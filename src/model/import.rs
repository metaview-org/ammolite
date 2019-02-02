use std::sync::Arc;
use std::path::Path;
use std::mem;
use vulkano::sampler::SamplerAddressMode;
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::format::*;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::sampler::Sampler;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::immutable::ImmutableImageInitialization;
use vulkano::image::Dimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImageLayout;
use vulkano::image::MipmapsCount;
use vulkano::image::traits::ImageViewAccess;
use gltf::{self, Document};
use gltf::mesh::Semantic;
use gltf::Node;
use gltf::accessor::DataType;
use gltf::image::Format as GltfFormat;
use gltf::texture::MagFilter;
use gltf::texture::MinFilter;
use failure::Error;
use crate::NodeUBO;
use crate::MaterialUBO;
use crate::iter::ArrayIterator;
use crate::iter::ForcedExactSizeIterator;
use crate::vertex::{GltfVertexPosition, GltfVertexNormal, GltfVertexTangent, GltfVertexTexCoord};
use crate::sampler::IntoVulkanEquivalent;
use crate::math::*;
use crate::model::{Model, ColorSpace, HelperResources};
use crate::model::resource::*;

fn convert_double_channel_to_triple_channel<'a>(data_slice: &'a [u8]) -> Box<ExactSizeIterator<Item=u8> + 'a> {
    let unsized_iterator = data_slice.chunks(3).flat_map(|rgb| {
        ArrayIterator::new([rgb[0], rgb[1], rgb[2], 0xFF])
    });
    let iterator_len = data_slice.len() / 3 * 4;
    Box::new(ForcedExactSizeIterator::new(unsized_iterator, iterator_len))
}

pub fn import_index_buffers_by_accessor_index<'a, I>(device: &Arc<Device>,
                                                     queue_families: &I,
                                                     document: &Document,
                                                     buffer_data_array: &[gltf::buffer::Data],
                                                     initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut converted_index_buffers_by_accessor_index: Vec<Option<Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>>> = vec![None; document.accessors().len()];

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(index_accessor) = primitive.indices() {
                if index_accessor.data_type() == DataType::U8 {
                    let index_slice = Model::get_semantic_byte_slice(buffer_data_array, &index_accessor);
                    let buffer_data: Vec<u8> = index_slice.into_iter().flat_map(|index| ArrayIterator::new([*index, 0])).collect(); // FIXME: Assumes byte order in u16
                    let converted_byte_len = mem::size_of::<u16>() * index_accessor.count();
                    let (device_index_buffer, index_buffer_initialization) = unsafe {
                        ImmutableBuffer::<[u16]>::raw(
                            device.clone(),
                            converted_byte_len,
                            BufferUsage {
                                transfer_destination: true,
                                index_buffer: true,
                                ..BufferUsage::none()
                            },
                            queue_families.clone(),
                        )
                    }?;
                    let index_buffer_initialization: BufferSlice<[u8], _> = unsafe {
                        BufferSlice::from_typed_buffer_access(index_buffer_initialization).reinterpret::<[u8]>()
                    };
                    initialization_tasks.push(InitializationTask::Buffer {
                        data: buffer_data,
                        initialization_buffer: Arc::new(index_buffer_initialization),
                    });
                    converted_index_buffers_by_accessor_index[index_accessor.index()] = Some(device_index_buffer);
                }
            }
        }
    }

    Ok(converted_index_buffers_by_accessor_index)
}

pub fn import_tangent_buffers<'a, I>(device: &Arc<Device>,
                                     queue_families: &I,
                                     document: &Document,
                                     buffer_data_array: &[gltf::buffer::Data],
                                     initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let primitive_count = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    let mut tangent_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>> = vec![Vec::new(); primitive_count];

    for (mesh_index, mesh) in document.meshes().enumerate() {
        tangent_buffers[mesh_index] = vec![None; mesh.primitives().len()];

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            // Compute tangents for the model if they are missing.
            if primitive.get(&Semantic::Tangents).is_none() {
                let vertex_count = primitive.get(&Semantic::Positions).unwrap().count();
                let index_count = primitive.indices()
                    .map(|index_accessor| index_accessor.count())
                    .unwrap_or(vertex_count);

                let converted_byte_len = mem::size_of::<GltfVertexTangent>() * vertex_count;
                let (device_tangent_buffer, tangent_buffer_initialization) = unsafe {
                    ImmutableBuffer::<[u8]>::raw(
                        device.clone(),
                        converted_byte_len,
                        BufferUsage {
                            transfer_destination: true,
                            vertex_buffer: true,
                            ..BufferUsage::none()
                        },
                        queue_families.clone(),
                    )
                }?;
                let tangent_buffer_initialization: BufferSlice<[u8], _> = unsafe {
                    BufferSlice::from_typed_buffer_access(tangent_buffer_initialization).reinterpret::<[u8]>()
                };
                let mut buffer_data: Vec<GltfVertexTangent> = vec![GltfVertexTangent([0.0; 4]); vertex_count];
                let vertices_per_face = 3;
                let face_count = index_count / vertices_per_face;

                let get_semantic_index: Box<Fn(usize, usize) -> usize> = if let Some(index_accessor) = primitive.indices() {
                    match index_accessor.data_type() {
                        DataType::U8 => {
                            let index_slice: &[u8] = Model::get_semantic_byte_slice(&buffer_data_array[..], &index_accessor);

                            Box::new(move |face_index, vertex_index| {
                                index_slice[face_index * vertices_per_face + vertex_index] as usize
                            })
                        },
                        DataType::U16 => {
                            let index_slice: &[u16] = Model::get_semantic_byte_slice(&buffer_data_array[..], &index_accessor);

                            Box::new(move |face_index, vertex_index| {
                                index_slice[face_index * vertices_per_face + vertex_index] as usize
                            })
                        },
                        _ => unreachable!(),
                    }
                } else {
                    Box::new(|face_index, vertex_index| { face_index * vertices_per_face + vertex_index })
                };

                let position_accessor = primitive.get(&Semantic::Positions).unwrap();
                let normal_accessor = primitive.get(&Semantic::Normals).unwrap();
                let tex_coord_accessor = primitive.get(&Semantic::TexCoords(0)).unwrap();
                let position_slice: &[GltfVertexPosition] = Model::get_semantic_byte_slice(&buffer_data_array[..], &position_accessor);
                let normal_slice: &[GltfVertexNormal] = Model::get_semantic_byte_slice(&buffer_data_array[..], &normal_accessor);
                let tex_coord_slice: &[GltfVertexTexCoord] = Model::get_semantic_byte_slice(&buffer_data_array[..], &tex_coord_accessor);

                mikktspace::generate_tangents(
                    &|| { vertices_per_face }, // vertices_per_face: &'a Fn() -> usize, 
                    &|| { face_count }, // face_count: &'a Fn() -> usize, 
                    &|face_index, vertex_index| { &position_slice[get_semantic_index(face_index, vertex_index)].0 }, // position: &'a Fn(usize, usize) -> &'a [f32; 3],
                    &|face_index, vertex_index| { &normal_slice[get_semantic_index(face_index, vertex_index)].0 }, // normal: &'a Fn(usize, usize) -> &'a [f32; 3],
                    &|face_index, vertex_index| { &tex_coord_slice[get_semantic_index(face_index, vertex_index)].0 }, // tex_coord: &'a Fn(usize, usize) -> &'a [f32; 2],
                    &mut |face_index, vertex_index, mut tangent| {
                        // The algorithm generates tangents in right-handed coordinate space,
                        // but models with pre-generated tangents seem to be in left-handed
                        // coordinate space.
                        tangent[3] *= -1.0;
                        buffer_data[get_semantic_index(face_index, vertex_index)] = GltfVertexTangent(tangent);
                    }, // set_tangent: &'a mut FnMut(usize, usize, [f32; 4])
                );

                initialization_tasks.push(InitializationTask::Buffer {
                    data: safe_transmute::guarded_transmute_to_bytes_pod_vec(buffer_data),
                    initialization_buffer: Arc::new(tangent_buffer_initialization),
                });
                tangent_buffers[mesh_index][primitive_index] = Some(device_tangent_buffer);
            }
        }
    }

    Ok(tangent_buffers)
}

pub fn import_device_buffers<'a, I>(device: &Arc<Device>,
                                    queue_families: &I,
                                    buffer_data_array: Vec<gltf::buffer::Data>,
                                    initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut device_buffers: Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = Vec::with_capacity(buffer_data_array.len());

    for gltf::buffer::Data(buffer_data) in buffer_data_array.into_iter() {
        let (device_buffer, buffer_initialization) = unsafe {
            ImmutableBuffer::raw(
                device.clone(),
                buffer_data.len(),
                BufferUsage { // TODO: Scan document for buffer usage and optimize
                    transfer_destination: true,
                    uniform_buffer: true,
                    storage_buffer: true,
                    index_buffer: true,
                    vertex_buffer: true,
                    indirect_buffer: true,
                    ..BufferUsage::none()
                },
                queue_families.clone(),
            )
        }?;
        initialization_tasks.push(InitializationTask::Buffer {
            data: buffer_data,
            initialization_buffer: Arc::new(buffer_initialization),
        });
        device_buffers.push(device_buffer);
    }

    Ok(device_buffers)
}

pub fn import_device_images<'a, I>(device: &Arc<Device>,
                                   queue_families: &I,
                                   helper_resources: &HelperResources,
                                   document: &Document,
                                   image_data_array: Vec<gltf::image::Data>,
                                   initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Arc<dyn ImageViewAccess + Send + Sync>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>> = vec![helper_resources.empty_image.clone(); image_data_array.len()];

    struct ArcImageData {
        pixels: Arc<Vec<u8>>,
        format: gltf::image::Format,
        width: u32,
        height: u32,
    };

    let image_data_array: Vec<ArcImageData> = image_data_array.into_iter()
        .map(|gltf::image::Data { pixels, format, width, height }| {
            ArcImageData { pixels: Arc::new(pixels), format, width, height }
        }).collect();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let images_slice = [(ColorSpace::Srgb,   pbr.base_color_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, pbr.metallic_roughness_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, material.normal_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, material.occlusion_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Srgb,   material.emissive_texture().map(|wrapped| wrapped.texture()))];

        for (space, image) in images_slice.into_iter()
                                 .filter(|(_, option)| option.is_some())
                                 .map(|(space, option)| (space, option.as_ref().unwrap())) {
            let &ArcImageData {
                ref pixels,
                format,
                width,
                height,
            } = &image_data_array[image.index()];

            macro_rules! insert_image_with_format_impl {
                ([$($vk_format:tt)+], $insert_init_tasks:expr) => {{
                    let (device_image, image_initialization) = ImmutableImage::uninitialized(
                        device.clone(),
                        Dimensions::Dim2d {
                            width,
                            height,
                        },
                        $($vk_format)+,
                        MipmapsCount::One,
                        ImageUsage {
                            transfer_source: true,
                            transfer_destination: true,
                            sampled: true,
                            ..ImageUsage::none()
                        },
                        ImageLayout::ShaderReadOnlyOptimal,
                        queue_families.clone(),
                    )?;
                    ($insert_init_tasks)(image_initialization);
                    device_images[image.index()] = device_image;
                }}
            }

            macro_rules! insert_image_with_format {
                ([$($vk_format:tt)+]) => {{
                    insert_image_with_format_impl! {
                        [$($vk_format)+],
                        |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                            initialization_tasks.push(InitializationTask::Image {
                                data: pixels.clone(),
                                device_image: Arc::new(image_initialization),
                                texel_conversion: None,
                            });
                        }
                    }
                }};

                ([$($vk_format:tt)+], $texel_conversion:expr) => {{
                    insert_image_with_format_impl! {
                        [$($vk_format)+],
                        |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                            initialization_tasks.push(InitializationTask::Image {
                                data: pixels.clone(),
                                device_image: Arc::new(image_initialization),
                                texel_conversion: Some($texel_conversion),
                            });
                        }
                    }
                }}
            }

            match (format, space) {
                (GltfFormat::R8, ColorSpace::Linear) => insert_image_with_format!([R8Unorm]),
                (GltfFormat::R8, ColorSpace::Srgb) => insert_image_with_format!([R8Srgb]),
                (GltfFormat::R8G8, ColorSpace::Linear) => insert_image_with_format!([R8G8Unorm]),
                (GltfFormat::R8G8, ColorSpace::Srgb) => insert_image_with_format!([R8G8Srgb]),
                (GltfFormat::R8G8B8, ColorSpace::Linear) => insert_image_with_format!([R8G8B8A8Unorm], Box::new(convert_double_channel_to_triple_channel)),
                (GltfFormat::R8G8B8, ColorSpace::Srgb) => insert_image_with_format!([R8G8B8A8Srgb], Box::new(convert_double_channel_to_triple_channel)),
                (GltfFormat::R8G8B8A8, ColorSpace::Linear) => insert_image_with_format!([R8G8B8A8Unorm]),
                (GltfFormat::R8G8B8A8, ColorSpace::Srgb) => insert_image_with_format!([R8G8B8A8Srgb]),
            }
        }
    }

    Ok(device_images)
}

fn get_node_matrices_impl(parent: Option<&Node>, node: &Node, results: &mut Vec<Option<Mat4>>) {
    // Matrix and its children already calculated, bail.
    if let Some(_) = results[node.index()] {
        return;
    }

    results[node.index()] = Some(if let Some(parent) = parent {
        results[parent.index()].as_ref().unwrap() * Mat4(node.transform().matrix())
    } else {
        Mat4(node.transform().matrix())
    });

    for child in node.children() {
        get_node_matrices_impl(Some(node), &child, results);
    }
}

/// Recursively calculates the final transformation matrix for each node of the document
fn get_node_matrices(document: &Document) -> Vec<Mat4> {
    let mut results = Vec::with_capacity(document.nodes().len());

    for _ in 0..document.nodes().len() {
        results.push(None);
    }

    for scene in document.scenes() {
        for node in scene.nodes() {
            get_node_matrices_impl(None, &node, &mut results);
        }
    }

    results.into_iter().map(|option| option.unwrap_or_else(Mat4::identity)).collect()
}

pub fn create_node_descriptor_sets(device: &Arc<Device>,
                                   pipeline: &Arc<GraphicsPipelineAbstract + Sync + Send>,
                                   document: &Document,
                                   initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Arc<dyn DescriptorSet + Send + Sync>>, Error> {
    let mut node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.nodes().len());
    let transform_matrices = get_node_matrices(&document);

    for node in document.nodes() {
        let node_ubo = NodeUBO::new(transform_matrices[node.index()].clone());
        let (device_buffer, buffer_initialization) = unsafe {
            ImmutableBuffer::<NodeUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 1)
                .add_buffer(device_buffer.clone()).unwrap()
                .build().unwrap()
        );

        initialization_tasks.push(InitializationTask::NodeDescriptorSet {
            data: node_ubo,
            initialization_buffer: Arc::new(buffer_initialization),
        });
        node_descriptor_sets.push(descriptor_set);
    }

    Ok(node_descriptor_sets)
}

pub fn create_samplers(device: &Arc<Device>, document: &Document) -> Result<Vec<Arc<Sampler>>, Error> {
    let mut device_samplers: Vec<Arc<Sampler>> = Vec::with_capacity(document.samplers().len());

    for gltf_sampler in document.samplers() {
        let (min_filter, mipmap_mode) = gltf_sampler.min_filter()
            .unwrap_or(MinFilter::LinearMipmapLinear)
            .into_vulkan_equivalent();
        let sampler = Sampler::new(
            device.clone(),
            gltf_sampler.mag_filter().unwrap_or(MagFilter::Linear).into_vulkan_equivalent(),
            min_filter,
            mipmap_mode,
            gltf_sampler.wrap_s().into_vulkan_equivalent(),
            gltf_sampler.wrap_t().into_vulkan_equivalent(),
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            1.0, // TODO check the range of LOD
        )?;

        device_samplers.push(sampler);
    }

    Ok(device_samplers)
}

pub fn create_material_descriptor_sets(device: &Arc<Device>,
                                       pipeline: &Arc<GraphicsPipelineAbstract + Sync + Send>,
                                       helper_resources: &HelperResources,
                                       document: &Document,
                                       device_images: &[Arc<dyn ImageViewAccess + Send + Sync>],
                                       initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Arc<dyn DescriptorSet + Send + Sync>>, Error> {
    let device_samplers = create_samplers(device, &document)?;
    let mut material_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.materials().len());

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base_color_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = pbr
            .base_color_texture()
            .map(|texture_info| {
                let image_index = texture_info.texture().source().index();
                device_images[image_index].clone()
            });
        let base_color_sampler_option: Option<Arc<Sampler>> = pbr
            .base_color_texture()
            .and_then(|texture_info| texture_info.texture().sampler().index())
            .map(|sampler_index| device_samplers[sampler_index].clone());
        let metallic_roughness_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = pbr
            .metallic_roughness_texture()
            .map(|texture_info| {
                let image_index = texture_info.texture().source().index();
                device_images[image_index].clone()
            });
        let metallic_roughness_sampler_option: Option<Arc<Sampler>> = pbr
            .metallic_roughness_texture()
            .and_then(|texture_info| texture_info.texture().sampler().index())
            .map(|sampler_index| device_samplers[sampler_index].clone());
        let normal_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = material
            .normal_texture()
            .map(|texture_info| {
                let image_index = texture_info.texture().source().index();
                device_images[image_index].clone()
            });
        let normal_sampler_option: Option<Arc<Sampler>> = material
            .normal_texture()
            .and_then(|texture_info| texture_info.texture().sampler().index())
            .map(|sampler_index| device_samplers[sampler_index].clone());
        let occlusion_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = material
            .occlusion_texture()
            .map(|texture_info| {
                let image_index = texture_info.texture().source().index();
                device_images[image_index].clone()
            });
        let occlusion_sampler_option: Option<Arc<Sampler>> = material
            .occlusion_texture()
            .and_then(|texture_info| texture_info.texture().sampler().index())
            .map(|sampler_index| device_samplers[sampler_index].clone());
        let emissive_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = material
            .emissive_texture()
            .map(|texture_info| {
                let image_index = texture_info.texture().source().index();
                device_images[image_index].clone()
            });
        let emissive_sampler_option: Option<Arc<Sampler>> = material
            .emissive_texture()
            .and_then(|texture_info| texture_info.texture().sampler().index())
            .map(|sampler_index| device_samplers[sampler_index].clone());
        let material_ubo = MaterialUBO::new(
            material.alpha_cutoff(),
            base_color_texture_option.is_some(),
            pbr.base_color_factor().into(),
            metallic_roughness_texture_option.is_some(),
            [pbr.metallic_factor(), pbr.roughness_factor()].into(),
            normal_texture_option.is_some(),
            material.normal_texture().map(|normal_texture| normal_texture.scale()).unwrap_or(1.0),
            occlusion_texture_option.is_some(),
            material.occlusion_texture().map(|occlusion_texture| occlusion_texture.strength()).unwrap_or(1.0),
            emissive_texture_option.is_some(),
            material.emissive_factor().into(),
        );
        let (device_material_ubo_buffer, material_ubo_buffer_initialization) = unsafe {
            ImmutableBuffer::<MaterialUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let base_color_texture: Arc<dyn ImageViewAccess + Send + Sync> = base_color_texture_option
            .unwrap_or_else(|| helper_resources.empty_image.clone());
        let base_color_sampler: Arc<Sampler> = base_color_sampler_option
            .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
        let metallic_roughness_texture: Arc<dyn ImageViewAccess + Send + Sync> = metallic_roughness_texture_option
            .unwrap_or_else(|| helper_resources.empty_image.clone());
        let metallic_roughness_sampler: Arc<Sampler> = metallic_roughness_sampler_option
            .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
        let normal_texture: Arc<dyn ImageViewAccess + Send + Sync> = normal_texture_option
            .unwrap_or_else(|| helper_resources.empty_image.clone());
        let normal_sampler: Arc<Sampler> = normal_sampler_option
            .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
        let occlusion_texture: Arc<dyn ImageViewAccess + Send + Sync> = occlusion_texture_option
            .unwrap_or_else(|| helper_resources.empty_image.clone());
        let occlusion_sampler: Arc<Sampler> = occlusion_sampler_option
            .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
        let emissive_texture: Arc<dyn ImageViewAccess + Send + Sync> = emissive_texture_option
            .unwrap_or_else(|| helper_resources.empty_image.clone());
        let emissive_sampler: Arc<Sampler> = emissive_sampler_option
            .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
        let descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 2)
                .add_buffer(device_material_ubo_buffer.clone()).unwrap()
                .add_image(base_color_texture).unwrap()
                .add_sampler(base_color_sampler).unwrap()
                .add_image(metallic_roughness_texture).unwrap()
                .add_sampler(metallic_roughness_sampler).unwrap()
                .add_image(normal_texture).unwrap()
                .add_sampler(normal_sampler).unwrap()
                .add_image(occlusion_texture).unwrap()
                .add_sampler(occlusion_sampler).unwrap()
                .add_image(emissive_texture).unwrap()
                .add_sampler(emissive_sampler).unwrap()
                .build().unwrap()
        );

        initialization_tasks.push(InitializationTask::MaterialDescriptorSet {
            data: material_ubo,
            initialization_buffer: Arc::new(material_ubo_buffer_initialization),
        });
        material_descriptor_sets.push(descriptor_set);
    }

    Ok(material_descriptor_sets)
}

pub fn import_model<'a, I, S>(device: &Arc<Device>,
                              queue_families: I,
                              pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>,
                              helper_resources: &HelperResources,
                              path: S)
        -> Result<SimpleUninitializedResource<Model>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone,
              S: AsRef<Path> {
    let (document, buffer_data_array, image_data_array) = gltf::import(path)?;
    let mut initialization_tasks: Vec<InitializationTask> = Vec::with_capacity(
        buffer_data_array.len() + image_data_array.len() + document.accessors().len() + document.nodes().len() + document.materials().len()
    );

    let converted_index_buffers_by_accessor_index = import_index_buffers_by_accessor_index(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks)?;
    let tangent_buffers = import_tangent_buffers(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks)?;
    let device_buffers = import_device_buffers(device, &queue_families, buffer_data_array, &mut initialization_tasks)?;
    let device_images = import_device_images(device, &queue_families, helper_resources, &document, image_data_array, &mut initialization_tasks)?;
    let node_descriptor_sets = create_node_descriptor_sets(device, &pipeline, &document, &mut initialization_tasks)?;
    let material_descriptor_sets = create_material_descriptor_sets(device, &pipeline, helper_resources, &document, &device_images[..], &mut initialization_tasks)?;

    Ok(SimpleUninitializedResource::new(Model {
        document,
        device_buffers,
        device_images,
        converted_index_buffers_by_accessor_index,
        tangent_buffers,
        node_descriptor_sets,
        material_descriptor_sets,
    }, initialization_tasks))
}
