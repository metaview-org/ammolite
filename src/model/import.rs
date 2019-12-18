use std::sync::{Arc, RwLock};
use std::path::Path;
use std::mem;
use core::num::NonZeroU32;
use arr_macro::arr;
use vulkano::sampler::SamplerAddressMode;
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::format::*;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::sampler::Sampler;
use vulkano::image::SyncImage;
use vulkano::image::Swizzle;
use vulkano::image::ImageDimensions;
use vulkano::image::ImageUsage;
use vulkano::image::layout::RequiredLayouts;



use vulkano::image::MipmapsCount;
use vulkano::image::view::ImageView;
use vulkano::image::traits::ImageViewAccess;
use vulkano::image::layout::typesafety;
use vulkano::image::sync::locker;
use byteorder::NativeEndian;
use byteorder::WriteBytesExt;
use gltf::{self, Document};
use gltf::mesh::Semantic;
use gltf::Node;
use gltf::accessor::DataType;
use gltf::image::Format as GltfFormat;
use gltf::texture::MagFilter;
use gltf::texture::MinFilter;
use failure::Error;
use ammolite_math::*;
use crate::NodeUBO;
use crate::MaterialUBO;
use crate::iter::ArrayIterator;
use crate::iter::ForcedExactSizeIterator;
use crate::iter::ByteBufferIterator;
use crate::vertex::{GltfVertexPosition, GltfVertexNormal, GltfVertexTangent, GltfVertexTexCoord};
use crate::sampler::IntoVulkanEquivalent;
use crate::pipeline::DescriptorSetMap;
use crate::pipeline::GltfGraphicsPipeline;
use crate::pipeline::GraphicsPipelineSetCache;
use crate::model::{Model, HelperResources};
use crate::model::resource::*;

enum ColorSpace {
    Srgb,
    Linear,
}

fn convert_double_channel_to_triple_channel<'a>(data_slice: &'a [u8]) -> Box<dyn ExactSizeIterator<Item=u8> + 'a> {
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
                    let index_iterator: ByteBufferIterator<u8> = ByteBufferIterator::from_accessor(buffer_data_array, &index_accessor);
                    let buffer_data: Vec<u8> = index_iterator.flat_map(|index| {
                        let mut array = [0u8; 2];
                        (&mut array[..]).write_u16::<NativeEndian>(index as u16).unwrap();
                        ArrayIterator::new(array)
                    }).collect();
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

pub fn precompute_missing_normal_buffers<'a, I>(device: &Arc<Device>,
                                                queue_families: &I,
                                                document: &Document,
                                                buffer_data_array: &[gltf::buffer::Data],
                                                initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<(
               Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>,
               Vec<Vec<Option<Vec<GltfVertexNormal>>>>
           ), Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut normal_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>> = vec![Vec::new(); document.meshes().len()];
    let mut normals: Vec<Vec<Option<Vec<GltfVertexNormal>>>> = vec![Vec::new(); document.meshes().len()];

    for (mesh_index, mesh) in document.meshes().enumerate() {
        normal_buffers[mesh_index] = vec![None; mesh.primitives().len()];
        normals[mesh_index] = vec![None; mesh.primitives().len()];

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if primitive.get(&Semantic::Normals).is_some() {
                continue;
            }

            let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
            let vertex_count = positions_accessor.count();
            // let index_count = primitive.indices()
            //     .map(|index_accessor| index_accessor.count())
            //     .unwrap_or(vertex_count);
            normals[mesh_index][primitive_index] = Some(vec![GltfVertexNormal([0.0; 3]); vertex_count]);
            let mut normals_data: Vec<GltfVertexNormal> = vec![GltfVertexNormal([0.0; 3]); vertex_count];
            let mut normals_count: Vec<u8> = vec![0; vertex_count];

            let mut index_iter: Box<dyn Iterator<Item=usize>> = if let Some(index_accessor) = primitive.indices() {
                match index_accessor.data_type() {
                    DataType::U8 => Box::new(
                        ByteBufferIterator::<u8>::from_accessor(buffer_data_array, &index_accessor)
                            .map(|index| index as usize)
                    ),
                    DataType::U16 => Box::new(
                        ByteBufferIterator::<u16>::from_accessor(buffer_data_array, &index_accessor)
                            .map(|index| index as usize)
                    ),
                    DataType::U32 => Box::new(
                        ByteBufferIterator::<u32>::from_accessor(buffer_data_array, &index_accessor)
                            .map(|index| index as usize)
                    ),
                    _ => unreachable!(),
                }
            } else {
                Box::new(0..vertex_count)
            };

            // Sum normals
            while let (Some(index_a), Some(index_b), Some(index_c)) = (index_iter.next(), index_iter.next(), index_iter.next()) {
                let a = Vec3(Model::index_byte_slice::<GltfVertexPosition>(buffer_data_array, &positions_accessor, index_a).0);
                let b = Vec3(Model::index_byte_slice::<GltfVertexPosition>(buffer_data_array, &positions_accessor, index_b).0);
                let c = Vec3(Model::index_byte_slice::<GltfVertexPosition>(buffer_data_array, &positions_accessor, index_c).0);
                let u = b - &a;
                let v = c - &a;
                let normal = u.cross(&v);

                for &index in [index_a, index_b, index_c].into_iter() {
                    let normals_count: &mut u8 = &mut normals_count[index];
                    let normal_sum = normal.clone() * (1f32 / (*normals_count + 1) as f32)
                        + Vec3(normals_data[index].0) * (*normals_count as f32 / (*normals_count + 1) as f32);

                    normals_data[index].0 = normal_sum.0;
                    *normals_count += 1;
                }
            }

            // Normalize normals
            for normal in &mut normals_data {
                let mut normal_vec = Vec3(normal.0);
                normal_vec.normalize();
                *normal = GltfVertexNormal(normal_vec.0);
            }

            let converted_byte_len = mem::size_of::<GltfVertexNormal>() * normals_data.len();
            let (device_normal_buffer, normal_buffer_initialization) = unsafe {
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

            // Copy data to a separate Vec to return
            normals[mesh_index][primitive_index].as_mut().unwrap().copy_from_slice(&mut normals_data[..]);

            initialization_tasks.push(InitializationTask::Buffer {
                data: safe_transmute::guarded_transmute_to_bytes_pod_vec(normals_data),
                initialization_buffer: Arc::new(normal_buffer_initialization),
            });
            normal_buffers[mesh_index][primitive_index] = Some(device_normal_buffer);
        }
    }

    Ok((normal_buffers, normals))
}

pub fn precompute_missing_tangent_buffers<'a, I>(device: &Arc<Device>,
                                                 queue_families: &I,
                                                 document: &Document,
                                                 buffer_data_array: &[gltf::buffer::Data],
                                                 initialization_tasks: &mut Vec<InitializationTask>,
                                                 normal_buffers: &[Vec<Option<Vec<GltfVertexNormal>>>])
        -> Result<Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut tangent_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>> = vec![Vec::new(); document.meshes().len()];

    for (mesh_index, mesh) in document.meshes().enumerate() {
        tangent_buffers[mesh_index] = vec![None; mesh.primitives().len()];

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if primitive.get(&Semantic::Tangents).is_some() {
                continue;
            }

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
            // TODO
            let vertices_per_face = 3;
            let face_count = index_count / vertices_per_face;

            let get_semantic_index: Box<dyn Fn(usize, usize) -> usize> = if let Some(index_accessor) = primitive.indices() {
                match index_accessor.data_type() {
                    DataType::U8 => {
                        Box::new(move |face_index, vertex_index| *Model::index_byte_slice::<u8>(
                            &buffer_data_array[..],
                            &index_accessor,
                            face_index * vertices_per_face + vertex_index as usize,
                        ) as usize)
                    },
                    DataType::U16 => {
                        Box::new(move |face_index, vertex_index| *Model::index_byte_slice::<u16>(
                            &buffer_data_array[..],
                            &index_accessor,
                            face_index * vertices_per_face + vertex_index as usize,
                        ) as usize)
                    },
                    DataType::U32 => {
                        Box::new(move |face_index, vertex_index| *Model::index_byte_slice::<u32>(
                            &buffer_data_array[..],
                            &index_accessor,
                            face_index * vertices_per_face + vertex_index as usize,
                        ) as usize)
                    },
                    _ => unreachable!(),
                }
            } else {
                Box::new(|face_index, vertex_index| { face_index * vertices_per_face + vertex_index })
            };

            let position_accessor = primitive.get(&Semantic::Positions).unwrap();
            let normal_accessor = primitive.get(&Semantic::Normals);
            let precomputed_normals = if normal_accessor.is_none() {
                Some(normal_buffers[mesh_index][primitive_index].as_ref().unwrap())
            } else {
                None
            };
            let tex_coord_accessor = primitive.get(&Semantic::TexCoords(0));
            let zero_tex_coord: [f32; 2] = Default::default();

            mikktspace::generate_tangents(
                &|| { vertices_per_face }, // vertices_per_face: &'a Fn() -> usize, 
                &|| { face_count }, // face_count: &'a Fn() -> usize, 
                &|face_index, vertex_index| &Model::index_byte_slice::<GltfVertexPosition>(
                    &buffer_data_array[..],
                    &position_accessor,
                    get_semantic_index(face_index, vertex_index),
                ).0, // position: &'a Fn(usize, usize) -> &'a [f32; 3],
                &|face_index, vertex_index| {
                    let semantic_index = get_semantic_index(face_index, vertex_index);

                    if let &Some(ref normal_accessor) = &normal_accessor {
                        &Model::index_byte_slice::<GltfVertexNormal>(
                            &buffer_data_array[..],
                            normal_accessor,
                            semantic_index,
                        ).0
                    } else {
                        &precomputed_normals.unwrap()[semantic_index].0
                    }
                }, // normal: &'a Fn(usize, usize) -> &'a [f32; 3],
                &|face_index, vertex_index| {
                    if let &Some(ref tex_coord_accessor) = &tex_coord_accessor {
                        &Model::index_byte_slice::<GltfVertexTexCoord>(
                            &buffer_data_array[..],
                            &tex_coord_accessor,
                            get_semantic_index(face_index, vertex_index),
                        ).0
                    } else {
                        &zero_tex_coord
                    }
                }, // tex_coord: &'a Fn(usize, usize) -> &'a [f32; 2],
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

    Ok(tangent_buffers)
}

pub fn import_device_buffers<'a, I>(device: &Arc<Device>,
                                    queue_families: &I,
                                    document: &Document,
                                    buffer_data_array: &[gltf::buffer::Data],
                                    initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>, Error>
        where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut device_buffers: Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = Vec::with_capacity(buffer_data_array.len());
    let mut buffer_usage_vec = vec![BufferUsage::none(); buffer_data_array.len()];

    // Scan document for buffer usage and optimize
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            macro_rules! check_accessor {
                (indexed [$($semantic:tt)+] $buffer_usage_field:ident) => {
                    for semantic_index in 0.. {
                        let semantic = Semantic::$($semantic)+(semantic_index);

                        if let Some(accessor) = primitive.get(&semantic) {
                            let buffer_index = accessor.view().buffer().index();

                            buffer_usage_vec[buffer_index].$buffer_usage_field = true;
                        } else {
                            break;
                        }
                    }
                };

                (unindexed [Indices] $buffer_usage_field:ident) => {
                    if let Some(accessor) = primitive.indices() {
                        let buffer_index = accessor.view().buffer().index();

                        buffer_usage_vec[buffer_index].$buffer_usage_field = true;
                    }
                };

                (unindexed [$($semantic:tt)+] $buffer_usage_field:ident) => {
                    if let Some(accessor) = primitive.get(&Semantic::$($semantic)+) {
                        let buffer_index = accessor.view().buffer().index();

                        buffer_usage_vec[buffer_index].$buffer_usage_field = true;
                    }
                };
            }

            check_accessor!(unindexed [Indices] index_buffer);
            check_accessor!(unindexed [Positions] vertex_buffer);
            check_accessor!(unindexed [Normals] vertex_buffer);
            check_accessor!(unindexed [Tangents] vertex_buffer);
            check_accessor!(indexed [Colors] vertex_buffer);
            check_accessor!(indexed [TexCoords] vertex_buffer);
            // check_accessor!(indexed [Joints] ???);
            // check_accessor!(indexed [Weights] ???);
        }
    }

    // Initialize the buffers with predetermined usages
    for (index, &gltf::buffer::Data(ref buffer_data)) in buffer_data_array.iter().enumerate() {
        let mut buffer_usage = buffer_usage_vec[index];

        if buffer_usage == BufferUsage::none() {
            panic!("Buffers with such usage are not yet implememented.");
        }

        buffer_usage.transfer_destination = true;

        let (device_buffer, buffer_initialization) = unsafe {
            ImmutableBuffer::raw(
                device.clone(),
                buffer_data.len(),
                buffer_usage,
                queue_families.clone(),
            )
        }?;
        initialization_tasks.push(InitializationTask::Buffer {
            // TODO: Avoid cloning
            data: buffer_data.iter().cloned().collect::<Vec<_>>(),
            initialization_buffer: Arc::new(buffer_initialization),
        });
        device_buffers.push(device_buffer);
    }

    Ok(device_buffers)
}

pub fn import_device_images<'a, I>(device: &Arc<Device>,
                                   _queue_families: &I,
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

    // Make image data accessible with an Arc
    let image_data_array: Vec<ArcImageData> = image_data_array.into_iter()
        .map(|gltf::image::Data { pixels, format, width, height }| {
            ArcImageData { pixels: Arc::new(pixels), format, width, height }
        }).collect();

    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let textures_slice = [(ColorSpace::Srgb,   pbr.base_color_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, pbr.metallic_roughness_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, material.normal_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Linear, material.occlusion_texture().map(|wrapped| wrapped.texture())),
                            (ColorSpace::Srgb,   material.emissive_texture().map(|wrapped| wrapped.texture()))];

        for (space, image) in textures_slice.into_iter()
                                 .filter(|(_, option)| option.is_some())
                                 .map(|(space, option)| (space, option.as_ref().unwrap().source())) {
            let &ArcImageData {
                ref pixels,
                format,
                width,
                height,
            } = &image_data_array[image.index()];

            macro_rules! insert_image_with_format_impl {
                ([$($vk_format:tt)+], $insert_init_tasks:expr) => {{
                    let usage = ImageUsage {
                        transfer_source: true,
                        transfer_destination: true,
                        sampled: true,
                        ..ImageUsage::none()
                    };

                    let device_image: Arc<SyncImage<locker::MatrixImageResourceLocker>> = Arc::new(
                        SyncImage::new(
                            device.clone(),
                            usage.clone(),
                            $($vk_format)+,
                            ImageDimensions::Dim2D {
                                width: NonZeroU32::new(width).expect("The image width must not be 0."),
                                height: NonZeroU32::new(height).expect("The image height must not be 0."),
                            },
                            NonZeroU32::new(1).unwrap(),
                            MipmapsCount::Log2,
                        )?
                    );

                    let device_image_view = {
                        // let mut required_layouts = RequiredLayouts::general();
                        let mut required_layouts = RequiredLayouts::none();
                        required_layouts.infer_mut(usage);
                        required_layouts.global = Some(typesafety::ImageLayoutEnd::ShaderReadOnlyOptimal);

                        Arc::new(ImageView::new::<$($vk_format)+>(
                            device_image.clone(),
                            None,
                            None,
                            Swizzle::identity(),
                            None,
                            required_layouts,
                        )?)
                    };

                    // let (device_image, image_initialization) = ImmutableImage::uninitialized(
                    //     device.clone(),
                    //     Dimensions::Dim2d {
                    //         width,
                    //         height,
                    //     },
                    //     $($vk_format)+,
                    //     MipmapsCount::One,
                    //     ImageUsage {
                    //         transfer_source: true,
                    //         transfer_destination: true,
                    //         sampled: true,
                    //         ..ImageUsage::none()
                    //     },
                    //     ImageLayout::ShaderReadOnlyOptimal,
                    //     queue_families.clone(),
                    // )?;
                    ($insert_init_tasks)(device_image);
                    device_images[image.index()] = device_image_view;
                }}
            }

            macro_rules! insert_image_with_format {
                ([$($vk_format:tt)+]) => {{
                    insert_image_with_format_impl! {
                        [$($vk_format)+],
                        |image: Arc<SyncImage<locker::MatrixImageResourceLocker>>| {
                            initialization_tasks.push(InitializationTask::ImageWithMipmaps {
                                data: pixels.clone(),
                                device_image: image,
                                texel_conversion: None,
                            });
                        }
                    }
                }};

                ([$($vk_format:tt)+], $texel_conversion:expr) => {{
                    insert_image_with_format_impl! {
                        [$($vk_format)+],
                        |image: Arc<SyncImage<locker::MatrixImageResourceLocker>>| {
                            initialization_tasks.push(InitializationTask::ImageWithMipmaps {
                                data: pixels.clone(),
                                device_image: image,
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
                (GltfFormat::B8G8R8, ColorSpace::Linear) => insert_image_with_format!([B8G8R8A8Unorm], Box::new(convert_double_channel_to_triple_channel)),
                (GltfFormat::B8G8R8, ColorSpace::Srgb) => insert_image_with_format!([B8G8R8A8Srgb], Box::new(convert_double_channel_to_triple_channel)),
                (GltfFormat::R8G8B8A8, ColorSpace::Linear) => insert_image_with_format!([R8G8B8A8Unorm]),
                (GltfFormat::R8G8B8A8, ColorSpace::Srgb) => insert_image_with_format!([R8G8B8A8Srgb]),
                (GltfFormat::B8G8R8A8, ColorSpace::Linear) => insert_image_with_format!([B8G8R8A8Unorm]),
                (GltfFormat::B8G8R8A8, ColorSpace::Srgb) => insert_image_with_format!([B8G8R8A8Srgb]),
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
        results[parent.index()].as_ref().unwrap() * Mat4::new(node.transform().matrix())
    } else {
        Mat4::new(node.transform().matrix())
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

pub fn create_node_descriptor_sets<'a>(device: &Arc<Device>,
                                       pipelines: impl IntoIterator<Item=&'a GltfGraphicsPipeline>,
                                       document: &Document,
                                       initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<(Vec<Mat4>, Vec<DescriptorSetMap>), Error> {
    let pipelines: Vec<_> = pipelines.into_iter().map(Clone::clone).collect();
    let mut node_descriptor_set_maps: Vec<DescriptorSetMap> = Vec::with_capacity(document.nodes().len());
    let transform_matrices = get_node_matrices(&document);

    for node in document.nodes() {
        let node_ubo = NodeUBO::new(transform_matrices[node.index()].clone());
        let (device_buffer, buffer_initialization) = unsafe {
            ImmutableBuffer::<NodeUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let descriptor_set_map = DescriptorSetMap::custom(&pipelines[..], |pipeline|
            Arc::new(
                PersistentDescriptorSet::start(pipeline.layout.clone(), 2)
                    .add_buffer(device_buffer.clone()).unwrap()
                    .build().unwrap()
            )
        );

        initialization_tasks.push(InitializationTask::NodeDescriptorSet {
            data: node_ubo,
            initialization_buffer: Arc::new(buffer_initialization),
        });
        node_descriptor_set_maps.push(descriptor_set_map);
    }

    Ok((transform_matrices, node_descriptor_set_maps))
}

pub fn create_samplers(device: &Arc<Device>, document: &Document) -> Result<Vec<Arc<Sampler>>, Error> {
    let mut device_samplers: Vec<Arc<Sampler>> = Vec::with_capacity(document.samplers().len());

    for gltf_sampler in document.samplers() {
        // let (min_filter, mipmap_mode) = gltf_sampler.min_filter()
        //     .unwrap_or(MinFilter::LinearMipmapLinear)
        //     .into_vulkan_equivalent();
        let (min_filter, mipmap_mode) = MinFilter::LinearMipmapLinear
            .into_vulkan_equivalent();
        let sampler = Sampler::new(
            device.clone(),
            gltf_sampler.mag_filter().unwrap_or(MagFilter::Linear).into_vulkan_equivalent(),
            min_filter,
            mipmap_mode,
            gltf_sampler.wrap_s().into_vulkan_equivalent(),
            gltf_sampler.wrap_t().into_vulkan_equivalent(),
            SamplerAddressMode::Repeat,
            /* These parameters affect how interpolation between Mip levels is calculated.
             *
             * We don't enforce any bounds on the resulting LOD value, `max_lod` is thus
             * `std::f32::MAX`.
             *
             * See the following link for how LOD is calculated.
             * https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#textures-level-of-detail-operation
             */
            0.0, // mip_lod_bias
            1.0, // max_anisotropy
            0.0, // min_lod
            std::f32::MAX, // max_lod
        )?;

        device_samplers.push(sampler);
    }

    Ok(device_samplers)
}

pub fn create_material_descriptor_sets<'a>(device: &Arc<Device>,
                                           pipelines: impl IntoIterator<Item=&'a GltfGraphicsPipeline>,
                                           helper_resources: &HelperResources,
                                           document: &Document,
                                           device_images: &[Arc<dyn ImageViewAccess + Send + Sync>],
                                           initialization_tasks: &mut Vec<InitializationTask>)
        -> Result<Vec<DescriptorSetMap>, Error> {
    let device_samplers = create_samplers(device, &document)?;
    let pipelines: Vec<_> = pipelines.into_iter().map(Clone::clone).collect();
    let mut material_descriptor_set_maps: Vec<DescriptorSetMap> = Vec::with_capacity(document.materials().len());

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
        let descriptor_set_map: DescriptorSetMap = DescriptorSetMap::custom(&pipelines[..], |pipeline|
            Arc::new(
                PersistentDescriptorSet::start(pipeline.layout.clone(), 3)
                    .add_buffer(device_material_ubo_buffer.clone()).unwrap()
                    .add_image(base_color_texture.clone()).unwrap()
                    .add_sampler(base_color_sampler.clone()).unwrap()
                    .add_image(metallic_roughness_texture.clone()).unwrap()
                    .add_sampler(metallic_roughness_sampler.clone()).unwrap()
                    .add_image(normal_texture.clone()).unwrap()
                    .add_sampler(normal_sampler.clone()).unwrap()
                    .add_image(occlusion_texture.clone()).unwrap()
                    .add_sampler(occlusion_sampler.clone()).unwrap()
                    .add_image(emissive_texture.clone()).unwrap()
                    .add_sampler(emissive_sampler.clone()).unwrap()
                    .build().unwrap()
            )
        );

        initialization_tasks.push(InitializationTask::MaterialDescriptorSet {
            data: material_ubo,
            initialization_buffer: Arc::new(material_ubo_buffer_initialization),
        });
        material_descriptor_set_maps.push(descriptor_set_map);
    }

    Ok(material_descriptor_set_maps)
}

fn import_model<'a, I>(
    device: &Arc<Device>,
    queue_families: I,
    pipeline_cache: &GraphicsPipelineSetCache,
    helper_resources: &HelperResources,
    document: Document,
    buffer_data_array: Vec<gltf::buffer::Data>,
    image_data_array: Vec<gltf::image::Data>,
) -> Result<SimpleUninitializedResource<Model>, Error>
where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let mut initialization_tasks: Vec<InitializationTask> = Vec::with_capacity(
        buffer_data_array.len() + image_data_array.len() + document.accessors().len() + document.nodes().len() + document.materials().len()
    );
    let pipelines = Model::get_pipelines_layouts(&document, pipeline_cache);

    let converted_index_buffers_by_accessor_index = import_index_buffers_by_accessor_index(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks)?;
    let (normal_buffers, normals) = precompute_missing_normal_buffers(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks)?;
    let tangent_buffers = precompute_missing_tangent_buffers(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks, &normals[..])?;
    let device_buffers = import_device_buffers(device, &queue_families, &document, &buffer_data_array[..], &mut initialization_tasks)?;
    let device_images = import_device_images(device, &queue_families, helper_resources, &document, image_data_array, &mut initialization_tasks)?;
    let (node_transform_matrices, node_descriptor_sets) = create_node_descriptor_sets(device, &pipelines[..], &document, &mut initialization_tasks)?;
    let material_descriptor_sets = create_material_descriptor_sets(device, &pipelines[..], helper_resources, &document, &device_images[..], &mut initialization_tasks)?;
    let scene_subpass_context_less_draw_calls = document.scenes().map(|_| arr![RwLock::new(None); 4]).collect();

    Ok(SimpleUninitializedResource::new(Model {
        document,
        buffer_data: buffer_data_array,
        device_buffers,
        device_images,
        converted_index_buffers_by_accessor_index,
        normal_buffers,
        tangent_buffers,
        node_transform_matrices,
        node_descriptor_sets,
        material_descriptor_sets,
        scene_subpass_context_less_draw_calls,
    }, initialization_tasks))
}

pub fn import_model_path<'a, I>(
    device: &Arc<Device>,
    queue_families: I,
    pipeline_cache: &GraphicsPipelineSetCache,
    helper_resources: &HelperResources,
    path: impl AsRef<Path>,
) -> Result<SimpleUninitializedResource<Model>, Error>
where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let (document, buffer_data_array, image_data_array) = gltf::import(path)?;
    import_model::<I>(
        device,
        queue_families,
        pipeline_cache,
        helper_resources,
        document,
        buffer_data_array,
        image_data_array,
    )
}

pub fn import_model_slice<'a, I>(
    device: &Arc<Device>,
    queue_families: I,
    pipeline_cache: &GraphicsPipelineSetCache,
    helper_resources: &HelperResources,
    slice: impl AsRef<[u8]>,
) -> Result<SimpleUninitializedResource<Model>, Error>
where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    let (document, buffer_data_array, image_data_array) = gltf::import_slice(slice)?;
    import_model::<I>(
        device,
        queue_families,
        pipeline_cache,
        helper_resources,
        document,
        buffer_data_array,
        image_data_array,
    )
}
