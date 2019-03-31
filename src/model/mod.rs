pub mod error;
pub mod resource;
pub mod import;

use std::sync::Arc;
use std::path::Path;
use std::mem;
use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::Filter;
use vulkano::sampler::MipmapMode;
use vulkano::command_buffer::{DynamicState, AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::format::*;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassDescClearValues;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::descriptor::descriptor_set::collection::DescriptorSetsCollection;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::sampler::Sampler;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::Dimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImageLayout;
use vulkano::image::MipmapsCount;
use vulkano::image::traits::ImageViewAccess;
use gltf::accessor::Accessor;
use gltf::{self, Document};
use gltf::material::AlphaMode;
use gltf::mesh::{Primitive, Mesh, Semantic};
use gltf::Node;
use gltf::accessor::DataType;
use failure::Error;
use safe_transmute::PodTransmutable;
use crate::shaders::{InstanceUBO, MaterialUBO, PushConstants};
use crate::vertex::*;
use crate::pipeline::GraphicsPipelineProperties;
use crate::pipeline::GraphicsPipelineSetCache;
use self::error::*;
use self::resource::*;

// TODO: Figure out a better way to provide the clear values, as they shouldn't need to be
// specified by the end user
pub trait FramebufferWithClearValues<C>: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {}

impl<C, F> FramebufferWithClearValues<C> for F where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {}

// TODO: Remove generics
#[derive(Clone)]
pub struct InitializationDrawContext<'a> {
    pub draw_context: DrawContext<'a>,
    pub framebuffer: &'a Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>,
    pub clear_values: &'a Vec<ClearValue>,
}

#[derive(Clone)]
pub struct DrawContext<'a> {
    pub device: &'a Arc<Device>,
    pub queue_family: &'a QueueFamily<'a>,
    pub pipeline_cache: &'a Arc<GraphicsPipelineSetCache>,
    pub dynamic: &'a DynamicState,
    pub descriptor_set_scene: &'a Arc<DescriptorSet + Send + Sync>,
    pub descriptor_set_instance: &'a Arc<DescriptorSet + Send + Sync>,
    pub descriptor_set_blend: &'a Arc<DescriptorSet + Send + Sync>,
    pub helper_resources: &'a HelperResources,
}

#[derive(Clone)]
pub struct HelperResources {
    pub empty_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub zero_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    pub default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    pub cheapest_sampler: Arc<Sampler>,
}

impl HelperResources {
    pub fn new<'a, I>(device: &Arc<Device>, queue_families: I, pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>)
            -> Result<SimpleUninitializedResource<HelperResources>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
        let (empty_device_image, empty_image_initialization) = ImmutableImage::uninitialized(
            device.clone(),
            Dimensions::Dim2d {
                width: 1,
                height: 1,
            },
            R8Uint,
            MipmapsCount::One,
            ImageUsage {
                transfer_destination: true,
                sampled: true,
                ..ImageUsage::none()
            },
            ImageLayout::ShaderReadOnlyOptimal,
            queue_families.clone(),
        )?;

        // Size of mat4 (square, rank 4 matrix of f32s):
        let zero_buffer_len = 4 * 4 * 4; 
        let (zero_device_buffer, zero_buffer_initialization) = unsafe {
            ImmutableBuffer::raw(
                device.clone(),
                zero_buffer_len,
                BufferUsage {
                    transfer_destination: true,
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                queue_families.clone(),
            )
        }?;

        // Create the cheapest sampler possible
        let cheapest_sampler = Sampler::new(
            device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0,
        )?;

        let (device_default_material_ubo_buffer, default_material_ubo_buffer_initialization) = unsafe {
            ImmutableBuffer::<MaterialUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 3)
                .add_buffer(device_default_material_ubo_buffer.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .build().unwrap()
        );

        let tasks = vec![
            InitializationTask::Image {
                data: Arc::new(vec![0]),
                device_image: Arc::new(empty_image_initialization),
                texel_conversion: None,
            },
            InitializationTask::ZeroBuffer {
                len: zero_buffer_len,
                initialization_buffer: Arc::new(zero_buffer_initialization),
            },
            InitializationTask::MaterialDescriptorSet {
                data: MaterialUBO::default(),
                initialization_buffer: Arc::new(default_material_ubo_buffer_initialization),
            },
        ];

        let output = HelperResources {
            empty_image: empty_device_image,
            zero_buffer: zero_device_buffer,
            default_material_descriptor_set,
            cheapest_sampler,
        };

        Ok(SimpleUninitializedResource::new(output, tasks))
    }
}

pub struct Model {
    document: Document,
    device_buffers: Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>,
    #[allow(dead_code)]
    device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
    /// In case indexes are specified as u8 values, convert and store them as u16 values in this
    /// field. This conversion is needed, because Vulkan doesn't support 8-bit indices.
    converted_index_buffers_by_accessor_index: Vec<Option<Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>>>,
    /// Precomputed normal buffers, in case they were not specified in the glTF document
    // FIXME: Should probably be of type `GltfVertexNormal` instead of `u8`
    normal_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>,
    /// Precomputed tangent buffers, in case they were not specified in the glTF document
    // FIXME: Should probably be of type `GltfVertexTangent` instead of `u8`
    tangent_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>,
    // Note: Do not ever try to express the descriptor set explicitly.
    node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
    material_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
}

impl Model {
    pub(crate) fn index_byte_slice<'a, T: PodTransmutable>(buffer_data_array: &'a [gltf::buffer::Data], accessor: &Accessor, item_index: usize) -> &'a T {
        let view = accessor.view();
        let stride = view.stride().unwrap_or_else(|| accessor.size());
        let slice_offset = view.offset() + accessor.offset();
        let slice_len = stride * accessor.count();
        let slice: &[u8] = &buffer_data_array[view.buffer().index()][slice_offset..(slice_offset + slice_len)];
        let item_slice_start_index = item_index * stride;
        let item_slice_range = item_slice_start_index..(item_slice_start_index + mem::size_of::<T>());
        let item_slice = &slice[item_slice_range];
        let item_ptr = item_slice.as_ptr();

        // safe_transmute::guarded_transmute_pod::<&'a T>(item_slice).unwrap()
        unsafe { &*(item_ptr as *const T) }
    }

    pub(crate) fn get_semantic_buffer_view<T>(&self, accessor: &Accessor) -> BufferSlice<[T], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> {
        let buffer_view = accessor.view();
        let buffer_index = buffer_view.buffer().index();
        let buffer_offset = accessor.offset() + buffer_view.offset();
        let buffer_bytes = accessor.size() * accessor.count();

        let buffer = self.device_buffers[buffer_index].clone();
        let slice: BufferSlice<[u8], _> = BufferSlice::from_typed_buffer_access(buffer)
            .slice(buffer_offset..(buffer_offset + buffer_bytes))
            .unwrap();

        // println!("buffer_view: [{}] offset: {}; len: {}", unsafe { std::intrinsics::type_name::<T>() }, buffer_offset, buffer_bytes);

        unsafe { slice.reinterpret::<[T]>() }
    }

    pub fn import<'a, I, S>(device: &Arc<Device>, queue_families: I, pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>, helper_resources: &HelperResources, path: S)
            -> Result<SimpleUninitializedResource<Model>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone,
                  S: AsRef<Path> {
        import::import_model(device, queue_families, pipeline, helper_resources, path)
    }

    pub fn draw_scene(
        &self,
        mut command_buffer: AutoCommandBufferBuilder,
        context: InitializationDrawContext,
        scene_index: usize,
    ) -> Result<AutoCommandBufferBuilder, Error> {
        if scene_index >= self.document.scenes().len() {
            return Err(ModelDrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let InitializationDrawContext {
            draw_context,
            framebuffer,
            clear_values,
        } = context;

        let scene = self.document.scenes().nth(scene_index).unwrap();
        command_buffer = command_buffer.begin_render_pass(framebuffer.clone(), false, clear_values.clone())
            .unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Opaque, 0);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Mask, 1);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Blend, 2);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Blend, 3);
        }

        command_buffer = command_buffer
            .end_render_pass().unwrap();

        Ok(command_buffer)
    }

    pub fn draw_main_scene(
        &self,
        command_buffer: AutoCommandBufferBuilder,
        context: InitializationDrawContext,
    ) -> Result<AutoCommandBufferBuilder, Error> {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(command_buffer, context, main_scene_index)
        } else {
            Err(ModelDrawError::NoDefaultScene.into())
        }
    }

    pub fn draw_node<'a>(
        &self,
        node: Node<'a>,
        mut command_buffer: AutoCommandBufferBuilder,
        context: &DrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
    ) -> AutoCommandBufferBuilder {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let material = primitive.material();

                if material.alpha_mode() == alpha_mode {
                    let properties = GraphicsPipelineProperties::from(&primitive, &material);
                    let pipeline_set = context.pipeline_cache.get_or_create_pipeline(&properties);
                    let pipeline = match (alpha_mode, subpass) {
                        (AlphaMode::Opaque, _) => &pipeline_set.opaque,
                        (AlphaMode::Mask, _) => &pipeline_set.mask,
                        (AlphaMode::Blend, 2) => &pipeline_set.blend_preprocess,
                        (AlphaMode::Blend, 3) => &pipeline_set.blend_finalize,
                        _ => panic!("Invalid alpha_mode/subpass combination."),
                    };

                    let material_descriptor_set = material.index().map(|material_index| {
                        self.material_descriptor_sets[material_index].clone()
                    }).unwrap_or_else(|| {
                        context.helper_resources.default_material_descriptor_set.clone()
                    });

                    match (alpha_mode, subpass) {
                        (AlphaMode::Blend, 3) => {
                            let descriptor_sets = (
                                context.descriptor_set_scene.clone(),
                                context.descriptor_set_instance.clone(),
                                self.node_descriptor_sets[node.index()].clone(),
                                material_descriptor_set,
                                context.descriptor_set_blend.clone(),
                            );

                            command_buffer = self.draw_primitive(
                                &mesh,
                                &primitive,
                                command_buffer,
                                context,
                                descriptor_sets.clone(),
                                pipeline,
                            );

                        },
                        _ => {
                            let descriptor_sets = (
                                context.descriptor_set_scene.clone(),
                                context.descriptor_set_instance.clone(),
                                self.node_descriptor_sets[node.index()].clone(),
                                material_descriptor_set,
                            );

                            command_buffer = self.draw_primitive(
                                &mesh,
                                &primitive,
                                command_buffer,
                                context,
                                descriptor_sets.clone(),
                                pipeline,
                            );
                        },
                    }
                }
            }
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, context, alpha_mode, subpass);
        }

        command_buffer
    }

    pub fn draw_primitive<'a, S>(
        &self,
        mesh: &Mesh<'a>,
        primitive: &Primitive<'a>,
        mut command_buffer: AutoCommandBufferBuilder,
        context: &DrawContext,
        sets: S,
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> AutoCommandBufferBuilder where S: DescriptorSetsCollection + Clone {
        let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
        let normals_accessor = primitive.get(&Semantic::Normals);
        let tangents_accessor = primitive.get(&Semantic::Tangents);
        // TODO: There may be multiple tex coord buffers per primitive
        let tex_coords_accessor = primitive.get(&Semantic::TexCoords(0));
        let indices_accessor = primitive.indices();
        // TODO: There may be multiple color buffers per primitive
        let vertex_color_accessor = primitive.get(&Semantic::Colors(0));

        let position_slice: BufferSlice<[GltfVertexPosition], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>
            = self.get_semantic_buffer_view(&positions_accessor);

        let normal_slice: BufferSlice<[GltfVertexNormal], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            let normal_slice: BufferSlice<[u8], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = normals_accessor.map(|normals_accessor| {
                self.get_semantic_buffer_view(&normals_accessor)
            }).unwrap_or_else(|| {
                let buffer = self.normal_buffers[mesh.index()][primitive.index()].as_ref()
                    .expect("No normals provided by the model and no normals were precomputed.");

                BufferSlice::from_typed_buffer_access(buffer.clone())
            });

            unsafe { normal_slice.reinterpret::<[GltfVertexNormal]>() }
        };

        let tangent_slice: BufferSlice<[GltfVertexTangent], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            let tangent_slice: BufferSlice<[u8], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = tangents_accessor.map(|tangents_accessor| {
                self.get_semantic_buffer_view(&tangents_accessor)
            }).unwrap_or_else(|| {
                let buffer = self.tangent_buffers[mesh.index()][primitive.index()].as_ref()
                    .expect("No tangents provided by the model and no tangents were precomputed.");

                BufferSlice::from_typed_buffer_access(buffer.clone())
            });

            unsafe { tangent_slice.reinterpret::<[GltfVertexTangent]>() }
        };

        let tex_coord_slice: BufferSlice<[GltfVertexTexCoord], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            if let &Some(ref tex_coord_accessor) = &tex_coords_accessor {
                self.get_semantic_buffer_view(tex_coord_accessor)
            } else {
                let zero_buffer = context.helper_resources.zero_buffer.clone();
                let zero_buffer_slice = BufferSlice::from_typed_buffer_access(zero_buffer);

                unsafe { zero_buffer_slice.reinterpret::<[GltfVertexTexCoord]>() }
            }
        };

        let vertex_color_slice: BufferSlice<[GltfVertexColor], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            if let &Some(ref vertex_color_accessor) = &vertex_color_accessor {
                self.get_semantic_buffer_view(vertex_color_accessor)
            } else {
                let zero_buffer = context.helper_resources.zero_buffer.clone();
                let zero_buffer_slice = BufferSlice::from_typed_buffer_access(zero_buffer);

                unsafe { zero_buffer_slice.reinterpret::<[GltfVertexColor]>() }
            }
        };

        let vertex_buffers = GltfVertexBuffers {
            position_buffer: Some(Arc::new(position_slice)),
            normal_buffer: Some(Arc::new(normal_slice)),
            tangent_buffer: Some(Arc::new(tangent_slice)),
            tex_coord_buffer: Some(Arc::new(tex_coord_slice)),
            vertex_color_buffer: Some(Arc::new(vertex_color_slice)),
        };

        let push_constants = PushConstants::new(
            vertex_color_accessor.is_some(),
        );

        if let Some(indices_accessor) = indices_accessor {
            macro_rules! draw_indexed {
                ($index_type:ty; $command_buffer:ident, $context:ident, $vertex_buffers:ident, $indices_accessor:ident, $sets:ident) => {
                    let index_slice: BufferSlice<[$index_type], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
                        let buffer_view = $indices_accessor.view();
                        let buffer_index = buffer_view.buffer().index();
                        let buffer_offset = $indices_accessor.offset() + buffer_view.offset();
                        let buffer_bytes = $indices_accessor.size() * $indices_accessor.count();

                        let index_buffer = self.device_buffers[buffer_index].clone();
                        let index_slice = BufferSlice::from_typed_buffer_access(index_buffer)
                            .slice(buffer_offset..(buffer_offset + buffer_bytes))
                            .unwrap();

                        unsafe { index_slice.reinterpret::<[$index_type]>() }
                    };

                    // unsafe {
                    //     let index_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(index_slice);
                    //     println!("index_slice: {:?}", index_slice);
                    // }

                    $command_buffer = $command_buffer.draw_indexed(
                        pipeline.clone(),
                        $context.dynamic,
                        $vertex_buffers.get_individual_buffers(),
                        index_slice,
                        $sets.clone(),
                        push_constants).unwrap();
                }
            }

            match indices_accessor.data_type() {
                DataType::U8 => {
                    let index_buffer = self.converted_index_buffers_by_accessor_index[indices_accessor.index()]
                        .as_ref()
                        .expect("Could not access a pre-generated `u16` index buffer, maybe it was not generated?")
                        .clone();
                    command_buffer = command_buffer.draw_indexed(
                        pipeline.clone(),
                        context.dynamic,
                        vertex_buffers.get_individual_buffers(),
                        index_buffer,
                        sets.clone(),
                        push_constants).unwrap();
                },
                DataType::U16 => {
                    draw_indexed!(u16; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                DataType::U32 => {
                    draw_indexed!(u32; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                _ => {
                    panic!("Index type not supported.");
                },
            }
        } else {
            command_buffer = command_buffer.draw(
                pipeline.clone(),
                context.dynamic,
                vertex_buffers.get_individual_buffers(),
                sets.clone(),
                push_constants).unwrap();
        }

        command_buffer
    }
}

// #[derive(Clone)]
// struct MeasuredDescriptorSetsCollection {
//     collection: Arc<dyn DescriptorSetsCollection>,
//     sets: usize,
// }

// #[derive(Clone)]
// struct DescriptorSetVec {
//     collections: Vec<MeasuredDescriptorSetsCollection>,
// }

// impl DescriptorSetVec {
//     pub fn new(slice: &[Arc<dyn DescriptorSetsCollection>]) -> Self {
//         let mut collections = Vec::with_capacity(slice.len());

//         for collection in slice {
//             let mut sets = 0;

//             while let Some(_) = collection.num_bindings_in_set(sets) {
//                 sets += 1;
//             }

//             collections.push(MeasuredDescriptorSetsCollection {
//                 collection: collection.clone(),
//                 sets,
//             });
//         }

//         DescriptorSetVec {
//             collections
//         }
//     }

//     pub fn collection_by_set(&self, set: usize) -> Option<(&MeasuredDescriptorSetsCollection, usize)> {
//         unimplemented!()
//     }
// }

// unsafe impl DescriptorSetsCollection for DescriptorSetVec {
//     fn into_vec(self) -> Vec<Box<DescriptorSet + Send + Sync>> {
//         let len = self.collections.iter().map(|collection| collection.sets).sum();
//         let mut result = Vec::with_capacity(len);

//         for measured_collection in self.collections.into_iter() {
//             let collection = measured_collection.collection;
//             let mut subresult = collection.into_vec();
//             result.append(&mut subresult);
//         }

//         result
//     }

//     fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
//         self.collection_by_set(set).and_then(|(collection, rel_index)| {
//             collection.collection.num_bindings_in_set(rel_index)
//         })
//     }

//     fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
//         self.collection_by_set(set).and_then(|(collection, rel_index)| {
//             collection.collection.descriptor(rel_index, binding)
//         })
//     }
// }

// #[derive(Clone)]
// struct DescriptorSetCollectionAppend<A, R>
//         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//               R: DescriptorSetsCollection {
//     rest_len: usize,
//     rest: R,
//     append: A,
// }

// // unsafe impl<A, R> Send for DescriptorSetCollectionAppend<A, R>
// //         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
// //               R: DescriptorSetsCollection + Send {}
// // unsafe impl<A, R> Sync for DescriptorSetCollectionAppend<A, R>
// //         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
// //               R: DescriptorSetsCollection + Sync {}

// impl<A, R> DescriptorSetCollectionAppend<A, R>
//         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//               R: DescriptorSetsCollection {
//     pub fn new(rest: R, append: A) -> Self {
//         let mut rest_len = 0;

//         while let Some(_) = rest.num_bindings_in_set(rest_len) {
//             rest_len += 1;
//         }

//         DescriptorSetCollectionAppend {
//             rest_len,
//             rest,
//             append,
//         }
//     }
// }

// unsafe impl<A, R> DescriptorSetsCollection for DescriptorSetCollectionAppend<A, R>
//     where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//           R: DescriptorSetsCollection {
//     fn into_vec(self) -> Vec<Box<DescriptorSet + Send + Sync>> {
//         let mut result: Vec<Box<DescriptorSet + Send + Sync>> = self.rest.into_vec();

//         result.push(Box::new(self.append));

//         result
//     }

//     fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
//         if set == self.rest_len {
//             Some(self.append.num_bindings())
//         } else {
//             self.rest.num_bindings_in_set(set)
//         }
//     }

//     fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
//         if set == self.rest_len {
//             self.append.descriptor(binding)
//         } else {
//             self.rest.descriptor(set, binding)
//         }
//     }
// }
