pub mod error;
pub mod resource;
pub mod import;

use std::marker::PhantomData;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use std::path::Path;
use std::mem;
use std::ops::Deref;
use std::collections::HashMap;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::command_buffer::{DynamicState, AutoCommandBuffer, AutoCommandBufferBuilder, DrawIndirectCommand, DrawIndexedIndirectCommand};
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::collection::DescriptorSetsCollection;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::device::Device;
use vulkano::format::*;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassDescClearValues;
use vulkano::image::Dimensions;
use vulkano::image::ImageLayout;
use vulkano::image::ImageUsage;
use vulkano::image::MipmapsCount;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::traits::ImageViewAccess;
use vulkano::instance::QueueFamily;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::pipeline::input_assembly::Index;
use vulkano::sampler::Filter;
use vulkano::sampler::MipmapMode;
use vulkano::sampler::Sampler;
use vulkano::sampler::SamplerAddressMode;
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
use crate::pipeline::GltfGraphicsPipeline;
use crate::pipeline::DescriptorSetMap;
use crate::iter::ArrayIterator;
use self::error::*;
use self::resource::*;

// TODO: Figure out a better way to provide the clear values, as they shouldn't need to be
// specified by the end user
pub trait FramebufferWithClearValues<C>: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {}

impl<C, F> FramebufferWithClearValues<C> for F where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {}

#[derive(Clone)]
pub struct InstanceDrawContext<'a> {
    pub draw_context: &'a DrawContext<'a>,
    pub descriptor_set_map_instance: &'a DescriptorSetMap,
}

#[derive(Clone)]
pub struct DrawContext<'a> {
    pub device: &'a Arc<Device>,
    pub queue_family: &'a QueueFamily<'a>,
    pub pipeline_cache: &'a GraphicsPipelineSetCache,
    pub dynamic: &'a DynamicState,
    pub helper_resources: &'a HelperResources,
}

#[derive(Clone)]
pub struct HelperResources {
    pub empty_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub zero_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    pub cheapest_sampler: Arc<Sampler>,
}

impl HelperResources {
    pub fn new<'a, I>(device: &Arc<Device>, queue_families: I)
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
            // ImageUsage::all(),
            ImageUsage {
                input_attachment: true,
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
        ];

        let output = HelperResources {
            empty_image: empty_device_image,
            zero_buffer: zero_device_buffer,
            cheapest_sampler,
        };

        Ok(SimpleUninitializedResource::new(output, tasks))
    }
}

#[derive(Clone)]
pub enum DynamicIndexBuffer {
    U16(Arc<TypedBufferAccess<Content = [u16]> + Send + Sync + 'static>),
    U32(Arc<TypedBufferAccess<Content = [u32]> + Send + Sync + 'static>),
}

type DynamicIndirectBuffer = Arc<TypedBufferAccess<Content = [DrawIndirectCommand]> + Send + Sync + 'static>;
type DynamicIndexedIndirectBuffer = Arc<TypedBufferAccess<Content = [DrawIndexedIndirectCommand]> + Send + Sync + 'static>;

#[derive(Clone)]
pub enum ContextLessDrawCallBuffers {
    Simple,
    Indexed {
        index_buffer: DynamicIndexBuffer,
    },
    Indirect {
        indirect_buffer: DynamicIndirectBuffer,
    },
    IndexedIndirect {
        index_buffer: DynamicIndexBuffer,
        indexed_indirect_buffer: DynamicIndexedIndirectBuffer,
    },
}

#[derive(Clone)]
pub struct ContextLessDrawCall<Gp, Gpl, V, Cd>
    where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,
          Gpl: PipelineLayoutAbstract + Send + Sync + Clone,
          V: Clone,
          Cd: Clone,
{
    pipeline: Gp,
    pipeline_layout: Gpl,
    vertex_source: V,
    buffers: ContextLessDrawCallBuffers,
    /// Data to provide additional draw call information, eg. to compute resulting descriptor sets
    /// and push constants
    custom_data: Cd,
}

// FIXME: Error handling
/// Takes a `ContextLessDrawCall` and constructs a final draw call from it
pub trait DrawCallIssuer<Gpl>
        where Gpl: PipelineLayoutAbstract + Send + Sync + Clone, {
    type Context;
    type CustomData: Clone;

    fn issue_draw_call<Gp, V>(
        command_buffer_builder: AutoCommandBufferBuilder,
        dynamic: &DynamicState,
        context_less: ContextLessDrawCall<Gp, Gpl, V, Self::CustomData>,
        context: &Self::Context,
    ) -> AutoCommandBufferBuilder
        where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,
              V: Clone;
}

#[derive(Clone)]
pub struct GltfContextLessDescriptorSets {
    descriptor_set_scene: Arc<dyn DescriptorSet + Send + Sync>,
    /// The instance descriptor set is left out to be filled in by the `DrawCallIssuer`.
    descriptor_set_instance: (),
    descriptor_set_node: Arc<dyn DescriptorSet + Send + Sync>,
    descriptor_set_material: Arc<dyn DescriptorSet + Send + Sync>,
    /// The blend descriptor set is only specified in the last subpass
    descriptor_set_blend: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}

#[derive(Clone)]
pub struct GltfContextLessDrawCallCustomData {
    incomplete_descriptor_sets: GltfContextLessDescriptorSets,
    push_constants: PushConstants,
}

pub type GltfContextLessDrawCall = ContextLessDrawCall<
    Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    Arc<dyn PipelineLayoutAbstract + Send + Sync>,
    Vec<Arc<(dyn BufferAccess + Send + Sync + 'static)>>,
    GltfContextLessDrawCallCustomData,
>;

pub struct GltfDrawCallContext<'a> {
    pub descriptor_set_map_instance: &'a DescriptorSetMap,
}

pub struct GltfDrawCallIssuer<'a> {
    _marker: PhantomData<&'a ()>,
}

// FIXME: Should be more generic
impl<'a> DrawCallIssuer<Arc<dyn PipelineLayoutAbstract + Send + Sync>> for GltfDrawCallIssuer<'a> {
    type Context = GltfDrawCallContext<'a>;
    type CustomData = GltfContextLessDrawCallCustomData;

    fn issue_draw_call<Gp, V>(
        command_buffer_builder: AutoCommandBufferBuilder,
        dynamic: &DynamicState,
        context_less: ContextLessDrawCall<Gp, Arc<dyn PipelineLayoutAbstract + Send + Sync>, V, Self::CustomData>,
        context: &Self::Context,
    ) -> AutoCommandBufferBuilder
            where Gp: GraphicsPipelineAbstract + VertexSource<V> + Send + Sync + 'static + Clone,
                  V: Clone {
        let ContextLessDrawCall {
            pipeline, pipeline_layout, vertex_source, buffers, custom_data
        } = context_less;
        let GltfContextLessDrawCallCustomData {
            incomplete_descriptor_sets,
            push_constants: constants,
        } = custom_data;
        let GltfContextLessDescriptorSets {
            descriptor_set_scene,
            descriptor_set_node,
            descriptor_set_material,
            descriptor_set_blend,
            ..
        } = incomplete_descriptor_sets;
        let descriptor_set_instance = context.descriptor_set_map_instance.map
            .get(&pipeline_layout)
            .expect("A descriptor set has not been generated for one of the required pipelines.")
            .clone();

        if let Some(descriptor_set_blend) = descriptor_set_blend {
            let sets = (
                descriptor_set_scene,
                descriptor_set_instance,
                descriptor_set_node,
                descriptor_set_material,
                descriptor_set_blend,
            );

            // TODO: remove code duplication
            use ContextLessDrawCallBuffers::*;

            match buffers {
                Simple => command_buffer_builder.draw(
                    pipeline,
                    dynamic,
                    vertex_source,
                    sets,
                    constants,
                ).unwrap(),

                Indexed {
                    index_buffer: DynamicIndexBuffer::U16(index_buffer),
                } => command_buffer_builder.draw_indexed(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    sets,
                    constants,
                ).unwrap(),

                Indexed {
                    index_buffer: DynamicIndexBuffer::U32(index_buffer),
                } => command_buffer_builder.draw_indexed(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    sets,
                    constants,
                ).unwrap(),

                Indirect {
                    indirect_buffer,
                } => command_buffer_builder.draw_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),

                IndexedIndirect {
                    index_buffer: DynamicIndexBuffer::U16(index_buffer),
                    indexed_indirect_buffer,
                } => command_buffer_builder.draw_indexed_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    indexed_indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),

                IndexedIndirect {
                    index_buffer: DynamicIndexBuffer::U32(index_buffer),
                    indexed_indirect_buffer,
                } => command_buffer_builder.draw_indexed_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    indexed_indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),
            }
        } else {
            let sets = (
                descriptor_set_scene,
                descriptor_set_instance,
                descriptor_set_node,
                descriptor_set_material,
            );

            use ContextLessDrawCallBuffers::*;

            match buffers {
                Simple => command_buffer_builder.draw(
                    pipeline,
                    dynamic,
                    vertex_source,
                    sets,
                    constants,
                ).unwrap(),

                Indexed {
                    index_buffer: DynamicIndexBuffer::U16(index_buffer),
                } => command_buffer_builder.draw_indexed(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    sets,
                    constants,
                ).unwrap(),

                Indexed {
                    index_buffer: DynamicIndexBuffer::U32(index_buffer),
                } => command_buffer_builder.draw_indexed(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    sets,
                    constants,
                ).unwrap(),

                Indirect {
                    indirect_buffer,
                } => command_buffer_builder.draw_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),

                IndexedIndirect {
                    index_buffer: DynamicIndexBuffer::U16(index_buffer),
                    indexed_indirect_buffer,
                } => command_buffer_builder.draw_indexed_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    indexed_indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),

                IndexedIndirect {
                    index_buffer: DynamicIndexBuffer::U32(index_buffer),
                    indexed_indirect_buffer,
                } => command_buffer_builder.draw_indexed_indirect(
                    pipeline,
                    dynamic,
                    vertex_source,
                    index_buffer,
                    indexed_indirect_buffer,
                    sets,
                    constants,
                ).unwrap(),
            }
        }
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
    node_descriptor_sets: Vec<DescriptorSetMap>,
    material_descriptor_sets: Vec<DescriptorSetMap>,
    /// A `Vec` of lazily created `ContextLessDrawCall`s for each scene and subpass.
    scene_subpass_context_less_draw_calls: Vec<[RwLock<Option<Vec<GltfContextLessDrawCall>>>; 4]>,
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

    pub fn import<'a, I, S>(device: &Arc<Device>,
                            queue_families: I,
                            pipeline_cache: &GraphicsPipelineSetCache,
                            helper_resources: &HelperResources,
                            path: S) -> Result<SimpleUninitializedResource<Model>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone,
                  S: AsRef<Path> {
        import::import_model(device, queue_families, pipeline_cache, helper_resources, path)
    }

    pub fn get_subpass_alpha_modes() -> impl Iterator<Item=AlphaMode> {
        ArrayIterator::new([
            AlphaMode::Opaque,
            AlphaMode::Mask,
            AlphaMode::Blend,
            AlphaMode::Blend,
        ])
    }

    pub fn get_used_pipelines_layouts(&self, pipeline_cache: &GraphicsPipelineSetCache) -> Vec<Arc<GltfGraphicsPipeline>> {
        Self::get_pipelines_layouts(&self.document, pipeline_cache)
    }

    // TODO: Cache result
    pub fn get_pipelines_layouts(document: &Document, pipeline_cache: &GraphicsPipelineSetCache) -> Vec<Arc<GltfGraphicsPipeline>> {
        let mut pipelines = HashMap::new();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let properties = GraphicsPipelineProperties::from(&primitive, &material);
                let pipeline_set = pipeline_cache.get_or_create_pipeline(&properties);

                for (subpass, alpha_mode) in Self::get_subpass_alpha_modes().enumerate() {
                    if material.alpha_mode() == alpha_mode {
                        let pipeline = match (alpha_mode, subpass) {
                            (AlphaMode::Opaque, _) => &pipeline_set.opaque,
                            (AlphaMode::Mask, _) => &pipeline_set.mask,
                            (AlphaMode::Blend, 2) => &pipeline_set.blend_preprocess,
                            (AlphaMode::Blend, 3) => &pipeline_set.blend_finalize,
                            _ => panic!("Invalid alpha_mode/subpass combination."),
                        };

                        if !pipelines.contains_key(&pipeline.layout) {
                            pipelines.insert(pipeline.layout.clone(), pipeline.clone());
                        }
                    }
                }
            }
        }

        pipelines.into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    pub fn draw_main_scene(
        &self,
        command_buffer: AutoCommandBufferBuilder,
        instance_context: InstanceDrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
    ) -> Result<AutoCommandBufferBuilder, Error> {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(command_buffer, instance_context, alpha_mode, subpass, main_scene_index)
        } else {
            Err(ModelDrawError::NoDefaultScene.into())
        }
    }

    pub fn draw_scene(
        &self,
        mut command_buffer: AutoCommandBufferBuilder,
        instance_context: InstanceDrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
        scene_index: usize,
    ) -> Result<AutoCommandBufferBuilder, Error> {
        if scene_index >= self.document.scenes().len() {
            return Err(ModelDrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let draw_call_read_guard = self.get_or_create_draw_calls_subpass_scene(
            &instance_context.draw_context,
            alpha_mode,
            subpass,
            scene_index,
        )?;

        let context = GltfDrawCallContext {
            descriptor_set_map_instance: instance_context.descriptor_set_map_instance,
        };

        if let Some(ref draw_calls) = *draw_call_read_guard {
            for draw_call in draw_calls {
                command_buffer = GltfDrawCallIssuer::issue_draw_call(
                    command_buffer,
                    &instance_context.draw_context.dynamic,
                    draw_call.clone(),
                    &context,
                );
            }
        }

        Ok(command_buffer)
    }

    fn get_or_create_draw_calls_subpass_scene<'a>(
        &'a self,
        draw_context: &DrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
        scene_index: usize,
    ) -> Result<RwLockReadGuard<'a, Option<Vec<GltfContextLessDrawCall>>>, Error> {
        if scene_index >= self.document.scenes().len() {
            return Err(ModelDrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let subpass_context_less_draw_calls = &self.scene_subpass_context_less_draw_calls[scene_index];
        let context_less_draw_calls = &subpass_context_less_draw_calls[subpass as usize];

        loop {
            {
                let read_lock = context_less_draw_calls.read().unwrap();

                if read_lock.is_some() {
                    return Ok(read_lock);
                }
            }

            {
                let mut write_lock = context_less_draw_calls.write().unwrap();

                if write_lock.is_some() {
                    continue;
                }

                *write_lock = Some(
                    self.create_draw_calls_scene(draw_context, alpha_mode, subpass, scene_index)?
                );
            }
        }
    }

    fn create_draw_calls_scene(
        &self,
        draw_context: &DrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
        scene_index: usize,
    ) -> Result<Vec<GltfContextLessDrawCall>, Error> {
        if scene_index >= self.document.scenes().len() {
            return Err(ModelDrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let mut draw_call_accumulator = Vec::new();
        let scene = self.document.scenes().nth(scene_index).unwrap();

        for node in scene.nodes() {
            self.create_draw_calls_node(node, draw_context, alpha_mode, subpass, &mut draw_call_accumulator);
        }

        Ok(draw_call_accumulator)
    }

    fn create_draw_calls_node<'a>(
        &self,
        node: Node<'a>,
        draw_context: &DrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
        draw_call_accumulator: &mut Vec<GltfContextLessDrawCall>,
    ) {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let material = primitive.material();

                if material.alpha_mode() == alpha_mode {
                    let properties = GraphicsPipelineProperties::from(&primitive, &material);
                    let pipeline_set = draw_context.pipeline_cache.get_or_create_pipeline(&properties);
                    let pipeline = match (alpha_mode, subpass) {
                        (AlphaMode::Opaque, _) => &pipeline_set.opaque,
                        (AlphaMode::Mask, _) => &pipeline_set.mask,
                        (AlphaMode::Blend, 2) => &pipeline_set.blend_preprocess,
                        (AlphaMode::Blend, 3) => &pipeline_set.blend_finalize,
                        _ => panic!("Invalid alpha_mode/subpass combination."),
                    };

                    let material_descriptor_set = material.index().map(|material_index| {
                        self.material_descriptor_sets[material_index].map[&pipeline.layout].clone()
                    }).unwrap_or_else(|| {
                        pipeline.layout_dependent_resources.default_material_descriptor_set.clone()
                    });

                    let mut incomplete_descriptor_sets = GltfContextLessDescriptorSets {
                        descriptor_set_scene: pipeline.layout_dependent_resources.descriptor_set_scene.clone(),
                        descriptor_set_instance: (),
                        descriptor_set_node: self.node_descriptor_sets[node.index()].map[&pipeline.layout].clone(),
                        descriptor_set_material: material_descriptor_set,
                        descriptor_set_blend: None,
                    };

                    if let (AlphaMode::Blend, 3) = (alpha_mode, subpass) {
                        incomplete_descriptor_sets.descriptor_set_blend
                            = Some(pipeline.layout_dependent_resources.descriptor_set_blend.as_ref()
                                   .unwrap().clone());
                    }

                    let draw_call = self.create_draw_call_primitive(
                        &mesh,
                        &primitive,
                        draw_context,
                        incomplete_descriptor_sets,
                        &pipeline.pipeline,
                        &pipeline.layout_dependent_resources.layout,
                    );

                    draw_call_accumulator.push(draw_call);
                }
            }
        }

        for child in node.children() {
            self.create_draw_calls_node(child, draw_context, alpha_mode, subpass, draw_call_accumulator);
        }
    }

    fn create_draw_call_primitive<'a>(
        &self,
        mesh: &Mesh<'a>,
        primitive: &Primitive<'a>,
        draw_context: &DrawContext,
        incomplete_descriptor_sets: GltfContextLessDescriptorSets,
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
        pipeline_layout: &Arc<PipelineLayoutAbstract + Send + Sync>,
    ) -> GltfContextLessDrawCall {
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
                let zero_buffer = draw_context.helper_resources.zero_buffer.clone();
                let zero_buffer_slice = BufferSlice::from_typed_buffer_access(zero_buffer);

                unsafe { zero_buffer_slice.reinterpret::<[GltfVertexTexCoord]>() }
            }
        };

        let vertex_color_slice: BufferSlice<[GltfVertexColor], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            if let &Some(ref vertex_color_accessor) = &vertex_color_accessor {
                self.get_semantic_buffer_view(vertex_color_accessor)
            } else {
                let zero_buffer = draw_context.helper_resources.zero_buffer.clone();
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

        let buffers = if let Some(indices_accessor) = indices_accessor {
            macro_rules! reinterpret_index_buffer_as_dynamic {
                ($index_type:ty, $index_ident:ident; $indices_accessor:ident) => {{
                    // FIXME: Isn't there a helper function to use?
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
                    let index_slice: Arc<dyn TypedBufferAccess<Content=[$index_type]> + Send + Sync> = Arc::new(index_slice);

                    // unsafe {
                    //     let index_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(index_slice);
                    //     println!("index_slice: {:?}", index_slice);
                    // }

                    DynamicIndexBuffer::$index_ident(index_slice)
                }}
            }

            let index_buffer = match indices_accessor.data_type() {
                DataType::U8 => {
                    let index_buffer = self.converted_index_buffers_by_accessor_index[indices_accessor.index()]
                        .as_ref()
                        .expect("Could not access a pre-generated `u16` index buffer, maybe it was not generated?")
                        .clone();

                    DynamicIndexBuffer::U16(index_buffer)
                },
                DataType::U16 => {
                    reinterpret_index_buffer_as_dynamic!(u16, U16; indices_accessor)
                },
                DataType::U32 => {
                    reinterpret_index_buffer_as_dynamic!(u32, U32; indices_accessor)
                },
                _ => {
                    panic!("Index type not supported.");
                },
            };

            ContextLessDrawCallBuffers::Indexed {
                index_buffer,
            }
        } else {
            ContextLessDrawCallBuffers::Simple
        };

        ContextLessDrawCall {
            pipeline: pipeline.clone(),
            pipeline_layout: pipeline_layout.clone(),
            vertex_source: vertex_buffers.get_individual_buffers(),
            buffers,
            custom_data: GltfContextLessDrawCallCustomData {
                incomplete_descriptor_sets,
                push_constants,
            }
        }
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
