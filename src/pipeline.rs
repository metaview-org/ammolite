use std::sync::{Arc, RwLock};
use std::ops::{BitOr, BitOrAssign, Not};
use std::collections::HashMap;
use vulkano::format::*;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSet;
use vulkano::instance::QueueFamily;
use vulkano::image::{AttachmentImage, ImageUsage};
use vulkano::image::swapchain::SwapchainImage;
use vulkano::buffer::{TypedBufferAccess, BufferAccess, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::pipeline::blend::AttachmentBlend;
use vulkano::pipeline::blend::BlendFactor;
use vulkano::pipeline::blend::BlendOp;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::depth_stencil::Compare;
use vulkano::pipeline::depth_stencil::DepthBounds;
use vulkano::swapchain::Swapchain;
use winit::Window;
use gltf::material::Material;
use gltf::mesh::Primitive;
use failure::Error;
use crate::vertex::{GltfVertexBufferDefinition, VertexAttributePropertiesSet};
use crate::model::{FramebufferWithClearValues, HelperResources};
use crate::model::resource::{InitializationTask, SimpleUninitializedResource};
use crate::buffer::StagedBuffer;
use crate::shaders::*;

#[derive(PartialEq, Eq)]
#[repr(C)]
pub enum GraphicsPipelineFlag {
    DoubleSided,
    Len,
}

#[derive(Debug, Clone, Eq, PartialEq, Default, Hash)]
pub struct GraphicsPipelineFlags(usize);

impl<'a, 'b> From<&'b Material<'a>> for GraphicsPipelineFlags {
    fn from(material: &'b Material<'a>) -> Self {
        let mut result = GraphicsPipelineFlags::default();

        if material.double_sided() {
            result |= GraphicsPipelineFlag::DoubleSided;
        }

        result
    }
}

impl BitOr for GraphicsPipelineFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        GraphicsPipelineFlags(self.0 | rhs.0)
    }
}

impl BitOr<GraphicsPipelineFlag> for GraphicsPipelineFlags {
    type Output = Self;

    fn bitor(self, rhs: GraphicsPipelineFlag) -> Self {
        if rhs == GraphicsPipelineFlag::Len {
            panic!("Invalid graphics pipeline flag.");
        }

        GraphicsPipelineFlags(self.0 | (1 << rhs as usize))
    }
}

impl BitOrAssign for GraphicsPipelineFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<GraphicsPipelineFlag> for GraphicsPipelineFlags {
    fn bitor_assign(&mut self, rhs: GraphicsPipelineFlag) {
        if rhs == GraphicsPipelineFlag::Len {
            panic!("Invalid graphics pipeline flag.");
        }

        self.0 |= 1 << rhs as usize;
    }
}

impl Not for GraphicsPipelineFlags {
    type Output = Self;

    fn not(self) -> Self {
        GraphicsPipelineFlags(!self.0)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Default)]
pub struct GraphicsPipelineProperties {
    flags: GraphicsPipelineFlags,
    vertex_attribute_properties_set: VertexAttributePropertiesSet,
}

impl GraphicsPipelineProperties {
    pub fn from<'a>(primitive: &Primitive<'a>, material: &Material<'a>) -> Self {
        GraphicsPipelineProperties {
            flags: material.into(),
            vertex_attribute_properties_set: primitive.into(),
        }
    }
}

#[derive(Clone)]
pub struct SharedGltfGraphicsPipelineResources {
    device: Arc<Device>,
    pub helper_resources: HelperResources,
    pub scene_ubo_buffer: StagedBuffer<SceneUBO>,
    pub default_material_ubo_buffer: Arc<ImmutableBuffer<MaterialUBO>>,
    // TODO: Probaby add another type to group dimensions dependent resources into a single Option
    pub depth_image: Option<Arc<AttachmentImage<D32Sfloat>>>,
    pub blend_accumulation_image: Option<Arc<AttachmentImage<R32G32B32A32Sfloat>>>,
    pub blend_revealage_image: Option<Arc<AttachmentImage<R32G32B32A32Sfloat>>>,
}

impl SharedGltfGraphicsPipelineResources {
    pub fn new(device: Arc<Device>, helper_resources: HelperResources, queue_family: QueueFamily)
            -> Result<SimpleUninitializedResource<Self>, Error> {
        let scene_ubo = SceneUBO::default();
        let scene_ubo_buffer = StagedBuffer::from_data(
            &device,
            queue_family,
            BufferUsage::uniform_buffer(),
            scene_ubo.clone(),
        );
        let (device_default_material_ubo_buffer, default_material_ubo_buffer_initialization) = unsafe {
            ImmutableBuffer::<MaterialUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let mut tasks = vec![
            InitializationTask::MaterialDescriptorSet {
                data: MaterialUBO::default(),
                initialization_buffer: Arc::new(default_material_ubo_buffer_initialization),
            },
        ];

        Ok(SimpleUninitializedResource::new(Self {
            device,
            helper_resources,
            scene_ubo_buffer,
            default_material_ubo_buffer: device_default_material_ubo_buffer,
            depth_image: None,
            blend_accumulation_image: None,
            blend_revealage_image: None,
        }, tasks))
    }

    pub fn construct_swapchain_framebuffers(&mut self,
                                            render_pass: Arc<RenderPassAbstract + Send + Sync>,
                                            swapchain_images: &[Arc<SwapchainImage<Window>>])
            -> Vec<Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>> {
        swapchain_images.iter().map(|image| {
            Arc::new(Framebuffer::start(render_pass)
                     .add(image.clone()).unwrap()
                     .add(self.depth_image.as_ref().unwrap().clone()).unwrap()
                     .add(self.blend_accumulation_image.as_ref().unwrap().clone()).unwrap()
                     .add(self.blend_revealage_image.as_ref().unwrap().clone()).unwrap()
                     .build().unwrap()) as Arc<dyn FramebufferWithClearValues<_>>
        }).collect()
    }

    pub fn reconstruct_dimensions_dependent_images(&mut self, dimensions: [u32; 2]) {
        self.depth_image = Some(AttachmentImage::with_usage(
                self.device.clone(),
                dimensions.clone(),
                D32Sfloat,
                ImageUsage {
                    depth_stencil_attachment: true,
                    .. ImageUsage::none()
                }
        ).unwrap());
        self.blend_accumulation_image = Some(AttachmentImage::with_usage(
                self.device.clone(),
                dimensions.clone(),
                R32G32B32A32Sfloat,
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                }
        ).unwrap());
        self.blend_revealage_image = Some(AttachmentImage::with_usage(
                self.device.clone(),
                dimensions.clone(),
                R32G32B32A32Sfloat, //FIXME
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                }
        ).unwrap());
    }
}

// FIXME: May not be necessary, if `GltfGraphicsPipeline` has no generics
// pub trait GltfGraphicsPipelineAbstract {
//     fn pipeline(&self) -> &Arc<dyn GraphicsPipelineAbstract + Send + Sync>;

//     fn reconstruct_descriptor_sets(&mut self,
//                                    shared_resources: &SharedGltfGraphicsPipelineResources);

//     fn next_instance_descriptor_set(
//         &mut self,
//         buffer: Arc<TypedBufferAccess<Content=InstanceUBO> + Send + Sync>)
//             -> Arc<dyn DescriptorSet + Send + Sync>;
// }

// FIXME: Cleanup comments
#[derive(Clone)]
pub struct GltfGraphicsPipeline/*<Layout>
        where Layout: PipelineLayoutAbstract + 'static + Clone + Send + Sync*/ {
    // layout: Layout,
    pub pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    pub descriptor_set_scene: Arc<dyn DescriptorSet + Send + Sync>,
    pub descriptor_set_pool_instance: Arc<FixedSizeDescriptorSetsPool<Arc<dyn GraphicsPipelineAbstract + Send + Sync>>>,
    pub descriptor_set_blend: Arc<dyn DescriptorSet + Send + Sync>,
    pub default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
}

impl/*<Layout>*/ GltfGraphicsPipeline/*<Layout>
        where Layout: PipelineLayoutAbstract + 'static + Clone + Send + Sync*/ {
    pub fn from/*<VertexDefinition, Layout, RenderP>*/(
        // pipeline: GraphicsPipeline<VertexDefinition, Layout, RenderP>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        shared_resources: &SharedGltfGraphicsPipelineResources) -> Self
            // where Self: GraphicsPipelineAbstract,
            //       Layout: 'static + Send + Sync,
            //       VertexDefinition: 'static + VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> + Send + Sync,
            //       RenderP: 'static + RenderPassAbstract + Send + Sync {
                      {
        // let layout: Layout = pipeline.layout().clone();
        // let pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync> = Arc::new(pipeline);
        let descriptor_set_scene = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 0)
                .add_buffer(shared_resources.scene_ubo_buffer.device_buffer().clone()).unwrap()
                .build().unwrap()
        );
        let descriptor_set_pool_instance = Arc::new(
            FixedSizeDescriptorSetsPool::new(pipeline.clone(), 1)
        );
        let descriptor_set_blend = Arc::new(
            // FIXME: Use a layout instead
            PersistentDescriptorSet::start(pipeline.clone(), 4)
                .add_image(shared_resources.blend_accumulation_image.as_ref().unwrap().clone()).unwrap()
                .add_image(shared_resources.blend_revealage_image.as_ref().unwrap().clone()).unwrap()
                .build().unwrap()
        );
        let default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 3)
                .add_buffer(shared_resources.default_material_ubo_buffer.clone()).unwrap()
                .add_image(shared_resources.helper_resources.empty_image.clone()).unwrap()
                .add_sampler(shared_resources.helper_resources.cheapest_sampler.clone()).unwrap()
                .add_image(shared_resources.helper_resources.empty_image.clone()).unwrap()
                .add_sampler(shared_resources.helper_resources.cheapest_sampler.clone()).unwrap()
                .add_image(shared_resources.helper_resources.empty_image.clone()).unwrap()
                .add_sampler(shared_resources.helper_resources.cheapest_sampler.clone()).unwrap()
                .add_image(shared_resources.helper_resources.empty_image.clone()).unwrap()
                .add_sampler(shared_resources.helper_resources.cheapest_sampler.clone()).unwrap()
                .add_image(shared_resources.helper_resources.empty_image.clone()).unwrap()
                .add_sampler(shared_resources.helper_resources.cheapest_sampler.clone()).unwrap()
                .build().unwrap()
        );

        Self {
            // layout,
            pipeline,
            descriptor_set_scene,
            descriptor_set_pool_instance,
            descriptor_set_blend,
            default_material_descriptor_set,
        }
    }
// }

// impl/*<Layout>*/ GltfGraphicsPipelineAbstract for GltfGraphicsPipeline/*<Layout>
//         where Layout: PipelineLayoutAbstract + 'static + Clone + Send + Sync*/ {
    // fn pipeline(&self) -> &Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
    //     &self.pipeline
    // }

    fn reconstruct_descriptor_sets(&mut self, shared_resources: &SharedGltfGraphicsPipelineResources) {
        self.descriptor_set_scene = Arc::new(
            // FIXME: Use a layout instead
            PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_buffer(shared_resources.scene_ubo_buffer.device_buffer().clone()).unwrap()
                .build().unwrap()
        );
        self.descriptor_set_blend = Arc::new(
                // FIXME: Use a layout instead
            PersistentDescriptorSet::start(self.pipeline.clone(), 4)
                .add_image(shared_resources.blend_accumulation_image.as_ref().unwrap().clone()).unwrap()
                .add_image(shared_resources.blend_revealage_image.as_ref().unwrap().clone()).unwrap()
                .build().unwrap()
        );
    }

    fn next_instance_descriptor_set(
        &mut self,
        buffer: Arc<TypedBufferAccess<Content=InstanceUBO> + Send + Sync>)
            -> Arc<dyn DescriptorSet + Send + Sync> {
        Arc::new(self.descriptor_set_pool_instance.next()
                 .add_buffer(buffer).unwrap()
                 .build().unwrap())
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineSet {
    pub opaque: Arc<GltfGraphicsPipeline>,
    pub mask: Arc<GltfGraphicsPipeline>,
    pub blend_preprocess: Arc<GltfGraphicsPipeline>,
    pub blend_finalize: Arc<GltfGraphicsPipeline>,
    // pub opaque: Arc<dyn GltfGraphicsPipelineAbstract>,
    // pub mask: Arc<dyn GltfGraphicsPipelineAbstract>,
    // pub blend_preprocess: Arc<dyn GltfGraphicsPipelineAbstract>,
    // pub blend_finalize: Arc<dyn GltfGraphicsPipelineAbstract>,
}

// Consider improving the synchronization data type
pub struct GraphicsPipelineSetCache {
    pub map: Arc<RwLock<HashMap<GraphicsPipelineProperties, Arc<GraphicsPipelineSet>>>>,
    pub shared_resources: SharedGltfGraphicsPipelineResources,
    pub device: Arc<Device>,
    pub render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pub vertex_shader: gltf_vert::Shader, // Stored here to avoid unnecessary reloading
}

macro_rules! construct_pipeline_opaque {
    ($cache:expr, $graphics_pipeline_builder:expr, $shared_resources:expr) => {{
        let fs = gltf_opaque_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 0).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_mask {
    ($cache:expr, $graphics_pipeline_builder:expr, $shared_resources:expr) => {{
        let fs = gltf_mask_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 1).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_blend_preprocess {
    ($cache:expr, $graphics_pipeline_builder:expr, $shared_resources:expr) => {{
        let fs = gltf_blend_preprocess_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil {
                depth_compare: Compare::Less,
                depth_write: false,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .fragment_shader(fs.main_entry_point(), ())
            .blend_individual([
                AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Add,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                },
                AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::Zero,
                    color_destination: BlendFactor::OneMinusSrcAlpha,
                    alpha_op: BlendOp::Add,
                    alpha_source: BlendFactor::Zero,
                    alpha_destination: BlendFactor::OneMinusSrcAlpha,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                },
            ].into_iter().cloned())
            .render_pass(Subpass::from($cache.render_pass.clone(), 2).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_blend_finalize {
    ($cache:expr, $graphics_pipeline_builder:expr, $shared_resources:expr) => {{
        let fs = gltf_blend_finalize_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil {
                depth_compare: Compare::Less,
                depth_write: false,
                depth_bounds_test: DepthBounds::Disabled,
                stencil_front: Default::default(),
                stencil_back: Default::default(),
            })
            .fragment_shader(fs.main_entry_point(), ())
            .blend_individual([
                AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::OneMinusSrcAlpha,
                    color_destination: BlendFactor::SrcAlpha,
                    alpha_op: BlendOp::Add,
                    alpha_source: BlendFactor::OneMinusSrcAlpha,
                    alpha_destination: BlendFactor::SrcAlpha,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                },
            ].into_iter().cloned())
            .render_pass(Subpass::from($cache.render_pass.clone(), 3).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

impl GraphicsPipelineSetCache {
    pub fn create(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>, helper_resources: HelperResources, queue_family: QueueFamily) -> Self {
        let result = GraphicsPipelineSetCache {
            map: Arc::new(RwLock::new(HashMap::new())),
            shared_resources: SharedGltfGraphicsPipelineResources::new(device.clone(), helper_resources, queue_family),
            device: device.clone(),
            render_pass: Self::create_render_pass(device, swapchain),
            vertex_shader: gltf_vert::Shader::load(device.clone())
                .expect("Failed to create shader module."),
        };

        result.create_pipeline(&GraphicsPipelineProperties::default());
            // FIXME: Add proper error handling (see `Self::get_default_pipeline`)
            // .expect("Couldn't create a pipeline set for default properties.");

        result
    }

    fn create_render_pass(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> Arc<RenderPassAbstract + Send + Sync> {
        Arc::new(ordered_passes_renderpass! {
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                },
                depth_stencil: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D32Sfloat,
                    samples: 1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                },
                transparency_accumulation: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R32G32B32A32Sfloat,
                    samples: 1,
                    // initial_layout: ImageLayout::Undefined,
                    // final_layout: ImageLayout::General,
                },
                transparency_revealage: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R32G32B32A32Sfloat, //FIXME: Could be just a single channel
                    samples: 1,
                    // initial_layout: ImageLayout::Undefined,
                    // final_layout: ImageLayout::General,
                }
            },
            passes: [
                {
                    color: [color],
                    depth_stencil: { depth_stencil },
                    input: []
                    // $(resolve: [$($resolve_atch:ident),*])*$(,)*
                },
                {
                    color: [color],
                    depth_stencil: { depth_stencil },
                    input: []
                },
                {
                    color: [transparency_accumulation, transparency_revealage],
                    depth_stencil: { depth_stencil },
                    input: []
                },
                {
                    color: [color],
                    depth_stencil: { depth_stencil },
                    input: [transparency_accumulation, transparency_revealage]
                }
            ]
        }.expect("Could not create a render pass."))
    }

    pub fn get_pipeline(&self, properties: &GraphicsPipelineProperties) -> Option<Arc<GraphicsPipelineSet>> {
        self.map
            .as_ref()
            .read()
            .expect("The Graphics Pipeline Cache became poisoned.")
            .get(properties)
            .map(|pipeline| pipeline.clone())
    }

    // FIXME: There shouldn't be a need for this function, use the pipeline layout instead.
    pub fn get_default_pipeline(&self) -> Option<Arc<GraphicsPipelineSet>> {
        self.get_pipeline(&GraphicsPipelineProperties::default())
    }

    pub fn get_or_create_pipeline(&self, properties: &GraphicsPipelineProperties) -> Arc<GraphicsPipelineSet> {
        if let Some(pipeline) = self.get_pipeline(properties) {
            pipeline
        } else {
            self.create_pipeline(properties)
        }
    }

    pub fn create_pipeline(&self, properties: &GraphicsPipelineProperties) -> Arc<GraphicsPipelineSet> {
        let mut builder = GraphicsPipeline::start();

        macro_rules! flag {
            ($flag:ident in $i:expr) => {
                ($i.0 & (1 << GraphicsPipelineFlag::$flag as usize)) != 0
            }
        }

        if flag!(DoubleSided in properties.flags) {
            builder = builder.cull_mode_disabled();
        } else {
            builder = builder.cull_mode_back();
        }

        let mut map = self.map
            .as_ref()
            .write()
            .expect("The Graphics Pipeline Cache became poisoned.");

        let vertex_input = GltfVertexBufferDefinition {
            properties_set: properties.vertex_attribute_properties_set.clone(),
        };

        let builder = builder
            .vertex_input(vertex_input)
            .vertex_shader(self.vertex_shader.main_entry_point(), ())
            // Configures the builder so that we use one viewport, and that the state of this viewport
            // is dynamic. This makes it possible to change the viewport for each draw command. If the
            // viewport state wasn't dynamic, then we would have to create a new pipeline object if we
            // wanted to draw to another image of a different size.
            //
            // Note: If you configure multiple viewports, you can use geometry shaders to choose which
            // viewport the shape is going to be drawn to. This topic isn't covered here.
            .viewports_dynamic_scissors_irrelevant(1);

        let pipeline_set = Arc::new(GraphicsPipelineSet {
            opaque: construct_pipeline_opaque!(self, builder, &self.shared_resources),
            mask: construct_pipeline_mask!(self, builder, &self.shared_resources),
            blend_preprocess: construct_pipeline_blend_preprocess!(self, builder, &self.shared_resources),
            blend_finalize: construct_pipeline_blend_finalize!(self, builder, &self.shared_resources),
        });

        map.insert(properties.clone(), pipeline_set.clone());

        pipeline_set
    }
}
