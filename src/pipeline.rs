use std::collections::hash_map::Entry;
use std::sync::{Arc, RwLock, Mutex};
use std::ops::{BitOr, BitOrAssign, Not};
use std::collections::HashMap;
use core::num::NonZeroU32;
use vulkano;
use vulkano::ordered_passes_renderpass;
use vulkano::format::*;
use vulkano::image::traits::ImageViewAccess;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetBuilder;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescAggregation;
use vulkano::instance::QueueFamily;

use vulkano::image::layout::RequiredLayouts;

use vulkano::image::Swizzle;
use vulkano::image::ImageDimensions;

use vulkano::image::ImageUsage;
use vulkano::image::MipmapsCount;
use vulkano::image::SyncImage;
use vulkano::image::view::ImageView;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::pipeline::blend::AttachmentBlend;
use vulkano::pipeline::blend::BlendFactor;
use vulkano::pipeline::blend::BlendOp;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::depth_stencil::Compare;
use vulkano::pipeline::depth_stencil::DepthBounds;


use gltf::material::Material;
use gltf::mesh::Primitive;
use failure::Error;
use fnv::FnvBuildHasher;
use crate::ViewSwapchain;
use crate::vertex::{GltfVertexBufferDefinition, VertexAttributePropertiesSet};
use crate::model::{FramebufferWithClearValues, HelperResources};
use crate::model::resource::{InitializationTask, UninitializedResource, SimpleUninitializedResource};
use crate::buffer::StagedBuffer;
use crate::iter::ArrayIterator;
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
pub struct SwapchainDependentResources {
    pub depth_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub blend_accumulation_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub blend_revealage_image: Arc<dyn ImageViewAccess + Send + Sync>,
}

#[derive(Clone)]
pub struct SharedGltfGraphicsPipelineResources {
    device: Arc<Device>,
    pub helper_resources: HelperResources,
    pub scene_ubo_buffer: StagedBuffer<SceneUBO>,
    pub default_material_ubo_buffer: Arc<ImmutableBuffer<MaterialUBO>>,
    pub swapchain_dependent_resources: Vec<Option<SwapchainDependentResources>>,
}

impl SharedGltfGraphicsPipelineResources {
    pub fn new(device: Arc<Device>, helper_resources: HelperResources, queue_family: QueueFamily, view_swapchains: &[&ViewSwapchain])
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
        let tasks = vec![
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
            swapchain_dependent_resources: vec![None; view_swapchains.len()],
        }, tasks))
    }

    pub fn construct_swapchain_framebuffers(&mut self,
                                            render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
                                            view_swapchain_index: usize,
                                            view_swapchain: &ViewSwapchain)
            -> Vec<Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>> {
        let render_pass = &render_pass;
        let resources = self.swapchain_dependent_resources[view_swapchain_index]
            .as_ref().expect("Framebuffer swapchain dependent images not initialized.");
        view_swapchain.swapchain.images().iter().map(|image| {
            Arc::new(Framebuffer::start(render_pass.clone())
                     .add(image.clone()).unwrap()
                     .add(resources.depth_image.clone()).unwrap()
                     .add(resources.blend_accumulation_image.clone()).unwrap()
                     .add(resources.blend_revealage_image.clone()).unwrap()
                     .build().unwrap()) as Arc<dyn FramebufferWithClearValues<_>>
        }).collect()
    }

    fn construct_attachment_image_view<F: FormatDesc>(
        &self,
        dimensions: [NonZeroU32; 2],
        format: F,
        usage: ImageUsage,
    ) -> Result<Arc<ImageView<SyncImage>>, Error> {
        let device_image = SyncImage::new(
            self.device.clone(),
            usage.clone(),
            format,
            ImageDimensions::Dim2D {
                width: dimensions[0],
                height: dimensions[1],
            },
            NonZeroU32::new(1).unwrap(),
            MipmapsCount::One,
        )?;

        let mut required_layouts = RequiredLayouts::none();
        // let mut required_layouts = RequiredLayouts::general();
        required_layouts.infer_mut(usage);
        // required_layouts.global = Some(typesafety::ImageLayoutEnd::ColorAttachmentOptimal);

        Ok(Arc::new(ImageView::new::<F>(
            device_image,
            None,
            None,
            Swizzle::identity(),
            None,
            required_layouts,
        )?))
    }

    pub fn reconstruct_dimensions_dependent_images(
        &mut self,
        view_swapchain_index: usize,
        view_swapchain: &ViewSwapchain,
    ) -> Result<(), Error> {
        let dimensions = view_swapchain.swapchain.dimensions();
        self.swapchain_dependent_resources[view_swapchain_index] = Some(SwapchainDependentResources {
            depth_image: self.construct_attachment_image_view(
                dimensions.clone(),
                D32Sfloat,
                ImageUsage {
                    depth_stencil_attachment: true,
                    .. ImageUsage::none()
                },
            )?,
            blend_accumulation_image: self.construct_attachment_image_view(
                dimensions.clone(),
                R32G32B32A32Sfloat,
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                }
            )?,
            blend_revealage_image: self.construct_attachment_image_view(
                dimensions.clone(),
                R32G32B32A32Sfloat, //FIXME
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                },
            )?,
        });

        Ok(())
    }
}

#[derive(Clone)]
pub struct GltfGraphicsPipeline {
    pub layout: Arc<PipelineLayout>,
    pub pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    pub layout_dependent_resources: GltfPipelineLayoutDependentResources,
}

impl GltfGraphicsPipeline {
    pub fn from(layout: Arc<PipelineLayout>,
                pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
                layout_dependent_resources: GltfPipelineLayoutDependentResources) -> Self {
        Self {
            layout,
            pipeline,
            layout_dependent_resources,
        }
    }
}

pub struct DescriptorSetMap {
    // Consider implementing a lightweight "map" that stores values linearly in a `Vec` until a
    // certain size is reached.
    pub map: HashMap<
        PipelineLayoutDescAggregation,
        Arc<dyn DescriptorSet + Send + Sync>,
        FnvBuildHasher,
    >,
}

impl DescriptorSetMap {
    pub fn custom<'a>(pipelines: impl IntoIterator<Item=&'a GltfGraphicsPipeline>,
                      set_provider: impl Fn(&'a GltfGraphicsPipeline)
                      -> Arc<dyn DescriptorSet + Send + Sync>) -> Self {
        let mut map = HashMap::<
            PipelineLayoutDescAggregation,
            Arc<dyn DescriptorSet + Send + Sync>,
            FnvBuildHasher,
        >::default();

        for pipeline in pipelines {
            match map.entry(pipeline.layout.desc().clone()) {
                Entry::Occupied(_) => (),
                Entry::Vacant(entry) => {
                    let set = (set_provider)(pipeline);

                    entry.insert(set);
                }
            }
        }

        Self { map }
    }

    pub fn new<'a>(pipelines: impl IntoIterator<Item=&'a GltfGraphicsPipeline>,
                   pool_provider: impl Fn(GltfPipelineLayoutDependentResources)
                                       -> Arc<Mutex<FixedSizeDescriptorSetsPool>>,
                   set_builder_fn: impl Fn(FixedSizeDescriptorSetBuilder<()>)
                                        -> Arc<dyn DescriptorSet + Send + Sync>) -> Self {
        Self::custom(pipelines, |pipeline| {
            let pool = pool_provider(pipeline.layout_dependent_resources.clone());
            let mut pool = pool.lock().unwrap();
            let set_builder = pool.next();

            (set_builder_fn)(set_builder)
        })
    }
}

#[derive(Clone)]
pub struct GltfPipelineLayoutDependentResources {
    pub layout: Arc<PipelineLayout>,
    pub descriptor_set_scene: Arc<dyn DescriptorSet + Send + Sync>,
    pub descriptor_set_pool_instance: Arc<Mutex<FixedSizeDescriptorSetsPool>>,
    pub descriptor_sets_blend: Option<Vec<Option<Arc<dyn DescriptorSet + Send + Sync>>>>,
    pub default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
}

impl GltfPipelineLayoutDependentResources {
    pub fn from(layout: Arc<PipelineLayout>,
                shared_resources: &SharedGltfGraphicsPipelineResources) -> Self {
        let descriptor_set_scene = Arc::new(
            PersistentDescriptorSet::start(layout.clone(), 0)
                .add_buffer(shared_resources.scene_ubo_buffer.device_buffer().clone()).unwrap()
                .build().unwrap()
        );
        let descriptor_set_pool_instance = Arc::new(Mutex::new(
            FixedSizeDescriptorSetsPool::new(layout.clone(), 1)
        ));
        let default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(layout.clone(), 3)
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
            layout,
            descriptor_set_scene,
            descriptor_set_pool_instance,
            descriptor_sets_blend: None, // late init with `reconstruct_descriptor_sets`
            default_material_descriptor_set,
        }
    }

    pub fn reconstruct_descriptor_sets(&mut self, shared_resources: &SharedGltfGraphicsPipelineResources, view_swapchains_len: usize, view_swapchain_index: usize, view_swapchain: &ViewSwapchain) {
        self.descriptor_set_scene = Arc::new(
            PersistentDescriptorSet::start(self.layout.clone(), 0)
                .add_buffer(shared_resources.scene_ubo_buffer.device_buffer().clone()).unwrap()
                .build().unwrap()
        );

        if self.descriptor_sets_blend.is_none() {
            self.descriptor_sets_blend = Some(vec![None; view_swapchains_len]);
        }

        self.descriptor_sets_blend.as_mut().unwrap()[view_swapchain_index] = self.layout.descriptor_set_layout(4).map(|_| {
            let swapchain_resources = shared_resources
                .swapchain_dependent_resources[view_swapchain_index]
                .as_ref().expect("Swapchain dependent resources not initialized.");

            Arc::new(PersistentDescriptorSet::start(self.layout.clone(), 4)
                .add_image(swapchain_resources.blend_accumulation_image.clone()).unwrap()
                .add_image(swapchain_resources.blend_revealage_image.clone()).unwrap()
                .build().unwrap()) as Arc<dyn DescriptorSet + Send + Sync>
        });
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineSet {
    pub opaque: GltfGraphicsPipeline,
    pub mask: GltfGraphicsPipeline,
    pub blend_preprocess: GltfGraphicsPipeline,
    pub blend_finalize: GltfGraphicsPipeline,
}

impl GraphicsPipelineSet {
    pub fn iter(&self) -> impl Iterator<Item=&GltfGraphicsPipeline> {
        ArrayIterator::new([
            &self.opaque,
            &self.mask,
            &self.blend_preprocess,
            &self.blend_finalize,
        ])
    }
}

// Consider improving the synchronization data type
pub struct GraphicsPipelineSetCache {
    pub pipeline_map: Arc<RwLock<HashMap<GraphicsPipelineProperties, GraphicsPipelineSet>>>,
    pub shared_resources: SharedGltfGraphicsPipelineResources,
    // TODO: Since the hashmap no longer has weak keys, the resources don't get freed automatically
    pub pipeline_layout_dependent_resources: Arc<RwLock<HashMap<
                                                 PipelineLayoutDescAggregation,
                                                 GltfPipelineLayoutDependentResources
                                             >>>,
    // pub pipeline_layout_dependent_resources: Arc<RwLock<WeakKeyHashMap<
    //                                              Weak<dyn PipelineLayoutDesc + Send + Sync>,
    //                                              GltfPipelineLayoutDependentResources
    //                                          >>>,
    pub device: Arc<Device>,
    pub render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pub vertex_shader: gltf_vert::Shader, // Stored here to avoid unnecessary reloading
}

macro_rules! cache_layout {
    ($cache:expr, $builder:expr) => {{
        let mut resources_map = $cache.pipeline_layout_dependent_resources
            .as_ref().write().expect("Layout dependent resources poisoned.");
        let layout_desc = $builder.construct_layout_desc(&[]).unwrap();
        let (layout, resources) = if let Some(resources) = resources_map.get(&layout_desc) {
            (resources.layout.clone(), resources.clone())
        } else {
            let layout: Arc<PipelineLayout> = layout_desc.clone()
                .build($cache.device.clone()).unwrap();
            let resources = GltfPipelineLayoutDependentResources::from(
                layout.clone(),
                &$cache.shared_resources,
            );

            resources_map.insert(layout_desc, resources.clone());

            (layout, resources)
        };

        let pipeline = Arc::new(
            $builder.with_pipeline_layout($cache.device.clone(), layout.clone())
                    .unwrap()
        );

        GltfGraphicsPipeline::from(layout, pipeline, resources)
    }}
}

macro_rules! construct_pipeline_opaque {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_opaque_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let builder = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 0).unwrap());

        cache_layout!($cache, builder)
            // .build($cache.device.clone())
            // .unwrap();

        // Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_mask {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_mask_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let builder = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 1).unwrap());

        cache_layout!($cache, builder)
            // .build($cache.device.clone())
            // .unwrap();

        // Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_blend_preprocess {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_blend_preprocess_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let builder = $graphics_pipeline_builder.clone()
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
            .render_pass(Subpass::from($cache.render_pass.clone(), 2).unwrap());

        cache_layout!($cache, builder)
            // .build($cache.device.clone())
            // .unwrap();

        // Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

macro_rules! construct_pipeline_blend_finalize {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_blend_finalize_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let builder = $graphics_pipeline_builder.clone()
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
            .render_pass(Subpass::from($cache.render_pass.clone(), 3).unwrap());

        cache_layout!($cache, builder)
            // .build($cache.device.clone())
            // .unwrap();

        // Arc::new(GltfGraphicsPipeline::from(Arc::new(pipeline), $shared_resources))
    }}
}

impl GraphicsPipelineSetCache {
    pub fn create(device: Arc<Device>, view_swapchains: &[&ViewSwapchain], helper_resources: HelperResources, queue_family: QueueFamily) -> impl UninitializedResource<Self> {
        let swapchain_format = view_swapchains[0].swapchain.format();

        for view_swapchain in view_swapchains.iter().skip(1) {
            assert_eq!(swapchain_format, view_swapchain.swapchain.format(), "All swapchains must use the same format.");
        }

        SharedGltfGraphicsPipelineResources::new(device.clone(), helper_resources, queue_family, view_swapchains)
            .unwrap()
            .map(move |shared_resources| {
                let result = GraphicsPipelineSetCache {
                    pipeline_map: Arc::new(RwLock::new(HashMap::new())),
                    shared_resources,
                    pipeline_layout_dependent_resources: Arc::new(RwLock::new(HashMap::new())),
                    // pipeline_layout_dependent_resources: Arc::new(RwLock::new(WeakKeyHashMap::new())),
                    device: device.clone(),
                    render_pass: Self::create_render_pass(&device, swapchain_format),
                    vertex_shader: gltf_vert::Shader::load(device.clone())
                        .expect("Failed to create shader module."),
                };

                // result.create_pipeline(&GraphicsPipelineProperties::default());
                    // FIXME: Add proper error handling (see `Self::get_default_pipeline`)
                    // .expect("Couldn't create a pipeline set for default properties.");

                result
            })
    }

    fn create_render_pass(device: &Arc<Device>, swapchain_format: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(ordered_passes_renderpass! {
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain_format,
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

    pub fn get_pipeline(&self, properties: &GraphicsPipelineProperties) -> Option<GraphicsPipelineSet> {
        self.pipeline_map
            .as_ref()
            .read()
            .expect("The Graphics Pipeline Cache became poisoned.")
            .get(properties)
            .map(|pipeline| pipeline.clone())
    }

    // FIXME: There shouldn't be a need for this function, use the pipeline layout instead.
    // pub fn get_default_pipeline(&self) -> Option<Arc<GraphicsPipelineSet>> {
    //     self.get_pipeline(&GraphicsPipelineProperties::default())
    // }

    pub fn get_or_create_pipeline(&self, properties: &GraphicsPipelineProperties) -> GraphicsPipelineSet {
        if let Some(pipeline) = self.get_pipeline(properties) {
            pipeline
        } else {
            self.create_pipeline(properties)
        }
    }

    pub fn create_pipeline(&self, properties: &GraphicsPipelineProperties) -> GraphicsPipelineSet {
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

        let mut pipeline_map = self.pipeline_map
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

        let pipeline_set = GraphicsPipelineSet {
            opaque: construct_pipeline_opaque!(self, builder),
            mask: construct_pipeline_mask!(self, builder),
            blend_preprocess: construct_pipeline_blend_preprocess!(self, builder),
            blend_finalize: construct_pipeline_blend_finalize!(self, builder),
        };

        pipeline_map.insert(properties.clone(), pipeline_set.clone());

        pipeline_set
    }
}
