use std::sync::{Arc, RwLock};
use std::ops::{BitOr, BitOrAssign, Not};
use std::collections::HashMap;
use vulkano::framebuffer::RenderPassAbstract;
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
use vulkano::swapchain::Swapchain;
use winit::Window;
use gltf::material::Material;
use gltf::mesh::Primitive;
use crate::vertex::{GltfVertexBufferDefinition, VertexAttributePropertiesSet};
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
pub struct GraphicsPipelineSet {
    pub opaque: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub mask: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub blend_preprocess: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub blend_finalize: Arc<GraphicsPipelineAbstract + Send + Sync>,
}

// Consider improving the synchronization data type
pub struct GraphicsPipelineSetCache {
    pub map: Arc<RwLock<HashMap<GraphicsPipelineProperties, Arc<GraphicsPipelineSet>>>>,
    pub device: Arc<Device>,
    pub render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pub vertex_shader: gltf_vert::Shader, // Stored here to avoid unnecessary reloading
}

macro_rules! construct_pipeline_opaque {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_opaque_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 0).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(pipeline)
    }}
}

macro_rules! construct_pipeline_mask {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
        let fs = gltf_mask_frag::Shader::load($cache.device.clone()).expect("Failed to create shader module.");
        let pipeline = $graphics_pipeline_builder.clone()
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from($cache.render_pass.clone(), 1).unwrap())
            .build($cache.device.clone())
            .unwrap();

        Arc::new(pipeline)
    }}
}

macro_rules! construct_pipeline_blend_preprocess {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
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

        Arc::new(pipeline)
    }}
}

macro_rules! construct_pipeline_blend_finalize {
    ($cache:expr, $graphics_pipeline_builder:expr) => {{
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

        Arc::new(pipeline)
    }}
}

impl GraphicsPipelineSetCache {
    pub fn create(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> Self {
        let result = GraphicsPipelineSetCache {
            map: Arc::new(RwLock::new(HashMap::new())),
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
            opaque: construct_pipeline_opaque!(self, builder),
            mask: construct_pipeline_mask!(self, builder),
            blend_preprocess: construct_pipeline_blend_preprocess!(self, builder),
            blend_finalize: construct_pipeline_blend_finalize!(self, builder),
        });

        map.insert(properties.clone(), pipeline_set.clone());

        pipeline_set
    }
}
