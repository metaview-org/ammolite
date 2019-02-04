use std::sync::Arc;
use std::ops::{BitOr, BitOrAssign, Not};
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::pipeline::blend::AttachmentBlend;
use vulkano::pipeline::blend::BlendFactor;
use vulkano::pipeline::blend::BlendOp;
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipelineBuilder};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::depth_stencil::Compare;
use vulkano::pipeline::depth_stencil::DepthBounds;
use vulkano::pipeline::shader::*;
use vulkano::pipeline::vertex::VertexDefinition;
use vulkano::framebuffer::RenderPassSubpassInterface;
use vulkano::swapchain::Swapchain;
use arrayvec::ArrayVec;
use winit::Window;
use crate::vertex::GltfVertexBufferDefinition;
use crate::shaders::*;

#[derive(PartialEq, Eq)]
#[repr(C)]
pub enum GraphicsPipelineFlag {
    DoubleSided,
    Len,
}

const GRAPHICS_PIPELINE_SET_LEN: usize = 1 << GraphicsPipelineFlag::Len as usize;

#[derive(Clone, Copy, Default, Debug)]
pub struct GraphicsPipelineFlags(usize);

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

#[derive(Clone)]
pub struct GraphicsPipelineSet(pub ArrayVec<[
    Arc<GraphicsPipelineAbstract + Send + Sync>;
    GRAPHICS_PIPELINE_SET_LEN
]>);

impl GraphicsPipelineSet {
    fn from_graphics_pipeline_builder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>(
        device: &Arc<Device>,
        graphics_pipeline_builder: GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>,
    ) -> GraphicsPipelineSet
        where Vdef: VertexDefinition<Vs::InputDefinition> + 'static + Send + Sync + Clone,
              Vs: GraphicsEntryPointAbstract + Clone,
              Fs: GraphicsEntryPointAbstract + Clone,
              Gs: GraphicsEntryPointAbstract + 'static + Clone,
              Tcs: GraphicsEntryPointAbstract + 'static + Clone,
              Tes: GraphicsEntryPointAbstract + 'static + Clone,
              Vss: SpecializationConstants + 'static + Clone,
              Tcss: SpecializationConstants + 'static + Clone,
              Tess: SpecializationConstants + 'static + Clone,
              Gss: SpecializationConstants + 'static + Clone,
              Fss: SpecializationConstants + 'static + Clone,
              Vs::PipelineLayout: 'static + Send + Sync + Clone,
              Fs::PipelineLayout: 'static + Send + Sync + Clone,
              Tcs::PipelineLayout: 'static + Send + Sync + Clone,
              Tes::PipelineLayout: 'static + Send + Sync + Clone,
              Gs::PipelineLayout: 'static + Send + Sync + Clone,
              Tcs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition> + 'static + Clone,
              Tes::InputDefinition: ShaderInterfaceDefMatch<Tcs::OutputDefinition> + 'static + Clone,
              Gs::InputDefinition: ShaderInterfaceDefMatch<Tes::OutputDefinition> + ShaderInterfaceDefMatch<Vs::OutputDefinition> + 'static + Clone,
              Fs::InputDefinition: ShaderInterfaceDefMatch<Gs::OutputDefinition> + ShaderInterfaceDefMatch<Tes::OutputDefinition> + ShaderInterfaceDefMatch<Vs::OutputDefinition> + 'static + Clone,
              Rp: RenderPassAbstract + RenderPassSubpassInterface<Fs::OutputDefinition> + 'static + Clone + Send + Sync {
        let mut pipelines = ArrayVec::<[
            Arc<GraphicsPipelineAbstract + Send + Sync>;
            GRAPHICS_PIPELINE_SET_LEN
        ]>::new();

        for i in 0..(1 << GraphicsPipelineFlag::Len as usize) {
            let mut builder = graphics_pipeline_builder.clone();

            macro_rules! flag {
                ($flag:ident in $i:expr) => {
                    ($i & (1 << GraphicsPipelineFlag::$flag as usize)) != 0
                }
            }

            if flag!(DoubleSided in i) {
                builder = builder.cull_mode_disabled();
            } else {
                builder = builder.cull_mode_back();
            }

            pipelines.push(Arc::new(builder.build(device.clone()).unwrap()));
        }

        GraphicsPipelineSet(pipelines)
    }

    pub fn get_pipeline(&self, flags: GraphicsPipelineFlags) -> &Arc<GraphicsPipelineAbstract + Send + Sync> {
        &self.0[flags.0]
    }

    fn create_gltf_opaque(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        shared_vs: &gltf_vert::Shader
    ) -> GraphicsPipelineSet {
        let fs = gltf_opaque_frag::Shader::load(device.clone()).expect("Failed to create shader module.");
        let builder/*FIXME remove : GraphicsPipelineBuilder<(), (), (), (), (), (), (), (), (), (), (), (), >*/ = GraphicsPipeline::start()
            // .with_pipeline_layout(device.clone(), pipeline_layout)
            // Specifies the vertex type
            .vertex_input(GltfVertexBufferDefinition)
            // .vertex_input_single_buffer::<Position>()
            .vertex_shader(shared_vs.main_entry_point(), ())
            // Configures the builder so that we use one viewport, and that the state of this viewport
            // is dynamic. This makes it possible to change the viewport for each draw command. If the
            // viewport state wasn't dynamic, then we would have to create a new pipeline object if we
            // wanted to draw to another image of a different size.
            //
            // Note: If you configure multiple viewports, you can use geometry shaders to choose which
            // viewport the shape is going to be drawn to. This topic isn't covered here.
            .viewports_dynamic_scissors_irrelevant(1)
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap());

        Self::from_graphics_pipeline_builder(device, builder)
    }

    fn create_gltf_mask(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        shared_vs: &gltf_vert::Shader
    ) -> GraphicsPipelineSet {
        let fs = gltf_mask_frag::Shader::load(device.clone()).expect("Failed to create shader module.");
        let builder = GraphicsPipeline::start()
            .vertex_input(GltfVertexBufferDefinition)
            .vertex_shader(shared_vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .depth_stencil(DepthStencil::simple_depth_test())
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 1).unwrap());

        Self::from_graphics_pipeline_builder(device, builder)
    }

    fn create_gltf_blend_preprocess(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        shared_vs: &gltf_vert::Shader
    ) -> GraphicsPipelineSet {
        let fs = gltf_blend_preprocess_frag::Shader::load(device.clone()).expect("Failed to create shader module.");
        let builder = GraphicsPipeline::start()
            .vertex_input(GltfVertexBufferDefinition)
            .vertex_shader(shared_vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
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
            .render_pass(Subpass::from(render_pass.clone(), 2).unwrap());

        GraphicsPipelineSet::from_graphics_pipeline_builder(device, builder)
    }

    fn create_gltf_blend_finalize(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        shared_vs: &gltf_vert::Shader
    ) -> GraphicsPipelineSet {
        let fs = gltf_blend_finalize_frag::Shader::load(device.clone()).expect("Failed to create shader module.");
        let builder = GraphicsPipeline::start()
            .vertex_input(GltfVertexBufferDefinition)
            .vertex_shader(shared_vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
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
            .render_pass(Subpass::from(render_pass.clone(), 3).unwrap());

        GraphicsPipelineSet::from_graphics_pipeline_builder(device, builder)
    }
}

#[derive(Clone)]
pub struct GraphicsPipelineSets {
    pub opaque: GraphicsPipelineSet,
    pub mask: GraphicsPipelineSet,
    pub blend_preprocess: GraphicsPipelineSet,
    pub blend_finalize: GraphicsPipelineSet,
}

impl GraphicsPipelineSets {
    pub fn create(device: &Arc<Device>,
                  swapchain: &Arc<Swapchain<Window>>) -> (Self, Arc<RenderPassAbstract + Send + Sync>) {
        let vertex_shader = gltf_vert::Shader::load(device.clone())
            .expect("Failed to create shader module.");
        let render_pass = Self::create_render_pass(device, swapchain);

        (Self {
            opaque: GraphicsPipelineSet::create_gltf_opaque(device, &render_pass, &vertex_shader),
            mask: GraphicsPipelineSet::create_gltf_mask(device, &render_pass, &vertex_shader),
            blend_preprocess: GraphicsPipelineSet::create_gltf_blend_preprocess(device, &render_pass, &vertex_shader),
            blend_finalize: GraphicsPipelineSet::create_gltf_blend_finalize(device, &render_pass, &vertex_shader),
        }, render_pass)
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
}
