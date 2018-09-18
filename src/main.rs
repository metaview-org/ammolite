#![feature(duration_as_u128)]

#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate det;
extern crate vulkano_win;
extern crate winit;
extern crate image;
extern crate typenum;
extern crate gltf;
extern crate byteorder;
extern crate rayon;
extern crate generic_array;
extern crate boolinator;

#[macro_use]
pub mod math;
pub mod model;
pub mod iter;
pub mod vertex;
pub mod camera;

use std::mem;
use std::collections::HashSet;
use std::sync::Arc;
use std::ops::Deref;
use std::ffi::CString;
use std::time::{Instant, Duration};
use vulkano::instance::RawInstanceExtensions;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, RawDeviceExtensions, DeviceExtensions, Queue};
use vulkano::instance::{Instance, PhysicalDevice, QueueFamily, Features};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::format::{self, Format, FormatTy};
use vulkano::image::{AttachmentImage, ImageUsage, Dimensions, ImageLayout};
use vulkano::image::immutable::{ImmutableImage, ImmutableImageInitialization};
use vulkano::image::swapchain::SwapchainImage;
use vulkano::framebuffer::{Framebuffer, RenderPass, RenderPassDesc, Subpass};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::swapchain::{self, PresentMode, SurfaceTransform, Swapchain, AcquireError, SwapchainCreationError, Surface};
use vulkano::sampler::{Sampler, SamplerAddressMode, BorderColor, MipmapMode, Filter};
use vulkano_win::VkSurfaceBuild;
use winit::{ElementState, MouseButton, Event, WindowEvent, KeyboardInput, VirtualKeyCode, EventsLoop, WindowBuilder, Window};
use image::{ImageBuffer, Pixel};
use math::matrix::*;
use math::vector::*;
use gltf::{Document, Gltf};
use gltf::mesh::util::ReadIndices;
use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian, LittleEndian};
use model::Model;
use model::DrawContext;
use model::InitializationDrawContext;
use model::UninitializedResource;
use model::SimpleUninitializedResource;
use model::HelperResources;
use vertex::GltfVertexBufferDefinition;
use camera::*;

pub type MainDescriptorSet<RPD: RenderPassDesc> = std::sync::Arc<vulkano::descriptor::descriptor_set::PersistentDescriptorSet<std::sync::Arc<vulkano::pipeline::GraphicsPipeline<GltfVertexBufferDefinition, std::boxed::Box<dyn vulkano::descriptor::PipelineLayoutAbstract + std::marker::Sync + std::marker::Send>, std::sync::Arc<vulkano::framebuffer::RenderPass<RPD>>>>, ((((), vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf<std::sync::Arc<vulkano::buffer::DeviceLocalBuffer<gltf_fs::ty::SceneUBO>>>), vulkano::descriptor::descriptor_set::PersistentDescriptorSetImg<std::sync::Arc<vulkano::image::AttachmentImage>>), vulkano::descriptor::descriptor_set::PersistentDescriptorSetSampler)>>;
pub type PipelineImpl<RPD: RenderPassDesc> = Arc<GraphicsPipeline<GltfVertexBufferDefinition, Box<(dyn PipelineLayoutAbstract + Sync + Send + 'static)>, Arc<RenderPass<RPD>>>>;

#[derive(Copy, Clone)]
pub struct Position {
    position: [f32; 3],
    tex_coord: [f32; 2],
}

impl_vertex!(Position, position, tex_coord);

#[derive(Copy, Clone)]
pub struct MainVertex {
    position: [f32; 3],
    tex_coord: [f32; 2],
}

impl_vertex!(MainVertex, position, tex_coord);

#[derive(Copy, Clone)]
struct ScreenVertex {
    position: [f32; 3],
}

impl_vertex!(ScreenVertex, position);

mod screen_vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[path = "src/shaders/screen.vert"]
    #[allow(dead_code)]
    struct Dummy;
}

mod screen_fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "src/shaders/screen.frag"]
    #[allow(dead_code)]
    struct Dummy;
}

mod gltf_fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "src/shaders/gltf.frag"]
    #[allow(dead_code)]
    struct Dummy;
}

mod gltf_vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[path = "src/shaders/gltf.vert"]
    #[allow(dead_code)]
    struct Dummy;
}

use gltf_fs::ty::*;

impl SceneUBO {
    pub fn new(dimensions: Vec2, model: Mat4, view: Mat4, projection: Mat4) -> SceneUBO {
        SceneUBO {
            dimensions: dimensions.0,
            _dummy0: Default::default(),
            model: model.0,
            view: view.0,
            projection: projection.0,
        }
    }
}

impl NodeUBO {
    pub fn new(matrix: Mat4) -> NodeUBO {
        NodeUBO {
            matrix: matrix.0,
        }
    }
}

impl MaterialUBO {
    pub fn new(base_color_factor: Vec4, metallic_factor: f32, roughness_factor: f32, base_color_texture_provided: bool) -> Self {
        MaterialUBO {
            base_color_factor: base_color_factor.0,
            metallic_factor,
            roughness_factor,
            base_color_texture_provided: base_color_texture_provided as u32,
        }
    }
}

impl Default for MaterialUBO {
    fn default() -> Self {
        Self::new(
            [1.0, 1.0, 1.0, 1.0].into(),
            1.0,
            1.0,
            false,
        )
    }
}

const SCREEN_DIMENSIONS: [u32; 2] = [3840, 1080];

fn vulkan_initialize<'a>(instance: &'a Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>, [u32; 2], Arc<Device>, QueueFamily<'a>, Arc<Queue>, Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    // TODO: Better device selection & SLI support
    let physical_device = PhysicalDevice::enumerate(instance).next().expect("No physical device available.");

    let events_loop = EventsLoop::new();
    // let displays = 
    // let display 
    // let surface = Surface::from_display_mode(

    // )
    let window = WindowBuilder::new()
        .with_title("metaview")
        .build_vk_surface(&events_loop, instance.clone()).unwrap();

    window.window().hide_cursor(true);

    let mut dimensions: [u32; 2] = {
        let (width, height) = window.window().get_inner_size().unwrap().into();
        [width, height]
    };

    // Queues are like CPU threads, queue families are groups of queues with certain capabilities.
    // println!("Available queue families:");

    // for queue_family in physical_device.queue_families() {
    //     println!("\tFamily #{} -- queues: {}, supports graphics: {}, supports compute: {}, supports transfers: {}, supports sparse binding: {}", queue_family.id(), queue_family.queues_count(), queue_family.supports_graphics(), queue_family.supports_compute(), queue_family.supports_transfers(), queue_family.supports_sparse_binding());
    // }

    let queue_family = physical_device.queue_families()
        .find(|&queue_family| {
            queue_family.supports_graphics() && window.is_supported(queue_family).unwrap_or(false)
        })
        .expect("Couldn't find a graphical queue family.");

    /*
     * Device Creation
     */

    // Create a device with a single queue
    let (device, mut queues) = {
        let safe_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            // ext_debug_marker: true,
            .. DeviceExtensions::none()
        };
        let raw_device_extensions = [/*CString::new("VK_EXT_debug_utils").unwrap()*/];
        let device_extensions = RawDeviceExtensions::new(raw_device_extensions.into_iter().cloned())
            .union(&(&safe_device_extensions).into());

        Device::new(physical_device,
                    &Features::none(),
                    device_extensions,
                    // A list of queues to use specified by an iterator of (QueueFamily, priority).
                    // In a real-life application, we would probably use at least a graphics queue
                    // and a transfers queue to handle data transfers in parallel. In this example
                    // we only use one queue.
                    [(queue_family, 0.5)].iter().cloned())
            .expect("Failed to create a Vulkan device.")
    };
    // println!("queues: {}", queues.len());
    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let capabilities = window.capabilities(physical_device)
            .expect("Failed to retrieve surface capabilities.");

        // Determines the behaviour of the alpha channel
        let alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
        dimensions = capabilities.current_extent.unwrap_or(dimensions);

        // Choosing the internal format that the images will have.
        let format = capabilities.supported_formats[0].0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            window.clone(),
            capabilities.min_image_count,
            format,
            dimensions,
            1,
            capabilities.supported_usage_flags,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Immediate, /* PresentMode::Relaxed TODO: Getting only ~60 FPS in a window */
            true,
            None,
        ).expect("failed to create swapchain")
    };

    (events_loop, window, dimensions, device, queue_family, queue, swapchain, images)
}

// fn vulkan_find_supported_format(physical_device: &Arc<PhysicalDevice>, candidates: &[Format]) -> Option<Format> {
//     // TODO: Querying available formats is not implemented (exposed) in vulkano.
//     unimplemented!()
// }

fn vulkan_main_pipeline(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> PipelineImpl<impl RenderPassDesc> {
    let main_vs = gltf_vs::Shader::load(device.clone()).expect("Failed to create shader module.");
    let main_fs = gltf_fs::Shader::load(device.clone()).expect("Failed to create shader module.");
    // let main_vs = main_vs::Shader::load(device.clone()).expect("Failed to create shader module.");
    // let main_fs = main_fs::Shader::load(device.clone()).expect("Failed to create shader module.");

    // A special GPU mode highly-optimized for rendering
    // This really shouldn't have to be Boxed, but making the type system happy is a struggle.
    let render_pass = Arc::new(single_pass_renderpass! { device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store, // for temporary images, use DontCare
                format: swapchain.format(),
                samples: 1,
            },
            depth_stencil: {
                load: Clear,
                store: DontCare,
                format: Format::D32Sfloat,
                samples: 1,
                initial_layout: ImageLayout::Undefined,
                final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                // initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
            }
        },
        pass: {
            color: [color],
            depth_stencil: { depth_stencil }
        }
    }.unwrap());

    Arc::new(GraphicsPipeline::start()
        // .with_pipeline_layout(device.clone(), pipeline_layout)
        // Specifies the vertex type
        .vertex_input(GltfVertexBufferDefinition)
        // .vertex_input_single_buffer::<Position>()
        .vertex_shader(main_vs.main_entry_point(), ())
        // Configures the builder so that we use one viewport, and that the state of this viewport
        // is dynamic. This makes it possible to change the viewport for each draw command. If the
        // viewport state wasn't dynamic, then we would have to create a new pipeline object if we
        // wanted to draw to another image of a different size.
        //
        // Note: If you configure multiple viewports, you can use geometry shaders to choose which
        // viewport the shape is going to be drawn to. This topic isn't covered here.
        .viewports_dynamic_scissors_irrelevant(1)
        .depth_stencil(DepthStencil::simple_depth_test())
        .fragment_shader(main_fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap())
}

fn create_staging_buffer_image<P, C>(device: &Arc<Device>, queue_family: QueueFamily, image: &ImageBuffer<P, C>)
    -> (Arc<CpuAccessibleBuffer<[u8]>>, Arc<ImmutableImage<format::R8G8B8A8Unorm>>, ImmutableImageInitialization<format::R8G8B8A8Unorm>, Arc<Sampler>)
    where P: Pixel<Subpixel=u8> + 'static,
          C: Deref<Target=[u8]> {
    // There should be a check whether the hardware supports the image format.
    // let pixel_size = P::channel_count() as usize * mem::size_of::<u8>();
    // let image_size = pixel_size * image.len();
    let staging_buffer = CpuAccessibleBuffer::<[u8]>::from_iter(
        device.clone(),
        BufferUsage {
            transfer_destination: true,
            transfer_source: true,
            .. BufferUsage::none()
        },
        Vec::from(&**image).into_iter(),
    ).unwrap();
    let (texture_image, initialization) = ImmutableImage::uninitialized(
        device.clone(),
        Dimensions::Dim2d {
            width: image.width(),
            height: image.height(),
        },
        format::R8G8B8A8Unorm,
        1,
        ImageUsage {
            transfer_destination: true,
            sampled: true,
            .. ImageUsage::none()
        },
        ImageLayout::ShaderReadOnlyOptimal,
        [queue_family].into_iter().cloned(),
    ).unwrap();
    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,  // magnifying filter
        Filter::Linear,  // minifying filter
        MipmapMode::Linear,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToEdge,
        0.0,  // mip_lod_bias
        // TODO: Turn anisotropic filtering on for better screen readability
        1.0,  // anisotropic filtering (1.0 = off, anything higher = on)
        1.0,  // min_lod
        1.0,  // max_lod
    ).unwrap();

    (staging_buffer, texture_image, initialization, sampler)
}

fn create_staging_buffers_data<T>(device: &Arc<Device>, queue_family: QueueFamily, usage: BufferUsage, data: T)
    -> (Arc<CpuAccessibleBuffer<T>>, Arc<DeviceLocalBuffer<T>>)
    where T: Sized + 'static {
    let staging_buffer = CpuAccessibleBuffer::<T>::from_data(
        device.clone(),
        BufferUsage {
            transfer_destination: true,
            transfer_source: true,
            .. usage.clone()
        },
        data,
    ).unwrap();
    let device_buffer = DeviceLocalBuffer::<T>::new(
        device.clone(),
        BufferUsage {
            transfer_destination: true,
            .. usage.clone()
        },
        [queue_family].into_iter().cloned(), // TODO: Figure out a way not to use a Vec
    ).unwrap();

    (staging_buffer, device_buffer)
}

fn create_staging_buffers_iter<T, I>(device: &Arc<Device>, queue_family: QueueFamily, usage: BufferUsage, iterator: I)
    -> (Arc<CpuAccessibleBuffer<[T]>>, Arc<DeviceLocalBuffer<[T]>>)
    where T: Clone + 'static,
          I: ExactSizeIterator<Item=T> {
    let iterator_len = iterator.len();
    let staging_buffer = CpuAccessibleBuffer::<[T]>::from_iter(
        device.clone(),
        BufferUsage {
            transfer_destination: true,
            transfer_source: true,
            .. usage.clone()
        },
        iterator,
    ).unwrap();
    let device_buffer = DeviceLocalBuffer::<[T]>::array(
        device.clone(),
        iterator_len,
        BufferUsage {
            transfer_destination: true,
            .. usage.clone()
        },
        [queue_family].into_iter().cloned(),
    ).unwrap();

    (staging_buffer, device_buffer)
}

fn create_vertex_index_buffers<V, I, VI, II>(device: &Arc<Device>, queue_family: QueueFamily, vertex_iterator: VI, index_iterator: II)
    -> ((Arc<CpuAccessibleBuffer<[V]>>, Arc<DeviceLocalBuffer<[V]>>),
        (Arc<CpuAccessibleBuffer<[I]>>, Arc<DeviceLocalBuffer<[I]>>))
    where V: vulkano::pipeline::vertex::Vertex + Clone + 'static,
          I: vulkano::pipeline::input_assembly::Index + Clone + 'static,
          VI: ExactSizeIterator<Item=V>,
          II: ExactSizeIterator<Item=I> {
    (
        create_staging_buffers_iter(
            &device,
            queue_family,
            BufferUsage::vertex_buffer(),
            vertex_iterator,
        ),
        create_staging_buffers_iter(
            &device,
            queue_family,
            BufferUsage::index_buffer(),
            index_iterator,
        ),
    )
}

fn vulkan_screen_pipeline(device: &Arc<Device>, swapchain: &Arc<Swapchain<Window>>) -> Arc<GraphicsPipeline<SingleBufferDefinition<ScreenVertex>, Box<dyn PipelineLayoutAbstract + Send + Sync>, Arc<RenderPass<impl RenderPassDesc>>>> {
    // A special GPU mode highly-optimized for rendering
    let screen_vs = screen_vs::Shader::load(device.clone()).expect("Failed to create shader module.");
    let screen_fs = screen_fs::Shader::load(device.clone()).expect("Failed to create shader module.");

    let render_pass = Arc::new(single_pass_renderpass! { device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store, // for temporary images, use DontCare
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    }.unwrap());

    // let pipeline_layout = PipelineLayour

    Arc::new(GraphicsPipeline::start()
        // .with_pipeline_layout(device.clone(), pipeline_layout)
        // Specifies the vertex type
        .vertex_input_single_buffer::<ScreenVertex>()
        .vertex_shader(screen_vs.main_entry_point(), ())
        // Configures the builder so that we use one viewport, and that the state of this viewport
        // is dynamic. This makes it possible to change the viewport for each draw command. If the
        // viewport state wasn't dynamic, then we would have to create a new pipeline object if we
        // wanted to draw to another image of a different size.
        //
        // Note: If you configure multiple viewports, you can use geometry shaders to choose which
        // viewport the shape is going to be drawn to. This topic isn't covered here.
        .viewports([Viewport {
            origin: [0.0, 0.0],
            dimensions: [SCREEN_DIMENSIONS[0] as f32, SCREEN_DIMENSIONS[1] as f32],
            depth_range: 0.0 .. 1.0,
        }].into_iter().cloned())
        .fragment_shader(screen_fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap())
}

fn construct_model_matrix(scale: f32, translation: &Vec3, rotation: &Vec3) -> Mat4 {
    Mat4::translation(translation)
        * Mat4::rotation_roll(rotation[2])
        * Mat4::rotation_yaw(rotation[1])
        * Mat4::rotation_pitch(rotation[0])
        * Mat4::scale(scale)
}

pub fn construct_view_matrix(translation: &Vec3, rotation: &Vec3) -> Mat4 {
    // construct_model_matrix(1.0, &-translation, &Vec3::zero())
    construct_model_matrix(1.0, &-translation, &-rotation)
}

#[allow(unused)]
fn construct_orthographic_projection_matrix(near_plane: f32, far_plane: f32, dimensions: Vec2) -> Mat4 {
    let z_n = near_plane;
    let z_f = far_plane;

    // Scale the X/Y-coordinates according to the dimensions. Translate and scale the Z-coordinate.
    mat4!([1.0 / dimensions[0],                  0.0,               0.0,                0.0,
                           0.0, -1.0 / dimensions[1],               0.0,                0.0,
                           0.0,                  0.0, 1.0 / (z_f - z_n), -z_n / (z_f - z_n),
                           0.0,                  0.0,               0.0,                1.0])
}

#[allow(unused)]
fn construct_perspective_projection_matrix(near_plane: f32, far_plane: f32, aspect_ratio: f32, fov_rad: f32) -> Mat4 {
    // The resulting `(x, y, z, w)` vector gets normalized following the execution of the vertex
    // shader to `(x/w, y/w, z/w)` (W-division). This makes it possible to create a perspective
    // projection matrix.
    // We copy the Z coordinate to the W coordinate so as to divide all coordinates by Z.
    let z_n = near_plane;
    let z_f = far_plane;
    // With normalization, it is actually `1 / (z * tan(FOV / 2))`, which is the width of the
    // screen at that point in space of the vector.
    // The X coordinate needs to be divided by the aspect ratio to make it independent of the
    // window size.
    // The Y coordinate is negated so as to adjust the vectors to the Vulkan coordinate system,
    // which has the Y axis pointing downwards, contrary to OpenGL.
    let f = 1.0 / (fov_rad / 2.0).tan();

    // We derive the coefficients for the Z coordinate from the following equation:
    // `f(z) = A*z + B`, because we know we need to translate and scale the Z coordinate.
    // The equation changes to the following, after the W-division:
    // `f(z) = A + B/z`
    // And must satisfy the following conditions:
    // `f(z_near) = 0`
    // `f(z_far) = 1`
    // Solving for A and B gives us the necessary coefficients to construct the matrix.
    mat4!([f / aspect_ratio, 0.0,                0.0,                       0.0,
                        0.0,  -f,                0.0,                       0.0,
                        0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
                        0.0, 0.0,                1.0,                       0.0])
}

fn main() {
    /*
     * Initialization
*/

    // TODO: Explore method arguments
    let win_extensions = vulkano_win::required_extensions();
    let raw_extensions = [/*CString::new("VK_EXT_debug_marker").unwrap()*/];
    let extensions = RawInstanceExtensions::new(raw_extensions.into_iter().cloned())
        .union(&(&win_extensions).into());
    let instance = Instance::new(None, extensions, None)
        .expect("Failed to create a Vulkan instance.");

    let (mut events_loop, window, mut dimensions, device, queue_family, queue, mut swapchain, mut images) = vulkan_initialize(&instance);

    let main_vertices = [
        MainVertex { position: [-0.5, -0.5,  0.0], tex_coord: [0.0, 1.0] },
        MainVertex { position: [ 0.5, -0.5,  0.0], tex_coord: [1.0, 1.0] },
        MainVertex { position: [ 0.5,  0.5,  0.0], tex_coord: [1.0, 0.0] },
        MainVertex { position: [-0.5,  0.5,  0.0], tex_coord: [0.0, 0.0] },

        MainVertex { position: [-0.5, -0.5, -0.5], tex_coord: [0.0, 1.0] },
        MainVertex { position: [ 0.5, -0.5, -0.5], tex_coord: [1.0, 1.0] },
        MainVertex { position: [ 0.5,  0.5, -0.5], tex_coord: [1.0, 0.0] },
        MainVertex { position: [-0.5,  0.5, -0.5], tex_coord: [0.0, 0.0] },
    ];

    let main_indices = [
        0, 1, 2,
        2, 3, 0,

        4, 5, 6,
        6, 7, 4u16,
    ];

    // let nom_obj::model::Interleaved { v_vt_vn, idx } = obj.objects[0].interleaved();

    // let main_vertices: Vec<MainVertex> = v_vt_vn.iter()
    //     .map(|&(v, vt, vn)| MainVertex { position: [v.0, v.1, v.2], tex_coord: [vt.0, 1.0 - vt.1] })
    //     .collect();

    // let main_indices: Vec<u32> = idx.iter()
    //     .map(|x| *x as u32)
    //     .collect();

    let (
        (main_vertex_staging_buffer, main_vertex_device_buffer),
        (main_index_staging_buffer, main_index_device_buffer),
    ) = create_vertex_index_buffers(
        &device,
        queue_family,
        // main_vertices.into_iter(),
        // main_indices.into_iter(),
        main_vertices.into_iter().cloned(),
        main_indices.into_iter().cloned(),
    );

    let screen_vertices = [
        ScreenVertex { position: [-1.0, -1.0,  0.0] },
        ScreenVertex { position: [-1.0,  1.0,  0.0] },
        ScreenVertex { position: [ 1.0,  1.0,  0.0] },
        ScreenVertex { position: [ 1.0, -1.0,  0.0] },
    ];

    let screen_indices = [
        0, 1, 2u16,
        // 2, 3, 0u16,
    ];

    let (
        (screen_vertex_staging_buffer, screen_vertex_device_buffer),
        (screen_index_staging_buffer, screen_index_device_buffer),
    ) = create_vertex_index_buffers(
        &device,
        queue_family,
        screen_vertices.into_iter().cloned(),
        screen_indices.into_iter().cloned(),
    );

    let main_pipeline = vulkan_main_pipeline(&device, &swapchain);
    let screen_pipeline = vulkan_screen_pipeline(&device, &swapchain);

    let mut main_ubo = SceneUBO::new(
        Vec2([dimensions[0] as f32, dimensions[1] as f32]),
        Mat4::identity(),
        Mat4::identity(),
        Mat4::identity(),
    );
    let (main_ubo_staging_buffer, main_ubo_device_buffer) = create_staging_buffers_data(
        &device,
        queue_family,
        BufferUsage::uniform_buffer(),
        main_ubo.clone(),
    );

    let screen_image = AttachmentImage::with_usage(
        device.clone(),
        SCREEN_DIMENSIONS.clone(),
        swapchain.format(),
        ImageUsage {
            sampled: true,
            .. ImageUsage::none()
        }
    ).unwrap();
    let border_color = match swapchain.format().ty() {
        FormatTy::Uint | FormatTy::Sint => BorderColor::IntTransparentBlack,
                                      _ => BorderColor::FloatTransparentBlack,
    };
    let screen_sampler = Sampler::new(
        device.clone(),
        Filter::Nearest,  // magnifying filter
        Filter::Linear,  // minifying filter
        MipmapMode::Nearest,
        SamplerAddressMode::ClampToBorder(border_color),
        SamplerAddressMode::ClampToBorder(border_color),
        SamplerAddressMode::ClampToBorder(border_color),
        0.0,  // mip_lod_bias
        // TODO: Turn anisotropic filtering on for better screen readability
        1.0,  // anisotropic filtering (1.0 = off, anything higher = on)
        1.0,  // min_lod
        1.0,  // max_lod
    ).unwrap();
    let main_descriptor_set = Arc::new(
        PersistentDescriptorSet::start(main_pipeline.clone(), 0)
            .add_buffer(main_ubo_device_buffer.clone()).unwrap()
            .add_sampled_image(screen_image.clone(), screen_sampler.clone()).unwrap()
            .build().unwrap()
    );

    let screen_framebuffers: Vec<Arc<Framebuffer<_, _>>> = images.iter().map(|_| {
        Arc::new(
            Framebuffer::start(screen_pipeline.render_pass().clone())
            .add(screen_image.clone()).unwrap()
            .build().unwrap()
        )
    }).collect();
    let mut main_framebuffers: Option<Vec<Arc<Framebuffer<_, _>>>> = None;
    let mut depth_image: Option<Arc<AttachmentImage>> = None;

    // We need to keep track of whether the swapchain is invalid for the current window,
    // for example when the window is resized.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone()));

    let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    let (init_command_buffer_builder, mut helper_resources) = HelperResources::new(
        &device,
        [queue_family].into_iter().cloned(),
        main_pipeline.clone(),
    ).unwrap().initialize_resource(
        &device,
        init_command_buffer_builder,
    ).unwrap();
    let (init_command_buffer_builder, mut model) = {
        let model_path = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!("No model path provided.");
            std::process::exit(1);
        });
        Model::import(
            &device,
            [queue_family].into_iter().cloned(),
            main_pipeline.clone(),
            &helper_resources,
            model_path,
        ).unwrap().initialize_resource(
            &device,
            init_command_buffer_builder
        ).unwrap()
    };
    let init_command_buffer = init_command_buffer_builder.build().unwrap();
    // let standard_command_pool = Device::standard_command_pool(&device, queue_family);
    // let init_unsafe_command_buffer = UnsafeCommandBufferBuilder::new(&standard_command_pool,
    //                                                                  Kind::primary(),
    //                                                                  Flags::OneTimeSubmit).unwrap()
    //     .build().unwrap();

    previous_frame_end = Box::new(previous_frame_end
        .then_execute(queue.clone(), init_command_buffer).unwrap()
        // .then_signal_fence()
        // .then_execute_same_queue(init_unsafe_command_buffer).unwrap()
        .then_signal_fence_and_flush().unwrap());
    let mut init_instant = Instant::now();
    let mut previous_frame_instant = init_instant.clone();
    let mut cursor_position: (f64, f64) = (dimensions[0] as f64, dimensions[1] as f64);
    let mut cursor_delta: (f64, f64) = (0.0, 0.0);
    let mut camera = PitchYawCamera3::new();
    let mut pressed_keys: HashSet<VirtualKeyCode> = HashSet::new();
    let mut pressed_mouse_buttons: HashSet<MouseButton> = HashSet::new();

    loop {
        let now = Instant::now();
        let delta_time = now.duration_since(previous_frame_instant);
        previous_frame_instant = now;
        let cursor_delta = {
            let (mut x, mut y) = cursor_position.into();
            x -= dimensions[0] as f64 / 2.0;
            y -= dimensions[1] as f64 / 2.0;
            (x, y)
        };

        camera.update(&delta_time, &cursor_delta, &pressed_keys, &pressed_mouse_buttons);

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            // println!("Recreating the swapchain.");

            dimensions = {
                let (width, height) = window.window().get_inner_size().unwrap().into();
                [width, height]
            };

            main_ubo.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            images = new_images;
            main_framebuffers = None;
            recreate_swapchain = false;
        }

        // Because framebuffers contains an Arc on the old swapchain, we need to
        // recreate framebuffers as well.
        if main_framebuffers.is_none() {
            depth_image = Some(AttachmentImage::with_usage(
                device.clone(),
                dimensions.clone(),
                Format::D32Sfloat,
                ImageUsage {
                    depth_stencil_attachment: true,
                    .. ImageUsage::none()
                }
            ).unwrap());
            main_framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(main_pipeline.render_pass().clone())
                         .add(image.clone()).unwrap()
                         .add(depth_image.as_ref().unwrap().clone()).unwrap()
                         .build().unwrap())
            }).collect::<Vec<_>>());
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok((image_num, acquire_future)) => {
                (image_num, acquire_future)
            },
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let nanoseconds_elapsed: u128 = delta_time.as_nanos();
        let seconds_elapsed: f64 = (nanoseconds_elapsed as f64) / 1.0e9;
        main_ubo = SceneUBO::new(
            Vec2([dimensions[0] as f32, dimensions[1] as f32]),
            construct_model_matrix(1.0,
                                   &[1.0, 0.0, 2.0].into(),
                                   &[seconds_elapsed as f32 * 0.0, seconds_elapsed as f32 * 0.0, 0.0].into()),
            camera.get_view_matrix(),
            // construct_view_matrix(&[(seconds_elapsed as f32 * 0.5).cos(), 0.0, 0.0].into(),
            //                       &[0.0, 0.0, 0.0].into()),
            construct_perspective_projection_matrix(0.1, 1000.0, dimensions[0] as f32 / dimensions[1] as f32, std::f32::consts::FRAC_PI_2),
            // construct_orthographic_projection_matrix(0.1, 1000.0, [dimensions[0] as f32 / dimensions[1] as f32, 1.0].into()),
        );

        let screen_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .copy_buffer(screen_vertex_staging_buffer.clone(), screen_vertex_device_buffer.clone()).unwrap()
            .copy_buffer(screen_index_staging_buffer.clone(), screen_index_device_buffer.clone()).unwrap()
            .begin_render_pass(screen_framebuffers[image_num].clone(),
                               false,
                               vec![[0.0, 1.0, 0.0, 1.0].into()]).unwrap()
            .draw_indexed(screen_pipeline.clone(),
                  &DynamicState::none(),
                  screen_vertex_device_buffer.clone(),
                  screen_index_device_buffer.clone(),
                  (), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        //let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        //    .update_buffer(main_ubo_staging_buffer.clone(), main_ubo.clone()).unwrap()
        //    .copy_buffer(main_ubo_staging_buffer.clone(), main_ubo_device_buffer.clone()).unwrap()
        //    .copy_buffer(main_vertex_staging_buffer.clone(), main_vertex_device_buffer.clone()).unwrap()
        //    .copy_buffer(main_index_staging_buffer.clone(), main_index_device_buffer.clone()).unwrap()
        //    // Before we can draw, we have to *enter a render pass*. There are two methods to do
        //    // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
        //    // not covered here.
        //    //
        //    // The third parameter builds the list of values to clear the attachments with. The API
        //    // is similar to the list of attachments when building the framebuffers, except that
        //    // only the attachments that use `load: Clear` appear in the list.
        //    .begin_render_pass(main_framebuffers.as_ref().unwrap()[image_num].clone(), false,
        //                       vec![[0.0, 0.0, 1.0, 1.0].into(), 1.0.into()])
        //    .unwrap()
        //    // We are now inside the first subpass of the render pass. We add a draw command.
        //    //
        //    // The last two parameters contain the list of resources to pass to the shaders.
        //    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
        //    .draw_indexed(main_pipeline.clone(),
        //          &DynamicState {
        //              line_width: None,
        //              // TODO: Find a way to do this without having to dynamically allocate a Vec every frame.
        //              viewports: Some(vec![Viewport {
        //                  origin: [0.0, 0.0],
        //                  dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        //                  depth_range: 0.0 .. 1.0,
        //              }]),
        //              scissors: None,
        //          },
        //          main_vertex_device_buffer.clone(),
        //          main_index_device_buffer.clone(),
        //          main_descriptor_set.clone(),
        //          ())
        //    .unwrap()
        //    // We leave the render pass by calling `draw_end`. Note that if we had multiple
        //    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
        //    // next subpass.
        //    .end_render_pass()
        //    .unwrap()
        //    // Finish building the command buffer by calling `build`.
        //    .build().unwrap();

        let buffer_updates = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
           .update_buffer(main_ubo_staging_buffer.clone(), main_ubo.clone()).unwrap()
           .copy_buffer(main_ubo_staging_buffer.clone(), main_ubo_device_buffer.clone()).unwrap()
           .build().unwrap();

        let current_framebuffer: Arc<Framebuffer<_, _>> = main_framebuffers.as_ref().unwrap()[image_num].clone();
        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), 1.0.into()];
        // TODO: Recreate only when screen dimensions change
        let dynamic_state = DynamicState {
            line_width: None,
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            scissors: None,
        };
        let command_buffer = model.draw_scene(
            InitializationDrawContext {
                draw_context: DrawContext {
                    device: device.clone(),
                    queue_family,
                    pipeline: main_pipeline.clone(),
                    dynamic: &dynamic_state,
                    main_descriptor_set: main_descriptor_set.clone(),
                    helper_resources: helper_resources.clone(),
                },
                framebuffer: current_framebuffer,
                clear_values,
            },
            0,
        ).unwrap();

        let result = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), screen_command_buffer).unwrap()
            .then_execute_same_queue(buffer_updates).unwrap()
            .then_signal_semaphore()
            .then_execute_same_queue(command_buffer).unwrap()
            .then_signal_fence()

            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        previous_frame_end = match result {
            Ok(future) => Box::new(future) as Box<GpuFuture>,
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>
            },
            Err(err) => panic!("{:?}", err),
        };

        if recreate_swapchain {
            continue;
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        //
        // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
        // wait would happen. Blocking may be the desired behavior, but if you don't want to
        // block you should spawn a separate thread dedicated to submissions.

        // Handling the window events in order to close the program when the user wants to close
        // it.
        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    },
                    ..
                } |
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => done = true,

                Event::WindowEvent {
                    // FIXME: "it should not be used to implement non-cursor-like interactions such as 3D camera control."
                    event: WindowEvent::CursorMoved {
                        position,
                        ..
                    },
                    ..
                } => {
                    cursor_position = position.into();
                    window.window().set_cursor_position(
                        (dimensions[0] as f64 / 2.0, dimensions[1] as f64 / 2.0).into()
                    ).expect("Could not center the cursor position.");
                },

                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state,
                            virtual_keycode: Some(virtual_code),
                            ..
                        },
                        ..
                    },
                    ..
                } => {
                    match state {
                        ElementState::Pressed => { pressed_keys.insert(virtual_code); }
                        ElementState::Released => { pressed_keys.remove(&virtual_code); }
                    }
                },

                Event::WindowEvent {
                    event: WindowEvent::MouseInput {
                        state,
                        button,
                        ..
                    },
                    ..
                } => {
                    match state {
                        ElementState::Pressed => { pressed_mouse_buttons.insert(button); }
                        ElementState::Released => { pressed_mouse_buttons.remove(&button); }
                    }
                }

                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => recreate_swapchain = true,

                _ => ()
            }
        });

        if done { return; }
    }
}
