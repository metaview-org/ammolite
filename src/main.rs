//! TODO:
//! * Precompute most of what `Model::draw_primitive` does
//! * Instancing
//! * VR rendering
//! * Animations
//! * Morph primitives

#![feature(core_intrinsics)]
#![feature(duration_float)]

#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
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
extern crate arrayvec;
extern crate boolinator;
extern crate mikktspace;
extern crate safe_transmute;

#[macro_use]
pub mod math;
pub mod iter;
pub mod shaders;
pub mod vertex;
pub mod sampler;
pub mod pipeline;
pub mod camera;
pub mod model;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::instance::RawInstanceExtensions;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, RawDeviceExtensions, DeviceExtensions, Queue, Features};
use vulkano::instance::{Instance, PhysicalDevice, QueueFamily};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageUsage};
use vulkano::image::swapchain::SwapchainImage;
use vulkano::framebuffer::{Framebuffer};
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::swapchain::{self, PresentMode, SurfaceTransform, Swapchain, AcquireError, SwapchainCreationError, Surface};
use vulkano_win::VkSurfaceBuild;
use winit::{ElementState, MouseButton, Event, DeviceEvent, WindowEvent, KeyboardInput, VirtualKeyCode, EventsLoop, WindowBuilder, Window};
use winit::dpi::PhysicalSize;
use crate::math::matrix::*;
use crate::math::vector::*;
use crate::model::Model;
use crate::model::DrawContext;
use crate::model::InitializationDrawContext;
use crate::model::HelperResources;
use crate::model::resource::UninitializedResource;
use crate::camera::*;
use crate::pipeline::GraphicsPipelineSetCache;

pub use crate::shaders::gltf_opaque_frag::ty::*;

fn swapchain_format_priority(format: &Format) -> u32 {
    match *format {
        Format::R8G8B8Srgb | Format::B8G8R8Srgb | Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb => 0,
        _ => 1,
    }
}

fn swapchain_format_compare(a: &Format, b: &Format) -> std::cmp::Ordering {
    swapchain_format_priority(a).cmp(&swapchain_format_priority(b))
}

fn vulkan_initialize<'a>(instance: &'a Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>, [u32; 2], Arc<Device>, QueueFamily<'a>, Arc<Queue>, Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    // TODO: Better device selection & SLI support
    let physical_device = PhysicalDevice::enumerate(instance).next().expect("No physical device available.");

    let events_loop = EventsLoop::new();
    let primary_monitor = events_loop.get_primary_monitor();

    // )
    let window = WindowBuilder::new()
        .with_title("ammolite")
        .with_dimensions(PhysicalSize::new(1280.0, 720.0).to_logical(primary_monitor.get_hidpi_factor()))
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
                    &Features {
                        independent_blend: true,
                        .. Features::none()
                    },
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

        // Order supported swapchain formats by priority and choose the most preferred one.
        // The swapchain format must be in SRGB space.
        let mut supported_formats: Vec<Format> = capabilities.supported_formats.iter()
            .map(|(current_format, _)| *current_format)
            .collect();
        supported_formats.sort_by(swapchain_format_compare);
        let format = supported_formats[0];

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
    mat4!([1.0 / dimensions[0],                 0.0,               0.0,                0.0,
                           0.0, 1.0 / dimensions[1],               0.0,                0.0,
                           0.0,                 0.0, 1.0 / (z_f - z_n), -z_n / (z_f - z_n),
                           0.0,                 0.0,               0.0,                1.0])
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
    // Even though we could negate the Y coordinate so as to adjust the vectors to the Vulkan
    // coordinate system, which has the Y axis pointing downwards, contrary to OpenGL, we need to
    // apply the same transformation to other vertex attributes such as normal and tangent vectors,
    // but those are not projected.
    let f = 1.0 / (fov_rad / 2.0).tan();

    // We derive the coefficients for the Z coordinate from the following equation:
    // `f(z) = A*z + B`, because we know we need to translate and scale the Z coordinate.
    // The equation changes to the following, after the W-division:
    // `f(z) = A + B/z`
    // And must satisfy the following conditions:
    // `f(z_near) = 0`
    // `f(z_far) = 1`
    // Solving for A and B gives us the necessary coefficients to construct the matrix.
    // mat4!([f / aspect_ratio, 0.0,                0.0,                       0.0,
    //                     0.0,  -f,                0.0,                       0.0,
    //                     0.0, 0.0, -z_f / (z_n - z_f), (z_n * z_f) / (z_n - z_f),
    //                     0.0, 0.0,                1.0,                       0.0])

    // TODO: Investigate the mysterious requirement of flipping the X coordinate
    mat4!([-f / aspect_ratio, 0.0,                0.0,                       0.0,
                         0.0,   f,                0.0,                       0.0,
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

    // let screen_vertices = [
    //     ScreenVertex { position: [-1.0, -1.0,  0.0] },
    //     ScreenVertex { position: [-1.0,  1.0,  0.0] },
    //     ScreenVertex { position: [ 1.0,  1.0,  0.0] },
    //     ScreenVertex { position: [ 1.0, -1.0,  0.0] },
    // ];

    // let screen_indices = [
    //     0, 1, 2u16,
    //     // 2, 3, 0u16,
    // ];

    // let (
    //     (screen_vertex_staging_buffer, screen_vertex_device_buffer),
    //     (screen_index_staging_buffer, screen_index_device_buffer),
    // ) = create_vertex_index_buffers(
    //     &device,
    //     queue_family,
    //     screen_vertices.into_iter().cloned(),
    //     screen_indices.into_iter().cloned(),
    // );

    let pipeline_cache = Arc::new(GraphicsPipelineSetCache::create(&device, &swapchain));

    let mut main_ubo = SceneUBO::new(
        0.0,
        [dimensions[0] as f32, dimensions[1] as f32].into(),
        [0.0, 0.0, 0.0].into(),
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

    // let screen_image = AttachmentImage::with_usage(
    //     device.clone(),
    //     SCREEN_DIMENSIONS.clone(),
    //     swapchain.format(),
    //     ImageUsage {
    //         sampled: true,
    //         .. ImageUsage::none()
    //     }
    // ).unwrap();
    // let border_color = match swapchain.format().ty() {
    //     FormatTy::Uint | FormatTy::Sint => BorderColor::IntTransparentBlack,
    //                                   _ => BorderColor::FloatTransparentBlack,
    // };
    // let screen_sampler = Sampler::new(
    //     device.clone(),
    //     Filter::Nearest,  // magnifying filter
    //     Filter::Linear,  // minifying filter
    //     MipmapMode::Nearest,
    //     SamplerAddressMode::ClampToBorder(border_color),
    //     SamplerAddressMode::ClampToBorder(border_color),
    //     SamplerAddressMode::ClampToBorder(border_color),
    //     0.0,  // mip_lod_bias
    //     // TODO: Turn anisotropic filtering on for better screen readability
    //     1.0,  // anisotropic filtering (1.0 = off, anything higher = on)
    //     1.0,  // min_lod
    //     1.0,  // max_lod
    // ).unwrap();
    let main_descriptor_set_gltf_opaque = Arc::new(
        // FIXME: Use a layout instead
        PersistentDescriptorSet::start(pipeline_cache.get_default_pipeline().unwrap().opaque.clone(), 0)
            .add_buffer(main_ubo_device_buffer.clone()).unwrap()
            .build().unwrap()
    );
    // let main_descriptor_set_gltf_mask = Arc::new(
    //     PersistentDescriptorSet::start(pipeline_gltf_mask.clone(), 0)
    //         .add_buffer(main_ubo_device_buffer.clone()).unwrap()
    //         .build().unwrap()
    // );

    // let screen_framebuffers: Vec<Arc<Framebuffer<_, _>>> = images.iter().map(|_| {
    //     Arc::new(
    //         Framebuffer::start(screen_pipeline.render_pass().clone())
    //         .add(screen_image.clone()).unwrap()
    //         .build().unwrap()
    //     )
    // }).collect();
    let mut main_framebuffers: Option<Vec<Arc<Framebuffer<_, _>>>> = None;
    let mut descriptor_set_gltf_blend: Option<Arc<DescriptorSet + Send + Sync>> = None;

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
    let (init_command_buffer_builder, helper_resources) = HelperResources::new(
        &device,
        [queue_family].into_iter().cloned(),
        // FIXME: Replace with a pipeline layout
        pipeline_cache.get_default_pipeline().unwrap().opaque.clone(),
    ).unwrap().initialize_resource(
        &device,
        queue_family.clone(),
        init_command_buffer_builder,
    ).unwrap();
    let (init_command_buffer_builder, model) = {
        let model_path = std::env::args().nth(1).unwrap_or_else(|| {
            eprintln!("No model path provided.");
            std::process::exit(1);
        });
        Model::import(
            &device,
            [queue_family].into_iter().cloned(),
            // FIXME: Replace with a pipeline layout
            pipeline_cache.get_default_pipeline().unwrap().opaque.clone(),
            &helper_resources,
            model_path,
        ).unwrap().initialize_resource(
            &device,
            queue_family.clone(),
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
    let init_instant = Instant::now();
    let mut previous_frame_instant = init_instant.clone();
    let mut mouse_delta: (f64, f64) = (0.0, 0.0);
    let mut camera = PitchYawCamera3::new();
    let mut pressed_keys: HashSet<VirtualKeyCode> = HashSet::new();
    let mut pressed_mouse_buttons: HashSet<MouseButton> = HashSet::new();
    let mut cursor_capture = true;

    loop {
        let now = Instant::now();
        let delta_time = now.duration_since(previous_frame_instant);
        previous_frame_instant = now;

        camera.update(&delta_time, &mouse_delta, &pressed_keys, &pressed_mouse_buttons);
        mouse_delta = (0.0, 0.0);

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            // println!("Recreating the swapchain.");

            dimensions = {
                let dpi = window.window().get_hidpi_factor();
                let (width, height) = window.window().get_inner_size().unwrap().to_physical(dpi)
                    .into();
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
            let depth_image = Some(AttachmentImage::with_usage(
                device.clone(),
                dimensions.clone(),
                Format::D32Sfloat,
                ImageUsage {
                    depth_stencil_attachment: true,
                    .. ImageUsage::none()
                }
            ).unwrap());
            let blend_accumulation_image = Some(AttachmentImage::with_usage(
                device.clone(),
                dimensions.clone(),
                Format::R32G32B32A32Sfloat,
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                }
            ).unwrap());
            let blend_revealage_image = Some(AttachmentImage::with_usage(
                device.clone(),
                dimensions.clone(),
                Format::R32G32B32A32Sfloat, //FIXME
                ImageUsage {
                    color_attachment: true,
                    input_attachment: true,
                    transient_attachment: true,
                    .. ImageUsage::none()
                }
            ).unwrap());
            descriptor_set_gltf_blend = Some(Arc::new(
                // FIXME: Use a layout instead
                PersistentDescriptorSet::start(pipeline_cache.get_default_pipeline().unwrap().blend_finalize.clone(), 3)
                    .add_image(blend_accumulation_image.as_ref().unwrap().clone()).unwrap()
                    .add_image(blend_revealage_image.as_ref().unwrap().clone()).unwrap()
                    .build().unwrap()
            ));
            main_framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(pipeline_cache.render_pass.clone())
                         .add(image.clone()).unwrap()
                         .add(depth_image.as_ref().unwrap().clone()).unwrap()
                         .add(blend_accumulation_image.as_ref().unwrap().clone()).unwrap()
                         .add(blend_revealage_image.as_ref().unwrap().clone()).unwrap()
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

        let time_elapsed = init_instant.elapsed().as_float_secs() as f32;

        // println!("Camera position: {:?}", camera.get_position());
        // println!("Camera direction: {:?}", camera.get_direction());
        main_ubo = SceneUBO::new(
            time_elapsed,
            Vec2([dimensions[0] as f32, dimensions[1] as f32]),
            camera.get_position(),
            construct_model_matrix(1.0,
                                   &[1.0, 0.0, 2.0].into(),
                                   &[time_elapsed.sin() * 0.0 * 1.0, time_elapsed.cos() * 0.0 * 3.0 / 2.0, 0.0].into()),
            camera.get_view_matrix(),
            // construct_view_matrix(&[(seconds_elapsed as f32 * 0.5).cos(), 0.0, 0.0].into(),
            //                       &[0.0, 0.0, 0.0].into()),
            construct_perspective_projection_matrix(0.001, 1000.0, dimensions[0] as f32 / dimensions[1] as f32, std::f32::consts::FRAC_PI_2),
            // construct_orthographic_projection_matrix(0.1, 1000.0, [dimensions[0] as f32 / dimensions[1] as f32, 1.0].into()),
        );

//         let screen_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
//             .copy_buffer(screen_vertex_staging_buffer.clone(), screen_vertex_device_buffer.clone()).unwrap()
//             .copy_buffer(screen_index_staging_buffer.clone(), screen_index_device_buffer.clone()).unwrap()
//             .begin_render_pass(screen_framebuffers[image_num].clone(),
//                                false,
//                                vec![[0.0, 1.0, 0.0, 1.0].into()]).unwrap()
//             .draw_indexed(screen_pipeline.clone(),
//                   &DynamicState::none(),
//                   screen_vertex_device_buffer.clone(),
//                   screen_index_device_buffer.clone(),
//                   (), ()).unwrap()
//             .end_render_pass().unwrap()
//             .build().unwrap();

        let buffer_updates = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
           .update_buffer(main_ubo_staging_buffer.clone(), main_ubo.clone()).unwrap()
           .copy_buffer(main_ubo_staging_buffer.clone(), main_ubo_device_buffer.clone()).unwrap()
           .build().unwrap();

        let current_framebuffer: Arc<Framebuffer<_, _>> = main_framebuffers.as_ref().unwrap()[image_num].clone();
        let clear_values = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            1.0.into(),
            // ClearValue::None,
            // ClearValue::None,
            [0.0, 0.0, 0.0, 0.0].into(),
            [1.0, 1.0, 1.0, 1.0].into(),
        ];
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
                    pipeline_cache: pipeline_cache.clone(),
                    dynamic: &dynamic_state,
                    main_descriptor_set: main_descriptor_set_gltf_opaque.clone(),
                    descriptor_set_blend: descriptor_set_gltf_blend.as_ref().unwrap().clone(),
                    helper_resources: helper_resources.clone(),
                },
                framebuffer: current_framebuffer,
                clear_values,
            },
            0,
        ).unwrap();

        let result = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), buffer_updates).unwrap()
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
                    event: WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state: ElementState::Released,
                            virtual_keycode: Some(VirtualKeyCode::LAlt),
                            ..
                        },
                        ..
                    },
                    ..
                } => cursor_capture ^= true,

                Event::DeviceEvent {
                    event: DeviceEvent::Motion { axis, value },
                    ..
                } if cursor_capture => {
                    match axis {
                        0 => mouse_delta.0 += value,
                        1 => mouse_delta.1 += value,
                        _ => (),
                    }
                },

                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { .. },
                    ..
                } if cursor_capture => {
                    window.window().set_cursor_position(
                        (dimensions[0] as f64 / 2.0, dimensions[1] as f64 / 2.0).into()
                    ).expect("Could not center the cursor position.");
                }

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
