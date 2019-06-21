//! TODO:
//! * Use secondary command buffers to parallelize their creation
//! * Mip Mapping
//! * VR rendering
//! * Instancing
//! * Animations
//! * Morph primitives

#![feature(core_intrinsics)]

#[macro_use]
pub mod math;
pub mod buffer;
pub mod camera;
pub mod iter;
pub mod model;
pub mod pipeline;
pub mod sampler;
pub mod shaders;
pub mod swapchain;
pub mod vertex;

use std::path::Path;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::cell::RefCell;
use std::ffi::{self, CString};
use core::num::NonZeroU32;
use openxr::Instance as XrInstance;
use vulkano;
use vulkano::VulkanObject;
use vulkano::swapchain::ColorSpace;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::instance::RawInstanceExtensions;
use vulkano::buffer::{TypedBufferAccess};
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, RawDeviceExtensions, DeviceExtensions, Queue, Features};
use vulkano::instance::{self, ApplicationInfo, Instance as VkInstance, PhysicalDevice, QueueFamily};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::format::{Format, ClearValue};
use vulkano::image::SwapchainImage;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{PresentMode, SurfaceTransform, Swapchain, AcquireError, SwapchainCreationError, Surface};
use vulkano_win::VkSurfaceBuild;
use winit::{ElementState, MouseButton, Event, DeviceEvent, WindowEvent, KeyboardInput, VirtualKeyCode, EventsLoop, WindowBuilder, Window};
use winit::dpi::PhysicalSize;
use openxr::Entry;
use crate::math::matrix::*;
use crate::math::vector::*;
use crate::model::FramebufferWithClearValues;
use crate::model::Model;
use crate::model::DrawContext;
use crate::model::InstanceDrawContext;
use crate::model::HelperResources;
use crate::model::resource::UninitializedResource;
use crate::camera::*;
use crate::pipeline::GltfGraphicsPipeline;
use crate::pipeline::GraphicsPipelineSetCache;
use crate::pipeline::DescriptorSetMap;

pub use crate::shaders::gltf_opaque_frag::ty::*;

pub type XrVkSession = openxr::Session<openxr::Vulkan>;
pub type XrVkFrameStream = openxr::FrameStream<openxr::Vulkan>;

fn swapchain_format_priority(format: &Format) -> u32 {
    match *format {
        Format::R8G8B8Srgb | Format::B8G8R8Srgb | Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb => 0,
        _ => 1,
    }
}

fn swapchain_format_compare(a: &Format, b: &Format) -> std::cmp::Ordering {
    swapchain_format_priority(a).cmp(&swapchain_format_priority(b))
}

// fn vulkan_find_supported_format(physical_device: &Arc<PhysicalDevice>, candidates: &[Format]) -> Option<Format> {
//     // TODO: Querying available formats is not implemented (exposed) in vulkano.
//     unimplemented!()
// }

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

pub struct WorldSpaceModel<'a> {
    pub model: &'a Model,
    pub matrix: Mat4,
}

pub struct Ammolite {
    pub vk_instance: Arc<VkInstance>,
    pub xr_instance: Arc<XrInstance>,
    pub xr_session: Arc<XrVkSession>,
    pub xr_frame_stream: XrVkFrameStream,
    pub xr_reference_space_stage: openxr::Space,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pipeline_cache: GraphicsPipelineSetCache,
    pub helper_resources: HelperResources,
    pub window: Arc<Surface<Window>>,
    pub window_events_loop: Rc<RefCell<EventsLoop>>,
    pub window_dimensions: [NonZeroU32; 2],
    pub window_swapchain: Arc<Swapchain<Window>>,
    pub window_swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    pub window_swapchain_framebuffers: Option<Vec<Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>>>,
    pub window_swapchain_recreate: bool,
    pub synchronization: Option<Box<dyn GpuFuture>>,
    // TODO Consider moving to SharedGltfGraphicsPipelineResources
    pub buffer_pool_uniform_instance: CpuBufferPool<InstanceUBO>,
    pub camera_position: Vec3,
    pub camera_view_matrix: Mat4,
    pub camera_projection_matrix: Mat4,
}

impl Ammolite {
    fn setup_openxr_instance() -> XrInstance {
        let entry = openxr::Entry::linked();
        let available_extensions = entry.enumerate_extensions().unwrap(); // FIXME: unwrap

        // TODO: Add support for the following extensions:
        // * XR_KHR_visibility_mask
        // * XR_EXT_debug_utils
        //
        // Low priority:
        // * XR_KHR_vulkan_swapchain_format_list
        // * XR_EXT_performance_settings
        // * XR_EXT_thermal_query

        println!("Supported OpenXR extensions: {:#?}", available_extensions);

        if !available_extensions.khr_vulkan_enable {
            panic!("The OpenXR runtime does not support Vulkan.");
        }

        let used_extensions = openxr::ExtensionSet {
            khr_vulkan_enable: true,
            khr_visibility_mask: available_extensions.khr_visibility_mask,
            ext_debug_utils: available_extensions.ext_debug_utils,
            ..Default::default()
        };

        struct OpenXRVersion((u16, u16, u16)); // (major, minor, patch)

        impl OpenXRVersion {
            fn from_crate() -> Self {
                Self((
                    env!("CARGO_PKG_VERSION_MAJOR").parse()
                        .expect("Invalid crate major version, must be u16."),
                    env!("CARGO_PKG_VERSION_MINOR").parse()
                        .expect("Invalid crate minor version, must be u16."),
                    env!("CARGO_PKG_VERSION_PATCH").parse()
                        .expect("Invalid crate patch version, must be u16."),
                ))
            }
        }

        impl Into<u32> for OpenXRVersion {
            fn into(self) -> u32 {
                  ((((self.0).0 & 0x3FF) as u32) << 22)
                | ((((self.0).1 & 0x3FF) as u32) << 12)
                | ((((self.0).2 & 0xFFF) as u32) <<  0)
            }
        }

        let app_info = openxr::ApplicationInfo::new()
            .api_version(OpenXRVersion((0, 90, 0)).into())
            .engine_name(env!("CARGO_PKG_NAME"))
            .engine_version(OpenXRVersion::from_crate().into())
            .application_name(env!("CARGO_PKG_NAME")) // TODO: Make customizable
            .application_version(OpenXRVersion::from_crate().into()); // TODO: Make customizable

        entry.create_instance(app_info, &used_extensions).unwrap()
    }

    fn vulkan_initialize<'a>(vk_instance: &'a Arc<VkInstance>, xr_instance: &'a Arc<XrInstance>, xr_system: openxr::SystemId) -> (EventsLoop, Arc<Surface<Window>>, [NonZeroU32; 2], Arc<Device>, QueueFamily<'a>, Arc<Queue>, Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let openxr::vulkan::Requirements {
            min_api_version_supported: min_api_version,
            max_api_version_supported: max_api_version,
        } = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system).unwrap();
        let min_api_version = min_api_version.into_raw();
        let max_api_version = max_api_version.into_raw();
        // TODO: Better device selection & SLI support
        let physical_device = PhysicalDevice::enumerate(vk_instance)
            .filter(|physical_device| {
                let api_version = physical_device.api_version().into_vulkan_version();

                api_version >= min_api_version && api_version <= max_api_version
            })
            .next().expect("No physical device available.");

        let events_loop = EventsLoop::new();
        let primary_monitor = events_loop.get_primary_monitor();

        // )
        let window = WindowBuilder::new()
            .with_title("ammolite")
            .with_dimensions(PhysicalSize::new(1280.0, 720.0).to_logical(primary_monitor.get_hidpi_factor()))
            // If this doesn't compile, you are probably using a conflicting version of winit
            .build_vk_surface(&events_loop, vk_instance.clone()).unwrap();

        window.window().hide_cursor(true);

        let mut dimensions: [NonZeroU32; 2] = {
            let (width, height) = window.window().get_inner_size().unwrap().into();
            [
                NonZeroU32::new(width).expect("The width of the window must not be 0."),
                NonZeroU32::new(height).expect("The height of the window must not be 0."),
            ]
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
            let xr_extensions: Vec<_> = xr_instance.vulkan_device_extensions(xr_system)
                .unwrap().split_ascii_whitespace()
                .map(|str_slice| CString::new(str_slice).unwrap()).collect();
            let raw_device_extensions = [/*CString::new("VK_EXT_debug_utils").unwrap()*/];
            let device_extensions = RawDeviceExtensions::new(raw_device_extensions.into_iter().cloned())
                .union(&(&safe_device_extensions).into())
                .union(&RawDeviceExtensions::new(xr_extensions.into_iter()));

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
            dimensions = capabilities.current_extent.map(|extent| [
                NonZeroU32::new(extent[0]).unwrap(),
                NonZeroU32::new(extent[1]).unwrap(),
            ]).unwrap_or(dimensions);

            // Order supported swapchain formats by priority and choose the most preferred one.
            // The swapchain format must be in SRGB space.
            let mut supported_formats: Vec<Format> = capabilities.supported_formats.iter()
                .map(|(current_format, _)| *current_format)
                .collect();
            supported_formats.sort_by(swapchain_format_compare);
            let format = supported_formats[0];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new::<_, &Arc<Queue>>(
                device.clone(),
                window.clone(),
                (&queue).into(),
                dimensions,
                NonZeroU32::new(1).unwrap(),
                NonZeroU32::new(capabilities.min_image_count).expect("Invalid swapchaing image count."),
                format,
                ColorSpace::SrgbNonLinear,
                capabilities.supported_usage_flags,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Immediate, /* PresentMode::Relaxed TODO: Getting only ~60 FPS in a window */
                true,
                None,
            ).expect("failed to create swapchain")
        };

        (events_loop, window, dimensions, device, queue_family, queue, swapchain, images)
    }

    fn setup_openxr_session(
        xr_instance: &Arc<XrInstance>,
        vk_instance: &Arc<VkInstance>,
        xr_system: openxr::SystemId,
        vk_device: &Arc<Device>,
        vk_queue: &Arc<Queue>,
    ) -> (XrVkSession, XrVkFrameStream) {
        let create_info = openxr::vulkan::SessionCreateInfo {
            instance: vk_instance.internal_object() as *const ffi::c_void,
            physical_device: vk_device.physical_device().internal_object() as *const ffi::c_void,
            device: vk_device.internal_object() as *const ffi::c_void,
            queue_family_index: vk_queue.family().id(),
            queue_index: vk_queue.id_within_family(),
        };

        unsafe {
            xr_instance.create_session::<openxr::Vulkan>(xr_system, &create_info)
        }.unwrap()
    }

    pub fn new() -> Self {
        // OpenXR
        let xr_instance = Arc::new(Self::setup_openxr_instance());
        let xr_system = xr_instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        // Instance
        let app_info = {
            let version = vulkano::instance::Version {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
            };
            let name = env!("CARGO_PKG_NAME");

            ApplicationInfo {
                application_name: Some(name.into()), // TODO: Make customizable
                application_version: Some(version.into()), // TODO: Make customizable
                engine_name: Some(name.into()),
                engine_version: Some(version.into()),
            }
        };
        let win_extensions = vulkano_win::required_extensions();
        let xr_extensions: Vec<_> = xr_instance.vulkan_instance_extensions(xr_system)
            .unwrap().split_ascii_whitespace()
            .map(|str_slice| CString::new(str_slice).unwrap()).collect();
        let raw_extensions = [/*CString::new("VK_EXT_debug_marker").unwrap()*/];
        let extensions = RawInstanceExtensions::new(raw_extensions.into_iter().cloned())
            .union(&(&win_extensions).into())
            .union(&RawInstanceExtensions::new(xr_extensions.into_iter()));
        let layers = [];
        let vk_instance = VkInstance::new(Some(&app_info), extensions, layers.into_iter().cloned())
            .expect("Failed to create a Vulkan instance.");

        // (EventsLoop, Arc<Surface<Window>>, [u32; 2], Arc<Device>, QueueFamily<'a>, Arc<Queue>, Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>)
        let (events_loop, window, dimensions, device, queue_family, queue, swapchain, images)
            = Self::vulkan_initialize(&vk_instance, &xr_instance, xr_system);
        let (xr_session, xr_frame_stream) = Self::setup_openxr_session(&xr_instance, &vk_instance, xr_system, &device, &queue);

        let xr_reference_space_stage = xr_session.create_reference_space(
            openxr::ReferenceSpaceType::STAGE,
            openxr::Posef {
                orientation: openxr::Quaternionf { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                position: openxr::Vector3f { x: 0.0, y: 0.0, z: 0.0 },
            },
        ).unwrap();

        let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
        let (init_command_buffer_builder, helper_resources) = HelperResources::new(
            &device,
            [queue_family].into_iter().cloned(),
        ).unwrap()
            .initialize_resource(&device, queue_family, init_command_buffer_builder).unwrap();
        let (init_command_buffer_builder, pipeline_cache) = GraphicsPipelineSetCache::create(device.clone(), swapchain.clone(), helper_resources.clone(), queue_family)
            .initialize_resource(&device, queue_family, init_command_buffer_builder).unwrap();
        let init_command_buffer = init_command_buffer_builder.build().unwrap();

        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
        // that, we store the submission of the previous frame here.
        let synchronization: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), init_command_buffer).unwrap()
            // .then_signal_fence()
            // .then_execute_same_queue(init_unsafe_command_buffer).unwrap()
            .then_signal_fence_and_flush().unwrap());

        Self {
            vk_instance,
            xr_instance,
            xr_session: Arc::new(xr_session),
            xr_frame_stream,
            xr_reference_space_stage,
            device: device.clone(),
            queue,
            pipeline_cache,
            helper_resources,
            window,
            window_events_loop: Rc::new(RefCell::new(events_loop)),
            window_dimensions: dimensions,
            window_swapchain: swapchain,
            window_swapchain_images: images,
            window_swapchain_framebuffers: None,
            window_swapchain_recreate: false,
            synchronization: Some(synchronization),
            buffer_pool_uniform_instance: CpuBufferPool::uniform_buffer(device),
            camera_position: Vec3::zero(),
            camera_view_matrix: Mat4::identity(),
            camera_projection_matrix: Mat4::identity(),
        }
    }

    pub fn load_model<S: AsRef<Path>>(&mut self, path: S) -> Model {
        let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family()).unwrap();
        let (init_command_buffer_builder, model) = {
            Model::import(
                &self.device,
                [self.queue.family()].into_iter().cloned(),
                &self.pipeline_cache,
                &self.helper_resources,
                path,
            ).unwrap().initialize_resource(
                &self.device,
                self.queue.family().clone(),
                init_command_buffer_builder
            ).unwrap()
        };
        let init_command_buffer = init_command_buffer_builder.build().unwrap();

        self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
            .then_execute(self.queue.clone(), init_command_buffer).unwrap()
            // .then_signal_fence()
            // .then_execute_same_queue(init_unsafe_command_buffer).unwrap()
            .then_signal_fence_and_flush().unwrap()));

        model
    }

    pub fn render<'a>(&mut self, elapsed: &Duration, model_provider: impl FnOnce() -> &'a [WorldSpaceModel<'a>]) {
        let world_space_models = model_provider();
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        self.synchronization.as_mut().unwrap().cleanup_finished();

        let state = self.xr_frame_stream.wait().unwrap();
        let (view_flags, views) = self.xr_session
            .locate_views(state.predicted_display_time, &self.xr_reference_space_stage)
            .unwrap();
        let status = self.xr_frame_stream.begin().unwrap();

        // A loop is used as a way to reset the rendering using `continue` if something goes wrong,
        // there is a `break` statement at the end.
        loop {
            if self.window_swapchain_recreate {
                self.window_dimensions = {
                    let dpi = self.window.window().get_hidpi_factor();
                    let (width, height): (u32, u32) = self.window.window().get_inner_size().unwrap().to_physical(dpi)
                        .into();
                    [
                        NonZeroU32::new(width).expect("The width of the window must not be 0."),
                        NonZeroU32::new(height).expect("The height of the window must not be 0."),
                    ]
                };

                let (new_swapchain, new_images) = match self.window_swapchain.recreate_with_dimension(self.window_dimensions) {
                    Ok(r) => r,
                    // This error tends to happen when the user is manually resizing the window.
                    // Simply restarting the loop is the easiest way to fix this issue.
                    Err(SwapchainCreationError::UnsupportedDimensions) => {
                        continue;
                    },
                    Err(err) => panic!("{:?}", err)
                };

                self.window_swapchain = new_swapchain;
                self.window_swapchain_images = new_images;
                self.window_swapchain_framebuffers = None;
                self.window_swapchain_recreate = false;
            }

            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            if self.window_swapchain_framebuffers.is_none() {
                self.pipeline_cache.shared_resources.reconstruct_dimensions_dependent_images(
                    self.window_dimensions.clone()
                ).expect("Could not reconstruct dimension dependent resources.");

                self.window_swapchain_framebuffers = Some(
                    self.pipeline_cache.shared_resources.construct_swapchain_framebuffers(
                        self.pipeline_cache.render_pass.clone(),
                        &self.window_swapchain_images,
                    )
                );

                for (_, pipeline) in self.pipeline_cache.pipeline_map.write().unwrap().iter_mut() {
                    let per_pipeline = |pipeline: &mut GltfGraphicsPipeline| {
                        pipeline.layout_dependent_resources
                            .reconstruct_descriptor_sets(&self.pipeline_cache.shared_resources);
                    };

                    (per_pipeline)(&mut pipeline.opaque);
                    (per_pipeline)(&mut pipeline.mask);
                    (per_pipeline)(&mut pipeline.blend_preprocess);
                    (per_pipeline)(&mut pipeline.blend_finalize);
                }
            }

            // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
            // no image is available (which happens if you submit draw commands too quickly), then the
            // function will block.
            // This operation returns the index of the image that we are allowed to draw upon.
            //
            // This function can block if no image is available. The parameter is an optional timeout
            // after which the function call will return an error.
            let (image_num, acquire_future) = match vulkano::swapchain::acquire_next_image(self.window_swapchain.clone(),
                                                                                  None) {
                Ok((image_num, acquire_future)) => {
                    (image_num, acquire_future)
                },
                Err(AcquireError::OutOfDate) => {
                    self.window_swapchain_recreate = true;
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            let secs_elapsed = ((elapsed.as_secs() as f64) + (elapsed.as_nanos() as f64) / (1_000_000_000f64)) as f32;

            let scene_ubo = SceneUBO::new(
                secs_elapsed,
                Vec2([self.window_dimensions[0].get() as f32, self.window_dimensions[1].get() as f32]),
                self.camera_position.clone(),
                self.camera_view_matrix.clone(),
                self.camera_projection_matrix.clone(),
            );

            let buffer_updates = AutoCommandBufferBuilder::primary_one_time_submit(
                self.device.clone(),
                self.queue.family(),
            ).unwrap()
                .update_buffer(
                    self.pipeline_cache.shared_resources.scene_ubo_buffer.staging_buffer().clone(),
                    scene_ubo.clone()
                ).unwrap()
                .copy_buffer(
                    self.pipeline_cache.shared_resources.scene_ubo_buffer.staging_buffer().clone(),
                    self.pipeline_cache.shared_resources.scene_ubo_buffer.device_buffer().clone()
                ).unwrap()
                .build().unwrap();


            // TODO don't Box the future, pass it as `impl GpuFuture` to render_instances,
            // which should then return an `impl GpuFuture`
            self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
                .then_execute(self.queue.clone(), buffer_updates).unwrap()
                .join(acquire_future)));

            let current_framebuffer: Arc<dyn FramebufferWithClearValues<_>> = self.window_swapchain_framebuffers
                .as_ref().unwrap()[image_num].clone();

            self.render_instances(current_framebuffer, world_space_models);


            let result = self.synchronization.take().unwrap()
                .then_signal_fence()
                // The color output is now expected to contain our triangle. But in order to show it on
                // the screen, we have to *present* the image by calling `present`.
                // This function does not actually present the image immediately. Instead it submits a
                // present command at the end of the queue. This means that it will only be presented once
                // the GPU has finished executing the command buffer that draws the triangle.
                .then_swapchain_present(self.queue.clone(), self.window_swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            self.synchronization = Some(match result {
                Ok(future) => Box::new(future) as Box<GpuFuture>,
                Err(FlushError::OutOfDate) => {
                    self.window_swapchain_recreate = true;
                    Box::new(vulkano::sync::now(self.device.clone())) as Box<GpuFuture>
                },
                Err(err) => panic!("{:?}", err),
            });

            if self.window_swapchain_recreate {
                continue;
            }

            // Note that in more complex programs it is likely that one of `acquire_next_image`,
            // `command_buffer::submit`, or `present` will block for some time. This happens when the
            // GPU's queue is full and the driver has to wait until the GPU finished some work.
            //
            // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
            // wait would happen. Blocking may be the desired behavior, but if you don't want to
            // block you should spawn a separate thread dedicated to submissions.
            break;
        }
    }

    fn render_instances<'a>(&mut self,
                            current_framebuffer: Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>,
                            // clear_values: Vec<ClearValue>,
                            // context: &'a DrawContext<'a>,
                            world_space_models: &'a [WorldSpaceModel<'a>]) {
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
                dimensions: [self.window_dimensions[0].get() as f32, self.window_dimensions[1].get() as f32],
                depth_range: 0.0 .. 1.0,
            }]),
            scissors: None,
        };

        let draw_context = DrawContext {
            device: &self.device.clone(),
            queue_family: &self.queue.family(),
            pipeline_cache: &self.pipeline_cache,
            dynamic: &dynamic_state,
            helper_resources: &self.helper_resources,
        };

        let instances = world_space_models.iter()
            .map(|WorldSpaceModel { model, matrix }| {
                let instance_ubo = InstanceUBO::new(matrix.clone());
                let instance_buffer: Arc<dyn TypedBufferAccess<Content=InstanceUBO> + Send + Sync>
                    = Arc::new(self.buffer_pool_uniform_instance.next(instance_ubo).unwrap());
                let used_layouts = model.get_used_pipelines_layouts(&self.pipeline_cache);
                let descriptor_set_map = DescriptorSetMap::new(
                    &used_layouts[..],
                    |layout_dependent_resources| {
                        layout_dependent_resources.descriptor_set_pool_instance.clone()
                    },
                    {
                        let instance_buffer_ref = &instance_buffer;
                        move |set_builder| {
                            Arc::new(set_builder.add_buffer(instance_buffer_ref.clone()).unwrap()
                                     .build().unwrap())
                        }
                    },
                );

                (
                    model,
                    descriptor_set_map,
                )
            })
            .collect::<Vec<_>>();

        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        ).unwrap()
            .begin_render_pass(
                current_framebuffer,
                false,
                clear_values,
            ).unwrap();

        for (subpass_index, alpha_mode) in Model::get_subpass_alpha_modes().enumerate() {
            if subpass_index > 0 {
                command_buffer = command_buffer.next_subpass(false).unwrap();
            }

            for (_index, world_space_model) in instances.iter().enumerate() {
                let &(ref model, ref descriptor_set_map_instance) = world_space_model;
                let instance_context = InstanceDrawContext {
                    draw_context: &draw_context,
                    descriptor_set_map_instance: &descriptor_set_map_instance,
                };

                command_buffer = model.draw_scene(
                    command_buffer,
                    instance_context,
                    alpha_mode,
                    subpass_index as u8,
                    0,
                ).unwrap();
            }
        }

        let command_buffer = command_buffer.end_render_pass().unwrap();

        self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
            .then_signal_semaphore()
            .then_execute_same_queue(command_buffer.build().unwrap()).unwrap()));
    }
}

fn main() {
    let mut ammolite = Ammolite::new();

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("No model path provided.");
        std::process::exit(1);
    });

    let model = ammolite.load_model(model_path);

    // Timing and camera controls are not handled by the renderer
    let mut mouse_delta: (f64, f64) = (0.0, 0.0);
    let mut camera = PitchYawCamera3::new();
    let mut pressed_keys: HashSet<VirtualKeyCode> = HashSet::new();
    let mut pressed_mouse_buttons: HashSet<MouseButton> = HashSet::new();
    let mut cursor_capture = true;

    let init_instant = Instant::now();
    let mut previous_frame_instant = init_instant.clone();
    let mut quit = false;

    while !quit {
        let now = Instant::now();
        let elapsed = now.duration_since(init_instant);
        let delta_time = now.duration_since(previous_frame_instant);
        previous_frame_instant = now;

        camera.update(&delta_time, &mouse_delta, &pressed_keys, &pressed_mouse_buttons);
        mouse_delta = (0.0, 0.0);

        ammolite.camera_position = camera.get_position();
        ammolite.camera_view_matrix = camera.get_view_matrix();
        ammolite.camera_projection_matrix = construct_perspective_projection_matrix(
            0.001,
            1000.0,
            ammolite.window_dimensions[0].get() as f32 / ammolite.window_dimensions[1].get() as f32,
            std::f32::consts::FRAC_PI_2,
        );
        // construct_orthographic_projection_matrix(0.1, 1000.0, [ammolite.window_dimensions[0] as f32 / ammolite.window_dimensions[1] as f32, 1.0].into()),

        let secs_elapsed = ((elapsed.as_secs() as f64) + (elapsed.as_nanos() as f64) / (1_000_000_000f64)) as f32;
        let model_matrices = [
            construct_model_matrix(1.0,
                                   &[1.0, 0.0, 2.0].into(),
                                   &[secs_elapsed.sin() * 0.0 * 1.0, secs_elapsed.cos() * 0.0 * 3.0 / 2.0, 0.0].into()),
            construct_model_matrix(1.0,
                                   &[1.0, 1.0, 2.0].into(),
                                   &[secs_elapsed.sin() * 0.0 * 1.0, secs_elapsed.cos() * 0.0 * 3.0 / 2.0, 0.0].into()),
        ];

        let world_space_models = [
            WorldSpaceModel { model: &model, matrix: model_matrices[0].clone() },
            // WorldSpaceModel { model: &model, matrix: model_matrices[1].clone() },
        ];

        ammolite.render(&elapsed, || &world_space_models[..]);

        ammolite.window_events_loop.clone().as_ref().borrow_mut().poll_events(|ev| {
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
                } => quit = true,

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
                    ammolite.window.window().set_cursor_position(
                        (ammolite.window_dimensions[0].get() as f64 / 2.0, ammolite.window_dimensions[1].get() as f64 / 2.0).into()
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
                } => ammolite.window_swapchain_recreate = true,

                _ => ()
            }
        });
    }
}
