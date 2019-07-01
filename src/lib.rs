//! TODO:
//! * Window/HMD event handling separation
//! * Use secondary command buffers to parallelize their creation
//! * Mip Mapping
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

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::path::Path;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::cell::RefCell;
use std::ffi::{self, CString};
use core::num::NonZeroU32;
use arrayvec::ArrayVec;
use openxr::Instance as XrInstance;
use vulkano;
use vulkano::VulkanObject;
use vulkano::swapchain::ColorSpace;
use vulkano::instance::RawInstanceExtensions;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, RawDeviceExtensions, DeviceExtensions, Queue, Features};
use vulkano::instance::{ApplicationInfo, Instance as VkInstance, PhysicalDevice, QueueFamily};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::format::{Format, ClearValue, R8G8B8A8Srgb};
use vulkano::image::ImageUsage;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{PresentMode, SurfaceTransform, AcquireError, SwapchainCreationError, Surface};
use vulkano_win::VkSurfaceBuild;
use winit::{ElementState, MouseButton, Event, DeviceEvent, WindowEvent, KeyboardInput, VirtualKeyCode, EventsLoop, WindowBuilder, Window};
use winit::dpi::PhysicalSize;

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
use crate::swapchain::{Swapchain, VkSwapchain, XrSwapchain};

pub use crate::shaders::gltf_opaque_frag::ty::*;

pub type XrVkSession = openxr::Session<openxr::Vulkan>;
pub type XrVkFrameStream = openxr::FrameStream<openxr::Vulkan>;

pub const NONZERO_ONE: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(1) };

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

pub struct ViewSwapchain {
    pub swapchain: Arc<dyn Swapchain>,
    pub index: usize,
    pub medium_index: usize,
    pub index_within_medium: usize,
    /// Swapchain images, those that get presented in a window or in an HMD
    pub framebuffers: Option<Vec<Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>>>,
    pub recreate: bool, // TODO: consider representing as Option<ViewSwapchain>
}

impl ViewSwapchain {
    fn new(
        swapchain: Arc<dyn Swapchain>,
        index: usize,
        medium_index: usize,
        index_within_medium: usize
    ) -> Self {
        Self {
            swapchain,
            index,
            medium_index,
            index_within_medium,
            framebuffers: None,
            recreate: false,
        }
    }
}

pub struct ViewSwapchains {
    pub inner: Vec<Arc<RwLock<ViewSwapchain>>>,
    pub format: Format,
}

impl<T> From<T> for ViewSwapchains
        where T: IntoIterator<Item=Box<dyn Medium>> {
    fn from(into_iter: T) -> Self {
        let inner: Vec<Arc<RwLock<ViewSwapchain>>> = {
            let mut inner = Vec::new();
            let mut index = 0;

            for (medium_index, medium) in into_iter.into_iter().enumerate() {
                for (index_within_medium, swapchain) in medium.swapchains().iter().enumerate() {
                    inner.push(Arc::new(RwLock::new(ViewSwapchain::new(
                        swapchain.clone(),
                        index,
                        medium_index,
                        index_within_medium,
                    ))));
                }

                index += 1;
            }

            inner
        };

        let format = {
            let format = inner[0].read().unwrap().swapchain.format();

            for view_swapchain in inner.iter().skip(1) {
                assert_eq!(format, view_swapchain.read().unwrap().swapchain.format(), "All swapchains must use the same format.");
            }

            format
        };

        ViewSwapchains { inner, format }
    }
}

/**
 * An internal type to represent a viewing medium, e.g. a Window or a stereo HMD.
 */
pub trait Medium {
    // type Event;
    fn swapchains(&self) -> &[Arc<dyn Swapchain>];
    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event);
}

pub struct WindowMedium {
    pub window: Arc<Surface<Window>>,
    // pub window_events_loop: Rc<RefCell<EventsLoop>>,
    swapchain: Arc<dyn Swapchain>,
}

impl Medium for WindowMedium {
    // type Event = winit::Event;

    fn swapchains(&self) -> &[Arc<dyn Swapchain>] {
        std::slice::from_ref(&self.swapchain)
    }

    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event) {
    //     self.window_events_loop.clone().as_ref().borrow_mut().poll_events(|event| {
    //         // TODO custom internal handling
    //         event_handler(event);
    //     })
    // }
}

pub struct XrMedium {
    pub xr_instance: XrInstance,
    pub xr_session: XrVkSession,
    pub xr_reference_space_stage: openxr::Space,
    pub xr_frame_stream: XrVkFrameStream,
    // HMD devices typically have 2 screens, one for each eye
    swapchains: ArrayVec<[Arc<dyn Swapchain>; 2]>,
}

impl<'a> Medium for XrMedium {
    // type Event = openxr::Event<'a>;

    fn swapchains(&self) -> &[Arc<dyn Swapchain>] {
        &self.swapchains[..]
    }

    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event) {
    //     let mut event_data_buffer = openxr::EventDataBuffer::new();

    //     while let Some(event) = self.xr_instance.poll_event().unwrap() {
    //         // TODO custom internal handling
    //         event_handler(event);
    //     }
    // }
}


pub struct ChosenQueues {
    pub graphics: Arc<Queue>,
    pub transfer: Arc<Queue>,
}

macro_rules! type_level_enum {
    ($trait:ident; $($variant:ident),+) => {
        paste::item! {
            pub trait [< $trait Trait >] {}

            pub mod $trait {
                $(
                    pub struct $variant;
                    impl super::[< $trait Trait >] for $variant {}
                )+
            }
        }
    }
}

type_level_enum!(OpenXrInitialized; True, False);
type_level_enum!(VulkanInitialized; True, False);

pub struct XrContext {
    instance: XrInstance,
    system: openxr::SystemId,
    stereo_hmd_mediums: ArrayVec<[XrMedium; 1]>,
}

pub struct AmmoliteBuilder<'a, A: OpenXrInitializedTrait, B: VulkanInitializedTrait> {
    application_name: &'a str,
    application_version: (u16, u16, u16),
    xr: Option<XrContext>,
    // TODO: Transform into a single `vk: Option<VkContext>`
    vk_instance: Option<Arc<VkInstance>>,
    vk_device: Option<Arc<Device>>,
    vk_queues: Option<ChosenQueues>,
    window_mediums: ArrayVec<[WindowMedium; 1]>,
    _marker: PhantomData<(A, B)>,
}

// TODO: refactor to use `Self` and remove generic parameters?
impl<'a, A: OpenXrInitializedTrait, B: VulkanInitializedTrait> AmmoliteBuilder<'a, A, B> {
    pub fn new(
        application_name: &'a str,
        application_version: (u16, u16, u16),
    ) -> AmmoliteBuilder<'a, OpenXrInitialized::False, VulkanInitialized::False> {
        AmmoliteBuilder {
            application_name,
            application_version,
            xr: None,
            vk_instance: None,
            vk_device: None,
            vk_queues: None,
            window_mediums: ArrayVec::new(),
            _marker: PhantomData,
        }
    }

    /**
     * A helper function to be only used within the builder pattern implementation
     */
    unsafe fn coerce<
        OA: OpenXrInitializedTrait,
        OB: VulkanInitializedTrait,
    >(self) -> AmmoliteBuilder<'a, OA, OB> {
        AmmoliteBuilder {
            application_name: self.application_name,
            application_version: self.application_version,
            xr: self.xr,
            vk_instance: self.vk_instance,
            vk_device: self.vk_device,
            vk_queues: self.vk_queues,
            window_mediums: self.window_mediums,
            _marker: PhantomData,
        }
    }
}

impl<'a, B: VulkanInitializedTrait> AmmoliteBuilder<'a, OpenXrInitialized::False, B> {
    // FIXME: Should return a Result and fail modifying the builder, if the
    //        OpenXR instance could not have been created.
    //
    // Example:
    // Result<AmmoliteBuilder<'a, OpenXrInitialized::True, B>, (Error, AmmoliteBuilder<'a, OpenXrInitialized::False, B>)>
    pub fn initialize_openxr(
        self,
        application_name: impl AsRef<str>,
        application_version: impl Into<openxr::Version>,
    ) -> AmmoliteBuilder<'a, OpenXrInitialized::True, B> {
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

        let engine_version = openxr::Version::new(
            env!("CARGO_PKG_VERSION_MAJOR").parse()
                .expect("Invalid crate major version, must be u16."),
            env!("CARGO_PKG_VERSION_MINOR").parse()
                .expect("Invalid crate minor version, must be u16."),
            env!("CARGO_PKG_VERSION_PATCH").parse()
                .expect("Invalid crate patch version, must be u16."),
        );

        let app_info = openxr::ApplicationInfo {
            engine_name: env!("CARGO_PKG_NAME"),
            engine_version: engine_version.into_raw(),
            application_name: application_name.as_ref(),
            application_version: application_version.into().into_raw(),
        };

        let xr_instance = Arc::new(entry.create_instance(&app_info, &used_extensions).unwrap());
        let xr_system = xr_instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        AmmoliteBuilder {
            xr: Some(XrContext {
                instance: xr_instance,
                system: xr_system,
                stereo_hmd_mediums: ArrayVec::new(),
            }),
            .. unsafe { self.coerce() }
        }
    }
}

impl<'a> AmmoliteBuilder<'a, OpenXrInitialized::True, VulkanInitialized::False> {
    pub fn initialize_vulkan<'b, 'c>(
        self,
        application_name: impl AsRef<str>,
        application_version: impl Into<vulkano::instance::Version>,
        xr_instance: &'b Arc<XrInstance>,
        xr_system: openxr::SystemId,
        window_builders: impl IntoIterator<Item=(&'c EventsLoop, WindowBuilder)>,
    ) -> AmmoliteBuilder<'a, OpenXrInitialized::True, VulkanInitialized::True> {
        let app_info = {
            let engine_name = env!("CARGO_PKG_NAME");
            let engine_version = vulkano::instance::Version {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
            };

            ApplicationInfo {
                application_name: Some(application_name.as_ref().into()),
                application_version: Some(application_version.into()),
                engine_name: Some(engine_name.into()),
                engine_version: Some(engine_version.into()),
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
        let vk_instance: Arc<VkInstance> = VkInstance::new(Some(&app_info), extensions, layers.into_iter().cloned())
            .expect("Failed to create a Vulkan instance.");

        let openxr::vulkan::Requirements {
            min_api_version_supported: min_api_version,
            max_api_version_supported: max_api_version,
        } = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system).unwrap();
        let min_api_version = min_api_version.into_raw();
        let max_api_version = max_api_version.into_raw();
        // TODO: Better device selection & SLI support
        let physical_device = PhysicalDevice::enumerate(&vk_instance)
            .filter(|physical_device| {
                let api_version = physical_device.api_version().into_vulkan_version();

                api_version >= min_api_version && api_version <= max_api_version
            })
            .next().expect("No physical device available.");

        let windows: Vec<_> = window_builders.into_iter()
            .map(|(events_loop, window_builder)| window_builder.build_vk_surface(events_loop, vk_instance.clone()).unwrap())
            .collect();

        struct ChosenQueueFamilies<'a> {
            graphics: QueueFamily<'a>,
            transfer: QueueFamily<'a>,
            compute: QueueFamily<'a>,
        }

        fn choose_queue_families<'a, 'b>(windows: &'b [Arc<Surface<Window>>], physical_device: PhysicalDevice<'a>) -> ChosenQueueFamilies<'a> {
            fn allow_graphics<'b>(windows: &'b [Arc<Surface<Window>>]) -> impl for<'a, 'c> FnMut(&'a QueueFamily<'c>) -> bool + 'b {
                |queue_family: &QueueFamily| {
                    queue_family.supports_graphics()
                        && windows.iter().all(|window| window.is_supported(*queue_family).unwrap_or(false))
                }
            }

            fn compare_graphics<'a>(a: &QueueFamily<'a>, b: &QueueFamily<'a>) -> Ordering {
                a.queues_count().cmp(&b.queues_count()).reverse()
            }

            fn allow_transfer<'a>(queue_family: &QueueFamily<'a>) -> bool {
                queue_family.explicitly_supports_transfers()
            }

            fn compare_transfer<'a>(a: &QueueFamily<'a>, b: &QueueFamily<'a>) -> Ordering {
                a.supports_graphics().cmp(&b.supports_graphics())
                    .then_with(|| a.supports_compute().cmp(&b.supports_compute()))
                    .then_with(|| a.queues_count().cmp(&b.queues_count()).reverse())
            }

            fn allow_compute<'a>(queue_family: &QueueFamily<'a>) -> bool {
                queue_family.supports_compute()
            }

            fn compare_compute<'a>(a: &QueueFamily<'a>, b: &QueueFamily<'a>) -> Ordering {
                a.supports_graphics().cmp(&b.supports_graphics())
                    .then_with(|| a.queues_count().cmp(&b.queues_count()).reverse())
            }

            fn custom_min<'a, T>(comparator: impl for<'b> FnMut(&'b T, &'b T) -> Ordering, a: &'a T, b: &'a T) -> &'a T {
                match comparator(a, b) {
                    Ordering::Greater => b,
                    _ => a,
                }
            }

            fn consider_choice<T: Copy>(predicate: impl for<'b> FnMut(&'b T) -> bool, comparator: impl for<'b> FnMut(&'b T, &'b T) -> Ordering, a: Option<T>, b: T) -> Option<T> {
                if predicate(&b) {
                    Some(if let Some(a) = a {
                        *custom_min(comparator, &a, &b)
                    } else {
                        b
                    })
                } else {
                    a
                }
            }

            let mut graphics = None;
            let mut transfer = None;
            let mut compute = None;

            for queue_family in physical_device.queue_families() {
                graphics = consider_choice(allow_graphics(windows), compare_graphics, graphics, queue_family);
                transfer = consider_choice(allow_transfer, compare_transfer, transfer, queue_family);
                compute = consider_choice(allow_compute, compare_compute, compute, queue_family);
            }

            ChosenQueueFamilies {
                graphics: graphics.expect("Could not find a suitable graphics queue family."),
                transfer: transfer.expect("Could not find a suitable transfer queue family."),
                compute: compute.expect("Could not find a suitable compute queue family."),
            }
        }

        let chosen_queue_families = choose_queue_families(&windows[..], physical_device);

        /*
         * Device Creation
         */

        // Create a device with a single queue
        let (vk_device, mut queue_iter) = {
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
                        // Used to handle data transfers between the CPU and the GPU in parallel.
                        // TODO: Request a compute queue as well, or use the graphics queue for
                        // compute operations (would require additional limits for the queue family
                        // selection in `allow_graphics`).
                        [(chosen_queue_families.graphics, 0.5),
                         (chosen_queue_families.transfer, 0.5)].iter().cloned())
                .expect("Failed to create a Vulkan device.")
        };

        let vk_queues = ChosenQueues {
            graphics: queue_iter.next().unwrap(),
            transfer: queue_iter.next().unwrap(),
        };

        AmmoliteBuilder {
            vk_instance: Some(vk_instance),
            vk_device: Some(vk_device),
            vk_queues: Some(vk_queues),
            .. unsafe { self.coerce() }
        }
    }

    // fn register_medium() {
    //     if !xr_instance.enumerate_view_configurations(xr_system).unwrap()
    //                    .contains(&openxr::ViewConfigurationType::PRIMARY_STEREO) {
    //         panic!("No HMD Stereo View available.");
    //     }

    //     let view_config_views = xr_instance
    //         .enumerate_view_configuration_views(xr_system, openxr::ViewConfigurationType::PRIMARY_STEREO)
    //         .unwrap();

    //     let (xr_session, xr_frame_stream) = Self::setup_openxr_session(&xr_instance, &vk_instance, xr_system, &device, &queue);
    //     let xr_session = Arc::new(xr_session);

    //     let swapchains = {
    //         if true {
    //             let mut swapchains = Vec::with_capacity(view_config_views.len());

    //             for (_index, view) in view_config_views.into_iter().enumerate() {
    //                 let dimensions = [
    //                     NonZeroU32::new(view.recommended_image_rect_width).unwrap(),
    //                     NonZeroU32::new(view.recommended_image_rect_height).unwrap(),
    //                 ];

    //                 let swapchain = XrSwapchain::new(
    //                     device.clone(),
    //                     xr_session.clone(),
    //                     dimensions,
    //                     crate::NONZERO_ONE,
    //                     R8G8B8A8Srgb,
    //                     ImageUsage::all(), // TODO
    //                     NonZeroU32::new(view.recommended_swapchain_sample_count).unwrap(),
    //                 );

    //                 swapchains.push(Box::new(swapchain) as Box<dyn Swapchain>);
    //             }

    //             swapchains
    //         } else {
    //             let capabilities = window.capabilities(physical_device)
    //                 .expect("Failed to retrieve surface capabilities.");

    //             // Determines the behaviour of the alpha channel
    //             let alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
    //             dimensions = capabilities.current_extent.map(|extent| [
    //                 NonZeroU32::new(extent[0]).unwrap(),
    //                 NonZeroU32::new(extent[1]).unwrap(),
    //             ]).unwrap_or(dimensions);

    //             // Order supported swapchain formats by priority and choose the most preferred one.
    //             // The swapchain format must be in SRGB space.
    //             let mut supported_formats: Vec<Format> = capabilities.supported_formats.iter()
    //                 .map(|(current_format, _)| *current_format)
    //                 .collect();
    //             supported_formats.sort_by(swapchain_format_compare);
    //             let format = supported_formats[0];

    //             // Please take a look at the docs for the meaning of the parameters we didn't mention.
    //             let swapchain: VkSwapchain<Window> = vulkano::swapchain::Swapchain::new::<_, &Arc<Queue>>(
    //                 device.clone(),
    //                 window.clone(),
    //                 (&queue).into(),
    //                 dimensions,
    //                 NonZeroU32::new(1).unwrap(),
    //                 NonZeroU32::new(capabilities.min_image_count).expect("Invalid swapchaing image count."),
    //                 format,
    //                 ColorSpace::SrgbNonLinear,
    //                 capabilities.supported_usage_flags,
    //                 SurfaceTransform::Identity,
    //                 alpha,
    //                 PresentMode::Immediate, /* PresentMode::Relaxed TODO: Getting only ~60 FPS in a window */
    //                 true,
    //                 None,
    //             ).expect("failed to create swapchain").into();

    //             vec![Box::new(swapchain) as Box<dyn Swapchain>]
    //         }
    //     };
    //     let view_swapchains: Arc<ViewSwapchains> = Arc::new(swapchains.into());

    //     xr_session.begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
    //               .unwrap();

    //     (xr_session, xr_frame_stream, events_loop, window, dimensions, device, queue_family, queue, view_swapchains)
    // }
}

impl<'a> AmmoliteBuilder<'a, OpenXrInitialized::True, VulkanInitialized::True> {
    pub fn build(self) -> Ammolite {
        let AmmoliteBuilder {
            xr,
            vk_instance,
            vk_device,
            vk_queues,
            window_mediums,
            ..
        } = self;
        let XrContext {
            instance: xr_instance,
            system: xr_system,
            stereo_hmd_mediums,
        } = xr.unwrap();
        let vk_instance = vk_instance.unwrap();
        let vk_device = vk_device.unwrap();
        let vk_queues = vk_queues.unwrap();


        let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
        let (init_command_buffer_builder, helper_resources) = HelperResources::new(
            &device,
            [queue_family].into_iter().cloned(),
        ).unwrap()
            .initialize_resource(&device, queue_family, init_command_buffer_builder).unwrap();
        let (init_command_buffer_builder, pipeline_cache) = GraphicsPipelineSetCache::create(device.clone(), &view_swapchains, helper_resources.clone(), queue_family)
            .initialize_resource(&device, queue_family, init_command_buffer_builder).unwrap();
        let init_command_buffer = init_command_buffer_builder.build().unwrap();

        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
        // that, we store the submission of the previous frame here.
        let synchronization: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone())
            .then_execute(queue.clone(), init_command_buffer).unwrap()
            // .then_signal_fence()
            // .then_execute_same_queue(init_unsafe_command_buffer).unwrap()
            .then_signal_fence_and_flush().unwrap());

        Ammolite {
            vk_instance: self.vk_instance.unwrap(),
            xr_instance: self.xr.as_,
            xr_session,
            xr_frame_stream,
            xr_reference_space_stage,
            device: device.clone(),
            queue,
            pipeline_cache,
            helper_resources,
            window,
            window_events_loop: Rc::new(RefCell::new(events_loop)),
            window_dimensions: dimensions,
            view_swapchains,
            synchronization: Some(synchronization),
            buffer_pool_uniform_instance: CpuBufferPool::uniform_buffer(device),
            camera_position: Vec3::zero(),
            camera_view_matrix: Mat4::identity(),
            camera_projection_matrix: Mat4::identity(),
        }
    }
}


impl<'a, A: OpenXrInitializedTrait> AmmoliteBuilder<'a, A, VulkanInitialized::True> {
    pub fn add_medium_window(
        self,
        window: Arc<Surface<Window>>,
    ) -> Self {
        let mut dimensions: [NonZeroU32; 2] = {
            let (width, height) = window.window().get_inner_size().unwrap().into();
            [
                NonZeroU32::new(width).expect("The width of the window must not be 0."),
                NonZeroU32::new(height).expect("The height of the window must not be 0."),
            ]
        };
        let vk_physical_device = self.vk_device.as_ref().unwrap().physical_device();
        let capabilities = window.capabilities(vk_physical_device)
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
        let swapchain: VkSwapchain<Window> = vulkano::swapchain::Swapchain::new::<_, &Arc<Queue>>(
            self.vk_device.as_ref().unwrap().clone(),
            window.clone(),
            (&self.vk_queues.as_ref().unwrap().graphics).into(),
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
        ).expect("failed to create swapchain").into();

        let window_medium = WindowMedium {
            window,
            swapchain: Arc::new(swapchain) as Arc<dyn Swapchain>,
        };

        self.window_mediums.push(window_medium);

        self
    }
}

impl<'a> AmmoliteBuilder<'a, OpenXrInitialized::True, VulkanInitialized::True> {
    pub fn add_medium_stereo_hmd(
        self,
    ) -> (Self, XrVkSession) {
        let XrContext {
            instance: xr_instance,
            system: xr_system,
            stereo_hmd_mediums,
        } = self.xr.take().unwrap();

        if !xr_instance.enumerate_view_configurations(xr_system).unwrap()
                       .contains(&openxr::ViewConfigurationType::PRIMARY_STEREO) {
            panic!("No HMD Stereo View available.");
        }

        let view_config_views = xr_instance
            .enumerate_view_configuration_views(xr_system, openxr::ViewConfigurationType::PRIMARY_STEREO)
            .unwrap();

        fn setup_openxr_session(
            xr_instance: &XrInstance,
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

        let (xr_session, xr_frame_stream) = setup_openxr_session(
            &xr_instance, self.vk_instance.as_ref().unwrap(), xr_system,
            self.vk_device.as_ref().unwrap(), &self.vk_queues.as_ref().unwrap().graphics,
        );

        let swapchains: ArrayVec<[Arc<dyn Swapchain>; 2]> = {
            let mut swapchains = ArrayVec::new();

            for (_index, view) in view_config_views.into_iter().enumerate() {
                let dimensions = [
                    NonZeroU32::new(view.recommended_image_rect_width).unwrap(),
                    NonZeroU32::new(view.recommended_image_rect_height).unwrap(),
                ];

                let swapchain = XrSwapchain::new(
                    self.vk_device.as_ref().unwrap().clone(),
                    xr_session.clone(),
                    dimensions,
                    crate::NONZERO_ONE,
                    R8G8B8A8Srgb,
                    ImageUsage::all(), // FIXME
                    NonZeroU32::new(view.recommended_swapchain_sample_count).unwrap(),
                );

                swapchains.push(Arc::new(swapchain) as Arc<dyn Swapchain>);
            }

            swapchains
        };

        xr_session.begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
                  .unwrap();

        let xr_reference_space_stage = xr_session.create_reference_space(
            openxr::ReferenceSpaceType::STAGE,
            openxr::Posef {
                orientation: openxr::Quaternionf { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                position: openxr::Vector3f { x: 0.0, y: 0.0, z: 0.0 },
            },
        ).unwrap();

        let xr_medium = XrMedium {
            xr_instance: xr_instance.clone(),
            xr_session: xr_session.clone(),
            xr_reference_space_stage,
            xr_frame_stream,
            swapchains,
        };

        stereo_hmd_mediums.push(xr_medium);

        (AmmoliteBuilder {
            xr: Some(XrContext {
                instance: xr_instance,
                system: xr_system,
                stereo_hmd_mediums,
            }),
            .. self
        }, xr_session)
    }
}

pub struct Ammolite {
    /// The Vulkan runtime implementation
    pub vk_instance: Arc<VkInstance>,
    /// The OpenXR runtime implementation
    pub xr_instance: XrInstance,
    pub xr_session: XrVkSession,
    pub xr_frame_stream: XrVkFrameStream,
    pub xr_reference_space_stage: openxr::Space,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pipeline_cache: GraphicsPipelineSetCache,
    pub helper_resources: HelperResources,
    pub window: Arc<Surface<Window>>,
    pub window_events_loop: Rc<RefCell<EventsLoop>>,
    pub window_dimensions: [NonZeroU32; 2],
    pub view_swapchains: Arc<ViewSwapchains>,
    pub synchronization: Option<Box<dyn GpuFuture>>,
    // TODO Consider moving to SharedGltfGraphicsPipelineResources
    pub buffer_pool_uniform_instance: CpuBufferPool<InstanceUBO>,
    pub camera_position: Vec3,
    pub camera_view_matrix: Mat4,
    pub camera_projection_matrix: Mat4,
}

impl Ammolite {
    pub fn builder<'a>(
        application_name: &'a str,
        application_version: (u16, u16, u16),
    ) -> AmmoliteBuilder<'a, OpenXrInitialized::False, VulkanInitialized::False> {
        AmmoliteBuilder::new(application_name, application_version)
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
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        self.synchronization.as_mut().unwrap().cleanup_finished();

        let state = self.xr_frame_stream.wait().unwrap();
        let (_view_flags, views) = self.xr_session
            .locate_views(state.predicted_display_time, &self.xr_reference_space_stage)
            .unwrap();
        let status = self.xr_frame_stream.begin().unwrap();

        if status != openxr::sys::Result::SESSION_VISIBILITY_UNAVAILABLE {
            let world_space_models = model_provider();

            for view_swapchain in &self.view_swapchains.clone().inner[..] {
                let mut view_swapchain = view_swapchain.write().unwrap();
                // A loop is used as a way to reset the rendering using `continue` if something goes wrong,
                // there is a `break` statement at the end.
                loop {
                    if view_swapchain.recreate {
                        // FIXME: uncomment, but enable only for window swapchains
                        // self.window_dimensions = {
                        //     let dpi = self.window.window().get_hidpi_factor();
                        //     let (width, height): (u32, u32) = self.window.window().get_inner_size().unwrap().to_physical(dpi)
                        //         .into();
                        //     [
                        //         NonZeroU32::new(width).expect("The width of the window must not be 0."),
                        //         NonZeroU32::new(height).expect("The height of the window must not be 0."),
                        //     ]
                        // };

                        match view_swapchain.swapchain.recreate_with_dimension(self.window_dimensions) {
                            Ok(()) => (),
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::UnsupportedDimensions) => {
                                continue;
                            },
                            Err(err) => panic!("{:?}", err)
                        };

                        view_swapchain.framebuffers = None;
                        view_swapchain.recreate = false;
                    }

                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    if view_swapchain.framebuffers.is_none() {
                        self.pipeline_cache.shared_resources
                            .reconstruct_dimensions_dependent_images(&view_swapchain)
                            .expect("Could not reconstruct dimension dependent resources.");

                        view_swapchain.framebuffers = Some(
                            self.pipeline_cache.shared_resources.construct_swapchain_framebuffers(
                                self.pipeline_cache.render_pass.clone(),
                                &view_swapchain,
                            )
                        );

                        for (_, pipeline) in self.pipeline_cache.pipeline_map.write().unwrap().iter_mut() {
                            let per_pipeline = |pipeline: &mut GltfGraphicsPipeline| {
                                pipeline.layout_dependent_resources
                                    .reconstruct_descriptor_sets(&self.pipeline_cache.shared_resources, &self.view_swapchains, &view_swapchain);
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
                    let (multilayer_image, acquire_future) = match view_swapchain.swapchain.acquire_next_image() {
                        Ok(result) => {
                            result
                        },
                        Err(AcquireError::OutOfDate) => {
                            view_swapchain.recreate = true;
                            continue;
                        },
                        Err(err) => panic!("{:?}", err)
                    };

                    let secs_elapsed = ((elapsed.as_secs() as f64) + (elapsed.as_nanos() as f64) / (1_000_000_000f64)) as f32;

                    self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
                        .join(acquire_future)));

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
                    // self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
                    //     .then_execute(self.queue.clone(), buffer_updates).unwrap()
                    //     .join(acquire_future)));
                    self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
                        .then_execute(self.queue.clone(), buffer_updates).unwrap()));

                    let current_framebuffer: Arc<dyn FramebufferWithClearValues<_>> = view_swapchain.framebuffers
                        .as_ref().unwrap()[multilayer_image.index()].clone();

                    self.render_instances(current_framebuffer, world_space_models, &view_swapchain);

                    let result = self.synchronization.take().unwrap()
                        .then_signal_fence();
                        // The color output is now expected to contain our triangle. But in order to show it on
                        // the screen, we have to *present* the image by calling `present`.
                        // This function does not actually present the image immediately. Instead it submits a
                        // present command at the end of the queue. This means that it will only be presented once
                        // the GPU has finished executing the command buffer that draws the triangle.
                    let result = view_swapchain.swapchain.present(Box::new(result), self.queue.clone(), multilayer_image.index())
                        .then_signal_fence_and_flush();

                    self.synchronization = Some(match result {
                        Ok(future) => Box::new(future) as Box<dyn GpuFuture>,
                        Err(FlushError::OutOfDate) => {
                            view_swapchain.recreate = true;
                            Box::new(vulkano::sync::now(self.device.clone())) as Box<dyn GpuFuture>
                        },
                        Err(err) => panic!("{:?}", err),
                    });

                    if view_swapchain.recreate {
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

            for view_swapchain in &self.view_swapchains.inner[..] {
                let mut view_swapchain = view_swapchain.write().unwrap();

                view_swapchain.swapchain.finish_rendering();
            }
        }

        // TODO: do not reallocate the vec every time render is called
        let view_swapchains = self.view_swapchains.inner.iter()
            .map(|view_swapchain| view_swapchain.read().unwrap())
            .collect::<Vec<_>>();
        let swapchains = view_swapchains.iter()
            .map(|view_swapchain| &view_swapchain.swapchain)
            .collect::<Vec<_>>();
        let composition_layers: Vec<openxr::CompositionLayerProjectionView<_>> = {
            let mut composition_layers = Vec::with_capacity(self.view_swapchains.inner.len());

            for (index, swapchain) in swapchains.iter().enumerate() {
                let dimensions = swapchain.dimensions();
                composition_layers.push(openxr::CompositionLayerProjectionView::new()
                    .pose(views[index].pose)
                    .fov(views[index].fov)
                    .sub_image(
                        openxr::SwapchainSubImage::new()
                        .swapchain(swapchain.downcast_xr().unwrap().inner())
                        .image_array_index(0)
                        .image_rect(openxr::Rect2Di {
                            offset: openxr::Offset2Di { x: 0, y: 0 },
                            extent: openxr::Extent2Di { // FIXME
                                width: dimensions[0].get() as i32,
                                height: dimensions[1].get() as i32,
                            },
                        })
                    )
                )
            }

            composition_layers
        };

        self.xr_frame_stream.end(
            state.predicted_display_time,
            openxr::EnvironmentBlendMode::OPAQUE,
            &[&openxr::CompositionLayerProjection::new()
                .space(&self.xr_reference_space_stage)
                .views(&composition_layers[..])]
        ).unwrap();
    }

    fn render_instances<'a>(&mut self,
                            current_framebuffer: Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>,
                            world_space_models: &'a [WorldSpaceModel<'a>],
                            view_swapchain: &'a ViewSwapchain) {
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
                dimensions: {
                    let dimensions = view_swapchain.swapchain.dimensions();
                    [dimensions[0].get() as f32, dimensions[1].get() as f32]
                },
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
            view_swapchain,
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
                } => {
                    // TODO: recreate only actual window swapchains, not HMD ones?
                    for view_swapchain in &ammolite.view_swapchains.inner[..] {
                        let mut view_swapchain = view_swapchain.write().unwrap();
                        view_swapchain.recreate = true;
                    }
                }

                _ => ()
            }
        });
    }
}
