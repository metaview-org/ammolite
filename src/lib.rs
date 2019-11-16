//! TODO:
//! * Window/HMD event handling separation
//! * Use secondary command buffers to parallelize their creation
//! * Mip Mapping
//! * Instancing
//! * Animations
//! * Morph primitives

#![feature(core_intrinsics)]

pub mod buffer;
pub mod camera;
pub mod iter;
pub mod model;
pub mod pipeline;
pub mod sampler;
pub mod shaders;
pub mod swapchain;
pub mod vertex;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::path::Path;
use std::collections::{VecDeque, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};
use std::ffi::{self, CString};
use core::num::NonZeroU32;
use arrayvec::ArrayVec;
use arr_macro::arr;
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
use vulkano::format::*;
use vulkano::image::ImageUsage;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{PresentMode, SurfaceTransform, AcquireError, SwapchainCreationError, Surface};
use vulkano_win::VkSurfaceBuild;
use winit::{ElementState, MouseButton, Event, DeviceEvent, WindowEvent, KeyboardInput, VirtualKeyCode, EventsLoop, WindowBuilder, Window};
use winit::dpi::PhysicalSize;
use smallvec::SmallVec;
use openxr::{View as XrView, FrameState as XrFrameState, FrameWaiter as XrFrameWaiter};

use ammolite_math::matrix::*;
use ammolite_math::vector::*;
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
use crate::iter::ArrayIterator;
use crate::swapchain::{Swapchain, VkSwapchain, XrSwapchain};

pub use crate::shaders::gltf_opaque_frag::ty::*;
pub use openxr::Instance as XrInstance;

pub type XrVkSession = openxr::Session<openxr::Vulkan>;
pub type XrVkFrameStream = openxr::FrameStream<openxr::Vulkan>;

pub const NONZERO_ONE: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(1) };

fn swapchain_format_priority(format: &Format) -> u32 {
    match *format {
        Format::R8G8B8Srgb | Format::R8G8B8A8Srgb => 0,
        Format::B8G8R8Srgb | Format::B8G8R8A8Srgb => 1,
        _ => 2,
    }
}

fn swapchain_format_compare(a: &Format, b: &Format) -> std::cmp::Ordering {
    swapchain_format_priority(a).cmp(&swapchain_format_priority(b))
}

fn into_raw_u32(version: (u16, u16, u16)) -> u32 {
    ((version.0 as u32 & 0x3FF) << 22)
        | ((version.1 as u32 & 0x3FF) << 12)
        | ((version.2 as u32 & 0xFFF) <<  0)
}

fn into_raw_u32_xr(version: openxr::Version) -> u32 {
    into_raw_u32((version.major(), version.minor(), version.patch() as u16))
}

// fn vulkan_find_supported_format(physical_device: &Arc<PhysicalDevice>, candidates: &[Format]) -> Option<Format> {
//     // TODO: Querying available formats is not implemented (exposed) in vulkano.
//     unimplemented!()
// }

#[derive(Clone)]
pub struct WorldSpaceModel<'a> {
    pub model: &'a Model,
    pub matrix: Mat4,
}

#[derive(Clone)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Clone)]
pub struct HomogeneousRay {
    pub origin: Vec4,
    pub direction: Vec4,
}

impl From<Ray> for HomogeneousRay {
    fn from(ray: Ray) -> Self {
        Self {
            origin: ray.origin.into_homogeneous_position(),
            direction: ray.direction.into_homogeneous_direction(),
        }
    }
}

impl<'a, 'b> std::ops::Mul<&'b Mat4> for &'a HomogeneousRay {
    type Output = HomogeneousRay;

    fn mul(self, matrix: &'b Mat4) -> HomogeneousRay {
        HomogeneousRay {
            origin: matrix * &self.origin,
            direction: matrix * &self.direction,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RayIntersection {
    pub distance: f32,
    pub node_index: usize,
    pub mesh_index: usize,
    pub primitive_index: usize,
}

pub fn intersect_convex_polygon(polygon: &[Vec3], ray: &HomogeneousRay) -> Option<f32> {
    if polygon.len() < 3 {
        return None
    }

    let projected_origin = ray.origin.into_projected();
    let projected_direction = Vec3([ray.direction.0[0], ray.direction.0[1], ray.direction.0[2]]);
    let mut sign: Option<bool> = None;
    let mut intersects_triangle = true;
    for edge_index in 0..polygon.len() {
        let vertex_from = &polygon[edge_index];
        let vertex_to = &polygon[(edge_index + 1) % polygon.len()];
        let edge = vertex_to - vertex_from;

        let new_origin = &projected_origin - vertex_from;
        let current_normal = edge.cross(&projected_direction);
        let current_sign = new_origin.dot(&current_normal);

        if current_sign != 0.0 {
            let current_sign_bool = current_sign > 0.0;
            if sign.is_some() && *sign.as_ref().unwrap() != current_sign_bool {
                intersects_triangle = false;
            }

            sign = Some(current_sign_bool)
        }
    }

    if intersects_triangle {
        let normal = (&polygon[1] - &polygon[0]).cross(&(&polygon[2] - &polygon[1])).normalize();
        let distance = -normal.dot(&(projected_origin - &polygon[0])) / normal.dot(&projected_direction);

        if distance >= 0.0 {
            return Some(distance);
        }
    }

    None
}

pub fn raytrace_distance(wsm: &WorldSpaceModel, ray: &Ray) -> Option<RayIntersection> {
    let instance_matrix_inverse = wsm.matrix.inverse();
    let ray: HomogeneousRay = ray.clone().into();
    let model = wsm.model;
    let scene = model.document().scenes().nth(0 /*TODO*/).unwrap();
    let mut closest: Option<RayIntersection> = None;
    let mut node_queue = VecDeque::new();

    for node in scene.nodes() {
        node_queue.push_back(node.index());
    }

    while let Some(node_index) = node_queue.pop_front() {
        let node = model.document().nodes().nth(node_index).unwrap();
        let node_transform_matrix = &model.node_transform_matrices()[node.index()];
        let ray_transform_matrix = node_transform_matrix.inverse();
        let transformed_ray = &(&ray * &instance_matrix_inverse) * &ray_transform_matrix;

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                for face in model.primitive_faces_iter(primitive.clone()) {
                    if let Some(distance) = intersect_convex_polygon(&face[..], &transformed_ray) {
                        if closest.is_none() || distance < closest.as_ref().unwrap().distance {
                            closest = Some(RayIntersection {
                                distance,
                                node_index: node.index(),
                                mesh_index: mesh.index(),
                                primitive_index: primitive.index(),
                            });
                        }
                    }
                }
            }
        }

        for child in node.children() {
            node_queue.push_back(child.index());
        }
    }

    closest
}

pub struct ViewSwapchain {
    pub swapchain: Box<dyn Swapchain>,
    // pub index: usize,
    // pub medium_index: usize,
    // pub index_within_medium: usize,
    /// Swapchain images, those that get presented in a window or in an HMD
    pub framebuffers: Option<Vec<Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>>>,
    pub recreate: bool, // TODO: consider representing as Option<ViewSwapchain>
}

impl ViewSwapchain {
    fn new(
        swapchain: Box<dyn Swapchain>,
        // index: usize,
        // medium_index: usize,
        // index_within_medium: usize
    ) -> Self {
        println!("format: {:?}", swapchain.format());
        Self {
            swapchain,
            // index,
            // medium_index,
            // index_within_medium,
            framebuffers: None,
            recreate: false,
        }
    }
}

// pub struct ViewSwapchains {
//     pub inner: Vec<Arc<RwLock<ViewSwapchain>>>,
//     pub format: Format,
// }

// impl<'a, T> From<T> for ViewSwapchains
//         where T: IntoIterator<Item=&'a dyn Medium> {
//     fn from(into_iter: T) -> Self {
//         let inner: Vec<Arc<RwLock<ViewSwapchain>>> = {
//             let mut inner = Vec::new();
//             let mut index = 0;

//             for (medium_index, medium) in into_iter.into_iter().enumerate() {
//                 for (index_within_medium, swapchain) in medium.swapchains().iter().enumerate() {
//                     inner.push(Arc::new(RwLock::new(ViewSwapchain::new(
//                         swapchain,
//                         index,
//                         medium_index,
//                         index_within_medium,
//                     ))));
//                 }

//                 index += 1;
//             }

//             inner
//         };

//         let format = {
//             let format = inner[0].read().unwrap().swapchain.format();

//             for view_swapchain in inner.iter().skip(1) {
//                 assert_eq!(format, view_swapchain.read().unwrap().swapchain.format(), "All swapchains must use the same format.");
//             }

//             format
//         };

//         ViewSwapchains { inner, format }
//     }
// }

#[derive(Clone, Default, Debug)]
pub struct ViewPose {
    pub orientation: Mat3,
    pub position: Vec3,
}

impl ViewPose {
    fn inverse_from(other: openxr::Posef) -> Self {
        Self {
            orientation: Mat3::from_quaternion([
                other.orientation.x,
                other.orientation.y,
                other.orientation.z,
                -other.orientation.w,
            ]),
            position: [
                -other.position.x,
                -other.position.y,
                -other.position.z,
            ].into(),
        }
    }
}

impl From<openxr::Posef> for ViewPose {
    fn from(other: openxr::Posef) -> Self {
        Self {
            orientation: Mat3::from_quaternion([
                other.orientation.x,
                other.orientation.y,
                other.orientation.z,
                other.orientation.w,
            ]),
            position: [
                other.position.x,
                other.position.y,
                other.position.z,
            ].into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ViewFov {
    pub angle_left: f32,
    pub angle_right: f32,
    pub angle_up: f32,
    pub angle_down: f32,
}

impl ViewFov {
    pub fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            angle_left:  -horizontal / 2.0,
            angle_right:  horizontal / 2.0,
            angle_up:     vertical   / 2.0,
            angle_down:  -vertical   / 2.0,
        }
    }
}

impl From<openxr::Fovf> for ViewFov {
    fn from(other: openxr::Fovf) -> Self {
        Self {
            angle_left: other.angle_left,
            angle_right: other.angle_right,
            angle_up: other.angle_up,
            angle_down: other.angle_down,
        }
    }
}

#[derive(Clone, Debug)]
pub struct View {
    pub pose: ViewPose,
    pub fov: ViewFov,
}

impl View {
    pub fn with_symmetric_fov(horizontal: f32, vertical: f32) -> Self {
        Self {
            pose: Default::default(),
            fov: ViewFov::symmetric(horizontal, vertical),
        }
    }

    fn inverse_from(other: openxr::View) -> Self {
        Self {
            pose: ViewPose::inverse_from(other.pose),
            fov: other.fov.into(),
        }
    }
}

impl From<openxr::View> for View {
    fn from(other: openxr::View) -> Self {
        Self {
            pose: other.pose.into(),
            fov: other.fov.into(),
        }
    }
}

/**
 * A command, which may not be implemented by all medium types.
 */
#[derive(Debug)]
pub enum MediumSpecificHandleEventsCommand {
    CenterCursorToWindow,
}

#[derive(Debug)]
pub enum HandleEventsCommand {
    Quit,
    MediumSpecific(MediumSpecificHandleEventsCommand),
    RecreateSwapchain(usize),
}

pub trait MediumData: Sized {
    // TODO: unify dimensions and view
    fn get_camera_transforms(&self, view_index: usize, view: &View, dimensions: [NonZeroU32; 2]) -> CameraTransforms;
    fn handle_events(&mut self, delta_time: &Duration) -> SmallVec<[HandleEventsCommand; 8]>;
}

/**
 * An internal type to represent a viewing medium, e.g. a Window or a stereo HMD.
 */
pub trait Medium<MD: MediumData> {
    // type Event;
    fn swapchains(&self) -> &[RefCell<ViewSwapchain>];
    // fn swapchains_mut(&mut self) -> &mut [ViewSwapchain];
    fn data(&self) -> &MD;
    fn data_mut(&mut self) -> &mut MD;
    // fn swapchains(&self) -> &[Box<dyn Swapchain>];
    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event);

    /**
     * Waits for a frame to become available. Returns `Some(vec)`, if rendering
     * for this medium should commence. Otherwise, returns `None`, then
     * rendering for this medium should be skipped.
     *
     * TODO: Remove allocation of Vec
     */
    fn wait_for_frame(&mut self) -> Option<Vec<View>>;

    /**
     * Gets called at the end of every frame, even when `wait_for_frame`
     * returned `false`.
     */
    fn finalize_frame(&mut self);

    /**
     * Returns the recommended render dimensions of the medium.
     */
    fn get_dimensions(&self) -> [NonZeroU32; 2];

    /**
     * Handle a medium-specific event command.
     * Returns `true`, when the command was handled, otherwise `false`, if this
     * medium does not handle this command.
     */
    fn handle_events_command(&mut self, command: MediumSpecificHandleEventsCommand) -> bool;
}



pub struct WindowMedium<MD: MediumData> {
    pub data: MD,
    pub window: Arc<Surface<Window>>,
    // pub window_events_loop: Rc<RefCell<EventsLoop>>,
    swapchain: RefCell<ViewSwapchain>,
    // swapchain: Box<dyn Swapchain>,
}

impl<MD: MediumData> Medium<MD> for WindowMedium<MD> {
    // type Event = winit::Event;

    fn swapchains(&self) -> &[RefCell<ViewSwapchain>] {
        std::slice::from_ref(&self.swapchain)
    }

    // fn swapchains_mut(&mut self) -> &mut [ViewSwapchain] {
    //     std::slice::from_mut(&mut self.swapchain)
    // }

    fn data(&self) -> &MD {
        &self.data
    }

    fn data_mut(&mut self) -> &mut MD {
        &mut self.data
    }

    // fn swapchains(&self) -> &[Box<dyn Swapchain>] {
    //     std::slice::from_ref(&self.swapchain)
    // }

    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event) {
    //     self.window_events_loop.clone().as_ref().borrow_mut().poll_events(|event| {
    //         // TODO custom internal handling
    //         event_handler(event);
    //     })
    // }

    fn wait_for_frame(&mut self) -> Option<Vec<View>> {
        let dimensions = self.get_dimensions();
        let aspect_ratio = dimensions[0].get() as f32 / dimensions[1].get() as f32;
        let vertical = std::f32::consts::FRAC_PI_2;
        let horizontal = 2.0 * (aspect_ratio * (vertical / 2.0).tan()).atan();

        Some(vec![View::with_symmetric_fov(horizontal, vertical)])
    }

    fn finalize_frame(&mut self) {}

    fn get_dimensions(&self) -> [NonZeroU32; 2] {
        let dpi = self.window.window().get_hidpi_factor();
        let (width, height): (u32, u32) = self.window.window().get_inner_size()
            .unwrap().to_physical(dpi).into();

        [
            NonZeroU32::new(width).expect("The width of the window must not be 0."),
            NonZeroU32::new(height).expect("The height of the window must not be 0."),
        ]
    }

    fn handle_events_command(&mut self, command: MediumSpecificHandleEventsCommand) -> bool {
        match command {
            CenterCursor => {
                let dimensions = self.get_dimensions();
                self.window.window().set_cursor_position(
                    (dimensions[0].get() as f64 / 2.0, dimensions[1].get() as f64 / 2.0).into()
                ).expect("Could not center the cursor position.");
            }
            _ => return false
        }

        true
    }
}

pub struct XrMedium<MD: MediumData> {
    pub data: MD,
    // TODO: Remove Arc
    pub xr_instance: Arc<XrInstance>,
    pub xr_session: XrVkSession,
    pub xr_reference_space_stage: openxr::Space,
    pub xr_frame_waiter: XrFrameWaiter,
    pub xr_frame_stream: RefCell<XrVkFrameStream>,
    // HMD devices typically have 2 screens, one for each eye
    swapchains: ArrayVec<[RefCell<ViewSwapchain>; 2]>,
    frame_state: Option<XrFrameState>,
    frame_views: Option<Vec<XrView>>,
    // swapchains: ArrayVec<[Box<dyn Swapchain>; 2]>,
}

impl<'a, MD: MediumData> Medium<MD> for XrMedium<MD> {
    // type Event = openxr::Event<'a>;

    fn swapchains(&self) -> &[RefCell<ViewSwapchain>] {
        &self.swapchains[..]
    }

    // fn swapchains_mut(&mut self) -> &mut [ViewSwapchain] {
    //     &mut self.swapchains[..]
    // }

    fn data(&self) -> &MD {
        &self.data
    }

    fn data_mut(&mut self) -> &mut MD {
        &mut self.data
    }

    // fn swapchains(&self) -> &[Box<dyn Swapchain>] {
    //     &self.swapchains[..]
    // }

    // fn poll_events<T>(&mut self, event_handler: T) where T: FnMut(Self::Event) {
    //     let mut event_data_buffer = openxr::EventDataBuffer::new();

    //     while let Some(event) = self.xr_instance.poll_event().unwrap() {
    //         // TODO custom internal handling
    //         event_handler(event);
    //     }
    // }

    fn wait_for_frame(&mut self) -> Option<Vec<View>> {
        let state = self.xr_frame_waiter.wait().unwrap();

        // Move to after framestream::begin, if necessary
        if !state.should_render {
            return None;
        }

        let (_view_flags, views) = self.xr_session
            .locate_views(openxr::ViewConfigurationType::PRIMARY_STEREO, state.predicted_display_time, &self.xr_reference_space_stage)
            .unwrap();
        self.frame_state = Some(state);
        self.frame_views = Some(views.clone());
        let status = self.xr_frame_stream.borrow_mut().begin().unwrap();

        Some(views.into_iter()
            .map(|mut view| {
                // Hacky HTC Vive tracking normalization
                // FIXME: There must be a way to simplify/fix this
                let mut v = View::inverse_from(view);
                v.pose.orientation = v.pose.orientation * {
                    let mut normalize = Mat3::identity();
                    normalize[1][1] = -1.0;
                    normalize[2][2] = -1.0;
                    normalize
                };
                v.pose.position[1] -= 1.6;
                v.pose.position[0] *= -1.0;
                v.pose.position = -v.pose.position;
                v
            })
            .collect())
    }

    fn finalize_frame(&mut self) {
        let views = self.frame_views.take().unwrap();
        let predicted_display_time = self.frame_state.take().unwrap().predicted_display_time;

        let view_swapchains = self.swapchains().iter()
            .map(|view_swapchain| view_swapchain.borrow())
            .collect::<Vec<_>>();

        let composition_layers: Vec<openxr::CompositionLayerProjectionView<_>> = {
            let mut composition_layers = Vec::with_capacity(self.swapchains().len());

            for (index, view_swapchain) in view_swapchains.iter().enumerate() {
                let swapchain = &view_swapchain.swapchain;
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

        self.xr_frame_stream.borrow_mut().end(
            predicted_display_time,
            openxr::EnvironmentBlendMode::OPAQUE,
            &[&openxr::CompositionLayerProjection::new()
                .space(&self.xr_reference_space_stage)
                .views(&composition_layers[..])]
        ).unwrap();
    }

    fn get_dimensions(&self) -> [NonZeroU32; 2] {
        unimplemented!()
    }

    fn handle_events_command(&mut self, command: MediumSpecificHandleEventsCommand) -> bool {
        match command {
            _ => return false
        }

        true
    }
}


pub struct ChosenQueues {
    pub graphics: Arc<Queue>,
    pub transfer: Arc<Queue>,
}

impl ChosenQueues {
    pub fn families<'a>(&'a self) -> impl IntoIterator<Item=QueueFamily<'a>> + Clone {
        ArrayIterator::new([self.graphics.family(), self.transfer.family()])
    }
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
type_level_enum!(WindowsAdded; True, False);
type_level_enum!(HmdsAdded; True, False);

pub struct XrContext<MD: MediumData> {
    // TODO: Remove Arc
    instance: Arc<XrInstance>,
    system: openxr::SystemId,
    stereo_hmd_mediums: ArrayVec<[XrMedium<MD>; 1]>,
}

pub struct UninitializedWindowMedium<MD> {
    pub events_loop: Rc<RefCell<EventsLoop>>,
    pub window_builder: WindowBuilder,
    pub window_handler: Option<Box<dyn FnOnce(&Arc<Surface<Window>>, &mut MD)>>,
    pub data: MD,
}

pub struct UninitializedStereoHmdMedium<MD> {
    pub instance_handler: Option<Box<dyn FnOnce(&Arc<XrInstance>, &XrVkSession, &mut MD)>>,
    pub data: MD,
}

pub struct AmmoliteBuilder<'a, MD: MediumData, A: OpenXrInitializedTrait, B: VulkanInitializedTrait, C: WindowsAddedTrait, D: HmdsAddedTrait> {
    application_name: &'a str,
    application_version: (u16, u16, u16),
    xr: Option<XrContext<MD>>,
    // TODO: Transform into a single `vk: Option<VkContext>`
    vk_instance: Option<Arc<VkInstance>>,
    vk_device: Option<Arc<Device>>,
    vk_queues: Option<ChosenQueues>,
    uninitialized_window_mediums: Option<ArrayVec<[UninitializedWindowMedium<MD>; 1]>>,
    window_mediums: ArrayVec<[WindowMedium<MD>; 1]>,
    _marker: PhantomData<(A, B, C, D)>,
}

impl<'a, MD: MediumData> AmmoliteBuilder<'a, MD, OpenXrInitialized::False, VulkanInitialized::False, WindowsAdded::False, HmdsAdded::False> {
    pub fn new(
        application_name: &'a str,
        application_version: (u16, u16, u16),
    ) -> Self {
        Self {
            application_name,
            application_version,
            xr: None,
            vk_instance: None,
            vk_device: None,
            vk_queues: None,
            uninitialized_window_mediums: None,
            window_mediums: ArrayVec::new(),
            _marker: PhantomData,
        }
    }
}

// TODO: refactor to use `Self` and remove generic parameters?
impl<'a, MD: MediumData, A: OpenXrInitializedTrait, B: VulkanInitializedTrait, C: WindowsAddedTrait, D: HmdsAddedTrait> AmmoliteBuilder<'a, MD, A, B, C, D> {
    /**
     * A helper function to be only used within the builder pattern implementation
     */
    unsafe fn coerce<
        OA: OpenXrInitializedTrait,
        OB: VulkanInitializedTrait,
        OC: WindowsAddedTrait,
        OD: HmdsAddedTrait,
    >(self) -> AmmoliteBuilder<'a, MD, OA, OB, OC, OD> {
        AmmoliteBuilder {
            application_name: self.application_name,
            application_version: self.application_version,
            xr: self.xr,
            vk_instance: self.vk_instance,
            vk_device: self.vk_device,
            vk_queues: self.vk_queues,
            uninitialized_window_mediums: self.uninitialized_window_mediums,
            window_mediums: self.window_mediums,
            _marker: PhantomData,
        }
    }
}

impl<'a, MD: MediumData, B: VulkanInitializedTrait> AmmoliteBuilder<'a, MD, OpenXrInitialized::False, B, WindowsAdded::False, HmdsAdded::False> {
    // FIXME: Should return a Result and fail modifying the builder, if the
    //        OpenXR instance could not have been created.
    //
    // Example:
    // Result<AmmoliteBuilder<'a, OpenXrInitialized::True, B>, (Error, AmmoliteBuilder<'a, OpenXrInitialized::False, B>)>
    pub fn initialize_openxr(
        self,
    ) -> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, B, WindowsAdded::False, HmdsAdded::False> {
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
            // khr_visibility_mask: available_extensions.khr_visibility_mask,
            // ext_debug_utils: available_extensions.ext_debug_utils,
            ..Default::default()
        };

        let engine_version = openxr::Version::new(
            env!("CARGO_PKG_VERSION_MAJOR").parse()
                .expect("Invalid crate major version, must be u16."),
            env!("CARGO_PKG_VERSION_MINOR").parse()
                .expect("Invalid crate minor version, must be u16."),
            env!("CARGO_PKG_VERSION_PATCH").parse()
                .expect("Invalid crate patch version, must be u32."),
        );

        let app_info = openxr::ApplicationInfo {
            engine_name: env!("CARGO_PKG_NAME"),
            engine_version: into_raw_u32_xr(engine_version),
            application_name: self.application_name.as_ref(),
            application_version: into_raw_u32(self.application_version),
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

impl<'a, MD: MediumData> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::False, WindowsAdded::False, HmdsAdded::False> {
    pub fn initialize_vulkan<'b, 'c>(
        mut self,
    ) -> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::True, WindowsAdded::False, HmdsAdded::False> {
        let xr = self.xr.take().unwrap();
        let app_info = {
            let engine_name = env!("CARGO_PKG_NAME");
            let engine_version = vulkano::instance::Version {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
            };

            ApplicationInfo {
                application_name: Some(Cow::Borrowed(self.application_name.as_ref())),
                application_version: Some(self.application_version.into()),
                engine_name: Some(engine_name.into()),
                engine_version: Some(engine_version.into()),
            }
        };
        let win_extensions = vulkano_win::required_extensions();
        let xr_extensions: Vec<_> = xr.instance.vulkan_instance_extensions(xr.system)
            .unwrap().split_ascii_whitespace()
            .map(|str_slice| CString::new(str_slice).unwrap()).collect();
        let raw_extensions = [/*CString::new("VK_EXT_debug_marker").unwrap()*/];
        let extensions = RawInstanceExtensions::new(raw_extensions.into_iter().cloned())
            .union(&(&win_extensions).into())
            .union(&RawInstanceExtensions::new(xr_extensions.into_iter()));
        let layers = [];
        let vk_instance: Arc<VkInstance> = VkInstance::new(Some(&app_info), extensions, layers.into_iter().cloned())
            .expect("Failed to create a Vulkan instance.");

        AmmoliteBuilder {
            vk_instance: Some(vk_instance),
            xr: Some(xr),
            uninitialized_window_mediums: Some(ArrayVec::new()),
            .. unsafe { self.coerce() }
        }
    }
}

impl<'a, MD: MediumData, A: OpenXrInitializedTrait> AmmoliteBuilder<'a, MD, A, VulkanInitialized::True, WindowsAdded::False, HmdsAdded::False> {
    pub fn add_medium_window(
        mut self,
        uninitialized_window_medium: UninitializedWindowMedium<MD>,
    ) -> Self {
        self.uninitialized_window_mediums.as_mut().unwrap().push(uninitialized_window_medium);
        self
    }

    fn register_medium_window(
        vk_device: &Arc<Device>,
        vk_queues: &ChosenQueues,
        window_mediums: &mut ArrayVec<[WindowMedium<MD>; 1]>,
        window: Arc<Surface<Window>>,
        data: MD,
    ) {
        let mut dimensions: [NonZeroU32; 2] = {
            let (width, height) = window.window().get_inner_size().unwrap().into();
            [
                NonZeroU32::new(width).expect("The width of the window must not be 0."),
                NonZeroU32::new(height).expect("The height of the window must not be 0."),
            ]
        };
        let vk_physical_device = vk_device.physical_device();
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
        println!("Supported formats: {:?}", &supported_formats);
        supported_formats.sort_by(swapchain_format_compare);
        let format = supported_formats[0];

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        let swapchain: VkSwapchain<Window> = vulkano::swapchain::Swapchain::new::<_, &Arc<Queue>>(
            vk_device.clone(),
            window.clone(),
            (&vk_queues.graphics).into(),
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

        let view_swapchain = ViewSwapchain::new(Box::new(swapchain) as Box<dyn Swapchain>);

        let window_medium = WindowMedium {
            data,
            window,
            swapchain: RefCell::new(view_swapchain),
            // swapchain: Box::new(swapchain) as Box<dyn Swapchain>,
        };

        window_mediums.push(window_medium);
    }

    pub fn finish_adding_mediums_window(
        mut self
    ) -> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::True, WindowsAdded::True, HmdsAdded::False> {
        let windows: Vec<(_, MD)> = self.uninitialized_window_mediums.take().unwrap().into_iter()
            .map(|uninitialized_window_medium| {
                let UninitializedWindowMedium {
                    events_loop,
                    window_builder,
                    window_handler,
                    mut data,
                } = uninitialized_window_medium;
                let events_loop = events_loop.as_ref().borrow();
                let window = window_builder.build_vk_surface(&events_loop, self.vk_instance.as_ref().unwrap().clone()).unwrap();

                if let Some(window_handler) = window_handler {
                    (window_handler)(&window, &mut data)
                }

                (window, data)
            })
            .collect();

        let openxr::vulkan::Requirements {
            min_api_version_supported: min_api_version,
            max_api_version_supported: max_api_version,
        } = self.xr.as_ref().unwrap().instance.graphics_requirements::<openxr::Vulkan>(self.xr.as_ref().unwrap().system).unwrap();
        let min_api_version = into_raw_u32_xr(min_api_version);
        let max_api_version = into_raw_u32_xr(max_api_version);
        // TODO: Better device selection & SLI support
        let physical_device = PhysicalDevice::enumerate(self.vk_instance.as_ref().unwrap())
            .filter(|physical_device| {
                println!("physical_device: {:?}", physical_device);
                let api_version = physical_device.api_version().into_vulkan_version();

                api_version >= min_api_version && api_version <= max_api_version
            })
            .next().expect("No physical device available.");


        struct ChosenQueueFamilies<'a> {
            graphics: QueueFamily<'a>,
            transfer: QueueFamily<'a>,
            compute: QueueFamily<'a>,
        }

        fn choose_queue_families<'a, 'b, MD>(windows: &'b [(Arc<Surface<Window>>, MD)], physical_device: PhysicalDevice<'a>) -> ChosenQueueFamilies<'a> {
            fn allow_graphics<'b, MD>(windows: &'b [(Arc<Surface<Window>>, MD)]) -> impl for<'a, 'c> FnMut(&'a QueueFamily<'c>) -> bool + 'b {
                move |queue_family: &QueueFamily| {
                    queue_family.supports_graphics()
                        && windows.iter().all(|(window, _)| window.is_supported(queue_family.clone()).unwrap_or(false))
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

            fn custom_min<'a, T>(mut comparator: impl for<'b> FnMut(&'b T, &'b T) -> Ordering, a: &'a T, b: &'a T) -> &'a T {
                match comparator(a, b) {
                    Ordering::Greater => b,
                    _ => a,
                }
            }

            fn consider_choice<T: Copy>(mut predicate: impl for<'b> FnMut(&'b T) -> bool, comparator: impl for<'b> FnMut(&'b T, &'b T) -> Ordering, a: Option<T>, b: T) -> Option<T> {
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
            let xr_extensions: Vec<_> = self.xr.as_ref().unwrap().instance.vulkan_device_extensions(self.xr.as_ref().unwrap().system)
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

        for (window, data) in windows.into_iter() {
            Self::register_medium_window(
                &vk_device,
                &vk_queues,
                &mut self.window_mediums,
                window,
                data,
            );
        }

        AmmoliteBuilder {
            vk_device: Some(vk_device),
            vk_queues: Some(vk_queues),
            .. unsafe { self.coerce() }
        }
    }
}

impl<'a, MD: MediumData> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::True, WindowsAdded::True, HmdsAdded::False> {
    pub fn add_medium_stereo_hmd(
        mut self,
        uninitialized_stereo_hmd_medium: UninitializedStereoHmdMedium<MD>,
    ) -> Self {
        let UninitializedStereoHmdMedium {
            instance_handler,
            mut data,
        } = uninitialized_stereo_hmd_medium;

        let XrContext {
            instance: xr_instance,
            system: xr_system,
            mut stereo_hmd_mediums,
        } = self.xr.take().unwrap();

        if !xr_instance.enumerate_view_configurations(xr_system).unwrap()
                       .contains(&openxr::ViewConfigurationType::PRIMARY_STEREO) {
            panic!("No HMD Stereo View available.");
        }

        let view_config_views = xr_instance
            .enumerate_view_configuration_views(xr_system, openxr::ViewConfigurationType::PRIMARY_STEREO)
            .unwrap();

        fn setup_openxr_session(
            xr_instance: &Arc<XrInstance>,
            vk_instance: &Arc<VkInstance>,
            xr_system: openxr::SystemId,
            vk_device: &Arc<Device>,
            vk_queue: &Arc<Queue>,
        ) -> (XrVkSession, XrFrameWaiter, XrVkFrameStream) {
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

        let (xr_session, xr_frame_waiter, xr_frame_stream) = setup_openxr_session(
            &xr_instance, self.vk_instance.as_ref().unwrap(), xr_system,
            self.vk_device.as_ref().unwrap(), &self.vk_queues.as_ref().unwrap().graphics,
        );

        let view_swapchains: ArrayVec<[RefCell<ViewSwapchain>; 2]> = {
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
                    // R8G8B8A8Srgb,
                    B8G8R8A8Srgb,
                    ImageUsage::all(), // FIXME
                    NonZeroU32::new(view.recommended_swapchain_sample_count).unwrap(),
                );

                swapchains.push(RefCell::new(ViewSwapchain::new(Box::new(swapchain) as Box<dyn Swapchain>)));
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

        if let Some(instance_handler) = instance_handler {
            (instance_handler)(&xr_instance, &xr_session, &mut data);
        }

        let xr_medium = XrMedium {
            data,
            xr_instance: xr_instance.clone(),
            xr_session: xr_session,
            xr_reference_space_stage,
            xr_frame_waiter,
            xr_frame_stream: RefCell::new(xr_frame_stream),
            swapchains: view_swapchains,
            frame_state: None,
            frame_views: None,
        };

        stereo_hmd_mediums.push(xr_medium);

        AmmoliteBuilder {
            xr: Some(XrContext {
                instance: xr_instance,
                system: xr_system,
                stereo_hmd_mediums,
            }),
            .. self
        }
    }

    pub fn finish_adding_mediums_stereo_hmd(
        mut self
    ) -> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::True, WindowsAdded::True, HmdsAdded::True> {
        unsafe { self.coerce() }
    }
}

impl<'a, MD: MediumData> AmmoliteBuilder<'a, MD, OpenXrInitialized::True, VulkanInitialized::True, WindowsAdded::True, HmdsAdded::True> {
    pub fn build(self) -> Ammolite<MD> {
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

        let medium_count = stereo_hmd_mediums.iter()
            .flat_map(|medium| medium.swapchains().into_iter())
            .chain(window_mediums.iter()
                   .flat_map(|medium| medium.swapchains().into_iter()))
            .count();

        if medium_count <= 0 {
            panic!("No mediums were specified during the creation of the Ammolite instance.");
        }

        let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(vk_device.clone(), vk_queues.graphics.family()).unwrap();
        let (init_command_buffer_builder, helper_resources) = HelperResources::new(
            &vk_device,
            vk_queues.families(),
        ).unwrap()
            .initialize_resource(&vk_device, vk_queues.graphics.family(), init_command_buffer_builder).unwrap();
        let pipeline_cache = {
            let view_swapchains = stereo_hmd_mediums.iter()
                .flat_map(|medium| medium.swapchains().into_iter())
                .chain(window_mediums.iter()
                       .flat_map(|medium| medium.swapchains().into_iter()))
                .collect::<Vec<_>>();
            GraphicsPipelineSetCache::create(vk_device.clone(), &view_swapchains, helper_resources.clone(), vk_queues.graphics.family())
        };
        let (init_command_buffer_builder, pipeline_cache) = pipeline_cache
            .initialize_resource(&vk_device, vk_queues.graphics.family(), init_command_buffer_builder).unwrap();
        let init_command_buffer = init_command_buffer_builder.build().unwrap();

        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
        // that, we store the submission of the previous frame here.
        let synchronization: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(vk_device.clone())
            .then_execute(vk_queues.graphics.clone(), init_command_buffer).unwrap()
            // .then_signal_fence()
            // .then_execute_same_queue(init_unsafe_command_buffer).unwrap()
            .then_signal_fence_and_flush().unwrap());

        Ammolite {
            vk_instance,
            xr: XrContext {
                instance: xr_instance,
                system: xr_system,
                stereo_hmd_mediums,
            },
            device: vk_device.clone(),
            vk_queues,
            pipeline_cache,
            helper_resources,
            window_mediums,
            // view_swapchains,
            synchronization: Some(synchronization),
            buffer_pool_uniform_instance: CpuBufferPool::uniform_buffer(vk_device),
        }
    }
}

pub struct CameraTransforms {
    pub position: Vec3,
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}

impl Default for CameraTransforms {
    fn default() -> Self {
        Self {
            position: Vec3::zero(),
            view_matrix: Mat4::identity(),
            projection_matrix: Mat4::identity(),
        }
    }
}

pub struct Ammolite<MD: MediumData> {
    /// The Vulkan runtime implementation
    pub vk_instance: Arc<VkInstance>,
    pub xr: XrContext<MD>,
    pub device: Arc<Device>,
    pub vk_queues: ChosenQueues,
    pub pipeline_cache: GraphicsPipelineSetCache,
    pub helper_resources: HelperResources,
    // pub window: Arc<Surface<Window>>,
    // pub window_events_loop: Rc<RefCell<EventsLoop>>,
    // pub window_dimensions: [NonZeroU32; 2],
    pub window_mediums: ArrayVec<[WindowMedium<MD>; 1]>,
    // pub view_swapchains: Arc<ViewSwapchains>,
    pub synchronization: Option<Box<dyn GpuFuture>>,
    // TODO Consider moving to SharedGltfGraphicsPipelineResources
    pub buffer_pool_uniform_instance: CpuBufferPool<InstanceUBO>,
}

impl<MD: MediumData> Ammolite<MD> {
    pub fn builder<'a>(
        application_name: &'a str,
        application_version: (u16, u16, u16),
    ) -> AmmoliteBuilder<'a, MD, OpenXrInitialized::False, VulkanInitialized::False, WindowsAdded::False, HmdsAdded::False> {
        AmmoliteBuilder::new(application_name, application_version)
    }

    /// returns `true`, to indicate to quit the application
    pub fn handle_events(&mut self, delta_time: &Duration) -> bool {
        let mut quit = false;

        'outer_loop:
        for medium in Self::mediums_mut(&mut self.xr.stereo_hmd_mediums,
                                        &mut self.window_mediums) {
            for command in medium.data_mut().handle_events(delta_time) {
                // println!("Handling events command: {:?}", command);

                match command {
                    HandleEventsCommand::Quit => {
                        quit = true;
                        break 'outer_loop;
                    },
                    HandleEventsCommand::MediumSpecific(command) => {
                        medium.handle_events_command(command);
                    },
                    HandleEventsCommand::RecreateSwapchain(swapchain_index) => {
                        let mut view_swapchain = medium.swapchains()[swapchain_index].borrow_mut();
                        view_swapchain.framebuffers = None;
                        view_swapchain.recreate = true;
                    }
                }
            }
        }

        quit
    }

    pub fn mediums<'a>(stereo_hmd_mediums: &'a ArrayVec<[XrMedium<MD>; 1]>, window_mediums: &'a ArrayVec<[WindowMedium<MD>; 1]>) -> impl Iterator<Item=&'a (dyn Medium<MD> + 'a)> {
        stereo_hmd_mediums.iter().map(|m| m as &dyn Medium<MD>)
            .chain(window_mediums.iter().map(|m| m as &dyn Medium<MD>))
    }

    // pub fn mediums<'a>(&'a self) -> impl Iterator<Item=&dyn Medium<MD>> {
    //     self.xr.stereo_hmd_mediums.iter().map(|m| m as &dyn Medium<MD>)
    //         .chain(self.window_mediums.iter().map(|m| m as &dyn Medium<MD>))
    // }

    pub fn mediums_mut<'a>(stereo_hmd_mediums: &'a mut ArrayVec<[XrMedium<MD>; 1]>, window_mediums: &'a mut ArrayVec<[WindowMedium<MD>; 1]>) -> impl Iterator<Item=&'a mut (dyn Medium<MD> + 'a)> {
        stereo_hmd_mediums.iter_mut().map(|m| m as &mut dyn Medium<MD>)
            .chain(window_mediums.iter_mut().map(|m| m as &mut dyn Medium<MD>))
    }

    // pub fn mediums_mut(&mut self) -> impl Iterator<Item=&mut dyn Medium<MD>> {
    //     self.xr.stereo_hmd_mediums.iter_mut().map(|m| m as &mut dyn Medium<MD>)
    //         .chain(self.window_mediums.iter_mut().map(|m| m as &mut dyn Medium<MD>))
    // }
    
    pub fn view_swapchains<'a>(stereo_hmd_mediums: &'a ArrayVec<[XrMedium<MD>; 1]>, window_mediums: &'a ArrayVec<[WindowMedium<MD>; 1]>) -> impl Iterator<Item=&'a RefCell<ViewSwapchain>> {
        Self::mediums(stereo_hmd_mediums, window_mediums).flat_map(|m| m.swapchains().into_iter().collect::<Vec<_>>())
    }

    // pub fn view_swapchains(&self) -> impl Iterator<Item=&RefCell<ViewSwapchain>> {
    //     self.mediums().flat_map(|m| m.swapchains().into_iter().collect::<Vec<_>>())
    // }

    // pub fn view_swapchains_mut(&mut self) -> impl Iterator<Item=&mut ViewSwapchain> {
    //     self.mediums_mut().flat_map(|m| m.swapchains_mut().into_iter().collect::<Vec<_>>())
    // }

    pub fn load_model<S: AsRef<Path>>(&mut self, path: S) -> Model {
        let init_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.vk_queues.graphics.family()).unwrap();
        let (init_command_buffer_builder, model) = {
            Model::import(
                &self.device,
                self.vk_queues.families(),
                &self.pipeline_cache,
                &self.helper_resources,
                path,
            ).unwrap().initialize_resource(
                &self.device,
                self.vk_queues.graphics.family().clone(),
                init_command_buffer_builder
            ).unwrap()
        };
        let init_command_buffer = init_command_buffer_builder.build().unwrap();

        self.synchronization = Some(Box::new(self.synchronization.take().unwrap()
            .then_execute(self.vk_queues.graphics.clone(), init_command_buffer).unwrap()
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

        let world_space_models = model_provider();
        let view_swapchains_len = Self::view_swapchains(&mut self.xr.stereo_hmd_mediums,
                                                       &mut self.window_mediums).count();

        // for stereo_hmd_medium in self.xr.stereo_hmd_mediums.iter_mut() {
        for medium in Self::mediums_mut(&mut self.xr.stereo_hmd_mediums,
                                        &mut self.window_mediums) {
            if let Some(views) = medium.wait_for_frame() {
                // let view_swapchains = self.view_swapchains().collect::<Vec<_>>();

                for (view_swapchain_index, view_swapchain) in medium.swapchains().iter().enumerate() {
                // for (view_swapchain_index, view_swapchain) in view_swapchains.iter().enumerate() {
                    let view = &views[view_swapchain_index];
                    let mut view_swapchain = view_swapchain.borrow_mut();

                    // A loop is used as a way to reset the rendering using `continue` if something goes wrong,
                    // there is a `break` statement at the end.
                    loop {
                        if view_swapchain.recreate {
                            let dimensions = medium.get_dimensions();

                            println!("Resizing to: [{}; {}]", dimensions[0], dimensions[1]);

                            match view_swapchain.swapchain.recreate_with_dimension(dimensions) {
                                Ok(()) => (),
                                // This error tends to happen when the user is manually resizing the window.
                                // Simply restarting the loop is the easiest way to fix this issue.
                                Err(SwapchainCreationError::UnsupportedDimensions) => {
                                    println!("Unsupported dimensions: [{}; {}]", dimensions[0], dimensions[1]);
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
                                .reconstruct_dimensions_dependent_images(view_swapchain_index, &view_swapchain)
                                .expect("Could not reconstruct dimension dependent resources.");

                            view_swapchain.framebuffers = Some(
                                self.pipeline_cache.shared_resources.construct_swapchain_framebuffers(
                                    self.pipeline_cache.render_pass.clone(),
                                    view_swapchain_index,
                                    &view_swapchain,
                                )
                            );

                            for (_, pipeline) in self.pipeline_cache.pipeline_map.write().unwrap().iter_mut() {
                                macro_rules! per_pipeline {
                                    ($pipeline:expr) => {
                                        $pipeline.layout_dependent_resources
                                            .reconstruct_descriptor_sets(&self.pipeline_cache.shared_resources, view_swapchains_len, view_swapchain_index, &view_swapchain);
                                    }
                                }

                                per_pipeline!(&mut pipeline.opaque);
                                per_pipeline!(&mut pipeline.mask);
                                per_pipeline!(&mut pipeline.blend_preprocess);
                                per_pipeline!(&mut pipeline.blend_finalize);
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

                        let dimensions = view_swapchain.swapchain.dimensions();
                        let camera_transforms = medium.data().get_camera_transforms(
                            view_swapchain_index,
                            view,
                            dimensions
                        );
                        let scene_ubo = SceneUBO::new(
                            secs_elapsed,
                            Vec2([dimensions[0].get() as f32, dimensions[1].get() as f32]),
                            camera_transforms.position.clone(),
                            camera_transforms.view_matrix.clone(),
                            camera_transforms.projection_matrix.clone(),
                        );

                        let buffer_updates = AutoCommandBufferBuilder::primary_one_time_submit(
                            self.device.clone(),
                            self.vk_queues.graphics.family(),
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
                            .then_execute(self.vk_queues.graphics.clone(), buffer_updates).unwrap()));

                        let current_framebuffer: Arc<dyn FramebufferWithClearValues<_>> = view_swapchain.framebuffers
                            .as_ref().unwrap()[multilayer_image.index()].clone();

                        let draw_context = DrawContext {
                            device: &self.device.clone(),
                            queue_family: &self.vk_queues.graphics.family(),
                            pipeline_cache: &self.pipeline_cache,
                            dynamic: DynamicState {
                                line_width: None,
                                viewports: None,
                                scissors: None,
                            },
                            helper_resources: &self.helper_resources,
                            view_swapchain_index,
                            view_swapchain: &view_swapchain,
                            vk_queues: &self.vk_queues,
                            buffer_pool_uniform_instance: &self.buffer_pool_uniform_instance,
                        };

                        self.synchronization = Some(Self::render_instances(
                            draw_context,
                            self.synchronization.take().unwrap(),
                            current_framebuffer,
                            world_space_models,
                            view_swapchain_index,
                            &view_swapchain
                        ));

                        let result = self.synchronization.take().unwrap()
                            .then_signal_fence();
                            // The color output is now expected to contain our triangle. But in order to show it on
                            // the screen, we have to *present* the image by calling `present`.
                            // This function does not actually present the image immediately. Instead it submits a
                            // present command at the end of the queue. This means that it will only be presented once
                            // the GPU has finished executing the command buffer that draws the triangle.
                        let result = view_swapchain.swapchain.present(Box::new(result), self.vk_queues.graphics.clone(), multilayer_image.index())
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

                for view_swapchain in medium.swapchains().iter() {
                    view_swapchain.borrow_mut().swapchain.finish_rendering();
                }
            }

            medium.finalize_frame();
        }
    }

    fn render_instances<'a>(mut draw_context: DrawContext,
                            synchronization: Box<dyn GpuFuture>,
                            current_framebuffer: Arc<dyn FramebufferWithClearValues<Vec<ClearValue>>>,
                            world_space_models: &'a [WorldSpaceModel<'a>],
                            view_swapchain_index: usize,
                            view_swapchain: &'a ViewSwapchain) -> Box<dyn GpuFuture> {
        let clear_values = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            1.0.into(),
            // ClearValue::None,
            // ClearValue::None,
            [0.0, 0.0, 0.0, 0.0].into(),
            [1.0, 1.0, 1.0, 1.0].into(),
        ];
        // TODO: Recreate only when screen dimensions change
        draw_context.dynamic = DynamicState {
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

        // let draw_context = DrawContext {
        //     device: &self.device.clone(),
        //     queue_family: &self.vk_queues.graphics.family(),
        //     pipeline_cache: &self.pipeline_cache,
        //     dynamic: &dynamic_state,
        //     helper_resources: &self.helper_resources,
        //     view_swapchain_index,
        //     view_swapchain,
        // };

        let instances = world_space_models.iter()
            .map(|WorldSpaceModel { model, matrix }| {
                let instance_ubo = InstanceUBO::new(matrix.clone());
                let instance_buffer: Arc<dyn TypedBufferAccess<Content=InstanceUBO> + Send + Sync>
                    = Arc::new(draw_context.buffer_pool_uniform_instance.next(instance_ubo).unwrap());
                let used_layouts = model.get_used_pipelines_layouts(&draw_context.pipeline_cache);
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
            draw_context.device.clone(),
            draw_context.vk_queues.graphics.family(),
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
                    0, // TODO
                ).unwrap();
            }
        }

        let command_buffer = command_buffer.end_render_pass().unwrap();

        Box::new(synchronization
                 .then_signal_semaphore()
                 .then_execute_same_queue(command_buffer.build().unwrap()).unwrap())
    }
}
