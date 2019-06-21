use std::sync::Arc;
use std::num::NonZeroU32;
use vulkano::image::ImageUsage;
use vulkano::device::{Device, DeviceOwned};
use vulkano::sync::NowFuture;
use vulkano::format::FormatDesc;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{self, Swapchain as VkSwapchain, SwapchainAcquireFuture};
use openxr;
use crate::XrVkSession;

pub enum SwapchainAcquireNextImageFuture<W> {
    SwapchainAcquireFuture(SwapchainAcquireFuture<W>),
    Now(NowFuture),
}

pub trait Swapchain<W>: DeviceOwned {
    // FIXME: error handling
    fn acquire_next_image(&mut self) -> (usize, SwapchainAcquireNextImageFuture<W>);

    // TODO: Image acquisition using `openxr::Swapchain::enumerate_images` and by converting them
    // to `vk::SwapchainImage`. This may not be possible without modifying this type's inner
    // structure, as it may currently only use the Vulkan Swapchain and nothing else.
}

impl<W> Swapchain<W> for Arc<VkSwapchain<W>> {
    fn acquire_next_image(&mut self) -> (usize, SwapchainAcquireNextImageFuture<W>) {
        let (index, future) = swapchain::acquire_next_image(self.clone(), None)
            .unwrap();
        (index, SwapchainAcquireNextImageFuture::SwapchainAcquireFuture(future))
    }
}

impl Swapchain<()> for XrSwapchain {
    fn acquire_next_image(&mut self) -> (usize, SwapchainAcquireNextImageFuture<()>) {
        let index = self.inner.acquire_image().unwrap();
        let future = vulkano::sync::now(self.vk_device.clone());
        (index as usize, SwapchainAcquireNextImageFuture::Now(future))
    }
}

struct XrSwapchain {
    vk_device: Arc<Device>,
    inner: openxr::Swapchain<openxr::Vulkan>,
}

impl XrSwapchain {
    pub fn new<F: FormatDesc>(
        vk_device: Arc<Device>,
        xr_session: Arc<XrVkSession>,
        dimensions: [NonZeroU32; 2],
        layers: NonZeroU32,
        format: F,
        usage: ImageUsage,
    ) -> Self {
        let create_info = openxr::SwapchainCreateInfo {
            create_flags: openxr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr_from_vk_usage(usage),
            format: format.format() as u32,
            sample_count: 1, // TODO customizability
            width: dimensions[0].get(),
            height: dimensions[1].get(),
            face_count: 1, // 6 for cubemaps, or 1
            array_size: layers.get(),
            mip_count: 1, // TODO customizability
        };
        let xr_swapchain = xr_session.create_swapchain(&create_info).unwrap();

        Self {
            vk_device,
            inner: xr_swapchain,
        }
    }
}

unsafe impl DeviceOwned for XrSwapchain {
    fn device(&self) -> &Arc<Device> {
        &self.vk_device
    }
}

fn xr_from_vk_usage(usage: ImageUsage) -> openxr::SwapchainUsageFlags {
    use openxr::SwapchainUsageFlags as SUF;

    let mut result = SUF::EMPTY;

    if usage.color_attachment {
        result |= SUF::COLOR_ATTACHMENT;
    }

    if usage.depth_stencil_attachment {
        result |= SUF::DEPTH_STENCIL_ATTACHMENT;
    }

    if usage.storage {
        result |= SUF::UNORDERED_ACCESS; // check validity
    }

    if usage.transfer_source {
        result |= SUF::TRANSFER_SRC;
    }

    if usage.transfer_destination {
        result |= SUF::TRANSFER_DST;
    }

    if usage.sampled {
        result |= SUF::SAMPLED;
    }

    //result |= SUF::MUTABLE_FORMAT;

    result
}
