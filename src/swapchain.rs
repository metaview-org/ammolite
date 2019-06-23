use std::fmt;
use std::sync::Arc;
use std::num::NonZeroU32;
use vulkano::image::ImageUsage;
use vulkano::device::{Device, DeviceOwned, Queue};
use vulkano::sync::{GpuFuture, NowFuture};
use vulkano::format::{Format, FormatDesc};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{self, PresentFuture, AcquireError, SwapchainAcquireFuture, SwapchainCreationError};
use vulkano::buffer::BufferAccess;
use vulkano::image::{
    sys::{UnsafeImage, UnsafeImageView, UnsafeImageViewCreationError},
    layout::{ImageLayout, ImageLayoutEnd, RequiredLayouts},
    ImageSubresourceLayoutError,
    SwapchainImage,
    ImageDimensions,
    ImageSubresourceRange,
    ImageViewType,
    ImageViewAccess,
    ImageAccess,
};
use vulkano::sync::AccessError;
use vulkano::OomError;
use openxr;
use crate::XrVkSession;

// pub enum SwapchainAcquireNextImageFuture<W> {
//     SwapchainAcquireFuture(SwapchainAcquireFuture<W>),
//     Now(NowFuture),
// }

pub trait Swapchain<W: 'static + Send + Sync>: DeviceOwned {
    // Cannot be used, because PresentFuture has a generic type argument of the previous future
    //type PresentFuture: GpuFuture;

    // FIXME: error handling
    fn images(&self) -> &[Arc<dyn SwapchainImage>];
    fn acquire_next_image_index(&mut self) -> Result<(usize, Box<dyn GpuFuture>), AcquireError>;
    fn acquire_next_image(&mut self) -> Result<(Arc<dyn SwapchainImage>, Box<dyn GpuFuture>), AcquireError> {
        let (index, future) = self.acquire_next_image_index()?;
        let image = self.images()[index].clone();
        Ok((image, future))
    }
    fn recreate_with_dimension(&mut self, dimensions: [NonZeroU32; 2]) -> Result<(), SwapchainCreationError>;
    fn format(&self) -> Format;

    // TODO: remove Box
    fn present(&self, sync: Box<dyn GpuFuture>, queue: Arc<Queue>, index: usize) -> Box<dyn GpuFuture>;
}

pub struct VkSwapchain<W: 'static> {
    vk_swapchain: Arc<swapchain::Swapchain<W>>,
    images: Vec<Arc<dyn SwapchainImage>>,
}

impl<W> From<(Arc<swapchain::Swapchain<W>>, Vec<Arc<dyn SwapchainImage>>)> for VkSwapchain<W> {
    fn from(other: (Arc<swapchain::Swapchain<W>>, Vec<Arc<dyn SwapchainImage>>)) -> Self {
        let (vk_swapchain, images) = other;
        return Self { vk_swapchain, images }
    }
}

impl<W: 'static + Send + Sync> Swapchain<W> for VkSwapchain<W> {
    fn acquire_next_image_index(&mut self) -> Result<(usize, Box<dyn GpuFuture>), AcquireError> {
        let (index, future) = swapchain::acquire_next_image(self.vk_swapchain.clone(), None)?;
        // Ok((index, SwapchainAcquireNextImageFuture::SwapchainAcquireFuture(future)))
        Ok((index, Box::new(future)))
    }

    fn images(&self) -> &[Arc<dyn SwapchainImage>] { &self.images[..] }

    fn recreate_with_dimension(&mut self, dimensions: [NonZeroU32; 2]) -> Result<(), SwapchainCreationError> {
        let (vk_swapchain, images) = self.vk_swapchain.recreate_with_dimension(dimensions)?;

        self.vk_swapchain = vk_swapchain;
        self.images = images;

        Ok(())
    }

    fn format(&self) -> Format {
        self.vk_swapchain.format()
    }

    fn present(&self, sync: Box<dyn GpuFuture>, queue: Arc<Queue>, index: usize) -> Box<dyn GpuFuture> {
        Box::new(sync.then_swapchain_present(queue, self.vk_swapchain.clone(), index))
    }
}

unsafe impl<W> DeviceOwned for VkSwapchain<W> {
    fn device(&self) -> &Arc<Device> {
        &self.vk_swapchain.device()
    }
}

pub struct XrSwapchain {
    vk_device: Arc<Device>,
    inner: openxr::Swapchain<openxr::Vulkan>,
    format: Format,
    images: Vec<Arc<dyn SwapchainImage>>,
}

impl XrSwapchain {
    pub fn new<F: FormatDesc + Clone>(
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
        let images = xr_swapchain.enumerate_images().unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, handle)| {
                let xr_swapchain_image = unsafe {
                    XrSwapchainImage::from_raw(
                        handle,
                        &vk_device,
                        index,
                        layers,
                        dimensions,
                        usage,
                        format.clone(),
                    )
                }.unwrap();

                Arc::new(xr_swapchain_image) as Arc<dyn SwapchainImage>
            })
            .collect();

        Self {
            vk_device,
            inner: xr_swapchain,
            format: format.format(),
            images,
        }
    }
}

impl Swapchain<()> for XrSwapchain {
    fn acquire_next_image_index(&mut self) -> Result<(usize, Box<dyn GpuFuture>), AcquireError> {
        // FIXME: Error handling
        let index = self.inner.acquire_image().unwrap();
        let future = vulkano::sync::now(self.vk_device.clone());
        // Ok((index as usize, SwapchainAcquireNextImageFuture::Now(future)))
        Ok((index as usize, Box::new(future)))
    }

    fn images(&self) -> &[Arc<dyn SwapchainImage>] { &self.images[..] }

    fn recreate_with_dimension(&mut self, dimensions: [NonZeroU32; 2]) -> Result<(), SwapchainCreationError> {
        // no-op
        Ok(())
    }

    fn format(&self) -> Format {
        self.format
    }

    fn present(&self, sync: Box<dyn GpuFuture>, queue: Arc<Queue>, index: usize) -> Box<dyn GpuFuture> {
        // FIXME
        sync
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

struct XrSwapchainImage {
    index: usize,
    image: UnsafeImage,
    view: UnsafeImageView,
}

impl XrSwapchainImage {
    const REQUIRED_LAYOUTS: RequiredLayouts =
        RequiredLayouts { global: Some(ImageLayoutEnd::PresentSrc), ..RequiredLayouts::none() };

    unsafe fn from_raw(handle: u64, device: &Arc<Device>, index: usize, layers: NonZeroU32, dimensions: [NonZeroU32; 2], usage: ImageUsage, format: impl FormatDesc) -> Result<Self, OomError> {
        let dims = if layers.get() == 1 {
            ImageDimensions::Dim2D { width: dimensions[0], height: dimensions[1] }
        } else {
            ImageDimensions::Dim2DArray {
                width: dimensions[0],
                height: dimensions[1],
                array_layers: layers
            }
        };

        let image = UnsafeImage::from_raw(
            device.clone(),
            handle,
            usage,
            format.format(),
            dims,
            crate::NONZERO_ONE,
            crate::NONZERO_ONE,
        );

        let view = match UnsafeImageView::new(
            &image,
            Some(ImageViewType::Dim2D),
            None,
            Default::default(),
            ImageSubresourceRange {
                array_layers: crate::NONZERO_ONE,
                array_layers_offset: 0,

                mipmap_levels: crate::NONZERO_ONE,
                mipmap_levels_offset: 0
            }
        ) {
            Ok(v) => v,
            Err(UnsafeImageViewCreationError::OomError(e)) => return Err(e),
            e => panic!("Could not create swapchain view: {:?}", e)
        };

        Ok(XrSwapchainImage { index, image, view })
    }
}

impl SwapchainImage for XrSwapchainImage {
    fn inner_dimensions(&self) -> [NonZeroU32; 2] {
        self.inner_image().dimensions().width_height()
    }

    fn inner_image(&self) -> &UnsafeImage {
        &self.image
    }

    fn index(&self) -> usize {
        self.index
    }
}

unsafe impl DeviceOwned for XrSwapchainImage {
    fn device(&self) -> &Arc<Device> { self.image.device() }
}

impl fmt::Debug for XrSwapchainImage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "XrSwapchainImage {{ index: {index}, view: {view:?} }}", index=self.index, view=self.view)
    }
}

// Basically copied from the source code of `vulkano::image::swapchain`
unsafe impl ImageAccess for XrSwapchainImage {
    fn inner(&self) -> &UnsafeImage { self.inner_image() }

    fn conflicts_buffer(&self, other: &BufferAccess) -> bool { false }

    fn conflicts_image(
        &self, subresource_range: ImageSubresourceRange, other: &dyn ImageAccess,
        other_subresource_range: ImageSubresourceRange
    ) -> bool {
        if ImageAccess::inner(self).key() == other.conflict_key() {
            subresource_range.overlaps_with(&other_subresource_range)
        } else {
            false
        }
    }

    fn conflict_key(&self) -> u64 { ImageAccess::inner(self).key() }

    fn current_layout(
        &self, _: ImageSubresourceRange
    ) -> Result<ImageLayout, ImageSubresourceLayoutError> {
        // TODO: Is this okay?
        Ok(ImageLayout::PresentSrc)
    }

    fn initiate_gpu_lock(
        &self, _: ImageSubresourceRange, _: bool, _: ImageLayout
    ) -> Result<(), AccessError> {
        // Swapchain image are only accessible after being acquired.
        // This is handled by the swapchain itself.
        Err(AccessError::SwapchainImageAcquireOnly)
    }

    unsafe fn increase_gpu_lock(&self, _: ImageSubresourceRange) {}

    unsafe fn decrease_gpu_lock(
        &self, _: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
    ) {
        // TODO: store that the image was initialized?
    }
}

// Basically copied from the source code of `vulkano::image::swapchain`
unsafe impl ImageViewAccess for XrSwapchainImage {
    fn parent(&self) -> &ImageAccess { self }

    fn inner(&self) -> &UnsafeImageView { &self.view }

    fn dimensions(&self) -> ImageDimensions { self.view.dimensions() }

    fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool { false }

    fn required_layouts(&self) -> &RequiredLayouts { &Self::REQUIRED_LAYOUTS }
}
