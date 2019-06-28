use std::iter;
use std::sync::Arc;
use std::fmt;
use std::marker::PhantomData;
use core::num::NonZeroU32;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::sampler::Filter;
use vulkano::image::traits::ImageViewAccess;
use vulkano::image::traits::ImageAccess;
use vulkano::image::view::ImageView;
use vulkano::image::sync::locker;
use vulkano::image::SyncImage;
use vulkano::image::ImageViewType;
use vulkano::image::ImageDimensions;
use vulkano::image::Swizzle;
use vulkano::image::ImageSubresourceRange;
use vulkano::image::layout::RequiredLayouts;
use vulkano::image::layout::typesafety;
use failure::Error;
use crate::NodeUBO;
use crate::MaterialUBO;
use crate::iter::ForcedExactSizeIterator;

pub enum InitializationTask {
    Buffer {
        data: Vec<u8>,
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    ZeroBuffer {
        len: usize,
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    // TODO: Investigate whether it should be merged with `ImageWithMipmaps`
    Image {
        data: Arc<Vec<u8>>,
        device_image: Arc<dyn ImageViewAccess + Send + Sync>,
        texel_conversion: Option<Box<dyn for<'a> Fn(&'a [u8]) -> Box<dyn ExactSizeIterator<Item=u8> + 'a>>>,
    },
    ImageWithMipmaps {
        data: Arc<Vec<u8>>,
        device_image: Arc<SyncImage<locker::MatrixImageResourceLocker>>,
        texel_conversion: Option<Box<dyn for<'a> Fn(&'a [u8]) -> Box<dyn ExactSizeIterator<Item=u8> + 'a>>>,
    },
    NodeDescriptorSet {
        data: NodeUBO,
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=NodeUBO> + Send + Sync>,
    },
    MaterialDescriptorSet {
        data: MaterialUBO,
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=MaterialUBO> + Send + Sync>,
    },
}

impl fmt::Debug for InitializationTask {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InitializationTask::Buffer { .. } => {
                write!(f, "buffer")
            },
            InitializationTask::ZeroBuffer { .. } => {
                write!(f, "zero buffer")
            },
            InitializationTask::Image { .. } => {
                write!(f, "image")
            },
            InitializationTask::ImageWithMipmaps { .. } => {
                write!(f, "image with mipmaps")
            },
            InitializationTask::NodeDescriptorSet { .. } => {
                write!(f, "node descriptor set")
            },
            InitializationTask::MaterialDescriptorSet { .. } => {
                write!(f, "material descriptor set")
            },
        }
    }
}

impl InitializationTask {
    fn initialize(self, device: &Arc<Device>, _queue_family: QueueFamily, command_buffer_builder: AutoCommandBufferBuilder) -> Result<AutoCommandBufferBuilder, Error> {
        match self {
            InitializationTask::Buffer { data, initialization_buffer, .. } => {
                let staging_buffer: Arc<CpuAccessibleBuffer<[u8]>> = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    data.iter().cloned(), // FIXME: Single memcpy call, do not iterate
                )?;

                Ok(command_buffer_builder.copy_buffer(staging_buffer, initialization_buffer)?)
            },
            InitializationTask::ZeroBuffer { len, initialization_buffer, .. } => {
                let staging_buffer: Arc<CpuAccessibleBuffer<[u8]>> = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    ForcedExactSizeIterator::new(iter::repeat(0u8).take(len), len),
                )?;

                Ok(command_buffer_builder.copy_buffer(staging_buffer, initialization_buffer)?)
            },
            InitializationTask::Image { data, device_image, texel_conversion, .. } => {
                let staging_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    if let Some(texel_conversion) = texel_conversion {
                        (texel_conversion)(&data[..])
                    } else {
                        Box::new(data.iter().cloned()) // FIXME: Single memcpy call, do not iterate
                    },
                )?;

                let command_buffer_builder = command_buffer_builder
                    .copy_buffer_to_image(staging_buffer, device_image.clone())?;

                Ok(command_buffer_builder)
            },
            // InitializationTask::ImageWithMipmaps { .. } => {
            //     unimplemented!();
            // },
            InitializationTask::ImageWithMipmaps { data, device_image, texel_conversion, .. } => {
                // let mut required_layouts = RequiredLayouts::general();
                let mut required_layouts = RequiredLayouts::none();
                required_layouts.infer_mut(device_image.usage());
                required_layouts.global = Some(typesafety::ImageLayoutEnd::ShaderReadOnlyOptimal);

                let mut source_layer = Arc::new(ImageView::new(
                    device_image.clone(),
                    Some(ImageViewType::Dim2D),
                    Some(device_image.format()),
                    Swizzle::identity(),
                    Some(ImageSubresourceRange {
                        array_layers: NonZeroU32::new(1).unwrap(),
                        array_layers_offset: 0,
                        mipmap_levels: NonZeroU32::new(1).unwrap(),
                        mipmap_levels_offset: 0,
                    }),
                    required_layouts.clone(),
                )?);
                let staging_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    if let Some(texel_conversion) = texel_conversion {
                        (texel_conversion)(&data[..])
                    } else {
                        Box::new(data.iter().cloned()) // FIXME: Single memcpy call, do not iterate
                    },
                )?;

                let mut command_buffer_builder = command_buffer_builder
                    .copy_buffer_to_image(staging_buffer, source_layer.clone())?;

                // TODO: Set mip count to Log2

                let (width, height) = if let ImageDimensions::Dim2D { width, height } = device_image.dimensions() {
                    (width.get(), height.get())
                } else {
                    panic!("Texture mipmap generation is not implemented for non-2d textures.");
                };

                for mip_level in 1..device_image.mipmap_levels().get() {
                    let destination_layer = Arc::new(ImageView::new(
                        device_image.clone(),
                        Some(ImageViewType::Dim2D),
                        Some(device_image.format()),
                        Swizzle::identity(),
                        Some(ImageSubresourceRange {
                            array_layers: NonZeroU32::new(1).unwrap(),
                            array_layers_offset: 0,
                            mipmap_levels: NonZeroU32::new(1).unwrap(),
                            mipmap_levels_offset: mip_level,
                        }),
                        required_layouts.clone(),
                    )?);
                    let source_mip_level = mip_level - 1;
                    let mip_dimensions = (width >> mip_level, height >> mip_level);
                    let source_mip_dimensions = (width >> source_mip_level, height >> source_mip_level);

                    command_buffer_builder = command_buffer_builder.blit_image(
                        source_layer.clone(),
                        [0, 0, 0],
                        [source_mip_dimensions.0 as i32, source_mip_dimensions.1 as i32, 1],
                        0,
                        source_mip_level,
                        destination_layer.clone(),
                        [0, 0, 0],
                        [mip_dimensions.0 as i32, mip_dimensions.1 as i32, 1],
                        0,
                        mip_level,
                        1,
                        Filter::Linear,
                    )?;
                    source_layer = destination_layer;
                }

                Ok(command_buffer_builder)
            },
            InitializationTask::NodeDescriptorSet { data, initialization_buffer, .. } => {
                let staging_buffer: Arc<CpuAccessibleBuffer<NodeUBO>> = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    data,
                )?;

                Ok(command_buffer_builder.copy_buffer(staging_buffer, initialization_buffer)?)
            },
            InitializationTask::MaterialDescriptorSet { data, initialization_buffer, .. } => {
                let staging_buffer: Arc<CpuAccessibleBuffer<MaterialUBO>> = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    data,
                )?;

                Ok(command_buffer_builder.copy_buffer(staging_buffer, initialization_buffer)?)
            },
        }
    }
}

pub trait UninitializedResource<T> {
    fn initialize_resource(
        self,
        device: &Arc<Device>,
        queue_family: QueueFamily,
        command_buffer_builder: AutoCommandBufferBuilder
    ) -> Result<(AutoCommandBufferBuilder, T), Error>;

    fn join<U, URU>(self, other: URU) -> JoinUninitializedResource<T, U, Self, URU>
            where Self: Sized,
                  URU: UninitializedResource<U> {
        JoinUninitializedResource {
            first: self,
            second: other,
            _marker: PhantomData,
        }
    }

    fn map<O, F>(self, map: F) -> MapUninitializedResource<T, O, Self, F>
            where Self: Sized,
                  F: FnOnce(T) -> O {
        MapUninitializedResource {
            input: self,
            map,
            _marker: PhantomData,
        }
    }
}

pub struct SimpleUninitializedResource<T> {
    output: T,
    tasks: Vec<InitializationTask>,
}

impl<T> SimpleUninitializedResource<T> {
    pub fn new(output: T, tasks: Vec<InitializationTask>) -> Self {
        Self {
            output,
            tasks,
        }
    }
}

impl<T> UninitializedResource<T> for SimpleUninitializedResource<T> {
    fn initialize_resource(self, device: &Arc<Device>, queue_family: QueueFamily, mut command_buffer_builder: AutoCommandBufferBuilder) -> Result<(AutoCommandBufferBuilder, T), Error> {
        for initialization_task in self.tasks.into_iter() {
            command_buffer_builder = initialization_task.initialize(device, queue_family, command_buffer_builder)?;
        }

        Ok((command_buffer_builder, self.output))
    }
}

pub struct JoinUninitializedResource<T, U, URT, URU> where URT: UninitializedResource<T>,
                                                           URU: UninitializedResource<U> {
    first: URT,
    second: URU,
    _marker: PhantomData<(T, U)>,
}

impl<T, U, URT, URU> UninitializedResource<(T, U)> for JoinUninitializedResource<T, U, URT, URU>
        where URT: UninitializedResource<T>,
              URU: UninitializedResource<U> {
    fn initialize_resource(self, device: &Arc<Device>, queue_family: QueueFamily, command_buffer_builder: AutoCommandBufferBuilder) -> Result<(AutoCommandBufferBuilder, (T, U)), Error> {
        let (command_buffer_builder, first_result) = self.first.initialize_resource(device, queue_family, command_buffer_builder)?;
        let (command_buffer_builder, second_result) = self.second.initialize_resource(device, queue_family, command_buffer_builder)?;

        Ok((command_buffer_builder, (first_result, second_result)))
    }
}

pub struct MapUninitializedResource<I, O, URI, F> where URI: UninitializedResource<I>,
                                                        F: FnOnce(I) -> O {
    input: URI,
    map: F,
    _marker: PhantomData<(I, O)>,
}

impl<I, O, URI, F> UninitializedResource<O> for MapUninitializedResource<I, O, URI, F>
        where URI: UninitializedResource<I>,
              F: FnOnce(I) -> O {
    fn initialize_resource(self, device: &Arc<Device>, queue_family: QueueFamily, command_buffer_builder: AutoCommandBufferBuilder) -> Result<(AutoCommandBufferBuilder, O), Error> {
        let (command_buffer_builder, result) = self.input.initialize_resource(device, queue_family, command_buffer_builder)?;
        let mapped_result = (self.map)(result);

        Ok((command_buffer_builder, mapped_result))
    }
}
