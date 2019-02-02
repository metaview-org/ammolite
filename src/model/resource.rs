use std::iter;
use std::sync::Arc;
use std::fmt;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::image::traits::ImageAccess;
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
    Image {
        data: Arc<Vec<u8>>,
        device_image: Arc<dyn ImageAccess + Send + Sync>,
        texel_conversion: Option<Box<dyn for<'a> Fn(&'a [u8]) -> Box<ExactSizeIterator<Item=u8> + 'a>>>,
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

                // TODO: Set mip count to Log2
                //
                // let (width, height) = if let ImageDimensions::Dim2d { width, height, .. } = device_image.dimensions() {
                //     (width, height)
                // } else {
                //     panic!("Texture mipmap generation is not implemented for non-2d textures.");
                // };

                // for mip_level in 1..device_image.mipmap_levels() {
                //     let source_mip_level = mip_level - 1;
                //     let mip_dimensions = (width >> mip_level, height >> mip_level);
                //     let source_mip_dimensions = (width >> source_mip_level, height >> source_mip_level);

                //     // unsafe {
                //     //     let pool = Device::standard_command_pool(&device, queue_family);
                //     //     let mut mipmap_layout_transition_commands = UnsafeCommandBufferBuilder::new(&pool, Kind::primary(), Flags::OneTimeSubmit)?;
                //     //     let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();

                //     //     barrier.add_image_memory_barrier(
                //     //         &device_image,
                //     //         mip_level..(mip_level + 1),
                //     //         0..1,
                //     //         PipelineStages {
                //     //             transfer: true,
                //     //             .. PipelineStages::none()
                //     //         },
                //     //         AccessFlagBits::none(),
                //     //         PipelineStages {
                //     //             host: true,
                //     //             .. PipelineStages::none()
                //     //         },
                //     //         AccessFlagBits {
                //     //             transfer_write: true,
                //     //             .. AccessFlagBits::none()
                //     //         },
                //     //         false, //????
                //     //         None,
                //     //         ImageLayout::Undefined,
                //     //         ImageLayout::TransferDstOptimal,
                //     //     );
                //     //     mipmap_layout_transition_commands.pipeline_barrier(&barrier);
                //     //     command_buffer_builder = command_buffer_builder
                //     //         .execute_commands(mipmap_layout_transition_commands)?;
                //     // }

                //     command_buffer_builder = command_buffer_builder.blit_image(
                //         device_image.clone(),
                //         [0, 0, 0],
                //         [source_mip_dimensions.0 as i32, source_mip_dimensions.1 as i32, 1],
                //         0,
                //         source_mip_level,
                //         device_image.clone(),
                //         [0, 0, 0],
                //         [mip_dimensions.0 as i32, mip_dimensions.1 as i32, 1],
                //         0,
                //         mip_level,
                //         1,
                //         Filter::Linear,
                //     )?;
                // }

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
