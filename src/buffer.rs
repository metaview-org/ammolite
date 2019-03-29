use std::sync::Arc;
use vulkano::buffer::{TypedBufferAccess, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::device::Device;
use vulkano::instance::QueueFamily;

pub struct StagedBuffer<T: Sized + Send + Sync + 'static> {
    staging_buffer: Arc<TypedBufferAccess<Content=T> + Send + Sync>,
    device_buffer: Arc<TypedBufferAccess<Content=T> + Send + Sync>,
}

impl<T: Send + Sync> StagedBuffer<T> {
    pub fn from_data(device: &Arc<Device>, queue_family: QueueFamily, usage: BufferUsage, data: T)
            -> StagedBuffer<T>
            where CpuAccessibleBuffer<T>: TypedBufferAccess<Content=T>,
                  DeviceLocalBuffer<T>: TypedBufferAccess<Content=T>, {
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

        StagedBuffer {
            staging_buffer,
            device_buffer,
        }
    }

    pub fn staging_buffer(&self) -> &Arc<TypedBufferAccess<Content=T>> {
        &self.staging_buffer
    }

    pub fn device_buffer(&self) -> &Arc<TypedBufferAccess<Content=T>> {
        &self.device_buffer
    }
}
