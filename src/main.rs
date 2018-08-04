#[macro_use]
extern crate vulkano;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily, Features};
use vulkano::sync::GpuFuture;

fn main() {
    /*
     * Initialization
     */

    // TODO: Explore method arguments
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("Failed to create a Vulkan instance.");

    // TODO: Better device selection & SLI support
    let physical_device = PhysicalDevice::enumerate(&instance).next().expect("No physical device available.");

    // Queues are like CPU threads, queue families are groups of queues with certain capabilities.
    println!("Available queue families:");

    /*
     * Device Creation
     */

    for queue_family in physical_device.queue_families() {
        println!("\tFamily #{} -- queues: {}, supports graphics: {}, supports compute: {}, supports transfers: {}, supports sparse binding: {}", queue_family.id(), queue_family.queues_count(), queue_family.supports_graphics(), queue_family.supports_compute(), queue_family.supports_transfers(), queue_family.supports_sparse_binding());
    }

    let queue_family = physical_device.queue_families()
        .find(QueueFamily::supports_graphics)
        .expect("Couldn't find a graphical queue family.");

    // Create a device with a single queue
    let (device, mut queues) = Device::new(physical_device,
                                           &Features::none(),
                                           &DeviceExtensions::none(),
                                           [(queue_family, 0.5)].iter().cloned())
            .expect("Failed to create a Vulkan device.");
    let queue = queues.next().unwrap();

    /*
     * Buffer Creation
     *
     * Vulkano does not provide a generic Buffer struct which you could create with Buffer::new.
     * Instead, it provides several different structs that all represent buffers, each of these
     * structs being optimized for a certain kind of usage. For example, if you want to
     * continuously upload data you should use a CpuBufferPool, while on the other hand if you have
     * some data that you are never going to modify you should use an ImmutableBuffer.
     */

    let in_data = 0..64;
    let in_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), in_data)
        .expect("Failed to create buffer.");
    let out_data = (0..64).map(|_| 0);
    let out_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), out_data)
        .expect("Failed to create buffer.");

    /*
     * Example Operation
     */

    // In order to execute commands efficiently, we store them in a command buffer and send them
    // to the device altogether.
    // Note that we specify the queue family when creating a command buffer.
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .expect("Could not create a command buffer builder.")
        .copy_buffer(in_buffer.clone(), out_buffer.clone()).unwrap()
        .build().expect("Could not build a command buffer.");

    let future = command_buffer.execute(queue.clone()).unwrap();

    // Wait for the execution to complete
    future.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    for value in out_buffer.read().unwrap().iter() {
        print!("{} ", value);
    }

    print!("\n");

    println!("Hello, world!");
}
