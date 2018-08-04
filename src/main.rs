#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate image;

use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily, Features};
use vulkano::sync::GpuFuture;
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use image::{ImageBuffer, Rgba};

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl_vertex!(Vertex, position);

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[path = "src/shaders/vertex.vert"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "src/shaders/fragment.frag"]
    struct Dummy;
}

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

    print!("\n");

    let vertices = vec![
        Vertex { position: [-0.5, -0.5 ] },
        Vertex { position: [ 0.0,  0.5 ] },
        Vertex { position: [ 0.5, -0.25] },
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(),
                                                       BufferUsage::all(),
                                                       vertices.into_iter()).unwrap();

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // A special GPU mode highly-optimized for rendering
    let render_pass = Arc::new(single_pass_renderpass! { device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store, // for temporary images, use DontCare
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    }.unwrap());

    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
                               .add(image.clone()).unwrap()
                               .build().unwrap());

    let vs = vs::Shader::load(device.clone()).expect("Failed to create shader module.");
    let fs = fs::Shader::load(device.clone()).expect("Failed to create shader module.");

    let pipeline = Arc::new(GraphicsPipeline::start()
        // Specifies the Vertex type
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        // Configures the builder so that we use one viewport, and that the state of this viewport
        // is dynamic. This makes it possible to change the viewport for each draw command. If the
        // viewport state wasn't dynamic, then we would have to create a new pipeline object if we
        // wanted to draw to another image of a different size.
        //
        // Note: If you configure multiple viewports, you can use geometry shaders to choose which
        // viewport the shape is going to be drawn to. This topic isn't covered here.
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                             (0 .. 1024 * 1024 * 4).map(|_| 0u8))
        .expect("failed to create buffer");

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        }]),
        .. DynamicState::none()
    };

    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
        .draw(pipeline.clone(), dynamic_state, vertex_buffer.clone(), (), ()).unwrap()
        .end_render_pass().unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();

    let future = command_buffer.execute(queue.clone()).unwrap();

    future.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();
}
