#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;
extern crate image;
extern crate winit;

use std::sync::Arc;
use std::mem;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily, Features};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::format::Format;
use vulkano::image::{Dimensions, StorageImage};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::swapchain::{self, PresentMode, SurfaceTransform, Swapchain, AcquireError, SwapchainCreationError};
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder};
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
    #[allow(dead_code)]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "src/shaders/fragment.frag"]
    #[allow(dead_code)]
    struct Dummy;
}

fn main() {
    /*
     * Initialization
     */

    // TODO: Explore method arguments
    let extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &extensions, None)
        .expect("Failed to create a Vulkan instance.");

    // TODO: Better device selection & SLI support
    let physical_device = PhysicalDevice::enumerate(&instance).next().expect("No physical device available.");

    // Queues are like CPU threads, queue families are groups of queues with certain capabilities.
    println!("Available queue families:");

    let mut events_loop = EventsLoop::new();
    let window = VkSurfaceBuild::build_vk_surface(WindowBuilder::new(), &events_loop, instance.clone()).unwrap();

    let mut dimensions = {
        let (width, height) = window.window().get_inner_size().unwrap();
        [width, height]
    };

    /*
     * Device Creation
     */

    for queue_family in physical_device.queue_families() {
        println!("\tFamily #{} -- queues: {}, supports graphics: {}, supports compute: {}, supports transfers: {}, supports sparse binding: {}", queue_family.id(), queue_family.queues_count(), queue_family.supports_graphics(), queue_family.supports_compute(), queue_family.supports_transfers(), queue_family.supports_sparse_binding());
    }

    let queue_family = physical_device.queue_families()
        .find(|&queue| {
            queue.supports_graphics() && window.is_supported(queue).unwrap_or(false)
        })
        .expect("Couldn't find a graphical queue family.");

    // Create a device with a single queue
    let (device, mut queues) = {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            .. DeviceExtensions::none()
        };

        Device::new(physical_device,
                    &Features::none(),
                    &device_extensions,
                    // A list of queues to use specified by an iterator of (QueueFamily, priority).
                    // In a real-life application, we would probably use at least a graphics queue
                    // and a transfers queue to handle data transfers in parallel. In this example
                    // we only use one queue.
                    [(queue_family, 0.5)].iter().cloned())
            .expect("Failed to create a Vulkan device.")
    };
    // println!("queues: {}", queues.len());
    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let capabilities = window.capabilities(physical_device)
            .expect("Failed to retrieve surface capabilities.");

        // Determines the behaviour of the alpha channel
        let alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
        dimensions = capabilities.current_extent.unwrap_or(dimensions);

        // Choosing the internal format that the images will have.
        let format = capabilities.supported_formats[0].0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(device.clone(), window.clone(), capabilities.min_image_count, format,
                       dimensions, 1, capabilities.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };

    /*
     * Buffer Creation
     *
     * Vulkano does not provide a generic Buffer struct which you could create with Buffer::new.
     * Instead, it provides several different structs that all represent buffers, each of these
     * structs being optimized for a certain kind of usage. For example, if you want to
     * continuously upload data you should use a CpuBufferPool, while on the other hand if you have
     * some data that you are never going to modify you should use an ImmutableBuffer.
     */

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
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    }.unwrap());

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

    let mut framebuffers: Option<Vec<Arc<Framebuffer<_, _>>>> = None;

    // We need to keep track of whether the swapchain is invalid for the current window,
    // for example when the window is resized.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone()));

    loop {
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            // Get the new dimensions for the viewport/framebuffers.
            dimensions = {
                let (new_width, new_height) = window.window().get_inner_size().unwrap();
                [new_width, new_height]
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);

            framebuffers = None;

            recreate_swapchain = false;
        }
        //
        // Because framebuffers contains an Arc on the old swapchain, we need to
        // recreate framebuffers as well.
        if framebuffers.is_none() {
            let new_framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            }).collect::<Vec<_>>());
            mem::replace(&mut framebuffers, new_framebuffers);
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok((image_num, acquire_future)) => {
                (image_num, acquire_future)
            },
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter builds the list of values to clear the attachments with. The API
            // is similar to the list of attachments when building the framebuffers, except that
            // only the attachments that use `load: Clear` appear in the list.
            .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(), false,
                               vec![[0.0, 0.0, 1.0, 1.0].into()])
            .unwrap()
            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw(pipeline.clone(),
                  DynamicState {
                      line_width: None,
                      // TODO: Find a way to do this without having to dynamically allocate a Vec every frame.
                      viewports: Some(vec![Viewport {
                          origin: [0.0, 0.0],
                          dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                          depth_range: 0.0 .. 1.0,
                      }]),
                      scissors: None,
                  },
                  vertex_buffer.clone(), (), ())
            .unwrap()
            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .end_render_pass()
            .unwrap()
            // Finish building the command buffer by calling `build`.
            .build().unwrap();

        let result = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).expect("DERP")

            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        previous_frame_end = match result {
            Ok(future) => Box::new(future) as Box<GpuFuture>,
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>
            },
            Err(err) => panic!("{:?}", err),
        };

        if recreate_swapchain {
            continue;
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        //
        // Unfortunately the Vulkan API doesn't provide any way to not wait or to detect when a
        // wait would happen. Blocking may be the desired behavior, but if you don't want to
        // block you should spawn a separate thread dedicated to submissions.

        // Handling the window events in order to close the program when the user wants to close
        // it.
        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::Resized(_, _), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}
