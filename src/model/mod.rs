use std::sync::Arc;
use std::path::Path;
use std::ops::Deref;
use std::mem;
use std::marker::PhantomData;
use vulkano;
use vulkano::command_buffer::{DynamicState, AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::device::Device;
use vulkano::instance::QueueFamily;
use vulkano::format::ClearValue;
use vulkano::framebuffer::RenderPassDesc;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassDescClearValues;
use vulkano::buffer::BufferSlice;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::descriptor::descriptor_set::collection::DescriptorSetsCollection;
use gltf::{self, Document, Gltf};
use gltf::mesh::util::ReadIndices;
use gltf::mesh::{Mesh, Semantic};
use gltf::accessor::Accessor as GltfAccessor;
use gltf::Node;
use ::Position;
use ::PipelineImpl;

pub struct Model {
    document: Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
    device_buffers: Vec<Arc<CpuAccessibleBuffer<[u8]>>>,
}

impl Model {
    pub fn import<S: AsRef<Path>>(device: Arc<Device>, path: S) -> Result<Model, gltf::Error> {
        let (document, buffers, images) = gltf::import(path)?;
        // TODO: setup buffer staging
        let device_buffers: Vec<Arc<CpuAccessibleBuffer<[u8]>>> = buffers.iter().map(|buffer| {
            let device_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (**buffer).into_iter().cloned());
            device_buffer.unwrap()
        }).collect();

        // let main_descriptor_set = Arc::new(
        //     PersistentDescriptorSet::start(main_pipeline.clone(), 0)
        //         .add_buffer(main_ubo_device_buffer.clone()).unwrap()
        //         .add_sampled_image(screen_image.clone(), screen_sampler.clone()).unwrap()
        //         .add_sampled_image(texture_device_image.clone(), texture_device_sampler.clone()).unwrap()
        //         .build().unwrap()
        // );

        Ok(Model {
            document,
            buffers,
            images,
            device_buffers,
        })
    }

    pub fn draw_scene<S, F, C, RPD>(&self, device: Arc<Device>, queue_family: QueueFamily,
                      framebuffer: Arc<F>, clear_values: C, pipeline: PipelineImpl<RPD>,
                      dynamic: &DynamicState, sets: S, scene_index: usize) -> Result<AutoCommandBuffer, ()>
            where S: DescriptorSetsCollection + Clone,
                  F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if scene_index >= self.document.scenes().len() {
            return Err(());
        }

        let scene = self.document.scenes().nth(scene_index).unwrap();
        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device, queue_family)
            .unwrap()
            .begin_render_pass(framebuffer, false, clear_values).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, pipeline.clone(), dynamic, sets.clone());
        }

        command_buffer = command_buffer.end_render_pass().unwrap();

        Ok(command_buffer.build().unwrap())
    }

    pub fn draw_main_scene<S, F, C, RPD>(&self, device: Arc<Device>, queue_family: QueueFamily,
                      framebuffer: Arc<F>, clear_values: C, pipeline: PipelineImpl<RPD>,
                      dynamic: &DynamicState, sets: S) -> Result<AutoCommandBuffer, ()>
            where S: DescriptorSetsCollection + Clone,
                  F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(device, queue_family, framebuffer, clear_values, pipeline, dynamic, sets, main_scene_index)
        } else {
            Err(())
        }
    }

    pub fn draw_node<'a, S, RPD>(&self, node: Node<'a>, mut command_buffer: AutoCommandBufferBuilder, pipeline: PipelineImpl<RPD>, dynamic: &DynamicState, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection + Clone,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {

        if let Some(mesh) = node.mesh() {
            command_buffer = self.draw_mesh(mesh, command_buffer, pipeline.clone(), dynamic, sets.clone());
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, pipeline.clone(), dynamic, sets.clone());
        }

        command_buffer
    }

    pub fn draw_mesh<'a, S, RPD>(&self, mesh: Mesh<'a>, mut command_buffer: AutoCommandBufferBuilder, pipeline: PipelineImpl<RPD>, dynamic: &DynamicState, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection + Clone,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        for primitive in mesh.primitives() {
            let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
            let indices_accessor = primitive.indices().unwrap();

            let vertex_slice: BufferSlice<[Position], Arc<CpuAccessibleBuffer<[u8]>>> = {
                let buffer_view = positions_accessor.view();
                let buffer_index = buffer_view.buffer().index();
                let buffer_offset = positions_accessor.offset() + buffer_view.offset();
                let buffer_bytes = positions_accessor.size() * positions_accessor.count();

                // println!("positions:");
                // println!("\tindex: {}", buffer_index);
                // println!("\toffset: {}", buffer_offset);
                // println!("\tbytes: {}", buffer_bytes);

                let vertex_buffer = self.device_buffers[buffer_index].clone();
                let vertex_slice = BufferSlice::from_typed_buffer_access(vertex_buffer)
                    .slice(buffer_offset..(buffer_offset + buffer_bytes))
                    .unwrap();

                unsafe { mem::transmute(vertex_slice) }
            };

            let index_slice: BufferSlice<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = {
                let buffer_view = indices_accessor.view();
                let buffer_index = buffer_view.buffer().index();
                let buffer_offset = indices_accessor.offset() + buffer_view.offset();
                let buffer_bytes = indices_accessor.size() * indices_accessor.count();

                // println!("indices:");
                // println!("\tindex: {}", buffer_index);
                // println!("\toffset: {}", buffer_offset);
                // println!("\tbytes: {}", buffer_bytes);

                let index_buffer = self.device_buffers[buffer_index].clone();
                let index_slice = BufferSlice::from_typed_buffer_access(index_buffer)
                    .slice(buffer_offset..(buffer_offset + buffer_bytes))
                    .unwrap();

                unsafe { mem::transmute(index_slice) }
            };

            // unsafe {
            //     let index_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(index_slice);
            //     println!("index_slice: {:?}", index_slice);
            // }

            // unsafe {
            //     let vertex_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(vertex_slice);
            //     println!("vertex_slice: {:?}", vertex_slice);
            // }

            command_buffer = command_buffer.draw_indexed(
                pipeline.clone(),
                dynamic,
                vertex_slice,
                index_slice,
                sets.clone(),
                () /* push_constants */).unwrap();
        }

        command_buffer
    }
}

// #[derive(Debug)]
// pub struct BufferSlicePublic<T: ?Sized, B> {
//     pub marker: PhantomData<T>,
//     pub resource: B,
//     pub offset: usize,
//     pub size: usize,
// }
