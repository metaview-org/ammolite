use std::sync::Arc;
use std::path::Path;
use std::ops::Deref;
use std::mem;
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
            let device_buffer = CpuAccessibleBuffer::from_iter(device, BufferUsage::all(), (**buffer).into_iter().cloned());
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

    pub fn draw_scene<F, C, RPD>(&self, device: Arc<Device>, queue_family: QueueFamily,
                      framebuffer: Arc<F>, clear_values: C, pipeline: PipelineImpl<RPD>,
                      dynamic: &DynamicState, scene_index: usize) -> Result<AutoCommandBuffer, ()>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Clone + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if scene_index >= self.document.scenes().len() {
            return Err(());
        }

        let scene = self.document.scenes().nth(scene_index).unwrap();
        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device, queue_family)
            .unwrap()
            .begin_render_pass(framebuffer, false, clear_values).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, pipeline, dynamic, ());
        }

        command_buffer = command_buffer.end_render_pass().unwrap();

        Ok(command_buffer.build().unwrap())
    }

    pub fn draw_main_scene<F, C, RPD>(&self, device: Arc<Device>, queue_family: QueueFamily,
                      framebuffer: Arc<F>, clear_values: C, pipeline: PipelineImpl<RPD>,
                      dynamic: &DynamicState) -> Result<AutoCommandBuffer, ()>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Clone + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(device, queue_family, framebuffer, clear_values, pipeline, dynamic, main_scene_index)
        } else {
            Err(())
        }
    }

    pub fn draw_node<'a, S, RPD>(&self, node: Node<'a>, command_buffer: AutoCommandBufferBuilder, pipeline: PipelineImpl<RPD>, dynamic: &DynamicState, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {

        if let Some(mesh) = node.mesh() {
            command_buffer = self.draw_mesh(mesh, command_buffer, pipeline, dynamic, sets);
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, pipeline, dynamic, sets);
        }

        command_buffer
    }

    pub fn draw_mesh<'a, S, RPD>(&self, mesh: Mesh<'a>, command_buffer: AutoCommandBufferBuilder, pipeline: PipelineImpl<RPD>, dynamic: &DynamicState, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        for primitive in mesh.primitives() {
            let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
            let indices_accessor = primitive.indices().unwrap();

            let vertex_slice: BufferSlice<[Position], _> = {
                let positions_buffer_index = positions_accessor.view().buffer().index();
                let positions_buffer_offset = positions_accessor.offset();
                let positions_buffer_bytes = positions_accessor.size() * positions_accessor.count();

                let vertex_buffer = self.device_buffers[positions_buffer_index];
                let vertex_slice = BufferSlice::from_typed_buffer_access(vertex_buffer)
                    .slice(positions_buffer_offset..(positions_buffer_offset + positions_buffer_bytes))
                    .unwrap();

                unsafe { mem::transmute(vertex_slice) }
            };

            let index_slice: BufferSlice<[u16], _> = {
                let indices_buffer_index = indices_accessor.view().buffer().index();
                let indices_buffer_offset = indices_accessor.offset();
                let indices_buffer_bytes = indices_accessor.size() * indices_accessor.count();

                let index_buffer = self.device_buffers[indices_buffer_index];
                let index_slice = BufferSlice::from_typed_buffer_access(index_buffer)
                    .slice(indices_buffer_offset..(indices_buffer_offset + indices_buffer_bytes))
                    .unwrap();

                unsafe { mem::transmute(index_slice) }
            };

            command_buffer = command_buffer.draw_indexed(
                pipeline,
                dynamic,
                vertex_slice,
                index_slice,
                sets,
                () /* push_constants */).unwrap();
        }

        command_buffer
    }
}
