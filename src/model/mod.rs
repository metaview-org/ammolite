pub mod error;

use std::sync::Arc;
use std::path::Path;
use std::ops::Deref;
use std::mem;
use std::marker::PhantomData;
use vulkano;
use vulkano::sync::GpuFuture;
use vulkano::command_buffer::{DynamicState, AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::device::Device;
use vulkano::device::Queue;
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
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf;
use vulkano::descriptor::descriptor_set::DescriptorSetDesc;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use gltf::{self, Document, Gltf};
use gltf::mesh::util::ReadIndices;
use gltf::mesh::{Mesh, Semantic};
use gltf::accessor::Accessor as GltfAccessor;
use gltf::Node;
use gltf::scene::Transform;
use gltf::Scene;
use gltf::accessor::DataType;
use failure::Error;
use ::Position;
use ::PipelineImpl;
use ::NodeUBO;
use ::MainDescriptorSet;
use math::matrix::Mat4;
use math::matrix::Matrix;
use self::error::DrawError;

// #[derive(Clone)]
// struct MeasuredDescriptorSetsCollection {
//     collection: Arc<dyn DescriptorSetsCollection>,
//     sets: usize,
// }

// #[derive(Clone)]
// struct DescriptorSetVec {
//     collections: Vec<MeasuredDescriptorSetsCollection>,
// }

// impl DescriptorSetVec {
//     pub fn new(slice: &[Arc<dyn DescriptorSetsCollection>]) -> Self {
//         let mut collections = Vec::with_capacity(slice.len());

//         for collection in slice {
//             let mut sets = 0;

//             while let Some(_) = collection.num_bindings_in_set(sets) {
//                 sets += 1;
//             }

//             collections.push(MeasuredDescriptorSetsCollection {
//                 collection: collection.clone(),
//                 sets,
//             });
//         }

//         DescriptorSetVec {
//             collections
//         }
//     }

//     pub fn collection_by_set(&self, set: usize) -> Option<(&MeasuredDescriptorSetsCollection, usize)> {
//         unimplemented!()
//     }
// }

// unsafe impl DescriptorSetsCollection for DescriptorSetVec {
//     fn into_vec(self) -> Vec<Box<DescriptorSet + Send + Sync>> {
//         let len = self.collections.iter().map(|collection| collection.sets).sum();
//         let mut result = Vec::with_capacity(len);

//         for measured_collection in self.collections.into_iter() {
//             let collection = measured_collection.collection;
//             let mut subresult = collection.into_vec();
//             result.append(&mut subresult);
//         }

//         result
//     }

//     fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
//         self.collection_by_set(set).and_then(|(collection, rel_index)| {
//             collection.collection.num_bindings_in_set(rel_index)
//         })
//     }

//     fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
//         self.collection_by_set(set).and_then(|(collection, rel_index)| {
//             collection.collection.descriptor(rel_index, binding)
//         })
//     }
// }

// #[derive(Clone)]
// struct DescriptorSetCollectionAppend<A, R>
//         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//               R: DescriptorSetsCollection {
//     rest_len: usize,
//     rest: R,
//     append: A,
// }

// // unsafe impl<A, R> Send for DescriptorSetCollectionAppend<A, R>
// //         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
// //               R: DescriptorSetsCollection + Send {}
// // unsafe impl<A, R> Sync for DescriptorSetCollectionAppend<A, R>
// //         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
// //               R: DescriptorSetsCollection + Sync {}

// impl<A, R> DescriptorSetCollectionAppend<A, R>
//         where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//               R: DescriptorSetsCollection {
//     pub fn new(rest: R, append: A) -> Self {
//         let mut rest_len = 0;

//         while let Some(_) = rest.num_bindings_in_set(rest_len) {
//             rest_len += 1;
//         }

//         DescriptorSetCollectionAppend {
//             rest_len,
//             rest,
//             append,
//         }
//     }
// }

// unsafe impl<A, R> DescriptorSetsCollection for DescriptorSetCollectionAppend<A, R>
//     where A: DescriptorSet + DescriptorSetDesc + Send + Sync + 'static,
//           R: DescriptorSetsCollection {
//     fn into_vec(self) -> Vec<Box<DescriptorSet + Send + Sync>> {
//         let mut result: Vec<Box<DescriptorSet + Send + Sync>> = self.rest.into_vec();

//         result.push(Box::new(self.append));

//         result
//     }

//     fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
//         if set == self.rest_len {
//             Some(self.append.num_bindings())
//         } else {
//             self.rest.num_bindings_in_set(set)
//         }
//     }

//     fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
//         if set == self.rest_len {
//             self.append.descriptor(binding)
//         } else {
//             self.rest.descriptor(set, binding)
//         }
//     }
// }

// TODO: Remove generics
#[derive(Clone)]
pub struct InitializationDrawContext<'a, F, C, RPD>
    where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
          RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
    pub device: Arc<Device>,
    pub queue_family: QueueFamily<'a>,
    pub framebuffer: Arc<F>,
    pub clear_values: C,
    pub pipeline: PipelineImpl<RPD>,
    pub dynamic: &'a DynamicState,
    pub main_descriptor_set: MainDescriptorSet<RPD>,
}

#[derive(Clone)]
pub struct DrawContext<'a, RPD>
    where RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
    device: Arc<Device>,
    queue_family: QueueFamily<'a>,
    pipeline: PipelineImpl<RPD>,
    dynamic: &'a DynamicState,
    main_descriptor_set: MainDescriptorSet<RPD>,
}

pub struct Model {
    document: Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
    device_buffers: Vec<Arc<CpuAccessibleBuffer<[u8]>>>,
    node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>, // Note: Do not ever try to express the descriptor set explicitly.
}

fn get_node_matrices_impl(parent: Option<&Node>, node: &Node, results: &mut Vec<Option<Mat4>>) {
    // Matrix and its children already calculated, bail.
    if let Some(_) = results[node.index()] {
        return;
    }

    results[node.index()] = Some(if let Some(parent) = parent {
        results[parent.index()].as_ref().unwrap() * Mat4(node.transform().matrix())
    } else {
        Mat4(node.transform().matrix())
    });

    for child in node.children() {
        get_node_matrices_impl(Some(node), &child, results);
    }
}

fn get_node_matrices(document: &Document) -> Vec<Mat4> {
    let mut results = Vec::with_capacity(document.nodes().len());

    for _ in 0..document.nodes().len() {
        results.push(None);
    }

    for scene in document.scenes() {
        for node in scene.nodes() {
            get_node_matrices_impl(None, &node, &mut results);
        }
    }

    results.into_iter().map(|option| option.unwrap_or_else(Mat4::identity)).collect()
}

impl Model {
    pub fn import<S: AsRef<Path>>(device: Arc<Device>, queue: Arc<Queue>, pipeline: PipelineImpl<impl RenderPassDesc + Send + Sync + 'static>, path: S) -> Result<(Model, Box<dyn GpuFuture>), Error> {
        let (document, buffers, images) = gltf::import(path)?;
        // TODO: setup buffer staging
        let device_buffers: Vec<Arc<CpuAccessibleBuffer<[u8]>>> = buffers.iter().map(|buffer| {
            let device_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (**buffer).into_iter().cloned());
            device_buffer.unwrap()
        }).collect();

        let mut initialization_future: Box<dyn GpuFuture> = Box::new(vulkano::sync::now(device.clone()));
        let mut node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.nodes().len());
        let transform_matrices = get_node_matrices(&document);

        for node in document.nodes() {
            let node_ubo = NodeUBO {
                matrix: transform_matrices[node.index()].clone().0,
            };
            let (node_ubo_buffer, gpu_future) = ImmutableBuffer::from_data(node_ubo, BufferUsage::uniform_buffer(), queue.clone())?;
            initialization_future = Box::new(initialization_future.join(gpu_future));
            let descriptor_set = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 1)
                    .add_buffer(node_ubo_buffer.clone()).unwrap()
                    .build().unwrap()
            );

            node_descriptor_sets.push(descriptor_set);
        }

        Ok((Model {
            document,
            buffers,
            images,
            device_buffers,
            node_descriptor_sets,
        }, initialization_future))
    }

    pub fn draw_scene<F, C, RPD>(&self, context: InitializationDrawContext<F, C, RPD>, scene_index: usize) -> Result<AutoCommandBuffer, Error>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if scene_index >= self.document.scenes().len() {
            return Err(DrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let InitializationDrawContext {
            device,
            queue_family,
            framebuffer,
            clear_values,
            pipeline,
            dynamic,
            main_descriptor_set,
        } = context;

        let scene = self.document.scenes().nth(scene_index).unwrap();
        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue_family.clone())
            .unwrap()
            .begin_render_pass(framebuffer.clone(), false, clear_values).unwrap();
        let draw_context = DrawContext {
            device: device,
            queue_family: queue_family,
            pipeline: pipeline,
            dynamic: dynamic,
            main_descriptor_set: main_descriptor_set,
        };

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context);
        }

        command_buffer = command_buffer.end_render_pass().unwrap();

        Ok(command_buffer.build().unwrap())
    }

    pub fn draw_main_scene<F, C, RPD>(&self, context: InitializationDrawContext<F, C, RPD>) -> Result<AutoCommandBuffer, Error>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(context, main_scene_index)
        } else {
            Err(DrawError::NoDefaultScene.into())
        }
    }

    pub fn draw_node<'a, RPD>(&self, node: Node<'a>, mut command_buffer: AutoCommandBufferBuilder, context: &DrawContext<RPD>)
        -> AutoCommandBufferBuilder
        where RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {

        let descriptor_sets = (
            context.main_descriptor_set.clone(),
            self.node_descriptor_sets[node.index()].clone(),
        );

        if let Some(mesh) = node.mesh() {
            command_buffer = self.draw_mesh(mesh, command_buffer, context, descriptor_sets.clone());
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, context);
        }

        command_buffer
    }

    pub fn draw_mesh<'a, S, RPD>(&self, mesh: Mesh<'a>, mut command_buffer: AutoCommandBufferBuilder, context: &DrawContext<RPD>, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection + Clone,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        for primitive in mesh.primitives() {
            let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
            let indices_accessor = primitive.indices();

            let vertex_slice: BufferSlice<[Position], Arc<CpuAccessibleBuffer<[u8]>>> = {
                let buffer_view = positions_accessor.view();
                let buffer_index = buffer_view.buffer().index();
                let buffer_offset = positions_accessor.offset() + buffer_view.offset();
                let buffer_bytes = positions_accessor.size() * positions_accessor.count();

                let vertex_buffer = self.device_buffers[buffer_index].clone();
                let vertex_slice = BufferSlice::from_typed_buffer_access(vertex_buffer)
                    .slice(buffer_offset..(buffer_offset + buffer_bytes))
                    .unwrap();

                unsafe { mem::transmute(vertex_slice) }
            };

            if let Some(indices_accessor) = indices_accessor {
                macro_rules! draw_indexed {
                    ($index_type:ty; $command_buffer:ident, $context:ident, $vertex_slice:ident, $indices_accessor:ident, $sets:ident) => {
                        let index_slice: BufferSlice<_, _> = {
                            let buffer_view = $indices_accessor.view();
                            let buffer_index = buffer_view.buffer().index();
                            let buffer_offset = $indices_accessor.offset() + buffer_view.offset();
                            let buffer_bytes = $indices_accessor.size() * $indices_accessor.count();

                            let index_buffer = self.device_buffers[buffer_index].clone();
                            let index_slice = BufferSlice::from_typed_buffer_access(index_buffer)
                                .slice(buffer_offset..(buffer_offset + buffer_bytes))
                                .unwrap();

                            unsafe { mem::transmute::<_, BufferSlice<[$index_type], Arc<CpuAccessibleBuffer<[u8]>>>>(index_slice) }
                        };

                        // unsafe {
                        //     let index_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(index_slice);
                        //     println!("index_slice: {:?}", index_slice);
                        // }

                        $command_buffer = $command_buffer.draw_indexed(
                            $context.pipeline.clone(),
                            $context.dynamic,
                            $vertex_slice,
                            index_slice,
                            $sets.clone(),
                            () /* push_constants */).unwrap();
                    }
                }

                match indices_accessor.data_type() {
                    DataType::U16 => {
                        draw_indexed!(u16; command_buffer, context, vertex_slice, indices_accessor, sets);
                    },
                    DataType::U32 => {
                        draw_indexed!(u32; command_buffer, context, vertex_slice, indices_accessor, sets);
                    },
                    _ => {
                        panic!("Index type not supported.");
                    }
                }
            } else {
                command_buffer = command_buffer.draw(
                    context.pipeline.clone(),
                    context.dynamic,
                    vertex_slice,
                    sets.clone(),
                    () /* push_constants */).unwrap();
            }
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
