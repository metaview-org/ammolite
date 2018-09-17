pub mod error;

use std::iter;
use std::sync::Arc;
use std::path::Path;
use std::ops::Deref;
use std::mem;
use std::fmt;
use std::marker::PhantomData;
use rayon::prelude::*;
use vulkano;
use vulkano::sync::GpuFuture;
use vulkano::command_buffer::{DynamicState, AutoCommandBuffer, AutoCommandBufferBuilder};
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::instance::QueueFamily;
use vulkano::format::ClearValue;
use vulkano::format::*;
use vulkano::framebuffer::RenderPassDesc;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassDescClearValues;
use vulkano::buffer::TypedBufferAccess;
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
use vulkano::sampler::Sampler;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::immutable::ImmutableImageInitialization;
use vulkano::image::Dimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImageLayout;
use vulkano::image::MipmapsCount;
use vulkano::image::traits::ImageAccess;
use vulkano::image::traits::ImageViewAccess;
use gltf::{self, Document, Gltf};
use gltf::mesh::util::ReadIndices;
use gltf::mesh::{Primitive, Mesh, Semantic};
use gltf::accessor::Accessor as GltfAccessor;
use gltf::Node;
use gltf::scene::Transform;
use gltf::Scene;
use gltf::accessor::DataType;
use gltf::image::Format as GltfFormat;
use generic_array::{GenericArray, ArrayLength};
use failure::Error;
use ::Position;
use ::PipelineImpl;
use ::NodeUBO;
use ::MaterialUBO;
use ::MainDescriptorSet;
use math::*;
use iter::ArrayIterator;
use iter::ForcedExactSizeIterator;
use vertex::{GltfVertexPosition, GltfVertexTexCoord, GltfVertexBuffers};
use self::error::*;

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
    pub draw_context: DrawContext<'a, RPD>,
    pub framebuffer: Arc<F>,
    pub clear_values: C,
}

#[derive(Clone)]
pub struct DrawContext<'a, RPD>
    where RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
    pub device: Arc<Device>,
    pub queue_family: QueueFamily<'a>,
    pub pipeline: PipelineImpl<RPD>,
    pub dynamic: &'a DynamicState,
    pub main_descriptor_set: MainDescriptorSet<RPD>,
    pub helper_resources: HelperResources,
}

// struct FormatConversionIterator<I, O> {
//     data: Vec<u8>,
//     input_chunk_len: usize,
//     conversion_fn: Box<dyn for<'a> Fn(&'a I) -> O>
// }

pub enum InitializationTask {
    Buffer {
        data: Vec<u8>,
        initialization_buffer: Box<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    ZeroBuffer {
        len: usize,
        initialization_buffer: Box<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    Image {
        data: Vec<u8>,
        device_image: Box<dyn ImageAccess + Send + Sync>,
        texel_conversion: Option<Box<dyn for<'a> Fn(&'a [u8]) -> Box<ExactSizeIterator<Item=u8> + 'a>>>,
    },
    NodeDescriptorSet {
        data: NodeUBO,
        initialization_buffer: Box<dyn TypedBufferAccess<Content=NodeUBO> + Send + Sync>,
    },
    MaterialDescriptorSet {
        data: MaterialUBO,
        initialization_buffer: Box<dyn TypedBufferAccess<Content=MaterialUBO> + Send + Sync>,
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
    fn initialize(self, device: &Arc<Device>, command_buffer_builder: AutoCommandBufferBuilder) -> Result<AutoCommandBufferBuilder, Error> {
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

                Ok(command_buffer_builder.copy_buffer_to_image(staging_buffer, device_image)?)
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
    fn initialize_resource(self, device: &Arc<Device>, mut command_buffer_builder: AutoCommandBufferBuilder) -> Result<(AutoCommandBufferBuilder, T), Error> {
        for initialization_task in self.tasks.into_iter() {
            command_buffer_builder = initialization_task.initialize(device, command_buffer_builder)?;
        }

        Ok((command_buffer_builder, self.output))
    }
}

#[derive(Clone)]
pub struct HelperResources {
    pub empty_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub zero_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    pub default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
}

impl HelperResources {
    pub fn new<'a, I>(device: &Arc<Device>, queue_families: I, pipeline: PipelineImpl<impl RenderPassDesc + Send + Sync + 'static>)
            -> Result<SimpleUninitializedResource<HelperResources>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
        let (empty_device_image, empty_image_initialization) = ImmutableImage::uninitialized(
            device.clone(),
            Dimensions::Dim2d {
                width: 1,
                height: 1,
            },
            R8Uint,
            MipmapsCount::One,
            ImageUsage {
                transfer_destination: true,
                sampled: true,
                ..ImageUsage::none()
            },
            ImageLayout::ShaderReadOnlyOptimal,
            queue_families.clone(),
        )?;

        let zero_buffer_len = (1 << 15) * 2; // FIXME: Figure out a way to dynamically resize the zero buffer
        let (zero_device_buffer, zero_buffer_initialization) = unsafe {
            ImmutableBuffer::raw(
                device.clone(),
                zero_buffer_len,
                BufferUsage {
                    transfer_destination: true,
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                queue_families.clone(),
            )
        }?;

        let (device_default_material_ubo_buffer, default_material_ubo_buffer_initialization) = unsafe {
            ImmutableBuffer::<MaterialUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 2)
                .add_buffer(device_default_material_ubo_buffer.clone()).unwrap()
                .add_sampled_image(
                    empty_device_image.clone(),
                    Sampler::simple_repeat_linear(device.clone()), // TODO
                ).unwrap()
                .build().unwrap()
        );

        let tasks = vec![
            InitializationTask::Image {
                data: vec![0],
                device_image: Box::new(empty_image_initialization),
                texel_conversion: None,
            },
            InitializationTask::ZeroBuffer {
                len: zero_buffer_len,
                initialization_buffer: Box::new(zero_buffer_initialization),
            },
            InitializationTask::MaterialDescriptorSet {
                data: MaterialUBO::default(),
                initialization_buffer: Box::new(default_material_ubo_buffer_initialization),
            },
        ];

        let output = HelperResources {
            empty_image: empty_device_image,
            zero_buffer: zero_device_buffer,
            default_material_descriptor_set,
        };

        Ok(SimpleUninitializedResource::new(output, tasks))
    }
}

pub struct Model {
    document: Document,
    device_buffers: Vec<Arc<ImmutableBuffer<[u8]>>>,
    device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
    // Note: Do not ever try to express the descriptor set explicitly.
    node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
    material_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
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
    pub fn import<'a, I, S>(device: &Arc<Device>, queue_families: I, pipeline: PipelineImpl<impl RenderPassDesc + Send + Sync + 'static>, helper_resources: &HelperResources, path: S)
            -> Result<SimpleUninitializedResource<Model>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone,
                  S: AsRef<Path> {
        let (document, buffer_data_array, image_data_array) = gltf::import(path)?;
        let mut initialization_tasks: Vec<InitializationTask> = Vec::with_capacity(
            buffer_data_array.len() + image_data_array.len() + document.nodes().len()
        );
        let mut device_buffers: Vec<Arc<ImmutableBuffer<[u8]>>> = Vec::with_capacity(buffer_data_array.len());

        for (index, gltf::buffer::Data(buffer_data)) in buffer_data_array.into_iter().enumerate() {
            let (device_buffer, buffer_initialization) = unsafe {
                ImmutableBuffer::raw(
                    device.clone(),
                    buffer_data.len(),
                    BufferUsage { // TODO: Scan document for buffer usage and optimize
                        transfer_destination: true,
                        uniform_buffer: true,
                        storage_buffer: true,
                        index_buffer: true,
                        vertex_buffer: true,
                        indirect_buffer: true,
                        ..BufferUsage::none()
                    },
                    queue_families.clone(),
                )
            }?;
            initialization_tasks.push(InitializationTask::Buffer {
                data: buffer_data,
                initialization_buffer: Box::new(buffer_initialization),
            });
            device_buffers.push(device_buffer);
        }

        let mut device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>> = Vec::with_capacity(document.textures().len());

        for (index, image_data) in image_data_array.into_iter().enumerate() {
            let gltf::image::Data {
                pixels,
                format,
                width,
                height,
            } = image_data;

            macro_rules! push_image_with_format_impl {
                ([$($vk_format:tt)+], $push_init_tasks:expr) => {{
                    let (device_image, image_initialization) = ImmutableImage::uninitialized(
                        device.clone(),
                        Dimensions::Dim2d {
                            width,
                            height,
                        },
                        $($vk_format)+,
                        MipmapsCount::One, // TODO: Figure out how mipmapping works
                        ImageUsage {
                            transfer_destination: true,
                            sampled: true,
                            ..ImageUsage::none()
                        },
                        ImageLayout::ShaderReadOnlyOptimal,
                        queue_families.clone(),
                    )?;
                    ($push_init_tasks)(image_initialization);
                    device_images.push(device_image);
                }}
            }

            macro_rules! push_image_with_format {
                ([$($vk_format:tt)+]) => {{
                    push_image_with_format_impl! {
                        [$($vk_format)+],
                        |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                            initialization_tasks.push(InitializationTask::Image {
                                data: pixels,
                                device_image: Box::new(image_initialization),
                                texel_conversion: None,
                            });
                        }
                    }
                }};

                ([$($vk_format:tt)+], $texel_conversion:expr) => {{
                    push_image_with_format_impl! {
                        [$($vk_format)+],
                        |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                            initialization_tasks.push(InitializationTask::Image {
                                data: pixels,
                                device_image: Box::new(image_initialization),
                                texel_conversion: Some($texel_conversion),
                            });
                        }
                    }
                }}
            }

            match format {
                GltfFormat::R8 => push_image_with_format!([R8Uint]),
                GltfFormat::R8G8 => push_image_with_format!([R8G8Uint]),
                GltfFormat::R8G8B8 => push_image_with_format!([R8G8B8A8Srgb], Box::new(|data_slice| {
                    let unsized_iterator = data_slice.chunks(3).flat_map(|rgb| {
                        ArrayIterator::new([rgb[0], rgb[1], rgb[2], 0xFF])
                    });
                    let iterator_len = data_slice.len() / 3 * 4;
                    Box::new(ForcedExactSizeIterator::new(unsized_iterator, iterator_len))
                })),
                GltfFormat::R8G8B8A8 => push_image_with_format!([R8G8B8A8Srgb]),
            }
        }

        let mut node_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.nodes().len());
        let transform_matrices = get_node_matrices(&document);

        for node in document.nodes() {
            let node_ubo = NodeUBO::new(transform_matrices[node.index()].clone());
            let (device_buffer, buffer_initialization) = unsafe {
                ImmutableBuffer::<NodeUBO>::uninitialized(
                    device.clone(),
                    BufferUsage::uniform_buffer_transfer_destination(),
                )
            }?;
            let descriptor_set = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 1)
                    .add_buffer(device_buffer.clone()).unwrap()
                    .build().unwrap()
            );

            initialization_tasks.push(InitializationTask::NodeDescriptorSet {
                data: node_ubo,
                initialization_buffer: Box::new(buffer_initialization),
            });
            node_descriptor_sets.push(descriptor_set);
        }

        let mut material_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.materials().len());

        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            let base_color_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = pbr
                .base_color_texture()
                .map(|it| it.texture().index())
                .map(|index| device_images[index].clone());
            let material_ubo = MaterialUBO::new(
                pbr.base_color_factor().into(),
                pbr.metallic_factor(),
                pbr.roughness_factor(),
                base_color_texture_option.is_some(),
            );
            let (device_material_ubo_buffer, material_ubo_buffer_initialization) = unsafe {
                ImmutableBuffer::<MaterialUBO>::uninitialized(
                    device.clone(),
                    BufferUsage::uniform_buffer_transfer_destination(),
                )
            }?;
            let base_color_texture: Arc<dyn ImageViewAccess + Send + Sync> = base_color_texture_option
                .unwrap_or_else(|| helper_resources.empty_image.clone());
            let descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 2)
                    .add_buffer(device_material_ubo_buffer.clone()).unwrap()
                    .add_sampled_image(
                        base_color_texture,
                        Sampler::simple_repeat_linear(device.clone()), // TODO
                    ).unwrap()
                    .build().unwrap()
            );

            initialization_tasks.push(InitializationTask::MaterialDescriptorSet {
                data: material_ubo,
                initialization_buffer: Box::new(material_ubo_buffer_initialization),
            });
            material_descriptor_sets.push(descriptor_set);
        }

        Ok(SimpleUninitializedResource::new(Model {
            document,
            device_buffers,
            node_descriptor_sets,
            material_descriptor_sets,
            device_images,
        }, initialization_tasks))
    }

    pub fn draw_scene<F, C, RPD>(&self, context: InitializationDrawContext<F, C, RPD>, scene_index: usize) -> Result<AutoCommandBuffer, Error>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static,
                  RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        if scene_index >= self.document.scenes().len() {
            return Err(ModelDrawError::InvalidSceneIndex { index: scene_index }.into());
        }

        let InitializationDrawContext {
            draw_context,
            framebuffer,
            clear_values,
        } = context;

        let scene = self.document.scenes().nth(scene_index).unwrap();
        let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(draw_context.device.clone(), draw_context.queue_family.clone())
            .unwrap()
            .begin_render_pass(framebuffer.clone(), false, clear_values).unwrap();

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
            Err(ModelDrawError::NoDefaultScene.into())
        }
    }

    pub fn draw_node<'a, RPD>(&self, node: Node<'a>, mut command_buffer: AutoCommandBufferBuilder, context: &DrawContext<RPD>)
        -> AutoCommandBufferBuilder
        where RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let material = primitive.material();
                let material_descriptor_set = material.index().map(|material_index| {
                    self.material_descriptor_sets[material_index].clone()
                }).unwrap_or_else(|| {
                    context.helper_resources.default_material_descriptor_set.clone()
                });

                let descriptor_sets = (
                    context.main_descriptor_set.clone(),
                    self.node_descriptor_sets[node.index()].clone(),
                    material_descriptor_set,
                );

                command_buffer = self.draw_primitive(&primitive, command_buffer, context, descriptor_sets.clone());
            }
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, context);
        }

        command_buffer
    }

    pub fn draw_primitive<'a, S, RPD>(&self, primitive: &Primitive<'a>, mut command_buffer: AutoCommandBufferBuilder, context: &DrawContext<RPD>, sets: S)
        -> AutoCommandBufferBuilder
        where S: DescriptorSetsCollection + Clone,
              RPD: RenderPassDesc + RenderPassDescClearValues<Vec<ClearValue>> + Send + Sync + 'static {
        let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
        let tex_coords_accessor = primitive.get(&Semantic::TexCoords(0));
        let indices_accessor = primitive.indices();

        let position_slice: BufferSlice<[GltfVertexPosition], Arc<ImmutableBuffer<[u8]>>> = {
            let buffer_view = positions_accessor.view();
            let buffer_index = buffer_view.buffer().index();
            let buffer_offset = positions_accessor.offset() + buffer_view.offset();
            let buffer_bytes = positions_accessor.size() * positions_accessor.count();

            let position_buffer = self.device_buffers[buffer_index].clone();
            let position_slice = BufferSlice::from_typed_buffer_access(position_buffer)
                .slice(buffer_offset..(buffer_offset + buffer_bytes))
                .unwrap();

            unsafe { position_slice.reinterpret::<[GltfVertexPosition]>() }
        };

        let tex_coord_slice: BufferSlice<[GltfVertexTexCoord], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            let tex_coord_slice: BufferSlice<[u8], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = tex_coords_accessor.map(|tex_coords_accessor| {
                let buffer_view = tex_coords_accessor.view();
                let buffer_index = buffer_view.buffer().index();
                let buffer_offset = tex_coords_accessor.offset() + buffer_view.offset();
                let buffer_bytes = tex_coords_accessor.size() * tex_coords_accessor.count();

                let tex_coord_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync> = self.device_buffers[buffer_index].clone();

                BufferSlice::from_typed_buffer_access(tex_coord_buffer)
                    .slice(buffer_offset..(buffer_offset + buffer_bytes))
                    .unwrap()
            }).unwrap_or_else(|| {
                let zero_buffer = context.helper_resources.zero_buffer.clone();

                BufferSlice::from_typed_buffer_access(zero_buffer)
            });

            unsafe { tex_coord_slice.reinterpret::<[GltfVertexTexCoord]>() }
        };

        let vertex_buffers = GltfVertexBuffers {
            position_buffer: Some(position_slice),
            tex_coord_buffer: Some(tex_coord_slice),
        };

        if let Some(indices_accessor) = indices_accessor {
            macro_rules! draw_indexed {
                ($index_type:ty; $command_buffer:ident, $context:ident, $vertex_buffers:ident, $indices_accessor:ident, $sets:ident) => {
                    let index_slice: BufferSlice<[$index_type], Arc<ImmutableBuffer<[u8]>>> = {
                        let buffer_view = $indices_accessor.view();
                        let buffer_index = buffer_view.buffer().index();
                        let buffer_offset = $indices_accessor.offset() + buffer_view.offset();
                        let buffer_bytes = $indices_accessor.size() * $indices_accessor.count();

                        let index_buffer = self.device_buffers[buffer_index].clone();
                        let index_slice = BufferSlice::from_typed_buffer_access(index_buffer)
                            .slice(buffer_offset..(buffer_offset + buffer_bytes))
                            .unwrap();

                        unsafe { index_slice.reinterpret::<[$index_type]>() }
                    };

                    // unsafe {
                    //     let index_slice: BufferSlicePublic<[u16], Arc<CpuAccessibleBuffer<[u8]>>> = mem::transmute(index_slice);
                    //     println!("index_slice: {:?}", index_slice);
                    // }

                    $command_buffer = $command_buffer.draw_indexed(
                        $context.pipeline.clone(),
                        $context.dynamic,
                        $vertex_buffers,
                        index_slice,
                        $sets.clone(),
                        () /* push_constants */).unwrap();
                }
            }

            match indices_accessor.data_type() {
                DataType::U16 => {
                    draw_indexed!(u16; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                DataType::U32 => {
                    draw_indexed!(u32; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                _ => {
                    panic!("Index type not supported.");
                }
            }
        } else {
            command_buffer = command_buffer.draw(
                context.pipeline.clone(),
                context.dynamic,
                vertex_buffers,
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
