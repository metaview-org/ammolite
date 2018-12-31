pub mod error;

use std::cmp;
use std::iter;
use std::sync::Arc;
use std::path::Path;
use std::ops::Deref;
use std::mem;
use std::fmt;
use std::marker::PhantomData;
use rayon::prelude::*;
use vulkano;
use vulkano::buffer::DeviceLocalBuffer;
use vulkano::command_buffer::sys::UnsafeCommandBufferBuilder;
use vulkano::command_buffer::sys::UnsafeCommandBufferBuilderPipelineBarrier;
use vulkano::command_buffer::sys::Flags;
use vulkano::command_buffer::sys::Kind;
use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::Filter;
use vulkano::sampler::MipmapMode;
use vulkano::sync::PipelineStages;
use vulkano::sync::AccessFlagBits;
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
use vulkano::descriptor::descriptor_set::PersistentDescriptorSetImg;
use vulkano::descriptor::descriptor_set::DescriptorSetDesc;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::sampler::Sampler;
use vulkano::image::AttachmentImage;
use vulkano::image::ImageDimensions;
use vulkano::image::immutable::ImmutableImage;
use vulkano::image::immutable::ImmutableImageInitialization;
use vulkano::image::Dimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImageLayout;
use vulkano::image::MipmapsCount;
use vulkano::image::traits::ImageAccess;
use vulkano::image::traits::ImageViewAccess;
use gltf::accessor::Accessor;
use gltf::{self, Document, Gltf};
use gltf::material::AlphaMode;
use gltf::mesh::util::ReadIndices;
use gltf::mesh::{Primitive, Mesh, Semantic};
use gltf::accessor::Accessor as GltfAccessor;
use gltf::Node;
use gltf::scene::Transform;
use gltf::Scene;
use gltf::accessor::DataType;
use gltf::image::Format as GltfFormat;
use gltf::texture::MagFilter;
use gltf::texture::MinFilter;
use generic_array::{GenericArray, ArrayLength};
use failure::Error;
use ::Position;
use ::SceneUBO;
use ::NodeUBO;
use ::MaterialUBO;
use ::MainDescriptorSet;
use math::*;
use iter::ArrayIterator;
use iter::ForcedExactSizeIterator;
use vertex::{GltfVertexPosition, GltfVertexNormal, GltfVertexTangent, GltfVertexTexCoord, GltfVertexBuffers};
use sampler::IntoVulkanEquivalent;
use safe_transmute::PodTransmutable;
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
pub struct InitializationDrawContext<'a, F, C>
        where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {
    pub draw_context: DrawContext<'a>,
    pub framebuffer: Arc<F>,
    pub clear_values: C,
}

#[derive(Clone)]
pub struct DrawContext<'a> {
    pub device: Arc<Device>,
    pub queue_family: QueueFamily<'a>,
    // pub combined_pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub pipeline_gltf_opaque: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub pipeline_gltf_mask: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub pipeline_gltf_blend_preprocess: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub pipeline_gltf_blend_finalize: Arc<GraphicsPipelineAbstract + Sync + Send>,
    pub dynamic: &'a DynamicState,
    pub main_descriptor_set: Arc<DescriptorSet + Send + Sync>,
    pub descriptor_set_blend: Arc<DescriptorSet + Send + Sync>,
    // pub main_descriptor_set: Arc<PersistentDescriptorSet<Arc<dyn GraphicsPipelineAbstract + Sync + Send>, (((), PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<SceneUBO>>>), PersistentDescriptorSetImg<Arc<vulkano::image::AttachmentImage>>)>>,
    // pub main_descriptor_set: Arc<PersistentDescriptorSet<Arc<dyn GraphicsPipelineAbstract + Sync + Send>, ((((), PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<SceneUBO>>>), PersistentDescriptorSetImg<Arc<AttachmentImage>>), PersistentDescriptorSetImg<Arc<AttachmentImage>>)>>,
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
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    ZeroBuffer {
        len: usize,
        initialization_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    },
    Image {
        data: Vec<u8>,
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
    fn initialize(self, device: &Arc<Device>, queue_family: QueueFamily, command_buffer_builder: AutoCommandBufferBuilder) -> Result<AutoCommandBufferBuilder, Error> {
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

                let mut command_buffer_builder = command_buffer_builder
                    .copy_buffer_to_image(staging_buffer, device_image.clone())?;

                let (width, height) = if let ImageDimensions::Dim2d { width, height, .. } = device_image.dimensions() {
                    (width, height)
                } else {
                    panic!("Texture mipmap generation is not implemented for non-2d textures.");
                };

                // TODO: Set mip count to Log2
                //
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

#[derive(Clone)]
pub struct HelperResources {
    pub empty_image: Arc<dyn ImageViewAccess + Send + Sync>,
    pub zero_buffer: Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>,
    pub default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    pub cheapest_sampler: Arc<Sampler>,
}

impl HelperResources {
    pub fn new<'a, I>(device: &Arc<Device>, queue_families: I, pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>)
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

        // Create the cheapest sampler possible
        let cheapest_sampler = Sampler::new(
            device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0,
        )?;

        let (device_default_material_ubo_buffer, default_material_ubo_buffer_initialization) = unsafe {
            ImmutableBuffer::<MaterialUBO>::uninitialized(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
            )
        }?;
        let default_material_descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 2)
                .add_buffer(device_default_material_ubo_buffer.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .add_image(empty_device_image.clone()).unwrap()
                .add_sampler(cheapest_sampler.clone()).unwrap()
                .build().unwrap()
        );

        let tasks = vec![
            InitializationTask::Image {
                data: vec![0],
                device_image: Arc::new(empty_image_initialization),
                texel_conversion: None,
            },
            InitializationTask::ZeroBuffer {
                len: zero_buffer_len,
                initialization_buffer: Arc::new(zero_buffer_initialization),
            },
            InitializationTask::MaterialDescriptorSet {
                data: MaterialUBO::default(),
                initialization_buffer: Arc::new(default_material_ubo_buffer_initialization),
            },
        ];

        let output = HelperResources {
            empty_image: empty_device_image,
            zero_buffer: zero_device_buffer,
            default_material_descriptor_set,
            cheapest_sampler,
        };

        Ok(SimpleUninitializedResource::new(output, tasks))
    }
}

pub struct Model {
    document: Document,
    device_buffers: Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>,
    device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
    converted_index_buffers_by_accessor_index: Vec<Option<Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>>>,
    tangent_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>>,
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

enum ColorSpace {
    Srgb,
    Linear,
}

fn convert_double_channel_to_triple_channel<'a>(data_slice: &'a [u8]) -> Box<ExactSizeIterator<Item=u8> + 'a> {
    let unsized_iterator = data_slice.chunks(3).flat_map(|rgb| {
        ArrayIterator::new([rgb[0], rgb[1], rgb[2], 0xFF])
    });
    let iterator_len = data_slice.len() / 3 * 4;
    Box::new(ForcedExactSizeIterator::new(unsized_iterator, iterator_len))
}

impl Model {
    fn get_semantic_byte_slice<'a, T: PodTransmutable>(buffer_data_array: &'a [gltf::buffer::Data], accessor: &Accessor) -> &'a [T] {
        let view = accessor.view();

        // TODO: Most buffers have the corresponding default stride, but some don't.
        // Stride is applied in `BuffersIter` of `GltfVertexBufferDefinition`.
        // view.stride().map(|stride| panic!("The stride of the view to a buffer of `{}` must be `None`, but is `{}`.",
        //                                   unsafe { std::intrinsics::type_name::<T>() },
        //                                   stride));

        let byte_offset = view.offset() + accessor.offset();
        let byte_len = accessor.size() * accessor.count();
        let byte_slice: &[u8] = &buffer_data_array[view.buffer().index()][byte_offset..(byte_offset + byte_len)];

        // println!("byte_slice: [{}] offset: {}; len: {}", unsafe { std::intrinsics::type_name::<T>() }, byte_offset, byte_len);

        safe_transmute::guarded_transmute_pod_many_pedantic(byte_slice)
            .unwrap_or_else(|err| panic!("Invalid byte slice to convert to &[{}]: {}", unsafe { std::intrinsics::type_name::<T>() }, err))
    }

    fn get_semantic_buffer_view<T>(&self, accessor: &Accessor) -> BufferSlice<[T], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> {
        let buffer_view = accessor.view();
        let buffer_index = buffer_view.buffer().index();
        let buffer_offset = accessor.offset() + buffer_view.offset();
        let buffer_bytes = accessor.size() * accessor.count();

        let buffer = self.device_buffers[buffer_index].clone();
        let slice: BufferSlice<[u8], _> = BufferSlice::from_typed_buffer_access(buffer)
            .slice(buffer_offset..(buffer_offset + buffer_bytes))
            .unwrap();

        // println!("buffer_view: [{}] offset: {}; len: {}", unsafe { std::intrinsics::type_name::<T>() }, buffer_offset, buffer_bytes);

        unsafe { slice.reinterpret::<[T]>() }
    }

    pub fn import<'a, I, S>(device: &Arc<Device>, queue_families: I, pipeline: Arc<GraphicsPipelineAbstract + Sync + Send>, helper_resources: &HelperResources, path: S)
            -> Result<SimpleUninitializedResource<Model>, Error>
            where I: IntoIterator<Item = QueueFamily<'a>> + Clone,
                  S: AsRef<Path> {
        let (document, buffer_data_array, image_data_array) = gltf::import(path)?;
        let mut initialization_tasks: Vec<InitializationTask> = Vec::with_capacity(
            buffer_data_array.len() + image_data_array.len() + document.accessors().len() + document.nodes().len() + document.materials().len()
        );

        let mut converted_index_buffers_by_accessor_index: Vec<Option<Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>>> = vec![None; document.accessors().len()];
        let primitive_count = document.meshes().map(|mesh| mesh.primitives().len()).sum();
        let mut tangent_buffers: Vec<Vec<Option<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>>> = vec![Vec::new(); primitive_count];

        for (mesh_index, mesh) in document.meshes().enumerate() {
            tangent_buffers[mesh_index] = vec![None; mesh.primitives().len()];

            for (primitive_index, primitive) in mesh.primitives().enumerate() {
                if let Some(index_accessor) = primitive.indices() {
                    if index_accessor.data_type() == DataType::U8 {
                        let index_slice = Self::get_semantic_byte_slice(&buffer_data_array[..], &index_accessor);
                        let buffer_data: Vec<u8> = index_slice.into_iter().flat_map(|index| ArrayIterator::new([*index, 0])).collect(); // FIXME: Assumes byte order in u16
                        let converted_byte_len = mem::size_of::<u16>() * index_accessor.count();
                        let (device_index_buffer, index_buffer_initialization) = unsafe {
                            ImmutableBuffer::<[u16]>::raw(
                                device.clone(),
                                converted_byte_len,
                                BufferUsage {
                                    transfer_destination: true,
                                    index_buffer: true,
                                    ..BufferUsage::none()
                                },
                                queue_families.clone(),
                            )
                        }?;
                        let index_buffer_initialization: BufferSlice<[u8], _> = unsafe {
                            BufferSlice::from_typed_buffer_access(index_buffer_initialization).reinterpret::<[u8]>()
                        };
                        initialization_tasks.push(InitializationTask::Buffer {
                            data: buffer_data,
                            initialization_buffer: Arc::new(index_buffer_initialization),
                        });
                        converted_index_buffers_by_accessor_index[index_accessor.index()] = Some(device_index_buffer);
                    }
                }

                // Compute tangents for the model if they are missing.
                if primitive.get(&Semantic::Tangents).is_none() {
                    let vertex_count = primitive.get(&Semantic::Positions).unwrap().count();
                    let index_count = primitive.indices()
                        .map(|index_accessor| index_accessor.count())
                        .unwrap_or(vertex_count);

                    let converted_byte_len = mem::size_of::<GltfVertexTangent>() * vertex_count;
                    let (device_tangent_buffer, tangent_buffer_initialization) = unsafe {
                        ImmutableBuffer::<[u8]>::raw(
                            device.clone(),
                            converted_byte_len,
                            BufferUsage {
                                transfer_destination: true,
                                vertex_buffer: true,
                                ..BufferUsage::none()
                            },
                            queue_families.clone(),
                        )
                    }?;
                    let tangent_buffer_initialization: BufferSlice<[u8], _> = unsafe {
                        BufferSlice::from_typed_buffer_access(tangent_buffer_initialization).reinterpret::<[u8]>()
                    };
                    let mut buffer_data: Vec<GltfVertexTangent> = vec![GltfVertexTangent([0.0; 4]); vertex_count];
                    let vertices_per_face = 3;
                    let face_count = index_count / vertices_per_face;

                    // TODO: No need for a closure.
                    let get_semantic_index: Box<Fn(usize, usize) -> usize> = if let Some(index_accessor) = primitive.indices() {
                        match index_accessor.data_type() {
                            DataType::U8 => {
                                let index_slice: &[u8] = Self::get_semantic_byte_slice(&buffer_data_array[..], &index_accessor);

                                Box::new(move |face_index, vertex_index| {
                                    index_slice[face_index * vertices_per_face + vertex_index] as usize
                                })
                            },
                            DataType::U16 => {
                                let index_slice: &[u16] = Self::get_semantic_byte_slice(&buffer_data_array[..], &index_accessor);

                                Box::new(move |face_index, vertex_index| {
                                    index_slice[face_index * vertices_per_face + vertex_index] as usize
                                })
                            },
                            _ => unreachable!(),
                        }
                    } else {
                        Box::new(|face_index, vertex_index| { face_index * vertices_per_face + vertex_index })
                    };

                    let position_accessor = primitive.get(&Semantic::Positions).unwrap();
                    let normal_accessor = primitive.get(&Semantic::Normals).unwrap();
                    let tex_coord_accessor = primitive.get(&Semantic::TexCoords(0)).unwrap();
                    let position_slice: &[GltfVertexPosition] = Self::get_semantic_byte_slice(&buffer_data_array[..], &position_accessor);
                    let normal_slice: &[GltfVertexNormal] = Self::get_semantic_byte_slice(&buffer_data_array[..], &normal_accessor);
                    let tex_coord_slice: &[GltfVertexTexCoord] = Self::get_semantic_byte_slice(&buffer_data_array[..], &tex_coord_accessor);

                    mikktspace::generate_tangents(
                        &|| { vertices_per_face }, // vertices_per_face: &'a Fn() -> usize, 
                        &|| { face_count }, // face_count: &'a Fn() -> usize, 
                        &|face_index, vertex_index| { &position_slice[get_semantic_index(face_index, vertex_index)].0 }, // position: &'a Fn(usize, usize) -> &'a [f32; 3],
                        &|face_index, vertex_index| { &normal_slice[get_semantic_index(face_index, vertex_index)].0 }, // normal: &'a Fn(usize, usize) -> &'a [f32; 3],
                        &|face_index, vertex_index| { &tex_coord_slice[get_semantic_index(face_index, vertex_index)].0 }, // tex_coord: &'a Fn(usize, usize) -> &'a [f32; 2],
                        &mut |face_index, vertex_index, tangent| {
                            // println!("{} {} [{}] -> {:?}", face_index, vertex_index, get_semantic_index(face_index, vertex_index), tangent);
                            buffer_data[get_semantic_index(face_index, vertex_index)] = GltfVertexTangent(tangent);
                        }, // set_tangent: &'a mut FnMut(usize, usize, [f32; 4])
                    );

                    initialization_tasks.push(InitializationTask::Buffer {
                        data: safe_transmute::guarded_transmute_to_bytes_pod_vec(buffer_data),
                        initialization_buffer: Arc::new(tangent_buffer_initialization),
                    });
                    tangent_buffers[mesh_index][primitive_index] = Some(device_tangent_buffer);
                }
            }
        }

        let mut device_buffers: Vec<Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = Vec::with_capacity(buffer_data_array.len());

        for gltf::buffer::Data(buffer_data) in buffer_data_array.into_iter() {
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
                initialization_buffer: Arc::new(buffer_initialization),
            });
            device_buffers.push(device_buffer);
        }

        let mut device_images: Vec<Arc<dyn ImageViewAccess + Send + Sync>> = vec![helper_resources.empty_image.clone(); image_data_array.len()]; // Vec::with_capacity(document.textures().len());

        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            let images_slice = [(ColorSpace::Srgb,   pbr.base_color_texture().map(|wrapped| wrapped.texture())),
                                (ColorSpace::Linear, pbr.metallic_roughness_texture().map(|wrapped| wrapped.texture())),
                                (ColorSpace::Linear, material.normal_texture().map(|wrapped| wrapped.texture())),
                                (ColorSpace::Linear, material.occlusion_texture().map(|wrapped| wrapped.texture())),
                                (ColorSpace::Srgb,   material.emissive_texture().map(|wrapped| wrapped.texture()))];

            for (space, image) in images_slice.into_iter()
                                     .filter(|(_, option)| option.is_some())
                                     .map(|(space, option)| (space, option.as_ref().unwrap())) {
                let gltf::image::Data {
                    pixels,
                    format,
                    width,
                    height,
                } = image_data_array[image.index()].clone(); // FIXME: Avoid cloning

                macro_rules! insert_image_with_format_impl {
                    ([$($vk_format:tt)+], $insert_init_tasks:expr) => {{
                        let (device_image, image_initialization) = ImmutableImage::uninitialized(
                            device.clone(),
                            Dimensions::Dim2d {
                                width,
                                height,
                            },
                            $($vk_format)+,
                            MipmapsCount::One,
                            ImageUsage {
                                transfer_source: true,
                                transfer_destination: true,
                                sampled: true,
                                ..ImageUsage::none()
                            },
                            ImageLayout::ShaderReadOnlyOptimal,
                            queue_families.clone(),
                        )?;
                        ($insert_init_tasks)(image_initialization);
                        device_images[image.index()] = device_image;
                    }}
                }

                macro_rules! insert_image_with_format {
                    ([$($vk_format:tt)+]) => {{
                        insert_image_with_format_impl! {
                            [$($vk_format)+],
                            |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                                initialization_tasks.push(InitializationTask::Image {
                                    data: pixels,
                                    device_image: Arc::new(image_initialization),
                                    texel_conversion: None,
                                });
                            }
                        }
                    }};

                    ([$($vk_format:tt)+], $texel_conversion:expr) => {{
                        insert_image_with_format_impl! {
                            [$($vk_format)+],
                            |image_initialization: ImmutableImageInitialization<$($vk_format)+>| {
                                initialization_tasks.push(InitializationTask::Image {
                                    data: pixels,
                                    device_image: Arc::new(image_initialization),
                                    texel_conversion: Some($texel_conversion),
                                });
                            }
                        }
                    }}
                }

                match (format, space) {
                    (GltfFormat::R8, ColorSpace::Linear) => insert_image_with_format!([R8Unorm]),
                    (GltfFormat::R8, ColorSpace::Srgb) => insert_image_with_format!([R8Srgb]),
                    (GltfFormat::R8G8, ColorSpace::Linear) => insert_image_with_format!([R8G8Unorm]),
                    (GltfFormat::R8G8, ColorSpace::Srgb) => insert_image_with_format!([R8G8Srgb]),
                    (GltfFormat::R8G8B8, ColorSpace::Linear) => insert_image_with_format!([R8G8B8A8Unorm], Box::new(convert_double_channel_to_triple_channel)),
                    (GltfFormat::R8G8B8, ColorSpace::Srgb) => insert_image_with_format!([R8G8B8A8Srgb], Box::new(convert_double_channel_to_triple_channel)),
                    (GltfFormat::R8G8B8A8, ColorSpace::Linear) => insert_image_with_format!([R8G8B8A8Unorm]),
                    (GltfFormat::R8G8B8A8, ColorSpace::Srgb) => insert_image_with_format!([R8G8B8A8Srgb]),
                }
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
                initialization_buffer: Arc::new(buffer_initialization),
            });
            node_descriptor_sets.push(descriptor_set);
        }

        let mut device_samplers: Vec<Arc<Sampler>> = Vec::with_capacity(document.samplers().len());

        for gltf_sampler in document.samplers() {
            let (min_filter, mipmap_mode) = gltf_sampler.min_filter()
                .unwrap_or(MinFilter::LinearMipmapLinear)
                .into_vulkan_equivalent();
            let sampler = Sampler::new(
                device.clone(),
                gltf_sampler.mag_filter().unwrap_or(MagFilter::Linear).into_vulkan_equivalent(),
                min_filter,
                mipmap_mode,
                gltf_sampler.wrap_s().into_vulkan_equivalent(),
                gltf_sampler.wrap_t().into_vulkan_equivalent(),
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                1.0, // TODO check the range of LOD
            )?;

            device_samplers.push(sampler);
        }

        let mut material_descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>> = Vec::with_capacity(document.materials().len());

        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            let base_color_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = pbr
                .base_color_texture()
                .map(|texture_info| {
                    let image_index = texture_info.texture().source().index();
                    device_images[image_index].clone()
                });
            let base_color_sampler_option: Option<Arc<Sampler>> = pbr
                .base_color_texture()
                .and_then(|texture_info| texture_info.texture().sampler().index())
                .map(|sampler_index| device_samplers[sampler_index].clone());
            let normal_texture_option: Option<Arc<dyn ImageViewAccess + Send + Sync>> = material
                .normal_texture()
                .map(|texture_info| {
                    let image_index = texture_info.texture().source().index();
                    device_images[image_index].clone()
                });
            let normal_sampler_option: Option<Arc<Sampler>> = material
                .normal_texture()
                .and_then(|texture_info| texture_info.texture().sampler().index())
                .map(|sampler_index| device_samplers[sampler_index].clone());
            let material_ubo = MaterialUBO::new(
                pbr.base_color_factor().into(),
                pbr.metallic_factor(),
                pbr.roughness_factor(),
                base_color_texture_option.is_some(),
                normal_texture_option.is_some(),
                material.normal_texture().map(|normal_texture| normal_texture.scale()).unwrap_or(1.0),
                material.alpha_cutoff(),
            );
            let (device_material_ubo_buffer, material_ubo_buffer_initialization) = unsafe {
                ImmutableBuffer::<MaterialUBO>::uninitialized(
                    device.clone(),
                    BufferUsage::uniform_buffer_transfer_destination(),
                )
            }?;
            let base_color_texture: Arc<dyn ImageViewAccess + Send + Sync> = base_color_texture_option
                .unwrap_or_else(|| helper_resources.empty_image.clone());
            let base_color_sampler: Arc<Sampler> = base_color_sampler_option
                .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
            let normal_texture: Arc<dyn ImageViewAccess + Send + Sync> = normal_texture_option
                .unwrap_or_else(|| helper_resources.empty_image.clone());
            let normal_sampler: Arc<Sampler> = normal_sampler_option
                .unwrap_or_else(|| helper_resources.cheapest_sampler.clone());
            let descriptor_set: Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), 2)
                    .add_buffer(device_material_ubo_buffer.clone()).unwrap()
                    .add_image(base_color_texture).unwrap()
                    .add_sampler(base_color_sampler).unwrap()
                    .add_image(normal_texture).unwrap()
                    .add_sampler(normal_sampler).unwrap()
                    .build().unwrap()
            );

            initialization_tasks.push(InitializationTask::MaterialDescriptorSet {
                data: material_ubo,
                initialization_buffer: Arc::new(material_ubo_buffer_initialization),
            });
            material_descriptor_sets.push(descriptor_set);
        }

        Ok(SimpleUninitializedResource::new(Model {
            document,
            device_buffers,
            device_images,
            converted_index_buffers_by_accessor_index,
            tangent_buffers,
            node_descriptor_sets,
            material_descriptor_sets,
        }, initialization_tasks))
    }

    pub fn draw_scene<F, C>(
        &self,
        context: InitializationDrawContext<F, C>,
        scene_index: usize,
    ) -> Result<AutoCommandBuffer, Error>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {
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
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Opaque, 0);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Mask, 1);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Blend, 2);
        }

        command_buffer = command_buffer.next_subpass(false).unwrap();

        for node in scene.nodes() {
            command_buffer = self.draw_node(node, command_buffer, &draw_context, AlphaMode::Blend, 3);
        }

        command_buffer = command_buffer
            .end_render_pass().unwrap();

        Ok(command_buffer.build().unwrap())
    }

    pub fn draw_main_scene<F, C>(
        &self,
        context: InitializationDrawContext<F, C>,
    ) -> Result<AutoCommandBuffer, Error>
            where F: FramebufferAbstract + RenderPassDescClearValues<C> + Send + Sync + 'static {
        if let Some(main_scene_index) = self.document.default_scene().map(|default_scene| default_scene.index()) {
            self.draw_scene(context, main_scene_index)
        } else {
            Err(ModelDrawError::NoDefaultScene.into())
        }
    }

    pub fn draw_node<'a>(
        &self,
        node: Node<'a>,
        mut command_buffer: AutoCommandBufferBuilder,
        context: &DrawContext,
        alpha_mode: AlphaMode,
        subpass: u8,
    ) -> AutoCommandBufferBuilder {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let material = primitive.material();

                if material.alpha_mode() == alpha_mode {
                    let pipeline = match (alpha_mode, subpass) {
                        (AlphaMode::Opaque, _) => &context.pipeline_gltf_opaque,
                        (AlphaMode::Mask, _) => &context.pipeline_gltf_mask,
                        (AlphaMode::Blend, 2) => &context.pipeline_gltf_blend_preprocess,
                        (AlphaMode::Blend, 3) => &context.pipeline_gltf_blend_finalize,
                        _ => panic!("Invalid alpha_mode/subpass combination."),
                    };

                    let material_descriptor_set = material.index().map(|material_index| {
                        self.material_descriptor_sets[material_index].clone()
                    }).unwrap_or_else(|| {
                        context.helper_resources.default_material_descriptor_set.clone()
                    });

                    match (alpha_mode, subpass) {
                        (AlphaMode::Blend, 3) => {
                            let descriptor_sets = (
                                context.main_descriptor_set.clone(),
                                self.node_descriptor_sets[node.index()].clone(),
                                material_descriptor_set,
                                context.descriptor_set_blend.clone(),
                            );

                            command_buffer = self.draw_primitive(
                                &mesh,
                                &primitive,
                                command_buffer,
                                context,
                                descriptor_sets.clone(),
                                pipeline,
                            );

                        },
                        _ => {
                            let descriptor_sets = (
                                context.main_descriptor_set.clone(),
                                self.node_descriptor_sets[node.index()].clone(),
                                material_descriptor_set,
                            );

                            command_buffer = self.draw_primitive(
                                &mesh,
                                &primitive,
                                command_buffer,
                                context,
                                descriptor_sets.clone(),
                                pipeline,
                            );
                        },
                    }
                }
            }
        }

        for child in node.children() {
            command_buffer = self.draw_node(child, command_buffer, context, alpha_mode, subpass);
        }

        command_buffer
    }

    pub fn draw_primitive<'a, S>(
        &self,
        mesh: &Mesh<'a>,
        primitive: &Primitive<'a>,
        mut command_buffer: AutoCommandBufferBuilder,
        context: &DrawContext,
        sets: S,
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
    ) -> AutoCommandBufferBuilder where S: DescriptorSetsCollection + Clone {
        let positions_accessor = primitive.get(&Semantic::Positions).unwrap();
        let normals_accessor = primitive.get(&Semantic::Normals).expect("Normals must be provided by the glTF model for now.");
        let tangents_accessor = primitive.get(&Semantic::Tangents);
        let tex_coords_accessor = primitive.get(&Semantic::TexCoords(0));
        let indices_accessor = primitive.indices();

        let position_slice: BufferSlice<[GltfVertexPosition], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>
            = self.get_semantic_buffer_view(&positions_accessor);
        let normal_slice: BufferSlice<[GltfVertexNormal], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>>
            = self.get_semantic_buffer_view(&normals_accessor);

        let tangent_slice: BufferSlice<[GltfVertexTangent], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
            let tangent_slice: BufferSlice<[u8], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = tangents_accessor.map(|tangents_accessor| {
                self.get_semantic_buffer_view(&tangents_accessor)
            }).unwrap_or_else(|| {
                let buffer = self.tangent_buffers[mesh.index()][primitive.index()].as_ref()
                    .expect("No tangents provided by the model and no tangents were precomputed.");

                BufferSlice::from_typed_buffer_access(buffer.clone())
            });

            unsafe { tangent_slice.reinterpret::<[GltfVertexTangent]>() }
        };

        // let tangent_slice: BufferSlice<[GltfVertexTangent], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
        //     let buffer_view = tangents_accessor.view();
        //     let buffer_index = buffer_view.buffer().index();
        //     let buffer_offset = tangents_accessor.offset() + buffer_view.offset();
        //     let buffer_bytes = tangents_accessor.size() * tangents_accessor.count();

        //     let tangent_buffer = self.device_buffers[buffer_index].clone();
        //     let tangent_slice = BufferSlice::from_typed_buffer_access(tangent_buffer)
        //         .slice(buffer_offset..(buffer_offset + buffer_bytes))
        //         .unwrap();

        //     unsafe { tangent_slice.reinterpret::<[GltfVertexTangent]>() }
        // };

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
            position_buffer: Some(Arc::new(position_slice)),
            normal_buffer: Some(Arc::new(normal_slice)),
            tangent_buffer: Some(Arc::new(tangent_slice)),
            tex_coord_buffer: Some(Arc::new(tex_coord_slice)),
        };

        if let Some(indices_accessor) = indices_accessor {
            macro_rules! draw_indexed {
                ($index_type:ty; $command_buffer:ident, $context:ident, $vertex_buffers:ident, $indices_accessor:ident, $sets:ident) => {
                    let index_slice: BufferSlice<[$index_type], Arc<dyn TypedBufferAccess<Content=[u8]> + Send + Sync>> = {
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
                        // $context.combined_pipeline.clone(),
                        pipeline.clone(),
                        $context.dynamic,
                        $vertex_buffers.get_individual_buffers(),
                        index_slice,
                        $sets.clone(),
                        () /* push_constants */).unwrap();
                }
            }

            match indices_accessor.data_type() {
                DataType::U8 => {
                    let index_buffer = self.converted_index_buffers_by_accessor_index[indices_accessor.index()]
                        .as_ref()
                        .expect("Could not access a pre-generated `u16` index buffer, maybe it was not generated?")
                        .clone();
                    command_buffer = command_buffer.draw_indexed(
                        // context.combined_pipeline.clone(),
                        pipeline.clone(),
                        context.dynamic,
                        vertex_buffers.get_individual_buffers(),
                        index_buffer,
                        sets.clone(),
                        () /* push_constants */).unwrap();
                },
                DataType::U16 => {
                    draw_indexed!(u16; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                DataType::U32 => {
                    draw_indexed!(u32; command_buffer, context, vertex_buffers, indices_accessor, sets);
                },
                _ => {
                    panic!("Index type not supported.");
                },
            }
        } else {
            command_buffer = command_buffer.draw(
                // context.combined_pipeline.clone(),
                pipeline.clone(),
                context.dynamic,
                vertex_buffers.get_individual_buffers(),
                sets.clone(),
                () /* push_constants */).unwrap();
        }

        command_buffer
    }
}
