use std::mem;
use std::sync::Arc;
use vulkano;
use vulkano::pipeline::vertex::VertexSource;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::TypedBufferAccess;
use vulkano::pipeline::vertex::IncompatibleVertexDefinitionError;
use vulkano::pipeline::vertex::VertexDefinition;
use vulkano::pipeline::vertex::InputRate;
use vulkano::pipeline::vertex::AttributeInfo;
use vulkano::pipeline::shader::ShaderInterfaceDef;
use typenum::*;
use iter::ArrayIterator;
use ::gltf_vert::MainInput;

#[repr(C)]
pub struct GltfVertexPosition([f32; 3]);

#[repr(C)]
pub struct GltfVertexTexCoord([f32; 2]);

macro_rules! impl_buffers {
    {
        $field_len:expr, $field_len_ty:ty;
        $([$field_name:ident: $($buffer_type_name:tt)+] of [$attribute_name:ident: $($attribute_type:tt)+]),+,
    } => {
        pub struct GltfVertexBuffers<PositionBuffer, TexCoordBuffer>
                where $($($buffer_type_name)+: TypedBufferAccess<Content=[$($attribute_type)+]> + Send + Sync + 'static,)+ {
            $(pub $field_name: Option<Arc<$($buffer_type_name)+>>,)+
        }

        impl<PositionBuffer, TexCoordBuffer> GltfVertexBuffers<PositionBuffer, TexCoordBuffer>
                where $($($buffer_type_name)+: TypedBufferAccess<Content=[$($attribute_type)+]> + Send + Sync + 'static,)+ {
            pub fn get_individual_buffers(&self) -> Vec<Arc<dyn BufferAccess + Send + Sync>> {
                let mut result: Vec<Arc<dyn BufferAccess + Send + Sync>> = Vec::new();

                $(
                    if let Some(ref buffer) = self.$field_name {
                        result.push(buffer.clone());
                    }
                )+

                result
            }
        }

        pub struct GltfVertexBufferDefinition;

        unsafe impl<$($($buffer_type_name)+,)+> VertexSource<GltfVertexBuffers<$($($buffer_type_name)+,)+>> for GltfVertexBufferDefinition
                where $($($buffer_type_name)+: TypedBufferAccess<Content=[$($attribute_type)+]> + Send + Sync + 'static,)+ {
            fn decode(&self, buffers: GltfVertexBuffers<$($($buffer_type_name)+,)+>)
                    -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
                let GltfVertexBuffers {$(
                    $field_name,
                )+} = buffers;
                let vertices = [$(
                    ($field_name).as_ref().map(|buffer| buffer.size() / mem::size_of::<$($attribute_type)+>()),
                )+].into_iter()
                   .cloned()
                   .filter(Option::is_some)
                   .map(Option::unwrap)
                   .min()
                   .unwrap();
                let instances = 1;

                let individual_buffers: Vec<Box<dyn BufferAccess + Send + Sync>> = {
                    vec![$(
                        ($field_name).map(|field| Box::new(field) as Box<dyn BufferAccess + Send + Sync>),
                    )+].into_iter()
                       .filter(Option::is_some)
                       .map(Option::unwrap)
                       .collect()
                };

                (individual_buffers, vertices, instances)
            }
        }

        unsafe impl VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> for GltfVertexBufferDefinition {
            fn decode(&self, buffers: Vec<Arc<dyn BufferAccess + Send + Sync>>)
                    -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
                let attribute_sizes = [$(
                    mem::size_of::<$($attribute_type)+>(),
                )+];
                let vertices = buffers.iter().zip(attribute_sizes.into_iter())
                    .map(|(buffer, attribute_size)| buffer.size() / attribute_size)
                    .min().unwrap();
                let instances = 1;

                let individual_buffers: Vec<Box<dyn BufferAccess + Send + Sync>> = {
                    buffers.into_iter()
                        .map(|x| Box::new(x) as Box<dyn BufferAccess + Send + Sync>)
                        .collect()
                };

                (individual_buffers, vertices, instances)
            }
        }

        unsafe impl VertexDefinition<MainInput> for GltfVertexBufferDefinition {
            type BuffersIter = ArrayIterator<(u32, usize, InputRate), $field_len_ty>;
            type AttribsIter = ArrayIterator<(u32, u32, AttributeInfo), $field_len_ty>;

            #[allow(unused_variables, unused_assignments)]
            fn definition(&self, interface: &MainInput)
                    -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
                let mut buffers: [(u32, usize, InputRate); $field_len] = [$({
                    let $field_name: ();
                    (0, 0, InputRate::Vertex)
                },)+];
                let mut attribs: [(u32, u32, AttributeInfo); $field_len] = [$({
                    let $field_name: ();
                    (0, 0, AttributeInfo {
                        offset: 0,
                        format: vulkano::format::Format::R4G4UnormPack8,
                    })
                },)+];
                let mut field_index = 0;

                $(
                    let element = interface.elements()
                        .nth(field_index)
                        .expect("Not enough vertex attributes in the shader.");

                    debug_assert_eq!(
                        element.name.expect("Shader input attribute has no name.").as_ref(),
                        stringify!($attribute_name)
                    );

                    buffers[field_index] = (
                        field_index as u32,
                        mem::size_of::<$($attribute_type)+>(),
                        InputRate::Vertex
                    );
                    attribs[field_index] = (
                        field_index as u32,
                        field_index as u32,
                        AttributeInfo {
                            offset: 0,
                            format: element.format,
                        }
                    );
                    field_index += 1;
                )+

                Ok((ArrayIterator::new(buffers), ArrayIterator::new(attribs)))
            }
        }
    }
}

impl_buffers! {
    2, U2;

    [position_buffer: PositionBuffer] of [position: GltfVertexPosition],
    [tex_coord_buffer: TexCoordBuffer] of [tex_coord: GltfVertexTexCoord],
}
