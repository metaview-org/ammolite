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
use safe_transmute::PodTransmutable;
use gltf::mesh::Semantic;
use gltf::mesh::Primitive;
use crate::iter::ArrayIterator;
use crate::shaders::gltf_vert::MainInput;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GltfVertexPosition(pub [f32; 3]);
unsafe impl PodTransmutable for GltfVertexPosition {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GltfVertexNormal(pub [f32; 3]);
unsafe impl PodTransmutable for GltfVertexNormal {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GltfVertexTangent(pub [f32; 4]);
unsafe impl PodTransmutable for GltfVertexTangent {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GltfVertexTexCoord(pub [f32; 2]);
unsafe impl PodTransmutable for GltfVertexTexCoord {}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GltfVertexColor(pub [f32; 4]);
unsafe impl PodTransmutable for GltfVertexColor {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VertexAttributeProperties {
    pub stride: usize,
}

macro_rules! impl_buffers {
    {
        $field_len:expr, $field_len_ty:ty;
        $([$field_name:ident: $($buffer_type_name:tt)+] of [$attribute_name:ident: $($attribute_type:tt)+] {
            default_stride: $default_stride:expr,
            missing_stride: $missing_stride:expr,
            semantic: $semantic:expr$(,)?
        }),+$(,)?
    } => {
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        pub struct VertexAttributePropertiesSet {
            $(pub $attribute_name: VertexAttributeProperties),*
        }

        impl<'a, 'b> From<&'b Primitive<'a>> for VertexAttributePropertiesSet {
            #[allow(unreachable_code)]
            fn from(primitive: &'b Primitive<'a>) -> Self {
                let mut result = VertexAttributePropertiesSet::default();

                $(
                    if let Some(accessor) = primitive.get(&$semantic) {
                        if let Some(stride) = accessor.view().stride() {
                            result.$attribute_name.stride = stride;
                        }
                    } else {
                        // For mandatory vertex attributes, set stride to default stride.
                        // For other attributes, set stride to 0 because the zero buffer is used.
                        result.$attribute_name.stride = $missing_stride;
                    }
                )+

                result
            }
        }

        impl Default for VertexAttributePropertiesSet {
            fn default() -> Self {
                VertexAttributePropertiesSet {
                    $(
                        $attribute_name: VertexAttributeProperties {
                            stride: $default_stride,
                        }
                    ),+
                }
            }
        }

        pub struct GltfVertexBuffers<$($($buffer_type_name),+),+>
                where $($($buffer_type_name)+: TypedBufferAccess<Content=[$($attribute_type)+]> + Send + Sync + 'static,)+ {
            $(pub $field_name: Option<Arc<$($buffer_type_name)+>>,)+
        }

        impl<$($($buffer_type_name),+),+> GltfVertexBuffers<$($($buffer_type_name),+),+>
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

        #[derive(Clone)]
        pub struct GltfVertexBufferDefinition {
            pub properties_set: VertexAttributePropertiesSet,
        }

        impl GltfVertexBufferDefinition {
            pub fn new(properties_set: VertexAttributePropertiesSet) -> Self {
                Self { properties_set }
            }
        }

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

                // for (index, element) in interface.elements().enumerate() {
                //     println!("element #{}: {:?}", index, element);
                // }

                let attribute_names = [$(stringify!($attribute_name)),+];
                // let attribute_type_sizes = [$(mem::size_of::<$($attribute_type)+>()),+];
                let attribute_properties = [$(&self.properties_set.$attribute_name),+];

                debug_assert_eq!(
                    interface.elements().len(),
                    $field_len,
                    "The number of fields in the shader and program code is inconsistent.",
                );

                for element in interface.elements() {
                    let field_index = element.location.start as usize;

                    debug_assert_eq!(
                        element.name.expect("Shader input attribute has no name.").as_ref(),
                        attribute_names[field_index],
                        "The field types in the shader and program code are inconsistent",
                    );

                    buffers[field_index] = (
                        field_index as u32,
                        // attribute_type_sizes[field_index],
                        attribute_properties[field_index].stride,
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
                }

                Ok((ArrayIterator::new(buffers), ArrayIterator::new(attribs)))
            }
        }
    }
}

impl_buffers! {
    5, U5;

    [position_buffer: PositionBuffer] of [position: GltfVertexPosition] {
        default_stride: 4 * 3,
        missing_stride: unreachable!(),
        semantic: Semantic::Positions,
    },
    [normal_buffer: NormalBuffer] of [normal: GltfVertexNormal] {
        default_stride: 4 * 3,
        missing_stride: 4 * 3,
        semantic: Semantic::Normals,
    },
    [tangent_buffer: TangentBuffer] of [tangent: GltfVertexTangent] {
        default_stride: 4 * 4,
        missing_stride: 4 * 4,
        semantic: Semantic::Tangents,
    },
    [tex_coord_buffer: TexCoordBuffer] of [tex_coord: GltfVertexTexCoord] {
        default_stride: 4 * 2,
        missing_stride: 0,
        semantic: Semantic::TexCoords(0), //TODO
    },
    [vertex_color_buffer: VertexColorBuffer] of [vertex_color: GltfVertexColor] {
        default_stride: 4 * 4,
        missing_stride: 0,
        semantic: Semantic::Colors(0), //TODO
    },
}
