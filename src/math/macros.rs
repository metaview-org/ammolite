macro_rules! impl_mat_macro {
    ($ty_name:ident, $dims:expr, $macro_name:ident, $vector_ty_name:ident) => {
        #[allow(unused)]
        macro_rules! $macro_name {
            {
                $component_array:expr
            } => {{
                use crate::math::matrix::{Matrix, $ty_name};
                let components: [f32; $dims * $dims] = $component_array;
                let mut result = <$ty_name as Matrix>::zero();

                for (index, component) in components.into_iter().enumerate() {
                    let column = index % $dims;
                    let row = index / $dims;

                    result.0[column][row] = *component;
                }

                result
            }};
        }
    }
}

impl_mat_macro!(Mat1, 1, mat1, Vec1);
impl_mat_macro!(Mat2, 2, mat2, Vec2);
impl_mat_macro!(Mat3, 3, mat3, Vec3);
impl_mat_macro!(Mat4, 4, mat4, Vec4);
