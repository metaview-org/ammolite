use std::mem;
use std::slice;
use std::ops::{Deref, DerefMut, Mul, Neg};
use std::fmt::{Debug, Formatter, Error};
use typenum::Unsigned;
use math::vector::*;

pub trait Matrix: Neg + Mul<Output=Self> + PartialEq + Sized + Debug {
    type Vector: Vector;

    fn zero() -> Self;
    fn identity() -> Self;
    fn transpose(&mut self);
}

pub trait AffineTransformation<V>: Matrix<Vector=V> where V: Homogeneous {
    fn scale(&mut self, coefficient: f32);
    fn translate(&mut self, translation: &<V as Homogeneous>::ProjectedVector);
}

pub trait Rotation3<V>: AffineTransformation<V> where V: Homogeneous {
    fn rotate(&mut self, translation: &<V as Homogeneous>::ProjectedVector);
}

macro_rules! impl_mat {
    ($ty_name:ident, $dims:expr, $macro_name:ident, $vector_ty_name:ident) => {
        #[derive(Clone, PartialEq)]
        pub struct $ty_name(pub [[f32; $dims]; $dims]);

        impl Matrix for $ty_name {
            type Vector = $vector_ty_name;

            #[inline]
            fn zero() -> Self {
                $ty_name([[0.0; $dims]; $dims])
            }

            #[inline]
            fn identity() -> Self {
                let mut result = Self::zero();

                for i in 0..$dims {
                    result.0[i][i] = 1.0;
                }

                result
            }

            #[inline]
            fn transpose(&mut self) {
                for column in 1..$dims {
                    for row in 0..column {
                        unsafe {
                            let dst = slice::from_raw_parts_mut(self.as_mut_ptr(), self.len());
                            mem::swap(&mut self[column][row], &mut dst[row][column]);
                        }
                    }
                }
            }
        }

        impl Neg for $ty_name {
            type Output = Self;

            #[inline]
            fn neg(mut self) -> Self::Output {
                for column in self.iter_mut() {
                    for component in column {
                        *component = -*component;
                    }
                }

                self
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [false];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&lhs, &rhs| {
                let mut result = $ty_name::identity();

                for result_row in 0..$dims {
                    for result_column in 0..$dims {
                        let mut result_cell = 0.0;

                        for cell_index in 0..$dims {
                            result_cell += lhs[cell_index][result_row] * rhs[result_column][cell_index];
                        }

                        result[result_column][result_row] = result_cell;
                    }
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [false];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, <$ty_name as Matrix>::Vector) -> <$ty_name as Matrix>::Vector;
            |&lhs, &rhs| {
                let mut result = <<$ty_name as Matrix>::Vector as Vector>::zero();

                for (result_row, result_component) in result.iter_mut().enumerate() {
                    for result_column in 0..$dims {
                        *result_component += lhs[result_column][result_row] * rhs[result_column];
                    }
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [false];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, f32) -> $ty_name;
            |&lhs, &rhs| {
                let mut result = lhs.clone();

                for column in result.iter_mut() {
                    for component in column.iter_mut() {
                        *component *= *rhs;
                    }
                }

                result
            }
        }

        impl Deref for $ty_name {
            type Target = [[f32; $dims]; $dims];

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $ty_name {
            #[inline]
            fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
                &mut self.0
            }
        }

        impl From<$ty_name> for [[f32; $dims]; $dims] {
            fn from(matrix: $ty_name) -> Self {
                *matrix
            }
        }

        impl Debug for $ty_name {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                let prefix = format!("{ty}!([", ty=stringify!($macro_name));

                write!(f, "{}", prefix)?;

                for row in 0..$dims {
                    if row > 0 {
                        write!(f, "\n")?;

                        // offset the values
                        for _ in 0..prefix.len() {
                            write!(f, " ")?;
                        }
                    }

                    for column in 0..$dims {
                        if column > 0 {
                            write!(f, "\t")?;
                        }

                        write!(f, "{}", self[column][row])?;
                    }
                }

                write!(f, "])")?;

                Ok(())
            }
        }
    }
}

macro_rules! impl_affine_transformation {
    ($ty_name:ident, $vector_ty_name:ident) => {
        impl AffineTransformation<$vector_ty_name> for $ty_name {
            #[inline]
            fn scale(&mut self, coefficient: f32) {
                let dims = <<<$vector_ty_name as Homogeneous>::ProjectedVector as Vector>::Dimensions as Unsigned>::to_usize();

                for i in 0..dims {
                    self[i][i] *= coefficient;
                }
            }

            #[inline]
            fn translate(&mut self, translation: &<$vector_ty_name as Homogeneous>::ProjectedVector) {
                let dims = <<$vector_ty_name as Vector>::Dimensions as Unsigned>::to_usize();

                for (index, component) in translation.iter().enumerate() {
                    self[dims - 1][index] += component;
                }
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [false];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, <$vector_ty_name as Homogeneous>::ProjectedVector) -> <$vector_ty_name as Homogeneous>::ProjectedVector;
            |&lhs, &rhs| {
                (lhs * rhs.into_homogeneous()).into_projected()
            }
        }
    }
}

impl_mat!(Mat1, 1, mat1, Vec1);
impl_mat!(Mat2, 2, mat2, Vec2);
impl_mat!(Mat3, 3, mat3, Vec3);
impl_mat!(Mat4, 4, mat4, Vec4);

impl_affine_transformation!(Mat2, Vec2);
impl_affine_transformation!(Mat3, Vec3);
impl_affine_transformation!(Mat4, Vec4);

impl Rotation3<Vec4> for Mat4 {
    #[inline]
    fn rotate(&mut self, euler_angles: &Vec3) {
        let e = euler_angles;

        if e[0] != 0.0 {
            let a = e[0];
            *self = mat4!([1.0,     0.0,      0.0, 0.0,
                           0.0, a.cos(), -a.sin(), 0.0,
                           0.0, a.sin(),  a.cos(), 0.0,
                           0.0,     0.0,      0.0, 1.0]) * &*self;
        }

        if e[1] != 0.0 {
            let b = e[1];
            *self = mat4!([ b.cos(), 0.0, b.sin(), 0.0,
                                0.0, 1.0,     0.0, 0.0,
                           -b.sin(), 0.0, b.cos(), 0.0,
                                0.0, 0.0,     0.0, 1.0]) * &*self;
        }

        if e[2] != 0.0 {
            let c = e[2];
            *self = mat4!([c.cos(), -c.sin(), 0.0, 0.0,
                           c.sin(),  c.cos(), 0.0, 0.0,
                               0.0,      0.0, 1.0, 0.0,
                               0.0,      0.0, 0.0, 1.0]) * &*self;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts;
    use super::*;

    #[test]
    fn matrix_construction() {
        let using_constructor = Mat3([[00.0, 10.0, 20.0],
                                      [01.0, 11.0, 21.0],
                                      [02.0, 12.0, 22.0]]);
        let using_macro = mat3!([00.0, 01.0, 02.0,
                                 10.0, 11.0, 12.0,
                                 20.0, 21.0, 22.0]);

        assert_eq!(using_constructor, using_macro);
    }

    #[test]
    fn matrix_matrix_multiplication() {
        let a = mat2!([0.0, 1.0,
                       2.0, 3.0]);
        let b = mat2!([4.0, 5.0,
                       6.0, 7.0]);

        assert_eq!(a * b, mat2!([6.0, 7.0,
                                 26.0, 31.0]));
    }

    #[test]
    fn matrix_vector_multiplication() {
        let mat = mat3!([1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0]);
        let vec: Vec3 = [10.0, 20.0, 40.0].into();

        assert_eq!(mat * vec, [170.0, 380.0, 590.0].into());
    }

    #[test]
    fn matrix_projected_vector_multiplication() {
        let mat = mat3!([1.0, 0.0, 8.0,
                         0.0, 1.0, 5.0,
                         0.0, 0.0, 1.0]);
        let vec: Vec2 = [10.0, 20.0].into();

        assert_eq!(mat * vec, [18.0, 25.0].into());
    }

    #[test]
    fn matrix_scalar_multiplication() {
        let mat = mat3!([1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0]);
        let scalar = 3.0;

        assert_eq!(mat * scalar, mat3!([3.0, 6.0, 9.0,
                                        12.0, 15.0, 18.0,
                                        21.0, 24.0, 27.0]));
    }

    #[test]
    fn matrix_vector_rotation() {
        let mut mat = Mat4::identity();

        mat.rotate(&[consts::FRAC_PI_2, 0.0, 0.0].into());
        mat.rotate(&[0.0, consts::FRAC_PI_2, 0.0].into());

        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [1.9999999, -3.0, -1.0000001].into());
    }

    #[test]
    fn matrix_vector_translation() {
        let mut mat = Mat4::identity();

        mat.translate(&[10.0, 20.0, 30.0].into());

        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [11.0, 22.0, 33.0].into());
    }

    #[test]
    fn matrix_vector_scale() {
        let mut mat = Mat4::identity();

        mat.scale(10.0);

        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [10.0, 20.0, 30.0].into());
    }
}

// macro_rules! mat1 {
//     {
//         $m00:expr$(;)*
//     } => {
//         Mat1([[$m00]])
//     };
// }

// macro_rules! mat2 {
//     {
//         $m00:expr, $m10:expr;
//         $m01:expr, $m11:expr$(;)*
//     } => {
//         Mat2([[$m00, $m01],
//               [$m10, $m11]])
//     };
// }

// macro_rules! mat3 {
//     {
//         $m00:expr, $m10:expr, $m20:expr;
//         $m01:expr, $m11:expr, $m21:expr;
//         $m02:expr, $m12:expr, $m22:expr$(;)*
//     } => {
//         Mat3([[$m00, $m01, $m02],
//               [$m10, $m11, $m12],
//               [$m20, $m21, $m22]])
//     };
// }

// macro_rules! mat4 {
//     {
//         $m00:expr, $m10:expr, $m20:expr, $m30:expr;
//         $m01:expr, $m11:expr, $m21:expr, $m31:expr;
//         $m02:expr, $m12:expr, $m22:expr, $m32:expr;
//         $m03:expr, $m13:expr, $m23:expr, $m33:expr$(;)*
//     } => {
//         Mat4([[$m00, $m01, $m02, $m03],
//               [$m10, $m11, $m12, $m13],
//               [$m20, $m21, $m22, $m23],
//               [$m30, $m31, $m32, $m33]])
//     };
// }
