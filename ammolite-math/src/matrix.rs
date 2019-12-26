use std::mem;
use std::slice;
use std::ops::{Div, Add, Deref, DerefMut, Mul, Neg};
use std::fmt::{Debug, Formatter, Error};
use serde::{Deserialize, Serialize};
use typenum::Unsigned;
use crate::vector::*;

pub trait Matrix: Neg + Mul<Output=Self> + Mul<f32, Output=Self> + PartialEq + Sized + Clone + Debug {
    type Vector: Vector;
    type LowerDim: Matrix;

    const DIM: usize;
    const ZERO: Self;
    const IDENTITY: Self;

    /**
     * A matrix with 1 fewer dimensions, with given row and column removed.
     */
    fn submatrix(&self, row: usize, col: usize) -> Self::LowerDim;

    /**
     * The determinant of the matrix with given row and column removed.
     * (The determinant of the given submatrix.)
     */
    fn minor(&self, row: usize, col: usize) -> f32 {
        let submatrix = self.submatrix(row, col);
        submatrix.determinant()
    }

    /**
     * The minor, multiplied by `(-1)^(row + col)`.
     */
    fn cofactor(&self, row: usize, col: usize) -> f32 {
        let minor = self.minor(row, col);
        if (row + col) % 2 == 0 { minor } else { -minor }
    }

    /**
     * A matrix made up of all cofactors of the current matrix.
     */
    fn cofactor_matrix(&self) -> Self;
    fn determinant(&self) -> f32;
    fn transpose_mut(&mut self);

    fn transpose(&self) -> Self {
        let mut result = self.clone();
        result.transpose_mut();
        result
    }

    fn adjugate(&self) -> Self {
        let mut result = self.cofactor_matrix();
        result.transpose_mut();
        result
    }

    fn inverse(&self) -> Self {
        let determinant = self.determinant();

        if determinant == 0.0 {
            panic!("Attempt to invert a non-invertible matrix.");
        }

        self.adjugate() * (1.0 / determinant)
    }
}

pub trait AffineTransformation<V>: Matrix<Vector=V> where V: Homogeneous {
    fn scale(coefficient: f32) -> Self;
    fn translation(translation: &<V as Homogeneous>::ProjectedVector) -> Self;
}

pub trait Rotation3<V> where V: Homogeneous {
    fn rotation_yaw(yaw: f32) -> Self;
    fn rotation_pitch(pitch: f32) -> Self;
    fn rotation_roll(roll: f32) -> Self;
}

macro_rules! impl_mat {
    ($ty_name:ident, $lower_dim_ty_name:ty, $dims:expr, $macro_name:ident, $vector_ty_name:ident) => {
        #[derive(Clone, PartialEq, Deserialize, Serialize)]
        pub struct $ty_name([[f32; $dims]; $dims]);

        impl $ty_name {
            pub fn new(inner: [[f32; $dims]; $dims]) -> Self {
                Self(inner)
            }

            pub fn as_ref(&self) -> &[[f32; $dims]; $dims] {
                &self.0
            }

            pub fn as_mut(&mut self) -> &mut [[f32; $dims]; $dims] {
                &mut self.0
            }

            pub fn as_flat_ref(&self) -> &[f32; $dims * $dims] {
                unsafe { mem::transmute(&self.0) }
            }

            pub fn as_flat_mut(&mut self) -> &mut [f32; $dims * $dims] {
                unsafe { mem::transmute(&mut self.0) }
            }

            pub fn as_slice_ref(&self) -> &[f32] {
                &self.as_flat_ref()[..]
            }

            pub fn as_slice_mut(&mut self) -> &mut [f32] {
                &mut self.as_flat_mut()[..]
            }

            pub fn inner(&self) -> &[[f32; $dims]; $dims] {
                &self.0
            }

            pub fn inner_mut(&mut self) -> &mut [[f32; $dims]; $dims] {
                &mut self.0
            }

            pub fn into_inner(self) -> [[f32; $dims]; $dims] {
                self.0
            }
        }

        impl Matrix for $ty_name {
            type Vector = $vector_ty_name;
            type LowerDim = $lower_dim_ty_name;

            const DIM: usize = $dims;
            const ZERO: Self = Self([[0.0; $dims]; $dims]);
            const IDENTITY: Self = {
                let mut result = Self::ZERO;
                let mut i = 0;

                while i < $dims {
                    result.0[i][i] = 1.0;
                    i += 1;
                }

                result
            };

            fn cofactor_matrix(&self) -> Self {
                let mut result = Self::ZERO;

                for column in 0..$dims {
                    for row in 0..$dims {
                        result.0[column][row] = self.cofactor(row, column)
                    }
                }

                result
            }

            fn determinant(&self) -> f32 {
                if $dims == 1 {
                    self.0[0][0]
                } else {
                    let mut result = 0.0;

                    for col in 0..$dims {
                        result += self.0[col][0] * self.cofactor(0, col);
                    }

                    result
                }
            }

            fn submatrix(&self, row: usize, col: usize) -> Self::LowerDim {
                if $dims <= 1 {
                    panic!("Cannot get a submatrix of a matrix with dimension 1 or lower.");
                }

                let mut result = Self::LowerDim::ZERO;

                for result_col in 0..($dims - 1) {
                    for result_row in 0..($dims - 1) {
                        let source_col = result_col + if result_col >= col { 1 } else { 0 };
                        let source_row = result_row + if result_row >= row { 1 } else { 0 };

                        result.0[result_col][result_row] = self.0[source_col][source_row];
                    }
                }

                result
            }

            #[inline]
            fn transpose_mut(&mut self) {
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

        impl Default for $ty_name {
            fn default() -> Self {
                Self::IDENTITY
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
                let mut result = $ty_name::IDENTITY;

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
                let mut result = <<$ty_name as Matrix>::Vector as Vector>::ZERO;

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

        impl_binary_operator! {
            operator_type: [Div];
            inline: [false];
            operator_fn: div;
            generics: [];
            header: ($ty_name, f32) -> $ty_name;
            |&lhs, &rhs| {
                lhs * (1.0 / rhs)
            }
        }

        impl_binary_operator! {
            operator_type: [Add];
            inline: [false];
            operator_fn: add;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&lhs, &rhs| {
                let mut result = lhs.clone();

                for (result_column, rhs_column) in result.iter_mut().zip(rhs.iter()) {
                    for (result_component, rhs_component) in result_column.iter_mut().zip(rhs_column.iter()) {
                        *result_component += *rhs_component;
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

                        write!(f, "{:.4}", self[column][row])?;
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
            fn scale(coefficient: f32) -> Self {
                let mut result = Self::IDENTITY;
                let dims = <<<$vector_ty_name as Homogeneous>::ProjectedVector as Vector>::Dimensions as Unsigned>::to_usize();

                for i in 0..dims {
                    result[i][i] = coefficient;
                }

                result
            }

            #[inline]
            fn translation(translation: &<$vector_ty_name as Homogeneous>::ProjectedVector) -> Self {
                let mut result = Self::IDENTITY;
                let dims = <<$vector_ty_name as Vector>::Dimensions as Unsigned>::to_usize();

                for (index, component) in translation.iter().enumerate() {
                    result[dims - 1][index] = *component;
                }

                result
            }
        }

        // Ambiguous transformation from ND to (N+1)D
        // impl_binary_operator! {
        //     operator_type: [Mul];
        //     inline: [false];
        //     operator_fn: mul;
        //     generics: [];
        //     header: ($ty_name, <$vector_ty_name as Homogeneous>::ProjectedVector) -> <$vector_ty_name as Homogeneous>::ProjectedVector;
        //     |&lhs, &rhs| {
        //         (lhs * rhs.into_homogeneous{_position,_direction}()).into_projected()
        //     }
        // }
    }
}

macro_rules! impl_conversion_to_homogeneous_space {
    ($ty_name_from:ident[$dim_from:expr] -> $ty_name_to:ident) => {
        impl $ty_name_from {
            pub fn to_homogeneous(self) -> $ty_name_to {
                let mut result = $ty_name_to::IDENTITY;

                for row in 0..$dim_from {
                    for column in 0..$dim_from {
                        result[column][row] = self[column][row];
                    }
                }

                result
            }
        }
    }
}

impl_mat!(Mat1, Mat1, 1, mat1, Vec1);
impl_mat!(Mat2, Mat1, 2, mat2, Vec2);
impl_mat!(Mat3, Mat2, 3, mat3, Vec3);
impl_mat!(Mat4, Mat3, 4, mat4, Vec4);

impl_conversion_to_homogeneous_space!(Mat1[1] -> Mat2);
impl_conversion_to_homogeneous_space!(Mat2[2] -> Mat3);
impl_conversion_to_homogeneous_space!(Mat3[3] -> Mat4);

impl_affine_transformation!(Mat2, Vec2);
impl_affine_transformation!(Mat3, Vec3);
impl_affine_transformation!(Mat4, Vec4);

impl Mat3 {
    pub fn from_quaternion(quaternion: [f32; 4]) -> Self {
        let [qx, qy, qz, qw] = quaternion;

        // source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        mat3!([1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qz*qw),       2.0 * (qx*qz + qy*qw),
               2.0 * (qx*qy + qz*qw),       1.0 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qx*qw),
               2.0 * (qx*qz - qy*qw),       2.0 * (qy*qz + qx*qw),       1.0 - 2.0 * (qx*qx + qy*qy)])
    }
}

impl Rotation3<Vec4> for Mat3 {
    #[inline]
    fn rotation_pitch(pitch: f32) -> Self {
        mat3!([1.0,         0.0,          0.0,
               0.0, pitch.cos(), -pitch.sin(),
               0.0, pitch.sin(),  pitch.cos()])
    }

    #[inline]
    fn rotation_yaw(yaw: f32) -> Self {
        mat3!([ yaw.cos(), 0.0, yaw.sin(),
                      0.0, 1.0,       0.0,
               -yaw.sin(), 0.0, yaw.cos()])
    }

    #[inline]
    fn rotation_roll(roll: f32) -> Self {
        mat3!([roll.cos(), -roll.sin(), 0.0,
               roll.sin(),  roll.cos(), 0.0,
                      0.0,         0.0, 1.0])
    }
}

impl Rotation3<Vec4> for Mat4 {
    #[inline]
    fn rotation_pitch(pitch: f32) -> Self {
        mat4!([1.0,         0.0,          0.0, 0.0,
               0.0, pitch.cos(), -pitch.sin(), 0.0,
               0.0, pitch.sin(),  pitch.cos(), 0.0,
               0.0,         0.0,          0.0, 1.0])
    }

    #[inline]
    fn rotation_yaw(yaw: f32) -> Self {
        mat4!([ yaw.cos(), 0.0, yaw.sin(), 0.0,
                      0.0, 1.0,       0.0, 0.0,
               -yaw.sin(), 0.0, yaw.cos(), 0.0,
                      0.0, 0.0,       0.0, 1.0])
    }

    #[inline]
    fn rotation_roll(roll: f32) -> Self {
        mat4!([roll.cos(), -roll.sin(), 0.0, 0.0,
               roll.sin(),  roll.cos(), 0.0, 0.0,
                      0.0,         0.0, 1.0, 0.0,
                      0.0,         0.0, 0.0, 1.0])
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
        let mat = Mat4::rotation_yaw(consts::FRAC_PI_2)
            * Mat4::rotation_pitch(consts::FRAC_PI_2);
        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [1.9999999, -3.0, -1.0000001].into());
    }

    #[test]
    fn matrix_vector_translation() {
        let mat = Mat4::translation(&[10.0, 20.0, 30.0].into());
        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [11.0, 22.0, 33.0].into());
    }

    #[test]
    fn matrix_vector_scale() {
        let mat = Mat4::scale(10.0);
        let vec: Vec3 = [1.0, 2.0, 3.0].into();

        assert_eq!(mat * vec, [10.0, 20.0, 30.0].into());
    }

    #[test]
    fn matrix_determinant() {
        let matrix = mat3!([-2.0,  2.0, -3.0,
                            -1.0,  1.0,  3.0,
                             2.0,  0.0, -1.0]);
        let determinant = matrix.determinant();
        assert_eq!(determinant, 18.0);
    }

    #[test]
    fn matrix_adjugate() {
        let matrix = mat3!([-3.0,  2.0, -5.0,
                            -1.0,  0.0, -2.0,
                             3.0, -4.0,  1.0]);
        let adjugate = matrix.adjugate();
        assert_eq!(adjugate, mat3!([-8.0, 18.0, -4.0,
                                    -5.0, 12.0, -1.0,
                                     4.0, -6.0,  2.0]));
    }

    #[test]
    fn matrix_inverse() {
        let matrix = mat3!([ 7.0,  2.0,  1.0,
                             0.0,  3.0, -1.0,
                            -3.0,  4.0, -2.0]);
        let inverse = matrix.inverse();
        assert_eq!(inverse, mat3!([-2.0,   8.0, -5.0,
                                    3.0, -11.0,  7.0,
                                    9.0, -34.0, 21.0]));
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
