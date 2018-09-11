use std::ops::{Add, Deref, DerefMut, Sub, Neg};
use std::fmt::{Debug, Formatter, Error};
use typenum::{Unsigned, U1, U2, U3, U4};
use math::matrix::Mat4;

pub trait Vector: Neg + Sized + Clone + Debug + PartialEq {
    type Dimensions: Unsigned + Add<U1> + Sub<U1>;

    fn zero() -> Self;
    fn dot(&self, other: &Self) -> f32;
}

pub trait Projected: Vector {
    type HomogeneousVector: Vector + Homogeneous<ProjectedVector=Self>;

    fn into_homogeneous(&self) -> Self::HomogeneousVector;
}

pub trait Homogeneous: Vector {
    type ProjectedVector: Vector + Projected<HomogeneousVector=Self>;

    fn into_projected(&self) -> Self::ProjectedVector;
}

pub trait UnitQuaternion {
    fn to_matrix(&self) -> Option<Mat4>;
}

macro_rules! impl_vec {
    ($ty_name:ident, $dims:expr, $dims_ty:ty) => {
        #[derive(Clone, PartialEq)]
        pub struct $ty_name(pub [f32; $dims]);

        impl Vector for $ty_name {
            type Dimensions = $dims_ty;

            fn zero() -> Self {
                $ty_name([0.0; $dims])
            }

            fn dot(&self, other: &Self) -> f32 {
                let mut result = 0.0;

                for (a, b) in self.iter().zip(other.iter()) {
                    result += a * b;
                }

                result
            }
        }

        impl_unary_operator! {
            operator_type: [Neg];
            inline: [true];
            operator_fn: neg;
            generics: [];
            header: ($ty_name) -> $ty_name;
            |&myself| {
                let mut result = myself.clone();

                for (result_component, myself_component) in result.iter_mut().zip(myself.iter()) {
                    *result_component = -*myself_component;
                }

                result
            }
        }

        impl From<$ty_name> for [f32; $dims] {
            fn from(vector: $ty_name) -> Self {
                *vector
            }
        }

        impl From<[f32; $dims]> for $ty_name {
            fn from(array: [f32; $dims]) -> Self {
                $ty_name(array)
            }
        }

        impl Debug for $ty_name {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                write!(f, "{ty}({comp:?})", ty=stringify!($ty_name), comp=self.0)
            }
        }

        impl Deref for $ty_name {
            type Target = [f32; $dims];

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
    }
}

macro_rules! impl_projected_homogeneous {
    ($lower_dim_ty_name:ident, $higher_dim_ty_name:ident) => {
        impl Projected for $lower_dim_ty_name {
            type HomogeneousVector = $higher_dim_ty_name;

            fn into_homogeneous(&self) -> Self::HomogeneousVector {
                let mut result = <Self::HomogeneousVector as Vector>::zero();

                for (result_component, self_component) in result.iter_mut().zip(self.iter()) {
                    *result_component = *self_component;
                }

                if let Some(last_component) = result.last_mut() {
                    *last_component = 1.0;
                } else {
                    panic!("No last element in vector {}.", stringify!($higher_dim_ty_name))
                }

                result
            }
        }

        impl Homogeneous for $higher_dim_ty_name {
            type ProjectedVector = $lower_dim_ty_name;

            fn into_projected(&self) -> Self::ProjectedVector {
                let mut result = <Self::ProjectedVector as Vector>::zero();

                let mut last_component = *self.last().expect(
                    &format!("No last element in vector {}.", stringify!($higher_dim_ty_name)));

                if last_component == 0.0 {
                    last_component = 1.0;
                }

                for (result_component, self_component) in result.iter_mut().zip(self.iter()) {
                    *result_component = *self_component / last_component;
                }

                result
            }
        }
    }
}

impl_vec!(Vec1, 1, U1);
impl_vec!(Vec2, 2, U2);
impl_vec!(Vec3, 3, U3);
impl_vec!(Vec4, 4, U4);

impl_projected_homogeneous!(Vec1, Vec2);
impl_projected_homogeneous!(Vec2, Vec3);
impl_projected_homogeneous!(Vec3, Vec4);

impl UnitQuaternion for Vec4 {
    fn to_matrix(&self) -> Option<Mat4> {
        // Not actually a unit quaternion? Bail.
        if self.dot(&self) != 1.0 {
            return None;
        }

        let q = self;

        Some(mat4!([1.0-2.0*(q[2]*q[2]+q[3]*q[3]),     2.0*(q[1]*q[2]-q[0]*q[3]),     2.0*(q[0]*q[2]+q[1]*q[3]), 0.0,
                        2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]*q[1]+q[3]*q[3]),     2.0*(q[2]*q[3]-q[0]*q[1]), 0.0,
                        2.0*(q[1]*q[3]-q[0]*q[2]),     2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]*q[1]+q[2]*q[2]), 0.0,
                                              0.0,                           0.0,                           0.0, 1.0]))
    }
}

