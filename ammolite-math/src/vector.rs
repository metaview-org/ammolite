use std::ops::{Add, Deref, DerefMut, Sub, Neg, AddAssign, SubAssign, Div, Mul, DivAssign, MulAssign, Index, IndexMut};
use std::convert::TryFrom;
use std::num::TryFromIntError;
use std::fmt::{Debug, Formatter, Error};
use det::det_copy;
use serde::{Deserialize, Serialize};
use typenum::{Unsigned, U1, U2, U3, U4};
use crate::matrix::Mat4;
use crate::ops::{DivEuclid, RemEuclid};
use paste::{item, expr};

pub trait Component {
    const ZERO: Self;

    fn to_f32(self) -> f32;
}

macro_rules! impl_component {
    ($($comp_ty:ty),*$(,)?) => {
        $(
            impl Component for $comp_ty {
                const ZERO: Self = 0 as $comp_ty;

                fn to_f32(self) -> f32 {
                    self as f32
                }
            }
        )*
    }
}

impl_component!(f32, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

pub trait Vector<C: Component>: Sized + Clone + Copy + Debug + PartialEq + Index<usize, Output=C> {
    type Dimensions: Unsigned + Add<U1> + Sub<U1>;

    const ZERO: Self;
    const DIMENSIONS: usize;

    fn dot(&self, other: &Self) -> C;
    fn norm_squared(&self) -> C;

    fn norm(&self) -> f32 {
        self.norm_squared().to_f32().sqrt()
    }

    fn distance_to_squared(&self, other: &Self) -> C;
    fn distance_to(&self, other: &Self) -> f32 {
        self.distance_to_squared(other).to_f32().sqrt()
    }
}

pub trait FloatVector<C: Component>: Vector<C> {
    fn normalize_mut(&mut self);
    fn normalize(&self) -> Self {
        let mut result = self.clone();
        result.normalize_mut();
        result
    }

    fn floor_mut(&mut self);
    fn floor(&self) -> Self {
        let mut result = self.clone();
        result.floor_mut();
        result
    }

    fn ceil_mut(&mut self);
    fn ceil(&self) -> Self {
        let mut result = self.clone();
        result.ceil_mut();
        result
    }

    fn floor_to_i32(&self) -> I32Vec3;
    fn ceil_to_i32(&self) -> I32Vec3;
}

pub trait Projected<C: Component>: Vector<C> {
    type HomogeneousVector: Vector<C> + Homogeneous<C, ProjectedVector=Self>;

    fn into_homogeneous_direction(&self) -> Self::HomogeneousVector;
    fn into_homogeneous_position(&self) -> Self::HomogeneousVector;
}

pub trait Homogeneous<C: Component>: Vector<C> {
    type ProjectedVector: Vector<C> + Projected<C, HomogeneousVector=Self>;

    fn into_projected(&self) -> Self::ProjectedVector;
}

pub trait UnitQuaternion {
    fn to_matrix(&self) -> Option<Mat4>;
}

macro_rules! impl_vec {
    ($ty_name:ident, $dims:expr, $dims_ty:ty, $comp_ty:ty) => {
        #[derive(Clone, Copy, PartialEq, Deserialize, Serialize)]
        pub struct $ty_name(pub [$comp_ty; $dims]);

        impl $ty_name {
            pub fn inner(&self) -> &[$comp_ty; $dims] {
                &self.0
            }

            pub fn inner_mut(&mut self) -> &mut [$comp_ty; $dims] {
                &mut self.0
            }

            pub fn into_inner(self) -> [$comp_ty; $dims] {
                self.0
            }

            pub fn min(&self, other: &Self) -> Self {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(self.iter().zip(other.iter())) {
                    *result_component = (*a_component).min(*b_component);
                }

                result
            }

            pub fn max(&self, other: &Self) -> Self {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(self.iter().zip(other.iter())) {
                    *result_component = (*a_component).max(*b_component);
                }

                result
            }
        }

        impl Vector<$comp_ty> for $ty_name {
            type Dimensions = $dims_ty;

            const ZERO: Self = Self([<$comp_ty as Component>::ZERO; $dims]);
            const DIMENSIONS: usize = $dims;

            fn dot(&self, other: &Self) -> $comp_ty {
                let mut result = <$comp_ty as Component>::ZERO;

                for (a, b) in self.iter().zip(other.iter()) {
                    result += a * b;
                }

                result
            }

            fn distance_to_squared(&self, other: &Self) -> $comp_ty {
                (self - other).norm_squared()
            }

            fn norm_squared(&self) -> $comp_ty {
                self.dot(&self)
            }
        }

        impl Default for $ty_name {
            fn default() -> Self {
                Self::ZERO
            }
        }

        impl Index<usize> for $ty_name {
            type Output = $comp_ty;

            fn index(&self, idx: usize) -> &Self::Output {
                &self.0[idx]
            }
        }

        impl IndexMut<usize> for $ty_name {
            fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
                &mut self.0[idx]
            }
        }

        impl_binary_operator! {
            operator_type: [Add];
            inline: [true];
            operator_fn: add;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = *a_component + *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Add];
            inline: [true];
            operator_fn: add;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = *a_component + *b;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Sub];
            inline: [true];
            operator_fn: sub;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = *a_component - *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Sub];
            inline: [true];
            operator_fn: sub;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = *a_component - *b;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [true];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = *a_component * *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [true];
            operator_fn: mul;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = *a_component * *b;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Mul];
            inline: [true];
            operator_fn: mul;
            generics: [];
            header: ($comp_ty, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, b_component) in result.iter_mut().zip(b.iter()) {
                    *result_component = *a * *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Div];
            inline: [true];
            operator_fn: div;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = *a_component / *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Div];
            inline: [true];
            operator_fn: div;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = *a_component / *b;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [Div];
            inline: [true];
            operator_fn: div;
            generics: [];
            header: ($comp_ty, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = $ty_name::ZERO;

                for (result_component, b_component) in result.iter_mut().zip(b.iter()) {
                    *result_component = *a / *b_component;
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [DivEuclid];
            inline: [true];
            operator_fn: div_euclid;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = (*a_component).div_euclid(*b_component);
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [DivEuclid];
            inline: [true];
            operator_fn: div_euclid;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = (*a_component).div_euclid(*b);
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [DivEuclid];
            inline: [true];
            operator_fn: div_euclid;
            generics: [];
            header: ($comp_ty, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, b_component) in result.iter_mut().zip(b.iter()) {
                    *result_component = (*a).div_euclid(*b_component);
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [RemEuclid];
            inline: [true];
            operator_fn: rem_euclid;
            generics: [];
            header: ($ty_name, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, (a_component, b_component)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
                    *result_component = (*a_component).rem_euclid(*b_component);
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [RemEuclid];
            inline: [true];
            operator_fn: rem_euclid;
            generics: [];
            header: ($ty_name, $comp_ty) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, a_component) in result.iter_mut().zip(a.iter()) {
                    *result_component = (*a_component).rem_euclid(*b);
                }

                result
            }
        }

        impl_binary_operator! {
            operator_type: [RemEuclid];
            inline: [true];
            operator_fn: rem_euclid;
            generics: [];
            header: ($comp_ty, $ty_name) -> $ty_name;
            |&a, &b| {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for (result_component, b_component) in result.iter_mut().zip(b.iter()) {
                    *result_component = (*a).rem_euclid(*b_component);
                }

                result
            }
        }

        impl<T> AddAssign<T> for $ty_name where Self: Add<T, Output=Self> {
            fn add_assign(&mut self, other: T) {
                *self = self.clone() + other;
            }
        }

        impl<T> SubAssign<T> for $ty_name where Self: Sub<T, Output=Self> {
            fn sub_assign(&mut self, other: T) {
                *self = self.clone() - other;
            }
        }

        impl<T> MulAssign<T> for $ty_name where Self: Mul<T, Output=Self> {
            fn mul_assign(&mut self, other: T) {
                *self = self.clone() * other;
            }
        }

        impl<T> DivAssign<T> for $ty_name where Self: Div<T, Output=Self> {
            fn div_assign(&mut self, other: T) {
                *self = self.clone() / other;
            }
        }

        impl From<$ty_name> for [$comp_ty; $dims] {
            fn from(vector: $ty_name) -> Self {
                *vector
            }
        }

        impl From<[$comp_ty; $dims]> for $ty_name {
            fn from(array: [$comp_ty; $dims]) -> Self {
                $ty_name(array)
            }
        }

        impl Debug for $ty_name {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                write!(f, "{ty}({comp:?})", ty=stringify!($ty_name), comp=self.0)
            }
        }

        impl Deref for $ty_name {
            type Target = [$comp_ty; $dims];

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

macro_rules! impl_vec_signed {
    ($ty_name:ident, $dims:expr, $dims_ty:ty) => {
        impl_unary_operator! {
            operator_type: [Neg];
            inline: [false];
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

        impl $ty_name {
            pub fn abs(&self) -> Self {
                let mut result = self.clone();

                for coord in result.iter_mut() {
                    *coord = coord.abs();
                }

                result
            }
        }
    }
}

macro_rules! impl_vec_f32 {
    ($ty_name:ident, $dims:expr, $dims_ty:ty) => {
        impl_vec!($ty_name, $dims, $dims_ty, f32);
        impl_vec_signed!($ty_name, $dims, $dims_ty);

        impl FloatVector<f32> for $ty_name {
            fn normalize_mut(&mut self) {
                let norm = self.norm();

                for coord in &mut self.0 {
                    *coord /= norm;
                }
            }

            fn floor_mut(&mut self) {
                for coord in self.inner_mut() {
                    *coord = coord.floor();
                }
            }

            fn ceil_mut(&mut self) {
                for coord in self.inner_mut() {
                    *coord = coord.ceil();
                }
            }

            fn floor_to_i32(&self) -> I32Vec3 {
                let floored = self.floor();
                let mut result = I32Vec3::ZERO;

                for (result_component, floored_component) in result.iter_mut().zip(floored.iter()) {
                    *result_component = *floored_component as i32;
                }

                result
            }

            fn ceil_to_i32(&self) -> I32Vec3 {
                let ceiled = self.ceil();
                let mut result = I32Vec3::ZERO;

                for (result_component, ceiled_component) in result.iter_mut().zip(ceiled.iter()) {
                    *result_component = *ceiled_component as i32;
                }

                result
            }
        }

        item! {
            use na::base::dimension::{
                $dims_ty as [< Na $dims_ty >],
            };

            #[cfg(feature = "nalgebra-interop")]
            impl From<na::base::VectorN<f32, [< Na $dims_ty >]>> for $ty_name {
                fn from(other: na::base::VectorN<f32, [< Na $dims_ty >]>) -> $ty_name {
                    (&other).into()
                }
            }

            #[cfg(feature = "nalgebra-interop")]
            impl<'a> From<&'a na::base::VectorN<f32, [< Na $dims_ty >]>> for $ty_name {
                fn from(other: &'a na::base::VectorN<f32, [< Na $dims_ty >]>) -> $ty_name {
                    let mut result = <$ty_name as Vector<f32>>::ZERO;

                    for (result_component, other_component) in result.iter_mut().zip(other.iter()) {
                        *result_component = *other_component;
                    }

                    result
                }
            }

            #[cfg(feature = "nalgebra-interop")]
            impl From<na::geometry::Point<f32, [< Na $dims_ty >]>> for $ty_name {
                fn from(other: na::geometry::Point<f32, [< Na $dims_ty >]>) -> $ty_name {
                    (&other.coords).into()
                }
            }

            #[cfg(feature = "nalgebra-interop")]
            impl<'a> From<&'a na::geometry::Point<f32, [< Na $dims_ty >]>> for $ty_name {
                fn from(other: &'a na::geometry::Point<f32, [< Na $dims_ty >]>) -> $ty_name {
                    (&other.coords).into()
                }
            }
        }
    }
}

macro_rules! impl_projected_homogeneous {
    ($lower_dim_ty_name:ident, $higher_dim_ty_name:ident) => {
        impl Projected<f32> for $lower_dim_ty_name {
            type HomogeneousVector = $higher_dim_ty_name;

            fn into_homogeneous_direction(&self) -> Self::HomogeneousVector {
                let mut result = <Self::HomogeneousVector as Vector<f32>>::ZERO;

                for (result_component, self_component) in result.iter_mut().zip(self.iter()) {
                    *result_component = *self_component;
                }

                result
            }

            fn into_homogeneous_position(&self) -> Self::HomogeneousVector {
                let mut result = self.into_homogeneous_direction();

                if let Some(last_component) = result.last_mut() {
                    *last_component = 1.0;
                } else {
                    panic!("No last element in vector {}.", stringify!($higher_dim_ty_name))
                }

                result
            }
        }

        impl Homogeneous<f32> for $higher_dim_ty_name {
            type ProjectedVector = $lower_dim_ty_name;

            fn into_projected(&self) -> Self::ProjectedVector {
                let mut result = <Self::ProjectedVector as Vector<f32>>::ZERO;

                let mut last_component = *self.last().unwrap_or_else(||
                    panic!("No last element in vector {}.", stringify!($higher_dim_ty_name)));

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

macro_rules! impl_vec_integer {
    (
        $ty_name:ident, $float_ty_name:ident, $dims:expr, $dims_ty:ty, $comp_ty:ty
        $(; $($lesser_comp_ty:ident),* $(,)? < $($greater_comp_ty:ident),* $(,)?)?
    ) => {
        impl_vec!($ty_name, $dims, $dims_ty, $comp_ty);

        impl Eq for $ty_name {}

        impl std::hash::Hash for $ty_name {
            fn hash<H>(&self, state: &mut H) where H: std::hash::Hasher {
                self.inner().hash(state);
            }
        }

        impl $ty_name {
            pub fn to_f32(self) -> $float_ty_name {
                let mut result = <$float_ty_name as Vector<f32>>::ZERO;

                for coord in 0..Self::DIMENSIONS {
                    result[coord] = self[coord] as f32;
                }

                result
            }

            pub fn from_f32(other: $float_ty_name) -> Self {
                let mut result = <$ty_name as Vector<_>>::ZERO;

                for coord in 0..Self::DIMENSIONS {
                    result[coord] = other[coord] as $comp_ty;
                }

                result
            }
        }

        $(
            $(
                impl From<$ty_name> for [$greater_comp_ty; $dims] {
                    fn from(vector: $ty_name) -> Self {
                        let mut result = [<$greater_comp_ty as Component>::ZERO; $dims];

                        for (result_component, vector_component) in result.iter_mut().zip(vector.iter()) {
                            *result_component = *vector_component as $greater_comp_ty;
                        }

                        result
                    }
                }

                impl TryFrom<[$greater_comp_ty; $dims]> for $ty_name {
                    type Error = TryFromIntError;

                    fn try_from(array: [$greater_comp_ty; $dims]) -> Result<Self, Self::Error> {
                        let mut result = <$ty_name as Vector<_>>::ZERO;

                        for (result_component, array_component) in result.iter_mut().zip(array.iter()) {
                            *result_component = <$comp_ty>::try_from(*array_component)?;
                        }

                        Ok(result)
                    }
                }

                // impl From<$ty_name> for $greater_ty {
                //     fn from(vector: $ty_name) -> Self {
                //         $greater_ty(<[$greater_comp_ty; $dims]>::from(vector))
                //     }
                // }
            )*

            $(
                impl From<[$lesser_comp_ty; $dims]> for $ty_name {
                    fn from(array: [$lesser_comp_ty; $dims]) -> Self {
                        let mut result = <$ty_name as Vector<_>>::ZERO;

                        for (result_component, array_component) in result.iter_mut().zip(array.iter()) {
                            *result_component = *array_component as $comp_ty;
                        }

                        result
                    }
                }

                impl TryFrom<$ty_name> for [$lesser_comp_ty; $dims] {
                    type Error = TryFromIntError;

                    fn try_from(vector: $ty_name) -> Result<Self, Self::Error> {
                        let mut result = [<$lesser_comp_ty as Component>::ZERO; $dims];

                        for (result_component, vector_component) in result.iter_mut().zip(vector.iter()) {
                            *result_component = <$lesser_comp_ty>::try_from(*vector_component)?;
                        }

                        Ok(result)
                    }
                }

                // impl From<$lesser_ty> for $ty_name {
                //     fn from(other: $lesser_ty) -> Self {
                //         <$ty_name>::from(other.0)
                //     }
                // }
            )*
        )?
    }
}

macro_rules! impl_vec_integer_signed {
    (
        $ty_name:ident, $float_ty_name:ident, $dims:expr, $dims_ty:ty, $comp_ty:ty $(,)?
        $(; $($lesser_comp_ty:ident),* $(,)? < $($greater_comp_ty:ident),* $(,)?)?
    ) => {
        impl_vec_integer!($ty_name, $float_ty_name, $dims, $dims_ty, $comp_ty$(; $($lesser_comp_ty),* < $($greater_comp_ty),*)?);
        impl_vec_signed!($ty_name, $dims, $dims_ty);
    }
}

// impl_vec_f32!(F32Vec1, 1, U1);
// impl_vec_f32!(F32Vec2, 2, U2);
// impl_vec_f32!(F32Vec3, 3, U3);
// impl_vec_f32!(F32Vec4, 4, U4);

// impl_projected_homogeneous!(F32Vec1, F32Vec2);
// impl_projected_homogeneous!(F32Vec2, F32Vec3);
// impl_projected_homogeneous!(F32Vec3, F32Vec4);

// pub type Vec1 = F32Vec1;
// pub type Vec2 = F32Vec2;
// pub type Vec3 = F32Vec3;
// pub type Vec4 = F32Vec4;

impl_vec_f32!(Vec1, 1, U1);
impl_vec_f32!(Vec2, 2, U2);
impl_vec_f32!(Vec3, 3, U3);
impl_vec_f32!(Vec4, 4, U4);

impl_projected_homogeneous!(Vec1, Vec2);
impl_projected_homogeneous!(Vec2, Vec3);
impl_projected_homogeneous!(Vec3, Vec4);

impl_vec_integer!(  U8Vec1, Vec1, 1, U1,   u8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer!(  U8Vec2, Vec2, 2, U2,   u8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer!(  U8Vec3, Vec3, 3, U3,   u8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer!(  U8Vec4, Vec4, 4, U4,   u8; < u16, i16, u32, i32, u64, i64, u128, i128);

impl_vec_integer!( U16Vec1, Vec1, 1, U1,  u16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer!( U16Vec2, Vec2, 2, U2,  u16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer!( U16Vec3, Vec3, 3, U3,  u16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer!( U16Vec4, Vec4, 4, U4,  u16; u8, i8, < u32, i32, u64, i64, u128, i128);

impl_vec_integer!( U32Vec1, Vec1, 1, U1,  u32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer!( U32Vec2, Vec2, 2, U2,  u32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer!( U32Vec3, Vec3, 3, U3,  u32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer!( U32Vec4, Vec4, 4, U4,  u32; u8, i8, u16, i16, < u64, i64, u128, i128);

impl_vec_integer!( U64Vec1, Vec1, 1, U1,  u64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer!( U64Vec2, Vec2, 2, U2,  u64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer!( U64Vec3, Vec3, 3, U3,  u64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer!( U64Vec4, Vec4, 4, U4,  u64; u8, i8, u16, i16, u32, i32, < u128, i128);

impl_vec_integer!(U128Vec1, Vec1, 1, U1, u128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer!(U128Vec2, Vec2, 2, U2, u128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer!(U128Vec3, Vec3, 3, U3, u128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer!(U128Vec4, Vec4, 4, U4, u128; u8, i8, u16, i16, u32, i32, u64, i64 <);

pub type UVec1 = U32Vec1;
pub type UVec2 = U32Vec2;
pub type UVec3 = U32Vec3;
pub type UVec4 = U32Vec4;

impl_vec_integer_signed!(  I8Vec1, Vec1, 1, U1,   i8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!(  I8Vec2, Vec2, 2, U2,   i8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!(  I8Vec3, Vec3, 3, U3,   i8; < u16, i16, u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!(  I8Vec4, Vec4, 4, U4,   i8; < u16, i16, u32, i32, u64, i64, u128, i128);

impl_vec_integer_signed!( I16Vec1, Vec1, 1, U1,  i16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!( I16Vec2, Vec2, 2, U2,  i16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!( I16Vec3, Vec3, 3, U3,  i16; u8, i8, < u32, i32, u64, i64, u128, i128);
impl_vec_integer_signed!( I16Vec4, Vec4, 4, U4,  i16; u8, i8, < u32, i32, u64, i64, u128, i128);

impl_vec_integer_signed!( I32Vec1, Vec1, 1, U1,  i32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer_signed!( I32Vec2, Vec2, 2, U2,  i32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer_signed!( I32Vec3, Vec3, 3, U3,  i32; u8, i8, u16, i16, < u64, i64, u128, i128);
impl_vec_integer_signed!( I32Vec4, Vec4, 4, U4,  i32; u8, i8, u16, i16, < u64, i64, u128, i128);

impl_vec_integer_signed!( I64Vec1, Vec1, 1, U1,  i64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer_signed!( I64Vec2, Vec2, 2, U2,  i64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer_signed!( I64Vec3, Vec3, 3, U3,  i64; u8, i8, u16, i16, u32, i32, < u128, i128);
impl_vec_integer_signed!( I64Vec4, Vec4, 4, U4,  i64; u8, i8, u16, i16, u32, i32, < u128, i128);

impl_vec_integer_signed!(I128Vec1, Vec1, 1, U1, i128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer_signed!(I128Vec2, Vec2, 2, U2, i128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer_signed!(I128Vec3, Vec3, 3, U3, i128; u8, i8, u16, i16, u32, i32, u64, i64 <);
impl_vec_integer_signed!(I128Vec4, Vec4, 4, U4, i128; u8, i8, u16, i16, u32, i32, u64, i64 <);

pub type IVec1 = I32Vec1;
pub type IVec2 = I32Vec2;
pub type IVec3 = I32Vec3;
pub type IVec4 = I32Vec4;

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

pub trait Cross {
    fn cross(&self, other: &Self) -> Self;
}

impl Cross for Vec3 {
    fn cross(&self, other: &Self) -> Self {
        let a = self;
        let b = other;

        det_copy!(Vec3([1.0, 0.0, 0.0]), Vec3([0.0, 1.0, 0.0]), Vec3([0.0, 0.0, 1.0]),
                                   a[0],                  a[1],                  a[2],
                                   b[0],                  b[1],                  b[2])
    }
}
