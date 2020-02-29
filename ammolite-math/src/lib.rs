#![feature(const_if_match)]
#![feature(const_loop)]

#[macro_use]
pub mod ops;
#[macro_use]
pub mod macros;
#[macro_use]
pub mod vector;
#[macro_use]
pub mod matrix;

pub use vector::*;
pub use matrix::*;
pub use ops::{DivEuclid, RemEuclid};
