//! Common utilities shared by unit tests.
#![cfg(test)]

use std::f32;

use nalgebra::Point3;

/// A vector represented as a tuple
pub type TupleVec = (f32, f32, f32);

/// Convert a `TupleVec` to a `nalgebra` point.
pub fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
    Point3::new(tpl.0, tpl.1, tpl.2)
}
