//! Common utilities shared by unit tests.
#![cfg(test)]

use std::f32;

use nalgebra::{Point3, Vector3};

use crate::bvh::aabb::{Bounded, AABB};
use crate::bvh::bounding_hierarchy::BHShape;

/// A vector represented as a tuple
pub type TupleVec = (f32, f32, f32);

/// Convert a `TupleVec` to a `nalgebra` point.
pub fn tuple_to_point(tpl: &TupleVec) -> Point3<f32> {
    Point3::new(tpl.0, tpl.1, tpl.2)
}

/// Define some `Bounded` structure.
pub struct UnitBox {
    pub id: i32,
    pub pos: Point3<f32>,
    node_index: usize,
}

impl UnitBox {
    pub fn new(id: i32, pos: Point3<f32>) -> UnitBox {
        UnitBox {
            id: id,
            pos: pos,
            node_index: 0,
        }
    }
}

/// `UnitBox`'s `AABB`s are unit `AABB`s centered on the box's position.
impl Bounded for UnitBox {
    fn aabb(&self) -> AABB {
        let min = self.pos + Vector3::new(-0.5, -0.5, -0.5);
        let max = self.pos + Vector3::new(0.5, 0.5, 0.5);
        AABB::with_bounds(min, max)
    }
}

impl BHShape for UnitBox {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Generate 21 `UnitBox`s along the X axis centered on whole numbers (-10,9,..,10).
/// The index is set to the rounded x-coordinate of the box center.
pub fn generate_aligned_boxes() -> Vec<UnitBox> {
    // Create 21 boxes along the x-axis
    let mut shapes = Vec::new();
    for x in -10..11 {
        shapes.push(UnitBox::new(x, Point3::new(x as f32, 0.0, 0.0)));
    }
    shapes
}
