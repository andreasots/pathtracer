//! Common utilities shared by unit tests.
#![cfg(test)]

use std::collections::HashSet;
use std::f32;

use nalgebra::{Point3, Vector3};

use crate::bvh::aabb::{Bounded, AABB};
use crate::bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::bvh::ray::Ray;

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

/// Creates a `BoundingHierarchy` for a fixed scene structure.
pub fn build_some_bh<BH: BoundingHierarchy>() -> (Vec<UnitBox>, BH) {
    let mut boxes = generate_aligned_boxes();
    let bh = BH::build(&mut boxes);
    (boxes, bh)
}

/// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
/// expected hits, verifies, whether the ray hits only the expected shapes.
fn traverse_and_verify<BH: BoundingHierarchy>(
    ray_origin: Point3<f32>,
    ray_direction: Vector3<f32>,
    all_shapes: &Vec<UnitBox>,
    bh: &BH,
    expected_shapes: &HashSet<i32>,
) {
    let ray = Ray::new(ray_origin, ray_direction);
    let hit_shapes = bh.traverse(&ray, all_shapes);

    assert_eq!(expected_shapes.len(), hit_shapes.len());
    for shape in hit_shapes {
        assert!(expected_shapes.contains(&shape.id));
    }
}

/// Perform some fixed intersection tests on BH structures.
pub fn traverse_some_bh<BH: BoundingHierarchy>() {
    let (all_shapes, bh) = build_some_bh::<BH>();

    {
        // Define a ray which traverses the x-axis from afar.
        let origin = Point3::new(-1000.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        let mut expected_shapes = HashSet::new();

        // It should hit everything.
        for id in -10..11 {
            expected_shapes.insert(id);
        }
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
    }

    {
        // Define a ray which traverses the y-axis from afar.
        let origin = Point3::new(0.0, -1000.0, 0.0);
        let direction = Vector3::new(0.0, 1.0, 0.0);

        // It should hit only one box.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(0);
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
    }

    {
        // Define a ray which intersects the x-axis diagonally.
        let origin = Point3::new(6.0, 0.5, 0.0);
        let direction = Vector3::new(-2.0, -1.0, 0.0);

        // It should hit exactly three boxes.
        let mut expected_shapes = HashSet::new();
        expected_shapes.insert(4);
        expected_shapes.insert(5);
        expected_shapes.insert(6);
        traverse_and_verify(origin, direction, &all_shapes, &bh, &expected_shapes);
    }
}
