//! This module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//!
//! [`BVH`]: struct.BVH.html
//! [`BVHNode`]: struct.BVHNode.html
//!

use crate::bvh::aabb::{Bounded, AABB};
use crate::bvh::bounding_hierarchy::BHShape;
use crate::bvh::ray::Ray;
use crate::bvh::utils::{concatenate_vectors, joint_aabb_of_shapes, Bucket};
use crate::bvh::EPSILON;
use std::f32;

/// The [`BVHNode`] enum that describes a node in a [`BVH`].
/// It's either a leaf node and references a shape (by holding its index)
/// or a regular node that has two child nodes.
/// The non-leaf node stores the [`AABB`]s of its children.
///
/// [`AABB`]: ../aabb/struct.AABB.html
/// [`BVH`]: struct.BVH.html
/// [`BVH`]: struct.BVHNode.html
///
#[derive(Debug, Copy, Clone)]
pub enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// Index of the left subtree's root node.
        child_l_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_l.
        child_l_aabb: AABB,

        /// Index of the right subtree's root node.
        child_r_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_r.
        child_r_aabb: AABB,
    },
}

impl PartialEq for BVHNode {
    // TODO Consider also comparing AABBs
    fn eq(&self, other: &BVHNode) -> bool {
        match (self, other) {
            (
                &BVHNode::Node {
                    parent_index: self_parent_index,
                    depth: self_depth,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BVHNode::Node {
                    parent_index: other_parent_index,
                    depth: other_depth,
                    child_l_index: other_child_l_index,
                    child_r_index: other_child_r_index,
                    ..
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_depth == other_depth
                    && self_child_l_index == other_child_l_index
                    && self_child_r_index == other_child_r_index
            }
            (
                &BVHNode::Leaf {
                    parent_index: self_parent_index,
                    depth: self_depth,
                    shape_index: self_shape_index,
                },
                &BVHNode::Leaf {
                    parent_index: other_parent_index,
                    depth: other_depth,
                    shape_index: other_shape_index,
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_depth == other_depth
                    && self_shape_index == other_shape_index
            }
            _ => false,
        }
    }
}

impl BVHNode {
    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf {
            parent_index: 0,
            depth: 0,
            shape_index: 0,
        }
    }

    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: BHShape + Bounded>(
        shapes: &mut [T],
        indices: &[usize],
        nodes: &mut Vec<BVHNode>,
        parent_index: usize,
        depth: u32,
    ) -> usize {
        // Helper function to accumulate the AABB joint and the centroids AABB
        fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            (
                convex_hull_aabbs.join(shape_aabb),
                convex_hull_centroids.grow(center),
            )
        }

        let mut convex_hull = Default::default();
        for index in indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BVHNode::Leaf {
                parent_index,
                depth,
                shape_index,
            });
            // Let the shape know the index of the node that represents it.
            shapes[shape_index].set_bh_node_index(node_index);
            return node_index;
        }

        // From here on we handle the recursive case. This dummy is required, because the children
        // must know their parent, and it's easier to update one parent node than the child nodes.
        let node_index = nodes.len();
        nodes.push(BVHNode::create_dummy());

        // Find the axis along which the shapes are spread the most.
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // The following `if` partitions `indices` for recursively calling `BVH::build`.
        let (child_l_index, child_l_aabb, child_r_index, child_r_aabb) = if split_axis_size
            < EPSILON
        {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at(indices.len() / 2);
            let child_l_aabb = joint_aabb_of_shapes(child_l_indices, shapes);
            let child_r_aabb = joint_aabb_of_shapes(child_r_indices, shapes);

            // Proceed recursively.
            let child_l_index =
                BVHNode::build(shapes, child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BVHNode::build(shapes, child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        } else {
            // Create six `Bucket`s, and six index assignment vector.
            const NUM_BUCKETS: usize = 6;
            let mut buckets = [Bucket::empty(); NUM_BUCKETS];
            let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number.
                let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = f32::INFINITY;
            let mut child_l_aabb = AABB::empty();
            let mut child_r_aabb = AABB::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

                let cost = (child_l.size as f32 * child_l.aabb.surface_area()
                    + child_r.size as f32 * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_r_aabb = child_r.aabb;
                }
            }

            // Join together all index buckets.
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);
            let child_l_indices = concatenate_vectors(l_assignments);
            let child_r_indices = concatenate_vectors(r_assignments);

            // Proceed recursively.
            let child_l_index =
                BVHNode::build(shapes, &child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BVHNode::build(shapes, &child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        assert!(!child_l_aabb.is_empty());
        assert!(!child_r_aabb.is_empty());
        nodes[node_index] = BVHNode::Node {
            parent_index,
            depth,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        };

        node_index
    }

    /// Traverses the [`BVH`] recursively and returns all shapes whose [`AABB`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: struct.BVH.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &Ray,
        indices: &mut Vec<usize>,
    ) {
        match nodes[node_index] {
            BVHNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                if ray.intersects_aabb(child_l_aabb) {
                    BVHNode::traverse_recursive(nodes, child_l_index, ray, indices);
                }
                if ray.intersects_aabb(child_r_aabb) {
                    BVHNode::traverse_recursive(nodes, child_r_index, ray, indices);
                }
            }
            BVHNode::Leaf { shape_index, .. } => {
                indices.push(shape_index);
            }
        }
    }
}

/// The [`BVH`] data structure. Contains the list of [`BVHNode`]s.
///
/// [`BVH`]: struct.BVH.html
///
pub struct BVH {
    /// The list of nodes of the [`BVH`].
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub nodes: Vec<BVHNode>,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes` slice.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn build<Shape: BHShape + Bounded>(shapes: &mut [Shape]) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2;
        let mut nodes = Vec::with_capacity(expected_node_count);
        BVHNode::build(shapes, &indices, &mut nodes, 0, 0);
        BVH { nodes }
    }

    /// Traverses the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse<'a, Shape>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape> {
        let mut indices = Vec::new();
        BVHNode::traverse_recursive(&self.nodes, 0, ray, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use crate::bvh::bvh::BVH;
    use crate::bvh::testbase::{generate_aligned_boxes, UnitBox};
    use nalgebra::{Point3, Vector3};
    use std::collections::HashSet;
    use crate::bvh::ray::Ray;

    /// Creates a `BoundingHierarchy` for a fixed scene structure.
    fn build_some_bh() -> (Vec<UnitBox>, BVH) {
        let mut boxes = generate_aligned_boxes();
        let bh = BVH::build(&mut boxes);
        (boxes, bh)
    }

    /// Given a ray, a bounding hierarchy, the complete list of shapes in the scene and a list of
    /// expected hits, verifies, whether the ray hits only the expected shapes.
    fn traverse_and_verify(
        ray_origin: Point3<f32>,
        ray_direction: Vector3<f32>,
        all_shapes: &Vec<UnitBox>,
        bh: &BVH,
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
    pub fn traverse_some_bh() {
        let (all_shapes, bh) = build_some_bh();

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


    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        build_some_bh();
    }

    #[test]
    /// Runs some primitive tests for intersections of a ray with a fixed scene given as a BVH.
    fn test_traverse_bvh() {
        traverse_some_bh();
    }
}
