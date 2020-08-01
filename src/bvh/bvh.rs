//! This module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//!
//! [`BVH`]: struct.BVH.html
//! [`BVHNode`]: struct.BVHNode.html
//!

use crate::bvh::aabb::{Bounded, AABB};
use crate::bvh::bounding_hierarchy::{Intersect, Distance};
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
        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
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

impl BVHNode {
    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf {
            shape_index: 0,
        }
    }

    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: Bounded>(
        shapes: &mut [T],
        indices: &[usize],
        nodes: &mut Vec<BVHNode>,
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
                shape_index,
            });
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
                BVHNode::build(shapes, child_l_indices, nodes);
            let child_r_index =
                BVHNode::build(shapes, child_r_indices, nodes);
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
                BVHNode::build(shapes, &child_l_indices, nodes);
            let child_r_index =
                BVHNode::build(shapes, &child_r_indices, nodes);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        assert!(!child_l_aabb.is_empty());
        assert!(!child_r_aabb.is_empty());
        nodes[node_index] = BVHNode::Node {
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
    pub fn traverse_recursive<'a, Shape: Intersect>(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &Ray,
        start: Option<&Shape>,
        shapes: &'a [Shape],
        max_distance: f32,
    ) -> Option<(&'a Shape, Shape::Intersection)> {
        match nodes[node_index] {
            BVHNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
            } => {
                let left_clip = ray.clip_aabb(child_l_aabb);
                let right_clip = ray.clip_aabb(child_r_aabb);

                match (left_clip, right_clip) {
                    (Some((l_min, l_max)), Some((r_min, r_max))) => {
                        //println!("Both: l: {:?}; r: {:?}", left_clip, right_clip);

                        let (first_child_index, first_child_max, second_child_index, second_child_min, second_child_max) = if l_min < r_min {
                            (child_l_index, l_max, child_r_index, r_min, r_max)
                        } else {
                            (child_r_index, r_max, child_l_index, l_min, l_max)
                        };

                        let first_intersection = BVHNode::traverse_recursive(nodes, first_child_index, ray, start, shapes, first_child_max.min(max_distance));
                        let first_distance = first_intersection.as_ref().map(|(_, intersection)| intersection.distance()).unwrap_or(f32::INFINITY);

                        if first_distance > second_child_min {
                            let second_intersection = BVHNode::traverse_recursive(nodes, second_child_index, ray, start, shapes, first_distance.min(max_distance).min(second_child_max));
                            let second_distance = second_intersection.as_ref().map(|(_, intersection)| intersection.distance()).unwrap_or(f32::INFINITY);

                            if first_distance < second_distance {
                                first_intersection
                            } else {
                                second_intersection
                            }
                        } else {
                            first_intersection
                        }
                    }
                    (Some((_, clip_max)), None) => {
                        BVHNode::traverse_recursive(nodes, child_l_index, ray, start, shapes, max_distance.min(clip_max))
                    }
                    (None, Some((_, clip_max))) => {
                        BVHNode::traverse_recursive(nodes, child_r_index, ray, start, shapes, max_distance.min(clip_max))
                    }
                    (None, None) => None,
                }
            }
            BVHNode::Leaf { shape_index } => {
                let shape = &shapes[shape_index];
                let start_ptr = start.map(|s| s as *const _).unwrap_or_else(std::ptr::null);
                if shape as *const _ != start_ptr {
                    // TODO: use `max_distance`
                    shape.intersect(ray, f32::INFINITY).map(|intersection| (shape, intersection))
                } else {
                    None
                }
            }
        }
    }
}

/// The [`BVH`] data structure. Contains the list of [`BVHNode`]s.
///
/// [`BVH`]: struct.BVH.html
///
pub struct BVH {
    /// The list of nodes of the [`BVH`].Vec<&Shape>
    ///
    pub nodes: Vec<BVHNode>,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes` slice.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn build<Shape: Bounded>(shapes: &mut [Shape]) -> BVH {
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2;
        let mut nodes = Vec::with_capacity(expected_node_count);
        BVHNode::build(shapes, &indices, &mut nodes);
        BVH { nodes }
    }

    /// Traverses the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse<'a, Shape: Intersect>(&'a self, ray: &Ray, start: Option<&Shape>, shapes: &'a [Shape]) -> Option<(&'a Shape, Shape::Intersection)> {
        BVHNode::traverse_recursive(&self.nodes, 0, ray, start, shapes, f32::INFINITY)
    }
}
