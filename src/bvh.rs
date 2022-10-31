/*
    MIT License

    Copyright (c) 2016 Sven-Hendrik Haase

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/// This is the [bvh](https://github.com/svenstaro/bvh) crate at the commit a16aaa560244f6470b1ca681cda2579d9ba2c74e
/// but stripped down and heavily modified to suit better large scenes.

const EPSILON: f32 = 0.00001;

use nalgebra::{Point3, Vector3};

#[derive(Debug, Copy, Clone)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

pub trait Bounded {
    fn aabb(&self) -> AABB;
}

impl AABB {
    pub fn with_bounds(min: Point3<f32>, max: Point3<f32>) -> AABB {
        AABB { min, max }
    }

    pub fn empty() -> AABB {
        AABB {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn join(&self, other: &AABB) -> AABB {
        AABB::with_bounds(
            Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }

    pub fn join_mut(&mut self, other: &AABB) {
        self.min = Point3::new(
            self.min.x.min(other.min.x),
            self.min.y.min(other.min.y),
            self.min.z.min(other.min.z),
        );
        self.max = Point3::new(
            self.max.x.max(other.max.x),
            self.max.y.max(other.max.y),
            self.max.z.max(other.max.z),
        );
    }

    pub fn grow(&self, other: &Point3<f32>) -> AABB {
        AABB::with_bounds(
            Point3::new(self.min.x.min(other.x), self.min.y.min(other.y), self.min.z.min(other.z)),
            Point3::new(self.max.x.max(other.x), self.max.y.max(other.y), self.max.z.max(other.z)),
        )
    }

    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    pub fn center(&self) -> Point3<f32> {
        self.min + (self.size() / 2.0)
    }

    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    pub fn largest_axis(&self) -> usize {
        self.size().argmax().0
    }
}

use std::f32;

pub trait Distance {
    fn distance(&self) -> f32;
}

pub trait Intersect {
    type Intersection: Distance;

    fn intersect(&self, ray: &Ray, max_distance: f32) -> Option<Self::Intersection>;
}

#[derive(Debug, Copy, Clone)]
pub enum BVHNode {
    Leaf { shape_index: usize },
    Node { child_l_index: usize, child_l_aabb: AABB, child_r_index: usize, child_r_aabb: AABB },
}

impl BVHNode {
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf { shape_index: 0 }
    }

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
            (convex_hull_aabbs.join(shape_aabb), convex_hull_centroids.grow(center))
        }

        let mut convex_hull = (AABB::empty(), AABB::empty());
        for index in indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BVHNode::Leaf { shape_index });
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
            let child_l_index = BVHNode::build(shapes, child_l_indices, nodes);
            let child_r_index = BVHNode::build(shapes, child_r_indices, nodes);
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
            let child_l_index = BVHNode::build(shapes, &child_l_indices, nodes);
            let child_r_index = BVHNode::build(shapes, &child_r_indices, nodes);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        assert!(!child_l_aabb.is_empty());
        assert!(!child_r_aabb.is_empty());
        nodes[node_index] =
            BVHNode::Node { child_l_aabb, child_l_index, child_r_aabb, child_r_index };

        node_index
    }

    pub fn traverse_recursive<'a, Shape: Intersect>(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &Ray,
        start: Option<&Shape>,
        shapes: &'a [Shape],
        max_distance: f32,
    ) -> Option<(&'a Shape, Shape::Intersection)> {
        match nodes[node_index] {
            BVHNode::Node { ref child_l_aabb, child_l_index, ref child_r_aabb, child_r_index } => {
                let left_clip = ray.intersects_aabb(child_l_aabb);
                let right_clip = ray.intersects_aabb(child_r_aabb);

                match (left_clip, right_clip) {
                    (Some(l_min), Some(r_min)) => {
                        let (first_child_index, second_child_index, second_child_min) =
                            if l_min < r_min {
                                (child_l_index, child_r_index, r_min)
                            } else {
                                (child_r_index, child_l_index, l_min)
                            };

                        let first_intersection = BVHNode::traverse_recursive(
                            nodes,
                            first_child_index,
                            ray,
                            start,
                            shapes,
                            max_distance,
                        );
                        let first_distance = first_intersection
                            .as_ref()
                            .map(|(_, intersection)| intersection.distance())
                            .unwrap_or(f32::INFINITY);

                        if first_distance > second_child_min {
                            let second_intersection = BVHNode::traverse_recursive(
                                nodes,
                                second_child_index,
                                ray,
                                start,
                                shapes,
                                first_distance.min(max_distance),
                            );
                            let second_distance = second_intersection
                                .as_ref()
                                .map(|(_, intersection)| intersection.distance())
                                .unwrap_or(f32::INFINITY);

                            if first_distance < second_distance {
                                first_intersection
                            } else {
                                second_intersection
                            }
                        } else {
                            first_intersection
                        }
                    }
                    (Some(_), None) => BVHNode::traverse_recursive(
                        nodes,
                        child_l_index,
                        ray,
                        start,
                        shapes,
                        max_distance,
                    ),
                    (None, Some(_)) => BVHNode::traverse_recursive(
                        nodes,
                        child_r_index,
                        ray,
                        start,
                        shapes,
                        max_distance,
                    ),
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

pub struct BVH {
    pub nodes: Vec<BVHNode>,
}

impl BVH {
    pub fn build<Shape: Bounded>(shapes: &mut [Shape]) -> BVH {
        if shapes.len() > 0 {
            let indices = (0..shapes.len()).collect::<Vec<usize>>();
            let expected_node_count = shapes.len() * 2;
            let mut nodes = Vec::with_capacity(expected_node_count);
            BVHNode::build(shapes, &indices, &mut nodes);
            BVH { nodes }
        } else {
            BVH { nodes: Vec::new() }
        }
    }

    pub fn traverse<'a, Shape: Intersect>(
        &'a self,
        ray: &Ray,
        start: Option<&Shape>,
        shapes: &'a [Shape],
    ) -> Option<(&'a Shape, Shape::Intersection)> {
        if self.nodes.len() > 0 {
            BVHNode::traverse_recursive(&self.nodes, 0, ray, start, shapes, f32::INFINITY)
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
    pub inv_direction: Vector3<f32>,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Ray {
        Ray {
            origin,
            direction,
            inv_direction: Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z),
        }
    }

    /// Implementation of the algorithm described [here]
    /// (https://tavianator.com/fast-branchless-raybounding-box-intersections/).
    pub fn intersects_aabb(&self, aabb: &AABB) -> Option<f32> {
        let t1 = (aabb.min - self.origin).component_mul(&self.inv_direction);
        let t2 = (aabb.max - self.origin).component_mul(&self.inv_direction);

        let tmin = t1[0].min(t2[0]);
        let tmax = t1[0].max(t2[0]);

        let tmin = tmin.max(t1[1].min(t2[1]));
        let tmax = tmax.min(t1[1].max(t2[1]));

        let tmin = tmin.max(t1[2].min(t2[2]));
        let tmax = tmax.min(t1[2].max(t2[2]));

        let tmin = tmin.max(0.0);

        if tmax >= tmin {
            Some(tmin)
        } else {
            None
        }
    }
}

pub fn concatenate_vectors<T: Sized>(vectors: &mut [Vec<T>]) -> Vec<T> {
    let mut result = Vec::new();
    for mut vector in vectors.iter_mut() {
        result.append(&mut vector);
    }
    result
}
#[derive(Copy, Clone)]
pub struct Bucket {
    pub size: usize,
    pub aabb: AABB,
}

impl Bucket {
    pub fn empty() -> Bucket {
        Bucket { size: 0, aabb: AABB::empty() }
    }

    pub fn add_aabb(&mut self, aabb: &AABB) {
        self.size += 1;
        self.aabb = self.aabb.join(aabb);
    }

    pub fn join_bucket(a: Bucket, b: &Bucket) -> Bucket {
        Bucket { size: a.size + b.size, aabb: a.aabb.join(&b.aabb) }
    }
}

pub fn joint_aabb_of_shapes<Shape: Bounded>(indices: &[usize], shapes: &[Shape]) -> AABB {
    let mut aabb = AABB::empty();
    for index in indices {
        let shape = &shapes[*index];
        aabb.join_mut(&shape.aabb());
    }
    aabb
}
