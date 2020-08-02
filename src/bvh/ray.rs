//! This module defines a Ray structure and intersection algorithms
//! for axis aligned bounding boxes and triangles.

use crate::bvh::AABB;
use nalgebra::{Point3, Vector3};

/// A struct which defines a ray and some of its cached values.
#[derive(Debug)]
pub struct Ray {
    /// The ray origin.
    pub origin: Point3<f32>,

    /// The ray direction.
    pub direction: Vector3<f32>,

    /// Inverse (1/x) ray direction. Cached for use in [`AABB`] intersections.
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    inv_direction: Vector3<f32>,
}

impl Ray {
    /// Creates a new [`Ray`] from an `origin` and a `direction`.
    /// `direction` will be normalized.
    ///
    /// # Examples
    /// ```
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// assert_eq!(ray.origin, origin);
    /// assert_eq!(ray.direction, direction);
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    ///
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Ray {
        let direction = direction.normalize();
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

#[cfg(test)]
mod tests {
    use crate::bvh::testbase::{tuple_to_point, TupleVec};
    use crate::bvh::{Ray, AABB};

    use quickcheck::quickcheck;

    /// Generates a random `Ray` which points at at a random `AABB`.
    fn gen_ray_to_aabb(data: (TupleVec, TupleVec, TupleVec)) -> (Ray, AABB) {
        // Generate a random AABB
        let aabb = AABB::empty()
            .grow(&tuple_to_point(&data.0))
            .grow(&tuple_to_point(&data.1));

        // Get its center
        let center = aabb.center();

        // Generate random ray pointing at the center
        let pos = tuple_to_point(&data.2);
        let ray = Ray::new(pos, center - pos);
        (ray, aabb)
    }

    // Test whether a `Ray` which points at the center of an `AABB` intersects it.
    // Uses the optimized algorithm.
    quickcheck! {
        fn test_ray_points_at_aabb_center(data: (TupleVec, TupleVec, TupleVec)) -> bool {
            let (ray, aabb) = gen_ray_to_aabb(data);
            ray.intersects_aabb(&aabb).is_some()
        }
    }
}
