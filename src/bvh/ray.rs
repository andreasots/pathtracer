//! This module defines a Ray structure and intersection algorithms
//! for axis aligned bounding boxes and triangles.

use crate::bvh::aabb::AABB;
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

    /// Sign of the direction. 0 means positive, 1 means negative.
    /// Cached for use in [`AABB`] intersections.
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    sign: Vector3<usize>,
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
            sign: Vector3::new(
                (direction.x < 0.0) as usize,
                (direction.y < 0.0) as usize,
                (direction.z < 0.0) as usize,
            ),
        }
    }

    /// Tests the intersection of a [`Ray`] with an [`AABB`] using the optimized algorithm
    /// from [this paper](http://www.cs.utah.edu/~awilliam/box/box.pdf).
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::ray::Ray;
    /// use bvh::nalgebra::{Point3,Vector3};
    ///
    /// let origin = Point3::new(0.0,0.0,0.0);
    /// let direction = Vector3::new(1.0,0.0,0.0);
    /// let ray = Ray::new(origin, direction);
    ///
    /// let point1 = Point3::new(99.9,-1.0,-1.0);
    /// let point2 = Point3::new(100.1,1.0,1.0);
    /// let aabb = AABB::with_bounds(point1, point2);
    ///
    /// assert!(ray.intersects_aabb(&aabb));
    /// ```
    ///
    /// [`Ray`]: struct.Ray.html
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        let mut ray_min = (aabb[self.sign.x].x - self.origin.x) * self.inv_direction.x;
        let mut ray_max = (aabb[1 - self.sign.x].x - self.origin.x) * self.inv_direction.x;

        let y_min = (aabb[self.sign.y].y - self.origin.y) * self.inv_direction.y;
        let y_max = (aabb[1 - self.sign.y].y - self.origin.y) * self.inv_direction.y;

        if (ray_min > y_max) || (y_min > ray_max) {
            return false;
        }

        if y_min > ray_min {
            ray_min = y_min;
        }
        // Using the following solution significantly decreases the performance
        // ray_min = ray_min.max(y_min);

        if y_max < ray_max {
            ray_max = y_max;
        }
        // Using the following solution significantly decreases the performance
        // ray_max = ray_max.min(y_max);

        let z_min = (aabb[self.sign.z].z - self.origin.z) * self.inv_direction.z;
        let z_max = (aabb[1 - self.sign.z].z - self.origin.z) * self.inv_direction.z;

        if (ray_min > z_max) || (z_min > ray_max) {
            return false;
        }

        // Only required for bounded intersection intervals.
        // if z_min > ray_min {
        // ray_min = z_min;
        // }

        if z_max < ray_max {
            ray_max = z_max;
        }
        // Using the following solution significantly decreases the performance
        // ray_max = ray_max.min(y_max);

        ray_max > 0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::bvh::aabb::AABB;
    use crate::bvh::ray::Ray;
    use crate::bvh::testbase::{tuple_to_point, TupleVec};

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
            ray.intersects_aabb(&aabb)
        }
    }
}
