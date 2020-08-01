//! Axis Aligned Bounding Boxes.

use nalgebra::{Point3, Vector3};

/// AABB struct.
#[derive(Debug, Copy, Clone)]
pub struct AABB {
    /// Minimum coordinates
    pub min: Point3<f32>,

    /// Maximum coordinates
    pub max: Point3<f32>,
}

/// A trait implemented by things which can be bounded by an [`AABB`].
///
/// [`AABB`]: struct.AABB.html
///
pub trait Bounded {
    /// Returns the geometric bounds of this object in the form of an [`AABB`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::{AABB, Bounded};
    /// use bvh::nalgebra::Point3;
    ///
    /// struct Something;
    ///
    /// impl Bounded for Something {
    ///     fn aabb(&self) -> AABB {
    ///         let point1 = Point3::new(0.0,0.0,0.0);
    ///         let point2 = Point3::new(1.0,1.0,1.0);
    ///         AABB::with_bounds(point1, point2)
    ///     }
    /// }
    ///
    /// let something = Something;
    /// let aabb = something.aabb();
    ///
    /// assert!(aabb.contains(&Point3::new(0.0,0.0,0.0)));
    /// assert!(aabb.contains(&Point3::new(1.0,1.0,1.0)));
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    fn aabb(&self) -> AABB;
}

impl AABB {
    /// Creates a new [`AABB`] with the given bounds.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// assert_eq!(aabb.min.x, -1.0);
    /// assert_eq!(aabb.max.z, 1.0);
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn with_bounds(min: Point3<f32>, max: Point3<f32>) -> AABB {
        AABB { min, max }
    }

    /// Creates a new empty [`AABB`].
    ///
    /// # Examples
    /// ```
    /// # extern crate bvh;
    /// # extern crate rand;
    /// use bvh::aabb::AABB;
    ///
    /// # fn main() {
    /// let aabb = AABB::empty();
    /// let min = &aabb.min;
    /// let max = &aabb.max;
    ///
    /// // For any point
    /// let x = rand::random();
    /// let y = rand::random();
    /// let z = rand::random();
    ///
    /// // An empty AABB should not contain it
    /// assert!(x < min.x && y < min.y && z < min.z);
    /// assert!(max.x < x && max.y < y && max.z < z);
    /// # }
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn empty() -> AABB {
        AABB {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Returns a new minimal [`AABB`] which contains both this [`AABB`] and `other`.
    /// The result is the convex hull of the both [`AABB`]s.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let aabb1 = AABB::with_bounds(Point3::new(-101.0, 0.0, 0.0), Point3::new(-100.0, 1.0, 1.0));
    /// let aabb2 = AABB::with_bounds(Point3::new(100.0, 0.0, 0.0), Point3::new(101.0, 1.0, 1.0));
    /// let joint = aabb1.join(&aabb2);
    ///
    /// let point_inside_aabb1 = Point3::new(-100.5, 0.5, 0.5);
    /// let point_inside_aabb2 = Point3::new(100.5, 0.5, 0.5);
    /// let point_inside_joint = Point3::new(0.0, 0.5, 0.5);
    ///
    /// # assert!(aabb1.contains(&point_inside_aabb1));
    /// # assert!(!aabb1.contains(&point_inside_aabb2));
    /// # assert!(!aabb1.contains(&point_inside_joint));
    /// #
    /// # assert!(!aabb2.contains(&point_inside_aabb1));
    /// # assert!(aabb2.contains(&point_inside_aabb2));
    /// # assert!(!aabb2.contains(&point_inside_joint));
    ///
    /// assert!(joint.contains(&point_inside_aabb1));
    /// assert!(joint.contains(&point_inside_aabb2));
    /// assert!(joint.contains(&point_inside_joint));
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
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

    /// Mutable version of [`AABB::join`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::{Point3, Vector3};
    ///
    /// let size = Vector3::new(1.0, 1.0, 1.0);
    /// let aabb_pos = Point3::new(-101.0, 0.0, 0.0);
    /// let mut aabb = AABB::with_bounds(aabb_pos, aabb_pos + size);
    ///
    /// let other_pos = Point3::new(100.0, 0.0, 0.0);
    /// let other = AABB::with_bounds(other_pos, other_pos + size);
    ///
    /// let point_inside_aabb = aabb_pos + size / 2.0;
    /// let point_inside_other = other_pos + size / 2.0;
    /// let point_inside_joint = Point3::new(0.0, 0.0, 0.0) + size / 2.0;
    ///
    /// # assert!(aabb.contains(&point_inside_aabb));
    /// # assert!(!aabb.contains(&point_inside_other));
    /// # assert!(!aabb.contains(&point_inside_joint));
    /// #
    /// # assert!(!other.contains(&point_inside_aabb));
    /// # assert!(other.contains(&point_inside_other));
    /// # assert!(!other.contains(&point_inside_joint));
    ///
    /// aabb.join_mut(&other);
    ///
    /// assert!(aabb.contains(&point_inside_aabb));
    /// assert!(aabb.contains(&point_inside_other));
    /// assert!(aabb.contains(&point_inside_joint));
    /// ```
    ///
    /// [`AABB::join`]: struct.AABB.html
    ///
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

    /// Returns a new minimal [`AABB`] which contains both
    /// this [`AABB`] and the [`Point3`] `other`.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let point1 = Point3::new(0.0, 0.0, 0.0);
    /// let point2 = Point3::new(1.0, 1.0, 1.0);
    /// let point3 = Point3::new(2.0, 2.0, 2.0);
    ///
    /// let aabb = AABB::empty();
    /// assert!(!aabb.contains(&point1));
    ///
    /// let aabb1 = aabb.grow(&point1);
    /// assert!(aabb1.contains(&point1));
    ///
    /// let aabb2 = aabb.grow(&point2);
    /// assert!(aabb2.contains(&point2));
    /// assert!(!aabb2.contains(&point3));
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    /// [`Point3`]: http://nalgebra.org/doc/nalgebra/struct.Point3.html
    ///
    pub fn grow(&self, other: &Point3<f32>) -> AABB {
        AABB::with_bounds(
            Point3::new(
                self.min.x.min(other.x),
                self.min.y.min(other.y),
                self.min.z.min(other.z),
            ),
            Point3::new(
                self.max.x.max(other.x),
                self.max.y.max(other.y),
                self.max.z.max(other.z),
            ),
        )
    }

    /// Returns the size of this [`AABB`] in all three dimensions.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let aabb = AABB::with_bounds(Point3::new(-1.0,-1.0,-1.0), Point3::new(1.0,1.0,1.0));
    /// let size = aabb.size();
    /// assert!(size.x == 2.0 && size.y == 2.0 && size.z == 2.0);
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn size(&self) -> Vector3<f32> {
        self.max - self.min
    }

    /// Returns the center [`Point3`] of the [`AABB`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let center = aabb.center();
    /// assert!(center.x == 42.0 && center.y == 42.0 && center.z == 42.0);
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    /// [`Point3`]: http://nalgebra.org/doc/nalgebra/struct.Point3.html
    ///
    pub fn center(&self) -> Point3<f32> {
        self.min + (self.size() / 2.0)
    }

    /// An empty [`AABB`] is an [`AABB`] where the lower bound is greater than
    /// the upper bound in at least one component
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let empty_aabb = AABB::empty();
    /// assert!(empty_aabb.is_empty());
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// assert!(!aabb.is_empty());
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Returns the total surface area of this [`AABB`].
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::nalgebra::Point3;
    ///
    /// let min = Point3::new(41.0,41.0,41.0);
    /// let max = Point3::new(43.0,43.0,43.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let surface_area = aabb.surface_area();
    /// assert!(surface_area == 24.0);
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.x * size.z + size.y * size.z)
    }

    /// Returns the axis along which the [`AABB`] is stretched the most.
    ///
    /// # Examples
    /// ```
    /// use bvh::aabb::AABB;
    /// use bvh::axis::Axis;
    /// use bvh::nalgebra::Point3;
    ///
    /// let min = Point3::new(-100.0,0.0,0.0);
    /// let max = Point3::new(100.0,0.0,0.0);
    ///
    /// let aabb = AABB::with_bounds(min, max);
    /// let axis = aabb.largest_axis();
    /// assert!(axis == Axis::X);
    /// ```
    ///
    /// [`AABB`]: struct.AABB.html
    ///
    pub fn largest_axis(&self) -> usize {
        self.size().argmax().0
    }
}

/// Default instance for [`AABB`]s. Returns an [`AABB`] which is [`empty()`].
///
/// [`AABB`]: struct.AABB.html
/// [`empty()`]: #method.empty
///
impl Default for AABB {
    fn default() -> AABB {
        AABB::empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::bvh::aabb::AABB;
    use crate::bvh::testbase::{tuple_to_point, TupleVec};
    use crate::bvh::EPSILON;

    use nalgebra::Vector3;
    use quickcheck::quickcheck;

    // Test whether the surface of a nonempty AABB is always positive.
    quickcheck! {
        fn test_surface_always_positive(a: TupleVec, b: TupleVec) -> bool {
            let aabb = AABB::empty()
                .grow(&tuple_to_point(&a))
                .grow(&tuple_to_point(&b));
            aabb.surface_area() >= 0.0
        }
    }

    // Compute and compare the surface area of an AABB by hand.
    quickcheck! {
        fn test_surface_area_cube(pos: TupleVec, size: f32) -> bool {
            // Generate some non-empty AABB
            let pos = tuple_to_point(&pos);
            let size_vec = Vector3::new(size, size, size);
            let aabb = AABB::with_bounds(pos, pos + size_vec);

            // Check its surface area
            let area_a = aabb.surface_area();
            let area_b = 6.0 * size * size;
            (1.0 - (area_a / area_b)).abs() < EPSILON
        }
    }
}
