use crate::bvh::ray::Ray;
use approx::relative_eq;
use nalgebra::{Point2, Point3, Vector3};

use crate::bvh::aabb::{Bounded, AABB};
use crate::bvh::bounding_hierarchy::{Distance, Intersect};

#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,

    uv: Option<[Point2<f32>; 3]>,
    vertex_normals: Option<[Vector3<f32>; 3]>,

    material_index: usize,

    n: Vector3<f32>,
    d: f32,

    n1: Vector3<f32>,
    d1: f32,

    n2: Vector3<f32>,
    d2: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Intersection {
    pub distance: f32,
    pub u: f32,
    pub v: f32,
    pub normal: Vector3<f32>,
}

impl Distance for Intersection {
    fn distance(&self) -> f32 {
        self.distance
    }
}

impl Triangle {
    pub fn new(
        a: Point3<f32>,
        b: Point3<f32>,
        c: Point3<f32>,
        uv: Option<[Point2<f32>; 3]>,
        vertex_normals: Option<[Vector3<f32>; 3]>,
        material_index: usize,
    ) -> Triangle {
        let ab = b - a;
        let ac = c - a;

        let n = ab.cross(&ac);
        let d = a.coords.dot(&n);

        let n1 = ac.cross(&n) / n.norm_squared();
        let d1 = -a.coords.dot(&n1);

        let n2 = n.cross(&ab) / n.norm_squared();
        let d2 = -a.coords.dot(&n2);

        Triangle {
            a,
            b,
            c,

            uv,
            vertex_normals,

            material_index,

            n,
            d,
            n1,
            d1,
            n2,
            d2,
        }
    }

    pub fn texture_coords(&self) -> Option<[Point2<f32>; 3]> {
        self.uv
    }

    pub fn material_index(&self) -> usize {
        self.material_index
    }

    pub fn vertex_normals(&self) -> Option<[Vector3<f32>; 3]> {
        self.vertex_normals
    }
}

impl Intersect for Triangle {
    type Intersection = Intersection;

    /// Yet Faster Ray-Triangle Intersection (Using SSE4) by Jiřı́ Havel and Adam Herout
    fn intersect(&self, ray: &Ray, max_distance: f32) -> Option<Intersection> {
        // Everything is scaled by `det` to avoid the division.
        let det = ray.direction.dot(&self.n);
        if det == 0.0 {
            return None;
        }

        let t = self.d - ray.origin.coords.dot(&self.n);

        if (max_distance * det - t).is_sign_positive() != t.is_sign_positive() {
            return None;
        }

        let p = ray.origin.coords * det + ray.direction * t;
        let u = p.dot(&self.n1) + det * self.d1;
        if (det - u).is_sign_positive() != u.is_sign_positive() {
            return None;
        }

        let v = p.dot(&self.n2) + det * self.d2;
        if (det - u - v).is_sign_positive() != v.is_sign_positive() {
            return None;
        }

        let inv_det = det.recip();

        let distance = t * inv_det;

        if relative_eq!(distance, 0.0) {
            return None;
        }

        Some(Intersection {
            distance,
            normal: self.n,
            u: u * inv_det,
            v: v * inv_det,
        })
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        AABB::with_bounds(self.a, self.a)
            .grow(&self.b)
            .grow(&self.c)
    }
}

#[cfg(test)]
mod test {
    use super::Triangle;
    use crate::bvh::bounding_hierarchy::Intersect;
    use crate::bvh::ray::Ray;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn intersection() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
            None,
            None,
            0,
        );

        let ray = Ray::new(
            Point3::new(1.0 / 3.0, 1.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        );

        let intersection = tri.intersect(&ray, f32::INFINITY).expect("no intersection");

        assert_eq!(intersection.distance, 1.0);
        assert_eq!(intersection.u, 1.0 / 3.0);
        assert_eq!(intersection.v, 1.0 / 3.0);
    }

    #[test]
    fn miss_t_negative() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 2.0),
            Point3::new(1.0, 0.0, 2.0),
            Point3::new(0.0, 1.0, 2.0),
            None,
            None,
            0,
        );

        let ray = Ray::new(
            Point3::new(2.0 / 3.0, 2.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let intersection = tri.intersect(&ray, f32::INFINITY);
        assert!(intersection.is_none(), "intersected at {:?}", intersection);
    }

    #[test]
    fn miss_u_plus_v_greater_than_one() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
            None,
            None,
            0,
        );

        let ray = Ray::new(
            Point3::new(2.0 / 3.0, 2.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let intersection = tri.intersect(&ray, f32::INFINITY);
        assert!(intersection.is_none(), "intersected at {:?}", intersection);
    }

    #[test]
    fn miss_u_negative() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
            None,
            None,
            0,
        );

        let ray = Ray::new(
            Point3::new(-2.0 / 3.0, 2.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let intersection = tri.intersect(&ray, f32::INFINITY);
        assert!(intersection.is_none(), "intersected at {:?}", intersection);
    }

    #[test]
    fn miss_v_negative() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(0.0, 1.0, -1.0),
            None,
            None,
            0,
        );

        let ray = Ray::new(
            Point3::new(2.0 / 3.0, -2.0 / 3.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let intersection = tri.intersect(&ray, f32::INFINITY);
        assert!(intersection.is_none(), "intersected at {:?}", intersection);
    }
}
