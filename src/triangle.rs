use approx::relative_eq;
use bvh::nalgebra::{Point2, Vector3};
use bvh::ray::Ray;

use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::nalgebra::Point3;

#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,

    uv: Option<[Point2<f32>; 3]>,
    vertex_normals: Option<[Vector3<f32>; 3]>,

    bh_node_index: usize,
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

            bh_node_index: usize::MAX,
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

    /// Yet Faster Ray-Triangle Intersection (Using SSE4) by Jiřı́ Havel and Adam Herout
    pub fn intersect(&self, ray: &Ray, max_distance: f32) -> Option<Intersection> {
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

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, i: usize) {
        self.bh_node_index = i;
    }

    fn bh_node_index(&self) -> usize {
        self.bh_node_index
    }
}

#[cfg(test)]
mod test {
    use super::Triangle;
    use approx::assert_relative_eq;
    use bvh::nalgebra::{Point3, Vector3};
    use bvh::ray::Ray;
    use quickcheck_macros::quickcheck;

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
            Vector3::new(0.0, 0.0, 1.0),
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

    #[quickcheck]
    fn fast_triangle_is_equivalent_to_slow_triangle(a: (f32, f32), b: (f32, f32)) -> bool {
        let a = Point3::new(a.0.sin() * a.1.cos(), a.0.sin() * a.1.sin(), a.0.cos());
        let b = Point3::new(b.0.sin() * b.1.cos(), b.0.sin() * b.1.sin(), b.0.cos());
        let dir = b - a;
        let ray = Ray::new(a, dir);

        let triangle = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(
                (2.0 * std::f32::consts::FRAC_PI_3).cos(),
                (2.0 * std::f32::consts::FRAC_PI_3).sin(),
                0.0,
            ),
            Point3::new(
                (4.0 * std::f32::consts::FRAC_PI_3).cos(),
                (4.0 * std::f32::consts::FRAC_PI_3).sin(),
                0.0,
            ),
            None,
            None,
            0,
        );

        let intersection = ray.intersects_triangle(&triangle.a, &triangle.b, &triangle.c);
        let fast_intersection = triangle.intersect(&ray, f32::INFINITY);

        if let Some(fast_intersection) = fast_intersection {
            assert_eq!(intersection.distance, fast_intersection.distance);
            assert_relative_eq!(intersection.u, fast_intersection.u);
            assert_relative_eq!(intersection.v, fast_intersection.v);
        } else {
            assert_eq!(intersection.distance, f32::INFINITY);
        }

        true
    }
}
