use crate::bvh::ray::Ray;

pub trait Distance {
    fn distance(&self) -> f32;
}

pub trait Intersect {
    type Intersection: Distance;

    fn intersect(&self, ray: &Ray, max_distance: f32) -> Option<Self::Intersection>;
}
