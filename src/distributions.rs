use bvh::nalgebra::Vector3;
use rand::Rng;
use rand_distr::Distribution;

pub struct CosineWeightedHemisphere;

impl Distribution<Vector3<f32>> for CosineWeightedHemisphere {
    fn sample<R>(&self, rng: &mut R) -> Vector3<f32>
    where
        R: Rng + ?Sized,
    {
        let r1 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();
        let r2 = rng.gen::<f32>();
        let r2s = r2.sqrt();

        let (sin_r1, cos_r1) = r1.sin_cos();

        Vector3::new(cos_r1 * r2s, sin_r1 * r2s, (1.0 - r2).sqrt())
    }
}
