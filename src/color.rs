use image::{Primitive, Rgba};
use nalgebra::{Matrix3, Vector3, SVector};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul};

fn to_linear_srgb(u: f32) -> f32 {
    if u <= 0.04045 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SRGB {}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum XYZ {}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color<S>(Vector3<f32>, PhantomData<S>);

impl Color<SRGB> {
    #[inline]
    pub fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self(Vector3::new(r, g, b), PhantomData)
    }

    pub fn r(&self) -> f32 {
        self.0.x
    }

    pub fn g(&self) -> f32 {
        self.0.y
    }

    pub fn b(&self) -> f32 {
        self.0.z
    }

    pub fn reflectance_at_one(&self, wavelength: f32) -> f32 {
        const TABLE: &[[f32; 3]] = &include!(concat!(env!("OUT_DIR"), "/srgb2reflectance.rs"));

        let offset = wavelength - 360.0;
        let index = offset.floor();
        let alpha = offset - index;
        let index = index as usize;

        let base_reflectance_0 = Vector3::from(TABLE.get(index).copied().unwrap_or([0.0; 3]));
        let base_reflectance_1 = Vector3::from(TABLE.get(index + 1).copied().unwrap_or([0.0; 3]));
        let base_reflectance = base_reflectance_0.lerp(&base_reflectance_1, alpha);

        self.0.dot(&base_reflectance)
    }

    pub fn reflectance_at<const N: usize>(&self, wavelengths: [f32; N]) -> SVector<f32, N> {
        SVector::from_fn(|i, _| self.reflectance_at_one(wavelengths[i]))
    }
}

impl Color<XYZ> {
    #[inline]
    pub fn xyz(x: f32, y: f32, z: f32) -> Self {
        Self(Vector3::new(x, y, z), PhantomData)
    }

    pub fn from_chromaticity_and_luminance(x: f32, y: f32, luminance: f32) -> Self {
        let scale = luminance / y;
        Self(Vector3::new(scale * x, luminance, scale * (1.0 - x - y)), PhantomData)
    }

    pub fn from_wavelength(wavelength: f32) -> Self {
        // Simple Analytic Approximations to the CIE XYZ Color Matching Functions by Chris Wyman, Peter-Pike Sloan, and Peter Shirley
        // http://jcgt.org/published/0002/02/01/paper.pdf

        fn gauss(wavelength: f32, weight: f32, mean: f32, stddev1: f32, stddev2: f32) -> f32 {
            weight
                * (-0.5
                    * ((wavelength - mean) * if wavelength < mean { stddev1 } else { stddev2 })
                        .powi(2))
                .exp()
        }

        Self::xyz(
            gauss(wavelength, 0.362, 442.0, 0.0624, 0.0374)
                + gauss(wavelength, 1.056, 599.8, 0.0264, 0.0323)
                + gauss(wavelength, -0.065, 501.1, 0.0490, 0.0382),
            gauss(wavelength, 0.821, 568.8, 0.0213, 0.0247)
                + gauss(wavelength, 0.286, 530.9, 0.0613, 0.0322),
            gauss(wavelength, 1.217, 437.0, 0.0845, 0.0278)
                + gauss(wavelength, 0.681, 459.0, 0.0385, 0.0725),
        )
    }

    pub fn x(&self) -> f32 {
        self.0.x
    }

    pub fn y(&self) -> f32 {
        self.0.y
    }

    pub fn z(&self) -> f32 {
        self.0.z
    }
}

impl From<Color<SRGB>> for Color<XYZ> {
    #[allow(clippy::excessive_precision)]
    fn from(srgb: Color<SRGB>) -> Color<XYZ> {
        let srgb2xyz = Matrix3::new(
            0.41239080, 0.35758434, 0.18048079, 0.21263901, 0.71516868, 0.07219232, 0.01933082,
            0.11919478, 0.95053215,
        );
        Color(srgb2xyz * srgb.0, PhantomData)
    }
}

impl<P> From<Rgba<P>> for Color<SRGB>
where
    P: Primitive,
{
    #[inline]
    fn from(rgba: Rgba<P>) -> Self {
        let inv_max = 1.0 / P::max_value().to_f32().expect("pixel max outside of range for f32");

        Color::srgb(
            to_linear_srgb(rgba[0].to_f32().expect("r outside of range for f32") * inv_max),
            to_linear_srgb(rgba[1].to_f32().expect("g outside of range for f32") * inv_max),
            to_linear_srgb(rgba[2].to_f32().expect("b outside of range for f32") * inv_max),
        )
    }
}

impl<S> Add for Color<S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, PhantomData)
    }
}

impl<S> AddAssign for Color<S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<S> Mul for Color<S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.component_mul(&rhs.0), PhantomData)
    }
}

impl<S> Mul<f32> for Color<S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs, PhantomData)
    }
}

impl<S> Mul<Color<S>> for f32 {
    type Output = Color<S>;

    fn mul(self, rhs: Color<S>) -> Self::Output {
        Color(self * rhs.0, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use super::{Color, XYZ};
    use approx::assert_abs_diff_eq;

    const COLOR_EPS: f32 = 0.001;

    #[test]
    fn srgb_primary_r() {
        assert_abs_diff_eq!(
            Color::<XYZ>::from(Color::srgb(1.0, 0.0, 0.0)).0,
            Color::from_chromaticity_and_luminance(0.6400, 0.3300, 0.2126).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_g() {
        assert_abs_diff_eq!(
            Color::<XYZ>::from(Color::srgb(0.0, 1.0, 0.0)).0,
            Color::from_chromaticity_and_luminance(0.3000, 0.6000, 0.7152).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_b() {
        assert_abs_diff_eq!(
            Color::<XYZ>::from(Color::srgb(0.0, 0.0, 1.0)).0,
            Color::from_chromaticity_and_luminance(0.1500, 0.0600, 0.0722).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_w() {
        assert_abs_diff_eq!(
            Color::<XYZ>::from(Color::srgb(1.0, 1.0, 1.0)).0,
            Color::from_chromaticity_and_luminance(0.3127, 0.3290, 1.0).0,
            epsilon = COLOR_EPS
        );
    }
}
