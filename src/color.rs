use image::{Primitive, Rgba};
use std::ops::{Add, AddAssign, Mul};
use bvh::nalgebra::{Vector3, Matrix3};

fn to_linear_srgb(u: f32) -> f32 {
    if u <= 0.04045 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color(Vector3<f32>);

impl Color {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Color {
        Self(Vector3::new(x, y, z))
    }

    pub fn from_linear_srgb(r: f32, g: f32, b: f32) -> Self {
        let srgb = Vector3::new(r, g, b);
        let srgb2xyz = Matrix3::new(
            0.41239080, 0.35758434, 0.18048079,
            0.21263901, 0.71516868, 0.07219232,
            0.01933082, 0.11919478, 0.95053215,
        );
        Color(srgb2xyz * srgb)
    }

    pub fn from_chromaticity_and_luminance(x: f32, y: f32, luminance: f32) -> Self {
        let scale = luminance / y;
        Self(Vector3::new(scale * x, luminance, scale * (1.0 - x - y)))
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

impl<P> From<Rgba<P>> for Color where P: Primitive {
    #[inline]
    fn from(rgba: Rgba<P>) -> Self {
        let inv_max = 1.0
        / P::max_value()
            .to_f32()
            .expect("pixel max outside of range for f32");

        Color::from_linear_srgb(
            to_linear_srgb(rgba[0].to_f32().expect("r outside of range for f32") * inv_max),
            to_linear_srgb(rgba[1].to_f32().expect("g outside of range for f32") * inv_max),
            to_linear_srgb(rgba[2].to_f32().expect("b outside of range for f32") * inv_max),
        )
    }
}

impl Add<Color> for Color {
    type Output = Color;

    #[inline]
    fn add(self, rhs: Color) -> Self::Output {
        Color(self.0 + rhs.0)
    }
}

impl AddAssign<Color> for Color {
    #[inline]
    fn add_assign(&mut self, rhs: Color) {
        self.0 += rhs.0;
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    #[inline]
    fn mul(self, rhs: Color) -> Self::Output {
        Color(self.0.component_mul(&rhs.0))
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Color(self.0 * rhs)
    }
}

impl Mul<Color> for f32 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color(self * rhs.0)
    }
}

#[cfg(test)]
mod test {
    use super::Color;
    use approx::assert_abs_diff_eq;

    const COLOR_EPS: f32 = 0.0001;

    #[test]
    fn srgb_primary_r() {
        assert_abs_diff_eq!(
            Color::from_linear_srgb(1.0, 0.0, 0.0).0,
            Color::from_chromaticity_and_luminance(0.6400, 0.3300, 0.2126).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_g() {
        assert_abs_diff_eq!(
            Color::from_linear_srgb(0.0, 1.0, 0.0).0,
            Color::from_chromaticity_and_luminance(0.3000, 0.6000, 0.7152).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_b() {
        assert_abs_diff_eq!(
            Color::from_linear_srgb(0.0, 0.0, 1.0).0,
            Color::from_chromaticity_and_luminance(0.1500, 0.0600, 0.0722).0,
            epsilon = COLOR_EPS
        );
    }

    #[test]
    fn srgb_primary_w() {
        assert_abs_diff_eq!(
            Color::from_linear_srgb(1.0, 1.0, 1.0).0,
            Color::from_chromaticity_and_luminance(0.3127, 0.3290, 1.0).0,
            epsilon = COLOR_EPS
        );
    }
}
