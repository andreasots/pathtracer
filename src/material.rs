use crate::bvh::Ray;
use crate::color::{Color, SRGB, XYZ};
use crate::triangle::{Intersection, Triangle};
use anyhow::{Context, Error};
use image::{DynamicImage, GenericImageView};
use nalgebra::{Vector3, Vector4};
use rand::Rng;
use rand_distr::UnitSphere;
use std::ops::{Add, Index, Mul};
use std::path::Path;

pub struct Image<P> {
    width: u32,
    height: u32,
    data: Vec<P>,
}

impl<P> Image<P> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl<P> Index<(u32, u32)> for Image<P> {
    type Output = P;

    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);

        &self.data[x as usize + y as usize * self.width as usize]
    }
}

impl From<DynamicImage> for Image<Color<SRGB>> {
    fn from(img: DynamicImage) -> Self {
        let width = img.width();
        let height = img.height();

        let mut data = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(img.get_pixel(x, y).into())
            }
        }

        Image { width, height, data }
    }
}

impl From<DynamicImage> for Image<f32> {
    fn from(img: DynamicImage) -> Self {
        let width = img.width();
        let height = img.height();

        let mut data = Vec::with_capacity(width as usize * height as usize);

        if img.color().has_alpha() {
            for y in 0..height {
                for x in 0..width {
                    data.push(img.get_pixel(x, y)[3] as f32 / 255.0);
                }
            }
        } else {
            for y in 0..height {
                for x in 0..width {
                    data.push(Color::<XYZ>::from(Color::<SRGB>::from(img.get_pixel(x, y))).y());
                }
            }
        }

        Image { width, height, data }
    }
}

pub enum Texture<P> {
    Nearest(Image<P>),
    Bilinear(Image<P>),
    Flat(P),
}

impl<P> Texture<P>
where
    P: Copy + Add<Output = P> + Mul<f32, Output = P>,
{
    pub fn sample(&self, u: f32, v: f32) -> P {
        match self {
            Texture::Nearest(tex) => {
                let width = tex.width();
                let height = tex.height();

                let x = ((u.rem_euclid(1.0)) * width as f32).floor() as u32;
                let y = ((1.0 - v.rem_euclid(1.0)) * height as f32).floor() as u32;

                tex[(x.rem_euclid(width), y.rem_euclid(height))]
            }
            Texture::Bilinear(tex) => {
                let width = tex.width();
                let height = tex.height();

                let x = (u.rem_euclid(1.0)) * width as f32;
                let y = (1.0 - v.rem_euclid(1.0)) * height as f32;

                let x0 = x.floor() as u32;
                let y0 = y.floor() as u32;

                let t_x = x - x.floor();
                let t_y = y - y.floor();

                let p00 = tex[(x0.rem_euclid(width), y0.rem_euclid(height))];
                let p10 = tex[((x0 + 1).rem_euclid(width), y0.rem_euclid(height))];
                let p01 = tex[(x0.rem_euclid(width), (y0 + 1).rem_euclid(height))];
                let p11 = tex[((x0 + 1).rem_euclid(width), (y0 + 1).rem_euclid(height))];

                let p0 = p00 * (1.0 - t_x) + p10 * t_x;
                let p1 = p01 * (1.0 - t_x) + p11 * t_x;

                p0 * (1.0 - t_y) + p1 * t_y
            }
            Texture::Flat(color) => *color,
        }
    }
}

pub struct D65;

impl D65 {
    pub fn sample(&self, wavelength: f32) -> f32 {
        const TABLE: &[f32] = &include!(concat!(env!("OUT_DIR"), "/d65.rs"));

        let offset = wavelength.floor() - 300.0 as f32;
        let index = offset.floor();
        let alpha = offset - index;
        let index = index as usize;

        TABLE[index] * (1.0 - alpha) + TABLE[index + 1] * alpha
    }

    pub fn sample4(&self, wavelengths: [f32; 4]) -> Vector4<f32> {
        Vector4::new(
            self.sample(wavelengths[0]),
            self.sample(wavelengths[1]),
            self.sample(wavelengths[2]),
            self.sample(wavelengths[3]),
        )
    }
}

pub struct Lambert {
    reflectance: Texture<Color<SRGB>>,
}

impl Lambert {
    fn sample<R>(
        &self,
        normal: Vector3<f32>,
        u: f32,
        v: f32,
        wavelengths: [f32; 4],
        rng: &mut R,
    ) -> (Vector3<f32>, Vector4<f32>)
    where
        R: Rng + ?Sized,
    {
        (
            // Lambertian Reflection Without Tangents
            // https://web.archive.org/web/20211207115922/http://amietia.com/lambertnotangent.html
            // TLDR: unit sphere offset by the shading normal
            normal + Vector3::from(rng.sample(UnitSphere)),
            // The normalization factor 1/pi is cancelled out by the sampler.
            self.reflectance.sample(u, v).reflectance_at4(wavelengths),
        )
    }
}

pub enum Emit {
    AbsorbedD65 { absorption: Texture<Color<SRGB>> },
    Null,
}

impl Emit {
    fn sample(&self, u: f32, v: f32, wavelengths: [f32; 4]) -> Vector4<f32> {
        match self {
            Self::AbsorbedD65 { absorption } => D65
                .sample4(wavelengths)
                .component_mul(&absorption.sample(u, v).reflectance_at4(wavelengths)),
            Self::Null => Vector4::from_element(0.0),
        }
    }
}

pub struct Sample {
    pub emit: Vector4<f32>,
    /// BRDF(..) * n.dot(&-ray.direction)
    pub weight: Vector4<f32>,
    pub new_ray: Ray,
}

pub struct Material {
    pub bsdf: Lambert,
    pub emit: Emit,
    pub base_dissolve: f32,
    pub dissolve: Texture<f32>,
}

impl Material {
    pub fn from_mtl<P: AsRef<Path>>(mtl_path: P, mtl: &obj::Material) -> Result<Material, Error> {
        println!("Material {}:", mtl.name);

        let mtl_path = mtl_path.as_ref();

        let diffuse = if let Some(ref path) = mtl.map_kd {
            println!("\tdiffuse: {}", path);
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load diffuse texture {:?}", path))?;
            Texture::Nearest(tex.into())
        } else if let Some(color) = mtl.kd {
            println!("\tdiffuse: {:?}", color);
            Texture::Flat(Color::srgb(color[0], color[1], color[2]))
        } else {
            println!("\tdiffuse: default 75% gray");
            Texture::Flat(Color::srgb(0.75, 0.75, 0.75))
        };

        let emit = if let Some(ref path) = mtl.map_ka {
            println!("\temit: {}", path);
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load emissive texture {:?}", path))?;
            Emit::AbsorbedD65 { absorption: Texture::Nearest(tex.into()) }
        } else if let Some(color) = mtl.ke {
            println!("\temit: {:?}", color);
            Emit::AbsorbedD65 {
                absorption: Texture::Flat(Color::srgb(color[0], color[1], color[2])),
            }
        } else {
            println!("\temit: nothing");
            Emit::Null
        };

        let base_dissolve = mtl.d.unwrap_or(1.0);
        println!("\tbase dissolve: {}", base_dissolve);

        let dissolve = if let Some(ref path) = mtl.map_d {
            println!("\tdissolve: {}", path);
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load dissolve texture {:?}", path))?;
            Texture::Nearest(tex.into())
        } else {
            println!("\tdissolve: no");
            Texture::Flat(1.0)
        };

        Ok(Material { bsdf: Lambert { reflectance: diffuse }, emit, base_dissolve, dissolve })
    }

    pub fn sample<R>(
        &self,
        ray: Ray,
        wavelengths: [f32; 4],
        intersection: Intersection,
        triangle: &Triangle,
        rng: &mut R,
    ) -> Option<Sample>
    where
        R: Rng + ?Sized,
    {
        let (u, v) = if let Some(uv_mapping) = triangle.texture_coords() {
            let tex = (1.0 - intersection.u - intersection.v) * uv_mapping[0].coords
                + intersection.u * uv_mapping[1].coords
                + intersection.v * uv_mapping[2].coords;
            (tex[0], tex[1])
        } else {
            (intersection.u, intersection.v)
        };

        if rng.gen::<f32>() >= self.base_dissolve * self.dissolve.sample(u, v) {
            return None;
        }

        let normal = if let Some(normals) = triangle.vertex_normals() {
            (1.0 - intersection.u - intersection.v) * normals[0]
                + intersection.u * normals[1]
                + intersection.v * normals[2]
        } else {
            intersection.normal
        };

        // `ray.direction` is facing *into* the surface and `normal` should *out of* the surface.
        let normal = if normal.dot(&ray.direction) < 0.0 { normal } else { -normal };

        let normal = normal.normalize();

        let p = ray.origin + ray.direction * intersection.distance;

        let (d, weight) = self.bsdf.sample(normal, u, v, wavelengths, rng);

        Some(Sample { emit: self.emit.sample(u, v, wavelengths), weight, new_ray: Ray::new(p, d) })
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            bsdf: Lambert { reflectance: Texture::Flat(Color::srgb(0.75, 0.75, 0.75)) },
            emit: Emit::Null,
            base_dissolve: 1.0,
            dissolve: Texture::Flat(1.0),
        }
    }
}
