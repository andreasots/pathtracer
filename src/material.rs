use crate::bvh::Ray;
use crate::color::{Color, SRGB, XYZ};
use crate::distributions::CosineWeightedHemisphere;
use crate::scene::Scene;
use crate::triangle::{Intersection, Triangle};
use anyhow::{Context, Error};
use image::{DynamicImage, GenericImageView};
use nalgebra::{Vector3, Vector4};
use rand::Rng;
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

        Image {
            width,
            height,
            data,
        }
    }
}

impl From<DynamicImage> for Image<f32> {
    fn from(img: DynamicImage) -> Self {
        let width = img.width();
        let height = img.height();

        let mut data = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(Color::<XYZ>::from(Color::<SRGB>::from(img.get_pixel(x, y))).y());
            }
        }

        Image {
            width,
            height,
            data,
        }
    }
}

pub enum Texture<P> {
    Texture(Image<P>),
    Flat(P),
}

impl<P> Texture<P>
where
    P: Copy + Add<Output = P> + Mul<f32, Output = P>,
{
    pub fn sample(&self, u: f32, v: f32) -> P {
        match self {
            Texture::Texture(tex) => {
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

pub trait Bsdf {
    /// returns new ray direction and the sample contribution `BSDF(d) / pdf(d)`
    fn sample<R>(
        &self,
        u: f32,
        v: f32,
        wavelengths: [f32; 4],
        use_russian_roulette: bool,
        rng: &mut R,
    ) -> Option<(Vector3<f32>, Vector4<f32>)>
    where
        R: Rng + ?Sized;
}

pub struct Lambert {
    reflectance: Texture<Color<SRGB>>,
}

impl Bsdf for Lambert {
    fn sample<R>(
        &self,
        u: f32,
        v: f32,
        wavelengths: [f32; 4],
        use_russian_roulette: bool,
        rng: &mut R,
    ) -> Option<(Vector3<f32>, Vector4<f32>)>
    where
        R: Rng + ?Sized,
    {
        let reflectance = self.reflectance.sample(u, v).reflectance_at4(wavelengths);
        let (_, max) = if use_russian_roulette {
            reflectance.argmax()
        } else {
            (0, 1.0)
        };
        if rng.gen::<f32>() < max {
            // The normalization factor is 1/pi and the PDF of the sampler is also 1/pi so they cancel out.
            Some((
                rng.sample(CosineWeightedHemisphere),
                reflectance * (1.0 / max),
            ))
        } else {
            None
        }
    }
}

pub struct Mix<A, B> {
    a: A,
    a_weight: f32,
    b: B,
}

impl<A, B> Bsdf for Mix<A, B>
where
    A: Bsdf,
    B: Bsdf,
{
    fn sample<R>(
        &self,
        u: f32,
        v: f32,
        wavelengths: [f32; 4],
        use_russian_roulette: bool,
        rng: &mut R,
    ) -> Option<(Vector3<f32>, Vector4<f32>)>
    where
        R: Rng + ?Sized,
    {
        if rng.gen::<f32>() < self.a_weight {
            self.a.sample(u, v, wavelengths, use_russian_roulette, rng)
        } else {
            self.b.sample(u, v, wavelengths, use_russian_roulette, rng)
        }
    }
}

pub struct Null;

impl Bsdf for Null {
    fn sample<R>(
        &self,
        _u: f32,
        _v: f32,
        _wavelengths: [f32; 4],
        _use_russian_roulette: bool,
        _rng: &mut R,
    ) -> Option<(Vector3<f32>, Vector4<f32>)>
    where
        R: Rng + ?Sized,
    {
        None
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
            Texture::Texture(tex.into())
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
            Emit::AbsorbedD65 {
                absorption: Texture::Texture(tex.into()),
            }
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
            Texture::Texture(tex.into())
        } else {
            println!("\tdissolve: no");
            Texture::Flat(1.0)
        };

        Ok(Material {
            bsdf: Lambert {
                reflectance: diffuse,
            },
            emit,
            base_dissolve,
            dissolve,
        })
    }

    pub fn radiance<R>(
        &self,
        ray: Ray,
        wavelengths: [f32; 4],
        intersection: Intersection,
        triangle: &Triangle,
        scene: &Scene,
        rng: &mut R,
        depth: usize,
    ) -> Vector4<f32>
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
            let p = ray.origin + ray.direction * intersection.distance;
            return scene.radiance(
                Ray::new(p, ray.direction),
                wavelengths,
                rng,
                Some(triangle),
                depth + 1,
            );
        }

        let emit = self.emit.sample(u, v, wavelengths);

        if let Some((d, weight)) = self.bsdf.sample(u, v, wavelengths, depth > 2, rng) {
            let normal = if let Some(normals) = triangle.vertex_normals() {
                (1.0 - intersection.u - intersection.v) * normals[0]
                    + intersection.u * normals[1]
                    + intersection.v * normals[2]
            } else {
                intersection.normal
            };

            // `ray.direction` is facing *into* the surface and `normal` should *out of* the surface.
            let normal = if normal.dot(&ray.direction) < 0.0 {
                normal
            } else {
                -normal
            };

            let normal = normal.normalize();
            // TODO: calculate `(u, v)` from the texture mapping.
            let v = if normal.x.abs() > 0.1 {
                Vector3::new(0.0, 1.0, 0.0)
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            };
            let u = v.cross(&normal).normalize();
            let v = normal.cross(&u);

            let d = u * d.x + v * d.y + normal * d.z;

            let p = ray.origin + ray.direction * intersection.distance;

            let incoming =
                scene.radiance(Ray::new(p, d), wavelengths, rng, Some(triangle), depth + 1);

            incoming.component_mul(&weight) + emit
        } else {
            emit
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            bsdf: Lambert {
                reflectance: Texture::Flat(Color::srgb(0.75, 0.75, 0.75)),
            },
            emit: Emit::Null,
            base_dissolve: 1.0,
            dissolve: Texture::Flat(1.0),
        }
    }
}
