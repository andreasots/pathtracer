use crate::color::Color;
use crate::scene::Scene;
use crate::triangle::{Intersection, Triangle};
use anyhow::{Context, Error};
use bvh::nalgebra::Vector3;
use bvh::ray::Ray;
use image::{DynamicImage, GenericImageView};
use rand::Rng;
use std::path::Path;

pub enum Texture {
    Texture(DynamicImage),
    Flat(Color),
}

impl Texture {
    pub fn sample(&self, u: f32, v: f32) -> Color {
        match self {
            Texture::Texture(tex) => {
                let w = tex.width();
                let h = tex.height();

                let x = (u.rem_euclid(1.0)) * w as f32;
                let y = (1.0 - v.rem_euclid(1.0)) * h as f32;

                let x0 = x.floor() as u32;
                let y0 = y.floor() as u32;

                let t_x = x - x.floor();
                let t_y = y - y.floor();

                let p00: Color = tex.get_pixel(x0.rem_euclid(w), y0.rem_euclid(h)).into();
                let p10: Color = tex.get_pixel((x0 + 1).rem_euclid(w), y0.rem_euclid(h)).into();
                let p01: Color = tex.get_pixel(x0.rem_euclid(w), (y0 + 1).rem_euclid(h)).into();
                let p11: Color = tex.get_pixel((x0 + 1).rem_euclid(w), (y0 + 1).rem_euclid(h)).into();

                let p0 = p00 * (1.0 - t_x) + p10 * t_x;
                let p1 = p01 * (1.0 - t_x) + p11 * t_x;

                p0 * (1.0 - t_y) + p1 * t_y
            }
            Texture::Flat(color) => *color,
        }
    }
}

pub struct Material {
    pub diffuse: Texture,
    pub emit: Texture,
    pub base_dissolve: f32,
    pub dissolve: Texture,
}

impl Material {
    pub fn from_mtl<P: AsRef<Path>>(mtl_path: P, mtl: &obj::Material) -> Result<Material, Error> {
        let mtl_path = mtl_path.as_ref();

        let diffuse = if let Some(ref path) = mtl.map_kd {
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load diffuse texture {:?}", path))?;
            Texture::Texture(tex)
        } else if let Some(color) = mtl.kd {
            Texture::Flat(Color::from_linear_srgb(color[0], color[1], color[2]))
        } else {
            Texture::Flat(Color::new(0.0, 0.75, 0.0))
        };

        let emit = if let Some(ref path) = mtl.map_ke {
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load emissive texture {:?}", path))?;
            Texture::Texture(tex)
        } else if let Some(color) = mtl.ke {
            Texture::Flat(Color::from_linear_srgb(color[0], color[1], color[2]))
        } else {
            Texture::Flat(Color::new(0.0, 0.0, 0.0))
        };

        let base_dissolve = mtl.d.unwrap_or(1.0);

        let dissolve = if let Some(ref path) = mtl.map_d {
            let tex = image::open(mtl_path.with_file_name(path))
                .with_context(|| format!("failed to load dissolve texture {:?}", path))?;
            Texture::Texture(tex)
        } else {
            Texture::Flat(Color::new(0.0, 1.0, 0.0))
        };

        Ok(Material { diffuse, emit, base_dissolve, dissolve })
    }

    pub fn radiance<R>(
        &self,
        ray: Ray,
        intersection: Intersection,
        triangle: &Triangle,
        scene: &Scene,
        rng: &mut R,
        depth: usize,
    ) -> Color
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

        if rng.gen::<f32>() >= self.base_dissolve * self.dissolve.sample(u, v).y() {
            let p = ray.origin + ray.direction * intersection.distance;
            return scene.radiance(Ray::new(p, ray.direction), rng, Some(triangle), depth + 1);
        }

        let diffuse = self.diffuse.sample(u, v);
        let emit = self.emit.sample(u, v);

        let diffuse = if depth > 5 {
            if rng.gen::<f32>() < diffuse.y() {
                diffuse * (1.0 / diffuse.y())
            } else {
                return emit;
            }
        } else {
            diffuse
        };

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

        let r1 = 2.0 * std::f32::consts::PI * rng.gen::<f32>();
        let r2 = rng.gen::<f32>();
        let r2s = r2.sqrt();

        let (sin_r1, cos_r1) = r1.sin_cos();

        let v = if normal.x.abs() > 0.1 {
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let u = v.cross(&normal).normalize();
        let v = normal.cross(&u);

        let d = u * cos_r1 * r2s + v * sin_r1 * r2s + normal * (1.0 - r2).sqrt();

        let p = ray.origin + ray.direction * intersection.distance;

        let incoming = scene.radiance(Ray::new(p, d), rng, Some(triangle), depth + 1);

        diffuse * incoming + emit
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            diffuse: Texture::Flat(Color::new(0.0, 0.75, 0.0)),
            emit: Texture::Flat(Color::new(0.0, 0.0, 0.0)),
            base_dissolve: 1.0,
            dissolve: Texture::Flat(Color::new(0.0, 1.0, 0.0)),
        }
    }
}
