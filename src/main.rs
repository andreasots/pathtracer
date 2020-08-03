use crate::camera::Camera;
use crate::color::Color;
use crate::scene::Scene;
use anyhow::{Context, Error};
use rand::Rng;
use rand_distr::StandardNormal;
use rand_distr::Uniform;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

mod bvh;
mod camera;
mod color;
mod distributions;
mod hosek_wilkie;
mod material;
mod renderer;
mod scene;
mod triangle;

const MIN_WAVELENGTH: f32 = 360.0;
const MAX_WAVELENGTH: f32 = 830.0;

const TILE_SIZE: usize = 32;

fn main() -> Result<(), Error> {
    simple_logger::init().context("failed to init logging")?;

    let scene_file_name = std::env::args_os()
        .skip(1)
        .next()
        .context("expected scene file name")?;
    let scene = Scene::load(&scene_file_name).context("failed to load the scene")?;
    let (width, height) = scene.camera.resolution;

    let mut buffer = Vec::with_capacity(4 * width * height);
    let uninitialized_buffer_color = loop {
        let wavelength = rand::thread_rng().sample(Uniform::new(MIN_WAVELENGTH, MAX_WAVELENGTH));
        let color = Color::from_wavelength(wavelength);
        if color.y() > 0.1 {
            break color;
        }
    };
    for _ in 0..height {
        for _ in 0..width {
            buffer.push(AtomicU32::new(uninitialized_buffer_color.x().to_bits()));
            buffer.push(AtomicU32::new(uninitialized_buffer_color.y().to_bits()));
            buffer.push(AtomicU32::new(uninitialized_buffer_color.z().to_bits()));
            buffer.push(AtomicU32::new(1.0f32.to_bits()));
        }
    }
    let buffer = Arc::new(buffer);
    let done = Arc::new(AtomicBool::new(false));

    {
        let buffer = buffer.clone();
        let done = done.clone();
        rayon::spawn(move || {
            let camera = Camera::from(scene.camera);
            let samples = scene.camera.samples;

            let start = std::time::Instant::now();

            (0..height)
                .into_par_iter()
                .step_by(TILE_SIZE)
                .flat_map(|y| (0..width).into_par_iter().map(move |x| (x, y)))
                .for_each(|(x, y)| {
                    let mut rng = rand::thread_rng();

                    let tile_height = std::cmp::min(height - y, TILE_SIZE);

                    for y in y..y + tile_height {
                        let mut accumulator = Color::xyz(0.0, 0.0, 0.0);

                        for _ in 0..samples {
                            let x = x as f32 + rng.sample::<f32, _>(StandardNormal) * 0.5;
                            let y = y as f32 + rng.sample::<f32, _>(StandardNormal) * 0.5;

                            let ray = camera.generate_ray(x, y);

                            let wavelength_width = MAX_WAVELENGTH - MIN_WAVELENGTH;
                            let hero_wavelength = rng.sample(Uniform::new(0.0, wavelength_width));
                            let wavelengths = [
                                MIN_WAVELENGTH + hero_wavelength,
                                MIN_WAVELENGTH
                                    + (hero_wavelength + wavelength_width * 0.25)
                                        .rem_euclid(wavelength_width),
                                MIN_WAVELENGTH
                                    + (hero_wavelength + wavelength_width * 0.50)
                                        .rem_euclid(wavelength_width),
                                MIN_WAVELENGTH
                                    + (hero_wavelength + wavelength_width * 0.75)
                                        .rem_euclid(wavelength_width),
                            ];

                            let radiance = scene.radiance(ray, wavelengths, &mut rng, None, 0);

                            accumulator +=
                                Color::from_wavelength(wavelengths[0]) * radiance[0] * 0.25;
                            accumulator +=
                                Color::from_wavelength(wavelengths[1]) * radiance[1] * 0.25;
                            accumulator +=
                                Color::from_wavelength(wavelengths[2]) * radiance[2] * 0.25;
                            accumulator +=
                                Color::from_wavelength(wavelengths[3]) * radiance[3] * 0.25;
                        }

                        let color = accumulator * (1.0 / (samples as f32));

                        buffer[4 * (x + y * width) + 0]
                            .store(color.x().to_bits(), Ordering::Relaxed);
                        buffer[4 * (x + y * width) + 1]
                            .store(color.y().to_bits(), Ordering::Relaxed);
                        buffer[4 * (x + y * width) + 2]
                            .store(color.z().to_bits(), Ordering::Relaxed);
                    }
                });

            let end = std::time::Instant::now();

            println!(
                "Rendered in {:.02} seconds",
                end.duration_since(start).as_secs_f64()
            );

            done.store(true, Ordering::Release);
        });
    }

    pollster::block_on(crate::renderer::renderer(
        width,
        height,
        &format!("pathtracer: {}", scene_file_name.to_string_lossy()),
        buffer,
        done,
    ))
    .context("failed to start the renderer")?;

    Ok(())
}
