use anyhow::{Context, Error};
use serde::Deserialize;
use std::fmt::Debug;
use std::path::Path;

fn csv_to_table<T, U, F>(src_path: &str, dst_path: &Path, f: F) -> Result<(), Error>
where
    T: for<'de> Deserialize<'de>,
    U: Debug,
    F: FnOnce(Vec<T>) -> Vec<U>,
{
    println!("cargo:rerun-if-changed={}", src_path);

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .has_headers(false)
        .trim(csv::Trim::All)
        .from_path(src_path)
        .with_context(|| format!("failed to open {:?}", src_path))?;

    let data = reader
        .deserialize::<T>()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read from {:?}", src_path))?;

    std::fs::write(dst_path, format!("{:?}", f(data)).as_bytes())
        .with_context(|| format!("failed to write {:?}", dst_path))?;

    Ok(())
}

fn main() -> Result<(), Error> {
    let base_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let base_dir = std::path::Path::new(&base_dir);

    csv_to_table::<[f32; 3], _, _>(
        "src/data/RGB-Components-CIE-1931-1nm.csv",
        &base_dir.join("srgb2reflectance.rs"),
        |data| data,
    )
    .context("failed to convert the sRGB to reflectance table")?;

    csv_to_table::<(u16, f32), _, _>(
        "src/data/Illuminantd65.csv",
        &base_dir.join("d65.rs"),
        |data| {
            let mut avg = 0.0;
            let mut n = 0.0;

            fn gauss(wavelength: f32, weight: f32, mean: f32, stddev1: f32, stddev2: f32) -> f32 {
                weight
                    * (-0.5
                        * ((wavelength - mean) * if wavelength < mean { stddev1 } else { stddev2 })
                            .powi(2))
                    .exp()
            }

            for &(wavelength, radiance) in &data {
                if wavelength < 360 || wavelength > 830 {
                    continue;
                }

                avg += radiance
                    * (gauss(wavelength as f32, 0.821, 568.8, 0.0213, 0.0247)
                        + gauss(wavelength as f32, 0.286, 530.9, 0.0613, 0.0322));
                n += 1.0;
            }
            avg /= n;

            data.into_iter().map(|(_, radiance)| radiance / avg).collect()
        },
    )
    .context("failed to convert the D64 illuminant table")?;

    Ok(())
}
