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
            data.into_iter().map(|(_, radiance)| radiance).collect()
        },
    )
    .context("failed to convert the D64 illuminant table")?;

    Ok(())
}
