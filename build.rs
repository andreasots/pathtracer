use anyhow::{Context, Error};
use serde::Deserialize;
use std::fmt::Debug;
use std::path::Path;

fn compile_shader(
    compiler: &mut shaderc::Compiler,
    kind: shaderc::ShaderKind,
    src_path: &str,
    dst_path: &Path,
) -> Result<(), Error> {
    println!("cargo:rerun-if-changed={}", src_path);

    let shader_src = std::fs::read_to_string(src_path)
        .with_context(|| format!("failed to read {:?}", src_path))?;

    let shader = compiler
        .compile_into_spirv(&shader_src, kind, src_path, "main", None)
        .with_context(|| format!("failed to compile {:?}", src_path))?;

    if shader.get_num_warnings() > 0 {
        println!("cargo:warning={}", shader.get_warning_messages());
    }

    std::fs::write(dst_path, format!("{:?}", shader.as_binary()).as_bytes())
        .with_context(|| format!("failed to write {:?}", dst_path))?;

    Ok(())
}

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

    let mut compiler = shaderc::Compiler::new().context("failed to create the shaderc compiler")?;

    compile_shader(
        &mut compiler,
        shaderc::ShaderKind::Fragment,
        "src/shader.frag.glsl",
        &base_dir.join("shader.frag.spv.rs"),
    )
    .context("failed to compile the fragment shader")?;

    compile_shader(
        &mut compiler,
        shaderc::ShaderKind::Vertex,
        "src/shader.vert.glsl",
        &base_dir.join("shader.vert.spv.rs"),
    )
    .context("failed to compile the vertex shader")?;

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

            data.into_iter()
                .map(|(_, radiance)| radiance / avg)
                .collect()
        },
    )
    .context("failed to convert the D64 illuminant table")?;

    cc::Build::new()
        .file("src/hosek-wilkie/ArHosekSkyModel.c")
        .compile("hosek-wilkie");

    println!("cargo:rerun-if-changed=src/hosek-wilkie/ArHosekSkyModel.h");
    let bindings = bindgen::Builder::default()
        .header("src/hosek-wilkie/ArHosekSkyModel.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect(
            "failed to generate bindings for the Hosek-Wilkie sky model reference implementation",
        );

    bindings
        .write_to_file(base_dir.join("ar_hosek_sky_model.rs"))
        .context(
            "failed to write the bindings for the Hosek-Wilkie sky model reference implementation",
        )?;

    Ok(())
}
