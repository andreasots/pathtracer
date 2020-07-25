use anyhow::{Context, Error};
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

fn main() -> Result<(), Error> {
    let mut compiler = shaderc::Compiler::new().context("failed to create the shaderc compiler")?;

    let base_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let base_dir = std::path::Path::new(&base_dir);

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

    Ok(())
}
