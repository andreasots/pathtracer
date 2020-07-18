use anyhow::{Context, Error};

fn main() -> Result<(), Error> {
    println!("cargo:rerun-if-changed=src/shader.frag.glsl");
    println!("cargo:rerun-if-changed=src/shader.vert.glsl");

    let mut compiler = shaderc::Compiler::new().context("failed to create the shaderc compiler")?;

    let frag_shader_src = std::fs::read_to_string("src/shader.frag.glsl")
        .context("failed to read 'src/shader.frag.glsl'")?;
    let vert_shader_src = std::fs::read_to_string("src/shader.vert.glsl")
        .context("failed to read 'src/shader.vert.glsl'")?;

    let frag_shader = compiler
        .compile_into_spirv(
            &frag_shader_src,
            shaderc::ShaderKind::Fragment,
            "src/shader.frag.glsl",
            "main",
            None,
        )
        .context("failed to compile 'src/shader.frag.glsl'")?;
    let vert_shader = compiler
        .compile_into_spirv(
            &vert_shader_src,
            shaderc::ShaderKind::Vertex,
            "src/shader.vert.glsl",
            "main",
            None,
        )
        .context("failed to compile 'src/shader.vert.glsl'")?;

    if frag_shader.get_num_warnings() > 0 {
        println!("cargo:warning={}", frag_shader.get_warning_messages());
    }
    if vert_shader.get_num_warnings() > 0 {
        println!("cargo:warning={}", vert_shader.get_warning_messages());
    }

    let base_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let base_dir = std::path::Path::new(&base_dir);

    std::fs::write(
        base_dir.join("shader.frag.spv.rs"),
        format!("{:?}", frag_shader.as_binary()).as_bytes(),
    )
    .context("failed to write 'shader.frag.spv.rs'")?;
    std::fs::write(
        base_dir.join("shader.vert.spv.rs"),
        format!("{:?}", vert_shader.as_binary()).as_bytes(),
    )
    .context("failed to write 'shader.vert.spv.rs'")?;

    Ok(())
}
