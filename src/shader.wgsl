struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    var ret: VertexOutput;
    ret.position = vec4(pos, 0.0, 1.0);
    ret.uv = uv;
    return ret;
}

struct ToneMapping {
    middle_gray: f32,
    avg_luminance: f32,
    max_luminance: f32,
}

@group(0)
@binding(0)
var frame: texture_2d<f32>;

@group(0)
@binding(1)
var frame_sampler: sampler;

@group(0)
@binding(2)
var<uniform> tone_mapping: ToneMapping;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let xyz2srgb = mat3x3<f32>(
         3.24096994, -1.53738318, -0.49861076,
        -0.96924364,  1.8759675,   0.04155506,
         0.05563008, -0.20397696,  1.05697151
    );
    let texel = textureSample(frame, frame_sampler, uv);
    var xyz = texel.xyz;
    xyz *= tone_mapping.middle_gray / tone_mapping.avg_luminance;
    xyz *= (1.0 + xyz.y / (tone_mapping.max_luminance * tone_mapping.max_luminance)) / (1.0 + xyz.y);
    return vec4(xyz * xyz2srgb, texel.w);
}
