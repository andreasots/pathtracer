#version 450 core

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 color;

layout(binding = 0) uniform texture2D frame;
layout(binding = 1) uniform sampler frame_sampler;
layout(binding = 2) uniform ToneMapping {
    float avg_luminance;
    float max_luminance;
} tone_mapping;

void main() {
    mat3 xyz2srgb = mat3(
         3.24096994, -1.53738318, -0.49861076,
        -0.96924364,  1.8759675,   0.04155506,
         0.05563008, -0.20397696,  1.05697151
    );
    vec4 texel = texture(sampler2D(frame, frame_sampler), v_uv);
    vec3 xyz = texel.xyz;
    xyz *= 0.18 / tone_mapping.avg_luminance;
    xyz *= (1 + xyz.y / (tone_mapping.max_luminance * tone_mapping.max_luminance)) / (1.0 + xyz.y);
    color = vec4(xyz * xyz2srgb, texel.w);
}
