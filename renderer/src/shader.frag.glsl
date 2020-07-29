#version 450 core

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 color;

layout(binding = 0) uniform texture2D frame;
layout(binding = 1) uniform sampler frame_sampler;

void main() {
    mat3 xyz2srgb = mat3(
         3.24096994, -1.53738318, -0.49861076,
        -0.96924364,  1.8759675,   0.04155506,
         0.05563008, -0.20397696,  1.05697151
    );
    vec4 texel = texture(sampler2D(frame, frame_sampler), v_uv);
    color = vec4(texel.xyz * xyz2srgb, texel.w);
}
