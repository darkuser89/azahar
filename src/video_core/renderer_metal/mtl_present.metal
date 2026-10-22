// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include <metal_stdlib>
using namespace metal;

struct PresentVertex {
    float2 position;
    float2 texcoord;
};

struct PresentUniforms {
    float4x4 modelview;
    float4 input_resolution;
    float4 output_resolution;
    int left_screen;
    int right_screen;
    int mode;
    int reverse_interlaced;
    float4 right_texcoord_transform;
};

struct PresentOutput {
    float4 position [[position]];
    float2 texcoord;
};

vertex PresentOutput present_vertex(const device PresentVertex* vertices [[buffer(0)]],
                                    constant PresentUniforms& uniforms [[buffer(1)]],
                                    uint index [[vertex_id]]) {
    const float4 position = float4(vertices[index].position, 0.0f, 1.0f) * uniforms.modelview;
    // Vulkan's positive-height viewport and Metal use opposite NDC Y directions.
    return {float4(position.x, -position.y, 0.0f, 1.0f), vertices[index].texcoord};
}

fragment float4 present_fragment(PresentOutput in [[stage_in]],
                                  constant PresentUniforms& uniforms [[buffer(1)]],
                                  array<texture2d<float>, 3> screens [[texture(0)]],
                                  sampler screen_sampler [[sampler(0)]]) {
    if (uniforms.mode == 2) {
        // Interlacing follows physical output rows, independent of a cache subrectangle
        // or rotated LCD texture coordinates. ReverseInterlaced exchanges the two eyes.
        const bool left = (int(in.position.y) % 2 == uniforms.reverse_interlaced);
        const float2 uv = left ? in.texcoord : in.texcoord * uniforms.right_texcoord_transform.xy +
                                                uniforms.right_texcoord_transform.zw;
        return screens[left ? uniforms.left_screen : uniforms.right_screen].sample(screen_sampler, uv);
    }
    const float4 left = screens[uniforms.left_screen].sample(screen_sampler, in.texcoord);
    if (uniforms.mode != 1) {
        return left;
    }
    const float2 right_uv = in.texcoord * uniforms.right_texcoord_transform.xy + uniforms.right_texcoord_transform.zw;
    const float4 right = screens[uniforms.right_screen].sample(screen_sampler, right_uv);
    // Dubois red/cyan projection, preserving the Vulkan shader's matrix orientation.
    const float3 red = float3(0.437f, 0.449f, 0.164f);
    const float3 green = float3(-0.062f, -0.062f, -0.024f);
    const float3 blue = float3(-0.048f, -0.050f, -0.017f);
    const float3 cyan_red = float3(-0.011f, -0.032f, -0.007f);
    const float3 cyan_green = float3(0.377f, 0.761f, 0.009f);
    const float3 cyan_blue = float3(-0.026f, -0.093f, 1.234f);
    return float4(dot(left.rgb, red) + dot(right.rgb, cyan_red),
                  dot(left.rgb, green) + dot(right.rgb, cyan_green),
                  dot(left.rgb, blue) + dot(right.rgb, cyan_blue), left.a);
}

vertex float4 cursor_vertex(const device float2* vertices [[buffer(0)]], uint index [[vertex_id]]) {
    return float4(vertices[index].x, -vertices[index].y, 0.0f, 1.0f);
}

fragment float4 cursor_fragment() {
    return float4(1.0f);
}
