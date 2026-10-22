// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <string_view>

namespace Metal {
// The translated color input/output use raster_order_group(0). Every covered
// fragment reads the density written by the preceding fragment at that pixel.
inline constexpr std::string_view GasAccumulationHelpers = R"GLSL(
layout(input_attachment_index=0, set=4, binding=0) uniform subpassInput metal_gas_destination;
layout(set=3, binding=5, std430) readonly buffer MetalGasRegisters { uint metal_gas_words[152]; };
layout(set=3, binding=6, std430) buffer MetalGasMaximum { uint metal_gas_maximum; };
layout(set=2, binding=2) uniform sampler2D metal_gas_depth;
vec4 metalGasAccumulate(float contribution, float fragment_depth) {
    uvec4 bytes = uvec4(round(clamp(subpassLoad(metal_gas_destination), 0.0, 1.0) * 255.0));
    vec2 previous = vec2(unpackHalf2x16((bytes.a << 8u) | bytes.b).x,
                         unpackHalf2x16((bytes.g << 8u) | bytes.r).x);
    uint delta = metal_gas_words[150];
    uint function = (delta >> 24u) & 3u;
    float attenuation = float(function);
    if (function >= 2u) {
        ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), textureSize(metal_gas_depth, 0) - 1);
        float scene_depth = texelFetch(metal_gas_depth, pixel, 0).r;
        attenuation = clamp((scene_depth - fragment_depth) * float(delta & 16777215u) / 256.0, 0.0, 1.0);
        if (function == 2u) attenuation = 1.0 - attenuation;
    }
    vec2 accumulated = previous + vec2(contribution, contribution * attenuation);
    uint plain = packHalf2x16(vec2(accumulated.x, 0.0)) & 65535u;
    uint depth = packHalf2x16(vec2(accumulated.y, 0.0)) & 65535u;
    if ((plain & 32768u) == 0u && (plain & 32767u) <= 31744u)
        atomicMax(metal_gas_maximum, plain);
    return vec4(depth & 255u, depth >> 8u, plain & 255u, plain >> 8u) / 255.0;
}
)GLSL";
}
