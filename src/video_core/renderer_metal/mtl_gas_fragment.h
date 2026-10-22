// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <string_view>

namespace Metal {
// Native fragment implementation of the same postsynthesis stages as the Gas
// compute runtime. Register words have the packed GasUniforms ABI (152 uints).
// The TEV source supplies guest RGBA8 bytes in reverse memory order. Filtering
// of encoded density pairs still needs a separate PICA hardware comparison.
inline constexpr std::string_view GasFragmentHelpers = R"GLSL(
layout(set=3, binding=5, std430) readonly buffer MetalGasRegisters { uint metal_gas_words[152]; };
float metalGasHalf(uint value) { return unpackHalf2x16(value & 65535u).x; }
float metalGasByte(uint word, uint shift) { return float((word >> shift) & 255u) / 255.0; }
float metalGasDiff(uint word, uint shift) {
    uint value = (word >> shift) & 255u;
    return float(value & 127u) * ((value & 128u) != 0u ? -1.0 : 1.0) / 127.0;
}
vec4 metalGasSynthesize(float xy, vec4 encoded) {
    uvec4 bytes = uvec4(round(clamp(encoded, 0.0, 1.0) * 255.0));
    float plain = metalGasHalf((bytes.a << 8u) | bytes.b);
    float depth_density = metalGasHalf((bytes.g << 8u) | bytes.r);
    float density = (metal_gas_words[149] != 0u ? depth_density : plain) * metalGasHalf(metal_gas_words[145]);
    uint light_xy = metal_gas_words[146], light_z = metal_gas_words[147], z_color = metal_gas_words[148];
    float xy_weight = xy * (1.0 - density * metalGasByte(light_xy, 16u));
    float xy_light = xy_weight * metalGasByte(light_xy, 8u) + (1.0 - xy_weight) * metalGasByte(light_xy, 0u);
    float z_weight = metalGasByte(z_color, 0u) * (1.0 - density * metalGasByte(light_z, 16u));
    float z_light = z_weight * metalGasByte(light_z, 8u) + (1.0 - z_weight) * metalGasByte(light_z, 0u);
    float position = clamp((z_color & 256u) != 0u ? xy_light + z_light : density, 0.0, 1.0) * 8.0;
    uint index = min(uint(position), 7u);
    uint base = metal_gas_words[index + 8u], diff = metal_gas_words[index];
    vec3 rgb = clamp(vec3(metalGasByte(base, 0u), metalGasByte(base, 8u), metalGasByte(base, 16u)) +
                     vec3(metalGasDiff(diff, 0u), metalGasDiff(diff, 8u), metalGasDiff(diff, 16u)) *
                     (position - float(index)), 0.0, 1.0);
    float fog_position = depth_density * metalGasHalf(metal_gas_words[144]) * 128.0;
    float fog_index = clamp(floor(fog_position), 0.0, 127.0);
    uint entry = metal_gas_words[16u + min(uint(fog_index), 127u)];
    int slope = (int(entry & 8191u) ^ 4096) - 4096;
    float alpha = clamp(float((entry >> 13u) & 2047u) / 2047.0 +
                        float(slope) / 2047.0 * (fog_position - fog_index), 0.0, 1.0);
    return vec4(rgb, alpha);
}
)GLSL";
}
