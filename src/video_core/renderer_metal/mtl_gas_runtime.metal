// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct GasUniforms {
    uint color_lut[16], fog_lut[128];
    uint attenuation, reciprocal_maximum, light_xy, light_z, light_z_color, shading_depth;
    uint delta_z_depth, padding;
};
struct GasParams { uint pixels, delta_z_depth; uint2 padding; };

uint gas_swap16(uint value) { return ((value & 255u) << 8u) | ((value >> 8u) & 255u); }
kernel void pica_gas_reciprocal(const device atomic_uint& maximum [[buffer(0)]],
    device uint& output [[buffer(1)]]) {
    const ushort bits = ushort(atomic_load_explicit(&maximum, memory_order_relaxed));
    output = uint(as_type<ushort>(half(1.0f / float(as_type<half>(bits)))));
}
float2 gas_unpack(uint raw) {
    return float2(float(as_type<half>(ushort(gas_swap16(raw)))),
                  float(as_type<half>(ushort(gas_swap16(raw >> 16u)))));
}
uint gas_pack(float2 density) {
    const ushort plain = as_type<ushort>(half(density.x));
    const ushort depth = as_type<ushort>(half(density.y));
    return gas_swap16(plain) | (gas_swap16(depth) << 16u);
}
float gas_attenuation(float surface_depth, float fragment_depth, uint delta_z_depth) {
    const float attenuation = clamp((surface_depth - fragment_depth) *
        (float(delta_z_depth & 0xffffffu) / 256.0f), 0.0f, 1.0f);
    switch ((delta_z_depth >> 24u) & 3u) {
    case 0: return 0.0f;
    case 1: return 1.0f;
    case 2: return 1.0f - attenuation;
    default: return attenuation;
    }
}
kernel void pica_gas_accumulate(device uint* gas [[buffer(0)]],
    const device float4* contribution [[buffer(1)]], const device float* scene_depth [[buffer(2)]],
    device atomic_uint& maximum [[buffer(3)]], constant GasParams& info [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= info.pixels || contribution[pixel].z == 0.0f) return;
    const float density = contribution[pixel].x;
    const float attenuation = gas_attenuation(scene_depth[pixel], contribution[pixel].y,
                                             info.delta_z_depth);
    const float2 accumulated = gas_unpack(gas[pixel]) + float2(density, density * attenuation);
    const uint packed = gas_pack(accumulated);
    gas[pixel] = packed;
    // Nonnegative half-float bit patterns have the same ordering as their values.
    const uint plain = gas_swap16(packed);
    if (!(plain & 0x8000u) && (plain & 0x7fffu) <= 0x7c00u)
        atomic_fetch_max_explicit(&maximum, plain, memory_order_relaxed);
}

float gas_byte(uint word, uint shift) { return float((word >> shift) & 255u) / 255.0f; }
float gas_difference(uint word, uint shift) {
    const uint value = (word >> shift) & 255u;
    return float(value & 127u) * ((value & 128u) ? -1.0f : 1.0f) / 127.0f;
}
float3 gas_color(float coordinate, constant GasUniforms& config) {
    const float position = clamp(coordinate, 0.0f, 1.0f) * 8.0f;
    const uint index = min(uint(position), 7u);
    const float fraction = position - float(index);
    const uint base = config.color_lut[index + 8u], diff = config.color_lut[index];
    return clamp(float3(gas_byte(base, 0), gas_byte(base, 8), gas_byte(base, 16)) +
        float3(gas_difference(diff, 0), gas_difference(diff, 8), gas_difference(diff, 16)) * fraction,
        0.0f, 1.0f);
}
float gas_opacity(float coordinate, constant GasUniforms& config) {
    const float position = coordinate * 128.0f;
    const float index = clamp(floor(position), 0.0f, 127.0f);
    // Bound the integer access too: conversion of a nonfinite coordinate must
    // never turn an unverified arithmetic special value into an out-of-range read.
    const uint entry = config.fog_lut[min(uint(index), 127u)];
    const int diff = (int(entry & 8191u) ^ 4096) - 4096;
    return clamp(float((entry >> 13u) & 2047u) / 2047.0f +
        float(diff) / 2047.0f * (position - index), 0.0f, 1.0f);
}
kernel void pica_gas_synthesize(const device uint* gas [[buffer(0)]],
    const device float* lighting [[buffer(1)]], device float4* output [[buffer(2)]],
    constant GasUniforms& config [[buffer(3)]], constant GasParams& info [[buffer(4)]],
    uint pixel [[thread_position_in_grid]]) {
    if (pixel >= info.pixels) return;
    const float2 density = gas_unpack(gas[pixel]);
    const float selected = config.shading_depth ? density.y : density.x;
    const float normalized = selected * float(as_type<half>(ushort(config.reciprocal_maximum)));
    const float xy = lighting[pixel];
    const float xy_weight = xy * (1.0f - normalized * gas_byte(config.light_xy, 16));
    const float xy_light = xy_weight * gas_byte(config.light_xy, 8) +
        (1.0f - xy_weight) * gas_byte(config.light_xy, 0);
    const float z_weight = gas_byte(config.light_z_color, 0) *
        (1.0f - normalized * gas_byte(config.light_z, 16));
    const float z_light = z_weight * gas_byte(config.light_z, 8) +
        (1.0f - z_weight) * gas_byte(config.light_z, 0);
    const float coordinate = (config.light_z_color & 256u) ? xy_light + z_light : normalized;
    const float attenuation = float(as_type<half>(ushort(config.attenuation)));
    output[pixel] = float4(gas_color(coordinate, config), gas_opacity(density.y * attenuation, config));
}
