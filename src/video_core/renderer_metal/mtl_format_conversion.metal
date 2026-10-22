// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include <metal_stdlib>
using namespace metal;

struct ClearParams { float4 color; float4 depth; };
struct ClearVertex { float4 position [[position]]; };
vertex ClearVertex clear_region_vertex(uint index [[vertex_id]]) {
    return {float4(index == 1u ? 3.0f : -1.0f, index == 2u ? 3.0f : -1.0f, 0.0f, 1.0f)};
}
fragment float4 clear_region_color(constant ClearParams& info [[buffer(0)]]) {
    return info.color;
}
struct DepthClearParams { float value; uint stencil; uint count; uint bytes; };
kernel void clear_depth_planes(device uchar* depth [[buffer(0)]],
                                device uchar* stencil [[buffer(1)]],
                                constant DepthClearParams& info [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= info.count) return;
    const uint word = info.bytes == 2u ? uint(round(info.value * 65535.0f)) : as_type<uint>(info.value);
    for (uint byte = 0; byte < info.bytes; ++byte) depth[ulong(gid) * info.bytes + byte] = uchar(word >> (byte * 8u));
    if (info.stencil & 256u) stencil[gid] = uchar(info.stencil);
}

struct ConversionParams {
    uint2 source_origin;
    uint2 dest_origin;
    uint2 extent;
    uint row_bytes;
    uint format;
};

uint depth24(float value) {
    // Truncation matches the PICA staging readback path; keep both endpoints in uint range.
    return uint(clamp(value, 0.0f, 1.0f) * 16777215.0f);
}

kernel void reinterpret_color16(texture2d<float, access::read> source [[texture(0)]],
                                texture2d<float, access::write> dest [[texture(2)]],
                                constant ConversionParams& info [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.extent)) return;
    const uint4 color = uint4(round(clamp(source.read(info.source_origin + gid), 0.0f, 1.0f) * 255.0f));
    uint word;
    switch (info.format & 255u) {
    case 2u: word = ((color.r >> 3u) << 11u) | ((color.g >> 3u) << 6u) |
                    ((color.b >> 3u) << 1u) | (color.a >> 7u); break;
    case 3u: word = ((color.r >> 3u) << 11u) | ((color.g >> 2u) << 5u) | (color.b >> 3u); break;
    case 4u: word = ((color.r >> 4u) << 12u) | ((color.g >> 4u) << 8u) |
                    ((color.b >> 4u) << 4u) | (color.a >> 4u); break;
    case 5u: word = (((color.r + color.g + color.b) / 3u) << 8u) | color.a; break;
    default: word = (color.r << 8u) | color.g; break;
    }
    uint4 result;
    switch (info.format >> 8u) {
    case 2u:
        result = uint4((word >> 11u) & 31u, (word >> 6u) & 31u, (word >> 1u) & 31u, 0u);
        result.rgb = (result.rgb << 3u) | (result.rgb >> 2u);
        result.a = (word & 1u) * 255u;
        break;
    case 3u:
        result = uint4((word >> 11u) & 31u, (word >> 5u) & 63u, word & 31u, 255u);
        result.rb = (result.rb << 3u) | (result.rb >> 2u);
        result.g = (result.g << 2u) | (result.g >> 4u);
        break;
    case 4u: result = uint4(word >> 12u, (word >> 8u) & 15u, (word >> 4u) & 15u, word & 15u) * 17u; break;
    case 5u: result = uint4(uint3(word >> 8u), word & 255u); break;
    default: result = uint4(word >> 8u, word & 255u, 0u, 255u); break;
    }
    dest.write(float4(result) / 255.0f, info.dest_origin + gid);
}

kernel void depth_stencil_to_buffer(
    depth2d<float, access::read> depth [[texture(0)]],
    texture2d<uint, access::read> stencil [[texture(1)]],
    device uchar* output [[buffer(0)]],
    constant ConversionParams& info [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.extent)) return;
    const uint2 coord = info.source_origin + gid;
    const float value = depth.read(coord);
    const uint d = depth24(value);
    const uint s = info.format & 256u ? stencil.read(coord).r & 255u : 0u;
    const uint format = info.format & 255u;
    const uint bytes = format == 1u ? 2u : 4u;
    const uint word = format == 1u ? uint(clamp(value, 0.0f, 1.0f) * 65535.0f)
                      : format == 2u ? as_type<uint>(value) : (d << 8u) | s;
    const ulong dest = ulong(gid.y) * info.row_bytes + ulong(gid.x) * bytes;
    for (uint byte = 0; byte < bytes; ++byte) output[dest + byte] = uchar(word >> (8u * byte));
}

kernel void depth_stencil_to_rgba8(
    depth2d<float, access::read> depth [[texture(0)]],
    texture2d<uint, access::read> stencil [[texture(1)]],
    texture2d<float, access::write> color [[texture(2)]],
    constant ConversionParams& info [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.extent)) return;
    const uint2 coord = info.source_origin + gid;
    const uint d = depth24(depth.read(coord));
    const uint s = info.format & 256u ? stencil.read(coord).r & 255u : 0u;
    const uint4 rgba{s, (d >> 16u) & 255u, (d >> 8u) & 255u, d & 255u};
    color.write(float4(rgba) / 255.0f, info.dest_origin + gid);
}

kernel void buffer_to_depth_stencil(
    const device uchar* packed [[buffer(0)]],
    constant ConversionParams& info [[buffer(1)]],
    device float* depth [[buffer(2)]],
    device uchar* stencil [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.extent)) return;
    const uint bytes = info.format == 1u ? 2u : 4u;
    const ulong source = ulong(gid.y) * info.row_bytes + ulong(gid.x) * bytes;
    uint word = 0;
    for (uint byte = 0; byte < bytes; ++byte) word |= uint(packed[source + byte]) << (8u * byte);
    float normalized;
    if (info.format == 2u) {
        normalized = as_type<float>(word);
    } else {
        const uint d = info.format == 1u ? word : word >> 8u;
        const float maximum = info.format == 1u ? 65535.0f : 16777215.0f;
        normalized = float(d) / maximum;
        // Choose the adjacent float when needed to preserve a truncating guest roundtrip.
        if (uint(normalized * maximum) < d) normalized = nextafter(normalized, 1.0f);
    }
    const ulong dest = ulong(gid.y) * info.extent.x + gid.x;
    depth[dest] = normalized;
    stencil[dest] = info.format == 0u ? packed[source] : uchar(0);
}

struct DepthBlitParams {
    uint2 source_origin;
    uint2 source_extent;
    uint2 dest_extent;
    uint flips;
    uint padding;
    uint2 dest_origin;
};

kernel void blit_depth_stencil(
    depth2d<float, access::read> source_depth [[texture(0)]],
    texture2d<uint, access::read> source_stencil [[texture(1)]],
    constant DepthBlitParams& info [[buffer(1)]],
    device uchar* depth [[buffer(2)]],
    device uchar* stencil [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.dest_extent)) return;
    // Exact nearest-neighbor mapping at destination pixel centers. Integer division
    // avoids a rounded floating-point coordinate choosing the neighboring source pixel.
    uint2 coord = uint2((ulong2(gid) * ulong2(info.source_extent) + ulong2(info.source_extent / 2u)) /
                        ulong2(info.dest_extent));
    if (info.flips & 1u) coord.x = info.source_extent.x - 1u - coord.x;
    if (info.flips & 2u) coord.y = info.source_extent.y - 1u - coord.y;
    coord += info.source_origin;
    const ulong dest = ulong(gid.y) * info.dest_extent.x + gid.x;
    const float value = source_depth.read(coord);
    const uint bytes = info.padding ? 2u : 4u;
    const uint word = info.padding ? uint(round(clamp(value, 0.0f, 1.0f) * 65535.0f)) : as_type<uint>(value);
    for (uint byte = 0; byte < bytes; ++byte) depth[dest * bytes + byte] = uchar(word >> (8u * byte));
    if (info.flips & 16u)
        stencil[dest] = info.flips & 8u ? uchar(source_stencil.read(coord).r & 255u) : uchar(0);
}

kernel void blit_color(
    texture2d<float, access::sample> source [[texture(0)]],
    texture2d<float, access::write> dest [[texture(2)]],
    sampler filtering [[sampler(0)]],
    constant DepthBlitParams& info [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.dest_extent)) return;
    float2 coord;
    if (info.flips & 4u) {
        coord = (float2(gid) + 0.5f) * float2(info.source_extent) / float2(info.dest_extent);
        if (info.flips & 1u) coord.x = float(info.source_extent.x) - coord.x;
        if (info.flips & 2u) coord.y = float(info.source_extent.y) - coord.y;
    } else {
        uint2 index = uint2((ulong2(gid) * ulong2(info.source_extent) + ulong2(info.source_extent / 2u)) /
                            ulong2(info.dest_extent));
        if (info.flips & 1u) index.x = info.source_extent.x - 1u - index.x;
        if (info.flips & 2u) index.y = info.source_extent.y - 1u - index.y;
        coord = float2(index) + 0.5f;
    }
    coord += float2(info.source_origin);
    dest.write(source.sample(filtering, coord, level(0.0f)), info.dest_origin + gid);
}

uint4 pica_transfer_color(uint4 value, uint format) {
    if (format == 1u || format == 3u) value.a = 255u;
    if (format == 2u || format == 3u) {
        const uint3 bits = format == 2u ? uint3(5, 5, 5) : uint3(5, 6, 5);
        const uint3 packed = value.rgb >> (8u - bits);
        value.rgb = (packed << (8u - bits)) | (packed >> (2u * bits - 8u));
        if (format == 2u) value.a = (value.a >> 7u) * 255u;
    } else if (format == 4u) {
        value = (value >> 4u) * 17u;
    }
    return value;
}

kernel void pica_color_transfer(texture2d<float, access::sample> source [[texture(0)]],
                                 texture2d<float, access::write> dest [[texture(2)]],
                                 sampler nearest [[sampler(0)]],
                                 constant DepthBlitParams& info [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (any(gid >= info.dest_extent)) return;
    const uint2 scale = info.source_extent / info.dest_extent;
    uint4 sum = 0;
    for (uint y = 0; y < scale.y; ++y) for (uint x = 0; x < scale.x; ++x) {
        uint2 coord = gid * scale + uint2(x, y);
        if (info.flips & 1u) coord.x = info.source_extent.x - 1u - coord.x;
        if (info.flips & 2u) coord.y = info.source_extent.y - 1u - coord.y;
        coord += info.source_origin;
        const float4 color = source.sample(nearest, float2(coord) + 0.5f, level(0.0f));
        sum += uint4(round(clamp(color, 0.0f, 1.0f) * 255.0f));
    }
    // Average decoded integer channels, then reduce to the destination guest precision.
    // Quantizing before sampling the result keeps in-cache pixels identical to a guest flush.
    const uint4 result = pica_transfer_color(sum / (scale.x * scale.y), info.padding);
    dest.write(float4(result) / 255.0f, info.dest_origin + gid);
}
