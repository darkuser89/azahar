// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include <metal_stdlib>
using namespace metal;

struct CodecParams {
    uint width, height, stride, format;
    uint bits_per_pixel, flags, linear_bpp, padding;
    ulong byte_begin, byte_count;
};
uint morton(uint x, uint y) {
    uint result = 0;
    for (uint i = 0; i < 3; ++i) result |= ((x >> i) & 1u) << (2u * i) | ((y >> i) & 1u) << (2u * i + 1u);
    return result;
}
uint unmorton(uint index, uint shift) {
    uint result = 0;
    for (uint i = 0; i < 3; ++i) result |= ((index >> (2u * i + shift)) & 1u) << i;
    return result;
}
uint load_word(const device uchar* bytes, ulong offset, uint count) {
    uint word = 0;
    for (uint i = 0; i < count; ++i) word |= uint(bytes[offset + i]) << (8u * i);
    return word;
}
ulong load_long(const device uchar* bytes, ulong offset) {
    ulong word = 0;
    for (uint i = 0; i < 8; ++i) word |= ulong(bytes[offset + i]) << (8u * i);
    return word;
}
uint expand5(uint value) { return ((value << 3u) | (value >> 2u)) & 255u; }
uint expand6(uint value) { return (value << 2u) | (value >> 4u); }

uint4 etc_color(const device uchar* input, ulong tile, uint x, uint y, bool alpha) {
    const ulong offset = tile + ((x / 4u) + 2u * (y / 4u)) * (alpha ? 16u : 8u);
    x %= 4u;
    y %= 4u;
    const ulong word = load_long(input, offset + (alpha ? 8u : 0u));
    const uint texel = x * 4u + y;
    const bool second = ((word >> 32u) & 1u) ? y >= 2u : x >= 2u;
    uint3 base;
    const uint3 shifts{56, 48, 40};
    for (uint c = 0; c < 3; ++c) {
        const uint shift = shifts[c];
        if ((word >> 33u) & 1u) {
            int value = int((word >> (shift + 3u)) & 31u);
            const int delta = (int((word >> shift) & 7u) ^ 4) - 4;
            if (second) value += delta;
            base[c] = expand5(uint(uchar(value)));
        } else {
            const uint value = uint((word >> (shift + (second ? 0u : 4u))) & 15u);
            base[c] = value * 17u;
        }
    }
    constexpr int2 table[8] = {int2(2,8), int2(5,17), int2(9,29), int2(13,42),
                              int2(18,60), int2(24,80), int2(33,106), int2(47,183)};
    const uint table_index = uint((word >> (second ? 34u : 37u)) & 7u);
    int modifier = table[table_index][uint((word >> texel) & 1u)];
    if ((word >> (texel + 16u)) & 1u) modifier = -modifier;
    const uint a = alpha ? uint((load_long(input, offset) >> (4u * texel)) & 15u) * 17u : 255u;
    return uint4(uint3(clamp(int3(base) + modifier, 0, 255)), a);
}
ulong guest_pixel(uint x, uint y, constant CodecParams& info) {
    if (!(info.flags & 1u)) return ulong(y) * info.stride + x;
    return (ulong(y / 8u) * (info.stride / 8u) + x / 8u) * 64u + morton(x, y);
}
uint4 decode_color(const device uchar* input, uint x, uint y, constant CodecParams& info) {
    const ulong pixel = guest_pixel(x, y, info);
    const ulong offset = pixel * info.bits_per_pixel / 8u;
    const uint format = info.format;
    if (format >= 12u) {
        const ulong tile = (ulong(y / 8u) * (info.stride / 8u) + x / 8u) * (format == 13u ? 64u : 32u);
        return etc_color(input, tile, x % 8u, y % 8u, format == 13u);
    }
    const uint word = load_word(input, offset, max(1u, info.bits_per_pixel / 8u));
    switch (format) {
    case 0: return uint4(word >> 24u, (word >> 16u) & 255u, (word >> 8u) & 255u, word & 255u);
    case 1: return uint4(word >> 16u, (word >> 8u) & 255u, word & 255u, 255u);
    case 2: return uint4(expand5((word >> 11u) & 31u), expand5((word >> 6u) & 31u), expand5((word >> 1u) & 31u), (word & 1u) * 255u);
    case 3: return uint4(expand5((word >> 11u) & 31u), expand6((word >> 5u) & 63u), expand5(word & 31u), 255u);
    case 4: return uint4((word >> 12u) & 15u, (word >> 8u) & 15u, (word >> 4u) & 15u, word & 15u) * 17u;
    case 5: return uint4(word >> 8u, word >> 8u, word >> 8u, word & 255u);
    case 6: return uint4(word >> 8u, word & 255u, 0u, 255u);
    case 7: return uint4(word, word, word, 255u);
    case 8: return uint4(0u, 0u, 0u, word);
    case 9: return uint4(word >> 4u, word >> 4u, word >> 4u, word & 15u) * 17u;
    case 10: { const uint i = ((word >> ((pixel & 1u) * 4u)) & 15u) * 17u; return uint4(i, i, i, 255u); }
    default: return uint4(0u, 0u, 0u, ((word >> ((pixel & 1u) * 4u)) & 15u) * 17u);
    }
}
kernel void decode_guest_texture(const device uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
    constant CodecParams& info [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= info.width || gid.y >= info.height) return;
    const bool flip = bool(info.flags & 1u) != bool(info.flags & 4u);
    const uint y = flip ? info.height - 1u - gid.y : gid.y;
    const ulong pixel = guest_pixel(gid.x, y, info);
    const ulong source = pixel * info.bits_per_pixel / 8u;
    const ulong dest = (ulong(gid.y) * info.width + gid.x) * info.linear_bpp;
    if (info.format < 14u && (info.format > 4u || (info.flags & 2u))) {
        const uint4 color = decode_color(input, gid.x, y, info);
        for (uint c = 0; c < 4; ++c) output[dest + c] = uchar(color[c]);
    } else {
        uint word = load_word(input, source, info.bits_per_pixel / 8u);
        if (info.format == 17u) word = (word << 8u) | (word >> 24u);
        if (info.format == 16u && (info.flags & 2u)) {
            float value = float(word) / 16777215.0f;
            if (uint(value * 16777215.0f) < word) value = nextafter(value, 1.0f);
            word = as_type<uint>(value);
        }
        for (uint c = 0; c < info.linear_bpp; ++c) output[dest + c] = uchar(word >> (8u * c));
    }
}
uint encode_word(const device uchar* input, uint x, uint y, constant CodecParams& info) {
    const bool flip = bool(info.flags & 1u) != bool(info.flags & 4u);
    if (flip) y = info.height - 1u - y;
    const ulong source = (ulong(y) * info.width + x) * info.linear_bpp;
    if (!(info.flags & 2u) && info.format <= 4u) return load_word(input, source, info.linear_bpp);
    if (info.format == 14u) return load_word(input, source, 2);
    if (info.format == 16u) {
        const uint word = load_word(input, source, 4);
        if (!(info.flags & 2u)) return word & 0xFFFFFFu;
        const float value = as_type<float>(word);
        return isfinite(value) ? uint(clamp(value, 0.0f, 1.0f) * 16777215.0f) : 0u;
    }
    if (info.format == 17u) { const uint word = load_word(input, source, 4); return (word >> 8u) | (word << 24u); }
    const uint4 color{input[source], input[source + 1], input[source + 2], input[source + 3]};
    const uint i = (color.r + color.g + color.b) / 3u;
    switch (info.format) {
    case 0: return (color.r << 24u) | (color.g << 16u) | (color.b << 8u) | color.a;
    case 1: return (color.r << 16u) | (color.g << 8u) | color.b;
    case 2: return ((color.r >> 3u) << 11u) | ((color.g >> 3u) << 6u) | ((color.b >> 3u) << 1u) | (color.a >> 7u);
    case 3: return ((color.r >> 3u) << 11u) | ((color.g >> 2u) << 5u) | (color.b >> 3u);
    case 4: return ((color.r >> 4u) << 12u) | ((color.g >> 4u) << 8u) | ((color.b >> 4u) << 4u) | (color.a >> 4u);
    case 5: return (i << 8u) | color.a;
    case 6: return (color.r << 8u) | color.g;
    case 7: return i;
    case 8: return color.a;
    case 9: return (i & 240u) | (color.a >> 4u);
    case 10: return i >> 4u;
    default: return color.a >> 4u;
    }
}
uint2 position(ulong pixel, constant CodecParams& info) {
    if (!(info.flags & 1u)) return uint2(uint(pixel % info.stride), uint(pixel / info.stride));
    const ulong tile = pixel / 64u;
    const uint index = uint(pixel % 64u);
    return uint2(uint(tile % (info.stride / 8u)) * 8u + unmorton(index, 0),
                 uint(tile / (info.stride / 8u)) * 8u + unmorton(index, 1));
}
kernel void encode_guest_texture(const device uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
    constant CodecParams& info [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= info.byte_count) return;
    const ulong byte = info.byte_begin + gid;
    if (info.bits_per_pixel == 4u) {
        uchar result = output[byte];
        for (uint n = 0; n < 2; ++n) {
            const uint2 p = position(byte * 2u + n, info);
            if (p.x < info.width && p.y < info.height)
                result = uchar((uint(result) & ~(15u << (n * 4u))) | (encode_word(input, p.x, p.y, info) << (n * 4u)));
        }
        output[byte] = result;
    } else {
        const uint bpp = info.bits_per_pixel / 8u;
        const uint2 p = position(byte / bpp, info);
        if (p.x < info.width && p.y < info.height)
            output[byte] = uchar(encode_word(input, p.x, p.y, info) >> (8u * uint(byte % bpp)));
    }
}

struct DepthCodecParams { CodecParams codec; uint2 origin; uint2 padding; };
kernel void encode_guest_depth(depth2d<float, access::read> depth [[texture(0)]],
    texture2d<uint, access::read> stencil [[texture(1)]], device uchar* output [[buffer(1)]],
    constant DepthCodecParams& params [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    constant CodecParams& info = params.codec;
    const uint bpp = info.bits_per_pixel / 8u;
    const ulong end = info.byte_begin + info.byte_count;
    const ulong pixel = info.byte_begin / bpp + gid;
    if (pixel >= (end + bpp - 1u) / bpp) return;
    uint2 p = position(pixel, info);
    if (p.x >= info.width || p.y >= info.height) return; // Preserve stride padding.
    if (bool(info.flags & 1u) != bool(info.flags & 4u)) p.y = info.height - 1u - p.y;
    const uint2 coord = params.origin + p;
    const float value = depth.read(coord);
    // D16 native storage is already quantized. Rounding recovers every stored
    // UNORM16 bit pattern; D24 retains the existing truncating guest contract.
    uint word = info.format == 14u ? uint(round(clamp(value, 0.0f, 1.0f) * 65535.0f))
              : isfinite(value) ? uint(clamp(value, 0.0f, 1.0f) * 16777215.0f) : 0u;
    if (info.format == 17u) word |= (stencil.read(coord).r & 255u) << 24u;
    for (uint c = 0; c < bpp; ++c) {
        const ulong byte = pixel * bpp + c;
        if (byte >= info.byte_begin && byte < end) output[byte] = uchar(word >> (8u * c));
    }
}

struct FillParams { uint pattern, size, phase, padding; ulong count, padding2; };
kernel void fill_guest_memory(device uchar* output [[buffer(0)]], constant FillParams& info [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= info.count) return;
    output[gid] = uchar(info.pattern >> (8u * uint((ulong(gid) + info.phase) % info.size)));
}

struct CopyParams {
    ulong count;
    uint input_width, input_gap, output_width, output_gap;
    ulong padding;
};
kernel void copy_guest_memory(const device uchar* input [[buffer(0)]], device uchar* output [[buffer(1)]],
                               constant CopyParams& info [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= info.count) return;
    const ulong source = ulong(gid) + ulong(gid / info.input_width) * info.input_gap;
    const ulong dest = ulong(gid) + ulong(gid / info.output_width) * info.output_gap;
    output[dest] = input[source];
}
