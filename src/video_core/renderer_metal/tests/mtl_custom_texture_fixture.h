// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "core/frontend/image_interface.h"
#include "video_core/custom_textures/material.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace Metal::Test::Custom {
using Color = std::array<u8, 4>;
inline void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Materials contain already decoded pixels; no disk, frontend, or emulator is opened.
class Image final : public Frontend::ImageInterface {
public:
    bool DecodePNG(std::vector<u8>&, u32&, u32&, std::span<const u8>) override { return false; }
    bool DecodeDDS(std::vector<u8>&, u32&, u32&, ddsktx_format&, std::span<const u8>) override { return false; }
    bool EncodePNG(const std::string&, u32, u32, std::span<const u8>) override { return false; }
};
struct Maps {
    VideoCore::CustomTexture color, normal;
    VideoCore::Material material{};
    Maps(Image& image, u32 width, u32 height, Color color_value, Color normal_value,
         bool has_normal = true) : color{image}, normal{image} {
        const auto fill = [&](VideoCore::CustomTexture& texture, VideoCore::MapType type, Color value) {
            texture.width = width; texture.height = height;
            texture.format = VideoCore::CustomPixelFormat::RGBA8;
            texture.type = type; texture.data.resize(width * height * 4);
            for (std::size_t i = 0; i < texture.data.size(); ++i) texture.data[i] = value[i % 4];
        };
        fill(color, VideoCore::MapType::Color, color_value);
        fill(normal, VideoCore::MapType::Normal, normal_value);
        material.width = width; material.height = height;
        material.format = VideoCore::CustomPixelFormat::RGBA8;
        material.textures = {&color, has_normal ? &normal : nullptr};
        material.state = VideoCore::DecodeState::Decoded;
    }
};

using Format = VideoCore::CustomPixelFormat;
constexpr std::array Formats{Format::RGBA8, Format::BC1, Format::BC3, Format::BC5,
                             Format::BC7, Format::ASTC4, Format::ASTC6, Format::ASTC8};
inline std::array<unsigned, 3> Block(Format format) {
    switch (format) {
    case Format::RGBA8: return {1, 1, 4};
    case Format::BC1: return {4, 4, 8};
    case Format::BC3: case Format::BC5: case Format::BC7: case Format::ASTC4: return {4, 4, 16};
    case Format::ASTC6: return {6, 6, 16};
    case Format::ASTC8: return {8, 6, 16}; // Matches the existing VK ASTC8 enum mapping.
    default: throw std::runtime_error("Unknown fixture format");
    }
}
inline Color BlockColor(Format format, unsigned x, unsigned y, unsigned seed) {
    const auto index = (x + 3 * y + seed) % 5;
    constexpr std::array<Color, 5> colors{{
        {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
        {255, 255, 0, 255}, {0, 0, 0, 0}}};
    if (format == Format::BC1) return colors[index];
    if (format == Format::BC3) {
        auto color = colors[index]; color[3] = u8(32 + index * 40); return color;
    }
    if (format == Format::BC5) return {u8(32 + index * 40), u8(224 - index * 40), 0, 255};
    // Even components are exactly representable by BC7 mode6 with P=0.
    return {u8(32 + index * 40), u8(224 - index * 40), u8(64 + index * 32), u8(48 + index * 40)};
}
inline std::vector<u8> ConstantBlock(Format format, Color color) {
    std::vector<u8> bytes(Block(format)[2]);
    const auto put16 = [&](unsigned offset, unsigned value) {
        bytes[offset] = u8(value); bytes[offset + 1] = u8(value >> 8);
    };
    if (format == Format::RGBA8) return {color.begin(), color.end()};
    if (format == Format::BC1 || format == Format::BC3) {
        const unsigned offset = format == Format::BC3 ? 8 : 0;
        const unsigned rgb = (unsigned(color[0] >> 3) << 11) |
                             (unsigned(color[1] >> 2) << 5) | (color[2] >> 3);
        put16(offset, rgb);
        if (format == Format::BC1 && !color[3]) {
            put16(2, 0xffff);
            std::fill(bytes.begin() + 4, bytes.end(), 0xff); // Transparent selector3.
        }
        if (format == Format::BC3) bytes[0] = color[3]; // All alpha selectors choose endpoint0.
    } else if (format == Format::BC5) {
        bytes[0] = color[0]; bytes[8] = color[1]; // Two BC4 blocks, all endpoint0 selectors.
    } else if (format == Format::BC7) {
        // Microsoft BC7 mode6: unary mode, component-major pair of7-bit endpoints,
        // two endpoint P-bits, then indices (index0 has its fix-up MSB omitted).
        unsigned position{};
        const auto put = [&](unsigned value, unsigned count) {
            for (unsigned bit = 0; bit < count; ++bit, ++position)
                bytes[position / 8] |= u8(((value >> bit) & 1) << (position % 8));
        };
        put(1U << 6, 7);
        for (const auto channel : color) { put(channel >> 1, 7); put(channel >> 1, 7); }
        put(0, 2); put(0, 3);
        for (unsigned i = 1; i < 16; ++i) put(0, 4);
        Check(position == 128, "BC7 fixture did not fill exactly one block");
    } else {
        // Khronos Data Format1.4, section23.22: LDR void-extent, reserved bits1,
        // all extent coordinates1, RGBA UNORM16. Each block has an exact known color.
        std::fill(bytes.begin(), bytes.begin() + 8, 0xff);
        bytes[0] = 0xfc; bytes[1] = 0xfd;
        for (unsigned channel = 0; channel < 4; ++channel) put16(8 + channel * 2, color[channel] * 257U);
    }
    return bytes;
}
inline void EncodeMap(VideoCore::CustomTexture& texture, Format format, unsigned seed) {
    texture.format = format; texture.data.clear();
    const auto block = Block(format);
    const auto columns = (texture.width + block[0] - 1) / block[0];
    const auto rows = (texture.height + block[1] - 1) / block[1];
    for (unsigned y = 0; y < rows; ++y) for (unsigned x = 0; x < columns; ++x) {
        const auto encoded = ConstantBlock(format, BlockColor(format, x, y, seed));
        texture.data.insert(texture.data.end(), encoded.begin(), encoded.end());
    }
}
} // namespace Metal::Test::Custom
