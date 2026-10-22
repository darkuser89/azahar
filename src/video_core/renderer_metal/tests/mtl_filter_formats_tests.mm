// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_filter.h"
#include "common/settings.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Color = std::array<double, 4>;
enum class Encoding { Unorm, Snorm, Half, Float, RGB10, RG11, RGB9, BC1, BC3, BC5, BC7, ASTC };
struct Format {
    MTLPixelFormat metal;
    const char* name;
    Encoding encoding;
    unsigned channels, bits, bw = 1, bh = 1, block_bytes = 0;
    bool srgb = false, bgra = false;
};
#define FORMAT(name, kind, channels, bits) Format{MTLPixelFormat##name, #name, Encoding::kind, channels, bits}
#define SRGB(name, kind, bw, bh, bytes) Format{MTLPixelFormat##name, #name, Encoding::kind, 4, 8, bw, bh, bytes, true}
#define BLOCK(name, kind, bw, bh, bytes) Format{MTLPixelFormat##name, #name, Encoding::kind, 4, 8, bw, bh, bytes}
const std::array Formats{
    FORMAT(R8Unorm, Unorm, 1, 8), FORMAT(R8Snorm, Snorm, 1, 8),
    FORMAT(RG8Unorm, Unorm, 2, 8), FORMAT(RG8Snorm, Snorm, 2, 8),
    FORMAT(RGBA8Unorm, Unorm, 4, 8), SRGB(RGBA8Unorm_sRGB, Unorm, 1, 1, 0),
    FORMAT(RGBA8Snorm, Snorm, 4, 8),
    Format{MTLPixelFormatBGRA8Unorm, "BGRA8Unorm", Encoding::Unorm, 4, 8, 1, 1, 0, false, true},
    Format{MTLPixelFormatBGRA8Unorm_sRGB, "BGRA8Unorm_sRGB", Encoding::Unorm, 4, 8, 1, 1, 0, true, true},
    FORMAT(R16Unorm, Unorm, 1, 16), FORMAT(R16Snorm, Snorm, 1, 16), FORMAT(R16Float, Half, 1, 16),
    FORMAT(RG16Unorm, Unorm, 2, 16), FORMAT(RG16Snorm, Snorm, 2, 16), FORMAT(RG16Float, Half, 2, 16),
    FORMAT(RGBA16Unorm, Unorm, 4, 16), FORMAT(RGBA16Snorm, Snorm, 4, 16), FORMAT(RGBA16Float, Half, 4, 16),
    FORMAT(R32Float, Float, 1, 32), FORMAT(RG32Float, Float, 2, 32), FORMAT(RGBA32Float, Float, 4, 32),
    FORMAT(RGB10A2Unorm, RGB10, 4, 8), FORMAT(RG11B10Float, RG11, 3, 8), FORMAT(RGB9E5Float, RGB9, 3, 8),
    BLOCK(BC1_RGBA, BC1, 4, 4, 8), SRGB(BC1_RGBA_sRGB, BC1, 4, 4, 8),
    BLOCK(BC3_RGBA, BC3, 4, 4, 16), SRGB(BC3_RGBA_sRGB, BC3, 4, 4, 16),
    BLOCK(BC5_RGUnorm, BC5, 4, 4, 16),
    BLOCK(BC7_RGBAUnorm, BC7, 4, 4, 16), SRGB(BC7_RGBAUnorm_sRGB, BC7, 4, 4, 16),
    BLOCK(ASTC_4x4_LDR, ASTC, 4, 4, 16), SRGB(ASTC_4x4_sRGB, ASTC, 4, 4, 16),
    BLOCK(ASTC_6x6_LDR, ASTC, 6, 6, 16), SRGB(ASTC_6x6_sRGB, ASTC, 6, 6, 16),
    BLOCK(ASTC_8x6_LDR, ASTC, 8, 6, 16), SRGB(ASTC_8x6_sRGB, ASTC, 8, 6, 16),
    BLOCK(ASTC_8x8_LDR, ASTC, 8, 8, 16), SRGB(ASTC_8x8_sRGB, ASTC, 8, 8, 16)};
#undef FORMAT
#undef SRGB
#undef BLOCK
constexpr std::array Destinations{4U, 5U, 20U};
constexpr Color Guard{17. / 255, 34. / 255, 51. / 255, 68. / 255};

void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
double Linear(double value) { return value <= .04045 ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4); }
double Srgb(double value) { return value <= .0031308 ? value * 12.92 : 1.055 * std::pow(value, 1. / 2.4) - .055; }
void Put(std::vector<std::byte>& bytes, std::uint32_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) bytes.push_back(std::byte(value >> (i * 8)));
}

struct Fixture {
    unsigned width, height, row_bytes;
    std::vector<std::byte> bytes;
    std::vector<Color> colors;
};

// These known finite dyadic values need only zero and normal binary16 encodings.
unsigned HalfBits(double value) {
    if (!value) return 0;
    const unsigned sign = value < 0 ? 0x8000 : 0;
    value = std::abs(value);
    const int exponent = int(std::floor(std::log2(value)));
    return sign | unsigned(exponent + 15) << 10 |
           unsigned(std::lround((std::ldexp(value, -exponent) - 1) * 1024));
}

Color EncodeTexel(const Format& format, unsigned x, unsigned y, unsigned revision,
                  std::vector<std::byte>& bytes) {
    const auto key = [&](unsigned c) { return x * 37 + y * 61 + c * 83 + revision * 47 + 19; };
    Color color{0, 0, 0, 1};
    if (format.encoding == Encoding::RGB10) {
        std::uint32_t word{};
        for (unsigned c = 0; c < 4; ++c) {
            const unsigned bits = c == 3 ? 2 : 10, raw = key(c) % (1U << bits);
            word |= raw << (c == 3 ? 30 : c * 10); color[c] = double(raw) / ((1U << bits) - 1);
        }
        Put(bytes, word, 4); return color;
    }
    if (format.encoding == Encoding::RG11) {
        std::uint32_t word{};
        for (unsigned c = 0; c < 3; ++c) {
            const unsigned mantissa_bits = c == 2 ? 5 : 6;
            const unsigned exponent = key(c) % 13 == 0 ? 0 : 14 + key(c) % 3;
            const unsigned mantissa = key(c) % (1U << mantissa_bits);
            word |= (exponent << mantissa_bits | mantissa) << (c == 2 ? 22 : c * 11);
            color[c] = exponent ? std::ldexp(1. + double(mantissa) / (1U << mantissa_bits), int(exponent) - 15)
                                : std::ldexp(double(mantissa), 1 - 15 - int(mantissa_bits));
        }
        Put(bytes, word, 4); return color;
    }
    if (format.encoding == Encoding::RGB9) {
        const unsigned exponent = 14 + key(0) % 4;
        std::uint32_t word = exponent << 27;
        for (unsigned c = 0; c < 3; ++c) {
            const unsigned mantissa = key(c) % 512;
            word |= mantissa << (c * 9); color[c] = std::ldexp(double(mantissa), int(exponent) - 24);
        }
        Put(bytes, word, 4); return color;
    }
    for (unsigned c = 0; c < format.channels; ++c) {
        const unsigned component = format.bgra && c != 1 && c != 3 ? 2 - c : c;
        double value{}; std::uint32_t raw{};
        switch (format.encoding) {
        case Encoding::Unorm:
            raw = format.bits == 8 ? key(component) % 256 : (key(component) * 251 + 17) % 65536;
            value = double(raw) / (format.bits == 8 ? 255 : 65535); break;
        case Encoding::Snorm: {
            const int signed_value = format.bits == 8 ? int(key(component) % 256) - 128 :
                                     int((key(component) * 251 + 17) % 65536) - 32768;
            raw = std::uint32_t(signed_value);
            value = std::max(-1., double(signed_value) / (format.bits == 8 ? 127 : 32767)); break;
        }
        case Encoding::Half: case Encoding::Float:
            value = (int(key(component) % 56) - 16) / 16.;
            raw = format.encoding == Encoding::Half ? HalfBits(value) : std::bit_cast<std::uint32_t>(float(value)); break;
        default: throw std::runtime_error("Invalid ordinary fixture format");
        }
        Put(bytes, raw, format.bits / 8);
        color[component] = format.srgb && component < 3 ? Linear(value) : value;
    }
    return color;
}

std::array<unsigned, 4> BlockColor(Encoding encoding, unsigned x, unsigned y, unsigned revision) {
    const unsigned index = (x + y * 3 + revision) % 5;
    constexpr std::array<std::array<unsigned, 4>, 5> Primary{{
        {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}, {0, 0, 0, 0}}};
    if (encoding == Encoding::BC1 || encoding == Encoding::BC3) {
        auto color = Primary[index]; if (encoding == Encoding::BC3) color[3] = 32 + index * 40;
        return color;
    }
    if (encoding == Encoding::BC5) return {32 + index * 40, 224 - index * 40, 0, 255};
    return {32 + index * 40, 224 - index * 40, 64 + index * 32, 48 + index * 40};
}

std::vector<std::byte> EncodeBlock(const Format& format, std::array<unsigned, 4> color) {
    std::vector<std::byte> out(format.block_bytes);
    const auto put16 = [&](unsigned offset, unsigned value) { out[offset] = std::byte(value); out[offset + 1] = std::byte(value >> 8); };
    if (format.encoding == Encoding::BC1 || format.encoding == Encoding::BC3) {
        const unsigned offset = format.encoding == Encoding::BC3 ? 8 : 0;
        put16(offset, (color[0] >> 3) << 11 | (color[1] >> 2) << 5 | (color[2] >> 3));
        if (format.encoding == Encoding::BC1 && !color[3]) {
            put16(2, 65535); std::fill(out.begin() + 4, out.end(), std::byte{255});
        }
        if (format.encoding == Encoding::BC3) out[0] = std::byte(color[3]);
    } else if (format.encoding == Encoding::BC5) {
        out[0] = std::byte(color[0]); out[8] = std::byte(color[1]);
    } else if (format.encoding == Encoding::BC7) {
        // Mode6, identical seven-bit endpoint pairs, P=0, all selectors0.
        unsigned position{};
        const auto put = [&](unsigned value, unsigned count) {
            for (unsigned i = 0; i < count; ++i, ++position)
                out[position / 8] |= std::byte(((value >> i) & 1) << (position % 8));
        };
        put(64, 7); for (auto c : color) { put(c >> 1, 7); put(c >> 1, 7); }
        put(0, 2); put(0, 3); for (unsigned i = 1; i < 16; ++i) put(0, 4);
        Check(position == 128, "Invalid BC7 test block");
    } else {
        // Khronos ASTC LDR void-extent, all extent coordinates1, UNORM16 color.
        std::fill(out.begin(), out.begin() + 8, std::byte{255}); out[0] = std::byte{0xfc}; out[1] = std::byte{0xfd};
        for (unsigned c = 0; c < 4; ++c) put16(8 + c * 2, color[c] * 257);
    }
    return out;
}

Fixture MakeFixture(const Format& format, unsigned width, unsigned height, unsigned revision, bool solid) {
    Fixture fixture{width, height, 0, {}, std::vector<Color>(width * height)};
    if (!format.block_bytes) {
        fixture.row_bytes = width * format.channels * format.bits / 8;
        if (format.encoding == Encoding::RGB10 || format.encoding == Encoding::RG11 || format.encoding == Encoding::RGB9)
            fixture.row_bytes = width * 4;
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x)
            fixture.colors[y * width + x] = EncodeTexel(format, solid ? 0 : x, solid ? 0 : y, revision, fixture.bytes);
    } else {
        const unsigned columns = (width + format.bw - 1) / format.bw;
        const unsigned rows = (height + format.bh - 1) / format.bh;
        fixture.row_bytes = columns * format.block_bytes;
        for (unsigned y = 0; y < rows; ++y) for (unsigned x = 0; x < columns; ++x) {
            const auto color = BlockColor(format.encoding, solid ? 0 : x, solid ? 0 : y, revision);
            const auto block = EncodeBlock(format, color); fixture.bytes.insert(fixture.bytes.end(), block.begin(), block.end());
        }
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
            const auto color = BlockColor(format.encoding, solid ? 0 : x / format.bw, solid ? 0 : y / format.bh, revision);
            for (unsigned c = 0; c < 4; ++c)
                fixture.colors[y * width + x][c] = format.srgb && c < 3 ? Linear(color[c] / 255.) : color[c] / 255.;
        }
    }
    Check(fixture.bytes.size() == fixture.row_bytes * ((height + format.bh - 1) / format.bh), "Invalid encoded fixture size");
    return fixture;
}

id<MTLTexture> Texture(Metal::Device& device, const Format& format, unsigned width, unsigned height, bool target) {
    auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format.metal width:width height:height mipmapped:NO];
    d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (target) d.usage |= MTLTextureUsageRenderTarget;
    return device.CreateTexture(d);
}
id<MTLTexture> View(id<MTLTexture> texture) {
    auto* d = [[MTLTextureViewDescriptor alloc] init]; d.pixelFormat = texture.pixelFormat;
    d.textureType = MTLTextureType2D; d.levelRange = NSMakeRange(0, 1); d.sliceRange = NSMakeRange(0, 1);
    const auto result = [texture newTextureViewWithDescriptor:d]; Check(result != nil, "Unable to create test view"); return result;
}
void Upload(Metal::Device& device, Metal::TransferRuntime& transfers, id<MTLTexture> texture, const Fixture& fixture) {
    const auto buffer = device.CreateBuffer(fixture.bytes.size(), true); std::memcpy(buffer.contents, fixture.bytes.data(), fixture.bytes.size());
    transfers.Upload(buffer, {0, fixture.row_bytes, 0}, {texture, 0, 0, MTLRegionMake2D(0, 0, fixture.width, fixture.height)});
}
double Spline(double x) {
    x = std::abs(x); return x < 1 ? (4 - 6 * x * x + 3 * x * x * x) / 6 : x < 2 ? std::pow(2 - x, 3) / 6 : 0;
}
Color Bicubic(const Fixture& fixture, double x, double y) {
    x -= .5; y -= .5; const int ix = int(std::floor(x)), iy = int(std::floor(y)); Color result{};
    for (int dy = -1; dy <= 2; ++dy) for (int dx = -1; dx <= 2; ++dx) {
        const auto color = fixture.colors[std::clamp(iy + dy, 0, int(fixture.height) - 1) * fixture.width +
                                           std::clamp(ix + dx, 0, int(fixture.width) - 1)];
        const double weight = Spline(x - ix - dx) * Spline(y - iy - dy);
        for (unsigned c = 0; c < 4; ++c) result[c] += color[c] * weight;
    }
    return result;
}
std::vector<Color> Reference(const Fixture& fixture, unsigned width, unsigned height, MTLRegion source,
                             MTLRegion dest, MTLRegion clip, bool fx, bool fy, std::vector<Color> output) {
    Check(output.size() == width * height, "Invalid scalar destination");
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        if (x < clip.origin.x || y < clip.origin.y || x >= clip.origin.x + clip.size.width || y >= clip.origin.y + clip.size.height) continue;
        double u = (x + .5 - dest.origin.x) / dest.size.width, v = (y + .5 - dest.origin.y) / dest.size.height;
        if (fx) u = 1 - u; if (fy) v = 1 - v;
        output[y * width + x] = Bicubic(fixture, source.origin.x + u * source.size.width, source.origin.y + v * source.size.height);
    }
    return output;
}

std::uint64_t Pixels{}, Cases{};
constexpr double FloatBudget = .006;
double MaxFloatError{}, MaxSrgbLinearError{};
unsigned MaxByteError{}, MaxSrgbByteError{};
void Compare(const Format& format, const std::vector<std::byte>& bytes, const std::vector<Color>& expected,
             unsigned width, const std::string& label) {
    const bool floating = format.encoding == Encoding::Float;
    Check(bytes.size() == expected.size() * (floating ? 16 : 4), "Invalid filter readback size");
    for (unsigned i = 0; i < expected.size(); ++i) for (unsigned c = 0; c < 4; ++c) {
        double actual{}, want = expected[i][c], tolerance{}; bool matches{};
        if (floating) {
            float value{}; std::memcpy(&value, bytes.data() + i * 16 + c * 4, 4); actual = value; tolerance = FloatBudget;
            Check(std::isfinite(actual), "Nonfinite filter output"); MaxFloatError = std::max(MaxFloatError, std::abs(actual - want));
            matches = std::abs(actual - want) <= tolerance;
        } else {
            actual = std::to_integer<unsigned>(bytes[i * 4 + c]); want = std::clamp(want, 0., 1.);
            if (format.srgb && c < 3) {
                // Apply the same linear-color precision budget as Float output. sRGB
                // expands near-black errors; a fixed encoded-byte budget changes meaning.
                const double lower = std::lround(Srgb(std::max(0., want - FloatBudget)) * 255) - 1;
                const double upper = std::lround(Srgb(std::min(1., want + FloatBudget)) * 255) + 1;
                MaxSrgbLinearError = std::max(MaxSrgbLinearError, std::abs(Linear(actual / 255.) - want));
                want = std::lround(Srgb(want) * 255);
                MaxSrgbByteError = std::max(MaxSrgbByteError, unsigned(std::abs(actual - want)));
                matches = actual >= lower && actual <= upper;
            } else {
                want = std::lround(want * 255); tolerance = 2;
                MaxByteError = std::max(MaxByteError, unsigned(std::abs(actual - want)));
                matches = std::abs(actual - want) <= tolerance;
            }
        }
        if (!matches) {
            std::cerr << label << " pixel=" << i % width << ',' << i / width << " channel=" << c
                      << " expected=" << want << " actual=" << actual << '\n';
            throw std::runtime_error("Native filter format/alias oracle mismatch");
        }
    }
    Pixels += expected.size(); ++Cases;
}
std::vector<std::byte> Read(Metal::TransferRuntime& transfers, id<MTLTexture> texture) {
    return transfers.Readback({texture, 0, 0, MTLRegionMake2D(0, 0, texture.width, texture.height)},
                             texture.width * (texture.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : 4));
}

void Aliases(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TransferRuntime& transfers,
             Metal::TextureFilterRuntime& filters, bool probe) {
    using F = Settings::TextureFilter;
    std::uint64_t scalar{}, counterpart{};
    for (unsigned revision = 0; revision < (probe ? 1U : 2U); ++revision)
        for (unsigned repeat = 0; repeat < (probe ? 1U : 2U); ++repeat)
            for (const auto index : Destinations) for (unsigned alias = probe ? 4U : 0U; alias < (probe ? 5U : 6U); ++alias)
                for (unsigned f = probe ? unsigned(F::Bicubic) : 1U; f <= (probe ? unsigned(F::Bicubic) : 5U); ++f)
                    for (unsigned flip = 0; flip < 4; ++flip) {
                    const auto& format = Formats[index];
                    const unsigned width = alias >= 4 ? 128 : 16, height = alias >= 4 ? 96 : 12;
                    const unsigned bpp = index == 20 ? 16 : 4;
                    const auto fixture = MakeFixture(format, width, height, revision, false);
                    auto source = Texture(device, format, width, height, true), dest = source;
                    auto initial = fixture; id<MTLBuffer> backing = nil;
                    if (alias == 1) source = View(source);
                    if (alias == 2) { source = View(source); dest = View(dest); }
                    if (alias == 3) { source = View(View(source)); dest = View(View(dest)); }
                    if (alias >= 4) {
                        const auto alignment = [device.Handle() minimumLinearTextureAlignmentForPixelFormat:format.metal];
                        const NSUInteger pitch = ((width * bpp + alignment - 1) / alignment) * alignment;
                        backing = device.CreateBuffer(pitch * (height + 1), true);
                        auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format.metal width:width height:height mipmapped:NO];
                        d.storageMode = MTLStorageModeShared;
                        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
                        d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
                        source = [backing newTextureWithDescriptor:d offset:0 bytesPerRow:pitch];
                        dest = [backing newTextureWithDescriptor:d offset:alias == 5 ? pitch : 0 bytesPerRow:pitch];
                        Check(source && dest && source != dest && !source.parentTexture && !dest.parentTexture &&
                              source.buffer == dest.buffer, "Fixture did not create distinct same-buffer roots");
                        auto* data = static_cast<std::byte*>(backing.contents);
                        for (unsigned y = 0; y <= height; ++y)
                            std::memcpy(data + y * pitch, fixture.bytes.data() + (y % height) * width * bpp, width * bpp);
                        if (alias == 5) {
                            for (unsigned y = 0; y < height; ++y) {
                                std::memcpy(initial.bytes.data() + y * width * bpp,
                                            fixture.bytes.data() + ((y + 1) % height) * width * bpp, width * bpp);
                                std::copy_n(fixture.colors.begin() + ((y + 1) % height) * width, width, initial.colors.begin() + y * width);
                            }
                        }
                    } else Upload(device, transfers, source, fixture);
                    auto independent = Texture(device, format, width, height, false);
                    auto reference = Texture(device, format, width, height, true);
                    Upload(device, transfers, independent, fixture); Upload(device, transfers, reference, initial);
                    const auto sr = MTLRegionMake2D(1, 1, width - 2, height - 2);
                    const auto dr = MTLRegionMake2D(2, 1, width - 4, height - 2);
                    const auto clip = MTLRegionMake2D(3, 2, width - 7, height - 4);
                    const bool fx = flip & 1, fy = flip & 2;
                    Check(filters.Filter(F(f), {independent, 0, 0, sr}, {reference, 0, 0, dr}, 2, fx, fy, clip), "Reference filter declined native path");
                    Check(filters.Filter(F(f), {source, 0, 0, sr}, {dest, 0, 0, dr}, 2, fx, fy, clip), "Alias filter declined native path");
                    const auto actual = Read(transfers, dest), other = Read(transfers, reference);
                    if (actual != other) {
                        std::cerr << "alias=" << alias << " filter=" << f << " format=" << format.name << " flip=" << flip << '\n';
                        if (f == unsigned(F::Bicubic)) {
                            const auto expected = Reference(fixture, width, height, sr, dr, clip, fx, fy, initial.colors);
                            Compare(format, other, expected, width, "immutable reference");
                            Compare(format, actual, expected, width, "aliased input");
                        }
                        throw std::runtime_error("Alias filter differs from immutable independent GPU input");
                    }
                    ++counterpart;
                    if (f == unsigned(F::Bicubic)) {
                        Compare(format, actual, Reference(fixture, width, height, sr, dr, clip, fx, fy, initial.colors), width,
                                "alias=" + std::to_string(alias) + " format=" + format.name); ++scalar;
                    }
                    scheduler.Finish();
                }
    Check(counterpart == (probe ? 12U : 1440U) && scalar == (probe ? 12U : 288U), "Incomplete alias matrix");
    std::cout << "aliases GPU-counterparts=" << counterpart << " independent-bicubic=" << scalar << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 2 || argc == 3, "Specify shader export directory, optionally alias-probe");
            const std::filesystem::path output{argv[1]}; Check(output.is_absolute(), "Use absolute export directory");
            const std::string mode = argc == 3 ? argv[2] : "";
            const bool probe = mode == "alias-probe";
            Check(mode.empty() || probe || mode == "precision-probe", "Unknown probe mode");
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TransferRuntime transfers{device, scheduler};
            Metal::TextureFilterRuntime filters{device, scheduler};
            if (probe) { Aliases(device, scheduler, transfers, filters, true); return 0; }
            if (mode == "precision-probe") {
                const auto fixture = MakeFixture(Formats[3], 17, 13, 0, false);
                auto source = Texture(device, Formats[3], 17, 13, false); Upload(device, transfers, source, fixture);
                const auto sr = MTLRegionMake2D(2, 1, 13, 10), dr = MTLRegionMake2D(2, 3, 19, 13), clip = MTLRegionMake2D(4, 5, 15, 9);
                const auto expected = Reference(fixture, 23, 19, sr, dr, clip, false, false, std::vector<Color>(23 * 19, Guard));
                for (const auto index : Destinations) {
                    auto target = Texture(device, Formats[index], 23, 19, true);
                    transfers.ClearColor(target, MTLClearColorMake(Guard[0], Guard[1], Guard[2], Guard[3]));
                    filters.Filter(Settings::TextureFilter::Bicubic, {source, 0, 0, sr}, {target, 0, 0, dr}, 2, false, false, clip);
                    const auto bytes = Read(transfers, target); const unsigned pixel = 13 * 23 + 10;
                    double actual{};
                    if (index == 20) { float value{}; std::memcpy(&value, bytes.data() + pixel * 16 + 4, 4); actual = value; }
                    else actual = std::to_integer<unsigned>(bytes[pixel * 4 + 1]) / 255.;
                    std::cout.precision(12);
                    std::cout << Formats[index].name << " green ideal=" << expected[pixel][1] << " stored=" << actual
                              << " linear=" << (Formats[index].srgb ? Linear(actual) : actual) << std::endl;
                }
                scheduler.Finish(); return 0;
            }
            Check(Formats.size() == 39, "Incomplete readable source format list");
            std::filesystem::create_directories(output);
            for (unsigned pass = 0; pass < 7; ++pass) {
                std::ofstream file{output / ("filter_" + std::to_string(pass) + ".metal")};
                file << Metal::GenerateTextureFilterShader(Metal::TextureFilterPass(pass)).source; Check(file.good(), "Shader export failed");
            }
            { std::ofstream file{output / "filter_vertex.metal"}; file << Metal::TextureFilterVertexSource; Check(file.good(), "Vertex export failed"); }
            std::uint64_t nonuniform{}, solids{}, raw_roundtrips{};
            for (unsigned revision = 0; revision < 2; ++revision) for (unsigned repeat = 0; repeat < 2; ++repeat) {
                for (const auto& format : Formats) {
                    auto source = Texture(device, format, 17, 13, false);
                    const auto fixture = MakeFixture(format, 17, 13, revision, false); Upload(device, transfers, source, fixture);
                    Check(transfers.Readback({source, 0, 0, MTLRegionMake2D(0, 0, 17, 13)}, fixture.row_bytes) == fixture.bytes,
                          "Nonuniform format bytes changed during native Upload/Readback"); ++raw_roundtrips;
                    for (const auto index : Destinations) for (unsigned flip = 0; flip < 4; ++flip) for (bool crop : {false, true}) {
                        const auto& target_format = Formats[index]; auto target = Texture(device, target_format, 23, 19, true);
                        transfers.ClearColor(target, MTLClearColorMake(Guard[0], Guard[1], Guard[2], Guard[3]));
                        const auto sr = crop ? MTLRegionMake2D(2, 1, 13, 10) : MTLRegionMake2D(0, 0, 17, 13);
                        const auto dr = MTLRegionMake2D(2, 3, 19, 13);
                        const auto clip = crop ? MTLRegionMake2D(4, 5, 15, 9) : dr;
                        const bool fx = flip & 1, fy = flip & 2;
                        Check(filters.Filter(Settings::TextureFilter::Bicubic, {source, 0, 0, sr}, {target, 0, 0, dr},
                                             2, fx, fy, clip), "Bicubic returned a fallback");
                        Compare(target_format, Read(transfers, target), Reference(fixture, 23, 19, sr, dr, clip, fx, fy,
                            std::vector<Color>(23 * 19, Guard)), 23, std::string(format.name) + "->" + target_format.name +
                            " rev=" + std::to_string(revision) + " flip=" + std::to_string(flip) + " crop=" + std::to_string(crop)); ++nonuniform;
                    }
                    const auto solid = MakeFixture(format, 17, 13, revision, true); Upload(device, transfers, source, solid);
                    Check(transfers.Readback({source, 0, 0, MTLRegionMake2D(0, 0, 17, 13)}, solid.row_bytes) == solid.bytes,
                          "Solid format bytes changed during native Upload/Readback"); ++raw_roundtrips;
                    for (const auto index : Destinations) for (unsigned f = 1; f <= 5; ++f) {
                        const auto& target_format = Formats[index]; auto target = Texture(device, target_format, 23, 19, true);
                        Check(filters.Filter(Settings::TextureFilter(f), {source, 0, 0, MTLRegionMake2D(0, 0, 17, 13)},
                            {target, 0, 0, MTLRegionMake2D(0, 0, 23, 19)}, 2), "Solid filter returned a fallback");
                        Compare(target_format, Read(transfers, target), std::vector<Color>(23 * 19, solid.colors[0]), 23,
                                std::string("solid ") + format.name + "->" + target_format.name + " filter=" + std::to_string(f)); ++solids;
                    }
                    std::cout << "revision=" << revision << " repeat=" << repeat << " format=" << format.name
                              << " nonuniform=" << nonuniform << " solid=" << solids << std::endl;
                }
            }
            Check(nonuniform == 3744 && solids == 2340 && raw_roundtrips == 312, "Incomplete source/destination format matrix");
            Aliases(device, scheduler, transfers, filters, false); scheduler.Finish();
            Check(Cases == 6372 && Pixels == 3875220, "Incomplete scalar pixel matrix");
            std::cout << "passed formats=39 nonuniform=" << nonuniform << " solids=" << solids << " raw_roundtrips=" << raw_roundtrips << " scalar_cases=" << Cases
                      << " rgba_pixels=" << Pixels << " max_byte_error=" << MaxByteError << " max_float_error=" << MaxFloatError
                      << " max_srgb_byte_error=" << MaxSrgbByteError << " max_srgb_linear_error=" << MaxSrgbLinearError << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
