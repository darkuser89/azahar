// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_filter.h"
#include "common/settings.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Color = std::array<double, 4>;
struct Pattern { bool diagonal; unsigned edge, orientation; };
constexpr Color Guard{17. / 255, 34. / 255, 51. / 255, 68. / 255};
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
Color Mix(Color a, Color b, double t) {
    for (unsigned c = 0; c < 4; ++c) a[c] += (b[c] - a[c]) * t;
    return a;
}
double AxisCoordinate(Pattern pattern, double x, double y) {
    const double s = pattern.orientation & 1 ? y : x;
    return pattern.orientation & 2 ? 16 - s : s;
}
double SourceAmount(Pattern pattern, double x, double y) {
    if (!pattern.diagonal) return std::floor(AxisCoordinate(pattern, x, y)) >= pattern.edge;
    if (pattern.orientation & 1) x = 16 - x;
    if (pattern.orientation & 2) y = 16 - y;
    return std::floor(x) + std::floor(y) >= 15;
}
double Spline(double distance) {
    distance = std::abs(distance);
    if (distance < 1) return (4 - 6 * distance * distance + 3 * distance * distance * distance) / 6;
    return distance < 2 ? std::pow(2 - distance, 3) / 6 : 0;
}
double SplineStep(double s, unsigned edge) {
    // Direct scalar convolution, independent of the four bilinear GPU taps.
    const int first = int(std::floor(s - .5)) - 1;
    double result{};
    for (int i = first; i < first + 4; ++i) if (i >= int(edge)) result += Spline(s - (i + .5));
    return result;
}
double ScaleForceStep(double s, unsigned edge, bool alpha) {
    // A 1D field has three identical neighbors on each side. The two
    // orthogonal neighbors equal the center and contribute zero distance.
    // For opaque gray, the YCbCr column sums reduce the distance to .6|a-b|.
    // For black alpha-only pixels it reduces to sqrt(|a-b|*a*b).
    const auto linear = [&](double p) { return std::clamp(p - edge + .5, 0., 1.); };
    const double center = linear(s);
    const auto distance = [&](double neighbor) {
        return alpha ? std::sqrt(std::abs(neighbor - center) * neighbor * center) :
                       .6 * std::abs(neighbor - center);
    };
    const double left = distance(linear(s - 1)), right = distance(linear(s + 1));
    if (left + right == 0) return center;
    const double limit = std::abs(right - left) / (right + left);
    const double shift = std::clamp(3 * (right - left), -limit, limit);
    return SplineStep(s - shift, edge);
}
double XbrzAmount(Pattern pattern, double x, double y, unsigned scale) {
    if (!pattern.diagonal) return SourceAmount(pattern, x, y);
    if (pattern.orientation & 1) x = 16 - x;
    if (pattern.orientation & 2) y = 16 - y;
    // Binary 45-degree half-plane: only one dominant corner can blend.
    // Both directional distances equal, excluding shallow/steep branches.
    // The geometric boundary is x+y=15.5, with normal (-1,-1)/sqrt(2).
    // The signed-distance smoothstep therefore reduces to this polynomial.
    const double t = std::clamp((scale * (x + y - 15.5) + 1) / 2, 0., 1.);
    return t * t * (3 - 2 * t);
}
id<MTLTexture> Texture(Metal::Device& device, MTLPixelFormat format, unsigned size, bool target) {
    auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:size height:size mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | (target ? MTLTextureUsageRenderTarget : 0);
    return device.CreateTexture(d);
}
double Srgb(double value) {
    return value <= .0031308 ? value * 12.92 : 1.055 * std::pow(value, 1 / 2.4) - .055;
}
struct Counts {
    std::uint64_t cases{}, pixels{}, golden{}, guards{};
    unsigned max_byte{}, max_srgb{};
    double max_float{}, max_guard_float{};
};
void Compare(Counts& counts, const std::vector<std::byte>& bytes, MTLPixelFormat format,
             unsigned width, unsigned x, unsigned y, Color expected, bool guard, const std::string& label) {
    const bool floating = format == MTLPixelFormatRGBA32Float;
    for (unsigned c = 0; c < 4; ++c) {
        const unsigned index = (y * width + x) * (floating ? 16 : 4) + c * (floating ? 4 : 1);
        double actual{}, want = expected[c], tolerance{}; bool matches{};
        if (floating) {
            float value{}; std::memcpy(&value, bytes.data() + index, 4); actual = value;
            tolerance = guard ? .000002 : .006;
            auto& maximum = guard ? counts.max_guard_float : counts.max_float;
            maximum = std::max(maximum, std::abs(actual - want));
            matches = std::abs(actual - want) <= tolerance;
        } else {
            actual = std::to_integer<unsigned>(bytes[index]);
            if (format == MTLPixelFormatRGBA8Unorm_sRGB && c < 3) {
                const double lower = std::lround(Srgb(std::max(0., want - (guard ? 0 : .006))) * 255) - 1;
                const double upper = std::lround(Srgb(std::min(1., want + (guard ? 0 : .006))) * 255) + 1;
                want = std::lround(Srgb(want) * 255); matches = actual >= lower && actual <= upper;
                counts.max_srgb = std::max(counts.max_srgb, unsigned(std::abs(actual - want)));
            } else {
                want = std::lround(want * 255); tolerance = guard ? 1 : 2;
                matches = std::abs(actual - want) <= tolerance;
                counts.max_byte = std::max(counts.max_byte, unsigned(std::abs(actual - want)));
            }
        }
        if (!std::isfinite(actual) || !matches) {
            std::cerr << label << " pixel=" << x << ',' << y << " component=" << c
                      << " actual=" << actual << " expected=" << want << " guard=" << guard << '\n';
            throw std::runtime_error("Independent edge filter golden failed");
        }
    }
    ++counts.pixels; ++(guard ? counts.guards : counts.golden);
}
Counts Run(bool xbrz, Metal::Device& device, Metal::Scheduler& scheduler, Metal::TransferRuntime& transfers,
           Metal::TextureFilterRuntime& filters) {
    std::vector<Pattern> patterns;
    for (unsigned edge : {1U, 8U, 15U}) for (unsigned o = 0; o < 4; ++o) patterns.push_back({false, edge, o});
    if (xbrz) for (unsigned o = 0; o < 4; ++o) patterns.push_back({true, 15, o});
    Counts counts;
    for (const bool source_float : {false, true}) for (const bool alpha : {false, true}) {
        const Color low = alpha ? Color{0, 0, 0, 0} : Color{0, 0, 0, 1};
        const Color high = alpha ? Color{0, 0, 0, 1} : Color{1, 1, 1, 1};
        for (const auto pattern : patterns) {
            auto source = Texture(device, source_float ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA8Unorm, 16, false);
            const unsigned bpp = source_float ? 16 : 4;
            auto buffer = device.CreateBuffer(256 * bpp, true);
            auto* raw = static_cast<std::byte*>(buffer.contents);
            for (unsigned y = 0; y < 16; ++y) for (unsigned x = 0; x < 16; ++x) {
                const auto color = Mix(low, high, SourceAmount(pattern, x + .5, y + .5));
                for (unsigned c = 0; c < 4; ++c) {
                    if (source_float) {
                        const float value = color[c]; std::memcpy(raw + (y * 16 + x) * 16 + c * 4, &value, 4);
                    } else raw[(y * 16 + x) * 4 + c] = std::byte(std::lround(color[c] * 255));
                }
            }
            transfers.Upload(buffer, {0, 16 * bpp, 0}, {source, 0, 0, MTLRegionMake2D(0, 0, 16, 16)});
            for (const auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA32Float})
                for (unsigned scale : {2U, 4U}) for (const bool clipped : {false, true}) {
                    // The diagonal crop keeps every 5x5 classification neighborhood
                    // inside the source. Straight edges include the texture borders.
                    const unsigned origin = pattern.diagonal ? 3 : 0, extent = pattern.diagonal ? 10 : 16;
                    const unsigned side = extent * scale, width = side + 4;
                    auto target = Texture(device, format, width, true);
                    transfers.ClearColor(target, MTLClearColorMake(Guard[0], Guard[1], Guard[2], Guard[3]));
                    const auto dr = MTLRegionMake2D(2, 2, side, side);
                    const auto clip = clipped ? MTLRegionMake2D(3, 4, side - 3, side - 5) : dr;
                    Check(filters.Filter(xbrz ? Settings::TextureFilter::xBRZ : Settings::TextureFilter::ScaleForce,
                        {source, 0, 0, MTLRegionMake2D(origin, origin, extent, extent)}, {target, 0, 0, dr},
                        scale, false, false, clip), "Edge filter declined the native path");
                    const auto bytes = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, width, width)},
                        width * (format == MTLPixelFormatRGBA32Float ? 16 : 4));
                    const auto label = std::string(xbrz ? "xBRZ" : "ScaleForce") + " diagonal=" + std::to_string(pattern.diagonal) +
                        " edge=" + std::to_string(pattern.edge) + " orientation=" + std::to_string(pattern.orientation) +
                        " alpha=" + std::to_string(alpha) + " source_float=" + std::to_string(source_float) +
                        " format=" + std::to_string(format) + " scale=" + std::to_string(scale) + " clipped=" + std::to_string(clipped);
                    for (unsigned y = 0; y < width; ++y) for (unsigned x = 0; x < width; ++x) {
                        const bool inside = x >= clip.origin.x && y >= clip.origin.y && x < clip.origin.x + clip.size.width &&
                                            y < clip.origin.y + clip.size.height;
                        Color expected = Guard;
                        if (inside) {
                            const double u = origin + (x + .5 - dr.origin.x) / scale;
                            const double v = origin + (y + .5 - dr.origin.y) / scale;
                            const double amount = xbrz ? XbrzAmount(pattern, u, v, scale) :
                                ScaleForceStep(AxisCoordinate(pattern, u, v), pattern.edge, alpha);
                            expected = Mix(low, high, amount);
                        }
                        Compare(counts, bytes, format, width, x, y, expected, !inside, label);
                    }
                    ++counts.cases;
                }
            scheduler.Finish();
        }
    }
    Check(counts.cases == (xbrz ? 768 : 576) && counts.pixels == (xbrz ? 1946112 : 1704960), "Incomplete edge matrix");
    Check(counts.golden + counts.guards == counts.pixels, "Incomplete golden/guard coverage");
    std::cout << (xbrz ? "xBRZ" : "ScaleForce") << " passed cases=" << counts.cases << " rgba_pixels=" << counts.pixels
              << " golden_rgba_pixels=" << counts.golden << " guard_rgba_pixels=" << counts.guards << " max_byte_error=" << counts.max_byte
              << " max_srgb_byte_error=" << counts.max_srgb << " max_float_error=" << counts.max_float
              << " max_guard_float_error=" << counts.max_guard_float << std::endl;
    return counts;
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 2 || argc == 3, "Specify shader export directory and optional scaleforce/xbrz selector");
            const std::filesystem::path output{argv[1]}; Check(output.is_absolute(), "Use an absolute path");
            const std::string mode = argc == 3 ? argv[2] : "";
            Check(mode.empty() || mode == "scaleforce" || mode == "xbrz", "Unknown edge selector");
            std::filesystem::create_directories(output);
            for (const auto pass : {Metal::TextureFilterPass::ScaleForce, Metal::TextureFilterPass::Xbrz}) {
                std::ofstream file{output / (pass == Metal::TextureFilterPass::ScaleForce ? "scaleforce.metal" : "xbrz.metal")};
                file << Metal::GenerateTextureFilterShader(pass).source; file.close(); Check(file.good(), "Shader export failed");
            }
            Metal::Device device; Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler}; Metal::TextureFilterRuntime filters{device, scheduler};
            if (mode != "xbrz") Run(false, device, scheduler, transfers, filters);
            if (mode != "scaleforce") Run(true, device, scheduler, transfers, filters);
            scheduler.Finish(); return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
