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
using Image = std::array<Color, 64>;
struct Fixture {
    const char* name;
    Image image;
    std::array<Color, 4> quadrants; // Top-left, top-right, bottom-left, bottom-right.
};
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
Color Mix(Color a, Color b, double weight) {
    for (unsigned c = 0; c < 4; ++c) a[c] = a[c] * (1 - weight) + b[c] * weight;
    return a;
}
Fixture Make(unsigned rule, const std::array<Color, 3>& palette) {
    const auto [e, b, d] = palette;
    Fixture f{"", {}, {e, e, e, e}};
    f.image.fill(e);
    const auto put = [&](int x, int y, Color value) { f.image[(y + 3) * 8 + x + 3] = value; };
    if (rule == 0) {
        // Only the top-left 1:1 rule fires: the distant upper pixel differs
        // from E; the lower and distant left pixels equal E. J = (B + E)/2.
        f.name = "upper-distance";
        put(-1, -1, b); put(0, -1, b); put(-1, 0, b); put(0, -2, b);
        f.quadrants[0] = Mix(e, b, .5);
    } else if (rule == 1) {
        // Independent 1:1 corners K=(B+E)/2 and L=(D+E)/2 precede
        // the sole 2:1 propagation J=(K+E)/2. L must not contribute to J.
        f.name = "quadrant-propagation";
        put(-1, -1, b); put(0, -1, b); put(1, 0, b); put(2, 0, b);
        put(-1, 0, d); put(0, 1, d);
        f.quadrants = {Mix(e, b, .25), Mix(e, b, .5), Mix(e, d, .5), e};
    } else {
        // The top-left 1:1 rule gives J=(B+E)/2. The lower-left 2:1
        // rule reads (+1,-2): equal B blocks it; a different value permits
        // L=(J+E)/2. (+1,+2) remains E in both fixtures.
        f.name = rule == 2 ? "upper-guard-blocks" : "upper-guard-permits";
        put(-1, 0, b); put(-1, 1, b); put(0, -1, b); put(0, -2, b);
        if (rule == 2) put(1, -2, b);
        f.quadrants[0] = Mix(e, b, .5);
        if (rule == 3) f.quadrants[2] = Mix(e, b, .25);
    }
    return f;
}
std::pair<int, int> Transform(int x, int y, unsigned symmetry) {
    if (symmetry & 4) x = -x;
    for (unsigned i = 0; i < (symmetry & 3); ++i) {
        const int previous = x; x = -y; y = previous;
    }
    return {x, y};
}
Fixture Transform(Fixture f, unsigned symmetry) {
    const auto original = f;
    // All non-background pixels lie within [-3,3]. The extra row/column
    // preserve a power-of-two texture, keeping exact center sampling dyadic.
    for (int y = -3; y <= 3; ++y) for (int x = -3; x <= 3; ++x) {
        const auto [u, v] = Transform(x, y, symmetry);
        f.image[(v + 3) * 8 + u + 3] = original.image[(y + 3) * 8 + x + 3];
    }
    for (unsigned i = 0; i < 4; ++i) {
        const auto [u, v] = Transform(i % 2 ? 1 : -1, i / 2 ? 1 : -1, symmetry);
        f.quadrants[(v > 0 ? 2 : 0) + (u > 0 ? 1 : 0)] = original.quadrants[i];
    }
    return f;
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
std::uint64_t Cases{}, GoldenPixels{}, GuardPixels{};
double MaxFloat{}; unsigned MaxByte{};
constexpr Color Guard{17. / 255, 34. / 255, 51. / 255, 68. / 255};
void Compare(const std::vector<std::byte>& bytes, MTLPixelFormat format, unsigned x, unsigned y,
             Color expected, const std::string& label) {
    const bool floating = format == MTLPixelFormatRGBA32Float;
    for (unsigned c = 0; c < 4; ++c) {
        double actual{}, want = expected[c], tolerance{};
        const unsigned index = (y * 20 + x) * (floating ? 16 : 4) + c * (floating ? 4 : 1);
        if (floating) {
            float value{}; std::memcpy(&value, bytes.data() + index, 4); actual = value;
            tolerance = .000002; MaxFloat = std::max(MaxFloat, std::abs(actual - want));
        } else {
            actual = std::to_integer<unsigned>(bytes[index]);
            if (format == MTLPixelFormatRGBA8Unorm_sRGB && c < 3) want = Srgb(want);
            want = std::lround(want * 255); tolerance = 1;
            MaxByte = std::max(MaxByte, unsigned(std::abs(actual - want)));
        }
        if (!std::isfinite(actual) || std::abs(actual - want) > tolerance) {
            std::cerr << label << " pixel=" << x << ',' << y << " component=" << c
                      << " actual=" << actual << " expected=" << want << '\n';
            throw std::runtime_error("MMPX independent neighborhood golden failed");
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 2 || argc == 3, "Specify an absolute shader export directory and optional rule 0..3");
            const std::string selected = argc == 3 ? argv[2] : "";
            Check(selected.empty() || selected == "0" || selected == "1" || selected == "2" || selected == "3", "Unknown rule");
            const unsigned first_rule = selected.empty() ? 0 : unsigned(selected[0] - '0');
            const unsigned last_rule = selected.empty() ? 4 : first_rule + 1;
            const std::filesystem::path output{argv[1]}; Check(output.is_absolute(), "Use an absolute path");
            std::filesystem::create_directories(output);
            std::ofstream shader{output / "mmpx.metal"};
            shader << Metal::GenerateTextureFilterShader(Metal::TextureFilterPass::Mmpx).source;
            shader.close(); Check(shader.good(), "MSL export failed");
            Metal::Device device; Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler}; Metal::TextureFilterRuntime filters{device, scheduler};
            for (const bool source_float : {false, true}) for (unsigned palette_index = 0; palette_index < 2; ++palette_index) {
                std::array<Color, 3> palette = palette_index == 0 ?
                    std::array<Color, 3>{{{1, 1, 1, 1}, {0, 0, 0, 1}, {1, 0, 0, 1}}} :
                    std::array<Color, 3>{{{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, .5}}};
                if (!source_float) for (auto& color : palette) for (auto& c : color) c = std::lround(c * 255) / 255.;
                for (unsigned rule = first_rule; rule < last_rule; ++rule) for (unsigned symmetry = 0; symmetry < 8; ++symmetry) {
                    const auto f = Transform(Make(rule, palette), symmetry);
                    const auto source_format = source_float ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA8Unorm;
                    auto source = Texture(device, source_format, 8, false);
                    const unsigned bpp = source_float ? 16 : 4;
                    auto buffer = device.CreateBuffer(64 * bpp, true);
                    auto* raw = static_cast<std::byte*>(buffer.contents);
                    for (unsigned i = 0; i < 64; ++i) for (unsigned c = 0; c < 4; ++c) {
                        if (source_float) {
                            const float value = f.image[i][c]; std::memcpy(raw + i * 16 + c * 4, &value, 4);
                        } else raw[i * 4 + c] = std::byte(std::lround(f.image[i][c] * 255));
                    }
                    transfers.Upload(buffer, {0, 8 * bpp, 0}, {source, 0, 0, MTLRegionMake2D(0, 0, 8, 8)});
                    for (const auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA32Float})
                        for (const bool clipped : {false, true}) {
                            auto target = Texture(device, format, 20, true);
                            transfers.ClearColor(target, MTLClearColorMake(Guard[0], Guard[1], Guard[2], Guard[3]));
                            const auto dr = MTLRegionMake2D(2, 2, 16, 16);
                            const auto clip = clipped ? MTLRegionMake2D(8, 8, 2, 2) : dr;
                            Check(filters.Filter(Settings::TextureFilter::MMPX, {source, 0, 0, MTLRegionMake2D(0, 0, 8, 8)},
                                {target, 0, 0, dr}, 2, false, false, clip), "MMPX declined the native path");
                            const auto bytes = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 20, 20)},
                                20 * (format == MTLPixelFormatRGBA32Float ? 16 : 4));
                            const auto label = std::string(f.name) + " symmetry=" + std::to_string(symmetry) +
                                " palette=" + std::to_string(palette_index) + " source_float=" + std::to_string(source_float) +
                                " format=" + std::to_string(format) + " clipped=" + std::to_string(clipped);
                            for (unsigned q = 0; q < 4; ++q) {
                                Compare(bytes, format, 8 + q % 2, 8 + q / 2, f.quadrants[q], label); ++GoldenPixels;
                            }
                            for (unsigned y = 0; y < 20; ++y) for (unsigned x = 0; x < 20; ++x) {
                                if (x >= clip.origin.x && y >= clip.origin.y && x < clip.origin.x + clip.size.width &&
                                    y < clip.origin.y + clip.size.height) continue;
                                Compare(bytes, format, x, y, Guard, label + " guard"); ++GuardPixels;
                            }
                            ++Cases;
                        }
                }
            }
            scheduler.Finish();
            const unsigned rule_count = last_rule - first_rule;
            Check(Cases == 192 * rule_count && GoldenPixels == 768 * rule_count && GuardPixels == 51840 * rule_count,
                  "Incomplete MMPX geometry matrix");
            std::cout << "passed cases=" << Cases << " golden_rgba_pixels=" << GoldenPixels << " guard_rgba_pixels=" << GuardPixels
                      << " max_float_error=" << MaxFloat << " max_byte_error=" << MaxByte << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
