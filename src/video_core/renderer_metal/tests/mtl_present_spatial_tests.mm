// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_screen_layout.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using Pixel = std::array<unsigned, 4>;
using FloatPixel = std::array<double, 4>;
using Image = std::vector<Pixel>;
constexpr unsigned Width = 48, Height = 32;
constexpr Pixel Background{17, 34, 51, 131};
constexpr std::array<unsigned, 3> SourceWidths{12, 16, 10}, SourceHeights{10, 12, 14};
// Store crop endpoints in sampling-axis order, independent of ScreenImage's rotated naming.
constexpr std::array<std::array<double, 4>, 3> Crops{{
    {.125 + 1. / 4096, .875 + 1. / 4096, .0625 + 3. / 4096, .8125 + 3. / 4096},
    {.0625 + 5. / 4096, .6875 + 5. / 4096, .1875 + 7. / 4096, .9375 + 7. / 4096},
    {.1875 + 9. / 4096, .9375 + 9. / 4096, .125 + 11. / 4096, .75 + 11. / 4096}}};

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

unsigned Quantize(double value) {
    return static_cast<unsigned>(std::lround(std::clamp(value, 0., 255.)));
}

Pixel SourcePixel(unsigned source, unsigned x, unsigned y, unsigned revision) {
    return {20 + (x * 13 + y * 7 + source * 43 + revision * 31) % 211,
            15 + (x * 5 + y * 17 + source * 61 + revision * 47) % 221,
            25 + (x * 19 + y * 3 + source * 29 + revision * 53) % 201,
            45 + (x * 11 + y * 13 + source * 37 + revision * 23) % 181};
}

id<MTLTexture> Texture(Metal::Device& device, unsigned width, unsigned height,
                       MTLPixelFormat format, bool sampled) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
        width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = sampled ? MTLTextureUsageShaderRead : MTLTextureUsageRenderTarget;
    return device.CreateTexture(descriptor);
}

void Upload(Metal::Device& device, Metal::TransferRuntime& transfers,
            id<MTLTexture> texture, unsigned source, unsigned revision) {
    auto buffer = device.CreateBuffer(texture.width * texture.height * 4, true);
    auto* data = static_cast<unsigned char*>(buffer.contents);
    for (unsigned y = 0; y < texture.height; ++y) for (unsigned x = 0; x < texture.width; ++x) {
        const auto color = SourcePixel(source, x, y, revision);
        for (unsigned c = 0; c < 4; ++c) data[(y * texture.width + x) * 4 + c] = color[c];
    }
    transfers.Upload(buffer, {0, texture.width * 4, 0},
        {texture, 0, 0, MTLRegionMake2D(0, 0, texture.width, texture.height)});
}

// A screen's physical orientation maps display fractions to memory fractions.
// The oracle never reads generated vertices, modelview matrices or stereo transforms.
std::array<double, 2> MemoryFraction(unsigned orientation, double x, double y) {
    switch (orientation) {
    case 0: return {1 - y, x};
    case 1: return {1 - x, 1 - y};
    case 2: return {y, 1 - x};
    case 3: return {x, y};
    default: throw std::runtime_error("Invalid oracle orientation");
    }
}

FloatPixel Sample(unsigned source, unsigned revision, std::array<double, 2> fraction,
                  bool linear) {
    const auto crop = Crops[source];
    const double u = crop[0] + fraction[0] * (crop[1] - crop[0]);
    const double v = crop[2] + fraction[1] * (crop[3] - crop[2]);
    const int width = SourceWidths[source], height = SourceHeights[source];
    const auto fetch = [&](int x, int y) {
        return SourcePixel(source, std::clamp(x, 0, width - 1),
                           std::clamp(y, 0, height - 1), revision);
    };
    FloatPixel result{};
    if (!linear) {
        const auto color = fetch(static_cast<int>(std::floor(u * width)),
                                 static_cast<int>(std::floor(v * height)));
        std::copy(color.begin(), color.end(), result.begin());
        return result;
    }
    const double sx = u * width - .5, sy = v * height - .5;
    const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
    const double fx = sx - x0, fy = sy - y0;
    for (unsigned j = 0; j < 2; ++j) for (unsigned i = 0; i < 2; ++i) {
        const auto color = fetch(x0 + static_cast<int>(i), y0 + static_cast<int>(j));
        const double weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
        for (unsigned c = 0; c < 4; ++c) result[c] += color[c] * weight;
    }
    return result;
}

struct Layer {
    double x, y, width, height;
    unsigned orientation, left, right, mode;
    bool reverse, linear;
    double opacity;
};

FloatPixel Shade(const Layer& layer, unsigned revision, unsigned x, unsigned y) {
    const auto fraction = MemoryFraction(layer.orientation,
        (x + .5 - layer.x) / layer.width, (y + .5 - layer.y) / layer.height);
    if (layer.mode == 2) {
        const unsigned eye = (y % 2 == unsigned(layer.reverse)) ? layer.left : layer.right;
        return Sample(eye, revision, fraction, layer.linear);
    }
    auto color = Sample(layer.left, revision, fraction, layer.linear);
    if (layer.mode == 1) {
        const auto right = Sample(layer.right, revision, fraction, layer.linear);
        constexpr double Left[3][3]{{.437, .449, .164}, {-.062, -.062, -.024}, {-.048, -.050, -.017}};
        constexpr double Right[3][3]{{-.011, -.032, -.007}, {.377, .761, .009}, {-.026, -.093, 1.234}};
        const auto left = color;
        for (unsigned c = 0; c < 3; ++c) {
            color[c] = 0;
            for (unsigned k = 0; k < 3; ++k) color[c] += Left[c][k] * left[k] + Right[c][k] * right[k];
        }
    }
    return color;
}

Image Reference(const std::vector<Layer>& layers, unsigned revision) {
    Image output(Width * Height, Background);
    for (const auto& layer : layers) {
        for (unsigned y = 0; y < Height; ++y) for (unsigned x = 0; x < Width; ++x) {
            if (x + .5 < layer.x || x + .5 >= layer.x + layer.width ||
                y + .5 < layer.y || y + .5 >= layer.y + layer.height) continue;
            const auto source = Shade(layer, revision, x, y);
            auto& dest = output[y * Width + x];
            // UNORM blending clamps source channels before combining with the destination.
            // Quantize the attachment after each ordered layer, including alpha.
            for (unsigned c = 0; c < 4; ++c)
                dest[c] = Quantize(std::clamp(source[c], 0., 255.) * layer.opacity + dest[c] * (1 - layer.opacity));
        }
    }
    return output;
}

Layout::FramebufferLayout BaseLayout() {
    Layout::FramebufferLayout layout{};
    layout.width = Width; layout.height = Height;
    layout.top_screen = {2, 3, 22, 23};
    layout.bottom_screen = {8, 11, 28, 29};
    layout.additional_screen = {14, 7, 34, 25};
    layout.cardboard.top_screen_right_eye = 1;
    layout.cardboard.bottom_screen_right_eye = 4;
    layout.render_3d_mode = Settings::StereoRenderOption::Off;
    return layout;
}

void Profile(Layout::FramebufferLayout& layout, unsigned profile) {
    // Eight explicit layouts cover each enable state and extra-screen source.
    // Opacities and extra/enable choices are intentionally coupled, not a Cartesian claim.
    layout.top_screen_enabled = profile <= 3;
    layout.bottom_screen_enabled = profile <= 2 || profile == 4;
    layout.additional_screen_enabled = profile != 0 && profile != 7;
    layout.additional_screen_is_bottom = profile == 2 || profile == 3 || profile == 6;
    constexpr std::array<float, 8> top{1, .75f, .25f, .5f, .75f, .25f, .75f, 1};
    constexpr std::array<float, 8> bottom{1, .25f, .75f, .75f, .5f, .75f, .25f, 1};
    layout.top_opacity = top[profile]; layout.bottom_opacity = bottom[profile];
}

// Separate scalar composition plan from public inputs. No BuildScreenDraws output is
// consumed by the oracle. Display mode numbers are the established public enum values.
std::vector<Layer> LayoutLayers(const Layout::FramebufferLayout& layout,
                                const Metal::ScreenLayoutOptions& options) {
    std::vector<Layer> layers;
    const unsigned mode = static_cast<unsigned>(layout.render_3d_mode);
    const auto add = [&](bool bottom, const Common::Rectangle<u32>& r) {
        Layer layer{double(r.left), double(r.top), double(r.right - r.left), double(r.bottom - r.top),
                    layout.is_rotated ? 0U : 1U, bottom ? 2U : unsigned(options.swap_eyes),
                    bottom ? 2U : unsigned(!options.swap_eyes), 0, false, options.linear_filter,
                    bottom ? layout.bottom_opacity : layout.top_opacity};
        switch (mode) {
        case 0:
            layer.left = layer.right = bottom ? 2 : options.mono_eye;
            layers.push_back(layer); break;
        case 1: case 2: case 6: {
            if (mode == 1) { layer.x *= .5; layer.width *= .5; }
            layer.right = layer.left;
            layers.push_back(layer);
            layer.left = layer.right = bottom ? 2 : unsigned(!options.swap_eyes);
            layer.x = mode == 6 ? Width / 2 + (bottom ? layout.cardboard.bottom_screen_right_eye :
                                             layout.cardboard.top_screen_right_eye) : layer.x + Width / 2;
            layers.push_back(layer); break;
        }
        case 3: layer.mode = 1; layers.push_back(layer); break;
        case 4: case 5:
            layer.mode = 2; layer.reverse = mode == 5; layers.push_back(layer); break;
        default: throw std::runtime_error("Invalid oracle stereo mode");
        }
    };
    const auto top = [&] { if (layout.top_screen_enabled) add(false, layout.top_screen); };
    const auto bottom = [&] { if (layout.bottom_screen_enabled) add(true, layout.bottom_screen); };
    if (options.swap_screen_order) { bottom(); top(); } else { top(); bottom(); }
    if (layout.additional_screen_enabled) add(layout.additional_screen_is_bottom, layout.additional_screen);
    return layers;
}

std::uint64_t Difference(const Image& a, const Image& b) {
    Check(a.size() == b.size(), "Invalid oracle image comparison");
    std::uint64_t count{};
    for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
    return count;
}

void CheckOracleSensitivity() {
    auto layout = BaseLayout(); Profile(layout, 1);
    layout.render_3d_mode = Settings::StereoRenderOption::Interlaced;
    Metal::ScreenLayoutOptions options{};
    const auto base = Reference(LayoutLayers(layout, options), 0);
    std::array<std::uint64_t, 7> changed{};
    changed[0] = Difference(base, Reference(LayoutLayers(layout, options), 1));
    options.linear_filter = true; changed[1] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    options = {}; options.swap_eyes = true; changed[2] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    options = {}; options.swap_screen_order = true; changed[3] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    options = {}; layout.is_rotated = false; changed[4] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    layout.is_rotated = true; layout.top_opacity = 1;
    changed[5] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    layout.top_opacity = .75f; layout.render_3d_mode = Settings::StereoRenderOption::ReverseInterlaced;
    changed[6] = Difference(base, Reference(LayoutLayers(layout, options), 0));
    for (const auto count : changed) Check(count > 0, "Presentation oracle lacks axis sensitivity");
    std::cout << "sensitivity revision/filter/eyes/order/rotation/opacity/parity";
    for (const auto count : changed) std::cout << ' ' << count;
    std::cout << '\n';
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            CheckOracleSensitivity();
            Metal::Device device{};
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::Compositor compositor{device, scheduler};
            std::array<id<MTLTexture>, 3> textures{};
            std::array<Metal::ScreenImage, 3> screens{};
            for (unsigned i = 0; i < 3; ++i) {
                textures[i] = Texture(device, SourceWidths[i], SourceHeights[i], MTLPixelFormatRGBA8Unorm, true);
                screens[i] = {textures[i], {float(Crops[i][2]), float(Crops[i][0]),
                                          float(Crops[i][3]), float(Crops[i][1])}};
            }
            std::array<id<MTLTexture>, 2> targets{
                Texture(device, Width, Height, MTLPixelFormatRGBA8Unorm, false),
                Texture(device, Width, Height, MTLPixelFormatBGRA8Unorm, false)};
            std::uint64_t renders{}, comparisons{}, direct{}, layouts{};
            unsigned max_error{};
            const auto verify = [&](unsigned format, const std::vector<Metal::PresentDraw>& draws,
                                    const std::vector<Layer>& layers, unsigned revision, const char* phase) {
                Check(draws.size() == layers.size(), "Presentation layout draw count mismatch");
                compositor.Render(targets[format], textures, draws,
                    MTLClearColorMake(17. / 255, 34. / 255, 51. / 255, 131. / 255));
                const auto bytes = transfers.Readback({targets[format], 0, 0,
                    MTLRegionMake2D(0, 0, Width, Height)}, Width * 4);
                Check(bytes.size() == Width * Height * 4, "Incomplete presentation readback");
                const auto expected = Reference(layers, revision);
                for (unsigned y = 0; y < Height; ++y) for (unsigned x = 0; x < Width; ++x) {
                    for (unsigned c = 0; c < 4; ++c) {
                        const unsigned channel = format && c != 1 && c != 3 ? 2 - c : c;
                        const auto actual = std::to_integer<unsigned>(bytes[(y * Width + x) * 4 + channel]);
                        const auto want = expected[y * Width + x][c];
                        const unsigned error = std::abs(int(actual) - int(want));
                        max_error = std::max(max_error, error);
                        if (error > 2) {
                            std::cerr << phase << " render=" << renders << " revision=" << revision
                                      << " format=" << format << " pixel=" << x << ',' << y
                                      << " channel=" << c << " expected=" << want << " actual=" << actual << '\n';
                            throw std::runtime_error("Nonuniform presentation scalar oracle mismatch");
                        }
                    }
                    ++comparisons;
                }
                ++renders;
            };
            constexpr std::array<std::array<float, 4>, 3> rectangles{{
                {3, 2, 41, 29}, {-3.25f, 4.25f, 31.75f, 35.25f}, {9.25f, -2.25f, 50.75f, 22.75f}}};
            for (unsigned revision = 0; revision < 2; ++revision) {
                for (unsigned i = 0; i < 3; ++i) Upload(device, transfers, textures[i], i, revision);
                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                    for (unsigned format = 0; format < 2; ++format)
                        for (unsigned orientation = 0; orientation < 4; ++orientation)
                            for (unsigned mode = 0; mode < 4; ++mode)
                                for (bool linear : {false, true}) for (bool swap : {false, true})
                                    for (bool bottom : {false, true})
                                    for (const auto& rect : rectangles) for (float opacity : {1.f, .75f, .25f}) {
                                        const unsigned shader_mode = mode < 2 ? mode : 2;
                                        const Layer layer{rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1],
                                            orientation, bottom ? 2U : unsigned(swap), bottom ? 2U : unsigned(!swap),
                                            shader_mode, mode == 3, linear, opacity};
                                        const auto draw = Metal::MakeScreenDraw(BaseLayout(), screens,
                                            {rect[0], rect[1], rect[2], rect[3]}, Layout::DisplayOrientation(orientation),
                                            layer.left, layer.right, Metal::PresentMode(shader_mode), layer.reverse, opacity, linear);
                                        verify(format, {draw}, {layer}, revision, "direct"); ++direct;
                                    }
                    for (unsigned format = 0; format < 2; ++format)
                        for (unsigned variant = 0; variant < 8; ++variant)
                            for (bool rotated : {false, true}) for (bool eyes : {false, true})
                                for (bool order : {false, true}) for (bool linear : {false, true})
                                    for (unsigned profile = 0; profile < 8; ++profile) {
                                        auto layout = BaseLayout(); Profile(layout, profile);
                                        layout.is_rotated = rotated;
                                        layout.render_3d_mode = Settings::StereoRenderOption(variant < 2 ? 0 : variant - 1);
                                        const Metal::ScreenLayoutOptions options{eyes, order, linear, variant == 1 ? 1U : 0U};
                                        verify(format, Metal::BuildScreenDraws(layout, screens, options),
                                               LayoutLayers(layout, options), revision, "layout"); ++layouts;
                                    }
                    std::cout << "revision=" << revision << " repeat=" << repeat
                              << " renders=" << renders << " pixel_comparisons=" << comparisons << std::endl;
                }
            }
            scheduler.Finish();
            Check(direct == 9216 && layouts == 8192 && renders == 17408 && comparisons == 26738688,
                  "Incomplete nonuniform presentation matrix");
            std::cout << "passed direct=" << direct << " layouts=" << layouts << " renders=" << renders
                      << " rgba_pixel_comparisons=" << comparisons << " max_byte_error=" << max_error << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
