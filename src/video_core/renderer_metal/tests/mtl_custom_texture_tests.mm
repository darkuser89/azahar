// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_texture_runtime.h"
#include "mtl_draw_runtime.h"
#include "mtl_custom_texture_fixture.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace Metal::Test::Custom;
Metal::TextureRegion Full(id<MTLTexture> texture, u32 level);
std::vector<std::byte> Read(Metal::TransferRuntime& transfers, id<MTLTexture> texture, u32 level);
void ExpectBlocks(Metal::TransferRuntime& transfers, id<MTLTexture> texture, unsigned level,
                  Format format, unsigned seed) {
    const auto region = Full(texture, level);
    const auto width = unsigned(region.region.size.width), height = unsigned(region.region.size.height);
    const auto pixels = Read(transfers, texture, level);
    const auto block = Block(format);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const auto expected = BlockColor(format, x / block[0], y / block[1], seed);
        for (unsigned channel = 0; channel < 4; ++channel) {
            const auto actual = std::to_integer<u8>(pixels[(y * width + x) * 4 + channel]);
            if (std::abs(int(actual) - expected[channel]) > 1)
                throw std::runtime_error("Custom compressed pixel mismatch format=" +
                    std::to_string(unsigned(format)) + " level=" + std::to_string(level) +
                    " x=" + std::to_string(x) + " y=" + std::to_string(y) +
                    " channel=" + std::to_string(channel) + " expected=" +
                    std::to_string(expected[channel]) + " actual=" + std::to_string(actual));
        }
    }
}
Metal::TextureRegion Full(id<MTLTexture> texture, u32 level) {
    return {texture, level, 0, MTLRegionMake2D(0, 0, std::max<NSUInteger>(1, texture.width >> level),
                                             std::max<NSUInteger>(1, texture.height >> level))};
}
std::vector<std::byte> Read(Metal::TransferRuntime& transfers, id<MTLTexture> texture, u32 level) {
    const auto region = Full(texture, level);
    return transfers.Readback(region, region.region.size.width * 4);
}
void Expect(Metal::TransferRuntime& transfers, id<MTLTexture> texture, u32 level, Color expected) {
    const auto pixels = Read(transfers, texture, level);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        Check(std::abs(int(std::to_integer<u8>(pixels[i])) - expected[i % 4]) <= 1,
              "Native custom material color/normal mip pixel mismatch");
}

unsigned GuestWriteback(Metal::TextureRuntime& runtime, Metal::TransferRuntime& transfers,
                       Metal::Surface& surface, unsigned level, Format format, unsigned seed,
                       unsigned& physical_cases, unsigned& rejected_cases) {
    const auto width = std::max(1U, surface.GetWidth() >> level);
    const auto height = std::max(1U, surface.GetHeight() >> level);
    const auto source_width = std::max<NSUInteger>(1, surface.Image().width >> level);
    const auto source_height = std::max<NSUInteger>(1, surface.Image().height >> level);
    const auto block = Block(format);
    const auto color_before = Read(transfers, surface.Image(), level);
    const auto normal_before = Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level);
    const auto base_region = Full(surface.Image(Metal::SurfaceImage::Base), level);
    const auto base_pitch = ((source_width + block[0] - 1) / block[0]) * block[2];
    const auto base_before = transfers.Readback(base_region, base_pitch);
    const auto expected = [&](unsigned x, unsigned y) {
        // Point samples at global guest texel centers, independently of the
        // production rectangle mapping or GPU conversion implementation.
        const auto sx = ((2 * x + 1) * source_width) / (2 * width);
        const auto sy = ((2 * y + 1) * source_height) / (2 * height);
        return BlockColor(format, unsigned(sx) / block[0], unsigned(sy) / block[1], seed);
    };
    for (bool partial : {false, true}) {
        const unsigned left = partial && width > 2 ? 1 : 0;
        const unsigned bottom = partial && height > 1 ? 1 : 0;
        const unsigned top = partial && height > 2 ? height - 1 : height;
        const unsigned copy_width = width - left, copy_height = top - bottom;
        const auto size = copy_width * copy_height * 4;
        auto staging = runtime.FindStaging(size + 32, false);
        std::memset(staging.mapped.data(), 0xcd, staging.size);
        surface.Download({staging.offset + 16, size, {left, top, width, bottom}, level}, staging);
        for (unsigned i = 0; i < staging.size; ++i) {
            if (i < 16 || i >= 16 + size) {
                Check(staging.mapped[i] == 0xcd, "Custom writeback changed a staging guard byte");
                continue;
            }
            const auto pixel = (i - 16) / 4, channel = (i - 16) % 4;
            const auto color = expected(left + pixel % copy_width, bottom + pixel / copy_width);
            if (std::abs(int(staging.mapped[i]) - color[channel]) > 1)
                throw std::runtime_error("Custom guest writeback mismatch format=" +
                    std::to_string(unsigned(format)) + " mip=" + std::to_string(level) +
                    " partial=" + std::to_string(partial) + " byte=" + std::to_string(i));
        }
    }
    if (!level) {
        VideoCore::SurfaceParams params;
        params.addr = surface.addr; params.width = width; params.height = height;
        params.stride = width; params.levels = 1; params.res_scale = 1;
        params.pixel_format = VideoCore::PixelFormat::RGBA8;
        params.type = VideoCore::SurfaceType::Texture; params.is_tiled = true;
        params.UpdateParams();
        constexpr std::array<unsigned, 8> morton{0, 1, 4, 5, 16, 17, 20, 21};
        for (bool partial : {false, true}) {
            const NSRange range = partial ? NSMakeRange(13, 224) : NSMakeRange(0, 256);
            std::array<u8, 288> bytes; bytes.fill(0xcd);
            runtime.DownloadGuest(surface, params, std::span<u8>{bytes}.subspan(16, 256), range);
            std::array<u8, 288> reference; reference.fill(0xcd);
            for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x) {
                // Tiled guest rows run opposite to the canonical sampled image.
                const auto color = expected(x, 7 - y);
                for (unsigned c = 0; c < 4; ++c) {
                    const auto offset = (morton[x] + 2 * morton[y]) * 4 + c;
                    if (offset >= range.location && offset - range.location < range.length)
                        reference[16 + offset] = color[3 - c]; // Guest ABGR byte order.
                }
            }
            for (unsigned i = 0; i < bytes.size(); ++i) {
                const bool changed = i >= 16 + range.location && i - 16 - range.location < range.length;
                if (!(changed ? std::abs(int(bytes[i]) - reference[i]) <= 1 : bytes[i] == 0xcd))
                    throw std::runtime_error("Custom physical writeback mismatch format=" +
                        std::to_string(unsigned(format)) + " partial=" + std::to_string(partial) +
                        " byte=" + std::to_string(i) + " expected=" + std::to_string(reference[i]) +
                        " actual=" + std::to_string(bytes[i]));
            }
            ++physical_cases;
        }
        auto staging = runtime.FindStaging(544, false);
        std::memset(staging.mapped.data(), 0xcd, staging.size);
        for (const auto copy : std::array<VideoCore::BufferTextureCopy, 4>{{
                 {16, 256, {0, 8, 8, 0}, 4}, {16, 288, {0, 8, 9, 0}, 0},
                 {16, 288, {0, 9, 8, 0}, 0}, {536, 256, {0, 8, 8, 0}, 0}}}) {
            bool caught{};
            try { auto bounded = copy; bounded.buffer_offset += staging.offset; surface.Download(bounded, staging); } catch (const std::invalid_argument&) { caught = true; }
            Check(caught && std::all_of(staging.mapped.begin(), staging.mapped.end(),
                                       [](u8 value) { return value == 0xcd; }),
                  "Invalid custom writeback changed staging bytes or escaped preflight");
            ++rejected_cases;
        }
    }
    Check(Read(transfers, surface.Image(), level) == color_before &&
          Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level) == normal_before &&
          transfers.Readback(base_region, base_pitch) == base_before,
          "Custom writeback changed a sampled color, normal or encoded base mip");
    return 2;
}

unsigned PhysicalGuestFormats(Metal::TextureRuntime& runtime, Metal::TransferRuntime& transfers,
                              const VideoCore::Material& material, Format color_format) {
    using Guest = VideoCore::PixelFormat;
    constexpr std::array formats{Guest::RGBA8, Guest::RGB8, Guest::RGB5A1, Guest::RGB565, Guest::RGBA4};
    constexpr std::array<unsigned, 5> bytes_per_pixel{4, 3, 2, 2, 2};
    constexpr std::array<unsigned, 8> morton{0, 1, 4, 5, 16, 17, 20, 21};
    const auto block = Block(color_format);
    unsigned cases{};
    for (unsigned format = 0; format < formats.size(); ++format)
        for (bool tiled : {false, true}) for (bool padded : {false, true}) {
            const unsigned stride = padded ? (tiled ? 24 : 21) : 16;
            VideoCore::SurfaceParams params;
            params.addr = 0x1000; params.width = params.height = 16; params.stride = stride;
            params.levels = 1; params.res_scale = 1; params.pixel_format = formats[format];
            params.type = VideoCore::SurfaceType::Texture; params.is_tiled = tiled;
            params.UpdateParams();
            VideoCore::SurfaceBase guest{params, {}};
            Metal::Surface surface{runtime, guest, &material};
            surface.UploadCustom(&material, 0);
            const auto color_before = Read(transfers, surface.Image(), 0);
            const auto normal_before = Read(transfers, surface.Image(Metal::SurfaceImage::Normal), 0);
            const auto base_region = Full(surface.Image(Metal::SurfaceImage::Base), 0);
            const auto base_pitch = ((material.width + block[0] - 1) / block[0]) * block[2];
            const auto base_before = transfers.Readback(base_region, base_pitch);
            const unsigned bpp = bytes_per_pixel[format];
            const unsigned pixels = tiled ? stride * 8 + 16 * 8 : stride * 15 + 16;
            const unsigned size = pixels * bpp;
            for (bool partial : {false, true}) {
                const NSRange range = partial ? NSMakeRange(7, size - 18) : NSMakeRange(0, size);
                std::vector<u8> bytes(size + 32, 0xcd), reference(size + 32, 0xcd);
                runtime.DownloadGuest(surface, params, std::span<u8>{bytes}.subspan(16, size), range);
                for (unsigned y = 0; y < 16; ++y) for (unsigned x = 0; x < 16; ++x) {
                    const auto sy = tiled ? 15 - y : y;
                    const auto sx = (2 * x + 1) * material.width / 32;
                    const auto ty = (2 * sy + 1) * material.height / 32;
                    const auto color = BlockColor(color_format, sx / block[0], ty / block[1], 7);
                    unsigned word{};
                    switch (formats[format]) {
                    case Guest::RGBA8: word = (unsigned(color[0]) << 24) | (unsigned(color[1]) << 16) |
                                              (unsigned(color[2]) << 8) | color[3]; break;
                    case Guest::RGB8: word = (unsigned(color[0]) << 16) | (unsigned(color[1]) << 8) | color[2]; break;
                    case Guest::RGB5A1: word = (unsigned(color[0] >> 3) << 11) | (unsigned(color[1] >> 3) << 6) |
                                               (unsigned(color[2] >> 3) << 1) | (color[3] >> 7); break;
                    case Guest::RGB565: word = (unsigned(color[0] >> 3) << 11) | (unsigned(color[1] >> 2) << 5) |
                                               (color[2] >> 3); break;
                    case Guest::RGBA4: word = (unsigned(color[0] >> 4) << 12) | (unsigned(color[1] >> 4) << 8) |
                                              (unsigned(color[2] >> 4) << 4) | (color[3] >> 4); break;
                    default: throw std::runtime_error("Unknown physical guest color fixture format");
                    }
                    const auto pixel = tiled ? ((y / 8) * (stride / 8) + x / 8) * 64 +
                                               morton[x % 8] + 2 * morton[y % 8] : y * stride + x;
                    for (unsigned c = 0; c < bpp; ++c) {
                        const auto offset = pixel * bpp + c;
                        if (offset >= range.location && offset - range.location < range.length)
                            reference[16 + offset] = u8(word >> (8 * c));
                    }
                }
                for (unsigned i = 0; i < bytes.size(); ++i)
                    if (bytes[i] != reference[i])
                        throw std::runtime_error("Custom physical guest format mismatch guest=" +
                            std::to_string(format) + " custom=" + std::to_string(unsigned(color_format)) +
                            " tiled=" + std::to_string(tiled) + " padded=" + std::to_string(padded) +
                            " partial=" + std::to_string(partial) + " byte=" + std::to_string(i) +
                            " expected=" + std::to_string(reference[i]) + " actual=" + std::to_string(bytes[i]));
                ++cases;
            }
            Check(Read(transfers, surface.Image(), 0) == color_before &&
                  Read(transfers, surface.Image(Metal::SurfaceImage::Normal), 0) == normal_before &&
                  transfers.Readback(base_region, base_pitch) == base_before,
                  "Physical guest format writeback changed a custom source image");
        }
    return cases;
}

struct DiffuseCase {
    std::array<std::array<float, 4>, 3> quaternions{{{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}}};
    std::array<float, 3> light{0, 0, -1}, diffuse{1, 1, 1}, ambient{}, global_ambient{};
    unsigned tolerance{1};
};

std::vector<DiffuseCase> RotationCases() {
    constexpr std::array<std::array<float, 4>, 10> rotations{{
        {0, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1},
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {1, 1, 1, 1},
        {0, -2, 0, -2}, {.25f, .5f, -.75f, 1}}};
    constexpr std::array<std::array<float, 3>, 4> lights{{
        {0, 0, -3}, {2, 0, 0}, {0, -4, 0}, {.75f, 1, -1.25f}}};
    std::vector<DiffuseCase> cases;
    for (unsigned rotation = 0; rotation <= rotations.size(); ++rotation) for (const auto light : lights) {
        DiffuseCase test;
        if (rotation < rotations.size()) test.quaternions.fill(rotations[rotation]);
        else test.quaternions = {{{0, 0, 0, 2}, {0, 2, 0, 2}, {2, 0, 0, 2}}};
        test.light = light; test.diffuse = {.75f, 1.25f, .375f};
        test.ambient = {.0625f, .125f, .1875f}; test.global_ambient = {.03125f, .0625f, .125f};
        test.tolerance = 2; cases.push_back(test);
    }
    return cases;
}

std::array<double, 3> DiffuseOracle(Color normal, const DiffuseCase& test, unsigned x, unsigned y) {
    // A 3x3 rotation matrix is independent of the shader's nested cross products.
    // Screen-center barycentrics for (-1,-1),(3,-1),(-1,3); normalize after interpolation.
    const double a = (x + .5) / 64, b = (y + .5) / 64;
    std::array<double, 4> q{};
    double norm{};
    for (unsigned c = 0; c < 4; ++c) {
        q[c] = (1 - a - b) * test.quaternions[0][c] + a * test.quaternions[1][c] + b * test.quaternions[2][c];
        norm += q[c] * q[c];
    }
    Check(norm > 0, "Diffuse quaternion fixture has zero length");
    for (auto& value : q) value /= std::sqrt(norm);
    const auto [qx, qy, qz, qw] = q;
    const std::array<std::array<double, 3>, 3> matrix{{
        {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
        {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
        {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}}};
    double light_length{}, dot{};
    for (const auto value : test.light) light_length += double(value) * value;
    Check(light_length > 0, "Diffuse light fixture has zero length");
    for (unsigned i = 0; i < 3; ++i) {
        double rotated{};
        for (unsigned j = 0; j < 3; ++j) rotated += matrix[i][j] * (2 * double(normal[j]) / 255 - 1);
        dot += rotated * test.light[i] / std::sqrt(light_length);
    }
    std::array<double, 3> result{};
    for (unsigned c = 0; c < 3; ++c)
        result[c] = std::clamp(test.global_ambient[c] + test.ambient[c] + test.diffuse[c] * std::max(dot, 0.0), 0.0, 1.0);
    return result;
}

class MaterialDraw {
public:
    MaterialDraw(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TextureRuntime& runtime)
        : geometry{device, scheduler}, draws{device, scheduler, geometry}, shaders{device},
          storage{device, scheduler, 4096}, sampler{runtime, {}} {
        using Tev = Pica::TexturingRegs::TevStageConfig;
        Pica::RegsInternal regs{};
        regs.lighting.disable.Assign(1);
        regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
        auto config = Pica::Shader::FSConfig{regs};
        config.lighting.enable.Assign(1); config.lighting.src_num.Assign(1);
        config.lighting.lights[0].directional.Assign(1);
        config.texture.texture0_type.Assign(Pica::TexturingRegs::TextureConfig::Texture2D);
        for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
        auto stage = static_cast<Tev>(config.texture.tev_stages[0]);
        stage.color_source1.Assign(Tev::Source::Texture0);
        stage.color_source2.Assign(Tev::Source::PrimaryFragmentColor);
        stage.color_op.Assign(Tev::Operation::Modulate);
        stage.alpha_source1.Assign(Tev::Source::Texture0);
        config.texture.tev_stages[0].sources_raw = stage.sources_raw;
        config.texture.tev_stages[0].ops_raw = stage.ops_raw;
        Pica::Shader::UserConfig user{}; user.use_custom_normal.Assign(1);
        fragment = shaders.Fragment(config, user, Metal::FragmentColorFormat::RGBA8);
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                          width:32 height:32 mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        target = device.CreateTexture(descriptor);
    }
    const std::string& Source() const { return fragment->program.shader.source; }
    void Verify(Metal::Surface& surface, unsigned level, Format color_format, Format normal_format,
                Metal::TransferRuntime& transfers, const DiffuseCase& lighting = {}) {
        auto vertices = storage.Allocate(3 * sizeof(Metal::CapturedVertex));
        std::memset(vertices.data, 0, vertices.size);
        auto* data = static_cast<Metal::CapturedVertex*>(vertices.data);
        constexpr std::array<std::array<float, 2>, 3> positions{{{-1, -1}, {3, -1}, {-1, 3}}};
        for (unsigned i = 0; i < 3; ++i) {
            data[i].position = {positions[i][0], positions[i][1], .5f, 1};
            data[i].texcoord0 = {(positions[i][0] + 1) / 2, (positions[i][1] + 1) / 2, 0, 0};
            data[i].quaternion = lighting.quaternions[i]; data[i].view = {0, 0, -1, 0};
            data[i].clip = {.5f, 0, 1, 0};
        }
        auto uniforms = storage.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
        std::memset(uniforms.data, 0, uniforms.size);
        auto* fs = static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
        fs->depth_offset = .5f;
        for (unsigned c = 0; c < 3; ++c) {
            fs->light_src[0].position[c] = lighting.light[c]; fs->light_src[0].diffuse[c] = lighting.diffuse[c];
            fs->light_src[0].ambient[c] = lighting.ambient[c]; fs->lighting_global_ambient[c] = lighting.global_ambient[c];
        }
        auto* pass = [[MTL4RenderPassDescriptor alloc] init];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 0);
        const auto color = surface.ImageView(Metal::SurfaceView::Mip0, Metal::SurfaceImage::Current, level);
        const auto normal = surface.ImageView(Metal::SurfaceView::Mip0, Metal::SurfaceImage::Normal, level);
        const std::array<Metal::TextureBinding, 2> images{{{color, 3}, {normal, 13}}};
        const std::array<Metal::SamplerBinding, 2> samplers{{{sampler.Handle(), 0}, {sampler.Handle(), 9}}};
        const std::array<Metal::BufferBinding, 1> buffers{{{uniforms.buffer, uniforms.offset, 2}}};
        Metal::PicaDrawState state{}; state.blend.color_write_mask = 15;
        draws.Draw(state, *fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                   {vertices, 3}, buffers, images, samplers);
        const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
        const auto color_block = Block(color_format), normal_block = Block(normal_format);
        for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 32; ++x) {
            const unsigned u = ((2 * x + 1) * unsigned(color.width)) / 64;
            const unsigned v = ((2 * y + 1) * unsigned(color.height)) / 64;
            const auto c = BlockColor(color_format, u / color_block[0], v / color_block[1], 7 + level);
            const auto n = BlockColor(normal_format, u / normal_block[0], v / normal_block[1], 11 + level);
            const auto diffuse = DiffuseOracle(n, lighting, x, y);
            for (unsigned channel = 0; channel < 4; ++channel) {
                const unsigned expected = channel == 3 ? c[3] : unsigned(std::lround(c[channel] * diffuse[channel]));
                const auto actual = std::to_integer<u8>(pixels[(y * 32 + x) * 4 + channel]);
                if (std::abs(int(actual) - int(expected)) > int(channel == 3 ? 1 : lighting.tolerance))
                    throw std::runtime_error("Custom material lighting/TEV mismatch color=" +
                        std::to_string(unsigned(color_format)) + " normal=" + std::to_string(unsigned(normal_format)) +
                        " level=" + std::to_string(level) + " x=" + std::to_string(x) + " y=" +
                        std::to_string(y) + " channel=" + std::to_string(channel) + " expected=" +
                        std::to_string(expected) + " actual=" + std::to_string(actual));
            }
        }
    }
private:
    Metal::GeometryRuntime geometry;
    Metal::DrawRuntime draws;
    Metal::ShaderRuntime shaders;
    Metal::StreamBuffer storage;
    Metal::Sampler sampler;
    std::shared_ptr<const Metal::CompiledFragmentShader> fragment;
    id<MTLTexture> target;
};
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Image image;
            Metal::Device device; Metal::Scheduler scheduler{device};
            Metal::TextureRuntime runtime{device, scheduler};
            Metal::TransferRuntime transfers{device, scheduler};
            Check(argc == 1 || argc == 2, "Optional custom material shader output path expected");
            MaterialDraw material_draw{device, scheduler, runtime};
            const auto rotation_cases = RotationCases();
            unsigned rotated_lighting_draws{};
            if (argc == 2) {
                Check(std::filesystem::path{argv[1]}.is_absolute(), "Shader fixture path must be absolute");
                std::ofstream file{argv[1]}; file << material_draw.Source();
                Check(file.good(), "Unable to write custom material lighting shader fixture");
            }
            unsigned material_cases{}, rejected{}, late_normal_cases{};
            for (u32 scale : {1U, 3U}) {
                VideoCore::SurfaceParams params;
                params.width = params.height = 8; params.levels = 4; params.res_scale = scale;
                params.pixel_format = VideoCore::PixelFormat::RGBA8;
                params.type = VideoCore::SurfaceType::Texture;
                VideoCore::SurfaceBase guest{params, {}};
                constexpr Color base_color{30, 70, 110, 190}, base_normal{128, 64, 255, 255};
                Maps base{image, 17, 11, base_color, base_normal};
                Metal::Surface surface{runtime, guest, &base.material};
                surface.UploadCustom(&base.material, 0);
                for (u32 level = 1; level < 4; ++level)
                    transfers.ClearColor(surface.Image(Metal::SurfaceImage::Normal),
                                         MTLClearColorMake(1, 0, 0, 1), level);
                runtime.GenerateMipmaps(surface);
                for (u32 level = 0; level < 4; ++level) {
                    Expect(transfers, surface.Image(), level, base_color);
                    Expect(transfers, surface.Image(Metal::SurfaceImage::Normal), level, base_normal);
                    ++material_cases;
                }
                for (u32 level = 1; level < 4; ++level) {
                    const Color color{u8(20 + level * 30), u8(60 + level * 10), 140, 230};
                    const Color normal{u8(80 + level * 20), 180, 220, 255};
                    Maps lower{image, 17U >> level, std::max(1U, 11U >> level), color, normal};
                    surface.UploadCustom(&lower.material, level);
                    Expect(transfers, surface.Image(), level, color);
                    Expect(transfers, surface.Image(Metal::SurfaceImage::Normal), level, normal);
                    const auto color_before = Read(transfers, surface.Image(), level);
                    const auto normal_before = Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level);
                    const auto reject = [&](auto mutation) {
                        mutation(); bool caught{};
                        try { surface.UploadCustom(&lower.material, level); }
                        catch (const std::invalid_argument&) { caught = true; }
                        Check(caught, "Malformed custom mip/material was accepted");
                        Check(Read(transfers, surface.Image(), level) == color_before &&
                              Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level) == normal_before,
                              "Rejected normal/material data partially overwrote a color or normal mip");
                        ++rejected;
                    };
                    lower.color.data[0] ^= 0xff;
                    reject([&] { lower.normal.data.pop_back(); });
                    lower.normal.data.push_back(normal[3]);
                    reject([&] { ++lower.material.width; });
                    --lower.material.width;
                    reject([&] { lower.material.textures[0] = nullptr; });
                    ++material_cases;
                }
                Maps color_only{image, 17, 11, base_color, base_normal, false};
                Metal::Surface late{runtime, guest, &color_only.material};
                late.UploadCustom(&color_only.material, 0);
                Check(!late.Image(Metal::SurfaceImage::Normal), "Color-only material exposed a normal image");
                constexpr Color added_normal{64, 128, 220, 255};
                Maps lower_normal{image, 8, 5, base_color, added_normal};
                late.UploadCustom(&lower_normal.material, 1);
                for (u32 level = 0; level < 4; ++level) {
                    Expect(transfers, late.Image(Metal::SurfaceImage::Normal), level,
                           level == 1 ? added_normal : Color{128, 128, 255, 255});
                    ++late_normal_cases;
                }
                Expect(transfers, late.Image(), 0, base_color);
            }
            unsigned compressed_cases{}, compressed_rejections{}, writeback_cases{}, physical_writebacks{}, writeback_rejections{}, physical_format_cases{};
            for (const auto color_format : Formats) for (const auto normal_format : Formats)
                for (const auto dimensions : {std::array{17U, 11U}, std::array{16U, 12U}}) {
                    VideoCore::SurfaceParams params;
                    params.width = params.height = 8; params.levels = 4;
                    params.pixel_format = VideoCore::PixelFormat::RGBA8;
                    params.type = VideoCore::SurfaceType::Texture;
                    params.addr = 0x1000; params.stride = 8; params.is_tiled = true;
                    params.UpdateParams();
                    VideoCore::SurfaceBase guest{params, {}};
                    Maps base{image, dimensions[0], dimensions[1], {}, {}};
                    base.material.format = color_format;
                    EncodeMap(base.color, color_format, 7);
                    EncodeMap(base.normal, normal_format, 11);
                    physical_format_cases += PhysicalGuestFormats(runtime, transfers, base.material, color_format);
                    Metal::Surface surface{runtime, guest, &base.material};
                    surface.UploadCustom(&base.material, 0);
                    for (unsigned level = 1; level < 4; ++level) {
                        Maps mip{image, std::max(1U, dimensions[0] >> level),
                                 std::max(1U, dimensions[1] >> level), {}, {}};
                        mip.material.format = color_format;
                        EncodeMap(mip.color, color_format, 7 + level);
                        EncodeMap(mip.normal, normal_format, 11 + level);
                        surface.UploadCustom(&mip.material, level);
                        if (level == 1) {
                            const auto before_color = Read(transfers, surface.Image(), level);
                            const auto before_normal = Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level);
                            mip.color.data[0] ^= 1;
                            mip.normal.data.pop_back();
                            bool caught{};
                            try { surface.UploadCustom(&mip.material, level); }
                            catch (const std::invalid_argument&) { caught = true; }
                            Check(caught, "Short mixed-format normal map was accepted");
                            Check(Read(transfers, surface.Image(), level) == before_color &&
                                  Read(transfers, surface.Image(Metal::SurfaceImage::Normal), level) == before_normal,
                                  "Rejected compressed normal map changed an existing color/normal mip");
                            ++compressed_rejections;
                        }
                    }
                    // Verify after all separate material objects have been destroyed and all
                    // mips uploaded, so lower-mip writes cannot silently overwrite another mip.
                    for (unsigned level = 0; level < 4; ++level) {
                        ExpectBlocks(transfers, surface.Image(), level, color_format, 7 + level);
                        ExpectBlocks(transfers, surface.Image(Metal::SurfaceImage::Normal), level, normal_format, 11 + level);
                        material_draw.Verify(surface, level, color_format, normal_format, transfers);
                        for (const auto& lighting : rotation_cases) {
                            material_draw.Verify(surface, level, color_format, normal_format, transfers, lighting);
                            ++rotated_lighting_draws;
                        }
                        writeback_cases += GuestWriteback(runtime, transfers, surface, level,
                                                         color_format, 7 + level,
                                                         physical_writebacks, writeback_rejections);
                        ++compressed_cases;
                    }
                }
            std::cout << material_cases << " actual custom Material color/normal mip cases: separate lower-mip objects,"
                         " odd replacement dimensions, scales1/3 and generated normal mips passed\n";
            std::cout << rejected << " malformed material/map preflight cases preserve both GPU images; "
                      << late_normal_cases << " late-normal mip cases use defined flat defaults passed\n";
            std::cout << compressed_cases << " mixed custom color/normal format mip cases: RGBA8/BC1/BC3/BC5/BC7/"
                         "ASTC4x4/6x6/8x6, all64 format pairs, odd/even dimensions, four mips, independent"
                         " per-block RGBA oracles including BC1 transparency passed; " << compressed_rejections
                      << " malformed mixed-map uploads preserve both GPU images passed\n";
            std::cout << compressed_cases << " actual Material -> native compressed upload -> color/normal mip views ->"
                         " real PICA directional diffuse lighting/TEV -> Metal4 render draws compare all1024 RGBA pixels passed\n";
            std::cout << rotated_lighting_draws << " custom normal/quaternion directional diffuse/TEV draws:"
                         " all64 color-normal format pairs, two replacement dimensions/four mips, ten rotations and"
                         " interpolated per-vertex quaternions, four normalized light directions, RGB ambient/clamping,"
                         " independent 3x3 matrix oracle for all1024 RGBA pixels passed\n";
            std::cout << writeback_cases << " custom guest writebacks: all8 color formats/all64 color-normal pairs,"
                         " odd/even replacement dimensions, four guest mips, full/partial global sampling phase,"
                         " staging guards and unchanged sampled/normal/encoded-base images passed\n";
            std::cout << physical_writebacks << " custom physical guest writebacks: GPU Morton/ABGR encoder,"
                         " full/partial byte ranges and all288 reference/guard bytes passed; "
                      << writeback_rejections << " invalid mip/extent/staging copies rejected without writes\n";
            std::cout << physical_format_cases << " custom physical format writebacks: RGBA8/RGB8/RGB5A1/RGB565/RGBA4,"
                         " linear/tiled, tight/padded strides16/21/24, full/unaligned partial ranges,"
                         " exact packed bytes/row padding/guards and unchanged color/normal/encoded-base images passed\n";
            std::cout << "No app/window/ROM/emulation or image decoding started\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
