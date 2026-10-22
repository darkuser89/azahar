// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_custom_texture_fixture.h"
#include "mtl_draw_runtime.h"
#include "mtl_texture_runtime.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
using namespace Metal::Test::Custom;
constexpr unsigned TableCount = 32 + 24 * 256;
unsigned Offset(unsigned table) { return 32 + ((table * 7) % 24) * 256; }
double Spot(unsigned light, double input) {
    return .125 + light / 32.0 + .625 * std::clamp(1 - 2 * std::abs(input - .125), 0.0, 1.0);
}
double Distance(unsigned light, double input) {
    return .1875 + light / 64.0 + .5 * std::clamp(1 - 4 * std::abs(input - .5), 0.0, 1.0);
}
struct Variant {
    unsigned light{}, mode{}, input_mode{};
    float scale{};
    bool directional{};
    std::shared_ptr<const Metal::CompiledFragmentShader> shader;
};

class LightingDraw {
public:
    LightingDraw(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TextureRuntime& runtime,
                 const char* export_prefix)
        : geometry{device, scheduler}, draws{device, scheduler, geometry}, shaders{device},
          stream{device, scheduler, 4096}, sampler{runtime, {}} {
        using C = Pica::TexturingRegs::TextureConfig;
        using T = Pica::TexturingRegs::TevStageConfig;
        using L = Pica::LightingRegs;
        constexpr std::array<float, 6> scales{.25f, .5f, 1, 2, 4, 8};
        for (unsigned light = 0; light < 8; ++light) for (unsigned mode = 0; mode < 4; ++mode)
            for (unsigned input_mode = 0; input_mode < 3; ++input_mode) {
                Variant v{light, mode, input_mode, scales[(light + mode + input_mode) % scales.size()], light % 2 == 0, {}};
                Pica::RegsInternal regs{}; regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                auto config = Pica::Shader::FSConfig{regs};
                config.lighting.enable.Assign(1); config.lighting.src_num.Assign(1);
                config.lighting.config.Assign(L::LightingConfig::Config0);
                auto& source = config.lighting.lights[0]; source.num.Assign(light);
                source.directional.Assign(v.directional); source.two_sided_diffuse.Assign(input_mode == 2);
                source.spot_atten_enable.Assign(mode & 1); source.dist_atten_enable.Assign(bool(mode & 2));
                config.lighting.lut_sp.enable.Assign(1);
                config.lighting.lut_sp.type.Assign(input_mode == 2 ? L::LightingLutInput::LN : L::LightingLutInput::SP);
                config.lighting.lut_sp.abs_input.Assign(input_mode != 0); config.lighting.lut_sp.SetScale(v.scale);
                config.texture.texture0_type.Assign(C::Texture2D);
                for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
                auto stage = static_cast<T>(config.texture.tev_stages[0]);
                stage.color_source1.Assign(T::Source::Texture0); stage.color_source2.Assign(T::Source::PrimaryFragmentColor);
                stage.color_op.Assign(T::Operation::Modulate); stage.alpha_source1.Assign(T::Source::Texture0);
                config.texture.tev_stages[0].sources_raw = stage.sources_raw;
                config.texture.tev_stages[0].ops_raw = stage.ops_raw;
                Pica::Shader::UserConfig user{}; user.use_custom_normal.Assign(1);
                v.shader = shaders.Fragment(config, user, Metal::FragmentColorFormat::RGBA8);
                if (export_prefix) {
                    std::ofstream output{std::string(export_prefix) + "-config" + std::to_string(variants.size()) + ".metal"};
                    output << v.shader->program.shader.source;
                    Check(bool(output), "Could not export lighting LUT shader");
                }
                variants.push_back(std::move(v));
            }
        for (unsigned domain = 0; domain < 2; ++domain) {
            const auto alignment = [device.Handle() minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatRG32Float];
            const auto bytes = TableCount * 8;
            tables[domain] = device.CreateBuffer(bytes + (alignment - bytes % alignment) % alignment, true);
            std::memset(tables[domain].contents, 0xcd, tables[domain].length);
            auto* values = static_cast<std::array<float, 2>*>(tables[domain].contents);
            for (unsigned table = 0; table < 24; ++table) for (unsigned i = 0; i < 256; ++i) {
                double input = i / 256.0, step = 1 / 256.0;
                if (table >= 8 && table < 16 && domain == 0) {
                    input = (i < 128 ? int(i) : int(i) - 256) / 128.0; step = 1 / 128.0;
                }
                const auto value = [&](double x) {
                    if (table >= 16) return Distance(table - 16, x);
                    if (table >= 8) return Spot(table - 8, x);
                    return (table + 1) / 32.0;
                };
                values[Offset(table) + i] = {float(value(input)), float(value(input + step) - value(input))};
            }
            views[domain] = device.CreateBufferTexture(tables[domain], 0, TableCount, MTLPixelFormatRG32Float);
            const auto* begin = static_cast<const std::byte*>(tables[domain].contents);
            snapshots[domain] = {begin, begin + tables[domain].length};
        }
        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                     width:32 height:32 mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate; desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        target = device.CreateTexture(desc);
    }

    unsigned Verify(Metal::Surface& surface, Format format, bool reverse_normal, Metal::TransferRuntime& transfers) {
        constexpr std::array<float, 9> spot_inputs{-1.5f, -1, -.375f, -.125f, 0, .125f, .5f, 1, 1.5f};
        constexpr std::array<float, 9> distance_inputs{-.25f, 0, .25f, .375f, .5f, .625f, .75f, 1, 1.25f};
        constexpr std::array<double, 3> diffuse{.75, .5, .25}, ambient{.125, .0625, .03125}, global{.03125, .0625, .125};
        unsigned count{};
        for (unsigned variant = 0; variant < variants.size(); ++variant) {
            const auto& v = variants[variant];
            for (unsigned sample = 0; sample < spot_inputs.size(); ++sample) {
                auto vertices = stream.Allocate(3 * sizeof(Metal::CapturedVertex));
                std::memset(vertices.data, 0, vertices.size);
                auto* data = static_cast<Metal::CapturedVertex*>(vertices.data);
                constexpr std::array<std::array<float, 2>, 3> positions{{{-1, -1}, {3, -1}, {-1, 3}}};
                for (unsigned i = 0; i < 3; ++i) {
                    const auto u = (positions[i][0] + 1) / 2, w = (positions[i][1] + 1) / 2;
                    data[i].position = {positions[i][0], positions[i][1], .5f, 1};
                    data[i].texcoord0 = {u, w, 0, 0}; data[i].quaternion = {0, 0, 0, 1};
                    data[i].view = {u - .5f, w - .5f, -.25f, 0}; data[i].clip = {1, 1, 1, 0};
                }
                auto uniforms = stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
                std::memset(uniforms.data, 0, uniforms.size);
                auto* fs = static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                fs->depth_offset = .5f;
                for (unsigned table = 0; table < 24; ++table) fs->lighting_lut_offset[table / 4][table % 4] = Offset(table);
                auto& light = fs->light_src[v.light];
                light.position = {0, 0, v.directional ? 2.f : 1.25f};
                light.spot_direction = {.25f, -.375f, spot_inputs[sample]};
                light.dist_atten_scale = sample % 2 ? .5f : -.25f;
                light.dist_atten_bias = distance_inputs[sample] - light.dist_atten_scale * (v.directional ? 2.f : 1.f);
                for (unsigned c = 0; c < 3; ++c) {
                    light.diffuse[c] = diffuse[c]; light.ambient[c] = ambient[c]; fs->lighting_global_ambient[c] = global[c];
                }
                const std::array<Metal::BufferBinding, 1> buffers{{{uniforms.buffer, uniforms.offset, 2}}};
                const std::array<Metal::TextureBinding, 3> images{{
                    {views[v.input_mode != 0], 0}, {surface.Image(), 3}, {surface.Image(Metal::SurfaceImage::Normal), 13}}};
                const std::array<Metal::SamplerBinding, 2> sampling{{{sampler.Handle(), 0}, {sampler.Handle(), 9}}};
                auto* pass = [[MTL4RenderPassDescriptor alloc] init];
                pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 0);
                Metal::PicaDrawState state{}; state.blend.color_write_mask = 15;
                draws.Draw(state, *v.shader, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32}, {vertices, 3}, buffers, images, sampling);
                const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 128);
                const auto block = Block(format);
                for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 32; ++x) {
                    // Closed-form triangular functions avoid reproducing shader LUT
                    // index/floor/delta code, and never read back a reference texture.
                    const double u = (x + .5) / 32, w = (y + .5) / 32;
                    const double lx = v.directional ? 0 : u - .5, ly = v.directional ? 0 : w - .5;
                    const double lz = v.directional ? 2 : 1;
                    const double length = std::sqrt(lx * lx + ly * ly + lz * lz);
                    const double dot = (lx / 255 + ly / 255 + lz * (reverse_normal ? -1 : 1)) / length;
                    const double lambert = v.input_mode == 2 ? std::abs(dot) : std::max(dot, 0.0);
                    double input = (lx * .25 - ly * .375 + lz * spot_inputs[sample]) / length;
                    if (v.input_mode == 1) input = std::max(input, 0.0);
                    // The selected active light enables two-sided LN here.
                    // Its physical ID chooses the SP table, not another slot's
                    // two-sided flag. Keep this scalar oracle independent of
                    // the old generator's erroneous physical-slot lookup.
                    if (v.input_mode == 2)
                        input = std::abs(dot);
                    const double spot = v.mode & 1 ? v.scale * Spot(v.light, input) : 1;
                    const double dist = v.mode & 2 ? Distance(v.light,
                        std::clamp(double(light.dist_atten_bias) + light.dist_atten_scale * length, 0.0, 1.0)) : 1;
                    const auto color = BlockColor(format, unsigned(u * 17) / block[0], unsigned(w * 11) / block[1], 7);
                    for (unsigned c = 0; c < 4; ++c) {
                        const auto expected = c == 3 ? color[c] : unsigned(std::lround(color[c] *
                            std::clamp(global[c] + (diffuse[c] * lambert + ambient[c]) * spot * dist, 0.0, 1.0)));
                        const auto actual = std::to_integer<u8>(pixels[(y * 32 + x) * 4 + c]);
                        if (std::abs(int(actual) - int(expected)) > 2)
                            throw std::runtime_error("Native lighting LUT mismatch format=" + std::to_string(unsigned(format)) +
                                " reversed=" + std::to_string(reverse_normal) + " variant=" + std::to_string(variant) +
                                " sample=" + std::to_string(sample) + " x=" + std::to_string(x) + " y=" + std::to_string(y) +
                                " channel=" + std::to_string(c) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
                    }
                }
                ++count;
            }
        }
        return count;
    }
    void CheckTables() const {
        for (unsigned i = 0; i < 2; ++i)
            Check(std::memcmp(tables[i].contents, snapshots[i].data(), snapshots[i].size()) == 0,
                  "Lighting draws changed LUT values, slopes, prefix or padding");
    }
private:
    Metal::GeometryRuntime geometry;
    Metal::DrawRuntime draws;
    Metal::ShaderRuntime shaders;
    Metal::StreamBuffer stream;
    Metal::Sampler sampler;
    std::vector<Variant> variants;
    std::array<id<MTLBuffer>, 2> tables;
    std::array<id<MTLTexture>, 2> views;
    std::array<std::vector<std::byte>, 2> snapshots;
    id<MTLTexture> target;
};
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc <= 2 && (argc == 1 || std::filesystem::path{argv[1]}.is_absolute()),
                  "Expected optional absolute lighting shader export prefix");
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TextureRuntime runtime{device, scheduler};
            Metal::TransferRuntime transfers{device, scheduler}; Image image;
            LightingDraw draws{device, scheduler, runtime, argc == 2 ? argv[1] : nullptr};
            unsigned cases{};
            for (auto format : Formats) for (bool reverse : {false, true}) {
                VideoCore::SurfaceParams params; params.width = params.height = 8; params.levels = 1;
                params.res_scale = 1; params.pixel_format = VideoCore::PixelFormat::RGBA8;
                params.type = VideoCore::SurfaceType::Texture;
                VideoCore::SurfaceBase guest{params, {}};
                Maps map{image, 17, 11, {}, Color{128, 128, u8(reverse ? 0 : 255), 255}};
                map.material.format = format; EncodeMap(map.color, format, 7);
                Metal::Surface surface{runtime, guest, &map.material}; surface.UploadCustom(&map.material, 0);
                cases += draws.Verify(surface, format, reverse, transfers);
            }
            draws.CheckTables();
            std::cout << cases << " native PICA Spot/Distance-LUT diffuse/TEV draws: all1024 RGBA pixels,"
                         " eight custom color formats/two normal signs, all8 light IDs, 96 shader variants,"
                         " disabled/spot/distance/both modes, signed/positive SP and active-light two-sided LN, six LUT scales,"
                         " directional/positional lights and nine paired input/clamp cases passed\n";
            std::cout << "Nonconstant triangular LUTs: independent closed-form interpolation oracle, shuffled24 table offsets,"
                         " negative slopes, unchanged shared buffer values/slopes/prefix/padding passed\n";
            std::cout << "No app/window/ROM/emulation or image decoding started\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
