// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_draw_runtime.h"
#include "mtl_gas_runtime.h"
#include "mtl_transfer.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
constexpr std::array<double, 4> Guard{17. / 255, 34. / 255, 51. / 255, 68. / 255};
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool AlphaPass(unsigned alpha, unsigned reference, unsigned function) {
    switch (function) {
    case 0: return false;
    case 1: return true;
    case 2: return alpha == reference;
    case 3: return alpha != reference;
    case 4: return alpha < reference;
    case 5: return alpha <= reference;
    case 6: return alpha > reference;
    case 7: return alpha >= reference;
    }
    throw std::runtime_error("Invalid alpha comparison");
}
id<MTLTexture> Texture(Metal::Device& device, MTLPixelFormat format, unsigned side) {
    auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:side height:side mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return device.CreateTexture(d);
}
void Compare(const std::vector<std::byte>& bytes, MTLPixelFormat format, unsigned x, unsigned y, bool rendered,
             unsigned index, const std::string& label) {
    for (unsigned c = 0; c < 4; ++c) {
        double want = rendered ? (c == 3 ? 136. / 255 : (85. + c * 17 + index * 8) / 255) : Guard[c];
        double actual{}; const bool floating = format == MTLPixelFormatRGBA32Float;
        if (floating) {
            float value{}; std::memcpy(&value, bytes.data() + (y * 20 + x) * 16 + c * 4, 4); actual = value;
        } else {
            actual = std::to_integer<unsigned>(bytes[(y * 20 + x) * 4 + c]) / 255.;
            if (format == MTLPixelFormatRGBA8Unorm_sRGB && c < 3) {
                // Compare encoded bytes against independently quantized sRGB.
                want = std::round(255 * (want <= .0031308 ? 12.92 * want :
                                       1.055 * std::pow(want, 1. / 2.4) - .055)) / 255;
            }
        }
        const double tolerance = floating ? .000002 :
            format == MTLPixelFormatRGBA8Unorm_sRGB ? 1. / 255 : .002;
        if (!std::isfinite(actual) || std::abs(actual - want) > tolerance) {
            std::cerr << label << " pixel=" << x << ',' << y << " component=" << c
                      << " actual=" << actual << " expected=" << want << '\n';
            throw std::runtime_error("Native Gas postsynthesis independent pixel failed");
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 2, "Specify an absolute shader export directory");
            const std::filesystem::path output{argv[1]}; Check(output.is_absolute(), "Use an absolute path");
            std::filesystem::create_directories(output);
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TransferRuntime transfers{device, scheduler};
            Metal::GeometryRuntime geometry{device, scheduler}; Metal::DrawRuntime draws{device, scheduler, geometry};
            Metal::ShaderRuntime shaders{device}; Metal::StreamBuffer storage{device, scheduler, 4096};
            std::uint64_t cases{}, pixels{}, shadow_words{};
            for (const bool shadow : {false}) for (unsigned function = 0; function < 8; ++function) {
                Pica::RegsInternal regs{};
                regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                auto config = Pica::Shader::FSConfig{regs};
                config.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Gas);
                config.framebuffer.shadow_rendering.Assign(shadow);
                config.framebuffer.alpha_test_func.Assign(Pica::FramebufferRegs::CompareFunc(function));
                for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
                using Tev = Pica::TexturingRegs::TevStageConfig;
                auto stage = static_cast<Tev>(config.texture.tev_stages[0]);
                stage.color_source1.Assign(Tev::Source::PrimaryColor);
                stage.alpha_source1.Assign(Tev::Source::PrimaryColor);
                config.texture.tev_stages[0].sources_raw = stage.sources_raw;
                stage = static_cast<Tev>(config.texture.tev_stages[5]);
                stage.color_source3.Assign(Tev::Source::Constant);
                if (function & 4) {
                    // Exercise the generated non-pass-through final stage too.
                    stage.color_source1.Assign(Tev::Source::PrimaryColor);
                    stage.alpha_source1.Assign(Tev::Source::PrimaryColor);
                }
                config.texture.tev_stages[5].sources_raw = stage.sources_raw;
                const auto fragment = shaders.Fragment(config, {}, Metal::FragmentColorFormat::RGBA8);
                const auto restored=Metal::RestorePicaFragmentShader(fragment->program.shader.spirv,config,{},
                    Metal::FragmentColorFormat::RGBA8,false);
                Check(restored.shader==fragment->program.shader && restored.operations==fragment->program.operations,
                    "Gas postsynthesis canonical IR changed source/reflection/flags/operations");
                Check((fragment->program.shader.source.find("[[depth(") != std::string::npos) == (function != 0),
                      "Gas postsynthesis lost its fragment depth output");
                if (function != 0)
                    Check(std::any_of(fragment->program.shader.resources.begin(), fragment->program.shader.resources.end(),
                        [](const auto& r) { return r.active && r.metal_index == 23 && r.byte_size == 608; }),
                        "Gas postsynthesis lost its raw register block");
                if (shadow) for (const auto& resource : fragment->program.shader.resources)
                    Check(!resource.active || resource.kind != Metal::ShaderResourceKind::StorageTexture,
                          "Gas fallback retained an active shadow writer");
                std::ofstream file{output / (std::string(shadow ? "shadow_" : "color_") + std::to_string(function) + ".metal")};
                file << fragment->program.shader.source; file.close(); Check(file.good(), "Shader export failed");
                for (unsigned alpha : {0U, 127U, 255U}) for (unsigned reference : {0U, 128U, 255U})
                    for (const auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA32Float})
                        for (const bool clipped : {false, true}) {
                            auto vertices = storage.Allocate(3 * sizeof(Metal::CapturedVertex));
                            std::memset(vertices.data, 0, vertices.size);
                            auto* v = static_cast<Metal::CapturedVertex*>(vertices.data);
                            constexpr std::array<std::array<float, 2>, 3> positions{{{-1,-1},{3,-1},{-1,3}}};
                            for (unsigned i = 0; i < 3; ++i) {
                                v[i].position = {positions[i][0], positions[i][1], .5f, 1};
                                v[i].color = {.2f, .4f, .6f, alpha / 255.f};
                                v[i].quaternion = {0, 0, 0, 1}; v[i].view = {0, 0, -1, 0};
                                v[i].clip = {.5f, 0, 1, 0};
                            }
                            auto uniforms = storage.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
                            std::memset(uniforms.data, 0, uniforms.size);
                            auto* fs = static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                            fs->alphatest_ref = reference;
                            fs->depth_offset = .875f;
                            // D1=1, D2=0.5, guest byte order {3C,00,38,00}.
                            fs->const_color[5] = {0, 56.f / 255, 0, 60.f / 255};
                            Metal::GasUniforms gas_uniforms{};
                            gas_uniforms.reciprocal_maximum = 0x3C00;
                            gas_uniforms.attenuation = 0x3C00;
                            gas_uniforms.shading_depth = function & 1;
                            for (unsigned i = 0; i < 8; ++i) gas_uniforms.color_lut[i + 8] = 0x776655 + i * 0x080808;
                            for (auto& entry : gas_uniforms.fog_lut) entry = 1092u << 13;
                            auto gas_buffer = device.CreateBuffer(sizeof(gas_uniforms), true);
                            std::memcpy(gas_buffer.contents, &gas_uniforms, sizeof(gas_uniforms));
                            auto target = Texture(device, format, 20);
                            auto utility = Texture(device, MTLPixelFormatR32Uint, 16);
                            std::array<std::uint32_t, 256> words;
                            for (unsigned i = 0; i < words.size(); ++i) words[i] = 0x12345678U ^ (i * 0x10203U);
                            auto input = device.CreateBuffer(sizeof(words), true); std::memcpy(input.contents, words.data(), sizeof(words));
                            transfers.Upload(input, {0, 64, 0}, {utility, 0, 0, MTLRegionMake2D(0, 0, 16, 16)});
                            auto* pass = [[MTL4RenderPassDescriptor alloc] init];
                            pass.colorAttachments[0].texture = target;
                            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                            pass.colorAttachments[0].clearColor = MTLClearColorMake(Guard[0], Guard[1], Guard[2], Guard[3]);
                            const std::array buffers{Metal::BufferBinding{uniforms.buffer, uniforms.offset, 2},
                                Metal::BufferBinding{gas_buffer, 0, 23}};
                            const std::array textures{Metal::TextureBinding{utility, 12}};
                            Metal::PicaDrawState state{}; state.blend.color_write_mask = 15; state.shadow = shadow;
                            const auto clip = clipped ? MTLScissorRect{5, 6, 9, 8} : MTLScissorRect{2, 2, 16, 16};
                            draws.Draw(state, *fragment, pass, {2, 2, 16, 16, 0, 1}, clip, {vertices, 3}, buffers, textures, {});
                            const auto actual = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 20, 20)},
                                20 * (format == MTLPixelFormatRGBA32Float ? 16 : 4));
                            const auto label = std::string(shadow ? "shadow" : "color") + " function=" + std::to_string(function) +
                                " alpha=" + std::to_string(alpha) + " reference=" + std::to_string(reference) +
                                " format=" + std::to_string(format) + " clipped=" + std::to_string(clipped);
                            for (unsigned y = 0; y < 20; ++y) for (unsigned x = 0; x < 20; ++x) {
                                const bool inside = x >= clip.x && y >= clip.y && x < clip.x + clip.width && y < clip.y + clip.height;
                                Compare(actual, format, x, y, !shadow && inside && AlphaPass(alpha, reference, function),
                                        function & 1 ? 4 : 7, label); ++pixels;
                            }
                            const auto untouched = transfers.Readback({utility, 0, 0, MTLRegionMake2D(0, 0, 16, 16)}, 64);
                            Check(untouched.size() == sizeof(words) && !std::memcmp(untouched.data(), words.data(), sizeof(words)),
                                  "Gas postsynthesis wrote the utility/shadow image");
                            shadow_words += words.size(); ++cases;
                        }
            }
            scheduler.Finish();
            auto bad = Pica::Shader::FSConfig{Pica::RegsInternal{}};
            bad.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Gas);
            bad.framebuffer.shadow_rendering.Assign(1);
            bool rejected{};
            try { shaders.Fragment(bad, {}, Metal::FragmentColorFormat::RGBA8); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Gas silently accepted the incompatible shadow operation");
            Check(cases == 432 && pixels == 172800 && shadow_words == 110592, "Incomplete native Gas fragment matrix");
            std::cout << "passed gas_native_postsynthesis cases=" << cases << " rgba_pixels=" << pixels
                      << " unchanged_shadow_words=" << shadow_words << " postsynthesis_matrix_only=1\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
