// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_draw_runtime.h"
#include "mtl_gas_runtime.h"
#include "mtl_transfer.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Independent host conversion verifies the selected IEEE half policy, not
// unmeasured PICA rounding. Each fragment stores a half before the next adds.
std::uint16_t Half(float v) { return std::bit_cast<std::uint16_t>(_Float16(v)); }
float Float(std::uint16_t v) { return float(std::bit_cast<_Float16>(v)); }
bool AlphaPass(unsigned f) {
    switch (f) {
    case 0: return false;
    case 1: return true;
    case 2: return false;
    case 3: return true;
    case 4: return true;
    case 5: return true;
    case 6: return false;
    case 7: return false;
    }
    throw std::runtime_error("Invalid alpha comparison");
}
id<MTLTexture> Texture(Metal::Device& device, MTLPixelFormat format, bool mips = false) {
    auto* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:20 height:20 mipmapped:mips];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    return device.CreateTexture(d);
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 2 && std::filesystem::path{argv[1]}.is_absolute(), "Specify an absolute shader export directory");
            const std::filesystem::path output{argv[1]};
            std::filesystem::create_directories(output);
            Metal::Device device{output / "cache"}; device.SwitchDiskCache(123);
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::GeometryRuntime geometry{device, scheduler};
            Metal::DrawRuntime draws{device, scheduler, geometry};
            Metal::ShaderRuntime shaders{device}; Metal::GasRuntime gas{device, scheduler};
            Metal::GasFeedback feedback{device}; Metal::StreamBuffer storage{device, scheduler, 1 << 20};
            auto* descriptor = [[MTLSamplerDescriptor alloc] init];
            descriptor.minFilter = descriptor.magFilter = MTLSamplerMinMagFilterNearest;
            const std::array samplers{Metal::SamplerBinding{device.CreateSampler(descriptor), 10}};
            std::uint64_t cases{}, pixels{}, overlapping_fragments{};
            std::vector<Pica::Shader::FSConfig> configurations;
            for (unsigned function = 0; function < 8; ++function) {
                Pica::RegsInternal regs{};
                regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                auto config = Pica::Shader::FSConfig{regs};
                config.framebuffer.alpha_test_func.Assign(Pica::FramebufferRegs::CompareFunc(function));
                for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
                using Tev = Pica::TexturingRegs::TevStageConfig;
                auto stage = static_cast<Tev>(config.texture.tev_stages[0]);
                stage.color_source1.Assign(Tev::Source::PrimaryColor);
                stage.alpha_source1.Assign(Tev::Source::PrimaryColor);
                config.texture.tev_stages[0].sources_raw = stage.sources_raw;
                configurations.push_back(config);
                const auto fragment = shaders.Fragment(config, {}, Metal::FragmentColorFormat::RGBA8, {}, true);
                const auto restored=Metal::RestorePicaFragmentShader(fragment->program.shader.spirv,config,{},
                    Metal::FragmentColorFormat::RGBA8,true);
                Check(restored.shader==fragment->program.shader && restored.operations==fragment->program.operations,
                    "Gas accumulation canonical IR changed source/reflection/flags/operations");
                Check(fragment != shaders.Fragment(config, {}, Metal::FragmentColorFormat::RGBA8),
                      "Gas accumulation aliased an ordinary fragment cache entry");
                Check(fragment->program.shader.source.find("[[depth(") == std::string::npos,
                      "Gas accumulation still exports an ordinary fragment depth");
                if (function) Check(fragment->program.shader.source.find("raster_order_group(0)") != std::string::npos,
                                    "Gas accumulation lacks per-pixel raster ordering");
                std::ofstream file{output / ("accumulate_" + std::to_string(function) + ".metal")};
                file << fragment->program.shader.source; file.close(); Check(file.good(), "Shader export failed");
                for (unsigned depth_function = 0; depth_function < 4; ++depth_function)
                    for (unsigned count : {1u, 257u}) for (bool clipped : {false, true})
                        for (auto depth_format : {MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8}) {
                            auto vertices = storage.Allocate(count * 3 * sizeof(Metal::CapturedVertex));
                            std::memset(vertices.data, 0, vertices.size);
                            auto* v = static_cast<Metal::CapturedVertex*>(vertices.data);
                            constexpr std::array<std::array<float, 2>, 3> positions{{{-1,-1},{3,-1},{-1,3}}};
                            std::uint16_t plain = Half(.5f), depth = Half(.25f);
                            for (unsigned triangle = 0; triangle < count; ++triangle) {
                                const float density = float(17 + (triangle % 3) * 34) / 255;
                                for (unsigned i = 0; i < 3; ++i) {
                                    auto& vertex = v[triangle * 3 + i];
                                    vertex.position = {positions[i][0], positions[i][1], .5f, 1};
                                    vertex.color = {density, .4f, .6f, 127.f / 255};
                                    vertex.quaternion = {0, 0, 0, 1}; vertex.view = {0, 0, -1, 0};
                                    vertex.clip = {.5f, 0, 1, 0};
                                }
                                const float attenuation = depth_function < 2 ? float(depth_function) : .5f;
                                plain = Half(Float(plain) + density);
                                depth = Half(Float(depth) + density * attenuation);
                            }
                            const auto fs_buffer = storage.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
                            std::memset(fs_buffer.data, 0, fs_buffer.size);
                            auto* fs = static_cast<Pica::Shader::Generator::FSUniformData*>(fs_buffer.data);
                            fs->alphatest_ref = 128; fs->depth_offset = .25f;
                            Metal::GasUniforms config_uniforms{};
                            config_uniforms.delta_z_depth = (depth_function << 24) | 256;
                            auto gas_buffer = device.CreateBuffer(sizeof(config_uniforms), true);
                            std::memcpy(gas_buffer.contents, &config_uniforms, sizeof(config_uniforms));
                            const auto target = Texture(device, MTLPixelFormatRGBA8Unorm);
                            const auto scene = Texture(device, depth_format);
                            transfers.ClearDepthStencil(scene, .75, 0xa7, true, depth_format == MTLPixelFormatDepth32Float_Stencil8);
                            const auto snapshot = gas.SnapshotDepth(scene);
                            // A later write must not alter the depth read by this draw.
                            transfers.ClearDepthStencil(scene, .125, 0x5c, true, depth_format == MTLPixelFormatDepth32Float_Stencil8);
                            feedback.Reset();
                            const std::array buffers{Metal::BufferBinding{fs_buffer.buffer, fs_buffer.offset, 2},
                                Metal::BufferBinding{gas_buffer, 0, 23}, gas.PrepareFragmentMaximum(feedback)};
                            const std::array textures{Metal::TextureBinding{snapshot, 14}};
                            auto* pass = [[MTL4RenderPassDescriptor alloc] init];
                            pass.colorAttachments[0].texture = target;
                            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                            // Canonical RGBA reverses guest {D1Hi,D1Lo,D2Hi,D2Lo}.
                            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 52. / 255, 0, 56. / 255);
                            Metal::PicaDrawState state{}; state.blend.color_write_mask = 15; state.gas_accumulation = true;
                            const auto clip = clipped ? MTLScissorRect{5, 6, 9, 8} : MTLScissorRect{2, 2, 16, 16};
                            draws.Draw(state, *fragment, pass, {2, 2, 16, 16, 0, 1}, clip,
                                       {vertices, count * 3}, buffers, textures, samplers);
                            const auto actual = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 20, 20)}, 80);
                            for (unsigned y = 0; y < 20; ++y) for (unsigned x = 0; x < 20; ++x) {
                                const bool covered = x >= clip.x && y >= clip.y && x < clip.x + clip.width && y < clip.y + clip.height;
                                const bool rendered = covered && AlphaPass(function);
                                const auto a = rendered ? plain : Half(.5f), b = rendered ? depth : Half(.25f);
                                const std::array<std::uint8_t, 4> expected{std::uint8_t(b), std::uint8_t(b >> 8), std::uint8_t(a), std::uint8_t(a >> 8)};
                                if (std::memcmp(actual.data() + (y * 20 + x) * 4, expected.data(), 4)) {
                                    std::cerr << "function=" << function << " depth=" << depth_function << " triangles=" << count
                                              << " pixel=" << x << ',' << y << '\n';
                                    throw std::runtime_error("Native ordered Gas accumulation byte mismatch");
                                }
                                ++pixels; if (rendered) overlapping_fragments += count;
                            }
                            Check(gas.ReadMaximum(feedback) == (AlphaPass(function) ? plain : 0),
                                  "Gas maximum includes discarded fragments or lost contributions");
                            ++cases;
                        }
            }
            // Snapshotting a nonzero mip must copy that mip's extent/content.
            const auto mipped = Texture(device, MTLPixelFormatDepth32Float, true);
            const auto source = Texture(device, MTLPixelFormatDepth32Float);
            transfers.ClearDepthStencil(source, .375, 0, true, false);
            transfers.CopyTexture({source, 0, 0, MTLRegionMake2D(0, 0, 10, 10)},
                                  {mipped, 1, 0, MTLRegionMake2D(0, 0, 10, 10)});
            const auto mip = gas.SnapshotDepth(mipped, 1);
            Check(mip.width == 10 && mip.height == 10, "Gas depth snapshot lost mip dimensions");
            const auto bytes = transfers.Readback({mip, 0, 0, MTLRegionMake2D(0, 0, 10, 10)}, 40);
            for (unsigned i = 0; i < 100; ++i) {
                float value{}; std::memcpy(&value, bytes.data() + i * 4, 4);
                Check(value == .375f, "Gas depth snapshot lost mip contents");
            }
            const auto zero = gas.SnapshotDepth(nil);
            Check(zero.width == 1 && zero.height == 1, "Gas Never/Always depth binding is incomplete");
            {
                const auto unused = gas.SceneDepthForDraw(nil, 0, false);
                const auto count = scheduler.EncoderCount();
                Check(gas.SceneDepthForDraw(mipped, 1, false) == unused &&
                      scheduler.EncoderCount() == count,
                      "Gas Never/Always rebuilt/copied an unused scene depth snapshot");
                auto first_read = device.CreateBuffer(400, true);
                const auto first = gas.SceneDepthForDraw(mipped, 1, true);
                transfers.Download({first, 0, 0, MTLRegionMake2D(0, 0, 10, 10)}, first_read,
                    {0, 40, 0, MTLBlitOptionNone});
                transfers.ClearDepthStencil(mipped, .625, 0, true, false, 1);
                const auto second = gas.SceneDepthForDraw(mipped, 1, true);
                Check(first == second, "Gas consecutive draws failed to reuse bounded depth scratch");
                const auto latest = transfers.Readback({second, 0, 0, MTLRegionMake2D(0, 0, 10, 10)}, 40);
                for (unsigned i = 0; i < 100; ++i) {
                    float value{}; std::memcpy(&value, latest.data() + i * 4, 4);
                    Check(value == .625f && static_cast<const float*>(first_read.contents)[i] == .375f,
                        "Gas scratch reuse changed an earlier GPU consumer or the next draw snapshot");
                }
                bool rejected{};
                try { gas.SceneDepthForDraw(mipped, 1000, true); }
                catch (const std::invalid_argument&) { rejected = true; }
                Check(rejected, "Gas cached scene-depth path accepted an invalid mip");
                std::cout << "Gas borrowed scene depth: unused binding without copies, nonzero-mip scratch reuse and earlier GPU consumer preserved\n";
            }
            std::uint64_t prewarm_hits{};
            // Production deliberately disables binary persistence during shader
            // instrumentation. Run this executable once more without that layer
            // to verify persistent identities without weakening the guard.
            if (device.DiskCacheStats().enabled) {
                Metal::ShaderRuntime warm{device};
                const std::atomic_bool stop{false};
                const auto preload = warm.Prewarm(stop);
                Check(preload.compiled == 16 && !preload.failed && !preload.cancelled,
                      "Ordinary/Gas disk prewarm lost distinct specialization records");
                for (const auto& configuration : configurations) {
                    const auto ordinary = warm.Fragment(configuration, {}, Metal::FragmentColorFormat::RGBA8);
                    const auto accumulation = warm.Fragment(configuration, {}, Metal::FragmentColorFormat::RGBA8, {}, true);
                    Check(ordinary != accumulation &&
                          accumulation->program.shader.source.find("[[depth(") == std::string::npos,
                          "Disk-prewarmed Gas accumulation reused an ordinary fragment");
                }
                prewarm_hits = warm.DiskStats().prewarm_hits;
                Check(prewarm_hits == 16 && !warm.DiskStats().translations,
                      "Warm Gas shader keys missed their compiled disk records");
            }
            scheduler.Finish();
            Check(cases == 256 && pixels == 102400, "Incomplete native Gas accumulation matrix");
            std::cout << "PASS: native ordered Gas accumulation cases=" << cases << " rgba_pixels=" << pixels
                      << " contributing_fragments=" << overlapping_fragments
                      << " max_overlap=257 depth_functions=4 alpha_functions=8 stable_depth_formats=2 "
                         "mip_snapshot_pixels=100 distinct_disk_prewarm_hits=" << prewarm_hits << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
