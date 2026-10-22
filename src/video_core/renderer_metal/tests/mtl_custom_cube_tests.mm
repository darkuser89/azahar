// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Real cache/Core/PICA/Memory objects only; no emulator initialization or window.
#include "mtl_pica_vertex_fixture.h"
#include "mtl_custom_texture_fixture.h"
#include "mtl_draw_runtime.h"
#include "mtl_texture_runtime.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
using namespace Metal::Test::Custom;
class Window final : public Frontend::EmuWindow { public: void PollEvents() override {} };
class Renderer final : public VideoCore::RendererBase {
public:
    Renderer(Core::System& system, Window& window) : RendererBase{system, window, nullptr} {}
    VideoCore::RasterizerInterface* Rasterizer() override { return nullptr; }
    void SwapBuffers() override {}
    void TryPresent(int, bool) override {}
};
unsigned Morton(unsigned x, unsigned y, unsigned width) {
    constexpr std::array<unsigned, 8> bits{0, 1, 4, 5, 16, 17, 20, 21};
    return ((y / 8) * (width / 8) + x / 8) * 64 + bits[x % 8] + 2 * bits[y % 8];
}
Color GuestColor(unsigned face, unsigned level, unsigned x, unsigned y) {
    return {u8(20 + face * 30), u8(11 * x + level * 17), u8(13 * y + level * 19), u8(210 + face)};
}
Color CustomColor(Format format, const Maps& map, unsigned seed, unsigned x, unsigned y, unsigned width) {
    const auto block = Block(format);
    const auto sx = (2 * x + 1) * map.material.width / (2 * width);
    const auto sy = (2 * y + 1) * map.material.height / (2 * width);
    return BlockColor(format, sx / block[0], sy / block[1], seed);
}

class CubeDraw {
public:
    CubeDraw(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TextureRuntime& runtime,
             const char* export_prefix)
        : geometry{device, scheduler}, draws{device, scheduler, geometry}, shaders{device},
          storage{device, scheduler, 4096} {
        using C = Pica::TexturingRegs::TextureConfig;
        using Tev = Pica::TexturingRegs::TevStageConfig;
        Pica::RegsInternal regs{}; regs.lighting.disable.Assign(1);
        regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
        auto config = Pica::Shader::FSConfig{regs}; config.texture.texture0_type.Assign(C::TextureCube);
        for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
        auto stage = static_cast<Tev>(config.texture.tev_stages[0]);
        stage.color_source1.Assign(Tev::Source::Texture0); stage.alpha_source1.Assign(Tev::Source::Texture0);
        config.texture.tev_stages[0].sources_raw = stage.sources_raw;
        for (unsigned level = 0; level < 2; ++level) for (bool linear : {false, true}) {
            const auto index = level * 2 + linear;
            Metal::PicaSamplerConfigs settings{};
            settings[0] = {C::ClampToEdge, C::ClampToEdge, unsigned(linear), unsigned(linear), 0, level, level};
            programs[index] = shaders.Fragment(config, {}, Metal::FragmentColorFormat::RGBA8, settings);
            samplers[index] = std::make_unique<Metal::Sampler>(runtime, VideoCore::SamplerParams{
                C::TextureFilter(linear), C::TextureFilter(linear), C::Nearest,
                C::ClampToEdge, C::ClampToEdge, 0, level, level, 0});
            if (export_prefix) {
                std::ofstream output{std::string(export_prefix) + "-mip" + std::to_string(level) +
                    (linear ? "-linear.metal" : "-nearest.metal")};
                output << programs[index]->program.shader.source;
                Check(bool(output), "Could not export custom cube fragment source");
            }
        }
        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                     width:32 height:32 mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate;
        desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        target = device.CreateTexture(desc);
    }

    template <class Texel>
    void Verify(id<MTLTexture> texture, unsigned face, unsigned level, bool linear,
                const Texel& texel, Metal::TransferRuntime& transfers, const std::string& context) {
        auto vertices = storage.Allocate(3 * sizeof(Metal::CapturedVertex));
        std::memset(vertices.data, 0, vertices.size);
        auto* data = static_cast<Metal::CapturedVertex*>(vertices.data);
        constexpr std::array<std::array<float, 2>, 3> positions{{{-1, -1}, {3, -1}, {-1, 3}}};
        for (unsigned i = 0; i < 3; ++i) {
            const auto u = (positions[i][0] + 1) / 2, v = (positions[i][1] + 1) / 2;
            const auto s = 2 * (linear ? .125f + .75f * u : u) - 1;
            const auto t = 2 * (linear ? .125f + .75f * v : v) - 1;
            // Six independent standard cube face bases; major axis stays constant.
            const std::array<std::array<float, 3>, 6> direction{{
                {1, -t, -s}, {-1, -t, s}, {s, 1, t}, {s, -1, -t}, {s, -t, 1}, {-s, -t, -1}}};
            data[i].position = {positions[i][0], positions[i][1], .5f, 1};
            data[i].color = {1, 1, 1, 1};
            data[i].texcoord0 = {direction[face][0], direction[face][1], 0, 0};
            data[i].clip = {1, 1, direction[face][2], 0};
        }
        auto uniforms = storage.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
        std::memset(uniforms.data, 0, uniforms.size);
        static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data)->depth_offset = .5f;
        const std::array<Metal::BufferBinding, 1> buffers{{{uniforms.buffer, uniforms.offset, 2}}};
        const std::array<Metal::TextureBinding, 1> images{{{texture, 3}}};
        const std::array<Metal::SamplerBinding, 1> sampling{{{samplers[level * 2 + linear]->Handle(), 0}}};
        auto* pass = [[MTL4RenderPassDescriptor alloc] init];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 0);
        Metal::PicaDrawState state{}; state.blend.color_write_mask = 15;
        draws.Draw(state, *programs[level * 2 + linear], pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                   {vertices, 3}, buffers, images, sampling);
        const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 128);
        const auto dim = unsigned(texture.width) >> level;
        for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 32; ++x) {
            const double u = (x + .5) / 32, v = (y + .5) / 32;
            std::array<double, 4> expected{};
            if (linear) {
                // Interior footprint avoids seams; each tap comes from the scalar
                // guest/custom upload oracle, never from a GPU reference image.
                const auto px = (.125 + .75 * u) * dim - .5, py = (.125 + .75 * v) * dim - .5;
                const auto tx = unsigned(std::floor(px)), ty = unsigned(std::floor(py));
                Check(tx + 1 < dim && ty + 1 < dim, "Linear cube oracle crossed a face seam");
                const auto fx = px - tx, fy = py - ty;
                for (unsigned j = 0; j < 2; ++j) for (unsigned i = 0; i < 2; ++i) {
                    const auto color = texel(tx + i, ty + j);
                    const auto weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
                    for (unsigned c = 0; c < 4; ++c) expected[c] += color[c] * weight;
                }
            } else {
                const auto color = texel(unsigned(u * dim), unsigned(v * dim));
                for (unsigned c = 0; c < 4; ++c) expected[c] = color[c];
            }
            for (unsigned c = 0; c < 4; ++c)
                if (std::abs(int(std::to_integer<u8>(pixels[(y * 32 + x) * 4 + c])) -
                             int(std::lround(expected[c]))) > (linear ? 2 : 1))
                    throw std::runtime_error("Custom cache cube fragment mismatch " + context +
                        " face=" + std::to_string(face) + " mip=" + std::to_string(level) +
                        " linear=" + std::to_string(linear) + " x=" + std::to_string(x) +
                        " y=" + std::to_string(y) + " channel=" + std::to_string(c) +
                        " expected=" + std::to_string(std::lround(expected[c])) +
                        " actual=" + std::to_string(std::to_integer<u8>(pixels[(y * 32 + x) * 4 + c])));
        }
    }
private:
    Metal::GeometryRuntime geometry;
    Metal::DrawRuntime draws;
    Metal::ShaderRuntime shaders;
    Metal::StreamBuffer storage;
    std::array<std::shared_ptr<const Metal::CompiledFragmentShader>, 4> programs;
    std::array<std::unique_ptr<Metal::Sampler>, 4> samplers;
    id<MTLTexture> target;
};
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc <= 2, "Expected optional fragment-source export prefix only");
            Core::System system;
            auto image = std::make_shared<Image>(); system.RegisterImageInterface(image);
            Memory::MemorySystem memory{system}; Pica::PicaCore pica{memory, nullptr};
            Metal::Test::InitializeFixtureUniforms(pica.vs_setup);
            Metal::Test::InitializeFixtureUniforms(pica.gs_setup);
            Window window; Renderer renderer{system, window}; VideoCore::CustomTexManager manager{system};
            Settings::values.custom_textures.SetValue(false); Settings::values.dump_textures.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device; Metal::Scheduler scheduler{device};
            Metal::TextureRuntime runtime{device, scheduler}; Metal::TransferRuntime transfer{device, scheduler};
            CubeDraw cube_draw{device, scheduler, runtime, argc == 2 ? argv[1] : nullptr};
            unsigned configurations{}, face_mips{}, physical_flushes{}, partial_copies{}, rejected_copies{};
            unsigned fragment_draws{};
            std::array<unsigned, 3> cube_widths{}, source_scales{};
            for (unsigned scale : {1U, 2U, 3U}) for (auto format : Formats)
                for (auto dimensions : {std::array{16U, 16U}, std::array{32U, 32U},
                                        std::array{33U, 25U}, std::array{11U, 9U}})
                    for (bool mixed : {false, true}) {
                        Settings::values.resolution_factor.SetValue(scale);
                        std::array<std::array<std::unique_ptr<Maps>, 2>, 6> maps;
                        Metal::RasterizerCache cache{memory, manager, runtime, pica.regs.internal, renderer};
                        std::array<PAddr, 6> addresses{};
                        std::array<VideoCore::SurfaceId, 6> ids{};
                        const auto is_custom = [&](unsigned face) { return !mixed || face % 2 == 0; };
                        for (unsigned face = 0; face < 6; ++face) {
                            addresses[face] = Memory::VRAM_PADDR + 0x10000 + face * 0x2000;
                            auto* bytes = memory.GetPhysicalPointer(addresses[face]);
                            std::memset(bytes - 16, 0xcd, 1280 + 32);
                            unsigned offset{};
                            for (unsigned level = 0; level < 2; ++level) {
                                const auto dim = 16U >> level;
                                for (unsigned y = 0; y < dim; ++y) for (unsigned x = 0; x < dim; ++x) {
                                    const auto color = GuestColor(face, level, x, dim - 1 - y);
                                    for (unsigned c = 0; c < 4; ++c)
                                        bytes[offset + Morton(x, y, dim) * 4 + c] = color[3 - c];
                                }
                                offset += dim * dim * 4;
                            }
                            Pica::Texture::TextureInfo info{};
                            info.physical_address = addresses[face]; info.width = info.height = 16;
                            info.format = Pica::TexturingRegs::TextureFormat::RGBA8;
                            ids[face] = cache.GetTextureSurface(info, 1);
                            if (!is_custom(face)) continue;
                            auto& source = cache.GetSurface(ids[face]);
                            Check(source.IsRegionValid(source.GetInterval()), "Cube fixture guest source was not validated");
                            for (unsigned level = 0; level < 2; ++level) {
                                maps[face][level] = std::make_unique<Maps>(*image,
                                    std::max(1U, dimensions[0] >> level), std::max(1U, dimensions[1] >> level),
                                    Color{}, Color{128, 128, 255, 255});
                                maps[face][level]->material.format = format;
                                EncodeMap(maps[face][level]->color, format, 7 + face + level);
                            }
                            Metal::Surface replacement{runtime, source, &maps[face][0]->material};
                            source = std::move(replacement);
                            for (unsigned level = 0; level < 2; ++level)
                                source.UploadCustom(&maps[face][level]->material, level);
                        }
                        // NoFilter keeps ordinary texture uploads at scale1, regardless
                        // of the resolution setting. Exercise real scaled source/cube
                        // allocations through the public native Surface operation.
                        for (const auto id : ids) cache.GetSurface(id).ScaleUp(scale);
                        const VideoCore::TextureCubeConfig config{addresses[0], addresses[1], addresses[2],
                            addresses[3], addresses[4], addresses[5], 16, 2, Pica::TexturingRegs::TextureFormat::RGBA8};
                        auto texture = cache.GetTextureCube(config).Image();
                        Check(texture.textureType == MTLTextureTypeCube && texture.mipmapLevelCount == 2,
                              "Cached custom cube lost its six-face/mip allocation");
                        cube_widths[scale - 1] = unsigned(texture.width);
                        source_scales[scale - 1] = cache.GetSurface(ids[0]).res_scale;
                        Check(texture.width == 16 * scale && texture.height == 16 * scale &&
                              source_scales[scale - 1] == scale,
                              "Custom cube test did not exercise the requested physical resolution scale");
                        const auto verify = [&](bool updated) {
                            for (unsigned face = 0; face < 6; ++face) for (unsigned level = 0; level < 2; ++level) {
                                const auto dim = unsigned(texture.width) >> level;
                                const auto pixels = transfer.Readback({texture, level, face,
                                    MTLRegionMake2D(0, 0, dim, dim)}, dim * 4);
                                const auto texel = [&](unsigned x, unsigned y) {
                                    return is_custom(face) ? CustomColor(format, *maps[face][level],
                                        7 + face + level + (updated && face == 2 ? 17 : 0), x, y, dim) :
                                        GuestColor(face, level, (2 * x + 1) * (16U >> level) / (2 * dim),
                                                   (2 * y + 1) * (16U >> level) / (2 * dim));
                                };
                                for (unsigned y = 0; y < dim; ++y) for (unsigned x = 0; x < dim; ++x) {
                                    const auto color = texel(x, y);
                                    for (unsigned c = 0; c < 4; ++c)
                                        if (std::abs(int(std::to_integer<u8>(pixels[(y * dim + x) * 4 + c])) - color[c]) > 1)
                                            throw std::runtime_error("Custom cache cube mismatch format=" +
                                                std::to_string(unsigned(format)) + " scale=" + std::to_string(scale) +
                                                " face=" + std::to_string(face) + " mip=" + std::to_string(level) +
                                                " updated=" + std::to_string(updated) + " x=" + std::to_string(x) +
                                                " y=" + std::to_string(y));
                                }
                                ++face_mips;
                                for (bool linear : {false, true}) {
                                    cube_draw.Verify(texture, face, level, linear, texel, transfer,
                                        "format=" + std::to_string(unsigned(format)) + " scale=" +
                                        std::to_string(scale) + " updated=" + std::to_string(updated));
                                    ++fragment_draws;
                                }
                            }
                        };
                        verify(false);
                        Check(cache.GetTextureCube(config).Image() == texture, "Unchanged cube cache lookup replaced its allocation");
                        auto& changed = cache.GetSurface(ids[2]);
                        for (unsigned level = 0; level < 2; ++level) {
                            EncodeMap(maps[2][level]->color, format, 7 + 2 + level + 17);
                            changed.UploadCustom(&maps[2][level]->material, level);
                        }
                        cache.InvalidateRegion(addresses[2], 1280, ids[2]);
                        Check(cache.GetTextureCube(config).Image() == texture, "Updated cube lookup replaced an unchanged-size allocation");
                        verify(true);
                        cache.FlushRegion(addresses[2], 1280);
                        unsigned offset{};
                        for (unsigned level = 0; level < 2; ++level) {
                            const auto dim = 16U >> level;
                            for (unsigned y = 0; y < dim; ++y) for (unsigned x = 0; x < dim; ++x) {
                                const auto color = CustomColor(format, *maps[2][level], 7 + 2 + level + 17, x, dim - 1 - y, dim);
                                for (unsigned c = 0; c < 4; ++c)
                                    Check(std::abs(int(memory.GetPhysicalPointer(addresses[2])[offset + Morton(x, y, dim) * 4 + c]) - color[3 - c]) <= 1,
                                          "Custom GPU-owner cache flush lost a physical VRAM color/mip byte");
                            }
                            offset += dim * dim * 4; ++physical_flushes;
                        }
                        for (unsigned face = 0; face < 6; ++face) {
                            const auto* bytes = memory.GetPhysicalPointer(addresses[face]);
                            for (unsigned i = 0; i < 16; ++i)
                                Check(bytes[int(i) - 16] == 0xcd && bytes[1280 + i] == 0xcd,
                                      "Custom cube owner flush changed a physical VRAM guard byte");
                        }
                        auto& cube = cache.GetTextureCube(config);
                        const unsigned dim = unsigned(texture.width);
                        const auto face_region = Metal::TextureRegion{texture, 0, 4, MTLRegionMake2D(0, 0, dim, dim)};
                        auto expected = transfer.Readback(face_region, dim * 4);
                        VideoCore::TextureCopy part{};
                        part.src_offset = {1, 2}; part.dst_offset = {3, 4}; part.dst_layer = 4; part.extent = {2, 2};
                        Check(runtime.CopyTextures(changed, cube, std::span{&part, 1}), "Partial custom cube copy was rejected");
                        const auto block = Block(format);
                        for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x) {
                            const auto color = BlockColor(format, (x + 1) / block[0], (y + 2) / block[1], 7 + 2 + 17);
                            for (unsigned c = 0; c < 4; ++c)
                                expected[((y + 4) * dim + x + 3) * 4 + c] = std::byte{color[c]};
                        }
                        const auto actual = transfer.Readback(face_region, dim * 4);
                        for (unsigned i = 0; i < actual.size(); ++i) {
                            const auto x = (i / 4) % dim, y = (i / 4) / dim;
                            const bool written = x >= 3 && x < 5 && y >= 4 && y < 6;
                            Check(written ? std::abs(int(std::to_integer<u8>(actual[i])) - std::to_integer<u8>(expected[i])) <= 1
                                          : actual[i] == expected[i],
                                  "Whole-face custom cube correction changed partial-copy coordinates or outside pixels");
                        }
                        ++partial_copies;
                        for (unsigned invalid = 0; invalid < 2; ++invalid) {
                            VideoCore::TextureCopy bad{};
                            bad.extent = {16 * scale, 16 * scale}; bad.dst_layer = 4;
                            if (invalid == 0) bad.src_level = UINT32_MAX; else bad.src_layer = 1;
                            bool caught{};
                            try { runtime.CopyTextures(changed, cube, std::span{&bad, 1}); }
                            catch (const std::invalid_argument&) { caught = true; }
                            Check(caught && transfer.Readback(face_region, dim * 4) == actual,
                                  "Invalid custom cube source mip/slice changed destination bytes");
                            ++rejected_copies;
                        }
                        cache.ClearAll(false); ++configurations;
                    }
            Check(!system.IsPoweredOn(), "Data-only cube test started emulation");
            std::cout << configurations << " actual custom cube cache configurations, " << face_mips
                      << " full RGBA face/mip oracles and " << physical_flushes
                      << " GPU-owner mip flushes to real VRAM passed\n";
            std::cout << "Resolution settings1/2/3: actual cube widths=" << cube_widths[0] << ','
                      << cube_widths[1] << ',' << cube_widths[2] << "; custom source scales="
                      << source_scales[0] << ',' << source_scales[1] << ',' << source_scales[2] << '\n';
            std::cout << "All8 custom formats, resolution settings1/2/3, four replacement dimensions, six faces/two mips,"
                         " mixed guest/custom faces, cache reuse and owner invalidation passed\n";
            std::cout << partial_copies << " partial custom cube copies preserve physical pixel coordinates/outside pixels; "
                      << rejected_copies << " invalid source mip/slice copies preserve destination bytes passed\n";
            std::cout << fragment_draws << " actual cache custom cube -> native PICA fragment/TEV draws:"
                         " all1024 RGBA pixels, nearest full-face and independent interior bilinear tap oracles,"
                         " six faces/two mips before/after owner invalidation passed\n";
            std::cout << "Public surface fixture replacement only; no manager/kernel injection,"
                         " app/window/ROM/emulation or image decoding started\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
