// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA commands -> raw LUTs -> RasterizerMetal -> physical guest VRAM.
// Core data objects only: no OS window, emulator initialization, ROM or title.
#include "mtl_rasterizer.h"
#include "mtl_pica_vertex_fixture.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/image_interface.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void Write(std::vector<u32>& words, u32 reg, u32 value) {
    words.insert(words.end(), {value, reg | (15U << 16)});
}
u32 Float24(float value) {
    if (value == 0) return 0;
    const auto bits = std::bit_cast<u32>(value);
    const auto raw = ((bits >> 31) << 23) | ((((bits >> 23) & 255) - 64) << 16) |
                     ((bits & 0x7fffff) >> 7);
    Check(Pica::f24::FromRaw(raw).ToFloat32() == value, "Inexact f24 fixture");
    return raw;
}
void Attribute(std::vector<u32>& words, const std::array<float, 4>& value) {
    const auto x = Float24(value[0]), y = Float24(value[1]);
    const auto z = Float24(value[2]), w = Float24(value[3]);
    words.insert(words.end(), {(w << 8) | (z >> 16),
        u32(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0])) |
            (15U << 16) | (2U << 20) | (1U << 31),
        (z << 16) | (y >> 8), (y << 24) | x});
}
// Deliberately independent base/delta fields: omitting or changing the sign of
// interpolation must be observable. These are raw register fixtures, not a
// claim that their slopes describe a continuous real-world material function.
u32 Entry(unsigned table, unsigned index, unsigned revision) {
    const auto value = 1400 + (table * 113 + index * 17 + revision * 379) % 1200;
    const auto difference = 320 + (table * 29 + index * 7 + revision * 97) % 320;
    return value | (difference << 12) | (((table + index + revision) & 1) << 23);
}
// Independent scalar decoding of raw fixed-point fields, using the LUT address
// contract. Do not call LutEntry::ToFloat/DiffToFloat, generator lookup helpers,
// SyncLUTs, or read the GPU LUT resource to obtain expected values.
double Lookup(unsigned table, double input, bool unsigned_domain, unsigned revision,
              bool use_difference = true) {
    const double position = input * (unsigned_domain ? 256 : 128);
    const int cell = std::clamp(int(std::floor(position)), unsigned_domain ? 0 : -128,
                                unsigned_domain ? 255 : 127);
    const auto raw = Entry(table, unsigned(cell < 0 ? cell + 256 : cell), revision);
    const double base = (raw & 4095) / 4095.0;
    const double delta = ((raw >> 12) & 2047) / 2047.0 * ((raw & (1U << 23)) ? -1 : 1);
    return base + (use_difference ? delta * (position - cell) : 0);
}
constexpr std::array<unsigned, 3> Diffuse{100, 140, 180}, Ambient{15, 25, 35};
constexpr std::array<unsigned, 3> Spec0{170, 100, 140}, Spec1{120, 180, 80};
constexpr std::array<unsigned, 3> Global{10, 20, 30};
void Color(Pica::LightingRegs::LightColor& color, const std::array<unsigned, 3>& rgb) {
    color.r.Assign(rgb[0]); color.g.Assign(rgb[1]); color.b.Assign(rgb[2]);
}
std::array<unsigned, 4> Expected(unsigned light, bool unsigned_domain, bool reverse,
                                 bool secondary, unsigned revision, bool use_difference = true) {
    const double cosine = 1 / std::sqrt(3.0);
    // Every selected physical light has two-sided diffuse enabled. Its active
    // slot therefore uses abs(LN) for an unsigned LUT, regardless of the physical
    // ID. SP/DA table addresses below still belong to that physical light.
    const double input = reverse && !unsigned_domain ? -cosine : cosine;
    const auto lut = [&](unsigned table) {
        return Lookup(table, input, unsigned_domain, revision, use_difference);
    };
    const double attenuation = lut(8 + light) * Lookup(16 + light, std::sqrt(3.0) / 4 + .125,
                                                     true, revision, use_difference);
    const std::array<unsigned, 3> reflect{6, 5, 4};
    std::array<unsigned, 4> result{};
    for (unsigned c = 0; c < 3; ++c) {
        const double value = secondary
            ? (Spec0[c] * lut(0) + Spec1[c] * lut(1) * lut(reflect[c])) / 255 * attenuation
            : ((Diffuse[c] * cosine + Ambient[c]) * attenuation + Global[c]) / 255;
        result[c] = unsigned(std::lround(std::clamp(value, 0.0, 1.0) * 255));
    }
    result[3] = unsigned(std::lround(std::clamp(lut(3), 0.0, 1.0) * 255));
    return result;
}
class Window final : public Frontend::EmuWindow {
public:
    void PollEvents() override {}
};
class Renderer final : public VideoCore::RendererBase {
public:
    Renderer(Core::System& system, Window& window) : RendererBase{system, window, nullptr} {}
    VideoCore::RasterizerInterface* Rasterizer() override { return rasterizer; }
    void SwapBuffers() override {}
    void TryPresent(int, bool) override {}
    VideoCore::RasterizerInterface* rasterizer{};
};
} // namespace

int main() {
    @autoreleasepool {
        try {
            Core::System system;
            system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());
            Memory::MemorySystem memory{system};
            Pica::PicaCore pica{memory, nullptr};
            Metal::Test::InitializeFixtureUniforms(pica.vs_setup);
            Metal::Test::InitializeFixtureUniforms(pica.gs_setup);
            Window window;
            Renderer renderer{system, window};
            VideoCore::CustomTexManager custom{system};
            Settings::values.resolution_factor.SetValue(1);
            Settings::values.dump_textures.SetValue(false);
            Settings::values.custom_textures.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory, pica, custom, renderer, device, scheduler};
            renderer.rasterizer = &rasterizer;
            pica.BindRasterizer(&rasterizer);
            auto& regs = pica.regs.internal;
            regs = {};
            Metal::Test::MovVertex(regs, pica.vs_setup);
            regs.vs.output_mask.Assign(15);
            regs.rasterizer.vs_output_total.Assign(4);
            regs.rasterizer.vs_output_attributes[3].raw = 0x1f141312; // View XYZ, unused W.
            pica.vs_setup.UpdateProgramCode(3, Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV, 3, 3));
            pica.vs_setup.UpdateProgramCode(4, u32(nihstro::OpCode::Id::END) << 26);
            regs.vs.max_input_attribute_index.Assign(3);
            regs.pipeline.max_input_attrib_index.Assign(3);
            regs.vs.input_attribute_to_register_map_low = 0x3210;
            regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
            regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR / 8);
            regs.framebuffer.framebuffer.width.Assign(32);
            regs.framebuffer.framebuffer.height.Assign(31);
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.viewport_size_x.Assign(67 << 16);
            regs.rasterizer.viewport_size_y.Assign(67 << 16);
            regs.rasterizer.viewport_depth_near_plane.Assign(62 << 16);
            Color(regs.lighting.global_ambient, Global);
            for (auto& light : regs.lighting.light) {
                Color(light.diffuse, Diffuse); Color(light.ambient, Ambient);
                Color(light.specular_0, Spec0); Color(light.specular_1, Spec1);
                light.x.Assign(15 << 10); light.y.Assign(15 << 10); light.z.Assign(15 << 10);
                light.config.directional.Assign(1); light.config.two_sided_diffuse.Assign(1);
                light.dist_atten_bias.Assign(60 << 12); // f20 0.125.
                light.dist_atten_scale.Assign(61 << 12); // f20 0.25.
            }
            pica.dirty_regs.SetAllDirty();
            auto target = memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize() >= 4096, "Missing physical VRAM");
            std::memset(target.GetPtr(), 0x11, 4096);
            const auto process = [&](const std::vector<u32>& words) {
                const auto bytes = words.size() * sizeof(u32);
                auto backing = memory.GetPhysicalRef(Memory::VRAM_PADDR + 0x8000);
                Check(backing && backing.GetSize() >= bytes, "Invalid command backing");
                std::memcpy(backing.GetPtr(), words.data(), bytes);
                pica.ProcessCmdList(Memory::VRAM_PADDR + 0x8000, u32(bytes), false);
            };
            constexpr std::array<std::array<float, 4>, 3> positions{{
                {-1,-1,-.5f,1}, {3,-1,-.5f,1}, {-1,3,-.5f,1}}};
            const auto triangle=[&](bool reverse) {
                std::vector<u32> words;
                Write(words, PICA_REG_INDEX(pipeline.triangle_topology), 0);
                Write(words, PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index), 15);
                for (const auto& position : positions) {
                    Attribute(words, position);
                    Attribute(words, reverse ? std::array<float,4>{1,0,0,0}
                                             : std::array<float,4>{0,0,0,1});
                    Attribute(words, {1,1,1,1});
                    Attribute(words, {0,0,1,0});
                }
                process(words);
            };
            unsigned draws{}, raw_entries{}, sensitive_channels{}, changed_channels{}, dormant_draws{};
            for (unsigned revision = 0; revision < 3; ++revision) {
                std::vector<u32> upload;
                const unsigned start = std::array<unsigned, 3>{239, 129, 255}[revision];
                for (unsigned table = 0; table < 24; ++table) {
                    Write(upload, PICA_REG_INDEX(lighting.lut_config), (table << 8) | start);
                    for (unsigned step = 0; step < 256; ++step) {
                        const auto index = (start + step) % 256;
                        Write(upload, PICA_REG_INDEX(lighting.lut_data[0]) + step % 8,
                              Entry(table, index, revision));
                    }
                }
                process(upload);
                Check(pica.lighting.lut_dirty == 0xffffff, "LUT commands missed dirty tables");
                Check(regs.lighting.lut_config.index == start && regs.lighting.lut_config.type == 23,
                      "PICA LUT index did not auto-increment and wrap");
                for (unsigned table = 0; table < 24; ++table) for (unsigned index = 0; index < 256; ++index) {
                    Check(pica.lighting.luts[table][index].raw == Entry(table, index, revision),
                          "Public command LUT raw value differs");
                    ++raw_entries;
                }
                std::vector<u32> dormant;
                Write(dormant, PICA_REG_INDEX(lighting.disable), 1);
                const std::array<unsigned,6> tev{PICA_REG_INDEX(texturing.tev_stage0),
                    PICA_REG_INDEX(texturing.tev_stage1), PICA_REG_INDEX(texturing.tev_stage2),
                    PICA_REG_INDEX(texturing.tev_stage3), PICA_REG_INDEX(texturing.tev_stage4),
                    PICA_REG_INDEX(texturing.tev_stage5)};
                for (unsigned i=0;i<tev.size();++i) {
                    Write(dormant,tev[i],i ? 0xf000f:0);
                    Write(dormant,tev[i]+1,0); Write(dormant,tev[i]+2,0); Write(dormant,tev[i]+4,0);
                }
                process(dormant);
                triangle(false);
                rasterizer.FlushRegion(Memory::VRAM_PADDR,4096);
                Check(pica.lighting.lut_dirty==0xffffff,"Inactive shader consumed lighting LUT writes");
                for (unsigned byte=0;byte<4096;++byte)
                    Check(target.GetPtr()[byte]==255,"Inactive lighting draw changed primary color");
                ++dormant_draws;
                for (unsigned physical = 0; physical < 8; ++physical)
                    for (unsigned domain = 0; domain < 2; ++domain)
                        for (unsigned reverse = 0; reverse < 2; ++reverse)
                            for (unsigned secondary = 0; secondary < 2; ++secondary) {
                                using T = Pica::TexturingRegs::TevStageConfig;
                                const auto source = secondary ? T::Source::SecondaryFragmentColor
                                                              : T::Source::PrimaryFragmentColor;
                                std::vector<u32> setup;
                                Write(setup, PICA_REG_INDEX(lighting.disable), 0);
                                Write(setup, PICA_REG_INDEX(lighting.max_light_index), 0);
                                Write(setup, PICA_REG_INDEX(lighting.light_enable), physical);
                                Write(setup, PICA_REG_INDEX(lighting.config0), (8 << 4) | 12);
                                Write(setup, PICA_REG_INDEX(lighting.config1), 255 | (1 << 18));
                                Write(setup, PICA_REG_INDEX(lighting.abs_lut_input), domain ? 0 : 0x02222222);
                                Write(setup, PICA_REG_INDEX(lighting.lut_input), 0x03333333); // LN for every LUT.
                                Write(setup, PICA_REG_INDEX(lighting.lut_scale), 0);
                                const std::array<unsigned, 6> tev{PICA_REG_INDEX(texturing.tev_stage0),
                                    PICA_REG_INDEX(texturing.tev_stage1), PICA_REG_INDEX(texturing.tev_stage2),
                                    PICA_REG_INDEX(texturing.tev_stage3), PICA_REG_INDEX(texturing.tev_stage4),
                                    PICA_REG_INDEX(texturing.tev_stage5)};
                                for (unsigned i = 0; i < tev.size(); ++i) {
                                    Write(setup, tev[i], i ? 0xf000f : u32(source) | (u32(source) << 16));
                                    Write(setup, tev[i] + 1, 0); Write(setup, tev[i] + 2, 0);
                                    Write(setup, tev[i] + 4, 0);
                                }
                                process(setup);
                                const Pica::Shader::FSConfig config{regs};
                                Check(config.lighting.lights[0].two_sided_diffuse == 1 &&
                                      (physical == 0 || config.lighting.lights[physical].two_sided_diffuse == 0),
                                      "Fixture no longer distinguishes active slots from physical light IDs");
                                const auto expected = Expected(physical, domain, reverse, secondary, revision);
                                const auto no_delta = Expected(physical, domain, reverse, secondary, revision, false);
                                for (unsigned c = 0; c < 4; ++c) {
                                    sensitive_channels += std::abs(int(expected[c]) - int(no_delta[c])) > 4;
                                    if (revision) changed_channels += std::abs(int(expected[c]) - int(
                                        Expected(physical, domain, reverse, secondary, revision - 1)[c])) > 4;
                                }
                                for (unsigned repeat = 0; repeat < 2; ++repeat) {
                                    // No ClearAll: the second draw reuses the LUT resources;
                                    // later revisions must invalidate them through real dirty flags.
                                    triangle(reverse);
                                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 4096);
                                    Check(pica.lighting.lut_dirty == 0, "Native draw did not consume LUT dirty flags");
                                    for (unsigned pixel = 0; pixel < 1024; ++pixel) for (unsigned c = 0; c < 4; ++c) {
                                        const auto actual = target.GetPtr()[pixel * 4 + 3 - c];
                                        if (std::abs(int(actual) - int(expected[c])) > 2) {
                                            std::cerr << "revision=" << revision << " physical=" << physical
                                                << " domain=" << domain << " reverse=" << reverse
                                                << " secondary=" << secondary << " repeat=" << repeat
                                                << " pixel=" << pixel << " channel=" << c
                                                << " actual=" << unsigned(actual) << " expected=" << expected[c] << '\n';
                                            throw std::runtime_error("Integrated raw LUT/lighting/TEV/VRAM mismatch");
                                        }
                                    }
                                    ++draws;
                                }
                            }
            }
            Check(draws == 384 && raw_entries == 18432 && dormant_draws==3 && sensitive_channels && changed_channels,
                  "Missing raw LUT, interpolation, mutation or reuse coverage");
            Check(!system.IsPoweredOn(), "Data-only harness initialized emulation");
            std::cout << draws << " integrated raw PICA LUT/native lighting/TEV/physical VRAM draws; "
                      << raw_entries << " raw entries; all 1024 RGBA pixels; tolerance 2\n"
                      << "Oracle coverage: interpolation-sensitive channels=" << sensitive_channels
                      << " mutation-sensitive channels=" << changed_channels << '\n';
            std::cout<<dormant_draws<<" inactive lighting draws preserve raw LUT writes before activation\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
