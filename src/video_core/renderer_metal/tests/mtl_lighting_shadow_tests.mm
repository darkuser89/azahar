// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA attributes/registers -> lighting shadow texture binding -> VRAM.
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
constexpr std::array<unsigned, 2> Physical{2, 7};
constexpr std::array<std::array<unsigned, 3>, 2> Diffuse{{{90,60,120}, {120,100,55}}};
constexpr std::array<std::array<unsigned, 3>, 2> Ambient{{{12,20,8}, {9,7,15}}};
constexpr std::array<std::array<unsigned, 3>, 2> Spec0{{{110,75,45}, {75,95,105}}};
constexpr std::array<std::array<unsigned, 3>, 2> Spec1{{{25,55,90}, {40,35,25}}};
constexpr std::array<unsigned, 3> Global{13,21,29};
constexpr std::array<std::array<float, 2>, 4> Probes{{
    {.1875f,.3125f}, {.5f,.5f}, {.03125f,.96875f}, {.90625f,.09375f}}};
void Color(Pica::LightingRegs::LightColor& color, const std::array<unsigned,3>& rgb) {
    color.r.Assign(rgb[0]); color.g.Assign(rgb[1]); color.b.Assign(rgb[2]);
}
unsigned PixelOffset(unsigned x, unsigned y) {
    unsigned offset{};
    for (unsigned bit = 0; bit < 3; ++bit)
        offset |= ((x >> bit) & 1) << (2 * bit) | ((y >> bit) & 1) << (2 * bit + 1);
    return offset;
}
std::array<unsigned,4> Texel(unsigned unit, unsigned x, unsigned y, unsigned revision) {
    return {32+unit*31+x*9+y*5+revision*17, 35+unit*17+x*3+y*13+revision*19,
            47+unit*23+x*11+y*2+revision*7, 61+unit*29+x*7+y*9+revision*23};
}
std::array<float,2> Coordinate(unsigned unit, unsigned probe) {
    const auto [u,v] = Probes[probe];
    return unit == 0 ? std::array{u,v} : unit == 1 ? std::array{v,1-u} : std::array{1-u,1-v};
}
// Scalar texture oracle: guest RGBA8 tiles have bottom-up rows, the ordinary
// native sampled image has top-down rows. No production codec or readback used.
std::array<double,4> Sample(unsigned unit, unsigned probe, bool linear, unsigned revision) {
    const auto [u,v] = Coordinate(unit, probe);
    const double px = u * 8 - (linear ? .5 : 0), py = v * 8 - (linear ? .5 : 0);
    const int x = int(std::floor(px)), y = int(std::floor(py));
    std::array<double,4> result{};
    for (unsigned j = 0; j < (linear ? 2U : 1U); ++j)
        for (unsigned i = 0; i < (linear ? 2U : 1U); ++i) {
            const double weight = linear ? (i ? px-x : 1-(px-x)) * (j ? py-y : 1-(py-y)) : 1;
            const auto value = Texel(unit, unsigned(std::clamp(x+int(i),0,7)),
                                    unsigned(7-std::clamp(y+int(j),0,7)), revision);
            for (unsigned c = 0; c < 4; ++c) result[c] += value[c] / 255.0 * weight;
        }
    return result;
}
// Flags: shadow globally enabled, primary RGB, secondary RGB, inversion, alpha.
std::array<unsigned,4> Expected(unsigned unit, unsigned flags, unsigned mask, bool secondary,
                                 bool linear, unsigned revision) {
    auto shadow = Sample(unit, flags % Probes.size(), linear, revision);
    if (flags & 8) for (auto& value : shadow) value = 1-value;
    std::array<unsigned,4> result{};
    for (unsigned c = 0; c < 3; ++c) {
        double sum = secondary ? 0 : Global[c] / 255.0;
        for (unsigned slot = 0; slot < 2; ++slot) {
            const double attenuation = (flags & 1) && (mask & (1U << slot)) &&
                (flags & (secondary ? 4U : 2U)) ? shadow[c] : 1;
            const double cosine = slot ? 1 / std::sqrt(2.0) : 1;
            sum += secondary ? (Spec0[slot][c]+Spec1[slot][c]) / 255.0 * attenuation
                : (Diffuse[slot][c]*cosine*attenuation+Ambient[slot][c]) / 255.0;
        }
        result[c] = unsigned(std::lround(std::clamp(sum,0.0,1.0)*255));
    }
    // Alpha is attenuated once after all lights, independently of per-light
    // shadow masks and primary/secondary RGB flags, when Fresnel is enabled.
    const double alpha = 3071 / 4095.0 * ((flags & 1) && (flags & 16) ? shadow[3] : 1);
    result[3] = unsigned(std::lround(alpha*255));
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
            Settings::values.use_disk_shader_cache.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory,pica,custom,renderer,device,scheduler};
            renderer.rasterizer = &rasterizer; pica.BindRasterizer(&rasterizer);
            auto& regs = pica.regs.internal;
            regs = {}; Metal::Test::MovVertex(regs,pica.vs_setup);
            regs.vs.output_mask.Assign(63); regs.rasterizer.vs_output_total.Assign(6);
            regs.rasterizer.vs_output_attributes[3].raw = 0x1f141312; // View XYZ.
            regs.rasterizer.vs_output_attributes[4].raw = 0x0f0e0d0c; // Texcoord0/1.
            regs.rasterizer.vs_output_attributes[5].raw = 0x1f1f1716; // Texcoord2.
            for (unsigned i = 3; i < 6; ++i)
                pica.vs_setup.UpdateProgramCode(i,Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,i,i));
            pica.vs_setup.UpdateProgramCode(6,u32(nihstro::OpCode::Id::END)<<26);
            regs.vs.max_input_attribute_index.Assign(5); regs.pipeline.max_input_attrib_index.Assign(5);
            regs.vs.input_attribute_to_register_map_low = 0x543210;
            regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask = 15<<8;
            regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR/8);
            regs.framebuffer.framebuffer.width.Assign(32); regs.framebuffer.framebuffer.height.Assign(31);
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.viewport_size_x.Assign(67<<16); regs.rasterizer.viewport_size_y.Assign(67<<16);
            regs.rasterizer.viewport_depth_near_plane.Assign(62<<16);
            Color(regs.lighting.global_ambient,Global);
            for (unsigned slot = 0; slot < 2; ++slot) {
                auto& light = regs.lighting.light[Physical[slot]];
                Color(light.diffuse,Diffuse[slot]); Color(light.ambient,Ambient[slot]);
                Color(light.specular_0,Spec0[slot]); Color(light.specular_1,Spec1[slot]);
                light.z.Assign(15<<10); light.y.Assign(slot ? 15<<10 : 0);
                light.config.directional.Assign(1); light.config.two_sided_diffuse.Assign(1);
            }
            using C = Pica::TexturingRegs::TextureConfig;
            using T = Pica::TexturingRegs::TevStageConfig;
            regs.texturing.main_config.texture0_enable.Assign(1);
            regs.texturing.main_config.texture1_enable.Assign(1);
            regs.texturing.main_config.texture2_enable.Assign(1);
            regs.texturing.texture0_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
            regs.texturing.texture1_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
            regs.texturing.texture2_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
            const std::array<C*,3> textures{&regs.texturing.texture0,&regs.texturing.texture1,&regs.texturing.texture2};
            std::array<PAddr,3> addresses{};
            for (unsigned unit = 0; unit < 3; ++unit) {
                addresses[unit] = Memory::VRAM_PADDR+0x120000+unit*0x1000;
                textures[unit]->width.Assign(8); textures[unit]->height.Assign(8);
                textures[unit]->address.Assign(addresses[unit]/8);
                textures[unit]->wrap_s.Assign(C::ClampToEdge); textures[unit]->wrap_t.Assign(C::ClampToEdge);
            }
            const std::array<T*,6> stages{&regs.texturing.tev_stage0,&regs.texturing.tev_stage1,
                &regs.texturing.tev_stage2,&regs.texturing.tev_stage3,&regs.texturing.tev_stage4,&regs.texturing.tev_stage5};
            for (auto* stage : stages) stage->sources_raw = 0xf000f;
            pica.dirty_regs.SetAllDirty();
            const auto process = [&](const std::vector<u32>& words) {
                auto backing = memory.GetPhysicalRef(Memory::VRAM_PADDR+0x8000);
                const auto bytes = words.size()*sizeof(u32);
                Check(backing && backing.GetSize() >= bytes,"Invalid packed command backing");
                std::memcpy(backing.GetPtr(),words.data(),bytes);
                pica.ProcessCmdList(Memory::VRAM_PADDR+0x8000,u32(bytes),false);
            };
            std::vector<u32> lut;
            for (unsigned table = 0; table < 24; ++table) {
                Write(lut,PICA_REG_INDEX(lighting.lut_config),table<<8);
                for (unsigned i = 0; i < 256; ++i)
                    Write(lut,PICA_REG_INDEX(lighting.lut_data[0])+i%8,table == 3 ? 3071 : 0);
            }
            process(lut);
            auto target = memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize() >= 4096,"Missing physical target VRAM");
            std::memset(target.GetPtr(),0x11,4096);
            constexpr std::array<std::array<float,4>,3> positions{{
                {-1,-1,-.5f,1},{3,-1,-.5f,1},{-1,3,-.5f,1}}};
            unsigned draws{}, mutation_sensitive{}, filter_sensitive{}, mask_sensitive{}, alpha_sensitive{};
            for (unsigned revision = 0; revision < 2; ++revision) {
                for (unsigned unit = 0; unit < 3; ++unit) {
                    auto backing = memory.GetPhysicalRef(addresses[unit]);
                    Check(backing && backing.GetSize() >= 256,"Missing guest shadow texture VRAM");
                    rasterizer.InvalidateRegion(addresses[unit],256);
                    for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x) {
                        const auto rgba = Texel(unit,x,y,revision);
                        for (unsigned c = 0; c < 4; ++c) backing.GetPtr()[PixelOffset(x,y)*4+3-c] = u8(rgba[c]);
                    }
                }
                for (unsigned unit = 0; unit < 3; ++unit) for (unsigned linear = 0; linear < 2; ++linear) {
                    for (auto* texture : textures) {
                        texture->min_filter.Assign(C::TextureFilter(linear));
                        texture->mag_filter.Assign(C::TextureFilter(linear));
                    }
                    pica.dirty_regs.SetAllDirty();
                    for (unsigned flags = 0; flags < 32; ++flags) for (unsigned mask = 0; mask < 4; ++mask)
                        for (unsigned secondary = 0; secondary < 2; ++secondary) {
                            const auto source = secondary ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor;
                            std::vector<u32> setup;
                            Write(setup,PICA_REG_INDEX(lighting.max_light_index),1);
                            Write(setup,PICA_REG_INDEX(lighting.light_enable),Physical[0] | (Physical[1]<<4));
                            Write(setup,PICA_REG_INDEX(lighting.config0),(flags&1) | 12 | (8<<4) |
                                ((flags&2)<<15) | ((flags&4)<<15) | ((flags&8)<<15) | ((flags&16)<<15) | (unit<<24));
                            const unsigned disabled = 255 ^ ((mask&1 ? 1U<<Physical[0] : 0) | (mask&2 ? 1U<<Physical[1] : 0));
                            Write(setup,PICA_REG_INDEX(lighting.config1),0xfff7ff00 | disabled);
                            Write(setup,PICA_REG_INDEX(texturing.tev_stage0),u32(source) | (u32(source)<<16));
                            process(setup);
                            const auto expected = Expected(unit,flags,mask,secondary,linear,revision);
                            for (unsigned c = 0; c < 4; ++c) {
                                if (revision) mutation_sensitive += std::abs(int(expected[c])-int(Expected(unit,flags,mask,secondary,linear,0)[c])) > 2;
                                filter_sensitive += std::abs(int(expected[c])-int(Expected(unit,flags,mask,secondary,!linear,revision)[c])) > 2;
                                mask_sensitive += std::abs(int(expected[c])-int(Expected(unit,flags,mask^3,secondary,linear,revision)[c])) > 2;
                            }
                            alpha_sensitive += std::abs(int(expected[3])-int(Expected(unit,flags^16,mask,secondary,linear,revision)[3])) > 2;
                            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                                std::vector<u32> triangle;
                                Write(triangle,PICA_REG_INDEX(pipeline.triangle_topology),0);
                                Write(triangle,PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index),15);
                                const auto a = Coordinate(0,flags%4), b = Coordinate(1,flags%4), c = Coordinate(2,flags%4);
                                for (const auto& position : positions) {
                                    Attribute(triangle,position); Attribute(triangle,{0,0,0,1});
                                    Attribute(triangle,{1,1,1,1}); Attribute(triangle,{0,0,1,0});
                                    Attribute(triangle,{a[0],a[1],b[0],b[1]}); Attribute(triangle,{c[0],c[1],0,0});
                                }
                                process(triangle); rasterizer.FlushRegion(Memory::VRAM_PADDR,4096);
                                for (unsigned pixel = 0; pixel < 1024; ++pixel) for (unsigned c = 0; c < 4; ++c) {
                                    const auto actual = target.GetPtr()[pixel*4+3-c];
                                    if (std::abs(int(actual)-int(expected[c])) > 2) {
                                        std::cerr << "revision=" << revision << " unit=" << unit << " linear=" << linear
                                            << " flags=" << flags << " mask=" << mask << " secondary=" << secondary
                                            << " repeat=" << repeat << " pixel=" << pixel << " channel=" << c
                                            << " actual=" << unsigned(actual) << " expected=" << expected[c] << '\n';
                                        throw std::runtime_error("Integrated lighting shadow/TEV/VRAM mismatch");
                                    }
                                }
                                ++draws;
                            }
                        }
                }
            }
            Check(draws == 6144 && mutation_sensitive && filter_sensitive && mask_sensitive && alpha_sensitive,
                  "Missing lighting shadow mutation/filter/mask/alpha coverage");
            Check(!system.IsPoweredOn(),"Data harness initialized emulation");
            std::cout << draws << " integrated lighting shadow/TEV/physical VRAM draws; all 1024 RGBA pixels; tolerance 2\n"
                << "Oracle coverage: mutation=" << mutation_sensitive << " filter=" << filter_sensitive
                << " light-mask=" << mask_sensitive << " alpha=" << alpha_sensitive << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n'; return 1;
        }
    }
}
