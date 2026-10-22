// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA attributes/registers -> raw Fog/Proctex LUT uploads -> VRAM.
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
#include <fstream>
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
using T = Pica::TexturingRegs;
constexpr std::array<unsigned,3> FogColor{31,149,211};
constexpr std::array<unsigned,4> Primary{191,64,128,159};
constexpr std::array<unsigned,4> Offsets{7,135,199,231};
constexpr std::array<std::array<float,2>,4> Probes{{
    {.12109375f,.31640625f},{.49609375f,.71484375f},{.00390625f,.99609375f},{.875f,.0625f}}};
int Signed(unsigned bits, unsigned width) {
    const auto sign = 1U << (width-1);
    return int(bits & (sign-1)) - int(bits & sign);
}
u32 FogEntry(unsigned index, unsigned revision) {
    const unsigned value = 200 + (index*53+revision*271)%1700;
    const int difference = -1100 + int((index*71+revision*149)%2200);
    return (value<<13) | (unsigned(difference)&8191);
}
u32 ValueEntry(unsigned table, unsigned index, unsigned revision) {
    if (table == 0) return (revision+1)*1024; // Constant noise interpolation map.
    const unsigned value = 500 + (index*29+table*173+revision*257)%3000;
    const int difference = -1000 + int((index*53+table*97+revision*137)%2000);
    return value | ((unsigned(difference)&4095)<<12);
}
u32 ColorEntry(unsigned index, unsigned revision, bool difference) {
    u32 raw{};
    for (unsigned c = 0; c < 4; ++c) {
        const int value = difference ? -45+int((index*11+c*23+revision*13)%90)
                                     : 60+int((index*19+c*37+revision*17)%130);
        raw |= (unsigned(value)&255) << (c*8);
    }
    return raw;
}
double FogFactor(double depth, bool flip, unsigned revision, bool difference = true) {
    const double position = (flip ? 1-depth : depth)*128;
    const int index = std::clamp(int(std::floor(position)),0,127);
    const auto raw = FogEntry(unsigned(index),revision);
    return std::clamp(((raw>>13)&2047)/2047.0 + (difference ?
        Signed(raw&8191,13)/2047.0*(position-index) : 0),0.0,1.0);
}
std::array<unsigned,4> FogExpected(double depth, bool flip, unsigned revision, bool difference = true) {
    const double factor = FogFactor(depth,flip,revision,difference);
    std::array<unsigned,4> rgba{};
    for (unsigned c = 0; c < 3; ++c)
        rgba[c] = unsigned(std::lround(FogColor[c]*(1-factor)+Primary[c]*factor));
    rgba[3] = Primary[3];
    return rgba;
}
double Map(unsigned table, double input, unsigned revision, bool difference = true) {
    const double position = input*128;
    const int index = std::clamp(int(std::floor(position)),0,127);
    const auto raw = ValueEntry(table,unsigned(index),revision);
    return std::clamp((raw&4095)/4095.0 + (difference ?
        Signed((raw>>12)&4095,12)/4095.0*(position-index) : 0),0.0,1.0);
}
double Combine(unsigned mode, double u, double v) {
    switch (mode) {
    case 0: return u;
    case 1: return u*u;
    case 2: return v;
    case 3: return v*v;
    case 4: return (u+v)/2;
    case 5: return (u*u+v*v)/2;
    case 6: return std::min(std::sqrt(u*u+v*v),1.0);
    case 7: return std::min(u,v);
    case 8: return std::max(u,v);
    // Current generator contract averages (u+v)/2 and sqrt, despite the
    // abbreviated public enum comment mentioning Max. No hardware claim.
    case 9: return std::min(((u+v)/2+std::sqrt(u*u+v*v))/2,1.0);
    }
    throw std::runtime_error("Invalid combiner fixture");
}
std::array<float,2> Coordinate(unsigned coord, unsigned probe) {
    const auto [u,v] = Probes[probe];
    return coord == 0 ? std::array{u,v} : coord == 1 ? std::array{v,1-u} : std::array{1-u,1-v};
}
std::array<unsigned,4> ProcExpected(unsigned combiner, unsigned filter, bool separate, bool noise,
                                    unsigned probe, unsigned revision, bool map_difference = true,
                                    bool color_difference = true, bool offsets = true,
                                    const std::array<unsigned,6>* table_revisions = nullptr) {
    const auto table_revision = [&](unsigned table) {
        return table_revisions ? (*table_revisions)[table] : revision;
    };
    const auto uv = Coordinate(combiner%3,probe);
    double u = uv[0], v = uv[1];
    if (noise) {
        // Frequency=phase=0 fixes the noise grid at its origin. The four
        // frozen-generator corner slopes yield g1+g2=0 and g3=6/5, so a
        // constant noise-map value t gives coefficient 6*t*t/5. This scalar
        // reduction does not duplicate the shader's hash/interpolation code;
        // general noise-frequency/phase semantics remain separate.
        const double t = (table_revision(0)+1)*1024/4095.0;
        const double coefficient = 6*t*t/5;
        // Public uniform contract normalizes signed amplitude by 4095.
        u = std::abs(u+(1024/4095.0)*coefficient);
        v = std::abs(v-(512/4095.0)*coefficient);
    }
    u = std::min(u,1.0); v = std::min(v,1.0);
    const double mapped = Map(2,Combine(combiner,u,v),table_revision(2),map_difference);
    const unsigned level = filter/2, width = 128>>level;
    const double position = mapped*(width-1);
    const bool linear = filter&1;
    const unsigned cell = unsigned(linear ? std::floor(position) : std::floor(position+.5));
    const unsigned index = cell+(offsets ? Offsets[level] : 0);
    Check(index < 256,"Scalar color LUT address overflow");
    const auto raw = ColorEntry(index,table_revision(4),false);
    const auto delta = ColorEntry(index,table_revision(5),true);
    std::array<unsigned,4> rgba{};
    for (unsigned c = 0; c < 4; ++c) {
        const double value = ((raw>>(8*c))&255) + (linear && color_difference ?
            2*Signed((delta>>(8*c))&255,8)*(position-std::floor(position)) : 0);
        rgba[c] = unsigned(std::lround(std::clamp(value,0.0,255.0)));
    }
    if (separate) rgba[3] = unsigned(std::lround(Map(3,Combine((combiner+3)%10,u,v),table_revision(3),map_difference)*255));
    return rgba;
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

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Core::System system;
            system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());
            Memory::MemorySystem memory{system}; Pica::PicaCore pica{memory,nullptr};
            Metal::Test::InitializeFixtureUniforms(pica.vs_setup);
            Metal::Test::InitializeFixtureUniforms(pica.gs_setup);
            Window window; Renderer renderer{system,window}; VideoCore::CustomTexManager custom{system};
            Settings::values.resolution_factor.SetValue(1);
            Settings::values.dump_textures.SetValue(false); Settings::values.custom_textures.SetValue(false);
            Settings::values.use_disk_shader_cache.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device; Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory,pica,custom,renderer,device,scheduler};
            renderer.rasterizer=&rasterizer; pica.BindRasterizer(&rasterizer);
            auto& regs=pica.regs.internal; regs={}; Metal::Test::MovVertex(regs,pica.vs_setup);
            regs.lighting.disable.Assign(1);
            regs.vs.output_mask.Assign(31); regs.rasterizer.vs_output_total.Assign(5);
            regs.rasterizer.vs_output_attributes[3].raw=0x0f0e0d0c;
            regs.rasterizer.vs_output_attributes[4].raw=0x1f1f1716;
            for (unsigned i=3;i<5;++i)
                pica.vs_setup.UpdateProgramCode(i,Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,i,i));
            pica.vs_setup.UpdateProgramCode(5,u32(nihstro::OpCode::Id::END)<<26);
            regs.vs.max_input_attribute_index.Assign(4); regs.pipeline.max_input_attrib_index.Assign(4);
            regs.vs.input_attribute_to_register_map_low=0x43210;
            regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask=15<<8;
            regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR/8);
            regs.framebuffer.framebuffer.width.Assign(32); regs.framebuffer.framebuffer.height.Assign(31);
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.viewport_size_x.Assign(67<<16); regs.rasterizer.viewport_size_y.Assign(67<<16);
            regs.rasterizer.viewport_depth_near_plane.Assign(62<<16);
            regs.texturing.fog_color.raw=FogColor[0] | (FogColor[1]<<8) | (FogColor[2]<<16);
            const std::array<T::TevStageConfig*,6> stages{&regs.texturing.tev_stage0,&regs.texturing.tev_stage1,
                &regs.texturing.tev_stage2,&regs.texturing.tev_stage3,&regs.texturing.tev_stage4,&regs.texturing.tev_stage5};
            for (auto* stage:stages) stage->sources_raw=0xf000f;
            pica.dirty_regs.SetAllDirty();
            const auto process=[&](const std::vector<u32>& words) {
                auto backing=memory.GetPhysicalRef(Memory::VRAM_PADDR+0x8000);
                const auto bytes=words.size()*sizeof(u32);
                Check(backing && backing.GetSize()>=bytes,"Invalid command backing");
                std::memcpy(backing.GetPtr(),words.data(),bytes);
                pica.ProcessCmdList(Memory::VRAM_PADDR+0x8000,u32(bytes),false);
            };
            // Initialize otherwise unused lighting storage through public commands too.
            std::vector<u32> init;
            for (unsigned table=0;table<24;++table) {
                Write(init,PICA_REG_INDEX(lighting.lut_config),table<<8);
                for (unsigned i=0;i<256;++i) Write(init,PICA_REG_INDEX(lighting.lut_data[0])+i%8,0);
            }
            process(init);
            auto target=memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize()>=4096,"Missing physical target VRAM");
            std::memset(target.GetPtr(),0x11,4096);
            constexpr std::array<std::array<float,4>,3> positions{{
                {-1,-1,-.5f,1},{3,-1,-.5f,1},{-1,3,-.5f,1}}};
            unsigned fog_draws{},proc_draws{},raw_entries{},exports{},dormant_draws{};
            unsigned fog_sensitive{},fog_clamped_low{},fog_clamped_high{};
            unsigned map_sensitive{},color_sensitive{},noise_sensitive{},offset_sensitive{},mutation_sensitive{};
            const auto export_shader=[&](const std::string& name) {
                if (argc<2) return;
                // Public-generator counterpart with current register arguments;
                // this is not a dump of RasterizerMetal's private shader cache.
                const auto program=Metal::GeneratePicaFragmentShader(Pica::Shader::FSConfig{regs},{},
                    Metal::FragmentColorFormat::RGBA8,Metal::BuildPicaSamplerConfigs(regs));
                std::ofstream file{std::string(argv[1])+"-"+name+".metal"}; file<<program.shader.source;
                Check(bool(file),"Cannot export public shader counterpart"); ++exports;
            };
            const auto encode=[&](unsigned probe) {
                std::vector<u32> triangle;
                Write(triangle,PICA_REG_INDEX(pipeline.triangle_topology),0);
                Write(triangle,PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index),15);
                const auto a=Coordinate(0,probe),b=Coordinate(1,probe),c=Coordinate(2,probe);
                for (const auto& position:positions) {
                    Attribute(triangle,position); Attribute(triangle,{0,0,0,1});
                    Attribute(triangle,{.75f,.25f,.5f,.625f});
                    Attribute(triangle,{a[0],a[1],b[0],b[1]}); Attribute(triangle,{c[0],c[1],0,0});
                }
                process(triangle);
            };
            const auto verify=[&](const std::array<unsigned,4>& expected,const std::string& label,
                                  PAddr address=Memory::VRAM_PADDR) {
                rasterizer.FlushRegion(address,4096);
                const auto pixels=memory.GetPhysicalRef(address);
                Check(pixels && pixels.GetSize()>=4096,"Missing LUT snapshot target");
                for (unsigned pixel=0;pixel<1024;++pixel) for (unsigned c=0;c<4;++c) {
                    const auto actual=pixels.GetPtr()[pixel*4+3-c];
                    if (std::abs(int(actual)-int(expected[c]))>2) {
                        std::cerr<<label<<" pixel="<<pixel<<" channel="<<c<<" actual="<<unsigned(actual)
                            <<" expected="<<expected[c]<<'\n';
                        throw std::runtime_error("Raw Fog/Proctex native FS/TEV/VRAM mismatch");
                    }
                }
            };
            const auto draw=[&](const std::array<unsigned,4>& expected,unsigned probe,
                                const std::string& label,unsigned lut_mask=1) {
                const bool fog_dirty=pica.fog.lut_dirty;
                const auto lighting_dirty=pica.lighting.lut_dirty;
                const auto proctex_dirty=pica.proctex.table_dirty;
                encode(probe);
                Check(pica.fog.lut_dirty==((lut_mask&1) ? false:fog_dirty) &&
                      pica.lighting.lut_dirty==((lut_mask&1) ? 0:lighting_dirty) &&
                      pica.proctex.table_dirty==((lut_mask&6) ? 0:proctex_dirty),
                      "LUT dirty flags were consumed without a reader or not consumed by a reader");
                verify(expected,label);
            };
            for (unsigned revision=0;revision<3;++revision) {
                std::vector<u32> upload;
                Write(upload,PICA_REG_INDEX(texturing.fog_lut_offset),0);
                for (unsigned i=0;i<128;++i) Write(upload,PICA_REG_INDEX(texturing.fog_lut_data[0])+i%8,FogEntry(i,revision));
                for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                    Write(upload,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                    const unsigned count=table<4 ? 128:256;
                    for (unsigned i=0;i<count;++i) Write(upload,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,
                        table<4 ? ValueEntry(table,i,revision):ColorEntry(i,revision,table==5));
                }
                process(upload);
                Check(pica.fog.lut_dirty && (pica.proctex.table_dirty&0x3d)==0x3d,"LUT commands missed dirty tables");
                for (unsigned i=0;i<128;++i) {
                    Check(pica.fog.lut[i].raw==FogEntry(i,revision),"Raw Fog entry mismatch");
                    Check(pica.proctex.noise_table[i].raw==ValueEntry(0,i,revision),"Raw Noise entry mismatch");
                    Check(pica.proctex.color_map_table[i].raw==ValueEntry(2,i,revision),"Raw ColorMap entry mismatch");
                    Check(pica.proctex.alpha_map_table[i].raw==ValueEntry(3,i,revision),"Raw AlphaMap entry mismatch");
                    raw_entries+=4;
                }
                for (unsigned i=0;i<256;++i) {
                    Check(pica.proctex.color_table[i].raw==ColorEntry(i,revision,false),"Raw Color entry mismatch");
                    Check(pica.proctex.color_diff_table[i].raw==ColorEntry(i,revision,true),"Raw ColorDiff entry mismatch");
                    raw_entries+=2;
                }
                std::vector<u32> dormant;
                Write(dormant,PICA_REG_INDEX(texturing.main_config),0);
                Write(dormant,PICA_REG_INDEX(texturing.fog_mode),0);
                Write(dormant,PICA_REG_INDEX(texturing.tev_stage0),0);
                process(dormant);
                draw({191,64,128,159},0,"Inactive LUT revision="+std::to_string(revision),0);
                ++dormant_draws;
                for (unsigned flip=0;flip<2;++flip) {
                    std::vector<u32> setup;
                    Write(setup,PICA_REG_INDEX(texturing.main_config),0);
                    Write(setup,PICA_REG_INDEX(texturing.fog_mode),5 | (flip<<16));
                    Write(setup,PICA_REG_INDEX(texturing.tev_stage0),0); // Primary vertex color.
                    process(setup);
                    if (!revision) export_shader("fog-flip"+std::to_string(flip));
                    for (unsigned sample=0;sample<258;++sample) {
                        const double depth=sample<256 ? (sample/2+(sample%2 ? .75:.25))/128
                                                     : sample==256 ? 0:1;
                        std::vector<u32> depth_command;
                        Write(depth_command,PICA_REG_INDEX(rasterizer.viewport_depth_near_plane),Float24(float(depth)));
                        process(depth_command);
                        const auto expected=FogExpected(depth,flip,revision),no_diff=FogExpected(depth,flip,revision,false);
                        for (unsigned c=0;c<3;++c) fog_sensitive+=std::abs(int(expected[c])-int(no_diff[c]))>4;
                        const auto factor=FogFactor(depth,flip,revision);
                        fog_clamped_low+=factor==0; fog_clamped_high+=factor==1;
                        for (unsigned repeat=0;repeat<2;++repeat) {
                            draw(expected,0,"fog revision="+std::to_string(revision)+" flip="+std::to_string(flip)+" sample="+std::to_string(sample));
                            ++fog_draws;
                        }
                    }
                }
                std::cout<<"revision "<<revision<<" Fog done: "<<fog_draws<<" draws\n"<<std::flush;
                for (unsigned filter=0;filter<6;++filter) for (unsigned combiner=0;combiner<10;++combiner)
                    for (unsigned separate=0;separate<2;++separate) for (unsigned noise=0;noise<2;++noise) {
                        std::vector<u32> setup;
                        Write(setup,PICA_REG_INDEX(texturing.fog_mode),0);
                        Write(setup,PICA_REG_INDEX(rasterizer.viewport_depth_near_plane),Float24(.5f));
                        Write(setup,PICA_REG_INDEX(texturing.main_config),(1<<10) | ((combiner%3)<<8));
                        Write(setup,PICA_REG_INDEX(texturing.proctex),1 | (1<<3) | (combiner<<6) |
                            (((combiner+3)%10)<<10) | (separate<<14) | (noise<<15));
                        Write(setup,PICA_REG_INDEX(texturing.proctex_noise_u),1024);
                        Write(setup,PICA_REG_INDEX(texturing.proctex_noise_v),u32(u16(-512)));
                        Write(setup,PICA_REG_INDEX(texturing.proctex_noise_frequency),0);
                        const unsigned level=filter/2;
                        Write(setup,PICA_REG_INDEX(texturing.proctex_lut),filter | (level<<3) | (level<<7) | (128<<11));
                        Write(setup,PICA_REG_INDEX(texturing.proctex_lut_offset),Offsets[0] | (Offsets[1]<<8) |
                            (Offsets[2]<<16) | (Offsets[3]<<24));
                        Write(setup,PICA_REG_INDEX(texturing.tev_stage0),u32(T::TevStageConfig::Source::Texture3) |
                            (u32(T::TevStageConfig::Source::Texture3)<<16));
                        process(setup);
                        const auto label="proc revision="+std::to_string(revision)+" filter="+std::to_string(filter)+
                            " combiner="+std::to_string(combiner)+" alpha="+std::to_string(separate)+" noise="+std::to_string(noise);
                        if (!revision) export_shader("proc-f"+std::to_string(filter)+"-c"+std::to_string(combiner)+
                            "-a"+std::to_string(separate)+"-n"+std::to_string(noise));
                        for (unsigned probe=0;probe<4;++probe) {
                            const auto expected=ProcExpected(combiner,filter,separate,noise,probe,revision);
                            const auto no_map=ProcExpected(combiner,filter,separate,noise,probe,revision,false);
                            const auto no_color=ProcExpected(combiner,filter,separate,noise,probe,revision,true,false);
                            const auto no_noise=ProcExpected(combiner,filter,separate,false,probe,revision);
                            const auto no_offset=ProcExpected(combiner,filter,separate,noise,probe,revision,true,true,false);
                            for (unsigned c=0;c<4;++c) {
                                map_sensitive+=std::abs(int(expected[c])-int(no_map[c]))>4;
                                color_sensitive+=std::abs(int(expected[c])-int(no_color[c]))>4;
                                noise_sensitive+=std::abs(int(expected[c])-int(no_noise[c]))>4;
                                offset_sensitive+=std::abs(int(expected[c])-int(no_offset[c]))>4;
                                if (revision) mutation_sensitive+=std::abs(int(expected[c])-int(ProcExpected(
                                    combiner,filter,separate,noise,probe,revision-1)[c]))>4;
                            }
                            for (unsigned repeat=0;repeat<2;++repeat) { draw(expected,probe,label+" probe="+std::to_string(probe),6); ++proc_draws; }
                        }
                    }
                std::cout<<"revision "<<revision<<" ProcTex done: "<<proc_draws<<" draws\n"<<std::flush;
            }
            // Upstream Vulkan #2627: changing any table must upload it even when
            // the color LUT is clean. Exercise each dirty bit independently.
            std::array<unsigned,6> table_revisions{};
            std::vector<u32> isolated_setup;
            Write(isolated_setup,PICA_REG_INDEX(texturing.main_config),1<<10);
            Write(isolated_setup,PICA_REG_INDEX(texturing.proctex),1 | (1<<3) | (3<<10) | (1<<14) | (1<<15));
            Write(isolated_setup,PICA_REG_INDEX(texturing.proctex_lut),1 | (128<<11));
            for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                Write(isolated_setup,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                for (unsigned i=0;i<(table<4 ? 128u:256u);++i)
                    Write(isolated_setup,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,
                          table<4 ? ValueEntry(table,i,0):ColorEntry(i,0,table==5));
            }
            process(isolated_setup);
            draw(ProcExpected(0,1,true,true,0,0),0,"Isolated LUT baseline",6);
            unsigned isolated_draws{};
            for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                const auto previous_revisions=table_revisions;
                table_revisions[table]=1;
                std::vector<u32> update;
                Write(update,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                for (unsigned i=0;i<(table<4 ? 128u:256u);++i)
                    Write(update,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,
                          table<4 ? ValueEntry(table,i,1):ColorEntry(i,1,table==5));
                process(update);
                Check(pica.proctex.table_dirty==(1u<<table),"Isolated LUT update dirtied unrelated tables");
                unsigned sensitive{};
                for (unsigned probe=0;probe<4;++probe) {
                    const auto expected=ProcExpected(0,1,true,true,probe,0,true,true,true,&table_revisions);
                    const auto stale=ProcExpected(0,1,true,true,probe,0,true,true,true,&previous_revisions);
                    for (unsigned c=0;c<4;++c) sensitive+=std::abs(int(expected[c])-int(stale[c]))>4;
                    draw(expected,probe,"Isolated LUT table="+std::to_string(table),6);
                    ++isolated_draws;
                }
                Check(sensitive,"Isolated LUT oracle cannot distinguish a skipped upload");
            }
            Check(isolated_draws==20,"Incomplete independent LUT update coverage");
            std::cout<<isolated_draws<<" isolated ProcTex LUT draws cover all five independent dirty bits\n";
            // Record different immutable LUT snapshots before submitting any of them.
            // A reused/overwritten physical allocation would change earlier draws too.
            scheduler.Finish();
            const auto queued_tick=scheduler.CurrentTick();
            std::array<std::array<unsigned,4>,8> queued_expected{};
            unsigned queued_sensitive{};
            for (unsigned index=0;index<queued_expected.size();++index) {
                const unsigned revision=index%3;
                std::vector<u32> setup;
                Write(setup,PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address),
                      (Memory::VRAM_PADDR+(index+1)*0x10000)/8);
                Write(setup,PICA_REG_INDEX(rasterizer.viewport_depth_near_plane),Float24(.34375f));
                if (index<4) {
                    Write(setup,PICA_REG_INDEX(texturing.main_config),0);
                    Write(setup,PICA_REG_INDEX(texturing.fog_mode),5);
                    Write(setup,PICA_REG_INDEX(texturing.tev_stage0),0);
                    Write(setup,PICA_REG_INDEX(texturing.fog_lut_offset),0);
                    for (unsigned i=0;i<128;++i)
                        Write(setup,PICA_REG_INDEX(texturing.fog_lut_data[0])+i%8,FogEntry(i,revision));
                    queued_expected[index]=FogExpected(.34375,false,revision);
                } else {
                    Write(setup,PICA_REG_INDEX(texturing.fog_mode),0);
                    Write(setup,PICA_REG_INDEX(texturing.main_config),1<<10);
                    Write(setup,PICA_REG_INDEX(texturing.proctex),1 | (1<<3) | (3<<10) | (1<<14) | (1<<15));
                    Write(setup,PICA_REG_INDEX(texturing.proctex_noise_u),1024);
                    Write(setup,PICA_REG_INDEX(texturing.proctex_noise_v),u32(u16(-512)));
                    Write(setup,PICA_REG_INDEX(texturing.proctex_noise_frequency),0);
                    Write(setup,PICA_REG_INDEX(texturing.proctex_lut),128<<11);
                    Write(setup,PICA_REG_INDEX(texturing.proctex_lut_offset),Offsets[0] | (Offsets[1]<<8) |
                          (Offsets[2]<<16) | (Offsets[3]<<24));
                    Write(setup,PICA_REG_INDEX(texturing.tev_stage0),u32(T::TevStageConfig::Source::Texture3) |
                          (u32(T::TevStageConfig::Source::Texture3)<<16));
                    for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                        Write(setup,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                        for (unsigned i=0;i<(table<4 ? 128u:256u);++i)
                            Write(setup,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,
                                  table<4 ? ValueEntry(table,i,revision):ColorEntry(i,revision,table==5));
                    }
                    queued_expected[index]=ProcExpected(0,0,true,true,1,revision);
                }
                process(setup);
                encode(1);
                if (index && index!=4) for (unsigned c=0;c<4;++c)
                    queued_sensitive+=std::abs(int(queued_expected[index][c])-int(queued_expected[index-1][c]))>4;
            }
            Check(scheduler.CurrentTick()==queued_tick && queued_sensitive,
                  "LUT snapshot draws were submitted early or have identical oracle colors");
            for (unsigned index=0;index<queued_expected.size();++index)
                verify(queued_expected[index],"Queued LUT snapshot="+std::to_string(index),
                       Memory::VRAM_PADDR+(index+1)*0x10000);
            Check(fog_draws==3096 && proc_draws==5760 && raw_entries==3072 && dormant_draws==3 &&
                  (argc<2 || exports==242),"Incomplete Fog/ProcTex matrix");
            Check(fog_sensitive && fog_clamped_low && fog_clamped_high && map_sensitive && color_sensitive &&
                  noise_sensitive && offset_sensitive && mutation_sensitive,"Missing raw LUT sensitivity/clamp coverage");
            Check(!system.IsPoweredOn(),"Data harness initialized emulation");
            std::cout<<fog_draws<<" Fog and "<<proc_draws<<" ProcTex integrated raw LUT/FS/TEV/physical VRAM draws; "
                <<raw_entries<<" raw entries; all 1024 RGBA pixels; tolerance 2\n"
                <<"Oracle coverage: fog-diff="<<fog_sensitive<<" fog-low="<<fog_clamped_low<<" fog-high="<<fog_clamped_high
                <<" map-diff="<<map_sensitive<<" color-diff="<<color_sensitive<<" noise="<<noise_sensitive
                <<" offset="<<offset_sensitive<<" mutation="<<mutation_sensitive<<'\n';
            std::cout<<dormant_draws<<" inactive LUT draws preserve dirty flags before activation; "
                     <<queued_expected.size()<<" queued Fog/ProcTex snapshots preserve independent VRAM images\n";
            return 0;
        } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
