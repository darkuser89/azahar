// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA attributes/registers -> extended Proctex LUT/Noise/Repeat/LOD -> VRAM.
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
#include <string_view>

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
constexpr std::array<unsigned,8> Offsets{7,135,199,231,240,248,252,254};
int Signed(unsigned bits, unsigned width) {
    const auto sign = 1U << (width-1);
    return int(bits & (sign-1)) - int(bits & sign);
}
u32 ValueEntry(unsigned table, unsigned index, unsigned revision) {
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
// Frozen current-generator integer lattice; each hex digit encodes
// the random numerator n for -1+2*n/15. No runtime shader hash in oracle.
constexpr std::array<std::string_view,25> NoiseGrid{{
    "a771940070c87940070c8010c",
    "b0608b1367db48b1367db307d",
    "859ff62e92ee1f62e92eeef2e",
    "0d177ea61a6697ea61a6667a6",
    "49db326ade22d326ade22abe2",
    "fc244fd72b9784fd72b9774b9",
    "e33558c43484b58c434844548",
    "968ee53181f96e53181f91e1f",
    "859ff62e92ee1f62e92eeef2e",
    "38e8039bef53c039bef53b8f5",
    "2ff91c88f840f1c88f8408984",
    "d242a9f545b5aa9f545b5525b",
    "5aca217dcd3d2217dcd3ddad3",
    "49db326ade22d326ade22abe2",
    "fc244fd72b9784fd72b9774b9",
    "e33558c43484b58c434844548",
    "968ee53181f96e53181f91e1f",
    "859ff62e92ee1f62e92eeef2e",
    "38e8039bef53c039bef53b8f5",
    "2ff91c88f840f1c88f8408984",
    "d242a9f545b5aa9f545b5525b",
    "968ee53181f96e53181f91e1f",
    "0d177ea61a6697ea61a6667a6",
    "38e8039bef53c039bef53b8f5",
    "2ff91c88f840f1c88f8408984",
}};
u32 Float16(float value) {
    if (value == 0) return 0;
    const auto raw=Float24(value); const auto exponent=(raw>>16)&127;
    Check((raw&63)==0 && exponent>=49 && exponent<=78,"Inexact f16 fixture");
    return ((raw>>8)&0x8000) | ((exponent-48)<<10) | ((raw&65535)>>6);
}
struct NoiseParameters {
    std::array<double,2> frequency, phase;
    std::array<int,2> amplitude;
};
constexpr std::array<NoiseParameters,2> NoiseProfiles{{
    {{.125,.25},{.125,-.375},{2048,-1024}},
    {{-.25,.125},{-.5,.25},{-1024,2048}}}};
constexpr std::array<std::array<float,2>,4> Probes{{
    {-2.87109375f,.06640625f},{.19140625f,1.37109375f},
    {1.87890625f,-.43359375f},{3.05859375f,2.12890625f}}};
constexpr std::array<std::array<double,2>,4> Spans{{{.125,.0625},{1.5,1},{5.5,3},{0,0}}};
constexpr std::array<float,5> Biases{0,.5f,1.125f,-2,64};
struct Case {
    unsigned clamp_u{},clamp_v{},shift_u{},shift_v{},filter{},combiner{},alpha{},coord{},level{};
    bool separate{},noise{},varying{};
};
Case WrapCase(unsigned code,bool noise,bool separate) {
    Case c{};c.filter=code%6;c.combiner=code%10;c.alpha=(c.combiner+3)%10;
    c.coord=code%3;c.level=c.filter/2;c.noise=noise;c.separate=separate;
    c.shift_v=code%3;code/=3;c.shift_u=code%3;code/=3;c.clamp_v=code%5;c.clamp_u=code/5;
    return c;
}
Case MipCase(unsigned filter,unsigned coord,bool separate) {
    Case c{};c.clamp_u=2;c.clamp_v=3;c.shift_u=1;c.shift_v=2;
    c.filter=filter;c.combiner=4;c.alpha=2;c.coord=coord;c.separate=separate;c.varying=true;
    return c;
}
std::array<double,2> Coordinate(unsigned coord,unsigned probe) {
    const auto [u,v]=Probes[probe];
    return coord==0 ? std::array<double,2>{u,v} : coord==1 ? std::array<double,2>{v,1-u} : std::array<double,2>{1-u,1-v};
}
std::array<double,2> Transform(unsigned coord,double u,double v) {
    return coord==0 ? std::array{u,v} : coord==1 ? std::array{.5*u+.0625,2*v+.125} : std::array{2*u+.125,.5*v+.0625};
}
std::array<double,2> VaryingCoordinate(unsigned coord,unsigned gradient,double x,double y) {
    // Dyadic margins keep arbitrary discontinuous LUT cells away from raster interpolation edges.
    return Transform(coord,.125+1.0/4096+Spans[gradient][0]*x,
                     .25+3.0/4096+Spans[gradient][1]*y);
}
double VaryingLod(unsigned coord,unsigned gradient,float bias) {
    const double du=Spans[gradient][0]/32*(coord==1 ? .5 : coord==2 ? 2 : 1);
    const double dv=Spans[gradient][1]/32*(coord==1 ? 2 : coord==2 ? .5 : 1);
    if (bias==0 || du+dv==0) return 0;
    return std::clamp(std::log2(std::abs(128*bias)*(du+dv)),0.0,7.0);
}
double RandomCorner(int x,int y) {
    Check(x>=-12 && x<=12 && y>=-12 && y<=12,"Noise grid fixture bounds exceeded");
    const char digit=NoiseGrid[y+12][x+12];
    const unsigned n=digit<='9' ? digit-'0' : digit-'a'+10;
    return -1+2*n/15.0;
}
double NoiseCoefficient(double u,double v,unsigned profile,unsigned revision,bool map_difference,bool variable_noise_map) {
    const auto& n=NoiseProfiles[profile];
    const double gx=n.frequency[0]*9*std::abs(u+n.phase[0]),gy=n.frequency[1]*9*std::abs(v+n.phase[1]);
    const int ix=int(std::floor(gx)),iy=int(std::floor(gy));
    const double fx=gx-ix,fy=gy-iy;
    const double tx=variable_noise_map ? Map(0,fx,revision,map_difference) : .5;
    const double ty=variable_noise_map ? Map(0,fy,revision,map_difference) : .5;
    // Independent tensor-product interpolation over precomputed corner data.
    // Shader's hash and nested-mix implementation are not duplicated here.
    const double d=fx+fy;
    return RandomCorner(ix,iy)*d*(1-tx)*(1-ty) + RandomCorner(ix+1,iy)*(d-1)*tx*(1-ty) +
           RandomCorner(ix,iy+1)*(d-1)*(1-tx)*ty + RandomCorner(ix+1,iy+1)*(d-2)*tx*ty;
}
double Shift(unsigned mode,unsigned clamp,double other) {
    if (!mode) return 0;
    return (clamp==3 ? 1 : .5)*((unsigned(other)+(mode==2))/2%2);
}
double Clamp(unsigned mode,double value) {
    switch (mode) {
    case 0:return value>1 ? 0 : value;
    case 1:return std::min(value,1.0);
    case 2:return value-std::floor(value);
    case 3:{const double f=value-std::floor(value);return unsigned(value)%2 ? 1-f : f;}
    case 4:return value>.5 ? 1 : 0;
    }
    throw std::runtime_error("Invalid clamp fixture");
}
std::array<double,4> LevelColor(double mapped,unsigned level,unsigned filter,unsigned revision,bool difference) {
    Check(level<8,"Out-of-bounds mip fixture");
    const auto width=128U>>level; const double p=mapped*(width-1);
    const bool linear=filter&1;const unsigned index=Offsets[level]+unsigned(std::floor(p+(linear ? 0 : .5)));
    Check(index<256,"Out-of-bounds color fixture");
    const auto raw=ColorEntry(index,revision,false),delta=ColorEntry(index,revision,true);
    std::array<double,4> rgba{};
    for (unsigned ch=0;ch<4;++ch) rgba[ch]=((raw>>(ch*8))&255) +
        (linear && difference ? 2*Signed((delta>>(ch*8))&255,8)*(p-std::floor(p)) : 0);
    return rgba; // Clamp after mip interpolation, not per level.
}
std::array<unsigned,4> Expected(Case c,std::array<double,2> uv,unsigned profile,unsigned revision,double lod,
                               bool map_difference=true,bool color_difference=true,bool noise=true,bool shifts=true,bool variable_noise_map=true) {
    double u=std::abs(uv[0]),v=std::abs(uv[1]);
    const double su=shifts ? Shift(c.shift_u,c.clamp_u,v) : 0,sv=shifts ? Shift(c.shift_v,c.clamp_v,u) : 0;
    if (c.noise && noise) {
        const double coef=NoiseCoefficient(u,v,profile,revision,map_difference,variable_noise_map);
        u=std::abs(u+NoiseProfiles[profile].amplitude[0]/4095.0*coef);
        v=std::abs(v+NoiseProfiles[profile].amplitude[1]/4095.0*coef);
    }
    u=Clamp(c.clamp_u,u+su);v=Clamp(c.clamp_v,v+sv);
    const double mapped=Map(2,Combine(c.combiner,u,v),revision,map_difference);
    unsigned level{};double fraction{};
    if (c.filter>=4) {level=unsigned(lod);fraction=lod-level;}
    else if (c.filter>=2) level=unsigned(std::floor(lod+.5));
    auto rgba=LevelColor(mapped,level,c.filter,revision,color_difference);
    if (c.filter>=4) {
        const auto next=LevelColor(mapped,std::min(level+1,7U),c.filter,revision,color_difference);
        for (unsigned ch=0;ch<4;++ch) rgba[ch]=rgba[ch]*(1-fraction)+next[ch]*fraction;
    }
    if (c.separate) rgba[3]=Map(3,Combine(c.alpha,u,v),revision,map_difference)*255;
    std::array<unsigned,4> result{};
    for (unsigned ch=0;ch<4;++ch) result[ch]=unsigned(std::lround(std::clamp(rgba[ch],0.0,255.0)));
    return result;
}
unsigned GuestPixel(unsigned x,unsigned y) {
    constexpr std::array<unsigned,8> morton{0,1,4,5,16,17,20,21};
    return ((y/8)*4+x/8)*64+morton[x%8]+2*morton[y%8];
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

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Check(argc<=2 && (argc==1 || std::filesystem::path{argv[1]}.is_absolute()),"Expected optional absolute shader prefix");
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
            for (auto* stage:std::array{&regs.texturing.tev_stage0,&regs.texturing.tev_stage1,&regs.texturing.tev_stage2,
                                       &regs.texturing.tev_stage3,&regs.texturing.tev_stage4,&regs.texturing.tev_stage5})
                stage->sources_raw=0xf000f;
            pica.dirty_regs.SetAllDirty();
            const auto process=[&](const std::vector<u32>& words) {
                const auto bytes=words.size()*sizeof(u32);auto backing=memory.GetPhysicalRef(Memory::VRAM_PADDR+0x8000);
                Check(backing && backing.GetSize()>=bytes,"Invalid public command backing");
                std::memcpy(backing.GetPtr(),words.data(),bytes);
                pica.ProcessCmdList(Memory::VRAM_PADDR+0x8000,u32(bytes),false);
            };
            std::vector<u32> init;
            for (unsigned table=0;table<24;++table) {
                Write(init,PICA_REG_INDEX(lighting.lut_config),table<<8);
                for (unsigned i=0;i<256;++i) Write(init,PICA_REG_INDEX(lighting.lut_data[0])+i%8,0);
            }
            Write(init,PICA_REG_INDEX(texturing.fog_lut_offset),0);
            for (unsigned i=0;i<128;++i) Write(init,PICA_REG_INDEX(texturing.fog_lut_data[0])+i%8,0);
            process(init);
            auto target=memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize()>=4096,"Missing physical framebuffer");std::memset(target.GetPtr(),0x11,4096);
            constexpr std::array<std::array<float,4>,3> positions{{{-1,-1,-.5f,1},{3,-1,-.5f,1},{-1,3,-.5f,1}}};
            unsigned wrap_draws{},mip_draws{},raw_entries{},exports{};
            unsigned noise_sensitive{},noise_map_sensitive{},shift_sensitive{},map_sensitive{},color_sensitive{},mutation_sensitive{},mip_sensitive{};
            unsigned fractional_lods{},clamped_high_lods{},zero_bias_cases{};
            const auto setup=[&](Case c,unsigned profile,float bias) {
                const auto& noise=NoiseProfiles[profile];const auto half=Float16(bias);
                std::vector<u32> words;
                Write(words,PICA_REG_INDEX(texturing.main_config),(1<<10)|(c.coord<<8));
                Write(words,PICA_REG_INDEX(texturing.proctex),c.clamp_u|(c.clamp_v<<3)|(c.combiner<<6)|(c.alpha<<10)|
                    (u32(c.separate)<<14)|(u32(c.noise)<<15)|(c.shift_u<<16)|(c.shift_v<<18)|((half&255)<<20));
                Write(words,PICA_REG_INDEX(texturing.proctex_noise_u),u32(u16(noise.amplitude[0]))|(Float16(float(noise.phase[0]))<<16));
                Write(words,PICA_REG_INDEX(texturing.proctex_noise_v),u32(u16(noise.amplitude[1]))|(Float16(float(noise.phase[1]))<<16));
                Write(words,PICA_REG_INDEX(texturing.proctex_noise_frequency),Float16(float(noise.frequency[0]))|
                    (Float16(float(noise.frequency[1]))<<16));
                Write(words,PICA_REG_INDEX(texturing.proctex_lut),c.filter|(c.varying ? 7<<7 : (c.level<<3)|(c.level<<7))|
                    (128<<11)|((half>>8)<<19));
                Write(words,PICA_REG_INDEX(texturing.proctex_lut_offset),Offsets[0]|(Offsets[1]<<8)|(Offsets[2]<<16)|(Offsets[3]<<24));
                using S=T::TevStageConfig::Source;
                Write(words,PICA_REG_INDEX(texturing.tev_stage0),u32(S::Texture3)|(u32(S::Texture3)<<16));
                process(words);
            };
            const auto export_shader=[&](const std::string& name) {
                if (argc<2) return;
                const auto program=Metal::GeneratePicaFragmentShader(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8,
                                                                     Metal::BuildPicaSamplerConfigs(regs));
                std::ofstream file{std::string(argv[1])+"-"+name+".metal"};file<<program.shader.source;
                Check(bool(file),"Cannot export public Proctex counterpart");++exports;
            };
            const auto draw=[&](Case c,unsigned probe_or_gradient,unsigned profile,float bias,unsigned revision,const std::string& label) {
                std::vector<u32> words;
                Write(words,PICA_REG_INDEX(pipeline.triangle_topology),0);
                Write(words,PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index),15);
                for (const auto& position:positions) {
                    std::array<std::array<double,2>,3> uv{};
                    for (unsigned unit=0;unit<3;++unit) uv[unit]=c.varying ? VaryingCoordinate(unit,probe_or_gradient,
                        (position[0]+1)/2,(position[1]+1)/2) : Coordinate(unit,probe_or_gradient);
                    Attribute(words,position);Attribute(words,{0,0,0,1});Attribute(words,{1,1,1,1});
                    Attribute(words,{float(uv[0][0]),float(uv[0][1]),float(uv[1][0]),float(uv[1][1])});
                    Attribute(words,{float(uv[2][0]),float(uv[2][1]),0,0});
                }
                process(words);rasterizer.FlushRegion(Memory::VRAM_PADDR,4096);
                Check(!pica.proctex.table_dirty,"Native Proctex draw left LUT dirty");
                const double lod=c.varying ? VaryingLod(c.coord,probe_or_gradient,bias) : c.level;
                const auto solid=Expected(c,Coordinate(c.coord,probe_or_gradient),profile,revision,lod);
                for (unsigned y=0;y<32;++y) for (unsigned x=0;x<32;++x) {
                    const auto uv=c.varying ? VaryingCoordinate(c.coord,probe_or_gradient,(x+.5)/32,(31-y+.5)/32)
                                           : Coordinate(c.coord,probe_or_gradient);
                    const auto expected=c.varying ? Expected(c,uv,profile,revision,lod) : solid;
                    for (unsigned ch=0;ch<4;++ch) {
                        const auto actual=target.GetPtr()[GuestPixel(x,y)*4+3-ch];
                        if (std::abs(int(actual)-int(expected[ch]))>2) {
                            std::cerr<<label<<" x="<<x<<" guest-y="<<y<<" channel="<<ch<<" actual="<<unsigned(actual)
                                <<" expected="<<expected[ch]<<" lod="<<lod<<'\n';
                            throw std::runtime_error("Extended Proctex/Noise/Repeat/LOD/physical VRAM mismatch");
                        }
                    }
                    if (c.varying && c.filter>=4) {
                        const auto integer=Expected(c,uv,profile,revision,std::floor(lod+.5));
                        for (unsigned ch=0;ch<4;++ch) mip_sensitive+=std::abs(int(expected[ch])-int(integer[ch]))>4;
                    }
                }
            };
            for (unsigned revision=0;revision<2;++revision) {
                std::vector<u32> upload;
                for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                    Write(upload,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                    const unsigned count=table<4 ? 128 : 256;
                    for (unsigned i=0;i<count;++i) Write(upload,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,
                        table<4 ? ValueEntry(table,i,revision) : ColorEntry(i,revision,table==5));
                }
                process(upload);Check((pica.proctex.table_dirty&0x3d)==0x3d,"Public uploads missed Proc tables");
                for (unsigned i=0;i<128;++i) {
                    Check(pica.proctex.noise_table[i].raw==ValueEntry(0,i,revision),"Raw nonconstant Noise mismatch");
                    Check(pica.proctex.color_map_table[i].raw==ValueEntry(2,i,revision),"Raw ColorMap mismatch");
                    Check(pica.proctex.alpha_map_table[i].raw==ValueEntry(3,i,revision),"Raw AlphaMap mismatch");raw_entries+=3;
                }
                for (unsigned i=0;i<256;++i) {
                    Check(pica.proctex.color_table[i].raw==ColorEntry(i,revision,false),"Raw Color mismatch");
                    Check(pica.proctex.color_diff_table[i].raw==ColorEntry(i,revision,true),"Raw ColorDiff mismatch");raw_entries+=2;
                }
                for (unsigned code=0;code<225;++code) for (unsigned noise:std::array<unsigned,2>{1,0})
                    for (unsigned separate=0;separate<2;++separate) {
                        const auto c=WrapCase(code,noise,separate);
                        for (unsigned profile=0;profile<(noise ? 2U : 1U);++profile) {
                            setup(c,profile,1.125f);
                            const auto name="wrap"+std::to_string(code)+"-n"+std::to_string(noise)+"-a"+std::to_string(separate);
                            if (!revision && !profile) export_shader(name);
                            for (unsigned probe=0;probe<4;++probe) {
                                const auto uv=Coordinate(c.coord,probe);const auto expected=Expected(c,uv,profile,revision,c.level);
                                const auto no_noise=Expected(c,uv,profile,revision,c.level,true,true,false);
                                const auto constant_noise_map=Expected(c,uv,profile,revision,c.level,true,true,true,true,false);
                                const auto no_shift=Expected(c,uv,profile,revision,c.level,true,true,true,false);
                                const auto no_map=Expected(c,uv,profile,revision,c.level,false);
                                const auto no_color=Expected(c,uv,profile,revision,c.level,true,false);
                                for (unsigned ch=0;ch<4;++ch) {
                                    noise_sensitive+=std::abs(int(expected[ch])-int(no_noise[ch]))>4;
                                    noise_map_sensitive+=std::abs(int(expected[ch])-int(constant_noise_map[ch]))>4;
                                    shift_sensitive+=std::abs(int(expected[ch])-int(no_shift[ch]))>4;
                                    map_sensitive+=std::abs(int(expected[ch])-int(no_map[ch]))>4;
                                    color_sensitive+=std::abs(int(expected[ch])-int(no_color[ch]))>4;
                                    if (revision) mutation_sensitive+=std::abs(int(expected[ch])-int(Expected(c,uv,profile,revision-1,c.level)[ch]))>4;
                                }
                                for (unsigned repeat=0;repeat<2;++repeat) {
                                    draw(c,probe,profile,1.125f,revision,"revision="+std::to_string(revision)+" "+name+
                                        " profile="+std::to_string(profile)+" probe="+std::to_string(probe));++wrap_draws;
                                }
                            }
                        }
                    }
                std::cout<<"revision "<<revision<<" wrap/noise complete: "<<wrap_draws<<" draws\n"<<std::flush;
                for (unsigned filter=0;filter<6;++filter) for (unsigned coord=0;coord<3;++coord)
                    for (unsigned separate=0;separate<2;++separate) {
                        const auto c=MipCase(filter,coord,separate);
                        const auto name="mip-f"+std::to_string(filter)+"-c"+std::to_string(coord)+"-a"+std::to_string(separate);
                        for (unsigned gradient=0;gradient<4;++gradient) for (unsigned bias_index=0;bias_index<Biases.size();++bias_index) {
                            const auto bias=Biases[bias_index];setup(c,0,bias);
                            if (!revision && !gradient && !bias_index) export_shader(name);
                            const auto lod=VaryingLod(coord,gradient,bias);
                            fractional_lods+=lod>0 && lod<7 && std::abs(lod-std::round(lod))>.01;
                            clamped_high_lods+=lod==7;zero_bias_cases+=bias==0;
                            for (unsigned repeat=0;repeat<2;++repeat) {
                                draw(c,gradient,0,bias,revision,"revision="+std::to_string(revision)+" "+name+
                                    " gradient="+std::to_string(gradient)+" bias="+std::to_string(bias_index));++mip_draws;
                            }
                        }
                    }
                std::cout<<"revision "<<revision<<" varying/mip complete: "<<mip_draws<<" draws\n"<<std::flush;
            }
            Check(wrap_draws==21600 && mip_draws==2880 && raw_entries==1792 && (argc<2 || exports==936),"Incomplete extended Proc matrix");
            Check(noise_sensitive && noise_map_sensitive && shift_sensitive && map_sensitive && color_sensitive && mutation_sensitive && mip_sensitive &&
                  fractional_lods && clamped_high_lods && zero_bias_cases,"Missing extended Proc sensitivity/LOD coverage");
            Check(!system.IsPoweredOn(),"Data harness initialized emulation");
            std::cout<<wrap_draws<<" wrap/noise and "<<mip_draws<<" varying/mip integrated Proctex/physical VRAM draws; "
                <<raw_entries<<" raw entries; all 1024 RGBA pixels; tolerance 2\n"
                <<"Oracle coverage: noise="<<noise_sensitive<<" noise-map="<<noise_map_sensitive<<" shift="<<shift_sensitive<<" map="<<map_sensitive<<" color="<<color_sensitive
                <<" mutation="<<mutation_sensitive<<" mip="<<mip_sensitive<<" fractional="<<fractional_lods
                <<" high-lod="<<clamped_high_lods<<" zero-bias="<<zero_bias_cases<<'\n';
            return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
