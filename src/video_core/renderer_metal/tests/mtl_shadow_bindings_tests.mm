// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA attributes/registers -> typed lighting-shadow texture bindings -> VRAM.
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
constexpr std::array<unsigned, 2> Physical{2, 7};
constexpr std::array<std::array<unsigned, 3>, 2> Diffuse{{{90,60,120}, {120,100,55}}};
constexpr std::array<std::array<unsigned, 3>, 2> Ambient{{{12,20,8}, {9,7,15}}};
constexpr std::array<std::array<unsigned, 3>, 2> Spec0{{{110,75,45}, {75,95,105}}};
constexpr std::array<std::array<unsigned, 3>, 2> Spec1{{{25,55,90}, {40,35,25}}};
constexpr std::array<unsigned, 3> Global{13,21,29};

using C = Pica::TexturingRegs::TextureConfig;
using Vec = std::array<double,3>;
constexpr unsigned TextureBytes = (32*32+16*16+8*8)*4;
constexpr std::array<unsigned,3> MipOffsets{0,32*32*4,(32*32+16*16)*4};
constexpr std::array<unsigned,4> Border{17,83,149,211};
struct Case {
    C::TextureType type; bool enabled,linear,orthographic;
    unsigned level,wrap_s,wrap_t,bias;
};
std::vector<Case> Cases() {
    std::vector<Case> result;
    for (auto type : {C::Shadow2D,C::ShadowCube}) for (unsigned profile=0;profile<4;++profile)
        result.push_back({type,true,bool(profile&1),bool(profile&1),0,profile&1 ? 4U : 1U,
                          profile&1 ? 5U : 1U,(profile/2)*0x80000});
    constexpr std::array<std::array<unsigned,2>,6> wraps{{{0,0},{1,0},{0,1},{1,1},{4,5},{5,4}}};
    for (auto type : {C::TextureCube,C::Projection2D,C::Texture2D}) for (unsigned profile=0;profile<6;++profile)
        result.push_back({type,true,bool(profile&1),false,profile/2,wraps[profile][0],wraps[profile][1],0});
    result.push_back({C::Disabled,true,false,false,0,0,0,0});
    for (unsigned type=0;type<6;++type) result.push_back({C::TextureType(type),false,false,false,0,0,0,0});
    return result;
}
void Color(Pica::LightingRegs::LightColor& color,const std::array<unsigned,3>& rgb) {
    color.r.Assign(rgb[0]);color.g.Assign(rgb[1]);color.b.Assign(rgb[2]);
}
unsigned Morton(unsigned x,unsigned y,unsigned width) {
    constexpr std::array<unsigned,8> interleave{0,1,4,5,16,17,20,21};
    return ((y/8)*(width/8)+x/8)*64+interleave[x%8]+2*interleave[y%8];
}
std::array<unsigned,4> Texel(unsigned face,unsigned level,unsigned x,unsigned y,unsigned revision) {
    return {25+(face*29+level*17+x*7+y*3+revision*31)%190,
            31+(face*11+level*29+x*3+y*9+revision*17)%180,
            39+(face*19+level*13+x*11+y*5+revision*23)%170,
            57+(face*23+level*31+x*5+y*7+revision*29)%150};
}
u32 ShadowWord(unsigned face,unsigned x,unsigned y,unsigned revision) {
    const auto depth=0x200000+(face*0x140000+x*0x170000+y*0x51000+revision*0x90000)%0xc00000;
    const auto intensity=30+(face*23+x*11+y*7+revision*29)%190;
    return (depth<<8)|intensity;
}
Vec Coordinate(C::TextureType type,unsigned flags) {
    const auto probe=(flags/6)%4;
    if (type==C::TextureCube || type==C::ShadowCube) {
        constexpr std::array<std::array<double,2>,4> interior{{{.1875,.3125},{.6875,.5625},{.34375,.71875},{.8125,.1875}}};
        constexpr std::array<std::array<double,2>,4> shadow{{{.1875,.3125},{.6875,.5625},{.0078125,.9921875},{.9921875,.0078125}}};
        auto uv=type==C::TextureCube ? interior[probe] : shadow[probe];
        // Keep native cube nearest footprints off face-projection texel boundaries.
        if (type==C::TextureCube) {uv[0]+=1.0/256;uv[1]+=3.0/256;}
        const double s=2*uv[0]-1,t=2*uv[1]-1,m=probe&1 ? .75 : .5;
        // Six standard cube bases; major axis and sign are independent of data.
        const std::array<Vec,6> directions{{{m,-t*m,-s*m},{-m,-t*m,s*m},{s*m,m,t*m},
                                          {s*m,-m,-t*m},{s*m,-t*m,m},{-s*m,-t*m,-m}}};
        return directions[flags%6];
    }
    constexpr std::array<Vec,4> coords{{{.125,.25,.5},{-.125,.25,.5},{.625,.875,.75},{.03125,.1875,-.5}}};
    auto value=coords[probe];if (type==C::Projection2D && probe==2) value[2]=2;
    return value;
}
unsigned CoordinateCode(unsigned flags,unsigned mask) {return flags/2+mask*17;}
struct Projection {unsigned face;double u,v,major;};
Projection Project(Vec d) {
    const std::array<double,3> a{std::abs(d[0]),std::abs(d[1]),std::abs(d[2])};
    unsigned axis=2;if (a[0]>a[1] && a[0]>a[2]) axis=0;else if (a[1]>a[2]) axis=1;
    const unsigned face=axis*2+(d[axis]<0);const auto major=a[axis];Check(major>0,"Invalid cube fixture direction");
    // Dot products against standard per-face tangent bases.
    constexpr std::array<Vec,6> horizontal{{{0,0,-1},{0,0,1},{1,0,0},{1,0,0},{1,0,0},{-1,0,0}}};
    constexpr std::array<Vec,6> vertical{{{0,-1,0},{0,-1,0},{0,0,1},{0,0,-1},{0,-1,0},{0,-1,0}}};
    double s{},t{};for (unsigned i=0;i<3;++i) {s+=d[i]*horizontal[face][i];t+=d[i]*vertical[face][i];}
    return {face,.5+s/(2*major),.5+t/(2*major),major};
}
int Wrap(int index,int size,unsigned mode,double coordinate) {
    if (mode==4) mode=coordinate<0 ? 2 : 0;
    if (mode==5) mode=coordinate<0 ? 2 : 1;
    if (mode==0) return std::clamp(index,0,size-1);
    if (mode==1) return index<0 || index>=size ? -1 : index;
    const auto period=mode==3 ? 2*size : size;
    const auto i=(index%period+period)%period;
    return mode==3 && i>=size ? 2*size-1-i : i;
}
std::array<double,4> Sample(Case c,Vec coordinate,unsigned revision,bool project=true,int forced_face=-1,bool null_bounds=true) {
    if (c.type==C::Disabled || (!c.enabled && (c.type!=C::Shadow2D || !null_bounds))) return {};
    unsigned face{};double u=coordinate[0],v=coordinate[1],w=coordinate[2];
    const bool cube=c.type==C::TextureCube || c.type==C::ShadowCube;
    const bool shadow=c.type==C::Shadow2D || c.type==C::ShadowCube;
    if (cube) {const auto p=Project(coordinate);face=p.face;u=p.u;v=p.v;w=p.major;}
    else if (project && (c.type==C::Projection2D || (shadow && !c.orthographic))) {u/=w;v/=w;}
    if (forced_face>=0) face=unsigned(forced_face);
    const auto level=shadow ? 0U : c.level;
    // A disabled Shadow2D still executes bounded reads on the canonical 1x1
    // black null surface. Out-of-range taps return one, including partial edges.
    const auto size=c.enabled ? 32U>>level : 1U;
    const bool linear=shadow || c.linear;
    const double px=u*size-(linear ? .5 : 0),py=v*size-(linear ? .5 : 0);
    const int bx=int(std::floor(px)),by=int(std::floor(py));
    const auto z=u32(std::max(0,int(std::min(std::abs(w),1.0)*16777215.0)-int(c.bias*2)));
    std::array<double,4> rgba{};
    for (unsigned j=0;j<(linear ? 2U : 1U);++j) for (unsigned i=0;i<(linear ? 2U : 1U);++i) {
        int x=bx+int(i),y=by+int(j);const double weight=linear ? (i ? px-bx : 1-(px-bx))*(j ? py-by : 1-(py-by)) : 1;
        std::array<double,4> tap{};
        if (shadow) {
            double value=1;
            if (cube || (x>=0 && y>=0 && x<int(size) && y<int(size))) {
                x=std::clamp(x,0,int(size)-1);y=std::clamp(y,0,int(size)-1);
                const auto raw=c.enabled ? ShadowWord(face,unsigned(x),unsigned(y),revision) : 0U;
                value=(raw>>8)>z ? (raw&255)/255.0 : 0;
            }
            tap.fill(value);
        } else {
            x=Wrap(x,int(size),cube ? 0 : c.wrap_s,u);y=Wrap(y,int(size),cube ? 0 : c.wrap_t,v);
            const auto value=x<0 || y<0 ? Border : Texel(face,level,unsigned(x),unsigned(y),revision);
            for (unsigned ch=0;ch<4;++ch) tap[ch]=value[ch]/255.0;
        }
        for (unsigned ch=0;ch<4;++ch) rgba[ch]+=weight*tap[ch];
    }
    return rgba;
}
std::array<unsigned,4> Expected(Case c,unsigned flags,unsigned mask,bool secondary,unsigned revision,
                                bool project=true,int forced_face=-1,int coordinate_code=-1,bool null_bounds=true) {
    const auto code=coordinate_code<0 ? CoordinateCode(flags,mask) : unsigned(coordinate_code);
    auto shadow=Sample(c,Coordinate(c.type,code),revision,project,forced_face,null_bounds);
    if (flags&8) for (auto& value:shadow) value=1-value;
    std::array<unsigned,4> result{};
    for (unsigned ch=0;ch<3;++ch) {
        double sum=secondary ? 0 : Global[ch]/255.0;
        for (unsigned slot=0;slot<2;++slot) {
            const double attenuation=(flags&1) && (mask&(1U<<slot)) && (flags&(secondary ? 4U : 2U)) ? shadow[ch] : 1;
            const double cosine=slot ? 1/std::sqrt(2.0) : 1;
            sum+=secondary ? (Spec0[slot][ch]+Spec1[slot][ch])/255.0*attenuation
                           : (Diffuse[slot][ch]*cosine*attenuation+Ambient[slot][ch])/255.0;
        }
        result[ch]=unsigned(std::lround(std::clamp(sum,0.0,1.0)*255));
    }
    result[3]=unsigned(std::lround(3071/4095.0*((flags&1) && (flags&16) ? shadow[3] : 1)*255));
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

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Check(argc<=2 && (argc==1 || std::filesystem::path{argv[1]}.is_absolute()),"Expected optional absolute shader prefix");
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
            regs.vs.output_mask.Assign(127); regs.rasterizer.vs_output_total.Assign(7);
            regs.rasterizer.vs_output_attributes[3].raw = 0x1f141312; // View XYZ.
            regs.rasterizer.vs_output_attributes[4].raw = 0x0f0e0d0c; // Texcoord0/1.
            regs.rasterizer.vs_output_attributes[5].raw = 0x1f1f1716; // Texcoord2.
            regs.rasterizer.vs_output_attributes[6].raw=0x1f1f1f10; // Texcoord0 W.
            for (unsigned i = 3; i < 7; ++i)
                pica.vs_setup.UpdateProgramCode(i,Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,i,i));
            pica.vs_setup.UpdateProgramCode(7,u32(nihstro::OpCode::Id::END)<<26);
            regs.vs.max_input_attribute_index.Assign(6); regs.pipeline.max_input_attrib_index.Assign(6);
            regs.vs.input_attribute_to_register_map_low = 0x6543210;
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
            regs.texturing.main_config.texture1_enable.Assign(0);
            regs.texturing.main_config.texture2_enable.Assign(0);
            regs.texturing.texture0_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
            regs.texturing.texture1_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
            regs.texturing.texture2_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);

            auto& texture=regs.texturing.texture0;
            texture.width.Assign(32);texture.height.Assign(32);
            texture.border_color.raw=Border[0]|(Border[1]<<8)|(Border[2]<<16)|(Border[3]<<24);
            std::array<PAddr,6> addresses{};
            for (unsigned face=0;face<6;++face) {
                addresses[face]=Memory::VRAM_PADDR+0x120000+face*0x10000;
                if (!face) texture.address.Assign(addresses[face]/8);
                else regs.texturing.cube_address[face-1].Assign(addresses[face]/8);
                Check(regs.texturing.GetCubePhysicalAddress(Pica::TexturingRegs::CubeFace(face))==addresses[face],"Wrong public cube face address");
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
            // Initialize unused Fog/Proc storage publicly before SyncLUTs.
            Write(lut,PICA_REG_INDEX(texturing.fog_lut_offset),0);
            for (unsigned i=0;i<128;++i) Write(lut,PICA_REG_INDEX(texturing.fog_lut_data[0])+i%8,0);
            for (unsigned table:std::array<unsigned,5>{0,2,3,4,5}) {
                Write(lut,PICA_REG_INDEX(texturing.proctex_lut_config),table<<8);
                for (unsigned i=0;i<(table<4 ? 128U : 256U);++i)
                    Write(lut,PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8,0);
            }
            process(lut);
            for (unsigned table=0;table<24;++table) for (unsigned i=0;i<256;++i)
                Check(pica.lighting.luts[table][i].raw==(table==3 ? 3071U : 0U),"Raw public Lighting LUT mismatch");
            auto target = memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize() >= 4096,"Missing physical target VRAM");
            std::memset(target.GetPtr(),0x11,4096);
            constexpr std::array<std::array<float,4>,3> positions{{
                {-1,-1,-.5f,1},{3,-1,-.5f,1},{-1,3,-.5f,1}}};

            const auto cases=Cases();Check(cases.size()==33,"Incomplete resource case matrix");
            unsigned draws{},exports{},mutation_sensitive{},filter_sensitive{},mask_sensitive{},alpha_sensitive{};
            unsigned projection_sensitive{},face_sensitive{},bias_sensitive{},mip_sensitive{},disabled_sensitive{},null_sensitive{};
            std::array<unsigned,6> face_coverage{},active_face_coverage{};std::array<unsigned,3> mip_coverage{};
            for (unsigned revision=0;revision<2;++revision) for (unsigned resource=0;resource<cases.size();++resource) {
                const auto c=cases[resource];
                // Public physical-memory edits with explicit invalidation, including
                // color/storage reinterpretation as resource type changes.
                for (unsigned face=0;face<6;++face) {
                    auto backing=memory.GetPhysicalRef(addresses[face]);
                    Check(backing && backing.GetSize()>=TextureBytes,"Missing physical face mip storage");
                    rasterizer.InvalidateRegion(addresses[face],TextureBytes);
                    for (unsigned level=0;level<3;++level) {
                        const auto size=32U>>level;
                        for (unsigned y=0;y<size;++y) for (unsigned x=0;x<size;++x) {
                            auto rgba=Texel(face,level,x,size-1-y,revision);
                            if (c.type==C::Shadow2D || c.type==C::ShadowCube) {
                                const auto raw=ShadowWord(face,x,size-1-y,revision);
                                for (unsigned ch=0;ch<4;++ch) rgba[ch]=(raw>>(ch*8))&255;
                            }
                            for (unsigned ch=0;ch<4;++ch)
                                backing.GetPtr()[MipOffsets[level]+Morton(x,y,size)*4+3-ch]=u8(rgba[ch]);
                        }
                    }
                }
                std::vector<u32> parameters;
                Write(parameters,PICA_REG_INDEX(texturing.main_config),u32(c.enabled));
                Write(parameters,PICA_REG_INDEX(texturing.texture0)+2,(u32(c.linear)<<1)|(u32(c.linear)<<2)|
                    (c.wrap_t<<8)|(c.wrap_s<<12)|(unsigned(c.type)<<28));
                Write(parameters,PICA_REG_INDEX(texturing.texture0)+3,(2<<16)|(c.level<<24));
                Write(parameters,PICA_REG_INDEX(texturing.shadow),u32(c.orthographic)|(c.bias<<1));
                process(parameters);
                for (unsigned flags=0;flags<32;++flags) for (unsigned mask=0;mask<4;++mask)
                    for (unsigned secondary=0;secondary<2;++secondary) {
                        const auto source=secondary ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor;
                        std::vector<u32> lighting;
                        Write(lighting,PICA_REG_INDEX(lighting.max_light_index),1);
                        Write(lighting,PICA_REG_INDEX(lighting.light_enable),Physical[0]|(Physical[1]<<4));
                        Write(lighting,PICA_REG_INDEX(lighting.config0),(flags&1)|12|(8<<4)|((flags&2)<<15)|
                              ((flags&4)<<15)|((flags&8)<<15)|((flags&16)<<15));
                        const auto disabled=255^((mask&1 ? 1U<<Physical[0] : 0)|(mask&2 ? 1U<<Physical[1] : 0));
                        Write(lighting,PICA_REG_INDEX(lighting.config1),0xfff7ff00|disabled);
                        Write(lighting,PICA_REG_INDEX(texturing.tev_stage0),u32(source)|(u32(source)<<16));
                        process(lighting);
                        if (argc==2 && !revision && flags==31 && mask==3) {
                            const auto program=Metal::GeneratePicaFragmentShader(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8,
                                                                                 Metal::BuildPicaSamplerConfigs(regs));
                            std::ofstream file{std::string(argv[1])+"-r"+std::to_string(resource)+"-s"+std::to_string(secondary)+".metal"};
                            file<<program.shader.source;Check(bool(file),"Cannot export public resource counterpart");++exports;
                        }
                        const auto code=CoordinateCode(flags,mask);
                        const auto expected=Expected(c,flags,mask,secondary,revision);
                        auto nearest=c;nearest.linear=!c.linear;auto no_bias=c;no_bias.bias=0;auto base_mip=c;base_mip.level=0;
                        auto enabled=c;enabled.enabled=true;if (enabled.type==C::Disabled) enabled.type=C::Texture2D;
                        for (unsigned ch=0;ch<4;++ch) {
                            if (revision) mutation_sensitive+=std::abs(int(expected[ch])-int(Expected(c,flags,mask,secondary,0)[ch]))>3;
                            filter_sensitive+=std::abs(int(expected[ch])-int(Expected(nearest,flags,mask,secondary,revision)[ch]))>3;
                            mask_sensitive+=std::abs(int(expected[ch])-int(Expected(c,flags,mask^3,secondary,revision,true,-1,int(code))[ch]))>3;
                            projection_sensitive+=std::abs(int(expected[ch])-int(Expected(c,flags,mask,secondary,revision,false)[ch]))>3;
                            if (c.type==C::TextureCube || c.type==C::ShadowCube)
                                face_sensitive+=std::abs(int(expected[ch])-int(Expected(c,flags,mask,secondary,revision,true,int((code%6+1)%6))[ch]))>3;
                            bias_sensitive+=std::abs(int(expected[ch])-int(Expected(no_bias,flags,mask,secondary,revision)[ch]))>3;
                            mip_sensitive+=std::abs(int(expected[ch])-int(Expected(base_mip,flags,mask,secondary,revision)[ch]))>3;
                            if (!c.enabled || c.type==C::Disabled)
                                disabled_sensitive+=std::abs(int(expected[ch])-int(Expected(enabled,flags,mask,secondary,revision)[ch]))>3;
                            if (!c.enabled && c.type==C::Shadow2D)
                                null_sensitive+=std::abs(int(expected[ch])-int(Expected(c,flags,mask,secondary,revision,true,-1,int(code),false)[ch]))>3;
                        }
                        alpha_sensitive+=std::abs(int(expected[3])-int(Expected(c,flags^16,mask,secondary,revision,true,-1,int(code))[3]))>3;
                        if (c.enabled && (c.type==C::TextureCube || c.type==C::ShadowCube)) {
                            ++face_coverage[code%6];
                            if ((flags&1) && (flags&(secondary ? 4U : 2U)) && mask) ++active_face_coverage[code%6];
                        }
                        if (c.enabled && (c.type==C::TextureCube || c.type==C::Texture2D || c.type==C::Projection2D)) ++mip_coverage[c.level];
                        for (unsigned repeat=0;repeat<2;++repeat) {
                            const auto coord=Coordinate(c.type,code);std::vector<u32> triangle;
                            Write(triangle,PICA_REG_INDEX(pipeline.triangle_topology),0);
                            Write(triangle,PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index),15);
                            for (const auto& position:positions) {
                                Attribute(triangle,position);Attribute(triangle,{0,0,0,1});Attribute(triangle,{1,1,1,1});
                                Attribute(triangle,{0,0,1,0});Attribute(triangle,{float(coord[0]),float(coord[1]),.25f,.75f});
                                Attribute(triangle,{.5f,.25f,0,0});Attribute(triangle,{float(coord[2]),0,0,0});
                            }
                            process(triangle);rasterizer.FlushRegion(Memory::VRAM_PADDR,4096);
                            for (unsigned pixel=0;pixel<1024;++pixel) for (unsigned ch=0;ch<4;++ch) {
                                const auto actual=target.GetPtr()[pixel*4+3-ch];
                                if (std::abs(int(actual)-int(expected[ch]))>2) {
                                    std::cerr<<"resource="<<resource<<" type="<<unsigned(c.type)<<" enabled="<<c.enabled<<" revision="<<revision
                                        <<" flags="<<flags<<" mask="<<mask<<" secondary="<<secondary<<" repeat="<<repeat
                                        <<" pixel="<<pixel<<" channel="<<ch<<" actual="<<unsigned(actual)<<" expected="<<expected[ch]<<'\n';
                                    throw std::runtime_error("Integrated typed lighting-shadow binding/TEV/VRAM mismatch");
                                }
                            }
                            ++draws;
                        }
                    }
                std::cout<<"revision "<<revision<<" resource "<<resource<<" complete: "<<draws<<" draws\n"<<std::flush;
            }
            Check(draws==33792 && (argc<2 || exports==66),"Incomplete typed shadow binding matrix");
            Check(mutation_sensitive && filter_sensitive && mask_sensitive && alpha_sensitive && projection_sensitive &&
                  face_sensitive && bias_sensitive && mip_sensitive && disabled_sensitive && null_sensitive,"Missing resource oracle sensitivity");
            for (auto n:face_coverage) Check(n>0,"Missing cube face coverage");
            for (auto n:active_face_coverage) Check(n>0,"Missing actually shadowed cube face coverage");
            for (auto n:mip_coverage) Check(n>0,"Missing mip coverage");
            Check(!system.IsPoweredOn(),"Data harness initialized emulation");
            std::cout<<draws<<" integrated typed lighting-shadow/TEV/physical VRAM draws; all 1024 RGBA pixels; tolerance 2\n"
                <<"Oracle coverage: mutation="<<mutation_sensitive<<" filter="<<filter_sensitive<<" light-mask="<<mask_sensitive
                <<" alpha="<<alpha_sensitive<<" projection="<<projection_sensitive<<" face="<<face_sensitive
                <<" bias="<<bias_sensitive<<" mip="<<mip_sensitive<<" disabled="<<disabled_sensitive<<" null-bounds="<<null_sensitive<<'\n';
            return 0;
        } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
