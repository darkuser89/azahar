// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_draw_runtime.h"
#include "mtl_texture_runtime.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using C=Pica::TexturingRegs::TextureConfig;
using Vec=std::array<double,3>;
void Check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct FaceUV {unsigned face;double u,v;};
// Independent dominant-axis projection, using the six standard cube face bases.
FaceUV Project(Vec d) {
    const std::array<double,3> a{std::abs(d[0]),std::abs(d[1]),std::abs(d[2])};
    unsigned axis=2;if(a[0]>a[1]&&a[0]>a[2])axis=0;else if(a[1]>a[2])axis=1;
    const auto major=a[axis];Check(major>0,"Test direction must be nonzero");
    unsigned face=axis*2+(d[axis]<0);double s{},t{};
    switch(face){case 0:s=-d[2];t=-d[1];break;case 1:s=d[2];t=-d[1];break;
    case 2:s=d[0];t=d[2];break;case 3:s=d[0];t=-d[2];break;
    case 4:s=d[0];t=-d[1];break;default:s=-d[0];t=-d[1];break;}
    return {face,.5+s/(2*major),.5+t/(2*major)};
}
Vec Direction(unsigned face,double u,double v,double major=1) {
    const double s=2*u-1,t=2*v-1;
    switch(face){case 0:return {major,-t*major,-s*major};case 1:return {-major,-t*major,s*major};
    case 2:return {s*major,major,t*major};case 3:return {s*major,-major,-t*major};
    case 4:return {s*major,-t*major,major};default:return {-s*major,-t*major,-major};}
}
std::array<unsigned,4> Color(unsigned face,unsigned level,unsigned x,unsigned y) {
    return {10+face*30+level*7+x*3,20+y*11,40+x*5+y*2,128+face*10};
}
std::uint32_t ShadowPixel(unsigned face,unsigned x,unsigned y) {
    const auto depth=0x300000U+face*0x100000U+x*0x180000U+y*0x20000U;
    const auto value=30U+face*25U+x*11U+y*5U;
    return (std::min(depth,0xffffffU)<<8)|std::min(value,255U);
}
double ShadowOracle(Vec d,bool cube,bool orthographic,int bias) {
    unsigned face{};double u=d[0],v=d[1],w=d[2];
    if(cube){const auto p=Project(d);face=p.face;u=p.u;v=p.v;w=std::max({std::abs(d[0]),std::abs(d[1]),std::abs(d[2])});}
    else if(!orthographic){u/=w;v/=w;}
    const auto z=std::uint32_t(std::max(0,int(std::min(std::abs(w),1.0)*16777215.0)-bias));
    const double px=u*4-.5,py=v*4-.5;const int x=int(std::floor(px)),y=int(std::floor(py));
    const double fx=px-x,fy=py-y;
    const auto tap=[&](int xx,int yy){
        if(!cube&&(xx<0||yy<0||xx>=4||yy>=4))return 1.0;
        xx=std::clamp(xx,0,3);yy=std::clamp(yy,0,3);
        const auto pixel=ShadowPixel(face,unsigned(xx),unsigned(yy));
        return (pixel>>8)>z ? double(pixel&255)/255.0 : 0.0;
    };
    return (1-fy)*((1-fx)*tap(x,y)+fx*tap(x+1,y))+fy*((1-fx)*tap(x,y+1)+fx*tap(x+1,y+1));
}
Pica::Shader::FSConfig Config(C::TextureType type,bool orthographic,unsigned s,unsigned t) {
    using Tev=Pica::TexturingRegs::TevStageConfig;
    Pica::RegsInternal regs{};regs.lighting.disable.Assign(1);
    regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
    auto config=Pica::Shader::FSConfig{regs};config.texture.texture0_type.Assign(type);
    config.texture.shadow_texture_orthographic.Assign(orthographic);
    config.texture.requested_wrap[0]={C::WrapMode(s),C::WrapMode(t)};
    for(auto& stage:config.texture.tev_stages)stage.sources_raw=0xF000F;
    auto stage=static_cast<Tev>(config.texture.tev_stages[0]);
    stage.color_source1.Assign(Tev::Source::Texture0);stage.alpha_source1.Assign(Tev::Source::Texture0);
    config.texture.tev_stages[0].sources_raw=stage.sources_raw;return config;
}
}

int main() {
    @autoreleasepool {
        try {
            Metal::Device device;Metal::Scheduler scheduler{device};Metal::TransferRuntime transfers{device,scheduler};
            Metal::GeometryRuntime geometry{device,scheduler};Metal::TextureRuntime textures{device,scheduler};
            Metal::ShaderRuntime shaders{device};Metal::DrawRuntime draws{device,scheduler,geometry};
            Metal::StreamBuffer stream{device,scheduler,1<<20};
            auto* target_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
            target_desc.storageMode=MTLStorageModePrivate;target_desc.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
            auto target=device.CreateTexture(target_desc);
            auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=target;
            pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            Pica::RegsInternal regs{};regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask=15<<8;regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            auto state=Metal::BuildPicaDrawState(regs);
            auto* cube_desc=[[MTLTextureDescriptor alloc] init];cube_desc.textureType=MTLTextureTypeCube;
            cube_desc.pixelFormat=MTLPixelFormatRGBA8Unorm;cube_desc.width=4;cube_desc.height=4;cube_desc.depth=1;
            cube_desc.arrayLength=1;cube_desc.mipmapLevelCount=3;cube_desc.storageMode=MTLStorageModePrivate;
            cube_desc.usage=MTLTextureUsageShaderRead;auto cube=device.CreateTexture(cube_desc);
            std::array<id<MTLTexture>,6> shadows{};
            for(unsigned face=0;face<6;++face){
                for(unsigned level=0;level<3;++level){const auto dim=4U>>level;auto data=device.CreateBuffer(dim*dim*4,true);
                    auto* bytes=static_cast<unsigned char*>(data.contents);
                    for(unsigned y=0;y<dim;++y)for(unsigned x=0;x<dim;++x){auto color=Color(face,level,x,y);
                        for(unsigned c=0;c<4;++c)bytes[(y*dim+x)*4+c]=color[c];}
                    transfers.Upload(data,{0,dim*4,0},{cube,level,face,MTLRegionMake2D(0,0,dim,dim)});
                }
                auto* desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:4 height:4 mipmapped:NO];
                desc.storageMode=MTLStorageModePrivate;desc.usage=MTLTextureUsageShaderRead;shadows[face]=device.CreateTexture(desc);
                auto data=device.CreateBuffer(4*4*4,true);auto* words=static_cast<std::uint32_t*>(data.contents);
                for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)words[y*4+x]=ShadowPixel(face,x,y);
                transfers.Upload(data,{0,16,0},{shadows[face],0,0,MTLRegionMake2D(0,0,4,4)});
            }
            const auto render=[&](const Metal::CompiledFragmentShader& program,Vec coord,int bias,
                                  std::span<const Metal::TextureBinding> images,std::span<const Metal::SamplerBinding> samplers){
                auto vertices=stream.Allocate(3*sizeof(Metal::CapturedVertex));
                auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);std::memset(data,0,vertices.size);
                constexpr std::array<std::array<float,4>,3> pos{{{-1,-1,0,1},{3,-1,0,1},{-1,3,0,1}}};
                for(unsigned i=0;i<3;++i){data[i].position=pos[i];data[i].color={1,1,1,1};
                    data[i].texcoord0={float(coord[0]),float(coord[1]),0,0};data[i].clip={1,1,float(coord[2]),0};}
                auto uniforms=stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));std::memset(uniforms.data,0,uniforms.size);
                auto* fs=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);fs->depth_offset=.5f;
                fs->shadow_texture_bias=bias;fs->tex_border_color[0]={.07f,.31f,.59f,.83f};
                const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
                draws.Draw(state,program,pass,{0,0,8,8,0,1},{0,0,8,8},{vertices,3},buffers,images,samplers);
                return transfers.Readback({target,0,0,MTLRegionMake2D(0,0,8,8)},8*4);
            };
            const std::array<std::pair<unsigned,unsigned>,6> wraps{{{0,0},{1,0},{0,1},{1,1},{4,5},{5,4}}};
            unsigned cube_cases{},shadow_cases{},cube_linear{};
            for(auto [s,t]:wraps)for(unsigned level=0;level<3;++level)for(bool linear:{false,true}){
                Metal::PicaSamplerConfigs settings{};settings[0]={s,t,unsigned(linear),unsigned(linear),0,level,level};
                auto program=shaders.Fragment(Config(C::TextureCube,false,s,t),{},Metal::FragmentColorFormat::RGBA8,settings);
                Metal::Sampler sampler{textures,{C::TextureFilter(linear),C::TextureFilter(linear),C::Nearest,C::WrapMode(s),C::WrapMode(t),0,level,level,0}};
                Metal::Sampler baseline{textures,{C::TextureFilter(linear),C::TextureFilter(linear),C::Nearest,C::ClampToEdge,C::ClampToEdge,0,level,level,0}};
                const std::array<Metal::TextureBinding,1> images{{{cube,3}}};
                const std::array<Metal::SamplerBinding,1> sampling{{{sampler.Handle(),0}}},edge{{{baseline.Handle(),0}}};
                for(unsigned face=0;face<6;++face)for(const auto uv:std::array<std::array<double,2>,5>{{{.5,.5},{.01,.3},{.99,.7},{.2,.01},{.8,.99}}}){
                    const auto direction=Direction(face,uv[0],uv[1]);const auto pixels=render(*program,direction,0,images,sampling);
                    if(linear){const auto reference=render(*program,direction,0,images,edge);
                        Check(pixels==reference,"Cube border/mixed wrap changed native seamless edge sampling");++cube_linear;}
                    else {const auto p=Project(direction);const auto size=4U>>level;
                        const auto color=Color(p.face,level,std::min(size-1,unsigned(p.u*size)),std::min(size-1,unsigned(p.v*size)));
                        for(unsigned i=0;i<pixels.size();++i)Check(std::abs(int(std::to_integer<unsigned>(pixels[i]))-int(color[i%4]))<=1,
                            "Cube native nearest face/UV/mip/color oracle mismatch");}
                    ++cube_cases;
                }
            }
            for(auto [s,t]:wraps)for(bool cube_shadow:{false,true})for(bool orthographic:{false,true}){
                Metal::PicaSamplerConfigs settings{};settings[0]={s,t,1,0,1,0,0};
                auto program=shaders.Fragment(Config(cube_shadow?C::ShadowCube:C::Shadow2D,orthographic,s,t),{},Metal::FragmentColorFormat::RGBA8,settings);
                Metal::Sampler sampler{textures,{C::Linear,C::Nearest,C::Linear,C::WrapMode(s),C::WrapMode(t),0,0,0,0}};
                std::vector<Metal::TextureBinding> images;std::vector<Metal::SamplerBinding> sampling;
                for(unsigned face=0;face<(cube_shadow?6U:1U);++face){images.push_back({shadows[face],3+face});sampling.push_back({sampler.Handle(),face});}
                std::vector<Vec> coords;
                if(cube_shadow)for(unsigned face=0;face<6;++face)for(auto uv:std::array<std::array<double,2>,3>{{{.5,.5},{.01,.3},{.99,.7}}})
                    coords.push_back(Direction(face,uv[0],uv[1],.5));
                else coords={{.25,.25,.5},{-.125,.5,.5},{1.125,.5,.5},{.5,-.125,.5},{.5,1.125,.5},{.0625,.1875,.5},{.4375,.3125,-.5}};
                for(auto coord:coords)for(int bias:{0,0x100000}){
                    const auto pixels=render(*program,coord,bias,images,sampling);
                    const auto expected=int(std::lround(ShadowOracle(coord,cube_shadow,orthographic,bias)*255));
                    for(auto value:pixels)Check(std::abs(int(std::to_integer<unsigned>(value))-expected)<=2,
                        "Shadow bounded-read/depth/intensity/bilinear oracle mismatch; sampler border must not replace shadow taps");
                    ++shadow_cases;
                }
            }
            std::cout<<cube_cases<<" native PICA cube draws: all6 faces/3 mips, nearest projection oracle, border/asymmetric flags and "<<cube_linear<<" seamless-linear clamp-reference comparisons passed\n";
            std::cout<<shadow_cases<<" native PICA Shadow2D/ShadowCube draws: independent packed depth/intensity/bilinear/bounds oracles, projection/orthographic/bias/face-edge cases and border/asymmetric flags passed\n";
            std::cout<<"No app/window/ROM/emulation started\n";return 0;
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
