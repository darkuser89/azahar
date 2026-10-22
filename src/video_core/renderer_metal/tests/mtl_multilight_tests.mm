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
using L = Pica::LightingRegs;
using Vec = std::array<double, 3>;
constexpr unsigned TableCount = 32 + 24 * 256;
constexpr unsigned VariantCount = 8 * 3 * 2 * 6 * 2;
constexpr std::array<unsigned,6> Tables{0,1,3,6,5,4};
constexpr std::array<float,6> Scales{.5f,1,1,.5f,1,2};
unsigned Offset(unsigned table) { return 32 + ((table * 7) % 24) * 256; }
double Lut(unsigned table, double x) {
    if (table >= 16)
        return .1875 + (table-16)/64.0 + .5*std::clamp(1-4*std::abs(x-.5),0.0,1.0);
    return .125 + (table>=8 ? (table-8)/32.0 : table/64.0) +
           .5*std::clamp(1-2*std::abs(x-.125),0.0,1.0);
}
double Dot(Vec a, Vec b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec Unit(Vec a) {
    const auto length=std::sqrt(Dot(a,a)); Check(length>0,"Fixture requires nonzero vectors");
    for (auto& x:a) x/=length;
    return a;
}
unsigned Physical(unsigned order, unsigned slot) {
    return order==0 ? slot : order==1 ? 7-slot : (3*slot+2)%8;
}
bool TwoSided(unsigned order, unsigned slot) { return (slot+order)%2!=0; }
Vec Position(unsigned id) { return {(int(id%3)-1)*.25,(double(id%4)-1.5)*.125,.75+id*.125}; }
Vec SpotDirection() { return {double(-.6f),0,double(.8f)}; }
float DistanceBias(unsigned id) { return id*.125f-.25f; }
float DistanceScale(unsigned id) { return id%2 ? -.25f : .5f; }
double LightColor(unsigned id, unsigned channel, unsigned term, float intensity) {
    constexpr std::array<Vec,4> coefficients{{{.75,.5,.25},{.25,.5,.75},{.5,.75,.25},{.75,.25,.5}}};
    const double weight=term==0 ? (1+id%3)/32.0 : term==1 ? (id+1)/512.0 :
                        term==2 ? (1+id%2)/16.0 : (1+id%3)/32.0;
    return intensity*weight*coefficients[term][channel];
}
struct Case {
    unsigned count{},order{},domain{},selector{},output{};
};
Case Decode(unsigned index) {
    Case c{}; c.output=index%2; index/=2; c.selector=index%6; index/=6;
    c.domain=index%2; index/=2; c.order=index%3; c.count=index/3+1; return c;
}
Pica::Shader::FSConfig BuildConfig(Case c) {
    using C=Pica::TexturingRegs::TextureConfig;
    using T=Pica::TexturingRegs::TevStageConfig;
    Pica::RegsInternal regs{}; regs.lighting.disable.Assign(1);
    regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
    auto config=Pica::Shader::FSConfig{regs};
    config.lighting.enable.Assign(1); config.lighting.src_num.Assign(c.count);
    config.lighting.config.Assign(L::LightingConfig::Config7);
    config.lighting.clamp_highlights.Assign(c.order%2);
    config.lighting.enable_primary_alpha.Assign(1); config.lighting.enable_secondary_alpha.Assign(1);
    for (unsigned slot=0;slot<c.count;++slot) {
        const auto id=Physical(c.order,slot); auto& light=config.lighting.lights[slot];
        light.num.Assign(id); light.directional.Assign(id%2==0);
        light.two_sided_diffuse.Assign(TwoSided(c.order,slot));
        light.spot_atten_enable.Assign(slot%3!=0); light.dist_atten_enable.Assign(slot%3!=1);
        light.geometric_factor_0.Assign(bool(slot&1)); light.geometric_factor_1.Assign(bool(slot&2));
    }
    std::array<Pica::Shader::LutConfig*,6> luts{&config.lighting.lut_d0,&config.lighting.lut_d1,
        &config.lighting.lut_fr,&config.lighting.lut_rr,&config.lighting.lut_rg,&config.lighting.lut_rb};
    for (unsigned i=0;i<luts.size();++i) {
        luts[i]->enable.Assign(1); luts[i]->abs_input.Assign(c.domain);
        luts[i]->type.Assign(static_cast<L::LightingLutInput>((c.selector+i)%6)); luts[i]->SetScale(Scales[i]);
    }
    config.lighting.lut_sp.enable.Assign(1); config.lighting.lut_sp.abs_input.Assign(c.domain);
    config.lighting.lut_sp.type.Assign(L::LightingLutInput::SP); config.lighting.lut_sp.SetScale(.5f);
    config.texture.texture0_type.Assign(C::Texture2D);
    for (auto& stage:config.texture.tev_stages) stage.sources_raw=0xf000f;
    auto stage=static_cast<T>(config.texture.tev_stages[0]);
    stage.color_source1.Assign(T::Source::Texture0);
    stage.color_source2.Assign(c.output ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor);
    stage.color_op.Assign(T::Operation::Modulate);
    stage.alpha_source1.Assign(c.output ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor);
    config.texture.tev_stages[0].sources_raw=stage.sources_raw; config.texture.tev_stages[0].ops_raw=stage.ops_raw;
    return config;
}
struct Variant {
    Case config;
    std::shared_ptr<const Metal::CompiledFragmentShader> shader;
};
class MultiDraw {
public:
    MultiDraw(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TextureRuntime& runtime, const char* prefix)
        : geometry{device,scheduler},draws{device,scheduler,geometry},shaders{device},stream{device,scheduler,4096},sampler{runtime,{}} {
        for (unsigned index=0;index<VariantCount;++index) {
            const auto c=Decode(index); Pica::Shader::UserConfig user{}; user.use_custom_normal.Assign(1);
            auto shader=shaders.Fragment(BuildConfig(c),user,Metal::FragmentColorFormat::RGBA8);
            if (prefix) {
                std::ofstream out{std::string(prefix)+"-config"+std::to_string(index)+".metal"};
                out<<shader->program.shader.source; Check(bool(out),"Could not export multilight shader");
            }
            variants.push_back({c,std::move(shader)});
        }
        for (unsigned domain=0;domain<2;++domain) {
            const auto alignment=[device.Handle() minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatRG32Float];
            const auto bytes=TableCount*8;
            tables[domain]=device.CreateBuffer(bytes+(alignment-bytes%alignment)%alignment,true);
            std::memset(tables[domain].contents,0xcd,tables[domain].length);
            auto* values=static_cast<std::array<float,2>*>(tables[domain].contents);
            for (unsigned table=0;table<24;++table) for (unsigned i=0;i<256;++i) {
                const bool unsigned_domain=domain || table>=16;
                const double x=unsigned_domain ? i/256.0 : (i<128 ? int(i) : int(i)-256)/128.0;
                const double step=unsigned_domain ? 1/256.0 : 1/128.0;
                values[Offset(table)+i]={float(Lut(table,x)),float(Lut(table,x+step)-Lut(table,x))};
            }
            views[domain]=device.CreateBufferTexture(tables[domain],0,TableCount,MTLPixelFormatRG32Float);
            const auto* begin=static_cast<const std::byte*>(tables[domain].contents);
            snapshots[domain]={begin,begin+tables[domain].length};
        }
        auto* desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:32 height:32 mipmapped:NO];
        desc.storageMode=MTLStorageModePrivate; desc.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        target=device.CreateTexture(desc);
    }
    unsigned Verify(Metal::Surface& surface, Format format, bool reverse, Metal::TransferRuntime& transfers) {
        constexpr std::array<float,3> Intensities{.25f,1,32};
        constexpr Vec global{.03125,.0625,.125};
        unsigned count{};
        for (unsigned index=0;index<variants.size();++index) for (const float intensity:Intensities) {
            const auto& v=variants[index]; const auto c=v.config;
            auto vertices=stream.Allocate(3*sizeof(Metal::CapturedVertex)); std::memset(vertices.data,0,vertices.size);
            auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);
            constexpr std::array<std::array<float,2>,3> positions{{{-1,-1},{3,-1},{-1,3}}};
            for (unsigned i=0;i<3;++i) {
                const float u=(positions[i][0]+1)/2,w=(positions[i][1]+1)/2;
                data[i].position={positions[i][0],positions[i][1],.5f,1}; data[i].texcoord0={u,w,0,0};
                data[i].quaternion={0,0,0,1}; data[i].view={u-.5f,w-.5f,-.25f,0}; data[i].clip={1,1,1,0};
            }
            auto uniforms=stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData)); std::memset(uniforms.data,0,uniforms.size);
            auto* fs=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data); fs->depth_offset=.5f;
            for (unsigned table=0;table<24;++table) fs->lighting_lut_offset[table/4][table%4]=Offset(table);
            for (unsigned id=0;id<8;++id) {
                auto& light=fs->light_src[id]; const auto pos=Position(id),spot=SpotDirection();
                for (unsigned ch=0;ch<3;++ch) {
                    light.position[ch]=pos[ch]; light.spot_direction[ch]=spot[ch];
                    light.diffuse[ch]=LightColor(id,ch,0,intensity); light.ambient[ch]=LightColor(id,ch,1,intensity);
                    light.specular_0[ch]=LightColor(id,ch,2,intensity); light.specular_1[ch]=LightColor(id,ch,3,intensity);
                    fs->lighting_global_ambient[ch]=global[ch];
                }
                light.dist_atten_bias=DistanceBias(id); light.dist_atten_scale=DistanceScale(id);
            }
            const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
            const std::array<Metal::TextureBinding,3> images{{{views[c.domain],0},{surface.Image(),3},{surface.Image(Metal::SurfaceImage::Normal),13}}};
            const std::array<Metal::SamplerBinding,2> sampling{{{sampler.Handle(),0},{sampler.Handle(),9}}};
            auto* pass=[[MTL4RenderPassDescriptor alloc] init]; pass.colorAttachments[0].texture=target;
            pass.colorAttachments[0].loadAction=MTLLoadActionClear; pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            pass.colorAttachments[0].clearColor=MTLClearColorMake(1,0,1,0);
            Metal::PicaDrawState state{}; state.blend.color_write_mask=15;
            draws.Draw(state,*v.shader,pass,{0,0,32,32,0,1},{0,0,32,32},{vertices,3},buffers,images,sampling);
            const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},128); const auto block=Block(format);
            for (unsigned y=0;y<32;++y) for (unsigned x=0;x<32;++x) {
                const double u=(x+.5)/32,w=(y+.5)/32; const Vec view{u-.5,w-.5,-.25};
                const Vec normal{1/255.0,1/255.0,reverse ? -1.0 : 1.0};
                Vec primary=global,secondary{}; double alpha=1,first_alpha=1;
                for (unsigned slot=0;slot<c.count;++slot) {
                    const auto id=Physical(c.order,slot); auto vector=Position(id);
                    if (id%2) for (unsigned ch=0;ch<3;++ch) vector[ch]+=view[ch];
                    const auto distance=std::sqrt(Dot(vector,vector)); const auto light=Unit(vector),eye=Unit(view);
                    const Vec half{light[0]+eye[0],light[1]+eye[1],light[2]+eye[2]}; const auto h=Unit(half);
                    const double ln=Dot(light,normal),nh=Dot(normal,h);
                    const double lambert=TwoSided(c.order,slot) ? std::abs(ln) : std::max(ln,0.0);
                    const double geometry=std::min(lambert/Dot(half,half),1.0);
                    const double highlight=c.order%2 && lambert==0 ? 0 : 1;
                    // LUT and diffuse inputs belong to the same enabled light,
                    // independently of the physical light-resource permutation.
                    const bool lut_two_sided=TwoSided(c.order,slot);
                    const auto domain=[&](double value) { return !c.domain ? value :
                        lut_two_sided ? std::abs(value) : std::max(value,0.0); };
                    const std::array<double,6> inputs{nh,Dot(eye,h),Dot(normal,eye),ln,Dot(light,SpotDirection()),h[0]-normal[0]*nh};
                    const auto lookup=[&](unsigned i) { return Scales[i]*Lut(Tables[i],domain(inputs[(c.selector+i)%6])); };
                    const double spot=slot%3 ? .5*Lut(8+id,domain(inputs[4])) : 1;
                    const double dist=slot%3!=1 ? Lut(16+id,std::clamp(double(DistanceBias(id))+DistanceScale(id)*distance,0.0,1.0)) : 1;
                    const Vec reflection{lookup(3),lookup(4),lookup(5)};
                    for (unsigned ch=0;ch<3;++ch) {
                        primary[ch]+=(LightColor(id,ch,0,intensity)*lambert+LightColor(id,ch,1,intensity))*spot*dist;
                        secondary[ch]+=(LightColor(id,ch,2,intensity)*lookup(0)*(slot&1 ? geometry : 1)+
                            LightColor(id,ch,3,intensity)*lookup(1)*reflection[ch]*(slot&2 ? geometry : 1))*highlight*spot*dist;
                    }
                    alpha=std::clamp(lookup(2),0.0,1.0); if (slot==0) first_alpha=alpha;
                }
                if (c.count>1 && std::abs(alpha-first_alpha)>1/255.0) ++distinct_fresnel;
                const auto color=BlockColor(format,unsigned(u*17)/block[0],unsigned(w*11)/block[1],11);
                for (unsigned ch=0;ch<4;++ch) {
                    if (ch<3) { primary_saturated+=primary[ch]>1; secondary_saturated+=secondary[ch]>1; }
                    const auto expected=unsigned(std::lround(ch==3 ? 255*alpha : color[ch]*
                        std::clamp(c.output ? secondary[ch] : primary[ch],0.0,1.0)));
                    const auto actual=std::to_integer<u8>(pixels[(y*32+x)*4+ch]);
                    if (std::abs(int(actual)-int(expected))>2)
                        throw std::runtime_error("Multilight mismatch format="+std::to_string(unsigned(format))+
                            " reversed="+std::to_string(reverse)+" variant="+std::to_string(index)+
                            " intensity="+std::to_string(intensity)+" x="+std::to_string(x)+" y="+std::to_string(y)+
                            " channel="+std::to_string(ch)+" expected="+std::to_string(expected)+" actual="+std::to_string(actual));
                }
            }
            ++count;
        }
        return count;
    }
    void CheckCoverage() const {
        for (unsigned domain=0;domain<2;++domain)
            Check(std::memcmp(tables[domain].contents,snapshots[domain].data(),snapshots[domain].size())==0,"Multilight changed LUT values/slopes/prefix/padding");
        Check(primary_saturated && secondary_saturated && distinct_fresnel,"Matrix missed saturation or distinct first/last Fresnel");
        std::cout<<"Oracle coverage: primary saturation="<<primary_saturated<<" secondary saturation="<<secondary_saturated
            <<" distinct first/last Fresnel="<<distinct_fresnel<<" pixel-channel/case observations passed\n";
    }
private:
    Metal::GeometryRuntime geometry;
    Metal::DrawRuntime draws;
    Metal::ShaderRuntime shaders;
    Metal::StreamBuffer stream;
    Metal::Sampler sampler;
    std::vector<Variant> variants;
    std::array<id<MTLBuffer>,2> tables;
    std::array<id<MTLTexture>,2> views;
    std::array<std::vector<std::byte>,2> snapshots;
    id<MTLTexture> target;
    u64 primary_saturated{},secondary_saturated{},distinct_fresnel{};
};
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Check(argc<=2 && (argc==1 || std::filesystem::path{argv[1]}.is_absolute()),"Expected optional absolute shader prefix");
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TextureRuntime runtime{device,scheduler};
            Metal::TransferRuntime transfers{device,scheduler}; Image image;
            MultiDraw draws{device,scheduler,runtime,argc==2 ? argv[1] : nullptr}; unsigned cases{};
            for (auto format:Formats) for (bool reverse:{false,true}) {
                VideoCore::SurfaceParams params; params.width=params.height=8; params.levels=1; params.res_scale=1;
                params.pixel_format=VideoCore::PixelFormat::RGBA8; params.type=VideoCore::SurfaceType::Texture;
                VideoCore::SurfaceBase guest{params,{}};
                Maps map{image,17,11,{},Color{128,128,u8(reverse ? 0 : 255),255}}; map.material.format=format;
                EncodeMap(map.color,format,11); Metal::Surface surface{runtime,guest,&map.material}; surface.UploadCustom(&map.material,0);
                cases+=draws.Verify(surface,format,reverse,transfers);
            }
            draws.CheckCoverage();
            std::cout<<cases<<" native multilight diffuse/specular/Fresnel/TEV draws: all1024 RGBA pixels, eight custom formats/two normal signs,"
                " 576 variants=counts1..8 x three physical orders x two domains x six selector rotations x two outputs,"
                " three intensities, mixed directional/positional lights and per-light attenuation/geometry passed\n";
            std::cout<<"Independent scalar sums/triangular LUTs/last-active-light Fresnel and unchanged buffer guards passed\n";
            std::cout<<"No app/window/ROM/emulation/image decoding started; no full SyncLUT or hardware golden claimed\n";
            return 0;
        } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
