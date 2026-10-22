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
constexpr std::array<unsigned, 8> Configs{0, 1, 2, 3, 4, 5, 6, 8};
constexpr std::array<unsigned, 8> Masks{0, 1, 2, 4, 8, 16, 32, 63};
constexpr std::array<unsigned, 6> Tables{0, 1, 3, 6, 5, 4};
constexpr std::array<float, 6> Scales{.25f, .5f, 1, 2, 4, 8};
constexpr unsigned TableCount = 32 + 24 * 256;
unsigned Offset(unsigned table) { return 32 + ((table * 7) % 24) * 256; }
double Lut(unsigned table, double x) {
    return .125 + table / 64.0 + .5 * std::clamp(1 - 2 * std::abs(x - .125), 0.0, 1.0);
}
double Dot(Vec a, Vec b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec Unit(Vec a) {
    const auto length = std::sqrt(Dot(a, a));
    Check(length > 0, "Fixture requires nonzero vectors");
    for (auto& x : a) x /= length;
    return a;
}
// Independent availability matrix from the documented PICA lighting modes:
// columns D0,D1,FR,RR,RG,RB. Never call the generator's support helper.
constexpr std::array<std::array<bool, 6>, 8> Available{{
    {1,0,0,1,0,0}, {0,0,1,1,0,0}, {1,1,0,1,0,0}, {1,1,1,0,0,0},
    {1,1,0,1,1,1}, {1,0,1,1,1,1}, {1,1,1,1,0,0}, {1,1,1,1,1,1}}};
unsigned Input(unsigned config, unsigned mode, unsigned lut, unsigned rotation = 0) {
    // CP is exercised in the all-LUT Config7, other modes use NH/VH/NV/LN/SP.
    return (mode + lut + rotation) % (config == 7 ? 6 : 5);
}
Pica::Shader::FSConfig BuildConfig(unsigned config_index, unsigned mode, unsigned geometry,
                                  unsigned rotation = 0, int domain = -1) {
    using C = Pica::TexturingRegs::TextureConfig;
    using T = Pica::TexturingRegs::TevStageConfig;
    Pica::RegsInternal regs{}; regs.lighting.disable.Assign(1);
    regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
    auto config = Pica::Shader::FSConfig{regs};
    config.lighting.enable.Assign(1); config.lighting.src_num.Assign(1);
    config.lighting.config.Assign(static_cast<L::LightingConfig>(Configs[config_index]));
    auto& light = config.lighting.lights[0]; light.num.Assign(0); light.directional.Assign(1);
    light.two_sided_diffuse.Assign(bool(mode & 4));
    light.geometric_factor_0.Assign(bool(geometry & 1)); light.geometric_factor_1.Assign(bool(geometry & 2));
    config.lighting.clamp_highlights.Assign(bool(mode & 2));
    const auto alpha = (mode + geometry) % 4;
    config.lighting.enable_primary_alpha.Assign(bool(alpha & 1));
    config.lighting.enable_secondary_alpha.Assign(bool(alpha & 2));
    std::array<Pica::Shader::LutConfig*, 6> luts{&config.lighting.lut_d0, &config.lighting.lut_d1,
        &config.lighting.lut_fr, &config.lighting.lut_rr, &config.lighting.lut_rg, &config.lighting.lut_rb};
    for (unsigned i = 0; i < luts.size(); ++i) {
        luts[i]->enable.Assign(bool(Masks[mode] & (1U << i)));
        luts[i]->abs_input.Assign(domain < 0 ? bool(mode & 1) : bool(domain));
        luts[i]->type.Assign(static_cast<L::LightingLutInput>(Input(config_index, mode, i, rotation)));
        luts[i]->SetScale(Scales[(mode + i) % Scales.size()]);
    }
    config.texture.texture0_type.Assign(C::Texture2D);
    for (auto& stage : config.texture.tev_stages) stage.sources_raw = 0xf000f;
    auto stage = static_cast<T>(config.texture.tev_stages[0]);
    stage.color_source1.Assign(T::Source::Texture0); stage.color_source2.Assign(T::Source::SecondaryFragmentColor);
    stage.color_op.Assign(T::Operation::Modulate);
    stage.alpha_source1.Assign(mode & 4 ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor);
    config.texture.tev_stages[0].sources_raw = stage.sources_raw;
    config.texture.tev_stages[0].ops_raw = stage.ops_raw;
    return config;
}
struct Variant {
    unsigned config{}, mode{}, geometry{}, rotation{};
    int domain{-1};
    std::shared_ptr<const Metal::CompiledFragmentShader> shader;
};

class SpecularDraw {
public:
    SpecularDraw(Metal::Device& device, Metal::Scheduler& scheduler, Metal::TextureRuntime& runtime,
                 const char* prefix)
        : geometry{device, scheduler}, draws{device, scheduler, geometry}, shaders{device},
          stream{device, scheduler, 4096}, sampler{runtime, {}} {
        for (unsigned index = 0; index < 304; ++index) {
            Variant v{};
            if (index < 256) {
                v.config = index / 32; v.mode = (index / 4) % 8; v.geometry = index % 4;
            } else {
                const auto extra = index - 256;
                v.config = 7; v.mode = 7; v.geometry = extra % 4;
                v.domain = (extra / 4) % 2; v.rotation = extra / 8;
            }
            Pica::Shader::UserConfig user{}; user.use_custom_normal.Assign(1);
            v.shader = shaders.Fragment(BuildConfig(v.config,v.mode,v.geometry,v.rotation,v.domain),user,
                                        Metal::FragmentColorFormat::RGBA8);
            if (prefix) {
                std::ofstream output{std::string(prefix) + "-config" + std::to_string(index) + ".metal"};
                output << v.shader->program.shader.source; Check(bool(output), "Could not export specular shader");
            }
            variants.push_back(std::move(v));
        }
        for (unsigned domain = 0; domain < 2; ++domain) {
            const auto alignment = [device.Handle() minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatRG32Float];
            const auto bytes = TableCount * 8;
            tables[domain] = device.CreateBuffer(bytes + (alignment - bytes % alignment) % alignment, true);
            std::memset(tables[domain].contents, 0xcd, tables[domain].length);
            auto* values = static_cast<std::array<float, 2>*>(tables[domain].contents);
            for (unsigned table = 0; table < 24; ++table) for (unsigned i = 0; i < 256; ++i) {
                const double x = domain ? i / 256.0 : (i < 128 ? int(i) : int(i) - 256) / 128.0;
                const double step = domain ? 1 / 256.0 : 1 / 128.0;
                values[Offset(table) + i] = {float(Lut(table, x)), float(Lut(table, x + step) - Lut(table, x))};
            }
            views[domain] = device.CreateBufferTexture(tables[domain], 0, TableCount, MTLPixelFormatRG32Float);
            const auto* begin = static_cast<const std::byte*>(tables[domain].contents);
            snapshots[domain] = {begin, begin + tables[domain].length};
        }
        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                     width:32 height:32 mipmapped:NO];
        desc.storageMode = MTLStorageModePrivate; desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        target = device.CreateTexture(desc);
    }
    unsigned Verify(Metal::Surface& surface, Format format, bool reverse, Metal::TransferRuntime& transfers) {
        constexpr Vec spec0{.75, .25, .5}, spec1{.375, .625, .25};
        unsigned count{};
        for (unsigned index = 0; index < variants.size(); ++index) for (unsigned sample = 0; sample < 3; ++sample) {
            const auto& v = variants[index];
            const Vec light_position = sample == 2 ? Vec{0, 0, 2} : Vec{.375, -.25, .5};
            auto vertices = stream.Allocate(3 * sizeof(Metal::CapturedVertex));
            std::memset(vertices.data, 0, vertices.size);
            auto* data = static_cast<Metal::CapturedVertex*>(vertices.data);
            constexpr std::array<std::array<float, 2>, 3> positions{{{-1,-1}, {3,-1}, {-1,3}}};
            for (unsigned i = 0; i < 3; ++i) {
                const float u = (positions[i][0] + 1) / 2, w = (positions[i][1] + 1) / 2;
                data[i].position = {positions[i][0], positions[i][1], .5f, 1};
                data[i].texcoord0 = {u, w, 0, 0}; data[i].quaternion = {0, 0, 0, 1};
                data[i].view = sample == 2 ? std::array<float,4>{.03125f, 0, -1, 0} :
                    std::array<float,4>{u - .5f, w - .5f, sample ? -.25f : .5f, 0};
                data[i].clip = {1, 1, 1, 0};
            }
            auto uniforms = stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
            std::memset(uniforms.data, 0, uniforms.size);
            auto* fs = static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
            fs->depth_offset = .5f;
            for (unsigned table = 0; table < 24; ++table) fs->lighting_lut_offset[table / 4][table % 4] = Offset(table);
            auto& light = fs->light_src[0];
            for (unsigned c = 0; c < 3; ++c) {
                light.position[c] = light_position[c]; light.specular_0[c] = spec0[c]; light.specular_1[c] = spec1[c];
            }
            light.spot_direction = {0, 0, 1};
            const std::array<Metal::BufferBinding, 1> buffers{{{uniforms.buffer, uniforms.offset, 2}}};
            const std::array<Metal::TextureBinding, 3> images{{{views[v.domain < 0 ? bool(v.mode & 1) : bool(v.domain)], 0},
                {surface.Image(), 3}, {surface.Image(Metal::SurfaceImage::Normal), 13}}};
            const std::array<Metal::SamplerBinding, 2> sampling{{{sampler.Handle(), 0}, {sampler.Handle(), 9}}};
            auto* pass = [[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 1, 0);
            Metal::PicaDrawState state{}; state.blend.color_write_mask = 15;
            draws.Draw(state, *v.shader, pass, {0,0,32,32,0,1}, {0,0,32,32}, {vertices,3}, buffers, images, sampling);
            const auto pixels = transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)}, 128);
            const auto block = Block(format);
            for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 32; ++x) {
                const double u = (x + .5) / 32, w = (y + .5) / 32;
                const Vec normal{1 / 255.0, 1 / 255.0, reverse ? -1.0 : 1.0};
                const auto l = Unit(light_position);
                const auto view = Unit(sample == 2 ? Vec{.03125,0,-1} : Vec{u-.5,w-.5,sample ? -.25 : .5});
                const Vec half{l[0]+view[0],l[1]+view[1],l[2]+view[2]};
                const auto h = Unit(half);
                const double ln = Dot(l,normal), nh = Dot(normal,h);
                const double lambert = v.mode & 4 ? std::abs(ln) : std::max(ln,0.0);
                const double geo = std::min(lambert / Dot(half,half),1.0);
                const std::array<double,6> inputs{nh,Dot(view,h),Dot(normal,view),ln,l[2],h[0]-normal[0]*nh};
                const auto enabled = [&](unsigned i) { return Available[v.config][i] && (Masks[v.mode] & (1U << i)); };
                const auto lookup = [&](unsigned i) {
                    double input = inputs[Input(v.config,v.mode,i,v.rotation)];
                    if (v.domain < 0 ? bool(v.mode & 1) : bool(v.domain)) input = v.mode & 4 ? std::abs(input) : std::max(input,0.0);
                    return Scales[(v.mode+i)%Scales.size()] * Lut(Tables[i],input);
                };
                const double d0 = enabled(0) ? lookup(0) : 1, d1 = enabled(1) ? lookup(1) : 1;
                const double rr = enabled(3) ? lookup(3) : 1;
                const Vec reflect{rr,enabled(4) ? lookup(4) : rr,enabled(5) ? lookup(5) : rr};
                const auto alpha_bits = (v.mode+v.geometry)%4;
                const bool alpha_enabled = v.mode & 4 ? alpha_bits & 2 : alpha_bits & 1;
                const double alpha = enabled(2) && alpha_enabled ? std::clamp(lookup(2),0.0,1.0) : 1;
                const double highlight = (v.mode & 2) && lambert == 0 ? 0 : 1;
                const auto color = BlockColor(format,unsigned(u*17)/block[0],unsigned(w*11)/block[1],9);
                for (unsigned c = 0; c < 4; ++c) {
                    const auto expected = unsigned(std::lround(c == 3 ? 255*alpha : color[c] * std::clamp(highlight *
                        (spec0[c]*d0*(v.geometry&1 ? geo : 1) + spec1[c]*d1*reflect[c]*(v.geometry&2 ? geo : 1)),0.0,1.0)));
                    const auto actual = std::to_integer<u8>(pixels[(y*32+x)*4+c]);
                    if (std::abs(int(actual)-int(expected)) > 2)
                        throw std::runtime_error("Specular LUT mismatch format="+std::to_string(unsigned(format))+
                            " reversed="+std::to_string(reverse)+" variant="+std::to_string(index)+
                            " sample="+std::to_string(sample)+" x="+std::to_string(x)+" y="+std::to_string(y)+
                            " channel="+std::to_string(c)+" expected="+std::to_string(expected)+" actual="+std::to_string(actual));
                }
            }
            ++count;
        }
        return count;
    }
    void CheckTables() const {
        for (unsigned i = 0; i < 2; ++i)
            Check(std::memcmp(tables[i].contents,snapshots[i].data(),snapshots[i].size())==0,
                  "Specular draws changed LUT values/slopes/prefix/padding");
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
};
}
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc<=2 && (argc==1 || std::filesystem::path{argv[1]}.is_absolute()),"Expected optional absolute shader prefix");
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TextureRuntime runtime{device,scheduler};
            Metal::TransferRuntime transfers{device,scheduler}; Image image;
            SpecularDraw draws{device,scheduler,runtime,argc==2 ? argv[1] : nullptr};
            unsigned cases{};
            for (auto format : Formats) for (bool reverse : {false,true}) {
                VideoCore::SurfaceParams params; params.width=params.height=8; params.levels=1; params.res_scale=1;
                params.pixel_format=VideoCore::PixelFormat::RGBA8; params.type=VideoCore::SurfaceType::Texture;
                VideoCore::SurfaceBase guest{params,{}};
                Maps map{image,17,11,{},Color{128,128,u8(reverse ? 0 : 255),255}};
                map.material.format=format; EncodeMap(map.color,format,9);
                Metal::Surface surface{runtime,guest,&map.material}; surface.UploadCustom(&map.material,0);
                cases+=draws.Verify(surface,format,reverse,transfers);
            }
            draws.CheckTables();
            std::cout<<cases<<" native specular/Fresnel/TEV draws: all1024 RGBA pixels, eight custom color formats/two normal signs,"
                " 304 variants: 256 over eight configs/eight masks/four geometry pairs, plus48 all-LUT Config7 cases,"
                " six selectors x two domains x four geometry pairs; scales distributed, highlight clamp/both Fresnel alpha outputs passed\n";
            std::cout<<"Independent closed-form LUT and per-pixel vector/color oracles, negative slopes/shuffled24 tables/unchanged buffer guards passed\n";
            std::cout<<"No app/window/ROM/emulation/image decoding started; no full Rasterizer SyncLUT or hardware golden claimed\n";
            return 0;
        } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
