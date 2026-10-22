// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_filter.h"
#include "mtl_compiler_gate_fixture.h"
#include "mtl_texture_runtime.h"
#include "common/settings.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
using Color=std::array<double,4>;
Color Pixel(const std::vector<u8>& data,unsigned width,unsigned height,int x,int y) {
    x=std::clamp(x,0,int(width)-1); y=std::clamp(y,0,int(height)-1);
    Color result{};
    for(unsigned c=0;c<4;++c) result[c]=data[(y*width+x)*4+c]/255.0;
    return result;
}
double Spline(double x) {
    x=std::abs(x);
    return x<1 ? (4-6*x*x+3*x*x*x)/6 : x<2 ? std::pow(2-x,3)/6 : 0;
}
Color Bicubic(const std::vector<u8>& data,unsigned width,unsigned height,double u,double v) {
    const double px=u*width-.5,py=v*height-.5;
    const int x=int(std::floor(px)),y=int(std::floor(py));
    Color result{};
    // Independent separable 16-tap B-spline oracle, rather than the shader's
    // four bilinear samples and optimized coordinate/weight construction.
    for(int dy=-1;dy<=2;++dy) for(int dx=-1;dx<=2;++dx) {
        const auto p=Pixel(data,width,height,x+dx,y+dy);
        const double weight=Spline(px-x-dx)*Spline(py-y-dy);
        for(unsigned c=0;c<4;++c) result[c]+=p[c]*weight;
    }
    return result;
}
Color Linear(const std::vector<u8>& data,unsigned width,unsigned height,double u,double v) {
    const double px=u*width-.5,py=v*height-.5;
    const int x=int(std::floor(px)),y=int(std::floor(py)); const double wx=px-x,wy=py-y;
    Color result{};
    for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
        const auto p=Pixel(data,width,height,x+dx,y+dy);
        const double weight=(dx?wx:1-wx)*(dy?wy:1-wy);
        for(unsigned c=0;c<4;++c) result[c]+=p[c]*weight;
    }
    return result;
}
std::vector<double> Sobel(const std::vector<u8>& data,unsigned width,unsigned height) {
    constexpr int kernel[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    std::vector<double> result(width*height);
    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
        double gx{},gy{};
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            const auto p=Pixel(data,width,height,int(x)+dx,int(y)+dy);
            const double luminance=.2627*p[0]+.6780*p[1]+.0593*p[2];
            gx+=luminance*kernel[dy+1][dx+1]; gy+=luminance*kernel[dx+1][dy+1];
        }
        result[y*width+x]=1.0f-float(std::hypot(gx,gy));
    }
    return result;
}
std::vector<Color> Anime(const std::vector<u8>& data,const std::vector<double>& sobel,unsigned width,unsigned height,double u,double v) {
    const auto gradient=[&](double s,double t) {
        const double px=s*width-.5,py=t*height-.5;
        const int x=int(std::floor(px)),y=int(std::floor(py)); const double wx=px-x,wy=py-y;
        double value{};
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            value+=sobel[std::clamp(y+dy,0,int(height)-1)*width+std::clamp(x+dx,0,int(width)-1)]*
                   (dx?wx:1-wx)*(dy?wy:1-wy);
        return value;
    };
    const auto center=Linear(data,width,height,u,v); const double center_gradient=gradient(u,v);
    // A bounded numerical oracle: preserve strict priority away from Float32
    // gradient ties, while admitting either branch within one-millionth. RGB
    // and alpha must match the same candidate, never independently chosen taps.
    constexpr double epsilon=1e-6;
    if(center_gradient>.4+epsilon) return {center};
    std::vector<Color> candidates;
    if(center_gradient>.4-epsilon) candidates.push_back(center);
    struct Point { int x,y; };
    struct Kernel { std::array<Point,3> light,dark; bool compare_center; };
    constexpr std::array<Kernel,8> kernels{{
        {{{{-1,-1},{0,-1},{1,-1}}},{{{1,1},{0,1},{-1,1}}},true},
        {{{{1,1},{0,1},{-1,1}}},{{{-1,-1},{0,-1},{1,-1}}},true},
        {{{{1,0},{0,-1},{1,-1}}},{{{0,0},{-1,0},{0,1}}},false},
        {{{{-1,1},{-1,0},{0,1}}},{{{0,0},{1,0},{0,-1}}},false},
        {{{{1,0},{1,1},{1,-1}}},{{{-1,0},{-1,-1},{-1,1}}},true},
        {{{{-1,0},{-1,-1},{-1,1}}},{{{1,0},{1,1},{1,-1}}},true},
        {{{{1,0},{1,1},{0,1}}},{{{0,0},{-1,0},{0,-1}}},false},
        {{{{0,-1},{-1,0},{-1,-1}}},{{{0,0},{1,0},{0,1}}},false},
    }};
    for(const auto& kernel:kernels) {
        double light=1e100,dark=-1e100;
        for(const auto p:kernel.light) light=std::min(light,gradient(u+double(p.x)/width,v+double(p.y)/height));
        for(const auto p:kernel.dark) dark=std::max(dark,gradient(u+double(p.x)/width,v+double(p.y)/height));
        if(light<dark-epsilon || (kernel.compare_center&&light<center_gradient-epsilon)) continue;
        auto result=center;
        for(auto& c:result)c*=.4;
        for(const auto p:kernel.light) {
            const auto color=Linear(data,width,height,u+double(p.x)/width,v+double(p.y)/height);
            for(unsigned c=0;c<4;++c)result[c]+=.2*color[c];
        }
        candidates.push_back(result);
        if(light>dark+epsilon && (!kernel.compare_center||light>center_gradient+epsilon)) return candidates;
    }
    candidates.push_back(center);
    return candidates;
}
std::vector<std::byte> Read(Metal::TransferRuntime& transfer,id<MTLTexture> texture,unsigned level=0,unsigned slice=0) {
    const auto width=std::max<NSUInteger>(1,texture.width>>level),height=std::max<NSUInteger>(1,texture.height>>level);
    return transfer.Readback({texture,level,slice,MTLRegionMake2D(0,0,width,height)},width*4);
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            using F=Settings::TextureFilter;
            Metal::Device device; Metal::Scheduler scheduler{device}; Metal::TransferRuntime transfer{device,scheduler};
            Metal::TextureFilterRuntime filters{device,scheduler};
            const auto make=[&](unsigned width,unsigned height,unsigned levels=1,unsigned layers=1) {
                auto* d=[[MTLTextureDescriptor alloc] init];
                d.textureType=layers>1?MTLTextureType2DArray:MTLTextureType2D;
                d.pixelFormat=MTLPixelFormatRGBA8Unorm; d.width=width; d.height=height;
                d.mipmapLevelCount=levels; d.arrayLength=layers; d.storageMode=MTLStorageModePrivate;
                d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;
                return device.CreateTexture(d);
            };
            const auto upload=[&](id<MTLTexture> texture,const std::vector<u8>& data,unsigned width,unsigned height,
                                  unsigned level=0,unsigned slice=0) {
                auto buffer=device.CreateBuffer(data.size(),true); std::memcpy(buffer.contents,data.data(),data.size());
                transfer.Upload(buffer,{0,width*4,0},{texture,level,slice,MTLRegionMake2D(0,0,width,height)});
            };
            auto* cube_descriptor=[[MTLTextureDescriptor alloc] init];
            cube_descriptor.textureType=MTLTextureTypeCubeArray;
            cube_descriptor.pixelFormat=MTLPixelFormatRGBA8Unorm;
            cube_descriptor.width=cube_descriptor.height=8;
            cube_descriptor.arrayLength=2;cube_descriptor.mipmapLevelCount=2;
            cube_descriptor.storageMode=MTLStorageModePrivate;
            cube_descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;
            const auto cube=device.CreateTexture(cube_descriptor),cube_target=device.CreateTexture(cube_descriptor),
                       cube_output=make(11,7);
            for (unsigned level=0;level<2;++level) for (unsigned face=0;face<12;++face) {
                const unsigned side=8u>>level;
                const std::array<unsigned,4> expected{17u+face*13u,29u+level*83u,211u-face*11u,53u+face*7u};
                transfer.ClearColor(cube,MTLClearColorMake(expected[0]/255.,expected[1]/255.,
                    expected[2]/255.,expected[3]/255.),level,face);
                filters.Filter(F::Bicubic,{cube,level,face,MTLRegionMake2D(0,0,side,side)},
                    {cube_output,0,0,MTLRegionMake2D(0,0,11,7)},2);
                const auto pixels=Read(transfer,cube_output);
                for (unsigned i=0;i<pixels.size();++i)
                    Check(std::abs(int(std::to_integer<unsigned>(pixels[i]))-int(expected[i%4]))<=1,
                          "Native filter selected the wrong cube-array face or mip");
                transfer.ClearColor(cube_target,MTLClearColorMake(0,0,0,0),level,11-face);
                filters.Filter(F::Bicubic,{cube_output,0,0,MTLRegionMake2D(0,0,11,7)},
                    {cube_target,level,11-face,MTLRegionMake2D(0,0,side,side)},2);
                const auto target_pixels=Read(transfer,cube_target,level,11-face);
                for (unsigned i=0;i<target_pixels.size();++i)
                    Check(std::abs(int(std::to_integer<unsigned>(target_pixels[i]))-int(expected[i%4]))<=1,
                          "Native filter wrote the wrong cube-array face or mip");
            }
            std::cout<<"48 native cube-array source/destination face/mip filter cases preserve all RGBA pixels\n";
            transfer.ClearColor(cube_output,MTLClearColorMake(17/255.,34/255.,51/255.,68/255.));
            const auto before_rejections=Read(transfer,cube_output);
            for (unsigned failure=0;failure<6;++failure) {
                auto* d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                    width:8 height:8 mipmapped:NO];
                d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead;
                if (failure==0) { d.textureType=MTLTextureType1D;d.height=1; }
                if (failure==1) { d.textureType=MTLTextureType3D;d.depth=2; }
                if (failure==2) { d.textureType=MTLTextureType2DMultisample;d.sampleCount=4;
                    d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget; }
                if (failure==3) d.pixelFormat=MTLPixelFormatR8Uint;
                if (failure==4) d.pixelFormat=MTLPixelFormatDepth32Float;
                if (failure==5) { d.storageMode=MTLStorageModeMemoryless;d.usage=MTLTextureUsageRenderTarget; }
                const auto invalid=device.CreateTexture(d);
                bool rejected{};
                try {
                    const Metal::TextureRegion bad{invalid,0,0,MTLRegionMake2D(0,0,8,failure==0?1:8)};
                    const Metal::TextureRegion good{cube_output,0,0,MTLRegionMake2D(0,0,11,7)};
                    filters.Filter(F::Bicubic,failure==5?good:bad,failure==5?bad:good,2);
                } catch(const std::invalid_argument&) { rejected=true; }
                Check(rejected,"Native filter failed to reject a texture type/sample/storage/format before GPU encoding");
                Check(Read(transfer,cube_output)==before_rejections,
                      "Native filter rejection modified destination pixels");
            }
            std::cout<<"Six native filter preflight rejections preserve the valid guard image: 1D/3D/MSAA/uint/depth/memoryless\n";
            for (bool destination:{false,true}) for (bool invalid_level:{false,true}) {
                bool rejected{};
                const Metal::TextureRegion bad{cube,invalid_level?2U:0U,invalid_level?0U:12U,
                                              MTLRegionMake2D(0,0,1,1)};
                const Metal::TextureRegion good{cube_output,0,0,MTLRegionMake2D(0,0,11,7)};
                try { filters.Filter(F::Bicubic,destination?good:bad,destination?bad:good,2); }
                catch(const std::invalid_argument&) { rejected=true; }
                Check(rejected && Read(transfer,cube_output)==before_rejections,
                      "Native cube-array filter accepted an invalid source/destination mip/slice");
            }
            std::cout<<"Four invalid cube-array source/destination mip/slice boundaries reject before encoding\n";
            unsigned fixture_count{};
            if(argc>1) {
                const std::filesystem::path output{argv[1]}; std::filesystem::create_directories(output);
                for(unsigned pass=0;pass<7;++pass) {
                    auto shader=Metal::GenerateTextureFilterShader(Metal::TextureFilterPass(pass));
                    std::ofstream file{output/("filter_"+std::to_string(pass)+".metal")}; file<<shader.source;
                    Check(file.good(),"Unable to write native filter fixture"); ++fixture_count;
                }
                std::ofstream file{output/"filter_vertex.metal"}; file<<Metal::TextureFilterVertexSource;
                Check(file.good(),"Unable to write native filter vertex fixture"); ++fixture_count;
            }
            unsigned bicubic_cases{};
            for(unsigned pattern=0;pattern<3;++pattern) for(unsigned scale:{2U,3U})
                for(bool flip_x:{false,true}) for(bool flip_y:{false,true}) {
                    constexpr unsigned sw=7,sh=5,dw=25,dh=20;
                    auto source=make(sw,sh),dest=make(dw,dh);
                    std::vector<u8> data(sw*sh*4);
                    for(unsigned y=0;y<sh;++y) for(unsigned x=0;x<sw;++x) for(unsigned c=0;c<4;++c)
                        data[(y*sw+x)*4+c]=pattern==0?u8(31+c*47):pattern==1?u8(x*29+y*13+c*19):u8(((x+y+c)%2)*255);
                    upload(source,data,sw,sh); transfer.ClearColor(dest,MTLClearColorMake(17/255.,34/255.,51/255.,68/255.));
                    const auto region=MTLRegionMake2D(3,2,5*scale,3*scale);
                    Check(filters.Filter(F::Bicubic,{source,0,0,MTLRegionMake2D(1,1,5,3)},{dest,0,0,region},scale,flip_x,flip_y),
                          "Native bicubic filter was not selected");
                    const auto pixels=Read(transfer,dest);
                    for(unsigned y=0;y<dh;++y) for(unsigned x=0;x<dw;++x) {
                        Color expected{17/255.,34/255.,51/255.,68/255.};
                        if(x>=3&&x<3+5*scale&&y>=2&&y<2+3*scale) {
                            double u=(x-3+.5)/(5*scale),v=(y-2+.5)/(3*scale);
                            if(flip_x)u=1-u; if(flip_y)v=1-v;
                            expected=Bicubic(data,sw,sh,(1+5*u)/sw,(1+3*v)/sh);
                        }
                        for(unsigned c=0;c<4;++c) {
                            const int wanted=std::lround(expected[c]*255),actual=std::to_integer<unsigned>(pixels[(y*dw+x)*4+c]);
                            if(std::abs(actual-wanted)>1) throw std::runtime_error("Native bicubic B-spline/guard mismatch pattern="+
                                std::to_string(pattern)+" x="+std::to_string(x)+" y="+std::to_string(y)+" expected="+
                                std::to_string(wanted)+" actual="+std::to_string(actual));
                        }
                    }
                    ++bicubic_cases;
                }
            std::cout<<bicubic_cases<<" native Bicubic cases: independent 16-tap B-spline, odd extents, crop/scale, both flips, alpha and destination guards passed\n";
            unsigned anime_cases{},float_tie_pixels{};
            for(unsigned pattern=0;pattern<2;++pattern) for(unsigned scale:{2U,3U})
                for(bool flip_x:{false,true}) for(bool flip_y:{false,true}) {
                    auto source=make(8,8),dest=make(8*scale,8*scale);
                    std::vector<u8> data(8*8*4);
                    for(unsigned y=0;y<8;++y) for(unsigned x=0;x<8;++x) {
                        const bool white=pattern ? x==3||y==4 : x>=y;
                        for(unsigned c=0;c<3;++c)data[(y*8+x)*4+c]=white?255:0;
                        data[(y*8+x)*4+3]=u8(17+x*19+y*5);
                    }
                    upload(source,data,8,8);
                    filters.Filter(F::Anime4K,{source,0,0,MTLRegionMake2D(0,0,8,8)},
                                   {dest,0,0,MTLRegionMake2D(0,0,8*scale,8*scale)},scale,flip_x,flip_y);
                    const auto gradient=Sobel(data,8,8);
                    const auto pixels=Read(transfer,dest);
                    for(unsigned y=0;y<8*scale;++y) for(unsigned x=0;x<8*scale;++x) {
                        double u=(x+.5)/(8*scale),v=(y+.5)/(8*scale); if(flip_x)u=1-u; if(flip_y)v=1-v;
                        const auto candidates=Anime(data,gradient,8,8,u,v);
                        if(candidates.size()>1) ++float_tie_pixels;
                        bool matches{};
                        for(const auto& expected:candidates) {
                            bool candidate_matches=true;
                            for(unsigned c=0;c<4;++c) {
                                const int wanted=std::lround(expected[c]*255),actual=std::to_integer<unsigned>(pixels[(y*8*scale+x)*4+c]);
                                candidate_matches&=std::abs(wanted-actual)<=2;
                            }
                            matches|=candidate_matches;
                        }
                        if(!matches) throw std::runtime_error("Native Anime4K Sobel/refine mismatch outside bounded Float32 ties pattern="+
                            std::to_string(pattern)+" scale="+std::to_string(scale)+" x="+std::to_string(x)+" y="+std::to_string(y));
                    }
                    ++anime_cases;
                }
            std::cout<<anime_cases<<" native Anime4K three-pass cases: independent Sobel convolution and priority-kernel reference, diagonals/crosses, both flips, scale2/3 and variable alpha passed; "<<float_tie_pixels<<" pixel comparisons admitted bounded Float32 gradient ties\n";
            unsigned filter_cases{};
            std::array<std::vector<std::byte>,5> patterned;
            for(unsigned filter=1;filter<=5;++filter) for(bool solid:{false,true}) {
                auto source=make(8,8),dest=make(24,24);
                std::vector<u8> data(8*8*4);
                for(unsigned y=0;y<8;++y) for(unsigned x=0;x<8;++x) {
                    const bool edge=x>=y;
                    const std::array<u8,4> color=solid?std::array<u8,4>{41,83,127,197}:
                        edge?std::array<u8,4>{255,31,83,255}:std::array<u8,4>{13,211,71,127};
                    std::copy(color.begin(),color.end(),data.begin()+(y*8+x)*4);
                }
                upload(source,data,8,8);
                auto fresh = std::make_unique<Metal::TextureFilterRuntime>(device,scheduler);
                scheduler.Finish();
                Metal::Test::CompilerGate gate{device};
                const auto compiled = device.DiskCacheStats().compiled_pipelines;
                const auto before = scheduler.WorkerStats();
                Check(fresh->Filter(F(filter),{source,0,0,MTLRegionMake2D(0,0,8,8)},
                                     {dest,0,0,MTLRegionMake2D(0,0,24,24)},3),"A native filter returned a CPU fallback");
                const auto held = scheduler.WorkerStats();
                const bool no_compile = device.DiskCacheStats().compiled_pipelines == compiled;
                const bool timely = gate.Open();
                Check(timely && no_compile && held.queued_draws == before.queued_draws +
                          (F(filter) == F::Anime4K ? 3 : 1) && held.encoded_draws == before.encoded_draws,
                      "First filter blocked/compiled on its owner, dropped a pass or encoded before its PSO");
                fresh.reset(); // Queued argument buffers, samplers and pipelines own their remaining lifetime.
                const auto pixels=Read(transfer,dest);
                Check(filters.Filter(F(filter),{source,0,0,MTLRegionMake2D(0,0,8,8)},
                                     {dest,0,0,MTLRegionMake2D(0,0,24,24)},3),"Reference filter failed");
                Check(pixels == Read(transfer,dest), "Deferred filter output changed after runtime destruction");
                if(solid) for(unsigned byte=0;byte<pixels.size();++byte)
                    Check(std::abs(int(std::to_integer<unsigned>(pixels[byte]))-int(std::array<unsigned,4>{41,83,127,197}[byte%4]))<=1,
                          "Native filter changed a solid color or alpha");
                else patterned[filter-1]=pixels;
                ++filter_cases;
            }
            for(unsigned a=0;a<5;++a) for(unsigned b=a+1;b<5;++b)
                Check(patterned[a]!=patterned[b],"Distinct native filter algorithms silently used the same output path");
            std::cout<<filter_cases<<" compiler-gated native filter cases: all5 algorithms, owned lifetime, complete repeated output, solid RGBA invariance and distinct nontrivial edge outputs passed\n";
            // A selected mip/slice must not sample level0 or another array layer.
            auto layered=make(16,12,3,3),dest=make(17,13);
            std::vector<u8> mip(8*6*4);
            for(unsigned y=0;y<6;++y) for(unsigned x=0;x<8;++x) for(unsigned c=0;c<4;++c) mip[(y*8+x)*4+c]=u8(x*23+y*17+c*31);
            upload(layered,mip,8,6,1,2);
            filters.Filter(F::Bicubic,{layered,1,2,MTLRegionMake2D(0,0,8,6)},{dest,0,0,MTLRegionMake2D(0,0,17,13)},2);
            const auto pixels=Read(transfer,dest);
            for(unsigned y=0;y<13;++y) for(unsigned x=0;x<17;++x) {
                const auto expected=Bicubic(mip,8,6,(x+.5)/17,(y+.5)/13);
                for(unsigned c=0;c<4;++c) Check(std::abs(int(std::to_integer<unsigned>(pixels[(y*17+x)*4+c]))-
                    int(std::lround(expected[c]*255)))<=1,"Native filter sampled an incorrect mip/slice or lost odd-size normalization");
            }
            auto overlap=make(8,6); upload(overlap,mip,8,6);
            auto distinct=make(8,6);
            filters.Filter(F::Bicubic,{overlap,0,0,MTLRegionMake2D(0,0,8,6)},{distinct,0,0,MTLRegionMake2D(0,0,8,6)},1,true,true);
            filters.Filter(F::Bicubic,{overlap,0,0,MTLRegionMake2D(0,0,8,6)},{overlap,0,0,MTLRegionMake2D(0,0,8,6)},1,true,true);
            Check(Read(transfer,overlap)==Read(transfer,distinct),"Native overlap filter did not preserve an immutable GPU source");
            bool rejected{};
            try { filters.Filter(F(99),{}, {},1); } catch(const std::invalid_argument&) {rejected=true;}
            Check(rejected,"Native texture filter accepted an unknown mode");
            rejected=false;
            try { filters.Filter(F::Bicubic,{overlap,0,0,MTLRegionMake2D(0,0,9,6)},
                                {dest,0,0,MTLRegionMake2D(0,0,17,13)},1); } catch(const std::invalid_argument&) {rejected=true;}
            Check(rejected,"Native filter accepted an out-of-bounds source region");
            Check(!filters.Filter(F::NoFilter,{}, {},1),"Disabled filter did not preserve the native blit path");
            std::cout<<"Native texture-filter mip/slice views, odd normalization, GPU overlap snapshot and explicit bounds/mode validation passed\n";
            Metal::TextureRuntime runtime{device,scheduler};
            F selected=F::NoFilter; runtime.SetTextureFilterProvider([&]{return selected;});
            VideoCore::SurfaceParams params{}; params.width=8; params.height=6; params.res_scale=3;
            params.pixel_format=VideoCore::PixelFormat::RGBA8; params.type=VideoCore::SurfaceType::Texture;
            for(unsigned filter=1;filter<=5;++filter) {
                selected=F(filter); Metal::Surface surface{runtime,params};
                auto staging=runtime.FindStaging(mip.size(),true); std::memcpy(staging.mapped.data(),mip.data(),mip.size());
                surface.Upload({staging.offset,u32(mip.size()),{0,6,8,0},0},staging);
                auto reference=make(24,18);
                filters.Filter(selected,{surface.Image(Metal::SurfaceImage::Base),0,0,MTLRegionMake2D(0,0,8,6)},
                               {reference,0,0,MTLRegionMake2D(0,0,24,18)},3);
                Check(Read(transfer,surface.Image())==Read(transfer,reference),"Native Surface upload ignored the selected filter");
            }
            std::cout<<"All5 native Surface upload filter choices match their direct GPU passes; base guest image retained\n";
            unsigned partial_cases{};
            for(unsigned filter=1;filter<=5;++filter) for(unsigned scale:{2U,3U}) for(unsigned level:{0U,1U}) {
                selected=F(filter);
                VideoCore::SurfaceParams p{};p.width=65;p.height=49;p.levels=2;p.res_scale=scale;
                p.pixel_format=VideoCore::PixelFormat::RGBA8;p.type=VideoCore::SurfaceType::Texture;
                Metal::Surface surface{runtime,p};
                const unsigned width=65U>>level,height=49U>>level;
                std::vector<u8> base(width*height*4);
                for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<4;++c)
                    base[(y*width+x)*4+c]=u8(x*11+y*19+c*37);
                auto full=runtime.FindStaging(base.size(),true);std::memcpy(full.mapped.data(),base.data(),base.size());
                surface.Upload({full.offset,u32(base.size()),{0,height,width,0},level},full);
                const unsigned x=width/2,y=height/2;
                auto patch=runtime.FindStaging(4,true);
                constexpr std::array<u8,4> color{255,0,211,99};
                std::memcpy(patch.mapped.data(),color.data(),4);
                surface.Upload({patch.offset,4,{x,y+1,x+1,y},level},patch);
                auto target=surface.Image();
                const unsigned dw=target.width>>level,dh=target.height>>level;
                auto reference=make(dw,dh);
                filters.Filter(selected,{surface.Image(Metal::SurfaceImage::Base),level,0,MTLRegionMake2D(0,0,width,height)},
                               {reference,0,0,MTLRegionMake2D(0,0,dw,dh)},scale);
                Check(Read(transfer,target,level)==Read(transfer,reference),
                      "Partial native filtered upload left neighboring pixels stale or changed odd-mip UV phase");
                VideoCore::TextureClear clear{};clear.texture_level=level;clear.texture_rect={1,4,4,1};
                clear.value.color={17/255.f,34/255.f,51/255.f,68/255.f};runtime.ClearTexture(surface,clear);
                std::copy(color.begin(),color.end(),base.begin()+(y*width+x)*4);
                base[(y*width+x)*4+1]=173;
                auto snapshot=runtime.FindStaging(base.size(),true);std::memcpy(snapshot.mapped.data(),base.data(),base.size());
                surface.Upload({snapshot.offset,u32(base.size()),{0,height,width,0},level},snapshot,Common::Rectangle<u32>{x,y+1,x+1,y});
                filters.Filter(selected,{surface.Image(Metal::SurfaceImage::Base),level,0,MTLRegionMake2D(0,0,width,height)},
                               {reference,0,0,MTLRegionMake2D(0,0,dw,dh)},scale);
                const auto actual=Read(transfer,target,level),expected=Read(transfer,reference);
                for(unsigned py=0;py<dh;++py) for(unsigned px=0;px<dw;++px) for(unsigned c=0;c<4;++c) {
                    const auto byte=(py*dw+px)*4+c;
                    const auto wanted=px>=1&&px<4&&py>=1&&py<4 ? std::byte(17*(c+1)) : expected[byte];
                    Check(actual[byte]==wanted,"Canonical full-mip snapshot with partial filter damage lost an unrelated GPU write or neighboring filter update");
                }
                const auto base_before=Read(transfer,surface.Image(Metal::SurfaceImage::Base),level);
                auto bad=runtime.FindStaging(4,true);std::memset(bad.mapped.data(),0,4);
                bool bad_damage{};
                try {surface.Upload({bad.offset,4,{x,y+1,x+1,y},level},bad,Common::Rectangle<u32>{width,y+1,width+1,y});}
                catch(const std::invalid_argument&){bad_damage=true;}
                Check(bad_damage && Read(transfer,surface.Image(Metal::SurfaceImage::Base),level)==base_before &&
                      Read(transfer,target,level)==actual,"Rejected filter damage partially changed a native base or scaled mip");
                ++partial_cases;
            }
            std::cout<<partial_cases<<" native partial-filter cases: all5 filters, scale2/3, odd mip0/1, one-pixel neighborhood refresh, full canonical snapshots and preserved outside GPU writes passed\n";
            rejected=false;
            try {filters.Filter(F::Bicubic,{overlap,0,0,MTLRegionMake2D(0,0,8,6)},
                {dest,0,0,MTLRegionMake2D(0,0,17,13)},1,false,false,MTLRegionMake2D(16,0,2,1));}
            catch(const std::invalid_argument&){rejected=true;}
            Check(rejected,"Native filter accepted damage outside its full-mip viewport");
            std::cout<<fixture_count<<" native Metal4 filter fixtures generated; no app/window/ROM/emulation started\n";
            return 0;
        } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
