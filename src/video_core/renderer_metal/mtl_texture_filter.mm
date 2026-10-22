// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_filter.h"
#include "mtl_shaders.h"
#include "common/settings.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <stdexcept>

namespace Metal {
namespace {
const char* Source(TextureFilterPass pass) {
    switch(pass) {
    case TextureFilterPass::Bicubic:return FilterBicubicSource;
    case TextureFilterPass::ScaleForce:return FilterScaleForceSource;
    case TextureFilterPass::Xbrz:return FilterXbrzSource;
    case TextureFilterPass::Mmpx:return FilterMmpxSource;
    case TextureFilterPass::XGradient:return FilterXGradientSource;
    case TextureFilterPass::YGradient:return FilterYGradientSource;
    case TextureFilterPass::Refine:return FilterRefineSource;
    }
    throw std::invalid_argument("Unknown native texture filter pass");
}
bool ReadableColor(MTLPixelFormat format) {
    // These backend formats are sampled by texture2d<float>; integer and depth
    // textures require different shader argument types and cannot enter a filter.
    switch(format) {
    case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm:
    case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Float:
    case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatR32Float: case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float:
    case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatRG11B10Float: case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA: case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC7_RGBAUnorm: case MTLPixelFormatBC7_RGBAUnorm_sRGB:
    case MTLPixelFormatASTC_4x4_LDR: case MTLPixelFormatASTC_4x4_sRGB:
    case MTLPixelFormatASTC_6x6_LDR: case MTLPixelFormatASTC_6x6_sRGB:
    case MTLPixelFormatASTC_8x6_LDR: case MTLPixelFormatASTC_8x6_sRGB:
    case MTLPixelFormatASTC_8x8_LDR: case MTLPixelFormatASTC_8x8_sRGB:
        return true;
    default:return false;
    }
}
void Validate(const Device& device,const TextureRegion& region,bool dest) {
    auto texture=region.texture;
    if(!texture || texture.device!=device.Handle() || region.level>=texture.mipmapLevelCount ||
       texture.sampleCount!=1 || texture.framebufferOnly || texture.storageMode==MTLStorageModeMemoryless ||
       (texture.textureType!=MTLTextureType2D && texture.textureType!=MTLTextureType2DArray &&
        texture.textureType!=MTLTextureTypeCube && texture.textureType!=MTLTextureTypeCubeArray) ||
       region.slice>=texture.arrayLength*((texture.textureType==MTLTextureTypeCube ||
                                          texture.textureType==MTLTextureTypeCubeArray)?6U:1U) ||
       region.region.origin.z || region.region.size.depth!=1 || !region.region.size.width || !region.region.size.height)
        throw std::invalid_argument("Invalid native filter texture region");
    const auto width=std::max<NSUInteger>(1,texture.width>>region.level);
    const auto height=std::max<NSUInteger>(1,texture.height>>region.level);
    const auto& r=region.region;
    if(r.origin.x>width || r.origin.y>height || r.size.width>width-r.origin.x || r.size.height>height-r.origin.y ||
       !(texture.usage&(dest?MTLTextureUsageRenderTarget:MTLTextureUsageShaderRead)))
        throw std::invalid_argument("Native filter exceeds its texture region or usage");
    if(dest && texture.pixelFormat!=MTLPixelFormatRGBA8Unorm && texture.pixelFormat!=MTLPixelFormatRGBA8Unorm_sRGB &&
       texture.pixelFormat!=MTLPixelFormatRGBA32Float)
        throw std::invalid_argument("Native texture filtering requires a color destination");
    if(!dest && !ReadableColor(texture.pixelFormat))
        throw std::invalid_argument("Native texture filtering requires a float-sampled color source");
}
id<MTLTexture> View(id<MTLTexture> texture,NSUInteger level,NSUInteger slice) {
    auto* descriptor=[[MTLTextureViewDescriptor alloc] init];
    descriptor.textureType=MTLTextureType2D; descriptor.pixelFormat=texture.pixelFormat;
    descriptor.levelRange=NSMakeRange(level,1); descriptor.sliceRange=NSMakeRange(slice,1);
    auto result=[texture newTextureViewWithDescriptor:descriptor];
    if(!result) throw std::runtime_error("Native filter texture view could not be created");
    return result;
}
id<MTLTexture> Root(id<MTLTexture> texture) {
    while(texture.parentTexture) texture=texture.parentTexture;
    return texture;
}
struct alignas(16) FilterInfo { std::array<float,4> uv; float scale; std::array<float,3> padding{}; };
static_assert(sizeof(FilterInfo)==32 && offsetof(FilterInfo,scale)==16);
}

TranslatedShader GenerateTextureFilterShader(TextureFilterPass pass) {
    const std::array bindings{
        ShaderBindingRemap{0,0,NoShaderBinding,0,0}, ShaderBindingRemap{0,1,NoShaderBinding,1,1},
        ShaderBindingRemap{0,2,NoShaderBinding,2,2}, RasterizerShaderBindings().back()};
    // The SPIR-V input dialect already defines VULKAN, selecting the original
    // push-constant source spelling; execution and compilation remain Metal4.
    auto shader=TranslateGuestGLSL(std::string("#version 450\n")+Source(pass),ShaderStage::Fragment,bindings);
    // SPIRV-Cross can emit local self assignments for xBRZ's rotated helper
    // arguments. They have no effect and fail strict MSL -Wself-assign checks.
    shader.source=std::regex_replace(shader.source,
        std::regex{R"(\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\s*;)"},"");
    return shader;
}

TextureFilterRuntime::TextureFilterRuntime(Device& device_,Scheduler& scheduler_)
    :device{device_},scheduler{scheduler_},transfers{device,scheduler},pipelines{device} {
    auto* descriptor=[[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter=MTLSamplerMinMagFilterLinear; descriptor.magFilter=MTLSamplerMinMagFilterLinear;
    descriptor.mipFilter=MTLSamplerMipFilterNearest;
    descriptor.sAddressMode=MTLSamplerAddressModeClampToEdge; descriptor.tAddressMode=MTLSamplerAddressModeClampToEdge;
    descriptor.lodMinClamp=0; descriptor.lodMaxClamp=0;
    sampler=device.CreateSampler(descriptor);
}
TextureFilterRuntime::Program& TextureFilterRuntime::GetProgram(TextureFilterPass pass) {
    if(const auto found=programs.find(pass);found!=programs.end()) {
        if(found->second.library.wait_for(std::chrono::seconds{0})!=std::future_status::ready) return found->second;
        try { (void)found->second.library.get(); return found->second; }
        catch (...) { programs.erase(found); }
    }
    auto shader=GenerateTextureFilterShader(pass);
    const auto bytes=shader.source.size();
    auto library=device.DeferCompilation([native=&device,source=shader.source] {
        return native->CompileLibrary(source,"Azahar native Metal4 texture filter");
    },bytes);
    return programs.emplace(pass,Program{std::move(shader),library}).first->second;
}
id<MTLTexture> TextureFilterRuntime::Allocate(NSUInteger width,NSUInteger height,MTLPixelFormat format,bool render_target) {
    auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.storageMode=MTLStorageModePrivate;
    descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
    if(render_target) descriptor.usage|=MTLTextureUsageRenderTarget;
    return device.CreateTexture(descriptor);
}
void TextureFilterRuntime::Pass(TextureFilterPass pass,const std::array<id<MTLTexture>,3>& images,
                               const TextureRegion& dest,std::array<float,4> uv,float scale,std::optional<MTLRegion> clip) {
    if(vertex.valid() && vertex.wait_for(std::chrono::seconds{0})==std::future_status::ready) {
        try { (void)vertex.get(); } catch (...) { vertex={}; }
    }
    if(!vertex.valid()) vertex=device.DeferCompilation([native=&device] {
        return native->CompileLibrary(TextureFilterVertexSource,"Azahar native Metal4 filter vertex");
    },TextureFilterVertexSource.size());
    auto& program=GetProgram(pass);
    auto target=View(dest.texture,dest.level,dest.slice);
    auto pipeline=pipelines.RenderAsync({nil,"filter_vertex",nil,program.shader.entry_point,
        target.pixelFormat,{},vertex,program.library});
    auto parameters=device.CreateBuffer(sizeof(FilterInfo),true);
    const FilterInfo info{uv,scale,{}};
    std::memcpy(parameters.contents,&info,sizeof(info));
    const std::array buffers{BufferBinding{parameters,0,21}};
    std::vector<TextureBinding> textures;
    std::vector<SamplerBinding> samplers;
    for(const auto& resource:program.shader.resources) {
        if(!resource.active || resource.kind!=ShaderResourceKind::Texture) continue;
        if(resource.metal_index>=images.size() || !images[resource.metal_index])
            throw std::logic_error("Native texture filter input is missing");
        textures.push_back({images[resource.metal_index],resource.metal_index});
        samplers.push_back({sampler,resource.sampler_index});
    }
    auto* render_pass=[[MTL4RenderPassDescriptor alloc] init];
    render_pass.colorAttachments[0].texture=target;
    render_pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
    render_pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    const auto& r=dest.region;
    const auto& scissor=clip ? *clip : r;
    const RenderCommand command{.pipeline=nil,.depth=nil,
        .viewport={double(r.origin.x),double(r.origin.y),double(r.size.width),double(r.size.height),0,1},
        .scissor={scissor.origin.x,scissor.origin.y,scissor.size.width,scissor.size.height},
        .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3,.pending_pipeline=pipeline};
    const auto snapshot=scheduler.SnapshotRenderPass(render_pass);
    scheduler.QueueRender(*snapshot,command,buffers,textures,samplers,false);
}
bool TextureFilterRuntime::Filter(Settings::TextureFilter filter,const TextureRegion& source,
                                  const TextureRegion& dest,float scale,bool flip_x,bool flip_y,std::optional<MTLRegion> clip) {
    using F=Settings::TextureFilter;
    if(filter==F::NoFilter) return false;
    TextureFilterPass pass;
    switch(filter) {
    case F::Bicubic:pass=TextureFilterPass::Bicubic;break;
    case F::ScaleForce:pass=TextureFilterPass::ScaleForce;break;
    case F::xBRZ:pass=TextureFilterPass::Xbrz;break;
    case F::MMPX:pass=TextureFilterPass::Mmpx;break;
    case F::Anime4K:pass=TextureFilterPass::Refine;break;
    default:throw std::invalid_argument("Unknown native texture filter");
    }
    Validate(device,source,false); Validate(device,dest,true);
    if(clip) {
        const auto& r=dest.region;
        if(clip->origin.z || clip->size.depth!=1 || !clip->size.width || !clip->size.height ||
           clip->origin.x<r.origin.x || clip->origin.y<r.origin.y ||
           clip->origin.x>r.origin.x+r.size.width || clip->origin.y>r.origin.y+r.size.height ||
           clip->size.width>r.origin.x+r.size.width-clip->origin.x ||
           clip->size.height>r.origin.y+r.size.height-clip->origin.y)
            throw std::invalid_argument("Native filter clip exceeds its viewport");
    }
    if(!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Invalid native filter resolution scale");
    auto input=View(source.texture,source.level,source.slice);
    const auto source_root=Root(source.texture),dest_root=Root(dest.texture);
    // Separate linear texture objects may still overlap the same buffer allocation.
    // Preserve the complete sampled image before any destination write can change it.
    if(source_root==dest_root || (source_root.buffer && source_root.buffer==dest_root.buffer)) {
        auto snapshot=Allocate(input.width,input.height,input.pixelFormat,false);
        transfers.CopyTexture({source.texture,source.level,source.slice,MTLRegionMake2D(0,0,input.width,input.height)},
                              {snapshot,0,0,MTLRegionMake2D(0,0,input.width,input.height)});
        input=snapshot;
    }
    std::array<float,4> uv{float(source.region.origin.x)/input.width,float(source.region.origin.y)/input.height,
        float(source.region.origin.x+source.region.size.width)/input.width,
        float(source.region.origin.y+source.region.size.height)/input.height};
    if(flip_x) std::swap(uv[0],uv[2]);
    if(flip_y) std::swap(uv[1],uv[3]);
    std::array<id<MTLTexture>,3> images{input,nil,nil};
    if(filter==F::Anime4K) {
        auto horizontal=Allocate(input.width,input.height,MTLPixelFormatRG32Float);
        auto gradient=Allocate(input.width,input.height,MTLPixelFormatR32Float);
        const auto full=MTLRegionMake2D(0,0,input.width,input.height);
        Pass(TextureFilterPass::XGradient,images,{horizontal,0,0,full},{0,0,1,1},scale);
        Pass(TextureFilterPass::YGradient,{nil,nil,horizontal},{gradient,0,0,full},{0,0,1,1},scale);
        images[1]=gradient;
    }
    Pass(pass,images,dest,uv,scale,clip);
    return true;
}
} // namespace Metal
