// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_runtime.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <chrono>

namespace {
void Check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
VideoCore::BufferTextureCopy Copy(const VideoCore::StagingData& staging, u32 width, u32 height, u32 bytes, u32 level = 0) {
    return {staging.offset, width * height * bytes, {0, height, width, 0}, level};
}
u64 GuestDepthPixel(u32 x, u32 y, u32 stride, bool tiled) {
    if (!tiled) return u64(y) * stride + x;
    unsigned morton{};
    for (unsigned bit = 0; bit < 3; ++bit)
        morton |= ((x >> bit) & 1) << (bit * 2) | ((y >> bit) & 1) << (bit * 2 + 1);
    return (u64(y / 8) * (stride / 8) + x / 8) * 64 + morton;
}
void ValidateDepthGuestEncoding(Metal::Device& device, Metal::Scheduler& scheduler,
                                Metal::TextureRuntime& runtime) {
    using F = VideoCore::PixelFormat;
    Metal::GuestTextureCodec codec{device, scheduler};
    unsigned cases{}, bytes_checked{}, single_pass{}, rejected{};
    for (auto format : {F::D16, F::D24, F::D24S8})
        for (unsigned scale : {1U, 3U}) for (bool tiled : {false, true}) {
            VideoCore::SurfaceParams params{};
            params.addr = 0x10000;
            params.width = params.height = 256;
            params.stride = 272; // Padding in every row / eight-row tile strip.
            params.is_tiled = tiled;
            params.pixel_format = format;
            params.res_scale = scale;
            params.UpdateParams();
            Metal::Surface surface{runtime, params};
            const auto linear_bpp = surface.GetInternalBytesPerPixel();
            const auto guest_bpp = format == F::D16 ? 2U : format == F::D24 ? 3U : 4U;
            auto staging = runtime.FindStaging(65536 * linear_bpp, true);
            std::vector<u32> words(65536);
            for (unsigned i = 0; i < words.size(); ++i) {
                if (format == F::D16) {
                    const auto word = u16(i); // Every representable native D16 value.
                    std::memcpy(staging.mapped.data() + i * 2, &word, 2);
                    words[i] = word;
                } else if (format == F::D24) {
                    float value = float((i * 7919U) & 0xffffffU) / 16777215.f;
                    switch (i % 16) {
                    case 0: value = 0; break;
                    case 1: value = 1; break;
                    case 2: value = -1; break;
                    case 3: value = 2; break;
                    case 4: value = std::numeric_limits<float>::quiet_NaN(); break;
                    case 5: value = std::numeric_limits<float>::infinity(); break;
                    case 6: value = std::nextafter(1.f, 0.f); break;
                    case 7: value = std::numeric_limits<float>::denorm_min(); break;
                    default: break;
                    }
                    std::memcpy(staging.mapped.data() + i * 4, &value, 4);
                    words[i] = std::isfinite(value) ? u32(std::clamp(value, 0.f, 1.f) * 16777215.f) : 0U;
                } else {
                    const auto depth = (i * 257U) & 0xffffffU;
                    const auto stencil = (i * 43U + (i >> 8)) & 255U;
                    const auto packed = (depth << 8) | stencil;
                    std::memcpy(staging.mapped.data() + i * 4, &packed, 4);
                    words[i] = depth | (stencil << 24);
                }
            }
            // Use the established linear native upload, not GuestCodec::Decode,
            // so shared guest-layout mistakes cannot make a roundtrip self-pass.
            surface.Upload(Copy(staging, 256, 256, linear_bpp), staging);
            if (format == F::D16) {
                std::vector<u8> complete(params.size, 0xa5);
                runtime.DownloadGuest(surface, params, complete, NSMakeRange(0, complete.size()));
                for (unsigned y=0;y<256;++y) for(unsigned x=0;x<256;++x) {
                    const auto pixel=GuestDepthPixel(x,tiled ? 255-y:y,272,tiled);
                    Check(complete[pixel*2]==u8(y*256+x) && complete[pixel*2+1]==u8(y),
                          "Fused native D16 encoding did not preserve every one of the 65536 stored values");
                }
                ++cases;bytes_checked+=65536*2;
            }
            VideoCore::TextureClear clear{};
            clear.texture_rect = {19 * scale, 21 * scale, 26 * scale, 13 * scale};
            // Exact endpoint avoids assuming how hardware quantizes a half-way
            // D16 clear. Exhaustive stored D16 values were checked before it.
            clear.value.depth = 1.f;
            clear.value.stencil = 0x6b;
            runtime.ClearTexture(surface, clear); // Actual later GPU write, including scaled storage.
            for (unsigned y = 13; y < 21; ++y) for (unsigned x = 19; x < 26; ++x)
                words[y * 256 + x] = format == F::D16 ? 65535U
                    : 16777215U | (format == F::D24S8 ? 0x6b000000U : 0U);
            std::vector<u8> expected(params.size, 0xa5);
            for (unsigned y = 0; y < 256; ++y) for (unsigned x = 0; x < 256; ++x) {
                const auto pixel = GuestDepthPixel(x, tiled ? 255-y : y, 272, tiled);
                for (unsigned c = 0; c < guest_bpp; ++c)
                    expected[pixel * guest_bpp + c] = u8(words[y * 256 + x] >> (8 * c));
            }
            for (auto range : {NSMakeRange(0, params.size), NSMakeRange(1, params.size-5),
                               NSMakeRange(63 * guest_bpp + 1, 5), NSMakeRange(params.size-1, 1),
                               NSMakeRange(0, 0)}) {
                std::vector<u8> result(params.size, 0xa5);
                const auto before = scheduler.EncoderCount();
                runtime.DownloadGuest(surface, params, result, range);
                if (scale == 1 && range.length) {
                    Check(scheduler.EncoderCount() == before + 1,
                          "Unscaled native depth guest download did not use exactly one GPU pass");
                    ++single_pass;
                }
                for (unsigned b = 0; b < result.size(); ++b) {
                    const auto wanted = b >= range.location && b-range.location < range.length ? expected[b] : 0xa5;
                    if (result[b] != wanted) {
                        std::cerr << "Depth mismatch format=" << unsigned(format) << " scale=" << scale
                                  << " tiled=" << tiled << " byte=" << b << " actual=" << unsigned(result[b])
                                  << " expected=" << unsigned(wanted) << " range=" << range.location << ',' << range.length << '\n';
                        throw std::runtime_error("Fused native depth guest bytes differ from independent layout/precision/padding oracle");
                    }
                }
                bytes_checked += result.size();
                ++cases;
            }
        }
    // Partial mip rectangles select a nonzero view level and source origin. The
    // byte interval crosses pixel/tile boundaries and must preserve both guards.
    for (auto format : {F::D16, F::D24, F::D24S8}) {
        VideoCore::SurfaceParams params{};
        params.addr = 0x20000; params.width = params.height = params.stride = 64;
        params.levels = 2; params.pixel_format = format; params.is_tiled = true;
        params.res_scale = 2; params.UpdateParams();
        Metal::Surface surface{runtime, params};
        const auto bpp = surface.GetInternalBytesPerPixel();
        auto input = runtime.FindStaging(32*32*bpp, true);
        std::vector<u32> wanted(1024);
        for (unsigned i = 0; i < wanted.size(); ++i) {
            if (format == F::D16) {
                const auto value = u16(i*61);std::memcpy(input.mapped.data()+i*bpp,&value,2);wanted[i]=value;
            } else if (format == F::D24) {
                const auto value = float(i)/1023;std::memcpy(input.mapped.data()+i*bpp,&value,4);
                wanted[i]=u32(value*16777215.f);
            } else {
                const auto depth = i*16399U & 0xffffffU, stencil = i*37 & 255U;
                const auto value = (depth<<8)|stencil;std::memcpy(input.mapped.data()+i*bpp,&value,4);
                wanted[i]=depth|(stencil<<24);
            }
        }
        surface.Upload(Copy(input,32,32,bpp,1),input);
        const auto guest_bpp = format == F::D16 ? 2U : format == F::D24 ? 3U : 4U;
        // One 8x8 tile at guest x8,y8 -> native x8,y16 in mip1.
        const auto address = params.LevelInterval(1).lower() + (4+1)*64*guest_bpp;
        const auto sub = surface.FromInterval({address,address+64*guest_bpp});
        Check(sub.width==8 && sub.height==8 && surface.LevelOf(sub.addr)==1,
              "Depth readback test did not select the intended nonzero mip tile");
        std::vector<u8> expected(sub.size,0xa5), result(sub.size,0xa5);
        for (unsigned y=0;y<8;++y) for(unsigned x=0;x<8;++x) {
            const auto pixel = GuestDepthPixel(x,7-y,8,true);
            for(unsigned c=0;c<guest_bpp;++c) expected[pixel*guest_bpp+c]=u8(wanted[(16+y)*32+8+x]>>(8*c));
        }
        const auto range=NSMakeRange(1,sub.size-3);
        runtime.DownloadGuest(surface,sub,result,range);
        for(unsigned b=0;b<result.size();++b)
            Check(result[b]==(b>=1 && b<sub.size-2 ? expected[b]:0xa5),
                  "Fused depth download selected the wrong mip, tile, origin or byte interval");
        ++cases;bytes_checked+=result.size();
        const auto depth_view=surface.ImageView(Metal::SurfaceView::Depth,Metal::SurfaceImage::Base,1);
        const auto stencil_view=format==F::D24S8
            ? surface.ImageView(Metal::SurfaceView::Stencil,Metal::SurfaceImage::Base,1):nil;
        for(bool flip : {false,true}) {
            Metal::GuestTextureLayout layout{16,16,24,format,true,true,flip};
            const auto size=Metal::GuestTextureCodec::GuestBytes(layout);
            auto buffer=device.CreateBuffer(size+48,true);
            std::memset(buffer.contents,0xc7,buffer.length);
            const Metal::BufferSlice output{buffer,17,size,static_cast<u8*>(buffer.contents)+17,{}};
            const auto span=NSMakeRange(1,size-3);
            const auto before=scheduler.EncoderCount();
            codec.EncodeDepth(layout,depth_view,stencil_view,MTLOriginMake(3,9,0),output,span);
            scheduler.Finish();
            Check(scheduler.EncoderCount()==before+1,"Direct native depth encoder split its GPU pass");
            std::vector<u8> wanted_bytes(buffer.length,0xc7);
            for(unsigned y=0;y<16;++y) for(unsigned x=0;x<16;++x) {
                const auto pixel=GuestDepthPixel(x,y,24,true);
                const auto word=wanted[(9+(flip ? y:15-y))*32+3+x];
                for(unsigned c=0;c<guest_bpp;++c) {
                    const auto b=pixel*guest_bpp+c;
                    if(b>=span.location && b-span.location<span.length)wanted_bytes[17+b]=u8(word>>(8*c));
                }
            }
            Check(!std::memcmp(buffer.contents,wanted_bytes.data(),buffer.length),
                  "Direct fused native depth encoding changed nonzero destination offset, flip, padding or guards");
            ++cases;bytes_checked+=buffer.length;
            if(format==F::D24S8 && !flip) {
                const auto reject=[&](Metal::GuestTextureLayout invalid,id<MTLTexture> depth,id<MTLTexture> stencil,
                                      MTLOrigin origin,Metal::BufferSlice dest,NSRange range) {
                    const auto count=scheduler.EncoderCount();
                    try { codec.EncodeDepth(invalid,depth,stencil,origin,dest,range); }
                    catch(const std::invalid_argument&) {
                        Check(scheduler.EncoderCount()==count,"Rejected native depth encoding mutated the GPU command stream");
                        ++rejected;return;
                    }
                    throw std::runtime_error("Invalid fused native depth encoding was accepted");
                };
                auto bad=layout;bad.format=F::RGBA8;
                reject(bad,depth_view,stencil_view,MTLOriginMake(0,0,0),output,span);
                bad=layout;bad.converted=false;
                reject(bad,depth_view,stencil_view,MTLOriginMake(0,0,0),output,span);
                reject(layout,nil,stencil_view,MTLOriginMake(0,0,0),output,span);
                reject(layout,depth_view,nil,MTLOriginMake(0,0,0),output,span);
                reject(layout,depth_view,depth_view,MTLOriginMake(0,0,0),output,span);
                reject(layout,depth_view,stencil_view,MTLOriginMake(32,0,0),output,span);
                reject(layout,depth_view,stencil_view,MTLOriginMake(0,0,1),output,span);
                reject(layout,depth_view,stencil_view,MTLOriginMake(0,0,0),output,NSMakeRange(size,1));
                auto short_slice=output;--short_slice.size;
                reject(layout,depth_view,stencil_view,MTLOriginMake(0,0,0),short_slice,span);
            }
        }
    }
    std::cout << cases << " fused depth guest cases, " << bytes_checked << " independent byte checks, "
              << single_pass << " single-pass checks; every D16 word, D24 endpoints/nonfinite/truncation, "
                 "D24S8 stencil, native/scaled GPU writes, Morton/linear padding, partial pixels, nonzero mip tiles, "
                 "view origins, destination guards and both flips; " << rejected << " pre-encoding rejections passed\n";
}
void ValidateStagingSnapshots(Metal::Device& device, Metal::Scheduler& scheduler,
                              Metal::TextureRuntime& runtime, Metal::TransferRuntime& transfer) {
    VideoCore::SurfaceParams params;
    params.addr = 0x1000;
    params.width = params.height = params.stride = 128;
    params.pixel_format = VideoCore::PixelFormat::RGBA8;
    params.is_tiled = true;
    params.UpdateParams();
    std::vector<std::unique_ptr<Metal::Surface>> surfaces;
    for (unsigned i = 0; i < 32; ++i)
        surfaces.push_back(std::make_unique<Metal::Surface>(runtime, params));
    std::vector<u8> raw(params.size), result(params.size);
    unsigned checked{};
    const auto value = [](unsigned round, unsigned image, unsigned byte) {
        return u8(round * 37 + image * 61 + byte * 19 + (byte >> 7));
    };
    // More than the retained arena capacity, with every group encoded before
    // any submission. Mutable caller memory must not become GPU source storage.
    for (unsigned round = 0; round < 12; ++round) {
        scheduler.Finish();
        const auto tick = scheduler.CurrentTick();
        for (unsigned i = 0; i < surfaces.size(); ++i) {
            for (unsigned b = 0; b < raw.size(); ++b) raw[b] = value(round, i, b);
            runtime.UploadGuest(*surfaces[i], params, raw);
            std::fill(raw.begin(), raw.end(), 0xee);
        }
        Check(scheduler.CurrentTick() == tick, "Texture staging forced an upload submission");
        for (unsigned i = 0; i < surfaces.size(); ++i) {
            for (bool partial : {false, true}) {
                std::fill(result.begin(), result.end(), 0xa5);
                const NSRange range = partial ? NSMakeRange(13, result.size() - 37)
                                              : NSMakeRange(0, result.size());
                runtime.DownloadGuest(*surfaces[i], params, result, range);
                for (unsigned b = 0; b < result.size(); ++b)
                    Check(result[b] == (b >= range.location && b - range.location < range.length
                              ? value(round, i, b) : 0xa5),
                          "Queued guest upload/readback lost a snapshot or partial flush guard");
                ++checked;
            }
        }
    }
    params.width = params.height = params.stride = 4;
    params.is_tiled = false;
    params.UpdateParams();
    Metal::Surface tiny{runtime, params};
    const auto stale = runtime.FindStaging(1024, true);
    const auto current = runtime.FindStaging(1024, true);
    Check(current.offset != 0, "Texture staging never exercised an arena offset");
    unsigned rejected{};
    const auto reject = [&](const VideoCore::StagingData& data) {
        try { tiny.Upload(Copy(data, 4, 4, 4), data); }
        catch (const std::invalid_argument&) { ++rejected; return; }
        throw std::runtime_error("Staging accepted a range outside its current CPU lease");
    };
    reject(stale);
    auto forged = current;
    forged.offset += current.size;
    forged.size = 64;
    forged.mapped = {current.mapped.data() + current.size, 64};
    reject(forged);
    auto bounded = current;
    bounded.offset += 256;
    bounded.size = 64;
    bounded.mapped = current.mapped.subspan(256, 64);
    std::fill(bounded.mapped.begin(), bounded.mapped.end(), 0x67);
    tiny.Upload(Copy(bounded, 4, 4, 4), bounded);
    const auto pixels = transfer.Readback({tiny.Image(), 0, 0, MTLRegionMake2D(0, 0, 4, 4)}, 16);
    Check(std::all_of(pixels.begin(), pixels.end(), [](std::byte v) { return v == std::byte{0x67}; }) && rejected == 2,
          "Bounded nonzero staging subrange changed its data or bypassed validation");
    std::cout << "384 queued guest uploads in 12 batches, 768 full/partial 65536-byte readbacks, "
                 "mutable caller snapshots, arena turnover, nonzero/subrange offsets and 2 stale/forged rejections passed\n";
}
void ValidateFramebufferPassCache(Metal::Device& device, Metal::Scheduler& scheduler,
                                  Metal::TextureRuntime& runtime, Metal::TransferRuntime& transfer) {
    struct Plane {
        id<MTLBuffer> output;
        NSUInteger bytes{};
        unsigned kind{}, expected{}; // RGBA8, D16, D32, stencil.
    };
    std::vector<Plane> planes;
    auto shadow_library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 cache_vertex(uint index [[vertex_id]]) {
    const float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[index], .5, 1);
}
fragment void cache_shadow(float4 position [[position]], constant uint& word [[buffer(0)]],
                           texture2d<uint, access::write> target [[texture(0)]]) {
    target.write(uint4(word, 0, 0, 0), uint2(position.xy));
}
)", "Attachment-free native framebuffer cache fixture");
    auto* pipeline_descriptor = [[MTL4RenderPipelineDescriptor alloc] init];
    pipeline_descriptor.vertexFunctionDescriptor = Metal::Device::Function(shadow_library, "cache_vertex");
    pipeline_descriptor.fragmentFunctionDescriptor = Metal::Device::Function(shadow_library, "cache_shadow");
    pipeline_descriptor.rasterSampleCount = 1;
    auto shadow_pipeline = device.CompileRender(pipeline_descriptor);
    unsigned cases{}, rejected{};
    const auto reject = [&](auto&& action) {
        try { action(); } catch (const std::invalid_argument&) { ++rejected; return; }
        throw std::runtime_error("Framebuffer descriptor cache bypassed invalid-state rejection");
    };
    for (auto format : {VideoCore::PixelFormat::D16, VideoCore::PixelFormat::D24,
                        VideoCore::PixelFormat::D24S8}) {
        VideoCore::SurfaceParams params;
        params.width = params.height = 16; params.levels = 3; params.res_scale = 1;
        params.pixel_format = VideoCore::PixelFormat::RGBA8; params.type = VideoCore::SurfaceType::Color;
        Metal::Surface color{runtime, params};
        params.pixel_format = format;
        params.type = format == VideoCore::PixelFormat::D24S8 ? VideoCore::SurfaceType::DepthStencil : VideoCore::SurfaceType::Depth;
        Metal::Surface depth{runtime, params};
        for (unsigned mode = 0; mode < 4; ++mode) {
            VideoCore::FramebufferParams fb_params{};
            fb_params.shadow_rendering = mode == 3;
            Metal::Framebuffer fb{runtime, fb_params, mode == 1 ? nullptr : &color,
                                  mode == 0 ? nullptr : &depth};
            MTL4RenderPassDescriptor* previous{};
            for (u32 scale : {1U, 2U}) {
                color.ScaleUp(scale); depth.ScaleUp(scale);
                for (u32 level = 0; level < 3; ++level) {
                    fb.color_level = level; fb.depth_level = (level + 1) % 3;
                    const auto old_width = previous.renderTargetWidth;
                    const auto old_texture = previous.colorAttachments[0].texture;
                    const auto old_level = previous.colorAttachments[0].level;
                    auto* pass = fb.RenderPass();
                    const auto snapshot = fb.RenderSnapshot(scheduler);
                    Check(!previous || pass != previous, "Changed framebuffer key reused an old descriptor");
                    if (previous) Check(previous.renderTargetWidth == old_width &&
                        previous.colorAttachments[0].texture == old_texture && previous.colorAttachments[0].level == old_level,
                        "Framebuffer descriptor replacement mutated a published pass");
                    for (unsigned repeat = 0; repeat < 100; ++repeat) {
                        Check(fb.RenderPass() == pass, "Warm framebuffer allocated another render-pass descriptor");
                        Check(fb.RenderSnapshot(scheduler) == snapshot, "Warm framebuffer rebuilt its immutable render-pass proof");
                    }
                    const bool use_color = mode == 0 || mode == 2, use_depth = mode == 1 || mode == 2;
                    const bool stencil = use_depth && format == VideoCore::PixelFormat::D24S8;
                    NSUInteger width = 16 * scale >> (use_color || mode == 3 ? level : fb.depth_level);
                    if (use_color && use_depth) width = std::min(width, NSUInteger(16 * scale >> fb.depth_level));
                    Check(pass.renderTargetWidth == width && pass.renderTargetHeight == width &&
                          pass.colorAttachments[0].texture == (use_color ? color.Image() : nil) &&
                          pass.depthAttachment.texture == (use_depth ? depth.Image() : nil) &&
                          pass.stencilAttachment.texture == (stencil ? depth.Image() : nil),
                          "Cached framebuffer lost dimensions, attachment mode or stencil selection");
                    if (mode == 3) Check(pass.defaultRasterSampleCount == 1,
                                         "Attachment-free shadow pass has no raster sample count");
                    for (NSUInteger slot = 1; slot < 8; ++slot)
                        Check(!pass.colorAttachments[slot].texture && !pass.colorAttachments[slot].resolveTexture,
                              "Cached framebuffer contains a stale secondary attachment");
                    Check(!pass.colorAttachments[0].resolveTexture && !pass.depthAttachment.resolveTexture &&
                          !pass.stencilAttachment.resolveTexture && !pass.visibilityResultBuffer && !pass.rasterizationRateMap,
                          "Cached framebuffer inherited an unrelated pass resource");
                    if (use_color) Check(pass.colorAttachments[0].level == level &&
                        pass.colorAttachments[0].loadAction == MTLLoadActionLoad && pass.colorAttachments[0].storeAction == MTLStoreActionStore,
                        "Cached color pass lost mip or Load/Store actions");
                    if (use_depth) Check(pass.depthAttachment.level == fb.depth_level &&
                        pass.depthAttachment.loadAction == MTLLoadActionLoad && pass.depthAttachment.storeAction == MTLStoreActionStore,
                        "Cached depth pass lost mip or Load/Store actions");
                    if (stencil) Check(pass.stencilAttachment.level == fb.depth_level &&
                        pass.stencilAttachment.loadAction == MTLLoadActionLoad && pass.stencilAttachment.storeAction == MTLStoreActionStore,
                        "Cached stencil pass lost mip or Load/Store actions");
                    auto copied = fb;
                    Check(copied.RenderPass() == pass, "Framebuffer copy lost its immutable descriptor");
                    Check(copied.RenderSnapshot(scheduler)==snapshot,"Framebuffer copy lost its immutable render-pass proof");
                    copied.shadow_rendering = !copied.shadow_rendering;
                    if (mode != 1) {
                        Check(copied.RenderPass() != pass && fb.RenderPass() == pass,
                              "Changing a copied framebuffer mutated the original descriptor cache");
                        Check(copied.RenderSnapshot(scheduler)!=snapshot && fb.RenderSnapshot(scheduler)==snapshot,
                              "Changed framebuffer reused a stale immutable pass proof");
                    }
                    // Clear through an independent copy, then consume the immutable Load/Store
                    // descriptor twice. Every render-area sample and guard byte is checked.
                    const unsigned red = 17 + cases, stamp = 80 + cases;
                    const auto clear_color = MTLClearColorMake(double(red) / 255, 33.0 / 255, 49.0 / 255, 1);
                    MTL4RenderPassDescriptor* initialize = [pass copy];
                    if (use_color) {
                        initialize.colorAttachments[0].loadAction = MTLLoadActionClear;
                        initialize.colorAttachments[0].clearColor = clear_color;
                    }
                    if (use_depth) {
                        initialize.depthAttachment.loadAction = MTLLoadActionClear;
                        initialize.depthAttachment.clearDepth = .25;
                    }
                    if (stencil) {
                        initialize.stencilAttachment.loadAction = MTLLoadActionClear;
                        initialize.stencilAttachment.clearStencil = stamp;
                    }
                    id<MTL4ArgumentTable> shadow_arguments{};
                    if (mode == 3) {
                        transfer.ClearColor(color.Image(), MTLClearColorMake(0, 0, 0, 0), level);
                        auto word_buffer = device.CreateBuffer(4, true);
                        const u32 word = red | (33U << 8) | (49U << 16) | (255U << 24);
                        std::memcpy(word_buffer.contents, &word, sizeof(word));
                        const std::array<Metal::BufferBinding, 1> buffers{{{word_buffer, 0, 0}}};
                        const std::array<Metal::TextureBinding, 1> textures{{{fb.ImageView(VideoCore::SurfaceType::Color), 0}}};
                        shadow_arguments = scheduler.MakeArguments(buffers, textures);
                        scheduler.KeepAlive(shadow_pipeline);
                    }
                    else { scheduler.BeginRender(initialize); scheduler.EndEncoding(); }
                    for (unsigned repeat = 0; repeat < 2; ++repeat) {
                        auto encoder = scheduler.BeginCompatibleRender(*snapshot);
                        if (mode == 3) {
                            [encoder setRenderPipelineState:shadow_pipeline];
                            [encoder setArgumentTable:shadow_arguments atStages:MTLRenderStageFragment];
                            [encoder setViewport:MTLViewport{0, 0, double(width), double(width), 0, 1}];
                            [encoder setScissorRect:MTLScissorRect{0, 0, width, width}];
                            [encoder setCullMode:MTLCullModeNone];
                            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        }
                        scheduler.EndEncoding();
                    }
                    const auto queue_plane = [&](id<MTLTexture> image, u32 mip, unsigned kind,
                                                 unsigned bytes, MTLBlitOption option, unsigned expected) {
                        const auto size = width * width * bytes;
                        auto output = device.CreateBuffer(size + 32, true);
                        transfer.FillBuffer(output, NSMakeRange(0, size + 32), 0xBC);
                        transfer.Download({image, mip, 0, MTLRegionMake2D(0, 0, width, width)}, output,
                                          {16, width * bytes, 0, option});
                        planes.push_back({output, size, kind, expected});
                    };
                    if (use_color || mode == 3) queue_plane(color.Image(), level, 0, 4, MTLBlitOptionNone, red);
                    if (use_depth) queue_plane(depth.Image(), fb.depth_level, format == VideoCore::PixelFormat::D16 ? 1 : 2,
                        format == VideoCore::PixelFormat::D16 ? 2 : 4,
                        stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone, 0);
                    if (stencil) queue_plane(depth.Image(), fb.depth_level, 3, 1, MTLBlitOptionStencilFromDepthStencil, stamp);
                    previous = pass;
                    ++cases;
                    if (cases % 7 == 0) scheduler.SubmitPending();
                }
                if (mode != 3) {
                    const auto valid = fb.RenderPass();
                    const auto color_level = fb.color_level, depth_level = fb.depth_level;
                    if (mode != 1) { fb.color_level = 3; reject([&] { (void)fb.RenderPass(); }); fb.color_level = color_level; }
                    if (mode != 0) { fb.depth_level = 3; reject([&] { (void)fb.RenderPass(); }); fb.depth_level = depth_level; }
                    Check(fb.RenderPass() == valid, "Rejected mip changed the last valid descriptor");
                }
            }
        }
    }
    VideoCore::FramebufferParams empty{};
    Metal::Framebuffer missing{runtime, empty, nullptr, nullptr};
    reject([&] { (void)missing.RenderPass(); });
    VideoCore::SurfaceParams invalid;
    invalid.pixel_format = VideoCore::PixelFormat::Invalid;
    Metal::Surface absent{runtime, invalid};
    Metal::Framebuffer no_image{runtime, empty, &absent, nullptr};
    reject([&] { (void)no_image.RenderPass(); });
    // All Surface and Framebuffer owners above are gone. Submitted encoders must
    // still own the exact old backings used by their queued readbacks.
    scheduler.Finish();
    NSUInteger samples{};
    for (const auto& plane : planes) {
        const auto* data = static_cast<const u8*>(plane.output.contents);
        for (unsigned i = 0; i < 16; ++i) Check(data[i] == 0xBC && data[16 + plane.bytes + i] == 0xBC,
                                               "Cached render pass readback overwrote its guards");
        data += 16;
        const unsigned step = plane.kind == 1 ? 2 : plane.kind == 3 ? 1 : 4;
        for (NSUInteger offset = 0; offset < plane.bytes; offset += step) {
            if (plane.kind == 0) Check(data[offset] == plane.expected && data[offset + 1] == 33 &&
                data[offset + 2] == 49 && data[offset + 3] == 255, "Cached render pass lost a color Load/Store sample");
            else if (plane.kind == 1) { u16 value; std::memcpy(&value, data + offset, 2);
                Check(value >= 16383 && value <= 16384, "Cached render pass lost a D16 Load/Store sample"); }
            else if (plane.kind == 2) { float value; std::memcpy(&value, data + offset, 4);
                Check(value == .25f, "Cached render pass lost a floating depth Load/Store sample"); }
            else Check(data[offset] == plane.expected, "Cached render pass lost a stencil Load/Store sample");
            ++samples;
        }
    }
    Check(cases == 72 && planes.size() == 102 && rejected == 26, "Framebuffer pass-cache fixture lost coverage");
    std::cout << "Native framebuffer pass cache: " << cases << " color/depth/stencil/shadow mip/scale cases, "
              << planes.size() << " GPU readback planes, " << samples << " render-area samples, 7200 warm hits, "
              << rejected << " invalid-state rejections, 36 attachment-free shadow draws, copied-framebuffer isolation and released old backings passed\n";
}

void ValidateViewCache(Metal::Device& device, Metal::Scheduler& scheduler,
                       Metal::TextureRuntime& runtime, Metal::TransferRuntime& transfer) {
    VideoCore::SurfaceParams params;
    params.width = params.height = 8;
    params.levels = 4;
    params.res_scale = 1;
    params.pixel_format = VideoCore::PixelFormat::RGBA8;
    params.type = VideoCore::SurfaceType::Texture;
    params.texture_type = VideoCore::TextureType::CubeMap;
    Metal::Surface cube{runtime, params};
    auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void write_view(texture2d<uint, access::write> target [[texture(0)]],
                       constant uint& word [[buffer(0)]]) {
    target.write(uint4(word, 0, 0, 0), uint2(0));
}
)", "Bounded native surface view cache fixture");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "write_view");
    auto pipeline = device.CompileCompute(descriptor);
    const auto write = [&](id<MTLTexture> view, u32 word) {
        auto data = device.CreateBuffer(4, true);
        std::memcpy(data.contents, &word, 4);
        const std::array<Metal::BufferBinding, 1> buffers{{{data, 0, 0}}};
        const std::array<Metal::TextureBinding, 1> textures{{{view, 0}}};
        auto table = scheduler.MakeArguments(buffers, textures);
        scheduler.KeepAlive(pipeline);
        auto encoder = scheduler.BeginCompute(pipeline);
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        scheduler.EndEncoding();
    };
    const auto read = [&](id<MTLTexture> texture, u32 level, u32 slice, u32 word, u32 outside) {
        const auto width = std::max<NSUInteger>(1, texture.width >> level);
        const auto height = std::max<NSUInteger>(1, texture.height >> level);
        const auto bytes = transfer.Readback({texture, level, slice, MTLRegionMake2D(0, 0, width, height)}, width * 4);
        for (NSUInteger i = 0; i < width * height; ++i) {
            u32 actual; std::memcpy(&actual, bytes.data() + i * 4, 4);
            Check(actual == (i ? outside : word), "Cached native view wrote a wrong mip/face or outside pixel");
        }
    };
    // Repeated native format/mip/slice keys across submissions. Later scale changes
    // clear aliases while commands may still need the preceding backing.
    for (u32 pass = 0; pass < 2; ++pass) for (u32 level = 0; level < 4; ++level) for (u32 face = 0; face < 6; ++face) {
        @autoreleasepool {
            VideoCore::TextureClear clear{};
            const auto dim = 8U >> level;
            clear.texture_level = level; clear.texture_layer = face;
            clear.texture_rect = {0, dim, dim, 0};
            clear.value.color = {.25, .5, .75, 1};
            runtime.ClearTexture(cube, clear);
            auto view = cube.StorageView(level, face);
            Check(view == cube.StorageView(level, face), "A live native view key was not reused");
            Check(view.textureType == MTLTextureType2D && view.pixelFormat == MTLPixelFormatR32Uint &&
                  view.width == dim && view.height == dim && view.mipmapLevelCount == 1,
                  "Cached storage view lost its exact native format, range or type");
            const auto rgba = cube.ImageView(Metal::SurfaceView::Mip0, Metal::SurfaceImage::Current, level, face);
            Check(rgba != view && rgba.pixelFormat == MTLPixelFormatRGBA8Unorm,
                  "A different view format reused the uint alias");
            write(view, 0xA0000000U | (pass << 16) | (level << 8) | face);
            if ((level * 6 + face) % 8 == 7) scheduler.Submit();
        }
    }
    scheduler.Finish();
    for (u32 level = 0; level < 4; ++level) for (u32 face = 0; face < 6; ++face)
        read(cube.Image(), level, face, 0xA0010000U | (level << 8) | face, 0xFFBF8040U);
    // Warm valid keys must not mask range/type validation.
    unsigned rejected = 0;
    for (const auto request : {std::pair{4U, 0U}, std::pair{0U, 6U}}) {
        try { (void)cube.StorageView(request.first, request.second); }
        catch (const std::invalid_argument&) { ++rejected; }
    }
    try { (void)cube.StencilView(); } catch (const std::invalid_argument&) { ++rejected; }
    Check(rejected == 3, "Native view cache bypassed range or format rejection");
    params.texture_type = VideoCore::TextureType::Texture2D;
    Metal::Surface surface{runtime, params};
    auto old_parent = surface.Image(); auto old_view = surface.StorageView();
    write(old_view, 0x10203040U);
    VideoCore::FramebufferParams fb_params{}; fb_params.shadow_rendering = true;
    Metal::Framebuffer framebuffer{runtime, fb_params, &surface, nullptr};
    Check(framebuffer.ImageView(VideoCore::SurfaceType::Color) == old_view,
          "Shadow framebuffer did not share the matching native storage alias");
    surface.ScaleUp(2); // Must preserve queued writes, invalidate aliases, and follow the new backing.
    auto scaled_parent = surface.Image(); auto scaled_view = surface.StorageView();
    Check(scaled_view != old_view && scaled_view.width == 16 &&
          framebuffer.ImageView(VideoCore::SurfaceType::Color) == scaled_view,
          "Native view cache or framebuffer retained a replaced backing");
    write(scaled_view, 0x50607080U);
    write(old_view, 0x11223344U); // A retained previous-allocation view remains independently usable.
    scheduler.Finish();
    auto old_bytes = transfer.Readback({old_parent, 0, 0, MTLRegionMake2D(0, 0, 1, 1)}, 4);
    auto scaled_bytes = transfer.Readback({scaled_parent, 0, 0, MTLRegionMake2D(0, 0, 1, 1)}, 4);
    u32 old_word, scaled_word;
    std::memcpy(&old_word, old_bytes.data(), 4); std::memcpy(&scaled_word, scaled_bytes.data(), 4);
    Check(old_word == 0x11223344U && scaled_word == 0x50607080U, "Old and replacement aliases crossed backing allocations");
    auto mip = surface.ImageView(Metal::SurfaceView::Mip0, Metal::SurfaceImage::Current, 1);
    auto tail = surface.ImageView(Metal::SurfaceView::Sample, Metal::SurfaceImage::Current, 1);
    Check(mip != tail && mip.mipmapLevelCount == 1 && tail.mipmapLevelCount == 3,
          "Native cache key omitted mip count");
    auto base_view = surface.ImageView(Metal::SurfaceView::Storage, Metal::SurfaceImage::Base);
    Check(base_view != scaled_view && base_view.width == 8, "Native cache key omitted base/current backing");
    Metal::Surface moved{std::move(surface)};
    Check(moved.StorageView() == scaled_view && framebuffer.ImageView(VideoCore::SurfaceType::Color) == scaled_view,
          "Moving a native surface lost its shared view allocation state");
    Metal::Surface assigned{runtime, params}; assigned = std::move(moved);
    Check(assigned.StorageView() == scaled_view, "Move assignment lost the source native view cache");
    assigned.ScaleUp(1);
    Check(assigned.StorageView().width == 8 && assigned.StorageView() != scaled_view &&
          framebuffer.ImageView(VideoCore::SurfaceType::Color) == assigned.StorageView(),
          "Scale-down did not invalidate aliases or follow the base backing");
    id<MTLTexture> released_parent;
    @autoreleasepool {
        Metal::Surface transient{runtime, params};
        released_parent = transient.Image();
        write(transient.StorageView(), 0x89ABCDEFU);
        scheduler.Submit();
    } // Surface and cached view owners released before waiting for GPU completion.
    scheduler.Finish();
    const auto released_bytes = transfer.Readback({released_parent, 0, 0, MTLRegionMake2D(0, 0, 1, 1)}, 4);
    u32 released_word; std::memcpy(&released_word, released_bytes.data(), 4);
    Check(released_word == 0x89ABCDEFU, "Submitted cached view did not survive destruction of its Surface");
    params.pixel_format = VideoCore::PixelFormat::D24S8; params.type = VideoCore::SurfaceType::DepthStencil;
    Metal::Surface depth{runtime, params};
    auto depth_view = depth.DepthView(), stencil_view = depth.StencilView();
    Check(depth_view == depth.DepthView() && stencil_view == depth.StencilView() && depth_view != stencil_view &&
          depth_view.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && stencil_view.pixelFormat == MTLPixelFormatX32_Stencil8,
          "Native depth/stencil view formats were merged or failed to reuse");
    try { (void)depth.StorageView(); } catch (const std::invalid_argument&) { ++rejected; }
    Check(rejected == 4, "Native cache allowed uint storage access to depth");
    scheduler.Finish();
    std::cout << "48 cube cached-view writes plus 4 scale/lifetime writes, 24 complete face/mip readbacks, 7 explicit submissions, format/count/backing keys, moves, scale replacement, Surface destruction and 4 rejections passed\n";
}

}

void ValidateReadbackSourceProfiles(Metal::Device& device) {
    struct Environment {
        std::map<std::string,std::optional<std::string>> saved;
        std::filesystem::path directory;
        Environment() {
            for(const auto* key:{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_HOST_ONLY","AZAHAR_METAL4_PROFILE_READBACK_SOURCES"}) {
                const auto* value=std::getenv(key);saved[key]=value ? std::optional<std::string>{value}:std::nullopt;
            }
            char path[]="/private/tmp/azahar-readback-source-test-XXXXXX";
            const auto* made=mkdtemp(path);Check(made,"Unable to create readback profile test directory");directory=made;
            setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
        }
        ~Environment() {
            for(const auto& [key,value]:saved) {
                if(value)setenv(key.c_str(),value->c_str(),1);else unsetenv(key.c_str());
            }
            std::error_code ec;std::filesystem::remove_all(directory,ec);
        }
    } environment;
    VideoCore::SurfaceParams params{};params.addr=0x123400;params.width=params.height=params.stride=8;
    params.pixel_format=VideoCore::PixelFormat::D16;params.is_tiled=true;params.UpdateParams();
    for(unsigned mode=0;mode<4;++mode) {
        const auto prefix=(environment.directory/std::to_string(mode)).string();
        if(mode==0)unsetenv("AZAHAR_METAL4_PROFILE");else setenv("AZAHAR_METAL4_PROFILE",prefix.c_str(),1);
        if(mode==1)unsetenv("AZAHAR_METAL4_PROFILE_READBACK_SOURCES");
        else setenv("AZAHAR_METAL4_PROFILE_READBACK_SOURCES",mode==2 ? "0":"1",1);
        {
            Metal::Scheduler queue{device};Metal::TextureRuntime runtime{device,queue};
            Metal::TransferRuntime transfer{device,queue};Metal::Surface depth{runtime,params};
            VideoCore::TextureClear clear{};clear.texture_rect={0,8,8,0};clear.value.depth=1.f;
            Check(runtime.ClearTexture(depth,clear),"Readback source test depth clear failed");
            std::vector<u8> bytes(params.size,0xa5);
            runtime.DownloadGuest(depth,params,bytes,NSMakeRange(0,bytes.size()));
            Check(std::all_of(bytes.begin(),bytes.end(),[](u8 x){return x==255;}),
                "Readback source profiling changed pending depth bytes");
            auto source=device.CreateBuffer(64,true),destination=device.CreateBuffer(64,true);
            std::memset(source.contents,0x5a,64);
            transfer.CopyBuffer(source,0,destination,0,64); // Unrelated pending GPU work.
            runtime.DownloadGuest(depth,params,bytes,NSMakeRange(0,bytes.size()));
            Check(std::all_of(bytes.begin(),bytes.end(),[](u8 x){return x==255;}) &&
                  !std::memcmp(source.contents,destination.contents,64),
                "Readback source profiling changed ready depth or queue completion");
            clear.value.depth=0.f;Check(runtime.ClearTexture(depth,clear),"Later depth write failed");
            std::fill(bytes.begin(),bytes.end(),0xa5);
            runtime.DownloadGuest(depth,params,bytes,NSMakeRange(3,9));
            for(unsigned i=0;i<bytes.size();++i)Check(bytes[i]==(i>=3 && i<12 ? 0:0xa5),
                "Readback source profiling changed partial bytes/guards after a later depth write");
            queue.Finish();
        }
        const auto path=prefix+".readback-sources.csv";
        if(mode<3) {Check(!std::filesystem::exists(path),"Readback source profile enabled without both optins");continue;}
        std::ifstream input(path);std::string line;Check(bool(std::getline(input,line)),"Missing readback source profile");
        const auto split=[](const std::string& s) {
            std::vector<std::string> out;std::stringstream in(s);std::string word;
            while(std::getline(in,word,','))out.push_back(word);return out;
        };
        const auto header=split(line);std::vector<std::map<std::string,u64>> rows;bool footer=false;
        while(std::getline(input,line)) {
            if(line.starts_with('#')) {footer=line=="# attempted=3,recorded=3,dropped=0";continue;}
            const auto words=split(line);Check(words.size()==header.size(),"Readback source CSV extent differs");
            std::map<std::string,u64> row;
            for(unsigned i=0;i<header.size();++i)row[header[i]]=std::stoull(words[i]);rows.push_back(std::move(row));
        }
        Check(rows.size()==3 && footer,"Readback source profile lost completed records or truncation status");
        Check(rows[0].at("current_last_use")==rows[0].at("tick_before") &&
              rows[0].at("current_ready")==0 && rows[0].at("resolved_ready")==0,
              "Pending unsubmitted depth source incorrectly reported complete");
        Check(rows[1].at("current_last_use")>0 && rows[1].at("current_last_use")<rows[1].at("tick_before") &&
              rows[1].at("current_ready")==1 && rows[1].at("resolved_ready")==1,
              "Completed source was confused with unrelated pending queue work");
        Check(rows[2].at("current_ready")==0 && rows[2].at("resolved_ready")==0 &&
              rows[2].at("byte_offset")==3 && rows[2].at("byte_length")==9,
              "Later queued depth write or partial extent was lost by source profiling");
        for(const auto& row:rows) {
            Check(row.at("end_host_ns")>=row.at("begin_host_ns") &&
                  row.at("end_host_ns")-row.at("begin_host_ns")==
                    row.at("prepare_ns")+row.at("encode_ns")+row.at("finish_ns")+row.at("copyout_ns") &&
                  row.at("raw_bytes")==params.size && row.at("address")==params.addr && row.at("depth")==1,
                  "Readback source timing or physical layout metadata differs");
        }
    }
    std::cout<<"Readback source profile: two optins, no-file controls, pending/ready-with-unrelated-work/later-write observations, exact full/partial depth bytes and timing extents passed\n";
}

// Compare the current native clear with an explicitly recreated scratch-image
// algorithm. GPU commit feedback includes all32 clears/copies; allocation time
// counts only the explicit temporary image allocation, never driver internals.
void ValidateClearCosts(const std::filesystem::path& output) {
    Check(output.is_absolute() && !std::filesystem::exists(output),"Clear cost output must be a new absolute path");
    Check(!std::getenv("AZAHAR_METAL4_PROFILE"),"Clear cost measurement requires frame profiling disabled");
    Metal::Device device;Metal::Scheduler scheduler{device};
    Metal::TextureRuntime runtime{device,scheduler};Metal::TransferRuntime transfer{device,scheduler};
    VideoCore::SurfaceParams params{};params.width=128;params.height=96;params.stride=128;
    params.pixel_format=VideoCore::PixelFormat::RGBA8;params.type=VideoCore::SurfaceType::Color;params.UpdateParams();
    Metal::Surface target{runtime,params};
    std::ofstream csv{output};csv<<"path,round,order,mode,calls,gpu_ms,host_clear_ms,explicit_temp_images,explicit_temp_alloc_ms,encoders,pixels_verified\n";
    unsigned verified{},submissions{};
    for(bool partial:{false,true})for(int round=-1;round<8;++round)for(unsigned order=0;order<4;++order) {
        const bool direct=((order==1 || order==2) != bool(round&1));
        const auto region=partial ? MTLRegionMake2D(7,5,100,80):MTLRegionMake2D(0,0,128,96);
        transfer.ClearColor(target.Image(),MTLClearColorMake(0,0,0,1));scheduler.Finish();
        VideoCore::TextureClear clear{};clear.texture_rect={u32(region.origin.x),u32(region.origin.y+region.size.height),
            u32(region.origin.x+region.size.width),u32(region.origin.y)};
        const auto promise=std::make_shared<std::promise<Metal::Scheduler::SubmissionTiming>>();auto ready=promise->get_future();
        scheduler.SetSubmissionObserver([promise](const auto& timing){promise->set_value(timing);});
        const auto encoders=scheduler.EncoderCount();unsigned allocations{};double allocation_ms{};
        const auto start=std::chrono::steady_clock::now();
        for(unsigned call=0;call<32;++call) {
            const bool green=call&1;clear.value.color={green ? 0.f:1.f,green ? 1.f:0.f,0.f,1.f};
            if(direct) Check(runtime.ClearTexture(target,clear),"Native clear cost call rejected");
            else {
                auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                    width:region.size.width height:region.size.height mipmapped:NO];
                descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
                const auto allocation_begin=std::chrono::steady_clock::now();
                auto scratch=device.CreateTexture(descriptor);
                allocation_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-allocation_begin).count();++allocations;
                transfer.ClearColor(scratch,MTLClearColorMake(green ? 0:1,green ? 1:0,0,1));
                transfer.CopyTexture({scratch,0,0,MTLRegionMake2D(0,0,region.size.width,region.size.height)},
                                     {target.Image(),0,0,region});
            }
        }
        const auto host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto tick=scheduler.Submit();scheduler.SetSubmissionObserver({});scheduler.Wait(tick);
        Check(ready.wait_for(std::chrono::seconds{5})==std::future_status::ready,"Missing native clear commit timing");
        const auto timing=ready.get();Check(!timing.failed && timing.tick==tick && std::isfinite(timing.gpu_begin_seconds) &&
            std::isfinite(timing.gpu_end_seconds) && timing.gpu_end_seconds>timing.gpu_begin_seconds && timing.gpu_begin_seconds>0,
            "Invalid native clear commit interval");
        const auto encoded=scheduler.EncoderCount()-encoders;
        const auto pixels=transfer.Readback({target.Image(),0,0,MTLRegionMake2D(0,0,128,96)},128*4);
        for(unsigned y=0;y<96;++y)for(unsigned x=0;x<128;++x)for(unsigned c=0;c<4;++c) {
            const bool inside=x>=region.origin.x && x<region.origin.x+region.size.width &&
                y>=region.origin.y && y<region.origin.y+region.size.height;
            const auto expected=(c==3 || (inside && c==1)) ? std::byte{255}:std::byte{};
            Check(pixels[(y*128+x)*4+c]==expected,"Clear cost algorithm changed raster bytes or outside neighbors");
        }
        if(round>=0) {
            csv<<(partial ? "rgba8_partial":"rgba8_full")<<','<<round<<','<<order<<','<<(direct ? "direct":"scratch")
                <<",32,"<<std::setprecision(12)<<(timing.gpu_end_seconds-timing.gpu_begin_seconds)*1000<<','<<host_ms<<','
                <<allocations<<','<<allocation_ms<<','<<encoded<<",12288\n";
            ++submissions;verified+=12288;
        }
    }
    csv.flush();Check(bool(csv) && submissions==64 && verified==786432,"Clear cost matrix lost samples or full raster checks");
    std::cout<<"Native clear cost matrix:64 ABBA/BAAB submissions,2048clear calls,786432 complete pixels; two warmup paths also checked; scratch allocations explicitly timed, no internal-driver allocation claim passed\n";
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            if(argc==3 && std::string_view{argv[1]}=="--clear-cost") {ValidateClearCosts(argv[2]);return 0;}
            Metal::Device device;
            ValidateReadbackSourceProfiles(device);
            Metal::Scheduler scheduler{device};
            Metal::TextureRuntime runtime{device, scheduler};
            Metal::TransferRuntime transfer{device, scheduler};
            ValidateDepthGuestEncoding(device, scheduler, runtime);
            ValidateStagingSnapshots(device, scheduler, runtime, transfer);
            ValidateViewCache(device, scheduler, runtime, transfer);
            ValidateFramebufferPassCache(device, scheduler, runtime, transfer);
            VideoCore::SurfaceParams params;
            params.width = 17;
            params.height = 11;
            params.levels = 3;
            params.res_scale = 2;
            params.pixel_format = VideoCore::PixelFormat::RGBA8;
            params.type = VideoCore::SurfaceType::Color;
            Metal::Surface color{runtime, params};
            const VideoCore::FramebufferParams fb_params{};
            Metal::Framebuffer fb{runtime, fb_params, &color, nullptr};
            auto upload = runtime.FindStaging(17 * 11 * 4, true);
            std::vector<u8> expected(upload.size);
            for (u32 i = 0; i < expected.size(); ++i) expected[i] = u8(i * 7);
            std::memcpy(upload.mapped.data(), expected.data(), expected.size());
            color.Upload(Copy(upload, 17, 11, 4), upload);
            for (u32 level = 1; level < 3; ++level) {
                const auto w = 17U >> level, h = 11U >> level;
                auto mip = runtime.FindStaging(w * h * 4, true);
                std::memset(mip.mapped.data(), 255, mip.size);
                color.Upload(Copy(mip, w, h, 4, level), mip);
                const auto native = color.Image();
                const auto pixels = transfer.Readback({native, level, 0, MTLRegionMake2D(0, 0, native.width >> level, native.height >> level)}, (native.width >> level) * 4);
                for (const auto value : pixels) Check(value == std::byte{255}, "Odd scaled mip upload left a right/bottom edge uninitialized");
            }
            VideoCore::TextureClear clear{};
            clear.texture_rect = {12, 8, 18, 2};
            clear.value.color = {17.f / 255, 34.f / 255, 51.f / 255, 68.f / 255};
            runtime.ClearTexture(color, clear);
            color.ScaleUp(3);
            Check(fb.Scale() == 3 && fb.RenderPass().colorAttachments[0].texture == color.Image(),
                  "Cached native framebuffer did not follow surface rescaling");
            auto download = runtime.FindStaging(17 * 11 * 4, false);
            color.Download(Copy(download, 17, 11, 4), download);
            for (u32 y = 0; y < 11; ++y) for (u32 x = 0; x < 17; ++x) for (u32 c = 0; c < 4; ++c) {
                const auto wanted = x >= 6 && x < 9 && y >= 1 && y < 4 ? 17U * (c + 1) : expected[(y * 17 + x) * 4 + c];
                Check(download.mapped[(y * 17 + x) * 4 + c] == wanted,
                      "Native scale replacement lost GPU-written contents or changed an outside pixel");
            }
            auto snapshot = color.CopyImageView();
            auto snapshot_pixels = transfer.Readback({snapshot, 2, 0, MTLRegionMake2D(0, 0, snapshot.width >> 2, snapshot.height >> 2)}, (snapshot.width >> 2) * 4);
            for (const auto value : snapshot_pixels) Check(value == std::byte{255}, "Native snapshot did not copy every valid mip extent");
            auto storage = color.StorageView();
            Check(storage.pixelFormat == MTLPixelFormatR32Uint, "Native RGBA8 storage view is not uint-addressable");
            Metal::Surface moved{std::move(color)};
            Check(fb.RenderPass().colorAttachments[0].texture == moved.Image(), "Native framebuffer lost its allocation when the surface moved");
            moved.ScaleUp(1);
            Check(fb.Scale() == 1 && fb.RenderPass().colorAttachments[0].texture == moved.Image(), "Native framebuffer did not follow scale-down to the base image");
            auto shadow_params = fb_params;
            shadow_params.shadow_rendering = true;
            Metal::Framebuffer shadow{runtime, shadow_params, &moved, nullptr};
            Check(!shadow.RenderPass().colorAttachments[0].texture && shadow.RenderPass().renderTargetWidth == 17,
                  "Native shadow framebuffer did not retain dimensions without a color attachment");
            auto writer_library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void write_shadow(texture2d<uint, access::write> target [[texture(0)]]) {
    target.write(uint4(0x12345678u, 0, 0, 0), uint2(0));
}
)", "Native uint surface alias test");
            auto* writer_descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
            writer_descriptor.computeFunctionDescriptor = Metal::Device::Function(writer_library, "write_shadow");
            auto writer = device.CompileCompute(writer_descriptor);
            const std::array<Metal::TextureBinding, 1> writer_textures{{{shadow.ImageView(VideoCore::SurfaceType::Color), 0}}};
            auto writer_table = scheduler.MakeArguments({}, writer_textures);
            scheduler.KeepAlive(writer);
            auto writer_encoder = scheduler.BeginCompute();
            [writer_encoder setComputePipelineState:writer];
            [writer_encoder setArgumentTable:writer_table];
            [writer_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            scheduler.EndEncoding();
            const auto alias = transfer.Readback({moved.Image(), 0, 0, MTLRegionMake2D(0, 0, 1, 1)}, 4);
            Check(alias == std::vector<std::byte>{std::byte{0x78}, std::byte{0x56}, std::byte{0x34}, std::byte{0x12}},
                  "Native uint storage writes did not alias the canonical RGBA8 surface bytes");
            // Upload/download every D16 value using actual native Depth16Unorm storage.
            for (const auto format : {VideoCore::PixelFormat::D16, VideoCore::PixelFormat::D24, VideoCore::PixelFormat::D24S8}) {
                params.width = params.height = 256;
                params.levels = 2;
                params.res_scale = 2;
                params.pixel_format = format;
                params.type = format == VideoCore::PixelFormat::D24S8 ? VideoCore::SurfaceType::DepthStencil : VideoCore::SurfaceType::Depth;
                Metal::Surface depth{runtime, params};
                const auto bpp = depth.GetInternalBytesPerPixel();
                auto depth_upload = runtime.FindStaging(256 * 256 * bpp, true);
                for (u32 i = 0; i < 65536; ++i) {
                    if (format == VideoCore::PixelFormat::D16) {
                        const auto value = u16(i);
                        std::memcpy(depth_upload.mapped.data() + i * 2, &value, 2);
                    } else if (format == VideoCore::PixelFormat::D24) {
                        const auto value = float(i) / 65535.f;
                        std::memcpy(depth_upload.mapped.data() + i * 4, &value, 4);
                    } else {
                        const auto value = ((i * 257U) << 8) | (i & 255U);
                        std::memcpy(depth_upload.mapped.data() + i * 4, &value, 4);
                    }
                }
                const auto wanted = std::vector<u8>{depth_upload.mapped.begin(), depth_upload.mapped.end()};
                depth.Upload(Copy(depth_upload, 256, 256, bpp), depth_upload);
                Metal::Framebuffer depth_fb{runtime, fb_params, nullptr, &depth};
                depth.ScaleUp(3);
                auto depth_download = runtime.FindStaging(256 * 256 * bpp, false);
                depth.Download(Copy(depth_download, 256, 256, bpp), depth_download);
                Check(std::equal(wanted.begin(), wanted.end(), depth_download.mapped.begin()), "Native depth staging or rescaling changed a D16/D24/D24S8 sample");
                Check(depth_fb.RenderPass().depthAttachment.texture == depth.Image(), "Cached native depth framebuffer retained an old allocation");
                auto mip_upload = runtime.FindStaging(128 * 128 * bpp, true);
                std::memcpy(mip_upload.mapped.data(), wanted.data(), mip_upload.size);
                depth.Upload(Copy(mip_upload, 128, 128, bpp, 1), mip_upload);
                const auto read_plane = [&](u32 level, bool stencil) {
                    const auto texture = depth.Image();
                    const auto width = texture.width >> level, height = texture.height >> level;
                    const auto options = stencil ? MTLBlitOptionStencilFromDepthStencil :
                        format == VideoCore::PixelFormat::D24S8 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone;
                    return transfer.Readback({texture, level, 0, MTLRegionMake2D(0, 0, width, height)},
                                             width * (stencil ? 1 : bpp), options);
                };
                const auto depth_before = read_plane(0, false), mip_before = read_plane(1, false);
                const auto stencil_before = format == VideoCore::PixelFormat::D24S8 ? read_plane(0, true) :
                    std::vector<std::byte>{};
                const auto stencil_mip_before = format == VideoCore::PixelFormat::D24S8 ? read_plane(1, true) :
                    std::vector<std::byte>{};
                VideoCore::TextureClear depth_clear{};
                depth_clear.texture_rect = {3, 8, 10, 2};
                depth_clear.value.depth = 1.f;
                depth_clear.value.stencil = 0xA5;
                runtime.ClearTexture(depth, depth_clear);
                auto cleared = transfer.Readback({depth.Image(), 0, 0, MTLRegionMake2D(3, 2, 7, 6)}, 7 * (format == VideoCore::PixelFormat::D16 ? 2 : 4),
                    format == VideoCore::PixelFormat::D24S8 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone);
                if (format == VideoCore::PixelFormat::D16) {
                    for (auto byte : cleared) Check(byte == std::byte{255}, "Native partial D16 clear failed");
                } else {
                    for (std::size_t i = 0; i < cleared.size(); i += 4) {
                        float value;
                        std::memcpy(&value, cleared.data() + i, 4);
                        Check(value == 1.f, "Native partial floating depth clear failed");
                    }
                }
                if (format == VideoCore::PixelFormat::D24S8) {
                    const auto stencil = transfer.Readback({depth.Image(), 0, 0, MTLRegionMake2D(3, 2, 7, 6)}, 7, MTLBlitOptionStencilFromDepthStencil);
                    for (auto byte : stencil) Check(byte == std::byte{0xA5}, "Native partial stencil clear failed");
                }
                const auto guard = [&](const std::vector<std::byte>& before, const std::vector<std::byte>& after,
                                       NSUInteger bytes_per_pixel) {
                    Check(after.size() == before.size(), "Partial native clear changed the readback extent");
                    const auto texture = depth.Image();
                    for (NSUInteger y = 0; y < texture.height; ++y) for (NSUInteger x = 0; x < texture.width; ++x) {
                        if (x >= 3 && x < 10 && y >= 2 && y < 8) continue;
                        for (NSUInteger byte = 0; byte < bytes_per_pixel; ++byte) {
                            const auto i = (y * texture.width + x) * bytes_per_pixel + byte;
                            Check(after[i] == before[i], "Partial native depth/stencil clear changed an outside pixel");
                        }
                    }
                };
                guard(depth_before, read_plane(0, false), bpp);
                Check(read_plane(1, false) == mip_before, "Partial native depth clear changed an untouched mip");
                if (format == VideoCore::PixelFormat::D24S8) {
                    guard(stencil_before, read_plane(0, true), 1);
                    Check(read_plane(1, true) == stencil_mip_before, "Partial native stencil clear changed an untouched mip");
                }
            }
            std::cout << "Native partial D16/D24/D24S8 clears preserve every outside depth/stencil pixel and untouched mip\n";
            params.width = params.height = 8;
            params.levels = 4;
            params.res_scale = 2;
            params.texture_type = VideoCore::TextureType::CubeMap;
            params.pixel_format = VideoCore::PixelFormat::RGBA8;
            params.type = VideoCore::SurfaceType::Texture;
            Metal::Surface cube{runtime, params};
            for (u32 face = 0; face < 6; ++face) for (u32 level = 0; level < 4; ++level) {
                VideoCore::TextureClear face_clear{};
                const auto dim = 16U >> level;
                face_clear.texture_level = level;
                face_clear.texture_layer = face;
                face_clear.texture_rect = {0, dim, dim, 0};
                face_clear.value.color = {float(face + 1) / 255, float(level + 1) / 255, 0, 1};
                runtime.ClearTexture(cube, face_clear);
            }
            cube.ScaleUp(3);
            auto cube_copy = cube.CopyImageView();
            for (u32 face = 0; face < 6; ++face) for (u32 level = 0; level < 4; ++level) {
                const auto dim = 24U >> level;
                const auto pixels = transfer.Readback({cube_copy, level, face, MTLRegionMake2D(0, 0, dim, dim)}, dim * 4);
                for (std::size_t i = 0; i < pixels.size(); i += 4)
                    Check(pixels[i] == std::byte(face + 1) && pixels[i + 1] == std::byte(level + 1) &&
                          pixels[i + 2] == std::byte{} && pixels[i + 3] == std::byte{255},
                          "Native cubemap clear, scale or snapshot selected the wrong face/mip");
            }
            for (u32 face = 0; face < 6; ++face) {
                const auto face_view = cube.StorageView(2, face);
                Check(face_view.width == 6 && face_view.height == 6 && face_view.textureType == MTLTextureType2D &&
                          face_view.pixelFormat == MTLPixelFormatR32Uint, "Native shadow-cube view selected the wrong mip/face/type");
                const auto words = transfer.Readback({face_view, 0, 0, MTLRegionMake2D(0, 0, 6, 6)}, 6 * 4);
                for (std::size_t i = 0; i < words.size(); i += 4)
                    Check(words[i] == std::byte(face + 1) && words[i + 1] == std::byte{3} &&
                          words[i + 2] == std::byte{} && words[i + 3] == std::byte{255},
                          "Native shadow-cube uint view did not alias the requested face");
            }
            std::cout << "30 native cube face/mip and uint-alias cases compare all four RGBA channels\n";
            using C = VideoCore::SamplerParams::TextureConfig;
            for (u32 wrap = 0; wrap < 8; ++wrap) {
                Metal::Sampler sampler{runtime, {C::Nearest, C::Linear, C::Nearest, C::WrapMode(wrap), C::Repeat, 0x12345678, 0, 3, -64}};
                Check(sampler.Handle() && sampler.Params().wrap_s == C::WrapMode(wrap), "Native sampler lost a PICA wrap mode");
            }
            runtime.Finish();
            Check(runtime.GetResourceFreeTick() >= runtime.GetResourceTick() - 1, "Native cache resource ticks did not advance after GPU completion");
            std::cout << "Native cache surfaces, bounded staging, partial clears, D16/D24/D24S8 staging, GPU-content-preserving scale changes, framebuffer relocation, uint storage views, all cubemap faces/mips and sampler descriptors passed\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
