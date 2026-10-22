// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_runtime.h"
#include "common/settings.h"
#include "video_core/custom_textures/material.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace Metal {

struct SurfaceTextureViewKey {
    id<MTLTexture> texture;
    MTLPixelFormat format{};
    u32 level{}, count{}, slice{};
    bool operator==(const SurfaceTextureViewKey& other) const {
        return texture == other.texture && format == other.format && level == other.level &&
               count == other.count && slice == other.slice;
    }
};
struct SurfaceTextureViewHash {
    std::size_t operator()(const SurfaceTextureViewKey& key) const {
        const auto backing = reinterpret_cast<std::uintptr_t>((__bridge void*)key.texture);
        return (backing >> 4) ^ (std::size_t(key.format) << 16) ^ (key.level << 12) ^
               (key.count << 6) ^ key.slice;
    }
};

struct SurfaceState {
    id<MTLTexture> base;
    id<MTLTexture> scaled;
    id<MTLTexture> current;
    id<MTLTexture> normal;
    id<MTLTexture> snapshot;
    VideoCore::PixelFormat format{VideoCore::PixelFormat::Invalid};
    u32 scale{1};
    bool depth{};
    bool custom{};
    // Only the three live image backings and their valid mip/slice/format ranges
    // can be requested. Replacing a backing clears aliases; submitted bindings
    // retain their views independently until GPU completion.
    std::unordered_map<SurfaceTextureViewKey, id<MTLTexture>, SurfaceTextureViewHash> views;
};

namespace {
u32 Scaled(u32 value, u32 scale) {
    if (!value || !scale || value > std::numeric_limits<u32>::max() / scale)
        throw std::invalid_argument("Invalid native surface extent or resolution scale");
    return value * scale;
}
u32 Slices(id<MTLTexture> texture) {
    return u32(texture.arrayLength) * (texture.textureType == MTLTextureTypeCube ? 6U : 1U);
}
TextureRegion Region(id<MTLTexture> texture, u32 level, u32 slice, Common::Rectangle<u32> rect) {
    const auto left = std::min(rect.left, rect.right);
    const auto bottom = std::min(rect.bottom, rect.top);
    return {texture, level, slice, MTLRegionMake2D(left, bottom,
            std::max(rect.left, rect.right) - left, std::max(rect.bottom, rect.top) - bottom)};
}
TextureRegion Full(id<MTLTexture> texture, u32 level, u32 slice = 0) {
    return {texture, level, slice, MTLRegionMake2D(0, 0,
            std::max<NSUInteger>(1, texture.width >> level),
            std::max<NSUInteger>(1, texture.height >> level))};
}
Common::Rectangle<u32> MapRect(Common::Rectangle<u32> rect, id<MTLTexture> source, id<MTLTexture> dest, u32 level) {
    const auto sw = std::max<NSUInteger>(1, source.width >> level), sh = std::max<NSUInteger>(1, source.height >> level);
    const auto dw = std::max<NSUInteger>(1, dest.width >> level), dh = std::max<NSUInteger>(1, dest.height >> level);
    return {u32(u64(rect.left) * dw / sw), u32(u64(rect.top) * dh / sh),
            u32(u64(rect.right) * dw / sw), u32(u64(rect.bottom) * dh / sh)};
}
id<MTLTexture> View(id<MTLTexture> texture, MTLPixelFormat format, u32 level, u32 count, u32 slice = 0) {
    if (!texture || level >= texture.mipmapLevelCount || !count || count > texture.mipmapLevelCount - level ||
        slice >= Slices(texture)) throw std::invalid_argument("Invalid native surface view range");
    auto* descriptor = [[MTLTextureViewDescriptor alloc] init];
    descriptor.pixelFormat = format;
    descriptor.textureType = MTLTextureType2D;
    descriptor.levelRange = NSMakeRange(level, count);
    descriptor.sliceRange = NSMakeRange(slice, 1);
    auto view = [texture newTextureViewWithDescriptor:descriptor];
    if (!view) throw std::runtime_error("Native surface texture view is unsupported");
    return view;
}
id<MTLTexture> CachedView(SurfaceState& state, id<MTLTexture> texture, MTLPixelFormat format,
                          u32 level, u32 count, u32 slice = 0) {
    // Validate before lookup, including cache hits, just like a newly created view.
    if (!texture || level >= texture.mipmapLevelCount || !count || count > texture.mipmapLevelCount - level ||
        slice >= Slices(texture)) throw std::invalid_argument("Invalid native surface view range");
    const SurfaceTextureViewKey key{texture, format, level, count, slice};
    if (const auto found = state.views.find(key); found != state.views.end()) return found->second;
    auto view = View(texture, format, level, count, slice);
    state.views.emplace(key, view);
    return view;
}
MTLPixelFormat CustomFormat(VideoCore::CustomPixelFormat format) {
    using F = VideoCore::CustomPixelFormat;
    switch (format) {
    case F::RGBA8: return MTLPixelFormatRGBA8Unorm;
    case F::BC1: return MTLPixelFormatBC1_RGBA;
    case F::BC3: return MTLPixelFormatBC3_RGBA;
    case F::BC5: return MTLPixelFormatBC5_RGUnorm;
    case F::BC7: return MTLPixelFormatBC7_RGBAUnorm;
    case F::ASTC4: return MTLPixelFormatASTC_4x4_LDR;
    case F::ASTC6: return MTLPixelFormatASTC_6x6_LDR;
    case F::ASTC8: return MTLPixelFormatASTC_8x6_LDR;
    default: throw std::invalid_argument("Unknown native custom texture format");
    }
}
std::array<u32, 3> CustomBlock(VideoCore::CustomPixelFormat format) {
    using F = VideoCore::CustomPixelFormat;
    switch (format) {
    case F::RGBA8: return {1, 1, 4};
    case F::BC1: return {4, 4, 8};
    case F::BC3: case F::BC5: case F::BC7: case F::ASTC4: return {4, 4, 16};
    case F::ASTC6: return {6, 6, 16};
    case F::ASTC8: return {8, 6, 16};
    default: throw std::invalid_argument("Unknown native custom texture block layout");
    }
}
void ValidateCopy(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging,
                  u32 bytes_per_pixel) {
    if (copy.texture_rect.left >= copy.texture_rect.right || copy.texture_rect.bottom >= copy.texture_rect.top ||
        copy.buffer_offset < staging.offset || copy.buffer_offset - staging.offset > staging.size ||
        copy.buffer_size > staging.size - (copy.buffer_offset - staging.offset))
        throw std::invalid_argument("Invalid native surface staging copy");
    const auto count = u64(copy.texture_rect.GetWidth()) * copy.texture_rect.GetHeight();
    if (count > copy.buffer_size / bytes_per_pixel)
        throw std::invalid_argument("Native surface staging copy exceeds its bounded data");
}
}

// Opt-in source observations; last-use ticks include reads and are NOT a
// last-write proof. No queue ordering, submissions or download bytes change.
struct TextureRuntime::ReadbackProfile {
    struct Row {
        u64 frame{},begin_ns{},end_ns{},tick_before{},tick_encode{},tick_after{};
        u64 current_use{},current_ready{},resolved_use{},resolved_ready{};
        u64 encoders_before{},encoders_encode{},encoders_after{};
        u64 prepare_ns{},encode_ns{},finish_ns{},copyout_ns{};
        u64 address{},width{},height{},stride{},format{},scale{},depth{};
        u64 raw_bytes{},byte_offset{},byte_length{},origin_x{},origin_y{};
    };
    static u64 Now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    explicit ReadbackProfile(const char* path):file{std::string(path)+".readback-sources.csv"} {
        if(!file) throw std::runtime_error("Unable to create requested Metal readback source profile");
        file<<"sequence,frame,begin_host_ns,end_host_ns,tick_before,tick_encode,tick_after,current_last_use,current_ready,resolved_last_use,resolved_ready,encoders_before,encoders_encode,encoders_after,prepare_ns,encode_ns,finish_ns,copyout_ns,address,width,height,stride,format,scale,depth,raw_bytes,byte_offset,byte_length,origin_x,origin_y\n";
    }
    ~ReadbackProfile() { file<<"# attempted="<<attempted<<",recorded="<<recorded<<",dropped="<<attempted-recorded<<'\n'; }
    void Write(const Row& r) {
        ++attempted;
        if(recorded==16384) return;
        ++recorded;
        const std::array values{attempted,r.frame,r.begin_ns,r.end_ns,r.tick_before,r.tick_encode,r.tick_after,
            r.current_use,r.current_ready,r.resolved_use,r.resolved_ready,r.encoders_before,r.encoders_encode,
            r.encoders_after,r.prepare_ns,r.encode_ns,r.finish_ns,r.copyout_ns,r.address,r.width,r.height,
            r.stride,r.format,r.scale,r.depth,r.raw_bytes,r.byte_offset,r.byte_length,r.origin_x,r.origin_y};
        for(std::size_t i=0;i<values.size();++i) {if(i) file<<',';file<<values[i];}
        file<<'\n';
    }
    std::ofstream file;
    u64 attempted{},recorded{};
};

TextureRuntime::TextureRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, transfers{device, scheduler}, conversions{device, scheduler}, guest_codec{device, scheduler}, filters{device,scheduler}, staging_pool{device,scheduler} {
    const auto* enabled=std::getenv("AZAHAR_METAL4_PROFILE_READBACK_SOURCES");
    const auto* path=std::getenv("AZAHAR_METAL4_PROFILE");
    if(enabled && std::string_view{enabled}=="1" && path && *path)
        readback_profile=std::make_shared<ReadbackProfile>(path);
}

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    auto& slice = upload ? upload_staging : download_staging;
    slice = staging_pool.Allocate(size);
    return {size, u32(slice.offset), {static_cast<u8*>(slice.data), size}};
}
id<MTLBuffer> TextureRuntime::StagingBuffer(const VideoCore::StagingData& staging, bool upload) const {
    const auto& slice = upload ? upload_staging : download_staging;
    auto buffer = slice.buffer;
    if (!buffer || staging.offset < slice.offset || staging.offset - slice.offset > slice.size ||
        staging.size > slice.size - (staging.offset - slice.offset) ||
        staging.mapped.size() != staging.size || staging.mapped.data() != static_cast<u8*>(buffer.contents) + staging.offset)
        throw std::invalid_argument("Native staging data does not refer to the current bounded allocation");
    return buffer;
}
u64 TextureRuntime::GetResourceTick() { return scheduler.CurrentTick(); }
u64 TextureRuntime::GetResourceFreeTick() {
    while (free_tick + 1 < scheduler.CurrentTick() && scheduler.IsComplete(free_tick + 1)) ++free_tick;
    return free_tick;
}
void TextureRuntime::Finish() { scheduler.Finish(); }

id<MTLTexture> TextureRuntime::Allocate(u32 width, u32 height, u32 levels, VideoCore::TextureType type,
                                       MTLPixelFormat format, bool compressed) {
    if (!width || !height || !levels || levels > VideoCore::MAX_PICA_LEVELS ||
        (type == VideoCore::TextureType::CubeMap && width != height) ||
        (type != VideoCore::TextureType::Texture2D && type != VideoCore::TextureType::CubeMap))
        throw std::invalid_argument("Invalid native surface allocation parameters");
    u32 max_levels = 1;
    for (u32 dim = std::max(width, height); dim > 1; dim >>= 1) ++max_levels;
    if (levels > max_levels) throw std::invalid_argument("Native surface has too many mip levels");
    auto* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = type == VideoCore::TextureType::CubeMap ? MTLTextureTypeCube : MTLTextureType2D;
    descriptor.pixelFormat = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.mipmapLevelCount = levels;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (!compressed) {
        descriptor.usage |= MTLTextureUsageRenderTarget;
        if (format == MTLPixelFormatRGBA8Unorm) descriptor.usage |= MTLTextureUsageShaderWrite;
    }
    return device.CreateTexture(descriptor);
}

bool TextureRuntime::NeedsConversion(const Surface& surface) const {
    return !surface.state->depth || surface.pixel_format == VideoCore::PixelFormat::D24;
}
void TextureRuntime::Blit(const TextureRegion& source, const TextureRegion& dest, bool linear, bool flip_x, bool flip_y) {
    const auto depth = [](MTLPixelFormat format) { return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float || format == MTLPixelFormatDepth32Float_Stencil8; };
    if (depth(source.texture.pixelFormat) && depth(dest.texture.pixelFormat))
        conversions.BlitDepthStencil(source, dest, flip_x, flip_y);
    else conversions.BlitColor(source, dest, linear, flip_x, flip_y);
}
bool TextureRuntime::BlitTextures(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
    if (source.state->depth != dest.state->depth) return false;
    const bool flip_x = (blit.src_rect.left > blit.src_rect.right) != (blit.dst_rect.left > blit.dst_rect.right);
    const bool flip_y = (blit.src_rect.bottom > blit.src_rect.top) != (blit.dst_rect.bottom > blit.dst_rect.top);
    Blit(Region(source.Image(), blit.src_level, blit.src_layer, blit.src_rect),
         Region(dest.Image(), blit.dst_level, blit.dst_layer, blit.dst_rect), false, flip_x, flip_y);
    return true;
}
void TextureRuntime::DisplayTransfer(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
    const bool flip_x = (blit.src_rect.left > blit.src_rect.right) != (blit.dst_rect.left > blit.dst_rect.right);
    const bool flip_y = (blit.src_rect.bottom > blit.src_rect.top) != (blit.dst_rect.bottom > blit.dst_rect.top);
    conversions.PicaColorTransfer(Region(source.Image(), blit.src_level, blit.src_layer, blit.src_rect),
        Region(dest.Image(), blit.dst_level, blit.dst_layer, blit.dst_rect), u32(dest.pixel_format), flip_x, flip_y);
}
bool TextureRuntime::CopyTextures(Surface& source, Surface& dest, std::span<const VideoCore::TextureCopy> copies) {
    if (source.state->depth != dest.state->depth) return false;
    for (const auto& copy : copies) {
        TextureRegion src{source.Image(), copy.src_level, copy.src_layer,
            MTLRegionMake2D(copy.src_offset.x, copy.src_offset.y, copy.extent.width, copy.extent.height)};
        const TextureRegion dst{dest.Image(), copy.dst_level, copy.dst_layer,
            MTLRegionMake2D(copy.dst_offset.x, copy.dst_offset.y, copy.extent.width, copy.extent.height)};
        if (source.state->custom && src.texture.textureType == MTLTextureType2D &&
            dst.texture.textureType == MTLTextureTypeCube &&
            copy.src_level < src.texture.mipmapLevelCount && copy.dst_level < dst.texture.mipmapLevelCount &&
            !copy.src_offset.x && !copy.src_offset.y && !copy.dst_offset.x && !copy.dst_offset.y &&
            copy.extent.width == std::max<u64>(1, (u64(source.width) * source.res_scale) >> copy.src_level) &&
            copy.extent.height == std::max<u64>(1, (u64(source.height) * source.res_scale) >> copy.src_level) &&
            copy.extent.width == std::max<NSUInteger>(1, dst.texture.width >> copy.dst_level) &&
            copy.extent.height == std::max<NSUInteger>(1, dst.texture.height >> copy.dst_level)) {
            // The shared cube cache describes a whole face in scaled guest
            // coordinates. Its replacement may have unrelated dimensions.
            // Sample the complete custom mip instead of its top-left crop.
            src.region = Full(src.texture, copy.src_level, copy.src_layer).region;
        }
        if (src.texture.pixelFormat == dst.texture.pixelFormat &&
            src.region.size.width == dst.region.size.width && src.region.size.height == dst.region.size.height)
            transfers.CopyTexture(src, dst);
        else Blit(src, dst, false);
    }
    return true;
}
bool TextureRuntime::Reinterpret(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy) {
    if (source.pixel_format == VideoCore::PixelFormat::D24S8 && dest.pixel_format == VideoCore::PixelFormat::RGBA8) {
        conversions.DepthStencilToRGBA8(
            {source.Image(), copy.src_level, copy.src_layer, MTLRegionMake2D(copy.src_offset.x, copy.src_offset.y, copy.extent.width, copy.extent.height)},
            {dest.Image(), copy.dst_level, copy.dst_layer, MTLRegionMake2D(copy.dst_offset.x, copy.dst_offset.y, copy.extent.width, copy.extent.height)});
        return true;
    }
    const auto source_format = u32(source.pixel_format), dest_format = u32(dest.pixel_format);
    if (source_format >= 2 && source_format <= 6 && dest_format >= 2 && dest_format <= 6 &&
        source.Image().pixelFormat == MTLPixelFormatRGBA8Unorm && dest.Image().pixelFormat == MTLPixelFormatRGBA8Unorm) {
        conversions.ReinterpretColor16(
            {source.Image(), copy.src_level, copy.src_layer, MTLRegionMake2D(copy.src_offset.x, copy.src_offset.y, copy.extent.width, copy.extent.height)},
            {dest.Image(), copy.dst_level, copy.dst_layer, MTLRegionMake2D(copy.dst_offset.x, copy.dst_offset.y, copy.extent.width, copy.extent.height)},
            source_format, dest_format);
        return true;
    }
    const auto src_bits = VideoCore::GetFormatBpp(source.pixel_format), dst_bits = VideoCore::GetFormatBpp(dest.pixel_format);
    if (src_bits != dst_bits || source.pixel_format == VideoCore::PixelFormat::ETC1 || source.pixel_format == VideoCore::PixelFormat::ETC1A4)
        return false;
    const GuestTextureLayout src_layout{copy.extent.width, copy.extent.height, copy.extent.width, source.pixel_format};
    const GuestTextureLayout dst_layout{copy.extent.width, copy.extent.height, copy.extent.width, dest.pixel_format};
    const auto raw_size = guest_codec.GuestBytes(src_layout);
    auto src_linear = device.CreateBuffer(guest_codec.LinearBytes(src_layout), false);
    auto raw = device.CreateBuffer(raw_size, false);
    transfers.FillBuffer(raw, NSMakeRange(0, raw_size), 0);
    auto dst_linear = device.CreateBuffer(guest_codec.LinearBytes(dst_layout), false);
    const TextureRegion src_region{source.Image(), copy.src_level, copy.src_layer,
        MTLRegionMake2D(copy.src_offset.x, copy.src_offset.y, copy.extent.width, copy.extent.height)};
    const TextureRegion dst_region{dest.Image(), copy.dst_level, copy.dst_layer,
        MTLRegionMake2D(copy.dst_offset.x, copy.dst_offset.y, copy.extent.width, copy.extent.height)};
    if (source.state->depth)
        conversions.DepthStencilToBuffer(src_region, src_linear, 0, copy.extent.width * source.GetInternalBytesPerPixel(), source.depth_staging);
    else transfers.Download(src_region, src_linear, {0, copy.extent.width * 4U, 0, MTLBlitOptionNone});
    // Re-encode the source guest bits before decoding the destination guest format. Both
    // canonical host textures may be RGBA8 while their guest bit layouts are different.
    guest_codec.Encode(src_layout, {src_linear, 0, src_linear.length, nullptr, {}}, {raw, 0, raw.length, nullptr, {}}, NSMakeRange(0, raw_size));
    guest_codec.Decode(dst_layout, {raw, 0, raw.length, nullptr, {}}, {dst_linear, 0, dst_linear.length, nullptr, {}});
    if (dest.state->depth)
        conversions.BufferToDepthStencil(dst_linear, 0, copy.extent.width * dest.GetInternalBytesPerPixel(), dst_region, dest.depth_staging);
    else transfers.Upload(dst_linear, {0, copy.extent.width * 4U, 0, MTLBlitOptionNone}, dst_region);
    return true;
}
bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear) {
    const auto dest = Region(surface.Image(), clear.texture_level, clear.texture_layer, clear.texture_rect);
    const auto color = surface.state->depth ? MTLClearColorMake(0, 0, 0, 0) :
        MTLClearColorMake(clear.value.color.x, clear.value.color.y, clear.value.color.z, clear.value.color.w);
    conversions.ClearRegion(dest, color, surface.state->depth ? clear.value.depth : 0.0,
                            surface.state->depth ? clear.value.stencil : 0U);
    return true;
}
void TextureRuntime::GenerateMipmaps(Surface& surface) {
    auto texture = surface.Image();
    if (!texture) throw std::invalid_argument("Cannot generate mipmaps for an empty native surface");
    if (texture.mipmapLevelCount == 1) return;
    if (!surface.state->depth) {
        transfers.GenerateMipmaps(texture);
        if (surface.state->normal && surface.state->normal.mipmapLevelCount > 1)
            transfers.GenerateMipmaps(surface.state->normal);
    }
    else for (u32 slice = 0; slice < Slices(texture); ++slice)
        for (u32 level = 1; level < texture.mipmapLevelCount; ++level)
            Blit(Full(texture, level - 1, slice), Full(texture, level, slice), false);
}

namespace {
GuestTextureLayout GuestLayout(const VideoCore::SurfaceParams& params, bool converted = true, bool flip_output = false) {
    return {params.width, params.height, params.stride ? params.stride : params.width, params.pixel_format,
            params.is_tiled, converted, flip_output};
}
u32 StagingSize(NSUInteger bytes) {
    if (bytes > std::numeric_limits<u32>::max()) throw std::overflow_error("Native guest staging exceeds the cache's size representation");
    return u32(bytes);
}
}
Settings::TextureFilter TextureRuntime::TextureFilterMode() const {
    return texture_filter ? texture_filter() : Settings::TextureFilter::NoFilter;
}
void TextureRuntime::UploadGuest(Surface& surface, const VideoCore::SurfaceParams& params, std::span<const u8> bytes,
                                std::optional<Common::Rectangle<u32>> filter_damage) {
    const auto layout = GuestLayout(params);
    const auto raw_size = guest_codec.GuestBytes(layout);
    if (bytes.size() < raw_size || surface.pixel_format != params.pixel_format)
        throw std::invalid_argument("Native guest upload data does not cover its physical layout");
    auto raw = staging_pool.Allocate(raw_size);
    std::memcpy(raw.data, bytes.data(), raw_size);
    const auto staging = FindStaging(StagingSize(guest_codec.LinearBytes(layout)), true);
    guest_codec.Decode(layout, raw, upload_staging);
    surface.Upload({staging.offset, staging.size, surface.GetSubRect(params), surface.LevelOf(params.addr)}, staging,filter_damage);
}
void TextureRuntime::CopyGuest(const GuestCopyLayout& layout, const BufferSlice& source, const BufferSlice& dest) {
    guest_codec.Copy(layout, source, dest);
}
void TextureRuntime::DownloadGuest(Surface& surface, const VideoCore::SurfaceParams& params, std::span<u8> bytes,
                                    NSRange byte_range) {
    const auto layout = GuestLayout(params);
    const auto raw_size = guest_codec.GuestBytes(layout);
    if (bytes.size() < raw_size || surface.pixel_format != params.pixel_format || byte_range.location > raw_size ||
        byte_range.length > raw_size - byte_range.location)
        throw std::invalid_argument("Native guest download data does not cover its physical layout or flush range");
    // Disabled: no clocks, output allocation or additional timeline observations.
    std::optional<ReadbackProfile::Row> profile;
    if(readback_profile) {
        auto& row=profile.emplace();row.begin_ns=ReadbackProfile::Now();
        row.frame=scheduler.CurrentFrame();row.tick_before=scheduler.CurrentTick();
        row.encoders_before=scheduler.EncoderCount();row.address=params.addr;
        row.width=layout.width;row.height=layout.height;row.stride=layout.stride;
        row.format=u64(layout.format);row.scale=surface.res_scale;
        row.depth=surface.state && surface.state->depth;row.raw_bytes=raw_size;
        row.byte_offset=byte_range.location;row.byte_length=byte_range.length;
        row.current_use=scheduler.LastUseTick(surface.state ? surface.state->current:nil);
        row.current_ready=row.current_use && scheduler.IsComplete(row.current_use);
    }
    auto raw = staging_pool.Allocate(raw_size);
    // Preserve padding, unused nibbles, and bytes outside the requested physical flush.
    std::memcpy(raw.data, bytes.data(), raw_size);
    if (surface.state && surface.state->depth) {
        // Preserve the same scaled-to-canonical update, then read cached plane
        // views directly into guest bytes without an intermediate linear buffer.
        const auto source = surface.PrepareDownload(surface.GetSubRect(params), surface.LevelOf(params.addr));
        const auto depth = CachedView(*surface.state, source.texture, source.texture.pixelFormat, source.level, 1);
        const auto stencil = surface.pixel_format == VideoCore::PixelFormat::D24S8
            ? CachedView(*surface.state, source.texture, MTLPixelFormatX32_Stencil8, source.level, 1) : nil;
        if(profile) {
            profile->resolved_use=scheduler.LastUseTick(source.texture);
            profile->resolved_ready=profile->resolved_use && scheduler.IsComplete(profile->resolved_use);
            profile->origin_x=source.region.origin.x;profile->origin_y=source.region.origin.y;
            profile->tick_encode=scheduler.CurrentTick();
            profile->encoders_encode=scheduler.EncoderCount();
            profile->prepare_ns=ReadbackProfile::Now()-profile->begin_ns;
        }
        guest_codec.EncodeDepth(layout, depth, stencil, source.region.origin, raw, byte_range);
        if(profile) profile->encode_ns=ReadbackProfile::Now()-profile->begin_ns-profile->prepare_ns;
        scheduler.Finish(Scheduler::WaitReason::Readback);
    } else {
        const auto staging = FindStaging(StagingSize(guest_codec.LinearBytes(layout)), false);
        surface.EncodeDownload({staging.offset, staging.size, surface.GetSubRect(params), surface.LevelOf(params.addr)}, staging);
        if(profile) {
            profile->tick_encode=scheduler.CurrentTick();profile->encoders_encode=scheduler.EncoderCount();
            profile->prepare_ns=ReadbackProfile::Now()-profile->begin_ns;
        }
        guest_codec.Encode(layout, download_staging, raw, byte_range);
        if(profile) profile->encode_ns=ReadbackProfile::Now()-profile->begin_ns-profile->prepare_ns;
        scheduler.Finish(Scheduler::WaitReason::Readback);
    }
    if(profile) profile->finish_ns=ReadbackProfile::Now()-profile->begin_ns-profile->prepare_ns-profile->encode_ns;
    std::memcpy(bytes.data() + byte_range.location, static_cast<u8*>(raw.data) + byte_range.location, byte_range.length);
    if(profile) {
        profile->end_ns=ReadbackProfile::Now();
        profile->copyout_ns=profile->end_ns-profile->begin_ns-profile->prepare_ns-profile->encode_ns-profile->finish_ns;
        profile->tick_after=scheduler.CurrentTick();profile->encoders_after=scheduler.EncoderCount();
        readback_profile->Write(*profile);
    }
}
std::vector<u8> TextureRuntime::DecodeGuest(const VideoCore::SurfaceParams& params, std::span<const u8> bytes,
                                           bool converted, bool flip_output) {
    const auto layout = GuestLayout(params, converted, flip_output);
    const auto raw_size = guest_codec.GuestBytes(layout);
    if (bytes.size() < raw_size) throw std::invalid_argument("Native guest readback source is too short");
    auto raw = staging_pool.Allocate(raw_size), linear = staging_pool.Allocate(guest_codec.LinearBytes(layout));
    std::memcpy(raw.data, bytes.data(), raw_size);
    guest_codec.Decode(layout, raw, linear);
    scheduler.Finish(Scheduler::WaitReason::Readback);
    const auto* begin = static_cast<const u8*>(linear.data);
    return {begin, begin + linear.size};
}
void TextureRuntime::DumpGuest(const VideoCore::SurfaceParams& params, u32 level, u64 hash, std::span<const u8> bytes) {
    if (!texture_dump_sink) throw std::logic_error("Native Metal texture dump sink has not been installed");
    texture_dump_sink(params, level, hash, DecodeGuest(params, bytes, true, true));
}
void TextureRuntime::FillGuest(std::span<u8> bytes, std::array<u8, 4> pattern, u32 pattern_size, u32 phase) {
    if (bytes.empty()) return;
    auto buffer = staging_pool.Allocate(bytes.size());
    guest_codec.Fill(buffer, pattern, pattern_size, phase);
    scheduler.Finish(Scheduler::WaitReason::Readback);
    std::memcpy(bytes.data(), buffer.data, bytes.size());
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params,
                 const VideoCore::SurfaceFlagBits& initial_flags)
    : SurfaceBase{params, initial_flags}, runtime{runtime_}, state{std::make_shared<SurfaceState>()} {
    state->format = pixel_format;
    state->scale = res_scale;
    if (pixel_format == VideoCore::PixelFormat::Invalid || type == VideoCore::SurfaceType::Fill) return;
    const auto raw = u32(pixel_format);
    if (raw >= VideoCore::PIXEL_FORMAT_COUNT || raw == 15) throw std::invalid_argument("Invalid native guest surface format");
    state->depth = raw >= u32(VideoCore::PixelFormat::D16);
    depth_staging = pixel_format == VideoCore::PixelFormat::D16 ? DepthStagingFormat::D16
                    : pixel_format == VideoCore::PixelFormat::D24 ? DepthStagingFormat::D32Float : DepthStagingFormat::D24S8;
    const auto format = pixel_format == VideoCore::PixelFormat::D16 ? MTLPixelFormatDepth16Unorm
                        : pixel_format == VideoCore::PixelFormat::D24 ? MTLPixelFormatDepth32Float
                        : state->depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA8Unorm;
    state->base = runtime.Allocate(width, height, levels, texture_type, format);
    if (res_scale != 1) state->scaled = runtime.Allocate(Scaled(width, res_scale), Scaled(height, res_scale), levels, texture_type, format);
    state->current = state->scaled ? state->scaled : state->base;
}
Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceBase& surface, const VideoCore::Material* mat)
    : SurfaceBase{surface}, runtime{runtime_}, state{std::make_shared<SurfaceState>()} {
    if (!mat || !mat->Map(VideoCore::MapType::Color)) throw std::invalid_argument("Missing native custom texture material");
    state->format = pixel_format;
    state->scale = res_scale;
    state->custom = true;
    material = mat;
    custom_format = mat->format;
    flags |= VideoCore::SurfaceFlagBits::Custom;
    const auto format = CustomFormat(mat->format);
    state->base = runtime.Allocate(mat->width, mat->height, levels, texture_type, format, mat->format != VideoCore::CustomPixelFormat::RGBA8);
    // Canonical native color images support sampling, render-target copies, and generated mips
    // irrespective of the replacement material's compressed block format.
    state->scaled = runtime.Allocate(mat->width, mat->height, levels, texture_type, MTLPixelFormatRGBA8Unorm);
    state->current = state->scaled;
    if (mat->Map(VideoCore::MapType::Normal)) state->normal = runtime.Allocate(mat->width, mat->height, levels, texture_type, MTLPixelFormatRGBA8Unorm);
}
Surface::~Surface() = default;
Surface& Surface::operator=(Surface&& other) {
    if (this == &other) return *this;
    if (&runtime != &other.runtime) throw std::invalid_argument("Native surfaces cannot move between runtimes");
    SurfaceBase::operator=(std::move(other));
    state = std::move(other.state);
    depth_staging = other.depth_staging;
    return *this;
}
id<MTLTexture> Surface::Image(SurfaceImage image) const {
    if (!state) throw std::logic_error("Moved-from native surface");
    switch (image) {
    case SurfaceImage::Current: return state->current;
    case SurfaceImage::Base: return state->base;
    case SurfaceImage::Normal: return state->normal;
    }
    throw std::invalid_argument("Unknown native surface image");
}
id<MTLTexture> Surface::ImageView(SurfaceView view, SurfaceImage image, u32 level, u32 slice) const {
    auto texture = Image(image);
    if (!texture) throw std::invalid_argument("Empty native surface image");
    switch (view) {
    case SurfaceView::Sample:
        if (!level && !slice) return texture;
        return CachedView(*state, texture, texture.pixelFormat, level, level < texture.mipmapLevelCount ? u32(texture.mipmapLevelCount) - level : 0U, slice);
    case SurfaceView::Mip0: return CachedView(*state, texture, texture.pixelFormat, level, 1, slice);
    case SurfaceView::Depth:
        if (!state->depth) throw std::invalid_argument("Color surface has no depth view");
        return CachedView(*state, texture, texture.pixelFormat, level, level < texture.mipmapLevelCount ? u32(texture.mipmapLevelCount) - level : 0U, slice);
    case SurfaceView::Stencil:
        if (texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) throw std::invalid_argument("Surface has no stencil view");
        return CachedView(*state, texture, MTLPixelFormatX32_Stencil8, level, level < texture.mipmapLevelCount ? u32(texture.mipmapLevelCount) - level : 0U, slice);
    case SurfaceView::Storage:
        if (state->depth || texture.pixelFormat != MTLPixelFormatRGBA8Unorm) throw std::invalid_argument("Native storage view requires canonical RGBA8");
        return CachedView(*state, texture, MTLPixelFormatR32Uint, level, 1, slice);
    }
    throw std::invalid_argument("Unknown native surface view");
}
id<MTLTexture> Surface::CopyImageView() {
    const auto source = Image();
    if (!source) throw std::invalid_argument("Cannot snapshot an empty native surface");
    if (!state->snapshot || state->snapshot.width != source.width || state->snapshot.height != source.height ||
        state->snapshot.mipmapLevelCount != source.mipmapLevelCount)
        state->snapshot = runtime.Allocate(u32(source.width), u32(source.height), u32(source.mipmapLevelCount), texture_type, source.pixelFormat);
    for (u32 slice = 0; slice < Slices(source); ++slice)
        for (u32 level = 0; level < source.mipmapLevelCount; ++level)
            runtime.transfers.CopyTexture(Full(source, level, slice), Full(state->snapshot, level, slice));
    return state->snapshot;
}
u32 Surface::GetInternalBytesPerPixel() const { return depth_staging == DepthStagingFormat::D16 && state->depth ? 2 : 4; }
void Surface::Upload(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging,
                     std::optional<Common::Rectangle<u32>> filter_damage) {
    if(!state || !state->base || copy.texture_level>=state->base.mipmapLevelCount)
        throw std::invalid_argument("Native surface upload has no valid base mip");
    ValidateCopy(copy, staging, GetInternalBytesPerPixel());
    const auto source_full=Full(state->base,copy.texture_level);
    const u64 sw=source_full.region.size.width,sh=source_full.region.size.height;
    if(copy.texture_rect.right>sw || copy.texture_rect.top>sh)
        throw std::invalid_argument("Native surface upload exceeds its base mip");
    const auto filter=state->scaled && !state->depth ? runtime.TextureFilterMode() : Settings::TextureFilter::NoFilter;
    if(u32(filter)>u32(Settings::TextureFilter::MMPX)) throw std::invalid_argument("Unknown native surface texture filter");
    const auto damage=filter_damage.value_or(copy.texture_rect);
    if(filter!=Settings::TextureFilter::NoFilter &&
       (damage.left>=damage.right || damage.bottom>=damage.top || damage.right>sw || damage.top>sh))
        throw std::invalid_argument("Native filter damage exceeds the source mip");
    const auto buffer = runtime.StagingBuffer(staging, true);
    const auto dest = Region(state->base, copy.texture_level, 0, copy.texture_rect);
    const auto pitch = dest.region.size.width * GetInternalBytesPerPixel();
    if (state->depth) runtime.conversions.BufferToDepthStencil(buffer, copy.buffer_offset, pitch, dest, depth_staging);
    else runtime.transfers.Upload(buffer, {copy.buffer_offset, pitch, 0, MTLBlitOptionNone}, dest);
    if (state->scaled) {
        const auto scaled=Region(state->scaled,copy.texture_level,0,MapRect(copy.texture_rect,state->base,state->scaled,copy.texture_level));
        if(!state->depth && filter!=Settings::TextureFilter::NoFilter) {
            const auto dest_full=Full(state->scaled,copy.texture_level);
            const u64 dw=dest_full.region.size.width,dh=dest_full.region.size.height;
            // Eight guest texels conservatively cover bicubic/ScaleForce and
            // xBRZ/MMPX taps, plus the Anime4K Sobel/refine dependency chain.
            constexpr u32 halo=8;
            const u64 left=damage.left>halo ? damage.left-halo : 0;
            const u64 bottom=damage.bottom>halo ? damage.bottom-halo : 0;
            const u64 right=std::min<u64>(sw,u64(damage.right)+halo);
            const u64 top=std::min<u64>(sh,u64(damage.top)+halo);
            const auto x=left*dw/sw,y=bottom*dh/sh;
            const auto end_x=(right*dw+sw-1)/sw,end_y=(top*dh+sh-1)/sh;
            // Keep the full-mip viewport/UV phase even for odd scaled mip
            // dimensions. Only the affected pixels are written by scissor.
            runtime.filters.Filter(filter,source_full,dest_full,float(res_scale),false,false,
                                   MTLRegionMake2D(x,y,end_x-x,end_y-y));
        } else
            runtime.Blit(dest,scaled,false);
    }
}
void Surface::Download(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging) {
    EncodeDownload(copy, staging);
    runtime.scheduler.Finish(Scheduler::WaitReason::Readback);
}
void Surface::EncodeDownload(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging) {
    if (!state || !state->base || !state->current || copy.texture_level >= levels ||
        copy.texture_level >= state->current.mipmapLevelCount)
        throw std::invalid_argument("Native surface download has no valid source mip");
    ValidateCopy(copy, staging, GetInternalBytesPerPixel());
    const auto buffer = runtime.StagingBuffer(staging, false);
    const auto source = PrepareDownload(copy.texture_rect, copy.texture_level);
    const auto pitch = source.region.size.width * GetInternalBytesPerPixel();
    if (state->depth) runtime.conversions.DepthStencilToBuffer(source, buffer, copy.buffer_offset, pitch, depth_staging);
    else runtime.transfers.Download(source, buffer, {copy.buffer_offset, pitch, 0, MTLBlitOptionNone});
}
TextureRegion Surface::PrepareDownload(Common::Rectangle<u32> rect, u32 level) {
    if (!state || !state->base || !state->current || level >= levels ||
        level >= state->current.mipmapLevelCount)
        throw std::invalid_argument("Native surface download has no valid source mip");
    const auto guest_width = std::max(1U, width >> level);
    const auto guest_height = std::max(1U, height >> level);
    if (rect.left >= rect.right || rect.bottom >= rect.top || rect.right > guest_width || rect.top > guest_height)
        throw std::invalid_argument("Native surface download exceeds its guest mip");
    TextureRegion source;
    if (state->custom) {
        // Replacement dimensions and block formats describe sampling storage,
        // not the guest's physical layout. Convert the entire guest mip before
        // selecting a partial download, preserving the global sampling phase.
        auto canonical = runtime.Allocate(guest_width, guest_height, 1,
                                         VideoCore::TextureType::Texture2D, MTLPixelFormatRGBA8Unorm);
        runtime.Blit(Full(state->current, level), Full(canonical, 0), false);
        source = Region(canonical, 0, 0, rect);
    } else {
        source = Region(state->base, level, 0, rect);
        if (state->scaled) runtime.Blit(Region(state->scaled, level, 0, MapRect(rect, state->base, state->scaled, level)), source, false);
    }
    return source;
}
void Surface::UploadCustom(const VideoCore::Material* mat, u32 level) {
    if (!state || !state->custom || !mat || !mat->Map(VideoCore::MapType::Color) || level >= levels)
        throw std::invalid_argument("Invalid native custom texture upload");
    const auto region = Full(state->scaled, level);
    if (mat->width != region.region.size.width || mat->height != region.region.size.height)
        throw std::invalid_argument("Native custom material does not match its destination mip");
    // The cache resolves a separate Material for each guest mip hash. Validate
    // every map before either color or normal is changed, without requiring the
    // base material's pointer identity for lower mip uploads.
    for (auto map : {VideoCore::MapType::Color, VideoCore::MapType::Normal}) {
        const auto* texture = mat->Map(map);
        if (!texture) continue;
        const auto block = CustomBlock(texture->format);
        const auto columns = (region.region.size.width + block[0] - 1) / block[0];
        const auto rows = (region.region.size.height + block[1] - 1) / block[1];
        if (texture->width != mat->width || texture->height != mat->height ||
            texture->data.size() != columns * rows * block[2])
            throw std::invalid_argument("Native custom mip data does not match the requested mip dimensions");
    }
    if (mat->Map(VideoCore::MapType::Normal) && !state->normal) {
        auto normal = runtime.Allocate(u32(state->scaled.width), u32(state->scaled.height), levels,
                                       texture_type, MTLPixelFormatRGBA8Unorm);
        // A normal map can first appear on a separately resolved lower mip.
        // Other mips must then sample a defined flat normal, not private memory.
        for (u32 slice = 0; slice < Slices(normal); ++slice)
            for (u32 mip = 0; mip < levels; ++mip)
                runtime.transfers.ClearColor(normal, MTLClearColorMake(.5, .5, 1, 1), mip, slice);
        state->normal = normal;
    }
    const auto upload = [&](VideoCore::MapType map, id<MTLTexture> target) {
        const auto* texture = mat->Map(map);
        if (!texture) return;
        const auto block = CustomBlock(texture->format);
        const auto columns = (region.region.size.width + block[0] - 1) / block[0];
        const auto rows = (region.region.size.height + block[1] - 1) / block[1];
        const auto bytes = columns * rows * block[2];
        auto buffer = runtime.device.CreateBuffer(bytes, true);
        std::memcpy(buffer.contents, texture->data.data(), bytes);
        auto encoded = runtime.Allocate(u32(region.region.size.width), u32(region.region.size.height), 1,
                                       VideoCore::TextureType::Texture2D, CustomFormat(texture->format), texture->format != VideoCore::CustomPixelFormat::RGBA8);
        runtime.transfers.Upload(buffer, {0, columns * block[2], 0, MTLBlitOptionNone}, Full(encoded, 0));
        runtime.Blit(Full(encoded, 0), Full(target, level), false);
        if (map == VideoCore::MapType::Color && encoded.pixelFormat == state->base.pixelFormat)
            runtime.transfers.CopyTexture(Full(encoded, 0), Full(state->base, level));
    };
    upload(VideoCore::MapType::Color, state->scaled);
    upload(VideoCore::MapType::Normal, state->normal);
}
void Surface::ScaleUp(u32 scale) {
    if (!scale) throw std::invalid_argument("Native resolution scale cannot be zero");
    if (scale == res_scale) return;
    if (!state->custom) {
        auto source = Image();
        auto dest = scale == 1 ? state->base : runtime.Allocate(Scaled(width, scale), Scaled(height, scale), levels, texture_type, source.pixelFormat);
        for (u32 slice = 0; slice < Slices(source); ++slice)
            for (u32 level = 0; level < levels; ++level) runtime.Blit(Full(source, level, slice), Full(dest, level, slice), false);
        state->views.clear();
        state->scaled = scale == 1 ? nil : dest;
        state->current = dest;
        state->snapshot = nil;
    }
    res_scale = scale;
    state->scale = scale;
}

Framebuffer::Framebuffer(TextureRuntime&, const VideoCore::FramebufferParams& params, Surface* color, Surface* depth)
    : VideoCore::FramebufferParams{params}, states{color ? color->state : nullptr, depth ? depth->state : nullptr} {
    if ((states[0] && states[0]->depth) || (states[1] && !states[1]->depth))
        throw std::invalid_argument("Native framebuffer attachment types do not match their slots");
}
u32 Framebuffer::Scale() const { return states[0] ? states[0]->scale : states[1] ? states[1]->scale : 1; }
VideoCore::PixelFormat Framebuffer::Format(VideoCore::SurfaceType type) const {
    const auto& state = states[Index(type)];
    return state ? state->format : VideoCore::PixelFormat::Invalid;
}
id<MTLTexture> Framebuffer::ImageView(VideoCore::SurfaceType type) const {
    const auto& state = states[Index(type)];
    if (!state || !state->current) return nil;
    if (shadow_rendering && Index(type) == 0) return CachedView(*state, state->current, MTLPixelFormatR32Uint, color_level, 1);
    return state->current;
}
MTL4RenderPassDescriptor* Framebuffer::RenderPass() const {
    const RenderPassKey key{states[0] ? states[0]->current : nil,
                           states[1] && !shadow_rendering ? states[1]->current : nil,
                           color_level, depth_level, shadow_rendering,
                           states[1] && states[1]->format == VideoCore::PixelFormat::D24S8};
    // Texture dimensions and mip counts are immutable. A validated matching key
    // can reuse the complete descriptor, including its fixed Load/Store actions.
    if (render_pass && key == render_pass_key) return render_pass;
    NSUInteger width = std::numeric_limits<NSUInteger>::max(), height = width;
    const auto dimensions = [&](const std::shared_ptr<SurfaceState>& state, u32 level) {
        if (!state || !state->current || level >= state->current.mipmapLevelCount)
            throw std::invalid_argument("Missing native framebuffer attachment or mip level");
        width = std::min(width, std::max<NSUInteger>(1, state->current.width >> level));
        height = std::min(height, std::max<NSUInteger>(1, state->current.height >> level));
    };
    if (states[0]) dimensions(states[0], color_level);
    if (states[1] && !shadow_rendering) dimensions(states[1], depth_level);
    if (width == std::numeric_limits<NSUInteger>::max()) throw std::invalid_argument("Native framebuffer has no dimensions");
    auto* pass = [[MTL4RenderPassDescriptor alloc] init];
    if (states[0]) {
        if (!shadow_rendering) {
            pass.colorAttachments[0].texture = states[0]->current;
            pass.colorAttachments[0].level = color_level;
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
    }
    if (states[1] && !shadow_rendering) {
        pass.depthAttachment.texture = states[1]->current;
        pass.depthAttachment.level = depth_level;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (states[1]->format == VideoCore::PixelFormat::D24S8) {
            pass.stencilAttachment.texture = states[1]->current;
            pass.stencilAttachment.level = depth_level;
            pass.stencilAttachment.loadAction = MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    }
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    // Shadow fragments write through a storage texture without attachments.
    // Metal requires an explicit raster sample count for this render pass.
    if (shadow_rendering) pass.defaultRasterSampleCount = 1;
    // Replace rather than mutate: queued encoders and copied Framebuffers may
    // still refer to the previously published descriptor and its attachments.
    render_pass_key = key;
    render_pass = pass;
    return render_pass;
}

std::shared_ptr<const ImmutableRenderPass> Framebuffer::RenderSnapshot(Scheduler& scheduler) const {
    auto* source = RenderPass();
    if (!render_snapshot || source != render_snapshot_source) {
        auto snapshot = scheduler.SnapshotRenderPass(source);
        render_snapshot = std::move(snapshot);
        render_snapshot_source = source;
    }
    return render_snapshot;
}

Sampler::Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params_) : params{params_} {
    using Config = VideoCore::SamplerParams::TextureConfig;
    if (params.min_filter > Config::Linear || params.mag_filter > Config::Linear || params.mip_filter > Config::Linear ||
        params.wrap_s > Config::Repeat3 || params.wrap_t > Config::Repeat3 || params.lod_min > params.lod_max)
        throw std::invalid_argument("Invalid native PICA sampler parameters");
    const auto wrap = [](Config::WrapMode mode) {
        switch (mode) {
        case Config::Repeat: case Config::Repeat2: case Config::Repeat3: return MTLSamplerAddressModeRepeat;
        case Config::MirroredRepeat: return MTLSamplerAddressModeMirrorRepeat;
        // GeneratePicaFragmentShader handles border/mixed 2D modes per texel;
        // these sampler address modes only serve the remaining native sample calls.
        default: return MTLSamplerAddressModeClampToEdge;
        }
    };
    auto* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.normalizedCoordinates = YES;
    descriptor.minFilter = params.min_filter == Config::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    descriptor.magFilter = params.mag_filter == Config::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    descriptor.mipFilter = params.mip_filter == Config::Linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    descriptor.sAddressMode = wrap(params.wrap_s);
    descriptor.tAddressMode = wrap(params.wrap_t);
    descriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.lodMinClamp = float(params.lod_min);
    descriptor.lodMaxClamp = float(params.lod_max);
    // LOD bias is supplied by the PICA fragment uniform block, not applied twice here.
    descriptor.maxAnisotropy = 1;
    sampler = runtime.device.CreateSampler(descriptor);
}

} // namespace Metal
