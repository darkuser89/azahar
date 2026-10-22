// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_format_conversion.h"
#include "mtl_guest_codec.h"
#include "mtl_texture_filter.h"
#include <optional>
#include <memory>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"

namespace Metal {

class Surface;
struct SurfaceState;
enum class SurfaceImage { Current, Base, Normal };
enum class SurfaceView { Sample, Mip0, Depth, Stencil, Storage };

class TextureRuntime final {
    friend class Surface;
    friend class Framebuffer;
    friend class Sampler;

public:
    TextureRuntime(Device& device, Scheduler& scheduler);
    VideoCore::StagingData FindStaging(u32 size, bool upload);
    u64 GetResourceTick();
    u64 GetResourceFreeTick();
    void Finish();
    bool NeedsConversion(const Surface& surface) const;
    bool Reinterpret(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);
    bool ClearTexture(Surface& surface, const VideoCore::TextureClear& clear);
    bool CopyTextures(Surface& source, Surface& dest, std::span<const VideoCore::TextureCopy> copies);
    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy) {
        return CopyTextures(source, dest, std::array{copy});
    }
    bool BlitTextures(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);
    void DisplayTransfer(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit);
    void GenerateMipmaps(Surface& surface);
    void UploadGuest(Surface& surface, const VideoCore::SurfaceParams& params, std::span<const u8> bytes,
                     std::optional<Common::Rectangle<u32>> filter_damage={});
    Settings::TextureFilter TextureFilterMode() const;
    void DownloadGuest(Surface& surface, const VideoCore::SurfaceParams& params, std::span<u8> bytes,
                       NSRange byte_range);
    std::vector<u8> DecodeGuest(const VideoCore::SurfaceParams& params, std::span<const u8> bytes,
                                bool converted, bool flip_output = false);
    using TextureDumpSink = std::function<void(const VideoCore::SurfaceParams&, u32, u64, std::vector<u8>)>;
    void SetTextureDumpSink(TextureDumpSink sink) { texture_dump_sink = std::move(sink); }
    void SetTextureFilterProvider(std::function<Settings::TextureFilter()> provider) { texture_filter=std::move(provider); }
    void DumpGuest(const VideoCore::SurfaceParams& params, u32 level, u64 hash, std::span<const u8> bytes);
    void FillGuest(std::span<u8> bytes, std::array<u8, 4> pattern, u32 pattern_size, u32 phase);
    void CopyGuest(const GuestCopyLayout& layout, const BufferSlice& source, const BufferSlice& dest);
    Scheduler& GetScheduler() { return scheduler; }

private:
    id<MTLBuffer> StagingBuffer(const VideoCore::StagingData& staging, bool upload) const;
    id<MTLTexture> Allocate(u32 width, u32 height, u32 levels, VideoCore::TextureType type,
                            MTLPixelFormat format, bool compressed = false);
    void Blit(const TextureRegion& source, const TextureRegion& dest, bool linear,
              bool flip_x = false, bool flip_y = false);
    Device& device;
    Scheduler& scheduler;
    TransferRuntime transfers;
    FormatConversionRuntime conversions;
    GuestTextureCodec guest_codec;
    TextureFilterRuntime filters;
    std::function<Settings::TextureFilter()> texture_filter;
    TextureDumpSink texture_dump_sink;
    SharedUploadPool staging_pool;
    // The public mapped staging range remains leased until the next request in
    // the same direction. GPU completion independently protects encoded users.
    BufferSlice upload_staging{};
    BufferSlice download_staging{};
    u64 free_tick{};
    struct ReadbackProfile;
    std::shared_ptr<ReadbackProfile> readback_profile;

};

// Shared state lets cached framebuffers follow a surface's replacement allocation even
// when the generic slot container moves the Surface object.
class Surface final : public VideoCore::SurfaceBase {
    friend class TextureRuntime;
    friend class Framebuffer;

public:
    Surface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params,
            const VideoCore::SurfaceFlagBits& initial_flags = {});
    Surface(TextureRuntime& runtime, const VideoCore::SurfaceBase& surface,
            const VideoCore::Material* material);
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    Surface(Surface&&) noexcept = default;
    Surface& operator=(Surface&& other);
    ~Surface();

    id<MTLTexture> Image(SurfaceImage image = SurfaceImage::Current) const;
    id<MTLTexture> ImageView(SurfaceView view = SurfaceView::Sample,
                            SurfaceImage image = SurfaceImage::Current, u32 level = 0, u32 slice = 0) const;
    id<MTLTexture> CopyImageView();
    id<MTLTexture> FramebufferView() const { return ImageView(SurfaceView::Mip0); }
    id<MTLTexture> DepthView() const { return ImageView(SurfaceView::Depth); }
    id<MTLTexture> StencilView() const { return ImageView(SurfaceView::Stencil); }
    id<MTLTexture> StorageView(u32 level = 0, u32 slice = 0) const {
        return ImageView(SurfaceView::Storage, SurfaceImage::Current, level, slice);
    }
    u32 GetWidth() const { return width; }
    u32 GetHeight() const { return height; }
    u32 GetResScale() const { return res_scale; }
    u32 GetInternalBytesPerPixel() const;
    void Upload(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging,
                std::optional<Common::Rectangle<u32>> filter_damage={});
    void Download(const VideoCore::BufferTextureCopy& copy, const VideoCore::StagingData& staging);
    void UploadCustom(const VideoCore::Material* material, u32 level);
    void ScaleUp(u32 scale);

private:
    friend class TextureRuntime;
    // Keep the public staging readback synchronous. Guest encoding can consume
    // the staging buffer directly on the GPU and wait once after both operations.
    void EncodeDownload(const VideoCore::BufferTextureCopy& copy,
                        const VideoCore::StagingData& staging);
    TextureRegion PrepareDownload(Common::Rectangle<u32> rect, u32 level);
    TextureRuntime& runtime;
    std::shared_ptr<SurfaceState> state;
    DepthStagingFormat depth_staging{DepthStagingFormat::D24S8};
};

class Framebuffer final : public VideoCore::FramebufferParams {
public:
    Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                Surface* color, Surface* depth);
    // Borrowed immutable descriptor. Copy it before changing render-pass actions.
    MTL4RenderPassDescriptor* RenderPass() const;
    std::shared_ptr<const ImmutableRenderPass> RenderSnapshot(Scheduler& scheduler) const;
    VideoCore::PixelFormat Format(VideoCore::SurfaceType type) const;
    id<MTLTexture> ImageView(VideoCore::SurfaceType type) const;
    u32 Scale() const;

private:
    std::array<std::shared_ptr<SurfaceState>, 2> states;
    struct RenderPassKey {
        id<MTLTexture> color{}, depth{};
        u32 color_level{}, depth_level{};
        bool shadow{}, stencil{};
        bool operator==(const RenderPassKey&) const = default;
    };
    mutable RenderPassKey render_pass_key;
    mutable MTL4RenderPassDescriptor* render_pass{};
    mutable MTL4RenderPassDescriptor* render_snapshot_source{};
    mutable std::shared_ptr<const ImmutableRenderPass> render_snapshot;
};

class Sampler final {
public:
    Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params);
    id<MTLSamplerState> Handle() const { return sampler; }
    // Custom border colors and asymmetric PICA wrap modes require shader handling.
    const VideoCore::SamplerParams& Params() const { return params; }

private:
    VideoCore::SamplerParams params;
    id<MTLSamplerState> sampler;
};

class DebugScope final {
public:
    template <class... Args>
    DebugScope(TextureRuntime&, Common::Vec4f, fmt::format_string<Args...>, Args&&...) {}
};

struct Traits {
    using Runtime = Metal::TextureRuntime;
    using Surface = Metal::Surface;
    using Sampler = Metal::Sampler;
    using Framebuffer = Metal::Framebuffer;
    using DebugScope = Metal::DebugScope;
};
using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Metal

// Declare Metal-only member specializations before a consumer instantiates the cache.
// Other backends keep the unchanged common template's codec paths.
namespace VideoCore {
template <> void RasterizerCache<Metal::Traits>::ValidateSurface(SurfaceId, PAddr, u32);
template <> void RasterizerCache<Metal::Traits>::UploadSurface(Metal::Surface&, SurfaceInterval);
template <> void RasterizerCache<Metal::Traits>::DownloadSurface(Metal::Surface&, SurfaceInterval);
template <> u64 RasterizerCache<Metal::Traits>::ComputeHash(const SurfaceParams&, std::span<u8>);
template <> void RasterizerCache<Metal::Traits>::DownloadFillSurface(Metal::Surface&, SurfaceInterval);
template <> bool RasterizerCache<Metal::Traits>::ValidateByReinterpretation(Metal::Surface&, SurfaceParams,
                                                                            const SurfaceInterval&);
} // namespace VideoCore
