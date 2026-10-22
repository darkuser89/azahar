// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_stream_buffer.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace Metal {

struct GuestTextureLayout {
    u32 width;
    u32 height;
    u32 stride;
    VideoCore::PixelFormat format;
    bool tiled{};
    bool converted{true};
    bool flip_output{};
};

struct GuestCopyLayout {
    std::uint64_t byte_count{};
    std::uint32_t input_width{};
    std::uint32_t input_gap{};
    std::uint32_t output_width{};
    std::uint32_t output_gap{};
};

// Guest memory is copied as bytes; pixel conversion and Morton/ETC decoding run on the GPU.
// ETC is a sampled guest texture format, not a PICA render-target format; encoding is rejected.
class GuestTextureCodec final {
public:
    GuestTextureCodec(Device& device, Scheduler& scheduler);
    static NSUInteger GuestBytes(const GuestTextureLayout& layout);
    static NSUInteger LinearBytes(const GuestTextureLayout& layout);
    static u32 LinearBytesPerPixel(const GuestTextureLayout& layout);
    void Decode(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest);
    void Encode(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest,
                NSRange byte_range);
    // Read native depth/stencil planes directly into guest bytes. Views select
    // one mip/slice; origin selects its bounded rectangle. No linear staging pass.
    void EncodeDepth(const GuestTextureLayout& layout, id<MTLTexture> depth,
                     id<MTLTexture> stencil, MTLOrigin origin, const BufferSlice& dest,
                     NSRange byte_range);
    void Fill(const BufferSlice& dest, std::array<u8, 4> pattern, u32 pattern_size, u32 phase);
    static std::uint64_t CopyBytes(std::uint64_t count, u32 width, u32 gap);
    void Copy(const GuestCopyLayout& layout, const BufferSlice& source, const BufferSlice& dest);

private:
    void Run(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest,
             NSRange byte_range, bool decode);
    Device& device;
    Scheduler& scheduler;
    StreamBuffer uniforms;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> decode_pipeline;
    id<MTLComputePipelineState> encode_pipeline;
    id<MTLComputePipelineState> depth_encode_pipeline;
    id<MTLComputePipelineState> fill_pipeline;
    id<MTLComputePipelineState> copy_pipeline;
    id<MTLTexture> empty_stencil;
};

} // namespace Metal
