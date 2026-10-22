// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_stream_buffer.h"
#include "mtl_transfer.h"
#include <map>

namespace Metal {

enum class DepthStagingFormat : std::uint32_t { D24S8, D16, D32Float };

// PICA staging words contain stencil in bits 0..7 and depth in bits 8..31.
// Color reinterpretation yields host RGBA = stencil, depth high, depth middle, depth low.
class FormatConversionRuntime final {
public:
    FormatConversionRuntime(Device& device, Scheduler& scheduler);
    void DepthStencilToBuffer(const TextureRegion& source, id<MTLBuffer> dest,
                              NSUInteger offset, NSUInteger bytes_per_row,
                              DepthStagingFormat format = DepthStagingFormat::D24S8);
    void BufferToDepthStencil(id<MTLBuffer> source, NSUInteger offset,
                              NSUInteger bytes_per_row, const TextureRegion& dest,
                              DepthStagingFormat format = DepthStagingFormat::D24S8);
    void BlitDepthStencil(const TextureRegion& source, const TextureRegion& dest,
                           bool flip_x = false, bool flip_y = false);
    void BlitColor(const TextureRegion& source, const TextureRegion& dest, bool linear,
                    bool flip_x = false, bool flip_y = false);
    void PicaColorTransfer(const TextureRegion& source, const TextureRegion& dest,
                           std::uint32_t guest_format, bool flip_x = false, bool flip_y = false);
    void DepthStencilToRGBA8(const TextureRegion& source, const TextureRegion& dest);
    void ReinterpretColor16(const TextureRegion& source, const TextureRegion& dest,
                            std::uint32_t source_format, std::uint32_t dest_format);
    void ClearRegion(const TextureRegion& dest, MTLClearColor color, double depth,
                     std::uint32_t stencil);

private:
    struct DepthViews {
        id<MTLTexture> depth;
        id<MTLTexture> stencil;
        bool has_stencil;
    };
    DepthViews MakeDepthViews(const TextureRegion& source);
    Device& device;
    Scheduler& scheduler;
    StreamBuffer parameters;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> buffer_pipeline;
    id<MTLComputePipelineState> color_pipeline;
    id<MTLComputePipelineState> unpack_pipeline;
    id<MTLComputePipelineState> depth_blit_pipeline;
    id<MTLComputePipelineState> color_blit_pipeline;
    id<MTLComputePipelineState> pica_transfer_pipeline;
    id<MTLComputePipelineState> color16_pipeline;
    id<MTLSamplerState> nearest_sampler;
    id<MTLSamplerState> linear_sampler;
    id<MTLTexture> empty_stencil;
    id<MTLBuffer> depth_plane, stencil_plane;
    id<MTLTexture> color_scratch;
    id<MTLBuffer> PlaneStorage(NSUInteger bytes, bool stencil);
    std::map<MTLPixelFormat, std::shared_future<id<MTLRenderPipelineState>>> clear_pipelines;
    id<MTLComputePipelineState> clear_depth_pipeline;
    ReusableArgumentTable clear_arguments;
};

} // namespace Metal
