// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_scheduler.h"
#include <vector>

namespace Metal {

struct TextureRegion {
    id<MTLTexture> texture;
    NSUInteger level{};
    NSUInteger slice{};
    MTLRegion region{};
};

// Conservative for shared buffer/heap/IOSurface backing and overlapping views.
// Distinct mip levels or array slices of one ordinary texture are disjoint.
bool TextureSubresourcesMayAlias(const TextureRegion& source, const TextureRegion& dest);

struct TextureBufferLayout {
    NSUInteger offset{};
    NSUInteger bytes_per_row{};
    NSUInteger bytes_per_image{};
    MTLBlitOption options{MTLBlitOptionNone};
};

// Copies, fills and mip generation use the Metal 4 compute encoder's native blit commands.
class TransferRuntime final {
public:
    TransferRuntime(const Device& device, Scheduler& scheduler);
    void CopyBuffer(id<MTLBuffer> source, NSUInteger source_offset, id<MTLBuffer> dest,
                    NSUInteger dest_offset, NSUInteger size);
    void FillBuffer(id<MTLBuffer> buffer, NSRange range, std::uint8_t value);
    void CopyTexture(const TextureRegion& source, const TextureRegion& dest);
    void Upload(id<MTLBuffer> source, const TextureBufferLayout& layout,
                const TextureRegion& dest);
    void Download(const TextureRegion& source, id<MTLBuffer> dest,
                  const TextureBufferLayout& layout);
    std::vector<std::byte> Readback(const TextureRegion& source, NSUInteger bytes_per_row,
                                    MTLBlitOption options = MTLBlitOptionNone);
    void GenerateMipmaps(id<MTLTexture> texture);
    void ClearColor(id<MTLTexture> texture, MTLClearColor color, NSUInteger level = 0,
                    NSUInteger slice = 0);
    void ClearDepthStencil(id<MTLTexture> texture, double depth, std::uint32_t stencil,
                           bool clear_depth, bool clear_stencil, NSUInteger level = 0,
                           NSUInteger slice = 0);

private:
    const Device& device;
    Scheduler& scheduler;
};

} // namespace Metal
