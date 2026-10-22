// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_transfer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace Metal {

namespace {

struct BlockLayout {
    NSUInteger width;
    NSUInteger height;
    NSUInteger bytes;
};

bool IsDepthStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth32Float_Stencil8;
}

BlockLayout GetBlockLayout(MTLPixelFormat format, MTLBlitOption options = MTLBlitOptionNone) {
    if (IsDepthStencil(format)) {
        if (options == MTLBlitOptionDepthFromDepthStencil) {
            return {1, 1, 4};
        }
        if (options == MTLBlitOptionStencilFromDepthStencil) {
            return {1, 1, 1};
        }
        throw std::invalid_argument("A depth/stencil transfer must select exactly one plane");
    }
    if (options != MTLBlitOptionNone) {
        throw std::invalid_argument("Unsupported options for the Metal texture format");
    }
    switch (format) {
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatStencil8:
    case MTLPixelFormatX32_Stencil8:
        return {1, 1, 1};
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatDepth16Unorm:
        return {1, 1, 2};
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatDepth32Float:
        return {1, 1, 4};
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRG32Float:
        return {1, 1, 8};
    case MTLPixelFormatRGBA32Float:
        return {1, 1, 16};
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB:
        return {4, 4, 8};
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC7_RGBAUnorm:
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
    case MTLPixelFormatASTC_4x4_LDR:
    case MTLPixelFormatASTC_4x4_sRGB:
        return {4, 4, 16};
    case MTLPixelFormatASTC_6x6_LDR:
    case MTLPixelFormatASTC_6x6_sRGB:
        return {6, 6, 16};
    case MTLPixelFormatASTC_8x6_LDR:
    case MTLPixelFormatASTC_8x6_sRGB:
        return {8, 6, 16};
    case MTLPixelFormatASTC_8x8_LDR:
    case MTLPixelFormatASTC_8x8_sRGB:
        return {8, 8, 16};
    default:
        throw std::invalid_argument("Metal texture format has no transfer layout");
    }
}

NSUInteger Multiply(NSUInteger a, NSUInteger b) {
    if (b && a > std::numeric_limits<NSUInteger>::max() / b) {
        throw std::overflow_error("Metal transfer size overflow");
    }
    return a * b;
}

NSUInteger Add(NSUInteger a, NSUInteger b) {
    if (a > std::numeric_limits<NSUInteger>::max() - b) {
        throw std::overflow_error("Metal transfer offset overflow");
    }
    return a + b;
}

void ValidateBuffer(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger size) {
    if (!buffer || offset > buffer.length || size > buffer.length - offset) {
        throw std::invalid_argument("Metal buffer transfer is outside its allocation");
    }
}

void ValidateRegion(const TextureRegion& region) {
    auto texture = region.texture;
    if (!texture || texture.sampleCount != 1 || texture.framebufferOnly ||
        texture.storageMode == MTLStorageModeMemoryless || region.level >= texture.mipmapLevelCount) {
        throw std::invalid_argument("Invalid Metal texture transfer resource or mip level");
    }
    NSUInteger slices = texture.arrayLength;
    if (texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray) {
        slices = Multiply(slices, 6);
    }
    if (region.slice >= slices) {
        throw std::invalid_argument("Metal texture transfer uses an invalid array slice");
    }
    const auto origin = region.region.origin;
    const auto size = region.region.size;
    const auto width = std::max<NSUInteger>(1, texture.width >> region.level);
    const auto height = std::max<NSUInteger>(1, texture.height >> region.level);
    const auto depth = std::max<NSUInteger>(1, texture.depth >> region.level);
    if (!size.width || !size.height || !size.depth || origin.x > width || origin.y > height ||
        origin.z > depth || size.width > width - origin.x || size.height > height - origin.y ||
        size.depth > depth - origin.z) {
        throw std::invalid_argument("Metal texture transfer is outside its mip dimensions");
    }
    if (!IsDepthStencil(texture.pixelFormat)) {
        const auto block = GetBlockLayout(texture.pixelFormat);
        if (origin.x % block.width || origin.y % block.height ||
            (size.width % block.width && size.width != width - origin.x) ||
            (size.height % block.height && size.height != height - origin.y)) {
            throw std::invalid_argument("Metal compressed transfer must align to blocks or mip edges");
        }
    }
}

NSUInteger RequiredBytes(const TextureRegion& region, const TextureBufferLayout& layout) {
    ValidateRegion(region);
    const auto block = GetBlockLayout(region.texture.pixelFormat, layout.options);
    const auto size = region.region.size;
    const auto columns = (size.width + block.width - 1) / block.width;
    const auto rows = (size.height + block.height - 1) / block.height;
    const auto row_size = Multiply(columns, block.bytes);
    if (layout.offset % block.bytes || layout.bytes_per_row % block.bytes ||
        layout.bytes_per_row < row_size ||
        (size.depth == 1 && layout.bytes_per_image != 0) ||
        (size.depth > 1 && (layout.bytes_per_image % block.bytes ||
                           layout.bytes_per_image < Multiply(rows, layout.bytes_per_row)))) {
        throw std::invalid_argument("Invalid Metal buffer/texture transfer pitch or alignment");
    }
    return Add(Multiply(size.depth - 1, layout.bytes_per_image),
               Add(Multiply(rows - 1, layout.bytes_per_row), row_size));
}

} // namespace

bool TextureSubresourcesMayAlias(const TextureRegion& source, const TextureRegion& dest) {
    const auto root = [](const TextureRegion& region) {
        auto texture = region.texture;
        auto level = region.level, slice = region.slice;
        while (texture.parentTexture) {
            level += texture.parentRelativeLevel;
            slice += texture.parentRelativeSlice;
            texture = texture.parentTexture;
        }
        return std::tuple{texture, level, slice};
    };
    const auto [a, al, as] = root(source);
    const auto [b, bl, bs] = root(dest);
    if (a == b) return al == bl && as == bs;
    return (a.buffer && a.buffer == b.buffer) || (a.heap && a.heap == b.heap) ||
           (a.iosurface && a.iosurface == b.iosurface);
}

TransferRuntime::TransferRuntime(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

void TransferRuntime::CopyBuffer(id<MTLBuffer> source, NSUInteger source_offset, id<MTLBuffer> dest,
                                 NSUInteger dest_offset, NSUInteger size) {
    ValidateBuffer(source, source_offset, size);
    ValidateBuffer(dest, dest_offset, size);
    if (!size || (source == dest && source_offset == dest_offset)) {
        return;
    }
    if (source == dest && source_offset < dest_offset + size && dest_offset < source_offset + size) {
        auto temporary = device.CreateBuffer(size, false);
        CopyBuffer(source, source_offset, temporary, 0, size);
        CopyBuffer(temporary, 0, dest, dest_offset, size);
        return;
    }
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::CopyBuffer};
    command.source_buffer=source; command.destination_buffer=dest;
    command.source_offset=source_offset; command.destination_offset=dest_offset; command.size=size;
    scheduler.QueueTransfer(command);
}

void TransferRuntime::FillBuffer(id<MTLBuffer> buffer, NSRange range, std::uint8_t value) {
    ValidateBuffer(buffer, range.location, range.length);
    if (!range.length) {
        return;
    }
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::FillBuffer};
    command.destination_buffer=buffer; command.destination_offset=range.location;
    command.size=range.length; command.value=value;
    scheduler.QueueTransfer(command);
}

void TransferRuntime::CopyTexture(const TextureRegion& source, const TextureRegion& dest) {
    ValidateRegion(source);
    ValidateRegion(dest);
    if (source.texture.pixelFormat != dest.texture.pixelFormat ||
        source.region.size.width != dest.region.size.width ||
        source.region.size.height != dest.region.size.height ||
        source.region.size.depth != dest.region.size.depth) {
        throw std::invalid_argument("A Metal texture copy must preserve format and region size");
    }
    if (TextureSubresourcesMayAlias(source, dest)) {
        // Texture views can overlap despite different resource IDs. Copy through separate storage.
        MTLTextureDescriptor* descriptor = [[MTLTextureDescriptor alloc] init];
        descriptor.textureType = source.region.size.depth > 1 ? MTLTextureType3D : MTLTextureType2D;
        descriptor.pixelFormat = source.texture.pixelFormat;
        descriptor.width = source.region.size.width;
        descriptor.height = source.region.size.height;
        descriptor.depth = source.region.size.depth;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead;
        const TextureRegion temporary{device.CreateTexture(descriptor), 0, 0,
                                      MTLRegion{{0, 0, 0}, source.region.size}};
        CopyTexture(source, temporary);
        CopyTexture(temporary, dest);
        return;
    }
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::CopyTexture};
    command.source_texture=source.texture; command.destination_texture=dest.texture;
    command.source_slice=source.slice; command.source_level=source.level; command.source_origin=source.region.origin;
    command.destination_slice=dest.slice; command.destination_level=dest.level; command.destination_origin=dest.region.origin;
    command.extent=source.region.size;
    scheduler.QueueTransfer(command);
}

void TransferRuntime::Upload(id<MTLBuffer> source, const TextureBufferLayout& layout,
                             const TextureRegion& dest) {
    ValidateBuffer(source, layout.offset, RequiredBytes(dest, layout));
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::Upload};
    command.source_buffer=source; command.source_offset=layout.offset; command.destination_texture=dest.texture;
    command.destination_slice=dest.slice; command.destination_level=dest.level; command.destination_origin=dest.region.origin;
    command.extent=dest.region.size; command.bytes_per_row=layout.bytes_per_row;
    command.bytes_per_image=layout.bytes_per_image; command.options=layout.options;
    scheduler.QueueTransfer(command);
}

void TransferRuntime::Download(const TextureRegion& source, id<MTLBuffer> dest,
                               const TextureBufferLayout& layout) {
    ValidateBuffer(dest, layout.offset, RequiredBytes(source, layout));
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::Download};
    command.source_texture=source.texture; command.destination_buffer=dest; command.destination_offset=layout.offset;
    command.source_slice=source.slice; command.source_level=source.level; command.source_origin=source.region.origin;
    command.extent=source.region.size; command.bytes_per_row=layout.bytes_per_row;
    command.bytes_per_image=layout.bytes_per_image; command.options=layout.options;
    scheduler.QueueTransfer(command);
}

std::vector<std::byte> TransferRuntime::Readback(const TextureRegion& source, NSUInteger bytes_per_row,
                                                MTLBlitOption options) {
    const TextureBufferLayout layout{0, bytes_per_row, 0, options};
    const auto size = RequiredBytes(source, layout);
    auto buffer = device.CreateBuffer(size, true);
    // GPU copies leave row padding untouched. Do not expose uninitialized padding to the CPU.
    FillBuffer(buffer, NSMakeRange(0, size), 0);
    Download(source, buffer, layout);
    scheduler.Finish(Scheduler::WaitReason::Readback);
    std::vector<std::byte> result(size);
    std::memcpy(result.data(), buffer.contents, size);
    return result;
}

void TransferRuntime::GenerateMipmaps(id<MTLTexture> texture) {
    if (!texture || texture.sampleCount != 1 || texture.framebufferOnly ||
        texture.pixelFormat == MTLPixelFormatDepth16Unorm ||
        texture.pixelFormat == MTLPixelFormatDepth32Float || IsDepthStencil(texture.pixelFormat)) {
        throw std::invalid_argument("Invalid Metal mipmap generation resource");
    }
    if (texture.mipmapLevelCount <= 1) {
        return;
    }
    const auto block = GetBlockLayout(texture.pixelFormat);
    if (block.width != 1 || texture.pixelFormat == MTLPixelFormatR32Uint ||
        texture.pixelFormat == MTLPixelFormatR16Uint || texture.pixelFormat == MTLPixelFormatR8Uint ||
        texture.pixelFormat == MTLPixelFormatStencil8 || texture.pixelFormat == MTLPixelFormatX32_Stencil8) {
        throw std::invalid_argument("Metal mipmap generation requires a filterable color format");
    }
    Scheduler::TransferCommand command{Scheduler::TransferCommand::Kind::GenerateMipmaps};
    command.destination_texture=texture;
    scheduler.QueueTransfer(command);
}

void TransferRuntime::ClearColor(id<MTLTexture> texture, MTLClearColor color, NSUInteger level,
                                 NSUInteger slice) {
    if (!texture || !(texture.usage & MTLTextureUsageRenderTarget) ||
        level >= texture.mipmapLevelCount || texture.textureType == MTLTextureType3D ||
        texture.pixelFormat == MTLPixelFormatDepth16Unorm ||
        texture.pixelFormat == MTLPixelFormatDepth32Float || IsDepthStencil(texture.pixelFormat) ||
        texture.pixelFormat == MTLPixelFormatStencil8 || texture.pixelFormat == MTLPixelFormatX32_Stencil8) {
        throw std::invalid_argument("Invalid Metal color clear target");
    }
    // Validate array bounds through the same subresource path used by transfers.
    ValidateRegion({texture, level, slice, MTLRegionMake2D(0, 0,
        std::max<NSUInteger>(1, texture.width >> level), std::max<NSUInteger>(1, texture.height >> level))});
    MTL4RenderPassDescriptor* descriptor = [[MTL4RenderPassDescriptor alloc] init];
    auto* attachment = descriptor.colorAttachments[0];
    attachment.texture = texture;
    attachment.level = level;
    attachment.slice = slice;
    attachment.loadAction = MTLLoadActionClear;
    attachment.storeAction = MTLStoreActionStore;
    attachment.clearColor = color;
    scheduler.QueueClear(descriptor);
}

void TransferRuntime::ClearDepthStencil(id<MTLTexture> texture, double depth, std::uint32_t stencil,
                                        bool clear_depth, bool clear_stencil, NSUInteger level,
                                        NSUInteger slice) {
    if (!texture || !(texture.usage & MTLTextureUsageRenderTarget) ||
        texture.textureType == MTLTextureType3D || level >= texture.mipmapLevelCount ||
        !std::isfinite(depth) || depth < 0.0 || depth > 1.0 ||
        stencil > 255 ||
        (clear_stencil && !IsDepthStencil(texture.pixelFormat)) ||
        (texture.pixelFormat != MTLPixelFormatDepth16Unorm &&
         texture.pixelFormat != MTLPixelFormatDepth32Float && !IsDepthStencil(texture.pixelFormat))) {
        throw std::invalid_argument("Invalid Metal depth/stencil clear");
    }
    if (!clear_depth && !clear_stencil) {
        return;
    }
    ValidateRegion({texture, level, slice, MTLRegionMake2D(0, 0,
        std::max<NSUInteger>(1, texture.width >> level), std::max<NSUInteger>(1, texture.height >> level))});
    MTL4RenderPassDescriptor* descriptor = [[MTL4RenderPassDescriptor alloc] init];
    descriptor.depthAttachment.texture = texture;
    descriptor.depthAttachment.level = level;
    descriptor.depthAttachment.slice = slice;
    descriptor.depthAttachment.loadAction = clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
    descriptor.depthAttachment.storeAction = MTLStoreActionStore;
    descriptor.depthAttachment.clearDepth = depth;
    if (IsDepthStencil(texture.pixelFormat)) {
        descriptor.stencilAttachment.texture = texture;
        descriptor.stencilAttachment.level = level;
        descriptor.stencilAttachment.slice = slice;
        descriptor.stencilAttachment.loadAction = clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
        descriptor.stencilAttachment.storeAction = MTLStoreActionStore;
        descriptor.stencilAttachment.clearStencil = stencil;
    }
    scheduler.QueueClear(descriptor);
}

} // namespace Metal
