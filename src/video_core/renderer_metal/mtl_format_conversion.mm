// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_format_conversion.h"
#include "mtl_shaders.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace Metal {

namespace {

struct ConversionParams {
    std::array<std::uint32_t, 2> source_origin;
    std::array<std::uint32_t, 2> dest_origin;
    std::array<std::uint32_t, 2> extent;
    std::uint32_t row_bytes;
    std::uint32_t format{};
};
static_assert(sizeof(ConversionParams) == 32);

struct DepthBlitParams {
    std::array<std::uint32_t, 2> source_origin;
    std::array<std::uint32_t, 2> source_extent;
    std::array<std::uint32_t, 2> dest_extent;
    std::uint32_t flips;
    std::uint32_t padding{};
    std::array<std::uint32_t, 2> dest_origin{};
};
static_assert(sizeof(DepthBlitParams) == 40);

void Validate2D(const TextureRegion& region, id<MTLDevice> device) {
    const auto texture = region.texture;
    if (!texture || texture.device != device || texture.sampleCount != 1 || texture.framebufferOnly ||
        texture.storageMode == MTLStorageModeMemoryless ||
        (texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray &&
         texture.textureType != MTLTextureTypeCube && texture.textureType != MTLTextureTypeCubeArray) ||
        region.level >= texture.mipmapLevelCount ||
        region.slice >= texture.arrayLength * ((texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray) ? 6 : 1) ||
        region.region.origin.z != 0 || region.region.size.depth != 1) {
        throw std::invalid_argument("Native format conversion requires a valid single-sample 2D subresource");
    }
    const auto width = std::max<NSUInteger>(1, texture.width >> region.level);
    const auto height = std::max<NSUInteger>(1, texture.height >> region.level);
    const auto origin = region.region.origin;
    const auto size = region.region.size;
    if (!size.width || !size.height || origin.x > width || origin.y > height ||
        size.width > width - origin.x || size.height > height - origin.y ||
        width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Native format conversion region exceeds its subresource");
    }
}

id<MTLTexture> View(id<MTLTexture> texture, MTLPixelFormat format, NSUInteger level, NSUInteger slice) {
    MTLTextureViewDescriptor* descriptor = [[MTLTextureViewDescriptor alloc] init];
    descriptor.pixelFormat = format;
    descriptor.textureType = MTLTextureType2D;
    descriptor.levelRange = NSMakeRange(level, 1);
    descriptor.sliceRange = NSMakeRange(slice, 1);
    auto view = [texture newTextureViewWithDescriptor:descriptor];
    if (!view) throw std::runtime_error("Unable to create native format conversion subresource view");
    return view;
}

struct ColorOutput {
    id<MTLTexture> texture;
    bool direct;
};
ColorOutput PrepareColorOutput(Device& device, __strong id<MTLTexture>& scratch,
                                const TextureRegion& source, const TextureRegion& dest) {
    if ((dest.texture.usage & MTLTextureUsageShaderWrite) && !TextureSubresourcesMayAlias(source, dest))
        return {View(dest.texture, dest.texture.pixelFormat, dest.level, dest.slice), true};
    if (scratch && scratch.pixelFormat == dest.texture.pixelFormat &&
        scratch.width >= dest.region.size.width && scratch.height >= dest.region.size.height)
        return {scratch, false};
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:dest.texture.pixelFormat
        width:dest.region.size.width height:dest.region.size.height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    auto texture = device.CreateTexture(descriptor);
    // Every consumer is encoded before the next call. Queue barriers order
    // reuse; submission payloads keep replaced textures alive until completion.
    if (texture.allocatedSize <= 16U * 1024 * 1024) scratch = texture;
    return {texture, false};
}

ConversionParams Params(const TextureRegion& source, const TextureRegion* dest, std::uint32_t words) {
    return {{std::uint32_t(source.region.origin.x), std::uint32_t(source.region.origin.y)},
            {dest ? std::uint32_t(dest->region.origin.x) : 0U,
             dest ? std::uint32_t(dest->region.origin.y) : 0U},
            {std::uint32_t(source.region.size.width), std::uint32_t(source.region.size.height)}, words};
}

bool WritableColor(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatRGBA32Float:
        return true;
    default:
        return false;
    }
}

NSUInteger DepthBytes(DepthStagingFormat format) {
    switch (format) {
    case DepthStagingFormat::D16: return 2;
    case DepthStagingFormat::D24S8:
    case DepthStagingFormat::D32Float: return 4;
    }
    throw std::invalid_argument("Unknown native depth staging format");
}

bool ReadableColor(MTLPixelFormat format) {
    if (WritableColor(format)) return true;
    switch (format) {
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatBC1_RGBA:
    case MTLPixelFormatBC1_RGBA_sRGB:
    case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB:
    case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC7_RGBAUnorm:
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
    case MTLPixelFormatASTC_4x4_LDR:
    case MTLPixelFormatASTC_4x4_sRGB:
    case MTLPixelFormatASTC_6x6_LDR:
    case MTLPixelFormatASTC_6x6_sRGB:
    case MTLPixelFormatASTC_8x6_LDR:
    case MTLPixelFormatASTC_8x6_sRGB:
    case MTLPixelFormatASTC_8x8_LDR:
    case MTLPixelFormatASTC_8x8_sRGB:
        return true;
    default:
        return false;
    }
}

} // namespace

FormatConversionRuntime::FormatConversionRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, parameters{device, scheduler, 64 << 10} {
    library = device.CompileLibrary(FormatConversionShaderSource, "Azahar Metal 4 PICA format conversion");
    const auto compile = [&](std::string_view entry) {
        MTL4ComputePipelineDescriptor* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor = Device::Function(library, entry);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        return device.CompileCompute(descriptor);
    };
    buffer_pipeline = compile("depth_stencil_to_buffer");
    color_pipeline = compile("depth_stencil_to_rgba8");
    unpack_pipeline = compile("buffer_to_depth_stencil");
    depth_blit_pipeline = compile("blit_depth_stencil");
    color_blit_pipeline = compile("blit_color");
    pica_transfer_pipeline = compile("pica_color_transfer");
    color16_pipeline = compile("reinterpret_color16");
    clear_depth_pipeline = compile("clear_depth_planes");
    auto* sampler = [[MTLSamplerDescriptor alloc] init];
    sampler.normalizedCoordinates = NO;
    sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler.minFilter = MTLSamplerMinMagFilterNearest;
    sampler.magFilter = MTLSamplerMinMagFilterNearest;
    nearest_sampler = device.CreateSampler(sampler);
    sampler.minFilter = MTLSamplerMinMagFilterLinear;
    sampler.magFilter = MTLSamplerMinMagFilterLinear;
    linear_sampler = device.CreateSampler(sampler);
    auto* empty_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint width:1 height:1 mipmapped:NO];
    empty_descriptor.storageMode = MTLStorageModePrivate;
    empty_descriptor.usage = MTLTextureUsageShaderRead;
    empty_stencil = device.CreateTexture(empty_descriptor);
    auto zero = device.CreateBuffer(1, true);
    *static_cast<std::uint8_t*>(zero.contents) = 0;
    TransferRuntime{device, scheduler}.Upload(zero, {0, 1, 0, MTLBlitOptionNone}, {empty_stencil, 0, 0, MTLRegionMake2D(0, 0, 1, 1)});
}

FormatConversionRuntime::DepthViews FormatConversionRuntime::MakeDepthViews(const TextureRegion& source) {
    Validate2D(source, device.Handle());
    const auto format = source.texture.pixelFormat;
    if ((format != MTLPixelFormatDepth32Float_Stencil8 && format != MTLPixelFormatDepth16Unorm && format != MTLPixelFormatDepth32Float) ||
        !(source.texture.usage & MTLTextureUsageShaderRead) ||
        !(source.texture.usage & MTLTextureUsagePixelFormatView)) {
        throw std::invalid_argument("Native depth/stencil conversion requires a readable D32S8 texture with plane views");
    }
    // A D32S8 view itself supports depth reads; the stencil plane uses X32_Stencil8.
    const bool has_stencil = format == MTLPixelFormatDepth32Float_Stencil8;
    return {View(source.texture, format, source.level, source.slice),
            has_stencil ? View(source.texture, MTLPixelFormatX32_Stencil8, source.level, source.slice) : empty_stencil,
            has_stencil};
}

id<MTLBuffer> FormatConversionRuntime::PlaneStorage(NSUInteger bytes, bool stencil) {
    auto& cached = stencil ? stencil_plane : depth_plane;
    if (bytes > 16U * 1024 * 1024) return device.CreateBuffer(bytes, false);
    if (!cached || cached.length < bytes) cached = device.CreateBuffer(bytes, false);
    // These buffers never escape the conversion call. All plane consumers
    // are encoded before reuse, with native RAW/WAR/WAW queue dependencies.
    return cached;
}

void FormatConversionRuntime::DepthStencilToBuffer(const TextureRegion& source, id<MTLBuffer> dest,
                                                   NSUInteger offset, NSUInteger bytes_per_row,
                                                   DepthStagingFormat format) {
    Validate2D(source, device.Handle());
    if (!dest || dest.device != device.Handle()) throw std::invalid_argument("Invalid native depth staging buffer");
    if ((source.texture.pixelFormat == MTLPixelFormatDepth16Unorm && format == DepthStagingFormat::D16) ||
        (source.texture.pixelFormat == MTLPixelFormatDepth32Float && format == DepthStagingFormat::D32Float)) {
        TransferRuntime{device, scheduler}.Download(source, dest, {offset, bytes_per_row, 0, MTLBlitOptionNone});
        return;
    }
    const auto views = MakeDepthViews(source);
    const auto size = source.region.size;
    const auto bytes = DepthBytes(format);
    if (!dest || dest.device != device.Handle() || offset % bytes || bytes_per_row % bytes ||
        bytes_per_row < size.width * bytes || bytes_per_row > std::numeric_limits<std::uint32_t>::max() ||
        offset > dest.length ||
        size.width * bytes > dest.length - offset ||
        size.height - 1 > (dest.length - offset - size.width * bytes) / bytes_per_row) {
        throw std::invalid_argument("Native packed depth/stencil output buffer or row pitch is invalid");
    }
    auto info = Params(source, nullptr, std::uint32_t(bytes_per_row));
    info.format = std::uint32_t(format) | (std::uint32_t(views.has_stencil) << 8);
    auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 2> buffers{{{dest, offset, 0}, {uniform.buffer, uniform.offset, 1}}};
    const std::array<TextureBinding, 2> textures{{{views.depth, 0}, {views.stencil, 1}}};
    auto table = scheduler.MakeArguments(buffers, textures);
    scheduler.KeepAlive(buffer_pipeline);
    auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:buffer_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
}

void FormatConversionRuntime::DepthStencilToRGBA8(const TextureRegion& source, const TextureRegion& dest) {
    const auto views = MakeDepthViews(source);
    Validate2D(dest, device.Handle());
    const auto size = source.region.size;
    if (dest.texture.pixelFormat != MTLPixelFormatRGBA8Unorm ||
        !(dest.texture.usage & MTLTextureUsageShaderWrite) ||
        size.width != dest.region.size.width || size.height != dest.region.size.height) {
        throw std::invalid_argument("Native depth/stencil reinterpretation requires an equally sized writable RGBA8 region");
    }
    const auto output = View(dest.texture, dest.texture.pixelFormat, dest.level, dest.slice);
    auto info = Params(source, &dest, 0);
    info.format = std::uint32_t(views.has_stencil) << 8;
    auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 1> buffers{{{uniform.buffer, uniform.offset, 1}}};
    const std::array<TextureBinding, 3> textures{{{views.depth, 0}, {views.stencil, 1}, {output, 2}}};
    auto table = scheduler.MakeArguments(buffers, textures);
    scheduler.KeepAlive(color_pipeline);
    auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:color_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
}

void FormatConversionRuntime::BufferToDepthStencil(id<MTLBuffer> source, NSUInteger offset,
                                                   NSUInteger bytes_per_row, const TextureRegion& dest,
                                                   DepthStagingFormat format) {
    Validate2D(dest, device.Handle());
    if (!source || source.device != device.Handle()) throw std::invalid_argument("Invalid native depth upload buffer");
    if ((dest.texture.pixelFormat == MTLPixelFormatDepth16Unorm && format == DepthStagingFormat::D16) ||
        (dest.texture.pixelFormat == MTLPixelFormatDepth32Float && format == DepthStagingFormat::D32Float)) {
        TransferRuntime{device, scheduler}.Upload(source, {offset, bytes_per_row, 0, MTLBlitOptionNone}, dest);
        return;
    }
    const auto size = dest.region.size;
    const auto bytes = DepthBytes(format);
    if (dest.texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
        !source || source.device != device.Handle() || bytes_per_row < size.width * bytes ||
        offset > source.length || size.width * bytes > source.length - offset ||
        size.height - 1 > (source.length - offset - size.width * bytes) / bytes_per_row ||
        bytes_per_row > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Native packed depth/stencil upload buffer or row pitch is invalid");
    }
    // Decode into private plane buffers and use the Metal 4 encoder's native plane copies.
    // Separate allocations prevent an overlapping source/destination view from corrupting input.
    const auto count = size.width * size.height;
    auto depth = PlaneStorage(count * 4, false);
    auto stencil = PlaneStorage(count, true);
    auto info = Params(dest, nullptr, std::uint32_t(bytes_per_row));
    info.format = std::uint32_t(format);
    auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 4> buffers{{{source, offset, 0}, {uniform.buffer, uniform.offset, 1},
                                               {depth, 0, 2}, {stencil, 0, 3}}};
    auto table = scheduler.MakeArguments(buffers);
    scheduler.KeepAlive(unpack_pipeline);
    auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:unpack_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
    TransferRuntime transfer{device, scheduler};
    transfer.Upload(depth, {0, size.width * 4, 0, MTLBlitOptionDepthFromDepthStencil}, dest);
    transfer.Upload(stencil, {0, size.width, 0, MTLBlitOptionStencilFromDepthStencil}, dest);
}

void FormatConversionRuntime::BlitDepthStencil(const TextureRegion& source, const TextureRegion& dest,
                                               bool flip_x, bool flip_y) {
    const auto views = MakeDepthViews(source);
    Validate2D(dest, device.Handle());
    const auto dest_format = dest.texture.pixelFormat;
    if (dest_format != MTLPixelFormatDepth32Float_Stencil8 && dest_format != MTLPixelFormatDepth16Unorm && dest_format != MTLPixelFormatDepth32Float) {
        throw std::invalid_argument("Native depth/stencil blit requires a D32S8 destination");
    }
    const auto size = dest.region.size;
    if (size.width > std::numeric_limits<NSUInteger>::max() / size.height / 4) {
        throw std::overflow_error("Native depth/stencil blit allocation size overflow");
    }
    const auto count = size.width * size.height;
    const bool depth16 = dest_format == MTLPixelFormatDepth16Unorm;
    const auto depth_bytes = depth16 ? 2U : 4U;
    auto depth = PlaneStorage(count * depth_bytes, false);
    const bool dest_stencil = dest_format == MTLPixelFormatDepth32Float_Stencil8;
    auto stencil = dest_stencil ? PlaneStorage(count, true) : nil;
    const DepthBlitParams info{
        {std::uint32_t(source.region.origin.x), std::uint32_t(source.region.origin.y)},
        {std::uint32_t(source.region.size.width), std::uint32_t(source.region.size.height)},
        {std::uint32_t(size.width), std::uint32_t(size.height)},
        std::uint32_t(flip_x) | (std::uint32_t(flip_y) << 1) | (std::uint32_t(views.has_stencil) << 3) |
            (std::uint32_t(dest_stencil) << 4),
        std::uint32_t(depth16)};
    auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    // Metal validates the declared buffer slot even when the dynamic stencil
    // branch is disabled. Bind existing valid storage; that branch never
    // accesses it for depth-only targets, so no stencil allocation is needed.
    const std::array<BufferBinding,3> buffers{{{uniform.buffer, uniform.offset, 1}, {depth, 0, 2},
                                              {stencil ? stencil : depth, 0, 3}}};
    const std::array<TextureBinding, 2> textures{{{views.depth, 0}, {views.stencil, 1}}};
    auto table = scheduler.MakeArguments(buffers, textures);
    scheduler.KeepAlive(depth_blit_pipeline);
    auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:depth_blit_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
    // Finish sampling into separate private planes before modifying any destination pixel.
    // This also makes overlapping copies within the same texture deterministic.
    TransferRuntime transfer{device, scheduler};
    transfer.Upload(depth, {0, size.width * depth_bytes, 0, dest_stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone}, dest);
    if (dest_stencil) transfer.Upload(stencil, {0, size.width, 0, MTLBlitOptionStencilFromDepthStencil}, dest);
}

void FormatConversionRuntime::BlitColor(const TextureRegion& source, const TextureRegion& dest, bool linear,
                                        bool flip_x, bool flip_y) {
    Validate2D(source, device.Handle());
    Validate2D(dest, device.Handle());
    if (!ReadableColor(source.texture.pixelFormat) || !WritableColor(dest.texture.pixelFormat) ||
        !(source.texture.usage & MTLTextureUsageShaderRead)) {
        throw std::invalid_argument("Native color blit requires a readable float color source and writable normalized/float destination format");
    }
    const auto size = dest.region.size;
    const auto output = PrepareColorOutput(device, color_scratch, source, dest);
    const auto input = View(source.texture, source.texture.pixelFormat, source.level, source.slice);
    const DepthBlitParams info{
        {std::uint32_t(source.region.origin.x), std::uint32_t(source.region.origin.y)},
        {std::uint32_t(source.region.size.width), std::uint32_t(source.region.size.height)},
        {std::uint32_t(size.width), std::uint32_t(size.height)},
        std::uint32_t(flip_x) | (std::uint32_t(flip_y) << 1) | (std::uint32_t(linear) << 2), 0,
        {output.direct ? std::uint32_t(dest.region.origin.x) : 0U,
         output.direct ? std::uint32_t(dest.region.origin.y) : 0U}};
    auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 1> buffers{{{uniform.buffer, uniform.offset, 1}}};
    const std::array<TextureBinding, 2> textures{{{input, 0}, {output.texture, 2}}};
    const std::array<SamplerBinding, 1> samplers{{{linear ? linear_sampler : nearest_sampler, 0}}};
    auto table = scheduler.MakeArguments(buffers, textures, samplers);
    scheduler.KeepAlive(color_blit_pipeline);
    auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:color_blit_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
    if (!output.direct)
        TransferRuntime{device, scheduler}.CopyTexture({output.texture, 0, 0, MTLRegionMake2D(0, 0, size.width, size.height)}, dest);
}

void FormatConversionRuntime::PicaColorTransfer(const TextureRegion& source, const TextureRegion& dest,
                                                std::uint32_t guest_format, bool flip_x, bool flip_y) {
    Validate2D(source, device.Handle());
    Validate2D(dest, device.Handle());
    const auto input_size = source.region.size, output_size = dest.region.size;
    const auto scale_x = input_size.width / output_size.width;
    const auto scale_y = input_size.height / output_size.height;
    if (guest_format > 4 || !ReadableColor(source.texture.pixelFormat) ||
        (dest.texture.pixelFormat != MTLPixelFormatRGBA8Unorm && dest.texture.pixelFormat != MTLPixelFormatBGRA8Unorm) ||
        !(source.texture.usage & MTLTextureUsageShaderRead) ||
        (scale_x != 1 && scale_x != 2) || (scale_y != 1 && scale_y != 2) ||
        (scale_x == 1 && scale_y != 1) ||
        input_size.width != output_size.width * scale_x || input_size.height != output_size.height * scale_y)
        throw std::invalid_argument("Native PICA transfer requires exact 1x1, 2x1 or 2x2 color regions");
    const auto output = PrepareColorOutput(device, color_scratch, source, dest);
    const auto input = View(source.texture, source.texture.pixelFormat, source.level, source.slice);
    const DepthBlitParams info{
        {std::uint32_t(source.region.origin.x), std::uint32_t(source.region.origin.y)},
        {std::uint32_t(input_size.width), std::uint32_t(input_size.height)},
        {std::uint32_t(output_size.width), std::uint32_t(output_size.height)},
        std::uint32_t(flip_x) | (std::uint32_t(flip_y) << 1), guest_format,
        {output.direct ? std::uint32_t(dest.region.origin.x) : 0U,
         output.direct ? std::uint32_t(dest.region.origin.y) : 0U}};
    const auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 1> buffers{{{uniform.buffer, uniform.offset, 1}}};
    const std::array<TextureBinding, 2> textures{{{input, 0}, {output.texture, 2}}};
    const std::array<SamplerBinding, 1> samplers{{{nearest_sampler, 0}}};
    const ComputeDispatch dispatch{.pipeline=pica_transfer_pipeline,
        .buffers={buffers.begin(),buffers.end()}, .textures={textures.begin(),textures.end()},
        .samplers={samplers.begin(),samplers.end()},
        .threads=MTLSizeMake(output_size.width,output_size.height,1), .group=MTLSizeMake(8,8,1)};
    scheduler.QueueCompute(std::span{&dispatch,1}, {}, @"PICA color transfer");
    // A separate image preserves source values for aliased/overlapping transfer regions.
    if (!output.direct) TransferRuntime{device, scheduler}.CopyTexture({output.texture, 0, 0,
        MTLRegionMake2D(0, 0, output_size.width, output_size.height)}, dest);
}

void FormatConversionRuntime::ReinterpretColor16(const TextureRegion& source,
                                                  const TextureRegion& dest,
                                                  std::uint32_t source_format,
                                                  std::uint32_t dest_format) {
    Validate2D(source, device.Handle());
    Validate2D(dest, device.Handle());
    const auto size = source.region.size;
    if (source_format < 2 || source_format > 6 || dest_format < 2 || dest_format > 6 ||
        source.texture.pixelFormat != MTLPixelFormatRGBA8Unorm ||
        dest.texture.pixelFormat != MTLPixelFormatRGBA8Unorm ||
        !(source.texture.usage & MTLTextureUsageShaderRead) ||
        size.width != dest.region.size.width || size.height != dest.region.size.height) {
        throw std::invalid_argument("Native 16-bit reinterpretation requires matching RGBA8 regions and guest color formats");
    }
    const auto output = PrepareColorOutput(device, color_scratch, source, dest);
    const auto input = View(source.texture, source.texture.pixelFormat, source.level, source.slice);
    auto info = Params(source, nullptr, 0);
    if (output.direct) info.dest_origin = {std::uint32_t(dest.region.origin.x), std::uint32_t(dest.region.origin.y)};
    info.format = source_format | (dest_format << 8);
    const auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array<BufferBinding, 1> buffers{{{uniform.buffer, uniform.offset, 1}}};
    const std::array<TextureBinding, 2> textures{{{input, 0}, {output.texture, 2}}};
    const auto table = scheduler.MakeArguments(buffers, textures);
    scheduler.KeepAlive(color16_pipeline);
    const auto encoder = scheduler.BeginCompute(color16_pipeline);
    [encoder setComputePipelineState:color16_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    scheduler.EndEncoding();
    // Snapshot every source texel before copying to possibly aliased destination subresources.
    if (!output.direct) TransferRuntime{device, scheduler}.CopyTexture({output.texture, 0, 0,
        MTLRegionMake2D(0, 0, size.width, size.height)}, dest);
}

void FormatConversionRuntime::ClearRegion(const TextureRegion& dest, MTLClearColor color,
                                           double depth, std::uint32_t stencil) {
    Validate2D(dest, device.Handle());
    const auto format = dest.texture.pixelFormat;
    const bool has_stencil = format == MTLPixelFormatDepth32Float_Stencil8;
    const bool is_depth = has_stencil || format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float;
    if (!(dest.texture.usage & MTLTextureUsageRenderTarget) || (!is_depth && !WritableColor(format)) ||
        (is_depth && (!std::isfinite(depth) || depth < 0 || depth > 1 || stencil > 255)))
        throw std::invalid_argument("Invalid native direct clear target or value");
    const auto width = std::max<NSUInteger>(1, dest.texture.width >> dest.level);
    const auto height = std::max<NSUInteger>(1, dest.texture.height >> dest.level);
    if (!dest.region.origin.x && !dest.region.origin.y &&
        dest.region.size.width == width && dest.region.size.height == height) {
        TransferRuntime transfer{device, scheduler};
        if (is_depth) transfer.ClearDepthStencil(dest.texture, depth, has_stencil ? stencil : 0U,
                                                true, has_stencil, dest.level, dest.slice);
        else transfer.ClearColor(dest.texture, color, dest.level, dest.slice);
        return;
    }
    if (is_depth) {
        // Loading/storing a depth attachment may canonicalize out-of-range or
        // nonfinite D32 values outside the scissor. Fill private native planes
        // and copy ONLY the cleared rectangle, preserving every other bit.
        const auto size = dest.region.size;
        const auto count = size.width * size.height;
        const auto bytes = format == MTLPixelFormatDepth16Unorm ? 2U : 4U;
        auto depth_buffer = PlaneStorage(count * bytes, false);
        auto stencil_buffer = has_stencil ? PlaneStorage(count, true) : depth_buffer;
        struct Params { float depth; std::uint32_t stencil, count, bytes; };
        const Params info{float(depth), has_stencil ? stencil | 256U : 0U, std::uint32_t(count), bytes};
        const auto uniform = parameters.Allocate(sizeof(info));
        std::memcpy(uniform.data, &info, sizeof(info));
        const auto table = scheduler.UpdateArguments(clear_arguments,
            std::array{BufferBinding{depth_buffer, 0, 0}, BufferBinding{stencil_buffer, 0, 1},
                       BufferBinding{uniform.buffer, uniform.offset, 2}});
        scheduler.UseResource(dest.texture);
        scheduler.KeepAlive(clear_depth_pipeline);
        auto encoder = scheduler.BeginCompute(clear_depth_pipeline);
        [encoder setComputePipelineState:clear_depth_pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageBlit
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder copyFromBuffer:depth_buffer sourceOffset:0 sourceBytesPerRow:size.width * bytes sourceBytesPerImage:0
                     sourceSize:size toTexture:dest.texture destinationSlice:dest.slice destinationLevel:dest.level
              destinationOrigin:dest.region.origin options:has_stencil ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
        if (has_stencil) {
            [encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit
                            visibilityOptions:MTL4VisibilityOptionDevice];
            [encoder copyFromBuffer:stencil_buffer sourceOffset:0 sourceBytesPerRow:size.width sourceBytesPerImage:0
                         sourceSize:size toTexture:dest.texture destinationSlice:dest.slice destinationLevel:dest.level
                  destinationOrigin:dest.region.origin options:MTLBlitOptionStencilFromDepthStencil];
        }
        scheduler.EndEncoding();
        return;
    }
    auto& pipeline = clear_pipelines[format];
    if (pipeline.valid() && pipeline.wait_for(std::chrono::seconds{0})==std::future_status::ready) {
        try { (void)pipeline.get(); } catch (...) { pipeline={}; }
    }
    if (!pipeline.valid()) pipeline=device.DeferCompilation([native=&device,source=library,format] {
        auto* descriptor = [[MTL4RenderPipelineDescriptor alloc] init];
        descriptor.label = @"Azahar Metal4 direct rectangular clear";
        descriptor.vertexFunctionDescriptor = Device::Function(source, "clear_region_vertex");
        descriptor.fragmentFunctionDescriptor = Device::Function(source, "clear_region_color");
        descriptor.rasterSampleCount = 1;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        descriptor.colorAttachments[0].pixelFormat = format;
        return native->CompileRender(descriptor);
    });
    struct ClearParams { std::array<float,4> color, depth; };
    const ClearParams info{{float(color.red), float(color.green), float(color.blue), float(color.alpha)},
                           {float(depth), 0, 0, 0}};
    const auto uniform = parameters.Allocate(sizeof(info));
    std::memcpy(uniform.data, &info, sizeof(info));
    const std::array buffers{BufferBinding{uniform.buffer, uniform.offset, 0}};
    auto* pass = [[MTL4RenderPassDescriptor alloc] init];
    pass.renderTargetWidth = width;
    pass.renderTargetHeight = height;
    pass.defaultRasterSampleCount = 1;
    auto attachment = pass.colorAttachments[0];
    attachment.texture = dest.texture;
    attachment.level = dest.level;
    attachment.slice = dest.slice;
    attachment.loadAction = MTLLoadActionLoad;
    attachment.storeAction = MTLStoreActionStore;
    const RenderCommand command{.pipeline=nil,.depth=nil,
        .viewport={0,0,double(width),double(height),0,1},
        .scissor={dest.region.origin.x,dest.region.origin.y,dest.region.size.width,dest.region.size.height},
        .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.stencil=stencil,
        .vertices=3,.pending_pipeline=pipeline};
    const auto snapshot=scheduler.SnapshotRenderPass(pass);
    scheduler.QueueRender(*snapshot,command,buffers,{}, {},false);
}

} // namespace Metal
