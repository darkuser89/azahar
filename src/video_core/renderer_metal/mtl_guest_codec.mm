// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_guest_codec.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace Metal {
namespace {
struct alignas(16) CodecParams {
    u32 width, height, stride, format;
    u32 bits_per_pixel, flags, linear_bpp, padding{};
    std::uint64_t byte_begin, byte_count;
};
static_assert(sizeof(CodecParams) == 48);
static_assert(offsetof(CodecParams, byte_begin) == 32);

u32 Bits(const GuestTextureLayout& layout) {
    const auto index = u32(layout.format);
    if (index >= VideoCore::FORMAT_MAP.size() || index == 15)
        throw std::invalid_argument("Unknown Metal guest texture format");
    return VideoCore::FORMAT_MAP[index].bits_per_block;
}
void Validate(const GuestTextureLayout& layout) {
    Bits(layout);
    if (!layout.width || !layout.height || layout.stride < layout.width ||
        std::uint64_t(layout.width) * layout.height > std::numeric_limits<u32>::max() ||
        (layout.tiled && (layout.width % 8 || layout.height % 8 || layout.stride % 8)) ||
        (!layout.tiled && (layout.format == VideoCore::PixelFormat::ETC1 || layout.format == VideoCore::PixelFormat::ETC1A4)))
        throw std::invalid_argument("Invalid Metal guest texture dimensions, stride or tile alignment");
}
void ValidateSlice(const BufferSlice& slice, NSUInteger required, id<MTLDevice> device) {
    if (!slice.buffer || slice.buffer.device != device || slice.offset > slice.buffer.length ||
        slice.size > slice.buffer.length - slice.offset || required > slice.size)
        throw std::invalid_argument("Metal guest codec buffer slice is too short or invalid");
}

void QueueCodec(Scheduler& scheduler, id<MTLComputePipelineState> pipeline,
                std::span<const BufferBinding> buffers, std::span<const TextureBinding> textures,
                MTLSize threads, MTLSize group) {
    const ComputeDispatch dispatch{.pipeline=pipeline, .buffers={buffers.begin(),buffers.end()},
        .textures={textures.begin(),textures.end()}, .threads=threads, .group=group};
    scheduler.QueueCompute(std::span{&dispatch,1}, {}, @"guest memory codec");
}
}

NSUInteger GuestTextureCodec::GuestBytes(const GuestTextureLayout& layout) {
    Validate(layout);
    const auto pixels = layout.tiled
        ? std::uint64_t(layout.stride) * 8 * (layout.height / 8 - 1) + std::uint64_t(layout.width) * 8
        : std::uint64_t(layout.stride) * (layout.height - 1) + layout.width;
    const auto bits = Bits(layout);
    if (pixels > (std::numeric_limits<NSUInteger>::max() - 7) / bits)
        throw std::overflow_error("Metal guest texture byte count overflows");
    return (pixels * bits + 7) / 8;
}
u32 GuestTextureCodec::LinearBytesPerPixel(const GuestTextureLayout& layout) {
    Bits(layout);
    if (layout.format == VideoCore::PixelFormat::D16) return 2;
    if (!layout.converted && u32(layout.format) <= u32(VideoCore::PixelFormat::RGBA4)) return Bits(layout) / 8;
    return 4;
}
NSUInteger GuestTextureCodec::LinearBytes(const GuestTextureLayout& layout) {
    Validate(layout);
    return NSUInteger(layout.width) * layout.height * LinearBytesPerPixel(layout);
}

GuestTextureCodec::GuestTextureCodec(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, uniforms{device, scheduler, 64 << 10} {
    library = device.CompileLibrary(GuestCodecShaderSource, "Azahar Metal 4 guest memory codec");
    const auto compile = [&](std::string_view entry) {
        auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor = Device::Function(library, entry);
        descriptor.maxTotalThreadsPerThreadgroup = 64;
        return device.CompileCompute(descriptor);
    };
    decode_pipeline = compile("decode_guest_texture");
    encode_pipeline = compile("encode_guest_texture");
    depth_encode_pipeline = compile("encode_guest_depth");
    fill_pipeline = compile("fill_guest_memory");
    copy_pipeline = compile("copy_guest_memory");
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint
        width:1 height:1 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    empty_stencil = device.CreateTexture(descriptor);
    auto zero = device.CreateBuffer(1, true);
    *static_cast<u8*>(zero.contents) = 0;
    TransferRuntime{device, scheduler}.Upload(zero, {0, 1, 0, MTLBlitOptionNone},
        {empty_stencil, 0, 0, MTLRegionMake2D(0, 0, 1, 1)});
}
void GuestTextureCodec::Decode(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest) {
    Run(layout, source, dest, {}, true);
}
void GuestTextureCodec::Encode(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest,
                               NSRange byte_range) {
    if (layout.format == VideoCore::PixelFormat::ETC1 || layout.format == VideoCore::PixelFormat::ETC1A4)
        throw std::invalid_argument("ETC is not a writable PICA framebuffer format");
    Run(layout, source, dest, byte_range, false);
}
void GuestTextureCodec::EncodeDepth(const GuestTextureLayout& layout, id<MTLTexture> depth,
                                    id<MTLTexture> stencil, MTLOrigin origin,
                                    const BufferSlice& dest, NSRange byte_range) {
    const auto bytes = GuestBytes(layout);
    const auto format = layout.format;
    const auto native = format == VideoCore::PixelFormat::D16 ? MTLPixelFormatDepth16Unorm
                      : format == VideoCore::PixelFormat::D24 ? MTLPixelFormatDepth32Float
                      : format == VideoCore::PixelFormat::D24S8 ? MTLPixelFormatDepth32Float_Stencil8
                      : MTLPixelFormatInvalid;
    ValidateSlice(dest, bytes, device.Handle());
    if (!layout.converted || native == MTLPixelFormatInvalid || !depth ||
        depth.device != device.Handle() || depth.pixelFormat != native ||
        depth.textureType != MTLTextureType2D || depth.sampleCount != 1 ||
        depth.framebufferOnly || depth.storageMode == MTLStorageModeMemoryless ||
        !(depth.usage & MTLTextureUsageShaderRead) || origin.z ||
        origin.x > depth.width || layout.width > depth.width - origin.x ||
        origin.y > depth.height || layout.height > depth.height - origin.y ||
        origin.x > std::numeric_limits<u32>::max() || origin.y > std::numeric_limits<u32>::max() ||
        byte_range.location > bytes || byte_range.length > bytes - byte_range.location ||
        byte_range.length > std::numeric_limits<u32>::max())
        throw std::invalid_argument("Native depth guest encoding exceeds its format, view or byte range");
    if (format == VideoCore::PixelFormat::D24S8 &&
        (!stencil || stencil.device != device.Handle() ||
         stencil.textureType != MTLTextureType2D || stencil.sampleCount != 1 ||
         stencil.pixelFormat != MTLPixelFormatX32_Stencil8 ||
         !(stencil.usage & MTLTextureUsageShaderRead) ||
         stencil.width != depth.width || stencil.height != depth.height))
        throw std::invalid_argument("Native guest depth encoding requires a matching stencil plane");
    if (!byte_range.length) return;
    struct alignas(16) DepthParams {
        CodecParams codec;
        std::array<u32, 2> origin;
        std::array<u32, 2> padding{};
    };
    static_assert(sizeof(DepthParams) == 64);
    const DepthParams info{{layout.width, layout.height, layout.stride, u32(format), Bits(layout),
        u32(layout.tiled) | (u32(layout.converted) << 1) | (u32(layout.flip_output) << 2),
        LinearBytesPerPixel(layout), 0, byte_range.location, byte_range.length},
        {u32(origin.x), u32(origin.y)}};
    auto params = uniforms.Allocate(sizeof(info));
    std::memcpy(params.data, &info, sizeof(info));
    const std::array<BufferBinding, 2> buffers{{{dest.buffer, dest.offset, 1},
                                               {params.buffer, params.offset, 2}}};
    const std::array<TextureBinding, 2> textures{{{depth, 0},
        {format == VideoCore::PixelFormat::D24S8 ? stencil : empty_stencil, 1}}};
    const auto bpp = Bits(layout) / 8;
    const auto pixels = (byte_range.location + byte_range.length + bpp - 1) / bpp - byte_range.location / bpp;
    QueueCodec(scheduler,depth_encode_pipeline,buffers,textures,MTLSizeMake(pixels,1,1),MTLSizeMake(64,1,1));
}
void GuestTextureCodec::Run(const GuestTextureLayout& layout, const BufferSlice& source, const BufferSlice& dest,
                            NSRange byte_range, bool decode) {
    const auto guest_bytes = GuestBytes(layout), linear_bytes = LinearBytes(layout);
    ValidateSlice(source, decode ? guest_bytes : linear_bytes, device.Handle());
    ValidateSlice(dest, decode ? linear_bytes : guest_bytes, device.Handle());
    if (source.buffer == dest.buffer && source.offset < dest.offset + dest.size && dest.offset < source.offset + source.size)
        throw std::invalid_argument("Metal guest codec input/output slices overlap");
    if (!decode && (byte_range.location > guest_bytes || byte_range.length > guest_bytes - byte_range.location ||
                    byte_range.length > std::numeric_limits<u32>::max()))
        throw std::invalid_argument("Metal guest encoding byte range exceeds its allocation");
    if (!decode && !byte_range.length) return;
    const CodecParams info{layout.width, layout.height, layout.stride, u32(layout.format), Bits(layout),
        u32(layout.tiled) | (u32(layout.converted) << 1) | (u32(layout.flip_output) << 2),
        LinearBytesPerPixel(layout), 0, byte_range.location, byte_range.length};
    auto params = uniforms.Allocate(sizeof(info));
    std::memcpy(params.data, &info, sizeof(info));
    const std::array<BufferBinding, 3> bindings{{{source.buffer, source.offset, 0}, {dest.buffer, dest.offset, 1},
                                               {params.buffer, params.offset, 2}}};
    QueueCodec(scheduler,decode ? decode_pipeline : encode_pipeline,bindings,{},
        decode ? MTLSizeMake(layout.width,layout.height,1) : MTLSizeMake(byte_range.length,1,1),
        decode ? MTLSizeMake(8,8,1) : MTLSizeMake(64,1,1));
}
void GuestTextureCodec::Fill(const BufferSlice& dest, std::array<u8, 4> pattern, u32 pattern_size, u32 phase) {
    ValidateSlice(dest, dest.size, device.Handle());
    if (pattern_size < 2 || pattern_size > 4 || phase >= pattern_size || dest.size > std::numeric_limits<u32>::max())
        throw std::invalid_argument("Invalid native guest fill pattern or byte range");
    if (!dest.size) return;
    struct alignas(16) FillParams { u32 pattern, size, phase, padding{}; std::uint64_t count, padding2{}; };
    static_assert(sizeof(FillParams) == 32);
    const FillParams info{u32(pattern[0]) | (u32(pattern[1]) << 8) | (u32(pattern[2]) << 16) | (u32(pattern[3]) << 24),
                          pattern_size, phase, 0, dest.size};
    auto params = uniforms.Allocate(sizeof(info));
    std::memcpy(params.data, &info, sizeof(info));
    const std::array<BufferBinding, 2> bindings{{{dest.buffer, dest.offset, 0}, {params.buffer, params.offset, 1}}};
    QueueCodec(scheduler,fill_pipeline,bindings,{},MTLSizeMake(dest.size,1,1),MTLSizeMake(64,1,1));
}

std::uint64_t GuestTextureCodec::CopyBytes(std::uint64_t count, u32 width, u32 gap) {
    if (!count) return 0;
    if (!width || count > std::numeric_limits<u32>::max())
        throw std::invalid_argument("Invalid native guest copy count or line width");
    // No gap follows the final partial/full line. Arithmetic is widened before multiplication.
    return count + ((count - 1) / width) * std::uint64_t(gap);
}

void GuestTextureCodec::Copy(const GuestCopyLayout& layout, const BufferSlice& source, const BufferSlice& dest) {
    const auto input_bytes = CopyBytes(layout.byte_count, layout.input_width, layout.input_gap);
    const auto output_bytes = CopyBytes(layout.byte_count, layout.output_width, layout.output_gap);
    if (!layout.byte_count) return;
    ValidateSlice(source, input_bytes, device.Handle());
    ValidateSlice(dest, output_bytes, device.Handle());
    if (source.buffer == dest.buffer && source.offset < dest.offset + output_bytes && dest.offset < source.offset + input_bytes)
        throw std::invalid_argument("Native guest copy requires nonoverlapping GPU snapshots");
    struct alignas(16) CopyParams {
        std::uint64_t count;
        u32 input_width, input_gap, output_width, output_gap;
        std::uint64_t padding{};
    };
    static_assert(sizeof(CopyParams) == 32);
    const CopyParams info{layout.byte_count, layout.input_width, layout.input_gap, layout.output_width, layout.output_gap};
    const auto params = uniforms.Allocate(sizeof(info));
    std::memcpy(params.data, &info, sizeof(info));
    const std::array<BufferBinding, 3> bindings{{{source.buffer, source.offset, 0}, {dest.buffer, dest.offset, 1},
                                               {params.buffer, params.offset, 2}}};
    QueueCodec(scheduler,copy_pipeline,bindings,{},MTLSizeMake(layout.byte_count,1,1),MTLSizeMake(64,1,1));
}
} // namespace Metal
