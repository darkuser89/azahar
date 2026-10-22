// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_gas_runtime.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace Metal {
namespace {
struct alignas(16) GasParams {
    std::uint32_t pixels, delta_z_depth{};
    std::array<std::uint32_t, 2> padding{};
};
void Validate(const BufferSlice& slice, std::uint64_t bytes, NSUInteger alignment, id<MTLDevice> device) {
    if (!slice.buffer || slice.buffer.device != device || slice.offset % alignment ||
        slice.offset > slice.buffer.length || slice.size > slice.buffer.length - slice.offset ||
        bytes > slice.size)
        throw std::invalid_argument("Invalid native Gas buffer/device/extent/alignment");
}
void Disjoint(const BufferSlice& a, const BufferSlice& b) {
    if (a.buffer == b.buffer && a.offset < b.offset + b.size && b.offset < a.offset + a.size)
        throw std::invalid_argument("Native Gas input and output buffers overlap");
}
}

GasFeedback::GasFeedback(const Device& device) : maximum{device.CreateBuffer(4, true)} {}
GasRuntime::GasRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, parameters{device, scheduler, 1 << 20} {
    library = device.CompileLibrary(GasShaderSource, "Azahar Metal4 Gas density and postsynthesis");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    descriptor.computeFunctionDescriptor = Device::Function(library, "pica_gas_accumulate");
    accumulate_pipeline = device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor = Device::Function(library, "pica_gas_synthesize");
    synthesize_pipeline = device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor = Device::Function(library, "pica_gas_reciprocal");
    reciprocal_pipeline = device.CompileCompute(descriptor);
    auto* sampler = [[MTLSamplerDescriptor alloc] init];
    sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterNearest;
    depth_sampler = device.CreateSampler(sampler);
}

void GasRuntime::Accumulate(const BufferSlice& gas, const BufferSlice& contributions,
    const BufferSlice& scene_depth, std::uint32_t pixels, std::uint32_t delta_z_depth,
    GasFeedback& feedback) {
    Validate(gas, std::uint64_t(pixels) * 4, 4, device.Handle());
    Validate(contributions, std::uint64_t(pixels) * sizeof(GasContribution), 16, device.Handle());
    Validate(scene_depth, std::uint64_t(pixels) * 4, 4, device.Handle());
    Disjoint(gas, contributions); Disjoint(gas, scene_depth);
    if (feedback.maximum.device != device.Handle())
        throw std::invalid_argument("Native Gas feedback belongs to another device");
    if (!pixels) return;
    if (feedback.needs_reset) {
        TransferRuntime{device, scheduler}.FillBuffer(feedback.maximum, NSMakeRange(0, 4), 0);
        feedback.needs_reset = false;
    }
    const GasParams info{pixels, delta_z_depth};
    const auto params = parameters.Allocate(sizeof(info));
    std::memcpy(params.data, &info, sizeof(info));
    const std::array<BufferBinding, 5> bindings{{{gas.buffer, gas.offset, 0},
        {contributions.buffer, contributions.offset, 1}, {scene_depth.buffer, scene_depth.offset, 2},
        {feedback.maximum, 0, 3}, {params.buffer, params.offset, 4}}};
    const auto table = scheduler.MakeArguments(bindings);
    scheduler.KeepAlive(accumulate_pipeline);
    const auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:accumulate_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(pixels, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    scheduler.EndEncoding();
}

void GasRuntime::Synthesize(const BufferSlice& gas, const BufferSlice& lighting,
    const BufferSlice& output, std::uint32_t pixels, const GasUniforms& config) {
    Validate(gas, std::uint64_t(pixels) * 4, 4, device.Handle());
    Validate(lighting, std::uint64_t(pixels) * 4, 4, device.Handle());
    Validate(output, std::uint64_t(pixels) * 16, 16, device.Handle());
    Disjoint(output, gas); Disjoint(output, lighting);
    if (config.shading_depth > 1 || config.padding)
        throw std::invalid_argument("Invalid native Gas synthesis configuration flags/padding");
    if (!pixels) return;
    const GasParams info{pixels};
    const auto params = parameters.Allocate(sizeof(info) + sizeof(config));
    std::memcpy(params.data, &info, sizeof(info));
    std::memcpy(static_cast<std::byte*>(params.data) + sizeof(info), &config, sizeof(config));
    const std::array<BufferBinding, 5> bindings{{{gas.buffer, gas.offset, 0},
        {lighting.buffer, lighting.offset, 1}, {output.buffer, output.offset, 2},
        {params.buffer, params.offset + sizeof(info), 3}, {params.buffer, params.offset, 4}}};
    const auto table = scheduler.MakeArguments(bindings);
    scheduler.KeepAlive(synthesize_pipeline);
    const auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:synthesize_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(pixels, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    scheduler.EndEncoding();
}

std::uint16_t GasRuntime::ReadMaximum(const GasFeedback& feedback) {
    if (feedback.maximum.device != device.Handle())
        throw std::invalid_argument("Native Gas feedback belongs to another device");
    const bool complete = scheduler.IsComplete(scheduler.LastUseTick(feedback.maximum));
    if (feedback.needs_reset) return 0;
    // Apple Silicon shared memory needs completion, but no staging allocation
    // or extra blit. GPU::ReadReg, GX feedback and saving explicitly publish
    // a deferred maximum before it becomes observable to the CPU/archive.
    // A completed resource is already coherent. Do not submit/drain unrelated
    // work on a repeated observation. A newly bound writer receives a newer
    // use tick, so its result still waits before publication to the CPU.
    if (!complete) scheduler.Finish(Scheduler::WaitReason::Readback);
    std::uint32_t result{};
    std::memcpy(&result, feedback.maximum.contents, 4);
    if (result > 0x7c00u) throw std::runtime_error("Invalid native Gas maximum feedback");
    return std::uint16_t(result);
}
void GasRuntime::LoadMaximum(GasFeedback& feedback, std::uint16_t maximum) {
    if (feedback.maximum.device != device.Handle() || maximum > 0x7c00u)
        throw std::invalid_argument("Invalid native Gas maximum feedback import");
    const auto source = parameters.Allocate(4);
    const std::uint32_t value = maximum;
    std::memcpy(source.data, &value, 4);
    TransferRuntime{device, scheduler}.CopyBuffer(source.buffer, source.offset, feedback.maximum, 0, 4);
    feedback.needs_reset = false;
}

std::uint16_t GasRuntime::ReadReciprocalMaximum(GasFeedback& feedback) {
    const auto input = PrepareFragmentMaximum(feedback);
    const auto output = device.CreateBuffer(4, true);
    const std::array bindings{BufferBinding{input.buffer, input.offset, 0}, BufferBinding{output, 0, 1}};
    const auto arguments = scheduler.MakeArguments(bindings);
    scheduler.KeepAlive(reciprocal_pipeline);
    const auto encoder = scheduler.BeginCompute();
    [encoder setComputePipelineState:reciprocal_pipeline];
    [encoder setArgumentTable:arguments];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    scheduler.EndEncoding();
    scheduler.Finish(Scheduler::WaitReason::Readback);
    std::uint32_t result{};
    std::memcpy(&result, output.contents, 4);
    if (result > 0x7c00) throw std::runtime_error("Invalid native Gas reciprocal maximum feedback");
    return std::uint16_t(result);
}

BufferBinding GasRuntime::PrepareFragmentMaximum(GasFeedback& feedback) {
    if (feedback.maximum.device != device.Handle())
        throw std::invalid_argument("Native Gas feedback belongs to another device");
    if (feedback.needs_reset) {
        TransferRuntime{device, scheduler}.FillBuffer(feedback.maximum, NSMakeRange(0, 4), 0);
        feedback.needs_reset = false;
    }
    return {feedback.maximum, 0, 24};
}

id<MTLTexture> GasRuntime::SnapshotDepth(id<MTLTexture> depth, std::uint32_t level) {
    if (depth && (depth.device != device.Handle() || depth.textureType != MTLTextureType2D ||
        depth.sampleCount != 1 || level >= depth.mipmapLevelCount ||
        (depth.pixelFormat != MTLPixelFormatDepth16Unorm && depth.pixelFormat != MTLPixelFormatDepth32Float &&
         depth.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)))
        throw std::invalid_argument("Invalid native Gas scene depth texture/device/mip");
    const NSUInteger width = depth ? std::max<NSUInteger>(1, depth.width >> level) : 1;
    const NSUInteger height = depth ? std::max<NSUInteger>(1, depth.height >> level) : 1;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:depth ? depth.pixelFormat : MTLPixelFormatDepth32Float
                                                                      width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    const auto copy = device.CreateTexture(descriptor);
    TransferRuntime transfer{device, scheduler};
    if (depth) transfer.CopyTexture({depth, level, 0, MTLRegionMake2D(0, 0, width, height)},
                                   {copy, 0, 0, MTLRegionMake2D(0, 0, width, height)});
    else transfer.ClearDepthStencil(copy, 1, 0, true, false);
    // A combined depth/stencil texture exposes depth to floating-point reads.
    // Metal does not permit reinterpreting it as a Depth32Float texture view.
    return copy;
}

id<MTLTexture> GasRuntime::SceneDepthForDraw(id<MTLTexture> depth, std::uint32_t level, bool reads_depth) {
    if (!reads_depth) {
        if (!unused_depth) unused_depth = SnapshotDepth(nil);
        return unused_depth;
    }
    if (!depth || depth.device != device.Handle() || depth.textureType != MTLTextureType2D ||
        depth.sampleCount != 1 || level >= depth.mipmapLevelCount ||
        (depth.pixelFormat != MTLPixelFormatDepth16Unorm && depth.pixelFormat != MTLPixelFormatDepth32Float &&
         depth.pixelFormat != MTLPixelFormatDepth32Float_Stencil8))
        throw std::invalid_argument("Invalid native Gas scene depth texture/device/mip");
    // Reuse only after the preceding draw was encoded. CopyTexture closes its
    // render pass and establishes All -> compute/blit ordering before overwriting.
    const auto width = std::max<NSUInteger>(1, depth.width >> level);
    const auto height = std::max<NSUInteger>(1, depth.height >> level);
    if (!scene_depth_scratch || scene_depth_scratch == depth ||
        scene_depth_scratch.width != width || scene_depth_scratch.height != height ||
        scene_depth_scratch.pixelFormat != depth.pixelFormat) {
        scene_depth_scratch = SnapshotDepth(depth, level);
    } else {
        TransferRuntime{device, scheduler}.CopyTexture({depth, level, 0, MTLRegionMake2D(0, 0, width, height)},
            {scene_depth_scratch, 0, 0, MTLRegionMake2D(0, 0, width, height)});
    }
    return scene_depth_scratch;
}
} // namespace Metal
