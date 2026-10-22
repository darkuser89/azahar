// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "mtl_stream_buffer.h"

namespace Metal {

// Register words remain unconverted on the host. Both density evaluation and
// look-up/interpolation execute on the GPU. Fog entries use the PICA 13+11 layout.
struct alignas(16) GasUniforms {
    std::array<std::uint32_t, 16> color_lut{};
    std::array<std::uint32_t, 128> fog_lut{};
    std::uint32_t attenuation{}, reciprocal_maximum{};
    std::uint32_t light_xy{}, light_z{}, light_z_color{}, shading_depth{};
    std::uint32_t delta_z_depth{}, padding{};
};
static_assert(sizeof(GasUniforms) == 608);

// One rasterized contribution per pixel: density, fragment depth, coverage, pad.
// The rasterization stage supplies these values; this runtime does not rasterize.
struct alignas(16) GasContribution {
    float density{}, depth{}, coverage{}, padding{};
};

class GasFeedback final {
    friend class GasRuntime;
public:
    explicit GasFeedback(const Device& device);
    void Reset() { needs_reset = true; }
private:
    id<MTLBuffer> maximum;
    bool needs_reset{true};
};

// Native Gas stages. Gas buffers contain the exact four guest bytes per pixel,
// {plainHigh, plainLow, depthHigh, depthLow}; no RGBA/unorm conversion occurs.
class GasRuntime final {
public:
    GasRuntime(Device& device, Scheduler& scheduler);
    void Accumulate(const BufferSlice& gas, const BufferSlice& contributions,
                    const BufferSlice& scene_depth, std::uint32_t pixels,
                    std::uint32_t delta_z_depth, GasFeedback& feedback);
    // lighting is the last TEV stage's first source red component, one float/pixel.
    // output is the Gas combiner result, float4/pixel, before alpha blending.
    void Synthesize(const BufferSlice& gas, const BufferSlice& lighting,
                    const BufferSlice& output, std::uint32_t pixels,
                    const GasUniforms& config);
    std::uint16_t ReadMaximum(const GasFeedback& feedback);
    std::uint16_t ReadReciprocalMaximum(GasFeedback& feedback);
    void LoadMaximum(GasFeedback& feedback, std::uint16_t maximum);
    BufferBinding PrepareFragmentMaximum(GasFeedback& feedback);
    // A stable GPU snapshot prevents the accumulation pass reading an attachment
    // that another output operation might also modify. A nil source binds an
    // initialized one-pixel texture for Never/Always, which do not read depth.
    id<MTLTexture> SnapshotDepth(id<MTLTexture> depth, std::uint32_t level = 0);
    // Borrowed only for the immediately encoded draw. Reuse is ordered on the
    // same scheduler; callers needing an independently retained image use SnapshotDepth.
    id<MTLTexture> SceneDepthForDraw(id<MTLTexture> depth, std::uint32_t level, bool reads_depth);
    id<MTLSamplerState> DepthSampler() const { return depth_sampler; }
private:
    Device& device;
    Scheduler& scheduler;
    StreamBuffer parameters;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> accumulate_pipeline, synthesize_pipeline, reciprocal_pipeline;
    id<MTLSamplerState> depth_sampler;
    id<MTLTexture> scene_depth_scratch, unused_depth;
};
} // namespace Metal
