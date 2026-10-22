// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#import <QuartzCore/CAMetalLayer.h>
#include "mtl_stream_buffer.h"
#include "mtl_pipeline_cache.h"

namespace Metal {

struct PresentVertex {
    std::array<float, 2> position;
    std::array<float, 2> texcoord;
};
static_assert(sizeof(PresentVertex) == 16);
static_assert(offsetof(PresentVertex, texcoord) == 8);

enum class PresentMode : std::int32_t { Single, Anaglyph, Interlaced };

struct alignas(16) PresentUniforms {
    std::array<float, 16> modelview;
    std::array<float, 4> input_resolution;
    std::array<float, 4> output_resolution;
    std::int32_t left_screen{};
    std::int32_t right_screen{};
    PresentMode mode{};
    std::int32_t reverse_interlaced{};
    std::array<float, 4> right_texcoord_transform{1, 1, 0, 0};
};
static_assert(sizeof(PresentUniforms) == 128);
static_assert(offsetof(PresentUniforms, input_resolution) == 64);
static_assert(offsetof(PresentUniforms, output_resolution) == 80);
static_assert(offsetof(PresentUniforms, left_screen) == 96);
static_assert(offsetof(PresentUniforms, reverse_interlaced) == 108);
static_assert(offsetof(PresentUniforms, right_texcoord_transform) == 112);

struct PresentDraw {
    std::array<PresentVertex, 4> vertices;
    PresentUniforms uniforms;
    float opacity{1.0f};
    bool linear_filter{};
};

class Compositor final {
public:
    Compositor(Device& device, Scheduler& scheduler);
    void Render(id<MTLTexture> target, const std::array<id<MTLTexture>, 3>& screens,
                std::span<const PresentDraw> draws, MTLClearColor background,
                std::span<const std::array<float, 2>> cursor = {});
    PipelinePrewarmResult Prewarm(const std::atomic_bool& stop,
        const std::function<void(std::size_t, std::size_t)>& progress = {});
    PipelineCacheStats PipelineStats() const { return cache.Stats(); }

private:
    struct Pipelines {
        std::shared_future<id<MTLRenderPipelineState>> screens;
        std::shared_future<id<MTLRenderPipelineState>> cursor;
    };
    Pipelines GetPipelines(MTLPixelFormat format);

    Device& device;
    Scheduler& scheduler;
    StreamBuffer stream;
    PipelineCache cache;
    id<MTLLibrary> library;
    std::array<id<MTLSamplerState>, 2> samplers;
};

class PresentSurface final {
public:
    PresentSurface(const Device& device, void* metal_layer);
    ~PresentSurface();
    PresentSurface(const PresentSurface&) = delete;
    PresentSurface& operator=(const PresentSurface&) = delete;
    void SetSize(NSUInteger width, NSUInteger height);
    void SetVSync(bool enabled);
    // Ready includes a completed native nil result. Never waits for the worker.
    bool HasReady();
    id<CAMetalDrawable> Acquire();
    // Request/consume one native image without waiting for WindowServer. Guest
    // GPU work is submitted independently when no presentation image is ready.
    id<CAMetalDrawable> TryAcquire();

private:
    id<CAMetalDrawable> Acquire(bool wait);
    struct Acquisition;
    std::unique_ptr<Acquisition> acquisition;
};

} // namespace Metal
