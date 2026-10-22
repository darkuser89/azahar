// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_pipeline_cache.h"
#include "mtl_shader_runtime.h"
#include "mtl_input_reference.h"
#include "video_core/pica/regs_internal.h"

namespace Metal {

struct PicaDrawState {
    BlendState blend;
    DepthStencilState depth;
    MTLCullMode cull{MTLCullModeNone};
    MTLWinding winding{MTLWindingCounterClockwise};
    std::uint32_t stencil_reference{};
    bool use_color{};
    bool use_depth{};
    bool shadow{};
    bool gas_accumulation{};
    FragmentColorFormat color_format{};
};

PicaDrawState BuildPicaDrawState(const Pica::RegsInternal& regs);

struct NativeMeshChunk {
    BufferSlice parameters;
    std::uint32_t groups{};
};

// Constructed only from the rasterizer's immutable uploads. It performs no
// history-buffer writes: complete list draws publish canonical zero history.
class NativeMeshDraw final {
    friend class RasterizerMetal;
    friend class DrawRuntime;
    friend struct InputReferenceTestAccess;
    NativeMeshDraw() = default;
    std::shared_ptr<const CompiledMeshVertexProgram> program;
    std::shared_ptr<const NativeVertexInputSnapshot> inputs;
    BufferSlice references, capture;
    std::vector<NativeMeshChunk> chunks;
    id<MTLBuffer> empty;
    std::uint32_t vertex_count{}, element_count{}, index_size{};
};

// One complete native render draw after GPU vertex fetch/capture/assembly. Argument tables,
// shaders and state objects are retained through submission by the scheduler.
class DrawRuntime final {
public:
    DrawRuntime(Device& device, Scheduler& scheduler, GeometryRuntime& geometry);
    void Draw(const PicaDrawState& state, const CompiledFragmentShader& fragment,
              MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
              const AssembledVertices& vertices, std::span<const BufferBinding> uniforms,
              std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers);
    void Clear();
    void DrawMesh(const PicaDrawState& state, const CompiledFragmentShader& fragment,
                  MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
                  const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
                  std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers);
    void DrawMesh(const PicaDrawState& state, const CompiledFragmentShader& fragment,
                  const ImmutableRenderPass& pass, MTLViewport viewport, MTLScissorRect scissor,
                  const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
                  std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers);
    PipelinePrewarmResult Prewarm(const std::atomic_bool& stop,
        const std::function<void(std::size_t, std::size_t)>& progress = {});
    PipelineCacheStats PipelineStats() const;

private:
    void DrawMeshImpl(const PicaDrawState& state, const CompiledFragmentShader& fragment,
                  MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
                  const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
                  std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers,
                  const ImmutableRenderPass* snapshot);
    Device& device;
    Scheduler& scheduler;
    GeometryRuntime& geometry;
    PipelineCache pipelines;
};

} // namespace Metal
