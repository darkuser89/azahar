// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_shader_translation.h"
#include "mtl_stream_buffer.h"

namespace Metal {

enum class TriangleTopology : std::uint32_t { List, Strip, Fan };

struct VertexCaptureProgram {
    TranslatedShader shader;
    id<MTLComputePipelineState> pipeline;
    id<MTLLibrary> library{};
    std::uint8_t uniform_mask{PicaUniformMask(shader)};
    std::shared_future<id<MTLComputePipelineState>> pending_pipeline{};
    std::shared_future<id<MTLLibrary>> pending_library{};
    id<MTLComputePipelineState> Pipeline() const {
        return pipeline ? pipeline : pending_pipeline.valid() ? pending_pipeline.get():nil;
    }
    id<MTLLibrary> Library() const {
        return library ? library : pending_library.valid() ? pending_library.get():nil;
    }
};

class NativeIndirectDraw final {
    friend class ShaderMachineRuntime;
    friend class DrawRuntime;
    friend struct InputReferenceTestAccess;
    friend class RasterizerMetal;
    NativeIndirectDraw()=default;
    BufferSlice arguments,status;
    std::uint32_t capacity{};
    std::uint64_t tick{};
};

struct AssembledVertices {
    BufferSlice storage;
    std::uint32_t vertex_count{};
    std::shared_ptr<const NativeIndirectDraw> indirect;
};
struct PrimitiveSnapshot {
    std::array<std::byte, 272> history{};
    std::uint32_t pending{};
    TriangleTopology topology{TriangleTopology::List};
    bool needs_reset{true};
};

// Only stream bookkeeping lives on the CPU. Buffered vertex values stay in private GPU
// memory, including across immediate vertices, draw calls and scheduler submissions.
class PrimitiveStream final {
    friend class GeometryRuntime;
    friend class RasterizerMetal;
    friend class ShaderMachineRuntime;
    friend struct InputReferenceTestAccess;
public:
    explicit PrimitiveStream(const Device& device);
    void Reset() { pending = 0; needs_reset = true; known_zero_history = false; deferred_list = {}; }
    std::uint32_t Pending() const { return pending; }
private:
    id<MTLBuffer> history;
    std::uint32_t history_offset{};
    std::uint32_t pending{};
    TriangleTopology topology{TriangleTopology::List};
    bool needs_reset{true};
    bool known_zero_history{};
    // GPU-produced draw/dispatch arguments and conditional list metadata.
    // A CPU snapshot resolves this only after GPU completion.
    BufferSlice deferred_list;
};

class GeometryRuntime final {
public:
    GeometryRuntime(Device& device, Scheduler& scheduler);
    VertexCaptureProgram CompileVertex(std::string_view source);
    BufferSlice Capture(const VertexCaptureProgram& program, const BufferSlice& canonical_inputs,
                        std::uint32_t vertex_count, std::uint32_t first_vertex,
                        std::span<const BufferBinding> uniforms);
    AssembledVertices Assemble(const BufferSlice& captured, std::uint32_t captured_count,
                                TriangleTopology topology, std::uint32_t element_count,
                                std::uint32_t vertex_base, std::uint32_t first_vertex,
                                const BufferSlice& indices, std::uint32_t index_size,
                                bool fix_quaternions);
    MTL4LibraryFunctionDescriptor* RasterVertexFunction() const;
    id<MTLLibrary> RasterVertexLibrary() const { return library; }
    AssembledVertices Append(PrimitiveStream& stream, const BufferSlice& captured,
                            std::uint32_t captured_count, TriangleTopology topology,
                            std::uint32_t element_count, std::uint32_t vertex_base,
                            std::uint32_t first_vertex, const BufferSlice& indices,
                            std::uint32_t index_size, bool fix_quaternions,
                            const BufferSlice& shader_winding = {});
    PrimitiveSnapshot SaveStream(const PrimitiveStream& stream);
    static void ValidateStream(const PrimitiveSnapshot& snapshot);
    void LoadStream(PrimitiveStream& stream, const PrimitiveSnapshot& snapshot);

private:
    friend class ShaderMachineRuntime;
    struct PreparedCapture {
        BufferSlice output,parameters;
        id<MTLComputePipelineState> pipeline;
        std::shared_future<id<MTLComputePipelineState>> pending_pipeline;
        std::vector<BufferBinding> bindings;
        std::uint32_t vertex_count{};
    };
    PreparedCapture PrepareCapture(const VertexCaptureProgram& program,const BufferSlice& canonical_inputs,
        std::uint32_t vertex_count,std::uint32_t first_vertex,std::span<const BufferBinding> uniforms);
    void EncodeCapture(const PreparedCapture& prepared,id<MTL4ComputeCommandEncoder> existing_encoder=nil);
    static ComputeDispatch CaptureDispatch(const PreparedCapture& prepared);
    Device& device;
    Scheduler& scheduler;
    StreamBuffer output_stream;
    StreamBuffer parameter_stream;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> assembly_pipeline;
    id<MTLComputePipelineState> stream_pipeline;
    id<MTLBuffer> empty_indices;
    // Separate complete layouts for direct assembly and retained primitive streams.
    id<MTL4ArgumentTable> assembly_arguments;
};

} // namespace Metal
