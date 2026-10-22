// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "mtl_shader_machine.h"
#include <vector>

namespace Metal {
enum class GeometryInputMode : std::uint32_t { Point, Fixed, Variable };
struct GeometryInputConfig {
    GeometryInputMode mode{};
    std::uint32_t vs_output_mask{};
    std::uint32_t vs_output_count{};
    std::uint32_t input_count{};
    std::array<std::uint32_t,16> input_registers{};
    std::uint32_t uniform_start{};
    std::uint32_t fixed_vertex_count{};
    std::uint32_t variable_main_count{};
    bool operator==(const GeometryInputConfig&) const = default;
};
struct GeometryInputSnapshot {
    std::array<std::byte,2128> history{};
    GeometryInputConfig config{};
    std::uint32_t cursor{},remaining{},main_remaining{},total_count{};
    bool configured{},initialized{},awaiting{};
    bool operator==(const GeometryInputSnapshot&) const = default;
};
class GeometryInputState final {
    friend class GeometryProgramRuntime;
public:
    explicit GeometryInputState(const Device& device);
    bool NeedsCount() const { return configured && config.mode==GeometryInputMode::Variable && !remaining && !awaiting; }
    bool IsEmpty() const { return !cursor && !remaining && !awaiting; }
private:
    struct UniformWrite {
        std::array<std::uint32_t,3> float_masks;
        std::uint32_t bool_mask,integer_mask,boolean_value_offset,data_offset,float_blocks;
    };
    // Bounded staging of explicit CPU writes, never a mirror of the GPU bank.
    // Unwritten slots are ignored. Boolean slots retain whole source words;
    // the GPU alone applies masks and preserves GPU-written inputs/b15.
    struct UniformStaging {
        UniformWrite info{};
        std::array<std::array<std::uint32_t,4>,96> floats{};
        std::array<std::array<std::uint32_t,4>,4> integers{};
        std::array<std::uint32_t,16> boolean_sources{};
        bool Empty() const {
            return !(info.float_masks[0]|info.float_masks[1]|info.float_masks[2]|info.bool_mask|info.integer_mask);
        }
        void Clear() { info={}; }
    };
    static_assert(sizeof(UniformStaging)==1696);
    mutable UniformStaging pending_uniforms;
    id<MTLBuffer> uniforms,point,inputs;
    GeometryInputConfig config{};
    std::uint32_t cursor{},remaining{},main_remaining{},total_count{};
    bool configured{},initialized{},awaiting{};
};
struct GeometryQueuedInvocation {
    ShaderMachinePendingResult execution;
    BufferSlice history;
    GeometryInputConfig config{};
    std::uint32_t cursor{},remaining{},main_remaining{},total_count{};
    bool configured{},initialized{},awaiting{};
};
// CPU bookkeeping describes groups and queues raw transport records.
// Attribute values, masked uniform updates and b15 remain on the GPU.
class GeometryProgramRuntime final {
public:
    GeometryProgramRuntime(Device& device,Scheduler& scheduler);
    void Configure(GeometryInputState& state,const GeometryInputConfig& config);
    void UpdateUniforms(GeometryInputState& state,const ShaderMachineUniforms& values,
        std::array<std::uint32_t,3> float_mask={0xffffffff,0xffffffff,0xffffffff},
        std::uint32_t bool_mask=0xffff,std::uint32_t integer_mask=15);
    void BeginVariable(GeometryInputState& state,std::uint32_t count);
    bool SubmitVertex(GeometryInputState& state,const BufferSlice& raw_vs_output);
    ShaderMachineResult Invoke(GeometryInputState& state,ShaderMachineRuntime& machine,
        ShaderMachineState& unit,const ShaderMachineProgramView& program,
        std::uint32_t emission_capacity=192);
    GeometryQueuedInvocation InvokeQueued(GeometryInputState& state,ShaderMachineRuntime& machine,
        ShaderMachineState& unit,const ShaderMachineProgramView& program,const BufferSlice& abort,
        std::uint32_t emission_capacity=192);
    // Complete a bounded chunk in order. Overflow retries only the failed GS
    // and guarded successors; successful prefixes can be consumed and released.
    void CompleteQueued(GeometryInputState& state,ShaderMachineRuntime& machine,ShaderMachineState& unit,
        std::span<GeometryQueuedInvocation> pending,const std::function<void(const ShaderMachineResult&)>& consume);
    GeometryInputSnapshot SaveState(const GeometryInputState& state);
    static void ValidateState(const GeometryInputSnapshot& snapshot);
    void LoadState(GeometryInputState& state,const GeometryInputSnapshot& snapshot);
private:
    friend class RasterizerMetal;
    friend struct InputReferenceTestAccess;
    GeometryQueuedInvocation InvokeQueuedImpl(GeometryInputState& state,ShaderMachineRuntime& machine,
        ShaderMachineState& unit,const ShaderMachineProgramView& program,const BufferSlice& abort,
        std::uint32_t emission_capacity,const BufferSlice& emissions,bool compact=false);
    void Invoked(GeometryInputState& state);
    void RestoreQueuedInput(GeometryInputState& state,const GeometryQueuedInvocation& pending);
    void CopyInputHistory(const GeometryInputState& state,const BufferSlice& history,bool restore);
    void Initialize(GeometryInputState& state);
    void FlushUniforms(const GeometryInputState& state);
    void EncodeInputs(GeometryInputState& state,const BufferSlice& source,
        std::uint32_t count,std::uint32_t destination,bool ready,std::uint32_t variable_count=0);
    Device& device;
    Scheduler& scheduler;
    StreamBuffer parameters;
    SharedUploadPool snapshots;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> input_pipeline,uniform_pipeline,invoked_pipeline;
    id<MTLBuffer> empty;
};
}
