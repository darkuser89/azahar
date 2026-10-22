// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_stream_buffer.h"

namespace Pica { struct RegsInternal; }

namespace Metal {

struct InputProvenance;
class InputReferenceRuntime;
class RasterizerMetal;

enum class VertexAttributeFormat : std::uint32_t { Byte, UnsignedByte, Short, Float };
using VertexRegisters = std::array<std::array<float, 4>, 16>;

inline VertexRegisters DefaultVertexRegisters() {
    VertexRegisters result{};
    for (auto& value : result) value[3] = 1;
    return result;
}

struct VertexAttribute {
    std::uint32_t buffer_index{};
    std::uint32_t offset{};
    std::uint32_t stride{};
    std::uint32_t components{4};
    VertexAttributeFormat format{VertexAttributeFormat::Float};
    std::uint32_t register_index{};
};

enum class PicaAttributeMode : std::uint32_t { Retained, Fixed, Buffer };
struct PicaAttributeInput {
    VertexAttribute fetch{};
    PicaAttributeMode mode{PicaAttributeMode::Retained};
};

struct PicaVertexLayout {
    std::vector<VertexAttribute> attributes;
    std::array<std::uint32_t, 12> loader_offsets{};
    VertexRegisters fixed;
    std::array<PicaAttributeInput, 16> attribute_inputs{};
    VertexRegisters fixed_attributes{};
    std::array<std::uint32_t, 16> register_map{};
    std::uint32_t mapped_attribute_count{};
};

// Input slices begin at each loader's data_offset relative to the guest attribute base.
// Fixed and retained attributes are indexed by guest attribute. Retained registers preserve
// shader inputs outside max_input_attribute_index; mapped attributes overwrite them in order.
PicaVertexLayout BuildPicaVertexLayout(const Pica::RegsInternal& regs,
                                       const VertexRegisters& fixed_attributes,
                                       const VertexRegisters& retained_attributes = DefaultVertexRegisters(),
                                       const VertexRegisters& retained_registers = DefaultVertexRegisters());
PicaVertexLayout BuildPicaImmediateLayout(const Pica::RegsInternal& regs,
                                         const VertexRegisters& submitted_attributes);
std::array<std::uint64_t, 12> RequiredVertexInputBytes(const PicaVertexLayout& layout,
                                                     std::uint32_t vertex_count,
                                                     std::uint32_t first_vertex = 0);
std::array<std::uint64_t, 12> RequiredPicaStateInputBytes(const PicaVertexLayout& layout,
                                                        std::uint32_t vertex_count);

// Attribute and mapped-register banks are private GPU data. Each fetch snapshots the
// previous banks, then commits the last actual input load after all parallel readers finish.
class VertexInputState final {
    friend class VertexFetchRuntime;
    friend class InputReferenceRuntime;
    friend class RasterizerMetal;
    friend struct InputReferenceTestAccess;
public:
    explicit VertexInputState(const Device& device);
    VertexInputState(const VertexInputState&) = delete;
    VertexInputState& operator=(const VertexInputState&) = delete;
    void Reset() { needs_reset = true; references.reset(); }
private:
    mutable id<MTLBuffer> history;
    mutable std::shared_ptr<void> history_lease{std::make_shared<std::byte>()};
    mutable std::shared_ptr<const InputProvenance> references;
    bool needs_reset{true};
};
struct VertexInputSnapshot {
    std::array<std::byte, 512> history{};
    bool needs_reset{true};
};

// Raw attributes are decoded on the GPU to sixteen Float4 registers per vertex. Integer
// attributes are unnormalized; missing components are (0,0,0,1). Fixed/default registers
// are supplied explicitly. Descriptors are applied in order, including register aliases.
class VertexFetchRuntime final {
public:
    VertexFetchRuntime(Device& device, Scheduler& scheduler);
    ~VertexFetchRuntime();
    BufferSlice Fetch(std::span<const BufferSlice> inputs, std::span<const VertexAttribute> attributes,
                      std::uint32_t vertex_count, std::uint32_t first_vertex = 0,
                      const VertexRegisters& fixed = DefaultVertexRegisters());
    BufferSlice FetchPica(VertexInputState& state, std::span<const BufferSlice> inputs,
                          const PicaVertexLayout& layout, std::uint32_t vertex_count,
                          std::uint32_t element_count, std::uint32_t vertex_base,
                          std::uint32_t first_vertex, const BufferSlice& indices = {},
                          std::uint32_t index_size = 0);
    VertexInputSnapshot SaveState(const VertexInputState& state);
    void LoadState(VertexInputState& state, const VertexInputSnapshot& snapshot);

private:
    friend class ShaderMachineRuntime;
    void ResolveReferences(const VertexInputState& state);
    std::unique_ptr<InputReferenceRuntime> reference_runtime;
    struct PreparedPicaFetch {
        // Source slices are borrowed for the synchronous public call. Its caller
        // retains their leases until encoding; prepared storage owns every upload.
        BufferSlice output, attributes, params, update_params;
        std::array<BufferBinding, 20> bindings{};
        std::array<BufferBinding, 7> update_bindings{};
        std::uint32_t vertex_count{};
        bool reset{};
    };
    PreparedPicaFetch PreparePica(VertexInputState& state, std::span<const BufferSlice> inputs,
                          const PicaVertexLayout& layout, std::uint32_t vertex_count,
                          std::uint32_t element_count, std::uint32_t vertex_base,
                          std::uint32_t first_vertex, const BufferSlice& indices,
                          std::uint32_t index_size,const BufferSlice& owned_output={});
    void EncodePica(VertexInputState& state,const PreparedPicaFetch& prepared,
                    id<MTL4ComputeCommandEncoder> __strong* continuation=nullptr,
                    const BufferSlice& fifo_plan={},const BufferSlice& status={},
                    id<MTL4ComputeCommandEncoder> existing_encoder=nil);
    std::array<ComputeDispatch,2> PreparePicaDispatches(VertexInputState& state,
                    const PreparedPicaFetch& prepared,const BufferSlice& fifo_plan={},
                    const BufferSlice& status={});
    void CommitPica(VertexInputState& state) { state.needs_reset=false; }
    Device& device;
    Scheduler& scheduler;
    StreamBuffer output_stream;
    StreamBuffer parameter_stream;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipeline;
    id<MTLComputePipelineState> pica_pipeline;
    id<MTLComputePipelineState> history_pipeline;
    id<MTLBuffer> empty_input;
    // Each complete buffer-only layout owns a separate table. Dispatch commands
    // snapshot bindings before the next call rebinds any of these tables.
    id<MTL4ArgumentTable> fetch_arguments;
    id<MTL4ArgumentTable> pica_arguments;
    id<MTL4ArgumentTable> history_arguments;
};

} // namespace Metal
