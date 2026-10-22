// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include "mtl_vertex_fetch.h"

namespace Metal {

// Pointer-free CPU representation of the std430 physical-reference ABI. Values
// are decoded on the GPU; constant payloads are copied without float arithmetic.
struct alignas(16) InputReferenceGPU {
    std::uint64_t address{};
    std::uint32_t stride{}, format{}, components{4}, kind{}, reserved0{}, reserved1{};
    std::array<std::uint32_t, 4> fixed{};
};
static_assert(sizeof(InputReferenceGPU) == 48 && offsetof(InputReferenceGPU, fixed) == 32);

struct InputReference {
    InputReferenceGPU gpu;
    id<MTLBuffer> storage{};
    std::shared_ptr<void> lease;
    static InputReference Constant(const std::array<float, 4>& value);
    InputReference Frozen(std::uint32_t vertex) const;
};

struct InputProvenance {
    std::array<InputReference, 16> attributes{}, registers{};
    bool needs_reset{true};
};

struct PreparedInputReferences {
    std::array<InputReference, 16> current_registers{};
    std::shared_ptr<const InputProvenance> next;
};

// Only the rasterizer may publish a snapshot of its own immutable guest uploads.
// The public borrowed-buffer shader-machine entry points retain their copy contract.
class NativeVertexInputSnapshot final {
    friend class RasterizerMetal;
    friend class ShaderMachineRuntime;
    friend class DrawRuntime;
    friend class InputReferenceRuntime;
    friend struct InputReferenceTestAccess;
    NativeVertexInputSnapshot(std::array<InputReference, 16> registers_, BufferSlice indices_,
                             id<MTLDevice> validated_device_ = nil, std::uint32_t validated_count_ = 0)
        : registers{std::move(registers_)}, indices{std::move(indices_)},
          validated_device{validated_device_}, validated_count{validated_count_} {}
    const std::array<InputReference, 16> registers;
    const BufferSlice indices;
    // Only PrepareNative publishes these after checking every immutable record.
    // Buffer device/address/length and this snapshot's records cannot change.
    id<MTLDevice> const validated_device;
    const std::uint32_t validated_count;
};

struct PreparedNativeInputReferences {
    std::shared_ptr<const NativeVertexInputSnapshot> snapshot;
    std::shared_ptr<const InputProvenance> next;
    BufferSlice metadata;
};

class InputReferenceRuntime final {
public:
    InputReferenceRuntime(Device& device, Scheduler& scheduler);
    ~InputReferenceRuntime();
    InputProvenance Current(const VertexInputState& state) const;
    PreparedInputReferences Prepare(const InputProvenance& old, std::span<const BufferSlice> inputs,
        const PicaVertexLayout& layout, std::uint32_t vertex_count, std::uint32_t element_count,
        std::uint32_t vertex_base, std::uint32_t first_vertex,
        std::span<const std::byte> indices, std::uint32_t index_size) const;
    void Validate(std::span<const InputReference> references, std::uint32_t vertex_count) const;
    void UseSources(std::span<const InputReference> references) const;
    BufferSlice Upload(std::span<const InputReference> references);
    BufferSlice Materialize(std::span<const InputReference> references, std::uint32_t vertex_count);
    void Resolve(const VertexInputState& state);

private:
    friend class RasterizerMetal;
    friend struct InputReferenceTestAccess;
    PreparedNativeInputReferences PrepareNative(const VertexInputState& state,
        std::span<const BufferSlice> inputs, const PicaVertexLayout& layout,
        std::uint32_t vertex_count, std::uint32_t element_count, std::uint32_t vertex_base,
        std::uint32_t first_vertex, const BufferSlice& indices, std::uint32_t index_size);
    BufferSlice UploadValidated(std::span<const InputReference> references);
    Device& device;
    Scheduler& scheduler;
    StreamBuffer parameters;
    std::shared_future<id<MTLComputePipelineState>> materializer;
    struct IndexStats;
    std::unique_ptr<IndexStats> index_stats;
};
} // namespace Metal
