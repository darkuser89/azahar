// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_input_reference.h"
#include "mtl_index_fifo.h"
#include "mtl_input_reference_shader.h"
#include "mtl_shader_translation.h"
#include "common/logging/log.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace Metal {

struct InputReferenceRuntime::IndexStats {
    struct Group { std::uint64_t calls{}, elements{}; };
    std::array<Group, 6> groups{};
};

InputReference InputReference::Constant(const std::array<float, 4>& value) {
    InputReference result{};
    std::memcpy(result.gpu.fixed.data(), value.data(), 16);
    return result;
}

InputReference InputReference::Frozen(std::uint32_t vertex) const {
    auto result = *this;
    if (result.gpu.kind) {
        result.gpu.address += std::uint64_t(vertex) * result.gpu.stride;
        result.gpu.stride = 0;
    }
    return result;
}

InputReferenceRuntime::InputReferenceRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, parameters{device, scheduler, 4 << 20} {
    std::string source{InputReferenceShaderGLSL};
    const auto count = source.find("input_refs[16]");
    if (count == std::string::npos) throw std::logic_error("Missing input-reference ABI");
    source.replace(count, std::string_view("input_refs[16]").size(), "input_refs[32]");
    source.insert(0, R"(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
layout(local_size_x=64) in;
)");
    source += R"(
layout(set=3,binding=2,std430) writeonly buffer MaterializedInputs {vec4 materialized_inputs[];};
layout(set=3,binding=3,std140) uniform MaterializeParams {uint count;uint registers;uint pad0;uint pad1;} materialize_params;
void main() {
 uint index=gl_GlobalInvocationID.x;
 if(index>=materialize_params.count)return;
 materialized_inputs[index]=mesh_load_register(index%materialize_params.registers,index/materialize_params.registers);
}
)";
    const auto bytes=source.size();
    materializer=device.DeferCompilation([native=&device,source=std::move(source)] {
        const auto translated=TranslateGuestGLSL(source,ShaderStage::Compute,RasterizerShaderBindings());
        const auto library=native->CompileLibrary(translated.source,"Azahar Metal 4 immutable input materialization");
        auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor=Device::Function(library,translated.entry_point);
        descriptor.maxTotalThreadsPerThreadgroup=64;
        return native->CompileCompute(descriptor);
    },bytes);
    const auto* profile = std::getenv("AZAHAR_METAL4_PROFILE");
    if (profile && *profile) index_stats = std::make_unique<IndexStats>();
}

InputReferenceRuntime::~InputReferenceRuntime() {
    if (!index_stats) return;
    for (unsigned n = 0; n < index_stats->groups.size(); ++n) {
        const auto& group = index_stats->groups[n];
        if (group.calls) LOG_INFO(Render,
            "Metal input index preparation: range={}, elements_per_draw={}, calls={}, elements={}",
            n / 2 == 0 ? "1..64" : n / 2 == 1 ? "65..256" : "257+",
            n % 2 == 0 ? "1..64" : "65+", group.calls, group.elements);
    }
}

InputProvenance InputReferenceRuntime::Current(const VertexInputState& state) const {
    (void)scheduler.LastUseTick(nil);
    if (state.history.device != device.Handle()) throw std::invalid_argument("Input reference state belongs to another device");
    if (state.references) return *state.references;
    InputProvenance result{};
    result.needs_reset = state.needs_reset;
    if (!state.needs_reset) {
        for (unsigned n = 0; n < 32; ++n) {
            InputReference ref{};
            ref.storage = state.history;
            ref.lease = state.history_lease;
            ref.gpu.address = state.history.gpuAddress + n * 16;
            ref.gpu.format = 3;
            ref.gpu.kind = 1;
            (n < 16 ? result.attributes[n] : result.registers[n - 16]) = std::move(ref);
        }
    }
    return result;
}

PreparedInputReferences InputReferenceRuntime::Prepare(const InputProvenance& old,
    std::span<const BufferSlice> inputs, const PicaVertexLayout& layout,
    std::uint32_t vertex_count, std::uint32_t element_count, std::uint32_t vertex_base,
    std::uint32_t first_vertex, std::span<const std::byte> indices, std::uint32_t index_size) const {
    (void)scheduler.LastUseTick(nil);
    if (!vertex_count || !element_count || layout.mapped_attribute_count > 16 || inputs.size() > 16 ||
        (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) ||
        std::uint64_t(vertex_base) + vertex_count > std::uint64_t(UINT32_MAX) + 1 ||
        (index_size && std::uint64_t(element_count) * index_size > indices.size()) ||
        (!index_size && std::uint64_t(first_vertex) + element_count > std::uint64_t(UINT32_MAX) + 1))
        throw std::invalid_argument("Invalid immutable input reference dimensions");
    PreparedInputReferences result{};
    auto attributes = old.attributes;
    result.current_registers = old.registers;
    if (old.needs_reset) { attributes = {}; result.current_registers = {}; }
    for (unsigned a = 0; a < 16; ++a) {
        const auto& input = layout.attribute_inputs[a];
        const auto& desc = input.fetch;
        if (unsigned(input.mode) > 2) throw std::invalid_argument("Invalid input reference mode");
        if (input.mode == PicaAttributeMode::Fixed) attributes[a] = InputReference::Constant(layout.fixed_attributes[a]);
        if (input.mode != PicaAttributeMode::Buffer) continue;
        if (desc.buffer_index >= inputs.size() || !desc.components || desc.components > 4 || unsigned(desc.format) > 3)
            throw std::invalid_argument("Invalid input reference descriptor");
        const auto& source = inputs[desc.buffer_index];
        const unsigned bytes = desc.format == VertexAttributeFormat::Float ? 4 : desc.format == VertexAttributeFormat::Short ? 2 : 1;
        const auto required = std::uint64_t(desc.offset) + std::uint64_t(vertex_count - 1) * desc.stride + desc.components * bytes;
        if (!source.buffer || source.buffer.device != device.Handle() || source.offset > source.buffer.length ||
            source.size > source.buffer.length - source.offset || required > source.size)
            throw std::invalid_argument("Input reference exceeds its immutable upload");
        auto& ref = attributes[a];
        ref = {};
        ref.storage = source.buffer;
        ref.lease = source.lease;
        ref.gpu.address = source.buffer.gpuAddress + source.offset + desc.offset;
        ref.gpu.stride = desc.stride;
        ref.gpu.format = unsigned(desc.format);
        ref.gpu.components = desc.components;
        ref.gpu.kind = 1;
    }
    for (unsigned a = 0; a < layout.mapped_attribute_count; ++a) {
        if (layout.register_map[a] >= 16) throw std::invalid_argument("Invalid input register reference");
        result.current_registers[layout.register_map[a]] = attributes[a];
    }
    Validate(result.current_registers, vertex_count);
    if (index_size && index_stats) {
        const unsigned group = (vertex_count <= 64 ? 0 : vertex_count <= 256 ? 2 : 4) +
                               (element_count > 64);
        ++index_stats->groups[group].calls;
        index_stats->groups[group].elements += element_count;
    }
    std::uint32_t last = 0;
    bool found = false;
    if (!index_size) {
        const auto id = first_vertex + element_count - 1;
        last = id - vertex_base;
        found = id >= vertex_base && last < vertex_count;
    } else if (vertex_count <= 256) {
        // Small relative IDs permit direct membership bits, including eviction
        // from the same FIFO. No attribute decoding or shader arithmetic moves
        // to the CPU. Larger ranges retain the general full-width ID table.
        std::array<std::uint64_t, 4> seen{};
        std::array<std::uint8_t, 64> order;
        unsigned size = 0, next = 0;
        for (std::uint32_t e = 0; e < element_count; ++e) {
            std::uint32_t id = 0;
            for (unsigned b = 0; b < index_size; ++b)
                id |= std::uint32_t(std::to_integer<unsigned char>(indices[std::size_t(e) * index_size + b])) << (b * 8);
            if (id < vertex_base || id - vertex_base >= vertex_count) continue;
            const auto local = id - vertex_base;
            const auto bit = std::uint64_t(1) << (local & 63);
            if (seen[local / 64] & bit) continue;
            if (size == order.size()) {
                const unsigned expired = order[next];
                seen[expired / 64] &= ~(std::uint64_t(1) << (expired & 63));
            } else {
                ++size;
            }
            order[next] = std::uint8_t(local);
            next = (next + 1) & 63;
            seen[local / 64] |= bit;
            last = local;
            found = true;
        }
    } else {
        // Identical 64-entry FIFO policy to the GPU fetch: hits do not move slots.
        // This is index bookkeeping, without reading or converting attribute values.
        IndexFifo fifo;
        for (std::uint32_t e = 0; e < element_count; ++e) {
            std::uint32_t id = 0;
            for (unsigned b = 0; b < index_size; ++b)
                id |= std::uint32_t(std::to_integer<unsigned char>(indices[std::size_t(e) * index_size + b])) << (b * 8);
            if (id < vertex_base || id - vertex_base >= vertex_count) continue;
            if (!fifo.Miss(id)) continue;
            last = id - vertex_base;
            found = true;
        }
    }
    auto next = std::make_shared<InputProvenance>();
    if (found) {
        for (unsigned a = 0; a < 16; ++a) {
            next->attributes[a] = attributes[a].Frozen(last);
            next->registers[a] = result.current_registers[a].Frozen(last);
        }
    } else if (!old.needs_reset) *next = old;
    next->needs_reset = false;
    result.next = std::move(next);
    return result;
}

void InputReferenceRuntime::Validate(std::span<const InputReference> refs, std::uint32_t count) const {
    (void)scheduler.LastUseTick(nil);
    if (!count || refs.empty() || refs.size() > 32) throw std::invalid_argument("Invalid input reference extent");
    for (const auto& ref : refs) {
        const auto& data = ref.gpu;
        if (data.kind > 1 || data.format > 3 || !data.components || data.components > 4 || data.reserved0 || data.reserved1)
            throw std::invalid_argument("Invalid GPU input reference ABI");
        if (!data.kind) continue;
        if (!ref.storage || ref.storage.device != device.Handle() || data.address < ref.storage.gpuAddress)
            throw std::invalid_argument("Invalid GPU input reference storage");
        const auto offset = data.address - ref.storage.gpuAddress;
        const auto stride_bytes = std::uint64_t(count - 1) * data.stride;
        const auto bytes = data.components * (data.format == 3 ? 4u : data.format == 2 ? 2u : 1u);
        if (offset > ref.storage.length || stride_bytes > ref.storage.length - offset ||
            bytes > ref.storage.length - offset - stride_bytes)
            throw std::invalid_argument("GPU input reference exceeds storage");
    }
}

void InputReferenceRuntime::UseSources(std::span<const InputReference> refs) const {
    (void)scheduler.LastUseTick(nil);
    for (const auto& ref : refs) if (ref.gpu.kind) scheduler.UseResource(ref.storage);
}

BufferSlice InputReferenceRuntime::Upload(std::span<const InputReference> refs) {
    Validate(refs, 1);
    return UploadValidated(refs);
}

BufferSlice InputReferenceRuntime::UploadValidated(std::span<const InputReference> refs) {
    auto result = parameters.Allocate(refs.size() * sizeof(InputReferenceGPU));
    auto* gpu = static_cast<InputReferenceGPU*>(result.data);
    for (std::size_t n = 0; n < refs.size(); ++n) gpu[n] = refs[n].gpu;
    return result;
}

PreparedNativeInputReferences InputReferenceRuntime::PrepareNative(const VertexInputState& state,
    std::span<const BufferSlice> inputs, const PicaVertexLayout& layout,
    std::uint32_t vertex_count, std::uint32_t element_count, std::uint32_t vertex_base,
    std::uint32_t first_vertex, const BufferSlice& indices, std::uint32_t index_size) {
    (void)scheduler.LastUseTick(nil);
    if (state.history.device != device.Handle())
        throw std::invalid_argument("Input reference state belongs to another device");
    if (index_size && (!indices.buffer || indices.buffer.device != device.Handle() || !indices.data ||
        indices.buffer.storageMode != MTLStorageModeShared ||
        indices.offset > indices.buffer.length || indices.size > indices.buffer.length - indices.offset ||
        std::uint64_t(element_count) * index_size > indices.size ||
        indices.data != static_cast<std::byte*>(indices.buffer.contents) + indices.offset))
        throw std::invalid_argument("Invalid native immutable input indices");
    if (!index_size && (first_vertex < vertex_base ||
        std::uint64_t(first_vertex) + element_count > std::uint64_t(vertex_base) + vertex_count))
        throw std::invalid_argument("Native input elements exceed their immutable vertex range");
    // Keep an existing immutable provenance object alive instead of copying its
    // 32 retained records. A materialized history needs a fresh description only.
    const auto retained = state.references;
    std::optional<InputProvenance> fallback;
    if (!retained) fallback.emplace(Current(state));
    auto prepared = Prepare(retained ? *retained : *fallback, inputs, layout, vertex_count,
        element_count, vertex_base, first_vertex, index_size ? std::span<const std::byte>{
            static_cast<const std::byte*>(indices.data), NSUInteger(element_count) * index_size}
            : std::span<const std::byte>{}, index_size);
    PreparedNativeInputReferences result;
    result.snapshot = std::shared_ptr<const NativeVertexInputSnapshot>(new NativeVertexInputSnapshot(
        std::move(prepared.current_registers), indices, device.Handle(), vertex_count));
    result.next = std::move(prepared.next);
    result.metadata = UploadValidated(result.snapshot->registers);
    return result;
}

BufferSlice InputReferenceRuntime::Materialize(std::span<const InputReference> refs, std::uint32_t count) {
    if ((refs.size() != 16 && refs.size() != 32) || (refs.size() == 32 && count != 1) || count > UINT32_MAX / refs.size())
        throw std::invalid_argument("Invalid input reference materialization");
    Validate(refs, count);
    std::array<InputReference, 32> padded{};
    std::copy(refs.begin(), refs.end(), padded.begin());
    // The original records were checked for the full extent above; padding is
    // value-initialized constant input. Publishing cannot introduce new sources.
    auto metadata = UploadValidated(padded);
    auto info = parameters.Allocate(16);
    const auto vectors = std::uint32_t(count * refs.size());
    const std::array<std::uint32_t, 4> dimensions{vectors, std::uint32_t(refs.size()), 0, 0};
    std::memcpy(info.data, dimensions.data(), 16);
    auto output = device.CreateBuffer(NSUInteger(vectors) * 16, false);
    UseSources(refs);
    ComputeDispatch command;
    command.pending_pipeline=materializer;
    command.buffers={{metadata.buffer,metadata.offset,17},{output,0,18},{info.buffer,info.offset,19}};
    command.threads=MTLSizeMake(vectors,1,1);command.group=MTLSizeMake(64,1,1);
    scheduler.QueueCompute(std::span{&command,1});
    return {output, 0, NSUInteger(vectors) * 16, nullptr, std::make_shared<std::byte>()};
}

void InputReferenceRuntime::Resolve(const VertexInputState& state) {
    (void)scheduler.LastUseTick(nil);
    if (state.history.device != device.Handle()) throw std::invalid_argument("Input reference state belongs to another device");
    if (!state.references) return;
    if (state.needs_reset) { state.references.reset(); return; }
    auto references = state.references;
    std::array<InputReference, 32> values{};
    std::copy(references->attributes.begin(), references->attributes.end(), values.begin());
    std::copy(references->registers.begin(), references->registers.end(), values.begin() + 16);
    auto output = Materialize(values, 1);
    state.history = output.buffer;
    state.history_lease = std::move(output.lease);
    state.references.reset();
}
} // namespace Metal
