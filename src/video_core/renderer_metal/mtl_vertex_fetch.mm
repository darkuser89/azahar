// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_vertex_fetch.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include "mtl_input_reference.h"
#include "video_core/pica/regs_internal.h"
#include <algorithm>
#include <cstring>
#include <utility>
#include <limits>
#include <optional>
#include <stdexcept>

namespace Metal {

namespace {
struct FetchAttribute {
    std::uint32_t buffer_index;
    std::uint32_t offset;
    std::uint32_t stride;
    std::uint32_t components;
    std::uint32_t format;
    std::uint32_t register_index;
    std::array<std::uint32_t, 2> padding{};
};
struct alignas(16) FetchParams {
    std::uint32_t vertex_count;
    std::uint32_t first_vertex;
    std::uint32_t attribute_count;
    std::uint32_t padding{};
    std::array<FetchAttribute, 16> attributes{};
    VertexRegisters fixed{};
};
static_assert(sizeof(FetchAttribute) == 32);
static_assert(offsetof(FetchParams, attributes) == 16);
static_assert(offsetof(FetchParams, fixed) == 528);
static_assert(sizeof(FetchParams) == 784);
struct alignas(16) PicaFetchParams {
    std::uint32_t vertex_count, mapped_count, reset, padding{};
    std::array<FetchAttribute, 16> attributes{};
    std::array<std::uint32_t, 16> register_map{};
    VertexRegisters fixed{};
};
static_assert(sizeof(PicaFetchParams) == 848);
static_assert(offsetof(PicaFetchParams, fixed) == 592);
struct HistoryParams {
    std::uint32_t element_count, vertex_count, vertex_base, first_vertex;
    std::uint32_t index_size, reset, emit_plan{}, padding1{};
};
static_assert(sizeof(HistoryParams) == 32);
}

PicaVertexLayout BuildPicaVertexLayout(const Pica::RegsInternal& regs,
                                       const VertexRegisters& fixed_attributes,
                                       const VertexRegisters& retained_attributes,
                                       const VertexRegisters& retained_registers) {
    const auto& config = regs.pipeline.vertex_attributes;
    PicaVertexLayout result{.attributes = {}, .loader_offsets = {}, .fixed = retained_registers};
    result.fixed_attributes = fixed_attributes;
    result.mapped_attribute_count = regs.vs.max_input_attribute_index.Value() + 1;
    std::array<std::optional<VertexAttribute>, 12> by_attribute{};
    for (unsigned loader_index = 0; loader_index < 12; ++loader_index) {
        const auto& loader = config.attribute_loaders[loader_index];
        result.loader_offsets[loader_index] = loader.data_offset;
        if (loader.component_count > 12) throw std::invalid_argument("PICA vertex loader has more than twelve components");
        unsigned offset{};
        for (unsigned component = 0; component < loader.component_count; ++component) {
            const auto attribute_index = loader.GetComponent(component);
            if (attribute_index >= 12) {
                offset = (offset + 3U) & ~3U;
                offset += (attribute_index - 11) * 4;
                continue;
            }
            const auto element_size = config.GetElementSizeInBytes(attribute_index);
            offset = (offset + element_size - 1) & ~(element_size - 1);
            by_attribute[attribute_index] = VertexAttribute{
                loader_index, offset, loader.byte_count, config.GetNumElements(attribute_index),
                static_cast<VertexAttributeFormat>(config.GetFormat(attribute_index)),
                regs.vs.GetRegisterForAttribute(attribute_index)};
            offset += config.GetStride(attribute_index);
        }
    }
    for (unsigned attribute = 0; attribute < 16; ++attribute) {
        result.register_map[attribute] = regs.vs.GetRegisterForAttribute(attribute);
        if (attribute >= config.GetNumTotalAttributes()) continue;
        if (attribute < 12 && by_attribute[attribute]) {
            result.attribute_inputs[attribute] = {*by_attribute[attribute], PicaAttributeMode::Buffer};
        } else if (config.IsDefaultAttribute(attribute)) {
            result.attribute_inputs[attribute].mode = PicaAttributeMode::Fixed;
        }
    }
    std::array<std::optional<VertexAttribute>, 16> by_register{};
    for (unsigned attribute = 0; attribute <= regs.vs.max_input_attribute_index; ++attribute) {
        const auto reg = regs.vs.GetRegisterForAttribute(attribute);
        if (attribute >= config.GetNumTotalAttributes()) {
            result.fixed[reg] = retained_attributes[attribute];
            by_register[reg].reset();
        } else if (attribute < 12 && by_attribute[attribute]) {
            // Hardware arrays override a fixed-attribute bit for the same attribute.
            by_register[reg] = by_attribute[attribute];
        } else if (config.IsDefaultAttribute(attribute)) {
            result.fixed[reg] = fixed_attributes[attribute];
            by_register[reg].reset();
        } else {
            result.fixed[reg] = retained_attributes[attribute];
            by_register[reg].reset();
        }
        // Later attributes overwrite an aliased shader register, including retained values.
    }
    for (const auto& attribute : by_register) if (attribute) result.attributes.push_back(*attribute);
    return result;
}

PicaVertexLayout BuildPicaImmediateLayout(const Pica::RegsInternal& regs,
                                         const VertexRegisters& submitted_attributes) {
    PicaVertexLayout result{};
    result.fixed_attributes = submitted_attributes;
    result.mapped_attribute_count = regs.vs.max_input_attribute_index.Value() + 1;
    for (unsigned attribute = 0; attribute < 16; ++attribute) {
        result.register_map[attribute] = regs.vs.GetRegisterForAttribute(attribute);
        if (attribute <= regs.pipeline.max_input_attrib_index)
            result.attribute_inputs[attribute].mode = PicaAttributeMode::Fixed;
    }
    return result;
}

std::array<std::uint64_t, 12> RequiredPicaStateInputBytes(const PicaVertexLayout& layout,
                                                       std::uint32_t vertex_count) {
    PicaVertexLayout all{};
    for (const auto& attribute : layout.attribute_inputs)
        if (attribute.mode == PicaAttributeMode::Buffer) all.attributes.push_back(attribute.fetch);
    return RequiredVertexInputBytes(all, vertex_count);
}

std::array<std::uint64_t, 12> RequiredVertexInputBytes(const PicaVertexLayout& layout,
                                                     std::uint32_t vertex_count,
                                                     std::uint32_t first_vertex) {
    std::array<std::uint64_t, 12> result{};
    if (!vertex_count) return result;
    if (first_vertex > std::numeric_limits<std::uint32_t>::max() - (vertex_count - 1)) {
        throw std::invalid_argument("PICA vertex input range overflows its index representation");
    }
    const auto last = std::uint64_t(first_vertex) + vertex_count - 1;
    for (const auto& attribute : layout.attributes) {
        if (attribute.buffer_index >= 12 || attribute.register_index >= 16 ||
            !attribute.components || attribute.components > 4 || attribute.format > VertexAttributeFormat::Float) {
            throw std::invalid_argument("Invalid PICA vertex input layout");
        }
        const auto bytes = attribute.format == VertexAttributeFormat::Float ? 4U
                           : attribute.format == VertexAttributeFormat::Short ? 2U : 1U;
        const auto required = std::uint64_t(attribute.offset) + last * attribute.stride + attribute.components * bytes;
        result[attribute.buffer_index] = std::max(result[attribute.buffer_index], required);
    }
    return result;
}

VertexInputState::VertexInputState(const Device& device) : history{device.CreateBuffer(2 * sizeof(VertexRegisters), false)} {}

VertexFetchRuntime::~VertexFetchRuntime() = default;

void VertexFetchRuntime::ResolveReferences(const VertexInputState& state) {
    if (state.references) {
        if (!reference_runtime) reference_runtime = std::make_unique<InputReferenceRuntime>(device, scheduler);
        reference_runtime->Resolve(state);
    }
    // A native snapshot may still read the old GPU history. Precise fetches
    // update their state in place, so pinning requires a separate destination.
    if (state.history_lease.use_count() > 1) {
        auto replacement = device.CreateBuffer(512, false);
        if (!state.needs_reset)
            TransferRuntime{device, scheduler}.CopyBuffer(state.history, 0, replacement, 0, 512);
        state.history = replacement;
        state.history_lease = std::make_shared<std::byte>();
    }
}

VertexFetchRuntime::VertexFetchRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, output_stream{device, scheduler, 64 << 20},
      parameter_stream{device, scheduler, 1 << 20} {
    library = device.CompileLibrary(VertexFetchShaderSource, "Azahar Metal 4 PICA vertex fetch");
    MTL4ComputePipelineDescriptor* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Device::Function(library, "fetch_vertices");
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    pipeline = device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor = Device::Function(library, "fetch_pica_vertices");
    descriptor.label = @"fetch_pica_vertices + update_pica_input_history";
    pica_pipeline = device.CompileCompute(descriptor);
    descriptor.label = nil;
    descriptor.computeFunctionDescriptor = Device::Function(library, "update_pica_input_history");
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    history_pipeline = device.CompileCompute(descriptor);
    if (history_pipeline.threadExecutionWidth < 16 || history_pipeline.threadExecutionWidth > 64)
        throw std::runtime_error("Native input-history FIFO requires a SIMD width between 16 and 64");
    empty_input = device.CreateBuffer(16, true);
    std::memset(empty_input.contents, 0, 16);
}

BufferSlice VertexFetchRuntime::Fetch(std::span<const BufferSlice> inputs,
                                     std::span<const VertexAttribute> attributes,
                                     std::uint32_t vertex_count, std::uint32_t first_vertex,
                                     const VertexRegisters& fixed) {
    if (inputs.size() > 16 || attributes.size() > 16 ||
        vertex_count > std::numeric_limits<std::uint32_t>::max() / 16 ||
        (vertex_count && first_vertex > std::numeric_limits<std::uint32_t>::max() - (vertex_count - 1))) {
        throw std::invalid_argument("Invalid native vertex-fetch dimensions or register count");
    }
    FetchParams info{.vertex_count = vertex_count, .first_vertex = first_vertex,
                     .attribute_count = std::uint32_t(attributes.size()), .fixed = fixed};
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        const auto& attr = attributes[i];
        if (attr.buffer_index >= inputs.size() || attr.register_index >= 16 ||
            !attr.components || attr.components > 4 || attr.format > VertexAttributeFormat::Float) {
            throw std::invalid_argument("Invalid native vertex attribute descriptor");
        }
        const auto& input = inputs[attr.buffer_index];
        if (!input.buffer || input.buffer.device != device.Handle() || input.offset > input.buffer.length ||
            input.size > input.buffer.length - input.offset) {
            throw std::invalid_argument("Invalid native vertex input slice");
        }
        const auto bytes = attr.format == VertexAttributeFormat::Float ? 4U
                           : attr.format == VertexAttributeFormat::Short ? 2U : 1U;
        const auto required = std::uint64_t(attr.offset) + std::uint64_t(attr.components) * bytes;
        if (vertex_count) {
            const auto last = std::uint64_t(first_vertex) + vertex_count - 1;
            if (required > input.size || last * attr.stride > input.size - required) {
                throw std::invalid_argument("Native vertex fetch exceeds its source slice");
            }
        }
        info.attributes[i] = {attr.buffer_index, attr.offset, attr.stride, attr.components,
                               std::uint32_t(attr.format), attr.register_index};
    }
    if (!vertex_count) return {};
    auto output = output_stream.Allocate(NSUInteger(vertex_count) * sizeof(VertexRegisters));
    auto parameters = parameter_stream.Allocate(sizeof(info));
    std::memcpy(parameters.data, &info, sizeof(info));
    std::array<BufferBinding, 18> bindings{};
    std::array<bool, 16> used{};
    for (const auto& attribute : attributes) used[attribute.buffer_index] = true;
    for (NSUInteger i = 0; i < 16; ++i) {
        bindings[i] = {used[i] ? inputs[i].buffer : empty_input, used[i] ? inputs[i].offset : 0, i};
        if (used[i] && inputs[i].buffer == output.buffer &&
            inputs[i].offset < output.offset + output.size && output.offset < inputs[i].offset + inputs[i].size) {
            throw std::invalid_argument("Native vertex-fetch input overlaps its output allocation");
        }
    }
    bindings[16] = {output.buffer, output.offset, 16};
    bindings[17] = {parameters.buffer, parameters.offset, 17};
    if (fetch_arguments) scheduler.RebindBufferArguments(fetch_arguments, bindings);
    else fetch_arguments = scheduler.MakeArguments(bindings);
    const auto table = fetch_arguments;
    scheduler.KeepAlive(pipeline);
    auto encoder = scheduler.BeginCompute(pipeline);
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(vertex_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    scheduler.EndEncoding();
    return output;
}

BufferSlice VertexFetchRuntime::FetchPica(VertexInputState& state, std::span<const BufferSlice> inputs,
                                         const PicaVertexLayout& layout, std::uint32_t vertex_count,
                                         std::uint32_t element_count, std::uint32_t vertex_base,
                                         std::uint32_t first_vertex, const BufferSlice& indices,
                                         std::uint32_t index_size) {
    if (!element_count) return {};
    auto prepared=PreparePica(state,inputs,layout,vertex_count,element_count,vertex_base,
                                   first_vertex,indices,index_size);
    EncodePica(state,prepared);
    scheduler.EndCompute();
    return std::move(prepared.output);
}
VertexFetchRuntime::PreparedPicaFetch VertexFetchRuntime::PreparePica(VertexInputState& state,
    std::span<const BufferSlice> inputs,const PicaVertexLayout& layout,std::uint32_t vertex_count,
    std::uint32_t element_count,std::uint32_t vertex_base,std::uint32_t first_vertex,
    const BufferSlice& indices,std::uint32_t index_size,const BufferSlice& owned_output) {
    if (!element_count || !vertex_count || vertex_count > UINT32_MAX / 16 || inputs.size() > 16 ||
        layout.mapped_attribute_count > 16 || state.history.device != device.Handle() ||
        std::uint64_t(vertex_base) + vertex_count > std::uint64_t(UINT32_MAX) + 1 ||
        (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) ||
        (index_size && element_count > UINT32_MAX / index_size) ||
        (!index_size && (first_vertex < vertex_base ||
            std::uint64_t(first_vertex) + element_count > std::uint64_t(vertex_base) + vertex_count)))
        throw std::invalid_argument("Invalid native PICA input-state dimensions or device");
    if (index_size && (!indices.buffer || indices.buffer.device != device.Handle() ||
        indices.offset > indices.buffer.length || indices.size > indices.buffer.length - indices.offset ||
        std::uint64_t(element_count) * index_size > indices.size))
        throw std::invalid_argument("Native PICA input-state index slice is too short or invalid");
    PicaFetchParams info{vertex_count, layout.mapped_attribute_count, state.needs_reset ? 1U : 0U};
    info.fixed = layout.fixed_attributes;
    info.register_map = layout.register_map;
    for (unsigned attribute = 0; attribute < info.register_map.size(); ++attribute)
        if (info.register_map[attribute] >= 16) throw std::invalid_argument("Invalid native PICA input register map");
    std::array<bool, 16> used{};
    for (unsigned attribute = 0; attribute < 16; ++attribute) {
        const auto& input = layout.attribute_inputs[attribute];
        if (input.mode > PicaAttributeMode::Buffer) throw std::invalid_argument("Invalid native PICA input source mode");
        const auto& fetch = input.fetch;
        if (input.mode == PicaAttributeMode::Buffer) {
            if (fetch.buffer_index >= inputs.size() || !fetch.components || fetch.components > 4 ||
                fetch.format > VertexAttributeFormat::Float)
                throw std::invalid_argument("Invalid native PICA stateful fetch descriptor");
            const auto& source = inputs[fetch.buffer_index];
            const auto bytes = fetch.format == VertexAttributeFormat::Float ? 4U : fetch.format == VertexAttributeFormat::Short ? 2U : 1U;
            const auto required = std::uint64_t(fetch.offset) + std::uint64_t(vertex_count - 1) * fetch.stride + fetch.components * bytes;
            if (!source.buffer || source.buffer.device != device.Handle() || source.offset > source.buffer.length ||
                source.size > source.buffer.length - source.offset || required > source.size)
                throw std::invalid_argument("Native stateful attribute fetch exceeds its physical source slice");
            used[fetch.buffer_index] = true;
        }
        info.attributes[attribute] = {fetch.buffer_index, fetch.offset, fetch.stride, fetch.components,
                                      std::uint32_t(fetch.format), std::uint32_t(input.mode)};
    }
    const auto output_bytes=NSUInteger(vertex_count)*sizeof(VertexRegisters);
    if(owned_output.buffer && (owned_output.buffer.device!=device.Handle() || owned_output.offset%16 ||
        owned_output.offset>owned_output.buffer.length || owned_output.size>owned_output.buffer.length-owned_output.offset ||
        output_bytes>owned_output.size || owned_output.buffer==state.history))
        throw std::invalid_argument("Invalid owned native vertex-fetch destination");
    ResolveReferences(state);
    PreparedPicaFetch prepared{};
    prepared.output=owned_output.buffer ? owned_output:output_stream.Allocate(output_bytes);
    prepared.output.size=output_bytes;
    prepared.attributes=output_stream.Allocate(NSUInteger(vertex_count)*sizeof(VertexRegisters));
    const auto& output=prepared.output;
    const auto& attributes=prepared.attributes;
    if (index_size) for (const auto& dest : {output, attributes})
        if (indices.buffer == dest.buffer && indices.offset < dest.offset + dest.size &&
            dest.offset < indices.offset + std::uint64_t(element_count) * index_size)
            throw std::invalid_argument("Native stateful index data overlaps its fetch output");
    prepared.params=parameter_stream.Allocate(sizeof(info));
    const auto& params=prepared.params;
    std::memcpy(params.data,&info,sizeof(info));
    auto& bindings=prepared.bindings;
    for (NSUInteger index = 0; index < 16; ++index) {
        bindings[index] = {used[index] ? inputs[index].buffer : empty_input, used[index] ? inputs[index].offset : 0, index};
        if (!used[index]) continue;
        const auto& source = inputs[index];
        for (const auto& dest : {output, attributes})
            if (source.buffer == dest.buffer && source.offset < dest.offset + dest.size && dest.offset < source.offset + source.size)
                throw std::invalid_argument("Native stateful vertex input overlaps its fetch output");
    }
    bindings[16] = {output.buffer, output.offset, 16};
    bindings[17] = {params.buffer, params.offset, 17};
    bindings[18] = {state.history, 0, 18};
    bindings[19] = {attributes.buffer, attributes.offset, 19};
    // Prepare both uploads before opening an encoder or publishing argument
    // bindings: upload-page recycling can submit/wait and change the queue slot.
    const HistoryParams update{element_count, vertex_count, vertex_base, first_vertex, index_size, info.reset};
    prepared.update_params=parameter_stream.Allocate(sizeof(update));
    const auto& update_params=prepared.update_params;
    std::memcpy(update_params.data,&update,sizeof(update));
    auto& update_bindings=prepared.update_bindings;
    update_bindings[0]={output.buffer,output.offset,16};
    update_bindings[1]={update_params.buffer,update_params.offset,17};
    update_bindings[2]={state.history,0,18};
    update_bindings[3]={attributes.buffer,attributes.offset,19};
    update_bindings[4]={index_size ? indices.buffer:empty_input,index_size ? indices.offset:0,20};
    update_bindings[5]={empty_input,0,21};
    update_bindings[6]={empty_input,0,22};
    prepared.vertex_count=vertex_count;prepared.reset=info.reset!=0;
    return prepared;
}
std::array<ComputeDispatch,2> VertexFetchRuntime::PreparePicaDispatches(VertexInputState& state,
    const PreparedPicaFetch& prepared,const BufferSlice& fifo_plan,const BufferSlice& status) {
    if(state.history!=prepared.bindings[18].buffer || state.needs_reset!=prepared.reset)
        throw std::invalid_argument("Prepared PICA fetch state changed before encoding");
    const auto& bindings=prepared.bindings;
    auto update_bindings=prepared.update_bindings;
    HistoryParams history_info{};
    std::memcpy(&history_info,prepared.update_params.data,sizeof(history_info));
    if(fifo_plan.buffer) {
        if(!history_info.index_size || fifo_plan.buffer.device!=device.Handle() || fifo_plan.offset%8 ||
           fifo_plan.offset>fifo_plan.buffer.length || fifo_plan.size>fifo_plan.buffer.length-fifo_plan.offset ||
           (NSUInteger(history_info.element_count)+1)*8>fifo_plan.size ||
           !status.buffer || status.buffer.device!=device.Handle() || status.offset%16 ||
           status.offset>status.buffer.length || status.size>status.buffer.length-status.offset || status.size<16)
            throw std::invalid_argument("Invalid shared native FIFO plan/status slice");
        history_info.emit_plan=1;
        update_bindings[5]={fifo_plan.buffer,fifo_plan.offset,21};
        update_bindings[6]={status.buffer,status.offset,22};
    }
    std::memcpy(prepared.update_params.data,&history_info,sizeof(history_info));
    std::array<ComputeDispatch,2> commands{};
    commands[0].pipeline=pica_pipeline;
    commands[0].buffers={bindings.begin(),bindings.end()};
    commands[0].threads=MTLSizeMake(prepared.vertex_count,1,1);
    commands[0].group=MTLSizeMake(64,1,1);
    commands[1].pipeline=history_pipeline;
    commands[1].buffers={update_bindings.begin(),update_bindings.end()};
    commands[1].simd_groups=1;
    return commands;
}
void VertexFetchRuntime::EncodePica(VertexInputState& state,const PreparedPicaFetch& prepared,
    id<MTL4ComputeCommandEncoder> __strong* continuation,
    const BufferSlice& fifo_plan,const BufferSlice& status,id<MTL4ComputeCommandEncoder> existing_encoder) {
    const auto commands=PreparePicaDispatches(state,prepared,fifo_plan,status);
    const auto& bindings=commands[0].buffers;
    const auto& update_bindings=commands[1].buffers;
    const auto vertex_count=prepared.vertex_count;
    if (pica_arguments) scheduler.RebindBufferArguments(pica_arguments, bindings);
    else pica_arguments = scheduler.MakeArguments(bindings);
    const auto table = pica_arguments;
    if (history_arguments) scheduler.RebindBufferArguments(history_arguments, update_bindings);
    else history_arguments = scheduler.MakeArguments(update_bindings);
    const auto update_table = history_arguments;
    scheduler.KeepAlive(pica_pipeline);
    scheduler.KeepAlive(history_pipeline);
    const auto encoder = existing_encoder ? existing_encoder:scheduler.BeginCompute(pica_pipeline);
    [encoder setComputePipelineState:pica_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(vertex_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    // Fetch reads the old history and writes both vertex banks. Retire all those
    // device accesses before copying the final FIFO miss back into history.
    [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setComputePipelineState:history_pipeline];
    [encoder setArgumentTable:update_table];
    const auto history_width = history_pipeline.threadExecutionWidth;
    [encoder dispatchThreads:MTLSizeMake(history_width, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(history_width, 1, 1)];
    state.needs_reset = false;
    if(continuation) *continuation=encoder;
}

VertexInputSnapshot VertexFetchRuntime::SaveState(const VertexInputState& state) {
    if (state.history.device != device.Handle()) throw std::invalid_argument("Vertex snapshot belongs to another Metal device");
    if (state.references) {
        if (!reference_runtime) reference_runtime = std::make_unique<InputReferenceRuntime>(device, scheduler);
        reference_runtime->Resolve(state);
    }
    VertexInputSnapshot result{.needs_reset = state.needs_reset};
    if (!state.needs_reset) {
        const auto readback = device.CreateBuffer(result.history.size(), true);
        TransferRuntime{device, scheduler}.CopyBuffer(state.history, 0, readback, 0, result.history.size());
        scheduler.Finish(Scheduler::WaitReason::Readback);
        std::memcpy(result.history.data(), readback.contents, result.history.size());
    }
    return result;
}

void VertexFetchRuntime::LoadState(VertexInputState& state, const VertexInputSnapshot& snapshot) {
    if (state.history.device != device.Handle()) throw std::invalid_argument("Vertex snapshot target belongs to another Metal device");
    if (snapshot.needs_reset && std::any_of(snapshot.history.begin(), snapshot.history.end(),
                                          [](std::byte value) { return value != std::byte{}; }))
        throw std::invalid_argument("A reset native vertex snapshot contains noncanonical history");
    const auto upload = device.CreateBuffer(snapshot.history.size(), true);
    std::memcpy(upload.contents, snapshot.history.data(), snapshot.history.size());
    const bool replace = state.references || state.history_lease.use_count() > 1;
    const auto destination = replace ? device.CreateBuffer(512, false) : state.history;
    TransferRuntime{device, scheduler}.CopyBuffer(upload, 0, destination, 0, snapshot.history.size());
    if (replace) {
        state.history = destination;
        state.history_lease = std::make_shared<std::byte>();
    }
    state.references.reset();
    state.needs_reset = snapshot.needs_reset;
}

} // namespace Metal
