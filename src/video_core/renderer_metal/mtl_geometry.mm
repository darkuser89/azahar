// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_geometry.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace Metal {

namespace {

void ValidateSlice(const BufferSlice& slice, NSUInteger required_size, NSUInteger alignment = 16) {
    if (!slice.buffer || slice.offset > slice.buffer.length ||
        slice.size > slice.buffer.length - slice.offset || required_size > slice.size ||
        slice.offset % alignment) {
        throw std::invalid_argument("Native geometry buffer is missing, misaligned or too small");
    }
}

} // namespace

PrimitiveStream::PrimitiveStream(const Device& device) : history{device.CreateBuffer(544, false)} {}

GeometryRuntime::GeometryRuntime(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, output_stream{device, scheduler, 64 << 20},
      parameter_stream{device, scheduler, 1 << 20} {
    library = device.CompileLibrary(GeometryShaderSource, "Azahar Metal 4 primitive assembly");
    MTL4ComputePipelineDescriptor* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Device::Function(library, "assemble_triangles");
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    assembly_pipeline = device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor = Device::Function(library, "append_primitive_stream");
    descriptor.label = @"append_primitive_stream with disjoint history";
    stream_pipeline = device.CompileCompute(descriptor);
    empty_indices = device.CreateBuffer(4, true);
    std::memset(empty_indices.contents, 0, 4);
}

VertexCaptureProgram GeometryRuntime::CompileVertex(std::string_view source) {
    auto shader = TranslateGuestGLSL(source, ShaderStage::VertexCompute, RasterizerShaderBindings());
    auto program_library = device.CompileLibrary(shader.source, "Azahar Metal 4 PICA vertex program");
    MTL4ComputePipelineDescriptor* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Device::Function(program_library, shader.entry_point);
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    return {std::move(shader), device.CompileCompute(descriptor), program_library};
}

BufferSlice GeometryRuntime::Capture(const VertexCaptureProgram& program,
                                     const BufferSlice& canonical_inputs, std::uint32_t vertex_count,
                                     std::uint32_t first_vertex, std::span<const BufferBinding> uniforms) {
    auto prepared=PrepareCapture(program,canonical_inputs,vertex_count,first_vertex,uniforms);
    EncodeCapture(prepared);
    return std::move(prepared.output);
}
GeometryRuntime::PreparedCapture GeometryRuntime::PrepareCapture(const VertexCaptureProgram& program,
                                     const BufferSlice& canonical_inputs, std::uint32_t vertex_count,
                                     std::uint32_t first_vertex, std::span<const BufferBinding> uniforms) {
    if (!vertex_count) {
        return {};
    }
    if ((!program.pipeline && !program.pending_pipeline.valid()) ||
        (program.pipeline && program.pipeline.device != device.Handle()) ||
        program.shader.stage != ShaderStage::VertexCompute || program.shader.needs_buffer_sizes ||
        vertex_count > UINT32_MAX / 16 ||
        std::uint64_t(first_vertex) + vertex_count > std::uint64_t(UINT32_MAX) + 1) {
        throw std::invalid_argument("Invalid native PICA vertex capture program");
    }
    ValidateSlice(canonical_inputs, NSUInteger(vertex_count) * 16 * sizeof(std::array<float, 4>));
    for (const auto& resource : program.shader.resources) {
        if (!resource.active) continue;
        if (resource.descriptor_set == 3 && resource.binding >= 1 && resource.binding <= 3) {
            continue;
        }
        if (resource.kind != ShaderResourceKind::UniformBuffer || resource.count != 1) {
            throw std::invalid_argument("PICA vertex capture contains an unsupported resource");
        }
        bool found{};
        for (const auto& binding : uniforms) {
            if (binding.index == resource.metal_index) {
                if (!binding.buffer || binding.offset > binding.buffer.length ||
                    resource.byte_size > binding.buffer.length - binding.offset) {
                    throw std::invalid_argument("PICA vertex uniform buffer is too small");
                }
                found = true;
            }
        }
        if (!found) {
            throw std::invalid_argument("PICA vertex capture is missing a uniform buffer");
        }
    }
    auto output = output_stream.Allocate(NSUInteger(vertex_count) * sizeof(CapturedVertex));
    const std::array<std::uint32_t, 4> info{vertex_count, first_vertex, 0, 0};
    auto parameters = parameter_stream.Allocate(sizeof(info));
    std::memcpy(parameters.data, info.data(), sizeof(info));
    std::vector<BufferBinding> bindings{uniforms.begin(), uniforms.end()};
    for (const auto& binding : uniforms) {
        if (binding.index >= 17) {
            throw std::invalid_argument("PICA vertex uniforms overlap capture bindings");
        }
    }
    bindings.push_back({canonical_inputs.buffer, canonical_inputs.offset, 17});
    bindings.push_back({output.buffer, output.offset, 18});
    bindings.push_back({parameters.buffer, parameters.offset, 19});
    // Keep slices alive until after bindings and consumers are encoded, including across allocation flushes.
    return {std::move(output),std::move(parameters),program.pipeline,program.pending_pipeline,std::move(bindings),vertex_count};
}
ComputeDispatch GeometryRuntime::CaptureDispatch(const PreparedCapture& prepared) {
    ComputeDispatch dispatch;
    dispatch.pipeline=prepared.pipeline;dispatch.pending_pipeline=prepared.pending_pipeline;
    dispatch.buffers=prepared.bindings;dispatch.threads=MTLSizeMake(prepared.vertex_count,1,1);
    dispatch.group=MTLSizeMake(64,1,1);
    return dispatch;
}
void GeometryRuntime::EncodeCapture(const PreparedCapture& prepared,id<MTL4ComputeCommandEncoder> existing_encoder) {
    if(!prepared.vertex_count) return;
    const auto dispatch=CaptureDispatch(prepared);
    scheduler.QueueCompute(std::span{&dispatch,1},{},nil,existing_encoder!=nil);
}

AssembledVertices GeometryRuntime::Assemble(const BufferSlice& captured, std::uint32_t captured_count,
                                            TriangleTopology topology, std::uint32_t element_count,
                                            std::uint32_t vertex_base, std::uint32_t first_vertex,
                                            const BufferSlice& indices, std::uint32_t index_size,
                                            bool fix_quaternions) {
    const auto triangle_count = topology == TriangleTopology::List ? element_count / 3
                               : element_count >= 3 ? element_count - 2 : 0;
    if (!triangle_count) {
        return {};
    }
    if (topology > TriangleTopology::Fan ||
        (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) ||
        triangle_count > std::numeric_limits<std::uint32_t>::max() / 3 ||
        (!index_size && std::uint64_t(first_vertex) + element_count > std::uint64_t(UINT32_MAX) + 1)) {
        throw std::invalid_argument("Invalid native PICA primitive assembly parameters");
    }
    ValidateSlice(captured, NSUInteger(captured_count) * sizeof(CapturedVertex));
    if (index_size) {
        if (element_count > std::numeric_limits<std::uint32_t>::max() / index_size) {
            throw std::invalid_argument("Native index byte offsets exceed the shader address range");
        }
        ValidateSlice(indices, NSUInteger(element_count) * index_size, 1);
    }
    const auto vertex_count = triangle_count * 3;
    auto output = output_stream.Allocate(NSUInteger(vertex_count) * sizeof(CapturedVertex));
    const std::array<std::uint32_t, 8> info{element_count, captured_count, vertex_base, first_vertex,
        static_cast<std::uint32_t>(topology), index_size, fix_quaternions ? 1U : 0U, triangle_count};
    auto parameters = parameter_stream.Allocate(sizeof(info));
    std::memcpy(parameters.data, info.data(), sizeof(info));
    const std::array<BufferBinding, 4> bindings{{
        {captured.buffer, captured.offset, 0},
        {index_size ? indices.buffer : empty_indices, index_size ? indices.offset : 0, 1},
        {parameters.buffer, parameters.offset, 2},
        {output.buffer, output.offset, 3},
    }};
    if (assembly_arguments) scheduler.RebindBufferArguments(assembly_arguments, bindings);
    else assembly_arguments = scheduler.MakeArguments(bindings);
    const auto table = assembly_arguments;
    scheduler.KeepAlive(assembly_pipeline);
    auto encoder = scheduler.BeginCompute(assembly_pipeline);
    [encoder setComputePipelineState:assembly_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreads:MTLSizeMake(triangle_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    scheduler.EndEncoding();
    return {std::move(output), vertex_count};
}

AssembledVertices GeometryRuntime::Append(PrimitiveStream& stream, const BufferSlice& captured,
                                         std::uint32_t captured_count, TriangleTopology topology,
                                         std::uint32_t element_count, std::uint32_t vertex_base,
                                         std::uint32_t first_vertex, const BufferSlice& indices,
                                         std::uint32_t index_size, bool fix_quaternions,
                                         const BufferSlice& shader_winding) {
    if (!element_count) return {};
    if (topology > TriangleTopology::Fan ||
        (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) ||
        (!index_size && std::uint64_t(first_vertex) + element_count > std::uint64_t(UINT32_MAX) + 1) ||
        (index_size && element_count > UINT32_MAX / index_size) ||
        stream.history.device != device.Handle() || captured.buffer.device != device.Handle() ||
        (index_size && indices.buffer.device != device.Handle())) {
        throw std::invalid_argument("Invalid native primitive stream input");
    }
    ValidateSlice(captured, NSUInteger(captured_count) * sizeof(CapturedVertex));
    if (shader_winding.buffer) {
        if (topology!=TriangleTopology::List || shader_winding.buffer.device!=device.Handle())
            throw std::invalid_argument("Native shader winding requires a list stream on the same device");
        ValidateSlice(shader_winding,sizeof(std::uint32_t),4);
    }
    if (index_size) ValidateSlice(indices, NSUInteger(element_count) * index_size, 1);
    // A deferred complete list always has pending0. This positive append
    // overwrites every logical history word; no previous vertex is consumed.
    stream.deferred_list = {};
    if (stream.topology != topology) stream.Reset();
    const auto total = std::uint64_t(stream.pending) + element_count;
    if (total > UINT32_MAX) throw std::length_error("Native primitive stream exceeds its element representation");
    const auto triangles = topology == TriangleTopology::List ? total / 3 : total > 2 ? total - 2 : 0;
    if (triangles > UINT32_MAX / 3) throw std::length_error("Native primitive stream output exceeds vertex storage");
    auto output = output_stream.Allocate(std::max<std::uint64_t>(1, triangles * 3) * sizeof(CapturedVertex));
    const std::array<std::uint32_t, 12> info{element_count, captured_count, vertex_base, first_vertex,
        std::uint32_t(topology), index_size, fix_quaternions ? 1U : 0U, std::uint32_t(triangles),
        stream.needs_reset ? 1U : 0U, stream.pending,shader_winding.buffer ? 1U:0U,0};
    const auto parameters = parameter_stream.Allocate(sizeof(info));
    std::memcpy(parameters.data, info.data(), sizeof(info));
    const auto next_history_offset=272u-stream.history_offset;
    const std::array<BufferBinding, 7> bindings{{
        {captured.buffer, captured.offset, 0}, {index_size ? indices.buffer : empty_indices, index_size ? indices.offset : 0, 1},
        {parameters.buffer, parameters.offset, 2}, {stream.history, stream.history_offset, 3}, {output.buffer, output.offset, 4},
        {shader_winding.buffer ? shader_winding.buffer:empty_indices,shader_winding.buffer ? shader_winding.offset:0,5},
        {stream.history,next_history_offset,6}}};
    ComputeDispatch command;
    command.pipeline=stream_pipeline; command.buffers={bindings.begin(),bindings.end()};
    command.threads=MTLSizeMake(std::max<std::uint64_t>(triangles,32),1,1);
    command.group=MTLSizeMake(64,1,1);
    scheduler.QueueCompute(std::span{&command,1});
    stream.history_offset=next_history_offset;
    stream.pending = topology == TriangleTopology::List ? total % 3 : std::min<std::uint64_t>(2, total);
    stream.topology = topology;
    stream.needs_reset = false;
    stream.known_zero_history = false;
    return {output, std::uint32_t(triangles * 3)};
}

MTL4LibraryFunctionDescriptor* GeometryRuntime::RasterVertexFunction() const {
    return Device::Function(library, "raster_vertex");
}

PrimitiveSnapshot GeometryRuntime::SaveStream(const PrimitiveStream& stream) {
    if (stream.history.device != device.Handle()) throw std::invalid_argument("Primitive snapshot belongs to another Metal device");
    PrimitiveSnapshot result{.pending = stream.pending, .topology = stream.topology, .needs_reset = stream.needs_reset};
    if (stream.deferred_list.buffer) {
        const auto& args=stream.deferred_list;
        if(args.buffer.device!=device.Handle() || !args.buffer.contents || args.offset%16 || args.size!=64 ||
            args.offset>args.buffer.length || args.size>args.buffer.length-args.offset ||
            stream.pending || stream.topology!=TriangleTopology::List)
            throw std::invalid_argument("Invalid deferred native list metadata");
        scheduler.Finish(Scheduler::WaitReason::Readback);
        const auto* words=reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(args.buffer.contents)+args.offset);
        if(words[8]>1 || (words[9]!=0 && words[9]!=272) || words[10] || words[11] || words[12])
            throw std::runtime_error("GPU rejected native list metadata/count");
        result.needs_reset=words[8];
        if(!result.needs_reset) {
            const auto readback=device.CreateBuffer(272,true);
            TransferRuntime{device,scheduler}.CopyBuffer(stream.history,words[9],readback,0,272);
            scheduler.Finish(Scheduler::WaitReason::Readback);std::memcpy(result.history.data(),readback.contents,272);
        }
        ValidateStream(result);return result;
    }
    if (!stream.needs_reset && !stream.known_zero_history) {
        const auto readback = device.CreateBuffer(result.history.size(), true);
        TransferRuntime{device, scheduler}.CopyBuffer(stream.history, stream.history_offset, readback, 0, result.history.size());
        scheduler.Finish(Scheduler::WaitReason::Readback);
        std::memcpy(result.history.data(), readback.contents, result.history.size());
    }
    ValidateStream(result);
    return result;
}

void GeometryRuntime::ValidateStream(const PrimitiveSnapshot& snapshot) {
    if (snapshot.topology > TriangleTopology::Fan || snapshot.pending > 2 ||
        (snapshot.needs_reset && snapshot.pending != 0))
        throw std::invalid_argument("Invalid native primitive snapshot metadata");
    std::array<std::uint32_t,4> state{};
    std::memcpy(state.data(), snapshot.history.data()+256, sizeof(state));
    const auto [index, ready, valid0, valid1] = state;
    if (ready > 1 || valid0 > 1 || valid1 > 1 ||
        (snapshot.needs_reset && std::any_of(snapshot.history.begin(),snapshot.history.end(),
                                          [](std::byte value) { return value != std::byte{}; })))
        throw std::invalid_argument("Invalid native primitive snapshot history flags");
    if (!snapshot.needs_reset) {
        const bool list = snapshot.topology == TriangleTopology::List;
        if ((list && (index != snapshot.pending || ready != 0)) ||
            (!list && (snapshot.pending == 0 || ready != (snapshot.pending == 2) || index > 1)) ||
            (snapshot.topology == TriangleTopology::Fan && index != 1) ||
            (snapshot.pending == 1 && !list && index != 1))
            throw std::invalid_argument("Native primitive snapshot disagrees with its buffered count/topology");
    }
}

void GeometryRuntime::LoadStream(PrimitiveStream& stream, const PrimitiveSnapshot& snapshot) {
    if (stream.history.device != device.Handle()) throw std::invalid_argument("Primitive snapshot target belongs to another Metal device");
    ValidateStream(snapshot);
    const auto upload = device.CreateBuffer(snapshot.history.size(), true);
    std::memcpy(upload.contents, snapshot.history.data(), snapshot.history.size());
    TransferRuntime{device, scheduler}.CopyBuffer(upload, 0, stream.history, stream.history_offset, snapshot.history.size());
    stream.pending = snapshot.pending;
    stream.topology = snapshot.topology;
    stream.needs_reset = snapshot.needs_reset;
    stream.known_zero_history = false;
    stream.deferred_list = {};
}

} // namespace Metal
