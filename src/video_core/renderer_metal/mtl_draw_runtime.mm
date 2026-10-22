// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_draw_runtime.h"
#include "common/logging/log.h"
#include <algorithm>
#include <stdexcept>

namespace Metal {

PicaDrawState BuildPicaDrawState(const Pica::RegsInternal& regs) {
    const auto& fb = regs.framebuffer.framebuffer;
    // Match Vulkan/OpenGL: cached framebuffer textures are logical pixel images,
    // so register 0x11B does not reject a draw. Physical Block32 addressing is
    // not represented by the shared surface cache or the guest codec yet.
    const auto& output = regs.framebuffer.output_merger;
    PicaDrawState state{};
    state.shadow = regs.framebuffer.IsShadowRendering();
    state.gas_accumulation = output.fragment_operation_mode == Pica::FramebufferRegs::FragmentOperationMode::Gas;
    state.blend.color_write_mask = fb.allow_color_write ? (output.depth_color_mask >> 8) & 15 : 0;
    // Both ordinary blending and logic operations run in the fragment shader with guest
    // channel precision. Fixed-function blending would blend twice or retain RGBA8 precision.
    state.blend.logic_operation = output.alphablend_enable ? 3 : std::uint8_t(output.logic_op.Value());
    state.depth.depth_test = output.depth_test_enable || output.depth_write_enable;
    state.depth.depth_compare = output.depth_test_enable ? std::uint8_t(output.depth_test_func.Value()) : 1;
    state.depth.depth_write = fb.allow_depth_stencil_write && output.depth_write_enable;
    const auto& stencil = output.stencil_test;
    state.depth.stencil_test = stencil.enable && regs.framebuffer.HasStencil();
    state.depth.stencil_compare = std::uint8_t(stencil.func.Value());
    state.depth.stencil_fail = std::uint8_t(stencil.action_stencil_fail.Value());
    state.depth.stencil_depth_fail = std::uint8_t(stencil.action_depth_fail.Value());
    state.depth.stencil_pass = std::uint8_t(stencil.action_depth_pass.Value());
    state.depth.stencil_read_mask = stencil.input_mask;
    state.depth.stencil_write_mask = fb.allow_depth_stencil_write ? stencil.write_mask.Value() : 0;
    state.stencil_reference = stencil.reference_value;
    state.use_color = fb.GetColorBufferPhysicalAddress() != 0 &&
                      (state.shadow || (state.blend.color_write_mask && state.blend.logic_operation != 6));
    state.use_depth = !state.shadow && fb.GetDepthBufferPhysicalAddress() != 0 &&
                      (state.depth.depth_test || state.depth.stencil_test);
    state.winding = fb.IsFlipped() ? MTLWindingClockwise : MTLWindingCounterClockwise;
    switch (regs.rasterizer.cull_mode.Value()) {
    case Pica::RasterizerRegs::CullMode::KeepAll:
    case Pica::RasterizerRegs::CullMode::KeepAll2: state.cull = MTLCullModeNone; break;
    // raster_vertex mirrors PICA clip Y before Metal's viewport transform.
    case Pica::RasterizerRegs::CullMode::KeepClockWise: state.cull = MTLCullModeBack; break;
    case Pica::RasterizerRegs::CullMode::KeepCounterClockWise: state.cull = MTLCullModeFront; break;
    default: throw std::invalid_argument("Invalid PICA cull mode");
    }
    switch (fb.color_format.Value()) {
    case Pica::FramebufferRegs::ColorFormat::RGBA8: state.color_format = FragmentColorFormat::RGBA8; break;
    case Pica::FramebufferRegs::ColorFormat::RGB8: state.color_format = FragmentColorFormat::RGB8; break;
    case Pica::FramebufferRegs::ColorFormat::RGB5A1: state.color_format = FragmentColorFormat::RGB5A1; break;
    case Pica::FramebufferRegs::ColorFormat::RGB565: state.color_format = FragmentColorFormat::RGB565; break;
    case Pica::FramebufferRegs::ColorFormat::RGBA4: state.color_format = FragmentColorFormat::RGBA4; break;
    default: throw std::invalid_argument("Invalid PICA framebuffer color format");
    }
    if (state.gas_accumulation) {
        if (state.color_format != FragmentColorFormat::RGBA8)
            throw std::invalid_argument("Native Gas density attachment must use the guest Gas/RGBA8 format");
        // Gas has its own density/depth operation, rather than ordinary fixed
        // depth/stencil testing, RGB logic or RGBA blending.
        state.depth = {};
        state.blend = {};
        state.blend.color_write_mask = fb.allow_color_write ? 15 : 0;
        state.use_color = fb.GetColorBufferPhysicalAddress() && fb.allow_color_write;
        state.use_depth = state.use_color && fb.GetDepthBufferPhysicalAddress() &&
                          ((regs.reg_array[0x126] >> 24) & 3) >= 2;
    }
    return state;
}

DrawRuntime::DrawRuntime(Device& device, Scheduler& scheduler_, GeometryRuntime& geometry_)
    : device{device}, scheduler{scheduler_}, geometry{geometry_}, pipelines{device} {}

void DrawRuntime::Draw(const PicaDrawState& state, const CompiledFragmentShader& fragment,
                       MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
                       const AssembledVertices& vertices, std::span<const BufferBinding> uniforms,
                       std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers) {
    if (!vertices.vertex_count || !scissor.width || !scissor.height) return;
    if (!pass || !vertices.storage.buffer || vertices.vertex_count % 3 ||
        std::uint64_t(vertices.vertex_count) * 128 > vertices.storage.size ||
        vertices.storage.offset % 16 || vertices.storage.offset > vertices.storage.buffer.length ||
        vertices.storage.size > vertices.storage.buffer.length - vertices.storage.offset) {
        throw std::invalid_argument("Invalid native PICA draw storage or render pass");
    }
    if(vertices.indirect) {
        const auto& proof=*vertices.indirect;const auto& args=proof.arguments;
        if(proof.capacity!=vertices.vertex_count || !args.buffer || args.buffer.device!=vertices.storage.buffer.device ||
            args.offset%4 || args.size!=64 || args.offset>args.buffer.length || args.size>args.buffer.length-args.offset)
            throw std::invalid_argument("Invalid bounded native indirect draw");
        scheduler.UseResource(args.buffer);
    }
    if (fragment.program.shader.needs_buffer_sizes)
        throw std::invalid_argument("Native fragment draw requires an explicit buffer-size table");
    for (const auto& resource : fragment.program.shader.resources) {
        if (!resource.active) continue;
        for (unsigned element = 0; element < resource.count; ++element) {
            const auto index = resource.metal_index + element;
            if (resource.kind == ShaderResourceKind::UniformBuffer || resource.kind == ShaderResourceKind::StorageBuffer) {
                const auto binding = std::find_if(uniforms.begin(), uniforms.end(),
                    [&](const auto& value) { return value.index == index; });
                if (binding == uniforms.end() || !binding->buffer || binding->offset > binding->buffer.length ||
                    resource.byte_size > binding->buffer.length - binding->offset)
                    throw std::invalid_argument("Native fragment draw has a missing or short buffer binding");
            } else if (resource.kind != ShaderResourceKind::Sampler) {
                if (std::none_of(textures.begin(), textures.end(),
                    [&](const auto& value) { return value.index == index && value.texture; }))
                    throw std::invalid_argument("Native fragment draw has a missing texture binding");
            }
            if (resource.sampler_index != NoShaderBinding &&
                std::none_of(samplers.begin(), samplers.end(), [&](const auto& value) {
                    return value.index == resource.sampler_index + element && value.sampler;
                })) throw std::invalid_argument("Native fragment draw has a missing sampler binding");
        }
    }
    RenderPipelineDescription description{
        geometry.RasterVertexLibrary(), "raster_vertex", fragment.library, fragment.program.shader.entry_point,
        pass.colorAttachments[0].texture ? pass.colorAttachments[0].texture.pixelFormat : MTLPixelFormatInvalid,
        state.blend, {}, fragment.pending_library};
    if (state.shadow) description.blend.color_write_mask = 0;
    const auto pipeline = pipelines.RenderAsync(description);
    const auto depth = pipelines.DepthStencil(state.shadow ? DepthStencilState{} : state.depth);
    std::vector<BufferBinding> buffers{uniforms.begin(), uniforms.end()};
    buffers.push_back({vertices.storage.buffer, vertices.storage.offset, 3});
    scheduler.KeepAlive(depth);
    // Both ordinary and indirect draws consume immutable captured ranges.
    // Any intervening fetch/assembly/write starts a compute encoder and closes
    // the render pass. Native framebuffer fetch remains raster ordered; only
    // shaders without arbitrary storage side effects may share this pass.
    const bool reusable = !state.shadow && !state.gas_accumulation &&
        std::none_of(fragment.program.shader.resources.begin(), fragment.program.shader.resources.end(), [](const auto& resource) {
            return resource.active && (resource.kind == ShaderResourceKind::StorageTexture || resource.kind == ShaderResourceKind::StorageBuffer);
        });
    RenderCommand command{.pipeline=nil, .depth=depth, .viewport=viewport, .scissor=scissor,
        .winding=state.winding, .cull=state.cull, .stencil=state.stencil_reference,
        .vertices=vertices.vertex_count, .pending_pipeline=pipeline};
    if (vertices.indirect) {
        command.indirect=vertices.indirect->arguments.buffer;
        command.indirect_offset=vertices.indirect->arguments.offset;
    }
    const auto snapshot=scheduler.SnapshotRenderPass(pass);
    const ReadOnlyBufferRenderProof proof;
    scheduler.QueueRender(*snapshot,command,buffers,textures,samplers,reusable,{},&proof);
}

void DrawRuntime::DrawMesh(const PicaDrawState& state, const CompiledFragmentShader& fragment,
    MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
    const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
    std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers) {
    DrawMeshImpl(state, fragment, pass, viewport, scissor, mesh, uniforms, textures, samplers, nullptr);
}

void DrawRuntime::DrawMesh(const PicaDrawState& state, const CompiledFragmentShader& fragment,
    const ImmutableRenderPass& pass, MTLViewport viewport, MTLScissorRect scissor,
    const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
    std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers) {
    if (pass.device != device.Handle()) throw std::invalid_argument("Immutable mesh pass belongs to another Metal device");
    DrawMeshImpl(state, fragment, pass.descriptor, viewport, scissor, mesh, uniforms, textures, samplers, &pass);
}

void DrawRuntime::DrawMeshImpl(const PicaDrawState& state, const CompiledFragmentShader& fragment,
    MTL4RenderPassDescriptor* pass, MTLViewport viewport, MTLScissorRect scissor,
    const NativeMeshDraw& mesh, std::span<const BufferBinding> uniforms,
    std::span<const TextureBinding> textures, std::span<const SamplerBinding> samplers,
    const ImmutableRenderPass* snapshot) {
    if (!scissor.width || !scissor.height) return;
    if (!pass || !mesh.program || (!mesh.program->library && !mesh.program->pending_library.valid()) ||
        mesh.program->shader.stage != ShaderStage::VertexMesh || mesh.program->shader.needs_buffer_sizes ||
        !mesh.inputs || !mesh.vertex_count || !mesh.element_count || mesh.element_count % 3 ||
        (mesh.index_size != 0 && mesh.index_size != 1 && mesh.index_size != 2 && mesh.index_size != 4) ||
        !mesh.references.buffer || mesh.references.size != 768 || !mesh.capture.buffer || mesh.capture.size != 16 ||
        !mesh.empty || mesh.empty.length < 512 || mesh.chunks.empty())
        throw std::invalid_argument("Invalid immutable native mesh draw");
    const auto* capture = static_cast<const std::uint32_t*>(mesh.capture.data);
    if (!capture || capture[0] != mesh.vertex_count || capture[2] || capture[3])
        throw std::invalid_argument("Invalid immutable mesh capture parameters");
    const auto& indices = mesh.inputs->indices;
    if (mesh.index_size && (!indices.buffer || indices.offset > indices.buffer.length ||
        indices.size > indices.buffer.length - indices.offset ||
        std::uint64_t(mesh.element_count) * mesh.index_size > indices.size))
        throw std::invalid_argument("Invalid immutable mesh index slice");
    std::uint32_t first_triangle = 0;
    for (const auto& chunk : mesh.chunks) {
        const auto* params = static_cast<const std::uint32_t*>(chunk.parameters.data);
        const auto remaining = mesh.element_count / 3 - first_triangle;
        const auto groups = std::min(1024u, (remaining + 31) / 32);
        if (!params || !chunk.parameters.buffer || chunk.parameters.size != 48 || !groups || chunk.groups != groups ||
            params[0] != mesh.element_count || params[1] != mesh.index_size ||
            params[4] || params[5] || params[6] != 1 || params[7] > 1 ||
            params[8] != mesh.element_count / 3 || params[9] || params[10] != 1 || params[11] != first_triangle)
            throw std::invalid_argument("Native mesh draw must have independent complete-list chunks");
        first_triangle += std::min(remaining, 1024u * 32);
    }
    if (first_triangle != mesh.element_count / 3) throw std::invalid_argument("Incomplete native mesh chunks");
    if (fragment.program.shader.needs_buffer_sizes)
        throw std::invalid_argument("Native fragment draw requires an explicit buffer-size table");
    for (const auto& resource : fragment.program.shader.resources) {
        if (!resource.active) continue;
        for (unsigned element = 0; element < resource.count; ++element) {
            const auto index = resource.metal_index + element;
            if (resource.kind == ShaderResourceKind::UniformBuffer || resource.kind == ShaderResourceKind::StorageBuffer) {
                const auto binding = std::find_if(uniforms.begin(), uniforms.end(),
                    [&](const auto& value) { return value.index == index; });
                if (binding == uniforms.end() || !binding->buffer || binding->offset > binding->buffer.length ||
                    resource.byte_size > binding->buffer.length - binding->offset)
                    throw std::invalid_argument("Native fragment draw has a missing or short buffer binding");
            } else if (resource.kind != ShaderResourceKind::Sampler) {
                if (std::none_of(textures.begin(), textures.end(),
                    [&](const auto& value) { return value.index == index && value.texture; }))
                    throw std::invalid_argument("Native fragment draw has a missing texture binding");
            }
            if (resource.sampler_index != NoShaderBinding &&
                std::none_of(samplers.begin(), samplers.end(), [&](const auto& value) {
                    return value.index == resource.sampler_index + element && value.sampler;
                })) throw std::invalid_argument("Native fragment draw has a missing sampler binding");
        }
    }
    RenderPipelineDescription description{mesh.program->library, mesh.program->shader.entry_point,
        fragment.library, fragment.program.shader.entry_point,
        pass.colorAttachments[0].texture ? pass.colorAttachments[0].texture.pixelFormat : MTLPixelFormatInvalid,
        state.blend, mesh.program->pending_library, fragment.pending_library};
    if (state.shadow) description.blend.color_write_mask = 0;
    const auto pipeline = pipelines.MeshAsync(description);
    const auto depth = pipelines.DepthStencil(state.shadow ? DepthStencilState{} : state.depth);
    std::vector<BufferBinding> buffers{uniforms.begin(), uniforms.end()};
    buffers.push_back({mesh.references.buffer, mesh.references.offset, 17});
    buffers.push_back({mesh.capture.buffer, mesh.capture.offset, 19});
    buffers.push_back({mesh.index_size ? indices.buffer : mesh.empty, mesh.index_size ? indices.offset : 0, 20});
    buffers.push_back({mesh.empty, 0, 18});
    buffers.push_back({mesh.empty, 0, 25});
    buffers.push_back({mesh.empty, 0, 26});
    buffers.push_back({mesh.chunks.front().parameters.buffer, mesh.chunks.front().parameters.offset, 3});
    for (const auto& resource : mesh.program->shader.resources) {
        if (!resource.active) continue;
        if (resource.kind != ShaderResourceKind::UniformBuffer && resource.kind != ShaderResourceKind::StorageBuffer)
            throw std::invalid_argument("Native mesh vertex program contains an unexpected resource");
        for (unsigned element = 0; element < resource.count; ++element) {
            const auto found = std::find_if(buffers.begin(), buffers.end(), [&](const auto& item) {
                return item.index == resource.metal_index + element;
            });
            if (found == buffers.end() || !found->buffer || found->offset > found->buffer.length ||
                resource.byte_size > found->buffer.length - found->offset)
                throw std::invalid_argument("Native mesh vertex program has a missing or short buffer");
        }
    }
    // Physical pointers are not separate argument-table bindings. Explicitly
    // retain every backing allocation and its complete residency chain.
    for (const auto& input : mesh.inputs->registers) if (input.gpu.kind) scheduler.UseResource(input.storage);
    scheduler.KeepAlive(depth);
    // Attachment framebuffer fetch is ordered across draws by the native
    // color(0), raster_order_group(0) interface emitted by TranslateGuestGLSL.
    // It stays inside tile memory; only actual storage side effects require
    // the conservative pass boundary below.
    const bool reusable = !state.shadow && !state.gas_accumulation &&
        std::none_of(fragment.program.shader.resources.begin(), fragment.program.shader.resources.end(), [](const auto& resource) {
            return resource.active && (resource.kind == ShaderResourceKind::StorageTexture || resource.kind == ShaderResourceKind::StorageBuffer);
        });
    const auto owned_snapshot=snapshot ? std::shared_ptr<const ImmutableRenderPass>{} : scheduler.SnapshotRenderPass(pass);
    const auto& immutable=snapshot ? *snapshot : *owned_snapshot;
    std::vector<MeshRenderChunk> chunks;
    chunks.reserve(mesh.chunks.size());
    for (const auto& chunk:mesh.chunks)
        chunks.push_back({{chunk.parameters.buffer,chunk.parameters.offset,3},chunk.groups});
    const RenderCommand command{.pipeline=nil, .depth=depth, .viewport=viewport, .scissor=scissor,
        .winding=state.winding, .cull=state.cull, .stencil=state.stencil_reference,
        .mesh_groups=mesh.chunks.front().groups, .pending_pipeline=pipeline};
    // The checked complete-list parameters disable both history stores
    // (reserved0 == 1) and debug output. Vertex evaluation only reads the
    // immutable register references/uniforms; fragment storage is gated above.
    const ReadOnlyBufferRenderProof proof;
    scheduler.QueueRender(immutable,command,buffers,textures,samplers,reusable,chunks,&proof);
}

void DrawRuntime::Clear() { pipelines.Clear(); }
PipelinePrewarmResult DrawRuntime::Prewarm(const std::atomic_bool& stop,
    const std::function<void(std::size_t, std::size_t)>& progress) {
    constexpr std::array<std::string_view, 1> entries{"raster_vertex"};
    const auto before = device.DiskCacheStats();
    const auto result = pipelines.Prewarm(stop, progress, entries, true);
    if (result.candidates) {
        const auto after = device.DiskCacheStats();
        LOG_INFO(Render, "Metal4 PICA pipeline prewarm: candidates={}, prepared={}, failed={}, cancelled={}, archive_hits={}, native_compiles={}",
                 result.candidates, result.prepared, result.failed, result.cancelled,
                 after.archive_hits - before.archive_hits, after.compiled_pipelines - before.compiled_pipelines);
    }
    return result;
}
PipelineCacheStats DrawRuntime::PipelineStats() const { return pipelines.Stats(); }

} // namespace Metal
