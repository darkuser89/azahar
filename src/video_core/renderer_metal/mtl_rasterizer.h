// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_draw_runtime.h"
#include "mtl_texture_runtime.h"
#include "mtl_vertex_fetch.h"
#include "mtl_screen_layout.h"
#include "mtl_geometry_program.h"
#include "mtl_state.h"
#include "mtl_gas_state.h"
#include "mtl_gas_runtime.h"
#include "video_core/rasterizer_accelerated.h"
#include <deque>

namespace Pica { struct FramebufferConfig; }

namespace Metal {

class RasterizerMetal final : public VideoCore::RasterizerAccelerated {
public:
    RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                    VideoCore::CustomTexManager& custom_textures, VideoCore::RendererBase& renderer,
                    Device& device, Scheduler& scheduler);
    ~RasterizerMetal() override;
    bool IsMetalRasterizer() const override { return true; }
    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;
    void SwitchDiskResources(u64 title_id) override;
    void AddTriangle(const Pica::OutputVertex&, const Pica::OutputVertex&, const Pica::OutputVertex&) override;
    void DrawTriangles() override;
    bool AccelerateDrawBatch(bool indexed) override;
    bool AccelerateImmediateVertex() override;
    void ResetMetalPrimitiveStream() override;
    void WriteMetalGasRegisters(u32 reg, std::span<const u32> values, u32 mask) override;
    void UpdateMetalGasAccumulatorMaximum() override;
    void SynchronizeMetalGasMaximum() override;
    void WriteMetalGSUniforms(u32 first,u32 count,u16 bool_mask,u8 integer_mask) override;
    std::vector<u8> SaveMetalState() override;
    void LoadMetalState(std::span<const u8> data) override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool GetScreen(const Pica::FramebufferConfig& config, PAddr address, u32 stride, ScreenImage& screen);
    void TickFrame();
    TextureRuntime& Textures() { return textures; }
    void SetPresentationPrewarm(std::function<void(const std::atomic_bool&,
        const VideoCore::DiskResourceLoadCallback&)> callback) { presentation_prewarm = std::move(callback); }

private:
    friend struct InputReferenceTestAccess;
    struct DeferredGeometry;
    void RetireGeometry(bool finish=false);
    void QueueGeometryDraw(const ShaderMachineProgramView& program,
                           std::vector<DeferredGeometry*>* batch=nullptr);
    u32 disk_cache_frames{};
    std::function<void(const std::atomic_bool&, const VideoCore::DiskResourceLoadCallback&)> presentation_prewarm;
    BufferSlice UploadGuestBytes(std::uint64_t address, std::uint64_t size, bool dedicated = false);
    struct UniformUploads {
        std::array<BufferSlice, 4> storage{};
        std::array<BufferBinding, 4> bindings{};
        std::size_t count{};
        std::span<const BufferBinding> Bindings() const { return {bindings.data(), count}; }
    };
    UniformUploads UploadUniforms(std::uint8_t mask);
    void SyncLUTs(const CompiledFragmentShader& fragment, std::vector<TextureBinding>& bindings);
    void BindTextures(const Framebuffer& framebuffer, const CompiledFragmentShader& fragment,
                      VideoCore::SurfaceId normal_source, std::vector<TextureBinding>& bindings,
                      std::vector<SamplerBinding>& samplers);
    TriangleTopology Topology() const;
    void DrawNativeVertices(const AssembledVertices& vertices, const NativeMeshDraw* mesh = nullptr);
    NativeMeshDraw PrepareNativeMesh(std::span<const BufferSlice> inputs, const PicaVertexLayout& layout,
        const BufferSlice& indices, std::uint32_t index_size, std::uint32_t vertex_count,
        std::uint32_t element_count, std::uint32_t first_vertex,
        ShaderMachineRuntime::NativeResetSelection&& selection);
    GeometryInputConfig GeometryConfig() const;
    void SubmitGeometryVertex(const BufferSlice& raw,const ShaderMachineProgramView& program);
    void DrawGeometryResult(const ShaderMachineResult& result);
    bool DrawGeometryBatch(bool indexed);
    bool DrawMachineBatch(bool indexed,bool use_geometry);
    void SubmitMachineVertex(const BufferSlice& output,std::uint32_t mask);
    Device& device;
    Scheduler& scheduler;
    TextureRuntime textures;
    RasterizerCache cache;
    GeometryRuntime geometry;
    PrimitiveStream primitive_stream;
    VertexInputState vertex_input_state;
    VertexFetchRuntime fetch;
    InputReferenceRuntime input_references;
    ShaderRuntime shaders;
    DrawRuntime draws;
    ShaderMachineRuntime machine;
    ShaderMachineState vs_unit,gs_unit;
    GeometryProgramRuntime geometry_program;
    GeometryInputState geometry_input;
    struct DeferredGeometry {
        GeometryQueuedInvocation invocation;
        AssembledVertices vertices;
        std::uint64_t tick{};
        bool ready{};
    };
    std::deque<DeferredGeometry> deferred_geometry;
    BufferSlice deferred_geometry_abort{},gs_emission_scratch{},gs_capture_scratch{};
    bool gs_deferred_logged{};
    bool gs_batch_logged{},gs_batch_ineligible_logged{};
    bool gs_vertex_batch_logged{};
    bool gs_uniforms_seeded{};
    GasTransferState gas;
    GasRuntime gas_runtime;
    GasFeedback gas_feedback;
    // GPU maximum newer than the CPU register/snapshot. This is distinct from
    // gas.feedback_pending, which records an ordered accumulator reset.
    bool gas_maximum_pending{};
    StreamBuffer uploads;
    SharedUploadPool vertex_uploads;
    ImmutableUniformCache uniform_uploads;
    LutTextureStream lighting_luts, proctex_value_luts, proctex_color_luts;
    id<MTLBuffer> mesh_empty;
    bool mesh_draw_logged{};
    bool gs_list_draw_logged{};
    std::array<id<MTLTexture>, 3> lut_textures{};
};

} // namespace Metal
