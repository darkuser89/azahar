// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_device.h"
#include <array>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <source_location>
#include <thread>
#include <unordered_set>

namespace Metal {

struct BufferBinding {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    NSUInteger index;
};

struct TextureBinding {
    id<MTLTexture> texture;
    NSUInteger index;
};

struct SamplerBinding {
    id<MTLSamplerState> sampler;
    NSUInteger index;
};

// Only native providers with buffer-only, read-only vertex/mesh inputs and no
// fragment storage writes may issue this proof. The scheduler separately
// rejects attachments that could alias those buffers (views included).
class ReadOnlyBufferRenderProof final {
    friend class DrawRuntime;
    friend class TransferRuntime;
    friend struct SchedulerTestAccess;
    friend struct StageTimingTestAccess;
    ReadOnlyBufferRenderProof() = default;
};

// A dedicated mutable binding set. Copying would alias the native table while
// allowing the two owners' occupied-slot masks to diverge.
class ReusableArgumentTable final {
    friend class Scheduler;
public:
    ReusableArgumentTable() = default;
    ReusableArgumentTable(const ReusableArgumentTable&) = delete;
    ReusableArgumentTable& operator=(const ReusableArgumentTable&) = delete;
private:
    id<MTL4ArgumentTable> table{nil};
    std::uint32_t buffers{};
    std::array<std::uint64_t, 2> textures{};
    std::uint16_t samplers{};
    // Native IDs may be reused after their objects die. Keep the current
    // objects alive until replacement/removal, independently of GPU submissions.
    // Buffer addresses are not cached: raw table aliases can rebind them.
    std::array<std::uint64_t, 128> texture_ids{};
    std::array<std::uint64_t, 16> sampler_ids{};
    std::array<id<MTLTexture>, 128> texture_objects{};
    std::array<id<MTLSamplerState>, 16> sampler_objects{};
};

// Own a private copy: callers cannot mutate the descriptor after its complete
// compatibility key has been checked. GPU encoders still retain all resources.
class ImmutableRenderPass final {
    friend class Scheduler;
    friend class DrawRuntime;
    ImmutableRenderPass() = default;
    MTL4RenderPassDescriptor* descriptor{};
    id<MTLDevice> device{};
    std::array<std::uint64_t, 128> key{};
    bool compatible{};
    bool buffer_disjoint_attachments{};
};

// Immutable native draw parameters. All buffer/texture/sampler bindings are
// copied separately; a mutable owner-side argument table never crosses threads.
struct RenderCommand {
    id<MTLRenderPipelineState> pipeline;
    id<MTLDepthStencilState> depth;
    MTLViewport viewport;
    MTLScissorRect scissor;
    MTLWinding winding;
    MTLCullMode cull;
    std::uint32_t stencil{};
    NSUInteger vertices{}, mesh_groups{};
    id<MTLBuffer> indirect{nil};
    NSUInteger indirect_offset{};
    std::shared_future<id<MTLRenderPipelineState>> pending_pipeline;
    MTLPrimitiveType primitive{MTLPrimitiveTypeTriangle};
    std::array<float, 4> blend_color{};
};
struct RenderBatchDraw {
    RenderCommand command{};
    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    std::vector<SamplerBinding> samplers;
};
struct MeshRenderChunk { BufferBinding parameters; NSUInteger groups; };
// Complete immutable compute inputs. Native tables never cross the ownership
// handoff; the encoding worker publishes its own table from these bindings.
struct ComputeDispatch {
    id<MTLComputePipelineState> pipeline{};
    std::shared_future<id<MTLComputePipelineState>> pending_pipeline{};
    std::vector<BufferBinding> buffers;
    std::vector<TextureBinding> textures;
    std::vector<SamplerBinding> samplers;
    MTLSize threads{1,1,1}, group{1,1,1};
    // A positive value dispatches exactly this many native SIMD groups. The
    // actual pipeline width is resolved/validated on the encoding worker.
    NSUInteger simd_groups{};
};
struct ComputeBufferCopy {
    id<MTLBuffer> source, destination;
    NSUInteger source_offset{}, destination_offset{}, size{};
};
struct EncodingWorkerStats {
    std::uint64_t queued_draws{}, encoded_draws{};
    bool separate_thread{};
    std::uint64_t profiled_cpu_nanoseconds{}, profiled_jobs{};
    std::uint64_t queued_dispatches{}, encoded_dispatches{};
    std::uint64_t buffer_readonly_render_passes{}, conservative_render_passes{};
};

// Encoding and submission are confined to one owner. A drained scheduler may explicitly
// release ownership for frontend initialization -> emulation handoff.
class Scheduler final {
public:
    explicit Scheduler(const Device& device);
    ~Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    id<MTL4RenderCommandEncoder> BeginRender(MTL4RenderPassDescriptor* descriptor, NSString* label = nil,
                                            const ReadOnlyBufferRenderProof* proof = nullptr,
                                            std::source_location source = std::source_location::current());
    id<MTL4RenderCommandEncoder> BeginCompatibleRender(MTL4RenderPassDescriptor* descriptor,
                                                      NSString* label = nil);
    std::shared_ptr<const ImmutableRenderPass> SnapshotRenderPass(MTL4RenderPassDescriptor* descriptor);
    id<MTL4RenderCommandEncoder> BeginCompatibleRender(const ImmutableRenderPass& pass,
                                                       NSString* label = nil);
    id<MTL4ComputeCommandEncoder> BeginCompute(id<MTLComputePipelineState> pipeline = nil,
                                              std::source_location source = std::source_location::current());
    void QueueRender(const ImmutableRenderPass& pass, const RenderCommand& command,
                     std::span<const BufferBinding> buffers,
                     std::span<const TextureBinding> textures = {},
                     std::span<const SamplerBinding> samplers = {}, bool reusable = true,
                     std::span<const MeshRenderChunk> chunks = {},
                     const ReadOnlyBufferRenderProof* proof = nullptr);
    // One complete logical pass, including one-time Clear/resolve/store actions.
    // Preflight all inputs before enqueueing; no caller table crosses threads.
    void QueueRenderBatch(const ImmutableRenderPass& pass, std::span<const RenderBatchDraw> draws);
    // Copies precede the dispatch chain in the same encoder. Each boundary has
    // an explicit native dependency, including the post-GS uniform update.
    void QueueCompute(std::span<const ComputeDispatch> dispatches,
                      std::span<const ComputeBufferCopy> copies = {},
                      NSString* label = nil, bool append_current = false);
    EncodingWorkerStats WorkerStats() const;
    struct SubmissionTiming {
        std::uint64_t tick{};
        double gpu_begin_seconds{}, gpu_end_seconds{};
        bool failed{};
    };
    // Optional native commit-feedback diagnostics. No timestamp commands or
    // additional GPU synchronization are inserted. Each submitted callback
    // owns its observer copy and may outlive this scheduler or a later setter.
    // Observers run on Metal's feedback thread, never the encoding owner.
    void SetSubmissionObserver(std::function<void(const SubmissionTiming&)> observer);
    // Keep adjacent compute/blit operations in one native encoder. Each new
    // operation retains a device-visible intra-encoder dependency. Worker-only
    // batches enqueue their close without waiting at scope exit. Rendering,
    // explicit EndEncoding and submission still force a real encoder boundary;
    // direct native owner access drains any queued worker commands first.
    class ComputeBatchScope final {
    public:
        explicit ComputeBatchScope(Scheduler& scheduler);
        ~ComputeBatchScope() noexcept;
        ComputeBatchScope(const ComputeBatchScope&) = delete;
        ComputeBatchScope& operator=(const ComputeBatchScope&) = delete;
    private:
        Scheduler& scheduler;
    };
    void EndCompute();
    // Only for owned, disjoint buffer ranges: immutable uploads and compute/blit
    // state, never render-produced inputs or storage still read by a render pass.
    // General transfers and rendering keep the default all-stage dependency.
    // A queued-only encoder boundary is enqueued without a CPU drain; direct
    // owner encoding still synchronizes before accessing native encoder state.
    class BufferComputeScope final {
    public:
        explicit BufferComputeScope(Scheduler& scheduler);
        ~BufferComputeScope();
        BufferComputeScope(const BufferComputeScope&) = delete;
        BufferComputeScope& operator=(const BufferComputeScope&) = delete;
    private:
        Scheduler& scheduler;
        MTLStages previous;
    };
    void EndEncoding(std::source_location source = std::source_location::current());
    void UseResource(id<MTLResource> resource);
    void KeepAlive(id object);
    // Validate GPU-produced status on the owner thread after its timeline retires.
    // The callback owns its readback storage; normal encoding never waits for it.
    void CheckAfterCompletion(std::function<void()> check);

    // Draw/dispatch commands snapshot bindings; the table can then be rebound.
    id<MTL4ArgumentTable> MakeArguments(std::span<const BufferBinding> buffers,
                                       std::span<const TextureBinding> textures = {},
                                       std::span<const SamplerBinding> samplers = {});

    // Rebind a dedicated table's complete fixed buffer-only layout. Keep distinct
    // tables when multiple logical argument sets await encoding at the same time.
    void RebindBufferArguments(id<MTL4ArgumentTable> table,
                               std::span<const BufferBinding> buffers);

    // Replace a variable layout, clearing slots omitted from the new binding set.
    // The returned table is for encoder binding only; mutate it through this API.
    // Encode all consumers before updating this same logical argument set again.
    id<MTL4ArgumentTable> UpdateArguments(ReusableArgumentTable& arguments,
                                         std::span<const BufferBinding> buffers,
                                         std::span<const TextureBinding> textures = {},
                                         std::span<const SamplerBinding> samplers = {});

    // A drawable is waited on before commit, signaled after commit, then presented.
    std::uint64_t Submit(id<MTLDrawable> drawable = nil, double minimum_present_duration = 0.0);
    // Flush guest work when presentation did not already submit it. An idle
    // scheduler still checks errors without creating an empty GPU submission.
    void SubmitPending();
    // Let guest work reach an already busy queue before the host waits for a
    // drawable. Keep idle/drained queues in the single presentation submission.
    // Completed resources are always retired. A ready prefetched image cannot
    // block acquisition and needs no separate submission of pending guest work.
    bool SubmitPendingBeforeDrawable(bool may_wait = true);
    // Opt-in host diagnostics distinguish CPU upload reuse from command-slot
    // reuse and genuine result reads. The reason does not change ordering.
    enum class WaitReason : unsigned { Explicit, CommandSlot, UploadPage, Readback, Drain, Count };
    static constexpr std::size_t SubmissionSlotCount = 4;
    void Wait(std::uint64_t tick, WaitReason reason = WaitReason::Explicit);
    void Finish(WaitReason reason = WaitReason::Drain,
                std::source_location source = std::source_location::current());
    void ReleaseThread();
    // Actual frontend frame boundaries, independent of submission/retirement ticks.
    void EndProfileFrame();
    bool ProfilingEnabled() const noexcept;
    // Opt-in exclusive host intervals at Metal entry points. Nested scopes are
    // charged only to their outer root, never summed twice. Disabled scopes
    // read no clocks, allocate nothing and do not claim scheduler ownership.
    enum class HostPhase : unsigned { Draw, Transfer, Coherency, Present, State, Count };
    class HostPhaseScope final {
    public:
        HostPhaseScope(Scheduler& scheduler, HostPhase phase) {
            if (scheduler.profile_host_phases) Begin(scheduler, phase);
        }
        ~HostPhaseScope() { if (owner) End(); }
        HostPhaseScope(const HostPhaseScope&) = delete;
        HostPhaseScope& operator=(const HostPhaseScope&) = delete;
    private:
        void Begin(Scheduler& scheduler, HostPhase phase);
        void End() noexcept;
        Scheduler* owner{};
        std::int64_t begin_host_ns{};
        std::int64_t begin_cpu_ns{-1};
        unsigned phase_index{};
        bool root{};
    };
    void MarkDiagnosticDraw();
    struct DrawableAcquireProfile {
        std::int64_t begin_host_ns{};
        std::uint64_t frame{}, tick{}, completed_tick{};
        bool commands_pending{};
    };
    // Opt-in host intervals correlate drawable waits with existing GPU frame/tick
    // records. Disabled profiling reads no clocks and creates no output files.
    DrawableAcquireProfile BeginProfileDrawableAcquire();
    void EndProfileDrawableAcquire(const DrawableAcquireProfile& begin,
                                   unsigned window_index, bool available);
    std::uint64_t CurrentFrame() const noexcept { return current_frame; }
    bool FinishNoThrow() noexcept;
    bool IsComplete(std::uint64_t tick) const;
    std::uint64_t LastUseTick(id<MTLResource> resource) const;
    NSUInteger CurrentRetainedObjectCount() const;
    // Successful reusable-table publications, for native command diagnostics.
    std::uint64_t ArgumentUpdateCount() const;
    std::uint64_t EncoderCount() const;
    std::uint64_t CurrentTick() const noexcept {
        return next_tick;
    }

private:
    friend class TransferRuntime;
    // TransferRuntime validates ranges/formats before publishing these owned
    // values. No caller encoder, argument table or borrowed storage is queued.
    struct TransferCommand {
        enum class Kind { CopyBuffer, FillBuffer, CopyTexture, Upload, Download, GenerateMipmaps };
        Kind kind;
        id<MTLBuffer> source_buffer{}, destination_buffer{};
        id<MTLTexture> source_texture{}, destination_texture{};
        NSUInteger source_offset{}, destination_offset{}, size{};
        NSUInteger source_slice{}, source_level{}, destination_slice{}, destination_level{};
        MTLOrigin source_origin{}, destination_origin{};
        MTLSize extent{};
        NSUInteger bytes_per_row{}, bytes_per_image{};
        MTLBlitOption options{MTLBlitOptionNone};
        std::uint8_t value{};
    };
    void QueueTransfer(const TransferCommand& transfer);
    void QueueClear(MTL4RenderPassDescriptor* descriptor);
    void QueueComputeInternal(std::span<const ComputeDispatch> dispatches,
                              std::span<const ComputeBufferCopy> copies,
                              NSString* label, bool append_current,
                              std::span<const TransferCommand> transfers);
    friend struct SchedulerTestAccess;
    struct CompletionState;
    struct ProfilingState;
    struct EncodingWorker;
    std::unique_ptr<EncodingWorker> encoding_worker;
    std::function<void(const SubmissionTiming&)> submission_observer;
    bool queued_render_open{};
    bool queued_render_buffer_readonly{};
    std::optional<std::array<std::uint64_t,128>> queued_render_key;
    std::uint64_t queued_draws{};
    std::uint64_t queued_dispatches{};
    std::uint64_t buffer_readonly_render_passes{}, conservative_render_passes{};
    bool compute_queued{}; // Worker owns the current batched compute encoder until drained.
    bool queued_compute_open{}, queued_compute_render_dependency{};
    void CloseQueuedRender();
    void CloseQueuedCompute();
    void DrainEncodingWorker(std::source_location source = std::source_location::current());
    struct Slot {
        id<MTL4CommandAllocator> allocator;
        id<MTL4CommandBuffer> commands;
        id<MTLResidencySet> residency;
        NSHashTable* retained;
        // Entries are valid only while retained owns the corresponding resources.
        std::unordered_set<const void*> used_resources;
        std::vector<std::shared_future<id<MTLRenderPipelineState>>> pending_pipelines;
        std::vector<std::shared_future<id<MTLComputePipelineState>>> pending_compute_pipelines;
        std::uint64_t tick{};
    };

    void BeginCommands();
    void ValidateQueuedRender(const ImmutableRenderPass& pass, const RenderCommand& command,
        std::span<const BufferBinding> buffers, std::span<const TextureBinding> textures,
        std::span<const SamplerBinding> samplers, std::span<const MeshRenderChunk> chunks) const;
    // Internal batch work only, after BeginCommands validated the owner/errors
    // and selected a live submission slot. Public entry points keep that gate.
    void RetainObject(id object);
    void RegisterResource(id<MTLResource> resource);
    void CheckThread() const;
    void CheckError() const;
    void BeginProfile();
    void EndProfile(NSString* label);
    void ResolveProfile(std::size_t index);

    id<MTLDevice> device;
    id<MTL4CommandQueue> queue;
    id<MTLSharedEvent> timeline;
    id<MTL4CommandEncoder> encoder;
    bool compute_encoder{};
    bool compute_render_dependency{};
    bool encoder_buffer_readonly{};
    unsigned compute_batch_depth{};
    std::optional<std::array<std::uint64_t, 128>> render_key;
    MTLStages compute_producers{MTLStageAll};
    NSMapTable<id<MTLResource>, NSNumber*>* resource_ticks;
    std::array<Slot, SubmissionSlotCount> slots{};
    std::shared_ptr<CompletionState> completion;
    std::unique_ptr<ProfilingState> profiling;
    bool profile_host_phases{};
    bool profile_drains{};
    mutable std::mutex owner_mutex;
    mutable std::thread::id owner;
    std::size_t slot_index{};
    std::uint64_t next_tick{1};
    std::uint64_t current_frame{};
    std::uint64_t argument_updates{};
    std::uint64_t encoder_count{};
    bool recording{};
    struct PendingCheck { std::uint64_t tick; std::function<void()> check; };
    mutable std::deque<PendingCheck> pending_checks;
    mutable std::exception_ptr validation_error;
    std::exception_ptr scope_error;
};

} // namespace Metal
