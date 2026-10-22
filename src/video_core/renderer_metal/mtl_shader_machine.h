// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "mtl_stream_buffer.h"
#include "mtl_vertex_fetch.h"
#include "mtl_geometry.h"
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <bitset>
#include <optional>

namespace Metal {
class NativeVertexInputSnapshot;
class ShaderMachineError final : public std::runtime_error {
public:
    ShaderMachineError(std::uint32_t code_,std::uint32_t instruction)
        :std::runtime_error("Native PICA shader machine error "+std::to_string(code_)+
                            " at instruction "+std::to_string(instruction)),code{code_} {}
    std::uint32_t Code() const { return code; }
private:
    std::uint32_t code;
};
struct ShaderMachineProgram {
    std::array<std::uint32_t,4096> code{};
    std::array<std::uint32_t,4096> swizzle{};
    std::uint32_t entry{};
    std::uint32_t output_mask{0xffff};
    bool allow_emission{true};
    bool operator==(const ShaderMachineProgram&) const = default;
};
// Only the shader machine can seal one of its owned, immutable program copies.
// A const shared_ptr supplied by a caller alone cannot establish immutability.
class NativeProgramSnapshot final {
public:
    NativeProgramSnapshot(const NativeProgramSnapshot&) = default;
    NativeProgramSnapshot(NativeProgramSnapshot&&) = default;
private:
    friend class ShaderMachineRuntime;
    friend class ShaderRuntime;
    explicit NativeProgramSnapshot(std::shared_ptr<const ShaderMachineProgram> source_)
        : source{std::move(source_)} {}
    std::shared_ptr<const ShaderMachineProgram> source;
};
// Call-bound source view only. Cached programs and GPU instructions retain their
// own complete snapshots; no borrowed bank is stored or used by queued GPU work.
struct ShaderMachineProgramView {
    const std::array<std::uint32_t,4096>& code;
    const std::array<std::uint32_t,4096>& swizzle;
    std::uint32_t entry;
    std::uint32_t output_mask;
    bool allow_emission;
    ShaderMachineProgramView(const ShaderMachineProgram& program)
        :code{program.code},swizzle{program.swizzle},entry{program.entry},
         output_mask{program.output_mask},allow_emission{program.allow_emission} {}
    ShaderMachineProgramView(const std::array<std::uint32_t,4096>& code_,
        const std::array<std::uint32_t,4096>& swizzle_,std::uint32_t entry_,
        std::uint32_t output_mask_,bool allow_emission_)
        :code{code_},swizzle{swizzle_},entry{entry_},output_mask{output_mask_},
         allow_emission{allow_emission_} {}
    bool Matches(const ShaderMachineProgram& program) const {
        return entry==program.entry && output_mask==program.output_mask &&
            allow_emission==program.allow_emission && code==program.code && swizzle==program.swizzle;
    }
};
struct ShaderMachineUniforms {
    std::array<std::array<float,4>,96> f{};
    std::array<std::array<std::uint32_t,4>,4> i{};
    std::uint32_t b{};
    std::array<std::uint32_t,3> padding{};
};
static_assert(sizeof(ShaderMachineUniforms)==1616);
struct ShaderMachineEmission {
    VertexRegisters attributes{};
    std::uint32_t winding{};
    std::array<std::uint32_t,3> padding{};
};
static_assert(sizeof(ShaderMachineEmission)==272);
struct GeometryEmissionConfig {
    std::array<std::uint32_t,7> semantics{};
    std::uint32_t attribute_count{};
    std::array<float,4> clip_coefficient{};
    std::uint32_t flip_viewport{};
    std::uint32_t enable_clip{};
    std::uint32_t fix_quaternions{};
    std::uint32_t honor_winding{1};
};
static_assert(sizeof(GeometryEmissionConfig)==64);
class ShaderMachineState final {
    friend class ShaderMachineRuntime;
public:
    explicit ShaderMachineState(const Device& device);
    void Reset() { deferred.reset();reset=true; }
    bool HasDeferredVertexBatch() const { return bool(deferred); }
private:
    id<MTLBuffer> storage;
    struct DeferredBatch;
    // Saving is logically const; materialization changes only cached GPU data
    // and these mutable bookkeeping fields, never the represented guest state.
    mutable std::shared_ptr<const DeferredBatch> deferred;
    mutable bool reset{true};
};
struct ShaderMachineResult {
    BufferSlice output;
    BufferSlice emissions;
    std::uint32_t emission_count{};
};
// Only the native encoder can publish this proof. It binds a statically proved
// physical emission reservation to the exact GPU result and status ranges.
class NativeEmissionStorage final {
    friend class ShaderMachineRuntime;
    NativeEmissionStorage() = default;
    BufferSlice emissions, status;
    std::uint32_t logical_capacity{}, storage_capacity{};
};
// Immutable inputs and separate status storage keep a queued GS retry independent
// of later input transport. A failed GS prevents subsequent guarded state writes.
struct ShaderMachinePendingResult {
    ShaderMachineResult result;
    BufferSlice status,uniforms,input,abort;
    std::shared_ptr<const ShaderMachineProgram> program;
    id<MTLBuffer> unit;
    std::uint32_t input_mask{},capacity{},instruction_limit{};
    bool needs_reset{};
    std::uint64_t tick{};
    std::shared_ptr<const NativeEmissionStorage> storage_proof;
};
struct ShaderMachineVertexBatchResult {
    BufferSlice output;
    BufferSlice captured;
};
struct ShaderMachineFetchedVertexBatchResult {
    BufferSlice input, output, captured;
};
struct ShaderMachineSnapshot {
    std::array<std::byte,1824> history{};
    bool needs_reset{true};
    bool operator==(const ShaderMachineSnapshot&) const = default;
};
class ShaderMachineRuntime final {
public:
    enum class VertexExecution { Ordered, Independent, MergeState, DependencyPasses, PrivateRecurrences };
    // Conservative compiler PC domain across every input and control uniform.
    // This examines instructions/stacks only; no vertex arithmetic runs on the host.
    static std::bitset<4096> ReachableVertexInstructions(const ShaderMachineProgram& program);
    static VertexExecution AnalyzeVertices(const ShaderMachineProgram& program,
                                          const ShaderMachineUniforms& uniforms={},
                                          std::uint32_t* unproven_instruction=nullptr,
                                          std::uint32_t* dependency_passes=nullptr,
                                          std::array<std::uint32_t,2>* recurrence_mask=nullptr,
                                          std::uint32_t* recurrence_capacity=nullptr,
                                          std::array<std::uint32_t,5>* always_written=nullptr,
                                          std::array<std::uint32_t,5>* possibly_written=nullptr,
                                          const GeometryEmissionConfig* native_reset_capture=nullptr);
    // A proof of invocation independence, never an assumption based on a game/title.
    static bool HasIndependentVertices(const ShaderMachineProgram& program,
                                       const ShaderMachineUniforms& uniforms={},
                                       std::uint32_t* unproven_instruction=nullptr);
    // Both compiler proofs are required: independent observed output and native
    // structured stack semantics. No title, float uniform or sampled values.
    static bool HasNativeResetControlFlow(const ShaderMachineProgram& program);
    // Conservative END-terminated IF/LOOP-only instruction bound under these
    // integer controls. No guest float arithmetic or data-dependent assumptions.
    // Unsupported flow, malformed structure and bounds above1M return no proof.
    static std::optional<std::uint32_t> NativeVertexInstructionBound(
        const ShaderMachineProgram& program,const ShaderMachineUniforms& uniforms);
    // No arithmetic/uniform evaluation: an entry..END path without branches,
    // calls or loops can execute each EMIT at most once (at most three vertices).
    static std::optional<std::uint32_t> LinearGeometryStorageBound(const ShaderMachineProgramView& program);
    bool CanCaptureNativeReset(const ShaderMachineProgramView& program,
                               const ShaderMachineUniforms& uniforms,
                               const GeometryEmissionConfig& config);
    ShaderMachineRuntime(Device& device,Scheduler& scheduler);
    ~ShaderMachineRuntime();
    ShaderMachineResult Run(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const ShaderMachineUniforms& uniforms,const BufferSlice& input,
                            std::uint32_t input_mask=0xffff,std::uint32_t emission_capacity=192,
                            std::uint32_t instruction_limit=1000000);
    ShaderMachineResult RunWithUniforms(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const BufferSlice& uniforms,const BufferSlice& input,
                            std::uint32_t input_mask=0xffff,std::uint32_t emission_capacity=192,
                            std::uint32_t instruction_limit=1000000);
    // Queue the exact scalar VS kernel without a CPU completion round trip.
    // Output and retained state remain GPU-ordered; validation is checked before
    // storage recycling/CPU readback. Emission programs require synchronous Run.
    BufferSlice RunVertexDeferred(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const ShaderMachineUniforms& uniforms,const BufferSlice& input,
                            std::uint32_t input_mask=0xffff,std::uint32_t instruction_limit=1000000);
    BufferSlice BeginGeometryBatch();
    ShaderMachinePendingResult QueueGeometry(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const BufferSlice& uniforms,const BufferSlice& input,const BufferSlice& abort,
                            std::uint32_t input_mask,std::uint32_t emission_capacity=192);
    ShaderMachinePendingResult RetryGeometry(ShaderMachineState& state,const ShaderMachinePendingResult& pending,
                            std::uint32_t emission_capacity);
    std::array<std::uint32_t,4> PendingStatus(const ShaderMachinePendingResult& pending) const;
    ShaderMachineResult ResolvePending(const ShaderMachinePendingResult& pending) const;
    void RestoreFailedGeometryReset(ShaderMachineState& state,const ShaderMachinePendingResult& pending);
    // Independent invocations run in parallel; dependencies execute in submission order
    // on the GPU. Both preserve temporary/output banks and the circular post-VS cache.
    BufferSlice RunVertexBatch(ShaderMachineState& state,const ShaderMachineProgramView& program,
                              const ShaderMachineUniforms& uniforms,const BufferSlice& input,
                              std::uint32_t vertex_count,std::uint32_t element_count,
                              std::uint32_t vertex_base,std::uint32_t first_vertex,
                              const BufferSlice& indices={},std::uint32_t index_size=0,
                              std::uint32_t instruction_limit=1000000,bool defer_validation=false);
    // Encode the unchanged VS and output conversion kernels together. Both private
    // slices retain their storage independently; the complete shader state survives.
    ShaderMachineVertexBatchResult RunVertexBatchAndCapture(ShaderMachineState& state,
                              const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,
                              const BufferSlice& input,const GeometryEmissionConfig& config,
                              std::uint32_t vertex_count,std::uint32_t element_count,
                              std::uint32_t vertex_base,std::uint32_t first_vertex,
                              const BufferSlice& indices={},std::uint32_t index_size=0,
                              std::uint32_t instruction_limit=1000000,bool defer_validation=false);
    ShaderMachineFetchedVertexBatchResult RunFetchedVertexBatchAndCapture(
        VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
        const PicaVertexLayout& layout,ShaderMachineState& state,
        const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,
        const GeometryEmissionConfig& config,std::uint32_t vertex_count,std::uint32_t element_count,
        std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices={},
        std::uint32_t index_size=0,std::uint32_t instruction_limit=1000000,bool defer_validation=false);
    AssembledVertices ConvertEmissions(const ShaderMachineResult& emissions,
                                      const GeometryEmissionConfig& config);
    // Complete independent lists fold semantic conversion, winding/quaternion
    // repair and canonical primitive history into one native GPU pass.
    AssembledVertices ConvertEmissionList(const ShaderMachineResult& emissions,
                                        const GeometryEmissionConfig& config,PrimitiveStream& stream);
    // Consume the GPU status/count directly; no CPU count read or completion.
    AssembledVertices ConvertEmissionListIndirect(const ShaderMachinePendingResult& pending,
        const GeometryEmissionConfig& config,PrimitiveStream& stream,const BufferSlice& destination={});
    void ValidateIndirectResult(const AssembledVertices& vertices) const;
    BufferSlice CaptureVertex(const BufferSlice& output,std::uint32_t mask,
                              const GeometryEmissionConfig& config);
    BufferSlice CaptureVertices(const BufferSlice& output,std::uint32_t mask,
                               const GeometryEmissionConfig& config,std::uint32_t count);
    // Snapshot a fresh VS batch for exact GPU materialization before SaveState or
    // any inherited-state invocation. Every borrowed input/index byte is copied
    // into owned private GPU storage now, ahead of later queued GPU writes.
    // Callers still respect the GPU read lifetime before changing shared CPU bytes.
    // Reset/LoadState discard pending work. No guest arithmetic runs on the CPU.
    void DeferResetVertexBatch(ShaderMachineState& state,const ShaderMachineProgramView& program,
                              const ShaderMachineUniforms& uniforms,const BufferSlice& input,
                              std::uint32_t vertex_count,std::uint32_t element_count,
                              std::uint32_t vertex_base,std::uint32_t first_vertex,
                              const BufferSlice& indices={},std::uint32_t index_size=0,
                              std::uint32_t instruction_limit=1000000);
    // Fetch directly into owned private deferred-state storage, then capture in
    // the same encoder. Only captured vertices escape; saved canonical inputs
    // are never exposed as mutable caller buffers. Index bytes remain copied.
    BufferSlice RunFetchedNativeResetAndCapture(VertexFetchRuntime& fetch,VertexInputState& input_state,
        std::span<const BufferSlice> inputs,const PicaVertexLayout& layout,ShaderMachineState& state,
        const ShaderMachineProgramView& source,const ShaderMachineUniforms& uniforms,
        const GeometryEmissionConfig& config,GeometryRuntime& geometry,const VertexCaptureProgram& native,
        std::span<const BufferBinding> bindings,std::uint32_t vertex_count,std::uint32_t element_count,
        std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices={},
        std::uint32_t index_size=0,std::uint32_t instruction_limit=1000000);
    ShaderMachineSnapshot SaveState(const ShaderMachineState& state);
    static void ValidateState(const ShaderMachineSnapshot& snapshot);
    void LoadState(ShaderMachineState& state,const ShaderMachineSnapshot& snapshot);
private:
    friend class GeometryProgramRuntime;
    // Internal transport only: snapshot before the GS dispatch, then set b15
    // on the live uniform bank in the same encoder. Retry still uses the snapshot.
    struct GeometryCapture {
        std::array<id<MTLBuffer>,3> sources;
        BufferSlice history;
        id<MTLComputePipelineState> invoked_pipeline;
        BufferBinding invoked_binding;
        // Private streaming storage; its GPU consumers must precede reuse.
        BufferSlice emissions;
        bool compact{};
    };
    ShaderMachinePendingResult QueueGeometryImpl(ShaderMachineState& state,const ShaderMachineProgramView& program,
        const BufferSlice& uniforms,const BufferSlice& input,const BufferSlice& abort,std::uint32_t input_mask,
        std::uint32_t emission_capacity,const GeometryCapture* capture);
    AssembledVertices ConvertEmissionsImpl(const ShaderMachineResult& emissions,
        const GeometryEmissionConfig& config,PrimitiveStream* stream);
    std::uint32_t EmissionStorageCapacity(const ShaderMachinePendingResult& pending) const;
    bool ProvesGeometryVertexCompletion(const ShaderMachineProgramView& program,
                                       const ShaderMachineUniforms& uniforms);
    // Raw per-element VS outputs for GS consumers, with the same fetched FIFO
    // plan as the captured batch. Only fully proved, terminating VS programs
    // may move ahead of earlier GS/render consumers.
    BufferSlice RunFetchedGeometryVertices(VertexFetchRuntime& fetch,VertexInputState& input_state,
        std::span<const BufferSlice> inputs,const PicaVertexLayout& layout,ShaderMachineState& state,
        const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,
        std::uint32_t vertex_count,std::uint32_t element_count,std::uint32_t vertex_base,
        std::uint32_t first_vertex,const BufferSlice& indices={},std::uint32_t index_size=0);
    ShaderMachinePendingResult EncodeScalar(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const BufferSlice& uniforms,const BufferSlice& input,std::uint32_t input_mask,
                            std::uint32_t emission_capacity,std::uint32_t instruction_limit,const BufferSlice& abort={},
                            const GeometryCapture* capture=nullptr,bool specialize=false);
    ShaderMachineResult RunWithUniformsImpl(ShaderMachineState& state,const ShaderMachineProgramView& program,
                            const BufferSlice& uniforms,const BufferSlice& input,std::uint32_t input_mask,
                            std::uint32_t emission_capacity,std::uint32_t instruction_limit,bool defer_vertex);
    friend class RasterizerMetal;
    friend struct InputReferenceTestAccess;
    // One draw owns this proved program/uniform snapshot. No borrowed source or
    // cache-node pointer survives selection, and publication consumes it once.
    class NativeResetSelection final {
    public:
        NativeResetSelection(NativeResetSelection&&) = default;
        NativeResetSelection& operator=(NativeResetSelection&&) = default;
    private:
        friend class ShaderMachineRuntime;
        NativeResetSelection(const ShaderMachineRuntime* owner_,
            std::shared_ptr<ShaderMachineState::DeferredBatch> pending_)
            : owner{owner_}, pending{std::move(pending_)} {}
        const ShaderMachineRuntime* owner;
        std::shared_ptr<ShaderMachineState::DeferredBatch> pending;
    };
    std::optional<NativeResetSelection> SelectNativeReset(const ShaderMachineProgramView& program,
        const ShaderMachineUniforms& uniforms, const GeometryEmissionConfig& config);
    NativeProgramSnapshot NativeProgram(const NativeResetSelection& selection) const;
    void DeferNativeReferenceBatch(ShaderMachineState& state, NativeResetSelection&& selection,
        std::shared_ptr<const NativeVertexInputSnapshot> snapshot, std::uint32_t vertex_count,
        std::uint32_t element_count, std::uint32_t vertex_base, std::uint32_t first_vertex,
        std::uint32_t index_size, std::uint32_t instruction_limit = 1000000);
    // Only raw/captured outputs escape this private transport. The canonical
    // fetched inputs remain immutable and leased until exact full-state replay.
    ShaderMachineVertexBatchResult RunOwnedFetchedVertexBatchAndCapture(
        VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
        const PicaVertexLayout& layout,ShaderMachineState& state,
        const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,
        const GeometryEmissionConfig& config,std::uint32_t vertex_count,std::uint32_t element_count,
        std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices={},
        std::uint32_t index_size=0,std::uint32_t instruction_limit=1000000,bool defer_validation=true);
    ShaderMachineFetchedVertexBatchResult RunFetchedVertexBatchAndCaptureImpl(
        VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
        const PicaVertexLayout& layout,ShaderMachineState& state,
        const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,
        const GeometryEmissionConfig& config,std::uint32_t vertex_count,std::uint32_t element_count,
        std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices,
        std::uint32_t index_size,std::uint32_t instruction_limit,bool defer_validation,bool owned_input);
    std::unique_ptr<InputReferenceRuntime> input_reference_runtime;
    BufferSlice RunVertexBatchImpl(ShaderMachineState& state,const ShaderMachineProgramView& program,
                              const ShaderMachineUniforms& uniforms,const BufferSlice& input,
                              std::uint32_t vertex_count,std::uint32_t element_count,
                              std::uint32_t vertex_base,std::uint32_t first_vertex,
                              const BufferSlice& indices,std::uint32_t index_size,
                              std::uint32_t instruction_limit,bool defer_validation,
                              const GeometryEmissionConfig* config,BufferSlice* captured,
                              VertexFetchRuntime* fetch_runtime=nullptr,
                              VertexInputState* fetch_state=nullptr,
                              VertexFetchRuntime::PreparedPicaFetch* prepared_fetch=nullptr,
                              bool generic_geometry=false,bool owned_input=false);
    void MaterializeDeferred(const ShaderMachineState& state);
    struct CaptureState;
    std::shared_ptr<CaptureState> capture;
    struct VertexProof {
        VertexExecution execution{VertexExecution::Ordered};
        std::uint32_t passes{},unproven{},instruction_bound{};
        bool profile_dumped{};
        std::array<std::uint32_t,2> recurrence_mask{};
        std::uint32_t recurrence_capacity{};
        // 64 output lanes, 64 temporary lanes, three addresses and two flags.
        std::array<std::uint32_t,5> always_written{},possibly_written{};
    };
    struct BatchPipelines {
        std::shared_future<id<MTLComputePipelineState>> vertex, finalize, simd_finalize;
    };
    struct CachedProgram {
        ShaderMachineProgram source;
        id<MTLBuffer> instructions;
        std::map<std::array<std::uint32_t,17>,VertexProof> control_proofs;
        // Five execution classes plus the bounded threadgroup dependency variant.
        std::array<BatchPipelines,6> vertex_pipelines{};
        std::array<BatchPipelines,6> reset_vertex_pipelines{};
        BatchPipelines owned_small_vertex_pipeline{}, owned_bounded_vertex_pipeline{}, reset_bounded_ordered_vertex_pipeline{};
        // Arithmetic source is fixed per program; masks are control-dependent.
        std::map<std::array<std::uint32_t,10>,BatchPipelines> owned_simd_vertex_pipelines;
        std::map<std::array<std::uint32_t,2>,BatchPipelines> recurrence_pipelines;
        std::map<std::array<std::uint32_t,2>,BatchPipelines> reset_recurrence_pipelines;
        std::shared_ptr<const ShaderMachineProgram> retained_source;
        std::optional<bool> native_reset_control_flow;
        std::map<std::pair<std::array<std::uint32_t,17>,std::array<std::uint32_t,8>>,bool> native_reset_proofs;
        std::shared_future<id<MTLComputePipelineState>> queued_scalar_pipeline{};
    };
    struct PrivateBatchBuffer {
        id<MTLBuffer> buffer;
        std::shared_ptr<void> lease;
    };
    // GPU-only storage: queue barriers order reuse after encoded consumers.
    // Outstanding CPU slices prevent a buffer from being reacquired.
    std::array<PrivateBatchBuffer,3> batch_buffers{};
    BufferSlice AcquireBatchStorage(NSUInteger size,id<MTLBuffer> avoid_a=nil,id<MTLBuffer> avoid_b=nil,
                                  std::span<const BufferSlice> avoid_inputs={},
                                  std::span<const BufferBinding> avoid_bindings={});
    CachedProgram& GetProgram(const ShaderMachineProgramView& program);
    // A positive capture query may return null: its unused raw proof is not cached.
    // Ordinary raw-only callers always receive the complete retained proof.
    VertexProof* GetVertexProof(CachedProgram& program, const ShaderMachineUniforms& uniforms,
        const GeometryEmissionConfig* native_reset_capture=nullptr,bool* capture_independent=nullptr);
    CachedProgram* FindNativeResetProgram(const ShaderMachineProgramView& program,
        const ShaderMachineUniforms& uniforms, const GeometryEmissionConfig& config);
    std::unordered_map<std::uint64_t,std::vector<CachedProgram>> program_cache;
    std::size_t cached_program_count{};
    std::uint64_t recent_program_hash{};
    bool recent_program_repeated{};
    bool profile_vertices{},profile_vertices_bounded{};
    Device& device;
    Scheduler& scheduler;
    StreamBuffer parameters;
    SharedUploadPool scalar_outputs;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipeline;
    id<MTLComputePipelineState> merge_pipeline;
    id<MTLComputePipelineState> resolve_pipeline;
    id<MTLComputePipelineState> plan_pipeline;
    id<MTLComputePipelineState> prefix_pipeline;
    id<MTLComputePipelineState> emission_pipeline;
    id<MTLComputePipelineState> emission_count_pipeline;
    id<MTLComputePipelineState> vertex_pipeline;
    // GS raw batches must not compile a new specialized library mid-race.
    // These native GPU kernels share the ordinary batch ABI and are prepared
    // once during runtime construction, before game execution.
    std::array<id<MTLComputePipelineState>,6> geometry_vertex_pipelines{};
    id<MTLComputePipelineState> geometry_finalize_pipeline,geometry_simd_finalize_pipeline;
    id<MTL4ArgumentTable> capture_arguments;
    id<MTLBuffer> empty_input;
    // Independent complete three-buffer layouts for the two conversion kernels.
    id<MTL4ArgumentTable> emission_arguments;
    ReusableArgumentTable emission_count_arguments,indirect_emission_arguments;
};
}
