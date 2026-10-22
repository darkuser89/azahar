// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_scheduler.h"
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace Metal {
namespace {
std::int64_t ThreadCpuNanoseconds() noexcept;
constexpr MTLStages RenderStages = MTLStageVertex | MTLStageObject | MTLStageMesh |
                                  MTLStageFragment | MTLStageTile;
constexpr MTLStages AttachmentStages = MTLStageFragment | MTLStageTile;
constexpr MTLStages ComputeStages = MTLStageDispatch | MTLStageBlit |
    MTLStageResourceState | MTLStageAccelerationStructure | MTLStageMachineLearning;

bool BufferDisjointAttachments(MTL4RenderPassDescriptor* pass) {
    if (pass.visibilityResultBuffer || pass.rasterizationRateMap || pass.imageblockSampleLength ||
        pass.threadgroupMemoryLength || pass.supportColorAttachmentMapping) return false;
    const auto disjoint = [](id<MTLTexture> texture) {
        while (texture.parentTexture) texture = texture.parentTexture;
        return !texture || (!texture.buffer && !texture.heap && !texture.iosurface && !texture.isSparse);
    };
    for (NSUInteger n = 0; n < 8; ++n)
        if (!disjoint(pass.colorAttachments[n].texture) ||
            !disjoint(pass.colorAttachments[n].resolveTexture)) return false;
    return disjoint(pass.depthAttachment.texture) && disjoint(pass.depthAttachment.resolveTexture) &&
           disjoint(pass.stencilAttachment.texture) && disjoint(pass.stencilAttachment.resolveTexture);
}
bool BufferRangesMayAlias(id<MTLBuffer> a, NSUInteger ao, NSUInteger an,
                          id<MTLBuffer> b, NSUInteger bo, NSUInteger bn) {
    const auto overlap=[](std::uint64_t x,std::uint64_t xn,std::uint64_t y,std::uint64_t yn) {
        return x<=y ? y-x<xn : x-y<yn;
    };
    if(a==b) return overlap(ao,an,bo,bn);
    if(a.heap && a.heap==b.heap) return true;
    if(overlap(a.gpuAddress+ao,an,b.gpuAddress+bo,bn)) return true;
    // Separate no-copy buffers may alias the same shared host pages.
    if(a.storageMode!=MTLStorageModePrivate && b.storageMode!=MTLStorageModePrivate && a.contents && b.contents)
        return overlap(reinterpret_cast<std::uintptr_t>(a.contents)+ao,an,
                       reinterpret_cast<std::uintptr_t>(b.contents)+bo,bn);
    return false;
}
}

// The owner retains resources and builds immutable jobs. Only this worker
// mutates its native encoder/table. Every direct encoding or submission drains
// it first, so one command buffer is never encoded from two threads at once.
struct Scheduler::EncodingWorker {
    using Job = std::function<void(EncodingWorker&)>;
    std::mutex mutex;
    std::condition_variable work_available, progress;
    struct WorkItem { Job command; bool timed; };
    std::deque<WorkItem> jobs;
    bool active{}, stopping{};
    std::exception_ptr error;
    std::thread thread;
    std::thread::id owner;
    id<MTL4RenderCommandEncoder> encoder;
    id<MTL4ComputeCommandEncoder> compute;
    id<MTL4CommandBuffer> compute_commands;
    id<MTL4CounterHeap> compute_counter_heap;
    NSUInteger compute_counter_index{};
    id<MTL4ArgumentTable> arguments;
    id<MTL4CommandBuffer> commands;
    id<MTL4CounterHeap> counter_heap;
    NSUInteger counter_index{};
    bool buffer_readonly{};
    std::array<float,4> blend_color{};
    std::array<bool,31> occupied_buffers{};
    std::array<bool,128> occupied_textures{};
    std::array<bool,16> occupied_samplers{};
    std::atomic_uint64_t encoded_draws{};
    std::atomic_uint64_t encoded_dispatches{};
    std::atomic_bool separate_thread{};
    std::atomic_uint64_t profiled_cpu_nanoseconds{}, profiled_jobs{};

    explicit EncodingWorker(id<MTLDevice> device) : owner{std::this_thread::get_id()} {
        auto* descriptor = [[MTL4ArgumentTableDescriptor alloc] init];
        descriptor.initializeBindings = YES;
        descriptor.maxBufferBindCount = 31;
        descriptor.maxTextureBindCount = 128;
        descriptor.maxSamplerStateBindCount = 16;
        NSError* failure = nil;
        arguments = [device newArgumentTableWithDescriptor:descriptor error:&failure];
        if (!arguments) throw std::runtime_error("Unable to create Metal4 encoding-worker arguments");
        thread = std::thread{[this] { Run(); }};
    }
    ~EncodingWorker() {
        { std::scoped_lock lock{mutex}; stopping=true; }
        work_available.notify_one();
        if (thread.joinable()) thread.join();
    }
    void Enqueue(Job job, bool timed=false) {
        std::unique_lock lock{mutex};
        progress.wait(lock,[&] {return jobs.size()<256 || error;});
        if (error) std::rethrow_exception(error);
        const bool wake=jobs.empty() && !active;
        jobs.push_back({std::move(job),timed});
        lock.unlock();
        // Admission and drain waiters must not compete with the consumer for
        // a work notification. The active worker already sees appended jobs.
        if (wake) work_available.notify_one();
    }
    void CheckError() {
        std::scoped_lock lock{mutex};
        if (error) std::rethrow_exception(error);
    }
    void Drain() {
        std::unique_lock lock{mutex};
        progress.wait(lock,[&] {return jobs.empty() && !active;});
        if (error) std::rethrow_exception(error);
    }
    void Close() {
        if (!encoder) return;
        // Opaque shaders may write buffers in any render stage. Publish that
        // dependency to every subsequent encoder, including future submissions.
        if (!buffer_readonly)
            [encoder barrierAfterStages:RenderStages beforeQueueStages:MTLStageAll
                      visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding]; encoder=nil;
        if (counter_heap) [commands writeTimestampIntoHeap:counter_heap atIndex:counter_index+1];
        counter_heap=nil; commands=nil;
    }
    void CloseCompute() {
        if (!compute) return;
        [compute barrierAfterStages:ComputeStages beforeQueueStages:MTLStageAll
                  visibilityOptions:MTL4VisibilityOptionDevice];
        [compute endEncoding]; compute=nil;
        if (compute_counter_heap)
            [compute_commands writeTimestampIntoHeap:compute_counter_heap atIndex:compute_counter_index+1];
        compute_counter_heap=nil; compute_commands=nil;
    }
    void Run() {
        pthread_setname_np("Azahar Metal4 encoding");
        separate_thread.store(std::this_thread::get_id()!=owner);
        for (;;) {
            WorkItem job;
            bool failed{}, freed_capacity{};
            {
                std::unique_lock lock{mutex};
                work_available.wait(lock,[&] {return stopping || !jobs.empty();});
                if (jobs.empty() && stopping) break;
                freed_capacity=jobs.size()==256;
                job=std::move(jobs.front());jobs.pop_front();active=true;failed=bool(error);
            }
            if (freed_capacity) progress.notify_one();
            @autoreleasepool {
                const auto begin=job.timed ? ThreadCpuNanoseconds() : -1;
                try { if (!failed) job.command(*this); }
                catch (...) {
                    const auto failure=std::current_exception();
                    try { Close(); } catch (...) {}
                    try { CloseCompute(); } catch (...) {}
                    std::scoped_lock lock{mutex}; if (!error) error=failure;
                }
                const auto end=job.timed ? ThreadCpuNanoseconds() : -1;
                if (begin>=0 && end>=begin) {
                    profiled_cpu_nanoseconds+=std::uint64_t(end-begin);
                    ++profiled_jobs;
                }
                job.command={}; // Release immutable job snapshots inside the worker pool.
            }
            bool idle{};
            { std::scoped_lock lock{mutex}; active=false; idle=jobs.empty() || bool(error); }
            if (idle) progress.notify_all();
        }
        @autoreleasepool { Close(); }
    }
    template <std::size_t N, typename Binding, typename Setter>
    static void Update(std::array<bool,N>& occupied, const std::vector<Binding>& bindings, Setter&& set) {
        std::array<bool,N> next{};
        for (const auto& binding:bindings) next[binding.index]=true;
        for (std::size_t i=0;i<N;++i) if (occupied[i] && !next[i]) set(i,nullptr);
        for (const auto& binding:bindings) set(binding.index,&binding);
        occupied=next;
    }
    void Bind(const std::vector<BufferBinding>& buffers, const std::vector<TextureBinding>& textures,
              const std::vector<SamplerBinding>& samplers) {
        Update(occupied_buffers,buffers,[&](NSUInteger index,const BufferBinding* binding) {
            [arguments setAddress:binding ? binding->buffer.gpuAddress+binding->offset : 0 atIndex:index];
        });
        Update(occupied_textures,textures,[&](NSUInteger index,const TextureBinding* binding) {
            [arguments setTexture:binding ? binding->texture.gpuResourceID : MTLResourceID{} atIndex:index];
        });
        Update(occupied_samplers,samplers,[&](NSUInteger index,const SamplerBinding* binding) {
            [arguments setSamplerState:binding ? binding->sampler.gpuResourceID : MTLResourceID{} atIndex:index];
        });
    }
};
namespace {
std::int64_t ThreadCpuNanoseconds() noexcept {
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return -1;
    return std::int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}
} // namespace

struct Scheduler::CompletionState {
    std::mutex mutex;
    std::string error;
};

// Opt-in encoder-boundary diagnostics. The normal path creates no counter heaps
// and performs no timestamp reads or file I/O. These timings include barriers;
// sampling itself can affect scheduling, so they are not game-FPS benchmarks.
// Host-only mode keeps the frame/submit/wait clocks without GPU counter work.
struct Scheduler::ProfilingState {
    static constexpr NSUInteger Capacity = 4096;
    struct Entry { NSUInteger index; std::string label; std::uint64_t frame; };
    struct Batch {
        id<MTL4CounterHeap> heap;
        std::vector<Entry> entries;
        NSUInteger used{}, dropped{}, retained{};
        NSUInteger allocated_bytes{};
    };
    std::array<Batch, SubmissionSlotCount> batches{};
    struct WaitStats {
        std::uint64_t calls{}, zero_tick{}, completed{}, submits{}, signal_waits{};
        double host_ms{}, submit_ms{}, signal_ms{};
    };
    std::array<WaitStats, static_cast<unsigned>(WaitReason::Count)> waits{};
    struct HostStats {
        std::uint64_t calls{}, cpu_valid_calls{};
        double host_ms{}, cpu_ms{};
    };
    std::array<HostStats, static_cast<unsigned>(HostPhase::Count)> host_phases{};
    unsigned host_depth{};
    unsigned host_phase_index{static_cast<unsigned>(HostPhase::Count)};
    struct DrainSite {
        std::string_view file;
        std::uint_least32_t line{};
        unsigned phase{};
        std::uint64_t calls{}, failures{}, cpu_valid_calls{};
        double host_ms{}, cpu_ms{}, max_host_ms{};
        std::int64_t max_begin_ns{}, max_end_ns{};
        std::size_t pending_render{}, pending_compute{};
    };
    std::array<DrainSite,256> drains{};
    std::size_t drain_count{};
    std::uint64_t drain_dropped{};
    DrainSite* Drain(std::source_location source, unsigned phase) {
        std::string_view file{source.file_name()};
        if (const auto slash=file.find_last_of('/');slash!=std::string_view::npos)file.remove_prefix(slash+1);
        for(std::size_t i=0;i<drain_count;++i)
            if(drains[i].file==file && drains[i].line==source.line() && drains[i].phase==phase)return &drains[i];
        if(drain_count==drains.size()) {++drain_dropped;return nullptr;}
        auto& site=drains[drain_count++];site.file=file;site.line=source.line();site.phase=phase;return &site;
    }
    struct ReadbackSite {
        std::string_view file;
        std::uint_least32_t line{};
        std::uint64_t calls{};
        double finish_ms{};
    };
    // Diagnostic capacity exceeds all current production readback callsites.
    // Fixed storage also keeps nested completion checks from invalidating timers.
    std::array<ReadbackSite, 64> readbacks{};
    std::size_t readback_count{};
    ReadbackSite& Readback(std::source_location source) {
        std::string_view file{source.file_name()};
        if (const auto slash = file.find_last_of('/'); slash != std::string_view::npos)
            file.remove_prefix(slash + 1);
        for (std::size_t i = 0; i < readback_count; ++i)
            if (readbacks[i].file == file && readbacks[i].line == source.line()) return readbacks[i];
        if (readback_count == readbacks.size())
            throw std::length_error("Metal readback callsite profile capacity exceeded");
        auto& site = readbacks[readback_count++];
        site = {file, source.line(), 0, 0};
        return site;
    }
    std::ofstream output, frames, drawables, control, wait_output, readback_output, host_output, cpu_output, worker_output, drain_output;
    EncodingWorkerStats previous_worker{};
    std::filesystem::path trigger;
    std::uint64_t frames_remaining{};
    bool enabled{true}, armed{}, host_only{};
    std::chrono::steady_clock::time_point frame_start{std::chrono::steady_clock::now()};
    std::thread::id frame_cpu_thread{};
    std::int64_t frame_cpu_ns{-1};
    double submit_ms{}, wait_ms{};
    std::uint64_t submit_calls{}, wait_calls{}, encoders{}, captured_draws{};
    std::uint64_t frequency{};
    bool active{};
};

bool Scheduler::ProfilingEnabled() const noexcept {
    return profiling && profiling->enabled;
}

void Scheduler::HostPhaseScope::Begin(Scheduler& scheduler, HostPhase phase) {
    if (!scheduler.ProfilingEnabled()) return;
    {
        std::scoped_lock lock{scheduler.owner_mutex};
        // Metadata-only entry points must not claim an unowned queue during
        // frontend -> emulation handoff merely because diagnostics are enabled.
        if (scheduler.owner == std::thread::id{}) return;
        if (scheduler.owner != std::this_thread::get_id())
            throw std::logic_error("Metal host profile must remain on the command owner");
    }
    if (static_cast<unsigned>(phase) >= static_cast<unsigned>(HostPhase::Count))
        throw std::invalid_argument("Invalid Metal host profile phase");
    auto& profile = *scheduler.profiling;
    if (profile.host_depth == std::numeric_limits<unsigned>::max())
        throw std::overflow_error("Metal host profile nesting overflow");
    owner = &scheduler;
    phase_index = static_cast<unsigned>(phase);
    root = profile.host_depth++ == 0;
    if (root) {
        profile.host_phase_index=phase_index;
        ++profile.host_phases[phase_index].calls;
        begin_host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        begin_cpu_ns = ThreadCpuNanoseconds();
    }
}

void Scheduler::HostPhaseScope::End() noexcept {
    auto& profile = *owner->profiling;
    if (root) {
        const auto cpu = ThreadCpuNanoseconds();
        const auto end = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        auto& phase = profile.host_phases[phase_index];
        phase.host_ms += double(end - begin_host_ns) / 1000000.0;
        if (begin_cpu_ns >= 0 && cpu >= begin_cpu_ns) {
            phase.cpu_ms += double(cpu - begin_cpu_ns) / 1000000.0;
            ++phase.cpu_valid_calls;
        }
    }
    --profile.host_depth;
}

namespace {
struct ProfileCallTimer {
    double* total{};
    std::chrono::steady_clock::time_point start{};
    explicit ProfileCallTimer(double* value) : total{value} {
        if (total) start=std::chrono::steady_clock::now();
    }
    ~ProfileCallTimer() {
        if (total) *total+=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
    }
};
}

void Scheduler::MarkDiagnosticDraw() {
    CheckThread();
    if (ProfilingEnabled()) ++profiling->captured_draws;
}

Scheduler::DrawableAcquireProfile Scheduler::BeginProfileDrawableAcquire() {
    if (!ProfilingEnabled()) return {};
    CheckThread();
    CheckError();
    DrawableAcquireProfile result{};
    result.frame = current_frame;
    result.tick = next_tick;
    result.completed_tick = timeline.signaledValue;
    result.commands_pending = recording;
    result.begin_host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return result;
}

void Scheduler::EndProfileDrawableAcquire(const DrawableAcquireProfile& begin,
                                         unsigned window_index, bool available) {
    if (!ProfilingEnabled()) return;
    const auto end = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    CheckThread();
    CheckError();
    if (window_index > 1 || begin.begin_host_ns <= 0 || end < begin.begin_host_ns ||
        begin.frame != current_frame || begin.tick != next_tick) {
        throw std::invalid_argument("Invalid Metal 4 drawable profile interval");
    }
    profiling->drawables << begin.frame << ',' << window_index << ',' << begin.tick
        << ',' << begin.commands_pending << ',' << begin.completed_tick << ','
        << timeline.signaledValue << ',' << begin.begin_host_ns << ',' << end << ','
        << std::setprecision(12) << double(end - begin.begin_host_ns) / 1000000.0
        << ',' << available << '\n';
    profiling->drawables.flush();
    if (!profiling->drawables) throw std::runtime_error("Unable to write Metal 4 drawable profile");
}

void Scheduler::EndProfileFrame() {
    CheckThread();
    if (profiling && profiling->host_depth)
        throw std::logic_error("Metal host profile scope crosses a frame boundary");
    EndEncoding();
    const auto transition = [&](bool enabled) {
        profiling->control << current_frame << ',' << enabled << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
            << ',' << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << '\n';
        profiling->control.flush();
        if (!profiling->control) throw std::runtime_error("Unable to write Metal 4 profile control transition");
    };
    bool stopped{};
    if (ProfilingEnabled()) {
        const auto cpu = profile_host_phases ? ThreadCpuNanoseconds() : -1;
        const auto now=std::chrono::steady_clock::now();
        const auto ns=[](auto t) {return std::chrono::duration_cast<std::chrono::nanoseconds>(
            t.time_since_epoch()).count();};
        profiling->frames << current_frame << ',' << ns(profiling->frame_start) << ',' << ns(now)
            << ',' << std::setprecision(12)
            << std::chrono::duration<double,std::milli>(now-profiling->frame_start).count()
            << ',' << profiling->submit_ms << ',' << profiling->wait_ms
            << ',' << profiling->submit_calls << ',' << profiling->wait_calls
            << ',' << profiling->encoders << ',' << profiling->captured_draws
            << ',' << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << '\n';
        profiling->frames.flush();
        if (!profiling->frames) throw std::runtime_error("Unable to write Metal 4 frame profile");
        constexpr std::array names{"explicit", "command_slot", "upload_page", "readback", "drain"};
        for (unsigned i = 0; i < profiling->waits.size(); ++i) {
            const auto& value = profiling->waits[i];
            if (!value.calls) continue;
            profiling->wait_output << current_frame << ',' << names[i] << ',' << value.calls
                << ',' << value.zero_tick << ',' << value.completed << ',' << value.submits
                << ',' << value.signal_waits << ',' << std::setprecision(12) << value.host_ms
                << ',' << value.submit_ms << ',' << value.signal_ms << '\n';
        }
        profiling->wait_output.flush();
        if (!profiling->wait_output) throw std::runtime_error("Unable to write Metal 4 wait profile");
        profiling->waits = {};
        for (std::size_t i = 0; i < profiling->readback_count; ++i) {
            const auto& site = profiling->readbacks[i];
            profiling->readback_output << current_frame << ',' << site.file << ',' << site.line
                << ',' << site.calls << ',' << std::setprecision(12) << site.finish_ms << '\n';
        }
        profiling->readback_output.flush();
        if (!profiling->readback_output) throw std::runtime_error("Unable to write Metal 4 readback callsites");
        profiling->readback_count = 0;
        if (profile_host_phases) {
            // EndEncoding drained the native jobs, so these per-frame deltas
            // exclude queued work from before the trigger and never sample a
            // partly encoded frame. Worker CPU clocks exclude compiler waits.
            const auto worker = WorkerStats();
            const auto& previous = profiling->previous_worker;
            profiling->worker_output << current_frame << ','
                << worker.queued_draws-previous.queued_draws << ','
                << worker.encoded_draws-previous.encoded_draws << ','
                << worker.queued_dispatches-previous.queued_dispatches << ','
                << worker.encoded_dispatches-previous.encoded_dispatches << ','
                << worker.profiled_cpu_nanoseconds-previous.profiled_cpu_nanoseconds << ','
                << worker.profiled_jobs-previous.profiled_jobs << ',' << worker.separate_thread << ','
                << worker.buffer_readonly_render_passes-previous.buffer_readonly_render_passes << ','
                << worker.conservative_render_passes-previous.conservative_render_passes << '\n';
            profiling->worker_output.flush();
            if (!profiling->worker_output) throw std::runtime_error("Unable to write Metal encoding-worker CPU profile");
            profiling->previous_worker = worker;
            // Compare only samples from the same owner lifetime. Thread CPU
            // excludes sleeping/descheduling and other compiler/driver threads;
            // it does not by itself explain why the owner was not running.
            const auto thread = std::this_thread::get_id();
            const bool valid = profiling->frame_cpu_thread == thread &&
                profiling->frame_cpu_ns >= 0 && cpu >= profiling->frame_cpu_ns;
            int relative_priority{};
            qos_class_t qos = QOS_CLASS_UNSPECIFIED;
            if (pthread_get_qos_class_np(pthread_self(), &qos, &relative_priority) != 0) {
                qos = QOS_CLASS_UNSPECIFIED;
                relative_priority = 0;
            }
            profiling->cpu_output << current_frame << ',' << ns(profiling->frame_start) << ',' << ns(now)
                << ',' << std::setprecision(12)
                << (valid ? double(cpu - profiling->frame_cpu_ns) / 1000000.0 : -1.0)
                << ',' << valid << ',' << unsigned(qos) << ',' << relative_priority << '\n';
            profiling->cpu_output.flush();
            if (!profiling->cpu_output) throw std::runtime_error("Unable to write Metal thread CPU profile");
            profiling->frame_cpu_thread = thread;
            profiling->frame_cpu_ns = cpu;
            constexpr std::array names{"draw", "transfer", "coherency", "present", "state"};
            for (unsigned i = 0; i < profiling->host_phases.size(); ++i) {
                const auto& value = profiling->host_phases[i];
                if (value.calls) profiling->host_output << current_frame << ',' << names[i] << ','
                    << value.calls << ',' << std::setprecision(12) << value.host_ms
                    << ',' << value.cpu_ms << ',' << value.cpu_valid_calls << '\n';
            }
            profiling->host_output.flush();
            if (!profiling->host_output) throw std::runtime_error("Unable to write Metal host phase profile");
            profiling->host_phases = {};
        }
        profiling->frame_start=now;
        profiling->submit_ms=profiling->wait_ms=0;
        profiling->submit_calls=profiling->wait_calls=profiling->encoders=profiling->captured_draws=0;
        if (profile_drains) {
            constexpr std::array names{"draw","transfer","coherency","present","state","none"};
            for(std::size_t i=0;i<profiling->drain_count;++i) {
                auto& site=profiling->drains[i];
                if(site.calls) profiling->drain_output << current_frame << ',' << site.file << ',' << site.line << ','
                    << names[site.phase] << ',' << site.calls << ',' << site.failures << ',' << std::setprecision(12)
                    << site.host_ms << ',' << site.cpu_ms << ',' << site.cpu_valid_calls << ',' << site.max_host_ms << ','
                    << site.max_begin_ns << ',' << site.max_end_ns << ',' << site.pending_render << ',' << site.pending_compute << ",0\n";
                site=ProfilingState::DrainSite{.file=site.file,.line=site.line,.phase=site.phase};
            }
            if(profiling->drain_dropped) profiling->drain_output << current_frame
                << ",<capacity>,0,none,0,0,0,0,0,0,0,0,0,0," << profiling->drain_dropped << '\n';
            profiling->drain_dropped=0;
            profiling->drain_output.flush();
            if(!profiling->drain_output) throw std::runtime_error("Unable to write Metal worker-drain profile");
        }
        if (profiling->frames_remaining && !--profiling->frames_remaining) {
            profiling->enabled=false; stopped=true;
        }
    }
    ++current_frame;
    if (stopped) transition(false);
    if (profiling && profiling->armed) {
        std::error_code error;
        const bool ready=std::filesystem::exists(profiling->trigger,error);
        if (error) throw std::runtime_error("Unable to inspect Metal 4 profile trigger");
        if (ready) {
            profiling->armed=false; profiling->enabled=true;
            profiling->frame_start=std::chrono::steady_clock::now();
            if (profile_host_phases) {
                profiling->previous_worker = WorkerStats();
                profiling->frame_cpu_ns = ThreadCpuNanoseconds();
                profiling->frame_cpu_thread = std::this_thread::get_id();
            }
            transition(true);
        }
    }
}

Scheduler::Scheduler(const Device& device_)
    : device{device_.Handle()}, completion{std::make_shared<CompletionState>()},
      owner{std::this_thread::get_id()} {
    MTL4CommandQueueDescriptor* queue_descriptor = [[MTL4CommandQueueDescriptor alloc] init];
    queue_descriptor.label = @"Azahar Metal 4 queue";
    NSError* queue_error = nil;
    queue = [device newMTL4CommandQueueWithDescriptor:queue_descriptor error:&queue_error];
    timeline = [device newSharedEvent];
    resource_ticks = [NSMapTable weakToStrongObjectsMapTable];
    if (!queue || !timeline) {
        throw std::runtime_error("Unable to create Metal 4 queue or shared event");
    }
    timeline.label = @"Azahar Metal 4 timeline";
    for (auto& slot : slots) {
        slot.allocator = [device newCommandAllocator];
        slot.commands = [device newCommandBuffer];
        // Resource IDs/addresses do not retain objects. Strong pointer identity
        // keeps each object once even when thousands of commands reuse it.
        slot.retained = [[NSHashTable alloc] initWithOptions:NSPointerFunctionsStrongMemory |
                                                            NSPointerFunctionsObjectPointerPersonality
                                                   capacity:256];
        slot.used_resources.reserve(256);
        MTLResidencySetDescriptor* descriptor = [[MTLResidencySetDescriptor alloc] init];
        descriptor.label = @"Azahar Metal 4 submission resources";
        descriptor.initialCapacity = 256;
        NSError* error = nil;
        slot.residency = [device newResidencySetWithDescriptor:descriptor error:&error];
        if (!slot.allocator || !slot.commands || !slot.residency) {
            throw std::runtime_error("Unable to create Metal 4 submission resources");
        }
    }
    if (const auto* path = std::getenv("AZAHAR_METAL4_PROFILE"); path && *path) {
        profiling = std::make_unique<ProfilingState>();
        if(const auto* mode=std::getenv("AZAHAR_METAL4_PROFILE_DRAINS");mode && *mode) {
            if(std::string_view{mode}!="0" && std::string_view{mode}!="1")
                throw std::invalid_argument("Invalid Metal worker-drain profile mode");
            profile_drains=*mode=='1';
        }
        if(profile_drains) {
            profiling->drain_output.open(std::string(path)+".drains.csv",std::ios::out|std::ios::trunc);
            profiling->drain_output << "frame,file,line,phase,calls,failures,host_ms,thread_cpu_ms,cpu_valid_calls,max_host_ms,max_begin_host_ns,max_end_host_ns,pending_render_psos_at_max,pending_compute_psos_at_max,dropped_sites\n";
            profiling->drain_output.flush();
            if(!profiling->drain_output) throw std::runtime_error("Unable to initialize Metal worker-drain profile");
        }
        if (const auto* mode = std::getenv("AZAHAR_METAL4_PROFILE_HOST_PHASES"); mode && *mode) {
            if (std::string_view{mode} != "0" && std::string_view{mode} != "1")
                throw std::invalid_argument("Invalid Metal host phase profile mode");
            profile_host_phases = std::string_view{mode} == "1";
        }
        if (profile_host_phases) {
            profiling->host_output.open(std::string(path) + ".host-phases.csv", std::ios::out | std::ios::trunc);
            profiling->host_output << "frame,phase,calls,host_ms,thread_cpu_ms,cpu_valid_calls\n";
            profiling->host_output.flush();
            if (!profiling->host_output) throw std::runtime_error("Unable to initialize Metal host phase profile");
            profiling->cpu_output.open(std::string(path) + ".thread-cpu.csv", std::ios::out | std::ios::trunc);
            profiling->cpu_output << "frame,begin_host_ns,end_host_ns,thread_cpu_ms,cpu_valid,qos_class,relative_priority\n";
            profiling->cpu_output.flush();
            if (!profiling->cpu_output) throw std::runtime_error("Unable to initialize Metal thread CPU profile");
            profiling->worker_output.open(std::string(path) + ".worker-cpu.csv", std::ios::out | std::ios::trunc);
            profiling->worker_output << "frame,queued_draws,encoded_draws,queued_dispatches,encoded_dispatches,worker_cpu_ns,profiled_jobs,separate_thread,buffer_readonly_passes,conservative_passes\n";
            profiling->worker_output.flush();
            if (!profiling->worker_output) throw std::runtime_error("Unable to initialize Metal encoding-worker CPU profile");
        }
        if (const auto* mode = std::getenv("AZAHAR_METAL4_PROFILE_HOST_ONLY"); mode && *mode) {
            if (std::string_view{mode} != "0" && std::string_view{mode} != "1")
                throw std::invalid_argument("Invalid Metal 4 host-only profile mode");
            profiling->host_only = *mode == '1';
        }
        if (const auto* trigger=std::getenv("AZAHAR_METAL4_PROFILE_TRIGGER");trigger && *trigger) {
            profiling->trigger=trigger;profiling->armed=true;profiling->enabled=false;
        }
        if (const auto* limit=std::getenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT");limit && *limit) {
            const auto end=limit+std::strlen(limit);
            const auto parsed=std::from_chars(limit,end,profiling->frames_remaining);
            if (parsed.ec!=std::errc{} || parsed.ptr!=end || profiling->frames_remaining>1000000)
                throw std::invalid_argument("Invalid Metal 4 profile frame limit");
        }
        if (profiling->armed && !profiling->frames_remaining)
            throw std::invalid_argument("Triggered Metal 4 profiling requires a finite frame limit");
        profiling->output.open(path, std::ios::out | std::ios::trunc);
        profiling->frames.open(std::string(path) + ".frames.csv", std::ios::out | std::ios::trunc);
        profiling->drawables.open(std::string(path) + ".drawables.csv", std::ios::out | std::ios::trunc);
        profiling->control.open(std::string(path) + ".control.csv", std::ios::out | std::ios::trunc);
        profiling->wait_output.open(std::string(path) + ".waits.csv", std::ios::out | std::ios::trunc);
        profiling->readback_output.open(std::string(path) + ".readbacks.csv", std::ios::out | std::ios::trunc);
        profiling->readback_output << "frame,file,line,calls,finish_host_ms\n";
        profiling->wait_output << "frame,reason,calls,zero_tick_calls,completed_calls,pending_submit_calls,signal_wait_calls,host_ms,submit_host_ms,signal_wait_host_ms\n";
        profiling->control << "frame,enabled,host_ns,wall_ns\n0," << profiling->enabled << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
            << ',' << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << '\n';
        profiling->control.flush();
        profiling->drawables << "frame,window,tick,commands_pending,completed_before,completed_after,begin_host_ns,end_host_ns,acquire_host_ms,available\n";
        profiling->frames << "frame,begin_host_ns,end_host_ns,wall_ms,submit_host_ms,wait_host_ms,submit_calls,wait_calls,encoders,captured_draws,end_wall_ns\n";
        if (!profiling->host_only) profiling->frequency = [device queryTimestampFrequency];
        if (!profiling->output || !profiling->frames || !profiling->drawables || !profiling->control || !profiling->wait_output || !profiling->readback_output ||
            (!profiling->host_only && !profiling->frequency))
            throw std::runtime_error("Unable to initialize requested Metal 4 GPU profiling");
        if (!profiling->host_only) for (auto& batch : profiling->batches) {
            auto* descriptor = [[MTL4CounterHeapDescriptor alloc] init];
            descriptor.type = MTL4CounterHeapTypeTimestamp;
            descriptor.count = ProfilingState::Capacity;
            NSError* error = nil;
            batch.heap = [device newCounterHeapWithDescriptor:descriptor error:&error];
            if (!batch.heap) throw std::runtime_error("Unable to create Metal 4 timestamp heap");
        }
        if (profiling->host_only) profiling->output << "# host_only=1\n";
        else profiling->output << "# gpu_timestamp_frequency_hz=" << profiling->frequency << '\n';
        profiling->output << "tick,encoder,label,start_ticks,end_ticks,gpu_ms,valid,retained_objects,device_allocated_bytes,dropped_encoders,frame\n";
        profiling->output.flush();
    }
}

void Scheduler::BeginProfile() {
    if (profiling->host_only) {
        ++profiling->encoders;
        return;
    }
    auto& batch = profiling->batches[slot_index];
    profiling->active = batch.used + 2 <= ProfilingState::Capacity;
    if (!profiling->active) { ++batch.dropped; return; }
    RetainObject(batch.heap);
    [slots[slot_index].commands writeTimestampIntoHeap:batch.heap atIndex:batch.used];
}

void Scheduler::EndProfile(NSString* label) {
    if (!profiling->active) return;
    auto& batch = profiling->batches[slot_index];
    [slots[slot_index].commands writeTimestampIntoHeap:batch.heap atIndex:batch.used + 1];
    batch.entries.push_back({batch.used, label.UTF8String ? label.UTF8String : "unlabelled", current_frame});
    ++profiling->encoders;
    batch.used += 2;
    profiling->active = false;
}

void Scheduler::ResolveProfile(std::size_t index) {
    auto& batch = profiling->batches[index];
    if (!batch.used) return;
    // Callers have waited on this slot's shared-event signal before CPU resolve
    // or invalidation, including during destruction and ownership handoff.
    NSData* data = [batch.heap resolveCounterRange:NSMakeRange(0, batch.used)];
    if (!data || data.length != batch.used * sizeof(MTL4TimestampHeapEntry))
        throw std::runtime_error("Unable to resolve Metal 4 GPU timestamps");
    const auto* times = static_cast<const MTL4TimestampHeapEntry*>(data.bytes);
    for (std::size_t encoder_index = 0; encoder_index < batch.entries.size(); ++encoder_index) {
        const auto& entry = batch.entries[encoder_index];
        const auto start = times[entry.index].timestamp, end = times[entry.index + 1].timestamp;
        const bool valid = start && end >= start;
        auto label = entry.label;
        // Quote CSV fields independently of shader/debug-label contents.
        for (std::size_t pos = 0; (pos = label.find('"', pos)) != std::string::npos; pos += 2)
            label.insert(pos, 1, '"');
        profiling->output << slots[index].tick << ',' << encoder_index << ",\"" << label << "\","
            << start << ',' << end << ',' << std::setprecision(12)
            << (valid ? double(end - start) * 1000.0 / profiling->frequency : 0.0) << ',' << valid
            << ',' << batch.retained << ',' << batch.allocated_bytes << ',' << batch.dropped
            << ',' << entry.frame << '\n';
    }
    profiling->output.flush();
    if (!profiling->output) throw std::runtime_error("Unable to write Metal 4 GPU profile");
    [batch.heap invalidateCounterRange:NSMakeRange(0, batch.used)];
    batch.entries.clear(); batch.used = batch.dropped = 0;
}

Scheduler::~Scheduler() {
    @autoreleasepool {
        try {
            Finish();
        } catch (const std::exception& error) {
            // Submission callbacks retain their allocations even if waiting fails during teardown.
            NSLog(@"Azahar Metal 4 shutdown: %s", error.what());
            try { DrainEncodingWorker(); } catch (...) {}
            if (recording) {
                [encoder endEncoding];
                encoder = nil;
                [slots[slot_index].commands endCommandBuffer];
            }
        }
    }
}

void Scheduler::CheckThread() const {
    std::scoped_lock lock{owner_mutex};
    if (owner == std::thread::id{}) {
        owner = std::this_thread::get_id();
    }
    if (std::this_thread::get_id() != owner) {
        throw std::logic_error("Metal 4 command encoding must remain on its owning thread");
    }
}

void Scheduler::ReleaseThread() {
    // Submit and retire initialization work before another thread can claim the queue.
    // Finish keeps ordinary submissions bound to the existing owner; only this explicit
    // lifecycle boundary permits migration.
    Finish();
    std::scoped_lock lock{owner_mutex};
    owner = {};
    if (profiling) {
        profiling->frame_cpu_ns = -1;
        profiling->frame_cpu_thread = {};
    }
}

void Scheduler::CheckError() const {
    if (encoding_worker) encoding_worker->CheckError();
    if (scope_error) std::rethrow_exception(scope_error);
    {
        std::scoped_lock lock{completion->mutex};
        if (!completion->error.empty()) throw std::runtime_error(completion->error);
    }
    if (validation_error) std::rethrow_exception(validation_error);
    if (pending_checks.empty()) return;
    const auto completed=timeline.signaledValue;
    while (!pending_checks.empty() && pending_checks.front().tick<=completed) {
        auto check=std::move(pending_checks.front().check);
        pending_checks.pop_front();
        try { check(); }
        catch (...) {
            validation_error=std::current_exception();
            throw;
        }
    }
}

void Scheduler::CheckAfterCompletion(std::function<void()> check) {
    if (!check) throw std::invalid_argument("Missing Metal 4 completion validation");
    BeginCommands();
    pending_checks.push_back({next_tick,std::move(check)});
}

void Scheduler::BeginCommands() {
    CheckThread();
    CheckError();
    if (recording) {
        return;
    }
    auto& slot = slots[slot_index];
    Wait(slot.tick, WaitReason::CommandSlot);
    if (profiling) ResolveProfile(slot_index);
    slot.used_resources.clear();
    [slot.retained removeAllObjects];
    [slot.residency removeAllAllocations];
    [slot.residency commit];
    [slot.allocator reset];
    [slot.commands beginCommandBufferWithAllocator:slot.allocator];
    recording = true;
}

void Scheduler::KeepAlive(id object) {
    BeginCommands();
    RetainObject(object);
}

void Scheduler::RetainObject(id object) {
    if (object && ![slots[slot_index].retained member:object]) {
        auto& slot = slots[slot_index];
        [slot.retained addObject:object];
        if ([object conformsToProtocol:@protocol(MTLAllocation)]) {
            // Pipeline states are allocations too; retaining them alone does not make them resident.
            [slot.residency addAllocation:(id<MTLAllocation>)object];
        }
    }
}

void Scheduler::UseResource(id<MTLResource> resource) {
    if (!resource) {
        return;
    }
    BeginCommands();
    RegisterResource(resource);
}

void Scheduler::RegisterResource(id<MTLResource> resource) {
    if (!resource) return;
    auto& slot = slots[slot_index];
    const auto identity = (__bridge const void*)resource;
    if (slot.used_resources.contains(identity)) {
        return;
    }
    if (resource.device != device) {
        throw std::invalid_argument("Metal resource belongs to a different device");
    }
    if (![slot.retained member:resource]) {
        [slot.residency addAllocation:resource];
        [slot.retained addObject:resource];
    }
    [resource_ticks setObject:@(next_tick) forKey:resource];
    // KeepAlive alone does not assign a use tick or register a texture's backing
    // resources. Mark only after first-use registration; clear before releasing
    // the slot's strong references. Public entry points still check owner/GPU errors.
    slot.used_resources.insert(identity);
    if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
        id<MTLTexture> texture = (id<MTLTexture>)resource;
        // Views and buffer-backed textures keep their backing allocation resident as well.
        RegisterResource(texture.parentTexture);
        RegisterResource(texture.buffer);
    }
}

void Scheduler::EndEncoding(std::source_location source) {
    CheckThread();
    DrainEncodingWorker(source);
    render_key.reset();
    if (encoder) {
        const bool measured = profiling && profiling->active;
        NSString* label = measured ? encoder.label : nil;
        if (!encoder_buffer_readonly)
            [encoder barrierAfterStages:compute_encoder ? ComputeStages : RenderStages
                     beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder endEncoding];
        encoder = nil;
        compute_encoder = false;
        compute_render_dependency = false;
        if (measured) EndProfile(label);
    }
}

void Scheduler::CloseQueuedRender() {
    if (!queued_render_open) return;
    queued_render_open=false;
    queued_render_key.reset();
    encoding_worker->Enqueue([](EncodingWorker& worker) {worker.Close();},ProfilingEnabled());
}

void Scheduler::CloseQueuedCompute() {
    if (!queued_compute_open) return;
    queued_compute_open=false; queued_compute_render_dependency=false;
    encoding_worker->Enqueue([](EncodingWorker& worker) {worker.CloseCompute();},ProfilingEnabled());
}

void Scheduler::DrainEncodingWorker(std::source_location source) {
    if (!encoding_worker) return;
    auto* site=profile_drains && ProfilingEnabled() ? profiling->Drain(source,
        profile_host_phases && profiling->host_depth ? profiling->host_phase_index : static_cast<unsigned>(HostPhase::Count)) : nullptr;
    struct Timer {
        ProfilingState::DrainSite* site;
        std::chrono::steady_clock::time_point begin;
        std::int64_t cpu{-1};
        std::size_t render{},compute{};
        bool failed{};
        ~Timer() {
            if(!site)return;
            const auto end=std::chrono::steady_clock::now();const auto end_cpu=ThreadCpuNanoseconds();
            const auto ms=std::chrono::duration<double,std::milli>(end-begin).count();
            ++site->calls;site->failures+=failed;site->host_ms+=ms;
            if(cpu>=0 && end_cpu>=cpu) {++site->cpu_valid_calls;site->cpu_ms+=double(end_cpu-cpu)/1000000.0;}
            if(ms>site->max_host_ms) {
                site->max_host_ms=ms;
                site->max_begin_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count();
                site->max_end_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
                site->pending_render=render;site->pending_compute=compute;
            }
        }
    } timer{site,site ? std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{},
        site ? ThreadCpuNanoseconds():-1,slots[slot_index].pending_pipelines.size(),slots[slot_index].pending_compute_pipelines.size()};
    try {
        std::exception_ptr failure;
        try { CloseQueuedRender(); } catch (...) { failure=std::current_exception(); }
        try { CloseQueuedCompute(); } catch (...) { if (!failure) failure=std::current_exception(); }
        try { encoding_worker->Drain(); } catch (...) { if (!failure) failure=std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
        // Only the owner mutates residency. Futures retain the complete PSOs until
        // this handoff, and slot retention then extends through GPU completion.
        auto& pending = slots[slot_index].pending_pipelines;
        for (const auto& result : pending) RetainObject(result.get());
        pending.clear();
        auto& compute_pending = slots[slot_index].pending_compute_pipelines;
        for (const auto& result : compute_pending) RetainObject(result.get());
        compute_pending.clear();
        compute_queued = false;
    } catch (...) {timer.failed=true;throw;}
}

EncodingWorkerStats Scheduler::WorkerStats() const {
    CheckThread();
    return {queued_draws, encoding_worker ? encoding_worker->encoded_draws.load() : 0,
            encoding_worker && encoding_worker->separate_thread.load(),
            encoding_worker ? encoding_worker->profiled_cpu_nanoseconds.load() : 0,
            encoding_worker ? encoding_worker->profiled_jobs.load() : 0,
            queued_dispatches, encoding_worker ? encoding_worker->encoded_dispatches.load() : 0,
            buffer_readonly_render_passes, conservative_render_passes};
}

namespace {
std::pair<std::array<std::uint64_t, 128>, bool> RenderCompatibilityKey(MTL4RenderPassDescriptor* descriptor) {
    if (!descriptor) throw std::invalid_argument("Missing compatible Metal 4 render pass");
    std::array<std::uint64_t, 128> key{};
    std::size_t cursor = 0;
    const auto push = [&](std::uint64_t value) { key[cursor++] = value; };
    const auto identity = [](id object) { return std::uint64_t(reinterpret_cast<std::uintptr_t>((__bridge void*)object)); };
    bool compatible = !descriptor.visibilityResultBuffer && !descriptor.imageblockSampleLength &&
                      !descriptor.threadgroupMemoryLength && !descriptor.supportColorAttachmentMapping &&
                      [descriptor getSamplePositions:nullptr count:0] == 0;
    const auto attachment = [&](MTLRenderPassAttachmentDescriptor* item) {
        push(identity(item.texture)); push(item.level); push(item.slice); push(item.depthPlane);
        push(identity(item.resolveTexture)); push(item.resolveLevel); push(item.resolveSlice); push(item.resolveDepthPlane);
        push(item.loadAction); push(item.storeAction);
        // Repeated clears/discards/resolves change the meaning of a new pass.
        // StoreActionOptions have no effect on the supported Apple Silicon GPUs.
        if (item.texture && (item.loadAction != MTLLoadActionLoad || item.storeAction != MTLStoreActionStore || item.resolveTexture))
            compatible = false;
    };
    for (NSUInteger n = 0; n < 8; ++n) attachment(descriptor.colorAttachments[n]);
    attachment(descriptor.depthAttachment);
    attachment(descriptor.stencilAttachment);
    push(descriptor.depthAttachment.depthResolveFilter);
    push(descriptor.stencilAttachment.stencilResolveFilter);
    push(descriptor.renderTargetArrayLength); push(descriptor.defaultRasterSampleCount);
    push(descriptor.renderTargetWidth); push(descriptor.renderTargetHeight);
    push(descriptor.tileWidth); push(descriptor.tileHeight);
    push(identity(descriptor.rasterizationRateMap));
    push(descriptor.visibilityResultType);
    return {key, compatible};
}
}

std::shared_ptr<const ImmutableRenderPass> Scheduler::SnapshotRenderPass(MTL4RenderPassDescriptor* descriptor) {
    CheckThread();
    CheckError();
    if (!descriptor) throw std::invalid_argument("Missing immutable Metal 4 render pass");
    auto result = std::shared_ptr<ImmutableRenderPass>{new ImmutableRenderPass};
    result->descriptor = [descriptor copy];
    if (!result->descriptor) throw std::runtime_error("Unable to copy Metal 4 render pass");
    const auto [key, compatible] = RenderCompatibilityKey(result->descriptor);
    result->key = key; result->compatible = compatible; result->device = device;
    result->buffer_disjoint_attachments = BufferDisjointAttachments(result->descriptor);
    return result;
}

void Scheduler::ValidateQueuedRender(const ImmutableRenderPass& pass, const RenderCommand& command,
    std::span<const BufferBinding> buffers, std::span<const TextureBinding> textures,
    std::span<const SamplerBinding> samplers, std::span<const MeshRenderChunk> chunks) const {
    if (pass.device != device || !pass.descriptor || (!command.pipeline && !command.pending_pipeline.valid()) ||
        (command.pipeline && (command.pending_pipeline.valid() || command.pipeline.device != device)) ||
        (command.depth && command.depth.device != device) ||
        (command.mesh_groups && (command.vertices || command.indirect)) ||
        (!command.mesh_groups && !command.vertices) ||
        (command.primitive != MTLPrimitiveTypeTriangle && command.primitive != MTLPrimitiveTypeTriangleStrip) ||
        ((command.mesh_groups || command.indirect) && command.primitive != MTLPrimitiveTypeTriangle) ||
        std::any_of(command.blend_color.begin(), command.blend_color.end(), [](float v) { return !std::isfinite(v); }) ||
        (command.indirect && (command.indirect.device != device || command.indirect_offset % 4 ||
         command.indirect_offset > command.indirect.length || command.indirect.length-command.indirect_offset<16)))
        throw std::invalid_argument("Invalid Metal4 queued render command/pass");
    if ((!chunks.empty() && (!command.mesh_groups || chunks.front().groups != command.mesh_groups)))
        throw std::invalid_argument("Invalid queued mesh chunk list");
    for (const auto& chunk:chunks)
        if (!chunk.groups || !chunk.parameters.buffer || chunk.parameters.buffer.device != device ||
            chunk.parameters.index != 3 || chunk.parameters.offset >= chunk.parameters.buffer.length)
            throw std::invalid_argument("Invalid queued mesh chunk arguments");
    if (!command.scissor.width || !command.scissor.height) return;
    std::array<bool,31> used_buffers{};
    std::array<bool,128> used_textures{};
    std::array<bool,16> used_samplers{};
    for (const auto& binding:buffers) {
        if (!binding.buffer || binding.buffer.device != device || binding.index>=31 ||
            binding.offset>=binding.buffer.length || used_buffers[binding.index])
            throw std::invalid_argument("Invalid immutable queued buffer binding");
        used_buffers[binding.index]=true;
    }
    for (const auto& binding:textures) {
        if (!binding.texture || binding.texture.device!=device || binding.index>=128 || used_textures[binding.index])
            throw std::invalid_argument("Invalid immutable queued texture binding");
        used_textures[binding.index]=true;
    }
    for (const auto& binding:samplers) {
        if (!binding.sampler || binding.sampler.device!=device || binding.index>=16 || used_samplers[binding.index])
            throw std::invalid_argument("Invalid immutable queued sampler binding");
        used_samplers[binding.index]=true;
    }
}

void Scheduler::QueueRenderBatch(const ImmutableRenderPass& pass, std::span<const RenderBatchDraw> draws) {
    CheckThread(); CheckError();
    if (draws.empty() || draws.size() > 65)
        throw std::invalid_argument("Invalid bounded Metal render batch");
    for (const auto& draw : draws) {
        if (!draw.command.scissor.width || !draw.command.scissor.height)
            throw std::invalid_argument("Empty queued batch draw");
        ValidateQueuedRender(pass, draw.command, draw.buffers, draw.textures, draw.samplers, {});
    }
    // Sharing is explicit only inside this complete batch. In particular a
    // Clear pass must start again for the next batch, even for the same key.
    auto continuation = pass;
    continuation.compatible = true;
    CloseQueuedRender();
    try {
        for (const auto& draw : draws)
            QueueRender(continuation, draw.command, draw.buffers, draw.textures, draw.samplers);
    } catch (...) { CloseQueuedRender(); throw; }
    CloseQueuedRender();
}

void Scheduler::QueueRender(const ImmutableRenderPass& pass, const RenderCommand& command,
    std::span<const BufferBinding> buffers, std::span<const TextureBinding> textures,
    std::span<const SamplerBinding> samplers, bool reusable, std::span<const MeshRenderChunk> chunks,
    const ReadOnlyBufferRenderProof* proof) {
    CheckThread(); CheckError();
    ValidateQueuedRender(pass, command, buffers, textures, samplers, chunks);
    if (!command.scissor.width || !command.scissor.height) return;
    CloseQueuedCompute();
    struct Draw {
        RenderCommand command;
        std::vector<BufferBinding> buffers;
        std::vector<TextureBinding> textures;
        std::vector<SamplerBinding> samplers;
        std::vector<MeshRenderChunk> chunks;
    };
    auto draw=std::make_shared<Draw>(Draw{command,{buffers.begin(),buffers.end()},
        {textures.begin(),textures.end()},{samplers.begin(),samplers.end()},{chunks.begin(),chunks.end()}});
    BeginCommands();
    if (encoder) EndEncoding();
    if (!encoding_worker) encoding_worker=std::make_unique<EncodingWorker>(device);
    auto& slot=slots[slot_index];
    for (const auto& binding:buffers) RegisterResource(binding.buffer);
    for (const auto& chunk:chunks) RegisterResource(chunk.parameters.buffer);
    for (const auto& binding:textures) RegisterResource(binding.texture);
    for (const auto& binding:samplers) RetainObject(binding.sampler);
    RegisterResource(command.indirect);
    RetainObject(command.pipeline); RetainObject(command.depth); RetainObject(encoding_worker->arguments);
    if (command.pending_pipeline.valid()) slot.pending_pipelines.push_back(command.pending_pipeline);
    const bool compatible=reusable && pass.compatible;
    const bool buffer_readonly = proof && reusable && pass.buffer_disjoint_attachments;
    const bool begin=!compatible || !queued_render_open || !queued_render_key || *queued_render_key!=pass.key ||
                     queued_render_buffer_readonly != buffer_readonly;
    if (begin) {
        CloseQueuedRender();
        auto* descriptor=pass.descriptor;
        for (NSUInteger i=0;i<8;++i) {
            RegisterResource(descriptor.colorAttachments[i].texture);
            RegisterResource(descriptor.colorAttachments[i].resolveTexture);
        }
        RegisterResource(descriptor.depthAttachment.texture); RegisterResource(descriptor.depthAttachment.resolveTexture);
        RegisterResource(descriptor.stencilAttachment.texture); RegisterResource(descriptor.stencilAttachment.resolveTexture);
        RegisterResource(descriptor.visibilityResultBuffer); RetainObject(descriptor.rasterizationRateMap);
        id<MTL4CounterHeap> heap=nil;NSUInteger counter{};
        if (ProfilingEnabled()) {
            ++profiling->encoders;
            auto& batch=profiling->batches[slot_index];
            if (!profiling->host_only) {
                if (batch.used+2<=ProfilingState::Capacity) {
                    heap=batch.heap;counter=batch.used;RetainObject(heap);
                    batch.entries.push_back({counter,"queued render",current_frame});batch.used+=2;
                } else ++batch.dropped;
            }
        }
        auto commands=slot.commands;
        encoding_worker->Enqueue([descriptor,commands,heap,counter,buffer_readonly](EncodingWorker& worker) {
            worker.commands=commands;worker.counter_heap=heap;worker.counter_index=counter;
            if (heap) [commands writeTimestampIntoHeap:heap atIndex:counter];
            worker.encoder=[commands renderCommandEncoderWithDescriptor:descriptor];
            if (!worker.encoder) throw std::runtime_error("Unable to create queued Metal4 render encoder");
            worker.buffer_readonly = buffer_readonly;
            worker.blend_color = {};
            // Compute/opaque-render producers publish to all future stages at
            // close. Only attachment RAW/WAR/WAW remains between proven draws;
            // their buffer-only vertex/mesh reads can overlap earlier fragments.
            [worker.encoder barrierAfterQueueStages:buffer_readonly ? AttachmentStages : RenderStages
                                     beforeStages:buffer_readonly ? AttachmentStages : MTLStageAll
                                visibilityOptions:MTL4VisibilityOptionDevice];
        },ProfilingEnabled());
        queued_render_open=true;
        queued_render_buffer_readonly=buffer_readonly;
        if (buffer_readonly) ++buffer_readonly_render_passes;
        else ++conservative_render_passes;
        if (compatible) queued_render_key=pass.key;
        ++encoder_count;
    }
    encoding_worker->Enqueue([draw, expected_device=device](EncodingWorker& worker) {
        const auto& command=draw->command;
        worker.Bind(draw->buffers,draw->textures,draw->samplers);
        auto encoder=worker.encoder;
        auto pipeline = command.pipeline ? command.pipeline : command.pending_pipeline.get();
        if (!pipeline || pipeline.device != expected_device)
            throw std::invalid_argument("Deferred Metal pipeline is missing or belongs to another device");
        [encoder setRenderPipelineState:pipeline];
        [encoder setDepthStencilState:command.depth];
        [encoder setStencilReferenceValue:command.stencil];
        [encoder setViewport:command.viewport];[encoder setScissorRect:command.scissor];
        [encoder setFrontFacingWinding:command.winding];[encoder setCullMode:command.cull];
        if (worker.blend_color != command.blend_color) {
            [encoder setBlendColorRed:command.blend_color[0] green:command.blend_color[1]
                                blue:command.blend_color[2] alpha:command.blend_color[3]];
            worker.blend_color = command.blend_color;
        }
        if (command.mesh_groups) {
            const auto draw_mesh = [&](NSUInteger groups) {
                [encoder setArgumentTable:worker.arguments atStages:MTLRenderStageMesh | MTLRenderStageFragment];
                [encoder drawMeshThreadgroups:MTLSizeMake(groups,1,1)
                    threadsPerObjectThreadgroup:MTLSizeMake(0,0,0) threadsPerMeshThreadgroup:MTLSizeMake(96,1,1)];
                ++worker.encoded_draws;
            };
            if (draw->chunks.empty()) draw_mesh(command.mesh_groups);
            else for (const auto& chunk:draw->chunks) {
                [worker.arguments setAddress:chunk.parameters.buffer.gpuAddress+chunk.parameters.offset atIndex:3];
                draw_mesh(chunk.groups);
            }
            return;
        }
        [encoder setArgumentTable:worker.arguments atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        if (command.indirect)
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle indirectBuffer:command.indirect.gpuAddress+command.indirect_offset];
        else [encoder drawPrimitives:command.primitive vertexStart:0 vertexCount:command.vertices];
        ++worker.encoded_draws;
    },ProfilingEnabled());
    const auto count=std::max<std::size_t>(1,chunks.size());
    argument_updates+=count; queued_draws+=count;
    if (!compatible) CloseQueuedRender();
}

void Scheduler::QueueCompute(std::span<const ComputeDispatch> dispatches,
                             std::span<const ComputeBufferCopy> copies,
                             NSString* label, bool append_current) {
    CheckThread();
    if (dispatches.empty()) throw std::invalid_argument("Empty queued Metal4 compute chain");
    QueueComputeInternal(dispatches, copies, label, append_current, {});
}

void Scheduler::QueueTransfer(const TransferCommand& transfer) {
    QueueComputeInternal({}, {}, @"queued transfer", false, std::span{&transfer, 1});
}

void Scheduler::QueueClear(MTL4RenderPassDescriptor* descriptor) {
    const auto pass=SnapshotRenderPass(descriptor);
    BeginCommands();
    if (encoder) EndEncoding();
    CloseQueuedRender(); CloseQueuedCompute();
    if (!encoding_worker) encoding_worker=std::make_unique<EncodingWorker>(device);
    descriptor=pass->descriptor;
    for (NSUInteger i=0;i<8;++i) {
        RegisterResource(descriptor.colorAttachments[i].texture);
        RegisterResource(descriptor.colorAttachments[i].resolveTexture);
    }
    RegisterResource(descriptor.depthAttachment.texture); RegisterResource(descriptor.depthAttachment.resolveTexture);
    RegisterResource(descriptor.stencilAttachment.texture); RegisterResource(descriptor.stencilAttachment.resolveTexture);
    const bool buffer_readonly=pass->buffer_disjoint_attachments;
    if (buffer_readonly) ++buffer_readonly_render_passes;
    else ++conservative_render_passes;
    ++encoder_count;
    id<MTL4CounterHeap> heap=nil; NSUInteger counter{};
    if (ProfilingEnabled()) {
        ++profiling->encoders;
        auto& batch=profiling->batches[slot_index];
        if (!profiling->host_only) {
            if (batch.used+2<=ProfilingState::Capacity) {
                heap=batch.heap;counter=batch.used;RetainObject(heap);
                batch.entries.push_back({counter,"queued clear",current_frame});batch.used+=2;
            } else ++batch.dropped;
        }
    }
    const auto commands=slots[slot_index].commands;
    // Clears remain separate native load-action passes, including repeated
    // clears of one attachment. No draw or caller-owned descriptor is borrowed.
    encoding_worker->Enqueue([pass,commands,heap,counter,buffer_readonly](EncodingWorker&) {
        if (heap) [commands writeTimestampIntoHeap:heap atIndex:counter];
        auto clear=[commands renderCommandEncoderWithDescriptor:pass->descriptor];
        if (!clear) throw std::runtime_error("Unable to create queued Metal4 clear encoder");
        [clear barrierAfterQueueStages:buffer_readonly ? AttachmentStages : RenderStages
                          beforeStages:buffer_readonly ? AttachmentStages : MTLStageAll
                     visibilityOptions:MTL4VisibilityOptionDevice];
        if (!buffer_readonly)
            [clear barrierAfterStages:RenderStages beforeQueueStages:MTLStageAll
                    visibilityOptions:MTL4VisibilityOptionDevice];
        [clear endEncoding];
        if (heap) [commands writeTimestampIntoHeap:heap atIndex:counter+1];
    },ProfilingEnabled());
}

void Scheduler::QueueComputeInternal(std::span<const ComputeDispatch> dispatches,
                                     std::span<const ComputeBufferCopy> copies,
                                     NSString* label, bool append_current,
                                     std::span<const TransferCommand> transfers) {
    CheckThread(); CheckError();
    if (dispatches.empty() && transfers.empty()) throw std::invalid_argument("Empty queued Metal4 compute chain");
    if (append_current && (!encoder || !compute_encoder))
        throw std::invalid_argument("Queued compute continuation requires the current compute encoder");
    const auto max_group = device.maxThreadsPerThreadgroup;
    for (const auto& dispatch : dispatches) {
        if ((!dispatch.pipeline && !dispatch.pending_pipeline.valid()) ||
            (dispatch.pipeline && (dispatch.pending_pipeline.valid() || dispatch.pipeline.device != device)) ||
            !dispatch.threads.width || !dispatch.threads.height || !dispatch.threads.depth ||
            !dispatch.group.width || !dispatch.group.height || !dispatch.group.depth ||
            dispatch.group.width > max_group.width || dispatch.group.height > max_group.height ||
            dispatch.group.depth > max_group.depth ||
            dispatch.group.width > NSUIntegerMax / dispatch.group.height ||
            dispatch.group.width * dispatch.group.height > NSUIntegerMax / dispatch.group.depth ||
            (dispatch.simd_groups && (dispatch.threads.width!=1 || dispatch.threads.height!=1 ||
             dispatch.threads.depth!=1 || dispatch.group.width!=1 || dispatch.group.height!=1 || dispatch.group.depth!=1)))
            throw std::invalid_argument("Invalid queued Metal4 dispatch dimensions/pipeline");
        std::array<bool,31> buffers{}; std::array<bool,128> textures{}; std::array<bool,16> samplers{};
        for (const auto& b : dispatch.buffers) {
            if (!b.buffer || b.buffer.device != device || b.index >= buffers.size() ||
                b.offset >= b.buffer.length || buffers[b.index])
                throw std::invalid_argument("Invalid queued compute buffer binding");
            buffers[b.index] = true;
        }
        for (const auto& b : dispatch.textures) {
            if (!b.texture || b.texture.device != device || b.index >= textures.size() || textures[b.index])
                throw std::invalid_argument("Invalid queued compute texture binding");
            textures[b.index] = true;
        }
        for (const auto& b : dispatch.samplers) {
            if (!b.sampler || b.sampler.device != device || b.index >= samplers.size() || samplers[b.index])
                throw std::invalid_argument("Invalid queued compute sampler binding");
            samplers[b.index] = true;
        }
    }
    for (const auto& copy : copies) {
        if (!copy.source || !copy.destination || copy.source.device != device || copy.destination.device != device ||
            !copy.size || copy.source_offset > copy.source.length || copy.size > copy.source.length-copy.source_offset ||
            copy.destination_offset > copy.destination.length || copy.size > copy.destination.length-copy.destination_offset ||
            (copy.source == copy.destination && copy.source_offset < copy.destination_offset+copy.size &&
             copy.destination_offset < copy.source_offset+copy.size))
            throw std::invalid_argument("Invalid queued compute copy range/overlap");
    }
    for (const auto& transfer : transfers)
        for (id<MTLResource> resource : {id<MTLResource>(transfer.source_buffer), id<MTLResource>(transfer.destination_buffer),
                                       id<MTLResource>(transfer.source_texture), id<MTLResource>(transfer.destination_texture)})
            if (resource && resource.device != device)
                throw std::invalid_argument("Queued transfer belongs to another Metal device");
    struct Chain {
        std::vector<ComputeDispatch> dispatches;
        std::vector<ComputeBufferCopy> copies;
        std::vector<TransferCommand> transfers;
    };
    auto chain = std::make_shared<const Chain>(Chain{{dispatches.begin(),dispatches.end()},
        {copies.begin(),copies.end()},{transfers.begin(),transfers.end()}});
    BeginCommands();
    id<MTL4ComputeCommandEncoder> borrowed = nil;
    if (append_current || (compute_batch_depth && encoder && compute_encoder)) {
        // An existing owner encoder may be handed to this worker, then drained
        // before the next direct owner command. Keep the native batch intact.
        if (!encoder || !compute_encoder) borrowed = BeginCompute();
        else borrowed = (id<MTL4ComputeCommandEncoder>)encoder;
        compute_queued = true;
    } else {
        if (encoder) EndEncoding();
        CloseQueuedRender();
    }
    const bool persistent = compute_batch_depth && !borrowed;
    const bool begin = persistent && !queued_compute_open;
    const bool previous_render_dependency = queued_compute_render_dependency;
    if (begin) {
        queued_compute_open=true;
        queued_compute_render_dependency=(compute_producers & RenderStages)!=0;
    }
    if (!encoding_worker) encoding_worker = std::make_unique<EncodingWorker>(device);
    auto& slot = slots[slot_index];
    for (const auto& copy : copies) { RegisterResource(copy.source); RegisterResource(copy.destination); }
    for (const auto& transfer : transfers) {
        RegisterResource(transfer.source_buffer); RegisterResource(transfer.destination_buffer);
        RegisterResource(transfer.source_texture); RegisterResource(transfer.destination_texture);
    }
    for (const auto& dispatch : dispatches) {
        for (const auto& b : dispatch.buffers) RegisterResource(b.buffer);
        for (const auto& b : dispatch.textures) RegisterResource(b.texture);
        for (const auto& b : dispatch.samplers) RetainObject(b.sampler);
        RetainObject(dispatch.pipeline);
        if (dispatch.pending_pipeline.valid()) slot.pending_compute_pipelines.push_back(dispatch.pending_pipeline);
    }
    RetainObject(encoding_worker->arguments);
    id<MTL4CounterHeap> heap = nil; NSUInteger counter{};
    if (!borrowed && (!persistent || begin)) {
        ++encoder_count;
        if (ProfilingEnabled()) {
            ++profiling->encoders;
            auto& batch = profiling->batches[slot_index];
            if (!profiling->host_only) {
                if (batch.used+2 <= ProfilingState::Capacity) {
                    heap=batch.heap; counter=batch.used; RetainObject(heap);
                    batch.entries.push_back({counter,label && label.UTF8String ? label.UTF8String:"queued compute",current_frame}); batch.used+=2;
                } else ++batch.dropped;
            }
        }
    }
    const auto commands = slot.commands;
    const auto producers = compute_producers;
    const bool restore_render_dependency = (producers & RenderStages) &&
        (borrowed ? !compute_render_dependency : persistent && !begin && !previous_render_dependency);
    encoding_worker->Enqueue([chain,borrowed,persistent,begin,commands,producers,restore_render_dependency,heap,counter,label,max_group,expected_device=device](EncodingWorker& worker) {
        std::vector<id<MTLComputePipelineState>> pipelines;
        struct Shape { MTLSize threads, group; };
        std::vector<Shape> shapes;
        pipelines.reserve(chain->dispatches.size());
        for (const auto& dispatch : chain->dispatches) {
            auto pipeline = dispatch.pipeline ? dispatch.pipeline : dispatch.pending_pipeline.get();
            auto threads=dispatch.threads, group=dispatch.group;
            if (!pipeline || pipeline.device != expected_device)
                throw std::invalid_argument("Invalid resolved Metal4 compute pipeline");
            if (dispatch.simd_groups) {
                const auto width=pipeline.threadExecutionWidth;
                if (!width || width>max_group.width || dispatch.simd_groups>NSUIntegerMax/width)
                    throw std::invalid_argument("Invalid native SIMD dispatch width/count");
                group=MTLSizeMake(width,1,1); threads=MTLSizeMake(width*dispatch.simd_groups,1,1);
            }
            if (group.width*group.height*group.depth > pipeline.maxTotalThreadsPerThreadgroup)
                throw std::invalid_argument("Invalid resolved Metal4 compute pipeline/group");
            pipelines.push_back(pipeline);
            shapes.push_back({threads,group});
        }
        if (heap) [commands writeTimestampIntoHeap:heap atIndex:counter];
        auto encoder = borrowed ? borrowed : persistent && !begin ? worker.compute : [commands computeCommandEncoder];
        if (!encoder) throw std::runtime_error("Unable to create queued Metal4 compute encoder");
        if (persistent && begin) {
            worker.compute=encoder; worker.compute_commands=commands;
            worker.compute_counter_heap=heap; worker.compute_counter_index=counter;
        }
        try {
            if (label) encoder.label=label;
            const auto stages = MTLStageDispatch | MTLStageBlit;
            if (restore_render_dependency)
                [encoder barrierAfterQueueStages:RenderStages beforeStages:MTLStageAll
                               visibilityOptions:MTL4VisibilityOptionDevice];
            if (borrowed || (persistent && !begin)) [encoder barrierAfterEncoderStages:stages beforeEncoderStages:stages
                                         visibilityOptions:MTL4VisibilityOptionDevice];
            else if (producers & RenderStages)
                [encoder barrierAfterQueueStages:RenderStages beforeStages:MTLStageAll
                               visibilityOptions:MTL4VisibilityOptionDevice];
            for (std::size_t i=0; i<chain->copies.size(); ++i) {
                const auto& copy=chain->copies[i];
                bool dependency{};
                for(std::size_t previous=0;previous<i && !dependency;++previous) {
                    const auto& a=chain->copies[previous];
                    dependency=BufferRangesMayAlias(a.destination,a.destination_offset,a.size,copy.source,copy.source_offset,copy.size) ||
                        BufferRangesMayAlias(a.destination,a.destination_offset,a.size,copy.destination,copy.destination_offset,copy.size) ||
                        BufferRangesMayAlias(a.source,a.source_offset,a.size,copy.destination,copy.destination_offset,copy.size);
                }
                if(dependency) [encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit
                                              visibilityOptions:MTL4VisibilityOptionDevice];
                [encoder copyFromBuffer:copy.source sourceOffset:copy.source_offset toBuffer:copy.destination
                      destinationOffset:copy.destination_offset size:copy.size];
            }
            for (std::size_t i=0; i<chain->transfers.size(); ++i) {
                if (i || !chain->copies.empty())
                    [encoder barrierAfterEncoderStages:stages beforeEncoderStages:stages
                                      visibilityOptions:MTL4VisibilityOptionDevice];
                const auto& t=chain->transfers[i];
                switch (t.kind) {
                case TransferCommand::Kind::CopyBuffer:
                    [encoder copyFromBuffer:t.source_buffer sourceOffset:t.source_offset toBuffer:t.destination_buffer
                          destinationOffset:t.destination_offset size:t.size]; break;
                case TransferCommand::Kind::FillBuffer:
                    [encoder fillBuffer:t.destination_buffer range:NSMakeRange(t.destination_offset,t.size) value:t.value]; break;
                case TransferCommand::Kind::CopyTexture:
                    [encoder copyFromTexture:t.source_texture sourceSlice:t.source_slice sourceLevel:t.source_level
                               sourceOrigin:t.source_origin sourceSize:t.extent toTexture:t.destination_texture
                           destinationSlice:t.destination_slice destinationLevel:t.destination_level
                          destinationOrigin:t.destination_origin]; break;
                case TransferCommand::Kind::Upload:
                    [encoder copyFromBuffer:t.source_buffer sourceOffset:t.source_offset sourceBytesPerRow:t.bytes_per_row
                         sourceBytesPerImage:t.bytes_per_image sourceSize:t.extent toTexture:t.destination_texture
                            destinationSlice:t.destination_slice destinationLevel:t.destination_level
                           destinationOrigin:t.destination_origin options:t.options]; break;
                case TransferCommand::Kind::Download:
                    [encoder copyFromTexture:t.source_texture sourceSlice:t.source_slice sourceLevel:t.source_level
                               sourceOrigin:t.source_origin sourceSize:t.extent toBuffer:t.destination_buffer
                          destinationOffset:t.destination_offset destinationBytesPerRow:t.bytes_per_row
                         destinationBytesPerImage:t.bytes_per_image options:t.options]; break;
                case TransferCommand::Kind::GenerateMipmaps:
                    [encoder generateMipmapsForTexture:t.destination_texture]; break;
                }
            }
            if (!chain->copies.empty() || !chain->transfers.empty()) [encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch
                                                       visibilityOptions:MTL4VisibilityOptionDevice];
            for (std::size_t i=0; i<chain->dispatches.size(); ++i) {
                if (i) [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                                         visibilityOptions:MTL4VisibilityOptionDevice];
                const auto& dispatch = chain->dispatches[i];
                worker.Bind(dispatch.buffers,dispatch.textures,dispatch.samplers);
                [encoder setComputePipelineState:pipelines[i]]; [encoder setArgumentTable:worker.arguments];
                [encoder dispatchThreads:shapes[i].threads threadsPerThreadgroup:shapes[i].group];
                ++worker.encoded_dispatches;
            }
            if (!borrowed && !persistent) {
                [encoder barrierAfterStages:ComputeStages beforeQueueStages:MTLStageAll
                         visibilityOptions:MTL4VisibilityOptionDevice];
                [encoder endEncoding];
            }
        } catch (...) { if (!borrowed && !persistent) [encoder endEncoding]; throw; }
        if (heap && !persistent) [commands writeTimestampIntoHeap:heap atIndex:counter+1];
    },ProfilingEnabled());
    if (restore_render_dependency) {
        if (borrowed) compute_render_dependency = true;
        else queued_compute_render_dependency = true;
    }
    queued_dispatches += dispatches.size(); argument_updates += dispatches.size();
}

id<MTL4RenderCommandEncoder> Scheduler::BeginCompatibleRender(const ImmutableRenderPass& pass, NSString* label) {
    BeginCommands();
    if (pass.device != device || !pass.descriptor)
        throw std::invalid_argument("Immutable render pass belongs to another Metal device");
    if (pass.compatible && encoder && render_key && *render_key == pass.key)
        return (id<MTL4RenderCommandEncoder>)encoder;
    auto result = BeginRender(pass.descriptor, label);
    if (pass.compatible) render_key = pass.key;
    return result;
}

id<MTL4RenderCommandEncoder> Scheduler::BeginCompatibleRender(MTL4RenderPassDescriptor* descriptor, NSString* label) {
    BeginCommands();
    const auto [key, compatible] = RenderCompatibilityKey(descriptor);
    if (compatible && encoder && render_key && *render_key == key)
        return (id<MTL4RenderCommandEncoder>)encoder;
    auto result = BeginRender(descriptor, label);
    if (compatible) render_key = key;
    return result;
}

id<MTL4RenderCommandEncoder> Scheduler::BeginRender(MTL4RenderPassDescriptor* descriptor, NSString* label,
                                                  const ReadOnlyBufferRenderProof* proof,std::source_location source) {
    if (!descriptor) {
        throw std::invalid_argument("Missing Metal 4 render pass descriptor");
    }
    BeginCommands();
    EndEncoding(source);
    for (NSUInteger i = 0; i < 8; ++i) {
        RegisterResource(descriptor.colorAttachments[i].texture);
        RegisterResource(descriptor.colorAttachments[i].resolveTexture);
    }
    RegisterResource(descriptor.depthAttachment.texture);
    RegisterResource(descriptor.depthAttachment.resolveTexture);
    RegisterResource(descriptor.stencilAttachment.texture);
    RegisterResource(descriptor.stencilAttachment.resolveTexture);
    RegisterResource(descriptor.visibilityResultBuffer);
    RetainObject(descriptor.rasterizationRateMap);
    if (ProfilingEnabled()) BeginProfile();
    id<MTL4RenderCommandEncoder> result =
        [slots[slot_index].commands renderCommandEncoderWithDescriptor:descriptor];
    if (!result) {
        throw std::runtime_error("Unable to create Metal 4 render encoder");
    }
    encoder = result;
    encoder_buffer_readonly = proof && BufferDisjointAttachments(descriptor);
    if (ProfilingEnabled()) encoder.label = label ? label : @"render";
    ++encoder_count;
    if (encoder_buffer_readonly) ++buffer_readonly_render_passes;
    else ++conservative_render_passes;
    [encoder barrierAfterQueueStages:encoder_buffer_readonly ? AttachmentStages : RenderStages
                        beforeStages:encoder_buffer_readonly ? AttachmentStages : MTLStageAll
                   visibilityOptions:MTL4VisibilityOptionDevice];
    return result;
}

Scheduler::BufferComputeScope::BufferComputeScope(Scheduler& scheduler_)
    : scheduler{scheduler_}, previous{MTLStageAll} {
    scheduler.CheckThread();
    scheduler.CheckError();
    // Preserve the encoder boundary in queue order. Worker-owned encoders do
    // not expose native state to this scope, so waiting for their pipelines is
    // unnecessary. An existing owner encoder (including borrowed worker jobs)
    // still needs the ordinary drain before it can be closed on this thread.
    if (scheduler.encoder) scheduler.EndEncoding();
    else {
        scheduler.CloseQueuedRender();
        scheduler.CloseQueuedCompute();
        scheduler.render_key.reset();
    }
    previous = scheduler.compute_producers;
    scheduler.compute_producers = MTLStageDispatch | MTLStageBlit;
}

Scheduler::BufferComputeScope::~BufferComputeScope() {
    scheduler.compute_producers = previous;
}

Scheduler::ComputeBatchScope::ComputeBatchScope(Scheduler& scheduler_) : scheduler{scheduler_} {
    scheduler.CheckThread();
    scheduler.CheckError();
    if (scheduler.compute_batch_depth == std::numeric_limits<unsigned>::max())
        throw std::overflow_error("Metal compute batch nesting overflow");
    ++scheduler.compute_batch_depth;
}

Scheduler::ComputeBatchScope::~ComputeBatchScope() noexcept {
    if (!--scheduler.compute_batch_depth) {
        try {
            scheduler.CloseQueuedCompute();
            if (scheduler.compute_encoder) scheduler.EndEncoding();
        } catch (...) {
            // RAII owners, including optional/nested scopes, must never throw
            // during destruction. The next explicit scheduler operation reports
            // the original failure without submitting partially encoded work.
            if(!scheduler.scope_error) scheduler.scope_error=std::current_exception();
        }
    }
}

void Scheduler::EndCompute() {
    CheckThread();
    CheckError();
    if (!compute_batch_depth) EndEncoding();
}

id<MTL4ComputeCommandEncoder> Scheduler::BeginCompute(id<MTLComputePipelineState> pipeline,std::source_location source) {
    BeginCommands();
    if (compute_queued) DrainEncodingWorker(source);
    if (compute_batch_depth && encoder && compute_encoder) {
        auto result = (id<MTL4ComputeCommandEncoder>)encoder;
        // A disjoint buffer scope can end inside a larger compute batch. The
        // next general operation may consume an earlier attachment even though
        // the native compute encoder is still open. Queue barriers apply to
        // prior render encoders, separately from this pass's dispatch ordering.
        if ((compute_producers & RenderStages) && !compute_render_dependency) {
            [result barrierAfterQueueStages:RenderStages beforeStages:MTLStageAll
                         visibilityOptions:MTL4VisibilityOptionDevice];
            compute_render_dependency = true;
        }
        // Dispatch and native buffer blits may both read or overwrite earlier
        // ranges. Keep RAW/WAR/WAW order without closing the current pass.
        const auto stages = MTLStageDispatch | MTLStageBlit;
        [result barrierAfterEncoderStages:stages beforeEncoderStages:stages
                       visibilityOptions:MTL4VisibilityOptionDevice];
        return result;
    }
    EndEncoding(source);
    if (ProfilingEnabled()) BeginProfile();
    id<MTL4ComputeCommandEncoder> result = [slots[slot_index].commands computeCommandEncoder];
    if (!result) {
        throw std::runtime_error("Unable to create Metal 4 compute encoder");
    }
    encoder = result;
    compute_encoder = true;
    encoder_buffer_readonly = false;
    compute_render_dependency = (compute_producers & RenderStages) != 0;
    if (ProfilingEnabled()) encoder.label = pipeline.label ? pipeline.label : @"compute";
    ++encoder_count;
    // Prior compute writes already carry a producer barrier to all following
    // encoders. General operations must additionally retire render reads/writes;
    // proven disjoint buffer work need not wait for those render stages.
    if (compute_producers & RenderStages)
        [encoder barrierAfterQueueStages:RenderStages beforeStages:MTLStageAll
                       visibilityOptions:MTL4VisibilityOptionDevice];
    return result;
}

id<MTL4ArgumentTable> Scheduler::MakeArguments(std::span<const BufferBinding> buffers,
                                             std::span<const TextureBinding> textures,
                                             std::span<const SamplerBinding> samplers) {
    BeginCommands();
    MTL4ArgumentTableDescriptor* descriptor = [[MTL4ArgumentTableDescriptor alloc] init];
    descriptor.initializeBindings = YES;
    descriptor.maxBufferBindCount = 31;
    descriptor.maxTextureBindCount = 128;
    descriptor.maxSamplerStateBindCount = 16;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:descriptor error:&error];
    if (!table) {
        throw std::runtime_error("Unable to create Metal 4 argument table");
    }
    std::array<bool, 31> used_buffers{};
    std::array<bool, 128> used_textures{};
    std::array<bool, 16> used_samplers{};
    for (const auto& binding : buffers) {
        if (!binding.buffer || binding.index >= 31 || binding.offset >= binding.buffer.length || used_buffers[binding.index]) {
            throw std::invalid_argument("Invalid Metal 4 buffer binding");
        }
        used_buffers[binding.index] = true;
        RegisterResource(binding.buffer);
        [table setAddress:binding.buffer.gpuAddress + binding.offset atIndex:binding.index];
    }
    for (const auto& binding : textures) {
        if (!binding.texture || binding.index >= 128 || used_textures[binding.index]) {
            throw std::invalid_argument("Invalid Metal 4 texture binding");
        }
        used_textures[binding.index] = true;
        RegisterResource(binding.texture);
        [table setTexture:binding.texture.gpuResourceID atIndex:binding.index];
    }
    for (const auto& binding : samplers) {
        if (!binding.sampler || binding.index >= 16 || binding.sampler.device != device || used_samplers[binding.index]) {
            throw std::invalid_argument("Invalid Metal 4 sampler binding");
        }
        used_samplers[binding.index] = true;
        RetainObject(binding.sampler);
        [table setSamplerState:binding.sampler.gpuResourceID atIndex:binding.index];
    }
    RetainObject(table);
    return table;
}

void Scheduler::RebindBufferArguments(id<MTL4ArgumentTable> table,
                                     std::span<const BufferBinding> buffers) {
    BeginCommands();
    if (!table || table.device != device) {
        throw std::invalid_argument("Invalid reusable Metal 4 argument table");
    }
    std::array<bool, 31> used{};
    for (const auto& binding : buffers) {
        if (!binding.buffer || binding.index >= 31 || binding.offset >= binding.buffer.length || used[binding.index]) {
            throw std::invalid_argument("Invalid Metal 4 buffer binding");
        }
        used[binding.index] = true;
        RegisterResource(binding.buffer);
        [table setAddress:binding.buffer.gpuAddress + binding.offset atIndex:binding.index];
    }
    RetainObject(table);
}

id<MTL4ArgumentTable> Scheduler::UpdateArguments(ReusableArgumentTable& arguments,
                                                std::span<const BufferBinding> buffers,
                                                std::span<const TextureBinding> textures,
                                                std::span<const SamplerBinding> samplers) {
    BeginCommands();
    if (arguments.table && arguments.table.device != device) {
        throw std::invalid_argument("Reusable Metal 4 argument table belongs to a different device");
    }
    std::uint32_t used_buffers{};
    std::array<std::uint64_t, 2> used_textures{};
    std::uint16_t used_samplers{};
    // Reject every malformed binding before clearing or replacing any native slot.
    for (const auto& binding : buffers) {
        if (!binding.buffer || binding.index >= 31 || binding.offset >= binding.buffer.length ||
            (used_buffers & (1u << binding.index)))
            throw std::invalid_argument("Invalid Metal 4 buffer binding");
        used_buffers |= 1u << binding.index;
    }
    for (const auto& binding : textures) {
        if (!binding.texture || binding.index >= 128)
            throw std::invalid_argument("Invalid Metal 4 texture binding");
        const auto bit = std::uint64_t{1} << (binding.index & 63);
        if (used_textures[binding.index >> 6] & bit)
            throw std::invalid_argument("Invalid Metal 4 texture binding");
        used_textures[binding.index >> 6] |= bit;
    }
    for (const auto& binding : samplers) {
        if (!binding.sampler || binding.index >= 16 || binding.sampler.device != device ||
            (used_samplers & (1u << binding.index)))
            throw std::invalid_argument("Invalid Metal 4 sampler binding");
        used_samplers |= 1u << binding.index;
    }
    if (!arguments.table) {
        arguments.table = MakeArguments(buffers, textures, samplers);
        for (const auto& binding : textures) {
            arguments.texture_ids[binding.index] = binding.texture.gpuResourceID._impl;
            arguments.texture_objects[binding.index] = binding.texture;
        }
        for (const auto& binding : samplers) {
            arguments.sampler_ids[binding.index] = binding.sampler.gpuResourceID._impl;
            arguments.sampler_objects[binding.index] = binding.sampler;
        }
    } else {
        // Registration checks device ownership and retains the full backing chain.
        // Finish it before changing slots so a rejected resource leaves bindings intact.
        for (const auto& binding : buffers) RegisterResource(binding.buffer);
        for (const auto& binding : textures) RegisterResource(binding.texture);
        for (const auto& binding : samplers) RetainObject(binding.sampler);
        RetainObject(arguments.table);
        for (auto removed = arguments.buffers & ~used_buffers; removed; removed &= removed - 1)
            [arguments.table setAddress:0 atIndex:std::countr_zero(removed)];
        for (unsigned bank = 0; bank < 2; ++bank)
            for (auto removed = arguments.textures[bank] & ~used_textures[bank]; removed; removed &= removed - 1) {
                const auto index = bank * 64 + std::countr_zero(removed);
                [arguments.table setTexture:MTLResourceID{0} atIndex:index];
                arguments.texture_objects[index] = nil;
                arguments.texture_ids[index] = 0;
            }
        for (auto removed = unsigned(arguments.samplers & ~used_samplers); removed; removed &= removed - 1) {
            const auto index = std::countr_zero(removed);
            [arguments.table setSamplerState:MTLResourceID{0} atIndex:index];
            arguments.sampler_objects[index] = nil;
            arguments.sampler_ids[index] = 0;
        }
        for (const auto& binding : buffers)
            [arguments.table setAddress:binding.buffer.gpuAddress + binding.offset atIndex:binding.index];
        // An unchanged ID still needs registration and retention above. Only
        // skip the native slot write; removed/re-added slots must be republished.
        for (const auto& binding : textures) {
            const auto resource_id = binding.texture.gpuResourceID;
            auto& previous = arguments.texture_ids[binding.index];
            if (previous != resource_id._impl || arguments.texture_objects[binding.index] != binding.texture ||
                !(arguments.textures[binding.index >> 6] & (std::uint64_t{1} << (binding.index & 63)))) {
                [arguments.table setTexture:resource_id atIndex:binding.index];
                arguments.texture_objects[binding.index] = binding.texture;
                previous = resource_id._impl;
            }
        }
        for (const auto& binding : samplers) {
            const auto resource_id = binding.sampler.gpuResourceID;
            auto& previous = arguments.sampler_ids[binding.index];
            if (previous != resource_id._impl || arguments.sampler_objects[binding.index] != binding.sampler ||
                !(arguments.samplers & (1u << binding.index))) {
                [arguments.table setSamplerState:resource_id atIndex:binding.index];
                arguments.sampler_objects[binding.index] = binding.sampler;
                previous = resource_id._impl;
            }
        }
    }
    arguments.buffers = used_buffers;
    arguments.textures = used_textures;
    arguments.samplers = used_samplers;
    ++argument_updates;
    return arguments.table;
}

void Scheduler::SetSubmissionObserver(std::function<void(const SubmissionTiming&)> observer) {
    CheckThread();
    CheckError();
    submission_observer=std::move(observer);
}

std::uint64_t Scheduler::Submit(id<MTLDrawable> drawable, double minimum_present_duration) {
    if (!std::isfinite(minimum_present_duration) || minimum_present_duration < 0.0)
        throw std::invalid_argument("Invalid Metal drawable presentation duration");
    if (profiling) CheckThread();
    const ProfileCallTimer timer{ProfilingEnabled() ? &profiling->submit_ms : nullptr};
    if (ProfilingEnabled()) ++profiling->submit_calls;
    // Snapshot enumeration creates autoreleased strong arrays. GPU completion alone
    // cannot release them on a plain C++ thread without a draining pool. The feedback
    // block and submission slots retain every in-flight resource independently.
    @autoreleasepool {
        BeginCommands();
        if (next_tick == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Metal 4 submission timeline exhausted");
        }
        EndEncoding();
        if (drawable) {
            RetainObject(drawable);
            [queue waitForDrawable:drawable];
        }
        auto& slot = slots[slot_index];
        if (profiling) {
            auto& batch = profiling->batches[slot_index];
            if (batch.used) {
                batch.retained = slot.retained.count;
                batch.allocated_bytes = device.currentAllocatedSize;
            }
        }
        [slot.residency commit];
        [slot.commands useResidencySet:slot.residency];
        [slot.commands endCommandBuffer];

        const auto state = completion;
        // Metal 4 does not retain resources referenced by GPU addresses or resource IDs.
        // The callback owns an immutable snapshot, independent of this scheduler's lifetime.
        NSMutableArray* payload = [slot.retained.allObjects mutableCopy];
        [payload addObject:slot.residency];
        [payload addObject:slot.allocator];
        [payload addObject:slot.commands];
        [payload addObject:queue];
        [payload addObject:timeline];
        NSArray* retained = [payload copy];
        const auto feedback_tick = next_tick;
        const auto feedback_frame = current_frame;
        const auto observer = submission_observer;
        MTL4CommitOptions* options = [[MTL4CommitOptions alloc] init];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            (void)retained.count;
            if (feedback.error) {
                std::scoped_lock lock{state->mutex};
                if (state->error.empty()) {
                    NSString* description = [NSString stringWithFormat:
                        @"Metal 4 submission tick=%llu frame=%llu GPU interval=%.3f ms: %@",
                        static_cast<unsigned long long>(feedback_tick),
                        static_cast<unsigned long long>(feedback_frame),
                        (feedback.GPUEndTime - feedback.GPUStartTime) * 1000.0,
                        feedback.error.description];
                    const char* detail = description.UTF8String;
                    state->error = detail ? detail : "Metal 4 queue execution failed";
                }
            }
            if (observer) {
                // A diagnostic callback must not throw across Metal's native
                // completion boundary. Preserve its failure for the owner.
                try {
                    observer({feedback_tick, feedback.GPUStartTime, feedback.GPUEndTime,
                              feedback.error != nil});
                } catch (const std::exception& e) {
                    std::scoped_lock lock{state->mutex};
                    if (state->error.empty()) state->error=std::string{"Metal 4 submission observer: "}+e.what();
                } catch (...) {
                    std::scoped_lock lock{state->mutex};
                    if (state->error.empty()) state->error="Metal 4 submission observer failed";
                }
            }
        }];
        const id<MTL4CommandBuffer> batch[] = {slot.commands};
        [queue commit:batch count:1 options:options];
        const auto submitted_tick = next_tick++;
        slot.tick = submitted_tick;
        [queue signalEvent:timeline value:submitted_tick];
        if (drawable) {
            [queue signalDrawable:drawable];
            if (minimum_present_duration > 0.0)
                [drawable presentAfterMinimumDuration:minimum_present_duration];
            else
                [drawable present];
        }
        recording = false;
        slot_index = (slot_index + 1) % slots.size();
        return submitted_tick;
    }
}

void Scheduler::SubmitPending() {
    CheckThread();
    CheckError();
    if (recording) Submit();
}

bool Scheduler::SubmitPendingBeforeDrawable(bool may_wait) {
    CheckThread();
    CheckError();
    const auto completed = timeline.signaledValue;
    for (std::size_t index = 0; index < slots.size(); ++index) {
        auto& slot = slots[index];
        if ((recording && index == slot_index) || !slot.tick || slot.tick > completed ||
            !slot.retained.count) continue;
        // A completed slot can still own a presented drawable. Release its
        // resource set before nextDrawable, independently of allocator reuse:
        // there can be more submission slots than layer drawables. In-flight
        // feedback payloads retain their own snapshots, and live CPU encoding
        // stays untouched. Do not reset allocators or profiling heaps here.
        slot.used_resources.clear();
        [slot.retained removeAllObjects];
        [slot.residency removeAllAllocations];
        [slot.residency commit];
    }
    if (!may_wait || !recording || next_tick == 1 || completed >= next_tick - 1) {
        return false;
    }
    Submit();
    return true;
}

bool Scheduler::IsComplete(std::uint64_t tick) const {
    CheckThread();
    CheckError();
    return timeline.signaledValue >= tick;
}

std::uint64_t Scheduler::LastUseTick(id<MTLResource> resource) const {
    CheckThread();
    return [resource_ticks objectForKey:resource].unsignedLongLongValue;
}

NSUInteger Scheduler::CurrentRetainedObjectCount() const {
    CheckThread();
    return recording ? slots[slot_index].retained.count : 0;
}

std::uint64_t Scheduler::ArgumentUpdateCount() const {
    CheckThread();
    return argument_updates;
}

std::uint64_t Scheduler::EncoderCount() const {
    CheckThread();
    return encoder_count;
}

void Scheduler::Wait(std::uint64_t tick, WaitReason reason) {
    CheckThread();
    if (static_cast<unsigned>(reason) >= static_cast<unsigned>(WaitReason::Count))
        throw std::invalid_argument("Invalid Metal 4 wait reason");
    const ProfileCallTimer timer{ProfilingEnabled() ? &profiling->wait_ms : nullptr};
    if (ProfilingEnabled()) ++profiling->wait_calls;
    auto* stats = ProfilingEnabled() ? &profiling->waits[static_cast<unsigned>(reason)] : nullptr;
    const ProfileCallTimer reason_timer{stats ? &stats->host_ms : nullptr};
    if (stats) ++stats->calls;
    CheckError();
    if (!tick) {
        if (stats) ++stats->zero_tick;
        return;
    }
    if (tick == next_tick && recording) {
        if (stats) ++stats->submits;
        const ProfileCallTimer submit_timer{stats ? &stats->submit_ms : nullptr};
        Submit();
    }
    if (tick >= next_tick) {
        throw std::invalid_argument("Cannot wait for an unsubmitted Metal 4 timeline value");
    }
    // Like Vulkan's timeline fast path, retired submissions need no OS wait.
    // Keep completion/error validation even when this call observes a ready tick.
    if (timeline.signaledValue >= tick) {
        if (stats) ++stats->completed;
        CheckError();
        return;
    }
    const ProfileCallTimer signal_timer{stats ? &stats->signal_ms : nullptr};
    if (stats) ++stats->signal_waits;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (![timeline waitUntilSignaledValue:tick timeoutMS:100]) {
        CheckError();
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("Timed out waiting for Metal 4 GPU completion");
        }
    }
    CheckError();
}

void Scheduler::Finish(WaitReason reason, std::source_location source) {
    CheckThread();
    if (static_cast<unsigned>(reason) >= static_cast<unsigned>(WaitReason::Count))
        throw std::invalid_argument("Invalid Metal 4 finish reason");
    auto* site = ProfilingEnabled() && reason == WaitReason::Readback ? &profiling->Readback(source) : nullptr;
    const ProfileCallTimer site_timer{site ? &site->finish_ms : nullptr};
    if (site) ++site->calls;
    if (recording) {
        Submit();
    }
    Wait(next_tick - 1, reason);
    if (profiling) for (std::size_t index = 0; index < slots.size(); ++index) ResolveProfile(index);
}

bool Scheduler::FinishNoThrow() noexcept {
    try {
        Finish();
        return true;
    } catch (const std::exception& error) {
        NSLog(@"Azahar Metal 4 finish: %s", error.what());
    } catch (...) {
        NSLog(@"Azahar Metal 4 finish: unknown C++ exception");
    }
    return false;
}

} // namespace Metal
