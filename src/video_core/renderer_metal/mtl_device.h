// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__) || !defined(__OBJC__)
#error "The native Metal 4 backend requires Objective-C++ on Apple Silicon"
#endif

#import <Metal/Metal.h>
#include <mutex>
#include <shared_mutex>
#include <filesystem>
#include <memory>
#include "mtl_pipeline_disk_cache.h"
#include "mtl_compilation_queue.h"
#include <string_view>
#include <map>
#include <optional>
#include <condition_variable>
#include <thread>
#include <array>
#include <cstdint>

namespace Metal {

// Copies share an immutable completion identity. Replacing a future creates a
// new identity, so an async pipeline cache cannot confuse retries or recycled
// metadata addresses. The producer must be submitted before dependent jobs.
class NativeLibraryFuture final {
public:
    NativeLibraryFuture() = default;
    NativeLibraryFuture(std::shared_future<id<MTLLibrary>> result)
        : state{result.valid() ? std::make_shared<const std::shared_future<id<MTLLibrary>>>(std::move(result)) : nullptr} {}
    bool valid() const noexcept { return bool(state); }
    id<MTLLibrary> get() const {
        if (!state) throw std::future_error(std::future_errc::no_state);
        return state->get();
    }
    template<class Rep, class Period>
    std::future_status wait_for(const std::chrono::duration<Rep, Period>& duration) const {
        if (!state) throw std::future_error(std::future_errc::no_state);
        return state->wait_for(duration);
    }
    std::uintptr_t Identity() const noexcept {
        return reinterpret_cast<std::uintptr_t>(state.get());
    }
private:
    std::shared_ptr<const std::shared_future<id<MTLLibrary>>> state;
};

// Shared Metal objects (buffers, textures, libraries and pipeline states) also belong to Metal 4.
// All command encoding and pipeline compilation use the MTL4 APIs.
class Device final {
public:
    explicit Device(std::filesystem::path disk_cache_root={},bool disk_cache_enabled=true);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    id<MTLDevice> Handle() const noexcept {
        return device;
    }

    id<MTLBuffer> CreateBuffer(NSUInteger size, bool cpu_visible) const;
    id<MTLTexture> CreateTexture(MTLTextureDescriptor* descriptor) const;
    id<MTLTexture> CreateBufferTexture(id<MTLBuffer> buffer, NSUInteger offset,
                                       NSUInteger count, MTLPixelFormat format) const;
    id<MTLSamplerState> CreateSampler(MTLSamplerDescriptor* descriptor) const;
    id<MTLLibrary> CompileLibrary(std::string_view source, std::string_view name);
    id<MTLComputePipelineState> CompileCompute(MTL4ComputePipelineDescriptor* descriptor);
    id<MTLRenderPipelineState> CompileRender(MTL4RenderPipelineDescriptor* descriptor);
    id<MTLRenderPipelineState> CompileMesh(MTL4MeshRenderPipelineDescriptor* descriptor);
    template <typename Builder>
    auto DeferCompilation(Builder&& builder, std::size_t payload_bytes = 0) {
        return compilation_queue.Submit([build=std::forward<Builder>(builder)]() mutable {
            @autoreleasepool { return build(); }
        },payload_bytes);
    }
    void DrainCompilations() { compilation_queue.Drain(); }
    void SwitchDiskCache(std::uint64_t title_id,const std::atomic_bool* stop=nullptr);
    void SetDiskCacheEnabled(bool enabled);
    bool FlushDiskCache();
    // Coalesce periodic saves on one worker. Detached serializers are never
    // used by new compiler tasks while their archives are published.
    void RequestDiskCacheFlush();
    PipelineDiskCacheStats DiskCacheStats() const;
    std::filesystem::path ShaderDiskCacheDirectory() const;
    std::filesystem::path RenderDiskCacheDirectory() const;
    std::optional<std::string> LibraryIdentity(id<MTLLibrary> library) const;
    bool SaveLibrarySource(id<MTLLibrary> library, const std::filesystem::path& directory);
    id<MTLLibrary> LibraryWithIdentity(std::string_view identity);
    static std::string SourceIdentity(std::string_view name, std::string_view source);
    static MTL4LibraryFunctionDescriptor* Function(id<MTLLibrary> library,
                                                  std::string_view name);

private:
    id<MTLDevice> device;
    id<MTL4Compiler> compiler;
    CompilationQueue compilation_queue;
    // Compilers may run concurrently, but a serializer/title rotation must
    // exclude every native call that still uses the old compiler context.
    mutable std::shared_mutex compiler_lifecycle;
    mutable std::mutex compiler_mutex;
    std::unique_ptr<PipelineDiskCache> disk_cache;
    struct CacheDirectories {
        std::filesystem::path shader, render;
    };
    mutable std::mutex cache_directory_mutex;
    CacheDirectories cache_directories;
    std::atomic_bool requested_cache_enabled{};
    std::mutex flush_request_mutex;
    std::condition_variable flush_request_cv;
    bool flush_requested{}, flush_stopping{};
    std::thread flush_worker;
    std::mutex publication_mutex;
    std::optional<PipelineArchiveBatch> publication_retry;
    struct CompilerDiagnostics;
    struct CompilerOperation;
    std::unique_ptr<CompilerDiagnostics> compiler_diagnostics;
    struct LibraryRecord {
        __weak id<MTLLibrary> library;
        std::string identity;
        struct Source { std::string name, text; };
        std::shared_ptr<const Source> source;
        std::filesystem::path persisted_directory;
        std::array<std::uint64_t, 8> persisted_stamp{};
    };
    std::map<std::uintptr_t, LibraryRecord> library_identities;
    // Full name/source equality also deduplicates requests that arrive before
    // the first native library has been published. Failed builds are retryable.
    std::map<std::pair<std::string, std::string>, std::shared_future<id<MTLLibrary>>> pending_libraries;
    void CreateCompiler();
    void PublishCacheDirectories(); // compiler_mutex held, except during construction
    void FlushWorker();
    bool PersistLibrary(LibraryRecord& record, const std::filesystem::path& directory,
                        CompilerOperation* profile=nullptr);
    id<MTL4Compiler> UncapturedCompiler();
};

} // namespace Metal
