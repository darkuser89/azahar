// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#import <Metal/Metal.h>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <string>
#include <optional>

namespace Metal {
struct PipelineDiskCacheStats {
    bool enabled{};
    std::uint64_t title_id{},loaded_batches{},archive_hits{},compiled_pipelines{},saved_batches{},rejected_batches{},io_failures{};
    std::uint64_t resident_batches{},resident_archive_bytes{},pruned_batches{},pruned_archive_bytes{};
    std::uint64_t scanned_entries{},scan_limit_hits{},cancelled_scans{},hashed_archive_bytes{};
    std::string last_error;
};

struct PipelineArchiveBatch {
    id<MTL4PipelineDataSetSerializer> serializer;
    std::filesystem::path root, directory;
    std::uint64_t pipelines;
    std::filesystem::path temporary{};
    bool serialized{};
};
struct PipelineArchivePublication {
    bool saved{};
    PipelineDiskCacheStats stats;
};

// Called under Device's compiler mutex. Immutable checksum-checked batches are
// published by an atomic directory rename; a cache failure never prevents native compilation.
class PipelineDiskCache final {
public:
    static constexpr std::size_t ScanEntryLimit=4096;
    PipelineDiskCache(id<MTLDevice> device,std::filesystem::path root,bool enabled=true);
    void SetEnabled(bool enabled);
    void Switch(std::uint64_t title_id,const std::atomic_bool* stop=nullptr);
    bool Flush(const std::atomic_bool* stop=nullptr);
    // Rotate the active serializer under the compiler lock, then serialize,
    // checksum and prune the detached immutable batch without that lock.
    std::optional<PipelineArchiveBatch> DetachFlush();
    static PipelineArchivePublication Publish(PipelineArchiveBatch& batch,
                                               const std::atomic_bool* stop=nullptr);
    void Published(const PipelineArchivePublication& publication);
    void RestoreDetached(const PipelineArchiveBatch& batch);
    id<MTL4PipelineDataSetSerializer> Serializer() const {return serializer;}
    MTL4CompilerTaskOptions* TaskOptions() const;
    id<MTLComputePipelineState> Find(MTL4ComputePipelineDescriptor* descriptor);
    id<MTLRenderPipelineState> Find(MTL4RenderPipelineDescriptor* descriptor);
    id<MTLRenderPipelineState> Find(MTL4MeshRenderPipelineDescriptor* descriptor);
    void Compiled(bool capture=true) {++stats.compiled_pipelines;if(stats.enabled&&capture)++pending;}
    PipelineDiskCacheStats Stats() const {return stats;}
    std::filesystem::path ShaderDirectory() const {
        return stats.enabled ? current / "shader-source" : std::filesystem::path{};
    }
    std::filesystem::path RenderDirectory() const {
        return stats.enabled ? current / "render-pipelines" : std::filesystem::path{};
    }
private:
    void Load(const std::filesystem::path& directory,const std::atomic_bool* stop,std::size_t& remaining,
              std::uintmax_t& hash_remaining);
    void Prune(const std::atomic_bool* stop=nullptr);
    id<MTLDevice> device;
    id<MTL4PipelineDataSetSerializer> serializer;
    std::filesystem::path root,current;
    std::vector<id<MTL4Archive>> archives;
    PipelineDiskCacheStats stats;
    std::uint64_t pending{};
    bool instrumented{};
    std::optional<PipelineArchiveBatch> synchronous_retry;
};
} // namespace Metal
