// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_device.h"
#include <compare>
#include <cstdint>
#include <map>
#include <string>
#include <functional>
#include <span>

namespace Metal {

// Values follow the PICA register encodings used by the existing shader configuration.
struct BlendState {
    bool enabled{};
    std::uint8_t color_write_mask{15};
    std::uint8_t logic_operation{3};
    std::uint8_t source_rgb{1};
    std::uint8_t dest_rgb{};
    std::uint8_t source_alpha{1};
    std::uint8_t dest_alpha{};
    std::uint8_t rgb_equation{};
    std::uint8_t alpha_equation{};
    bool emulate_rgb_minmax{};
    bool emulate_alpha_minmax{};
    auto operator<=>(const BlendState&) const = default;
};

struct DepthStencilState {
    bool depth_test{};
    bool depth_write{};
    std::uint8_t depth_compare{1};
    bool stencil_test{};
    std::uint8_t stencil_compare{1};
    std::uint8_t stencil_fail{};
    std::uint8_t stencil_depth_fail{};
    std::uint8_t stencil_pass{};
    std::uint8_t stencil_read_mask{255};
    std::uint8_t stencil_write_mask{255};
    auto operator<=>(const DepthStencilState&) const = default;
};

struct RenderPipelineDescription {
    id<MTLLibrary> vertex_library;
    std::string vertex_entry;
    id<MTLLibrary> fragment_library;
    std::string fragment_entry;
    MTLPixelFormat color_format{MTLPixelFormatInvalid};
    BlendState blend;
    NativeLibraryFuture pending_vertex_library{}, pending_fragment_library{};
};
struct RenderPipelineRecord {
    std::string vertex_identity, vertex_entry, fragment_identity, fragment_entry;
    MTLPixelFormat color_format{MTLPixelFormatInvalid};
    BlendState blend;
    bool mesh{};
};
struct PipelinePrewarmResult {
    std::size_t candidates{}, visited{}, prepared{}, failed{};
    bool cancelled{}, budget_limited{};
};
struct PipelineCacheStats {
    std::uint64_t render_builds{}, prewarmed{}, prewarm_hits{}, rejected{}, manifest_failures{};
};
class RenderPipelineManifest;

// Deferred factories own their descriptions and publish complete states through
// shared futures. Synchronous clients use the same validated native factories.
class PipelineCache final {
public:
    explicit PipelineCache(Device& device);
    ~PipelineCache();
    id<MTLRenderPipelineState> Render(const RenderPipelineDescription& description);
    id<MTLRenderPipelineState> Mesh(const RenderPipelineDescription& description);
    std::shared_future<id<MTLRenderPipelineState>> RenderAsync(const RenderPipelineDescription& description);
    std::shared_future<id<MTLRenderPipelineState>> MeshAsync(const RenderPipelineDescription& description);
    id<MTLComputePipelineState> Compute(id<MTLLibrary> library, std::string_view entry,
                                        NSUInteger max_threads = 64);
    id<MTLDepthStencilState> DepthStencil(const DepthStencilState& state);
    PipelinePrewarmResult Prewarm(const std::atomic_bool& stop,
        const std::function<void(std::size_t, std::size_t)>& progress = {},
        std::span<const std::string_view> vertex_entries = {}, bool include_mesh = false);
    PipelineCacheStats Stats() const;
    void Clear();

private:
    struct RenderKey {
        std::uintptr_t vertex_library;
        std::string vertex_entry;
        std::uintptr_t fragment_library;
        std::string fragment_entry;
        MTLPixelFormat color_format;
        BlendState blend;
        bool pending_vertex{}, pending_fragment{};
        auto operator<=>(const RenderKey&) const = default;
    };
    struct ComputeKey {
        std::uintptr_t library;
        std::string entry;
        NSUInteger max_threads;
        auto operator<=>(const ComputeKey&) const = default;
    };
    struct RenderEntry {
        id<MTLRenderPipelineState> pipeline;
        id<MTLLibrary> vertex_library;
        id<MTLLibrary> fragment_library;
    };
    struct ComputeEntry {
        id<MTLComputePipelineState> pipeline;
        id<MTLLibrary> library;
    };
    struct DeferredEntry {
        std::shared_future<id<MTLRenderPipelineState>> result;
        // Pointer keys cannot outlive their library objects.
        id<MTLLibrary> vertex_library;
        id<MTLLibrary> fragment_library;
        NativeLibraryFuture pending_vertex{}, pending_fragment{};
    };
    Device& device;
    mutable std::mutex mutex;
    // Warm draw lookups and depth state never wait behind a native compiler API.
    std::mutex deferred_mutex, depth_mutex;
    std::filesystem::path deferred_directory;
    std::map<RenderKey, DeferredEntry> deferred_render, deferred_mesh;
    std::shared_future<id<MTLRenderPipelineState>> Request(const RenderPipelineDescription& description, bool mesh);
    std::map<RenderKey, RenderEntry> render_pipelines;
    std::map<RenderKey, RenderEntry> mesh_pipelines;
    std::map<ComputeKey, ComputeEntry> compute_pipelines;
    std::map<DepthStencilState, id<MTLDepthStencilState>> depth_states;
    std::map<std::string, RenderEntry> prewarmed_pipelines;
    std::filesystem::path prewarm_directory;
    std::uint64_t prewarm_epoch{};
    std::unique_ptr<RenderPipelineManifest> manifests;
    PipelineCacheStats stats;
    id<MTLRenderPipelineState> Compile(const RenderPipelineDescription& description, std::string_view label);
    id<MTLRenderPipelineState> CompileMesh(const RenderPipelineDescription& description, std::string_view label);
    void CheckDirectory(const std::filesystem::path& directory);
};

} // namespace Metal
