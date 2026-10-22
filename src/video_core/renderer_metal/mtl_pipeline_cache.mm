// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_pipeline_cache.h"
#include "mtl_pipeline_manifest.h"
#include <array>
#include <algorithm>
#include <stdexcept>

namespace Metal {

namespace {

MTLBlendOperation BlendEquation(std::uint8_t value) {
    constexpr std::array values{MTLBlendOperationAdd, MTLBlendOperationSubtract,
        MTLBlendOperationReverseSubtract, MTLBlendOperationMin, MTLBlendOperationMax};
    // Undefined PICA equations behave as add, matching the hardware-tested source behavior.
    return value < values.size() ? values[value] : MTLBlendOperationAdd;
}

MTLBlendFactor BlendFactor(std::uint8_t value) {
    constexpr std::array values{
        MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor, MTLBlendFactorDestinationColor,
        MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorSourceAlpha,
        MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha, MTLBlendFactorBlendColor,
        MTLBlendFactorOneMinusBlendColor, MTLBlendFactorBlendAlpha,
        MTLBlendFactorOneMinusBlendAlpha, MTLBlendFactorSourceAlphaSaturated,
    };
    return value < values.size() ? values[value] : MTLBlendFactorOne;
}

MTLCompareFunction Compare(std::uint8_t value) {
    constexpr std::array values{MTLCompareFunctionNever, MTLCompareFunctionAlways,
        MTLCompareFunctionEqual, MTLCompareFunctionNotEqual, MTLCompareFunctionLess,
        MTLCompareFunctionLessEqual, MTLCompareFunctionGreater, MTLCompareFunctionGreaterEqual};
    if (value >= values.size()) {
        throw std::invalid_argument("Invalid PICA comparison function");
    }
    return values[value];
}

MTLStencilOperation Stencil(std::uint8_t value) {
    constexpr std::array values{MTLStencilOperationKeep, MTLStencilOperationZero,
        MTLStencilOperationReplace, MTLStencilOperationIncrementClamp,
        MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
        MTLStencilOperationIncrementWrap, MTLStencilOperationDecrementWrap};
    if (value >= values.size()) {
        throw std::invalid_argument("Invalid PICA stencil operation");
    }
    return values[value];
}

MTLColorWriteMask ColorMask(std::uint8_t value) {
    if (value >= 16) {
        throw std::invalid_argument("Invalid PICA color write mask");
    }
    // PICA numbers R..A from low to high, while Metal numbers them from high to low.
    return MTLColorWriteMask((value & 1 ? MTLColorWriteMaskRed : 0) |
                             (value & 2 ? MTLColorWriteMaskGreen : 0) |
                             (value & 4 ? MTLColorWriteMaskBlue : 0) |
                             (value & 8 ? MTLColorWriteMaskAlpha : 0));
}

std::uintptr_t Identity(id object) {
    return reinterpret_cast<std::uintptr_t>((__bridge void*)object);
}

bool ContainsEntry(id<MTLLibrary> library, std::string_view entry) {
    auto* name = [[NSString alloc] initWithBytes:entry.data() length:entry.size() encoding:NSUTF8StringEncoding];
    return name && [library.functionNames containsObject:name];
}

} // namespace

PipelineCache::PipelineCache(Device& device_) : device{device_}, manifests{std::make_unique<RenderPipelineManifest>()} {}
PipelineCache::~PipelineCache() {
    // Factory captures are owned until their final publication, including errors.
    device.DrainCompilations();
}

std::shared_future<id<MTLRenderPipelineState>> PipelineCache::RenderAsync(const RenderPipelineDescription& info) {
    return Request(info, false);
}

std::shared_future<id<MTLRenderPipelineState>> PipelineCache::MeshAsync(const RenderPipelineDescription& info) {
    return Request(info, true);
}

std::shared_future<id<MTLRenderPipelineState>> PipelineCache::Request(const RenderPipelineDescription& info, bool mesh) {
    const bool pending_vertex = info.pending_vertex_library.valid();
    const bool pending_fragment = info.pending_fragment_library.valid();
    if ((!info.vertex_library && !pending_vertex) || (info.vertex_library && pending_vertex) ||
        (info.fragment_library && pending_fragment) ||
        (info.vertex_library && info.vertex_library.device != device.Handle()) || info.vertex_entry.empty() ||
        (info.fragment_library && info.fragment_library.device != device.Handle()) ||
        ((info.fragment_library || pending_fragment) && info.fragment_entry.empty()) ||
        info.blend.logic_operation >= 16)
        throw std::invalid_argument("Invalid deferred Metal pipeline description");
    const RenderKey key{pending_vertex ? info.pending_vertex_library.Identity() : Identity(info.vertex_library),
        info.vertex_entry, pending_fragment ? info.pending_fragment_library.Identity() : Identity(info.fragment_library),
        info.fragment_entry, info.color_format, info.blend, pending_vertex, pending_fragment};
    std::scoped_lock lock{deferred_mutex};
    const auto directory = device.RenderDiskCacheDirectory();
    if (directory != deferred_directory) {
        deferred_render.clear(); deferred_mesh.clear(); deferred_directory = directory;
    }
    auto& entries = mesh ? deferred_mesh : deferred_render;
    if (auto found = entries.find(key); found != entries.end()) {
        // Failed factories retain their original error for existing waiters,
        // while a later request is allowed to retry.
        if (found->second.result.wait_for(std::chrono::seconds{0}) != std::future_status::ready)
            return found->second.result;
        try { (void)found->second.result.get(); return found->second.result; }
        catch (...) { entries.erase(found); }
    }
    auto result = device.DeferCompilation([this, info, mesh] {
        // Library producers are queued before their dependent PSO. Resolve them
        // without holding a native cache/compiler lock, then use the ordinary
        // validated native key so independent futures still deduplicate.
        auto ready = info;
        if (ready.pending_vertex_library.valid()) {
            ready.vertex_library = ready.pending_vertex_library.get();
            if (!ready.vertex_library) throw std::invalid_argument("Deferred vertex library is nil");
            ready.pending_vertex_library = {};
        }
        if (ready.pending_fragment_library.valid()) {
            ready.fragment_library = ready.pending_fragment_library.get();
            if (!ready.fragment_library) throw std::invalid_argument("Deferred fragment library is nil");
            ready.pending_fragment_library = {};
        }
        // Report invalid metadata through the future before Metal validation
        // can abort the process for a missing native entry point.
        if (!ContainsEntry(ready.vertex_library, ready.vertex_entry) ||
            (ready.fragment_library && !ContainsEntry(ready.fragment_library, ready.fragment_entry)))
            throw std::invalid_argument("Deferred library does not export the requested entry point");
        return mesh ? Mesh(ready) : Render(ready);
    }, info.vertex_entry.size() + info.fragment_entry.size());
    entries.emplace(key, DeferredEntry{result, info.vertex_library, info.fragment_library,
        info.pending_vertex_library, info.pending_fragment_library});
    return result;
}

void PipelineCache::CheckDirectory(const std::filesystem::path& directory) {
    if (directory == prewarm_directory) return;
    render_pipelines.clear();
    mesh_pipelines.clear();
    prewarmed_pipelines.clear(); prewarm_directory = directory; ++prewarm_epoch;
}

PipelineCacheStats PipelineCache::Stats() const {
    std::scoped_lock lock{mutex};
    auto result = stats;
    result.rejected = manifests->Rejected(); result.manifest_failures = manifests->Failures();
    return result;
}

id<MTLRenderPipelineState> PipelineCache::Render(const RenderPipelineDescription& info) {
    if (info.pending_vertex_library.valid() || info.pending_fragment_library.valid() ||
        !info.vertex_library || info.vertex_library.device != device.Handle() || info.vertex_entry.empty() ||
        (info.fragment_library && info.fragment_library.device != device.Handle()) ||
        (info.fragment_library && info.fragment_entry.empty()) || info.blend.logic_operation >= 16) {
        throw std::invalid_argument("Invalid native render pipeline shader or blend description");
    }
    const RenderKey key{Identity(info.vertex_library), info.vertex_entry, Identity(info.fragment_library),
                         info.fragment_entry, info.color_format, info.blend};
    std::scoped_lock lock{mutex};
    const auto directory = device.RenderDiskCacheDirectory();
    CheckDirectory(directory);
    if (auto found = render_pipelines.find(key); found != render_pipelines.end()) {
        return found->second.pipeline;
    }
    const auto vertex_identity = device.LibraryIdentity(info.vertex_library);
    const auto fragment_identity = info.fragment_library ? device.LibraryIdentity(info.fragment_library) : std::optional<std::string>{std::string{}};
    std::optional<RenderPipelineRecord> record;
    std::string stable_key;
    if (vertex_identity && fragment_identity) {
        record = RenderPipelineRecord{*vertex_identity, info.vertex_entry, *fragment_identity, info.fragment_entry,
                                      info.color_format, info.blend};
        try { stable_key = RenderPipelineManifest::Identity(*record); }
        catch (const std::exception&) { record.reset(); }
    }
    id<MTLRenderPipelineState> pipeline = nil;
    if (!stable_key.empty()) if (const auto found = prewarmed_pipelines.find(stable_key); found != prewarmed_pipelines.end()) {
        pipeline = found->second.pipeline; ++stats.prewarm_hits;
    }
    if (!pipeline) pipeline = Compile(info, stable_key.empty() ? "Azahar Metal 4 PICA render pipeline" : "azm4-rp1-" + stable_key);
    if (record && !directory.empty() && device.SaveLibrarySource(info.vertex_library, directory) &&
        (!info.fragment_library || device.SaveLibrarySource(info.fragment_library, directory)))
        manifests->Save(directory, *record);
    // Strong library references prevent object-address reuse from aliasing a live cache key.
    render_pipelines.emplace(key, RenderEntry{pipeline, info.vertex_library, info.fragment_library});
    return pipeline;
}

id<MTLRenderPipelineState> PipelineCache::Mesh(const RenderPipelineDescription& info) {
    if (info.pending_vertex_library.valid() || info.pending_fragment_library.valid() ||
        !info.vertex_library || info.vertex_library.device != device.Handle() || info.vertex_entry.empty() ||
        (info.fragment_library && info.fragment_library.device != device.Handle()) ||
        (info.fragment_library && info.fragment_entry.empty()) || info.blend.logic_operation >= 16)
        throw std::invalid_argument("Invalid native mesh pipeline description");
    const RenderKey key{Identity(info.vertex_library), info.vertex_entry, Identity(info.fragment_library),
                        info.fragment_entry, info.color_format, info.blend};
    std::scoped_lock lock{mutex};
    const auto directory = device.RenderDiskCacheDirectory();
    CheckDirectory(directory);
    if (const auto found = mesh_pipelines.find(key); found != mesh_pipelines.end()) return found->second.pipeline;
    const auto vertex_identity = device.LibraryIdentity(info.vertex_library);
    const auto fragment_identity = info.fragment_library ? device.LibraryIdentity(info.fragment_library) : std::optional<std::string>{std::string{}};
    std::optional<RenderPipelineRecord> record;
    std::string stable_key;
    if (vertex_identity && fragment_identity) {
        record = RenderPipelineRecord{*vertex_identity, info.vertex_entry, *fragment_identity, info.fragment_entry,
                                      info.color_format, info.blend, true};
        try { stable_key = RenderPipelineManifest::Identity(*record); }
        catch (const std::exception&) { record.reset(); }
    }
    id<MTLRenderPipelineState> pipeline = nil;
    if (!stable_key.empty()) if (const auto found = prewarmed_pipelines.find(stable_key); found != prewarmed_pipelines.end()) {
        pipeline = found->second.pipeline; ++stats.prewarm_hits;
    }
    if (!pipeline) pipeline = CompileMesh(info, stable_key.empty() ? "Azahar Metal 4 PICA mesh raster" : "azm4-mp1-" + stable_key);
    if (record && !directory.empty() && device.SaveLibrarySource(info.vertex_library, directory) &&
        (!info.fragment_library || device.SaveLibrarySource(info.fragment_library, directory)))
        manifests->Save(directory, *record);
    mesh_pipelines.emplace(key, RenderEntry{pipeline, info.vertex_library, info.fragment_library});
    return pipeline;
}

id<MTLRenderPipelineState> PipelineCache::CompileMesh(const RenderPipelineDescription& info, std::string_view label) {
    auto* descriptor = [[MTL4MeshRenderPipelineDescriptor alloc] init];
    descriptor.label = [[NSString alloc] initWithBytes:label.data() length:label.size() encoding:NSUTF8StringEncoding];
    descriptor.meshFunctionDescriptor = Device::Function(info.vertex_library, info.vertex_entry);
    if (info.fragment_library) descriptor.fragmentFunctionDescriptor = Device::Function(info.fragment_library, info.fragment_entry);
    else descriptor.rasterizationEnabled = NO;
    descriptor.rasterSampleCount = 1;
    descriptor.maxTotalThreadsPerMeshThreadgroup = 96;
    auto* color = descriptor.colorAttachments[0];
    color.pixelFormat = info.color_format;
    color.writeMask = ColorMask(info.blend.color_write_mask);
    if (!info.blend.enabled && info.blend.logic_operation == 6) {
        color.writeMask = MTLColorWriteMaskNone;
    }
    color.blendingState = info.blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
    color.sourceRGBBlendFactor = info.blend.emulate_rgb_minmax ? MTLBlendFactorOne : BlendFactor(info.blend.source_rgb);
    color.destinationRGBBlendFactor = info.blend.emulate_rgb_minmax ? MTLBlendFactorZero : BlendFactor(info.blend.dest_rgb);
    color.rgbBlendOperation = info.blend.emulate_rgb_minmax ? MTLBlendOperationAdd : BlendEquation(info.blend.rgb_equation);
    color.sourceAlphaBlendFactor = info.blend.emulate_alpha_minmax ? MTLBlendFactorOne : BlendFactor(info.blend.source_alpha);
    color.destinationAlphaBlendFactor = info.blend.emulate_alpha_minmax ? MTLBlendFactorZero : BlendFactor(info.blend.dest_alpha);
    color.alphaBlendOperation = info.blend.emulate_alpha_minmax ? MTLBlendOperationAdd : BlendEquation(info.blend.alpha_equation);
    ++stats.render_builds;
    return device.CompileMesh(descriptor);
}

id<MTLRenderPipelineState> PipelineCache::Compile(const RenderPipelineDescription& info, std::string_view label) {
    MTL4RenderPipelineDescriptor* descriptor = [[MTL4RenderPipelineDescriptor alloc] init];
    descriptor.label = Device::Function(info.vertex_library, label).name;
    descriptor.vertexFunctionDescriptor = Device::Function(info.vertex_library, info.vertex_entry);
    if (info.fragment_library) {
        descriptor.fragmentFunctionDescriptor = Device::Function(info.fragment_library, info.fragment_entry);
    } else {
        descriptor.rasterizationEnabled = NO;
    }
    descriptor.rasterSampleCount = 1;
    descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    // Metal 4 derives depth/stencil attachment formats from the render pass, not the pipeline.
    auto* color = descriptor.colorAttachments[0];
    color.pixelFormat = info.color_format;
    color.writeMask = ColorMask(info.blend.color_write_mask);
    if (!info.blend.enabled && info.blend.logic_operation == 6) {
        color.writeMask = MTLColorWriteMaskNone;
    }
    color.blendingState = info.blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
    color.sourceRGBBlendFactor = info.blend.emulate_rgb_minmax ? MTLBlendFactorOne : BlendFactor(info.blend.source_rgb);
    color.destinationRGBBlendFactor = info.blend.emulate_rgb_minmax ? MTLBlendFactorZero : BlendFactor(info.blend.dest_rgb);
    color.rgbBlendOperation = info.blend.emulate_rgb_minmax ? MTLBlendOperationAdd : BlendEquation(info.blend.rgb_equation);
    color.sourceAlphaBlendFactor = info.blend.emulate_alpha_minmax ? MTLBlendFactorOne : BlendFactor(info.blend.source_alpha);
    color.destinationAlphaBlendFactor = info.blend.emulate_alpha_minmax ? MTLBlendFactorZero : BlendFactor(info.blend.dest_alpha);
    color.alphaBlendOperation = info.blend.emulate_alpha_minmax ? MTLBlendOperationAdd : BlendEquation(info.blend.alpha_equation);
    ++stats.render_builds;
    return device.CompileRender(descriptor);
}

PipelinePrewarmResult PipelineCache::Prewarm(const std::atomic_bool& stop,
    const std::function<void(std::size_t, std::size_t)>& progress,
    std::span<const std::string_view> vertex_entries, bool include_mesh) {
    PipelinePrewarmResult result{};
    if (stop.load()) { result.cancelled = true; return result; }
    const auto directory = device.RenderDiskCacheDirectory();
    if (directory.empty()) return result;
    std::vector<HistoricalPipelineHint> records;
    std::uint64_t epoch{};
    {
        std::scoped_lock lock{mutex};
        CheckDirectory(directory); epoch = ++prewarm_epoch;
        try {
            std::size_t remaining_visits=16384;
            auto current=manifests->Load(directory,stop,result.budget_limited,&remaining_visits);
            for(auto& record:current)records.push_back({std::move(record),{}, {}});
            bool limited=false;
            auto historical=manifests->LoadHistorical(directory,stop,limited,4096-records.size(),&remaining_visits);
            result.budget_limited=result.budget_limited||limited;
            for(auto& hint:historical)records.push_back(std::move(hint));
        }
        catch (const std::exception&) { ++result.failed; }
        if (!vertex_entries.empty()) std::erase_if(records, [&](const auto& hint) {
            if (hint.record.mesh) return !include_mesh;
            return std::find(vertex_entries.begin(), vertex_entries.end(), hint.record.vertex_entry) == vertex_entries.end();
        });
    }
    result.candidates = records.size();
    if (progress) progress(0, result.candidates);
    for (const auto& hint : records) {
        if (stop.load()) { result.cancelled = true; break; }
        {
            std::scoped_lock lock{mutex};
            if (epoch != prewarm_epoch || directory != device.RenderDiskCacheDirectory()) {
                result.cancelled = true; break;
            }
            ++result.visited;
            auto record=hint.record;
            // Only current complete keys enter the ready map. Historical identities
            // validate source files; they never select a pipeline for a new draw.
            if(hint.vertex_source)record.vertex_identity=Device::SourceIdentity(hint.vertex_source->first,hint.vertex_source->second);
            if(hint.fragment_source)record.fragment_identity=Device::SourceIdentity(hint.fragment_source->first,hint.fragment_source->second);
            const auto key = RenderPipelineManifest::Identity(record);
            if (!prewarmed_pipelines.contains(key)) {
                try {
                    auto vertex = hint.vertex_source ? device.CompileLibrary(hint.vertex_source->second,hint.vertex_source->first) :
                                                       device.LibraryWithIdentity(record.vertex_identity);
                    auto fragment = hint.fragment_source ? device.CompileLibrary(hint.fragment_source->second,hint.fragment_source->first) :
                                    record.fragment_identity.empty() ? nil : device.LibraryWithIdentity(record.fragment_identity);
                    if (!vertex || (!record.fragment_identity.empty() && !fragment))
                        throw std::runtime_error("Native pipeline manifest has no matching source library");
                    const auto contains = [](id<MTLLibrary> library, std::string_view entry) {
                        auto* name = [[NSString alloc] initWithBytes:entry.data() length:entry.size()
                                                          encoding:NSUTF8StringEncoding];
                        return name && [library.functionNames containsObject:name];
                    };
                    // A valid file checksum cannot establish that an entry still
                    // exists. Reject stale hints before API validation sees them.
                    if (!contains(vertex, record.vertex_entry) ||
                        (fragment && !contains(fragment, record.fragment_entry)))
                        throw std::runtime_error("Native pipeline manifest function is not exported");
                    const RenderPipelineDescription description{vertex, record.vertex_entry, fragment,
                        record.fragment_entry, record.color_format, record.blend};
                    auto pipeline = record.mesh ? CompileMesh(description, "azm4-mp1-" + key) :
                                                  Compile(description, "azm4-rp1-" + key);
                    if(epoch!=prewarm_epoch||directory!=device.RenderDiskCacheDirectory()) {
                        result.cancelled=true;break;
                    }
                    prewarmed_pipelines.emplace(key, RenderEntry{pipeline, vertex, fragment});
                    if(hint.vertex_source)manifests->Save(directory,record);
                    ++stats.prewarmed; ++result.prepared;
                } catch (const std::exception&) { ++result.failed; }
            }
        }
        if (progress) progress(result.visited, result.candidates);
    }
    result.cancelled = result.cancelled || stop.load();
    return result;
}

id<MTLComputePipelineState> PipelineCache::Compute(id<MTLLibrary> library, std::string_view entry,
                                                  NSUInteger max_threads) {
    if (!library || library.device != device.Handle() || entry.empty() || !max_threads) {
        throw std::invalid_argument("Invalid native compute pipeline description");
    }
    const ComputeKey key{Identity(library), std::string{entry}, max_threads};
    std::scoped_lock lock{mutex};
    if (auto found = compute_pipelines.find(key); found != compute_pipelines.end()) {
        return found->second.pipeline;
    }
    MTL4ComputePipelineDescriptor* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.label = @"Azahar Metal 4 PICA compute pipeline";
    descriptor.computeFunctionDescriptor = Device::Function(library, entry);
    descriptor.maxTotalThreadsPerThreadgroup = max_threads;
    auto pipeline = device.CompileCompute(descriptor);
    compute_pipelines.emplace(key, ComputeEntry{pipeline, library});
    return pipeline;
}

id<MTLDepthStencilState> PipelineCache::DepthStencil(const DepthStencilState& state) {
    std::scoped_lock lock{depth_mutex};
    if (auto found = depth_states.find(state); found != depth_states.end()) {
        return found->second;
    }
    MTLDepthStencilDescriptor* descriptor = [[MTLDepthStencilDescriptor alloc] init];
    descriptor.label = @"Azahar Metal 4 PICA depth/stencil state";
    descriptor.depthCompareFunction = state.depth_test ? Compare(state.depth_compare) : MTLCompareFunctionAlways;
    descriptor.depthWriteEnabled = state.depth_write;
    if (state.stencil_test) {
        MTLStencilDescriptor* stencil = [[MTLStencilDescriptor alloc] init];
        stencil.stencilCompareFunction = Compare(state.stencil_compare);
        stencil.stencilFailureOperation = Stencil(state.stencil_fail);
        stencil.depthFailureOperation = Stencil(state.stencil_depth_fail);
        stencil.depthStencilPassOperation = Stencil(state.stencil_pass);
        stencil.readMask = state.stencil_read_mask;
        stencil.writeMask = state.stencil_write_mask;
        descriptor.frontFaceStencil = stencil;
        descriptor.backFaceStencil = stencil;
    }
    auto depth_stencil = [device.Handle() newDepthStencilStateWithDescriptor:descriptor];
    if (!depth_stencil) {
        throw std::runtime_error("Unable to create native PICA depth/stencil state");
    }
    depth_states.emplace(state, depth_stencil);
    return depth_stencil;
}

void PipelineCache::Clear() {
    std::scoped_lock deferred_lock{deferred_mutex};
    device.DrainCompilations();
    deferred_render.clear(); deferred_mesh.clear();
    std::scoped_lock depth_lock{depth_mutex};
    std::scoped_lock lock{mutex};
    render_pipelines.clear();
    mesh_pipelines.clear();
    compute_pipelines.clear();
    depth_states.clear();
    prewarmed_pipelines.clear(); ++prewarm_epoch;
}

} // namespace Metal
