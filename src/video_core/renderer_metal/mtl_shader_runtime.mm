// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_shader_runtime.h"
#include "mtl_shader_machine.h"
#include <cstring>
#include <stdexcept>

namespace Metal {
namespace {
void KeyWord(std::vector<std::byte>& key, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) key.push_back(std::byte(value >> (8 * i)));
}
void ValidateEntry(id<MTLLibrary> library, const TranslatedShader& shader) {
    if (![library.functionNames containsObject:Device::Function(library, shader.entry_point).name])
        throw std::invalid_argument("Cached PICA shader entry point is absent from its native library");
}
std::size_t ShaderPayloadBytes(const TranslatedShader& shader,std::size_t key_bytes) {
    std::size_t bytes=shader.source.size()+shader.entry_point.size()+shader.spirv.size()*sizeof(std::uint32_t)+key_bytes;
    for(const auto& resource:shader.resources) {
        bytes+=sizeof(resource)+resource.name.size()+resource.members.size()*sizeof(ShaderMemberLayout);
        for(const auto& member:resource.members)bytes+=member.name.size();
    }
    return bytes;
}
template<class Future>
bool LibraryFailed(const Future& pending) {
    if(!pending.valid() || pending.wait_for(std::chrono::seconds{0})!=std::future_status::ready)return false;
    try{(void)pending.get();return false;}catch(...){return true;}
}
} // namespace

ShaderRuntime::ShaderRuntime(Device& device_) : device{device_} {}
ShaderRuntime::~ShaderRuntime() {
    if(mesh_stats.owned_hits || mesh_stats.owned_registrations)
        LOG_INFO(Render,"Metal4 owned mesh lookup: hits={}, registrations={}, content_lookups={}, entries={}",
            mesh_stats.owned_hits,mesh_stats.owned_registrations,mesh_stats.content_lookups,owned_mesh.size());
    const auto sources=source_store.Stats();
    if(sources.saved||sources.rejected||sources.failures)
        LOG_INFO(Render,"Metal4 source input store: saved={}, rejected={}, failures={}",
                 sources.saved,sources.rejected,sources.failures);
}

void ShaderRuntime::CheckPrewarmDirectory(const std::filesystem::path& directory) {
    if (directory == prewarm_directory) return;
    prewarmed_fragments.clear(); prewarmed_vertices.clear(); prewarmed_mesh_vertices.clear();
    prewarm_directory = directory;
    ++prewarm_epoch;
}

id<MTLLibrary> ShaderRuntime::CompileLibrary(std::string_view source, std::string_view name) {
    // Explicit title preload validates native results before publishing ready
    // objects. On-demand IR hits instead rebuild/compare canonical CPU metadata
    // and publish owned native futures without waiting on the draw owner.
    auto library=device.DeferCompilation([native=&device,source=std::string(source),name=std::string(name)] {
        return native->CompileLibrary(source,name);
    },source.size()+name.size()).get();
    disk_cache->LibraryCompiled();return library;
}

std::shared_ptr<const VertexCaptureProgram> ShaderRuntime::CompileVertex(TranslatedShader shader) {
    auto owned=std::make_shared<const TranslatedShader>(std::move(shader));
    const auto native=device.DeferCompilation([native=&device,cache=disk_cache,owned] {
        auto library=native->CompileLibrary(owned->source,"Azahar Metal 4 PICA vertex program");
        cache->LibraryCompiled();ValidateEntry(library,*owned);
        auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor=Device::Function(library,owned->entry_point);
        descriptor.maxTotalThreadsPerThreadgroup=64;
        return std::pair<id<MTLComputePipelineState>,id<MTLLibrary>>{native->CompileCompute(descriptor),library};
    },ShaderPayloadBytes(*owned,0)).get();
    return std::make_shared<const VertexCaptureProgram>(VertexCaptureProgram{*owned,native.first,native.second});
}

ShaderPrewarmResult ShaderRuntime::Prewarm(const std::atomic_bool& stop,
    const std::function<void(std::size_t, std::size_t)>& progress) {
    ShaderPrewarmResult result{};
    if (stop.load()) { result.cancelled = true; return result; }
    // Include validated records still being published by owned on-demand
    // factories. Never drain while holding the runtime/cache lookup mutex.
    device.DrainCompilations();
    const auto directory = device.ShaderDiskCacheDirectory();
    if (directory.empty()) return result;
    std::vector<ShaderDiskEntry> entries;
    std::uint64_t epoch{};
    {
        std::scoped_lock lock{mutex};
        CheckPrewarmDirectory(directory);
        epoch = ++prewarm_epoch;
        try { entries = disk_cache->Entries(directory, stop, result.budget_limited); }
        catch (const std::exception&) { disk_cache->Reject(); ++result.failed; }
    }
    result.candidates = entries.size();
    if (progress) progress(0, result.candidates);
    const auto sources=source_store.Prewarm(directory,stop,[&] {
        std::scoped_lock lock{mutex};
        return epoch==prewarm_epoch&&directory==device.ShaderDiskCacheDirectory();
    });
    result.source_candidates=sources.candidates;result.source_prepared=sources.prepared;result.source_failed=sources.failed;
    result.budget_limited=result.budget_limited||sources.budget_limited;
    if(sources.candidates)
        LOG_INFO(Render,"Metal4 source input prewarm: candidates={}, prepared={}, failed={}, cancelled={}, limited={}",
                 sources.candidates,sources.prepared,sources.failed,sources.cancelled,sources.budget_limited);
    if(sources.cancelled){result.cancelled=true;return result;}
    constexpr std::uintmax_t MaxBytes = 256ULL * 1024 * 1024;
    std::uintmax_t bytes{};
    for (const auto& entry : entries) {
        if (stop.load()) { result.cancelled = true; break; }
        if (entry.bytes > MaxBytes - bytes) { result.budget_limited = true; break; }
        bytes += entry.bytes;
        {
            std::scoped_lock lock{mutex};
            if (epoch != prewarm_epoch || directory != device.ShaderDiskCacheDirectory()) {
                result.cancelled = true; break;
            }
            ++result.visited;
            if (!prewarmed_fragments.contains(entry.identity) && !prewarmed_vertices.contains(entry.identity) && !prewarmed_mesh_vertices.contains(entry.identity)) {
                auto record = disk_cache->Load(entry);
                if (!record) ++result.failed;
                else try {
                    if (record->shader.stage == ShaderStage::VertexCompute) {
                        prewarmed_vertices.emplace(entry.identity, CompileVertex(std::move(record->shader)));
                    } else if (record->shader.stage == ShaderStage::VertexMesh) {
                        auto library = CompileLibrary(record->shader.source, "Azahar Metal 4 PICA mesh program");
                        ValidateEntry(library, record->shader);
                        prewarmed_mesh_vertices.emplace(entry.identity, std::make_shared<const CompiledMeshVertexProgram>(
                            CompiledMeshVertexProgram{std::move(record->shader), library}));
                    } else if (record->shader.stage == ShaderStage::Fragment) {
                        auto library = CompileLibrary(record->shader.source, "Azahar Metal 4 PICA fragment program");
                        ValidateEntry(library, record->shader);
                        auto compiled = std::make_shared<const CompiledFragmentShader>(
                            CompiledFragmentShader{{std::move(record->shader), record->operations}, library});
                        prewarmed_fragments.emplace(entry.identity, std::move(compiled));
                    } else throw std::invalid_argument("Unsupported preloaded PICA shader stage");
                    disk_cache->Prewarmed(); ++result.compiled;
                } catch (const std::exception&) { disk_cache->Reject(); ++result.failed; }
            }
        }
        // The client may query stats, clear the cache or cancel from its callback.
        if (progress) progress(result.visited, result.candidates);
    }
    result.cancelled = result.cancelled || stop.load();
    return result;
}

std::shared_ptr<const CompiledFragmentShader> ShaderRuntime::Fragment(
    const Pica::Shader::FSConfig& config, const Pica::Shader::UserConfig& user,
    FragmentColorFormat color_format, const PicaSamplerConfigs& samplers, bool gas_accumulation) {
    static_assert(std::has_unique_object_representations_v<Pica::Shader::FSConfig>);
    Key key{.config = {}, .user = user.raw, .color_format = color_format, .samplers = samplers,
            .gas_accumulation = gas_accumulation};
    std::memcpy(key.config.data(), &config, sizeof(config));
    std::scoped_lock lock{mutex};
    if (const auto found = fragments.find(key); found != fragments.end()) {
        const auto& pending = found->second->pending_library;
        if (!pending.valid() || pending.wait_for(std::chrono::seconds{0}) != std::future_status::ready)
            return found->second;
        try { (void)pending.get(); return found->second; }
        catch (...) { fragments.erase(found); } // Existing waiters retain the original failure; new requests retry.
    }
    std::vector<std::byte> disk_key;
    KeyWord(disk_key, std::uint32_t(ShaderStage::Fragment));
    KeyWord(disk_key, gas_accumulation);
    disk_key.insert(disk_key.end(), key.config.begin(), key.config.end());
    KeyWord(disk_key, key.user); KeyWord(disk_key, std::uint32_t(key.color_format));
    for (const auto& sampler : samplers) {
        KeyWord(disk_key, sampler.wrap_s); KeyWord(disk_key, sampler.wrap_t);
        KeyWord(disk_key, sampler.min_filter); KeyWord(disk_key, sampler.mag_filter);
        KeyWord(disk_key, sampler.mip_filter); KeyWord(disk_key, sampler.lod_min);
        KeyWord(disk_key, sampler.lod_max);
    }
    const auto directory = device.ShaderDiskCacheDirectory();
    CheckPrewarmDirectory(directory);
    if (const auto found = prewarmed_fragments.find(ShaderDiskCache::Identity(disk_key));
        found != prewarmed_fragments.end()) {
        disk_cache->PrewarmHit();
        fragments.emplace(key, found->second);
        return found->second;
    }
    auto cached = disk_cache->Load(directory, disk_key, ShaderStage::Fragment);
    PicaFragmentProgram program{};
    bool cached_valid{};
    if (cached) {
        try {
            program = RestorePicaFragmentShader(cached->shader.spirv, config, user, color_format, gas_accumulation);
            if (program.shader != cached->shader || program.operations != cached->operations)
                throw std::invalid_argument("Cached PICA fragment source/reflection differs from its IR");
            cached_valid = true;
        } catch (const std::exception&) { disk_cache->Reject(); }
    }
    if (!cached_valid) {
        TranslationInputScope capture{[&](auto source, auto stage, auto bindings) {
            source_store.Save(directory, source, stage, bindings);
        }};
        program = GeneratePicaFragmentShader(config, user, color_format, samplers, gas_accumulation);
        disk_cache->Translated();
    }
    // Metadata is complete before native compilation. Texture/LUT binding and
    // uniform uploads can proceed while this owned factory builds the library.
    auto complete = std::make_shared<CompiledFragmentShader>(CompiledFragmentShader{std::move(program), nil});
    const auto& shader = complete->program.shader;
    std::size_t payload = shader.source.size() + shader.entry_point.size() +
        shader.spirv.size() * sizeof(std::uint32_t) + disk_key.size();
    for (const auto& resource : shader.resources) {
        payload += sizeof(resource) + resource.name.size() + resource.members.size() * sizeof(ShaderMemberLayout);
        for (const auto& member : resource.members) payload += member.name.size();
    }
    // The callable must not retain the object that owns its own future.
    // A separately owned immutable input also survives Clear/runtime teardown.
    auto native_program = std::make_shared<const PicaFragmentProgram>(complete->program);
    complete->pending_library = device.DeferCompilation(
        [native = &device, cache = disk_cache, native_program, directory, disk_key = std::move(disk_key), save = !cached_valid] {
            auto library = native->CompileLibrary(native_program->shader.source,
                "Azahar Metal 4 PICA fragment program");
            ValidateEntry(library, native_program->shader);
            cache->LibraryCompiled();
            if (save) cache->Save(directory, disk_key, {native_program->shader, native_program->operations});
            return library;
        }, payload);
    fragments.emplace(key, complete);
    return complete;

}

std::size_t ShaderRuntime::CachedFragmentCount() const {
    std::scoped_lock lock{mutex};
    return fragments.size();
}

std::shared_ptr<const VertexCaptureProgram> ShaderRuntime::Vertex(const Pica::RegsInternal& regs,
                                                                Pica::ShaderSetup& setup, bool reset_machine) {
    std::scoped_lock lock{mutex};
    if (regs.vs.main_offset >= Pica::MAX_PROGRAM_CODE_LENGTH ||
        setup.GetBiggestProgramSize() > Pica::MAX_PROGRAM_CODE_LENGTH ||
        setup.GetBiggestSwizzleSize() > Pica::MAX_SWIZZLE_DATA_LENGTH) {
        throw std::invalid_argument("Invalid PICA vertex cache input bounds");
    }
    setup.DoProgramCodeFixup();
    const auto& program = setup.GetProgramCode();
    const auto& swizzle = setup.GetSwizzleData();
    VertexKeyView view{regs.vs.main_offset, regs.vs.output_mask, regs.rasterizer.vs_output_total,
                       regs.lighting.disable, std::uint32_t(reset_machine), {}, program, swizzle};
    for (std::size_t i = 0; i < view.semantics.size(); ++i) {
        std::memcpy(&view.semantics[i], &regs.rasterizer.vs_output_attributes[i], sizeof(std::uint32_t));
    }
    if (const auto found = vertices.find(view); found != vertices.end()) {
        if(!LibraryFailed(found->second->pending_library))return found->second;
        vertices.erase(found);
    }
    // Only misses need an owned snapshot. The view never outlives this lookup.
    VertexKey key{view.main_offset, view.output_mask, view.output_total, view.lighting_disable,
                  view.reset_machine, view.semantics, program, swizzle};
    std::vector<std::byte> disk_key;
    KeyWord(disk_key, std::uint32_t(ShaderStage::VertexCompute));
    for (const auto value : {key.main_offset, key.output_mask, key.output_total, key.lighting_disable})
        KeyWord(disk_key, value);
    for (const auto value : key.semantics) KeyWord(disk_key, value);
    for (const auto value : key.program) KeyWord(disk_key, value);
    for (const auto value : key.swizzle) KeyWord(disk_key, value);
    if(reset_machine) {KeyWord(disk_key,0x52535431u);KeyWord(disk_key,1);}
    const auto directory = device.ShaderDiskCacheDirectory();
    CheckPrewarmDirectory(directory);
    if (const auto found = prewarmed_vertices.find(ShaderDiskCache::Identity(disk_key));
        found != prewarmed_vertices.end()) {
        disk_cache->PrewarmHit();
        vertices.emplace(key, found->second);
        return found->second;
    }
    auto cached = disk_cache->Load(directory, disk_key, ShaderStage::VertexCompute);
    TranslatedShader shader{};
    bool cached_valid{};
    if (cached) {
        try {
            shader = RestorePicaVertexShader(cached->shader.spirv, ShaderStage::VertexCompute, reset_machine);
            if (shader != cached->shader || cached->operations != FragmentOperations{})
                throw std::invalid_argument("Cached PICA vertex source/reflection differs from its IR");
            cached_valid = true;
        } catch (const std::exception&) { disk_cache->Reject(); }
    }
    if (!cached_valid) {
        TranslationInputScope capture{[&](auto source, auto stage, auto bindings) {
            source_store.Save(directory, source, stage, bindings);
        }};
        shader = GeneratePicaVertexShader(regs, setup, reset_machine);
        disk_cache->Translated();
    }
    auto complete=std::make_shared<VertexCaptureProgram>(VertexCaptureProgram{std::move(shader),nil,nil});
    auto owned=std::make_shared<const TranslatedShader>(complete->shader);
    struct Publication {std::promise<id<MTLComputePipelineState>> pipeline;std::promise<id<MTLLibrary>> library;};
    auto publication=std::make_shared<Publication>();
    complete->pending_pipeline=publication->pipeline.get_future().share();
    complete->pending_library=publication->library.get_future().share();
    const auto bytes=ShaderPayloadBytes(*owned,disk_key.size());
    (void)device.DeferCompilation([native=&device,cache=disk_cache,owned,publication,directory,disk_key=std::move(disk_key),save=!cached_valid] {
        try {
            auto library=native->CompileLibrary(owned->source,"Azahar Metal 4 PICA vertex program");
            ValidateEntry(library,*owned);
            auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
            descriptor.computeFunctionDescriptor=Device::Function(library,owned->entry_point);
            descriptor.maxTotalThreadsPerThreadgroup=64;
            auto pipeline=native->CompileCompute(descriptor);
            cache->LibraryCompiled();if(save)cache->Save(directory,disk_key,{*owned,{}});
            publication->library.set_value(library);publication->pipeline.set_value(pipeline);
        }catch(...) {
            const auto failure=std::current_exception();
            try{publication->library.set_exception(failure);}catch(const std::future_error&){}
            try{publication->pipeline.set_exception(failure);}catch(const std::future_error&){}
        }
    },bytes);
    vertices.emplace(std::move(key),complete);return complete;

}

std::shared_ptr<const CompiledMeshVertexProgram> ShaderRuntime::MeshVertex(const Pica::RegsInternal& regs,
                                                                Pica::ShaderSetup& setup) {
    constexpr bool reset_machine = true;
    std::scoped_lock lock{mutex};
    if (regs.vs.main_offset >= Pica::MAX_PROGRAM_CODE_LENGTH ||
        setup.GetBiggestProgramSize() > Pica::MAX_PROGRAM_CODE_LENGTH ||
        setup.GetBiggestSwizzleSize() > Pica::MAX_SWIZZLE_DATA_LENGTH) {
        throw std::invalid_argument("Invalid PICA vertex cache input bounds");
    }
    setup.DoProgramCodeFixup();
    const auto& program = setup.GetProgramCode();
    const auto& swizzle = setup.GetSwizzleData();
    VertexKeyView view{regs.vs.main_offset, regs.vs.output_mask, regs.rasterizer.vs_output_total,
                       regs.lighting.disable, std::uint32_t(reset_machine), {}, program, swizzle};
    for (std::size_t i = 0; i < view.semantics.size(); ++i) {
        std::memcpy(&view.semantics[i], &regs.rasterizer.vs_output_attributes[i], sizeof(std::uint32_t));
    }
    return MeshVertex(view,[&] { return GeneratePicaVertexMeshShader(regs,setup); });
}

std::shared_ptr<const CompiledMeshVertexProgram> ShaderRuntime::MeshVertex(const Pica::RegsInternal& regs,
                                                               const NativeProgramSnapshot& snapshot) {
    std::scoped_lock lock{mutex};
    const auto& source=snapshot.source;
    if(!source || source->allow_emission || source->entry!=regs.vs.main_offset ||
       source->output_mask!=regs.vs.output_mask)
        throw std::invalid_argument("Native program snapshot does not match draw metadata");
    VertexKeyView view{source->entry,source->output_mask,regs.rasterizer.vs_output_total,
                       regs.lighting.disable,1,{},source->code,source->swizzle};
    for(std::size_t i=0;i<view.semantics.size();++i)
        std::memcpy(&view.semantics[i],&regs.rasterizer.vs_output_attributes[i],sizeof(std::uint32_t));
    OwnedMeshKey key{{view.main_offset,view.output_mask,view.output_total,view.lighting_disable,view.reset_machine},source.get(),source};
    std::copy(view.semantics.begin(),view.semantics.end(),key.metadata.begin()+5);
    const auto hit=[&](const OwnedMeshMap::value_type& entry) {
        if(auto compiled=entry.second.lock(); compiled && !LibraryFailed(compiled->pending_library)) { ++mesh_stats.owned_hits;return compiled; }
        return std::shared_ptr<const CompiledMeshVertexProgram>{};
    };
    const OwnedMeshLess less;
    if(recent_owned_mesh && !less(key,recent_owned_mesh->first) && !less(recent_owned_mesh->first,key))
        if(auto compiled=hit(*recent_owned_mesh))return compiled;
    if(const auto found=owned_mesh.find(key);found!=owned_mesh.end())
        if(auto compiled=hit(*found)) { recent_owned_mesh=std::addressof(*found);return compiled; }
    // First registration still uses the complete existing bank/metadata key.
    // Generation, if needed, reads the sealed copy, never a mutable caller bank.
    auto compiled=MeshVertex(view,[&] {
        Pica::ShaderSetup setup;
        setup.UpdateProgramCode(source->code);setup.UpdateSwizzleData(source->swizzle);
        return GeneratePicaVertexMeshShader(regs,setup);
    });
    if(owned_mesh.size()>=64) { recent_owned_mesh=nullptr;owned_mesh.clear(); }
    auto found=owned_mesh.insert_or_assign(std::move(key),compiled).first;
    recent_owned_mesh=std::addressof(*found);++mesh_stats.owned_registrations;
    return compiled;
}

std::shared_ptr<const CompiledMeshVertexProgram> ShaderRuntime::MeshVertex(const VertexKeyView& view,
                                                    const std::function<TranslatedShader()>& generate) {
    ++mesh_stats.content_lookups;
    if (recent_mesh_vertex && VertexKeyLess::Equal(view, recent_mesh_vertex->first) &&
        !LibraryFailed(recent_mesh_vertex->second->pending_library)) return recent_mesh_vertex->second;
    const auto remember = [this](MeshVertexMap::const_iterator found) {
        recent_mesh_vertex = std::addressof(*found);
        return found->second;
    };
    if (const auto found = mesh_vertices.find(view); found != mesh_vertices.end()) {
        if(!LibraryFailed(found->second->pending_library))return remember(found);
        recent_mesh_vertex=nullptr;mesh_vertices.erase(found);
    }
    // Only misses need an owned snapshot. The view never outlives this lookup.
    VertexKey key{view.main_offset, view.output_mask, view.output_total, view.lighting_disable,
                  view.reset_machine, view.semantics, view.program, view.swizzle};
    std::vector<std::byte> disk_key;
    KeyWord(disk_key, std::uint32_t(ShaderStage::VertexMesh));
    for (const auto value : {key.main_offset, key.output_mask, key.output_total, key.lighting_disable})
        KeyWord(disk_key, value);
    for (const auto value : key.semantics) KeyWord(disk_key, value);
    for (const auto value : key.program) KeyWord(disk_key, value);
    for (const auto value : key.swizzle) KeyWord(disk_key, value);
    if(view.reset_machine) {KeyWord(disk_key,0x52535431u);KeyWord(disk_key,1);}
    const auto directory = device.ShaderDiskCacheDirectory();
    CheckPrewarmDirectory(directory);
    if (const auto found = prewarmed_mesh_vertices.find(ShaderDiskCache::Identity(disk_key));
        found != prewarmed_mesh_vertices.end()) {
        disk_cache->PrewarmHit();
        return remember(mesh_vertices.emplace(key, found->second).first);
    }
    auto cached = disk_cache->Load(directory, disk_key, ShaderStage::VertexMesh);
    TranslatedShader shader{};
    bool cached_valid{};
    if (cached) {
        try {
            shader = RestorePicaVertexShader(cached->shader.spirv, ShaderStage::VertexMesh, true);
            if (shader != cached->shader || cached->operations != FragmentOperations{})
                throw std::invalid_argument("Cached PICA mesh source/reflection differs from its IR");
            cached_valid = true;
        } catch (const std::exception&) { disk_cache->Reject(); }
    }
    if (!cached_valid) {
        TranslationInputScope capture{[&](auto source, auto stage, auto bindings) {
            source_store.Save(directory, source, stage, bindings);
        }};
        shader = generate();
        disk_cache->Translated();
    }
    auto complete=std::make_shared<CompiledMeshVertexProgram>(CompiledMeshVertexProgram{std::move(shader),nil});
    auto owned=std::make_shared<const TranslatedShader>(complete->shader);
    const auto bytes=ShaderPayloadBytes(*owned,disk_key.size());
    complete->pending_library=device.DeferCompilation([native=&device,cache=disk_cache,owned,directory,disk_key=std::move(disk_key),save=!cached_valid] {
        auto library=native->CompileLibrary(owned->source,"Azahar Metal 4 PICA mesh program");
        ValidateEntry(library,*owned);cache->LibraryCompiled();if(save)cache->Save(directory,disk_key,{*owned,{}});
        return library;
    },bytes);
    return remember(mesh_vertices.emplace(std::move(key),complete).first);

}

std::size_t ShaderRuntime::CachedVertexCount() const {
    std::scoped_lock lock{mutex};
    return vertices.size() + mesh_vertices.size();
}

ShaderDiskCacheStats ShaderRuntime::DiskStats() const {
    std::scoped_lock lock{mutex};
    return disk_cache->Stats();
}

MeshLookupStats ShaderRuntime::MeshStats() const {
    std::scoped_lock lock{mutex};
    auto result=mesh_stats;result.owned_entries=owned_mesh.size();
    for(const auto& entry:owned_mesh)result.expired_owners+=entry.first.owner.expired();
    return result;
}

void ShaderRuntime::Clear() {
    std::scoped_lock lock{mutex};
    fragments.clear();
    vertices.clear();
    recent_mesh_vertex = nullptr;
    recent_owned_mesh = nullptr;
    owned_mesh.clear();
    source_store.Clear();
    mesh_vertices.clear();
    prewarmed_fragments.clear(); prewarmed_vertices.clear(); prewarmed_mesh_vertices.clear();
    ++prewarm_epoch;
}

} // namespace Metal
