// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_device.h"
#include <stdexcept>
#include <string>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include "mtl_shader_disk_cache.h"
#include "mtl_shader_cache_version.h"
#include "mtl_pipeline_manifest.h"

namespace Metal {

namespace {

NSString* MakeString(std::string_view text) {
    NSString* result = [[NSString alloc] initWithBytes:text.data()
                                               length:text.size()
                                             encoding:NSUTF8StringEncoding];
    if (!result) {
        throw std::invalid_argument("Metal shader source or name is not valid UTF-8");
    }
    return result;
}

[[noreturn]] void Fail(const char* operation, NSError* error) {
    const char* detail = error ? error.localizedDescription.UTF8String : "no object returned";
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "unknown error"));
}

} // namespace

// Optional bounded CPU-operation records. They contain names and clocks only,
// no shader payloads, GPU timestamp instructions or CPU stack sampling. Device
// operations can run before the Scheduler exists; correlate these clocks with
// its frame begin/end clocks instead of inventing a frame number here.
struct Device::CompilerDiagnostics {
    std::mutex mutex;
    std::ofstream output;
    unsigned remaining{4096}, sequence{};
    std::atomic_bool failed{};
    static std::unique_ptr<CompilerDiagnostics> Create() {
        const auto* enabled=std::getenv("AZAHAR_METAL4_PROFILE_COMPILER");
        if (!enabled || !*enabled || std::string_view{enabled}=="0") return {};
        if (std::string_view{enabled}!="1") throw std::invalid_argument("Invalid Metal compiler profile mode");
        // A compiler-only output must not activate Scheduler timestamps or
        // shader evidence dumps merely to measure CPU compilation costs.
        const auto* path=std::getenv("AZAHAR_METAL4_PROFILE_COMPILER_OUTPUT");
        if (!path || !*path) path=std::getenv("AZAHAR_METAL4_PROFILE");
        if (!path || !*path) throw std::invalid_argument("Metal compiler profile needs an output path");
        auto result=std::make_unique<CompilerDiagnostics>();
        if (const auto* limit=std::getenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT");limit && *limit) {
            const auto end=limit+std::strlen(limit);
            const auto parsed=std::from_chars(limit,end,result->remaining);
            if (parsed.ec!=std::errc{} || parsed.ptr!=end || !result->remaining || result->remaining>4096)
                throw std::invalid_argument("Invalid Metal compiler profile operation limit");
        }
        result->output.open(std::string(path)+".compiler.csv",std::ios::out|std::ios::trunc);
        result->output<<"sequence,operation,name,begin_host_ns,api_begin_host_ns,api_end_host_ns,end_host_ns,host_ms,cache_hit,failed,source_bytes,compiler_lock_wait_ns,compiler_lock_held_ns,publication_lock_wait_ns,persistence_ns,persistence_calls,persistence_fast_hits\n";
        result->output.flush();
        if (!result->output) throw std::runtime_error("Unable to initialize Metal compiler profile");
        return result;
    }
};
struct Device::CompilerOperation {
    CompilerDiagnostics* diagnostics{};
    std::string_view operation;
    std::string name;
    unsigned sequence{};
    std::int64_t begin{}, api_begin{}, api_end{};
    int exceptions{};
    bool cache_hit{};
    std::uint64_t source_bytes{}, compiler_lock_wait_ns{}, compiler_lock_held_ns{};
    std::uint64_t publication_lock_wait_ns{}, persistence_ns{}, persistence_calls{}, persistence_fast_hits{};
    static std::int64_t Now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    CompilerOperation(CompilerDiagnostics* value,std::string_view kind,std::string_view label={}) {
        if (!value) return;
        std::scoped_lock lock{value->mutex};
        if (value->failed) throw std::runtime_error("Unable to write Metal compiler profile");
        if (!value->remaining) return;
        --value->remaining; sequence=++value->sequence; diagnostics=value;
        operation=kind; name=label.substr(0,256); exceptions=std::uncaught_exceptions(); begin=Now();
    }
    CompilerOperation(const CompilerOperation&)=delete;
    CompilerOperation& operator=(const CompilerOperation&)=delete;
    void Name(NSString* label) {
        if (diagnostics && label) if (const auto* text=label.UTF8String) name=std::string_view{text}.substr(0,256);
    }
    void BeginApi() { if (diagnostics) api_begin=Now(); }
    void EndApi() { if (diagnostics) api_end=Now(); }
    // Measure acquisition separately from time inside the critical section.
    // Disabled diagnostics neither read clocks nor change lock ownership/order.
    struct Lock {
        CompilerOperation& profile;
        std::unique_lock<std::mutex> lock;
        std::int64_t acquired{};
        bool publication{};
        Lock(CompilerOperation& value, std::mutex& mutex, bool is_publication=false)
            : profile{value}, lock{mutex,std::defer_lock}, publication{is_publication} {
            const auto begin=profile.diagnostics ? Now():0;
            lock.lock();
            if (profile.diagnostics) {
                acquired=Now();
                (publication ? profile.publication_lock_wait_ns:profile.compiler_lock_wait_ns)+=acquired-begin;
            }
        }
        ~Lock() {
            if (profile.diagnostics && !publication) profile.compiler_lock_held_ns+=Now()-acquired;
        }
    };
    struct Persistence {
        CompilerOperation* profile;
        std::int64_t begin{};
        explicit Persistence(CompilerOperation* value) : profile{value && value->diagnostics ? value:nullptr} {
            if (profile) {begin=Now();++profile->persistence_calls;}
        }
        ~Persistence() {if (profile) profile->persistence_ns+=Now()-begin;}
    };
    ~CompilerOperation() noexcept {
        if (!diagnostics) return;
        const auto end=Now();
        try {
            std::scoped_lock lock{diagnostics->mutex};
            auto& out=diagnostics->output;
            out<<sequence<<','<<operation<<",\"";
            for (const auto c:name) {
                if (c=='\"') out<<"\"\"";
                else out<<(c=='\r' || c=='\n' ? ' ':c);
            }
            out<<"\","<<begin<<','<<api_begin<<','<<api_end<<','<<end<<','<<std::setprecision(12)
                <<double(end-begin)/1000000.0<<','<<cache_hit<<','
                <<(std::uncaught_exceptions()>exceptions)<<','<<source_bytes<<','
                <<compiler_lock_wait_ns<<','<<compiler_lock_held_ns<<','<<publication_lock_wait_ns<<','
                <<persistence_ns<<','<<persistence_calls<<','<<persistence_fast_hits<<'\n';
            out.flush(); diagnostics->failed=!out;
        } catch (...) { diagnostics->failed=true; }
    }
};

Device::Device(std::filesystem::path disk_cache_root,bool disk_cache_enabled) {
    compiler_diagnostics=CompilerDiagnostics::Create();
    if (@available(macOS 26.0, *)) {
        device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            throw std::runtime_error("Metal 4 requires a supported Apple Silicon GPU");
        }
        disk_cache=std::make_unique<PipelineDiskCache>(device,std::move(disk_cache_root),disk_cache_enabled);
        CreateCompiler();
        PublishCacheDirectories();
        requested_cache_enabled.store(disk_cache_enabled);
        flush_worker=std::thread{[this] { FlushWorker(); }};
    } else {
        throw std::runtime_error("Metal 4 requires macOS 26 or later");
    }
}
void Device::CreateCompiler() {
    CompilerOperation profile{compiler_diagnostics.get(),"compiler"};
    auto descriptor=[[MTL4CompilerDescriptor alloc] init];descriptor.label=@"Azahar Metal 4 compiler";
    descriptor.pipelineDataSetSerializer=disk_cache->Serializer();NSError* error=nil;
    auto next=[device newCompilerWithDescriptor:descriptor error:&error];
    if(!next)Fail("Create Metal 4 compiler",error);compiler=next;
}
id<MTL4Compiler> Device::UncapturedCompiler() {
    auto descriptor=[[MTL4CompilerDescriptor alloc] init];descriptor.label=@"Azahar validated Metal4 compiler";
    NSError* error=nil;auto next=[device newCompilerWithDescriptor:descriptor error:&error];
    if(!next)Fail("Create uncaptured Metal4 compiler",error);return next;
}
Device::~Device() {
    // Factories capture this device, and their native serializers must be idle
    // before final persistence or destruction of compiler/library metadata.
    compilation_queue.Drain();
    {
        std::scoped_lock lock{flush_request_mutex};
        flush_stopping=true;
    }
    flush_request_cv.notify_one();
    if(flush_worker.joinable())flush_worker.join();
    try {FlushDiskCache();}catch(...) {}
}
void Device::PublishCacheDirectories() {
    std::scoped_lock lock{cache_directory_mutex};
    cache_directories={disk_cache->ShaderDirectory(),disk_cache->RenderDirectory()};
}
void Device::RequestDiskCacheFlush() {
    {
        std::scoped_lock lock{flush_request_mutex};
        if(flush_stopping)return;
        flush_requested=true;
    }
    flush_request_cv.notify_one();
}
void Device::FlushWorker() {
    for(;;) {
        {
            std::unique_lock lock{flush_request_mutex};
            flush_request_cv.wait(lock,[this] {return flush_requested||flush_stopping;});
            if(flush_stopping)return;
            flush_requested=false;
        }
        @autoreleasepool {
            try {
                CompilerOperation profile{compiler_diagnostics.get(),"flush_disk_async"};
                FlushDiskCache();
            } catch(...) {
                // Persistence is optional; the next request or final flush can retry.
            }
        }
    }
}
void Device::SwitchDiskCache(std::uint64_t title_id,const std::atomic_bool* stop) {
    if (stop && stop->load()) return;
    // A queued factory must finish in its title's serializer namespace before
    // the owner rotates archives. Warm unchanged settings do not take this path.
    compilation_queue.Drain();
    std::scoped_lock publication{publication_mutex};
    std::unique_lock lifecycle{compiler_lifecycle};
    std::scoped_lock lock{compiler_mutex};
    if(stop && stop->load())return;
    disk_cache->Switch(title_id,stop);CreateCompiler();
    PublishCacheDirectories();
}
bool Device::FlushDiskCache() {
    CompilerOperation profile{compiler_diagnostics.get(),"flush_disk"};
    CompilerOperation::Lock publication{profile,publication_mutex,true};
    const auto publish = [&] {
        const auto result=[&] {
            CompilerOperation::Persistence io{&profile};
            return PipelineDiskCache::Publish(*publication_retry);
        }();
        { CompilerOperation::Lock lock{profile,compiler_mutex}; disk_cache->Published(result); }
        if (result.saved) publication_retry.reset();
        return result.saved;
    };
    if (publication_retry && !publish()) return false;
    {
        std::unique_lock lifecycle{compiler_lifecycle};
        CompilerOperation::Lock lock{profile,compiler_mutex};
        publication_retry = disk_cache->DetachFlush();
        if (publication_retry && publication_retry->serializer) {
            try { CreateCompiler(); }
            catch (...) {
                disk_cache->RestoreDetached(*publication_retry);
                publication_retry.reset();
                throw;
            }
        }
    }
    return !publication_retry || publish();
}
void Device::SetDiskCacheEnabled(bool enabled) {
    // TickFrame calls this even when settings did not change. A warm frame
    // must not wait for the cache writer's compiler/serializer exclusion lock.
    if(requested_cache_enabled.load()==enabled)return;
    compilation_queue.Drain();
    std::scoped_lock publication{publication_mutex};
    std::unique_lock lifecycle{compiler_lifecycle};
    std::scoped_lock lock{compiler_mutex};const auto before=disk_cache->Stats().enabled;
    disk_cache->SetEnabled(enabled);if(before!=disk_cache->Stats().enabled)CreateCompiler();
    PublishCacheDirectories();
    requested_cache_enabled.store(enabled);
}
PipelineDiskCacheStats Device::DiskCacheStats() const {std::scoped_lock lock{compiler_mutex};return disk_cache->Stats();}
std::filesystem::path Device::ShaderDiskCacheDirectory() const {
    std::scoped_lock lock{cache_directory_mutex};
    return cache_directories.shader;
}
std::filesystem::path Device::RenderDiskCacheDirectory() const {
    std::scoped_lock lock{cache_directory_mutex};
    return cache_directories.render;
}
std::optional<std::string> Device::LibraryIdentity(id<MTLLibrary> library) const {
    std::scoped_lock lock{compiler_mutex};
    const auto id = reinterpret_cast<std::uintptr_t>((__bridge void*)library);
    const auto found = library_identities.find(id);
    if (found == library_identities.end() || found->second.library != library) return std::nullopt;
    return found->second.identity;
}
bool Device::SaveLibrarySource(id<MTLLibrary> library, const std::filesystem::path& directory) {
    CompilerOperation profile{compiler_diagnostics.get(),"persist_library"};
    CompilerOperation::Lock lock{profile,compiler_mutex};
    if (directory.empty() || directory != disk_cache->RenderDirectory()) return false;
    const auto id = reinterpret_cast<std::uintptr_t>((__bridge void*)library);
    const auto found = library_identities.find(id);
    if (found == library_identities.end() || found->second.library != library || !found->second.source)
        return false;
    profile.source_bytes=found->second.source->text.size();
    return PersistLibrary(found->second, directory, &profile);
}

bool Device::PersistLibrary(LibraryRecord& record, const std::filesystem::path& directory,
                            CompilerOperation* profile) {
    if (directory.empty() || !record.source) return false;
    CompilerOperation::Persistence io{profile};
    const auto path = directory / "libraries" / ShaderDiskCacheSourceVersion / (record.identity + ".library");
    const auto stamp = [&]() -> std::array<std::uint64_t,8> {
        struct stat info{};
        if (lstat(path.c_str(), &info) || !S_ISREG(info.st_mode)) return {};
        return {std::uint64_t(info.st_dev), std::uint64_t(info.st_ino), std::uint64_t(info.st_size),
                std::uint64_t(info.st_mtimespec.tv_sec), std::uint64_t(info.st_mtimespec.tv_nsec),
                std::uint64_t(info.st_ctimespec.tv_sec), std::uint64_t(info.st_ctimespec.tv_nsec), 1};
    };
    // Full bytes/name equality was checked on live reuse. Only a changed disk
    // file needs checksum/source verification again; missing/corrupt/replaced
    // files still go through the original validating atomic publisher.
    const auto current = stamp();
    if (current.back() && record.persisted_directory == directory && record.persisted_stamp == current) {
        if (io.profile) ++io.profile->persistence_fast_hits;
        return true;
    }
    if (!RenderPipelineManifest::SaveLibrary(directory, record.identity, record.source->name, record.source->text))
        return false;
    record.persisted_directory = directory;
    record.persisted_stamp = stamp();
    return record.persisted_stamp.back();
}
id<MTLLibrary> Device::LibraryWithIdentity(std::string_view identity) {
    std::filesystem::path directory;
    {
        std::scoped_lock lock{compiler_mutex};
        for (const auto& [id, record] : library_identities) {
            (void)id;
            if (record.identity == identity) if (auto library = record.library) return library;
        }
        directory = disk_cache->RenderDirectory();
    }
    const auto saved = RenderPipelineManifest::LoadLibrary(directory, identity);
    if (!saved) return nil;
    return CompileLibrary(saved->second, saved->first);
}

id<MTLBuffer> Device::CreateBuffer(NSUInteger size, bool cpu_visible) const {
    if (!size || size > device.maxBufferLength) {
        throw std::invalid_argument("Metal buffer size exceeds the device limits");
    }
    const auto storage = cpu_visible ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                              options:storage | MTLResourceHazardTrackingModeUntracked];
    if (!buffer) {
        Fail("Allocate Metal buffer", nil);
    }
    return buffer;
}

id<MTLTexture> Device::CreateTexture(MTLTextureDescriptor* descriptor) const {
    if (!descriptor || !descriptor.width || !descriptor.height || !descriptor.depth) {
        throw std::invalid_argument("Invalid Metal texture descriptor");
    }
    MTLTextureDescriptor* copy = [descriptor copy];
    copy.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    id<MTLTexture> texture = [device newTextureWithDescriptor:copy];
    if (!texture) {
        Fail("Allocate Metal texture", nil);
    }
    return texture;
}

id<MTLTexture> Device::CreateBufferTexture(id<MTLBuffer> buffer, NSUInteger offset,
                                         NSUInteger count, MTLPixelFormat format) const {
    const NSUInteger bytes = format == MTLPixelFormatRG32Float ? 8 :
                             format == MTLPixelFormatRGBA32Float ? 16 : 0;
    if (!bytes || !buffer || buffer.device != device || !count || offset > buffer.length ||
        count > (buffer.length - offset) / bytes ||
        offset % [device minimumTextureBufferAlignmentForPixelFormat:format]) {
        throw std::invalid_argument("Invalid native PICA LUT buffer view");
    }
    auto* descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureTypeTextureBuffer;
    descriptor.pixelFormat = format;
    descriptor.width = count;
    descriptor.storageMode = buffer.storageMode;
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    const auto alignment = [device minimumTextureBufferAlignmentForPixelFormat:format];
    const auto row_bytes = count * bytes;
    const auto padding = (alignment - row_bytes % alignment) % alignment;
    if (padding > buffer.length - offset - row_bytes)
        throw std::invalid_argument("Native PICA LUT buffer is missing aligned row padding");
    const auto texture = [buffer newTextureWithDescriptor:descriptor offset:offset bytesPerRow:row_bytes + padding];
    if (!texture) Fail("Allocate native PICA LUT view", nil);
    return texture;
}

id<MTLSamplerState> Device::CreateSampler(MTLSamplerDescriptor* descriptor) const {
    if (!descriptor) {
        throw std::invalid_argument("Missing Metal sampler descriptor");
    }
    MTLSamplerDescriptor* copy = [descriptor copy];
    copy.supportArgumentBuffers = YES;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:copy];
    if (!sampler) {
        Fail("Create Metal sampler", nil);
    }
    return sampler;
}

id<MTLLibrary> Device::CompileLibrary(std::string_view source, std::string_view name) {
    CompilerOperation profile{compiler_diagnostics.get(),"library",name};
    profile.source_bytes=source.size();
    std::shared_lock lifecycle{compiler_lifecycle};
    std::pair<std::string,std::string> key;
    std::shared_future<id<MTLLibrary>> pending;
    std::shared_ptr<std::promise<id<MTLLibrary>>> producer;
    {
        CompilerOperation::Lock lock{profile,compiler_mutex};
        std::erase_if(library_identities, [](const auto& item) { return !item.second.library; });
        for (auto& [address, record] : library_identities) {
            (void)address;
            if (!record.source || record.source->name != name || record.source->text != source) continue;
            id<MTLLibrary> live = record.library;
            if (live) {
                PersistLibrary(record, disk_cache->RenderDirectory(), &profile);
                profile.cache_hit = true;
                return live;
            }
        }
        key={std::string{name},std::string{source}};
        if (const auto found=pending_libraries.find(key);found!=pending_libraries.end()) pending=found->second;
        else {
            producer=std::make_shared<std::promise<id<MTLLibrary>>>();
            pending=producer->get_future().share();
            pending_libraries.emplace(key,pending);
        }
    }
    if (!producer) {
        auto library=pending.get();
        profile.cache_hit=true;
        return library;
    }
    try {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        MTL4LibraryDescriptor* descriptor = [[MTL4LibraryDescriptor alloc] init];
        descriptor.source = MakeString(source);
        descriptor.name = MakeString(name);
        descriptor.options = options;
        NSError* error = nil;
        profile.BeginApi();
        id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
        profile.EndApi();
        if (!library) {
            Fail("Compile Metal 4 library", error);
        }
        const auto identity = SourceIdentity(name, source);
        const auto id = reinterpret_cast<std::uintptr_t>((__bridge void*)library);
        {
            CompilerOperation::Lock lock{profile,compiler_mutex};
            // Remove expired metadata before retaining source text. Identical live
            // libraries share one bounded source record even across title/cache toggles.
            std::erase_if(library_identities, [](const auto& item) { return !item.second.library; });
            std::shared_ptr<const LibraryRecord::Source> retained_source;
            for (const auto& [address, record] : library_identities) {
                (void)address;
                if (record.identity == identity) { retained_source = record.source; break; }
            }
            if (!retained_source && name.size() <= 8 * 1024 * 1024 && source.size() <= 8 * 1024 * 1024)
                retained_source = std::make_shared<LibraryRecord::Source>(LibraryRecord::Source{std::string{name}, std::string{source}});
            auto [record, inserted] = library_identities.insert_or_assign(id,
                LibraryRecord{library, identity, std::move(retained_source), {}, {}});
            (void)inserted;
            PersistLibrary(record->second, disk_cache->RenderDirectory(), &profile);
            pending_libraries.erase(key);
        }
        producer->set_value(library);
        return library;
    } catch (...) {
        const auto failure=std::current_exception();
        // Existing waiters keep their original exception; a later request may
        // retry the full source. Never wait on a future under the metadata lock.
        try { producer->set_exception(failure); } catch (...) {}
        { CompilerOperation::Lock lock{profile,compiler_mutex}; pending_libraries.erase(key); }
        std::rethrow_exception(failure);
    }
}

std::string Device::SourceIdentity(std::string_view name, std::string_view source) {
    std::vector<std::byte> key;
    const auto append = [&](std::string_view value) {
        const auto size = std::uint64_t(value.size());
        for (unsigned i = 0; i < 8; ++i) key.push_back(std::byte(size >> (8 * i)));
        const auto bytes = std::as_bytes(std::span(value));
        key.insert(key.end(), bytes.begin(), bytes.end());
    };
    append(ShaderDiskCacheSourceVersion); append(name); append(source);
    return ShaderDiskCache::Identity(key);
}

id<MTLComputePipelineState> Device::CompileCompute(MTL4ComputePipelineDescriptor* descriptor) {
    if (!descriptor || !descriptor.computeFunctionDescriptor) {
        throw std::invalid_argument("Missing Metal 4 compute function");
    }
    // Preserve the caller's reusable descriptor while giving diagnostics a
    // stable function name, including when a disk-cached PSO is returned.
    descriptor = [descriptor copy];
    if (!descriptor.label && [descriptor.computeFunctionDescriptor isKindOfClass:[MTL4LibraryFunctionDescriptor class]])
        descriptor.label = ((MTL4LibraryFunctionDescriptor*)descriptor.computeFunctionDescriptor).name;
    CompilerOperation profile{compiler_diagnostics.get(),"compute"}; profile.Name(descriptor.label);
    std::shared_lock lifecycle{compiler_lifecycle};
    NSError* error = nil;
    const bool validated=descriptor.options.shaderValidation==MTLShaderValidationEnabled;
    id<MTL4Compiler> active;
    MTL4CompilerTaskOptions* options;
    {
        CompilerOperation::Lock lock{profile,compiler_mutex};
        if(!validated)if(auto cached=disk_cache->Find(descriptor)){profile.cache_hit=true;return cached;}
        active=validated && disk_cache->Serializer() ? UncapturedCompiler() : compiler;
        options=validated ? nil:disk_cache->TaskOptions();
    }
    profile.BeginApi();
    id<MTLComputePipelineState> pipeline =
        [active newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:options error:&error];
    profile.EndApi();
    if (!pipeline) {
        Fail("Compile Metal 4 compute pipeline", error);
    }
    { CompilerOperation::Lock lock{profile,compiler_mutex}; disk_cache->Compiled(!validated); }
    return pipeline;
}

id<MTLRenderPipelineState> Device::CompileMesh(MTL4MeshRenderPipelineDescriptor* descriptor) {
    if (!descriptor || !descriptor.meshFunctionDescriptor)
        throw std::invalid_argument("Missing Metal 4 mesh function");
    CompilerOperation profile{compiler_diagnostics.get(),"mesh"}; profile.Name(descriptor.label);
    std::shared_lock lifecycle{compiler_lifecycle};
    NSError* error = nil;
    const bool validated=descriptor.options.shaderValidation==MTLShaderValidationEnabled;
    id<MTL4Compiler> active;
    MTL4CompilerTaskOptions* options;
    {
        CompilerOperation::Lock lock{profile,compiler_mutex};
        if(!validated)if(auto cached=disk_cache->Find(descriptor)){profile.cache_hit=true;return cached;}
        active=validated && disk_cache->Serializer() ? UncapturedCompiler() : compiler;
        options=validated ? nil:disk_cache->TaskOptions();
    }
    profile.BeginApi();
    auto pipeline = [active newRenderPipelineStateWithDescriptor:descriptor
                                                  compilerTaskOptions:options error:&error];
    profile.EndApi();
    if (!pipeline) Fail("Compile Metal 4 mesh pipeline", error);
    { CompilerOperation::Lock lock{profile,compiler_mutex}; disk_cache->Compiled(!validated); }
    return pipeline;
}

id<MTLRenderPipelineState> Device::CompileRender(MTL4RenderPipelineDescriptor* descriptor) {
    if (!descriptor || !descriptor.vertexFunctionDescriptor) {
        throw std::invalid_argument("Missing Metal 4 vertex function");
    }
    CompilerOperation profile{compiler_diagnostics.get(),"render"}; profile.Name(descriptor.label);
    std::shared_lock lifecycle{compiler_lifecycle};
    NSError* error = nil;
    const bool validated=descriptor.options.shaderValidation==MTLShaderValidationEnabled;
    id<MTL4Compiler> active;
    MTL4CompilerTaskOptions* options;
    {
        CompilerOperation::Lock lock{profile,compiler_mutex};
        if(!validated)if(auto cached=disk_cache->Find(descriptor)){profile.cache_hit=true;return cached;}
        active=validated && disk_cache->Serializer() ? UncapturedCompiler() : compiler;
        options=validated ? nil:disk_cache->TaskOptions();
    }
    profile.BeginApi();
    id<MTLRenderPipelineState> pipeline =
        [active newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:options error:&error];
    profile.EndApi();
    if (!pipeline) {
        Fail("Compile Metal 4 render pipeline", error);
    }
    { CompilerOperation::Lock lock{profile,compiler_mutex}; disk_cache->Compiled(!validated); }
    return pipeline;
}

MTL4LibraryFunctionDescriptor* Device::Function(id<MTLLibrary> library, std::string_view name) {
    if (!library || name.empty()) {
        throw std::invalid_argument("Missing Metal library or function name");
    }
    MTL4LibraryFunctionDescriptor* descriptor = [[MTL4LibraryFunctionDescriptor alloc] init];
    descriptor.library = library;
    descriptor.name = MakeString(name);
    return descriptor;
}

} // namespace Metal
