// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_pipeline_disk_cache.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstdlib>
#include <stdexcept>

namespace Metal {
namespace {
constexpr std::uintmax_t MaxArchiveBytes=256ULL*1024*1024;
constexpr std::size_t MaxLoadedBatches=64;
constexpr std::uintmax_t MaxStoredArchiveBytes=512ULL*1024*1024;
constexpr std::size_t MaxStoredBatches=128;
bool Cancelled(PipelineDiskCacheStats& stats,const std::atomic_bool* stop) {
    if(!stop || !stop->load())return false;
    ++stats.cancelled_scans;return true;
}
bool ScanEntry(PipelineDiskCacheStats& stats,const std::atomic_bool* stop,std::size_t& remaining) {
    if(Cancelled(stats,stop))return false;
    if(!remaining){++stats.scan_limit_hits;return false;}
    --remaining;++stats.scanned_entries;return true;
}
bool TitleName(std::string_view name) {
    return name.size()==16 && name.find_first_not_of("0123456789abcdef")==std::string_view::npos;
}
bool BatchName(std::string_view name) {
    if(!name.starts_with("batch-") || name.size()!=42)return false;
    name.remove_prefix(6);
    for(std::size_t i=0;i<name.size();++i) {
        if(i==8 || i==13 || i==18 || i==23) {if(name[i]!='-')return false;}
        else if(std::string_view("0123456789abcdefABCDEF").find(name[i])==std::string_view::npos)return false;
    }
    return true;
}
NSURL* URL(const std::filesystem::path& path) {
    auto text=[[NSString alloc] initWithBytes:path.native().data() length:path.native().size() encoding:NSUTF8StringEncoding];
    return text?[NSURL fileURLWithPath:text]:nil;
}
std::string Hex(const unsigned char* bytes,std::size_t size) {
    constexpr char digits[]="0123456789abcdef";
    std::string result(size*2,'0');
    for(std::size_t i=0;i<size;++i) {result[i*2]=digits[bytes[i]>>4];result[i*2+1]=digits[bytes[i]&15];}
    return result;
}
std::string Digest(const std::filesystem::path& path,std::uintmax_t size,const std::atomic_bool* stop=nullptr,
                   std::uint64_t* hashed_bytes=nullptr) {
    if(!size || size>MaxArchiveBytes)return {};
    std::ifstream input(path,std::ios::binary);
    if(!input)return {};
    CC_SHA256_CTX context;CC_SHA256_Init(&context);
    std::array<char,64*1024> buffer{};
    // Hash exactly the validated size: a concurrently growing file must not
    // turn this optional cache lookup into an unbounded read.
    while(size) {
        if(stop && stop->load())return {};
        const auto count=std::min<std::uintmax_t>(size,buffer.size());
        input.read(buffer.data(),std::streamsize(count));
        if(hashed_bytes && input.gcount()>0)*hashed_bytes+=std::uint64_t(input.gcount());
        if(input.gcount()!=std::streamsize(count))return {};
        CC_SHA256_Update(&context,buffer.data(),CC_LONG(count));size-=count;
    }
    if(input.peek()!=std::char_traits<char>::eof() || input.bad())return {};
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};CC_SHA256_Final(digest.data(),&context);
    return Hex(digest.data(),digest.size());
}
std::string Environment(id<MTLDevice> device) {
    const auto version=NSProcessInfo.processInfo.operatingSystemVersionString;
    const std::string identity=std::string("azahar-mtl4-v1-arm64-msl4-safe-")+std::to_string(device.registryID)+
        "-"+(version.UTF8String?version.UTF8String:"");
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(identity.data(),CC_LONG(identity.size()),digest.data());
    return Hex(digest.data(),digest.size());
}
std::string Title(std::uint64_t id) {std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<id;return out.str();}
}

PipelineDiskCache::PipelineDiskCache(id<MTLDevice> device_,std::filesystem::path root_,bool enabled):device{device_} {
    if(root_.empty() || !root_.is_absolute())return;
    const char* validation=std::getenv("MTL_SHADER_VALIDATION");
    if(validation && std::string_view(validation)!="0") {
        // Instrumented binaries exported by this API can carry a deployment
        // target rejected on reload. Keep validation active and compile natively.
        stats.last_error="Binary persistence disabled while Metal Shader Validation instrumentation is enabled";
        instrumented=true;
        return;
    }
    root=root_/Environment(device);if(enabled)SetEnabled(true);
}
void PipelineDiskCache::SetEnabled(bool enabled) {
    enabled=enabled&&!root.empty()&&!instrumented;
    if(enabled==stats.enabled)return;
    if(!enabled){stats.enabled=false;serializer=nil;archives.clear();pending=0;
        if (synchronous_retry && !synchronous_retry->temporary.empty()) {
            std::error_code error; std::filesystem::remove_all(synchronous_retry->temporary,error);
        }
        synchronous_retry.reset();
        stats.resident_batches=stats.resident_archive_bytes=0;return;}
    stats.enabled=true;Switch(stats.title_id);
}
void PipelineDiskCache::Load(const std::filesystem::path& directory,const std::atomic_bool* stop,std::size_t& remaining,
                             std::uintmax_t& hash_remaining) {
    if(!remaining || Cancelled(stats,stop))return;
    std::error_code error;
    struct Batch {
        std::filesystem::path path;
        std::filesystem::file_time_type modified;
    };
    std::vector<Batch> batches;
    for(std::filesystem::directory_iterator it(directory,error),end;!error && it!=end;it.increment(error)) {
        if(!ScanEntry(stats,stop,remaining)) {
            if(stop && stop->load())return;
            break;
        }
        const auto name=it->path().filename().string();
        const auto status=it->symlink_status(error);
        if(name.starts_with("batch-") && !error && std::filesystem::is_directory(status)) {
            std::error_code timestamp_error;
            auto modified=it->last_write_time(timestamp_error);
            if(timestamp_error)modified=std::filesystem::file_time_type::min();
            batches.push_back({it->path(),modified});
        }
    }
    std::sort(batches.begin(),batches.end(),[](const auto& a,const auto& b){
        return a.modified==b.modified ? a.path>b.path : a.modified>b.modified;
    });
    for(const auto& entry:batches) {
        if(Cancelled(stats,stop))return;
        if(archives.size()>=MaxLoadedBatches || stats.resident_archive_bytes>=MaxArchiveBytes)return;
        const auto& batch=entry.path;
        const auto file=batch/"pipelines.metalar",checksum=batch/"sha256";
        const auto status=std::filesystem::symlink_status(file,error);
        if(error || !std::filesystem::is_regular_file(status)){++stats.rejected_batches;error.clear();continue;}
        const auto size=std::filesystem::file_size(file,error);
        if(error || !size || size>MaxArchiveBytes || size>hash_remaining || size>MaxArchiveBytes-stats.resident_archive_bytes ||
           archives.size()>=MaxLoadedBatches) {
            ++stats.rejected_batches;error.clear();continue;
        }
        const auto checksum_status=std::filesystem::symlink_status(checksum,error);
        if(error || !std::filesystem::is_regular_file(checksum_status)) {
            ++stats.rejected_batches;error.clear();continue;
        }
        const auto checksum_size=std::filesystem::file_size(checksum,error);
        if(error || checksum_size!=64){++stats.rejected_batches;error.clear();continue;}
        std::ifstream check(checksum,std::ios::binary);std::string expected(64,'\0');check.read(expected.data(),64);
        if(check)hash_remaining-=size;
        const auto digest=check?Digest(file,size,stop,&stats.hashed_archive_bytes):std::string{};
        if(Cancelled(stats,stop))return;
        if(digest!=expected){++stats.rejected_batches;continue;}
        NSError* failure=nil;auto url=URL(file);
        auto archive=url?[device newArchiveWithURL:url error:&failure]:nil;
        if(!archive){++stats.rejected_batches;stats.last_error=failure.localizedDescription.UTF8String?
            failure.localizedDescription.UTF8String:"Native archive loader returned nil";continue;}
        archives.push_back(archive);stats.resident_archive_bytes+=size;
        stats.resident_batches=archives.size();++stats.loaded_batches;
    }
}
namespace {
void PruneBatches(const std::filesystem::path& root, PipelineDiskCacheStats& stats, const std::atomic_bool* stop) {
    struct Batch {std::filesystem::path path;std::filesystem::file_time_type modified;std::uintmax_t bytes;};
    std::vector<Batch> batches;std::error_code error;
    std::size_t remaining=PipelineDiskCache::ScanEntryLimit;
    if(Cancelled(stats,stop))return;
    // Only our exact title/UUID namespace and two-file batch layout are owned.
    // Unknown files, interrupted publications and symlinks are never removed.
    for(std::filesystem::directory_iterator titles(root,error),end;!error && titles!=end;titles.increment(error)) {
        if(!ScanEntry(stats,stop,remaining))return;
        if(!TitleName(titles->path().filename().string()))continue;
        const auto status=titles->symlink_status(error);
        if(error || !std::filesystem::is_directory(status)){error.clear();continue;}
        std::error_code batch_error;
        for(std::filesystem::directory_iterator it(titles->path(),batch_error),batch_end;
            !batch_error && it!=batch_end;it.increment(batch_error)) {
            if(!ScanEntry(stats,stop,remaining))return;
            if(!BatchName(it->path().filename().string()))continue;
            const auto batch_status=it->symlink_status(batch_error);
            if(batch_error || !std::filesystem::is_directory(batch_status)){batch_error.clear();continue;}
            unsigned count{};bool managed=true;std::uintmax_t bytes{};
            std::error_code file_error;
            for(std::filesystem::directory_iterator files(it->path(),file_error),file_end;
                !file_error && files!=file_end;files.increment(file_error)) {
                if(!ScanEntry(stats,stop,remaining))return;
                ++count;const auto name=files->path().filename().string();
                const auto file_status=files->symlink_status(file_error);
                if(file_error || !std::filesystem::is_regular_file(file_status) ||
                   (name!="pipelines.metalar" && name!="sha256")){managed=false;break;}
                const auto size=files->file_size(file_error);
                if(file_error || (name=="sha256" && size!=64)){managed=false;break;}
                if(name=="pipelines.metalar")bytes=size;
            }
            if(file_error || !managed || count!=2 || !bytes)continue;
            auto modified=it->last_write_time(file_error);
            if(file_error)modified=std::filesystem::file_time_type::min();
            batches.push_back({it->path(),modified,bytes});
        }
        if(batch_error)return;
    }
    // Never derive eviction decisions from a partial environment inventory.
    if(error || Cancelled(stats,stop))return;
    std::sort(batches.begin(),batches.end(),[](const auto& a,const auto& b){
        return a.modified==b.modified ? a.path>b.path : a.modified>b.modified;
    });
    std::uintmax_t retained_bytes{};std::size_t retained{};
    for(const auto& batch:batches) {
        if(Cancelled(stats,stop))return;
        if(retained<MaxStoredBatches && batch.bytes<=MaxStoredArchiveBytes-retained_bytes) {
            retained_bytes+=batch.bytes;++retained;continue;
        }
        // Avoid recursive deletion: new/unknown content survives concurrent writers.
        std::error_code failure;
        const bool removed=std::filesystem::remove(batch.path/"pipelines.metalar",failure);
        if(!failure && removed) {
            ++stats.pruned_batches;stats.pruned_archive_bytes+=batch.bytes;
            std::filesystem::remove(batch.path/"sha256",failure);
            if(!failure)std::filesystem::remove(batch.path,failure);
        }
        if(failure){++stats.io_failures;stats.last_error=failure.message();}
    }
}
} // namespace
void PipelineDiskCache::Prune(const std::atomic_bool* stop) { PruneBatches(root, stats, stop); }
void PipelineDiskCache::Switch(std::uint64_t title_id,const std::atomic_bool* stop) {
    if(!stats.enabled || (stop && stop->load()))return;
    if (!Flush(stop)) {
        if (stop && stop->load()) return;
        SetEnabled(false);
        return;
    }
    if(Cancelled(stats,stop))return;
    archives.clear();stats.resident_batches=stats.resident_archive_bytes=0;
    stats.title_id=title_id;current=root/Title(title_id);
    std::error_code error;std::filesystem::create_directories(current,error);
    if(error){++stats.io_failures;stats.last_error=error.message();stats.enabled=false;serializer=nil;return;}
    auto* descriptor=[[MTL4PipelineDataSetSerializerDescriptor alloc] init];
    descriptor.configuration=MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    serializer=[device newPipelineDataSetSerializerWithDescriptor:descriptor];
    if(!serializer){++stats.io_failures;stats.enabled=false;return;}
    Prune(stop);
    // A large common archive must not starve the active title's own pipelines.
    std::size_t remaining=ScanEntryLimit;
    std::uintmax_t hash_remaining=MaxArchiveBytes;
    Load(current,stop,remaining,hash_remaining);
    if(title_id)Load(root/Title(0),stop,remaining,hash_remaining);
}
MTL4CompilerTaskOptions* PipelineDiskCache::TaskOptions() const {
    if(archives.empty())return nil;
    auto options=[[MTL4CompilerTaskOptions alloc] init];auto array=[NSMutableArray arrayWithCapacity:archives.size()];
    for(auto archive:archives)[array addObject:archive];options.lookupArchives=array;return options;
}
id<MTLComputePipelineState> PipelineDiskCache::Find(MTL4ComputePipelineDescriptor* descriptor) {
    for(auto archive:archives) {NSError* error=nil;auto pipeline=[archive newComputePipelineStateWithDescriptor:descriptor error:&error];
        if(pipeline){++stats.archive_hits;return pipeline;}}
    return nil;
}
id<MTLRenderPipelineState> PipelineDiskCache::Find(MTL4RenderPipelineDescriptor* descriptor) {
    for(auto archive:archives) {NSError* error=nil;auto pipeline=[archive newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if(pipeline){++stats.archive_hits;return pipeline;}}
    return nil;
}
id<MTLRenderPipelineState> PipelineDiskCache::Find(MTL4MeshRenderPipelineDescriptor* descriptor) {
    // MTL4Archive accepts the native mesh descriptor through MTL4PipelineDescriptor.
    // Keep the concrete type; no vertex-descriptor reinterpretation is involved.
    for(auto archive:archives) {NSError* error=nil;auto pipeline=[archive newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if(pipeline){++stats.archive_hits;return pipeline;}}
    return nil;
}
PipelineArchivePublication PipelineDiskCache::Publish(PipelineArchiveBatch& batch, const std::atomic_bool* stop) {
    PipelineArchivePublication result{};
    auto& stats = result.stats;
    if(Cancelled(stats,stop))return result;
    const auto identifier=NSUUID.UUID.UUIDString;
    const std::string id=batch.temporary.empty() ?
        (identifier.UTF8String?identifier.UTF8String:"") : batch.temporary.filename().string().substr(9);
    if(id.empty()){++stats.io_failures;return result;}
    const auto temporary=batch.directory/(".pending-"+id),final=batch.directory/("batch-"+id);
    std::error_code error;std::filesystem::create_directory(temporary,error);
    if(error){++stats.io_failures;stats.last_error=error.message();return result;}
    NSError* failure=nil;
    batch.temporary=temporary;
    if (!batch.serialized) {
        {std::ofstream output(temporary/"pipelines.metalar",std::ios::binary);}
        batch.serialized=[batch.serializer serializeAsArchiveAndFlushToURL:URL(temporary/"pipelines.metalar") error:&failure];
    }
    const bool saved=batch.serialized;
    bool valid=saved;
    if(saved) {
        const auto size=std::filesystem::file_size(temporary/"pipelines.metalar",error);
        const auto checksum=!error && size && size<=MaxArchiveBytes ? Digest(temporary/"pipelines.metalar",size) : std::string{};
        std::ofstream output(temporary/"sha256",std::ios::binary);output.write(checksum.data(),checksum.size());output.close();
        valid=checksum.size()==64 && bool(output);
    }
    if(valid){std::filesystem::rename(temporary,final,error);valid=!error;}
    if(!valid){
        const auto details=failure.localizedDescription.UTF8String?failure.localizedDescription.UTF8String:"no NSError";
        stats.last_error="serialize="+std::to_string(saved)+", pending="+std::to_string(batch.pipelines)+", "+details;
        if(error)stats.last_error+="; "+error.message();
        // A successful native serialize flushes its data set. Retain that one
        // unpublished archive for checksum/rename retry instead of discarding it.
        if (!batch.serialized) {
            std::filesystem::remove_all(temporary,error);
            batch.temporary.clear();
        }
        ++stats.io_failures;return result;
    }
    batch.temporary.clear();
    ++stats.saved_batches; PruneBatches(batch.root, stats, stop); result.saved=true; return result;
}

void PipelineDiskCache::Published(const PipelineArchivePublication& publication) {
    const auto& report = publication.stats;
    stats.saved_batches += report.saved_batches;
    stats.io_failures += report.io_failures;
    stats.pruned_batches += report.pruned_batches;
    stats.pruned_archive_bytes += report.pruned_archive_bytes;
    stats.scanned_entries += report.scanned_entries;
    stats.scan_limit_hits += report.scan_limit_hits;
    stats.cancelled_scans += report.cancelled_scans;
    if (!report.last_error.empty()) stats.last_error = report.last_error;
}
std::optional<PipelineArchiveBatch> PipelineDiskCache::DetachFlush() {
    if (!stats.enabled || !serializer || !pending) return std::nullopt;
    if (synchronous_retry && synchronous_retry->serialized) {
        auto batch=std::move(*synchronous_retry);
        synchronous_retry.reset();
        pending-=batch.pipelines;
        // No native serializer operation is needed for this publication retry.
        // The current serializer already contains only later compiler captures.
        batch.serializer=nil;
        return batch;
    }
    auto* descriptor = [[MTL4PipelineDataSetSerializerDescriptor alloc] init];
    descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    auto next = [device newPipelineDataSetSerializerWithDescriptor:descriptor];
    if (!next) throw std::runtime_error("Unable to rotate native pipeline serializer");
    PipelineArchiveBatch batch{serializer, root, current, pending};
    if (synchronous_retry) {
        batch.temporary=std::move(synchronous_retry->temporary);
        synchronous_retry.reset();
    }
    serializer = next;
    pending = 0;
    return batch;
}
void PipelineDiskCache::RestoreDetached(const PipelineArchiveBatch& batch) {
    serializer=batch.serializer;
    pending+=batch.pipelines;
}
bool PipelineDiskCache::Flush(const std::atomic_bool* stop) {
    if(Cancelled(stats,stop))return false;
    if(!stats.enabled || !serializer || !pending)return true;
    const auto publish = [&] {
        const auto publication = Publish(*synchronous_retry, stop);
        Published(publication);
        if (publication.saved) {
            pending -= synchronous_retry->pipelines;
            synchronous_retry.reset();
        }
        return publication.saved;
    };
    if (synchronous_retry && !publish()) return false;
    if (!pending) return true;
    synchronous_retry.emplace(PipelineArchiveBatch{serializer, root, current, pending});
    return publish();
}
} // namespace Metal
