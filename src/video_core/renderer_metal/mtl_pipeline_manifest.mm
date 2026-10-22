// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_pipeline_manifest.h"
#include "mtl_shader_disk_cache.h"
#include "mtl_shader_cache_version.h"
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace Metal {
namespace {
using Bytes = std::vector<std::byte>;
constexpr std::size_t MaxBytes = 4096, MaxRecords = 4096, MaxVisited = 16384;
constexpr std::array<std::byte,8> Magic{std::byte{'A'},std::byte{'Z'},std::byte{'M'},std::byte{'4'},
    std::byte{'R'},std::byte{'P'},std::byte{'S'},std::byte{1}};
constexpr std::array<std::byte,8> MeshMagic{std::byte{'A'},std::byte{'Z'},std::byte{'M'},std::byte{'4'},
    std::byte{'M'},std::byte{'P'},std::byte{'S'},std::byte{1}};
void Word(Bytes& bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.push_back(std::byte(value >> (8 * i)));
}
void String(Bytes& bytes, std::string_view value) {
    if (value.size() > 256) throw std::invalid_argument("Native pipeline manifest string too long");
    Word(bytes, std::uint32_t(value.size()));
    const auto data = std::as_bytes(std::span(value));
    bytes.insert(bytes.end(),data.begin(),data.end());
}
bool HashName(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
Bytes Serialize(const RenderPipelineRecord& record, std::string_view version = ShaderDiskCacheSourceVersion) {
    if (!HashName(record.vertex_identity) || record.vertex_entry.empty() ||
        (!record.fragment_identity.empty() && (!HashName(record.fragment_identity) || record.fragment_entry.empty())) ||
        (record.fragment_identity.empty() && !record.fragment_entry.empty()) ||
        record.blend.color_write_mask > 15 || record.blend.logic_operation > 15 ||
        std::uint64_t(record.color_format) > UINT32_MAX)
        throw std::invalid_argument("Invalid native pipeline manifest specialization");
    const auto& magic=record.mesh?MeshMagic:Magic;
    Bytes bytes(magic.begin(),magic.end());
    String(bytes,version);
    String(bytes,record.vertex_identity); String(bytes,record.vertex_entry);
    String(bytes,record.fragment_identity); String(bytes,record.fragment_entry);
    Word(bytes,std::uint32_t(record.color_format));
    const auto& b = record.blend;
    Word(bytes,b.enabled); Word(bytes,b.color_write_mask); Word(bytes,b.logic_operation);
    Word(bytes,b.source_rgb); Word(bytes,b.dest_rgb); Word(bytes,b.source_alpha); Word(bytes,b.dest_alpha);
    Word(bytes,b.rgb_equation); Word(bytes,b.alpha_equation);
    Word(bytes,b.emulate_rgb_minmax); Word(bytes,b.emulate_alpha_minmax);
    // These fixed descriptor fields are part of the persisted ABI too.
    Word(bytes,1); // rasterSampleCount
    // Typed schemas keep existing vertex manifests byte-for-byte compatible.
    // A mesh descriptor has a maximum thread count, not inputPrimitiveTopology.
    Word(bytes,record.mesh?96:std::uint32_t(MTLPrimitiveTopologyClassTriangle));
    Word(bytes,!record.fragment_identity.empty()); // rasterizationEnabled
    return bytes;
}
class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes_) : bytes{bytes_} {}
    std::span<const std::byte> Take(std::size_t size) {
        if (size > bytes.size()) throw std::invalid_argument("Truncated native pipeline manifest");
        const auto result=bytes.first(size); bytes=bytes.subspan(size); return result;
    }
    std::uint32_t Word() {
        const auto value=Take(4);std::uint32_t result{};
        for(unsigned i=0;i<4;++i)result|=std::to_integer<std::uint32_t>(value[i])<<(8*i);
        return result;
    }
    std::uint32_t Value(std::uint32_t maximum) {
        const auto value=Word();if(value>maximum)throw std::invalid_argument("Invalid pipeline manifest field");return value;
    }
    bool Bool() { return Value(1)!=0; }
    std::string String() {
        const auto data=Take(Value(256));return {reinterpret_cast<const char*>(data.data()),data.size()};
    }
    bool Empty() const { return bytes.empty(); }
private:
    std::span<const std::byte> bytes;
};
std::array<unsigned char,32> Checksum(std::span<const std::byte> bytes) {
    std::array<unsigned char,32> result{};CC_SHA256(bytes.data(),CC_LONG(bytes.size()),result.data());return result;
}
RenderPipelineRecord Read(const std::filesystem::path& path, std::string_view version = ShaderDiskCacheSourceVersion) {
    std::error_code error;
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path,error)) || error)
        throw std::invalid_argument("Pipeline manifest is not a regular file");
    const auto size=std::filesystem::file_size(path,error);
    if(error || size<40 || size>MaxBytes)throw std::invalid_argument("Invalid pipeline manifest size");
    Bytes bytes(size);std::ifstream input(path,std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(size));
    if(!input || input.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("Incomplete pipeline manifest read");
    const auto payload=std::span<const std::byte>{bytes}.first(size-32);
    const auto checksum=Checksum(payload);
    if(std::memcmp(checksum.data(),bytes.data()+size-32,32))throw std::invalid_argument("Pipeline manifest checksum mismatch");
    if(ShaderDiskCache::Identity(payload)!=path.stem().string())throw std::invalid_argument("Pipeline manifest key mismatch");
    Reader reader{payload};
    RenderPipelineRecord record{};
    const auto magic=reader.Take(Magic.size());
    if(std::ranges::equal(magic,MeshMagic))record.mesh=true;
    else if(!std::ranges::equal(magic,Magic))throw std::invalid_argument("Pipeline manifest schema mismatch");
    if(reader.String()!=version)
        throw std::invalid_argument("Pipeline manifest source version mismatch");
    record.vertex_identity=reader.String();record.vertex_entry=reader.String();
    record.fragment_identity=reader.String();record.fragment_entry=reader.String();
    record.color_format=MTLPixelFormat(reader.Word());
    auto& b=record.blend;
    b.enabled=reader.Bool();b.color_write_mask=std::uint8_t(reader.Value(15));b.logic_operation=std::uint8_t(reader.Value(15));
    b.source_rgb=std::uint8_t(reader.Value(255));b.dest_rgb=std::uint8_t(reader.Value(255));
    b.source_alpha=std::uint8_t(reader.Value(255));b.dest_alpha=std::uint8_t(reader.Value(255));
    b.rgb_equation=std::uint8_t(reader.Value(255));b.alpha_equation=std::uint8_t(reader.Value(255));
    b.emulate_rgb_minmax=reader.Bool();b.emulate_alpha_minmax=reader.Bool();
    if(reader.Word()!=1 || reader.Word()!=(record.mesh?96:std::uint32_t(MTLPrimitiveTopologyClassTriangle)) ||
       reader.Bool()!=!record.fragment_identity.empty() || !reader.Empty() || Serialize(record,version)!=Bytes(payload.begin(),payload.end()))
        throw std::invalid_argument("Pipeline manifest descriptor ABI mismatch");
    return record;
}
std::filesystem::path File(const std::filesystem::path& directory, const RenderPipelineRecord& record) {
    return directory/ShaderDiskCacheSourceVersion/(RenderPipelineManifest::Identity(record)+".pipeline");
}
} // namespace

std::string RenderPipelineManifest::Identity(const RenderPipelineRecord& record) {
    return ShaderDiskCache::Identity(Serialize(record));
}
bool RenderPipelineManifest::Save(const std::filesystem::path& directory,const RenderPipelineRecord& record) {
    if(directory.empty())return false;
    std::filesystem::path temporary;
    try {
        const auto path=File(directory,record);
        try { (void)Read(path);return true; } catch(const std::exception&) {}
        std::error_code error;std::filesystem::create_directories(path.parent_path(),error);
        if(error){++failures;return false;}
        auto bytes=Serialize(record);const auto checksum=Checksum(bytes);
        const auto digest=std::as_bytes(std::span(checksum));bytes.insert(bytes.end(),digest.begin(),digest.end());
        temporary=path.string()+".pending-"+NSUUID.UUID.UUIDString.UTF8String;
        std::ofstream output(temporary,std::ios::binary);output.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));output.close();
        if(!output){std::filesystem::remove(temporary,error);++failures;return false;}
        std::filesystem::rename(temporary,path,error);
        if(error){std::filesystem::remove(temporary,error);++failures;return false;}
        return true;
    } catch(const std::exception&) {
        std::error_code error;if(!temporary.empty())std::filesystem::remove(temporary,error);
        ++failures;return false;
    }
}
std::vector<RenderPipelineRecord> RenderPipelineManifest::Load(const std::filesystem::path& directory,
    const std::atomic_bool& stop,bool& limited,std::size_t* remaining_visits) {
    std::vector<RenderPipelineRecord> records;limited=false;
    if(directory.empty() || stop.load())return records;
    const auto path=directory/ShaderDiskCacheSourceVersion;
    std::error_code error;const auto status=std::filesystem::symlink_status(path,error);
    if(error || !std::filesystem::exists(status))return records;
    if(!std::filesystem::is_directory(status)){++rejected;return records;}
    std::size_t local_visits=MaxVisited;
    auto& remaining=remaining_visits ? *remaining_visits:local_visits;
    for(std::filesystem::directory_iterator it(path,error),end;!error && it!=end && !stop.load();it.increment(error)) {
        if(!remaining){limited=true;break;}
        --remaining;
        if(it->path().extension()!=".pipeline")continue;
        try { records.push_back(Read(it->path())); } catch(const std::exception&) {++rejected;}
        if(records.size()==MaxRecords){limited=true;break;}
    }
    if(error)++failures;
    return records;
}
} // namespace Metal

namespace Metal {
namespace {
constexpr std::size_t MaxLibraryBytes=16*1024*1024;
constexpr std::string_view LibraryMagic="AZM4LIB1";
void LibraryString(Bytes& bytes,std::string_view value) {
    if(value.size()>8*1024*1024)throw std::invalid_argument("Native library source too large");
    Word(bytes,std::uint32_t(value.size()));const auto data=std::as_bytes(std::span(value));
    bytes.insert(bytes.end(),data.begin(),data.end());
}
std::filesystem::path LibraryFile(const std::filesystem::path& directory,std::string_view identity) {
    return directory/"libraries"/ShaderDiskCacheSourceVersion/(std::string(identity)+".library");
}
}
bool RenderPipelineManifest::SaveLibrary(const std::filesystem::path& directory,std::string_view identity,
                                         std::string_view name,std::string_view source) {
    if(directory.empty())return false;
    std::filesystem::path temporary;
    try {
        if(!HashName(identity)||Device::SourceIdentity(name,source)!=identity)return false;
        const auto path=LibraryFile(directory,identity);std::error_code error;
        if(std::filesystem::is_regular_file(std::filesystem::symlink_status(path,error))&&!error)
            if(LoadLibrary(directory,identity))return true;
        error.clear();std::filesystem::create_directories(path.parent_path(),error);if(error)return false;
        Bytes bytes;const auto magic=std::as_bytes(std::span(LibraryMagic));bytes.insert(bytes.end(),magic.begin(),magic.end());
        LibraryString(bytes,name);LibraryString(bytes,source);const auto checksum=Checksum(bytes);
        const auto digest=std::as_bytes(std::span(checksum));bytes.insert(bytes.end(),digest.begin(),digest.end());
        if(bytes.size()>MaxLibraryBytes)return false;
        temporary=path.string()+".pending-"+NSUUID.UUID.UUIDString.UTF8String;
        std::ofstream output(temporary,std::ios::binary);output.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));output.close();
        if(!output){std::filesystem::remove(temporary,error);return false;}
        std::filesystem::rename(temporary,path,error);if(error){std::filesystem::remove(temporary,error);return false;}
        return true;
    }catch(const std::exception&){std::error_code error;if(!temporary.empty())std::filesystem::remove(temporary,error);return false;}
}
std::optional<std::pair<std::string,std::string>> RenderPipelineManifest::LoadLibrary(
    const std::filesystem::path& directory,std::string_view identity) {
    if(directory.empty()||!HashName(identity))return std::nullopt;
    try {
        const auto path=LibraryFile(directory,identity);std::error_code error;
        if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path,error))||error)return std::nullopt;
        const auto size=std::filesystem::file_size(path,error);if(error||size<48||size>MaxLibraryBytes)return std::nullopt;
        Bytes bytes(size);std::ifstream input(path,std::ios::binary);input.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(size));
        if(!input||input.peek()!=std::char_traits<char>::eof())return std::nullopt;
        const auto payload=std::span<const std::byte>{bytes}.first(size-32);const auto checksum=Checksum(payload);
        if(std::memcmp(checksum.data(),bytes.data()+size-32,32))return std::nullopt;
        Reader reader{payload};const auto magic=std::as_bytes(std::span(LibraryMagic));
        if(!std::ranges::equal(reader.Take(8),magic))return std::nullopt;
        const auto get_string=[&] {
            const auto data=reader.Take(reader.Value(8*1024*1024));
            return std::string(reinterpret_cast<const char*>(data.data()),data.size());
        };
        auto name=get_string(),source=get_string();
        if(!reader.Empty()||source.empty()||Device::SourceIdentity(name,source)!=identity)return std::nullopt;
        return std::pair{std::move(name),std::move(source)};
    }catch(const std::exception&){return std::nullopt;}
}

namespace {
struct HistoricalLimit {};
struct HistoricalBudget {
    std::size_t& visits;
    std::uintmax_t bytes{256ULL*1024*1024};
    void Visit() { if(!visits)throw HistoricalLimit{};--visits; }
    void ReadBytes(std::uintmax_t size) {
        if(size>bytes)throw HistoricalLimit{};
        bytes-=size;
    }
};
bool PlainDirectory(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_directory(std::filesystem::symlink_status(path,error))&&!error;
}
// AZM4LIB1's historical identity is length-prefixed version/name/full MSL.
// This validates the old file only. Actual PSO keys always use Device's CURRENT identity.
std::string HistoricalLibraryIdentity(std::string_view version,std::string_view name,std::string_view source) {
    Bytes bytes;
    for(const auto value:{version,name,source}) {
        const auto size=std::uint64_t(value.size());
        for(unsigned i=0;i<8;++i)bytes.push_back(std::byte(size>>(8*i)));
        const auto data=std::as_bytes(std::span(value));bytes.insert(bytes.end(),data.begin(),data.end());
    }
    return ShaderDiskCache::Identity(bytes);
}
std::shared_ptr<const HistoricalPipelineHint::Source> HistoricalLibrary(
    const std::filesystem::path& directory,std::string_view version,std::string_view identity,
    HistoricalBudget& budget) {
    budget.Visit();
    if(!HashName(version)||!HashName(identity)||!PlainDirectory(directory/"libraries")||
       !PlainDirectory(directory/"libraries"/version))
        throw std::invalid_argument("Invalid historical native library directory/key");
    const auto path=directory/"libraries"/version/(std::string(identity)+".library");
    std::error_code error;
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path,error))||error)
        throw std::invalid_argument("Historical native library is not a regular file");
    const auto size=std::filesystem::file_size(path,error);
    if(error||size<48||size>MaxLibraryBytes)throw std::invalid_argument("Historical native library size");
    budget.ReadBytes(size);
    Bytes bytes(size);std::ifstream input(path,std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(size));
    if(!input||input.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("Historical library read");
    const auto payload=std::span<const std::byte>{bytes}.first(size-32);
    const auto checksum=Checksum(payload);
    if(std::memcmp(checksum.data(),bytes.data()+size-32,32))throw std::invalid_argument("Historical library checksum");
    Reader reader{payload};
    if(!std::ranges::equal(reader.Take(8),std::as_bytes(std::span(LibraryMagic))))
        throw std::invalid_argument("Historical library schema");
    const auto string=[&] {
        const auto data=reader.Take(reader.Value(8*1024*1024));
        return std::string(reinterpret_cast<const char*>(data.data()),data.size());
    };
    auto name=string(),source=string();
    if(!reader.Empty()||source.empty()||HistoricalLibraryIdentity(version,name,source)!=identity)
        throw std::invalid_argument("Historical library identity");
    return std::make_shared<const HistoricalPipelineHint::Source>(std::move(name),std::move(source));
}
}

std::vector<HistoricalPipelineHint> RenderPipelineManifest::LoadHistorical(
    const std::filesystem::path& directory,const std::atomic_bool& stop,bool& limited,
    std::size_t max_records,std::size_t* remaining_visits) {
    std::vector<HistoricalPipelineHint> result;limited=false;
    if(directory.empty()||stop.load()||!max_records)return result;
    if(!PlainDirectory(directory)){++rejected;return result;}
    max_records=std::min(max_records,MaxRecords);
    std::size_t local_visits=MaxVisited;
    HistoricalBudget budget{remaining_visits?*remaining_visits:local_visits};
    std::map<std::pair<std::string,std::string>,std::shared_ptr<const HistoricalPipelineHint::Source>> sources;
    const auto source=[&](const std::string& version,const std::string& identity) {
        const auto key=std::make_pair(version,identity);
        if(const auto found=sources.find(key);found!=sources.end())return found->second;
        std::shared_ptr<const HistoricalPipelineHint::Source> value;
        if(!stop.load())try { value=HistoricalLibrary(directory,version,identity,budget); }
        catch(const std::exception&) {} // Cache a rejection too; never reread it per descriptor.
        sources.emplace(key,value);return value;
    };
    std::error_code error;
    try {
        for(std::filesystem::directory_iterator versions(directory,error),end;!error&&versions!=end&&!stop.load();versions.increment(error)) {
            budget.Visit();
            const auto version=versions->path().filename().string();
            if(!HashName(version)||version==ShaderDiskCacheSourceVersion)continue;
            if(!PlainDirectory(versions->path())){++rejected;continue;}
            std::error_code scan_error;
            for(std::filesystem::directory_iterator files(versions->path(),scan_error),last;!scan_error&&files!=last&&!stop.load();files.increment(scan_error)) {
                budget.Visit();
                if(files->path().extension()!=".pipeline")continue;
                try {
                    const auto size=std::filesystem::file_size(files->path(),scan_error);
                    if(scan_error){scan_error.clear();++rejected;continue;}
                    if(size>MaxBytes){++rejected;continue;}
                    budget.ReadBytes(size);
                    auto record=Read(files->path(),version);
                    auto vertex=source(version,record.vertex_identity);
                    auto fragment=record.fragment_identity.empty()?nullptr:source(version,record.fragment_identity);
                    if(stop.load())return result;
                    if(!vertex||(!record.fragment_identity.empty()&&!fragment)){++rejected;continue;}
                    result.push_back({std::move(record),std::move(vertex),std::move(fragment)});
                    if(result.size()==max_records){limited=true;return result;}
                } catch(const std::exception&) {++rejected;}
            }
            if(scan_error)++failures;
        }
    } catch(const HistoricalLimit&) {limited=true;}
    if(error)++failures;
    return result;
}
} // namespace Metal
