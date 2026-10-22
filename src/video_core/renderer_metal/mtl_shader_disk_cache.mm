// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_shader_disk_cache.h"
#include "mtl_shader_cache_version.h"
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <set>
#include <stdexcept>

namespace Metal {
namespace {
constexpr std::size_t MaxFileBytes = 16 * 1024 * 1024;
constexpr std::size_t MaxStringBytes = 8 * 1024 * 1024;
constexpr std::array<std::byte, 8> Magic{std::byte{'A'}, std::byte{'Z'}, std::byte{'M'},
    std::byte{'4'}, std::byte{'S'}, std::byte{'R'}, std::byte{'C'}, std::byte{1}};
using Bytes = std::vector<std::byte>;
using Hash = std::array<std::byte, CC_SHA256_DIGEST_LENGTH>;
Hash Digest(std::span<const std::byte> bytes) {
    Hash hash{};
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), reinterpret_cast<unsigned char*>(hash.data()));
    return hash;
}
std::string Hex(const Hash& hash) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(hash.size() * 2);
    for (const auto value : hash) {
        const auto byte = std::to_integer<unsigned>(value);
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}
void Word(Bytes& bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.push_back(std::byte(value >> (8 * i)));
}
void String(Bytes& bytes, const std::string& value) {
    if (value.size() > MaxStringBytes) throw std::length_error("Native shader cache string too large");
    Word(bytes, std::uint32_t(value.size()));
    const auto data = std::as_bytes(std::span(value));
    bytes.insert(bytes.end(), data.begin(), data.end());
}
class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes_) : bytes{bytes_} {}
    std::span<const std::byte> Take(std::size_t count) {
        if (count > bytes.size()) throw std::invalid_argument("Truncated native shader cache");
        const auto result = bytes.first(count);
        bytes = bytes.subspan(count);
        return result;
    }
    std::uint32_t Word() {
        const auto data = Take(4);
        std::uint32_t result{};
        for (unsigned i = 0; i < 4; ++i) result |= std::to_integer<std::uint32_t>(data[i]) << (8 * i);
        return result;
    }
    std::uint32_t Count(std::uint32_t maximum) {
        const auto value = Word();
        if (value > maximum) throw std::invalid_argument("Invalid native shader cache count");
        return value;
    }
    bool Bool() { return Count(1) != 0; }
    std::string String() {
        const auto data = Take(Count(MaxStringBytes));
        return {reinterpret_cast<const char*>(data.data()), data.size()};
    }
    bool Empty() const { return bytes.empty(); }
private:
    std::span<const std::byte> bytes;
};
void Channel(Bytes& bytes, const FragmentBlendChannel& channel) {
    Word(bytes, channel.enabled); Word(bytes, channel.equation);
    Word(bytes, channel.source_factor); Word(bytes, channel.dest_factor);
}
FragmentBlendChannel Channel(Reader& reader) {
    return {reader.Bool(), std::uint8_t(reader.Count(4)), std::uint8_t(reader.Count(15)),
            std::uint8_t(reader.Count(15))};
}
Bytes Serialize(std::span<const std::byte> key, const ShaderDiskRecord& record) {
    Bytes bytes(Magic.begin(), Magic.end());
    const auto hash = Digest(key);
    bytes.insert(bytes.end(), hash.begin(), hash.end());
    const auto& shader = record.shader;
    Word(bytes, std::uint32_t(shader.stage));
    String(bytes, shader.source); String(bytes, shader.entry_point);
    Word(bytes, shader.needs_buffer_sizes); Word(bytes, shader.uses_framebuffer_fetch);
    if (shader.resources.size() > 128 || shader.spirv.size() > MaxStringBytes / 4)
        throw std::length_error("Native shader cache metadata too large");
    Word(bytes, std::uint32_t(shader.resources.size()));
    for (const auto& resource : shader.resources) {
        String(bytes, resource.name); Word(bytes, std::uint32_t(resource.kind));
        Word(bytes, resource.descriptor_set); Word(bytes, resource.binding);
        Word(bytes, resource.metal_index); Word(bytes, resource.sampler_index);
        Word(bytes, resource.count); Word(bytes, resource.byte_size);
        Word(bytes, resource.runtime_array_stride); Word(bytes, resource.active);
        if (resource.members.size() > 1024) throw std::length_error("Native shader members too large");
        Word(bytes, std::uint32_t(resource.members.size()));
        for (const auto& member : resource.members) {
            String(bytes, member.name); Word(bytes, member.offset); Word(bytes, member.size);
        }
    }
    Word(bytes, std::uint32_t(shader.spirv.size()));
    for (const auto word : shader.spirv) Word(bytes, word);
    const auto& operations = record.operations;
    Word(bytes, operations.logic_enabled); Word(bytes, operations.logic_operation);
    Word(bytes, std::uint32_t(operations.color_format));
    Channel(bytes, operations.rgb); Channel(bytes, operations.alpha);
    Word(bytes, operations.blend_color_from_pica_uniform);
    const auto checksum = Digest(bytes);
    bytes.insert(bytes.end(), checksum.begin(), checksum.end());
    if (bytes.size() > MaxFileBytes) throw std::length_error("Native shader cache file too large");
    return bytes;
}
ShaderDiskRecord Deserialize(std::span<const std::byte> bytes, const Hash& key_hash,
                             std::optional<ShaderStage> stage) {
    if (bytes.size() < Magic.size() + 2 * Hash{}.size())
        throw std::invalid_argument("Short native shader cache header");
    const auto payload = bytes.first(bytes.size() - Hash{}.size());
    const auto checksum = Digest(payload);
    if (!std::equal(checksum.begin(), checksum.end(), bytes.last(checksum.size()).begin()))
        throw std::invalid_argument("Native shader cache checksum mismatch");
    Reader reader{payload};
    if (!std::ranges::equal(reader.Take(Magic.size()), Magic) ||
        !std::ranges::equal(reader.Take(Hash{}.size()), key_hash))
        throw std::invalid_argument("Native shader cache version/key mismatch");
    ShaderDiskRecord record{};
    auto& shader = record.shader;
    shader.stage = ShaderStage(reader.Count(std::uint32_t(ShaderStage::VertexMesh)));
    if (stage && shader.stage != *stage) throw std::invalid_argument("Native shader cache stage mismatch");
    shader.source = reader.String(); shader.entry_point = reader.String();
    if (shader.source.empty() || shader.entry_point.empty())
        throw std::invalid_argument("Native shader cache has no source or entry point");
    shader.needs_buffer_sizes = reader.Bool(); shader.uses_framebuffer_fetch = reader.Bool();
    const auto resource_count = reader.Count(128);
    std::array<bool,31> buffer_slots{};
    std::array<bool,128> texture_slots{};
    std::array<bool,16> sampler_slots{};
    std::set<std::pair<std::uint32_t,std::uint32_t>> descriptors;
    const auto claim = [](auto& slots,std::uint32_t index,std::uint32_t count) {
        for (std::uint32_t slot=index;slot<index+count;++slot) {
            if (slots[slot]) throw std::invalid_argument("Overlapping native shader cache bindings");
            slots[slot]=true;
        }
    };
    for (std::uint32_t i = 0; i < resource_count; ++i) {
        ShaderResource resource{};
        resource.name = reader.String();
        resource.kind = ShaderResourceKind(reader.Count(std::uint32_t(ShaderResourceKind::Sampler)));
        resource.descriptor_set = reader.Word(); resource.binding = reader.Word();
        resource.metal_index = reader.Word(); resource.sampler_index = reader.Word();
        resource.count = reader.Count(128); resource.byte_size = reader.Word();
        resource.runtime_array_stride = reader.Word(); resource.active = reader.Bool();
        if (!resource.count) throw std::invalid_argument("Empty native shader resource array");
        const bool buffer = resource.kind == ShaderResourceKind::UniformBuffer ||
                            resource.kind == ShaderResourceKind::StorageBuffer;
        const auto slots = buffer ? 31U : resource.kind == ShaderResourceKind::Sampler ? 16U : 128U;
        if (resource.metal_index >= slots || resource.count > slots - resource.metal_index ||
            (resource.sampler_index != NoShaderBinding &&
             (resource.sampler_index >= 16 || resource.count > 16 - resource.sampler_index)))
            throw std::invalid_argument("Native shader resource exceeds Metal slots");
        if (!descriptors.emplace(resource.descriptor_set,resource.binding).second)
            throw std::invalid_argument("Duplicate native shader cache descriptor");
        if (buffer) claim(buffer_slots,resource.metal_index,resource.count);
        else if (resource.kind==ShaderResourceKind::Sampler) {
            if (resource.sampler_index!=NoShaderBinding && resource.sampler_index!=resource.metal_index)
                throw std::invalid_argument("Inconsistent native shader cache sampler binding");
            claim(sampler_slots,resource.metal_index,resource.count);
        } else {
            claim(texture_slots,resource.metal_index,resource.count);
            if (resource.sampler_index!=NoShaderBinding)
                claim(sampler_slots,resource.sampler_index,resource.count);
        }
        const auto member_count = reader.Count(1024);
        for (std::uint32_t j = 0; j < member_count; ++j) {
            ShaderMemberLayout member{reader.String(), reader.Word(), reader.Word()};
            if (member.offset > resource.byte_size || member.size > resource.byte_size - member.offset)
                throw std::invalid_argument("Native shader member exceeds its reflected buffer");
            resource.members.push_back(std::move(member));
        }
        shader.resources.push_back(std::move(resource));
    }
    if (shader.needs_buffer_sizes && buffer_slots[ShaderBufferSizesBinding])
        throw std::invalid_argument("Native shader cache overlaps its buffer-size binding");
    const auto words = reader.Count(MaxStringBytes / 4);
    if (words < 5) throw std::invalid_argument("Native shader cache has no compiler IR");
    shader.spirv.reserve(words);
    for (std::uint32_t i = 0; i < words; ++i) shader.spirv.push_back(reader.Word());
    if (shader.spirv[0] != 0x07230203) throw std::invalid_argument("Native shader IR magic mismatch");
    auto& operations = record.operations;
    operations.logic_enabled = reader.Bool(); operations.logic_operation = std::uint8_t(reader.Count(15));
    operations.color_format = FragmentColorFormat(reader.Count(std::uint32_t(FragmentColorFormat::RGBA4)));
    operations.rgb = Channel(reader); operations.alpha = Channel(reader);
    operations.blend_color_from_pica_uniform = reader.Bool();
    if (!reader.Empty()) throw std::invalid_argument("Trailing native shader cache bytes");
    return record;
}
std::filesystem::path File(const std::filesystem::path& directory, std::span<const std::byte> key) {
    return directory / ShaderDiskCacheSourceVersion / (Hex(Digest(key)) + ".shader");
}
std::optional<Hash> ParseIdentity(std::string_view identity) {
    if (identity.size() != 64) return std::nullopt;
    const auto digit = [](char c) { return c >= '0' && c <= '9' ? c - '0' :
                                        c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    Hash hash{};
    for (std::size_t i = 0; i < hash.size(); ++i) {
        const auto high = digit(identity[i * 2]), low = digit(identity[i * 2 + 1]);
        if (high < 0 || low < 0) return std::nullopt;
        hash[i] = std::byte((high << 4) | low);
    }
    return hash;
}
ShaderDiskRecord Read(const std::filesystem::path& path, const Hash& key_hash,
                      std::optional<ShaderStage> stage) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status))
        throw std::invalid_argument("Native shader cache is not a regular file");
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > MaxFileBytes) throw std::invalid_argument("Invalid native shader cache size");
    Bytes bytes(size);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("Incomplete native shader cache read");
    return Deserialize(bytes, key_hash, stage);
}
} // namespace

std::string ShaderDiskCache::Identity(std::span<const std::byte> key) { return Hex(Digest(key)); }

std::vector<ShaderDiskEntry> ShaderDiskCache::Entries(const std::filesystem::path& directory,
                                                    const std::atomic_bool& stop, bool& limited) {
    std::scoped_lock lock{mutex};
    std::vector<ShaderDiskEntry> entries;
    limited = false;
    if (directory.empty() || stop.load()) return entries;
    std::error_code error;
    const auto version_directory = directory / ShaderDiskCacheSourceVersion;
    const auto status = std::filesystem::symlink_status(version_directory, error);
    if (error || !std::filesystem::exists(status)) return entries;
    if (!std::filesystem::is_directory(status)) { ++stats.rejected; return entries; }
    constexpr std::size_t MaxVisited = 16384, MaxEntries = 4096;
    std::size_t visited{};
    for (std::filesystem::directory_iterator it(version_directory, error), end;
         !error && it != end && !stop.load() && visited++ < MaxVisited; it.increment(error)) {
        const auto path = it->path();
        if (path.extension() != ".shader") continue;
        const auto identity = path.stem().string();
        const auto file_status = it->symlink_status(error);
        if (error || !std::filesystem::is_regular_file(file_status) || !ParseIdentity(identity)) {
            ++stats.rejected; error.clear(); continue;
        }
        const auto size = it->file_size(error);
        if (error || size > MaxFileBytes) { ++stats.rejected; error.clear(); continue; }
        entries.push_back({path, identity, size});
        if (entries.size() == MaxEntries) { limited = true; break; }
    }
    if (visited >= MaxVisited) limited = true;
    if (error) ++stats.io_failures;
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.identity < b.identity; });
    return entries;
}

std::optional<ShaderDiskRecord> ShaderDiskCache::Load(const ShaderDiskEntry& entry) {
    std::scoped_lock lock{mutex};
    try {
        const auto identity = ParseIdentity(entry.identity);
        if (!identity || entry.path.stem().string() != entry.identity)
            throw std::invalid_argument("Invalid preloaded native shader identity");
        auto record = Read(entry.path, *identity, std::nullopt);
        ++stats.hits;
        return record;
    } catch (const std::exception&) { ++stats.rejected; return std::nullopt; }
}

std::optional<ShaderDiskRecord> ShaderDiskCache::Load(const std::filesystem::path& directory,
                                                    std::span<const std::byte> key, ShaderStage stage) {
    std::scoped_lock lock{mutex};
    if (directory.empty()) return std::nullopt;
    try {
        const auto path = File(directory, key);
        std::error_code error;
        const auto status = std::filesystem::symlink_status(path, error);
        if (error || !std::filesystem::exists(status)) return std::nullopt;
        auto record = Read(path, Digest(key), stage);
        ++stats.hits;
        return record;
    } catch (const std::exception&) { ++stats.rejected; return std::nullopt; }
}
bool ShaderDiskCache::Save(const std::filesystem::path& directory, std::span<const std::byte> key,
                           const ShaderDiskRecord& record) {
    std::scoped_lock lock{mutex};
    if (directory.empty()) return false;
    std::filesystem::path temporary;
    try {
        const auto path = File(directory, key);
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) { ++stats.io_failures; return false; }
        const auto bytes = Serialize(key, record);
        temporary = path.string() + ".pending-" + NSUUID.UUID.UUIDString.UTF8String;
        std::ofstream output(temporary, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        output.close();
        if (!output) { std::filesystem::remove(temporary, error); ++stats.io_failures; return false; }
        std::filesystem::rename(temporary, path, error);
        if (error) { std::filesystem::remove(temporary, error); ++stats.io_failures; return false; }
        ++stats.saved;
        return true;
    } catch (const std::exception&) {
        std::error_code error;
        if (!temporary.empty()) std::filesystem::remove(temporary, error);
        ++stats.io_failures;
        return false;
    }
}
} // namespace Metal
