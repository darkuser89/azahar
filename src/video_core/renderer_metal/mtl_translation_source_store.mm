// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include <algorithm>
#include <cstring>
#include <fstream>
#include <optional>
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#include "mtl_shader_disk_cache.h"
#include "mtl_translation_source_store.h"

namespace Metal {
namespace {
using Bytes = std::vector<std::byte>;
constexpr std::string_view Magic = "AZM4TXT1";
constexpr std::size_t MaxSource = 8 * 1024 * 1024, MaxBindings = 128,
                      MaxFile = MaxSource + 20 * MaxBindings + 52;
struct Input {
    ShaderStage stage;
    std::string source;
    std::vector<ShaderBindingRemap> bindings;
};
void Word(Bytes& bytes, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
        bytes.push_back(std::byte(value >> (8 * i)));
}
void Validate(const Input& input) {
    if (std::uint32_t(input.stage) > std::uint32_t(ShaderStage::VertexMesh) ||
        input.source.empty() || input.source.size() > MaxSource ||
        input.bindings.size() > MaxBindings)
        throw std::invalid_argument("Invalid portable translation input");
}
Bytes Serialize(const Input& input) {
    Validate(input);
    Bytes bytes;
    for (char c : Magic)
        bytes.push_back(std::byte(c));
    Word(bytes, std::uint32_t(input.stage));
    Word(bytes, std::uint32_t(input.source.size()));
    const auto source = std::as_bytes(std::span(input.source));
    bytes.insert(bytes.end(), source.begin(), source.end());
    Word(bytes, std::uint32_t(input.bindings.size()));
    for (const auto& b : input.bindings)
        for (auto value : {b.descriptor_set, b.binding, b.buffer, b.texture, b.sampler})
            Word(bytes, value);
    return bytes;
}
std::array<unsigned char, 32> Digest(std::span<const std::byte> bytes) {
    std::array<unsigned char, 32> result{};
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), result.data());
    return result;
}
bool PlainDirectory(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_directory(std::filesystem::symlink_status(path, error)) && !error;
}
bool HashName(std::string_view name) {
    return name.size() == 64 && std::all_of(name.begin(), name.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}
Input Read(const std::filesystem::path& path) {
    std::error_code error;
    if (!HashName(path.stem().string()) ||
        !std::filesystem::is_regular_file(std::filesystem::symlink_status(path, error)) || error)
        throw std::invalid_argument("Translation source is not a regular keyed file");
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < 52 || size > MaxFile)
        throw std::invalid_argument("Translation source size");
    Bytes bytes(size);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(size));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("Incomplete translation source");
    const auto payload = std::span<const std::byte>(bytes).first(size - 32);
    const auto checksum = Digest(payload);
    if (std::memcmp(checksum.data(), bytes.data() + size - 32, 32) ||
        ShaderDiskCache::Identity(payload) != path.stem().string())
        throw std::invalid_argument("Translation source checksum/key mismatch");
    std::size_t pos{};
    const auto take = [&](std::size_t count) {
        if (count > payload.size() - pos)
            throw std::invalid_argument("Truncated translation source");
        const auto value = payload.subspan(pos, count);
        pos += count;
        return value;
    };
    const auto word = [&] {
        const auto value = take(4);
        std::uint32_t result{};
        for (unsigned i = 0; i < 4; ++i)
            result |= std::to_integer<std::uint32_t>(value[i]) << (8 * i);
        return result;
    };
    if (!std::ranges::equal(take(8), std::as_bytes(std::span(Magic))))
        throw std::invalid_argument("Translation source schema");
    Input input{ShaderStage(word()), {}, {}};
    const auto length = word();
    if (length > MaxSource)
        throw std::invalid_argument("Translation source length");
    const auto source = take(length);
    input.source.assign(reinterpret_cast<const char*>(source.data()), source.size());
    const auto count = word();
    if (count > MaxBindings)
        throw std::invalid_argument("Translation source binding count");
    for (unsigned i = 0; i < count; ++i)
        input.bindings.push_back({word(), word(), word(), word(), word()});
    Validate(input);
    if (pos != payload.size())
        throw std::invalid_argument("Translation source trailing data");
    return input;
}
} // namespace
std::filesystem::path TranslationSourceStore::Directory(
    const std::filesystem::path& shader_directory) {
    return shader_directory.empty() ? std::filesystem::path{}
                                    : shader_directory.parent_path() / "translation-inputs" / "v1";
}
void TranslationSourceStore::Remember(const std::filesystem::path& path,
                                      const std::string& identity) {
    std::scoped_lock lock{mutex};
    if (saved.size() >= 512)
        saved.clear();
    saved.emplace(path, identity);
}
TranslationSourceStats TranslationSourceStore::Stats() const {
    std::scoped_lock lock{mutex};
    return stats;
}
void TranslationSourceStore::Clear() {
    std::scoped_lock lock{mutex};
    saved.clear();
}
bool TranslationSourceStore::Save(const std::filesystem::path& shader_directory,
                                  std::string_view source, ShaderStage stage,
                                  std::span<const ShaderBindingRemap> bindings) {
    const auto directory = Directory(shader_directory);
    if (directory.empty())
        return false;
    std::filesystem::path temporary;
    try {
        // Check before memo hits, reads or directory creation: an existing
        // symlink must not redirect even a valid input or an empty mkdir.
        for (const auto& path : {directory.parent_path(), directory}) {
            std::error_code error;
            const auto status = std::filesystem::symlink_status(path, error);
            if (std::filesystem::exists(status) && !std::filesystem::is_directory(status))
                throw std::runtime_error("Translation source directory is not plain");
            if (error && error != std::errc::no_such_file_or_directory)
                throw std::runtime_error("Cannot inspect translation source directory");
        }
        if (source.empty() || source.size() > MaxSource || bindings.size() > MaxBindings ||
            std::uint32_t(stage) > std::uint32_t(ShaderStage::VertexMesh))
            throw std::invalid_argument("Translation source input bounds");
        const Input input{stage, std::string(source), {bindings.begin(), bindings.end()}};
        auto bytes = Serialize(input);
        const auto identity = ShaderDiskCache::Identity(bytes);
        {
            std::scoped_lock lock{mutex};
            if (saved.contains({directory, identity}))
                return true;
        }
        const auto path = directory / (identity + ".input");
        try {
            (void)Read(path);
            Remember(directory, identity);
            return true;
        } catch (const std::exception&) {
        }
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error || !PlainDirectory(directory) || !PlainDirectory(directory.parent_path()))
            throw std::runtime_error("Translation source directory");
        const auto checksum = Digest(bytes);
        const auto digest = std::as_bytes(std::span(checksum));
        bytes.insert(bytes.end(), digest.begin(), digest.end());
        temporary = path.string() + ".pending-" + NSUUID.UUID.UUIDString.UTF8String;
        std::ofstream out(temporary, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        out.close();
        if (!out)
            throw std::runtime_error("Write translation source");
        std::filesystem::rename(temporary, path, error);
        if (error)
            throw std::runtime_error("Publish translation source");
        Remember(directory, identity);
        {
            std::scoped_lock lock{mutex};
            ++stats.saved;
        }
        return true;
    } catch (const std::exception&) {
        std::error_code error;
        if (!temporary.empty())
            std::filesystem::remove(temporary, error);
        std::scoped_lock lock{mutex};
        ++stats.failures;
        return false;
    }
}
TranslationSourcePrewarm TranslationSourceStore::Prewarm(
    const std::filesystem::path& shader_directory, const std::atomic_bool& stop,
    const std::function<bool()>& current, std::size_t max_records) {
    TranslationSourcePrewarm result{};
    const auto directory = Directory(shader_directory);
    const auto cancelled = [&] { return stop.load() || (current && !current()); };
    if (cancelled()) {
        result.cancelled = true;
        return result;
    }
    if (directory.empty() || !max_records)
        return result;
    std::error_code error;
    if (!std::filesystem::exists(std::filesystem::symlink_status(directory, error)))
        return result;
    if (!PlainDirectory(directory) || !PlainDirectory(directory.parent_path())) {
        std::scoped_lock lock{mutex};
        ++stats.rejected;
        return result;
    }
    max_records = std::min<std::size_t>(max_records, 256);
    std::size_t visits = 4096;
    std::uintmax_t bytes = 64ULL * 1024 * 1024;
    std::vector<std::filesystem::path> paths;
    for (std::filesystem::directory_iterator it(directory, error), end;
         !error && it != end && !cancelled(); it.increment(error)) {
        if (!visits) {
            result.budget_limited = true;
            break;
        }
        --visits;
        if (it->path().extension() != ".input")
            continue;
        const auto size = it->file_size(error);
        if (error) {
            error.clear();
            std::scoped_lock lock{mutex};
            ++stats.rejected;
            continue;
        }
        if (size > MaxFile) {
            std::scoped_lock lock{mutex};
            ++stats.rejected;
            continue;
        }
        if (size > bytes) {
            result.budget_limited = true;
            break;
        }
        bytes -= size;
        paths.push_back(it->path());
        if (paths.size() == max_records) {
            result.budget_limited = true;
            break;
        }
    }
    if (error) {
        std::scoped_lock lock{mutex};
        ++stats.failures;
    }
    std::sort(paths.begin(), paths.end());
    result.candidates = paths.size();
    TranslationInputScope muted;
    for (const auto& path : paths) {
        if (cancelled()) {
            result.cancelled = true;
            break;
        }
        ++result.visited;
        std::optional<Input> input;
        try {
            input = Read(path);
        } catch (const std::exception&) {
            std::scoped_lock lock{mutex};
            ++stats.rejected;
            ++result.failed;
            continue;
        }
        try {
            (void)TranslateGuestGLSL(input->source, input->stage, input->bindings);
            if (cancelled()) {
                result.cancelled = true;
                break;
            }
            Remember(directory, path.stem().string());
            ++result.prepared;
        } catch (const std::exception&) {
            ++result.failed;
        }
    }
    result.cancelled = result.cancelled || cancelled();
    return result;
}
} // namespace Metal
