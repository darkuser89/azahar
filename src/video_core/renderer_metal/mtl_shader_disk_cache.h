// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include "mtl_fragment_operations.h"
#include <filesystem>
#include <optional>
#include <atomic>
#include <mutex>

namespace Metal {
struct ShaderDiskRecord {
    TranslatedShader shader;
    FragmentOperations operations;
};
struct ShaderDiskCacheStats {
    std::uint64_t hits{}, translations{}, saved{}, rejected{}, io_failures{};
    std::uint64_t library_compiles{}, prewarmed{}, prewarm_hits{};
};
struct ShaderDiskEntry {
    std::filesystem::path path;
    std::string identity;
    std::uintmax_t bytes{};
};

// Owner lookups and deferred compiler publication share this cache safely.
// Keys contain the complete PICA specialization;
// the directory version also covers the source generator, translation and ABI code.
class ShaderDiskCache final {
public:
    std::optional<ShaderDiskRecord> Load(const std::filesystem::path& directory,
                                       std::span<const std::byte> key, ShaderStage stage);
    bool Save(const std::filesystem::path& directory, std::span<const std::byte> key,
              const ShaderDiskRecord& record);
    static std::string Identity(std::span<const std::byte> key);
    std::vector<ShaderDiskEntry> Entries(const std::filesystem::path& directory,
                                         const std::atomic_bool& stop, bool& limited);
    std::optional<ShaderDiskRecord> Load(const ShaderDiskEntry& entry);
    void Translated() { std::scoped_lock lock{mutex}; ++stats.translations; }
    void Reject() { std::scoped_lock lock{mutex}; ++stats.rejected; }
    void LibraryCompiled() { std::scoped_lock lock{mutex}; ++stats.library_compiles; }
    void Prewarmed() { std::scoped_lock lock{mutex}; ++stats.prewarmed; }
    void PrewarmHit() { std::scoped_lock lock{mutex}; ++stats.prewarm_hits; }
    ShaderDiskCacheStats Stats() const { std::scoped_lock lock{mutex}; return stats; }

private:
    mutable std::mutex mutex;
    ShaderDiskCacheStats stats;
};
} // namespace Metal
