// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <atomic>
#include <filesystem>
#include <mutex>
#include <set>
#include "mtl_shader_translation.h"

namespace Metal {
struct TranslationSourceStats {
    std::uint64_t saved{}, rejected{}, failures{};
};
struct TranslationSourcePrewarm {
    std::size_t candidates{}, visited{}, prepared{}, failed{};
    bool cancelled{}, budget_limited{};
};
// Portable source hints, never translated IR/reflection. Every warm result is
// rebuilt by the current compiler and selected through its complete input key.
class TranslationSourceStore final {
public:
    static std::filesystem::path Directory(const std::filesystem::path& shader_directory);
    bool Save(const std::filesystem::path& shader_directory, std::string_view source,
              ShaderStage stage, std::span<const ShaderBindingRemap> bindings);
    TranslationSourcePrewarm Prewarm(const std::filesystem::path& shader_directory,
                                     const std::atomic_bool& stop,
                                     const std::function<bool()>& current,
                                     std::size_t max_records = 256);
    TranslationSourceStats Stats() const;
    void Clear();

private:
    mutable std::mutex mutex;
    std::set<std::pair<std::filesystem::path, std::string>> saved;
    TranslationSourceStats stats{};
    void Remember(const std::filesystem::path& path, const std::string& identity);
};
} // namespace Metal
