// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "mtl_pipeline_cache.h"

namespace Metal {
struct HistoricalPipelineHint {
    using Source = std::pair<std::string, std::string>; // Full library name and MSL bytes.
    RenderPipelineRecord record;
    std::shared_ptr<const Source> vertex_source, fragment_source;
};
// Each record contains every field used to construct the backend's MTL4 render descriptor.
// Dynamic viewport/scissor/cull/depth state is encoded separately and is not part of a PSO.
class RenderPipelineManifest final {
public:
    static bool SaveLibrary(const std::filesystem::path& directory, std::string_view identity,
                             std::string_view name, std::string_view source);
    static std::optional<std::pair<std::string, std::string>> LoadLibrary(
        const std::filesystem::path& directory, std::string_view identity);
    static std::string Identity(const RenderPipelineRecord& record);
    bool Save(const std::filesystem::path& directory, const RenderPipelineRecord& record);
    std::vector<RenderPipelineRecord> Load(const std::filesystem::path& directory,
                                          const std::atomic_bool& stop, bool& limited,
                                          std::size_t* remaining_visits = nullptr);
    // Hints only: old CPU shader IR/reflection is never loaded. The caller rebuilds
    // native libraries/PSOs with the current factory and complete current key.
    std::vector<HistoricalPipelineHint> LoadHistorical(const std::filesystem::path& directory,
        const std::atomic_bool& stop, bool& limited, std::size_t max_records = 4096,
        std::size_t* remaining_visits = nullptr);
    std::uint64_t Rejected() const { return rejected; }
    std::uint64_t Failures() const { return failures; }
private:
    std::uint64_t rejected{}, failures{};
};
} // namespace Metal
