// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_device.h"
#include "mtl_pica_shader_provider.h"
#include "mtl_geometry.h"
#include "mtl_shader_disk_cache.h"
#include "mtl_translation_source_store.h"
#include <compare>
#include <map>
#include <memory>
#include <functional>
#include <cstring>
#include <tuple>
#include <type_traits>

namespace Metal {
class NativeProgramSnapshot;
struct ShaderMachineProgram;

struct CompiledFragmentShader {
    PicaFragmentProgram program;
    id<MTLLibrary> library;
    std::uint8_t uniform_mask{PicaUniformMask(program.shader)};
    std::bitset<128> texture_mask{PicaTextureMask(program.shader)};
    NativeLibraryFuture pending_library{};
    id<MTLLibrary> Library() const {
        return library ? library : pending_library.get();
    }
};
struct CompiledMeshVertexProgram {
    TranslatedShader shader;
    id<MTLLibrary> library;
    std::uint8_t uniform_mask{PicaUniformMask(shader)};
    NativeLibraryFuture pending_library{};
    id<MTLLibrary> Library() const {
        return library ? library : pending_library.valid() ? pending_library.get():nil;
    }
};
struct ShaderPrewarmResult {
    std::size_t candidates{}, visited{}, compiled{}, failed{};
    bool cancelled{}, budget_limited{};
    std::size_t source_candidates{},source_prepared{},source_failed{};
};
struct MeshLookupStats {
    std::uint64_t owned_hits{}, owned_registrations{}, content_lookups{};
    std::size_t owned_entries{}, expired_owners{};
};

// CPU generation publishes immutable metadata with owned native completion futures.
// The device is fixed for this cache's lifetime; caller-held programs survive Clear().
class ShaderRuntime final {
public:
    explicit ShaderRuntime(Device& device);
    ~ShaderRuntime();
    std::shared_ptr<const CompiledMeshVertexProgram> MeshVertex(const Pica::RegsInternal& regs,
                                                             Pica::ShaderSetup& setup);
    std::shared_ptr<const CompiledFragmentShader> Fragment(const Pica::Shader::FSConfig& config,
                                                         const Pica::Shader::UserConfig& user,
                                                         FragmentColorFormat color_format,
                                                         const PicaSamplerConfigs& samplers = {},
                                                         bool gas_accumulation = false);
    std::size_t CachedFragmentCount() const;
    std::shared_ptr<const VertexCaptureProgram> Vertex(const Pica::RegsInternal& regs,
                                                     Pica::ShaderSetup& setup, bool reset_machine = false);
    std::size_t CachedVertexCount() const;
    ShaderDiskCacheStats DiskStats() const;
    MeshLookupStats MeshStats() const;
    ShaderPrewarmResult Prewarm(const std::atomic_bool& stop,
                                const std::function<void(std::size_t, std::size_t)>& progress = {});
    void Clear();

private:
    friend class RasterizerMetal;
    friend struct InputReferenceTestAccess;
    std::shared_ptr<const CompiledMeshVertexProgram> MeshVertex(const Pica::RegsInternal& regs,
                                                               const NativeProgramSnapshot& snapshot);
    struct Key {
        std::array<std::byte, sizeof(Pica::Shader::FSConfig)> config;
        std::uint32_t user;
        FragmentColorFormat color_format;
        PicaSamplerConfigs samplers;
        bool gas_accumulation;
        auto operator<=>(const Key&) const = default;
    };
    struct VertexKey {
        std::uint32_t main_offset;
        std::uint32_t output_mask;
        std::uint32_t output_total;
        std::uint32_t lighting_disable;
        std::uint32_t reset_machine;
        std::array<std::uint32_t, 7> semantics;
        Pica::ProgramCode program;
        Pica::SwizzleData swizzle;
        auto operator<=>(const VertexKey&) const = default;
    };
    struct VertexKeyView {
        std::uint32_t main_offset;
        std::uint32_t output_mask;
        std::uint32_t output_total;
        std::uint32_t lighting_disable;
        std::uint32_t reset_machine;
        std::array<std::uint32_t, 7> semantics;
        const Pica::ProgramCode& program;
        const Pica::SwizzleData& swizzle;
    };
    struct VertexKeyLess {
        static_assert(std::has_unique_object_representations_v<Pica::ProgramCode> &&
                      std::has_unique_object_representations_v<Pica::SwizzleData>);
        using is_transparent = void;
        template <typename Left, typename Right>
        static bool Equal(const Left& left, const Right& right) {
            return std::tie(left.main_offset, left.output_mask, left.output_total,
                            left.lighting_disable, left.reset_machine, left.semantics) ==
                       std::tie(right.main_offset, right.output_mask, right.output_total,
                                right.lighting_disable, right.reset_machine, right.semantics) &&
                   !std::memcmp(left.program.data(), right.program.data(), sizeof(Pica::ProgramCode)) &&
                   !std::memcmp(left.swizzle.data(), right.swizzle.data(), sizeof(Pica::SwizzleData));
        }
        template <typename Left, typename Right>
        bool operator()(const Left& left, const Right& right) const {
            const auto metadata = std::tie(left.main_offset, left.output_mask, left.output_total,
                                           left.lighting_disable, left.reset_machine, left.semantics) <=>
                                  std::tie(right.main_offset, right.output_mask, right.output_total,
                                           right.lighting_disable, right.reset_machine, right.semantics);
            if (metadata != 0) return metadata < 0;
            // Byte order only sorts this private cache. Full bank equality defines
            // identity; disk serialization and guest word values remain unchanged.
            if (const auto order = std::memcmp(left.program.data(), right.program.data(),
                                              sizeof(Pica::ProgramCode)); order != 0)
                return order < 0;
            return std::memcmp(left.swizzle.data(), right.swizzle.data(),
                               sizeof(Pica::SwizzleData)) < 0;
        }
    };
    struct OwnedMeshKey {
        std::array<std::uint32_t,12> metadata;
        const ShaderMachineProgram* address;
        std::weak_ptr<const ShaderMachineProgram> owner;
    };
    struct OwnedMeshLess {
        bool operator()(const OwnedMeshKey& left,const OwnedMeshKey& right) const {
            if(const auto order=left.metadata<=>right.metadata;order!=0)return order<0;
            if(left.address!=right.address)return std::less<const ShaderMachineProgram*>{}(left.address,right.address);
            return left.owner.owner_before(right.owner);
        }
    };
    using OwnedMeshMap=std::map<OwnedMeshKey,std::weak_ptr<const CompiledMeshVertexProgram>,OwnedMeshLess>;
    OwnedMeshMap owned_mesh;
    const OwnedMeshMap::value_type* recent_owned_mesh{};
    MeshLookupStats mesh_stats{};
    std::shared_ptr<const CompiledMeshVertexProgram> MeshVertex(const VertexKeyView& view,
                                               const std::function<TranslatedShader()>& generate);
    Device& device;
    mutable std::mutex mutex;
    std::map<Key, std::shared_ptr<const CompiledFragmentShader>> fragments;
    std::map<VertexKey, std::shared_ptr<const VertexCaptureProgram>, VertexKeyLess> vertices;
    using MeshVertexMap = std::map<VertexKey, std::shared_ptr<const CompiledMeshVertexProgram>, VertexKeyLess>;
    MeshVertexMap mesh_vertices;
    // Map nodes own immutable keys and stay stable until Clear(). A recent hit
    // still checks every metadata/code/swizzle byte, without repeated tree comparisons.
    const MeshVertexMap::value_type* recent_mesh_vertex{};
    std::shared_ptr<ShaderDiskCache> disk_cache{std::make_shared<ShaderDiskCache>()};
    TranslationSourceStore source_store;
    std::map<std::string, std::shared_ptr<const CompiledFragmentShader>> prewarmed_fragments;
    std::map<std::string, std::shared_ptr<const VertexCaptureProgram>> prewarmed_vertices;
    std::map<std::string, std::shared_ptr<const CompiledMeshVertexProgram>> prewarmed_mesh_vertices;
    std::filesystem::path prewarm_directory;
    std::uint64_t prewarm_epoch{};
    void CheckPrewarmDirectory(const std::filesystem::path& directory);
    id<MTLLibrary> CompileLibrary(std::string_view source, std::string_view name);
    std::shared_ptr<const VertexCaptureProgram> CompileVertex(TranslatedShader shader);
};

} // namespace Metal
