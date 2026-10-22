// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Metal {

enum class ShaderStage { Vertex, Fragment, Compute, VertexCompute, VertexMesh };
enum class ShaderResourceKind { UniformBuffer, StorageBuffer, Texture, StorageTexture, Sampler };

inline constexpr auto NoShaderBinding = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t ShaderBufferSizesBinding = 30;

// Compute capture writes eight vec4 fields, avoiding ABI differences in packed vertex attributes.
struct alignas(16) CapturedVertex {
    std::array<float, 4> position;
    std::array<float, 4> color;
    std::array<float, 4> texcoord0;
    std::array<float, 4> texcoord1;
    std::array<float, 4> texcoord2;
    std::array<float, 4> quaternion;
    std::array<float, 4> view;
    std::array<float, 4> clip; // Clip distances in xy, texture coordinate 0 w in z.
};
static_assert(sizeof(CapturedVertex) == 128);

struct ShaderBindingRemap {
    std::uint32_t descriptor_set;
    std::uint32_t binding;
    std::uint32_t buffer{NoShaderBinding};
    std::uint32_t texture{NoShaderBinding};
    std::uint32_t sampler{NoShaderBinding};
};

struct ShaderMemberLayout {
    std::string name;
    std::uint32_t offset;
    std::uint32_t size;
    bool operator==(const ShaderMemberLayout&) const = default;
};

struct ShaderResource {
    std::string name;
    ShaderResourceKind kind;
    std::uint32_t descriptor_set;
    std::uint32_t binding;
    std::uint32_t metal_index;
    std::uint32_t sampler_index{NoShaderBinding};
    std::uint32_t count{1};
    std::uint32_t byte_size{};
    std::uint32_t runtime_array_stride{};
    std::vector<ShaderMemberLayout> members;
    bool active{true}; // Declared resources remain in ABI reflection even when optimized away.
    bool operator==(const ShaderResource&) const = default;
};

struct TranslatedShader {
    ShaderStage stage;
    std::string source;
    std::string entry_point;
    std::vector<ShaderResource> resources;
    bool needs_buffer_sizes{};
    bool uses_framebuffer_fetch{};
    // Retain the guest compiler IR alongside the final MSL for the native disk cache.
    std::vector<std::uint32_t> spirv{};
    bool operator==(const TranslatedShader&) const = default;
};

// Compute once when publishing an immutable PICA program, including disk loads.
inline std::uint8_t PicaUniformMask(const TranslatedShader& shader) {
    constexpr std::array<std::uint32_t, 4> slots{0, 1, 2, 23};
    std::uint8_t mask{};
    for (const auto& resource : shader.resources) {
        if (!resource.active || (resource.kind != ShaderResourceKind::UniformBuffer &&
                                 resource.kind != ShaderResourceKind::StorageBuffer)) continue;
        for (unsigned i = 0; i < slots.size(); ++i)
            if (resource.metal_index <= slots[i] && slots[i] - resource.metal_index < resource.count)
                mask |= 1u << i;
    }
    return mask;
}

// Both sampled and storage images are consumers, including shadow writes.
// Cache this with the immutable program rather than scanning reflection per draw.
inline std::bitset<128> PicaTextureMask(const TranslatedShader& shader) {
    std::bitset<128> mask;
    for (const auto& resource : shader.resources) {
        if (!resource.active || (resource.kind != ShaderResourceKind::Texture &&
                                 resource.kind != ShaderResourceKind::StorageTexture)) continue;
        for (unsigned index = 0; index < mask.size(); ++index)
            if (resource.metal_index <= index && index - resource.metal_index < resource.count)
                mask.set(index);
    }
    return mask;
}

// These are CPU-only operations. No Vulkan device, loader or GPU command is involved.
std::vector<std::uint32_t> CompileGuestGLSL(std::string_view source, ShaderStage stage);
TranslatedShader TranslateGuestSPIRV(std::span<const std::uint32_t> words, ShaderStage stage,
                                     std::span<const ShaderBindingRemap> bindings);
TranslatedShader TranslateGuestGLSL(std::string_view source, ShaderStage stage,
                                    std::span<const ShaderBindingRemap> bindings);
std::span<const ShaderBindingRemap> RasterizerShaderBindings();

// Stack-scoped and thread-local. Records only successful full GLSL inputs,
// including cache hits; no CPU compiler output or borrowed storage escapes.
class TranslationInputScope final {
public:
    using Recorder=std::function<void(std::string_view,ShaderStage,std::span<const ShaderBindingRemap>)>;
    explicit TranslationInputScope(Recorder recorder = {});
    ~TranslationInputScope();
    TranslationInputScope(const TranslationInputScope&)=delete;
    TranslationInputScope& operator=(const TranslationInputScope&)=delete;
private:
    friend TranslatedShader TranslateGuestGLSL(std::string_view,ShaderStage,std::span<const ShaderBindingRemap>);
    Recorder recorder;
    TranslationInputScope* previous;
};

} // namespace Metal
