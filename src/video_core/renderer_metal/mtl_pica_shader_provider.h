// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_fragment_operations.h"
#include "video_core/shader/generator/pica_fs_config.h"
#include "video_core/pica/shader_setup.h"

namespace Metal {

struct PicaSamplerConfig {
    std::uint32_t wrap_s{}, wrap_t{};
    std::uint32_t min_filter{}, mag_filter{}, mip_filter{};
    std::uint32_t lod_min{}, lod_max{};
    auto operator<=>(const PicaSamplerConfig&) const = default;
};
using PicaSamplerConfigs = std::array<PicaSamplerConfig, 3>;
PicaSamplerConfigs BuildPicaSamplerConfigs(const Pica::RegsInternal& regs);

struct PicaFragmentProgram {
    TranslatedShader shader;
    FragmentOperations operations;
};

// Reuse the PICA generator as a CPU source generator, then compile for native Metal execution.
// Its Vulkan profile flag selects source bindings only; no Vulkan runtime is linked or called.
PicaFragmentProgram GeneratePicaFragmentShader(const Pica::Shader::FSConfig& config,
                                              const Pica::Shader::UserConfig& user,
                                              FragmentColorFormat color_format,
                                              const PicaSamplerConfigs& samplers = {},
                                              bool gas_accumulation = false);
TranslatedShader GeneratePicaVertexShader(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup, bool reset_machine = false);
TranslatedShader GeneratePicaVertexMeshShader(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup);

// Reconstruct canonical MSL/reflection from stored guest IR without a native
// compiler call or another GLSL translation. Callers compare the entire result
// before exposing cached metadata; native completion is still validated later.
TranslatedShader RestorePicaVertexShader(std::span<const std::uint32_t> words,
                                         ShaderStage stage, bool reset_machine);
PicaFragmentProgram RestorePicaFragmentShader(std::span<const std::uint32_t> words,
    const Pica::Shader::FSConfig& config, const Pica::Shader::UserConfig& user,
    FragmentColorFormat color_format, bool gas_accumulation);

} // namespace Metal
