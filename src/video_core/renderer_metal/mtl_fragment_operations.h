// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_shader_translation.h"

namespace Metal {

enum class FragmentColorFormat { RGBA8, RGB8, RGB5A1, RGB565, RGBA4 };

struct FragmentBlendChannel {
    bool enabled{};
    std::uint8_t equation{3}; // Add/Subtract/ReverseSubtract/Min/Max = 0..4.
    std::uint8_t source_factor{1};
    std::uint8_t dest_factor{1};
    bool operator==(const FragmentBlendChannel&) const = default;
};

struct FragmentOperations {
    bool logic_enabled{};
    std::uint8_t logic_operation{3}; // Raw PICA register encoding.
    FragmentColorFormat color_format{FragmentColorFormat::RGBA8};
    FragmentBlendChannel rgb{};
    FragmentBlendChannel alpha{};
    bool blend_color_from_pica_uniform{};
    bool operator==(const FragmentOperations&) const = default;
};

// Apply after texture combiner, alpha test, fog and guest depth calculation. The upstream
// generator must have logic-op emulation disabled: these operations use native tile memory.
// No framebuffer snapshot or extra texture descriptor is used.
TranslatedShader TranslateRasterizerFragmentGLSL(std::string_view source,
                                                  const FragmentOperations& operations);

} // namespace Metal
