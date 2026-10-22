// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_fragment_operations.h"
#include "mtl_shader_text.h"
#include <array>
#include <regex>
#include <stdexcept>

namespace Metal {

namespace {

std::array<unsigned, 4> ChannelMasks(FragmentColorFormat format) {
    switch (format) {
    case FragmentColorFormat::RGBA8: return {255, 255, 255, 255};
    case FragmentColorFormat::RGB8: return {255, 255, 255, 0};
    case FragmentColorFormat::RGB5A1: return {31, 31, 31, 1};
    case FragmentColorFormat::RGB565: return {31, 63, 31, 0};
    case FragmentColorFormat::RGBA4: return {15, 15, 15, 15};
    }
    throw std::invalid_argument("Invalid PICA fragment color format");
}

bool UsesConstant(const FragmentBlendChannel& channel) {
    const auto constant = [](unsigned factor) { return factor >= 10 && factor <= 13; };
    return channel.enabled && (constant(channel.source_factor) || constant(channel.dest_factor));
}

std::string Factor(unsigned factor, std::string_view blend_color) {
    constexpr std::array factors{
        "vec4(0.0)", "vec4(1.0)", "azahar_native_s", "vec4(1.0) - azahar_native_s",
        "azahar_native_d", "vec4(1.0) - azahar_native_d", "azahar_native_s.aaaa",
        "vec4(1.0) - azahar_native_s.aaaa", "azahar_native_d.aaaa",
        "vec4(1.0) - azahar_native_d.aaaa",
    };
    if (factor < factors.size()) return factors[factor];
    switch (factor) {
    case 10: return std::string{blend_color};
    case 11: return "vec4(1.0) - " + std::string{blend_color};
    case 12: return std::string{blend_color} + ".aaaa";
    case 13: return "vec4(1.0) - " + std::string{blend_color} + ".aaaa";
    case 14: return "vec4(vec3(min(azahar_native_s.a, 1.0 - azahar_native_d.a)), 1.0)";
    default: return "vec4(1.0)"; // Undefined factors match the existing hardware-tested fallback.
    }
}

} // namespace

TranslatedShader TranslateRasterizerFragmentGLSL(std::string_view source,
                                                  const FragmentOperations& operations) {
    if (operations.logic_operation >= 16) {
        throw std::invalid_argument("Invalid PICA logic operation");
    }
    const auto masks = ChannelMasks(operations.color_format);
    const bool blending = operations.rgb.enabled || operations.alpha.enabled;
    if (operations.logic_enabled && blending) {
        throw std::invalid_argument("Invalid or mutually exclusive native fragment operations");
    }
    if ((!operations.logic_enabled && !blending) ||
        (operations.logic_enabled && operations.logic_operation == 6)) {
        // NoOp is enforced by the render pipeline's zero color mask, including depth-only
        // passes. Such a pass has no color attachment available for framebuffer fetch.
        return TranslateGuestGLSL(source, ShaderStage::Fragment, RasterizerShaderBindings());
    }
    if (source.size() > 8 * 1024 * 1024 || source.find("azahar_native_") != std::string_view::npos) {
        throw std::invalid_argument("Invalid source for native fragment operations");
    }
    std::string text{source};
    std::cmatch output;
    static const std::regex output_pattern{
        R"(layout\s*\(\s*location\s*=\s*0\s*\)\s*out\s+vec4\s+([A-Za-z0-9_]+)\s*;)"};
    ShaderTextMatcher outputs{text, output_pattern, {"layout"}};
    std::size_t position{};
    if (!outputs.Next(position, output)) {
        throw std::invalid_argument("Native fragment operations require one vec4 color output at location zero");
    }
    const auto color = output[1].str();
    if (outputs.Next(position, output)) {
        throw std::invalid_argument("Native fragment operations require one vec4 color output at location zero");
    }
    static const std::regex main_pattern{R"(void\s+main\s*\(\s*\))"};
    ShaderTextMatcher mains{text, main_pattern, {"void"}};
    std::cmatch main;
    if (!mains.Next(position, main) || mains.Next(position, main)) {
        throw std::invalid_argument("Native fragment operations require one main function");
    }
    text = ReplaceShaderText(text, main_pattern, {"void"}, "void azahar_native_guest_fragment()");
    // Appending declarations leaves the generator's leading version/extension directives intact.
    const bool needs_source = operations.logic_operation != 0 && operations.logic_operation != 4 &&
                              operations.logic_operation != 6 && operations.logic_operation != 7;
    const bool needs_dest = blending || (operations.logic_operation != 0 && operations.logic_operation != 3 &&
                                      operations.logic_operation != 4 && operations.logic_operation != 5);
    if (needs_dest) {
        text += "\nlayout(input_attachment_index=0, set=4, binding=0) uniform subpassInput azahar_native_destination;\n";
    }
    const bool constants = UsesConstant(operations.rgb) || (masks[3] && UsesConstant(operations.alpha));
    if (constants && !operations.blend_color_from_pica_uniform) {
        text += "layout(set=3, binding=4, std140) uniform AzaharNativeBlendConstants { vec4 azahar_native_blend_color; };\n";
    }
    text += R"(
void main() {
    azahar_native_guest_fragment();
)";
    if (blending) {
        text += "vec4 azahar_native_s = " + color + ";\n";
        text += "vec4 azahar_native_d = subpassLoad(azahar_native_destination);\n";
        for (unsigned c = 0; c < 4; ++c) {
            const auto component = std::string(1, "rgba"[c]);
            const auto scale = std::to_string(masks[c]) + ".0";
            text += "azahar_native_d." + component + " = " + (masks[c]
                ? "round(azahar_native_d." + component + " * " + scale + ") / " + scale
                : "1.0") + ";\n";
        }
        const auto blend_color = operations.blend_color_from_pica_uniform ? "blend_color"
                                                                         : "azahar_native_blend_color";
        const auto apply = [&](const FragmentBlendChannel& channel, std::string_view components) {
            if (!channel.enabled) return;
            const auto part = std::string{components};
            const auto s = "azahar_native_s." + part + " * (" + Factor(channel.source_factor, blend_color) + ")." + part;
            const auto d = "azahar_native_d." + part + " * (" + Factor(channel.dest_factor, blend_color) + ")." + part;
            std::string value;
            switch (channel.equation) {
            case 1: value = "(" + s + ") - (" + d + ")"; break;
            case 2: value = "(" + d + ") - (" + s + ")"; break;
            case 3: value = "min(" + s + ", " + d + ")"; break;
            case 4: value = "max(" + s + ", " + d + ")"; break;
            default: value = "(" + s + ") + (" + d + ")"; break;
            }
            text += color + "." + part + " = " + value + ";\n";
        };
        apply(operations.rgb, "rgb");
        // RGB guest targets always store opaque alpha. Do not generate a dead
        // alpha expression (or its otherwise unused constant-buffer dependency).
        if (masks[3]) apply(operations.alpha, "a");
        // Quantize before storing to canonical RGBA8, including Add/Subtract channels.
        // Otherwise subsequent primitives would blend with host rather than guest precision.
        for (unsigned c = 0; c < 4; ++c) {
            const auto component = std::string(1, "rgba"[c]);
            if (masks[c]) {
                const auto scale = std::to_string(masks[c]) + ".0";
                text += color + "." + component + " = round(clamp(" + color + "." + component +
                        ", 0.0, 1.0) * " + scale + ") / " + scale + ";\n";
            } else {
                text += color + "." + component + " = 1.0;\n";
            }
        }
        text += "}\n";
        return TranslateGuestGLSL(text, ShaderStage::Fragment, RasterizerShaderBindings());
    }
    std::string mask = "uvec4(";
    std::string scale = "vec4(";
    for (unsigned i = 0; i < 4; ++i) {
        if (i) { mask += ", "; scale += ", "; }
        mask += std::to_string(masks[i]) + "u";
        scale += std::to_string(masks[i] ? masks[i] : 1) + ".0";
    }
    mask += ")";
    scale += ")";
    text += "const uvec4 azahar_native_mask = " + mask + ";\n";
    text += "const vec4 azahar_native_scale = " + scale + ";\n";
    if (needs_source) {
        text += "uvec4 s = uvec4(round(clamp(" + color +
                ", 0.0, 1.0) * azahar_native_scale)) & azahar_native_mask;\n";
    }
    if (needs_dest) {
        text += "uvec4 d = uvec4(round(clamp(subpassLoad(azahar_native_destination), 0.0, 1.0) * "
                "azahar_native_scale)) & azahar_native_mask;\n";
    }
    constexpr std::array expressions{
        "uvec4(0)", "s & d", "s & ~d", "s", "azahar_native_mask", "~s", "d", "~d",
        "~(s & d)", "s | d", "~(s | d)", "s ^ d", "~(s ^ d)", "~s & d", "s | ~d", "~s | d",
    };
    text += color + " = vec4((" + expressions[operations.logic_operation] +
            ") & azahar_native_mask) / azahar_native_scale;\n";
    if (!masks[3]) {
        text += color + ".a = 1.0;\n";
    }
    text += "}\n";
    return TranslateGuestGLSL(text, ShaderStage::Fragment, RasterizerShaderBindings());
}

} // namespace Metal
