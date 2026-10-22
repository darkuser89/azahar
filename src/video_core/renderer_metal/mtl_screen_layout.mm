// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_screen_layout.h"
#include "video_core/pica/regs_external.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Metal {

PAddr GetActiveLCDAddress(const Pica::FramebufferConfig& config, bool right_eye) {
    const PAddr left = config.active_fb == 0 ? config.address_left1 : config.address_left2;
    const PAddr right = config.active_fb == 0 ? config.address_right1 : config.address_right2;
    return right_eye && right ? right : left;
}

PresentDraw MakeScreenDraw(const Layout::FramebufferLayout& layout,
                           const std::array<ScreenImage, 3>& screens,
                           const Common::Rectangle<float>& rect, Layout::DisplayOrientation orientation,
                           unsigned left, unsigned right, PresentMode mode, bool reverse,
                           float opacity, bool linear_filter) {
    if (!layout.width || !layout.height || left >= screens.size() || right >= screens.size() ||
        !screens[left].texture || !screens[right].texture || !std::isfinite(opacity) || opacity < 0 || opacity > 1 ||
        !std::isfinite(rect.left) || !std::isfinite(rect.top) || !std::isfinite(rect.right) || !std::isfinite(rect.bottom) ||
        rect.right <= rect.left || rect.bottom <= rect.top) {
        throw std::invalid_argument("Invalid native screen layout or image");
    }
    const auto uv = screens[left].texcoords;
    std::array<std::array<float, 2>, 4> coords;
    float w = rect.right - rect.left, h = rect.bottom - rect.top;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        coords = {{{uv.bottom, uv.left}, {uv.bottom, uv.right}, {uv.top, uv.left}, {uv.top, uv.right}}}; break;
    case Layout::DisplayOrientation::Portrait:
        coords = {{{uv.bottom, uv.right}, {uv.top, uv.right}, {uv.bottom, uv.left}, {uv.top, uv.left}}};
        std::swap(w, h); break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        coords = {{{uv.top, uv.right}, {uv.top, uv.left}, {uv.bottom, uv.right}, {uv.bottom, uv.left}}}; break;
    case Layout::DisplayOrientation::PortraitFlipped:
        coords = {{{uv.top, uv.left}, {uv.bottom, uv.left}, {uv.top, uv.right}, {uv.bottom, uv.right}}};
        std::swap(w, h); break;
    default: throw std::invalid_argument("Invalid native screen orientation");
    }
    PresentDraw draw{};
    draw.vertices = {{{{rect.left, rect.top}, coords[0]}, {{rect.right, rect.top}, coords[1]},
                      {{rect.left, rect.bottom}, coords[2]}, {{rect.right, rect.bottom}, coords[3]}}};
    // MSL multiplies the row vector by this column-major storage, matching the guest layout.
    draw.uniforms.modelview = {2.f / layout.width, 0, 0, -1, 0, 2.f / layout.height, 0, -1,
                              0, 0, 1, 0, 0, 0, 0, 1};
    const float iw = screens[left].texture.width, ih = screens[left].texture.height;
    draw.uniforms.input_resolution = {iw, ih, 1.f / iw, 1.f / ih};
    draw.uniforms.output_resolution = {h, w, 1.f / h, 1.f / w};
    draw.uniforms.left_screen = left;
    draw.uniforms.right_screen = right;
    draw.uniforms.mode = mode;
    draw.uniforms.reverse_interlaced = reverse;
    const auto right_uv = screens[right].texcoords;
    const float du = uv.top - uv.bottom, dv = uv.right - uv.left;
    if (!std::isfinite(du) || !std::isfinite(dv) || !du || !dv)
        throw std::invalid_argument("Invalid native screen texture coordinates");
    const float su = (right_uv.top - right_uv.bottom) / du, sv = (right_uv.right - right_uv.left) / dv;
    draw.uniforms.right_texcoord_transform = {su, sv, right_uv.bottom - uv.bottom * su, right_uv.left - uv.left * sv};
    for (const auto value : draw.uniforms.right_texcoord_transform)
        if (!std::isfinite(value)) throw std::invalid_argument("Invalid native stereo texture coordinate transform");
    draw.opacity = opacity;
    draw.linear_filter = linear_filter;
    return draw;
}

std::vector<PresentDraw> BuildScreenDraws(const Layout::FramebufferLayout& layout,
                                        const std::array<ScreenImage, 3>& screens,
                                        const ScreenLayoutOptions& options) {
    if (!layout.width || !layout.height) return {};
    if (options.mono_eye > 1) throw std::invalid_argument("Invalid native mono eye");
    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape : Layout::DisplayOrientation::Portrait;
    std::vector<PresentDraw> result;
    const auto add_screen = [&](bool bottom, const Common::Rectangle<u32>& rectangle, float opacity) {
        if (rectangle.right <= rectangle.left || rectangle.bottom <= rectangle.top) return;
        const unsigned left = bottom ? 2 : options.swap_eyes ? 1 : 0;
        const unsigned right = bottom ? 2 : options.swap_eyes ? 0 : 1;
        Common::Rectangle<float> rect{float(rectangle.left), float(rectangle.top), float(rectangle.right), float(rectangle.bottom)};
        const auto draw = [&](const auto& area, unsigned l, unsigned r, PresentMode mode = PresentMode::Single, bool reverse = false) {
            result.push_back(MakeScreenDraw(layout, screens, area, orientation, l, r, mode, reverse, opacity, options.linear_filter));
        };
        using Stereo = Settings::StereoRenderOption;
        switch (layout.render_3d_mode) {
        case Stereo::Off: draw(rect, bottom ? 2 : options.mono_eye, bottom ? 2 : options.mono_eye); break;
        case Stereo::SideBySide:
            rect.left /= 2; rect.right /= 2;
            draw(rect, left, left);
            rect.left += layout.width / 2; rect.right += layout.width / 2;
            draw(rect, right, right); break;
        case Stereo::SideBySideFull:
            draw(rect, left, left);
            rect.left += layout.width / 2; rect.right += layout.width / 2;
            draw(rect, right, right); break;
        case Stereo::CardboardVR: {
            draw(rect, left, left);
            const float x = (bottom ? layout.cardboard.bottom_screen_right_eye : layout.cardboard.top_screen_right_eye) + layout.width / 2;
            rect.right += x - rect.left; rect.left = x;
            draw(rect, right, right); break;
        }
        case Stereo::Anaglyph: draw(rect, left, right, PresentMode::Anaglyph); break;
        case Stereo::Interlaced: draw(rect, left, right, PresentMode::Interlaced); break;
        case Stereo::ReverseInterlaced: draw(rect, left, right, PresentMode::Interlaced, true); break;
        default: throw std::invalid_argument("Invalid native stereo mode");
        }
    };
    const auto top = [&] { if (layout.top_screen_enabled) add_screen(false, layout.top_screen, layout.top_opacity); };
    const auto bottom = [&] { if (layout.bottom_screen_enabled) add_screen(true, layout.bottom_screen, layout.bottom_opacity); };
    if (options.swap_screen_order) { bottom(); top(); } else { top(); bottom(); }
    if (layout.additional_screen_enabled)
        add_screen(layout.additional_screen_is_bottom, layout.additional_screen,
                   layout.additional_screen_is_bottom ? layout.bottom_opacity : layout.top_opacity);
    return result;
}

std::vector<std::array<float, 2>> BuildScreenCursor(const Layout::FramebufferLayout& layout,
                                                  bool visible, float local_x, float local_y) {
    if (!visible || !layout.bottom_screen_enabled || !layout.width || !layout.height ||
        !std::isfinite(local_x) || !std::isfinite(local_y)) return {};
    const auto rect = layout.bottom_screen;
    if (rect.right <= rect.left || rect.bottom <= rect.top) return {};
    const float x = rect.left + local_x, y = rect.top + local_y;
    const float radius = (rect.bottom - rect.top) / 60.f;
    std::vector<std::array<float, 2>> vertices;
    const auto bar = [&](float l, float t, float r, float b) {
        l = std::max(l, float(rect.left)); r = std::min(r, float(rect.right));
        t = std::max(t, float(rect.top)); b = std::min(b, float(rect.bottom));
        if (r <= l || b <= t) return;
        const auto ndc = [&](float px, float py) { return std::array{2.f * px / layout.width - 1, 2.f * py / layout.height - 1}; };
        vertices.insert(vertices.end(), {ndc(l, t), ndc(r, t), ndc(r, b), ndc(l, t), ndc(r, b), ndc(l, b)});
    };
    bar(x - radius / 5, y - radius, x + radius / 5, y + radius);
    bar(x - radius, y - radius / 5, x + radius, y + radius / 5);
    return vertices;
}

} // namespace Metal
