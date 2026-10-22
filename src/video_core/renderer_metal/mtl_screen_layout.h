// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_present.h"
#include "core/frontend/framebuffer_layout.h"
#include <vector>

namespace Pica { struct FramebufferConfig; }

namespace Metal {

struct ScreenImage {
    id<MTLTexture> texture{};
    // The LCD memory is rotated relative to the physical display, as in the PICA cache.
    Common::Rectangle<float> texcoords{};
};

struct ScreenLayoutOptions {
    bool swap_eyes{};
    bool swap_screen_order{};
    bool linear_filter{};
    unsigned mono_eye{};
};

PAddr GetActiveLCDAddress(const Pica::FramebufferConfig& config, bool right_eye);

PresentDraw MakeScreenDraw(const Layout::FramebufferLayout& layout,
                           const std::array<ScreenImage, 3>& screens,
                           const Common::Rectangle<float>& rect, Layout::DisplayOrientation orientation,
                           unsigned left, unsigned right, PresentMode mode, bool reverse,
                           float opacity, bool linear_filter);
std::vector<PresentDraw> BuildScreenDraws(const Layout::FramebufferLayout& layout,
                                        const std::array<ScreenImage, 3>& screens,
                                        const ScreenLayoutOptions& options);
std::vector<std::array<float, 2>> BuildScreenCursor(const Layout::FramebufferLayout& layout,
                                                  bool visible, float local_x, float local_y);

} // namespace Metal
