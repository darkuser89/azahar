// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_rasterizer.h"
#include "video_core/renderer_base.h"
#include <unordered_set>

namespace Common { template <class> class StatefulThreadWorker; }

namespace Metal {

class RendererMetal final : public VideoCore::RendererBase {
public:
    RendererMetal(Core::System& system, Pica::PicaCore& pica, Frontend::EmuWindow& window,
                  Frontend::EmuWindow* secondary_window);
    ~RendererMetal() override;
    VideoCore::RasterizerInterface* Rasterizer() override { return &rasterizer; }
    void SwapBuffers() override;
    // Metal presents from SwapBuffers on the scheduler's owner thread.
    void TryPresent(int, bool) override {}
    void NotifySurfaceChanged(bool second) override;

private:
    void PrepareScreens();
    void RenderWindow(Frontend::EmuWindow& window, unsigned index);
    void RenderTarget(id<MTLTexture> target, const Layout::FramebufferLayout& layout,
                      const Frontend::EmuWindow& window);
    void RenderScreenshot();
    void DumpTexture(const VideoCore::SurfaceParams& params, u32 level, u64 hash, std::vector<u8> pixels);

    Pica::PicaCore& pica;
    Device device;
    Scheduler scheduler;
    TransferRuntime transfers;
    Compositor compositor;
    RasterizerMetal rasterizer;
    std::array<ScreenImage, 3> screens{};
    std::array<id<MTLTexture>, 3> fill_images{};
    std::array<std::unique_ptr<PresentSurface>, 2> surfaces;
    std::mutex surface_mutex;
    std::array<id, 2> pending_layers{};
    std::array<bool, 2> changed_surfaces{};
    std::unique_ptr<Common::StatefulThreadWorker<void>> dump_worker;
    std::unordered_set<std::string> dumped_paths;
};

} // namespace Metal
