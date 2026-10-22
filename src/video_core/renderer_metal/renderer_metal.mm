// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "renderer_metal.h"
#include "common/file_util.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/image_interface.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/process.h"
#include "video_core/pica/pica_core.h"
#include <bit>
#include <cstring>
#include <stdexcept>

namespace Metal {

RendererMetal::RendererMetal(Core::System& system_, Pica::PicaCore& pica_, Frontend::EmuWindow& window,
                             Frontend::EmuWindow* secondary)
    : RendererBase{system_, window, secondary}, pica{pica_},
      device{std::filesystem::path(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir))/"native_metal4",
             Settings::values.use_disk_shader_cache.GetValue()},
      scheduler{device},
      transfers{device, scheduler}, compositor{device, scheduler},
      rasterizer{system.Memory(), pica, system.CustomTexManager(), *this, device, scheduler} {
    Settings::values.use_hw_shader.SetValue(true);
    for (auto& texture : fill_images) {
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                            width:1 height:1 mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        texture = device.CreateTexture(descriptor);
        transfers.ClearColor(texture, MTLClearColorMake(0, 0, 0, 1));
    }
    NotifySurfaceChanged(false);
    if (secondary) NotifySurfaceChanged(true);
    rasterizer.Textures().SetTextureDumpSink([this](const auto& params, u32 level, u64 hash, std::vector<u8> pixels) {
        DumpTexture(params, level, hash, std::move(pixels));
    });
    rasterizer.SetPresentationPrewarm([this](const auto& stop, const auto& callback) {
        compositor.Prewarm(stop, [&](std::size_t done, std::size_t total) {
            if (callback) callback(VideoCore::LoadCallbackStage::Build, done, total,
                                   "Native Metal4 presentation pipelines");
        });
    });
    // Qt constructs the renderer on the UI thread, then starts EmuThread. Complete
    // the initial clears and let the first emulation operation claim the scheduler.
    scheduler.ReleaseThread();
}

RendererMetal::~RendererMetal() {
    rasterizer.SetPresentationPrewarm({});
    rasterizer.Textures().SetTextureDumpSink({});
    // Queue failures during shutdown must not escape the noexcept destructor.
    // Feedback snapshots still own submitted resources if waiting fails.
    scheduler.FinishNoThrow();
    if (dump_worker) dump_worker->WaitForRequests();
}

void RendererMetal::NotifySurfaceChanged(bool second) {
    auto* window = second ? secondary_window : &render_window;
    id layer = window ? (__bridge id)window->GetWindowInfo().render_surface : nil;
    // Retain the new layer on the notification thread. Rebuilding and encoding remain on
    // the scheduler's owner thread; a replaced frontend layer cannot disappear meanwhile.
    std::scoped_lock lock{surface_mutex};
    pending_layers[second] = layer;
    changed_surfaces[second] = true;
}

void RendererMetal::PrepareScreens() {
    for (unsigned i = 0; i < screens.size(); ++i) {
        const unsigned fb_index = i == 2 ? 1 : 0;
        const auto& config = pica.regs.framebuffer_config[fb_index];
        const auto fill = fb_index ? pica.regs_lcd.color_fill_bottom : pica.regs_lcd.color_fill_top;
        if (fill.is_enabled) {
            const auto color = fill.AsVector();
            transfers.ClearColor(fill_images[i], MTLClearColorMake(color.x / 255.0, color.y / 255.0, color.z / 255.0, 1));
            screens[i] = {fill_images[i], {0, 0, 1, 1}};
            continue;
        }
        const PAddr address = GetActiveLCDAddress(config, i == 1);
        if (unsigned(config.color_format.Value()) > unsigned(Pica::PixelFormat::RGBA4))
            throw std::invalid_argument("Native LCD framebuffer has an invalid color format");
        const u32 bytes_per_pixel = Pica::BytesPerPixel(config.color_format);
        if (!bytes_per_pixel || config.stride % bytes_per_pixel)
            throw std::invalid_argument("Native LCD framebuffer has an invalid byte stride");
        if (!rasterizer.GetScreen(config, address, config.stride / bytes_per_pixel, screens[i])) {
            transfers.ClearColor(fill_images[i], MTLClearColorMake(0, 0, 0, 1));
            screens[i] = {fill_images[i], {0, 0, 1, 1}};
        }
    }
}

void RendererMetal::RenderTarget(id<MTLTexture> target, const Layout::FramebufferLayout& layout,
                                 const Frontend::EmuWindow& window) {
    const ScreenLayoutOptions options{Settings::values.swap_eyes_3d.GetValue(),
        Settings::values.swap_screen.GetValue(), Settings::values.filter_mode.GetValue(),
        unsigned(Settings::values.mono_render_option.GetValue())};
    const auto draws = BuildScreenDraws(layout, screens, options);
    const auto cursor = window.GetCursorInfo();
    const auto source_layout = window.GetMetalFramebufferLayoutSnapshot();
    const float sx = source_layout.bottom_screen.GetWidth(), sy = source_layout.bottom_screen.GetHeight();
    const auto vertices = BuildScreenCursor(layout, cursor.visible && sx > 0 && sy > 0,
        sx > 0 ? cursor.projected_x * layout.bottom_screen.GetWidth() / sx : 0,
        sy > 0 ? cursor.projected_y * layout.bottom_screen.GetHeight() / sy : 0);
    const std::array<id<MTLTexture>, 3> images{screens[0].texture, screens[1].texture, screens[2].texture};
    compositor.Render(target, images, draws, MTLClearColorMake(Settings::values.bg_red.GetValue(),
        Settings::values.bg_green.GetValue(), Settings::values.bg_blue.GetValue(), 1), vertices);
}

void RendererMetal::RenderWindow(Frontend::EmuWindow& window, unsigned index) {
    id layer = nil;
    bool changed{};
    {
        std::scoped_lock lock{surface_mutex};
        changed = std::exchange(changed_surfaces[index], false);
        if (changed) layer = pending_layers[index];
    }
    if (changed) {
        surfaces[index].reset();
        if (layer) surfaces[index] = std::make_unique<PresentSurface>(device, (__bridge void*)layer);
    }
    if (!surfaces[index]) return;
    const auto layout = window.GetMetalFramebufferLayoutSnapshot();
    surfaces[index]->SetSize(layout.width, layout.height);
    surfaces[index]->SetVSync(Settings::values.use_vsync.GetValue());
    scheduler.SubmitPendingBeforeDrawable(!surfaces[index]->HasReady());
    const auto acquire_profile = scheduler.BeginProfileDrawableAcquire();
    // WindowServer can hold nextDrawable for a refresh or a native timeout.
    // Continue every guest GPU frame while the single bounded producer waits;
    // compose the latest screens once its native drawable becomes available.
    const auto drawable = surfaces[index]->TryAcquire();
    scheduler.EndProfileDrawableAcquire(acquire_profile, index, drawable != nil);
    if (!drawable) return;
    RenderTarget(drawable.texture, layout, window);
    // At normal emulation speed the LCD produces about 59.83 refreshes/sec.
    // Ask Core Animation to space presentations across 60 Hz display periods,
    // rather than allowing bursts on consecutive 120 Hz ProMotion refreshes.
    // Guest GPU progress and fast-forward remain independent of this request.
    const double minimum_duration = Settings::values.use_vsync.GetValue() &&
        Settings::GetFrameLimit() == 100.0 ? 1.0 / 60.0 : 0.0;
    scheduler.Submit(drawable, minimum_duration);
}

void RendererMetal::RenderScreenshot() {
    if (!settings.screenshot_requested.exchange(false)) return;
    const auto layout = settings.screenshot_framebuffer_layout;
    if (!layout.width || !layout.height || !settings.screenshot_bits)
        throw std::invalid_argument("Native screenshot has no destination or dimensions");
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                        width:layout.width height:layout.height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget;
    const auto target = device.CreateTexture(descriptor);
    RenderTarget(target, layout, render_window);
    // The frontend's QImage::Format_RGB32 expects BGRA byte order and top-down rows.
    const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, layout.width, layout.height)},
                                          NSUInteger(layout.width) * 4);
    std::memcpy(settings.screenshot_bits, pixels.data(), pixels.size());
    if (settings.screenshot_complete_callback) settings.screenshot_complete_callback(false);
}

void RendererMetal::SwapBuffers() {
    @autoreleasepool {
        system.perf_stats->StartSwap();
        {
            Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Present};
            PrepareScreens();
            RenderScreenshot();
            RenderWindow(render_window, 0);
            if (secondary_window && Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
                RenderWindow(*secondary_window, 1);
                secondary_window->PollEvents();
            }
            // Commit guest draws even if the window is hidden or its layer cannot supply a drawable.
            scheduler.SubmitPending();
            rasterizer.TickFrame();
            system.perf_stats->EndSwap();
            settings.bg_color_update_requested = false;
            settings.shader_update_requested = false;
        }
        scheduler.EndProfileFrame();
        EndFrame();
    }
}

void RendererMetal::DumpTexture(const VideoCore::SurfaceParams& params, u32 level, u64 hash,
                                std::vector<u8> pixels) {
    if (!std::has_single_bit(params.width) || !std::has_single_bit(params.height)) return;
    if (pixels.size() != std::uint64_t(params.width) * params.height * 4)
        throw std::invalid_argument("Native PNG dump is not a complete GPU-decoded RGBA8 image");
    const auto process = system.Kernel().GetCurrentProcess();
    const auto image = system.GetImageInterface();
    if (!process || !process->codeset || !image) return;
    const auto directory = fmt::format("{}textures/{:016X}/", FileUtil::GetUserPath(FileUtil::UserPath::DumpDir),
                                       process->codeset->program_id);
    const auto path = fmt::format("{}tex1_{}x{}_{:016X}_{}_mip{}.png", directory, params.width, params.height,
                                  hash, unsigned(params.pixel_format), level);
    if (dumped_paths.contains(path) || FileUtil::Exists(path) || !FileUtil::CreateFullPath(directory)) return;
    if (!dump_worker) dump_worker = std::make_unique<Common::ThreadWorker>(1, "Metal PNG export");
    dump_worker->QueueWork([image, path, width = params.width, height = params.height, pixels = std::move(pixels)] {
        // PNG compression is a file-format operation. Guest decoding and row flipping have
        // already run on Metal; this worker never captures renderer or guest-memory pointers.
        if (!image->EncodePNG(path, width, height, pixels)) LOG_ERROR(Render, "Native Metal PNG export failed: {}", path);
    });
    dumped_paths.insert(path);
}

} // namespace Metal
