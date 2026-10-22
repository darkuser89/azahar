// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_screen_layout.h"
#include "mtl_compiler_gate_fixture.h"
#include "mtl_transfer.h"
#include "video_core/pica/regs_external.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <cstring>
#include <iostream>
#include <stdexcept>

// A controllable acquisition boundary tests the real PresentSurface handoff.
// Native Metal textures still identify the requested device/size. No fake object
// is submitted to a GPU queue; the pixel tests below exercise actual MTL4 draws.
struct SurfaceGate {
    std::mutex mutex;
    std::condition_variable condition;
    unsigned allowed{}, nil_at{}, fail_at{};
    std::atomic_uint calls{}, produced{};
    std::vector<CGSize> sizes;
    std::vector<bool> vsync;
    std::vector<std::thread::id> threads;
    std::atomic_uint alive{};
    void Wait(unsigned count, bool complete = false) {
        std::unique_lock lock{mutex};
        if (!condition.wait_for(lock, std::chrono::seconds{5}, [&] {
            return (complete ? produced : calls) >= count;
        })) throw std::runtime_error("Timed out at controlled drawable boundary");
    }
    void Allow(unsigned count) {
        { std::scoped_lock lock{mutex}; allowed = std::max(allowed, count); }
        condition.notify_all();
    }
};

@interface SurfaceTestDrawable : NSObject {
@public
    id<MTLTexture> image;
    unsigned identity;
    std::shared_ptr<SurfaceGate> gate;
}
@end
@implementation SurfaceTestDrawable
- (id<MTLTexture>)texture { return image; }
- (void)dealloc { if (gate) --gate->alive; }
@end

@interface SurfaceTestLayer : CAMetalLayer {
@public
    std::shared_ptr<SurfaceGate> gate;
}
@end
@implementation SurfaceTestLayer
- (id<CAMetalDrawable>)nextDrawable {
    const auto size = self.drawableSize;
    const bool vsync = self.displaySyncEnabled;
    unsigned identity;
    {
        std::unique_lock lock{gate->mutex};
        identity = ++gate->calls;
        gate->sizes.push_back(size);
        gate->vsync.push_back(vsync);
        gate->threads.push_back(std::this_thread::get_id());
        gate->condition.notify_all();
        if (!gate->condition.wait_for(lock, std::chrono::seconds{5}, [&] {
            return gate->allowed >= identity;
        })) throw std::runtime_error("Controlled nextDrawable was not released");
    }
    if (identity == gate->fail_at) throw std::runtime_error("Injected native acquisition failure");
    id<CAMetalDrawable> value = nil;
    if (identity != gate->nil_at) {
        auto* frame = [[SurfaceTestDrawable alloc] init];
        frame->gate = gate; ++gate->alive; frame->identity = identity;
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
            width:NSUInteger(size.width) height:NSUInteger(size.height) mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget;
        frame->image = [self.device newTextureWithDescriptor:descriptor];
        value = (id<CAMetalDrawable>)frame;
    }
    {
        std::scoped_lock lock{gate->mutex};
        gate->produced = identity;
    }
    gate->condition.notify_all();
    return value;
}
@end

namespace {

using Color = std::array<unsigned, 4>;
constexpr Color Red{255, 0, 0, 255}, Green{0, 255, 0, 255}, Blue{0, 0, 255, 255};
constexpr Color Yellow{255, 255, 0, 255}, Poison{255, 0, 255, 255}, Background{17, 34, 51, 255};

void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void ValidateSurfaceHandoff(Metal::Device& device) {
    const auto acquire = [](Metal::PresentSurface& surface, const auto& gate, const auto& release) {
        std::exception_ptr failure;
        std::jthread helper{[&] {
            try { release(); }
            catch (...) { failure = std::current_exception(); gate->Allow(UINT_MAX); }
        }};
        id<CAMetalDrawable> result = surface.Acquire();
        helper.join();
        if (failure) std::rethrow_exception(failure);
        return result;
    };
    const auto identity = [](id<CAMetalDrawable> drawable) {
        return ((__bridge SurfaceTestDrawable*)(__bridge void*)drawable)->identity;
    };
    const auto gate = std::make_shared<SurfaceGate>();
    @autoreleasepool {
        auto* layer = [SurfaceTestLayer layer]; layer->gate = gate;
        {
            Metal::PresentSurface surface{device, (__bridge void*)layer};
            surface.SetSize(48, 32);
            Check(!surface.HasReady(), "Unstarted surface fabricated a prepared drawable");
            auto first = acquire(surface, gate, [&] { gate->Wait(1); gate->Allow(1); });
            Check(identity(first) == 1, "Drawable handoff reordered its first image");
            gate->Wait(2);
            Check(!surface.HasReady(), "Blocked native acquisition was reported ready");
            Check(gate->threads[0] != std::this_thread::get_id(),
                  "Native acquisition still ran on the emulation/consumer thread");
            // Complete the speculative next acquisition WITHOUT a second Acquire.
            gate->Allow(2); gate->Wait(2, true);
            const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
            while (!surface.HasReady() && std::chrono::steady_clock::now() < ready_deadline)
                std::this_thread::yield();
            Check(surface.HasReady(), "Completed speculative drawable was not published as ready");
            Check(gate->calls == 2, "Drawable producer exceeded its one-image bound");
            auto second = surface.Acquire();
            Check(identity(second) == 2, "Prepared drawable was skipped or replaced");
            gate->Wait(3);
            surface.SetSize(32, 16);
            Check(!surface.HasReady(), "Size change retained stale readiness");
            auto resized = acquire(surface, gate, [&] {
                gate->Allow(3); gate->Wait(4);
                Check(gate->sizes[3].width == 32 && gate->sizes[3].height == 16,
                      "Resized native acquisition used stale dimensions");
                gate->Allow(4);
            });
            Check(identity(resized) == 4 && resized.texture.width == 32 && resized.texture.height == 16,
                  "Obsolete in-flight resize image reached the consumer");
            gate->Wait(5); surface.SetVSync(false);
            auto changed = acquire(surface, gate, [&] {
                gate->Allow(5); gate->Wait(6);
                Check(!gate->vsync[5], "VSync change did not reach native acquisition");
                gate->Allow(6);
            });
            Check(identity(changed) == 6, "Obsolete in-flight VSync image reached the consumer");
            gate->Wait(7);
            std::atomic_uint rejected{};
            std::jthread foreign{[&] {
                try { surface.SetSize(1, 1); } catch (const std::logic_error&) { ++rejected; }
                try { surface.SetVSync(true); } catch (const std::logic_error&) { ++rejected; }
                try { (void)surface.Acquire(); } catch (const std::logic_error&) { ++rejected; }
                try { (void)surface.TryAcquire(); } catch (const std::logic_error&) { ++rejected; }
                try { (void)surface.HasReady(); } catch (const std::logic_error&) { ++rejected; }
            }}; foreign.join();
            Check(rejected == 5, "Drawable handoff accepted a foreign consumer");
            surface.SetSize(0, 0);
            Check(surface.Acquire() == nil, "Suspended surface waited for an in-flight acquisition");
            Check(surface.TryAcquire() == nil, "Suspended surface published an in-flight image");
            // Destructor must retire an acquisition already blocked in the native
            // boundary and release its layer/ready image before state destruction.
            gate->Allow(UINT_MAX);
            first = second = resized = changed = nil;
        }
    }
    Check(gate->alive == 0, "Stopped drawable handoff retained an image after both owners released it");
    for (bool nil_result : {true, false}) {
        const auto isolated = std::make_shared<SurfaceGate>();
        isolated->nil_at = nil_result ? 1 : 0;
        isolated->fail_at = nil_result ? 0 : 1;
        @autoreleasepool {
            auto* layer = [SurfaceTestLayer layer]; layer->gate = isolated;
            Metal::PresentSurface surface{device, (__bridge void*)layer}; surface.SetSize(8, 8);
            bool failed{};
            id<CAMetalDrawable> initial = nil;
            try {
                initial = acquire(surface, isolated, [&] { isolated->Wait(1); isolated->Allow(1); });
            } catch (const std::runtime_error& error) {
                if (std::string_view{error.what()} != "Injected native acquisition failure") throw;
                failed = true;
            }
            Check(initial == nil && failed == !nil_result && isolated->calls == 1,
                  "Native acquisition failure was swallowed or producer retried without a consumer");
            if (nil_result) {
                auto value = acquire(surface, isolated, [&] { isolated->Wait(2); isolated->Allow(2); });
                Check(identity(value) == 2, "A native nil result prevented a later real frame");
                value = nil; isolated->Allow(UINT_MAX);
            } else {
                for (unsigned action = 0; action < 5; ++action) {
                    bool retained{};
                    try {
                        if (action == 0) surface.SetSize(8, 8);
                        else if (action == 1) surface.SetVSync(true);
                        else if (action == 2) (void)surface.Acquire();
                        else if (action == 3) (void)surface.TryAcquire();
                        else (void)surface.HasReady();
                    } catch (const std::runtime_error& error) {
                        retained = std::string_view{error.what()} == "Injected native acquisition failure";
                    }
                    Check(retained, "Pending native acquisition error disappeared");
                }
            }
        }
        Check(isolated->alive == 0, "Nil/error handoff leaked a retained image");
    }
    std::cout << "Native surface handoff: one prefetched image, distinct acquisition thread, ordered consumers, resize/vsync generations, suspend, nil recovery, error propagation, foreign callers and joined lifetime passed\n";
}

void ValidateNonblockingSurface(Metal::Device& device) {
    for (const bool initial_nil : {false, true}) {
        const auto gate = std::make_shared<SurfaceGate>();
        gate->nil_at = initial_nil ? 1 : 0;
        @autoreleasepool {
            auto* layer = [SurfaceTestLayer layer]; layer->gate = gate;
            Metal::PresentSurface surface{device, (__bridge void*)layer};
            // Release a blocked test producer before destroying the surface,
            // including when an assertion fails.
            struct Release {
                std::shared_ptr<SurfaceGate> gate;
                ~Release() { gate->Allow(UINT_MAX); }
            } release{gate};
            surface.SetSize(16, 16);
            Check(surface.TryAcquire() == nil, "Pending drawable blocked the consumer");
            gate->Wait(1);
            for (unsigned n = 0; n < 32; ++n)
                Check(surface.TryAcquire() == nil, "Blocked producer fabricated an image");
            Check(gate->calls == 1 && gate->produced == 0,
                  "Nonblocking polling started more than one native acquisition");

            // Mirror SwapBuffers' unconditional guest submission while the
            // window producer is deliberately still blocked. Check actual
            // GPU-written bytes before allowing any drawable to complete.
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            const auto bytes = device.CreateBuffer(256, true);
            for (unsigned frame = 1; frame <= 3; ++frame) {
                transfers.FillBuffer(bytes, NSMakeRange(0, 256), std::uint8_t(frame + 40));
                scheduler.SubmitPendingBeforeDrawable(!surface.HasReady());
                Check(surface.TryAcquire() == nil, "Pending window stopped guest GPU progress");
                scheduler.SubmitPending();
                scheduler.Finish();
                const auto* data = static_cast<const unsigned char*>(bytes.contents);
                Check(std::all_of(data, data + 256, [&](auto value) { return value == frame + 40; }),
                      "Guest Metal commands were lost when the drawable was unavailable");
                Check(gate->produced == 0, "GPU test accidentally released its blocked window");
            }
            const auto await_ready = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
                while (!surface.HasReady() && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                Check(surface.HasReady(), "Nonblocking producer did not publish its completed result");
            };
            gate->Allow(1); gate->Wait(1, true); await_ready();
            if (initial_nil) {
                Check(surface.TryAcquire() == nil && gate->calls == 1,
                      "Consuming a native nil result retried without another guest frame");
                Check(surface.TryAcquire() == nil, "Nil recovery fabricated a drawable");
                gate->Wait(2); gate->Allow(2); gate->Wait(2, true); await_ready();
            }
            auto drawable = surface.TryAcquire();
            Check(drawable != nil && drawable.texture.width == 16,
                  "Ready native image was lost during nonblocking handoff");
            Check(((__bridge SurfaceTestDrawable*)(__bridge void*)drawable)->identity ==
                      (initial_nil ? 2u : 1u),
                  "Nonblocking consumer skipped a prepared native image");
            gate->Wait(initial_nil ? 3 : 2);
            surface.SetSize(0, 0);
            Check(surface.TryAcquire() == nil, "Suspended nonblocking surface exposed a stale image");
            drawable = nil;
        }
        Check(gate->alive == 0, "Nonblocking handoff retained an image after destruction");
    }
    std::cout << "Nonblocking surface: bounded polling, GPU submission and verified GPU writes while nextDrawable is blocked, ready image delivery, nil recovery and suspended lifetime passed\n";
}

Layout::FramebufferLayout LayoutFor(unsigned width, unsigned height) {
    return {.width = width, .height = height, .top_screen_enabled = true, .bottom_screen_enabled = false,
            .top_screen = {0, 0, width, height}, .bottom_screen = {}, .cardboard = {},
            .render_3d_mode = Settings::StereoRenderOption::Off};
}

id<MTLTexture> Texture(Metal::Device& device, unsigned width, unsigned height, MTLPixelFormat format,
                       bool sampled) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | (sampled ? MTLTextureUsageShaderRead : MTLTextureUsageUnknown);
    if (sampled) descriptor.usage |= MTLTextureUsagePixelFormatView;
    return device.CreateTexture(descriptor);
}

void Upload(Metal::Device& device, Metal::TransferRuntime& transfers, id<MTLTexture> texture,
             const std::vector<Color>& colors) {
    Check(colors.size() == texture.width * texture.height, "Invalid presentation fixture size");
    auto buffer = device.CreateBuffer(colors.size() * 4, true);
    auto* data = static_cast<unsigned char*>(buffer.contents);
    for (std::size_t i = 0; i < colors.size(); ++i)
        for (unsigned c = 0; c < 4; ++c) data[i * 4 + c] = colors[i][c];
    transfers.Upload(buffer, {0, texture.width * 4, 0}, {texture, 0, 0, MTLRegionMake2D(0, 0, texture.width, texture.height)});
}

Color Pixel(const std::vector<std::byte>& data, unsigned width, unsigned x, unsigned y, bool bgra = false) {
    Color result{};
    for (unsigned c = 0; c < 4; ++c) result[c] = std::to_integer<unsigned>(data[(y * width + x) * 4 + c]);
    if (bgra) std::swap(result[0], result[2]);
    return result;
}

void Same(Color actual, Color expected, const char* message, unsigned tolerance = 0) {
    for (unsigned c = 0; c < 4; ++c) {
        if (std::abs(int(actual[c]) - int(expected[c])) > int(tolerance)) {
            std::cerr << "channel " << c << ": expected " << expected[c] << " got " << actual[c] << '\n';
            throw std::runtime_error(message);
        }
    }
}

Color Dubois(Color left, Color right) {
    constexpr std::array<std::array<double, 3>, 3> l{{{.437, .449, .164}, {-.062, -.062, -.024}, {-.048, -.050, -.017}}};
    constexpr std::array<std::array<double, 3>, 3> r{{{-.011, -.032, -.007}, {.377, .761, .009}, {-.026, -.093, 1.234}}};
    Color output{0, 0, 0, left[3]};
    for (unsigned c = 0; c < 3; ++c) {
        double value{};
        for (unsigned k = 0; k < 3; ++k) value += l[c][k] * left[k] + r[c][k] * right[k];
        output[c] = std::lround(std::clamp(value, 0.0, 255.0));
    }
    return output;
}

Color Blend(Color source, Color dest, double opacity) {
    Color result{};
    for (unsigned c = 0; c < 4; ++c) result[c] = std::lround(source[c] * opacity + dest[c] * (1 - opacity));
    return result;
}

void ValidateDeferredPresentation(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    Metal::TransferRuntime transfer{device, scheduler};
    auto red = Texture(device, 2, 2, MTLPixelFormatRGBA8Unorm, true);
    auto green = Texture(device, 2, 2, MTLPixelFormatRGBA8Unorm, true);
    Upload(device, transfer, red, std::vector<Color>(4, Red));
    Upload(device, transfer, green, std::vector<Color>(4, Green));
    const std::array<id<MTLTexture>, 3> images{red, red, green};
    const std::array<Metal::ScreenImage, 3> screens{{{red, {0,0,1,1}},
        {red, {0,0,1,1}}, {green, {0,0,1,1}}}};
    auto layout = LayoutFor(128, 128);
    layout.bottom_screen_enabled = true;
    layout.top_screen = {0, 0, 64, 128}; layout.bottom_screen = {32, 0, 96, 128};
    layout.top_opacity = .75f; layout.bottom_opacity = .25f;
    const auto original = Metal::BuildScreenDraws(layout, screens, {});
    const auto original_cursor = Metal::BuildScreenCursor(layout, true, 64.5f, 64.5f);
    Check(original.size() == 2 && !original_cursor.empty(), "Invalid deferred presentation fixture");
    const auto background = MTLClearColorMake(17.0/255, 34.0/255, 51.0/255, 1);
    for (const bool bgra : {false, true}) for (const unsigned count : {2U, 64U}) {
        auto target = Texture(device, 128, 128, bgra ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm, false);
        auto compositor = std::make_unique<Metal::Compositor>(device, scheduler);
        compositor->Render(target, images, {}, background);
        scheduler.Finish();
        auto draws = original;
        while (draws.size() < count) draws.insert(draws.end(), original.begin(), original.end());
        auto cursor = original_cursor;
        Metal::Test::CompilerGate gate{device};
        const auto before = scheduler.WorkerStats();
        const auto encoders = scheduler.EncoderCount();
        const auto compiled = device.DiskCacheStats().compiled_pipelines;
        compositor->Render(target, images, draws, background, cursor);
        const auto held = scheduler.WorkerStats();
        const bool no_compile = device.DiskCacheStats().compiled_pipelines == compiled;
        const bool one_pass = scheduler.EncoderCount() == encoders + 1;
        std::fill(draws.begin(), draws.end(), Metal::PresentDraw{});
        std::fill(cursor.begin(), cursor.end(), std::array<float,2>{});
        const bool timely = gate.Open();
        Check(timely && no_compile && one_pass && held.queued_draws == before.queued_draws + count + 1 &&
                  held.encoded_draws == before.encoded_draws,
              "Presentation blocked/compiled on its owner, cleared between draws or lost a batch item");
        compositor.reset();
        const auto pixels = transfer.Readback({target,0,0,MTLRegionMake2D(0,0,128,128)},128*4);
        // Independent opacity/coverage oracle away from the cursor, including
        // overlapping layers and all untouched background pixels.
        for (unsigned x = 0; x < 128; ++x) {
            auto expected = Background;
            for (unsigned i = 0; i < count / 2; ++i) {
                if (x < 64) expected = Blend(Red, expected, .75);
                if (x >= 32 && x < 96) expected = Blend(Green, expected, .25);
            }
            Same(Pixel(pixels,128,x,4,bgra), expected, "Deferred presentation changed coverage/opacity/clear order", 1);
        }
        // Compare every pixel after releasing all caller-owned arrays and the
        // original compositor, including the mixed strip/triangle cursor path.
        Metal::Compositor repeat{device, scheduler};
        draws.clear();
        while (draws.size() < count) draws.insert(draws.end(), original.begin(), original.end());
        repeat.Render(target, images, draws, background, original_cursor);
        Check(pixels == transfer.Readback({target,0,0,MTLRegionMake2D(0,0,128,128)},128*4),
              "Queued presentation lost immutable input or compositor lifetime");
        repeat.Render(target, images, std::span{original.data(),1}, background);
        const auto next = transfer.Readback({target,0,0,MTLRegionMake2D(0,0,128,128)},128*4);
        for (unsigned y = 0; y < 128; ++y) for (unsigned x = 0; x < 128; ++x)
            Same(Pixel(next,128,x,y,bgra), x < 64 ? Blend(Red,Background,.75) : Background,
                 "Same-target presentation batch retained a prior layer or cursor", 1);
        repeat.Render(target, images, {}, background);
        const auto empty = transfer.Readback({target,0,0,MTLRegionMake2D(0,0,128,128)},128*4);
        for (unsigned y = 0; y < 128; ++y) for (unsigned x = 0; x < 128; ++x)
            Same(Pixel(empty,128,x,y,bgra), Background, "Background-only presentation failed to clear");
    }
    std::cout << "Compiler-gated presentation: RGBA/BGRA, 2/64 layers plus cursor, one clear/pass, immutable arrays, released compositor, exact full-image replay, independent blend/coverage, next-batch clear and background-only output passed\n";
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const bool surface_only = argc == 2 && std::string_view{argv[1]} == "--surface-only";
            Check(argc == 1 || argc == 3 || surface_only, "Specify optional cold/warm/offcold/offwarm and isolated root");
            const std::string mode = argc == 3 ? argv[1] : "";
            const std::filesystem::path root = argc == 3 ? argv[2] : "";
            Check(root.empty() || root.is_absolute(), "Presentation cache root must be absolute");
            Check(mode.empty() || mode == "cold" || mode == "warm" || mode == "offcold" || mode == "offwarm",
                  "Invalid presentation cache mode");
            Metal::Device device{root, mode != "offcold"};
            ValidateSurfaceHandoff(device);
            ValidateNonblockingSurface(device);
            if (surface_only) return 0;
            if (mode.empty()) ValidateDeferredPresentation(device);
            std::atomic_bool stop{false};
            if (mode == "warm" || mode == "offwarm") {
                device.SwitchDiskCache(0xfeed);
                Metal::PipelineCache fresh{device};
                const auto ready = fresh.Prewarm(stop);
                Check(ready.prepared == 4 && !ready.failed && device.DiskCacheStats().compiled_pipelines == 0,
                      "Fresh presentation manifests failed to restore standalone native sources/PSOs");
            }
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::Compositor compositor{device, scheduler};
            if (!mode.empty()) {
                device.SetDiskCacheEnabled(true); device.SwitchDiskCache(0xfeed);
                if (mode == "warm" || mode == "offwarm") {
                    const auto ready = compositor.Prewarm(stop, [&](std::size_t, std::size_t) {
                        (void)compositor.PipelineStats();
                    });
                    Check(ready.prepared == 4 && !ready.failed && device.DiskCacheStats().compiled_pipelines == 0,
                          "Native presentation family prewarm failed");
                }
            }
            const auto before_render = compositor.PipelineStats().render_builds;
            std::array<Metal::ScreenImage, 3> screens;
            const auto left = Texture(device, 8, 8, MTLPixelFormatRGBA8Unorm, true);
            const auto right = Texture(device, 16, 8, MTLPixelFormatRGBA8Unorm, true);
            const auto bottom = Texture(device, 2, 2, MTLPixelFormatRGBA8Unorm, true);
            std::vector<Color> left_pixels(64, Poison), right_pixels(128, Poison);
            for (unsigned y = 0; y < 8; ++y) {
                for (unsigned x = 0; x < 4; ++x) left_pixels[y * 8 + x] = Red;
                for (unsigned x = 8; x < 16; ++x) right_pixels[y * 16 + x] = Blue;
            }
            Upload(device, transfers, left, left_pixels);
            Upload(device, transfers, right, right_pixels);
            Upload(device, transfers, bottom, std::vector<Color>(4, Green));
            screens = {{{left, {0, 0, 1, .5f}}, {right, {0, .5f, 1, 1}}, {bottom, {0, 0, 1, 1}}}};
            const std::array<id<MTLTexture>, 3> images{left, right, bottom};
            const auto render = [&](id<MTLTexture> target, const auto& layout, const Metal::ScreenLayoutOptions& options,
                                    std::span<const std::array<float, 2>> cursor = {}) {
                const auto draws = Metal::BuildScreenDraws(layout, screens, options);
                compositor.Render(target, images, draws, MTLClearColorMake(17.0 / 255, 34.0 / 255, 51.0 / 255, 1), cursor);
                return transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, target.width, target.height)}, target.width * 4);
            };

            Pica::FramebufferConfig lcd{};
            lcd.address_left1 = 100; lcd.address_left2 = 200;
            lcd.address_right1 = 300; lcd.address_right2 = 0;
            Check(Metal::GetActiveLCDAddress(lcd, true) == 300, "Native LCD ignored a valid active right-eye buffer");
            lcd.active_fb = 1;
            Check(Metal::GetActiveLCDAddress(lcd, true) == 200, "Native LCD did not fall back to the active left-eye buffer");
            lcd.address_right2 = 400;
            Check(Metal::GetActiveLCDAddress(lcd, true) == 400 && Metal::GetActiveLCDAddress(lcd, false) == 200,
                  "Native LCD selected the wrong buffer generation");

            for (unsigned mode = 0; mode < 7; ++mode) {
                for (bool swap : {false, true}) {
                    for (bool rotated : {false, true}) {
                        auto layout = LayoutFor(mode == 1 || mode == 2 || mode == 6 ? 32 : 16, 8);
                        layout.render_3d_mode = Settings::StereoRenderOption(mode);
                        layout.is_rotated = rotated;
                        if (mode == 2) layout.top_screen = {0, 0, 16, 8};
                        if (mode == 6) layout.top_screen = {0, 0, 8, 8};
                        auto target = Texture(device, layout.width, layout.height, MTLPixelFormatRGBA8Unorm, false);
                        const auto pixels = render(target, layout, {.swap_eyes = swap});
                        const Color l = swap ? Blue : Red, r = swap ? Red : Blue;
                        for (unsigned y = 0; y < layout.height; ++y) {
                            for (unsigned x = 0; x < layout.width; ++x) {
                                Color expected;
                                if (mode == 0) expected = Red;
                                else if (mode == 1 || mode == 2) expected = x < 16 ? l : r;
                                else if (mode == 6) expected = x < 8 ? l : x >= 16 && x < 24 ? r : Background;
                                else if (mode == 3) expected = Dubois(l, r);
                                else expected = ((y & 1) == unsigned(mode == 5)) ? l : r;
                                Same(Pixel(pixels, layout.width, x, y), expected,
                                     "Native stereo, eye swap, LCD orientation, cache crop or physical-row parity mismatch", mode == 3 ? 1 : 0);
                            }
                        }
                    }
                }
            }
            auto mono_layout = LayoutFor(16, 8);
            auto mono_target = Texture(device, 16, 8, MTLPixelFormatRGBA8Unorm, false);
            const auto mono_pixels = render(mono_target, mono_layout, {.mono_eye = 1});
            for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 16; ++x)
                Same(Pixel(mono_pixels, 16, x, y), Blue, "Native mono right-eye selection mismatch");
            std::cout << "Seven stereo modes, both eye orders/orientations, physical-row parity, mono and independent cache crops passed\n";

            auto corner = Texture(device, 2, 2, MTLPixelFormatRGBA8Unorm, true);
            Upload(device, transfers, corner, {Red, Green, Blue, Yellow});
            const std::array<Metal::ScreenImage, 3> corners{{{corner, {0, 0, 1, 1}}, {corner, {0, 0, 1, 1}}, {corner, {0, 0, 1, 1}}}};
            const std::array<id<MTLTexture>, 3> corner_images{corner, corner, corner};
            constexpr std::array<std::array<Color, 4>, 4> expected_corners{{
                {Green, Yellow, Red, Blue}, {Yellow, Blue, Green, Red},
                {Blue, Red, Yellow, Green}, {Red, Green, Blue, Yellow}}};
            for (unsigned format = 0; format < 2; ++format) {
                for (unsigned orientation = 0; orientation < 4; ++orientation) {
                    const auto layout = LayoutFor(16, 12);
                    const auto draw = Metal::MakeScreenDraw(layout, corners, {4, 2, 12, 10}, Layout::DisplayOrientation(orientation),
                                                            0, 0, Metal::PresentMode::Single, false, 1, false);
                    auto target = Texture(device, 16, 12, format ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm, false);
                    compositor.Render(target, corner_images, std::span{&draw, 1}, MTLClearColorMake(17.0 / 255, 34.0 / 255, 51.0 / 255, 1));
                    const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 16, 12)}, 16 * 4);
                    for (unsigned c = 0; c < 4; ++c)
                        Same(Pixel(pixels, 16, c & 1 ? 10 : 5, c & 2 ? 8 : 3, format), expected_corners[orientation][c],
                             "Native LCD corner rotation, offset viewport or BGRA screenshot order mismatch");
                    Same(Pixel(pixels, 16, 0, 0, format), Background, "Native screenshot changed a pixel outside its display rectangle");
                }
            }
            std::cout << "All four LCD rotations, offset rectangles, top-down RGBA/BGRA screenshot readback passed\n";

            auto layers = LayoutFor(16, 8);
            layers.bottom_screen_enabled = true;
            layers.top_screen = {0, 0, 8, 8}; layers.bottom_screen = {4, 0, 12, 8};
            layers.top_opacity = .75f; layers.bottom_opacity = .25f;
            auto layer_target = Texture(device, 16, 8, MTLPixelFormatRGBA8Unorm, false);
            for (bool swap : {false, true}) {
                const auto pixels = render(layer_target, layers, {.swap_screen_order = swap});
                const auto expected = swap ? Blend(Red, Blend(Green, Background, .25), .75)
                                           : Blend(Green, Blend(Red, Background, .75), .25);
                Same(Pixel(pixels, 16, 6, 4), expected, "Native layer opacity or screen order mismatch", 1);
            }
            auto additional = LayoutFor(16, 16);
            additional.top_screen = {0, 0, 8, 8};
            additional.additional_screen_enabled = true; additional.additional_screen = {8, 8, 16, 16};
            auto additional_target = Texture(device, 16, 16, MTLPixelFormatRGBA8Unorm, false);
            for (bool bottom_extra : {false, true}) {
                additional.additional_screen_is_bottom = bottom_extra;
                const auto pixels = render(additional_target, additional, {});
                Same(Pixel(pixels, 16, 12, 12), bottom_extra ? Green : Red, "Native additional-screen selection mismatch");
                Same(Pixel(pixels, 16, 12, 4), Background, "Native additional screen covered the wrong rectangle");
            }
            auto secondary = LayoutFor(16, 8);
            secondary.top_screen_enabled = false; secondary.bottom_screen_enabled = true;
            secondary.bottom_screen = {0, 0, 16, 8};
            const auto secondary_pixels = render(layer_target, secondary, {});
            Same(Pixel(secondary_pixels, 16, 8, 4), Green, "Native secondary bottom-screen layout mismatch");
            std::cout << "Opacity, overlapping layer order, additional screens and secondary-window layouts passed\n";

            auto cursor_layout = LayoutFor(128, 128);
            cursor_layout.top_screen_enabled = false; cursor_layout.bottom_screen_enabled = true;
            cursor_layout.bottom_screen = {0, 0, 128, 128};
            auto cursor_target = Texture(device, 128, 128, MTLPixelFormatRGBA8Unorm, false);
            const auto cursor = Metal::BuildScreenCursor(cursor_layout, true, 64.5f, 64.5f);
            const auto cursor_pixels = render(cursor_target, cursor_layout, {}, cursor);
            Same(Pixel(cursor_pixels, 128, 64, 63), Poison, "Native cursor failed to invert the vertical arm");
            Same(Pixel(cursor_pixels, 128, 63, 64), Poison, "Native cursor failed to invert the horizontal arm");
            Same(Pixel(cursor_pixels, 128, 64, 64), Green, "Native cursor intersection did not preserve double-inversion behavior");
            Same(Pixel(cursor_pixels, 128, 20, 20), Green, "Native cursor changed pixels outside its arms");
            for (const auto& p : Metal::BuildScreenCursor(cursor_layout, true, .5f, .5f))
                Check(p[0] >= -1 && p[1] >= -1 && p[0] <= 1 && p[1] <= 1, "Native cursor escaped its screen bounds");
            Check(Metal::BuildScreenCursor(cursor_layout, true, -100, -100).empty(), "Native off-screen cursor was not clipped");
            std::cout << "Native cursor inversion, intersection, clipping and bounds passed\n";

            auto linear_layout = LayoutFor(8, 8);
            const auto nearest = Metal::MakeScreenDraw(linear_layout, corners, {0, 0, 8, 8}, Layout::DisplayOrientation::PortraitFlipped,
                                                      0, 0, Metal::PresentMode::Single, false, 1, false);
            auto linear = nearest; linear.linear_filter = true;
            auto linear_target = Texture(device, 8, 8, MTLPixelFormatRGBA8Unorm, false);
            compositor.Render(linear_target, corner_images, std::span{&nearest, 1}, MTLClearColorMake(0, 0, 0, 1));
            const auto near_pixels = transfers.Readback({linear_target, 0, 0, MTLRegionMake2D(0, 0, 8, 8)}, 32);
            compositor.Render(linear_target, corner_images, std::span{&linear, 1}, MTLClearColorMake(0, 0, 0, 1));
            const auto linear_pixels = transfers.Readback({linear_target, 0, 0, MTLRegionMake2D(0, 0, 8, 8)}, 32);
            Check(Pixel(near_pixels, 8, 3, 3) != Pixel(linear_pixels, 8, 3, 3), "Native nearest/linear presentation filtering did not differ");
            Same(Pixel(linear_pixels, 8, 3, 3), {135, 96, 60, 255}, "Native linear screen sampling differs from bilinear reference", 1);
            std::cout << "Native nearest/linear display sampling passed\n";
            unsigned alias_cases{};
            for (bool bgra : {false, true}) for (bool view : {false, true})
                for (bool filtered : {false, true})
                    for (auto mode : {Metal::PresentMode::Single, Metal::PresentMode::Anaglyph,
                                      Metal::PresentMode::Interlaced}) {
                        const auto format = bgra ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm;
                        auto alias_target = Texture(device, 8, 8, format, true);
                        auto independent_source = Texture(device, 8, 8, format, true);
                        auto reference_target = Texture(device, 8, 8, format, false);
                        std::vector<Color> input(64);
                        for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x)
                            input[y * 8 + x] = {20 + x * 23, 30 + y * 19, 40 + (x + y) * 11, 80 + x * 17};
                        Upload(device, transfers, alias_target, input);
                        Upload(device, transfers, independent_source, input);
                        id<MTLTexture> alias_source = alias_target;
                        if (view) {
                            auto* desc = [[MTLTextureViewDescriptor alloc] init];
                            desc.pixelFormat = format; desc.textureType = MTLTextureType2D;
                            desc.levelRange = NSMakeRange(0, 1); desc.sliceRange = NSMakeRange(0, 1);
                            alias_source = [alias_target newTextureViewWithDescriptor:desc];
                            Check(alias_source && alias_source != alias_target,
                                  "Alias fixture failed to create a separate view of the output allocation");
                        }
                        const std::array<id<MTLTexture>, 3> aliased_images{alias_source, alias_source, alias_source};
                        const std::array<id<MTLTexture>, 3> independent_images{independent_source, independent_source, independent_source};
                        const std::array<Metal::ScreenImage, 3> alias_screens{{
                            {alias_source, {0, 0, 1, 1}}, {alias_source, {0, 0, 1, 1}}, {alias_source, {0, 0, 1, 1}}}};
                        const auto draw = Metal::MakeScreenDraw(LayoutFor(8, 8), alias_screens, {0, 0, 8, 8},
                            Layout::DisplayOrientation::PortraitFlipped, 0, 1, mode, false, 1, filtered);
                        compositor.Render(reference_target, independent_images, std::span{&draw, 1},
                                          MTLClearColorMake(0, 0, 0, 1));
                        compositor.Render(alias_target, aliased_images, std::span{&draw, 1},
                                          MTLClearColorMake(0, 0, 0, 1));
                        const auto reference = transfers.Readback({reference_target, 0, 0, MTLRegionMake2D(0, 0, 8, 8)}, 32);
                        const auto actual = transfers.Readback({alias_target, 0, 0, MTLRegionMake2D(0, 0, 8, 8)}, 32);
                        Check(actual == reference, "Native presentation alias/view sampling did not preserve immutable input before clear");
                        for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x) {
                            auto expected = input[y * 8 + x];
                            if (bgra) std::swap(expected[0], expected[2]);
                            if (mode == Metal::PresentMode::Anaglyph) expected = Dubois(expected, expected);
                            Same(Pixel(actual, 8, x, y, bgra), expected,
                                 "Native alias presentation source color/alpha or stereo projection oracle mismatch", 2);
                        }
                        ++alias_cases;
                    }
            std::cout << alias_cases << " native compositor alias cases: direct/parent views, RGBA/BGRA, nearest/linear, single/anaglyph/interlaced, immutable source and all64 color/alpha pixels passed\n";
            scheduler.Finish();
            if (!mode.empty()) {
                Check(device.FlushDiskCache(), "Native presentation archive flush failed");
                if (mode == "warm" || mode == "offwarm")
                    Check(compositor.PipelineStats().prewarm_hits == 4 &&
                          compositor.PipelineStats().render_builds == before_render &&
                          device.DiskCacheStats().compiled_pipelines == 0,
                          "First prepared presentation repeated PSO creation");
                std::cout << "Native presentation closure/prewarm: four RGBA/BGRA screen/cursor manifests, independent fresh library restoration, first draws4 ready hits/0 render builds/0 PSO compiles in warm mode passed\n";
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
