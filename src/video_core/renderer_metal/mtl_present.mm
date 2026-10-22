// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_present.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include <cmath>
#include <condition_variable>
#include <optional>
#include <thread>
#include <pthread.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace Metal {

namespace {
id<MTLTexture> RootTexture(id<MTLTexture> texture) {
    while (texture.parentTexture) texture = texture.parentTexture;
    return texture;
}
}

Compositor::Compositor(Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, stream{device, scheduler, 1 << 20}, cache{device} {
    library = device.CompileLibrary(PresentShaderSource, "Azahar Metal 4 presentation");
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
    descriptor.minFilter = MTLSamplerMinMagFilterNearest;
    descriptor.magFilter = MTLSamplerMinMagFilterNearest;
    samplers[0] = device.CreateSampler(descriptor);
    descriptor.minFilter = MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplers[1] = device.CreateSampler(descriptor);
}

Compositor::Pipelines Compositor::GetPipelines(MTLPixelFormat format) {
    if (format != MTLPixelFormatBGRA8Unorm && format != MTLPixelFormatRGBA8Unorm) {
        throw std::invalid_argument("Unsupported Metal presentation target format");
    }
    BlendState screen{};
    screen.enabled = true; screen.source_rgb = screen.source_alpha = 12;
    screen.dest_rgb = screen.dest_alpha = 13;
    BlendState cursor{};
    cursor.enabled = true; cursor.source_rgb = 5; cursor.dest_rgb = 0;
    cursor.source_alpha = 1; cursor.dest_alpha = 0;
    Pipelines result;
    result.screens = cache.RenderAsync({library, "present_vertex", library, "present_fragment", format, screen});
    result.cursor = cache.RenderAsync({library, "cursor_vertex", library, "cursor_fragment", format, cursor});
    return result;
}

PipelinePrewarmResult Compositor::Prewarm(const std::atomic_bool& stop,
    const std::function<void(std::size_t, std::size_t)>& progress) {
    if (stop.load()) { PipelinePrewarmResult result{}; result.cancelled = true; return result; }
    cache.Clear();
    constexpr std::array<std::string_view, 2> entries{"present_vertex", "cursor_vertex"};
    return cache.Prewarm(stop, progress, entries);
}

void Compositor::Render(id<MTLTexture> target, const std::array<id<MTLTexture>, 3>& screens,
                         std::span<const PresentDraw> draws, MTLClearColor background,
                         std::span<const std::array<float, 2>> cursor) {
    if (!target || !target.width || !target.height || target.sampleCount != 1 ||
        target.textureType != MTLTextureType2D) {
        throw std::invalid_argument("Invalid Metal presentation target");
    }
    for (id<MTLTexture> screen : screens) {
        if (!screen || screen.textureType != MTLTextureType2D || screen.sampleCount != 1 ||
            screen.framebufferOnly) {
            throw std::invalid_argument("Invalid Metal presentation screen texture");
        }
    }
    if (draws.size() > 64 || cursor.size() > 1024 || cursor.size() % 3) {
        throw std::invalid_argument("Metal presentation geometry exceeds its bounded batch");
    }
    if (target.pixelFormat != MTLPixelFormatBGRA8Unorm && target.pixelFormat != MTLPixelFormatRGBA8Unorm)
        throw std::invalid_argument("Unsupported Metal presentation target format");
    if (draws.empty() && cursor.empty()) {
        TransferRuntime transfers{device, scheduler};
        transfers.ClearColor(target, background);
        return;
    }
    const auto pipeline = GetPipelines(target.pixelFormat);

    auto input_screens = screens;
    if (!draws.empty()) {
        TransferRuntime transfers{device, scheduler};
        const auto target_root = RootTexture(target);
        for (std::size_t i = 0; i < screens.size(); ++i) {
            const auto source_root = RootTexture(screens[i]);
            if (source_root != target_root &&
                (!source_root.buffer || source_root.buffer != target_root.buffer)) continue;
            bool reused{};
            for (std::size_t previous = 0; previous < i; ++previous)
                if (screens[previous] == screens[i]) {
                    input_screens[i] = input_screens[previous]; reused = true; break;
                }
            if (reused) continue;
            auto* copy = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:screens[i].pixelFormat
                width:screens[i].width height:screens[i].height mipmapped:NO];
            copy.storageMode = MTLStorageModePrivate; copy.usage = MTLTextureUsageShaderRead;
            input_screens[i] = device.CreateTexture(copy);
            // Copy before the output pass clears its attachment. Sampling a
            // sibling view of the same allocation must also see immutable input.
            const auto region = MTLRegionMake2D(0, 0, screens[i].width, screens[i].height);
            transfers.CopyTexture({screens[i], 0, 0, region}, {input_screens[i], 0, 0, region});
        }
    }

    // Allocate the complete batch before recording any consumers. Allocate can submit older work.
    std::vector<RenderBatchDraw> prepared;
    prepared.reserve(draws.size() + !cursor.empty());
    const MTLViewport viewport{0,0,double(target.width),double(target.height),0,1};
    const MTLScissorRect scissor{0,0,target.width,target.height};
    const NSUInteger stride = 512;
    const NSUInteger cursor_offset = draws.size() * stride;
    const NSUInteger bytes = cursor_offset + cursor.size_bytes();
    const auto batch = bytes ? stream.Allocate(bytes) : BufferSlice{};
    for (std::size_t i = 0; i < draws.size(); ++i) {
        const auto& draw = draws[i];
        if (draw.uniforms.left_screen < 0 || draw.uniforms.left_screen >= 3 ||
            draw.uniforms.right_screen < 0 || draw.uniforms.right_screen >= 3 ||
            draw.uniforms.mode < PresentMode::Single || draw.uniforms.mode > PresentMode::Interlaced ||
            draw.uniforms.reverse_interlaced < 0 || draw.uniforms.reverse_interlaced > 1 ||
            !std::isfinite(draw.opacity) || draw.opacity < 0.0f || draw.opacity > 1.0f) {
            throw std::invalid_argument("Invalid Metal presentation uniforms");
        }
        auto* data = static_cast<std::byte*>(batch.data) + i * stride;
        std::memcpy(data, draw.vertices.data(), sizeof(draw.vertices));
        std::memcpy(data + 256, &draw.uniforms, sizeof(draw.uniforms));
        const std::array<BufferBinding, 2> buffers{{
            {batch.buffer, batch.offset + i * stride, 0},
            {batch.buffer, batch.offset + i * stride + 256, 1},
        }};
        const std::array<TextureBinding, 3> textures{{{input_screens[0], 0}, {input_screens[1], 1}, {input_screens[2], 2}}};
        const std::array<SamplerBinding, 1> filter{{{samplers[draw.linear_filter], 0}}};
        RenderCommand command{.pipeline=nil,.depth=nil,.viewport=viewport,.scissor=scissor,
            .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=4,
            .pending_pipeline=pipeline.screens,.primitive=MTLPrimitiveTypeTriangleStrip,
            .blend_color={0,0,0,draw.opacity}};
        prepared.push_back({command,{buffers.begin(),buffers.end()},{textures.begin(),textures.end()},
            {filter.begin(),filter.end()}});
    }
    if (!cursor.empty()) {
        std::memcpy(static_cast<std::byte*>(batch.data) + cursor_offset, cursor.data(),
                    cursor.size_bytes());
        const std::array<BufferBinding, 1> buffers{{{batch.buffer, batch.offset + cursor_offset, 0}}};
        RenderCommand command{.pipeline=nil,.depth=nil,.viewport=viewport,.scissor=scissor,
            .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=cursor.size(),
            .pending_pipeline=pipeline.cursor};
        prepared.push_back({command,{buffers.begin(),buffers.end()},{},{}});
    }
    MTL4RenderPassDescriptor* descriptor = [[MTL4RenderPassDescriptor alloc] init];
    descriptor.colorAttachments[0].texture = target;
    descriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
    descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
    descriptor.colorAttachments[0].clearColor = background;
    const auto snapshot=scheduler.SnapshotRenderPass(descriptor);
    scheduler.QueueRenderBatch(*snapshot,prepared);
}

// Acquire at most one image ahead. This worker never encodes GPU commands or
// touches guest/framebuffer state. A consuming drawable stays owned by its caller
// and the scheduler; the sole ready drawable stays owned by this handoff.
struct PresentSurface::Acquisition {
    struct Request {
        NSUInteger width{}, height{};
        bool vsync{true};
        std::uint64_t generation{};
    };
    struct Result {
        id<CAMetalDrawable> drawable;
        std::uint64_t generation{};
    };

    CAMetalLayer* layer;
    const std::thread::id owner{std::this_thread::get_id()};
    std::mutex mutex;
    std::condition_variable condition;
    Request desired{};
    std::optional<Request> pending;
    std::optional<std::uint64_t> working;
    std::optional<Result> ready;
    std::exception_ptr error;
    bool started{}, stopping{};
    std::jthread worker;

    explicit Acquisition(CAMetalLayer* layer_) : layer{layer_} {
        worker = std::jthread{[this] { Run(); }};
    }

    ~Acquisition() {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
            pending.reset();
            ready.reset();
        }
        condition.notify_all();
        // allowsNextDrawableTimeout bounds a native acquisition in progress.
        // Join before releasing the retained layer and synchronization objects.
        worker.join();
    }

    void CheckOwner() const {
        if (owner != std::this_thread::get_id())
            throw std::logic_error("Metal drawable handoff belongs to another thread");
    }

    void RequestNext() {
        if (stopping || error || !desired.width || !desired.height || ready ||
            (working && *working == desired.generation) ||
            (pending && pending->generation == desired.generation)) return;
        pending = desired;
        condition.notify_all();
    }

    void Changed() {
        if (desired.generation == UINT64_MAX)
            throw std::overflow_error("Metal drawable generation exhausted");
        ++desired.generation;
        ready.reset();
        pending.reset();
        if (started) RequestNext();
    }

    void Run() noexcept {
        pthread_setname_np("MetalDrawable");
        try {
            for (;;) {
                Request request;
                {
                    std::unique_lock lock{mutex};
                    condition.wait(lock, [this] { return stopping || pending.has_value(); });
                    if (stopping) return;
                    request = *pending;
                    pending.reset();
                    working = request.generation;
                }
                @autoreleasepool {
                    const auto size = CGSizeMake(request.width, request.height);
                    if (!CGSizeEqualToSize(layer.drawableSize, size)) layer.drawableSize = size;
                    if (bool(layer.displaySyncEnabled) != request.vsync)
                        layer.displaySyncEnabled = request.vsync;
                    id<CAMetalDrawable> drawable = [layer nextDrawable];
                    if (drawable && (!drawable.texture || drawable.texture.device != layer.device ||
                        drawable.texture.width != request.width || drawable.texture.height != request.height))
                        throw std::runtime_error("Native Metal drawable does not match its requested surface");
                    {
                        std::scoped_lock lock{mutex};
                        working.reset();
                        if (stopping) return;
                        if (request.generation == desired.generation)
                            ready = Result{drawable, request.generation};
                        else RequestNext(); // Discard an obsolete resize/vsync result.
                    }
                    condition.notify_all();
                }
            }
        } catch (...) {
            {
                std::scoped_lock lock{mutex};
                error = std::current_exception();
                working.reset();
                pending.reset();
                ready.reset();
            }
            condition.notify_all();
        }
    }
};

PresentSurface::PresentSurface(const Device& device, void* metal_layer) {
    auto* layer = (__bridge CAMetalLayer*)metal_layer;
    if (!layer || ![layer isKindOfClass:[CAMetalLayer class]]) {
        throw std::invalid_argument("Native Metal presentation requires a CAMetalLayer");
    }
    layer.device = device.Handle();
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.opaque = YES;
    layer.maximumDrawableCount = 3;
    layer.allowsNextDrawableTimeout = YES;
    acquisition = std::make_unique<Acquisition>(layer);
}

PresentSurface::~PresentSurface() = default;

void PresentSurface::SetSize(NSUInteger width, NSUInteger height) {
    auto& state = *acquisition;
    state.CheckOwner();
    std::scoped_lock lock{state.mutex};
    if (state.error) std::rethrow_exception(state.error);
    if (state.desired.width == width && state.desired.height == height) return;
    state.desired.width = width;
    state.desired.height = height;
    state.Changed();
}

void PresentSurface::SetVSync(bool enabled) {
    auto& state = *acquisition;
    state.CheckOwner();
    std::scoped_lock lock{state.mutex};
    if (state.error) std::rethrow_exception(state.error);
    if (state.desired.vsync == enabled) return;
    state.desired.vsync = enabled;
    state.Changed();
}

bool PresentSurface::HasReady() {
    auto& state = *acquisition;
    state.CheckOwner();
    std::scoped_lock lock{state.mutex};
    if (state.error) std::rethrow_exception(state.error);
    return state.ready.has_value();
}

id<CAMetalDrawable> PresentSurface::Acquire() {
    return Acquire(true);
}

id<CAMetalDrawable> PresentSurface::TryAcquire() {
    return Acquire(false);
}

id<CAMetalDrawable> PresentSurface::Acquire(bool wait) {
    auto& state = *acquisition;
    state.CheckOwner();
    std::unique_lock lock{state.mutex};
    if (state.error) std::rethrow_exception(state.error);
    if (!state.desired.width || !state.desired.height) return nil;
    state.started = true;
    state.RequestNext();
    if (wait)
        state.condition.wait(lock, [&] { return state.error || state.stopping || state.ready.has_value(); });
    if (state.error) std::rethrow_exception(state.error);
    if (state.stopping) throw std::logic_error("Metal drawable handoff stopped");
    if (!state.ready) return nil;
    if (state.ready->generation != state.desired.generation)
        throw std::logic_error("Obsolete Metal drawable reached its consumer");
    id<CAMetalDrawable> drawable = state.ready->drawable;
    state.ready.reset();
    // A nil native timeout/suspended result is not retried in the background.
    // The next actual emulation frame may request again, as in the direct path.
    if (drawable) state.RequestNext();
    return drawable;
}

} // namespace Metal
