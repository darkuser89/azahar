// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#import <CommonCrypto/CommonDigest.h>
#include "mtl_pipeline_cache.h"
#include "mtl_scheduler.h"
#include "mtl_shader_disk_cache.h"
#include "mtl_shader_runtime.h"
#include "mtl_transfer.h"
#include "mtl_translation_cache.h"
#include "mtl_translation_source_store.h"

namespace {
using namespace Metal;
using Bytes = std::vector<std::byte>;
constexpr std::string_view Vertex = R"(
void main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex],0,1);}
)";
constexpr std::string_view Fragment = R"(
layout(set=0,binding=0,std140) uniform color_data {vec4 value;};
layout(location=0) out vec4 color;
void main(){color=value;}
)";
const std::array<ShaderBindingRemap, 1> Bindings{{{0, 0, 7, NoShaderBinding, NoShaderBinding}}};
void Check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
Bytes Read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c{std::istreambuf_iterator<char>{f}, {}};
    Bytes b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}
void Write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    Check(f.good(), "Fixture write failed");
}
void Word(Bytes& b, std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i)
        b.push_back(std::byte(v >> (i * 8)));
}
void ReplaceWord(Bytes& b, unsigned at, std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i)
        b.at(at + i) = std::byte(v >> (i * 8));
}
Bytes Payload(std::string_view source, ShaderStage stage,
              std::span<const ShaderBindingRemap> bindings) {
    Bytes b;
    for (char c : std::string_view{"AZM4TXT1"})
        b.push_back(std::byte(c));
    Word(b, std::uint32_t(stage));
    Word(b, std::uint32_t(source.size()));
    for (char c : source)
        b.push_back(std::byte(c));
    Word(b, std::uint32_t(bindings.size()));
    for (auto a : bindings) {
        Word(b, a.descriptor_set);
        Word(b, a.binding);
        Word(b, a.buffer);
        Word(b, a.texture);
        Word(b, a.sampler);
    }
    return b;
}
std::filesystem::path Fixture(const std::filesystem::path& directory, Bytes b) {
    const auto identity = ShaderDiskCache::Identity(b);
    std::array<unsigned char, 32> h{};
    CC_SHA256(b.data(), CC_LONG(b.size()), h.data());
    for (auto c : h)
        b.push_back(std::byte(c));
    auto p = directory / (identity + ".input");
    Write(p, b);
    return p;
}
std::vector<std::filesystem::path> Inputs(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> paths;
    for (const auto& e : std::filesystem::directory_iterator(directory))
        if (e.path().extension() == ".input")
            paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    return paths;
}
void CompareComplete(const std::filesystem::path& root, const TranslatedShader& shader,
                     unsigned index, bool capture) {
    const std::array key{std::byte(index)};
    ShaderDiskCache store;
    const auto base = root / "cold-result", now = root / "warm-result";
    Check(store.Save(capture ? base : now, key, {shader, {}}),
          "Complete result fixture save failed");
    if (!capture) {
        const auto name = ShaderDiskCache::Identity(key) + ".shader";
        Check(Read(base / name) == Read(now / name),
              "Complete current MSL/SPIRV/reflection/member ABI differs cold/warm");
    }
}
void Draw(const TranslatedShader& vertex, const TranslatedShader& fragment) {
    Check(fragment.resources.size() == 1 && fragment.resources[0].metal_index == 7 &&
              fragment.resources[0].byte_size == 16,
          "Remapped uniform reflection changed");
    Device device;
    PipelineCache pipelines{device};
    Scheduler scheduler{device};
    TransferRuntime transfers{device, scheduler};
    auto vl = device.CompileLibrary(vertex.source, "source fixture vertex"),
         fl = device.CompileLibrary(fragment.source, "source fixture fragment");
    RenderPipelineDescription d{
        vl, vertex.entry_point, fl, fragment.entry_point, MTLPixelFormatRGBA8Unorm, {}};
    auto pipeline = pipelines.Render(d);
    auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                 width:8
                                                                height:4
                                                             mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    auto target = device.CreateTexture(td);
    auto buffer = device.CreateBuffer(16, true);
    const std::array values{.25f, .5f, .75f, 1.f};
    std::memcpy(buffer.contents, values.data(), 16);
    const std::array buffers{BufferBinding{buffer, 0, 7}};
    auto args = scheduler.MakeArguments(buffers);
    auto pass = [[MTL4RenderPassDescriptor alloc] init];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    scheduler.KeepAlive(pipeline);
    auto encoder = scheduler.BeginRender(pass);
    [encoder setRenderPipelineState:pipeline];
    [encoder setArgumentTable:args atStages:MTLRenderStageFragment];
    [encoder setViewport:MTLViewport{0, 0, 8, 4, 0, 1}];
    [encoder setScissorRect:MTLScissorRect{0, 0, 8, 4}];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    scheduler.EndEncoding();
    auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 8, 4)}, 32);
    const std::array<unsigned, 4> expected{64, 128, 191, 255};
    Check(pixels.size() == 128, "GPU readback extent changed");
    for (unsigned i = 0; i < 128; ++i)
        Check(std::to_integer<unsigned>(pixels[i]) == expected[i % 4],
              "Cold/warm full GPU pixels differ");
}
void CaptureOrWarm(const std::filesystem::path& root, bool capture) {
    TranslationSourceStore store;
    const auto shaders = root / "title/shader-source";
    std::atomic_bool stop{};
    if (!capture) {
        auto stats = GetTranslationCacheStats();
        Check(stats.entries == 0, "Warm test did not start in fresh process");
        const auto warm = store.Prewarm(shaders, stop, [] { return true; });
        Check(warm.candidates == 2 && warm.visited == 2 && warm.prepared == 2 && !warm.failed &&
                  !warm.cancelled && !warm.budget_limited,
              "Fresh process prewarm did not rebuild both inputs");
        stats = GetTranslationCacheStats();
        Check(stats.misses[0] == 2 && stats.misses[1] == 2,
              "Prewarm did not use current complete GLSL/SPIRV compiler factory");
    }
    const auto before = GetTranslationCacheStats();
    unsigned recorded{};
    TranslatedShader vs, fs;
    {
        TranslationInputScope scope{[&](auto source, auto stage, auto remaps) {
            ++recorded;
            Check(store.Save(shaders, source, stage, remaps), "Input save failed");
        }};
        vs = TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        fs = TranslateGuestGLSL(Fragment, ShaderStage::Fragment, Bindings);
    }
    Check(recorded == 2, "Successful translations were not captured");
    const auto after = GetTranslationCacheStats();
    if (!capture)
        Check(after.hits[0] == before.hits[0] + 2 && after.misses == before.misses &&
                  after.lookups[1] == before.lookups[1],
              "Actual warm calls compiled instead of complete input cache hits");
    CompareComplete(root, vs, 1, capture);
    CompareComplete(root, fs, 2, capture);
    Draw(vs, fs);
    auto paths = Inputs(TranslationSourceStore::Directory(shaders));
    Check(paths.size() == 2, "Unexpected portable input count");
    // Independent test codec verifies all input bytes, not the production reader.
    for (const auto& [source, stage, bindings] :
         std::array<std::tuple<std::string_view, ShaderStage, std::span<const ShaderBindingRemap>>,
                    2>{
             {{Vertex, ShaderStage::Vertex, {}}, {Fragment, ShaderStage::Fragment, Bindings}}}) {
        auto bytes = Payload(source, stage, bindings);
        const auto name = ShaderDiskCache::Identity(bytes) + ".input";
        auto disk = Read(TranslationSourceStore::Directory(shaders) / name);
        Check(disk.size() == bytes.size() + 32 &&
                  std::equal(bytes.begin(), bytes.end(), disk.begin()),
              "Persisted full GLSL/stage/ordered binding bytes changed");
    }
    std::cout << (capture ? "capture" : "warm")
              << ": 2 complete current compiler outputs; 128 exact GPU bytes; full input codec; "
                 "cache misses="
              << after.misses[0] << " hits=" << after.hits[0] << '\n';
}
void Scopes() {
    unsigned outer{}, inner{};
    {
        TranslationInputScope scope{[&](auto, auto, auto) { ++outer; }};
        (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        {
            TranslationInputScope mute;
            (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        }
        {
            TranslationInputScope nested{[&](auto, auto, auto) { ++inner; }};
            (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        }
        (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        try {
            (void)TranslateGuestGLSL("bad invalid source", ShaderStage::Fragment, {});
        } catch (const std::exception&) {
        }
        Check(outer == 2 && inner == 1, "Nested/muted scope or failed translation capture leaked");
        {
            TranslationInputScope throwing{
                [](auto, auto, auto) { throw std::runtime_error("optional recorder"); }};
            (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
        }
        std::vector<std::future<unsigned>> jobs;
        for (unsigned i = 0; i < 8; ++i)
            jobs.push_back(std::async(std::launch::async, [] {
                unsigned count{};
                TranslationInputScope local{[&](auto, auto, auto) { ++count; }};
                for (unsigned n = 0; n < 8; ++n)
                    (void)TranslateGuestGLSL(Vertex, ShaderStage::Vertex, {});
                return count;
            }));
        for (auto& job : jobs)
            Check(job.get() == 8, "Thread-local input scope crossed compiler threads");
        Check(outer == 2, "Foreign thread called outer recorder");
    }
}
void Edge(const std::filesystem::path& root) {
    Scopes();
    TranslationSourceStore store;
    std::atomic_bool stop{};
    Check(TranslationSourceStore::Directory({}).empty(), "Disabled cache produced path");
    Check(!store.Save({}, Vertex, ShaderStage::Vertex, {}), "Disabled cache wrote input");
    auto directory = root / "edges/shader-source",
         inputs = TranslationSourceStore::Directory(directory);
    Check(store.Prewarm(directory, stop, [] { return true; }).candidates == 0,
          "Missing input directory not empty");
    Check(!store.Save(directory, {}, ShaderStage::Vertex, {}), "Empty source accepted");
    Check(!store.Save(directory, Vertex, ShaderStage(99), {}), "Invalid stage accepted");
    std::vector<ShaderBindingRemap> too_many(129);
    Check(!store.Save(directory, Vertex, ShaderStage::Vertex, too_many),
          "Oversized binding count accepted");
    Check(!store.Save(directory, std::string(8 * 1024 * 1024 + 1, ' '), ShaderStage::Vertex, {}),
          "Oversized source accepted");
    Check(store.Save(directory, Vertex, ShaderStage::Vertex, {}), "Valid neighbor save failed");
    const auto valid = Inputs(inputs).front();
    const auto bytes = Read(valid);
    auto bad = Payload(Vertex, ShaderStage::Vertex, {});
    bad[0] = std::byte('X');
    Fixture(inputs, bad);
    bad = Payload(Vertex, ShaderStage::Vertex, {});
    ReplaceWord(bad, 8, 99);
    Fixture(inputs, bad);
    bad = Payload(Vertex, ShaderStage::Vertex, {});
    ReplaceWord(bad, 12, 0x7fffffff);
    Fixture(inputs, bad);
    bad = Payload(Vertex, ShaderStage::Vertex, {});
    ReplaceWord(bad, unsigned(bad.size() - 4), 129);
    Fixture(inputs, bad);
    bad = Payload(Vertex, ShaderStage::Vertex, {});
    bad.push_back(std::byte(1));
    Fixture(inputs, bad);
    Fixture(inputs, Payload("invalid shader", ShaderStage::Fragment, {}));
    bad = bytes;
    bad.back() ^= std::byte(1);
    Write(inputs / (std::string(64, '0') + ".input"), bad);
    Write(inputs / "bad-name.input", bytes);
    std::filesystem::create_symlink(valid, inputs / (std::string(64, '1') + ".input"));
    Write(inputs / (std::string(64, '2') + ".input"), {});
    auto warm = store.Prewarm(directory, stop, [] { return true; });
    Check(warm.prepared == 1 && warm.failed == 10 && warm.candidates == 11,
          "Corruption did not reject independently while preparing valid neighbor");
    stop = true;
    warm = store.Prewarm(directory, stop, [] { return true; });
    Check(warm.cancelled && warm.visited == 0, "Initial stop ignored");
    stop = false;
    warm = store.Prewarm(directory, stop, [] { return false; });
    Check(warm.cancelled && warm.visited == 0, "Stale title epoch accepted");
    unsigned calls{};
    warm = store.Prewarm(directory, stop, [&] { return ++calls < 3; });
    Check(warm.cancelled && !warm.prepared, "Cancellation during inventory ignored");
    auto cap = root / "cap/shader-source";
    Check(store.Save(cap, Vertex, ShaderStage::Vertex, {}) &&
              store.Save(cap, Fragment, ShaderStage::Fragment, Bindings),
          "Cap fixtures failed");
    warm = store.Prewarm(cap, stop, [] { return true; }, 1);
    Check(warm.budget_limited && warm.visited == 1 && warm.prepared == 1, "Record budget ignored");
    store.Clear();
    Check(store.Save(cap, Vertex, ShaderStage::Vertex, {}), "Clear lost valid full input");
    auto symlink = root / "symlink/shader-source";
    std::filesystem::create_directories(symlink.parent_path());
    std::filesystem::create_directory_symlink(inputs, symlink.parent_path() / "translation-inputs");
    Check(!store.Save(symlink, Vertex, ShaderStage::Vertex, {}), "Symlink input parent accepted");
    Check(!std::filesystem::exists(inputs / "v1"), "Rejected symlink parent was mutated");
    Check(store.Prewarm(symlink, stop, [] { return true; }).prepared == 0,
          "Symlink input parent prewarmed");
    // Every stage and every ordered binding field participates in the portable identity.
    auto identity = root / "identity/shader-source";
    for (unsigned stage = 0; stage < 5; ++stage)
        Check(store.Save(identity, Vertex, ShaderStage(stage), {}),
              "Stage identity fixture failed");
    std::array<ShaderBindingRemap, 2> remaps{{{0, 0, 7, 8, 9}, {1, 2, 3, 4, 5}}};
    Check(store.Save(identity, Vertex, ShaderStage::Vertex, remaps),
          "Full binding identity save failed");
    for (unsigned field = 0; field < 5; ++field) {
        auto changed = remaps;
        auto& b = changed[0];
        switch (field) {
        case 0:
            ++b.descriptor_set;
            break;
        case 1:
            ++b.binding;
            break;
        case 2:
            ++b.buffer;
            break;
        case 3:
            ++b.texture;
            break;
        case 4:
            ++b.sampler;
            break;
        }
        Check(store.Save(identity, Vertex, ShaderStage::Vertex, changed),
              "Binding field save failed");
    }
    std::swap(remaps[0], remaps[1]);
    Check(store.Save(identity, Vertex, ShaderStage::Vertex, remaps), "Binding order save failed");
    Check(store.Save(identity, std::string(Vertex) + " ", ShaderStage::Vertex, {}),
          "Source identity save failed");
    Check(Inputs(TranslationSourceStore::Directory(identity)).size() == 13,
          "Complete stage/source/binding field/order identities aliased");
    auto visits = root / "visits/shader-source";
    auto visitdir = TranslationSourceStore::Directory(visits);
    std::filesystem::create_directories(visitdir);
    for (unsigned i = 0; i < 4100; ++i) {
        std::ofstream file(visitdir / (std::to_string(i) + ".skip"));
    }
    warm = store.Prewarm(visits, stop, [] { return true; });
    Check(warm.budget_limited && !warm.candidates, "Directory visit budget ignored");
    std::filesystem::remove_all(visitdir);
    auto oversized = root / "oversized/shader-source";
    auto overdir = TranslationSourceStore::Directory(oversized);
    std::filesystem::create_directories(overdir);
    auto overfile = overdir / (std::string(64, '3') + ".input");
    Write(overfile, {});
    std::filesystem::resize_file(overfile, 9 * 1024 * 1024);
    const auto rejected = store.Stats().rejected;
    warm = store.Prewarm(oversized, stop, [] { return true; });
    Check(!warm.candidates && store.Stats().rejected == rejected + 1,
          "Oversized file was read or accepted");
    std::filesystem::remove_all(overdir);
    // Limit aggregate bytes independently of file and record limits. Valid
    // checksums surround deliberately invalid GLSL, so these are read attempts,
    // never cached compiler results or unbounded allocations.
    auto bytecap = root / "bytes/shader-source";
    auto bytedir = TranslationSourceStore::Directory(bytecap);
    std::string large(8 * 1024 * 1024, ' ');
    large.replace(0, 14, "invalid shader");
    for (unsigned i = 0; i < 9; ++i) {
        large[14] = char('a' + i);
        Fixture(bytedir, Payload(large, ShaderStage::Vertex, {}));
    }
    warm = store.Prewarm(bytecap, stop, [] { return true; });
    Check(warm.budget_limited && warm.candidates == 7 && warm.visited == 7 && warm.failed == 7 &&
              !warm.prepared,
          "Aggregate 64-MiB read budget ignored");
    std::filesystem::remove_all(bytedir);
    auto cancellation = root / "cancel/shader-source";
    Check(store.Save(cancellation, Vertex, ShaderStage::Vertex, {}), "Cancel fixture failed");
    unsigned recorded{};
    calls = 0;
    {
        TranslationInputScope outer{[&](auto, auto, auto) { ++recorded; }};
        warm = store.Prewarm(cancellation, stop, [&] {
            store.Clear();
            return ++calls < 4;
        });
    }
    Check(warm.cancelled && warm.visited == 1 && !warm.prepared && !recorded,
          "Post-compile epoch cancellation or muted capture failed");
    std::cout << "edge: full input identities, corruption, valid neighbor, scoped/concurrent "
                 "capture, disabled cache, stop/title cancellation, record/visit budgets and "
                 "symlink rejection passed\n";
}
void Runtime(const std::filesystem::path& root) {
    Device device{root / "native"};
    device.SwitchDiskCache(0x1111);
    const auto first = device.ShaderDiskCacheDirectory();
    Check(!first.empty(), "Runtime persistence test requires production shader mode");
    TranslationSourceStore store;
    Check(store.Save(first, Vertex, ShaderStage::Vertex, {}) &&
              store.Save(first, Fragment, ShaderStage::Fragment, Bindings),
          "Runtime input fixtures failed");
    ShaderRuntime runtime{device};
    std::atomic_bool stop{};
    auto result = runtime.Prewarm(stop);
    Check(!result.candidates && result.source_candidates == 2 && result.source_prepared == 2 &&
              !result.source_failed && !result.cancelled,
          "Runtime failed to prewarm portable inputs without current CPU shader records");
    result = runtime.Prewarm(stop, [&](auto, auto) { runtime.Clear(); });
    Check(result.cancelled && !result.source_prepared,
          "Runtime Clear callback failed to cancel old input epoch");
    result = runtime.Prewarm(stop, [&](auto, auto) { device.SwitchDiskCache(0x2222); });
    Check(result.cancelled && !result.source_prepared,
          "Runtime title change callback published old-title inputs");
    Check(device.ShaderDiskCacheDirectory() != first, "Title switch did not isolate input paths");
    result = runtime.Prewarm(stop);
    Check(!result.source_candidates && !result.source_prepared && !result.cancelled,
          "Different title reused old-title input files");
    device.SetDiskCacheEnabled(false);
    result = runtime.Prewarm(stop);
    Check(!result.source_candidates && !result.source_prepared,
          "Disabled runtime cache prewarmed inputs");
    stop = true;
    result = runtime.Prewarm(stop);
    Check(result.cancelled, "Runtime initial stop ignored");
    std::cout << "runtime: current compiler input prewarm without CPU results; Clear, title "
                 "switch, disabled cache and initial stop passed\n";
}
} // namespace
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 3, "Expected capture|warm|edge ROOT");
            const std::string mode = argv[1];
            const std::filesystem::path root = argv[2];
            if (mode == "capture")
                CaptureOrWarm(root, true);
            else if (mode == "warm")
                CaptureOrWarm(root, false);
            else if (mode == "edge")
                Edge(root);
            else if (mode == "runtime")
                Runtime(root);
            else
                throw std::runtime_error("Unknown mode");
            return 0;
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
