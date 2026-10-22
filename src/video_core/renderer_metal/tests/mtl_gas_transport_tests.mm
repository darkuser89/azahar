// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Data objects and native Metal resources only; no application, ROM or OS window.
#include "mtl_pica_vertex_fixture.h"
#include "mtl_rasterizer.h"
#include "common/archives.h"
#include "core/core.h"
#include "core/memory.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/image_interface.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include <cstring>
#include <iostream>
#include <sstream>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
class Window final : public Frontend::EmuWindow {
public:
    void PollEvents() override {}
};
class Renderer final : public VideoCore::RendererBase {
public:
    Renderer(Core::System& system, Window& window) : RendererBase{system, window, nullptr} {}
    VideoCore::RasterizerInterface* Rasterizer() override { return rasterizer; }
    void SwapBuffers() override {}
    void TryPresent(int, bool) override {}
    VideoCore::RasterizerInterface* rasterizer{};
};
struct Event {
    u32 reg, value, mask;
    bool operator==(const Event&) const = default;
};
class Capture final : public VideoCore::RasterizerInterface {
public:
    bool IsMetalRasterizer() const override { return native; }
    bool native{true};
    std::vector<Event> events;
    void WriteMetalGasRegisters(u32 reg, std::span<const u32> values, u32 mask) override {
        for (auto value : values) events.push_back({reg, value, mask});
    }
    void AddTriangle(const Pica::OutputVertex&, const Pica::OutputVertex&, const Pica::OutputVertex&) override {}
    void DrawTriangles() override {}
    void FlushAll() override {}
    void FlushRegion(PAddr, u32) override {}
    void InvalidateRegion(PAddr, u32) override {}
    void FlushAndInvalidateRegion(PAddr, u32) override {}
    void ClearAll(bool) override {}
};
}

int main() {
    @autoreleasepool {
        try {
            const auto original_api = Settings::values.graphics_api.GetValue();
            for (const auto api : {
#ifdef ENABLE_SOFTWARE_RENDERER
                     Settings::GraphicsAPI::Software,
#endif
#ifdef ENABLE_VULKAN
                     Settings::GraphicsAPI::Vulkan,
#endif
                     Settings::GraphicsAPI::Metal}) {
                Settings::values.graphics_api.SetValue(api);
                Check(Settings::values.graphics_api.GetValue() == api &&
                          Settings::GetWorkingGraphicsAPI() == api,
                      "Renderer setting was clamped or overridden instead of selecting its backend");
            }
            Settings::values.graphics_api.SetValue(original_api);
            Core::System system;
            system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());
            Memory::MemorySystem memory{system};
            Pica::PicaCore pica{memory, nullptr};
            Metal::Test::InitializeFixtureUniforms(pica.vs_setup);
            Metal::Test::InitializeFixtureUniforms(pica.gs_setup);
            constexpr auto address = Memory::VRAM_PADDR + 0x8000;
            auto submit = [&](std::vector<u32> words) {
                if (words.size() % 2) words.push_back(0);
                std::memcpy(memory.GetPhysicalPointer(address), words.data(), words.size() * 4);
                pica.ProcessCmdList(address, u32(words.size() * 4), false);
                std::memset(memory.GetPhysicalPointer(address), 0, words.size() * 4);
            };
            Capture capture;
            capture.events.push_back({0x125, 0xabcdef01, 15});
            const auto before_default_hook = capture.events;
            capture.SynchronizeMetalGasMaximum();
            Check(capture.events == before_default_hook,
                  "Default Gas synchronization changed a backend without an override");
            pica.BindRasterizer(&capture);
            unsigned event_count{};
            for (u32 mask = 0; mask < 16; ++mask) {
                for (u32 mode = 0; mode < 3; ++mode) {
                    capture.events.clear();
                    std::vector<u32> words;
                    std::vector<Event> expected;
                    if (mode == 2) {
                        words = {0x12345678, 0x123u | (mask << 16) | (2u << 20) | (1u << 31),
                                 0xabcdef90, 0x76543210};
                        expected = {{0x123, words[0], mask}, {0x124, words[2], mask},
                                    {0x125, words[3], mask}};
                    } else for (u32 reg = 0x123; reg <= 0x125; ++reg) {
                        std::array<u32, 16> values{};
                        for (u32 i = 0; i < values.size(); ++i) {
                            values[i] = 0x80000000u ^ (reg << 20) ^ (i * 0x10203u);
                            expected.push_back({reg, values[i], mask});
                            if (mode == 0) words.insert(words.end(), {values[i], reg | (mask << 16)});
                        }
                        if (mode == 1) {
                            words.insert(words.end(), {values[0], reg | (mask << 16) | (15u << 20)});
                            words.insert(words.end(), values.begin() + 1, values.end());
                            if (words.size() % 2) words.push_back(0);
                        }
                    }
                    submit(words);
                    Check(capture.events == expected, "Public PICA lost/reordered/masked raw Gas writes");
                    event_count += unsigned(expected.size());
                }
            }
            Window window;
            Renderer renderer{system, window};
            VideoCore::CustomTexManager custom{system};
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory, pica, custom, renderer, device, scheduler};
            renderer.rasterizer = &rasterizer;
            pica.BindRasterizer(&rasterizer);
            const auto clean = rasterizer.SaveMetalState();
            Check(Metal::GasArchiveWord(clean, 8) == 2, "Clean snapshot compatibility changed");
            // Check every retained byte, including arbitrary shader/primitive banks.
            // Padding is canonical zero; this tests the archive codec, not PICA math.
            for (u32 seed = 0; seed < 256; ++seed) {
                auto payload = clean;
                for (std::size_t i = 32; i < payload.size(); ++i)
                    payload[i] = Metal::GasArchiveOmitted(i) ? 0 : u8((i * 71 + seed * 97) & 255);
                Metal::GasArchiveWord(payload, 12, seed % 3);
                Metal::GasArchiveWord(payload, 16, (seed / 3) % 3);
                Metal::GasArchiveWord(payload, 20, seed & 1);
                Metal::GasArchiveWord(payload, 24, (seed >> 1) & 1);
                Metal::GasArchiveWord(payload, 28, (seed >> 2) & 1);
                Metal::GasTransferState original{}, restored{};
                for (unsigned i = 0; i < original.lut.size(); ++i)
                    original.lut[i] = seed * 0x01020304u ^ (i * 0x10203040u);
                original.cursor = seed * 257;
                original.feedback_pending = seed & 1;
                const auto packed = Metal::PackGasArchive(payload, original);
                Check(Metal::UnpackGasArchive(packed, restored) == payload && restored == original,
                      "Compact archive lost a shader/primitive/Gas byte");
                original.feedback_pending = false;
                original.maximum = seed ? seed * 0x7c00u / 255 : 0x7c00;
                const auto with_maximum = Metal::PackGasArchive(payload, original);
                Check(Metal::GasArchiveWord(with_maximum, 8) == 4 &&
                      Metal::UnpackGasArchive(with_maximum, restored) == payload && restored == original,
                      "Version 4 archive lost a retained byte or GPU maximum");
            }
            // A restored maximum must become actual GPU state, survive cache
            // eviction, and reset when an old snapshot or ordered reset loads.
            for (u32 maximum : {1u, 0x3555u, 0x7bffu, 0x7c00u}) {
                Metal::GasTransferState restored{}; restored.maximum = maximum;
                const auto state = Metal::PackGasArchive(clean, restored);
                rasterizer.LoadMetalState(state);
                rasterizer.ClearAll(false);
                Check(rasterizer.SaveMetalState() == state, "GPU Gas maximum was lost on import/cache clear");
                for (u32 bad_maximum : {0x7c01u, 0x8000u, 0xffffffffu}) {
                    auto bad = state;
                    Metal::GasArchiveWord(bad, Metal::GasArchiveOffset + 72, bad_maximum);
                    bool rejected{};
                    try { rasterizer.LoadMetalState(bad); } catch (const std::invalid_argument&) { rejected = true; }
                    Check(rejected && rasterizer.SaveMetalState() == state, "Invalid maximum changed live GPU state");
                }
                auto pending = state;
                Metal::GasArchiveWord(pending, Metal::GasArchiveOffset + 68, 1);
                bool rejected{};
                try { rasterizer.LoadMetalState(pending); } catch (const std::invalid_argument&) { rejected = true; }
                Check(rejected && rasterizer.SaveMetalState() == state, "Pending reset retained an impossible maximum");
                rasterizer.LoadMetalState(clean);
                Check(rasterizer.SaveMetalState() == clean, "Version 2 import retained a stale GPU maximum");
            }
            unsigned lut_count{};
            for (bool grouped : {false, true}) for (u32 seed = 0; seed < 32; ++seed) {
                rasterizer.LoadMetalState(clean);
                std::vector<u32> words{0, 0x123u | (15u << 16)};
                std::array<u32, 16> lut{};
                // Includes all-black and distinct tables with the same final zero word.
                for (u32 i = 0; i < 15; ++i) lut[i] = seed * 0x10001u + i * seed;
                if (grouped) {
                    words.insert(words.end(), {lut[0], 0x124u | (15u << 16) | (15u << 20)});
                    words.insert(words.end(), lut.begin() + 1, lut.end());
                } else for (auto value : lut) words.insert(words.end(), {value, 0x124u | (15u << 16)});
                submit(words);
                Metal::GasTransferState gas{};
                const auto saved = rasterizer.SaveMetalState();
                Check(saved.size() == clean.size(), "Gas changed the shared archive envelope size");
                const auto expanded = Metal::UnpackGasArchive(saved, gas);
                Check(gas.lut == lut && gas.cursor == 16, "Native Gas bank lost LUT words/cursor");
                Check(expanded == clean, "Gas archive changed unrelated shader/primitive state");
                rasterizer.ClearAll(false);
                Check(rasterizer.SaveMetalState() == saved, "Cache clear erased persistent Gas state");
                rasterizer.LoadMetalState(clean);
                rasterizer.LoadMetalState(saved);
                Check(rasterizer.SaveMetalState() == saved, "Gas snapshot roundtrip failed");
                for (auto offset : {12u, unsigned(Metal::GasArchiveOffset + 68),
                                    unsigned(Metal::GasArchiveOffset + 72)}) {
                    auto bad = saved;
                    Metal::GasArchiveWord(bad, offset, 0xffffffffu);
                    bool rejected{};
                    try { rasterizer.LoadMetalState(bad); } catch (const std::invalid_argument&) { rejected = true; }
                    Check(rejected && rasterizer.SaveMetalState() == saved,
                          "Malformed Gas archive changed live state or was accepted");
                }
                // Exercise the unchanged public PICA serialization envelope too.
                std::stringstream stream;
                { oarchive archive{stream}; archive << pica; }
                rasterizer.LoadMetalState(clean);
                { iarchive archive{stream}; archive >> pica; }
                Check(rasterizer.SaveMetalState() == saved, "PICA archive lost native Gas state");
                ++lut_count;
            }
            rasterizer.LoadMetalState(clean);
            submit({0, 0x125u | (3u << 16)});
            Metal::GasTransferState feedback{};
            const auto feedback_state = rasterizer.SaveMetalState();
            Metal::UnpackGasArchive(feedback_state, feedback);
            Check(feedback.feedback_pending, "Zero-valued Gas feedback action was coalesced away");
            for (u32 mask = 0; mask < 15; ++mask) {
                bool rejected{};
                try { submit({0x12345678, 0x124u | (mask << 16)}); }
                catch (const std::runtime_error&) { rejected = true; }
                Check(rejected && rasterizer.SaveMetalState() == feedback_state,
                      "Unverified masked FIFO write silently changed the native table");
            }
            submit({0xffff, 0x123u | (15u << 16)});
            const auto invalid_cursor_state = rasterizer.SaveMetalState();
            bool rejected{};
            try { submit({0x12345678, 0x124u | (15u << 16)}); }
            catch (const std::runtime_error&) { rejected = true; }
            Check(rejected && rasterizer.SaveMetalState() == invalid_cursor_state,
                  "Out-of-range Gas FIFO write silently aliased a valid table entry");
            // Both implementations coexist in the app. Native hooks must stay off the
            // legacy path, and loading a state for the wrong backend must fail early.
            rasterizer.LoadMetalState(clean);
            std::stringstream native_archive;
            { oarchive archive{native_archive}; archive << pica; }
            std::stringstream legacy_metal_archive;
            {
                oarchive archive{legacy_metal_archive};
                boost::serialization::access::serialize(archive, pica, 2);
            }
            { iarchive archive{legacy_metal_archive};
              boost::serialization::access::serialize(archive, pica, 2); }
            Capture legacy;
            legacy.native = false;
            pica.BindRasterizer(&legacy);
            submit({7, 0x123u | (15u << 16), 0x12345678, 0x124u | (15u << 16),
                    0, 0x125u | (15u << 16), 3, PICA_REG_INDEX(gs.bool_uniforms) | (15u << 16)});
            Check(legacy.events.empty(), "Legacy backend received native Gas callbacks");
            const auto expected_reg = pica.regs.internal.reg_array[0x124];
            bool mismatch{};
            try { iarchive archive{native_archive}; archive >> pica; }
            catch (const std::runtime_error&) { mismatch = true; }
            Check(mismatch && pica.regs.internal.reg_array[0x124] == expected_reg,
                  "Native state entered a legacy backend or changed registers before rejection");
            std::stringstream legacy_archive;
            { oarchive archive{legacy_archive}; archive << pica; }
            pica.regs.internal.reg_array[0x124] = 0;
            { iarchive archive{legacy_archive}; archive >> pica; }
            Check(pica.regs.internal.reg_array[0x124] == expected_reg,
                  "Legacy CPU geometry/register state did not roundtrip");
            std::stringstream version_zero;
            { oarchive archive{version_zero}; boost::serialization::access::serialize(archive, pica, 0); }
            pica.regs.internal.reg_array[0x124] = 0;
            { iarchive archive{version_zero}; boost::serialization::access::serialize(archive, pica, 0); }
            Check(pica.regs.internal.reg_array[0x124] == expected_reg,
                  "Version-0 legacy PICA payload could not be restored");
            legacy_archive.clear();
            legacy_archive.seekg(0);
            pica.BindRasterizer(&rasterizer);
            mismatch = false;
            try { iarchive archive{legacy_archive}; archive >> pica; }
            catch (const std::runtime_error&) { mismatch = true; }
            Check(mismatch && rasterizer.SaveMetalState() == clean,
                  "Legacy state entered the native backend or changed native state");
            std::cout << "Mixed-backend dispatch: legacy Gas/GS writes avoid native hooks; "
                         "native v2 and legacy v0 payloads plus tagged v3 states restore; "
                         "both cross-backend loads reject before mutation.\n";
            std::cout << "PASS: " << event_count << " exact ordered PICA Gas events across all 16 masks, "
                      << lut_count << " native LUTs, cache lifetime, v2/v3/v4 and PICA archive roundtrips, "
                         "512 full-byte codec roundtrips, 208 atomic malformed-state rejections, "
                         "4 GPU maximum imports including infinity, "
                         "feedback action and 16 unsupported FIFO rejections. No app, ROM or OS window started.\n";
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
