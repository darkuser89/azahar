// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_gas_runtime.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace Metal {
struct SchedulerTestAccess {
    static void BlockQueue(Scheduler& scheduler, id<MTLSharedEvent> event) {
        scheduler.CheckThread();
        [scheduler.queue waitForEvent:event value:1];
    }
};
}

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// Oracle for the selected IEEE half storage policy. PICA rounding still needs
// hardware measurements; these tests must not be reported as hardware goldens.
std::uint16_t Half(float v) { return std::bit_cast<std::uint16_t>(_Float16(v)); }
float Float(std::uint16_t v) { return float(std::bit_cast<_Float16>(v)); }
std::uint32_t Pack(std::uint16_t a, std::uint16_t b) {
    return std::bit_cast<std::uint32_t>(std::array<std::uint8_t, 4>{
        std::uint8_t(a >> 8), std::uint8_t(a), std::uint8_t(b >> 8), std::uint8_t(b)});
}
Metal::BufferSlice Slice(id<MTLBuffer> b, NSUInteger offset = 16, NSUInteger size = 0) {
    return {b, offset, size ? size : b.length - offset - 16, nullptr, {}};
}
template <typename F> void Reject(F&& f) {
    bool rejected{};
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, "Gas accepted invalid buffers/configuration");
}
void ValidateFeedbackCompletion(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    Metal::TransferRuntime transfer{device, scheduler};
    Metal::GasRuntime gas{device, scheduler};
    Metal::GasFeedback feedback{device};
    gas.LoadMaximum(feedback, 0x3555);
    Check(gas.ReadMaximum(feedback) == 0x3555, "Gas initial completion lost imported bits");

    auto gate = [device.Handle() newSharedEvent];
    // Avoid hanging a regression if the ready read accidentally waits for
    // unrelated queue work. Normal completion releases this guard immediately.
    struct GateRelease {
        std::mutex mutex;
        std::condition_variable changed;
        bool done{};
        id<MTLSharedEvent> event;
        std::jthread watchdog;
        explicit GateRelease(id<MTLSharedEvent> event_) : event{event_}, watchdog{[this] {
            std::unique_lock lock{mutex};
            changed.wait_for(lock, std::chrono::seconds{2}, [&] { return done; });
            event.signaledValue = 1;
        }} {}
        ~GateRelease() {
            { std::scoped_lock lock{mutex}; done = true; }
            changed.notify_all();
        }
    } release{gate};
    Metal::SchedulerTestAccess::BlockQueue(scheduler, gate);
    auto unrelated = device.CreateBuffer(64, true);
    transfer.FillBuffer(unrelated, NSMakeRange(0, 64), 0xa7);
    const auto blocked = scheduler.Submit();
    Check(!scheduler.IsComplete(blocked), "Gas fixture failed to hold unrelated GPU work");
    transfer.FillBuffer(unrelated, NSMakeRange(16, 16), 0x5c);
    const auto tick = scheduler.CurrentTick(), encoders = scheduler.EncoderCount();
    Check(gas.ReadMaximum(feedback) == 0x3555 && gate.signaledValue == 0 &&
          !scheduler.IsComplete(blocked) && scheduler.CurrentTick() == tick &&
          scheduler.EncoderCount() == encoders,
          "Completed Gas maximum submitted/waited for unrelated work");

    bool rejected{};
    std::thread foreign{[&] {
        try { (void)gas.ReadMaximum(feedback); }
        catch (const std::logic_error&) { rejected = true; }
    }};
    foreign.join();
    Check(rejected, "Completed Gas feedback bypassed command ownership");

    // A fresh write to the same four bytes must invalidate the ready path.
    gas.LoadMaximum(feedback, 0x7c00);
    const auto writer = scheduler.LastUseTick(gas.PrepareFragmentMaximum(feedback).buffer);
    Check(!scheduler.IsComplete(writer), "Gas replacement writer was not genuinely pending");
    gate.signaledValue = 1;
    Check(gas.ReadMaximum(feedback) == 0x7c00 && scheduler.IsComplete(writer),
          "Gas observation published a stale or incomplete replacement maximum");
    for (unsigned i = 0; i < 64; ++i)
        Check(static_cast<const unsigned char*>(unrelated.contents)[i] ==
                  (i >= 16 && i < 32 ? 0x5c : 0xa7),
              "Gas completion changed unrelated ordered GPU bytes");
    feedback.Reset();
    Check(gas.ReadMaximum(feedback) == 0, "Gas reset did not supersede the completed maximum");
    scheduler.CheckAfterCompletion([] { throw std::runtime_error("expected Gas completion failure"); });
    try { scheduler.Finish(); } catch (const std::runtime_error&) {}
    bool retained_error{};
    try { (void)gas.ReadMaximum(feedback); }
    catch (const std::runtime_error& error) {
        retained_error = std::string_view{error.what()} == "expected Gas completion failure";
    }
    Check(retained_error, "Ready/reset Gas observation swallowed a retained GPU failure");
    std::cout << "Gas feedback completion: gated unrelated GPU work and pending commands remain unsubmitted; exact replacement bits,64 unrelated bytes, reset, owner and retained errors passed\n";
}
std::array<double, 4> Reference(std::uint16_t plain, std::uint16_t depth, float xy,
                                const Metal::GasUniforms& c) {
    const auto byte = [](std::uint32_t w, unsigned s) { return double((w >> s) & 255) / 255; };
    const double d = (c.shading_depth ? Float(depth) : Float(plain)) * Float(c.reciprocal_maximum);
    // Patent equations: mix(min,max,(1-att*D/AMAX)*incoming_light).
    const double wx = (1 - d * byte(c.light_xy, 16)) * xy;
    const double wz = (1 - d * byte(c.light_z, 16)) * byte(c.light_z_color, 0);
    const double light = byte(c.light_xy, 0) + wx * (byte(c.light_xy, 8) - byte(c.light_xy, 0)) +
                         byte(c.light_z, 0) + wz * (byte(c.light_z, 8) - byte(c.light_z, 0));
    const double p = std::clamp(c.light_z_color & 256 ? light : d, 0.0, 1.0) * 8;
    const auto i = std::min(unsigned(p), 7u);
    std::array<double, 4> out{};
    for (unsigned k = 0; k < 3; ++k) {
        const auto v = (c.color_lut[i] >> (8 * k)) & 255;
        out[k] = std::clamp(byte(c.color_lut[8 + i], 8 * k) +
            double(v & 127) / 127 * (v & 128 ? -1 : 1) * (p - i), 0.0, 1.0);
    }
    const double fp = Float(depth) * Float(c.attenuation) * 128;
    const auto fi = unsigned(std::clamp(std::floor(fp), 0.0, 127.0));
    const auto entry = c.fog_lut[fi];
    const int diff = entry & 4096 ? int(entry & 8191) - 8192 : int(entry & 8191);
    out[3] = std::clamp(double((entry >> 13) & 2047) / 2047 + double(diff) / 2047 * (fp - fi), 0.0, 1.0);
    return out;
}
}

int main() {
    @autoreleasepool {
        try {
            Metal::Device device;
            ValidateFeedbackCompletion(device);
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfer{device, scheduler};
            Metal::GasRuntime gas{device, scheduler};
            Metal::GasFeedback feedback{device};
            for (unsigned seed = 0; seed < 258; ++seed) {
                const auto maximum = std::uint16_t(seed == 257 ? 0x7c00 : seed * 123u % 0x7c01u);
                gas.LoadMaximum(feedback, maximum);
                Check(gas.ReadReciprocalMaximum(feedback) == Half(1.f / Float(maximum)),
                      "Native GPU reciprocal maximum differs from the independent half policy");
                Check(gas.ReadMaximum(feedback) == maximum, "Reciprocal calculation erased the accumulated maximum");
            }
            feedback.Reset();
            Check(gas.ReadReciprocalMaximum(feedback) == 0x7c00 && gas.ReadMaximum(feedback) == 0,
                  "Native reciprocal retained maximum feedback across a pending reset");
            std::cout << "PASS: 258 native GPU reciprocal maximum cases including zero/infinity and ordered reset\n";
            constexpr unsigned count = 31744, active = 2053;
            auto guest = device.CreateBuffer(count * 4 + 32, false);
            auto host = device.CreateBuffer(guest.length, true);
            auto contribution = device.CreateBuffer(count * sizeof(Metal::GasContribution) + 32, true);
            auto depths = device.CreateBuffer(count * 4 + 32, true);
            auto* words = reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(host.contents) + 16);
            auto* input = reinterpret_cast<Metal::GasContribution*>(static_cast<std::byte*>(contribution.contents) + 16);
            auto* scene = reinterpret_cast<float*>(static_cast<std::byte*>(depths.contents) + 16);
            std::memset(host.contents, 0xD3, host.length);
            for (unsigned i = 0; i < count; ++i) {
                words[i] = Pack(i, count - 1 - i);
                input[i] = {0, 0.25f, 1, 0}; scene[i] = 0.5f;
            }
            transfer.CopyBuffer(host, 0, guest, 0, guest.length);
            gas.Accumulate(Slice(guest), Slice(contribution), Slice(depths), count, 1 << 24, feedback);
            transfer.CopyBuffer(guest, 0, host, 0, guest.length); scheduler.Finish();
            for (unsigned i = 0; i < count; ++i)
                Check(words[i] == Pack(i, count - 1 - i), "Gas half-pair bytes changed during zero accumulation");
            Check(gas.ReadMaximum(feedback) == 0x7BFF, "Gas maximum missed a finite half value");
            const auto guards = [&] {
                const auto* b = static_cast<const std::uint8_t*>(host.contents);
                for (unsigned i = 0; i < 16; ++i)
                    Check(b[i] == 0xD3 && b[host.length - 1 - i] == 0xD3, "Gas changed a buffer guard");
            };
            guards();
            std::vector<std::array<std::uint16_t, 2>> expected(active);
            for (unsigned mode = 0; mode < 4; ++mode) {
                feedback.Reset(); Check(gas.ReadMaximum(feedback) == 0, "Gas reset retained feedback");
                for (unsigned i = 0; i < active; ++i) {
                    expected[i] = {Half(float(i % 23) / 16), Half(float(i % 17) / 32)};
                    words[i] = Pack(expected[i][0], expected[i][1]);
                    input[i] = {float(i % 29) / 64, float(i % 19) / 16, i % 7 ? 1.0f : 0.0f, 0};
                    scene[i] = float(i % 31) / 16;
                }
                transfer.CopyBuffer(host, 0, guest, 0, guest.length);
                std::uint16_t maximum{};
                for (unsigned pass = 0; pass < 3; ++pass) {
                    for (unsigned i = 0; i < active; ++i) {
                        if (!input[i].coverage) continue;
                        float a = std::clamp((scene[i] - input[i].depth) * 1.5f, 0.0f, 1.0f);
                        if (mode < 2) a = float(mode); else if (mode == 2) a = 1 - a;
                        expected[i][0] = Half(Float(expected[i][0]) + input[i].density);
                        expected[i][1] = Half(Float(expected[i][1]) + input[i].density * a);
                        maximum = std::max(maximum, expected[i][0]);
                    }
                    gas.Accumulate(Slice(guest), Slice(contribution), Slice(depths), active, mode << 24 | 384, feedback);
                    transfer.CopyBuffer(guest, 0, host, 0, guest.length); scheduler.Finish();
                    for (unsigned i = 0; i < active; ++i)
                        Check(words[i] == Pack(expected[i][0], expected[i][1]), "Gas coverage/depth/ordered half accumulation differs");
                    Check(gas.ReadMaximum(feedback) == maximum, "Gas GPU feedback differs from stored density");
                    guards();
                }
            }
            for (auto value : {0x7C00u, 0x3555u, 0u}) {
                gas.LoadMaximum(feedback, value);
                Check(gas.ReadMaximum(feedback) == value, "Gas feedback import/export changed bits");
            }
            Reject([&] { gas.LoadMaximum(feedback, 0x7C01); });
            auto lighting = device.CreateBuffer(active * 4 + 32, true);
            auto output = device.CreateBuffer(active * 16 + 32, false);
            auto result = device.CreateBuffer(output.length, true);
            auto* xy = reinterpret_cast<float*>(static_cast<std::byte*>(lighting.contents) + 16);
            for (unsigned i = 0; i < active; ++i) {
                expected[i] = {Half(float(i % 257) / 256), Half(float((i * 31) % 513) / 256)};
                words[i] = Pack(expected[i][0], expected[i][1]); xy[i] = float(i % 129) / 128;
            }
            transfer.CopyBuffer(host, 0, guest, 0, guest.length);
            std::memset(result.contents, 0xC7, result.length);
            transfer.CopyBuffer(result, 0, output, 0, output.length);
            for (unsigned variant = 0; variant < 64; ++variant) {
                Metal::GasUniforms c{};
                c.attenuation = Half(variant & 4 ? 0.75f : 1.5f);
                c.reciprocal_maximum = Half(variant & 8 ? 0.5f : 1.0f);
                c.shading_depth = variant & 1;
                c.light_z_color = (variant & 2 ? 256 : 0) | ((variant * 17) & 255);
                c.light_xy = 0x40C020u ^ (variant * 33547u); c.light_z = 0x807030u ^ (variant * 13573u);
                for (unsigned i = 0; i < 8; ++i) {
                    c.color_lut[i + 8] = (0x346EA1u + i * 13271u + variant * 7411u) & 0xFFFFFF;
                    c.color_lut[i] = (0x7F8040u + i * 8191u + variant * 3167u) & 0xFFFFFF;
                }
                for (unsigned i = 0; i < 128; ++i)
                    c.fog_lut[i] = ((i * 37 + variant * 53) & 2047) << 13 |
                                  (std::uint32_t(int((i * 71 + variant * 29) & 8191) - 4096) & 8191);
                gas.Synthesize(Slice(guest), Slice(lighting), Slice(output), active, c);
                transfer.CopyBuffer(output, 0, result, 0, output.length); scheduler.Finish();
                const auto* actual = reinterpret_cast<const float*>(static_cast<const std::byte*>(result.contents) + 16);
                for (unsigned i = 0; i < active; ++i) {
                    const auto wanted = Reference(expected[i][0], expected[i][1], xy[i], c);
                    for (unsigned k = 0; k < 4; ++k)
                        Check(std::isfinite(actual[i * 4 + k]) && std::abs(actual[i * 4 + k] - wanted[k]) < 0.00002,
                              "Gas density/XY/Z lighting/color/opacity synthesis differs");
                }
                const auto* b = static_cast<const std::uint8_t*>(result.contents);
                for (unsigned i = 0; i < 16; ++i)
                    Check(b[i] == 0xC7 && b[result.length - 1 - i] == 0xC7, "Gas synthesis changed output guards");
            }
            Reject([&] { gas.Accumulate(Slice(guest, 17), Slice(contribution), Slice(depths), 1, 0, feedback); });
            Reject([&] { gas.Accumulate(Slice(guest, 16, 3), Slice(contribution), Slice(depths), 1, 0, feedback); });
            Reject([&] { gas.Accumulate(Slice(guest), Slice(contribution, 17), Slice(depths), 1, 0, feedback); });
            Reject([&] { gas.Accumulate(Slice(guest), Slice(contribution), Slice(guest), 1, 0, feedback); });
            Reject([&] { gas.Synthesize(Slice(guest), Slice(lighting), Slice(guest), 1, {}); });
            Reject([&] { gas.Synthesize(Slice(guest), Slice(lighting), Slice(output, 17), 1, {}); });
            Reject([&] { Metal::GasUniforms c{}; c.shading_depth = 2; gas.Synthesize(Slice(guest), Slice(lighting), Slice(output), 1, c); });
            Reject([&] { Metal::GasUniforms c{}; c.padding = 1; gas.Synthesize(Slice(guest), Slice(lighting), Slice(output), 1, c); });
            scheduler.Finish();
            std::cout << "Metal4 Gas stages passed: 31744 half-pairs, 24636 accumulation pixels, 525568 synthesis channels, feedback/reset/import, guards, invalid arguments.\n"
                      << "Algorithm tests; PICA hardware precision remains unproven. "
                         "Draw integration is covered by separate native/PICA tests.\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
