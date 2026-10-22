// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include "mtl_device.h"
#include <array>
#include <condition_variable>
#include <thread>

namespace Metal::Test {
// A regression must fail, not hang, if the draw owner waits on the compiler.
class CompilerGate final {
public:
    explicit CompilerGate(Device& device) {
        auto wait = release.get_future().share();
        std::array<std::promise<void>, 2> started;
        for (unsigned i = 0; i < jobs.size(); ++i)
            jobs[i] = device.DeferCompilation([&, i, wait] { started[i].set_value(); wait.wait(); });
        for (auto& start : started) start.get_future().wait();
        watchdog = std::thread([this] {
            std::unique_lock lock{mutex};
            if (!changed.wait_for(lock, std::chrono::seconds{5}, [this] { return opened; })) {
                timed_out = true;
                opened = true;
                release.set_value();
            }
        });
    }
    bool Open() {
        std::scoped_lock lock{mutex};
        if (!opened) { opened = true; release.set_value(); changed.notify_all(); }
        return !timed_out;
    }
    ~CompilerGate() {
        Open();
        if (watchdog.joinable()) watchdog.join();
        for (auto& job : jobs) job.wait();
    }
private:
    std::promise<void> release;
    std::array<std::shared_future<void>, 2> jobs;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread watchdog;
    bool opened{}, timed_out{};
};
} // namespace Metal::Test
