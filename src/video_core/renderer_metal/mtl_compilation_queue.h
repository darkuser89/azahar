// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace Metal {

// Owned compiler inputs only. Admission bounds include active jobs, not just
// the waiting deque. One oversized valid input may run alone without truncation.
class CompilationQueue final {
public:
    explicit CompilationQueue(unsigned workers = 2, std::size_t capacity = 128,
                              std::size_t payload_budget = 32U * 1024 * 1024)
        : capacity{capacity}, payload_budget{payload_budget} {
        if (!workers || workers > 16 || !capacity || !payload_budget)
            throw std::invalid_argument("Invalid Metal compilation queue limits");
        try {
            for (unsigned n=0;n<workers;++n) threads.emplace_back([this] { Run(); });
        } catch (...) { Stop(); throw; }
    }
    ~CompilationQueue() { Stop(); }
    CompilationQueue(const CompilationQueue&) = delete;
    CompilationQueue& operator=(const CompilationQueue&) = delete;

    template <typename Builder>
    auto Submit(Builder&& builder, std::size_t payload_bytes = 0) {
        using Result = std::invoke_result_t<Builder>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Builder>(builder));
        auto result = task->get_future().share();
        // Native factories may invoke another factory. Executing that nested
        // request here prevents an exhausted worker pool waiting for itself.
        if (current == this) { (*task)(); return result; }
        std::unique_lock lock{mutex};
        changed.wait(lock,[&] {
            return stopping || (jobs.size()+active<capacity &&
                (!retained_payload || (payload_bytes<=payload_budget && retained_payload<=payload_budget-payload_bytes)));
        });
        if (stopping) throw std::logic_error("Metal compilation queue is stopping");
        jobs.push_back({[task] { (*task)(); },payload_bytes});
        retained_payload += payload_bytes;
        changed.notify_all();
        return result;
    }
    void Drain() {
        if (current == this) throw std::logic_error("A Metal compiler worker cannot drain itself");
        std::unique_lock lock{mutex};
        changed.wait(lock,[&] {return jobs.empty() && !active;});
    }
    bool IsWorker() const noexcept { return current == this; }

private:
    struct Job { std::function<void()> run; std::size_t bytes; };
    void Stop() noexcept {
        { std::scoped_lock lock{mutex}; stopping=true; }
        changed.notify_all();
        for (auto& thread:threads) if (thread.joinable()) thread.join();
    }
    void Run() {
        current=this;
        for (;;) {
            Job job;
            {
                std::unique_lock lock{mutex};
                changed.wait(lock,[&] {return stopping || !jobs.empty();});
                if (stopping && jobs.empty()) break;
                job=std::move(jobs.front());jobs.pop_front();++active;
            }
            // packaged_task preserves the original exception for every waiter.
            job.run();
            job.run={};
            { std::scoped_lock lock{mutex}; --active;retained_payload-=job.bytes; }
            changed.notify_all();
        }
        current=nullptr;
    }
    inline static thread_local const CompilationQueue* current{};
    const std::size_t capacity, payload_budget;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Job> jobs;
    std::vector<std::thread> threads;
    std::size_t active{}, retained_payload{};
    bool stopping{};
};
} // namespace Metal
