// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_stream_buffer.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace Metal {

SharedUploadPool::SharedUploadPool(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

BufferSlice SharedUploadPool::Allocate(NSUInteger size) {
    if (!size) throw std::invalid_argument("Invalid Metal shared upload size");
    // Reject cross-thread calls before touching the arena. UseResource checks
    // retained GPU errors before any returned range can be modified by the CPU.
    (void)scheduler.LastUseTick(nil);
    constexpr NSUInteger budget = 32 * 1024 * 1024;
    const auto fits = [size](const Entry& entry) {
        if (!entry.buffer || entry.offset > std::numeric_limits<NSUInteger>::max() - 255) return false;
        const auto offset = (entry.offset + 255) & ~NSUInteger(255);
        return offset <= entry.buffer.length && size <= entry.buffer.length - offset;
    };
    const auto append = [&](Entry& entry) -> BufferSlice {
        const auto offset = (entry.offset + 255) & ~NSUInteger(255);
        scheduler.UseResource(entry.buffer);
        entry.offset = offset + size;
        return {entry.buffer, offset, size, static_cast<std::byte*>(entry.buffer.contents) + offset, entry.lease};
    };
    if (fits(entries[active_index])) return append(entries[active_index]);
    // After all retained pages filled in one unsubmitted batch, repeatedly
    // scanning their unfinished GPU ticks cannot produce more writable space.
    if (unavailable_tick != scheduler.CurrentTick() || size < unavailable_size) {
        Entry* replaceable = nullptr;
        for (std::size_t index = 0; index < entries.size(); ++index) {
            auto& entry = entries[index];
            if (fits(entry)) { active_index = index; return append(entry); }
            if (entry.buffer && (entry.lease.use_count() != 1 ||
                !scheduler.IsComplete(scheduler.LastUseTick(entry.buffer)))) continue;
            if (entry.buffer && entry.buffer.length >= size) {
                scheduler.UseResource(entry.buffer);
                entry.offset = 0; active_index = index;
                return append(entry);
            }
            if (!replaceable || entry.buffer.length > replaceable->buffer.length) replaceable = &entry;
        }
        NSUInteger capacity = 1024 * 1024;
        if (size <= budget) {
            while (capacity < size) capacity *= 2;
            if (replaceable && retained_bytes - replaceable->buffer.length <= budget - capacity) {
                auto buffer = device.CreateBuffer(capacity, true);
                scheduler.UseResource(buffer);
                retained_bytes = retained_bytes - replaceable->buffer.length + capacity;
                replaceable->buffer = buffer; replaceable->lease = std::make_shared<std::byte>();
                replaceable->offset = 0;
                active_index = std::size_t(replaceable - entries.data());
                return append(*replaceable);
            }
        }
        unavailable_tick = scheduler.CurrentTick();
        unavailable_size = size;
    }
    // Retention limits do not restrict guest input size or CPU lease count.
    // A full arena obtains separate native storage without waiting or truncating.
    auto buffer = device.CreateBuffer(size, true);
    scheduler.UseResource(buffer);
    return {buffer, 0, size, buffer.contents, {}};
}

ImmutableUniformCache::ImmutableUniformCache(const Device& device, Scheduler& scheduler_)
    : scheduler{scheduler_}, pool{device, scheduler_} {}

BufferSlice ImmutableUniformCache::Upload(unsigned slot, std::span<const std::byte> bytes) {
    (void)scheduler.LastUseTick(nil);
    if (slot >= entries.size() || bytes.empty()) {
        throw std::invalid_argument("Invalid Metal immutable uniform block");
    }
    auto& entry = entries[slot];
    // These shared ranges contain CPU-written, GPU-read-only constants. Compare
    // their immutable bytes directly, without a second CPU shadow or per-change
    // heap allocation. This must never be used for a GPU-written shader bank.
    if (entry.size == bytes.size() &&
        std::memcmp(entry.data, bytes.data(), bytes.size()) == 0) {
        // Reusing bytes in a later submission must renew their GPU use tick.
        scheduler.UseResource(entry.buffer);
        return entry;
    }
    // Finish all potentially throwing allocations before publishing a new
    // snapshot. Never overwrite the previous CPU- or GPU-consumed range.
    auto storage = pool.Allocate(bytes.size());
    std::memcpy(storage.data, bytes.data(), bytes.size());
    entry = std::move(storage);
    return entry;
}

void ImmutableUniformCache::Clear() {
    (void)scheduler.LastUseTick(nil);
    entries = {};
}

LutTextureStream::LutTextureStream(const Device& device_, Scheduler& scheduler_, MTLPixelFormat format_,
                                 NSUInteger page_bytes_)
    : device{device_}, scheduler{scheduler_}, pool{device_, scheduler_}, format{format_},
      page_bytes{page_bytes_}, element_bytes{format == MTLPixelFormatRG32Float ? 8u :
          format == MTLPixelFormatRGBA32Float ? 16u : 0u} {
    if (!element_bytes || !page_bytes ||
        page_bytes % std::max<NSUInteger>(256,
            [device.Handle() minimumTextureBufferAlignmentForPixelFormat:format])) {
        throw std::invalid_argument("Invalid native LUT texture stream page");
    }
}

bool LutTextureStream::BeginUpdate(NSUInteger maximum_bytes) {
    (void)scheduler.LastUseTick(nil);
    if (!maximum_bytes || maximum_bytes > page_bytes || maximum_bytes % element_bytes) {
        throw std::invalid_argument("Invalid native LUT update reservation");
    }
    const bool replaced = !texture || maximum_bytes > page_bytes - cursor;
    if (replaced) {
        auto next = pool.Allocate(page_bytes);
        auto view = device.CreateBufferTexture(next.buffer, next.offset, page_bytes / element_bytes, format);
        scheduler.UseResource(view);
        storage = std::move(next);
        texture = view;
        cursor = 0;
    } else {
        scheduler.UseResource(texture);
    }
    remaining = maximum_bytes;
    return replaced;
}

BufferSlice LutTextureStream::Append(NSUInteger bytes) {
    (void)scheduler.LastUseTick(nil);
    if (!texture || !bytes || bytes % element_bytes || bytes > remaining) {
        throw std::invalid_argument("Native LUT append exceeds its reservation");
    }
    scheduler.UseResource(texture);
    BufferSlice result{storage.buffer, storage.offset + cursor, bytes,
                      static_cast<std::byte*>(storage.data) + cursor, storage.lease};
    cursor += bytes;
    remaining -= bytes;
    return result;
}

NSUInteger LutTextureStream::TexelOffset(const BufferSlice& slice) const {
    (void)scheduler.LastUseTick(nil);
    if (!texture || slice.buffer != storage.buffer || slice.offset < storage.offset ||
        slice.offset - storage.offset > cursor || slice.size > cursor - (slice.offset - storage.offset) ||
        (slice.offset - storage.offset) % element_bytes) {
        throw std::invalid_argument("Native LUT slice is not in the current texture page");
    }
    return (slice.offset - storage.offset) / element_bytes;
}

void LutTextureStream::Clear() {
    (void)scheduler.LastUseTick(nil);
    storage = {};
    texture = nil;
    cursor = remaining = 0;
}

StreamBuffer::StreamBuffer(const Device& device_, Scheduler& scheduler_, NSUInteger page_size_)
    : device{device_}, scheduler{scheduler_}, page_size{page_size_} {
    pages.push_back({.buffer = device.CreateBuffer(page_size, true)});
}

BufferSlice StreamBuffer::Allocate(NSUInteger size, NSUInteger alignment) {
    if (!size || !alignment || (alignment & (alignment - 1))) {
        throw std::invalid_argument("Invalid Metal stream allocation size or alignment");
    }
    // Reject foreign callers before reading or changing CPU-owned page state.
    (void)scheduler.LastUseTick(nil);
    if (size > page_size) {
        // Oversized valid draws receive separate storage, retained by both the caller and
        // scheduler. It is never recycled into upload pages or silently truncated.
        auto buffer = device.CreateBuffer(size, true);
        scheduler.UseResource(buffer);
        return {buffer, 0, size, buffer.contents, std::make_shared<std::byte>()};
    }
    auto align_offset = [alignment](NSUInteger offset) {
        if (offset > std::numeric_limits<NSUInteger>::max() - (alignment - 1)) {
            throw std::overflow_error("Metal stream alignment overflow");
        }
        return (offset + alignment - 1) & ~(alignment - 1);
    };
    NSUInteger offset = align_offset(pages[page_index].offset);
    if (offset > page_size || size > page_size - offset) {
        // An in-flight page cannot be reset, including one used by the current unsubmitted batch.
        std::size_t reusable = pages.size();
        for (std::size_t i = 0; i < pages.size(); ++i) {
            if (pages[i].lease.use_count() == 1 &&
                scheduler.IsComplete(std::max(pages[i].tick, scheduler.LastUseTick(pages[i].buffer)))) {
                reusable = i;
                break;
            }
        }
        if (reusable == pages.size()) {
            // Bound memory consumption. Waiting flushes the current batch when necessary.
            if (pages.size() >= 3) {
                for (std::size_t i = 0; i < pages.size(); ++i) {
                    if (pages[i].lease.use_count() == 1) {
                        reusable = i;
                        break;
                    }
                }
                if (reusable == pages.size()) {
                    throw std::length_error("Metal upload pages are held by unconsumed CPU slices");
                }
                scheduler.Wait(std::max(pages[reusable].tick,
                                        scheduler.LastUseTick(pages[reusable].buffer)),
                               Scheduler::WaitReason::UploadPage);
            } else {
                pages.push_back({.buffer = device.CreateBuffer(page_size, true)});
            }
        }
        page_index = reusable;
        pages[page_index].offset = 0;
        offset = 0;
    }
    auto& page = pages[page_index];
    page.offset = offset + size;
    // UseResource may acquire a new submission slot, but does not advance the timeline.
    scheduler.UseResource(page.buffer);
    page.tick = scheduler.CurrentTick();
    return {page.buffer, offset, size, static_cast<std::byte*>(page.buffer.contents) + offset, page.lease};
}

} // namespace Metal
