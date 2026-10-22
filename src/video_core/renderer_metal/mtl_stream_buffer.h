// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_scheduler.h"
#include <vector>

namespace Metal {

struct BufferSlice {
    id<MTLBuffer> buffer;
    NSUInteger offset;
    NSUInteger size;
    void* data;
    // An outstanding CPU slice prevents its page from being recycled before consumers are encoded.
    std::shared_ptr<void> lease;
};

// Append disjoint immutable CPU uploads without waiting. Resetting a page
// requires released CPU leases AND completion of its last GPU submission.
class SharedUploadPool final {
public:
    SharedUploadPool(const Device& device, Scheduler& scheduler);
    BufferSlice Allocate(NSUInteger size);

private:
    struct Entry {
        id<MTLBuffer> buffer;
        std::shared_ptr<void> lease;
        NSUInteger offset{};
    };
    const Device& device;
    Scheduler& scheduler;
    std::array<Entry, 16> entries{};
    NSUInteger retained_bytes{};
    std::size_t active_index{};
    std::uint64_t unavailable_tick{};
    NSUInteger unavailable_size{};
};

// CPU-owned draw constants only. Each slot holds its latest immutable byte
// snapshot; outstanding slices and GPU submissions retain earlier versions.
// A separate append arena avoids pinning the three general upload pages.
class ImmutableUniformCache final {
public:
    ImmutableUniformCache(const Device& device, Scheduler& scheduler);
    BufferSlice Upload(unsigned slot, std::span<const std::byte> bytes);
    void Clear();

private:
    Scheduler& scheduler;
    SharedUploadPool pool;
    std::array<BufferSlice, 4> entries{};
};

// Append-only native LUT texture-buffer pages. A page change invalidates every
// logical table offset, so the caller republishes all tables for that texture.
class LutTextureStream final {
public:
    LutTextureStream(const Device& device, Scheduler& scheduler, MTLPixelFormat format,
                     NSUInteger page_bytes = 1024 * 1024);
    bool BeginUpdate(NSUInteger maximum_bytes);
    BufferSlice Append(NSUInteger bytes);
    id<MTLTexture> Texture() const { return texture; }
    NSUInteger TexelOffset(const BufferSlice& slice) const;
    void Clear();

private:
    const Device& device;
    Scheduler& scheduler;
    SharedUploadPool pool;
    MTLPixelFormat format;
    NSUInteger page_bytes, element_bytes, cursor{}, remaining{};
    BufferSlice storage{};
    id<MTLTexture> texture;
};

// Shared upload pages are never overwritten before the last submission using them completes.
class StreamBuffer final {
public:
    StreamBuffer(const Device& device, Scheduler& scheduler, NSUInteger page_size);
    BufferSlice Allocate(NSUInteger size, NSUInteger alignment = 256);

private:
    struct Page {
        id<MTLBuffer> buffer;
        NSUInteger offset{};
        std::uint64_t tick{};
        std::shared_ptr<void> lease{std::make_shared<std::byte>()};
    };

    const Device& device;
    Scheduler& scheduler;
    const NSUInteger page_size;
    std::vector<Page> pages;
    std::size_t page_index{};
};

} // namespace Metal
