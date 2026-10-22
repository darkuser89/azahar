// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <array>
#include <cstdint>

namespace Metal {
// Membership lookup is independent of FIFO replacement: hits never refresh a slot.
class IndexFifo final {
public:
    bool Miss(std::uint32_t id) noexcept {
        const auto key = std::uint64_t(id) + 1;
        auto slot = Hash(id);
        while (entries[slot]) {
            if (entries[slot] == key) return false;
            slot = (slot + 1) & Mask;
        }
        if (size == 64) {
            Erase(order[next]);
            slot = Hash(id);
            while (entries[slot]) slot = (slot + 1) & Mask;
        } else {
            ++size;
        }
        entries[slot] = key;
        order[next] = id;
        next = (next + 1) & 63;
        return true;
    }

private:
    static constexpr unsigned Mask = 127;
    static unsigned Hash(std::uint32_t id) noexcept {
        return (id * std::uint32_t{0x9e3779b9}) >> 25;
    }
    // At most64 of128 entries are occupied. Close the deleted probe-chain
    // hole instead of accumulating tombstones across repeated FIFO wraps.
    void Erase(std::uint32_t id) noexcept {
        const auto key = std::uint64_t(id) + 1;
        auto hole = Hash(id);
        while (entries[hole] != key) hole = (hole + 1) & Mask;
        auto scan = (hole + 1) & Mask;
        while (entries[scan]) {
            const auto home = Hash(std::uint32_t(entries[scan] - 1));
            if (((hole - home) & Mask) < ((scan - home) & Mask)) {
                entries[hole] = entries[scan];
                hole = scan;
            }
            scan = (scan + 1) & Mask;
        }
        entries[hole] = 0;
    }
    // Zero marks an empty slot; uint64(id)+1 also represents UINT32_MAX.
    std::array<std::uint64_t, 128> entries{};
    std::array<std::uint32_t, 64> order{};
    unsigned next{}, size{};
};
} // namespace Metal
