// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include "mtl_state.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace Metal {

// Raw Gas state, independent of color conversion and shader/cache lifetime.
// The documented table consists of eight difference words and eight color words.
struct GasTransferState {
    std::array<std::uint32_t, 16> lut{};
    std::uint32_t cursor{};
    bool feedback_pending{};
    std::uint32_t maximum{};

    bool Empty() const { return *this == GasTransferState{}; }
    bool operator==(const GasTransferState&) const = default;

    void Write(std::uint32_t reg, std::span<const std::uint32_t> values,
               std::uint32_t mask) {
        if (reg < 0x123 || reg > 0x125 || mask > 15 || values.empty())
            throw std::invalid_argument("Invalid native Gas register transport");
        std::uint32_t bytes{};
        for (unsigned i = 0; i < 4; ++i)
            if (mask & (1u << i)) bytes |= 0xffu << (8 * i);
        if (reg == 0x123) {
            for (auto value : values) cursor = ((cursor & ~bytes) | (value & bytes)) & 0xffff;
        } else if (reg == 0x124) {
            // Hardware behavior for masked FIFO writes and out-of-range indices has
            // not been established. Never alias them to another table entry.
            if (mask != 15)
                throw std::runtime_error("Masked native Gas LUT FIFO writes require hardware validation");
            if (cursor > lut.size() || values.size() > lut.size() - cursor)
                throw std::runtime_error("Native Gas LUT FIFO index is outside the documented table");
            std::copy(values.begin(), values.end(), lut.begin() + cursor);
            cursor += static_cast<std::uint32_t>(values.size());
        } else {
            // A reset is an ordered action even when its value equals the old value.
            feedback_pending = true;
            maximum = 0;
        }
    }
};

// Keep the existing fixed-size PICA archive envelope. Version 3 removes only
// canonical padding and packs five small header fields, leaving room for Gas.
// Version 4 uses the last reserved word for the nonnegative half maximum.
// Version 1/2/3 blobs remain readable without changing the shared archive schema.
inline constexpr std::size_t GasArchiveOffset = StateArchiveBytes - 76;

inline bool GasArchiveOmitted(std::size_t offset) {
    if (offset >= 16 && offset < 32) return true;
    for (auto base : {LegacyStateBytes, LegacyStateBytes + ShaderStateBlockBytes}) {
        if ((offset >= base + 4 && offset < base + 16) ||
            (offset >= base + 16 + 1804 && offset < base + 16 + 1808)) return true;
    }
    return (offset >= GeometryStateOffset + 112 && offset < GeometryStateOffset + 128) ||
           (offset >= GeometryStateOffset + 128 + 1604 &&
            offset < GeometryStateOffset + 128 + 1616);
}

inline std::uint32_t GasArchiveWord(std::span<const std::uint8_t> data, std::size_t offset) {
    std::uint32_t value{};
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t(data[offset + i]) << (8 * i);
    return value;
}
inline void GasArchiveWord(std::span<std::uint8_t> data, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data[offset + i] = std::uint8_t(value >> (8 * i));
}

inline std::vector<std::uint8_t> PackGasArchive(std::span<const std::uint8_t> original,
                                             const GasTransferState& gas) {
    if (original.size() != StateArchiveBytes || GasArchiveWord(original, 8) != 2)
        throw std::invalid_argument("Gas packing requires a version 2 native snapshot");
    if (gas.cursor > 0xffff || gas.maximum > 0x7c00 || (gas.feedback_pending && gas.maximum))
        throw std::invalid_argument("Invalid native Gas snapshot cursor/maximum/reset");
    const auto topology = GasArchiveWord(original, 12), pending = GasArchiveWord(original, 16);
    if (topology > 2 || pending > 2 || GasArchiveWord(original, 20) > 1 ||
        GasArchiveWord(original, 24) > 1 || GasArchiveWord(original, 28) > 1)
        throw std::invalid_argument("Invalid native snapshot header during Gas packing");
    std::vector<std::uint8_t> result(StateArchiveBytes);
    std::size_t dest{};
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (!GasArchiveOmitted(i)) result[dest++] = original[i];
        else if (i >= 32 && original[i])
            throw std::invalid_argument("Gas packing would discard nonzero native padding");
    }
    if (dest != GasArchiveOffset) throw std::logic_error("Invalid compact native Gas archive size");
    GasArchiveWord(result, 8, gas.maximum ? 4 : 3);
    GasArchiveWord(result, 12, topology | (pending << 2) |
        (GasArchiveWord(original, 20) << 4) | (GasArchiveWord(original, 24) << 5) |
        (GasArchiveWord(original, 28) << 6));
    for (unsigned i = 0; i < gas.lut.size(); ++i)
        GasArchiveWord(result, GasArchiveOffset + i * 4, gas.lut[i]);
    GasArchiveWord(result, GasArchiveOffset + 64, gas.cursor);
    GasArchiveWord(result, GasArchiveOffset + 68, gas.feedback_pending);
    GasArchiveWord(result, GasArchiveOffset + 72, gas.maximum);
    return result;
}

inline std::vector<std::uint8_t> UnpackGasArchive(std::span<const std::uint8_t> compact,
                                               GasTransferState& gas) {
    if (compact.size() != StateArchiveBytes ||
        (GasArchiveWord(compact, 8) != 3 && GasArchiveWord(compact, 8) != 4))
        throw std::invalid_argument("Invalid compact native Gas snapshot");
    const auto flags = GasArchiveWord(compact, 12);
    const auto maximum = GasArchiveWord(compact, GasArchiveOffset + 72);
    if (flags > 127 || (flags & 3) > 2 || ((flags >> 2) & 3) > 2 ||
        GasArchiveWord(compact, GasArchiveOffset + 68) > 1 ||
        maximum > 0x7c00 || (maximum &&
        (GasArchiveWord(compact, 8) == 3 || GasArchiveWord(compact, GasArchiveOffset + 68))))
        throw std::invalid_argument("Invalid compact native Gas snapshot flags/padding");
    GasTransferState decoded{};
    for (unsigned i = 0; i < decoded.lut.size(); ++i)
        decoded.lut[i] = GasArchiveWord(compact, GasArchiveOffset + i * 4);
    decoded.cursor = GasArchiveWord(compact, GasArchiveOffset + 64);
    if (decoded.cursor > 0xffff) throw std::invalid_argument("Invalid Gas snapshot cursor");
    decoded.feedback_pending = GasArchiveWord(compact, GasArchiveOffset + 68) != 0;
    decoded.maximum = maximum;
    std::vector<std::uint8_t> result(StateArchiveBytes);
    std::size_t source{};
    for (std::size_t i = 0; i < result.size(); ++i)
        if (!GasArchiveOmitted(i)) result[i] = compact[source++];
    GasArchiveWord(result, 8, 2);
    GasArchiveWord(result, 12, flags & 3);
    GasArchiveWord(result, 16, (flags >> 2) & 3);
    GasArchiveWord(result, 20, (flags >> 4) & 1);
    GasArchiveWord(result, 24, (flags >> 5) & 1);
    GasArchiveWord(result, 28, (flags >> 6) & 1);
    gas = decoded;
    return result;
}
} // namespace Metal
