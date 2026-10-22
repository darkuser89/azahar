// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <array>
#include <cstdint>

namespace Metal {
// Version 1: 40 little-endian uint32 words, followed without padding by code,
// swizzles, uniforms, initial history, fetched vertices, indices, raw VS outputs,
// and final history. Both histories contain the complete 1824-byte GPU unit.
// Reset histories are canonical zero. Float payloads are copied as raw bits.
// This is a VS-stage diagnostic; it does not capture a complete render frame.
struct VertexDiagnostic {
    enum Field : unsigned {
        Magic, Version, HeaderBytes, CodeWords, SwizzleWords, UniformBytes, StateBytes,
        VertexBytes, Entry, OutputMask, AllowEmission, Execution, Passes,
        RecurrenceLow, RecurrenceHigh, RecurrenceCapacity,
        Vertices, Elements, VertexBase, FirstVertex, IndexWidth, InstructionLimit,
        InitialReset, FinalReset, FrameLow, FrameHigh, DrawLow, DrawHigh,
        TickLow, TickHigh, InputBytes, IndexBytes, OutputBytes, InputIndexAlias,
        RequestLow, RequestHigh, ProgramLow, ProgramHigh, ControlLow, ControlHigh
    };
    static constexpr std::uint32_t Signature=0x34434456; // VDC4
    using Header=std::array<std::uint32_t,40>;
};
}
