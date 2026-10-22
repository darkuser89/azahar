// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <cstddef>

namespace Metal {
inline constexpr std::size_t LegacyStateBytes=816;
inline constexpr std::size_t ShaderStateBlockBytes=1840;
inline constexpr std::size_t GeometryStateOffset=LegacyStateBytes+2*ShaderStateBlockBytes;
inline constexpr std::size_t StateArchiveBytes=GeometryStateOffset+128+2128;
static_assert(StateArchiveBytes==6752);
}
