// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_pica_shader_provider.h"
#include <nihstro/shader_bytecode.h>

namespace Metal::Test {

inline constexpr std::uint32_t IdentitySwizzle =
    15U | (0x1bU << 5U) | (0x1bU << 14U) | (0x1bU << 23U);

inline void InitializeFixtureUniforms(Pica::ShaderSetup& setup) {
    // ShaderSetup's common constructor does not initialize uniforms. Data-only
    // fixtures must define every value before writes inspect the previous bank
    // or the Metal adapter uploads it. This is not a guest/reset policy.
    for (auto& value : setup.uniforms.f)
        value = {Pica::f24::FromFloat32(0), Pica::f24::FromFloat32(0),
                 Pica::f24::FromFloat32(0), Pica::f24::FromFloat32(0)};
    setup.uniforms.b.fill(false);
    for (auto& value : setup.uniforms.i) value = {0, 0, 0, 0};
}

inline std::uint32_t Arithmetic(nihstro::OpCode::Id op, unsigned dest, unsigned src1,
                                 unsigned src2 = 0) {
    return (std::uint32_t(op) << 26U) | (dest << 21U) | (src1 << 12U) | (src2 << 7U);
}

inline void MovVertex(Pica::RegsInternal& regs, Pica::ShaderSetup& setup) {
    regs.vs.main_offset.Assign(0);
    regs.vs.output_mask.Assign(7);
    regs.rasterizer.vs_output_total.Assign(3);
    regs.rasterizer.vs_output_attributes[0].raw = 0x03020100; // Position.
    regs.rasterizer.vs_output_attributes[1].raw = 0x07060504; // Quaternion.
    regs.rasterizer.vs_output_attributes[2].raw = 0x0b0a0908; // Color.
    setup.UpdateSwizzleData(0, IdentitySwizzle);
    for (unsigned i = 0; i < 3; ++i) {
        setup.UpdateProgramCode(i, Arithmetic(nihstro::OpCode::Id::MOV, i, i));
    }
    setup.UpdateProgramCode(3, std::uint32_t(nihstro::OpCode::Id::END) << 26U);
}

} // namespace Metal::Test
