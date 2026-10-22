// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Matches ShaderMachineUniforms; GS transport stays in its own small library.
struct MachineUniforms { float4 f[96]; uint4 i[4]; uint b,pad0,pad1,pad2; };
static_assert(sizeof(MachineUniforms)==1616,"Native GS uniform layout");
struct GeometryInputParams {
    uint mode,mask,count,destination,ready,input_count;
    uint registers[16],begin_count,variable_count;
};
static_assert(sizeof(GeometryInputParams)==96,"Native GS input parameter layout");
kernel void pica_geometry_inputs(const device float4* source [[buffer(0)]],
    device float4* point [[buffer(1)]],device MachineUniforms& uniforms [[buffer(2)]],
    device float4* inputs [[buffer(3)]],constant GeometryInputParams& info [[buffer(4)]]) {
    if (info.begin_count) uniforms.f[0]=float4(float(info.variable_count));
    uint copied=0;
    for (uint reg=0;reg<16 && copied<info.count;++reg) if (info.mask&(1u<<reg)) {
        if (info.mode==0) point[info.destination+copied]=source[reg];
        else uniforms.f[info.destination+copied]=source[reg];
        ++copied;
    }
    if (info.mode==0 && info.ready)
        for (uint attribute=0;attribute<info.input_count;++attribute)
            inputs[info.registers[attribute]]=point[attribute];
}
// Staged records contain only the last explicit CPU writer for each requested
// slot. Masked application remains here, preserving all GPU-owned bank values.
struct GeometryUniformWrite {
 uint float_masks[3];
 uint bool_mask,integer_mask,boolean_value_offset,data_offset,float_blocks;
};
static_assert(sizeof(GeometryUniformWrite)==32,"Native GS uniform write layout");
kernel void pica_geometry_uniform_batch(constant GeometryUniformWrite& update [[buffer(0)]],
 device MachineUniforms& dest [[buffer(1)]],
 const device uint4* values [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
 if(gid<96) {
  if(update.float_masks[gid/32]&(1u<<(gid%32))) {
   const uint index=update.data_offset+popcount(update.float_blocks&((1u<<(gid/8))-1))*8+gid%8;
   dest.f[gid]=as_type<float4>(values[index]);
  }
 } else if(gid<100) {
  const uint reg=gid-96;
  if(update.integer_mask&(1u<<reg)) {
   const uint index=update.data_offset+popcount(update.float_blocks)*8+popcount(update.integer_mask&((1u<<reg)-1));
   dest.i[reg]=values[index];
  }
 } else if(gid==100) {
  uint value=dest.b;
  for(uint bit=0;bit<16;++bit) if(update.bool_mask&(1u<<bit)) {
   const uint source=values[update.boolean_value_offset+bit/4][bit%4];
   value=(value&~(1u<<bit))|(source&(1u<<bit));
  }
  dest.b=value;
 }
}
kernel void pica_geometry_invoked(device MachineUniforms& uniforms [[buffer(0)]]) {
    uniforms.b|=0x8000u;
}
