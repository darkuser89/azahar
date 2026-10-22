// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include <metal_stdlib>
using namespace metal;

struct FetchAttribute {
    uint buffer_index;
    uint offset;
    uint stride;
    uint components;
    uint format;
    uint register_index;
    uint2 padding;
};
struct FetchParams {
    uint vertex_count;
    uint first_vertex;
    uint attribute_count;
    uint padding;
    FetchAttribute attributes[16];
    float4 fixed[16];
};

uint read_le(const device uchar* input, ulong offset, uint bytes) {
    uint value = 0;
    for (uint i = 0; i < bytes; ++i) value |= uint(input[offset + i]) << (i * 8u);
    return value;
}

float4 load_attribute(const device uchar* source, FetchAttribute attr, ulong input_id) {
    const uint bytes = attr.format == 3 ? 4 : attr.format == 2 ? 2 : 1;
    const ulong base = ulong(attr.offset) + input_id * attr.stride;
    float4 value{0, 0, 0, 1};
    for (uint c = 0; c < attr.components; ++c) {
        const uint raw = read_le(source, base + c * bytes, bytes);
        switch (attr.format) {
        case 0: value[c] = float(raw < 128u ? int(raw) : int(raw) - 256); break;
        case 1: value[c] = float(raw); break;
        case 2: value[c] = float(raw < 32768u ? int(raw) : int(raw) - 65536); break;
        case 3: value[c] = as_type<float>(raw); break;
        }
    }
    return value;
}

kernel void fetch_vertices(
    const device uchar* b0 [[buffer(0)]], const device uchar* b1 [[buffer(1)]],
    const device uchar* b2 [[buffer(2)]], const device uchar* b3 [[buffer(3)]],
    const device uchar* b4 [[buffer(4)]], const device uchar* b5 [[buffer(5)]],
    const device uchar* b6 [[buffer(6)]], const device uchar* b7 [[buffer(7)]],
    const device uchar* b8 [[buffer(8)]], const device uchar* b9 [[buffer(9)]],
    const device uchar* b10 [[buffer(10)]], const device uchar* b11 [[buffer(11)]],
    const device uchar* b12 [[buffer(12)]], const device uchar* b13 [[buffer(13)]],
    const device uchar* b14 [[buffer(14)]], const device uchar* b15 [[buffer(15)]],
    device float4* output [[buffer(16)]], constant FetchParams& info [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= info.vertex_count) return;
    const device uchar* sources[16] = {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15};
    for (uint reg = 0; reg < 16; ++reg) output[gid * 16u + reg] = info.fixed[reg];
    const ulong vertex_index = ulong(info.first_vertex) + gid;
    for (uint i = 0; i < info.attribute_count; ++i) {
        const FetchAttribute attr = info.attributes[i];
        const device uchar* source = sources[attr.buffer_index];
        output[gid * 16u + attr.register_index] = load_attribute(source, attr, vertex_index);
    }
}

struct PicaFetchParams {
    uint vertex_count, mapped_count, reset, padding;
    FetchAttribute attributes[16]; // register_index is the PicaAttributeMode here.
    uint register_map[16];
    float4 fixed[16];
};
struct InputHistory { float4 attributes[16]; float4 registers[16]; };

kernel void fetch_pica_vertices(
    const device uchar* b0 [[buffer(0)]], const device uchar* b1 [[buffer(1)]],
    const device uchar* b2 [[buffer(2)]], const device uchar* b3 [[buffer(3)]],
    const device uchar* b4 [[buffer(4)]], const device uchar* b5 [[buffer(5)]],
    const device uchar* b6 [[buffer(6)]], const device uchar* b7 [[buffer(7)]],
    const device uchar* b8 [[buffer(8)]], const device uchar* b9 [[buffer(9)]],
    const device uchar* b10 [[buffer(10)]], const device uchar* b11 [[buffer(11)]],
    const device uchar* b12 [[buffer(12)]], const device uchar* b13 [[buffer(13)]],
    const device uchar* b14 [[buffer(14)]], const device uchar* b15 [[buffer(15)]],
    device float4* output [[buffer(16)]], constant PicaFetchParams& info [[buffer(17)]],
    const device InputHistory& history [[buffer(18)]], device float4* attributes [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= info.vertex_count) return;
    const device uchar* sources[16] = {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15};
    float4 values[16];
    for (uint attribute = 0; attribute < 16; ++attribute) {
        const FetchAttribute input = info.attributes[attribute];
        float4 value = info.reset ? float4(0) : history.attributes[attribute];
        if (input.register_index == 1) value = info.fixed[attribute];
        else if (input.register_index == 2) value = load_attribute(sources[input.buffer_index], input, gid);
        values[attribute] = value;
        attributes[gid * 16u + attribute] = value;
    }
    float4 registers[16];
    for (uint reg = 0; reg < 16; ++reg) registers[reg] = info.reset ? float4(0) : history.registers[reg];
    for (uint attribute = 0; attribute < info.mapped_count; ++attribute)
        registers[info.register_map[attribute]] = values[attribute];
    for (uint reg = 0; reg < 16; ++reg) output[gid * 16u + reg] = registers[reg];
}

struct HistoryParams {
    uint element_count, vertex_count, vertex_base, first_vertex;
    uint index_size, reset, emit_plan, padding1;
};
// Shared with the native shader machine: cached source and preceding miss.
struct InputVertexPlan {uint cached,previous;};
kernel void update_pica_input_history(const device float4* registers [[buffer(16)]],
    constant HistoryParams& info [[buffer(17)]], device InputHistory& history [[buffer(18)]],
    const device float4* attributes [[buffer(19)]],const device uchar* indices [[buffer(20)]],
    device InputVertexPlan* plan [[buffer(21)]],device uint4& status [[buffer(22)]],
    uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    uint last=0;bool found=false;
    const bool shared=info.emit_plan!=0;
    if(!info.index_size) {
        const uint id=info.first_vertex+info.element_count-1u;
        last=id-info.vertex_base;found=id>=info.vertex_base && last<info.vertex_count;
    } else {
        uint4 keys=uint4(0),values=uint4(~0u);
        uint previous=~0u,replacement=0,replacement_lane=0,replacement_slot=0;
        bool plan_live=true;
        for(ulong base=0;base<info.element_count;base+=width) {
            InputVertexPlan own={~0u,~0u};
            uint prefetched=0;
            if(base+lane<info.element_count) prefetched=read_le(indices,(base+lane)*info.index_size,info.index_size);
            const uint count=min(ulong(width),ulong(info.element_count)-base);
            uint plan_count=plan_live ? count:0u;
            for(uint offset=0;offset<count;++offset) {
                const uint element=uint(base)+offset,id=simd_broadcast(prefetched,offset);
                if(id<info.vertex_base || id-info.vertex_base>=info.vertex_count) {
                    if(shared && plan_live) {
                        if(lane==0) status=uint4(8,element,0,0);
                        plan_count=offset;plan_live=false;
                    }
                    // Fetch history continues to the last VALID miss, as before.
                    // The shader plan retains only the prefix before its first error.
                    continue;
                }
                uint matching=keys.x==id ? values.x:~0u;
                if(width<64) matching=min(matching,keys.y==id ? values.y:~0u);
                if(width<32) matching=min(matching,min(keys.z==id ? values.z:~0u,keys.w==id ? values.w:~0u));
                const uint cached=simd_min(matching);
                if(shared && plan_live && lane==offset) own={cached,previous};
                if(cached!=~0u) continue;
                previous=element;
                if(lane==replacement_lane) {keys[replacement_slot]=id;values[replacement_slot]=element;}
                ++replacement;++replacement_lane;
                if(replacement==64) {replacement=0;replacement_lane=0;replacement_slot=0;}
                else if(replacement_lane==width) {replacement_lane=0;++replacement_slot;}
                last=id-info.vertex_base;found=true;
            }
            if(shared && lane<plan_count) plan[uint(base)+lane]=own;
        }
        // The independent finalizer consumes this instead of searching again.
        if(shared && lane==0) plan[info.element_count]={plan_live ? previous:~0u,~0u};
    }
    if(lane>=16) return;
    if(found) {
        history.attributes[lane]=attributes[ulong(last)*16+lane];
        history.registers[lane]=registers[ulong(last)*16+lane];
    } else if(info.reset) {
        history.attributes[lane]=float4(0);history.registers[lane]=float4(0);
    }
}
