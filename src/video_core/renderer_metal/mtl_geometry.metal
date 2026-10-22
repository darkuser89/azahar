// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include <metal_stdlib>
using namespace metal;

struct CapturedVertex {
    float4 position;
    float4 color;
    float4 texcoord0;
    float4 texcoord1;
    float4 texcoord2;
    float4 quaternion;
    float4 view;
    float4 clip;
};

struct AssemblyParams {
    uint element_count;
    uint vertex_count;
    uint vertex_base;
    uint first_vertex;
    uint topology; // 0 list, 1 strip, 2 fan.
    uint index_size; // 0 nonindexed, 1 byte, 2 ushort, 4 uint.
    uint fix_quaternions;
    uint triangle_count;
};

uint source_index(const device uchar* indices, uint element, constant AssemblyParams& info) {
    if (info.index_size == 0) {
        return info.first_vertex + element;
    }
    const uint offset = element * info.index_size;
    uint value = uint(indices[offset]);
    if (info.index_size >= 2) {
        value |= uint(indices[offset + 1]) << 8;
    }
    if (info.index_size == 4) {
        value |= uint(indices[offset + 2]) << 16;
        value |= uint(indices[offset + 3]) << 24;
    }
    return value;
}

kernel void assemble_triangles(const device CapturedVertex* source [[buffer(0)]],
                                const device uchar* indices [[buffer(1)]],
                                constant AssemblyParams& info [[buffer(2)]],
                                device CapturedVertex* output [[buffer(3)]],
                                uint triangle [[thread_position_in_grid]]) {
    if (triangle >= info.triangle_count) {
        return;
    }
    uint3 elements;
    if (info.topology == 0) {
        elements = uint3(triangle * 3, triangle * 3 + 1, triangle * 3 + 2);
    } else if (info.topology == 1) {
        elements = uint3(triangle, triangle + 1, triangle + 2);
        if (triangle & 1u) {
            elements = elements.yxz;
        }
    } else {
        elements = uint3(0, triangle + 1, triangle + 2);
    }
    uint3 vertex_indices = uint3(source_index(indices, elements.x, info),
                                 source_index(indices, elements.y, info),
                                 source_index(indices, elements.z, info));
    if (any(vertex_indices < uint3(info.vertex_base)) ||
        any(vertex_indices - uint3(info.vertex_base) >= uint3(info.vertex_count))) {
        // An invalid guest index never reads outside capture storage. Clip the whole primitive.
        CapturedVertex invalid = {};
        invalid.position = float4(0.0f, 0.0f, 2.0f, 1.0f);
        invalid.clip = float4(-1.0f, -1.0f, 0.0f, 0.0f);
        output[triangle * 3] = invalid;
        output[triangle * 3 + 1] = invalid;
        output[triangle * 3 + 2] = invalid;
        return;
    }
    vertex_indices -= uint3(info.vertex_base);
    CapturedVertex a = source[vertex_indices.x];
    CapturedVertex b = source[vertex_indices.y];
    CapturedVertex c = source[vertex_indices.z];
    if (info.fix_quaternions != 0) {
        if (dot(a.quaternion, b.quaternion) < 0.0f) {
            b.quaternion = -b.quaternion;
        }
        if (dot(a.quaternion, c.quaternion) < 0.0f) {
            c.quaternion = -c.quaternion;
        }
    }
    output[triangle * 3] = a;
    output[triangle * 3 + 1] = b;
    output[triangle * 3 + 2] = c;
}

struct PrimitiveHistory {
    CapturedVertex vertices[2];
    uint index;
    uint ready;
    uint valid[2];
};

struct StreamParams {
    AssemblyParams assembly;
    uint reset;
    uint pending;
    uint shader_winding,padding;
};

// Address the prior history followed by this submission in chronological order.
// Strip history has alternating physical slots; fan history retains its center.
CapturedVertex stream_vertex(uint logical, const device CapturedVertex* source,
                              const device uchar* indices, constant StreamParams& info,
                              const device PrimitiveHistory& history, thread bool& valid) {
    if (logical < info.pending) {
        uint slot = logical;
        if (info.assembly.topology == 1 && info.pending == 2) slot ^= history.index;
        valid = info.reset == 0 && history.valid[slot] != 0;
        if (valid) return history.vertices[slot];
        return {};
    }
    const uint index = source_index(indices, logical - info.pending, info.assembly);
    valid = index >= info.assembly.vertex_base && index - info.assembly.vertex_base < info.assembly.vertex_count;
    if (valid) return source[index - info.assembly.vertex_base];
    return {};
}

// Address one raw component without materializing two private 128-byte vertices.
uint stream_history_word(uint logical,uint word,const device CapturedVertex* source,
    const device uchar* indices,constant StreamParams& info,
    const device PrimitiveHistory& history,thread bool& valid) {
    if(logical<info.pending) {
        uint slot=logical;
        if(info.assembly.topology==1 && info.pending==2) slot^=history.index;
        valid=info.reset==0 && history.valid[slot]!=0;
        return valid ? reinterpret_cast<const device uint*>(&history.vertices[slot])[word]:0u;
    }
    const uint index=source_index(indices,logical-info.pending,info.assembly);
    valid=index>=info.assembly.vertex_base && index-info.assembly.vertex_base<info.assembly.vertex_count;
    return valid ? reinterpret_cast<const device uint*>(source)[ulong(index-info.assembly.vertex_base)*32+word]:0u;
}
void write_primitive_history_word(const device CapturedVertex* source,
    const device uchar* indices,constant StreamParams& info,
    const device PrimitiveHistory& history,device PrimitiveHistory& next_history,uint word) {
    const uint total=info.pending+info.assembly.element_count;
    const uint remaining=info.assembly.topology==0 ? total%3:min(total,2u);
    const uint next_slot=info.assembly.topology==1 ?
        ((info.reset==0 ? history.index:0)+info.assembly.element_count)&1:remaining;
    bool valid_a=false,valid_b=false;
    uint a=0,b=0;
    if(remaining) {
        const uint first=info.assembly.topology==2 ? 0:total-remaining;
        a=stream_history_word(first,word,source,indices,info,history,valid_a);
        if(remaining==2) b=stream_history_word(total-1,word,source,indices,info,history,valid_b);
    }
    const uint slot=info.assembly.topology==1 && remaining==2 ? next_slot:0;
    reinterpret_cast<device uint*>(&next_history.vertices[slot])[word]=a;
    reinterpret_cast<device uint*>(&next_history.vertices[slot^1])[word]=b;
    if(!word) {
        next_history.valid[slot]=valid_a;next_history.valid[slot^1]=valid_b;
        next_history.index=info.assembly.topology==2 ? 1:next_slot;
        next_history.ready=info.assembly.topology!=0 && remaining==2;
    }
}

kernel void append_primitive_stream(const device CapturedVertex* source [[buffer(0)]],
                                    const device uchar* indices [[buffer(1)]],
                                    constant StreamParams& info [[buffer(2)]],
                                    const device PrimitiveHistory& history [[buffer(3)]],
                                    device CapturedVertex* output [[buffer(4)]],
                                    const device uint& winding [[buffer(5)]],
                                    device PrimitiveHistory& next_history [[buffer(6)]],
                                    uint triangle [[thread_position_in_grid]]) {
    // Each history component has one writer in the disjoint destination half.
    // All primitive readers use only the old half, including other workgroups.
    if(triangle<32) write_primitive_history_word(source,indices,info,history,next_history,triangle);
    if (triangle >= info.assembly.triangle_count) return;
    uint3 elements;
    if (info.assembly.topology == 0) {
        elements = uint3(triangle * 3, triangle * 3 + 1, triangle * 3 + 2);
    } else if (info.assembly.topology == 1) {
        elements = uint3(triangle, triangle + 1, triangle + 2);
        const uint prior_parity = info.pending == 2 && info.reset == 0 ? history.index : 0;
        if ((triangle ^ prior_parity) & 1) elements = elements.yxz;
    } else {
        elements = uint3(0, triangle + 1, triangle + 2);
    }
    if (info.shader_winding && winding && triangle==0) elements=elements.yxz;
    bool valid_a, valid_b, valid_c;
    CapturedVertex a = stream_vertex(elements.x, source, indices, info, history, valid_a);
    CapturedVertex b = stream_vertex(elements.y, source, indices, info, history, valid_b);
    CapturedVertex c = stream_vertex(elements.z, source, indices, info, history, valid_c);
    if (!valid_a || !valid_b || !valid_c) {
        a = {};
        a.position = float4(0, 0, 2, 1);
        a.clip = float4(-1, -1, 0, 0);
        b = c = a;
    } else if (info.assembly.fix_quaternions != 0) {
        if (dot(a.quaternion, b.quaternion) < 0) b.quaternion = -b.quaternion;
        if (dot(a.quaternion, c.quaternion) < 0) c.quaternion = -c.quaternion;
    }
    output[triangle * 3] = a;
    output[triangle * 3 + 1] = b;
    output[triangle * 3 + 2] = c;
}


struct RasterVertex {
    float4 position [[position, invariant]];
    float4 primary_color [[user(locn1)]];
    float2 texcoord0 [[user(locn2)]];
    float2 texcoord1 [[user(locn3)]];
    float2 texcoord2 [[user(locn4)]];
    float texcoord0_w [[user(locn5)]];
    float4 normquat [[user(locn6)]];
    float3 view [[user(locn7)]];
    float clip_distance [[clip_distance]] [2];
};

vertex RasterVertex raster_vertex(const device CapturedVertex* vertices [[buffer(3)]],
                                  uint index [[vertex_id]]) {
    const auto vtx = vertices[index];
    RasterVertex output;
    // Capture retains the PICA/Vulkan clip-space orientation. Convert Y once, at rasterization.
    output.position = float4(vtx.position.x, -vtx.position.y, vtx.position.zw);
    output.primary_color = vtx.color;
    output.texcoord0 = vtx.texcoord0.xy;
    output.texcoord1 = vtx.texcoord1.xy;
    output.texcoord2 = vtx.texcoord2.xy;
    output.texcoord0_w = vtx.clip.z;
    output.normquat = vtx.quaternion;
    output.view = vtx.view.xyz;
    output.clip_distance[0] = vtx.clip.x;
    output.clip_distance[1] = vtx.clip.y;
    return output;
}
