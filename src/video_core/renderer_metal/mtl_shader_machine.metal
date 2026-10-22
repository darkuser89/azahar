// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct MachineUnit {
    float4 input[16],temporary[16],output[2][16],emit_buffer[3][16];
    int4 address;
    uint4 flags; // condition X/Y, output bank, SETEMIT bits.
};
struct MachineUniforms { float4 f[96]; uint4 i[4]; uint b,pad0,pad1,pad2; };
struct MachineParams { uint entry,output_mask,input_mask,reset,capacity,limit,allow_emission,pad1; };
struct Emission { float4 attributes[16]; uint winding,pad0,pad1,pad2; };
struct FlowFrame { uint end,resume,begin,remaining,increment; };
void machine_push(thread FlowFrame* stack,thread uint& depth,uint capacity,FlowFrame frame) {
    if (depth==capacity) for(uint i=1;i<capacity;++i) stack[i-1]=stack[i];
    else ++depth;
    stack[depth-1]=frame;
}
struct GeometryEmissionConfig {
    uint semantics[7],attribute_count;
    float4 clip_coefficient;
    uint flip_viewport,enable_clip,fix_quaternions,honor_winding;
};
struct GeometryEmissionParams { GeometryEmissionConfig config; uint4 count; };
struct MachineCapturedVertex {
    float4 position,color,texcoord0,texcoord1,texcoord2,quaternion,view,clip;
};

MachineCapturedVertex machine_capture(thread const Emission& input,
    constant GeometryEmissionConfig& info) {
    float slots[32];
    for (uint i=0;i<32;++i) slots[i]=1;
    for (uint attribute=0;attribute<info.attribute_count;++attribute)
        for (uint component=0;component<4;++component)
            slots[(info.semantics[attribute]>>(8*component))&31]=input.attributes[attribute][component];
    float4 position=float4(slots[0],slots[1],slots[2],slots[3]);
    const float ndc=position.z/position.w;
    if (ndc>0 && ndc<.000001f) position.z=0;
    if (ndc<-1 && ndc>-1.00001f) position.z=-position.w;
    if (info.flip_viewport) position.y=-position.y;
    MachineCapturedVertex result={};
    result.position=float4(position.xy,-position.z,position.w);
    const float4 color=float4(slots[8],slots[9],slots[10],slots[11]);
    for (uint i=0;i<4;++i) {
        const float magnitude=abs(color[i]);
        // A saturation instruction maps NaN to zero. PICA's comparison maps it to one.
        result.color[i]=isnan(color[i]) ? 1 : magnitude<1 ? magnitude : 1;
    }
    result.texcoord0=float4(slots[12],slots[13],0,0);
    result.texcoord1=float4(slots[14],slots[15],0,0);
    result.texcoord2=float4(slots[22],slots[23],0,0);
    result.quaternion=float4(slots[4],slots[5],slots[6],slots[7]);
    result.view=float4(slots[18],slots[19],slots[20],0);
    result.clip=float4(-position.z,info.enable_clip ? dot(info.clip_coefficient,position) : 0,slots[16],0);
    return result;
}

kernel void pica_geometry_emissions(const device Emission* source [[buffer(0)]],
    constant GeometryEmissionParams& info [[buffer(1)]],
    device MachineCapturedVertex* output [[buffer(2)]],
    device uint* next_history [[buffer(3)]],const device uint4& draw_arguments [[buffer(4)]],
    uint triangle [[thread_position_in_grid]]) {
    const uint vertices=info.count.w ? draw_arguments.x:info.count.x;
    // Every nonempty complete list leaves the same 272 canonical zero bytes
    // as appending its triangles separately. One writer, no old-history reads.
    if (!triangle && info.count.z) for (uint word=0;word<68;++word) next_history[word]=0;
    if (triangle>=vertices/3) return;
    const uint base=triangle*3;
    const bool reverse=info.config.honor_winding && source[base].winding;
    const Emission first=source[base+(reverse ? 1:0)],second=source[base+(reverse ? 0:1)],third=source[base+2];
    auto a=machine_capture(first,info.config);
    auto b=machine_capture(second,info.config);
    auto c=machine_capture(third,info.config);
    if (info.config.fix_quaternions) {
        if (dot(a.quaternion,b.quaternion)<0) b.quaternion=-b.quaternion;
        if (dot(a.quaternion,c.quaternion)<0) c.quaternion=-c.quaternion;
    }
    output[base]=a;output[base+1]=b;output[base+2]=c;
}

// 64 bytes: MTLDrawPrimitivesIndirectArguments, dispatch groups, conditional
// primitive metadata, then validation. Counts are bounded before any consumer.
kernel void pica_geometry_count(const device uint4& status [[buffer(0)]],
    constant uint4& info [[buffer(1)]],device uint4* arguments [[buffer(2)]],
    device uint* history [[buffer(3)]],const device uint4* previous [[buffer(4)]]) {
    uint4 state=info.w ? previous[2]:uint4(info.y,info.z,0,0);
    const uint invalid=!status.x && (status.y>info.x || status.y%3u);
    const uint count=status.x || invalid ? 0u:status.y;
    if(count) {
        state.x=0;state.y^=((count/3u)&1u)*272u;
        for(uint word=0;word<68;++word) history[state.y/4u+word]=0;
    }
    arguments[0]=uint4(count,1,0,0);
    arguments[1]=uint4(max(1u,(count+191u)/192u),1,1,0);
    arguments[2]=state;
    arguments[3]=uint4(invalid ? 8u:0u,0,0,0);
}
kernel void pica_vertex_capture(const device float4* source [[buffer(0)]],
    constant GeometryEmissionParams& info [[buffer(1)]],device MachineCapturedVertex* output [[buffer(2)]],
    uint vertex_id [[thread_position_in_grid]]) {
    if (vertex_id>=info.count.x) return;
    Emission captured={};uint attribute=0;
    for(uint reg=0;reg<16;++reg) if(info.count.y&(1u<<reg)) captured.attributes[attribute++]=source[vertex_id*16u+reg];
    output[vertex_id]=machine_capture(captured,info.config);
}

// PICA checks each stack against the fall-through PC before applying a jump.
// Each stack receives its own PC copy; LOOP has priority over IF over CALL.
uint machine_advance(uint next,uint jump,bool breaking,thread FlowFrame* calls,
    thread FlowFrame* branches,thread FlowFrame* loops,thread uint& call_depth,
    thread uint& branch_depth,thread uint& loop_depth,thread MachineUnit& unit) {
    uint result=jump;
    uint call_pc=next;
    uint call_result=next,pops=0;
    bool call_pop=false;
    while (call_depth && call_pc==calls[call_depth-1].end) {
        call_pc=calls[--call_depth].resume;
        // The fourth consecutive CALL return loses its PC update on PICA.
        if(++pops<4) call_result=call_pc;
        call_pop=true;
    }
    if (call_pop) result=call_result;
    if (branch_depth && next==branches[branch_depth-1].end)
        result=branches[--branch_depth].resume;
    if (loop_depth && (breaking || next==loops[loop_depth-1].end)) {
        thread FlowFrame& frame=loops[loop_depth-1];
        if (!breaking) unit.address.z+=int(frame.increment);
        if (!breaking && --frame.remaining) result=frame.begin;
        else { result=frame.resume;--loop_depth; }
    }
    return result;
}

float4 machine_source(uint reg,uint address_index,thread MachineUnit& unit,
                      constant MachineUniforms& uniforms) {
    if (reg<16) return unit.input[reg];
    if (reg<32) return unit.temporary[reg-16];
    int offset=address_index ? unit.address[address_index-1] : 0;
    if (offset<-128 || offset>127) offset=0;
    const uint index=uint(int(reg-32)+offset)&127u;
    return index<96 ? uniforms.f[index] : float4(1);
}
float4 machine_swizzle(float4 value,uint desc,uint source) {
    const uint shift=source==0 ? 5 : source==1 ? 14 : 23;
    const uint sign=source==0 ? 4 : source==1 ? 13 : 22;
    float4 result;
    for (uint component=0;component<4;++component)
        result[component]=value[(desc>>(shift+6-component*2))&3u];
    return desc&(1u<<sign) ? -result : result;
}
float4 machine_mul(float4 a,float4 b) {
    float4 result=a*b;
    for (uint c=0;c<4;++c) if (isnan(result[c]) && !isnan(a[c]) && !isnan(b[c])) result[c]=0;
    return result;
}
bool machine_condition(uint word,thread MachineUnit& unit) {
    const bool x=(unit.flags.x!=0)==bool((word>>25)&1);
    const bool y=(unit.flags.y!=0)==bool((word>>24)&1);
    switch ((word>>22)&3) {
    case 0:return x||y;
    case 1:return x&&y;
    case 2:return x;
    default:return y;
    }
}
bool machine_compare(float a,float b,uint mode) {
    switch (mode) {
    case 0:return a==b;case 1:return a!=b;case 2:return a<b;
    case 3:return a<=b;case 4:return a>b;case 5:return a>=b;default:return true;
    }
}

struct RecurrenceEvent { uint lane,operation,flags,operand; };
static_assert(sizeof(RecurrenceEvent)==16,"Native scalar recurrence event layout");
template<bool TrackWrites=false,bool TrackRecurrences=false>
uint4 machine_execute(const device uint* code,const device uint* swizzle,
    constant MachineUniforms& uniforms,thread MachineUnit& unit,
    device float4* output,device Emission* emitted,constant MachineParams& info,
    thread uint* writes=nullptr,device RecurrenceEvent* events=nullptr,
    uint2 recurrence_mask=uint2(0),uint recurrence_capacity=0,thread uint* event_count=nullptr) {
    FlowFrame calls[4]={},branches[8]={},loops[4]={};
    uint call_depth=0,branch_depth=0,loop_depth=0;
    uint pc=info.entry,next=pc,jump=pc,count=0,steps=0,error=0;
    bool ended=false;
    bool breaking=false;
    while (!ended && !error && steps<info.limit) {
        if (steps) pc=machine_advance(next,jump,breaking,calls,branches,loops,
            call_depth,branch_depth,loop_depth,unit);
        breaking=false;
        if (pc>=4096) { error=1;break; }
        const uint word=code[pc],op=word>>26;
        const uint target=(word>>10)&4095u,length=word&255u;
        ++pc;++steps;
        next=jump=pc;
        if (op==0x21) continue;
        if (op==0x22) { ended=true;continue; }
        if ((op==0x2a || op==0x2b) && !info.allow_emission) { error=2;break; }
        if (op==0x2b) { unit.flags.w=(word>>22)&15u;continue; }
        if (op==0x2a) {
            const uint vertex_id=(unit.flags.w>>2)&3u;
            if (vertex_id>=3) { error=3;break; }
            uint slot=0;
            for (uint reg=0;reg<16;++reg) if (info.output_mask&(1u<<reg))
                unit.emit_buffer[vertex_id][slot++]=unit.output[unit.flags.z][reg];
            if (unit.flags.w&2u) {
                if (count>info.capacity || 3u>info.capacity-count) { error=4;break; }
                for (uint v=0;v<3;++v) {
                    for (uint reg=0;reg<16;++reg) emitted[count+v].attributes[reg]=unit.emit_buffer[v][reg];
                    emitted[count+v].winding=v==0 ? unit.flags.w&1u : 0u;
                    emitted[count+v].pad0=emitted[count+v].pad1=emitted[count+v].pad2=0;
                }
                count+=3;
            }
            unit.flags.z^=1u;
            continue;
        }
        if (op==0x20 || op==0x23) {
            if (op==0x23 && !machine_condition(word,unit)) continue;
            if (!loop_depth) { error=7;break; }
            breaking=true;
            continue;
        }
        if (op>=0x24 && op<=0x2d) {
            if (op==0x2c || op==0x2d) {
                const bool take=op==0x2c ? machine_condition(word,unit) :
                    bool(uniforms.b&(1u<<((word>>22)&15u)))!=bool(length&1u);
                if (take) jump=target;
                continue;
            }
            if (op==0x29) {
                const uint4 loop=uniforms.i[(word>>22)&3u];
                if (target<pc || target>=4096) { error=5;break; }
                unit.address.z=int(loop.y&255u);
                if(TrackWrites) writes[32]|=4;
                machine_push(loops,loop_depth,4,{target+1,target+1,pc,(loop.x&255u)+1,loop.z&255u});
                continue;
            }
            const bool take=op==0x24 ? true : (op==0x25 || op==0x28) ?
                machine_condition(word,unit) : bool(uniforms.b&(1u<<((word>>22)&15u)));
            const bool branch=op==0x27 || op==0x28;
            if (target+length>4096) { error=5;break; }
            if (branch) {
                if (target<pc) { error=5;break; }
                if (take) {
                    if (pc==target) jump=target+length;
                    else {
                        machine_push(branches,branch_depth,8,{target,target+length,0,0,0});
                    }
                } else {
                    jump=target;
                }
            } else if (take && length) {
                machine_push(calls,call_depth,4,{target+length,pc,0,0,0});jump=target;
            }
            continue;
        }
        const bool mad=op>=0x30;
        const bool inverted=op==0x18 || op==0x19 || op==0x1a || op==0x1b || (mad && op<0x38);
        const uint desc=swizzle[word&(mad ? 31u : 127u)];
        const uint address=(word>>(mad ? 22 : 19))&3u;
        uint src1,src2,src3=0,dest;
        if (mad) {
            src1=(word>>17)&31u;
            src2=(word>>(inverted ? 12 : 10))&(inverted ? 31u : 127u);
            src3=(word>>5)&(inverted ? 127u : 31u);
            dest=(word>>24)&31u;
        } else {
            src1=(word>>(inverted ? 14 : 12))&(inverted ? 31u : 127u);
            src2=(word>>7)&(inverted ? 127u : 31u);
            dest=(word>>21)&31u;
        }
        float4 a=machine_swizzle(machine_source(src1,(!mad && !inverted) ? address : 0,unit,uniforms),desc,0);
        float4 b=machine_swizzle(machine_source(src2,(mad && !inverted) || (!mad && inverted) ? address : 0,unit,uniforms),desc,1);
        float4 c=machine_swizzle(machine_source(src3,mad && inverted ? address : 0,unit,uniforms),desc,2);
        float4 value=0;
        if (mad) value=machine_mul(a,b)+c;
        else switch (op) {
        case 0x00:value=a+b;break;
        case 0x01:value=float4(dot(machine_mul(a,b).xyz,float3(1)));break;
        case 0x02:value=float4(dot(machine_mul(a,b),float4(1)));break;
        case 0x03:case 0x18:a.w=1;value=float4(dot(machine_mul(a,b),float4(1)));break;
        case 0x04:value=float4(1,machine_mul(a,b).y,a.z,b.w);break;
        case 0x19:value=float4(1,machine_mul(a,b).y,b.z,a.w);break;
        case 0x05:value=float4(exp2(a.x));break;
        case 0x06:value=float4(log2(a.x));break;
        case 0x07:
            value=float4(a.x>=0 ? a.x : 0,clamp(a.y,-127.99609375f,127.99609375f),0,a.w>=0 ? a.w : 0);
            unit.flags.x=a.x>=0;unit.flags.y=a.w>=0;
            if(TrackWrites) writes[33]|=3;
            break;
        case 0x08:value=machine_mul(a,b);break;
        case 0x09:case 0x1a:value=float4(a>=b);break;
        case 0x0a:case 0x1b:value=float4(a<b);break;
        case 0x0b:value=floor(a);break;
        case 0x0c:for(uint i=0;i<4;++i) value[i]=a[i]>b[i] ? a[i] : b[i];break;
        case 0x0d:for(uint i=0;i<4;++i) value[i]=a[i]<b[i] ? a[i] : b[i];break;
        case 0x0e:value=float4(1.0f/a.x);break;
        case 0x0f:value=float4(rsqrt(a.x));break;
        case 0x12:
            for (uint i=0;i<2;++i) if (desc&(8u>>i))
                {
                unit.address[i]=isfinite(a[i]) ? int(clamp(a[i],-2147483648.0f,2147483520.0f)) : 0;
                if(TrackWrites) writes[32]|=1u<<i;
            }
            continue;
        case 0x13:value=a;break;
        case 0x2e:case 0x2f:
            unit.flags.x=machine_compare(a.x,b.x,(word>>24)&7u);
            unit.flags.y=machine_compare(a.y,b.y,(word>>21)&7u);
            if(TrackWrites) writes[33]|=3;
            continue;
        default:error=2;break;
        }
        if (error) break;
        for (uint component=0;component<4;++component) if (desc&(8u>>component)) {
            if(TrackWrites) writes[dest]|=8u>>component;
            if(TrackRecurrences && dest>=16) {
                const uint lane=(dest-16)*4+component;
                if(recurrence_mask[lane/32]&(1u<<(lane%32))) {
                    if(*event_count>=recurrence_capacity) { error=9;break; }
                    const bool self_a=src1==dest && ((desc>>(11-component*2))&3u)==component;
                    const bool self_b=src2==dest && ((desc>>(20-component*2))&3u)==component;
                    RecurrenceEvent event={lane,0,0,as_type<uint>(value[component])};
                    if(!mad && op==0x13 && self_a) { event.operation=3;event.flags=(desc>>4)&1u; }
                    else if(!mad && (op==0x00 || op==0x08) && self_a!=self_b) {
                        event.operation=op==0x08 ? 1:2;
                        event.flags=(self_a ? (desc>>4)&1u:(desc>>13)&1u)|(self_b ? 2u:0u);
                        event.operand=as_type<uint>(self_a ? b[component]:a[component]);
                    }
                    events[(*event_count)++]=event;
                }
            }
            if (dest<16) unit.output[unit.flags.z][dest][component]=value[component];
            else unit.temporary[dest-16][component]=value[component];
        }
    }
    if (!ended && !error) error=6;
    for (uint reg=0;reg<16;++reg) output[reg]=unit.output[unit.flags.z][reg];
    return uint4(error,count,steps,pc);
}


kernel void pica_shader_machine(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* emitted [[buffer(6)]],
    device uint4& status [[buffer(7)]],constant MachineParams& info [[buffer(8)]],
    device uint& abort [[buffer(9)]]) {
    // Guarded GS invocations are ordered by command-encoder barriers. On failure
    // later invocations must not consume the missing retained GS state.
    if (info.pad1 && abort) { status=uint4(8,0,0,0);return; }
    MachineUnit unit={};
    if (!info.reset) unit=stored;
    for (uint reg=0;reg<16;++reg) if (info.input_mask&(1u<<reg)) unit.input[reg]=input[reg];
    status=machine_execute(code,swizzle,uniforms,unit,output,emitted,info);
    if (!status.x) stored=unit;
    else if (info.pad1) abort=1;
}

struct VertexBatchParams {
    MachineParams machine;
    uint vertex_count,element_count,vertex_base,first_vertex,index_size,pad0,parallel_merge,shared_fifo_plan;
    uint2 recurrence_mask;uint recurrence_capacity,aliased_state;
    uint always_written[5],possibly_written[5];
};
static_assert(sizeof(VertexBatchParams)==120,"Native vertex write-proof parameter layout");
bool vertex_may_write(constant VertexBatchParams& info,uint lane) {
    return info.possibly_written[lane/32]&(1u<<(lane%32));
}
kernel void pica_vertex_machine_batch(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device uint4& status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]]) {
    MachineUnit unit={};
    if (!info.machine.reset) unit=stored;
    uint ids[64]={},results[64]={},size=0,replacement=0;
    status=uint4(0);
    for (uint element=0;element<info.element_count;++element) {
        uint vertex_id=info.first_vertex+element;
        if (info.index_size) {
            vertex_id=0;
            for(uint byte=0;byte<info.index_size;++byte)
                vertex_id|=uint(indices[ulong(element)*info.index_size+byte])<<(8u*byte);
        }
        if (vertex_id<info.vertex_base || vertex_id-info.vertex_base>=info.vertex_count) {
            status=uint4(8,element,0,0);return;
        }
        uint cached=~0u;
        // Live FIFO IDs are unique. Scan their bounded prefix without an early
        // exit so repeated-hit runs retain the correct compiled source index.
        if (info.index_size) for(uint slot=0;slot<size;++slot)
            if(ids[slot]==vertex_id) cached=results[slot];
        if (cached!=~0u) {
            for(uint reg=0;reg<16;++reg) output[ulong(element)*16+reg]=output[ulong(cached)*16+reg];
            continue;
        }
        for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(vertex_id-info.vertex_base)*16+reg];
        const uint4 result=machine_execute(code,swizzle,uniforms,unit,
            output+ulong(element)*16,unused_emissions,info.machine);
        if (result.x) { status=result;return; }
        if (info.index_size) {
            ids[replacement]=vertex_id;results[replacement]=element;
            replacement=(replacement+1u)&63u;size=min(size+1u,64u);
        }
    }
    stored=unit;
}

uint batch_vertex(const device uchar* indices,constant VertexBatchParams& info,uint element) {
    uint id=info.first_vertex+element;
    if(info.index_size) {
        id=0;
        for(uint byte=0;byte<info.index_size;++byte)
            id|=uint(indices[ulong(element)*info.index_size+byte])<<(8u*byte);
    }
    return id;
}
void batch_error(device atomic_uint* status,uint4 result) {
    if(!result.x) return;
    uint expected=0;
    while(!atomic_compare_exchange_weak_explicit(status,&expected,result.x,memory_order_relaxed,memory_order_relaxed))
        if(expected) return;
    {
        // Other lanes only use the atomic error word; details are read after completion.
        device uint* details=reinterpret_cast<device uint*>(status);
        details[1]=result.y;details[2]=result.z;details[3]=result.w;
    }
}
kernel void pica_vertex_machine_parallel(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],const device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],uint element [[thread_position_in_grid]]) {
    if(element>=info.element_count) return;
    const uint id=batch_vertex(indices,info,element);
    if(id<info.vertex_base || id-info.vertex_base>=info.vertex_count) {
        batch_error(status,uint4(8,element,0,0));return;
    }
    MachineUnit unit={};
    if(!info.machine.reset) unit=stored;
    for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(id-info.vertex_base)*16+reg];
    batch_error(status,machine_execute(code,swizzle,uniforms,unit,output+ulong(element)*16,
                                      unused_emissions,info.machine));
}
// Only components actually written by a vertex may replace the inherited state.
// Input-dependent branches can leave different components untouched on each invocation.
struct VertexPlan { uint cached,previous; };
static_assert(sizeof(VertexPlan)==8,"Native vertex dependency plan layout");
struct VertexCarry {
    float4 temporary[16];
    int4 address;
    uint4 flags;
    uint writes[34];
};
static_assert(sizeof(VertexCarry)==432,"Native vertex carry layout");
kernel void pica_vertex_machine_carry(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],const device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],device VertexCarry* carry [[buffer(10)]],
    const device VertexPlan* plan [[buffer(12)]],
    uint element [[thread_position_in_grid]]) {
    if(element>=info.element_count) return;
    // The preceding FIFO plan identifies the only invocations that execute.
    // Cached outputs and retained state are handled by the existing merge.
    if(info.index_size && plan[element].cached!=~0u) return;
    const uint id=batch_vertex(indices,info,element);
    if(id<info.vertex_base || id-info.vertex_base>=info.vertex_count) {
        batch_error(status,uint4(8,element,0,0));return;
    }
    MachineUnit unit={};
    if(!info.machine.reset) unit=stored;
    uint writes[34]={};
    for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(id-info.vertex_base)*16+reg];
    batch_error(status,machine_execute<true>(code,swizzle,uniforms,unit,
        output+ulong(element)*16,unused_emissions,info.machine,writes));
    for(uint reg=0;reg<16;++reg) carry[element].temporary[reg]=unit.temporary[reg];
    carry[element].address=unit.address;carry[element].flags=unit.flags;
    for(uint reg=0;reg<34;++reg) carry[element].writes[reg]=writes[reg];
}
// The plan records actual FIFO misses. Cache hits neither execute a vertex nor
// advance retained state, including when more than 64 distinct IDs wrap the FIFO.
struct VertexSeed { float4 temporary[16];int4 address;uint4 flags; };
static_assert(sizeof(VertexSeed)==288,"Native vertex dependency seed layout");
VertexPlan vertex_plan(const device VertexPlan* plan,constant VertexBatchParams& info,uint element) {
    // The host validates the complete sequential range before queuing a draw.
    // Every non-indexed element is a miss, so no plan allocation or pass is needed.
    if(!info.index_size) return {~0u,element ? element-1:~0u};
    return plan[element];
}
template<bool WritePlan>
uint vertex_fifo_plan(device uint4& status,constant VertexBatchParams& info,
    const device uchar* indices,device VertexPlan* plan,uint lane,uint width) {
    // One SIMD group searches all 64 entries, preserving the scalar FIFO's
    // exact miss order and cached source element. Hits never advance state.
    uint4 keys=uint4(0),values=uint4(~0u);
    uint previous=~0u,replacement=0,replacement_lane=0,replacement_slot=0;
    for(ulong base=0;base<info.element_count;base+=width) {
        VertexPlan prefetched_plan={~0u,~0u};
        uint prefetched=0;
        if(base+lane<info.element_count) prefetched=batch_vertex(indices,info,uint(base)+lane);
        const uint count=min(ulong(width),ulong(info.element_count)-base);
        for(uint offset=0;offset<count;++offset) {
            const uint element=uint(base)+offset,id=simd_broadcast(prefetched,offset);
            if(id<info.vertex_base || id-info.vertex_base>=info.vertex_count) {
                // Retire exactly the valid prefix of this chunk. The invalid
                // entry and its suffix must remain untouched, including lane zero.
                if(WritePlan && lane<offset) plan[uint(base)+lane]=prefetched_plan;
                if(lane==0) status=uint4(8,element,0,0);
                return ~0u;
            }
            // A SIMD group of 16/32/64 lanes needs at most 4/2/1 slot banks.
            uint matching=keys.x==id ? values.x:~0u;
            if(width<64) matching=min(matching,keys.y==id ? values.y:~0u);
            if(width<32) matching=min(matching,min(keys.z==id ? values.z:~0u,keys.w==id ? values.w:~0u));
            const uint cached=info.index_size ? simd_min(matching):~0u;
            if(WritePlan && lane==offset) prefetched_plan={cached,previous};
            if(cached!=~0u) continue;
            previous=element;
            if(info.index_size) {
                if(lane==replacement_lane) {
                    keys[replacement_slot]=id;values[replacement_slot]=element;
                }
                // Only misses rotate the exact 64-entry FIFO. Track the lane
                // and bank directly instead of dividing each replacement index.
                ++replacement;++replacement_lane;
                if(replacement==64) {replacement=0;replacement_lane=0;replacement_slot=0;}
                else if(replacement_lane==width) {replacement_lane=0;++replacement_slot;}
            }
        }
        // Each lane owns one prefetched entry: write the complete chunk together.
        if(WritePlan && lane<count) plan[uint(base)+lane]=prefetched_plan;
    }
    return previous;
}
kernel void pica_vertex_machine_plan(device uint4& status [[buffer(7)]],
    constant VertexBatchParams& info [[buffer(8)]],const device uchar* indices [[buffer(9)]],
    device VertexPlan* plan [[buffer(12)]],uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    vertex_fifo_plan<true>(status,info,indices,plan,lane,width);
}
kernel void pica_vertex_machine_dependency(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],const device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],device VertexCarry* carry [[buffer(10)]],
    const device VertexSeed* seed [[buffer(11)]],const device VertexPlan* plan [[buffer(12)]],
    uint element [[thread_position_in_grid]]) {
    if(element>=info.element_count || atomic_load_explicit(status,memory_order_relaxed)) return;
    const VertexPlan entry=vertex_plan(plan,info,element);
    if(entry.cached!=~0u) return;
    MachineUnit unit={};
    if(!info.machine.reset) unit=stored;
    if(info.pad0 && entry.previous!=~0u) {
        const uint previous=entry.previous;
        for(uint reg=0;reg<16;++reg) unit.temporary[reg]=seed[previous].temporary[reg];
        unit.address=seed[previous].address;unit.flags=seed[previous].flags;
    }
    const uint id=batch_vertex(indices,info,element);
    for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(id-info.vertex_base)*16+reg];
    uint writes[34]={};
    batch_error(status,machine_execute<true>(code,swizzle,uniforms,unit,
        output+ulong(element)*16,unused_emissions,info.machine,writes));
    for(uint reg=0;reg<16;++reg) carry[element].temporary[reg]=unit.temporary[reg];
    carry[element].address=unit.address;carry[element].flags=unit.flags;
    for(uint reg=0;reg<34;++reg) carry[element].writes[reg]=writes[reg];
}
// Eight non-indexed invocations share exact component prefixes locally. All
// 64 lanes reach every barrier, including inactive lanes and GPU failures.
// The arithmetic is the same machine_execute used by the ordinary dependency
// kernel. Only its carry/seed/output prefix moves into threadgroup storage.
kernel void pica_vertex_machine_small_dependency(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    uint element [[thread_index_in_threadgroup]]) {
    threadgroup VertexCarry carry[8];
    threadgroup VertexSeed seed[8];
    threadgroup float4 raw_output[8][16];
    const bool active=element<info.element_count;
    for(uint pass=0;pass<info.pad0;++pass) {
        if(active && !atomic_load_explicit(status,memory_order_relaxed)) {
            MachineUnit unit={};
            if(!info.machine.reset) unit=stored;
            if(pass) {
                for(uint reg=0;reg<16;++reg) unit.temporary[reg]=seed[element].temporary[reg];
                unit.address=seed[element].address;unit.flags=seed[element].flags;
            }
            const uint vertex_id=info.first_vertex-info.vertex_base+element;
            for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(vertex_id)*16+reg];
            uint writes[34]={};
            batch_error(status,machine_execute<true>(code,swizzle,uniforms,unit,
                output+ulong(element)*16,unused_emissions,info.machine,writes));
            for(uint reg=0;reg<16;++reg) {
                carry[element].temporary[reg]=unit.temporary[reg];
                raw_output[element][reg]=unit.output[unit.flags.z][reg];
            }
            carry[element].address=unit.address;carry[element].flags=unit.flags;
            for(uint reg=0;reg<34;++reg) carry[element].writes[reg]=writes[reg];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
        if(pass+1<info.pad0 && !atomic_load_explicit(status,memory_order_relaxed)) {
            const device uint* initial=reinterpret_cast<const device uint*>(&stored);
            for(uint item=element;item<info.element_count*72;item+=64) {
                const uint row=item/72,component=item%72;
                uint value=info.machine.reset ? 0:initial[component<64 ? 64+component:448+component-64];
                for(uint prior=0;prior<row;++prior) {
                    bool written=false;
                    if(component<64) written=carry[prior].writes[16+component/4]&(8u>>(component%4));
                    else if(component<67) written=carry[prior].writes[32]&(1u<<(component-64));
                    else if(component>=68 && component<70) written=carry[prior].writes[33]&(1u<<(component-68));
                    if(written) value=reinterpret_cast<const threadgroup uint*>(&carry[prior])[component];
                }
                reinterpret_cast<threadgroup uint*>(&seed[row])[component]=value;
            }
        }
        // No lane overwrites carry until every seed reader has finished.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint bank=info.machine.reset ? 0:stored.flags.z;
    const bool success=!atomic_load_explicit(status,memory_order_relaxed);
    if(success) for(uint item=element;item<info.element_count*64;item+=64) {
        const uint row=item/64,component=item%64,reg=component/4,lane=component%4;
        uint value=info.machine.reset ? 0:as_type<uint>(stored.output[bank][reg][lane]);
        for(uint prior=0;prior<=row;++prior)
            if(carry[prior].writes[reg]&(8u>>lane)) value=as_type<uint>(raw_output[prior][reg][lane]);
        reinterpret_cast<device uint*>(output)[item]=value;
    }
    // Other lanes still read the original stored output during their prefix.
    threadgroup_barrier(mem_flags::mem_device);
    if(success) for(uint word=element;word<456;word+=64) {
        const uint last=info.element_count-1,vertex_id=info.first_vertex-info.vertex_base+last;
        uint value=0;
        bool written=true;
        if(word<64) value=reinterpret_cast<const device uint*>(input)[ulong(vertex_id)*64+word];
        else if(word<128) value=reinterpret_cast<const threadgroup uint*>(&carry[last])[word-64];
        else if(word>=128+bank*64 && word<192+bank*64)
            value=reinterpret_cast<const device uint*>(output)[ulong(last)*64+word-128-bank*64];
        else if(word>=448) value=reinterpret_cast<const threadgroup uint*>(&carry[last])[64+word-448];
        else written=bool(info.machine.reset);
        if(written) {
            reinterpret_cast<device uint*>(&stored)[word]=value;
        }
    }
}
kernel void pica_vertex_machine_recurrence(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],const device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],device VertexCarry* carry [[buffer(10)]],
    const device VertexSeed* seed [[buffer(11)]],const device VertexPlan* plan [[buffer(12)]],
    device RecurrenceEvent* events [[buffer(14)]],uint element [[thread_position_in_grid]]) {
    if(element>=info.element_count || atomic_load_explicit(status,memory_order_relaxed)) return;
    const VertexPlan entry=vertex_plan(plan,info,element);
    if(entry.cached!=~0u) return;
    MachineUnit unit={};
    if(!info.machine.reset) unit=stored;
    if(info.pad0 && entry.previous!=~0u) {
        const uint previous=entry.previous;
        for(uint reg=0;reg<16;++reg) unit.temporary[reg]=seed[previous].temporary[reg];
        unit.address=seed[previous].address;unit.flags=seed[previous].flags;
    }
    const uint id=batch_vertex(indices,info,element);
    for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(id-info.vertex_base)*16+reg];
    uint writes[34]={};
    uint count=0;
    device RecurrenceEvent* recorded=events+ulong(element)*(info.recurrence_capacity+1u);
    batch_error(status,machine_execute<true,true>(code,swizzle,uniforms,unit,
        output+ulong(element)*16,unused_emissions,info.machine,writes,recorded+1,
        info.recurrence_mask,info.recurrence_capacity,&count));
    recorded[0]={count,0,0,0};
    for(uint reg=0;reg<16;++reg) carry[element].temporary[reg]=unit.temporary[reg];
    carry[element].address=unit.address;carry[element].flags=unit.flags;
    for(uint reg=0;reg<34;++reg) carry[element].writes[reg]=writes[reg];
}
kernel void pica_vertex_machine_prefix(const device MachineUnit& stored [[buffer(4)]],
    const device uint4& status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device VertexCarry* carry [[buffer(10)]],device VertexSeed* seed [[buffer(11)]],
    const device VertexPlan* plan [[buffer(12)]],uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    if(status.x) return;
    const device uint* initial=reinterpret_cast<const device uint*>(&stored);
    device uint* seeds=reinterpret_cast<device uint*>(seed);
    const uint scan_begin=(info.element_count+63u)&~63u;
    if(gid<scan_begin) {
        // Direct rows and SIMD scans occupy disjoint, SIMD-aligned ranges.
        if(gid<info.element_count && vertex_plan(plan,info,gid).cached==~0u) {
            for(uint component=0;component<72;++component) {
                const bool immutable=component==67 || component>=70;
                const uint retained_lane=component<67 ? component+64:component-68+131;
                const bool always=!immutable && (info.always_written[retained_lane/32]&
                    (1u<<(retained_lane%32)));
                const bool possible=!immutable && vertex_may_write(info,retained_lane);
                if(always || !possible) {
                    const uint value=always ? reinterpret_cast<const device uint*>(&carry[gid])[component]:
                        info.machine.reset ? 0:initial[component<64 ? 64+component:448+component-64];
                    seeds[ulong(gid)*72+component]=value;
                }
            }
        }
        return;
    }
    const uint lane=(gid-scan_begin)/width;
    if(lane>=72) return;
    const bool immutable=lane==67 || lane>=70;
    const uint retained_lane=lane<67 ? lane+64:lane-68+131;
    if(immutable || !vertex_may_write(info,retained_lane) ||
       (info.always_written[retained_lane/32]&(1u<<(retained_lane%32)))) return;
    const uint inherited=info.machine.reset ? 0:initial[lane<64 ? 64+lane:448+lane-64];
    uint previous_writer=0;
    for(ulong base=0;base<info.element_count;base+=width) {
        const uint element=uint(base)+simd_lane;
        const bool miss=element<info.element_count && vertex_plan(plan,info,element).cached==~0u;
        bool written=false;
        if(miss) {
            if(lane<64) written=carry[element].writes[16+lane/4]&(8u>>(lane%4));
            else if(lane<67) written=carry[element].writes[32]&(1u<<(lane-64));
            else written=carry[element].writes[33]&(1u<<(lane-68));
        }
        uint writer=written ? element+1u:0u;
        // Inclusive maximum over integer writer IDs is exactly associative.
        // All SIMD lanes participate, including the final partial input block.
        for(ushort shift=1;shift<width;shift*=2) {
            const uint preceding=simd_shuffle_up(writer,shift);
            if(simd_lane>=shift) writer=max(writer,preceding);
        }
        writer=max(writer,previous_writer);
        if(miss) seeds[ulong(element)*72+lane]=writer ?
            reinterpret_cast<const device uint*>(&carry[writer-1u])[lane]:inherited;
        previous_writer=simd_broadcast(writer,width-1u);
    }
}
// The 133 mutable retained components are independent once vertex arithmetic has
// finished. Scalar uint stores preserve float bits and cannot overwrite adjacent
// vector components. Cache hits copy their original output without advancing state.
static_assert(sizeof(MachineUnit)==1824,"Native complete unit layout");
// Statically isolated temporary components replay their exact scalar operations
// in FIFO-miss order. Their merge lane is the sole writer, so no intermediate
// normal merge or separate replay dispatch is required.
uint replay_vertex_component(uint current,uint lane,constant VertexBatchParams& info,
    const device VertexPlan* plan,const device RecurrenceEvent* events) {
    for(uint element=0;element<info.element_count;++element) {
        if(vertex_plan(plan,info,element).cached!=~0u) continue;
        const device RecurrenceEvent* recorded=events+ulong(element)*(info.recurrence_capacity+1u);
        for(uint index=0;index<recorded[0].lane;++index) {
            const RecurrenceEvent event=recorded[index+1];
            if(event.lane!=lane) continue;
            if(event.operation==0) { current=event.operand;continue; }
            float value=as_type<float>(current);
            if(event.flags&1u) value=-value;
            if(event.operation==3) { current=as_type<uint>(value);continue; }
            const float operand=as_type<float>(event.operand);
            const float a=event.flags&2u ? operand:value,b=event.flags&2u ? value:operand;
            current=as_type<uint>(event.operation==1 ? machine_mul(float4(a),float4(b)).x:a+b);
        }
    }
    return current;
}
// Resolve inherited output components before the final retained-state commit.
// Integer last-writer IDs admit an exact SIMD prefix scan. Raw uint copies
// preserve produced scalar bits without floating-point conversion.
void pica_vertex_machine_output(const device MachineUnit& stored,
    device uint* output,const device uint4& status,
    constant VertexBatchParams& info,const device VertexCarry* carry,
    const device VertexPlan* plan,uint component,
    uint lane,uint width) {
    if(component>=64 || status.x) return;
    const bool always_written=info.always_written[component/32]&(1u<<(component%32));
    if(always_written) {
        if(info.index_size) for(ulong element=lane;element<info.element_count;element+=width) {
            const uint cached=plan[element].cached;
            if(cached!=~0u) output[element*64+component]=output[ulong(cached)*64+component];
        }
        return;
    }
    const uint bank=info.machine.reset ? 0:stored.flags.z;
    const uint initial=info.machine.reset ? 0:reinterpret_cast<const device uint*>(&stored)[128+bank*64+component];
    if(!vertex_may_write(info,component)) {
        for(ulong element=lane;element<info.element_count;element+=width) output[element*64+component]=initial;
        return;
    }
    uint previous_writer=0;
    for(ulong base=0;base<info.element_count;base+=width) {
        const ulong element=base+lane;
        const uint cached=element<info.element_count ? vertex_plan(plan,info,uint(element)).cached:~0u;
        const bool miss=element<info.element_count && cached==~0u;
        const bool written=miss && (carry[element].writes[component/4]&(8u>>(component%4)));
        uint writer=written ? uint(element)+1u:0u;
        for(ushort shift=1;shift<width;shift*=2) {
            const uint preceding=simd_shuffle_up(writer,shift);
            if(lane>=shift) writer=max(writer,preceding);
        }
        writer=max(writer,previous_writer);
        // A hit in this block selects the writer at its original miss, rather
        // than the writer preceding the hit. Clamp inactive shuffle sources.
        const ushort source=cached!=~0u && ulong(cached)>=base ? ushort(cached-base):0;
        const uint cached_writer=simd_shuffle(writer,source);
        if(element<info.element_count && !written) {
            if(cached!=~0u && ulong(cached)<base) output[element*64+component]=output[ulong(cached)*64+component];
            else {
                const uint selected=cached==~0u ? writer:cached_writer;
                output[element*64+component]=selected ? output[ulong(selected-1u)*64+component]:initial;
            }
        }
        // Actual writer outputs are immutable. Earlier inherited hit sources
        // must be visible before this SIMD group advances to the next block.
        simdgroup_barrier(mem_flags::mem_device);
        previous_writer=simd_broadcast(writer,width-1u);
    }
}
// Select conditional non-output writers in reverse SIMD-sized blocks.
// Integer maximum preserves the exact final writer and its original bits.
void pica_vertex_machine_retained(const device MachineUnit& stored,
    const device uint4& status,constant VertexBatchParams& info,
    device VertexCarry* carry,const device VertexPlan* plan,
    uint component,uint lane,
    uint width) {
    if(component>=69 || status.x) return;
    const uint retained_lane=component+64;
    if(!vertex_may_write(info,retained_lane) ||
       (info.always_written[retained_lane/32]&(1u<<(retained_lane%32))) ||
       (component<64 && (info.recurrence_mask[component/32]&(1u<<(component%32))))) return;
    const uint offset=component<64 ? retained_lane:component<67 ? 448+component-64:452+component-67;
    uint current=info.machine.reset ? 0:reinterpret_cast<const device uint*>(&stored)[offset];
    for(ulong end=info.element_count;end;) {
        const bool valid=lane<end;
        const uint element=valid ? uint(end-1u-lane):0u;
        const bool miss=valid && vertex_plan(plan,info,element).cached==~0u;
        bool written=false;
        if(miss) {
            if(component<64) written=carry[element].writes[retained_lane/4]&(8u>>(retained_lane%4));
            else if(component<67) written=carry[element].writes[32]&(1u<<(component-64));
            else written=carry[element].writes[33]&(1u<<(component-67));
        }
        const uint writer=simd_max(written ? element+1u:0u);
        if(writer) {
            const device uint* values=reinterpret_cast<const device uint*>(&carry[writer-1u]);
            current=values[component<67 ? component:component+1u];
            break;
        }
        end=end>width ? end-width:0;
    }
    if(!lane) {
        const uint last=info.element_count-1;
        const VertexPlan entry=vertex_plan(plan,info,last);
        const uint last_miss=entry.cached==~0u ? last:entry.previous;
        reinterpret_cast<device uint*>(&carry[last_miss])[component<67 ? component:component+1u]=current;
    }
}
// Prefetch consecutive FIFO-miss events together, then replay them in their
// original element/operation order in every lane. Floating-point operations
// are never reassociated; only lane zero commits the resulting scalar bits.
void pica_vertex_machine_replay(const device MachineUnit& stored,
    const device uint4& status,constant VertexBatchParams& info,
    device VertexCarry* carry,const device VertexPlan* plan,
    const device RecurrenceEvent* events,uint component,
    uint lane,uint width) {
    if(component>=64 || status.x || !(info.recurrence_mask[component/32]&(1u<<(component%32)))) return;
    uint current=info.machine.reset ? 0:reinterpret_cast<const device uint*>(&stored)[64+component];
    for(ulong base=0;base<info.element_count;base+=width) {
        const ulong element=base+lane;
        const bool miss=element<info.element_count && vertex_plan(plan,info,uint(element)).cached==~0u;
        const device RecurrenceEvent* recorded=events+element*(info.recurrence_capacity+1u);
        const uint count=miss ? recorded[0].lane:0u;
        RecurrenceEvent own=count ? recorded[1]:RecurrenceEvent{0,0,0,0};
        for(ushort source=0;source<width && base+source<info.element_count;++source) {
            const uint iterations=simd_broadcast(count,source);
            for(uint index=0;index<iterations;++index) {
                const RecurrenceEvent event={simd_broadcast(own.lane,source),simd_broadcast(own.operation,source),
                    simd_broadcast(own.flags,source),simd_broadcast(own.operand,source)};
                if(event.lane==component) {
                    if(event.operation==0) current=event.operand;
                    else {
                        float value=as_type<float>(current);
                        if(event.flags&1u) value=-value;
                        if(event.operation==3) current=as_type<uint>(value);
                        else {
                            const float operand=as_type<float>(event.operand);
                            const float a=event.flags&2u ? operand:value,b=event.flags&2u ? value:operand;
                            current=as_type<uint>(event.operation==1 ? machine_mul(float4(a),float4(b)).x:a+b);
                        }
                    }
                }
                if(lane==source && index+1u<iterations) own=recorded[index+2u];
            }
        }
    }
    if(!lane) {
        const uint last=info.element_count-1;
        const VertexPlan entry=vertex_plan(plan,info,last);
        const uint last_miss=entry.cached==~0u ? last:entry.previous;
        reinterpret_cast<device uint*>(&carry[last_miss])[component]=current;
    }
}
kernel void pica_vertex_machine_resolve(const device MachineUnit& stored [[buffer(4)]],
    device uint* output [[buffer(5)]],const device uint4& status [[buffer(7)]],
    constant VertexBatchParams& info [[buffer(8)]],device VertexCarry* carry [[buffer(10)]],
    const device VertexPlan* plan [[buffer(12)]],const device RecurrenceEvent* events [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    if(group<64) pica_vertex_machine_output(stored,output,status,info,carry,plan,group,lane,width);
    else {
        const uint component=group-64;
        if(component<64 && (info.recurrence_mask[component/32]&(1u<<(component%32))))
            pica_vertex_machine_replay(stored,status,info,carry,plan,events,component,lane,width);
        else pica_vertex_machine_retained(stored,status,info,carry,plan,component,lane,width);
    }
}
kernel void pica_vertex_machine_merge(const device float4* input [[buffer(3)]],
    device MachineUnit& stored [[buffer(4)]],device uint* output [[buffer(5)]],
    const device uint4& status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],const device VertexCarry* carry [[buffer(10)]],
    const device VertexPlan* plan [[buffer(12)]],const device RecurrenceEvent* events [[buffer(14)]],
    uint lane [[thread_position_in_grid]]) {
    if(lane>=133 || status.x) return;
    const uint bank=info.machine.reset ? 0:stored.flags.z;
    const uint offset=lane<64 ? 128+bank*64+lane:lane<128 ? lane:
                      lane<131 ? 448+lane-128:452+lane-131;
    device uint* retained=reinterpret_cast<device uint*>(&stored);
    uint current=info.machine.reset ? 0:retained[offset];
    const uint last=info.element_count-1;
    const VertexPlan last_entry=vertex_plan(plan,info,last);
    const uint last_miss=last_entry.cached==~0u ? last:last_entry.previous;
    const bool may_write=vertex_may_write(info,lane);
    const bool always_written=info.always_written[lane/32]&(1u<<(lane%32));
    if(info.parallel_merge) {
        if(lane<64) current=output[ulong(last_miss)*64+lane];
        else if(may_write || (lane<128 && (info.recurrence_mask[(lane-64)/32]&(1u<<((lane-64)%32))))) {
            const device uint* values=reinterpret_cast<const device uint*>(&carry[last_miss]);
            current=values[lane<128 ? lane-64:lane<131 ? 64+lane-128:68+lane-131];
        }
    } else {
        if(lane>=64 && lane<128 && (info.recurrence_mask[(lane-64)/32]&(1u<<((lane-64)%32)))) {
            current=replay_vertex_component(current,lane-64,info,plan,events);
        } else if(always_written) {
            // Every actual miss writes this lane. Its final retained value comes
            // directly from the last miss, with no ordered carry scan. Output hits
            // still reuse the original miss, never the preceding invocation.
            const device uint* values=reinterpret_cast<const device uint*>(&carry[last_miss]);
            current=lane<64 ? output[ulong(last_miss)*64+lane]:
                    values[lane<128 ? lane-64:lane<131 ? 64+lane-128:68+lane-131];
            if(lane<64 && info.index_size) for(uint element=0;element<info.element_count;++element) {
                const uint cached=plan[element].cached;
                if(cached!=~0u) output[ulong(element)*64+lane]=output[ulong(cached)*64+lane];
            }
        } else if(!may_write) {
            // Unwritten retained lanes need no carry scan. Every output,
            // including cache hits, inherits the same initial scalar bit pattern.
            if(lane<64) for(uint element=0;element<info.element_count;++element)
                output[ulong(element)*64+lane]=current;
        } else if(lane>=64) {
            // Only the final retained value is needed for non-output components.
            // Select the last actual miss that wrote the component, preserving raw
            // bits; recurrence arithmetic remains in its original ordered path.
            for(uint remaining=info.element_count;remaining;--remaining) {
                const uint element=remaining-1;
                if(vertex_plan(plan,info,element).cached!=~0u) continue;
                bool written;
                if(lane<128) written=carry[element].writes[lane/4]&(8u>>(lane%4));
                else if(lane<131) written=carry[element].writes[32]&(1u<<(lane-128));
                else written=carry[element].writes[33]&(1u<<(lane-131));
                if(written) {
                    const device uint* values=reinterpret_cast<const device uint*>(&carry[element]);
                    current=values[lane<128 ? lane-64:lane<131 ? 64+lane-128:68+lane-131];
                    break;
                }
            }
        } else for(uint element=0;element<info.element_count;++element) {
            const uint cached=vertex_plan(plan,info,element).cached;
            if(cached!=~0u) {
                if(lane<64) output[ulong(element)*64+lane]=output[ulong(cached)*64+lane];
                continue;
            }
            const device uint* values=reinterpret_cast<const device uint*>(&carry[element]);
            if(lane<64) {
                if(carry[element].writes[lane/4]&(8u>>(lane%4))) current=output[ulong(element)*64+lane];
                output[ulong(element)*64+lane]=current;
            } else if(lane<128) {
                if(carry[element].writes[lane/4]&(8u>>(lane%4))) current=values[lane-64];
            } else if(lane<131) {
                if(carry[element].writes[32]&(1u<<(lane-128))) current=values[64+lane-128];
            } else {
                if(carry[element].writes[33]&(1u<<(lane-131))) current=values[68+lane-131];
            }
        }
    }
    uint final_input=0;
    if(lane<64) {
        const uint id=batch_vertex(indices,info,last_miss);
        final_input=reinterpret_cast<const device uint*>(input)[ulong(id-info.vertex_base)*64+lane];
    }
    // Disjoint inputs permit independent component groups. Aliased input/index
    // views use one group and retire every read before the first state store.
    if(info.aliased_state) threadgroup_barrier(mem_flags::mem_device);
    // Each lane reads and commits only its own retained scalar. The output bank
    // is immutable in VS mode; reset selects bank zero without reading stored.
    // All arithmetic/status writers precede this dispatch. No merge lane can
    // fail, so a failed batch still leaves every byte of stored untouched.
    retained[offset]=current;
    if(lane<64) retained[lane]=final_input;
    if(info.machine.reset) for(uint word=lane;word<456;word+=133) {
        // Inputs and mutable components already have a sole writer above.
        // Clear the inactive output bank, emission slots and immutable flags.
        if((word>=192 && word<448) || word==451 || word>=454) retained[word]=0;
    }
}
kernel void pica_vertex_machine_finalize(const device uint* code [[buffer(0)]],
    const device uint* swizzle [[buffer(1)]],constant MachineUniforms& uniforms [[buffer(2)]],
    const device float4* input [[buffer(3)]],device MachineUnit& stored [[buffer(4)]],
    device float4* output [[buffer(5)]],device Emission* unused_emissions [[buffer(6)]],
    device uint4& status [[buffer(7)]],constant VertexBatchParams& info [[buffer(8)]],
    const device uchar* indices [[buffer(9)]],const device VertexPlan* plan [[buffer(12)]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    if(status.x) return;
    uint last=info.element_count-1;
    // Reuse the exact SIMD FIFO search without writing an intermediate plan.
    // Every lane gets the last miss; only lane zero executes and commits it.
    if(info.index_size) last=info.shared_fifo_plan ? plan[info.element_count].cached:
        vertex_fifo_plan<false>(status,info,indices,nullptr,lane,width);
    if(lane || last==~0u) return;
    // The proof guarantees that written lanes do not depend on an earlier invocation.
    // Reexecute the last actual cache miss to commit an exact complete unit snapshot,
    // including lanes outside output_mask and untouched inherited register banks.
    const uint id=batch_vertex(indices,info,last);
    MachineUnit unit={};
    if(!info.machine.reset) unit=stored;
    for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(id-info.vertex_base)*16+reg];
    status=machine_execute(code,swizzle,uniforms,unit,output+ulong(last)*16,unused_emissions,info.machine);
    if(!status.x) stored=unit;
}
