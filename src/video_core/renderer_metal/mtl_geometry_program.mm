// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_geometry_program.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>

namespace Metal {
GeometryInputState::GeometryInputState(const Device& device)
    :uniforms{device.CreateBuffer(sizeof(ShaderMachineUniforms),false)},
     point{device.CreateBuffer(sizeof(VertexRegisters),false)},inputs{device.CreateBuffer(sizeof(VertexRegisters),false)} {}
GeometryProgramRuntime::GeometryProgramRuntime(Device& device_,Scheduler& scheduler_)
    :device{device_},scheduler{scheduler_},parameters{device,scheduler,1<<20},snapshots{device,scheduler} {
    // Keep GS transport kernels separate from the interpreter helpers. Besides
    // reducing compilation scope, this avoids their shader-validation compiler
    // interaction; the GPU layouts and ordered writes are unchanged.
    library=device.CompileLibrary(GeometryProgramShaderSource,"Azahar Metal 4 geometry input groups");
    auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.maxTotalThreadsPerThreadgroup=1;
    descriptor.computeFunctionDescriptor=Device::Function(library,"pica_geometry_inputs");
    input_pipeline=device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor=Device::Function(library,"pica_geometry_uniform_batch");
    descriptor.maxTotalThreadsPerThreadgroup=128;
    uniform_pipeline=device.CompileCompute(descriptor);
    descriptor.maxTotalThreadsPerThreadgroup=1;
    descriptor.computeFunctionDescriptor=Device::Function(library,"pica_geometry_invoked");
    invoked_pipeline=device.CompileCompute(descriptor);
    empty=device.CreateBuffer(4,true);
    std::memset(empty.contents,0,4);
}
void GeometryProgramRuntime::Initialize(GeometryInputState& state) {
    if (state.uniforms.device!=device.Handle()) throw std::invalid_argument("Geometry input state belongs to another device");
    if (state.initialized) return;
    TransferRuntime transfers{device,scheduler};
    for (auto buffer:{state.uniforms,state.point,state.inputs}) transfers.FillBuffer(buffer,NSMakeRange(0,buffer.length),0);
    state.initialized=true;
}
namespace {
void ValidateConfig(const GeometryInputConfig& config) {
    if (config.mode>GeometryInputMode::Variable || config.vs_output_mask>0xffff ||
        !config.vs_output_count || config.vs_output_count>16 ||
        std::popcount(config.vs_output_mask)<int(config.vs_output_count))
        throw std::invalid_argument("Invalid native GS mode/VS output configuration");
    if (config.mode==GeometryInputMode::Point) {
        if (!config.input_count || config.input_count>16 || config.input_count%config.vs_output_count)
            throw std::invalid_argument("Native point GS input group is not divisible by VS outputs");
        for (unsigned i=0;i<config.input_count;++i)
            if (config.input_registers[i]>=16) throw std::invalid_argument("Native GS input mapping exceeds register bank");
    } else if (config.mode==GeometryInputMode::Fixed) {
        if (!config.fixed_vertex_count || config.uniform_start>=96 ||
            std::uint64_t(config.vs_output_count)*config.fixed_vertex_count>96-config.uniform_start)
            throw std::invalid_argument("Native fixed GS input group exceeds uniform bank");
    } else if (!config.variable_main_count || config.variable_main_count>96) {
        throw std::invalid_argument("Invalid native variable GS main vertex count");
    }
}
}
void GeometryProgramRuntime::Configure(GeometryInputState& state,const GeometryInputConfig& config) {
    if (state.uniforms.device!=device.Handle()) throw std::invalid_argument("Geometry input state belongs to another device");
    ValidateConfig(config);
    if (state.configured && state.config==config) return;
    if (!state.IsEmpty()) throw std::invalid_argument("Cannot reconfigure an incomplete native GS group");
    Initialize(state);
    state.config=config;state.configured=true;
}
void GeometryProgramRuntime::UpdateUniforms(GeometryInputState& state,const ShaderMachineUniforms& values,
    std::array<std::uint32_t,3> float_mask,std::uint32_t bool_mask,std::uint32_t integer_mask) {
    (void)scheduler.LastUseTick(nil);
    if (bool_mask>0xffff || integer_mask>15 || state.awaiting)
        throw std::invalid_argument("Invalid or in-flight native GS uniform write");
    Initialize(state);
    auto& pending=state.pending_uniforms;
    for(unsigned word=0;word<3;++word) {
        auto mask=float_mask[word];
        while(mask) {
            const auto first=unsigned(std::countr_zero(mask));
            const auto count=unsigned(std::countr_one(mask>>first));
            std::memcpy(pending.floats.data()+word*32+first,values.f.data()+word*32+first,count*16);
            mask&=~(count==32 ? ~0u:((1u<<count)-1)<<first);
        }
        pending.info.float_masks[word]|=float_mask[word];
    }
    for(auto mask=integer_mask;mask;mask&=mask-1) {
        const auto reg=unsigned(std::countr_zero(mask));pending.integers[reg]=values.i[reg];
    }
    for(auto mask=bool_mask;mask;mask&=mask-1)
        pending.boolean_sources[std::countr_zero(mask)]=values.b;
    pending.info.integer_mask|=integer_mask;pending.info.bool_mask|=bool_mask;
    // No bank reader occurs here. Keep the compatible render pass open even
    // across arbitrarily many writes; storage stays bounded by register count.
}
void GeometryProgramRuntime::FlushUniforms(const GeometryInputState& state) {
    if(state.pending_uniforms.Empty()) return;
    static_assert(sizeof(GeometryInputState::UniformWrite)==32);
    const auto& pending=state.pending_uniforms;auto info=pending.info;
    for(unsigned block=0;block<12;++block)
        if((info.float_masks[block/4]>>((block%4)*8))&0xffu) info.float_blocks|=1u<<block;
    info.boolean_value_offset=std::popcount(info.float_blocks)*8+std::popcount(info.integer_mask);
    const auto count=info.boolean_value_offset+(info.bool_mask ? 4:0);
    const auto values=parameters.Allocate(count*16);auto* destination=static_cast<std::byte*>(values.data);
    auto blocks=info.float_blocks;unsigned offset=0;
    while(blocks) {
        const auto first=unsigned(std::countr_zero(blocks));
        const auto length=unsigned(std::countr_one(blocks>>first));
        std::memcpy(destination+offset*16,pending.floats.data()+first*8,length*128);
        offset+=length*8;blocks&=~(((1u<<length)-1)<<first);
    }
    for(auto mask=info.integer_mask;mask;mask&=mask-1) {
        const auto reg=unsigned(std::countr_zero(mask));
        std::memcpy(destination+offset*16,pending.integers[reg].data(),16);++offset;
    }
    if(info.bool_mask) std::memcpy(destination+offset*16,pending.boolean_sources.data(),64);
    const auto source=parameters.Allocate(sizeof(info));std::memcpy(source.data,&info,sizeof(info));
    const std::array<BufferBinding,3> bindings{{{source.buffer,source.offset,0},{state.uniforms,0,1},
        {values.buffer,values.offset,3}}};
    const auto args=scheduler.MakeArguments(bindings);scheduler.KeepAlive(uniform_pipeline);
    const auto encoder=scheduler.BeginCompute(uniform_pipeline);
    [encoder setComputePipelineState:uniform_pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(101,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    scheduler.EndCompute();state.pending_uniforms.Clear();
}

void GeometryProgramRuntime::EncodeInputs(GeometryInputState& state,const BufferSlice& source,
    std::uint32_t count,std::uint32_t destination,bool ready,std::uint32_t variable_count) {
    FlushUniforms(state);
    struct Params {
        std::uint32_t mode,mask,count,destination,ready,input_count;
        std::array<std::uint32_t,16> registers;
        std::uint32_t begin_count,variable_count;
    };
    static_assert(sizeof(Params)==96);
    const Params info{std::uint32_t(state.config.mode),state.config.vs_output_mask,count,destination,
        ready ? 1U:0U,state.config.input_count,state.config.input_registers,variable_count ? 1U:0U,variable_count};
    const auto params=parameters.Allocate(sizeof(info));std::memcpy(params.data,&info,sizeof(info));
    const std::array<BufferBinding,5> bindings{{
        {count ? source.buffer:empty,count ? source.offset:0,0},{state.point,0,1},
        {state.uniforms,0,2},{state.inputs,0,3},{params.buffer,params.offset,4}}};
    const auto args=scheduler.MakeArguments(bindings);scheduler.KeepAlive(input_pipeline);
    const auto encoder=scheduler.BeginCompute(input_pipeline);
    [encoder setComputePipelineState:input_pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    scheduler.EndCompute();
}
void GeometryProgramRuntime::BeginVariable(GeometryInputState& state,std::uint32_t count) {
    if (!state.NeedsCount() || !count) throw std::invalid_argument("Native variable GS requires a nonzero count at a group boundary");
    const auto main=std::min(count,state.config.variable_main_count);
    const auto size=1+std::uint64_t(main)*state.config.vs_output_count+(count-main);
    if (size>96) throw std::invalid_argument("Native variable GS input group exceeds uniform bank");
    EncodeInputs(state,{},0,0,false,count);
    state.cursor=1;state.remaining=count;state.main_remaining=main;state.total_count=count;
}
bool GeometryProgramRuntime::SubmitVertex(GeometryInputState& state,const BufferSlice& source) {
    if (state.uniforms.device!=device.Handle() || !state.configured || state.awaiting || (state.config.mode==GeometryInputMode::Variable && !state.remaining))
        throw std::invalid_argument("Native GS vertex submitted without a configured/available group");
    if (!source.buffer || source.buffer.device!=device.Handle() || source.offset%16 ||
        source.offset>source.buffer.length || source.size>source.buffer.length-source.offset || source.size<sizeof(VertexRegisters))
        throw std::invalid_argument("Native GS raw VS output is missing, misaligned or too small");
    const auto& config=state.config;
    const auto count=config.mode==GeometryInputMode::Variable && !state.main_remaining ? 1U:config.vs_output_count;
    const auto end=config.mode==GeometryInputMode::Point ? config.input_count:
        config.mode==GeometryInputMode::Fixed ? config.vs_output_count*config.fixed_vertex_count:0;
    const bool ready=config.mode==GeometryInputMode::Variable ? state.remaining==1:state.cursor+count==end;
    const auto destination=config.mode==GeometryInputMode::Fixed ? config.uniform_start+state.cursor:state.cursor;
    EncodeInputs(state,source,count,destination,ready);
    state.cursor+=count;
    if (config.mode==GeometryInputMode::Variable) { --state.remaining;if(state.main_remaining) --state.main_remaining; }
    if (ready) { state.cursor=0;state.awaiting=true; }
    return ready;
}
ShaderMachineResult GeometryProgramRuntime::Invoke(GeometryInputState& state,ShaderMachineRuntime& machine,
    ShaderMachineState& unit,const ShaderMachineProgramView& program,std::uint32_t emission_capacity) {
    if (state.uniforms.device!=device.Handle() || !state.awaiting) throw std::invalid_argument("Native GS invocation requires a complete input group on its owning device");
    FlushUniforms(state);
    std::uint32_t mask{};
    if (state.config.mode==GeometryInputMode::Point)
        for (unsigned i=0;i<state.config.input_count;++i) mask|=1U<<state.config.input_registers[i];
    const BufferSlice uniforms{state.uniforms,0,state.uniforms.length,nullptr,{}};
    const BufferSlice inputs{state.inputs,0,state.inputs.length,nullptr,{}};
    auto result=machine.RunWithUniforms(unit,program,uniforms,inputs,mask,emission_capacity);
    Invoked(state);
    return result;
}
void GeometryProgramRuntime::Invoked(GeometryInputState& state) {
    const std::array<BufferBinding,1> bindings{{{state.uniforms,0,0}}};
    const auto args=scheduler.MakeArguments(bindings);scheduler.KeepAlive(invoked_pipeline);
    const auto encoder=scheduler.BeginCompute(invoked_pipeline);
    [encoder setComputePipelineState:invoked_pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    scheduler.EndEncoding();state.awaiting=false;state.total_count=0;
}
GeometryQueuedInvocation GeometryProgramRuntime::InvokeQueued(GeometryInputState& state,ShaderMachineRuntime& machine,
    ShaderMachineState& unit,const ShaderMachineProgramView& program,const BufferSlice& abort,std::uint32_t emission_capacity) {
    return InvokeQueuedImpl(state,machine,unit,program,abort,emission_capacity,{});
}
GeometryQueuedInvocation GeometryProgramRuntime::InvokeQueuedImpl(GeometryInputState& state,ShaderMachineRuntime& machine,
    ShaderMachineState& unit,const ShaderMachineProgramView& program,const BufferSlice& abort,
    std::uint32_t emission_capacity,const BufferSlice& emissions,bool compact) {
    if(state.uniforms.device!=device.Handle() || !state.awaiting)
        throw std::invalid_argument("Queued native GS invocation requires a complete input group");
    if(compact && !ShaderMachineRuntime::LinearGeometryStorageBound(program))
        throw std::invalid_argument("Compact native GS invocation has no linear storage proof");
    FlushUniforms(state);
    const auto history=snapshots.Allocate(2128);
    GeometryQueuedInvocation pending{.history=history,.config=state.config,.cursor=state.cursor,
        .remaining=state.remaining,.main_remaining=state.main_remaining,.total_count=state.total_count,
        .configured=state.configured,.initialized=state.initialized,.awaiting=state.awaiting};
    std::uint32_t mask{};
    if(state.config.mode==GeometryInputMode::Point)
        for(unsigned i=0;i<state.config.input_count;++i) mask|=1U<<state.config.input_registers[i];
    const BufferSlice uniforms{history.buffer,history.offset,sizeof(ShaderMachineUniforms),nullptr,history.lease};
    const BufferSlice inputs{history.buffer,history.offset+sizeof(ShaderMachineUniforms)+sizeof(VertexRegisters),
        sizeof(VertexRegisters),nullptr,history.lease};
    const ShaderMachineRuntime::GeometryCapture capture{{state.uniforms,state.point,state.inputs},history,
        invoked_pipeline,{state.uniforms,0,0},emissions,compact};
    pending.execution=machine.QueueGeometryImpl(unit,program,uniforms,inputs,abort,mask,emission_capacity,&capture);
    // Snapshot blits, exact scalar GS and known b15 transport remain ordered
    // inside one encoder. CPU bookkeeping never reads or evaluates these banks.
    state.awaiting=false;state.total_count=0;
    return pending;
}
void GeometryProgramRuntime::RestoreQueuedInput(GeometryInputState& state,const GeometryQueuedInvocation& pending) {
    CopyInputHistory(state,pending.history,true);
    state.config=pending.config;state.cursor=pending.cursor;state.remaining=pending.remaining;
    state.main_remaining=pending.main_remaining;state.total_count=pending.total_count;
    state.configured=pending.configured;state.initialized=pending.initialized;state.awaiting=pending.awaiting;
    state.pending_uniforms.Clear();
}
void GeometryProgramRuntime::CopyInputHistory(const GeometryInputState& state,const BufferSlice& history,bool restore) {
    static_assert(sizeof(ShaderMachineUniforms)+2*sizeof(VertexRegisters)==2128);
    if(!history.buffer || history.buffer.device!=device.Handle() || history.offset%4 ||
        history.offset>history.buffer.length || history.size>history.buffer.length-history.offset || history.size<2128)
        throw std::invalid_argument("Invalid native GS input history storage");
    scheduler.UseResource(history.buffer);
    for(auto buffer:{state.uniforms,state.point,state.inputs}) scheduler.UseResource(buffer);
    const auto encoder=scheduler.BeginCompute();NSUInteger offset{};
    for(auto buffer:{state.uniforms,state.point,state.inputs}) {
        if(restore) [encoder copyFromBuffer:history.buffer sourceOffset:history.offset+offset
            toBuffer:buffer destinationOffset:0 size:buffer.length];
        else [encoder copyFromBuffer:buffer sourceOffset:0 toBuffer:history.buffer
            destinationOffset:history.offset+offset size:buffer.length];
        offset+=buffer.length;
    }
    scheduler.EndEncoding();
}
void GeometryProgramRuntime::CompleteQueued(GeometryInputState& state,ShaderMachineRuntime& machine,
    ShaderMachineState& unit,std::span<GeometryQueuedInvocation> pending,
    const std::function<void(const ShaderMachineResult&)>& consume) {
    if(pending.empty()) return;
    if(pending.size()>32 || !consume || state.uniforms.device!=device.Handle())
        throw std::invalid_argument("Invalid native GS completion chunk");
    const auto abort=pending.front().execution.abort;
    for(const auto& item:pending)
        if(item.execution.abort.buffer!=abort.buffer || item.execution.abort.offset!=abort.offset || !item.history.buffer)
            throw std::invalid_argument("Native GS completion chunk mixes guards or lacks immutable input history");
    scheduler.Finish(Scheduler::WaitReason::Readback);
    const auto clear_abort=[&] {
        // All preceding uses of this four-byte range have completed. Other
        // disjoint upload ranges in its shared buffer may still be GPU-owned.
        std::memset(static_cast<std::byte*>(abort.buffer.contents)+abort.offset,0,4);
    };
    const auto fail=[&](GeometryQueuedInvocation& item,const std::array<std::uint32_t,4>& flags) {
        RestoreQueuedInput(state,item);
        machine.RestoreFailedGeometryReset(unit,item.execution);
        throw ShaderMachineError(flags[0],flags[3]);
    };
    for(std::size_t first=0;first<pending.size();) {
        auto& item=pending[first];
        auto flags=machine.PendingStatus(item.execution);
        bool replay_tail=false;
        while(flags[0]) {
            if(flags[0]!=4 || item.execution.capacity==1000000) fail(item,flags);
            // Retire successful prefixes before a large retry. At most one
            // invocation grows at a time, preserving the original peak bound.
            clear_abort();
            item.execution=machine.RetryGeometry(unit,item.execution,std::min(item.execution.capacity*2,1000000u));
            scheduler.Finish(Scheduler::WaitReason::Readback);flags=machine.PendingStatus(item.execution);replay_tail=true;
        }
        const auto retry_capacity=item.execution.capacity;
        consume(machine.ResolvePending(item.execution));
        item={};++first;
        if(replay_tail && first<pending.size()) {
            clear_abort();
            for(auto index=first;index<pending.size();++index) {
                auto& entry=pending[index].execution;
                // Share small learned capacities, but never multiply a huge
                // exceptional allocation across the entire remaining chunk.
                entry=machine.RetryGeometry(unit,entry,std::max(entry.capacity,std::min(retry_capacity,2048u)));
            }
            scheduler.Finish(Scheduler::WaitReason::Readback);
        }
    }
}
GeometryInputSnapshot GeometryProgramRuntime::SaveState(const GeometryInputState& state) {
    if (state.uniforms.device!=device.Handle()) throw std::invalid_argument("Geometry input state belongs to another device");
    FlushUniforms(state);
    GeometryInputSnapshot result{.config=state.config,.cursor=state.cursor,.remaining=state.remaining,
        .main_remaining=state.main_remaining,.total_count=state.total_count,.configured=state.configured,
        .initialized=state.initialized,.awaiting=state.awaiting};
    if (state.initialized) {
        const auto readback=device.CreateBuffer(result.history.size(),true);
        TransferRuntime transfer{device,scheduler};
        NSUInteger offset{};
        for (auto buffer:{state.uniforms,state.point,state.inputs}) {
            transfer.CopyBuffer(buffer,0,readback,offset,buffer.length);offset+=buffer.length;
        }
        scheduler.Finish(Scheduler::WaitReason::Readback);std::memcpy(result.history.data(),readback.contents,result.history.size());
    }
    ValidateState(result);
    return result;
}
void GeometryProgramRuntime::ValidateState(const GeometryInputSnapshot& snapshot) {
    const auto zeros=[](auto begin,auto end) {
        return std::all_of(begin,end,[](std::byte value) { return value==std::byte{}; });
    };
    if (!snapshot.initialized && (!zeros(snapshot.history.begin(),snapshot.history.end()) || snapshot.configured))
        throw std::invalid_argument("Uninitialized GS snapshot contains noncanonical GPU state");
    if (!snapshot.configured) {
        if (snapshot.config!=GeometryInputConfig{} || snapshot.cursor || snapshot.remaining ||
            snapshot.main_remaining || snapshot.total_count || snapshot.awaiting ||
            !zeros(snapshot.history.begin()+1616,snapshot.history.end()))
            throw std::invalid_argument("Unconfigured GS snapshot contains group metadata/values");
    } else {
        const auto& config=snapshot.config;ValidateConfig(config);
        if (config.mode==GeometryInputMode::Variable) {
            if (!snapshot.total_count) {
                if (snapshot.cursor || snapshot.remaining || snapshot.main_remaining || snapshot.awaiting)
                    throw std::invalid_argument("Empty variable GS snapshot contains pending metadata");
            } else {
                const auto main=std::min(snapshot.total_count,config.variable_main_count);
                const auto size=1+std::uint64_t(main)*config.vs_output_count+(snapshot.total_count-main);
                if (size>96 || snapshot.remaining>snapshot.total_count)
                    throw std::invalid_argument("Variable GS snapshot exceeds uniform bank/count");
                const auto consumed=snapshot.total_count-snapshot.remaining;
                const auto expected=1+std::uint64_t(std::min(consumed,main))*config.vs_output_count+
                    (consumed>main ? consumed-main:0);
                if (snapshot.main_remaining!=(consumed<main ? main-consumed:0) ||
                    (snapshot.awaiting ? snapshot.remaining!=0 || snapshot.cursor!=0 :
                     snapshot.remaining==0 || snapshot.cursor!=expected))
                    throw std::invalid_argument("Variable GS snapshot cursor does not match its group progress");
            }
        } else {
            const auto limit=config.mode==GeometryInputMode::Point ? config.input_count:
                config.vs_output_count*config.fixed_vertex_count;
            if (snapshot.remaining || snapshot.main_remaining || snapshot.total_count ||
                snapshot.cursor>=limit || snapshot.cursor%config.vs_output_count || (snapshot.awaiting && snapshot.cursor))
                throw std::invalid_argument("Point/fixed GS snapshot contains invalid group progress");
        }
    }
    if (!zeros(snapshot.history.begin()+1604,snapshot.history.begin()+1616))
        throw std::invalid_argument("GS uniform snapshot contains nonzero padding");
    if (snapshot.history[1602]!=std::byte{} || snapshot.history[1603]!=std::byte{})
        throw std::invalid_argument("GS uniform snapshot contains unknown boolean bits");
}
void GeometryProgramRuntime::LoadState(GeometryInputState& state,const GeometryInputSnapshot& snapshot) {
    if (state.uniforms.device!=device.Handle()) throw std::invalid_argument("Geometry input state belongs to another device");
    ValidateState(snapshot);
    if (snapshot.initialized) {
        const auto source=parameters.Allocate(snapshot.history.size());
        std::memcpy(source.data,snapshot.history.data(),snapshot.history.size());
        TransferRuntime transfer{device,scheduler};
        NSUInteger offset{};
        for (auto buffer:{state.uniforms,state.point,state.inputs}) {
            transfer.CopyBuffer(source.buffer,source.offset+offset,buffer,0,buffer.length);offset+=buffer.length;
        }
    }
    state.pending_uniforms.Clear();
    state.config=snapshot.config;state.cursor=snapshot.cursor;state.remaining=snapshot.remaining;
    state.main_remaining=snapshot.main_remaining;state.total_count=snapshot.total_count;
    state.configured=snapshot.configured;state.initialized=snapshot.initialized;state.awaiting=snapshot.awaiting;
}
}
