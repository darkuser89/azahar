// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_shader_machine.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include "mtl_compiler_gate_fixture.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <set>
namespace Metal {
struct InputReferenceTestAccess {
    static ShaderMachineVertexBatchResult Owned(ShaderMachineRuntime& machine,
        VertexFetchRuntime& fetch,VertexInputState& input,std::span<const BufferSlice> sources,
        const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgram& program,
        const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config,
        unsigned vertices,unsigned elements,unsigned base,unsigned first,const BufferSlice& indices={},
        unsigned width=0,unsigned limit=1000000,bool deferred=true) {
        return machine.RunOwnedFetchedVertexBatchAndCapture(fetch,input,sources,layout,state,
            program,uniforms,config,vertices,elements,base,first,indices,width,limit,deferred);
    }
    static bool CachedRawMatches(ShaderMachineRuntime& machine,const ShaderMachineProgram& program,
                                 const ShaderMachineUniforms& uniforms,bool required=true) {
        auto& cached=machine.GetProgram(program);
        std::array<std::uint32_t,17> control{uniforms.b};
        std::memcpy(control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
        const auto found=cached.control_proofs.find(control);
        if(found==cached.control_proofs.end()) return !required;
        ShaderMachineRuntime::VertexProof expected{};
        expected.execution=ShaderMachineRuntime::AnalyzeVertices(program,uniforms,&expected.unproven,&expected.passes,
            &expected.recurrence_mask,&expected.recurrence_capacity,&expected.always_written,&expected.possibly_written);
        if((expected.execution==ShaderMachineRuntime::VertexExecution::DependencyPasses ||
            expected.execution==ShaderMachineRuntime::VertexExecution::Ordered) &&
           ShaderMachineRuntime::ReachableVertexInstructions(program).count()<=512)
            expected.instruction_bound=ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms).value_or(0);
        const auto& actual=found->second;
        return actual.execution==expected.execution && actual.unproven==expected.unproven &&
            actual.passes==expected.passes && actual.recurrence_mask==expected.recurrence_mask &&
            actual.recurrence_capacity==expected.recurrence_capacity && actual.always_written==expected.always_written &&
            actual.possibly_written==expected.possibly_written && actual.instruction_bound==expected.instruction_bound;
    }
    static std::pair<unsigned,unsigned> ProofCounts(ShaderMachineRuntime& machine,const ShaderMachineProgram& program) {
        const auto& cached=machine.GetProgram(program);
        return {unsigned(cached.control_proofs.size()),unsigned(cached.native_reset_proofs.size())};
    }
    static void Evict(ShaderMachineRuntime& machine,const ShaderMachineProgram& program) {
        auto other=program;
        for(unsigned i=0;i<257;++i) {other.code[4095]=0x40000000u+i;machine.GetProgram(other);}
    }
};
struct SchedulerTestAccess {
    static void BlockQueue(Scheduler& scheduler, id<MTLSharedEvent> event) {
        scheduler.CheckThread();
        [scheduler.queue waitForEvent:event value:1];
    }
};
}
namespace {
void Check(bool yes,const char* message) { if(!yes) throw std::runtime_error(message); }
}
void CheckNativeInstructionBounds() {
    using Runtime=Metal::ShaderMachineRuntime;
    using Op=nihstro::OpCode::Id;
    const auto flow=[](unsigned op,unsigned target,unsigned length=0,unsigned id=0) {
        return op<<26 | id<<22 | target<<10 | length;
    };
    const auto blank=[] {
        Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=7;
        p.code.fill(0x21u<<26);p.swizzle[0]=Metal::Test::IdentitySwizzle;return p;
    };
    unsigned checked=0;
    const auto expect=[&](const Metal::ShaderMachineProgram& p,const Metal::ShaderMachineUniforms& u,
                          std::optional<unsigned> wanted) {
        const auto actual=Runtime::NativeVertexInstructionBound(p,u);
        if(actual!=wanted)throw std::runtime_error("Conservative native instruction bound case "+
            std::to_string(checked)+": expected "+(wanted ? std::to_string(*wanted):"none")+
            ", got "+(actual ? std::to_string(*actual):"none"));
        ++checked;
    };
    auto p=blank();p.code[0]=unsigned(Op::END)<<26;expect(p,{},1);
    p.entry=4095;p.code[4095]=unsigned(Op::END)<<26;expect(p,{},1);
    p.entry=4096;expect(p,{},{});p=blank();expect(p,{},{});
    p.code[5]=unsigned(Op::END)<<26;expect(p,{},6);
    p.allow_emission=true;expect(p,{},{});p.allow_emission=false;p.output_mask=65536;expect(p,{},{});
    p=blank();p.code[0]=flow(0x27,3,1);p.code[4]=unsigned(Op::END)<<26;expect(p,{},4);
    p.code[0]=flow(0x28,3,1);expect(p,{},4);
    p.code[1]=unsigned(Op::END)<<26;expect(p,{},3); // Early END: conservatively includes the later tail.
    p.code[3]=unsigned(Op::END)<<26;expect(p,{},2);
    for(unsigned count:{0u,1u,6u,255u,256u,511u}) {
        p=blank();p.code[0]=flow(0x29,2);p.code[3]=unsigned(Op::END)<<26;
        Metal::ShaderMachineUniforms u{};u.i[0][0]=count;expect(p,u,2+2*((count&255)+1));
    }
    p=blank();p.code[0]=flow(0x29,6);p.code[1]=flow(0x29,4,0,1);p.code[7]=unsigned(Op::END)<<26;
    Metal::ShaderMachineUniforms u{};u.i[0][0]=6;u.i[1][0]=1;expect(p,u,65);
    u.i[0][0]=255;u.i[1][0]=255;expect(p,u,197378);
    p=blank();p.code[0]=flow(0x29,7);p.code[1]=flow(0x29,5,0,1);p.code[2]=flow(0x29,3,0,2);
    p.code[8]=unsigned(Op::END)<<26;u.i[2][0]=255;expect(p,u,{}); // More than1M, without integer overflow.
    for(unsigned opcode:{0x20u,0x23u,0x2cu,0x2du,0x10u}) {
        p=blank();p.code[0]=flow(opcode,2,1);p.code[1]=unsigned(Op::END)<<26;p.code[3]=unsigned(Op::END)<<26;
        expect(p,{},{});
    }
    for(unsigned opcode:{0x24u,0x25u,0x26u}) {
        p=blank();p.code[0]=flow(opcode,16,3);p.code[1]=unsigned(Op::END)<<26;
        expect(p,{},5); // CALL plus its three instructions and the caller END.
        p.code[0]=flow(opcode,16,0);expect(p,{},2);
        p.code[0]=flow(opcode,4095,2);expect(p,{},{});
        p.code[0]=flow(opcode,16,3);p.code[16]=unsigned(Op::END)<<26;
        expect(p,{},opcode==0x24u ? 2:3); // Conditional END may be skipped.
    }
    p=blank();p.code[0]=flow(0x24,16,2);p.code[1]=unsigned(Op::END)<<26;
    p.code[16]=flow(0x24,32,2);p.code[32]=flow(0x24,48,2);expect(p,{},8);
    p.code[48]=flow(0x24,64,1);expect(p,{},{}); // Structural stack depth still applies.
    p.code[16]=flow(0x24,16,2);expect(p,{},{}); // Recursive range is never accepted.
    p=blank();p.code[0]=flow(0x24,16,3);p.code[1]=unsigned(Op::END)<<26;
    p.code[16]=flow(0x29,17);u={};u.i[0][0]=255;expect(p,u,260);
    p.code[16]=flow(0x2c,17);expect(p,{},{}); // Calls do not admit excluded JMPs.
    p=blank();for(unsigned i=0;i<3;++i)p.code[i]=flow(0x24,16,255);
    p.code[3]=unsigned(Op::END)<<26;expect(p,{},769);
    p.code[3]=flow(0x24,16,255);p.code[4]=unsigned(Op::END)<<26;
    expect(p,{},{}); // Repeated callees must also fit the expanded native source budget.
    p=blank();p.code[0]=flow(0x27,4095,255);expect(p,{},{});
    p=blank();p.code[0]=flow(0x29,0);p.code[1]=unsigned(Op::END)<<26;expect(p,{},{});
    p=blank();p.code[0]=flow(0x29,3);p.code[1]=flow(0x29,3,0,1);p.code[4]=unsigned(Op::END)<<26;expect(p,{},{});
    p=blank();for(unsigned i=0;i<5;++i)p.code[i]=flow(0x29,11-i,0,i%4);
    p.code[12]=unsigned(Op::END)<<26;expect(p,{},{});
    std::cout<<checked<<" conservative instruction-bound cases: END/IF/LOOP/CALL/CALLC/CALLU, all loop-count byte limits, nested costs, malformed/unterminated/control flow/over-budget rejection passed\n";
}

void CheckBoundedOwnedControl(Metal::Device& device) {
    using Op=nihstro::OpCode::Id;
    Metal::ShaderMachineProgram original{};original.allow_emission=false;original.output_mask=7;
    original.swizzle[0]=Metal::Test::IdentitySwizzle;
    original.code[0]=Metal::Test::Arithmetic(Op::MOV,0,16);
    original.code[1]=Metal::Test::Arithmetic(Op::MOV,16,0);
    original.code[2]=0x29u<<26 | 6u<<10;
    original.code[3]=0x27u<<26 | 5u<<10 | 1u;
    original.code[4]=Metal::Test::Arithmetic(Op::MOV,1,32);
    original.code[5]=Metal::Test::Arithmetic(Op::MOV,1,33);
    // Make iteration count observable in the raw/captured output, not only in
    // deferred replay: read a distinct uniform through the live loop address.
    original.code[6]=Metal::Test::Arithmetic(Op::MOV,2,32) | 3u<<19;
    original.code[7]=unsigned(Op::END)<<26;
    Metal::Scheduler queue{device};Metal::ShaderMachineRuntime machine{device,queue};
    Metal::VertexFetchRuntime fetch{device,queue};Metal::TransferRuntime transfer{device,queue};
    Metal::PicaVertexLayout layout{};layout.mapped_attribute_count=16;
    for(unsigned i=0;i<16;++i) {
        layout.register_map[i]=i;layout.attribute_inputs[i].mode=Metal::PicaAttributeMode::Fixed;
        layout.fixed_attributes[i]={float(i)*.125f,-.25f,.5f,1};
    }
    Metal::GeometryEmissionConfig config{};config.attribute_count=3;config.honor_winding=0;
    config.semantics={0x03020100,0x07060504,0x0b0a0908};
    unsigned checked=0;
    for(unsigned call:{0u,0x24u,0x25u,0x26u}) {
    auto program=original;
    if(call) {
        program.code[2]=call<<26 | 16u<<10 | 6u;
        program.code[3]=unsigned(Op::END)<<26;
        program.code[16]=0x29u<<26 | 20u<<10;
        program.code[17]=0x27u<<26 | 19u<<10 | 1u;
        program.code[18]=original.code[4];program.code[19]=original.code[5];
        program.code[20]=original.code[6];program.code[21]=0x21u<<26;
    }
    for(unsigned count:{0u,1u,6u,255u,256u,511u})for(unsigned boolean:{0u,1u}) {
        Metal::ShaderMachineUniforms uniforms{};uniforms.b=boolean;uniforms.i[0]={count,2,3,0};
        for(unsigned i=0;i<uniforms.f.size();++i)uniforms.f[i]={float(i)*.03125f,-.25f,.5f,1};
        uniforms.f[0]={.125f,-.5f,.25f,1};uniforms.f[1]={-.75f,.25f,.5f,1};
        if(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)!=
            Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses)
            throw std::runtime_error("Bounded control dependency proof: call="+std::to_string(call)+
                ", count="+std::to_string(count)+", bool="+std::to_string(boolean));
        Check(Metal::ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms)==(call ? 6u:4u)+3*((count&255)+1),
            "Bounded control fixture cost differs");
        Metal::VertexInputState reference_input{device},input{device};
        Metal::ShaderMachineState reference{device},state{device};
        const auto expected=machine.RunFetchedVertexBatchAndCapture(fetch,reference_input,{},layout,reference,
            program,uniforms,config,6,6,9,9);
        const auto actual=Metal::InputReferenceTestAccess::Owned(machine,fetch,input,{},layout,state,
            program,uniforms,config,6,6,9,9);
        Check(state.HasDeferredVertexBatch(),"Bounded control fixture failed to use fresh owned state");
        const std::array<Metal::BufferSlice,4> slices{expected.output,actual.output,expected.captured,actual.captured};
        std::array<id<MTLBuffer>,4> reads{};
        for(unsigned i=0;i<4;++i) {
            reads[i]=device.CreateBuffer(slices[i].size,true);
            transfer.CopyBuffer(slices[i].buffer,slices[i].offset,reads[i],0,slices[i].size);
        }
        queue.Finish();
        Check(!std::memcmp(reads[0].contents,reads[1].contents,reads[0].length) &&
              !std::memcmp(reads[2].contents,reads[3].contents,reads[2].length) &&
              machine.SaveState(state)==machine.SaveState(reference) &&
              fetch.SaveState(input).history==fetch.SaveState(reference_input).history,
              "Bounded native control changed full raw/capture/1824/512-byte state");
        // Short caller budgets must still use the guarded path, with the same PC and unchanged state.
        std::string reference_error;
        for(bool owned:{false,true}) {
            Metal::ShaderMachineState limited{device};Metal::VertexInputState limited_input{device};
            bool caught=false;
            try {
                if(owned)Metal::InputReferenceTestAccess::Owned(machine,fetch,limited_input,{},layout,limited,
                    program,uniforms,config,6,6,9,9,{},0,3,false);
                else machine.RunFetchedVertexBatchAndCapture(fetch,limited_input,{},layout,limited,
                    program,uniforms,config,6,6,9,9,{},0,3,false);
            }catch(const Metal::ShaderMachineError& e){
                caught=e.Code()==6;
                if(!owned)reference_error=e.what();
                else caught=caught && reference_error==e.what();
            }
            Check(caught && machine.SaveState(limited)==Metal::ShaderMachineSnapshot{},
                "Bounded native control bypassed a short limit or changed its failure PC/state");
        }
        ++checked;
    }
    }
    std::cout<<checked<<" bounded owned IF/LOOP/CALL/CALLC/CALLU control cases: full raw/capture/1824/512-byte state and 96 short-budget errors exact\n";
}

void CheckOwnedDeferredState(Metal::Device& device) {
    using Op=nihstro::OpCode::Id;
    Metal::ShaderMachineProgram dependent{};dependent.allow_emission=false;dependent.output_mask=7;
    dependent.swizzle[0]=Metal::Test::IdentitySwizzle;
    dependent.code[0]=Metal::Test::Arithmetic(Op::MOV,0,16);
    dependent.code[1]=Metal::Test::Arithmetic(Op::MOV,16,0);
    dependent.code[2]=Metal::Test::Arithmetic(Op::MOV,1,32);
    dependent.code[3]=Metal::Test::Arithmetic(Op::MOV,2,1);
    dependent.code[4]=unsigned(Op::END)<<26;
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(dependent,{})==
        Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,"Owned-state fixture lost dependency proof");
    Metal::Scheduler reference_queue{device},queue{device};
    Metal::ShaderMachineRuntime oracle{device,reference_queue},machine{device,queue};
    Metal::VertexFetchRuntime oracle_fetch{device,reference_queue},fetch{device,queue};
    Metal::GeometryRuntime oracle_geometry{device,reference_queue},geometry{device,queue};
    Metal::TransferRuntime reference_copies{device,reference_queue},copies{device,queue};
    struct Trial {
        unsigned count{},vertices{},base{},first{},width{};bool deferred{};
        Metal::ShaderMachineProgram program;
        Metal::ShaderMachineUniforms uniforms{};
        Metal::PicaVertexLayout layout{};Metal::GeometryEmissionConfig config{};
        Metal::ShaderMachineSnapshot initial{},final{};
        Metal::VertexInputSnapshot input_initial{},input_final{};
        Metal::PrimitiveSnapshot primitive_final{};
        Metal::TriangleTopology topology{};
        Metal::BufferSlice source,indices,canonical;
        std::unique_ptr<Metal::ShaderMachineState> state;
        std::unique_ptr<Metal::VertexInputState> input;
        std::unique_ptr<Metal::PrimitiveStream> stream;
        std::array<id<MTLBuffer>,3> expected{},read{};
    };
    std::vector<Trial> trials;
    for(unsigned n=0;n<48;++n) {
        Trial t;t.count=std::array<unsigned,5>{1,2,3,6,8}[n%5];t.vertices=t.count+3;
        t.base=9+n;t.first=t.base+1;t.program=dependent;t.topology=Metal::TriangleTopology(n%3);
        if(n>=30)t.width=std::array<unsigned,4>{0,1,2,4}[(n-30)%4];
        if(n==30){t.count=9;t.vertices=12;}
        t.initial.needs_reset=n<30 || n==30 || n%4==0;
        if(!t.initial.needs_reset) {
            for(unsigned i=0;i<448;++i) {
                const float value=float(i%23)*.015625f;std::memcpy(t.initial.history.data()+i*4,&value,4);
            }
            const std::array<int,4> address{3,-2,1,0};const std::array<unsigned,4> flags{1,0,n%2,9};
            std::memcpy(t.initial.history.data()+1792,address.data(),16);
            std::memcpy(t.initial.history.data()+1808,flags.data(),16);
        }
        if(n>=45) {t.program.code[0]=Metal::Test::Arithmetic(Op::MOV,0,0);t.initial={};t.width=0;}
        t.deferred=t.initial.needs_reset && !t.width && t.count<=8 && n<45;
        t.uniforms.f[0]={float(n)*.0625f,-.5f,.25f,1};t.uniforms.b=n;t.uniforms.i[0]={n,3,2,1};
        auto bytes=device.CreateBuffer(3+t.vertices*263,true);
        for(NSUInteger i=0;i<bytes.length;++i)static_cast<unsigned char*>(bytes.contents)[i]=
            std::array<unsigned char,4>{0,0x80,0x3f,0}[i%4];
        t.source={bytes,3,t.vertices*263,static_cast<std::byte*>(bytes.contents)+3,{}};
        t.layout.mapped_attribute_count=n%17;
        for(unsigned a=0;a<16;++a) {
            t.layout.register_map[a]=(a*7+n)%16;t.layout.fixed_attributes[a]={float(a)*.125f,-.25f,.5f,1};
            auto& input=t.layout.attribute_inputs[a];input.mode=Metal::PicaAttributeMode((a+n)%3);
            input.fetch={0,a*16+1,263,1+(a+n)%4,Metal::VertexAttributeFormat((a+n)%4),a};
        }
        t.layout.register_map[15]=t.layout.register_map[0];t.input_initial.needs_reset=n%2;
        if(!t.input_initial.needs_reset)for(unsigned i=0;i<128;++i) {
            const float value=float(int(i%19)-9)*.03125f;std::memcpy(t.input_initial.history.data()+i*4,&value,4);
        }
        auto index=device.CreateBuffer(3+std::max(t.width,1u)*t.count,true);
        for(unsigned e=0;e<t.count;++e)for(unsigned b=0;b<t.width;++b)
            static_cast<unsigned char*>(index.contents)[3+e*t.width+b]=(t.base+(e*3)%t.vertices)>>(b*8);
        t.indices={index,3,t.width*t.count,nullptr,{}};
        t.config.attribute_count=3;t.config.honor_winding=0;t.config.flip_viewport=n%2;t.config.enable_clip=(n/2)%2;
        t.config.semantics={0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x1f171614,0x1f1f1f1f};
        t.config.clip_coefficient={.25f,-.5f,.125f,1};
        const std::array<Metal::BufferSlice,1> sources{t.source};
        Metal::VertexInputState input{device};oracle_fetch.LoadState(input,t.input_initial);
        t.canonical=oracle_fetch.FetchPica(input,sources,t.layout,t.vertices,t.count,t.base,t.first,
            t.width ? t.indices:Metal::BufferSlice{},t.width);
        t.input_final=oracle_fetch.SaveState(input);
        Metal::ShaderMachineState state{device};oracle.LoadState(state,t.initial);
        const auto raw=oracle.RunVertexBatch(state,t.program,t.uniforms,t.canonical,t.vertices,t.count,t.base,t.first,
            t.width ? t.indices:Metal::BufferSlice{},t.width);
        const auto captured=oracle.CaptureVertices(raw,t.program.output_mask,t.config,t.count);
        t.final=oracle.SaveState(state);
        Metal::PrimitiveStream stream{device};
        const auto assembled=oracle_geometry.Append(stream,captured,t.count,t.topology,t.count,0,0,{},0,n%2);
        t.primitive_final=oracle_geometry.SaveStream(stream);
        const std::array<Metal::BufferSlice,3> outputs{raw,captured,assembled.storage};
        for(unsigned slot=0;slot<3;++slot)if(slot!=2 || assembled.vertex_count) {
            t.expected[slot]=device.CreateBuffer(outputs[slot].size,true);
            reference_copies.CopyBuffer(outputs[slot].buffer,outputs[slot].offset,t.expected[slot],0,outputs[slot].size);
            t.read[slot]=device.CreateBuffer(outputs[slot].size+32,true);std::memset(t.read[slot].contents,0xA7,t.read[slot].length);
        }
        t.state=std::make_unique<Metal::ShaderMachineState>(device);machine.LoadState(*t.state,t.initial);
        t.input=std::make_unique<Metal::VertexInputState>(device);fetch.LoadState(*t.input,t.input_initial);
        t.stream=std::make_unique<Metal::PrimitiveStream>(device);trials.push_back(std::move(t));
    }
    reference_queue.Finish();queue.Finish();
    auto gate=[device.Handle() newSharedEvent];
    struct ReleaseEvent {id<MTLSharedEvent> event;~ReleaseEvent(){event.signaledValue=1;}} release{gate};
    Metal::SchedulerTestAccess::BlockQueue(queue,gate);
    const auto tick=queue.CurrentTick();unsigned deferred=0;
    auto zero=device.CreateBuffer(16*263,true);std::memset(zero.contents,0,zero.length);
    for(unsigned n=0;n<trials.size();++n) {
        auto& t=trials[n];auto program=t.program;auto uniforms=t.uniforms;auto layout=t.layout;auto config=t.config;
        const std::array<Metal::BufferSlice,1> sources{t.source};Metal::AssembledVertices assembled;
        Metal::ShaderMachineVertexBatchResult result;
        std::unique_ptr<Metal::Test::CompilerGate> compiler_gate;
        if (!n) compiler_gate=std::make_unique<Metal::Test::CompilerGate>(device);
        const auto before_worker=queue.WorkerStats();
        const auto before_encoders=queue.EncoderCount();
        {
            Metal::Scheduler::BufferComputeScope buffers{queue};Metal::Scheduler::ComputeBatchScope batch{queue};
            result=Metal::InputReferenceTestAccess::Owned(machine,fetch,*t.input,sources,layout,*t.state,
                program,uniforms,config,t.vertices,t.count,t.base,t.first,t.width ? t.indices:Metal::BufferSlice{},t.width);
            assembled=geometry.Append(*t.stream,result.captured,t.count,t.topology,t.count,0,0,{},0,n%2);
        }
        if (compiler_gate) {
            const auto held=queue.WorkerStats();
            const bool one_encoder=queue.EncoderCount()==before_encoders+1;
            const bool timely=compiler_gate->Open();
            Check(timely && one_encoder && held.queued_dispatches>=before_worker.queued_dispatches+5 &&
                      held.encoded_dispatches==before_worker.encoded_dispatches,
                  "Fetched VS/capture/primitive batch waited for its compiler, split its encoder or encoded a partial chain");
            std::cout<<"Compiler-gated owned fetch+history+VS+capture+primitive append: owner and scope exit return, one worker encoder, all native dispatches retained\n";
        }
        Check(t.state->HasDeferredVertexBatch()==t.deferred,"Owned-state eligibility or publication differs from its proof");
        deferred+=t.deferred;program={};uniforms={};layout={};config={};
        const std::array<Metal::BufferSlice,3> outputs{result.output,result.captured,assembled.storage};
        for(unsigned slot=0;slot<3;++slot)if(t.read[slot])
            copies.CopyBuffer(outputs[slot].buffer,outputs[slot].offset,t.read[slot],16,outputs[slot].size);
        // A deferred state must retain fetched values, not replay these borrowed
        // bytes after a later ordered GPU writer changes and releases them.
        copies.CopyBuffer(zero,0,t.source.buffer,t.source.offset,t.source.size);t.source={};t.indices={};
    }
    Check(queue.CurrentTick()==tick,"Owned-state batch caused per-draw submission/wait");
    Metal::InputReferenceTestAccess::Evict(machine,dependent);queue.SubmitPending();
    Check(!queue.IsComplete(tick),"Native event failed to hold owned-state GPU work");
    gate.signaledValue=1;queue.Finish();
    for(unsigned n=0;n<trials.size();++n) {
        auto& t=trials[n];
        for(unsigned slot=0;slot<3;++slot)if(t.read[slot]) {
            const auto bytes=t.expected[slot].length;
            Check(!std::memcmp(static_cast<std::byte*>(t.read[slot].contents)+16,t.expected[slot].contents,bytes),
                "Owned-state path changed guarded complete raw/captured/assembled output");
            for(unsigned i=0;i<16;++i)Check(static_cast<unsigned char*>(t.read[slot].contents)[i]==0xA7 &&
                static_cast<unsigned char*>(t.read[slot].contents)[16+bytes+i]==0xA7,"Owned-state output damaged guards");
        }
        const auto input=fetch.SaveState(*t.input);const auto primitive=geometry.SaveStream(*t.stream);
        Check(input.needs_reset==t.input_final.needs_reset && input.history==t.input_final.history &&
            primitive.needs_reset==t.primitive_final.needs_reset && primitive.pending==t.primitive_final.pending &&
            primitive.topology==t.primitive_final.topology && primitive.history==t.primitive_final.history,
            "Owned-state path changed complete512/272-byte input/primitive states");
        if(n%3==0 && t.deferred) {
            Metal::ShaderMachineState next{device};oracle.LoadState(next,t.final);
            const auto expected=oracle.RunVertexBatch(next,t.program,t.uniforms,t.canonical,t.vertices,1,t.base,t.first);
            const auto actual=machine.RunVertexBatch(*t.state,t.program,t.uniforms,t.canonical,t.vertices,1,t.base,t.first);
            auto a=device.CreateBuffer(actual.size,true),b=device.CreateBuffer(expected.size,true);
            copies.CopyBuffer(actual.buffer,actual.offset,a,0,actual.size);
            reference_copies.CopyBuffer(expected.buffer,expected.offset,b,0,expected.size);
            reference_queue.Finish();queue.Finish();
            Check(!std::memcmp(a.contents,b.contents,a.length) && machine.SaveState(*t.state)==oracle.SaveState(next),
                "Inherited invocation failed to materialize exact owned full state");
        } else Check(machine.SaveState(*t.state)==t.final,"Owned deferred full1824-byte state differs after mutation/cache eviction");
        Check(!t.state->HasDeferredVertexBatch(),"Observed owned state was not materialized");
    }
    Metal::PicaVertexLayout fixed{};fixed.mapped_attribute_count=16;
    for(unsigned a=0;a<16;++a) {
        fixed.register_map[a]=a;fixed.attribute_inputs[a].mode=Metal::PicaAttributeMode::Fixed;
        fixed.fixed_attributes[a]={float(a)*.125f,-.25f,.5f,1};
    }
    Metal::GeometryEmissionConfig config{};config.attribute_count=3;config.honor_winding=0;
    config.semantics={0x03020100,0x07060504,0x0b0a0908};
    for(unsigned policy=0;policy<3;++policy) {
        Metal::ShaderMachineState state{device},reference{device};
        Metal::VertexInputState input{device},reference_input{device};
        Metal::InputReferenceTestAccess::Owned(machine,fetch,input,{},fixed,state,dependent,{},config,6,6,9,9);
        Check(state.HasDeferredVertexBatch(),"Discard/continuation fixture did not defer full state");
        oracle.RunFetchedVertexBatchAndCapture(oracle_fetch,reference_input,{},fixed,reference,dependent,{},config,6,6,9,9);
        const auto expected=oracle.SaveState(reference);
        if(policy==0) {
            const auto count=queue.EncoderCount();state.Reset();
            Check(machine.SaveState(state)==Metal::ShaderMachineSnapshot{} && queue.EncoderCount()==count,
                "Reset materialized discarded deferred state or retained noncanonical bytes");
        } else if(policy==1) {
            machine.LoadState(state,expected);
            Check(!state.HasDeferredVertexBatch() && machine.SaveState(state)==expected,
                "LoadState failed to discard and replace exact owned deferred state");
        } else {
            const auto actual=Metal::InputReferenceTestAccess::Owned(machine,fetch,input,{},fixed,state,dependent,{},config,6,6,9,9);
            const auto expected_output=oracle.RunFetchedVertexBatchAndCapture(oracle_fetch,reference_input,{},fixed,reference,
                dependent,{},config,6,6,9,9);
            auto a=device.CreateBuffer(actual.output.size,true),b=device.CreateBuffer(expected_output.output.size,true);
            copies.CopyBuffer(actual.output.buffer,actual.output.offset,a,0,a.length);
            reference_copies.CopyBuffer(expected_output.output.buffer,expected_output.output.offset,b,0,b.length);
            reference_queue.Finish();queue.Finish();
            Check(!std::memcmp(a.contents,b.contents,a.length) && !state.HasDeferredVertexBatch() &&
                machine.SaveState(state)==oracle.SaveState(reference),"Private inherited call changed exact materialization/state");
        }
    }
    for(unsigned inherited=0;inherited<2;++inherited)for(unsigned failure=0;failure<2;++failure) {
        auto program=dependent;if(failure)program.code[1]=0x10u<<26;
        const unsigned limit=failure ? 1000000:1;
        Metal::ShaderMachineState state{device},reference{device};
        Metal::VertexInputState input{device},reference_input{device};
        Metal::ShaderMachineSnapshot initial{};
        if(inherited) {initial=trials[31].initial;initial.needs_reset=false;}
        machine.LoadState(state,initial);oracle.LoadState(reference,initial);
        std::string actual_error,reference_error;
        try {oracle.RunFetchedVertexBatchAndCapture(oracle_fetch,reference_input,{},fixed,reference,program,{},config,6,6,9,9,{},0,limit);}
        catch(const Metal::ShaderMachineError& e){reference_error=e.what();}
        try {Metal::InputReferenceTestAccess::Owned(machine,fetch,input,{},fixed,state,program,{},config,6,6,9,9,{},0,limit,false);}
        catch(const Metal::ShaderMachineError& e){actual_error=e.what();}
        const auto actual_input=fetch.SaveState(input),expected_input=oracle_fetch.SaveState(reference_input);
        Check(!actual_error.empty() && actual_error==reference_error && machine.SaveState(state)==initial &&
            oracle.SaveState(reference)==initial && actual_input.history==expected_input.history &&
            actual_input.needs_reset==expected_input.needs_reset,"Owned short-limit/invalid opcode changed exact error PC or failure states");
    }
    for(unsigned failure=0;failure<2;++failure) {
        Metal::Scheduler failed_queue{device};Metal::ShaderMachineRuntime failed_machine{device,failed_queue};
        Metal::VertexFetchRuntime failed_fetch{device,failed_queue};Metal::VertexInputState input{device};
        Metal::ShaderMachineState state{device};auto program=dependent;if(failure)program.code[1]=0x10u<<26;
        const auto tick=failed_queue.CurrentTick();
        Metal::InputReferenceTestAccess::Owned(failed_machine,failed_fetch,input,{},fixed,state,program,{},config,
            6,6,9,9,{},0,failure ? 1000000:1,true);
        Check(!state.HasDeferredVertexBatch() && failed_queue.CurrentTick()==tick,"Owned error path deferred full state or submitted early");
        bool caught=false,sticky=false;const unsigned code=failure ? 2:6;
        try{failed_queue.Finish();}catch(const Metal::ShaderMachineError& e){caught=e.Code()==code && std::string(e.what()).ends_with(failure ? "instruction 2":"instruction 1");}
        try{failed_queue.IsComplete(0);}catch(const Metal::ShaderMachineError& e){sticky=e.Code()==code;}
        Check(caught && sticky,"Owned GPU failure lost exact code/PC or sticky validation");
    }
    {
        const auto path=std::filesystem::temp_directory_path()/("azahar-owned-state-diagnostic-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
        struct Environment {
            std::optional<std::string> old;
            Environment(const std::filesystem::path& path) {
                if(const auto* value=std::getenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS"))old=value;
                setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",path.c_str(),1);
            }
            ~Environment(){if(old)setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",old->c_str(),1);else unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");}
        } environment{path};
        {std::ofstream request(path/"capture.request");request<<1;}
        Metal::Scheduler diagnostic_queue{device};Metal::ShaderMachineRuntime diagnostic{device,diagnostic_queue};
        Metal::VertexFetchRuntime diagnostic_fetch{device,diagnostic_queue};Metal::VertexInputState input{device};
        Metal::ShaderMachineState state{device},reference{device};Metal::VertexInputState reference_input{device};
        Metal::InputReferenceTestAccess::Owned(diagnostic,diagnostic_fetch,input,{},fixed,state,dependent,{},config,6,6,9,9);
        oracle.RunFetchedVertexBatchAndCapture(oracle_fetch,reference_input,{},fixed,reference,dependent,{},config,6,6,9,9);
        Check(!state.HasDeferredVertexBatch() && diagnostic.SaveState(state)==oracle.SaveState(reference),
            "Owned diagnostics failed to retain its eager complete-state contract");
        unsigned count=0;for(const auto& item:std::filesystem::directory_iterator(path))count+=item.path().extension()==".vdc";
        Check(count==1,"Owned diagnostic state check did not actually capture a full record");
        std::filesystem::remove_all(path);
    }
    std::cout<<"Owned-state reset/load discard, private inherited continuation,4 exact synchronous failure states,2 sticky GPU code/PC failures,1 full diagnostic record passed\n";
    Check(deferred>=30,"Owned-state matrix failed to execute its fresh small deferred class");
    std::cout<<"48 queue-blocked owned-state cases, "<<deferred<<" deferred: all guarded raw/captured/assembled outputs,1824/512/272-byte states, "
        "counts1/2/3/6/8/9, both inherited banks/all index widths, formats/fixed/retained/unaligned/aliases, "
        "unequal first/base,257 cache evictions, source/control mutation and exact inherited GPU materialization passed\n";
}

void CheckOwnedConditionalPrefix(Metal::Device& device) {
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=3;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    program.code[0]=Metal::Test::Arithmetic(Op::MOV,0,16);
    // CMP's second source is only five bits. Put c0 in the seven-bit first
    // source and compare c0>v0, so negative inputs actually take the branch.
    program.code[1]=word(Op::CMP)|(4u<<24)|(4u<<21)|(32u<<12);
    program.code[2]=word(Op::IFC)|(6u<<10)|1u|(2u<<22)|(1u<<25);
    program.code[3]=Metal::Test::Arithmetic(Op::MOV,16,0);
    program.code[4]=Metal::Test::Arithmetic(Op::MOV,1,0);
    program.code[5]=word(Op::NOP);program.code[6]=word(Op::NOP);
    program.code[7]=word(Op::IFU)|(9u<<10)|1u;
    program.code[8]=Metal::Test::Arithmetic(Op::MOV,16,1);
    program.code[9]=word(Op::NOP);program.code[10]=word(Op::END);
    Metal::Scheduler queue{device};Metal::ShaderMachineRuntime machine{device,queue};
    Metal::VertexFetchRuntime fetch{device,queue};Metal::TransferRuntime transfer{device,queue};
    Metal::PicaVertexLayout layout{};layout.mapped_attribute_count=16;
    for(unsigned reg=0;reg<16;++reg) {
        layout.register_map[reg]=reg;layout.attribute_inputs[reg].mode=Metal::PicaAttributeMode::Buffer;
        layout.attribute_inputs[reg].fetch={0,reg*16,256,4,Metal::VertexAttributeFormat::Float,reg};
    }
    Metal::GeometryEmissionConfig config{};config.attribute_count=2;config.honor_winding=0;
    config.semantics={0x03020100,0x0b0a0908};
    unsigned cases=0;
    for(unsigned pattern=0;pattern<6;++pattern)for(unsigned count:{1u,2u,3u,6u,8u})
    for(unsigned boolean:{0u,1u,0u}) {
        auto bytes=device.CreateBuffer(8*sizeof(Metal::VertexRegisters),true);
        auto* vertices=static_cast<Metal::VertexRegisters*>(bytes.contents);
        for(unsigned v=0;v<8;++v) {
            const bool written=pattern==0 ? v%2==0:pattern==1 ? v%2==1:
                pattern==2 ? v==0:pattern==3 ? v==count-1:pattern==4;
            for(unsigned reg=0;reg<16;++reg)
                vertices[v][reg]={written ? -float(v+1):float(v+1),float(v)*.125f,float(reg)*.0625f,1};
        }
        const Metal::BufferSlice input{bytes,0,bytes.length,bytes.contents,{}};
        const std::array<Metal::BufferSlice,1> sources{input};
        Metal::ShaderMachineUniforms uniforms{};
        uniforms.b=boolean;
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
            Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,
            "Conditional prefix fixture lost dependency proof");
        Check(Metal::ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms)!=0,
            "Conditional prefix fixture lost termination proof");
        Metal::ShaderMachineState reference{device},state{device};
        Metal::VertexInputState reference_input{device},owned_input{device};
        const auto expected=machine.RunFetchedVertexBatchAndCapture(fetch,reference_input,sources,layout,
            reference,program,uniforms,config,count,count,0,0);
        const auto actual=Metal::InputReferenceTestAccess::Owned(machine,fetch,owned_input,sources,layout,
            state,program,uniforms,config,count,count,0,0);
        Check(state.HasDeferredVertexBatch(),"Conditional prefix fixture missed owned bounded path");
        const std::array<Metal::BufferSlice,4> slices{expected.output,actual.output,expected.captured,actual.captured};
        std::array<id<MTLBuffer>,4> reads{};
        for(unsigned i=0;i<4;++i) {
            reads[i]=device.CreateBuffer(slices[i].size,true);
            transfer.CopyBuffer(slices[i].buffer,slices[i].offset,reads[i],0,slices[i].size);
        }
        queue.Finish();
        float previous=0,previous_output=0;
        const auto* reference_raw=static_cast<const float*>(reads[0].contents);
        for(unsigned v=0;v<count;++v) {
            Check(reference_raw[v*64]==previous,"Conditional fixture failed its literal previous-writer oracle");
            if(vertices[v][0][0]<0)previous_output=vertices[v][0][0];
            Check(reference_raw[v*64+4]==previous_output,"Conditional fixture failed its literal inclusive-output oracle");
            if(boolean || vertices[v][0][0]<0)previous=vertices[v][0][0];
        }
        Check(!std::memcmp(reads[0].contents,reads[1].contents,reads[0].length),
            "Conditional owned prefix changed complete raw output");
        Check(!std::memcmp(reads[2].contents,reads[3].contents,reads[2].length) &&
            machine.SaveState(state)==machine.SaveState(reference) &&
            fetch.SaveState(owned_input).history==fetch.SaveState(reference_input).history,
            "Conditional owned prefix changed capture or complete shader/input state");
        ++cases;
    }
    std::cout<<cases<<" conditional owned prefixes: alternating/sparse/late/all/no writers,counts1/2/3/6/8, mixed false/true/false write-proof caches, full raw/capture/1824/512-byte state exact\n";
}

void CheckSmallRecordedBatches(Metal::Device& device,int argc,char** argv) {
    Check(argc>2,"Small native batch benchmark requires recorded shader files");
    Metal::Scheduler scheduler{device};Metal::ShaderMachineRuntime machine{device,scheduler};
    Metal::TransferRuntime transfer{device,scheduler};
    const auto input=device.CreateBuffer(16*sizeof(Metal::VertexRegisters),true);
    auto* vertices=static_cast<Metal::VertexRegisters*>(input.contents);
    for(unsigned v=0;v<16;++v) for(unsigned reg=0;reg<16;++reg)
        vertices[v][reg]={float(v+1)*.0625f,float(reg+1)*.03125f,-.25f,1};
    const Metal::BufferSlice inputs{input,0,input.length,nullptr,{}};
    unsigned cases=0,checked_vertices=0;
    for(int argument=2;argument<argc;++argument) {
        std::ifstream file(argv[argument],std::ios::binary);std::array<std::uint32_t,12> header{};
        Metal::ShaderMachineProgram program{};Metal::ShaderMachineUniforms uniforms{};
        file.read(reinterpret_cast<char*>(header.data()),sizeof(header));
        Check(file && header[0]==0x34544c4d && header[1]==1 && header[2]==4096 &&
            header[3]==4096 && header[4]==sizeof(uniforms),"Invalid small recorded shader header");
        program.entry=header[5];program.output_mask=header[6];program.allow_emission=header[7];
        file.read(reinterpret_cast<char*>(program.code.data()),sizeof(program.code));
        file.read(reinterpret_cast<char*>(program.swizzle.data()),sizeof(program.swizzle));
        file.read(reinterpret_cast<char*>(&uniforms),sizeof(uniforms));
        Check(file && file.peek()==std::char_traits<char>::eof(),"Truncated small recorded shader");
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
            Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,
            "Recorded small batch no longer exercises proven dependency passes");
        for(unsigned count:{1u,3u,6u,8u,9u,16u}) for(unsigned reset:{0u,1u}) for(unsigned bank:{0u,1u}) {
            Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
            for(unsigned word=0;word<448;++word) {
                const std::uint32_t value=0x3e000000u+word*257u;
                std::memcpy(initial.history.data()+word*4,&value,4);
            }
            const std::array<std::int32_t,4> address{3,-2,1,0};
            const std::array<std::uint32_t,4> flags{1,0,bank,9};
            std::memcpy(initial.history.data()+1792,address.data(),sizeof(address));
            std::memcpy(initial.history.data()+1808,flags.data(),sizeof(flags));
            if(reset) initial.history.fill(std::byte{});
            Metal::ShaderMachineState scalar{device},batched{device};
            machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
            std::vector<Metal::VertexRegisters> expected(count);
            for(unsigned v=0;v<count;++v) {
                const auto result=machine.Run(scalar,program,uniforms,
                    {input,v*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
                std::memcpy(&expected[v],result.output.data,sizeof(Metal::VertexRegisters));
            }
            const auto result=machine.RunVertexBatch(batched,program,uniforms,inputs,16,count,5,5);
            const auto readback=device.CreateBuffer(result.size,true);
            transfer.CopyBuffer(result.buffer,result.offset,readback,0,result.size);scheduler.Finish();
            Check(!std::memcmp(readback.contents,expected.data(),result.size) &&
                machine.SaveState(batched)==machine.SaveState(scalar),
                "Small recorded batch changed full output or1824-byte retained native state");
            if(reset && count<=8) {
                Metal::VertexFetchRuntime fetch{device,scheduler};Metal::VertexInputState fetched{device};
                Metal::ShaderMachineState owned{device};Metal::PicaVertexLayout layout{};layout.mapped_attribute_count=16;
                for(unsigned reg=0;reg<16;++reg) {
                    layout.register_map[reg]=reg;layout.attribute_inputs[reg].mode=Metal::PicaAttributeMode::Buffer;
                    layout.attribute_inputs[reg].fetch={0,reg*16,256,4,Metal::VertexAttributeFormat::Float,reg};
                }
                Metal::GeometryEmissionConfig config{};config.attribute_count=std::min(7,std::popcount(program.output_mask));config.honor_winding=0;
                config.semantics={0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x1f171614,0x1f1f1f1f};
                const auto expected_capture=machine.CaptureVertices(result,program.output_mask,config,count);
                const std::array<Metal::BufferSlice,1> sources{inputs};
                const auto actual=Metal::InputReferenceTestAccess::Owned(machine,fetch,fetched,sources,layout,owned,program,uniforms,config,16,count,5,5);
                Check(owned.HasDeferredVertexBatch(),"Recorded hot shader did not execute owned deferred-state pipeline");
                auto raw_read=device.CreateBuffer(actual.output.size,true),capture_read=device.CreateBuffer(actual.captured.size,true),capture_expected=device.CreateBuffer(expected_capture.size,true);
                transfer.CopyBuffer(actual.output.buffer,actual.output.offset,raw_read,0,raw_read.length);
                transfer.CopyBuffer(actual.captured.buffer,actual.captured.offset,capture_read,0,capture_read.length);
                transfer.CopyBuffer(expected_capture.buffer,expected_capture.offset,capture_expected,0,capture_expected.length);scheduler.Finish();
                Check(!std::memcmp(raw_read.contents,expected.data(),raw_read.length) &&
                    !std::memcmp(capture_read.contents,capture_expected.contents,capture_read.length) &&
                    machine.SaveState(owned)==machine.SaveState(scalar),"Recorded owned shader changed full outputs/capture/1824-byte state");
                std::cout<<"OWNED_RECORDED count="<<count<<" bank="<<bank<<" bound="<<
                    Metal::ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms).value_or(0)<<
                    " full outputs/state exact\n";
            }
            ++cases;checked_vertices+=count;
        }
        Metal::ShaderMachineState batched{device};
        for(unsigned count:{6u,8u,9u,16u}) {
            // Compile and warm outside the measured block. Resets reproduce the
            // captured draw class; every invocation still executes on the GPU.
            batched.Reset();machine.RunVertexBatch(batched,program,uniforms,inputs,16,count,5,5);
            scheduler.EndProfileFrame();scheduler.Finish();
            for(unsigned round=0;round<3;++round) {
                const auto start=std::chrono::steady_clock::now();
                for(unsigned repeat=0;repeat<64;++repeat) {
                    batched.Reset();machine.RunVertexBatch(batched,program,uniforms,inputs,16,count,5,5,{},0,1000000,true);
                }
                scheduler.EndProfileFrame();scheduler.Finish();
                const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-start).count();
                std::cout<<"SMALL_BENCH program="<<argument-2<<" count="<<count<<" round="<<round<<
                    " draws=64 host_ns="<<elapsed<<std::endl;
            }
        }
    }
    std::cout<<"Small recorded native batches: "<<cases<<" cases/"<<checked_vertices<<
        " vertices, full output bytes and all1824 state bytes exact; counts1/3/6/8/9/16, reset/inherited, both output banks\n";
}
void CheckTriggeredWarmProof(Metal::Device& device, bool host_only = false) {
    struct Environment {
        std::array<const char*,4> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_TRIGGER","AZAHAR_METAL4_PROFILE_FRAME_LIMIT","AZAHAR_METAL4_PROFILE_HOST_ONLY"};
        std::array<std::optional<std::string>,4> old;
        Environment(){for(unsigned i=0;i<keys.size();++i){if(const auto* v=std::getenv(keys[i]))old[i]=v;unsetenv(keys[i]);}}
        ~Environment(){for(unsigned i=0;i<keys.size();++i){if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}}
    } environment;
    char directory[]="/private/tmp/azahar-warm-profile-proof-XXXXXX";
    Check(mkdtemp(directory),"Unable to create warm bounded proof directory");
    const auto prefix=std::filesystem::path(directory)/"gpu.csv",trigger=std::filesystem::path(directory)/"start";
    setenv("AZAHAR_METAL4_PROFILE",prefix.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_TRIGGER",trigger.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","1",1);
    if(host_only)setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    {
        Metal::Scheduler scheduler{device};Metal::ShaderMachineRuntime trial{device,scheduler},reference{device,scheduler};
        Metal::TransferRuntime transfer{device,scheduler};Metal::ShaderMachineState batched{device},scalar{device};
        using Op=nihstro::OpCode::Id;
        Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=1;
        program.swizzle[0]=Metal::Test::IdentitySwizzle;
        program.code[0]=Metal::Test::Arithmetic(Op::ADD,16,16,0);
        program.code[1]=Metal::Test::Arithmetic(Op::MOV,0,16);program.code[2]=u32(Op::END)<<26;
        const auto input=device.CreateBuffer(3*sizeof(Metal::VertexRegisters),true),indices=device.CreateBuffer(4,true);
        std::memset(input.contents,0,input.length);
        for(unsigned v=0;v<3;++v)static_cast<Metal::VertexRegisters*>(input.contents)[v][0].fill(float(v+1));
        const std::array<unsigned char,4> order{0,1,0,2};std::memcpy(indices.contents,order.data(),order.size());
        const auto dumps=[&] {std::vector<std::filesystem::path> files;
            for(const auto& e:std::filesystem::directory_iterator(directory))if(e.path().extension()==".bin")files.push_back(e.path());return files;};
        std::vector<std::byte> recorded;
        for(unsigned frame=0;frame<4;++frame) {
            Check(scheduler.ProfilingEnabled()==(frame==2),"Warm shader proof capture activated outside its single requested frame");
            Metal::ShaderMachineUniforms uniforms{};
            std::array<Metal::VertexRegisters,3> expected{};
            for(unsigned v=0;v<3;++v) {
                const auto result=reference.Run(scalar,program,uniforms,{input,v*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
                std::memcpy(&expected[v],result.output.data,sizeof(Metal::VertexRegisters));
            }
            const auto output=trial.RunVertexBatch(batched,program,uniforms,{input,0,input.length,nullptr,{}},3,4,0,0,
                {indices,0,indices.length,nullptr,{}},1);
            const auto readback=device.CreateBuffer(output.size,true);
            transfer.CopyBuffer(output.buffer,output.offset,readback,0,output.size);scheduler.Finish();
            for(unsigned e=0;e<order.size();++e)Check(!std::memcmp(static_cast<const std::byte*>(readback.contents)+e*sizeof(Metal::VertexRegisters),
                &expected[order[e]],sizeof(Metal::VertexRegisters)),"Bounded diagnostics changed full indexed vertex output");
            Check(trial.SaveState(batched)==reference.SaveState(scalar),"Bounded diagnostics changed complete1824-byte retained state");
            const auto files=dumps();
            if(frame<2 || host_only)Check(files.empty(),"Inactive or host-only profile wrote shader payloads");
            else {
                Check(files.size()==1,"Warm pre-trigger proof was omitted or dumped more than once");
                std::ifstream file(files[0],std::ios::binary);const std::vector<char> bytes{std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
                const auto* begin=reinterpret_cast<const std::byte*>(bytes.data());std::vector<std::byte> value(begin,begin+bytes.size());
                Check(value.size()==48+sizeof(program.code)+sizeof(program.swizzle)+sizeof(uniforms) &&
                    !std::memcmp(value.data()+48,program.code.data(),sizeof(program.code)) &&
                    !std::memcmp(value.data()+48+sizeof(program.code),program.swizzle.data(),sizeof(program.swizzle)) &&
                    !std::memcmp(value.data()+48+sizeof(program.code)+sizeof(program.swizzle),&uniforms,sizeof(uniforms)),
                    "Triggered warm proof lost full instructions/swizzles/active uniforms");
                if(frame==2)recorded=std::move(value);else Check(value==recorded,"Disabled profiling rewrote the retained warm shader dump");
            }
            if(frame==1){std::ofstream signal(trigger);signal<<"start";}
            scheduler.EndProfileFrame();
        }
    }
    std::filesystem::remove_all(directory);
    if(host_only)std::cout<<"Host-only warm shader proof: all four frames emit no shader dumps; all16 indexed outputs and complete1824-byte states match scalar GPU\n";
    else std::cout<<"Triggered warm shader proof: two pre-trigger native batches emit no dumps; one active frame publishes full32768 instruction/swizzle bytes and1616 uniforms from an already cached proof; post-limit batch and all16 indexed outputs/full1824-byte states remain exact\n";
}

void CheckFirstNativeProofReuse(Metal::Device& device) {
    using Runtime=Metal::ShaderMachineRuntime;using Execution=Runtime::VertexExecution;
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    const auto arithmetic=[](Op op,unsigned dest,unsigned a,unsigned b=0) {
        return Metal::Test::Arithmetic(op,dest,a,b);
    };
    std::array<Metal::ShaderMachineProgram,5> programs{};
    for(auto& p:programs) {p.allow_emission=false;p.output_mask=7;p.swizzle[0]=Metal::Test::IdentitySwizzle;}
    programs[0].code[0]=arithmetic(Op::MOV,0,16);
    programs[0].code[1]=arithmetic(Op::ADD,16,16,0);programs[0].code[2]=word(Op::END);
    for(unsigned i=0;i<3;++i)programs[1].code[i]=arithmetic(Op::MOV,i,i);
    programs[1].code[3]=word(Op::END);
    programs[2].code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
    programs[2].code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
    programs[2].code[2]=arithmetic(Op::MOV,0,0);programs[2].code[3]=word(Op::NOP);
    programs[2].code[4]=word(Op::NOP);programs[2].code[5]=word(Op::END);
    programs[3].code[0]=arithmetic(Op::MOV,0,16);
    programs[3].code[1]=arithmetic(Op::MOV,16,0);programs[3].code[2]=word(Op::END);
    programs[4].code[0]=arithmetic(Op::MOV,0,0);
    programs[4].code[1]=arithmetic(Op::ADD,16,16,0);programs[4].code[2]=word(Op::END);
    auto hidden=programs[1];hidden.output_mask=1;
    hidden.code[3]=arithmetic(Op::MUL,16,16,16);hidden.code[4]=word(Op::END);
    Check(Runtime::AnalyzeVertices(hidden,{})==Execution::Ordered,
        "Native proof fixture must retain ordered hidden raw state");
    std::vector<Metal::ShaderMachineProgram> fixtures(programs.begin(),programs.end());fixtures.push_back(hidden);
    auto branch=programs[1];branch.code.fill(word(Op::NOP));
    branch.code[0]=0x27u<<26|3u<<10|1u;branch.code[1]=arithmetic(Op::MOV,0,0);
    branch.code[2]=arithmetic(Op::MOV,16,0);branch.code[3]=arithmetic(Op::MOV,0,16);
    branch.code[4]=word(Op::END);fixtures.push_back(branch);
    Metal::Scheduler queue{device};Metal::ShaderMachineRuntime machine{device,queue};
    Metal::GeometryEmissionConfig config{};config.honor_winding=0;config.attribute_count=3;
    config.semantics={0x03020100,0x07060504,0x0b0a0908};
    unsigned checked=0,independent_ordered=0;
    for(const auto& program:fixtures) {
        Check(Runtime::HasNativeResetControlFlow(program),"Native proof fixture lacks supported control flow");
        std::vector<Metal::ShaderMachineUniforms> controls(1);
        for(unsigned bit=0;bit<16;++bit){auto u=controls[0];u.b=1u<<bit;controls.push_back(u);}
        for(unsigned lane=0;lane<16;++lane){auto u=controls[0];u.i[lane/4][lane%4]=lane%4==0 ? 6:0x80000100u;controls.push_back(u);}
        for(const auto& uniforms:controls)for(unsigned variant=0;variant<3;++variant) {
            auto capture=config;
            if(variant==1)capture.attribute_count=1;
            if(variant==2)capture.semantics.fill(0x1f1f1f1fu);
            const bool expected=Runtime::AnalyzeVertices(program,uniforms,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&capture)==Execution::Independent;
            Check(machine.CanCaptureNativeReset(program,uniforms,capture)==expected,
                "First native proof changed capture verdict across full control key/semantics");
            Check(Metal::InputReferenceTestAccess::CachedRawMatches(machine,program,uniforms,!expected),
                "Negative native capture proof did not retain the exact complete raw/control proof");
            if(expected && Runtime::AnalyzeVertices(program,uniforms)==Execution::Ordered)++independent_ordered;
            ++checked;
        }
    }
    Check(independent_ordered>0,"Native capture success incorrectly implies independent raw state");
    // Saturate both bounded caches using a fresh program. Different capture
    // semantics must still be evaluated after its raw proof already exists.
    branch.code[4095]=0x12345678;
    const auto run=[&](unsigned value) {
        Metal::ShaderMachineUniforms u{};u.b=value&1;u.i[0][0]=value;
        const bool expected=Runtime::AnalyzeVertices(branch,u,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&config)==Execution::Independent;
        Check(machine.CanCaptureNativeReset(branch,u,config)==expected &&
              Metal::InputReferenceTestAccess::CachedRawMatches(machine,branch,u),
              "Native/raw bounded proof caches changed control-dependent decisions");
    };
    for(unsigned value=0;value<128;++value)run(value);
    Check(Metal::InputReferenceTestAccess::ProofCounts(machine,branch)==std::pair{128u,128u},
        "Native/raw proof caches failed to saturate at128 entries");
    for(unsigned value=0;value<128;++value)run(value);
    run(128);
    Check(Metal::InputReferenceTestAccess::ProofCounts(machine,branch)==std::pair{1u,1u},
        "Native/raw proof caches lost bounded replacement at a real miss");
    run(0);Metal::InputReferenceTestAccess::Evict(machine,branch);run(0);
    auto positive=programs[1];positive.code[4095]=0x87654321;
    const auto native_only=[&](unsigned control) {
        Metal::ShaderMachineUniforms u{};u.i[0][0]=control;
        Check(machine.CanCaptureNativeReset(positive,u,config),"Independent native fixture lost selection");
    };
    for(unsigned value=0;value<128;++value)native_only(value);
    Check(Metal::InputReferenceTestAccess::ProofCounts(machine,positive)==std::pair{0u,128u},
        "Positive native selections must not populate or evict the raw proof cache");
    for(unsigned value=0;value<128;++value)native_only(value);
    native_only(128);
    Check(Metal::InputReferenceTestAccess::ProofCounts(machine,positive)==std::pair{0u,1u},
        "Positive native proof replacement changed raw cache occupancy");
    // Positive controls of the SAME program must not evict its saturated
    // negative raw proofs, even if the separate native cache rolls over.
    auto mixed=programs[1];mixed.code.fill(word(Op::NOP));
    mixed.code[0]=0x27u<<26|3u<<10;
    for(unsigned i=0;i<3;++i)mixed.code[i+1]=arithmetic(Op::MOV,i,i);
    mixed.code[4]=word(Op::END);mixed.code[4095]=0x12121212;
    for(unsigned control=0;control<128;++control) {
        Metal::ShaderMachineUniforms u{};u.i[0][0]=control;
        Check(!machine.CanCaptureNativeReset(mixed,u,config),"Negative mixed-control fixture selected native");
    }
    for(unsigned control=0;control<129;++control) {
        Metal::ShaderMachineUniforms u{};u.b=1;u.i[0][0]=control;
        Check(machine.CanCaptureNativeReset(mixed,u,config),"Positive mixed-control fixture lost native selection");
    }
    Check(Metal::InputReferenceTestAccess::ProofCounts(machine,mixed).first==128,
        "Positive native controls evicted the same program's saturated negative raw proofs");
    for(unsigned control=0;control<128;++control) {
        Metal::ShaderMachineUniforms u{};u.i[0][0]=control;
        Check(Metal::InputReferenceTestAccess::CachedRawMatches(machine,mixed,u),
            "Positive native pressure lost an existing complete negative raw proof");
    }
    queue.Finish();
    std::cout<<checked<<" first native proof queries: full raw facts/bool+16integer controls/semantic changes, independent capture with ordered raw, saturated hits, eviction/reselection, positive native leaves raw cache empty and preserves saturated same-program negative proofs passed\n";
}

void CheckControlProofCache(Metal::Device& device, Metal::Scheduler& scheduler) {
    // The existing opt-in analysis dumps distinguish a retained proof from a
    // repeated analysis without adding instrumentation to the production API.
    struct ProfileScope {
        bool had_value{};
        std::string previous;
        std::filesystem::path directory;
        ProfileScope() {
            if(const auto* value=std::getenv("AZAHAR_METAL4_PROFILE")) {
                had_value=true;previous=value;
            }
            char path[]="/private/tmp/azahar-metal4-proof-cache-XXXXXX";
            const auto* made=mkdtemp(path);
            Check(made,"Unable to create control-proof test directory");directory=made;
            const auto prefix=(directory/"proof").string();
            Check(setenv("AZAHAR_METAL4_PROFILE",prefix.c_str(),1)==0,"Unable to enable analysis evidence");
        }
        ~ProfileScope() {
            if(had_value) setenv("AZAHAR_METAL4_PROFILE",previous.c_str(),1);
            else unsetenv("AZAHAR_METAL4_PROFILE");
            std::error_code error;std::filesystem::remove_all(directory,error);
        }
    } profile;
    Metal::ShaderMachineRuntime trial{device,scheduler},reference{device,scheduler};
    Metal::ShaderMachineState batched{device},scalar{device};
    Metal::ShaderMachineProgram program{};program.output_mask=7;program.allow_emission=false;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    program.code[0]=(0x27u<<26)|(3u<<10); // Uniform-b0 conditional with two MOVs.
    program.code[1]=(0x13u<<26);
    program.code[2]=(0x13u<<26)|(1u<<21)|(1u<<12);
    program.code[3]=(0x13u<<26)|(2u<<21)|(2u<<12);
    program.code[4]=0x22u<<26;
    auto inputs=device.CreateBuffer(3*sizeof(Metal::VertexRegisters),true);
    auto* values=static_cast<Metal::VertexRegisters*>(inputs.contents);
    constexpr std::array<unsigned char,5> order{0,1,0,2,1};
    auto index_buffer=device.CreateBuffer(order.size(),true);
    std::memcpy(index_buffer.contents,order.data(),order.size());
    const Metal::BufferSlice input{inputs,0,inputs.length,inputs.contents,{}},
        indices{index_buffer,0,index_buffer.length,index_buffer.contents,{}};
    unsigned batches{};
    const auto run=[&](unsigned control) {
        for(unsigned vertex=0;vertex<3;++vertex)for(unsigned reg=0;reg<16;++reg)
            values[vertex][reg]={float(batches+vertex+1)*.125f,float(reg)*.25f,-.5f,.75f};
        Metal::ShaderMachineUniforms uniforms{};
        uniforms.b=control&1u;uniforms.i[0][0]=control;
        std::array<Metal::VertexRegisters,3> misses{};
        for(unsigned vertex=0;vertex<3;++vertex) {
            const auto result=reference.Run(scalar,program,uniforms,
                {inputs,vertex*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),values+vertex,{}});
            std::memcpy(&misses[vertex],result.output.data,sizeof(Metal::VertexRegisters));
        }
        // Exercise the production selection-before-fallback sequence: the
        // native query may now populate the complete raw proof used below.
        Metal::GeometryEmissionConfig capture{};capture.attribute_count=3;capture.honor_winding=0;
        capture.semantics={0x03020100,0x07060504,0x0b0a0908};
        (void)trial.CanCaptureNativeReset(program,uniforms,capture);
        const auto output=trial.RunVertexBatch(batched,program,uniforms,input,3,order.size(),0,0,indices,1);
        auto readback=device.CreateBuffer(output.size,true);
        Metal::TransferRuntime transfer{device,scheduler};
        transfer.CopyBuffer(output.buffer,output.offset,readback,0,output.size);scheduler.Finish();
        for(unsigned element=0;element<order.size();++element)
            Check(!std::memcmp(static_cast<const std::byte*>(readback.contents)+element*sizeof(Metal::VertexRegisters),
                &misses[order[element]],sizeof(Metal::VertexRegisters)),"Control-proof cache changed complete vertex outputs");
        const auto expected=reference.SaveState(scalar),actual=trial.SaveState(batched);
        Check(expected.needs_reset==actual.needs_reset && expected.history==actual.history,
            "Control-proof cache changed complete retained shader state");
        ++batches;
    };
    for(unsigned control=0;control<128;++control) run(control);
    std::vector<std::filesystem::path> proofs;
    constexpr std::string_view marker="retained-control-proof";
    for(const auto& entry:std::filesystem::directory_iterator(profile.directory)) {
        if(entry.path().extension()!=".bin") continue;
        proofs.push_back(entry.path());
        std::ofstream file(entry.path(),std::ios::binary|std::ios::trunc);file<<marker;
        Check(bool(file),"Unable to mark existing proof evidence");
    }
    Check(proofs.size()==128,"Control fixture did not create 128 distinct analyses");
    for(unsigned control=0;control<128;++control) run(control);
    const auto retained=[&](const auto& path) {
        std::ifstream file(path,std::ios::binary);
        const std::string bytes{std::istreambuf_iterator<char>{file},std::istreambuf_iterator<char>{}};
        return bytes==marker;
    };
    for(const auto& path:proofs) Check(retained(path),"A full control-proof cache reanalyzed an existing key");
    run(128); // A real miss still triggers the bounded-cache replacement policy.
    run(0);   // Evicted controls must be analyzed again, with exact output/state.
    unsigned replaced{};
    for(const auto& path:proofs) replaced+=!retained(path);
    Check(replaced==1,"A real control miss did not preserve bounded-cache replacement behavior");
    std::cout<<batches<<" control-proof batches preserve full outputs/state: native-selection-first, 128 saturated-cache hits avoid reanalysis, real misses still replace bounded proofs\n";
}
void CheckPrefixBoundaries(Metal::Device& device, Metal::Scheduler& scheduler) {
    // An independent scalar integer oracle selects the last actual FIFO miss
    // that wrote each component. Include raw special float bits without doing
    // floating-point arithmetic, and verify cache-hit rows remain untouched.
    const auto library = device.CompileLibrary(Metal::ShaderMachineSource, "Native prefix boundary regression");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "pica_vertex_machine_prefix");
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    const auto pipeline = device.CompileCompute(descriptor);
    const unsigned simd_width = unsigned(pipeline.threadExecutionWidth);
    Check(simd_width >= 16 && simd_width <= 64 && (simd_width & (simd_width - 1)) == 0,
          "Unsupported native prefix SIMD width");
    unsigned cases = 0;
    for (unsigned count : {1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,
                           255u,256u,257u,511u,512u,513u,1023u,1024u,1025u,4095u,4096u,4097u})
    for (unsigned pattern = 0; pattern < 8; ++pattern) {
        const bool indexed = pattern & 1u, reset = pattern & 4u;
        constexpr unsigned guard = 16, canary = 0xcdcdcdcdu;
        auto stored = device.CreateBuffer(1824, true);
        auto status = device.CreateBuffer(16, true);
        auto params = device.CreateBuffer(120, true);
        auto carry = device.CreateBuffer(guard + NSUInteger(count) * 432 + guard, true);
        auto seeds = device.CreateBuffer(guard + NSUInteger(count) * 288 + guard, true);
        auto plan = device.CreateBuffer(NSUInteger(count) * 8, true);
        auto* initial = static_cast<unsigned*>(stored.contents);
        auto* flags = static_cast<unsigned*>(status.contents);
        auto* config = static_cast<unsigned*>(params.contents);
        auto* values = reinterpret_cast<unsigned*>(static_cast<std::byte*>(carry.contents) + guard);
        auto* entries = static_cast<unsigned*>(plan.contents);
        std::fill_n(initial, 456, 0x80000000u);
        for (unsigned lane = 0; lane < 72; ++lane)
            initial[lane < 64 ? 64 + lane : 448 + lane - 64] = 0x7fc00000u + lane;
        std::fill_n(flags, 4, 0u);
        std::fill_n(config, 30, 0u);
        std::memset(carry.contents, 0xcd, carry.length);
        std::memset(seeds.contents, 0xcd, seeds.length);
        config[3] = reset;
        config[9] = count;
        config[12] = indexed ? 2 : 0;
        std::array<unsigned, 72> modes{}, current{};
        for (unsigned lane = 0; lane < 72; ++lane) {
            const bool immutable = lane == 67 || lane >= 70;
            modes[lane] = immutable ? 0u : (lane + pattern) % 3u;
            const unsigned retained = lane < 67 ? lane + 64 : lane - 68 + 131;
            if (modes[lane]) config[25 + retained / 32] |= 1u << (retained % 32);
            if (modes[lane] == 1) config[20 + retained / 32] |= 1u << (retained % 32);
            current[lane] = reset ? 0u : initial[lane < 64 ? 64 + lane : 448 + lane - 64];
        }
        std::vector<unsigned> expected(NSUInteger(count) * 72, canary);
        std::array<std::pair<unsigned, unsigned>, 64> fifo{};
        unsigned size = 0, replacement = 0, previous = ~0u;
        for (unsigned element = 0; element < count; ++element) {
            const unsigned id = element < 75 ? element :
                element % 3 == 0 ? (element - 1) % 90 : (element * 7) % 90;
            unsigned cached = ~0u;
            if (indexed) for (unsigned slot = 0; slot < size; ++slot)
                if (fifo[slot].first == id) cached = fifo[slot].second;
            entries[element * 2] = cached;
            entries[element * 2 + 1] = previous;
            std::fill_n(values + NSUInteger(element) * 108 + 72, 34, 0u);
            for (unsigned lane = 0; lane < 72; ++lane) {
                const unsigned bits = (element % 5 == 0 ? 0x80000000u :
                    element % 5 == 1 ? 0x7fc12345u : 0x3e000000u) ^ (element * 4099u + lane * 67u);
                values[NSUInteger(element) * 108 + lane] = bits;
                const bool conditional = pattern % 4 == 0 ? false : pattern % 4 == 1 ? true :
                    pattern % 4 == 2 ? (element + lane) % 37 == 0 : element + 1 == count;
                const bool written = modes[lane] == 1 || (modes[lane] == 2 && conditional);
                if (written) {
                    const unsigned reg = lane < 64 ? 16 + lane / 4 : lane < 67 ? 32 : 33;
                    const unsigned bit = lane < 64 ? 8u >> (lane % 4) : 1u << (lane < 67 ? lane - 64 : lane - 68);
                    values[NSUInteger(element) * 108 + 72 + reg] |= bit;
                }
                if (cached == ~0u) {
                    if (written) current[lane] = bits;
                    expected[NSUInteger(element) * 72 + lane] = current[lane];
                }
            }
            if (cached == ~0u) {
                previous = element;
                if (indexed) {
                    fifo[replacement] = {id, element};
                    replacement = (replacement + 1) & 63u;
                    size = std::min(size + 1u, 64u);
                }
            }
        }
        const std::array bindings{
            Metal::BufferBinding{stored, 0, 4}, Metal::BufferBinding{status, 0, 7},
            Metal::BufferBinding{params, 0, 8}, Metal::BufferBinding{carry, guard, 10},
            Metal::BufferBinding{seeds, guard, 11}, Metal::BufferBinding{plan, 0, 12}};
        const auto arguments = scheduler.MakeArguments(bindings);
        scheduler.KeepAlive(pipeline);
        const auto dispatch = [&] {
            const auto encoder = scheduler.BeginCompute(pipeline);
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:arguments];
            [encoder dispatchThreads:MTLSizeMake(((count + 63u) & ~63u) + 72u * simd_width, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            scheduler.EndEncoding();
            scheduler.Finish();
        };
        dispatch();
        const auto* actual = static_cast<const std::byte*>(seeds.contents) + guard;
        Check(!std::memcmp(actual, expected.data(), expected.size() * 4),
              "Native prefix changed a conditional last writer, inherited raw bit or cache-hit row");
        const auto* words = static_cast<const unsigned*>(seeds.contents);
        Check(std::all_of(words, words + 4, [](unsigned value) { return value == canary; }) &&
              std::all_of(words + 4 + count * 72, words + 8 + count * 72,
                          [](unsigned value) { return value == canary; }),
              "Native prefix overwrote a guard outside seed storage");
        if (count == 33) {
            std::memset(seeds.contents, 0xcd, seeds.length);
            flags[0] = 6;
            dispatch();
            Check(std::all_of(words, words + seeds.length / 4,
                              [](unsigned value) { return value == canary; }),
                  "Native prefix wrote seed storage after an earlier GPU error");
        }
        ++cases;
    }
    std::cout << cases << " native prefix boundary cases preserve every seed bit, masked writes, sparse/late writers, reset/inheritance, FIFO hits/eviction, partial SIMD groups and error guards\n";
}

void CheckMergeBoundaries(Metal::Device& device, Metal::Scheduler& scheduler) {
    const auto library = device.CompileLibrary(Metal::ShaderMachineSource, "Native final-state boundary regression");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "pica_vertex_machine_merge");
    descriptor.maxTotalThreadsPerThreadgroup = 256;
    const auto pipeline = device.CompileCompute(descriptor);
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "pica_vertex_machine_resolve");
    descriptor.maxTotalThreadsPerThreadgroup = 64;
    const auto resolve_pipeline = device.CompileCompute(descriptor);
    unsigned cases = 0;
    for (unsigned count : {1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,
                           255u,256u,257u,511u,512u,513u,1023u,1024u,1025u,4095u,4096u,4097u})
    for (unsigned pattern = 0; pattern < 8; ++pattern) {
        constexpr unsigned guard = 16, canary = 0xcdcdcdcdu;
        const bool indexed = pattern & 1u, reset = pattern & 4u, aliased = pattern % 4 == 3;
        const unsigned vertices = aliased ? 6u : indexed ? 90u : count;
        auto stored = device.CreateBuffer(guard + 1824 + guard, true);
        auto input = aliased ? stored : device.CreateBuffer(guard + NSUInteger(vertices) * 256 + guard, true);
        auto output = device.CreateBuffer(guard + NSUInteger(count) * 256 + guard, true);
        auto carry = device.CreateBuffer(NSUInteger(count) * 432, true);
        auto plan = device.CreateBuffer(NSUInteger(count) * 8, true);
        auto indices = device.CreateBuffer(NSUInteger(count) * 2, true);
        auto params = device.CreateBuffer(120, true);
        auto status = device.CreateBuffer(16, true);
        auto events = device.CreateBuffer(16, true);
        std::memset(stored.contents, 0xcd, stored.length);
        std::memset(output.contents, 0xcd, output.length);
        std::memset(params.contents, 0, params.length);
        std::memset(status.contents, 0, status.length);
        std::memset(events.contents, 0, events.length);
        auto* state = reinterpret_cast<unsigned*>(static_cast<std::byte*>(stored.contents) + guard);
        auto* source = reinterpret_cast<unsigned*>(static_cast<std::byte*>(input.contents) + guard);
        auto* raw_output = reinterpret_cast<unsigned*>(static_cast<std::byte*>(output.contents) + guard);
        auto* values = static_cast<unsigned*>(carry.contents);
        auto* entries = static_cast<unsigned*>(plan.contents);
        auto* ids = static_cast<std::uint16_t*>(indices.contents);
        auto* config = static_cast<unsigned*>(params.contents);
        for (unsigned word = 0; word < 456; ++word) state[word] = 0x7fc00000u ^ (word * 919u);
        state[454] = (pattern >> 1) & 1u;
        if (!aliased) for (unsigned word = 0; word < vertices * 64; ++word)
            source[word] = 0x80000000u ^ (word * 1297u);
        const std::vector<unsigned> original_state(state, state + 456);
        const std::vector<unsigned> original_input(source, source + vertices * 64);
        std::vector<unsigned> expected_state = reset ? std::vector<unsigned>(456, 0u) : original_state;
        std::array<unsigned, 133> modes{};
        config[3] = reset; config[8] = vertices; config[9] = count;
        config[12] = indexed ? 2u : 0u; config[14] = count >= 128u; config[19] = aliased;
        for (unsigned lane = 0; lane < 133; ++lane) {
            modes[lane] = (lane + pattern) % 3u;
            if (modes[lane]) config[25 + lane / 32] |= 1u << (lane % 32);
            if (modes[lane] == 1) config[20 + lane / 32] |= 1u << (lane % 32);
        }
        std::array<std::pair<unsigned, unsigned>, 64> fifo{};
        unsigned size = 0, replacement = 0, previous = ~0u;
        for (unsigned element = 0; element < count; ++element) {
            const unsigned id = indexed ? (element < 75 ? element :
                element % 3 == 0 ? (element - 1) % 90 : (element * 7) % 90) % vertices : element;
            ids[element] = id;
            unsigned cached = ~0u;
            if (indexed) for (unsigned slot = 0; slot < size; ++slot)
                if (fifo[slot].first == id) cached = fifo[slot].second;
            entries[element * 2] = cached; entries[element * 2 + 1] = previous;
            if (cached == ~0u) {
                previous = element;
                if (indexed) {
                    fifo[replacement] = {id, element}; replacement = (replacement + 1) & 63u;
                    size = std::min(size + 1u, 64u);
                }
            }
        }
        const unsigned last_miss = previous;
        for (unsigned element = 0; element < count; ++element) {
            for (unsigned lane = 0; lane < 64; ++lane)
                raw_output[NSUInteger(element) * 64 + lane] = 0x7fc12345u ^ (element * 4099u + lane * 67u);
            std::fill_n(values + NSUInteger(element) * 108, 108, 0u);
            for (unsigned word = 0; word < 72; ++word)
                values[NSUInteger(element) * 108 + word] = 0x80000000u ^ (element * 8191u + word * 37u);
            for (unsigned lane = 0; lane < 133; ++lane) {
                const bool conditional = pattern % 4 == 0 ? false : pattern % 4 == 1 ? true :
                    pattern % 4 == 2 ? (element + lane) % 37 == 0 : element == last_miss;
                if (modes[lane] == 1 || (modes[lane] == 2 && conditional)) {
                    const unsigned reg = lane < 128 ? lane / 4 : lane < 131 ? 32 : 33;
                    const unsigned bit = lane < 128 ? 8u >> (lane % 4) : 1u << (lane < 131 ? lane - 128 : lane - 131);
                    values[NSUInteger(element) * 108 + 72 + reg] |= bit;
                }
            }
        }
        std::vector<unsigned> expected_output(raw_output, raw_output + NSUInteger(count) * 64);
        const unsigned bank = reset ? 0u : original_state[454];
        for (unsigned lane = 0; lane < 133; ++lane) {
            const unsigned offset = lane < 64 ? 128 + bank * 64 + lane : lane < 128 ? lane :
                lane < 131 ? 448 + lane - 128 : 452 + lane - 131;
            unsigned current = reset ? 0u : original_state[offset];
            for (unsigned element = 0; element < count; ++element) {
                const unsigned cached = entries[element * 2];
                if (cached != ~0u) {
                    if (lane < 64) expected_output[NSUInteger(element) * 64 + lane] = expected_output[NSUInteger(cached) * 64 + lane];
                    continue;
                }
                const unsigned reg = lane < 128 ? lane / 4 : lane < 131 ? 32 : 33;
                const unsigned bit = lane < 128 ? 8u >> (lane % 4) : 1u << (lane < 131 ? lane - 128 : lane - 131);
                if (values[NSUInteger(element) * 108 + 72 + reg] & bit)
                    current = lane < 64 ? expected_output[NSUInteger(element) * 64 + lane] :
                        values[NSUInteger(element) * 108 + (lane < 128 ? lane - 64 : lane < 131 ? 64 + lane - 128 : 68 + lane - 131)];
                if (lane < 64) expected_output[NSUInteger(element) * 64 + lane] = current;
            }
            expected_state[offset] = current;
            if (lane < 64) expected_state[lane] = original_input[NSUInteger(ids[last_miss]) * 64 + lane];
        }
        const auto arguments = scheduler.MakeArguments(std::array{
            Metal::BufferBinding{input, guard, 3}, Metal::BufferBinding{stored, guard, 4},
            Metal::BufferBinding{output, guard, 5}, Metal::BufferBinding{status, 0, 7},
            Metal::BufferBinding{params, 0, 8}, Metal::BufferBinding{indices, 0, 9},
            Metal::BufferBinding{carry, 0, 10}, Metal::BufferBinding{plan, 0, 12},
            Metal::BufferBinding{events, 0, 14}});
        const auto dispatch = [&] {
            scheduler.KeepAlive(pipeline); scheduler.KeepAlive(resolve_pipeline);
            const auto encoder = scheduler.BeginCompute(resolve_pipeline);
            if (config[14]) {
                [encoder setComputePipelineState:resolve_pipeline]; [encoder setArgumentTable:arguments];
                [encoder dispatchThreads:MTLSizeMake(resolve_pipeline.threadExecutionWidth * 133, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(resolve_pipeline.threadExecutionWidth, 1, 1)];
                [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            }
            [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:arguments];
            [encoder dispatchThreads:MTLSizeMake(133, 1, 1) threadsPerThreadgroup:MTLSizeMake(aliased ? 256 : 64, 1, 1)];
            scheduler.EndEncoding(); scheduler.Finish();
        };
        dispatch();
        Check(!std::memcmp(state, expected_state.data(), 1824), "Native merge changed complete retained state or aliased final input");
        Check(!std::memcmp(raw_output, expected_output.data(), expected_output.size() * 4), "Native merge changed raw outputs or FIFO-hit history");
        const auto guards = [&](id<MTLBuffer> buffer, unsigned body_words) {
            const auto* words = static_cast<const unsigned*>(buffer.contents);
            return std::all_of(words, words + 4, [](unsigned word) { return word == canary; }) &&
                std::all_of(words + 4 + body_words, words + 8 + body_words, [](unsigned word) { return word == canary; });
        };
        Check(guards(stored, 456) && guards(output, count * 64), "Native merge overwrote state/output guards");
        if (count == 33) {
            std::memcpy(state, original_state.data(), 1824);
            std::fill_n(raw_output, count * 64, canary);
            static_cast<unsigned*>(status.contents)[0] = 6;
            dispatch();
            Check(!std::memcmp(state, original_state.data(), 1824) &&
                  std::all_of(raw_output, raw_output + count * 64, [](unsigned word) { return word == canary; }),
                  "Native merge changed state/output after an earlier GPU error");
        }
        ++cases;
    }
    std::cout << cases << " native merge boundary cases preserve full state/output, last conditional writers, no writers, FIFO hits/eviction, both banks, reset/inheritance, aliased input, guards and GPU-error rollback\n";
}

void CheckScalarOutputRetention(Metal::Device& device, Metal::Scheduler& scheduler,
                                Metal::ShaderMachineRuntime& machine) {
    using Op=nihstro::OpCode::Id;
    Metal::ShaderMachineProgram program{};
    program.output_mask=1;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    unsigned pc=0;
    for(unsigned slot=0;slot<3;++slot) {
        program.code[pc++]=Metal::Test::Arithmetic(Op::MOV,0,32);
        program.code[pc++]=(unsigned(Op::SETEMIT)<<26)|(slot<<24)|(slot==2 ? 1u<<23:0);
        program.code[pc++]=unsigned(Op::EMIT)<<26;
    }
    program.code[pc]=unsigned(Op::END)<<26;
    Metal::ShaderMachineState state{device};
    Metal::ShaderMachineUniforms uniforms{};
    struct Retained { Metal::ShaderMachineResult result; Metal::VertexRegisters expected; };
    std::vector<Retained> held;
    const auto run=[&](unsigned value,unsigned capacity) {
        uniforms.f[0]={float(value),float(value+1),-float(value+2),0.5f};
        state.Reset();
        auto result=machine.Run(state,program,uniforms,{},0,capacity);
        Metal::VertexRegisters expected{};expected[0]=uniforms.f[0];
        Check(result.emission_count==3,"Scalar output pool lost emitted vertices");
        Check(result.output.size==sizeof(expected) &&
              !std::memcmp(result.output.data,&expected,sizeof(expected)),
              "Scalar output pool changed complete vertex registers");
        const auto* emitted=static_cast<const Metal::ShaderMachineEmission*>(result.emissions.data);
        for(unsigned i=0;i<3;++i)
            Check(!std::memcmp(&emitted[i].attributes,&expected,sizeof(expected)) && emitted[i].winding==0,
                  "Scalar output pool changed complete geometry attributes/winding");
        return Retained{std::move(result),expected};
    };
    const auto check_held=[&] {
        for(const auto& value:held) {
            Check(!std::memcmp(value.result.output.data,&value.expected,sizeof(value.expected)),
                  "Later scalar invocation overwrote retained vertex output");
            if(value.result.emissions.buffer) {
                const auto* emitted=static_cast<const Metal::ShaderMachineEmission*>(value.result.emissions.data);
                for(unsigned i=0;i<3;++i)
                    Check(!std::memcmp(&emitted[i].attributes,&value.expected,sizeof(value.expected)) && emitted[i].winding==0,
                          "Later scalar invocation overwrote retained geometry output");
            }
        }
    };
    // More than 32 MiB of live reservations; exceeding retention must preserve
    // every old result, including buffers held only through one sibling slice.
    for(unsigned i=0;i<700;++i)held.push_back(run(i,192));
    check_held();
    for(unsigned i=0;i<held.size();++i)if(i%2)held[i].result.emissions={};
    for(unsigned i=0;i<96;++i)(void)run(1000+i,i%2 ? 3:4096);
    check_held();
    held.clear();
    // Release all leases, churn the arena, then retain new values across more
    // completions. A shared buffer identity is allowed; aliasing ranges are not.
    for(unsigned i=0;i<700;++i) {
        auto result=run(2000+i,192);
        if(i%23==0)held.push_back(std::move(result));
    }
    check_held();
    std::cout<<"1496 scalar GPU invocations preserve complete output/emission attributes across >32MiB live reservations, sibling lease release and arena reuse\n";
    scheduler.Finish();
}

void CheckBorrowedProgramSnapshots(Metal::Device& device,Metal::Scheduler& scheduler,
                                   Metal::ShaderMachineRuntime& machine) {
    Metal::ShaderMachineState state{device};
    Metal::VertexRegisters inputs{};
    for(unsigned reg=0;reg<16;++reg) for(unsigned c=0;c<4;++c)
        inputs[reg][c]=float(reg*4+c+1);
    auto buffer=device.CreateBuffer(sizeof(inputs),true);
    std::memcpy(buffer.contents,inputs.data(),sizeof(inputs));
    const Metal::BufferSlice input{buffer,0,sizeof(inputs),nullptr,{}};
    std::vector<Metal::BufferSlice> outputs;
    std::vector<Metal::VertexRegisters> expected;
    const auto tick=scheduler.CurrentTick();
    {
        std::array<std::uint32_t,4096> code{},swizzle{};
        for(unsigned i=0;i<8;++i) {
            code.fill(0);swizzle.fill(0);
            const unsigned source=(i/2)%2,entry=(i&1) ? 8:0;
            code[entry]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,source);
            code[entry+1]=unsigned(nihstro::OpCode::Id::END)<<26;
            code[4095]=i+1;swizzle[4095]=0xA5000000U+i;
            const bool negate=(i&4)!=0;
            swizzle[0]=Metal::Test::IdentitySwizzle^(negate ? 1U<<4:0U);
            Metal::ShaderMachineProgramView view{code,swizzle,entry,1,false};
            state.Reset();
            outputs.push_back(machine.RunVertexBatch(state,view,{},input,1,1,0,0,{},0,1000000,true));
            Metal::VertexRegisters value{};value[0]=inputs[source];
            if(negate) for(auto& c:value[0]) c=-c;
            expected.push_back(value);
            // Neither queued instruction uploads nor cached source identities may
            // retain these borrowed arrays. Destroy them before GPU completion.
            code.fill(UINT32_MAX);swizzle.fill(UINT32_MAX);
        }
    }
    Check(scheduler.CurrentTick()==tick,"Borrowed program batch unexpectedly submitted or waited");
    std::vector<id<MTLBuffer>> readbacks;
    for(const auto& output:outputs) {
        auto read=device.CreateBuffer(sizeof(Metal::VertexRegisters)+32,true);
        std::memset(read.contents,0xA7,read.length);
        Metal::TransferRuntime{device,scheduler}.CopyBuffer(output.buffer,output.offset,read,16,output.size);
        readbacks.push_back(read);
    }
    scheduler.Finish();
    for(unsigned i=0;i<readbacks.size();++i) {
        const auto* bytes=static_cast<const std::byte*>(readbacks[i].contents);
        Check(!std::memcmp(bytes+16,&expected[i],sizeof(expected[i])),
              "Borrowed program mutation/destruction changed queued native output");
        for(unsigned guard=0;guard<16;++guard)
            Check(bytes[guard]==std::byte{0xA7} && bytes[16+sizeof(expected[i])+guard]==std::byte{0xA7},
                  "Borrowed program native output changed guard bytes");
    }
    Metal::ShaderMachineSnapshot snapshot{};snapshot.needs_reset=false;
    std::memcpy(snapshot.history.data(),inputs.data(),sizeof(inputs));
    std::memcpy(snapshot.history.data()+512,&expected.back(),sizeof(expected.back()));
    Check(machine.SaveState(state)==snapshot,"Borrowed program changed complete retained GPU state");
    std::cout<<"8 borrowed program views: destructive code/swizzle mutation and bank destruction before GPU completion; "
               "2048 exact output bytes, guards and1824 complete retained-state bytes passed\n";
}

void CheckResetBankPipelines(Metal::Device& device,Metal::Scheduler& scheduler,
                            Metal::ShaderMachineRuntime& machine) {
    Metal::ShaderMachineRuntime reference{device,scheduler};
    Metal::TransferRuntime transfer{device,scheduler};
    auto input_buffer=device.CreateBuffer(5*sizeof(Metal::VertexRegisters),true);
    auto* values=static_cast<Metal::VertexRegisters*>(input_buffer.contents);
    for(unsigned v=0;v<5;++v) for(unsigned reg=0;reg<16;++reg)
        values[v][reg]={float(v+1)*.125f,float(reg+1)*.25f,-.5f,.75f};
    const Metal::BufferSlice input{input_buffer,0,input_buffer.length,values,{}};
    auto index_buffer=device.CreateBuffer(5,true);
    const std::array<unsigned char,5> ids{0,1,2,0,1};
    std::memcpy(index_buffer.contents,ids.data(),ids.size());
    const Metal::BufferSlice indices{index_buffer,0,5,index_buffer.contents,{}};
    std::array<Metal::ShaderMachineProgram,5> programs{};
    for(auto& program:programs) {program.allow_emission=false;program.output_mask=1;
        program.swizzle[0]=Metal::Test::IdentitySwizzle;}
    programs[0].code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,0);
    programs[0].code[1]=Metal::Test::Arithmetic(nihstro::OpCode::Id::ADD,16,16,0);
    programs[1].code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,16);
    programs[1].code[1]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,16,0);
    for(unsigned p=0;p<2;++p) programs[p].code[2]=unsigned(nihstro::OpCode::Id::END)<<26;
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    programs[2].code[0]=Metal::Test::Arithmetic(Op::MOV,0,0);
    programs[2].code[1]=word(Op::END);
    programs[3].code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
    programs[3].code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
    programs[3].code[2]=Metal::Test::Arithmetic(Op::MOV,0,0);
    programs[3].code[3]=word(Op::NOP);programs[3].code[4]=word(Op::NOP);
    programs[3].code[5]=word(Op::END);
    programs[4].code[0]=Metal::Test::Arithmetic(Op::MOV,0,16);
    programs[4].code[1]=Metal::Test::Arithmetic(Op::ADD,16,16,0);
    programs[4].code[2]=word(Op::END);
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(programs[0],{})==
          Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences,"Reset fixture lost recurrence proof");
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(programs[1],{})==
          Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,"Reset fixture lost dependency proof");
    using Execution=Metal::ShaderMachineRuntime::VertexExecution;
    const std::array expected_classes{Execution::PrivateRecurrences,Execution::DependencyPasses,
        Execution::Independent,Execution::MergeState,Execution::Ordered};
    for(unsigned p=0;p<programs.size();++p)
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(programs[p],{})==expected_classes[p],
              "Reset transition fixture lost its execution class");
    struct Trial {
        Metal::ShaderMachineSnapshot initial,expected_state;
        std::array<Metal::VertexRegisters,5> expected_output{};
        std::unique_ptr<Metal::ShaderMachineState> state;
        id<MTLBuffer> readback;
        unsigned program{},index_size{};
    };
    std::vector<Trial> trials;
    // Revisit each cache in reset/bank-one/bank-zero/reset order. Build a
    // scalar GPU oracle first, then queue all candidate calls without a wait.
    for(unsigned round=0;round<4;++round) for(unsigned p=0;p<programs.size();++p)
    for(unsigned width:{0u,1u}) {
        Trial t{};t.program=p;t.index_size=width;t.initial.needs_reset=round==0 || round==3;
        for(unsigned word=0;!t.initial.needs_reset && word<448;++word) {
            const float value=float(word+1)*.015625f;
            std::memcpy(t.initial.history.data()+word*4,&value,4);
        }
        const unsigned bank=round==1 ? 1:0;
        std::memcpy(t.initial.history.data()+454*4,&bank,4);
        Metal::ShaderMachineState oracle{device};reference.LoadState(oracle,t.initial);
        std::array<Metal::VertexRegisters,5> misses{};
        for(unsigned v=0;v<(width ? 3u:5u);++v) {
            const Metal::BufferSlice source{input_buffer,v*sizeof(Metal::VertexRegisters),
                sizeof(Metal::VertexRegisters),values+v,{}};
            const auto result=reference.Run(oracle,programs[p],{},source);
            std::memcpy(&misses[v],result.output.data,sizeof(Metal::VertexRegisters));
        }
        for(unsigned e=0;e<5;++e) t.expected_output[e]=misses[width ? ids[e]:e];
        t.expected_state=reference.SaveState(oracle);
        t.state=std::make_unique<Metal::ShaderMachineState>(device);
        machine.LoadState(*t.state,t.initial);
        t.readback=device.CreateBuffer(sizeof(t.expected_output),true);
        trials.push_back(std::move(t));
    }
    const auto tick=scheduler.CurrentTick();
    for(auto& t:trials) {
        const auto output=machine.RunVertexBatch(*t.state,programs[t.program],{},input,5,5,0,0,
            t.index_size ? indices:Metal::BufferSlice{},t.index_size,1000000,true);
        transfer.CopyBuffer(output.buffer,output.offset,t.readback,0,output.size);
    }
    Check(scheduler.CurrentTick()==tick,"Reset/bank pipeline selection unexpectedly waited");
    scheduler.Finish();
    for(auto& t:trials) {
        Check(!std::memcmp(t.readback.contents,t.expected_output.data(),sizeof(t.expected_output)),
              "Reset/inherited-bank pipeline cache changed queued outputs");
        Check(machine.SaveState(*t.state)==t.expected_state,
              "Reset/inherited-bank pipeline cache changed complete retained state");
    }
    std::cout<<"40 queued reset/bank-one/bank-zero pipeline transitions: all five execution classes, sequential IDs/FIFO hits, "
               "complete outputs and1824-byte states match scalar GPU execution\n";
}

void CheckCombinedVertexCapture(Metal::Device& device,Metal::Scheduler& scheduler,
                               Metal::ShaderMachineRuntime& machine) {
    using Execution=Metal::ShaderMachineRuntime::VertexExecution;
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    const auto arithmetic=[](Op op,unsigned dest,unsigned a,unsigned b=0) {
        return Metal::Test::Arithmetic(op,dest,a,b);
    };
    std::array<Metal::ShaderMachineProgram,5> programs{};
    for(auto& p:programs) {p.allow_emission=false;p.output_mask=7;p.swizzle[0]=Metal::Test::IdentitySwizzle;}
    programs[0].code[0]=arithmetic(Op::MOV,0,16);
    programs[0].code[1]=arithmetic(Op::ADD,16,16,0);programs[0].code[2]=word(Op::END);
    for(unsigned i=0;i<3;++i)programs[1].code[i]=arithmetic(Op::MOV,i,i);
    programs[1].code[3]=word(Op::END);
    programs[2].code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
    programs[2].code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
    programs[2].code[2]=arithmetic(Op::MOV,0,0);programs[2].code[3]=word(Op::NOP);
    programs[2].code[4]=word(Op::NOP);programs[2].code[5]=word(Op::END);
    programs[3].code[0]=arithmetic(Op::MOV,0,16);
    programs[3].code[1]=arithmetic(Op::MOV,16,0);programs[3].code[2]=word(Op::END);
    programs[4].code[0]=arithmetic(Op::MOV,0,0);
    programs[4].code[1]=arithmetic(Op::ADD,16,16,0);programs[4].code[2]=word(Op::END);
    for(unsigned i=0;i<programs.size();++i)
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(programs[i],{})==Execution(i),
              "Combined capture fixture lost its execution class");
    Metal::ShaderMachineRuntime reference{device,scheduler};Metal::TransferRuntime transfer{device,scheduler};
    constexpr unsigned base=9,max_vertices=257;
    auto source=device.CreateBuffer(16+max_vertices*sizeof(Metal::VertexRegisters),true);
    auto* values=reinterpret_cast<Metal::VertexRegisters*>(static_cast<std::byte*>(source.contents)+16);
    for(unsigned v=0;v<max_vertices;++v)for(unsigned reg=0;reg<16;++reg)
        values[v][reg]={float(int(v%7)-3)*.125f,float(reg+1)*.0625f,-.5f,1};
    struct Trial {
        unsigned program{},count{},vertices{},index_size{};
        Metal::GeometryEmissionConfig config{};
        Metal::ShaderMachineSnapshot initial{},final{};
        std::unique_ptr<Metal::ShaderMachineState> state;
        id<MTLBuffer> index,raw_expected,captured_expected,raw_read,captured_read;
        Metal::ShaderMachineVertexBatchResult retained;
    };
    std::vector<Trial> trials;
    for(unsigned p=0;p<5;++p)for(unsigned bank=0;bank<3;++bank)for(unsigned width:{0u,1u,2u,4u}) {
        Trial t;t.program=p;t.index_size=width;t.count=std::array<unsigned,3>{1,65,257}[trials.size()%3];
        t.vertices=width ? 70:t.count;
        t.initial.needs_reset=bank==0;
        if(bank) {
            for(unsigned lane=0;lane<448;++lane) {
                const float value=float(lane%23)*.015625f;
                std::memcpy(t.initial.history.data()+lane*4,&value,4);
            }
            const unsigned active=bank-1;std::memcpy(t.initial.history.data()+1816,&active,4);
        }
        t.config.attribute_count=trials.size()%8;t.config.honor_winding=0;
        t.config.semantics={0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x1f171614,0x1f1f1f1f};
        t.config.flip_viewport=trials.size()&1;t.config.enable_clip=(trials.size()>>1)&1;
        t.config.clip_coefficient={.25f,-.5f,.125f,float(trials.size()%5)};
        t.index=device.CreateBuffer(3+std::max(1u,width)*t.count,true);
        auto* bytes=static_cast<unsigned char*>(t.index.contents)+3;
        for(unsigned e=0;e<t.count;++e) {
            const unsigned id=base+(e<70 ? e:(e*7)%70);
            for(unsigned b=0;b<width;++b)bytes[e*width+b]=id>>(8*b);
        }
        const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}},
            indices{t.index,3,width*t.count,bytes,{}};
        Metal::ShaderMachineState oracle{device};reference.LoadState(oracle,t.initial);
        const auto raw=reference.RunVertexBatch(oracle,programs[p],{},input,t.vertices,t.count,base,base,
            width ? indices:Metal::BufferSlice{},width,1000000,true);
        const auto captured=reference.CaptureVertices(raw,7,t.config,t.count);
        t.raw_expected=device.CreateBuffer(raw.size,true);t.captured_expected=device.CreateBuffer(captured.size,true);
        transfer.CopyBuffer(raw.buffer,raw.offset,t.raw_expected,0,raw.size);
        transfer.CopyBuffer(captured.buffer,captured.offset,t.captured_expected,0,captured.size);
        t.final=reference.SaveState(oracle);
        t.state=std::make_unique<Metal::ShaderMachineState>(device);machine.LoadState(*t.state,t.initial);
        t.raw_read=device.CreateBuffer(raw.size+32,true);t.captured_read=device.CreateBuffer(captured.size+32,true);
        std::memset(t.raw_read.contents,0xA7,t.raw_read.length);std::memset(t.captured_read.contents,0xA7,t.captured_read.length);
        trials.push_back(std::move(t));
    }
    const auto tick=scheduler.CurrentTick();
    for(auto& t:trials) {
        const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}},
            indices{t.index,3,t.index_size*t.count,static_cast<std::byte*>(t.index.contents)+3,{}};
        auto config=t.config;
        t.retained=machine.RunVertexBatchAndCapture(*t.state,programs[t.program],{},input,config,
            t.vertices,t.count,base,base,t.index_size ? indices:Metal::BufferSlice{},t.index_size,1000000,true);
        config={}; // Conversion arguments must be an immutable queued snapshot.
        Check(t.retained.output.lease && t.retained.captured.lease && !t.retained.output.data &&
            !t.retained.captured.data && t.retained.output.size==t.raw_expected.length &&
            t.retained.captured.size==t.captured_expected.length,"Combined capture lost output ownership/extents");
    }
    Check(scheduler.CurrentTick()==tick,"Combined capture unexpectedly submitted or waited");
    scheduler.Submit();
    // Encode all consumers late, after other runtimes have reused their argument
    // tables and after submission. Both returned slices must remain alive.
    for(auto& t:trials) {
        transfer.CopyBuffer(t.retained.output.buffer,t.retained.output.offset,t.raw_read,16,t.retained.output.size);
        transfer.CopyBuffer(t.retained.captured.buffer,t.retained.captured.offset,t.captured_read,16,t.retained.captured.size);
    }
    scheduler.Finish();
    for(auto& t:trials) {
        Check(!std::memcmp(static_cast<std::byte*>(t.raw_read.contents)+16,t.raw_expected.contents,t.raw_expected.length) &&
            !std::memcmp(static_cast<std::byte*>(t.captured_read.contents)+16,t.captured_expected.contents,t.captured_expected.length),
            "Combined capture changed complete raw/converted output or delayed lifetime");
        for(unsigned i=0;i<16;++i)
            Check(static_cast<unsigned char*>(t.raw_read.contents)[i]==0xA7 &&
                static_cast<unsigned char*>(t.raw_read.contents)[16+t.raw_expected.length+i]==0xA7 &&
                static_cast<unsigned char*>(t.captured_read.contents)[i]==0xA7 &&
                static_cast<unsigned char*>(t.captured_read.contents)[16+t.captured_expected.length+i]==0xA7,
                "Combined capture damaged readback guards");
        Check(machine.SaveState(*t.state)==t.final,"Combined capture changed complete1824-byte shader state");
    }
    auto& t=trials.back();const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}};
    auto config=t.config;config.honor_winding=0;config.attribute_count=1;config.semantics[0]=0x03020100;
    const auto stable=machine.SaveState(*t.state);
    for(unsigned invalid=0;invalid<8;++invalid) {
        auto bad=config;auto program=programs[1];
        if(invalid==0)bad.attribute_count=8;
        if(invalid==1)bad.flip_viewport=2;
        if(invalid==2)bad.enable_clip=2;
        if(invalid==3)bad.fix_quaternions=1;
        if(invalid==4)bad.honor_winding=1;
        if(invalid==5)bad.semantics[0]=24;
        if(invalid==6)program.output_mask=0x10000;
        if(invalid==7)bad.semantics[0]=30;
        const auto before=scheduler.CurrentTick();bool rejected=false;
        try {machine.RunVertexBatchAndCapture(*t.state,program,{},input,bad,t.vertices,1,base,base);}
        catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected && scheduler.CurrentTick()==before && machine.SaveState(*t.state)==stable,
            "Invalid combined capture metadata queued work or changed shader state");
    }
    for(unsigned failure=0;failure<2;++failure) {
        auto bad=programs[1];if(failure)bad.code[1]=0x10u<<26;bool rejected=false;
        try {machine.RunVertexBatchAndCapture(*t.state,bad,{},input,config,t.vertices,1,base,base,{},0,failure ? 1000000:1);}
        catch(const Metal::ShaderMachineError& e) {rejected=e.Code()==(failure ? 2:6);}
        Check(rejected && machine.SaveState(*t.state)==stable,"Failed combined capture committed partial shader state");
    }
    {
        constexpr unsigned count=4096;
        Metal::Scheduler queue{device};Metal::ShaderMachineRuntime runtime{device,queue};Metal::ShaderMachineState state{device};
        Metal::TransferRuntime copies{device,queue};auto program=programs[1];program.output_mask=1;
        program.code[0]=arithmetic(Op::MOV,0,32);program.code[1]=word(Op::END);
        auto read=device.CreateBuffer(count*(sizeof(Metal::VertexRegisters)+sizeof(Metal::CapturedVertex)),true);
        const auto before=queue.CurrentTick();Metal::ShaderMachineUniforms uniforms{};
        for(unsigned draw=0;draw<count;++draw) {
            uniforms.f[0]={float(draw),-.5f,float(draw%17)*.25f,1};
            config.flip_viewport=draw&1;config.enable_clip=1;config.clip_coefficient={1,0,0,float(draw%5)};
            const auto result=runtime.RunVertexBatchAndCapture(state,program,uniforms,input,config,t.vertices,1,base,base,{},0,1000000,true);
            const auto offset=draw*(sizeof(Metal::VertexRegisters)+sizeof(Metal::CapturedVertex));
            copies.CopyBuffer(result.output.buffer,result.output.offset,read,offset,result.output.size);
            copies.CopyBuffer(result.captured.buffer,result.captured.offset,read,offset+sizeof(Metal::VertexRegisters),result.captured.size);
        }
        Check(queue.CurrentTick()>before,"Combined capture did not exercise upload-page recycling");queue.Finish();
        for(unsigned draw=0;draw<count;++draw) {
            Metal::VertexRegisters raw{};raw[0]={float(draw),-.5f,float(draw%17)*.25f,1};
            Metal::CapturedVertex captured{};captured.position={float(draw),draw&1 ? .5f:-.5f,-float(draw%17)*.25f,1};
            captured.color=captured.quaternion={1,1,1,1};captured.texcoord0=captured.texcoord1=captured.texcoord2={1,1,0,0};
            captured.view={1,1,1,0};captured.clip={captured.position[2],float(draw+draw%5),1,0};
            const auto* bytes=static_cast<std::byte*>(read.contents)+draw*(sizeof(raw)+sizeof(captured));
            Check(!std::memcmp(bytes,&raw,sizeof(raw)) && !std::memcmp(bytes+sizeof(raw),&captured,sizeof(captured)),
                "Combined upload recycling overwrote raw output, conversion metadata or constants");
        }
        Metal::ShaderMachineState oracle{device};runtime.Run(oracle,program,uniforms,input);
        Check(runtime.SaveState(state)==runtime.SaveState(oracle),"Combined upload recycling changed complete retained state");
    }
    for(unsigned failure_draw:{127u,767u}) {
        Metal::Scheduler queue{device};Metal::ShaderMachineRuntime runtime{device,queue};
        Metal::ShaderMachineState state{device};auto program=programs[1];
        bool rejected=false,sticky=false;unsigned queued=0;
        try {
            for(;queued<4096;++queued)
                runtime.RunVertexBatchAndCapture(state,program,{},input,config,t.vertices,1,base,base,{},0,
                    queued==failure_draw ? 1:1000000,true);
            queue.Finish();
        } catch(const Metal::ShaderMachineError& e) {rejected=e.Code()==6;}
        try {queue.IsComplete(0);}catch(const Metal::ShaderMachineError& e) {sticky=e.Code()==6;}
        Check(rejected && sticky && queued>failure_draw && queued<4096,
            "Combined capture lost deferred shader failure during upload recycling");
    }
    std::cout<<"60 combined VS/capture batches: all5 execution classes, reset/both inherited banks, all index widths, FIFO eviction, delayed owned slices, immutable conversion metadata, guards/full1824-byte states;8 metadata rejects,2 atomic shader failures and4096 upload-recycling draws plus2 sticky deferred failures passed\n";
}
void CheckFetchedVertexCapture(Metal::Device& device,Metal::Scheduler& scheduler,
                               Metal::ShaderMachineRuntime& machine) {
    using Execution=Metal::ShaderMachineRuntime::VertexExecution;
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    const auto arithmetic=[](Op op,unsigned dest,unsigned a,unsigned b=0) {
        return Metal::Test::Arithmetic(op,dest,a,b);
    };
    std::array<Metal::ShaderMachineProgram,5> programs{};
    for(auto& p:programs) {p.allow_emission=false;p.output_mask=7;p.swizzle[0]=Metal::Test::IdentitySwizzle;}
    programs[0].code[0]=arithmetic(Op::MOV,0,16);
    programs[0].code[1]=arithmetic(Op::ADD,16,16,0);programs[0].code[2]=word(Op::END);
    for(unsigned i=0;i<3;++i)programs[1].code[i]=arithmetic(Op::MOV,i,i);
    programs[1].code[3]=word(Op::END);
    programs[2].code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
    programs[2].code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
    programs[2].code[2]=arithmetic(Op::MOV,0,0);programs[2].code[3]=word(Op::NOP);
    programs[2].code[4]=word(Op::NOP);programs[2].code[5]=word(Op::END);
    programs[3].code[0]=arithmetic(Op::MOV,0,16);
    programs[3].code[1]=arithmetic(Op::MOV,16,0);programs[3].code[2]=word(Op::END);
    programs[4].code[0]=arithmetic(Op::MOV,0,0);
    programs[4].code[1]=arithmetic(Op::ADD,16,16,0);programs[4].code[2]=word(Op::END);
    for(unsigned i=0;i<programs.size();++i)
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(programs[i],{})==Execution(i),
              "Combined capture fixture lost its execution class");
    Metal::ShaderMachineRuntime reference{device,scheduler};Metal::TransferRuntime transfer{device,scheduler};
    Metal::VertexFetchRuntime fetch{device,scheduler},reference_fetch{device,scheduler};
    Metal::PicaVertexLayout identity{};identity.mapped_attribute_count=16;
    identity.fixed_attributes=Metal::DefaultVertexRegisters();
    for(unsigned i=0;i<16;++i) {
        identity.register_map[i]=i;identity.attribute_inputs[i].mode=Metal::PicaAttributeMode::Buffer;
        identity.attribute_inputs[i].fetch={0,i*16,256,4,Metal::VertexAttributeFormat::Float,i};
    }
    constexpr unsigned base=9,max_vertices=257;
    auto source=device.CreateBuffer(16+max_vertices*sizeof(Metal::VertexRegisters),true);
    auto* values=reinterpret_cast<Metal::VertexRegisters*>(static_cast<std::byte*>(source.contents)+16);
    for(unsigned v=0;v<max_vertices;++v)for(unsigned reg=0;reg<16;++reg)
        values[v][reg]={float(int(v%7)-3)*.125f,float(reg+1)*.0625f,-.5f,1};
    struct Trial {
        unsigned program{},count{},vertices{},index_size{};
        Metal::GeometryEmissionConfig config{};
        Metal::ShaderMachineSnapshot initial{},final{};
        std::unique_ptr<Metal::ShaderMachineState> state;
        id<MTLBuffer> index,raw_expected,captured_expected,raw_read,captured_read;
        Metal::ShaderMachineFetchedVertexBatchResult retained;
        Metal::PicaVertexLayout layout{};
        Metal::VertexInputSnapshot input_initial{},input_final{};
        std::unique_ptr<Metal::VertexInputState> input_state;
        id<MTLBuffer> input_expected,input_read;
    };
    std::vector<Trial> trials;
    for(unsigned p=0;p<5;++p)for(unsigned bank=0;bank<3;++bank)for(unsigned width:{0u,1u,2u,4u}) {
        Trial t;t.program=p;t.index_size=width;t.count=std::array<unsigned,3>{1,65,257}[trials.size()%3];
        t.vertices=width ? 70:t.count;
        t.initial.needs_reset=bank==0;
        if(bank) {
            for(unsigned lane=0;lane<448;++lane) {
                const float value=float(lane%23)*.015625f;
                std::memcpy(t.initial.history.data()+lane*4,&value,4);
            }
            const unsigned active=bank-1;std::memcpy(t.initial.history.data()+1816,&active,4);
        }
        t.layout=identity;
        t.layout.mapped_attribute_count=trials.size()%17;
        for(unsigned i=0;i<16;++i) {
            t.layout.register_map[i]=(i*7+trials.size())%16;
            t.layout.fixed_attributes[i]={float(i)*.125f,-.25f,.5f,1};
            auto& a=t.layout.attribute_inputs[i];
            a.mode=Metal::PicaAttributeMode((i+trials.size())%3);
            a.fetch.format=Metal::VertexAttributeFormat((i+trials.size())%4);
            a.fetch.components=1+(i+trials.size())%4;
        }
        t.layout.register_map[15]=t.layout.register_map[0]; // Ordered alias and retained hole.
        t.input_initial.needs_reset=(trials.size()&1)==0;
        if(!t.input_initial.needs_reset) for(unsigned i=0;i<128;++i) {
            const float value=float(int(i%19)-9)*.03125f;
            std::memcpy(t.input_initial.history.data()+i*4,&value,4);
        }
        t.config.attribute_count=trials.size()%8;t.config.honor_winding=0;
        t.config.semantics={0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x1f171614,0x1f1f1f1f};
        t.config.flip_viewport=trials.size()&1;t.config.enable_clip=(trials.size()>>1)&1;
        t.config.clip_coefficient={.25f,-.5f,.125f,float(trials.size()%5)};
        t.index=device.CreateBuffer(3+std::max(1u,width)*t.count,true);
        auto* bytes=static_cast<unsigned char*>(t.index.contents)+3;
        for(unsigned e=0;e<t.count;++e) {
            const unsigned id=base+(e<70 ? e:(e*7)%70);
            for(unsigned b=0;b<width;++b)bytes[e*width+b]=id>>(8*b);
        }
        const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}},
            indices{t.index,3,width*t.count,bytes,{}};
        const std::array<Metal::BufferSlice,1> sources{{input}};
        Metal::VertexInputState input_oracle{device};reference_fetch.LoadState(input_oracle,t.input_initial);
        const auto canonical=reference_fetch.FetchPica(input_oracle,sources,t.layout,t.vertices,t.count,base,base,
            width ? indices:Metal::BufferSlice{},width);
        t.input_expected=device.CreateBuffer(canonical.size,true);
        transfer.CopyBuffer(canonical.buffer,canonical.offset,t.input_expected,0,canonical.size);
        t.input_final=reference_fetch.SaveState(input_oracle);
        Metal::ShaderMachineState oracle{device};reference.LoadState(oracle,t.initial);
        const auto raw=reference.RunVertexBatch(oracle,programs[p],{},canonical,t.vertices,t.count,base,base,
            width ? indices:Metal::BufferSlice{},width,1000000,true);
        const auto captured=reference.CaptureVertices(raw,7,t.config,t.count);
        t.raw_expected=device.CreateBuffer(raw.size,true);t.captured_expected=device.CreateBuffer(captured.size,true);
        transfer.CopyBuffer(raw.buffer,raw.offset,t.raw_expected,0,raw.size);
        transfer.CopyBuffer(captured.buffer,captured.offset,t.captured_expected,0,captured.size);
        t.final=reference.SaveState(oracle);
        t.state=std::make_unique<Metal::ShaderMachineState>(device);machine.LoadState(*t.state,t.initial);
        t.input_state=std::make_unique<Metal::VertexInputState>(device);fetch.LoadState(*t.input_state,t.input_initial);
        t.input_read=device.CreateBuffer(t.input_expected.length+32,true);
        std::memset(t.input_read.contents,0xA7,t.input_read.length);
        t.raw_read=device.CreateBuffer(raw.size+32,true);t.captured_read=device.CreateBuffer(captured.size+32,true);
        std::memset(t.raw_read.contents,0xA7,t.raw_read.length);std::memset(t.captured_read.contents,0xA7,t.captured_read.length);
        trials.push_back(std::move(t));
    }
    const auto tick=scheduler.CurrentTick();
    for(auto& t:trials) {
        const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}},
            indices{t.index,3,t.index_size*t.count,static_cast<std::byte*>(t.index.contents)+3,{}};
        auto config=t.config;
        const std::array<Metal::BufferSlice,1> inputs{{input}};auto layout=t.layout;
        t.retained=machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,inputs,layout,
            *t.state,programs[t.program],{},config,
            t.vertices,t.count,base,base,t.index_size ? indices:Metal::BufferSlice{},t.index_size,1000000,true);
        config={};layout={}; // Both fetch and conversion metadata must be immutable snapshots.
        Check(t.retained.input.lease && t.retained.input.size==t.input_expected.length,
              "Fetched batch lost canonical input ownership/extents");
        Check(t.retained.output.lease && t.retained.captured.lease && !t.retained.output.data &&
            !t.retained.captured.data && t.retained.output.size==t.raw_expected.length &&
            t.retained.captured.size==t.captured_expected.length,"Combined capture lost output ownership/extents");
    }
    Check(scheduler.CurrentTick()==tick,"Combined capture unexpectedly submitted or waited");
    scheduler.Submit();
    // Encode all consumers late, after other runtimes have reused their argument
    // tables and after submission. Both returned slices must remain alive.
    for(auto& t:trials) {
        transfer.CopyBuffer(t.retained.input.buffer,t.retained.input.offset,t.input_read,16,t.retained.input.size);
        transfer.CopyBuffer(t.retained.output.buffer,t.retained.output.offset,t.raw_read,16,t.retained.output.size);
        transfer.CopyBuffer(t.retained.captured.buffer,t.retained.captured.offset,t.captured_read,16,t.retained.captured.size);
    }
    scheduler.Finish();
    for(auto& t:trials) {
        Check(!std::memcmp(static_cast<std::byte*>(t.raw_read.contents)+16,t.raw_expected.contents,t.raw_expected.length) &&
            !std::memcmp(static_cast<std::byte*>(t.captured_read.contents)+16,t.captured_expected.contents,t.captured_expected.length),
            "Combined capture changed complete raw/converted output or delayed lifetime");
        for(unsigned i=0;i<16;++i)
            Check(static_cast<unsigned char*>(t.raw_read.contents)[i]==0xA7 &&
                static_cast<unsigned char*>(t.raw_read.contents)[16+t.raw_expected.length+i]==0xA7 &&
                static_cast<unsigned char*>(t.captured_read.contents)[i]==0xA7 &&
                static_cast<unsigned char*>(t.captured_read.contents)[16+t.captured_expected.length+i]==0xA7,
                "Combined capture damaged readback guards");
        Check(machine.SaveState(*t.state)==t.final,"Fetched batch changed complete1824-byte shader state");
        Check(!std::memcmp(static_cast<std::byte*>(t.input_read.contents)+16,t.input_expected.contents,t.input_expected.length),
              "Fetched batch changed complete canonical inputs");
        for(unsigned i=0;i<16;++i) Check(static_cast<unsigned char*>(t.input_read.contents)[i]==0xA7 &&
            static_cast<unsigned char*>(t.input_read.contents)[16+t.input_expected.length+i]==0xA7,
            "Fetched batch damaged canonical input guards");
        const auto input_state=fetch.SaveState(*t.input_state);
        Check(input_state.needs_reset==t.input_final.needs_reset && input_state.history==t.input_final.history,
              "Fetched batch changed complete512-byte input history");
    }
    for(auto& t:trials) {
        const auto input_bytes=t.vertices*sizeof(Metal::VertexRegisters);
        const auto index_offset=16+input_bytes+3;
        const auto shared=device.CreateBuffer(index_offset+t.index_size*t.count+16,true);
        std::memset(shared.contents,0x5A,shared.length);
        std::memcpy(static_cast<std::byte*>(shared.contents)+16,values,input_bytes);
        std::memcpy(static_cast<std::byte*>(shared.contents)+index_offset,
                    static_cast<std::byte*>(t.index.contents)+3,t.index_size*t.count);
        const Metal::BufferSlice input{shared,16,input_bytes,static_cast<std::byte*>(shared.contents)+16,{}},
            indices{shared,index_offset,t.index_size*t.count,static_cast<std::byte*>(shared.contents)+index_offset,{}};
        const std::array<Metal::BufferSlice,1> inputs{{input}};
        machine.LoadState(*t.state,t.initial);fetch.LoadState(*t.input_state,t.input_initial);
        const auto result=machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,inputs,t.layout,*t.state,
            programs[t.program],{},t.config,t.vertices,t.count,base,base,t.index_size ? indices:Metal::BufferSlice{},t.index_size,1000000,true);
        transfer.CopyBuffer(result.input.buffer,result.input.offset,t.input_read,16,result.input.size);
        transfer.CopyBuffer(result.output.buffer,result.output.offset,t.raw_read,16,result.output.size);
        transfer.CopyBuffer(result.captured.buffer,result.captured.offset,t.captured_read,16,result.captured.size);
        scheduler.Finish();
        Check(!std::memcmp(static_cast<std::byte*>(t.input_read.contents)+16,t.input_expected.contents,t.input_expected.length) &&
              !std::memcmp(static_cast<std::byte*>(t.raw_read.contents)+16,t.raw_expected.contents,t.raw_expected.length) &&
              !std::memcmp(static_cast<std::byte*>(t.captured_read.contents)+16,t.captured_expected.contents,t.captured_expected.length),
              "Shared input/index allocation changed fused inputs or complete outputs");
        const auto history=fetch.SaveState(*t.input_state);
        Check(history.needs_reset==t.input_final.needs_reset && history.history==t.input_final.history &&
              machine.SaveState(*t.state)==t.final,"Shared input/index allocation changed fused histories");
        for(unsigned i=0;i<16;++i) Check(static_cast<unsigned char*>(shared.contents)[i]==0x5A &&
            static_cast<unsigned char*>(shared.contents)[shared.length-16+i]==0x5A,
            "Fused shared input/index allocation damaged source guards");
    }
    unsigned index_error_cases=0;
    for(unsigned p=0;p<5;++p) for(unsigned width:{1u,2u,4u})
    for(unsigned bad:{0u,1u,31u,32u,63u,64u,65u,128u,256u}) {
        auto& t=trials[p*12+4];
        const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}};
        const std::array<Metal::BufferSlice,1> inputs{{input}};
        constexpr unsigned elements=257;
        const auto index=device.CreateBuffer(3+elements*width+16,true);
        std::memset(index.contents,0xA7,index.length);
        auto* bytes=static_cast<unsigned char*>(index.contents)+3;
        for(unsigned e=0;e<elements;++e) {
            const unsigned id=base+(e==bad ? t.vertices:e%t.vertices);
            for(unsigned byte=0;byte<width;++byte)bytes[e*width+byte]=id>>(8*byte);
        }
        const Metal::BufferSlice ix{index,3,elements*width,bytes,{}};
        Metal::VertexInputState input_oracle{device};reference_fetch.LoadState(input_oracle,t.input_initial);
        reference_fetch.FetchPica(input_oracle,inputs,t.layout,t.vertices,elements,base,base,ix,width);
        const auto expected_history=reference_fetch.SaveState(input_oracle);
        machine.LoadState(*t.state,t.initial);fetch.LoadState(*t.input_state,t.input_initial);bool rejected=false;
        try {machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,inputs,t.layout,*t.state,
            programs[p],{},t.config,t.vertices,elements,base,base,ix,width);}
        catch(const Metal::ShaderMachineError& e) {
            rejected=e.Code()==8;
        }
        const auto history=fetch.SaveState(*t.input_state);
        Check(rejected && machine.SaveState(*t.state)==t.initial &&
              history.needs_reset==expected_history.needs_reset && history.history==expected_history.history,
              "Fused GPU index error changed error code, atomic VS state or valid-suffix fetch history");
        for(unsigned i=0;i<3;++i)Check(static_cast<unsigned char*>(index.contents)[i]==0xA7,
              "Fused index validation damaged prefix guards");
        for(unsigned i=0;i<16;++i)Check(bytes[elements*width+i]==0xA7,
              "Fused index validation damaged suffix guards");
        ++index_error_cases;
    }
    Check(index_error_cases==135,"Incomplete fetched first-index-error boundary matrix");
    {
        const auto path=std::filesystem::temp_directory_path()/(
            "azahar-metal4-fetched-diagnostic-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
        struct CaptureEnvironment {
            bool had{};std::string previous;
            CaptureEnvironment(const std::filesystem::path& p) {
                const auto* old=std::getenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");had=old!=nullptr;if(old)previous=old;
                setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",p.c_str(),1);
            }
            ~CaptureEnvironment(){if(had)setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",previous.c_str(),1);else unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");}
        };
        Metal::Scheduler queue{device};Metal::VertexFetchRuntime input_runtime{device,queue};
        std::unique_ptr<Metal::ShaderMachineRuntime> recorded;
        {CaptureEnvironment env{path};recorded=std::make_unique<Metal::ShaderMachineRuntime>(device,queue);}
        const Metal::BufferSlice input{source,16,max_vertices*sizeof(Metal::VertexRegisters),values,{}};
        const std::array<Metal::BufferSlice,1> inputs{{input}};
        for(unsigned p=0;p<5;++p) {
            {std::ofstream request(path/"capture.request");request<<(p+1);}
            if(p)queue.EndProfileFrame();
            Metal::VertexInputState input_state{device};Metal::ShaderMachineState state{device};
            Metal::GeometryEmissionConfig config{};config.attribute_count=1;config.semantics[0]=0x03020100;config.honor_winding=0;
            recorded->RunFetchedVertexBatchAndCapture(input_runtime,input_state,inputs,identity,state,programs[p],{},config,max_vertices,3,base,base);
        }
        queue.Finish();unsigned count=0;
        for(const auto& entry:std::filesystem::directory_iterator(path))count+=entry.path().extension()==".vdc";
        Check(count==5,"Fused input diagnostics lost an execution class or complete record");
        std::cout<<"FUSED_DIAGNOSTIC_DIR="<<path.string()<<'\n';
    }
    auto& t=trials.back();const Metal::BufferSlice input{source,16,t.vertices*sizeof(Metal::VertexRegisters),values,{}};
    auto config=t.config;config.honor_winding=0;config.attribute_count=1;config.semantics[0]=0x03020100;
    const auto stable=machine.SaveState(*t.state);
    const auto stable_input=fetch.SaveState(*t.input_state);
    const std::array<Metal::BufferSlice,1> sources{{input}};
    for(unsigned invalid=0;invalid<8;++invalid) {
        auto bad=config;auto program=programs[1];
        if(invalid==0)bad.attribute_count=8;
        if(invalid==1)bad.flip_viewport=2;
        if(invalid==2)bad.enable_clip=2;
        if(invalid==3)bad.fix_quaternions=1;
        if(invalid==4)bad.honor_winding=1;
        if(invalid==5)bad.semantics[0]=24;
        if(invalid==6)program.output_mask=0x10000;
        if(invalid==7)bad.semantics[0]=30;
        const auto before=scheduler.CurrentTick();bool rejected=false;
        try {machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,sources,t.layout,*t.state,program,{},bad,t.vertices,1,base,base);}
        catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected && scheduler.CurrentTick()==before && machine.SaveState(*t.state)==stable,
            "Invalid fetched metadata queued work or changed shader state");
        const auto input_after=fetch.SaveState(*t.input_state);
        Check(input_after.needs_reset==stable_input.needs_reset && input_after.history==stable_input.history,
              "Invalid fetched metadata changed input state");
    }
    for(unsigned invalid=0;invalid<12;++invalid) {
        auto bad=programs[1];auto layout=t.layout;auto local_config=config;
        unsigned vertices=t.vertices,elements=1,first=base,width=0,limit=1000000;
        Metal::BufferSlice ix{};
        if(invalid==0)bad.entry=4096;
        if(invalid==1)bad.allow_emission=true;
        if(invalid==2)limit=0;
        if(invalid==3)limit=1000001;
        if(invalid==4)vertices=0;
        if(invalid==5)elements=0;
        if(invalid==6)first=base-1;
        if(invalid==7)layout.mapped_attribute_count=17;
        if(invalid==8)layout.register_map[0]=16;
        if(invalid==9){layout.attribute_inputs[0].mode=Metal::PicaAttributeMode::Buffer;layout.attribute_inputs[0].fetch.format=Metal::VertexAttributeFormat(4);}
        if(invalid==10)width=3;
        if(invalid==11){width=2;ix={t.index,0,1,t.index.contents,{}};}
        const auto before=scheduler.CurrentTick();bool rejected=false;
        try {machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,sources,layout,*t.state,bad,{},local_config,
                 vertices,elements,base,first,ix,width,limit,true);}
        catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && scheduler.CurrentTick()==before && machine.SaveState(*t.state)==stable,
              "Invalid fetched dimensions/layout/program committed VS work");
        const auto after=fetch.SaveState(*t.input_state);
        Check(after.history==stable_input.history && after.needs_reset==stable_input.needs_reset,
              "Invalid fetched dimensions/layout/program changed input state");
    }
    {
        const auto before=scheduler.CurrentTick();auto bad=programs[1];bad.entry=4096;
        for(unsigned i=0;i<4096;++i) {
            bool rejected=false;
            try {machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,sources,t.layout,*t.state,bad,{},config,t.vertices,1,base,base);}
            catch(const std::invalid_argument&){rejected=true;}
            Check(rejected && scheduler.CurrentTick()==before,"Invalid source allocated enough fetch pages to submit/wait");
        }
        Metal::Scheduler other_queue{device};Metal::VertexFetchRuntime other_fetch{device,other_queue};
        Metal::VertexInputState other_input{device};bool rejected=false;
        try {machine.RunFetchedVertexBatchAndCapture(other_fetch,other_input,sources,t.layout,*t.state,programs[1],{},config,t.vertices,1,base,base);}
        catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && scheduler.CurrentTick()==before,"Foreign fetch queue accepted by composed shader runtime");
    }
    for(unsigned failure=0;failure<2;++failure) {
        auto bad=programs[1];if(failure)bad.code[1]=0x10u<<26;bool rejected=false;
        try {machine.RunFetchedVertexBatchAndCapture(fetch,*t.input_state,sources,t.layout,*t.state,bad,{},config,t.vertices,1,base,base,{},0,failure ? 1000000:1);}
        catch(const Metal::ShaderMachineError& e) {rejected=e.Code()==(failure ? 2:6);}
        Check(rejected && machine.SaveState(*t.state)==stable,"Failed combined capture committed partial shader state");
    }
    {
        constexpr unsigned count=4096;
        Metal::Scheduler queue{device};Metal::ShaderMachineRuntime runtime{device,queue};Metal::ShaderMachineState state{device};
        Metal::TransferRuntime copies{device,queue};Metal::VertexFetchRuntime input_runtime{device,queue};
        Metal::VertexInputState input_state{device};auto varying_layout=identity;
        varying_layout.attribute_inputs[0].mode=Metal::PicaAttributeMode::Fixed;
        auto program=programs[1];program.output_mask=1;
        program.code[0]=arithmetic(Op::MOV,0,32);program.code[1]=word(Op::END);
        const auto input_bytes=t.vertices*sizeof(Metal::VertexRegisters);
        const auto stride=sizeof(Metal::VertexRegisters)+sizeof(Metal::CapturedVertex)+input_bytes;
        auto read=device.CreateBuffer(count*stride,true);
        const auto before=queue.CurrentTick();Metal::ShaderMachineUniforms uniforms{};
        for(unsigned draw=0;draw<count;++draw) {
            uniforms.f[0]={float(draw),-.5f,float(draw%17)*.25f,1};
            config.flip_viewport=draw&1;config.enable_clip=1;config.clip_coefficient={1,0,0,float(draw%5)};
            varying_layout.fixed_attributes[0]=uniforms.f[0];
            const auto result=runtime.RunFetchedVertexBatchAndCapture(input_runtime,input_state,sources,varying_layout,
                state,program,uniforms,config,t.vertices,1,base,base,{},0,1000000,true);
            const auto offset=draw*stride;
            copies.CopyBuffer(result.input.buffer,result.input.offset,read,offset+sizeof(Metal::VertexRegisters)+sizeof(Metal::CapturedVertex),result.input.size);
            copies.CopyBuffer(result.output.buffer,result.output.offset,read,offset,result.output.size);
            copies.CopyBuffer(result.captured.buffer,result.captured.offset,read,offset+sizeof(Metal::VertexRegisters),result.captured.size);
        }
        Check(queue.CurrentTick()>before,"Combined capture did not exercise upload-page recycling");queue.Finish();
        for(unsigned draw=0;draw<count;++draw) {
            Metal::VertexRegisters raw{};raw[0]={float(draw),-.5f,float(draw%17)*.25f,1};
            Metal::CapturedVertex captured{};captured.position={float(draw),draw&1 ? .5f:-.5f,-float(draw%17)*.25f,1};
            captured.color=captured.quaternion={1,1,1,1};captured.texcoord0=captured.texcoord1=captured.texcoord2={1,1,0,0};
            captured.view={1,1,1,0};captured.clip={captured.position[2],float(draw+draw%5),1,0};
            const auto* bytes=static_cast<std::byte*>(read.contents)+draw*stride;
            for(unsigned v=0;v<t.vertices;++v) {
                auto expected=values[v];expected[0]={float(draw),-.5f,float(draw%17)*.25f,1};
                Check(!std::memcmp(bytes+sizeof(raw)+sizeof(captured)+v*sizeof(expected),&expected,sizeof(expected)),
                      "Fetched upload recycling overwrote canonical inputs or immutable fetch metadata");
            }
            Check(!std::memcmp(bytes,&raw,sizeof(raw)) && !std::memcmp(bytes+sizeof(raw),&captured,sizeof(captured)),
                "Combined upload recycling overwrote raw output, conversion metadata or constants");
        }
        const auto input_history=input_runtime.SaveState(input_state);
        auto expected_input=values[0];expected_input[0]=uniforms.f[0];
        Check(!input_history.needs_reset && !std::memcmp(input_history.history.data(),&expected_input,256) &&
              !std::memcmp(input_history.history.data()+256,&expected_input,256),
              "Fetched upload recycling changed complete input history");
        const auto actual=runtime.SaveState(state);
        const auto expected_buffer=device.CreateBuffer(256,true);
        std::memcpy(expected_buffer.contents,&expected_input,256);
        const Metal::BufferSlice expected_slice{expected_buffer,0,256,expected_buffer.contents,{}};
        Metal::ShaderMachineState oracle{device};runtime.Run(oracle,program,uniforms,expected_slice);
        Check(actual==runtime.SaveState(oracle),"Fetched upload recycling changed complete retained state");
    }
    {
        constexpr unsigned draws=4096;
        Metal::Scheduler queue{device};Metal::VertexFetchRuntime input_runtime{device,queue};
        Metal::VertexInputState input_state{device};Metal::ShaderMachineRuntime runtime{device,queue},oracle{device,queue};
        Metal::ShaderMachineState state{device};Metal::SharedUploadPool sources{device,queue};
        Metal::TransferRuntime copies{device,queue};auto program=programs[1];program.output_mask=1;
        program.code[0]=arithmetic(Op::MOV,0,0);program.code[1]=word(Op::END);
        Metal::GeometryEmissionConfig config{};config.attribute_count=1;config.honor_winding=0;
        config.semantics[0]=0x03020100;
        constexpr unsigned stride=256+256+128;
        auto read=device.CreateBuffer(draws*stride,true);
        std::vector<Metal::VertexRegisters> expected_inputs(draws),expected_raw(draws);
        std::map<const void*,NSUInteger> last_offset;unsigned recycled=0;
        for(unsigned draw=0;draw<draws;++draw) {
            auto source=sources.Allocate(32*1024);source.size=256;
            const void* allocation=(__bridge const void*)source.buffer;
            if(last_offset.contains(allocation) && source.offset<=last_offset[allocation])++recycled;
            last_offset[allocation]=source.offset;
            for(unsigned r=0;r<16;++r)for(unsigned c=0;c<4;++c)
                expected_inputs[draw][r][c]=float(int(draw%31)-15)*.125f+float(r*4+c)*.03125f;
            expected_raw[draw][0]=expected_inputs[draw][0];
            std::memcpy(source.data,&expected_inputs[draw],256);
            const std::array<Metal::BufferSlice,1> borrowed{{source}};state.Reset();
            const auto result=runtime.RunFetchedVertexBatchAndCapture(input_runtime,input_state,borrowed,identity,
                state,program,{},config,1,1,base,base,{},0,1000000,true);
            copies.CopyBuffer(result.input.buffer,result.input.offset,read,draw*stride,256);
            copies.CopyBuffer(result.output.buffer,result.output.offset,read,draw*stride+256,256);
            copies.CopyBuffer(result.captured.buffer,result.captured.offset,read,draw*stride+512,128);
            if((draw+1)%16==0)queue.Submit();
        }
        queue.Finish();Check(recycled>0,"Borrowed-source test never recycled upload storage");
        const auto raw_reference=device.CreateBuffer(draws*256,true);
        std::memcpy(raw_reference.contents,expected_raw.data(),raw_reference.length);
        const Metal::BufferSlice reference_slice{raw_reference,0,raw_reference.length,raw_reference.contents,{}};
        const auto converted=oracle.CaptureVertices(reference_slice,1,config,draws);
        const auto converted_read=device.CreateBuffer(draws*128,true);
        copies.CopyBuffer(converted.buffer,converted.offset,converted_read,0,converted.size);queue.Finish();
        for(unsigned draw=0;draw<draws;++draw) {
            const auto* bytes=static_cast<const std::byte*>(read.contents)+draw*stride;
            Check(!std::memcmp(bytes,&expected_inputs[draw],256) &&
                  !std::memcmp(bytes+256,&expected_raw[draw],256) &&
                  !std::memcmp(bytes+512,static_cast<const std::byte*>(converted_read.contents)+draw*128,128),
                  "Borrowed source lease release/recycling changed complete input/raw/capture bytes");
        }
        const auto history=input_runtime.SaveState(input_state);
        Check(!history.needs_reset && !std::memcmp(history.history.data(),&expected_inputs.back(),256) &&
              !std::memcmp(history.history.data()+256,&expected_inputs.back(),256),
              "Borrowed source recycling changed complete input history");
        const auto final_input=device.CreateBuffer(256,true);std::memcpy(final_input.contents,&expected_inputs.back(),256);
        Metal::ShaderMachineState reference_state{device};
        oracle.Run(reference_state,program,{},Metal::BufferSlice{final_input,0,256,final_input.contents,{}});
        Check(runtime.SaveState(state)==oracle.SaveState(reference_state),"Borrowed source recycling changed complete VS history");
        std::cout<<"4096 borrowed-source lease/recycling draws; source pages reused "<<recycled<<" times, complete inputs/raw/capture and512/1824-byte histories match\n";
    }
    for(unsigned failure_draw:{127u,767u}) {
        Metal::Scheduler queue{device};Metal::ShaderMachineRuntime runtime{device,queue};
        Metal::ShaderMachineState state{device};Metal::VertexFetchRuntime input_runtime{device,queue};
        Metal::VertexInputState input_state{device};auto program=programs[1];
        bool rejected=false,sticky=false;unsigned queued=0;
        try {
            for(;queued<4096;++queued)
                runtime.RunFetchedVertexBatchAndCapture(input_runtime,input_state,sources,identity,state,program,{},config,t.vertices,1,base,base,{},0,
                    queued==failure_draw ? 1:1000000,true);
            queue.Finish();
        } catch(const Metal::ShaderMachineError& e) {rejected=e.Code()==6;}
        try {queue.IsComplete(0);}catch(const Metal::ShaderMachineError& e) {sticky=e.Code()==6;}
        Check(rejected && sticky && queued>failure_draw && queued<4096,
            "Combined capture lost deferred shader failure during upload recycling");
    }
    {
        // Compare the actual fetch -> VS/capture -> primitive stream path with
        // separate, completed GPU operations. Block the trial queue so a host
        // wait or premature recycling cannot masquerade as correct ordering.
        Metal::Scheduler queue{device};
        Metal::ShaderMachineRuntime runtime{device,queue};
        Metal::VertexFetchRuntime inputs_runtime{device,queue};
        Metal::GeometryRuntime primitives{device,queue},oracle_primitives{device,scheduler};
        Metal::TransferRuntime copies{device,queue};
        const auto seed=device.CreateBuffer(256,true);
        for(unsigned word=0;word<64;++word)
            static_cast<float*>(seed.contents)[word]=float(int(word%11)-5)*.125f;
        struct PrimitiveTrial {
            unsigned fixture{},vertices{},count{},width{};
            Metal::TriangleTopology topology{};
            Metal::PrimitiveSnapshot primitive_initial{},primitive_final{};
            Metal::ShaderMachineSnapshot final{};
            Metal::VertexInputSnapshot input_final{};
            std::unique_ptr<Metal::ShaderMachineState> state;
            std::unique_ptr<Metal::VertexInputState> input_state;
            std::unique_ptr<Metal::PrimitiveStream> stream;
            id<MTLBuffer> indices;
            std::array<id<MTLBuffer>,4> expected{},read{};
            Metal::ShaderMachineFetchedVertexBatchResult retained;
            Metal::AssembledVertices assembled;
        };
        std::vector<PrimitiveTrial> pending;
        for(unsigned i=0;i<trials.size();++i) {
            const auto& fixture=trials[i];
            PrimitiveTrial test;test.fixture=i;test.width=fixture.index_size;
            test.count=std::array<unsigned,8>{1,2,3,6,8,9,65,257}[i%8];
            // The profiled hot path is a reset dependent six-vertex draw.
            // Do not let the count/index-width matrix miss that combination.
            if(i==36) {
                Check(fixture.program==3 && !test.width && fixture.initial.needs_reset,
                    "Six-vertex batching fixture lost its reset/dependency class");
                test.count=6;
            }
            test.vertices=test.width ? 70:test.count;
            test.topology=Metal::TriangleTopology(i%3);
            test.indices=device.CreateBuffer(3+std::max(1u,test.width)*test.count,true);
            auto* index_bytes=static_cast<unsigned char*>(test.indices.contents)+3;
            for(unsigned e=0;e<test.count;++e) for(unsigned b=0;b<test.width;++b)
                index_bytes[e*test.width+b]=(base+e%test.vertices)>>(8*b);
            const Metal::BufferSlice input{source,16,test.vertices*256,values,{}},
                indices{test.indices,3,test.width*test.count,index_bytes,{}};
            const std::array<Metal::BufferSlice,1> sources{{input}};
            Metal::VertexInputState input_oracle{device};
            reference_fetch.LoadState(input_oracle,fixture.input_initial);
            const auto canonical=reference_fetch.FetchPica(input_oracle,sources,fixture.layout,
                test.vertices,test.count,base,base,test.width ? indices:Metal::BufferSlice{},test.width);
            test.input_final=reference_fetch.SaveState(input_oracle);
            Metal::ShaderMachineState oracle{device};reference.LoadState(oracle,fixture.initial);
            const auto raw=reference.RunVertexBatch(oracle,programs[fixture.program],{},canonical,
                test.vertices,test.count,base,base,test.width ? indices:Metal::BufferSlice{},test.width);
            const auto captured=reference.CaptureVertices(raw,7,fixture.config,test.count);
            test.final=reference.SaveState(oracle);
            Metal::PrimitiveStream stream{device};
            if(!fixture.initial.needs_reset)
                oracle_primitives.Append(stream,{seed,0,256,nullptr,{}},2,test.topology,
                    1+(i&1),0,0,{},0,false);
            test.primitive_initial=oracle_primitives.SaveStream(stream);
            const auto assembled=oracle_primitives.Append(stream,captured,test.count,test.topology,
                test.count,0,0,{},0,(i&1)!=0);
            test.primitive_final=oracle_primitives.SaveStream(stream);
            const std::array<Metal::BufferSlice,4> outputs{canonical,raw,captured,assembled.storage};
            for(unsigned slot=0;slot<outputs.size();++slot) {
                // Append always returns storage, including a partial triangle.
                if(slot==3 && !assembled.vertex_count) continue;
                test.expected[slot]=device.CreateBuffer(outputs[slot].size,true);
                transfer.CopyBuffer(outputs[slot].buffer,outputs[slot].offset,test.expected[slot],0,outputs[slot].size);
                test.read[slot]=device.CreateBuffer(outputs[slot].size+32,true);
                std::memset(test.read[slot].contents,0xA7,test.read[slot].length);
            }
            test.state=std::make_unique<Metal::ShaderMachineState>(device);
            runtime.LoadState(*test.state,fixture.initial);
            test.input_state=std::make_unique<Metal::VertexInputState>(device);
            inputs_runtime.LoadState(*test.input_state,fixture.input_initial);
            test.stream=std::make_unique<Metal::PrimitiveStream>(device);
            primitives.LoadStream(*test.stream,test.primitive_initial);
            pending.push_back(std::move(test));
        }
        scheduler.Finish();queue.Finish();
        auto gate=[device.Handle() newSharedEvent];
        struct ReleaseEvent {id<MTLSharedEvent> event;~ReleaseEvent(){event.signaledValue=1;}} release{gate};
        Metal::SchedulerTestAccess::BlockQueue(queue,gate);
        const auto tick=queue.CurrentTick();unsigned one_encoder=0;
        for(auto& test:pending) {
            const auto& fixture=trials[test.fixture];
            const Metal::BufferSlice input{source,16,test.vertices*256,values,{}},
                indices{test.indices,3,test.width*test.count,nullptr,{}};
            const std::array<Metal::BufferSlice,1> sources{{input}};
            {
                Metal::Scheduler::BufferComputeScope buffers{queue};
                Metal::Scheduler::ComputeBatchScope batch{queue};
                const auto count=queue.EncoderCount();
                test.retained=runtime.RunFetchedVertexBatchAndCapture(inputs_runtime,*test.input_state,
                    sources,fixture.layout,*test.state,programs[fixture.program],{},fixture.config,
                    test.vertices,test.count,base,base,test.width ? indices:Metal::BufferSlice{},test.width,1000000,true);
                Check(queue.EncoderCount()==count+1,"Fetched VS/capture unexpectedly split its encoder");
                test.assembled=primitives.Append(*test.stream,test.retained.captured,test.count,test.topology,
                    test.count,0,0,{},0,(test.fixture&1)!=0);
                Check(queue.EncoderCount()==count+1,"Fetched VS/capture and primitive append used separate encoders");
                ++one_encoder;
                // Explicit submission must still close and replace a live encoder.
                if(test.fixture==13) queue.SubmitPending();
            }
            const std::array<Metal::BufferSlice,4> outputs{
                test.retained.input,test.retained.output,test.retained.captured,test.assembled.storage};
            for(unsigned slot=0;slot<outputs.size();++slot) if(test.read[slot])
                copies.CopyBuffer(outputs[slot].buffer,outputs[slot].offset,test.read[slot],16,outputs[slot].size);
        }
        queue.SubmitPending();
        Check(queue.CurrentTick()>tick && !queue.IsComplete(tick),"Native event did not hold batched VS/append work");
        gate.signaledValue=1;queue.Finish();
        for(auto& test:pending) {
            for(unsigned slot=0;slot<test.read.size();++slot) if(test.read[slot]) {
                const auto size=test.expected[slot].length;
                Check(!std::memcmp(static_cast<std::byte*>(test.read[slot].contents)+16,test.expected[slot].contents,size),
                    "Batched fetched VS/primitive append changed complete output bytes");
                for(unsigned byte=0;byte<16;++byte) Check(
                    static_cast<unsigned char*>(test.read[slot].contents)[byte]==0xA7 &&
                    static_cast<unsigned char*>(test.read[slot].contents)[16+size+byte]==0xA7,
                    "Batched fetched VS/primitive append damaged guards");
            }
            Check(runtime.SaveState(*test.state)==test.final,"Batched append changed full1824-byte VS state");
            const auto fetched=inputs_runtime.SaveState(*test.input_state);
            Check(fetched.needs_reset==test.input_final.needs_reset && fetched.history==test.input_final.history,
                "Batched append changed full512-byte fetch state");
            const auto primitive=primitives.SaveStream(*test.stream);
            Check(primitive.history==test.primitive_final.history && primitive.pending==test.primitive_final.pending &&
                primitive.topology==test.primitive_final.topology && primitive.needs_reset==test.primitive_final.needs_reset,
                "Batched append changed full272-byte primitive state/metadata");
        }
        std::cout<<one_encoder<<" queue-blocked fetch/VS/capture/append cases use exactly one compute encoder: "
            "all5 execution classes, list/strip/fan, reset/inherited banks, all index widths, counts1/2/3/6/8/9/65/257, "
            "explicit submissions, guarded complete outputs and1824/512/272-byte states match unbatched GPU operations\n";
    }
    std::cout<<"120 fetched VS/capture batches (60 shared input/index allocations),135 GPU index errors across FIFO/SIMD boundaries and all index widths,12 invalid request cases,4096 early program rejects,1 foreign queue reject,5 diagnostic records; mixed formats, retained/fixed inputs, register aliases, full512-byte input history and canonical guards: all5 execution classes, reset/both inherited banks, all index widths, FIFO eviction, delayed owned slices, immutable conversion metadata, guards/full1824-byte states;8 metadata rejects,2 atomic shader failures and4096 upload-recycling draws plus2 sticky deferred failures passed\n";
}


void CheckDeferredBatchCompiler(Metal::Device& device) {
    Metal::Scheduler scheduler{device};Metal::ShaderMachineRuntime oracle{device,scheduler};
    auto actual=std::make_unique<Metal::ShaderMachineRuntime>(device,scheduler);
    Metal::ShaderMachineState reference{device},state{device};
    Metal::TransferRuntime transfer{device,scheduler};
    constexpr unsigned count=129;
    auto input=device.CreateBuffer(count*sizeof(Metal::VertexRegisters),true);
    auto indices=device.CreateBuffer(count*4,true);
    for(unsigned v=0;v<count;++v) {
        Metal::VertexRegisters registers{};
        for(unsigned r=0;r<16;++r)registers[r]={float(v)*.25f,float(r)*.5f,-.75f,1};
        std::memcpy(static_cast<std::byte*>(input.contents)+v*sizeof(registers),registers.data(),sizeof(registers));
        static_cast<unsigned*>(indices.contents)[v]=(v*17u)%count;
    }
    Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=3;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    program.code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,0);
    program.code[1]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,1,32);
    program.code[2]=unsigned(nihstro::OpCode::Id::END)<<26;
    Metal::ShaderMachineUniforms uniforms{};uniforms.f[0]={-.25f,.5f,2,-4};
    std::array<Metal::VertexRegisters,count> expected;
    for(unsigned v=0;v<count;++v) {
        const auto index=static_cast<const unsigned*>(indices.contents)[v];
        const auto output=oracle.Run(reference,program,uniforms,
            {input,index*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}},0xffff);
        std::memcpy(expected[v].data(),output.output.data,sizeof(Metal::VertexRegisters));
    }
    const auto expected_state=oracle.SaveState(reference);
    auto read=device.CreateBuffer(sizeof(expected)+32,true);std::memset(read.contents,0xB4,read.length);
    std::promise<void> release;
    struct Release {std::promise<void>& promise;bool open{};void Open(){if(!open){promise.set_value();open=true;}}~Release(){Open();}} gate{release};
    const auto wait=release.get_future().share();
    std::array<std::promise<void>,2> started;std::array<std::shared_future<void>,2> blockers;
    for(unsigned i=0;i<2;++i)blockers[i]=device.DeferCompilation([&,i,wait]{started[i].set_value();wait.wait();});
    for(auto& start:started)start.get_future().wait();
    const auto tick=scheduler.CurrentTick(),before=scheduler.WorkerStats().queued_dispatches;
    const auto output=actual->RunVertexBatch(state,program,uniforms,{input,0,input.length,nullptr,{}},
        count,count,0,0,{indices,0,indices.length,nullptr,{}},4,1000000,true);
    Check(scheduler.CurrentTick()==tick && scheduler.WorkerStats().queued_dispatches==before+2 && !output.data,
        "First independent vertex batch waited for native compilation or omitted finalization");
    program.code.fill(UINT32_MAX);program.swizzle.fill(UINT32_MAX);uniforms.f[0]={99,98,97,96};
    actual.reset();input=nil;indices=nil;
    gate.Open();transfer.CopyBuffer(output.buffer,output.offset,read,16,sizeof(expected));scheduler.Finish();
    Check(!std::memcmp(static_cast<const std::byte*>(read.contents)+16,expected.data(),sizeof(expected)) &&
        oracle.SaveState(state)==expected_state,"Deferred batch publication borrowed destroyed runtime/program/uniforms or changed final SIMD state");
    for(unsigned i=0;i<16;++i)Check(static_cast<const std::byte*>(read.contents)[i]==std::byte{0xB4} &&
        static_cast<const std::byte*>(read.contents)[16+sizeof(expected)+i]==std::byte{0xB4},"Deferred batch changed output guards");
    std::cout<<"First independent indexed batch compiler-gate: owner returned with compiler workers held; runtime destruction, changed program/uniforms, released inputs preserve129 complete512-byte outputs/guards and1824-byte state against generic scalar GPU oracle; native SIMD finalization passed\n";
}

void CheckDeferredScalarVertices(Metal::Device& device,Metal::Scheduler& scheduler) {
    Metal::ShaderMachineRuntime queued{device,scheduler},oracle{device,scheduler};
    Metal::ShaderMachineState state{device},reference{device};
    constexpr unsigned count=64,rounds=4;
    auto source=device.CreateBuffer(count*sizeof(Metal::VertexRegisters)+32,true);
    auto* bytes=static_cast<std::byte*>(source.contents);
    std::memset(bytes,0xA6,source.length);
    for(unsigned v=0;v<count;++v) {
        Metal::VertexRegisters value{};
        for(unsigned reg=0;reg<16;++reg) value[reg]={float(v+1)*.125f,float(reg)*.25f,-.5f,.75f};
        std::memcpy(bytes+16+v*sizeof(value),value.data(),sizeof(value));
    }
    Metal::ShaderMachineProgram program{};
    program.allow_emission=false;program.output_mask=15;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    // A genuine cross-invocation dependency and partial output bank updates.
    program.code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::ADD,16,16,0);
    program.code[1]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,16);
    program.code[2]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,1,32);
    program.code[3]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,3,1);
    program.code[4]=unsigned(nihstro::OpCode::Id::END)<<26;
    {
        const auto input=Metal::BufferSlice{source,16,sizeof(Metal::VertexRegisters),nullptr,{}};
        const auto expected=oracle.Run(reference,program,{},input,0xffff);
        Metal::VertexRegisters expected_bytes{};
        std::memcpy(expected_bytes.data(),expected.output.data,sizeof(expected_bytes));
        const auto expected_state=oracle.SaveState(reference);
        Metal::TransferRuntime transfer{device,scheduler};
        auto read=device.CreateBuffer(sizeof(Metal::VertexRegisters),true);
        std::promise<void> release;
        struct Release {std::promise<void>& promise;bool open{};void Open(){if(!open){promise.set_value();open=true;}}~Release(){Open();}} gate{release};
        const auto wait=release.get_future().share();
        std::array<std::promise<void>,2> started;
        std::array<std::shared_future<void>,2> blockers;
        for(unsigned i=0;i<2;++i)blockers[i]=device.DeferCompilation([&,i,wait]{started[i].set_value();wait.wait();});
        for(auto& start:started)start.get_future().wait();
        const auto tick=scheduler.CurrentTick(),before=scheduler.WorkerStats().queued_dispatches;
        const auto output=queued.RunVertexDeferred(state,program,{},input,0xffff);
        Check(scheduler.CurrentTick()==tick && scheduler.WorkerStats().queued_dispatches==before+1 && !output.data,
            "First scalar program compiled synchronously, submitted early or exposed uncompleted data");
        const auto restore=program;program.code.fill(UINT32_MAX);program.swizzle.fill(UINT32_MAX);
        gate.Open();
        transfer.CopyBuffer(output.buffer,output.offset,read,0,read.length);scheduler.Finish();
        Check(!std::memcmp(read.contents,expected_bytes.data(),sizeof(expected_bytes)) &&
            oracle.SaveState(state)==expected_state,"Deferred scalar compiler borrowed mutable program banks or changed complete state");
        program=restore;
        std::cout<<"First scalar VS compiler-gate: owner returned with compiler workers held; changed borrowed banks preserve complete512-byte output and1824-byte state against generic GPU oracle\n";
    }
    struct Expected { Metal::VertexRegisters output; Metal::ShaderMachineUniforms uniforms; unsigned mask; };
    std::vector<Metal::BufferSlice> retained;
    std::vector<Metal::VertexRegisters> retained_values;
    for(unsigned round=0;round<rounds;++round) {
        const auto initial=oracle.SaveState(reference);
        queued.LoadState(state,initial);
        std::array<Expected,count> expected{};
        for(unsigned v=0;v<count;++v) {
            auto& value=expected[v];value.uniforms.f[0]={float(round*count+v)*.25f,-0.f,1.f,-1.f};
            value.mask=v%4==0 ? 0:v%4==1 ? 1:v%4==2 ? 3:0xffff;
            if(v%17==0 && (round==0 || v!=0)) reference.Reset();
            const auto output=oracle.Run(reference,program,value.uniforms,
                {source,16+v*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}},value.mask);
            std::memcpy(value.output.data(),output.output.data,sizeof(value.output));
        }
        const auto reference_snapshot=oracle.SaveState(reference);
        const auto tick=scheduler.CurrentTick();
        for(unsigned v=0;v<count;++v) {
            if(v%17==0 && (round==0 || v!=0)) state.Reset();
            const auto output=queued.RunVertexDeferred(state,program,expected[v].uniforms,
                {source,16+v*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}},expected[v].mask);
            Check(output.buffer && !output.data,"Queued VS exposed uncompleted CPU data");
            retained.push_back(output);retained_values.push_back(expected[v].output);
        }
        Check(scheduler.CurrentTick()==tick,"Scalar VS queue introduced per-vertex submission/wait");
        Check(queued.SaveState(state)==reference_snapshot,"Queued scalar VS changed complete1824-byte state/reset history");
        // Re-check all earlier held outputs after subsequent states and resets.
        auto readback=device.CreateBuffer(retained.size()*sizeof(Metal::VertexRegisters)+32,true);
        std::memset(readback.contents,0xDA,readback.length);
        for(unsigned v=0;v<retained.size();++v)
            Metal::TransferRuntime{device,scheduler}.CopyBuffer(retained[v].buffer,retained[v].offset,
                readback,16+v*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters));
        scheduler.Finish();
        const auto* actual=static_cast<const std::byte*>(readback.contents);
        for(unsigned v=0;v<retained.size();++v)
            Check(!std::memcmp(actual+16+v*sizeof(Metal::VertexRegisters),retained_values[v].data(),sizeof(Metal::VertexRegisters)),
                "Queued scalar VS changed a complete output or overwrote a retained earlier range");
        for(unsigned b=0;b<16;++b)
            Check(actual[b]==std::byte{0xDA} && actual[readback.length-16+b]==std::byte{0xDA},"Queued scalar VS readback damaged guards");
    }
    auto geometry_program=program;geometry_program.allow_emission=true;
    bool rejected=false;const auto tick=scheduler.CurrentTick();
    try {queued.RunVertexDeferred(state,geometry_program,{},{});}catch(const std::invalid_argument&){rejected=true;}
    Check(rejected && scheduler.CurrentTick()==tick,"Queued scalar VS accepted a geometry/emission program");
    for(unsigned failure=0;failure<3;++failure) {
        Metal::Scheduler queue{device};Metal::ShaderMachineRuntime machine{device,queue};Metal::ShaderMachineState failed{device};
        auto invalid=program;
        if(failure==0) invalid.code[0]=0x10u<<26;
        if(failure==1) invalid.code[0]=unsigned(nihstro::OpCode::Id::EMIT)<<26;
        const auto before=queue.CurrentTick();
        machine.RunVertexDeferred(failed,invalid,{}, {source,16,sizeof(Metal::VertexRegisters),nullptr,{}},0xffff,failure==2 ? 1:1000000);
        // Later successful dispatches cannot erase the earlier status record.
        for(unsigned v=0;v<8;++v) machine.RunVertexDeferred(failed,program,{}, {source,16,sizeof(Metal::VertexRegisters),nullptr,{}});
        Check(queue.CurrentTick()==before,"Deferred scalar error caused a premature submit");
        bool caught=false,sticky=false;
        const unsigned code=failure==2 ? 6:2;
        try {queue.Finish();}catch(const Metal::ShaderMachineError& e){caught=e.Code()==code;}
        try {queue.IsComplete(0);}catch(const Metal::ShaderMachineError& e){sticky=e.Code()==code;}
        Check(caught && sticky,"Queued scalar shader failure was lost or failed to remain sticky");
    }
    std::cout<<"256 queued scalar VS invocations: exact synchronous GPU oracle outputs/full1824-byte state, inherited temporaries, reset/masks/uniforms, retained ranges/guards, no per-vertex submit, GS rejection and three sticky GPU failures passed\n";
}

void CheckStructuredOrderedBoundaries(Metal::Device& device, Metal::Scheduler& scheduler,
                                    Metal::ShaderMachineRuntime& machine) {
    using Op=nihstro::OpCode::Id;
    auto input=device.CreateBuffer(sizeof(Metal::VertexRegisters),true);
    std::memset(input.contents,0,input.length);
    for(unsigned expanded:{0u,1u}) {
        Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=1;
        program.swizzle[0]=Metal::Test::IdentitySwizzle;
        const auto call=[](unsigned target,unsigned length) {return (unsigned(Op::CALL)<<26)|(target<<10)|length;};
        if(!expanded) {
            program.entry=4094;program.code[4094]=call(0,1);
            program.code[0]=Metal::Test::Arithmetic(Op::RCP,16,16);
            program.code[4095]=unsigned(Op::NOP)<<26;
        } else {
            // Only 52 distinct PCs, but >4K instructions if every call is inlined.
            // NOP tails keep calls away from simultaneous return boundaries.
            program.code[0]=call(32,17);program.code[1]=unsigned(Op::END)<<26;
            for(unsigned i=0;i<16;++i) {
                program.code[32+i]=call(64,17);program.code[64+i]=call(96,16);
                program.code[96+i]=Metal::Test::Arithmetic(Op::RCP,16,16);
            }
            program.code[48]=program.code[80]=unsigned(Op::NOP)<<26;
        }
        std::cout<<"Structured boundary fixture "<<expanded<<": control="
            <<Metal::ShaderMachineRuntime::HasNativeResetControlFlow(program)
            <<" reachable="<<Metal::ShaderMachineRuntime::ReachableVertexInstructions(program).count()
            <<" execution="<<unsigned(Metal::ShaderMachineRuntime::AnalyzeVertices(program))<<'\n';
        Check(Metal::ShaderMachineRuntime::HasNativeResetControlFlow(program) &&
            Metal::ShaderMachineRuntime::ReachableVertexInstructions(program).count()<=512 &&
            Metal::ShaderMachineRuntime::AnalyzeVertices(program)==Metal::ShaderMachineRuntime::VertexExecution::Ordered,
            "Structured boundary fixture lost its ordered, bounded native-control domain");
        for(unsigned inherited:{0u,1u}) for(unsigned limit:{1u,2u,3u,4u,1000000u}) {
            if(expanded && limit!=1000000) continue;
            Metal::ShaderMachineSnapshot initial{};initial.needs_reset=!inherited;
            if(inherited) {
                for(unsigned lane=0;lane<448;++lane) {
                    const std::uint32_t bits=0x3f000000u;std::memcpy(initial.history.data()+lane*4,&bits,4);
                }
                const std::uint32_t bank=1;std::memcpy(initial.history.data()+1816,&bank,4);
            }
            Metal::ShaderMachineState scalar{device},batch{device};
            machine.LoadState(scalar,initial);machine.LoadState(batch,initial);
            Metal::VertexRegisters expected{};std::string reference_error,actual_error;
            try {
                const auto result=machine.Run(scalar,program,{},
                    {input,0,input.length,input.contents,{}},0xffff,192,limit);
                std::memcpy(&expected,result.output.data,sizeof(expected));
            } catch(const Metal::ShaderMachineError& error) {reference_error=error.what();}
            try {
                const auto result=machine.RunVertexBatch(batch,program,{},
                    {input,0,input.length,input.contents,{}},1,1,0,0,{},0,limit);
                auto read=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                scheduler.Finish();
                Check(read.length==sizeof(expected) && std::memcmp(read.contents,&expected,sizeof(expected))==0,
                    "Expanded call-source budget changed the scalar GPU output");
            } catch(const Metal::ShaderMachineError& error) {actual_error=error.what();}
            Check(reference_error==actual_error && machine.SaveState(batch)==machine.SaveState(scalar),
                "Structured call end/PC4096 or source-budget path changed native error code, PC or full state");
            if(!reference_error.empty()) Check(machine.SaveState(batch)==initial,"Call boundary error committed state");
        }
    }
    std::cout<<"Structured ordered calls: PC4096 versus instruction-limit priority, inherited bank1/reset, expanded-source budget and exact scalar GPU state/output passed\n";
}

void CheckRecordedOrderedBatches(Metal::Device& device, Metal::Scheduler& scheduler,
    Metal::ShaderMachineRuntime& machine, const Metal::ShaderMachineProgram& program,
    const Metal::ShaderMachineUniforms& uniforms) {
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
        Metal::ShaderMachineRuntime::VertexExecution::Ordered,"Recorded regression must exercise ordered execution");
    for(unsigned count:{65u,129u,372u,2355u}) for(unsigned initial_case:{0u,1u,2u}) {
        const unsigned vertices=count==2355 ? 911:90,base=7;
        auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
        auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
        for(unsigned vertex=0;vertex<vertices;++vertex) for(unsigned reg=0;reg<16;++reg)
            values[vertex][reg]={float((vertex+reg)%7)*.125f,float((vertex+reg)%5)*.25f,
                float((vertex+reg)%3)*.25f,1};
        auto indices=device.CreateBuffer(count*2,true);
        std::vector<unsigned> ids(count);
        for(unsigned element=0;element<count;++element) {
            ids[element]=base+((element/3)*7)%vertices;
            static_cast<std::uint16_t*>(indices.contents)[element]=ids[element];
        }
        Metal::ShaderMachineSnapshot initial{};initial.needs_reset=initial_case==0;
        if(!initial.needs_reset) {
            for(unsigned lane=0;lane<448;++lane) {
                const auto bits=0x3f000000u+(lane%32)*257u;
                std::memcpy(initial.history.data()+lane*4,&bits,4);
            }
            const std::uint32_t bank=initial_case==2;
            std::memcpy(initial.history.data()+1816,&bank,4);
        }
        Metal::ShaderMachineState scalar{device},batch{device};
        machine.LoadState(scalar,initial);machine.LoadState(batch,initial);
        std::vector<Metal::VertexRegisters> expected(count);
        std::array<std::pair<unsigned,unsigned>,64> fifo{};
        unsigned cached_count{},replacement{};
        for(unsigned element=0;element<count;++element) {
            unsigned hit=~0u;
            for(unsigned slot=0;slot<cached_count;++slot)
                if(fifo[slot].first==ids[element]) hit=fifo[slot].second;
            if(hit!=~0u) {expected[element]=expected[hit];continue;}
            const auto output=machine.Run(scalar,program,uniforms,
                {input,(ids[element]-base)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
            std::memcpy(&expected[element],output.output.data,sizeof(Metal::VertexRegisters));
            fifo[replacement]={ids[element],element};replacement=(replacement+1)&63u;
            cached_count=std::min(cached_count+1,64u);
        }
        const auto begin=std::chrono::steady_clock::now();
        const auto output=machine.RunVertexBatch(batch,program,uniforms,
            {input,0,input.length,input.contents,{}},vertices,count,base,base,
            {indices,0,indices.length,indices.contents,{}},2);
        auto readback=device.CreateBuffer(output.size,true);
        Metal::TransferRuntime{device,scheduler}.CopyBuffer(output.buffer,output.offset,readback,0,readback.length);
        scheduler.Finish();
        const auto host_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        Check(readback.length==expected.size()*sizeof(Metal::VertexRegisters) &&
            std::memcmp(readback.contents,expected.data(),readback.length)==0 &&
            machine.SaveState(batch)==machine.SaveState(scalar),
            "Recorded ordered batch changed raw outputs, FIFO behavior or complete inherited GPU state");
        std::cout<<"Recorded ordered batch: elements="<<count<<" initial="<<initial_case
            <<" exact outputs/full state passed; diagnostic host_ms="<<host_ms<<'\n';
        if(count==372 && initial_case==0) {
            // Same shader, uniforms, FIFO and inputs, alternating the proven
            // full-budget pipeline with a semantically sufficient guarded
            // budget. Warm both outside timing; compare complete final state.
            for(unsigned limit:{999999u,1000000u}) {
                batch.Reset();machine.RunVertexBatch(batch,program,uniforms,
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,
                    {indices,0,indices.length,indices.contents,{}},2,limit,true);
                scheduler.Finish();
            }
            for(unsigned round=0;round<3;++round)for(unsigned variant=0;variant<2;++variant) {
                const auto limit=((round+variant)&1) ? 1000000u:999999u;
                const auto start=std::chrono::steady_clock::now();
                for(unsigned repeat=0;repeat<16;++repeat) {
                    batch.Reset();machine.RunVertexBatch(batch,program,uniforms,
                        {input,0,input.length,input.contents,{}},vertices,count,base,base,
                        {indices,0,indices.length,indices.contents,{}},2,limit,true);
                }
                scheduler.Finish();
                const auto ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-start).count();
                Check(machine.SaveState(batch)==machine.SaveState(scalar),
                    "Alternating ordered limit benchmark changed complete final state");
                std::cout<<"ORDERED_LIMIT_BENCH elements=372 round="<<round<<" limit="<<limit
                    <<" draws=16 host_ms="<<ms<<'\n';
            }
        }
        if(count==65 && initial_case==0) {
            const auto bound=Metal::ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms);
            if(bound) {
                // The full-budget proof-selected kernel must agree with the
                // ordinary guarded kernel at its conservative finite bound.
                machine.LoadState(batch,initial);
                const auto limited=machine.RunVertexBatch(batch,program,uniforms,
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,
                    {indices,0,indices.length,indices.contents,{}},2,*bound);
                auto bounded_read=device.CreateBuffer(limited.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(limited.buffer,limited.offset,
                    bounded_read,0,bounded_read.length);
                scheduler.Finish();
                Check(bounded_read.length==expected.size()*sizeof(Metal::VertexRegisters) &&
                    !std::memcmp(bounded_read.contents,expected.data(),bounded_read.length) &&
                    machine.SaveState(batch)==machine.SaveState(scalar),
                    "Proof-selected ordered kernel differs at its finite guarded instruction bound");
            }
        }
        if(count==65) for(unsigned limit:{1u,2u,8u,16u,32u}) {
            machine.LoadState(scalar,initial);machine.LoadState(batch,initial);
            std::string expected_error,actual_error;
            try {
                (void)machine.Run(scalar,program,uniforms,
                    {input,(ids[0]-base)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}},
                    0xffff,192,limit);
            } catch(const Metal::ShaderMachineError& error) {expected_error=error.what();}
            try {
                (void)machine.RunVertexBatch(batch,program,uniforms,
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,
                    {indices,0,indices.length,indices.contents,{}},2,limit);
            } catch(const Metal::ShaderMachineError& error) {actual_error=error.what();}
            Check(!expected_error.empty() && expected_error==actual_error &&
                machine.SaveState(batch)==initial && machine.SaveState(scalar)==initial,
                "Recorded ordered call changed instruction-limit code/PC or error rollback");
        }
    }
}

int main(int argc,char** argv) {
    @autoreleasepool { try {
        std::cout.setf(std::ios::unitbuf);
        if(argc>2 && (std::string_view(argv[1])=="--analyze-recorded" ||
                     std::string_view(argv[1])=="--recorded-ordered-only")) {
            const bool execute=std::string_view(argv[1])=="--recorded-ordered-only";
            std::unique_ptr<Metal::Device> device;
            std::unique_ptr<Metal::Scheduler> scheduler;
            std::unique_ptr<Metal::ShaderMachineRuntime> machine;
            if(execute) {
                device=std::make_unique<Metal::Device>();scheduler=std::make_unique<Metal::Scheduler>(*device);
                machine=std::make_unique<Metal::ShaderMachineRuntime>(*device,*scheduler);
                CheckStructuredOrderedBoundaries(*device,*scheduler,*machine);
            }
            for(int argument=2;argument<argc;++argument) {
                std::ifstream stream(argv[argument],std::ios::binary);
                std::array<std::uint32_t,12> header{};
                Metal::ShaderMachineProgram program;Metal::ShaderMachineUniforms uniforms;
                stream.read(reinterpret_cast<char*>(header.data()),sizeof(header));
                Check(stream && header[0]==0x34544c4d && header[1]==1 && header[2]==4096 &&
                    header[3]==4096 && header[4]==sizeof(uniforms),"Invalid recorded shader header");
                program.entry=header[5];program.output_mask=header[6];program.allow_emission=header[7];
                stream.read(reinterpret_cast<char*>(program.code.data()),sizeof(program.code));
                stream.read(reinterpret_cast<char*>(program.swizzle.data()),sizeof(program.swizzle));
                stream.read(reinterpret_cast<char*>(&uniforms),sizeof(uniforms));
                Check(stream && stream.peek()==std::char_traits<char>::eof(),"Truncated/trailing recorded shader");
                std::uint32_t unproven{},passes{};
                const auto execution=Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms,&unproven,&passes);
                std::cout<<argv[argument]<<" b="<<uniforms.b<<" execution="<<unsigned(execution)
                    <<" passes="<<passes<<" last_pc="<<unproven
                    <<" reachable="<<Metal::ShaderMachineRuntime::ReachableVertexInstructions(program).count()
                    <<" structured_control="<<Metal::ShaderMachineRuntime::HasNativeResetControlFlow(program)
                    <<" instruction_bound="<<Metal::ShaderMachineRuntime::NativeVertexInstructionBound(program,uniforms).value_or(0)<<'\n';
                if(execute) CheckRecordedOrderedBatches(*device,*scheduler,*machine,program,uniforms);
            }
            return 0;
        }
        CheckNativeInstructionBounds();
        if(argc==2 && std::string_view(argv[1])=="--instruction-bound-only")return 0;
        Metal::Device device;
        CheckDeferredBatchCompiler(device);
        if(argc==2 && std::string_view(argv[1])=="--batch-compiler-only")return 0;
        if(argc==2 && std::string_view(argv[1])=="--first-native-proof-only") {
            CheckFirstNativeProofReuse(device);return 0;
        }
        if(argc>1 && std::string_view(argv[1])=="--small-batch-benchmark") {
            CheckSmallRecordedBatches(device,argc,argv);return 0;
        }
        if(argc==2 && std::string_view(argv[1])=="--owned-state-only") {
            CheckOwnedDeferredState(device);CheckBoundedOwnedControl(device);CheckOwnedConditionalPrefix(device);return 0;
        }
        if(argc==2 && std::string_view(argv[1])=="--conditional-prefix-only") {CheckOwnedConditionalPrefix(device);return 0;}
        CheckBoundedOwnedControl(device);
        CheckOwnedDeferredState(device);
        CheckOwnedConditionalPrefix(device);
        CheckTriggeredWarmProof(device);
        CheckTriggeredWarmProof(device,true);
        if(argc==2 && std::string_view(argv[1])=="--profile-window-only")return 0;
        Metal::Scheduler scheduler{device};
        Metal::ShaderMachineRuntime machine{device,scheduler};
        CheckDeferredScalarVertices(device,scheduler);
        if(argc==2 && std::string_view(argv[1])=="--deferred-scalar-only") return 0;
        CheckFetchedVertexCapture(device,scheduler,machine);
        if(argc==2 && std::string_view(argv[1])=="--fetched-capture-only") return 0;
        CheckCombinedVertexCapture(device,scheduler,machine);
        if(argc==2 && std::string_view(argv[1])=="--combined-capture-only") return 0;
        CheckResetBankPipelines(device,scheduler,machine);
        if(argc==2 && std::string_view(argv[1])=="--reset-bank-only") return 0;
        CheckBorrowedProgramSnapshots(device,scheduler,machine);
        if(argc==2 && std::string_view(argv[1])=="--program-view-only") return 0;
        CheckFirstNativeProofReuse(device);
        CheckControlProofCache(device,scheduler);
        if(argc==2 && std::string_view(argv[1])=="--control-cache-only") return 0;
        CheckScalarOutputRetention(device,scheduler,machine);
        if(argc==2 && std::string_view(argv[1])=="--scalar-output-only") return 0;
        CheckPrefixBoundaries(device,scheduler);
        CheckMergeBoundaries(device,scheduler);
        if(argc==2 && std::string_view(argv[1])=="--prefix-only") return 0;
        {
            // One program switches between 129 proven recurrence masks while
            // earlier pipelines remain queued. Include both mask words and
            // return to the first mask after bounded pipeline-cache eviction.
            Metal::ShaderMachineRuntime reference{device,scheduler};
            Metal::ShaderMachineProgram rp{};
            rp.code.fill(0);rp.output_mask=1;rp.allow_emission=false;
            rp.swizzle[0]=Metal::Test::IdentitySwizzle;
            for(unsigned component=0;component<4;++component)
                rp.swizzle[1+component]=(Metal::Test::IdentitySwizzle&~15u)|(8u>>component);
            rp.code[0]=(0x13u<<26);unsigned pc=1;
            for(unsigned reg=0;reg<16;++reg) {
                rp.code[pc]=(0x27u<<26)|(reg<<22)|((pc+2u)<<10);++pc;
                rp.code[pc++]=(16u+reg)<<21|((16u+reg)<<12)|(1u+reg%4);
            }
            rp.code[pc]=0x22u<<26;
            Metal::ShaderMachineProgram tp{};
            tp.code=rp.code;tp.swizzle=rp.swizzle;tp.entry=rp.entry;tp.output_mask=rp.output_mask;tp.allow_emission=false;
            constexpr unsigned count=5,rounds=130;
            auto inputs=device.CreateBuffer(3*sizeof(Metal::VertexRegisters),true);
            auto* values=static_cast<Metal::VertexRegisters*>(inputs.contents);
            for(unsigned vertex=0;vertex<3;++vertex)for(unsigned reg=0;reg<16;++reg)
                values[vertex][reg]={float(vertex+1)*.125f,.25f,-.5f,.75f};
            auto indices=device.CreateBuffer(count,true);
            for(unsigned i=0;i<count;++i)static_cast<unsigned char*>(indices.contents)[i]=i%3;
            const Metal::BufferSlice input{inputs,0,inputs.length,inputs.contents,{}},ix{indices,0,indices.length,indices.contents,{}};
            Metal::ShaderMachineSnapshot initial{};initial.needs_reset=false;
            const std::array<unsigned,5> flags{0,1,0,1,15};
            std::memcpy(initial.history.data()+451*4,flags.data(),20);
            Metal::ShaderMachineSnapshot trial_initial{};trial_initial.history=initial.history;trial_initial.needs_reset=false;
            Metal::ShaderMachineState baseline_state{device};Metal::ShaderMachineState trial_state{device};
            reference.LoadState(baseline_state,initial);machine.LoadState(trial_state,trial_initial);scheduler.Finish();
            auto expected=device.CreateBuffer(rounds*count*sizeof(Metal::VertexRegisters),true);
            auto actual=device.CreateBuffer(expected.length,true);
            Metal::TransferRuntime transfer{device,scheduler};
            std::set<std::array<unsigned,2>> masks;
            // Build an independent ordered scalar GPU oracle first. Its FIFO
            // has three misses then two hits for this five-element index stream.
            for(unsigned round=0;round<rounds;++round) {
                const unsigned control=round==129 ? 1:round+1;
                Metal::ShaderMachineUniforms u{};u.b=control|((control&1u) ? 0x8100u:0u);
                std::array<unsigned,2> mask{};unsigned capacity{};
                Check(Metal::ShaderMachineRuntime::AnalyzeVertices(rp,u,nullptr,nullptr,&mask,&capacity)==
                    Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences,
                    "Changing-mask fixture lost its exact private recurrence proof");
                masks.insert(mask);
                std::array<Metal::VertexRegisters,3> misses{};
                for(unsigned vertex=0;vertex<3;++vertex) {
                    const Metal::BufferSlice source{inputs,vertex*sizeof(Metal::VertexRegisters),
                        sizeof(Metal::VertexRegisters),values+vertex,{}};
                    const auto a=reference.Run(baseline_state,rp,u,source);
                    std::memcpy(&misses[vertex],a.output.data,sizeof(Metal::VertexRegisters));
                }
                for(unsigned element=0;element<count;++element)
                    std::memcpy(static_cast<std::byte*>(expected.contents)+
                        (round*count+element)*sizeof(Metal::VertexRegisters),
                        &misses[element%3],sizeof(Metal::VertexRegisters));
            }
            const auto expected_state=reference.SaveState(baseline_state);
            for(unsigned round=0;round<rounds;++round) {
                const unsigned control=round==129 ? 1:round+1;
                Metal::ShaderMachineUniforms tu{};tu.b=control|((control&1u) ? 0x8100u:0u);
                auto b=machine.RunVertexBatch(trial_state,tp,tu,input,3,count,0,0,ix,1,1000000,true);
                transfer.CopyBuffer(b.buffer,b.offset,actual,round*count*sizeof(Metal::VertexRegisters),b.size);
            }
            Check(masks.size()==129,"Changing-mask fixture did not exceed the pipeline-cache bound");
            scheduler.Finish();
            Check(!std::memcmp(expected.contents,actual.contents,expected.length),
                "Changing recurrence masks or pipeline eviction changed complete outputs");
            const auto b=machine.SaveState(trial_state);
            Check(expected_state.needs_reset==b.needs_reset && expected_state.history==b.history,
                "Changing recurrence masks or in-flight pipeline eviction changed complete retained state");
            std::cout<<"130 changing-mask batches preserve full output/state through 129 masks, both mask words, FIFO hits and in-flight cache eviction\n";
        }

        {
            // Captured/emitted outputs may outlive both later draws and submissions.
            // Check every output field and defer consumers for three retained slices.
            constexpr unsigned batches=64;
            std::array<id<MTLBuffer>,batches> readbacks{};
            std::array<std::vector<Metal::CapturedVertex>,batches> expected{};
            std::vector<Metal::BufferSlice> held;
            std::vector<unsigned> held_batches;
            for(unsigned batch=0;batch<batches;++batch) {
                const bool emission=batch&1;
                unsigned count=emission ? std::array<unsigned,4>{3,6,66,258}[(batch/2)%4]:
                    std::array<unsigned,4>{1,3,65,257}[(batch/2)%4];
                if(batch==62)count=65537;
                if(batch==63)count=65538;
                Metal::GeometryEmissionConfig config{};
                config.attribute_count=2;config.semantics[0]=0x03020100;config.semantics[1]=0x07060504;
                config.flip_viewport=(batch>>1)&1;config.enable_clip=(batch>>2)&1;
                config.fix_quaternions=emission ? (batch>>3)&1:0;
                config.honor_winding=emission ? (batch>>4)&1:0;
                config.clip_coefficient={1,0,0,0};
                const NSUInteger stride=emission ? sizeof(Metal::ShaderMachineEmission):sizeof(Metal::VertexRegisters);
                auto upload=device.CreateBuffer(count*stride+32,true);
                std::memset(upload.contents,0,upload.length);
                for(unsigned vertex=0;vertex<count;++vertex) {
                    Metal::VertexRegisters attributes{};
                    attributes[0]={float(batch*17+vertex),float(batch)+.25f,-.5f,1};
                    attributes[1]={vertex&1 ? -1.f:1.f,0,0,0};
                    auto destination=static_cast<std::byte*>(upload.contents)+16+vertex*stride;
                    std::memcpy(destination,&attributes,sizeof(attributes));
                    if(emission) {
                        const std::uint32_t winding=(vertex/3+batch)&1;
                        std::memcpy(destination+sizeof(attributes),&winding,sizeof(winding));
                    }
                }
                auto input=device.CreateBuffer(upload.length,false);
                Metal::TransferRuntime transfer{device,scheduler};
                transfer.CopyBuffer(upload,0,input,0,upload.length);
                const Metal::BufferSlice source{input,16,count*stride,nullptr,{}};
                const auto tick=scheduler.CurrentTick();
                Metal::BufferSlice output;
                if(emission) {
                    Metal::ShaderMachineResult emitted{};emitted.emissions=source;emitted.emission_count=count;
                    const auto captured=machine.ConvertEmissions(emitted,config);
                    Check(captured.vertex_count==count,"Captured emission count changed");output=captured.storage;
                } else output=machine.CaptureVertices(source,3,config,count);
                Check(scheduler.CurrentTick()==tick,"Native capture unexpectedly submitted or waited");
                Check(output.size==count*sizeof(Metal::CapturedVertex) && !output.data && output.lease,
                      "Native capture lost its private output lease or logical size");
                expected[batch].resize(count);
                for(unsigned vertex=0;vertex<count;++vertex) {
                    unsigned source_vertex=vertex;
                    const unsigned first=vertex/3*3;
                    if(emission && config.honor_winding && ((vertex/3+batch)&1) && vertex%3<2)
                        source_vertex=first+1-vertex%3;
                    auto& value=expected[batch][vertex];
                    const float x=float(batch*17+source_vertex),y=float(batch)+.25f;
                    value.position={x,config.flip_viewport ? -y:y,.5f,1};value.color={1,1,1,1};
                    value.texcoord0=value.texcoord1=value.texcoord2={1,1,0,0};
                    const float sign=source_vertex&1 ? -1.f:1.f;
                    value.quaternion={sign,0,0,0};
                    if(emission && config.fix_quaternions) {
                        const unsigned first_source=first+((config.honor_winding && ((vertex/3+batch)&1)) ? 1:0);
                        const float first_sign=first_source&1 ? -1.f:1.f;
                        if(sign!=first_sign)for(auto& component:value.quaternion)component=-component;
                    }
                    value.view={1,1,1,0};
                    value.clip={.5f,config.enable_clip ? x:0,1,0};
                }
                readbacks[batch]=device.CreateBuffer(output.size,true);
                transfer.CopyBuffer(output.buffer,output.offset,readbacks[batch],0,output.size);
                if(batch==0 || batch==3 || batch==8) {held.push_back(output);held_batches.push_back(batch);}
                if(batch%8==7)scheduler.Submit();
            }
            std::vector<id<MTLBuffer>> late;
            for(const auto& output:held) {
                auto read=device.CreateBuffer(output.size,true);
                Metal::TransferRuntime transfer{device,scheduler};
                transfer.CopyBuffer(output.buffer,output.offset,read,0,output.size);late.push_back(read);
            }
            scheduler.Finish();
            for(unsigned batch=0;batch<batches;++batch)
                Check(!std::memcmp(readbacks[batch].contents,expected[batch].data(),readbacks[batch].length),
                      "Native capture/emission output changed across reuse or submissions");
            for(unsigned i=0;i<late.size();++i)
                Check(!std::memcmp(late[i].contents,expected[held_batches[i]].data(),late[i].length),
                      "Native capture overwrote an output retained for a later consumer");
            std::cout<<batches<<" native capture/emission cases preserve full output bytes and held leases across submissions\n";
        }
        {
            // Release each output only after its GPU consumer has been encoded.
            // Keep three outputs for later consumers while reusing other storage.
            constexpr unsigned batches=64;
            Metal::ShaderMachineProgram program{};
            program.allow_emission=false;program.output_mask=1;
            program.swizzle[0]=Metal::Test::IdentitySwizzle;
            program.code[0]=std::uint32_t(nihstro::OpCode::Id::MOV)<<26;
            program.code[1]=std::uint32_t(nihstro::OpCode::Id::END)<<26;
            Metal::ShaderMachineState state{device};
            std::array<id<MTLBuffer>,batches> readbacks{};
            std::array<std::vector<Metal::VertexRegisters>,batches> expected{};
            std::vector<Metal::BufferSlice> held;
            std::vector<unsigned> held_batches;
            std::array<std::uint64_t,8> first_addresses{};
            bool reused_before_submission=false;
            for(unsigned batch=0;batch<batches;++batch) {
                const unsigned count=batch==batches-1 ? 32769:
                    std::array<unsigned,4>{1,3,65,257}[batch%4];
                expected[batch].resize(count);
                for(unsigned i=0;i<count;++i)
                    expected[batch][i][0]={float(batch*17+i),float(batch),-.5f,1};
                auto input=device.CreateBuffer(count*sizeof(Metal::VertexRegisters),true);
                std::memcpy(input.contents,expected[batch].data(),input.length);
                state.Reset();
                const auto tick=scheduler.CurrentTick();
                const auto output=machine.RunVertexBatch(state,program,{},
                    {input,0,input.length,nullptr,{}},count,count,0,0,{},0,1000000,true);
                if(batch<first_addresses.size()) {
                    const auto address=output.buffer.gpuAddress;
                    reused_before_submission|=std::find(first_addresses.begin(),
                        first_addresses.begin()+batch,address)!=first_addresses.begin()+batch;
                    first_addresses[batch]=address;
                }
                readbacks[batch]=device.CreateBuffer(output.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(output.buffer,output.offset,
                    readbacks[batch],0,output.size);
                if(batch==0 || batch==9 || batch==19) {
                    held.push_back(output);held_batches.push_back(batch);
                }
                Check(scheduler.CurrentTick()==tick,
                      "Private GPU storage reuse submitted or waited within the draw stream");
                if(batch%8==7) scheduler.Submit();
            }
            Check(reused_before_submission,
                  "Private lifetime fixture did not exercise reuse before submission");
            std::array<id<MTLBuffer>,3> held_readbacks{};
            for(unsigned i=0;i<held.size();++i) {
                held_readbacks[i]=device.CreateBuffer(held[i].size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(held[i].buffer,held[i].offset,
                    held_readbacks[i],0,held[i].size);
            }
            scheduler.Finish();
            for(unsigned batch=0;batch<batches;++batch)
                Check(std::memcmp(readbacks[batch].contents,expected[batch].data(),
                                  readbacks[batch].length)==0,
                      "Private buffer reuse overwrote an earlier encoded GPU consumer");
            for(unsigned i=0;i<held.size();++i)
                Check(std::memcmp(held_readbacks[i].contents,expected[held_batches[i]].data(),
                                  held_readbacks[i].length)==0,
                      "Private buffer reuse overwrote an outstanding CPU output slice");
            std::cout<<"64 native private-buffer lifetime cases: pre-submit reuse, encoded consumers, held slices, triple-slot submissions and oversized valid output passed\n";
        }
        {
            // Check every FIFO plan entry against a scalar ring, including partial
            // SIMD chunks and the untouched suffix after the first invalid ID.
            auto library=device.CompileLibrary(Metal::ShaderMachineSource,"Native FIFO plan regression");
            auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
            descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"pica_vertex_machine_plan");
            descriptor.maxTotalThreadsPerThreadgroup=64;
            auto pipeline=device.CompileCompute(descriptor);
            const auto lanes=pipeline.threadExecutionWidth;
            Check(lanes>=16 && lanes<=64,"Unsupported native FIFO test SIMD width");
            unsigned cases=0;
            for(unsigned index_size:{0u,1u,2u,4u})
            for(unsigned count:{1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,257u,8193u})
            for(unsigned pattern=0;pattern<8+lanes;++pattern) {
                // Exercise every partial SIMD-prefix flush after one complete
                // chunk, as well as errors at the first/chunk-boundary/last ID.
                if(pattern>=8 && (!index_size || count!=129)) continue;
                const unsigned vertex_count=index_size==1 ? 192:512;
                const unsigned base=index_size==4 ? UINT32_MAX-vertex_count+1u:0u;
                std::vector<unsigned> ids(count);
                unsigned random=0x12345678u;
                for(unsigned i=0;i<count;++i) {
                    random=random*1664525u+1013904223u;
                    ids[i]=base+(pattern==0 ? i%vertex_count:pattern==1 ? i%48:(random>>16)%90);
                }
                if(index_size && pattern>=3)
                    ids[pattern==3 ? 0:pattern==4 ? count/2:pattern==5 ? count-1:
                        pattern==6 ? std::min<unsigned>(lanes-1,count-1):
                        pattern==7 ? std::min<unsigned>(lanes,count-1):lanes+pattern-8]=base ? base-1:vertex_count;
                const std::array<unsigned,2> canary{0xa5a5a5a5u,0x5a5a5a5au};
                std::vector<std::array<unsigned,2>> expected(count,canary);
                std::array<std::pair<unsigned,unsigned>,64> ring{};
                unsigned size=0,replacement=0,previous=~0u;
                std::array<unsigned,4> expected_status{};
                for(unsigned i=0;i<count;++i) {
                    const unsigned id=index_size ? ids[i]:base+i;
                    if(id<base || id-base>=vertex_count) { expected_status={8,i,0,0};break; }
                    unsigned cached=~0u;
                    if(index_size) for(unsigned slot=0;slot<size;++slot)
                        if(ring[slot].first==id) cached=ring[slot].second;
                    expected[i]={cached,previous};
                    if(cached!=~0u) continue;
                    previous=i;
                    if(index_size) {
                        ring[replacement]={id,i};replacement=(replacement+1u)&63u;
                        size=std::min(size+1u,64u);
                    }
                }
                auto indices=device.CreateBuffer(std::max(count*index_size,1u),true);
                for(unsigned i=0;i<count;++i) for(unsigned byte=0;byte<index_size;++byte)
                    static_cast<std::uint8_t*>(indices.contents)[i*index_size+byte]=std::uint8_t(ids[i]>>(8*byte));
                auto plan=device.CreateBuffer(count*sizeof(canary),true);
                for(unsigned i=0;i<count;++i) std::memcpy(static_cast<std::byte*>(plan.contents)+i*sizeof(canary),canary.data(),sizeof(canary));
                auto status=device.CreateBuffer(16,true);std::memset(status.contents,0,16);
                std::array<unsigned,30> config{};
                config[8]=vertex_count;config[9]=count;config[10]=base;config[11]=base;config[12]=index_size;
                auto params=device.CreateBuffer(sizeof(config),true);std::memcpy(params.contents,config.data(),sizeof(config));
                const std::array<Metal::BufferBinding,4> bindings{{{status,0,7},{params,0,8},{indices,0,9},{plan,0,12}}};
                auto arguments=scheduler.MakeArguments(bindings);
                auto encoder=scheduler.BeginCompute(pipeline);
                [encoder setComputePipelineState:pipeline];[encoder setArgumentTable:arguments];
                [encoder dispatchThreads:MTLSizeMake(lanes,1,1) threadsPerThreadgroup:MTLSizeMake(lanes,1,1)];
                scheduler.EndEncoding();scheduler.Finish();
                Check(std::memcmp(plan.contents,expected.data(),plan.length)==0 &&
                    std::memcmp(status.contents,expected_status.data(),16)==0,
                    "SIMD FIFO plan changed a cached source, preceding miss, first error or untouched suffix");
                ++cases;
            }
            std::cout<<cases<<" native SIMD FIFO plans preserve every entry, partial chunks, all index widths, UINT_MAX and first-error suffixes\n";
        }
        Metal::ShaderMachineState state{device};
        Metal::ShaderMachineProgram program{};
        program.swizzle[0]=Metal::Test::IdentitySwizzle;
        Metal::ShaderMachineUniforms uniforms{};
        Metal::VertexRegisters inputs{};
        inputs[0]={.5f,.25f,-.5f,2}; inputs[1]={.25f,.5f,.125f,1};
        const auto buffer=device.CreateBuffer(sizeof(inputs),true);
        const auto upload=[&] { std::memcpy(buffer.contents,inputs.data(),sizeof(inputs)); };
        upload();
        const Metal::BufferSlice slice{buffer,0,buffer.length,buffer.contents,{}};
        using Op=nihstro::OpCode::Id;
        struct RecordedCase { Metal::ShaderMachineProgram program;Metal::ShaderMachineUniforms uniforms; };
        std::vector<RecordedCase> recorded;
        for(int argument=1;argument<argc;++argument) {
            std::ifstream stream(argv[argument],std::ios::binary);std::array<std::uint32_t,12> header{};
            RecordedCase value;
            stream.read(reinterpret_cast<char*>(header.data()),sizeof(header));
            Check(stream && header[0]==0x34544c4d && header[1]==1 && header[2]==4096 &&
                header[3]==4096 && header[4]==sizeof(value.uniforms),"Invalid native recorded shader case header");
            value.program.entry=header[5];value.program.output_mask=header[6];value.program.allow_emission=header[7];
            stream.read(reinterpret_cast<char*>(value.program.code.data()),sizeof(value.program.code));
            stream.read(reinterpret_cast<char*>(value.program.swizzle.data()),sizeof(value.program.swizzle));
            stream.read(reinterpret_cast<char*>(&value.uniforms),sizeof(value.uniforms));
            Check(bool(stream) && stream.peek()==std::char_traits<char>::eof(),"Truncated/trailing native recorded shader case");
            recorded.push_back(std::move(value));
        }
        const auto word=[](Op op) { return std::uint32_t(op)<<26; };
        const auto arithmetic=[](Op op, unsigned dest, unsigned src1, unsigned src2=0) {
            return Metal::Test::Arithmetic(op,dest,src1,src2);
        };
        {
            // RCP of an inherited temporary requires ordered execution. Dense
            // hit runs must keep their original source indices after element 7;
            // the larger ranges also cross the 64-entry FIFO eviction boundary.
            unsigned cases=0;
            for(unsigned body:{0u,8u,32u,128u}) for(unsigned count:{16u,65u,129u})
            for(unsigned width:{1u,2u,4u}) for(unsigned reset:{0u,1u}) {
                Metal::ShaderMachineProgram p{};p.allow_emission=false;
                p.swizzle[0]=Metal::Test::IdentitySwizzle;
                unsigned pc=0;p.code[pc++]=arithmetic(Op::RCP,16,16);
                for(unsigned i=0;i<body;++i)
                    p.code[pc++]=arithmetic(Op::ADD,(i%15)+17,(i%15)+17,i%16);
                p.code[pc++]=arithmetic(Op::MUL,17,16,0);
                p.code[pc++]=arithmetic(Op::MOV,0,17);p.code[pc]=word(Op::END);
                Check(Metal::ShaderMachineRuntime::AnalyzeVertices(p)==
                    Metal::ShaderMachineRuntime::VertexExecution::Ordered,
                    "Dense FIFO regression shader did not select ordered execution");
                const unsigned vertices=count==16 ? 21:90,base=5;
                auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
                auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
                for(unsigned i=0;i<vertices;++i) for(unsigned reg=0;reg<16;++reg)
                    values[i][reg]={float((i+reg)%13+1)*.125f,float((i+reg)%7+1)*.25f,-.5f,1};
                auto indices=device.CreateBuffer(count*width,true);
                std::vector<unsigned> ids(count);
                for(unsigned i=0;i<count;++i) {
                    ids[i]=base+(i*7)%vertices;
                    for(unsigned byte=0;byte<width;++byte)
                        static_cast<std::uint8_t*>(indices.contents)[i*width+byte]=
                            std::uint8_t(ids[i]>>(byte*8));
                }
                Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                if(!reset) for(unsigned component=0;component<448;++component) {
                    const std::uint32_t bits=0x3f000000u+(component%64)*257u;
                    std::memcpy(initial.history.data()+component*4,&bits,4);
                }
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                std::vector<Metal::VertexRegisters> expected(count);
                std::array<std::pair<unsigned,unsigned>,64> fifo{};
                unsigned size=0,replacement=0;
                for(unsigned element=0;element<count;++element) {
                    unsigned cached=~0u;
                    for(unsigned slot=0;slot<size;++slot)
                        if(fifo[slot].first==ids[element]) cached=fifo[slot].second;
                    if(cached!=~0u) {expected[element]=expected[cached];continue;}
                    const auto result=machine.Run(scalar,p,{},
                        {input,(ids[element]-base)*sizeof(Metal::VertexRegisters),
                            sizeof(Metal::VertexRegisters),nullptr,{}});
                    std::memcpy(&expected[element],result.output.data,sizeof(Metal::VertexRegisters));
                    fifo[replacement]={ids[element],element};
                    replacement=(replacement+1)&63u;size=std::min(size+1,64u);
                }
                const auto result=machine.RunVertexBatch(batched,p,{},
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,
                    {indices,0,indices.length,indices.contents,{}},width);
                auto readback=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(
                    result.buffer,result.offset,readback,0,readback.length);
                scheduler.Finish();
                Check(std::memcmp(readback.contents,expected.data(),readback.length)==0 &&
                    machine.SaveState(batched)==machine.SaveState(scalar),
                    "Ordered FIFO hit/eviction changed output or complete scalar GPU state");
                ++cases;
            }
            std::cout<<cases<<" dense ordered FIFO cases preserve cached source indices, eviction, all index widths and complete GPU state\n";
        }
        {
            // Mutate one source object, reuse copies at other addresses and cross
            // the bounded program-cache eviction limit. Neither a hot cache hit
            // nor eviction may reuse stale instruction/swizzle uploads or proofs.
            Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
            p.code[0]=arithmetic(Op::MOV,0,0);p.code[1]=word(Op::END);
            p.code[8]=arithmetic(Op::MOV,0,1);p.code[9]=word(Op::END);
            p.swizzle[0]=Metal::Test::IdentitySwizzle;
            const auto anchor=p;
            unsigned cases=0;
            const auto verify=[&](const Metal::ShaderMachineProgram& source,unsigned input,bool negate,bool batch) {
                Metal::VertexRegisters expected{};expected[0]=inputs[input];
                if(negate) for(auto& component:expected[0]) component=-component;
                Metal::ShaderMachineSnapshot snapshot{};snapshot.needs_reset=false;
                std::memcpy(snapshot.history.data(),inputs.data(),sizeof(inputs));
                std::memcpy(snapshot.history.data()+512,expected.data(),sizeof(expected));
                state.Reset();
                if(batch) {
                    const auto result=machine.RunVertexBatch(state,source,{},slice,1,1,0,0);
                    auto read=device.CreateBuffer(result.size,true);
                    Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                    scheduler.Finish();
                    Check(read.length==sizeof(expected) && std::memcmp(read.contents,expected.data(),sizeof(expected))==0,
                          "Recent program cache reused stale specialized vertex source");
                } else {
                    const auto result=machine.Run(state,source,{},slice);
                    Check(std::memcmp(result.output.data,expected.data(),sizeof(expected))==0,
                          "Recent program cache reused stale instruction/swizzle upload");
                }
                Check(machine.SaveState(state)==snapshot,"Recent program cache changed complete native state");
                ++cases;
            };
            verify(anchor,0,false,false);
            for(unsigned i=0;i<270;++i) {
                p.code[0]=arithmetic(Op::MOV,0,i&1u);
                p.code[4095]=i+1;p.swizzle[4095]=0xa5a50000u+i;
                const bool negate=i%3==0;
                p.swizzle[0]=Metal::Test::IdentitySwizzle^(negate ? 1u<<4:0u);
                verify(p,i&1u,negate,false);
                if(i%16==0) {
                    const auto copy=p;
                    verify(copy,i&1u,negate,false);verify(copy,i&1u,negate,false);
                }
            }
            verify(anchor,0,false,false);verify(anchor,0,false,false);verify(anchor,0,false,false);
            p=anchor;
            for(unsigned i=0;i<8;++i) {
                p.entry=i&1u ? 8:0;p.output_mask=i&2u ? 0xffff:1;
                verify(p,i&1u,false,true);verify(p,i&1u,false,true);verify(p,i&1u,false,true);
            }
            p.allow_emission=true;verify(p,1,false,false);
            p.allow_emission=false;verify(p,1,false,true);
            std::cout<<cases<<" recent-program source mutations, copied sources, metadata changes and bounded eviction cases preserve all native output registers and complete state\n";
        }
        const auto check=[&](std::array<float,4> expected,std::uint32_t mask=0xffff) {
            const auto result=machine.Run(state,program,uniforms,mask ? slice:Metal::BufferSlice{},mask);
            const auto& actual=(*static_cast<const Metal::VertexRegisters*>(result.output.data))[0];
            for(unsigned c=0;c<4;++c) Check(std::abs(actual[c]-expected[c])<.00001f,"Native shader machine arithmetic/control-flow result mismatch");
        };
        unsigned operations{};
        for(auto op:{Op::ADD,Op::MUL,Op::DP3,Op::DP4,Op::DPH,Op::DST,Op::EX2,Op::LG2,Op::LIT,Op::SGE,Op::SLT,Op::FLR,Op::MAX,Op::MIN,Op::RCP,Op::RSQ,Op::MOV}) {
            program.code.fill(0);program.code[0]=arithmetic(op,0,0,1);program.code[1]=word(Op::END);state.Reset();
            std::array<float,4> expected{};
            for(unsigned c=0;c<4;++c) switch(op) {
            case Op::ADD:expected[c]=inputs[0][c]+inputs[1][c];break;
            case Op::MUL:expected[c]=inputs[0][c]*inputs[1][c];break;
            case Op::DP3:expected[c]=.5f*.25f+.25f*.5f-.5f*.125f;break;
            case Op::DP4:expected[c]=.5f*.25f+.25f*.5f-.5f*.125f+2;break;
            case Op::DPH:expected[c]=.5f*.25f+.25f*.5f-.5f*.125f+1;break;
            case Op::DST:expected=std::array<float,4>{1,.125f,-.5f,1};break;
            case Op::EX2:expected[c]=std::exp2(.5f);break;
            case Op::LG2:expected[c]=std::log2(.5f);break;
            case Op::LIT:expected=std::array<float,4>{.5f,.25f,0,2};break;
            case Op::SGE:expected[c]=inputs[0][c]>=inputs[1][c];break;
            case Op::SLT:expected[c]=inputs[0][c]<inputs[1][c];break;
            case Op::FLR:expected[c]=std::floor(inputs[0][c]);break;
            case Op::MAX:expected[c]=std::max(inputs[0][c],inputs[1][c]);break;
            case Op::MIN:expected[c]=std::min(inputs[0][c],inputs[1][c]);break;
            case Op::RCP:expected[c]=2;break;
            case Op::RSQ:expected[c]=std::sqrt(2.f);break;
            default:expected[c]=inputs[0][c];break;
            }
            check(expected);++operations;
        }
        state.Reset();program.code[0]=arithmetic(Op::MOV,16,0);
        program.code[1]=arithmetic(Op::MOV,0,16);program.code[2]=word(Op::END);check(inputs[0]);
        program.code[0]=arithmetic(Op::ADD,0,16,1);program.code[1]=word(Op::END);
        check({.75f,.75f,-.375f,3}); // Temporary register survives another invocation.
        inputs[1]={9,9,9,9};upload();
        check({.75f,.75f,-.375f,3},1); // Unloaded input register survives too.
        check({.75f,.75f,-.375f,3},0); // No input buffer is needed for a retained bank.
        inputs[1]={.25f,.5f,.125f,1};upload();
        // Descriptor swizzling, negation and destination masking preserve unwritten lanes.
        state.Reset();program.code.fill(0);
        program.swizzle[1]=(Metal::Test::IdentitySwizzle&~(15u|(255u<<5)))|10u|(0xe4u<<5)|(1u<<4);
        program.code[0]=arithmetic(Op::MOV,0,1);
        program.code[1]=arithmetic(Op::MOV,0,0)|1u;program.code[2]=word(Op::END);
        check({-2,.5f,-.25f,1});
        program.swizzle[1]=0;
        state.Reset();program.code.fill(0);
        program.code[0]=word(Op::CALL)|(8u<<10)|1u;program.code[1]=word(Op::END);
        program.code[8]=arithmetic(Op::MOV,0,1);program.code[9]=0xffffffff;
        check(inputs[1]); // A dead word after the call boundary is never interpreted.
        // CALL completion overrides a JMP on its last instruction.
        state.Reset();program.code.fill(0);
        program.code[0]=word(Op::CALL)|(8u<<10)|1u;
        program.code[1]=arithmetic(Op::MOV,0,0);program.code[2]=word(Op::END);
        program.code[8]=word(Op::JMPC)|(2u<<22)|(20u<<10);
        program.code[20]=arithmetic(Op::MOV,0,1);program.code[21]=word(Op::END);
        check(inputs[0]);
        // Hardware CALL stack has four slots and loses the fourth consecutive return-PC update.
        for(unsigned calls:{4u,5u}) {
            state.Reset();program.code.fill(0);
            program.code[0]=word(Op::CALL)|(10u<<10)|1u;
            for(unsigned call=1;call<calls;++call)
                program.code[call*10]=word(Op::CALL)|((call+1)*10u<<10)|1u;
            program.code[calls*10]=arithmetic(Op::MOV,0,0);
            const unsigned final_pc=calls==4 ? 11:21;
            program.code[final_pc]=arithmetic(Op::MOV,0,1);program.code[final_pc+1]=word(Op::END);
            program.code[1]=arithmetic(Op::MOV,0,32);program.code[2]=word(Op::END);
            check(inputs[1]);
        }
        // IF/LOOP overflow discards the oldest entry; only one entry is checked per instruction.
        state.Reset();program.code.fill(0);uniforms.b=1;
        for(unsigned i=0;i<9;++i) program.code[i]=word(Op::IFU)|(30u<<10)|1u;
        program.code[9]=arithmetic(Op::MOV,0,0);program.code[10]=word(Op::JMPU)|(29u<<10);
        program.code[29]=word(Op::NOP);program.code[30]=arithmetic(Op::MOV,0,1);program.code[31]=word(Op::END);
        check(inputs[0]);
        state.Reset();program.code.fill(0);uniforms.i[0]={0,0,0,0};
        for(unsigned i=0;i<5;++i) program.code[i]=word(Op::LOOP)|(8u<<10);
        program.code[5]=arithmetic(Op::MOV,0,0);program.code[6]=word(Op::NOP);program.code[7]=word(Op::NOP);
        program.code[8]=word(Op::NOP);program.code[9]=word(Op::END);check(inputs[0]);
        // Each stack checks its own fall-through PC: LOOP wins over IF over CALL.
        state.Reset();program.code.fill(0);uniforms.b=1;uniforms.i[0]={0,0,0,0};
        program.code[0]=word(Op::LOOP)|(9u<<10);program.code[1]=word(Op::IFU)|(10u<<10)|1u;
        program.code[2]=word(Op::CALL)|(8u<<10)|2u;
        program.code[8]=arithmetic(Op::MOV,0,0);program.code[9]=word(Op::NOP);
        program.code[10]=arithmetic(Op::MOV,0,1);program.code[11]=word(Op::END);check(inputs[1]);
        const auto compare=[](float a,float b,unsigned mode) {
            switch(mode) { case 0:return a==b;case 1:return a!=b;case 2:return a<b;
            case 3:return a<=b;case 4:return a>b;case 5:return a>=b;default:return true; }
        };
        unsigned conditions{};
        for(unsigned comparison=0;comparison<8;++comparison) {
            for(unsigned condition=0;condition<4;++condition) for(unsigned refs=0;refs<4;++refs) {
                state.Reset();program.code.fill(0);
                program.code[0]=word(Op::CMP)|(comparison<<24)|(comparison<<21)|(1u<<7);
                program.code[1]=word(Op::IFC)|(3u<<10)|1u|(condition<<22)|((refs&1u)<<25)|((refs>>1)<<24);
                program.code[2]=arithmetic(Op::MOV,0,0);program.code[3]=arithmetic(Op::MOV,0,1);program.code[4]=word(Op::END);
                const bool x=compare(inputs[0][0],inputs[1][0],comparison)==bool(refs&1);
                const bool y=compare(inputs[0][1],inputs[1][1],comparison)==bool(refs&2);
                const bool take=condition==0 ? x||y : condition==1 ? x&&y : condition==2 ? x:y;
                check(inputs[take ? 0:1]);++conditions;
            }
        }
        for(bool condition:{false,true}) {
            state.Reset();program.code.fill(0);uniforms.b=condition;
            program.code[0]=word(Op::IFU)|(2u<<10)|1u;
            program.code[1]=arithmetic(Op::MOV,0,0);program.code[2]=arithmetic(Op::MOV,0,1);program.code[3]=word(Op::END);
            check(inputs[condition ? 0:1]);
        }
        state.Reset();program.code.fill(0);uniforms.i[0]={2,3,2,0};
        program.code[0]=word(Op::LOOP)|(1u<<10);
        program.code[1]=arithmetic(Op::ADD,16,16,1);program.code[2]=arithmetic(Op::MOV,0,16);program.code[3]=word(Op::END);
        check({.75f,1.5f,.375f,3});
        // A jump on the final LOOP instruction is overridden by the loop stack.
        state.Reset();program.code[1]=word(Op::JMPC)|(2u<<22)|(20u<<10);
        program.code[2]=arithmetic(Op::MOV,0,0);program.code[3]=word(Op::END);
        program.code[20]=arithmetic(Op::MOV,0,1);program.code[21]=word(Op::END);check(inputs[0]);
        for(auto stop:{Op::BREAK,Op::BREAKC}) {
            state.Reset();program.code.fill(0);program.code[0]=word(Op::LOOP)|(2u<<10);
            program.code[1]=word(stop)|(2u<<22);program.code[2]=arithmetic(Op::ADD,16,16,1);
            program.code[3]=arithmetic(Op::MOV,0,16);program.code[4]=word(Op::END);check({0,0,0,0});
        }
        // Different-width source encodings and non-fused multiply/add.
        uniforms.f[0]={.125f,.25f,.5f,1};
        for(auto op:{Op::DPHI,Op::SGEI,Op::SLTI}) {
            state.Reset();program.code.fill(0);
            program.code[0]=word(op)|(0u<<14)|(32u<<7);program.code[1]=word(Op::END);
            std::array<float,4> expected{};
            for(unsigned c=0;c<4;++c) expected[c]=op==Op::DPHI ? .875f : op==Op::SGEI ? inputs[0][c]>=uniforms.f[0][c] : inputs[0][c]<uniforms.f[0][c];
            check(expected);
        }
        state.Reset();program.code.fill(0);program.code[0]=word(Op::DSTI)|(32u<<7);program.code[1]=word(Op::END);
        check({1,.0625f,.5f,2});
        // The wide SRC2 field, its own descriptor and relative uniform index are independent.
        inputs[3]={2,0,0,0};upload();uniforms.f[2]={2,3,4,5};
        program.swizzle[1]=Metal::Test::IdentitySwizzle|(1u<<13);
        program.code[0]=arithmetic(Op::MOVA,0,3);program.code[1]=word(Op::DSTI)|(32u<<7)|(1u<<19)|1u;program.code[2]=word(Op::END);
        state.Reset();check({1,-.75f,-4,2});program.swizzle[1]=0;
        inputs[0].fill(1.f+std::ldexp(1.f,-13));inputs[1].fill(1.f-std::ldexp(1.f,-13));inputs[2].fill(-1);upload();
        for(auto op:{Op::MAD,Op::MADI}) {
            state.Reset();program.code.fill(0);
            program.code[0]=word(op)|(1u<<(op==Op::MAD ? 10:12))|(2u<<5);program.code[1]=word(Op::END);
            const auto result=machine.Run(state,program,uniforms,slice);
            const auto& actual=(*static_cast<const Metal::VertexRegisters*>(result.output.data))[0];
            Check(actual==std::array<float,4>{0,0,0,0},"MAD/MADI incorrectly fused multiply and add");
        }
        inputs[0]={.5f,.25f,-.5f,2};inputs[1]={.25f,.5f,.125f,1};upload();
        // Relative uniform addressing, out-of-range constants, and MOVA truncation.
        state.Reset();program.code[0]=arithmetic(Op::MOVA,0,0);
        program.code[1]=arithmetic(Op::MOV,0,32)|(1u<<19);program.code[2]=word(Op::END);
        uniforms.f[0]={3,4,5,6};check(uniforms.f[0]);
        inputs[0][0]=100;upload();state.Reset();check({1,1,1,1});
        inputs[0][0]=200;upload();state.Reset();check(uniforms.f[0]); // Outside signed-byte range disables offset.
        inputs[0][0]=-1;uniforms.f[95]={7,6,5,4};upload();state.Reset();check({1,1,1,1});
        // c95 with offset -1 reaches c94; c0 with offset +95 reaches c95.
        program.code[1]=arithmetic(Op::MOV,0,32+95)|(1u<<19);uniforms.f[94]={8,7,6,5};
        state.Reset();check(uniforms.f[94]);
        inputs[0][0]=95;upload();program.code[1]=arithmetic(Op::MOV,0,32)|(1u<<19);
        state.Reset();check(uniforms.f[95]);
        inputs[0]={0,0,0,0};inputs[1].fill(std::numeric_limits<float>::infinity());upload();
        state.Reset();program.code[0]=arithmetic(Op::MUL,0,0,1);program.code[1]=word(Op::END);
        check({0,0,0,0});
        // Every EMIT switches output banks, even without primitive emission.
        inputs[0]={1,2,3,4};inputs[1]={5,6,7,8};upload();state.Reset();
        program.output_mask=1;program.code.fill(0);
        program.code[0]=arithmetic(Op::MOV,0,0);
        program.code[1]=word(Op::SETEMIT);program.code[2]=word(Op::EMIT);
        program.code[3]=arithmetic(Op::MOV,0,1);
        program.code[4]=word(Op::SETEMIT)|(1u<<24);program.code[5]=word(Op::EMIT);
        program.code[6]=word(Op::SETEMIT)|(2u<<24)|(1u<<23)|(1u<<22);
        program.code[7]=word(Op::EMIT);program.code[8]=word(Op::END);
        const auto emitted=machine.Run(state,program,uniforms,slice);
        Check(emitted.emission_count==3,"Native GS did not emit a complete three-vertex primitive");
        const auto* vertices=static_cast<const Metal::ShaderMachineEmission*>(emitted.emissions.data);
        Check(vertices[0].attributes[0]==inputs[0] && vertices[1].attributes[0]==inputs[1] &&
              vertices[2].attributes[0]==inputs[0] && vertices[0].winding==1 &&
              vertices[1].winding==0 && vertices[2].winding==0,
              "Native GS SETEMIT slots, winding or two output banks were incorrect");
        // Reuse two slots from the previous invocation; compact noncontiguous output registers.
        const auto saved_geometry_unit=machine.SaveState(state);
        program.code.fill(0);program.output_mask=9;
        program.code[0]=arithmetic(Op::MOV,0,1);program.code[1]=arithmetic(Op::MOV,3,0);
        program.code[2]=word(Op::SETEMIT)|(2u<<24)|(1u<<23);program.code[3]=word(Op::EMIT);program.code[4]=word(Op::END);
        const auto reused=machine.Run(state,program,uniforms,slice);
        const auto* reused_vertices=static_cast<const Metal::ShaderMachineEmission*>(reused.emissions.data);
        Check(reused.emission_count==3 && reused_vertices[0].attributes[0]==inputs[0] &&
              reused_vertices[1].attributes[0]==inputs[1] && reused_vertices[2].attributes[0]==inputs[1] &&
              reused_vertices[2].attributes[1]==inputs[0] && reused_vertices[2].attributes[2]==std::array<float,4>{},
              "Native GS did not preserve emission slots or compact its output register mask");
        Metal::ShaderMachineState restored_unit{device};machine.LoadState(restored_unit,saved_geometry_unit);
        Check(machine.SaveState(restored_unit)==saved_geometry_unit,"Shader snapshot did not restore both output banks and emission slots");
        const auto resumed=machine.Run(restored_unit,program,uniforms,slice);
        Check(resumed.emission_count==reused.emission_count &&
              std::memcmp(resumed.emissions.data,reused.emissions.data,reused.emission_count*sizeof(Metal::ShaderMachineEmission))==0,
              "Recreated GS unit could not resume bank/slot state after snapshot restore");
        const auto stable_unit=machine.SaveState(restored_unit);
        for (unsigned failure=1;failure<=7;++failure) {
            Metal::ShaderMachineProgram failing{};
            failing.swizzle[0]=Metal::Test::IdentitySwizzle;
            failing.code[0]=arithmetic(Op::MOV,16,1);
            auto failure_uniforms=uniforms;
            failure_uniforms.b=1;
            unsigned capacity=192;
            switch (failure) {
            case 1:
                failing.entry=4095;failing.code[4095]=word(Op::NOP);break;
            case 2:
                failing.code[1]=word(static_cast<Op>(0x10));break;
            case 3:
                failing.code[1]=word(Op::SETEMIT)|(3u<<24);
                failing.code[2]=word(Op::EMIT);break;
            case 4:
                capacity=2;failing.code[1]=word(Op::SETEMIT)|(2u<<24)|(1u<<23);
                failing.code[2]=word(Op::EMIT);break;
            case 5:
                failing.code[1]=word(Op::CALL)|(4095u<<10)|2u;break;
            case 6:
                failing.code[1]=word(Op::JMPU)|(1u<<10);break;
            case 7:
                failing.code[1]=word(Op::BREAK);break;
            }
            bool exact_error{};
            try { machine.Run(restored_unit,failing,failure_uniforms,slice,0xffff,capacity,32); }
            catch (const Metal::ShaderMachineError& error) { exact_error=error.Code()==failure; }
            Check(exact_error,"Native shader machine reported the wrong failure class");
            Check(machine.SaveState(restored_unit)==stable_unit,
                  "Failed native shader execution changed its complete committed GPU snapshot");
        }
        std::cout<<"All seven native shader error classes preserve the entire 1824-byte committed snapshot\n";
        for(unsigned offset:{1804u,1808u,1812u,1816u,1820u}) {
            auto invalid=stable_unit;invalid.history[offset]=std::byte{255};bool failed=false;
            try { machine.LoadState(restored_unit,invalid); } catch(const std::invalid_argument&) { failed=true; }
            Check(failed && machine.SaveState(restored_unit)==stable_unit,"Invalid shader snapshot partially changed the GPU unit");
        }
        Metal::ShaderMachineState reset_unit{device};auto invalid_reset=machine.SaveState(reset_unit);
        invalid_reset.history[0]=std::byte{1};bool failed_reset=false;
        try { machine.LoadState(restored_unit,invalid_reset); } catch(const std::invalid_argument&) { failed_reset=true; }
        Check(failed_reset && machine.SaveState(restored_unit)==stable_unit,"Noncanonical reset shader snapshot was accepted");
        bool rejected=false;
        try { machine.Run(state,program,uniforms,slice,0xffff,2); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected,"Native GS overflowed its emission buffer");
        state.Reset();program.code.fill(0);program.code[0]=word(Op::JMPC)|(2u<<22);
        rejected=false;
        try { machine.Run(state,program,uniforms,slice,0xffff,192,32); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected,"Native shader machine did not bound an infinite control-flow loop");
        state.Reset();program.code[0]=word(Op::CALL)|2u;program.code[1]=word(Op::END);
        rejected=false;
        try { machine.Run(state,program,uniforms,slice,0xffff,192,32); }
        catch(const Metal::ShaderMachineError& error) { rejected=error.Code()==6; }
        Check(rejected,"Native shader machine did not retain ring-stack behavior while bounding infinite recursion");
        state.Reset();program.code[0]=word(Op::SETEMIT)|(3u<<24);program.code[1]=word(Op::EMIT);program.code[2]=word(Op::END);
        rejected=false;
        try { machine.Run(state,program,uniforms,slice); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected,"Native GS accepted reserved emission vertex ID 3");
        state.Reset();program.code.fill(0);program.code[1]=word(Op::END);
        program.code[0]=word(static_cast<Op>(0x10));
        rejected=false;
        try { machine.Run(state,program,uniforms,slice); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected,"Unsupported native opcode was silently executed");
        program.allow_emission=false;
        for(auto opcode:{Op::SETEMIT,Op::EMIT}) {
            program.code[0]=word(opcode);rejected=false;
            try { machine.Run(state,program,uniforms,slice); } catch(const Metal::ShaderMachineError& error) { rejected=error.Code()==2; }
            Check(rejected,"VS mode executed a geometry emission opcode");
        }
        program.allow_emission=true;
        // Failed execution may not overwrite a previously committed shader unit.
        state.Reset();program.code[0]=arithmetic(Op::MOV,16,0);program.code[1]=word(Op::END);
        machine.Run(state,program,uniforms,slice);
        program.code[0]=arithmetic(Op::MOV,16,1);program.code[1]=word(static_cast<Op>(0x10));
        rejected=false;
        try { machine.Run(state,program,uniforms,slice); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected,"Native shader machine accepted a reserved opcode after a state write");
        program.code[0]=arithmetic(Op::MOV,0,16);program.code[1]=word(Op::END);check(inputs[0]);
        // Batched execution must preserve cross-invocation temporary registers,
        // circular cache replacement, call boundaries and every submitted output.
        program={};program.allow_emission=false;program.swizzle[0]=Metal::Test::IdentitySwizzle;
        program.code[0]=word(Op::CALL)|(8u<<10)|2u;program.code[1]=word(Op::END);
        program.code[8]=arithmetic(Op::ADD,16,16,0);program.code[9]=arithmetic(Op::MOV,0,16);
        constexpr unsigned batch_vertices=90,batch_elements=180,vertex_base=5;
        std::array<Metal::VertexRegisters,batch_vertices> batch_inputs{};
        for(unsigned vertex=0;vertex<batch_vertices;++vertex)
            batch_inputs[vertex][0]={float(vertex+1),float(vertex%7),-.25f,float(vertex%3)};
        const auto batch_input_buffer=device.CreateBuffer(sizeof(batch_inputs),true);
        std::memcpy(batch_input_buffer.contents,batch_inputs.data(),sizeof(batch_inputs));
        const Metal::BufferSlice batch_input{batch_input_buffer,0,batch_input_buffer.length,batch_input_buffer.contents,{}};
        {
            // A linear entry can share code memory with unrelated control flow.
            // Exercise the final instruction address and exact instruction limits,
            // plus an indexed suffix made entirely of hits after the last miss.
            unsigned cases=0;
            for(unsigned entry:{0u,123u,4094u}) for(unsigned width:{0u,1u,2u,4u})
            for(unsigned reset:{0u,1u}) {
                Metal::ShaderMachineProgram linear{};linear.allow_emission=false;
                linear.entry=entry;linear.output_mask=1;linear.swizzle[0]=Metal::Test::IdentitySwizzle;
                linear.code.fill(word(Op::CALL)|(4000u<<10)|20u);
                linear.code[entry]=arithmetic(Op::MOV,0,0);linear.code[entry+1]=word(Op::END);
                auto initial=reset ? Metal::ShaderMachineSnapshot{}:stable_unit;
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                constexpr unsigned count=65;
                auto indices=device.CreateBuffer(count*4,true);
                std::array<unsigned,count> ids{};
                for(unsigned i=0;i<count;++i) {
                    ids[i]=vertex_base+(width ? i%17:i);
                    for(unsigned byte=0;byte<width;++byte)
                        static_cast<std::uint8_t*>(indices.contents)[i*width+byte]=std::uint8_t(ids[i]>>(8*byte));
                }
                const Metal::BufferSlice indexed{indices,0,indices.length,indices.contents,{}};
                bool scalar_failed=false,batch_failed=false;
                try {machine.Run(scalar,linear,uniforms,batch_input,0xffff,192,1);}
                catch(const Metal::ShaderMachineError& error){scalar_failed=error.Code()==6;}
                try {machine.RunVertexBatch(batched,linear,uniforms,batch_input,batch_vertices,count,
                    vertex_base,vertex_base,width ? indexed:Metal::BufferSlice{},width,1);}
                catch(const Metal::ShaderMachineError& error){batch_failed=error.Code()==6;}
                Check(scalar_failed && batch_failed && machine.SaveState(scalar)==initial &&
                    machine.SaveState(batched)==initial,"Linear specialization changed instruction-limit rollback");
                std::array<Metal::VertexRegisters,count> expected{};
                for(unsigned i=0;i<count;++i) {
                    if(width && i>=17) {expected[i]=expected[i%17];continue;}
                    auto input=batch_input;input.offset=(ids[i]-vertex_base)*sizeof(Metal::VertexRegisters);
                    input.size=sizeof(Metal::VertexRegisters);
                    const auto result=machine.Run(scalar,linear,uniforms,input,0xffff,192,2);
                    std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                }
                const auto result=machine.RunVertexBatch(batched,linear,uniforms,batch_input,batch_vertices,count,
                    vertex_base,vertex_base,width ? indexed:Metal::BufferSlice{},width,2);
                const auto readback=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,readback,0,result.size);
                scheduler.Finish();
                Check(std::memcmp(readback.contents,expected.data(),result.size)==0 &&
                    machine.SaveState(batched)==machine.SaveState(scalar),
                    "Linear specialization/FIFO finalization changed output or complete state");
                ++cases;
            }
            std::cout<<cases<<" linear-entry/FIFO-finalization cases preserve scalar GPU output/state, address 4095, instruction-limit rollback, all index widths, reset and trailing cache hits\n";
        }
        {
            // Vertex execution must retain inactive GS banks/slots and emitter
            // bits, including a trailing FIFO hit with a different last input.
            // An emission reached through control flow remains an atomic VS error.
            unsigned cases=0;
            for(auto emission:{Op::EMIT,Op::SETEMIT}) for(unsigned control:{0u,1u})
            for(unsigned reset:{0u,1u}) for(unsigned bank:{0u,1u})
            for(unsigned width:{0u,1u,2u,4u}) {
                if(reset && bank) continue;
                Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
                p.code.fill(word(static_cast<Op>(0x10)));p.swizzle[0]=Metal::Test::IdentitySwizzle;
                p.code[0]=arithmetic(Op::LIT,16,0);p.code[1]=word(Op::IFU)|(8u<<10)|1u;
                p.code[2]=arithmetic(Op::MOV,0,16);p.code[3]=arithmetic(Op::MOVA,0,0);
                p.code[4]=arithmetic(Op::MOV,17,0);
                p.code[5]=p.code[6]=p.code[7]=word(Op::NOP);
                p.code[8]=word(emission)|(15u<<22);p.code[9]=word(Op::END);
                Metal::ShaderMachineUniforms u{};u.b=control;
                auto initial=reset ? Metal::ShaderMachineSnapshot{}:stable_unit;
                const std::array<unsigned,5> flags{0,0,1,bank,15};
                if(!reset) std::memcpy(initial.history.data()+451*4,flags.data(),20);
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                constexpr unsigned count=9;
                auto index=device.CreateBuffer(count*4,true);
                std::array<unsigned,count> ids{};
                for(unsigned i=0;i<count;++i) {
                    ids[i]=vertex_base+(width ? (i==count-1 ? 0:i%3):i);
                    for(unsigned byte=0;byte<width;++byte)
                        static_cast<std::uint8_t*>(index.contents)[i*width+byte]=std::uint8_t(ids[i]>>(byte*8));
                }
                std::array<Metal::VertexRegisters,count> expected{};
                unsigned failure=0;std::string message;
                for(unsigned i=0;i<count;++i) {
                    if(width && i>=3) {expected[i]=expected[ids[i]-vertex_base];continue;}
                    auto source=batch_input;source.offset=(ids[i]-vertex_base)*sizeof(Metal::VertexRegisters);
                    source.size=sizeof(Metal::VertexRegisters);
                    try {const auto out=machine.Run(scalar,p,u,source);
                        std::memcpy(&expected[i],out.output.data,sizeof(Metal::VertexRegisters));}
                    catch(const Metal::ShaderMachineError& error) {failure=error.Code();message=error.what();break;}
                }
                Check(failure==(control ? 0u:2u),"Scalar vertex fixture changed conditional VS emission legality");
                if(failure) {
                    unsigned actual=0;std::string actual_message;
                    try {machine.RunVertexBatch(batched,p,u,batch_input,batch_vertices,count,vertex_base,vertex_base,
                        width ? Metal::BufferSlice{index,0,index.length,index.contents,{}}:Metal::BufferSlice{},width);}
                    catch(const Metal::ShaderMachineError& error) {actual=error.Code();actual_message=error.what();}
                    Check(actual==failure && actual_message==message && machine.SaveState(scalar)==initial &&
                          machine.SaveState(batched)==initial,
                          "Control-reached vertex emission changed exact error PC or full GS-state rollback");
                } else {
                    const auto expected_state=machine.SaveState(scalar);
                    const auto out=machine.RunVertexBatch(batched,p,u,batch_input,batch_vertices,count,
                        vertex_base,vertex_base,width ? Metal::BufferSlice{index,0,index.length,index.contents,{}}:
                        Metal::BufferSlice{},width);
                    auto read=device.CreateBuffer(out.size,true);
                    Metal::TransferRuntime{device,scheduler}.CopyBuffer(out.buffer,out.offset,read,0,read.length);
                    scheduler.Finish();
                    Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                          machine.SaveState(batched)==expected_state,
                          "Vertex execution changed inactive GS banks/slots, emitter bits or final miss state");
                }
                ++cases;
            }
            std::cout<<cases<<" native VS/GS-state cases preserve both banks, inactive emission slots, emitter bits, reset, all index widths, trailing hits and control-reached EMIT/SETEMIT rollback\n";
        }
        for(unsigned mode=0;mode<235+recorded.size();++mode) {
        std::cout<<"Vertex proof regression mode "<<mode<<" (including sequential eight-vertex group)\n";
        const bool independent=mode!=0 && mode!=8 && mode!=9 && mode!=11 && mode!=13;
        program.output_mask=1;
        program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|10u;
        program.code[9]=arithmetic(Op::MOV,0,16)|(mode==2 ? 1u:0u);
        program.code[8]=arithmetic(independent ? Op::MOV:Op::ADD,16,independent ? 0:16,0);
        program.code.fill(0);
        program.code[0]=word(Op::CALL)|(8u<<10)|2u;program.code[1]=word(Op::END);
        program.code[8]=arithmetic(independent ? Op::MOV:Op::ADD,16,independent ? 0:16,0);
        program.code[9]=arithmetic(Op::MOV,0,16)|(mode==2 ? 1u:0u);
        uniforms={};uniforms.i[0]={2,1,1,0};uniforms.b=mode==4 ? 1:0;
        if(mode==3 || mode==4) {
            program.code[0]=word(Op::IFU)|(4u<<10)|1u;
            program.code[1]=word(Op::CALL)|(8u<<10)|2u;
            program.code[2]=program.code[3]=program.code[9]=word(Op::NOP);
            program.code[4]=arithmetic(Op::MOV,16,1);
            program.code[5]=arithmetic(Op::MOV,0,16);program.code[6]=word(Op::END);
        } else if(mode==5) {
            program.code[0]=arithmetic(Op::MOV,16,0);program.code[1]=word(Op::LOOP)|(4u<<10);
            program.code[2]=arithmetic(Op::ADD,16,16,1);
            program.code[3]=program.code[4]=word(Op::NOP);
            program.code[5]=arithmetic(Op::MOV,0,16);program.code[6]=word(Op::END);
        } else if(mode==6) {
            uniforms.b=1;uniforms.i[0]={0,0,0,0};
            program.code[0]=word(Op::LOOP)|(9u<<10);
            program.code[1]=word(Op::IFU)|(10u<<10)|1u;
            program.code[2]=word(Op::CALL)|(8u<<10)|2u;
            program.code[8]=arithmetic(Op::MOV,16,0);program.code[9]=word(Op::NOP);
            program.code[10]=arithmetic(Op::MOV,0,1);program.code[11]=word(Op::END);
        }
        if(mode>=7 && mode<=9) {
            program.code.fill(0);uniforms.f[0]={45,45,45,45};
            program.code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
            program.code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
            program.code[2]=arithmetic(Op::MOV,mode==9 ? 1:16,0);
            program.code[3]=word(Op::NOP);program.code[4]=word(Op::NOP);
            program.code[5]=arithmetic(Op::MOV,0,mode==9 ? 0:16);program.code[6]=word(Op::END);
            if(mode==7) {
                // Both arms define the temporary, despite different vertex-dependent paths.
                program.code[4]=arithmetic(Op::MOV,16,1);
            }
        }
        if(mode>=10) {
            program.code.fill(0);program.code[0]=word(Op::CALL)|(8u<<10)|2u;program.code[1]=word(Op::END);
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(mode>=12 ? 14u:10u);
            program.code[8]=arithmetic(Op::MOV,16,0)|1u;
            program.code[9]=arithmetic(mode==12 ? Op::DP3:mode==13 ? Op::DP4:Op::MOV,0,16,1)|(mode==10 ? 1u:0u);
            if(mode==11 || mode==13) {
                // Unwritten lanes are only dependencies when another invocation can change them.
                program.swizzle[2]=(Metal::Test::IdentitySwizzle&~15u)|(mode==11 ? 5u:1u);
                program.code[0]=word(Op::CALL)|(8u<<10)|3u;
                program.code[10]=arithmetic(Op::MOV,16,0)|2u;
            }
        }
        if(mode>=14 && mode<142) {
            std::uint32_t random=0x9e3779b9u*mode;
            const auto next=[&] { random^=random<<13;random^=random>>17;random^=random<<5;return random; };
            constexpr std::array ops{Op::ADD,Op::MUL,Op::DP3,Op::DP4,Op::DPH,Op::DST,Op::EX2,Op::LG2,
                Op::LIT,Op::SGE,Op::SLT,Op::FLR,Op::MAX,Op::MIN,Op::RCP,Op::RSQ,Op::MOV};
            program.code.fill(0);program.code[0]=word(Op::CALL)|(8u<<10)|3u;program.code[1]=word(Op::END);
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(1u+next()%15);
            program.swizzle[2]=(next()&0x003ffff0u)|(1u+next()%15);
            program.swizzle[3]=(Metal::Test::IdentitySwizzle&~15u)|(1u+next()%15);
            program.code[8]=arithmetic(Op::MOV,16,0)|1u;
            program.code[9]=arithmetic(ops[next()%ops.size()],0,16,1)|2u;
            program.code[10]=arithmetic(Op::MOV,16,2)|3u;
        }
        if(mode>=142 && mode<158) {
            program.code.fill(0);uniforms.f[0]={45,45,45,45};
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(1u+(mode-142)%15u);
            program.code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
            program.code[1]=word(Op::IFC)|(8u<<10)|1u|(2u<<22)|(1u<<25);
            program.code[2]=arithmetic(Op::MOV,1,0)|1u;
            program.code[3]=arithmetic(Op::MOV,16,0)|1u;
            program.code[4]=arithmetic(Op::MOVA,0,0);
            program.code[5]=word(Op::LOOP)|(6u<<10);
            program.code[6]=word(Op::NOP);
            program.code[7]=arithmetic(Op::LIT,2,0)|1u;
            program.code[8]=word(Op::NOP);
            program.code[9]=arithmetic(Op::MOV,0,0);program.code[10]=word(Op::END);
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
                Metal::ShaderMachineRuntime::VertexExecution::MergeState,
                "Conditional component writes did not select native GPU state merging");
        }
        if(mode>=158 && mode<166) {
            program.code.fill(0);
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(8u>>((mode-158)%4));
            program.code[0]=arithmetic(Op::MOV,16,17)|1u;
            program.code[1]=arithmetic(Op::MOV,0,mode<162 ? 0:16)|1u;
            program.code[2]=arithmetic(Op::MOV,16,0);
            program.code[3]=arithmetic(Op::MOV,17,1);
            program.code[4]=word(Op::END);
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
                (mode<162 ? Metal::ShaderMachineRuntime::VertexExecution::Independent:
                            Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses),
                "Dead intermediate reads and live inherited output dependencies were conflated");
        }
        if(mode>=166 && mode<198) {
            // Old temporary values flow through one to sixteen registers across
            // invocations. The final root is replaced from input, so the component
            // graph is acyclic even though a single parallel pass is insufficient.
            const unsigned length=1+(mode-166)%16;
            const bool conditional=mode>=182;
            program.code.fill(0);program.swizzle[0]=Metal::Test::IdentitySwizzle;
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(8u>>((mode-166)%4));
            unsigned pc=0;
            program.code[pc++]=arithmetic(Op::MOV,0,16);
            for(unsigned reg=0;reg+1<length;++reg)
                program.code[pc++]=arithmetic(Op::MOV,16+reg,17+reg);
            if(conditional) {
                uniforms.f[0]={45,45,45,45};
                program.code[pc++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
                const unsigned branch=pc++;
                program.code[branch]=word(Op::IFC)|((branch+2u)<<10)|1u|(2u<<22)|(1u<<25);
                program.code[pc++]=arithmetic(Op::MOV,15+length,0)|1u;
                program.code[pc++]=word(Op::NOP);
            } else program.code[pc++]=arithmetic(Op::MOV,15+length,0);
            program.code[pc++]=word(Op::END);
            std::uint32_t passes=0;
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms,nullptr,&passes)==
                  Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses && passes==length+1,
                  "Acyclic retained-register chain has the wrong native GPU pass count");
        }
        if(mode==198) {
            // Address-relative uniforms consume the preceding invocation's MOVA.
            program.code.fill(0);uniforms.f[0]={2,3,4,5};uniforms.f[1]={7,8,9,10};
            program.code[0]=arithmetic(Op::MOV,0,32)|(1u<<19);
            program.code[1]=arithmetic(Op::MOVA,0,0);program.code[2]=word(Op::END);
            std::uint32_t passes=0;
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms,nullptr,&passes)==
                  Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses && passes==2,
                  "Retained MOVA dependency did not select two native GPU passes");
        }
        if(mode>=199 && mode<202) {
            program.code.fill(0);
            if(mode==199) { // A true accumulator cannot converge in a fixed DAG depth.
                program.code[0]=arithmetic(Op::ADD,16,16,0);
                program.code[1]=arithmetic(Op::MOV,0,16);program.code[2]=word(Op::END);
            } else if(mode==200) { // A two-register recurrence is still a cycle.
                program.code[0]=arithmetic(Op::MOV,16,17);
                program.code[1]=arithmetic(Op::MOV,17,16);
                program.code[2]=arithmetic(Op::MOV,0,17);program.code[3]=word(Op::END);
            } else { // Speculation must never change a branch/error path.
                program.code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(16u<<12);
                program.code[1]=word(Op::IFC)|(4u<<10)|1u|(2u<<22)|(1u<<25);
                program.code[2]=arithmetic(Op::MOV,0,0);program.code[3]=word(Op::NOP);
                program.code[4]=word(Op::NOP);program.code[5]=arithmetic(Op::MOV,16,0);
                program.code[6]=word(Op::END);
            }
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
                  Metal::ShaderMachineRuntime::VertexExecution::Ordered,
                  "A retained-state cycle or changing control predicate was speculated");
        }
        if(mode>=202 && mode<227) {
            program.code.fill(0);program.swizzle[0]=Metal::Test::IdentitySwizzle;
            program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|(1u+(mode-202)%15u);
            program.swizzle[2]=program.swizzle[1]|(1u<<4)|(1u<<13);
            uniforms={};uniforms.f[0].fill(45);uniforms.i[0]={2,0,0,0};
            unsigned pc=0;
            program.code[pc++]=arithmetic(Op::MOV,0,0);
            if(mode<218 || mode==225) {
                // Private accumulators are not discarded: every update, masked lane,
                // sign and conditional reset must survive the complete GPU snapshot.
                program.code[pc++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<7);
                const unsigned branch=pc++;
                program.code[branch]=word(Op::IFC)|((branch+2u)<<10)|1u|(2u<<22)|(1u<<25);
                program.code[pc++]=arithmetic(Op::MOV,16,1)|1u;
                program.code[pc++]=word(Op::NOP);
                const unsigned loop=pc++;
                program.code[pc++]=arithmetic(Op::MUL,16,16,0)|2u;
                program.code[pc++]=arithmetic(Op::ADD,16,0,16)|2u;
                program.code[pc++]=arithmetic(Op::MOV,16,16)|2u;
                program.code[loop]=word(Op::LOOP)|((pc-1u)<<10);
                if(mode==225) program.code[pc++]=arithmetic(Op::ADD,17,17,0);
            } else if(mode==218) {
                program.code[pc++]=arithmetic(Op::MOV,0,17);
                program.code[pc++]=arithmetic(Op::MOV,17,0);
                program.code[pc++]=arithmetic(Op::MUL,16,16,0);
            } else if(mode==219) {
                program.code[pc++]=arithmetic(Op::MUL,16,0,16)|2u;
                program.code[pc++]=arithmetic(Op::ADD,17,17,0)|1u;
            } else if(mode==226) {
                // One merge dispatch must give both recurrence-mask words their
                // own writers while ordinary temporary/output lanes still merge.
                program.code[pc++]=arithmetic(Op::MOV,17,1);
                program.code[pc++]=arithmetic(Op::MUL,16,16,0)|2u;
                program.code[pc++]=arithmetic(Op::ADD,31,31,0)|1u;
            } else if(mode==220) program.code[pc++]=arithmetic(Op::FLR,16,16);
            else if(mode==221) {
                program.code[pc++]=arithmetic(Op::MOV,17,16);
                program.code[pc++]=arithmetic(Op::ADD,16,16,17);
                program.code[pc++]=arithmetic(Op::MOV,17,0);
            } else if(mode==222) program.code[pc++]=arithmetic(Op::ADD,16,16,16);
            else if(mode==223) {
                program.code[pc++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(16u<<12);
                const unsigned branch=pc++;
                program.code[branch]=word(Op::IFC)|((branch+2u)<<10)|1u|(2u<<22)|(1u<<25);
                program.code[pc++]=arithmetic(Op::MUL,16,16,0);program.code[pc++]=word(Op::NOP);
            } else if(mode==224) {
                uniforms.i[0]={64,0,0,0};program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|8u;
                program.code[pc++]=word(Op::LOOP)|(2u<<10);
                program.code[pc++]=arithmetic(Op::MUL,16,16,0)|1u;
            }
            program.code[pc++]=word(Op::END);
            std::array<std::uint32_t,2> mask{};std::uint32_t capacity{},passes{};
            const bool excluded=mode>=220 && mode<=224;
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms,nullptr,&passes,&mask,&capacity)==
                  (excluded ? Metal::ShaderMachineRuntime::VertexExecution::Ordered:
                              Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences),
                  "Private recurrence proof accepted a dependent operand/control path or rejected exact scalar replay");
            if(!excluded) Check((mask[0]|mask[1]) && capacity && capacity<=64 &&
                (mode!=218 || passes==2),"Private recurrence mask/capacity/dependency depth is invalid");
            if(mode==226) Check(mask[0] && mask[1],"Mixed recurrence did not cover both mask words");
        }
        if(mode>=227 && mode<235) {
            // Twelve diamonds inside a two-iteration loop exhaust path enumeration.
            // Joins must preserve optional writes, inherited values, replay isolation,
            // actual output dependencies and both sides of condition-dependent control.
            program.code.fill(0);program.swizzle[1]=(Metal::Test::IdentitySwizzle&~15u)|8u;
            uniforms={};uniforms.i[0]={1,0,0,0};uniforms.f[0]={45,45,45,45};
            unsigned pc=1;const auto loop=pc++;unsigned last{};
            for(unsigned branch=0;branch<12;++branch) {
                const unsigned source=mode==231 ? 17:branch%2;
                program.code[pc++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(source<<12)|(32u<<7);
                const auto conditional=pc++;
                const bool both_arms=mode==234;
                program.code[conditional]=word(Op::IFC)|((pc+1u)<<10)|(both_arms ? 1u:0u)|(2u<<22)|(1u<<25);
                const unsigned dest=mode==229 ? 16:mode==233 ? 1:17;
                program.code[pc++]=arithmetic(mode==228 || mode==230 ? Op::ADD:Op::MOV,
                    dest,mode==228 || mode==230 ? dest:1,1)|1u;
                if(both_arms) program.code[pc++]=arithmetic(Op::MOV,17,0)|1u;
            }
            last=pc;program.code[pc++]=word(Op::NOP);
            program.code[0]=word(Op::NOP);program.code[loop]=word(Op::LOOP)|(last<<10);
            if(mode==232) {
                program.code[pc++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(17u<<12)|(32u<<7);
                program.code[pc]=word(Op::IFC)|((pc+2u)<<10)|(2u<<22)|(1u<<25);++pc;
                program.code[pc++]=arithmetic(Op::MOV,1,0);
            }
            if(mode==233) program.code[pc++]=arithmetic(Op::MOV,17,0);
            program.code[pc++]=arithmetic(Op::MOV,0,mode==230 || mode==233 ? 17:mode==234 || mode==231 || mode==232 ? 0:16);
            program.code[pc++]=arithmetic(Op::MOV,16,0);program.code[pc]=word(Op::END);
            std::uint32_t passes{};
            const auto actual=Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms,nullptr,&passes);
            using Execution=Metal::ShaderMachineRuntime::VertexExecution;
            const auto expected=mode==228 ? Execution::PrivateRecurrences:
                mode==230 || mode==231 || mode==232 ? Execution::Ordered:
                mode==233 ? Execution::MergeState:mode==234 ? Execution::Independent:Execution::DependencyPasses;
            Check(actual==expected,"Joined branch/loop proof lost a write, inherited value, recurrence or control dependency");
            if(mode==227 || mode==229) Check(passes==2,"Joined inherited-state proof did not preserve two GPU passes");
        }
        if(mode>=235) {
            program=recorded[mode-235].program;uniforms=recorded[mode-235].uniforms;
            std::cout<<"Recorded native shader case "<<mode-235<<" b="<<uniforms.b<<" execution="<<
                unsigned(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms))<<std::endl;
        }
        if(mode<14) Check(Metal::ShaderMachineRuntime::HasIndependentVertices(program,uniforms)==independent,
              "Native vertex independence proof misclassified temporary-state dependence");
        // The extra tag replays every proof/branch/loop/NaN program through a
        // full eight-vertex sequential group, including bounded-failure checks.
        for(unsigned index_tag:{0u,1u,2u,4u,8u}) {
            const unsigned index_bytes=index_tag==8 ? 0:index_tag;
            const auto elements=index_tag==8 ? 8u:
                mode>=14 && mode<142 ? (index_bytes ? 24u:8u):(index_bytes ? batch_elements:batch_vertices);
            const auto index_buffer=device.CreateBuffer(batch_elements*4,true);
            auto* index_data=static_cast<std::uint8_t*>(index_buffer.contents);
            std::vector<unsigned> ids(elements);
            for(unsigned element=0;element<elements;++element) {
                // First evict six entries, then repeatedly hit and evict wrapped slots.
                ids[element]=vertex_base+(index_bytes ? (element<70 ? element : (element*7)%90):element);
                if(index_bytes) for(unsigned byte=0;byte<index_bytes;++byte)
                    index_data[element*index_bytes+byte]=std::uint8_t(ids[element]>>(8*byte));
            }
            Metal::ShaderMachineState scalar{device},batched{device};
            Metal::ShaderMachineProgram seed{};seed.allow_emission=false;
            seed.swizzle[0]=Metal::Test::IdentitySwizzle;
            seed.code[0]=arithmetic(Op::MOV,0,0);seed.code[1]=word(Op::END);
            machine.Run(scalar,seed,uniforms,batch_input);machine.Run(batched,seed,uniforms,batch_input);
            if(mode>=142) {
                // Nonzero inactive banks and emission slots must survive a VS
                // commit, and must be cleared when Reset precedes the batch.
                auto snapshot=machine.SaveState(scalar);
                for(unsigned word=128;word<448;++word) {
                    const std::uint32_t bits=0x3e000000u+word*257u;
                    std::memcpy(snapshot.history.data()+word*4,&bits,4);
                }
                const std::uint32_t emitter_flags=15;
                std::memcpy(snapshot.history.data()+1820,&emitter_flags,4);
                machine.LoadState(scalar,snapshot);machine.LoadState(batched,snapshot);
            }
            if(mode>=142 && (mode&1u)) {
                auto snapshot=machine.SaveState(scalar);
                const std::uint32_t alternate_bank=1;
                std::memcpy(snapshot.history.data()+1816,&alternate_bank,sizeof(alternate_bank));
                machine.LoadState(scalar,snapshot);machine.LoadState(batched,snapshot);
            }
            if(mode>=166 && mode<202 && !(mode&1u)) { scalar.Reset();batched.Reset(); }
            if(mode>=202) {
                auto snapshot=machine.SaveState(scalar);
                constexpr std::array<std::uint32_t,8> bits{0x3f000001,0xbf400001,0,0x80000000,
                    0x7f800000,0xff800000,0x7fc00123,1};
                for(unsigned lane=0;lane<64;++lane) {
                    const auto value=bits[(lane+mode)%bits.size()];
                    std::memcpy(snapshot.history.data()+256+lane*4,&value,4);
                }
                machine.LoadState(scalar,snapshot);machine.LoadState(batched,snapshot);
            }
            if(mode==226 && !index_bytes) { scalar.Reset();batched.Reset(); }
            Metal::ShaderMachineSnapshot queued_initial{};
            if(index_tag==8) queued_initial=machine.SaveState(batched);
            std::vector<Metal::VertexRegisters> expected(elements);
            struct Entry { unsigned id{},element{};bool valid{}; };
            std::array<Entry,64> entries{};unsigned replacement=0;
            for(unsigned element=0;element<elements;++element) {
                bool hit=false;
                if(index_bytes) for(const auto& entry:entries) if(entry.valid && entry.id==ids[element]) {
                    expected[element]=expected[entry.element];hit=true;break;
                }
                if(hit) continue;
                auto source=batch_input;
                source.offset=(ids[element]-vertex_base)*sizeof(Metal::VertexRegisters);
                source.size=sizeof(Metal::VertexRegisters);
                const auto raw=machine.Run(scalar,program,uniforms,source);
                std::memcpy(&expected[element],raw.output.data,sizeof(Metal::VertexRegisters));
                if(index_bytes) { entries[replacement]={ids[element],element,true};replacement=(replacement+1)%64; }
            }
            const auto expected_state=machine.SaveState(scalar);
            if(index_tag==8) {
                // Every proof/arithmetic/control fixture also traverses the
                // specialized scalar queue. The synchronous oracle above keeps
                // the original generic GPU instruction/descriptor loads.
                Metal::ShaderMachineState queued{device};machine.LoadState(queued,queued_initial);
                auto queued_readback=device.CreateBuffer(elements*sizeof(Metal::VertexRegisters),true);
                for(unsigned element=0;element<elements;++element) {
                    auto source=batch_input;
                    source.offset=(ids[element]-vertex_base)*sizeof(Metal::VertexRegisters);
                    source.size=sizeof(Metal::VertexRegisters);
                    const auto result=machine.RunVertexDeferred(queued,program,uniforms,source);
                    Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,
                        queued_readback,element*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters));
                }
                scheduler.Finish();
                Check(!std::memcmp(queued_readback.contents,expected.data(),queued_readback.length) &&
                      machine.SaveState(queued)==expected_state,
                      "Specialized scalar queue changed full output/history against generic GPU oracle");
            }
            const Metal::BufferSlice batch_indices{index_buffer,0,index_buffer.length,index_buffer.contents,{}};
            const auto before_tick=scheduler.CurrentTick();
            const auto raw=machine.RunVertexBatch(batched,program,uniforms,batch_input,batch_vertices,elements,
                vertex_base,vertex_base,index_bytes ? batch_indices:Metal::BufferSlice{},index_bytes,1000000,index_bytes==2);
            if(index_bytes==2) Check(scheduler.CurrentTick()==before_tick,"Deferred vertex batch submitted/waited before its consumers");
            const auto readback=device.CreateBuffer(raw.size,true);
            Metal::TransferRuntime{device,scheduler}.CopyBuffer(raw.buffer,raw.offset,readback,0,raw.size);
            scheduler.Finish();
            Check(std::memcmp(readback.contents,expected.data(),raw.size)==0,"Batched vertex output/cache differs from scalar GPU execution");
            Check(machine.SaveState(batched)==expected_state,"Batched vertex history differs from scalar GPU execution");
            // A second dispatch must retain the final committed temporary/output bank.
            const auto next=machine.RunVertexBatch(batched,program,uniforms,batch_input,batch_vertices,1,vertex_base,vertex_base);
            const auto scalar_next=machine.Run(scalar,program,uniforms,batch_input);
            Metal::TransferRuntime{device,scheduler}.CopyBuffer(next.buffer,next.offset,readback,0,sizeof(Metal::VertexRegisters));
            scheduler.Finish();
            Check(std::memcmp(readback.contents,scalar_next.output.data,sizeof(Metal::VertexRegisters))==0,
                  "Batched vertex state did not survive another dispatch");
            Check(machine.SaveState(batched)==machine.SaveState(scalar),
                  "Resumed batched execution changed the complete scalar GPU state");
            const auto stable=machine.SaveState(batched);
            if(mode>=142) {
                bool bounded=false;
                try { machine.RunVertexBatch(batched,program,uniforms,batch_input,batch_vertices,2,
                    vertex_base,vertex_base,{},0,2); }
                catch(const Metal::ShaderMachineError& error) { bounded=error.Code()==6; }
                Check(bounded && machine.SaveState(batched)==stable,
                    "Failed parallel carry execution committed a partial unit state");
            }
            if(((mode>=166 && mode<199) || mode>=202) && index_bytes) {
                auto* bytes=static_cast<std::uint8_t*>(index_buffer.contents);
                const auto saved=std::vector<std::uint8_t>(bytes,bytes+index_bytes);
                std::memset(bytes,0,index_bytes);bool rejected_index=false;
                try { machine.RunVertexBatch(batched,program,uniforms,batch_input,batch_vertices,elements,
                    vertex_base,vertex_base,batch_indices,index_bytes); }
                catch(const Metal::ShaderMachineError& error) { rejected_index=error.Code()==8; }
                std::memcpy(bytes,saved.data(),index_bytes);
                Check(rejected_index && machine.SaveState(batched)==stable,
                      "Failed dependency plan committed a partial shader state");
            }
            auto invalid_program=program;invalid_program.code[mode>=235 ? program.entry:mode==6 ? 11:mode>=3 && mode<=9 ? 5:1]=word(static_cast<Op>(0x10));
            bool failed=false;
            try { machine.RunVertexBatch(batched,invalid_program,uniforms,batch_input,batch_vertices,2,vertex_base,vertex_base); }
            catch(const Metal::ShaderMachineError& error) { failed=error.Code()==2; }
            Check(failed && machine.SaveState(batched)==stable,"Failed vertex batch committed partial shader state");
        }
        }
        {
            // The compiler PC domain must cover both uniform outcomes, maximum
            // loop counts and exact ring-stack priority/overflow. Poison every
            // other PC so an incorrect target or boundary cannot silently pass.
            unsigned cases=0;
            for(unsigned mode=0;mode<9;++mode) {
                Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;
                p.code.fill(word(static_cast<Op>(0x10)));p.swizzle[0]=Metal::Test::IdentitySwizzle;
                std::bitset<4096> expected_domain;
                const auto pcs=[&](std::initializer_list<unsigned> list) {for(auto pc:list) expected_domain.set(pc);};
                if(mode==0) {
                    p.entry=128;p.code[128]=word(Op::CALL)|(16u<<10)|3u;p.code[129]=word(Op::END);
                    p.code[16]=arithmetic(Op::MOV,0,0);p.code[17]=word(Op::NOP);
                    p.code[18]=arithmetic(Op::MOV,16,1);pcs({16,17,18,128,129});
                } else if(mode==1 || mode==2) {
                    const unsigned depth=mode==1 ? 4:5;
                    for(unsigned i=0;i<depth;++i) p.code[i*10]=word(Op::CALL)|((i+1)*10u<<10)|1u;
                    p.code[depth*10]=arithmetic(Op::MOV,0,0);
                    const unsigned resume=mode==1 ? 11:21;
                    p.code[resume]=arithmetic(Op::MOV,1,1);p.code[resume+1]=word(Op::END);
                    for(unsigned i=0;i<=depth;++i) expected_domain.set(i*10);
                    pcs({resume,resume+1});
                } else if(mode==3) {
                    p.code[0]=word(Op::LOOP)|(5u<<10);p.code[1]=word(Op::IFU)|(6u<<10)|2u;
                    p.code[2]=word(Op::CALL)|(5u<<10)|1u;p.code[5]=arithmetic(Op::MOV,0,0);
                    p.code[6]=arithmetic(Op::MOV,1,1);p.code[7]=word(Op::NOP);p.code[8]=word(Op::END);
                    pcs({0,1,2,5,6,7,8});
                } else if(mode==4) {
                    p.code[0]=word(Op::LOOP)|(5u<<10);p.code[1]=word(Op::CALL)|(10u<<10)|3u;
                    p.code[2]=arithmetic(Op::MOV,1,1);p.code[3]=p.code[4]=word(Op::NOP);
                    p.code[5]=arithmetic(Op::MOV,16,2);p.code[6]=word(Op::END);
                    p.code[10]=word(Op::CMP)|(2u<<24)|(2u<<21)|(1u<<7);
                    p.code[11]=word(Op::BREAKC)|(2u<<22)|(1u<<25);p.code[12]=arithmetic(Op::MOV,0,0);
                    pcs({0,1,2,3,4,5,6,10,11,12});
                } else if(mode==5) {
                    p.entry=4093;p.code[4093]=word(Op::JMPU)|(4095u<<10);
                    p.code[4094]=word(Op::END);pcs({4093,4094,4095});
                } else if(mode==6) {
                    p.entry=4093;p.code[4093]=word(Op::CALL)|(4095u<<10)|1u;
                    p.code[4094]=word(Op::END);p.code[4095]=arithmetic(Op::MOV,0,0);
                    pcs({4093,4094,4095});
                } else if(mode==7) {
                    for(unsigned i=0;i<9;++i) p.code[i]=word(Op::IFU)|(20u<<10)|(i+1);
                    for(unsigned i=9;i<20;++i) p.code[i]=word(Op::NOP);
                    p.code[20]=word(Op::END);p.code[29]=arithmetic(Op::MOV,1,1);p.code[30]=word(Op::END);
                    for(unsigned i=0;i<=20;++i) expected_domain.set(i);
                    pcs({29,30});
                } else {
                    for(unsigned i=0;i<5;++i) p.code[i]=word(Op::LOOP)|(20u<<10);
                    p.code[5]=arithmetic(Op::MOV,0,0);
                    for(unsigned i=6;i<20;++i) p.code[i]=word(Op::NOP);
                    p.code[20]=arithmetic(Op::MOV,1,1);p.code[21]=word(Op::END);
                    for(unsigned i=0;i<=21;++i) expected_domain.set(i);
                }
                Check(Metal::ShaderMachineRuntime::ReachableVertexInstructions(p)==expected_domain,
                      "Compiler PC domain lost a boundary/ring path or included an unreachable poison PC");
                constexpr unsigned count=17,vertices=22,base=5;
                auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
                auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
                std::memset(values,0,input.length);
                for(unsigned i=0;i<vertices;++i) {
                    values[i][0]={float(i%3)*.5f,-.5f,2,1};
                    values[i][1]={.25f,1,0,2};values[i][2]={float(i)*.125f,1,2,3};
                }
                for(unsigned control=0;control<4;++control)
                for(unsigned reset=0;reset<2;++reset)
                for(unsigned width:{0u,2u}) {
                    Metal::ShaderMachineUniforms u{};u.b=control&1u;
                    u.i[0]={control&2u ? 255u:0u,3,2,0};
                    auto indices=device.CreateBuffer(count*2,true);
                    std::array<unsigned,count> ids{};
                    for(unsigned i=0;i<count;++i) {
                        ids[i]=base+(width ? (i*7)%11:i);
                        static_cast<unsigned short*>(indices.contents)[i]=ids[i];
                    }
                    Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                    if(!reset) {
                        for(unsigned i=0;i<456;++i) {
                            const unsigned bits=0x3e000000u+(i%64)*257u;
                            std::memcpy(initial.history.data()+i*4,&bits,4);
                        }
                        const std::array<unsigned,5> flags{0,1,0,control&1u,15};
                        std::memcpy(initial.history.data()+451*4,flags.data(),20);
                    }
                    Metal::ShaderMachineState scalar{device},batched{device};
                    machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                    std::vector<Metal::VertexRegisters> expected(count);
                    std::array<Metal::VertexRegisters,64> fifo{};
                    std::array<unsigned,64> keys{};keys.fill(UINT32_MAX);unsigned next=0,failure=0;
                    for(unsigned i=0;i<count;++i) {
                        unsigned hit=64;
                        if(width) for(unsigned slot=0;slot<64;++slot) if(keys[slot]==ids[i]) {hit=slot;break;}
                        if(hit<64) expected[i]=fifo[hit];
                        else try {
                            const auto result=machine.Run(scalar,p,u,
                                {input,(ids[i]-base)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
                            std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                            if(width) {keys[next]=ids[i];fifo[next]=expected[i];next=(next+1)%64;}
                        } catch(const Metal::ShaderMachineError& error) {failure=error.Code();break;}
                    }
                    Check(failure==(mode==5 && (control&1u) ? 2u:0u),"Scalar control-domain fixture took an unexpected error path");
                    if(failure) {
                        unsigned actual=0;
                        try {machine.RunVertexBatch(batched,p,u,{input,0,input.length,input.contents,{}},
                            vertices,count,base,base,width ? Metal::BufferSlice{indices,0,indices.length,indices.contents,{}}:
                            Metal::BufferSlice{},width);}
                        catch(const Metal::ShaderMachineError& error) {actual=error.Code();}
                        Check(actual==failure && machine.SaveState(batched)==initial,
                              "Reachable high-PC error changed native batch rollback");
                    } else {
                        const auto expected_state=machine.SaveState(scalar);
                        const auto result=machine.RunVertexBatch(batched,p,u,{input,0,input.length,input.contents,{}},
                            vertices,count,base,base,width ? Metal::BufferSlice{indices,0,indices.length,indices.contents,{}}:
                            Metal::BufferSlice{},width);
                        auto read=device.CreateBuffer(result.size,true);
                        Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                        scheduler.Finish();
                        Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                              machine.SaveState(batched)==expected_state,
                              "Control-domain specialization changed full scalar GPU outputs or retained state");
                    }
                    ++cases;
                }
            }
            std::cout<<cases<<" native control-domain cases preserve below-entry calls, fourth-return loss, CALL/IF/LOOP ring overflow, simultaneous boundary priority, conditional BREAK, 256-iteration loops, uniform-only changes, PC4095 and complete error rollback\n";
        }
        {
            // Sequential subset draws derive their predecessor on the GPU without
            // a plan buffer. Compare partial groups and UINT_MAX-ending ranges to
            // scalar GPU execution, including all retained bytes and reset state.
            unsigned cases=0;
            for(unsigned mode=0;mode<3;++mode)
            for(unsigned count:{1u,64u,65u,257u})
            for(unsigned reset=0;reset<2;++reset) {
                const unsigned vertices=count+5,base=reset ? UINT32_MAX-vertices+1u:5u;
                Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
                p.swizzle[0]=Metal::Test::IdentitySwizzle;
                if(mode==0) {
                    p.code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(1u<<7);
                    p.code[1]=word(Op::IFC)|(3u<<10)|(2u<<22)|(1u<<25);
                    p.code[2]=arithmetic(Op::MOV,16,0);
                    p.code[3]=arithmetic(Op::MOV,0,0);p.code[4]=word(Op::END);
                } else if(mode==1) {
                    p.code[0]=arithmetic(Op::MOV,0,16);
                    p.code[1]=arithmetic(Op::MOV,16,0);p.code[2]=word(Op::END);
                } else {
                    p.code[0]=arithmetic(Op::ADD,16,16,1);
                    p.code[1]=arithmetic(Op::MOV,0,0);p.code[2]=word(Op::END);
                }
                const auto expected_execution=mode==0 ? Metal::ShaderMachineRuntime::VertexExecution::MergeState:
                    mode==1 ? Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses:
                              Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences;
                Check(Metal::ShaderMachineRuntime::AnalyzeVertices(p)==expected_execution,
                      "Sequential subset test has the wrong native execution class");
                auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
                auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
                std::memset(values,0,input.length);
                for(unsigned i=0;i<vertices;++i) {
                    values[i][0]={float(i%37)*.125f,float(i%5),-.5f,1};
                    values[i][1]={.125f,float(i%3),-.25f,2};
                }
                Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                if(!reset) {
                    for(unsigned i=0;i<456;++i) {
                        const std::uint32_t bits=0x3e000000u+(i%64)*257u;
                        std::memcpy(initial.history.data()+i*4,&bits,4);
                    }
                    const std::array<std::uint32_t,5> flags{0,1,0,1,15};
                    std::memcpy(initial.history.data()+451*4,flags.data(),20);
                }
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                std::vector<Metal::VertexRegisters> expected(count);
                for(unsigned i=0;i<count;++i) {
                    const Metal::BufferSlice vertex{input,(i+5)*sizeof(Metal::VertexRegisters),
                        sizeof(Metal::VertexRegisters),nullptr,{}};
                    const auto result=machine.Run(scalar,p,{},vertex);
                    std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                }
                const auto expected_state=machine.SaveState(scalar);
                const auto tick=scheduler.CurrentTick();
                const auto result=machine.RunVertexBatch(batched,p,{},
                    {input,0,input.length,input.contents,{}},vertices,count,base,base+5,{},0,1000000,true);
                Check(scheduler.CurrentTick()==tick,"Sequential batch submitted before its consumers");
                auto read=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                scheduler.Finish();
                Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                      machine.SaveState(batched)==expected_state,
                      "Sequential subset output or complete state differs from scalar GPU execution");
                const auto stable_tick=scheduler.CurrentTick();bool invalid=false;
                try { machine.RunVertexBatch(batched,p,{},
                    {input,0,input.length,input.contents,{}},vertices,count,base,base+6); }
                catch(const std::invalid_argument&) { invalid=true; }
                Check(invalid && scheduler.CurrentTick()==stable_tick &&
                      machine.SaveState(batched)==expected_state,
                      "Invalid sequential range queued work or changed retained state");
                ++cases;
            }
            std::cout<<cases<<" native sequential subset batches preserve scalar GPU output and complete state, partial groups, UINT_MAX-ending ranges, deferred submission and invalid-range rollback\n";
        }
        {
            // Chain private outputs through deferred multi-dispatch batches and
            // transfer consumers without CPU waits between twelve submissions.
            // This exercises the bounded compute pass and its queue boundaries.
            constexpr unsigned count=65,batches=36;
            std::array<Metal::ShaderMachineProgram,3> programs{};
            for(unsigned mode=0;mode<programs.size();++mode) {
                auto& p=programs[mode];p.allow_emission=false;p.output_mask=3;
                p.swizzle[0]=Metal::Test::IdentitySwizzle;
                unsigned end=0;
                if(mode==0) {
                    p.code[end++]=word(Op::CMP)|(2u<<24)|(2u<<21)|(1u<<7);
                    p.code[end++]=word(Op::IFC)|(3u<<10)|(2u<<22)|(1u<<25);
                    p.code[end++]=arithmetic(Op::MOV,16,0);
                    p.code[end++]=arithmetic(Op::MOV,0,0);
                } else if(mode==1) {
                    p.code[end++]=arithmetic(Op::MOV,0,16);
                    p.code[end++]=arithmetic(Op::MOV,16,0);
                } else {
                    p.code[end++]=arithmetic(Op::ADD,16,16,1);
                    p.code[end++]=arithmetic(Op::MOV,0,0);
                }
                p.code[end++]=arithmetic(Op::MOV,1,1);p.code[end]=word(Op::END);
                Check(Metal::ShaderMachineRuntime::AnalyzeVertices(p)==
                    (mode==0 ? Metal::ShaderMachineRuntime::VertexExecution::MergeState:
                     mode==1 ? Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses:
                               Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences),
                      "Deferred chain has the wrong native execution class");
            }
            std::vector<Metal::VertexRegisters> source(count);
            for(unsigned i=0;i<count;++i) {
                source[i][0]={float(i%13)*.125f,float(i%5),-.5f,1};
                source[i][1]={.125f,.25f,-.125f,2};
            }
            const auto initial=source;
            std::array<std::vector<Metal::VertexRegisters>,batches> expected;
            Metal::ShaderMachineState scalar{device},batched{device};
            for(unsigned batch=0;batch<batches;++batch) {
                const unsigned width=std::array{0u,1u,2u,4u}[batch%4];
                std::array<unsigned,count> fifo{};unsigned size=0;
                expected[batch].resize(count);
                auto input=device.CreateBuffer(count*sizeof(Metal::VertexRegisters),true);
                std::memcpy(input.contents,source.data(),input.length);
                for(unsigned element=0;element<count;++element) {
                    const unsigned id=width ? (element*11+batch)%49:element;
                    unsigned cached=~0u;
                    for(unsigned slot=0;slot<size;++slot) if(fifo[slot]==id) cached=slot;
                    if(cached!=~0u) { expected[batch][element]=expected[batch][cached];continue; }
                    const auto result=machine.Run(scalar,programs[batch%3],{},
                        {input,id*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
                    std::memcpy(&expected[batch][element],result.output.data,sizeof(Metal::VertexRegisters));
                    // This fixture has at most 49 unique IDs and sequential first misses.
                    if(width) fifo[size++]=id;
                }
                source=expected[batch];
            }
            const auto expected_state=machine.SaveState(scalar);
            auto upload=device.CreateBuffer(initial.size()*sizeof(Metal::VertexRegisters),true);
            std::memcpy(upload.contents,initial.data(),upload.length);
            Metal::BufferSlice previous{upload,0,upload.length,nullptr,{}};
            std::array<id<MTLBuffer>,batches> readbacks{};
            for(unsigned batch=0;batch<batches;++batch) {
                const unsigned width=std::array{0u,1u,2u,4u}[batch%4];
                auto indices=device.CreateBuffer(std::max(count*width,1u),true);
                for(unsigned element=0;element<count;++element) for(unsigned byte=0;byte<width;++byte)
                    static_cast<std::uint8_t*>(indices.contents)[element*width+byte]=
                        std::uint8_t(((element*11+batch)%49)>>(byte*8));
                const auto tick=scheduler.CurrentTick();
                previous=machine.RunVertexBatch(batched,programs[batch%3],{},previous,count,count,0,0,
                    {indices,0,indices.length,nullptr,{}},width,1000000,true);
                Check(scheduler.CurrentTick()==tick,"Deferred chain submitted within a bounded vertex pass");
                readbacks[batch]=device.CreateBuffer(previous.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(previous.buffer,previous.offset,
                    readbacks[batch],0,previous.size);
                if(batch%3==2) scheduler.Submit();
            }
            upload=nil;previous={};scheduler.Finish();
            for(unsigned batch=0;batch<batches;++batch)
                Check(std::memcmp(readbacks[batch].contents,expected[batch].data(),readbacks[batch].length)==0,
                      "Deferred private-output chain changed scalar GPU outputs across encoder/slot boundaries");
            Check(machine.SaveState(batched)==expected_state,
                  "Deferred private-output chain changed complete scalar GPU state");
            std::cout<<"36 deferred native vertex batches chain private outputs across three execution classes, all index widths, transfer consumers and twelve triple-slot submissions with exact scalar GPU outputs/state\n";
        }
        {
            // Three dependent passes must preserve LIT condition flags used by
            // later control, including when its destination is an output register.
            // Keep the complete-state reference ready for seed-pass optimizations.
            Metal::ShaderMachineProgram p{};p.allow_emission=false;
            p.swizzle[0]=Metal::Test::IdentitySwizzle;
            p.code[0]=arithmetic(Op::MOV,0,17);
            p.code[1]=arithmetic(Op::MOV,17,16);
            p.code[2]=arithmetic(Op::MOV,16,0);
            p.code[3]=arithmetic(Op::LIT,1,1);
            p.code[4]=word(Op::IFC)|(6u<<10)|(2u<<22)|(1u<<25);
            p.code[5]=arithmetic(Op::MOV,18,2);p.code[6]=word(Op::END);
            unsigned passes=0;
            Check(Metal::ShaderMachineRuntime::AnalyzeVertices(p,{},nullptr,&passes)==
                      Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses && passes==3,
                  "Seed-output test does not exercise two intermediate dependency passes");
            constexpr unsigned count=129,vertices=192,base=5;
            auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
            auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
            for(unsigned i=0;i<vertices;++i) {
                values[i][0]={float(i)*.125f,-.5f,2,1};
                values[i][1]={i%2 ? -.25f:.5f,140,0,i%3 ? 1.0f:-1.0f};
                values[i][2]={float(i%7),1,2,3};
                for(unsigned reg=3;reg<16;++reg) values[i][reg]={};
            }
            unsigned cases=0;
            for(unsigned reset=0;reset<2;++reset)
            for(unsigned bank=0;bank<2;++bank)
            for(unsigned width:{0u,1u,2u,4u}) {
                auto indices=device.CreateBuffer(count*4,true);
                std::array<unsigned,count> ids{};
                for(unsigned i=0;i<count;++i) {
                    ids[i]=width ? base+(i<80 ? i:i%3 ? (i*7)%vertices:(i-1)%vertices):base+i;
                    std::memcpy(static_cast<std::byte*>(indices.contents)+i*width,&ids[i],width);
                }
                Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                if(!reset) {
                    for(unsigned i=0;i<456;++i) {
                        const unsigned bits=0x3e000000u+(i%64)*257u;
                        std::memcpy(initial.history.data()+i*4,&bits,4);
                    }
                    const std::array<unsigned,5> flags{0,1,0,bank,15};
                    std::memcpy(initial.history.data()+451*4,flags.data(),20);
                }
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                std::vector<Metal::VertexRegisters> expected(count);
                std::array<Metal::VertexRegisters,64> fifo{};
                std::array<unsigned,64> keys{};keys.fill(UINT32_MAX);unsigned next=0,hits=0,misses=0;
                for(unsigned i=0;i<count;++i) {
                    unsigned hit=64;
                    if(width) for(unsigned slot=0;slot<64;++slot) if(keys[slot]==ids[i]) {hit=slot;break;}
                    if(hit<64) {expected[i]=fifo[hit];++hits;}
                    else {
                        ++misses;
                        const auto result=machine.Run(scalar,p,{},
                            {input,(ids[i]-base)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}});
                        std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                        if(width) {keys[next]=ids[i];fifo[next]=expected[i];next=(next+1)%64;}
                    }
                }
                Check(!width || (hits && misses>64),"Seed-output index sequence lacks FIFO hits/eviction");
                const auto expected_state=machine.SaveState(scalar);
                const auto result=machine.RunVertexBatch(batched,p,{},
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,
                    width ? Metal::BufferSlice{indices,0,count*width,indices.contents,{}}:Metal::BufferSlice{},width);
                auto read=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                scheduler.Finish();
                if(std::memcmp(read.contents,expected.data(),read.length)!=0 ||
                   machine.SaveState(batched)!=expected_state) {
                    std::cerr<<"Seed case reset="<<reset<<" bank="<<bank<<" index="<<width<<'\n';
                    const auto* actual=static_cast<const unsigned*>(read.contents);
                    const auto* reference=reinterpret_cast<const unsigned*>(expected.data());
                    for(unsigned i=0;i<read.length/4;++i) if(actual[i]!=reference[i]) {
                        std::cerr<<"output word "<<i<<" actual="<<actual[i]<<" expected="<<reference[i]<<'\n';break;
                    }
                    const auto snapshot=machine.SaveState(batched);
                    for(unsigned i=0;i<1824;++i) if(snapshot.history[i]!=expected_state.history[i]) {
                        std::cerr<<"state byte "<<i<<'\n';break;
                    }
                }
                Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                      machine.SaveState(batched)==expected_state,
                      "Three-pass native execution changed LIT flags, conditional carries, full outputs or retained banks");
                bool failed=false;
                try {machine.RunVertexBatch(batched,p,{},
                    {input,0,input.length,input.contents,{}},vertices,count,base,base,{},0,2);}
                catch(const Metal::ShaderMachineError& error) {failed=error.Code()==6;}
                Check(failed && machine.SaveState(batched)==expected_state,
                      "Failed intermediate seed pass committed retained state");
                ++cases;
            }
            std::cout<<cases<<" three-pass retained-state cases preserve LIT flags/control, conditional carries, full outputs, both banks, all index widths, FIFO hits/eviction and failed-pass rollback\n";
        }
        {
            // Uniform-only changes keep the same MergeState pipeline but change
            // which output/temporary/address lanes are always or never written.
            // Alternate proofs with retained bank-one state and FIFO hits whose
            // source is not the final actual miss. Compare the complete GPU unit.
            Metal::ShaderMachineProgram p{};p.allow_emission=false;
            p.swizzle[0]=Metal::Test::IdentitySwizzle;
            for(unsigned i=1;i<7;++i) p.swizzle[i]=(Metal::Test::IdentitySwizzle&~15u)|
                std::array<unsigned,6>{9,6,12,3,10,8}[i-1];
            p.code[0]=word(Op::CMP)|(2u<<24)|(2u<<21)|(1u<<7);
            p.code[1]=word(Op::IFC)|(3u<<10)|(2u<<22)|(1u<<25);
            p.code[2]=arithmetic(Op::MOV,16,0);
            p.code[3]=word(Op::CALLU)|(16u<<10)|3u;
            p.code[4]=word(Op::CALLU)|(32u<<10)|3u|(1u<<22);
            p.code[5]=word(Op::END);
            p.code[16]=arithmetic(Op::MOV,0,0)|1u;
            p.code[17]=arithmetic(Op::MOV,31,1)|2u;
            p.code[18]=arithmetic(Op::MOVA,0,0)|3u;
            p.code[32]=arithmetic(Op::MOV,15,1)|4u;
            p.code[33]=arithmetic(Op::MOV,17,0)|5u;
            p.code[34]=arithmetic(Op::MOVA,0,1)|6u;
            constexpr unsigned vertices=90,base=7;
            auto input=device.CreateBuffer(vertices*sizeof(Metal::VertexRegisters),true);
            auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
            std::memset(values,0,input.length);
            for(unsigned i=0;i<vertices;++i) {
                values[i][0]={float(i%5),float(i%3),-.5f,1};
                values[i][1]={2,float(i%7),-.25f,2};
            }
            unsigned cases=0;
            for(unsigned dependency=0;dependency<2;++dependency) {
            if(dependency) {
                // Reading r0 before its conditional write forces a second native
                // pass. This mixes conditional, definite and untouched seeds.
                for(unsigned pc=6;pc>0;--pc) p.code[pc]=p.code[pc-1];
                p.code[0]=arithmetic(Op::MOV,2,16);
                p.code[2]=word(Op::IFC)|(4u<<10)|(2u<<22)|(1u<<25);
            }
            for(unsigned reset=0;reset<2;++reset) for(unsigned width:{0u,1u,2u,4u}) {
                Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                if(!reset) {
                    for(unsigned i=0;i<456;++i) {
                        const std::uint32_t bits=0x3e000000u+(i%64)*257u;
                        std::memcpy(initial.history.data()+i*4,&bits,4);
                    }
                    const std::array<std::uint32_t,5> flags{0,1,0,1,15};
                    std::memcpy(initial.history.data()+451*4,flags.data(),20);
                }
                Metal::ShaderMachineState scalar{device},batched{device};
                machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
                std::vector<unsigned> order;
                for(unsigned i=0;i<(width ? 133u:85u);++i)
                    order.push_back(base+(width ? (i<70 ? i:(i*7)%90):i+2));
                if(width) order.insert(order.end(),{order.back(),order[order.size()-2]});
                auto indices=device.CreateBuffer(order.size()*std::max(width,1u),true);
                if(width) for(unsigned i=0;i<order.size();++i) for(unsigned byte=0;byte<width;++byte)
                    static_cast<std::uint8_t*>(indices.contents)[i*width+byte]=std::uint8_t(order[i]>>(byte*8));
                for(unsigned selection:{0u,1u,2u,3u,2u,0u,3u,1u}) {
                    Metal::ShaderMachineUniforms selected{};selected.b=selection;
                    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(p,selected)==
                        (dependency ? Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses:
                                      Metal::ShaderMachineRuntime::VertexExecution::MergeState),
                        "Uniform write-proof case changed execution class");
                    std::vector<Metal::VertexRegisters> expected(order.size());
                    std::array<std::pair<unsigned,unsigned>,64> fifo{};unsigned size=0,replacement=0;
                    for(unsigned i=0;i<order.size();++i) {
                        unsigned cached=~0u;
                        if(width) for(unsigned slot=0;slot<size;++slot)
                            if(fifo[slot].first==order[i]) cached=fifo[slot].second;
                        if(cached!=~0u) { expected[i]=expected[cached];continue; }
                        const Metal::BufferSlice vertex{input,(order[i]-base)*sizeof(Metal::VertexRegisters),
                            sizeof(Metal::VertexRegisters),nullptr,{}};
                        const auto result=machine.Run(scalar,p,selected,vertex);
                        std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                        if(width) { fifo[replacement]={order[i],i};replacement=(replacement+1u)&63u;size=std::min(size+1u,64u); }
                    }
                    const auto result=machine.RunVertexBatch(batched,p,selected,
                        {input,0,input.length,input.contents,{}},vertices,unsigned(order.size()),base,base+2,
                        width ? Metal::BufferSlice{indices,0,indices.length,indices.contents,{}}:Metal::BufferSlice{},width);
                    auto read=device.CreateBuffer(result.size,true);
                    Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,result.offset,read,0,read.length);
                    scheduler.Finish();
                    Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                        machine.SaveState(batched)==machine.SaveState(scalar),
                        "Uniform-only write-proof transition changed full scalar GPU output/state");
                    ++cases;
                }
            }
            }
            std::cout<<cases<<" uniform-only write-proof transitions through native merge/dependency pipelines preserve every output and all state bytes, masked first/last registers, addresses, both banks, FIFO hits/eviction and all index widths\n";
        }
        {
            // Force full collision clusters, wraparound deletion and UINT_MAX IDs.
            // The reference is scalar GPU execution plus the contractual FIFO,
            // not another implementation of the accelerated hash table.
            constexpr unsigned count=16384;
            auto input=device.CreateBuffer(count*sizeof(Metal::VertexRegisters),true);
            auto* values=static_cast<Metal::VertexRegisters*>(input.contents);
            std::memset(values,0,input.length);
            for(unsigned i=0;i<count;++i) { values[i][0].fill(float(i));values[i][1].fill(.125f); }
            Metal::BufferSlice source{input,0,input.length,input.contents,{}};
            Metal::ShaderMachineProgram p{};p.allow_emission=false;p.swizzle[0]=Metal::Test::IdentitySwizzle;
            p.code[0]=arithmetic(Op::ADD,16,16,1);p.code[1]=arithmetic(Op::MOV,0,0);p.code[2]=word(Op::END);
            for(unsigned test=0;test<3;++test) {
                const unsigned base=test==2 ? UINT32_MAX-count+1u:0u,width=test==2 ? 4u:2u;
                const unsigned bucket=test==0 ? 0:test==1 ? 127:((UINT32_MAX*0x9e3779b9u)>>25);
                std::vector<unsigned> candidates,order;
                for(unsigned i=0;i<count && candidates.size()<90;++i)
                    if(((std::uint32_t(base+i)*0x9e3779b9u)>>25)==bucket) candidates.push_back(base+i);
                Check(candidates.size()==90,"Collision test did not find enough valid vertex IDs");
                for(unsigned i=0;i<320;++i) order.push_back(candidates[i<70 ? i:(i*7)%90]);
                if(test==2) order.insert(order.end(),{UINT32_MAX,UINT32_MAX,candidates[0]});
                auto indices=device.CreateBuffer(order.size()*width,true);
                for(unsigned i=0;i<order.size();++i) for(unsigned byte=0;byte<width;++byte)
                    static_cast<std::uint8_t*>(indices.contents)[i*width+byte]=std::uint8_t(order[i]>>(byte*8));
                Metal::ShaderMachineState scalar{device},batched{device};
                std::array<std::pair<unsigned,unsigned>,64> fifo{};unsigned size=0,replacement=0;
                std::vector<Metal::VertexRegisters> expected(order.size());
                for(unsigned i=0;i<order.size();++i) {
                    unsigned cached=~0u;
                    for(unsigned slot=0;slot<size;++slot) if(fifo[slot].first==order[i]) cached=fifo[slot].second;
                    if(cached!=~0u) { expected[i]=expected[cached];continue; }
                    auto vertex=source;vertex.offset=std::size_t(order[i]-base)*sizeof(Metal::VertexRegisters);
                    vertex.size=sizeof(Metal::VertexRegisters);
                    const auto result=machine.Run(scalar,p,{},vertex);
                    std::memcpy(&expected[i],result.output.data,sizeof(Metal::VertexRegisters));
                    fifo[replacement]={order[i],i};replacement=(replacement+1u)&63u;size=std::min(size+1u,64u);
                }
                const auto result=machine.RunVertexBatch(batched,p,{},source,count,unsigned(order.size()),base,base,
                    {indices,0,indices.length,indices.contents,{}},width);
                auto read=device.CreateBuffer(result.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(result.buffer,0,read,0,result.size);scheduler.Finish();
                Check(std::memcmp(read.contents,expected.data(),read.length)==0 &&
                    machine.SaveState(batched)==machine.SaveState(scalar),"Collision/FIFO deletion changed vertex outputs or full native state");
            }
            std::cout<<"Three adversarial native FIFO collision/wraparound/UINT_MAX programs preserve every output and complete scalar GPU state\n";
        }
        {
            // One immutable program selects all four execution classes solely through
            // boolean uniforms. Revisit each compiled pipeline with retained unit state.
            Metal::ShaderMachineProgram variants{};variants.allow_emission=false;
            variants.swizzle[0]=Metal::Test::IdentitySwizzle;
            for(unsigned branch=0;branch<4;++branch) {
                const unsigned target=16+branch*16;
                variants.code[branch]=word(Op::CALLU)|(target<<10)|4u|(branch<<22);
                for(unsigned i=0;i<4;++i) variants.code[target+i]=word(Op::NOP);
            }
            variants.code[4]=word(Op::END);
            variants.code[16]=arithmetic(Op::ADD,16,16,0);
            variants.code[17]=arithmetic(Op::MOV,0,16);
            variants.code[32]=arithmetic(Op::MOV,16,0);
            variants.code[33]=arithmetic(Op::MOV,0,16);
            variants.code[48]=word(Op::CMP)|(2u<<24)|(2u<<21)|(1u<<7);
            variants.code[49]=word(Op::IFC)|(51u<<10)|(2u<<22)|(1u<<25);
            variants.code[50]=arithmetic(Op::MOV,16,0);
            variants.code[64]=arithmetic(Op::MOV,0,16);
            variants.code[65]=arithmetic(Op::MOV,16,0);
            Metal::ShaderMachineState scalar{device},batched{device};
            for(unsigned selection:{1u,0u,2u,3u,1u,3u,2u,0u}) {
                uniforms={};uniforms.b=1u<<selection;
                const auto expected_execution=selection==0 ? Metal::ShaderMachineRuntime::VertexExecution::Ordered:
                    selection==1 ? Metal::ShaderMachineRuntime::VertexExecution::Independent:
                    selection==2 ? Metal::ShaderMachineRuntime::VertexExecution::MergeState:
                                   Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses;
                Check(Metal::ShaderMachineRuntime::AnalyzeVertices(variants,uniforms)==expected_execution,
                      "Uniform-selected program has the wrong vertex execution class");
                std::array<Metal::VertexRegisters,3> expected{};
                for(unsigned element=0;element<3;++element) {
                    auto source=batch_input;source.offset=element*sizeof(Metal::VertexRegisters);
                    source.size=sizeof(Metal::VertexRegisters);
                    const auto raw=machine.Run(scalar,variants,uniforms,source);
                    std::memcpy(&expected[element],raw.output.data,sizeof(Metal::VertexRegisters));
                }
                const auto raw=machine.RunVertexBatch(batched,variants,uniforms,batch_input,
                    batch_vertices,3,vertex_base,vertex_base);
                const auto readback=device.CreateBuffer(raw.size,true);
                Metal::TransferRuntime{device,scheduler}.CopyBuffer(raw.buffer,raw.offset,readback,0,raw.size);
                scheduler.Finish();
                Check(std::memcmp(readback.contents,expected.data(),raw.size)==0 &&
                      machine.SaveState(batched)==machine.SaveState(scalar),
                      "Cached execution-class transition changed native output or retained state");
            }
            std::cout<<"One immutable program revisits all four specialized execution classes with uniform-only changes and exact retained GPU state\n";
        }
        {
            // Exceed all three parameter pages without explicit per-draw submits
            // or waits. Unique constants detect premature page reuse; completed
            // status callbacks must release their CPU leases before recycling.
            constexpr unsigned draws=4096;
            static_assert(draws*sizeof(Metal::ShaderMachineUniforms)>3*(1u<<20));
            Metal::Scheduler pressure_scheduler{device};
            Metal::ShaderMachineRuntime pressure_machine{device,pressure_scheduler};
            Metal::ShaderMachineState pressure_state{device},scalar_state{device};
            Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
            p.swizzle[0]=Metal::Test::IdentitySwizzle;
            p.code[0]=arithmetic(Op::MOV,0,32);p.code[1]=word(Op::END);
            Metal::ShaderMachineUniforms constants{};
            auto source=device.CreateBuffer(sizeof(Metal::VertexRegisters),true);
            std::memset(source.contents,0,source.length);
            const Metal::BufferSlice input{source,0,source.length,source.contents,{}};
            auto read=device.CreateBuffer(draws*sizeof(Metal::VertexRegisters),true);
            const auto first_tick=pressure_scheduler.CurrentTick();
            for(unsigned draw=0;draw<draws;++draw) {
                constants.f[0]={float(draw),-.5f,float(draw%17)*.25f,1};
                const auto output=pressure_machine.RunVertexBatch(pressure_state,p,constants,input,
                    1,1,0,0,{},0,1000000,true);
                Metal::TransferRuntime{device,pressure_scheduler}.CopyBuffer(output.buffer,output.offset,
                    read,draw*sizeof(Metal::VertexRegisters),output.size);
            }
            Check(pressure_scheduler.CurrentTick()>first_tick,
                  "Parameter-page pressure failed to exercise submission and recycling");
            pressure_scheduler.Finish();
            for(unsigned draw=0;draw<draws;++draw) {
                Metal::VertexRegisters expected{};
                expected[0]={float(draw),-.5f,float(draw%17)*.25f,1};
                Check(std::memcmp(static_cast<const std::byte*>(read.contents)+draw*sizeof(expected),
                    expected.data(),sizeof(expected))==0,"Parameter-page reuse overwrote deferred constants or output");
            }
            pressure_machine.Run(scalar_state,p,constants,input);
            Check(pressure_machine.SaveState(pressure_state)==pressure_machine.SaveState(scalar_state),
                  "Parameter-page reuse changed complete scalar GPU state");
            std::cout<<draws<<" deferred native draws recycle bounded upload pages with unique constants, every output and complete scalar GPU state\n";
        }
        for(unsigned failure_draw:{127u,767u}) {
            Metal::Scheduler pressure_scheduler{device};
            Metal::ShaderMachineRuntime pressure_machine{device,pressure_scheduler};
            Metal::ShaderMachineState pressure_state{device};
            Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
            p.swizzle[0]=Metal::Test::IdentitySwizzle;
            p.code[0]=arithmetic(Op::MOV,0,0);p.code[1]=word(Op::END);
            bool failed=false,sticky=false;unsigned queued=0;
            try {
                for(;queued<4096;++queued)
                    pressure_machine.RunVertexBatch(pressure_state,p,{},slice,1,1,0,0,{},0,
                        queued==failure_draw ? 1:1000000,true);
                pressure_scheduler.Finish();
            } catch(const Metal::ShaderMachineError& error) {
                failed=error.Code()==6 && std::string(error.what())==
                    "Native PICA shader machine error 6 at instruction 1";
            }
            try { pressure_scheduler.IsComplete(0); }
            catch(const Metal::ShaderMachineError& error) { sticky=error.Code()==6; }
            Check(failed && sticky && queued>failure_draw && queued<4096,
                  "Parameter-page pressure lost a deferred error or its sticky validation state");
        }
        std::cout<<"Deferred failures on two upload pages survive later successful draws and bounded-page recycling with exact code/PC and sticky errors\n";
        {
            Metal::Scheduler failed_scheduler{device};
            Metal::ShaderMachineRuntime failed_machine{device,failed_scheduler};
            Metal::ShaderMachineState failed_state{device};
            auto invalid=program;invalid.code[0]=word(static_cast<Op>(0x10));
            failed_machine.RunVertexBatch(failed_state,invalid,uniforms,batch_input,batch_vertices,2,
                vertex_base,vertex_base,{},0,1000000,true);
            bool failed=false,sticky=false;
            try { failed_scheduler.Finish(); }
            catch(const Metal::ShaderMachineError& error) { failed=error.Code()==2; }
            try { failed_scheduler.IsComplete(0); }
            catch(const Metal::ShaderMachineError& error) { sticky=error.Code()==2; }
            Check(failed && sticky,"Deferred GPU shader failure was lost after completion");
        }
        std::cout<<"Deferred native GPU validation: no per-draw submit/wait, ordered readback and sticky shader errors passed\n";
        std::cout<<"33 acyclic retained-register programs (2-17 GPU passes, conditional lane writes and MOVA), three cycle/control exclusions, all index widths and cache eviction, complete state and bounded/index failure rollback passed\n";
        std::cout<<"Eight branch/loop join programs preserve optional/definite writes, isolated/observable recurrence and control exclusions, all index widths, complete native state and failure rollback\n";
        std::cout<<"20 private recurrence programs (including both mask words with ordinary writes and reset), five unsafe recurrence exclusions and "<<recorded.size()<<
            " recorded shader/control cases: complete bitwise scalar GPU state/output, all index widths, FIFO eviction, sign/NaN/zero/reset/loop semantics and failure rollback passed\n";
        std::cout<<"8 live/dead inherited-component programs distinguish overwritten intermediates from observable output dependencies and preserve complete GPU state\n";
        std::cout<<"16 conditional component-write programs preserve outputs, temporaries, MOVA/LOOP addresses, flags, both output banks, cache misses/hits, resumed state and failure rollback through GPU state merging\n";
        std::cout<<"128 deterministic masked/swizzled arithmetic programs compared against ordered scalar GPU outputs and complete states across all index widths\n";
        std::cout<<"Batched native vertex machine: exact scalar GPU output/state in ordered/parallel modes, partial output writes, CALL/IFU/IFC/LOOP and simultaneous boundary priority, conditional component/state merging including address/flags and both output banks, per-component swizzled dependencies/DP3 versus DP4, retained temporaries, 64-entry cache hits/eviction, all four index widths, resumed state and failure rollback passed\n";
        std::cout<<operations<<" native GPU arithmetic operations, "<<conditions<<" CMP/IFC combinations, inverted operations and non-fused MAD/MADI, persistent registers, swizzles/masks, CALL/IFU/LOOP and boundary-jump priority, BREAK/BREAKC, MOVA/uniform bounds, zero*infinity, GS EMIT/SETEMIT/winding/two banks/reused slots/compact outputs and atomic bounded failures passed\n";
        std::cout<<"Native shader snapshots: recreated GS banks/slots resume and malformed/reset payload rollback passed\n";
        std::cout<<"Native DSTI swapped sources/relative negated uniforms, CMP6/7, CALL4/IF8/LOOP4 ring overflow, fourth-return quirk and independent simultaneous stack priority passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }
}
