// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_shader_runtime.h"
#include "mtl_shader_machine.h"
#include "mtl_transfer.h"
#include "mtl_pica_vertex_fixture.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <bit>
#include <cstring>
#include <iostream>
#include <array>
#include <limits>
#include <thread>
void Check(bool yes,const char* message) {if(!yes)throw std::runtime_error(message);}
void ValidateNativeSelection(Metal::ShaderMachineRuntime& machine) {
    unsigned checks=0;
    Metal::ShaderMachineUniforms u{};
    Metal::GeometryEmissionConfig c{};c.attribute_count=1;c.semantics[0]=0x03020100;c.honor_winding=0;
    const auto fresh=[] {Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;p.swizzle[0]=Metal::Test::IdentitySwizzle;p.code[0]=0x13u<<26;p.code[1]=0x22u<<26;return p;};
    const auto check=[&](const char* name,const Metal::ShaderMachineProgram& p,const Metal::ShaderMachineUniforms& values,const Metal::GeometryEmissionConfig& config,bool expected) {
        const auto actual=machine.CanCaptureNativeReset(p,values,config);++checks;
        if(actual!=expected) throw std::runtime_error(std::string("Native selection: ")+name);
        Check(machine.CanCaptureNativeReset(p,values,config)==actual,"Cached native selection changed");
    };
    auto p=fresh();check("complete direct input",p,u,c,true);
    p.code[0]|=16u<<12;check("read-only reset-zero temporary",p,u,c,true);
    p=fresh();p.code[0]=(16u<<21)|(16u<<12);p.code[1]=(0x13u<<26)|(16u<<12);p.code[2]=0x22u<<26;
    check("observable retained recurrence",p,u,c,false);
    p.code[1]=0x13u<<26;check("unobservable retained recurrence",p,u,c,true);
    p=fresh();p.code[0]=(0x27u<<26)|(2u<<10);p.code[1]=0x13u<<26;p.code[2]=0x22u<<26;
    check("skipped output",p,u,c,false);u.b=1;check("uniform definite output",p,u,c,true);u.b=0;
    p.code[0]=(0x27u<<26)|(2u<<10)|1;p.code[2]=0x13u<<26;p.code[3]=0x22u<<26;
    check("both uniform paths complete",p,u,c,true);u.b=1;check("opposite uniform path complete",p,u,c,true);u.b=0;
    p=fresh();p.swizzle[0]&=~1u;check("missing live output w",p,u,c,false);
    auto xy=c;xy.semantics[0]=0x1f1f0100;check("unmapped output w",p,u,xy,true);
    p=fresh();auto reserved=c;reserved.semantics[0]|=0xe0e0e0e0u;check("reserved semantic bits mask",p,u,reserved,true);
    auto invalid=c;invalid.semantics[0]=0x18181818;check("unknown used semantic",p,u,invalid,false);
    invalid=c;invalid.attribute_count=8;check("too many attributes",p,u,invalid,false);
    invalid=c;invalid.fix_quaternions=1;check("per primitive operation excluded",p,u,invalid,false);
    invalid=c;invalid.honor_winding=1;check("GS winding excluded",p,u,invalid,false);
    invalid=c;invalid.enable_clip=2;check("invalid flag",p,u,invalid,false);
    auto surplus=c;surplus.attribute_count=2;surplus.semantics[1]=0x03020100;
    check("surplus metadata clamped to compact output count",p,u,surplus,true);
    p=fresh();p.output_mask=3;p.code[0]=(16u<<21)|(16u<<12);p.code[1]=(0x13u<<26);p.code[2]=(0x13u<<26)|(1u<<21)|(16u<<12);p.code[3]=0x22u<<26;
    check("unmapped unstable output register",p,u,c,true);
    surplus=c;surplus.attribute_count=2;surplus.semantics[1]=0x07060504;
    check("mapped unstable output register",p,u,surplus,false);
    p=fresh();p.code[0]=0x12u<<26;p.code[1]=(0x13u<<26)|(1u<<19)|(32u<<12);p.code[2]=0x22u<<26;
    check("same vertex relative uniform address",p,u,c,true);
    p=fresh();p.code[0]=(0x2eu<<26)|(6u<<21);p.code[1]=0x13u<<26;p.code[2]=0x22u<<26;
    check("CMP reserved mode native exclusion",p,u,c,false);
    p=fresh();p.allow_emission=true;check("GS native exclusion",p,u,c,false);
    // Bounded cache replacement must not mix controls or semantic maps.
    for(unsigned i=0;i<140;++i) {u.b=i;check("control cache replacement",fresh(),u,c,true);}
    u={};p=fresh();check("original source survives cache replacement",p,u,c,true);
    std::cout<<"Native reset selection: "<<checks<<" shared-dataflow/metadata/control/cache checks passed\n";
}
int ValidateNativeFlow() {
 unsigned checks=0,failures=0;
 const auto fresh=[] {Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;p.code.fill(0x84000000);p.code[0]=0x88000000;return p;};
 const auto check=[&](const char* name,const Metal::ShaderMachineProgram& p,bool expected) {
  const bool actual=Metal::ShaderMachineRuntime::HasNativeResetControlFlow(p);++checks;
  if(actual!=expected){++failures;std::cerr<<name<<" actual="<<actual<<" expected="<<expected<<'\n';}
 };
 constexpr unsigned mov=0x4c000000,end=0x88000000;
 auto p=fresh();check("empty",p,true);p.allow_emission=true;check("GS metadata",p,false);
 p=fresh();p.entry=4096;check("invalid entry",p,false);
 for(unsigned op:std::array{4u,7u,0x10u,0x11u,0x19u,0x20u,0x23u,0x2au,0x2bu}) {
  p=fresh();p.code[0]=op<<26;p.code[1]=end;check("unsupported precise instruction",p,false);
 }
 p=fresh();p.code[0]=(0x2eu<<26)|(6u<<21);p.code[1]=end;check("CMP6",p,false);
 p=fresh();p.code[0]=end;p.code[1]=0x2au<<26;check("dead emission",p,true);
 p=fresh();p.code[0]=(0x24u<<26)|(3u<<10)|2;p.code[1]=end;p.code[3]=mov;check("normal CALL",p,true);
 p=fresh();p.code[0]=(0x24u<<26)|(3u<<10)|1;p.code[1]=end;p.code[3]=(0x24u<<26)|(5u<<10)|1;p.code[5]=mov;check("tail CALL boundary",p,false);
 p=fresh();p.code[0]=(0x24u<<26)|1;p.code[1]=end;check("recursive CALL",p,false);
 for(unsigned depth:std::array{3u,4u}) {
  p=fresh();p.code[0]=(0x24u<<26)|(10u<<10)|2;p.code[1]=end;
  for(unsigned level=1;level<depth;++level)p.code[level*10]=(0x24u<<26)|((level+1)*10u<<10)|2;
  p.code[depth*10]=mov;check("bounded CALL depth",p,depth==3);
 }
 p=fresh();p.code[0]=(0x27u<<26)|(2u<<10)|1;p.code[1]=mov;p.code[2]=mov;p.code[3]=end;check("ordinary IF",p,true);
 p=fresh();p.code[0]=(0x27u<<26)|(3u<<10)|1;p.code[1]=(0x27u<<26)|(3u<<10);p.code[2]=mov;p.code[4]=end;check("colliding IF returns",p,false);
 for(unsigned depth:std::array{8u,9u}) {
  p=fresh();for(unsigned i=0;i<depth;++i)p.code[i]=(0x27u<<26)|((2*depth-i)<<10);
  p.code[depth]=mov;p.code[2*depth+1]=end;check("IF stack bound",p,depth==8);
 }
 for(unsigned depth:std::array{4u,5u}) {
  p=fresh();for(unsigned i=0;i<depth;++i)p.code[i]=(0x29u<<26)|((2*depth-i-1)<<10);
  p.code[depth]=mov;p.code[2*depth]=end;check("LOOP stack bound",p,depth==4);
 }
 p=fresh();p.code[0]=(0x2du<<26)|(2u<<10);p.code[1]=mov;p.code[2]=end;check("forward JMPU",p,true);
 p.code[0]=(0x2cu<<26)|(2u<<10);check("forward JMPC",p,true);
 p.code[1]=0x2cu<<26;check("backward JMP",p,false);
 p=fresh();p.code[0]=(0x24u<<26)|(3u<<10)|3;p.code[1]=end;p.code[3]=(0x2du<<26)|(6u<<10);check("JMP to CALL end",p,false);
 p=fresh();p.code[0]=(0x2du<<26)|(2u<<10);p.code[1]=(0x27u<<26)|(3u<<10)|1;p.code[2]=mov;p.code[3]=mov;p.code[4]=end;check("JMP enters nested IF",p,false);
 p=fresh();p.code[0]=(0x27u<<26)|(4u<<10)|1;p.code[1]=(0x2du<<26)|(4u<<10);p.code[5]=end;check("JMP to IF boundary",p,false);
 std::cout<<"Native structured control proof: "<<checks<<" bounded CALL/IF/LOOP/JMP and exclusion checks, failures="<<failures<<'\n';return failures ? 1:0;
}

void ValidateNativePrimitives(Metal::Device& device,Metal::Scheduler& scheduler,
    Metal::ShaderRuntime& shaders,Metal::ShaderMachineRuntime& machine,Metal::GeometryRuntime& geometry) {
    Metal::TransferRuntime transfer{device,scheduler};
    constexpr unsigned vertices=7,base=8;
    const auto input=device.CreateBuffer(vertices*256,true);
    auto* registers=static_cast<Metal::VertexRegisters*>(input.contents);
    for(unsigned v=0;v<vertices;++v) {
        registers[v]={};registers[v][0]={float(v)*.125f,-float(v)*.0625f,-.25f,1};
        registers[v][1]={v&1 ? -1.f:1.f,0,0,0};registers[v][2]={.25f,.5f,.75f,1};
    }
    Pica::RegsInternal regs{};Pica::ShaderSetup setup;Metal::Test::InitializeFixtureUniforms(setup);Metal::Test::MovVertex(regs,setup);
    auto compiled=shaders.Vertex(regs,setup,true);Metal::ShaderMachineProgram p{};p.code=setup.GetProgramCode();p.swizzle=setup.GetSwizzleData();p.allow_emission=false;p.output_mask=7;
    Metal::GeometryEmissionConfig capture{};capture.attribute_count=3;capture.semantics={0x03020100,0x07060504,0x0b0a0908};capture.honor_winding=0;
    Check(machine.CanCaptureNativeReset(p,{},capture),"MOV primitive test was not selected native");
    auto u=device.CreateBuffer(1616,true),vs=device.CreateBuffer(32,true);std::memset(u.contents,0,1616);std::memset(vs.contents,0,32);
    const std::array<Metal::BufferBinding,2> bindings{{{u,0,0},{vs,0,1}}};unsigned cases=0;
    for(auto topology:std::array{Metal::TriangleTopology::List,Metal::TriangleTopology::Strip,Metal::TriangleTopology::Fan})
    for(unsigned width:std::array{0u,1u,2u,4u}) for(bool fix:std::array{false,true}) {
        Metal::PrimitiveStream strict_stream{device},native_stream{device};
        Metal::ShaderMachineState strict_unit{device},native_unit{device};
        for(unsigned batch=0;batch<4;++batch) {
            const std::vector<unsigned> ids=width ? std::vector<unsigned>{8,10,8,14,10,9,13,8,12,14,11}:std::vector<unsigned>{8,9,10,11,12,13,14};
            auto index=device.CreateBuffer(ids.size()*std::max(1u,width),true);
            if(width)for(unsigned e=0;e<ids.size();++e)std::memcpy(static_cast<std::byte*>(index.contents)+e*width,&ids[e],width);
            Metal::BufferSlice ix=width ? Metal::BufferSlice{index,0,index.length,index.contents,{}}:Metal::BufferSlice{};
            Metal::BufferSlice in{input,0,input.length,input.contents,{}};
            strict_unit.Reset();native_unit.Reset();
            const auto strict=machine.RunVertexBatchAndCapture(strict_unit,p,{},in,capture,vertices,ids.size(),base,base,ix,width);
            machine.DeferResetVertexBatch(native_unit,p,{},in,vertices,ids.size(),base,base,ix,width);
            const auto native=geometry.Capture(*compiled,in,vertices,base,bindings);
            const auto a=geometry.Append(strict_stream,strict.captured,ids.size(),topology,ids.size(),0,0,{},0,fix);
            const auto b=geometry.Append(native_stream,native,vertices,topology,ids.size(),base,base,ix,width,fix);
            Check(a.vertex_count==b.vertex_count,"Native primitive count differs");
            if(a.vertex_count) {
                const auto bytes=NSUInteger(a.vertex_count)*128;auto out=device.CreateBuffer(bytes*2,true);
                transfer.CopyBuffer(a.storage.buffer,a.storage.offset,out,0,bytes);
                transfer.CopyBuffer(b.storage.buffer,b.storage.offset,out,bytes,bytes);scheduler.Finish();
                Check(!std::memcmp(out.contents,static_cast<std::byte*>(out.contents)+bytes,bytes),"Native complete primitive vertices differ");
            }
            const auto ah=geometry.SaveStream(strict_stream),bh=geometry.SaveStream(native_stream);
            Check(ah.pending==bh.pending && ah.topology==bh.topology && ah.needs_reset==bh.needs_reset && ah.history==bh.history,
                  "Native complete272-byte primitive history differs");
            Check(machine.SaveState(strict_unit)==machine.SaveState(native_unit),"Native primitive complete deferred VS state differs");++cases;
        }
    }
    std::cout<<"Native indexed primitive stream: "<<cases<<" list/strip/fan,0/1/2/4-byte indices, FIFO hits, retained stream and quaternion-repair cases bit exact\n";
}
void ValidateOwnedNativeBatch(Metal::Device& device,Metal::Scheduler& scheduler,
    Metal::ShaderRuntime& shaders,Metal::ShaderMachineRuntime& machine,Metal::GeometryRuntime& geometry) {
    Metal::TransferRuntime transfer{device,scheduler};Metal::VertexFetchRuntime fetch{device,scheduler};
    constexpr unsigned vertices=67,base=8;unsigned cases=0;
    Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=7;p.swizzle[0]=Metal::Test::IdentitySwizzle;
    p.code[0]=(16u<<21)|(16u<<12)|(15u<<7); // Unobservable retained r0 += v15.
    p.code[1]=(17u<<21)|(32u<<12)|(17u<<7); // Unobservable retained r1 += c0.
    for(unsigned i=0;i<3;++i)p.code[2+i]=(0x13u<<26)|(i<<21)|(i<<12);
    p.code[5]=0x22u<<26;
    Metal::ShaderMachineUniforms values{};values.f[0]={.25f,-.5f,.125f,1};
    Metal::GeometryEmissionConfig capture{};capture.attribute_count=3;capture.semantics={0x03020100,0x07060504,0x0b0a0908};capture.honor_winding=0;
    Check(machine.CanCaptureNativeReset(p,values,capture),"Owned native recurrence fixture not selected");
    Pica::RegsInternal regs{};Pica::ShaderSetup setup;Metal::Test::InitializeFixtureUniforms(setup);Metal::Test::MovVertex(regs,setup);
    setup.UpdateProgramCode(p.code);setup.UpdateSwizzleData(p.swizzle);auto compiled=shaders.Vertex(regs,setup,true);
    auto ub=device.CreateBuffer(1616,true),vs=device.CreateBuffer(32,true);
    Pica::Shader::Generator::VSPicaUniformData guest{};std::memcpy(guest.f.data(),values.f.data(),sizeof(values.f));
    std::memcpy(ub.contents,&guest,sizeof(guest));std::memset(vs.contents,0,32);
    const std::array<Metal::BufferBinding,2> bindings{{{ub,0,0},{vs,0,1}}};
    for(unsigned mode=0;mode<2;++mode)for(unsigned width:std::array{0u,1u,2u,4u})
    for(auto topology:std::array{Metal::TriangleTopology::List,Metal::TriangleTopology::Strip,Metal::TriangleTopology::Fan}) {
        Metal::VertexInputState oracle_input{device},owned_input{device};
        Metal::PrimitiveStream oracle_stream{device},owned_stream{device};
        auto input=device.CreateBuffer(vertices*256+64,true),packed=device.CreateBuffer(vertices*8+64,true);
        std::vector<unsigned> ids;
        for(unsigned v=0;v<vertices;++v)ids.push_back(base+v);
        if(width)for(unsigned v:std::array{0u,65u,1u,66u,0u,64u,1u})ids.push_back(base+v);
        auto indices=device.CreateBuffer(ids.size()*std::max(1u,width)+8,true);
        for(unsigned batch=0;batch<2;++batch) {
            scheduler.Finish();std::memset(input.contents,0,input.length);std::memset(packed.contents,0,packed.length);std::memset(indices.contents,0,indices.length);
            for(unsigned v=0;v<vertices;++v) {
                auto* data=static_cast<std::byte*>(input.contents)+32;
                if(!mode) {
                    Metal::VertexRegisters registers{};
                    for(unsigned reg=0;reg<16;++reg) registers[reg]={float(v+reg)*.125f,-float(v+batch)*.0625f,.5f,1};
                    registers[1]={v&1 ? -1.f:1.f,0,0,0};
                    std::memcpy(data+v*256,&registers,256);
                } else {
                    const std::array<std::int16_t,3> position{std::int16_t(v),std::int16_t(-int(v)-int(batch)),-2};
                    std::memcpy(data+v*9+1,position.data(),6);
                    const std::array<std::int8_t,4> quaternion{std::int8_t(v&1 ? -1:1),0,0,0};
                    std::memcpy(static_cast<std::byte*>(packed.contents)+16+v*8,quaternion.data(),4);
                }
            }
            if(width)for(unsigned e=0;e<ids.size();++e)std::memcpy(static_cast<std::byte*>(indices.contents)+3+e*width,&ids[e],width);
            const std::array<Metal::BufferSlice,2> inputs{{
                {input,32,vertices*256,static_cast<std::byte*>(input.contents)+32,{}},
                {packed,16,vertices*8,static_cast<std::byte*>(packed.contents)+16,{}}}};
            const Metal::BufferSlice ix=width ? Metal::BufferSlice{indices,3,ids.size()*width,static_cast<std::byte*>(indices.contents)+3,{}}:Metal::BufferSlice{};
            Metal::PicaVertexLayout layout{};layout.fixed_attributes=Metal::DefaultVertexRegisters();layout.mapped_attribute_count=mode ? 7:16;
            for(unsigned attr=0;attr<16;++attr) {
                layout.register_map[attr]=attr;
                if(!mode) {layout.attribute_inputs[attr].mode=Metal::PicaAttributeMode::Buffer;layout.attribute_inputs[attr].fetch={0,attr*16,256,4,Metal::VertexAttributeFormat::Float,attr};}
            }
            if(mode) {
                layout.attribute_inputs[0].mode=Metal::PicaAttributeMode::Buffer;
                layout.attribute_inputs[0].fetch={0,1,9,3,Metal::VertexAttributeFormat::Short,0};
                layout.attribute_inputs[1].mode=batch ? Metal::PicaAttributeMode::Retained:Metal::PicaAttributeMode::Buffer;
                layout.attribute_inputs[1].fetch={1,0,8,4,Metal::VertexAttributeFormat::Byte,1};
                layout.attribute_inputs[2].mode=Metal::PicaAttributeMode::Fixed;layout.fixed_attributes[2]={.25f,.5f,.75f,1};
                layout.attribute_inputs[4].mode=Metal::PicaAttributeMode::Buffer;
                layout.attribute_inputs[4].fetch={1,1,8,2,Metal::VertexAttributeFormat::UnsignedByte,4};
            }
            Metal::ShaderMachineState oracle{device},owned{device};
            const auto strict=machine.RunFetchedVertexBatchAndCapture(fetch,oracle_input,inputs,layout,oracle,p,values,capture,
                vertices,ids.size(),base,base,ix,width,1000000,true);
            const auto expected_state=machine.SaveState(oracle);const auto expected_fetch=fetch.SaveState(oracle_input);
            auto caller_program=p;auto caller_uniforms=values;
            const auto native=machine.RunFetchedNativeResetAndCapture(fetch,owned_input,inputs,layout,owned,caller_program,caller_uniforms,capture,
                geometry,*compiled,bindings,vertices,ids.size(),base,base,ix,width);
            Check(owned.HasDeferredVertexBatch(),"Owned native state was not retained");
            const auto a=geometry.Append(oracle_stream,strict.captured,ids.size(),topology,ids.size(),0,0,{},0,batch!=0);
            const auto b=geometry.Append(owned_stream,native,vertices,topology,ids.size(),base,base,ix,width,batch!=0);
            Check(a.vertex_count==b.vertex_count,"Owned native primitive count differs");
            const auto bytes=NSUInteger(a.vertex_count)*128;auto readback=device.CreateBuffer(std::max<NSUInteger>(16,bytes*2),true);
            if(bytes) {transfer.CopyBuffer(a.storage.buffer,a.storage.offset,readback,0,bytes);transfer.CopyBuffer(b.storage.buffer,b.storage.offset,readback,bytes,bytes);}
            // No CPU wait between the owned operation and later GPU mutation.
            transfer.FillBuffer(input,NSMakeRange(0,input.length),0xa5);
            transfer.FillBuffer(packed,NSMakeRange(0,packed.length),0xcc);
            transfer.FillBuffer(indices,NSMakeRange(0,indices.length),0xff);
            caller_program.code.fill(0x84000000);caller_program.swizzle.fill(0);caller_uniforms={};
            bool rejected=false;
            try {machine.RunFetchedNativeResetAndCapture(fetch,owned_input,inputs,layout,owned,p,values,capture,
                geometry,*compiled,bindings,0,ids.size(),base,base,ix,width);}catch(const std::invalid_argument&){rejected=true;}
            Check(rejected && owned.HasDeferredVertexBatch(),"Invalid owned dimensions replaced existing state");
            std::array<bool,1> foreign_rejected{};
            std::thread foreign([&]{try {machine.RunFetchedNativeResetAndCapture(fetch,owned_input,inputs,layout,owned,p,values,capture,
                geometry,*compiled,bindings,vertices,ids.size(),base,base,ix,width);}catch(const std::logic_error&){foreign_rejected[0]=true;}});foreign.join();
            Check(foreign_rejected[0] && owned.HasDeferredVertexBatch(),"Foreign thread touched owned state");
            auto invalid=expected_state;invalid.history[1804]=std::byte{1};rejected=false;
            try {machine.LoadState(owned,invalid);}catch(const std::invalid_argument&){rejected=true;}
            Check(rejected && owned.HasDeferredVertexBatch(),"Invalid load discarded owned snapshot");
            Check(machine.SaveState(owned)==expected_state,"Owned canonical/index/program/uniform snapshot changed after caller mutation");
            const auto actual_fetch=fetch.SaveState(owned_input);
            Check(actual_fetch.needs_reset==expected_fetch.needs_reset && actual_fetch.history==expected_fetch.history,"Owned complete512-byte input history differs");
            Check(!bytes || !std::memcmp(readback.contents,static_cast<std::byte*>(readback.contents)+bytes,bytes),"Owned complete128-byte primitive output differs");
            const auto ah=geometry.SaveStream(oracle_stream),bh=geometry.SaveStream(owned_stream);
            Check(ah.pending==bh.pending && ah.topology==bh.topology && ah.needs_reset==bh.needs_reset && ah.history==bh.history,
                "Owned complete272-byte primitive history differs");++cases;
        }
    }
    Check(cases==48,"Owned packed primitive corpus size changed");
    Metal::VertexRegisters fixed=Metal::DefaultVertexRegisters();fixed[0]={.25f,-.5f,-.25f,1};fixed[1]={1,0,0,0};fixed[2]={.25f,.5f,.75f,1};
    auto immediate=Metal::BuildPicaImmediateLayout(Pica::RegsInternal{},fixed);
    // Explicit mapping avoids a default register-count specialization in the fixture.
    immediate.mapped_attribute_count=16;for(unsigned i=0;i<16;++i) {
        immediate.register_map[i]=i;immediate.attribute_inputs[i].mode=Metal::PicaAttributeMode::Fixed;
    }
    Metal::VertexInputState input_state{device};Metal::ShaderMachineState retained{device};
    const auto oracle_snapshot=[&](const Metal::PicaVertexLayout& layout) {
        Metal::VertexInputState oracle_input{device};Metal::ShaderMachineState oracle{device};
        machine.RunFetchedVertexBatchAndCapture(fetch,oracle_input,{},layout,oracle,p,values,capture,1,1,0,0,{},0,1000000,true);
        return machine.SaveState(oracle);
    };
    const auto expected=oracle_snapshot(immediate);
    const auto defer=[&](Metal::ShaderMachineState& target,const Metal::PicaVertexLayout& layout) {
        machine.RunFetchedNativeResetAndCapture(fetch,input_state,{},layout,target,p,values,capture,geometry,*compiled,bindings,1,1,0,0);
        Check(target.HasDeferredVertexBatch(),"Owned lifecycle did not retain snapshot");
    };
    defer(retained,immediate);const auto tick=scheduler.CurrentTick();retained.Reset();
    Check(!retained.HasDeferredVertexBatch() && machine.SaveState(retained)==Metal::ShaderMachineSnapshot{} && scheduler.CurrentTick()==tick,
        "Owned Reset materialized discarded work");++cases;
    defer(retained,immediate);machine.LoadState(retained,Metal::ShaderMachineSnapshot{});
    Check(!retained.HasDeferredVertexBatch() && machine.SaveState(retained)==Metal::ShaderMachineSnapshot{},"Owned reset load retained pending work");++cases;
    defer(retained,immediate);machine.LoadState(retained,expected);
    Check(!retained.HasDeferredVertexBatch() && machine.SaveState(retained)==expected,"Owned load did not replace pending state");++cases;
    for(unsigned bad=0;bad<5;++bad) {
        defer(retained,immediate);const auto input_before=fetch.SaveState(input_state);
        auto invalid_program=p;auto invalid_layout=immediate;auto invalid_native=*compiled;auto invalid_bindings=bindings;
        if(bad==0)invalid_native.pipeline=nil;
        if(bad==1)invalid_bindings[1].buffer=nil;
        if(bad==2)invalid_layout.attribute_inputs[0].mode=Metal::PicaAttributeMode::Buffer;
        if(bad==3)invalid_program.code[2]=(0x13u<<26)|(16u<<12);
        bool rejected=false;
        try {machine.RunFetchedNativeResetAndCapture(fetch,input_state,{},invalid_layout,retained,invalid_program,values,capture,
            geometry,invalid_native,invalid_bindings,1,1,0,0,{},0,bad==4 ? 0:1000000);}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && retained.HasDeferredVertexBatch(),"Invalid owned native/fetch/proof metadata replaced state");
        const auto input_after=fetch.SaveState(input_state);
        Check(input_before.needs_reset==input_after.needs_reset && input_before.history==input_after.history,"Invalid owned metadata changed input history");
        Check(machine.SaveState(retained)==expected,"Invalid owned metadata changed represented VS state");++cases;
    }
    defer(retained,immediate);
    auto alternate=immediate;alternate.fixed_attributes[0][0]=.75f;defer(retained,alternate);
    Check(machine.SaveState(retained)==oracle_snapshot(alternate),"Owned pending batches composed instead of replacing");++cases;
    defer(retained,immediate);Metal::ShaderMachineState continued_oracle{device};machine.LoadState(continued_oracle,expected);
    auto next=p;next.output_mask=1;next.code[0]=(0x13u<<26)|(16u<<12);next.code[1]=0x22u<<26;
    auto shared=device.CreateBuffer(256,true);std::memcpy(shared.contents,&fixed,256);
    const Metal::BufferSlice input{shared,0,256,shared.contents,{}};
    machine.Run(continued_oracle,next,values,input);machine.Run(retained,next,values,input);
    Check(machine.SaveState(retained)==machine.SaveState(continued_oracle),"Owned inherited invocation skipped exact state materialization");++cases;
    {
        Metal::ShaderMachineRuntime temporary{device,scheduler};
        temporary.RunFetchedNativeResetAndCapture(fetch,input_state,{},immediate,retained,p,values,capture,
            geometry,*compiled,bindings,1,1,0,0);
    }
    Check(retained.HasDeferredVertexBatch() && machine.SaveState(retained)==expected,"Owned snapshot borrowed its destroyed runtime storage");++cases;
    std::cout<<"Owned native pipeline: "<<cases<<" cases (48 packed/full primitives plus reset/load/replacement/inherited/runtime-lifetime and invalid metadata); complete128/272/512/1824-byte output/state checks, all index widths/FIFO and later GPU/CPU-object mutation preserved\n";
}

int main() {@autoreleasepool {try {
 Metal::Device device{{},false};Metal::Scheduler scheduler{device};Metal::TransferRuntime transfer{device,scheduler};
 Metal::ShaderRuntime shaders{device};Metal::ShaderMachineRuntime machine{device,scheduler};Metal::GeometryRuntime geometry{device,scheduler};
 auto input=device.CreateBuffer(256,true);Metal::VertexRegisters regs{};
 Metal::ShaderMachineUniforms uniforms{};uniforms.f[0]={.25f,0,-.5f,1};
 for(unsigned i=1;i<96;++i) uniforms.f[i]={float(i)*.125f,-float(i)*.0625f,.25f,1};
 Pica::Shader::Generator::VSPicaUniformData native_values{};
 const auto ub=device.CreateBuffer(sizeof(native_values),true);
 const auto vsb=device.CreateBuffer(32,true);
 const auto actual=device.CreateBuffer(128,true),expected=device.CreateBuffer(128,true);
 constexpr std::array<unsigned,16> bits{0,0x80000000,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc12345,0x7fa54321,1,0x80000001,0x007fffff,0x807fffff,0x7f7fffff,0xff7fffff,0x3eaaaaab,0xc3000000};
 Check(!ValidateNativeFlow(),"Native structured control corpus failed");
 ValidateNativeSelection(machine);
 ValidateOwnedNativeBatch(device,scheduler,shaders,machine,geometry);
 ValidateNativePrimitives(device,scheduler,shaders,machine,geometry);
 unsigned checks=0,failures=0;
 const auto run=[&](Metal::ShaderMachineProgram& p,Metal::GeometryEmissionConfig capture,const char* label,unsigned sample) {
  Pica::RegsInternal config{};Pica::ShaderSetup setup;Metal::Test::InitializeFixtureUniforms(setup);
  config.vs.main_offset.Assign(p.entry);config.vs.output_mask.Assign(p.output_mask);
  config.rasterizer.vs_output_total.Assign(capture.attribute_count);config.lighting.disable.Assign(1);
  for(unsigned i=0;i<7;++i) config.rasterizer.vs_output_attributes[i].raw=capture.semantics[i];
  setup.UpdateProgramCode(p.code);setup.UpdateSwizzleData(p.swizzle);
  auto compiled=shaders.Vertex(config,setup,true);
  native_values.b=uniforms.b;std::memcpy(native_values.f.data(),uniforms.f.data(),sizeof(uniforms.f));
  std::memcpy(native_values.i.data(),uniforms.i.data(),sizeof(uniforms.i));std::memcpy(ub.contents,&native_values,sizeof(native_values));
  Pica::Shader::Generator::VSUniformData vs{};vs.enable_clip1=capture.enable_clip;vs.flip_viewport=capture.flip_viewport;
  std::memcpy(&vs.clip_coef,capture.clip_coefficient.data(),16);std::memcpy(vsb.contents,&vs,sizeof(vs));
  std::memcpy(input.contents,&regs,256);
  Metal::ShaderMachineState state{device};const auto raw=machine.Run(state,p,uniforms,{input,0,256,input.contents,{}});
  auto oracle=machine.CaptureVertex(raw.output,p.output_mask,capture);
  const std::array<Metal::BufferBinding,2> bindings{{{ub,0,0},{vsb,0,1}}};
  auto native=geometry.Capture(*compiled,{input,0,256,input.contents,{}},1,0,bindings);
  transfer.CopyBuffer(oracle.buffer,oracle.offset,expected,0,128);
  transfer.CopyBuffer(native.buffer,native.offset,actual,0,128);scheduler.Finish();++checks;
  if(std::memcmp(actual.contents,expected.contents,128)) {
   ++failures;const auto* aa=static_cast<const unsigned*>(actual.contents);const auto* ee=static_cast<const unsigned*>(expected.contents);
   if(failures<=24)for(unsigned c=0;c<32;++c)if(aa[c]!=ee[c])std::cerr<<label<<" sample="<<sample<<" lane="<<c<<" actual="<<std::hex<<aa[c]<<" expected="<<ee[c]<<std::dec<<'\n';
  }
 };
 Metal::GeometryEmissionConfig capture{};capture.attribute_count=2;capture.semantics={0x03020100,0x07060504};capture.honor_winding=0;
 capture.enable_clip=1;capture.clip_coefficient={.25f,-.5f,.125f,1};
 for(unsigned op:std::array{0u,1u,2u,3u,5u,6u,8u,9u,10u,11u,12u,13u,14u,15u,0x13u,0x18u,0x1au,0x1bu,0x30u,0x38u}) {
  Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;p.swizzle[0]=Metal::Test::IdentitySwizzle;
  p.code[0]=(0x13u<<26)|(32u<<12);
  p.code[1]=op>=0x30 ? (op<<26)|(1u<<24)|(1u<<(op<0x38 ? 12:10))|(2u<<5):
    (op<<26)|(1u<<21)|(1u<<7);
  p.code[2]=0x22u<<26;
  for(unsigned sample=0;sample<32;++sample) {
   for(unsigned v=0;v<4;++v) for(unsigned c=0;c<4;++c) regs[v][c]=std::bit_cast<float>(bits[(sample+v*5+c*3)%bits.size()]);
   run(p,capture,"arithmetic",op*100+sample);
  }
 }
 // Both inherited read-only temp.w and unwritten output.w reset to zero.
 for(unsigned kind=0;kind<2;++kind) {
  Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;p.swizzle[0]=Metal::Test::IdentitySwizzle;
  p.swizzle[1]=Metal::Test::IdentitySwizzle&~1u;
  p.code[0]=(0x13u<<26)|(32u<<12);
  p.code[1]=(0x13u<<26)|(1u<<21)|(kind ? 0u:16u<<12)|(kind ? 1u:0u);
  p.code[2]=0x22u<<26;run(p,capture,"reset-banks",kind);
 }
 for(unsigned address=1;address<=2;++address) {
  Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;p.swizzle[0]=Metal::Test::IdentitySwizzle;
  p.code[0]=(0x13u<<26)|(32u<<12);p.code[1]=0x12u<<26;
  p.code[2]=(0x13u<<26)|(1u<<21)|(address<<19)|(32u<<12);p.code[3]=0x22u<<26;
  for(unsigned sample=0;sample<32;++sample) {
   for(unsigned c=0;c<4;++c) regs[0][c]=std::bit_cast<float>(bits[(sample+c)%bits.size()]);
   run(p,capture,"mova",address*100+sample);
  }
 }
 // Non-byte integer values must retain the machine's eight-bit loop fields.
 for(unsigned mask:std::array{0u,1u,2u,255u,256u,257u,0xffffff00u,0xffffffffu}) {
  Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;p.swizzle[0]=Metal::Test::IdentitySwizzle;
  p.code[0]=(0x13u<<26)|(32u<<12);p.code[1]=(0x13u<<26)|(16u<<21);
  p.code[2]=(0x29u<<26)|(3u<<10);p.code[3]=(16u<<21)|(16u<<12)|(1u<<7);
  p.code[4]=(0x13u<<26)|(1u<<21)|(16u<<12);p.code[5]=0x22u<<26;
  uniforms.i[0]={mask,259,0x800000ffu,0};regs[0]={.25f,.5f,-.25f,1};regs[1]={.125f,.25f,.5f,1};
  run(p,capture,"loop-byte-fields",mask);
 }uniforms.i={};
 // Explicit NaN color saturation and viewport/clip signed-zero preservation.
 for(unsigned sample=0;sample<32;++sample) {
  Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=3;p.swizzle[0]=Metal::Test::IdentitySwizzle;
  p.code[0]=0x13u<<26;p.code[1]=(0x13u<<26)|(1u<<21)|(1u<<12);p.code[2]=0x22u<<26;
  auto c=capture;c.semantics[1]=0x0b0a0908;c.flip_viewport=sample&1;c.enable_clip=(sample>>1)&1;
  for(unsigned v=0;v<2;++v)for(unsigned lane=0;lane<4;++lane)regs[v][lane]=std::bit_cast<float>(bits[(sample+lane*3+v*5)%bits.size()]);
  run(p,c,"nan-color-and-position",sample);
 }
 std::cout<<"Reset-only native shader: "<<checks<<" exact128-byte comparisons, failures="<<failures<<'\n';
 scheduler.Finish();return failures ? 1:0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
