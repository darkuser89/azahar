// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_shader_machine.h"
#include "mtl_vertex_diagnostic.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <thread>

namespace {
void Check(bool yes,const char* message) {if(!yes) throw std::runtime_error(message);}
using D=Metal::VertexDiagnostic;
void Replay(Metal::Device& device,Metal::Scheduler& scheduler,Metal::ShaderMachineRuntime& machine,
            const std::filesystem::path& path,bool scalar,bool queued=false) {
    std::ifstream stream(path,std::ios::binary);
    const auto read=[&](void* dest,std::size_t size) {
        stream.read(static_cast<char*>(dest),size);Check(bool(stream),"Truncated vertex diagnostic");
    };
    D::Header h{};read(h.data(),sizeof(h));
    Check(h[D::Magic]==D::Signature && h[D::Version]==1 && h[D::HeaderBytes]==sizeof(h) &&
          h[D::CodeWords]==4096 && h[D::SwizzleWords]==4096 && h[D::UniformBytes]==1616 &&
          h[D::StateBytes]==1824 && h[D::VertexBytes]==256 && !h[D::AllowEmission] &&
          h[D::InitialReset]<=1 && !h[D::FinalReset],"Unsupported vertex diagnostic schema");
    Check(h[D::Vertices] && h[D::Elements] &&
          h[D::InputBytes]==std::uint64_t(h[D::Vertices])*256 &&
          h[D::IndexBytes]==std::uint64_t(h[D::Elements])*h[D::IndexWidth] &&
          h[D::OutputBytes]==std::uint64_t(h[D::Elements])*256 &&
          std::uint64_t(h[D::InputBytes])+h[D::IndexBytes]+h[D::OutputBytes]<=16u*1024u*1024u &&
          (h[D::IndexWidth]==0 || h[D::IndexWidth]==1 || h[D::IndexWidth]==2 || h[D::IndexWidth]==4),
          "Invalid vertex diagnostic extents");
    Metal::ShaderMachineProgram p{};p.entry=h[D::Entry];p.output_mask=h[D::OutputMask];p.allow_emission=false;
    read(p.code.data(),sizeof(p.code));read(p.swizzle.data(),sizeof(p.swizzle));
    Metal::ShaderMachineUniforms u{};read(&u,sizeof(u));
    Metal::ShaderMachineSnapshot initial{};initial.needs_reset=h[D::InitialReset];
    read(initial.history.data(),initial.history.size());Metal::ShaderMachineRuntime::ValidateState(initial);
    auto input=device.CreateBuffer(h[D::InputBytes],true);read(input.contents,input.length);
    auto indices=device.CreateBuffer(std::max(h[D::IndexBytes],4u),true);
    read(indices.contents,h[D::IndexBytes]);
    std::vector<std::byte> expected(h[D::OutputBytes]);read(expected.data(),expected.size());
    Metal::ShaderMachineSnapshot final{};final.needs_reset=false;
    read(final.history.data(),final.history.size());Metal::ShaderMachineRuntime::ValidateState(final);
    Check(stream.peek()==std::char_traits<char>::eof(),"Trailing vertex diagnostic data");
    Metal::ShaderMachineState state{device};machine.LoadState(state,initial);
    const Metal::BufferSlice source{input,0,input.length,input.contents,{}};
    const Metal::BufferSlice ix{indices,0,h[D::IndexBytes],indices.contents,{}};
    const auto raw=machine.RunVertexBatch(state,p,u,source,h[D::Vertices],h[D::Elements],
        h[D::VertexBase],h[D::FirstVertex],h[D::IndexWidth] ? ix:Metal::BufferSlice{},
        h[D::IndexWidth],h[D::InstructionLimit],true);
    auto output=device.CreateBuffer(raw.size,true);
    Metal::TransferRuntime{device,scheduler}.CopyBuffer(raw.buffer,raw.offset,output,0,raw.size);
    scheduler.Finish();
    Check(std::memcmp(output.contents,expected.data(),expected.size())==0,"Captured raw outputs do not replay bit exactly");
    Check(machine.SaveState(state)==final,"Captured complete GPU history does not replay bit exactly");
    if(initial.needs_reset) {
        machine.DeferResetVertexBatch(state,p,u,source,h[D::Vertices],h[D::Elements],
            h[D::VertexBase],h[D::FirstVertex],h[D::IndexWidth] ? ix:Metal::BufferSlice{},h[D::IndexWidth],h[D::InstructionLimit]);
        Metal::TransferRuntime transfer{device,scheduler};
        transfer.FillBuffer(input,NSMakeRange(0,input.length),0xa5);
        transfer.FillBuffer(indices,NSMakeRange(0,indices.length),0xff);
        Check(machine.SaveState(state)==final,"Recorded deferred batch lost full unit state after caller mutation");
        // Restore exact caller data for the independent scalar oracle below.
        stream.clear();stream.seekg(sizeof(h)+sizeof(p.code)+sizeof(p.swizzle)+sizeof(u)+1824);
        read(input.contents,input.length);read(indices.contents,h[D::IndexBytes]);
    }
    if (!scalar) return;
    // Independent native single-invocation GPU execution, with the documented
    // 64-entry circular post-VS cache represented by IDs only on the host.
    machine.LoadState(state,initial);
    std::array<std::pair<unsigned,unsigned>,64> cache{};unsigned replacement=0,valid=0;
    std::vector<Metal::VertexRegisters> scalar_output(h[D::Elements]);
    for(unsigned e=0;e<h[D::Elements];++e) {
        unsigned id=h[D::FirstVertex]+e;
        if(h[D::IndexWidth]) {id=0;std::memcpy(&id,static_cast<const std::byte*>(indices.contents)+e*h[D::IndexWidth],h[D::IndexWidth]);}
        Check(id>=h[D::VertexBase] && std::uint64_t(id)-h[D::VertexBase]<h[D::Vertices],"Captured index outside vertex range");
        bool hit=false;
        if(h[D::IndexWidth]) for(unsigned c=0;c<valid;++c) if(cache[c].first==id) {
            scalar_output[e]=scalar_output[cache[c].second];hit=true;break;
        }
        if(hit) continue;
        auto vertex=source;vertex.offset=(id-h[D::VertexBase])*256;vertex.size=256;
        const auto result=machine.Run(state,p,u,vertex,0xffff,192,h[D::InstructionLimit]);
        std::memcpy(&scalar_output[e],result.output.data,256);
        if(h[D::IndexWidth]) {cache[replacement]={id,e};replacement=(replacement+1)%64;valid=std::min(valid+1,64u);}
    }
    Check(std::memcmp(scalar_output.data(),expected.data(),expected.size())==0,"Captured output differs from independent scalar GPU oracle");
    Check(machine.SaveState(state)==final,"Captured history differs from independent scalar GPU oracle");
    if(queued) {
        machine.LoadState(state,initial);
        auto readback=device.CreateBuffer(expected.size(),true);
        Metal::TransferRuntime transfer{device,scheduler};
        cache={};replacement=valid=0;
        for(unsigned e=0;e<h[D::Elements];++e) {
            unsigned id=h[D::FirstVertex]+e;
            if(h[D::IndexWidth]) {id=0;std::memcpy(&id,static_cast<const std::byte*>(indices.contents)+e*h[D::IndexWidth],h[D::IndexWidth]);}
            bool hit=false;
            if(h[D::IndexWidth]) for(unsigned c=0;c<valid;++c) if(cache[c].first==id) {
                transfer.CopyBuffer(readback,cache[c].second*256,readback,e*256,256);hit=true;break;
            }
            if(hit) continue;
            const auto offset=NSUInteger(id-h[D::VertexBase])*256;
            const auto value=machine.RunVertexDeferred(state,p,u,
                {input,offset,256,static_cast<std::byte*>(input.contents)+offset,{}},
                0xffff,h[D::InstructionLimit]);
            transfer.CopyBuffer(value.buffer,value.offset,readback,e*256,256);
            if(h[D::IndexWidth]) {cache[replacement]={id,e};replacement=(replacement+1)%64;valid=std::min(valid+1,64u);}
        }
        scheduler.Finish();
        Check(!std::memcmp(readback.contents,expected.data(),expected.size()),
              "Queued scalar real record changed captured output/FIFO bytes");
        Check(machine.SaveState(state)==final,"Queued scalar real record changed full1824-byte history");
    }
}
void ValidateStructuredSmall(Metal::Device& device,Metal::Scheduler& scheduler,
                             Metal::ShaderMachineRuntime& machine) {
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return std::uint32_t(op)<<26;};
    Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
    p.swizzle[0]=Metal::Test::IdentitySwizzle;
    p.code[0]=Metal::Test::Arithmetic(Op::MOV,17,16);
    p.code[1]=word(Op::LOOP)|(5u<<10);
    p.code[2]=word(Op::LOOP)|(1u<<22)|(4u<<10);
    p.code[3]=Metal::Test::Arithmetic(Op::MOV,16,0);
    p.code[4]=word(Op::NOP);p.code[5]=word(Op::NOP);
    p.code[6]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<12);
    p.code[7]=word(Op::IFC)|(10u<<10)|1u|(2u<<22)|(1u<<25);
    p.code[8]=Metal::Test::Arithmetic(Op::MOV,0,17);
    p.code[9]=word(Op::NOP);
    p.code[10]=Metal::Test::Arithmetic(Op::MOV,0,1);
    p.code[11]=word(Op::END);
    Check(Metal::ShaderMachineRuntime::HasNativeResetControlFlow(p),
          "Structured small fixture does not satisfy stack-boundary proof");
    // PICA processes just one IF pop per advance. A taken child whose resume
    // equals its parent's end reaches that PC without popping the parent.
    // Ordinary nested host/GPU control would incorrectly skip the parent's else.
    auto colliding=p;
    colliding.code.fill(0);colliding.code[0]=Metal::Test::Arithmetic(Op::MOV,17,16);
    colliding.code[1]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<12);
    colliding.code[2]=word(Op::IFC)|(8u<<10)|1u|(2u<<22)|(1u<<25);
    colliding.code[3]=word(Op::IFC)|(6u<<10)|2u|(2u<<22)|(1u<<25);
    colliding.code[4]=Metal::Test::Arithmetic(Op::MOV,0,17);
    colliding.code[5]=word(Op::NOP);
    colliding.code[6]=Metal::Test::Arithmetic(Op::MOV,16,0);
    colliding.code[7]=word(Op::NOP);
    colliding.code[8]=Metal::Test::Arithmetic(Op::MOV,16,1);
    colliding.code[9]=word(Op::END);
    Check(!Metal::ShaderMachineRuntime::HasNativeResetControlFlow(colliding),
          "Nested IF resume collision incorrectly accepted native structured control");
    auto branch_end=p;branch_end.code[9]=word(Op::END);
    auto loop_end=p;loop_end.code[4]=word(Op::END);
    Check(Metal::ShaderMachineRuntime::HasNativeResetControlFlow(branch_end) &&
          Metal::ShaderMachineRuntime::HasNativeResetControlFlow(loop_end),
          "Legal early-END structured fixtures were rejected");
    auto input=device.CreateBuffer(6*256,true);
    const Metal::BufferSlice source{input,0,input.length,input.contents,{}};
    Metal::TransferRuntime transfer{device,scheduler};unsigned cases=0,failures=0;
    for(const auto& program:std::array{p,colliding,branch_end,loop_end})
    for(const auto counts:std::array<std::array<unsigned,2>,5>{{{0,0},{1,2},{255,0},{0,255},{3,4}}})
    for(unsigned reset:{0u,1u}) for(unsigned bank:{0u,1u}) for(unsigned branch:{0u,1u}) {
        Metal::ShaderMachineUniforms u{};
        u.i[0]={0xa500u|counts[0],0xb703u,0xc905u,0xffffffffu};
        u.i[1]={0xfe00u|counts[1],0xed07u,0xdc0bu,0xabcdu};
        std::uint32_t passes{};
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,u,nullptr,&passes)==
              Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses && passes==2,
              "Structured small fixture lost its two-pass dependency");
        auto* vertices=static_cast<Metal::VertexRegisters*>(input.contents);
        for(unsigned v=0;v<6;++v) for(unsigned r=0;r<16;++r)
            vertices[v][r]={branch ? -.5f:.5f,float(r)*.125f,-.25f,.75f};
        Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
        if(!reset) {
            for(unsigned at=0;at<448;++at) {
                const std::uint32_t bits=0x3e000000u+at*257u;
                std::memcpy(initial.history.data()+at*4,&bits,4);
            }
            const std::array<std::uint32_t,8> tail{3,4,7,0,1,0,bank,13};
            std::memcpy(initial.history.data()+1792,tail.data(),32);
        }
        for(unsigned limit:{1u,2u,5u,6u,7u,12u,1024u,1000000u}) {
            Metal::ShaderMachineState scalar{device},batched{device};
            machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
            std::vector<Metal::VertexRegisters> expected(6);
            std::string scalar_error,batch_error;
            try {
                for(unsigned v=0;v<6;++v) {
                    const auto result=machine.Run(scalar,program,u,
                        {input,v*256,256,vertices+v,{}},0xffff,192,limit);
                    std::memcpy(&expected[v],result.output.data,256);
                }
            } catch(const Metal::ShaderMachineError& error) {scalar_error=error.what();}
            Metal::BufferSlice raw{};
            try {raw=machine.RunVertexBatch(batched,program,u,source,6,6,0,0,{},0,limit);}
            catch(const Metal::ShaderMachineError& error) {batch_error=error.what();}
            Check(scalar_error==batch_error,"Structured small instruction limit changed exact error/PC");
            if(!batch_error.empty()) {
                Check(machine.SaveState(batched)==initial,"Structured small failure committed partial state");
                ++failures;
            } else {
                auto readback=device.CreateBuffer(raw.size,true);
                transfer.CopyBuffer(raw.buffer,raw.offset,readback,0,raw.size);scheduler.Finish();
                Check(!std::memcmp(readback.contents,expected.data(),raw.size) &&
                      machine.SaveState(batched)==machine.SaveState(scalar),
                      "Structured small nested loop/branch changed output or full1824-byte state");
            }
            ++cases;
        }
    }
    std::cout<<"Structured small native control: "<<cases<<" exact cases, "<<failures<<
        " atomic failures; nested masked256-iteration loops, address increments, early END, excluded IF-resume collisions, both branches/banks, reset/inherited, instruction-limit PC and complete outputs/1824-byte states match scalar GPU\n";
}
void ValidateSmallSerialRanges(Metal::Device& device,Metal::Scheduler& scheduler,
                               Metal::ShaderMachineRuntime& machine) {
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return std::uint32_t(op)<<26;};
    Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=1;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    program.code[0]=Metal::Test::Arithmetic(Op::MOV,17,16);
    program.code[1]=Metal::Test::Arithmetic(Op::MOV,16,0);
    program.code[2]=Metal::Test::Arithmetic(Op::MOV,0,17);
    program.code[3]=word(Op::END);
    Metal::ShaderMachineUniforms uniforms{};
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==
          Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,
          "Small range fixture lost its retained-state dependency");
    constexpr unsigned vertices=10,base=37;
    auto shared=device.CreateBuffer(vertices*256+64,true);
    auto input=device.CreateBuffer(shared.length,false);
    auto* values=reinterpret_cast<Metal::VertexRegisters*>(static_cast<std::byte*>(shared.contents)+32);
    Metal::TransferRuntime transfer{device,scheduler};
    const Metal::BufferSlice source{input,32,vertices*256,nullptr,{}};
    unsigned exact=0,failures=0;
    const auto initial_state=[](bool reset,unsigned bank) {
        Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
        if(!reset) {
            for(unsigned at=0;at<448;++at) {
                const std::uint32_t bits=0x3e000000u+at*257u;
                std::memcpy(initial.history.data()+at*4,&bits,4);
            }
            const std::array<std::uint32_t,8> tail{3,4,7,0,1,0,bank,13};
            std::memcpy(initial.history.data()+1792,tail.data(),32);
        }
        return initial;
    };
    for(unsigned count=1;count<=8;++count) for(unsigned reset:{0u,1u})
    for(unsigned bank:{0u,1u}) for(unsigned first:{0u,2u}) {
        for(unsigned v=0;v<vertices;++v) for(unsigned r=0;r<16;++r)
            values[v][r]={float(v+1)*.125f,float(r+1)*-.25f,.5f,-.75f};
        transfer.CopyBuffer(shared,0,input,0,shared.length);
        Metal::ShaderMachineState scalar{device},batched{device};
        const auto initial=initial_state(reset,bank);
        machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
        std::vector<Metal::VertexRegisters> expected(count);
        for(unsigned v=0;v<count;++v) {
            const auto result=machine.Run(scalar,program,uniforms,
                {input,32+(first+v)*256,256,nullptr,{}},0xffff);
            std::memcpy(&expected[v],result.output.data,256);
        }
        const auto raw=machine.RunVertexBatch(batched,program,uniforms,source,
                                              vertices,count,base,base+first);
        Check(raw.size==count*256,"Small range returned the wrong output extent");
        auto readback=device.CreateBuffer(raw.size+32,true);
        std::memset(readback.contents,0xa5,readback.length);
        transfer.CopyBuffer(raw.buffer,raw.offset,readback,16,raw.size);scheduler.Finish();
        Check(!std::memcmp(static_cast<std::byte*>(readback.contents)+16,expected.data(),raw.size) &&
              machine.SaveState(batched)==machine.SaveState(scalar),
              "Small ordered range changed output or complete1824-byte state");
        const auto* bytes=static_cast<const unsigned char*>(readback.contents);
        for(unsigned at=0;at<16;++at)
            Check(bytes[at]==0xa5 && bytes[16+raw.size+at]==0xa5,
                  "Small ordered range readback changed canaries");
        ++exact;
    }
    // Earlier invocations finish successfully; a later input takes two extra
    // instructions and exceeds its own limit. The whole batch must leave both
    // physical banks and every other stored word at the original snapshot.
    auto failing=program;
    failing.code[2]=word(Op::CMP)|(2u<<24)|(2u<<21)|(32u<<12);
    failing.code[3]=word(Op::IFC)|(6u<<10)|(2u<<22)|(1u<<25);
    failing.code[4]=failing.code[5]=word(Op::NOP);
    failing.code[6]=Metal::Test::Arithmetic(Op::MOV,0,17);
    failing.code[7]=word(Op::END);
    Check(Metal::ShaderMachineRuntime::AnalyzeVertices(failing,uniforms)==
          Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,
          "Late failure fixture lost its retained-state dependency");
    for(unsigned count=1;count<=8;++count) for(unsigned fault=0;fault<count;++fault)
    for(unsigned reset:{0u,1u}) for(unsigned bank:{0u,1u}) {
        for(unsigned v=0;v<vertices;++v) for(unsigned r=0;r<16;++r)
            values[v][r]={v==fault+2 ? .5f:-.5f,.125f,-.25f,.75f};
        transfer.CopyBuffer(shared,0,input,0,shared.length);
        Metal::ShaderMachineState scalar{device},batched{device};
        const auto initial=initial_state(reset,bank);
        machine.LoadState(scalar,initial);machine.LoadState(batched,initial);
        std::string scalar_error,batch_error;unsigned completed=0;
        try {
            for(unsigned v=0;v<count;++v) {
                machine.Run(scalar,failing,uniforms,{input,32+(v+2)*256,256,nullptr,{}},
                            0xffff,192,6);
                ++completed;
            }
        } catch(const Metal::ShaderMachineError& error) {scalar_error=error.what();}
        try {machine.RunVertexBatch(batched,failing,uniforms,source,vertices,count,
                                    base,base+2,{},0,6);}
        catch(const Metal::ShaderMachineError& error) {batch_error=error.what();}
        Check(completed==fault && !scalar_error.empty() && scalar_error==batch_error,
              "Small ordered late failure changed first failing invocation or exact PC");
        Check(machine.SaveState(batched)==initial,
              "Small ordered late failure committed a successful prefix to stored state");
        ++failures;
    }
    std::cout<<"Small native ordered ranges: "<<exact<<" exact outputs/full-state cases and "
        <<failures<<" first/late atomic failures; counts1..8, nonzero base/offset, private GPU input, "
          "reset/inherited banks, output canaries and exact first-failure PC passed\n";
}
void BenchmarkSmallRecords(Metal::Device& device,Metal::Scheduler& scheduler,
                           Metal::ShaderMachineRuntime& machine,
                           const std::filesystem::path& directory,bool queued_scalar=false) {
    std::vector<std::filesystem::path> paths;
    for(const auto& entry:std::filesystem::directory_iterator(directory))
        if(entry.path().extension()==".vdc") paths.push_back(entry.path());
    std::sort(paths.begin(),paths.end());
    unsigned records=0;
    for(const auto& path:paths) {
        std::ifstream stream(path,std::ios::binary);D::Header h{};
        stream.read(reinterpret_cast<char*>(h.data()),sizeof(h));
        Check(stream && h[D::Magic]==D::Signature && h[D::Version]==1,
              "Invalid real-input benchmark record");
        if(h[D::Execution]!=unsigned(Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses) ||
           h[D::IndexWidth] || !h[D::InitialReset] || h[D::Elements]>8) continue;
        // Validate every recorded output/state byte and the scalar oracle before
        // timing. These are the actual inputs/uniforms, not synthetic vertices.
        Replay(device,scheduler,machine,path,true);
        Metal::ShaderMachineProgram program{};program.allow_emission=false;
        program.entry=h[D::Entry];program.output_mask=h[D::OutputMask];
        stream.read(reinterpret_cast<char*>(program.code.data()),sizeof(program.code));
        stream.read(reinterpret_cast<char*>(program.swizzle.data()),sizeof(program.swizzle));
        Metal::ShaderMachineUniforms uniforms{};stream.read(reinterpret_cast<char*>(&uniforms),sizeof(uniforms));
        stream.seekg(1824,std::ios::cur);
        auto input=device.CreateBuffer(h[D::InputBytes],true);
        stream.read(static_cast<char*>(input.contents),input.length);
        Check(bool(stream),"Truncated real-input benchmark record");
        const Metal::BufferSlice source{input,0,input.length,input.contents,{}};
        Metal::ShaderMachineState state{device};
        const auto scalar=[&](unsigned element) {
            const auto offset=NSUInteger(h[D::FirstVertex]-h[D::VertexBase]+element)*256;
            return machine.RunVertexDeferred(state,program,uniforms,
                {input,offset,256,static_cast<std::byte*>(input.contents)+offset,{}},
                0xffff,h[D::InstructionLimit]);
        };
        if(queued_scalar) {
            // Independently captured outputs and complete final history remain
            // the oracle even when queued scalar code gets a new compiler path.
            std::vector<std::byte> expected(h[D::OutputBytes]);
            stream.read(reinterpret_cast<char*>(expected.data()),expected.size());
            Metal::ShaderMachineSnapshot final{};final.needs_reset=false;
            stream.read(reinterpret_cast<char*>(final.history.data()),final.history.size());
            Check(bool(stream),"Truncated queued-scalar expected output/history");
            auto readback=device.CreateBuffer(expected.size(),true);
            Metal::TransferRuntime transfer{device,scheduler};
            for(unsigned element=0;element<h[D::Elements];++element) {
                const auto value=scalar(element);
                transfer.CopyBuffer(value.buffer,value.offset,readback,element*256,256);
            }
            scheduler.Finish();
            Check(!std::memcmp(readback.contents,expected.data(),expected.size()) &&
                  machine.SaveState(state)==final,
                  "Queued scalar real inputs changed recorded output or1824-byte state");
        }
        const auto draw=[&] {
            state.Reset();
            if(queued_scalar) for(unsigned element=0;element<h[D::Elements];++element) scalar(element);
            else machine.RunVertexBatch(state,program,uniforms,source,h[D::Vertices],
                h[D::Elements],h[D::VertexBase],h[D::FirstVertex],{},0,h[D::InstructionLimit],true);
        };
        draw();scheduler.EndProfileFrame();scheduler.Finish();
        for(unsigned round=0;round<3;++round) {
            const auto frame=scheduler.CurrentFrame();
            const auto start=std::chrono::steady_clock::now();
            for(unsigned repeat=0;repeat<64;++repeat) draw();
            scheduler.EndProfileFrame();scheduler.Finish();
            const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-start).count();
            std::cout<<(queued_scalar ? "REAL_SCALAR_BENCH record=":"REAL_SMALL_BENCH record=")<<path.filename().string()<<" frame="<<frame<<
                " elements="<<h[D::Elements]<<" round="<<round<<" draws=64 host_ns="<<elapsed<<'\n';
        }
        ++records;
    }
    Check(records>0,"No reset nonindexed small dependency records found");
    std::cout<<(queued_scalar ? "Real queued scalar benchmark: ":"Real small dependency benchmark: ")<<records<<
        " complete recorded input/uniform/output/state cases match scalar GPU; 3 warm blocks of64 draws each\n";
}
void ValidateTargetFilter(Metal::Device& device,Metal::Scheduler& scheduler,
                          Metal::ShaderMachineRuntime& reference,
                          const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",directory.c_str(),1);
    Metal::ShaderMachineRuntime captured{device,scheduler};
    unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
    Metal::ShaderMachineState state{device};
    Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
    p.swizzle[0]=Metal::Test::IdentitySwizzle;p.code[0]=0x13u<<26;p.code[1]=0x22u<<26;
    auto input=device.CreateBuffer(7*256,true);
    auto* vertices=static_cast<Metal::VertexRegisters*>(input.contents);
    for(unsigned v=0;v<7;++v)for(unsigned r=0;r<16;++r)
        vertices[v][r]={float(v+r)*.125f,-.25f,.5f,.75f};
    const Metal::BufferSlice source{input,0,input.length,input.contents,{}};
    const auto records=[&] {
        unsigned count=0;for(const auto& entry:std::filesystem::directory_iterator(directory))
            count+=entry.path().extension()==".vdc";
        return count;
    };
    const auto arm=[&](const char* text) {
        {std::ofstream request(directory/"capture.request");request<<text;}
        scheduler.EndProfileFrame();
    };
    const auto draw=[&](unsigned count,bool no_capture) {
        const auto tick=scheduler.CurrentTick();
        captured.RunVertexBatch(state,p,{},source,7,count,0,0,{},0,1000000,true);
        if(no_capture)Check(scheduler.CurrentTick()==tick,
            "Excluded/invalid/same-class capture forced GPU submission or wait");
    };
    // Five and seven share a size class. An earlier excluded draw must neither
    // consume that class nor force diagnostic GPU readbacks/submission.
    arm("1 7");draw(5,true);Check(records()==0,"Wrong-sized draw was captured");
    draw(7,false);Check(records()==1,"Exact-size target was lost behind a smaller size-class peer");
    draw(7,true);Check(records()==1,"Same target class was captured twice");
    arm("1 5");draw(5,true);draw(7,true);
    Check(records()==1,"Changing target without a new epoch rearmed capture");
    arm("2 5");draw(7,true);draw(5,false);
    Check(records()==2,"New epoch did not change exact-size target");
    arm("3");draw(5,false);draw(7,true);
    Check(records()==3,"Legacy epoch-only request lost size-class deduplication");
    for(const char* invalid:{"4 nope","4 7 trailing","4 4294967296","4 -1","0 7","","4 7.5","4 frame trailing","4 frames","4 frame 7","4 fram","4 Frame"}) {
        arm(invalid);draw(5,true);draw(7,true);
        Check(records()==3,"Malformed/disarmed request captured or consumed a draw");
    }
    arm("5 7\n");draw(5,true);draw(7,false);
    Check(records()==4,"Valid target did not recover after malformed requests");
    arm("6 0\n");draw(5,false);draw(7,true);
    Check(records()==5,"Explicit zero did not retain legacy all-size mode");
    for(const auto& entry:std::filesystem::directory_iterator(directory))
        if(entry.path().extension()==".vdc")Replay(device,scheduler,reference,entry.path(),true);
    std::cout<<"Five exact-filter/legacy diagnostic records replay bit exactly against scalar GPU outputs/state; excluded/same-class/malformed requests never force submission\n";
}
void ValidateFrameCapture(Metal::Device& device,Metal::Scheduler& scheduler,
                          Metal::ShaderMachineRuntime& reference,const std::filesystem::path& root) {
    for(unsigned mode=0;mode<3;++mode) {
        const auto directory=root/std::to_string(mode);std::filesystem::create_directories(directory);
        setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",directory.c_str(),1);
        Metal::ShaderMachineRuntime captured{device,scheduler};
        unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
        Metal::ShaderMachineState state{device};Metal::ShaderMachineProgram p{};
        p.allow_emission=false;p.output_mask=1;p.swizzle[0]=Metal::Test::IdentitySwizzle;
        p.code[0]=(0x13u<<26)|(32u<<12);p.code[1]=0x22u<<26;
        auto input=device.CreateBuffer(256,true);std::memset(input.contents,0,256);
        auto indices=device.CreateBuffer(66000,true);std::memset(indices.contents,0,indices.length);
        const Metal::BufferSlice source{input,0,input.length,input.contents,{}};
        const auto frame=scheduler.CurrentFrame();
        {std::ofstream arm(directory/"capture.request");arm<<"100 frame\n";}
        const unsigned draws=mode==0 ? 20:mode==1 ? 513:9;
        const unsigned expected=mode==0 ? 20:mode==1 ? 512:7;
        for(unsigned draw=0;draw<draws;++draw) {
            Metal::ShaderMachineUniforms u{};u.f[0]={float(draw)*.125f,-.25f,.5f,1};
            state.Reset();const unsigned count=mode==2 ? draw==8 ? 66000:65200:1;
            if(!mode) {
                Metal::GeometryEmissionConfig config{};config.attribute_count=1;config.honor_winding=0;
                config.semantics[0]=0x03020100;config.flip_viewport=draw&1;config.enable_clip=1;
                config.clip_coefficient={1,.5f,-.25f,0};
                captured.RunVertexBatchAndCapture(state,p,u,source,config,1,count,0,0,{},0,1000000,true);
            } else captured.RunVertexBatch(state,p,u,source,1,count,0,0,
                mode==2 ? Metal::BufferSlice{indices,0,count,indices.contents,{}}:Metal::BufferSlice{},
                mode==2 ? 1:0,1000000,true);
        }
        Check(!std::filesystem::exists(directory/("frame-100-"+std::to_string(frame)+".json")),
              "Frame capture claimed completion before the next observed frame");
        scheduler.EndProfileFrame();
        const auto tick=scheduler.CurrentTick();
        captured.RunVertexBatch(state,p,{},source,1,1,0,0,{},0,1000000,true);
        Check(scheduler.CurrentTick()==tick,"Closed frame capture forced a GPU wait");
        {std::ofstream arm(directory/"capture.request");arm<<"101 frame\n";}
        scheduler.EndProfileFrame();
        captured.RunVertexBatch(state,p,{},source,1,1,0,0,{},0,1000000,true);
        Check(scheduler.CurrentTick()==tick,"Consumed one-frame capture was rearmed");
        const auto manifest=directory/("frame-100-"+std::to_string(frame)+".json");
        std::ifstream stream(manifest);const std::string json{std::istreambuf_iterator<char>(stream),{}};
        Check(bool(stream) || stream.eof(),"Frame capture manifest missing");
        const auto number=[&](std::string key) {
            const auto pos=json.find("\""+key+"\":");Check(pos!=std::string::npos,"Frame manifest field missing");
            return std::stoull(json.substr(pos+key.size()+3));
        };
        Check(number("frame")==frame && number("attempted")==draws && number("selected")==expected &&
              number("written")==expected && number("skipped_record_limit")==unsigned(mode==1) &&
              number("skipped_byte_budget")==unsigned(mode==2) && number("skipped_size")==unsigned(mode==2) &&
              number("skipped_alias")==0 && number("reserved_bytes")==number("written_bytes") &&
              number("conversion_records")==unsigned(mode==0 ? 20:0),
              "Frame capture counts/bounds disagree with actual records");
        Check(json.find(mode==0 ? "\"complete_observed_stage\":true":"\"complete_observed_stage\":false")!=std::string::npos &&
              json.find("\"complete_render_frame\":false")!=std::string::npos,
              "Incomplete/stage-only frame capture claimed full renderer coverage");
        unsigned records=0;std::uint64_t bytes=0;
        for(const auto& entry:std::filesystem::directory_iterator(directory)) {
            Check(entry.path().extension()!=".part","Frame capture left an incomplete publication");
            if(entry.path().extension()!=".vdc")continue;
            D::Header h{};std::ifstream f(entry.path(),std::ios::binary);f.read(reinterpret_cast<char*>(h.data()),sizeof(h));
            Check(bool(f) && h[D::FrameLow]==std::uint32_t(frame) && h[D::FrameHigh]==std::uint32_t(frame>>32) &&
                  h[D::RequestLow]==100 && h[D::RequestHigh]==0,"Frame record belongs to another epoch/frame");
            if(!mode) {
                const auto sidecar=std::filesystem::path(entry.path().string()+".capture.bin");
                Metal::GeometryEmissionConfig config{};std::ifstream c(sidecar,std::ios::binary);
                c.read(reinterpret_cast<char*>(&config),sizeof(config));
                Check(bool(c) && c.peek()==std::char_traits<char>::eof() && config.attribute_count==1 &&
                      config.semantics[0]==0x03020100 && config.flip_viewport==((h[D::DrawLow]-1)&1u) &&
                      config.enable_clip==1 && config.honor_winding==0 &&
                      config.clip_coefficient==std::array<float,4>{1,.5f,-.25f,0},
                      "Frame conversion metadata changed or outlived its immutable snapshot");
                bytes+=std::filesystem::file_size(sidecar);
            }
            Replay(device,scheduler,reference,entry.path(),true);++records;bytes+=entry.file_size();
        }
        Check(records==expected && bytes==number("written_bytes"),"Frame manifest did not cover all actual files/bytes");
    }
    std::cout<<"3 one-frame captures /542 observed draws:20 duplicate draws retained;512-record and128MiB bounds, oversized skips, exact manifests/20 real conversion sidecars, atomic files and one-shot disarming; all539 records replay against scalar GPU outputs/full1824 states\n";
}

void ValidateDeferredState(Metal::Device& device,Metal::Scheduler& scheduler,
                           Metal::ShaderMachineRuntime& machine) {
    Metal::TransferRuntime transfer{device,scheduler};
    Metal::ShaderMachineRuntime reference{device,scheduler};
    constexpr unsigned vertices=67,base=8;
    Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=3;
    program.swizzle[0]=Metal::Test::IdentitySwizzle;
    program.code[0]=(16u<<21)|(16u<<12); // ADD r0,r0,v0: retained recurrence.
    program.code[1]=(17u<<21)|(32u<<12)|(1u<<7); // ADD r1,c0,v1.
    program.code[2]=(0x13u<<26)|(16u<<12);
    program.code[3]=(0x13u<<26)|(1u<<21)|(17u<<12);
    program.code[4]=0x22u<<26;
    Metal::ShaderMachineUniforms uniforms{};uniforms.f[0]={1.25f,-.5f,.125f,0};
    std::vector<Metal::VertexRegisters> values(vertices);
    for(unsigned v=0;v<vertices;++v) for(unsigned reg=0;reg<16;++reg)
        values[v][reg]={float(v+reg)*.125f,-.25f,.5f,.75f};
    constexpr std::array<unsigned,4> payload{0x80000000,0x7fc12345,0x7f800000,0xff800000};
    std::memcpy(values.back()[15].data(),payload.data(),16);
    unsigned cases=0;
    for(unsigned width:std::array{0u,1u,2u,4u}) {
        std::vector<unsigned> ids;
        if(width) {
            for(unsigned e=0;e<67;++e) ids.push_back(base+e);
            for(unsigned e:std::array{0u,65u,1u,66u,0u,64u,1u}) ids.push_back(base+e);
        } else for(unsigned v=0;v<vertices;++v) ids.push_back(base+v);
        auto input=device.CreateBuffer(vertices*256+64,true);std::memset(input.contents,0,input.length);
        std::memcpy(static_cast<std::byte*>(input.contents)+32,values.data(),vertices*256);
        auto indices=device.CreateBuffer(ids.size()*std::max(width,1u)+8,true);std::memset(indices.contents,0,indices.length);
        if(width) for(unsigned e=0;e<ids.size();++e)
            std::memcpy(static_cast<std::byte*>(indices.contents)+3+e*width,&ids[e],width);
        const Metal::BufferSlice source{input,32,vertices*256,static_cast<std::byte*>(input.contents)+32,{}};
        const Metal::BufferSlice ix{indices,3,ids.size()*width,static_cast<std::byte*>(indices.contents)+3,{}};
        const auto index=width ? ix:Metal::BufferSlice{};
        Metal::ShaderMachineState oracle{device};
        reference.RunVertexBatch(oracle,program,uniforms,source,vertices,ids.size(),base,base,index,width,1000000,true);
        const auto expected=reference.SaveState(oracle);
        Metal::ShaderMachineState state{device};
        const auto defer=[&](const Metal::ShaderMachineProgram& p,const Metal::ShaderMachineUniforms& u) {
            machine.DeferResetVertexBatch(state,p,u,source,vertices,ids.size(),base,base,index,width);
            Check(state.HasDeferredVertexBatch(),"Deferred batch did not own its pending snapshot");
        };
        // Later GPU writes, before the first materialization, must not change a
        // single saved input/temp/output/address/flag/emission-bank byte.
        auto caller_program=program;auto caller_uniforms=uniforms;
        defer(caller_program,caller_uniforms);
        transfer.FillBuffer(input,NSMakeRange(0,input.length),0xa5);
        transfer.FillBuffer(indices,NSMakeRange(0,indices.length),0xff);
        caller_program.code.fill(0x84000000);caller_program.swizzle.fill(0);
        caller_uniforms={};
        const Metal::ShaderMachineState& logical_const=state;
        Check(machine.SaveState(logical_const)==expected && !state.HasDeferredVertexBatch(),
              "Deferred snapshot borrowed mutated input/index/program/uniform storage");++cases;
        // Restore callers after retiring the queued mutation. Subsequent test
        // cases still avoid a wait between snapshot copies and their consumers.
        const auto restore=[&] {
            scheduler.Finish();std::memcpy(static_cast<std::byte*>(input.contents)+32,values.data(),vertices*256);
            if(width) for(unsigned e=0;e<ids.size();++e)
                std::memcpy(static_cast<std::byte*>(indices.contents)+3+e*width,&ids[e],width);
        };restore();
        // Reset cancels without executing a VS or forcing a submission.
        defer(program,uniforms);const auto tick=scheduler.CurrentTick();state.Reset();
        Check(machine.SaveState(state)==Metal::ShaderMachineSnapshot{} && !state.HasDeferredVertexBatch() &&
              scheduler.CurrentTick()==tick,"Reset materialized a discarded deferred batch");++cases;
        defer(program,uniforms);auto invalid=expected;invalid.history[1804]=std::byte{1};
        bool rejected=false;try {machine.LoadState(state,invalid);} catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected && state.HasDeferredVertexBatch() && machine.SaveState(state)==expected,
              "Invalid LoadState discarded pending represented state");++cases;
        defer(program,uniforms);machine.LoadState(state,Metal::ShaderMachineSnapshot{});
        Check(!state.HasDeferredVertexBatch() && machine.SaveState(state)==Metal::ShaderMachineSnapshot{},
              "Valid reset LoadState retained stale pending data");++cases;
        defer(program,uniforms);machine.LoadState(state,expected);
        Check(!state.HasDeferredVertexBatch() && machine.SaveState(state)==expected,
              "Valid inherited LoadState retained stale pending data");++cases;
        // Invalid host calls must reject before materialization/copy submission.
        defer(program,uniforms);const auto before=scheduler.CurrentTick();rejected=false;
        try {machine.DeferResetVertexBatch(state,program,uniforms,source,vertices,ids.size(),base,base,index,3);}
        catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected && state.HasDeferredVertexBatch() && scheduler.CurrentTick()==before,
              "Invalid deferred metadata changed or submitted pending state");
        rejected=false;try {machine.RunVertexBatch(state,program,uniforms,{},vertices,ids.size(),base,base,index,width);}
        catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected && state.HasDeferredVertexBatch() && machine.SaveState(state)==expected,
              "Invalid ordinary batch materialized/discarded pending state");++cases;
        // Rejected foreign-thread calls must not detach or replace the snapshot.
        defer(program,uniforms);std::array<bool,2> thread_rejected{};
        std::thread outsider([&] {
            try {machine.DeferResetVertexBatch(state,program,uniforms,source,vertices,ids.size(),base,base,index,width);}
            catch(const std::logic_error&) {thread_rejected[0]=true;}
            try {machine.SaveState(state);} catch(const std::logic_error&) {thread_rejected[1]=true;}
        });outsider.join();
        Check(thread_rejected[0] && thread_rejected[1] && state.HasDeferredVertexBatch() &&
              machine.SaveState(state)==expected,"Foreign thread detached/changed a deferred snapshot");++cases;
        // An inherited scalar invocation and inherited batch materialize first.
        const Metal::BufferSlice one{input,32,256,source.data,{}};
        defer(program,uniforms);reference.LoadState(oracle,expected);
        const auto scalar_expected=reference.Run(oracle,program,uniforms,one);
        const auto scalar_actual=machine.Run(state,program,uniforms,one);
        Check(!std::memcmp(scalar_expected.output.data,scalar_actual.output.data,256) &&
              machine.SaveState(state)==reference.SaveState(oracle),"Inherited scalar skipped deferred VS history");++cases;
        defer(program,uniforms);reference.LoadState(oracle,expected);
        const auto raw_expected=reference.RunVertexBatch(oracle,program,uniforms,source,vertices,ids.size(),base,base,index,width,1000000,true);
        const auto raw_actual=machine.RunVertexBatch(state,program,uniforms,source,vertices,ids.size(),base,base,index,width,1000000,true);
        auto bytes=device.CreateBuffer(raw_expected.size*2,true);
        transfer.CopyBuffer(raw_expected.buffer,raw_expected.offset,bytes,0,raw_expected.size);
        transfer.CopyBuffer(raw_actual.buffer,raw_actual.offset,bytes,raw_expected.size,raw_actual.size);scheduler.Finish();
        Check(!std::memcmp(bytes.contents,static_cast<std::byte*>(bytes.contents)+raw_expected.size,raw_expected.size) &&
              machine.SaveState(state)==reference.SaveState(oracle),"Inherited batch skipped deferred VS history");++cases;
        // New deferred reset replaces the old batch, never composes with it.
        auto second=uniforms;second.f[0]={-.25f,.5f,-.75f,1};
        defer(program,uniforms);defer(program,second);oracle.Reset();
        reference.RunVertexBatch(oracle,program,second,source,vertices,ids.size(),base,base,index,width);
        Check(machine.SaveState(state)==reference.SaveState(oracle),"Deferred reset batches composed instead of replacing");++cases;
        // Program cache eviction and originating runtime destruction cannot
        // invalidate an owned program/uniform/input snapshot.
        if(width==1) {
            defer(program,uniforms);Metal::ShaderMachineState scratch{device};
            for(unsigned i=0;i<260;++i) {
                auto variant=program;variant.swizzle[4095]=0x1000+i;
                machine.DeferResetVertexBatch(scratch,variant,uniforms,one,1,1,0,0);scratch.Reset();
            }
            Check(machine.SaveState(state)==expected,"Program eviction invalidated deferred source snapshot");++cases;
            {Metal::ShaderMachineRuntime temporary{device,scheduler};
                temporary.DeferResetVertexBatch(state,program,uniforms,source,vertices,ids.size(),base,base,index,width);}
            Check(machine.SaveState(state)==expected,"Originating runtime destruction invalidated deferred snapshot");++cases;
        }
    }
    // Deferred reached errors must propagate on materialization and remain
    // recoverable by Reset, with no partially represented saved state.
    Metal::ShaderMachineState bad{device};auto invalid=program;
    invalid.code[0]=(0x13u<<26);invalid.code[1]=0x2au<<26;invalid.code[2]=0x22u<<26;
    const auto input=device.CreateBuffer(256,true);std::memcpy(input.contents,values.data(),256);
    machine.DeferResetVertexBatch(bad,invalid,uniforms,{input,0,256,input.contents,{}},1,1,0,0);
    bool failed=false;try {machine.SaveState(bad);} catch(const Metal::ShaderMachineError& e) {failed=e.Code()==2;}
    Check(failed && bad.HasDeferredVertexBatch(),"Deferred illegal instruction did not preserve/propagate its error");
    bad.Reset();Check(machine.SaveState(bad)==Metal::ShaderMachineSnapshot{},"Reset failed after deferred execution error");++cases;
    std::cout<<"Deferred full VS state: "<<cases<<" mutation/reset/load/inherited/FIFO/eviction/lifetime/error cases preserve complete1824-byte GPU state\n";
}
}
int main(int argc,char** argv) {
    @autoreleasepool {try {
        std::cout.setf(std::ios::unitbuf);
        if(argc==2 && std::string_view(argv[1])=="--structured-small-only") {
            unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
            Metal::Device device;Metal::Scheduler scheduler{device};
            Metal::ShaderMachineRuntime machine{device,scheduler};
            ValidateStructuredSmall(device,scheduler,machine);
            ValidateSmallSerialRanges(device,scheduler,machine);return 0;
        }
        if(argc==3 && (std::string_view(argv[1])=="--benchmark-small" ||
                      std::string_view(argv[1])=="--benchmark-scalar")) {
            unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
            Metal::Device device;Metal::Scheduler scheduler{device};
            Metal::ShaderMachineRuntime machine{device,scheduler};
            BenchmarkSmallRecords(device,scheduler,machine,argv[2],
                std::string_view(argv[1])=="--benchmark-scalar");return 0;
        }
        Check(argc==2 || (argc==3 && (std::string_view(argv[1])=="--replay" ||
              std::string_view(argv[1])=="--replay-queued")),"Expected output directory or replay directory");
        const bool replay=argc==3;
        const std::filesystem::path directory{argv[replay ? 2:1]};
        unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
        Metal::Device device;Metal::Scheduler scheduler{device};
        Metal::ShaderMachineRuntime reference{device,scheduler};
        if(!replay) {
            std::filesystem::create_directories(directory);
            Check(std::filesystem::is_empty(directory),"Diagnostic test directory must be empty");
            setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",directory.c_str(),1);
            Metal::ShaderMachineRuntime captured{device,scheduler};
            unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
            unsigned cases=0;
            for(unsigned kind=0;kind<3;++kind) for(unsigned reset=0;reset<2;++reset)
                for(unsigned width:std::array{0u,1u,2u,4u}) {
                ++cases;{std::ofstream arm(directory/"capture.request");arm<<cases;}
                scheduler.EndProfileFrame();
                Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
                p.swizzle[0]=Metal::Test::IdentitySwizzle;
                if(kind==0) {p.code[0]=0x13u<<26;p.code[1]=0x22u<<26;}
                else if(kind==1) {p.code[0]=(16u<<21)|(16u<<12);p.code[1]=(0x13u<<26)|(16u<<12);p.code[2]=0x22u<<26;}
                else {p.code[0]=(0x13u<<26)|(17u<<21)|(16u<<12);p.code[1]=(0x13u<<26)|(16u<<21);p.code[2]=(0x13u<<26)|(17u<<12);p.code[3]=0x22u<<26;}
                Metal::ShaderMachineUniforms u{};u.b=cases;u.f[0]={1.25f,-.5f,.125f,0};
                constexpr unsigned vertices=7,base=8;
                const std::vector<unsigned> ids=width ? std::vector<unsigned>{8,10,8,14,10,9,13,8,12,14}:std::vector<unsigned>{9,10,11,12,13};
                auto shared=device.CreateBuffer(vertices*256+64,true);
                auto* data=reinterpret_cast<Metal::VertexRegisters*>(static_cast<std::byte*>(shared.contents)+32);
                for(unsigned v=0;v<vertices;++v) for(unsigned r=0;r<16;++r) data[v][r]={float(v+r)*.125f,-.25f,.5f,.75f};
                auto input=device.CreateBuffer(shared.length,false);
                Metal::TransferRuntime transfer{device,scheduler};transfer.CopyBuffer(shared,0,input,0,shared.length);
                auto index_shared=device.CreateBuffer(128,true);std::memset(index_shared.contents,0,128);
                if(width) for(unsigned e=0;e<ids.size();++e) std::memcpy(static_cast<std::byte*>(index_shared.contents)+16+e*width,&ids[e],width);
                auto index=device.CreateBuffer(128,false);transfer.CopyBuffer(index_shared,0,index,0,128);
                Metal::ShaderMachineSnapshot initial{};initial.needs_reset=reset;
                if(!reset) {
                    for(unsigned word=0;word<451;++word) {const unsigned bits=0x3e000000u+word*23;std::memcpy(initial.history.data()+word*4,&bits,4);}
                    const std::array<unsigned,5> flags{0,1,0,kind%2,15};std::memcpy(initial.history.data()+1804,flags.data(),20);
                }
                Metal::ShaderMachineState state{device};captured.LoadState(state,initial);
                captured.RunVertexBatch(state,p,u,{input,32,vertices*256,nullptr,{}},vertices,ids.size(),base,9,
                    {index,16,128-16,nullptr,{}},width,1000000,true);
            }
            Check(cases==24,"Unexpected capture case count");
            const auto bounded=directory/"bounded";
            std::filesystem::create_directories(bounded);
            setenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS",bounded.c_str(),1);
            Metal::ShaderMachineRuntime limited{device,scheduler};
            unsetenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");
            Metal::ShaderMachineState state{device};
            Metal::ShaderMachineProgram p{};p.allow_emission=false;p.output_mask=1;
            p.swizzle[0]=Metal::Test::IdentitySwizzle;p.code[0]=0x13u<<26;p.code[1]=0x22u<<26;
            const auto input=device.CreateBuffer(256,true);std::memset(input.contents,0,256);
            const Metal::BufferSlice source{input,0,256,input.contents,{}};
            const auto tick=scheduler.CurrentTick();
            limited.RunVertexBatch(state,p,{},source,1,1,0,0,{},0,1000000,true);
            Check(scheduler.CurrentTick()==tick,"Disarmed capture forced a submission/wait");
            for(unsigned request=1;request<=5;++request) {
                {std::ofstream arm(bounded/"capture.request");arm<<request;}
                scheduler.EndProfileFrame();
                for(unsigned control=0;control<17;++control) {
                    Metal::ShaderMachineUniforms u{};u.b=control;
                    limited.RunVertexBatch(state,p,u,source,1,1,0,0,{},0,1000000,true);
                }
            }
            unsigned bounded_records=0;
            for(const auto& entry:std::filesystem::directory_iterator(bounded))
                if(entry.path().extension()==".vdc") ++bounded_records;
            Check(bounded_records==64,"Capture did not obey per-request/global bounds");
            scheduler.EndProfileFrame();
            ValidateTargetFilter(device,scheduler,reference,directory/"filtered");
            ValidateFrameCapture(device,scheduler,reference,directory/"frames");
            ValidateDeferredState(device,scheduler,reference);
        }
        unsigned records=0;
        for(const auto& entry:std::filesystem::directory_iterator(directory)) if(entry.path().extension()==".vdc") {
            Replay(device,scheduler,reference,entry.path(),true,
                argc==3 && std::string_view(argv[1])=="--replay-queued");++records;
        }
        Check(replay ? records>0:records==24,"Incomplete capture record count");
        std::cout<<"Native vertex diagnostics: "<<records<<" complete records replay bit exactly";
        std::cout<<" and match independent scalar GPU outputs/history";
        if(argc==3 && std::string_view(argv[1])=="--replay-queued")
            std::cout<<" plus specialized scalar queue outputs/full history with exact64-entry FIFO";
        std::cout<<'\n';scheduler.Finish();scheduler.EndProfileFrame();return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}}
}
