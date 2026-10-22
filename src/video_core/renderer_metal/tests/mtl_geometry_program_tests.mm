// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_geometry_program.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_draw_runtime.h"
#include "mtl_transfer.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <string_view>

namespace Metal {
struct InputReferenceTestAccess {
    static GeometryQueuedInvocation Compact(GeometryProgramRuntime& groups,GeometryInputState& state,
        ShaderMachineRuntime& machine,ShaderMachineState& unit,const ShaderMachineProgramView& program,
        const BufferSlice& abort,std::uint32_t capacity=1000000) {
        return groups.InvokeQueuedImpl(state,machine,unit,program,abort,capacity,{},true);
    }
    static bool VertexCompletes(ShaderMachineRuntime& machine,const ShaderMachineProgramView& program,
                                const ShaderMachineUniforms& uniforms={}) {
        return machine.ProvesGeometryVertexCompletion(program,uniforms);
    }
    static BufferSlice Fetched(ShaderMachineRuntime& machine,VertexFetchRuntime& fetch,VertexInputState& state,
        std::span<const BufferSlice> inputs,const PicaVertexLayout& layout,ShaderMachineState& unit,
        const ShaderMachineProgramView& program,const ShaderMachineUniforms& uniforms,unsigned vertices,
        unsigned elements,unsigned base,unsigned first,const BufferSlice& indices,unsigned width) {
        return machine.RunFetchedGeometryVertices(fetch,state,inputs,layout,unit,program,uniforms,
            vertices,elements,base,first,indices,width);
    }
    static BufferSlice Indirect(const AssembledVertices& vertices) { return vertices.indirect->arguments; }
    static BufferSlice History(const PrimitiveStream& stream) {
        return {stream.history,stream.history_offset,272,nullptr,{}};
    }
};
}

namespace {
void Check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
Metal::CapturedVertex Reference(const Metal::ShaderMachineEmission& emission,
    const Metal::GeometryEmissionConfig& config) {
    Pica::RasterizerRegs regs{};
    regs.vs_output_total.Assign(config.attribute_count);
    for(unsigned i=0;i<7;++i) regs.vs_output_attributes[i].raw=config.semantics[i];
    Pica::AttributeBuffer attributes{};
    for(unsigned i=0;i<16;++i) for(unsigned c=0;c<4;++c)
        attributes[i][c]=Pica::f24::FromFloat32(emission.attributes[i][c]);
    const Pica::OutputVertex vertex{regs,attributes};
    Metal::CapturedVertex result{};
    for(unsigned c=0;c<4;++c) {
        result.position[c]=vertex.pos[c].ToFloat32();
        result.color[c]=vertex.color[c].ToFloat32();
        result.quaternion[c]=vertex.quat[c].ToFloat32();
    }
    const float ndc=result.position[2]/result.position[3];
    if(ndc>0 && ndc<.000001f) result.position[2]=0;
    if(ndc<-1 && ndc>-1.00001f) result.position[2]=-result.position[3];
    if(config.flip_viewport) result.position[1]=-result.position[1];
    result.clip[0]=-result.position[2];
    if(config.enable_clip) for(unsigned c=0;c<4;++c) result.clip[1]+=config.clip_coefficient[c]*result.position[c];
    result.position[2]=-result.position[2];
    for(unsigned c=0;c<2;++c) {
        result.texcoord0[c]=vertex.tc0[c].ToFloat32();
        result.texcoord1[c]=vertex.tc1[c].ToFloat32();
        result.texcoord2[c]=vertex.tc2[c].ToFloat32();
    }
    result.clip[2]=vertex.tc0_w.ToFloat32();
    for(unsigned c=0;c<3;++c) result.view[c]=vertex.view[c].ToFloat32();
    return result;
}
}
Metal::ShaderMachineProgram RealLinearGeometryProgram() {
    Metal::ShaderMachineProgram p{};p.entry=148;p.output_mask=7;p.allow_emission=true;
    constexpr std::array<std::uint32_t,34> words{1327939606u,1327497275u,1317015557u,1317015591u,1279275038u,1319112764u,1277607997u,2885681152u,1275068478u,575027900u,2367u,2818572288u,1279275038u,1277608000u,2902458368u,1275068478u,1321291784u,1321291841u,575028025u,2367u,2818572288u,1277608002u,2927624192u,575028028u,2367u,2818572288u,1277607994u,2898264064u,575027897u,2367u,2818572288u,2214592512u,2214592512u,2281701376u};
    std::copy(words.begin(),words.end(),p.code.begin()+p.entry);
    p.swizzle[5]=2147489096u;
    p.swizzle[8]=2147494568u;
    p.swizzle[22]=2147909646u;
    p.swizzle[30]=2147926895u;
    p.swizzle[39]=2793444u;
    p.swizzle[57]=344748u;
    p.swizzle[58]=687u;
    p.swizzle[59]=8177u;
    p.swizzle[60]=352940u;
    p.swizzle[61]=2063u;
    p.swizzle[62]=867u;
    p.swizzle[63]=344940u;
    p.swizzle[64]=2735u;
    p.swizzle[65]=20u;
    p.swizzle[66]=15u;
    return p;
}

void CheckFetchedGeometryVertices(Metal::Device& device,Metal::Scheduler& scheduler) {
    using Op=nihstro::OpCode::Id;
    Metal::ShaderMachineRuntime machine{device,scheduler},oracle{device,scheduler};
    Metal::VertexFetchRuntime fetch{device,scheduler},reference_fetch{device,scheduler};
    Metal::TransferRuntime transfer{device,scheduler};
    unsigned cases{},elements{};
    for(unsigned count:{3u,67u,131u}) for(unsigned index_size:{0u,1u,2u}) for(unsigned mode=0;mode<4;++mode) {
        const unsigned vertices=index_size ? 66:count,base=7,stride=23,guard=19;
        const auto source=device.CreateBuffer(guard+vertices*stride+guard,true);
        std::memset(source.contents,0x6d,source.length);
        auto* data=static_cast<std::byte*>(source.contents)+guard;
        for(unsigned v=0;v<vertices;++v) {
            const std::array<std::int16_t,3> a{std::int16_t(v-31),std::int16_t(v*3),std::int16_t(-int(v))};
            std::memcpy(data+v*stride+1,a.data(),sizeof(a));
            const float value=float(v+1)*.125f;std::memcpy(data+v*stride+9,&value,sizeof(value));
        }
        Metal::PicaVertexLayout layout{};
        layout.fixed_attributes=Metal::DefaultVertexRegisters();layout.mapped_attribute_count=4;
        for(unsigned a=0;a<16;++a)layout.register_map[a]=a;
        layout.attribute_inputs[0]={{0,1,stride,3,Metal::VertexAttributeFormat::Short,0},Metal::PicaAttributeMode::Buffer};
        layout.attribute_inputs[1]={{0,9,stride,1,Metal::VertexAttributeFormat::Float,1},Metal::PicaAttributeMode::Buffer};
        layout.attribute_inputs[2].mode=Metal::PicaAttributeMode::Fixed;
        layout.fixed_attributes[2]={.125f,.25f,.5f,1};
        // A retained attribute aliases the first mapped register; all untouched
        // attribute/register words remain observable in the full history bank.
        layout.register_map[3]=0;
        const auto indexed=device.CreateBuffer(3+count*std::max(index_size,1u),true);
        std::vector<unsigned> ids(count);
        for(unsigned n=0;n<count;++n) {
            // 0..65 fills and evicts the FIFO, then revisits old and recent IDs.
            const unsigned id=index_size ? (n<66 ? n:n==66 ? 0:n==67 ? 1:(n*17)%66):n;
            ids[n]=id;
            if(index_size) {auto* bytes=static_cast<std::uint8_t*>(indexed.contents)+3+n*index_size;
                bytes[0]=std::uint8_t(base+id);if(index_size==2)bytes[1]=std::uint8_t((base+id)>>8);}
        }
        const Metal::BufferSlice indices=index_size ? Metal::BufferSlice{indexed,3,count*index_size,nullptr,{}}:Metal::BufferSlice{};
        Metal::ShaderMachineProgram program{};program.allow_emission=false;program.output_mask=5;
        program.swizzle[0]=Metal::Test::IdentitySwizzle;
        program.code[0]=Metal::Test::Arithmetic(Op::MOV,17,mode==1 ? 16:1);
        program.code[1]=Metal::Test::Arithmetic(Op::MOV,16,1);
        program.code[2]=Metal::Test::Arithmetic(Op::MOV,0,mode==1 ? 17:16);
        program.code[3]=Metal::Test::Arithmetic(Op::MOV,2,0);
        program.code[4]=unsigned(Op::END)<<26;
        Metal::ShaderMachineUniforms uniforms{};
        if(mode==2) {
            program.code[0]=Metal::Test::Arithmetic(Op::ADD,16,16,1);
            program.code[1]=Metal::Test::Arithmetic(Op::MOV,0,1);
            program.code[2]=Metal::Test::Arithmetic(Op::MOV,2,0);
            program.code[3]=unsigned(Op::END)<<26;
        } else if(mode==3) {
            uniforms.f[0].fill(4);
            program.code[0]=(unsigned(Op::CMP)<<26)|(2u<<24)|(2u<<21)|(32u<<12)|(1u<<7);
            program.code[1]=(unsigned(Op::IFC)<<26)|(1u<<25)|(2u<<22)|(3u<<10);
            program.code[2]=Metal::Test::Arithmetic(Op::MOV,0,1);
            program.code[3]=Metal::Test::Arithmetic(Op::MOV,2,0);
        }
        const std::array<Metal::ShaderMachineRuntime::VertexExecution,4> execution{
            Metal::ShaderMachineRuntime::VertexExecution::Independent,Metal::ShaderMachineRuntime::VertexExecution::DependencyPasses,
            Metal::ShaderMachineRuntime::VertexExecution::PrivateRecurrences,Metal::ShaderMachineRuntime::VertexExecution::MergeState};
        Check(Metal::ShaderMachineRuntime::AnalyzeVertices(program,uniforms)==execution[mode],
            "Raw GS fixture did not exercise its intended GPU execution class");
        Metal::ShaderMachineState unit{device},expected_unit{device};
        Metal::VertexInputState input_state{device},expected_input{device};
        Metal::ShaderMachineSnapshot initial{};Metal::VertexInputSnapshot input_initial{};
        initial.needs_reset=input_initial.needs_reset=(count==3);
        for(unsigned w=0;!initial.needs_reset && w<1804/4;++w) {
            const float value=float((w%7)+1)*.125f;std::memcpy(initial.history.data()+w*4,&value,4);
        }
        for(unsigned w=0;!input_initial.needs_reset && w<input_initial.history.size()/4;++w) {
            const float value=float(w+1)*.25f;std::memcpy(input_initial.history.data()+w*4,&value,4);
        }
        machine.LoadState(unit,initial);oracle.LoadState(expected_unit,initial);
        fetch.LoadState(input_state,input_initial);reference_fetch.LoadState(expected_input,input_initial);
        const auto expected=device.CreateBuffer(NSUInteger(count)*sizeof(Metal::VertexRegisters),true);
        struct Entry {unsigned id{};bool valid{};Metal::BufferSlice output;};
        std::array<Entry,64> fifo{};unsigned replacement{};
        for(unsigned n=0;n<count;++n) {
            const auto id=ids[n];Metal::BufferSlice result{};
            if(index_size)for(const auto& e:fifo)if(e.valid && e.id==id){result=e.output;break;}
            if(!result.buffer) {
                const std::array<Metal::BufferSlice,1> sources{{{source,guard+id*stride,stride,nullptr,{}}}};
                const auto input=reference_fetch.FetchPica(expected_input,sources,layout,1,1,0,0);
                result=oracle.RunVertexDeferred(expected_unit,program,uniforms,input);
                if(index_size){fifo[replacement]={id,true,result};replacement=(replacement+1)%64;}
            }
            transfer.CopyBuffer(result.buffer,result.offset,expected,NSUInteger(n)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters));
        }
        scheduler.Finish();
        const auto vs_expected=oracle.SaveState(expected_unit);
        const auto fetch_expected=reference_fetch.SaveState(expected_input);
        const std::array<Metal::BufferSlice,1> sources{{{source,guard,vertices*stride,nullptr,{}}}};
        const auto tick=scheduler.CurrentTick();
        const auto compiled=device.DiskCacheStats().compiled_pipelines;
        const auto actual=Metal::InputReferenceTestAccess::Fetched(machine,fetch,input_state,sources,layout,
            unit,program,uniforms,vertices,count,base,base,indices,index_size);
        Check(scheduler.CurrentTick()==tick,"Raw fetched GS vertices waited/read back before GPU consumers");
        Check(device.DiskCacheStats().compiled_pipelines==compiled,
            "Raw GS vertex batch compiled a pipeline during execution");
        const auto readback=device.CreateBuffer(expected.length,true);
        transfer.CopyBuffer(actual.buffer,actual.offset,readback,0,expected.length);scheduler.Finish();
        Check(std::memcmp(readback.contents,expected.contents,expected.length)==0,
            "Raw fetched GS vertices differ from scalar FIFO outputs");
        const auto fetched=fetch.SaveState(input_state);
        Check(machine.SaveState(unit)==vs_expected && fetched.history==fetch_expected.history && fetched.needs_reset==fetch_expected.needs_reset,
            "Raw fetched GS batch changed full1824-byte VS or512-byte input history");
        auto illegal=program;illegal.code[0]=0x11u<<26;
        const auto encoded=scheduler.EncoderCount();bool rejected=false;
        try{(void)Metal::InputReferenceTestAccess::Fetched(machine,fetch,input_state,sources,layout,
            unit,illegal,uniforms,vertices,count,base,base,indices,index_size);}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && scheduler.EncoderCount()==encoded && machine.SaveState(unit)==vs_expected,
            "Unproved raw GS vertex program changed queued work or retained VS state");
        ++cases;elements+=count;
    }
    std::cout<<cases<<" raw fetched GS vertex batches: "<<elements<<" exact per-element outputs, unaligned short/float fetches, retained/map aliases, reset/inherited full1824/512-byte banks,64-FIFO hits/evictions and pre-encoding proof rejection passed\n";
}

void CheckCompactGeometry(Metal::Device& device,Metal::Scheduler& scheduler) {
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    auto real=RealLinearGeometryProgram();
    Check(Metal::ShaderMachineRuntime::LinearGeometryStorageBound(real)==12,"Real linear GS storage bound changed");
    for(auto op:{Op::CALL,Op::IFU,Op::LOOP,Op::JMPU,Op::BREAK}) {
        auto invalid=real;invalid.code[invalid.entry]=word(op);
        Check(!Metal::ShaderMachineRuntime::LinearGeometryStorageBound(invalid),"Unbounded GS flow accepted compact storage");
    }
    auto no_end=real;no_end.code.fill(word(Op::NOP));
    Check(!Metal::ShaderMachineRuntime::LinearGeometryStorageBound(no_end),"Missing END accepted a compact GS proof");
    Metal::ShaderMachineRuntime probe{device,scheduler};
    Metal::ShaderMachineProgram vs{};vs.allow_emission=false;vs.output_mask=1;vs.swizzle[0]=Metal::Test::IdentitySwizzle;
    vs.code[0]=Metal::Test::Arithmetic(Op::MOV,0,0);vs.code[1]=word(Op::END);
    Check(Metal::InputReferenceTestAccess::VertexCompletes(probe,vs),"Safe scalar VS completion was not proved");
    vs.code[0]=0x11u<<26;
    Check(!Metal::InputReferenceTestAccess::VertexCompletes(probe,vs),"Illegal scalar VS passed deferred-render completion proof");
    constexpr unsigned width=16,height=16;
    Metal::GeometryRuntime geometry{device,scheduler};Metal::TransferRuntime transfer{device,scheduler};
    Metal::DrawRuntime draws{device,scheduler,geometry};
    Metal::FragmentOperations blend{};blend.rgb={true,0,6,7};blend.alpha={true,0,1,7};
    auto fs=Metal::TranslateRasterizerFragmentGLSL(R"(#version 450
layout(location=1) in vec4 primary_color;
layout(location=0) out vec4 color;
void main(){color=primary_color;}
)",blend);
    auto library=device.CompileLibrary(fs.source,"Batched compact GS exact ordered blend");
    const Metal::CompiledFragmentShader fragment{{std::move(fs),blend},library};
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
    const auto texture=device.CreateTexture(td);
    auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=texture;
    pass.colorAttachments[0].loadAction=MTLLoadActionLoad;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    Metal::PicaDrawState draw_state{};draw_state.use_color=true;
    const auto render=[&](const Metal::AssembledVertices& vertices) {
        draws.Draw(draw_state,fragment,pass,{0,0,width,height,0,1},{0,0,width,height},vertices,{},{},{});
    };
    unsigned cases{},invocations{},pixels{};
    for(unsigned kind=0;kind<3;++kind)
    for(auto mode:{Metal::GeometryInputMode::Point,Metal::GeometryInputMode::Fixed,Metal::GeometryInputMode::Variable}) {
        auto program=real;
        if(kind!=0) {
            program={};program.allow_emission=true;program.output_mask=7;program.swizzle[0]=Metal::Test::IdentitySwizzle;
            unsigned pc=0;
            if(kind==1) for(unsigned v=0;v<4;++v) {
                program.code[pc++]=Metal::Test::Arithmetic(Op::MOV,0,32+v);
                program.code[pc++]=Metal::Test::Arithmetic(Op::MOV,1,40);
                program.code[pc++]=Metal::Test::Arithmetic(Op::MOV,2,41);
                program.code[pc++]=word(Op::SETEMIT)|(((v<3 ? v:0)*4+(v>=2 ? 2:0)+(v==3 ? 1:0))<<22);
                program.code[pc++]=word(Op::EMIT);
            }
            program.code[pc]=word(Op::END);
        }
        Metal::GeometryProgramRuntime groups{device,scheduler},oracle_groups{device,scheduler};
        Metal::ShaderMachineRuntime machine{device,scheduler},oracle{device,scheduler};
        Metal::GeometryInputState state{device},expected_state{device};
        Metal::ShaderMachineState unit{device},expected_unit{device};
        Metal::PrimitiveStream stream{device},expected_stream{device};
        Metal::GeometryInputConfig config{};config.mode=mode;config.vs_output_mask=7;config.vs_output_count=3;
        config.input_count=3;config.input_registers={0,1,2};config.fixed_vertex_count=1;config.uniform_start=40;config.variable_main_count=1;
        groups.Configure(state,config);oracle_groups.Configure(expected_state,config);
        Metal::GeometryEmissionConfig capture{};capture.attribute_count=3;capture.semantics={0x03020100,0x0b0a0908,0x07060504};
        capture.honor_winding=1;capture.fix_quaternions=1;
        const auto source=device.CreateBuffer(8*sizeof(Metal::VertexRegisters),true);
        for(unsigned n=0;n<8;++n) {
            auto* raw=static_cast<Metal::VertexRegisters*>(source.contents)+n;
            for(unsigned reg=0;reg<16;++reg)(*raw)[reg]={float(n+reg)*.0625f,-0.f,.5f,1};
        }
        const auto prepare=[&](auto& runtime,auto& target,unsigned n) {
            Metal::ShaderMachineUniforms values{};
            for(unsigned reg=0;reg<96;++reg)values.f[reg]={float(reg+n)*.03125f,.125f,-0.f,1};
            values.f[0]={-1,-1,-.5f,1};values.f[1]={1,-1,-.5f,1};values.f[2]={-1,1,-.5f,1};values.f[3]={1,1,-.5f,1};
            values.f[8]={float(n+1)/9,.25f,float(8-n)/9,.125f+float(n%3)*.125f};values.f[9]={0,0,0,1};
            values.b=n&1;values.i[2]={n,3,5,7};
            runtime.UpdateUniforms(target,values);
            if(mode==Metal::GeometryInputMode::Variable)runtime.BeginVariable(target,1);
            Check(runtime.SubmitVertex(target,{source,n*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}}),
                  "Compact GS fixture did not produce a ready group");
        };
        struct Expected {Metal::VertexRegisters raw{};std::vector<std::byte> emissions;unsigned count{};Metal::AssembledVertices captured;};
        std::vector<Expected> expected;
        transfer.ClearColor(texture,MTLClearColorMake(.0625,.125,.25,1));
        for(unsigned n=0;n<8;++n) {
            prepare(oracle_groups,expected_state,n);
            const auto result=oracle_groups.Invoke(expected_state,oracle,expected_unit,program,24);
            Expected e{};std::memcpy(e.raw.data(),result.output.data,sizeof(e.raw));e.count=result.emission_count;
            const auto* bytes=static_cast<const std::byte*>(result.emissions.data);
            e.emissions.assign(bytes,bytes+e.count*sizeof(Metal::ShaderMachineEmission));
            e.captured=oracle.ConvertEmissionList(result,capture,expected_stream);
            render(e.captured);expected.push_back(std::move(e));
        }
        const auto expected_pixels=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
        if(kind==1 && mode==Metal::GeometryInputMode::Point)
            Check(expected_pixels[0]!=std::byte{16},"Ordered blend fixture rendered only its clear color");
        const auto expected_group=oracle_groups.SaveState(expected_state);
        const auto expected_history=geometry.SaveStream(expected_stream);
        const auto expected_shader=oracle.SaveState(expected_unit);
        transfer.ClearColor(texture,MTLClearColorMake(.0625,.125,.25,1));
        const auto abort=machine.BeginGeometryBatch();
        std::vector<Metal::GeometryQueuedInvocation> pending;std::vector<Metal::AssembledVertices> actual;
        const auto tick=scheduler.CurrentTick();
        {
            Metal::Scheduler::ComputeBatchScope batch{scheduler};
            for(unsigned n=0;n<8;++n) {
                prepare(groups,state,n);
                auto item=Metal::InputReferenceTestAccess::Compact(groups,state,machine,unit,program,abort);
                const auto bound=*Metal::ShaderMachineRuntime::LinearGeometryStorageBound(program);
                Check(item.execution.capacity==1000000 && item.execution.storage_proof &&
                    item.execution.result.emissions.size==bound*sizeof(Metal::ShaderMachineEmission),
                    "Compact GS changed logical limit or lacked exact physical proof");
                if(n==0) for(unsigned bad=0;bad<3;++bad) {
                    auto invalid=item.execution;
                    if(bad==0)--invalid.capacity;else if(bad==1)invalid.result.emissions.offset+=16;else invalid.status.offset+=4;
                    const auto encoded=scheduler.EncoderCount();bool rejected=false;
                    try{machine.ConvertEmissionListIndirect(invalid,capture,stream);}catch(const std::invalid_argument&){rejected=true;}
                    Check(rejected && scheduler.EncoderCount()==encoded,"Mismatched compact GS proof encoded GPU consumers");
                }
                actual.push_back(machine.ConvertEmissionListIndirect(item.execution,capture,stream));pending.push_back(std::move(item));
            }
            const auto encoded=scheduler.EncoderCount();
            for(const auto& vertices:actual)render(vertices);
            // A zero-emission list still uses an indirect count and one render pass.
            Check(scheduler.EncoderCount()==encoded+1,"Batched GS outputs did not share one native render encoder");
        }
        Check(scheduler.CurrentTick()==tick,"Compact GS batch inserted CPU completion");
        Check(expected_pixels==transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4),
              "Batched compact GS changed complete ordered blend framebuffer");pixels+=width*height;
        for(unsigned n=0;n<8;++n) {
            machine.ValidateIndirectResult(actual[n]);const auto result=machine.ResolvePending(pending[n].execution);const auto& e=expected[n];
            Check(result.emission_count==e.count && !std::memcmp(result.output.data,e.raw.data(),sizeof(e.raw)) &&
                (e.emissions.empty() || !std::memcmp(result.emissions.data,e.emissions.data(),e.emissions.size())),
                "Compact GS changed full raw output/emission bytes");
            if(e.count) {
                const auto bytes=NSUInteger(e.count)*sizeof(Metal::CapturedVertex);const auto check=device.CreateBuffer(bytes*2,true);
                transfer.CopyBuffer(actual[n].storage.buffer,actual[n].storage.offset,check,0,bytes);
                transfer.CopyBuffer(e.captured.storage.buffer,e.captured.storage.offset,check,bytes,bytes);scheduler.Finish();
                Check(!std::memcmp(check.contents,static_cast<std::byte*>(check.contents)+bytes,bytes),"Batched GS overwrote earlier captured vertices");
            }
            ++invocations;
        }
        Check(groups.SaveState(state)==expected_group && machine.SaveState(unit)==expected_shader,
              "Compact GS changed full input/uniform/group or shader state");
        const auto history=geometry.SaveStream(stream);
        Check(history.history==expected_history.history && history.pending==expected_history.pending && history.needs_reset==expected_history.needs_reset,
              "Compact GS batch changed complete primitive history");++cases;
    }
    for(unsigned failure=0;failure<3;++failure) {
        Metal::GeometryProgramRuntime groups{device,scheduler},oracle_groups{device,scheduler};
        Metal::ShaderMachineRuntime machine{device,scheduler},oracle{device,scheduler};
        Metal::GeometryInputState state{device},expected_state{device};
        Metal::ShaderMachineState unit{device},expected_unit{device};
        Metal::PrimitiveStream stream{device};
        Metal::GeometryInputConfig config{};config.vs_output_mask=7;config.vs_output_count=3;
        config.input_count=3;config.input_registers={0,1,2};
        groups.Configure(state,config);oracle_groups.Configure(expected_state,config);
        auto program=real,bad=real;
        if(failure==1)bad.code[155]=(bad.code[155]&~(15u<<22))|(12u<<22);
        if(failure==2)bad.code[159]=0x11u<<26;
        const auto source=device.CreateBuffer(sizeof(Metal::VertexRegisters),true);std::memset(source.contents,0,source.length);
        const auto prepare=[&](auto& runtime,auto& target,unsigned n) {
            Metal::ShaderMachineUniforms values{};
            for(unsigned i=0;i<96;++i)values.f[i]={float(n+i)*.0625f,.25f,-0.f,1};
            values.b=n&1;runtime.UpdateUniforms(target,values);
            Check(runtime.SubmitVertex(target,{source,0,source.length,nullptr,{}}),"Compact failure fixture lacks a ready GS group");
        };
        std::vector<Metal::VertexRegisters> outputs;std::vector<std::vector<std::byte>> emitted;
        std::string expected_error;
        for(unsigned n=0;n<4;++n) {
            prepare(oracle_groups,expected_state,n);
            try {
                const auto result=oracle_groups.Invoke(expected_state,oracle,expected_unit,n==2 && failure ? bad:program,24);
                Metal::VertexRegisters value{};std::memcpy(value.data(),result.output.data,sizeof(value));outputs.push_back(value);
                const auto* bytes=static_cast<const std::byte*>(result.emissions.data);
                emitted.emplace_back(bytes,bytes+result.emission_count*sizeof(Metal::ShaderMachineEmission));
            } catch(const Metal::ShaderMachineError& error) {expected_error=error.what();break;}
        }
        const auto expected_group=oracle_groups.SaveState(expected_state);const auto expected_shader=oracle.SaveState(expected_unit);
        std::vector<Metal::GeometryQueuedInvocation> pending;std::vector<Metal::AssembledVertices> captured;
        const auto abort=machine.BeginGeometryBatch();
        {
            Metal::Scheduler::ComputeBatchScope batch{scheduler};
            for(unsigned n=0;n<4;++n) {
                prepare(groups,state,n);
                auto item=Metal::InputReferenceTestAccess::Compact(groups,state,machine,unit,n==2 && failure ? bad:program,abort,failure ? 1000000:4);
                if(failure)captured.push_back(machine.ConvertEmissionListIndirect(item.execution,{},stream));
                pending.push_back(std::move(item));
            }
        }
        scheduler.Finish();
        if(failure)for(unsigned n=0;n<4;++n) {
            const auto args=Metal::InputReferenceTestAccess::Indirect(captured[n]);
            const auto* words=reinterpret_cast<const unsigned*>(static_cast<const std::byte*>(args.buffer.contents)+args.offset);
            Check(words[0]==(n<2 ? 6u:0u),"Failed/skipped compact GS published an indirect draw or lost a valid prefix");
        }
        unsigned consumed{};std::string error;
        try {
            groups.CompleteQueued(state,machine,unit,pending,[&](const auto& result) {
                Check(consumed<outputs.size() && result.emission_count*sizeof(Metal::ShaderMachineEmission)==emitted[consumed].size() &&
                    !std::memcmp(result.output.data,outputs[consumed].data(),sizeof(Metal::VertexRegisters)) &&
                    !std::memcmp(result.emissions.data,emitted[consumed].data(),emitted[consumed].size()),
                    "Compact overflow/fatal handling changed successful raw/emission prefix");++consumed;
            });
        } catch(const Metal::ShaderMachineError& e) {error=e.what();}
        Check(error==expected_error && consumed==outputs.size(),"Compact GS changed first fatal code/PC or overflow retries");
        Check(groups.SaveState(state)==expected_group && machine.SaveState(unit)==expected_shader,
            "Compact GS lost complete ready-input/b15/reset/unit state after retry/fatal rollback");
    }
    std::cout<<"Compact GS failures: logical4 overflow retries and late invalid-vertex/opcode errors preserve full state, exact error PCs and indirect successful prefixes\n";
    std::cout<<"Compact GS: "<<cases<<" point/fixed/variable real/synthetic/zero programs, "<<invocations
             <<" retained invocations, "<<pixels<<" ordered blend pixels, full1824/2128/272-byte states, logical1M limits and rejected forged storage bindings exact\n";
}

void CheckQueuedGeometry(Metal::Device& device,Metal::Scheduler& scheduler,unsigned batch_mode=0) {
    const bool batched=batch_mode!=0;
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op) {return unsigned(op)<<26;};
    const auto mov=[](unsigned d,unsigned s){return Metal::Test::Arithmetic(Op::MOV,d,s);};
    constexpr unsigned invocations=65;
    unsigned cases{},checked_emissions{},encoder_checks{};
    for(auto mode:{Metal::GeometryInputMode::Point,Metal::GeometryInputMode::Fixed,Metal::GeometryInputMode::Variable})
    for(unsigned emitted:{0u,3u,195u}) {
        Metal::GeometryProgramRuntime groups{device,scheduler},reference_groups{device,scheduler};
        Metal::ShaderMachineRuntime machine{device,scheduler},reference_machine{device,scheduler};
        Metal::GeometryInputState state{device},reference_state{device};
        Metal::ShaderMachineState unit{device},reference_unit{device};
        Metal::GeometryInputConfig config{};config.mode=mode;config.vs_output_mask=13;config.vs_output_count=2;
        config.input_count=4;config.input_registers={5,7,5,9};config.uniform_start=20;config.fixed_vertex_count=3;config.variable_main_count=2;
        groups.Configure(state,config);reference_groups.Configure(reference_state,config);
        Metal::ShaderMachineUniforms initial{};initial.i[0]={emitted ? emitted/3-1:0,0,1,0};
        groups.UpdateUniforms(state,initial);reference_groups.UpdateUniforms(reference_state,initial);
        const unsigned vertices=mode==Metal::GeometryInputMode::Point ? 2:3;
        auto source=device.CreateBuffer(invocations*3*sizeof(Metal::VertexRegisters),true);
        for(unsigned v=0;v<invocations*3;++v) {
            Metal::VertexRegisters raw{};
            for(unsigned reg=0;reg<16;++reg) raw[reg]={float(v+1)*.03125f,float(reg)*.125f,-0.f,1.f};
            std::memcpy(static_cast<std::byte*>(source.contents)+v*sizeof(raw),raw.data(),sizeof(raw));
        }
        const auto input=[&](unsigned group,unsigned vertex) {
            return Metal::BufferSlice{source,(group*3+vertex)*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}};
        };
        Metal::ShaderMachineProgram base{};base.allow_emission=true;base.output_mask=3;base.swizzle[0]=Metal::Test::IdentitySwizzle;
        const unsigned source_register=mode==Metal::GeometryInputMode::Point ? 5:mode==Metal::GeometryInputMode::Fixed ? 52:33;
        base.code[0]=word(Op::IFU)|(15u<<22)|(2u<<10)|1u;
        base.code[1]=Metal::Test::Arithmetic(Op::ADD,16,16,source_register);
        base.code[2]=mov(16,source_register);base.code[3]=mov(1,122);
        base.code[4]=emitted ? word(Op::LOOP)|(13u<<10):word(Op::END);
        for(unsigned v=0;v<3;++v) {
            base.code[5+v*3]=mov(0,16);
            base.code[6+v*3]=word(Op::SETEMIT)|((v*4+(v==2 ? 2:0))<<22);
            base.code[7+v*3]=word(Op::EMIT);
        }
        base.code[14]=word(Op::END);
        struct Expected {Metal::VertexRegisters output{};std::vector<std::byte> emissions;unsigned count{};};
        std::vector<Expected> expected;
        std::map<unsigned,std::pair<Metal::GeometryInputSnapshot,Metal::ShaderMachineSnapshot>> checkpoints;
        const auto update=[&](auto& runtime,auto& target,unsigned index) {
            Metal::ShaderMachineUniforms patch{};patch.f[90]={float(index),-.5f,1.f,.25f};patch.b=(index&1)<<5;patch.i[2]={index,3,5,7};
            runtime.UpdateUniforms(target,patch,{0,0,1u<<26},1u<<5,1u<<2);
            if(mode==Metal::GeometryInputMode::Variable) runtime.BeginVariable(target,3);
            for(unsigned v=0;v<vertices;++v)
                Check(runtime.SubmitVertex(target,input(index,v))==(v+1==vertices),"Queued GS group boundary differs from configured inputs");
        };
        const auto program=[&](unsigned index) {auto value=base;value.code[4095]=index%3;value.code[6]|=(index&1)<<22;return value;};
        for(unsigned index=0;index<invocations;++index) {
            update(reference_groups,reference_state,index);
            const auto result=reference_groups.Invoke(reference_state,reference_machine,reference_unit,program(index),384);
            Expected value{};value.count=result.emission_count;
            std::memcpy(value.output.data(),result.output.data,sizeof(value.output));
            const auto* bytes=static_cast<const std::byte*>(result.emissions.data);
            value.emissions.assign(bytes,bytes+value.count*sizeof(Metal::ShaderMachineEmission));expected.push_back(std::move(value));
            if((index+1)%32==0) checkpoints.emplace(index+1,std::pair{reference_groups.SaveState(reference_state),reference_machine.SaveState(reference_unit)});
        }
        if(mode==Metal::GeometryInputMode::Variable) reference_groups.BeginVariable(reference_state,3);
        Check(!reference_groups.SubmitVertex(reference_state,input(0,0)),"Incomplete oracle GS group unexpectedly became ready");
        const auto expected_group=reference_groups.SaveState(reference_state);
        const auto expected_unit=reference_machine.SaveState(reference_unit);
        std::vector<Metal::GeometryQueuedInvocation> pending;pending.reserve(32);
        Metal::BufferSlice abort{};unsigned consumed{};auto tick=scheduler.CurrentTick();
        const auto complete=[&] {
            Check(scheduler.CurrentTick()==tick,"Queued GS submitted before its bounded chunk completed");
            groups.CompleteQueued(state,machine,unit,pending,[&](const auto& result) {
                const auto& value=expected.at(consumed++);
                Check(result.emission_count==value.count && !std::memcmp(result.output.data,value.output.data(),sizeof(value.output)),
                    "Queued GS changed full raw output or its emission count");
                Check(value.emissions.empty() || !std::memcmp(result.emissions.data,value.emissions.data(),value.emissions.size()),
                    "Queued GS changed complete emission attributes/winding/padding");
                checked_emissions+=result.emission_count;
            });
            if(!emitted || emitted==3) Check(scheduler.CurrentTick()==tick+1,"Successful GS chunk used more than one completion");
            pending.clear();abort={};
            if(const auto found=checkpoints.find(consumed);found!=checkpoints.end()) {
                Check(groups.SaveState(state)==found->second.first,"Queued GS changed2128-byte input/uniform/group history");
                Check(machine.SaveState(unit)==found->second.second,"Queued GS changed1824-byte retained unit state");
            }
            tick=scheduler.CurrentTick();
        };
        std::optional<Metal::Scheduler::ComputeBatchScope> draw_batch;
        if(batch_mode==2)draw_batch.emplace(scheduler);
        for(unsigned index=0;index<invocations;++index) {
            if(pending.empty()) {abort=machine.BeginGeometryBatch();tick=scheduler.CurrentTick();}
            std::optional<Metal::Scheduler::ComputeBatchScope> batch;
            const auto before_batch=scheduler.EncoderCount();
            if(batch_mode==1)batch.emplace(scheduler);
            update(groups,state,index);auto owned=program(index);
            const auto encoders=scheduler.EncoderCount();
            pending.push_back(groups.InvokeQueued(state,machine,unit,owned,abort));
            const auto encoded=scheduler.EncoderCount()-encoders;
            Check(encoded==(batched ? 0:1),"Queued GS snapshot/execution/b15 split across multiple encoders");
            batch.reset();
            if(batched)Check(scheduler.EncoderCount()==before_batch+
                (batch_mode==1 || index%32==0 ? 1:0),
                "GS input updates/snapshot/execution/b15 did not share one compute encoder");
            ++encoder_checks;
            // Destroy borrowed banks before any completion or overflow retry.
            owned.code.fill(UINT32_MAX);owned.swizzle.fill(UINT32_MAX);
            if(pending.size()==32) complete();
        }
        if(mode==Metal::GeometryInputMode::Variable) groups.BeginVariable(state,3);
        Check(!groups.SubmitVertex(state,input(0,0)),"Incomplete queued GS group unexpectedly became ready");
        complete();
        Check(consumed==invocations && groups.SaveState(state)==expected_group && machine.SaveState(unit)==expected_unit,
            "Queued GS failed full state/partial-group continuation after65 ordered invocations");
        ++cases;
    }
    for(unsigned failed_index:{0u,1u}) {
        Metal::GeometryProgramRuntime groups{device,scheduler},oracle{device,scheduler};
        Metal::ShaderMachineRuntime machine{device,scheduler},reference{device,scheduler};
        Metal::GeometryInputState state{device},expected_state{device};
        Metal::ShaderMachineState unit{device},expected_unit{device};
        Metal::GeometryInputConfig config{};config.vs_output_mask=1;config.vs_output_count=1;config.input_count=1;
        groups.Configure(state,config);oracle.Configure(expected_state,config);
        groups.UpdateUniforms(state,{});oracle.UpdateUniforms(expected_state,{});
        auto source=device.CreateBuffer(3*sizeof(Metal::VertexRegisters),true);
        std::memset(source.contents,0,source.length);
        for(unsigned v=0;v<3;++v)static_cast<Metal::VertexRegisters*>(source.contents)[v][0]={float(v+1),0.f,0.f,1.f};
        const auto input=[&](unsigned index){return Metal::BufferSlice{source,index*sizeof(Metal::VertexRegisters),sizeof(Metal::VertexRegisters),nullptr,{}};};
        Metal::ShaderMachineProgram good{};good.allow_emission=true;good.output_mask=1;good.swizzle[0]=Metal::Test::IdentitySwizzle;
        good.code[0]=mov(0,0);good.code[1]=word(Op::END);
        auto bad=good;bad.code[0]=0x10u<<26;
        for(unsigned index=0;index<=failed_index;++index) {
            Check(oracle.SubmitVertex(expected_state,input(index)),"Failure oracle group did not become ready");
            if(index<failed_index)oracle.Invoke(expected_state,reference,expected_unit,good);
        }
        const auto expected_group=oracle.SaveState(expected_state);
        const auto before_unit=reference.SaveState(expected_unit);
        std::string expected_error;
        try{oracle.Invoke(expected_state,reference,expected_unit,bad);}catch(const Metal::ShaderMachineError& e){expected_error=e.what();}
        Check(!expected_error.empty(),"Failure oracle did not reject illegal GS instruction");
        const auto abort=machine.BeginGeometryBatch();std::vector<Metal::GeometryQueuedInvocation> pending;
        for(unsigned index=0;index<3;++index) {
            Check(groups.SubmitVertex(state,input(index)),"Queued failure group did not become ready");
            pending.push_back(groups.InvokeQueued(state,machine,unit,index==failed_index ? bad:good,abort));
        }
        bool premature=false;
        try{(void)machine.ResolvePending(pending.front().execution);}catch(const std::invalid_argument&){premature=true;}
        Check(premature,"Uncompleted queued GS result was exposed to the CPU");
        scheduler.Finish();
        Check(machine.PendingStatus(pending[failed_index].execution)[0]==2 &&
            machine.PendingStatus(pending.back().execution)[0]==8,"GPU failure guard did not suppress later GS invocations");
        unsigned consumed{};std::string actual_error;
        try{groups.CompleteQueued(state,machine,unit,pending,[&](const auto&){++consumed;});}
        catch(const Metal::ShaderMachineError& e){actual_error=e.what();}
        Check(actual_error==expected_error && consumed==failed_index,"Queued fatal GS changed error code/PC or published a failed successor");
        Check(groups.SaveState(state)==expected_group && machine.SaveState(unit)==before_unit,
            "Queued fatal GS lost its ready input group, b15, reset flag or successful-prefix unit state");
        const auto expected=oracle.Invoke(expected_state,reference,expected_unit,good);
        const auto actual=groups.Invoke(state,machine,unit,good);
        Check(!std::memcmp(actual.output.data,expected.output.data,sizeof(Metal::VertexRegisters)) &&
            groups.SaveState(state)==oracle.SaveState(expected_state) && machine.SaveState(unit)==reference.SaveState(expected_unit),
            "Failed queued GS could not resume through the existing synchronous invocation contract");
    }
    std::cout<<cases<<" queued native GS streams/585 invocations: Point/Fixed/Variable,32/32/1 chunks, zero/3/195 emissions, exact full outputs/emissions and1824/2128-byte states, b15/masked uniforms, source mutation, partial-group continuation and bounded overflow retries; "<<checked_emissions<<" emitted vertices checked; two fatal prefix/first-invocation guards preserve exact code/PC, ready group/reset/state and corrected retry\n";
    Check(encoder_checks==585,"Missing single-encoder GS invocation coverage");
    std::cout<<"Single-encoder GS snapshot/execution/b15: "<<encoder_checks<<" exact native invocation checks passed; full1824/2128-byte states, immutable retries and fatal rollback preserved\n";
}

void CheckEmissionLists(Metal::Device& device,Metal::Scheduler& scheduler) {
    Metal::ShaderMachineRuntime machine{device,scheduler};
    Metal::GeometryRuntime geometry{device,scheduler};
    Metal::TransferRuntime transfer{device,scheduler};
    Metal::DrawRuntime draws{device,scheduler,geometry};
    constexpr unsigned maximum=210,width=32,height=32;
    std::array<Metal::ShaderMachineEmission,maximum> records{};
    for(unsigned v=0;v<maximum;++v) {
        const auto triangle=v/3;
        records[v].attributes[0]={std::array{-1.f,3.f,-1.f}[v%3],std::array{-1.f,-1.f,3.f}[v%3],-.5f,1};
        records[v].attributes[1]={v%3==1 ? -1.f:1.f,.25f,-0.f,1};
        records[v].attributes[2]={float(triangle%7+1)/8,float(triangle%5+1)/8,.375f,.25f};
        records[v].winding=v%3==0 && (triangle&1);
    }
    const auto upload=device.CreateBuffer(sizeof(records)+32,true),source=device.CreateBuffer(upload.length,false);
    std::memset(upload.contents,0xa7,upload.length);
    std::memcpy(static_cast<std::byte*>(upload.contents)+16,records.data(),sizeof(records));
    transfer.CopyBuffer(upload,0,source,0,source.length);
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    const auto texture=device.CreateTexture(td);
    constexpr std::string_view fs=R"(#version 450
layout(location=1) in vec4 primary_color;
layout(location=6) in vec4 normquat;
layout(location=0) out vec4 color;
void main(){color=primary_color*0.75+abs(normquat)*0.125;}
)";
    std::array<Metal::CompiledFragmentShader,3> fragments;
    for(unsigned i=0;i<fragments.size();++i) {
        Metal::FragmentOperations operations{};
        if(!i) {operations.rgb={true,0,6,7};operations.alpha={true,0,1,7};}
        else {operations.logic_enabled=true;operations.logic_operation=i==1 ? 11:7;}
        auto translated=Metal::TranslateRasterizerFragmentGLSL(fs,operations);
        Check(translated.uses_framebuffer_fetch,"GS list fixture lost ordered framebuffer dependency");
        auto library=device.CompileLibrary(translated.source,"GS complete list ordered pixels");
        fragments[i]={{std::move(translated),operations},library};
    }
    Metal::PicaDrawState draw_state{};draw_state.use_color=true;
    const auto same_snapshot=[](const auto& a,const auto& b) {
        return a.history==b.history && a.pending==b.pending && a.topology==b.topology && a.needs_reset==b.needs_reset;
    };
    unsigned cases{},pixel_cases{},vertices{};
    for(unsigned count:{0u,3u,6u,9u,99u,210u}) for(unsigned flags=0;flags<16;++flags) for(bool seed_strip:{false,true}) {
        Metal::PrimitiveStream original{device},batched{device};
        Metal::GeometryEmissionConfig config{};config.attribute_count=3;
        config.semantics={0x03020100,0x07060504,0x0b0a0908};
        config.flip_viewport=flags&1;config.enable_clip=(flags>>1)&1;
        config.fix_quaternions=(flags>>2)&1;config.honor_winding=(flags>>3)&1;
        config.clip_coefficient={.5f,-.25f,.125f,.75f};
        if(seed_strip) {
            auto seed=machine.ConvertEmissions({{},{source,16,3*sizeof(records[0]),nullptr,{}},3},config);
            for(auto* stream:{&original,&batched})
                geometry.Append(*stream,seed.storage,3,Metal::TriangleTopology::Strip,2,0,0,{},0,false);
        }
        const auto tick=scheduler.CurrentTick();
        Metal::ShaderMachineResult emitted{{},{source,16,sizeof(records),nullptr,{}},count};
        std::vector<Metal::AssembledVertices> expected;
        const auto packed=device.CreateBuffer(std::max(16u,count*unsigned(sizeof(Metal::CapturedVertex)))+32,true);
        std::memset(packed.contents,0x6d,packed.length);
        auto legacy_config=config;legacy_config.fix_quaternions=legacy_config.honor_winding=0;
        for(unsigned first=0;first<count;first+=3) {
            auto primitive=emitted;primitive.emission_count=3;primitive.emissions.offset+=first*sizeof(records[0]);primitive.emissions.size=3*sizeof(records[0]);
            const auto capture=machine.ConvertEmissions(primitive,legacy_config);
            Metal::BufferSlice winding{};
            if(config.honor_winding) winding={source,primitive.emissions.offset+256,4,nullptr,{}};
            auto assembled=geometry.Append(original,capture.storage,3,Metal::TriangleTopology::List,3,0,0,{},0,config.fix_quaternions,winding);
            transfer.CopyBuffer(assembled.storage.buffer,assembled.storage.offset,packed,16+first*sizeof(Metal::CapturedVertex),3*sizeof(Metal::CapturedVertex));
            expected.push_back(std::move(assembled));
        }
        const auto actual=machine.ConvertEmissionList(emitted,config,batched);
        Check(scheduler.CurrentTick()==tick,"Complete GS list forced an early submission/completion");
        Check(actual.vertex_count==count,"Complete GS list changed vertex count");
        if(count) {
            const auto readback=device.CreateBuffer(packed.length,true);
            std::memset(readback.contents,0x6d,readback.length);
            transfer.CopyBuffer(actual.storage.buffer,actual.storage.offset,readback,16,count*sizeof(Metal::CapturedVertex));
            scheduler.Finish();
            Check(!std::memcmp(readback.contents,packed.contents,packed.length),"Complete GS list changed captured bytes, winding/quaternion order or guards");
        }
        Check(same_snapshot(geometry.SaveStream(original),geometry.SaveStream(batched)),"Complete GS list changed complete272-byte primitive snapshot or metadata");
        if(count || seed_strip) {
            const auto a=Metal::InputReferenceTestAccess::History(original),b=Metal::InputReferenceTestAccess::History(batched);
            auto readback=device.CreateBuffer(544,true);
            transfer.CopyBuffer(a.buffer,a.offset,readback,0,272);transfer.CopyBuffer(b.buffer,b.offset,readback,272,272);scheduler.Finish();
            Check(!std::memcmp(readback.contents,static_cast<std::byte*>(readback.contents)+272,272),"Complete GS list marked zero history without writing the actual GPU bytes");
        }
        // Blend and two logic operations depend on the exact order of all
        // overlapping primitives, including both winding orientations.
        if(count && flags>=12 && !seed_strip) for(const auto& fragment:fragments) {
            const auto render=[&](bool batch) {
                transfer.ClearColor(texture,MTLClearColorMake(.0625,.125,.25,1));
                auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=texture;
                pass.colorAttachments[0].loadAction=MTLLoadActionLoad;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                const auto before=scheduler.ArgumentUpdateCount();
                if(batch) draws.Draw(draw_state,fragment,pass,{0,0,width,height,0,1},{0,0,width,height},actual,{},{},{});
                else for(const auto& triangle:expected) draws.Draw(draw_state,fragment,pass,{0,0,width,height,0,1},{0,0,width,height},triangle,{},{},{});
                Check(scheduler.ArgumentUpdateCount()-before==(batch ? 1:count/3),"GS list failed to replace per-primitive draws with one draw");
                return transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
            };
            Check(render(false)==render(true),"GS complete-list draw changed ordered framebuffer pixels");++pixel_cases;
        }
        vertices+=count;++cases;
    }
    Metal::PrimitiveStream partial{device};Metal::GeometryEmissionConfig config{};
    const auto capture=machine.ConvertEmissions({{},{source,16,sizeof(records),nullptr,{}},3},config);
    geometry.Append(partial,capture.storage,3,Metal::TriangleTopology::List,1,0,0,{},0,false);
    const auto snapshot=geometry.SaveStream(partial);bool rejected=false;
    try{machine.ConvertEmissionList({{},{source,16,sizeof(records),nullptr,{}},3},config,partial);}catch(const std::invalid_argument&){rejected=true;}
    Check(rejected && same_snapshot(geometry.SaveStream(partial),snapshot),"GS complete list accepted or mutated a partial list");
    Metal::PrimitiveStream foreign{device};bool foreign_rejected=false;
    std::thread worker([&]{try{machine.ConvertEmissionList({{},{source,16,sizeof(records),nullptr,{}},3},config,foreign);}catch(const std::logic_error&){foreign_rejected=true;}});worker.join();
    Check(foreign_rejected,"GS complete list accepted foreign-thread history mutation");
    std::cout<<"Native GS complete lists: "<<cases<<" full captured-byte/272-byte GPU-history cases, "<<vertices
             <<" vertices; "<<pixel_cases<<" complete ordered blend/logic pixel comparisons, zero/odd/even/multiworkgroup counts, strip transitions, winding/quaternion/clip/viewport flags, guards and one draw instead of per-triangle draws; partial-list/foreign-thread rejects passed\n";
}

void CheckIndirectEmissionLists(Metal::Device& device,Metal::Scheduler& scheduler) {
    using Op=nihstro::OpCode::Id;
    const auto word=[](Op op){return unsigned(op)<<26;};
    Metal::ShaderMachineRuntime machine{device,scheduler},reference{device,scheduler};
    Metal::GeometryRuntime geometry{device,scheduler};Metal::TransferRuntime transfer{device,scheduler};
    Metal::DrawRuntime draws{device,scheduler,geometry};
    constexpr unsigned capacity=1023,width=32,height=32;
    constexpr std::string_view fs=R"(#version 450
layout(location=1) in vec4 primary_color;
layout(location=6) in vec4 normquat;
layout(location=0) out vec4 color;
void main(){color=primary_color*0.75+abs(normquat)*0.125;}
)";
    std::array<Metal::CompiledFragmentShader,3> fragments;
    for(unsigned variant=0;variant<3;++variant) {
        Metal::FragmentOperations op{};
        if(!variant) {op.rgb={true,0,6,7};op.alpha={true,0,1,7};}
        else {op.logic_enabled=true;op.logic_operation=variant==1 ? 11:7;}
        auto shader=Metal::TranslateRasterizerFragmentGLSL(fs,op);
        auto library=device.CompileLibrary(shader.source,"GPU-counted GS ordered pixels");
        fragments[variant]={{std::move(shader),op},library};
    }
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    const auto texture=device.CreateTexture(td);
    Metal::PicaDrawState draw_state{};draw_state.use_color=true;
    const auto render=[&](const Metal::AssembledVertices& vertices,const Metal::CompiledFragmentShader& fragment) {
        transfer.ClearColor(texture,MTLClearColorMake(.0625,.125,.25,1));
        auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=texture;
        pass.colorAttachments[0].loadAction=MTLLoadActionLoad;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        draws.Draw(draw_state,fragment,pass,{0,0,width,height,0,1},{0,0,width,height},vertices,{},{},{});
    };
    unsigned cases{},pixel_cases{},vertices_checked{};
    for(unsigned count:{0u,3u,6u,195u,210u,768u}) for(unsigned flags=0;flags<16;++flags) for(bool seeded:{false,true}) {
        Metal::ShaderMachineState unit{device},reference_unit{device};
        Metal::PrimitiveStream stream{device},expected_stream{device};
        if(seeded) {Metal::PrimitiveSnapshot snapshot{};snapshot.needs_reset=false;geometry.LoadStream(stream,snapshot);geometry.LoadStream(expected_stream,snapshot);}
        Metal::ShaderMachineUniforms uniforms{};uniforms.i[0]={count ? count/3-1:0,0,1,0};
        for(unsigned v=0;v<3;++v) {
            uniforms.f[v]={std::array{-1.f,3.f,-1.f}[v],std::array{-1.f,-1.f,3.f}[v],-.5f,1};
            uniforms.f[3+v]={.125f+v*.125f,.5f,.75f,.25f};
            uniforms.f[6+v]={v==1 ? -1.f:1.f,.25f,-0.f,1};
        }
        const auto constants=device.CreateBuffer(sizeof(uniforms),true);std::memcpy(constants.contents,&uniforms,sizeof(uniforms));
        Metal::ShaderMachineProgram program{};program.allow_emission=true;program.output_mask=7;program.swizzle[0]=Metal::Test::IdentitySwizzle;
        program.code[0]=count ? word(Op::LOOP)|(15u<<10):word(Op::END);
        for(unsigned v=0;v<3;++v) {
            const auto pc=1+v*5;
            for(unsigned reg=0;reg<3;++reg) program.code[pc+reg]=Metal::Test::Arithmetic(Op::MOV,reg,32+reg*3+v);
            program.code[pc+3]=word(Op::SETEMIT)|((v*4+(v==2 ? 2:0)+(flags&1))<<22);
            program.code[pc+4]=word(Op::EMIT);
        }
        program.code[16]=word(Op::END);
        Metal::GeometryEmissionConfig config{};config.attribute_count=3;
        config.semantics={0x03020100,0x0b0a0908,0x07060504};
        config.flip_viewport=flags&1;config.enable_clip=(flags>>1)&1;
        config.fix_quaternions=(flags>>2)&1;config.honor_winding=(flags>>3)&1;
        config.clip_coefficient={.5f,-.25f,.125f,.75f};
        const auto abort=machine.BeginGeometryBatch();
        const auto tick=scheduler.CurrentTick();
        const auto pending=machine.QueueGeometry(unit,program,{constants,0,sizeof(uniforms),nullptr,{}},{},abort,0,capacity);
        const auto actual=machine.ConvertEmissionListIndirect(pending,config,stream);
        render(actual,fragments[0]);
        Check(scheduler.CurrentTick()==tick,"GPU-counted GS draw forced CPU completion before rasterization");
        const auto actual_pixels=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
        machine.ValidateIndirectResult(actual);
        const auto expected=reference.Run(reference_unit,program,uniforms,{},0,capacity);
        Check(expected.emission_count==count,"Native GS fixture emitted an unexpected count");
        const auto converted=reference.ConvertEmissionList(expected,config,expected_stream);
        render(converted,fragments[0]);
        Check(actual_pixels==transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4),"GPU-counted GS changed complete ordered blend pixels");++pixel_cases;
        if(count) {
            const auto bytes=NSUInteger(count)*sizeof(Metal::CapturedVertex);
            const auto readback=device.CreateBuffer(bytes*2,true);
            transfer.CopyBuffer(actual.storage.buffer,actual.storage.offset,readback,0,bytes);
            transfer.CopyBuffer(converted.storage.buffer,converted.storage.offset,readback,bytes,bytes);scheduler.Finish();
            Check(!std::memcmp(readback.contents,static_cast<std::byte*>(readback.contents)+bytes,bytes),"GPU indirect GS count/conversion changed complete vertex bytes");
        }
        for(unsigned variant=1;variant<3;++variant) {
            render(actual,fragments[variant]);const auto a=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
            render(converted,fragments[variant]);Check(a==transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4),"GPU-counted GS changed complete ordered logic pixels");++pixel_cases;
        }
        const auto a=geometry.SaveStream(stream),b=geometry.SaveStream(expected_stream);
        Check(a.history==b.history && a.pending==b.pending && a.topology==b.topology && a.needs_reset==b.needs_reset,"GPU-counted GS changed complete conditional272-byte primitive state");
        Check(machine.SaveState(unit)==reference.SaveState(reference_unit),"GPU indirect consumer changed full1824-byte GS state");
        const auto args=Metal::InputReferenceTestAccess::Indirect(actual);
        const auto* words=reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(args.buffer.contents)+args.offset);
        Check(words[0]==count && words[1]==1 && !words[2] && !words[3] && words[4]==std::max(1u,(count+191)/192) && words[5]==1 && words[6]==1 && !words[7] && !words[10] && !words[11] && !words[12],"GPU indirect dispatch/draw ABI or validation bytes differ");
        vertices_checked+=count;++cases;
    }
    // Keep the conditional metadata and both reusable GPU output buffers live
    // across a whole queued chain, including zero counts before/after output.
    unsigned chained_lists{},chained_vertices{},mixed_continuations{};
    for(bool seeded:{false,true}) for(unsigned ending=0;ending<3;++ending) {
        Metal::PrimitiveStream stream{device},expected_stream{device};
        Metal::PrimitiveSnapshot seed{};seed.needs_reset=!seeded;
        if(seeded) for(unsigned i=0;i<256;++i) seed.history[i]=std::byte((i*13+7)&255);
        geometry.LoadStream(stream,seed);geometry.LoadStream(expected_stream,seed);
        const auto source=device.CreateBuffer(capacity*sizeof(Metal::ShaderMachineEmission),false);
        const auto destination=device.CreateBuffer(capacity*sizeof(Metal::CapturedVertex),false);
        const auto comparisons=device.CreateBuffer(12*2*capacity*sizeof(Metal::CapturedVertex),true);
        std::vector<Metal::AssembledVertices> pending_draws;
        std::vector<std::pair<NSUInteger,NSUInteger>> compared_ranges;
        const auto tick=scheduler.CurrentTick();
        const std::array<unsigned,12> counts{0,0,3,0,6,195,0,3,0,6,3,ending==0 ? 0u:ending==1 ? 3u:6u};
        for(unsigned n=0;n<counts.size();++n) {
            const auto count=counts[n];
            const auto input=device.CreateBuffer(source.length,true);std::memset(input.contents,0,input.length);
            auto* emitted=static_cast<Metal::ShaderMachineEmission*>(input.contents);
            for(unsigned v=0;v<count;++v) {
                emitted[v].attributes[0]={std::array{-1.f,3.f,-1.f}[v%3],std::array{-1.f,-1.f,3.f}[v%3],-.5f,1};
                emitted[v].attributes[1]={float(n+1)/16,.5f,.25f,1};
            }
            transfer.CopyBuffer(input,0,source,0,source.length);
            const auto status=device.CreateBuffer(16,true);
            const std::array<unsigned,4> flags{0,count,0,0};std::memcpy(status.contents,flags.data(),16);
            Metal::ShaderMachinePendingResult pending{};pending.capacity=capacity;pending.status={status,0,16,nullptr,{}};
            pending.result.emissions={source,0,source.length,nullptr,{}};
            Metal::GeometryEmissionConfig config{};config.attribute_count=2;config.semantics={0x03020100,0x0b0a0908};
            const auto actual=machine.ConvertEmissionListIndirect(pending,config,stream,{destination,0,destination.length,nullptr,{}});
            const Metal::ShaderMachineResult reference_source{{},{input,0,input.length,nullptr,{}},count};
            const auto expected=reference.ConvertEmissionList(reference_source,config,expected_stream);
            render(actual,fragments[0]);
            if(count) {
                const auto offset=NSUInteger(n)*2*capacity*sizeof(Metal::CapturedVertex),bytes=NSUInteger(count)*sizeof(Metal::CapturedVertex);
                transfer.CopyBuffer(actual.storage.buffer,actual.storage.offset,comparisons,offset,bytes);
                transfer.CopyBuffer(expected.storage.buffer,expected.storage.offset,comparisons,offset+bytes,bytes);
                compared_ranges.emplace_back(offset,bytes);
            }
            pending_draws.push_back(actual);chained_vertices+=count;++chained_lists;
        }
        Check(scheduler.CurrentTick()==tick,"Chained indirect GS reuse forced CPU completion");
        scheduler.Finish();
        for(const auto& vertices:pending_draws) machine.ValidateIndirectResult(vertices);
        for(auto [offset,bytes]:compared_ranges) Check(!std::memcmp(static_cast<std::byte*>(comparisons.contents)+offset,
            static_cast<std::byte*>(comparisons.contents)+offset+bytes,bytes),"Reused indirect GS scratch changed an earlier complete vertex result");
        const auto a=geometry.SaveStream(stream),b=geometry.SaveStream(expected_stream);
        Check(a.history==b.history && a.pending==b.pending && a.topology==b.topology && a.needs_reset==b.needs_reset,
            "Chained zero/nonzero GPU metadata changed complete272-byte history");
        // A positive ordinary append replaces conditional metadata, including
        // incomplete list history and transitions to strip/fan assembly.
        const auto vertices=device.CreateBuffer(3*sizeof(Metal::CapturedVertex),true);std::memset(vertices.contents,0,vertices.length);
        for(auto topology:{Metal::TriangleTopology::List,Metal::TriangleTopology::Strip,Metal::TriangleTopology::Fan}) {
            geometry.LoadStream(stream,a);geometry.LoadStream(expected_stream,b);
            const auto status=device.CreateBuffer(16,true);std::memset(status.contents,0,16);
            Metal::ShaderMachinePendingResult zero{};zero.capacity=capacity;zero.status={status,0,16,nullptr,{}};
            zero.result.emissions={source,0,source.length,nullptr,{}};
            const auto empty=machine.ConvertEmissionListIndirect(zero,{},stream,{destination,0,destination.length,nullptr,{}});
            for(unsigned count:{1u,2u}) {
                geometry.Append(stream,{vertices,0,vertices.length,nullptr,{}},3,topology,count,0,0,{},0,false);
                geometry.Append(expected_stream,{vertices,0,vertices.length,nullptr,{}},3,topology,count,0,0,{},0,false);
                const auto x=geometry.SaveStream(stream),y=geometry.SaveStream(expected_stream);
                Check(x.history==y.history && x.pending==y.pending && x.topology==y.topology && x.needs_reset==y.needs_reset,
                    "Deferred list continuation changed ordinary partial primitive state");++mixed_continuations;
            }
            machine.ValidateIndirectResult(empty);
        }
    }
    std::cout<<"Queued indirect GS reuse: "<<chained_lists<<" zero/nonzero lists, "<<chained_vertices
        <<" complete captured vertices, six uninterrupted GPU metadata chains and "<<mixed_continuations<<" archive/partial/topology continuations passed\n";
    // Malformed producer counts and real shader errors must never reach the
    // vertex buffer. Guard the complete destination, including both ends.
    unsigned rejected_counts{};
    const auto source=device.CreateBuffer(capacity*sizeof(Metal::ShaderMachineEmission),false);
    for(auto flags:std::array{std::array{0u,capacity+3,0u,0u},std::array{0u,2u,0u,0u},std::array{6u,9u,0u,42u}}) {
        Metal::PrimitiveStream stream{device};const auto status=device.CreateBuffer(16,true);std::memcpy(status.contents,flags.data(),16);
        const auto destination=device.CreateBuffer(capacity*sizeof(Metal::CapturedVertex)+32,false);
        transfer.FillBuffer(destination,NSMakeRange(0,destination.length),0x6d);
        Metal::ShaderMachinePendingResult pending{};pending.capacity=capacity;
        pending.status={status,0,16,nullptr,{}};pending.result.emissions={source,0,source.length,nullptr,{}};
        const auto actual=machine.ConvertEmissionListIndirect(pending,{},stream,{destination,16,destination.length-32,nullptr,{}});
        render(actual,fragments[0]);scheduler.Finish();
        bool rejected=false;try{machine.ValidateIndirectResult(actual);}catch(const Metal::ShaderMachineError& error){rejected=error.Code()==(flags[0] ? flags[0]:8);}
        Check(rejected,"GPU indirect count validation accepted a malformed count/error");
        const auto readback=device.CreateBuffer(destination.length,true);transfer.CopyBuffer(destination,0,readback,0,readback.length);scheduler.Finish();
        const auto* bytes=static_cast<const std::byte*>(readback.contents);
        Check(std::all_of(bytes,bytes+readback.length,[](auto byte){return byte==std::byte{0x6d};}),"Invalid GPU count wrote output or crossed destination guards");++rejected_counts;
    }
    std::cout<<"GPU-counted native GS lists: "<<cases<<" actual shader-status/dispatch/draw cases, "<<vertices_checked<<" complete vertices, "<<pixel_cases<<" ordered full-pixel comparisons; zero/3/6/195/210/768, all flags, conditional272-byte history/full1824-byte GS state, no pre-draw completion and "<<rejected_counts<<" malformed/error guard cases passed\n";
}

int main(int argc,char** argv) {
    @autoreleasepool { try {
        Metal::Device device;
        Metal::Scheduler scheduler{device};
        CheckFetchedGeometryVertices(device,scheduler);
        if(argc==2 && std::string_view{argv[1]}=="--fetched-gs-only")return 0;
        CheckCompactGeometry(device,scheduler);
        if(argc==2 && std::string_view{argv[1]}=="--compact-only")return 0;
        CheckIndirectEmissionLists(device,scheduler);
        CheckEmissionLists(device,scheduler);
        CheckQueuedGeometry(device,scheduler);
        CheckQueuedGeometry(device,scheduler,1);
        CheckQueuedGeometry(device,scheduler,2);
        Metal::TransferRuntime transfer{device,scheduler};
        Metal::ShaderMachineRuntime machine{device,scheduler};
        Metal::GeometryProgramRuntime groups{device,scheduler};
        Metal::ShaderMachineState unit{device};
        Metal::GeometryInputState state{device};
        Metal::ShaderMachineProgram program{};
        program.swizzle[0]=Metal::Test::IdentitySwizzle;
        using Op=nihstro::OpCode::Id;
        const auto word=[](Op op) { return std::uint32_t(op)<<26; };
        const auto mov=[](unsigned dest,unsigned source) { return Metal::Test::Arithmetic(Op::MOV,dest,source); };
        Metal::VertexRegisters raw{};
        const auto private_raw=device.CreateBuffer(sizeof(raw),false);
        const Metal::BufferSlice source{private_raw,0,private_raw.length,nullptr,{}};
        const auto upload=[&] {
            const auto visible=device.CreateBuffer(sizeof(raw),true);
            std::memcpy(visible.contents,raw.data(),sizeof(raw));
            transfer.CopyBuffer(visible,0,private_raw,0,private_raw.length);
        };
        const auto output=[](const Metal::ShaderMachineResult& result,unsigned index) {
            return (*static_cast<const Metal::VertexRegisters*>(result.output.data))[index];
        };
        Metal::GeometryInputConfig config{};
        config.vs_output_mask=13;config.vs_output_count=2;config.input_count=4;
        config.input_registers[0]=5;config.input_registers[1]=7;config.input_registers[2]=5;config.input_registers[3]=9;
        {
            // Compare every bank byte after bounded batches, queue submissions and
            // immediate caller mutation. No test assumes a CPU mirror of GPU b15.
            unsigned cases=0;
            for(unsigned count:{1u,2u,7u,63u,64u,65u,127u,128u,129u,257u,4097u})
            for(unsigned pattern=0;pattern<6;++pattern) {
                Metal::GeometryInputState queued{device};
                Metal::ShaderMachineUniforms expected{};
                unsigned random=0x12345678u;
                for(unsigned update=0;update<count;++update) {
                    Metal::ShaderMachineUniforms write{};
                    for(unsigned word_index=0;word_index<400;++word_index) {
                        random=random*1664525u+1013904223u;
                        std::memcpy(reinterpret_cast<std::byte*>(&write)+word_index*4,&random,4);
                    }
                    write.b=random&0xffffu;
                    std::array<std::uint32_t,3> floats{};std::uint32_t booleans{},integers{};
                    if(pattern==0) { floats.fill(~0u);booleans=0xffff;integers=15; }
                    else if(pattern==1) {
                        floats[(update%96)/32]=1u<<(update%32);
                        booleans=1u<<(update%16);integers=1u<<(update%4);
                    } else if(pattern==2) { booleans=update%2 ? 0x5555:0xaaaa;integers=update%3 ? 0:8; }
                    else if(pattern==3) {
                        floats={random,random^0xaaaaaaaa,random^0x55555555};
                        booleans=random&0xffff;integers=random&15;
                    } else if(pattern==4) {
                        // Empty masks must not read a nonexistent packed-value buffer.
                        if(update%3) { floats={0x80000000,1,0x80000000};integers=10; }
                    } else {
                        // Contiguous runs include complete32-bit words and both ends.
                        floats=update%2 ? std::array<std::uint32_t,3>{0x7fffffff,0xfffffffe,0x80000000}:
                            std::array<std::uint32_t,3>{0xffffffff,0x80000001,0};
                        booleans=0x8000;integers=5;
                    }
                    for(unsigned reg=0;reg<96;++reg) if(floats[reg/32]&(1u<<(reg%32)))
                        std::memcpy(&expected.f[reg],&write.f[reg],16);
                    for(unsigned reg=0;reg<4;++reg) if(integers&(1u<<reg)) expected.i[reg]=write.i[reg];
                    expected.b=(expected.b&~booleans)|(write.b&booleans);
                    groups.UpdateUniforms(queued,write,floats,booleans,integers);
                    std::memset(&write,0xcc,sizeof(write));
                    if(update%17==16) scheduler.Finish();
                }
                const auto saved=groups.SaveState(queued);
                Check(saved.initialized && std::memcmp(saved.history.data(),&expected,sizeof(expected))==0,
                    "Batched GS uniform writes changed raw floats, masks, integer values, b15 or padding");
                Check(std::all_of(saved.history.begin()+sizeof(expected),saved.history.end(),
                    [](std::byte value) { return value==std::byte{}; }),
                    "Batched GS uniform writes changed point/input banks");
                ++cases;
            }
            Metal::GeometryInputState queued{device},isolated{device};
            Metal::ShaderMachineUniforms base{};base.f[95]={.25f,.5f,.75f,1};base.b=0x8000;
            groups.UpdateUniforms(queued,base);const auto baseline=groups.SaveState(queued);
            auto replacement=base;replacement.f[95]={4,3,2,1};replacement.b=1;
            groups.UpdateUniforms(queued,replacement,{0,0,1u<<31},1,0);
            auto invalid=baseline;invalid.history[1602]=std::byte{1};
            bool rejected=false;try { groups.LoadState(queued,invalid); } catch(const std::invalid_argument&) { rejected=true; }
            const auto retained=groups.SaveState(queued);
            base.f[95]=replacement.f[95];base.b|=1;
            Check(rejected && std::memcmp(retained.history.data(),&base,sizeof(base))==0,
                "Rejected GS load discarded pending masked writes");
            groups.UpdateUniforms(isolated,replacement);
            for(unsigned i=0;i<65;++i) groups.UpdateUniforms(queued,replacement);
            groups.LoadState(queued,baseline);
            Check(groups.SaveState(queued)==baseline,"GS load replayed stale pending writes after restoration");
            const auto other=groups.SaveState(isolated);
            Check(std::memcmp(other.history.data(),&replacement,sizeof(replacement))==0,
                "Pending GS writes escaped their owning input state");
            groups.UpdateUniforms(queued,replacement);groups.LoadState(queued,{});
            Check(groups.SaveState(queued)==Metal::GeometryInputSnapshot{},
                "Uninitialized GS restore retained a pending write");
            groups.UpdateUniforms(queued,replacement,{0,0,1u<<31},0,0);
            Metal::ShaderMachineUniforms reset{};reset.f[95]=replacement.f[95];
            const auto reset_saved=groups.SaveState(queued);
            Check(std::memcmp(reset_saved.history.data(),&reset,sizeof(reset))==0,
                "Reinitialized GS state reused pre-restore uniform data");
            std::cout<<cases<<" bounded native uniform batches: all1616 bank bytes, caller mutation, submissions, masks, b15, padding, state isolation and pending save/load/reset rollback passed\n";
        }
        {
            // CPU uniform writes have no GS reader here. They must preserve a
            // compatible render pass until a later native bank consumer.
            Metal::GeometryInputState pending{device};
            Metal::ShaderMachineUniforms expected{};
            groups.UpdateUniforms(pending,expected);(void)groups.SaveState(pending);
            auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:8 height:8 mipmapped:NO];
            descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget;
            const auto texture=device.CreateTexture(descriptor);
            transfer.ClearColor(texture,MTLClearColorMake(.25,.5,.75,1));
            auto* pass=[[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture=texture;pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            auto encoder=scheduler.BeginCompatibleRender(pass);unsigned breaks=0;
            for(unsigned n=0;n<257;++n) {
                Metal::ShaderMachineUniforms write{};write.f[n%96]={float(n),-float(n),.25f,1};
                write.i[n%4]={n,n^255u,n+1,0};write.b=n*31337u;
                std::array<std::uint32_t,3> mask{};mask[(n%96)/32]=1u<<(n%32);
                const auto bool_mask=1u<<(n%16);
                groups.UpdateUniforms(pending,write,mask,bool_mask,1u<<(n%4));
                expected.f[n%96]=write.f[n%96];expected.i[n%4]=write.i[n%4];
                expected.b=(expected.b&~bool_mask)|(write.b&bool_mask);
                const auto current=scheduler.BeginCompatibleRender(pass);
                if(current!=encoder) {++breaks;encoder=current;}
            }
            const auto saved=groups.SaveState(pending);
            Check(!std::memcmp(saved.history.data(),&expected,sizeof(expected)),
                "Deferred uniform consumer lost last masked values or GPU padding");
            const auto pixels=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,8,8)},32);
            constexpr std::array<unsigned,4> color{64,128,191,255};
            for(unsigned byte=0;byte<pixels.size();++byte)
                Check(std::abs(int(std::to_integer<unsigned>(pixels[byte]))-int(color[byte%4]))<=1,
                    "Uniform-only staging changed unrelated render pixels");
            std::cout<<"257 GS uniform writes without a consumer: render-pass breaks="<<breaks
                <<", complete1616 bank bytes and256 unrelated image bytes exact\n";
            Check(!breaks,"Unconsumed GS uniform writes interrupted compatible rendering");
            bool rejected=false;
            std::thread foreign([&] {
                try { groups.UpdateUniforms(pending,{}); }
                catch(const std::exception&) { rejected=true; }
            });foreign.join();
            Check(rejected && groups.SaveState(pending)==saved,
                "Foreign thread changed native pending uniform transport");
        }
        groups.Configure(state,config);
        Metal::ShaderMachineUniforms values{};values.f[95]={.25f,.5f,.75f,1};groups.UpdateUniforms(state,values);
        raw[0]={1,2,3,4};raw[2]={5,6,7,8};raw[3].fill(99);upload();
        Check(!groups.SubmitVertex(state,source),"Point GS invoked before its group was full");
        scheduler.Finish(); // The incomplete point group survives a real queue submission.
        const auto saved_partial_point=groups.SaveState(state);
        const auto saved_initial_unit=machine.SaveState(unit);
        auto other=config;other.input_count=2;
        bool rejected=false;
        try { groups.Configure(state,other); } catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected,"Native GS discarded a partial group during reconfiguration");
        groups.Configure(state,config);
        raw[0]={9,10,11,12};raw[2]={13,14,15,16};upload();
        Check(groups.SubmitVertex(state,source),"Point GS did not complete its group");
        program.code[0]=mov(0,5);program.code[1]=mov(1,7);program.code[2]=mov(2,9);program.code[3]=word(Op::END);
        auto result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==raw[0] && output(result,1)==std::array<float,4>{5,6,7,8} && output(result,2)==raw[2],
              "Point GS compact outputs, alias order or register mapping was incorrect");
        Check(state.IsEmpty(),"Point GS did not leave an empty group after invocation");
        groups.LoadState(state,saved_partial_point);machine.LoadState(unit,saved_initial_unit);
        Check(groups.SaveState(state)==saved_partial_point,"Native GS partial point snapshot changed its banks/configuration");
        groups.SubmitVertex(state,source);result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==raw[0] && output(result,1)==std::array<float,4>{5,6,7,8},
              "Native GS partial point snapshot did not resume the original group");
        config.input_count=2;config.input_registers[1]=9;groups.Configure(state,config);
        program.code.fill(0);program.code[0]=word(Op::IFU)|(15u<<22)|(2u<<10)|1u;
        program.code[1]=mov(0,5);program.code[2]=mov(0,32+95);program.code[3]=mov(1,7);program.code[4]=word(Op::END);
        Check(groups.SubmitVertex(state,source),"Point GS one-vertex group was not ready");
        result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==raw[0] && output(result,1)==std::array<float,4>{5,6,7,8},
              "GS b15 or retained unmapped input register was lost");
        values.b=0;values.f[95].fill(99);
        groups.UpdateUniforms(state,values,{0,0,0},1,0); // Does not overwrite b15 or f95.
        groups.SubmitVertex(state,source);result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==raw[0],"An unrelated boolean write cleared GPU b15");
        groups.UpdateUniforms(state,values,{0,0,0},0x8000,0);
        groups.SubmitVertex(state,source);result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==std::array<float,4>{.25f,.5f,.75f,1},"Explicit b15 write or retained float uniform failed");
        config.mode=Metal::GeometryInputMode::Fixed;config.uniform_start=90;config.fixed_vertex_count=2;
        groups.Configure(state,config);program.code.fill(0);
        program.code[0]=mov(0,32+90);program.code[1]=mov(1,32+91);program.code[2]=mov(2,32+92);program.code[3]=word(Op::END);
        raw[0]={1,2,3,4};raw[2]={5,6,7,8};upload();groups.SubmitVertex(state,source);
        const auto saved_fixed=groups.SaveState(state);
        Metal::GeometryInputState restored_fixed{device};groups.LoadState(restored_fixed,saved_fixed);
        Check(groups.SaveState(restored_fixed)==saved_fixed,"Recreated fixed GS group lost its partially filled uniform range");
        groups.LoadState(state,groups.SaveState(restored_fixed));
        values.f[90]={.125f,.25f,.5f,1};values.f[91].fill(999);
        groups.UpdateUniforms(state,values,{0,0,1u<<26},0,0);
        raw[0]={9,10,11,12};raw[2]={13,14,15,16};upload();
        Check(groups.SubmitVertex(state,source),"Fixed GS did not complete its uniform group");
        result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==values.f[90] && output(result,1)==std::array<float,4>{5,6,7,8} && output(result,2)==raw[0],
              "Fixed GS uniform range, partial write or retained slots were incorrect");
        config.mode=Metal::GeometryInputMode::Variable;config.variable_main_count=2;groups.Configure(state,config);
        Check(state.NeedsCount(),"Variable GS did not request its count index");
        for(auto count:{0u,97u}) {
            rejected=false;try { groups.BeginVariable(state,count); } catch(const std::invalid_argument&) { rejected=true; }
            Check(rejected && state.NeedsCount(),"Invalid variable GS count changed group state");
        }
        values.f[0].fill(999);groups.UpdateUniforms(state,values,{1,0,0},0,0);
        groups.BeginVariable(state,4);
        for(unsigned i=0;i<4;++i) {
            raw[0].fill(float(10+i));raw[2].fill(float(20+i));upload();
            Check(groups.SubmitVertex(state,source)==(i==3),"Variable GS completed at the wrong vertex");
            if(i==1 || i==3) {
                const auto saved=groups.SaveState(state);
                Metal::GeometryInputState temporary{device};groups.LoadState(temporary,saved);
                Check(groups.SaveState(temporary)==saved,"Recreated variable GS group lost progress/private uniform data");
                groups.LoadState(state,groups.SaveState(temporary));
            }
        }
        program.code.fill(0);
        for(unsigned i=0;i<5;++i) program.code[i]=mov(i,32+std::array<unsigned,5>{0,2,4,5,6}[i]);
        program.code[5]=word(Op::END);result=groups.Invoke(state,machine,unit,program);
        for(unsigned i=0;i<5;++i) {
            std::array<float,4> expected{};expected.fill(std::array<float,5>{4,20,21,12,13}[i]);
            Check(output(result,i)==expected,"Variable GS count/main/all versus trailing/first attribute policy failed");
        }
        Check(state.NeedsCount(),"Variable GS did not return to its count boundary");
        groups.BeginVariable(state,93); // Exactly fills f0 through f95 with two main vertices.
        for(unsigned i=0;i<93;++i) {
            raw[0].fill(float(i));raw[2].fill(float(100+i));upload();
            Check(groups.SubmitVertex(state,source)==(i==92),"Maximum variable GS group completed early");
        }
        program.code.fill(0);program.code[0]=mov(0,32);program.code[1]=mov(1,32+95);program.code[2]=word(Op::END);
        result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==std::array<float,4>{93,93,93,93} && output(result,1)==std::array<float,4>{92,92,92,92},
              "Native variable GS group did not reach the exact end of its uniform bank");
        for(unsigned failure=0;failure<4;++failure) {
            auto bad=config;
            if(failure==0) bad.vs_output_mask=0x10000;
            if(failure==1) bad.vs_output_count=0;
            if(failure==2) { bad.mode=Metal::GeometryInputMode::Fixed;bad.fixed_vertex_count=1;bad.uniform_start=95; }
            if(failure==3) { bad.mode=Metal::GeometryInputMode::Point;bad.input_count=3; }
            rejected=false;try { groups.Configure(state,bad); } catch(const std::invalid_argument&) { rejected=true; }
            Check(rejected && state.NeedsCount(),"Invalid native GS configuration changed the previous empty group");
        }
        config.mode=Metal::GeometryInputMode::Point;config.input_count=2;config.input_registers[0]=0;config.input_registers[1]=1;
        groups.Configure(state,config);values.b=0;groups.UpdateUniforms(state,values,{0,0,0},0x8000,0);
        raw[0]={1,2,3,4};upload();groups.SubmitVertex(state,source);
        rejected=false;try { groups.UpdateUniforms(state,values); } catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected,"A completed native GS group accepted unrelated uniform writes before invocation");
        program.code.fill(0);program.code[0]=word(static_cast<Op>(0x10));program.code[1]=word(Op::END);
        rejected=false;try { groups.Invoke(state,machine,unit,program); } catch(const std::runtime_error&) { rejected=true; }
        Check(rejected && !state.IsEmpty(),"Failed native GS discarded its complete input group");
        const auto retry_group=groups.SaveState(state);
        Metal::GeometryInputState restored_retry{device};groups.LoadState(restored_retry,retry_group);
        Check(groups.SaveState(restored_retry)==retry_group,"Native GS snapshot lost a complete group awaiting retry");
        groups.LoadState(state,groups.SaveState(restored_retry));
        program.code[0]=word(Op::IFU)|(15u<<22)|(2u<<10)|1u;
        program.code[1]=mov(0,0);program.code[2]=mov(0,32+95);program.code[3]=word(Op::END);
        result=groups.Invoke(state,machine,unit,program);
        Check(output(result,0)==std::array<float,4>{92,92,92,92},"Failed native GS set b15 before a successful retry");
        values.i[0]={2,0,1,0};groups.UpdateUniforms(state,values,{0,0,0},0,1);
        program.code.fill(0);program.code[0]=word(Op::LOOP)|(1u<<10);
        program.code[1]=Metal::Test::Arithmetic(Op::ADD,16,16,0);program.code[2]=mov(0,16);program.code[3]=word(Op::END);
        for(unsigned iteration=0;iteration<2;++iteration) {
            unit.Reset();groups.SubmitVertex(state,source);result=groups.Invoke(state,machine,unit,program);
            Check(output(result,0)==std::array<float,4>{3,6,9,12},"Native GS integer uniform write/retention or LOOP count was incorrect");
            values.i[0]={9,0,0,0};values.i[1]={7,0,1,0};
            groups.UpdateUniforms(state,values,{0,0,0},0,2); // i1 changes; GPU i0 remains {2,0,1,0}.
        }
        const auto stable_group=groups.SaveState(state);
        for(unsigned failure=0;failure<6;++failure) {
            auto invalid=stable_group;
            if(failure==0) invalid.cursor=16;
            if(failure==1) invalid.config.input_registers[0]=16;
            if(failure==2) invalid.awaiting=true,invalid.cursor=2;
            if(failure==3) invalid.remaining=1;
            if(failure==4) invalid.history[1604]=std::byte{1};
            if(failure==5) invalid.history[1602]=std::byte{1};
            bool rejected_snapshot=false;
            try { groups.LoadState(state,invalid); } catch(const std::invalid_argument&) { rejected_snapshot=true; }
            Check(rejected_snapshot && groups.SaveState(state)==stable_group,"Invalid GS group snapshot partially changed GPU banks or metadata");
        }
        std::cout<<"Native Point/Fixed/Variable GS groups, compact VS masks, mapped aliases, retained inputs/uniforms, explicit partial writes, GPU b15 and count bounds passed\n";
        std::cout<<"Native GS group snapshots: partial point/fixed/variable continuation, recreated private banks and malformed-payload rollback passed\n";

        // 70 triangles exercise more than one threadgroup; reference uses original PICA semantic mapping.
        std::array<Metal::ShaderMachineEmission,210> emissions{};
        for(unsigned i=0;i<emissions.size();++i) {
            auto& v=emissions[i];v.winding=i%3==0 && (i/3)%2;
            v.attributes[0]={float(i)/256,-.25f,std::array<float,3>{-.5f,.0000005f,-1.000005f}[i%3],1};
            v.attributes[1]={i%3==1 ? -1.f:1.f,0,0,0};
            v.attributes[2]={-.25f,1.5f,.5f,i%3==0 ? std::numeric_limits<float>::quiet_NaN():.75f};
            v.attributes[3]={.125f,.25f,.375f,.5f};v.attributes[4]={.625f,.75f,.875f,1};
            v.attributes[5]={1.125f,1.25f,1.375f,1.5f};v.attributes[6]={-.125f,.375f,99,98};
        }
        const auto emission_visible=device.CreateBuffer(sizeof(emissions)+16,true),emission_private=device.CreateBuffer(sizeof(emissions)+16,false);
        std::memcpy(static_cast<std::byte*>(emission_visible.contents)+16,emissions.data(),sizeof(emissions));
        transfer.CopyBuffer(emission_visible,0,emission_private,0,emission_private.length);
        Metal::ShaderMachineResult emitted{{},{emission_private,16,sizeof(emissions),nullptr,{}},210};
        Metal::GeometryEmissionConfig conversion{};
        conversion.semantics={0x03020100,0x07060504,0x0b0a0908,0x0f0e0d0c,0x13121110,0x17161514,0x1f1f0a00};
        conversion.clip_coefficient={.5f,-.25f,.125f,.75f};
        unsigned converted_cases{};
        for(unsigned attributes=0;attributes<=7;++attributes) for(unsigned flags=0;flags<16;++flags) {
            conversion.attribute_count=attributes;conversion.flip_viewport=flags&1;
            conversion.enable_clip=(flags>>1)&1;conversion.fix_quaternions=(flags>>2)&1;conversion.honor_winding=(flags>>3)&1;
            const auto converted=machine.ConvertEmissions(emitted,conversion);
            Check(converted.storage.buffer.storageMode==MTLStorageModePrivate,"Native GS capture output must stay in private GPU storage");
            const auto readback=device.CreateBuffer(converted.vertex_count*sizeof(Metal::CapturedVertex),true);
            transfer.CopyBuffer(converted.storage.buffer,0,readback,0,readback.length);scheduler.Finish();
            const auto* actual=static_cast<const Metal::CapturedVertex*>(readback.contents);
            for(unsigned triangle=0;triangle<70;++triangle) {
                const unsigned base=triangle*3;
                const bool reverse=conversion.honor_winding && emissions[base].winding;
                std::array<Metal::CapturedVertex,3> expected{Reference(emissions[base+(reverse ? 1:0)],conversion),
                    Reference(emissions[base+(reverse ? 0:1)],conversion),Reference(emissions[base+2],conversion)};
                if(conversion.fix_quaternions) for(unsigned v=1;v<3;++v) {
                    float dot{};for(unsigned c=0;c<4;++c) dot+=expected[0].quaternion[c]*expected[v].quaternion[c];
                    if(dot<0) for(auto& component:expected[v].quaternion) component=-component;
                }
                for(unsigned v=0;v<3;++v) {
                    const auto* a=reinterpret_cast<const float*>(&actual[base+v]);
                    const auto* e=reinterpret_cast<const float*>(&expected[v]);
                    for(unsigned component=0;component<32;++component)
                        if(!(std::abs(a[component]-e[component])<.00001f))
                            throw std::runtime_error("Native GS reference mismatch attributes="+std::to_string(attributes)+
                                " flags="+std::to_string(flags)+" triangle="+std::to_string(triangle)+
                                " vertex="+std::to_string(v)+" component="+std::to_string(component)+
                                " actual="+std::to_string(a[component])+" expected="+std::to_string(e[component]));
                }
            }
            ++converted_cases;
        }
        rejected=false;auto invalid=emitted;invalid.emission_count=2;
        try { machine.ConvertEmissions(invalid,conversion); } catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected,"Native GS converter accepted an incomplete primitive");
        invalid=emitted;invalid.emissions.size=272;
        rejected=false;try { machine.ConvertEmissions(invalid,conversion); } catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected,"Native GS converter read past its emission slice");
        std::cout<<converted_cases<<" native GS output configurations against original PICA semantics: defaults/aliases, color saturation/NaN, winding, quaternion, viewport flip, epsilon clipping and multiple workgroups passed\n";

        // Real VS -> GS EMIT program -> private GPU capture -> original PICA FS -> all pixels.
        Metal::GeometryInputState draw_inputs{device};Metal::ShaderMachineState draw_unit{device};
        Metal::GeometryInputConfig draw_config{};draw_config.vs_output_mask=1;draw_config.vs_output_count=1;draw_config.input_count=1;
        groups.Configure(draw_inputs,draw_config);
        Metal::ShaderMachineUniforms draw_uniforms{};
        draw_uniforms.f[0]={-1,-1,-.5f,1};draw_uniforms.f[1]={3,-1,-.5f,1};draw_uniforms.f[2]={-1,3,-.5f,1};
        groups.UpdateUniforms(draw_inputs,draw_uniforms);
        Metal::ShaderMachineProgram draw_program{};draw_program.swizzle[0]=Metal::Test::IdentitySwizzle;draw_program.output_mask=3;
        for(unsigned vertex=0;vertex<3;++vertex) {
            const unsigned pc=vertex*4;
            draw_program.code[pc]=mov(0,32+vertex);draw_program.code[pc+1]=mov(1,0);
            draw_program.code[pc+2]=word(Op::SETEMIT)|(vertex<<24)|(vertex==2 ? 1u<<23:0);
            draw_program.code[pc+3]=word(Op::EMIT);
        }
        draw_program.code[12]=word(Op::END);
        raw[0]={0,1,1,1};upload();
        Metal::ShaderMachineState vs_unit{device};Metal::ShaderMachineProgram vs_program{};
        vs_program.swizzle[0]=Metal::Test::IdentitySwizzle;vs_program.output_mask=1;
        vs_program.code[0]=mov(0,0);vs_program.code[1]=word(Op::END);
        const auto vs_result=machine.Run(vs_unit,vs_program,{},source);
        Check(!vs_result.emission_count,"Native test VS unexpectedly emitted a GS primitive");
        groups.SubmitVertex(draw_inputs,vs_result.output);
        auto drawn=groups.Invoke(draw_inputs,machine,draw_unit,draw_program);
        Metal::GeometryEmissionConfig draw_conversion{};draw_conversion.attribute_count=2;draw_conversion.semantics[0]=0x03020100;draw_conversion.semantics[1]=0x0b0a0908;
        const auto vertices=machine.ConvertEmissions(drawn,draw_conversion);
        Metal::GeometryRuntime geometry{device,scheduler};Metal::DrawRuntime draws{device,scheduler,geometry};Metal::ShaderRuntime shaders{device};
        Pica::RegsInternal regs{};regs.lighting.disable.Assign(1);regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
        regs.framebuffer.framebuffer.allow_color_write.Assign(1);regs.framebuffer.output_merger.depth_color_mask=15<<8;
        const auto fragment=shaders.Fragment(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8);
        const auto fs=device.CreateBuffer(sizeof(Pica::Shader::Generator::FSUniformData),true);
        std::memset(fs.contents,0,fs.length);static_cast<Pica::Shader::Generator::FSUniformData*>(fs.contents)->depth_offset=.5f;
        auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:32 height:32 mipmapped:NO];
        descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        const auto target=device.CreateTexture(descriptor);
        auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=target;
        pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        const std::array<Metal::BufferBinding,1> bindings{{{fs,0,2}}};
        draws.Draw(Metal::BuildPicaDrawState(regs),*fragment,pass,{0,0,32,32,0,1},{0,0,32,32},vertices,bindings,{},{});
        const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},32*4);
        for(unsigned i=0;i<pixels.size();++i) Check(pixels[i]==std::byte(i%4==0 ? 0:255),"Native GS emission to real PICA FS offscreen draw produced incorrect pixels");
        std::cout<<"Real native VS -> GS EMIT/SETEMIT -> private GPU capture -> original PICA FS -> all 1024 offscreen pixels passed; no app/window/emulation started\n";
        // Check the same NaN-color rule through the existing translated No-GS vertex path.
        Pica::ShaderSetup no_gs_setup{};Pica::RegsInternal no_gs_regs{};
        Metal::Test::MovVertex(no_gs_regs,no_gs_setup);
        raw[0]={0,0,-.5f,1};raw[1]={1,0,0,0};
        raw[2]={-.5f,2,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN()};upload();
        const auto no_gs=shaders.Vertex(no_gs_regs,no_gs_setup);
        Pica::Shader::Generator::VSPicaUniformData guest_uniforms{}; // MOV fixture reads no constants.
        const auto no_gs_guest=device.CreateBuffer(sizeof(guest_uniforms),true),no_gs_common=device.CreateBuffer(sizeof(Pica::Shader::Generator::VSUniformData),true);
        std::memcpy(no_gs_guest.contents,&guest_uniforms,sizeof(guest_uniforms));std::memset(no_gs_common.contents,0,no_gs_common.length);
        const std::array<Metal::BufferBinding,2> no_gs_uniforms{{{no_gs_guest,0,0},{no_gs_common,0,1}}};
        const auto captured=geometry.Capture(*no_gs,source,1,0,no_gs_uniforms);
        const auto no_gs_readback=device.CreateBuffer(sizeof(Metal::CapturedVertex),true);
        transfer.CopyBuffer(captured.buffer,captured.offset,no_gs_readback,0,no_gs_readback.length);scheduler.Finish();
        Check(static_cast<const Metal::CapturedVertex*>(no_gs_readback.contents)->color==std::array<float,4>{.5f,1,1,1},
              "Translated native No-GS vertex color does not match PICA abs/saturation/NaN semantics");
        std::cout<<"Translated native No-GS PICA color abs/saturation/NaN regression passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }
}
