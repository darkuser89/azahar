// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_draw_runtime.h"
#include "mtl_shader_machine.h"
#include "mtl_transfer.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_compiler_gate_fixture.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

namespace Metal {
struct InputReferenceTestAccess {
    static auto MeshProgram(const NativeMeshDraw& mesh) { return mesh.program; }
    static void MeshProgram(NativeMeshDraw& mesh, std::shared_ptr<const CompiledMeshVertexProgram> program) {
        mesh.program = std::move(program);
    }
    static NativeProgramSnapshot Program(ShaderMachineRuntime& runtime,const ShaderMachineProgram& source,
        const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config) {
        auto selected=runtime.SelectNativeReset(source,uniforms,config);
        if(!selected)throw std::invalid_argument("Owned mesh fixture requires native selection");
        return runtime.NativeProgram(*selected);
    }
    static auto OwnedMesh(ShaderRuntime& shaders,const Pica::RegsInternal& regs,const NativeProgramSnapshot& snapshot) {
        return shaders.MeshVertex(regs,snapshot);
    }
    static NativeProgramSnapshot SeparateOwner(Device& device,Scheduler& scheduler,ShaderMachineRuntime& runtime,
        const ShaderMachineProgram& source,const GeometryEmissionConfig& config) {
        ShaderMachineRuntime other{device,scheduler};auto selected=other.SelectNativeReset(source,{},config);
        if(!selected)throw std::runtime_error("Separate owner fixture was not selected");
        bool rejected=false;try{(void)runtime.NativeProgram(*selected);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Foreign runtime sealed another runtime's selected source");
        auto moved=std::move(*selected);rejected=false;
        try{(void)other.NativeProgram(*selected);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Moved-from native selection produced a sealed source");
        return other.NativeProgram(moved);
    }
    static void CheckSelection(Device& device,Scheduler& scheduler,ShaderMachineRuntime& runtime,
        InputReferenceRuntime& references,const VertexRegisters& inputs,
        const ShaderMachineProgram& original,const GeometryEmissionConfig& config);
    static void Commit(VertexInputState& state, const PreparedInputReferences& prepared) {
        state.references = prepared.next;
        state.needs_reset = false;
    }
    static PreparedNativeInputReferences PrepareNative(InputReferenceRuntime& runtime,
        const VertexInputState& state, std::span<const BufferSlice> inputs, const PicaVertexLayout& layout,
        unsigned vertices, unsigned elements, unsigned base, const BufferSlice& indices, unsigned width) {
        return runtime.PrepareNative(state, inputs, layout, vertices, elements, base, base, indices, width);
    }
    static void Adopt(NativeMeshDraw& mesh, const PreparedNativeInputReferences& prepared,
                      Device& device, unsigned count) {
        if (!prepared.snapshot || prepared.snapshot->validated_device != device.Handle() ||
            prepared.snapshot->validated_count != count)
            throw std::runtime_error("Native reference preparation did not certify its exact extent");
        if (std::memcmp(mesh.references.data, prepared.metadata.data, 16 * sizeof(InputReferenceGPU)))
            throw std::runtime_error("Native reference metadata differs from the checked borrowed path");
        mesh.inputs = prepared.snapshot;
        mesh.references = prepared.metadata;
    }
    static void Commit(VertexInputState& state, const PreparedNativeInputReferences& prepared) {
        state.references = prepared.next;
        state.needs_reset = false;
    }
    static bool Retains(const VertexInputState& state, const std::shared_ptr<const InputProvenance>& next) {
        return state.references == next && !state.needs_reset;
    }
    static std::shared_ptr<const NativeVertexInputSnapshot> Snapshot(
        const std::array<InputReference, 16>& refs, const BufferSlice& indices) {
        return std::shared_ptr<const NativeVertexInputSnapshot>(new NativeVertexInputSnapshot(refs, indices));
    }
    static void Defer(ShaderMachineRuntime& runtime, ShaderMachineState& state,
        const ShaderMachineProgram& program, const ShaderMachineUniforms& values,
        const GeometryEmissionConfig& config, std::shared_ptr<const NativeVertexInputSnapshot> snapshot,
        unsigned vertices, unsigned elements, unsigned base, unsigned width) {
        auto selected=runtime.SelectNativeReset(program,values,config);
        if (!selected)
            throw std::invalid_argument("Reference test requires proven native outputs");
        runtime.DeferNativeReferenceBatch(state, std::move(*selected), std::move(snapshot), vertices, elements, base, base, width);
    }
    static void KnownZero(PrimitiveStream& stream) {
        stream.pending = 0; stream.topology = TriangleTopology::List;
        stream.needs_reset = false; stream.known_zero_history = true;
    }
    static void Poison(TransferRuntime& transfer, PrimitiveStream& stream) {
        transfer.FillBuffer(stream.history, NSMakeRange(stream.history_offset, 272), 0x5a);
    }
    static NativeMeshDraw Mesh(Device& device, InputReferenceRuntime& references, ShaderRuntime& shaders,ShaderMachineRuntime& runtime,
        const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
        const PreparedInputReferences& prepared, const BufferSlice& indices,
        unsigned vertices, unsigned elements, unsigned base, unsigned width, bool fix) {
        NativeMeshDraw mesh;
        ShaderMachineProgram source{setup.GetProgramCode(),setup.GetSwizzleData(),regs.vs.main_offset,regs.vs.output_mask,false};
        GeometryEmissionConfig config{};config.attribute_count=regs.rasterizer.vs_output_total;config.honor_winding=0;
        for(unsigned i=0;i<7;++i)config.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
        mesh.program = OwnedMesh(shaders,regs,Program(runtime,source,{},config));
        mesh.inputs = Snapshot(prepared.current_registers, indices);
        mesh.references = references.Upload(prepared.current_registers);
        const std::array<std::uint32_t, 4> capture{vertices, base, 0, 0};
        auto cp = device.CreateBuffer(16, true); std::memcpy(cp.contents, capture.data(), 16);
        mesh.capture = {cp, 0, 16, cp.contents, {}};
        mesh.vertex_count = vertices; mesh.element_count = elements; mesh.index_size = width;
        mesh.empty = device.CreateBuffer(512, true); std::memset(mesh.empty.contents, 0x5a, 512);
        const auto triangles = elements / 3;
        for (unsigned first = 0; first < triangles;) {
            const unsigned count = std::min(32768u, triangles - first);
            const std::array<unsigned, 12> values{elements, width, base, base, 0, 0, 1, fix, triangles, 0, 1, first};
            auto buffer = device.CreateBuffer(48, true); std::memcpy(buffer.contents, values.data(), 48);
            mesh.chunks.push_back({{buffer, 0, 48, buffer.contents, {}}, (count + 31) / 32}); first += count;
        }
        return mesh;
    }
    static bool DummyUntouched(const NativeMeshDraw& mesh) {
        const auto* bytes = static_cast<const unsigned char*>(mesh.empty.contents);
        return std::all_of(bytes, bytes + 512, [](auto byte) { return byte == 0x5a; });
    }
};
} // namespace Metal

void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

void Metal::InputReferenceTestAccess::CheckSelection(Device& device,Scheduler& scheduler,
    ShaderMachineRuntime& runtime,InputReferenceRuntime& references,const VertexRegisters& inputs,
    const ShaderMachineProgram& original,const GeometryEmissionConfig& config) {
    auto program=original;
    program.code[3]=Metal::Test::Arithmetic(nihstro::OpCode::Id::ADD,16,32,16);
    ShaderMachineUniforms values{};values.f[0]={.125f,.25f,.5f,1.f};
    const auto saved_program=program;const auto saved_values=values;
    std::array<InputReference,16> fixed{};
    for(unsigned n=0;n<16;++n) fixed[n]=InputReference::Constant(inputs[n]);
    const auto snapshot=Snapshot(fixed,{});
    auto canonical=references.Materialize(fixed,3);
    ShaderMachineState golden{device},unit{device};
    (void)runtime.RunVertexBatch(golden,program,values,canonical,3,3,0,0);
    const auto wanted=runtime.SaveState(golden);
    const std::array<float,4> expected_temporary{.375f,.75f,1.5f,3.f};
    Check(!std::memcmp(wanted.history.data()+256,expected_temporary.data(),16),
        "Selection oracle did not execute the original uniform-dependent GPU recurrence");
    auto selected=runtime.SelectNativeReset(program,values,config);
    Check(bool(selected),"Independent selection was unexpectedly rejected");
    program.code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,2);
    program.code[4095]^=0xabcdef01u;program.swizzle[0]^=1u<<4;
    program.entry=4;program.output_mask=1;program.allow_emission=true;
    values.f[0]={7,8,9,10};values.b=65535;values.i[0]={1,2,3,4};
    for(unsigned n=0;n<270;++n) {
        auto other=saved_program;other.code[4095]=n+1;other.swizzle[4095]=n^0x5a5a0000u;
        Check(runtime.CanCaptureNativeReset(other,saved_values,config),"Eviction fixture lost native selection");
    }
    runtime.DeferNativeReferenceBatch(unit,std::move(*selected),snapshot,3,3,0,0,0);
    Check(runtime.SaveState(unit)==wanted,"Selected owned program/uniforms changed after source mutation/cache eviction");
    bool consumed_rejected=false;
    try{(void)runtime.NativeProgram(*selected);}catch(const std::invalid_argument&){consumed_rejected=true;}
    Check(consumed_rejected,"Consumed native selection produced a new sealed source");
    bool rejected=false;
    try { runtime.DeferNativeReferenceBatch(unit,std::move(*selected),snapshot,3,3,0,0,0); }
    catch(const std::invalid_argument&) { rejected=true; }
    Check(rejected && runtime.SaveState(unit)==wanted,"Consumed selection was reused or changed complete state");
    selected=runtime.SelectNativeReset(saved_program,saved_values,config);
    auto moved=std::move(*selected);rejected=false;
    try { runtime.DeferNativeReferenceBatch(unit,std::move(*selected),snapshot,3,3,0,0,0); }
    catch(const std::invalid_argument&) { rejected=true; }
    Check(rejected && runtime.SaveState(unit)==wanted,"Moved-from selection was reused or changed complete state");
    rejected=false;
    try { runtime.DeferNativeReferenceBatch(unit,std::move(moved),nullptr,3,3,0,0,0); }
    catch(const std::invalid_argument&) { rejected=true; }
    Check(rejected && runtime.SaveState(unit)==wanted,"Invalid inputs published selected state");
    {
        ShaderMachineRuntime foreign{device,scheduler};rejected=false;
        try { foreign.DeferNativeReferenceBatch(unit,std::move(moved),snapshot,3,3,0,0,0); }
        catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected && runtime.SaveState(unit)==wanted,"Another runtime accepted a selected pending batch");
    }
    runtime.DeferNativeReferenceBatch(unit,std::move(moved),snapshot,3,3,0,0,0);
    Check(runtime.SaveState(unit)==wanted,"Rejected publication consumed the valid selection");
    auto invalid=config;invalid.semantics[0]=0x03020118;rejected=false;
    try { (void)runtime.SelectNativeReset(saved_program,saved_values,invalid); }
    catch(const std::invalid_argument&) { rejected=true; }
    Check(rejected && runtime.SaveState(unit)==wanted,"Selection accepted invalid output semantics");
    auto dependent=saved_program;dependent.output_mask=1;
    dependent.code[0]=Metal::Test::Arithmetic(nihstro::OpCode::Id::ADD,16,16,0);
    dependent.code[1]=Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,0,16);
    dependent.code[2]=std::uint32_t(nihstro::OpCode::Id::END)<<26;
    Check(!runtime.SelectNativeReset(dependent,saved_values,config),"Selection accepted outputs depending on previous invocations");
    Check(runtime.SaveState(unit)==wanted,"Unselected program changed pending native state");
    std::cout<<"Native selection ownership: code/swizzle/entry/mask/emission/uniform mutation,270 cache evictions, consumed/moved/foreign rejection, invalid-input retry/semantics and dependent-output rejection preserve complete1824-byte GPU state\n";
}

void CheckExactMeshVertexCache(Metal::Device& device) {
    Metal::ShaderRuntime cache{device};
    Pica::RegsInternal regs{};
    Pica::ShaderSetup setup;
    Metal::Test::InitializeFixtureUniforms(setup);
    Metal::Test::MovVertex(regs, setup);
    const auto original_regs = regs;
    const auto original_program = setup.GetProgramCode();
    const auto original_swizzle = setup.GetSwizzleData();
    const auto restore = [&] {
        regs = original_regs;
        setup.UpdateProgramCode(original_program);
        setup.UpdateSwizzleData(original_swizzle);
    };
    const auto first = cache.MeshVertex(regs, setup);
    const auto first_source = first->shader.source;
    std::vector<std::function<void()>> mutations;
    for (unsigned field = 0; field < 11; ++field) mutations.push_back([&, field] {
        switch (field) {
        case 0: regs.vs.main_offset.Assign(1); break;
        case 1: regs.vs.output_mask.Assign(3); break;
        case 2: regs.rasterizer.vs_output_total.Assign(2); break;
        case 3: regs.lighting.disable.Assign(1); break;
        default: regs.rasterizer.vs_output_attributes[field - 4].raw ^= 1; break;
        }
    });
    // Even unreachable bank words are part of cache identity. A recent-key
    // shortcut must never treat an equal pointer or active instruction prefix as a hit.
    for (unsigned offset : {4, 31, 32, 63, 64, 1023, 2047, 4095})
        for (u32 value : {1U, 256U, 0x01000000U}) {
            mutations.push_back([&, offset, value] { setup.UpdateProgramCode(offset, value); });
            mutations.push_back([&, offset, value] { setup.UpdateSwizzleData(offset, value); });
        }
    std::vector<std::shared_ptr<const Metal::CompiledMeshVertexProgram>> programs;
    for (const auto& mutate : mutations) {
        restore(); mutate();
        const auto program = cache.MeshVertex(regs, setup);
        Check(program != first && program->Library(), "Exact mesh key mutation reused another program");
        Check(cache.CachedVertexCount() == programs.size() + 2, "Exact mesh key bytes collapsed");
        Check(cache.MeshVertex(regs, setup) == program, "Recent mesh key was not a stable hit");
        programs.push_back(program);
        restore();
        Check(cache.MeshVertex(regs, setup) == first, "Mutable mesh view changed an owned key");
        Check(cache.MeshVertex(regs, setup) == first, "Restored recent mesh key missed");
    }
    for (std::size_t i = mutations.size(); i-- > 0;) {
        restore(); mutations[i]();
        Check(cache.MeshVertex(regs, setup) == programs[i], "Reverse mesh lookup returned another key");
        Check(cache.MeshVertex(regs, setup) == programs[i], "Reverse recent mesh lookup changed identity");
    }
    Check(cache.CachedVertexCount() == 60 && first->shader.source == first_source,
          "Mesh cache changed its immutable program or key count");
    cache.Clear();
    Check(!cache.CachedVertexCount() && first->Library(), "Mesh Clear destroyed a caller-held program");
    restore();
    const auto replacement = cache.MeshVertex(regs, setup);
    Check(replacement != first && cache.CachedVertexCount() == 1,
          "Mesh Clear retained a stale recent-node pointer");
    Check(cache.MeshVertex(regs, setup) == replacement && replacement->shader.source == first_source,
          "Repopulated mesh recent key changed the shader");
    std::cout << "60 exact mesh keys: all metadata and code/swizzle boundary mutations, recent/reverse hits, "
                 "owned source and Clear lifetime passed\n";
}

void CheckOwnedMeshVertexCache(Metal::Device& device,Metal::Scheduler& scheduler) {
    using Access=Metal::InputReferenceTestAccess;
    static_assert(!std::is_default_constructible_v<Metal::NativeProgramSnapshot>);
    static_assert(!std::is_constructible_v<Metal::NativeProgramSnapshot,std::shared_ptr<const Metal::ShaderMachineProgram>>);
    Metal::ShaderRuntime cache{device};Metal::ShaderMachineRuntime machine{device,scheduler};
    Pica::RegsInternal regs{};Pica::ShaderSetup setup;
    Metal::Test::InitializeFixtureUniforms(setup);Metal::Test::MovVertex(regs,setup);
    Metal::ShaderMachineProgram source{setup.GetProgramCode(),setup.GetSwizzleData(),regs.vs.main_offset,regs.vs.output_mask,false};
    const auto original=source;const auto original_regs=regs;
    Metal::GeometryEmissionConfig config{};config.attribute_count=regs.rasterizer.vs_output_total;config.honor_winding=0;
    for(unsigned i=0;i<7;++i)config.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
    const auto snapshot=Access::Program(machine,source,{},config);
    const auto first=Access::OwnedMesh(cache,regs,snapshot);
    const auto generated=Metal::GeneratePicaVertexMeshShader(regs,setup);
    Check(first->shader.source==generated.source && first->shader.spirv==generated.spirv &&
          first->shader.entry_point==generated.entry_point && first->shader.resources.size()==generated.resources.size(),
          "Owned first-generation shader differs from ordinary current generation");
    const auto content=cache.MeshStats().content_lookups;
    for(unsigned i=0;i<100;++i)Check(Access::OwnedMesh(cache,regs,snapshot)==first,"Owned repeated lookup changed program");
    Check(cache.MeshStats().content_lookups==content && cache.MeshStats().owned_hits==100,
          "Owned repeated lookup repeated full bank comparison");
    source.code[0]=0;source.code[4095]^=0xabcdef01;source.swizzle[0]^=1;source.entry=3;source.output_mask=1;source.allow_emission=true;
    Check(Access::OwnedMesh(cache,regs,snapshot)==first,"Owned program borrowed mutable source storage");
    for(unsigned field=0;field<9;++field) {
        regs=original_regs;
        if(field==0)regs.rasterizer.vs_output_total.Assign(2);
        else if(field==1)regs.lighting.disable.Assign(1);
        else regs.rasterizer.vs_output_attributes[field-2].raw^=1;
        const auto other=Access::OwnedMesh(cache,regs,snapshot);
        Check(other!=first && Access::OwnedMesh(cache,regs,snapshot)==other,"Owned metadata key collapsed");
    }
    regs=original_regs;
    for(unsigned field=0;field<2;++field) {
        if(field==0)regs.vs.main_offset.Assign(1);else {regs=original_regs;regs.vs.output_mask.Assign(1);}
        bool rejected=false;try{(void)Access::OwnedMesh(cache,regs,snapshot);}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected,"Owned snapshot accepted mismatched entry/output metadata");
    }
    regs=original_regs;
    for(unsigned bank=0;bank<2;++bank) {
        auto changed=original;if(bank)changed.swizzle[4095]^=0x01020304;else changed.code[4095]^=0xabcde;
        const auto other=Access::OwnedMesh(cache,regs,Access::Program(machine,changed,{},config));
        Check(other!=first,"Distinct immutable source bank aliased an owned mesh program");
        Check(Access::OwnedMesh(cache,regs,snapshot)==first,"New owner replaced original immutable program");
    }
    {
        auto local=Access::SeparateOwner(device,scheduler,machine,original,config);
        Check(Access::OwnedMesh(cache,regs,local)==first,"Separate owner with identical full bytes changed program identity");
    }
    Check(cache.MeshStats().expired_owners>0,"Owned index kept a retired source object alive");
    auto orphan=[&] {Metal::ShaderMachineRuntime temporary{device,scheduler};return Access::Program(temporary,original,{},config);}();
    Check(Access::OwnedMesh(cache,regs,orphan)==first,"Sealed snapshot failed after machine destruction");
    for(unsigned i=0;i<70;++i) {
        auto changed=original;changed.code[4095]=i+10000;
        (void)Access::OwnedMesh(cache,regs,Access::Program(machine,changed,{},config));
        Check(cache.MeshStats().owned_entries<=64,"Owned index exceeded its retention bound");
    }
    const auto before=cache.MeshStats().content_lookups;
    Check(Access::OwnedMesh(cache,regs,snapshot)==first && cache.MeshStats().content_lookups==before+1,
          "Evicted owner skipped full re-registration or lost valid program");
    std::array<std::thread,8> workers;std::atomic_bool mismatch=false;
    for(auto& worker:workers)worker=std::thread([&] {
        for(unsigned i=0;i<64;++i)if(Access::OwnedMesh(cache,regs,snapshot)!=first)mismatch=true;
    });
    for(auto& worker:workers)worker.join();Check(!mismatch,"Concurrent owned hits published another program");
    if(std::getenv("AZAHAR_METAL4_BENCH_OWNED_MESH")) {
        constexpr unsigned iterations=30000;
        for(unsigned pair=0;pair<12;++pair) {
            const auto measure=[&](bool owned) {
                const auto start=std::chrono::steady_clock::now();
                for(unsigned i=0;i<iterations;++i) {
                    const auto value=owned?Access::OwnedMesh(cache,regs,snapshot):cache.MeshVertex(regs,setup);
                    if(value!=first)throw std::runtime_error("Benchmark changed complete shader identity");
                }
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            };
            double ordinary{},owned{};
            if(pair&1){owned=measure(true);ordinary=measure(false);}else{ordinary=measure(false);owned=measure(true);}
            std::cout<<"OWNED_MESH_BENCH "<<pair<<' '<<iterations<<' '<<ordinary<<' '<<owned<<'\n';
        }
    }
    cache.Clear();Check(!cache.MeshStats().owned_entries && first->Library(),"Clear lost retained program or owned index state");
    const auto replacement=Access::OwnedMesh(cache,regs,snapshot);
    Check(replacement!=first && replacement->shader.source==first->shader.source,
          "Clear reused a stale owner node or changed regenerated code");
    std::cout<<"Owned mesh source: sealed construction, exact first generation,100 repeated hits without bank checks,all metadata/bank changes,foreign/moved rejection,retired/orphan owners,70 bounded replacements,8concurrent callers and Clear lifetime passed\n";
}

void CheckDeferredInputMaterializer(Metal::Device& device) {
    Metal::Scheduler queue{device}; Metal::TransferRuntime transfer{device,queue};
    Metal::VertexInputState state{device};
    constexpr unsigned count=7;
    std::array<std::uint32_t,count*16*4> expected{};
    auto source=device.CreateBuffer(count*16,true);
    for(unsigned v=0;v<count;++v) for(unsigned c=0;c<4;++c) {
        const std::uint32_t bits=c==0 ? 0x80000000u:c==1 ? 0x7fc12345u:0x3f000000u+v*17+c;
        static_cast<std::uint32_t*>(source.contents)[v*4+c]=bits;
        expected[(v*16)*4+c]=bits;
    }
    Metal::PicaVertexLayout layout{};layout.mapped_attribute_count=16;
    for(unsigned n=0;n<16;++n) {
        layout.register_map[n]=n;
        layout.attribute_inputs[n].mode=Metal::PicaAttributeMode::Fixed;
        for(unsigned c=0;c<4;++c) {
            const std::uint32_t bits=c==0 ? 0x80000000u:c==1 ? 0x7fc12000u+n:0x3f000000u+n*16+c;
            std::memcpy(&layout.fixed_attributes[n][c],&bits,4);
            if(n)for(unsigned v=0;v<count;++v)expected[(v*16+n)*4+c]=bits;
        }
    }
    layout.attribute_inputs[0].mode=Metal::PicaAttributeMode::Buffer;
    layout.attribute_inputs[0].fetch={0,0,16,4,Metal::VertexAttributeFormat::Float,0};
    Metal::BufferSlice output;
    Metal::Test::CompilerGate gate{device};
    const auto compiled=device.DiskCacheStats().compiled_pipelines;
    {
        auto runtime=std::make_unique<Metal::InputReferenceRuntime>(device,queue);
        const std::array<Metal::BufferSlice,1> inputs{{{source,0,source.length,source.contents,{}}}};
        auto prepared=runtime->Prepare(runtime->Current(state),inputs,layout,count,count,11,11,{},0);
        output=runtime->Materialize(prepared.current_registers,count);
        prepared={};source=nil;layout={};
        runtime.reset(); // Neither compiler nor encoder jobs may capture this owner.
    }
    const auto held=queue.WorkerStats();
    const bool no_compile=device.DiskCacheStats().compiled_pipelines==compiled;
    const bool timely=gate.Open();
    Check(timely && no_compile && held.queued_dispatches==1 && !held.encoded_dispatches && queue.CurrentTick()==1,
          "Lazy input materializer compiled/waited on its owner, lost its job or ran before its pipeline");
    const auto read=device.CreateBuffer(sizeof(expected)+32,true);
    std::memset(read.contents,0xA7,read.length);
    transfer.CopyBuffer(output.buffer,output.offset,read,16,sizeof(expected));queue.Finish();
    Check(!std::memcmp(static_cast<const std::byte*>(read.contents)+16,expected.data(),sizeof(expected)),
          "Deferred materializer lost source/runtime lifetime or changed raw fixed/buffer bits");
    for(unsigned i=0;i<16;++i)Check(static_cast<const std::byte*>(read.contents)[i]==std::byte{0xA7} &&
        static_cast<const std::byte*>(read.contents)[read.length-16+i]==std::byte{0xA7},"Deferred materializer changed guards");
    std::cout<<"Compiler-gated lazy input materializer: constructor/dispatch/runtime destruction return, no owner native compilation, released sources/layout, all448 words including NaN/signed-zero bits and guards passed\n";
}

int main() { @autoreleasepool { try {
    Metal::Device device{{}, false}; Metal::Scheduler scheduler{device};
    CheckDeferredInputMaterializer(device);
    CheckExactMeshVertexCache(device);
    CheckOwnedMeshVertexCache(device,scheduler);
    Metal::TransferRuntime transfer{device, scheduler}; Metal::VertexFetchRuntime fetch{device, scheduler};
    Metal::InputReferenceRuntime references{device, scheduler}; Metal::GeometryRuntime geometry{device, scheduler};
    Metal::ShaderRuntime shaders{device}; Metal::ShaderMachineRuntime machine{device, scheduler};
    Metal::DrawRuntime draws{device, scheduler, geometry};
    Pica::RegsInternal regs{}; Pica::ShaderSetup setup;
    Metal::Test::InitializeFixtureUniforms(setup); Metal::Test::MovVertex(regs, setup);
    setup.UpdateProgramCode(3, Metal::Test::Arithmetic(nihstro::OpCode::Id::ADD, 16, 16, 15));
    setup.UpdateProgramCode(4, std::uint32_t(nihstro::OpCode::Id::END) << 26);
    Metal::ShaderMachineProgram program{}; program.allow_emission = false; program.output_mask = 7;
    program.code = setup.GetProgramCode(); program.swizzle = setup.GetSwizzleData();
    Metal::ShaderMachineUniforms values{};
    Metal::GeometryEmissionConfig config{}; config.attribute_count = 3;
    config.semantics = {0x03020100, 0x07060504, 0x0b0a0908}; config.honor_winding = 0;
    Check(machine.CanCaptureNativeReset(program, values, config), "Reference fixture was not selected native");
    const auto capture = shaders.Vertex(regs, setup, true);
    constexpr std::string_view fragment_source=R"(
#version 450
layout(location=1) in vec4 primary_color;
layout(location=6) in vec4 normquat;
layout(location=0) out vec4 color;
void main(){color=abs(normquat)*0.25+primary_color*0.5;}
)";
    auto fragment_shader = Metal::TranslateGuestGLSL(fragment_source, Metal::ShaderStage::Fragment, Metal::RasterizerShaderBindings());
    auto fragment_library = device.CompileLibrary(fragment_shader.source, "Reference mesh varying pixels");
    Metal::CompiledFragmentShader fragment{{std::move(fragment_shader), {}}, fragment_library};
    std::array<Metal::CompiledFragmentShader,3> ordered_fragments;
    for(unsigned variant=0;variant<ordered_fragments.size();++variant) {
        Metal::FragmentOperations operations{};
        if(variant==0) {
            operations.rgb={true,0,6,7}; operations.alpha={true,0,1,7};
        }else {operations.logic_enabled=true;operations.logic_operation=variant==1 ? 11:7;}
        auto translated=Metal::TranslateRasterizerFragmentGLSL(fragment_source,operations);
        Check(translated.uses_framebuffer_fetch,"Ordered fixture lost destination dependency");
        auto library=device.CompileLibrary(translated.source,"Ordered mesh destination pixel fixture");
        ordered_fragments[variant]={{std::move(translated),operations},library};
    }
    auto ubo = device.CreateBuffer(1616, true), clip = device.CreateBuffer(32, true);
    std::memset(ubo.contents, 0, ubo.length); std::memset(clip.contents, 0, clip.length);
    const std::array<Metal::BufferBinding, 2> uniforms{{{ubo, 0, 0}, {clip, 0, 1}}};
    constexpr unsigned count = 70, base = 8, width = 128, height = 96;
    auto input = device.CreateBuffer(count * 256, true); auto* data = static_cast<Metal::VertexRegisters*>(input.contents);
    for (unsigned v = 0; v < count; ++v) {
        data[v] = {};
        data[v][0] = {-.75f + float(v % 3) * .75f, -.75f + float((v / 3) % 3) * .75f, .25f, 1};
        data[v][1] = {v & 1 ? -.5f : .5f, .25f, 0, 1}; data[v][2] = {.125f, .5f, .75f, 1};
        data[v][15] = {.125f, .25f, -.5f, 1};
    }
    const std::array<Metal::BufferSlice, 1> sources{{{input, 0, input.length, input.contents, std::make_shared<std::byte>()}}};
    Metal::PicaVertexLayout layout{}; layout.mapped_attribute_count = 16;
    for (unsigned a = 0; a < 16; ++a) {
        layout.register_map[a] = a; layout.attribute_inputs[a].mode = Metal::PicaAttributeMode::Buffer;
        layout.attribute_inputs[a].fetch = {0, a * 16, 256, 4, Metal::VertexAttributeFormat::Float, a};
    }
    auto* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    auto texture = device.CreateTexture(td);
    const MTLViewport viewport{0, 0, width, height, 0, 1}; const MTLScissorRect scissor{0, 0, width, height};
    Metal::PicaDrawState draw_state{}; draw_state.use_color = true;
    unsigned cases = 0, ordered_chains = 0, argument_checks = 0, multi_chunk_checks = 0;
    constexpr std::array index_widths{0u, 1u, 2u, 4u, 2u};
    for (unsigned variant = 0; variant < index_widths.size(); ++variant) for (bool fix : std::array{false, true}) {
        const bool large = variant == 4;
        if (large && fix) continue;
        const unsigned index_width = index_widths[variant];
        const unsigned elements = large ? 98307 : index_width ? 138 : 69;
        auto index = device.CreateBuffer(std::max(1u, elements * index_width), true);
        for (unsigned e = 0; e < elements && index_width; ++e) {
            const unsigned id = base + e % count;
            std::memcpy(static_cast<std::byte*>(index.contents) + e * index_width, &id, index_width);
        }
        const auto indices = index_width ? Metal::BufferSlice{index, 0, index.length, index.contents, std::make_shared<std::byte>()} : Metal::BufferSlice{};
        const auto index_bytes = index_width ? std::span<const std::byte>{static_cast<const std::byte*>(index.contents), elements * index_width} : std::span<const std::byte>{};
        Metal::VertexInputState expected_input{device}, current_input{device};
        const auto prepared = references.Prepare(references.Current(current_input), sources, layout, count, elements, base, base, index_bytes, index_width);
        auto mesh = Metal::InputReferenceTestAccess::Mesh(device, references, shaders, machine, regs, setup, prepared, indices, count, elements, base, index_width, fix);
        const auto published = Metal::InputReferenceTestAccess::PrepareNative(references, current_input,
            sources, layout, count, elements, base, indices, index_width);
        Metal::InputReferenceTestAccess::Adopt(mesh, published, device, count);
        if (!index_width && !fix) {
            Metal::ShaderMachineState rejected_state{device};
            const auto unchanged = machine.SaveState(rejected_state);
            bool rejected{};
            try { Metal::InputReferenceTestAccess::Defer(machine, rejected_state, program, values,
                config, published.snapshot, count * 2, elements, base, 0); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected && machine.SaveState(rejected_state) == unchanged,
                  "Certified reference extent was incorrectly reused for a larger count");
        }
        const auto canonical = fetch.FetchPica(expected_input, sources, layout, count, elements, base, base, indices, index_width);
        Metal::ShaderMachineState expected_vs{device}, current_vs{device};
        const auto expected_batch = machine.RunVertexBatchAndCapture(expected_vs, program, values, canonical, config, count, elements, base, base, indices, index_width);
        const auto expected_state = machine.SaveState(expected_vs);
        Metal::PrimitiveStream primitive{device};
        const auto triangles = geometry.Append(primitive, expected_batch.captured, elements, Metal::TriangleTopology::List, elements, 0, 0, {}, 0, fix);
        const auto history = geometry.SaveStream(primitive);
        Check(std::all_of(history.history.begin(), history.history.end(), [](auto byte) { return byte == std::byte{}; }), "Complete-list history is not canonical zero");
        const auto render = [&](bool native) {
            transfer.ClearColor(texture, MTLClearColorMake(.0625, .125, .25, 1));
            auto* pass = [[MTL4RenderPassDescriptor alloc] init]; pass.colorAttachments[0].texture = texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            if (native) {
                const auto before = scheduler.ArgumentUpdateCount();
                const auto snapshot=scheduler.SnapshotRenderPass(pass);
                draws.DrawMesh(draw_state, fragment, *snapshot, viewport, scissor, mesh, uniforms, {}, {});
                const auto actual = scheduler.ArgumentUpdateCount() - before;
                const auto expected = (std::uint64_t(elements / 3) + 32767) / 32768;
                if (actual != expected)
                    std::cerr << "Mesh argument-table publications=" << actual << ", expected=" << expected << '\n';
                Check(actual == expected, "Mesh draw republished an unchanged first-chunk argument table");
                ++argument_checks;
                if (expected > 1) ++multi_chunk_checks;
            }
            else draws.Draw(draw_state, fragment, pass, viewport, scissor, triangles, {}, {}, {});
            return transfer.Readback({texture, 0, 0, MTLRegionMake2D(0, 0, width, height)}, width * 4);
        };
        Check(render(false) == render(true), "Production mesh/precise full pixel image differs");
        if (variant == 0 && !fix) {
            const auto expected_pixels = render(false);
            transfer.ClearColor(texture, MTLClearColorMake(.0625, .125, .25, 1));
            scheduler.Finish();
            Metal::Test::CompilerGate gate{device};
            const auto original = Metal::InputReferenceTestAccess::MeshProgram(mesh);
            auto pending_mesh = std::make_shared<Metal::CompiledMeshVertexProgram>(*original);
            const auto mesh_source = std::make_shared<const Metal::TranslatedShader>(original->shader);
            pending_mesh->library = nil;
            pending_mesh->pending_library = device.DeferCompilation([&device, mesh_source] {
                return device.CompileLibrary(mesh_source->source, "Gated complete mesh draw");
            });
            auto pending_fragment = fragment;
            pending_fragment.library = nil;
            const auto fragment_source = std::make_shared<const Metal::TranslatedShader>(fragment.program.shader);
            pending_fragment.pending_library = device.DeferCompilation([&device, fragment_source] {
                return device.CompileLibrary(fragment_source->source, "Gated complete mesh fragment");
            });
            Metal::InputReferenceTestAccess::MeshProgram(mesh, pending_mesh);
            auto* pass = [[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture = texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            const auto snapshot = scheduler.SnapshotRenderPass(pass);
            draws.DrawMesh(draw_state, pending_fragment, *snapshot, viewport, scissor, mesh, uniforms, {}, {});
            const bool both_pending = pending_mesh->pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout &&
                pending_fragment.pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout;
            const bool timely = gate.Open();
            Check(timely && both_pending, "Full DrawMesh waited for a native mesh/fragment library on the owner");
            std::weak_ptr<const Metal::CompiledMeshVertexProgram> owner = pending_mesh;
            Metal::InputReferenceTestAccess::MeshProgram(mesh, original);
            pending_mesh.reset();
            pending_fragment = {};
            Check(owner.expired(), "Full deferred mesh draw retained a mutable metadata owner");
            Check(expected_pixels == transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)}, width*4),
                "Full deferred mesh draw lost pixels after input metadata release");
            std::cout << "Full DrawMesh: both libraries compiler-gated, owner progress, released metadata and complete128x96 GPU pixels passed\n";
        }
        // Compare accumulated overlapping draws with the previous per-draw
        // render-pass boundary. Each final pixel depends on earlier draw calls;
        // mixed pipelines and changing scissors must retain primitive order.
        for(unsigned pattern=0;pattern<3 && !large;++pattern) {
            const auto render_ordered=[&](unsigned mode) {
                transfer.ClearColor(texture, MTLClearColorMake(.0625, .125, .25, 1));
                auto* pass=[[MTL4RenderPassDescriptor alloc] init];
                pass.colorAttachments[0].texture=texture;
                pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
                pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                const auto snapshot=scheduler.SnapshotRenderPass(pass);
                const auto before = scheduler.EncoderCount();
                for(unsigned step=0;step<17;++step) {
                    const auto* selected=pattern==0 ? &ordered_fragments[0]:&ordered_fragments[(step+pattern)%3];
                    if(pattern==2 && step%5==1) selected=&fragment;
                    auto clip=scissor;
                    if(pattern==2) {clip.x=(step%3)*8;clip.y=(step%4)*4;clip.width-=clip.x;clip.height-=clip.y;}
                    if(mode == 1) {
                        draws.DrawMesh(draw_state,*selected,*snapshot,viewport,clip,mesh,uniforms,{},{});
                    } else {
                        draws.Draw(draw_state,*selected,pass,viewport,clip,triangles,{},{},{});
                        if (mode == 0) scheduler.EndEncoding();
                    }
                }
                if (mode) Check(scheduler.EncoderCount() == before + 1,
                    "Queued compatible draws failed to share one render encoder");
                const auto pixels=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
                const auto worker=scheduler.WorkerStats();
                Check(worker.separate_thread && worker.encoded_draws==worker.queued_draws,
                    "Native draw work did not retire on its separate encoding thread");
                return pixels;
            };
            const auto separated = render_ordered(0);
            Check(separated==render_ordered(1),"Multi-draw raster-order pixels differ from separated passes");
            Check(separated==render_ordered(2),"Ordinary shared-pass pixels differ from separated passes");
            ++ordered_chains;
        }
        Check(Metal::InputReferenceTestAccess::DummyUntouched(mesh), "Independent mesh wrote its dummy state buffer");
        Metal::InputReferenceTestAccess::Defer(machine, current_vs, program, values, config,
            published.snapshot, count, elements, base, index_width);
        Metal::InputReferenceTestAccess::Commit(current_input, prepared);
        Check(fetch.SaveState(current_input).history == fetch.SaveState(expected_input).history, "Production complete512-byte reference history differs");
        Check(machine.SaveState(current_vs) == expected_state, "Production deferred complete1824-byte state differs");
        Metal::InputReferenceTestAccess::Poison(transfer, primitive);
        Metal::InputReferenceTestAccess::KnownZero(primitive);
        Check(geometry.SaveStream(primitive).history == history.history, "Known-zero primitive state differs");
        geometry.LoadStream(primitive, history);
        Check(geometry.SaveStream(primitive).history == history.history, "Known-zero primitive load differs");
        auto* continued = [[MTL4RenderPassDescriptor alloc] init];
        continued.colorAttachments[0].texture = texture; continued.colorAttachments[0].loadAction = MTLLoadActionLoad;
        continued.colorAttachments[0].storeAction = MTLStoreActionStore;
        auto first_encoder = scheduler.BeginCompatibleRender(continued);
        Check(scheduler.BeginCompatibleRender(continued) == first_encoder, "Compatible mesh draw did not retain its render encoder");
        draws.DrawMesh(draw_state, fragment, continued, viewport, scissor, mesh, uniforms, {}, {});
        auto after_worker=scheduler.BeginCompatibleRender(continued);
        Check(after_worker!=first_encoder,"Direct encoding failed to drain/close a queued render pass");
        first_encoder=after_worker;
        auto replacement = device.CreateTexture(td); continued.colorAttachments[0].texture = replacement;
        auto changed_encoder = scheduler.BeginCompatibleRender(continued);
        Check(changed_encoder != first_encoder, "Attachment change reused the old render encoder");
        scheduler.EndEncoding();
        continued.colorAttachments[0].loadAction = MTLLoadActionClear;
        auto cleared = scheduler.BeginCompatibleRender(continued);
        Check(scheduler.BeginCompatibleRender(continued) != cleared, "Repeated clear incorrectly retained its render encoder");
        scheduler.EndEncoding();
        continued.colorAttachments[0].loadAction = MTLLoadActionLoad;
        auto before_compute = scheduler.BeginCompatibleRender(continued);
        transfer.FillBuffer(ubo, NSMakeRange(0, 16), 0);
        Check(scheduler.BeginCompatibleRender(continued) != before_compute, "Compute transfer did not break render-pass reuse");
        scheduler.EndEncoding();
        ++cases;
    }

    // A snapshot pinning a preceding GPU input-history allocation must survive
    // both a precise fetch overwrite and a load into the original state object.
    for (unsigned replace = 0; replace < 2; ++replace) {
        Metal::VertexInputSnapshot seeded{}; seeded.needs_reset = false;
        std::array<Metal::VertexRegisters, 2> seeded_values{};
        seeded_values[1] = data[0]; std::memcpy(seeded.history.data(), seeded_values.data(), 512);
        Metal::VertexInputState state{device}; fetch.LoadState(state, seeded);
        Metal::PicaVertexLayout retained{}; retained.mapped_attribute_count = 0;
        const auto prepared = references.Prepare(references.Current(state), {}, retained, 1, 3, 0, 0, {}, 0);
        const auto snapshot = Metal::InputReferenceTestAccess::Snapshot(prepared.current_registers, {});
        Metal::ShaderMachineState unit{device}, golden{device};
        auto canonical = references.Materialize(prepared.current_registers, 3);
        (void)machine.RunVertexBatch(golden, program, values, canonical, 3, 3, 0, 0);
        const auto wanted = machine.SaveState(golden);
        Metal::InputReferenceTestAccess::Defer(machine, unit, program, values, config, snapshot, 3, 3, 0, 0);
        if (replace) { auto replacement = seeded; std::fill(replacement.history.begin(), replacement.history.end(), std::byte{0}); fetch.LoadState(state, replacement); }
        else { (void)fetch.FetchPica(state, sources, layout, count, 69, base, base); }
        Check(machine.SaveState(unit) == wanted, "Pinned GPU history changed after precise write/load");
        ++cases;
    }
    Metal::InputReferenceTestAccess::CheckSelection(device,scheduler,machine,references,data[0],program,config);
    if (std::getenv("AZAHAR_METAL4_BENCH_NATIVE_SELECTION")) {
        std::array<Metal::InputReference,16> fixed{};
        for(unsigned n=0;n<16;++n) fixed[n]=Metal::InputReference::Constant(data[0][n]);
        auto snapshot=Metal::InputReferenceTestAccess::Snapshot(fixed,{});
        Metal::ShaderMachineState unit{device};
        constexpr unsigned iterations=10000;
        for(unsigned round=0;round<7;++round) {
            const auto start=std::chrono::steady_clock::now();
            for(unsigned n=0;n<iterations;++n) {
                unit.Reset();
                Metal::InputReferenceTestAccess::Defer(machine,unit,program,values,config,snapshot,3,3,0,0);
            }
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::cout<<"Native selection host benchmark: round="<<round<<", iterations="<<iterations<<", ms="<<ms<<'\n';
        }
        auto canonical=references.Materialize(fixed,3);
        Metal::ShaderMachineState golden{device};
        (void)machine.RunVertexBatch(golden,program,values,canonical,3,3,0,0);
        Check(machine.SaveState(unit)==machine.SaveState(golden),"Timed native selection changed full GPU state");
    }
    // Compare the complete published512-byte input bank with the existing
    // independent GPU FIFO, including eviction, aliases, invalid indices and
    // retained sources. This exercises the production CPU membership lookup.
    unsigned history_cases=0;
    for(unsigned index_width:std::array{0u,1u,2u,4u}) for(unsigned mode=0;mode<5;++mode) {
        constexpr unsigned vertices=70,vertex_base=8;
        Metal::VertexInputState golden{device},current{device};
        Metal::PicaVertexLayout history_layout{};history_layout.mapped_attribute_count=16;
        history_layout.fixed_attributes=Metal::DefaultVertexRegisters();
        std::array<Metal::BufferSlice,12> history_inputs{};
        const auto format=Metal::VertexAttributeFormat(mode%4);
        const unsigned bytes=mode%4==3 ? 4:mode%4==2 ? 2:1,stride=4*bytes+1;
        auto history_input=device.CreateBuffer(vertices*stride,true);std::memset(history_input.contents,0x5a,history_input.length);
        for(unsigned v=0;v<vertices;++v)for(unsigned c=0;c<4;++c) {
            const std::uint32_t bits=format==Metal::VertexAttributeFormat::Float ?
                (c==0 ? 0x80000000u:c==1 ? 0x7fc12345u:c==2 ? 0x3f800000u+v:0):0x81234567u+v+c;
            std::memcpy(static_cast<std::byte*>(history_input.contents)+v*stride+1+c*bytes,&bits,bytes);
        }
        for(auto& source:history_inputs)source={history_input,0,history_input.length,history_input.contents,std::make_shared<std::byte>()};
        for(unsigned a=0;a<16;++a) {
            history_layout.register_map[a]=a;history_layout.attribute_inputs[a].mode=Metal::PicaAttributeMode::Buffer;
            history_layout.attribute_inputs[a].fetch={a%12,1,stride,1+a%4,format,a};
        }
        if(mode==4) {
            history_layout.mapped_attribute_count=7;history_layout.register_map[0]=3;history_layout.register_map[1]=3;
            history_layout.register_map[2]=4;history_layout.register_map[3]=1;
            history_layout.attribute_inputs[1].mode=Metal::PicaAttributeMode::Retained;
            history_layout.attribute_inputs[2].mode=Metal::PicaAttributeMode::Fixed;
            history_layout.fixed_attributes[2]={-.0f,.25f,-.5f,1};
        }
        for(unsigned batch=0;batch<9;++batch) {
            if(batch==1) {
                Metal::VertexInputSnapshot seed{};seed.needs_reset=false;
                for(unsigned b=0;b<512;++b)seed.history[b]=std::byte((b*17)&255);
                fetch.LoadState(golden,seed);fetch.LoadState(current,seed);
            }
            if(batch==2) {golden.Reset();current.Reset();}
            const unsigned elements=index_width ? (batch>=3 ? 137:7):vertices;
            std::vector<std::uint32_t> ids(elements);
            for(unsigned e=0;e<elements;++e)ids[e]=vertex_base+e%70;
            if(index_width && batch==3) {
                for(unsigned e=0;e<65;++e)ids[e]=vertex_base+e;
                for(unsigned e=65;e<elements;++e)ids[e]=vertex_base+(e%2==0 ? 0:64);
            }
            if(index_width && batch==4)for(unsigned e=0;e<elements;++e)ids[e]=e%3 ? vertex_base+e%70:UINT32_MAX;
            if(index_width && batch==5)std::fill(ids.begin(),ids.end(),0);
            if(batch==6)for(unsigned a=0;a<16;++a)if(a%2==0)history_layout.attribute_inputs[a].mode=Metal::PicaAttributeMode::Retained;
            if(batch==7) {history_layout.register_map[0]=15;history_layout.register_map[1]=15;history_layout.mapped_attribute_count=2;}
            auto index=device.CreateBuffer(std::max(4u,elements*index_width),true);std::memset(index.contents,0,index.length);
            for(unsigned e=0;e<elements && index_width;++e)std::memcpy(static_cast<std::byte*>(index.contents)+e*index_width,&ids[e],index_width);
            const auto indices=index_width ? Metal::BufferSlice{index,0,elements*index_width,index.contents,{}}:Metal::BufferSlice{};
            const auto prepared=Metal::InputReferenceTestAccess::PrepareNative(references,current,history_inputs,
                history_layout,vertices,elements,vertex_base,indices,index_width);
            (void)fetch.FetchPica(golden,history_inputs,history_layout,vertices,elements,vertex_base,vertex_base,indices,index_width);
            Metal::InputReferenceTestAccess::Commit(current,prepared);
            const auto actual_history=fetch.SaveState(current),expected_history=fetch.SaveState(golden);
            Check(actual_history.history==expected_history.history && actual_history.needs_reset==expected_history.needs_reset,
                "Complete hashed-FIFO512-byte input history differs from GPU fetch");
            ++history_cases;
        }
    }
    // Compare the bounded membership path and both sides of its range boundary
    // against the independent GPU fetch, including every retained history byte.
    unsigned bounded_history_cases = 0;
    for (unsigned vertices : {1u, 4u, 8u, 63u, 64u, 65u, 128u, 256u, 257u, 512u})
    for (unsigned index_width : {1u, 2u, 4u})
    for (bool inherited : {false, true})
    for (unsigned pattern = 0; pattern < 2; ++pattern) {
        const unsigned vertex_base = index_width == 4 ? UINT32_MAX - vertices + 1 :
                                     index_width == 2 ? 2048 : 0;
        constexpr unsigned elements = 193;
        Metal::VertexInputState golden{device}, current{device};
        if (inherited) {
            Metal::VertexInputSnapshot seed{}; seed.needs_reset = false;
            for (unsigned b = 0; b < seed.history.size(); ++b) seed.history[b] = std::byte((b * 17) & 255);
            fetch.LoadState(golden, seed); fetch.LoadState(current, seed);
        }
        auto history_layout = layout;
        history_layout.mapped_attribute_count = 7;
        history_layout.register_map[0] = 3; history_layout.register_map[1] = 3;
        history_layout.register_map[2] = 4; history_layout.register_map[3] = 1;
        history_layout.attribute_inputs[1].mode = Metal::PicaAttributeMode::Retained;
        history_layout.attribute_inputs[2].mode = Metal::PicaAttributeMode::Fixed;
        history_layout.fixed_attributes[2] = {-.0f, .25f, -.5f, 1};
        auto buffer = device.CreateBuffer(vertices * sizeof(Metal::VertexRegisters), true);
        auto* registers = static_cast<Metal::VertexRegisters*>(buffer.contents);
        for (unsigned v = 0; v < vertices; ++v) {
            for (unsigned a = 0; a < 16; ++a)
                registers[v][a] = {float(v), float(a), -.0f, 1};
        }
        const std::array<Metal::BufferSlice, 1> inputs{{{buffer, 0, buffer.length, buffer.contents, std::make_shared<std::byte>()}}};
        auto index = device.CreateBuffer(elements * index_width, true);
        for (unsigned e = 0; e < elements; ++e) {
            // Pattern 1 fills 64 slots, hits the oldest ID, evicts it, then
            // revisits it. Refresh-on-hit or forgetting eviction must fail.
            const unsigned local = pattern == 0 ? (e * 7 + e / 9) % vertices :
                (e < 64 ? e % vertices : e == 64 ? 0 : e == 65 ? 64 % vertices :
                 e == 66 ? 0 : e % 2 == 0 ? 0 : 64 % vertices);
            const unsigned id = vertex_base + local;
            std::memcpy(static_cast<std::byte*>(index.contents) + e * index_width, &id, index_width);
        }
        const Metal::BufferSlice indices{index, 0, index.length, index.contents, {}};
        const auto prepared = Metal::InputReferenceTestAccess::PrepareNative(references, current, inputs,
            history_layout, vertices, elements, vertex_base, indices, index_width);
        (void)fetch.FetchPica(golden, inputs, history_layout, vertices, elements, vertex_base, vertex_base, indices, index_width);
        Metal::InputReferenceTestAccess::Commit(current, prepared);
        const auto actual = fetch.SaveState(current), expected = fetch.SaveState(golden);
        Check(actual.history == expected.history && actual.needs_reset == expected.needs_reset,
              "Bounded FIFO complete512-byte GPU history differs");
        ++bounded_history_cases;
    }
    std::cout << "Bounded FIFO histories: " << bounded_history_cases
              << " complete512-byte GPU comparisons;1/2/4-byte IDs, ranges1..512,"
                 "63/64/65 and256/257 boundaries, UINT_MAX, reset/inherited, aliases/fixed/retained, eviction/no-hit-refresh\n";
    {
        Metal::VertexInputState state{device};
        Metal::PicaVertexLayout retained{};
        for (auto& input : retained.attribute_inputs) input.mode = Metal::PicaAttributeMode::Retained;
        auto malformed = std::make_shared<Metal::InputProvenance>();
        malformed->needs_reset = false;
        Metal::PreparedInputReferences setup{}; setup.next = malformed;
        Metal::InputReferenceTestAccess::Commit(state, setup);
        unsigned rejected{};
        for (unsigned test = 0; test < 3; ++test) {
            malformed->registers[0] = {};
            if (test == 0) malformed->registers[0].gpu.reserved0 = 1;
            if (test == 1) malformed->registers[0].gpu.kind = 2;
            if (test == 2) malformed->registers[0].gpu.components = 0;
            try { (void)Metal::InputReferenceTestAccess::PrepareNative(references, state, {}, retained, 3, 3, 0, {}, 0); }
            catch (const std::invalid_argument&) { ++rejected; }
            Check(Metal::InputReferenceTestAccess::Retains(state, malformed),
                  "Rejected private reference preparation changed retained state");
        }
        Check(rejected == 3, "Private reference factory accepted malformed retained records");
        malformed->registers[0] = {};
        auto indices = device.CreateBuffer(32, true);
        Metal::BufferSlice wrong{indices, 0, 3, static_cast<std::byte*>(indices.contents) + 1, {}};
        bool wrong_bytes{};
        try { (void)Metal::InputReferenceTestAccess::PrepareNative(references,state,{},retained,3,3,0,wrong,1); }
        catch (const std::invalid_argument&) { wrong_bytes = true; }
        Check(wrong_bytes, "Private index provenance accepted CPU bytes different from its GPU slice");
        bool foreign{};
        std::thread other{[&] {
            try { (void)Metal::InputReferenceTestAccess::PrepareNative(references,state,{},retained,3,3,0,{},0); }
            catch (const std::logic_error&) { foreign = true; }
        }};
        other.join(); Check(foreign && Metal::InputReferenceTestAccess::Retains(state, malformed),
                            "Foreign private preparation bypassed owner checks or changed state");
    }
    bool rejected = false;
    std::thread foreign([&] { try { Metal::VertexInputState state{device}; (void)references.Current(state); } catch (const std::exception&) { rejected = true; } }); foreign.join();
    Check(rejected, "Foreign thread entered input-reference state");
    scheduler.Finish();
    std::cout << "Production hashed FIFO histories: " << history_cases << " complete512-byte GPU comparisons passed\n";
    std::cout << "Private native reference publication: complete768-byte metadata matches checked public path, "
                 "9 mesh/full1824-byte shader-state cases and all180 retained512-byte GPU histories passed\n";
    Check(argument_checks >= 9 && multi_chunk_checks, "Missing one-/two-chunk argument publication coverage");
    std::cout << "Native mesh argument publications: " << argument_checks
              << " exact chunk-count checks, including98307 indexed elements/two chunks; no redundant first publication; "
                 "all complete GPU pixels and512/272/1824-byte states passed\n";
    std::cout << "Ordered framebuffer-fetch chains: " << ordered_chains << " across17 draws each, full pixels, one queued encoder per chain and explicit direct/queued transitions passed\n";
    std::cout << "Production reference mesh: " << cases << " complete pixels/input/primitive/VS-state and pinned-history cases passed; all index widths, quaternion repair, immutable GPU addresses, exact512/272/1824-byte state, foreign-thread rejection\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } } }
