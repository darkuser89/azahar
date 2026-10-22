// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_draw_runtime.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_transfer.h"
#include "mtl_compiler_gate_fixture.h"
#include "video_core/shader/generator/shader_uniforms.h"
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<std::filesystem::path> Files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator it(root, error), end;
         !error && it != end; it.increment(error)) {
        if (it->path().extension() == ".shader" && it->is_regular_file(error)) files.push_back(it->path());
    }
    return files;
}
Pica::RegsInternal Regs(Pica::ShaderSetup& setup) {
    Pica::RegsInternal regs{};
    Metal::Test::MovVertex(regs, setup);
    regs.lighting.disable.Assign(1);
    regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Xor);
    regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
    regs.framebuffer.framebuffer.allow_color_write.Assign(1);
    return regs;
}
void Scene(Metal::Device& device, Metal::ShaderRuntime& runtime,
           const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
           std::shared_ptr<const Metal::CompiledFragmentShader> supplied_fragment = {},
           std::shared_ptr<const Metal::VertexCaptureProgram> supplied_vertex = {},
           const std::function<void()>& after_capture = {},
           const std::function<std::shared_ptr<const Metal::CompiledFragmentShader>()>& before_draw = {},
           const std::function<void()>& after_draw = {}) {
    Metal::Scheduler scheduler{device};
    Metal::GeometryRuntime geometry{device, scheduler};
    Metal::TransferRuntime transfers{device, scheduler};
    Metal::StreamBuffer stream{device, scheduler, 64 << 10};
    const auto vertex = supplied_vertex ? supplied_vertex : runtime.Vertex(regs, setup);
    auto fragment = supplied_fragment ? supplied_fragment :
        runtime.Fragment(Pica::Shader::FSConfig{regs}, {}, Metal::FragmentColorFormat::RGBA8);
    Check(vertex->shader.spirv.size() >= 5 && fragment->program.shader.spirv.size() >= 5,
          "Native shader cache lost original guest compiler IR");
    Check(vertex->shader.stage == Metal::ShaderStage::VertexCompute &&
          fragment->program.shader.stage == Metal::ShaderStage::Fragment &&
          fragment->program.shader.uses_framebuffer_fetch,
          "Native shader cache lost stage or fragment fetch metadata");
    auto input = stream.Allocate(3 * 16 * sizeof(std::array<float, 4>));
    std::memset(input.data, 0, input.size);
    const std::array<std::array<float, 4>, 3> positions{{{-1,-1,0,1}, {3,-1,0,1}, {-1,3,0,1}}};
    auto* registers = static_cast<std::array<float, 4>*>(input.data);
    for (unsigned i = 0; i < 3; ++i) {
        registers[i * 16] = positions[i];
        registers[i * 16 + 1] = {0,0,0,1};
        registers[i * 16 + 2] = {1,.25f,.5f,1};
    }
    auto vs_data = stream.Allocate(sizeof(Pica::Shader::Generator::VSPicaUniformData));
    auto vs_extra = stream.Allocate(32);
    auto fs_data = stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
    std::memset(vs_data.data, 0, vs_data.size);
    std::memset(vs_extra.data, 0, vs_extra.size);
    std::memset(fs_data.data, 0, fs_data.size);
    static_cast<Pica::Shader::Generator::FSUniformData*>(fs_data.data)->depth_offset = .5f;
    const std::array<Metal::BufferBinding, 2> uniforms{{{vs_data.buffer,vs_data.offset,0},
                                                     {vs_extra.buffer,vs_extra.offset,1}}};
    const auto captured = geometry.Capture(*vertex, input, 3, 0, uniforms);
    if(after_capture)after_capture();
    auto readback = device.CreateBuffer(captured.size, true);
    transfers.CopyBuffer(captured.buffer, captured.offset, readback, 0, captured.size);
    scheduler.Finish();
    const auto* vertices = static_cast<const Metal::CapturedVertex*>(readback.contents);
    for (unsigned i = 0; i < 3; ++i) {
        if (vertices[i].position != positions[i] || vertices[i].color != std::array<float,4>{1,.25f,.5f,1}) {
            std::cerr << "Captured vertex " << i << " position";
            for (auto value : vertices[i].position) std::cerr << ' ' << value;
            std::cerr << " color";
            for (auto value : vertices[i].color) std::cerr << ' ' << value;
            std::cerr << '\n';
        }
        Check(vertices[i].position == positions[i] &&
              vertices[i].color == std::array<float,4>{1,.25f,.5f,1},
              "Disk-loaded PICA VS produced incorrect captured GPU attributes");
    }
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                        width:8 height:8 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    auto target = device.CreateTexture(descriptor);
    auto* pass = [[MTL4RenderPassDescriptor alloc] init];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    Metal::DrawRuntime draws{device, scheduler, geometry};
    const std::array<Metal::BufferBinding, 1> fs_uniforms{{{fs_data.buffer,fs_data.offset,2}}};
    if (before_draw) fragment = before_draw();
    draws.Draw(Metal::BuildPicaDrawState(regs), *fragment, pass, {0,0,8,8,0,1}, {0,0,8,8},
               {captured,3}, fs_uniforms, {}, {});
    if (after_draw) after_draw();
    const auto pixels = transfers.Readback({target,0,0,MTLRegionMake2D(0,0,8,8)}, 8 * 4);
    const std::array<unsigned, 4> expected{255,64,128,255};
    for (unsigned i = 0; i < pixels.size(); ++i) {
        if (std::abs(int(std::to_integer<unsigned>(pixels[i])) - int(expected[i % 4])) > 1)
            std::cerr << "Native cached scene byte " << i << " expected " << expected[i % 4]
                      << " actual " << std::to_integer<unsigned>(pixels[i]) << '\n';
        Check(std::abs(int(std::to_integer<unsigned>(pixels[i])) - int(expected[i % 4])) <= 1,
              "Disk-loaded PICA FS/reflection/operations produced incorrect GPU pixels");
    }
}

void DeferredFragmentTest(const std::filesystem::path& root) {
    Check(Files(root).empty(),"Deferred fragment test requires a fresh root");
    Metal::Device device{root};device.SwitchDiskCache(123);
    Pica::ShaderSetup setup{};const auto regs=Regs(setup);
    std::promise<void> release;
    struct Release {
        std::promise<void>& promise;bool open{};
        void Open(){if(!open){promise.set_value();open=true;}}
        ~Release(){Open();}
    } gate{release};
    const auto wait=release.get_future().share();
    std::array<std::promise<void>,2> started;
    std::array<std::shared_future<void>,2> blockers;
    for(unsigned i=0;i<2;++i) blockers[i]=device.DeferCompilation([&,i,wait]{started[i].set_value();wait.wait();});
    for(auto& start:started)start.get_future().wait();
    std::shared_ptr<const Metal::CompiledFragmentShader> shader;
    std::shared_ptr<const Metal::VertexCaptureProgram> vertex;
    std::shared_ptr<const Metal::CompiledMeshVertexProgram> mesh;
    {
        Metal::ShaderRuntime runtime{device};
        std::array<std::future<std::shared_ptr<const Metal::CompiledFragmentShader>>,8> requests;
        for(auto& request:requests) request=std::async(std::launch::async,[&] {
            @autoreleasepool {return runtime.Fragment(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8);}
        });
        shader=requests[0].get();
        for(unsigned i=1;i<requests.size();++i)Check(requests[i].get()==shader,"Deferred fragment first use duplicated its cache entry");
        vertex=runtime.Vertex(regs,setup);mesh=runtime.MeshVertex(regs,setup);
        Check(vertex->pending_pipeline.valid() && mesh->pending_library.valid() &&
            vertex->pending_pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
            mesh->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout,
            "Generated vertex/mesh program waited for native compilation before returning metadata");
        Check(shader->pending_library.valid() &&
            shader->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
            shader->program.shader.spirv.size()>=5 && runtime.DiskStats().translations==3 &&
            runtime.DiskStats().library_compiles==0 && Files(root).empty(),
            "Fragment metadata waited for the native compiler or persisted an unvalidated native result");
        runtime.Clear();
    }
    // No factory may capture the destroyed runtime/cache owner or a caller's source bytes.
    Metal::ShaderRuntime after{device};
    Scene(device,after,regs,setup,shader,vertex,[&] {
        const bool held=vertex->pending_pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout;
        gate.Open();
        Check(held,"Native capture waited for compilation on its owner before queuing the immutable dispatch");
    });
    Check(mesh->Library() && vertex->Pipeline() && vertex->Library(),"Deferred vertex/mesh publication lost a complete native result");
    Metal::Test::CompilerGate reload_gate{device};
    const auto files_before_reload = Files(root);
    std::vector<std::filesystem::file_time_type> stamps;
    for (const auto& path : files_before_reload) stamps.push_back(std::filesystem::last_write_time(path));
    const auto loaded_vertex=after.Vertex(regs,setup);const auto loaded_mesh=after.MeshVertex(regs,setup);
    const auto loaded=after.Fragment(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8);
    const bool held = loaded_vertex->pending_pipeline.valid() && loaded_mesh->pending_library.valid() &&
        loaded->pending_library.valid() &&
        loaded_vertex->pending_pipeline.wait_for(std::chrono::seconds{0}) == std::future_status::timeout &&
        loaded_mesh->pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout &&
        loaded->pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout;
    const bool timely_reload = reload_gate.Open();
    Check(held && timely_reload, "Loaded VS/FS/mesh IR waited for the native compiler on its owner");
    Check(loaded_vertex->shader == vertex->shader && loaded_mesh->shader == mesh->shader &&
        loaded->program.shader == shader->program.shader && loaded->program.operations == shader->program.operations,
        "Canonical IR reload changed MSL, entry, reflection, flags or fragment operations");
    Scene(device,after,regs,setup,loaded,loaded_vertex);
    Check(loaded_vertex->Pipeline() && loaded_mesh->Library() == mesh->Library() && loaded->Library() &&
        after.DiskStats().hits==3 && after.DiskStats().translations==0 && after.DiskStats().saved==0 && Files(root).size()==3,
        "Deferred IR reload regenerated GLSL, rewrote a valid record or lost a complete native result");
    for (unsigned i = 0; i < files_before_reload.size(); ++i)
        Check(std::filesystem::last_write_time(files_before_reload[i]) == stamps[i], "Valid IR reload rewrote its persistent record");
    for (unsigned corruption=0; corruption<5; ++corruption) {
        for (const auto& path : Files(root)) {
            std::ifstream input(path, std::ios::binary);
            std::vector<char> bytes{std::istreambuf_iterator<char>{input}, {}};
            const auto word = [&](std::size_t pos) {
                Check(pos+4 <= bytes.size(), "Corrupt IR fixture word is outside file");
                std::uint32_t value{};
                for (unsigned n=0;n<4;++n) value |= std::uint32_t(static_cast<unsigned char>(bytes[pos+n])) << (8*n);
                return value;
            };
            const auto source_size = word(44);
            const auto entry_size = word(48+source_size);
            const auto flag = 52ULL+source_size+entry_size;
            Check(flag+4 < bytes.size()-CC_SHA256_DIGEST_LENGTH, "Corrupt IR fixture metadata is outside record");
            if (corruption==0) bytes[48]='!';
            else if (corruption==1) bytes[52+source_size]='!';
            else if (corruption==2) bytes[flag]^=1; // Valid buffer-size boolean, inconsistent with IR.
            else if (corruption==3) {
                Check(word(flag+8)>0, "Cached IR fixture has no resource to corrupt");
                const auto name_size=word(flag+12);
                const auto active=flag+16+name_size+8*4;
                Check(active<bytes.size()-CC_SHA256_DIGEST_LENGTH, "Cached active flag is outside record");
                bytes[active]^=1; // Valid active mask that would suppress/introduce an owner binding.
            } else bytes[bytes.size()-CC_SHA256_DIGEST_LENGTH-48]^=1; // Valid fragment-operation metadata.
            CC_SHA256(bytes.data(), CC_LONG(bytes.size()-CC_SHA256_DIGEST_LENGTH),
                reinterpret_cast<unsigned char*>(bytes.data()+bytes.size()-CC_SHA256_DIGEST_LENGTH));
            std::ofstream output(path,std::ios::binary|std::ios::trunc);
            output.write(bytes.data(),std::streamsize(bytes.size()));
        }
        Metal::Test::CompilerGate repair_gate{device};
        auto repair = std::make_unique<Metal::ShaderRuntime>(device);
        const auto repaired_vertex=repair->Vertex(regs,setup);
        const auto repaired_mesh=repair->MeshVertex(regs,setup);
        const auto repaired_fragment=repair->Fragment(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8);
        const bool pending = repaired_vertex->pending_pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
            repaired_mesh->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
            repaired_fragment->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout;
        const auto stats=repair->DiskStats();
        const bool metadata = repaired_vertex->shader==vertex->shader && repaired_mesh->shader==mesh->shader &&
            repaired_fragment->program.shader==shader->program.shader && repaired_fragment->program.operations==shader->program.operations;
        repair->Clear();repair.reset();
        const bool timely=repair_gate.Open();
        Check(timely && pending && metadata && stats.rejected==3 && stats.translations==3 && stats.saved==0,
            "On-demand malformed MSL/entry/reflection exposed corrupt metadata or waited for native repair");
        Scene(device,after,regs,setup,repaired_fragment,repaired_vertex);
        Check(repaired_mesh->Library()==mesh->Library(), "Recovered mesh IR changed its complete native source");
    }
    std::cout << "Loaded IR: compiler-gated VS/FS/mesh, complete canonical MSL/entry/reflection/flags/operations,0 GLSL translations/0 rewrites;5 checksum-valid corruptions per stage repaired before metadata use, released runtime and exact GPU attributes/all64pixels passed\n";
    std::weak_ptr<const Metal::CompiledFragmentShader> released=shader;
    std::weak_ptr<const Metal::VertexCaptureProgram> released_vertex=vertex;
    std::weak_ptr<const Metal::CompiledMeshVertexProgram> released_mesh=mesh;
    shader.reset();vertex.reset();mesh.reset();
    device.DrainCompilations();
    Check(released.expired() && released_vertex.expired() && released_mesh.expired(),"Completed compiler factory retained its future owner in a cycle");
    Check(device.FlushDiskCache(),"Deferred fragment native archive failed to persist");
    {
        Metal::Device fresh{{}, false};
        Metal::ShaderRuntime runtime{fresh};
        std::unique_ptr<Metal::Test::CompilerGate> draw_gate;
        std::weak_ptr<const Metal::CompiledFragmentShader> owner;
        Scene(fresh, runtime, regs, setup, {}, {}, {}, [&] {
            draw_gate = std::make_unique<Metal::Test::CompilerGate>(fresh);
            Metal::ShaderRuntime temporary{fresh};
            auto selected = temporary.Fragment(Pica::Shader::FSConfig{regs}, {}, Metal::FragmentColorFormat::RGBA8);
            Check(selected->pending_library.valid() &&
                selected->pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout,
                "Full draw gate did not hold the fresh fragment library");
            owner = selected;
            temporary.Clear();
            return selected;
        }, [&] {
            const auto held = owner.lock();
            const bool pending = held && held->pending_library.wait_for(std::chrono::seconds{0}) == std::future_status::timeout;
            const bool timely = draw_gate->Open();
            Check(timely && pending, "Full ordinary Draw waited for its native fragment library on the owner");
        });
        fresh.DrainCompilations();
        Check(owner.expired(), "Deferred full draw retained the fragment metadata owner");
    }
    std::cout << "Full ordinary Draw: compiler-gated fresh fragment library, destroyed shader runtime, owner progress and all64 native GPU pixels passed\n";
    std::cout<<"Deferred PICA VS/FS/mesh: compiler-gated metadata, eight FS first-use callers/one FS translation, no premature IR publication, Clear/runtime destruction, complete captured GPU attributes/all64 pixels, three persisted IR reloads and no future-owner cycles passed\n";
}

void DeferredWarmIRTest(const std::filesystem::path& root) {
    Check(Files(root).size()==3, "Fresh-process gated IR test requires the three validated source records");
    Metal::Device device{root};device.SwitchDiskCache(123);
    Pica::ShaderSetup setup{};const auto regs=Regs(setup);
    auto runtime=std::make_unique<Metal::ShaderRuntime>(device);
    Metal::Test::CompilerGate gate{device};
    const auto vertex=runtime->Vertex(regs,setup);
    const auto mesh=runtime->MeshVertex(regs,setup);
    const auto fragment=runtime->Fragment(Pica::Shader::FSConfig{regs},{},Metal::FragmentColorFormat::RGBA8);
    const bool pending=vertex->pending_pipeline.valid() && mesh->pending_library.valid() && fragment->pending_library.valid() &&
        vertex->pending_pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
        mesh->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
        fragment->pending_library.wait_for(std::chrono::seconds{0})==std::future_status::timeout;
    const auto stats=runtime->DiskStats();
    runtime->Clear();runtime.reset();
    const bool timely=gate.Open();
    Check(timely && pending && stats.hits==3 && stats.translations==0 && stats.saved==0 && stats.library_compiles==0,
        "Fresh-process loaded IR metadata waited for native compilation, translated GLSL or wrote unvalidated data");
    Metal::ShaderRuntime scene{device};Scene(device,scene,regs,setup,fragment,vertex);
    Check(mesh->Library() && vertex->Pipeline() && fragment->Library(), "Fresh-process IR futures lost their complete native result");
    std::cout << "Fresh-process loaded VS/FS/mesh IR: native compiler held, owner metadata progress,3disk hits,0guest translations/0writes, Clear/runtime destruction and exact captured attributes/all64pixels passed\n";
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc == 3, "Specify cold/warm/edge mode and an isolated absolute cache root");
            const std::string mode{argv[1]};
            const std::filesystem::path root{argv[2]};
            Check(root.is_absolute(), "Native shader test root must be absolute");
            if (mode == "deferred") { DeferredFragmentTest(root); return 0; }
            if (mode == "deferred_warm") { DeferredWarmIRTest(root); return 0; }
            if (mode == "reflection_edge") {
                Metal::ShaderDiskCache cache;
                auto shader = Metal::TranslateGuestGLSL(
                    "layout(local_size_x=1) in;"
                    "layout(set=3,binding=0,std430) buffer data_block { vec4 values[]; };"
                    "layout(set=0,binding=0,std140) uniform constants { vec4 value; };"
                    "layout(set=1,binding=0) uniform sampler2D images[6];"
                    "layout(set=1,binding=1) uniform sampler2D normal_image;"
                    "void main() { if (gl_GlobalInvocationID.x < values.length())"
                    "values[gl_GlobalInvocationID.x] = value + textureLod(images[0],vec2(0.5),0)"
                    "+ textureLod(normal_image,vec2(0.5),0); }",
                    Metal::ShaderStage::Compute, Metal::RasterizerShaderBindings());
                Check(shader.needs_buffer_sizes && shader.resources.size() == 4,
                      "Reflection corruption fixture lost its active resources");
                const std::array key{std::byte{0x17}};
                Check(cache.Save(root,key,{shader,{}}) && cache.Load(root,key,shader.stage).has_value(),
                      "Valid shader reflection failed its disk roundtrip");
                for (unsigned failure=0;failure<5;++failure) {
                    auto invalid=shader;
                    auto duplicate=shader.resources.front();
                    duplicate.descriptor_set=42;duplicate.binding=failure;
                    if (failure==1) {
                        duplicate=shader.resources.front();duplicate.metal_index=29;
                        // Same guest descriptor, distinct native slot.
                    }
                    if (failure==2 || failure==3) {
                        const auto found=std::find_if(shader.resources.begin(),shader.resources.end(),
                            [](const auto& r) { return r.kind==Metal::ShaderResourceKind::Texture; });
                        Check(found!=shader.resources.end(),"Missing reflection texture fixture");
                        duplicate=*found;duplicate.descriptor_set=42;duplicate.binding=failure;
                        if (failure==2) duplicate.sampler_index=10;
                        else duplicate.metal_index=100;
                    }
                    if (failure==4) duplicate.metal_index=Metal::ShaderBufferSizesBinding;
                    invalid.resources.push_back(duplicate);
                    Check(cache.Save(root,key,{invalid,{}}),"Unable to write checksum-valid malformed reflection");
                    Check(!cache.Load(root,key,shader.stage),
                          "Checksum-valid shader cache accepted overlapping reflection bindings");
                }
                Check(cache.Stats().rejected==5,"Malformed reflection rejection count changed");
                std::cout << "Five checksum-valid reflection corruptions reject duplicate descriptors, overlapping buffer/texture/sampler slots and buffer-size collision\n";
                return 0;
            }
            Pica::ShaderSetup setup;
            auto regs = Regs(setup);
            if (mode == "cold") Check(Files(root).empty(), "Cold source test requires an empty isolated root");
            if (mode == "edge" || mode == "prewarm_edge" || mode == "entry_edge") {
                auto files = Files(root);
                std::erase_if(files, [](const auto& path) {
                    return path.parent_path().parent_path().parent_path().filename() != "000000000000007b";
                });
                Check(files.size() == 2, "Corruption test requires the isolated title's VS/FS records");
                for (unsigned i = 0; i < files.size(); ++i) {
                    std::ifstream input(files[i], std::ios::binary);
                    std::vector<char> bytes{std::istreambuf_iterator<char>{input}, {}};
                    Check(bytes.size() > 100, "Native source record has no MSL payload");
                    if (mode == "entry_edge") {
                        const auto source_size = std::uint32_t(static_cast<unsigned char>(bytes[44])) |
                            (std::uint32_t(static_cast<unsigned char>(bytes[45])) << 8) |
                            (std::uint32_t(static_cast<unsigned char>(bytes[46])) << 16) |
                            (std::uint32_t(static_cast<unsigned char>(bytes[47])) << 24);
                        Check(source_size + 52ULL < bytes.size() - CC_SHA256_DIGEST_LENGTH,
                              "Native shader entry metadata lies outside source record");
                        bytes[source_size + 52] = '!'; // Valid MSL, but no such native function.
                        CC_SHA256(bytes.data(), CC_LONG(bytes.size() - CC_SHA256_DIGEST_LENGTH),
                                  reinterpret_cast<unsigned char*>(bytes.data() + bytes.size() - CC_SHA256_DIGEST_LENGTH));
                    } else if (i == 0) bytes[80] ^= 0x55; // Broken checksum must reject before compilation.
                    else {
                        bytes[48] = '!'; // Valid checksum, invalid MSL must regenerate after compiler rejection.
                        CC_SHA256(bytes.data(), CC_LONG(bytes.size() - CC_SHA256_DIGEST_LENGTH),
                                  reinterpret_cast<unsigned char*>(bytes.data() + bytes.size() - CC_SHA256_DIGEST_LENGTH));
                    }
                    std::ofstream output(files[i], std::ios::binary | std::ios::trunc);
                    output.write(bytes.data(), std::streamsize(bytes.size()));
                }
            }
            Metal::Device device{root};
            Check(device.DiskCacheStats().enabled, "Native source tests require optional persistence enabled");
            device.SwitchDiskCache(123);
            Metal::ShaderRuntime runtime{device};
            if (mode == "prewarm_edge" || mode == "entry_edge") {
                const std::atomic_bool stop{false};
                const auto result = runtime.Prewarm(stop);
                Check(result.candidates == 2 && result.failed == 2 && !result.compiled && !result.cancelled,
                      "Corrupt/syntax-invalid prewarm published incomplete native shader objects");
            }
            if (mode == "prewarm") {
                std::atomic_bool stop{false};
                std::vector<std::pair<std::size_t, std::size_t>> progress;
                const auto preload = runtime.Prewarm(stop, [&](std::size_t done, std::size_t total) {
                    progress.emplace_back(done, total);
                    (void)runtime.DiskStats(); // A callback must be able to re-enter the stats API.
                });
                Check(preload.candidates == 2 && preload.compiled == 2 && !preload.failed &&
                      !preload.cancelled && !preload.budget_limited && progress.front().first == 0 &&
                      progress.back().first == 2, "Native shader prewarm did not complete both source records");
                const auto before = runtime.DiskStats();
                Scene(device, runtime, regs, setup);
                const auto after = runtime.DiskStats();
                Check(after.prewarm_hits == 2 && !after.translations &&
                      after.library_compiles == before.library_compiles,
                      "First GPU scene repeated guest translation or PICA library compilation after prewarm");
                const auto repeated = runtime.Prewarm(stop);
                Check(repeated.visited == 2 && !repeated.compiled &&
                      runtime.DiskStats().library_compiles == after.library_compiles,
                      "Repeated prewarm compiled already complete native objects");
                stop.store(true);
                const auto cancelled = runtime.Prewarm(stop);
                Check(cancelled.cancelled && !cancelled.visited && !cancelled.compiled,
                      "Initially cancelled prewarm touched source records");
                runtime.Clear(); stop.store(false);
                const auto interrupted = runtime.Prewarm(stop, [&](std::size_t done, std::size_t) {
                    if (done == 1) stop.store(true);
                });
                Check(interrupted.cancelled && interrupted.compiled == 1 && interrupted.visited == 1,
                      "Mid-prewarm cancellation compiled a subsequent record");
                Scene(device, runtime, regs, setup);
                Check(!runtime.DiskStats().translations, "Partial prewarm lost valid on-demand disk reuse");
                runtime.Clear(); stop.store(false);
                const auto cleared = runtime.Prewarm(stop, [&](std::size_t done, std::size_t) {
                    if (!done) runtime.Clear();
                });
                Check(cleared.cancelled && !cleared.compiled,
                      "Callback cache clear failed to invalidate the in-progress prewarm epoch");
                runtime.Clear(); device.SetDiskCacheEnabled(false);
                const auto disabled = runtime.Prewarm(stop);
                Check(!disabled.candidates && !disabled.compiled, "Disabled cache still prewarmed persistent sources");
                device.SetDiskCacheEnabled(true);
                Scene(device, runtime, regs, setup);
                Check(!runtime.DiskStats().translations, "Cancelled/disabled preload damaged existing disk records");
                Check(device.FlushDiskCache(), "Native pipeline archive flush after prewarm failed");
                std::cout << "Native shader prewarm: 2 source records, native VS pipeline/FS library ready; first GPU scene 0 guest translations/0 new PICA library compiles; all64 pixels and captured attributes passed\n";
                std::cout << "Native prewarm repeated/initial-cancel/mid-cancel/reentrant-stats/clear-epoch/disabled cases passed; no app/window/ROM/emulation started\n";
                return 0;
            }
            Scene(device, runtime, regs, setup);
            const auto stats = runtime.DiskStats();
            if (mode == "warm") {
                Check(stats.hits == 2 && stats.translations == 0 && stats.saved == 0,
                      "Fresh-process VS/FS cache hit repeated guest shader generation/translation");
                std::cout << "Fresh-process PICA VS/FS MSL/IR/reflection cache: 2 hits, 0 guest translations; native captured attributes and all64 GPU pixels passed\n";
            } else if (mode == "cold" || mode == "edge" || mode == "prewarm_edge" || mode == "entry_edge") {
                const auto expected_hits = mode == "entry_edge" ? 4U : mode == "prewarm_edge" ? 2U : mode == "edge" ? 1U : 0U;
                Check(stats.translations == 2 && stats.saved == 2 && stats.hits == expected_hits,
                      "Native source miss/corruption did not generate and atomically save both PICA shaders");
                if (mode == "edge" || mode == "prewarm_edge" || mode == "entry_edge")
                    Check(stats.rejected >= 2, "Native shader checksum corruption was not rejected");
                std::cout << "Native source cold/corrupt regeneration: original IR, MSL, full reflection and fragment operations retained; GPU VS/FS results passed\n";
            } else throw std::invalid_argument("Unknown native shader disk test mode");
            runtime.Clear();
            Scene(device, runtime, regs, setup);
            Check(runtime.DiskStats().hits == stats.hits + 2 && runtime.DiskStats().translations == stats.translations,
                  "Native source records could not be reused after RAM cache clear");
            runtime.Clear();
            device.SwitchDiskCache(mode == "cold" ? 456 : mode == "warm" ? 556 : mode == "entry_edge" ? 756 : 656);
            Scene(device, runtime, regs, setup);
            Check(runtime.DiskStats().translations == stats.translations + 2,
                  "Native source cache leaked a record across title partitions");
            runtime.Clear();
            device.SwitchDiskCache(123);
            Scene(device, runtime, regs, setup);
            Check(runtime.DiskStats().hits == stats.hits + 4,
                  "Native source title return lost persisted VS/FS programs");
            runtime.Clear();
            device.SetDiskCacheEnabled(false);
            Scene(device, runtime, regs, setup);
            Check(runtime.DiskStats().translations == stats.translations + 4 &&
                  runtime.DiskStats().saved == stats.saved + 2,
                  "Disabled native source cache still read or wrote persistent shader records");
            device.SetDiskCacheEnabled(true);
            Check(device.FlushDiskCache(), "Native PSO flush after source cache failed");
            std::cout << "Native shader cache RAM clear/title isolation/title return/runtime disable preserve GPU output; no app/window/ROM/emulation started\n";
            if (mode == "prewarm_edge")
                std::cout << "Native prewarm rejected both checksum/syntax failures, then on-demand PICA regeneration repaired files and GPU output\n";
            if (mode == "entry_edge")
                std::cout << "Native prewarm rejected missing VS/FS entry names despite valid MSL/checksum, then PICA regeneration restored all64 GPU pixels and captured attributes\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
