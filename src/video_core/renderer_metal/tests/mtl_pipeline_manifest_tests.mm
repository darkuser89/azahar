// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_pipeline_cache.h"
#include "mtl_pipeline_manifest.h"
#include "mtl_shader_disk_cache.h"
#include "mtl_scheduler.h"
#include "mtl_transfer.h"
#import <CommonCrypto/CommonDigest.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
constexpr std::string_view Vertex=R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 manifest_vertex(uint i [[vertex_id]]) {
    const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);
})";
constexpr std::string_view Fragment=R"(
#include <metal_stdlib>
using namespace metal;
fragment float4 manifest_fragment(){return float4(1,.25,.5,1);}
)";
std::vector<std::filesystem::path> Files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;std::error_code error;
    for(std::filesystem::recursive_directory_iterator it(root,error),end;!error&&it!=end;it.increment(error))
        if(it->path().extension()==".pipeline"&&it->is_regular_file(error))files.push_back(it->path());
    return files;
}
void Draw(Metal::Device& device,Metal::PipelineCache& cache,const Metal::RenderPipelineDescription& description,
          std::array<unsigned,4> foreground={255,64,128,255}) {
    const auto pipeline=cache.Render(description);
    auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:4 mipmapped:NO];
    descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    auto target=device.CreateTexture(descriptor);
    Metal::Scheduler scheduler{device};Metal::TransferRuntime transfers{device,scheduler};
    auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=target;
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    const std::array<unsigned,4> background{17,33,65,129};
    pass.colorAttachments[0].clearColor=MTLClearColorMake(17./255,33./255,65./255,129./255);
    scheduler.KeepAlive(pipeline);auto encoder=scheduler.BeginRender(pass);[encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{0,0,8,4,0,1}];[encoder setScissorRect:MTLScissorRect{0,0,8,4}];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];scheduler.EndEncoding();
    const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,8,4)},8*4);
    for(unsigned i=0;i<pixels.size();++i) {
        const auto channel=i%4;
        const bool writes=(description.blend.color_write_mask&(1U<<channel)) &&
                           (description.blend.enabled || description.blend.logic_operation!=6);
        const auto expected=writes?foreground[channel]:background[channel];
        Check(std::abs(int(std::to_integer<unsigned>(pixels[i]))-int(expected))<=1,
              "Manifest/prepared native PSO lost GPU color-mask/blend/NoOp pixels");
    }
}
std::vector<Metal::BlendState> States() {
    std::vector<Metal::BlendState> states;
    for(unsigned mask=0;mask<16;++mask) {Metal::BlendState state{};state.color_write_mask=mask;states.push_back(state);}
    Metal::BlendState no_op{};no_op.logic_operation=6;states.push_back(no_op);
    Metal::BlendState blend{};blend.enabled=true;blend.source_rgb=1;blend.dest_rgb=0;
    blend.source_alpha=1;blend.dest_alpha=0;states.push_back(blend);
    blend.emulate_rgb_minmax=true;blend.emulate_alpha_minmax=true;states.push_back(blend);
    return states;
}
constexpr std::string_view Mesh=R"(
#include <metal_stdlib>
using namespace metal;
struct MeshVertex {float4 position [[position]];};
using MeshOutput=mesh<MeshVertex,void,3,1,topology::triangle>;
[[mesh]] void manifest_mesh(uint i [[thread_index_in_threadgroup]],MeshOutput output) {
    if(i==0) {
        output.set_vertex(0,MeshVertex{float4(-1,-1,0,1)});
        output.set_vertex(1,MeshVertex{float4(3,-1,0,1)});
        output.set_vertex(2,MeshVertex{float4(-1,3,0,1)});
        output.set_index(0,0);output.set_index(1,1);output.set_index(2,2);
        output.set_primitive_count(1);
    }
})";
std::vector<std::byte> MeshPixels(Metal::Device& device,id<MTLRenderPipelineState> pipeline,
                                const Metal::RenderPipelineDescription& description,
                                std::array<unsigned,4> foreground={255,64,128,255}) {
    auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:4 mipmapped:NO];
    descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    auto target=device.CreateTexture(descriptor);Metal::Scheduler scheduler{device};Metal::TransferRuntime transfer{device,scheduler};
    auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=target;
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    const std::array<unsigned,4> background{17,33,65,129};
    pass.colorAttachments[0].clearColor=MTLClearColorMake(17./255,33./255,65./255,129./255);
    scheduler.KeepAlive(pipeline);auto encoder=scheduler.BeginRender(pass);[encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{0,0,8,4,0,1}];[encoder setScissorRect:MTLScissorRect{0,0,8,4}];
    [encoder drawMeshThreadgroups:MTLSizeMake(1,1,1) threadsPerObjectThreadgroup:MTLSizeMake(0,0,0)
        threadsPerMeshThreadgroup:MTLSizeMake(96,1,1)];scheduler.EndEncoding();
    const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,8,4)},8*4);
    for(unsigned i=0;i<pixels.size();++i) {
        const auto c=i%4;const bool writes=description.fragment_library && (description.blend.color_write_mask&(1u<<c)) &&
            (description.blend.enabled || description.blend.logic_operation!=6);
        Check(std::abs(int(std::to_integer<unsigned>(pixels[i]))-int(writes?foreground[c]:background[c]))<=1,
              "Native archived/prewarmed mesh changed full color/mask/blend/NoOp/no-fragment pixels");
    }
    return {pixels.begin(),pixels.end()};
}
void MeshManifestTest(const std::string& mode,const std::filesystem::path& root) {
    const bool cold=mode=="mesh-cold",warm=mode=="mesh-warm",validated=mode=="mesh-validation";
    Check(cold||warm||validated,"Unknown mesh cache test mode");
    if(cold||validated)Check(Files(root).empty(),"Mesh cold/validation root must be fresh");
    Metal::Device device{root};device.SwitchDiskCache(123);Metal::PipelineCache cache{device};
    Check(device.DiskCacheStats().enabled!=validated,"Mesh cache instrumentation isolation mismatch");
    const auto vi=Metal::Device::SourceIdentity("Manifest mesh fixture",Mesh);
    const auto fi=Metal::Device::SourceIdentity("Manifest fragment fixture",Fragment);
    id<MTLLibrary> vertex=nil,fragment=nil;
    const auto states=States();constexpr unsigned count=20;
    std::atomic_bool stop{false};Metal::PipelinePrewarmResult ready{};
    if(warm) {
        ready=cache.Prewarm(stop,[&](std::size_t,std::size_t){(void)cache.Stats();});
        Check(ready.prepared==count&&!ready.failed&&!ready.cancelled,"Fresh-process mesh PSOs were not all prepared");
        Check(device.DiskCacheStats().archive_hits==count&&device.DiskCacheStats().compiled_pipelines==0,
              "Fresh-process mesh prewarm did not use native archive binaries");
        vertex=device.LibraryWithIdentity(vi);fragment=device.LibraryWithIdentity(fi);
    } else {vertex=device.CompileLibrary(Mesh,"Manifest mesh fixture");fragment=device.CompileLibrary(Fragment,"Manifest fragment fixture");}
    Check(vertex&&fragment,"Missing persisted mesh library");
    Metal::RenderPipelineRecord record{vi,"manifest_mesh",fi,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{},true};
    auto traditional=record;traditional.mesh=false;
    Check(Metal::RenderPipelineManifest::Identity(record)!=Metal::RenderPipelineManifest::Identity(traditional),
          "Mesh and traditional pipeline manifest types alias");
    std::vector<std::byte> all_pixels;const auto builds=cache.Stats().render_builds;
    for(unsigned i=0;i<count;++i) {
        Metal::RenderPipelineDescription description{vertex,"manifest_mesh",i==19?nil:fragment,
            i==19?"":"manifest_fragment",MTLPixelFormatRGBA8Unorm,i==19?Metal::BlendState{}:states[i]};
        const auto pipeline=cache.Mesh(description);const auto pixels=MeshPixels(device,pipeline,description);
        all_pixels.insert(all_pixels.end(),pixels.begin(),pixels.end());
        Check(cache.Mesh(description)==pipeline,"Mesh cache lost a live native PSO");
    }
    if(warm)Check(cache.Stats().prewarm_hits==count&&cache.Stats().render_builds==builds,
                  "Prepared mesh draws rebuilt a descriptor/PSO");
    Check(device.FlushDiskCache(),"Native mesh archive flush failed");
    const auto baseline=root/"mesh-cold-pixels.bin";
    if(cold) {
        Check(Files(root).size()==count&&device.DiskCacheStats().compiled_pipelines==count,
              "Cold mesh capture lost descriptor variants");
        std::ofstream out(baseline,std::ios::binary);out.write(reinterpret_cast<const char*>(all_pixels.data()),all_pixels.size());
        Check(bool(out),"Could not save complete cold mesh pixels");
    }
    if(warm) {
        std::vector<std::byte> expected(all_pixels.size());std::ifstream in(baseline,std::ios::binary);
        in.read(reinterpret_cast<char*>(expected.data()),expected.size());
        Check(bool(in)&&in.peek()==std::char_traits<char>::eof()&&expected==all_pixels,
              "Fresh-process archived/prewarmed mesh pixels differ from cold native pixels");
        cache.Clear();const std::array<std::string_view,1> entries{"raster_vertex"};
        Check(!cache.Prewarm(stop,{},entries).candidates,"Traditional-only prewarm included mesh records");
        Check(cache.Prewarm(stop,{},entries,true).prepared==count,"PICA prewarm omitted persisted mesh entries");
        cache.Clear();stop.store(true);const auto cancelled=cache.Prewarm(stop);
        Check(cancelled.cancelled&&!cancelled.visited,"Cancelled mesh prewarm touched pipelines");
        stop.store(false);const auto partial=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(done==1)stop.store(true);});
        Check(partial.cancelled&&partial.prepared==1&&partial.visited==1,"Mesh prewarm cancellation advanced too far");
        cache.Clear();stop.store(false);const auto cleared=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(!done)cache.Clear();});
        Check(cleared.cancelled&&!cleared.prepared,"Mesh prewarm ignored reentrant Clear epoch");
        cache.Clear();device.SwitchDiskCache(456);Check(!cache.Prewarm(stop).candidates,"Mesh manifests leaked between titles");
        cache.Clear();device.SwitchDiskCache(123);Check(cache.Prewarm(stop).prepared==count,"Mesh title return lost archived variants");
        // Per-descriptor instrumentation must compile without reading or writing binary archives.
        const auto saved=device.DiskCacheStats().saved_batches,hits=device.DiskCacheStats().archive_hits;
        auto* diagnostic=[[MTL4MeshRenderPipelineDescriptor alloc] init];diagnostic.meshFunctionDescriptor=Metal::Device::Function(vertex,"manifest_mesh");
        diagnostic.fragmentFunctionDescriptor=Metal::Device::Function(fragment,"manifest_fragment");
        diagnostic.maxTotalThreadsPerMeshThreadgroup=96;diagnostic.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
        diagnostic.options=[[MTL4PipelineOptions alloc] init];diagnostic.options.shaderValidation=MTLShaderValidationEnabled;
        (void)MeshPixels(device,device.CompileMesh(diagnostic),{vertex,"manifest_mesh",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}});
        device.FlushDiskCache();Check(device.DiskCacheStats().saved_batches==saved&&device.DiskCacheStats().archive_hits==hits,
              "Validated mesh pipeline reused or polluted native archives");
        cache.Clear();device.SetDiskCacheEnabled(false);
        Check(!cache.Prewarm(stop).candidates,"Disabled cache prepared mesh manifests");
        Metal::RenderPipelineDescription original{vertex,"manifest_mesh",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}};
        (void)MeshPixels(device,cache.Mesh(original),original);
        device.FlushDiskCache();Check(device.DiskCacheStats().saved_batches==saved&&device.DiskCacheStats().archive_hits==hits,
              "Disabled mesh disk cache read/wrote binaries");
        cache.Clear();device.SetDiskCacheEnabled(true);Check(cache.Prewarm(stop).prepared==count,"Reenabled mesh cache lost prior manifests");
        const auto before=device.DiskCacheStats().compiled_pipelines;
        auto changed=device.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;fragment float4 manifest_fragment(){return float4(.5,.75,.125,1);}","Manifest fragment fixture");
        Metal::RenderPipelineDescription other{vertex,"manifest_mesh",changed,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}};
        (void)MeshPixels(device,cache.Mesh(other),other,{128,191,32,255});
        Check(device.DiskCacheStats().compiled_pipelines==before+1,"Changed mesh fragment source reused an incompatible native PSO");
    }
    if(validated)Check(!device.DiskCacheStats().archive_hits&&!device.DiskCacheStats().saved_batches&&Files(root).empty(),
                        "Global shader validation wrote or reused mesh binaries/manifests");
    std::cout<<"20 native mesh variants: masks/NoOp/blend/minmax/no-fragment, all2560 pixel bytes checked; "
             <<(warm?"fresh-process20 archive hits/0 PSO recompiles/20 prepared first draws; exact cold/warm pixels; ":"")
             <<"typed identity, cancellation/title/validation/disable lifetime checks passed, mode="<<mode<<'\n';
}

using Bytes=std::vector<std::byte>;
void Word(Bytes& bytes,std::uint32_t value) {
    for(unsigned i=0;i<4;++i)bytes.push_back(std::byte(value>>(8*i)));
}
void String(Bytes& bytes,std::string_view value) {
    Word(bytes,std::uint32_t(value.size()));
    const auto view=std::as_bytes(std::span(value));bytes.insert(bytes.end(),view.begin(),view.end());
}
Bytes Checked(Bytes bytes) {
    std::array<unsigned char,32> hash{};CC_SHA256(bytes.data(),CC_LONG(bytes.size()),hash.data());
    const auto view=std::as_bytes(std::span(hash));bytes.insert(bytes.end(),view.begin(),view.end());return bytes;
}
void Write(const std::filesystem::path& path,std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    Check(bool(out),"Could not write historical native fixture");
}
std::string OldIdentity(std::string_view version,std::string_view name,std::string_view source) {
    Bytes bytes;
    for(const auto value:{version,name,source}) {
        const auto size=std::uint64_t(value.size());
        for(unsigned i=0;i<8;++i)bytes.push_back(std::byte(size>>(8*i)));
        const auto view=std::as_bytes(std::span(value));bytes.insert(bytes.end(),view.begin(),view.end());
    }
    return Metal::ShaderDiskCache::Identity(bytes);
}
std::string OldLibrary(const std::filesystem::path& dir,std::string_view version,
                       std::string_view name,std::string_view source) {
    const auto id=OldIdentity(version,name,source);Bytes bytes;
    for(char c:std::string_view("AZM4LIB1"))bytes.push_back(std::byte(c));
    String(bytes,name);String(bytes,source);
    Write(dir/"libraries"/version/(id+".library"),Checked(std::move(bytes)));return id;
}
Bytes OldRecord(std::string_view version,const Metal::RenderPipelineRecord& record) {
    Bytes bytes;for(char c:record.mesh?std::string_view("AZM4MPS"):std::string_view("AZM4RPS"))bytes.push_back(std::byte(c));
    bytes.push_back(std::byte{1});String(bytes,version);
    String(bytes,record.vertex_identity);String(bytes,record.vertex_entry);
    String(bytes,record.fragment_identity);String(bytes,record.fragment_entry);
    Word(bytes,std::uint32_t(record.color_format));const auto& b=record.blend;
    for(auto value:std::array<std::uint32_t,11>{b.enabled,b.color_write_mask,b.logic_operation,b.source_rgb,
        b.dest_rgb,b.source_alpha,b.dest_alpha,b.rgb_equation,b.alpha_equation,b.emulate_rgb_minmax,b.emulate_alpha_minmax})Word(bytes,value);
    Word(bytes,1);Word(bytes,record.mesh?96:std::uint32_t(MTLPrimitiveTopologyClassTriangle));
    Word(bytes,!record.fragment_identity.empty());return bytes;
}
std::filesystem::path OldPipeline(const std::filesystem::path& dir,std::string_view version,Bytes bytes) {
    const auto path=dir/version/(Metal::ShaderDiskCache::Identity(bytes)+".pipeline");
    Write(path,Checked(std::move(bytes)));return path;
}
void HistoricalTest(const std::filesystem::path& root) {
    Check(!std::filesystem::exists(root),"Historical test needs a fresh isolated root");
    Metal::Device device{root};device.SwitchDiskCache(123);Metal::PipelineCache cache{device};
    const auto dir=device.RenderDiskCacheDirectory();Check(!dir.empty(),"Native persistence must be enabled");
    const std::string version(64,'1'),second(64,'2');const auto states=States();
    const auto fixture=[&](const std::filesystem::path& where,std::string_view ver,bool mesh=false) {
        return Metal::RenderPipelineRecord{OldLibrary(where,ver,mesh?"Historical mesh":"Historical vertex",mesh?Mesh:Vertex),
            mesh?"manifest_mesh":"manifest_vertex",OldLibrary(where,ver,"Historical fragment",Fragment),
            "manifest_fragment",MTLPixelFormatRGBA8Unorm,{},mesh};
    };
    for(const auto& ver:{version,second}) {
        auto record=fixture(dir,ver);for(const auto& state:states){record.blend=state;OldPipeline(dir,ver,OldRecord(ver,record));}
        record=fixture(dir,ver,true);for(const auto& state:states){record.blend=state;OldPipeline(dir,ver,OldRecord(ver,record));}
        record.fragment_identity.clear();record.fragment_entry.clear();record.blend={};OldPipeline(dir,ver,OldRecord(ver,record));
    }
    std::map<std::filesystem::path,Bytes> originals;
    for(const auto& path:Files(root)) {
        std::ifstream in(path,std::ios::binary);std::string value{std::istreambuf_iterator<char>{in},{}};
        const auto view=std::as_bytes(std::span(value));originals.emplace(path,Bytes(view.begin(),view.end()));
    }
    Metal::RenderPipelineManifest manifest;std::atomic_bool stop{false};bool limited=false;
    Check(manifest.Load(dir,stop,limited).empty(),"Normal current-version manifest loader accepted historical PSOs");
    const auto hints=manifest.LoadHistorical(dir,stop,limited);
    Check(hints.size()==78&&!limited,"Historical sources/descriptors were not fully validated");
    const auto old_vertex=OldIdentity(version,"Historical vertex",Vertex);
    Check(!Metal::RenderPipelineManifest::LoadLibrary(dir,old_vertex),"Normal library loader crossed source-version boundary");
    const auto ready=cache.Prewarm(stop,[&](std::size_t,std::size_t){(void)cache.Stats();});
    Check(ready.candidates==78&&ready.visited==78&&ready.prepared==39&&!ready.failed&&!ready.cancelled&&!ready.budget_limited,
          "Historical prewarm did not rebuild/deduplicate complete CURRENT keys");
    Check(manifest.Load(dir,stop,limited).size()==39,"Historical prewarm failed to publish current descriptors");
    for(const auto& [path,expected]:originals) {
        std::ifstream in(path,std::ios::binary);std::string value{std::istreambuf_iterator<char>{in},{}};
        Check(std::ranges::equal(std::as_bytes(std::span(value)),expected),"Prewarm modified original historical records");
    }
    auto vertex=device.CompileLibrary(Vertex,"Historical vertex"),mesh=device.CompileLibrary(Mesh,"Historical mesh"),
         fragment=device.CompileLibrary(Fragment,"Historical fragment");
    const auto builds=cache.Stats().render_builds,compiled=device.DiskCacheStats().compiled_pipelines;
    for(const auto& state:states) {
        Draw(device,cache,{vertex,"manifest_vertex",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,state});
        Metal::RenderPipelineDescription description{mesh,"manifest_mesh",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,state};
        (void)MeshPixels(device,cache.Mesh(description),description);
    }
    Metal::RenderPipelineDescription no_fragment{mesh,"manifest_mesh",nil,"",MTLPixelFormatRGBA8Unorm,{}};
    (void)MeshPixels(device,cache.Mesh(no_fragment),no_fragment);
    Check(cache.Stats().prewarm_hits==39&&cache.Stats().render_builds==builds&&device.DiskCacheStats().compiled_pipelines==compiled,
          "First historical-prepared GPU draws rebuilt PSOs");
    auto green=device.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;fragment float4 manifest_fragment(){return float4(.5,.75,.125,1);}","Historical fragment");
    Draw(device,cache,{vertex,"manifest_vertex",green,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}},{128,191,32,255});
    auto renamed=device.CompileLibrary(Fragment,"Historical fragment renamed");
    Draw(device,cache,{vertex,"manifest_vertex",renamed,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}});
    Check(device.DiskCacheStats().compiled_pipelines==compiled+2,"Full current library name/source did not isolate PSOs");
    Check(device.FlushDiskCache(),"Historical recompile could not persist current archive");
    cache.Clear();const auto again=cache.Prewarm(stop);
    Check(again.prepared==41&&!again.failed,"Current+historical descriptors did not deduplicate on repeat");
    cache.Clear();stop.store(true);Check(cache.Prewarm(stop).cancelled,"Initially stopped historical prewarm ran");
    stop.store(false);const auto partial=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(done==1)stop.store(true);});
    Check(partial.cancelled&&partial.prepared==1&&partial.visited==1,"Historical cancellation published a later record");
    stop.store(false);cache.Clear();const auto cleared=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(!done)cache.Clear();});
    Check(cleared.cancelled&&!cleared.prepared,"Historical prewarm ignored reentrant Clear");
    cache.Clear();const auto switched=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(!done)device.SwitchDiskCache(456);});
    Check(switched.cancelled&&!switched.prepared&&!cache.Prewarm(stop).candidates,"Historical prewarm crossed a title switch");
    device.SwitchDiskCache(123);cache.Clear();device.SetDiskCacheEnabled(false);
    Check(!cache.Prewarm(stop).candidates,"Disabled disk cache loaded historical hints");
    device.SetDiskCacheEnabled(true);
    Check(cache.Prewarm(stop).prepared==41,"Reenabled historical/current cache lost valid records");
    cache.Clear();device.SwitchDiskCache(789);
    const auto factory_dir=device.RenderDiskCacheDirectory();
    auto valid=fixture(factory_dir,version);OldPipeline(factory_dir,version,OldRecord(version,valid));
    auto invalid=valid;invalid.vertex_entry="missing_historical_vertex";
    OldPipeline(factory_dir,version,OldRecord(version,invalid));
    const auto factory=cache.Prewarm(stop);
    Check(factory.candidates==2&&factory.prepared==1&&factory.failed==1&&!factory.cancelled,
          "Current compiler failure published an invalid historical PSO or lost its valid neighbor");
    const auto valid_builds=cache.Stats().render_builds;
    Draw(device,cache,{vertex,"manifest_vertex",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}});
    Check(cache.Stats().render_builds==valid_builds,"Current factory failure discarded a valid prepared neighbor");
    const auto negative=root/"negative";
    const auto rejected=[&](const char* name,const auto& damage) {
        const auto where=negative/name;auto record=fixture(where,version);
        auto payload=OldRecord(version,record);const auto path=OldPipeline(where,version,payload);
        damage(where,record,path,payload);Metal::RenderPipelineManifest other;bool bounded=false;
        Check(other.LoadHistorical(where,stop,bounded).empty()&&other.Rejected()>0&&!bounded,
              "Historical loader accepted corrupt/missing/aliased source, schema or path");
    };
    rejected("checksum",[&](const auto&,const auto&,const auto& path,const auto& payload){auto bytes=Checked(payload);bytes[80]^=std::byte{1};Write(path,bytes);});
    rejected("schema",[&](const auto& where,const auto&,const auto& path,auto payload){std::filesystem::remove(path);payload[0]=std::byte{'X'};OldPipeline(where,version,payload);});
    rejected("fixed-abi",[&](const auto& where,const auto&,const auto& path,auto payload){std::filesystem::remove(path);payload[payload.size()-8]=std::byte{95};OldPipeline(where,version,payload);});
    rejected("filename",[&](const auto&,const auto&,const auto& path,const auto&){std::filesystem::rename(path,path.parent_path()/(std::string(64,'e')+".pipeline"));});
    rejected("missing-library",[&](const auto& where,const auto& record,const auto&,const auto&){std::filesystem::remove(where/"libraries"/version/(record.vertex_identity+".library"));});
    rejected("library-identity",[&](const auto& where,const auto& record,const auto&,const auto&){
        const auto other=OldLibrary(where,version,"Different name",Vertex);
        std::filesystem::copy_file(where/"libraries"/version/(other+".library"),where/"libraries"/version/(record.vertex_identity+".library"),std::filesystem::copy_options::overwrite_existing);
    });
    rejected("pipeline-symlink",[&](const auto&,const auto&,const auto& path,const auto&){const auto moved=path.string()+".original";std::filesystem::rename(path,moved);std::filesystem::create_symlink(moved,path);});
    rejected("version-symlink",[&](const auto& where,const auto&,const auto&,const auto&){std::filesystem::rename(where/version,where/"original");std::filesystem::create_directory_symlink(where/"original",where/version);});
    rejected("library-symlink",[&](const auto& where,const auto& record,const auto&,const auto&){const auto path=where/"libraries"/version/(record.vertex_identity+".library");const auto moved=path.string()+".original";std::filesystem::rename(path,moved);std::filesystem::create_symlink(moved,path);});
    rejected("library-directory-symlink",[&](const auto& where,const auto&,const auto&,const auto&){std::filesystem::rename(where/"libraries"/version,where/"libraries"/"original");std::filesystem::create_directory_symlink(where/"libraries"/"original",where/"libraries"/version);});
    std::size_t remaining=0;
    Check(manifest.LoadHistorical(dir,stop,limited,4096,&remaining).empty()&&limited&&!remaining,"Historical visit budget was ignored");
    Check(manifest.LoadHistorical(dir,stop,limited,2).size()==2&&limited,"Historical record budget was ignored");
    stop.store(true);Check(manifest.LoadHistorical(dir,stop,limited).empty()&&!limited,"Initially stopped historical load touched files");stop.store(false);
    // Sparse invalid libraries exercise the real cumulative read cap without allocating huge fixture strings.
    const auto pressure=root/"read-budget";
    for(unsigned i=0;i<17;++i) {
        const auto id=Metal::ShaderDiskCache::Identity(std::as_bytes(std::span{&i,1}));
        Metal::RenderPipelineRecord record{id,"manifest_vertex","","",MTLPixelFormatRGBA8Unorm,{}};
        OldPipeline(pressure,version,OldRecord(version,record));
        const auto path=pressure/"libraries"/version/(id+".library");std::filesystem::create_directories(path.parent_path());
        std::ofstream sparse(path,std::ios::binary);sparse.seekp(16*1024*1024-1);sparse.put('\0');Check(bool(sparse),"Sparse budget fixture failed");
    }
    Check(manifest.LoadHistorical(pressure,stop,limited).empty()&&limited,"Historical source read budget was ignored");
    std::cout<<"Historical native hints:78 descriptors/2 old versions ->39 CURRENT PSOs; all4992 GPU pixel bytes checked; first draws39 ready hits/0 compiles; full name/source isolation, strict current loaders, title/cancel/clear/disable,10 rejection cases and visit/record/256MiB read budgets passed\n";
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Check(argc==3,"Specify cold/warm/edge, mesh-* or historical and an isolated absolute manifest root");
            const std::string mode{argv[1]};const std::filesystem::path root{argv[2]};
            Check(root.is_absolute(),"Manifest test root must be absolute");
            if(mode=="historical"){HistoricalTest(root);return 0;}
            if(mode.starts_with("mesh-")){MeshManifestTest(mode,root);return 0;}
            if(mode=="cold")Check(Files(root).empty(),"Cold manifest test requires a fresh root");
            if(mode=="edge") {
                auto files=Files(root);
                auto damaged_state=States().front();
                const auto damaged_key=Metal::RenderPipelineManifest::Identity({
                    Metal::Device::SourceIdentity("Manifest vertex fixture",Vertex),"manifest_vertex",
                    Metal::Device::SourceIdentity("Manifest fragment fixture",Fragment),"manifest_fragment",
                    MTLPixelFormatRGBA8Unorm,damaged_state});
                std::erase_if(files,[&](const auto& path){return
                    path.parent_path().parent_path().parent_path().filename()!="000000000000007b" || path.stem()!=damaged_key;});
                Check(!files.empty(),"Manifest corruption needs isolated cold/warm files");
                std::fstream file(files[0],std::ios::binary|std::ios::in|std::ios::out);
                char byte{};file.seekg(80);file.read(&byte,1);byte^=0x55;file.seekp(80);file.write(&byte,1);file.close();
                std::filesystem::create_directory(files[0].parent_path()/".pending-interrupted");
                std::filesystem::create_symlink(files[0],files[0].parent_path()/(std::string(64,'e')+".pipeline"));
            }
            Metal::Device device{root};device.SwitchDiskCache(123);
            id<MTLLibrary> vertex=nil,fragment=nil;
            if(mode=="cold") {
                vertex=device.CompileLibrary(Vertex,"Manifest vertex fixture");
                fragment=device.CompileLibrary(Fragment,"Manifest fragment fixture");
            }
            Metal::PipelineCache cache{device};const auto states=States();
            auto existing_files=Files(root);
            std::erase_if(existing_files,[](const auto& path){return
                path.parent_path().parent_path().parent_path().filename()!="000000000000007b" ||
                std::filesystem::is_symlink(std::filesystem::symlink_status(path));});
            std::atomic_bool stop{false};
            Metal::PipelinePrewarmResult ready{};
            if(mode!="cold") {
                ready=cache.Prewarm(stop,[&](std::size_t,std::size_t){(void)cache.Stats();});
                Check(!ready.failed && !ready.cancelled && !ready.budget_limited,"Native manifest preparation failed");
                if(mode=="warm")Check(ready.prepared==existing_files.size(),"Fresh process failed to prepare every persisted native PSO");
                else Check(ready.prepared==existing_files.size()-1 && cache.Stats().rejected>=2,
                           "Corrupt/symlink manifest was used or valid neighboring PSOs were lost");
                Check(device.DiskCacheStats().compiled_pipelines==0,"Native archive preparation recompiled valid PSOs");
                vertex=device.LibraryWithIdentity(Metal::Device::SourceIdentity("Manifest vertex fixture",Vertex));
                fragment=device.LibraryWithIdentity(Metal::Device::SourceIdentity("Manifest fragment fixture",Fragment));
            }
            Check(device.LibraryIdentity(vertex) && device.LibraryIdentity(fragment),"Native library source identities missing");
            const auto before=cache.Stats().render_builds;
            for(const auto& state:states) Draw(device,cache,{vertex,"manifest_vertex",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,state});
            if(mode=="warm")Check(cache.Stats().prewarm_hits==states.size() && cache.Stats().render_builds==before,
                                  "First prepared draws repeated render descriptor/PSO creation");
            if(mode=="edge")Check(cache.Stats().prewarm_hits==states.size()-1 && cache.Stats().render_builds==before+1,
                                  "Corrupt manifest fallback did not repair only the missing ready pipeline");
            Check(device.FlushDiskCache(),"Native binary flush after manifest GPU draws failed");
            if(mode=="cold")Check(Files(root).size()==states.size(),"Render manifests did not cover all complete descriptor variants");
            cache.Clear();
            stop.store(true);const auto cancelled=cache.Prewarm(stop);
            Check(cancelled.cancelled && !cancelled.visited && !cancelled.prepared,"Initially cancelled PSO preparation touched records");
            stop.store(false);
            const auto partial=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(done==1)stop.store(true);});
            Check(partial.cancelled && partial.prepared==1 && partial.visited==1,"Partial PSO cancellation prepared a later record");
            cache.Clear();stop.store(false);
            const auto cleared=cache.Prewarm(stop,[&](std::size_t done,std::size_t){if(!done)cache.Clear();});
            Check(cleared.cancelled && !cleared.prepared,"Callback cache clear failed to invalidate render prewarm epoch");
            cache.Clear();device.SwitchDiskCache(mode=="cold"?456:mode=="warm"?556:656);
            Check(!cache.Prewarm(stop).candidates,"Render manifest leaked into another title");
            cache.Clear();device.SwitchDiskCache(123);
            Check(cache.Prewarm(stop).prepared==(mode=="cold"?states.size():existing_files.size()),
                  "Returning to a title lost complete native PSOs");
            cache.Clear();device.SetDiskCacheEnabled(false);
            Check(!cache.Prewarm(stop).candidates,"Disabled disk cache still preloaded render manifests");
            Draw(device,cache,{vertex,"manifest_vertex",fragment,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}});
            device.SetDiskCacheEnabled(true);
            const auto count=device.DiskCacheStats().compiled_pipelines;
            const auto values=mode=="cold"?".5,.75,.125,1":mode=="warm"?".75,.5,.125,1":".125,.5,.75,1";
            const auto changed_source=std::string("#include <metal_stdlib>\nusing namespace metal;fragment float4 manifest_fragment(){return float4(")+values+");}";
            auto changed=device.CompileLibrary(changed_source,"Manifest fragment fixture");
            Check(device.LibraryIdentity(changed)!=device.LibraryIdentity(fragment),"Same library name hid changed source identity");
            const std::array<unsigned,4> expected_changed=mode=="cold"?std::array<unsigned,4>{128,191,32,255}:
                mode=="warm"?std::array<unsigned,4>{191,128,32,255}:std::array<unsigned,4>{32,128,191,255};
            Draw(device,cache,{vertex,"manifest_vertex",changed,"manifest_fragment",MTLPixelFormatRGBA8Unorm,{}},expected_changed);
            Check(device.DiskCacheStats().compiled_pipelines==count+1,"Changed source reused an incompatible ready/archive render PSO");
            Check(device.FlushDiskCache(),"Native final manifest archive flush failed");
            std::cout<<"19 native render manifest variants: all16 color masks, NoOp, hardware blend and min/max-emulation descriptor state; every32 GPU pixel output passed\n";
            if(mode=="warm")std::cout<<"Fresh-process full render PSO preparation: "<<ready.prepared<<" native archive hits/0 PSO recompiles; first draws 19 ready hits/0 render builds\n";
            if(mode=="edge")std::cout<<"Native checksum/symlink/interrupted manifest rejection and on-demand repair preserve neighboring PSOs/GPU pixels\n";
            std::cout<<"Native render manifest initial/mid cancel, reentrant stats, clear epoch, title return/isolation, disabled cache and changed-source cases passed; no app/window/ROM/emulation started\n";
            return 0;
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
