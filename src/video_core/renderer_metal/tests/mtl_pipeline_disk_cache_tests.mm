// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_device.h"
#include "mtl_scheduler.h"
#include "mtl_transfer.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>

namespace {
void Check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
id<MTLComputePipelineState> Compute(Metal::Device& device,unsigned value,bool validation=false) {
    const auto source=std::string("#include <metal_stdlib>\nusing namespace metal;\n")+
        "kernel void disk_compute(device uint* out [[buffer(0)]],uint i [[thread_position_in_grid]]){out[i]=i*7+"+
        std::to_string(value)+"u;}";
    auto library=device.CompileLibrary(source,"Metal4 disk compute fixture");
    auto descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"disk_compute");
    descriptor.maxTotalThreadsPerThreadgroup=64;
    descriptor.label=[NSString stringWithFormat:@"disk-compute-%u",value];
    if(validation){descriptor.options=[[MTL4PipelineOptions alloc] init];descriptor.options.shaderValidation=MTLShaderValidationEnabled;}
    return device.CompileCompute(descriptor);
}
void RunCompute(Metal::Device& device,id<MTLComputePipelineState> pipeline,unsigned value) {
    Metal::Scheduler scheduler{device};auto output=device.CreateBuffer(16*sizeof(unsigned),true);
    std::memset(output.contents,0xff,output.length);
    const std::array buffers{Metal::BufferBinding{output,0,0}};
    auto args=scheduler.MakeArguments(buffers);scheduler.KeepAlive(pipeline);
    auto encoder=scheduler.BeginCompute();[encoder setComputePipelineState:pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];scheduler.EndEncoding();scheduler.Finish();
    for(unsigned i=0;i<16;++i)Check(static_cast<unsigned*>(output.contents)[i]==i*7+value,
        "Cold/warm/changed Metal4 archive compute returned incorrect GPU bytes");
}
void Render(Metal::Device& device) {
    constexpr std::string_view source=R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 disk_vertex(uint i [[vertex_id]]) {
    const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);
}
fragment float4 disk_fragment(){return float4(1,.25,.5,1);}
)";
    auto library=device.CompileLibrary(source,"Metal4 disk render fixture");
    auto descriptor=[[MTL4RenderPipelineDescriptor alloc] init];
    descriptor.vertexFunctionDescriptor=Metal::Device::Function(library,"disk_vertex");
    descriptor.fragmentFunctionDescriptor=Metal::Device::Function(library,"disk_fragment");
    descriptor.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
    descriptor.label=@"disk-render";
    auto pipeline=device.CompileRender(descriptor);
    auto* texture_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:4 mipmapped:NO];
    texture_descriptor.storageMode=MTLStorageModePrivate;texture_descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    auto texture=device.CreateTexture(texture_descriptor);
    Metal::Scheduler scheduler{device};Metal::TransferRuntime transfers{device,scheduler};
    auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=texture;
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    scheduler.KeepAlive(pipeline);auto encoder=scheduler.BeginRender(pass);[encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{0,0,8,4,0,1}];[encoder setScissorRect:MTLScissorRect{0,0,8,4}];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];scheduler.EndEncoding();
    const auto pixels=transfers.Readback({texture,0,0,MTLRegionMake2D(0,0,8,4)},8*4);
    constexpr std::array<unsigned,4> expected{255,64,128,255};
    for(unsigned i=0;i<pixels.size();++i)Check(std::abs(int(std::to_integer<unsigned>(pixels[i]))-int(expected[i%4]))<=1,
        "Cold/warm Metal4 archive render returned incorrect GPU pixels");
}
std::vector<std::filesystem::path> Archives(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> result;std::error_code error;
    for(std::filesystem::recursive_directory_iterator it(root,error),end;!error&&it!=end;it.increment(error))
        if(it->is_regular_file(error)&&it->path().filename()=="pipelines.metalar")result.push_back(it->path());
    return result;
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Check(argc==3,"Specify cold/warm/async/edge mode and absolute isolated temporary cache root");
            const std::string mode=argv[1];const std::filesystem::path root=argv[2];
            Check(root.is_absolute(),"Disk cache test root must be absolute");
            if(mode=="async") {
                Check(Archives(root).empty(),"Async disk test requires an isolated empty cache");
                {
                    Metal::Device device{root};
                    Check(device.DiskCacheStats().enabled,"Async persistence test needs native binary capture");
                    RunCompute(device,Compute(device,3001),3001);Render(device);
                    const auto shader=device.ShaderDiskCacheDirectory();
                    const auto render=device.RenderDiskCacheDirectory();
                    for(unsigned i=0;i<256;++i) {
                        device.RequestDiskCacheFlush();
                        device.SetDiskCacheEnabled(true);
                        Check(device.ShaderDiskCacheDirectory()==shader && device.RenderDiskCacheDirectory()==render,
                              "Background publication changed live cache directories");
                    }
                    RunCompute(device,Compute(device,3002),3002);
                    device.RequestDiskCacheFlush();
                    device.SwitchDiskCache(0xab);
                    Check(device.RenderDiskCacheDirectory()!=render,"Async title switch kept the old cache directory");
                    RunCompute(device,Compute(device,3003),3003);
                    device.RequestDiskCacheFlush();
                    device.SwitchDiskCache(0);
                    const auto hits=device.DiskCacheStats().archive_hits;
                    RunCompute(device,Compute(device,3001),3001);Render(device);
                    RunCompute(device,Compute(device,3002),3002);
                    Check(device.DiskCacheStats().archive_hits==hits+3,
                          "Background save/title switch lost or corrupted the common native pipelines");
                    device.SetDiskCacheEnabled(false);
                    Check(device.ShaderDiskCacheDirectory().empty() && device.RenderDiskCacheDirectory().empty(),
                          "Disabled async cache retained published paths");
                    device.RequestDiskCacheFlush();
                    device.SetDiskCacheEnabled(true);
                    Check(device.ShaderDiskCacheDirectory()==shader && device.RenderDiskCacheDirectory()==render,
                          "Async cache re-enable did not restore paths");
                    RunCompute(device,Compute(device,3004),3004);
                    device.RequestDiskCacheFlush(); // destruction must drain/join and persist this last PSO
                }
                {
                    Metal::Device device{root};
                    RunCompute(device,Compute(device,3004),3004);
                    Check(device.DiskCacheStats().archive_hits==1 && device.DiskCacheStats().compiled_pipelines==0,
                          "Shutdown lost a queued background cache save");
                    device.SwitchDiskCache(0xab);
                    const auto hits=device.DiskCacheStats().archive_hits;
                    RunCompute(device,Compute(device,3003),3003);
                    Check(device.DiskCacheStats().archive_hits==hits+1,
                          "Async publication mixed title-specific binaries");
                }
                std::cout<<"Async native Metal4 persistence: GPU bytes/pixels, coalesced requests, title isolation, settings and shutdown reload passed\n";
            } else if(mode=="cold") {
                Check(Archives(root).empty(),"Cold disk test requires a root without prior archive batches");
                Metal::Device device{root};Check(device.DiskCacheStats().enabled,"Native Metal4 serializer cache was not enabled");
                RunCompute(device,Compute(device,3),3);Render(device);
                if(!device.FlushDiskCache())throw std::runtime_error("Unable to serialize native Metal4 archive: "+device.DiskCacheStats().last_error);
                RunCompute(device,Compute(device,11),11);
                Check(device.FlushDiskCache(),"Unable to serialize incremental native Metal4 archive batch");
                const auto stats=device.DiskCacheStats();
                Check(stats.compiled_pipelines==3&&stats.saved_batches==2&&Archives(root).size()==2,
                      "Native Metal4 cold capture lost compute/render or incremental batches");
                Check(device.FlushDiskCache()&&device.DiskCacheStats().saved_batches==2,"Empty cache flush rewrote a batch");
                std::cout<<"Cold native Metal4 serializer: compute/render GPU results, two immutable atomic batches, incremental capture and no-op flush passed\n";
            } else if(mode=="warm") {
                Metal::Device device{root};
                if(device.DiskCacheStats().loaded_batches<2)throw std::runtime_error("Warm archive load: rejected="+
                    std::to_string(device.DiskCacheStats().rejected_batches)+", "+device.DiskCacheStats().last_error);
                RunCompute(device,Compute(device,3),3);Render(device);RunCompute(device,Compute(device,11),11);
                auto stats=device.DiskCacheStats();
                Check(stats.archive_hits==3&&stats.compiled_pipelines==0,"Fresh-process native Metal4 archive lookup did not reuse all three PSOs");
                RunCompute(device,Compute(device,99),99);
                Check(device.DiskCacheStats().compiled_pipelines==1,"Changed shader source reused an incompatible native binary");
                std::atomic_bool stop=true;const auto previous=device.DiskCacheStats().title_id;
                device.SwitchDiskCache(123,&stop);Check(device.DiskCacheStats().title_id==previous,"Cancelled title switch changed the native cache");
                device.SwitchDiskCache(123);RunCompute(device,Compute(device,77),77);device.SwitchDiskCache(456);
                const auto count=device.DiskCacheStats().compiled_pipelines;RunCompute(device,Compute(device,77),77);
                Check(device.DiskCacheStats().compiled_pipelines==count+1,"Native archive title isolation failed");
                device.SwitchDiskCache(123);const auto hits=device.DiskCacheStats().archive_hits;
                RunCompute(device,Compute(device,77),77);Check(device.DiskCacheStats().archive_hits==hits+1,"Returning to a title lost its archived PSO");
                const auto saved=device.DiskCacheStats().saved_batches;
                RunCompute(device,Compute(device,8080,true),8080);device.FlushDiskCache();
                Check(device.DiskCacheStats().saved_batches==saved,"Per-pipeline Shader Validation was captured into a binary archive");
                device.SetDiskCacheEnabled(false);const auto disabled_hits=device.DiskCacheStats().archive_hits;
                RunCompute(device,Compute(device,8081),8081);device.FlushDiskCache();
                Check(!device.DiskCacheStats().enabled&&device.DiskCacheStats().saved_batches==saved&&
                      device.DiskCacheStats().archive_hits==disabled_hits,"Runtime cache disable still looked up or wrote binaries");
                device.SetDiskCacheEnabled(true);RunCompute(device,Compute(device,77),77);
                Check(device.DiskCacheStats().archive_hits==disabled_hits+1,"Runtime re-enable did not restore a title's binary lookup");
                std::cout<<"Native archive per-pipeline validation bypass and runtime disable/re-enable preserve GPU output and existing binaries\n";
                std::cout<<"Warm native Metal4 archive: fresh-process compute/render direct hits, changed-source native miss, cancellation and title isolation/return passed\n";
            } else if(mode=="bounds") {
                Metal::Device device{root};
                auto files=Archives(root);Check(files.size()==2,"Bounds test requires only two cold common batches");
                const auto common=files.front().parent_path().parent_path();
                const auto environment=common.parent_path();
                device.SwitchDiskCache(0xb0);
                RunCompute(device,Compute(device,77),77);RunCompute(device,Compute(device,78),78);
                Check(device.FlushDiskCache(),"Unable to publish title fixture before bounding native archives");
                const auto title=environment/"00000000000000b0";
                const auto title_seed=Archives(title).front().parent_path();
                const auto common_seed=files.front().parent_path();
                const auto time=std::filesystem::file_time_type::clock::now();
                const auto clone=[&](const std::filesystem::path& source,const std::filesystem::path& dest,int age) {
                    const auto name=std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String;
                    const auto path=dest/name;std::filesystem::create_directory(path);
                    std::filesystem::copy_file(source/"pipelines.metalar",path/"pipelines.metalar");
                    std::filesystem::copy_file(source/"sha256",path/"sha256");
                    std::filesystem::last_write_time(path,time+std::chrono::seconds(age));return path;
                };
                for(unsigned i=0;i<69;++i){clone(common_seed,common,int(i));clone(title_seed,title,200+int(i));}
                const auto protected_batch=common/(std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String);
                std::filesystem::create_directory(protected_batch);
                std::filesystem::copy_file(common_seed/"pipelines.metalar",protected_batch/"pipelines.metalar");
                std::filesystem::copy_file(common_seed/"sha256",protected_batch/"sha256");
                {std::ofstream note(protected_batch/"user-note");note<<"preserve unknown content";}
                const auto pending=common/".pending-keep";std::filesystem::create_directory(pending);
                {std::ofstream note(pending/"user-note");note<<"preserve interrupted content";}
                const auto external=root/"external-keep";std::filesystem::create_directory(external);
                {std::ofstream note(external/"user-note");note<<"preserve linked target";}
                const auto linked=common/(std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String);
                std::filesystem::create_directory_symlink(external,linked);
                device.SwitchDiskCache(0xb0);
                auto stats=device.DiskCacheStats();
                Check(stats.resident_batches==64 && stats.resident_archive_bytes<=256ULL*1024*1024,
                      "Common/title archives exceeded their shared native resident bound");
                Check(stats.pruned_batches>=13,"Native archive disk count did not evict old managed batches");
                const auto hits=stats.archive_hits;RunCompute(device,Compute(device,77),77);
                Check(device.DiskCacheStats().archive_hits==hits+1,"Common archive count starved the active title's native pipeline");
                // Keep valid common archives newer than every live title archive.
                // Flush must prune the old files while retained native objects remain usable.
                const auto common_source=Archives(common).front().parent_path();
                for(unsigned i=0;i<128;++i)clone(common_source,common,500+int(i));
                RunCompute(device,Compute(device,88),88);Check(device.FlushDiskCache(),"Bounded capture flush failed");
                Check(!std::filesystem::exists(title_seed),"Old loaded title archive was not evicted from disk");
                const auto live_hits=device.DiskCacheStats().archive_hits;
                RunCompute(device,Compute(device,78),78);
                Check(device.DiskCacheStats().archive_hits==live_hits+1,
                      "Retained native archive could not create a previously unused PSO after its disk eviction");
                Check(std::filesystem::exists(protected_batch/"user-note") &&
                      std::filesystem::exists(protected_batch/"pipelines.metalar") &&
                      std::filesystem::exists(pending/"user-note") &&
                      std::filesystem::is_symlink(linked) && std::filesystem::exists(external/"user-note"),
                      "Native archive pruning removed unknown, pending or linked content");
                const auto oversized=common/(std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String);
                std::filesystem::create_directory(oversized);
                {std::ofstream file(oversized/"pipelines.metalar",std::ios::binary);
                    file.seekp(512ULL*1024*1024);file.put(0);}
                {std::ofstream checksum(oversized/"sha256");checksum<<std::string(64,'0');}
                std::filesystem::last_write_time(oversized,time+std::chrono::seconds(1000));
                device.SwitchDiskCache(0xb0);
                Check(!std::filesystem::exists(oversized),"Newest oversized native archive escaped the disk byte bound");
                Check(device.DiskCacheStats().resident_batches<=64 &&
                      device.DiskCacheStats().resident_archive_bytes<=256ULL*1024*1024,
                      "Switch after archive eviction escaped resident limits");
                RunCompute(device,Compute(device,78),78);
                std::cout<<"Native archive bounds: shared common/title64-batch/256MiB load, active-title priority, environment128-batch/512MiB disk pruning, retained archive PSO/GPU execution after unlink, oversized newest rejection and unknown/pending/symlink preservation passed\n";
            } else if(mode=="inventory") {
                Metal::Device device{root};
                const auto originals=Archives(root);Check(originals.size()==2,"Inventory test requires two cold batches");
                RunCompute(device,Compute(device,77),77);Check(device.FlushDiskCache(),"Inventory seed capture failed");
                const auto files=Archives(root);
                const auto captured=std::find_if(files.begin(),files.end(),[&](const auto& file) {
                    return std::find(originals.begin(),originals.end(),file)==originals.end();
                });
                Check(captured!=files.end(),"Inventory seed archive was not published");
                const auto seed=*captured;
                const auto common=seed.parent_path().parent_path(),environment=common.parent_path();
                const auto flood=environment/"00000000000000b1";
                std::filesystem::create_directory(flood);
                const auto time=std::filesystem::file_time_type::clock::now();
                std::vector<std::filesystem::path> clones;
                for(unsigned i=0;i<140;++i) {
                    const auto path=common/(std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String);
                    std::filesystem::create_directory(path);
                    std::filesystem::copy_file(seed,path/"pipelines.metalar");
                    std::filesystem::copy_file(seed.parent_path()/"sha256",path/"sha256");
                    std::filesystem::last_write_time(path,time+std::chrono::seconds(100+i));clones.push_back(path);
                }
                for(std::size_t i=0;i<Metal::PipelineDiskCache::ScanEntryLimit+32;++i) {
                    std::ofstream file(flood/("unknown-"+std::to_string(i)));file<<"preserve";
                    Check(bool(file),"Unable to create bounded inventory fixture");
                }
                const auto before=device.DiskCacheStats();device.SwitchDiskCache(0xb1);
                const auto limited=device.DiskCacheStats();
                Check(limited.scan_limit_hits>=before.scan_limit_hits+2 &&
                      limited.scanned_entries-before.scanned_entries<=2*Metal::PipelineDiskCache::ScanEntryLimit &&
                      limited.pruned_batches==before.pruned_batches && limited.resident_batches==0,
                      "Native environment/title scans exceeded bounds or pruned an incomplete inventory");
                for(const auto& path:clones)Check(std::filesystem::exists(path/"pipelines.metalar"),
                    "Incomplete native inventory deleted a managed batch");
                const auto compiled=limited.compiled_pipelines;RunCompute(device,Compute(device,77),77);
                Check(device.DiskCacheStats().compiled_pipelines==compiled+1 && device.FlushDiskCache(),
                      "Bounded inventory did not preserve native compilation and publication");
                std::atomic_bool stop=true;const auto title=device.DiskCacheStats().title_id;
                device.SwitchDiskCache(0,&stop);
                Check(device.DiskCacheStats().title_id==title,"Cancelled bounded scan changed the title");
                Metal::PipelineDiskCache cancelled{device.Handle(),root,false};
                Check(!cancelled.Flush(&stop) && cancelled.Stats().cancelled_scans==1,
                      "Cancelled native cache flush did not stop before serializer/filesystem work");
                for(std::size_t i=0;i<Metal::PipelineDiskCache::ScanEntryLimit+32;++i)
                    Check(std::filesystem::remove(flood/("unknown-"+std::to_string(i))),"Inventory fixture disappeared");
                const auto pruning=device.DiskCacheStats().pruned_batches;device.SwitchDiskCache(0);
                Check(device.DiskCacheStats().pruned_batches>pruning && device.DiskCacheStats().resident_batches==64,
                      "Complete inventory did not resume quotas and native archive loading");
                const auto hits=device.DiskCacheStats().archive_hits;RunCompute(device,Compute(device,77),77);
                Check(device.DiskCacheStats().archive_hits==hits+1,"Inventory recovery lost the archived native PSO");
                std::cout<<"Native archive inventory: global4096-entry prune and shared title/common4096-entry load limits, incomplete-inventory zero eviction, native GPU compilation/publication, cancellation and complete-inventory quota/archive recovery passed\n";
            } else if(mode=="hashbounds") {
                Metal::Device device{root};
                const auto files=Archives(root);Check(files.size()==2,"Hash bounds test requires two cold batches");
                const auto common=files.front().parent_path().parent_path();
                std::vector<std::filesystem::path> corrupt;
                const auto time=std::filesystem::file_time_type::clock::now();
                for(unsigned i=0;i<3;++i) {
                    const auto path=common/(std::string("batch-")+NSUUID.UUID.UUIDString.UTF8String);
                    std::filesystem::create_directory(path);
                    {std::ofstream file(path/"pipelines.metalar",std::ios::binary);
                        file.seekp(128ULL*1024*1024-1);file.put(0);Check(bool(file),"Unable to create sparse hash fixture");}
                    {std::ofstream checksum(path/"sha256");checksum<<std::string(64,'0');}
                    std::filesystem::last_write_time(path,time+std::chrono::seconds(100+i));corrupt.push_back(path);
                }
                const auto before=device.DiskCacheStats();device.SwitchDiskCache(0);
                const auto bounded=device.DiskCacheStats();
                Check(bounded.hashed_archive_bytes-before.hashed_archive_bytes==256ULL*1024*1024 &&
                      bounded.rejected_batches>=before.rejected_batches+3 && bounded.resident_batches==0,
                      "Checksum-invalid archives escaped the shared256MiB hashing budget");
                RunCompute(device,Compute(device,3),3);
                for(const auto& path:corrupt)std::filesystem::remove_all(path);
                device.SwitchDiskCache(0);const auto hits=device.DiskCacheStats().archive_hits;
                RunCompute(device,Compute(device,11),11);
                Check(device.DiskCacheStats().archive_hits==hits+1,"Hash-budget recovery lost a valid native pipeline");
                std::cout<<"Native archive hash bounds: three checksum-invalid128MiB files consume at most256MiB total per switch, native GPU miss fallback and valid archive recovery passed\n";
            } else if(mode=="edge") {
                auto files=Archives(root);
                std::erase_if(files,[](const auto& path){return path.parent_path().parent_path().filename()!="0000000000000000";});
                Check(!files.empty(),"Corruption test requires an isolated previously captured common archive");
                std::fstream file(files[0],std::ios::binary|std::ios::in|std::ios::out);
                char value{};file.read(&value,1);value^=0x7f;file.seekp(0);file.write(&value,1);file.close();
                const auto title=files[0].parent_path().parent_path();
                std::filesystem::create_directories(title/".pending-interrupted-test");
                Check(files.size()>=2,"Metadata test requires another intact common archive");
                const auto metadata=title/"batch-symlink-metadata";
                std::filesystem::create_directory(metadata);
                std::filesystem::copy_file(files[1],metadata/"pipelines.metalar");
                std::filesystem::create_symlink(files[1].parent_path()/"sha256",metadata/"sha256");
                std::filesystem::create_directory_symlink(files[1].parent_path(),title/"batch-symlink-directory");
                Metal::Device device{root};Check(device.DiskCacheStats().rejected_batches>=2,"Corrupted archive or symlink metadata was not rejected before native loading");
                RunCompute(device,Compute(device,1234),1234);
                const auto blocked=root/"blocked-root";{std::ofstream output(blocked);output<<"not a directory";}
                Metal::Device unavailable{blocked};Check(!unavailable.DiskCacheStats().enabled,"Unwritable cache location did not disable optional persistence");
                RunCompute(unavailable,Compute(unavailable,5678),5678);
                Metal::Device disabled;RunCompute(disabled,Compute(disabled,9012),9012);
                Check(!disabled.DiskCacheStats().enabled,"Default standalone device unexpectedly wrote a disk cache");
                std::cout<<"Native Metal4 cache checksum corruption/symlink metadata/interrupted publication, unavailable storage and disabled-cache native compilation passed\n";
            } else if(mode=="validation") {
                Metal::Device device{root};Check(!device.DiskCacheStats().enabled,"Shader-validated device loaded or persisted uninstrumented pipeline binaries");
                RunCompute(device,Compute(device,2026),2026);Render(device);
                std::cout<<"Native Metal4 cache bypass under Shader Validation preserves instrumented compute/render GPU execution\n";
            } else throw std::invalid_argument("Unknown native disk cache test mode");
            std::cout<<"No app/window/ROM/emulation started; isolated cache root "<<root<<'\n';return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
}
