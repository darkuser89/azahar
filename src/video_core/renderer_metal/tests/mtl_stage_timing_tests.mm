// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_scheduler.h"
#include "mtl_transfer.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace Metal {
// Same native provider proof used by scheduler regressions. These fixtures
// have immutable, disjoint input ranges and no fragment stores except the
// explicitly conservative storage control.
struct StageTimingTestAccess {
    static ReadOnlyBufferRenderProof ReadOnlyBuffers() { return {}; }
};
}
namespace {
void Check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
unsigned Value(unsigned value) {for(unsigned i=0;i<64;++i)value=value*1664525u+1013904223u;return value;}
}

// Whole-submission GPU execution times from Metal4 commit feedback. No GPU
// counter/timestamp commands are inserted between encoders, so sampling does
// not add dependencies that would erase the overlap under comparison.
// This measures the complete synchronization policy's effect on identical
// work; it cannot assign an intrinsic nanosecond cost to one barrier API call.
void ValidateStageTiming(Metal::Device& device,const std::filesystem::path& output) {
    Check(output.is_absolute(),"Stage timing requires an absolute output path");
    Check(!std::getenv("AZAHAR_METAL4_PROFILE"),"Stage timing requires GPU encoder profiling to be disabled");
    std::optional<std::filesystem::path> capture_directory;
    if(const auto* path=std::getenv("AZAHAR_METAL4_STAGE_CAPTURE");path && *path) {
        capture_directory=path;
        Check(capture_directory->is_absolute() && std::filesystem::is_directory(*capture_directory),
              "Stage capture requires an existing absolute temporary directory");
        Check([[MTLCaptureManager sharedCaptureManager] supportsDestination:MTLCaptureDestinationGPUTraceDocument],
              "Native Metal GPU trace capture is unavailable");
    }
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
struct V {float4 p [[position]];uint value [[flat]];};
vertex V timing_vertex(uint id [[vertex_id]],constant uint4& config [[buffer(0)]]) {
    id%=3;uint value=config.x;
    for(uint i=0;i<64;++i)value=value*1664525u+1013904223u;
    return {float4(id==1 ? 3.0f:-1.0f,id==2 ? 3.0f:-1.0f,0,1),value};
}
fragment uint timing_fragment(V v [[stage_in]],uint previous [[color(0),raster_order_group(0)]]) {
    uint2 p=uint2(v.p.xy);return previous+(v.value^((p.x+1)*131u+(p.y+1)*37u));
}
fragment uint timing_storage(V v [[stage_in]],uint previous [[color(0),raster_order_group(0)]],
                             device atomic_uint* output [[buffer(1)]]) {
    uint2 p=uint2(v.p.xy);atomic_store_explicit(output+p.y*128+p.x,v.value,memory_order_relaxed);
    return previous+(v.value^((p.x+1)*131u+(p.y+1)*37u));
}
kernel void timing_produce(constant uint4& input [[buffer(0)]],device uint4& output [[buffer(1)]]) {
    output=input;
}
)","Native identical-work stage-policy timing fixture");
    auto* rd=[[MTL4RenderPipelineDescriptor alloc] init];
    rd.vertexFunctionDescriptor=Metal::Device::Function(library,"timing_vertex");
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"timing_fragment");
    rd.colorAttachments[0].pixelFormat=MTLPixelFormatR32Uint;
    const auto render=device.CompileRender(rd);
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"timing_storage");
    const auto storage_render=device.CompileRender(rd);
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"timing_produce");
    const auto compute=device.CompileCompute(cd);
    constexpr unsigned width=128,height=64,draws=32,triangles=4;
    const auto proof=Metal::StageTimingTestAccess::ReadOnlyBuffers();
    std::ofstream csv{output};Check(bool(csv),"Unable to create stage timing CSV");
    csv<<"path,round,order,mode,tick,gpu_begin_seconds,gpu_end_seconds,gpu_ms,host_encode_ms,submit_host_ms,encoders,draws,dispatches,buffer_passes,conservative_passes,pixels_verified,storage_words_verified\n";
    unsigned samples{},pixels_verified{};
    const std::array<const char*,5> paths{"direct_render","queued_render_reuse","compute_render","storage_control","buffer_alias_control"};
    for(unsigned path=0;path<paths.size();++path) for(int round=-2;round<12;++round) {
        // Alternating ABBA/BAAB order limits monotonic thermal/cache drift.
        const std::array<bool,4> order=round%2 ? std::array{true,false,false,true}:std::array{false,true,true,false};
        std::vector<std::byte> paired;
        for(unsigned order_index=0;order_index<order.size();++order_index) {
            if(capture_directory && (path>2 || round!=0 || order_index>1))continue;
            const bool narrow=order[order_index];
            Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
            auto input=device.CreateBuffer(draws*16,true),produced=device.CreateBuffer(draws*16,false);
            auto storage=device.CreateBuffer(width*height*4,true);
            std::memset(storage.contents,0xa7,storage.length);
            for(unsigned i=0;i<draws;++i) {
                const std::array<unsigned,4> config{unsigned(97+round)*37u+i*7919u,0,0,0};
                std::memcpy(static_cast<std::byte*>(input.contents)+i*16,config.data(),16);
            }
            auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:width height:height mipmapped:NO];
            td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
            id<MTLTexture> target;
            id<MTLBuffer> backing;
            if(path==4) {
                const auto alignment=[device.Handle() minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Uint];
                const auto pitch=(width*4+alignment-1)/alignment*alignment;
                backing=device.CreateBuffer(pitch*height,true);td.storageMode=MTLStorageModeShared;
                target=[backing newTextureWithDescriptor:td offset:0 bytesPerRow:pitch];
                Check(target!=nil,"Unable to create native buffer-backed alias-control attachment");
            }else target=device.CreateTexture(td);
            transfer.ClearColor(target,MTLClearColorMake(0,0,0,0));queue.Finish();
            auto* descriptor=[[MTL4RenderPassDescriptor alloc] init];
            descriptor.colorAttachments[0].texture=target;
            descriptor.colorAttachments[0].loadAction=MTLLoadActionLoad;
            descriptor.colorAttachments[0].storeAction=MTLStoreActionStore;
            const auto pass=queue.SnapshotRenderPass(descriptor);
            const auto promise=std::make_shared<std::promise<Metal::Scheduler::SubmissionTiming>>();
            auto timing=promise->get_future();
            queue.SetSubmissionObserver([promise](const auto& value){promise->set_value(value);});
            const auto before=queue.WorkerStats();const auto encoders=queue.EncoderCount();
            if(capture_directory) {
                const auto trace=*capture_directory/(std::string{paths[path]}+(narrow ? "-narrow.gputrace":"-broad.gputrace"));
                Check(!std::filesystem::exists(trace),"Refusing to replace a previous GPU trace");
                auto* capture=[[MTLCaptureDescriptor alloc] init];
                capture.captureObject=device.Handle();capture.destination=MTLCaptureDestinationGPUTraceDocument;
                capture.outputURL=[NSURL fileURLWithPath:[NSString stringWithUTF8String:trace.c_str()]];
                NSError* error=nil;
                if(![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:capture error:&error])
                    throw std::runtime_error(error.localizedDescription.UTF8String ?: "Unable to capture native Metal stage workload");
            }
            const auto encode_begin=std::chrono::steady_clock::now();
            for(unsigned i=0;i<draws;++i) {
                const std::array bindings{Metal::BufferBinding{path==2 ? produced:input,i*16,0},
                                          Metal::BufferBinding{storage,0,1}};
                if(path==2) {
                    std::optional<Metal::Scheduler::BufferComputeScope> scope;
                    if(narrow)scope.emplace(queue);
                    Metal::ComputeDispatch dispatch;dispatch.pipeline=compute;
                    dispatch.buffers={{input,i*16,0},{produced,i*16,1}};
                    queue.QueueCompute(std::span{&dispatch,1});
                }
                const auto* selected=narrow && path!=3 ? &proof:nullptr;
                const auto pipeline=path==3 ? storage_render:render;
                if(path==0) {
                    const auto args=queue.MakeArguments(bindings);
                    queue.KeepAlive(pipeline);
                    const auto encoder=queue.BeginRender(descriptor,@"direct timing",selected);
                    [encoder setRenderPipelineState:pipeline];
                    [encoder setArgumentTable:args atStages:MTLRenderStageVertex|MTLRenderStageFragment];
                    [encoder setViewport:MTLViewport{0,0,width,height,0,1}];
                    [encoder setScissorRect:MTLScissorRect{0,0,width,height}];
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3*triangles];
                    queue.EndEncoding();
                }else {
                    Metal::RenderCommand command{.pipeline=pipeline,.viewport={0,0,width,height,0,1},
                        .scissor={0,0,width,height},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,
                        .vertices=3*triangles};
                    queue.QueueRender(*pass,command,bindings,{}, {},(path==1 && narrow) || path==2 || path==4,{},selected);
                }
            }
            const auto encode_end=std::chrono::steady_clock::now();
            const auto tick=queue.Submit();
            const auto submit_end=std::chrono::steady_clock::now();
            queue.SetSubmissionObserver({});queue.Wait(tick);
            if(capture_directory)[[MTLCaptureManager sharedCaptureManager] stopCapture];
            Check(timing.wait_for(std::chrono::seconds{5})==std::future_status::ready,"Missing native commit timing feedback");
            const auto value=timing.get();
            Check(value.tick==tick && !value.failed && std::isfinite(value.gpu_begin_seconds) &&
                  std::isfinite(value.gpu_end_seconds) && value.gpu_begin_seconds>0 &&
                  value.gpu_end_seconds>value.gpu_begin_seconds,"Invalid native commit GPU timestamps");
            const auto after=queue.WorkerStats();const auto encoder_count=queue.EncoderCount()-encoders;
            const auto expected_encoders=path==2 ? draws*2:((path==1 && narrow) || path==4) ? 1u:draws;
            Check(encoder_count==expected_encoders,"Stage timing changed the expected workload encoder count");
            if(path==4 || path==3)
                Check(after.buffer_readonly_render_passes==before.buffer_readonly_render_passes,
                      "Alias or storage control incorrectly acquired a narrow render dependency");
            const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,width,height)},width*4);
            for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x) {
                unsigned expected{},actual{};
                for(unsigned i=0;i<draws;++i)expected+=triangles*(Value(unsigned(97+round)*37u+i*7919u)^((x+1)*131u+(y+1)*37u));
                std::memcpy(&actual,pixels.data()+(y*width+x)*4,4);
                Check(actual==expected,"Stage timing lost framebuffer-fetch RAW/WAW or complete raster pixels");
                if(path==3)Check(static_cast<const unsigned*>(storage.contents)[y*width+x]==Value(unsigned(97+round)*37u+(draws-1)*7919u),
                                 "Conservative storage timing lost complete fragment-store state");
            }
            if(paired.empty())paired=pixels;
            else Check(pixels==paired,"Paired stage policies changed a framebuffer byte");
            if(round>=0) {
                csv<<paths[path]<<','<<round<<','<<order_index<<','<<narrow<<','<<tick<<','<<std::setprecision(17)
                   <<value.gpu_begin_seconds<<','<<value.gpu_end_seconds<<','
                   <<(value.gpu_end_seconds-value.gpu_begin_seconds)*1000<<','
                   <<std::chrono::duration<double,std::milli>(encode_end-encode_begin).count()<<','
                   <<std::chrono::duration<double,std::milli>(submit_end-encode_end).count()<<','
                   <<encoder_count<<','<<draws<<','<<(path==2 ? draws:0)<<','
                   <<after.buffer_readonly_render_passes-before.buffer_readonly_render_passes<<','
                   <<after.conservative_render_passes-before.conservative_render_passes<<','
                   <<width*height<<','<<(path==3 ? width*height:0)<<'\n';
                ++samples;pixels_verified+=width*height;
            }
        }
    }
    csv.close();Check(bool(csv),"Unable to finish stage timing CSV");
    Check(samples==(capture_directory ? 6u:240u) && pixels_verified==(capture_directory ? 49152u:1966080u),
          "Incomplete paired stage timing matrix");
    if(capture_directory)std::cout<<"Native Metal4 capture:6 exact broad/narrow direct/queued/compute-render GPU traces and49152 full pixels verified; capture timings are excluded from the native performance comparison\n";
    else std::cout<<"Native stage policy timing:240 ABBA/BAAB submissions across5paths, no inserted GPU timestamps, exact1966080 measured pixels plus warmups and storage words; encoder counts and conservative controls passed\n";
}
