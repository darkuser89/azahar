// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_stream_buffer.h"
#include "mtl_transfer.h"
#include "mtl_shader_runtime.h"
#include "mtl_pipeline_manifest.h"
#include "mtl_compiler_gate_fixture.h"
#include "mtl_guest_codec.h"
#include "mtl_format_conversion.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <map>
#include <string_view>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <type_traits>
#include <sstream>

@interface LifetimeProbe : NSObject {
@public
    std::shared_ptr<std::atomic_uint> counter;
}
@end
@implementation LifetimeProbe
- (void)dealloc { if (counter) ++*counter; }
@end

void ValidateStageTiming(Metal::Device& device,const std::filesystem::path& output);

namespace Metal {
struct SchedulerTestAccess {
    static ReadOnlyBufferRenderProof ReadOnlyBuffers() { return {}; }
    static bool QueuedRenderOpen(const Scheduler& scheduler) { return scheduler.queued_render_open; }
    static bool QueuedComputeOpen(const Scheduler& scheduler) { return scheduler.queued_compute_open; }
    static void BlockQueue(Scheduler& scheduler, id<MTLSharedEvent> event) {
        scheduler.CheckThread();
        [scheduler.queue waitForEvent:event value:1];
    }
};
}

namespace {
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void ValidateReadyDrawableSubmission(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    Metal::TransferRuntime transfers{device, scheduler};
    auto gate = [device.Handle() newSharedEvent];
    struct ReleaseEvent {
        id<MTLSharedEvent> event;
        ~ReleaseEvent() { event.signaledValue = 1; }
    } release{gate};
    Metal::SchedulerTestAccess::BlockQueue(scheduler, gate);
    auto bytes = device.CreateBuffer(64, true);
    transfers.FillBuffer(bytes, NSMakeRange(0, 64), 0x21);
    scheduler.SubmitPending();
    Check(!scheduler.IsComplete(1), "Native event failed to hold the earlier submission");
    transfers.FillBuffer(bytes, NSMakeRange(0, 64), 0x73);
    const auto validated = std::make_shared<std::atomic_bool>(false);
    scheduler.CheckAfterCompletion([validated] { *validated = true; });
    const auto tick = scheduler.CurrentTick();
    Check(!scheduler.SubmitPendingBeforeDrawable(false) && scheduler.CurrentTick() == tick,
          "Ready drawable unnecessarily split pending guest work behind a busy GPU queue");
    Check(!scheduler.IsComplete(1) && !*validated,
          "Ready handoff waited for GPU completion or ran deferred validation early");
    std::atomic_uint rejected{};
    std::jthread foreign{[&] {
        try { scheduler.SubmitPendingBeforeDrawable(false); }
        catch (const std::logic_error&) { ++rejected; }
    }}; foreign.join();
    Check(rejected == 1, "Ready fast path bypassed scheduler ownership");
    // A genuinely blocking acquisition still needs the former progress path.
    Check(scheduler.SubmitPendingBeforeDrawable(true) && scheduler.CurrentTick() == tick + 1,
          "Unavailable drawable failed to submit guest work behind the held queue");
    Check(!*validated, "Submit-before-wait prematurely validated blocked GPU work");
    transfers.FillBuffer(bytes, NSMakeRange(0, 64), 0x74);
    Check(!scheduler.SubmitPendingBeforeDrawable(false) && scheduler.CurrentTick() == tick + 1,
          "Ready drawable split a second queued guest batch");
    gate.signaledValue = 1;
    scheduler.SubmitPending(); scheduler.Finish();
    Check(*validated && scheduler.CurrentTick() == tick + 2,
          "Ready handoff lost its one final submission or deferred validation");
    for (unsigned i = 0; i < 64; ++i)
        Check(static_cast<const unsigned char*>(bytes.contents)[i] == 0x74,
              "Ready handoff lost ordered native GPU bytes");
    scheduler.CheckAfterCompletion([] { throw std::runtime_error("expected ready deferred error"); });
    scheduler.SubmitPending();
    try { scheduler.Finish(); } catch (const std::runtime_error&) {}
    bool retained_error{};
    try { scheduler.SubmitPendingBeforeDrawable(false); }
    catch (const std::runtime_error& error) {
        retained_error = std::string_view{error.what()} == "expected ready deferred error";
    }
    Check(retained_error, "Ready fast path swallowed a retained GPU failure");
    std::cout << "Native ready drawable: externally gated busy queue, no split/wait/early validation, explicit progress for unavailable images, one final ready commit, exact ordered64 GPU bytes, owner and retained errors passed\n";
}

void ValidateImmutableRenderPass(Metal::Device& device) {
    Metal::Scheduler queue{device}; Metal::TransferRuntime transfer{device, queue};
    auto* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    auto first = device.CreateTexture(td), second = device.CreateTexture(td);
    auto* pass = [[MTL4RenderPassDescriptor alloc] init];
    pass.colorAttachments[0].texture = first;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
    const auto red = queue.SnapshotRenderPass(pass);
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 1, 1);
    const auto blue = queue.SnapshotRenderPass(pass);
    const auto check_pixels = [&](unsigned r, unsigned g, unsigned b) {
        const auto bytes = transfer.Readback({first, 0, 0, MTLRegionMake2D(0, 0, 4, 4)}, 16);
        for (unsigned i=0; i<16; ++i) Check(bytes[i*4]==std::byte(r) && bytes[i*4+1]==std::byte(g) && bytes[i*4+2]==std::byte(b) && bytes[i*4+3]==std::byte{255},
            "Immutable render-pass copy lost complete ordered pixel contents");
    };
    queue.BeginCompatibleRender(*red); queue.EndEncoding(); check_pixels(255, 0, 0);
    queue.BeginCompatibleRender(*blue); queue.EndEncoding(); check_pixels(0, 0, 255);
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    const auto load = queue.SnapshotRenderPass(pass), equal = queue.SnapshotRenderPass(pass);
    auto encoder = queue.BeginCompatibleRender(*load);
    Check(queue.BeginCompatibleRender(*equal)==encoder && queue.BeginCompatibleRender(pass)==encoder,
        "Identical independently copied render-pass keys did not reuse the live encoder");
    constexpr unsigned repeats=10000;
    const auto count=queue.EncoderCount(),tick=queue.CurrentTick();
    auto start=std::chrono::steady_clock::now();
    for(unsigned n=0;n<repeats;++n) queue.BeginCompatibleRender(pass);
    const auto generic_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    start=std::chrono::steady_clock::now();
    for(unsigned n=0;n<repeats;++n) queue.BeginCompatibleRender(*load);
    const auto snapshot_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    Check(queue.EncoderCount()==count && queue.CurrentTick()==tick,
        "Warm immutable/generic render-pass lookup encoded or submitted additional work");
    // The mutable API must inspect the descriptor again, including when its
    // address is unchanged and a private snapshot is currently active.
    pass.colorAttachments[0].loadAction=MTLLoadActionClear;
    queue.BeginCompatibleRender(pass); queue.EndEncoding(); check_pixels(0, 0, 255);
    const auto clears=queue.EncoderCount();
    queue.BeginCompatibleRender(*red); queue.BeginCompatibleRender(*red);
    Check(queue.EncoderCount()==clears+2,"Immutable clear snapshot incorrectly merged repeated clears");
    queue.EndEncoding();check_pixels(255,0,0);
    for(unsigned n=0;n<24;++n) {
        auto before=queue.BeginCompatibleRender(*load);
        Check(queue.BeginCompatibleRender(*equal)==before,"Warm immutable load pass did not remain open");
        if(n%3==0) queue.SubmitPending();
        else if(n%3==1) { auto buffer=device.CreateBuffer(16,true);transfer.FillBuffer(buffer,NSMakeRange(0,16),std::uint8_t(n)); }
        else queue.EndEncoding();
        const auto encoded=queue.EncoderCount();queue.BeginCompatibleRender(*load);
        Check(queue.EncoderCount()==encoded+1,"Render snapshot retained a stale encoder across submit/compute/end");
    }
    queue.EndEncoding();check_pixels(255,0,0);
    pass.colorAttachments[0].texture=second;pass.colorAttachments[0].loadAction=MTLLoadActionClear;
    pass.colorAttachments[0].clearColor=MTLClearColorMake(0,1,0,1);
    queue.BeginCompatibleRender(pass);queue.EndEncoding();
    pass=nil;second=nil;
    queue.BeginCompatibleRender(*load);queue.EndEncoding();check_pixels(255,0,0);
    bool nil_rejected{};try{queue.SnapshotRenderPass(nil);}catch(const std::invalid_argument&){nil_rejected=true;}
    Check(nil_rejected,"Missing immutable render pass was accepted");
    queue.Finish();
    std::cout<<"Immutable native render passes: full64-byte red/blue images, private copy/source mutation, equal keys, repeated clears, released source/attachments,24 submit/compute/end transitions and20000 warm lookups without encoding/submission passed; generic_ns="
        <<generic_ns<<", snapshot_ns="<<snapshot_ns<<" (host lookup diagnostic, not game FPS)\n";
}
void ValidateRegistrationGates(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    auto input = device.CreateBuffer(64, true), poison = device.CreateBuffer(64, true);
    auto output = device.CreateBuffer(128, true);
    for (unsigned i = 0; i < 16; ++i) {
        static_cast<u32*>(input.contents)[i] = 0x12340000 + i;
        static_cast<u32*>(poison.contents)[i] = 0xDEAD0000 + i;
    }
    std::memset(output.contents, 0, 128);
    const std::array first{Metal::BufferBinding{input, 0, 0}, Metal::BufferBinding{output, 0, 1}};
    const std::array second{Metal::BufferBinding{input, 0, 0}, Metal::BufferBinding{output, 64, 1}};
    const std::array replace_first{Metal::BufferBinding{poison, 0, 0}, Metal::BufferBinding{output, 0, 1}};
    const std::array replace_second{Metal::BufferBinding{poison, 0, 0}, Metal::BufferBinding{output, 64, 1}};
    auto table = scheduler.MakeArguments(first);
    Metal::ReusableArgumentTable warm, cold;
    auto updated = scheduler.UpdateArguments(warm, second);
    auto* pass = [[MTL4RenderPassDescriptor alloc] init];
    pass.renderTargetWidth = pass.renderTargetHeight = 1; pass.defaultRasterSampleCount = 1;
    const auto immutable_pass=scheduler.SnapshotRenderPass(pass);
    const std::array<std::function<void()>, 16> calls{{
        [&] { scheduler.UseResource(input); }, [&] { scheduler.KeepAlive(input); },
        [&] { scheduler.KeepAlive(nil); }, [&] { (void)scheduler.MakeArguments(first); },
        [&] { scheduler.RebindBufferArguments(table, replace_first); },
        [&] { (void)scheduler.UpdateArguments(warm, replace_second); },
        [&] { (void)scheduler.UpdateArguments(cold, first); },
        [&] { (void)scheduler.BeginCompute(); }, [&] { (void)scheduler.BeginRender(pass); },
        [&] { (void)scheduler.BeginCompatibleRender(pass); },
        [&] { (void)scheduler.BeginCompatibleRender(*immutable_pass); },
        [&] { (void)scheduler.SnapshotRenderPass(pass); },
        [&] { scheduler.CheckAfterCompletion([] {}); },
        [&] { Metal::Scheduler::BufferComputeScope buffers{scheduler}; },
        [&] { Metal::Scheduler::ComputeBatchScope batch{scheduler}; },
        [&] { scheduler.EndCompute(); }
    }};
    const auto tick = scheduler.CurrentTick();
    const auto retained = scheduler.CurrentRetainedObjectCount();
    for (const auto& call : calls) {
        bool rejected{};
        std::thread foreign{[&] { try { call(); } catch (const std::logic_error&) { rejected = true; } }};
        foreign.join();
        Check(rejected && scheduler.CurrentTick() == tick && scheduler.CurrentRetainedObjectCount() == retained,
              "Public scheduler batch bypassed its owner gate or changed a rejected submission");
    }
    // Rejected binding updates must leave the actual GPU-visible tables intact.
    auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void gate_read(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
                      uint index [[thread_position_in_grid]]) { output[index] = input[index]; }
)", "Native registration gate fixture");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "gate_read");
    auto pipeline = device.CompileCompute(descriptor); scheduler.KeepAlive(pipeline);
    for (auto arguments : {table, updated}) {
        auto encoder = scheduler.BeginCompute(pipeline);
        [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:arguments];
        [encoder dispatchThreads:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
        scheduler.EndEncoding();
    }
    scheduler.Finish();
    for (unsigned i = 0; i < 32; ++i)
        Check(static_cast<const u32*>(output.contents)[i] == 0x12340000 + i % 16,
              "Rejected public binding update changed a GPU-visible argument snapshot");
    unsigned invoked{};
    scheduler.CheckAfterCompletion([&] { ++invoked; throw std::runtime_error("registration gate sentinel"); });
    bool failed{};
    try { scheduler.Finish(); } catch (const std::runtime_error&) { failed = true; }
    Check(failed && invoked == 1, "Registration gate fixture lost its deferred GPU error");
    for (const auto& call : calls) {
        bool rejected{};
        try { call(); } catch (const std::runtime_error& error) {
            rejected = std::string_view{error.what()} == "registration gate sentinel";
        }
        Check(rejected && invoked == 1, "Public scheduler batch bypassed or re-executed a retained GPU error");
    }
    std::cout << "Native registration gates: 14 foreign-thread rejections, 14 retained-error rejections, "
                 "two unchanged GPU tables and 128 readback bytes passed\n";
}
void ValidateArgumentBindingLifetimes(Metal::Device& device) {
    for(bool buffer_view : {false,true}) {
        Metal::ReusableArgumentTable arguments;
        __weak id<MTLTexture> texture_probe=nil;
        __weak id<MTLSamplerState> sampler_probe=nil;
        @autoreleasepool {
            Metal::Scheduler scheduler{device};
            id<MTLTexture> texture;
            if(buffer_view) {
                auto buffer=device.CreateBuffer(4096,true);
                texture=device.CreateBufferTexture(buffer,0,256,MTLPixelFormatRG32Float);
            } else {
                auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                    width:2 height:1 mipmapped:NO];
                descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                auto parent=device.CreateTexture(descriptor);
                texture=[parent newTextureViewWithPixelFormat:MTLPixelFormatRGBA32Float];
            }
            auto sampler=device.CreateSampler([[MTLSamplerDescriptor alloc] init]);
            texture_probe=texture;sampler_probe=sampler;
            scheduler.UpdateArguments(arguments,{},std::array{Metal::TextureBinding{texture,127}},
                std::array{Metal::SamplerBinding{sampler,15}});
            scheduler.Finish();
        }
        // Retire completion callbacks as well as the scheduler's slots. Only
        // the logical binding owner should now keep its resource IDs alive.
        for(unsigned i=0;i<100 && texture_probe;++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(texture_probe && sampler_probe,
              "Argument cache outlived its texture/sampler objects and permits resource-ID reuse");
        @autoreleasepool {
            Metal::Scheduler scheduler{device};
            scheduler.UpdateArguments(arguments,{});
            scheduler.Finish();
        }
        for(unsigned i=0;i<1000 && texture_probe;++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!texture_probe,"Removed argument binding retained an obsolete texture object");
    }
    std::cout<<"Native argument owners keep texture/view and sampler IDs alive after GPU/scheduler retirement, "
                 "and release ordinary/buffer texture views when their slots are removed\n";
}
void ValidateVariableArguments(Metal::Device& device,bool null_probe=false) {
    Metal::Scheduler scheduler{device};Metal::TransferRuntime transfer{device,scheduler};
    std::map<unsigned,id<MTLComputePipelineState>> pipelines;
    const auto get_pipeline=[&](unsigned mask,bool have_base,bool have_extra) {
        const unsigned key=null_probe ? 31:mask|unsigned(have_base)*8|unsigned(have_extra)*16;
        if(auto found=pipelines.find(key);found!=pipelines.end()) return found->second;
        std::string source="#define HAVE_EXTRA "+std::to_string(bool(key&16))+"\n#define HAVE_BASE "+
            std::to_string(bool(key&8))+"\n#define HAVE_LOW "+std::to_string(bool(key&1))+
            "\n#define HAVE_HIGH "+std::to_string(bool(key&2))+"\n#define HAVE_LAST "+std::to_string(bool(key&4))+"\n";
        source+=R"(
#include <metal_stdlib>
using namespace metal;
kernel void variable_arguments(device float* output [[buffer(0)]],constant float2& tag [[buffer(1)]]
#if HAVE_EXTRA
    ,device const float* extra [[buffer(30)]]
#endif
#if HAVE_BASE
    ,texture2d<float,access::sample> base [[texture(0)]],sampler first_sampler [[sampler(0)]]
#endif
#if HAVE_LOW
    ,texture2d<float,access::read> low [[texture(63)]]
#endif
#if HAVE_HIGH
    ,texture2d<float,access::read> high [[texture(64)]]
#endif
#if HAVE_LAST
    ,texture2d<float,access::sample> last [[texture(127)]],sampler last_sampler [[sampler(15)]]
#endif
) {
    output[0]=tag.x;output[7]=tag.y;
#if HAVE_EXTRA
    output[1]=extra ? extra[0]:-1;
#else
    output[1]=-1;
#endif
#if HAVE_LOW
    output[2]=is_null_texture(low) ? -1:low.read(uint2(0)).r;
#else
    output[2]=-1;
#endif
#if HAVE_HIGH
    output[3]=is_null_texture(high) ? -1:high.read(uint2(0)).r;
#else
    output[3]=-1;
#endif
#if HAVE_LAST
    output[4]=is_null_texture(last) ? -1:last.read(uint2(0)).r;
    output[5]=is_null_texture(last) ? -1:last.sample(last_sampler,float2(0.6,0.5)).r;
#else
    output[4]=-1;output[5]=-1;
#endif
#if HAVE_BASE
    output[6]=is_null_texture(base) ? -1:base.sample(first_sampler,float2(0.6,0.5)).r;
#else
    output[6]=-1;
#endif
}
)";
        auto library=device.CompileLibrary(source,"Native variable argument layouts");
        auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"variable_arguments");
        descriptor.maxTotalThreadsPerThreadgroup=1;
        const auto result=device.CompileCompute(descriptor);pipelines.emplace(key,result);return result;
    };
    constexpr unsigned calls=128,stride=64;
    auto output=device.CreateBuffer(calls*stride+64,true);std::memset(output.contents,0xA7,output.length);
    std::array<std::array<float,8>,calls> expected{};
    unsigned submits=0,rejected=0;
    std::chrono::steady_clock::duration repeated_binding_time{};
    constexpr unsigned repeats=64;
    const auto tick=scheduler.CurrentTick();
    {
        Metal::ReusableArgumentTable arguments;
        id<MTL4ArgumentTable> first=nil;
        for(unsigned i=0;i<calls;++i) {
            @autoreleasepool {
                if(i%8==0) scheduler.UpdateArguments(arguments,{});
                const unsigned mask=(i/2)%8;
                const bool dense=i%4==0,have_base=i%13!=0,have_extra=i%3!=0;
                auto tag=device.CreateBuffer(64,true),extra=device.CreateBuffer(64,true);
                const NSUInteger offset=(i&1)?16:32;
                const float tags[2]{float(i+10),float(3*i+1)},extra_value=float(7*i+2);
                std::memcpy(static_cast<std::byte*>(tag.contents)+offset,tags,sizeof(tags));
                std::memcpy(static_cast<std::byte*>(extra.contents)+offset,&extra_value,sizeof(extra_value));
                auto* image_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                    width:2 height:1 mipmapped:NO];
                image_descriptor.storageMode=MTLStorageModeShared;
                image_descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
                auto parent=device.CreateTexture(image_descriptor);
                const float base=float(i*4),pixels[8]{base,0,0,1,base+2,0,0,1};
                [parent replaceRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0 withBytes:pixels bytesPerRow:sizeof(pixels)];
                auto view=[parent newTextureViewWithPixelFormat:MTLPixelFormatRGBA32Float];
                auto* sampler_descriptor=[[MTLSamplerDescriptor alloc] init];
                sampler_descriptor.minFilter=sampler_descriptor.magFilter=(i&1)?MTLSamplerMinMagFilterLinear:MTLSamplerMinMagFilterNearest;
                sampler_descriptor.sAddressMode=sampler_descriptor.tAddressMode=MTLSamplerAddressModeClampToEdge;
                auto sampler=device.CreateSampler(sampler_descriptor);
                std::vector<Metal::BufferBinding> buffers{{output,32+i*stride,0},{tag,offset,1}};
                if(have_extra) buffers.push_back({extra,offset,30});
                if(dense) for(unsigned slot=2;slot<30;++slot) buffers.push_back({extra,offset,slot});
                std::vector<Metal::TextureBinding> textures;
                if(have_base) textures.push_back({view,0});
                if(mask&1) textures.push_back({view,63});
                if(mask&2) textures.push_back({view,64});
                if(mask&4) textures.push_back({view,127});
                if(dense) for(unsigned slot=1;slot<127;++slot)
                    if(slot!=63 && slot!=64) textures.push_back({view,slot});
                std::vector<Metal::SamplerBinding> samplers;
                if(have_base) samplers.push_back({sampler,0});
                if(mask&4) samplers.push_back({sampler,15});
                if(dense) for(unsigned slot=1;slot<15;++slot) samplers.push_back({sampler,slot});
                auto table=scheduler.UpdateArguments(arguments,buffers,textures,samplers);
                if(first) Check(table==first,"Variable argument table was reallocated");
                else first=table;
                if(i%17==0) {scheduler.Submit();++submits;}
                // Rebind's raw table alias changes buffers only. Updating the
                // same logical set must restore them, even if textures/samplers
                // remain identical and their previous submission has retired.
                auto altered=buffers;altered[1]={extra,offset,1};
                scheduler.RebindBufferArguments(table,altered);
                const auto binding_begin=std::chrono::steady_clock::now();
                for(unsigned repeat=0;repeat<repeats;++repeat)
                    Check(scheduler.UpdateArguments(arguments,buffers,textures,samplers)==table,
                          "Stable native bindings replaced their argument table");
                repeated_binding_time+=std::chrono::steady_clock::now()-binding_begin;
                if(!textures.empty()) {
                    Check(scheduler.LastUseTick(view)==scheduler.CurrentTick() &&
                          scheduler.LastUseTick(parent)==scheduler.CurrentTick(),
                          "Variable argument update omitted view/backing registration");
                }
                // A valid replacement precedes each malformed late binding. The
                // following GPU dispatch must still see the previous complete set.
                if(i%8==3) {
                    auto replacement=device.CreateBuffer(64,true);
                    const float poison[2]{-900,-901};std::memcpy(replacement.contents,poison,sizeof(poison));
                    auto invalid_buffers=buffers;invalid_buffers[1]={replacement,0,1};
                    auto reject=[&](const auto& b,const auto& t,const auto& s) {
                        bool caught=false;try {scheduler.UpdateArguments(arguments,b,t,s);}
                        catch(const std::invalid_argument&) {caught=true;}
                        Check(caught,"Malformed variable argument set accepted");++rejected;
                    };
                    auto bad=invalid_buffers;bad.push_back({extra,0,31});reject(bad,textures,samplers);
                    bad=invalid_buffers;bad.push_back({extra,0,1});reject(bad,textures,samplers);
                    bad=invalid_buffers;bad.push_back({extra,extra.length,29});reject(bad,textures,samplers);
                    auto bad_textures=textures;bad_textures.push_back({view,128});reject(invalid_buffers,bad_textures,samplers);
                    bad_textures={{view,0},{view,0}};reject(invalid_buffers,bad_textures,samplers);
                    const std::vector<Metal::SamplerBinding> bad_samplers{{sampler,0},{sampler,0}};
                    reject(invalid_buffers,textures,bad_samplers);
                    const std::vector<Metal::SamplerBinding> range_sampler{{sampler,16}};
                    reject(invalid_buffers,textures,range_sampler);
                    bad=invalid_buffers;bad.push_back({nil,0,29});reject(bad,textures,samplers);
                    bad_textures={{nil,0}};reject(invalid_buffers,bad_textures,samplers);
                    const std::vector<Metal::SamplerBinding> null_sampler{{nil,0}};
                    reject(invalid_buffers,textures,null_sampler);
                }
                if(i==7) {
                    bool caught=false;
                    std::thread other{[&] {try {scheduler.UpdateArguments(arguments,buffers,textures,samplers);}
                        catch(const std::logic_error&) {caught=true;}}};other.join();
                    Check(caught,"Cross-thread variable argument update accepted");++rejected;
                }
                const auto pipeline=get_pipeline(mask,have_base,have_extra);
                scheduler.KeepAlive(pipeline);
                auto encoder=scheduler.BeginCompute(pipeline);
                [encoder setComputePipelineState:pipeline];[encoder setArgumentTable:table];
                [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                scheduler.EndEncoding();
                const auto filtered=base+((i&1)?1.4f:2.f);
                expected[i]={tags[0],have_extra?extra_value:-1.f,(mask&1)?base:-1.f,(mask&2)?base:-1.f,
                    (mask&4)?base:-1.f,(mask&4)?filtered:-1.f,have_base?filtered:-1.f,tags[1]};
            }
            if((i+1)%17==0) {scheduler.Submit();++submits;}
            Check(scheduler.CurrentTick()==tick+submits,"Variable argument update submitted early");
        }
        scheduler.UpdateArguments(arguments,{});
        first=nil; // The scheduler owns queued tables and all resource snapshots.
    }
    scheduler.Finish();
    const auto* bytes=static_cast<const std::byte*>(output.contents);
    for(unsigned i=0;i<calls;++i) {
        float actual[8];std::memcpy(actual,bytes+32+i*stride,sizeof(actual));
        for(unsigned channel=0;channel<8;++channel)
            Check(std::isfinite(actual[channel]) && std::abs(actual[channel]-expected[i][channel])<0.002f,
                  "Variable argument slots, rejected-update snapshot or GPU lifetime corrupted");
        for(unsigned b=0;b<32;++b)
            Check(bytes[32+i*stride+32+b]==std::byte{0xA7},"Variable argument output guard overwritten");
    }
    for(unsigned i=0;i<32;++i) Check(bytes[i]==std::byte{0xA7},"Variable argument prefix overwritten");
    std::cout<<calls<<" variable argument GPU snapshots: dense/sparse/empty layouts, buffer30, texture63/64/127, sampler0/15 changes, "
        <<submits<<" submits, null-probe="<<null_probe<<", "<<rejected<<" malformed/owner rejections, view parents and released CPU/table owners passed\n";
    std::cout<<calls*repeats<<" repeated native argument lookups including raw-buffer-alias restoration: "
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(repeated_binding_time).count()
        <<" host_ns (API-call diagnostic, not game FPS)\n";
}

void ValidateLutTextureStream(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void lut_snapshot(texture_buffer<float, access::read> lut [[texture(0)]],
                         constant uint* offsets [[buffer(0)]], device float4* output [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    const uint block = i < 6144 ? i / 256 : 24;
    const uint element = i < 6144 ? i % 256 : i - 6144;
    output[i] = lut.read(offsets[block] + element);
}
)", "Native immutable LUT offset fixture");
    auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "lut_snapshot");
    auto pipeline = device.CompileCompute(descriptor);
    constexpr unsigned calls = 192, texels = 24 * 256 + 128, stride = texels * 16 + 32;
    for (auto format : {MTLPixelFormatRG32Float, MTLPixelFormatRGBA32Float}) {
        const unsigned channels = format == MTLPixelFormatRG32Float ? 2 : 4;
        auto output = device.CreateBuffer(calls * stride, true);
        std::memset(output.contents, 0xA7, output.length);
        std::vector<std::array<float, 4>> expected;
        unsigned rollovers{}, partial{};
        {
            // Nonzero buffer-view base offsets and frequent page changes are
            // deliberate; shader texel offsets must be relative to the view.
            Metal::LutTextureStream stream{device, scheduler, format, channels * 64 * 1024};
            bool rejected{};
            try { stream.Append(16); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "LUT append without reservation accepted");
            std::array<std::uint32_t, 25> offsets{};
            std::vector<std::array<float, 4>> current(texels);
            for (unsigned call = 0; call < calls; ++call) {
                if (call && call % 61 == 0) stream.Clear();
                const auto before = stream.Texture();
                const bool replace = stream.BeginUpdate(texels * channels * sizeof(float));
                if (replace) ++rollovers; else ++partial;
                if (!replace) Check(stream.Texture() == before, "Partial LUT update replaced its texture view");
                for (unsigned block = 0; block < 25; ++block) {
                    if (!replace && block != call % 25) continue;
                    const unsigned count = block == 24 ? 128 : 256, first = block * 256;
                    auto storage = stream.Append(count * channels * sizeof(float));
                    offsets[block] = stream.TexelOffset(storage);
                    auto* values = static_cast<float*>(storage.data);
                    for (unsigned i = 0; i < count; ++i)
                        for (unsigned c = 0; c < 4; ++c) {
                            const float value = float(call * 10000 + block * 300 + i + c);
                            current[first + i][c] = c < channels ? value : c == 3 ? 1.f : 0.f;
                            if (c < channels) values[i * channels + c] = value;
                        }
                }
                if (!call) {
                    bool bad{};
                    try { stream.BeginUpdate(0); } catch (const std::invalid_argument&) { bad = true; }
                    Check(bad, "Empty LUT reservation accepted");
                    bad = false;
                    try { stream.Append(3); } catch (const std::invalid_argument&) { bad = true; }
                    Check(bad, "Misaligned LUT append accepted");
                    unsigned foreign{};
                    std::thread other{[&] {
                        try { stream.BeginUpdate(16); } catch (const std::logic_error&) { ++foreign; }
                        try { stream.Clear(); } catch (const std::logic_error&) { ++foreign; }
                    }};
                    other.join(); Check(foreign == 2, "Foreign LUT update changed the page");
                }
                expected.insert(expected.end(), current.begin(), current.end());
                auto indices = device.CreateBuffer(sizeof(offsets), true);
                std::memcpy(indices.contents, offsets.data(), sizeof(offsets));
                const std::array buffers{Metal::BufferBinding{indices, 0, 0},
                    Metal::BufferBinding{output, call * stride + 16, 1}};
                const std::array textures{Metal::TextureBinding{stream.Texture(), 0}};
                auto arguments = scheduler.MakeArguments(buffers, textures);
                scheduler.KeepAlive(pipeline);
                auto encoder = scheduler.BeginCompute(pipeline);
                [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:arguments];
                [encoder dispatchThreads:MTLSizeMake(texels, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                scheduler.EndEncoding();
                if (call % 13 == 0) scheduler.Submit();
                if (call % 37 == 0) scheduler.Finish();
            }
        } // Drop the current texture, page and CPU leases before final completion.
        scheduler.Finish();
        const auto* bytes = static_cast<const std::byte*>(output.contents);
        for (unsigned call = 0; call < calls; ++call) {
            Check(std::memcmp(bytes + call * stride + 16, expected.data() + call * texels, texels * 16) == 0,
                  "Native LUT texels, preserved offsets or old texture-page lifetime corrupted");
            for (unsigned b = 0; b < 16; ++b)
                Check(bytes[call * stride + b] == std::byte{0xA7} &&
                      bytes[call * stride + stride - 16 + b] == std::byte{0xA7}, "LUT GPU output guard corrupted");
        }
        Check(rollovers > 3 && partial > 100, "LUT stream missed page rollovers or partial reuse");
        std::cout << "Native LUT stream " << channels << " channels: " << calls << " complete6272-texel GPU snapshots, "
                  << rollovers << " page publications, " << partial << " partial updates, nonzero view offsets, clear, "
                  << "GPU/CPU lifetime, submit/finish and invalid/foreign updates passed\n";
    }
}

void ValidateImmutableUniforms(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    Metal::TransferRuntime transfer{device, scheduler};
    constexpr unsigned calls = 192, stride = 8192 + 32;
    auto output = device.CreateBuffer(calls * stride, true);
    std::memset(output.contents, 0xA7, output.length);
    std::vector<std::vector<std::byte>> expected;
    unsigned reused{}, changed{}, submits{};
    {
        Metal::ImmutableUniformCache cache{device, scheduler};
        std::array<std::vector<std::byte>, 4> values;
        std::array<Metal::BufferSlice, 4> previous{};
        // Different sizes, raw NaN/negative-zero bits, and changing sizes test
        // byte identity rather than floating-point equality or pointer identity.
        for (unsigned slot = 0; slot < 4; ++slot) {
            values[slot].resize((slot + 1) * 256);
            for (unsigned b = 0; b < values[slot].size(); ++b)
                values[slot][b] = std::byte((b * 37 + slot * 113) & 255);
            previous[slot] = cache.Upload(slot, values[slot]);
        }
        for (unsigned slot = 0; slot < 4; ++slot)
            for (unsigned repeat = 0; repeat < 2500; ++repeat) {
                auto same = cache.Upload(slot, values[slot]);
                Check(same.buffer == previous[slot].buffer && same.offset == previous[slot].offset,
                      "Unchanged uniform bytes allocated another range");
            }
        bool rejected{};
        try { cache.Upload(4, values[0]); } catch (const std::invalid_argument&) { rejected = true; }
        Check(rejected, "Invalid uniform cache slot accepted");
        rejected = false;
        try { cache.Upload(0, {}); } catch (const std::invalid_argument&) { rejected = true; }
        Check(rejected, "Empty uniform block accepted");
        unsigned foreign_rejected{};
        std::thread foreign{[&] {
            try { cache.Upload(0, values[0]); } catch (const std::logic_error&) { ++foreign_rejected; }
            try { cache.Clear(); } catch (const std::logic_error&) { ++foreign_rejected; }
        }};
        foreign.join(); Check(foreign_rejected == 2, "Foreign uniform cache access changed owner state");
        for (unsigned call = 0; call < calls; ++call) {
            const unsigned slot = (call * 7) & 3;
            if (call == 64 || call == 128) {
                cache.Clear(); previous = {};
            }
            if (call % 3 == 0) {
                values[slot][(call * 31) % values[slot].size()] ^= std::byte{0x80};
            }
            if (call % 19 == 0) values[slot].resize(256 + (call % 9) * 768, std::byte{0x5A});
            auto source = values[slot];
            auto current = cache.Upload(slot, source);
            if (previous[slot].buffer == current.buffer && previous[slot].offset == current.offset) ++reused;
            else ++changed;
            expected.push_back(source);
            std::memset(source.data(), 0xCC, source.size()); // cache cannot borrow caller bytes
            transfer.CopyBuffer(current.buffer, current.offset, output, call * stride + 16, current.size);
            previous[slot] = current;
            if (call % 11 == 0) { scheduler.Submit(); ++submits; }
            if (call % 37 == 0) scheduler.Finish(); // later reuse must renew use ticks
        }
        // Drop cache and CPU owners while the last GPU copies are still pending.
    }
    scheduler.Finish();
    const auto* actual = static_cast<const std::byte*>(output.contents);
    for (unsigned call = 0; call < calls; ++call) {
        const auto begin = call * stride;
        for (unsigned b = 0; b < stride; ++b) {
            const auto wanted = b >= 16 && b < 16 + expected[call].size()
                ? expected[call][b - 16] : std::byte{0xA7};
            Check(actual[begin + b] == wanted, "Uniform snapshot/lifetime or GPU output guard corrupted");
        }
    }
    Check(reused > 0 && changed > 0, "Uniform test missed either reuse or replacement");
    std::cout << "Immutable draw uniforms: 10000 unchanged lookups keep range identity; " << calls
              << " full guarded GPU snapshots, four slots, raw bits, resize, caller poisoning, "
              << reused << " reuses, " << changed << " replacements, " << submits
              << " submits, completion/reuse, clear and released CPU/cache ownership passed\n";
}

void ValidateSharedUploads(Metal::Device& device) {
    Metal::Scheduler scheduler{device}; Metal::TransferRuntime transfer{device,scheduler};
    Metal::SharedUploadPool pool{device,scheduler};
    bool rejected=false;try { pool.Allocate(0); } catch(const std::invalid_argument&) { rejected=true; }
    Check(rejected,"Zero upload accepted");
    rejected=false;
    std::thread other{[&] {try { pool.Allocate(64); } catch(const std::logic_error&) { rejected=true; }}};
    other.join(); Check(rejected,"Cross-thread pool mutation accepted");
    constexpr NSUInteger page=1024*1024;
    auto held=pool.Allocate(page);auto held_buffer=held.buffer;
    std::memset(held.data,17,held.size);scheduler.Finish();
    auto second=pool.Allocate(page);Check(held.buffer!=second.buffer,"CPU-held upload overwritten after completion");
    auto second_buffer=second.buffer;second={};
    auto third=pool.Allocate(page);Check(third.buffer!=second_buffer,"Unsubmitted GPU use was overwritten");
    auto readback=device.CreateBuffer(257,true);
    transfer.CopyBuffer(held.buffer,0,readback,0,257);
    held={};scheduler.Finish();
    auto reused=pool.Allocate(page);Check(reused.buffer==held_buffer,"Completed upload allocation was not reused");
    std::memset(reused.data,91,reused.size);
    for(unsigned i=0;i<257;++i)Check(static_cast<const unsigned char*>(readback.contents)[i]==17,"Earlier GPU consumer lost bytes");
    // A later submission using an old allocation must postpone recycling again.
    transfer.CopyBuffer(reused.buffer,0,readback,0,257);reused={};
    auto later=pool.Allocate(page);Check(later.buffer!=held_buffer,"Later GPU use tick ignored");scheduler.Finish();
    for(unsigned i=0;i<257;++i)Check(static_cast<const unsigned char*>(readback.contents)[i]==91,"Later GPU consumer lost bytes");
    third={};later={};
    {
        Metal::SharedUploadPool concurrent{device,scheduler};
        auto output=device.CreateBuffer(12*128,true);
        for(unsigned i=0;i<12;++i) {
            auto first=concurrent.Allocate(64);
            std::memset(first.data,i+1,64);
            transfer.CopyBuffer(first.buffer,first.offset,output,i*128,64);
            scheduler.Submit();
            auto next=concurrent.Allocate(64);
            Check(first.buffer==next.buffer&&first.offset+first.size<=next.offset,
                  "Disjoint in-flight appends did not share their arena page safely");
            std::memset(next.data,i+101,64);
            transfer.CopyBuffer(next.buffer,next.offset,output,i*128+64,64);
        }
        scheduler.Finish();
        auto* bytes=static_cast<const unsigned char*>(output.contents);
        for(unsigned i=0;i<12;++i)for(unsigned b=0;b<64;++b)
            Check(bytes[i*128+b]==i+1&&bytes[i*128+b+64]==i+101,"In-flight append changed an earlier snapshot");
    }
    for(unsigned round=0;round<12;++round) {
        std::vector<Metal::BufferSlice> slices;
        NSUInteger total=0;
        for(unsigned i=0;i<24;++i) {
            auto slice=pool.Allocate((i+1)*257+round*13);
            for(const auto& old:slices)Check(old.buffer!=slice.buffer || old.offset+old.size<=slice.offset ||
                slice.offset+slice.size<=old.offset,"Simultaneous immutable upload ranges overlap");
            std::memset(slice.data,(round*29+i*7)&255,slice.size);total+=slice.size;
            slices.push_back(std::move(slice));
        }
        auto output=device.CreateBuffer(total+32,true);std::memset(output.contents,0xA7,total+32);
        NSUInteger offset=16;
        for(auto& slice:slices) {transfer.CopyBuffer(slice.buffer,slice.offset,output,offset,slice.size);offset+=slice.size;}
        slices.clear(); // GPU lifetime cannot depend on CPU slice handles.
        scheduler.Submit();scheduler.Finish();
        auto* bytes=static_cast<const unsigned char*>(output.contents);offset=16;
        for(unsigned i=0;i<24;++i) {
            const auto size=(i+1)*257+round*13;
            for(unsigned b=0;b<size;++b)Check(bytes[offset+b]==((round*29+i*7)&255),"GPU upload bytes corrupted");
            offset+=size;
        }
        for(unsigned i=0;i<16;++i)Check(bytes[i]==0xA7&&bytes[total+16+i]==0xA7,"Upload consumer overwrote canary");
    }
    // More retained bytes than the pool budget and a single oversized upload
    // remain valid native allocations with exactly their requested visible size.
    std::vector<Metal::BufferSlice> large;
    for(unsigned i=0;i<12;++i)large.push_back(pool.Allocate(4*1024*1024));
    auto oversized=pool.Allocate(32*1024*1024+1);
    Check(oversized.size==32*1024*1024+1&&oversized.buffer.length>=oversized.size,"Oversized valid upload truncated");
    for(auto& slice:large)Check(slice.size==4*1024*1024&&slice.data,"Budget pressure rejected valid upload");
    scheduler.Finish();
    std::cout<<"Shared upload lifecycle: zero/owner rejection, CPU leases, unsubmitted/later GPU ticks, reuse, 288 full snapshots, canaries, 12 large leases and oversized allocation passed\n";
}

void ValidateBoundedProfiles(Metal::Device& device, bool host_only = false) {
    struct Environment {
        std::array<const char*,4> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_TRIGGER","AZAHAR_METAL4_PROFILE_FRAME_LIMIT","AZAHAR_METAL4_PROFILE_HOST_ONLY"};
        std::array<std::optional<std::string>,4> old;
        Environment() {for(unsigned i=0;i<keys.size();++i) {if(const auto* v=std::getenv(keys[i]))old[i]=v;unsetenv(keys[i]);}}
        ~Environment() {for(unsigned i=0;i<keys.size();++i) {if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}}
    } environment;
    char directory[]="/private/tmp/azahar-bounded-profile-test-XXXXXX";
    Check(mkdtemp(directory),"Unable to create bounded profile test directory");
    const auto path=std::filesystem::path(directory)/"gpu.csv",trigger=std::filesystem::path(directory)/"start";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_TRIGGER",trigger.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","2",1);
    if (host_only) setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        const auto output=device.CreateBuffer(64,true);const auto tick=queue.CurrentTick();
        Check(!queue.ProfilingEnabled(),"Triggered profile started before its signal");
        for(unsigned frame=0;frame<10;++frame) {
            if(frame==3) {std::ofstream signal(trigger);signal<<"start";}
            const bool active=frame==4 || frame==5;
            Check(queue.ProfilingEnabled()==active,"Bounded profile did not match exact frame boundaries");
            transfer.FillBuffer(output,NSMakeRange(0,64),std::uint8_t(frame+31));
            queue.MarkDiagnosticDraw();
            const auto acquire=queue.BeginProfileDrawableAcquire();
            Check(bool(acquire.begin_host_ns)==active,"Inactive bounded profile read drawable timing");
            queue.EndProfileDrawableAcquire(acquire,active ? 0:999,false);
            queue.EndProfileFrame();
            Check(queue.CurrentTick()==tick,"Profile trigger/limit submitted or waited for GPU work");
        }
        Check(!queue.ProfilingEnabled(),"Exhausted trigger profile rearmed from the same signal");
        queue.Finish();
        for(unsigned i=0;i<64;++i) Check(static_cast<const unsigned char*>(output.contents)[i]==40,
            "Bounded timestamp activation/retirement changed native GPU output");
    }
    const auto fields=[](const std::string& line) {
        std::istringstream input{line};std::vector<std::string> result;std::string field;
        while(std::getline(input,field,','))result.push_back(field);return result;
    };
    std::string line;unsigned rows{};
    std::ifstream frames(path.string()+".frames.csv");std::getline(frames,line);
    Check(fields(line).size()==11,"Bounded frame profile lost its wall-clock schema");
    while(std::getline(frames,line)) {
        const auto f=fields(line);Check(f.size()==11 && rows<2 && std::stoul(f[0])==4+rows &&
            std::stoul(f[8])==1 && std::stoul(f[9])==1 && std::stoll(f[10])>std::stoll(f[2]),
            "Bounded profile included waiting/post-limit frames or lost exact attribution");++rows;
    }
    Check(rows==2,"Bounded profile lost complete active frames");
    std::ifstream control(path.string()+".control.csv");std::getline(control,line);rows=0;
    for(const auto expected:std::array{std::pair{0u,0u},std::pair{4u,1u},std::pair{6u,0u}}) {
        Check(bool(std::getline(control,line)),"Missing bounded control transition");const auto f=fields(line);
        Check(f.size()==4 && std::stoul(f[0])==expected.first && std::stoul(f[1])==expected.second &&
            std::stoll(f[2])>0 && std::stoll(f[3])>std::stoll(f[2]),"Bounded control transition has wrong frame/state/clocks");++rows;
    }
    Check(!std::getline(control,line),"Bounded profile recorded unexpected transitions");
    std::ifstream gpu(path);rows=0;
    while(std::getline(gpu,line)) {
        if(line.empty() || line[0]=='#' || line.starts_with("tick,"))continue;
        const auto f=fields(line);Check(f.size()==11 && rows<2 && std::stoul(f[10])==4+rows &&
            f[6]=="1" && std::stoull(f[4])>=std::stoull(f[3]),"Retired bounded GPU timestamp belongs to a wrong/invalid frame");++rows;
    }
    Check(rows==(host_only ? 0u : 2u),"Host-only profile emitted GPU timestamps or bounded timestamps did not retire");
    if (host_only) {
        std::ifstream mode(path);std::getline(mode,line);
        Check(line=="# host_only=1","Host-only output did not identify its measurement mode");
    }
    for(const char* limit:{"-1","3x","1000001","18446744073709551616","0"}) {
        const auto invalid=std::filesystem::path(directory)/("bad-"+std::string(limit));
        setenv("AZAHAR_METAL4_PROFILE",invalid.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT",limit,1);
        bool rejected{};try{Metal::Scheduler queue{device};}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && !std::filesystem::exists(invalid),"Malformed/unbounded trigger request opened output or was accepted");
    }
    for (const char* mode : {"true", "2", "1x"}) {
        const auto invalid=std::filesystem::path(directory)/("bad-mode-"+std::string(mode));
        setenv("AZAHAR_METAL4_PROFILE",invalid.c_str(),1);
        setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","2",1);
        setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY",mode,1);
        bool rejected{};try{Metal::Scheduler queue{device};}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && !std::filesystem::exists(invalid),"Invalid host-only mode opened output or was accepted");
    }
    unsetenv("AZAHAR_METAL4_PROFILE");
    {Metal::Scheduler off{device};Check(!off.ProfilingEnabled(),"Profile control enabled diagnostics without an output request");}
    std::filesystem::remove_all(directory);
    std::cout<<(host_only ? "Bounded host-only profiles: " : "Bounded native GPU profiles: ")
        <<"ten GPU frames, exact4/5 capture, three clocked transitions, no trigger submit/wait, automatic disable before retirement, no rearm,64 output bytes and eight rejected requests passed\n";
}

void ValidateCompilerProfiles() {
    struct Environment {
        std::array<const char*,7> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_TRIGGER",
            "AZAHAR_METAL4_PROFILE_FRAME_LIMIT","AZAHAR_METAL4_PROFILE_HOST_ONLY",
            "AZAHAR_METAL4_PROFILE_COMPILER","AZAHAR_METAL4_PROFILE_COMPILER_LIMIT",
            "AZAHAR_METAL4_PROFILE_COMPILER_OUTPUT"};
        std::array<std::optional<std::string>,7> old;
        Environment() {for(unsigned i=0;i<keys.size();++i) {
            if(const auto* value=std::getenv(keys[i]))old[i]=value;unsetenv(keys[i]);
        }}
        ~Environment() {for(unsigned i=0;i<keys.size();++i) {
            if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);
        }}
    } environment;
    char directory[]="/private/tmp/azahar-compiler-profile-test-XXXXXX";
    Check(mkdtemp(directory),"Unable to create isolated compiler profile directory");
    const auto path=std::filesystem::path(directory)/"host.csv";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    setenv("AZAHAR_METAL4_PROFILE_COMPILER","1",1);
    setenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT","3",1);
    std::size_t expected_source_bytes{};
    {
        Metal::Device measured{{},false};
        constexpr std::string_view source=R"(#include <metal_stdlib>
using namespace metal;
kernel void diagnostic_compiler_write(device uint* output [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    output[index]=index^0x51a7e3d9u;
})";
        expected_source_bytes=source.size();
        auto library=measured.CompileLibrary(source,"diagnostic \"quoted\", line\nlibrary");
        auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
        descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"diagnostic_compiler_write");
        descriptor.maxTotalThreadsPerThreadgroup=64;
        auto pipeline=measured.CompileCompute(descriptor);
        Check(measured.CompileLibrary(source,"beyond bounded operation budget")!=nil,
              "Exhausted compiler diagnostics prevented native compilation");
        Metal::Scheduler queue{measured};
        const auto output=measured.CreateBuffer(128,true);
        std::memset(output.contents,0xcd,128);
        const std::array bindings{Metal::BufferBinding{output,0,0}};
        const auto args=queue.MakeArguments(bindings);
        const auto encoder=queue.BeginCompute(pipeline);
        [encoder setComputePipelineState:pipeline];[encoder setArgumentTable:args];
        [encoder dispatchThreads:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        queue.EndEncoding();queue.Finish();
        for(unsigned i=0;i<32;++i)Check(static_cast<const u32*>(output.contents)[i]==
            (i<16 ? i^0x51a7e3d9u:0xcdcdcdcdu),"Compiler profiling changed native output or guards");
    }
    const auto fields=[](const std::string& line) {
        std::vector<std::string> result;std::string value;bool quoted{};
        for(std::size_t i=0;i<line.size();++i) {
            const auto c=line[i];
            if(c=='\"') {
                if(quoted && i+1<line.size() && line[i+1]=='\"'){value+='\"';++i;}
                else quoted=!quoted;
            } else if(c==',' && !quoted){result.push_back(value);value.clear();}
            else value+=c;
        }
        Check(!quoted,"Compiler CSV contains an unterminated quoted name");result.push_back(value);return result;
    };
    std::ifstream input(path.string()+".compiler.csv");std::string line;std::getline(input,line);
    Check(fields(line).size()==17,"Compiler operation schema lost clock/attribution fields");
    unsigned rows{};const std::array names{"compiler","library","compute"};
    while(std::getline(input,line)) {
        const auto f=fields(line);Check(f.size()==17 && rows<3 && std::stoul(f[0])==rows+1 && f[1]==names[rows],
            "Bounded compiler profile lost exact operation order or exceeded its limit");
        const auto begin=std::stoll(f[3]),api_begin=std::stoll(f[4]),api_end=std::stoll(f[5]),end=std::stoll(f[6]);
        Check(begin>0 && end>=begin && std::stod(f[7])>=0 && f[8]=="0" && f[9]=="0",
              "Compiler profile contains invalid host interval or failure/cache status");
        if(rows)Check(api_begin>=begin && api_end>=api_begin && api_end<=end,
            "Compiler API subinterval escaped its inclusive operation interval");
        const auto lock_wait=std::stoull(f[11]),lock_held=std::stoull(f[12]);
        Check(lock_wait+lock_held<=std::uint64_t(end-begin) && f[13]=="0" && f[14]=="0" && f[15]=="0" && f[16]=="0",
              "Compiler lock/persistence subintervals escaped operation or invented disabled I/O");
        if(rows)Check(lock_held>0,"Compiler profile missed cache metadata critical sections");
        if(rows==1)Check(f[2]=="diagnostic \"quoted\", line library" && std::stoull(f[10])==expected_source_bytes,
                        "Compiler CSV damaged a quoted/comma/newline name or source byte count");
        else Check(f[10]=="0","Non-library compiler operation invented source bytes");
        ++rows;
    }
    Check(rows==3,"Compiler profile did not record its full bounded operation budget");
    for(const auto* limit:{"0","-1","3x","4097","18446744073709551616"}) {
        const auto bad=std::filesystem::path(directory)/("bad-limit-"+std::string(limit));
        setenv("AZAHAR_METAL4_PROFILE",bad.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT",limit,1);
        bool rejected{};try{Metal::Device invalid{{},false};}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && !std::filesystem::exists(bad.string()+".compiler.csv"),"Invalid compiler budget opened output or was accepted");
    }
    setenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT","3",1);
    for(const auto* mode:{"true","2","1x"}) {
        const auto bad=std::filesystem::path(directory)/("bad-mode-"+std::string(mode));
        setenv("AZAHAR_METAL4_PROFILE",bad.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_COMPILER",mode,1);
        bool rejected{};try{Metal::Device invalid{{},false};}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && !std::filesystem::exists(bad.string()+".compiler.csv"),"Invalid compiler mode opened output or was accepted");
    }
    setenv("AZAHAR_METAL4_PROFILE_COMPILER","1",1);unsetenv("AZAHAR_METAL4_PROFILE");
    bool missing{};try{Metal::Device invalid{{},false};}catch(const std::invalid_argument&){missing=true;}
    Check(missing,"Compiler profile accepted a missing output path");
    const auto independent=std::filesystem::path(directory)/"independent";
    setenv("AZAHAR_METAL4_PROFILE_COMPILER_OUTPUT",independent.c_str(),1);
    {
        Metal::Device measured{{},false};Metal::Scheduler queue{measured};
        Check(!queue.ProfilingEnabled(),"Compiler-only output activated Scheduler profiling");
    }
    Check(std::filesystem::exists(independent.string()+".compiler.csv") &&
          !std::filesystem::exists(independent.string()+".frames.csv"),
          "Compiler-only diagnostics lost output or activated frame diagnostics");
    unsetenv("AZAHAR_METAL4_PROFILE_COMPILER_OUTPUT");
    const auto disabled=std::filesystem::path(directory)/"disabled";
    setenv("AZAHAR_METAL4_PROFILE",disabled.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_COMPILER","0",1);
    setenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT","0",1);
    {Metal::Device off{{},false};}
    Check(!std::filesystem::exists(disabled.string()+".compiler.csv"),"Disabled compiler profiling opened a diagnostic file");
    std::filesystem::remove_all(directory);
    std::cout<<"Bounded compiler operations: three real Metal4 calls, full128 output/guard bytes, exact CPU/API intervals, quoted CSV name, exhausted-budget compilation, nine rejected requests and disabled output passed\n";
}

// A bounded native measurement, with real compiler contention and concurrent
// archive publication. The CSV separates mutex acquisition, held time, source
// size, native API time and persistence work; it never attributes all pre-API
// time to a lock. Every resulting kernel is dispatched and fully checked.
void ValidateCompilerCosts(const std::filesystem::path& output) {
    Check(output.is_absolute() && !std::filesystem::exists(output.string()+".compiler.csv"),
          "Compiler cost output must be a new absolute path");
    struct Environment {
        const std::array<const char*,3> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_COMPILER","AZAHAR_METAL4_PROFILE_COMPILER_LIMIT"};
        std::array<std::optional<std::string>,3> old;
        Environment() {for(unsigned i=0;i<keys.size();++i)if(const auto* value=std::getenv(keys[i]))old[i]=value;}
        ~Environment() {for(unsigned i=0;i<keys.size();++i)if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}
    } environment;
    setenv("AZAHAR_METAL4_PROFILE",output.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_COMPILER","1",1);
    setenv("AZAHAR_METAL4_PROFILE_COMPILER_LIMIT","4096",1);
    char directory[]="/private/tmp/azahar-compiler-cost-XXXXXX";
    Check(mkdtemp(directory),"Unable to create isolated compiler cost cache");
    unsigned words{};
    {
        Metal::Device measured{directory,true};
        measured.SwitchDiskCache(0x1234c057);
        Metal::Scheduler queue{measured};
        struct Program {id<MTLLibrary> library;id<MTLComputePipelineState> pipeline;unsigned steps,seed;};
        std::promise<void> release;
        auto start=release.get_future().share();
        std::array<std::shared_future<std::vector<Program>>,2> jobs;
        for(unsigned worker=0;worker<2;++worker) jobs[worker]=measured.DeferCompilation([&,worker,start] {
            start.wait();std::vector<Program> programs;
            for(unsigned round=0;round<3;++round)for(unsigned steps:{1U,16U,64U,256U}) {
                const unsigned seed=worker*100+round*7+steps;
                const std::string name="compiler-cost-"+std::to_string(worker)+"-"+std::to_string(round)+"-"+std::to_string(steps);
                std::string source="#include <metal_stdlib>\nusing namespace metal;\nkernel void cost_write(device uint* o [[buffer(0)]],uint i [[thread_position_in_grid]]) {uint v=i^"+std::to_string(seed)+"u;\n";
                for(unsigned n=0;n<steps;++n)source+="v=v*1664525u+1013904223u;\n";
                source+="o[i]=v;}\n";
                auto library=measured.CompileLibrary(source,name);
                Check(measured.CompileLibrary(source,name)==library,"Warm library cost probe lost identity");
                const auto descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
                descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"cost_write");
                descriptor.maxTotalThreadsPerThreadgroup=64;
                auto pipeline=measured.CompileCompute(descriptor);
                programs.push_back({library,pipeline,steps,seed});
                measured.RequestDiskCacheFlush();
            }
            return programs;
        });
        release.set_value();
        for(auto& job:jobs)for(const auto& program:job.get()) {
            const auto buffer=measured.CreateBuffer(256,true);std::memset(buffer.contents,0xcd,256);
            const std::array bindings{Metal::BufferBinding{buffer,0,0}};
            const auto args=queue.MakeArguments(bindings);
            auto encoder=queue.BeginCompute(program.pipeline);
            [encoder setComputePipelineState:program.pipeline];[encoder setArgumentTable:args];
            [encoder dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            queue.EndEncoding();queue.Finish();
            for(unsigned i=0;i<64;++i) {
                unsigned expected=i^program.seed;
                for(unsigned n=0;n<program.steps;++n)expected=expected*1664525U+1013904223U;
                Check(static_cast<const unsigned*>(buffer.contents)[i]==(i<32 ? expected:0xcdcdcdcdU),
                      "Compiler/persistence timing probe changed native output or guards");
                ++words;
            }
        }
        Check(measured.FlushDiskCache(),"Compiler cost final publication failed");
        Check(measured.DiskCacheStats().saved_batches>0,"Compiler cost probe did not publish a native archive");
    }
    std::ifstream csv(output.string()+".compiler.csv");std::string line;std::getline(csv,line);
    unsigned libraries{},warm{},persisted{},publication{};
    std::uint64_t max_wait{},io_ns{};
    std::vector<std::pair<std::uint64_t,std::uint64_t>> native_intervals;
    while(std::getline(csv,line)) {
        std::vector<std::string> fields;std::stringstream row{line};std::string field;
        while(std::getline(row,field,','))fields.push_back(field);
        Check(fields.size()==17,"Compiler cost schema lost intervals");
        Check(fields[9]=="0","Compiler cost probe recorded a failed operation");
        const auto begin=std::stoull(fields[3]),end=std::stoull(fields[6]);
        const auto wait=std::stoull(fields[11]),held=std::stoull(fields[12]),io=std::stoull(fields[14]);
        Check(wait+held<=end-begin && io<=end-begin,"Compiler cost subinterval escaped its operation");
        max_wait=std::max(max_wait,std::uint64_t(wait));io_ns+=io;
        if(fields[1]=="library" || fields[1]=="compute") {
            const auto api_begin=std::stoull(fields[4]),api_end=std::stoull(fields[5]);
            if(api_begin && api_end>api_begin) native_intervals.emplace_back(api_begin,api_end);
        }
        if(fields[1]=="library") {
            ++libraries;
            Check(std::stoull(fields[10])>0 && held>0,"Library source size/critical section not measured");
            if(fields[8]=="1") {++warm;persisted+=std::stoul(fields[16]);}
        }
        if(fields[1]=="flush_disk" && std::stoul(fields[15]))++publication;
    }
    Check(libraries==48 && warm==24 && persisted==24 && publication>0 && io_ns>0 && words==1536,
          "Compiler cost probe missed warm fingerprints, real publication or full GPU checks");
    unsigned overlapping{};
    for(std::size_t i=0;i<native_intervals.size();++i)
        for(std::size_t j=0;j<i;++j)
            overlapping+=std::max(native_intervals[i].first,native_intervals[j].first)<
                         std::min(native_intervals[i].second,native_intervals[j].second);
    Check(overlapping>0,"Independent native Metal compilations remained serialized");
    std::filesystem::remove_all(directory);
    std::cout<<"Compiler cost probe:24 cold+24 warm libraries,24 native kernels,1536 full output/guard words, concurrent native publication, overlapping_native_api_pairs="<<overlapping<<", compiler-lock wait max_ns="<<max_wait<<", persistence_ns="<<io_ns<<" passed\n";
}

void ValidateHostPhaseProfiles(Metal::Device& device) {
    struct Environment {
        std::array<const char*,5> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_TRIGGER",
            "AZAHAR_METAL4_PROFILE_FRAME_LIMIT","AZAHAR_METAL4_PROFILE_HOST_ONLY","AZAHAR_METAL4_PROFILE_HOST_PHASES"};
        std::array<std::optional<std::string>,5> old;
        Environment(){for(unsigned i=0;i<keys.size();++i){if(const auto* v=std::getenv(keys[i]))old[i]=v;unsetenv(keys[i]);}}
        ~Environment(){for(unsigned i=0;i<keys.size();++i){if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}}
    } environment;
    using Phase=Metal::Scheduler::HostPhase;using Scope=Metal::Scheduler::HostPhaseScope;
    char directory[]="/private/tmp/azahar-host-phase-test-XXXXXX";
    Check(mkdtemp(directory),"Unable to create host phase test directory");
    const auto path=std::filesystem::path(directory)/"host.csv",trigger=std::filesystem::path(directory)/"start";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    setenv("AZAHAR_METAL4_PROFILE_TRIGGER",trigger.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","2",1);
    setenv("AZAHAR_METAL4_PROFILE_HOST_PHASES","1",1);
    const auto worker_library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void host_profile_worker(device atomic_uint& value [[buffer(0)]]) {
    atomic_fetch_xor_explicit(&value,0u,memory_order_relaxed);
}
)","Native host-profile worker fixture");
    auto* worker_descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
    worker_descriptor.computeFunctionDescriptor=Metal::Device::Function(worker_library,"host_profile_worker");
    const auto worker_pipeline=device.CompileCompute(worker_descriptor);
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        const auto output=device.CreateBuffer(64,true);const auto tick=queue.CurrentTick();
        for(unsigned frame=0;frame<6;++frame) {
            if(frame==1){std::ofstream signal(trigger);signal<<1;}
            const bool active=frame==2 || frame==3;
            Check(queue.ProfilingEnabled()==active,"Host phase trigger/frame limit differs");
            {
                Scope root{queue,Phase::Present};Scope nested{queue,Phase::Coherency};
                Scope deeper{queue,Phase::Draw};transfer.FillBuffer(output,NSMakeRange(0,64),std::uint8_t(frame+17));
            }
            {Scope root{queue,Phase::Draw};transfer.FillBuffer(output,NSMakeRange(0,64),std::uint8_t(frame+31));}
            try{Scope root{queue,Phase::Transfer};throw std::runtime_error("controlled scope unwind");}
            catch(const std::runtime_error&){}
            {
                Scope root{queue,Phase::State};
                if(active) {
                    bool rejected=false;
                    try{queue.EndProfileFrame();}catch(const std::logic_error&){rejected=true;}
                    Check(rejected && queue.CurrentFrame()==frame,"Host scope crossed a frame boundary or advanced it on failure");
                }
            }
            bool foreign_rejected=false;
            std::thread foreign{[&]{try{Scope root{queue,Phase::State};}catch(const std::logic_error&){foreign_rejected=true;}}};
            foreign.join();Check(foreign_rejected==active,"Host phase scope changed owner checks outside its active window");
            bool invalid=false;
            try{Scope root{queue,Phase(999)};}catch(const std::invalid_argument&){invalid=true;}
            Check(invalid==active,"Host phase invalid input changed active/inactive semantics");
            Metal::ComputeDispatch dispatch;dispatch.pipeline=worker_pipeline;dispatch.buffers={{output,0,0}};
            queue.QueueCompute(std::span{&dispatch,1});
            queue.EndProfileFrame();Check(queue.CurrentTick()==tick,"Host phase timing submitted or waited for GPU work");
        }
        queue.Finish();for(unsigned i=0;i<64;++i)Check(static_cast<const unsigned char*>(output.contents)[i]==36,
            "Host phase timing changed ordered GPU output");
    }
    std::ifstream file(path.string()+".host-phases.csv");std::string line;
    Check(bool(std::getline(file,line)) && line=="frame,phase,calls,host_ms,thread_cpu_ms,cpu_valid_calls","Host phase output schema changed");
    std::map<std::pair<unsigned,std::string>,double> durations;unsigned rows=0;
    while(std::getline(file,line)) {
        std::istringstream row{line};std::array<std::string,6> fields;
        for(auto& field:fields)Check(bool(std::getline(row,field,',')),"Truncated host phase record");
        const auto frame=std::stoul(fields[0]);const auto ms=std::stod(fields[3]);
        Check((frame==2 || frame==3) && fields[2]=="1" && std::isfinite(ms) && ms>=0 &&
              fields[1]!="coherency" && durations.emplace(std::make_pair(frame,fields[1]),ms).second,
              "Host phase attribution double-counted nested calls, retained inactive frames or lost valid time");++rows;
        const auto cpu=std::stod(fields[4]);
        Check(fields[5]=="1" && std::isfinite(cpu) && cpu>=0 && cpu<=ms+.05,
              "Root CPU time is unavailable, invalid or not bounded by its wall interval");
    }
    Check(rows==8 && durations.size()==8,"Host phase attribution lost exclusive roots or exception unwinding");
    for(unsigned frame:{2u,3u})for(const char* phase:{"present","draw","transfer","state"})
        Check(durations.contains({frame,phase}),"Host phase record assigned a root to the wrong category/frame");
    const auto cpu_rows=[&](const std::filesystem::path& base) {
        std::ifstream input(base.string()+".thread-cpu.csv");std::string text;
        Check(bool(std::getline(input,text)) && text=="frame,begin_host_ns,end_host_ns,thread_cpu_ms,cpu_valid,qos_class,relative_priority",
              "Thread CPU profile schema missing or changed");
        std::vector<std::array<std::string,7>> records;
        while(std::getline(input,text)) {
            std::istringstream row{text};std::array<std::string,7> fields;
            for(auto& field:fields)Check(bool(std::getline(row,field,',')),"Truncated thread CPU record");
            records.push_back(std::move(fields));
        }
        return records;
    };
    auto triggered_cpu=cpu_rows(path);
    Check(triggered_cpu.size()==2,"Thread CPU records escaped the exact triggered frame window");
    for(unsigned i=0;i<2;++i)Check(std::stoul(triggered_cpu[i][0])==i+2 && triggered_cpu[i][4]=="1" && std::stod(triggered_cpu[i][3])>=0,
        "Triggered CPU frame did not use a valid owner sample from its trigger boundary");
    std::ifstream workers(path.string()+".worker-cpu.csv");
    Check(bool(std::getline(workers,line)) && line=="frame,queued_draws,encoded_draws,queued_dispatches,encoded_dispatches,worker_cpu_ns,profiled_jobs,separate_thread,buffer_readonly_passes,conservative_passes",
          "Encoding-worker profile schema missing or changed");
    unsigned worker_rows{};
    while(std::getline(workers,line)) {
        std::istringstream row{line};std::array<std::string,10> fields;
        for(auto& field:fields)Check(bool(std::getline(row,field,',')),"Truncated encoding-worker CPU record");
        Check(std::stoul(fields[0])==worker_rows+2 && fields[1]=="0" && fields[2]=="0" &&
              fields[3]=="1" && fields[4]=="1" && std::stoull(fields[5])>0 && fields[6]=="3" &&
              fields[7]=="1" && fields[8]=="0" && fields[9]=="0",
              "Worker CPU/command deltas lost a native job or included pre-trigger/inactive work");
        ++worker_rows;
    }
    Check(worker_rows==2,"Encoding-worker CPU records escaped the triggered frame window");
    unsetenv("AZAHAR_METAL4_PROFILE_TRIGGER");unsetenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT");
    const auto sleeping=std::filesystem::path(directory)/"sleeping.csv";
    setenv("AZAHAR_METAL4_PROFILE",sleeping.c_str(),1);
    {
        Metal::Scheduler queue{device};
        queue.EndProfileFrame(); // First sample has no previous owner CPU baseline.
        {
            Scope root{queue,Phase::Draw};Scope nested{queue,Phase::Transfer};
            std::uint64_t value=0xabc123;
            for(unsigned i=0;i<100000;++i)value=(value*0x100000001b3ULL)^i;
            Check(value!=0,"Bounded CPU fixture lost its observable work");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        queue.EndProfileFrame();
        queue.ReleaseThread();
        bool transferred=false;
        std::thread owner{[&]{try{queue.EndProfileFrame();queue.EndProfileFrame();queue.ReleaseThread();transferred=true;}catch(const std::exception&){} }};
        owner.join();Check(transferred,"Thread CPU diagnostics changed explicit scheduler handoff");
        queue.EndProfileFrame();
    }
    const auto sleeping_cpu=cpu_rows(sleeping);
    Check(sleeping_cpu.size()==5 && sleeping_cpu[0][4]=="0" && sleeping_cpu[0][3]=="-1" &&
          sleeping_cpu[1][4]=="1" && sleeping_cpu[2][4]=="0" && sleeping_cpu[2][3]=="-1" &&
          sleeping_cpu[3][4]=="1" && sleeping_cpu[4][4]=="0" && sleeping_cpu[4][3]=="-1",
          "First samples or owner transfers subtracted unrelated thread CPU clocks");
    const auto& slept=sleeping_cpu[1];
    const auto slept_wall=(std::stoll(slept[2])-std::stoll(slept[1]))/1000000.0;
    const auto slept_cpu=std::stod(slept[3]);
    Check(slept_wall>=45 && slept_cpu>0 && slept_cpu<slept_wall*.5,
          "Thread CPU timing counted controlled sleep as owner CPU work");
    std::ifstream sleeping_phases(sleeping.string()+".host-phases.csv");std::getline(sleeping_phases,line);
    Check(bool(std::getline(sleeping_phases,line)),"Controlled sleep root was not measured");
    std::istringstream sleeping_row{line};std::array<std::string,6> sleeping_fields;
    for(auto& field:sleeping_fields)Check(bool(std::getline(sleeping_row,field,',')),"Truncated controlled CPU root");
    Check(sleeping_fields[0]=="1" && sleeping_fields[1]=="draw" && sleeping_fields[2]=="1" &&
          sleeping_fields[5]=="1" && std::stod(sleeping_fields[3])>=45 &&
          std::stod(sleeping_fields[4])>0 && std::stod(sleeping_fields[4])<std::stod(sleeping_fields[3])*.5 &&
          !std::getline(sleeping_phases,line),
          "CPU root counted nested work twice or measured sleep as active CPU time");
    std::cout<<"Controlled owner CPU: wall="<<slept_wall<<"ms cpu="<<slept_cpu<<"ms; valid sample/explicit owner handoff/trigger bounds passed\n";
    const auto unclaimed=std::filesystem::path(directory)/"unclaimed.csv";
    setenv("AZAHAR_METAL4_PROFILE",unclaimed.c_str(),1);
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        const auto output=device.CreateBuffer(64,true);
        queue.ReleaseThread(); // Match the renderer's explicit frontend handoff.
        {Scope metadata{queue,Phase::State};} // No GPU owner exists yet.
        bool encoded=false;
        std::thread actual_owner{[&]{try {
            transfer.FillBuffer(output,NSMakeRange(0,64),77);queue.Finish();queue.ReleaseThread();encoded=true;
        }catch(const std::exception&){} }};
        actual_owner.join();Check(encoded,"Host metadata timing claimed GPU ownership before frontend handoff");
        queue.EndProfileFrame();
        for(unsigned i=0;i<64;++i)Check(static_cast<const unsigned char*>(output.contents)[i]==77,
            "Host metadata timing changed owner-handoff GPU output");
    }
    std::ifstream unclaimed_file(unclaimed.string()+".host-phases.csv");
    Check(bool(std::getline(unclaimed_file,line)) && !std::getline(unclaimed_file,line),
        "Host phase timing charged metadata before a GPU owner existed");
    const auto disabled=std::filesystem::path(directory)/"disabled.csv";
    setenv("AZAHAR_METAL4_PROFILE",disabled.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_HOST_PHASES","0",1);
    {Metal::Scheduler queue{device};{Scope root{queue,Phase::Present};}queue.EndProfileFrame();}
    Check(!std::filesystem::exists(disabled.string()+".host-phases.csv"),"Disabled host phase timing created a file");
    Check(!std::filesystem::exists(disabled.string()+".thread-cpu.csv"),"Disabled host phase timing created a CPU file");
    Check(!std::filesystem::exists(disabled.string()+".worker-cpu.csv"),"Disabled host timing created a worker CPU file");
    setenv("AZAHAR_METAL4_PROFILE_HOST_PHASES","bad",1);bool invalid=false;
    try{Metal::Scheduler queue{device};}catch(const std::invalid_argument&){invalid=true;}
    Check(invalid,"Host phase timing accepted an invalid mode");
    unsetenv("AZAHAR_METAL4_PROFILE");setenv("AZAHAR_METAL4_PROFILE_HOST_PHASES","1",1);
    {Metal::Scheduler queue{device};{Scope root{queue,Phase(999)};}Check(!queue.ProfilingEnabled(),"Host phase mode alone enabled profiling");}
    std::filesystem::remove_all(directory);
    std::cout<<"Host phase attribution: exclusive nested roots, two exact triggered frames, exception unwind, owner/invalid/boundary rejection, unclaimed frontend handoff, disabled/no-path modes and128 GPU bytes passed\n";
}

void ValidateWaitProfiles(Metal::Device& device) {
    struct Environment {
        std::array<const char*, 4> keys{"AZAHAR_METAL4_PROFILE", "AZAHAR_METAL4_PROFILE_TRIGGER",
            "AZAHAR_METAL4_PROFILE_FRAME_LIMIT", "AZAHAR_METAL4_PROFILE_HOST_ONLY"};
        std::array<std::optional<std::string>, 4> old;
        Environment() { for (unsigned i=0;i<keys.size();++i) {
            if (const auto* value=std::getenv(keys[i])) old[i]=value; unsetenv(keys[i]);
        }}
        ~Environment() { for (unsigned i=0;i<keys.size();++i) {
            if (old[i]) setenv(keys[i],old[i]->c_str(),1); else unsetenv(keys[i]);
        }}
    } environment;
    char directory[]="/private/tmp/azahar-wait-profile-test-XXXXXX";
    Check(mkdtemp(directory),"Unable to create isolated wait profile directory");
    const auto path=std::filesystem::path(directory)/"host.csv", trigger=std::filesystem::path(directory)/"start";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_TRIGGER",trigger.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","2",1);
    setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    unsigned default_readback_line{};
    const auto repeated_readback_site = std::source_location::current();
    {
        Metal::Scheduler queue{device}; Metal::TransferRuntime transfer{device,queue};
        const auto output=device.CreateBuffer(256,true);
        queue.Wait(0); // No record before the trigger.
        queue.Finish(Metal::Scheduler::WaitReason::Readback, repeated_readback_site);
        { std::ofstream signal(trigger); signal<<"start"; }
        queue.EndProfileFrame();
        Check(queue.ProfilingEnabled(),"Wait profile did not arm at the frame boundary");
        queue.Wait(0);
        transfer.FillBuffer(output,NSMakeRange(0,256),0);
        default_readback_line = __LINE__ + 1;
        queue.Finish(Metal::Scheduler::WaitReason::Readback);
        const auto retired=queue.CurrentTick()-1;
        queue.Wait(retired); // Completed-tick fast path must preserve native bytes.
        const auto unchanged=queue.CurrentTick();
        for (const auto invalid : {Metal::Scheduler::WaitReason::Count,
                                  static_cast<Metal::Scheduler::WaitReason>(~0u)}) {
            bool wait_rejected{}, finish_rejected{};
            try { queue.Wait(unchanged,invalid); } catch (const std::invalid_argument&) { wait_rejected=true; }
            try { queue.Finish(invalid); } catch (const std::invalid_argument&) { finish_rejected=true; }
            Check(wait_rejected && finish_rejected && queue.CurrentTick()==unchanged,
                  "Invalid wait classification submitted work or changed the timeline");
        }
        // All three CPU-upload pages belong to the same unsubmitted tick.
        // The fourth allocation must submit/wait before overwriting page zero.
        Metal::StreamBuffer uploads{device,queue,64};
        for (unsigned i=0;i<4;++i) {
            auto slice=uploads.Allocate(64);
            std::memset(slice.data,0x31+i,64);
            transfer.CopyBuffer(slice.buffer,slice.offset,output,i*64,64);
        }
        queue.Finish(Metal::Scheduler::WaitReason::Readback, repeated_readback_site);
        for (unsigned retired=3;retired<Metal::Scheduler::SubmissionSlotCount;++retired) {
            transfer.FillBuffer(output,NSMakeRange(192,64),0x34);
            queue.Finish(Metal::Scheduler::WaitReason::Readback, repeated_readback_site);
        }
        for (unsigned i=0;i<256;++i) Check(static_cast<const u8*>(output.contents)[i]==0x31+i/64,
            "Upload-page wait profiling changed queued immutable snapshots");
        queue.EndProfileFrame();
        // The first command slot is retired now; exercising actual slot reuse
        // must be attributed separately from a CPU result read and page reuse.
        transfer.FillBuffer(output,NSMakeRange(0,64),0x72);
        queue.Finish();
        queue.EndProfileFrame();
        Check(!queue.ProfilingEnabled(),"Wait profile exceeded its two-frame limit");
        queue.Wait(retired); queue.EndProfileFrame(); // No post-limit record.
        queue.Finish(Metal::Scheduler::WaitReason::Readback, repeated_readback_site);
        Check(static_cast<const u8*>(output.contents)[0]==0x72,
              "Command-slot fast path changed GPU execution");
    }
    const auto fields=[](const std::string& line) {
        std::istringstream input{line}; std::vector<std::string> result; std::string field;
        while (std::getline(input,field,',')) result.push_back(field); return result;
    };
    std::map<std::pair<unsigned,std::string>,std::vector<std::string>> records;
    std::map<unsigned,unsigned> calls;
    std::ifstream waits(path.string()+".waits.csv"); std::string line; std::getline(waits,line);
    Check(fields(line).size()==10,"Wait profile schema lost attribution or component timings");
    while (std::getline(waits,line)) {
        auto f=fields(line); Check(f.size()==10,"Malformed wait attribution row");
        const auto frame=std::stoul(f[0]); Check(frame==1 || frame==2,"Wait profile captured disabled frames");
        for (unsigned i=7;i<10;++i) Check(std::isfinite(std::stod(f[i])) && std::stod(f[i])>=0,
            "Wait profile contains invalid elapsed host time");
        Check(std::stod(f[8])<=std::stod(f[7])+0.000001 && std::stod(f[9])<=std::stod(f[7])+0.000001,
              "Wait component exceeds its inclusive caller interval");
        calls[frame]+=std::stoul(f[2]);
        Check(records.emplace(std::pair{frame,f[1]},std::move(f)).second,"Duplicate wait-reason frame row");
    }
    const auto& explicit_wait=records.at({1,"explicit"});
    Check(std::stoul(explicit_wait[2])==2 && std::stoul(explicit_wait[3])==1 && std::stoul(explicit_wait[4])==1,
          "Completed/zero ticks were not distinguished from OS waits");
    const auto& page=records.at({1,"upload_page"});
    Check(std::stoul(page[2])==1 && std::stoul(page[5])==1,
          "Unsubmitted CPU-upload recycling did not retain its own wait reason");
    Check(std::stoul(records.at({1,"readback"})[2])==Metal::Scheduler::SubmissionSlotCount-1 &&
          std::stoul(records.at({2,"drain"})[2])==1 &&
          std::stoul(records.at({2,"command_slot"})[4])==1,
          "Readback, lifecycle drain or completed command-slot reuse was misattributed");
    std::ifstream frames(path.string()+".frames.csv"); std::getline(frames,line); unsigned rows{};
    while (std::getline(frames,line)) {
        const auto f=fields(line); Check(f.size()==11 && calls.at(std::stoul(f[0]))==std::stoul(f[7]),
            "Wait reason totals disagree with existing aggregate frame counts"); ++rows;
    }
    Check(rows==2,"Wait profile lost its exact bounded frame coverage");
    std::ifstream sites(path.string()+".readbacks.csv");
    Check(bool(std::getline(sites,line)) && line=="frame,file,line,calls,finish_host_ms",
          "Readback callsite profile has the wrong schema");
    std::map<unsigned,unsigned> site_calls;
    double site_ms{};
    while (std::getline(sites,line)) {
        const auto f=fields(line);
        Check(f.size()==5 && f[0]=="1" && f[1]=="mtl_scheduler_tests.mm",
              "Readback profile captured inactive, post-limit or unrelated calls");
        const auto source_line=std::stoul(f[2]),count=std::stoul(f[3]);
        Check(source_line==default_readback_line || source_line==repeated_readback_site.line(),
              "Readback source_location identifies Finish instead of its caller");
        Check(site_calls.emplace(source_line,count).second && std::isfinite(std::stod(f[4])) && std::stod(f[4])>=0,
              "Readback callsite record is duplicated or has invalid elapsed time");
        site_ms+=std::stod(f[4]);
    }
    Check(site_calls.size()==2 && site_calls.at(default_readback_line)==1 &&
          site_calls.at(repeated_readback_site.line())==Metal::Scheduler::SubmissionSlotCount-2,
          "Readback callsites did not retain default/repeated caller grouping");
    Check(site_ms+0.000001>=std::stod(records.at({1,"readback"})[7]),
          "Inclusive Finish callsites lost part of the nested readback Wait interval");
    std::filesystem::remove_all(directory);
    std::cout<<"Native wait attribution: exact two-frame trigger/limit, actual three-page unsubmitted reuse, full256 output bytes, completed/zero tick fast paths, readback/drain/slot separation, default/repeated callsite and aggregate reconciliation, inactive suppression and four pre-submit reason rejections passed\n";
}

void ValidateDrawableProfiles(Metal::Device& device) {
    struct RestoreEnvironment {
        std::optional<std::string> value;
        RestoreEnvironment() {
            if (const auto* original = std::getenv("AZAHAR_METAL4_PROFILE")) value = original;
        }
        ~RestoreEnvironment() {
            if (value) setenv("AZAHAR_METAL4_PROFILE", value->c_str(), 1);
            else unsetenv("AZAHAR_METAL4_PROFILE");
        }
    } restore;
    unsetenv("AZAHAR_METAL4_PROFILE");
    {
        Metal::Scheduler disabled{device};
        const auto begin = disabled.BeginProfileDrawableAcquire();
        Check(!begin.begin_host_ns && !begin.tick, "Disabled drawable profiling read a timestamp");
        disabled.EndProfileDrawableAcquire(begin, 999, false);
        Check(disabled.CurrentTick() == 1, "Disabled drawable profiling submitted GPU work");
    }
    char directory[] = "/private/tmp/azahar-drawable-profile-test-XXXXXX";
    Check(mkdtemp(directory), "Unable to create isolated drawable profile test directory");
    const auto path = std::filesystem::path(directory) / "gpu.csv";
    setenv("AZAHAR_METAL4_PROFILE", path.c_str(), 1);
    {
        Metal::Scheduler scheduler{device};
        Metal::TransferRuntime transfer{device, scheduler};
        const auto empty = scheduler.BeginProfileDrawableAcquire();
        scheduler.EndProfileDrawableAcquire(empty, 0, false);
        auto output = device.CreateBuffer(64, true);
        transfer.FillBuffer(output, NSMakeRange(0, 64), 0x63);
        const auto pending = scheduler.BeginProfileDrawableAcquire();
        Check(pending.commands_pending && pending.frame == 0 && pending.tick == 1,
              "Drawable profiling did not identify pending unsubmitted GPU work");
        bool begin_rejected{}, end_rejected{};
        std::thread foreign{[&] {
            try { (void)scheduler.BeginProfileDrawableAcquire(); }
            catch (const std::logic_error&) { begin_rejected = true; }
            try { scheduler.EndProfileDrawableAcquire(pending, 0, false); }
            catch (const std::logic_error&) { end_rejected = true; }
        }};
        foreign.join();
        Check(begin_rejected && end_rejected, "Drawable profiling accepted a foreign owner");
        bool bad_window{}, future{};
        try { scheduler.EndProfileDrawableAcquire(pending, 2, false); }
        catch (const std::invalid_argument&) { bad_window = true; }
        auto invalid = pending;
        invalid.begin_host_ns = std::numeric_limits<std::int64_t>::max();
        try { scheduler.EndProfileDrawableAcquire(invalid, 0, false); }
        catch (const std::invalid_argument&) { future = true; }
        Check(bad_window && future, "Drawable profiling accepted invalid index/time");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        scheduler.EndProfileDrawableAcquire(pending, 1, true);
        Check(scheduler.CurrentTick() == 1, "Drawable timing prematurely submitted pending GPU work");
        scheduler.SubmitPending();
        scheduler.Finish();
        scheduler.EndProfileFrame();
        for (unsigned i = 0; i < 64; ++i)
            Check(static_cast<const unsigned char*>(output.contents)[i] == 0x63,
                  "Drawable profiling lost pending GPU work");
        bool stale{};
        try { scheduler.EndProfileDrawableAcquire(pending, 0, false); }
        catch (const std::invalid_argument&) { stale = true; }
        Check(stale, "Drawable profiling assigned a stale interval to a new frame");
        const auto retired = scheduler.BeginProfileDrawableAcquire();
        Check(!retired.commands_pending && retired.frame == 1 && retired.tick == 2 &&
              retired.completed_tick == 1, "Drawable profiling lost retired frame/timeline attribution");
        scheduler.EndProfileDrawableAcquire(retired, 0, false);
    }
    std::ifstream csv(path.string() + ".drawables.csv");
    std::string line;
    Check(bool(std::getline(csv, line)) && line ==
        "frame,window,tick,commands_pending,completed_before,completed_after,begin_host_ns,end_host_ns,acquire_host_ms,available",
        "Drawable profile CSV has an invalid schema");
    unsigned rows{};
    while (std::getline(csv, line)) {
        std::istringstream input{line};
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(input, field, ',')) fields.push_back(field);
        Check(fields.size() == 10 && rows < 3, "Drawable profile added malformed or rejected rows");
        const auto begin = std::stoll(fields[6]), end = std::stoll(fields[7]);
        const auto elapsed = std::stod(fields[8]);
        Check(begin > 0 && end >= begin && std::isfinite(elapsed) &&
              std::abs(elapsed - double(end - begin) / 1000000.0) < 1e-8,
              "Drawable profile host interval has inconsistent units or endpoints");
        Check(std::stoul(fields[0]) == (rows == 2 ? 1u : 0u) &&
              std::stoul(fields[1]) == (rows == 1 ? 1u : 0u) &&
              std::stoul(fields[2]) == (rows == 2 ? 2u : 1u) &&
              std::stoul(fields[3]) == (rows == 1 ? 1u : 0u) &&
              std::stoul(fields[9]) == (rows == 1 ? 1u : 0u),
              "Drawable profile lost frame, window, submission, pending or availability identity");
        if (rows == 1) Check(elapsed >= 2.0, "Drawable interval omitted measured host blocking");
        ++rows;
    }
    Check(rows == 3, "Drawable profile lost valid records");
    std::filesystem::remove_all(directory);
    std::cout << "Drawable host profiles: disabled gate, three frame/window/tick intervals, five rejected calls, pending native GPU bytes and retired timeline passed\n";
}
}

void ValidateComputeBatches(Metal::Device& device) {
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void batch_xor(const device uint* input [[buffer(0)]],device uint* output [[buffer(1)]]) {
    output[0]=input[0]^0x51a7e3d9u;
}
struct V { float4 position [[position]]; uint value [[flat]]; };
vertex V batch_vertex(uint id [[vertex_id]],const device uint* input [[buffer(0)]]) {
    const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return {float4(p[id],0,1),input[0]};
}
fragment uint batch_fragment(V v [[stage_in]]) { return v.value; }
kernel void batch_read(texture2d<uint,access::read> input [[texture(0)]],device uint* output [[buffer(0)]],
                       uint x [[thread_position_in_grid]]) { output[x]=input.read(uint2(x,0)).r; }
)","Compute batch dispatch/blit/render WAR fixture");
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"batch_xor");
    const auto compute=device.CompileCompute(cd);
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"batch_read");
    const auto read=device.CompileCompute(cd);
    auto* rd=[[MTL4RenderPipelineDescriptor alloc] init];
    rd.vertexFunctionDescriptor=Metal::Device::Function(library,"batch_vertex");
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"batch_fragment");
    rd.colorAttachments[0].pixelFormat=MTLPixelFormatR32Uint;
    const auto render=device.CompileRender(rd);
    constexpr unsigned count=257,guard=0xabcdef01u;
    Metal::Scheduler queue{device};
    auto source=device.CreateBuffer(count*4,true),scratch=device.CreateBuffer(16,false);
    auto output=device.CreateBuffer((count+2)*4,true),readback=device.CreateBuffer(count*4,true);
    for(unsigned i=0;i<count;++i)static_cast<unsigned*>(source.contents)[i]=i*7919u+37u;
    std::fill_n(static_cast<unsigned*>(output.contents),count+2,guard);
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:count height:1 mipmapped:NO];
    td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    const auto texture=device.CreateTexture(td);
    queue.KeepAlive(compute);queue.KeepAlive(read);queue.KeepAlive(render);
    for(unsigned i=0;i<count;++i) {
        Metal::Scheduler::ComputeBatchScope batch{queue};
        const auto before=queue.EncoderCount();
        auto table=queue.MakeArguments(std::array{Metal::BufferBinding{source,i*4,0},Metal::BufferBinding{scratch,0,1}});
        const auto first=queue.BeginCompute(compute);
        [first setComputePipelineState:compute];[first setArgumentTable:table];
        [first dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        queue.EndCompute();
        if(i%19==0)queue.EndEncoding(); // An explicit boundary must still work inside the scope.
        const auto blit=queue.BeginCompute();
        [blit copyFromBuffer:scratch sourceOffset:0 toBuffer:scratch destinationOffset:4 size:4];
        queue.EndCompute();
        try { Metal::Scheduler::ComputeBatchScope nested{queue};throw std::runtime_error("nested unwind"); }
        catch(const std::runtime_error&) {}
        queue.RebindBufferArguments(table,std::array{Metal::BufferBinding{scratch,4,0},Metal::BufferBinding{output,(i+1)*4,1}});
        const auto second=queue.BeginCompute(compute);
        [second setComputePipelineState:compute];[second setArgumentTable:table];
        [second dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        queue.EndCompute();
        Check(second==blit && queue.EncoderCount()==before+(i%19==0 ? 2:1),
              "Compute batch did not preserve its encoder across dependent dispatch/blit operations");
        if(i%17==0)queue.SubmitPending(); // Submission must force the pending encoder to finish.
        auto* pass=[[MTL4RenderPassDescriptor alloc] init];
        pass.renderTargetWidth=count;pass.renderTargetHeight=1;pass.defaultRasterSampleCount=1;
        pass.colorAttachments[0].texture=texture;pass.colorAttachments[0].loadAction=i ? MTLLoadActionLoad:MTLLoadActionClear;
        pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        table=queue.MakeArguments(std::array{Metal::BufferBinding{scratch,4,0}});
        const auto draw=queue.BeginRender(pass);
        [draw setRenderPipelineState:render];[draw setArgumentTable:table atStages:MTLRenderStageVertex];
        [draw setViewport:MTLViewport{0,0,count,1,0,1}];[draw setScissorRect:MTLScissorRect{i,0,1,1}];
        [draw drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        queue.EndEncoding();
        table=queue.MakeArguments(std::array{Metal::BufferBinding{readback,0,0}},std::array{Metal::TextureBinding{texture,0}});
        const auto reader=queue.BeginCompute(read);
        [reader setComputePipelineState:read];[reader setArgumentTable:table];
        [reader dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        queue.EndCompute();
        // The next iteration overwrites scratch consumed by the preceding render.
    }
    queue.Finish();
    const auto* words=static_cast<const unsigned*>(output.contents);
    Check(words[0]==guard && words[count+1]==guard,"Compute batch damaged output guards");
    for(unsigned i=0;i<count;++i) {
        const auto value=i*7919u+37u;
        Check(words[i+1]==value && static_cast<const unsigned*>(readback.contents)[i]==(value^0x51a7e3d9u),
              "Compute batch changed dependent dispatch/blit results or render-read/scratch-write ordering");
    }
    std::cout<<"Compute batches:257 dispatch/blit/dispatch/render/readback chains, scratch WAR, nested unwind, explicit encoder ends and16 cross-slot submissions exact\n";
}

void ValidateStageDependencies(Metal::Device& device, bool use_gate=true) {
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
struct StageV { float4 position [[position]]; uint value [[flat]]; };
vertex StageV stage_vertex(uint id [[vertex_id]],const device uint* input [[buffer(0)]]) {
    const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return {float4(p[id],0,1),input[0]};
}
fragment uint stage_fragment(StageV v [[stage_in]]) { return v.value; }
fragment uint stage_writer(StageV v [[stage_in]],device uint* output [[buffer(1)]]) {
    output[0]=v.value^0x51a7e3d9u;return v.value;
}
kernel void stage_mutate(device uint* value [[buffer(0)]]) { value[0]^=0x13579bdfu; }
kernel void stage_read(texture2d<uint,access::read> image [[texture(0)]],
    device uint* output [[buffer(0)]],constant uint& x [[buffer(1)]]) {output[x+1]=image.read(uint2(x,0)).r;}
)","Native split-stage RAW WAR WAW fixture");
    auto* rd=[[MTL4RenderPipelineDescriptor alloc] init];
    rd.vertexFunctionDescriptor=Metal::Device::Function(library,"stage_vertex");
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"stage_fragment");
    rd.colorAttachments[0].pixelFormat=MTLPixelFormatR32Uint;
    const auto render=device.CompileRender(rd);
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"stage_writer");
    const auto writer=device.CompileRender(rd);
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"stage_mutate");
    const auto mutate=device.CompileCompute(cd);
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"stage_read");
    const auto read=device.CompileCompute(cd);
    constexpr unsigned count=192,width=count*3,guard=0xdecafbad;
    const auto proof=Metal::SchedulerTestAccess::ReadOnlyBuffers();
    for(bool narrowed:{false,true}) {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
            width:width height:1 mipmapped:NO];
        td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
        const auto target=device.CreateTexture(td);
        td.width=1;const auto scratch=device.CreateTexture(td);
        transfer.ClearColor(target,MTLClearColorMake(0,0,0,0));
        transfer.ClearColor(scratch,MTLClearColorMake(0,0,0,0));
        const auto make_pass=[&](id<MTLTexture> texture) {
            auto* pass=[[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture=texture;pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;return pass;
        };
        const auto pass=queue.SnapshotRenderPass(make_pass(target));
        auto* opaque_pass=make_pass(scratch);
        const auto opaque=queue.SnapshotRenderPass(opaque_pass);
        auto state=device.CreateBuffer(4,true),observed=device.CreateBuffer((width+2)*4,true);
        auto disjoint=device.CreateBuffer(64,true);
        std::memset(disjoint.contents,0xa7,64);
        std::fill_n(static_cast<unsigned*>(observed.contents),width+2,guard);
        std::vector<unsigned> expected(width);
        const auto before=queue.WorkerStats();
        auto gate=[device.Handle() newSharedEvent];
        struct Release {id<MTLSharedEvent> event;~Release(){event.signaledValue=1;}} release{gate};
        if(use_gate) Metal::SchedulerTestAccess::BlockQueue(queue,gate);
        const auto pure=[&](unsigned x) {
            const std::array bindings{Metal::BufferBinding{state,0,0}};
            const Metal::RenderCommand command{.pipeline=render,.depth=nil,.viewport={0,0,width,1,0,1},
                .scissor={x,0,1,1},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
            queue.QueueRender(*pass,command,bindings,{},{},true,{},narrowed ? &proof:nullptr);
        };
        for(unsigned i=0;i<count;++i) {
            const auto value=0x01010101u*((i*13+7)&255);
            transfer.FillBuffer(state,NSMakeRange(0,4),std::uint8_t(value));
            pure(i*3);expected[i*3]=value;
            // WAR: finish the previous vertex read before overwriting its
            // exact buffer. Exercise direct, one-shot and borrowed encoders.
            const std::array bindings{Metal::BufferBinding{state,0,0}};
            if(i%3==0) {
                const auto table=queue.MakeArguments(bindings);
                queue.KeepAlive(mutate);const auto encoder=queue.BeginCompute(mutate);
                [encoder setComputePipelineState:mutate];[encoder setArgumentTable:table];
                [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                queue.EndCompute();
            } else {
                std::optional<Metal::Scheduler::ComputeBatchScope> batch;
                if(i%3==2)batch.emplace(queue);
                Metal::ComputeDispatch dispatch;dispatch.pipeline=mutate;
                dispatch.buffers.assign(bindings.begin(),bindings.end());
                queue.QueueCompute(std::span{&dispatch,1});
            }
            pure(i*3+1);expected[i*3+1]=value^0x13579bdfu;
            // Opaque fragment writes feed a later pure vertex: no weaker
            // dependency is allowed just because the consumer has a proof.
            const std::array write_bindings{Metal::BufferBinding{state,0,0},Metal::BufferBinding{state,0,1}};
            if(i%2) {
                const Metal::RenderCommand command{.pipeline=writer,.depth=nil,.viewport={0,0,1,1,0,1},
                    .scissor={0,0,1,1},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
                queue.QueueRender(*opaque,command,write_bindings,{},{},false);
            } else {
                const auto table=queue.MakeArguments(write_bindings);
                queue.KeepAlive(writer);const auto encoder=queue.BeginRender(opaque_pass);
                [encoder setRenderPipelineState:writer];
                [encoder setArgumentTable:table atStages:MTLRenderStageVertex|MTLRenderStageFragment];
                [encoder setViewport:MTLViewport{0,0,1,1,0,1}];[encoder setScissorRect:MTLScissorRect{0,0,1,1}];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];queue.EndEncoding();
            }
            pure(i*3+2);expected[i*3+2]=value^0x13579bdfu^0x51a7e3d9u;
            auto position=device.CreateBuffer(4,true);*static_cast<unsigned*>(position.contents)=i*3+2;
            const std::array read_bindings{Metal::BufferBinding{observed,0,0},Metal::BufferBinding{position,0,1}};
            const std::array textures{Metal::TextureBinding{target,0}};
            {
                std::optional<Metal::Scheduler::ComputeBatchScope> outer;
                if(i%4<2) {
                    outer.emplace(queue);
                    Metal::Scheduler::BufferComputeScope inner{queue};
                    const std::array inputs{Metal::BufferBinding{disjoint,0,0}};
                    const auto table=queue.MakeArguments(inputs);
                    queue.KeepAlive(mutate);const auto encoder=queue.BeginCompute(mutate);
                    [encoder setComputePipelineState:mutate];[encoder setArgumentTable:table];
                    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    queue.EndCompute(); // Keep this encoder open past inner scope exit.
                }
                if(i%4==1) {
                    Metal::ComputeDispatch dispatch;dispatch.pipeline=read;
                    dispatch.buffers.assign(read_bindings.begin(),read_bindings.end());
                    dispatch.textures.assign(textures.begin(),textures.end());
                    queue.QueueCompute(std::span{&dispatch,1});
                } else {
                    const auto table=queue.MakeArguments(read_bindings,textures);
                    queue.KeepAlive(read);const auto encoder=queue.BeginCompute(read);
                    [encoder setComputePipelineState:read];[encoder setArgumentTable:table];
                    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];queue.EndCompute();
                }
            }
            if(i%17==16) {
                const auto tick=queue.Submit();
                if(i==16 && use_gate) {Check(!queue.IsComplete(tick),"Stage fixture lost external GPU gate");gate.signaledValue=1;}
            }
        }
        queue.Finish();
        const auto bytes=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,width,1)},width*4);
        Check(std::memcmp(bytes.data(),expected.data(),width*4)==0,"Split stages changed RAW/WAR/WAW raster pixels");
        const auto* words=static_cast<const unsigned*>(observed.contents);
        for(unsigned x=0;x<64;++x)
            Check(static_cast<const unsigned char*>(disjoint.contents)[x]==0xa7,
                  "Nested buffer/general transitions changed disjoint values or guards");
        Check(words[0]==guard && words[width+1]==guard,"Split stages overwrote readback guards");
        for(unsigned x=0;x<width;++x)
            Check(words[x+1]==(x%3==2 ? expected[x]:guard),"Split stages changed fragment-to-compute values/guards");
        const auto after=queue.WorkerStats();
        Check(after.buffer_readonly_render_passes-before.buffer_readonly_render_passes==(narrowed ? width:0),
              "Read-only render proof did not control the actual native pass path");
        const auto switch_before=queue.EncoderCount();
        const std::array switch_bindings{Metal::BufferBinding{state,0,0}};
        for(unsigned x=0;x<3;++x) {
            const Metal::RenderCommand command{.pipeline=render,.depth=nil,.viewport={0,0,width,1,0,1},
                .scissor={x,0,1,1},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
            queue.QueueRender(*pass,command,switch_bindings,{},{},true,{},x==1 ? &proof:nullptr);
        }
        // Mode changes must close/reopen even though all three immutable
        // attachment keys match; check before readback adds its own encoders.
        Check(queue.EncoderCount()==switch_before+3,"Barrier-mode switch reused an incompatible live pass");
        const auto switched=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,3,1)},12);
        for(unsigned x=0;x<3;++x)
            Check(std::memcmp(switched.data()+x*4,&expected.back(),4)==0,"Barrier-mode switch changed complete pixels");
        // A heap-backed view cannot use the buffer-disjoint proof, even with
        // read-only shaders. Render/read it natively through the conservative path.
        const auto size=[device.Handle() heapTextureSizeAndAlignWithDescriptor:td];
        auto* hd=[[MTLHeapDescriptor alloc] init];hd.storageMode=MTLStorageModePrivate;
        hd.size=((size.size+size.align-1)/size.align)*size.align;
        const auto heap=[device.Handle() newHeapWithDescriptor:hd];
        const auto root=[heap newTextureWithDescriptor:td];
        Check(root!=nil,"Unable to create native heap alias fixture");
        const auto view=[root newTextureViewWithPixelFormat:MTLPixelFormatR32Uint];
        transfer.ClearColor(view,MTLClearColorMake(0,0,0,0));
        const auto heap_pass=queue.SnapshotRenderPass(make_pass(view));
        const std::array bindings{Metal::BufferBinding{state,0,0}};
        const Metal::RenderCommand command{.pipeline=render,.depth=nil,.viewport={0,0,1,1,0,1},
            .scissor={0,0,1,1},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
        const auto heap_before=queue.WorkerStats();
        queue.QueueRender(*heap_pass,command,bindings,{},{},true,{},&proof);
        const auto pixel=transfer.Readback({view,0,0,MTLRegionMake2D(0,0,1,1)},4);
        Check(std::memcmp(pixel.data(),&expected.back(),4)==0 &&
              queue.WorkerStats().buffer_readonly_render_passes==heap_before.buffer_readonly_render_passes &&
              queue.WorkerStats().conservative_render_passes==heap_before.conservative_render_passes+1,
              "Heap-backed attachment view incorrectly narrowed memory dependencies");
        std::cout<<"Split-stage mode="<<narrowed<<" gpu_gate="<<use_gate<<":192 mixed direct/queued/borrowed RAW/WAR/WAW chains,576 exact raster words,192 fragment readbacks plus guards,96 nested buffer/general transitions,3 barrier-mode switches,11 cross-slot submissions and native heap-view conservative path passed\n";
    }
}

void ValidateBufferComputeScope(Metal::Device& device) {
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; uint value [[flat]]; };
kernel void buffer_scope_compute(const device uint* input [[buffer(0)]],device uint* output [[buffer(1)]]) {
    output[0]=input[0]^0x51a7e3d9u;
}
vertex V buffer_scope_vertex(uint id [[vertex_id]],const device uint* input [[buffer(0)]]) {
    const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    return {float4(p[id],0,1),input[0]};
}
fragment uint buffer_scope_fragment(V v [[stage_in]]) { return v.value; }
kernel void buffer_scope_read(texture2d<uint,access::read> input [[texture(0)]],
    device uint* output [[buffer(0)]],uint x [[thread_position_in_grid]]) { output[x]=input.read(uint2(x,0)).r; }
)","Buffer compute/render ordering fixture");
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"buffer_scope_compute");
    const auto compute=device.CompileCompute(cd);
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"buffer_scope_read");
    const auto read=device.CompileCompute(cd);
    auto* rd=[[MTL4RenderPipelineDescriptor alloc] init];
    rd.vertexFunctionDescriptor=Metal::Device::Function(library,"buffer_scope_vertex");
    rd.fragmentFunctionDescriptor=Metal::Device::Function(library,"buffer_scope_fragment");
    rd.colorAttachments[0].pixelFormat=MTLPixelFormatR32Uint;
    const auto render=device.CompileRender(rd);
    constexpr unsigned count=257,guard=0xabcdef01u;
    for(bool scoped:{false,true}) {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        auto source=device.CreateBuffer(count*4,true),input=device.CreateBuffer(count*4,false);
        auto output=device.CreateBuffer((count+2)*4,true),readback=device.CreateBuffer(count*4,true);
        std::fill_n(static_cast<unsigned*>(output.contents),count+2,guard);
        for(unsigned i=0;i<count;++i)static_cast<unsigned*>(source.contents)[i]=i*7919u+37u;
        transfer.CopyBuffer(source,0,input,0,source.length); // Blit producer of scoped input.
        auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
            width:count height:1 mipmapped:NO];
        td.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        const auto texture=device.CreateTexture(td);
        queue.KeepAlive(compute);queue.KeepAlive(read);queue.KeepAlive(render);
        for(unsigned i=0;i<count;++i) {
            {
                std::optional<Metal::Scheduler::BufferComputeScope> scope;
                if(scoped)scope.emplace(queue);
                const std::array bindings{Metal::BufferBinding{input,i*4,0},Metal::BufferBinding{output,(i+1)*4,1}};
                const auto table=queue.MakeArguments(bindings);
                const auto encoder=queue.BeginCompute(compute);
                [encoder setComputePipelineState:compute];[encoder setArgumentTable:table];
                [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                queue.EndEncoding();
                if(scoped && i%17==0) {
                    try { Metal::Scheduler::BufferComputeScope nested{queue};throw std::runtime_error("scope unwind"); }
                    catch(const std::runtime_error&) {}
                }
            }
            auto* pass=[[MTL4RenderPassDescriptor alloc] init];
            pass.renderTargetWidth=count;pass.renderTargetHeight=1;pass.defaultRasterSampleCount=1;
            pass.colorAttachments[0].texture=texture;
            pass.colorAttachments[0].loadAction=i ? MTLLoadActionLoad:MTLLoadActionClear;
            pass.colorAttachments[0].storeAction=MTLStoreActionStore;
            const std::array bindings{Metal::BufferBinding{output,(i+1)*4,0}};
            const auto table=queue.MakeArguments(bindings);
            const auto encoder=queue.BeginRender(pass);
            [encoder setRenderPipelineState:render];
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
            [encoder setViewport:MTLViewport{0,0,count,1,0,1}];
            [encoder setScissorRect:MTLScissorRect{i,0,1,1}];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            queue.EndEncoding();
            // Scope exit must restore the render->compute dependency. Read all
            // pixels repeatedly, including after nested exception unwinding.
            const std::array reads{Metal::BufferBinding{readback,0,0}};
            const std::array textures{Metal::TextureBinding{texture,0}};
            const auto args=queue.MakeArguments(reads,textures);
            const auto reader=queue.BeginCompute(read);
            [reader setComputePipelineState:read];[reader setArgumentTable:args];
            [reader dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            queue.EndEncoding();
            if(i%17==16)queue.Submit();
        }
        queue.Finish();
        const auto* words=static_cast<const unsigned*>(output.contents);
        Check(words[0]==guard && words[count+1]==guard,"Scoped compute overwrote disjoint-range guards");
        for(unsigned i=0;i<count;++i) {
            const auto expected=(i*7919u+37u)^0x51a7e3d9u;
            Check(words[i+1]==expected && static_cast<const unsigned*>(readback.contents)[i]==expected,
                  "Scoped compute changed blit/vertex/fragment/readback ordering or output bytes");
        }
    }
    std::cout<<"Buffer compute scope: 514 compute/render/readback chains, nested unwind, 30 cross-slot submits and disjoint guards passed\n";
}

void ValidateCompletedPresentationResources(Metal::Device& device) {
    Metal::Scheduler scheduler{device};
    Metal::TransferRuntime transfer{device, scheduler};
    auto completed = std::make_shared<std::atomic_uint>();
    auto pending = std::make_shared<std::atomic_uint>();
    auto output = device.CreateBuffer(128, true);
    for (unsigned i = 0; i < Metal::Scheduler::SubmissionSlotCount - 1; ++i) {
        @autoreleasepool {
            auto* probe = [[LifetimeProbe alloc] init];
            probe->counter = completed;
            scheduler.KeepAlive(probe);
            transfer.FillBuffer(output, NSMakeRange(i * 16, 16), std::uint8_t(61 + i));
            scheduler.Submit();
        }
    }
    scheduler.Finish();
    Check(completed->load() == 0, "Fixture lost submitted resource before slot retirement");
    @autoreleasepool {
        auto* probe = [[LifetimeProbe alloc] init];
        probe->counter = pending;
        scheduler.KeepAlive(probe);
        transfer.FillBuffer(output, NSMakeRange(64, 64), 93);
    }
    const auto tick = scheduler.CurrentTick();
    const auto retained = scheduler.CurrentRetainedObjectCount();
    Check(!scheduler.SubmitPendingBeforeDrawable() && scheduler.CurrentTick() == tick &&
              scheduler.CurrentRetainedObjectCount() == retained && pending->load() == 0,
          "Completed-slot retirement submitted or released pending CPU/GPU resources");
    // A completed command's feedback block can finish releasing its independent
    // payload just after the shared event. No further submission/slot wrap is used.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed->load() != Metal::Scheduler::SubmissionSlotCount - 1 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(completed->load() == Metal::Scheduler::SubmissionSlotCount - 1,
          "Completed presentation resources remain pinned until a later slot wrap");
    scheduler.Submit();
    scheduler.Finish();
    const auto idle_tick = scheduler.CurrentTick();
    Check(!scheduler.SubmitPendingBeforeDrawable() && scheduler.CurrentTick() == idle_tick,
          "Retiring completed idle resources submitted an empty command buffer");
    const auto pending_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!pending->load() && std::chrono::steady_clock::now() < pending_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(pending->load() == 1, "Retired last presentation payload remains pinned");
    const auto* bytes = static_cast<const std::uint8_t*>(output.contents);
    for (unsigned i = 0; i < Metal::Scheduler::SubmissionSlotCount - 1; ++i)
        for (unsigned j = 0; j < 16; ++j)
            Check(bytes[i * 16 + j] == 61 + i, "Completed retirement changed earlier native GPU bytes");
    for (unsigned i = 64; i < 128; ++i)
        Check(bytes[i] == 93, "Completed retirement lost pending native GPU bytes");
    bool rejected{};
    std::thread other{[&] {
        try { scheduler.SubmitPendingBeforeDrawable(); }
        catch (const std::logic_error&) { rejected = true; }
    }};
    other.join();
    Check(rejected, "Completed-slot retirement bypassed scheduler thread ownership");
    std::cout << "Completed presentation retention: retired payloads released before slot wrap, pending payload protected, 112 native GPU bytes and foreign-owner rejection passed\n";
}

void ValidateLiveLibraryReuse(Metal::Device& device) {
    const std::string source = R"(#include <metal_stdlib>
using namespace metal;
struct Output { float4 position [[position]]; };
vertex Output live_vertex(uint n [[vertex_id]]) {
    const float2 positions[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
    return Output{float4(positions[n],0,1)};
}
fragment float4 live_fragment() { return float4(1,0,0,1); }
)";
    auto first = device.CompileLibrary(source, "Live native library full source");
    auto second = device.CompileLibrary(source, "Live native library full source");
    auto named = device.CompileLibrary(source, "Live native library other full name");
    auto green_source = source;
    constexpr std::string_view red = "float4(1,0,0,1)";
    const auto color = green_source.find(red);
    Check(color != std::string::npos, "Missing live-library color fixture");
    green_source.replace(color, red.size(), "float4(0,1,0,1)");
    auto green = device.CompileLibrary(green_source, "Live native library full source");
    Check(first == second && first != named && first != green,
          "Live native libraries lost full source/name identity or were rebuilt");
    Metal::PipelineCache pipelines{device};
    const auto pipeline = [&](id<MTLLibrary> library) {
        return pipelines.Render({library,"live_vertex",library,"live_fragment",MTLPixelFormatRGBA8Unorm,{}});
    };
    auto first_pipeline = pipeline(first), second_pipeline = pipeline(second), green_pipeline = pipeline(green);
    Check(first_pipeline == second_pipeline && first_pipeline != green_pipeline,
          "Full native pipeline key missed reused library or aliased a different source");
    for (unsigned n = 0; n < 64; ++n) {
        const auto library = device.CompileLibrary(source, "Live native library full source");
        Check(library == first && pipeline(library) == first_pipeline,
              "Repeated live library changed immutable native pipeline identity");
    }
    Check(pipelines.Stats().render_builds == 2, "Live native library repeatedly built a pipeline");
    std::array<id<MTLLibrary>,8> simultaneous{};
    std::array<std::jthread,8> workers;
    for (unsigned n = 0; n < workers.size(); ++n) workers[n] = std::jthread{[&,n] {
        @autoreleasepool { simultaneous[n] = device.CompileLibrary(source, "Concurrent live native library"); }
    }};
    for (auto& worker : workers) worker.join();
    for (auto library : simultaneous)
        Check(library == simultaneous[0], "Concurrent live-library publication duplicated native objects");

    Metal::Scheduler queue{device};
    Metal::TransferRuntime transfer{device,queue};
    auto* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget;
    const auto texture = device.CreateTexture(td);
    const auto draw = [&](id<MTLRenderPipelineState> state, unsigned r, unsigned g) {
        auto* pass = [[MTL4RenderPassDescriptor alloc] init];
        pass.renderTargetWidth = pass.renderTargetHeight = 4;
        pass.defaultRasterSampleCount = 1;
        pass.colorAttachments[0].texture = texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        queue.KeepAlive(state);
        auto encoder = queue.BeginRender(pass);
        [encoder setRenderPipelineState:state];
        [encoder setViewport:MTLViewport{0,0,4,4,0,1}];
        [encoder setScissorRect:MTLScissorRect{0,0,4,4}];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        queue.EndEncoding();
        const auto pixels = transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,4,4)},16);
        for (unsigned n = 0; n < 16; ++n)
            Check(pixels[n*4] == std::byte(r) && pixels[n*4+1] == std::byte(g) &&
                  pixels[n*4+2] == std::byte{0} && pixels[n*4+3] == std::byte{255},
                  "Live-library reuse changed full native GPU pixel output");
    };
    draw(first_pipeline,255,0);
    draw(green_pipeline,0,255);
    draw(second_pipeline,255,0);
    char directory[] = "/private/tmp/azahar-live-library-test-XXXXXX";
    Check(mkdtemp(directory), "Unable to create isolated live-library cache directory");
    bool persistence_checked{};
    {
        Metal::Device persisted{std::filesystem::path(directory),true};
        persistence_checked = persisted.DiskCacheStats().enabled;
        const auto* validation = std::getenv("MTL_SHADER_VALIDATION");
        Check(persistence_checked || (validation && std::string_view{validation} != "0"),
              "Native cache unexpectedly disabled outside shader instrumentation");
        persisted.SwitchDiskCache(0x12340001);
        auto original = persisted.CompileLibrary(source,"Published live native library");
        const auto identity = Metal::Device::SourceIdentity("Published live native library",source);
        Check(!persistence_checked || bool(Metal::RenderPipelineManifest::LoadLibrary(persisted.RenderDiskCacheDirectory(),identity)),
              "Initial live-library source was not published");
        if (persistence_checked) {
            const auto directory = persisted.RenderDiskCacheDirectory();
            std::filesystem::path library_file;
            for (const auto& item : std::filesystem::recursive_directory_iterator(directory / "libraries"))
                if (item.path().filename() == identity + ".library") library_file = item.path();
            Check(!library_file.empty(), "Published live-library file is missing");
            for (unsigned repeat = 0; repeat < 64; ++repeat)
                Check(persisted.CompileLibrary(source,"Published live native library") == original &&
                      persisted.SaveLibrarySource(original,directory), "Persisted live-library warm reuse changed identity");
            std::filesystem::remove(library_file);
            Check(persisted.CompileLibrary(source,"Published live native library") == original &&
                  bool(Metal::RenderPipelineManifest::LoadLibrary(directory,identity)),
                  "Warm persistence marker hid a removed source file");
            { std::ofstream damaged(library_file,std::ios::binary|std::ios::trunc); damaged << "corrupt"; }
            Check(persisted.SaveLibrarySource(original,directory) &&
                  bool(Metal::RenderPipelineManifest::LoadLibrary(directory,identity)),
                  "Warm persistence marker hid a replaced/corrupt source file");
        }
        persisted.SwitchDiskCache(0x12340002);
        auto switched = persisted.CompileLibrary(source,"Published live native library");
        Check(switched == original && (!persistence_checked || bool(Metal::RenderPipelineManifest::LoadLibrary(persisted.RenderDiskCacheDirectory(),identity))),
              "Title switch lost live reuse or source publication");
        Metal::PipelineCache after_switch{persisted};
        (void)after_switch.Render({switched,"live_vertex",switched,"live_fragment",MTLPixelFormatRGBA8Unorm,{}});
        persisted.SetDiskCacheEnabled(false);
        auto disabled = persisted.CompileLibrary(source,"Published live native library");
        persisted.SetDiskCacheEnabled(true);
        auto enabled = persisted.CompileLibrary(source,"Published live native library");
        Check(original == disabled && original == enabled &&
              (!persistence_checked || bool(Metal::RenderPipelineManifest::LoadLibrary(persisted.RenderDiskCacheDirectory(),identity))),
              "Disk-cache toggles lost valid live library or source publication");
        Check(persisted.FlushDiskCache(), "Live-library cache serializer did not flush");
    }
    std::filesystem::remove_all(directory);
    std::cout << "Live native library reuse: full source/name, 64 warm lookups, exact pipeline reuse, eight concurrent publications, 48 GPU pixels and title/disk-cache changes passed; source persistence "
              << (persistence_checked ? "verified in native disk mode" : "disabled by shader instrumentation; requires separate native-mode check") << '\n';
}

void ValidateQueuedTextureTransfers(Metal::Device& device) {
    auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void prepare_texture(device uint* pixels [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    pixels[i]=0x11223344u;
}
)","Queued texture upload ordering fixture");
    auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"prepare_texture");
    descriptor.maxTotalThreadsPerThreadgroup=64;
    auto pipeline=device.CompileCompute(descriptor);
    for (bool batched : {false,true}) {
        Metal::Scheduler queue{device};
        Metal::GuestTextureCodec codec{device,queue};
        Metal::FormatConversionRuntime conversion{device,queue};
        Metal::TransferRuntime transfer{device,queue};
        queue.Finish();
        auto raw=device.CreateBuffer(32*32*4,false),linear=device.CreateBuffer(32*32*4,false);
        auto downloaded=device.CreateBuffer(16*16*4,false),encoded=device.CreateBuffer(16*16*4,false);
        auto output=device.CreateBuffer(16*16*4+32,true);
        auto texture=[&](unsigned size) {
            auto* d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:size height:size mipmapped:YES];
            d.storageMode=MTLStorageModePrivate;
            d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
            return device.CreateTexture(d);
        };
        auto uploaded=texture(32),scaled=texture(16),copied=texture(16);
        auto cleared=texture(16);
        auto color_output=device.CreateBuffer(16*16*4,true);
        auto* depth_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
            width:16 height:16 mipmapped:NO];
        depth_descriptor.storageMode=MTLStorageModePrivate;
        depth_descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        auto depth=device.CreateTexture(depth_descriptor);
        auto depth_output=device.CreateBuffer(16*16*4,true),stencil_output=device.CreateBuffer(16*16,true);
        const Metal::TextureRegion input{uploaded,0,0,MTLRegionMake2D(0,0,32,32)};
        const Metal::TextureRegion reduced{scaled,0,0,MTLRegionMake2D(0,0,16,16)};
        const Metal::TextureRegion final{copied,0,0,MTLRegionMake2D(0,0,16,16)};
        const auto slice=[](id<MTLBuffer> buffer) {return Metal::BufferSlice{buffer,0,buffer.length,nullptr,{}};};
        const auto tick=queue.CurrentTick();
        Metal::Test::CompilerGate gate{device};
        auto pending=device.DeferCompilation([pipeline] {return pipeline;});
        Metal::ComputeDispatch dispatch{.pending_pipeline=pending,.buffers={{raw,0,0}},
            .threads=MTLSizeMake(32*32,1,1),.group=MTLSizeMake(64,1,1)};
        {
            std::unique_ptr<Metal::Scheduler::ComputeBatchScope> batch;
            if(batched)batch=std::make_unique<Metal::Scheduler::ComputeBatchScope>(queue);
            queue.QueueCompute(std::span{&dispatch,1});
            transfer.ClearColor(cleared,MTLClearColorMake(1,0,0,1));
            transfer.Download({cleared,0,0,MTLRegionMake2D(0,0,16,16)},color_output,{0,16*4,0,MTLBlitOptionNone});
            transfer.ClearDepthStencil(depth,0.5,0x5A,true,true);
            transfer.Download({depth,0,0,MTLRegionMake2D(0,0,16,16)},depth_output,
                {0,16*4,0,MTLBlitOptionDepthFromDepthStencil});
            transfer.Download({depth,0,0,MTLRegionMake2D(0,0,16,16)},stencil_output,
                {0,16,0,MTLBlitOptionStencilFromDepthStencil});
            codec.Decode({32,32,32,VideoCore::PixelFormat::RGBA8,true,true},slice(raw),slice(linear));
            transfer.Upload(linear,{0,32*4,0,MTLBlitOptionNone},input);
            conversion.PicaColorTransfer(input,reduced,0,false,false);
            transfer.CopyTexture(reduced,final);
            transfer.GenerateMipmaps(copied);
            transfer.Download(final,downloaded,{0,16*4,0,MTLBlitOptionNone});
            codec.Encode({16,16,16,VideoCore::PixelFormat::RGBA8,true,true},slice(downloaded),slice(encoded),
                NSMakeRange(0,encoded.length));
            transfer.FillBuffer(output,NSMakeRange(0,output.length),0xDA);
            transfer.CopyBuffer(encoded,0,output,16,encoded.length);
        }
        const bool progressed=queue.CurrentTick()==tick && !queue.WorkerStats().encoded_dispatches;
        const bool timely=gate.Open();
        queue.Finish();
        Check(timely && progressed,"Texture upload/codec/transfer drained a compiler-blocked encoding worker");
        for(unsigned i=0;i<16*16;++i) {
            Check(static_cast<const std::uint32_t*>(color_output.contents)[i]==0xFF0000FFu,
                "Queued color clear lost its load action or transfer ordering");
            Check(static_cast<const float*>(depth_output.contents)[i]==0.5f &&
                static_cast<const std::uint8_t*>(stencil_output.contents)[i]==0x5A,
                "Queued depth/stencil clear lost values or transfer ordering");
        }
        const auto* words=static_cast<const std::uint32_t*>(output.contents);
        for(unsigned i=0;i<output.length/4;++i)
            Check(words[i]==(i<4 || i>=260 ? 0xDADADADAu:0x11223344u),
                  "Queued texture transfers lost codec/copy ordering, resource lifetime or byte guards");
    }
    std::cout<<"Queued texture transfers: compiler-gated owner progress in direct/batched scopes, exact color/depth/stencil clears, native32x32 decode/upload,2x2 PICA downscale,texture copy/mipmaps/download/encode,1024 exact bytes and guards passed\n";
}

void ValidateQueuedCompute(Metal::Device& device) {
    auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void queued_compute(device uint* values [[buffer(0)]], constant uint& salt [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    values[i]=(values[i]*1664525u+1013904223u)^salt;
}
kernel void queued_after(device uint* values [[buffer(0)]], uint i [[thread_position_in_grid]]) {values[i]+=3u;}
)","Immutable queued compute fixture");
    auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
    descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"queued_after");
    descriptor.maxTotalThreadsPerThreadgroup=64;
    auto after=device.CompileCompute(descriptor);
    struct Environment {
        std::array<const char*,4> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_HOST_ONLY",
            "AZAHAR_METAL4_PROFILE_TRIGGER","AZAHAR_METAL4_PROFILE_FRAME_LIMIT"};
        std::array<std::optional<std::string>,4> old;
        Environment(){for(unsigned i=0;i<keys.size();++i){if(const auto* v=std::getenv(keys[i]))old[i]=v;unsetenv(keys[i]);}}
        ~Environment(){for(unsigned i=0;i<keys.size();++i){if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}}
    } environment;
    char directory[]="/private/tmp/azahar-compute-worker-XXXXXX";
    Check(mkdtemp(directory),"Unable to create compute worker profile directory");
    const auto path=std::filesystem::path(directory)/"compute.csv";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","1",1);
    std::shared_future<id<MTLComputePipelineState>> pipeline;
    {
        Metal::Scheduler queue{device};
        auto gpu_gate=[device.Handle() newSharedEvent];
        struct ReleaseGPU{id<MTLSharedEvent> event;~ReleaseGPU(){event.signaledValue=1;}} gpu_release{gpu_gate};
        Metal::SchedulerTestAccess::BlockQueue(queue,gpu_gate);
        std::promise<void> release;
        struct Release {std::promise<void>& promise;bool open{};void Open(){if(!open){promise.set_value();open=true;}}~Release(){Open();}} gate{release};
        const auto wait=release.get_future().share();
        std::array<std::promise<void>,2> started;
        std::array<std::shared_future<void>,2> blockers;
        for(unsigned i=0;i<2;++i)blockers[i]=device.DeferCompilation([&,i,wait]{started[i].set_value();wait.wait();});
        for(auto& start:started)start.get_future().wait();
        pipeline=device.DeferCompilation([native=&device,library] {
            auto* desc=[[MTL4ComputePipelineDescriptor alloc] init];
            desc.computeFunctionDescriptor=Metal::Device::Function(library,"queued_compute");
            desc.maxTotalThreadsPerThreadgroup=64;return native->CompileCompute(desc);
        });
        std::vector<id<MTLBuffer>> outputs;
        for(unsigned job=0;job<320;++job) {
            auto source=device.CreateBuffer(256,true),history=device.CreateBuffer(288,true),target=device.CreateBuffer(288,true);
            auto salt=device.CreateBuffer(4,true);*static_cast<unsigned*>(salt.contents)=job;
            for(unsigned i=0;i<64;++i)static_cast<unsigned*>(source.contents)[i]=job*1009u+i;
            std::memset(target.contents,0xDA,target.length);
            std::array<Metal::ComputeBufferCopy,2> copies{{{source,history,0,16,256},{history,target,16,16,256}}};
            std::array<Metal::ComputeDispatch,2> commands;
            commands[0].pending_pipeline=pipeline;commands[0].buffers={{target,16,0},{salt,0,1}};
            commands[1].pipeline=after;commands[1].buffers={{target,16,0}};
            for(auto& command:commands){command.threads=MTLSizeMake(64,1,1);command.group=MTLSizeMake(64,1,1);}
            queue.QueueCompute(commands,copies);outputs.push_back(target);
            commands[0].buffers.clear();copies[0].source=nil;source=nil;salt=nil;history=nil;
            if(job==7) {
                Check(pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout && queue.CurrentTick()==1 &&
                    queue.WorkerStats().queued_dispatches==16,"Pending compiler blocked or dropped owner compute work");
                gate.Open();
            }
        }
        const auto count=queue.WorkerStats().queued_dispatches;
        bool rejected{};
        Metal::ComputeDispatch invalid;invalid.pipeline=after;invalid.buffers={{outputs[0],outputs[0].length,0}};
        try{queue.QueueCompute(std::span{&invalid,1});}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected && queue.WorkerStats().queued_dispatches==count,"Invalid queued binding changed command state");
        rejected=false;
        std::thread foreign{[&]{try{queue.QueueCompute({});}catch(const std::logic_error&){rejected=true;}}};foreign.join();
        Check(rejected,"Queued compute bypassed owner validation");
        queue.SubmitPending();
        Check(!queue.IsComplete(1),"Queued compute did not retain blocked GPU work");
        gpu_gate.signaledValue=1;queue.Finish();
        for(unsigned job=0;job<outputs.size();++job) {
            const auto* bytes=static_cast<const std::byte*>(outputs[job].contents);
            const auto* values=reinterpret_cast<const std::uint32_t*>(bytes+16);
            for(unsigned i=0;i<64;++i)Check(values[i]==(((job*1009u+i)*1664525u+1013904223u)^job)+3u,
                "Queued compute lost RAW copy/dispatch dependencies, immutable bindings or GPU lifetime");
            for(unsigned i=0;i<16;++i)Check(bytes[i]==std::byte{0xDA} && bytes[272+i]==std::byte{0xDA},
                "Queued compute overwrote range guards");
        }
        const auto stats=queue.WorkerStats();
        Check(stats.queued_dispatches==640 && stats.encoded_dispatches==640 && stats.separate_thread && queue.EncoderCount()==320,
            "Queued compute changed dispatch/encoder count or worker ownership");
        queue.EndProfileFrame();
    }
    std::ifstream profile(path);std::string line;unsigned valid{};
    while(std::getline(profile,line))if(line.find("queued compute")!=std::string::npos) {
        std::vector<std::string> fields;std::istringstream row{line};std::string field;
        while(std::getline(row,field,','))fields.push_back(field);
        if(fields.size()==11 && fields[6]=="1")++valid;
    }
    Check(valid==320,"Queued compute lost native timestamp attribution");
    unsetenv("AZAHAR_METAL4_PROFILE");unsetenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT");
    {
        Metal::Scheduler queue{device};auto data=device.CreateBuffer(256,true),salt=device.CreateBuffer(4,true);
        for(unsigned i=0;i<64;++i)static_cast<unsigned*>(data.contents)[i]=i;
        *static_cast<unsigned*>(salt.contents)=99;
        const std::array<Metal::BufferBinding,1> binding{{{data,0,0}}};
        const auto before=queue.EncoderCount();
        {
            Metal::Scheduler::ComputeBatchScope batch{queue};
            const auto direct=[&] {
                auto encoder=queue.BeginCompute(after);auto arguments=queue.MakeArguments(binding);
                [encoder setComputePipelineState:after];[encoder setArgumentTable:arguments];
                [encoder dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            };
            direct();
            Metal::ComputeDispatch dispatch;dispatch.pending_pipeline=pipeline;dispatch.buffers={{data,0,0},{salt,0,1}};
            dispatch.threads=dispatch.group=MTLSizeMake(64,1,1);
            queue.QueueCompute(std::span{&dispatch,1});direct();
            Check(queue.EncoderCount()==before+1,"Owner/worker compute handoff split the native batch");
        }
        queue.Finish();
        for(unsigned i=0;i<64;++i)Check(static_cast<const unsigned*>(data.contents)[i]==(((i+3u)*1664525u+1013904223u)^99u)+3u,
            "Owner/worker compute handoff reordered operations");
    }
    {
        Metal::Scheduler failed{device};Metal::ComputeDispatch dispatch;
        dispatch.pending_pipeline=device.DeferCompilation([]()->id<MTLComputePipelineState>{throw std::runtime_error("original queued compute failure");});
        failed.QueueCompute(std::span{&dispatch,1});bool original{};
        try{failed.SubmitPending();}catch(const std::runtime_error& e){original=std::string_view(e.what())=="original queued compute failure";}
        Check(original && failed.CurrentTick()==1,"Queued compute hid compiler failure or submitted incomplete work");
    }
    {
        Metal::Scheduler queue{device};
        const auto native=pipeline.get();const auto width=native.threadExecutionWidth;
        auto data=device.CreateBuffer(width*2*4+32,true),salt=device.CreateBuffer(4,true);
        std::memset(data.contents,0xD3,data.length);*static_cast<unsigned*>(salt.contents)=71;
        for(unsigned i=0;i<width*2;++i)static_cast<unsigned*>(data.contents)[4+i]=i;
        Metal::ComputeDispatch dispatch;dispatch.pending_pipeline=pipeline;dispatch.buffers={{data,16,0},{salt,0,1}};dispatch.simd_groups=2;
        queue.QueueCompute(std::span{&dispatch,1});queue.Finish();
        for(unsigned i=0;i<width*2;++i)Check(static_cast<const unsigned*>(data.contents)[4+i]==((i*1664525u+1013904223u)^71u),
            "Deferred native SIMD dispatch used guessed width/count");
        for(unsigned i=0;i<16;++i)Check(static_cast<const std::byte*>(data.contents)[i]==std::byte{0xD3} &&
            static_cast<const std::byte*>(data.contents)[data.length-16+i]==std::byte{0xD3},"Native SIMD dispatch exceeded its exact grid");
        bool rejected{};dispatch.threads.width=2;
        try{queue.QueueCompute(std::span{&dispatch,1});}catch(const std::invalid_argument&){rejected=true;}
        Check(rejected,"Native SIMD dispatch accepted conflicting fixed dimensions");
    }
    std::filesystem::remove_all(directory);
    std::cout<<"640 queued native compute dispatches: compiler-gated owner progress, bounded worker, immutable bindings/copies, RAW/WAR/WAW ordering, exact20480 words/guards, blocked GPU residency/lifetime, owner rejection, one batched owner-worker-owner encoder, timestamps, native SIMD widths/guards and original errors passed\n";
}

void ValidatePersistentCompute(Metal::Device& device) {
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void persistent_step(device uint* words [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    words[i]=words[i]*3u+7u;
}
struct V {float4 p [[position]];};
vertex V persistent_vertex(uint i [[vertex_id]]) {
    return {float4(i==1 ? 3.0f:-1.0f,i==2 ? 3.0f:-1.0f,0,1)};
}
fragment uint persistent_fragment(V in [[stage_in]],const device uint* words [[buffer(0)]]) {
    return words[uint(in.p.x)];
}
)","Worker-owned persistent compute fixture");
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"persistent_step");
    const auto compute=device.CompileCompute(cd);
    Metal::PipelineCache cache{device};
    const auto render=cache.Render({library,"persistent_vertex",library,"persistent_fragment",MTLPixelFormatR32Uint,{}});
    const auto directory=std::filesystem::temp_directory_path()/
        ("azahar-persistent-compute-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto path=directory/"gpu.csv";
    struct Environment {
        std::optional<std::string> old;
        explicit Environment(const std::filesystem::path& path) {
            if(const auto* v=std::getenv("AZAHAR_METAL4_PROFILE")) old=v;
            setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
        }
        ~Environment(){if(old)setenv("AZAHAR_METAL4_PROFILE",old->c_str(),1);else unsetenv("AZAHAR_METAL4_PROFILE");}
    } environment{path};
    {
        Metal::Scheduler queue{device}; Metal::TransferRuntime transfer{device,queue};
        auto words=device.CreateBuffer(34*4,true);
        std::fill_n(static_cast<unsigned*>(words.contents),34,0xA7B8C9D0u);
        for(unsigned i=0;i<32;++i)static_cast<unsigned*>(words.contents)[i+1]=i;
        auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:32 height:1 mipmapped:NO];
        td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
        const auto texture=device.CreateTexture(td);
        auto* descriptor=[[MTL4RenderPassDescriptor alloc] init];
        descriptor.colorAttachments[0].texture=texture;
        descriptor.colorAttachments[0].loadAction=MTLLoadActionClear;
        descriptor.colorAttachments[0].storeAction=MTLStoreActionStore;
        const auto pass=queue.SnapshotRenderPass(descriptor);
        Metal::Test::CompilerGate gate{device};
        const auto pending=device.DeferCompilation([compute]{return compute;});
        Metal::ComputeDispatch step;step.pending_pipeline=pending;step.buffers={{words,4,0}};
        step.threads=MTLSizeMake(32,1,1);step.group=MTLSizeMake(32,1,1);
        {
            Metal::Scheduler::ComputeBatchScope batch{queue};
            {
                Metal::Scheduler::BufferComputeScope disjoint{queue};
                queue.QueueCompute(std::span{&step,1});
            }
            // Restore the general render dependency while the same worker
            // encoder is still pending, then append a RAW/WAR/WAW consumer.
            step.pending_pipeline={};step.pipeline=compute;
            queue.QueueCompute(std::span{&step,1});
        }
        Metal::RenderCommand draw{.pipeline=render,.viewport={0,0,32,1,0,1},.scissor={0,0,32,1},
            .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
        const std::array bindings{Metal::BufferBinding{words,4,0}};
        queue.QueueRender(*pass,draw,bindings);
        const auto held=queue.WorkerStats();
        const bool two_encoders=queue.EncoderCount()==2;
        const bool timely=gate.Open();
        Check(timely && two_encoders && held.queued_dispatches==2 && !held.encoded_dispatches &&
                  held.queued_draws==1 && !held.encoded_draws && queue.CurrentTick()==1,
              "Persistent compute scope exit/render transition waited on the compiler or split/dropped its chain");
        const auto pixels=transfer.Readback({texture,0,0,MTLRegionMake2D(0,0,32,1)},128);
        for(unsigned i=0;i<32;++i) {
            unsigned value{};std::memcpy(&value,pixels.data()+i*4,4);
            Check(value==i*9+28 && static_cast<unsigned*>(words.contents)[i+1]==value,
                  "Persistent compute/render handoff lost a RAW/WAR/WAW dependency");
        }
        Check(static_cast<unsigned*>(words.contents)[0]==0xA7B8C9D0u &&
              static_cast<unsigned*>(words.contents)[33]==0xA7B8C9D0u,"Persistent compute changed buffer guards");
    }
    std::ifstream stream(path);std::string line;unsigned measured{};
    while(std::getline(stream,line)) if(line.find("queued compute")!=std::string::npos) {
        std::vector<std::string> fields;std::istringstream row{line};std::string field;
        while(std::getline(row,field,','))fields.push_back(field);
        Check(fields.size()==11 && fields[6]=="1","Persistent compute lost its complete native timestamp pair");
        ++measured;
    }
    Check(measured==1,"Persistent compute was timestamped once per dispatch instead of once per encoder");
    setenv("AZAHAR_METAL4_PROFILE",(directory/"failed.csv").c_str(),1);
    static_assert(std::is_nothrow_destructible_v<Metal::Scheduler::ComputeBatchScope>);
    static_assert(std::is_nothrow_destructible_v<std::optional<Metal::Scheduler::ComputeBatchScope>>);
    for (const bool optional : {false,true}) for (const bool unwind : {false,true}) {
        Metal::Scheduler failed{device};
        auto words=device.CreateBuffer(4,true);*static_cast<unsigned*>(words.contents)=19;
        std::promise<id<MTLComputePipelineState>> publication;
        publication.set_exception(std::make_exception_ptr(std::runtime_error("original persistent PSO failure")));
        Metal::ComputeDispatch step;step.pipeline=compute;step.buffers={{words,0,0}};
        bool original{},caller{};
        try {
            const auto enqueue=[&] {
                failed.QueueCompute(std::span{&step,1});
                step.pipeline=nil;step.pending_pipeline=publication.get_future().share();
                failed.QueueCompute(std::span{&step,1});
                bool observed{};
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds{2};
                while(!observed && std::chrono::steady_clock::now()<deadline) {
                    try { failed.UseResource(words); }
                    catch (const std::runtime_error& e) {observed=std::string_view{e.what()}=="original persistent PSO failure";}
                    if(!observed)std::this_thread::yield();
                }
                Check(observed,"Failed worker did not publish its original error before scope destruction");
                if (unwind) throw std::runtime_error("original caller unwind");
            };
            if(optional) {
                std::optional<Metal::Scheduler::ComputeBatchScope> batch{std::in_place,failed};
                enqueue();batch.reset();
            } else {
                Metal::Scheduler::ComputeBatchScope batch{failed};enqueue();
            }
            failed.SubmitPending();
        } catch (const std::runtime_error& e) {
            original=std::string_view{e.what()}=="original persistent PSO failure";
            caller=std::string_view{e.what()}=="original caller unwind";
        }
        if (unwind) {
            Check(caller,"Worker failure replaced an unwinding caller exception");
            try { failed.SubmitPending(); }
            catch (const std::runtime_error& e) {original=std::string_view{e.what()}=="original persistent PSO failure";}
        }
        Check(original && failed.CurrentTick()==1 && *static_cast<unsigned*>(words.contents)==19,
              "Persistent batch hid its original failure or submitted partially encoded GPU work");
    }
    std::filesystem::remove_all(directory);
    std::cout<<"Persistent worker compute: compiler-gated scope exit and render handoff, one compute encoder/timestamp pair, nested buffer/general dependency restoration, exact32 R32Uint pixels/words/guards, original sticky PSO error, no partial submit and caller unwind in direct/optional scopes passed\n";
}

void ValidateQueuedBufferBoundaries(Metal::Device& device) {
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void boundary_step(device uint* words [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    words[i]=words[i]*3u+7u;
}
struct V {float4 p [[position]];};
vertex V boundary_vertex(uint i [[vertex_id]]) {
    return {float4(i==1 ? 3.0f:-1.0f,i==2 ? 3.0f:-1.0f,0,1)};
}
fragment uint boundary_fragment(V in [[stage_in]],const device uint* words [[buffer(0)]]) {
    return words[uint(in.p.x)];
}
)","Queued buffer scope boundary fixture");
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"boundary_step");
    const auto compute=device.CompileCompute(cd);
    Metal::PipelineCache cache{device};
    const auto render=cache.Render({library,"boundary_vertex",library,"boundary_fragment",MTLPixelFormatR32Uint,{}});
    constexpr unsigned count=64,guard=0x51a7e3d9;
    for(unsigned mode=0;mode<3;++mode) for(bool unwind:{false,true}) {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        auto words=device.CreateBuffer((count+2)*4,true),source=device.CreateBuffer(count*4,true);
        std::fill_n(static_cast<unsigned*>(words.contents),count+2,guard);
        for(unsigned i=0;i<count;++i) {
            static_cast<unsigned*>(words.contents)[i+1]=i;
            static_cast<unsigned*>(source.contents)[i]=i^0xa7b8c9d0u;
        }
        const auto texture=[&] {
            auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                width:count height:1 mipmapped:NO];
            td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
            return device.CreateTexture(td);
        };
        auto first=texture(),last=texture();
        const auto pass=[&](id<MTLTexture> target,bool reusable) {
            auto* descriptor=[[MTL4RenderPassDescriptor alloc] init];
            descriptor.colorAttachments[0].texture=target;
            descriptor.colorAttachments[0].loadAction=MTLLoadActionClear;
            descriptor.colorAttachments[0].storeAction=MTLStoreActionStore;
            if(reusable) {
                descriptor.colorAttachments[0].loadAction=MTLLoadActionLoad;
                // Explicit immutable pass compatibility is already validated by
                // the scheduler; prime its clear before this gated workload.
                transfer.ClearColor(target,MTLClearColorMake(0,0,0,0));
            }
            return queue.SnapshotRenderPass(descriptor);
        };
        const auto initial=pass(first,mode==2),final=pass(last,false);
        queue.Finish();
        const auto tick=queue.CurrentTick();
        Metal::Test::CompilerGate gate{device};
        Metal::ComputeDispatch step;step.pipeline=compute;step.buffers={{words,4,0}};
        step.threads=MTLSizeMake(count,1,1);step.group=MTLSizeMake(count,1,1);
        Metal::RenderCommand draw{.pipeline=render,.viewport={0,0,count,1,0,1},.scissor={0,0,count,1},
            .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
        const auto readonly=Metal::SchedulerTestAccess::ReadOnlyBuffers();
        {
            Metal::Scheduler::ComputeBatchScope batch{queue};
            if(!mode) {
                auto pending=step;pending.pipeline=nil;pending.pending_pipeline=device.DeferCompilation([compute]{return compute;});
                queue.QueueCompute(std::span{&pending,1});
            } else {
                auto pending=draw;pending.pipeline=nil;pending.pending_pipeline=device.DeferCompilation([render]{return render;});
                queue.QueueRender(*initial,pending,std::array{Metal::BufferBinding{source,0,0}}, {}, {}, mode==2, {}, &readonly);
            }
            Check(Metal::SchedulerTestAccess::QueuedComputeOpen(queue)==(mode==0) &&
                  Metal::SchedulerTestAccess::QueuedRenderOpen(queue)==(mode==2),
                  "Queued buffer boundary fixture did not exercise its intended encoder kind");
            bool caller{};
            try {
                Metal::Scheduler::BufferComputeScope disjoint{queue};
                Check(!Metal::SchedulerTestAccess::QueuedComputeOpen(queue) &&
                      !Metal::SchedulerTestAccess::QueuedRenderOpen(queue),
                      "Queued buffer boundary did not close the preceding native encoder in queue order");
                // These writes are disjoint from every preceding render input
                // and attachment, while the compute mode retains RAW/WAR/WAW.
                queue.QueueCompute(std::span{&step,1});
                {
                    Metal::Scheduler::BufferComputeScope nested{queue};
                    queue.QueueCompute(std::span{&step,1});
                    if(unwind)throw std::runtime_error("queued buffer caller unwind");
                }
            }catch(const std::runtime_error& e){caller=std::string_view{e.what()}=="queued buffer caller unwind";}
            Check(caller==unwind,"Queued buffer scope changed the caller exception");
            // Restored general mode must still join the preceding dependencies.
            queue.QueueCompute(std::span{&step,1});
        }
        queue.QueueRender(*final,draw,std::array{Metal::BufferBinding{words,4,0}});
        const auto held=queue.WorkerStats();
        const bool timely=gate.Open();
        Check(timely && queue.CurrentTick()==tick && !held.encoded_dispatches && !held.encoded_draws &&
              held.queued_dispatches==(!mode ? 4u:3u) && held.queued_draws==(!mode ? 1u:2u),
              "Queued-only buffer scope waited on the compiler, submitted, or lost a command");
        const auto pixels=transfer.Readback({last,0,0,MTLRegionMake2D(0,0,count,1)},count*4);
        for(unsigned i=0;i<count;++i) {
            unsigned expected=i;for(unsigned n=0;n<(!mode ? 4u:3u);++n)expected=expected*3+7;
            unsigned value{};std::memcpy(&value,pixels.data()+i*4,4);
            Check(value==expected && static_cast<unsigned*>(words.contents)[i+1]==expected,
                  "Queued buffer scope lost ordered compute state or final raster words");
        }
        Check(static_cast<unsigned*>(words.contents)[0]==guard && static_cast<unsigned*>(words.contents)[count+1]==guard,
              "Queued buffer scope changed a disjoint range guard");
        if(mode) {
            const auto before=transfer.Readback({first,0,0,MTLRegionMake2D(0,0,count,1)},count*4);
            Check(!std::memcmp(before.data(),source.contents,count*4),"Queued buffer scope changed prior render input pixels");
        }
    }
    {
        Metal::Scheduler queue{device};
        auto words=device.CreateBuffer(count*4,true);
        for(unsigned i=0;i<count;++i)static_cast<unsigned*>(words.contents)[i]=i;
        Metal::Test::CompilerGate gate{device};
        Metal::ComputeDispatch step;step.pending_pipeline=device.DeferCompilation([compute]{return compute;});
        step.buffers={{words,0,0}};step.threads=MTLSizeMake(count,1,1);step.group=MTLSizeMake(count,1,1);
        std::atomic_bool opened{};std::jthread release;
        {
            Metal::Scheduler::ComputeBatchScope batch{queue};
            (void)queue.BeginCompute(compute);
            queue.QueueCompute(std::span{&step,1},{},nil,true);
            release=std::jthread{[&] {
                std::this_thread::sleep_for(std::chrono::milliseconds{40});
                opened.store(true);gate.Open();
            }};
            Metal::Scheduler::BufferComputeScope disjoint{queue};
            Check(opened.load() && queue.WorkerStats().encoded_dispatches==1,
                  "Buffer scope failed to drain borrowed owner encoding before closing its native encoder");
            step.pending_pipeline={};step.pipeline=compute;
            queue.QueueCompute(std::span{&step,1});
        }
        release.join();queue.Finish();
        for(unsigned i=0;i<count;++i)
            Check(static_cast<unsigned*>(words.contents)[i]==i*9+28,
                  "Borrowed owner/buffer boundary lost its dependent native writes");
    }
    {
        Metal::Scheduler queue{device};auto words=device.CreateBuffer(4,true);
        *static_cast<unsigned*>(words.contents)=19;
        std::promise<id<MTLComputePipelineState>> failure;
        failure.set_exception(std::make_exception_ptr(std::runtime_error("queued buffer boundary PSO failure")));
        Metal::ComputeDispatch step;step.pending_pipeline=failure.get_future().share();step.buffers={{words,0,0}};
        queue.QueueCompute(std::span{&step,1});
        bool observed{};const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds{2};
        while(!observed && std::chrono::steady_clock::now()<deadline) {
            try {queue.UseResource(words);}
            catch(const std::runtime_error& e){observed=std::string_view{e.what()}=="queued buffer boundary PSO failure";}
            if(!observed)std::this_thread::yield();
        }
        Check(observed,"Queued buffer boundary fixture failed to observe the original worker error");
        bool original{},sticky{};
        try {Metal::Scheduler::BufferComputeScope disjoint{queue};}
        catch(const std::runtime_error& e){original=std::string_view{e.what()}=="queued buffer boundary PSO failure";}
        try {queue.SubmitPending();}
        catch(const std::runtime_error& e){sticky=std::string_view{e.what()}=="queued buffer boundary PSO failure";}
        Check(original && sticky && queue.CurrentTick()==1 && *static_cast<unsigned*>(words.contents)==19,
              "Queued buffer boundary hid the original failure or submitted partial native commands");
    }
    std::cout<<"Queued buffer boundaries:6 compiler-gated compute/closed-render/open-render transitions, nested normal/unwind scopes, no CPU drain/submit, exact384 final raster/state words plus256 earlier pixels/guards; borrowed owner drain64words and sticky original failure/no partial submit passed\n";
}

void ValidateSubmissionObservers(Metal::Device& device) {
    using Timing=Metal::Scheduler::SubmissionTiming;
    auto first=std::make_shared<std::promise<Timing>>(),second=std::make_shared<std::promise<Timing>>();
    auto first_result=first->get_future(),second_result=second->get_future();
    auto bytes=device.CreateBuffer(64,true);
    std::uint64_t first_tick{},second_tick{};
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        auto event=[device.Handle() newSharedEvent];
        struct Release {id<MTLSharedEvent> event;~Release(){event.signaledValue=1;}} release{event};
        queue.SetSubmissionObserver([first](const Timing& value){first->set_value(value);});
        bool rejected{};
        std::thread foreign{[&] {
            try{queue.SetSubmissionObserver({});}catch(const std::logic_error&){rejected=true;}
        }};foreign.join();Check(rejected,"Submission observer setter bypassed owner confinement");
        Metal::SchedulerTestAccess::BlockQueue(queue,event);
        transfer.FillBuffer(bytes,NSMakeRange(0,64),0x21);first_tick=queue.Submit();
        queue.SetSubmissionObserver([second](const Timing& value){second->set_value(value);});
        transfer.FillBuffer(bytes,NSMakeRange(0,64),0x57);second_tick=queue.Submit();
        queue.SetSubmissionObserver({});first.reset();second.reset();
        Check(first_result.wait_for(std::chrono::seconds{0})==std::future_status::timeout &&
              second_result.wait_for(std::chrono::seconds{0})==std::future_status::timeout,
              "Submission timing ran before its externally blocked GPU work completed");
        event.signaledValue=1;queue.Finish();
    }
    Check(first_result.wait_for(std::chrono::seconds{2})==std::future_status::ready &&
          second_result.wait_for(std::chrono::seconds{2})==std::future_status::ready,
          "Submission observer lost a callback after setter/source/scheduler teardown");
    const std::array values{first_result.get(),second_result.get()};
    for(unsigned i=0;i<values.size();++i) {
        const auto& value=values[i];
        Check(value.tick==(i ? second_tick:first_tick) && !value.failed &&
              std::isfinite(value.gpu_begin_seconds) && std::isfinite(value.gpu_end_seconds) &&
              value.gpu_begin_seconds>0 && value.gpu_end_seconds>=value.gpu_begin_seconds,
              "Native commit observer changed tick identity or GPU timestamp units/order");
    }
    for(unsigned i=0;i<64;++i)Check(static_cast<const unsigned char*>(bytes.contents)[i]==0x57,
                                  "Submission observer changed ordered native GPU bytes");
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        queue.SetSubmissionObserver([](const Timing&){throw std::runtime_error("expected diagnostic callback failure");});
        transfer.FillBuffer(bytes,NSMakeRange(0,64),0x73);queue.SubmitPending();
        try{queue.Finish();}catch(const std::runtime_error&){}
        bool original{};const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds{2};
        while(!original && std::chrono::steady_clock::now()<deadline) {
            try{queue.UseResource(bytes);}
            catch(const std::runtime_error& e){original=std::string_view{e.what()}=="Metal 4 submission observer: expected diagnostic callback failure";}
            if(!original)std::this_thread::yield();
        }
        Check(original && queue.CurrentTick()==2,"Submission observer exception escaped Metal or lost its sticky failure");
        for(unsigned i=0;i<64;++i)Check(static_cast<const unsigned char*>(bytes.contents)[i]==0x73,
                                      "Failing diagnostic observer changed completed GPU writes");
    }
    std::cout<<"Native commit observer:2 externally gated immutable callback snapshots, released source/setter/scheduler lifetime, exact64GPUbytes, tick/timestamp order, foreign-owner rejection and contained sticky callback exception passed\n";
}

void ValidateDrainAttribution(Metal::Device& device) {
    struct Environment {
        std::array<const char*,6> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_HOST_ONLY",
            "AZAHAR_METAL4_PROFILE_HOST_PHASES","AZAHAR_METAL4_PROFILE_DRAINS",
            "AZAHAR_METAL4_PROFILE_FRAME_LIMIT","AZAHAR_METAL4_PROFILE_TRIGGER"};
        std::array<std::optional<std::string>,6> old;
        Environment(){for(unsigned i=0;i<keys.size();++i){if(const auto* v=std::getenv(keys[i]))old[i]=v;unsetenv(keys[i]);}}
        ~Environment(){for(unsigned i=0;i<keys.size();++i){if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);}}
    } environment;
    char directory[]="/private/tmp/azahar-drain-attribution-XXXXXX";
    Check(mkdtemp(directory),"Unable to create worker-drain profile directory");
    const auto path=std::filesystem::path{directory}/"gpu.csv";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_HOST_ONLY","1",1);
    setenv("AZAHAR_METAL4_PROFILE_HOST_PHASES","1",1);
    setenv("AZAHAR_METAL4_PROFILE_DRAINS","1",1);
    setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","2",1);
    const auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void drain_step(device uint* words [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    words[i]=(words[i]^0x51a7e3d9u)*1664525u+1013904223u;
}
)","Native drain source/timing fixture");
    auto* cd=[[MTL4ComputePipelineDescriptor alloc] init];
    cd.computeFunctionDescriptor=Metal::Device::Function(library,"drain_step");
    const auto compute=device.CompileCompute(cd);
    const auto source=std::source_location::current();
    auto words=device.CreateBuffer(64*4,true);
    for(unsigned i=0;i<64;++i)static_cast<unsigned*>(words.contents)[i]=i;
    {
        Metal::Scheduler queue{device};
        for(unsigned frame=0;frame<3;++frame) {
            Metal::Test::CompilerGate gate{device};
            Metal::ComputeDispatch command;command.pending_pipeline=device.DeferCompilation([compute]{return compute;});
            command.buffers={{words,0,0}};command.threads=MTLSizeMake(64,1,1);command.group=MTLSizeMake(64,1,1);
            queue.QueueCompute(std::span{&command,1});
            std::promise<void> requested;
            std::jthread release{[&] {
                requested.get_future().wait();std::this_thread::sleep_for(std::chrono::milliseconds{40});gate.Open();
            }};
            {
                Metal::Scheduler::HostPhaseScope root{queue,frame%2 ? Metal::Scheduler::HostPhase::Present:Metal::Scheduler::HostPhase::Draw};
                // The nested phase must not replace the exclusive root label.
                Metal::Scheduler::HostPhaseScope nested{queue,Metal::Scheduler::HostPhase::Transfer};
                requested.set_value();
                if(frame%2)(void)queue.BeginCompute(compute,source);
                else queue.EndEncoding(source);
            }
            release.join();queue.EndProfileFrame();
        }
        queue.Finish();
    }
    for(unsigned i=0;i<64;++i) {
        unsigned expected=i;for(unsigned n=0;n<3;++n)expected=(expected^0x51a7e3d9u)*1664525u+1013904223u;
        Check(static_cast<unsigned*>(words.contents)[i]==expected,"Drain attribution changed completed native GPU words");
    }
    unsigned matched{};std::set<std::string> frames;
    std::ifstream input(path.string()+".drains.csv");std::string line;std::getline(input,line);
    Check(line=="frame,file,line,phase,calls,failures,host_ms,thread_cpu_ms,cpu_valid_calls,max_host_ms,max_begin_host_ns,max_end_host_ns,pending_render_psos_at_max,pending_compute_psos_at_max,dropped_sites",
          "Worker drain profile schema changed");
    while(std::getline(input,line)) {
        std::vector<std::string> fields;std::string field;std::istringstream row{line};
        while(std::getline(row,field,','))fields.push_back(field);
        Check(fields.size()==15 && fields[14]=="0","Worker-drain profile truncated a row or dropped source sites");
        if(fields[1]!="mtl_scheduler_tests.mm" || fields[2]!=std::to_string(source.line()))continue;
        Check(fields[4]=="1" && fields[5]=="0" && fields[8]=="1" && fields[13]=="1" &&
              std::stod(fields[9])>=25 && std::stod(fields[7])<std::stod(fields[6])*.5 &&
              std::stoll(fields[11])>std::stoll(fields[10]),
              "Drain source/timing confused blocked host time with owner CPU or lost its pending PSO");
        Check(fields[3]==(matched ? "present":"draw"),"Nested drain attribution replaced its exclusive root phase");
        frames.insert(fields[0]);++matched;
    }
    Check(matched==2 && frames.size()==2,"Drain attribution ignored its exact two-frame window");
    for(const char* mode : {"0",""}) {
        const auto off=std::filesystem::path{directory}/(std::string{mode}+"disabled.csv");
        setenv("AZAHAR_METAL4_PROFILE",off.c_str(),1);
        if(*mode)setenv("AZAHAR_METAL4_PROFILE_DRAINS",mode,1);else unsetenv("AZAHAR_METAL4_PROFILE_DRAINS");
        {Metal::Scheduler queue{device};Metal::ComputeDispatch command;command.pipeline=compute;
            command.buffers={{words,0,0}};queue.QueueCompute(std::span{&command,1});queue.EndProfileFrame();queue.Finish();}
        Check(!std::filesystem::exists(off.string()+".drains.csv"),"Disabled worker-drain mode created an output");
    }
    setenv("AZAHAR_METAL4_PROFILE_DRAINS","2",1);
    bool rejected{};try{Metal::Scheduler invalid{device};}catch(const std::invalid_argument&){rejected=true;}
    Check(rejected,"Worker-drain profile accepted an invalid mode");
    std::filesystem::remove_all(directory);
    std::cout<<"Native drain attribution: compiler-held waits at EndEncoding/BeginCompute, exact source/line and draw/present roots, separate wall/thread-CPU intervals, pending PSO,2-frame limit,64 GPU words,disabled/no-output and invalid mode passed\n";
}

void ValidateDeferredPipelines(Metal::Device& device) {
    auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 deferred_vertex(uint i [[vertex_id]]) {
    return float4(i==1 ? 3.0f:-1.0f,i==2 ? 3.0f:-1.0f,0,1);
}
fragment float4 deferred_fragment() {return float4(1,0,1,1);}
)","Deferred native PSO publication fixture");
    Metal::PipelineCache cache{device};
    Metal::Scheduler queue{device};
    Metal::TransferRuntime transfer{device,queue};
    auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
    td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
    auto target=device.CreateTexture(td);
    auto* pass=[[MTL4RenderPassDescriptor alloc] init];
    pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;
    pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    const auto snapshot=queue.SnapshotRenderPass(pass);
    std::promise<void> release;
    struct Release {
        std::promise<void>& promise;bool open{};
        void Open(){if(!open){promise.set_value();open=true;}}
        ~Release(){Open();}
    } gate{release};
    const auto wait=release.get_future().share();
    std::array<std::shared_future<void>,2> blockers;
    std::array<std::promise<void>,2> started;
    for(unsigned i=0;i<2;++i) blockers[i]=device.DeferCompilation([&,i,wait]{started[i].set_value();wait.wait();});
    for(auto& start:started)start.get_future().wait();
    Metal::RenderPipelineDescription description{library,"deferred_vertex",library,"deferred_fragment",MTLPixelFormatRGBA8Unorm,{}};
    auto pipeline=cache.RenderAsync(description);
    std::array<std::future<std::shared_future<id<MTLRenderPipelineState>>>,8> readers;
    for(auto& reader:readers)reader=std::async(std::launch::async,[&]{return cache.RenderAsync(description);});
    for(auto& reader:readers) Check(reader.get().wait_for(std::chrono::seconds{0})==std::future_status::timeout,
        "Deferred PSO lookup waited behind a compiler job or published a partial result");
    auto depth=cache.DepthStencil({});
    Check(depth && pipeline.wait_for(std::chrono::seconds{0})==std::future_status::timeout,
        "Depth state creation waited for native pipeline compilation");
    Metal::RenderCommand draw{.pipeline=nil,.depth=depth,.viewport={0,0,8,8,0,1},.scissor={0,0,8,8},
        .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3,.pending_pipeline=pipeline};
    queue.QueueRender(*snapshot,draw,{});
    description.vertex_entry="caller mutated";description.fragment_library=nil;library=nil;
    Check(queue.WorkerStats().queued_draws==1 && queue.CurrentTick()==1,
        "Pending native pipeline skipped/submitted its draw");
    gate.Open();
    cache.Clear(); // Ready state must survive cache eviction and released library/description inputs.
    Check(cache.Stats().render_builds==1,"Concurrent native pipeline requests duplicated compilation");
    pipeline={};draw.pending_pipeline={};
    const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,8,8)},32);
    for(unsigned i=0;i<pixels.size();++i)
        Check(pixels[i]==std::byte(i%4==1 ? 0:255),"Deferred native PSO lost residency, draw or immutable input lifetime");
    Check(queue.WorkerStats().encoded_draws==1,"Deferred render was not encoded exactly once");
    {
        Metal::Scheduler failed{device};
        auto failure=device.DeferCompilation([]()->id<MTLRenderPipelineState> {
            throw std::runtime_error("original deferred PSO failure");
        });
        draw.pending_pipeline=failure;
        failed.QueueRender(*snapshot,draw,{});
        bool original{};
        try {failed.SubmitPending();}catch(const std::runtime_error& e){original=std::string_view(e.what())=="original deferred PSO failure";}
        Check(original && failed.CurrentTick()==1,"Deferred pipeline error was hidden or submitted incomplete GPU work");
    }
    std::cout<<"Deferred native PSO: compiler-gated owner progress, eight concurrent lookups, one factory, immutable library/description lifetime, cache eviction, residency, exact64 GPU pixels and original failure propagation passed\n";
}

void ValidateRenderBatch(Metal::Device& device) {
    const auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 batch_vertex(uint i [[vertex_id]]) {
    return float4(i==1 ? 3.0f:-1.0f,i==2 ? 3.0f:-1.0f,0,1);
}
fragment float4 batch_fragment() { return float4(1,0,1,1); }
)", "Complete render-batch fixture");
    Metal::PipelineCache cache{device};
    const auto pipeline = cache.Render({library,"batch_vertex",library,"batch_fragment",MTLPixelFormatRGBA8Unorm,{}});
    Metal::Scheduler queue{device};
    Metal::TransferRuntime transfer{device,queue};
    auto* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
    td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
    const auto target = device.CreateTexture(td);
    auto* descriptor = [[MTL4RenderPassDescriptor alloc] init];
    descriptor.colorAttachments[0].texture = target;
    descriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
    descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
    descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0,1,0,1);
    const auto pass = queue.SnapshotRenderPass(descriptor);
    descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0,0,1,1);
    std::array<Metal::RenderBatchDraw,2> draws{};
    draws[0].command = {.pipeline=pipeline,.viewport={0,0,8,8,0,1},.scissor={0,0,4,8},
        .winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
    draws[1].command = draws[0].command; draws[1].command.scissor.x = 4;
    const auto before = queue.WorkerStats();
    const auto encoders = queue.EncoderCount();
    draws[1].buffers.push_back({nil,0,0});
    bool rejected{};
    try { queue.QueueRenderBatch(*pass,draws); } catch (const std::invalid_argument&) { rejected=true; }
    Check(rejected && queue.WorkerStats().queued_draws == before.queued_draws &&
              queue.EncoderCount() == encoders && queue.CurrentTick() == 1,
          "Late invalid batch item partially queued a clear/draw");
    draws[1].buffers.clear();
    queue.QueueRenderBatch(*pass,draws);
    Check(queue.EncoderCount() == encoders + 1,"Complete batch opened more than one render pass");
    const auto first = transfer.Readback({target,0,0,MTLRegionMake2D(0,0,8,8)},32);
    for (unsigned i = 0; i < first.size(); ++i)
        Check(first[i] == std::byte(i%4 == 1 ? 0 : 255),"Batch clear discarded an earlier draw");
    queue.QueueRenderBatch(*pass,std::span{draws.data(),1});
    const auto second = transfer.Readback({target,0,0,MTLRegionMake2D(0,0,8,8)},32);
    for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x) for (unsigned c = 0; c < 4; ++c) {
        const unsigned expected = c == 3 || (x < 4 ? c != 1 : c == 1) ? 255 : 0;
        Check(second[(y*8+x)*4+c] == std::byte(expected),"New batch missed its clear or used a mutated pass descriptor");
    }
    std::cout << "Complete render batch: late-invalid preflight leaves GPU state untouched, one pass/clear, both scissors, fresh same-key clear and immutable descriptor exact128 pixels passed\n";
}

void ValidateDeferredLibraries(Metal::Device& device) {
    constexpr std::string_view source = R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 library_vertex(uint i [[vertex_id]]) {
    return float4(i==1 ? 3.0f:-1.0f,i==2 ? 3.0f:-1.0f,0,1);
}
fragment float4 library_fragment() { return float4(0,1,0,1); }
)";
    Metal::PipelineCache cache{device};
    Metal::Test::CompilerGate gate{device};
    Metal::NativeLibraryFuture first = device.DeferCompilation([&device, source] {
        return device.CompileLibrary(source, "Deferred library identity");
    });
    auto copy = first;
    Check(copy.Identity() == first.Identity(), "Copied native library future changed identity");
    Metal::RenderPipelineDescription description{nil,"library_vertex",nil,"library_fragment",
        MTLPixelFormatRGBA8Unorm,{},first,copy};
    auto pending = cache.RenderAsync(description);
    std::array<std::future<std::shared_future<id<MTLRenderPipelineState>>>,8> readers;
    for (auto& reader : readers) reader = std::async(std::launch::async, [&] { return cache.RenderAsync(description); });
    for (auto& reader : readers) Check(reader.get().wait_for(std::chrono::seconds{0}) == std::future_status::timeout,
        "Library-dependent pipeline lookup waited on compiler publication");
    bool rejected{};
    try { (void)cache.Render(description); } catch (const std::invalid_argument&) { rejected=true; }
    Check(rejected, "Synchronous native factory silently accepted unresolved libraries");
    const bool timely = gate.Open();
    Check(timely, "Library-dependent pipeline lookup blocked its caller");
    const auto expected = pending.get();
    const auto library = first.get();
    Check(expected && cache.Stats().render_builds == 1, "Library-dependent concurrent requests duplicated a native PSO");
    auto independent = Metal::NativeLibraryFuture{device.DeferCompilation([library] { return library; })};
    Check(independent.Identity() != first.Identity(), "Independent future states shared a cache identity");
    description.pending_vertex_library = independent;
    description.pending_fragment_library = independent;
    Check(cache.RenderAsync(description).get() == expected && cache.Stats().render_builds == 1,
        "Independent library futures bypassed native-library PSO deduplication");
    description.vertex_library = library;
    rejected=false;
    try { (void)cache.RenderAsync(description); } catch (const std::invalid_argument&) { rejected=true; }
    Check(rejected, "Ambiguous ready/pending vertex library was accepted");
    description.vertex_library = nil;
    for (unsigned failure=0; failure<3; ++failure) {
        auto failed = device.DeferCompilation([failure,library]() -> id<MTLLibrary> {
            if (failure == 0) throw std::runtime_error("original library factory failure");
            return failure == 1 ? nil : library;
        });
        description.pending_fragment_library = failed;
        description.fragment_entry = failure == 2 ? "missing_fragment_entry" : "library_fragment";
        auto result = cache.RenderAsync(description);
        rejected=false;
        try { (void)result.get(); }
        catch (const std::exception& e) {
            rejected = failure != 0 || std::string_view{e.what()} == "original library factory failure";
        }
        Check(rejected, "Deferred library failure, nil result or missing entry was hidden");
        // Retrying a failed cache entry preserves the original failed future,
        // while a new library completion can succeed without stale reuse.
        if (failure == 0) {
            rejected=false;
            try { (void)cache.RenderAsync(description).get(); }
            catch (const std::runtime_error& e) { rejected=std::string_view{e.what()} == "original library factory failure"; }
            Check(rejected, "Library retry lost its original failure");
        }
        description.pending_fragment_library = device.DeferCompilation([library] { return library; });
        description.fragment_entry = "library_fragment";
        Check(cache.RenderAsync(description).get() == expected, "Fresh library completion reused a failed pending cache state");
    }
    description.pending_vertex_library = {}; description.pending_fragment_library = {};
    description.vertex_library = library; description.fragment_library = library;
    Check(cache.Render(description) == expected, "Ready native cache diverged from resolved library futures");
    std::cout << "Deferred library cache: compiler gate,8 concurrent readers, immutable copied/separate identities, native dedup, ambiguous/synchronous rejection, original error/retry,nil and missing entry passed\n";
}

void ValidateQueuedRendering(Metal::Device& device) {
    struct Environment {
        std::array<const char*,4> keys{"AZAHAR_METAL4_PROFILE","AZAHAR_METAL4_PROFILE_HOST_ONLY",
            "AZAHAR_METAL4_PROFILE_TRIGGER","AZAHAR_METAL4_PROFILE_FRAME_LIMIT"};
        std::array<std::optional<std::string>,4> old;
        Environment() {for(unsigned i=0;i<keys.size();++i) {
            if(const auto* value=std::getenv(keys[i]))old[i]=value;unsetenv(keys[i]);
        }}
        ~Environment() {for(unsigned i=0;i<keys.size();++i) {
            if(old[i])setenv(keys[i],old[i]->c_str(),1);else unsetenv(keys[i]);
        }}
    } environment;
    char directory[]="/private/tmp/azahar-encoding-worker-XXXXXX";
    Check(mkdtemp(directory),"Unable to create native encoding-worker test directory");
    const auto path=std::filesystem::path(directory)/"worker.csv";
    setenv("AZAHAR_METAL4_PROFILE",path.c_str(),1);
    setenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT","1",1);
    auto library=device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 queued_vertex(uint index [[vertex_id]]) {
    return float4(index==1 ? 3.0f:-1.0f,index==2 ? 3.0f:-1.0f,0,1);
}
fragment float4 queued_fragment(constant uint& value [[buffer(0)]]) {
    return float4(float(value&255u)/255.0f,float((value>>8u)&255u)/255.0f,0,1);
}
)","Native queued draw snapshot fixture");
    auto* pipeline_desc=[[MTL4RenderPipelineDescriptor alloc] init];
    pipeline_desc.vertexFunctionDescriptor=Metal::Device::Function(library,"queued_vertex");
    pipeline_desc.fragmentFunctionDescriptor=Metal::Device::Function(library,"queued_fragment");
    pipeline_desc.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
    auto pipeline=device.CompileRender(pipeline_desc);
    {
        Metal::Scheduler queue{device};Metal::TransferRuntime transfer{device,queue};
        auto* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:32 height:8 mipmapped:NO];
        td.storageMode=MTLStorageModePrivate;td.usage=MTLTextureUsageRenderTarget;
        auto target=device.CreateTexture(td),untouched=device.CreateTexture(td);
        transfer.ClearColor(target,MTLClearColorMake(0,0,0,1));
        transfer.ClearColor(untouched,MTLClearColorMake(0,0,1,1));
        auto* pass=[[MTL4RenderPassDescriptor alloc] init];
        pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        const auto snapshot=queue.SnapshotRenderPass(pass);
        pass.colorAttachments[0].texture=untouched; // Must never affect the queued descriptor.
        const auto before=queue.EncoderCount(),tick=queue.CurrentTick();
        auto gate=[device.Handle() newSharedEvent];
        struct Release {id<MTLSharedEvent> event;~Release(){event.signaledValue=1;}} release{gate};
        Metal::SchedulerTestAccess::BlockQueue(queue,gate);
        for(unsigned draw=0;draw<320;++draw) {
            auto buffer=device.CreateBuffer(4,true);*static_cast<unsigned*>(buffer.contents)=draw;
            std::array bindings{Metal::BufferBinding{buffer,0,0}};
            Metal::RenderCommand command{.pipeline=pipeline,.depth=nil,.viewport={0,0,32,8,0,1},
                .scissor={2*(draw%16),0,2,8},.winding=MTLWindingCounterClockwise,.cull=MTLCullModeNone,.vertices=3};
            queue.QueueRender(*snapshot,command,bindings);
            bindings[0]={nil,0,30};command.scissor={0,0,0,0};buffer=nil;
        }
        Check(queue.CurrentTick()==tick && queue.EncoderCount()==before+1,
            "Bounded worker introduced submission/waits or split compatible render passes");
        bool rejected{};
        std::thread foreign{[&] {try {
            queue.QueueRender(*snapshot,{},{});
        } catch(const std::logic_error&) {rejected=true;}}};foreign.join();
        Check(rejected,"Queued encoding bypassed owner-thread validation");
        queue.SubmitPending();
        Check(!queue.IsComplete(tick),"Worker fixture did not retain the blocked GPU timeline");
        gate.signaledValue=1;
        const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,32,8)},128);
        for(unsigned y=0;y<8;++y)for(unsigned x=0;x<32;++x) {
            const unsigned value=304+x/2,offset=(y*32+x)*4;
            Check(pixels[offset]==std::byte(value&255) && pixels[offset+1]==std::byte(value>>8) &&
                pixels[offset+2]==std::byte{0} && pixels[offset+3]==std::byte{255},
                "Queued worker lost immutable bindings/scissor, released buffers or draw order");
        }
        const auto other=transfer.Readback({untouched,0,0,MTLRegionMake2D(0,0,32,8)},128);
        for(unsigned i=0;i<other.size();++i)Check(other[i]==std::byte(i%4>=2 ? 255:0),
            "Queued render used a subsequently mutated pass descriptor");
        const auto stats=queue.WorkerStats();
        Check(stats.queued_draws==320 && stats.encoded_draws==320 && stats.separate_thread,
            "Native driver encoding did not execute all draws on a separate worker");
        Check(stats.profiled_jobs>=322 && stats.profiled_cpu_nanoseconds>0,
            "Queued worker CPU diagnostics omitted real native encoding work");
        std::cout<<"Encoding-worker diagnostic: draws=320 jobs="<<stats.profiled_jobs<<
            " worker_cpu_ns="<<stats.profiled_cpu_nanoseconds<<" (synthetic correctness workload, no game-FPS claim)\n";
        queue.EndProfileFrame();
        queue.ReleaseThread();
        std::exception_ptr failure;
        std::thread next_owner{[&] {try {queue.Finish();queue.ReleaseThread();}catch(...){failure=std::current_exception();}}};
        next_owner.join();if(failure)std::rethrow_exception(failure);queue.Finish();
    }
    std::ifstream gpu(path);std::string line;std::getline(gpu,line);unsigned valid{},queued{};
    while(std::getline(gpu,line)) {
        if(line.find("queued render")!=std::string::npos)++queued;
        std::vector<std::string> fields;std::istringstream row{line};std::string field;
        while(std::getline(row,field,','))fields.push_back(field);
        if(fields.size()==11 && fields[6]=="1")++valid;
    }
    Check(queued==1 && valid>=3,"Queued worker lost bounded GPU timestamp attribution");
    std::filesystem::remove_all(directory);
    std::cout<<"320 native worker draws: bounded queue, one reused pass, immutable bindings/scissors/descriptor, released inputs, blocked GPU/submission lifetime, exact1024 pixels, foreign-owner rejection, thread handoff and GPU timestamps passed\n";
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            Metal::Device device;
            if(argc==3 && std::string_view{argv[1]}=="--compiler-cost") {
                ValidateCompilerCosts(argv[2]);return 0;
            }
            if(argc==3 && std::string_view{argv[1]}=="--stage-timing") {
                ValidateStageTiming(device,argv[2]);return 0;
            }
            if(argc==2 && std::string_view{argv[1]}=="--deferred-library-only") {
                ValidateDeferredPipelines(device); ValidateDeferredLibraries(device); return 0;
            }
            if(argc==2 && std::string_view{argv[1]}=="--queued-texture-only") {
                ValidateQueuedTextureTransfers(device);return 0;
            }
            if(argc==2 && std::string_view{argv[1]}=="--stage-dependency-only") {
                ValidateStageDependencies(device);return 0;
            }
            if(argc==2 && std::string_view{argv[1]}=="--buffer-boundary-only") {
                ValidateQueuedBufferBoundaries(device);ValidateSubmissionObservers(device);return 0;
            }
            // Profilers can synchronize a submission internally. Do not hold
            // that submission on an event released only after Submit returns.
            // The normal regression above retains its external GPU gate.
            if(argc==2 && std::string_view{argv[1]}=="--stage-profile-only") {
                ValidateStageDependencies(device,false);return 0;
            }
            ValidateLiveLibraryReuse(device);
            if(argc==2 && std::string_view{argv[1]}=="--live-library-only")return 0;
            ValidateQueuedCompute(device);
            ValidateQueuedTextureTransfers(device);
            ValidatePersistentCompute(device);
            ValidateQueuedBufferBoundaries(device);
            ValidateSubmissionObservers(device);
            ValidateDrainAttribution(device);
            ValidateDeferredPipelines(device);
            ValidateDeferredLibraries(device);
            ValidateRenderBatch(device);
            ValidateQueuedRendering(device);
            if(argc==2 && std::string_view{argv[1]}=="--encoding-worker-only")return 0;
            ValidateHostPhaseProfiles(device);
            if(argc==2 && std::string_view{argv[1]}=="--host-phase-only")return 0;
            if(argc==2 && std::string_view{argv[1]}=="--binding-owner-only") {
                ValidateArgumentBindingLifetimes(device);return 0;
            }
            ValidateCompletedPresentationResources(device);
            if(argc==2 && std::string_view{argv[1]}=="--present-retirement-only")return 0;
            ValidateCompilerProfiles();
            if(argc==2 && std::string_view{argv[1]}=="--compiler-profile-only")return 0;
            ValidateBoundedProfiles(device);
            ValidateBoundedProfiles(device,true);
            ValidateWaitProfiles(device);
            if(argc==2 && std::string_view{argv[1]}=="--profile-window-only")return 0;
            ValidateImmutableRenderPass(device);
            ValidateBufferComputeScope(device);
            ValidateStageDependencies(device);
            ValidateComputeBatches(device);
            if(argc==2 && std::string_view{argv[1]}=="--buffer-scope-only")return 0;
            ValidateDrawableProfiles(device);
            if(argc==2 && std::string_view{argv[1]}=="--variable-null-only") {
                ValidateVariableArguments(device,true);return 0;
            }
            for (unsigned mask = 0; mask < 16; ++mask) {
                for (auto kind : {Metal::ShaderResourceKind::UniformBuffer, Metal::ShaderResourceKind::StorageBuffer}) {
                    Metal::TranslatedShader shader{};
                    constexpr std::array<unsigned, 4> slots{0, 1, 2, 23};
                    for (unsigned i = 0; i < slots.size(); ++i) {
                        Metal::ShaderResource resource{};
                        resource.kind = kind; resource.metal_index = slots[i];
                        resource.active = mask & (1u << i);
                        shader.resources.push_back(resource);
                        resource.kind = Metal::ShaderResourceKind::Texture; resource.active = true;
                        shader.resources.push_back(resource);
                    }
                    const Metal::VertexCaptureProgram vertex{shader, nil, nil};
                    const Metal::CompiledFragmentShader fragment{{std::move(shader), {}}, nil};
                    Check(vertex.uniform_mask == mask && fragment.uniform_mask == mask,
                          "Published vertex/fragment uniform mask lost active/type selection");
                }
            }
            {
                Metal::TranslatedShader shader{};
                Metal::ShaderResource resource{};
                resource.kind = Metal::ShaderResourceKind::UniformBuffer;
                resource.metal_index = 22; resource.count = 2;
                shader.resources.push_back(resource);
                resource.metal_index = UINT32_MAX; resource.count = UINT32_MAX;
                shader.resources.push_back(resource);
                resource.metal_index = 0; resource.count = 0;
                shader.resources.push_back(resource);
                Check(Metal::PicaUniformMask(shader) == 8,
                      "Uniform mask mishandled ranges, zero count or overflowing source index");
            }
            std::cout << "Native immutable uniform masks: 32 vertex/fragment publications plus inactive/type/range boundaries passed\n";
            ValidateVariableArguments(device);
            ValidateArgumentBindingLifetimes(device);
            ValidateSharedUploads(device);
            ValidateImmutableUniforms(device);
            ValidateLutTextureStream(device);
            ValidateRegistrationGates(device);
            ValidateReadyDrawableSubmission(device);
            {
                Metal::Scheduler scheduler{device};
                Metal::TransferRuntime transfer{device, scheduler};
                const auto initial_tick = scheduler.CurrentTick();
                for (unsigned i = 0; i < 100; ++i) {
                    scheduler.SubmitPending();
                    Check(!scheduler.SubmitPendingBeforeDrawable(),
                          "Idle drawable preparation reported a submission");
                }
                Check(scheduler.CurrentTick() == initial_tick,
                      "Idle frame flushing created an empty GPU submission");
                auto output = device.CreateBuffer(64, true);
                bool validated = false;
                transfer.FillBuffer(output, NSMakeRange(0, 64), 73);
                scheduler.CheckAfterCompletion([&] { validated = true; });
                Check(!scheduler.SubmitPendingBeforeDrawable() && scheduler.CurrentTick() == initial_tick,
                      "First drawable preparation split work without an older submission");
                scheduler.SubmitPending();
                Check(scheduler.CurrentTick() == initial_tick + 1,
                      "Pending frame flushing did not submit guest GPU work exactly once");
                scheduler.Finish();
                Check(validated, "Pending frame flushing lost deferred validation");
                for (unsigned i = 0; i < 64; ++i)
                    Check(static_cast<const std::uint8_t*>(output.contents)[i] == 73,
                          "Pending frame flushing lost native GPU bytes");
                transfer.FillBuffer(output, NSMakeRange(0, 64), 74);
                const auto drained_tick = scheduler.CurrentTick();
                Check(!scheduler.SubmitPendingBeforeDrawable() && scheduler.CurrentTick() == drained_tick,
                      "Drained drawable preparation split work after GPU completion");
                scheduler.SubmitPending();
                scheduler.Finish();
                for (unsigned i = 0; i < 64; ++i)
                    Check(static_cast<const std::uint8_t*>(output.contents)[i] == 74,
                          "Drained drawable preparation lost pending native GPU bytes");
                bool wrong_thread = false;
                std::thread other{[&] {
                    try { scheduler.SubmitPending(); }
                    catch (const std::logic_error&) { wrong_thread = true; }
                }};
                other.join();
                Check(wrong_thread, "Idle frame flushing bypassed owner-thread validation");
                wrong_thread = false;
                std::thread drawable_other{[&] {
                    try { scheduler.SubmitPendingBeforeDrawable(); }
                    catch (const std::logic_error&) { wrong_thread = true; }
                }};
                drawable_other.join();
                Check(wrong_thread, "Drawable preparation bypassed owner-thread validation");
                scheduler.CheckAfterCompletion([] { throw std::runtime_error("expected deferred error"); });
                scheduler.SubmitPending();
                bool caught = false;
                try { scheduler.Finish(); }
                catch (const std::runtime_error&) { caught = true; }
                Check(caught, "Pending frame flushing lost a deferred GPU error");
                caught = false;
                try { scheduler.SubmitPending(); }
                catch (const std::runtime_error&) { caught = true; }
                Check(caught, "Idle frame flushing bypassed a retained GPU error");
                caught = false;
                try { scheduler.SubmitPendingBeforeDrawable(); }
                catch (const std::runtime_error&) { caught = true; }
                Check(caught, "Drawable preparation bypassed a retained GPU error");
                std::cout << "Native pending-frame and drawable submission preserve idle/first/drained timelines, guest GPU bytes, deferred checks/errors and thread ownership\n";
            }
            // A retained texture still needs first-use registration of its backing
            // allocation; reused slots must refresh every related timeline tick.
            {
                Metal::Scheduler scheduler{device};
                auto backing = device.CreateBuffer(4096, true);
                auto* pairs = static_cast<float*>(backing.contents);
                for (unsigned i = 0; i < 32; ++i) pairs[i] = float(i + 1);
                auto buffer_texture = device.CreateBufferTexture(backing, 0, 16, MTLPixelFormatRG32Float);
                auto* texture_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                    MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
                texture_descriptor.storageMode = MTLStorageModeShared;
                texture_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
                auto parent = device.CreateTexture(texture_descriptor);
                std::array<float, 64> pixels{};
                for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = float(100 + i);
                [parent replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
                            withBytes:pixels.data() bytesPerRow:16 * sizeof(float)];
                auto view = [parent newTextureViewWithPixelFormat:MTLPixelFormatRGBA32Float];
                Check(view && view.parentTexture == parent && buffer_texture.buffer == backing,
                      "Texture residency fixture has no expected backing allocation");
                auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void backing_read(texture_buffer<float, access::read> pairs [[texture(0)]],
                         texture2d<float, access::read> pixels [[texture(1)]],
                         device float2* output [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    output[i] = pairs.read(i).xy + pixels.read(uint2(i % 4, i / 4)).zw;
}
)", "Native scheduler texture backing fixture");
                auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
                descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "backing_read");
                auto pipeline = device.CompileCompute(descriptor);
                std::array<id<MTLBuffer>, 12> outputs{};
                for (unsigned batch = 0; batch < outputs.size(); ++batch) {
                    outputs[batch] = device.CreateBuffer(32 * sizeof(float), true);
                    scheduler.KeepAlive(view); scheduler.KeepAlive(buffer_texture);
                    scheduler.KeepAlive(pipeline);
                    const std::array buffers{Metal::BufferBinding{outputs[batch], 0, 0}};
                    const std::array textures{Metal::TextureBinding{buffer_texture, 0},
                                              Metal::TextureBinding{view, 1}};
                    auto table = scheduler.MakeArguments(buffers, textures);
                    for (unsigned reuse = 0; reuse < 100; ++reuse) {
                        scheduler.UseResource(view); scheduler.UseResource(buffer_texture);
                        scheduler.UseResource(backing); scheduler.UseResource(parent);
                    }
                    for (id<MTLResource> resource : std::array<id<MTLResource>, 5>{
                             view, parent, buffer_texture, backing, outputs[batch]})
                        Check(scheduler.LastUseTick(resource) == scheduler.CurrentTick(),
                              "Retained texture/backing resource did not receive the new submission tick");
                    Check(scheduler.CurrentRetainedObjectCount() == 7,
                          "Texture backing uses duplicated or omitted submission lifetime objects");
                    if (!batch) {
                        bool wrong_thread{};
                        std::thread other{[&] { try { scheduler.UseResource(view); }
                            catch (const std::logic_error&) { wrong_thread = true; } }};
                        other.join();
                        Check(wrong_thread, "Repeated resource use bypassed owner-thread validation");
                    }
                    auto encoder = scheduler.BeginCompute();
                    [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
                    [encoder dispatchThreads:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
                    scheduler.EndEncoding(); scheduler.Submit();
                }
                view = nil; parent = nil; buffer_texture = nil; backing = nil;
                scheduler.Finish();
                for (auto output : outputs)
                    for (unsigned i = 0; i < 16; ++i) {
                        const auto* values = static_cast<const float*>(output.contents);
                        Check(values[i * 2] == float(i * 2 + 1) + pixels[i * 4 + 2] &&
                              values[i * 2 + 1] == float(i * 2 + 2) + pixels[i * 4 + 3],
                              "Texture view/buffer backing residency lost native GPU values across slot reuse");
                    }
                std::cout << "Native scheduler: retained-before-use textures, parent/buffer residency and ticks, 12 rotating-slot submissions and repeated-use owner rejection passed\n";
            }
            // Argument tables are snapshotted by dispatch. Overwrite one table
            // before submission, release CPU inputs, then reuse every submission slot.
            {
                Metal::Scheduler scheduler{device};
                auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void tagged(const device uint* input [[buffer(0)]],
                   device uint* output [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = input[i] ^ (i * 0x9e3779b9u);
}
)", "Native reusable argument-table snapshot fixture");
                auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
                descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "tagged");
                descriptor.maxTotalThreadsPerThreadgroup = 64;
                auto pipeline = device.CompileCompute(descriptor);
                id<MTL4ArgumentTable> table = nil;
                struct Output {
                    id<MTLBuffer> result, poison;
                    unsigned count, tag;
                };
                std::array<Output, 42> outputs{};
                for (unsigned epoch = 0; epoch < outputs.size(); ++epoch) {
                    @autoreleasepool {
                        const unsigned count = std::array<unsigned, 7>{1, 31, 32, 33, 65, 257, 513}[epoch % 7];
                        const unsigned tag = 0x12340000u + epoch * 911u;
                        auto input = device.CreateBuffer((count + 1) * sizeof(unsigned), true);
                        auto result = device.CreateBuffer(count * sizeof(unsigned), true);
                        auto poison_input = device.CreateBuffer(input.length, true);
                        auto poison_result = device.CreateBuffer(result.length, true);
                        auto* values = static_cast<unsigned*>(input.contents);
                        auto* poison = static_cast<unsigned*>(poison_input.contents);
                        for (unsigned i = 0; i < count; ++i) {
                            values[i] = tag + i * 271u;
                            poison[i] = 0xdeadbeefu;
                        }
                        values[count] = poison[count] = count;
                        std::memset(result.contents, 0xcd, result.length);
                        std::memset(poison_result.contents, 0, poison_result.length);
                        const std::array bindings{
                            Metal::BufferBinding{input, 0, 0},
                            Metal::BufferBinding{result, 0, 1},
                            Metal::BufferBinding{input, count * sizeof(unsigned), 2}};
                        if (table) scheduler.RebindBufferArguments(table, bindings);
                        else table = scheduler.MakeArguments(bindings);
                        auto encoder = scheduler.BeginCompute(pipeline);
                        [encoder setComputePipelineState:pipeline];
                        [encoder setArgumentTable:table];
                        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        const std::array overwritten{
                            Metal::BufferBinding{poison_input, 0, 0},
                            Metal::BufferBinding{poison_result, 0, 1},
                            Metal::BufferBinding{poison_input, count * sizeof(unsigned), 2}};
                        scheduler.RebindBufferArguments(table, overwritten);
                        scheduler.EndEncoding();
                        scheduler.Submit();
                        outputs[epoch] = {result, poison_result, count, tag};
                    }
                }
                scheduler.Finish();
                for (const auto& output : outputs)
                    for (unsigned i = 0; i < output.count; ++i) {
                        Check(static_cast<unsigned*>(output.result.contents)[i] ==
                                  ((output.tag + i * 271u) ^ (i * 0x9e3779b9u)),
                              "Rebinding before submission changed an encoded argument snapshot");
                        Check(static_cast<unsigned*>(output.poison.contents)[i] == 0,
                              "An encoded dispatch used overwritten argument-table bindings");
                    }
                bool nil_rejected{};
                try { scheduler.RebindBufferArguments(nil, {}); }
                catch (const std::invalid_argument&) { nil_rejected = true; }
                Check(nil_rejected, "Reusable argument binding accepted a nil table");
                auto buffer = device.CreateBuffer(16, true);
                bool duplicate{};
                try {
                    const std::array bindings{Metal::BufferBinding{buffer, 0, 0},
                                              Metal::BufferBinding{buffer, 0, 0}};
                    scheduler.RebindBufferArguments(table, bindings);
                } catch (const std::invalid_argument&) { duplicate = true; }
                Check(duplicate, "Reusable argument binding accepted duplicate buffer slots");
                bool wrong_thread{};
                std::thread other{[&] {
                    try { scheduler.RebindBufferArguments(table, {}); }
                    catch (const std::logic_error&) { wrong_thread = true; }
                }};
                other.join();
                Check(wrong_thread, "Reusable argument binding bypassed owner-thread validation");
                scheduler.Finish();
                std::cout << "Native reusable tables: 42 overwritten-before-submit snapshots, released inputs, seven dispatch sizes, rotating-slot reuse and invalid/foreign-thread rejection passed\n";
            }
            auto released = std::make_shared<std::atomic_uint>();
            @autoreleasepool {
                Metal::Scheduler scheduler{device};
                Metal::TransferRuntime transfer{device, scheduler};
                auto buffer = device.CreateBuffer(64, true);
                for (unsigned i = 0; i < 10000; ++i) {
                    scheduler.UseResource(buffer); scheduler.KeepAlive(buffer);
                }
                Check(scheduler.CurrentRetainedObjectCount() == 1,
                      "Repeated resource binding grows the submission retain set");
                transfer.FillBuffer(buffer, NSMakeRange(0, 64), 37);
                const auto pending = scheduler.LastUseTick(buffer);
                Check(pending == scheduler.CurrentTick(), "Resource was not assigned its pending submission tick");
                scheduler.Wait(pending);
                Check(scheduler.IsComplete(pending), "Waiting for a pending resource did not flush and finish it");
                for (unsigned i = 0; i < 64; ++i)
                    Check(static_cast<std::uint8_t*>(buffer.contents)[i] == 37, "Pending-wait GPU bytes mismatch");
                auto library = device.CompileLibrary(R"(
#include <metal_stdlib>
using namespace metal;
kernel void timeline_copy(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) { output[i] = input[i] ^ 0xa55a1234u; }
)", "Native scheduler lifetime fixture");
                auto* descriptor = [[MTL4ComputePipelineDescriptor alloc] init];
                descriptor.computeFunctionDescriptor = Metal::Device::Function(library, "timeline_copy");
                auto pipeline = device.CompileCompute(descriptor);
                std::array<id<MTLBuffer>, 48> outputs{};
                for (unsigned batch = 0; batch < outputs.size(); ++batch) {
                    @autoreleasepool {
                        auto input = device.CreateBuffer(64, true);
                        for (unsigned i = 0; i < 16; ++i)
                            static_cast<std::uint32_t*>(input.contents)[i] = batch * 101 + i;
                        outputs[batch] = device.CreateBuffer(64, true);
                        std::memset(outputs[batch].contents, 0, 64);
                        auto* probe = [[LifetimeProbe alloc] init]; probe->counter = released;
                        scheduler.KeepAlive(probe); scheduler.KeepAlive(pipeline);
                        const std::array bindings{Metal::BufferBinding{input, 0, 0},
                                                 Metal::BufferBinding{outputs[batch], 0, 1}};
                        auto table = scheduler.MakeArguments(bindings);
                        for (unsigned reuse = 0; reuse < 1000; ++reuse) {
                            scheduler.UseResource(input); scheduler.UseResource(outputs[batch]);
                            scheduler.KeepAlive(table); scheduler.KeepAlive(pipeline);
                        }
                        Check(scheduler.CurrentRetainedObjectCount() == 5,
                              "Repeated PSO/table/buffer references are not deduplicated by object identity");
                        auto encoder = scheduler.BeginCompute();
                        [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
                        [encoder dispatchThreads:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
                        scheduler.EndEncoding(); scheduler.Submit();
                    }
                }
                scheduler.Finish();
                for (unsigned batch = 0; batch < outputs.size(); ++batch)
                    for (unsigned i = 0; i < 16; ++i)
                        Check(static_cast<std::uint32_t*>(outputs[batch].contents)[i] == ((batch * 101 + i) ^ 0xa55a1234u),
                              "Triple-slot reuse lost released CPU input or argument-table GPU state");
                bool wrong_thread{};
                std::thread other{[&] { try { scheduler.BeginCompute(); }
                    catch (const std::logic_error&) { wrong_thread = true; } }};
                other.join(); Check(wrong_thread, "Foreign-thread command encoding was accepted");
                bool finish_failed{};
                std::thread failed_finish{[&] { finish_failed = !scheduler.FinishNoThrow(); }};
                failed_finish.join();
                Check(finish_failed && scheduler.FinishNoThrow(),
                      "Nonthrowing finish lost failure reporting or broke a valid owner-thread finish");
                transfer.FillBuffer(buffer, NSMakeRange(0, 64), 41);
                scheduler.ReleaseThread();
                std::exception_ptr handoff_error;
                std::thread emulation{[&] {
                    try {
                        for (unsigned i = 0; i < 64; ++i)
                            Check(static_cast<std::uint8_t*>(buffer.contents)[i] == 41,
                                  "Ownership release did not retire initialization GPU work");
                        transfer.FillBuffer(buffer, NSMakeRange(0, 64), 53);
                        scheduler.ReleaseThread();
                    } catch (...) { handoff_error = std::current_exception(); }
                }};
                emulation.join();
                if (handoff_error) std::rethrow_exception(handoff_error);
                scheduler.Finish();
                for (unsigned i = 0; i < 64; ++i)
                    Check(static_cast<std::uint8_t*>(buffer.contents)[i] == 53,
                          "Emulation handoff lost GPU writes or return ownership");
                std::cout << "Explicit drained frontend/emulation ownership handoff preserves GPU writes\n";
                std::cout << "Native nonthrowing finish reports a caught owner violation and preserves a successful owner-thread finish\n";
                bool future{};
                try { scheduler.Wait(scheduler.CurrentTick() + 1); }
                catch (const std::invalid_argument&) { future = true; }
                Check(future, "Unsubmitted future timeline wait was accepted");
                {
                    Metal::StreamBuffer owned{device, scheduler, 4096};
                    std::array<Metal::BufferSlice, 3> slices;
                    slices[0] = owned.Allocate(64);
                    for (unsigned attempt = 0; attempt < 2; ++attempt) {
                        bool rejected{};
                        std::thread foreign{[&] {
                            try { owned.Allocate(attempt ? 8192 : 64); }
                            catch (const std::logic_error&) { rejected = true; }
                        }};
                        foreign.join();
                        Check(rejected, "Foreign stream allocation was accepted");
                        slices[attempt + 1] = owned.Allocate(64);
                        Check(slices[attempt + 1].offset == (attempt + 1) * 256,
                              "Rejected stream allocation mutated page offset");
                        Check(slices[attempt + 1].buffer == slices[0].buffer,
                              "Rejected stream allocation replaced the active page");
                    }
                    auto copied = device.CreateBuffer(192, true);
                    for (unsigned i = 0; i < slices.size(); ++i) {
                        std::memset(slices[i].data, 0x31 + i, 64);
                        transfer.CopyBuffer(slices[i].buffer, slices[i].offset, copied, i * 64, 64);
                    }
                    slices = {};
                    scheduler.Finish();
                    for (unsigned i = 0; i < 192; ++i)
                        Check(static_cast<const std::uint8_t*>(copied.contents)[i] == 0x31 + i / 64,
                              "Stream rejection damaged subsequent native GPU uploads");
                    std::cout << "Native stream owner rejection preserves offsets/page identity and 192 GPU-consumed bytes, including oversized foreign calls\n";
                }
                Metal::StreamBuffer stream{device, scheduler, 64};
                std::array<Metal::BufferSlice, 3> held;
                for (unsigned i = 0; i < held.size(); ++i) {
                    held[i] = stream.Allocate(64); std::memset(held[i].data, 80 + i, 64);
                }
                bool lease{};
                try { stream.Allocate(64); } catch (const std::length_error&) { lease = true; }
                Check(lease, "An unconsumed CPU upload lease was overwritten");
                auto output = device.CreateBuffer(192, true);
                for (unsigned i = 0; i < held.size(); ++i)
                    transfer.CopyBuffer(held[i].buffer, held[i].offset, output, i * 64, 64);
                for (auto& slice : held) slice = {};
                // Page reuse must submit/wait the still pending copy before CPU writes.
                auto reused = stream.Allocate(64); std::memset(reused.data, 99, 64);
                scheduler.Finish();
                for (unsigned i = 0; i < 192; ++i)
                    Check(static_cast<std::uint8_t*>(output.contents)[i] == 80 + i / 64,
                          "Stream page recycling raced its pending native GPU copy");
                auto oversized = stream.Allocate(256); std::memset(oversized.data, 123, 256);
                auto oversized_output = device.CreateBuffer(256, true);
                transfer.CopyBuffer(oversized.buffer, 0, oversized_output, 0, 256);
                oversized = {}; scheduler.Finish();
                for (unsigned i = 0; i < 256; ++i)
                    Check(static_cast<std::uint8_t*>(oversized_output.contents)[i] == 123,
                          "Dedicated oversized upload storage was released before GPU completion");
                for (auto [size, alignment] : std::array<std::pair<NSUInteger, NSUInteger>, 3>{{{0, 16}, {1, 0}, {1, 3}}}) {
                    bool invalid{}; try { stream.Allocate(size, alignment); }
                    catch (const std::invalid_argument&) { invalid = true; }
                    Check(invalid, "Invalid upload size/alignment was accepted");
                }
            }
            for (unsigned attempt = 0; attempt < 1000 && *released != 48; ++attempt)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            Check(*released == 48, "Submission payloads retained CPU lifetime probes after scheduler teardown");
            std::cout << "Native scheduler: 10000 duplicate resource uses and 48 rotating-slot GPU submissions, unique PSO/table/buffer retention, pending wait, released CPU inputs, ownership/future rejection and teardown lifetime passed\n";
            std::cout << "Native stream: three held CPU leases, pending GPU-copy page reuse, dedicated oversized buffers and invalid alignment checks passed\n";
            // An outer pool deliberately spans every submission. Core memory flushes
            // can call Metal from plain C++ threads without a per-frame/draw pool.
            // GPU completion must release submission snapshots before that pool drains.
            auto unpooled_released = std::make_shared<std::atomic_uint>();
            {
                Metal::Scheduler scheduler{device};
                Metal::TransferRuntime transfer{device, scheduler};
                id<MTLBuffer> last_buffer;
                for (unsigned batch = 0; batch < 96; ++batch) {
                    for (unsigned draw = 0; draw < 16; ++draw) {
                        auto buffer = device.CreateBuffer(65536, true);
                        transfer.FillBuffer(buffer, NSMakeRange(0, 65536), batch + draw);
                        auto* probe = [[LifetimeProbe alloc] init];
                        probe->counter = unpooled_released;
                        scheduler.KeepAlive(probe);
                        last_buffer = buffer;
                    }
                    scheduler.Submit();
                    if (batch % 8 == 7) {
                        scheduler.Finish();
                        const auto minimum = (batch + 1 - Metal::Scheduler::SubmissionSlotCount) * 16;
                        for (unsigned attempt = 0; attempt < 1000 && *unpooled_released < minimum; ++attempt)
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        Check(*unpooled_released >= minimum,
                              "Completed submission snapshots wait for an external autorelease pool");
                        const auto* bytes = static_cast<const std::uint8_t*>(last_buffer.contents);
                        for (unsigned i = 0; i < 65536; ++i)
                            Check(bytes[i] == batch + 15,
                                  "Internal submission pool released an in-flight GPU buffer");
                    }
                }
            }
            for (unsigned attempt = 0; attempt < 1000 && *unpooled_released != 1536; ++attempt)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            Check(*unpooled_released == 1536, "Unpooled submission snapshots survived scheduler teardown");
            std::cout << "Native scheduler: 96 submissions without an external draining pool release 1536 lifetime probes, preserve GPU bytes and release all retired slots\n";
            std::cout << "No app/window/ROM/emulation started\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
