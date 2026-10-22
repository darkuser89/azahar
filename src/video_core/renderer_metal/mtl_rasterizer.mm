// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_rasterizer.h"
#include "mtl_gas_runtime.h"
#include "core/core.h"
#include "core/loader/loader.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/texture/texture_decode.h"
#include "common/logging/log.h"
#include <cstring>
#include <bit>
#include <limits>
#include <optional>
#include <stdexcept>

namespace Metal {
namespace {
ShaderMachineUniforms MachineUniforms(const Pica::ShaderSetup& setup,unsigned first=0,unsigned count=96,
    std::uint32_t bool_mask=0xffff,std::uint32_t integer_mask=15) {
    ShaderMachineUniforms values{};
    for(unsigned i=first;i<first+count;++i) for(unsigned c=0;c<4;++c) values.f[i][c]=setup.uniforms.f[i][c].ToFloat32();
    for(unsigned i=0;i<16;++i) if((bool_mask&(1u<<i)) && setup.uniforms.b[i]) values.b|=1u<<i;
    for(unsigned i=0;i<4;++i) if(integer_mask&(1u<<i))
        for(unsigned c=0;c<4;++c) values.i[i][c]=setup.uniforms.i[i][c];
    return values;
}
ShaderMachineProgramView MachineProgram(Pica::ShaderSetup& setup,const Pica::ShaderRegs& config,bool geometry) {
    setup.DoProgramCodeFixup();
    return {setup.GetProgramCode(),setup.GetSwizzleData(),config.main_offset.Value(),
            config.output_mask.Value(),geometry};
}
bool NeedsMachine(Pica::ShaderSetup& setup,const Pica::ShaderRegs& config) {
    setup.DoProgramCodeFixup();
    const auto& code=setup.GetProgramCode();
    for(unsigned pc=config.main_offset.Value();pc<code.size();++pc) {
        const auto word=code[pc],op=word>>26;
        if(op==0x22) return false;
        if(op==0x04 || op==0x07 || op==0x19 || op==0x20 || (op>=0x23 && op<=0x2d) ||
           (op>=0x2e && op<=0x2f && (((word>>24)&7)>5 || ((word>>21)&7)>5))) return true;
        if((op>=0x10 && op<=0x11) || (op>=0x14 && op<=0x17) || (op>=0x1c && op<=0x1f)) return true;
        const bool mad=op>=0x30,inverted=op==0x18 || op==0x1a || op==0x1b || (mad && op<0x38);
        const auto temporary=[](unsigned reg) { return reg>=16 && reg<32; };
        if(mad) {
            if(temporary((word>>17)&31) || temporary((word>>(inverted ? 12:10))&(inverted ? 31:127)) ||
               temporary((word>>5)&(inverted ? 127:31))) return true;
        } else if(temporary((word>>(inverted ? 14:12))&(inverted ? 31:127)) ||
                  temporary((word>>7)&(inverted ? 127:31))) return true;
    }
    return true; // Missing END is handled by the bounded GPU machine, never by recursive decompilation.
}
}

RasterizerMetal::RasterizerMetal(Memory::MemorySystem& memory_, Pica::PicaCore& pica_,
                                VideoCore::CustomTexManager& custom_textures,
                                VideoCore::RendererBase& renderer, Device& device_, Scheduler& scheduler_)
    : RasterizerAccelerated{memory_, pica_}, device{device_}, scheduler{scheduler_},
      textures{device, scheduler}, cache{memory, custom_textures, textures, regs, renderer},
      geometry{device, scheduler}, primitive_stream{device}, vertex_input_state{device}, fetch{device, scheduler},
      input_references{device, scheduler}, shaders{device},
      draws{device, scheduler, geometry}, machine{device,scheduler},vs_unit{device},gs_unit{device},
      geometry_program{device,scheduler},geometry_input{device},gas_runtime{device,scheduler},gas_feedback{device},
      uploads{device, scheduler, 4 * 1024 * 1024}, vertex_uploads{device, scheduler},
      uniform_uploads{device, scheduler}, lighting_luts{device, scheduler, MTLPixelFormatRG32Float},
      proctex_value_luts{device, scheduler, MTLPixelFormatRG32Float},
      proctex_color_luts{device, scheduler, MTLPixelFormatRGBA32Float} {
    mesh_empty = device.CreateBuffer(512, true);
    std::memset(mesh_empty.contents, 0, 512);
    textures.SetTextureFilterProvider([] { return Settings::values.texture_filter.GetValue(); });
    pica.dirty_regs.SetAllDirty();
}

RasterizerMetal::~RasterizerMetal() {
    try { RetireGeometry(true); }
    catch(const std::exception& error) { LOG_ERROR(Render,"Metal4 deferred GS shutdown: {}",error.what()); }
    scheduler.FinishNoThrow();
}

void RasterizerMetal::RetireGeometry(bool finish) {
    if(deferred_geometry.empty()) return;
    if(finish) scheduler.Finish(Scheduler::WaitReason::Readback);
    while(!deferred_geometry.empty()) {
        auto& item=deferred_geometry.front();
        if(!item.ready || !scheduler.IsComplete(item.tick)) break;
        const auto status=machine.PendingStatus(item.invocation.execution);
        try {
            if(status[0]) throw ShaderMachineError(status[0],status[3]);
            if(item.vertices.indirect) machine.ValidateIndirectResult(item.vertices);
        } catch(...) {
            // The shared abort guard prevented every later GS unit commit.
            // Retire its already queued transport before restoring the failed
            // ready group and original CPU reset marker, then report its error.
            scheduler.Finish(Scheduler::WaitReason::Readback);
            geometry_program.RestoreQueuedInput(geometry_input,item.invocation);
            if(status[0]) machine.RestoreFailedGeometryReset(gs_unit,item.invocation.execution);
            std::uint32_t boolean_word{};
            std::memcpy(&boolean_word,static_cast<const std::byte*>(item.invocation.history.buffer.contents)+
                item.invocation.history.offset+offsetof(ShaderMachineUniforms,b),sizeof(boolean_word));
            pica.gs_setup.uniforms.b[15]=(boolean_word&0x8000u)!=0;
            scheduler.Finish(Scheduler::WaitReason::Readback);deferred_geometry.clear();deferred_geometry_abort={};
            throw;
        }
        deferred_geometry.pop_front();
    }
    if(deferred_geometry.empty()) deferred_geometry_abort={};
}

void RasterizerMetal::QueueGeometryDraw(const ShaderMachineProgramView& program,
                                      std::vector<DeferredGeometry*>* batch) {
    if(!batch || batch->empty()) RetireGeometry();
    // Match the existing GS machine's full accepted emission ceiling. These
    // two native scratch allocations are reused only after encoded consumers,
    // with queue-stage barriers protecting both dispatch and vertex reads.
    constexpr std::uint32_t capacity=1000000;
    const bool compact=bool(ShaderMachineRuntime::LinearGeometryStorageBound(program));
    if(batch && !compact) throw std::logic_error("Batched native GS lacks disjoint proved storage");
    if(!compact && !gs_emission_scratch.buffer) {
        const auto bytes=NSUInteger(capacity)*sizeof(ShaderMachineEmission);
        const auto captured=NSUInteger(capacity-capacity%3)*sizeof(CapturedVertex);
        auto emissions=device.CreateBuffer(bytes,false),vertices=device.CreateBuffer(captured,false);
        gs_emission_scratch={emissions,0,bytes,nullptr,{}};gs_capture_scratch={vertices,0,captured,nullptr,{}};
    }
    if(!deferred_geometry_abort.buffer) deferred_geometry_abort=machine.BeginGeometryBatch();
    auto pending=geometry_program.InvokeQueuedImpl(geometry_input,machine,gs_unit,program,
        deferred_geometry_abort,capacity,compact ? BufferSlice{}:gs_emission_scratch,compact);
    deferred_geometry.push_back({std::move(pending),{},0,false});
    auto& item=deferred_geometry.back();
    try {
        pica.gs_setup.uniforms.b[15]=true;
        SyncDrawUniforms();
        GeometryEmissionConfig config{};config.attribute_count=regs.rasterizer.vs_output_total.Value();
        for(unsigned i=0;i<7;++i) config.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
        for(unsigned c=0;c<4;++c) config.clip_coefficient[c]=vs_data.clip_coef[c];
        config.flip_viewport=vs_data.flip_viewport;config.enable_clip=vs_data.enable_clip1;
        config.honor_winding=regs.pipeline.triangle_topology==Pica::PipelineRegs::TriangleTopology::Shader;
        config.fix_quaternions=regs.lighting.disable==0;
        item.vertices=machine.ConvertEmissionListIndirect(item.invocation.execution,config,primitive_stream,
            compact ? BufferSlice{}:gs_capture_scratch);
        item.tick=item.vertices.indirect->tick;
        if(batch) batch->push_back(&item);
        else { DrawNativeVertices(item.vertices);
            item.tick=std::max(item.vertices.indirect->tick,scheduler.LastUseTick(item.vertices.storage.buffer));
            item.ready=true; }
    } catch(...) {
        item.tick=item.vertices.indirect ? item.vertices.indirect->tick:item.invocation.execution.tick;
        item.ready=true;throw;
    }
    if(!gs_deferred_logged) {
        LOG_INFO(Render,"Metal4 deferred GS raster active: GPU emission count, logical_capacity={}, compact={}, emission_bytes={}, scratch_bytes={}",
            capacity,compact,item.invocation.execution.result.emissions.size,gs_emission_scratch.size+gs_capture_scratch.size);
        gs_deferred_logged=true;
    }
    // Bounded producer batches reach the GPU without a count/readback wait.
    // The scheduler's three submission slots bound CPU lead and retained leases.
    if(deferred_geometry.size()>=256 && deferred_geometry.size()%256==0) scheduler.SubmitPending();
}

void RasterizerMetal::LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                               const VideoCore::DiskResourceLoadCallback& callback) {
    if(callback)callback(VideoCore::LoadCallbackStage::Prepare,0,0,"");
    if(!stop_loading.load())device.SetDiskCacheEnabled(Settings::values.use_disk_shader_cache.GetValue());
    if(!stop_loading.load() && Settings::values.use_disk_shader_cache.GetValue() && device.DiskCacheStats().enabled) {
        u64 title_id{};
        if(Core::System::GetInstance().GetAppLoader().ReadProgramId(title_id)!=Loader::ResultStatus::Success)title_id=0;
        if(callback)callback(VideoCore::LoadCallbackStage::Preload,0,1,"Native Metal4 pipeline archives");
        device.SwitchDiskCache(title_id,&stop_loading);
        shaders.Clear(); draws.Clear();
        if(callback)callback(VideoCore::LoadCallbackStage::Preload,1,1,"Native Metal4 pipeline archives");
        shaders.Prewarm(stop_loading, [&](std::size_t done, std::size_t total) {
            if(callback)callback(VideoCore::LoadCallbackStage::Build,done,total,"Native Metal4 PICA shaders");
        });
        draws.Prewarm(stop_loading, [&](std::size_t done, std::size_t total) {
            if(callback)callback(VideoCore::LoadCallbackStage::Build,done,total,"Native Metal4 render pipelines");
        });
        if (!stop_loading.load() && presentation_prewarm) presentation_prewarm(stop_loading, callback);
    }
    if(callback)callback(VideoCore::LoadCallbackStage::Complete,0,0,"");
}
void RasterizerMetal::SwitchDiskResources(u64 title_id) {
    if(switch_disk_resources_callback)switch_disk_resources_callback(VideoCore::LoadCallbackStage::Prepare,0,0,"");
    device.SetDiskCacheEnabled(Settings::values.use_disk_shader_cache.GetValue());
    if(Settings::values.use_disk_shader_cache.GetValue() && device.DiskCacheStats().enabled) {
        device.SwitchDiskCache(title_id);shaders.Clear();draws.Clear();
        const std::atomic_bool stop{false};
        shaders.Prewarm(stop, [&](std::size_t done, std::size_t total) {
            if(switch_disk_resources_callback)
                switch_disk_resources_callback(VideoCore::LoadCallbackStage::Build,done,total,"Native Metal4 PICA shaders");
        });
        draws.Prewarm(stop, [&](std::size_t done, std::size_t total) {
            if(switch_disk_resources_callback)
                switch_disk_resources_callback(VideoCore::LoadCallbackStage::Build,done,total,"Native Metal4 render pipelines");
        });
        if (presentation_prewarm) presentation_prewarm(stop, switch_disk_resources_callback);
    }
    if(switch_disk_resources_callback)switch_disk_resources_callback(VideoCore::LoadCallbackStage::Complete,0,0,"");
}

void RasterizerMetal::AddTriangle(const Pica::OutputVertex&, const Pica::OutputVertex&, const Pica::OutputVertex&) {
    throw std::runtime_error("Metal received CPU-processed vertices instead of native GPU input");
}

void RasterizerMetal::DrawTriangles() {
    if (!vertex_batch.empty()) throw std::runtime_error("Metal cannot execute a CPU vertex fallback");
}

BufferSlice RasterizerMetal::UploadGuestBytes(std::uint64_t address, std::uint64_t size, bool dedicated) {
    if (!size || address > std::numeric_limits<PAddr>::max() ||
        size > std::numeric_limits<u32>::max() || address + size > std::uint64_t(std::numeric_limits<PAddr>::max()) + 1) {
        throw std::invalid_argument("Metal vertex/index range exceeds physical address storage");
    }
    const auto source = memory.GetPhysicalRef(PAddr(address));
    if (!source || source.GetSize() < size) throw std::invalid_argument("Metal vertex/index range exceeds physical memory");
    cache.FlushRegion(PAddr(address), u32(size));
    RetireGeometry();
    BufferSlice result{};
    if (dedicated) {
        // A compiled batch may hold all twelve loaders plus its index stream
        // before encoding any consumer. Immutable upload ranges avoid
        // exhausting the three CPU-leased upload pages for a valid large draw.
        result = vertex_uploads.Allocate(NSUInteger(size));
    } else {
        result = uploads.Allocate(size);
    }
    std::memcpy(result.data, source.GetPtr(), size);
    return result;
}

RasterizerMetal::UniformUploads RasterizerMetal::UploadUniforms(std::uint8_t mask) {
    constexpr std::array<NSUInteger, 4> slots{0, 1, 2, 23};
    UniformUploads result{};
    const auto block = [&](unsigned index, const auto& value) {
        auto& storage = result.storage[index];
        storage = uniform_uploads.Upload(index, std::as_bytes(std::span{&value, 1}));
        result.bindings[result.count++] = {storage.buffer, storage.offset, slots[index]};
    };
    if (mask & 1) {
        Pica::Shader::Generator::VSPicaUniformData guest{};
        guest.SetFromRegs(pica.vs_setup);
        block(0, guest);
    }
    if (mask & 2) block(1, vs_data);
    if (mask & 4) block(2, fs_data);
    if (mask & 8) {
        GasUniforms gas_uniforms{};
        gas_uniforms.color_lut = gas.lut;
        for (unsigned i = 0; i < gas_uniforms.fog_lut.size(); ++i)
            gas_uniforms.fog_lut[i] = pica.fog.lut[i].raw;
        gas_uniforms.attenuation = regs.reg_array[0xE4];
        gas_uniforms.reciprocal_maximum = regs.reg_array[0xE5];
        gas_uniforms.light_xy = regs.reg_array[0x120];
        gas_uniforms.light_z = regs.reg_array[0x121];
        gas_uniforms.light_z_color = regs.reg_array[0x122];
        gas_uniforms.shading_depth = (regs.reg_array[0xE0] >> 3) & 1;
        gas_uniforms.delta_z_depth = regs.reg_array[0x126];
        block(3, gas_uniforms);
    }
    // Compare all CPU-owned constant bytes, including values changed by load,
    // clip state or Gas. Dirty flags and shader mask changes cannot hide writes.
    // Every returned block retains its own immutable range until encoding.
    return result;
}

void RasterizerMetal::SyncLUTs(const CompiledFragmentShader& fragment, std::vector<TextureBinding>& bindings) {
    // Keep all current offsets in one texture page, like Vulkan's LUT ring.
    // Only a page rollover requires republishing unchanged logical tables.
    if (fragment.texture_mask.test(0) &&
        (!lut_textures[0] || pica.lighting.lut_dirty || pica.fog.lut_dirty)) {
        const bool replace = lighting_luts.BeginUpdate((24 * 256 + 128) * sizeof(Common::Vec2f));
        const u32 dirty = replace ? pica.lighting.LutAllDirty : pica.lighting.lut_dirty & pica.lighting.LutAllDirty;
        for (unsigned lut = 0; lut < 24; ++lut) {
            if (!(dirty & (1u << lut))) continue;
            const auto storage = lighting_luts.Append(256 * sizeof(Common::Vec2f));
            auto* dest = static_cast<Common::Vec2f*>(storage.data);
            for (unsigned i = 0; i < 256; ++i) {
                const auto& entry = pica.lighting.luts[lut][i];
                dest[i] = {entry.ToFloat(), entry.DiffToFloat()};
            }
            fs_data.lighting_lut_offset[lut / 4][lut % 4] = lighting_luts.TexelOffset(storage);
        }
        if (replace || pica.fog.lut_dirty) {
            const auto storage = lighting_luts.Append(128 * sizeof(Common::Vec2f));
            auto* dest = static_cast<Common::Vec2f*>(storage.data);
            for (unsigned i = 0; i < 128; ++i)
                dest[i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
            fs_data.fog_lut_offset = lighting_luts.TexelOffset(storage);
        }
        lut_textures[0] = lighting_luts.Texture();
        pica.lighting.lut_dirty = 0;
        pica.fog.lut_dirty = false;
    }
    // Value and color tables have separate texture consumers. Leave writes for
    // an inactive consumer pending; changing only noise must not lose color data.
    constexpr u8 value_bits = (1u << 0) | (1u << 2) | (1u << 3);
    constexpr u8 color_bits = (1u << 4) | (1u << 5);
    if (fragment.texture_mask.test(1) && (!lut_textures[1] || (pica.proctex.table_dirty & value_bits))) {
        const bool replace = proctex_value_luts.BeginUpdate(3 * 128 * sizeof(Common::Vec2f));
        const auto dirty = replace ? value_bits : pica.proctex.table_dirty & value_bits;
        const std::array sources{&pica.proctex.noise_table, &pica.proctex.color_map_table, &pica.proctex.alpha_map_table};
        const std::array bits{1u << 0, 1u << 2, 1u << 3};
        const std::array offsets{&fs_data.proctex_noise_lut_offset, &fs_data.proctex_color_map_offset,
                                 &fs_data.proctex_alpha_map_offset};
        for (unsigned lut = 0; lut < 3; ++lut) {
            if (!(dirty & bits[lut])) continue;
            const auto storage = proctex_value_luts.Append(128 * sizeof(Common::Vec2f));
            auto* dest = static_cast<Common::Vec2f*>(storage.data);
            for (unsigned i = 0; i < 128; ++i)
                dest[i] = {(*sources[lut])[i].ToFloat(), (*sources[lut])[i].DiffToFloat()};
            *offsets[lut] = proctex_value_luts.TexelOffset(storage);
        }
        lut_textures[1] = proctex_value_luts.Texture();
        pica.proctex.table_dirty &= color_bits;
    }
    if (fragment.texture_mask.test(2) && (!lut_textures[2] || (pica.proctex.table_dirty & color_bits))) {
        const bool replace = proctex_color_luts.BeginUpdate(2 * 256 * sizeof(Common::Vec4f));
        const auto dirty = replace ? color_bits : pica.proctex.table_dirty & color_bits;
        if (dirty & (1u << 4)) {
            const auto storage = proctex_color_luts.Append(256 * sizeof(Common::Vec4f));
            auto* colors = static_cast<Common::Vec4f*>(storage.data);
            for (unsigned i = 0; i < 256; ++i) colors[i] = pica.proctex.color_table[i].ToVector() / 255.f;
            fs_data.proctex_lut_offset = proctex_color_luts.TexelOffset(storage);
        }
        if (dirty & (1u << 5)) {
            const auto storage = proctex_color_luts.Append(256 * sizeof(Common::Vec4f));
            auto* colors = static_cast<Common::Vec4f*>(storage.data);
            for (unsigned i = 0; i < 256; ++i) colors[i] = pica.proctex.color_diff_table[i].ToVector() / 255.f;
            fs_data.proctex_diff_lut_offset = proctex_color_luts.TexelOffset(storage);
        }
        lut_textures[2] = proctex_color_luts.Texture();
        pica.proctex.table_dirty &= value_bits;
    }
    for (NSUInteger i = 0; i < lut_textures.size(); ++i)
        if (fragment.texture_mask.test(i)) bindings.push_back({lut_textures[i], i});
}

void RasterizerMetal::BindTextures(const Framebuffer& framebuffer, const CompiledFragmentShader& fragment,
                                  VideoCore::SurfaceId normal_source, std::vector<TextureBinding>& bindings,
                                  std::vector<SamplerBinding>& samplers) {
    using Config = Pica::TexturingRegs::TextureConfig;
    using Face = Pica::TexturingRegs::CubeFace;
    const auto units = regs.texturing.GetTextures();
    if (normal_source && fragment.texture_mask.test(13)) {
        const auto normal = cache.GetSurface(normal_source).Image(SurfaceImage::Normal);
        bindings.push_back({normal, 13});
        samplers.push_back({cache.GetSampler(units[0].config).Handle(), 9});
    }
    for (unsigned unit = 0; unit < units.size(); ++unit) {
        const auto& texture = units[unit];
        const NSUInteger image_index = unit ? unit + 8 : 3;
        const NSUInteger sampler_index = unit ? unit + 5 : 0;
        const unsigned count = unit == 0 && texture.config.type == Config::ShadowCube ? 6 : 1;
        bool consumed{};
        for (unsigned face = 0; face < count; ++face)
            consumed |= fragment.texture_mask.test(image_index + face);
        if (!consumed) continue;
        if (!texture.enabled) {
            const bool cube = unit == 0 && texture.config.type == Config::TextureCube;
            auto& surface = cache.GetSurface(cube ? VideoCore::NULL_SURFACE_CUBE_ID : VideoCore::NULL_SURFACE_ID);
            auto image = surface.ImageView();
            if (unit == 0 && (texture.config.type == Config::Shadow2D || texture.config.type == Config::ShadowCube)) image = surface.StorageView();
            for (unsigned face = 0; face < count; ++face) {
                if (!fragment.texture_mask.test(image_index + face)) continue;
                bindings.push_back({image, image_index + face});
                samplers.push_back({cache.GetSampler(VideoCore::NULL_SAMPLER_ID).Handle(), sampler_index + face});
            }
            continue;
        }
        const auto sampler = cache.GetSampler(texture.config).Handle();
        if (unit == 0 && texture.config.type == Config::ShadowCube) {
            auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config, texture.format);
            for (unsigned face = 0; face < 6; ++face) {
                if (!fragment.texture_mask.test(3 + face)) continue;
                info.physical_address = regs.texturing.GetCubePhysicalAddress(Face(face));
                auto& surface = cache.GetSurface(cache.GetTextureSurface(info));
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                bindings.push_back({surface.StorageView(), 3 + face});
                samplers.push_back({sampler, face});
            }
            continue;
        }
        if (unit == 0 && texture.config.type == Config::TextureCube) {
            const VideoCore::TextureCubeConfig cube{
                regs.texturing.GetCubePhysicalAddress(Face::PositiveX), regs.texturing.GetCubePhysicalAddress(Face::NegativeX),
                regs.texturing.GetCubePhysicalAddress(Face::PositiveY), regs.texturing.GetCubePhysicalAddress(Face::NegativeY),
                regs.texturing.GetCubePhysicalAddress(Face::PositiveZ), regs.texturing.GetCubePhysicalAddress(Face::NegativeZ),
                texture.config.width.Value(), texture.config.lod.max_level + 1, texture.format};
            bindings.push_back({cache.GetTextureCube(cube).ImageView(), 3});
        } else {
            auto& surface = unit == 0 && normal_source ? cache.GetSurface(normal_source) :
                                                       cache.GetTextureSurface(texture);
            id<MTLTexture> image;
            if (unit == 0 && texture.config.type == Config::Shadow2D) {
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                image = surface.StorageView();
            } else {
                // Framebuffer views may be separate objects. Compare the backing images.
                const bool feedback = framebuffer.color_id && surface.Image() == cache.GetSurface(framebuffer.color_id).Image();
                image = feedback ? surface.CopyImageView() : surface.ImageView();
            }
            bindings.push_back({image, image_index});
        }
        samplers.push_back({sampler, sampler_index});
    }
    if (!fragment.texture_mask.test(12)) return;
    auto utility = cache.GetSurface(VideoCore::NULL_SURFACE_ID).StorageView();
    if (framebuffer.shadow_rendering) {
        utility = framebuffer.ImageView(VideoCore::SurfaceType::Color);
    } else if (regs.lighting.config0.enable_shadow) {
        const auto unit = regs.lighting.config0.shadow_selector.Value();
        if (unit >= units.size()) throw std::invalid_argument("Invalid PICA lighting shadow texture unit");
        if (units[unit].enabled) utility = cache.GetTextureSurface(units[unit]).StorageView();
    }
    bindings.push_back({utility, 12});
}

TriangleTopology RasterizerMetal::Topology() const {
    switch (regs.pipeline.triangle_topology.Value()) {
    case Pica::PipelineRegs::TriangleTopology::List: return TriangleTopology::List;
    case Pica::PipelineRegs::TriangleTopology::Strip: return TriangleTopology::Strip;
    case Pica::PipelineRegs::TriangleTopology::Fan: return TriangleTopology::Fan;
    case Pica::PipelineRegs::TriangleTopology::Shader: return TriangleTopology::List;
    default: throw std::invalid_argument("Invalid native PICA primitive topology");
    }
}

void RasterizerMetal::ResetMetalPrimitiveStream() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    RetireGeometry();primitive_stream.Reset();
}
void RasterizerMetal::WriteMetalGSUniforms(u32 first,u32 count,u16 bool_mask,u8 integer_mask) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    RetireGeometry();
    if (first>96 || count>96-first || integer_mask>15)
        throw std::invalid_argument("Invalid native GS uniform write range");
    if (!gs_uniforms_seeded) {
        geometry_program.UpdateUniforms(geometry_input,MachineUniforms(pica.gs_setup));gs_uniforms_seeded=true;
    } else {
        std::array<std::uint32_t,3> mask{};
        for (unsigned i=first;i<first+count;++i) mask[i/32]|=1u<<(i%32);
        geometry_program.UpdateUniforms(geometry_input,MachineUniforms(pica.gs_setup,first,count,bool_mask,integer_mask),
            mask,bool_mask,integer_mask);
    }
}
GeometryInputConfig RasterizerMetal::GeometryConfig() const {
    if (regs.pipeline.use_gs!=Pica::PipelineRegs::UseGS::Yes || regs.gs.shader_mode!=Pica::ShaderRegs::ShaderMode::GS ||
        regs.pipeline.vs_outmap_total_minus_1_a!=regs.pipeline.vs_outmap_total_minus_1_b)
        throw std::invalid_argument("Invalid native geometry shader mode/output counts");
    GeometryInputConfig config{};
    config.vs_output_mask=regs.vs.output_mask.Value();
    config.vs_output_count=regs.pipeline.vs_outmap_total_minus_1_a+1;
    switch(regs.pipeline.gs_config.mode.Value()) {
    case Pica::PipelineRegs::GSMode::Point:
        if(regs.pipeline.variable_primitive || regs.gs.input_to_uniform)
            throw std::invalid_argument("Point GS requires register inputs and fixed grouping");
        config.mode=GeometryInputMode::Point;config.input_count=regs.gs.max_input_attribute_index+1;
        for(unsigned i=0;i<config.input_count;++i) config.input_registers[i]=regs.gs.GetRegisterForAttribute(i);
        break;
    case Pica::PipelineRegs::GSMode::FixedPrimitive:
        if(regs.pipeline.variable_primitive || regs.gs.input_to_uniform!=1 ||
           regs.pipeline.gs_config.stride_minus_1+1!=config.vs_output_count)
            throw std::invalid_argument("Invalid fixed GS uniform input stride/mode");
        config.mode=GeometryInputMode::Fixed;config.uniform_start=regs.pipeline.gs_config.start_index;
        config.fixed_vertex_count=regs.pipeline.gs_config.fixed_vertex_num_minus_1+1;
        break;
    case Pica::PipelineRegs::GSMode::VariablePrimitive:
        if(!regs.pipeline.variable_primitive || regs.gs.input_to_uniform!=1 || regs.pipeline.variable_vertex_main_num_minus_1>=96)
            throw std::invalid_argument("Invalid variable GS input mode/main count");
        config.mode=GeometryInputMode::Variable;config.variable_main_count=regs.pipeline.variable_vertex_main_num_minus_1+1;
        break;
    default:throw std::invalid_argument("Unknown native GS input mode");
    }
    return config;
}
void RasterizerMetal::SubmitGeometryVertex(const BufferSlice& raw,const ShaderMachineProgramView& program) {
    RetireGeometry(true);
    if(!geometry_program.SubmitVertex(geometry_input,raw)) return;
    ShaderMachineResult result{};
    for(unsigned capacity=192;;capacity=std::min(capacity*2,1000000u)) {
        try { result=geometry_program.Invoke(geometry_input,machine,gs_unit,program,capacity);break; }
        catch(const ShaderMachineError& error) {
            // Failed execution leaves the unit and ready input group unchanged.
            if(error.Code()!=4 || capacity==1000000) throw;
        }
    }
    DrawGeometryResult(result);
}
void RasterizerMetal::DrawGeometryResult(const ShaderMachineResult& result) {
    pica.gs_setup.uniforms.b[15]=true; // Known invocation bookkeeping; uniform data stays on GPU.
    SyncDrawUniforms();
    GeometryEmissionConfig config{};
    config.attribute_count=regs.rasterizer.vs_output_total.Value();
    for(unsigned i=0;i<7;++i) config.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
    for(unsigned c=0;c<4;++c) config.clip_coefficient[c]=vs_data.clip_coef[c];
    config.flip_viewport=vs_data.flip_viewport;config.enable_clip=vs_data.enable_clip1;
    config.honor_winding=0;config.fix_quaternions=0; // The shared primitive stream applies these after grouping.
    const auto draw_state=BuildPicaDrawState(regs);
    if (result.emission_count && Topology()==TriangleTopology::List &&
        (primitive_stream.topology!=TriangleTopology::List || !primitive_stream.Pending()) &&
        !draw_state.shadow && !draw_state.gas_accumulation) {
        config.honor_winding=regs.pipeline.triangle_topology==Pica::PipelineRegs::TriangleTopology::Shader;
        config.fix_quaternions=regs.lighting.disable==0;
        const auto assembled=machine.ConvertEmissionList(result,config,primitive_stream);
        DrawNativeVertices(assembled);
        if (!gs_list_draw_logged) {
            LOG_INFO(Render,"Metal4 GS complete-list raster active: one GPU conversion/history pass and draw per invocation, first_vertices={}",result.emission_count);
            gs_list_draw_logged=true;
        }
        return;
    }
    for(unsigned first=0;first<result.emission_count;first+=3) {
        auto primitive=result;primitive.emission_count=3;
        primitive.emissions.offset+=NSUInteger(first)*sizeof(ShaderMachineEmission);
        primitive.emissions.size=3*sizeof(ShaderMachineEmission);primitive.emissions.data=nullptr;
        const auto captured=machine.ConvertEmissions(primitive,config);
        BufferSlice winding{};
        if(regs.pipeline.triangle_topology==Pica::PipelineRegs::TriangleTopology::Shader)
            winding={primitive.emissions.buffer,primitive.emissions.offset+256,4,nullptr,{}};
        const auto assembled=geometry.Append(primitive_stream,captured.storage,3,Topology(),3,0,0,{},0,
                                             regs.lighting.disable==0,winding);
        DrawNativeVertices(assembled);
    }
}
void RasterizerMetal::SubmitMachineVertex(const BufferSlice& output,std::uint32_t mask) {
    SyncDrawUniforms();
    GeometryEmissionConfig config{};
    config.attribute_count=std::min(unsigned(regs.rasterizer.vs_output_total.Value()),unsigned(std::popcount(mask)));
    for(unsigned i=0;i<7;++i) config.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
    for(unsigned c=0;c<4;++c) config.clip_coefficient[c]=vs_data.clip_coef[c];
    config.flip_viewport=vs_data.flip_viewport;config.enable_clip=vs_data.enable_clip1;config.honor_winding=0;
    const auto captured=machine.CaptureVertex(output,mask,config);
    const auto assembled=geometry.Append(primitive_stream,captured,1,Topology(),1,0,0,{},0,regs.lighting.disable==0);
    DrawNativeVertices(assembled);
}
bool RasterizerMetal::DrawGeometryBatch(bool indexed) { return DrawMachineBatch(indexed,true); }
bool RasterizerMetal::DrawMachineBatch(bool indexed,bool use_geometry) {
    RetireGeometry();
    const auto count=regs.pipeline.num_vertices;
    const auto config=use_geometry ? GeometryConfig():GeometryInputConfig{};
    if(use_geometry && config.mode==GeometryInputMode::Variable && !indexed)
        throw std::invalid_argument("Variable GS requires count indices");
    if(!indexed && std::uint64_t(regs.pipeline.vertex_offset)+count>std::uint64_t(UINT32_MAX)+1)
        throw std::invalid_argument("Native GS vertex range overflows");
    if(use_geometry) {
        if(!gs_uniforms_seeded) WriteMetalGSUniforms(0,0,0,0);
        geometry_program.Configure(geometry_input,config);
    }
    const auto vs_program=MachineProgram(pica.vs_setup,regs.vs,false);
    std::optional<ShaderMachineProgramView> gs_program;
    if(use_geometry) gs_program.emplace(MachineProgram(pica.gs_setup,regs.gs,true));
    const auto uniforms=MachineUniforms(pica.vs_setup);
    vs_unit.Reset(); // Existing batch contract: fresh VS unit, then retained temporary/output state for misses.
    const auto base=std::uint64_t(regs.pipeline.vertex_attributes.GetPhysicalBaseAddress());
    BufferSlice indices{};unsigned index_size{};
    if(indexed) {
        index_size=regs.pipeline.index_array.format ? 2:1;
        indices=UploadGuestBytes(base+regs.pipeline.index_array.offset,std::uint64_t(count)*index_size);
    }
    auto defaults=DefaultVertexRegisters();
    for(unsigned i=0;i<16;++i) for(unsigned c=0;c<4;++c) defaults[i][c]=pica.input_default_attributes[i][c].ToFloat32();
    const auto layout=BuildPicaVertexLayout(regs,defaults);
    if (!use_geometry) {
        u32 first=regs.pipeline.vertex_offset,last=first+count-1;
        if (indexed) {
            first=UINT32_MAX;last=0;
            const auto* bytes=static_cast<const u8*>(indices.data);
            for(u32 element=0;element<count;++element) {
                const auto offset=NSUInteger(element)*index_size;
                const auto vertex=u32(bytes[offset])|(index_size==2 ? u32(bytes[offset+1])<<8:0);
                first=std::min(first,vertex);last=std::max(last,vertex);
            }
        }
        const auto vertex_count=u32(std::uint64_t(last)-first+1);
        const auto required=RequiredPicaStateInputBytes(layout,vertex_count);
        std::array<BufferSlice,12> inputs{};
        for(unsigned loader=0;loader<inputs.size();++loader) {
            if(!required[loader]) continue;
            const auto stride=regs.pipeline.vertex_attributes.attribute_loaders[loader].byte_count.Value();
            inputs[loader]=UploadGuestBytes(base+layout.loader_offsets[loader]+std::uint64_t(first)*stride,required[loader],true);
        }
        SyncDrawUniforms();
        GeometryEmissionConfig capture{};
        capture.attribute_count=std::min(unsigned(regs.rasterizer.vs_output_total.Value()),unsigned(std::popcount(vs_program.output_mask)));
        for(unsigned i=0;i<7;++i) capture.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
        for(unsigned c=0;c<4;++c) capture.clip_coefficient[c]=vs_data.clip_coef[c];
        capture.flip_viewport=vs_data.flip_viewport;capture.enable_clip=vs_data.enable_clip1;capture.honor_winding=0;
        const bool tried_native_mesh = Topology() == TriangleTopology::List && count % 3 == 0 &&
            (primitive_stream.topology != TriangleTopology::List || primitive_stream.pending == 0);
        if (tried_native_mesh) {
            if(auto selected=machine.SelectNativeReset(vs_program,uniforms,capture)) {
                const auto mesh = PrepareNativeMesh(inputs, layout, indices, index_size, vertex_count, count,
                                                   first, std::move(*selected));
                DrawNativeVertices({}, &mesh);
                return true;
            }
        }
        const auto assembled=[&] {
            // Guest bytes are immutable uploads. VS/fetch/primitive histories
            // are compute-only; output leases reserve fresh disjoint ranges.
            Scheduler::BufferComputeScope buffers{scheduler};
            // A successful selection already returned through the mesh draw.
            // Otherwise its negative result applies to these unchanged local
            // program/control/capture values; other topologies still prove it.
            if(!tried_native_mesh && machine.CanCaptureNativeReset(vs_program,uniforms,capture)) {
                const auto native=shaders.Vertex(regs,pica.vs_setup,true);
                const auto bindings=UploadUniforms(native->uniform_mask);
                const auto captured=machine.RunFetchedNativeResetAndCapture(fetch,vertex_input_state,inputs,layout,
                    vs_unit,vs_program,uniforms,capture,geometry,*native,bindings.Bindings(),
                    vertex_count,count,first,first,indices,index_size);
                return geometry.Append(primitive_stream,captured,vertex_count,Topology(),count,first,first,
                    indices,index_size,regs.lighting.disable==0);
            }
            // Keep VM/capture and primitive append in one compute encoder.
            // BeginCompute supplies the device-visible dependency; allocations,
            // submission and the final render consumer still close it as needed.
            Scheduler::ComputeBatchScope compute_batch{scheduler};
            const auto batch=machine.RunOwnedFetchedVertexBatchAndCapture(fetch,vertex_input_state,inputs,layout,
                vs_unit,vs_program,uniforms,capture,vertex_count,count,first,first,indices,index_size,1000000,true);
            return geometry.Append(primitive_stream,batch.captured,count,Topology(),count,0,0,{},0,regs.lighting.disable==0);
        }();
        DrawNativeVertices(assembled);
        return true;
    }
    const auto required=RequiredPicaStateInputBytes(layout,1);
    const auto draw_state=BuildPicaDrawState(regs);
    const bool indirect=Topology()==TriangleTopology::List && primitive_stream.topology==TriangleTopology::List &&
        !primitive_stream.Pending() && !draw_state.shadow && !draw_state.gas_accumulation;
    if(!indirect) RetireGeometry(true);
    struct CacheEntry { u32 id{};bool valid{};BufferSlice output; };
    std::array<CacheEntry,64> cache_entries{};unsigned replacement{};
    std::vector<GeometryQueuedInvocation> pending;
    pending.reserve(32);
    BufferSlice abort{};
    const auto complete=[&] {
        if(pending.empty()) return;
        geometry_program.CompleteQueued(geometry_input,machine,gs_unit,pending,
            [&](const auto& result) { DrawGeometryResult(result); });
        pending.clear();abort={};
    };
    // Preserve the former draw-before-next-guest-read ordering when VS bytes
    // alias either framebuffer. Four bytes per aligned guest texel is a
    // conservative bound for every supported color/depth packing.
    const auto& fb=regs.framebuffer.framebuffer;
    const auto framebuffer_bytes=std::uint64_t((fb.GetWidth()+31)&~31u)*((fb.GetHeight()+31)&~31u)*4;
    const auto aliases_framebuffer=[&](std::uint64_t address,std::uint64_t bytes) {
        const auto overlaps=[&](std::uint64_t target) {
            return address<target+framebuffer_bytes && target<address+bytes;
        };
        return framebuffer_bytes && (overlaps(fb.GetColorBufferPhysicalAddress()) || overlaps(fb.GetDepthBufferPhysicalAddress()));
    };
    // Only disjoint texture reads can be moved past preceding GS work. A
    // exact same mip interval is used by the texture cache;
    // cube/shadow modes retain the existing immediately ordered native path.
    const auto gs_storage_bound=use_geometry ? ShaderMachineRuntime::LinearGeometryStorageBound(*gs_program):std::nullopt;
    const bool vs_completed=use_geometry && indirect && gs_storage_bound &&
        machine.ProvesGeometryVertexCompletion(vs_program,uniforms);
    bool batch_draws=vs_completed;
    for(unsigned unit=0;batch_draws && unit<3;++unit) {
        const auto texture=regs.texturing.GetTextures()[unit];
        if(!texture.enabled) continue;
        using Config=Pica::TexturingRegs::TextureConfig;
        if(unit==0 && texture.config.type!=Config::Texture2D) {batch_draws=false;break;}
        const auto info=Pica::Texture::TextureInfo::FromPicaRegister(texture.config,texture.format);
        if(info.width<8 || info.height<8 || info.width>1024 || info.height>1024 ||
            info.width%8 || info.height%8 || info.is_shadow_source || unsigned(info.format)>13) {
            batch_draws=false;break;
        }
        VideoCore::SurfaceParams texture_range;
        texture_range.addr=info.physical_address;texture_range.width=info.width;texture_range.height=info.height;
        texture_range.levels=VideoCore::MipLevels(info.width,info.height,texture.config.lod.max_level);
        texture_range.is_tiled=true;texture_range.pixel_format=VideoCore::PixelFormatFromTextureFormat(info.format);
        texture_range.UpdateParams();
        if(texture_range.end<texture_range.addr || aliases_framebuffer(texture_range.addr,texture_range.size))
            batch_draws=false;
    }
    if(use_geometry && !(batch_draws ? gs_batch_logged:gs_batch_ineligible_logged)) {
        LOG_INFO(Render,"Metal4 GS grouping: enabled={}, storage_bound={}, proved_vs_completion={}, mode={}, vertices={}",
            batch_draws,gs_storage_bound.value_or(0),vs_completed,unsigned(config.mode),count);
        (batch_draws ? gs_batch_logged:gs_batch_ineligible_logged)=true;
    }
    std::vector<DeferredGeometry*> geometry_draws;
    if(batch_draws)geometry_draws.reserve(64);
    const auto flush_geometry_draws=[&] {
        try {
            for(auto* item:geometry_draws) {
                DrawNativeVertices(item->vertices);
                item->tick=std::max(item->vertices.indirect->tick,scheduler.LastUseTick(item->vertices.storage.buffer));item->ready=true;
            }
        } catch(...) {
            for(auto* item:geometry_draws) {item->tick=std::max(item->vertices.indirect->tick,scheduler.LastUseTick(item->vertices.storage.buffer));item->ready=true;}
            geometry_draws.clear();throw;
        }
        geometry_draws.clear();
    };
    // No render-produced GS scratch is reused by this proved batch. Rendering,
    // submission, alias transfers and each bounded flush still close the pass.
    std::optional<Scheduler::ComputeBatchScope> geometry_compute;
    if(batch_draws)geometry_compute.emplace(scheduler);
    BufferSlice geometry_vertices{};
    // Snapshot each loader once, like the existing No-GS batch. Count words in
    // variable GS streams are not vertex IDs. Aliased loaders retain the exact
    // draw-before-next-read path below, including its FIFO and cache flushes.
    if(batch_draws && config.mode!=GeometryInputMode::Variable && count>=3) {
        u32 first=regs.pipeline.vertex_offset,last=first+count-1;
        if(indexed) {
            first=UINT32_MAX;last=0;
            const auto* bytes=static_cast<const u8*>(indices.data);
            for(u32 element=0;element<count;++element) {
                const auto offset=NSUInteger(element)*index_size;
                const auto vertex=u32(bytes[offset])|(index_size==2 ? u32(bytes[offset+1])<<8:0);
                first=std::min(first,vertex);last=std::max(last,vertex);
            }
        }
        const auto vertices=u32(std::uint64_t(last)-first+1);
        const auto bytes=RequiredPicaStateInputBytes(layout,vertices);
        bool disjoint=true;
        for(unsigned loader=0;loader<bytes.size();++loader) {
            if(!bytes[loader]) continue;
            const auto stride=regs.pipeline.vertex_attributes.attribute_loaders[loader].byte_count.Value();
            const auto address=base+layout.loader_offsets[loader]+std::uint64_t(first)*stride;
            // Preflight without uploads: invalid guest ranges keep their former
            // scalar error/prefix ordering instead of failing earlier in a batch.
            if(address>UINT32_MAX || bytes[loader]>UINT32_MAX ||
               address+bytes[loader]>std::uint64_t(UINT32_MAX)+1 ||
               aliases_framebuffer(address,bytes[loader])) {disjoint=false;break;}
            const auto source=memory.GetPhysicalRef(PAddr(address));
            if(!source || source.GetSize()<bytes[loader]) {disjoint=false;break;}
        }
        if(disjoint) {
            std::array<BufferSlice,12> inputs{};
            for(unsigned loader=0;loader<inputs.size();++loader) {
                if(!bytes[loader]) continue;
                const auto stride=regs.pipeline.vertex_attributes.attribute_loaders[loader].byte_count.Value();
                inputs[loader]=UploadGuestBytes(base+layout.loader_offsets[loader]+std::uint64_t(first)*stride,
                                               bytes[loader],true);
            }
            geometry_vertices=machine.RunFetchedGeometryVertices(fetch,vertex_input_state,inputs,layout,
                vs_unit,vs_program,uniforms,vertices,count,first,first,indices,index_size);
            if(!gs_vertex_batch_logged) {
                LOG_INFO(Render,"Metal4 GS vertex batch: raw GPU outputs, elements={}, vertices={}, indexed={}, mode={}",
                    count,vertices,indexed,unsigned(config.mode));
                gs_vertex_batch_logged=true;
            }
        }
    }
    try {
        for(unsigned element=0;element<count;++element) {
            // Fetch, scalar VS, GS inputs and invocation share a compute pass until
            // the native draw consumes them. Render and submission boundaries keep
            // their full queue dependencies, including scratch reuse after draws.
            std::optional<Scheduler::ComputeBatchScope> compute_batch;
            if(use_geometry) compute_batch.emplace(scheduler);
            u32 vertex=regs.pipeline.vertex_offset+element;
            if(indexed) {
                const auto* bytes=static_cast<const u8*>(indices.data)+NSUInteger(element)*index_size;
                vertex=u32(bytes[0])|(index_size==2 ? u32(bytes[1])<<8:0);
                if(use_geometry && geometry_input.NeedsCount()) { geometry_program.BeginVariable(geometry_input,vertex);continue; }
            }
            BufferSlice output{};
            if(geometry_vertices.buffer) {
                output={geometry_vertices.buffer,geometry_vertices.offset+NSUInteger(element)*sizeof(VertexRegisters),
                    sizeof(VertexRegisters),nullptr,geometry_vertices.lease};
            } else if(indexed) for(const auto& entry:cache_entries) if(entry.valid && entry.id==vertex) { output=entry.output;break; }
            if(!output.buffer) {
                std::array<BufferSlice,12> inputs{};
                for(unsigned loader=0;loader<12;++loader) {
                    if(!required[loader]) continue;
                    const auto stride=regs.pipeline.vertex_attributes.attribute_loaders[loader].byte_count.Value();
                    if(aliases_framebuffer(base+layout.loader_offsets[loader]+std::uint64_t(vertex)*stride,required[loader])) {
                        flush_geometry_draws();complete();break;
                    }
                }
                for(unsigned loader=0;loader<12;++loader) {
                    if(!required[loader]) continue;
                    const auto stride=regs.pipeline.vertex_attributes.attribute_loaders[loader].byte_count.Value();
                    inputs[loader]=UploadGuestBytes(base+layout.loader_offsets[loader]+std::uint64_t(vertex)*stride,required[loader]);
                }
                const auto input=fetch.FetchPica(vertex_input_state,inputs,layout,1,1,0,0);
                // The next GS input/dispatch consumes this GPU range in queue order.
                // Only GS emission counts require CPU completion, not every VS value.
                output=machine.RunVertexDeferred(vs_unit,vs_program,uniforms,input);
                if(indexed) {
                    cache_entries[replacement]={vertex,true,output};replacement=(replacement+1)%64;
                }
            }
            if(use_geometry) {
                if(geometry_program.SubmitVertex(geometry_input,output)) {
                    if(indirect) {
                        QueueGeometryDraw(*gs_program,batch_draws ? &geometry_draws:nullptr);
                        if(geometry_draws.size()==64)flush_geometry_draws();
                    }
                    else {
                        if(!abort.buffer) abort=machine.BeginGeometryBatch();
                        pending.push_back(geometry_program.InvokeQueued(geometry_input,machine,gs_unit,*gs_program,abort));
                        if(pending.size()==32) complete();
                    }
                }
            }
            else SubmitMachineVertex(output,vs_program.output_mask);
        }
    } catch(...) {
        flush_geometry_draws();throw;
    }
    flush_geometry_draws();
    complete();
    return true;
}

bool RasterizerMetal::AccelerateImmediateVertex() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Draw};
    @autoreleasepool {
        RetireGeometry();
        VertexRegisters attributes{};
        for (unsigned attribute = 0; attribute <= regs.pipeline.max_input_attrib_index; ++attribute) {
            for (unsigned channel = 0; channel < 4; ++channel)
                attributes[attribute][channel] = pica.immediate.input_vertex[attribute][channel].ToFloat32();
        }
        const auto layout = BuildPicaImmediateLayout(regs, attributes);
        if(regs.pipeline.use_gs==Pica::PipelineRegs::UseGS::No && NeedsMachine(pica.vs_setup,regs.vs)) {
            vs_unit.Reset();
            SyncDrawUniforms();
            const auto program=MachineProgram(pica.vs_setup,regs.vs,false);
            GeometryEmissionConfig capture{};
            capture.attribute_count=std::min(unsigned(regs.rasterizer.vs_output_total.Value()),unsigned(std::popcount(program.output_mask)));
            for(unsigned i=0;i<7;++i) capture.semantics[i]=regs.rasterizer.vs_output_attributes[i].raw;
            for(unsigned c=0;c<4;++c) capture.clip_coefficient[c]=vs_data.clip_coef[c];
            capture.flip_viewport=vs_data.flip_viewport;capture.enable_clip=vs_data.enable_clip1;capture.honor_winding=0;
            const auto assembled=[&] {
                Scheduler::BufferComputeScope buffers{scheduler};
                const auto values=MachineUniforms(pica.vs_setup);
                if(machine.CanCaptureNativeReset(program,values,capture)) {
                    const auto native=shaders.Vertex(regs,pica.vs_setup,true);
                    const auto bindings=UploadUniforms(native->uniform_mask);
                    const auto captured=machine.RunFetchedNativeResetAndCapture(fetch,vertex_input_state,{},layout,
                        vs_unit,program,values,capture,geometry,*native,bindings.Bindings(),1,1,0,0);
                    return geometry.Append(primitive_stream,captured,1,Topology(),1,0,0,{},0,regs.lighting.disable==0);
                }
                const auto batch=machine.RunFetchedVertexBatchAndCapture(fetch,vertex_input_state,{},layout,
                    vs_unit,program,values,capture,1,1,0,0,{},0,1000000,true);
                return geometry.Append(primitive_stream,batch.captured,1,Topology(),1,0,0,{},0,regs.lighting.disable==0);
            }();
            DrawNativeVertices(assembled);
            return true;
        }
        if(regs.pipeline.use_gs!=Pica::PipelineRegs::UseGS::No) {
            const auto config=GeometryConfig();
            if(config.mode==GeometryInputMode::Variable) throw std::invalid_argument("Variable GS does not accept immediate vertices");
            if(!gs_uniforms_seeded) WriteMetalGSUniforms(0,0,0,0);
            geometry_program.Configure(geometry_input,config);
            vs_unit.Reset(); // Immediate VS uses a fresh unit in the existing PICA contract.
            const auto input=fetch.FetchPica(vertex_input_state,{},layout,1,1,0,0);
            const auto raw=machine.RunVertexDeferred(vs_unit,MachineProgram(pica.vs_setup,regs.vs,false),MachineUniforms(pica.vs_setup),input);
            SubmitGeometryVertex(raw,MachineProgram(pica.gs_setup,regs.gs,true));
            return true;
        }
        SyncDrawUniforms();
        const auto program = shaders.Vertex(regs, pica.vs_setup);
        const auto uniforms = UploadUniforms(program->uniform_mask);
        const auto input = fetch.FetchPica(vertex_input_state, {}, layout, 1, 1, 0, 0);
        const auto captured = geometry.Capture(*program, input, 1, 0,
                                              uniforms.Bindings());
        const auto assembled = geometry.Append(primitive_stream, captured, 1, Topology(), 1, 0, 0, {}, 0,
                                               regs.lighting.disable == 0);
        DrawNativeVertices(assembled);
        return true;
    }
}

bool RasterizerMetal::AccelerateDrawBatch(bool indexed) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Draw};
    @autoreleasepool {
        RetireGeometry();
        const auto count = regs.pipeline.num_vertices;
        if (!count) return true;
        if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) return DrawGeometryBatch(indexed);
        if(NeedsMachine(pica.vs_setup,regs.vs)) return DrawMachineBatch(indexed,false);
        const auto topology = Topology();
        BufferSlice indices{};
        u32 first = regs.pipeline.vertex_offset, last;
        u32 index_size = 0;
        const auto base = std::uint64_t(regs.pipeline.vertex_attributes.GetPhysicalBaseAddress());
        if (indexed) {
            index_size = regs.pipeline.index_array.format ? 2 : 1;
            indices = UploadGuestBytes(base + regs.pipeline.index_array.offset, std::uint64_t(count) * index_size, true);
            first = 65535;
            last = 0;
            const auto* bytes = static_cast<const u8*>(indices.data);
            for (u32 i = 0; i < count; ++i) {
                const auto offset = std::size_t(i) * index_size;
                const auto index = u32(bytes[offset]) | (index_size == 2 ? u32(bytes[offset + 1]) << 8 : 0);
                first = std::min(first, index);
                last = std::max(last, index);
            }
        } else {
            if (std::uint64_t(first) + count > std::uint64_t(std::numeric_limits<u32>::max()) + 1)
                throw std::invalid_argument("PICA vertex range overflows");
            last = first + count - 1;
        }
        const auto vertex_count64 = std::uint64_t(last) - first + 1;
        if (vertex_count64 > std::numeric_limits<u32>::max()) throw std::length_error("PICA vertex count exceeds native storage");
        const auto vertex_count = u32(vertex_count64);
        auto defaults = DefaultVertexRegisters();
        for (unsigned attribute = 0; attribute < defaults.size(); ++attribute)
            for (unsigned channel = 0; channel < 4; ++channel)
                defaults[attribute][channel] = pica.input_default_attributes[attribute][channel].ToFloat32();
        const auto layout = BuildPicaVertexLayout(regs, defaults);
        const auto required = RequiredPicaStateInputBytes(layout, vertex_count);
        std::array<BufferSlice, 12> inputs{};
        for (unsigned loader = 0; loader < required.size(); ++loader) {
            if (!required[loader]) continue;
            const auto& config = regs.pipeline.vertex_attributes.attribute_loaders[loader];
            inputs[loader] = UploadGuestBytes(base + layout.loader_offsets[loader] + std::uint64_t(first) * config.byte_count.Value(), required[loader], true);
        }
        SyncDrawUniforms();
        const auto program = shaders.Vertex(regs, pica.vs_setup);
        const auto uniforms = UploadUniforms(program->uniform_mask);
        const auto canonical = fetch.FetchPica(vertex_input_state, inputs, layout, vertex_count, count,
                                              first, first, indices, index_size);
        const auto captured = geometry.Capture(*program, canonical, vertex_count, first,
                                              uniforms.Bindings());
        const auto assembled = geometry.Append(primitive_stream, captured, vertex_count, topology, count,
                                               first, first, indices, index_size, regs.lighting.disable == 0);
        DrawNativeVertices(assembled);
        return true;
    }
}

NativeMeshDraw RasterizerMetal::PrepareNativeMesh(std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout, const BufferSlice& indices, std::uint32_t index_size,
    std::uint32_t vertex_count, std::uint32_t element_count, std::uint32_t first_vertex,
    ShaderMachineRuntime::NativeResetSelection&& selection) {
    NativeMeshDraw result;
    result.program = shaders.MeshVertex(regs, machine.NativeProgram(selection));
    auto prepared = input_references.PrepareNative(vertex_input_state, inputs, layout, vertex_count,
        element_count, first_vertex, first_vertex, indices, index_size);
    const auto& snapshot = prepared.snapshot;
    result.inputs = snapshot;
    result.references = std::move(prepared.metadata);
    result.capture = uploads.Allocate(16);
    const std::array<std::uint32_t, 4> capture_params{vertex_count, first_vertex, 0, 0};
    std::memcpy(result.capture.data, capture_params.data(), 16);
    result.vertex_count = vertex_count;
    result.element_count = element_count;
    result.index_size = index_size;
    result.empty = mesh_empty;
    const auto triangles = element_count / 3;
    result.chunks.reserve((std::uint64_t(triangles) + 32767) / 32768);
    for (std::uint32_t first = 0; first < triangles;) {
        const auto count = std::min(32768u, triangles - first);
        const std::array<std::uint32_t, 12> params{element_count, index_size, first_vertex, first_vertex,
            0, 0, 1, regs.lighting.disable == 0 ? 1u : 0u, triangles, 0, 1, first};
        auto storage = uploads.Allocate(sizeof(params));
        std::memcpy(storage.data, params.data(), sizeof(params));
        result.chunks.push_back({storage, (count + 31) / 32});
        first += count;
    }
    // Publication matches the previous fetch-before-raster contract. All data
    // validation and allocation precede it, and no mutable borrowed input enters
    // this private path: UploadGuestBytes created every slice above.
    machine.DeferNativeReferenceBatch(vs_unit, std::move(selection), snapshot,
        vertex_count, element_count, first_vertex, first_vertex, index_size);
    vertex_input_state.references = std::move(prepared.next);
    vertex_input_state.needs_reset = false;
    primitive_stream.pending = 0;
    primitive_stream.topology = TriangleTopology::List;
    primitive_stream.needs_reset = false;
    primitive_stream.known_zero_history = true;
    primitive_stream.deferred_list = {};
    return result;
}

void RasterizerMetal::DrawNativeVertices(const AssembledVertices& assembled, const NativeMeshDraw* mesh) {
    if (mesh ? !mesh->element_count : !assembled.vertex_count) return;
    SyncDrawUniforms();
    const auto state = BuildPicaDrawState(regs);
    if (!state.use_color && !state.use_depth) return;
    const auto helper = cache.GetFramebufferSurfaces(state.use_color, state.use_depth);
    const auto rect = helper.DrawRect();
    if (!rect.GetArea()) return;
    const auto* framebuffer = helper.Framebuffer();
    fs_data.framebuffer_scale = framebuffer->Scale();
    const auto scissor = helper.Scissor();
    fs_data.scissor_x1 = scissor.left;
    fs_data.scissor_x2 = scissor.right;
    fs_data.scissor_y1 = scissor.bottom;
    fs_data.scissor_y2 = scissor.top;
    std::vector<TextureBinding> bindings;
    std::vector<SamplerBinding> samplers;
    // A replacement normal map can affect lighting even if texture0's color is
    // unused. Resolve that potential consumer before selecting the final shader.
    // Do not infer cache contents from live settings: cache mode changes at TickFrame.
    user_config = {};
    VideoCore::SurfaceId normal_source{};
    const auto texture0 = regs.texturing.GetTextures()[0];
    using TextureConfig = Pica::TexturingRegs::TextureConfig;
    if (!regs.lighting.disable && texture0.enabled &&
        texture0.config.type != TextureConfig::TextureCube &&
        texture0.config.type != TextureConfig::ShadowCube) {
        const auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture0.config, texture0.format);
        normal_source = cache.GetTextureSurface(info,
            VideoCore::MipLevels(info.width, info.height, texture0.config.lod.max_level) - 1);
        auto& surface = cache.GetSurface(normal_source);
        if (texture0.config.type == TextureConfig::Shadow2D)
            surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
        user_config.use_custom_normal.Assign(surface.Image(SurfaceImage::Normal) != nil);
    }
    const auto fragment = shaders.Fragment(Pica::Shader::FSConfig{regs}, user_config, state.color_format,
                                           BuildPicaSamplerConfigs(regs), state.gas_accumulation);
    BindTextures(*framebuffer, *fragment, normal_source, bindings, samplers);
    SyncLUTs(*fragment, bindings);
    const auto base_uniforms = UploadUniforms(fragment->uniform_mask | (mesh ? mesh->program->uniform_mask : 0));
    const auto uniform_bindings = base_uniforms.Bindings();
    std::vector<BufferBinding> uniforms{uniform_bindings.begin(), uniform_bindings.end()};
    if (state.gas_accumulation) {
        const auto depth = framebuffer->ImageView(VideoCore::SurfaceType::Depth);
        if (!depth && ((regs.reg_array[0x126] >> 24) & 3) >= 2)
            throw std::invalid_argument("Native Gas depth attenuation requires a scene depth attachment");
        const auto snapshot = gas_runtime.SceneDepthForDraw(depth, depth ? framebuffer->depth_level : 0,
                                                           ((regs.reg_array[0x126] >> 24) & 3) >= 2);
        bindings.push_back({snapshot, 14});
        samplers.push_back({gas_runtime.DepthSampler(), 10});
        uniforms.push_back(gas_runtime.PrepareFragmentMaximum(gas_feedback));
        gas.feedback_pending = false;
    }
    const auto viewport = helper.Viewport();
    const MTLViewport native_viewport{double(viewport.x), double(viewport.y), double(viewport.width), double(viewport.height), 0, 1};
    const MTLScissorRect native_scissor{rect.left, rect.bottom, rect.GetWidth(), rect.GetHeight()};
    if (mesh) {
        const auto pass = framebuffer->RenderSnapshot(scheduler);
        draws.DrawMesh(state, *fragment, *pass, native_viewport, native_scissor,
                       *mesh, uniforms, bindings, samplers);
        if (!mesh_draw_logged) {
            LOG_INFO(Render, "Metal4 direct mesh raster active: immutable vertex input references, cached immutable render pass, complete triangle lists");
            mesh_draw_logged = true;
        }
    }
    else draws.Draw(state, *fragment, framebuffer->RenderPass(), native_viewport, native_scissor,
                    assembled, uniforms, bindings, samplers);
    if (state.gas_accumulation) gas_maximum_pending = true;
}

void RasterizerMetal::FlushAll() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Coherency};
    cache.FlushAll();
}
void RasterizerMetal::FlushRegion(PAddr addr, u32 size) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Coherency};
    cache.FlushRegion(addr, size);
}
void RasterizerMetal::InvalidateRegion(PAddr addr, u32 size) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Coherency};
    cache.InvalidateRegion(addr, size);
}
void RasterizerMetal::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Coherency};
    cache.FlushRegion(addr, size);
    cache.InvalidateRegion(addr, size);
}
void RasterizerMetal::ClearAll(bool flush) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Coherency};
    cache.ClearAll(flush);
    shaders.Clear();
    draws.Clear();
    uniform_uploads.Clear();
    lighting_luts.Clear();
    proctex_value_luts.Clear();
    proctex_color_luts.Clear();
    lut_textures = {};
    pica.dirty_regs.SetAllDirty();
}
void RasterizerMetal::WriteMetalGasRegisters(u32 reg, std::span<const u32> values, u32 mask) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    gas.Write(reg, values, mask);
    if (reg == 0x125) {
        gas_feedback.Reset();
        gas_maximum_pending = false;
    }
    if (reg == 0x124) {
        // Reflect the FIFO cursor rather than retaining the initial index forever.
        auto& index = pica.regs.internal.reg_array[0x123];
        index = (index & 0xffff0000u) | gas.cursor;
        pica.dirty_regs.Set(0x123);
    }
}
void RasterizerMetal::UpdateMetalGasAccumulatorMaximum() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    const auto reciprocal = gas_runtime.ReadReciprocalMaximum(gas_feedback);
    // The reciprocal already completed the writer. Publish its maximum without
    // a second submission/wait at this CPU-visible GX command boundary.
    SynchronizeMetalGasMaximum();
    gas.feedback_pending = false;
    auto& value = pica.regs.internal.reg_array[0xE5];
    value = (value & 0xffff0000u) | reciprocal;
    pica.dirty_regs.Set(0xE5);
}
void RasterizerMetal::SynchronizeMetalGasMaximum() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    if (!gas_maximum_pending) return;
    const auto maximum = gas_runtime.ReadMaximum(gas_feedback);
    gas.maximum = maximum;
    auto& feedback = pica.regs.internal.reg_array[0x125];
    feedback = (feedback & 0xffff0000u) | maximum;
    pica.dirty_regs.Set(0x125);
    gas_maximum_pending = false;
}
std::vector<u8> RasterizerMetal::SaveMetalState() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    RetireGeometry(true);
    SynchronizeMetalGasMaximum();
    const auto primitive = geometry.SaveStream(primitive_stream);
    const auto input = fetch.SaveState(vertex_input_state);
    const auto vertex_shader=machine.SaveState(vs_unit),geometry_shader=machine.SaveState(gs_unit);
    const auto group=geometry_program.SaveState(geometry_input);
    std::vector<u8> data(StateArchiveBytes);
    constexpr std::array<u8,8> magic{'A','Z','M','4','S','T','A','T'};
    std::copy(magic.begin(),magic.end(),data.begin());
    const auto word = [&](unsigned offset,u32 value) {
        for (unsigned byte=0; byte<4; ++byte) data[offset+byte]=u8(value>>(8*byte));
    };
    word(8,2); word(12,u32(primitive.topology)); word(16,primitive.pending);
    word(20,primitive.needs_reset); word(24,input.needs_reset); word(28,gs_uniforms_seeded);
    std::memcpy(data.data()+32,primitive.history.data(),primitive.history.size());
    std::memcpy(data.data()+32+primitive.history.size(),input.history.data(),input.history.size());
    const auto shader=[&](unsigned offset,const ShaderMachineSnapshot& snapshot) {
        word(offset,snapshot.needs_reset);
        std::memcpy(data.data()+offset+16,snapshot.history.data(),snapshot.history.size());
    };
    shader(LegacyStateBytes,vertex_shader);shader(LegacyStateBytes+ShaderStateBlockBytes,geometry_shader);
    const auto g=unsigned(GeometryStateOffset);
    word(g,u32(group.configured)|(u32(group.initialized)<<1)|(u32(group.awaiting)<<2));
    word(g+4,u32(group.config.mode));word(g+8,group.config.vs_output_mask);word(g+12,group.config.vs_output_count);
    word(g+16,group.config.input_count);word(g+20,group.config.uniform_start);
    word(g+24,group.config.fixed_vertex_count);word(g+28,group.config.variable_main_count);
    word(g+32,group.cursor);word(g+36,group.remaining);word(g+40,group.main_remaining);word(g+44,group.total_count);
    for (unsigned i=0;i<16;++i) word(g+48+i*4,group.config.input_registers[i]);
    std::memcpy(data.data()+g+128,group.history.data(),group.history.size());
    return gas.Empty() ? data : PackGasArchive(data, gas);
}
void RasterizerMetal::LoadMetalState(std::span<const u8> data) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    RetireGeometry(true);
    constexpr std::array<u8,8> magic{'A','Z','M','4','S','T','A','T'};
    if ((data.size()!=LegacyStateBytes && data.size()!=StateArchiveBytes) || !std::equal(magic.begin(),magic.end(),data.begin()))
        throw std::invalid_argument("Invalid native Metal state size or signature");
    GasTransferState restored_gas{};
    std::vector<u8> expanded;
    if (GasArchiveWord(data, 8) == 3 || GasArchiveWord(data, 8) == 4) {
        expanded = UnpackGasArchive(data, restored_gas);
        data = expanded;
    }
    const auto word = [&](unsigned offset) {
        u32 value{};
        for (unsigned byte=0; byte<4; ++byte) value|=u32(data[offset+byte])<<(8*byte);
        return value;
    };
    const bool legacy=word(8)==1;
    if ((legacy ? data.size()!=LegacyStateBytes || word(28)!=0 :
                  word(8)!=2 || data.size()!=StateArchiveBytes || word(28)>1) || word(20)>1 || word(24)>1)
        throw std::invalid_argument("Unsupported native Metal state version or flags");
    PrimitiveSnapshot primitive{.pending=word(16),.topology=TriangleTopology(word(12)),.needs_reset=word(20)!=0};
    VertexInputSnapshot input{.needs_reset=word(24)!=0};
    std::memcpy(primitive.history.data(),data.data()+32,primitive.history.size());
    std::memcpy(input.history.data(),data.data()+32+primitive.history.size(),input.history.size());
    ShaderMachineSnapshot vertex_shader{},geometry_shader{};
    GeometryInputSnapshot group{};
    if (!legacy) {
        const auto shader=[&](unsigned offset,ShaderMachineSnapshot& snapshot) {
            if (word(offset)>1 || word(offset+4) || word(offset+8) || word(offset+12))
                throw std::invalid_argument("Invalid native shader snapshot flags/padding");
            snapshot.needs_reset=word(offset)!=0;
            std::memcpy(snapshot.history.data(),data.data()+offset+16,snapshot.history.size());
        };
        shader(LegacyStateBytes,vertex_shader);shader(LegacyStateBytes+ShaderStateBlockBytes,geometry_shader);
        const auto g=unsigned(GeometryStateOffset),flags=word(g);
        if (flags>7 || word(g+112) || word(g+116) || word(g+120) || word(g+124))
            throw std::invalid_argument("Invalid native GS grouping flags/padding");
        group.configured=flags&1;group.initialized=flags&2;group.awaiting=flags&4;
        group.config.mode=GeometryInputMode(word(g+4));group.config.vs_output_mask=word(g+8);group.config.vs_output_count=word(g+12);
        group.config.input_count=word(g+16);group.config.uniform_start=word(g+20);
        group.config.fixed_vertex_count=word(g+24);group.config.variable_main_count=word(g+28);
        group.cursor=word(g+32);group.remaining=word(g+36);group.main_remaining=word(g+40);group.total_count=word(g+44);
        for(unsigned i=0;i<16;++i) group.config.input_registers[i]=word(g+48+i*4);
        std::memcpy(group.history.data(),data.data()+g+128,group.history.size());
        if (bool(word(28))!=group.initialized)
            throw std::invalid_argument("Native GS uniform seeding flag does not match initialized GPU banks");
    }
    // Validate the whole blob before clearing caches or replacing either GPU bank.
    GeometryRuntime::ValidateStream(primitive);
    if (input.needs_reset && std::any_of(input.history.begin(),input.history.end(),
                                        [](std::byte value) { return value!=std::byte{}; }))
        throw std::invalid_argument("Invalid reset input bank in native Metal state");
    ShaderMachineRuntime::ValidateState(vertex_shader);ShaderMachineRuntime::ValidateState(geometry_shader);
    GeometryProgramRuntime::ValidateState(group);
    ClearAll(false);
    geometry.LoadStream(primitive_stream,primitive);
    fetch.LoadState(vertex_input_state,input);
    machine.LoadState(vs_unit,vertex_shader);machine.LoadState(gs_unit,geometry_shader);
    geometry_program.LoadState(geometry_input,group);gs_uniforms_seeded=!legacy && word(28)!=0;
    gas = restored_gas;
    if (gas.feedback_pending) gas_feedback.Reset();
    else gas_runtime.LoadMaximum(gas_feedback, std::uint16_t(gas.maximum));
    gas_maximum_pending = false;
    scheduler.Finish();
}
void RasterizerMetal::TickFrame() {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::State};
    RetireGeometry();
    if (vertex_input_state.references) {
        // Compact long-lived retained inputs once per frame. This reads only
        // immutable buffer data, so it need not wait for preceding mesh stages.
        Scheduler::BufferComputeScope buffers{scheduler};
        input_references.Resolve(vertex_input_state);
    }
    cache.TickFrame();
    device.SetDiskCacheEnabled(Settings::values.use_disk_shader_cache.GetValue());
    if(++disk_cache_frames>=120) {
        disk_cache_frames=0;
        if(Settings::values.use_disk_shader_cache.GetValue())device.RequestDiskCacheFlush();
    }
}
bool RasterizerMetal::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Transfer};
    @autoreleasepool {
        if (!config.output_width || !config.output_height) return true;
        // Match the Vulkan/OpenGL accelerated cache path: block_32 does not change
        // logical GPU images. The existing guest-memory layout remains unchanged;
        // this compatibility behavior is not a physical Block32 address codec.
        if (config.input_format > Pica::PixelFormat::RGBA4 || config.output_format > Pica::PixelFormat::RGBA4 ||
            config.scaling > config.ScaleXY || config.output_width > config.input_width || config.output_height > config.input_height)
            throw std::invalid_argument("Invalid native PICA display-transfer format, dimensions or scaling");
        const auto params = [&](std::uint64_t address, u32 width, u32 height, u32 stride,
                                Pica::PixelFormat format, bool tiled) {
            const auto pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(format);
            const auto bytes = GuestTextureCodec::GuestBytes({width, height, stride, pixel_format, tiled});
            if (address > std::numeric_limits<PAddr>::max() || bytes > std::numeric_limits<u32>::max() ||
                address + bytes > std::uint64_t(std::numeric_limits<PAddr>::max()) + 1)
                throw std::invalid_argument("Native display transfer has an invalid physical interval");
            const auto backing = memory.GetPhysicalRef(PAddr(address));
            if (!backing || backing.GetSize() < bytes)
                throw std::invalid_argument("Native display transfer exceeds physical backing");
            VideoCore::SurfaceParams result{};
            result.addr = PAddr(address); result.width = width; result.height = height; result.stride = stride;
            result.is_tiled = tiled; result.pixel_format = pixel_format;
            result.UpdateParams();
            if (result.size != bytes) throw std::logic_error("Native display-transfer/cache byte layout mismatch");
            return result;
        };
        // Transfer bit 5 selects tiled-to-tiled and takes precedence over bit 1.
        const bool source_tiled = config.dont_swizzle || !config.input_linear;
        const bool dest_tiled = config.dont_swizzle || config.input_linear;
        const auto source_params = params(std::uint64_t(config.input_address) * 8, config.output_width,
            config.output_height, config.input_width, config.input_format, source_tiled);
        const auto width = config.output_width.Value() >> (config.scaling != config.NoScale ? 1 : 0);
        const auto height = config.output_height.Value() >> (config.scaling == config.ScaleXY ? 1 : 0);
        auto dest_address = std::uint64_t(config.output_address) * 8;
        if (config.flip_vertically && config.crop_input_lines)
            dest_address += std::uint64_t(config.input_width - config.output_width) * (config.output_height - 1) *
                            Pica::BytesPerPixel(config.output_format);
        auto dest_params = params(dest_address, width, height, width, config.output_format, dest_tiled);
        auto [source_id, source_rect] = cache.GetSurfaceSubRect(source_params, VideoCore::ScaleMatch::Ignore, true);
        if (!source_id) throw std::runtime_error("Native display-transfer source surface could not be resolved");
        dest_params.res_scale = cache.GetSurface(source_id).res_scale;
        auto [dest_id, dest_rect] = cache.GetSurfaceSubRect(dest_params, VideoCore::ScaleMatch::Upscale, false);
        if (!dest_id) throw std::runtime_error("Native display-transfer destination surface could not be resolved");
        // Resolve both handles after insertion; the slot container can move its surfaces.
        auto& source = cache.GetSurface(source_id);
        auto& dest = cache.GetSurface(dest_id);
        if (source.is_tiled != dest.is_tiled) std::swap(source_rect.top, source_rect.bottom);
        if (config.flip_vertically) std::swap(source_rect.top, source_rect.bottom);
        textures.DisplayTransfer(source, dest, {
            .src_level = source.LevelOf(source_params.addr), .dst_level = dest.LevelOf(dest_params.addr),
            .src_rect = source_rect, .dst_rect = dest_rect});
        cache.InvalidateRegion(dest_params.addr, dest_params.size, dest_id);
        return true;
    }
}
bool RasterizerMetal::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Transfer};
    @autoreleasepool {
        const auto count = config.texture_copy.size & ~15U;
        if (!count) return true;
        const auto width = [&](u32 value, u32 gap) { return gap ? value * 16 : count; };
        const GuestCopyLayout layout{count,
            width(config.texture_copy.input_width, config.texture_copy.input_gap), config.texture_copy.input_gap * 16U,
            width(config.texture_copy.output_width, config.texture_copy.output_gap), config.texture_copy.output_gap * 16U};
        const auto input_size = GuestTextureCodec::CopyBytes(count, layout.input_width, layout.input_gap);
        const auto output_size = GuestTextureCodec::CopyBytes(count, layout.output_width, layout.output_gap);
        const auto input_address = std::uint64_t(config.input_address) * 8;
        const auto output_address = std::uint64_t(config.output_address) * 8;
        const auto physical = [&](std::uint64_t address, std::uint64_t size) {
            if (address > std::numeric_limits<PAddr>::max() || size > std::numeric_limits<u32>::max() ||
                address + size > std::uint64_t(std::numeric_limits<PAddr>::max()) + 1)
                throw std::invalid_argument("Native texture copy has an invalid physical range");
            const auto ref = memory.GetPhysicalRef(PAddr(address));
            if (!ref || ref.GetSize() < size) throw std::invalid_argument("Native texture copy exceeds physical memory");
            return ref;
        };
        const auto source = physical(input_address, input_size);
        auto dest = physical(output_address, output_size);
        // Retain resolution-enhanced cached copies when their rectangles are compatible.
        if (cache.AccelerateTextureCopy(config)) return true;
        cache.FlushRegion(PAddr(input_address), u32(input_size));
        cache.FlushRegion(PAddr(output_address), u32(output_size));
        const auto input = vertex_uploads.Allocate(input_size);
        const auto output = vertex_uploads.Allocate(output_size);
        std::memcpy(input.data, source.GetPtr(), input_size);
        std::memcpy(output.data, dest.GetPtr(), output_size);
        // Both physical snapshots are immutable until the GPU completes. This also makes
        // overlapping guest addresses safe and preserves every destination gap byte.
        // CPU leases and submission ticks protect disjoint ranges in the shared guest arena.
        textures.CopyGuest(layout, input, output);
        scheduler.Finish(Scheduler::WaitReason::Readback);
        std::memcpy(dest.GetPtr(), output.data, output_size);
        cache.InvalidateRegion(PAddr(output_address), u32(output_size));
        return true;
    }
}
bool RasterizerMetal::AccelerateFill(const Pica::MemoryFillConfig& config) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Transfer};
    const auto start = std::uint64_t(config.address_start) * 8;
    const auto end = std::uint64_t(config.address_end) * 8;
    if (start > std::numeric_limits<PAddr>::max() || end > std::numeric_limits<PAddr>::max() || end < start)
        throw std::invalid_argument("Native memory fill has an invalid physical interval");
    if (start == end) return true;
    const auto backing = memory.GetPhysicalRef(PAddr(start));
    if (!backing || end - start > backing.GetSize())
        throw std::invalid_argument("Native memory fill exceeds physical memory");
    // Validate before the shared cache registers pages or makes this lazy GPU fill
    // the interval owner. DownloadFillSurface performs the actual fill on Metal.
    return cache.AccelerateFill(config);
}

bool RasterizerMetal::GetScreen(const Pica::FramebufferConfig& config, PAddr address, u32 stride, ScreenImage& screen) {
    Scheduler::HostPhaseScope host_profile{scheduler, Scheduler::HostPhase::Present};
    if (!address || !stride || !config.width || !config.height) return false;
    if (config.color_format > Pica::PixelFormat::RGBA4)
        throw std::invalid_argument("Native LCD screen has an invalid color format");
    VideoCore::SurfaceParams params{};
    params.addr = address;
    params.width = std::min(config.width.Value(), stride);
    params.height = config.height;
    params.stride = stride;
    params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    const auto bytes = GuestTextureCodec::GuestBytes({params.width, params.height, stride, params.pixel_format});
    if (bytes > std::numeric_limits<u32>::max() ||
        std::uint64_t(address) + bytes > std::numeric_limits<PAddr>::max())
        throw std::invalid_argument("Native LCD screen has an invalid physical interval");
    const auto backing = memory.GetPhysicalRef(address);
    if (!backing || backing.GetSize() < bytes)
        throw std::invalid_argument("Native LCD screen exceeds physical memory");
    params.UpdateParams();
    if (params.size != bytes) throw std::logic_error("Native LCD/cache byte layout mismatch");
    const auto [id, rect] = cache.GetSurfaceSubRect(params, VideoCore::ScaleMatch::Ignore, true);
    if (!id) return false;
    const auto& surface = cache.GetSurface(id);
    screen.texture = surface.ImageView();
    screen.texcoords = {float(rect.bottom) / surface.GetScaledHeight(), float(rect.left) / surface.GetScaledWidth(),
                        float(rect.top) / surface.GetScaledHeight(), float(rect.right) / surface.GetScaledWidth()};
    return true;
}

} // namespace Metal
