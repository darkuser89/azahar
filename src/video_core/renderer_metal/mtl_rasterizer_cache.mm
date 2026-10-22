// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_texture_runtime.h"
#include "common/settings.h"
#include "video_core/rasterizer_cache/rasterizer_cache.h"

namespace VideoCore {

template <>
void RasterizerCache<Metal::Traits>::ValidateSurface(SurfaceId surface_id,PAddr address,u32 size) {
    if(!size) return;
    auto& surface=slot_surfaces[surface_id];
    if(address<surface.addr || address>surface.end || size>surface.end-address)
        throw std::invalid_argument("Native surface validation exceeds its guest interval");
    SurfaceInterval validate_interval(address,address+size);
    if(surface.type==SurfaceType::Fill) {
        if(!surface.IsRegionValid(validate_interval)) throw std::logic_error("Invalid native fill surface interval");
        return;
    }
    const bool filtered=surface.res_scale!=1 && !surface.IsCustom() &&
        (surface.type==SurfaceType::Color || surface.type==SurfaceType::Texture) &&
        runtime.TextureFilterMode()!=Settings::TextureFilter::NoFilter;
    if(filtered) {
        const auto last_level=surface.LevelOf(address+size-1);
        // A requested neighbor may depend on a dirty pixel outside its byte
        // interval. Validate the canonical mip dependencies, including level0
        // when the existing cache policy generates filtered mips from it.
        validate_interval=SurfaceInterval(surface.LevelInterval(0).lower(),surface.LevelInterval(last_level).upper());
    }
    SurfaceRegions validate_regions=surface.invalid_regions&validate_interval;
    if(validate_regions.empty()) return;
    const auto validated=[&](SurfaceInterval interval) {
        surface.MarkValid(interval); validate_regions.erase(interval);
    };
    u32 level=surface.LevelOf(validate_interval.lower());
    SurfaceInterval level_interval=surface.LevelInterval(level);
    while(!validate_regions.empty()) {
        const auto interval=*validate_regions.begin()&level_interval;
        if(boost::icl::is_empty(interval)) {level_interval=surface.LevelInterval(++level);continue;}
        const auto params=surface.FromInterval(interval);
        const auto source_id=FindMatch<MatchFlags::Copy>(params,ScaleMatch::Ignore,interval);
        if(source_id && source_id!=surface_id) {
            auto& source=slot_surfaces[source_id];
            const auto copy_interval=source.GetCopyableInterval(params);
            CopySurface(source,surface,copy_interval); validated(copy_interval); continue;
        }
        if(ValidateByReinterpretation(surface,params,interval)) {validated(interval);continue;}
        FlushRegion(params.addr,params.size);
        if(!use_custom_textures || !UploadCustomSurface(surface_id,interval)) UploadSurface(surface,interval);
        validated(params.GetInterval());
    }
    if(surface.res_scale!=1 && level!=0) runtime.GenerateMipmaps(surface);
}

template <>
u64 RasterizerCache<Metal::Traits>::ComputeHash(const SurfaceParams& params, std::span<u8> data) {
    if (custom_tex_manager.UseNewHash())
        return Common::ComputeHash64<Common::HashAlgo64::CityHash>(data.data(), data.size());
    const auto decoded = runtime.DecodeGuest(params, data, false);
    return Common::ComputeHash64<Common::HashAlgo64::CityHash>(decoded.data(), decoded.size());
}

template <>
void RasterizerCache<Metal::Traits>::UploadSurface(Metal::Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_UploadSurface);
    auto info = surface.FromInterval(interval);
    std::optional<Common::Rectangle<u32>> filter_damage;
    if(surface.res_scale!=1 && !surface.IsCustom() &&
       (surface.type==SurfaceType::Color || surface.type==SurfaceType::Texture) &&
       runtime.TextureFilterMode()!=Settings::TextureFilter::NoFilter) {
        filter_damage=surface.GetSubRect(info);
        info=surface.FromInterval(surface.LevelInterval(surface.LevelOf(info.addr)));
        // Neighbor taps need the whole canonical mip, including intervals owned
        // by another GPU surface. Flush those owners before taking the physical
        // snapshot; the scaled destination keeps pixels outside the damage halo.
        FlushRegion(info.addr,info.size);
    }
    if (info.addr < surface.addr || info.end > surface.end)
        throw std::invalid_argument("Native cache upload rectangle exceeds the guest surface");
    const auto source = memory.GetPhysicalRef(info.addr);
    if (!source || source.GetSize() < info.size)
        throw std::invalid_argument("Native cache upload exceeds available physical memory");
    const auto data = source.GetReadBytes<u8>(info.size);
    runtime.UploadGuest(surface, info, data,filter_damage);
    if (dump_textures && False(surface.flags & SurfaceFlagBits::Custom) && False(surface.flags & SurfaceFlagBits::RenderTarget)) {
        // Hashing needs mutable span only for the unchanged common method signature.
        auto source_write = memory.GetPhysicalRef(info.addr);
        const auto hash = ComputeHash(info, source_write.GetWriteBytes(info.size));
        runtime.DumpGuest(info, surface.LevelOf(info.addr), hash, data);
    }
}

template <>
void RasterizerCache<Metal::Traits>::DownloadSurface(Metal::Surface& surface, SurfaceInterval interval) {
    MICROPROFILE_SCOPE(RasterizerCache_DownloadSurface);
    const auto info = surface.FromInterval(interval);
    if (info.addr < surface.addr || info.end > surface.end || interval.lower() < info.addr || interval.upper() > info.end)
        throw std::invalid_argument("Native cache flush rectangle exceeds the guest surface");
    auto dest = memory.GetPhysicalRef(info.addr);
    if (!dest || dest.GetSize() < info.size)
        throw std::invalid_argument("Native cache flush exceeds available physical memory");
    runtime.DownloadGuest(surface, info, dest.GetWriteBytes(info.size),
                          NSMakeRange(interval.lower() - info.addr, interval.upper() - interval.lower()));
}

template <>
void RasterizerCache<Metal::Traits>::DownloadFillSurface(Metal::Surface& surface, SurfaceInterval interval) {
    if (interval.lower() < surface.addr || interval.upper() > surface.end || surface.fill_size < 2 || surface.fill_size > 4)
        throw std::invalid_argument("Native cache memory fill exceeds its guest interval or pattern");
    auto dest = memory.GetPhysicalRef(interval.lower());
    const auto size = interval.upper() - interval.lower();
    if (!dest || dest.GetSize() < size) throw std::invalid_argument("Native cache memory fill exceeds physical memory");
    runtime.FillGuest(dest.GetWriteBytes(size), surface.fill_data, surface.fill_size,
                      (interval.lower() - surface.addr) % surface.fill_size);
}

template <>
bool RasterizerCache<Metal::Traits>::ValidateByReinterpretation(Metal::Surface& surface, SurfaceParams params,
                                                               const SurfaceInterval& interval) {
    const auto source_id = FindMatch<MatchFlags::Reinterpret>(params, ScaleMatch::Ignore, interval);
    if (source_id) {
        auto& source = slot_surfaces[source_id];
        const auto copy_interval = source.GetCopyableInterval(params);
        if (!boost::icl::is_empty(copy_interval & interval)) {
            if (source.res_scale > surface.res_scale) surface.ScaleUp(source.res_scale);
            const auto copy_params = surface.FromInterval(copy_interval);
            const auto src = source.GetScaledSubRect(copy_params), dst = surface.GetScaledSubRect(copy_params);
            if (runtime.Reinterpret(source, surface, {
                    .src_level = source.LevelOf(interval.lower()), .dst_level = surface.LevelOf(interval.lower()),
                    .src_offset = {src.left, src.bottom}, .dst_offset = {dst.left, dst.bottom},
                    .extent = {src.GetWidth(), src.GetHeight()}})) return true;
        }
    }
    // A different guest stride changes the interpretation of the same physical bytes.
    // Flush the owner's bytes with the GPU encoder, then load them through the destination
    // GPU decoder. Merely reporting the region as valid would leave old/uninitialized pixels.
    const bool stride_alias = std::any_of(RangeFromInterval(dirty_regions, interval).begin(),
                                         RangeFromInterval(dirty_regions, interval).end(), [&](const auto& item) {
        return item.second && slot_surfaces[item.second].stride != surface.stride;
    });
    if (!stride_alias) return false;
    FlushRegion(params.addr, params.size);
    UploadSurface(surface, interval);
    return true;
}

template class RasterizerCache<Metal::Traits>;
} // namespace VideoCore
