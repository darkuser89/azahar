// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#pragma once

#include "mtl_pipeline_cache.h"
#include "mtl_shader_translation.h"
#include "mtl_transfer.h"
#include <optional>

namespace Settings { enum class TextureFilter : std::uint32_t; }

namespace Metal {
enum class TextureFilterPass { Bicubic, ScaleForce, Xbrz, Mmpx, XGradient, YGradient, Refine };
TranslatedShader GenerateTextureFilterShader(TextureFilterPass pass);

inline constexpr std::string_view TextureFilterVertexSource=R"(
#include <metal_stdlib>
using namespace metal;
struct FilterInfo { float4 rect; float scale; float padding0, padding1, padding2; };
struct FilterVertex { float4 position [[position]]; float2 uv [[user(locn0)]]; };
vertex FilterVertex filter_vertex(uint index [[vertex_id]], constant FilterInfo& info [[buffer(21)]]) {
    const float2 positions[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    const float2 p=positions[index];
    FilterVertex out;
    out.position=float4(p.x,-p.y,0,1);
    out.uv=mix(info.rect.xy,info.rect.zw,p*.5f+.5f);
    return out;
}
)";

// Original filter GLSL is CPU-translated to MSL4.0. All filtering, intermediate
// gradients and overlap snapshots execute through native Metal4 commands.
class TextureFilterRuntime final {
public:
    TextureFilterRuntime(Device& device, Scheduler& scheduler);
    bool Filter(Settings::TextureFilter filter, const TextureRegion& source,
                const TextureRegion& dest, float scale, bool flip_x=false, bool flip_y=false,
                std::optional<MTLRegion> clip={});
private:
    struct Program { TranslatedShader shader; NativeLibraryFuture library; };
    Program& GetProgram(TextureFilterPass pass);
    id<MTLTexture> Allocate(NSUInteger width, NSUInteger height, MTLPixelFormat format,
                            bool render_target=true);
    void Pass(TextureFilterPass pass, const std::array<id<MTLTexture>,3>& images,
              const TextureRegion& dest, std::array<float,4> uv, float scale,
              std::optional<MTLRegion> clip={});
    Device& device;
    Scheduler& scheduler;
    TransferRuntime transfers;
    PipelineCache pipelines;
    NativeLibraryFuture vertex;
    id<MTLSamplerState> sampler;
    std::map<TextureFilterPass,Program> programs;
};
} // namespace Metal
