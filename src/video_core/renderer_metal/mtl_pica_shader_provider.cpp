// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_pica_shader_provider.h"
#include "mtl_shader_text.h"
#include "mtl_gas_fragment.h"
#include "mtl_gas_accumulation.h"
#include "video_core/shader/generator/glsl_fs_shader_gen.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/shader/generator/shader_gen.h"
#include <nihstro/shader_bytecode.h>
#include <fmt/format.h>
#include <regex>
#include <bit>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace Metal {

PicaSamplerConfigs BuildPicaSamplerConfigs(const Pica::RegsInternal& regs) {
    PicaSamplerConfigs result{};
    const auto textures = regs.texturing.GetTextures();
    for (unsigned i = 0; i < result.size(); ++i) {
        if (!textures[i].enabled) continue;
        const auto& config = textures[i].config;
        result[i] = {config.wrap_s.Value(), config.wrap_t.Value(), config.min_filter.Value(),
                     config.mag_filter.Value(), config.mip_filter.Value(),
                     config.lod.min_level.Value(), config.lod.max_level.Value()};
    }
    return result;
}

namespace {
void ApplyPicaLightBindings(std::string& source, const Pica::Shader::FSConfig& config) {
    // The generator uses light.num both for physical resources and to index its
    // active-slot config array. Generate with slot IDs, then restore only the
    // physical uniform/LUT bindings. This keeps each slot's two-sided LUT rule.
    static const std::regex binding{
        R"(\blight_src\[([0-7])\]|\bLookupLightingLUT(Unsigned|Signed)\((8|9|1[0-9]|2[0-3]),)"};
    std::string remapped;
    remapped.reserve(source.size());
    std::size_t consumed{};
    ShaderTextMatcher bindings{source, binding,
        {"light_src[", "LookupLightingLUTUnsigned(", "LookupLightingLUTSigned("}};
    std::size_t begin{};
    std::cmatch match;
    while (bindings.Next(begin, match)) {
        const bool uniform = match[1].matched;
        const auto index = unsigned(std::stoul(match[uniform ? 1 : 3].str()));
        const auto slot = uniform ? index : (index - 8) % 8;
        if (slot >= config.lighting.src_num)
            throw std::logic_error("Generated lighting binding refers to an inactive slot");
        const auto physical = config.lighting.lights[slot].num.Value();
        remapped.append(source, consumed, begin - consumed);
        remapped += uniform ? fmt::format("light_src[{}]", physical) :
            fmt::format("LookupLightingLUT{}({},", match[2].str(), index - slot + physical);
        consumed = begin + std::size_t(match.length());
    }
    remapped.append(source, consumed, std::string::npos);
    source = std::move(remapped);
}

std::string GasTevSource(Pica::TexturingRegs::TevStageConfig::Source source) {
    using S = Pica::TexturingRegs::TevStageConfig::Source;
    switch (source) {
    case S::PrimaryColor: return "rounded_primary_color";
    case S::PrimaryFragmentColor: return "primary_fragment_color";
    case S::SecondaryFragmentColor: return "secondary_fragment_color";
    case S::Texture0: return "sampleTexUnit0()";
    case S::Texture1: return "sampleTexUnit1()";
    case S::Texture2: return "sampleTexUnit2()";
    case S::Texture3: return "sampleTexUnit3()";
    case S::PreviousBuffer: return "combiner_buffer";
    case S::Constant: return "const_color[5]";
    case S::Previous: return "combiner_output";
    default: throw std::invalid_argument("Unknown native Gas TEV source");
    }
}

void ApplyGasPostsynthesis(std::string& source, const Pica::Shader::FSConfig& config) {
    if (config.framebuffer.alpha_test_func == Pica::FramebufferRegs::CompareFunc::Never) return;
    constexpr std::string_view final_color = "combiner_output = byteround(combiner_output);";
    const auto final = source.rfind(final_color);
    if (final == std::string::npos) throw std::logic_error("PICA final fragment color is absent");
    source.replace(final, final_color.size(),
                   "combiner_output = byteround(metalGasSynthesize(metal_gas_xy, metal_gas_density));");
    // Capture the last stage's inputs before it overwrites Previous or its buffer.
    auto position = source.rfind("combiner_buffer = next_combiner_buffer;");
    if (const auto output = source.find("vec3 color_output_5 ="); output != std::string::npos)
        position = source.rfind("color_results_1 =", output);
    if (position == std::string::npos) throw std::logic_error("PICA final TEV stage is absent");
    const Pica::TexturingRegs::TevStageConfig stage = config.texture.tev_stages[5];
    source.insert(position, fmt::format("float metal_gas_xy = ({}).r;\nvec4 metal_gas_density = {};\n",
        GasTevSource(stage.color_source1.Value()), GasTevSource(stage.color_source3.Value())));
    const auto main = source.find("void main()");
    if (main == std::string::npos) throw std::logic_error("PICA fragment main is absent");
    source.insert(main, GasFragmentHelpers);
}

void ApplyGasAccumulation(std::string& source, const Pica::Shader::FSConfig& config) {
    if (config.framebuffer.alpha_test_func == Pica::FramebufferRegs::CompareFunc::Never) return;
    const auto terminal = source.rfind("gl_FragDepth = depth;");
    const auto main = source.find("void main()");
    if (terminal == std::string::npos || main == std::string::npos)
        throw std::logic_error("PICA accumulation fragment body is absent");
    // Preserve TEV, clipping, scissor and alpha testing. Gas replaces the output
    // operation itself; ordinary RGB blending/logic must not run a second time.
    source.replace(terminal, source.size() - terminal,
                   "color = metalGasAccumulate(combiner_output.r, depth);\n}");
    source.insert(main, GasAccumulationHelpers);
}

bool ShaderSampler(const PicaSamplerConfig& config) {
    const auto special = [](unsigned wrap) { return wrap == 1 || wrap == 4 || wrap == 5; };
    return special(config.wrap_s) || special(config.wrap_t);
}

void ApplyPicaSamplers(std::string& source, const Pica::Shader::FSConfig& config,
                       const Pica::Shader::UserConfig& user, const PicaSamplerConfigs& samplers) {
    bool helpers_added{};
    for (unsigned unit = 0; unit < samplers.size(); ++unit) {
        const auto& s = samplers[unit];
        if (s.wrap_s > 7 || s.wrap_t > 7 || s.min_filter > 1 || s.mag_filter > 1 ||
            s.mip_filter > 1 || s.lod_min > s.lod_max || s.lod_max > 15)
            throw std::invalid_argument("Invalid native shader sampler configuration");
        if (!ShaderSampler(s)) continue;
        using C = Pica::TexturingRegs::TextureConfig;
        // The unchanged cube path performs native directional sampling; the shadow
        // paths perform their own bounded integer reads. Neither uses the 2D
        // normalized-coordinate wrap helpers or the guest sampler border color.
        if (unit == 0 && (config.texture.texture0_type == C::TextureCube ||
                          config.texture.texture0_type == C::Shadow2D ||
                          config.texture.texture0_type == C::ShadowCube)) continue;
        if (unit == 0 && config.texture.texture0_type != C::Texture2D &&
            config.texture.texture0_type != C::Projection2D && config.texture.texture0_type != C::Disabled)
            throw std::runtime_error("Native cube/shadow border sampling requires its own shader path");
        if (unit == 0 && config.texture.texture0_type == C::Disabled) continue;
        const auto signature = fmt::format("vec4 sampleTexUnit{}() {{", unit);
        const auto begin = source.find(signature);
        if (begin == std::string::npos) throw std::logic_error("PICA sampler function was not generated");
        const auto body = begin + signature.size();
        std::size_t end = body;
        unsigned depth = 1;
        while (end < source.size() && depth) {
            if (source[end] == '{') ++depth;
            if (source[end] == '}') --depth;
            ++end;
        }
        if (depth) throw std::logic_error("PICA sampler function has unmatched braces");
        auto helpers = std::string{};
        if (!helpers_added) {
            helpers += R"(
int metalWrapMode(int mode, float coordinate) {
    return mode == 4 ? (coordinate < 0.0 ? 2 : 0) :
           mode == 5 ? (coordinate < 0.0 ? 2 : 1) : mode;
}
float metalWrapCoordinate(float coordinate, int mode) {
    if (mode == 2 || mode == 6 || mode == 7) return fract(coordinate);
    if (mode == 3) { float p = mod(coordinate, 2.0); return p > 1.0 ? 2.0 - p : p; }
    return clamp(coordinate, -1.0, 2.0);
}
int metalWrapIndex(int index, int size, int mode) {
    if (mode == 0) return clamp(index, 0, size - 1);
    if (mode == 1) return index < 0 || index >= size ? -1 : index;
    if (mode == 3) { int p = ((index % (2 * size)) + 2 * size) % (2 * size); return p < size ? p : 2 * size - 1 - p; }
    return ((index % size) + size) % size;
}
)";
            helpers_added = true;
        }
        helpers += fmt::format(R"(
vec4 metalRead{0}(ivec2 p, ivec2 size, ivec2 modes, int level) {{
    p = ivec2(metalWrapIndex(p.x, size.x, modes.x), metalWrapIndex(p.y, size.y, modes.y));
    if (p.x < 0 || p.y < 0) return tex_border_color[{0}];
    return texelFetch(tex{0}, p, level);
}}
vec4 metalLevel{0}(vec2 uv, int level, bool linear_filter) {{
    ivec2 modes = ivec2(metalWrapMode({1}, uv.x), metalWrapMode({2}, uv.y));
    uv = vec2(metalWrapCoordinate(uv.x, modes.x), metalWrapCoordinate(uv.y, modes.y));
    ivec2 size = textureSize(tex{0}, level);
    vec2 p = uv * vec2(size);
    if (!linear_filter) return metalRead{0}(ivec2(floor(p)), size, modes, level);
    p -= vec2(0.5);
    ivec2 base = ivec2(floor(p));
    vec2 weight = fract(p);
    return mix(mix(metalRead{0}(base, size, modes, level),
                   metalRead{0}(base + ivec2(1,0), size, modes, level), weight.x),
               mix(metalRead{0}(base + ivec2(0,1), size, modes, level),
                   metalRead{0}(base + ivec2(1,1), size, modes, level), weight.x), weight.y);
}}
vec4 metalSample{0}(vec2 uv, float lod) {{
    lod = clamp(lod, float({5}), float({6}));
    bool linear_filter = lod <= 0.0 ? {3} : {4};
    float last = float(textureQueryLevels(tex{0}) - 1);
    lod = min(lod, last);
    if (!{7}) return metalLevel{0}(uv, int(floor(lod + 0.5)), linear_filter);
    int low = int(floor(lod));
    return mix(metalLevel{0}(uv, low, linear_filter),
               metalLevel{0}(uv, min(low + 1, int(last)), linear_filter), fract(lod));
}}
)", unit, s.wrap_s, s.wrap_t, s.mag_filter ? "true" : "false", s.min_filter ? "true" : "false",
            s.lod_min, s.lod_max, s.mip_filter ? "true" : "false");
        const std::string normal_lookup="texture(tex_normal, texcoord0)";
        const bool normal_sampler=unit==0 && user.use_custom_normal && source.find(normal_lookup)!=std::string::npos;
        if(normal_sampler) {
            // Normal maps share unit0's sampler, but retain the original implicit LOD
            // and unprojected UV contract; guest texture bias is not applied to them.
            const auto definitions=helpers.find("\nvec4 metalRead0(");
            if(definitions==std::string::npos) throw std::logic_error("Native normal sampler template is missing");
            auto normal_helpers=helpers.substr(definitions);
            static const std::regex normal_function{R"(\bmetal(Read|Level|Sample)0\b)"};
            static const std::regex normal_texture{R"(\btex0\b)"};
            normal_helpers=ReplaceShaderText(normal_helpers,normal_function,
                {"metalRead0","metalLevel0","metalSample0"},"metal$1Normal");
            normal_helpers=ReplaceShaderText(normal_helpers,normal_texture,{"tex0"},"tex_normal");
            helpers+=normal_helpers;
        }
        const auto coord = unit == 2 && config.texture.texture2_use_coord1 ? 1U : unit;
        const auto uv = unit == 0 && config.texture.texture0_type == C::Projection2D ?
                        "texcoord0 / texcoord0_w" : fmt::format("texcoord{}", coord);
        const auto replacement = fmt::format(
            "{}\n{}\nreturn metalSample{}({}, getLod(({}) * vec2(textureSize(tex{}, 0))) + tex_lod_bias[{}]);\n}}",
            helpers, signature, unit, uv, uv, unit, unit);
        source.replace(begin, end - begin, replacement);
        if(normal_sampler) {
            const auto lookup=source.find(normal_lookup);
            source.replace(lookup,normal_lookup.size(),
                "metalSampleNormal(texcoord0, getLod(texcoord0 * vec2(textureSize(tex_normal, 0))))");
        }
    }
}
} // namespace

static TranslatedShader FinalizeVertexMSL(TranslatedShader shader, bool reset_machine) {
    if(reset_machine) shader.source.insert(0,"#pragma clang fp contract(off)\n");
    static const std::regex temporary_pattern{
        R"(\b((?:float|int|bool)[234]?\s+)(reg_tmp[0-9]+|conditional_code|address_registers)(\s*=))"};
    static const std::regex call_pattern{
        R"(\b(bool\s+_[0-9]+\s*=\s*(?:exec_shader|sub_[0-9]+_[0-9]+)\s*\())"};
    shader.source = ReplaceShaderText(shader.source, temporary_pattern,
        {"float","int","bool"}, "[[maybe_unused]] $1$2$3");
    shader.source = ReplaceShaderText(shader.source, call_pattern, {"bool"}, "[[maybe_unused]] $1");
    return shader;
}

static TranslatedShader GeneratePicaVertexShaderStage(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
                                                     bool reset_machine, ShaderStage stage) {
    if (regs.vs.main_offset >= Pica::MAX_PROGRAM_CODE_LENGTH ||
        regs.rasterizer.vs_output_total > 7 ||
        setup.GetBiggestProgramSize() > Pica::MAX_PROGRAM_CODE_LENGTH ||
        setup.GetBiggestSwizzleSize() > Pica::MAX_SWIZZLE_DATA_LENGTH) {
        throw std::invalid_argument("Invalid PICA vertex program or semantic configuration bounds");
    }
    for (unsigned i = 0; i < regs.rasterizer.vs_output_total; ++i) {
        const auto mapping = regs.rasterizer.vs_output_attributes[i].raw;
        for (unsigned c = 0; c < 4; ++c) {
            const auto semantic = (mapping >> (c * 8)) & 31U;
            if (semantic >= 24 && semantic != 31) {
                throw std::invalid_argument("Unknown PICA vertex output semantic");
            }
        }
    }
    // Follow all possible control-flow edges before the shared recursive decompiler. Reject
    // out-of-range blocks and reached GS instructions instead of silently dropping emissions.
    using Op = nihstro::OpCode::Id;
    using RangePoint = std::pair<std::uint32_t, std::uint32_t>;
    std::set<RangePoint> visited;
    std::vector<RangePoint> pending{{regs.vs.main_offset.Value(), Pica::MAX_PROGRAM_CODE_LENGTH}};
    const auto push = [&](std::uint32_t offset, std::uint32_t end) {
        if (offset < Pica::MAX_PROGRAM_CODE_LENGTH && offset != end && !visited.contains({offset, end})) {
            pending.emplace_back(offset, end);
        }
    };
    while (!pending.empty()) {
        const auto [offset, limit] = pending.back();
        pending.pop_back();
        if (!visited.emplace(offset, limit).second) continue;
        const nihstro::Instruction instruction{setup.GetProgramCode()[offset]};
        const auto op = instruction.opcode.Value().EffectiveOpCode();
        if (op == Op::END) continue;
        if (op == Op::EMIT || op == Op::SETEMIT) {
            throw std::runtime_error("PICA geometry emission requires the native geometry-program path");
        }
        if (op == Op::CALL || op == Op::CALLC || op == Op::CALLU || op == Op::IFC || op == Op::IFU) {
            const auto begin = instruction.flow_control.dest_offset.Value();
            const auto end = begin + instruction.flow_control.num_instructions.Value();
            if (end > Pica::MAX_PROGRAM_CODE_LENGTH) {
                throw std::invalid_argument("PICA vertex control-flow block exceeds program storage");
            }
            if (op == Op::IFC || op == Op::IFU) {
                push(offset + 1, begin);
                push(begin, end);
                push(end, limit);
                continue;
            }
            push(begin, end);
        } else if (op == Op::JMPC || op == Op::JMPU || op == Op::LOOP) {
            const auto target = instruction.flow_control.dest_offset.Value();
            if (op == Op::LOOP) {
                push(offset + 1, target + 1);
                push(target + 1, limit);
                continue;
            }
            push(target, limit);
        }
        push(offset + 1, limit);
    }
    auto capture_regs=regs;
    if(reset_machine) capture_regs.rasterizer.vs_output_total.Assign(std::min(
        unsigned(regs.rasterizer.vs_output_total.Value()),unsigned(std::popcount(regs.vs.output_mask.Value()))));
    const Pica::Shader::Generator::PicaVSConfig config{capture_regs, setup};
    Pica::Shader::Generator::ExtraVSConfig extra{};
    extra.use_clip_planes = 1;
    extra.separable_shader = 1;
    extra.sanitize_mul = 1; // Includes correct DPH homogeneous components and PICA zero*infinity.
    extra.load_flags.fill(Pica::Shader::Generator::AttribLoadFlags::Float);
    auto source = Pica::Shader::Generator::GLSL::GenerateVertexShader(setup, config, extra);
    if (source.empty()) throw std::runtime_error("Unable to decompile the PICA vertex program for Metal");
    if(reset_machine) {
        // This policy is selected only for a proven fresh-unit capture. Keep the
        // legacy native shader's defaults and cache identity unchanged otherwise.
        static const std::regex reset_pattern{R"((reg_tmp[0-9]+|vs_out_attr[0-9]+)(\s*=\s*)vec4\(0\.0, 0\.0, 0\.0, 1\.0\))"};
        static const std::regex mova_pattern{R"(\bivec2\()"};
        static const std::regex integer_pattern{R"(uniforms\.i\[([0-3])\])"};
        source=ReplaceShaderText(source,reset_pattern,{"reg_tmp","vs_out_attr"},"$1$2vec4(0.0)");
        source=ReplaceShaderText(source,mova_pattern,{"ivec2("},"pica_mova(");
        source=ReplaceShaderText(source,integer_pattern,{"uniforms.i["},"(uniforms.i[$1] & uvec4(255u))");
        const std::string color="primary_color = min(abs(vtx_color), vec4(1.0));";
        const auto at=source.find(color);
        if(at==std::string::npos) throw std::logic_error("PICA reset capture color boundary is missing");
        source.replace(at,color.size(),"primary_color = vec4(0.0); for(int component=0;component<4;++component) { float v=vtx_color[component]; primary_color[component]=isnan(v) ? 1.0 : abs(v)<1.0 ? abs(v) : 1.0; }");
        const auto extension=source.rfind("#extension");
        const auto insert=extension==std::string::npos ? 0:source.find('\n',extension)+1;
        source.insert(insert,R"(
int pica_mova_scalar(float v) {
    if(isnan(v) || isinf(v)) return 0;
    return int(clamp(v,-2147483648.0,2147483520.0));
}
ivec2 pica_mova(vec4 v) { return ivec2(pica_mova_scalar(v.x),pica_mova_scalar(v.y)); }
)");
    }
    auto shader = TranslateGuestGLSL(source, stage, RasterizerShaderBindings());
    return FinalizeVertexMSL(std::move(shader), reset_machine);
}

TranslatedShader GeneratePicaVertexShader(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup, bool reset_machine) {
    return GeneratePicaVertexShaderStage(regs, setup, reset_machine, ShaderStage::VertexCompute);
}

TranslatedShader GeneratePicaVertexMeshShader(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup) {
    return GeneratePicaVertexShaderStage(regs, setup, true, ShaderStage::VertexMesh);
}

static TranslatedShader FinalizeFragmentMSL(TranslatedShader shader, const Pica::Shader::FSConfig& config) {
    static const std::regex sampler_pattern{
        R"(\b(sampler (?:tex[012]|tex_normal|metal_gas_depth)Smplr)\b)"};
    shader.source = ReplaceShaderText(shader.source, sampler_pattern, {"sampler "}, "[[maybe_unused]] $1");
    // The real generator emits common temporaries for every TEV/lighting specialization. Mark
    // these known locals explicitly, retaining strict compiler diagnostics for all other code.
    static const std::regex temporary{
        R"(\b((?:float|uint|int|bool)[234]?\s+)(rounded_primary_color|primary_fragment_color|secondary_fragment_color|combiner_buffer|next_combiner_buffer|color_results_[123]|alpha_results_[123]|diffuse_sum|specular_sum|light_vector|light_distance|refl_value|spot_dir|half_vector|dot_product|clamp_highlights|geo_factor|shadow|tangent)(\s*(?:=|;)))"};
    shader.source = ReplaceShaderText(shader.source, temporary,
        {"float","uint","int","bool"}, "[[maybe_unused]] $1$2$3");
    if (config.texture.fog_mode == Pica::TexturingRegs::FogMode::Gas) {
        // Postsynthesis consumes the two captured stage inputs, not the final TEV result.
        static const std::regex gas_temporary{R"(\b(float[234]?\s+)(depth|combiner_output)(\s*(?:=|;)))"};
        shader.source = ReplaceShaderText(shader.source, gas_temporary, {"float"}, "[[maybe_unused]] $1$2$3");
    }
    if (config.texture.texture0_type == Pica::TexturingRegs::TextureConfig::Shadow2D ||
        config.texture.texture0_type == Pica::TexturingRegs::TextureConfig::ShadowCube) {
        // Integer shadow images use texel reads, while SPIRV-Cross forwards their combined
        // sampler through helpers. That sampler is deliberately unused in these helpers.
        static const std::regex shadow_sampler{R"(\b((?:sampler|thread const array<sampler, 6>&) tex0Smplr))"};
        shader.source = ReplaceShaderText(shader.source, shadow_sampler,
            {"sampler","thread const array<sampler, 6>&"}, "[[maybe_unused]] $1");
    }
    return shader;
}

static FragmentOperations FragmentOperationState(const Pica::Shader::FSConfig& config,
    const Pica::Shader::UserConfig& user, FragmentColorFormat color_format, bool gas_accumulation) {
    if (config.lighting.src_num > 8 || user.raw > 1 || config.proctex.coord > 2 ||
        static_cast<std::uint32_t>(config.framebuffer.requested_logic_op) >= 16 ||
        static_cast<std::uint32_t>(config.framebuffer.requested_rgb_blend.src_factor) >= 16 ||
        static_cast<std::uint32_t>(config.framebuffer.requested_rgb_blend.dst_factor) >= 16 ||
        static_cast<std::uint32_t>(config.framebuffer.requested_alpha_blend.src_factor) >= 16 ||
        static_cast<std::uint32_t>(config.framebuffer.requested_alpha_blend.dst_factor) >= 16) {
        throw std::invalid_argument("Invalid PICA fragment configuration");
    }
    const bool gas_postsynthesis = config.texture.fog_mode == Pica::TexturingRegs::FogMode::Gas;
    if (gas_accumulation && (gas_postsynthesis || config.framebuffer.shadow_rendering ||
                             color_format != FragmentColorFormat::RGBA8))
        throw std::invalid_argument("Native Gas accumulation requires its RGBA8 operation without postsynthesis/shadow");
    if (gas_postsynthesis && config.framebuffer.shadow_rendering)
        throw std::invalid_argument("Native Gas postsynthesis cannot use the shadow operation mode");
    FragmentOperations operations{};
    operations.color_format = color_format;
    operations.logic_enabled = !gas_accumulation && !config.framebuffer.alphablend_enable && !config.framebuffer.shadow_rendering;
    operations.logic_operation = static_cast<std::uint8_t>(config.framebuffer.requested_logic_op);
    operations.blend_color_from_pica_uniform = true;
    const auto channel = [](const Pica::Shader::BlendConfig& blend) {
        return FragmentBlendChannel{true, std::uint8_t(blend.eq),
                                 std::uint8_t(blend.src_factor), std::uint8_t(blend.dst_factor)};
    };
    if (!gas_accumulation && config.framebuffer.alphablend_enable && !config.framebuffer.shadow_rendering) {
        operations.rgb = channel(config.framebuffer.requested_rgb_blend);
        operations.alpha = channel(config.framebuffer.requested_alpha_blend);
    }
    return operations;
}

PicaFragmentProgram GeneratePicaFragmentShader(const Pica::Shader::FSConfig& config,
                                              const Pica::Shader::UserConfig& user,
                                              FragmentColorFormat color_format,
                                              const PicaSamplerConfigs& samplers,
                                              bool gas_accumulation) {
    const bool gas_postsynthesis = config.texture.fog_mode == Pica::TexturingRegs::FogMode::Gas;
    Pica::Shader::Profile profile{};
    profile.is_vulkan = 1;
    profile.has_separable_shaders = 1;
    profile.has_clip_planes = 1;
    profile.has_fragment_shader_interlock = 1;
    profile.has_logic_op = 1;
    profile.has_blend_minmax_factor = 1;

    const auto operations = FragmentOperationState(config, user, color_format, gas_accumulation);
    auto normalized = config;
    // Stage zero's Previous selects source three, rather than an earlier output.
    // Resolve its first inputs before the shared generator's passthrough test;
    // otherwise Replace/Previous incorrectly leaves the initial zero color intact.
    using Tev = Pica::TexturingRegs::TevStageConfig;
    Tev first = normalized.texture.tev_stages[0];
    if (first.color_source1 == Tev::Source::Previous)
        first.color_source1.Assign(first.color_source3.Value());
    if (first.alpha_source1 == Tev::Source::Previous)
        first.alpha_source1.Assign(first.alpha_source3.Value());
    normalized.texture.tev_stages[0].sources_raw = first.sources_raw;
    bool remap_lights{};
    if (normalized.lighting.enable) {
        for (unsigned slot = 0; slot < normalized.lighting.src_num; ++slot) {
            remap_lights |= normalized.lighting.lights[slot].num != slot;
            normalized.lighting.lights[slot].num.Assign(slot);
        }
    }
    if (gas_accumulation) {
        normalized.framebuffer.alphablend_enable.Assign(0);
        normalized.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::None);
    }
    // Generate the common TEV/depth/alpha interface without entering the frozen
    // generator's unimplemented Gas branch. Native postsynthesis is added below.
    if (gas_postsynthesis) normalized.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::None);
    // A transferable config can contain fields derived for another host. Reset those fields
    // before applying our profile so old float logic-op or framebuffer snapshot code cannot leak in.
    normalized.framebuffer.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
    for (auto& border : normalized.texture.texture_border_color) border.raw = 0;
    using Texture = Pica::TexturingRegs::TextureConfig;
    if (config.texture.texture0_type == Texture::TextureCube ||
        config.texture.texture0_type == Texture::Shadow2D ||
        config.texture.texture0_type == Texture::ShadowCube) {
        // GenerateFragmentShader applies its profile after copying the config.
        // Disable its normalized 2D border prechecks for directional cube sampling
        // and the shadow functions' explicitly bounded integer reads.
        normalized.texture.requested_wrap[0] = {Texture::ClampToEdge, Texture::ClampToEdge};
    }
    auto source = Pica::Shader::Generator::GLSL::GenerateFragmentShader(normalized, user, profile);
    if (remap_lights) ApplyPicaLightBindings(source, config);
    ApplyPicaSamplers(source, config, user, samplers);
    if (gas_postsynthesis) ApplyGasPostsynthesis(source, config);
    if (gas_accumulation) ApplyGasAccumulation(source, config);
    if (config.framebuffer.shadow_rendering) {
        // Shadow rendering writes its storage image only. Avoid exporting an uninitialized color.
        const std::string output = "layout (location = 0) out vec4 color;";
        if (const auto position = source.find(output); position != std::string::npos) {
            source.erase(position, output.size());
        }
    }
    if (config.proctex.enable) {
        // At LOD 7 the shared mip interpolation samples level 8 even when its weight is zero.
        // Keep the native access inside the eight-entry PICA mip-offset table.
        const std::string lookup = "SampleProcTexColor(lut_coord, lod_i + 1)";
        if (const auto position = source.find(lookup); position != std::string::npos) {
            source.replace(position, lookup.size(), "SampleProcTexColor(lut_coord, min(lod_i + 1, 7))");
        }
    }
    auto shader = TranslateRasterizerFragmentGLSL(source, operations);
    return {FinalizeFragmentMSL(std::move(shader), config), operations};
}

TranslatedShader RestorePicaVertexShader(std::span<const std::uint32_t> words,
    ShaderStage stage, bool reset_machine) {
    if (stage != ShaderStage::VertexCompute && stage != ShaderStage::VertexMesh)
        throw std::invalid_argument("Invalid cached PICA vertex stage");
    return FinalizeVertexMSL(TranslateGuestSPIRV(words, stage, RasterizerShaderBindings()), reset_machine);
}

PicaFragmentProgram RestorePicaFragmentShader(std::span<const std::uint32_t> words,
    const Pica::Shader::FSConfig& config, const Pica::Shader::UserConfig& user,
    FragmentColorFormat color_format, bool gas_accumulation) {
    const auto operations = FragmentOperationState(config, user, color_format, gas_accumulation);
    return {FinalizeFragmentMSL(TranslateGuestSPIRV(words, ShaderStage::Fragment,
        RasterizerShaderBindings()), config), operations};
}

} // namespace Metal
