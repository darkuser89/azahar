// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_pica_shader_provider.h"
#include "mtl_pica_vertex_fixture.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        Check(argc == 2, "Specify a temporary directory for PICA shader fixtures");
        const std::filesystem::path output{argv[1]};
        std::filesystem::create_directories(output);
        Pica::RegsInternal regs{};
        regs.lighting.disable.Assign(1);
        regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
        auto config = Pica::Shader::FSConfig{regs};
        auto program = Metal::GeneratePicaFragmentShader(config, {}, Metal::FragmentColorFormat::RGBA8);
        bool found{};
        for (const auto& resource : program.shader.resources) {
            if (resource.kind == Metal::ShaderResourceKind::UniformBuffer && resource.metal_index == 2) {
                Check(resource.byte_size == sizeof(Pica::Shader::Generator::FSUniformData), "PICA FS uniform ABI mismatch");
                using Uniform = Pica::Shader::Generator::FSUniformData;
                const std::array expected{
                    std::pair{"framebuffer_scale", offsetof(Uniform, framebuffer_scale)},
                    std::pair{"alphatest_ref", offsetof(Uniform, alphatest_ref)},
                    std::pair{"depth_scale", offsetof(Uniform, depth_scale)},
                    std::pair{"depth_offset", offsetof(Uniform, depth_offset)},
                    std::pair{"lighting_lut_offset", offsetof(Uniform, lighting_lut_offset)},
                    std::pair{"fog_color", offsetof(Uniform, fog_color)},
                    std::pair{"proctex_noise_f", offsetof(Uniform, proctex_noise_f)},
                    std::pair{"proctex_noise_a", offsetof(Uniform, proctex_noise_a)},
                    std::pair{"proctex_noise_p", offsetof(Uniform, proctex_noise_p)},
                    std::pair{"lighting_global_ambient", offsetof(Uniform, lighting_global_ambient)},
                    std::pair{"light_src", offsetof(Uniform, light_src)},
                    std::pair{"const_color", offsetof(Uniform, const_color)},
                    std::pair{"tev_combiner_buffer_color", offsetof(Uniform, tev_combiner_buffer_color)},
                    std::pair{"tex_lod_bias", offsetof(Uniform, tex_lod_bias)},
                    std::pair{"tex_border_color", offsetof(Uniform, tex_border_color)},
                    std::pair{"blend_color", offsetof(Uniform, blend_color)},
                };
                for (const auto& [name, offset] : expected) {
                    bool matched{};
                    for (const auto& member : resource.members) matched |= member.name == name && member.offset == offset;
                    Check(matched, "Real PICA fragment uniform member offset mismatch");
                }
                found = true;
            }
        }
        Check(found, "Real PICA fragment generator lost its uniform block");
        std::size_t fixture_count{};
        const auto save = [&](const std::string& name, const Pica::Shader::FSConfig& variant,
                              Pica::Shader::UserConfig user = {}, const Metal::PicaSamplerConfigs& samplers = {}) {
            auto result = Metal::GeneratePicaFragmentShader(variant, user, Metal::FragmentColorFormat::RGBA8, samplers);
            const auto restored=Metal::RestorePicaFragmentShader(result.shader.spirv, variant, user,
                Metal::FragmentColorFormat::RGBA8, false);
            Check(restored.shader==result.shader && restored.operations==result.operations,
                "Canonical cached PICA fragment IR changed complete source/reflection/flags/operations");
            std::ofstream file{output / (name + ".metal")};
            file << result.shader.source;
            Check(file.good(), "Unable to save PICA fragment fixture");
            ++fixture_count;
            return result;
        };
        save("pica_default", config);
        using Tev = Pica::TexturingRegs::TevStageConfig;
        for (unsigned op = 0; op < 10; ++op) {
            auto variant = config;
            auto stage = static_cast<Tev>(variant.texture.tev_stages[0]);
            stage.color_op.Assign(static_cast<Tev::Operation>(op));
            stage.alpha_op.Assign(static_cast<Tev::Operation>(op == 6 || op == 7 ? 1 : op));
            variant.texture.tev_stages[0].ops_raw = stage.ops_raw;
            save("pica_tev_" + std::to_string(op), variant);
        }
        for (unsigned type = 0; type <= 5; ++type) {
            auto variant = config;
            variant.texture.texture0_type.Assign(static_cast<Pica::TexturingRegs::TextureConfig::TextureType>(type));
            auto stage = static_cast<Tev>(variant.texture.tev_stages[0]);
            stage.color_source1.Assign(Tev::Source::Texture0);
            stage.alpha_source1.Assign(Tev::Source::Texture0);
            for (auto& later : variant.texture.tev_stages) later.sources_raw = 0xF000F;
            variant.texture.tev_stages[0].sources_raw = stage.sources_raw;
            save("pica_texture_" + std::to_string(type), variant);
        }
        for (unsigned compare = 0; compare < 8; ++compare) {
            auto variant = config;
            variant.framebuffer.alpha_test_func.Assign(static_cast<Pica::FramebufferRegs::CompareFunc>(compare));
            save("pica_alpha_" + std::to_string(compare), variant);
        }
        for (unsigned unit = 0; unit < 3; ++unit) for (unsigned wrap = 0; wrap < 8; ++wrap)
            for (bool linear : {false, true}) {
                auto variant = config;
                variant.texture.texture0_type.Assign(Pica::TexturingRegs::TextureConfig::Texture2D);
                variant.texture.texture2_use_coord1.Assign(unit == 2);
                for (auto& later : variant.texture.tev_stages) later.sources_raw = 0xF000F;
                auto stage = static_cast<Tev>(variant.texture.tev_stages[0]);
                stage.color_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0) + unit));
                stage.alpha_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0) + unit));
                variant.texture.tev_stages[0].sources_raw = stage.sources_raw;
                Metal::PicaSamplerConfigs samplers{};
                samplers[unit] = {wrap, 1, unsigned(linear), unsigned(!linear), unsigned(linear), 0, 15};
                save("pica_sampler_" + std::to_string(unit) + "_" + std::to_string(wrap) + "_" +
                     std::to_string(linear), variant, {}, samplers);
                if (unit == 0) {
                    variant.texture.texture0_type.Assign(Pica::TexturingRegs::TextureConfig::Projection2D);
                    save("pica_sampler_project_" + std::to_string(wrap) + "_" + std::to_string(linear),
                         variant, {}, samplers);
                }
            }
        std::cout << "64 native PICA sampler fixtures: all 3 units, coord1 alias, projection, border/asymmetric modes, distinct min/mag and mip filters generated\n";
        auto lighting = config;
        lighting.lighting.enable.Assign(1);
        lighting.lighting.src_num.Assign(8);
        for (unsigned i = 0; i < 8; ++i) lighting.lighting.lights[i].num.Assign(i);
        save("pica_lighting", lighting);
        Pica::Shader::UserConfig custom_normal{};
        custom_normal.use_custom_normal.Assign(1);
        const auto normal = save("pica_normal", lighting, custom_normal);
        bool normal_binding{};
        for (const auto& resource : normal.shader.resources) normal_binding |= resource.descriptor_set == 2 && resource.binding == 1;
        Check(normal_binding, "Custom normal configuration lost its native texture binding");
        for(unsigned wrap=0;wrap<8;++wrap) for(bool linear:{false,true}) {
            auto variant=lighting;
            variant.texture.texture0_type.Assign(Pica::TexturingRegs::TextureConfig::Texture2D);
            for(auto& later:variant.texture.tev_stages) later.sources_raw=0xF000F;
            auto stage=static_cast<Tev>(variant.texture.tev_stages[0]);
            stage.color_source1.Assign(Tev::Source::PrimaryFragmentColor);
            stage.alpha_source1.Assign(Tev::Source::PrimaryFragmentColor);
            variant.texture.tev_stages[0].sources_raw=stage.sources_raw;
            Metal::PicaSamplerConfigs samplers{};
            samplers[0]={wrap,1,unsigned(linear),unsigned(!linear),unsigned(linear),0,15};
            auto result=save("pica_sampler_normal_"+std::to_string(wrap)+"_"+std::to_string(linear),
                             variant,custom_normal,samplers);
            Check(result.shader.source.find("metalSampleNormal")!=std::string::npos,
                  "Native custom-normal sampler fixture did not contain the per-tap path");
        }
        std::cout<<"16 native custom-normal sampler fixtures with per-tap border/asymmetric modes and distinct filters generated\n";
        auto fog = config;
        fog.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Fog);
        save("pica_fog", fog);
        auto wbuffer = config;
        wbuffer.framebuffer.depthmap_enable.Assign(Pica::RasterizerRegs::DepthBuffering::WBuffering);
        save("pica_wbuffer", wbuffer);
        auto shadow = config;
        shadow.framebuffer.shadow_rendering.Assign(1);
        auto shadow_program = save("pica_shadow", shadow);
        Check(shadow_program.shader.source.find("raster_order_group(0)") != std::string::npos,
              "Real PICA shadow shader lost ordered image updates");
        for (unsigned filter = 0; filter < 6; ++filter) {
            auto variant = config;
            variant.proctex.enable.Assign(1);
            variant.proctex.noise_enable.Assign(1);
            variant.proctex.lut_width = 128;
            variant.proctex.lod_max = 7;
            variant.proctex.lut_filter.Assign(static_cast<Pica::TexturingRegs::ProcTexFilter>(filter));
            auto stage = static_cast<Tev>(variant.texture.tev_stages[0]);
            stage.color_source1.Assign(Tev::Source::Texture3);
            variant.texture.tev_stages[0].sources_raw = stage.sources_raw;
            save("pica_proctex_" + std::to_string(filter), variant);
        }
        auto blended = config;
        blended.framebuffer.alphablend_enable.Assign(1);
        blended.framebuffer.requested_rgb_blend = {Pica::FramebufferRegs::BlendEquation::Min,
            Pica::FramebufferRegs::BlendFactor::ConstantColor, Pica::FramebufferRegs::BlendFactor::DestColor};
        blended.framebuffer.requested_alpha_blend = {Pica::FramebufferRegs::BlendEquation::Max,
            Pica::FramebufferRegs::BlendFactor::SourceAlpha, Pica::FramebufferRegs::BlendFactor::ConstantAlpha};
        auto blend_program = save("pica_minmax", blended);
        Check(blend_program.operations.rgb.enabled && blend_program.operations.alpha.enabled &&
              blend_program.shader.uses_framebuffer_fetch, "PICA Min/Max blending lost native framebuffer fetch");
        for (const auto& resource : blend_program.shader.resources) {
            Check(!(resource.descriptor_set == 1 && resource.binding == 3) && resource.metal_index != 22,
                  "PICA blending added a framebuffer snapshot or duplicate blend constants");
        }
        auto gas = config;
        gas.texture.fog_mode.Assign(Pica::TexturingRegs::FogMode::Gas);
        const auto gas_program = save("pica_gas_native_postsynthesis", gas);
        Check(gas_program.shader.source.find("[[depth(") != std::string::npos,
              "Native Gas postsynthesis omitted fragment depth");
        Pica::ShaderSetup vertex_setup;
        Metal::Test::MovVertex(regs, vertex_setup);
        auto vertex = Metal::GeneratePicaVertexShader(regs, vertex_setup);
        Check(Metal::RestorePicaVertexShader(vertex.spirv,Metal::ShaderStage::VertexCompute,false)==vertex,
            "Canonical cached PICA vertex IR changed complete source/reflection/flags");
        Check(vertex.stage == Metal::ShaderStage::VertexCompute &&
              vertex.source.find("[[stage_in]]") == std::string::npos,
              "Real PICA vertex program did not use explicit native compute capture");
        std::ofstream vertex_file{output / "pica_vertex_mov.metal"};
        vertex_file << vertex.source;
        Check(vertex_file.good(), "Unable to save PICA vertex fixture");
        constexpr std::array arithmetic_ops{nihstro::OpCode::Id::ADD, nihstro::OpCode::Id::MUL,
            nihstro::OpCode::Id::DP3, nihstro::OpCode::Id::DP4, nihstro::OpCode::Id::DPH,
            nihstro::OpCode::Id::MIN, nihstro::OpCode::Id::MAX, nihstro::OpCode::Id::FLR,
            nihstro::OpCode::Id::RCP, nihstro::OpCode::Id::RSQ, nihstro::OpCode::Id::EX2,
            nihstro::OpCode::Id::LG2, nihstro::OpCode::Id::SGE, nihstro::OpCode::Id::SLT};
        for (auto op : arithmetic_ops) {
            vertex_setup.UpdateProgramCode(2, Metal::Test::Arithmetic(op, 2, 2, 3));
            auto arithmetic = Metal::GeneratePicaVertexShader(regs, vertex_setup);
            Check(Metal::RestorePicaVertexShader(arithmetic.spirv,Metal::ShaderStage::VertexCompute,false)==arithmetic,
                "Canonical cached arithmetic IR changed complete source/reflection/flags");
            std::ofstream file{output / ("pica_vertex_op_" + std::to_string(unsigned(op)) + ".metal")};
            file << arithmetic.source;
            Check(file.good(), "Unable to save real PICA arithmetic fixture");
        }
        bool rejected_entry{};
        try {
            auto bad_regs = regs;
            bad_regs.vs.main_offset.Assign(4096);
            Metal::GeneratePicaVertexShader(bad_regs, vertex_setup);
        } catch (const std::invalid_argument&) { rejected_entry = true; }
        Check(rejected_entry, "Invalid PICA vertex entry point reached the decompiler");
        bool rejected_block{};
        vertex_setup.UpdateProgramCode(0, (unsigned(nihstro::OpCode::Id::CALL) << 26U) | (4095U << 10U) | 255U);
        try { Metal::GeneratePicaVertexShader(regs, vertex_setup); }
        catch (const std::invalid_argument&) { rejected_block = true; }
        Check(rejected_block, "A PICA call block outside program storage reached the decompiler");
        Metal::Test::MovVertex(regs, vertex_setup);
        bool rejected_semantic{};
        auto bad_semantic_regs = regs;
        bad_semantic_regs.rasterizer.vs_output_attributes[0].map_x.Assign(
            static_cast<Pica::RasterizerRegs::VSOutputAttributes::Semantic>(24));
        try { Metal::GeneratePicaVertexShader(bad_semantic_regs, vertex_setup); }
        catch (const std::invalid_argument&) { rejected_semantic = true; }
        Check(rejected_semantic, "An invalid output semantic was silently used as a native constant");
        vertex_setup.UpdateProgramCode(0, std::uint32_t(nihstro::OpCode::Id::EMIT) << 26U);
        bool rejected_emission{};
        try { Metal::GeneratePicaVertexShader(regs, vertex_setup); }
        catch (const std::runtime_error&) { rejected_emission = true; }
        Check(rejected_emission, "Guest geometry emission was silently omitted by the vertex provider");
        std::cout << "Real PICA MOV vertex program and entry/emission validation passed\n";
        std::cout << fixture_count << " real PICA fragment variants, TEV, textures, alpha tests, lighting, normals, fog, shadows, procedural textures, Min/Max and full uniform ABI passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
