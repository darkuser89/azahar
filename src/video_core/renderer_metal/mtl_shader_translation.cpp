// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_shader_translation.h"
#include "mtl_translation_cache.h"
#include "mtl_shader_text.h"
#include "mtl_input_reference_shader.h"
#include "mtl_mesh_shader.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv_msl.hpp>

namespace Metal {

namespace {

constexpr std::size_t MaxShaderBytes = 8 * 1024 * 1024;
thread_local TranslationInputScope* input_scope{};

struct CompilerTranslations {
    TranslationCache cache;
    std::string profile;

    CompilerTranslations() {
        if (const auto* path = std::getenv("AZAHAR_METAL4_PROFILE"); path && *path)
            profile = std::string{path} + ".translations.csv";
    }

    ~CompilerTranslations() noexcept {
        if (profile.empty()) return;
        try {
            const auto stats = cache.Stats();
            std::ofstream out{profile, std::ios::trunc};
            out << "glsl_lookups,glsl_hits,glsl_misses,spirv_lookups,spirv_hits,spirv_misses,"
                   "entries,retained_payload_bytes,evictions,unretained,duplicate_compiles\n"
                << stats.lookups[0] << ',' << stats.hits[0] << ',' << stats.misses[0] << ','
                << stats.lookups[1] << ',' << stats.hits[1] << ',' << stats.misses[1] << ','
                << stats.entries << ',' << stats.retained_payload_bytes << ','
                << stats.evictions << ',' << stats.unretained << ',' << stats.duplicate_compiles
                << '\n';
        } catch (...) {} // Optional process-wide diagnostics must never affect compilation.
    }
};

TranslationCache& CompilerCache() {
    static CompilerTranslations translations;
    return translations.cache;
}

constexpr std::array RasterizerBindings{
    ShaderBindingRemap{0, 0, 0}, ShaderBindingRemap{0, 1, 1}, ShaderBindingRemap{0, 2, 2},
    ShaderBindingRemap{0, 3, NoShaderBinding, 0},
    ShaderBindingRemap{0, 4, NoShaderBinding, 1},
    ShaderBindingRemap{0, 5, NoShaderBinding, 2},
    ShaderBindingRemap{1, 0, NoShaderBinding, 3, 0},
    ShaderBindingRemap{1, 1, NoShaderBinding, 9, 6},
    ShaderBindingRemap{1, 2, NoShaderBinding, 10, 7},
    ShaderBindingRemap{1, 3, NoShaderBinding, 11, 8},
    ShaderBindingRemap{2, 0, NoShaderBinding, 12},
    ShaderBindingRemap{2, 1, NoShaderBinding, 13, 9},
    ShaderBindingRemap{2, 2, NoShaderBinding, 14, 10}, // Immutable Gas scene depth.
    // Compute assembly may provide per-primitive data to the fragment stage.
    ShaderBindingRemap{3, 0, 16},
    ShaderBindingRemap{3, 1, 17}, ShaderBindingRemap{3, 2, 18}, ShaderBindingRemap{3, 3, 19},
    ShaderBindingRemap{3, 4, 22}, // Dynamic blend constants for standalone fragment operations.
    ShaderBindingRemap{3, 5, 23}, // Raw Gas LUT and postsynthesis registers.
    ShaderBindingRemap{3, 6, 24}, // Atomic Gas maximum, shared by all fragments.
    ShaderBindingRemap{4, 0, 3}, ShaderBindingRemap{4, 1, 20},
    ShaderBindingRemap{4, 2, 25}, ShaderBindingRemap{4, 3, 26},
    ShaderBindingRemap{5, 4, 17}, // Immutable physical input references, mesh/materializer only.
    ShaderBindingRemap{spirv_cross::kPushConstDescSet, spirv_cross::kPushConstBinding, 21},
};

EShLanguage GLSLStage(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return EShLangVertex;
    case ShaderStage::Fragment:
        return EShLangFragment;
    case ShaderStage::Compute:
    case ShaderStage::VertexCompute:
        return EShLangCompute;
    case ShaderStage::VertexMesh: return EShLangMesh;
    }
    throw std::invalid_argument("Unsupported Metal shader stage");
}

spv::ExecutionModel SPIRVStage(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return spv::ExecutionModelVertex;
    case ShaderStage::Fragment:
        return spv::ExecutionModelFragment;
    case ShaderStage::Compute:
    case ShaderStage::VertexCompute:
        return spv::ExecutionModelGLCompute;
    case ShaderStage::VertexMesh: return spv::ExecutionModelMeshEXT;
    }
    throw std::invalid_argument("Unsupported Metal shader stage");
}

const char* EntryName(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return "azahar_vertex";
    case ShaderStage::Fragment:
        return "azahar_fragment";
    case ShaderStage::Compute:
        return "azahar_compute";
    case ShaderStage::VertexCompute:
        return "azahar_vertex_compute";
    case ShaderStage::VertexMesh: return "azahar_vertex_mesh";
    }
    throw std::invalid_argument("Unsupported Metal shader stage");
}

const ShaderBindingRemap& FindBinding(std::span<const ShaderBindingRemap> bindings,
                                     std::uint32_t set, std::uint32_t binding) {
    const ShaderBindingRemap* found = nullptr;
    for (const auto& candidate : bindings) {
        if (candidate.descriptor_set == set && candidate.binding == binding) {
            if (found) {
                throw std::invalid_argument("Duplicate Metal shader descriptor mapping");
            }
            found = &candidate;
        }
    }
    if (!found) {
        throw std::invalid_argument("Metal shader resource has no explicit binding mapping");
    }
    return *found;
}

template <std::size_t Size>
void Claim(std::array<bool, Size>& slots, std::uint32_t index, std::uint32_t count) {
    if (!count || index >= Size || count > Size - index) {
        throw std::invalid_argument("Metal shader resource exceeds its binding table");
    }
    for (std::uint32_t i = index; i < index + count; ++i) {
        if (slots[i]) {
            throw std::invalid_argument("Metal shader resources overlap in the binding table");
        }
        slots[i] = true;
    }
}

std::uint32_t BufferLayoutSize(std::uint64_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Guest buffer layout exceeds its reflection ABI");
    }
    return static_cast<std::uint32_t>(size);
}

std::uint32_t BufferStructSize(const spirv_cross::CompilerMSL& compiler,
                               const spirv_cross::SPIRType& structure, unsigned depth);

std::uint32_t BufferMemberSize(const spirv_cross::CompilerMSL& compiler,
                               const spirv_cross::SPIRType& structure, std::uint32_t index,
                               unsigned depth) {
    const auto& type = compiler.get_type(structure.member_types[index]);
    if (!type.array.empty()) {
        auto array_id = structure.member_types[index];
        for (unsigned nesting = 0; !compiler.get_type(array_id).array.empty(); ++nesting) {
            const auto& array = compiler.get_type(array_id);
            if (nesting > 64 || !array.array_size_literal.back()) {
                throw std::invalid_argument("Guest buffer layout requires bounded literal arrays");
            }
            BufferLayoutSize(std::uint64_t(compiler.get_decoration(array_id, spv::DecorationArrayStride)) *
                             array.array.back());
            array_id = array.parent_type;
        }
        if (type.basetype == spirv_cross::SPIRType::Struct) {
            BufferStructSize(compiler, type, depth + 1);
        }
        // SPIRV-Cross multiplies these two uint32 values before widening to size_t.
        return BufferLayoutSize(std::uint64_t(compiler.type_struct_member_array_stride(structure, index)) *
                                type.array.back());
    }
    if (type.basetype == spirv_cross::SPIRType::Struct) {
        return BufferStructSize(compiler, type, depth + 1);
    }
    if (type.columns > 1) {
        const auto stride = compiler.type_struct_member_matrix_stride(structure, index);
        const auto flags = compiler.get_member_decoration_bitset(structure.self, index);
        const auto vectors = flags.get(spv::DecorationRowMajor) ? type.vecsize : type.columns;
        return BufferLayoutSize(std::uint64_t(stride) * vectors);
    }
    return BufferLayoutSize(compiler.get_declared_struct_member_size(structure, index));
}

std::uint32_t BufferStructSize(const spirv_cross::CompilerMSL& compiler,
                               const spirv_cross::SPIRType& structure, unsigned depth) {
    if (depth > 64 || structure.member_types.empty()) {
        throw std::invalid_argument("Guest buffer layout is empty or too deeply nested");
    }
    std::uint64_t size{};
    for (std::uint32_t i = 0; i < structure.member_types.size(); ++i) {
        const auto end = std::uint64_t(compiler.type_struct_member_offset(structure, i)) +
                         BufferMemberSize(compiler, structure, i, depth);
        size = std::max(size, end);
    }
    return BufferLayoutSize(size);
}

std::string MakeVertexComputeSource(std::string_view source) {
    // The PICA generator emits scalar/vector attributes with explicit locations. Turn those
    // into globals loaded from canonical float4 registers, without legacy compute stage-in.
    static const std::regex interface_pattern{
        R"(layout\s*\(\s*location\s*=\s*([0-9]+)\s*\)\s*(in|out)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;)"};
    std::string input{source};
    static const std::regex position_pattern{R"(\bgl_Position\s*=)"};
    std::cmatch match;
    std::size_t position{};
    if (!ShaderTextMatcher{input, position_pattern, {"gl_Position"}}.Next(position, match)) {
        throw std::invalid_argument("Native PICA vertex capture requires a complete position output");
    }
    std::string body;
    std::string load;
    std::string store;
    std::size_t consumed{};
    std::array<bool, 16> input_locations{};
    std::array<bool, 8> output_locations{};
    ShaderTextMatcher interfaces{input, interface_pattern, {"layout"}};
    while (interfaces.Next(position, match)) {
        body.append(input, consumed, position - consumed);
        const auto location = static_cast<std::uint32_t>(std::stoul(match[1].str()));
        const auto direction = match[2].str();
        const auto type = match[3].str();
        const auto name = match[4].str();
        body += type + " " + name + ";";
        std::size_t components{};
        if (type == "float" || type == "int" || type == "uint") {
            components = 1;
        } else if ((type.starts_with("vec") || type.starts_with("ivec") || type.starts_with("uvec")) &&
                   type.back() >= '2' && type.back() <= '4') {
            components = type.back() - '0';
        } else {
            throw std::invalid_argument("PICA compute capture requires scalar or vector attributes");
        }
        constexpr std::array swizzles{"x", "xy", "xyz", "xyzw"};
        if (direction == "in") {
            if (location >= 16 || input_locations[location]) {
                throw std::invalid_argument("Invalid or duplicate PICA input register");
            }
            input_locations[location] = true;
            load += name + " = " + type + "(azahar_inputs[azahar_index * 16u + " +
                    std::to_string(location) + "u]." + swizzles[components - 1] + ");\n";
        } else {
            if (location == 0 || location >= 8 || output_locations[location]) {
                throw std::invalid_argument("Invalid or duplicate PICA output semantic");
            }
            output_locations[location] = true;
            constexpr std::array fields{"position", "color", "texcoord0", "texcoord1", "texcoord2",
                                         "clip", "quaternion", "view"};
            if (location == 5) {
                if (components != 1) {
                    throw std::invalid_argument("PICA texture coordinate w must be a scalar");
                }
                store += "azahar_out.clip.z = float(" + name + ");\n";
            } else {
                store += "azahar_out." + std::string(fields[location]) + "." +
                         swizzles[components - 1] + " = " +
                         (components == 1 ? "float" : "vec" + std::to_string(components)) +
                         "(" + name + ");\n";
            }
        }
        consumed = position + static_cast<std::size_t>(match.length());
    }
    body.append(input, consumed, input.size() - consumed);
    static const std::regex per_vertex_pattern{R"(out\s+gl_PerVertex\s*\{[^}]*\}\s*;)"};
    body = ReplaceShaderText(body, per_vertex_pattern, {"out"}, "");
    static const std::regex main_pattern{R"(void\s+main\s*\(\s*\))"};
    ShaderTextMatcher mains{body, main_pattern, {"void"}};
    unsigned main_count{};
    while (mains.Next(position, match)) ++main_count;
    if (main_count != 1) {
        throw std::invalid_argument("PICA vertex source must have one main entry point");
    }
    body = ReplaceShaderText(body, main_pattern, {"void"}, "void azahar_guest_vertex()");
    static const std::regex vertex_index_pattern{R"(\bgl_VertexIndex\b)"};
    static const std::regex instance_index_pattern{R"(\bgl_InstanceIndex\b)"};
    body = ReplaceShaderText(body, vertex_index_pattern, {"gl_VertexIndex"},
                              "int(gl_GlobalInvocationID.x + azahar_capture.first_vertex)");
    body = ReplaceShaderText(body, instance_index_pattern, {"gl_InstanceIndex"},
                              "int(azahar_capture.instance_index)");
    // Insert after leading directives so #version and extension declarations retain their order.
    std::size_t insertion{};
    while (insertion < body.size()) {
        const auto line_end = body.find('\n', insertion);
        const auto line = std::string_view{body}.substr(insertion,
            (line_end == std::string::npos ? body.size() : line_end) - insertion);
        const auto start = line.find_first_not_of(" \t\r");
        if (start != std::string_view::npos && line[start] != '#' && !line.substr(start).starts_with("//")) {
            break;
        }
        if (line_end == std::string::npos) {
            insertion = body.size();
            break;
        }
        insertion = line_end + 1;
    }
    constexpr std::string_view capture = R"(
layout(local_size_x=64) in;
struct AzaharCapturedVertex {
    vec4 position;
    vec4 color;
    vec4 texcoord0;
    vec4 texcoord1;
    vec4 texcoord2;
    vec4 quaternion;
    vec4 view;
    vec4 clip;
};
layout(set=3, binding=1, std430) readonly buffer AzaharCaptureInputs { vec4 azahar_inputs[]; };
layout(set=3, binding=2, std430) writeonly buffer AzaharCaptureOutputs { AzaharCapturedVertex azahar_outputs[]; };
layout(set=3, binding=3, std140) uniform AzaharCaptureParams {
    uint vertex_count;
    uint first_vertex;
    uint instance_index;
    uint padding;
} azahar_capture;
vec4 gl_Position;
float gl_ClipDistance[2];
)";
    body.insert(insertion, capture);
    body += "\nvoid main() {\nuint azahar_index = gl_GlobalInvocationID.x;\n"
            "if (azahar_index >= azahar_capture.vertex_count) return;\n"
            "gl_Position = vec4(0.0, 0.0, 2.0, 1.0);\n"
            "gl_ClipDistance[0] = 0.0; gl_ClipDistance[1] = 0.0;\n";
    body += load + "azahar_guest_vertex();\n";
    body += R"(
AzaharCapturedVertex azahar_out;
azahar_out.position = gl_Position;
azahar_out.color = vec4(1.0);
azahar_out.texcoord0 = vec4(0.0);
azahar_out.texcoord1 = vec4(0.0);
azahar_out.texcoord2 = vec4(0.0);
azahar_out.quaternion = vec4(0.0, 0.0, 0.0, 1.0);
azahar_out.view = vec4(0.0);
azahar_out.clip = vec4(gl_ClipDistance[0], gl_ClipDistance[1], 0.0, 0.0);
)";
    body += store + "azahar_outputs[azahar_index] = azahar_out;\n}\n";
    static const std::regex position_name_pattern{R"(\bgl_Position\b)"};
    static const std::regex clip_name_pattern{R"(\bgl_ClipDistance\b)"};
    body = ReplaceShaderText(body, position_name_pattern, {"gl_Position"}, "azahar_position");
    body = ReplaceShaderText(body, clip_name_pattern, {"gl_ClipDistance"}, "azahar_clip_distance");
    return body;
}

std::string MakeVertexMeshSource(std::string_view source) {
 auto text=MakeVertexComputeSource(source);
 const auto replace=[&](std::string_view from,std::string_view to) {
  const auto at=text.find(from);if(at==std::string::npos)throw std::logic_error("Missing mesh conversion boundary: "+std::string(from));
  text.replace(at,from.size(),to);
 };
 replace("layout(local_size_x=64) in;","layout(local_size_x=96) in;\nlayout(triangles,max_vertices=96,max_primitives=32) out;");
 const auto version=text.find('\n');text.insert(version+1,"#extension GL_EXT_mesh_shader : require\n");
 replace("void main() {\nuint azahar_index = gl_GlobalInvocationID.x;\nif (azahar_index >= azahar_capture.vertex_count) return;",
         "AzaharCapturedVertex azahar_evaluate(uint azahar_index) {");
 replace("azahar_outputs[azahar_index] = azahar_out;","return azahar_out;");
 // VertexIndex must follow the selected input index, rather than mesh workgroup geometry.
 static const std::regex invocation_pattern{R"(gl_GlobalInvocationID\.x)"};
 text=ReplaceShaderText(text,invocation_pattern,{"gl_GlobalInvocationID.x"},"azahar_selected_index");
 const auto at=text.find("vec4 azahar_position;");
 if(at==std::string::npos)throw std::logic_error("Missing mesh position declaration");
 text.insert(at,"uint azahar_selected_index;\n");
 replace("AzaharCapturedVertex azahar_evaluate(uint azahar_index) {",
         "AzaharCapturedVertex azahar_evaluate(uint azahar_index) {\nazahar_selected_index=azahar_index;");
 // History maintenance may evaluate vertices before this thread's raster vertex.
 // Every evaluation has a fresh machine, including output and control registers.
 std::string resets;
 static const std::regex reset_pattern{R"(\b(vec4|bvec2|ivec3)\s+((?:reg_tmp|vs_out_attr)[0-9]+|conditional_code|address_registers)\s*=\s*([^;]+);)"};
 ShaderTextMatcher reset_matches{text,reset_pattern,{"vec4","bvec2","ivec3"}};
 std::cmatch reset_match;std::size_t reset_position{};
 while(reset_matches.Next(reset_position,reset_match))
  resets+=reset_match[2].str()+" = "+reset_match[3].str()+";\n";
 const auto evaluation=text.find("azahar_selected_index=azahar_index;");
 text.insert(evaluation+std::string_view("azahar_selected_index=azahar_index;").size(),"\n"+resets);
 text+=VertexMeshShaderGLSL;

 {
  text.insert(text.find('\n')+1,R"EXT(
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
)EXT");
  static const std::regex input_pattern{R"(azahar_inputs\[azahar_index \* 16u \+ ([0-9]+)u\])"};
  text=ReplaceShaderText(text,input_pattern,{"azahar_inputs["},"mesh_load_register($1u,azahar_index)");
  const std::string declaration="layout(set=3, binding=1, std430) readonly buffer AzaharCaptureInputs { vec4 azahar_inputs[]; };";
  const auto begin=text.find(declaration);if(begin==std::string::npos)throw std::logic_error("Missing canonical input declaration");
  text.erase(begin,declaration.size());
  text.insert(text.find("uint azahar_selected_index;"),InputReferenceShaderGLSL);
  const std::string old="if(triangle==0u && local==0u)mesh_write_history();";
  const auto history=text.find(old);if(history==std::string::npos)throw std::logic_error("Missing history entry");
  text.replace(history,old.size(),"if(mesh_params.reserved0==0u && triangle==0u && local==0u)mesh_write_history();");
 }

 return text;
}

} // namespace

std::span<const ShaderBindingRemap> RasterizerShaderBindings() {
    return RasterizerBindings;
}

std::vector<std::uint32_t> CompileGuestGLSL(std::string_view source, ShaderStage stage) {
    if (source.empty() || source.size() > MaxShaderBytes) {
        throw std::invalid_argument("Guest shader source is empty or exceeds its size limit");
    }
    static std::once_flag initialize;
    std::call_once(initialize, [] {
        if (!glslang::InitializeProcess()) {
            throw std::runtime_error("Unable to initialize the guest shader compiler");
        }
        std::atexit([] { glslang::FinalizeProcess(); });
    });
    // Keep source storage alive for the entire parse and link operation.
    std::string text = stage == ShaderStage::VertexMesh ? MakeVertexMeshSource(source) : stage == ShaderStage::VertexCompute ? MakeVertexComputeSource(source)
                                                          : std::string{source};
    if (text.find("#version") == std::string::npos) {
        text.insert(0, "#version 450\n");
    }
    const char* source_ptr = text.data();
    const int source_length = static_cast<int>(text.size());
    const auto language = GLSLStage(stage);
    glslang::TShader shader{language};
    shader.setStringsWithLengths(&source_ptr, &source_length, 1);
    // This selects a SPIR-V source dialect only; GPU execution is native Metal 4.
    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 450);
    shader.setEnvClient(glslang::EShClientVulkan, stage == ShaderStage::VertexMesh ? glslang::EShTargetVulkan_1_2 : glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EShTargetSpv, stage == ShaderStage::VertexMesh ? glslang::EShTargetSpv_1_5 : glslang::EShTargetSpv_1_3);
    const auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    glslang::TShader::ForbidIncluder includer;
    if (!shader.parse(GetDefaultResources(), 450, ECoreProfile, false, true, messages, includer)) {
        throw std::runtime_error(std::string("Guest shader parsing failed: ") + shader.getInfoLog() +
                                 shader.getInfoDebugLog());
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        throw std::runtime_error(std::string("Guest shader linking failed: ") + program.getInfoLog());
    }
    auto* intermediate = program.getIntermediate(language);
    if (!intermediate) {
        throw std::runtime_error("Guest shader compiler returned no intermediate program");
    }
    std::vector<std::uint32_t> words;
    glslang::SpvOptions options;
    options.disableOptimizer = true;
    options.optimizeSize = false;
    spv::SpvBuildLogger logger;
    glslang::GlslangToSpv(*intermediate, words, &logger, &options);
    if (words.empty() || words.size() > MaxShaderBytes / sizeof(std::uint32_t)) {
        throw std::runtime_error("Guest shader compiler returned invalid or oversized SPIR-V");
    }
    return words;
}

static TranslatedShader TranslateGuestSPIRVUncached(std::span<const std::uint32_t> words, ShaderStage stage,
                                     std::span<const ShaderBindingRemap> bindings) {
    if (words.size() < 5 || words.size() > MaxShaderBytes / sizeof(std::uint32_t) ||
        words[0] != spv::MagicNumber) {
        throw std::invalid_argument("Invalid guest SPIR-V program header or size");
    }
    spirv_cross::CompilerMSL compiler{words.data(), words.size()};
    const auto model = SPIRVStage(stage);
    compiler.set_entry_point("main", model);
    compiler.rename_entry_point("main", EntryName(stage), model);
    auto options = compiler.get_msl_options();
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.set_msl_version(4, 0);
    options.argument_buffers = false;
    options.texture_buffer_native = true;
    options.use_framebuffer_fetch_subpasses = true;
    options.buffer_size_buffer_index = ShaderBufferSizesBinding;
    options.shader_input_buffer_index = 17;
    options.shader_output_buffer_index = 18;
    options.indirect_params_buffer_index = 19;
    options.shader_index_buffer_index = 20;
    compiler.set_msl_options(options);
    auto common = compiler.get_common_options();
    common.vertex.fixup_clipspace = false;
    common.vertex.flip_vert_y = true;
    compiler.set_common_options(common);

    const auto resources = compiler.get_shader_resources();
    const auto active_resources = compiler.get_active_interface_variables();
    TranslatedShader result{.stage = stage, .source = {}, .entry_point = {}, .resources = {}};
    result.spirv.assign(words.begin(), words.end());
    std::array<bool, 31> buffer_slots{};
    std::array<bool, 128> texture_slots{};
    std::array<bool, 16> sampler_slots{};
    const auto add_resources = [&](const auto& list, ShaderResourceKind kind, bool combined_sampler) {
        for (const auto& resource : list) {
            const auto set = kind == ShaderResourceKind::UniformBuffer &&
                                     compiler.get_storage_class(resource.id) == spv::StorageClassPushConstant
                                 ? spirv_cross::kPushConstDescSet
                                 : compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
            const auto binding = set == spirv_cross::kPushConstDescSet
                                     ? spirv_cross::kPushConstBinding
                                     : compiler.get_decoration(resource.id, spv::DecorationBinding);
            const auto& map = FindBinding(bindings, set, binding);
            const auto& type = compiler.get_type(resource.type_id);
            std::uint32_t count = 1;
            for (std::size_t i = 0; i < type.array.size(); ++i) {
                if (!type.array_size_literal[i] || !type.array[i] || type.array[i] > 128 / count) {
                    throw std::invalid_argument("Guest descriptor arrays must have bounded literal sizes");
                }
                count *= type.array[i];
            }
            const bool buffer = kind == ShaderResourceKind::UniformBuffer || kind == ShaderResourceKind::StorageBuffer;
            const bool texture = kind == ShaderResourceKind::Texture || kind == ShaderResourceKind::StorageTexture;
            const bool sampler = kind == ShaderResourceKind::Sampler ||
                                 (combined_sampler && type.image.dim != spv::DimBuffer);
            const auto index = buffer ? map.buffer : texture ? map.texture : map.sampler;
            if (buffer) {
                Claim(buffer_slots, index, count);
            } else if (texture) {
                Claim(texture_slots, index, count);
            }
            if (sampler) {
                Claim(sampler_slots, map.sampler, count);
            }
            spirv_cross::MSLResourceBinding remap{};
            remap.stage = model;
            remap.desc_set = set;
            remap.binding = binding;
            remap.count = count;
            remap.msl_buffer = map.buffer;
            remap.msl_texture = map.texture;
            remap.msl_sampler = map.sampler;
            compiler.add_msl_resource_binding(remap);
            ShaderResource info{.name = resource.name, .kind = kind, .descriptor_set = set,
                                .binding = binding, .metal_index = index,
                                .sampler_index = sampler ? map.sampler : NoShaderBinding,
                                .count = count, .byte_size = 0, .runtime_array_stride = 0, .members = {}};
            info.active = active_resources.contains(resource.id);
            if (buffer) {
                const auto& structure = compiler.get_type(resource.base_type_id);
                // Runtime-sized storage tails have no fixed total size, but their fixed prefix does.
                info.byte_size = BufferStructSize(compiler, structure, 0);
                for (std::uint32_t i = 0; i < structure.member_types.size(); ++i) {
                    const auto& member_type = compiler.get_type(structure.member_types[i]);
                    if (!member_type.array.empty() && member_type.array.back() == 0) {
                        info.runtime_array_stride = compiler.type_struct_member_array_stride(structure, i);
                        continue;
                    }
                    info.members.push_back({compiler.get_member_name(structure.self, i),
                        compiler.type_struct_member_offset(structure, i),
                        BufferMemberSize(compiler, structure, i, 0)});
                }
            }
            result.resources.push_back(std::move(info));
        }
    };
    add_resources(resources.uniform_buffers, ShaderResourceKind::UniformBuffer, false);
    add_resources(resources.storage_buffers, ShaderResourceKind::StorageBuffer, false);
    add_resources(resources.push_constant_buffers, ShaderResourceKind::UniformBuffer, false);
    add_resources(resources.sampled_images, ShaderResourceKind::Texture, true);
    add_resources(resources.separate_images, ShaderResourceKind::Texture, false);
    add_resources(resources.storage_images, ShaderResourceKind::StorageTexture, false);
    add_resources(resources.separate_samplers, ShaderResourceKind::Sampler, false);
    for (const auto& input : resources.subpass_inputs) {
        const auto& type = compiler.get_type(input.type_id);
        if (stage != ShaderStage::Fragment || resources.subpass_inputs.size() != 1 ||
            !compiler.has_decoration(input.id, spv::DecorationInputAttachmentIndex) ||
            compiler.get_decoration(input.id, spv::DecorationInputAttachmentIndex) != 0 ||
            !type.array.empty() || type.image.ms ||
            compiler.get_type(type.image.type).basetype != spirv_cross::SPIRType::Float) {
            throw std::invalid_argument("Native framebuffer fetch requires one float color attachment at index zero");
        }
        // This is a tile-memory input, not a sampled texture or an argument-table resource.
        result.uses_framebuffer_fetch = true;
    }
    if (!resources.acceleration_structures.empty() ||
        !resources.atomic_counters.empty()) {
        throw std::invalid_argument("Guest shader uses a resource unsupported by the native binding layout");
    }
    result.source = compiler.compile();
    if (result.uses_framebuffer_fetch) {
        // SPIRV-Cross emits framebuffer color parameters without a raster order group. Both
        // the input and output must participate so overlapping primitives observe prior writes.
        static const std::regex color_attribute{R"(\[\[color\(0\)\]\])"};
        result.source = ReplaceShaderText(result.source, color_attribute, {"[[color(0)]]"},
                                         "[[color(0), raster_order_group(0)]]");
    }
    result.entry_point = compiler.get_cleansed_entry_point_name(EntryName(stage), model);
    result.needs_buffer_sizes = compiler.needs_buffer_size_buffer();
    if (result.needs_buffer_sizes && buffer_slots[ShaderBufferSizesBinding]) {
        throw std::invalid_argument("Guest shader overlaps the Metal buffer-size binding");
    }
    return result;
}

TranslatedShader TranslateGuestSPIRV(std::span<const std::uint32_t> words, ShaderStage stage,
                                     std::span<const ShaderBindingRemap> bindings) {
    if (words.size() < 5 || words.size() > MaxShaderBytes / sizeof(std::uint32_t) ||
        words[0] != spv::MagicNumber) {
        throw std::invalid_argument("Invalid guest SPIR-V program header or size");
    }
    const std::string_view bytes{reinterpret_cast<const char*>(words.data()), words.size_bytes()};
    return CompilerCache().Get({TranslationInputKind::SPIRV, stage, bytes, bindings}, [&] {
        return TranslateGuestSPIRVUncached(words, stage, bindings);
    });
}

TranslationInputScope::TranslationInputScope(Recorder recorder_)
    : recorder{std::move(recorder_)},previous{input_scope} { input_scope=this; }
TranslationInputScope::~TranslationInputScope() { input_scope=previous; }
TranslationCacheStats GetTranslationCacheStats() { return CompilerCache().Stats(); }

TranslatedShader TranslateGuestGLSL(std::string_view source, ShaderStage stage,
                                    std::span<const ShaderBindingRemap> bindings) {
    if (source.empty() || source.size() > MaxShaderBytes) {
        throw std::invalid_argument("Guest shader source is empty or exceeds its size limit");
    }
    auto result=CompilerCache().Get({TranslationInputKind::GLSL, stage, source, bindings}, [&] {
        return TranslateGuestSPIRV(CompileGuestGLSL(source, stage), stage, bindings);
    });
    // Persistence is optional; a recording failure cannot change a successful
    // shader translation. The callback runs after the compiler-cache lock.
    if(input_scope && input_scope->recorder)
        try { input_scope->recorder(source,stage,bindings); } catch(const std::exception&) {}
    return result;
}

} // namespace Metal
