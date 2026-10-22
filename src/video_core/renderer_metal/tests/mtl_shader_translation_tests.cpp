// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_shader_translation.h"
#include "mtl_fragment_operations.h"
#include "mtl_shader_text.h"
#include "mtl_translation_cache.h"
#include "mtl_compilation_queue.h"
#include <atomic>
#include <barrier>
#include <chrono>
#include <thread>
#include <tuple>
#include <random>
#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {

void Check(bool condition, const char* detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

template <typename Function>
void Reject(Function&& function, const char* detail) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(detail);
}

const Metal::ShaderResource& Resource(const Metal::TranslatedShader& shader, std::uint32_t set,
                                      std::uint32_t binding) {
    for (const auto& resource : shader.resources) {
        if (resource.descriptor_set == set && resource.binding == binding) {
            return resource;
        }
    }
    throw std::runtime_error("Required resource missing from native shader reflection");
}

constexpr std::string_view Vertex = R"(
layout(location=0) in vec4 input_position;
layout(location=1) out vec4 primary_color;
layout(set=0, binding=0, std140) uniform pica {
    uint b;
    uvec4 i[4];
    vec4 f[96];
} uniforms;
layout(set=0, binding=1, std140) uniform draw {
    bool enable_clip1;
    bool flip_viewport;
    vec4 clip_coef;
};
out gl_PerVertex { invariant vec4 gl_Position; float gl_ClipDistance[2]; };
void main() {
    vec4 p = input_position + uniforms.f[0];
    gl_Position = vec4(p.x, flip_viewport ? -p.y : p.y, -p.z, p.w);
    gl_ClipDistance[0] = -p.z;
    gl_ClipDistance[1] = enable_clip1 ? dot(clip_coef, p) : 0.0;
    primary_color = uniforms.f[1];
}
)";

constexpr std::string_view Fragment = R"(
layout(location=1) in vec4 primary_color;
layout(location=0) out vec4 color;
layout(set=0, binding=2, std140) uniform fs_data {
    int framebuffer_scale;
    int alpha_ref;
    float depth_scale;
    float depth_offset;
    ivec4 lighting_lut_offset[6];
    vec3 fog_color;
    vec2 noise;
};
layout(set=0, binding=3) uniform samplerBuffer lut;
layout(set=1, binding=0) uniform sampler2D screens[6];
void main() {
    vec2 value = texelFetch(lut, lighting_lut_offset[0].x).xy;
    color = primary_color * texture(screens[5], vec2(value.x)) + vec4(fog_color, 0.0);
    gl_FragDepth = clamp(-gl_FragCoord.z * depth_scale + depth_offset, 0.0, 1.0);
}
)";

constexpr std::string_view Shadow = R"(
#extension GL_ARB_fragment_shader_interlock : require
layout(pixel_interlock_ordered) in;
layout(set=2, binding=0, r32ui) uniform uimage2D shadow;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    beginInvocationInterlockARB();
    uint old_value = imageLoad(shadow, p).r;
    imageStore(shadow, p, uvec4(min(old_value, uint(gl_FragCoord.z * 16777215.0))));
    endInvocationInterlockARB();
}
)";

constexpr std::string_view Compute = R"(
layout(local_size_x=64) in;
layout(set=3, binding=0, std430) buffer vertices { vec4 data[]; };
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < data.length()) { data[index] = vec4(float(index)); }
}
)";

void Save(const std::filesystem::path& directory, const char* name,
          const Metal::TranslatedShader& shader) {
    std::ofstream output{directory / name};
    output << shader.source;
    Check(output.good(), "Unable to save translated shader for offline Metal compilation");
}

} // namespace

struct Rule {
    std::string expression, replacement;
    std::regex pattern;
    std::vector<std::string_view> prefixes;
    Rule(const char* expression_, const char* replacement_,
         std::initializer_list<std::string_view> prefixes_)
        : expression{expression_}, replacement{replacement_}, pattern{expression}, prefixes(prefixes_) {}
};

void CheckShaderTextFiltering() {
    std::vector<Rule> rules{
        {R"((reg_tmp[0-9]+|vs_out_attr[0-9]+)(\s*=\s*)vec4\(0\.0, 0\.0, 0\.0, 1\.0\))", "$1$2vec4(0.0)", {"reg_tmp","vs_out_attr"}},
        {R"(\bivec2\()", "pica_mova(", {"ivec2("}},
        {R"(uniforms\.i\[([0-3])\])", "(uniforms.i[$1] & uvec4(255u))", {"uniforms.i["}},
        {R"(\b((?:float|int|bool)[234]?\s+)(reg_tmp[0-9]+|conditional_code|address_registers)(\s*=))", "[[maybe_unused]] $1$2$3", {"float","int","bool"}},
        {R"(\b(bool\s+_[0-9]+\s*=\s*(?:exec_shader|sub_[0-9]+_[0-9]+)\s*\())", "[[maybe_unused]] $1", {"bool"}},
        {R"(layout\s*\(\s*location\s*=\s*([0-9]+)\s*\)\s*(in|out)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;)", "$1/$2/$3/$4", {"layout"}},
        {R"(\bgl_Position\s*=)", "$&", {"gl_Position"}},
        {R"(out\s+gl_PerVertex\s*\{[^}]*\}\s*;)", "", {"out"}},
        {R"(void\s+main\s*\(\s*\))", "void azahar_guest_vertex()", {"void"}},
        {R"(\bgl_VertexIndex\b)", "int(gl_GlobalInvocationID.x + azahar_capture.first_vertex)", {"gl_VertexIndex"}},
        {R"(\bgl_InstanceIndex\b)", "int(azahar_capture.instance_index)", {"gl_InstanceIndex"}},
        {R"(\bgl_Position\b)", "azahar_position", {"gl_Position"}},
        {R"(\bgl_ClipDistance\b)", "azahar_clip_distance", {"gl_ClipDistance"}},
        {R"(gl_GlobalInvocationID\.x)", "azahar_selected_index", {"gl_GlobalInvocationID.x"}},
        {R"(\b(vec4|bvec2|ivec3)\s+((?:reg_tmp|vs_out_attr)[0-9]+|conditional_code|address_registers)\s*=\s*([^;]+);)", "$1/$2/$3", {"vec4","bvec2","ivec3"}},
        {R"(azahar_inputs\[azahar_index \* 16u \+ ([0-9]+)u\])", "mesh_load_register($1u,azahar_index)", {"azahar_inputs["}},
        {R"(\[\[color\(0\)\]\])", "[[color(0), raster_order_group(0)]]", {"[[color(0)]]"}},
        {R"(\blight_src\[([0-7])\]|\bLookupLightingLUT(Unsigned|Signed)\((8|9|1[0-9]|2[0-3]),)", "$1/$2/$3", {"light_src[","LookupLightingLUTUnsigned(","LookupLightingLUTSigned("}},
        {R"(\bmetal(Read|Level|Sample)0\b)", "metal$1Normal", {"metalRead0","metalLevel0","metalSample0"}},
        {R"(\btex0\b)", "tex_normal", {"tex0"}},
        {R"(\b(sampler (?:tex[012]|tex_normal|metal_gas_depth)Smplr)\b)", "[[maybe_unused]] $1", {"sampler "}},
        {R"(\b((?:float|uint|int|bool)[234]?\s+)(rounded_primary_color|primary_fragment_color|secondary_fragment_color|combiner_buffer|next_combiner_buffer|color_results_[123]|alpha_results_[123]|diffuse_sum|specular_sum|light_vector|light_distance|refl_value|spot_dir|half_vector|dot_product|clamp_highlights|geo_factor|shadow|tangent)(\s*(?:=|;)))", "[[maybe_unused]] $1$2$3", {"float","uint","int","bool"}},
        {R"(\b(float[234]?\s+)(depth|combiner_output)(\s*(?:=|;)))", "[[maybe_unused]] $1$2$3", {"float"}},
        {R"(\b((?:sampler|thread const array<sampler, 6>&) tex0Smplr))", "[[maybe_unused]] $1", {"sampler","thread const array<sampler, 6>&"}},
        {R"(layout\s*\(\s*location\s*=\s*0\s*\)\s*out\s+vec4\s+([A-Za-z0-9_]+)\s*;)", "$1", {"layout"}},
    };
    std::vector<std::string> sources;
    const std::string fixture=R"(
reg_tmp12 = vec4(0.0, 0.0, 0.0, 1.0);vs_out_attr3)" "\t=\t" R"(vec4(0.0, 0.0, 0.0, 1.0);
ivec2(address);xivec2(address);ivec2x(address);uniforms.i[0];uniforms.i[3];uniforms.i[4];
float4 reg_tmp2 = float4(1); bool conditional_code = false;int3 address_registers = int3(0);
bool _123 = exec_shader(); bool _9 = sub_2_31();bool _55 = sub_2_();floatx reg_tmp1 = 0;
layout ( location = 4 ) in vec4 attr;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[2]; };
void main () { gl_Position = vec4(gl_VertexIndex,gl_InstanceIndex,0,1); }
vec4 reg_tmp8 = vec4(0); bvec2 conditional_code = bvec2(true); ivec3 address_registers = ivec3(0);
azahar_inputs[azahar_index * 16u + 15u];
light_src[0];light_src[7];light_src[8];LookupLightingLUTUnsigned(8,x);LookupLightingLUTSigned(23,x);
metalRead0();metalLevel0();metalSample0();tex0;tex01;tex_normal;
sampler tex0Smplr;sampler tex2Smplr;sampler tex_normalSmplr;sampler metal_gas_depthSmplr;
thread const array<sampler, 6>& tex0Smplr;
float4 rounded_primary_color = 0;uint4 color_results_1;bool shadow;float depth;float4 combiner_output = 0;
layout (location=0) out vec4 color; [[color(0)]]; gl_GlobalInvocationID.x;
)";
    for(const std::string prefix: {"","a","_","0"," ","\n","\r","\t",";","/*","\\"})
        for(const std::string suffix: {"","_","9"," ","\n",";"})sources.push_back(prefix+fixture+suffix);
    sources.emplace_back();
    std::mt19937 random{0xabbaf00d};
    for(unsigned trial=0;trial<40;++trial) {
        std::string source;
        for(unsigned n=0;n<2000;++n)source+=char(random()%256);
        for(unsigned n=0;n<8;++n)source.insert(random()%(source.size()+1),fixture);
        sources.push_back(std::move(source));
    }
    std::size_t checks=0,matches=0;
    for(const auto& source:sources) for(const auto& rule:rules) {
        const auto expected=std::regex_replace(source,rule.pattern,rule.replacement);
        auto actual=Metal::ReplaceShaderText(source,rule.pattern,rule.prefixes,rule.replacement);
        if(expected!=actual)throw std::runtime_error("Replacement bytes differ from original std::regex_replace");
        Metal::ShaderTextMatcher matcher{source,rule.pattern,rule.prefixes};std::cmatch match;std::size_t at{};
        for(std::sregex_iterator it{source.begin(),source.end(),rule.pattern},end;it!=end;++it) {
            if(!matcher.Next(at,match) || at!=std::size_t(it->position()) || match.size()!=it->size() || match.length()!=it->length())throw std::runtime_error("Match positions/captures differ from original iterator");
            for(unsigned n=0;n<match.size();++n)if(match[n].matched!=(*it)[n].matched || match[n].str()!=(*it)[n].str())throw std::runtime_error("Match capture bytes differ");
            ++matches;
        }
        if(matcher.Next(at,match))throw std::runtime_error("Extra match");
        ++checks;
    }
    std::cout<<"EXACT checks="<<checks<<" matches="<<matches<<" rules="<<rules.size()<<"; output bytes + every capture/position; boundaries, whitespace, UTF8/high bytes,NUL, absent alternatives\n"<<std::flush;
}

namespace {
void Same(const Metal::TranslatedShader& a,const Metal::TranslatedShader& b) {
 Check(a.stage==b.stage && a.source==b.source && a.entry_point==b.entry_point && a.spirv==b.spirv &&
       a.needs_buffer_sizes==b.needs_buffer_sizes && a.uses_framebuffer_fetch==b.uses_framebuffer_fetch && a.resources.size()==b.resources.size(),"Complete source/SPIRV/entry/flags differ");
 for(unsigned n=0;n<a.resources.size();++n) {
  const auto& x=a.resources[n];const auto& y=b.resources[n];
  Check(std::tie(x.name,x.kind,x.descriptor_set,x.binding,x.metal_index,x.sampler_index,x.count,x.byte_size,x.runtime_array_stride,x.active)==std::tie(y.name,y.kind,y.descriptor_set,y.binding,y.metal_index,y.sampler_index,y.count,y.byte_size,y.runtime_array_stride,y.active) && x.members.size()==y.members.size(),"Resource ABI differs");
  for(unsigned i=0;i<x.members.size();++i)Check(std::tie(x.members[i].name,x.members[i].offset,x.members[i].size)==std::tie(y.members[i].name,y.members[i].offset,y.members[i].size),"Member ABI differs");
 }
}

struct CollisionHash { std::size_t operator()(const Metal::TranslationInput&) const { return 0; } };
inline Metal::TranslatedShader MockShader(unsigned id) {
    Metal::TranslatedShader s{};s.stage=Metal::ShaderStage::Compute;s.source="source"+std::to_string(id);
    s.entry_point="entry"+std::to_string(id);s.spirv={0x07230203,1,id,0,0};s.needs_buffer_sizes=id%2;
    s.uses_framebuffer_fetch=id%3;s.resources.push_back({.name="resource"+std::to_string(id),
        .kind=Metal::ShaderResourceKind::UniformBuffer,.descriptor_set=1,.binding=2,.metal_index=3,
        .sampler_index=Metal::NoShaderBinding,.count=1,.byte_size=16,.runtime_array_stride=0,
        .members={{"member"+std::to_string(id),0,16}},.active=true});return s;
}
inline void CheckTranslationCache() {
    using namespace Metal;
    BasicTranslationCache<CollisionHash> cache;
    std::string bytes{"a\0b",3};std::vector<ShaderBindingRemap> bindings{{0,1,2,3,4}};
    TranslationInput input{TranslationInputKind::GLSL,ShaderStage::Compute,bytes,bindings};
    unsigned compiled=0;
    const auto first=cache.Get(input,[&]{return MockShader(++compiled);});
    Same(first,cache.Get(input,[&]{return MockShader(++compiled);}));
    Check(compiled==1,"Exact translation hit rebuilt");
    auto changed=bytes;changed[2]='c';auto variant=input;variant.bytes=changed;
    auto second=cache.Get(variant,[&]{return MockShader(++compiled);});
    Check(second.entry_point!=first.entry_point && compiled==2,"Full source bytes lost under hash collision");
    variant=input;variant.kind=TranslationInputKind::SPIRV;
    (void)cache.Get(variant,[&]{return MockShader(++compiled);});
    variant=input;variant.stage=ShaderStage::Fragment;
    (void)cache.Get(variant,[&]{return MockShader(++compiled);});
    Check(compiled==4,"Stage or input kind lost under collision");
    for(unsigned field=0;field<5;++field) {
        auto remap=bindings;
        switch(field){case 0:++remap[0].descriptor_set;break;case 1:++remap[0].binding;break;
            case 2:++remap[0].buffer;break;case 3:++remap[0].texture;break;case 4:++remap[0].sampler;break;}
        variant=input;variant.bindings=remap;const auto before=compiled;
        (void)cache.Get(variant,[&]{return MockShader(++compiled);});
        Check(compiled==before+1,"Binding field collapsed under collision");
    }
    auto larger=bindings;larger.push_back({1,2,3,4,5});variant=input;variant.bindings=larger;
    (void)cache.Get(variant,[&]{return MockShader(++compiled);});
    std::swap(larger[0],larger[1]);variant.bindings=larger;
    (void)cache.Get(variant,[&]{return MockShader(++compiled);});
    Check(compiled==11,"Binding length/order was weakened");
    auto copy=cache.Get(input,[&]{return MockShader(++compiled);});copy.source="mutated";
    copy.spirv[0]=0;copy.resources[0].members[0].name="mutated";
    Same(first,cache.Get(input,[&]{return MockShader(++compiled);}));
    bytes[0]='z';bindings[0].buffer=99;
    std::array<ShaderBindingRemap,1> original{{{0,1,2,3,4}}};
    TranslationInput original_input{TranslationInputKind::GLSL,ShaderStage::Compute,std::string_view{"a\0b",3},original};
    Same(first,cache.Get(original_input,[&]{return MockShader(++compiled);}));
    Check(compiled==11,"Borrowed source/remap mutation changed owned cache key");
    {
        TranslationCache small{2,1<<20};unsigned builds=0;
        for(auto key:{"one","two","three","one"}) {
            auto in=original_input;in.bytes=key;(void)small.Get(in,[&]{return MockShader(++builds);});
        }
        const auto s=small.Stats();Check(builds==4 && s.entries==2 && s.evictions==2 && s.retained_payload_bytes<=(1u<<20),"Translation entry bound/eviction failed");
    }
    for(unsigned limit:{0u,64u,1024u}) {
        TranslationCache small{limit?256u:0u,limit};unsigned builds=0;
        std::string large(4096,'x');auto in=original_input;in.bytes=large;
        for(unsigned n=0;n<2;++n)(void)small.Get(in,[&]{return MockShader(++builds);});
        Check(builds==2 && !small.Stats().entries && !small.Stats().retained_payload_bytes,"Translation payload bound retained oversized entry");
    }
    {
        TranslationCache bounded{256,4096};unsigned builds=0;
        for(unsigned n=0;n<5;++n) {
            std::string text(1024,char('a'+n));auto in=original_input;in.bytes=text;
            (void)bounded.Get(in,[&]{return MockShader(++builds);});
            Check(bounded.Stats().retained_payload_bytes<=4096,"Translation cumulative payload exceeded bound");
        }
        Check(builds==5 && bounded.Stats().entries>0 && bounded.Stats().entries<5 && bounded.Stats().evictions>0,
              "Translation payload pressure did not evict fitting entries");
    }
    {
        TranslationCache failed;unsigned builds=0;bool rejected=false;
        try{(void)failed.Get(original_input,[&]()->TranslatedShader{++builds;throw std::runtime_error("original compiler error");});}
        catch(const std::runtime_error& e){rejected=std::string_view(e.what())=="original compiler error";}
        Check(rejected && failed.Stats().entries==0,"Compiler exception changed or failure cached");
        (void)failed.Get(original_input,[&]{return MockShader(++builds);});Check(builds==2,"Failed compile prevented retry");
    }
    {
        TranslationCache nested;auto inner=original_input;inner.kind=TranslationInputKind::SPIRV;
        const auto shader=nested.Get(original_input,[&]{return nested.Get(inner,[&]{return MockShader(42);});});
        Same(shader,MockShader(42));Check(nested.Stats().entries==2,"Nested translation publication failed");
    }
    {
        TranslationCache shared;std::barrier barrier{8};std::atomic<unsigned> builds{};
        std::promise<void> release;auto ready=release.get_future().share();
        std::array<std::future<TranslatedShader>,8> futures;
        for(auto& result:futures)result=std::async(std::launch::async,[&]{barrier.arrive_and_wait();return shared.Get(original_input,[&]{++builds;ready.wait();return MockShader(888);});});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(shared.Stats().lookups[0]!=8 && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        release.set_value();
        for(auto& result:futures)Same(result.get(),MockShader(888));
        auto stats=shared.Stats();Check(builds==1 && stats.entries==1 && stats.duplicate_compiles==0 && stats.coalesced_compiles==7,
            "Concurrent first use did not share exactly one complete translation");
        Same(shared.Get(original_input,[&]{return MockShader(++builds);}),MockShader(888));
        Check(shared.Stats().hits[0]==1,"Concurrent cache did not publish reusable result");
    }
    {
        TranslationCache failed;std::barrier start{4};std::promise<void> release;
        auto ready=release.get_future().share();std::atomic<unsigned> builds{};
        std::array<std::future<bool>,4> futures;
        for(auto& result:futures)result=std::async(std::launch::async,[&]{
            start.arrive_and_wait();
            try { failed.Get(original_input,[&]()->TranslatedShader {
                ++builds;ready.wait();throw std::runtime_error("shared original compiler error");
            }); } catch(const std::runtime_error& e) {return std::string_view(e.what())=="shared original compiler error";}
            return false;
        });
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(failed.Stats().lookups[0]!=4 && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        release.set_value();
        for(auto& result:futures)Check(result.get(),"Shared translation failure lost its original exception");
        Check(builds==1 && failed.Stats().entries==0 && failed.Stats().coalesced_compiles==3,
            "Failed translation duplicated work or poisoned the result cache");
        Same(failed.Get(original_input,[&]{++builds;return MockShader(999);}),MockShader(999));
        Check(builds==2,"Shared compiler failure prevented a later successful retry");
    }
    {
        TranslationCache recursive;
        Same(recursive.Get(original_input,[&]{return recursive.Get(original_input,[&]{return MockShader(42);});}),MockShader(42));
        Check(recursive.Stats().entries==1,"Nested same-key translation failed to publish");
    }
    std::cout<<"CACHE_EXACT full bytes including NUL, input-kind/stage/all five binding fields/count/order, forced collisions, owned key/result isolation, entry/payload bounds, eviction, compiler errors/retry, nested compilation and8-way concurrent publication passed\n"<<std::flush;
}

void CheckCompilationQueue() {
    using namespace Metal;
    CompilationQueue queue{2,4,64};
    const auto owner=std::this_thread::get_id();
    std::promise<void> release;
    auto gate=release.get_future().share();
    auto held=queue.Submit([gate,owner] { gate.wait(); return std::this_thread::get_id()!=owner; },64);
    std::promise<void> submitting;
    auto attempted=submitting.get_future();
    auto next=std::async(std::launch::async,[&] {
        submitting.set_value();
        return queue.Submit([] {return 123;},1);
    });
    attempted.wait();
    const bool bounded=next.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
    release.set_value();
    Check(bounded && held.get() && next.get().get()==123,"Compiler queue lost active-payload bounds or separate execution");
    auto nested=queue.Submit([&] {return queue.Submit([] {return 456;},1024).get();});
    Check(nested.get()==456,"Nested compiler factory deadlocked or changed result");
    auto failed=queue.Submit([]()->int {throw std::runtime_error("original native compilation failure");});
    for(unsigned waiter=0;waiter<2;++waiter) {
        bool original{};
        try {(void)failed.get();}catch(const std::runtime_error& e) {original=std::string_view(e.what())=="original native compilation failure";}
        Check(original,"Compiler queue lost original shared failure");
    }
    Check(queue.Submit([]{return 789;},1024).get()==789,"Oversized valid compiler input was truncated/rejected");
    queue.Drain();
    std::shared_future<unsigned> shutdown;
    {CompilationQueue ending{1};shutdown=ending.Submit([]{return 321u;});}
    Check(shutdown.get()==321,"Compiler queue shutdown abandoned an owned factory");
    std::cout<<"Bounded compiler queue: separate threads, active payload admission, nested factories, original shared errors, oversized input and draining shutdown passed\n";
}
}

int main(int argc, char** argv) {
    try {
        Check(argc == 2, "Specify a temporary directory for offline shader fixtures");
        const std::filesystem::path directory{argv[1]};
        std::filesystem::create_directories(directory);
        CheckShaderTextFiltering();
        CheckTranslationCache();
        CheckCompilationQueue();
        // Concurrent first use must initialize the GLSL compiler once, without races.
        std::array<std::future<Metal::TranslatedShader>, 4> concurrent;
        for (auto& request : concurrent) {
            request = std::async(std::launch::async, [] {
                return Metal::TranslateGuestGLSL(Vertex, Metal::ShaderStage::Vertex,
                                                 Metal::RasterizerShaderBindings());
            });
        }
        auto vertex = concurrent[0].get();
        for (std::size_t i = 1; i < concurrent.size(); ++i) {
            Check(concurrent[i].get().source == vertex.source,
                  "Concurrent shader translation produced inconsistent MSL");
        }
        Check(vertex.entry_point == "azahar_vertex", "Wrong native vertex entry point");
        const auto& pica = Resource(vertex, 0, 0);
        Check(pica.metal_index == 0 && pica.byte_size == 1616, "Wrong PICA uniform block binding or ABI");
        Check(pica.members.size() == 3 && pica.members[1].offset == 16 && pica.members[2].offset == 80,
              "PICA uniform arrays do not preserve std140 offsets");
        const auto& draw = Resource(vertex, 0, 1);
        Check(draw.metal_index == 1 && draw.byte_size == 32 && draw.members[2].offset == 16,
              "Boolean draw uniforms do not preserve their host ABI");
        auto fragment = Metal::TranslateGuestGLSL(Fragment, Metal::ShaderStage::Fragment,
                                                  Metal::RasterizerShaderBindings());
        Check(Resource(fragment, 1, 0).metal_index == 3 && Resource(fragment, 1, 0).count == 6 &&
              Resource(fragment, 1, 0).sampler_index == 0, "Wrong shadow-cube descriptor array mapping");
        Check(Resource(fragment, 0, 3).metal_index == 0 &&
              Resource(fragment, 0, 3).sampler_index == Metal::NoShaderBinding,
              "Native LUT texture buffer incorrectly requires a sampler");
        auto shadow = Metal::TranslateGuestGLSL(Shadow, Metal::ShaderStage::Fragment,
                                                Metal::RasterizerShaderBindings());
        Check(shadow.source.find("raster_order_group") != std::string::npos,
              "Shadow fragment interlock lost raster-order semantics");
        Check(Resource(shadow, 2, 0).metal_index == 12, "Wrong shadow storage binding");
        {
            const auto mask = Metal::PicaTextureMask(fragment);
            Check(mask.test(0), "LUT texture consumer missing from immutable mask");
            for (unsigned index = 3; index < 9; ++index)
                Check(mask.test(index), "Texture array element missing from immutable mask");
            Check(Metal::PicaTextureMask(shadow).test(12), "Shadow storage write omitted from texture consumers");
            for (const auto kind : {Metal::ShaderResourceKind::Texture, Metal::ShaderResourceKind::StorageTexture,
                                    Metal::ShaderResourceKind::Sampler, Metal::ShaderResourceKind::UniformBuffer,
                                    Metal::ShaderResourceKind::StorageBuffer}) {
                for (bool active : {false, true}) {
                    Metal::TranslatedShader fixture{};
                    fixture.resources.push_back({.name = "consumer", .kind = kind,
                        .descriptor_set = 0, .binding = 0, .metal_index = 63, .count = 3, .active = active});
                    const auto actual = Metal::PicaTextureMask(fixture);
                    const bool consumes = active && (kind == Metal::ShaderResourceKind::Texture ||
                                                     kind == Metal::ShaderResourceKind::StorageTexture);
                    Check(actual.count() == (consumes ? 3U : 0U) && actual.test(63) == consumes &&
                          actual.test(64) == consumes && actual.test(65) == consumes,
                          "Immutable texture mask confuses kinds, inactive resources or array banks");
                }
            }
            std::cout << "Immutable texture consumers: reflected arrays, LUTs, shadow writes and ten kind/activity cases passed\n";
        }
        auto compute = Metal::TranslateGuestGLSL(Compute, Metal::ShaderStage::Compute,
                                                 Metal::RasterizerShaderBindings());
        Check(compute.needs_buffer_sizes && Resource(compute, 3, 0).metal_index == 16,
              "Runtime array length lost native buffer-size metadata");
        auto vertex_compute = Metal::TranslateGuestGLSL(Vertex, Metal::ShaderStage::VertexCompute,
                                                        Metal::RasterizerShaderBindings());
        Check(vertex_compute.source.find("kernel") != std::string::npos,
              "Guest vertex program was not translated as a native compute kernel");
        Check(vertex_compute.source.find("[[stage_in]]") == std::string::npos &&
              Resource(vertex_compute, 3, 1).metal_index == 17 &&
              Resource(vertex_compute, 3, 2).metal_index == 18 &&
              Resource(vertex_compute, 3, 3).metal_index == 19 &&
              Resource(vertex_compute, 3, 2).runtime_array_stride == sizeof(Metal::CapturedVertex),
              "Native vertex compute capture lost its explicit buffer ABI");
        Reject([] { Metal::CompileGuestGLSL("invalid shader", Metal::ShaderStage::Vertex); },
               "Malformed GLSL was accepted");
        Reject([] { Metal::TranslateGuestSPIRV({}, Metal::ShaderStage::Vertex, {}); },
               "Empty SPIR-V was accepted");
        const auto oversized_block = Metal::CompileGuestGLSL(
            "layout(local_size_x=1) in;"
            "layout(set=3, binding=0, std430) buffer huge { vec4 data[268435456]; };"
            "void main() { data[gl_GlobalInvocationID.x] = vec4(1.0); }",
            Metal::ShaderStage::Compute);
        bool oversized_rejected{};
        try {
            Metal::TranslateGuestSPIRV(oversized_block, Metal::ShaderStage::Compute,
                                      Metal::RasterizerShaderBindings());
        } catch (const std::invalid_argument& error) {
            oversized_rejected = std::string_view{error.what()}.find(
                                     "buffer layout exceeds its reflection ABI") !=
                                 std::string_view::npos;
        }
        Check(oversized_rejected, "A 4-GiB storage block was accepted with a truncated reflection size");
        const auto large_block = Metal::TranslateGuestGLSL(
            "layout(local_size_x=1) in;"
            "layout(set=3, binding=0, std430) buffer large { vec4 data[268435455]; };"
            "void main() { data[gl_GlobalInvocationID.x] = vec4(1.0); }",
            Metal::ShaderStage::Compute, Metal::RasterizerShaderBindings());
        Check(Resource(large_block, 3, 0).byte_size == 4294967280u &&
                  Resource(large_block, 3, 0).members[0].size == 4294967280u,
              "A representable large storage layout lost its exact reflection size");
        std::cout << "Storage reflection: 4-GiB overflow rejected and 4-GiB-minus-16 layout preserved\n";
        Reject([] { Metal::TranslateGuestGLSL(Vertex, Metal::ShaderStage::Vertex, {}); },
               "Unmapped uniforms were assigned implicit native bindings");
        constexpr std::array overlapping{Metal::ShaderBindingRemap{0, 0, 0},
                                         Metal::ShaderBindingRemap{0, 1, 0}};
        Reject([&] { Metal::TranslateGuestGLSL(Vertex, Metal::ShaderStage::Vertex, overlapping); },
               "Overlapping native buffer slots were accepted");
        constexpr std::array out_of_range{Metal::ShaderBindingRemap{0, 0, 31},
                                          Metal::ShaderBindingRemap{0, 1, 1}};
        Reject([&] { Metal::TranslateGuestGLSL(Vertex, Metal::ShaderStage::Vertex, out_of_range); },
               "A native buffer index outside the table was accepted");
        Save(directory, "guest_vertex.metal", vertex);
        Save(directory, "guest_fragment.metal", fragment);
        Save(directory, "guest_shadow.metal", shadow);
        Save(directory, "guest_compute.metal", compute);
        Save(directory, "guest_vertex_compute.metal", vertex_compute);
        auto logic = Metal::TranslateRasterizerFragmentGLSL(
            "layout(location=0) out vec4 color; void main() { color = vec4(0.3); }",
            {true, 11, Metal::FragmentColorFormat::RGB565});
        Check(logic.uses_framebuffer_fetch && logic.resources.empty() &&
              logic.source.find("raster_order_group(0)") != std::string::npos &&
              logic.source.find("[[texture(") == std::string::npos,
              "Native logic operations did not use ordered tile-memory framebuffer fetch");
        Save(directory, "guest_fragment_logic.metal", logic);
        Reject([] {
            Metal::TranslateRasterizerFragmentGLSL("void main() {}", {true, 16});
        }, "Invalid logic operation was accepted");
        Reject([] {
            Metal::TranslateRasterizerFragmentGLSL("void main() {}", {true, 11});
        }, "Native logic operation accepted a shader without a color output");
        Reject([] {
            Metal::TranslateGuestGLSL(
                "layout(input_attachment_index=1, set=4, binding=0) uniform subpassInput prior;"
                "layout(location=0) out vec4 color; void main() { color = subpassLoad(prior); }",
                Metal::ShaderStage::Fragment, Metal::RasterizerShaderBindings());
        }, "An unsupported native framebuffer attachment was accepted");
        std::cout << "Shader ABI, descriptor arrays, LUTs, interlock, compute capture, concurrency and rejection checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
