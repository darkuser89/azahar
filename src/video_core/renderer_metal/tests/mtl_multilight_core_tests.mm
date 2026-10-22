// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Public PICA commands -> mixed multilight/raw LUTs -> RasterizerMetal -> physical guest VRAM.
// Core data objects only: no OS window, emulator initialization, ROM or title.
#include "mtl_rasterizer.h"
#include "mtl_pica_vertex_fixture.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/image_interface.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace {
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void Write(std::vector<u32>& words, u32 reg, u32 value) {
    words.insert(words.end(), {value, reg | (15U << 16)});
}
u32 Float24(float value) {
    if (value == 0) return 0;
    const auto bits = std::bit_cast<u32>(value);
    const auto raw = ((bits >> 31) << 23) | ((((bits >> 23) & 255) - 64) << 16) |
                     ((bits & 0x7fffff) >> 7);
    Check(Pica::f24::FromRaw(raw).ToFloat32() == value, "Inexact f24 fixture");
    return raw;
}
void Attribute(std::vector<u32>& words, const std::array<float, 4>& value) {
    const auto x = Float24(value[0]), y = Float24(value[1]);
    const auto z = Float24(value[2]), w = Float24(value[3]);
    words.insert(words.end(), {(w << 8) | (z >> 16),
        u32(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0])) |
            (15U << 16) | (2U << 20) | (1U << 31),
        (z << 16) | (y >> 8), (y << 24) | x});
}
// Deliberately independent base/delta fields: omitting or changing the sign of
// interpolation must be observable. These are raw register fixtures, not a
// claim that their slopes describe a continuous real-world material function.
u32 Entry(unsigned table, unsigned index, unsigned revision) {
    const auto value = 1400 + (table * 113 + index * 17 + revision * 379) % 1200;
    const auto difference = 320 + (table * 29 + index * 7 + revision * 97) % 320;
    return value | (difference << 12) | (((table + index + revision) & 1) << 23);
}
// Independent scalar decoding of raw fixed-point fields, using the LUT address
// contract. Do not call LutEntry::ToFloat/DiffToFloat, generator lookup helpers,
// SyncLUTs, or read the GPU LUT resource to obtain expected values.
double Lookup(unsigned table, double input, bool unsigned_domain, unsigned revision,
              bool use_difference = true) {
    const double position = input * (unsigned_domain ? 256 : 128);
    const int cell = std::clamp(int(std::floor(position)), unsigned_domain ? 0 : -128,
                                unsigned_domain ? 255 : 127);
    const auto raw = Entry(table, unsigned(cell < 0 ? cell + 256 : cell), revision);
    const double base = (raw & 4095) / 4095.0;
    const double delta = ((raw >> 12) & 2047) / 2047.0 * ((raw & (1U << 23)) ? -1 : 1);
    return base + (use_difference ? delta * (position - cell) : 0);
}
using Vec = std::array<double, 3>;
constexpr std::array<unsigned, 8> Configs{0, 1, 2, 3, 4, 5, 6, 8};
// Independent supported-component matrix, columns D0,D1,FR,RR,RG,RB.
constexpr std::array<std::array<bool, 6>, 8> Available{{
    {1,0,0,1,0,0}, {0,0,1,1,0,0}, {1,1,0,1,0,0}, {1,1,1,0,0,0},
    {1,1,0,1,1,1}, {1,0,1,1,1,1}, {1,1,1,1,0,0}, {1,1,1,1,1,1}}};
constexpr std::array<unsigned, 6> Tables{0, 1, 3, 6, 5, 4};
constexpr std::array<unsigned, 6> Shifts{0, 4, 12, 24, 20, 16};
constexpr std::array<unsigned, 6> ScaleBits{6, 7, 0, 1, 2, 3};
constexpr std::array<double, 6> Scales{.25, .5, 1, 2, 4, 8};
constexpr Vec Global{8, 12, 20};
struct Case { unsigned config, count, order, domain, selector, output; };
std::vector<Case> Cases() {
    std::vector<Case> cases;
    for (unsigned count = 1; count <= 8; ++count)
        for (unsigned order = 0; order < 3; ++order)
            for (unsigned domain = 0; domain < 2; ++domain)
                for (unsigned selector = 0; selector < 6; ++selector)
                    for (unsigned output = 0; output < 2; ++output)
                        cases.push_back({7, count, order, domain, selector, output});
    // Other lighting modes: counts1/4/8, one coupled order/selector per mode.
    // These supplement the full Config7 product; they are not a full product.
    for (unsigned config = 0; config < 7; ++config)
        for (unsigned count : std::array<unsigned, 3>{1, 4, 8})
            for (unsigned domain = 0; domain < 2; ++domain)
                for (unsigned output = 0; output < 2; ++output)
                    cases.push_back({config, count, config % 3, domain, config % 5, output});
    Check(cases.size() == 660, "Incomplete mixed multilight cases");
    return cases;
}
unsigned Physical(unsigned order, unsigned slot) {
    return order == 0 ? slot : order == 1 ? 7 - slot : (3 * slot + 2) % 8;
}
bool TwoSided(unsigned order, unsigned slot) { return (slot + order) % 2 != 0; }
Vec Position(unsigned id) {
    return {(int(id % 3) - 1) * .25, (int(id % 4) - 1) * .125,
            (id % 2 ? -1 : 1) * (.75 + id * .125)};
}
constexpr Vec View{.25, -.125, .5}, Spot{-.5, .25, .75};
// Observed public uniform contract for signed fixed1.1.11: divide by2047.
Vec SpotDirection() { return {-1024/2047.0, 512/2047.0, 1536/2047.0}; }
double Bias(unsigned id) { return id * .125 - .25; }
double DistanceScale(unsigned id) { return id % 2 ? -.25 : .5; }
unsigned LightColor(unsigned id, unsigned channel, unsigned term, unsigned brightness) {
    constexpr std::array<std::array<unsigned, 3>, 4> coefficients{{{3,2,1},{1,2,3},{2,3,1},{3,1,2}}};
    const auto base = term == 0 ? (1 + id % 3) * 8 : term == 1 ? id + 1 :
                      term == 2 ? (1 + id % 2) * 12 : (1 + id % 3) * 8;
    return base * coefficients[term][channel] * (brightness ? 8 : 1);
}
u32 PackColor(unsigned id, unsigned term, unsigned brightness) {
    const auto r = LightColor(id, 0, term, brightness);
    const auto g = LightColor(id, 1, term, brightness);
    const auto b = LightColor(id, 2, term, brightness);
    Check(r < 1024 && g < 1024 && b < 1024, "10bit light color overflow");
    return (r << 20) | (g << 10) | b;
}
u32 Float16(float value) {
    if (value == 0) return 0;
    const auto raw = Float24(value);
    Check((raw & 63) == 0 && ((raw >> 16) & 127) >= 49 && ((raw >> 16) & 127) <= 78,
          "Inexact/out-of-range f16 light fixture");
    return ((raw >> 8) & 0x8000) | (((((raw >> 16) & 127) - 48) & 31) << 10) |
           ((raw & 65535) >> 6);
}
u32 Float20(float value) {
    const auto raw = Float24(value);
    Check((raw & 15) == 0, "Inexact f20 attenuation fixture");
    return raw >> 4;
}
double Dot(Vec a, Vec b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
Vec Unit(Vec a) {
    const auto length = std::sqrt(Dot(a, a)); Check(length > 0, "Zero light/half/view vector");
    for (auto& x : a) x /= length;
    return a;
}
unsigned Selector(Case c, unsigned lut) { return (c.selector + lut) % (c.config == 7 ? 6 : 5); }
double Scale(Case c, unsigned lut) { return Scales[(c.selector + lut) % 6]; }
struct Result {
    std::array<unsigned, 4> rgba{};
    std::array<double, 3> unbounded{};
    double alpha{1}, first_alpha{1};
};
Result Expected(Case c, bool reverse, unsigned brightness, unsigned revision,
                bool difference = true, bool force_all_supported = false) {
    const Vec normal{0, 0, reverse ? -1.0 : 1.0}, eye = Unit(View);
    Vec primary = Global, secondary{};
    Result result{};
    for (unsigned slot = 0; slot < c.count; ++slot) {
        const auto id = Physical(c.order, slot); auto vector = Position(id);
        if (id % 2) for (unsigned ch = 0; ch < 3; ++ch) vector[ch] += View[ch];
        const auto distance = std::sqrt(Dot(vector, vector)); const auto light = Unit(vector);
        const Vec half{light[0]+eye[0], light[1]+eye[1], light[2]+eye[2]}; const auto h = Unit(half);
        const double ln = Dot(light, normal), nh = Dot(normal, h);
        const double lambert = TwoSided(c.order, slot) ? std::abs(ln) : std::max(ln, 0.0);
        const double geometry = std::min(lambert / Dot(half, half), 1.0);
        const double highlight = c.order % 2 && lambert == 0 ? 0 : 1;
        // LUT input clamping and diffuse/geometry use the same enabled light's
        // two-sided flag, even when its physical ID differs from its slot.
        const bool lut_two_sided = TwoSided(c.order, slot);
        const auto domain = [&](double value) {
            return !c.domain ? value : lut_two_sided ? std::abs(value) : std::max(value, 0.0);
        };
        const std::array<double, 6> inputs{nh, Dot(eye,h), Dot(normal,eye), ln,
                                         Dot(light,SpotDirection()), h[0]-normal[0]*nh};
        const auto supported = [&](unsigned i) { return force_all_supported || Available[c.config][i]; };
        const auto lookup = [&](unsigned i) {
            return Scale(c,i) * Lookup(Tables[i], domain(inputs[Selector(c,i)]), c.domain, revision, difference);
        };
        const double spot = slot % 3 && (force_all_supported || (c.config != 2 && c.config != 3))
            ? Scales[(c.selector+2)%6] * Lookup(8+id,domain(inputs[4]),c.domain,revision,difference) : 1;
        const double dist = slot % 3 != 1 ? Lookup(16+id,
            std::clamp(Bias(id)+DistanceScale(id)*distance,0.0,1.0),true,revision,difference) : 1;
        const double d0 = supported(0) ? lookup(0) : 1, d1 = supported(1) ? lookup(1) : 1;
        const double rr = supported(3) ? lookup(3) : 1;
        const Vec reflection{rr, supported(4) ? lookup(4) : rr, supported(5) ? lookup(5) : rr};
        for (unsigned ch = 0; ch < 3; ++ch) {
            primary[ch] += (LightColor(id,ch,0,brightness)*lambert + LightColor(id,ch,1,brightness))*spot*dist;
            secondary[ch] += (LightColor(id,ch,2,brightness)*d0*(slot&1 ? geometry : 1) +
                LightColor(id,ch,3,brightness)*d1*reflection[ch]*(slot&2 ? geometry : 1))*highlight*spot*dist;
        }
        result.alpha = supported(2) ? std::clamp(lookup(2),0.0,1.0) : 1;
        if (slot == 0) result.first_alpha = result.alpha;
    }
    result.unbounded = c.output ? secondary : primary;
    for (unsigned ch = 0; ch < 3; ++ch)
        result.rgba[ch] = unsigned(std::lround(std::clamp(result.unbounded[ch],0.0,255.0)));
    result.rgba[3] = unsigned(std::lround(255 * result.alpha));
    return result;
}
class Window final : public Frontend::EmuWindow {
public:
    void PollEvents() override {}
};
class Renderer final : public VideoCore::RendererBase {
public:
    Renderer(Core::System& system, Window& window) : RendererBase{system, window, nullptr} {}
    VideoCore::RasterizerInterface* Rasterizer() override { return rasterizer; }
    void SwapBuffers() override {}
    void TryPresent(int, bool) override {}
    VideoCore::RasterizerInterface* rasterizer{};
};
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Check(argc <= 2 && (argc == 1 || std::filesystem::path{argv[1]}.is_absolute()), "Expected optional absolute shader prefix");
            Core::System system;
            system.RegisterImageInterface(std::make_shared<Frontend::ImageInterface>());
            Memory::MemorySystem memory{system};
            Pica::PicaCore pica{memory, nullptr};
            Metal::Test::InitializeFixtureUniforms(pica.vs_setup);
            Metal::Test::InitializeFixtureUniforms(pica.gs_setup);
            Window window;
            Renderer renderer{system, window};
            VideoCore::CustomTexManager custom{system};
            Settings::values.resolution_factor.SetValue(1);
            Settings::values.use_disk_shader_cache.SetValue(false);
            Settings::values.dump_textures.SetValue(false);
            Settings::values.custom_textures.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory, pica, custom, renderer, device, scheduler};
            renderer.rasterizer = &rasterizer;
            pica.BindRasterizer(&rasterizer);
            auto& regs = pica.regs.internal;
            regs = {};
            Metal::Test::MovVertex(regs, pica.vs_setup);
            regs.vs.output_mask.Assign(15);
            regs.rasterizer.vs_output_total.Assign(4);
            regs.rasterizer.vs_output_attributes[3].raw = 0x1f141312; // View XYZ, unused W.
            pica.vs_setup.UpdateProgramCode(3, Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV, 3, 3));
            pica.vs_setup.UpdateProgramCode(4, u32(nihstro::OpCode::Id::END) << 26);
            regs.vs.max_input_attribute_index.Assign(3);
            regs.pipeline.max_input_attrib_index.Assign(3);
            regs.vs.input_attribute_to_register_map_low = 0x3210;
            regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
            regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR / 8);
            regs.framebuffer.framebuffer.width.Assign(32);
            regs.framebuffer.framebuffer.height.Assign(31);
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.viewport_size_x.Assign(67 << 16);
            regs.rasterizer.viewport_size_y.Assign(67 << 16);
            regs.rasterizer.viewport_depth_near_plane.Assign(62 << 16);
            pica.dirty_regs.SetAllDirty();
            auto target = memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize() >= 4096, "Missing physical VRAM");
            std::memset(target.GetPtr(), 0x11, 4096);
            const auto process = [&](const std::vector<u32>& words) {
                const auto bytes = words.size() * sizeof(u32);
                auto backing = memory.GetPhysicalRef(Memory::VRAM_PADDR + 0x8000);
                Check(backing && backing.GetSize() >= bytes, "Invalid command backing");
                std::memcpy(backing.GetPtr(), words.data(), bytes);
                pica.ProcessCmdList(Memory::VRAM_PADDR + 0x8000, u32(bytes), false);
            };
            // Initialize unused Fog/Proc storage via public commands as well.
            std::vector<u32> init;
            Write(init, PICA_REG_INDEX(texturing.fog_lut_offset), 0);
            for (unsigned i = 0; i < 128; ++i) Write(init, PICA_REG_INDEX(texturing.fog_lut_data[0])+i%8, 0);
            for (unsigned table : std::array<unsigned,5>{0,2,3,4,5}) {
                Write(init, PICA_REG_INDEX(texturing.proctex_lut_config), table << 8);
                for (unsigned i = 0; i < (table < 4 ? 128U : 256U); ++i)
                    Write(init, PICA_REG_INDEX(texturing.proctex_lut_data[0])+i%8, 0);
            }
            process(init);
            constexpr std::array<std::array<float,4>,3> positions{{
                {-1,-1,-.5f,1}, {3,-1,-.5f,1}, {-1,3,-.5f,1}}};
            const auto cases = Cases();
            unsigned draws{}, raw_entries{}, exports{};
            unsigned interpolation{}, mutation{}, availability{}, order_sensitive{};
            unsigned primary_saturated{}, secondary_saturated{}, distinct_fresnel{};
            for (unsigned revision = 0; revision < 3; ++revision) {
                std::vector<u32> upload;
                const unsigned start = std::array<unsigned,3>{239,129,255}[revision];
                for (unsigned table = 0; table < 24; ++table) {
                    Write(upload, PICA_REG_INDEX(lighting.lut_config), (table << 8) | start);
                    for (unsigned step = 0; step < 256; ++step)
                        Write(upload, PICA_REG_INDEX(lighting.lut_data[0])+step%8,
                              Entry(table,(start+step)%256,revision));
                }
                process(upload);
                Check(pica.lighting.lut_dirty == 0xffffff, "Raw commands missed dirty tables");
                Check(regs.lighting.lut_config.index == start && regs.lighting.lut_config.type == 23,
                      "Raw Lighting index/autoincrement/wrap mismatch");
                for (unsigned table = 0; table < 24; ++table) for (unsigned i = 0; i < 256; ++i) {
                    Check(pica.lighting.luts[table][i].raw == Entry(table,i,revision), "Raw Lighting value mismatch");
                    ++raw_entries;
                }
                for (unsigned variant = 0; variant < cases.size(); ++variant)
                    for (unsigned reverse = 0; reverse < 2; ++reverse)
                        for (unsigned brightness = 0; brightness < 2; ++brightness) {
                            const auto c = cases[variant];
                            std::vector<u32> setup;
                            Write(setup, PICA_REG_INDEX(lighting.disable), 0);
                            Write(setup, PICA_REG_INDEX(lighting.max_light_index), c.count-1);
                            u32 enabled{}, spot_disabled{255}, distance_disabled{255};
                            for (unsigned slot = 0; slot < 8; ++slot) {
                                const auto id = Physical(c.order,slot); enabled |= id << (4*slot);
                                if (slot < c.count && slot%3) spot_disabled &= ~(1U<<id);
                                if (slot < c.count && slot%3!=1) distance_disabled &= ~(1U<<id);
                                const auto base = PICA_REG_INDEX(lighting.light[0].specular_0)+16*id;
                                Write(setup, base, PackColor(id,2,brightness));
                                Write(setup, base+1, PackColor(id,3,brightness));
                                Write(setup, base+2, PackColor(id,0,brightness));
                                Write(setup, base+3, PackColor(id,1,brightness));
                                const auto pos = Position(id);
                                Write(setup, base+4, Float16(float(pos[0])) | (Float16(float(pos[1]))<<16));
                                Write(setup, base+5, Float16(float(pos[2])));
                                const auto fixed = [](double x) { return u32(int(x*2048)) & 8191; };
                                Write(setup, base+6, fixed(Spot[0]) | (fixed(Spot[1])<<16));
                                Write(setup, base+7, fixed(Spot[2]));
                                Write(setup, base+9, u32(id%2==0) | (u32(TwoSided(c.order,slot))<<1) |
                                    (u32(bool(slot&1))<<2) | (u32(bool(slot&2))<<3));
                                Write(setup, base+10, Float20(float(Bias(id))));
                                Write(setup, base+11, Float20(float(DistanceScale(id))));
                            }
                            Write(setup, PICA_REG_INDEX(lighting.light_enable), enabled);
                            Write(setup, PICA_REG_INDEX(lighting.global_ambient), (8<<20) | (12<<10) | 20);
                            Write(setup, PICA_REG_INDEX(lighting.config0), (Configs[c.config]<<4) | 12 | ((c.order%2)<<27));
                            Write(setup, PICA_REG_INDEX(lighting.config1), 255 | (1<<18) |
                                  (spot_disabled<<8) | (distance_disabled<<24));
                            Write(setup, PICA_REG_INDEX(lighting.abs_lut_input), c.domain ? 0 : 0x02222222);
                            u32 inputs = 4<<8, scales = ScaleBits[(c.selector+2)%6]<<8;
                            for (unsigned i = 0; i < 6; ++i) {
                                inputs |= Selector(c,i)<<Shifts[i];
                                scales |= ScaleBits[(c.selector+i)%6]<<Shifts[i];
                            }
                            Write(setup, PICA_REG_INDEX(lighting.lut_input), inputs);
                            Write(setup, PICA_REG_INDEX(lighting.lut_scale), scales);
                            using T = Pica::TexturingRegs::TevStageConfig;
                            const auto source = c.output ? T::Source::SecondaryFragmentColor : T::Source::PrimaryFragmentColor;
                            const std::array<unsigned,6> tev{PICA_REG_INDEX(texturing.tev_stage0),PICA_REG_INDEX(texturing.tev_stage1),
                                PICA_REG_INDEX(texturing.tev_stage2),PICA_REG_INDEX(texturing.tev_stage3),
                                PICA_REG_INDEX(texturing.tev_stage4),PICA_REG_INDEX(texturing.tev_stage5)};
                            for (unsigned i = 0; i < tev.size(); ++i) {
                                Write(setup, tev[i], i ? 0xf000f : u32(source) | (u32(source)<<16));
                                Write(setup, tev[i]+1, 0); Write(setup, tev[i]+2, 0); Write(setup, tev[i]+4, 0);
                            }
                            process(setup);
                            const Pica::Shader::FSConfig config{regs};
                            Check(config.lighting.src_num == c.count && unsigned(config.lighting.config.Value()) == Configs[c.config],
                                  "Public raw lighting config/count mismatch");
                            for (unsigned slot = 0; slot < 8; ++slot)
                                Check(config.lighting.lights[slot].two_sided_diffuse == (slot < c.count && TwoSided(c.order,slot)) &&
                                      (slot >= c.count || config.lighting.lights[slot].num == Physical(c.order,slot)),
                                      "Unexpected active/physical FSConfig contract");
                            if (argc == 2 && revision == 0 && reverse == 0 && brightness == 0) {
                                // Public config counterpart, not private Rasterizer cache output.
                                const auto shader = Metal::GeneratePicaFragmentShader(config,{},Metal::FragmentColorFormat::RGBA8,
                                                                                     Metal::BuildPicaSamplerConfigs(regs));
                                std::ofstream file{std::string(argv[1])+"-config"+std::to_string(variant)+".metal"};
                                file << shader.shader.source; Check(bool(file), "Cannot export public shader counterpart"); ++exports;
                            }
                            const auto expected = Expected(c,reverse,brightness,revision);
                            const auto no_diff = Expected(c,reverse,brightness,revision,false);
                            const auto all_supported = Expected(c,reverse,brightness,revision,true,true);
                            auto other_order = c; other_order.order = (c.order+1)%3;
                            const auto order_result = Expected(other_order,reverse,brightness,revision);
                            for (unsigned ch = 0; ch < 4; ++ch) {
                                interpolation += std::abs(int(expected.rgba[ch])-int(no_diff.rgba[ch]))>4;
                                availability += std::abs(int(expected.rgba[ch])-int(all_supported.rgba[ch]))>4;
                                order_sensitive += std::abs(int(expected.rgba[ch])-int(order_result.rgba[ch]))>4;
                                if (revision) mutation += std::abs(int(expected.rgba[ch])-int(
                                    Expected(c,reverse,brightness,revision-1).rgba[ch]))>4;
                                if (ch < 3 && expected.unbounded[ch] > 255)
                                    (c.output ? secondary_saturated : primary_saturated)++;
                            }
                            distinct_fresnel += c.count>1 && std::abs(expected.alpha-expected.first_alpha)>4/255.0;
                            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                                std::vector<u32> triangle;
                                Write(triangle,PICA_REG_INDEX(pipeline.triangle_topology),0);
                                Write(triangle,PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index),15);
                                for (const auto& position : positions) {
                                    Attribute(triangle,position);
                                    Attribute(triangle,reverse ? std::array<float,4>{1,0,0,0} : std::array<float,4>{0,0,0,1});
                                    Attribute(triangle,{1,1,1,1});
                                    Attribute(triangle,{float(View[0]),float(View[1]),float(View[2]),0});
                                }
                                process(triangle); rasterizer.FlushRegion(Memory::VRAM_PADDR,4096);
                                Check(pica.lighting.lut_dirty == 0, "Native multi draw left LUT dirty");
                                for (unsigned pixel = 0; pixel < 1024; ++pixel) for (unsigned ch = 0; ch < 4; ++ch) {
                                    const auto actual = target.GetPtr()[pixel*4+3-ch];
                                    if (std::abs(int(actual)-int(expected.rgba[ch]))>2) {
                                        std::cerr << "revision=" << revision << " variant=" << variant << " config=" << c.config
                                            << " count=" << c.count << " order=" << c.order << " domain=" << c.domain
                                            << " selector=" << c.selector << " output=" << c.output << " reverse=" << reverse
                                            << " brightness=" << brightness << " repeat=" << repeat << " pixel=" << pixel
                                            << " channel=" << ch << " actual=" << unsigned(actual) << " expected=" << expected.rgba[ch] << '\n';
                                        throw std::runtime_error("Public raw multilight/FS/TEV/physical VRAM mismatch");
                                    }
                                }
                                ++draws;
                            }
                        }
                std::cout << "revision " << revision << " complete: " << draws << " draws\n" << std::flush;
            }
            Check(draws == 15840 && raw_entries == 18432 && (argc < 2 || exports == 660), "Incomplete raw multilight matrix");
            Check(interpolation && mutation && availability && order_sensitive && primary_saturated && secondary_saturated &&
                  distinct_fresnel, "Missing raw multilight sensitivity/saturation/Fresnel coverage");
            Check(!system.IsPoweredOn(), "Data-only harness initialized emulation");
            std::cout << draws << " integrated raw multilight/FS/TEV/physical VRAM draws; " << raw_entries
                << " raw entries; all 1024 RGBA pixels; tolerance 2\n"
                << "Oracle coverage: interpolation=" << interpolation << " mutation=" << mutation << " availability=" << availability
                << " order=" << order_sensitive << " primary-saturation=" << primary_saturated << " secondary-saturation=" << secondary_saturated
                << " distinct-Fresnel=" << distinct_fresnel << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
