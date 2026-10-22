// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// This harness constructs core data objects only. It never initializes/loads/runs the
// emulator, creates a window, or opens a ROM. All drawing goes through RasterizerMetal.
#include "mtl_rasterizer.h"
#include "mtl_pica_vertex_fixture.h"
#include "common/color.h"
#include "common/archives.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/image_interface.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include <cstring>
#include <cmath>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace Metal {
struct InputReferenceTestAccess {
    static bool GasPending(const RasterizerMetal& r) { return r.gas_maximum_pending; }
    static BufferBinding GasMaximum(RasterizerMetal& r) {
        return r.gas_runtime.PrepareFragmentMaximum(r.gas_feedback);
    }
    static bool VertexBatchUsed(const RasterizerMetal& r) {return r.gs_vertex_batch_logged;}
    static void ResetVertexBatchMarker(RasterizerMetal& r) {r.gs_vertex_batch_logged=false;}
    static std::size_t PendingGS(const RasterizerMetal& r) {return r.deferred_geometry.size();}
    static NSUInteger ScratchBytes(const RasterizerMetal& r) {return r.gs_emission_scratch.size+r.gs_capture_scratch.size;}
    static bool CompactGS(const RasterizerMetal& r) {
        if(r.deferred_geometry.empty())return false;
        for(const auto& item:r.deferred_geometry) {
            const auto& p=item.invocation.execution;
            if(!p.program || !p.storage_proof || p.capacity!=1000000)return false;
            const auto bound=ShaderMachineRuntime::LinearGeometryStorageBound(*p.program);
            if(!bound || p.result.emissions.size!=NSUInteger(*bound)*sizeof(ShaderMachineEmission) ||
                item.vertices.storage.size!=NSUInteger(*bound)*sizeof(CapturedVertex))return false;
        }
        return true;
    }
    static void DrainGS(RasterizerMetal& r) {r.RetireGeometry(true);}
    static ShaderMachineSnapshot Unit(RasterizerMetal& r) {return r.machine.SaveState(r.gs_unit);}
    static void ResetUnit(RasterizerMetal& r) {r.gs_unit.Reset();}
    static GeometryInputSnapshot Group(RasterizerMetal& r) {return r.geometry_program.SaveState(r.geometry_input);}
    static GeometryInputSnapshot CapturedGroup(const RasterizerMetal& r) {
        const auto& p=r.deferred_geometry.front().invocation;
        GeometryInputSnapshot s{.config=p.config,.cursor=p.cursor,.remaining=p.remaining,.main_remaining=p.main_remaining,
            .total_count=p.total_count,.configured=p.configured,.initialized=p.initialized,.awaiting=p.awaiting};
        std::memcpy(s.history.data(),static_cast<const std::byte*>(p.history.buffer.contents)+p.history.offset,s.history.size());return s;
    }
    static void ResumeReady(RasterizerMetal& r,const ShaderMachineProgramView& program) {
        r.DrawGeometryResult(r.geometry_program.Invoke(r.geometry_input,r.machine,r.gs_unit,program));
    }
};
struct SchedulerTestAccess {
    static void BlockQueue(Scheduler& scheduler, id<MTLSharedEvent> event) {
        scheduler.CheckThread();
        [scheduler.queue waitForEvent:event value:1];
    }
};
}

namespace {
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void Encode(Pica::PixelFormat format, const Common::Vec4<u8>& color, u8* bytes) {
    using namespace Common::Color;
    switch (format) {
    case Pica::PixelFormat::RGBA8: EncodeRGBA8(color, bytes); break;
    case Pica::PixelFormat::RGB8: EncodeRGB8(color, bytes); break;
    case Pica::PixelFormat::RGB565: EncodeRGB565(color, bytes); break;
    case Pica::PixelFormat::RGB5A1: EncodeRGB5A1(color, bytes); break;
    case Pica::PixelFormat::RGBA4: EncodeRGBA4(color, bytes); break;
    }
}
Common::Vec4<u8> Decode(Pica::PixelFormat format, const u8* bytes) {
    using namespace Common::Color;
    switch (format) {
    case Pica::PixelFormat::RGBA8: return DecodeRGBA8(bytes);
    case Pica::PixelFormat::RGB8: return DecodeRGB8(bytes);
    case Pica::PixelFormat::RGB565: return DecodeRGB565(bytes);
    case Pica::PixelFormat::RGB5A1: return DecodeRGB5A1(bytes);
    case Pica::PixelFormat::RGBA4: return DecodeRGBA4(bytes);
    }
    throw std::runtime_error("Invalid fixture color format");
}
unsigned PixelOffset(unsigned x, unsigned y, unsigned width, bool tiled) {
    constexpr std::array<unsigned,8> interleave{0,1,4,5,16,17,20,21};
    return tiled ? ((y / 8) * (width / 8) + x / 8) * 64 + interleave[x % 8] + 2 * interleave[y % 8] : y * width + x;
}
void Write(std::vector<u32>& words, u32 reg, u32 value) {
    words.insert(words.end(), {value, reg | (15U << 16)});
}
u32 Float24(float value) {
    if (value == 0) return 0;
    const auto bits = std::bit_cast<u32>(value);
    const auto raw = ((bits >> 31) << 23) | ((((bits >> 23) & 255) - 64) << 16) |
                     ((bits & 0x7fffff) >> 7);
    Check(Pica::f24::FromRaw(raw).ToFloat32() == value, "Fixture value is not exactly representable in f24");
    return raw;
}
void Attribute(std::vector<u32>& words, const std::array<float, 4>& value) {
    const auto x = Float24(value[0]), y = Float24(value[1]);
    const auto z = Float24(value[2]), w = Float24(value[3]);
    // Grouped command writes exercise the real packed immediate-attribute handler.
    words.insert(words.end(), {(w << 8) | (z >> 16),
        u32(PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0])) |
            (15U << 16) | (2U << 20) | (1U << 31),
        (z << 16) | (y >> 8), (y << 24) | x});
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
}

int main() {
    @autoreleasepool {
        try {
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
            Settings::values.dump_textures.SetValue(false);
            Settings::values.custom_textures.SetValue(false);
            Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::RasterizerMetal rasterizer{memory, pica, custom, renderer, device, scheduler};
            renderer.rasterizer = &rasterizer;
            pica.BindRasterizer(&rasterizer);
            {
                const auto saved = Settings::values.use_disk_shader_cache.GetValue();
                std::vector<VideoCore::LoadCallbackStage> stages;
                const VideoCore::DiskResourceLoadCallback callback =
                    [&](VideoCore::LoadCallbackStage stage, std::size_t, std::size_t, const std::string&) {
                        stages.push_back(stage);
                    };
                const std::vector expected{VideoCore::LoadCallbackStage::Prepare,
                                           VideoCore::LoadCallbackStage::Complete};
                std::atomic_bool cancelled{true};
                rasterizer.LoadDefaultDiskResources(cancelled, callback);
                Check(stages == expected, "Cancelled disk preload callback sequence");
                stages.clear();
                Settings::values.use_disk_shader_cache.SetValue(false);
                cancelled.store(false);
                rasterizer.LoadDefaultDiskResources(cancelled, callback);
                Check(stages == expected, "Disabled disk preload callback sequence");
                stages.clear();
                VideoCore::RasterizerInterface::SetSwitchDiskResourcesCallback(callback);
                rasterizer.SwitchDiskResources(123);
                VideoCore::RasterizerInterface::SetSwitchDiskResourcesCallback({});
                Check(stages == expected && !device.DiskCacheStats().enabled,
                      "Optional storage-free title switch");
                Settings::values.use_disk_shader_cache.SetValue(saved);
                std::cout << "Native disk-resource cancelled/disabled preload and storage-free title callbacks passed without loader/window/emulation\n";
            }
            auto& regs = pica.regs.internal;
            regs = {};
            Metal::Test::MovVertex(regs, pica.vs_setup);
            regs.vs.max_input_attribute_index.Assign(2);
            regs.pipeline.max_input_attrib_index.Assign(2);
            regs.vs.input_attribute_to_register_map_low = 0x210;
            regs.lighting.disable.Assign(1);
            regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
            regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR / 8);
            regs.framebuffer.framebuffer.width.Assign(32);
            regs.framebuffer.framebuffer.height.Assign(31);
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.viewport_size_x.Assign(67 << 16); // PICA f24: 16, half viewport width.
            regs.rasterizer.viewport_size_y.Assign(67 << 16);
            regs.rasterizer.viewport_depth_near_plane.Assign(62 << 16); // 0.5
            pica.dirty_regs.SetAllDirty();
            auto target = memory.GetPhysicalRef(Memory::VRAM_PADDR);
            Check(target && target.GetSize() >= 32 * 32 * 4, "No physical VRAM backing in the data-only harness");
            std::memset(target.GetPtr(), 0x11, 32 * 32 * 4);
            const std::array<std::array<float, 4>, 3> positions{{{-1,-1,-.5f,1}, {3,-1,-.5f,1}, {-1,3,-.5f,1}}};
            for (unsigned i = 0; i < 3; ++i) {
                for (unsigned c = 0; c < 4; ++c) {
                    pica.immediate.input_vertex[0][c] = Pica::f24::FromFloat32(positions[i][c]);
                    pica.immediate.input_vertex[1][c] = Pica::f24::FromFloat32(c == 3 ? 1.f : 0.f);
                    pica.immediate.input_vertex[2][c] = Pica::f24::FromFloat32(c == 1 ? 0.f : 1.f);
                }
                Check(rasterizer.AccelerateImmediateVertex(), "Native rasterizer rejected immediate input");
                rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                if (i < 2) {
                    for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                        Check(target.GetPtr()[byte] == 0x11, "Incomplete immediate triangle changed physical VRAM");
                }
            }
            // Guest RGBA8 is stored A,B,G,R. A solid magenta output avoids sharing the
            // production Morton decoder as an oracle and checks every physical guest byte.
            for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                Check(target.GetPtr()[byte] == (byte % 4 == 2 ? 0 : 255),
                      "Native immediate VS/FS/cache/guest flush did not produce the expected VRAM bytes");

            // Match Vulkan/OpenGL's logical framebuffer contract. The source must
            // remain GPU-owned: stale guest RAM cannot produce this magenta image.
            for (unsigned framebuffer_mode = 0; framebuffer_mode < 2; ++framebuffer_mode)
                for (bool transfer_block32 : {false, true}) {
                    rasterizer.ClearAll(false);
                    rasterizer.ResetMetalPrimitiveStream();
                    regs.reg_array[0x11b] = framebuffer_mode;
                    pica.dirty_regs.SetAllDirty();
                    std::memset(target.GetPtr(), 0x11, 32 * 32 * 4);
                    constexpr PAddr screen_address = Memory::VRAM_PADDR + 0x4000;
                    auto* output = memory.GetPhysicalPointer(screen_address);
                    std::memset(output, 0xCD, 32 * 32 * 4 + 32);
                    for (unsigned vertex = 0; vertex < 3; ++vertex) {
                        for (unsigned c = 0; c < 4; ++c) {
                            pica.immediate.input_vertex[0][c] = Pica::f24::FromFloat32(positions[vertex][c]);
                            pica.immediate.input_vertex[1][c] = Pica::f24::FromFloat32(c == 3 ? 1.f : 0.f);
                            pica.immediate.input_vertex[2][c] = Pica::f24::FromFloat32(c == 1 ? 0.f : 1.f);
                        }
                        Check(rasterizer.AccelerateImmediateVertex(), "Logical block32 draw required a CPU renderer");
                    }
                    Pica::DisplayTransferConfig config{};
                    config.input_address = Memory::VRAM_PADDR / 8;
                    config.output_address = (screen_address + 16) / 8;
                    config.input_width.Assign(32); config.input_height.Assign(32);
                    config.output_width.Assign(32); config.output_height.Assign(32);
                    config.block_32.Assign(transfer_block32);
                    Check(rasterizer.AccelerateDisplayTransfer(config), "Logical block32 transfer required a CPU renderer");
                    Pica::FramebufferConfig lcd{};
                    lcd.width.Assign(32); lcd.height.Assign(32);
                    lcd.color_format.Assign(Pica::PixelFormat::RGBA8);
                    Metal::ScreenImage screen{};
                    Check(rasterizer.GetScreen(lcd, screen_address + 16, 32, screen),
                          "GPU-owned block32-compatible framebuffer was unavailable to LCD output");
                    const auto origin_x = NSUInteger(std::lround(screen.texcoords.top * screen.texture.width));
                    const auto origin_y = NSUInteger(std::lround(screen.texcoords.left * screen.texture.height));
                    const auto pixels = Metal::TransferRuntime{device, scheduler}.Readback({screen.texture, 0, 0,
                        MTLRegionMake2D(origin_x, origin_y, 32, 32)}, 128);
                    for (unsigned byte = 0; byte < pixels.size(); ++byte)
                        Check(std::to_integer<u8>(pixels[byte]) == (byte % 4 == 1 ? 0 : 255),
                              "Logical block32 draw/transfer/LCD chain lost the GPU-rendered source");
                    for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                        Check(target.GetPtr()[byte] == 0x11,
                              "Logical block32 presentation unnecessarily flushed the source framebuffer");
                    rasterizer.FlushRegion(screen_address + 16, 32 * 32 * 4);
                    for (unsigned byte = 0; byte < 32 * 32 * 4 + 32; ++byte) {
                        const auto expected = byte < 16 || byte >= 16 + 32 * 32 * 4 ? 0xCD :
                            (byte % 4 == 2 ? 0 : 255);
                        Check(output[byte] == expected, "Logical block32 output flush lost color or guards");
                    }
                }
            regs.reg_array[0x11b] = 0;
            std::cout << "4 GPU-owned framebuffer-mode/transfer-bit combinations: native draw to cached display/LCD, stale source RAM, output flush and guards passed; logical cache compatibility only\n";

            {
                // Exercise full display dimensions with a nonuniform GPU-owned
                // source. Uniform colors cannot detect a per-row pixel shift.
                // The Block32 flag cases test the logical cache contract only;
                // their oracle is not an independently captured Block32 layout.
                const auto saved_regs = regs;
                constexpr PAddr source_address = Memory::VRAM_PADDR + 0x100000;
                constexpr PAddr output_address = Memory::VRAM_PADDR + 0x200000;
                const auto tile_color = [](unsigned x, unsigned y) {
                    const auto tx = x / 32, ty = y / 32;
                    return Common::Vec4<u8>{u8(tx * 31), u8(ty * 19),
                                           u8(tx * 13 + ty * 7), 255};
                };
                unsigned cases = 0;
                for (const auto size : {std::array<unsigned, 2>{240, 400},
                                        std::array<unsigned, 2>{256, 416}})
                    for (unsigned framebuffer_mode = 0; framebuffer_mode < 2; ++framebuffer_mode) {
                        rasterizer.ClearAll(false);
                        rasterizer.ResetMetalPrimitiveStream();
                        const auto [width, height] = size;
                        const auto source_bytes = width * height * 4;
                        auto* source = memory.GetPhysicalPointer(source_address);
                        std::memset(source, 0x11, source_bytes);
                        regs.framebuffer.framebuffer.color_buffer_address.Assign(source_address / 8);
                        regs.framebuffer.framebuffer.width.Assign(width);
                        regs.framebuffer.framebuffer.height.Assign(height - 1);
                        regs.framebuffer.framebuffer.flip.Assign(1);
                        regs.reg_array[0x11b] = framebuffer_mode;
                        regs.rasterizer.viewport_size_x.Assign(Float24(width / 2.f));
                        regs.rasterizer.viewport_size_y.Assign(Float24(height / 2.f));
                        regs.rasterizer.scissor_test.mode.Assign(Pica::RasterizerRegs::ScissorMode::Include);
                        for (unsigned y = 0; y < height; y += 32)
                            for (unsigned x = 0; x < width; x += 32) {
                                regs.rasterizer.scissor_test.x1.Assign(x);
                                regs.rasterizer.scissor_test.x2.Assign(std::min(x + 32, width) - 1);
                                regs.rasterizer.scissor_test.y1.Assign(y);
                                regs.rasterizer.scissor_test.y2.Assign(std::min(y + 32, height) - 1);
                                pica.dirty_regs.SetAllDirty();
                                const auto color = tile_color(x, y);
                                for (unsigned vertex = 0; vertex < 3; ++vertex) {
                                    for (unsigned c = 0; c < 4; ++c) {
                                        pica.immediate.input_vertex[0][c] = Pica::f24::FromFloat32(positions[vertex][c]);
                                        pica.immediate.input_vertex[1][c] = Pica::f24::FromFloat32(c == 3 ? 1.f : 0.f);
                                        pica.immediate.input_vertex[2][c] = Pica::f24::FromFloat32(color[c] / 255.f);
                                    }
                                    Check(rasterizer.AccelerateImmediateVertex(), "Display-size pattern draw required CPU execution");
                                }
                            }
                        for (const auto format : {Pica::PixelFormat::RGBA8, Pica::PixelFormat::RGB8})
                            for (bool block32 : {false, true}) for (bool flip : {false, true}) {
                                const auto bpp = Pica::BytesPerPixel(format);
                                const auto bytes = width * height * bpp;
                                rasterizer.InvalidateRegion(output_address, source_bytes + 32);
                                auto* output = memory.GetPhysicalPointer(output_address);
                                std::memset(output, 0xCD, source_bytes + 32);
                                Pica::DisplayTransferConfig config{};
                                config.input_address = source_address / 8;
                                config.output_address = (output_address + 16) / 8;
                                config.input_width.Assign(width); config.input_height.Assign(height);
                                config.output_width.Assign(width); config.output_height.Assign(height);
                                config.output_format.Assign(format); config.block_32.Assign(block32);
                                config.flip_vertically.Assign(flip);
                                Check(rasterizer.AccelerateDisplayTransfer(config), "Display-size transfer required CPU execution");
                                Pica::FramebufferConfig lcd{};
                                lcd.width.Assign(width); lcd.height.Assign(height); lcd.color_format.Assign(format);
                                Metal::ScreenImage screen{};
                                Check(rasterizer.GetScreen(lcd, output_address + 16, width, screen),
                                      "Display-size GPU transfer was unavailable to LCD output");
                                const auto ox = NSUInteger(std::lround(screen.texcoords.top * screen.texture.width));
                                const auto oy = NSUInteger(std::lround(screen.texcoords.left * screen.texture.height));
                                const auto pixels = Metal::TransferRuntime{device, scheduler}.Readback(
                                    {screen.texture, 0, 0, MTLRegionMake2D(ox, oy, width, height)}, width * 4);
                                rasterizer.FlushRegion(output_address + 16, bytes);
                                std::vector<u8> expected(source_bytes + 32, 0xCD);
                                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                                    const auto color = tile_color(x, flip ? y : height - 1 - y);
                                    Encode(format, color, expected.data() + 16 + (y * width + x) * bpp);
                                    for (unsigned c = 0; c < 4; ++c)
                                        Check(std::to_integer<u8>(pixels[(y * width + x) * 4 + c]) == color[c],
                                              "Display-size LCD readback shifted a row or channel");
                                }
                                Check(std::equal(expected.begin(), expected.end(), output),
                                      "Display-size transfer shifted rows/channels or changed physical guards");
                                for (unsigned byte = 0; byte < source_bytes; ++byte)
                                    Check(source[byte] == 0x11, "Display-size transfer flushed its GPU-owned source");
                                ++cases;
                            }
                    }
                Check(cases == 32, "Incomplete display-size GPU transfer matrix");
                regs = saved_regs;
                pica.dirty_regs.SetAllDirty();
                rasterizer.ClearAll(false);
                rasterizer.ResetMetalPrimitiveStream();
                std::cout << "32 GPU-owned 240x400/256x416 transfers: patterned scissor draws, RGBA8/RGB8 LCD and physical readback, flips, both framebuffer modes/transfer flags, stale source RAM and guards passed; logical Block32 contract only\n";
            }

            // In stage zero Previous selects that component's third source;
            // it becomes zero only if the third source is Previous as well.
            // Later stages must keep reading the actual preceding output.
            {
                using Tev = Pica::TexturingRegs::TevStageConfig;
                const auto saved_regs = regs;
                std::array<Tev*, 6> stages{&regs.texturing.tev_stage0, &regs.texturing.tev_stage1,
                    &regs.texturing.tev_stage2, &regs.texturing.tev_stage3,
                    &regs.texturing.tev_stage4, &regs.texturing.tev_stage5};
                constexpr std::array sources{Tev::Source::Previous, Tev::Source::Constant,
                                              Tev::Source::PrimaryColor};
                constexpr std::array<unsigned, 4> constant{17, 67, 149, 211};
                constexpr std::array<float, 4> primary{.25f, .5f, .75f, 1.f};
                constexpr std::array<unsigned, 4> primary_bytes{64, 128, 191, 255};
                unsigned cases{}, mismatches{};
                for (auto color_source : sources) for (auto alpha_source : sources) {
                    rasterizer.ClearAll(false);
                    rasterizer.ResetMetalPrimitiveStream();
                    for (auto* stage : stages) {
                        *stage = {};
                        stage->color_source1.Assign(Tev::Source::Previous);
                        stage->alpha_source1.Assign(Tev::Source::Previous);
                        stage->color_source3.Assign(Tev::Source::Constant);
                        stage->alpha_source3.Assign(Tev::Source::Constant);
                        stage->const_color = 0x7351290B;
                    }
                    auto& first = *stages[0];
                    first.color_source3.Assign(color_source);
                    first.alpha_source3.Assign(alpha_source);
                    first.const_color = constant[0] | (constant[1] << 8) |
                                        (constant[2] << 16) | (constant[3] << 24);
                    pica.dirty_regs.SetAllDirty();
                    std::memset(target.GetPtr(), 0x11, 32 * 32 * 4);
                    for (unsigned vertex = 0; vertex < 3; ++vertex) {
                        for (unsigned c = 0; c < 4; ++c) {
                            pica.immediate.input_vertex[0][c] = Pica::f24::FromFloat32(positions[vertex][c]);
                            pica.immediate.input_vertex[1][c] = Pica::f24::FromFloat32(c == 3 ? 1.f : 0.f);
                            pica.immediate.input_vertex[2][c] = Pica::f24::FromFloat32(primary[c]);
                        }
                        Check(rasterizer.AccelerateImmediateVertex(), "First-stage Previous draw required a CPU renderer");
                    }
                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                    const auto expected = [&](unsigned c) {
                        const auto source = c == 3 ? alpha_source : color_source;
                        return source == Tev::Source::Constant ? constant[c] :
                            source == Tev::Source::PrimaryColor ? primary_bytes[c] : 0U;
                    };
                    unsigned case_mismatches{};
                    for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                        case_mismatches += target.GetPtr()[byte] != expected(3 - byte % 4);
                    if (case_mismatches)
                        std::cerr << "First-stage Previous color3=" << unsigned(color_source)
                                  << " alpha3=" << unsigned(alpha_source) << ": "
                                  << case_mismatches << " incorrect guest bytes\n";
                    mismatches += case_mismatches;
                    ++cases;
                }
                regs = saved_regs;
                rasterizer.ClearAll(false);
                rasterizer.ResetMetalPrimitiveStream();
                pica.dirty_regs.SetAllDirty();
                Check(!mismatches, "First-stage Previous was incorrectly treated as passthrough");
                std::cout << cases << " native stage-zero Previous color/alpha source pairs: all 36864 guest bytes exact; later-stage Previous preserved\n";
            }

            const auto command_address = Memory::VRAM_PADDR + 0x8000;
            const auto vertex_address = Memory::VRAM_PADDR + 0x10000;
            const auto index_address = vertex_address + 0x2000;
            const auto process = [&](const std::vector<u32>& words) {
                const auto bytes = words.size() * sizeof(u32);
                auto backing = memory.GetPhysicalRef(command_address);
                Check(backing && backing.GetSize() >= bytes, "Invalid fixture command backing");
                std::memcpy(backing.GetPtr(), words.data(), bytes);
                pica.ProcessCmdList(command_address, u32(bytes), false);
            };
            const auto check_target = [&](bool complete) {
                rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                    Check(target.GetPtr()[byte] == (complete ? (byte % 4 == 2 ? 0 : 255) : 0x11),
                          "PICA command dispatch/native primitive continuation produced incorrect VRAM bytes");
            };
            const auto reset_target = [&] {
                rasterizer.ClearAll(false);
                std::memset(target.GetPtr(), 0x11, 32 * 32 * 4);
                std::vector<u32> words;
                Write(words, PICA_REG_INDEX(pipeline.triangle_topology), 0);
                Write(words, PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index), 15);
                process(words);
            };
            regs.pipeline.max_input_attrib_index.Assign(2);
            const auto immediate_vertex = [&](unsigned index) {
                std::vector<u32> words;
                Attribute(words, positions[index]);
                Attribute(words, {0, 0, 0, 1});
                Attribute(words, {1, 0, 1, 1});
                process(words);
            };
            reset_target();
            for (unsigned index = 0; index < 3; ++index) {
                immediate_vertex(index);
                check_target(index == 2);
            }
            std::cout << "PICA packed command-list immediate dispatch and partial primitive preservation passed\n";

            regs.pipeline.vertex_attributes.base_address.Assign(vertex_address / 16);
            regs.pipeline.vertex_attributes.max_attribute_index.Assign(2);
            regs.pipeline.vertex_attributes.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            regs.pipeline.vertex_attributes.format1.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            regs.pipeline.vertex_attributes.format2.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            regs.pipeline.vertex_attributes.size0.Assign(3);
            regs.pipeline.vertex_attributes.size1.Assign(3);
            regs.pipeline.vertex_attributes.size2.Assign(3);
            auto& loader = regs.pipeline.vertex_attributes.attribute_loaders[0];
            loader.comp0.Assign(0); loader.comp1.Assign(1); loader.comp2.Assign(2);
            loader.component_count.Assign(3); loader.byte_count.Assign(48);
            std::array<std::array<float, 12>, 3> raw_vertices{};
            for (unsigned index = 0; index < 3; ++index) {
                std::copy(positions[index].begin(), positions[index].end(), raw_vertices[index].begin());
                raw_vertices[index][7] = 1;
                raw_vertices[index][8] = raw_vertices[index][10] = raw_vertices[index][11] = 1;
            }
            std::memcpy(memory.GetPhysicalPointer(vertex_address), raw_vertices.data(), sizeof(raw_vertices));
            const auto batch = [&](u32 first, u32 count, bool indexed) {
                std::vector<u32> words;
                Write(words, PICA_REG_INDEX(pipeline.vertex_offset), first);
                Write(words, PICA_REG_INDEX(pipeline.num_vertices), count);
                Write(words, indexed ? PICA_REG_INDEX(pipeline.trigger_draw_indexed) :
                                       PICA_REG_INDEX(pipeline.trigger_draw), 1);
                process(words);
            };
            reset_target();
            batch(0, 3, false);
            check_target(true);
            {
                const auto saved = regs;
                using Tev = Pica::TexturingRegs::TevStageConfig;
                const auto texture_address = Memory::VRAM_PADDR + 0x40000;
                constexpr std::array<Common::Vec4<u8>, 2> colors{{{37, 89, 173, 255}, {211, 43, 67, 255}}};
                const auto select = [&](unsigned unit, bool active) {
                    std::array<Tev*, 6> stages{&regs.texturing.tev_stage0, &regs.texturing.tev_stage1,
                        &regs.texturing.tev_stage2, &regs.texturing.tev_stage3,
                        &regs.texturing.tev_stage4, &regs.texturing.tev_stage5};
                    for (auto* stage : stages) stage->sources_raw = 0xF000F;
                    const auto source = active ? Tev::Source(unsigned(Tev::Source::Texture0) + unit) :
                                                 Tev::Source::PrimaryColor;
                    stages[0]->color_source1.Assign(source);
                    stages[0]->alpha_source1.Assign(source);
                    pica.dirty_regs.SetAllDirty();
                };
                const auto verify = [&](const Common::Vec4<u8>& color) {
                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                    std::array<u8, 4> expected{};
                    Encode(Pica::PixelFormat::RGBA8, color, expected.data());
                    for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                        Check(target.GetPtr()[byte] == expected[byte % 4],
                              "Inactive texture activation/invalidation lost native VRAM contents");
                };
                for (unsigned unit = 0; unit < 3; ++unit) {
                    regs = saved;
                    reset_target();
                    regs.texturing.main_config.texture0_enable.Assign(1);
                    regs.texturing.main_config.texture1_enable.Assign(1);
                    regs.texturing.main_config.texture2_enable.Assign(1);
                    for (auto* texture : {&regs.texturing.texture0, &regs.texturing.texture1, &regs.texturing.texture2}) {
                        texture->width.Assign(8); texture->height.Assign(8);
                        texture->address.Assign(texture_address / 8);
                    }
                    regs.texturing.texture0_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                    regs.texturing.texture1_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                    regs.texturing.texture2_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                    for (unsigned pixel = 0; pixel < 64; ++pixel)
                        Encode(Pica::PixelFormat::RGBA8, colors[0], memory.GetPhysicalPointer(texture_address) + pixel * 4);
                    select(unit, false);
                    batch(0, 3, false);
                    verify({255, 0, 255, 255});
                    select(unit, true);
                    batch(0, 3, false);
                    verify(colors[0]);
                    select(unit, false);
                    for (unsigned pixel = 0; pixel < 64; ++pixel)
                        Encode(Pica::PixelFormat::RGBA8, colors[1], memory.GetPhysicalPointer(texture_address) + pixel * 4);
                    rasterizer.InvalidateRegion(texture_address, 8 * 8 * 4);
                    batch(0, 3, false);
                    verify({255, 0, 255, 255});
                    select(unit, true);
                    batch(0, 3, false);
                    verify(colors[1]);
                }
                regs = saved;
                reset_target();
                batch(0, 3, false);
                check_target(true);
                std::cout << "All three enabled texture units: inactive/cold, activation, inactive invalidation and reactivation; 49152 native VRAM bytes passed\n";
            }
            reset_target();
            batch(0, 2, false);
            check_target(false);
            immediate_vertex(2);
            check_target(true);
            for (unsigned index_size : {1U, 2U}) {
                reset_target();
                immediate_vertex(0);
                check_target(false);
                const std::array<u8, 4> indices = index_size == 1 ? std::array<u8, 4>{1, 2, 0, 0} :
                                                                 std::array<u8, 4>{1, 0, 2, 0};
                std::memcpy(memory.GetPhysicalPointer(index_address), indices.data(), indices.size());
                regs.pipeline.index_array.offset.Assign(index_address - vertex_address);
                regs.pipeline.index_array.format.Assign(index_size == 2 ? decltype(regs.pipeline.index_array)::SHORT :
                                                                          decltype(regs.pipeline.index_array)::BYTE);
                batch(99, 2, true); // Indexed draws must ignore vertex_offset.
                check_target(true);
            }
            reset_target();
            batch(0, 2, false);
            std::vector<u32> restart;
            Write(restart, PICA_REG_INDEX(pipeline.restart_primitive), 1);
            process(restart);
            immediate_vertex(2);
            check_target(false);
            batch(0, 2, false);
            check_target(true);
            std::cout << "PICA batch, indexed 8/16-bit, mixed immediate/batch and restart dispatch passed through real cache/VRAM\n";

            {
                const auto saved_regs = regs;
                const auto saved_fog = pica.fog;
                const auto saved_lighting = pica.lighting;
                const auto saved_native = rasterizer.SaveMetalState();
                reset_target();
                // Known initial banks; subsequent changes go through real PICA
                // register commands and preserve all inactive LUT tables.
                for (auto& lut : pica.lighting.luts) for (auto& entry : lut) entry.raw = 0;
                pica.lighting.lut_dirty = pica.lighting.LutAllDirty;
                regs.texturing.fog_mode.Assign(Pica::TexturingRegs::FogMode::Fog);
                regs.texturing.fog_color.raw = 0x00FF00; // green vs the magenta triangle
                pica.dirty_regs.SetAllDirty();
                unsigned fog_value{};
                for (unsigned round = 0; round < 440; ++round) {
                    std::vector<u32> commands;
                    if (!round || round % 3) {
                        fog_value = (round & 1) ? 2047 : 0;
                        Write(commands, PICA_REG_INDEX(texturing.fog_lut_offset), 0);
                        for (unsigned i = 0; i < 128; ++i)
                            Write(commands, PICA_REG_INDEX(texturing.fog_lut_data[0]), fog_value << 13);
                    }
                    // A single Lighting table change must preserve the Fog
                    // offset, including at the native texture-page boundary.
                    Write(commands, PICA_REG_INDEX(lighting.lut_config), (round % 24) << 8);
                    Write(commands, PICA_REG_INDEX(lighting.lut_data[0]), round + 1);
                    process(commands);
                    batch(0, 3, false);
                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                    for (unsigned b = 0; b < 32 * 32 * 4; ++b) {
                        const auto expected = b % 4 == 0 ? 255u : b % 4 == 2 ?
                            (fog_value ? 0u : 255u) : (fog_value ? 255u : 0u);
                        if (target.GetPtr()[b] != expected)
                            throw std::runtime_error("Dirty Lighting/Fog stream mismatch at round " +
                                std::to_string(round) + ", byte " + std::to_string(b) + ": got " +
                                std::to_string(target.GetPtr()[b]) + ", expected " + std::to_string(expected));
                    }
                    if (round == 417) {
                        const auto state = rasterizer.SaveMetalState();
                        rasterizer.LoadMetalState(state); // invalidate views, retain actual PICA LUT values
                    }
                }
                regs = saved_regs; pica.fog = saved_fog; pica.lighting = saved_lighting;
                rasterizer.LoadMetalState(saved_native);
                rasterizer.ClearAll(false); pica.dirty_regs.SetAllDirty();
                std::cout << "Dirty native LUT stream:440 public PICA Lighting/Fog draws, unchanged Fog offsets, "
                             "page append, Save/Load and all1802240 physical VRAM bytes exact\n";
            }

            {
                // Public command lists -> raw Gas/Fog banks -> native uniforms ->
                // generated MSL -> actual cache writeback. No private LUT injection.
                const auto saved_regs = regs;
                const auto saved_fog = pica.fog;
                const auto saved_native = rasterizer.SaveMetalState();
                reset_target();
                std::vector<u32> commands;
                Write(commands, 0x123, 0);
                for (unsigned i = 0; i < 16; ++i)
                    Write(commands, 0x124, i < 8 ? 0 : 0x776655 + (i - 8) * 0x080808);
                Write(commands, PICA_REG_INDEX(texturing.fog_lut_offset), 0);
                for (unsigned i = 0; i < 128; ++i)
                    Write(commands, PICA_REG_INDEX(texturing.fog_lut_data[0]), 1092u << 13);
                Write(commands, 0xE4, 0x3C00);
                Write(commands, 0xE5, 0x3C00);
                auto stage = regs.texturing.tev_stage5;
                stage.color_source3.Assign(Pica::TexturingRegs::TevStageConfig::Source::Constant);
                Write(commands, PICA_REG_INDEX(texturing.tev_stage5.sources_raw), stage.sources_raw);
                // TEV register RGBA components encode D1=1 and D2=0.5.
                Write(commands, PICA_REG_INDEX(texturing.tev_stage5.const_color), 0x3C003800);
                process(commands);
                for (unsigned shading = 0; shading < 2; ++shading) {
                    reset_target();
                    std::vector<u32> mode;
                    Write(mode, 0xE0, (saved_regs.reg_array[0xE0] & ~15u) | 7u | shading << 3);
                    process(mode);
                    batch(0, 3, false);
                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                    const unsigned index = shading ? 4 : 7;
                    const std::array<u8, 4> expected{136, u8(119 + index * 8),
                                                   u8(102 + index * 8), u8(85 + index * 8)};
                    for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                        Check(target.GetPtr()[byte] == expected[byte % 4],
                              "Public PICA Gas postsynthesis lost raw LUT/Fog registers, density selection or VRAM bytes");
                }
                regs = saved_regs;
                pica.fog = saved_fog;
                rasterizer.LoadMetalState(saved_native);
                rasterizer.ClearAll(false);
                pica.dirty_regs.SetAllDirty();
                std::cout << "Public PICA Gas command lists -> native postsynthesis -> 2048 physical VRAM pixels passed\n";
            }

            {
                const auto saved_regs = regs;
                const auto saved_native = rasterizer.SaveMetalState();
                constexpr auto depth_address = Memory::VRAM_PADDR + 0x4000;
                constexpr unsigned triangles = 32;
                unsigned cases{};
                for (auto format : {Pica::FramebufferRegs::DepthFormat::D16,
                                    Pica::FramebufferRegs::DepthFormat::D24,
                                    Pica::FramebufferRegs::DepthFormat::D24S8})
                    for (unsigned function = 0; function < 4; ++function)
                        for (unsigned path = 0; path < 4; ++path) {
                            regs = saved_regs;
                            reset_target();
                            std::memset(target.GetPtr(), 0, 32 * 32 * 4);
                            regs.framebuffer.framebuffer.depth_buffer_address.Assign(depth_address / 8);
                            regs.framebuffer.framebuffer.depth_format.Assign(format);
                            // Gas must read scene depth even when ordinary depth testing is off.
                            regs.framebuffer.output_merger.depth_test_enable.Assign(0);
                            regs.framebuffer.output_merger.depth_write_enable.Assign(0);
                            std::vector<u32> commands;
                            Write(commands, 0x100, (regs.reg_array[0x100] & ~3u) | 1u);
                            Write(commands, 0x126, (function << 24) | 256);
                            Write(commands, 0x125, 0);
                            process(commands);
                            pica.dirty_regs.SetAllDirty();
                            const unsigned bpp = format == Pica::FramebufferRegs::DepthFormat::D16 ? 2 :
                                                 format == Pica::FramebufferRegs::DepthFormat::D24 ? 3 : 4;
                            std::vector<u8> scene(32 * 32 * bpp);
                            for (unsigned i = 0; i < 32 * 32; ++i) {
                                scene[i * bpp] = 0xff;
                                scene[i * bpp + 1] = bpp == 2 ? 0xbf : 0xff;
                                if (bpp >= 3) scene[i * bpp + 2] = 0xbf;
                                if (bpp == 4) scene[i * bpp + 3] = 0xa7;
                            }
                            std::memcpy(memory.GetPhysicalPointer(depth_address), scene.data(), scene.size());
                            std::vector<std::array<float, 12>> vertices(triangles * 3);
                            for (unsigned i = 0; i < vertices.size(); ++i) vertices[i] = raw_vertices[i % 3];
                            std::memcpy(memory.GetPhysicalPointer(vertex_address), vertices.data(), vertices.size() * sizeof(vertices[0]));
                            if (path == 0) batch(0, triangles * 3, false);
                            else if (path == 1) {
                                for (unsigned i = 0; i < triangles * 3; ++i) immediate_vertex(i % 3);
                            } else if (path == 2) {
                                for (unsigned i = 0; i < triangles * 3; ++i)
                                    memory.GetPhysicalPointer(index_address)[i] = i % 3;
                                regs.pipeline.index_array.offset.Assign(index_address - vertex_address);
                                regs.pipeline.index_array.format.Assign(decltype(regs.pipeline.index_array)::BYTE);
                                batch(99, triangles * 3, true);
                            } else {
                                batch(0, triangles * 3 - 2, false);
                                immediate_vertex(1); immediate_vertex(2);
                            }
                            rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                            const std::array<u8, 4> expected{0x50, 0,
                                u8(function == 0 ? 0 : function == 1 ? 0x50 : function == 2 ? 0x4e : 0x48), 0};
                            for (unsigned i = 0; i < 32 * 32 * 4; ++i)
                                if (target.GetPtr()[i] != expected[i % 4]) {
                                    std::cerr << "Gas core depth_format=" << unsigned(format) << " function=" << function
                                              << " path=" << path << " byte=" << i << " actual=" << unsigned(target.GetPtr()[i])
                                              << " expected=" << unsigned(expected[i % 4]) << '\n';
                                    throw std::runtime_error("Public PICA Gas accumulation produced incorrect VRAM bytes");
                                }
                            Check((regs.reg_array[0x125] & 0xffff) == 0 &&
                                  Metal::InputReferenceTestAccess::GasPending(rasterizer),
                                  "Gas draw published CPU feedback before an observation boundary");
                            renderer.Rasterizer()->SynchronizeMetalGasMaximum();
                            Check((regs.reg_array[0x125] & 0xffff) == 0x5000 &&
                                  !Metal::InputReferenceTestAccess::GasPending(rasterizer),
                                  "Native GPU Gas maximum did not reach the PICA feedback register");
                            std::vector<u32> reciprocal;
                            Write(reciprocal, 0xE5, 0xabcd0101);
                            process(reciprocal);
                            rasterizer.UpdateMetalGasAccumulatorMaximum();
                            Check(regs.reg_array[0xE5] == 0xabcd2800,
                                  "Native GPU reciprocal feedback lost value/register upper bits");
                            rasterizer.FlushRegion(depth_address, u32(scene.size()));
                            Check(!std::memcmp(memory.GetPhysicalPointer(depth_address), scene.data(), scene.size()),
                                  "Gas accumulation altered scene depth/stencil VRAM");
                            const auto native = rasterizer.SaveMetalState();
                            Metal::GasTransferState state{};
                            Metal::UnpackGasArchive(native, state);
                            Check(Metal::GasArchiveWord(native, 8) == 4 && state.maximum == 0x5000 && !state.feedback_pending,
                                  "Actual PICA Gas maximum was not serialized in native version 4");
                            std::stringstream stream;
                            { oarchive archive{stream}; archive << pica; }
                            rasterizer.LoadMetalState(saved_native);
                            { iarchive archive{stream}; archive >> pica; }
                            Check(rasterizer.SaveMetalState() == native, "PICA archive lost actual accumulated GPU maximum");
                            std::vector<u32> reset;
                            Write(reset, 0x125, 0);
                            process(reset);
                            Metal::UnpackGasArchive(rasterizer.SaveMetalState(), state);
                            Check(state.maximum == 0 && state.feedback_pending, "Ordered feedback reset retained the GPU maximum");
                            rasterizer.UpdateMetalGasAccumulatorMaximum();
                            Check(regs.reg_array[0xE5] == 0xabcd7c00,
                                  "Native GPU reciprocal after reset retained a stale maximum");
                            ++cases;
                        }
                // The cache/shaders/depth images above are warm. Hold the native
                // queue while actual public PICA draws accumulate, then observe
                // through each CPU boundary. A watchdog makes regressions fail
                // without hanging the suite if a draw reintroduces a CPU wait.
                for (unsigned boundary = 0; boundary < 4; ++boundary) {
                    auto gate = [device.Handle() newSharedEvent];
                    struct Release {
                        std::mutex mutex;
                        std::condition_variable changed;
                        bool done{};
                        id<MTLSharedEvent> event;
                        std::jthread thread;
                        explicit Release(id<MTLSharedEvent> e) : event{e}, thread{[this] {
                            std::unique_lock lock{mutex};
                            changed.wait_for(lock, std::chrono::seconds{2}, [&] { return done; });
                            event.signaledValue = 1;
                        }} {}
                        ~Release() {
                            { std::scoped_lock lock{mutex}; done = true; }
                            changed.notify_all();
                        }
                    } release{gate};
                    // Keep the depth texture warm but reset the guest Gas target.
                    reset_target();
                    std::memset(target.GetPtr(), 0, 32 * 32 * 4);
                    std::vector<u32> reset;
                    Write(reset, 0x125, 0xa5c30000);
                    process(reset);
                    scheduler.Finish();
                    const auto tick = scheduler.CurrentTick();
                    Metal::SchedulerTestAccess::BlockQueue(scheduler, gate);
                    for (unsigned i = 0; i < 32; ++i) batch(0, 3, false);
                    Check(gate.signaledValue == 0 && scheduler.CurrentTick() == tick &&
                          regs.reg_array[0x125] == 0xa5c30000 &&
                          Metal::InputReferenceTestAccess::GasPending(rasterizer),
                          "Gas draw waited/submitted or prematurely published a deferred maximum");
                    bool foreign_rejected{};
                    std::thread foreign{[&] {
                        try { renderer.Rasterizer()->SynchronizeMetalGasMaximum(); }
                        catch (const std::logic_error&) { foreign_rejected = true; }
                    }};
                    foreign.join();
                    Check(foreign_rejected && Metal::InputReferenceTestAccess::GasPending(rasterizer),
                          "Gas observation ignored owner or discarded pending feedback on error");
                    auto invalid = saved_native;
                    invalid[0] ^= 1;
                    bool invalid_rejected{};
                    // Invalid-state preflight must not discard the deferred value.
                    // LoadMetalState may finish already queued work before parsing.
                    gate.signaledValue = 1;
                    try { rasterizer.LoadMetalState(invalid); }
                    catch (const std::invalid_argument&) { invalid_rejected = true; }
                    Check(invalid_rejected && Metal::InputReferenceTestAccess::GasPending(rasterizer),
                          "Invalid native import discarded unpublished Gas feedback");
                    if (boundary == 0) {
                        const auto maximum = Metal::InputReferenceTestAccess::GasMaximum(rasterizer);
                        Metal::TransferRuntime transfer{device, scheduler};
                        transfer.FillBuffer(maximum.buffer, NSMakeRange(maximum.offset, 4), 0xff);
                        bool rejected{};
                        try { renderer.Rasterizer()->SynchronizeMetalGasMaximum(); }
                        catch (const std::runtime_error&) { rejected = true; }
                        Check(rejected && Metal::InputReferenceTestAccess::GasPending(rasterizer) &&
                              regs.reg_array[0x125] == 0xa5c30000,
                              "Failed Gas read published invalid bits or discarded pending feedback");
                        const auto restored = device.CreateBuffer(4, true);
                        const u32 value = 0x5000;
                        std::memcpy(restored.contents, &value, 4);
                        transfer.CopyBuffer(restored, 0, maximum.buffer, maximum.offset, 4);
                    }
                    const auto before = scheduler.CurrentTick();
                    if (boundary == 0) renderer.Rasterizer()->SynchronizeMetalGasMaximum();
                    else if (boundary == 1) rasterizer.UpdateMetalGasAccumulatorMaximum();
                    else if (boundary == 2) {
                        Metal::GasTransferState saved{};
                        Metal::UnpackGasArchive(rasterizer.SaveMetalState(), saved);
                        Check(saved.maximum == 0x5000 && !saved.feedback_pending,
                              "Save failed to publish the latest deferred maximum");
                    } else {
                        // A reset supersedes unobserved writes without publishing
                        // them; the following new draw must start a new maximum.
                        process(reset);
                        Check(!Metal::InputReferenceTestAccess::GasPending(rasterizer) &&
                              regs.reg_array[0x125] == 0xa5c30000,
                              "Ordered reset published a superseded deferred maximum");
                        reset_target();
                        std::memset(target.GetPtr(), 0, 32 * 32 * 4);
                        batch(0, 3, false);
                        renderer.Rasterizer()->SynchronizeMetalGasMaximum();
                    }
                    Check(regs.reg_array[0x125] == (boundary == 3 ? 0xa5c33c00u : 0xa5c35000u) &&
                          !Metal::InputReferenceTestAccess::GasPending(rasterizer),
                          "Gas CPU boundary lost maximum, upper bits or pending state");
                    if (boundary == 1)
                        Check(scheduler.CurrentTick() == before + 1 &&
                              (regs.reg_array[0xE5] & 0xffff) == 0x2800,
                              "GX maximum publication introduced a second submit or stale reciprocal");
                    const auto published_tick = scheduler.CurrentTick();
                    renderer.Rasterizer()->SynchronizeMetalGasMaximum();
                    Check(scheduler.CurrentTick() == published_tick,
                          "Repeated Gas observation submitted unrelated work");
                    std::stringstream stream;
                    { oarchive archive{stream}; archive << pica; }
                    const auto expected = regs.reg_array[0x125];
                    rasterizer.LoadMetalState(saved_native);
                    { iarchive archive{stream}; archive >> pica; }
                    Check(regs.reg_array[0x125] == expected &&
                          !Metal::InputReferenceTestAccess::GasPending(rasterizer),
                          "Published raw PICA register/native payload diverged after archive load");
                }
                std::cout << "Deferred Gas:128 actual PICA draws returned with GPU queue gated, no submit/readback; "
                             "CPU publication/GX/save/reset, owner/read errors, rejected import, upper bits and raw PICA archive exact\n";
                regs = saved_regs;
                rasterizer.LoadMetalState(saved_native);
                std::memcpy(memory.GetPhysicalPointer(vertex_address), raw_vertices.data(), sizeof(raw_vertices));
                rasterizer.ClearAll(false);
                pica.dirty_regs.SetAllDirty();
                Check(cases == 48, "Incomplete public PICA Gas accumulation matrix");
                std::cout << "Public PICA Gas accumulation -> 49152 physical VRAM pixels passed: 32 overlapping triangles, "
                             "4 depth functions, D16/D24/D24S8, batch/immediate/indexed/mixed, GPU feedback and PICA archives\n";
            }

            const auto save_pica = [&] {
                std::ostringstream stream(std::ios::binary);
                { oarchive archive{stream}; archive & pica; }
                return stream.str();
            };
            const auto load_pica = [&](const std::string& state) {
                std::istringstream stream{state,std::ios::binary};
                iarchive archive{stream}; archive & pica;
            };
            reset_target();
            immediate_vertex(0); immediate_vertex(1);
            check_target(false);
            rasterizer.ClearAll(true); // Exact pre-save cache operation used by Core::System.
            const auto native_before=rasterizer.SaveMetalState();
            Check(native_before.size()==Metal::StateArchiveBytes && native_before[16]==2,"Pre-save cache clear destroyed the incomplete native primitive");
            {
                std::vector<u32> gas_commands;
                Write(gas_commands, 0x123, 0);
                for (u32 i=0; i<16; ++i) Write(gas_commands, 0x124, 0x10203040u+i);
                process(gas_commands);
                const auto gas_state=rasterizer.SaveMetalState();
                Metal::GasTransferState gas;
                Check(Metal::UnpackGasArchive(gas_state,gas)==native_before && gas.cursor==16,
                      "Gas archive changed real pending GPU primitive/input banks");
                rasterizer.LoadMetalState(native_before);
                rasterizer.LoadMetalState(gas_state);
                Check(rasterizer.SaveMetalState()==gas_state,"Gas archive lost actual native GPU history");
                rasterizer.LoadMetalState(native_before);
            }
            const auto saved_pica=save_pica();
            for (unsigned version : {0U,4U}) {
                std::ostringstream empty_archive(std::ios::binary);
                { oarchive archive{empty_archive}; }
                std::istringstream source{empty_archive.str(),std::ios::binary};
                iarchive archive{source};
                bool rejected_version=false;
                try { boost::serialization::access::serialize(archive,pica,version); }
                catch (const std::runtime_error&) { rejected_version=true; }
                Check(rejected_version,"Metal PICA loader accepted a legacy or unsupported future archive version");
            }
            Check(rasterizer.SaveMetalState()==native_before,"Exporting PICA state changed GPU history");
            immediate_vertex(2); check_target(true);
            regs.rasterizer.viewport_size_x.Assign(0);
            rasterizer.ClearAll(false);
            std::memset(target.GetPtr(),0x11,32*32*4);
            load_pica(saved_pica);
            Check(rasterizer.SaveMetalState()==native_before,"Metal archive did not restore both GPU banks and primitive metadata byte-for-byte");
            immediate_vertex(2); check_target(true);
            // Reject malformed payloads before either history bank or cache is changed.
            const auto stable=rasterizer.SaveMetalState();
            for (unsigned offset : {0U,8U,12U,16U,20U,24U,28U,32U+256U}) {
                auto corrupt=stable;
                corrupt[offset]=255;
                bool rejected=false;
                try { rasterizer.LoadMetalState(corrupt); } catch (const std::invalid_argument&) { rejected=true; }
                Check(rejected,"Native state import accepted invalid signature/version/topology/count/flags");
                Check(rasterizer.SaveMetalState()==stable,"Rejected native state partially changed GPU history");
            }
            bool rejected_state=false;
            try { rasterizer.LoadMetalState(std::span{stable}.first(stable.size()-1)); }
            catch (const std::invalid_argument&) { rejected_state=true; }
            Check(rejected_state,"Native state import accepted a truncated payload");
            // A separate, newly constructed native rasterizer receives the same GPU state.
            rasterizer.ClearAll(false);
            {
                Metal::RasterizerMetal restored{memory,pica,custom,renderer,device,scheduler};
                renderer.rasterizer=&restored;
                pica.BindRasterizer(&restored);
                restored.LoadMetalState(native_before);
                Check(restored.SaveMetalState()==native_before,"Recreated renderer lost native opaque state");
                std::memset(target.GetPtr(),0x11,32*32*4);
                immediate_vertex(2);
                restored.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                for (unsigned byte=0; byte<32*32*4; ++byte)
                    Check(target.GetPtr()[byte]==(byte%4==2 ? 0 : 255),"Recreated native renderer could not finish a restored partial triangle");
                restored.ClearAll(false);
            }
            renderer.rasterizer=&rasterizer;
            pica.BindRasterizer(&rasterizer);
            std::cout << "Real PICA binary archive: pre-save ClearAll, incomplete GPU primitive/input banks, byte-exact load, malformed-payload rollback and recreated-rasterizer continuation passed\n";

            // Real register commands must overwrite GPU-modified uniforms even when the CPU value is unchanged.
            const auto saved_native=rasterizer.SaveMetalState();
            const auto saved_gs_uniforms=pica.gs_setup.uniforms,saved_vs_uniforms=pica.vs_setup.uniforms;
            const auto saved_gs_regs=regs.gs,saved_vs_regs=regs.vs;
            const auto read_word=[](const auto& bytes,unsigned offset) {
                u32 value{};for(unsigned byte=0;byte<4;++byte) value|=u32(bytes[offset+byte])<<(byte*8);return value;
            };
            const auto put_word=[](auto& bytes,unsigned offset,u32 value) {
                for(unsigned byte=0;byte<4;++byte) bytes[offset+byte]=u8(value>>(byte*8));
            };
            constexpr unsigned gs_uniform_offset=Metal::GeometryStateOffset+128;
            auto gpu_modified=saved_native;
            put_word(gpu_modified,28,1);put_word(gpu_modified,Metal::GeometryStateOffset,2);
            for(unsigned channel=0;channel<4;++channel) {
                pica.gs_setup.uniforms.f[5][channel]=Pica::f24::FromFloat32(2);
                put_word(gpu_modified,gs_uniform_offset+5*16+channel*4,std::bit_cast<u32>(77.f));
                put_word(gpu_modified,gs_uniform_offset+6*16+channel*4,std::bit_cast<u32>(88.f));
                put_word(gpu_modified,gs_uniform_offset+1536+channel*4,9);
                put_word(gpu_modified,gs_uniform_offset+1552+channel*4,33);
            }
            pica.gs_setup.uniforms.i[0]={1,2,3,4};pica.gs_setup.uniforms.b.fill(false);
            put_word(gpu_modified,gs_uniform_offset+1600,0x8000);
            rasterizer.LoadMetalState(gpu_modified);
            const auto float_uniform=[&](bool vertex_stage,u32 index,const std::array<float,4>& value,bool grouped) {
                std::vector<u32> words;
                const auto setup=vertex_stage ? PICA_REG_INDEX(vs.uniform_setup):PICA_REG_INDEX(gs.uniform_setup);
                const u32 data=vertex_stage ? PICA_REG_INDEX(vs.uniform_setup.set_value[0]):PICA_REG_INDEX(gs.uniform_setup.set_value[0]);
                Write(words,setup,(1u<<31)|index);
                if(grouped) {
                    words.insert(words.end(),{std::bit_cast<u32>(value[3]),data|(15u<<16)|(3u<<20)|(1u<<31),
                        std::bit_cast<u32>(value[2]),std::bit_cast<u32>(value[1]),std::bit_cast<u32>(value[0]),0});
                } else for(int c=3;c>=0;--c) Write(words,data,std::bit_cast<u32>(value[c]));
                process(words);
            };
            for(bool grouped:{false,true}) {
                rasterizer.LoadMetalState(gpu_modified);
                pica.gs_setup.uniforms_dirty=false;
                float_uniform(false,5,{2,2,2,2},grouped);
                auto state=rasterizer.SaveMetalState();
                for(unsigned c=0;c<4;++c) {
                    Check(read_word(state,gs_uniform_offset+5*16+c*4)==std::bit_cast<u32>(2.f),"Repeated same-value GS float command did not overwrite native GPU uniform");
                    Check(read_word(state,gs_uniform_offset+6*16+c*4)==std::bit_cast<u32>(88.f),"GS float command destroyed an unrelated native uniform");
                }
                Check(!pica.gs_setup.uniforms_dirty && read_word(state,gs_uniform_offset+1600)==0x8000,
                      "GS write must be forwarded independently of CPU dirty flags and preserve unrelated GPU b15");
            }
            std::vector<u32> uniform_commands;
            Write(uniform_commands,PICA_REG_INDEX(gs.bool_uniforms),0);
            Write(uniform_commands,PICA_REG_INDEX(gs.int_uniforms[0]),0x04030201);process(uniform_commands);
            auto gs_state=rasterizer.SaveMetalState();
            Check(read_word(gs_state,gs_uniform_offset+1600)==0,"Repeated GS boolean write did not clear GPU-generated b15");
            for(unsigned c=0;c<4;++c) {
                Check(read_word(gs_state,gs_uniform_offset+1536+c*4)==c+1,"GS integer command did not reach private uniform bank");
                Check(read_word(gs_state,gs_uniform_offset+1552+c*4)==33,"GS integer command overwrote an unrelated private uniform");
            }
            // Shared VS writes follow the existing GS-exclusive/use_gs conditions.
            for(bool grouped:{false,true}) {
                auto poisoned=rasterizer.SaveMetalState();
                for(unsigned c=0;c<4;++c) put_word(poisoned,gs_uniform_offset+6*16+c*4,std::bit_cast<u32>(88.f));
                rasterizer.LoadMetalState(poisoned);
                float_uniform(true,6,{3,4,5,6},grouped);
                gs_state=rasterizer.SaveMetalState();
                for(unsigned c=0;c<4;++c) Check(read_word(gs_state,gs_uniform_offset+6*16+c*4)==std::bit_cast<u32>(float(c+3)),"Shared VS single/batch float command was not mirrored to native GS uniforms");
            }
            const auto original_exclusive=regs.pipeline.gs_unit_exclusive_configuration.Value();
            const auto original_use_gs=regs.pipeline.use_gs.Value();
            for(bool exclusive:{false,true}) {
                regs.pipeline.gs_unit_exclusive_configuration.Assign(exclusive);
                regs.pipeline.use_gs.Assign(exclusive ? Pica::PipelineRegs::UseGS::No:Pica::PipelineRegs::UseGS::Yes);
                float_uniform(true,6,{9,9,9,9},false);
                const auto isolated=rasterizer.SaveMetalState();
                for(unsigned c=0;c<4;++c) Check(read_word(isolated,gs_uniform_offset+6*16+c*4)==std::bit_cast<u32>(float(c+3)),"Exclusive/enabled-GS configuration incorrectly mirrored a VS uniform write");
            }
            regs.pipeline.gs_unit_exclusive_configuration.Assign(original_exclusive);regs.pipeline.use_gs.Assign(original_use_gs);
            const auto gs_archive=save_pica();
            float_uniform(false,5,{4,4,4,4},false);load_pica(gs_archive);
            Check(rasterizer.SaveMetalState()==gs_state,"Real PICA v2 binary archive lost private GS uniform banks");
            // Native v1 blobs have no GS banks and migrate to canonical empty native GS state.
            std::vector<u8> old_native(saved_native.begin(),saved_native.begin()+Metal::LegacyStateBytes);
            put_word(old_native,8,1);put_word(old_native,28,0);rasterizer.LoadMetalState(old_native);
            const auto migrated=rasterizer.SaveMetalState();
            Check(migrated.size()==Metal::StateArchiveBytes && read_word(migrated,8)==2 && read_word(migrated,28)==0 &&
                  read_word(migrated,Metal::GeometryStateOffset)==0,"Native Metal v1 blob did not migrate to empty v2 GS state");
            for(unsigned offset : {unsigned(Metal::LegacyStateBytes),unsigned(Metal::LegacyStateBytes+4),
                unsigned(Metal::LegacyStateBytes+16+1816),unsigned(Metal::GeometryStateOffset),
                unsigned(Metal::GeometryStateOffset+32),unsigned(Metal::GeometryStateOffset+112),
                gs_uniform_offset+1602,gs_uniform_offset+1604}) {
                auto corrupt=gs_state;corrupt[offset]=255;bool rejected=false;
                try { rasterizer.LoadMetalState(corrupt); } catch(const std::invalid_argument&) { rejected=true; }
                Check(rejected && rasterizer.SaveMetalState()==migrated,"Malformed native GS/archive metadata partially changed existing GPU state");
            }
            pica.gs_setup.uniforms=saved_gs_uniforms;pica.vs_setup.uniforms=saved_vs_uniforms;
            regs.gs=saved_gs_regs;regs.vs=saved_vs_regs;rasterizer.LoadMetalState(saved_native);
            std::cout<<"Real PICA GS uniform commands: same-value single/batch floats despite clean CPU dirty flags, isolated bool/int writes, shared VS writes, v2 binary archive and native v1 blob migration passed\n";

            {
                const auto original=save_pica();using Op=nihstro::OpCode::Id;
                const auto opcode=[](Op op) { return u32(op)<<26; };
                float_uniform(true,0,{.25f,.25f,.25f,.25f},false);
                pica.vs_setup.UpdateProgramCode(2,opcode(Op::DSTI)|(2u<<21)|(2u<<14)|(32u<<7));
                pica.vs_setup.UpdateProgramCode(3,opcode(Op::END));
                reset_target();batch(0,3,false);rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                for(unsigned byte=0;byte<32*32*4;++byte) {
                    const auto expected=std::array<int,4>{255,64,0,255}[byte%4];
                    Check(std::abs(int(target.GetPtr()[byte])-expected)<=1,"Native No-GS DSTI opcode did not reach correct physical VRAM colors");
                }
                pica.vs_setup.UpdateProgramCode(2,opcode(Op::CMP)|(6u<<24)|(7u<<21)|(2u<<12)|(2u<<7));
                pica.vs_setup.UpdateProgramCode(3,opcode(Op::IFC)|(5u<<10)|1u|(1u<<22)|(1u<<25)|(1u<<24));
                pica.vs_setup.UpdateProgramCode(4,Metal::Test::Arithmetic(Op::MOV,2,2));
                pica.vs_setup.UpdateProgramCode(5,Metal::Test::Arithmetic(Op::MOV,2,32));pica.vs_setup.UpdateProgramCode(6,opcode(Op::END));
                reset_target();immediate_vertex(0);immediate_vertex(1);const auto pending=save_pica();
                immediate_vertex(2);check_target(true);
                std::memset(target.GetPtr(),0x11,32*32*4);load_pica(pending);immediate_vertex(2);check_target(true);
                reset_target();batch(0,3,false);check_target(true);
                pica.vs_setup.UpdateProgramCode(2,Metal::Test::Arithmetic(Op::ADD,16,32,16));
                pica.vs_setup.UpdateProgramCode(3,Metal::Test::Arithmetic(Op::MOV,2,16));pica.vs_setup.UpdateProgramCode(4,opcode(Op::END));
                for(unsigned bytes:{1u,2u}) {
                    const std::array<u8,6> ids=bytes==1 ? std::array<u8,6>{0,1,0,0,0,0}:std::array<u8,6>{0,0,1,0,0,0};
                    std::memcpy(memory.GetPhysicalPointer(index_address),ids.data(),ids.size());
                    regs.pipeline.index_array.offset.Assign(index_address-vertex_address);
                    regs.pipeline.index_array.format.Assign(bytes==2 ? decltype(regs.pipeline.index_array)::SHORT:decltype(regs.pipeline.index_array)::BYTE);
                    reset_target();batch(0,3,true);
                    const auto no_gs_state=rasterizer.SaveMetalState();
                    Check(read_word(no_gs_state,Metal::LegacyStateBytes+16+256)==std::bit_cast<u32>(.5f),
                          "Native No-GS cache hit re-executed temporary arithmetic instead of retaining GPU output");
                }
                // Immediate invocations reset VS temporaries while preserving the
                // input banks and partially assembled primitive across commands/archives.
                // Earlier command-queue tests can leave a partial uniform packet;
                // seed this fixture's complete scalar uniform explicitly.
                for(unsigned channel=0;channel<4;++channel)
                    pica.vs_setup.uniforms.f[0][channel]=Pica::f24::FromFloat32(.25f);
                reset_target();
                for(unsigned vertex=0;vertex<3;++vertex) {
                    immediate_vertex(vertex);
                    const auto state=rasterizer.SaveMetalState();
                    for(unsigned channel=0;channel<4;++channel) {
                        const auto actual=read_word(state,Metal::LegacyStateBytes+16+256+channel*4);
                        Check(actual==std::bit_cast<u32>(.25f),
                              "Native immediate machine retained a prior invocation's temporary");
                    }
                    if(vertex<2) check_target(false);
                }
                rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                const std::vector<u8> immediate_pixels(target.GetPtr(),target.GetPtr()+32*32*4);
                for(const auto byte:immediate_pixels)
                    Check(std::abs(int(byte)-64)<=1,"Native immediate machine lost its converted output/physical VRAM color");
                reset_target();immediate_vertex(0);immediate_vertex(1);const auto partial=save_pica();
                immediate_vertex(2);rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                const auto completed=rasterizer.SaveMetalState();
                load_pica(partial);std::memset(target.GetPtr(),0x11,32*32*4);
                immediate_vertex(2);rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                Check(rasterizer.SaveMetalState()==completed &&
                      !std::memcmp(target.GetPtr(),immediate_pixels.data(),immediate_pixels.size()),
                      "Native immediate machine archive continuation changed full native state or VRAM");
                pica.vs_setup.UpdateProgramCode(0,opcode(Op::SETEMIT));
                reset_target();bool rejected=false;
                try { batch(0,1,false); } catch(const Metal::ShaderMachineError& error) { rejected=error.Code()==2; }
                Check(rejected,"Actual native No-GS mode accepted a GS emission instruction");
                load_pica(original);
                std::cout<<"Actual native No-GS machine routing: DSTI physical VRAM colors, CMP6/7 IFC, immediate/batch/PICA-archive continuation, 8/16-bit temporary cache hits and GS-emission rejection passed\n";
            }

            {
                const auto original = save_pica();
                using Op = nihstro::OpCode::Id;
                const auto mov = [](unsigned dest, unsigned source) {
                    return Metal::Test::Arithmetic(Op::MOV, dest, source);
                };
                // Isolate program mutations from the partially filled uniform
                // command queue restored by the preceding archive tests.
                for (unsigned c = 0; c < 4; ++c) {
                    pica.vs_setup.uniforms.f[0][c] = Pica::f24::FromFloat32(
                        std::array<float, 4>{.25f, .5f, .75f, 1.f}[c]);
                    pica.vs_setup.uniforms.f[1][c] = Pica::f24::FromFloat32(
                        std::array<float, 4>{.75f, .25f, .5f, 1.f}[c]);
                }
                const std::array<u32, 6> low{mov(0, 0), mov(1, 1), mov(16, 32),
                    mov(2, 16) | 127u, mov(3, 33), u32(Op::END) << 26};
                auto high = low;
                high[2] = mov(16, 33);
                pica.vs_setup.UpdateProgramCodeRange(0, low.data(), low.size());
                pica.vs_setup.UpdateProgramCodeRange(4090, high.data(), high.size());
                pica.vs_setup.UpdateSwizzleData(127, Metal::Test::IdentitySwizzle);
                regs.vs.main_offset.Assign(0);
                regs.vs.output_mask.Assign(7);
                unsigned images{};
                const auto verify = [&](std::array<int, 4> abgr, bool immediate = false) {
                    reset_target();
                    if (immediate) for (unsigned i = 0; i < 3; ++i) immediate_vertex(i);
                    else batch(0, 3, false);
                    rasterizer.FlushRegion(Memory::VRAM_PADDR, 4096);
                    for (unsigned byte = 0; byte < 4096; ++byte) {
                        if (std::abs(int(target.GetPtr()[byte]) - abgr[byte % 4]) > 1)
                            std::cerr << "Program mutation image " << images << ", byte " << byte
                                      << ": actual " << int(target.GetPtr()[byte])
                                      << ", expected " << abgr[byte % 4] << '\n';
                        Check(std::abs(int(target.GetPtr()[byte]) - abgr[byte % 4]) <= 1,
                              "Native program reuse lost code/swizzle/entry/mask/archive mutation");
                    }
                    ++images;
                };
                const std::array<int, 4> first{255, 191, 128, 64}, second{255, 128, 64, 191};
                verify(first); verify(first); // Warm repeated program.
                const auto snapshot = save_pica();
                pica.vs_setup.UpdateSwizzleData(127, Metal::Test::IdentitySwizzle & ~(255u << 5));
                verify({64, 64, 64, 64});
                pica.vs_setup.UpdateSwizzleData(127, Metal::Test::IdentitySwizzle);
                verify(first, true);
                pica.vs_setup.UpdateProgramCode(2, mov(16, 33));
                verify(second);
                pica.vs_setup.UpdateProgramCodeRange(0, low.data(), low.size());
                verify(first);
                regs.vs.main_offset.Assign(4090); // END is the last word in the bank.
                verify(second); verify(second, true);
                regs.vs.main_offset.Assign(0);
                regs.vs.output_mask.Assign(11); // Pack o3 instead of o2 as color.
                verify(second);
                regs.vs.output_mask.Assign(7);
                verify(first);
                // A complete bank replacement must also invalidate the snapshot.
                auto replacement = pica.vs_setup.GetProgramCode();
                replacement[2] = mov(16, 33);
                pica.vs_setup.UpdateProgramCode(replacement, 4096);
                verify(second);
                load_pica(snapshot); verify(first);
                regs.vs.main_offset.Assign(4090);
                pica.vs_setup.uniforms.b[0] = true;
                // The final bank word changes valid control flow: jump back to
                // the low program, which replaces the high program's color.
                pica.vs_setup.UpdateProgramCode(4095, u32(Op::JMPU) << 26);
                verify(first);
                pica.vs_setup.UpdateProgramCode(4095, u32(Op::END) << 26);
                verify(second);
                load_pica(snapshot); verify(first, true);
                load_pica(original);
                std::cout << "Native program snapshot mutations: " << images
                          << " full VRAM images, last code word, swizzle 127, single/range/full writes, entry, output mask, immediate/batch and PICA archive passed\n";
            }

            {
                const auto no_gs_archive=save_pica();
                const auto original_rasterizer=regs.rasterizer;
                using Op=nihstro::OpCode::Id;
                const auto op=[](Op id) { return u32(id)<<26; };
                const auto setup_gs=[&](Pica::PipelineRegs::GSMode mode) {
                    regs.pipeline.use_gs.Assign(Pica::PipelineRegs::UseGS::Yes);
                    regs.pipeline.variable_primitive.Assign(mode==Pica::PipelineRegs::GSMode::VariablePrimitive);
                    regs.pipeline.vs_outmap_total_minus_1_a.Assign(2);regs.pipeline.vs_outmap_total_minus_1_b.Assign(2);
                    regs.pipeline.gs_config.mode.Assign(mode);regs.pipeline.gs_config.stride_minus_1.Assign(2);
                    regs.pipeline.gs_config.fixed_vertex_num_minus_1.Assign(2);regs.pipeline.gs_config.start_index.Assign(0);
                    regs.pipeline.variable_vertex_main_num_minus_1=2;
                    regs.gs.shader_mode.Assign(Pica::ShaderRegs::ShaderMode::GS);
                    regs.gs.input_to_uniform.Assign(mode==Pica::PipelineRegs::GSMode::Point ? 0:1);
                    regs.gs.max_input_attribute_index.Assign(8);regs.gs.input_attribute_to_register_map_low=0x76543210;
                    regs.gs.input_attribute_to_register_map_high=0xfedcba98;regs.gs.main_offset.Assign(0);regs.gs.output_mask.Assign(3);
                    pica.gs_setup.UpdateSwizzleData(0,Metal::Test::IdentitySwizzle);
                    for(unsigned vertex=0;vertex<3;++vertex) {
                        const unsigned input=vertex*3+(mode==Pica::PipelineRegs::GSMode::VariablePrimitive ? 1:0);
                        const unsigned source=mode==Pica::PipelineRegs::GSMode::Point ? input:32+input;
                        pica.gs_setup.UpdateProgramCode(vertex*4,Metal::Test::Arithmetic(Op::MOV,0,source));
                        pica.gs_setup.UpdateProgramCode(vertex*4+1,Metal::Test::Arithmetic(Op::MOV,1,source+2));
                        pica.gs_setup.UpdateProgramCode(vertex*4+2,op(Op::SETEMIT)|(vertex<<24)|(vertex==2 ? 1u<<23:0));
                        pica.gs_setup.UpdateProgramCode(vertex*4+3,op(Op::EMIT));
                    }
                    pica.gs_setup.UpdateProgramCode(12,op(Op::END));
                    regs.rasterizer.vs_output_total.Assign(2);
                    regs.rasterizer.vs_output_attributes[0].raw=0x03020100;
                    regs.rasterizer.vs_output_attributes[1].raw=0x0b0a0908;
                    std::vector<u32> topology;Write(topology,PICA_REG_INDEX(pipeline.triangle_topology),3u<<8);process(topology);
                };
                const auto indexed_gs=[&](std::span<const u16> ids,unsigned bytes) {
                    std::vector<u8> storage(ids.size()*bytes);
                    for(unsigned i=0;i<ids.size();++i) { storage[i*bytes]=u8(ids[i]);if(bytes==2) storage[i*bytes+1]=u8(ids[i]>>8); }
                    std::memcpy(memory.GetPhysicalPointer(index_address),storage.data(),storage.size());
                    regs.pipeline.index_array.offset.Assign(index_address-vertex_address);
                    regs.pipeline.index_array.format.Assign(bytes==2 ? decltype(regs.pipeline.index_array)::SHORT:decltype(regs.pipeline.index_array)::BYTE);
                    batch(0,u32(ids.size()),true);
                };
                unsigned gs_cases{};
                for(auto mode:{Pica::PipelineRegs::GSMode::Point,Pica::PipelineRegs::GSMode::FixedPrimitive,
                               Pica::PipelineRegs::GSMode::VariablePrimitive}) {
                    setup_gs(mode);
                    reset_target();
                    if(mode!=Pica::PipelineRegs::GSMode::VariablePrimitive) {
                        const auto scratch_before=Metal::InputReferenceTestAccess::ScratchBytes(rasterizer);
                        const auto tick=scheduler.CurrentTick();batch(0,3,false);
                        Check(Metal::InputReferenceTestAccess::PendingGS(rasterizer)>0 && scheduler.CurrentTick()==tick,
                            "Actual GS list draw still completed/read its count on the CPU");
                        Check(Metal::InputReferenceTestAccess::ScratchBytes(rasterizer)==scratch_before &&
                              Metal::InputReferenceTestAccess::CompactGS(rasterizer),
                            "Actual GS did not retain logical1M limits with private proved physical storage");
                        check_target(true);++gs_cases;
                        if(mode==Pica::PipelineRegs::GSMode::Point) {
                            const auto before_multi=save_pica();
                            pica.vs_setup.UpdateSwizzleData(0,Metal::Test::IdentitySwizzle);
                            for(unsigned a=0;a<3;++a)pica.vs_setup.UpdateProgramCode(a,Metal::Test::Arithmetic(Op::MOV,a,a));
                            pica.vs_setup.UpdateProgramCode(3,op(Op::END));regs.vs.main_offset.Assign(0);regs.vs.output_mask.Assign(7);
                            regs.texturing.main_config.texture0_enable.Assign(0);
                            regs.texturing.main_config.texture1_enable.Assign(0);
                            regs.texturing.main_config.texture2_enable.Assign(0);
                            constexpr unsigned vertices=192,bytes=vertices*sizeof(raw_vertices[0]);
                            std::vector<u8> saved_input(bytes);std::memcpy(saved_input.data(),memory.GetPhysicalPointer(vertex_address),bytes);
                            for(unsigned n=0;n<vertices;++n)std::memcpy(memory.GetPhysicalPointer(vertex_address)+n*sizeof(raw_vertices[0]),raw_vertices[n%3].data(),sizeof(raw_vertices[0]));
                            reset_target();setup_gs(mode);
                            Metal::InputReferenceTestAccess::ResetVertexBatchMarker(rasterizer);
                            const auto encoded=scheduler.EncoderCount(),tick=scheduler.CurrentTick();
                            batch(0,vertices,false);
                            Check(scheduler.CurrentTick()==tick && scheduler.EncoderCount()-encoded<64 &&
                                  Metal::InputReferenceTestAccess::CompactGS(rasterizer) &&
                                  Metal::InputReferenceTestAccess::VertexBatchUsed(rasterizer),
                                  "Actual public PICA GS batch did not group its64 GPU draws without count completion");
                            check_target(true);
                            // A later vertex reads pixels written by the preceding GS draw.
                            // Bulk processing must match two separate public command lists.
                            const auto alias_start=save_pica();std::vector<u8> alias_pixels;
                            for(bool split:{true,false}) {
                                load_pica(alias_start);reset_target();setup_gs(mode);
                                regs.pipeline.vertex_attributes.base_address.Assign(Memory::VRAM_PADDR/16);
                                for(unsigned n=0;n<6;++n)std::memcpy(target.GetPtr()+n*sizeof(raw_vertices[0]),raw_vertices[n%3].data(),sizeof(raw_vertices[0]));
                                Metal::InputReferenceTestAccess::ResetVertexBatchMarker(rasterizer);
                                if(split){batch(0,3,false);batch(3,3,false);}else batch(0,6,false);
                                Check(!Metal::InputReferenceTestAccess::VertexBatchUsed(rasterizer),
                                    "Framebuffer-aliased PICA loaders entered an early VS batch");
                                rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                                if(split)alias_pixels.assign(target.GetPtr(),target.GetPtr()+32*32*4);
                                else Check(std::equal(alias_pixels.begin(),alias_pixels.end(),target.GetPtr()),
                                    "Batched GS read aliased framebuffer bytes before publishing its earlier draw");
                            }
                            std::memcpy(memory.GetPhysicalPointer(vertex_address),saved_input.data(),bytes);load_pica(before_multi);
                            std::cout<<"Actual PICA compact GS:64 deferred GPU draws, one bounded batch,4096 framebuffer-alias bytes equal split command lists, no count/readback wait\n";
                            Metal::InputReferenceTestAccess::DrainGS(rasterizer);
                            for(bool reset_unit:{false,true}) {
                                reset_target();
                                if(reset_unit) Metal::InputReferenceTestAccess::ResetUnit(rasterizer);
                                std::vector<u32> clear_boolean;Write(clear_boolean,PICA_REG_INDEX(gs.bool_uniforms),0);process(clear_boolean);
                                const auto before_unit=Metal::InputReferenceTestAccess::Unit(rasterizer);
                                const auto good=pica.gs_setup.GetProgramCode()[0];pica.gs_setup.UpdateProgramCode(0,0x10u<<26);
                                batch(0,3,false);scheduler.Finish();
                                Check(Metal::InputReferenceTestAccess::PendingGS(rasterizer)>0,"Deferred GS failure was not retained for ordered retirement");
                                const auto ready_group=Metal::InputReferenceTestAccess::CapturedGroup(rasterizer);
                                bool rejected=false;
                                try{Metal::InputReferenceTestAccess::DrainGS(rasterizer);}catch(const Metal::ShaderMachineError& error){rejected=error.Code()==2;}
                                Check(rejected && Metal::InputReferenceTestAccess::PendingGS(rasterizer)==0,
                                    "Deferred GS did not report/retire its original fatal error");
                                Check(Metal::InputReferenceTestAccess::Unit(rasterizer)==before_unit &&
                                    Metal::InputReferenceTestAccess::Group(rasterizer)==ready_group && !pica.gs_setup.uniforms.b[15],
                                    "Deferred fatal GS changed full1824/2128-byte state, ready group, reset marker or CPU b15 transport");
                                pica.gs_setup.UpdateProgramCode(0,good);pica.gs_setup.DoProgramCodeFixup();
                                const Metal::ShaderMachineProgramView corrected{pica.gs_setup.GetProgramCode(),pica.gs_setup.GetSwizzleData(),
                                    regs.gs.main_offset.Value(),regs.gs.output_mask.Value(),true};
                                Metal::InputReferenceTestAccess::ResumeReady(rasterizer,corrected);check_target(true);
                            }
                            std::cout<<"Actual deferred GS: no pre-draw count completion; proved compact storage/logical1M; two fatal errors preserve full1824/2128-byte state, reset/ready/b15 and corrected ready-group VRAM retry\n";
                        }
                    }
                    for(unsigned bytes:{1u,2u}) {
                        reset_target();
                        const std::array<u16,4> variable{3,0,1,2};const std::array<u16,3> fixed{0,1,2};
                        indexed_gs(mode==Pica::PipelineRegs::GSMode::VariablePrimitive ? std::span<const u16>{variable}:std::span<const u16>{fixed},bytes);
                        check_target(true);++gs_cases;
                    }
                    reset_target();
                    if(mode==Pica::PipelineRegs::GSMode::VariablePrimitive) {
                        const std::array<u16,3> partial{3,0,1};indexed_gs(partial,2);
                    } else batch(0,2,false);
                    check_target(false);
                    const auto partial_native=rasterizer.SaveMetalState();const auto partial_archive=save_pica();
                    const auto finish=[&] {
                        if(mode==Pica::PipelineRegs::GSMode::VariablePrimitive) {
                            const std::array<u16,1> last{2};indexed_gs(last,2);
                        } else immediate_vertex(2);
                    };
                    finish();check_target(true);
                    std::memset(target.GetPtr(),0x11,32*32*4);load_pica(partial_archive);
                    Check(rasterizer.SaveMetalState()==partial_native,"Actual GS archive lost its incomplete input group");
                    finish();check_target(true);++gs_cases;
                    rasterizer.ClearAll(false);
                    {
                        Metal::RasterizerMetal recreated{memory,pica,custom,renderer,device,scheduler};
                        renderer.rasterizer=&recreated;pica.BindRasterizer(&recreated);
                        recreated.LoadMetalState(partial_native);std::memset(target.GetPtr(),0x11,32*32*4);
                        finish();recreated.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                        for(unsigned byte=0;byte<32*32*4;++byte)
                            Check(target.GetPtr()[byte]==(byte%4==2 ? 0:255),"Recreated native rasterizer failed actual GS archive-to-VRAM continuation");
                        recreated.ClearAll(false);
                    }
                    renderer.rasterizer=&rasterizer;pica.BindRasterizer(&rasterizer);
                    rasterizer.LoadMetalState(partial_native);finish();rasterizer.ClearAll(false);++gs_cases;
                }
                setup_gs(Pica::PipelineRegs::GSMode::VariablePrimitive);reset_target();
                const std::array<u16,6> trailing{5,0,1,2,0,1};indexed_gs(trailing,2);check_target(true);
                const auto trailing_state=rasterizer.SaveMetalState();
                for(unsigned c=0;c<4;++c) {
                    Check(read_word(trailing_state,gs_uniform_offset+10*16+c*4)==std::bit_cast<u32>(positions[0][c]) &&
                          read_word(trailing_state,gs_uniform_offset+11*16+c*4)==std::bit_cast<u32>(positions[1][c]),
                          "Actual variable GS did not copy only the first attribute of trailing vertices");
                }
                setup_gs(Pica::PipelineRegs::GSMode::Point);
                const auto shader_target=[&] { reset_target();std::vector<u32> words;Write(words,PICA_REG_INDEX(pipeline.triangle_topology),3u<<8);process(words); };
                const auto original_flip=regs.framebuffer.framebuffer.flip.Value();
                unsigned winding_cases{};
                for(bool flipped:{false,true}) for(bool winding:{false,true}) for(unsigned cull=0;cull<4;++cull) {
                    regs.framebuffer.framebuffer.flip.Assign(!flipped);regs.rasterizer.cull_mode.Assign(Pica::RasterizerRegs::CullMode(cull));
                    pica.gs_setup.UpdateProgramCode(10,op(Op::SETEMIT)|(2u<<24)|(1u<<23)|(winding ? 1u<<22:0));
                    shader_target();batch(0,3,false);
                    check_target(cull==0 || cull==3 || (cull==1 ? winding:!winding));++winding_cases;
                }
                regs.framebuffer.framebuffer.flip.Assign(original_flip);regs.rasterizer.cull_mode.Assign(Pica::RasterizerRegs::CullMode::KeepAll);
                pica.gs_setup.UpdateProgramCode(10,op(Op::SETEMIT)|(2u<<24)|(1u<<23));
                for(unsigned topology:{0u,1u,2u}) {
                    reset_target();std::vector<u32> words;Write(words,PICA_REG_INDEX(pipeline.triangle_topology),topology<<8);process(words);
                    batch(0,3,false);check_target(true);
                }
                // Old buffered No-GS vertices and a GS emission share the same Shader topology stream.
                shader_target();regs.pipeline.use_gs.Assign(Pica::PipelineRegs::UseGS::No);regs.rasterizer=original_rasterizer;
                immediate_vertex(0);immediate_vertex(1);check_target(false);
                regs.pipeline.use_gs.Assign(Pica::PipelineRegs::UseGS::Yes);regs.rasterizer.vs_output_total.Assign(2);
                regs.rasterizer.vs_output_attributes[1].raw=0x0b0a0908;
                pica.gs_setup.UpdateProgramCode(10,op(Op::SETEMIT)|(2u<<24)|(1u<<23)|(1u<<22));
                batch(0,3,false);check_target(false); // First mixed triangle is degenerate; two GS vertices remain.
                const auto mixed=rasterizer.SaveMetalState();Check(mixed[16]==2,"GS emission discarded old Shader-topology pending vertices");
                regs.pipeline.use_gs.Assign(Pica::PipelineRegs::UseGS::No);regs.rasterizer=original_rasterizer;
                immediate_vertex(0);check_target(true);
                // A valid GS output larger than the initial allocation retries without committing a partial shader unit.
                setup_gs(Pica::PipelineRegs::GSMode::Point);shader_target();
                const auto emit_code=pica.gs_setup.GetProgramCode();
                for(unsigned pc=0;pc<12;++pc) pica.gs_setup.UpdateProgramCode(pc+1,emit_code[pc]);
                pica.gs_setup.UpdateProgramCode(0,op(Op::LOOP)|(12u<<10));pica.gs_setup.UpdateProgramCode(13,op(Op::END));
                std::vector<u32> loop_count;Write(loop_count,PICA_REG_INDEX(gs.int_uniforms[0]),64);process(loop_count);
                batch(0,3,false);check_target(true);
                std::cout<<winding_cases<<" actual GS Shader-topology winding/culling/flip cases, List/Strip/Fan, mixed No-GS/GS pending vertices, variable trailing inputs and 195-vertex adaptive emission passed\n";
                setup_gs(Pica::PipelineRegs::GSMode::Point);
                const auto normal_emit=pica.gs_setup.GetProgramCode();
                pica.gs_setup.UpdateProgramCode(0,op(Op::IFU)|(15u<<22)|(6u<<10)|12u);
                pica.gs_setup.UpdateProgramCode(1,Metal::Test::Arithmetic(Op::MOV,0,6));
                pica.gs_setup.UpdateProgramCode(2,Metal::Test::Arithmetic(Op::MOV,1,8));
                pica.gs_setup.UpdateProgramCode(3,op(Op::SETEMIT)|(2u<<24)|(1u<<23));
                pica.gs_setup.UpdateProgramCode(4,op(Op::EMIT));pica.gs_setup.UpdateProgramCode(5,op(Op::NOP));
                for(unsigned pc=0;pc<12;++pc) pica.gs_setup.UpdateProgramCode(6+pc,normal_emit[pc]);
                pica.gs_setup.UpdateProgramCode(18,op(Op::END));
                std::vector<u32> first_invocation;Write(first_invocation,PICA_REG_INDEX(gs.bool_uniforms),0);process(first_invocation);
                shader_target();batch(0,3,false);check_target(true);
                const auto retained_slots_archive=save_pica();
                auto changed_positions=raw_vertices;
                changed_positions[0][0]=changed_positions[0][1]=10;
                changed_positions[1][0]=11;changed_positions[1][1]=10;
                std::memcpy(memory.GetPhysicalPointer(vertex_address),changed_positions.data(),sizeof(changed_positions));
                shader_target();batch(0,3,false);check_target(true); // b15 selects reuse; first two emit slots remain from the previous invocation.
                load_pica(retained_slots_archive);shader_target();batch(0,3,false);check_target(true);
                std::memcpy(memory.GetPhysicalPointer(vertex_address),raw_vertices.data(),sizeof(raw_vertices));
                std::cout<<"Actual GS GPU b15 branch, retained emission slots/two banks across cache clear and binary archive passed to VRAM\n";
                // Temporary-state accumulation makes cache hits/replacement observable in the private GS bank.
                setup_gs(Pica::PipelineRegs::GSMode::FixedPrimitive);
                regs.vs.output_mask.Assign(1);regs.pipeline.vs_outmap_total_minus_1_a.Assign(0);regs.pipeline.vs_outmap_total_minus_1_b.Assign(0);
                regs.pipeline.gs_config.stride_minus_1.Assign(0);regs.pipeline.gs_config.start_index.Assign(90);
                pica.vs_setup.UpdateProgramCode(0,Metal::Test::Arithmetic(Op::ADD,16,16,2));
                pica.vs_setup.UpdateProgramCode(1,Metal::Test::Arithmetic(Op::MOV,0,16));pica.vs_setup.UpdateProgramCode(2,op(Op::END));
                pica.gs_setup.UpdateProgramCode(0,op(Op::END));
                auto cache_vertices=raw_vertices;
                for(auto& vertex:cache_vertices) { vertex[8]=.125f;vertex[9]=vertex[10]=0;vertex[11]=1; }
                std::memcpy(memory.GetPhysicalPointer(vertex_address),cache_vertices.data(),sizeof(cache_vertices));
                const std::array<u16,3> hits{0,1,0};indexed_gs(hits,1);
                auto cache_state=rasterizer.SaveMetalState();
                for(unsigned i=0;i<3;++i) Check(read_word(cache_state,gs_uniform_offset+(90+i)*16)==std::bit_cast<u32>(i==1 ? .25f:.125f),
                                              "Actual GS VS cache hit re-executed temporary-state arithmetic");
                regs.pipeline.gs_config.fixed_vertex_num_minus_1.Assign(0);
                std::vector<std::array<float,12>> many(65,cache_vertices[0]);
                std::memcpy(memory.GetPhysicalPointer(vertex_address),many.data(),many.size()*sizeof(many[0]));
                std::vector<u16> eviction;for(unsigned i=0;i<65;++i) eviction.push_back(u16(i));eviction.push_back(0);eviction.push_back(1);
                for(unsigned bytes:{1u,2u}) {
                    indexed_gs(eviction,bytes);cache_state=rasterizer.SaveMetalState();
                    Check(read_word(cache_state,gs_uniform_offset+90*16)==std::bit_cast<u32>(8.375f),
                          "Actual GS 64-entry circular VS cache replacement/order was incorrect");
                }
                std::memcpy(memory.GetPhysicalPointer(vertex_address),raw_vertices.data(),sizeof(raw_vertices));
                // A preceding GS draw overwrites the bytes fetched by the next
                // VS. Delaying both draws until after all uploads would read the
                // original blue/green colors and visibly violate VRAM feedback.
                setup_gs(Pica::PipelineRegs::GSMode::Point);shader_target();
                regs.pipeline.vertex_attributes={};
                auto& alias_attributes=regs.pipeline.vertex_attributes;
                alias_attributes.base_address.Assign(Memory::VRAM_PADDR/16);
                alias_attributes.max_attribute_index.Assign(0);
                alias_attributes.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                alias_attributes.size0.Assign(3);
                auto& alias_loader=alias_attributes.attribute_loaders[0];
                alias_loader.comp0.Assign(0);alias_loader.component_count.Assign(1);alias_loader.byte_count.Assign(16);
                regs.vs.max_input_attribute_index.Assign(0);regs.pipeline.max_input_attrib_index.Assign(0);
                regs.vs.input_attribute_to_register_map_low=0;regs.vs.input_attribute_to_register_map_high=0;
                regs.vs.output_mask.Assign(1);regs.vs.main_offset.Assign(0);
                regs.pipeline.vs_outmap_total_minus_1_a.Assign(0);regs.pipeline.vs_outmap_total_minus_1_b.Assign(0);
                regs.gs.max_input_attribute_index.Assign(0);regs.gs.input_attribute_to_register_map_low=0;
                pica.vs_setup.UpdateProgramCode(0,Metal::Test::Arithmetic(Op::MOV,0,0));pica.vs_setup.UpdateProgramCode(1,op(Op::END));
                for(unsigned vertex=0;vertex<3;++vertex) {
                    for(unsigned c=0;c<4;++c) pica.gs_setup.uniforms.f[vertex][c]=Pica::f24::FromFloat32(positions[vertex][c]);
                    pica.gs_setup.UpdateProgramCode(vertex*4,Metal::Test::Arithmetic(Op::MOV,0,32+vertex));
                    pica.gs_setup.UpdateProgramCode(vertex*4+1,Metal::Test::Arithmetic(Op::MOV,1,0));
                    pica.gs_setup.UpdateProgramCode(vertex*4+2,op(Op::SETEMIT)|(vertex<<24)|(vertex==2 ? 1u<<23:0));
                    pica.gs_setup.UpdateProgramCode(vertex*4+3,op(Op::EMIT));
                }
                pica.gs_setup.UpdateProgramCode(12,op(Op::END));rasterizer.WriteMetalGSUniforms(0,3,0,0);
                regs.framebuffer.output_merger.alpha_test.enable.Assign(0);
                const std::array<std::array<float,4>,3> colors{{{1,0,0,1},{0,0,1,1},{0,1,0,1}}};
                alias_attributes.base_address.Assign(vertex_address/16);
                std::memcpy(memory.GetPhysicalPointer(vertex_address),colors.data(),sizeof(colors));pica.dirty_regs.SetAllDirty();
                batch(0,3,false);rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                for(unsigned byte=0;byte<32*32*4;++byte)
                    Check(target.GetPtr()[byte]==(byte%4==0 || byte%4==2 ? 255:0),
                          "Queued GS nonaliased control failed to render its final green vertex color");
                rasterizer.ClearAll(false);rasterizer.ResetMetalPrimitiveStream();
                alias_attributes.base_address.Assign(Memory::VRAM_PADDR/16);
                std::memcpy(target.GetPtr(),colors.data(),sizeof(colors));pica.dirty_regs.SetAllDirty();
                batch(0,3,false);rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                // Red RGBA8 words (0xff0000ff) become large negative floats.
                // PICA takes color magnitude before clamping, producing white;
                // the next white words are NaNs, which PICA also maps to one.
                for(unsigned byte=0;byte<32*32*4;++byte) {
                    if(target.GetPtr()[byte]!=255)
                        std::cerr<<"Queued GS alias byte "<<byte<<": actual "<<unsigned(target.GetPtr()[byte])<<", expected 255\n";
                    Check(target.GetPtr()[byte]==255,"Queued GS read stale vertex bytes aliased with its preceding framebuffer write");
                }
                std::cout<<"Actual queued GS framebuffer/vertex alias feedback: nonaliased RGB control ends green; prior aliased red draw changes later float4 colors to white through PICA magnitude/NaN rules, all8192 physical VRAM bytes exact\n";
                std::memcpy(memory.GetPhysicalPointer(vertex_address),raw_vertices.data(),sizeof(raw_vertices));
                load_pica(no_gs_archive);
                std::cout<<gs_cases<<" actual RasterizerMetal GS Point/Fixed/Variable draw cases: immediate/batch, 8/16-bit indices, count-header filtering, binary archive and recreated-rasterizer VRAM continuation passed\n";
                std::cout<<"Actual GS ordered VS temporary-state execution, cache hits and 64-entry circular replacement for 8/16-bit indices passed\n";
            }

            // Poison CPU-side fixed/immediate data so retention must come from the GPU banks.
            const auto old_fixed_color = pica.input_default_attributes[2];
            const auto old_immediate_color = pica.immediate.input_vertex[2];
            for (unsigned channel=0; channel<4; ++channel) {
                const auto green = channel==1 || channel==3 ? 1.f : 0.f;
                pica.input_default_attributes[2][channel] = Pica::f24::FromFloat32(green);
                pica.immediate.input_vertex[2][channel] = Pica::f24::FromFloat32(green);
            }
            regs.pipeline.vertex_attributes.attribute_mask.Assign(1U<<2);
            reset_target();
            batch(0,3,false);
            check_target(true); // Dynamic magenta array must override fixed green.
            regs.pipeline.vertex_attributes.attribute_mask.Assign(0);
            loader.component_count.Assign(2);
            auto green_vertices=raw_vertices;
            for (auto& vertex : green_vertices) { vertex[8]=0; vertex[9]=1; vertex[10]=0; vertex[11]=1; }
            std::memcpy(memory.GetPhysicalPointer(vertex_address),green_vertices.data(),sizeof(green_vertices));
            reset_target();
            batch(0,3,false);
            check_target(true); // Missing color array retains GPU magenta, not CPU green.
            regs.vs.max_input_attribute_index.Assign(1);
            reset_target();
            batch(0,3,false);
            check_target(true); // Real MOV shader still reads retained input register 2.
            auto cyan_vertices=raw_vertices;
            for (auto& vertex : cyan_vertices) { vertex[8]=0; vertex[9]=1; vertex[10]=1; vertex[11]=1; }
            std::memcpy(memory.GetPhysicalPointer(vertex_address),cyan_vertices.data(),sizeof(cyan_vertices));
            loader.component_count.Assign(3);
            reset_target();
            batch(0,3,false);
            check_target(true); // Attribute 2 loads cyan, but unmapped register 2 stays magenta.
            regs.vs.max_input_attribute_index.Assign(2);
            loader.component_count.Assign(2);
            reset_target();
            batch(0,3,false);
            rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
            constexpr std::array<u8,4> cyan{255,255,255,0};
            for (unsigned byte=0; byte<32*32*4; ++byte)
                Check(target.GetPtr()[byte]==cyan[byte%4],"GPU-loaded attribute outside the previous VS mapping was not retained/remapped");
            loader.component_count.Assign(3);
            std::memcpy(memory.GetPhysicalPointer(vertex_address),raw_vertices.data(),sizeof(raw_vertices));
            reset_target(); batch(0,3,false); check_target(true); // Seed GPU magenta again.
            loader.component_count.Assign(2);
            regs.pipeline.max_input_attrib_index.Assign(1);
            reset_target();
            for (unsigned index=0; index<3; ++index) {
                std::vector<u32> words;
                Attribute(words,positions[index]); Attribute(words,{0,0,0,1});
                process(words);
            }
            check_target(true); // Partial immediate packets inherit the prior batch's color.
            regs.pipeline.max_input_attrib_index.Assign(2);
            reset_target();
            for (unsigned index=0; index<3; ++index) {
                std::vector<u32> words;
                Attribute(words,positions[index]); Attribute(words,{0,0,0,1}); Attribute(words,{0,1,1,1});
                process(words);
            }
            reset_target();
            batch(0,3,false); // Array packets now inherit the immediate cyan color.
            rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
            for (unsigned byte=0; byte<32*32*4; ++byte)
                Check(target.GetPtr()[byte]==cyan[byte%4],"Immediate-to-array native GPU attribute retention failed in real VS/FS/VRAM");
            pica.input_default_attributes[2]=old_fixed_color;
            pica.immediate.input_vertex[2]=old_immediate_color;
            loader.component_count.Assign(3);
            std::memcpy(memory.GetPhysicalPointer(vertex_address),raw_vertices.data(),sizeof(raw_vertices));
            std::cout << "Real PICA VS/FS/VRAM: arrays override fixed attributes; retained GPU attributes/unmapped registers survive cache clear and batch/immediate transitions despite poisoned CPU inputs\n";

            reset_target();
            batch(0, 3, false);
            rasterizer.FlushRegion(Memory::VRAM_PADDR + 13, 17);
            for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte) {
                const bool changed = byte >= 13 && byte < 30;
                Check(target.GetPtr()[byte] == (changed ? (byte % 4 == 2 ? 0 : 255) : 0x11),
                      "Partial native cache flush changed bytes outside the requested physical interval");
            }
            check_target(true);
            for (unsigned mask = 0; mask < 16; ++mask) {
                reset_target();
                regs.framebuffer.output_merger.depth_color_mask = mask << 8;
                pica.dirty_regs.SetAllDirty();
                batch(0, 3, false);
                rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                const std::array<unsigned, 4> channel_bits{8, 4, 2, 1};
                for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte) {
                    const auto channel = byte % 4;
                    const auto expected = mask & channel_bits[channel] ? (channel == 2 ? 0 : 255) : 17;
                    Check(target.GetPtr()[byte] == expected, "Real PICA draw/cache integration lost a guest color write mask");
                }
            }
            regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
            constexpr std::array<unsigned, 8> morton_x{0, 1, 4, 5, 16, 17, 20, 21};
            for (unsigned flip = 0; flip < 2; ++flip) for (auto mode :
                 {Pica::RasterizerRegs::ScissorMode::Include, Pica::RasterizerRegs::ScissorMode::Exclude}) {
                reset_target();
                regs.framebuffer.framebuffer.flip.Assign(flip);
                regs.rasterizer.scissor_test.mode.Assign(mode);
                regs.rasterizer.scissor_test.x1.Assign(5); regs.rasterizer.scissor_test.x2.Assign(17);
                regs.rasterizer.scissor_test.y1.Assign(7); regs.rasterizer.scissor_test.y2.Assign(14);
                pica.dirty_regs.SetAllDirty();
                batch(0, 3, false);
                rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 32; ++x) {
                    const auto pixel = ((y / 8) * 4 + x / 8) * 64 + morton_x[x % 8] + 2 * morton_x[y % 8];
                    // Tiled framebuffer bytes use the opposite row order from cache
                    // images. PICA flip=0 additionally mirrors the scissor in the cache.
                    const auto scissor_y = flip ? 31 - y : y;
                    const bool inside = x >= 5 && x <= 17 && scissor_y >= 7 && scissor_y <= 14;
                    const bool drawn = mode == Pica::RasterizerRegs::ScissorMode::Include ? inside : !inside;
                    for (unsigned channel = 0; channel < 4; ++channel) {
                        const auto expected = drawn ? (channel == 2 ? 0 : 255) : 17;
                        const auto actual = target.GetPtr()[pixel * 4 + channel];
                        if (actual != expected) throw std::runtime_error("Real PICA scissor/flip/cache/Morton mismatch: x=" +
                            std::to_string(x) + " y=" + std::to_string(y) + " channel=" + std::to_string(channel) +
                            " mode=" + std::to_string(unsigned(mode)) + " expected=" + std::to_string(expected) +
                            " actual=" + std::to_string(actual));
                    }
                }
            }
            regs.framebuffer.framebuffer.flip.Assign(1);
            regs.rasterizer.scissor_test.mode.Assign(Pica::RasterizerRegs::ScissorMode::Disabled);
            reset_target();
            regs.pipeline.vertex_attributes.base_address.Assign((Memory::VRAM_PADDR + Memory::VRAM_SIZE - 16) / 16);
            bool rejected = false;
            try { batch(0, 3, false); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native rasterizer accepted a vertex source beyond physical VRAM backing");
            check_target(false);
            std::cout << "Real native cache partial flush guards, 16 color masks, scissor Include/Exclude with Morton guest pixels and physical VRAM bounds passed\n";
            const auto fill_address = Memory::VRAM_PADDR + 0x40000;
            auto* fill_bytes = memory.GetPhysicalPointer(fill_address);
            for (unsigned width : {2U, 3U, 4U}) {
                rasterizer.ClearAll(false);
                std::memset(fill_bytes, 0x77, 88);
                Pica::MemoryFillConfig config{};
                config.address_start = (fill_address + 8) / 8;
                config.address_end = (fill_address + 80) / 8;
                config.value_32bit = 0xe93a7f12;
                config.fill_24bit.Assign(width == 3);
                config.fill_32bit.Assign(width == 4);
                Check(rasterizer.AccelerateFill(config), "Native cache rejected a valid guest fill");
                constexpr std::array<u8, 4> pattern{0x12, 0x7f, 0x3a, 0xe9};
                rasterizer.FlushRegion(fill_address + 21, 17);
                for (unsigned byte = 0; byte < 88; ++byte) {
                    const auto expected = byte >= 21 && byte < 38 ? pattern[(byte - 8) % width] : 0x77;
                    Check(fill_bytes[byte] == expected, "Native cache partial fill lost phase or changed guard bytes");
                }
                rasterizer.FlushRegion(fill_address + 8, 72);
                for (unsigned byte = 0; byte < 88; ++byte) {
                    const auto expected = byte >= 8 && byte < 80 ? pattern[(byte - 8) % width] : 0x77;
                    Check(fill_bytes[byte] == expected, "Native cache fill produced incorrect physical guest bytes");
                }
            }
            Pica::MemoryFillConfig bad_fill{};
            bad_fill.address_start = (Memory::VRAM_PADDR + Memory::VRAM_SIZE - 8) / 8;
            bad_fill.address_end = (Memory::VRAM_PADDR + Memory::VRAM_SIZE + 8) / 8;
            rejected = false;
            try { rasterizer.AccelerateFill(bad_fill); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native memory fill crossed its physical backing");
            bad_fill.address_start = 0xffffffff;
            rejected = false;
            try { rasterizer.AccelerateFill(bad_fill); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native memory fill wrapped its register address");
            std::cout << "Real native cache 16/24/32-bit GPU fills, partial interval phase/guards and physical-address overflow rejection passed\n";
            const auto copy_address = Memory::VRAM_PADDR + 0x60000;
            auto* copy_bytes = memory.GetPhysicalPointer(copy_address);
            for (bool overlapping : {false, true}) for (bool gaps : {false, true}) {
                rasterizer.ClearAll(false);
                std::vector<u8> original(4096);
                for (unsigned byte = 0; byte < original.size(); ++byte) original[byte] = u8(byte * 41 + byte / 7);
                std::memcpy(copy_bytes, original.data(), original.size());
                Pica::DisplayTransferConfig config{};
                config.input_address = (copy_address + 64) / 8;
                config.output_address = (copy_address + (overlapping ? 72 : 2048)) / 8;
                config.texture_copy.size = 527; // Low four bits ignored: exactly 512 bytes.
                config.texture_copy.input_width.Assign(gaps ? 2 : 0);
                config.texture_copy.input_gap.Assign(gaps ? 1 : 0);
                config.texture_copy.output_width.Assign(gaps ? 3 : 0);
                config.texture_copy.output_gap.Assign(gaps ? 2 : 0);
                Check(rasterizer.AccelerateTextureCopy(config), "Native raw copy fell back to CPU rendering");
                auto expected = original;
                for (unsigned byte = 0; byte < 512; ++byte) {
                    const auto src = 64 + byte + (gaps ? byte / 32 * 16 : 0);
                    const auto dst = (overlapping ? 72 : 2048) + byte + (gaps ? byte / 48 * 32 : 0);
                    expected[dst] = original[src];
                }
                Check(std::equal(expected.begin(), expected.end(), copy_bytes),
                      "Real native raw copy lost truncation, overlap snapshots, partial lines, gaps or guards");
            }
            {
                rasterizer.ClearAll(false);
                constexpr unsigned window = 1024 * 1024;
                std::vector<u8> original(window);
                unsigned cases{};
                std::uint64_t snapshot_bytes{};
                for (unsigned round = 0; round < 8; ++round)
                    for (unsigned count : {16u, 65536u, 262144u})
                        for (unsigned overlap = 0; overlap < 3; ++overlap)
                            for (bool gaps : {false, true}) {
                                for (unsigned byte = 0; byte < window; ++byte)
                                    original[byte] = u8(byte * 41 + byte / 7 + cases * 53);
                                std::memcpy(copy_bytes, original.data(), window);
                                constexpr unsigned source = 65536;
                                const unsigned dest = overlap == 0 ? window / 2 :
                                    overlap == 1 ? source + 8 : source - 8;
                                Pica::DisplayTransferConfig config{};
                                config.input_address = (copy_address + source) / 8;
                                config.output_address = (copy_address + dest) / 8;
                                config.texture_copy.size = count + 7;
                                config.texture_copy.input_width.Assign(gaps ? 2 : 0);
                                config.texture_copy.input_gap.Assign(gaps ? 1 : 0);
                                config.texture_copy.output_width.Assign(gaps ? 3 : 0);
                                config.texture_copy.output_gap.Assign(gaps ? 2 : 0);
                                Check(rasterizer.AccelerateTextureCopy(config),
                                      "Repeated physical raw transfer left the native GPU path");
                                auto expected = original;
                                for (unsigned byte = 0; byte < count; ++byte)
                                    expected[dest + byte + (gaps ? byte / 48 * 32 : 0)] =
                                        original[source + byte + (gaps ? byte / 32 * 16 : 0)];
                                Check(std::equal(expected.begin(), expected.end(), copy_bytes),
                                      "Repeated raw transfer corrupted snapshots, physical overlap, gaps or guards");
                                snapshot_bytes += Metal::GuestTextureCodec::CopyBytes(count, gaps ? 32 : count, gaps ? 16 : 0);
                                snapshot_bytes += Metal::GuestTextureCodec::CopyBytes(count, gaps ? 48 : count, gaps ? 32 : 0);
                                ++cases;
                            }
                Check(snapshot_bytes > 32u * 1024u * 1024u,
                      "Repeated raw transfer fixture did not exercise enough arena turnover");
                std::cout << cases << " repeated native physical TextureCopy cases, " << snapshot_bytes
                          << " snapshot bytes and " << std::uint64_t(cases) * window
                          << " checked guest bytes: changing source/destination, forward/backward overlap, gaps, partial rows and arena reuse passed\n";
            }
            reset_target();
            regs.pipeline.vertex_attributes.base_address.Assign(vertex_address / 16);
            batch(0, 3, false); // Leave GPU-owned framebuffer bytes unflushed in physical VRAM.
            rasterizer.InvalidateRegion(copy_address, 4096);
            std::memset(copy_bytes, 0x55, 4096);
            Pica::DisplayTransferConfig dirty_copy{};
            dirty_copy.input_address = Memory::VRAM_PADDR / 8;
            dirty_copy.output_address = copy_address / 8;
            dirty_copy.texture_copy.size = 512;
            dirty_copy.texture_copy.input_width.Assign(1); dirty_copy.texture_copy.input_gap.Assign(1);
            dirty_copy.texture_copy.output_width.Assign(3); dirty_copy.texture_copy.output_gap.Assign(1);
            Check(rasterizer.AccelerateTextureCopy(dirty_copy), "Dirty framebuffer raw copy was not handled natively");
            std::vector<u8> expected_copy(4096, 0x55);
            for (unsigned byte = 0; byte < 512; ++byte) {
                const auto source_byte = byte + byte / 16 * 16;
                expected_copy[byte + byte / 48 * 16] = source_byte % 4 == 2 ? 0 : 255;
            }
            Check(std::equal(expected_copy.begin(), expected_copy.end(), copy_bytes),
                  "Native raw transfer read stale physical bytes instead of flushing GPU-owned framebuffer data");
            Pica::DisplayTransferConfig bad_copy = dirty_copy;
            bad_copy.input_address = (Memory::VRAM_PADDR + Memory::VRAM_SIZE - 8) / 8;
            rejected = false;
            try { rasterizer.AccelerateTextureCopy(bad_copy); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native raw copy exceeded physical input backing");
            bad_copy = dirty_copy; bad_copy.output_address = 0xffffffff;
            rejected = false;
            try { rasterizer.AccelerateTextureCopy(bad_copy); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native raw copy wrapped its physical output address");
            bad_copy = dirty_copy; bad_copy.texture_copy.input_width.Assign(0);
            rejected = false;
            try { rasterizer.AccelerateTextureCopy(bad_copy); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native raw copy accepted a zero-width gapped input");
            bad_copy.texture_copy.size = 15;
            Check(rasterizer.AccelerateTextureCopy(bad_copy), "Empty aligned raw transfer required a CPU fallback");
            std::cout << "Real RasterizerMetal raw TextureCopy fallback: gaps/partial rows, size truncation, physical overlap, GPU framebuffer coherence, guards and bounds passed\n";
            const auto display_input = Memory::VRAM_PADDR + 0x90000;
            const auto display_output = Memory::VRAM_PADDR + 0xc0000;
            unsigned display_cases{};
            for (unsigned input_format = 0; input_format < 5; ++input_format)
                for (unsigned output_format = 0; output_format < 5; ++output_format)
                    for (unsigned scaling = 0; scaling < 3; ++scaling)
                        for (unsigned tiling = 0; tiling < 4; ++tiling)
                            for (bool flip : {false, true}) for (bool block32 : {false, true}) {
                                rasterizer.ClearAll(false);
                                const auto src_format = Pica::PixelFormat(input_format);
                                const auto dst_format = Pica::PixelFormat(output_format);
                                const auto src_bpp = Pica::BytesPerPixel(src_format), dst_bpp = Pica::BytesPerPixel(dst_format);
                                const bool input_tiled = tiling != 2, output_tiled = tiling != 0;
                                const unsigned scale_x = scaling ? 2 : 1, scale_y = scaling == 2 ? 2 : 1;
                                const unsigned width = 64 / scale_x, height = 32 / scale_y;
                                std::vector<u8> source(64 * 32 * src_bpp);
                                for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 64; ++x) {
                                    // Low-value checkerboard catches nearest/rounding errors;
                                    // gradients exercise every channel's guest bit precision.
                                    const bool checker = x < 4 && y < 4;
                                    Encode(src_format, {u8(checker ? x % 2 : x * 29 + y * 5),
                                        u8(checker ? y % 2 + 1 : x * 7 + y * 31), u8(x * 19 + y * 13),
                                        u8(checker ? 127 + x % 2 : x * 11 + y * 23)},
                                        source.data() + PixelOffset(x,y,64,input_tiled) * src_bpp);
                                }
                                std::memcpy(memory.GetPhysicalPointer(display_input), source.data(), source.size());
                                const auto output_size = width * height * dst_bpp;
                                auto* output = memory.GetPhysicalPointer(display_output);
                                std::memset(output, 0xcd, output_size + 32);
                                Pica::DisplayTransferConfig config{};
                                config.input_address = display_input / 8;
                                config.output_address = (display_output + 16) / 8;
                                config.input_width.Assign(64); config.input_height.Assign(32);
                                config.output_width.Assign(64); config.output_height.Assign(32);
                                config.input_format.Assign(src_format); config.output_format.Assign(dst_format);
                                config.input_linear.Assign(tiling >= 2);
                                config.dont_swizzle.Assign(tiling == 1 || tiling == 3);
                                config.scaling.Assign(Pica::DisplayTransferConfig::ScalingMode(scaling));
                                config.flip_vertically.Assign(flip);
                                config.block_32.Assign(block32);
                                Check(rasterizer.AccelerateDisplayTransfer(config), "Native display transfer required a CPU fallback");
                                std::vector<u8> expected(output_size + 32, 0xcd);
                                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                                    std::array<unsigned,4> sum{};
                                    for (unsigned dy = 0; dy < scale_y; ++dy) for (unsigned dx = 0; dx < scale_x; ++dx) {
                                        const auto sx = x * scale_x + dx;
                                        const auto sy = flip ? 31 - (y * scale_y + dy) : y * scale_y + dy;
                                        const auto color = Decode(src_format, source.data() + PixelOffset(sx,sy,64,input_tiled) * src_bpp);
                                        for (unsigned channel = 0; channel < 4; ++channel) sum[channel] += color[channel];
                                    }
                                    const auto samples = scale_x * scale_y;
                                    Encode(dst_format, {u8(sum[0]/samples), u8(sum[1]/samples), u8(sum[2]/samples), u8(sum[3]/samples)},
                                           expected.data() + 16 + PixelOffset(x,y,width,output_tiled) * dst_bpp);
                                }
                                if (!output_tiled) {
                                    Pica::FramebufferConfig lcd{};
                                    lcd.width.Assign(width); lcd.height.Assign(height); lcd.color_format.Assign(dst_format);
                                    Metal::ScreenImage screen{};
                                    Check(rasterizer.GetScreen(lcd, display_output + 16, width, screen),
                                          "Native display output was not available from its GPU-owned cache surface");
                                    const auto origin_x = NSUInteger(std::lround(screen.texcoords.top * screen.texture.width));
                                    const auto origin_y = NSUInteger(std::lround(screen.texcoords.left * screen.texture.height));
                                    const auto pixels = Metal::TransferRuntime{device, scheduler}.Readback({screen.texture,0,0,
                                        MTLRegionMake2D(origin_x,origin_y,width,height)}, width * 4);
                                    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                                        const auto color = Decode(dst_format, expected.data() + 16 + (y * width + x) * dst_bpp);
                                        for (unsigned channel = 0; channel < 4; ++channel)
                                            Check(std::to_integer<u8>(pixels[(y * width + x) * 4 + channel]) == color[channel],
                                                  "Native display transfer cache pixels lacked guest precision before physical flush");
                                    }
                                }
                                rasterizer.FlushRegion(display_output + 16, output_size);
                                for (unsigned byte = 0; byte < expected.size(); ++byte)
                                    if (output[byte] != expected[byte]) throw std::runtime_error("Native PICA display-transfer mismatch: src=" +
                                        std::to_string(input_format) + " dst=" + std::to_string(output_format) + " scaling=" +
                                        std::to_string(scaling) + " tiling=" + std::to_string(tiling) + " flip=" + std::to_string(flip) +
                                        " byte=" + std::to_string(byte) + " expected=" + std::to_string(expected[byte]) +
                                        " actual=" + std::to_string(output[byte]));
                                ++display_cases;
                            }
            std::cout << display_cases << " real native PICA DisplayTransfer cases: 25 format pairs, NoScale/ScaleX/ScaleXY, all four tiling flag combinations including bit-5 precedence, flips, both block32 compatibility flags, integer box filtering, pre-flush cache precision and existing-layout guard bytes passed\n";
            unsigned stride_cases{};
            for (unsigned scaling = 0; scaling < 3; ++scaling)
                for (unsigned tiling = 0; tiling < 4; ++tiling) for (bool flip : {false,true})
                    for (bool crop : {false,true}) for (bool block_32 : {false,true}) {
                rasterizer.ClearAll(false);
                const bool input_tiled = tiling != 2, output_tiled = tiling != 0;
                std::vector<u8> source(72 * 32 * 4, 0xa5);
                for (unsigned y = 0; y < 32; ++y) for (unsigned x = 0; x < 64; ++x)
                    Encode(Pica::PixelFormat::RGBA8, {u8(x*3),u8(y*7),u8(x+y),u8(127+x%2)},
                           source.data() + PixelOffset(x,y,72,input_tiled)*4);
                std::memcpy(memory.GetPhysicalPointer(display_input),source.data(),source.size());
                auto* output = memory.GetPhysicalPointer(display_output);
                std::memset(output,0xcd,10000);
                Pica::DisplayTransferConfig config{};
                config.input_address = display_input/8; config.output_address = (display_output+16)/8;
                config.input_width.Assign(72); config.input_height.Assign(32);
                config.output_width.Assign(64); config.output_height.Assign(32);
                config.input_linear.Assign(tiling >= 2);
                config.dont_swizzle.Assign(tiling == 1 || tiling == 3);
                config.crop_input_lines.Assign(crop); config.flip_vertically.Assign(flip);
                config.block_32.Assign(block_32);
                config.scaling.Assign(Pica::DisplayTransferConfig::ScalingMode(scaling));
                const auto scale_x = scaling ? 2U : 1U, scale_y = scaling == 2 ? 2U : 1U;
                const auto width = 64 / scale_x, height = 32 / scale_y;
                const auto shift = flip && crop ? (72U - 64U) * 31U * 4U : 0U;
                Check(rasterizer.AccelerateDisplayTransfer(config), "Native reference display transfer required CPU execution");
                rasterizer.FlushRegion(display_output+16+shift,width*height*4);
                std::vector<u8> expected(10000,0xcd);
                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                    std::array<unsigned,4> sum{};
                    for (unsigned dy = 0; dy < scale_y; ++dy) for (unsigned dx = 0; dx < scale_x; ++dx) {
                        const auto sx = x*scale_x+dx, sy = flip ? 31-(y*scale_y+dy) : y*scale_y+dy;
                        const auto color = Decode(Pica::PixelFormat::RGBA8,source.data()+PixelOffset(sx,sy,72,input_tiled)*4);
                        for (unsigned channel = 0; channel < 4; ++channel) sum[channel] += color[channel];
                    }
                    Encode(Pica::PixelFormat::RGBA8,{u8(sum[0]/(scale_x*scale_y)),u8(sum[1]/(scale_x*scale_y)),
                        u8(sum[2]/(scale_x*scale_y)),u8(sum[3]/(scale_x*scale_y))},
                        expected.data()+16+shift+PixelOffset(x,y,width,output_tiled)*4);
                }
                Check(std::equal(expected.begin(),expected.end(),output),
                      "Native display transfer lost source stride, crop/flip skew, box filtering or physical guard bytes");
                ++stride_cases;
            }
            Check(display_cases == 1200 && stride_cases == 96, "Incomplete native display-transfer matrix");
            rasterizer.ClearAll(false);
            Pica::DisplayTransferConfig invalid_display{};
            invalid_display.input_address = display_input/8; invalid_display.output_address = display_output/8;
            invalid_display.input_width.Assign(64); invalid_display.input_height.Assign(32);
            invalid_display.output_width.Assign(64); invalid_display.output_height.Assign(32);
            invalid_display.output_address = 0xffffffff;
            rejected = false;
            try { rasterizer.AccelerateDisplayTransfer(invalid_display); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected,"Native display transfer wrapped its physical target address");
            invalid_display.output_address = display_output/8;
            invalid_display.input_address = (Memory::VRAM_PADDR+Memory::VRAM_SIZE-8)/8;
            rejected = false;
            try { rasterizer.AccelerateDisplayTransfer(invalid_display); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected,"Native display transfer exceeded physical input backing");
            invalid_display.input_address = display_input/8;
            invalid_display.scaling.Assign(Pica::DisplayTransferConfig::ScalingMode(3));
            rejected = false;
            try { rasterizer.AccelerateDisplayTransfer(invalid_display); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected,"Native display transfer accepted reserved scale mode");
            invalid_display.scaling.Assign(Pica::DisplayTransferConfig::NoScale);
            invalid_display.output_width.Assign(0);
            Check(rasterizer.AccelerateDisplayTransfer(invalid_display),"Empty display transfer required CPU execution");
            auto block32_regs = regs;
            block32_regs.reg_array[0x11b] |= 1u;
            (void)Metal::BuildPicaDrawState(block32_regs);
            std::cout << stride_cases << " native display-transfer stride/flag cases: 48 cases per block32 flag; "
                         "block32 framebuffer accepted through logical GPU-cache compatibility; "
                         "physical 32x32 addressing remains unverified\n";
            {
                using C = Pica::TexturingRegs::TextureConfig;
                using Tev = Pica::TexturingRegs::TevStageConfig;
                regs = {};
                Metal::Test::MovVertex(regs,pica.vs_setup);
                regs.vs.output_mask.Assign(15); regs.rasterizer.vs_output_total.Assign(4);
                regs.rasterizer.vs_output_attributes[3].raw=0x1f1f0d0c;
                pica.vs_setup.UpdateProgramCode(3,Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,3,3));
                pica.vs_setup.UpdateProgramCode(4,std::uint32_t(nihstro::OpCode::Id::END)<<26U);
                regs.vs.max_input_attribute_index.Assign(3); regs.pipeline.max_input_attrib_index.Assign(3);
                regs.vs.input_attribute_to_register_map_low=0x3210;
                regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                regs.framebuffer.output_merger.depth_color_mask=15<<8;
                regs.framebuffer.framebuffer.allow_color_write.Assign(1);
                regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR/8);
                regs.framebuffer.framebuffer.width.Assign(32); regs.framebuffer.framebuffer.height.Assign(31);
                regs.framebuffer.framebuffer.flip.Assign(1);
                regs.rasterizer.viewport_size_x.Assign(67<<16); regs.rasterizer.viewport_size_y.Assign(67<<16);
                regs.rasterizer.viewport_depth_near_plane.Assign(62<<16);
                regs.texturing.main_config.texture0_enable.Assign(1);
                auto& texture=regs.texturing.texture0;
                texture.width.Assign(8); texture.height.Assign(8); texture.address.Assign(display_input/8);
                texture.border_color.raw=17U|(83U<<8)|(149U<<16)|(211U<<24);
                regs.texturing.texture0_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                // Later zero-initialized TEV stages read Previous when explicitly configured.
                std::array<Tev*,6> stages{&regs.texturing.tev_stage0,&regs.texturing.tev_stage1,
                    &regs.texturing.tev_stage2,&regs.texturing.tev_stage3,&regs.texturing.tev_stage4,&regs.texturing.tev_stage5};
                for(auto* tev:stages) tev->sources_raw=0xF000F;
                stages[0]->color_source1.Assign(Tev::Source::Texture0);
                stages[0]->alpha_source1.Assign(Tev::Source::Texture0);
                constexpr std::array<unsigned,4> interior{84,107,130,233},border{17,83,149,211};
                unsigned cases{};
                for(unsigned path=0;path<5;++path) {
                    const unsigned unit=path<2 ? 0 : path==2 ? 1 : 2;
                    regs.texturing.main_config.texture0_enable.Assign(unit==0);
                    regs.texturing.main_config.texture1_enable.Assign(unit==1);
                    regs.texturing.main_config.texture2_enable.Assign(unit==2);
                    regs.texturing.main_config.texture2_use_coord1.Assign(path==4);
                    regs.texturing.texture0.type.Assign(path==1 ? C::Projection2D : C::Texture2D);
                    regs.texturing.texture1=texture; regs.texturing.texture2=texture;
                    regs.texturing.texture1_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                    regs.texturing.texture2_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                    std::array<C*,3> configurations{&regs.texturing.texture0,&regs.texturing.texture1,&regs.texturing.texture2};
                    auto& selected_texture=*configurations[unit];
                    regs.rasterizer.vs_output_attributes[3].raw=path==1 ? 0x1f100d0c :
                        path==2 || path==4 ? 0x1f1f0f0e : path==3 ? 0x1f1f1716 : 0x1f1f0d0c;
                    stages[0]->color_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0)+unit));
                    stages[0]->alpha_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0)+unit));
                for(auto mode:{C::ClampToBorder,C::ClampToEdge2,C::ClampToBorder2})
                    for(bool linear:{false,true}) for(float u:{-.03125f,0.f,1.f,1.03125f,-.0625f,-1.0625f,.5f,1.0625f}) {
                        reset_target();
                        for(unsigned pixel=0;pixel<64;++pixel)
                            Encode(Pica::PixelFormat::RGBA8,{84,107,130,233},memory.GetPhysicalPointer(display_input)+pixel*4);
                        selected_texture.wrap_s.Assign(mode); selected_texture.wrap_t.Assign(C::ClampToEdge);
                        selected_texture.min_filter.Assign(C::TextureFilter(linear)); selected_texture.mag_filter.Assign(C::TextureFilter(linear));
                        pica.dirty_regs.SetAllDirty();
                        for(unsigned vertex=0;vertex<3;++vertex) {
                            std::vector<u32> words;
                            Attribute(words,positions[vertex]); Attribute(words,{0,0,0,1});
                            Attribute(words,{1,0,1,1});
                            Attribute(words,path==1 ? std::array<float,4>{2*u,.375f,2,0} :
                                                     std::array<float,4>{u,.1875f,0,0}); process(words);
                        }
                        rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                        const bool border_mode=mode==C::ClampToBorder || (mode==C::ClampToBorder2 && u>=0);
                        double fraction{};
                        if(border_mode) {
                            const double p=u*8-(linear?.5:0);
                            const int tap=int(std::floor(p)); const double weight=p-tap;
                            const auto outside=[](int index){return index<0||index>=8 ? 1.0 : 0.0;};
                            fraction=linear ? outside(tap)*(1-weight)+outside(tap+1)*weight : outside(tap);
                        }
                        for(unsigned pixel=0;pixel<1024;++pixel) {
                            const auto actual=Decode(Pica::PixelFormat::RGBA8,target.GetPtr()+pixel*4);
                            for(unsigned c=0;c<4;++c) {
                                const int expected=std::lround(interior[c]*(1-fraction)+border[c]*fraction);
                                Check(std::abs(int(actual[c])-expected)<=1,"Actual native sampler command/VS/FS/cache/VRAM mismatch");
                            }
                        }
                        ++cases;
                    }
                }
                std::cout<<cases<<" actual RasterizerMetal extended sampler draws: all3 units, coord1 alias, projection division, packed PICA commands, GPU VS/FS, custom border/alpha, linear edge taps and physical VRAM passed\n";
            }
            {
                using C=Pica::TexturingRegs::TextureConfig;
                using Tev=Pica::TexturingRegs::TevStageConfig;
                regs={};Metal::Test::MovVertex(regs,pica.vs_setup);
                regs.vs.output_mask.Assign(15);regs.rasterizer.vs_output_total.Assign(4);
                regs.rasterizer.vs_output_attributes[3].raw=0x1f100d0c;
                pica.vs_setup.UpdateProgramCode(3,Metal::Test::Arithmetic(nihstro::OpCode::Id::MOV,3,3));
                pica.vs_setup.UpdateProgramCode(4,std::uint32_t(nihstro::OpCode::Id::END)<<26U);
                regs.vs.max_input_attribute_index.Assign(3);regs.pipeline.max_input_attrib_index.Assign(3);
                regs.vs.input_attribute_to_register_map_low=0x3210;
                regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                regs.framebuffer.output_merger.depth_color_mask=15<<8;regs.framebuffer.framebuffer.allow_color_write.Assign(1);
                regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR/8);
                regs.framebuffer.framebuffer.width.Assign(32);regs.framebuffer.framebuffer.height.Assign(31);
                regs.framebuffer.framebuffer.flip.Assign(1);
                regs.rasterizer.viewport_size_x.Assign(67<<16);regs.rasterizer.viewport_size_y.Assign(67<<16);
                regs.rasterizer.viewport_depth_near_plane.Assign(62<<16);
                regs.texturing.main_config.texture0_enable.Assign(1);
                regs.texturing.texture0_format.Assign(Pica::TexturingRegs::TextureFormat::RGBA8);
                auto& texture=regs.texturing.texture0;
                texture.width.Assign(8);texture.height.Assign(8);texture.lod.min_level.Assign(0);texture.lod.max_level.Assign(0);
                texture.border_color.raw=17U|(83U<<8)|(149U<<16)|(211U<<24);
                std::array<Tev*,6> stages{&regs.texturing.tev_stage0,&regs.texturing.tev_stage1,&regs.texturing.tev_stage2,
                    &regs.texturing.tev_stage3,&regs.texturing.tev_stage4,&regs.texturing.tev_stage5};
                for(auto* stage:stages)stage->sources_raw=0xF000F;
                stages[0]->color_source1.Assign(Tev::Source::Texture0);stages[0]->alpha_source1.Assign(Tev::Source::Texture0);
                std::array<PAddr,6> addresses{};
                for(unsigned face=0;face<6;++face){addresses[face]=Memory::VRAM_PADDR+0x110000+face*0x1000;
                    if(face==0)texture.address.Assign(addresses[face]/8);
                    else regs.texturing.cube_address[face-1].Assign(addresses[face]/8);}
                constexpr std::array<std::array<float,4>,6> directions{{{.5f,0,0,0},{-.5f,0,0,0},
                    {0,.5f,0,0},{0,-.5f,0,0},{0,0,.5f,0},{0,0,-.5f,0}}};
                unsigned cube_cases{},shadow_cases{};
                for(auto type:{C::TextureCube,C::Shadow2D,C::ShadowCube})
                    for(auto wrap:{C::ClampToBorder,C::ClampToEdge2,C::ClampToBorder2})for(bool linear:{false,true}) {
                        texture.type.Assign(type);texture.wrap_s.Assign(wrap);texture.wrap_t.Assign(C::ClampToBorder);
                        texture.mag_filter.Assign(C::TextureFilter(linear));texture.min_filter.Assign(C::TextureFilter(linear));
                        for(unsigned face=0;face<6;++face)for(unsigned pixel=0;pixel<64;++pixel){
                            const Common::Vec4<u8> color=type==C::TextureCube?
                                Common::Vec4<u8>{u8(20+face*30),u8(40+face*17),u8(60+face*11),u8(180+face*9)}:
                                Common::Vec4<u8>{u8(64+face*20),0,0,192};
                            // Shadow reads alias the canonical host RGBA8 image as R32Uint:
                            // low byte intensity, high24 depth. Guest encode/decode remains unchanged.
                            Encode(Pica::PixelFormat::RGBA8,color,memory.GetPhysicalPointer(addresses[face])+pixel*4);
                        }
                        for(unsigned face=0;face<(type==C::Shadow2D?2U:6U);++face){
                            reset_target();pica.dirty_regs.SetAllDirty();
                            const auto coordinate=type==C::Shadow2D?
                                (face==0?std::array<float,4>{.25f,.25f,.5f,0}:std::array<float,4>{-.25f,.25f,.5f,0}):directions[face];
                            for(unsigned vertex=0;vertex<3;++vertex){std::vector<u32> words;
                                Attribute(words,positions[vertex]);Attribute(words,{0,0,0,1});
                                Attribute(words,{1,1,1,1});Attribute(words,coordinate);process(words);}
                            rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                            const std::array<unsigned,4> expected=type==C::TextureCube?
                                std::array<unsigned,4>{20+face*30,40+face*17,60+face*11,180+face*9}:
                                type==C::Shadow2D&&face==1?std::array<unsigned,4>{255,255,255,255}:
                                std::array<unsigned,4>{64+face*20,64+face*20,64+face*20,64+face*20};
                            for(unsigned pixel=0;pixel<1024;++pixel){const auto actual=Decode(Pica::PixelFormat::RGBA8,target.GetPtr()+pixel*4);
                                for(unsigned c=0;c<4;++c)if(std::abs(int(actual[c])-int(expected[c]))>1) {
                                    const auto snapshot=rasterizer.SaveMetalState();
                                    std::cerr<<"special sampler GPU input/register banks:";
                                    for(unsigned slot=0;slot<4;++slot){std::array<float,4> values{};
                                        std::memcpy(values.data(),snapshot.data()+32+272+256+slot*16,16);
                                        std::cerr<<" r"<<slot;for(float value:values)std::cerr<<' '<<value;}
                                    std::cerr<<" output="<<unsigned(actual[0])<<','<<unsigned(actual[1])<<','<<unsigned(actual[2])<<','<<unsigned(actual[3])<<'\n';
                                    throw std::runtime_error("Actual special sampler mismatch: type="+std::to_string(unsigned(type))+
                                        " wrap="+std::to_string(unsigned(wrap))+" linear="+std::to_string(linear)+
                                        " face="+std::to_string(face)+" pixel="+std::to_string(pixel)+" channel="+
                                        std::to_string(c)+" expected="+std::to_string(expected[c])+" actual="+std::to_string(actual[c]));}}
                            if(type==C::TextureCube)++cube_cases;else++shadow_cases;
                        }
                }
                rasterizer.ClearAll(false);
                std::cout<<cube_cases<<" actual PICA cube-border draws and "<<shadow_cases<<" actual PICA shadow-border draws: all6 physical faces, mixed modes, filters, typed storage, bounded shadow reads, packed commands/GPU VS/FS/cache/VRAM passed\n";
            }
            {
                using C=Pica::TexturingRegs::TextureConfig;
                using Tev=Pica::TexturingRegs::TevStageConfig;
                Settings::values.resolution_factor.SetValue(3);
                regs.texturing.main_config.texture0_enable.Assign(1);
                regs.texturing.main_config.texture1_enable.Assign(0);
                regs.texturing.main_config.texture2_enable.Assign(0);
                regs.texturing.texture0.type.Assign(C::Texture2D);
                regs.texturing.texture0.wrap_s.Assign(C::ClampToEdge);
                regs.texturing.texture0.wrap_t.Assign(C::ClampToEdge);
                regs.texturing.texture0.mag_filter.Assign(C::Nearest);
                regs.texturing.texture0.min_filter.Assign(C::Nearest);
                regs.texturing.texture0.address.Assign(display_input/8);
                regs.rasterizer.vs_output_attributes[3].raw=0x1f1f0d0c;
                regs.texturing.tev_stage0.color_source1.Assign(Tev::Source::Texture0);
                regs.texturing.tev_stage0.alpha_source1.Assign(Tev::Source::Texture0);
                std::array<std::vector<u8>,5> results;
                for(unsigned filter=1;filter<=5;++filter) {
                    Settings::values.texture_filter.SetValue(Settings::TextureFilter(filter));
                    rasterizer.TickFrame(); reset_target();
                    for(unsigned y=0;y<8;++y) for(unsigned x=0;x<8;++x)
                        Encode(Pica::PixelFormat::RGBA8,x>=y ? Common::Vec4<u8>{255,31,83,255} :
                                                               Common::Vec4<u8>{13,211,71,127},
                               memory.GetPhysicalPointer(display_input)+PixelOffset(x,y,8,true)*4);
                    pica.dirty_regs.SetAllDirty();
                    for(unsigned vertex=0;vertex<3;++vertex) {
                        std::vector<u32> words;
                        Attribute(words,positions[vertex]); Attribute(words,{0,0,0,1}); Attribute(words,{1,0,1,1});
                        Attribute(words,{(positions[vertex][0]+1)*.5f,(positions[vertex][1]+1)*.5f,0,0});
                        process(words);
                    }
                    rasterizer.FlushRegion(Memory::VRAM_PADDR,32*32*4);
                    results[filter-1].assign(target.GetPtr(),target.GetPtr()+32*32*4);
                }
                for(unsigned a=0;a<5;++a) for(unsigned b=a+1;b<5;++b)
                    Check(results[a]!=results[b],"Real Metal filter settings/cache upload/PICA draw paths produced identical filter outputs");
                Settings::values.texture_filter.SetValue(Settings::TextureFilter::NoFilter);
                Settings::values.resolution_factor.SetValue(1); rasterizer.TickFrame();
                std::cout<<"All5 real RasterizerMetal filter choices: settings change/cache tick, scaled guest upload, packed PICA VS/FS, native color cache and physical VRAM produce distinct edge images without emulation\n";
            }
            {
                using F=Settings::TextureFilter;
                constexpr unsigned width=64,height=64,scale=3;
                const PAddr address=Memory::VRAM_PADDR+0x190000;
                Settings::values.resolution_factor.SetValue(scale);
                rasterizer.ClearAll(false);
                unsigned cases{};
                for(unsigned filter=1;filter<=5;++filter) {
                    Settings::values.texture_filter.SetValue(F(filter));
                    Metal::RasterizerCache cache{memory,custom,rasterizer.Textures(),regs,renderer};
                    VideoCore::SurfaceParams params{};params.addr=address;params.width=width;params.height=height;
                    params.stride=width;params.res_scale=scale;params.is_tiled=true;params.pixel_format=VideoCore::PixelFormat::RGBA8;
                    params.UpdateParams();
                    auto* backing=memory.GetPhysicalPointer(address);
                    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x)
                        Encode(Pica::PixelFormat::RGBA8,{u8(x*11+y*3),u8(x*7+y*17),u8(x*19+y*5),u8(31+x+y)},
                               backing+PixelOffset(x,y,width,true)*4);
                    const auto id=cache.GetSurface(params,VideoCore::ScaleMatch::Exact,true);
                    auto& surface=cache.GetSurface(id);
                    Metal::TransferRuntime transfer{device,scheduler};
                    VideoCore::TextureClear clear{};clear.texture_rect={0,height*scale,8*scale,(height-8)*scale};
                    clear.value.color={17/255.f,34/255.f,51/255.f,68/255.f};
                    rasterizer.Textures().ClearTexture(surface,clear);
                    cache.InvalidateRegion(address,8*8*4,id);
                    const auto before=transfer.Readback({surface.Image(),0,0,MTLRegionMake2D(0,0,width*scale,height*scale)},width*scale*4);
                    const auto changed=address+PixelOffset(32,32,width,true)*4;
                    Encode(Pica::PixelFormat::RGBA8,{255,0,211,99},memory.GetPhysicalPointer(changed));
                    cache.InvalidateRegion(changed,8*8*4);
                    auto neighbor=params;neighbor.addr=address+PixelOffset(24,32,width,true)*4;
                    neighbor.width=8;neighbor.height=8;neighbor.UpdateParams();
                    Check(neighbor.end<=changed,"Filter cache fixture unexpectedly validated the dirty tile directly");
                    const auto [neighbor_id,rectangle]=cache.GetSurfaceSubRect(neighbor,VideoCore::ScaleMatch::Exact,true);
                    Check(neighbor_id==id && rectangle.GetArea(),"Neighbor-only filter request did not reuse the cached surface");
                    Check(surface.IsRegionValid(VideoCore::SurfaceInterval(changed,changed+8*8*4)),
                          "Native filtered neighbor validation left its out-of-request dependency dirty");
                    for(unsigned pixel=0;pixel<64;++pixel)
                        Check(Decode(Pica::PixelFormat::RGBA8,backing+pixel*4)==Common::Vec4<u8>{17,34,51,68},
                              "Full canonical filter snapshot did not flush its outside GPU owner into guest memory");
                    Metal::Surface reference{rasterizer.Textures(),params};
                    rasterizer.Textures().UploadGuest(reference,params,{backing,params.size});
                    const auto expected=transfer.Readback({reference.Image(),0,0,MTLRegionMake2D(0,0,width*scale,height*scale)},width*scale*4);
                    const auto actual=transfer.Readback({surface.Image(),0,0,MTLRegionMake2D(0,0,width*scale,height*scale)},width*scale*4);
                    for(unsigned y=0;y<height*scale;++y) for(unsigned x=0;x<width*scale;++x) for(unsigned c=0;c<4;++c) {
                        const auto byte=(y*width*scale+x)*4+c;
                        const bool affected=x>=24*scale&&x<48*scale&&y>=16*scale&&y<40*scale;
                        Check(actual[byte]==(affected?expected[byte]:before[byte]),
                              "Native cache neighbor refresh was stale, lost UV phase or overwrote unrelated GPU-owned scaled pixels");
                    }
                    cache.ClearAll(false);++cases;
                }
                Settings::values.texture_filter.SetValue(F::NoFilter);
                Settings::values.resolution_factor.SetValue(1);rasterizer.TickFrame();
                std::cout<<cases<<" real native filter-cache neighbor cases: out-of-request dirty tiles, full canonical mip snapshots, outside GPU-owner flush, scoped halo writes and scaled-pixel preservation passed\n";
            }
            {
                Pica::FramebufferConfig lcd{};
                lcd.width.Assign(8); lcd.height.Assign(2); lcd.color_format.Assign(Pica::PixelFormat::RGBA8);
                Metal::ScreenImage screen{};
                Check(!rasterizer.GetScreen(lcd, 0, 8, screen) &&
                      !rasterizer.GetScreen(lcd, Memory::VRAM_PADDR, 0, screen), "Empty LCD address/stride was resolved");
                lcd.width.Assign(0);
                Check(!rasterizer.GetScreen(lcd, Memory::VRAM_PADDR, 8, screen), "Zero-width LCD was resolved");
                lcd.width.Assign(8); lcd.height.Assign(0);
                Check(!rasterizer.GetScreen(lcd, Memory::VRAM_PADDR, 8, screen), "Zero-height LCD underflow was accepted");
                lcd.height.Assign(2);
                for (unsigned kind = 0; kind < 3; ++kind) {
                    auto invalid = lcd;
                    if (kind == 0) invalid.color_format.Assign(Pica::PixelFormat(5));
                    bool rejected{};
                    try { rasterizer.GetScreen(invalid, kind == 1 ? Memory::VRAM_PADDR_END - 16 : Memory::VRAM_PADDR,
                                               kind == 2 ? UINT32_MAX : 8, screen); }
                    catch (const std::invalid_argument&) { rejected = true; }
                    Check(rejected, "Invalid LCD format/backing/stride overflow was accepted");
                }
                std::cout << "Native LCD preflight: empty address/stride/dimensions, invalid format, short physical backing and64-bit stride overflow rejected before cache mutation passed\n";
            }
            {
                rasterizer.ClearAll(false); regs = {};
                Metal::Test::MovVertex(regs, pica.vs_setup);
                regs.vs.max_input_attribute_index.Assign(3);
                regs.pipeline.max_input_attrib_index.Assign(3);
                regs.vs.input_attribute_to_register_map_low = 0x3210;
                regs.lighting.disable.Assign(1);
                regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
                regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
                regs.framebuffer.framebuffer.allow_color_write.Assign(1);
                regs.framebuffer.framebuffer.color_buffer_address.Assign(Memory::VRAM_PADDR / 8);
                regs.framebuffer.framebuffer.width.Assign(32); regs.framebuffer.framebuffer.height.Assign(31);
                regs.framebuffer.framebuffer.flip.Assign(1);
                regs.rasterizer.viewport_size_x.Assign(67 << 16);
                regs.rasterizer.viewport_size_y.Assign(67 << 16);
                regs.rasterizer.viewport_depth_near_plane.Assign(62 << 16);
                constexpr u32 count = 3 * 65536, plane_bytes = count * 16;
                constexpr PAddr base = Memory::FCRAM_PADDR + 0x1000000;
                regs.pipeline.vertex_attributes.base_address.Assign(base / 16);
                regs.pipeline.vertex_attributes.max_attribute_index.Assign(3);
                auto& attributes = regs.pipeline.vertex_attributes;
                attributes.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attributes.format1.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attributes.format2.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attributes.format3.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attributes.size0.Assign(3); attributes.size1.Assign(3);
                attributes.size2.Assign(3); attributes.size3.Assign(3);
                regs.pipeline.num_vertices = count;
                for (unsigned attribute = 0; attribute < 4; ++attribute) {
                    auto& source = regs.pipeline.vertex_attributes.attribute_loaders[attribute];
                    source.data_offset.Assign(attribute * plane_bytes);
                    source.comp0.Assign(attribute); source.component_count.Assign(1); source.byte_count.Assign(16);
                    auto memory_plane = memory.GetPhysicalRef(base + attribute * plane_bytes);
                    Check(memory_plane && memory_plane.GetSize() >= plane_bytes,
                          "Large batch fixture has no bounded FCRAM attribute plane");
                    auto* values = reinterpret_cast<float*>(memory_plane.GetPtr());
                    for (u32 vertex = 0; vertex < count; ++vertex) {
                        const auto value = attribute == 0 ?
                            (vertex < count - 3 ? std::array<float, 4>{8, 8, -.5f, 1} : positions[vertex - (count - 3)]) :
                            attribute == 1 ? std::array<float, 4>{0, 0, 0, 1} :
                            attribute == 2 ? std::array<float, 4>{1, 0, 1, 1} : std::array<float, 4>{.25f, .75f, 0, 1};
                        std::copy(value.begin(), value.end(), values + std::size_t(vertex) * 4);
                    }
                }
                std::memset(target.GetPtr(), 0x11, 32 * 32 * 4); pica.dirty_regs.SetAllDirty();
                Check(rasterizer.AccelerateDrawBatch(false), "Large four-loader draw was not accepted natively");
                rasterizer.FlushRegion(Memory::VRAM_PADDR, 32 * 32 * 4);
                for (unsigned byte = 0; byte < 32 * 32 * 4; ++byte)
                    Check(target.GetPtr()[byte] == (byte % 4 == 2 ? 0 : 255),
                          "Large batch lost an immutable loader/uniform snapshot or the final triangle's VRAM pixels");
                std::cout << "196608-vertex actual native batch: four3MiB immutable FCRAM attribute planes, GPU fetch/VS/assembly, clipped preceding primitives and final triangle's1024 VRAM pixels passed\n";
            }
            Check(!system.IsPoweredOn(), "The data-only harness unexpectedly started emulation");
            rasterizer.ClearAll(false);
            std::cout << "Real RasterizerMetal immediate input -> GPU VS/stream/FS -> cache -> physical VRAM flush passed without starting emulation\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
