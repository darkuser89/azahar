// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_format_conversion.h"
#include "mtl_compiler_gate_fixture.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void WriteWord(std::uint8_t* bytes, std::uint32_t word) {
    for (unsigned i = 0; i < 4; ++i) bytes[i] = std::uint8_t(word >> (8 * i));
}
id<MTLTexture> DepthTexture(Metal::Device& device, NSUInteger width, NSUInteger height) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                       width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    return device.CreateTexture(descriptor);
}

void CheckDirectClears(Metal::Device& device, Metal::Scheduler& scheduler,
                       Metal::FormatConversionRuntime& conversions) {
    Metal::TransferRuntime transfer{device, scheduler};
    unsigned cases = 0;
    for (auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth16Unorm,
                        MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8}) {
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
            width:20 height:12 mipmapped:YES];
        descriptor.textureType = MTLTextureType2DArray;
        descriptor.arrayLength = 2;
        descriptor.mipmapLevelCount = 3;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
        auto texture = device.CreateTexture(descriptor);
        for (unsigned slice = 0; slice < 2; ++slice) for (unsigned level = 0; level < 3; ++level) {
            const auto width = 20U >> level, height = 12U >> level;
            const Metal::TextureRegion full{texture, level, slice, MTLRegionMake2D(0, 0, width, height)};
            conversions.ClearRegion(full, MTLClearColorMake(1, 0, 0, 1), 1.0, 0xA5);
        }
        for (unsigned slice = 0; slice < 2; ++slice) for (unsigned level = 0; level < 3; ++level) {
            const auto width = 20U >> level, height = 12U >> level;
            const Metal::TextureRegion part{texture, level, slice, MTLRegionMake2D(1, 1, width - 2, height - 2)};
            std::unique_ptr<Metal::Test::CompilerGate> gate;
            if (!slice && !level) {
                scheduler.Finish();
                gate = std::make_unique<Metal::Test::CompilerGate>(device);
            }
            const auto compiled = device.DiskCacheStats().compiled_pipelines;
            const auto queued = scheduler.WorkerStats().queued_draws;
            const auto before = scheduler.EncoderCount();
            conversions.ClearRegion(part, MTLClearColorMake(0, 1, 0, 1), 0.25, 0x3C);
            Check(scheduler.EncoderCount() == before + 1, "Partial clear split its native encoder");
            if (gate) {
                const bool no_compile = device.DiskCacheStats().compiled_pipelines == compiled;
                const bool color_queued = format != MTLPixelFormatRGBA8Unorm ||
                    scheduler.WorkerStats().queued_draws == queued + 1;
                const bool timely = gate->Open();
                Check(timely && no_compile && color_queued,
                      "First partial clear waited for the compiler or compiled a native PSO on its owner");
            }
            // Inspect every subresource, so a clear of the wrong mip or slice is visible.
            for (unsigned s = 0; s < 2; ++s) for (unsigned l = 0; l < 3; ++l) {
                const auto w = 20U >> l, h = 12U >> l;
                const bool changed = s < slice || (s == slice && l <= level);
                const Metal::TextureRegion full{texture, l, s, MTLRegionMake2D(0, 0, w, h)};
                const bool color = format == MTLPixelFormatRGBA8Unorm;
                auto read = device.CreateBuffer(w * h * 4, true);
                if (color) transfer.Download(full, read, {0, w * 4, 0, MTLBlitOptionNone});
                else conversions.DepthStencilToBuffer(full, read, 0, w * 4,
                    format == MTLPixelFormatDepth32Float_Stencil8 ? Metal::DepthStagingFormat::D24S8 : Metal::DepthStagingFormat::D32Float);
                scheduler.Finish();
                for (unsigned y = 0; y < h; ++y) for (unsigned x = 0; x < w; ++x) {
                    const bool inside = changed && x && x + 1 < w && y && y + 1 < h;
                    if (color) {
                        const auto* bytes = static_cast<const unsigned char*>(read.contents) + (y * w + x) * 4;
                        Check(bytes[0] == (inside ? 0 : 255) && bytes[1] == (inside ? 255 : 0) &&
                            bytes[2] == 0 && bytes[3] == 255, "Direct clear changed a neighboring color/subresource");
                    } else if (format == MTLPixelFormatDepth32Float_Stencil8) {
                        const auto word = static_cast<const unsigned*>(read.contents)[y * w + x];
                        Check(word == (inside ? 0x3FFFFF3CU : 0xFFFFFFA5U), "Direct clear changed depth/stencil or neighboring subresource");
                    } else {
                        const auto value = static_cast<const float*>(read.contents)[y * w + x];
                        Check(std::abs(value - (inside ? 0.25f : 1.f)) < 0.00002f, "Direct clear changed depth or neighboring subresource");
                    }
                }
            }
            ++cases;
        }
    }
    std::cout << cases << " direct partial clears: compiler-gated first use, RGBA8/D16/D32/D32S8, three mips/two slices, all neighbor subresources and one native encoder passed\n";
}

void CheckDisjointCopies(Metal::Device& device, Metal::Scheduler& scheduler) {
    Metal::TransferRuntime transfer{device,scheduler};
    for (const auto indices : {std::array<unsigned,4>{0,0,1,0}, {1,0,1,1}, {1,1,0,0}}) {
        const auto [sl,ss,dl,ds] = indices;
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:20 height:12 mipmapped:YES];
        descriptor.textureType = MTLTextureType2DArray;
        descriptor.arrayLength = 2;
        descriptor.mipmapLevelCount = 3;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
        auto texture = device.CreateTexture(descriptor);
        const auto pixel = [](unsigned l,unsigned s,unsigned x,unsigned y,unsigned c) {
            return std::uint8_t(l*31+s*73+x*(c+1)+y*17+c*43);
        };
        for (unsigned s=0;s<2;++s) for (unsigned l=0;l<3;++l) {
            const auto w=20U>>l,h=12U>>l;
            auto upload=device.CreateBuffer(w*h*4,true);
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)for(unsigned c=0;c<4;++c)
                static_cast<std::uint8_t*>(upload.contents)[(y*w+x)*4+c]=pixel(l,s,x,y,c);
            transfer.Upload(upload,{0,w*4,0,MTLBlitOptionNone},{texture,l,s,MTLRegionMake2D(0,0,w,h)});
        }
        const auto view = [&](NSUInteger level,NSUInteger slice) {
            auto* vd=[[MTLTextureViewDescriptor alloc] init];vd.pixelFormat=MTLPixelFormatRGBA8Unorm;
            vd.textureType=MTLTextureType2D;vd.levelRange=NSMakeRange(level,1);vd.sliceRange=NSMakeRange(slice,1);
            auto first=[texture newTextureViewWithDescriptor:vd];
            vd.levelRange=NSMakeRange(0,1);vd.sliceRange=NSMakeRange(0,1);
            auto second=[first newTextureViewWithDescriptor:vd];
            Check(first && second,"Nested subresource-copy fixture could not create views");
            return second;
        };
        const Metal::TextureRegion source{view(sl,ss),0,0,MTLRegionMake2D(1,1,4,2)},
            dest{view(dl,ds),0,0,MTLRegionMake2D(2,1,4,2)};
        Check(!Metal::TextureSubresourcesMayAlias(source,dest),"Disjoint nested mip/slice views were treated as overlapping");
        const auto before=scheduler.EncoderCount();
        transfer.CopyTexture(source,dest);
        Check(scheduler.EncoderCount()==before+1,"Disjoint subresource copy allocated an intermediate blit");
        for (unsigned s=0;s<2;++s) for (unsigned l=0;l<3;++l) {
            const auto w=20U>>l,h=12U>>l;
            const auto read=transfer.Readback({texture,l,s,MTLRegionMake2D(0,0,w,h)},w*4);
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)for(unsigned c=0;c<4;++c) {
                const bool copied=l==dl && s==ds && x>=2 && x<6 && y>=1 && y<3;
                const auto expected=copied ? pixel(sl,ss,x-1,y,c):pixel(l,s,x,y,c);
                Check(std::to_integer<unsigned>(read[(y*w+x)*4+c])==expected,
                    "Disjoint subresource copy changed pixels or neighboring mip/slice");
            }
        }
    }
    std::cout<<"3 disjoint nested-view copies: distinct mips/slices, nonuniform pixels, all neighboring subresources and one blit encoder passed\n";
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::FormatConversionRuntime conversions{device, scheduler};
            CheckDirectClears(device, scheduler, conversions);
            CheckDisjointCopies(device, scheduler);
            auto target = DepthTexture(device, 17, 11);
            transfers.ClearDepthStencil(target, 1.0, 0xA5, true, true);
            constexpr NSUInteger pitch = 57;
            auto source = device.CreateBuffer(1 + pitch * 9, true);
            auto* bytes = static_cast<std::uint8_t*>(source.contents);
            std::memset(bytes, 0xCD, source.length);
            std::array<std::uint32_t, 13 * 9> expected{};
            for (unsigned i = 0; i < expected.size(); ++i) {
                const auto depth = i == 0 ? 0U : i == expected.size() - 1 ? 0xFFFFFFU : (i * 147911U) & 0xFFFFFFU;
                expected[i] = (depth << 8) | ((i * 7) & 255U);
                WriteWord(bytes + 1 + (i / 13) * pitch + (i % 13) * 4, expected[i]);
            }
            auto private_source = device.CreateBuffer(source.length, false);
            transfers.CopyBuffer(source, 0, private_source, 0, source.length);
            const Metal::TextureRegion partial{target, 0, 0, MTLRegionMake2D(2, 1, 13, 9)};
            conversions.BufferToDepthStencil(private_source, 1, pitch, partial);
            auto packed = device.CreateBuffer(17 * 11 * 4, true);
            conversions.DepthStencilToBuffer({target, 0, 0, MTLRegionMake2D(0, 0, 17, 11)}, packed, 0, 17 * 4);
            scheduler.Finish();
            const auto* words = static_cast<const std::uint32_t*>(packed.contents);
            for (unsigned y = 0; y < 11; ++y) {
                for (unsigned x = 0; x < 17; ++x) {
                    const auto wanted = x >= 2 && x < 15 && y >= 1 && y < 10
                        ? expected[(y - 1) * 13 + x - 2] : 0xFFFFFFA5U;
                    Check(words[y * 17 + x] == wanted, "Packed PICA upload changed a depth word, stencil value or outside pixel");
                }
            }
            bool rejected{};
            try { conversions.BufferToDepthStencil(private_source, 1, 51, partial); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Packed upload accepted a short source row");
            rejected = false;
            try { conversions.BufferToDepthStencil(private_source, source.length, pitch, partial); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Packed upload accepted an out-of-range source buffer");
            auto scaled = DepthTexture(device, 23, 15);
            transfers.ClearDepthStencil(scaled, 1.0, 0xA5, true, true);
            const Metal::TextureRegion scaled_part{scaled, 0, 0, MTLRegionMake2D(3, 2, 17, 11)};
            conversions.BlitDepthStencil(partial, scaled_part, true, true);
            auto scaled_packed = device.CreateBuffer(23 * 15 * 4, true);
            conversions.DepthStencilToBuffer({scaled, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, scaled_packed, 0, 23 * 4);
            scheduler.Finish();
            const auto* scaled_words = static_cast<const std::uint32_t*>(scaled_packed.contents);
            for (unsigned y = 0; y < 15; ++y) {
                for (unsigned x = 0; x < 23; ++x) {
                    auto wanted = 0xFFFFFFA5U;
                    if (x >= 3 && x < 20 && y >= 2 && y < 13) {
                        const auto sx = 12 - ((2 * (x - 3) + 1) * 13) / (2 * 17);
                        const auto sy = 8 - ((2 * (y - 2) + 1) * 9) / (2 * 11);
                        wanted = expected[sy * 13 + sx];
                    }
                    Check(scaled_words[y * 23 + x] == wanted, "Scaled native depth/stencil blit lost a flip, stencil byte or outside pixel");
                }
            }
            conversions.BlitDepthStencil(partial, {target, 0, 0, MTLRegionMake2D(1, 1, 13, 9)}, true, false);
            conversions.DepthStencilToBuffer({target, 0, 0, MTLRegionMake2D(0, 0, 17, 11)}, packed, 0, 17 * 4);
            scheduler.Finish();
            for (unsigned y = 0; y < 11; ++y) {
                for (unsigned x = 0; x < 17; ++x) {
                    auto wanted = x >= 2 && x < 15 && y >= 1 && y < 10
                        ? expected[(y - 1) * 13 + x - 2] : 0xFFFFFFA5U;
                    if (x >= 1 && x < 14 && y >= 1 && y < 10) wanted = expected[(y - 1) * 13 + 13 - x];
                    Check(words[y * 17 + x] == wanted, "Overlapping native depth/stencil blit modified its source before sampling finished");
                }
            }
            auto* color_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                       width:13 height:9 mipmapped:NO];
            color_descriptor.storageMode = MTLStorageModePrivate;
            color_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
            auto color_source = device.CreateTexture(color_descriptor);
            auto color_data = device.CreateBuffer(13 * 9 * 4, true);
            auto* color_bytes = static_cast<std::uint8_t*>(color_data.contents);
            for (unsigned y = 0; y < 9; ++y) {
                for (unsigned x = 0; x < 13; ++x) {
                    for (unsigned c = 0; c < 4; ++c) {
                        color_bytes[(y * 13 + x) * 4 + c] = std::uint8_t(x * (17 + c) + y * (7 + c * 3) + c * 29);
                    }
                }
            }
            transfers.Upload(color_data, {0, 13 * 4, 0, MTLBlitOptionNone},
                             {color_source, 0, 0, MTLRegionMake2D(0, 0, 13, 9)});
            color_descriptor.width = 23;
            color_descriptor.height = 15;
            auto color_target = device.CreateTexture(color_descriptor);
            const Metal::TextureRegion color_part{color_source, 0, 0, MTLRegionMake2D(2, 1, 9, 7)};
            for (const bool linear : {false, true}) {
                transfers.ClearColor(color_target, MTLClearColorMake(17.0 / 255, 34.0 / 255, 51.0 / 255, 68.0 / 255));
                conversions.BlitColor(color_part, {color_target, 0, 0, MTLRegionMake2D(3, 2, 17, 11)}, linear, true, false);
                const auto pixels = transfers.Readback({color_target, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, 23 * 4);
                for (unsigned y = 0; y < 15; ++y) {
                    for (unsigned x = 0; x < 23; ++x) {
                        for (unsigned c = 0; c < 4; ++c) {
                            int wanted = 17 * (c + 1);
                            const bool inside = x >= 3 && x < 20 && y >= 2 && y < 13;
                            if (inside && !linear) {
                                const auto sx = 2 + 8 - ((2 * (x - 3) + 1) * 9) / (2 * 17);
                                const auto sy = 1 + ((2 * (y - 2) + 1) * 7) / (2 * 11);
                                wanted = color_bytes[(sy * 13 + sx) * 4 + c];
                            } else if (inside) {
                                const auto px = 2.0 + 9.0 - (double(x - 3) + 0.5) * 9.0 / 17.0 - 0.5;
                                const auto py = 1.0 + (double(y - 2) + 0.5) * 7.0 / 11.0 - 0.5;
                                const auto x0 = int(std::floor(px));
                                const auto y0 = int(std::floor(py));
                                const auto fx = px - x0;
                                const auto fy = py - y0;
                                const auto sample = [&](int sx, int sy) {
                                    return double(color_bytes[(std::clamp(sy, 0, 8) * 13 + std::clamp(sx, 0, 12)) * 4 + c]);
                                };
                                const auto top = sample(x0, y0) * (1 - fx) + sample(x0 + 1, y0) * fx;
                                const auto bottom = sample(x0, y0 + 1) * (1 - fx) + sample(x0 + 1, y0 + 1) * fx;
                                wanted = int(std::lround(top * (1 - fy) + bottom * fy));
                            }
                            const auto actual = std::to_integer<int>(pixels[(y * 23 + x) * 4 + c]);
                            Check(std::abs(actual - wanted) <= (inside && linear ? 1 : 0),
                                  "Native color blit changed pixel-center sampling, flipping, filtering or outside pixels");
                        }
                    }
                }
            }
            conversions.BlitColor(color_part, {color_source, 0, 0, MTLRegionMake2D(1, 1, 9, 7)}, false, true, true);
            const auto alias_pixels = transfers.Readback({color_source, 0, 0, MTLRegionMake2D(0, 0, 13, 9)}, 13 * 4);
            for (unsigned y = 0; y < 9; ++y) {
                for (unsigned x = 0; x < 13; ++x) {
                    const bool inside = x >= 1 && x < 10 && y >= 1 && y < 8;
                    const auto sx = inside ? 2 + 9 - x : x;
                    const auto sy = inside ? 1 + 7 - y : y;
                    for (unsigned c = 0; c < 4; ++c) {
                        Check(std::to_integer<unsigned>(alias_pixels[(y * 13 + x) * 4 + c]) == color_bytes[(sy * 13 + sx) * 4 + c],
                              "Overlapping native color blit overwrote source pixels during sampling");
                    }
                }
            }
            // Views must select the requested mip and array slice rather than mip zero/slice zero.
            color_descriptor.textureType = MTLTextureType2DArray;
            color_descriptor.arrayLength = 2;
            color_descriptor.mipmapLevelCount = 3;
            color_descriptor.width = 17;
            color_descriptor.height = 11;
            auto array_source = device.CreateTexture(color_descriptor);
            color_descriptor.width = 23;
            color_descriptor.height = 15;
            auto array_dest = device.CreateTexture(color_descriptor);
            transfers.ClearColor(array_source, MTLClearColorMake(17.0 / 255, 34.0 / 255, 51.0 / 255, 68.0 / 255), 1, 1);
            transfers.ClearColor(array_dest, MTLClearColorMake(1.0 / 255, 2.0 / 255, 3.0 / 255, 4.0 / 255), 1, 1);
            conversions.BlitColor({array_source, 1, 1, MTLRegionMake2D(1, 1, 6, 3)},
                                  {array_dest, 1, 1, MTLRegionMake2D(1, 1, 9, 5)}, true);
            const auto array_pixels = transfers.Readback({array_dest, 1, 1, MTLRegionMake2D(0, 0, 11, 7)}, 11 * 4);
            for (unsigned y = 0; y < 7; ++y) {
                for (unsigned x = 0; x < 11; ++x) {
                    for (unsigned c = 0; c < 4; ++c) {
                        const bool inside = x >= 1 && x < 10 && y >= 1 && y < 6;
                        Check(std::to_integer<unsigned>(array_pixels[(y * 11 + x) * 4 + c]) == (inside ? 17U : 1U) * (c + 1),
                              "Native blit did not select the requested array slice and mip level");
                    }
                }
            }
            color_descriptor.textureType = MTLTextureType2D;
            color_descriptor.arrayLength = 1;
            color_descriptor.mipmapLevelCount = 1;
            color_descriptor.width = 2;
            color_descriptor.height = 2;
            color_descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm_sRGB;
            color_descriptor.usage = MTLTextureUsageShaderRead;
            auto srgb_source = device.CreateTexture(color_descriptor);
            auto srgb_data = device.CreateBuffer(16, true);
            const std::array<unsigned, 4> srgb_values{128, 64, 192, 80};
            auto* srgb_bytes = static_cast<std::uint8_t*>(srgb_data.contents);
            for (unsigned i = 0; i < 16; ++i) srgb_bytes[i] = std::uint8_t(srgb_values[i % 4]);
            transfers.Upload(srgb_data, {0, 8, 0, MTLBlitOptionNone}, {srgb_source, 0, 0, MTLRegionMake2D(0, 0, 2, 2)});
            conversions.BlitColor({srgb_source, 0, 0, MTLRegionMake2D(0, 0, 2, 2)},
                                  {color_target, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, false);
            const auto srgb_pixels = transfers.Readback({color_target, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, 23 * 4);
            for (unsigned c = 0; c < 4; ++c) {
                const auto value = srgb_values[c] / 255.0;
                const auto linear = c == 3 ? value : value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
                const auto wanted = int(std::lround(linear * 255));
                for (std::size_t i = c; i < srgb_pixels.size(); i += 4) {
                    Check(std::abs(std::to_integer<int>(srgb_pixels[i]) - wanted) <= 1,
                          "Native color sampling did not decode sRGB or preserve linear alpha");
                }
            }
            if (device.Handle().supportsBCTextureCompression) {
                color_descriptor.pixelFormat = MTLPixelFormatBC1_RGBA;
                color_descriptor.width = 7;
                color_descriptor.height = 5;
                auto bc_source = device.CreateTexture(color_descriptor);
                auto bc_data = device.CreateBuffer(32, true);
                auto* bc_bytes = static_cast<std::uint8_t*>(bc_data.contents);
                std::memset(bc_bytes, 0, 32);
                for (unsigned block = 0; block < 4; ++block) {
                    bc_bytes[block * 8] = 0x00;
                    bc_bytes[block * 8 + 1] = 0xF8; // Opaque red RGB565 endpoint, all selectors zero.
                }
                transfers.Upload(bc_data, {0, 16, 0, MTLBlitOptionNone}, {bc_source, 0, 0, MTLRegionMake2D(0, 0, 7, 5)});
                conversions.BlitColor({bc_source, 0, 0, MTLRegionMake2D(0, 0, 7, 5)},
                                      {color_target, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, true);
                const auto bc_pixels = transfers.Readback({color_target, 0, 0, MTLRegionMake2D(0, 0, 23, 15)}, 23 * 4);
                for (std::size_t i = 0; i < bc_pixels.size(); ++i) {
                    Check(std::to_integer<unsigned>(bc_pixels[i]) == ((i % 4 == 0 || i % 4 == 3) ? 255U : 0U),
                          "Native compressed color sampling lost a block edge or channel");
                }
                std::cout << "Native BC1 sampling with partial edge blocks passed\n";
            }
            const bool exhaustive = argc == 2 && std::string_view(argv[1]) == "--exhaustive";
            const NSUInteger width = exhaustive ? 4096 : 256;
            const NSUInteger height = width;
            const auto count = width * height;
            auto sweep_source = device.CreateBuffer(count * 4, true);
            auto* sweep_words = static_cast<std::uint32_t*>(sweep_source.contents);
            for (NSUInteger i = 0; i < count; ++i) {
                const auto depth = exhaustive ? std::uint32_t(i) : std::uint32_t((i * 257U) & 0xFFFFFFU);
                sweep_words[i] = (depth << 8) | ((depth ^ (depth >> 8) ^ (depth >> 16)) & 255U);
            }
            auto sweep_texture = DepthTexture(device, width, height);
            const Metal::TextureRegion sweep{sweep_texture, 0, 0, MTLRegionMake2D(0, 0, width, height)};
            conversions.BufferToDepthStencil(sweep_source, 0, width * 4, sweep);
            auto sweep_output = device.CreateBuffer(count * 4, true);
            conversions.DepthStencilToBuffer(sweep, sweep_output, 0, width * 4);
            scheduler.Finish();
            const auto* sweep_result = static_cast<const std::uint32_t*>(sweep_output.contents);
            for (NSUInteger i = 0; i < count; ++i) {
                if (sweep_words[i] != sweep_result[i]) {
                    throw std::runtime_error("Packed PICA roundtrip mismatch at depth sample " + std::to_string(i));
                }
            }
            std::cout << "Native packed D24S8 upload and scaled/flipped/overlapping blits: private/unaligned input, odd pitch/extents, partial-region preservation, endpoints, bounds and "
                      << count << " depth/stencil roundtrip samples passed\n";
            std::cout << "Native nearest/linear color scaling, flips, partial-region preservation and overlapping copies passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
