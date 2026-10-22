// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_guest_codec.h"
#include "mtl_transfer.h"
#include "mtl_texture_runtime.h"
#include "common/color.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"
#include <bit>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
Metal::BufferSlice Slice(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger size) {
    return {buffer, offset, size, buffer.storageMode == MTLStorageModeShared ? static_cast<u8*>(buffer.contents) + offset : nullptr, {}};
}
u64 Pixel(const Metal::GuestTextureLayout& layout, u32 x, u32 y) {
    return layout.tiled ? ((u64(y / 8) * (layout.stride / 8) + x / 8) * 64 + VideoCore::MortonInterleave(x, y))
                        : u64(y) * layout.stride + x;
}
u32 Word(const u8* data, u32 bytes) { u32 word{}; for (u32 i = 0; i < bytes; ++i) word |= u32(data[i]) << (8 * i); return word; }
Common::Vec4<u8> Color(VideoCore::PixelFormat format, const u8* data, u64 pixel) {
    using namespace Common::Color;
    using F = VideoCore::PixelFormat;
    switch (format) {
    case F::RGBA8: return DecodeRGBA8(data);
    case F::RGB8: return DecodeRGB8(data);
    case F::RGB5A1: return DecodeRGB5A1(data);
    case F::RGB565: return DecodeRGB565(data);
    case F::RGBA4: return DecodeRGBA4(data);
    case F::IA8: return DecodeIA8(data);
    case F::RG8: return DecodeRG8(data);
    case F::I8: return DecodeI8(data);
    case F::A8: return DecodeA8(data);
    case F::IA4: return DecodeIA4(data);
    case F::I4: { const auto i = u8(((data[0] >> ((pixel % 2) * 4)) & 15) * 17); return {i, i, i, 255}; }
    case F::A4: return {0, 0, 0, u8(((data[0] >> ((pixel % 2) * 4)) & 15) * 17)};
    default: throw std::runtime_error("Unexpected CPU reference format");
    }
}

void EncodeColor16(unsigned format, const Common::Vec4<u8>& color, u8* bytes) {
    using namespace Common::Color;
    switch(format) {
    case 2: EncodeRGB5A1(color,bytes);break;
    case 3: EncodeRGB565(color,bytes);break;
    case 4: EncodeRGBA4(color,bytes);break;
    case 5: EncodeIA8(color,bytes);break;
    case 6: EncodeRG8(color,bytes);break;
    default: throw std::runtime_error("Unexpected 16-bit reference format");
    }
}

void CheckColor16Reinterpret(Metal::Device& device, Metal::Scheduler& scheduler,
                            Metal::TextureRuntime& runtime) {
    Metal::TransferRuntime transfers{device,scheduler};
    constexpr unsigned width=288,count=256*256;
    unsigned cases{};
    auto upload=device.CreateBuffer(width*width*4,true);
    auto* pixels=static_cast<u8*>(upload.contents);
    for(unsigned source_format=2;source_format<=6;++source_format) {
        VideoCore::SurfaceParams params{};
        params.width=params.height=params.stride=width*2;params.levels=2;
        params.pixel_format=VideoCore::PixelFormat(source_format);params.UpdateParams();
        Metal::Surface source{runtime,params};
        for(unsigned pattern=0;pattern<2;++pattern) {
            std::memset(pixels,0xD7,upload.length);
            std::vector<std::array<u8,2>> words(count);
            for(unsigned i=0;i<count;++i) {
                const std::array<u8,2> word{u8(i),u8(i>>8)};
                auto color=Color(params.pixel_format,word.data(),0);
                if(pattern) color={u8(i*73),u8((i>>8)*29),u8((i*43)>>7),u8(i*31+(i>>8))};
                EncodeColor16(source_format,color,words[i].data());
                std::memcpy(pixels+((i/256+13)*width+i%256+7)*4,color.AsArray(),4);
            }
            transfers.Upload(upload,{0,width*4,0,MTLBlitOptionNone},
                {source.Image(),1,0,MTLRegionMake2D(0,0,width,width)});
            for(unsigned dest_format=2;dest_format<=6;++dest_format) {
                auto dest_params=params;dest_params.pixel_format=VideoCore::PixelFormat(dest_format);dest_params.UpdateParams();
                Metal::Surface dest{runtime,dest_params};
                transfers.ClearColor(dest.Image(),MTLClearColorMake(0.25,0.5,0.75,1),0);
                transfers.ClearColor(dest.Image(),MTLClearColorMake(0,0,0,0),1);
                Check(runtime.Reinterpret(source,dest,{.src_level=1,.dst_level=1,
                    .src_offset={7,13},.dst_offset={11,5},.extent={256,256}}),
                    "Native 16-bit reinterpretation rejected a supported format pair");
                auto result=transfers.Readback({dest.Image(),1,0,MTLRegionMake2D(0,0,width,width)},width*4);
                for(unsigned y=0;y<width;++y)for(unsigned x=0;x<width;++x) {
                    Common::Vec4<u8> expected{};
                    if(x>=11 && x<267 && y>=5 && y<261)
                        expected=Color(dest_params.pixel_format,words[(y-5)*256+x-11].data(),0);
                    Check(!std::memcmp(result.data()+(y*width+x)*4,expected.AsArray(),4),
                        "Complete 16-bit reinterpretation space, independent encoding or outside pixel differs");
                }
                const auto untouched=transfers.Readback({dest.Image(),0,0,MTLRegionMake2D(0,0,2,2)},8);
                for(unsigned p=0;p<4;++p) {
                    const auto* bytes=reinterpret_cast<const u8*>(untouched.data())+p*4;
                    Check(bytes[0]==64 && bytes[1]==128 && bytes[2]==191 && bytes[3]==255,
                        "Native reinterpretation changed an unrequested mip");
                }
                ++cases;
            }
        }
    }
    // The existing public surface path also accepts cube subresources and
    // overlapping self-copies. Build the oracle from the pre-copy host image.
    for(unsigned format=2;format<=6;++format)for(unsigned cube=0;cube<2;++cube) {
        VideoCore::SurfaceParams params{};
        params.width=params.height=params.stride=64;params.levels=2;
        params.texture_type=cube ? VideoCore::TextureType::CubeMap:VideoCore::TextureType::Texture2D;
        params.pixel_format=VideoCore::PixelFormat(format);params.UpdateParams();
        Metal::Surface surface{runtime,params};
        const unsigned layer=cube ? 5:0;
        for(unsigned direction=0;direction<2;++direction) {
            auto input=device.CreateBuffer(32*32*4,true);
            auto* data=static_cast<u8*>(input.contents);
            for(unsigned i=0;i<input.length;++i)data[i]=u8(i*73+(i>>5)*31);
            transfers.Upload(input,{0,128,0,MTLBlitOptionNone},
                {surface.Image(),1,layer,MTLRegionMake2D(0,0,32,32)});
            std::vector<u8> expected(data,data+input.length);
            const unsigned sx=direction ? 7:1,sy=direction ? 9:2,
                           dx=direction ? 1:7,dy=direction ? 2:9;
            for(unsigned y=0;y<11;++y)for(unsigned x=0;x<13;++x) {
                Common::Vec4<u8> color{};std::memcpy(color.AsArray(),data+((sy+y)*32+sx+x)*4,4);
                std::array<u8,2> word{};EncodeColor16(format,color,word.data());
                color=Color(params.pixel_format,word.data(),0);
                std::memcpy(expected.data()+((dy+y)*32+dx+x)*4,color.AsArray(),4);
            }
            Check(runtime.Reinterpret(surface,surface,{.src_level=1,.dst_level=1,.src_layer=layer,.dst_layer=layer,
                .src_offset={sx,sy},.dst_offset={dx,dy},.extent={13,11}}),"Native alias reinterpretation rejected");
            const auto actual=transfers.Readback({surface.Image(),1,layer,MTLRegionMake2D(0,0,32,32)},128);
            Check(!std::memcmp(actual.data(),expected.data(),expected.size()),
                "Native reinterpretation lost the source snapshot for an overlapping cube/mip/2D region");
            ++cases;
        }
    }
    std::cout<<cases<<" native 16-bit reinterpretation cases: all65536 words for25 format pairs, arbitrary8-bit channels, nonzero offsets/mip1, outside pixels/mip preservation and overlapping2D/cube subresources\n";
}
}

int main() {
    @autoreleasepool {
        try {
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::GuestTextureCodec codec{device, scheduler};
            Metal::TextureRuntime checked_runtime{device,scheduler};
            CheckColor16Reinterpret(device,scheduler,checked_runtime);
            unsigned variants{};
            for (u32 format = 0; format < 18; ++format) {
                if (format == 15) continue;
                for (const bool tiled : {false, true}) {
                    if (!tiled && (format == 12 || format == 13)) continue;
                    for (const bool converted : {false, true}) for (const bool flip_output : {false, true}) {
                        const Metal::GuestTextureLayout layout{tiled ? 16U : 7U, tiled ? 16U : 5U,
                            tiled ? 24U : 9U, VideoCore::PixelFormat(format), tiled, converted, flip_output};
                        const auto guest_size = codec.GuestBytes(layout), linear_size = codec.LinearBytes(layout);
                        const auto bits = VideoCore::GetFormatBpp(layout.format), bpp = codec.LinearBytesPerPixel(layout);
                        auto raw = device.CreateBuffer(guest_size + 2, true);
                        auto* bytes = static_cast<u8*>(raw.contents) + 1;
                        for (NSUInteger i = 0; i < guest_size; ++i) bytes[i] = u8(i * 73 + (i >> 2) * 19 + format * 29);
                        auto private_raw = device.CreateBuffer(raw.length, false);
                        transfers.CopyBuffer(raw, 0, private_raw, 0, raw.length);
                        auto linear = device.CreateBuffer(linear_size + 2, true);
                        transfers.FillBuffer(linear, NSMakeRange(0, linear.length), 0xCD);
                        codec.Decode(layout, Slice(private_raw, 1, guest_size), Slice(linear, 1, linear_size));
                        scheduler.Finish();
                        const auto* actual = static_cast<const u8*>(linear.contents) + 1;
                        Check(static_cast<const u8*>(linear.contents)[0] == 0xCD && actual[linear_size] == 0xCD, "Guest decode overwrote guard bytes");
                        for (u32 y = 0; y < layout.height; ++y) for (u32 x = 0; x < layout.width; ++x) {
                            const auto sy = tiled != flip_output ? layout.height - y - 1 : y;
                            const auto pixel = Pixel(layout, x, sy);
                            const auto* source = bytes + pixel * bits / 8;
                            const auto* result = actual + (y * layout.width + x) * bpp;
                            if (format <= 13 && (format > 4 || converted)) {
                                Common::Vec4<u8> wanted;
                                if (format == 12 || format == 13) {
                                    Pica::Texture::TextureInfo info{};
                                    info.width = layout.width;
                                    info.height = layout.height;
                                    info.format = Pica::TexturingRegs::TextureFormat(format);
                                    info.stride = ptrdiff_t(layout.stride / 8) * Pica::Texture::CalculateTileSize(info.format);
                                    wanted = Pica::Texture::LookupTexture(bytes, x, sy, info);
                                } else wanted = Color(layout.format, source, pixel);
                                for (u32 c = 0; c < 4; ++c) Check(result[c] == wanted[c], "Guest decode differs from independent scalar PICA color/ETC reference");
                            } else if (format == 16 && converted) {
                                float value;
                                std::memcpy(&value, result, 4);
                                const auto d = Word(source, 3);
                                Check(u32(value * 16777215.f) == d, "Guest D24 float conversion did not preserve the 24-bit value");
                            } else {
                                auto word = Word(source, bits / 8);
                                if (format == 17) word = std::rotl(word, 8);
                                Check(Word(result, bpp) == word, "Guest raw/hash/depth staging conversion differs from PICA words");
                            }
                        }
                        if (format != 12 && format != 13) {
                            auto encoded = device.CreateBuffer(guest_size + 2, true);
                            for (const bool partial : {false, true}) {
                                transfers.FillBuffer(encoded, NSMakeRange(0, encoded.length), 0xA5);
                                const auto range = partial ? NSMakeRange(3, guest_size - 7) : NSMakeRange(0, guest_size);
                                codec.Encode(layout, Slice(linear, 1, linear_size), Slice(encoded, 1, guest_size), range);
                                scheduler.Finish();
                                std::vector<u8> wanted(guest_size + 2, 0xA5);
                                for (u32 y = 0; y < layout.height; ++y) for (u32 x = 0; x < layout.width; ++x) {
                                    const auto pixel = Pixel(layout, x, y);
                                    const auto offset = pixel * bits / 8;
                                    for (u32 b = 0; b < (bits == 4 ? 1U : bits / 8); ++b) {
                                        const auto byte = offset + b;
                                        if (byte < range.location || byte >= range.location + range.length) continue;
                                        if (bits == 4) {
                                            const auto mask = u8(15U << ((pixel % 2) * 4));
                                            wanted[byte + 1] = u8((wanted[byte + 1] & ~mask) | (bytes[byte] & mask));
                                        } else wanted[byte + 1] = bytes[byte];
                                    }
                                }
                                Check(std::memcmp(encoded.contents, wanted.data(), wanted.size()) == 0,
                                      "Guest encode lost stride padding, partial-byte interval, nibble ownership or guard bytes");
                            }
                        }
                        ++variants;
                    }
                }
            }
            // Exhaust the packed 16-bit color space for all three PICA render-target formats.
            for (auto format : {VideoCore::PixelFormat::RGB5A1, VideoCore::PixelFormat::RGB565, VideoCore::PixelFormat::RGBA4}) {
                const Metal::GuestTextureLayout layout{256, 256, 256, format, true, true};
                auto raw = device.CreateBuffer(65536 * 2, true);
                auto* words = static_cast<u16*>(raw.contents);
                for (u32 i = 0; i < 65536; ++i) words[i] = u16(i);
                auto linear = device.CreateBuffer(65536 * 4, true);
                codec.Decode(layout, Slice(raw, 0, raw.length), Slice(linear, 0, linear.length));
                auto back = device.CreateBuffer(raw.length, true);
                codec.Encode(layout, Slice(linear, 0, linear.length), Slice(back, 0, back.length), NSMakeRange(0, back.length));
                scheduler.Finish();
                Check(std::memcmp(raw.contents, back.contents, raw.length) == 0, "Packed 16-bit color space did not survive GPU decode/encode");
            }
            Metal::TextureRuntime runtime{device, scheduler};
            for (u32 format = 0; format < 18; ++format) {
                if (format == 15) continue;
                VideoCore::SurfaceParams params;
                params.addr = 0x1000;
                params.width = params.height = 16;
                params.stride = 24;
                params.res_scale = 2;
                params.pixel_format = VideoCore::PixelFormat(format);
                params.is_tiled = true;
                params.UpdateParams();
                Metal::Surface surface{runtime, params};
                std::vector<u8> raw(params.size);
                for (u32 i = 0; i < raw.size(); ++i) raw[i] = u8(i * 47 + format * 13);
                runtime.UploadGuest(surface, params, raw);
                auto from_surface = runtime.FindStaging(16 * 16 * surface.GetInternalBytesPerPixel(), false);
                surface.Download({from_surface.offset, from_surface.size, params.GetRect(), 0}, from_surface);
                const auto decoded = runtime.DecodeGuest(params, raw, true);
                Check(std::equal(decoded.begin(), decoded.end(), from_surface.mapped.begin()),
                      "Native guest upload -> scaled surface -> staging differs from the GPU codec");
                const auto hash_pixels = runtime.DecodeGuest(params, raw, false);
                Check(hash_pixels.size() == 16 * 16 * codec.LinearBytesPerPixel({16, 16, 24, params.pixel_format, true, false}),
                      "Native legacy hash readback has the wrong byte layout");
                if (format != 12 && format != 13) {
                    std::vector<u8> full(raw.size(), 0xA5);
                    const auto full_tick = scheduler.CurrentTick();
                    runtime.DownloadGuest(surface, params, full, NSMakeRange(0, full.size()));
                    Check(scheduler.CurrentTick() == full_tick + 1,
                          "Native guest flush split GPU readback and encoding across submissions");
                    std::vector<u8> partial(raw.size(), 0xA5);
                    const auto partial_tick = scheduler.CurrentTick();
                    runtime.DownloadGuest(surface, params, partial, NSMakeRange(3, partial.size() - 7));
                    Check(scheduler.CurrentTick() == partial_tick + 1,
                          "Native partial guest flush introduced an intermediate GPU wait");
                    for (u32 i = 0; i < partial.size(); ++i)
                        Check(partial[i] == (i >= 3 && i < partial.size() - 4 ? full[i] : 0xA5),
                              "Native guest flush changed bytes outside its physical interval");
                }
                if (format == 0) {
                    bool dumped{};
                    runtime.SetTextureDumpSink([&](const VideoCore::SurfaceParams& info, u32 level, u64 hash, std::vector<u8> pixels) {
                        Check(info.width == 16 && level == 2 && hash == 0x1234 && pixels.size() == decoded.size(),
                              "Native dump sink lost texture metadata");
                        for (u32 y = 0; y < 16; ++y) for (u32 x = 0; x < 16 * 4; ++x)
                            Check(pixels[y * 16 * 4 + x] == decoded[(15 - y) * 16 * 4 + x],
                                  "Native PNG-oriented decode was not flipped on the GPU");
                        dumped = true;
                    });
                    runtime.DumpGuest(params, 2, 0x1234, raw);
                    Check(dumped, "Native dump did not reach its installed pixel sink");
                    runtime.SetTextureDumpSink({});
                }
            }
            unsigned reinterprets{};
            for (u32 src_format = 0; src_format < 18; ++src_format) {
                if (src_format == 12 || src_format == 13 || src_format == 15) continue;
                VideoCore::SurfaceParams src_params;
                src_params.width = src_params.height = src_params.stride = 16;
                src_params.pixel_format = VideoCore::PixelFormat(src_format);
                src_params.is_tiled = true;
                src_params.UpdateParams();
                Metal::Surface source{runtime, src_params};
                std::vector<u8> raw(src_params.size);
                for (u32 i = 0; i < raw.size(); ++i) raw[i] = u8(i * 61 + src_format * 17);
                runtime.UploadGuest(source, src_params, raw);
                for (u32 dst_format = 0; dst_format < 18; ++dst_format) {
                    if (dst_format == 12 || dst_format == 13 || dst_format == 15) continue;
                    const auto dst = VideoCore::PixelFormat(dst_format);
                    if (VideoCore::GetFormatBpp(dst) != src_params.GetFormatBpp()) continue;
                    auto dst_params = src_params;
                    dst_params.pixel_format = dst;
                    dst_params.UpdateParams();
                    Metal::Surface dest{runtime, dst_params};
                    Check(runtime.Reinterpret(source, dest, {.extent = {16, 16}}), "Native guest-bit reinterpretation rejected an equal-width format pair");
                    auto staging = runtime.FindStaging(16 * 16 * dest.GetInternalBytesPerPixel(), false);
                    dest.Download({staging.offset, staging.size, dest.GetRect(), 0}, staging);
                    const auto wanted = runtime.DecodeGuest(dst_params, raw, true);
                    Check(std::equal(wanted.begin(), wanted.end(), staging.mapped.begin()),
                          "Native guest-bit reinterpretation changed color/depth/stencil bit semantics");
                    ++reinterprets;
                }
            }
            bool rejected{};
            for (const bool tiled : {false, true}) {
                VideoCore::SurfaceParams wide;
                wide.width = wide.stride = 16;
                wide.height = 16;
                wide.pixel_format = VideoCore::PixelFormat::RGBA8;
                wide.is_tiled = tiled;
                wide.UpdateParams();
                Metal::Surface source{runtime, wide};
                std::vector<u8> bytes(wide.size);
                for (u32 i = 0; i < bytes.size(); ++i) bytes[i] = u8(i * 31);
                runtime.UploadGuest(source, wide, bytes);
                VideoCore::TextureClear clear{};
                clear.texture_rect = {4, 12, 12, 4};
                clear.value.color = {1, 0, 1, 1};
                runtime.ClearTexture(source, clear);
                runtime.DownloadGuest(source, wide, bytes, NSMakeRange(0, bytes.size()));
                auto narrow = wide;
                narrow.width = narrow.stride = 8;
                narrow.height = 32;
                narrow.UpdateParams();
                Metal::Surface dest{runtime, narrow};
                runtime.UploadGuest(dest, narrow, bytes);
                const auto wanted = runtime.DecodeGuest(narrow, bytes, true);
                auto staging = runtime.FindStaging(u32(wanted.size()), false);
                dest.Download({staging.offset, staging.size, narrow.GetRect(), 0}, staging);
                Check(std::equal(wanted.begin(), wanted.end(), staging.mapped.begin()),
                      "GPU-written pixels were lost while reshaping the same physical bytes to a different guest stride");
            }
            for (u32 size = 2; size <= 4; ++size) for (u32 phase = 0; phase < size; ++phase) {
                auto fill = device.CreateBuffer(131, true);
                transfers.FillBuffer(fill, NSMakeRange(0, fill.length), 0xCD);
                const std::array<u8, 4> pattern{0x17, 0x39, 0xA5, 0xE1};
                codec.Fill(Slice(fill, 1, 129), pattern, size, phase);
                scheduler.Finish();
                const auto* data = static_cast<const u8*>(fill.contents);
                Check(data[0] == 0xCD && data[130] == 0xCD, "Native guest fill overwrote guard bytes");
                for (u32 i = 0; i < 129; ++i) Check(data[i + 1] == pattern[(i + phase) % size], "Native guest fill lost its pattern phase");
                std::array<u8, 35> target{};
                runtime.FillGuest(target, pattern, size, phase);
                for (u32 i = 0; i < target.size(); ++i) Check(target[i] == pattern[(i + phase) % size], "Native cache fill readback lost its pattern");
            }
            unsigned copies{};
            for (unsigned count : {1U, 63U, 64U, 65U, 257U})
                for (unsigned input_width : {1U, 16U, 33U})
                    for (unsigned output_width : {1U, 17U, 64U})
                        for (bool gaps : {false, true}) {
                            const Metal::GuestCopyLayout copy{count, input_width, gaps ? 7U : 0U, output_width, gaps ? 11U : 0U};
                            const auto input_size = codec.CopyBytes(count, copy.input_width, copy.input_gap);
                            const auto output_size = codec.CopyBytes(count, copy.output_width, copy.output_gap);
                            auto source = device.CreateBuffer(input_size + 2, true);
                            auto dest = device.CreateBuffer(output_size + 2, true);
                            for (NSUInteger byte = 0; byte < source.length; ++byte)
                                static_cast<u8*>(source.contents)[byte] = u8(byte * 53 + (byte / 9) * 7);
                            auto private_source = device.CreateBuffer(source.length, false);
                            transfers.CopyBuffer(source, 0, private_source, 0, source.length);
                            std::memset(dest.contents, 0xcd, dest.length);
                            codec.Copy(copy, Slice(private_source, 1, input_size), Slice(dest, 1, output_size));
                            scheduler.Finish();
                            std::vector<u8> expected(output_size + 2, 0xcd);
                            for (unsigned byte = 0; byte < count; ++byte) {
                                const auto src = byte / input_width * (input_width + copy.input_gap) + byte % input_width;
                                const auto dst = byte / output_width * (output_width + copy.output_gap) + byte % output_width;
                                expected[dst + 1] = static_cast<const u8*>(source.contents)[src + 1];
                            }
                            Check(std::equal(expected.begin(), expected.end(), static_cast<const u8*>(dest.contents)),
                                  "Native raw GPU copy lost partial-row bytes, gaps or unaligned guards");
                            ++copies;
                        }
            {
                std::vector<id<MTLBuffer>> buffers;
                std::vector<std::vector<u8>> expected;
                for (unsigned i = 0; i < 32; ++i) {
                    const Metal::GuestCopyLayout copy{257 + i, 33, 7, 17, 11};
                    const auto input_size = codec.CopyBytes(copy.byte_count, copy.input_width, copy.input_gap);
                    const auto output_size = codec.CopyBytes(copy.byte_count, copy.output_width, copy.output_gap);
                    auto storage = device.CreateBuffer(8192, true);
                    auto* bytes = static_cast<u8*>(storage.contents);
                    for (unsigned byte = 0; byte < storage.length; ++byte)
                        bytes[byte] = u8(byte * 53 + byte / 9 + i * 17);
                    expected.emplace_back(bytes, bytes + storage.length);
                    const NSUInteger source = i % 3 == 0 ? 4097 : 37;
                    const NSUInteger dest = i % 3 == 0 ? 37 :
                        i % 3 == 1 ? 4097 : source + input_size;
                    for (unsigned byte = 0; byte < copy.byte_count; ++byte) {
                        const auto src = source + byte / copy.input_width * (copy.input_width + copy.input_gap) + byte % copy.input_width;
                        const auto dst = dest + byte / copy.output_width * (copy.output_width + copy.output_gap) + byte % copy.output_width;
                        expected.back()[dst] = bytes[src];
                    }
                    codec.Copy(copy, Slice(storage, source, input_size), Slice(storage, dest, output_size));
                    buffers.push_back(storage);
                }
                scheduler.Finish();
                for (unsigned i = 0; i < buffers.size(); ++i)
                    Check(std::equal(expected[i].begin(), expected[i].end(), static_cast<const u8*>(buffers[i].contents)),
                          "Queued same-buffer guest copy changed disjoint source, gaps, adjacent ranges or guards");
                std::cout << "32 queued disjoint same-buffer raw GPU copies: nonzero/unaligned offsets, forward/backward/adjacent ranges and 262144 full buffer bytes passed\n";
            }
            rejected = false;
            try { codec.CopyBytes(16, 0, 1); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Raw GPU copy accepted zero line width");
            rejected = false;
            try { codec.CopyBytes(std::uint64_t(UINT32_MAX) + 1, 16, 0); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Raw GPU copy overflowed its thread-index representation");
            auto copy_buffer = device.CreateBuffer(32, true);
            rejected = false;
            try { codec.Copy({17,16,16,16,0}, Slice(copy_buffer,0,32), Slice(copy_buffer,0,32)); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Raw GPU copy accepted an out-of-bounds gapped source");
            rejected = false;
            try { codec.Copy({16,16,0,16,0}, Slice(copy_buffer,0,16), Slice(copy_buffer,8,16)); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Raw GPU copy accepted overlapping GPU slices without a snapshot");
            codec.Copy({}, {}, {});
            std::cout << copies << " raw GPU copy variants with mismatched widths, partial rows, gaps, private unaligned source, guards and bounds passed\n";
            rejected = false;
            try { codec.GuestBytes({8, 8, 7, VideoCore::PixelFormat::RGBA8, true}); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Guest codec accepted a shorter-than-width stride");
            const std::array invalid_layouts{
                Metal::GuestTextureLayout{0, 8, 8, VideoCore::PixelFormat::RGBA8, false},
                Metal::GuestTextureLayout{8, 0, 8, VideoCore::PixelFormat::RGBA8, false},
                Metal::GuestTextureLayout{7, 8, 8, VideoCore::PixelFormat::RGBA8, true},
                Metal::GuestTextureLayout{8, 7, 8, VideoCore::PixelFormat::RGBA8, true},
                Metal::GuestTextureLayout{8, 8, 9, VideoCore::PixelFormat::RGBA8, true},
                Metal::GuestTextureLayout{65536, 65536, 65536, VideoCore::PixelFormat::RGBA8, false},
                Metal::GuestTextureLayout{8, 8, 8, VideoCore::PixelFormat(15), true},
                Metal::GuestTextureLayout{8, 8, 8, VideoCore::PixelFormat(18), true},
                Metal::GuestTextureLayout{8, 8, 8, VideoCore::PixelFormat::ETC1, false},
                Metal::GuestTextureLayout{8, 8, 8, VideoCore::PixelFormat::ETC1A4, false},
            };
            for (const auto& invalid : invalid_layouts) {
                rejected = false;
                try { codec.GuestBytes(invalid); }
                catch (const std::invalid_argument&) { rejected = true; }
                Check(rejected, "Guest codec accepted an invalid format, dimension or tile layout");
            }
            const auto wide_stride = UINT32_MAX - 7u;
            Check(codec.GuestBytes({8, 16, wide_stride, VideoCore::PixelFormat::RGBA8, true}) ==
                      (std::uint64_t(wide_stride) * 8u + 64u) * 4u,
                  "Guest codec truncated a valid 64-bit tiled byte extent");
            Check(codec.CopyBytes(UINT32_MAX, 1, UINT32_MAX) ==
                      std::uint64_t(UINT32_MAX) * UINT32_MAX,
                  "Guest codec truncated a valid 64-bit gapped copy extent");
            std::cout << "Native codec preflight: 11 independent invalid layouts and two 64-bit byte extents passed\n";
            const Metal::GuestTextureLayout layout{8, 8, 8, VideoCore::PixelFormat::RGBA8, true};
            auto short_buffer = device.CreateBuffer(255, true), valid_buffer = device.CreateBuffer(256, true);
            rejected = false;
            try { codec.Decode(layout, Slice(short_buffer, 0, 255), Slice(valid_buffer, 0, 256)); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Guest codec accepted an out-of-bounds source read");
            rejected = false;
            try { codec.Decode(layout, Slice(valid_buffer, 0, 256), Slice(valid_buffer, 0, 256)); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Guest codec accepted overlapping decode buffers");
            std::cout << variants << " guest format/layout/conversion/orientation variants, scalar ETC/PICA reference, private unaligned input, strides, partial byte/nibble ranges, guards, bounds and all 65536 RGB5A1/RGB565/RGBA4 words passed\n";
            std::cout << "Native guest codec -> scaled cache surfaces -> guest flush, legacy hash readback and GPU-oriented dump pixels passed\n";
            std::cout << reinterprets << " equal-bit-width guest color/depth/stencil reinterpretation pairs passed\n";
            std::cout << "Native 16/24/32-bit guest fill patterns, arbitrary phase, tails and guard bytes passed\n";
            std::cout << "GPU-written physical bytes reshaped from 16x16 to 8x32 in linear and tiled layouts passed\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
