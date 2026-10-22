// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_vertex_fetch.h"
#include "mtl_transfer.h"
#include "mtl_shaders.h"
#include <algorithm>
#include <array>
#include <vector>
#include "video_core/pica/regs_internal.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void ValidateArgumentSnapshots(Metal::Device& device, Metal::Scheduler& scheduler,
                               Metal::TransferRuntime& transfers, Metal::VertexFetchRuntime& fetch) {
    std::array<Metal::VertexInputState, 2> states{Metal::VertexInputState{device}, Metal::VertexInputState{device}};
    std::array<Metal::VertexRegisters, 2> attribute_history{}, register_history{};
    unsigned cases = 0, snapshots = 0;
    for (unsigned round = 0; round < 4; ++round) {
        constexpr unsigned draws = 24;
        auto readback = device.CreateBuffer(draws * 65 * sizeof(Metal::VertexRegisters) + 32, true);
        std::memset(readback.contents, 0xA7, readback.length);
        std::vector<Metal::VertexRegisters> expected;
        for (unsigned draw = 0; draw < draws; ++draw) {
            const unsigned bank = draw & 1, count = draw % 3 == 0 ? 1 : draw % 3 == 1 ? 4 : 12;
            const unsigned vertices = draw & 1 ? 65 : 3;
            const bool pica = draw % 3 != 0;
            if (pica && (draw == 7 || draw == 8)) {
                states[bank].Reset(); attribute_history[bank] = {}; register_history[bank] = {};
            }
            auto input = device.CreateBuffer(count * (vertices * 16 + 7), true);
            std::vector<Metal::BufferSlice> inputs;
            std::vector<Metal::VertexAttribute> descriptors;
            Metal::PicaVertexLayout layout{};
            layout.mapped_attribute_count = draw % 17;
            for (unsigned a = 0; a < 16; ++a) layout.register_map[a] = (a * 3 + draw) % 16;
            layout.register_map[3] = layout.register_map[0]; // Later mapped attributes overwrite aliases.
            const auto value = [&](unsigned loader, unsigned vertex) {
                return std::array<float, 4>{float(round * 1000 + draw * 20 + loader),
                    float(vertex) + .5f, -float(loader), float(draw) + .25f};
            };
            for (unsigned loader = 0; loader < count; ++loader) {
                const NSUInteger offset = loader * (vertices * 16 + 7) + 3;
                for (unsigned v = 0; v < vertices; ++v) {
                    const auto bytes = value(loader, v);
                    std::memcpy(static_cast<std::byte*>(input.contents) + offset + v * 16, bytes.data(), 16);
                }
                inputs.push_back({input, offset, vertices * 16, nullptr, {}});
                descriptors.push_back({loader, 0, 16, 4, Metal::VertexAttributeFormat::Float, loader});
                layout.attribute_inputs[loader] = {descriptors.back(), Metal::PicaAttributeMode::Buffer};
            }
            const std::array<float, 4> fixed_value{float(round), float(draw), -9, 1};
            auto fixed = Metal::DefaultVertexRegisters(); fixed[13] = fixed_value;
            layout.attribute_inputs[13].mode = Metal::PicaAttributeMode::Fixed;
            layout.fixed_attributes[13] = fixed_value;
            const unsigned width = draw % 4 == 0 ? 0 : draw % 4 == 1 ? 1 : draw % 4 == 2 ? 2 : 4;
            auto indices = device.CreateBuffer(16, true);
            for (unsigned i = 0; i < 3; ++i) for (unsigned b = 0; b < width; ++b)
                static_cast<u8*>(indices.contents)[1 + i * width + b] = u8((i == 1 ? 0 : vertices - 1) >> (b * 8));
            const auto old_size = expected.size();
            for (unsigned v = 0; v < vertices; ++v) {
                auto attributes = attribute_history[bank]; attributes[13] = fixed_value;
                for (unsigned loader = 0; loader < count; ++loader) attributes[loader] = value(loader, v);
                auto registers = pica ? register_history[bank] : fixed;
                if (pica) for (unsigned a = 0; a < layout.mapped_attribute_count; ++a)
                    registers[layout.register_map[a]] = attributes[a];
                else for (unsigned loader = 0; loader < count; ++loader) registers[loader] = value(loader, v);
                expected.push_back(registers);
            }
            const auto result = pica ? fetch.FetchPica(states[bank], inputs, layout, vertices,
                width ? 3 : vertices, 0, 0, {indices, 1, width * 3, nullptr, {}}, width)
                : fetch.Fetch(inputs, descriptors, vertices, 0, fixed);
            transfers.CopyBuffer(result.buffer, result.offset, readback,
                                 16 + old_size * sizeof(Metal::VertexRegisters), result.size);
            if (pica) {
                const unsigned last = width ? 0 : vertices - 1;
                attribute_history[bank][13] = fixed_value;
                for (unsigned loader = 0; loader < count; ++loader) attribute_history[bank][loader] = value(loader, last);
                register_history[bank] = expected[old_size + last];
            }
            if (draw % 8 == 7) scheduler.Submit(); // Rebind while prior submissions are still in flight.
            ++cases;
        }
        scheduler.Finish();
        const auto expected_bytes = expected.size() * sizeof(Metal::VertexRegisters);
        Check(!std::memcmp(static_cast<const std::byte*>(readback.contents) + 16, expected.data(), expected_bytes),
              "Rebound Fetch/PICA/history tables changed queued register snapshots");
        for (unsigned b = 0; b < 16; ++b) Check(static_cast<const u8*>(readback.contents)[b] == 0xA7 &&
            static_cast<const u8*>(readback.contents)[16 + expected_bytes + b] == 0xA7,
            "Rebound fetch output overwrote a readback canary");
        for (unsigned bank = 0; bank < 2; ++bank) {
            const auto snapshot = fetch.SaveState(states[bank]);
            Check(!snapshot.needs_reset &&
                !std::memcmp(snapshot.history.data(), &attribute_history[bank], sizeof(Metal::VertexRegisters)) &&
                !std::memcmp(snapshot.history.data() + sizeof(Metal::VertexRegisters),
                             &register_history[bank], sizeof(Metal::VertexRegisters)),
                "Rebound history table changed complete attribute/register banks");
            ++snapshots;
        }
    }
    Check(cases == 96 && snapshots == 8, "Argument-table snapshot coverage was omitted");
    std::cout << cases << " mixed Fetch/PICA argument snapshots and " << snapshots
              << " full512 history snapshots passed across 12 GPU submissions\n";
}


void ValidateFetchUploadRecycling(Metal::Device& device, Metal::Scheduler& scheduler,
                                 Metal::TransferRuntime& transfers) {
    // Cross all three 1 MiB parameter pages without explicit submission. In
    // particular, the small history upload can be the allocation that wraps.
    Metal::VertexFetchRuntime fetch{device, scheduler};
    std::array<Metal::VertexInputState, 2> states{Metal::VertexInputState{device}, Metal::VertexInputState{device}};
    std::array<Metal::VertexRegisters, 2> attributes{}, registers{};
    constexpr unsigned draws = 4096;
    auto readback = device.CreateBuffer(draws * sizeof(Metal::VertexRegisters) + 32, true);
    std::memset(readback.contents, 0xA7, readback.length);
    std::vector<Metal::VertexRegisters> expected;
    expected.reserve(draws);
    const auto initial_tick = scheduler.CurrentTick();
    for (unsigned draw = 0; draw < draws; ++draw) {
        const unsigned bank = draw & 1;
        if (draw % 257 == 0) {
            states[bank].Reset(); attributes[bank] = {}; registers[bank] = {};
        }
        Metal::PicaVertexLayout layout{};
        layout.mapped_attribute_count = 2;
        for (unsigned a = 0; a < 16; ++a) layout.register_map[a] = a;
        layout.register_map[0] = (draw / 16) % 16;
        layout.register_map[1] = (draw / 31) % 16; // Includes later-attribute aliases.
        layout.attribute_inputs[0].mode = Metal::PicaAttributeMode::Fixed;
        layout.attribute_inputs[1].mode = draw % 3 == 0 ? Metal::PicaAttributeMode::Fixed
                                                      : Metal::PicaAttributeMode::Retained;
        const std::array<std::uint32_t, 4> raw{0x7FC00000U | (draw + 1), 0x80000000U,
                                             draw + 1, 0x7F800000U};
        std::memcpy(layout.fixed_attributes[0].data(), raw.data(), sizeof(raw));
        layout.fixed_attributes[1] = {float(draw), -float(draw), .25f, 1};
        attributes[bank][0] = layout.fixed_attributes[0];
        if (layout.attribute_inputs[1].mode == Metal::PicaAttributeMode::Fixed)
            attributes[bank][1] = layout.fixed_attributes[1];
        registers[bank][layout.register_map[0]] = attributes[bank][0];
        registers[bank][layout.register_map[1]] = attributes[bank][1];
        expected.push_back(registers[bank]);
        const auto result = fetch.FetchPica(states[bank], {}, layout, 1, 1, 0, 0);
        transfers.CopyBuffer(result.buffer, result.offset, readback,
                             16 + NSUInteger(draw) * sizeof(Metal::VertexRegisters), result.size);
    }
    Check(scheduler.CurrentTick() > initial_tick,
          "Fetch recycling regression did not cross the automatic submit boundary");
    scheduler.Finish();
    Check(!std::memcmp(static_cast<const std::byte*>(readback.contents) + 16,
                       expected.data(), expected.size() * sizeof(Metal::VertexRegisters)),
          "Fetch/history lost queued snapshots while upload pages recycled");
    for (unsigned b = 0; b < 16; ++b)
        Check(static_cast<const u8*>(readback.contents)[b] == 0xA7 &&
              static_cast<const u8*>(readback.contents)[readback.length - 16 + b] == 0xA7,
              "Fetch upload recycling overwrote a guard");
    for (unsigned bank = 0; bank < states.size(); ++bank) {
        const auto snapshot = fetch.SaveState(states[bank]);
        Check(!snapshot.needs_reset &&
              !std::memcmp(snapshot.history.data(), &attributes[bank], sizeof(Metal::VertexRegisters)) &&
              !std::memcmp(snapshot.history.data() + sizeof(Metal::VertexRegisters),
                           &registers[bank], sizeof(Metal::VertexRegisters)),
              "Fetch upload recycling changed a complete512 history bank");
    }
    std::cout << draws << " Fetch/history snapshots preserved raw bits, aliases and both512 banks across upload-page recycling\n";
}

unsigned ValidateInputHistoryBoundaries(Metal::Device& device,Metal::Scheduler& scheduler,
                                       id<MTLComputePipelineState> pipeline) {
    unsigned cases=0;
    for(unsigned count:{1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,
                        255u,256u,257u,511u,512u,513u,1023u,1024u,1025u,4095u,4096u,4097u})
    for(unsigned index_size:{0u,1u,2u,4u})
    for(unsigned reset:{0u,1u})
    for(unsigned emit_plan:{0u,1u})
    for(unsigned pattern=0;pattern<12;++pattern){@autoreleasepool{
        if(emit_plan && !index_size) continue;
        const unsigned vertices=index_size ? index_size==1 ? 192u:512u:count+7;
        const unsigned base=!index_size ? 10000u:!(pattern&1) ? 0u:
            index_size==4 ? UINT32_MAX-vertices+1u:index_size==2 ? 10000u:16u;
        const unsigned first=index_size ? base:base+3,invalid=base ? base-1:vertices;
        auto registers=device.CreateBuffer(NSUInteger(vertices)*256,true);
        auto attributes=device.CreateBuffer(NSUInteger(vertices)*256,true);
        auto* raw_registers=static_cast<unsigned*>(registers.contents);
        auto* raw_attributes=static_cast<unsigned*>(attributes.contents);
        constexpr std::array<unsigned,8> special{0u,0x80000000u,1u,0x7fc12345u,
            0xff812345u,0x7f800000u,0xff800000u,0x3f800000u};
        for(unsigned row=0;row<vertices;++row)for(unsigned word=0;word<64;++word){
            raw_registers[NSUInteger(row)*64+word]=special[word%8]^(row*8191u);
            raw_attributes[NSUInteger(row)*64+word]=special[(word+3)%8]^(row*4099u+0x100u);
        }
        std::vector<unsigned> ids(count);unsigned random=0x12345678u;
        for(unsigned e=0;e<count;++e){
            random=random*1664525u+1013904223u;
            unsigned id=base+(pattern==1 ? e%48:pattern==2 || pattern==9 ? (random>>16)%90:e%vertices);
            if(pattern==3 || (pattern==4 && !e) || (pattern==5 && e+1==count) ||
               (pattern==6 && e%32==31) || (pattern==7 && e>=64 && !(e&1)) ||
               (pattern==9 && (e&1)) || (pattern==11 && e+1!=count))id=invalid;
            else if(pattern==7 && e>=64)id=base+e%32;
            else if(pattern==8 && e>=70)id=base+(e&1 ? 0u:69u);
            else if(pattern==10 && e>=64)id=base;
            else if(pattern==11)id=base+vertices-1;
            ids[e]=id;
        }
        auto indices=device.CreateBuffer(16+NSUInteger(count)*index_size,true);
        std::memset(indices.contents,0xad,indices.length);
        for(unsigned e=0;e<count;++e)for(unsigned b=0;b<index_size;++b)
            static_cast<std::uint8_t*>(indices.contents)[3+NSUInteger(e)*index_size+b]=std::uint8_t(ids[e]>>(8*b));
        bool found=false;unsigned last=0,replacement=0,size=0;
        std::array<unsigned,64> fifo{};
        if(!index_size){last=first+count-1-base;found=true;}
        else for(unsigned id:ids){
            if(id<base || id-base>=vertices)continue;
            bool cached=false;for(unsigned slot=0;slot<size;++slot)cached|=fifo[slot]==id;
            if(cached)continue;
            fifo[replacement]=id;replacement=(replacement+1)&63u;size=std::min(size+1,64u);
            last=id-base;found=true;
        }
        auto history=device.CreateBuffer(544,true);std::memset(history.contents,0xa7,history.length);
        auto* history_words=reinterpret_cast<unsigned*>(static_cast<std::byte*>(history.contents)+16);
        std::array<unsigned,128> expected{};
        for(unsigned word=0;word<128;++word)history_words[word]=special[word%8]^(word*919u);
        if(found){std::memcpy(expected.data(),raw_attributes+NSUInteger(last)*64,256);
            std::memcpy(expected.data()+64,raw_registers+NSUInteger(last)*64,256);}
        else if(!reset)std::memcpy(expected.data(),history_words,512);
        // The shader plan stops at its first invalid ID. The independent
        // history oracle above continues through valid suffix misses.
        constexpr unsigned plan_canary=0xcdcdcdcdu,status_canary=0xabcdef01u;
        std::vector<unsigned> expected_plan((NSUInteger(count)+1)*2,plan_canary);
        std::array<unsigned,4> expected_status{};
        if(!emit_plan) expected_status.fill(status_canary);
        if(emit_plan) {
            std::array<std::pair<unsigned,unsigned>,64> entries{};
            unsigned used=0,next=0,previous=~0u;
            for(unsigned e=0;e<count;++e) {
                const auto id=ids[e];
                if(id<base || id-base>=vertices) {expected_status={8,e,0,0};break;}
                unsigned cached=~0u;
                for(unsigned slot=0;slot<used;++slot)
                    if(entries[slot].first==id) cached=entries[slot].second;
                expected_plan[NSUInteger(e)*2]=cached;
                expected_plan[NSUInteger(e)*2+1]=previous;
                if(cached!=~0u) continue;
                previous=e;entries[next]={id,e};next=(next+1)&63u;used=std::min(used+1,64u);
            }
            expected_plan[NSUInteger(count)*2]=expected_status[0] ? ~0u:previous;
            expected_plan[NSUInteger(count)*2+1]=~0u;
        }
        auto plan=device.CreateBuffer(32+expected_plan.size()*4,true);
        auto status=device.CreateBuffer(48,true);
        std::memset(plan.contents,0xcd,plan.length);std::memset(status.contents,0xa7,status.length);
        auto* plan_words=reinterpret_cast<unsigned*>(static_cast<std::byte*>(plan.contents)+16);
        auto* status_words=reinterpret_cast<unsigned*>(static_cast<std::byte*>(status.contents)+16);
        std::fill_n(status_words,4,emit_plan ? 0u:status_canary);
        const std::array<unsigned,8> config{count,vertices,base,first,index_size,reset,emit_plan,0};
        auto params=device.CreateBuffer(sizeof(config),true);std::memcpy(params.contents,config.data(),sizeof(config));
        const std::array<Metal::BufferBinding,7> bindings{{{registers,0,16},{params,0,17},
            {history,16,18},{attributes,0,19},{indices,3,20},{plan,16,21},{status,16,22}}};
        auto arguments=scheduler.MakeArguments(bindings);scheduler.KeepAlive(pipeline);
        auto encoder=scheduler.BeginCompute(pipeline);[encoder setComputePipelineState:pipeline];[encoder setArgumentTable:arguments];
        const auto width=pipeline.threadExecutionWidth;
        [encoder dispatchThreads:MTLSizeMake(width,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        scheduler.EndEncoding();scheduler.Finish();
        Check(!std::memcmp(history_words,expected.data(),512),
              "Input-history FIFO changed complete512 bank bits, skip-invalid order or last actual miss");
        for(unsigned b=0;b<16;++b)Check(static_cast<const std::uint8_t*>(history.contents)[b]==0xa7 &&
            static_cast<const std::uint8_t*>(history.contents)[528+b]==0xa7,"Input-history guard overwritten");
        Check(!std::memcmp(plan_words,expected_plan.data(),expected_plan.size()*4) &&
              !std::memcmp(status_words,expected_status.data(),16),
              "Shared history/FIFO plan changed a cached source, previous/last miss, first error or untouched suffix");
        for(unsigned byte=0;byte<16;++byte) {
            Check(static_cast<const std::uint8_t*>(plan.contents)[byte]==0xcd &&
                  static_cast<const std::uint8_t*>(plan.contents)[plan.length-16+byte]==0xcd &&
                  static_cast<const std::uint8_t*>(status.contents)[byte]==0xa7 &&
                  static_cast<const std::uint8_t*>(status.contents)[status.length-16+byte]==0xa7,
                  "Shared history/FIFO plan damaged plan/status guards");
        }
        ++cases;
    }}
    std::cout<<cases<<" native input-history/shared-plan boundary cases preserve all512 raw bank bytes, every plan entry, last miss, first error and untouched suffix, skips, FIFO hits/eviction, reset/inheritance, all index widths, UINT_MAX, partial chunks and guards\n";
    return cases;
}

}

int main() {
    @autoreleasepool {
        try {
            Metal::Device device;
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::VertexFetchRuntime fetch{device, scheduler};
            ValidateArgumentSnapshots(device, scheduler, transfers, fetch);
            ValidateFetchUploadRecycling(device, scheduler, transfers);
            const auto history_library=device.CompileLibrary(Metal::VertexFetchShaderSource,"Native input-history boundary regression");
            auto* history_descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
            history_descriptor.computeFunctionDescriptor=Metal::Device::Function(history_library,"update_pica_input_history");
            history_descriptor.maxTotalThreadsPerThreadgroup=64;
            const auto history_pipeline=device.CompileCompute(history_descriptor);
            Check(history_pipeline.threadExecutionWidth>=16 && history_pipeline.threadExecutionWidth<=64,
                  "Unsupported input-history SIMD width");
            Check(ValidateInputHistoryBoundaries(device,scheduler,history_pipeline)==4200,
                  "Input-history boundary cases omitted");
            constexpr unsigned source_stride = 160;
            auto raw = device.CreateBuffer(1 + 4 * source_stride, true);
            std::memset(raw.contents, 0, raw.length);
            auto* bytes = static_cast<std::uint8_t*>(raw.contents) + 1;
            std::array<Metal::VertexAttribute, 16> attrs{};
            std::array<Metal::VertexRegisters, 4> expected{};
            unsigned offset = 1;
            for (unsigned format = 0; format < 4; ++format) {
                const unsigned size = format == 3 ? 4 : format == 2 ? 2 : 1;
                for (unsigned components = 1; components <= 4; ++components) {
                    const auto reg = format * 4 + components - 1;
                    attrs[reg] = {0, offset, source_stride, components,
                                  static_cast<Metal::VertexAttributeFormat>(format), reg};
                    for (unsigned v = 0; v < 4; ++v) {
                        expected[v][reg] = {0, 0, 0, 1};
                        for (unsigned c = 0; c < components; ++c) {
                            const int integer = format == 0 ? -128 + int(v * 17 + c * 7)
                                              : format == 1 ? 255 - int(v * 19 + c * 11)
                                              : -32768 + int(v * 1207 + c * 809);
                            const float value = format == 3 ? (float(v * 7 + c) - 4.5f) * 0.125f : float(integer);
                            expected[v][reg][c] = value;
                            std::uint32_t word = std::uint32_t(integer);
                            if (format == 3) std::memcpy(&word, &value, 4);
                            for (unsigned b = 0; b < size; ++b) {
                                bytes[v * source_stride + offset + c * size + b] = std::uint8_t(word >> (b * 8));
                            }
                        }
                    }
                    offset += components * size + 1; // Deliberately unaligned SHORT/FLOAT fields.
                }
            }
            auto private_input = device.CreateBuffer(raw.length, false);
            transfers.CopyBuffer(raw, 0, private_input, 0, raw.length);
            const std::array<Metal::BufferSlice, 1> inputs{{{private_input, 1, raw.length - 1, nullptr, {}}}};
            auto decoded = fetch.Fetch(inputs, attrs, 2, 1);
            auto cpu = device.CreateBuffer(decoded.size, true);
            transfers.CopyBuffer(decoded.buffer, decoded.offset, cpu, 0, decoded.size);
            scheduler.Finish();
            const auto* result = static_cast<const Metal::VertexRegisters*>(cpu.contents);
            for (unsigned v = 0; v < 2; ++v) {
                for (unsigned reg = 0; reg < 16; ++reg) {
                    for (unsigned c = 0; c < 4; ++c) {
                        Check(result[v][reg][c] == expected[v + 1][reg][c],
                              "GPU vertex fetch format, component fill, offset, stride or first vertex mismatch");
                    }
                }
            }
            auto fixed = Metal::DefaultVertexRegisters();
            fixed[15] = {3, -4, 5, -6};
            auto fixed_output = fetch.Fetch({}, {}, 1, 0, fixed);
            transfers.CopyBuffer(fixed_output.buffer, fixed_output.offset, cpu, 0, fixed_output.size);
            scheduler.Finish();
            const auto* fixed_result = static_cast<const Metal::VertexRegisters*>(cpu.contents);
            Check(*fixed_result == fixed, "Native fixed/default attribute register values changed");
            auto constant_attr = attrs[0];
            constant_attr.stride = 0;
            auto overwritten = attrs[1];
            overwritten.register_index = 0;
            const std::array aliases{overwritten, constant_attr}; // Last descriptor wins at register zero.
            auto aliased = fetch.Fetch(inputs, aliases, 1, 3, fixed);
            transfers.CopyBuffer(aliased.buffer, aliased.offset, cpu, 0, aliased.size);
            scheduler.Finish();
            Check((*static_cast<const Metal::VertexRegisters*>(cpu.contents))[0] == expected[0][0],
                  "Zero-stride native attribute fetch used a per-vertex address");
            bool rejected{};
            try { fetch.Fetch(inputs, attrs, 5); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native vertex fetch accepted source reads outside the allocation");
            Pica::RegsInternal regs{};
            auto& config = regs.pipeline.vertex_attributes;
            config.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::BYTE);
            config.size0.Assign(2);
            config.format1.Assign(Pica::PipelineRegs::VertexAttributeFormat::SHORT);
            config.size1.Assign(1);
            config.format2.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            config.size2.Assign(2);
            config.format3.Assign(Pica::PipelineRegs::VertexAttributeFormat::UBYTE);
            config.size3.Assign(0);
            config.attribute_mask.Assign(1U << 3);
            config.max_attribute_index.Assign(15);
            regs.vs.max_input_attribute_index.Assign(15);
            regs.vs.input_attribute_to_register_map_low = 0x76042215;
            regs.vs.input_attribute_to_register_map_high = 0xFEDCBA98;
            auto& loader0 = config.attribute_loaders[0];
            loader0.data_offset.Assign(32);
            loader0.component_count.Assign(5);
            loader0.byte_count.Assign(40);
            loader0.comp0.Assign(0);
            loader0.comp1.Assign(13); // Three bytes, align to four, then eight bytes of padding.
            loader0.comp2.Assign(1);
            loader0.comp3.Assign(2);
            loader0.comp4.Assign(3);
            auto& loader1 = config.attribute_loaders[1];
            loader1.data_offset.Assign(256);
            loader1.component_count.Assign(1);
            loader1.byte_count.Assign(0); // Last loader overrides attribute zero with a constant record.
            loader1.comp0.Assign(0);
            fixed[3] = {-1, -2, -3, -4};
            auto retained = Metal::DefaultVertexRegisters();
            retained[5] = {9, 8, 7, 6};
            const auto layout = Metal::BuildPicaVertexLayout(regs, fixed, retained);
            Check(layout.attributes.size() == 3 && layout.loader_offsets[0] == 32 &&
                      layout.loader_offsets[1] == 256, "PICA loader/register mapping mismatch");
            const auto lengths = Metal::RequiredVertexInputBytes(layout, 2, 1);
            Check(lengths[0] == 109 && lengths[1] == 3, "PICA padding or bounded upload length mismatch");
            auto layout_raw = device.CreateBuffer(320, true);
            std::memset(layout_raw.contents, 0, layout_raw.length);
            auto* layout_bytes = static_cast<std::uint8_t*>(layout_raw.contents);
            layout_bytes[256] = 128;
            layout_bytes[257] = 2;
            layout_bytes[258] = 3;
            for (unsigned v = 0; v < 3; ++v) {
                const std::array<std::int16_t, 2> pair{std::int16_t(-200 + int(v)), std::int16_t(1234 + v)};
                std::memcpy(layout_bytes + 32 + v * 40 + 12, pair.data(), sizeof(pair));
                layout_bytes[32 + v * 40 + 28] = std::uint8_t(20 + v);
            }
            auto layout_private = device.CreateBuffer(layout_raw.length, false);
            transfers.CopyBuffer(layout_raw, 0, layout_private, 0, layout_raw.length);
            const std::array<Metal::BufferSlice, 2> layout_inputs{{
                {layout_private, layout.loader_offsets[0], lengths[0], nullptr, {}},
                {layout_private, layout.loader_offsets[1], lengths[1], nullptr, {}}}};
            auto from_regs = fetch.Fetch(layout_inputs, layout.attributes, 2, 1, layout.fixed);
            transfers.CopyBuffer(from_regs.buffer, from_regs.offset, cpu, 0, from_regs.size);
            scheduler.Finish();
            const auto* mapped = static_cast<const Metal::VertexRegisters*>(cpu.contents);
            for (unsigned v = 0; v < 2; ++v) {
                auto wanted = Metal::DefaultVertexRegisters();
                wanted[0] = retained[5];
                wanted[1] = {float(-199 + int(v)), float(1235 + v), 0, 1};
                wanted[2] = {float(21+v),0,0,1}; // Configured array overrides fixed attr 3.
                wanted[5] = {-128, 2, 3, 1};
                for (unsigned reg = 12; reg < 16; ++reg) wanted[reg] = fixed[reg];
                Check(mapped[v] == wanted, "GPU fetch derived from PICA registers changed padding, aliases or retained inputs");
            }
            regs.vs.max_input_attribute_index.Assign(2);
            const auto limited = Metal::BuildPicaVertexLayout(regs, fixed, retained);
            Check(limited.attributes.size() == 3 && Metal::RequiredVertexInputBytes(limited, 2, 1)[0] == 108,
                  "Native fetch ignored the PICA shader input attribute limit");
            // A later absent attribute must override an earlier dynamic alias with its retained value.
            regs.vs.max_input_attribute_index.Assign(5);
            regs.vs.input_attribute_to_register_map_low = 0x76542215;
            const auto absent_alias = Metal::BuildPicaVertexLayout(regs, fixed, retained);
            Check(absent_alias.attributes.size() == 2 && absent_alias.fixed[5] == retained[5],
                  "Missing PICA attribute did not preserve its retained value when registers alias");
            loader0.component_count.Assign(13);
            rejected = false;
            try { Metal::BuildPicaVertexLayout(regs, fixed); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native layout accepted an overflowing PICA component list");
            rejected = false;
            try { Metal::RequiredVertexInputBytes(layout, 2, UINT32_MAX); }
            catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected, "Native layout accepted an overflowing vertex index range");
            // Separate allocations also cover geometry/fetch batches larger than a stream page.
            Metal::StreamBuffer small_stream{device, scheduler, 64};
            auto large = small_stream.Allocate(1024);
            std::memset(large.data, 0x5A, large.size);
            auto next_large = small_stream.Allocate(1024);
            std::memset(next_large.data, 0xA5, next_large.size);
            transfers.CopyBuffer(large.buffer, large.offset, cpu, 0, 256);
            scheduler.Finish();
            const auto* large_bytes = static_cast<const std::uint8_t*>(cpu.contents);
            for (unsigned i = 0; i < 256; ++i) Check(large_bytes[i] == 0x5A, "Oversized stream storage was prematurely reused");
            const auto read = [&](const Metal::BufferSlice& values) {
                auto copy = device.CreateBuffer(values.size, true);
                transfers.CopyBuffer(values.buffer, values.offset, copy, 0, values.size);
                scheduler.Finish();
                std::vector<Metal::VertexRegisters> result(values.size / sizeof(Metal::VertexRegisters));
                std::memcpy(result.data(), copy.contents, values.size);
                return result;
            };
            unsigned preserved_state_cases{};
            const auto same_state = [&](const Metal::VertexInputState& state, const Metal::VertexInputSnapshot& before) {
                const auto after = fetch.SaveState(state);
                Check(after.history == before.history && after.needs_reset == before.needs_reset,
                      "Rejected/empty native vertex input changed its complete GPU bank snapshot");
                ++preserved_state_cases;
            };
            Pica::RegsInternal history_regs{};
            auto& history_config = history_regs.pipeline.vertex_attributes;
            history_config.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            history_config.size0.Assign(3);
            auto& history_loader = history_config.attribute_loaders[0];
            history_loader.component_count.Assign(1); history_loader.byte_count.Assign(16);
            std::array<std::array<float,4>,65> records{};
            for (unsigned index=0; index<records.size(); ++index) records[index]={float(index)+.5f,-float(index),float(index*2),1};
            auto host_records=device.CreateBuffer(sizeof(records)+1,true);
            std::memcpy(static_cast<u8*>(host_records.contents)+1,records.data(),sizeof(records));
            auto private_records=device.CreateBuffer(host_records.length,false);
            transfers.CopyBuffer(host_records,0,private_records,0,host_records.length);
            const std::array<Metal::BufferSlice,1> history_inputs{{{private_records,1,sizeof(records),nullptr,{}}}};
            auto history_layout=Metal::BuildPicaVertexLayout(history_regs,{});
            history_loader.component_count.Assign(0);
            const auto retain_layout=Metal::BuildPicaVertexLayout(history_regs,{});
            Metal::VertexInputState history_state{device};
            fetch.FetchPica(history_state,history_inputs,history_layout,65,65,100,100);
            const auto kept=fetch.FetchPica(history_state,{},retain_layout,1,1,0,0);
            Check(read(kept)[0][0]==records[64],"GPU input bank did not retain the last sequential vertex across encoders");
            scheduler.Wait(scheduler.Submit());
            Check(read(fetch.FetchPica(history_state,{},retain_layout,1,1,0,0))[0][0]==records[64],
                  "Private GPU input state was lost across queue submissions");
            const auto saved_input=fetch.SaveState(history_state);
            history_state.Reset();
            Check(read(fetch.FetchPica(history_state,{},retain_layout,1,1,0,0))[0]==Metal::VertexRegisters{},
                  "Input snapshot test did not perturb the live GPU bank");
            fetch.LoadState(history_state,saved_input);
            Check(read(fetch.FetchPica(history_state,{},retain_layout,1,1,0,0))[0][0]==records[64],
                  "Input snapshot failed to restore the private GPU bank before the next dispatch");
            Metal::VertexInputState restored_input{device};
            fetch.LoadState(restored_input,saved_input);
            const auto roundtrip=fetch.SaveState(restored_input);
            Check(roundtrip.history==saved_input.history && roundtrip.needs_reset==saved_input.needs_reset,
                  "Recreated GPU input state did not preserve snapshot bytes");
            auto corrupt_input=saved_input; corrupt_input.needs_reset=true;
            bool rejected_snapshot=false;
            try { fetch.LoadState(restored_input,corrupt_input); }
            catch (const std::invalid_argument&) { rejected_snapshot=true; }
            Check(rejected_snapshot,"Reset input snapshot accepted noncanonical retained bytes");
            same_state(restored_input, saved_input);
            for (unsigned width : {1U,2U,4U}) {
                for (bool eviction : {false,true}) {
                    Metal::VertexInputState indexed_state{device};
                    std::vector<unsigned> order;
                    if (eviction) {
                        for (unsigned index=0; index<64; ++index) order.push_back(index);
                        order.insert(order.end(),{0,64,0,0,62});
                    } else order={2,0,2};
                    auto indices=device.CreateBuffer(order.size()*width+1,true);
                    auto* data=static_cast<u8*>(indices.contents)+1;
                    for (unsigned element=0; element<order.size(); ++element)
                        for (unsigned byte=0; byte<width; ++byte) data[element*width+byte]=u8((order[element]+100)>>(byte*8));
                    auto private_indices=device.CreateBuffer(indices.length,false);
                    transfers.CopyBuffer(indices,0,private_indices,0,indices.length);
                    fetch.FetchPica(indexed_state,history_inputs,history_layout,65,u32(order.size()),100,0,
                                   {private_indices,1,indices.length-1,nullptr,{}},width);
                    const auto value=read(fetch.FetchPica(indexed_state,{},retain_layout,1,1,0,0));
                    Check(value[0][0]==records[0],"Indexed GPU input history used final/highest capture instead of last 64-entry-cache miss");
                }
            }
            // Adversarial collision chains must retain the same last FIFO miss,
            // including backward-shift wraparound and the valid UINT_MAX ID.
            constexpr unsigned collision_count=16384;
            auto collision_records=device.CreateBuffer(collision_count*16,true);
            for(unsigned i=0;i<collision_count;++i)
                static_cast<std::array<float,4>*>(collision_records.contents)[i]={float(i),float(i)+.5f,0,1};
            const std::array<Metal::BufferSlice,1> collision_inputs{{
                {collision_records,0,collision_records.length,collision_records.contents,{}}}};
            for(unsigned test=0;test<3;++test) {
                const unsigned base=test==2 ? UINT32_MAX-collision_count+1u:0u,width=test==2 ? 4u:2u;
                const unsigned bucket=test==0 ? 0:test==1 ? 127:((UINT32_MAX*0x9e3779b9u)>>25);
                std::vector<unsigned> candidates,order;
                for(unsigned i=0;i<collision_count && candidates.size()<90;++i)
                    if(((u32(base+i)*0x9e3779b9u)>>25)==bucket) candidates.push_back(base+i);
                Check(candidates.size()==90,"Input collision fixture has too few IDs");
                for(unsigned i=0;i<320;++i) order.push_back(candidates[i<70 ? i:(i*7)%90]);
                if(test==2) order.insert(order.end(),{UINT32_MAX,UINT32_MAX,candidates[0]});
                auto indices=device.CreateBuffer(order.size()*width,true);
                std::array<unsigned,64> fifo{};unsigned size=0,replacement=0,last=0;
                for(unsigned i=0;i<order.size();++i) {
                    for(unsigned byte=0;byte<width;++byte) static_cast<u8*>(indices.contents)[i*width+byte]=u8(order[i]>>(byte*8));
                    bool hit=false;for(unsigned slot=0;slot<size;++slot) hit|=fifo[slot]==order[i];
                    if(hit) continue;
                    fifo[replacement]=order[i];replacement=(replacement+1u)&63u;size=std::min(size+1u,64u);last=order[i]-base;
                }
                Metal::VertexInputState indexed_state{device};
                fetch.FetchPica(indexed_state,collision_inputs,history_layout,collision_count,unsigned(order.size()),base,base,
                    {indices,0,indices.length,indices.contents,{}},width);
                const auto retained=read(fetch.FetchPica(indexed_state,{},retain_layout,1,1,0,0));
                Check(retained[0][0]==static_cast<std::array<float,4>*>(collision_records.contents)[last],
                    "GPU collision/wraparound/UINT_MAX lookup changed the last FIFO-miss input bank");
            }
            std::cout<<"Three adversarial input-history collision/wraparound/UINT_MAX FIFO cases passed\n";
            // Load attribute 2 even when VS maps only 0/1, then expose the retained bank.
            Pica::RegsInternal bank_regs{};
            auto& bank_config=bank_regs.pipeline.vertex_attributes;
            bank_config.max_attribute_index.Assign(2);
            bank_config.format0.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT); bank_config.size0.Assign(3);
            bank_config.format1.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT); bank_config.size1.Assign(3);
            bank_config.format2.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT); bank_config.size2.Assign(3);
            auto& bank_loader=bank_config.attribute_loaders[0];
            bank_loader.component_count.Assign(3); bank_loader.byte_count.Assign(48);
            bank_loader.comp0.Assign(0); bank_loader.comp1.Assign(1); bank_loader.comp2.Assign(2);
            bank_regs.vs.max_input_attribute_index.Assign(1); bank_regs.vs.input_attribute_to_register_map_low=0x210;
            std::array<std::array<std::array<float,4>,3>,2> bank_records{{
                {{{1,2,3,4},{5,6,7,8},{9,10,11,12}}}, {{{13,14,15,16},{17,18,19,20},{21,22,23,24}}}}};
            auto bank_buffer=device.CreateBuffer(sizeof(bank_records),true);
            std::memcpy(bank_buffer.contents,bank_records.data(),sizeof(bank_records));
            const std::array<Metal::BufferSlice,1> bank_inputs{{{bank_buffer,0,bank_buffer.length,bank_buffer.contents,{}}}};
            Metal::VertexInputState bank_state{device};
            const auto bank_layout=Metal::BuildPicaVertexLayout(bank_regs,{});
            Check(Metal::RequiredPicaStateInputBytes(bank_layout,2)[0]==96,
                  "Stateful upload planning omitted attributes beyond the current shader mapping count");
            fetch.FetchPica(bank_state,bank_inputs,bank_layout,2,2,0,0);
            bank_loader.component_count.Assign(0);
            bank_regs.vs.max_input_attribute_index.Assign(2);
            const auto bank_values=read(fetch.FetchPica(bank_state,{},Metal::BuildPicaVertexLayout(bank_regs,{}),1,1,0,0));
            for (unsigned attribute=0; attribute<3; ++attribute)
                Check(bank_values[0][attribute]==bank_records[1][attribute],"Previously unmapped GPU attribute was not retained");
            bank_regs.vs.max_input_attribute_index.Assign(1); bank_regs.vs.input_attribute_to_register_map_low=0x55;
            bank_loader.component_count.Assign(1); bank_config.attribute_mask.Assign(1U<<1);
            Metal::VertexRegisters fixed_alias{}; fixed_alias[1]={.125f,.25f,.5f,1};
            const auto fixed_alias_values=read(fetch.FetchPica(bank_state,bank_inputs,
                Metal::BuildPicaVertexLayout(bank_regs,fixed_alias),1,1,0,0));
            Check(fixed_alias_values[0][5]==fixed_alias[1],"A later fixed attribute did not overwrite an earlier dynamic register alias");
            bank_config.attribute_mask.Assign(0); bank_loader.component_count.Assign(3);
            bank_regs.vs.max_input_attribute_index.Assign(2); bank_regs.vs.input_attribute_to_register_map_low=0x210;
            fetch.FetchPica(bank_state,bank_inputs,Metal::BuildPicaVertexLayout(bank_regs,{}),2,2,0,0);
            // A retained later alias wins, but the earlier dynamic attribute still updates its bank.
            bank_regs.vs.max_input_attribute_index.Assign(1); bank_regs.vs.input_attribute_to_register_map_low=0x55;
            bank_loader.component_count.Assign(1);
            fetch.FetchPica(bank_state,bank_inputs,Metal::BuildPicaVertexLayout(bank_regs,{}),1,1,0,0);
            bank_loader.component_count.Assign(0); bank_regs.vs.max_input_attribute_index.Assign(0);
            bank_regs.vs.input_attribute_to_register_map_low=0;
            const auto alias_bank=read(fetch.FetchPica(bank_state,{},Metal::BuildPicaVertexLayout(bank_regs,{}),1,1,0,0));
            Check(alias_bank[0][0]==bank_records[0][0] && alias_bank[0][5]==bank_records[1][1],
                  "GPU alias handling lost raw attribute state or an unmapped register");
            // Immediate attributes ignore array loaders and share the same GPU banks.
            bank_regs.pipeline.max_input_attrib_index.Assign(0);
            Metal::VertexRegisters submitted{}; submitted[0]={.125f,.25f,.5f,1};
            fetch.FetchPica(bank_state,{},Metal::BuildPicaImmediateLayout(bank_regs,submitted),1,1,0,0);
            const auto after_immediate=read(fetch.FetchPica(bank_state,{},Metal::BuildPicaVertexLayout(bank_regs,{}),1,1,0,0));
            Check(after_immediate[0][0]==submitted[0] && after_immediate[0][5]==bank_records[1][1],
                  "Immediate-to-batch GPU history or unmapped registers changed");
            bank_state.Reset();
            const auto reset_values=read(fetch.FetchPica(bank_state,{},Metal::BuildPicaVertexLayout(bank_regs,{}),1,1,0,0));
            Check(reset_values[0]==Metal::VertexRegisters{},"GPU input reset did not clear both private banks");
            Metal::VertexInputState isolated{device};
            Check(read(fetch.FetchPica(isolated,{},retain_layout,1,1,0,0))[0]==Metal::VertexRegisters{},
                  "Separate input-state objects shared GPU history");
            rejected=false;
            const auto before_rejections=fetch.SaveState(history_state);
            try { fetch.FetchPica(history_state,history_inputs,history_layout,66,66,0,0); }
            catch (const std::invalid_argument&) { rejected=true; }
            Check(rejected,"Stateful GPU fetch accepted out-of-bounds input");
            same_state(history_state,before_rejections);
            auto invalid_map=retain_layout; invalid_map.register_map[0]=16;
            rejected=false;
            try { fetch.FetchPica(history_state,{},invalid_map,1,1,0,0); }
            catch (const std::invalid_argument&) { rejected=true; }
            Check(rejected,"Stateful GPU fetch accepted an invalid register alias");
            same_state(history_state,before_rejections);
            Check(!fetch.FetchPica(history_state,{},retain_layout,1,0,0,0).buffer,
                  "Empty stateful draw allocated or changed GPU input state");
            same_state(history_state,before_rejections);
            Metal::VertexInputState invalid_index_state{device};
            auto invalid_indices=device.CreateBuffer(3,true);
            std::memset(invalid_indices.contents,255,3);
            fetch.FetchPica(invalid_index_state,history_inputs,history_layout,1,3,0,0,
                           {invalid_indices,0,3,invalid_indices.contents,{}},1);
            Check(read(fetch.FetchPica(invalid_index_state,{},retain_layout,1,1,0,0))[0]==Metal::VertexRegisters{},
                  "A reset input state with no valid indices exposed uninitialized/private history");
            auto bad_mode=retain_layout; bad_mode.attribute_inputs[0].mode=Metal::PicaAttributeMode(3);
            rejected=false;
            try { fetch.FetchPica(history_state,{},bad_mode,1,1,0,0); }
            catch (const std::invalid_argument&) { rejected=true; }
            Check(rejected,"Stateful GPU fetch accepted a reserved attribute source mode");
            same_state(history_state,before_rejections);
            rejected=false;
            try { fetch.FetchPica(history_state,history_inputs,history_layout,65,1,0,0,
                                 {invalid_indices,0,3,invalid_indices.contents,{}},3); }
            catch (const std::invalid_argument&) { rejected=true; }
            Check(rejected,"Stateful GPU fetch accepted a three-byte index width");
            same_state(history_state,before_rejections);
            Check(preserved_state_cases==6,"Native vertex input state-preservation cases were omitted");
            std::cout << preserved_state_cases << " rejected/empty native vertex input cases preserve all512 GPU-bank bytes and reset metadata\n";
            std::cout << "Private GPU attribute/register banks, all-loaded/unmapped attributes, aliases, fixed/array priority, immediate/batch sharing, 8/16/32-bit indexed last-cache-miss/eviction order, submissions, save/load, reset/isolation and bounds passed\n";
            std::cout << "16 PICA attribute format/component combinations, register-derived layouts, padding, aliases, retained inputs, unaligned/private buffers, strides, first vertex, fixed values, bounds and oversized stream allocations passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
