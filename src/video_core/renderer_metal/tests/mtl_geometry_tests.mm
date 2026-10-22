// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

#include "mtl_geometry.h"
#include "mtl_draw_runtime.h"
#include "mtl_fragment_operations.h"
#include "mtl_format_conversion.h"
#include "mtl_texture_runtime.h"
#include "mtl_pica_shader_provider.h"
#include "mtl_shader_runtime.h"
#include "mtl_pica_vertex_fixture.h"
#include "mtl_vertex_fetch.h"
#include "video_core/shader/generator/shader_uniforms.h"
#include "video_core/pica/primitive_assembly.h"
#include <algorithm>
#include "mtl_pipeline_cache.h"
#include "mtl_transfer.h"
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void Check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::array<double, 4> SamplerReference(double u, double v, unsigned level, unsigned wrap_s,
                                      unsigned wrap_t, bool linear) {
    const unsigned size = 4U >> level;
    // Reference evaluates original coordinates and integer taps directly. It does not
    // share the generated shader's coordinate folding or wrapped-index implementation.
    const auto index = [](int value, int extent, unsigned mode, double coordinate) {
        if (mode == 4) mode = coordinate < 0 ? 2 : 0;
        if (mode == 5) mode = coordinate < 0 ? 2 : 1;
        if (mode == 0) return std::clamp(value, 0, extent - 1);
        if (mode == 1) return value < 0 || value >= extent ? -1 : value;
        if (mode == 3) {
            while (value < 0) value += extent * 2;
            while (value >= extent * 2) value -= extent * 2;
            return value >= extent ? extent * 2 - value - 1 : value;
        }
        while (value < 0) value += extent;
        while (value >= extent) value -= extent;
        return value;
    };
    const auto read = [&](int x, int y) {
        x = index(x, size, wrap_s, u); y = index(y, size, wrap_t, v);
        if (x < 0 || y < 0) return std::array<double,4>{17, 83, 149, 211};
        return std::array<double,4>{double(30 + level*60 + x*13), double(40 + y*21),
                                    double(50 + x*7 + y*11), double(100 + level*40 + x*5)};
    };
    const double x = u*size - (linear ? .5 : 0), y = v*size - (linear ? .5 : 0);
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    if (!linear) return read(ix,iy);
    const double wx = x - ix, wy = y - iy;
    const auto a=read(ix,iy), b=read(ix+1,iy), c=read(ix,iy+1), d=read(ix+1,iy+1);
    std::array<double,4> result{};
    for (unsigned channel=0; channel<4; ++channel)
        result[channel]=(a[channel]*(1-wx)+b[channel]*wx)*(1-wy)+
                        (c[channel]*(1-wx)+d[channel]*wx)*wy;
    return result;
}

unsigned LogicReference(unsigned s, unsigned d, unsigned op, unsigned mask) {
    switch (op) {
    case 0: return 0;
    case 1: return s & d;
    case 2: return s & (~d & mask);
    case 3: return s;
    case 4: return mask;
    case 5: return ~s & mask;
    case 6: return d;
    case 7: return ~d & mask;
    case 8: return ~(s & d) & mask;
    case 9: return s | d;
    case 10: return ~(s | d) & mask;
    case 11: return s ^ d;
    case 12: return ~(s ^ d) & mask;
    case 13: return (~s & mask) & d;
    case 14: return s | (~d & mask);
    case 15: return (~s & mask) | d;
    }
    throw std::runtime_error("Invalid reference logic operation");
}

std::array<double, 4> BlendFactorReference(unsigned factor, const std::array<double, 4>& source,
                                          const std::array<double, 4>& dest,
                                          const std::array<double, 4>& constant) {
    std::array<double, 4> result{};
    for (unsigned c = 0; c < 4; ++c) {
        switch (factor) {
        case 0: result[c] = 0; break;
        case 1: result[c] = 1; break;
        case 2: result[c] = source[c]; break;
        case 3: result[c] = 1 - source[c]; break;
        case 4: result[c] = dest[c]; break;
        case 5: result[c] = 1 - dest[c]; break;
        case 6: result[c] = source[3]; break;
        case 7: result[c] = 1 - source[3]; break;
        case 8: result[c] = dest[3]; break;
        case 9: result[c] = 1 - dest[3]; break;
        case 10: result[c] = constant[c]; break;
        case 11: result[c] = 1 - constant[c]; break;
        case 12: result[c] = constant[3]; break;
        case 13: result[c] = 1 - constant[3]; break;
        case 14: result[c] = c == 3 ? 1 : std::min(source[3], 1 - dest[3]); break;
        default: result[c] = 1; break;
        }
    }
    return result;
}

constexpr std::string_view VertexSource = R"(
layout(location=0) in vec4 input_position;
layout(location=1) in vec4 input_quaternion;
layout(location=2) in vec4 input_color;
layout(location=1) out vec4 primary_color;
layout(location=6) out vec4 normquat;
layout(set=0, binding=1, std140) uniform vs_data {
    bool enable_clip1;
    bool flip_viewport;
    vec4 clip_coef;
};
out gl_PerVertex { invariant vec4 gl_Position; float gl_ClipDistance[2]; };
void main() {
    vec4 p = input_position;
    if (flip_viewport) p.y = -p.y;
    gl_Position = vec4(p.xy, -p.z, p.w);
    gl_ClipDistance[0] = -input_position.z;
    gl_ClipDistance[1] = enable_clip1 ? dot(clip_coef, input_position) : 0.0;
    primary_color = input_color;
    normquat = input_quaternion;
}
)";

std::vector<Metal::CapturedVertex> Download(Metal::TransferRuntime& transfers,
                                           Metal::Scheduler& scheduler, const Metal::Device& device,
                                           const Metal::AssembledVertices& vertices) {
    const auto size = vertices.vertex_count * sizeof(Metal::CapturedVertex);
    auto buffer = device.CreateBuffer(size, true);
    transfers.CopyBuffer(vertices.storage.buffer, vertices.storage.offset, buffer, 0, size);
    scheduler.Finish();
    std::vector<Metal::CapturedVertex> result(vertices.vertex_count);
    std::memcpy(result.data(), buffer.contents, size);
    return result;
}

void Verify(const std::vector<Metal::CapturedVertex>& vertices, std::span<const unsigned> order,
            float quaternion_w) {
    Check(vertices.size() == order.size(), "Incorrect assembled vertex count");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        Check(vertices[i].position[0] == float(order[i]) && vertices[i].position[2] == 0.5f &&
              vertices[i].position[3] == 1.0f, "GPU primitive assembly produced incorrect positions");
        Check(vertices[i].quaternion[3] == quaternion_w,
              "GPU quaternion correction did not preserve a consistent triangle hemisphere");
        Check(vertices[i].clip[0] == 0.5f && vertices[i].clip[1] == 0.0f,
              "GPU vertex capture did not preserve PICA clip distances");
        Check(vertices[i].color[0] == float(order[i]) / 4.0f && vertices[i].color[3] == 1.0f,
              "GPU vertex capture lost vertex attributes");
    }
}

} // namespace


void CheckPrimitiveUploadRecycling(Metal::Device& device, Metal::Scheduler& scheduler,
                                   Metal::TransferRuntime& transfers) {
    constexpr unsigned draws = 12293, source_vertices = draws * 4;
    constexpr NSUInteger stride = 12 * sizeof(Metal::CapturedVertex) + 32;
    std::vector<Metal::CapturedVertex> values(source_vertices);
    for (unsigned vertex = 0; vertex < values.size(); ++vertex) {
        std::array<std::uint32_t, 32> bits{};
        for (unsigned lane = 0; lane < bits.size(); ++lane)
            bits[lane] = 0x3F000000U + vertex * 32 + lane;
        bits[4] = 0x7FC00000U | (vertex + 1); bits[5] = 0x80000000U;
        bits[6] = vertex + 1; bits[7] = 0x7F800000U;
        std::memcpy(&values[vertex], bits.data(), sizeof(Metal::CapturedVertex));
    }
    auto upload = device.CreateBuffer(values.size() * sizeof(Metal::CapturedVertex), true);
    std::memcpy(upload.contents, values.data(), upload.length);
    auto source = device.CreateBuffer(upload.length, false);
    transfers.CopyBuffer(upload, 0, source, 0, upload.length);
    auto index_buffer = device.CreateBuffer(draws * 17, true);
    std::memset(index_buffer.contents, 0, index_buffer.length);
    for (unsigned draw = 0; draw < draws; ++draw) {
        const unsigned width = draw % 4 == 0 ? 0 : draw % 4 == 1 ? 1 : draw % 4 == 2 ? 2 : 4;
        for (unsigned element = 0; element < 4 && width; ++element)
            static_cast<u8*>(index_buffer.contents)[draw * 17 + 1 + element * width] = (element * 3 + draw) % 4;
    }
    for (unsigned topology = 0; topology < 3; ++topology) {
        Metal::GeometryRuntime geometry{device, scheduler};
        const auto mode = Metal::TriangleTopology(topology);
        std::array<Metal::PrimitiveStream, 2> streams{Metal::PrimitiveStream{device}, Metal::PrimitiveStream{device}};
        std::array<Pica::PrimitiveAssembler, 2> references{
            Pica::PrimitiveAssembler{Pica::PipelineRegs::TriangleTopology(topology)},
            Pica::PrimitiveAssembler{Pica::PipelineRegs::TriangleTopology(topology)}};
        std::array<std::vector<unsigned>, 2> submitted;
        std::vector<std::vector<Metal::CapturedVertex>> expected(draws);
        auto readback = device.CreateBuffer(draws * stride, true);
        std::memset(readback.contents, 0xA7, readback.length);
        const auto initial_tick = scheduler.CurrentTick();
        for (unsigned draw = 0; draw < draws; ++draw) {
            const unsigned bank = draw & 1, count = 1 + draw % 4;
            if (draw % 257 == 0) { streams[bank].Reset(); references[bank].Reset(); submitted[bank].clear(); }
            const unsigned width = draw % 4 == 0 ? 0 : draw % 4 == 1 ? 1 : draw % 4 == 2 ? 2 : 4;
            for (unsigned element = 0; element < count; ++element) {
                const unsigned id = draw * 4 + (width ? (element * 3 + draw) % 4 : element);
                Pica::OutputVertex vertex{}; vertex.pos.x = Pica::f24::FromFloat32(float(id));
                references[bank].SubmitVertex(vertex, [&](const auto& a, const auto& b, const auto& c) {
                    for (const auto* v : {&a, &b, &c}) expected[draw].push_back(values[unsigned(v->pos.x.ToFloat32())]);
                });
                submitted[bank].push_back(id);
            }
            const Metal::BufferSlice captured{source, NSUInteger(draw) * 4 * sizeof(Metal::CapturedVertex), 4 * sizeof(Metal::CapturedVertex), nullptr, {}};
            const Metal::BufferSlice indices{index_buffer, draw * 17 + 1, count * width, nullptr, {}};
            const auto result = geometry.Append(streams[bank], captured, 4, mode, count, 0, 0, indices, width, false);
            Check(result.vertex_count == expected[draw].size(), "Recycled primitive output count differs");
            if (result.vertex_count) transfers.CopyBuffer(result.storage.buffer, result.storage.offset,
                readback, draw * stride + 16, result.vertex_count * sizeof(Metal::CapturedVertex));
        }
        // 48-byte parameters aligned to256 fill three1MiB pages before this wrap.
        Check(scheduler.CurrentTick() > initial_tick, "Primitive recycling did not cross an automatic submission");
        scheduler.Finish();
        for (unsigned draw = 0; draw < draws; ++draw) {
            const auto* bytes = static_cast<const u8*>(readback.contents) + draw * stride;
            const auto size = expected[draw].size() * sizeof(Metal::CapturedVertex);
            Check(!size || !std::memcmp(bytes + 16, expected[draw].data(), size), "Recycled primitive output lost raw bits/history/parity");
            for (unsigned b = 0; b < 16; ++b) Check(bytes[b] == 0xA7 && bytes[16 + size + b] == 0xA7, "Primitive recycling damaged a guard");
        }
        for (unsigned bank = 0; bank < 2; ++bank) {
            const auto& ids = submitted[bank];
            const unsigned pending = topology == 0 ? ids.size() % 3 : std::min<std::size_t>(2, ids.size());
            const unsigned index = topology == 0 ? pending : topology == 1 ? ids.size() & 1 : 1;
            const unsigned slot = topology == 1 && pending == 2 ? index : 0;
            std::array<Metal::CapturedVertex, 2> slots{};
            std::array<std::uint32_t, 4> flags{index, topology != 0 && pending == 2, 0, 0};
            if (pending) { slots[slot] = values[topology == 2 ? ids.front() : ids[ids.size() - pending]]; flags[2 + slot] = 1; }
            if (pending == 2) { slots[slot ^ 1] = values[ids.back()]; flags[2 + (slot ^ 1)] = 1; }
            std::array<std::byte, 272> reference{};
            std::memcpy(reference.data(), slots.data(), sizeof(slots)); std::memcpy(reference.data() + sizeof(slots), flags.data(), sizeof(flags));
            const auto snapshot = geometry.SaveStream(streams[bank]);
            Check(!snapshot.needs_reset && snapshot.pending == pending && snapshot.topology == mode && snapshot.history == reference,
                  "Recycled primitive full272 history differs");
        }
    }
    std::cout << draws * 3 << " List/Strip/Fan Append snapshots and six full272 histories preserve raw bits, resets, parity, zero-triangle updates and index widths across upload-page recycling\n";
}

// Loading an archive must target the current physical history half. Queue the
// load and following updates without waiting, including an empty Append and a
// rejected request, then compare complete output and the canonical 272 bytes.
void CheckPrimitiveHistoryReloads(Metal::Device& device, Metal::Scheduler& scheduler,
                                 Metal::TransferRuntime& transfers, Metal::GeometryRuntime& geometry) {
    std::array<Metal::CapturedVertex,8> values{};
    for(unsigned vertex=0;vertex<values.size();++vertex) {
        std::array<std::uint32_t,32> bits{};
        for(unsigned word=0;word<bits.size();++word) bits[word]=0x3F000000U+vertex*32+word;
        bits[4]=0x7FC00001U+vertex;bits[5]=0x80000000U;bits[6]=vertex+1;bits[7]=0x7F800000U;
        std::memcpy(&values[vertex],bits.data(),sizeof(Metal::CapturedVertex));
    }
    auto upload=device.CreateBuffer(sizeof(values),true);
    std::memcpy(upload.contents,values.data(),sizeof(values));
    auto source=device.CreateBuffer(sizeof(values),false);
    transfers.CopyBuffer(upload,0,source,0,source.length);
    const Metal::BufferSlice captured{source,0,source.length,nullptr,{}};
    unsigned cases=0;
    for(unsigned topology=0;topology<3;++topology) for(unsigned parity=0;parity<2;++parity)
    for(unsigned prefix=1;prefix<=2;++prefix) for(unsigned width:{0u,1u,2u,4u}) {
        const auto mode=Metal::TriangleTopology(topology);
        Metal::PrimitiveStream stream{device};
        if(parity) geometry.Append(stream,captured,8,mode,1,0,0,{},0,false);
        Metal::PrimitiveSnapshot initial{};
        initial.topology=mode;initial.pending=prefix;initial.needs_reset=false;
        const unsigned index=topology==0 ? prefix:topology==1 ? prefix&1u:1u;
        const unsigned slot=topology==1 && prefix==2 ? index:0;
        std::array<Metal::CapturedVertex,2> slots{};
        slots[slot]=values[0];if(prefix==2) slots[slot^1]=values[1];
        const std::array<unsigned,4> flags{index,topology!=0 && prefix==2,1,prefix==2};
        std::memcpy(initial.history.data(),slots.data(),sizeof(slots));
        std::memcpy(initial.history.data()+sizeof(slots),flags.data(),sizeof(flags));
        geometry.LoadStream(stream,initial);
        Pica::PrimitiveAssembler reference{Pica::PipelineRegs::TriangleTopology(topology)};
        std::vector<unsigned> submitted;
        const auto seed=[&](unsigned id,const auto& handler) {
            Pica::OutputVertex v{};v.pos.x=Pica::f24::FromFloat32(float(id));
            reference.SubmitVertex(v,handler);submitted.push_back(id);
        };
        for(unsigned i=0;i<prefix;++i) seed(i,[](const auto&,const auto&,const auto&) {});
        bool rejected=false;
        try { geometry.Append(stream,captured,8,mode,1,0,0,{},3,false); }
        catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected,"Invalid history request was accepted");
        std::vector<id<MTLBuffer>> readbacks;
        std::vector<std::vector<Metal::CapturedVertex>> expected;
        for(unsigned count:{0u,1u,2u,width ? 257u:3u,3u}) {
            auto indices=device.CreateBuffer(std::max(1u,count*width),true);
            std::memset(indices.contents,0,indices.length);
            expected.emplace_back();
            for(unsigned element=0;element<count;++element) {
                const unsigned id=width ? (element*3+count)%8:element;
                if(width) static_cast<u8*>(indices.contents)[element*width]=id;
                seed(id,[&](const auto& a,const auto& b,const auto& c) {
                    for(const auto* v:{&a,&b,&c}) expected.back().push_back(values[unsigned(v->pos.x.ToFloat32())]);
                });
            }
            const auto result=geometry.Append(stream,captured,8,mode,count,0,0,
                width ? Metal::BufferSlice{indices,0,count*width,nullptr,{}}:Metal::BufferSlice{},width,false);
            Check(result.vertex_count==expected.back().size(),"Reloaded primitive count differs");
            auto check=device.CreateBuffer(expected.back().size()*128+32,true);
            std::memset(check.contents,0xA7,check.length);
            if(result.vertex_count) transfers.CopyBuffer(result.storage.buffer,result.storage.offset,
                check,16,result.vertex_count*128ull);
            readbacks.push_back(check);++cases;
        }
        const unsigned pending=topology==0 ? submitted.size()%3:2;
        const unsigned final_index=topology==0 ? pending:topology==1 ? submitted.size()&1u:1u;
        const unsigned final_slot=topology==1 && pending==2 ? final_index:0;
        slots={};
        if(pending) slots[final_slot]=values[topology==2 ? submitted.front():submitted[submitted.size()-pending]];
        if(pending==2) slots[final_slot^1]=values[submitted.back()];
        const std::array<unsigned,4> final_flags{final_index,topology!=0 && pending==2,pending!=0,pending==2};
        std::array<std::byte,272> history{};
        std::memcpy(history.data(),slots.data(),sizeof(slots));
        std::memcpy(history.data()+sizeof(slots),final_flags.data(),sizeof(final_flags));
        const auto saved=geometry.SaveStream(stream);
        Check(saved.history==history && saved.pending==pending && saved.topology==mode && !saved.needs_reset,
              "Reloaded full272 history differs across physical half changes");
        for(unsigned draw=0;draw<readbacks.size();++draw) {
            const auto size=expected[draw].size()*128;
            const auto* bytes=static_cast<const u8*>(readbacks[draw].contents);
            Check(!size || !std::memcmp(bytes+16,expected[draw].data(),size),"Reloaded primitive lost raw output bits");
            for(unsigned word=0;word<16;++word)
                Check(bytes[word]==0xA7 && bytes[16+size+word]==0xA7,"Reloaded primitive damaged guards");
        }
    }
    std::cout<<cases<<" queued primitive archive reload/empty/rejected request cases preserve raw outputs, guards and full272 history in either half\n";
}

// Queue mixed layouts before completion, with two independently retained streams.
void CheckQueuedGeometryBindings(Metal::Device& device, Metal::Scheduler& scheduler,
                                 Metal::TransferRuntime& transfers, Metal::GeometryRuntime& geometry) {
    struct Readback {
        id<MTLBuffer> buffer;
        std::vector<Metal::CapturedVertex> expected;
    };
    std::vector<Readback> readbacks;
    unsigned submits{};
    const auto queue = [&](const Metal::AssembledVertices& result,
                           const std::vector<Metal::CapturedVertex>& expected) {
        Check(result.vertex_count == expected.size(), "Queued geometry count mismatch");
        if (expected.empty()) return;
        auto buffer = device.CreateBuffer(expected.size() * sizeof(Metal::CapturedVertex) + 32, true);
        std::memset(buffer.contents, 0x5C, buffer.length);
        transfers.CopyBuffer(result.storage.buffer, result.storage.offset, buffer, 16,
                             expected.size() * sizeof(Metal::CapturedVertex));
        readbacks.push_back({buffer, expected});
    };
    const auto fix = [](Metal::CapturedVertex a, Metal::CapturedVertex b,
                        Metal::CapturedVertex c, bool enabled) {
        if (enabled) {
            if (a.quaternion[0] * b.quaternion[0] < 0)
                for (auto& v : b.quaternion) v = -v;
            if (a.quaternion[0] * c.quaternion[0] < 0)
                for (auto& v : c.quaternion) v = -v;
        }
        return std::array{a, b, c};
    };
    for (unsigned topology = 0; topology < 3; ++topology) {
        const auto mode = Metal::TriangleTopology(topology);
        std::array<Metal::PrimitiveStream, 2> streams{Metal::PrimitiveStream{device}, Metal::PrimitiveStream{device}};
        std::array<Pica::PrimitiveAssembler, 2> references{
            Pica::PrimitiveAssembler{Pica::PipelineRegs::TriangleTopology(topology)},
            Pica::PrimitiveAssembler{Pica::PipelineRegs::TriangleTopology(topology)}};
        std::vector<Metal::CapturedVertex> values;
        std::array<std::vector<Metal::CapturedVertex>, 2> history;
        for (unsigned draw = 0; draw < 24; ++draw) {
            constexpr std::array counts{1U, 2U, 3U, 65U, 4U, 257U};
            const unsigned width = draw % 4 == 0 ? 0 : draw % 4 == 1 ? 1 : draw % 4 == 2 ? 2 : 4;
            const auto count = width ? counts[draw % counts.size()] : std::min(8U, counts[draw % counts.size()]);
            const auto first_id = values.size();
            for (unsigned vertex = 0; vertex < 8; ++vertex) {
                std::array<float, 32> words{};
                for (unsigned lane = 0; lane < words.size(); ++lane)
                    words[lane] = float((first_id + vertex) * 32 + lane) * .125f;
                Metal::CapturedVertex value{};
                std::memcpy(&value, words.data(), sizeof(value));
                value.quaternion = {vertex & 1 ? -1.f : 1.f, 0, 0, 0};
                values.push_back(value);
            }
            auto upload = device.CreateBuffer(8 * sizeof(Metal::CapturedVertex), true);
            std::memcpy(upload.contents, values.data() + first_id, upload.length);
            auto source = device.CreateBuffer(upload.length + 32, false);
            transfers.CopyBuffer(upload, 0, source, 16, upload.length);
            const Metal::BufferSlice captured{source, 16, upload.length, nullptr, {}};
            Metal::BufferSlice indices{};
            if (width) {
                auto buffer = device.CreateBuffer(count * width + 2, true);
                std::memset(buffer.contents, 0, buffer.length);
                for (unsigned element = 0; element < count; ++element)
                    static_cast<unsigned char*>(buffer.contents)[1 + element * width] = (element * 3 + draw) % 8;
                indices = {buffer, 1, count * width, nullptr, {}};
            }
            std::vector<Metal::CapturedVertex> assembled_expected, stream_expected;
            Pica::PrimitiveAssembler direct{Pica::PipelineRegs::TriangleTopology(topology)};
            const bool repair = draw & 2;
            const auto handler = [&](auto& output) {
                return [&](const auto& a, const auto& b, const auto& c) {
                    const auto triangle = fix(values[unsigned(a.pos.x.ToFloat32())],
                        values[unsigned(b.pos.x.ToFloat32())], values[unsigned(c.pos.x.ToFloat32())], repair);
                    output.insert(output.end(), triangle.begin(), triangle.end());
                };
            };
            for (unsigned element = 0; element < count; ++element) {
                const unsigned vertex = width ? (element * 3 + draw) % 8 : element;
                Pica::OutputVertex v{};
                v.pos.x = Pica::f24::FromFloat32(float(first_id + vertex));
                direct.SubmitVertex(v, handler(assembled_expected));
                references[draw & 1].SubmitVertex(v, handler(stream_expected));
                history[draw & 1].push_back(values[first_id + vertex]);
            }
            const auto tick = scheduler.CurrentTick();
            queue(geometry.Assemble(captured, 8, mode, count, 0, 0, indices, width, repair), assembled_expected);
            queue(geometry.Append(streams[draw & 1], captured, 8, mode, count, 0, 0, indices, width, repair), stream_expected);
            Check(scheduler.CurrentTick() == tick, "Geometry binding reuse introduced a submit/wait");
            if (draw % 7 == 6) { scheduler.Submit(); ++submits; }
        }
        for (unsigned bank = 0; bank < 2; ++bank) {
            const auto& submitted = history[bank];
            const unsigned pending = topology == 0 ? submitted.size() % 3 : std::min<std::size_t>(2, submitted.size());
            const unsigned index = topology == 0 ? pending : topology == 1 ? submitted.size() & 1 : 1;
            std::array<Metal::CapturedVertex, 2> slots{};
            std::array<std::uint32_t, 4> flags{index, topology != 0 && pending == 2, 0, 0};
            const unsigned slot = topology == 1 && pending == 2 ? index : 0;
            if (pending) { slots[slot] = topology == 2 ? submitted.front() : submitted[submitted.size() - pending]; flags[2 + slot] = 1; }
            if (pending == 2) { slots[slot ^ 1] = submitted.back(); flags[2 + (slot ^ 1)] = 1; }
            std::array<std::byte, 272> expected{};
            std::memcpy(expected.data(), slots.data(), sizeof(slots));
            std::memcpy(expected.data() + sizeof(slots), flags.data(), sizeof(flags));
            const auto saved = geometry.SaveStream(streams[bank]);
            Check(!saved.needs_reset && saved.pending == pending && saved.topology == mode && saved.history == expected,
                  "Full queued 272-byte primitive history mismatch");
        }
    }
    Metal::PrimitiveStream winding_stream{device};
    for (unsigned draw = 0; draw < 12; ++draw) {
        std::vector<Metal::CapturedVertex> expected(3);
        for (unsigned vertex = 0; vertex < 3; ++vertex) {
            std::array<float, 32> words{};
            for (unsigned lane = 0; lane < 32; ++lane) words[lane] = float(draw * 128 + vertex * 32 + lane);
            std::memcpy(&expected[vertex], words.data(), sizeof(Metal::CapturedVertex));
        }
        auto upload = device.CreateBuffer(3 * sizeof(Metal::CapturedVertex), true);
        std::memcpy(upload.contents, expected.data(), upload.length);
        auto source = device.CreateBuffer(upload.length, false);
        transfers.CopyBuffer(upload, 0, source, 0, upload.length);
        Metal::BufferSlice winding{};
        if (draw % 3 != 2) {
            auto buffer = device.CreateBuffer(8, true);
            const std::uint32_t word = draw % 3 == 1;
            std::memcpy(static_cast<std::byte*>(buffer.contents) + 4, &word, 4);
            winding = {buffer, 4, 4, nullptr, {}};
            if (word) std::swap(expected[0], expected[1]);
        }
        winding_stream.Reset();
        queue(geometry.Append(winding_stream, {source, 0, source.length, nullptr, {}}, 3,
            Metal::TriangleTopology::List, 3, 0, 0, {}, 0, false, winding), expected);
    }
    scheduler.Finish();
    for (const auto& result : readbacks) {
        const auto bytes = result.expected.size() * sizeof(Metal::CapturedVertex);
        const auto* data = static_cast<const unsigned char*>(result.buffer.contents);
        Check(!std::memcmp(data + 16, result.expected.data(), bytes), "Queued full geometry output mismatch");
        for (unsigned i = 0; i < 16; ++i)
            Check(data[i] == 0x5C && data[16 + bytes + i] == 0x5C, "Queued geometry readback canary mismatch");
    }
    std::cout << "Queued geometry: 156 mixed Assemble/Append calls, full 128-byte vertices, six full histories, "
              << submits << " explicit submits, two streams, all index widths, alternating winding/empty slots and released inputs passed\n";
}

namespace {
// CPU oracle input for varying complete native Capture and Draw resource layouts.
struct BindingScene {
    std::array<Metal::VertexCaptureProgram,4> captures;
    std::array<Metal::CompiledFragmentShader,4> fragments;
    std::array<id<MTLSamplerState>,2> samplers;
    explicit BindingScene(Metal::Device& device) {
        for(unsigned mode=0;mode<4;++mode) {
            const auto slots=Slots(mode);
            std::string source="#include <metal_stdlib>\nusing namespace metal;\nstruct V{float4 f[8];};\n"
                "kernel void capture_bindings(device const float4* input [[buffer(17)]],device V* output [[buffer(18)]],"
                "constant uint4& info [[buffer(19)]],uint id [[thread_position_in_grid]]";
            for(auto slot:slots) source+=",constant float4& u"+std::to_string(slot)+" [[buffer("+std::to_string(slot)+")]]";
            source+=") {if(id>=info.x)return;float4 shift=float4(0);";
            for(auto slot:slots) source+="shift+=u"+std::to_string(slot)+";";
            source+="for(uint j=0;j<8;++j)output[id].f[j]=input[id*16+j];"
                "for(uint j=1;j<7;++j)output[id].f[j]+=shift;output[id].f[6].x+=float(info.y);}";
            auto library=device.CompileLibrary(source,"Native Capture binding benchmark");
            auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
            descriptor.computeFunctionDescriptor=Metal::Device::Function(library,"capture_bindings");
            descriptor.maxTotalThreadsPerThreadgroup=64;
            Metal::TranslatedShader shader{};shader.stage=Metal::ShaderStage::VertexCompute;shader.entry_point="capture_bindings";
            for(auto slot:slots) {
                Metal::ShaderResource resource{};resource.kind=Metal::ShaderResourceKind::UniformBuffer;
                resource.metal_index=slot;resource.byte_size=16;shader.resources.push_back(resource);
            }
            captures[mode]={std::move(shader),device.CompileCompute(descriptor),library};
            source="#include <metal_stdlib>\nusing namespace metal;\nstruct F{float4 color [[user(locn1)]];};\n"
                "fragment float4 fragment_bindings(F input [[stage_in]],constant float4& fs [[buffer(2)]]";
            if(mode==3) source+=",constant float4& gas [[buffer(23)]]";
            constexpr unsigned texture_slots[3]{3,9,13},sampler_slots[3]{0,6,9};
            for(unsigned i=0;i<mode;++i) source+=",texture2d<float> t"+std::to_string(i)+" [[texture("+
                std::to_string(texture_slots[i])+")]],sampler s"+std::to_string(i)+" [[sampler("+std::to_string(sampler_slots[i])+")]]";
            source+="){float4 result=input.color+fs;";
            if(mode==3) source+="result+=gas;";
            for(unsigned i=0;i<mode;++i) source+="result+=t"+std::to_string(i)+".sample(s"+std::to_string(i)+",float2(0.625,0.5));";
            source+="return result;}";
            library=device.CompileLibrary(source,"Native Draw binding benchmark");
            shader={};shader.stage=Metal::ShaderStage::Fragment;shader.entry_point="fragment_bindings";
            Metal::ShaderResource fs{};fs.kind=Metal::ShaderResourceKind::UniformBuffer;fs.metal_index=2;fs.byte_size=16;
            shader.resources.push_back(fs);
            if(mode==3) {fs.metal_index=23;shader.resources.push_back(fs);}
            for(unsigned i=0;i<mode;++i) {
                Metal::ShaderResource resource{};resource.kind=Metal::ShaderResourceKind::Texture;
                resource.metal_index=texture_slots[i];resource.sampler_index=sampler_slots[i];shader.resources.push_back(resource);
            }
            fragments[mode]={{std::move(shader),{}},library};
        }
        for(unsigned linear=0;linear<2;++linear) {
            auto* descriptor=[[MTLSamplerDescriptor alloc] init];
            descriptor.minFilter=descriptor.magFilter=linear?MTLSamplerMinMagFilterLinear:MTLSamplerMinMagFilterNearest;
            descriptor.sAddressMode=descriptor.tAddressMode=MTLSamplerAddressModeClampToEdge;
            samplers[linear]=device.CreateSampler(descriptor);
        }
    }
    static std::vector<unsigned> Slots(unsigned mode) {
        if(mode==0)return {};if(mode==1)return {0};if(mode==2)return {1,2};return {0,1,2,16};
    }
    static std::array<float,4> Uniform(unsigned draw,unsigned slot) {
        return {float(draw+slot)*.125f,.25f,.5f,1};
    }
    static Metal::CapturedVertex Vertex(unsigned draw,unsigned vertex) {
        Metal::CapturedVertex value{};
        constexpr float x[3]{-1,3,-1},y[3]{1,1,-3};
        value.position={x[vertex%3],y[vertex%3],0,1};value.color={float(draw)*.125f,.25f,.5f,1};
        value.texcoord0={.6f,.5f,0,0};value.texcoord1={.25f,.75f,0,0};value.texcoord2={.125f,.875f,0,0};
        value.quaternion={1,0,0,0};value.view={float(vertex),1,2,3};value.clip={1,1,1,0};return value;
    }
    static Metal::CapturedVertex Captured(unsigned draw,unsigned vertex,unsigned mode,unsigned first) {
        auto result=Vertex(draw,vertex);std::array<float,4> shift{};
        for(auto slot:Slots(mode)) {const auto u=Uniform(draw,slot);for(unsigned c=0;c<4;++c)shift[c]+=u[c];}
        // Explicit field assignment keeps the CPU reference independent of GPU struct indexing.
        for(auto* field:{&result.color,&result.texcoord0,&result.texcoord1,&result.texcoord2,&result.quaternion,&result.view})
            for(unsigned c=0;c<4;++c)(*field)[c]+=shift[c];
        result.view[0]+=float(first);return result;
    }
};

void ValidateVariableDrawCapture(Metal::Device& device,Metal::Scheduler& scheduler,Metal::GeometryRuntime& geometry) {
    Metal::TransferRuntime transfer{device,scheduler};
    constexpr unsigned calls=36,vertices=3,stride=vertices*128+32;
    auto capture_readback=device.CreateBuffer(calls*stride,true);
    std::memset(capture_readback.contents,0xA7,capture_readback.length);
    std::array<std::array<Metal::CapturedVertex,vertices>,calls> expected_capture{};
    std::array<std::array<float,4>,calls> expected_pixels{};
    auto* target_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
        width:calls height:1 mipmapped:NO];target_descriptor.storageMode=MTLStorageModePrivate;
    target_descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    auto target=device.CreateTexture(target_descriptor);
    transfer.ClearColor(target,MTLClearColorMake(-1,-1,-1,-1));
    const auto tick=scheduler.CurrentTick();unsigned submits=0,rejected=0;
    {
        BindingScene scene{device};Metal::DrawRuntime draws{device,scheduler,geometry};
        for(unsigned draw=0;draw<calls;++draw) {
            @autoreleasepool {
                const auto mode=draw%4;
                auto input=device.CreateBuffer(vertices*256,true),uniforms=device.CreateBuffer(6*256,true);
                std::vector<Metal::BufferBinding> capture_uniforms,fragment_uniforms;
                const auto slots=BindingScene::Slots(mode);
                for(unsigned j=0;j<4;++j) {
                    constexpr unsigned indices[4]{0,1,2,16};const auto u=BindingScene::Uniform(draw,indices[j]);
                    std::memcpy(static_cast<std::byte*>(uniforms.contents)+j*256,u.data(),16);
                    if(std::find(slots.begin(),slots.end(),indices[j])!=slots.end())
                        capture_uniforms.push_back({uniforms,j*256,indices[j]});
                }
                for(unsigned vertex=0;vertex<vertices;++vertex) {
                    const auto value=BindingScene::Vertex(draw,vertex);
                    std::memcpy(static_cast<std::byte*>(input.contents)+vertex*256,&value,128);
                    expected_capture[draw][vertex]=BindingScene::Captured(draw,vertex,mode,draw*2);
                }
                if(mode) {
                    bool caught=false;try {geometry.Capture(scene.captures[mode],{input,0,input.length,nullptr,{}},vertices,draw*2,{});}
                    catch(const std::invalid_argument&) {caught=true;}
                    Check(caught,"Variable Capture accepted missing active uniform");++rejected;
                }
                const auto captured=geometry.Capture(scene.captures[mode],{input,0,input.length,nullptr,{}},vertices,draw*2,capture_uniforms);
                const auto fs=BindingScene::Uniform(draw,4),gas=BindingScene::Uniform(draw,5);
                std::memcpy(static_cast<std::byte*>(uniforms.contents)+4*256,fs.data(),16);
                std::memcpy(static_cast<std::byte*>(uniforms.contents)+5*256,gas.data(),16);
                fragment_uniforms.push_back({uniforms,4*256,2});
                if(mode==3)fragment_uniforms.push_back({uniforms,5*256,23});
                if(draw&1)fragment_uniforms.push_back({uniforms,0,0});
                auto pixel=expected_capture[draw][0].color;
                for(unsigned c=0;c<4;++c) {pixel[c]+=fs[c];if(mode==3)pixel[c]+=gas[c];}
                std::vector<Metal::TextureBinding> textures;std::vector<Metal::SamplerBinding> samplers;
                for(unsigned texture=0;texture<mode;++texture) {
                    auto* descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                        width:2 height:1 mipmapped:NO];descriptor.storageMode=MTLStorageModeShared;descriptor.usage=MTLTextureUsageShaderRead;
                    auto image=device.CreateTexture(descriptor);const float base=float(draw+texture)*.125f;
                    const float values[8]{base,base,base,base,base+2,base+2,base+2,base+2};
                    [image replaceRegion:MTLRegionMake2D(0,0,2,1) mipmapLevel:0 withBytes:values bytesPerRow:sizeof(values)];
                    constexpr unsigned ti[3]{3,9,13},si[3]{0,6,9};const bool linear=(draw+texture)&1;
                    textures.push_back({image,ti[texture]});samplers.push_back({scene.samplers[linear],si[texture]});
                    for(auto& component:pixel)component+=base+(linear?1.5f:2.f);
                }
                expected_pixels[draw]=pixel;
                auto* pass=[[MTL4RenderPassDescriptor alloc] init];pass.colorAttachments[0].texture=target;
                pass.colorAttachments[0].loadAction=MTLLoadActionLoad;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
                pass.renderTargetWidth=calls;pass.renderTargetHeight=1;
                if(mode) {
                    bool caught=false;
                    try {draws.Draw({},scene.fragments[mode],pass,{double(draw),0,1,1,0,1},{draw,0,1,1},
                        {captured,vertices},fragment_uniforms,textures,{});}
                    catch(const std::invalid_argument&) {caught=true;}
                    Check(caught,"Variable Draw accepted missing active samplers");++rejected;
                }
                draws.Draw({},scene.fragments[mode],pass,{double(draw),0,1,1,0,1},{draw,0,1,1},
                    {captured,vertices},fragment_uniforms,textures,samplers);
                transfer.CopyBuffer(captured.buffer,captured.offset,capture_readback,draw*stride+16,vertices*128);
                if(draw%8==7)draws.Clear(); // Queued pipelines and bindings must outlive cache clearing.
            }
            if(draw%7==6) {scheduler.Submit();++submits;}
            Check(scheduler.CurrentTick()==tick+submits,"Variable Draw/Capture submitted early");
        }
    } // Release Draw tables, shader libraries and samplers before the final GPU completion.
    const auto pixels=transfer.Readback({target,0,0,MTLRegionMake2D(0,0,calls,1)},calls*16);
    for(unsigned draw=0;draw<calls;++draw) {
        auto* bytes=static_cast<const std::byte*>(capture_readback.contents)+draw*stride+16;
        Check(!std::memcmp(bytes,expected_capture[draw].data(),vertices*128),"Variable Capture full output changed");
        for(unsigned g=0;g<16;++g)Check(bytes[int(g)-16]==std::byte{0xA7}&&bytes[vertices*128+g]==std::byte{0xA7},"Variable Capture guard changed");
        float actual[4];std::memcpy(actual,pixels.data()+draw*16,16);
        for(unsigned c=0;c<4;++c)Check(std::isfinite(actual[c])&&std::abs(actual[c]-expected_pixels[draw][c])<.003f,
            "Variable Draw vertex/fragment snapshot changed");
    }
    std::cout<<calls<<" variable native Capture->Draw pairs: complete 128-byte vertices, changing uniforms/textures/samplers, "
        <<submits<<" submits, "<<rejected<<" missing-resource rejections, cache clears and released owners passed\n";
}

void CheckExactVertexCache(Metal::Device& device) {
    Metal::ShaderRuntime cache{device};
    Pica::RegsInternal regs{};
    Pica::ShaderSetup setup;
    Metal::Test::MovVertex(regs, setup);
    const auto original_regs = regs;
    const auto original_program = setup.GetProgramCode();
    const auto original_swizzle = setup.GetSwizzleData();
    const auto restore = [&] {
        regs = original_regs;
        setup.UpdateProgramCode(original_program);
        setup.UpdateSwizzleData(original_swizzle);
    };
    const auto first = cache.Vertex(regs, setup);
    const auto first_source = first->shader.source;
    using Mutation = std::function<void()>;
    std::vector<Mutation> mutations;
    for (unsigned field = 0; field < 11; ++field) mutations.push_back([&, field] {
        switch (field) {
        case 0: regs.vs.main_offset.Assign(1); break;
        case 1: regs.vs.output_mask.Assign(3); break;
        case 2: regs.rasterizer.vs_output_total.Assign(2); break;
        case 3: regs.lighting.disable.Assign(1); break;
        default: regs.rasterizer.vs_output_attributes[field - 4].raw ^= 1; break;
        }
    });
    // Unreached words are still part of the exact cache contract. Exercise both
    // ends, vector/block boundaries, and word values with different byte orders.
    for (unsigned offset : {4, 31, 32, 63, 64, 1023, 2047, 4095})
        for (u32 value : {1U, 256U, 0x01000000U}) {
            mutations.push_back([&, offset, value] { setup.UpdateProgramCode(offset, value); });
            mutations.push_back([&, offset, value] { setup.UpdateSwizzleData(offset, value); });
        }
    std::vector<std::shared_ptr<const Metal::VertexCaptureProgram>> programs;
    for (const auto& mutate : mutations) {
        restore(); mutate();
        const auto program = cache.Vertex(regs, setup);
        Check(program != first && program->Pipeline(), "Exact VS key mutation reused another program");
        Check(cache.CachedVertexCount() == programs.size() + 2,
              "Exact VS metadata or unused bank words collapsed into another key");
        Check(cache.Vertex(regs, setup) == program, "New VS key did not produce a stable cache hit");
        programs.push_back(program);
        restore();
        Check(cache.Vertex(regs, setup) == first, "Mutable lookup view changed an owned VS key");
    }
    for (std::size_t i = mutations.size(); i-- > 0;) {
        restore(); mutations[i]();
        Check(cache.Vertex(regs, setup) == programs[i], "Reversed VS lookup returned another key");
    }
    Check(cache.CachedVertexCount() == 60 && first->shader.source == first_source,
          "VS cache identity or immutable program changed after input mutation");
    cache.Clear();
    Check(!cache.CachedVertexCount() && first->Pipeline(),
          "VS clear invalidated a caller-held native pipeline");
    restore();
    Check(cache.Vertex(regs, setup) != first && cache.CachedVertexCount() == 1,
          "VS cache retained a dangling lookup view across clear");
    std::cout << "60 exact native VS keys: all11 metadata words, 48 code/swizzle boundary mutations, "
                 "stable/reversed hits, owned snapshots and caller-held Clear lifetime passed\n";
}

}
int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const bool bindings_only = argc == 2 && std::string_view{argv[1]} == "--bindings-only";
            const bool variable_bindings_only = argc == 2 && std::string_view{argv[1]} == "--variable-bindings-only";
            const bool vertex_cache_only = argc == 2 && std::string_view{argv[1]} == "--vertex-cache-identity-only";
            Check(bindings_only || variable_bindings_only || vertex_cache_only || argc == 1 || argc == 3, "Optional source cache arguments: absolute isolated root cold|warm");
            const auto disk_root = argc == 3 ? std::filesystem::path{argv[1]} : std::filesystem::path{};
            if (argc == 3) Check(disk_root.is_absolute(), "Geometry source cache root must be absolute");
            if (argc == 3) Check(std::string_view{argv[2]} == "cold" || std::string_view{argv[2]} == "warm" ||
                                 std::string_view{argv[2]} == "prewarm" || std::string_view{argv[2]} == "renderwarm",
                                 "Geometry cache mode must be cold, warm, prewarm or renderwarm");
            const bool render_corpus = argc == 3 && std::string_view{argv[2]} == "renderwarm";
            const bool prewarm_corpus = argc == 3 && (std::string_view{argv[2]} == "prewarm" || render_corpus);
            Metal::Device device{disk_root};
            if (vertex_cache_only) { CheckExactVertexCache(device); return 0; }
            if (argc == 3) Check(device.DiskCacheStats().enabled,
                                 "Persistent geometry corpus requires native disk cache enabled");
            Metal::Scheduler scheduler{device};
            Metal::TransferRuntime transfers{device, scheduler};
            Metal::GeometryRuntime geometry{device, scheduler};
            ValidateVariableDrawCapture(device,scheduler,geometry);
            if(variable_bindings_only) return 0;
            CheckQueuedGeometryBindings(device, scheduler, transfers, geometry);
            CheckPrimitiveUploadRecycling(device, scheduler, transfers);
            CheckPrimitiveHistoryReloads(device, scheduler, transfers, geometry);
            if (bindings_only) return 0;
            Metal::StreamBuffer input_stream{device, scheduler, 1 << 20};
            auto program = geometry.CompileVertex(VertexSource);
            auto inputs = input_stream.Allocate(4 * 16 * sizeof(std::array<float, 4>));
            std::memset(inputs.data, 0, inputs.size);
            auto* registers = static_cast<std::array<float, 4>*>(inputs.data);
            for (unsigned i = 0; i < 4; ++i) {
                registers[i * 16] = {float(i), 0.25f, -0.5f, 1.0f};
                registers[i * 16 + 1] = {0.0f, 0.0f, 0.0f, i & 1 ? -1.0f : 1.0f};
                registers[i * 16 + 2] = {float(i) / 4.0f, 0.0f, 0.0f, 1.0f};
            }
            auto uniform_data = input_stream.Allocate(32);
            std::memset(uniform_data.data, 0, uniform_data.size);
            const std::array<Metal::BufferBinding, 1> uniforms{{
                {uniform_data.buffer, uniform_data.offset, 1},
            }};
            auto captured = geometry.Capture(program, inputs, 4, 0, uniforms);
            Pica::RegsInternal real_vertex_regs{};
            Pica::ShaderSetup real_vertex_setup;
            Metal::Test::MovVertex(real_vertex_regs, real_vertex_setup);
            Metal::ShaderRuntime vertex_runtime{device};
            const std::atomic_bool keep_loading{false};
            std::uint64_t vs_library_baseline{};
            const auto prewarm_vertex = [&] {
                const auto result = vertex_runtime.Prewarm(keep_loading);
                Check(result.compiled && !result.failed && !result.cancelled && !result.budget_limited,
                      "Native vertex corpus preload failed or skipped source records");
                vs_library_baseline = vertex_runtime.DiskStats().library_compiles;
            };
            if (prewarm_corpus) prewarm_vertex();
            auto real_vertex = vertex_runtime.Vertex(real_vertex_regs, real_vertex_setup);
            Check(vertex_runtime.Vertex(real_vertex_regs, real_vertex_setup) == real_vertex &&
                  vertex_runtime.CachedVertexCount() == 1, "Real PICA vertex cache did not reuse its native compute pipeline");
            auto pica_uniform_data = input_stream.Allocate(sizeof(Pica::Shader::Generator::VSPicaUniformData));
            std::memset(pica_uniform_data.data, 0, pica_uniform_data.size);
            const std::array<Metal::BufferBinding, 2> real_uniforms{{
                {pica_uniform_data.buffer, pica_uniform_data.offset, 0},
                {uniform_data.buffer, uniform_data.offset, 1},
            }};
            auto real_captured = geometry.Capture(*real_vertex, inputs, 4, 0, real_uniforms);
            auto real_fan = geometry.Assemble(real_captured, 4, Metal::TriangleTopology::Fan, 4, 0, 0, {}, 0, true);
            constexpr std::array<unsigned, 6> real_fan_order{0, 1, 2, 0, 2, 3};
            Verify(Download(transfers, scheduler, device, real_fan), real_fan_order, 1.0f);
            vertex_runtime.Clear();
            Check(vertex_runtime.CachedVertexCount() == 0 && real_vertex->Pipeline(),
                  "Clearing the vertex cache invalidated a held native compute pipeline");
            if (prewarm_corpus) prewarm_vertex();
            std::cout << "Real PICA MOV microprogram, compute capture, semantic mapping and vertex-cache lifetime passed\n";
            Metal::VertexFetchRuntime fetch{device, scheduler};
            auto raw_vertices = device.CreateBuffer(4 * 48, true);
            for (unsigned i = 0; i < 4; ++i) {
                auto* bytes = static_cast<std::byte*>(raw_vertices.contents) + i * 48;
                std::memcpy(bytes, &registers[i * 16], 16);
                std::memcpy(bytes + 16, &registers[i * 16 + 1], 16);
                std::memcpy(bytes + 32, &registers[i * 16 + 2], 16);
            }
            auto raw_private = device.CreateBuffer(raw_vertices.length, false);
            transfers.CopyBuffer(raw_vertices, 0, raw_private, 0, raw_vertices.length);
            const std::array<Metal::BufferSlice, 1> raw_slices{{{raw_private, 0, raw_private.length, nullptr, {}}}};
            const std::array<Metal::VertexAttribute, 3> raw_attributes{{
                {0, 0, 48, 4, Metal::VertexAttributeFormat::Float, 0},
                {0, 16, 48, 4, Metal::VertexAttributeFormat::Float, 1},
                {0, 32, 48, 4, Metal::VertexAttributeFormat::Float, 2},
            }};
            auto fetched_inputs = fetch.Fetch(raw_slices, raw_attributes, 4);
            auto fetched_capture = geometry.Capture(*real_vertex, fetched_inputs, 4, 0, real_uniforms);
            auto fetched_fan = geometry.Assemble(fetched_capture, 4, Metal::TriangleTopology::Fan, 4, 0, 0, {}, 0, true);
            Verify(Download(transfers, scheduler, device, fetched_fan), real_fan_order, 1.0f);
            std::cout << "Private raw vertex buffers -> native GPU fetch -> real PICA microprogram -> fan assembly passed\n";
            auto arithmetic_inputs = input_stream.Allocate(16 * sizeof(std::array<float, 4>));
            std::memset(arithmetic_inputs.data, 0, arithmetic_inputs.size);
            auto* arithmetic_registers = static_cast<std::array<float, 4>*>(arithmetic_inputs.data);
            arithmetic_registers[0] = {0, 0.25f, -0.5f, 1};
            arithmetic_registers[1] = {0, 0, 0, 1};
            arithmetic_registers[2] = {0.25f, 0.5f, 0.75f, 0.9f};
            arithmetic_registers[3] = {0.5f, 0.25f, 0.125f, 0.2f};
            const auto check_vertex = [&](const std::array<float, 4>& expected) {
                auto executable = vertex_runtime.Vertex(real_vertex_regs, real_vertex_setup);
                auto output = geometry.Capture(*executable, arithmetic_inputs, 1, 0, real_uniforms);
                auto cpu = device.CreateBuffer(sizeof(Metal::CapturedVertex), true);
                transfers.CopyBuffer(output.buffer, output.offset, cpu, 0, cpu.length);
                scheduler.Finish();
                const auto& value = *static_cast<const Metal::CapturedVertex*>(cpu.contents);
                for (unsigned c = 0; c < 4; ++c) {
                    Check(std::abs(value.color[c] - std::min(std::abs(expected[c]), 1.0f)) < 0.00001f,
                          "Real PICA arithmetic/control-flow instruction or color semantic failed on Metal compute");
                }
            };
            using Op = nihstro::OpCode::Id;
            constexpr std::array arithmetic_ops{Op::ADD, Op::MUL, Op::DP3, Op::DP4, Op::DPH,
                Op::MIN, Op::MAX, Op::FLR, Op::RCP, Op::RSQ, Op::EX2, Op::LG2, Op::SGE, Op::SLT};
            for (auto op : arithmetic_ops) {
                real_vertex_setup.UpdateProgramCode(2, Metal::Test::Arithmetic(op, 2, 2, 3));
                std::array<float, 4> expected{};
                const auto& a = arithmetic_registers[2];
                const auto& b = arithmetic_registers[3];
                for (unsigned c = 0; c < 4; ++c) {
                    switch (op) {
                    case Op::ADD: expected[c] = a[c] + b[c]; break;
                    case Op::MUL: expected[c] = a[c] * b[c]; break;
                    case Op::DP3: expected[c] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; break;
                    case Op::DP4: expected[c] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; break;
                    case Op::DPH: expected[c] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + b[3]; break;
                    case Op::MIN: expected[c] = std::min(a[c], b[c]); break;
                    case Op::MAX: expected[c] = std::max(a[c], b[c]); break;
                    case Op::FLR: expected[c] = std::floor(a[c]); break;
                    case Op::RCP: expected[c] = 1 / a[0]; break;
                    case Op::RSQ: expected[c] = 1 / std::sqrt(a[0]); break;
                    case Op::EX2: expected[c] = std::exp2(a[0]); break;
                    case Op::LG2: expected[c] = std::log2(a[0]); break;
                    case Op::SGE: expected[c] = a[c] >= b[c]; break;
                    case Op::SLT: expected[c] = a[c] < b[c]; break;
                    default: throw std::runtime_error("Invalid arithmetic test opcode");
                    }
                }
                check_vertex(expected);
            }
            real_vertex_setup.UpdateProgramCode(2, Metal::Test::Arithmetic(Op::MUL, 2, 2, 3));
            const auto original_a = arithmetic_registers[2];
            const auto original_b = arithmetic_registers[3];
            arithmetic_registers[2].fill(0);
            arithmetic_registers[3].fill(std::numeric_limits<float>::infinity());
            check_vertex({0, 0, 0, 0});
            arithmetic_registers[2] = original_a;
            arithmetic_registers[3] = original_b;
            // CALL executes exactly one instruction. A dead EMIT immediately after its return
            // boundary must not be mistaken for a reached geometry instruction.
            real_vertex_setup.UpdateProgramCode(2, (unsigned(Op::CALL) << 26U) | (8U << 10U) | 1U);
            real_vertex_setup.UpdateProgramCode(8, Metal::Test::Arithmetic(Op::ADD, 2, 2, 3));
            real_vertex_setup.UpdateProgramCode(9, unsigned(Op::EMIT) << 26U);
            check_vertex({0.75f, 0.75f, 0.875f, 1.1f});
            // Uniform-controlled IFU: both outcomes share one compiled program.
            real_vertex_setup.UpdateProgramCode(2, (unsigned(Op::IFU) << 26U) | (5U << 10U) | 1U);
            real_vertex_setup.UpdateProgramCode(3, Metal::Test::Arithmetic(Op::MUL, 2, 2, 3));
            real_vertex_setup.UpdateProgramCode(4, unsigned(Op::NOP) << 26U);
            real_vertex_setup.UpdateProgramCode(5, Metal::Test::Arithmetic(Op::ADD, 2, 2, 3));
            real_vertex_setup.UpdateProgramCode(6, unsigned(Op::END) << 26U);
            auto* pica_uniforms = static_cast<Pica::Shader::Generator::VSPicaUniformData*>(pica_uniform_data.data);
            pica_uniforms->b = 1;
            check_vertex({0.125f, 0.125f, 0.09375f, 0.18f});
            const auto compiled_before = vertex_runtime.CachedVertexCount();
            pica_uniforms->b = 0;
            check_vertex({0.75f, 0.75f, 0.875f, 1.1f});
            Check(vertex_runtime.CachedVertexCount() == compiled_before,
                  "Dynamic PICA uniform values unnecessarily changed the compiled vertex cache key");
            // A LOOP accumulates in temporary register zero for i.x+1 iterations.
            real_vertex_setup.UpdateProgramCode(2, Metal::Test::Arithmetic(Op::MOV, 16, 2));
            real_vertex_setup.UpdateProgramCode(3, (unsigned(Op::LOOP) << 26U) | (4U << 10U));
            real_vertex_setup.UpdateProgramCode(4, Metal::Test::Arithmetic(Op::ADD, 16, 16, 3));
            real_vertex_setup.UpdateProgramCode(5, Metal::Test::Arithmetic(Op::MOV, 2, 16));
            arithmetic_registers[3] = {0.125f, 0.0625f, 0.03125f, 0.02f};
            pica_uniforms->i[0].x = 2;
            check_vertex({0.625f, 0.6875f, 0.84375f, 0.96f});
            std::cout << "14 real PICA arithmetic operations, zero*infinity, CALL, both IFU paths, LOOP and dynamic uniform cache reuse passed\n";
            {
                constexpr std::array<std::array<unsigned, 4>, 5> splits{{{1,1,1,1}, {2,2,0,0}, {1,3,0,0}, {3,1,0,0}, {4,0,0,0}}};
                for (auto topology : {Metal::TriangleTopology::List, Metal::TriangleTopology::Strip, Metal::TriangleTopology::Fan}) {
                    for (const auto& split : splits) {
                        Metal::PrimitiveStream stream{device};
                        Pica::PrimitiveAssembler reference{Pica::PipelineRegs::TriangleTopology(unsigned(topology))};
                        unsigned first{};
                        for (unsigned count : split) {
                            if (!count) continue;
                            std::vector<unsigned> expected;
                            for (unsigned i = first; i < first + count; ++i) {
                                Pica::OutputVertex v{};
                                v.pos.x = Pica::f24::FromFloat32(float(i));
                                reference.SubmitVertex(v, [&](const auto& a, const auto& b, const auto& c) {
                                    expected.insert(expected.end(), {unsigned(a.pos.x.ToFloat32()), unsigned(b.pos.x.ToFloat32()), unsigned(c.pos.x.ToFloat32())});
                                });
                            }
                            auto assembled = geometry.Append(stream, captured, 4, topology, count, 0, first, {}, 0, true);
                            Check(assembled.vertex_count == expected.size(), "Native stream count differs from PICA assembly");
                            if (!expected.empty()) Verify(Download(transfers, scheduler, device, assembled), expected, 1.f);
                            const auto saved = geometry.SaveStream(stream);
                            stream.Reset();
                            geometry.Append(stream,captured,4,topology,1,0,3,{},0,true);
                            geometry.LoadStream(stream,saved);
                            const auto restored = geometry.SaveStream(stream);
                            Check(restored.history==saved.history && restored.pending==saved.pending &&
                                  restored.topology==saved.topology && restored.needs_reset==saved.needs_reset,
                                  "Native List/Strip/Fan primitive snapshot lost GPU history or parity");
                            scheduler.Wait(scheduler.Submit());
                            first += count;
                        }
                        stream.Reset();
                        Check(geometry.Append(stream, captured, 4, topology, 1, 0, 3, {}, 0, true).vertex_count == 0,
                              "Native primitive restart leaked history");
                    }
                    for (unsigned chunk : {1U, 2U, 3U, 5U, 63U, 64U, 257U}) {
                        Metal::PrimitiveStream stream{device};
                        Pica::PrimitiveAssembler reference{Pica::PipelineRegs::TriangleTopology(unsigned(topology))};
                        const unsigned width = chunk == 63 ? 2 : chunk == 64 ? 4 : 1;
                        for (unsigned first = 0; first < 257;) {
                            const auto count = std::min(chunk, 257 - first);
                            auto indices = input_stream.Allocate(count * width);
                            std::memset(indices.data, 0, indices.size);
                            std::vector<unsigned> expected;
                            for (unsigned element = 0; element < count; ++element) {
                                const auto index = (first + element) * 3 % 4;
                                static_cast<std::uint8_t*>(indices.data)[element * width] = std::uint8_t(index);
                                Pica::OutputVertex vertex{};
                                vertex.pos.x = Pica::f24::FromFloat32(float(index));
                                reference.SubmitVertex(vertex, [&](const auto& a, const auto& b, const auto& c) {
                                    expected.insert(expected.end(), {unsigned(a.pos.x.ToFloat32()), unsigned(b.pos.x.ToFloat32()), unsigned(c.pos.x.ToFloat32())});
                                });
                            }
                            const auto assembled = geometry.Append(stream, captured, 4, topology, count, 0, 0, indices, width, true);
                            Check(assembled.vertex_count == expected.size(), "Parallel stream count differs from the original PICA assembler");
                            if (!expected.empty()) {
                                const auto values = Download(transfers, scheduler, device, assembled);
                                for (unsigned triangle = 0; triangle < values.size(); triangle += 3) {
                                    const std::vector<Metal::CapturedVertex> vertices{values.begin() + triangle, values.begin() + triangle + 3};
                                    Verify(vertices, std::span{expected}.subspan(triangle, 3), expected[triangle] & 1 ? -1.f : 1.f);
                                }
                            }
                            scheduler.Wait(scheduler.Submit());
                            first += count;
                        }
                    }
                }
                Pica::RegsInternal immediate_regs{};
                Pica::ShaderSetup immediate_setup;
                Metal::Test::MovVertex(immediate_regs, immediate_setup);
                const auto immediate_program = vertex_runtime.Vertex(immediate_regs, immediate_setup);
                Metal::PrimitiveStream immediate_stream{device};
                const auto* input_bytes = static_cast<const std::byte*>(inputs.data);
                for (unsigned i = 0; i < 4; ++i) {
                    auto input = input_stream.Allocate(16 * sizeof(std::array<float, 4>));
                    std::memcpy(input.data, input_bytes + i * input.size, input.size);
                    const auto value = geometry.Capture(*immediate_program, input, 1, 0, real_uniforms);
                    const auto assembled = geometry.Append(immediate_stream, value, 1, Metal::TriangleTopology::Fan, 1, 0, 0, {}, 0, true);
                    if (i >= 2) Verify(Download(transfers, scheduler, device, assembled), std::array<unsigned,3>{0,i-1,i}, 1.f);
                    else Check(assembled.vertex_count == 0, "Immediate stream emitted an incomplete triangle");
                    scheduler.Wait(scheduler.Submit());
                }
                for (unsigned width : {1U,2U,4U}) {
                    Metal::PrimitiveStream stream{device};
                    for (unsigned element = 0; element < 3; ++element) {
                        auto index = input_stream.Allocate(width);
                        std::memset(index.data, 0, width);
                        *static_cast<std::uint8_t*>(index.data) = 3 - element;
                        const auto assembled = geometry.Append(stream, captured, 4, Metal::TriangleTopology::List, 1, 0, 0, index, width, true);
                        if (element == 2) Verify(Download(transfers, scheduler, device, assembled), std::array<unsigned,3>{3,2,1}, -1.f);
                    }
                }
                Metal::PrimitiveStream invalid_stream{device};
                auto bad = input_stream.Allocate(1);
                *static_cast<std::uint8_t*>(bad.data) = 255;
                geometry.Append(invalid_stream, captured, 4, Metal::TriangleTopology::List, 1, 0, 0, bad, 1, true);
                const auto clipped = geometry.Append(invalid_stream, captured, 4, Metal::TriangleTopology::List, 2, 0, 1, {}, 0, true);
                for (const auto& v : Download(transfers, scheduler, device, clipped))
                    Check(v.position[2] == 2 && v.clip[0] < 0, "Invalid stream index did not clip the whole primitive");
                std::cout << "Parallel GPU primitive history matches PICA List/Strip/Fan across 36 split patterns/submissions, including 257 indexed elements and multiple workgroups; immediate MOV capture, restart, save/load parity and invalid clipping passed\n";
            }
            auto fan = geometry.Assemble(captured, 4, Metal::TriangleTopology::Fan,
                                         4, 0, 0, {}, 0, true);
            constexpr std::array<unsigned, 6> fan_order{0, 1, 2, 0, 2, 3};
            Verify(Download(transfers, scheduler, device, fan), fan_order, 1.0f);
            auto strip = geometry.Assemble(captured, 4, Metal::TriangleTopology::Strip,
                                           4, 0, 0, {}, 0, true);
            constexpr std::array<unsigned, 6> strip_order{0, 1, 2, 2, 1, 3};
            Verify(Download(transfers, scheduler, device, strip), strip_order, 1.0f);
            constexpr std::array<unsigned, 3> indexed_order{3, 2, 1};
            for (unsigned width : {1U, 2U, 4U}) {
                auto indices = input_stream.Allocate(3 * width);
                std::memset(indices.data, 0, indices.size);
                auto* bytes = static_cast<std::uint8_t*>(indices.data);
                for (unsigned i = 0; i < 3; ++i) {
                    bytes[i * width] = indexed_order[i];
                }
                auto indexed = geometry.Assemble(captured, 4, Metal::TriangleTopology::List,
                                                  3, 0, 0, indices, width, true);
                Verify(Download(transfers, scheduler, device, indexed), indexed_order, -1.0f);
            }
            auto invalid_indices = input_stream.Allocate(3);
            const std::array<std::uint8_t, 3> invalid_values{255, 1, 0};
            std::memcpy(invalid_indices.data, invalid_values.data(), invalid_values.size());
            auto invalid = geometry.Assemble(captured, 4, Metal::TriangleTopology::List,
                                             3, 0, 0, invalid_indices, 1, true);
            for (const auto& vertex : Download(transfers, scheduler, device, invalid)) {
                Check(vertex.position[2] == 2.0f && vertex.clip[0] < 0.0f && vertex.clip[1] < 0.0f,
                      "Invalid guest indices were not clipped before reading capture data");
            }
            // Reusing completed capture storage must not overwrite vertices still referenced by callers.
            auto repeat = geometry.Assemble(captured, 4, Metal::TriangleTopology::Fan,
                                            4, 0, 0, {}, 0, true);
            Verify(Download(transfers, scheduler, device, repeat), fan_order, 1.0f);

            // Render through native Metal 4 as well: no window, application or ROM is involved.
            auto raster_inputs = input_stream.Allocate(3 * 16 * sizeof(std::array<float, 4>));
            std::memset(raster_inputs.data, 0, raster_inputs.size);
            auto* raster_registers = static_cast<std::array<float, 4>*>(raster_inputs.data);
            constexpr std::array<std::array<float, 4>, 3> positions{{
                {-1.0f, -1.0f, -0.5f, 1.0f}, {3.0f, -1.0f, -0.5f, 1.0f}, {-1.0f, 3.0f, -0.5f, 1.0f},
            }};
            for (unsigned i = 0; i < 3; ++i) {
                raster_registers[i * 16] = positions[i];
                raster_registers[i * 16 + 1] = {0.0f, 0.0f, 0.0f, 1.0f};
                raster_registers[i * 16 + 2] = {1.0f, 0.0f, 1.0f, 1.0f};
            }
            auto raw_raster = device.CreateBuffer(3 * 48, true);
            for (unsigned i = 0; i < 3; ++i) {
                auto* bytes = static_cast<std::byte*>(raw_raster.contents) + i * 48;
                std::memcpy(bytes, &raster_registers[i * 16], 16);
                std::memcpy(bytes + 16, &raster_registers[i * 16 + 1], 16);
                std::memcpy(bytes + 32, &raster_registers[i * 16 + 2], 16);
            }
            const std::array<Metal::BufferSlice, 1> raster_raw_slices{{{raw_raster, 0, raw_raster.length, raw_raster.contents, {}}}};
            auto raster_fetched = fetch.Fetch(raster_raw_slices, raw_attributes, 3);
            auto raster_capture = geometry.Capture(*real_vertex, raster_fetched, 3, 0, real_uniforms);
            auto raster_vertices = geometry.Assemble(raster_capture, 3, Metal::TriangleTopology::List,
                                                     3, 0, 0, {}, 0, true);
            auto fragment = Metal::TranslateGuestGLSL(
                "layout(location=1) in vec4 primary_color; layout(location=0) out vec4 color;"
                "void main() { color = primary_color; }", Metal::ShaderStage::Fragment,
                Metal::RasterizerShaderBindings());
            auto fragment_library = device.CompileLibrary(fragment.source, "Metal 4 raster test");
            Metal::PipelineCache pipelines{device};
            std::uint64_t generic_render_baseline{};
            if (render_corpus) {
                const auto ready = pipelines.Prewarm(keep_loading);
                Check(ready.prepared && !ready.failed && !ready.cancelled && !ready.budget_limited,
                      "Generic native render corpus manifests failed to reconstruct");
                generic_render_baseline = pipelines.Stats().render_builds;
            }
            const auto vertex_function = geometry.RasterVertexFunction();
            Metal::RenderPipelineDescription pipeline_description{
                vertex_function.library, "raster_vertex", fragment_library, fragment.entry_point,
                MTLPixelFormatRGBA8Unorm, {}};
            auto pipeline = pipelines.Render(pipeline_description);
            Check(pipelines.Render(pipeline_description) == pipeline,
                  "Native pipeline cache did not reuse identical state");
            auto depth_state = pipelines.DepthStencil({});
            Check(pipelines.DepthStencil({}) == depth_state,
                  "Native depth/stencil cache did not reuse identical state");
            auto* texture_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                           width:32 height:32 mipmapped:NO];
            texture_descriptor.storageMode = MTLStorageModePrivate;
            texture_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            auto target = device.CreateTexture(texture_descriptor);
            const std::array<Metal::BufferBinding, 1> raster_buffers{{
                {raster_vertices.storage.buffer, raster_vertices.storage.offset, 3},
            }};
            auto table = scheduler.MakeArguments(raster_buffers);
            scheduler.KeepAlive(pipeline);
            scheduler.KeepAlive(depth_state);
            MTL4RenderPassDescriptor* pass = [[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture = target;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
            auto encoder = scheduler.BeginRender(pass);
            [encoder setRenderPipelineState:pipeline];
            [encoder setDepthStencilState:depth_state];
            [encoder setArgumentTable:table atStages:MTLRenderStageVertex];
            [encoder setViewport:MTLViewport{0, 0, 32, 32, 0, 1}];
            [encoder setScissorRect:MTLScissorRect{0, 0, 32, 32}];
            [encoder setCullMode:MTLCullModeNone];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            scheduler.EndEncoding();
            auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
            for (std::size_t i = 0; i < pixels.size(); i += 4) {
                Check(pixels[i] == std::byte{255} && pixels[i + 1] == std::byte{0} &&
                      pixels[i + 2] == std::byte{255} && pixels[i + 3] == std::byte{255},
                      "Native Metal 4 rasterization, clip distances or vertex/fragment linkage is incorrect");
            }
            for (std::uint8_t mask = 0; mask < 16; ++mask) {
                pipeline_description.blend.color_write_mask = mask;
                auto masked_pipeline = pipelines.Render(pipeline_description);
                scheduler.KeepAlive(masked_pipeline);
                scheduler.KeepAlive(depth_state);
                auto masked_table = scheduler.MakeArguments(raster_buffers);
                pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 0.125);
                auto masked_encoder = scheduler.BeginRender(pass);
                [masked_encoder setRenderPipelineState:masked_pipeline];
                [masked_encoder setDepthStencilState:depth_state];
                [masked_encoder setArgumentTable:masked_table atStages:MTLRenderStageVertex];
                [masked_encoder setViewport:MTLViewport{0, 0, 32, 32, 0, 1}];
                [masked_encoder setScissorRect:MTLScissorRect{0, 0, 32, 32}];
                [masked_encoder setCullMode:MTLCullModeNone];
                [masked_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                scheduler.EndEncoding();
                const auto masked = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                constexpr std::array<unsigned, 4> background{64, 128, 191, 32};
                constexpr std::array<unsigned, 4> foreground{255, 0, 255, 255};
                for (std::size_t channel = 0; channel < 4; ++channel) {
                    const auto expected = mask & (1 << channel) ? foreground[channel] : background[channel];
                    Check(std::to_integer<unsigned>(masked[channel]) == expected,
                          "PICA channel write mask was not translated correctly to Metal");
                }
            }
            // Two overlapping triangles in ONE draw must see each other's writes. A sampled
            // framebuffer snapshot would fail XOR, invert and several other logic operations.
            auto overlap = input_stream.Allocate(6 * sizeof(Metal::CapturedVertex));
            std::memcpy(overlap.data, raster_vertices.storage.data, 3 * sizeof(Metal::CapturedVertex));
            std::memcpy(static_cast<std::byte*>(overlap.data) + 3 * sizeof(Metal::CapturedVertex),
                        raster_vertices.storage.data, 3 * sizeof(Metal::CapturedVertex));
            const std::array<Metal::BufferBinding, 1> overlap_buffers{{
                {overlap.buffer, overlap.offset, 3},
            }};
            constexpr std::array formats{Metal::FragmentColorFormat::RGBA8, Metal::FragmentColorFormat::RGB8,
                Metal::FragmentColorFormat::RGB5A1, Metal::FragmentColorFormat::RGB565,
                Metal::FragmentColorFormat::RGBA4};
            constexpr std::array<std::array<unsigned, 4>, 5> channel_masks{{
                {255, 255, 255, 255}, {255, 255, 255, 0}, {31, 31, 31, 1},
                {31, 63, 31, 0}, {15, 15, 15, 15},
            }};
            constexpr std::array<unsigned, 4> source_bytes{85, 170, 204, 51};
            constexpr std::array<unsigned, 4> dest_bytes{80, 144, 208, 96};
            for (std::size_t format = 0; format < formats.size(); ++format) {
                for (std::uint8_t op = 0; op < 16; ++op) {
                    const auto native_fragment = Metal::TranslateRasterizerFragmentGLSL(
                        "layout(location=0) out vec4 color; void main() {"
                        "color = vec4(85.0,170.0,204.0,51.0) / 255.0; }",
                        {true, op, formats[format]});
                    const auto native_library = device.CompileLibrary(native_fragment.source, "PICA logic GPU test");
                    pipeline_description.fragment_library = native_library;
                    pipeline_description.fragment_entry = native_fragment.entry_point;
                    pipeline_description.blend = {};
                    pipeline_description.blend.logic_operation = op;
                    auto logic_pipeline = pipelines.Render(pipeline_description);
                    auto logic_table = scheduler.MakeArguments(overlap_buffers);
                    scheduler.KeepAlive(logic_pipeline);
                    scheduler.KeepAlive(depth_state);
                    pass.colorAttachments[0].clearColor = MTLClearColorMake(
                        dest_bytes[0] / 255.0, dest_bytes[1] / 255.0, dest_bytes[2] / 255.0, dest_bytes[3] / 255.0);
                    auto logic_encoder = scheduler.BeginRender(pass);
                    [logic_encoder setRenderPipelineState:logic_pipeline];
                    [logic_encoder setDepthStencilState:depth_state];
                    [logic_encoder setArgumentTable:logic_table atStages:MTLRenderStageVertex];
                    [logic_encoder setViewport:MTLViewport{0, 0, 32, 32, 0, 1}];
                    [logic_encoder setScissorRect:MTLScissorRect{0, 0, 32, 32}];
                    [logic_encoder setCullMode:MTLCullModeNone];
                    [logic_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                    scheduler.EndEncoding();
                    const auto result = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                    for (std::size_t i = 0; i < result.size(); ++i) {
                        const auto channel = i % 4;
                        const auto mask = channel_masks[format][channel];
                        const auto s = unsigned(std::lround(source_bytes[channel] * mask / 255.0));
                        const auto d = unsigned(std::lround(dest_bytes[channel] * mask / 255.0));
                        const auto once = LogicReference(s, d, op, mask);
                        const auto twice = LogicReference(s, once, op, mask);
                        // NoOp is implemented by the pipeline's write mask and keeps host precision.
                        const auto expected = op == 6 ? dest_bytes[channel]
                            : mask ? unsigned(std::lround(twice * 255.0 / mask)) : 255U;
                        if (std::to_integer<unsigned>(result[i]) != expected) {
                            throw std::runtime_error("Native logic mismatch: format " + std::to_string(format) +
                                " operation " + std::to_string(op) + " channel " + std::to_string(channel) +
                                " expected " + std::to_string(expected) + " got " +
                                std::to_string(std::to_integer<unsigned>(result[i])));
                        }
                    }
                }
            }
            std::cout << "All 16 native logic operations in 5 guest formats with overlapping primitives in one draw passed\n";
            const std::array<float, 4> constants{17.0f / 255, 119.0f / 255, 221.0f / 255, 153.0f / 255};
            auto blend_constants = input_stream.Allocate(sizeof(constants));
            std::memcpy(blend_constants.data, constants.data(), sizeof(constants));
            const std::array<Metal::BufferBinding, 2> blend_buffers{{
                {overlap.buffer, overlap.offset, 3}, {blend_constants.buffer, blend_constants.offset, 22},
            }};
            for (std::uint8_t equation : {3, 4}) {
                for (std::uint8_t sf = 0; sf < 16; ++sf) {
                    for (std::uint8_t df = 0; df < 16; ++df) {
                        Metal::FragmentOperations operations{};
                        operations.rgb = {true, equation, sf, df};
                        operations.alpha = {true, std::uint8_t(7 - equation), df, sf};
                        const auto shader = Metal::TranslateRasterizerFragmentGLSL(
                            "layout(location=0) out vec4 color; void main() {"
                            "color = vec4(85.0,170.0,204.0,51.0) / 255.0; }", operations);
                        pipeline_description.fragment_library = device.CompileLibrary(shader.source, "PICA Min/Max GPU test");
                        pipeline_description.fragment_entry = shader.entry_point;
                        pipeline_description.blend = {};
                        pipeline_description.blend.enabled = true;
                        pipeline_description.blend.emulate_rgb_minmax = true;
                        pipeline_description.blend.emulate_alpha_minmax = true;
                        auto blend_pipeline = pipelines.Render(pipeline_description);
                        auto blend_table = scheduler.MakeArguments(blend_buffers);
                        scheduler.KeepAlive(blend_pipeline);
                        scheduler.KeepAlive(depth_state);
                        pass.colorAttachments[0].clearColor = MTLClearColorMake(
                            dest_bytes[0] / 255.0, dest_bytes[1] / 255.0, dest_bytes[2] / 255.0, dest_bytes[3] / 255.0);
                        auto blend_encoder = scheduler.BeginRender(pass);
                        [blend_encoder setRenderPipelineState:blend_pipeline];
                        [blend_encoder setDepthStencilState:depth_state];
                        [blend_encoder setArgumentTable:blend_table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
                        [blend_encoder setViewport:MTLViewport{0, 0, 32, 32, 0, 1}];
                        [blend_encoder setScissorRect:MTLScissorRect{0, 0, 32, 32}];
                        [blend_encoder setCullMode:MTLCullModeNone];
                        [blend_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                        scheduler.EndEncoding();
                        const auto blended = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                        std::array<double, 4> s{}, d{}, constant{};
                        std::array<unsigned, 4> expected{};
                        for (unsigned c = 0; c < 4; ++c) {
                            s[c] = source_bytes[c] / 255.0;
                            d[c] = dest_bytes[c] / 255.0;
                            constant[c] = constants[c];
                        }
                        for (unsigned primitive = 0; primitive < 2; ++primitive) {
                            const auto src_rgb = BlendFactorReference(sf, s, d, constant);
                            const auto dst_rgb = BlendFactorReference(df, s, d, constant);
                            const auto src_alpha = BlendFactorReference(df, s, d, constant);
                            const auto dst_alpha = BlendFactorReference(sf, s, d, constant);
                            for (unsigned c = 0; c < 4; ++c) {
                                const auto src_factor = c == 3 ? src_alpha[c] : src_rgb[c];
                                const auto dst_factor = c == 3 ? dst_alpha[c] : dst_rgb[c];
                                const auto eq = c == 3 ? 7 - equation : equation;
                                const auto value = eq == 3 ? std::min(s[c] * src_factor, d[c] * dst_factor)
                                                           : std::max(s[c] * src_factor, d[c] * dst_factor);
                                expected[c] = unsigned(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
                            }
                            for (unsigned c = 0; c < 4; ++c) d[c] = expected[c] / 255.0;
                        }
                        for (std::size_t i = 0; i < blended.size(); ++i) {
                            const auto actual = std::to_integer<unsigned>(blended[i]);
                            Check(std::abs(int(actual) - int(expected[i % 4])) <= 1,
                                  "Native PICA Min/Max factor, dynamic constant, independent alpha or primitive order mismatch");
                        }
                    }
                }
            }
            std::cout << "512 native PICA Min/Max cases with all factors, dynamic constants and overlapping primitives passed\n";
            Metal::DrawRuntime blend_draws{device, scheduler, geometry};
            for (std::size_t format = 0; format < formats.size(); ++format) {
                for (std::uint8_t equation = 0; equation < 3; ++equation) {
                    for (std::uint8_t sf = 0; sf < 16; ++sf) {
                        const auto df = std::uint8_t(15 - sf);
                        Metal::FragmentOperations operations{};
                        operations.color_format = formats[format];
                        operations.rgb = {true, equation, sf, df};
                        operations.alpha = {true, std::uint8_t((equation + 1) % 3), df, sf};
                        auto shader = Metal::TranslateRasterizerFragmentGLSL(
                            "layout(location=0) out vec4 color; void main() {"
                            "color = vec4(85.0,170.0,204.0,51.0) / 255.0; }", operations);
                        const auto library = device.CompileLibrary(shader.source, "PICA ordinary blend precision");
                        const Metal::CompiledFragmentShader fragment{{std::move(shader), operations}, library};
                        pass.colorAttachments[0].clearColor = MTLClearColorMake(
                            dest_bytes[0] / 255.0, dest_bytes[1] / 255.0, dest_bytes[2] / 255.0, dest_bytes[3] / 255.0);
                        const std::array<Metal::BufferBinding, 1> constants_binding{{
                            {blend_constants.buffer, blend_constants.offset, 22}}};
                        blend_draws.Draw({}, fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                                         {overlap, 6}, constants_binding, {}, {});
                        const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                        std::array<double, 4> source{}, dest{}, constant{};
                        std::array<unsigned, 4> expected = dest_bytes;
                        for (unsigned c = 0; c < 4; ++c) { source[c] = source_bytes[c] / 255.0; constant[c] = constants[c]; }
                        for (unsigned primitive = 0; primitive < 2; ++primitive) {
                            for (unsigned c = 0; c < 4; ++c) {
                                const auto mask = channel_masks[format][c];
                                dest[c] = mask ? std::round(expected[c] * mask / 255.0) / mask : 1.0;
                            }
                            const auto sr = BlendFactorReference(sf, source, dest, constant);
                            const auto dr = BlendFactorReference(df, source, dest, constant);
                            const auto sa = BlendFactorReference(df, source, dest, constant);
                            const auto da = BlendFactorReference(sf, source, dest, constant);
                            for (unsigned c = 0; c < 4; ++c) {
                                const auto mask = channel_masks[format][c];
                                const auto eq = c == 3 ? (equation + 1) % 3 : equation;
                                const auto sv = source[c] * (c == 3 ? sa[c] : sr[c]);
                                const auto dv = dest[c] * (c == 3 ? da[c] : dr[c]);
                                const auto value = eq == 1 ? sv - dv : eq == 2 ? dv - sv : sv + dv;
                                const auto guest = std::round(std::clamp(value, 0.0, 1.0) * mask);
                                expected[c] = mask ? unsigned(std::lround(guest * 255.0 / mask)) : 255;
                            }
                        }
                        for (std::size_t i = 0; i < pixels.size(); ++i) {
                            if (std::abs(int(std::to_integer<unsigned>(pixels[i])) - int(expected[i % 4])) > 1)
                                throw std::runtime_error("Native ordinary blend mismatch: format " + std::to_string(format) +
                                    " equation " + std::to_string(equation) + " source factor " + std::to_string(sf) +
                                    " channel " + std::to_string(i % 4));
                        }
                    }
                }
            }
            std::cout << "240 native Add/Subtract/ReverseSubtract cases with all factors, 5 guest formats, independent alpha and overlapping primitives passed\n";
            Pica::RegsInternal pica_regs{};
            pica_regs.lighting.disable.Assign(1);
            pica_regs.framebuffer.output_merger.logic_op.Assign(Pica::FramebufferRegs::LogicOp::Copy);
            Metal::ShaderRuntime shader_runtime{device};
            std::uint64_t fs_library_baseline{};
            if (prewarm_corpus) {
                const auto result = shader_runtime.Prewarm(keep_loading);
                Check(result.compiled && !result.failed && !result.cancelled && !result.budget_limited,
                      "Native fragment corpus preload failed or skipped source records");
                fs_library_baseline = shader_runtime.DiskStats().library_compiles;
            }
            const auto pica_config = Pica::Shader::FSConfig{pica_regs};
            std::array<std::future<std::shared_ptr<const Metal::CompiledFragmentShader>>, 4> requests;
            for (auto& request : requests) {
                request = std::async(std::launch::async, [&] {
                    @autoreleasepool { return shader_runtime.Fragment(pica_config, {}, Metal::FragmentColorFormat::RGBA8); }
                });
            }
            auto pica_fragment = requests[0].get();
            for (std::size_t i = 1; i < requests.size(); ++i) {
                Check(requests[i].get() == pica_fragment, "Concurrent native shader compilation published different cache entries");
            }
            Check(shader_runtime.Fragment(pica_config, {}, Metal::FragmentColorFormat::RGBA8) == pica_fragment &&
                  shader_runtime.CachedFragmentCount() == 1, "Native shader cache did not reuse a completed PICA program");
            Pica::Shader::UserConfig normal_user{};
            normal_user.use_custom_normal.Assign(1);
            auto lit_config = pica_config;
            lit_config.lighting.enable.Assign(1);
            lit_config.lighting.src_num.Assign(1);
            for (auto& tev : lit_config.texture.tev_stages) tev.sources_raw = 0xF000F;
            lit_config.texture.tev_stages[0].sources_raw = 0x10001;
            auto lit_fragment = shader_runtime.Fragment(lit_config, {}, Metal::FragmentColorFormat::RGBA8);
            auto normal_fragment = shader_runtime.Fragment(lit_config, normal_user, Metal::FragmentColorFormat::RGBA8);
            Check(normal_fragment != lit_fragment && normal_fragment->program.shader.source != lit_fragment->program.shader.source &&
                  shader_runtime.CachedFragmentCount() == 3,
                  "Native shader cache ignored the custom normal user configuration");
            shader_runtime.Clear();
            if (prewarm_corpus) {
                const auto result = shader_runtime.Prewarm(keep_loading);
                Check(result.compiled && !result.failed && !result.cancelled && !result.budget_limited,
                      "Native fragment corpus re-preload after explicit RAM clear failed");
                fs_library_baseline = shader_runtime.DiskStats().library_compiles;
            }
            Check(shader_runtime.CachedFragmentCount() == 0 && pica_fragment->Library(),
                  "Native shader cache clear invalidated a caller-held program");
            bool rejected_invalid_shader{};
            try {
                auto invalid_config = pica_config;
                invalid_config.lighting.src_num.Assign(9);
                shader_runtime.Fragment(invalid_config, {}, Metal::FragmentColorFormat::RGBA8);
            } catch (const std::invalid_argument&) { rejected_invalid_shader = true; }
            Check(rejected_invalid_shader && shader_runtime.CachedFragmentCount() == 0,
                  "A failed shader compilation published an incomplete native cache entry");
            pipeline_description.fragment_library = pica_fragment->Library();
            pipeline_description.fragment_entry = pica_fragment->program.shader.entry_point;
            pipeline_description.blend = {};
            Metal::DrawRuntime draw_runtime{device, scheduler, geometry};
            std::uint64_t pica_render_baseline{};
            if (render_corpus) {
                const auto ready = draw_runtime.Prewarm(keep_loading);
                Check(ready.prepared && !ready.failed && !ready.cancelled && !ready.budget_limited,
                      "PICA native render corpus manifests failed to reconstruct");
                pica_render_baseline = draw_runtime.PipelineStats().render_builds;
            }
            auto fs_data = input_stream.Allocate(sizeof(Pica::Shader::Generator::FSUniformData));
            auto* fs_uniform = static_cast<Pica::Shader::Generator::FSUniformData*>(fs_data.data);
            std::memset(fs_uniform, 0, sizeof(*fs_uniform));
            fs_uniform->depth_offset = 0.5f;
            pica_regs.framebuffer.framebuffer.allow_color_write.Assign(1);
            pica_regs.framebuffer.output_merger.depth_color_mask = 15 << 8;
            const auto draw_state = Metal::BuildPicaDrawState(pica_regs);
            Check(!draw_state.blend.enabled && draw_state.blend.color_write_mask == 15 &&
                  draw_state.cull == MTLCullModeNone, "Native PICA draw state did not select shader blending or color masks");
            const std::array<Metal::BufferBinding, 1> pica_buffers{{{fs_data.buffer, fs_data.offset, 2}}};
            draw_runtime.Draw(draw_state, *pica_fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                              {overlap, 3}, pica_buffers, {}, {});
            const auto pica_pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
            for (std::size_t i = 0; i < pica_pixels.size(); i += 4) {
                Check(pica_pixels[i] == std::byte{255} && pica_pixels[i + 1] == std::byte{0} &&
                      pica_pixels[i + 2] == std::byte{255} && pica_pixels[i + 3] == std::byte{255},
                      "The real PICA fragment generator, FS uniforms or TEV output failed on native Metal 4");
            }
            std::cout << "Real PICA fragment generator and 1328-byte FS uniforms rendered correctly through native Metal 4\n";
            {
                using C = Pica::TexturingRegs::TextureConfig;
                using Tev = Pica::TexturingRegs::TevStageConfig;
                auto config = pica_config;
                config.texture.texture0_type.Assign(C::Texture2D);
                for (auto& tev : config.texture.tev_stages) tev.sources_raw = 0xF000F;
                auto tev = static_cast<Tev>(config.texture.tev_stages[0]);
                tev.color_source1.Assign(Tev::Source::Texture0);
                tev.alpha_source1.Assign(Tev::Source::Texture0);
                config.texture.tev_stages[0].sources_raw = tev.sources_raw;
                Metal::TextureRuntime runtime{device,scheduler};
                auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:4 height:4 mipmapped:YES];
                descriptor.storageMode=MTLStorageModePrivate; descriptor.usage=MTLTextureUsageShaderRead;
                auto texture=device.CreateTexture(descriptor);
                for (unsigned level=0;level<3;++level) {
                    const unsigned size=4U>>level;
                    auto data=device.CreateBuffer(size*size*4,true);
                    auto* pixels=static_cast<unsigned char*>(data.contents);
                    for (unsigned y=0;y<size;++y) for(unsigned x=0;x<size;++x) {
                        const auto color=SamplerReference((x+.5)/size,(y+.5)/size,level,0,0,false);
                        for(unsigned c=0;c<4;++c) pixels[(y*size+x)*4+c]=unsigned(color[c]);
                    }
                    transfers.Upload(data,{0,size*4,0},{texture,level,0,MTLRegionMake2D(0,0,size,size)});
                }
                const std::array<Metal::TextureBinding,1> images{{{texture,3}}};
                unsigned cases{};
                for(unsigned s=0;s<8;++s) for(unsigned t=0;t<8;++t)
                    for(bool linear:{false,true}) for(bool mip_linear:{false,true}) {
                        Metal::PicaSamplerConfigs settings{};
                        settings[0]={s,t,unsigned(linear),unsigned(linear),unsigned(mip_linear),0,15};
                        auto program=shader_runtime.Fragment(config,{},Metal::FragmentColorFormat::RGBA8,settings);
                        Check(shader_runtime.Fragment(config,{},Metal::FragmentColorFormat::RGBA8,settings)==program,
                              "Native sampler shader configuration was not cached");
                        Metal::Sampler sampler{runtime,{C::TextureFilter(linear),C::TextureFilter(linear),
                            C::TextureFilter(mip_linear),C::WrapMode(s),C::WrapMode(t),0,0,15,0}};
                        const std::array<Metal::SamplerBinding,1> samplers{{{sampler.Handle(),0}}};
                        for(float lod:{-1.0f,.75f,1.25f,7.0f}) for(bool transpose:{false,true}) {
                            auto vertices=input_stream.Allocate(3*sizeof(Metal::CapturedVertex));
                            std::memcpy(vertices.data,overlap.data,vertices.size);
                            auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);
                            for(unsigned i=0;i<3;++i) data[i].texcoord0=transpose ?
                                std::array<float,4>{-.0625f,4*data[i].position[0],0,0} :
                                std::array<float,4>{4*data[i].position[0],-.0625f,0,0};
                            auto uniforms=input_stream.Allocate(sizeof(*fs_uniform));
                            std::memcpy(uniforms.data,fs_uniform,sizeof(*fs_uniform));
                            auto* uniform=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                            uniform->tex_lod_bias[0]=lod;
                            uniform->tex_border_color[0]={17/255.f,83/255.f,149/255.f,211/255.f};
                            const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
                            draw_runtime.Draw(draw_state,*program,pass,{0,0,32,32,0,1},{0,0,32,32},
                                              {vertices,3},buffers,images,samplers);
                            const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},32*4);
                            const double clamped=std::clamp(double(lod),0.0,2.0);
                            const unsigned low=mip_linear ? unsigned(std::floor(clamped)) : unsigned(std::floor(clamped+.5));
                            const unsigned high=std::min(low+1,2U);
                            const double weight=mip_linear ? clamped-low : 0;
                            for(unsigned y=0;y<32;++y) for(unsigned x=0;x<32;++x) {
                                const double u=4*((x+.5)/16-1);
                                const auto a=SamplerReference(transpose ? -.0625 : u,transpose ? u : -.0625,low,s,t,linear);
                                const auto b=SamplerReference(transpose ? -.0625 : u,transpose ? u : -.0625,high,s,t,linear);
                                for(unsigned c=0;c<4;++c) {
                                    const int expected=std::lround(a[c]*(1-weight)+b[c]*weight);
                                    const int actual=std::to_integer<unsigned>(pixels[(y*32+x)*4+c]);
                                    if(std::abs(actual-expected)>1) throw std::runtime_error("Native PICA sampler mismatch s="+
                                        std::to_string(s)+" t="+std::to_string(t)+" linear="+std::to_string(linear)+
                                        " mip="+std::to_string(mip_linear)+" lod="+std::to_string(lod)+" x="+
                                        std::to_string(x)+" channel="+std::to_string(c)+" expected="+
                                        std::to_string(expected)+" actual="+std::to_string(actual));
                                }
                            }
                            ++cases;
                        }
                    }
                std::cout<<cases<<" native PICA sampler draws: 64 wrap pairs, nearest/linear texels and mips, custom per-tap border/alpha, asymmetric negative repeat and LOD bounds passed\n";
                unsigned extended_cases{};
                for(unsigned path=0;path<4;++path) for(unsigned wrap:{1U,4U,5U})
                    for(bool min_linear:{false,true}) for(bool mip_linear:{false,true})
                        for(unsigned bounds=0;bounds<4;++bounds) {
                            const unsigned unit=path==3 ? 0 : path==0 ? 1 : 2;
                            const unsigned coordinate=path==2 ? 1 : unit;
                            auto variant=config;
                            variant.texture.texture0_type.Assign(path==3 ? C::Projection2D : C::Texture2D);
                            variant.texture.texture2_use_coord1.Assign(path==2);
                            auto stage=static_cast<Tev>(variant.texture.tev_stages[0]);
                            stage.color_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0)+unit));
                            stage.alpha_source1.Assign(Tev::Source(unsigned(Tev::Source::Texture0)+unit));
                            variant.texture.tev_stages[0].sources_raw=stage.sources_raw;
                            Metal::PicaSamplerConfigs settings{};
                            const unsigned minimum=bounds==1 ? 1 : bounds==2 ? 15 : 0;
                            const unsigned maximum=bounds==3 ? 0 : bounds==1 ? 1 : 15;
                            settings[unit]={wrap,1,unsigned(min_linear),unsigned(!min_linear),
                                            unsigned(mip_linear),minimum,maximum};
                            auto program=shader_runtime.Fragment(variant,{},Metal::FragmentColorFormat::RGBA8,settings);
                            Metal::Sampler sampler{runtime,{C::TextureFilter(!min_linear),C::TextureFilter(min_linear),
                                C::TextureFilter(mip_linear),C::WrapMode(wrap),C::ClampToBorder,0,minimum,maximum,0}};
                            const std::array<Metal::TextureBinding,1> images{{{texture,unit ? unit+8 : 3}}};
                            const std::array<Metal::SamplerBinding,1> samplers{{{sampler.Handle(),unit ? unit+5 : 0}}};
                            for(float lod:{-1.f,.75f,1.25f}) for(bool transpose:{false,true}) {
                                auto vertices=input_stream.Allocate(3*sizeof(Metal::CapturedVertex));
                                std::memcpy(vertices.data,overlap.data,vertices.size);
                                auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);
                                for(unsigned i=0;i<3;++i) {
                                    data[i].texcoord0={23,29,0,0}; data[i].texcoord1={31,37,0,0};
                                    data[i].texcoord2={41,43,0,0}; data[i].clip[2]=path==3 ? 2 : 1;
                                    const float divisor=path==3 ? 2 : 1;
                                    const std::array<float,4> uv=transpose ?
                                        std::array<float,4>{-.0625f*divisor,4*data[i].position[0]*divisor,0,0} :
                                        std::array<float,4>{4*data[i].position[0]*divisor,-.0625f*divisor,0,0};
                                    if(coordinate==0) data[i].texcoord0=uv;
                                    if(coordinate==1) data[i].texcoord1=uv;
                                    if(coordinate==2) data[i].texcoord2=uv;
                                }
                                auto uniforms=input_stream.Allocate(sizeof(*fs_uniform));
                                std::memcpy(uniforms.data,fs_uniform,sizeof(*fs_uniform));
                                auto* uniform=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                                uniform->tex_lod_bias={7,7,7}; uniform->tex_lod_bias[unit]=lod;
                                for(unsigned i=0;i<3;++i) uniform->tex_border_color[i]={1,0,1,0};
                                uniform->tex_border_color[unit]={17/255.f,83/255.f,149/255.f,211/255.f};
                                const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
                                draw_runtime.Draw(draw_state,*program,pass,{0,0,32,32,0,1},{0,0,32,32},
                                                  {vertices,3},buffers,images,samplers);
                                const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},32*4);
                                const double clamped=std::clamp(double(lod),double(std::min(minimum,2U)),
                                                                                         double(std::min(maximum,2U)));
                                const unsigned low=mip_linear ? unsigned(std::floor(clamped)) : unsigned(std::floor(clamped+.5));
                                const unsigned high=std::min(low+1,2U);
                                const double weight=mip_linear ? clamped-low : 0;
                                const bool linear=clamped<=0 ? !min_linear : min_linear;
                                for(unsigned y=0;y<32;++y) for(unsigned x=0;x<32;++x) {
                                    const double u=4*((x+.5)/16-1);
                                    const auto a=SamplerReference(transpose ? -.0625 : u,transpose ? u : -.0625,low,wrap,1,linear);
                                    const auto b=SamplerReference(transpose ? -.0625 : u,transpose ? u : -.0625,high,wrap,1,linear);
                                    for(unsigned c=0;c<4;++c) {
                                        const int expected=std::lround(a[c]*(1-weight)+b[c]*weight);
                                        const int actual=std::to_integer<unsigned>(pixels[(y*32+x)*4+c]);
                                        if(std::abs(actual-expected)>1) throw std::runtime_error("Extended native sampler mismatch path="+
                                            std::to_string(path)+" wrap="+std::to_string(wrap)+" min="+std::to_string(min_linear)+
                                            " bounds="+std::to_string(bounds)+" lod="+std::to_string(lod)+" x="+
                                            std::to_string(x)+" channel="+std::to_string(c)+" expected="+
                                            std::to_string(expected)+" actual="+std::to_string(actual));
                                    }
                                }
                                ++extended_cases;
                            }
                        }
                std::cout<<extended_cases<<" extended native PICA sampler draws: units1/2, coord1 alias, projection division, distinct min/mag, LOD min/max beyond physical mip levels and poisoned unused coordinates/uniforms passed\n";
                unsigned normal_cases{};
                for(unsigned wrap:{1U,4U,5U}) for(bool linear:{false,true})
                    for(bool projection:{false,true}) {
                        auto variant=config;
                        variant.lighting.enable.Assign(1); variant.lighting.src_num.Assign(1);
                        variant.lighting.lights[0].directional.Assign(1);
                        variant.texture.texture0_type.Assign(projection ? C::Projection2D : C::Texture2D);
                        auto stage=static_cast<Tev>(variant.texture.tev_stages[0]);
                        stage.color_source1.Assign(Tev::Source::PrimaryFragmentColor);
                        stage.alpha_source1.Assign(Tev::Source::PrimaryFragmentColor);
                        variant.texture.tev_stages[0].sources_raw=stage.sources_raw;
                        Pica::Shader::UserConfig user{}; user.use_custom_normal.Assign(1);
                        Metal::PicaSamplerConfigs settings{};
                        settings[0]={wrap,1,unsigned(linear),unsigned(linear),1,0,15};
                        auto program=shader_runtime.Fragment(variant,user,Metal::FragmentColorFormat::RGBA8,settings);
                        Check(program->program.shader.source.find("metalSampleNormal")!=std::string::npos,
                              "Custom normal map did not use native per-tap sampler handling");
                        Metal::Sampler sampler{runtime,{C::TextureFilter(linear),C::TextureFilter(linear),
                            C::Linear,C::WrapMode(wrap),C::ClampToBorder,0,0,15,0}};
                        const std::array<Metal::TextureBinding,1> images{{{texture,13}}};
                        const std::array<Metal::SamplerBinding,1> samplers{{{sampler.Handle(),9}}};
                        for(unsigned level:{0U,1U}) for(bool transpose:{false,true}) {
                            const float scale=level ? 8 : 4;
                            auto vertices=input_stream.Allocate(3*sizeof(Metal::CapturedVertex));
                            std::memcpy(vertices.data,overlap.data,vertices.size);
                            auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);
                            for(unsigned i=0;i<3;++i) {
                                data[i].texcoord0=transpose ? std::array<float,4>{-.0625f,scale*data[i].position[0],0,0} :
                                    std::array<float,4>{scale*data[i].position[0],-.0625f,0,0};
                                data[i].clip[2]=2; // Normal-map UVs must remain unprojected.
                                data[i].view={0,0,-1,0}; data[i].quaternion={0,0,0,1};
                            }
                            auto uniforms=input_stream.Allocate(sizeof(*fs_uniform));
                            std::memcpy(uniforms.data,fs_uniform,sizeof(*fs_uniform));
                            auto* uniform=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                            uniform->tex_lod_bias={7,7,7}; // Normal-map implicit LOD must ignore guest texture bias.
                            uniform->tex_border_color[0]={17/255.f,83/255.f,149/255.f,211/255.f};
                            uniform->light_src[0].position={0,0,-1}; uniform->light_src[0].diffuse={1,1,1};
                            const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
                            draw_runtime.Draw(draw_state,*program,pass,{0,0,32,32,0,1},{0,0,32,32},
                                              {vertices,3},buffers,images,samplers);
                            const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},32*4);
                            for(unsigned y=0;y<32;++y) for(unsigned x=0;x<32;++x) {
                                const double u=scale*((x+.5)/16-1);
                                const auto sample=SamplerReference(transpose ? -.0625 : u,transpose ? u : -.0625,level,wrap,1,linear);
                                const int expected=std::lround(std::clamp(255-2*sample[2],0.0,255.0));
                                for(unsigned c=0;c<4;++c) {
                                    const int actual=std::to_integer<unsigned>(pixels[(y*32+x)*4+c]);
                                    const int channel_expected=c==3 ? 255 : expected;
                                    if(std::abs(actual-channel_expected)>1) throw std::runtime_error("Native normal sampler mismatch wrap="+
                                        std::to_string(wrap)+" linear="+std::to_string(linear)+" projection="+
                                        std::to_string(projection)+" level="+std::to_string(level)+" x="+
                                        std::to_string(x)+" expected="+std::to_string(channel_expected)+" actual="+std::to_string(actual));
                                }
                            }
                            ++normal_cases;
                        }
                    }
                std::cout<<normal_cases<<" native custom-normal sampler draws: per-tap border/asymmetric filtering, both axes, implicit mip LOD without guest bias, unprojected UVs and independent diffuse-lighting pixels passed\n";
                auto* single_descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:4 height:4 mipmapped:NO];
                single_descriptor.storageMode=MTLStorageModePrivate; single_descriptor.usage=MTLTextureUsageShaderRead;
                auto single_level=device.CreateTexture(single_descriptor);
                transfers.CopyTexture({texture,0,0,MTLRegionMake2D(0,0,4,4)},
                                      {single_level,0,0,MTLRegionMake2D(0,0,4,4)});
                unsigned native_comparisons{};
                for(bool min_linear:{false,true}) for(bool mip_linear:{false,true})
                    for(unsigned bounds=0;bounds<4;++bounds) for(float lod:{-1.f,.75f,1.25f})
                        for(bool one_level:{false,true}) {
                        std::array<std::array<unsigned,4>,2> results{};
                        for(unsigned manual=0;manual<2;++manual) {
                            const unsigned minimum=bounds==1 ? 1 : bounds==2 ? 15 : 0;
                            const unsigned maximum=bounds==3 ? 0 : bounds==1 ? 1 : 15;
                            Metal::PicaSamplerConfigs settings{};
                            settings[0]={manual,manual,unsigned(min_linear),unsigned(!min_linear),
                                            unsigned(mip_linear),minimum,maximum};
                            auto program=shader_runtime.Fragment(config,{},Metal::FragmentColorFormat::RGBA8,settings);
                            Metal::Sampler sampler{runtime,{C::TextureFilter(!min_linear),C::TextureFilter(min_linear),
                                C::TextureFilter(mip_linear),C::WrapMode(manual),C::WrapMode(manual),0,minimum,maximum,0}};
                            const std::array<Metal::SamplerBinding,1> samplers{{{sampler.Handle(),0}}};
                            const std::array<Metal::TextureBinding,1> images{{{one_level ? single_level : texture,3}}};
                            auto vertices=input_stream.Allocate(3*sizeof(Metal::CapturedVertex));
                            std::memcpy(vertices.data,overlap.data,vertices.size);
                            auto* data=static_cast<Metal::CapturedVertex*>(vertices.data);
                            for(unsigned i=0;i<3;++i) data[i].texcoord0={.49609375f+.125f*data[i].position[0],.5f,0,0};
                            auto uniforms=input_stream.Allocate(sizeof(*fs_uniform));
                            std::memcpy(uniforms.data,fs_uniform,sizeof(*fs_uniform));
                            auto* uniform=static_cast<Pica::Shader::Generator::FSUniformData*>(uniforms.data);
                            uniform->tex_lod_bias[0]=lod+5;
                            uniform->tex_border_color[0]={17/255.f,83/255.f,149/255.f,211/255.f};
                            const std::array<Metal::BufferBinding,1> buffers{{{uniforms.buffer,uniforms.offset,2}}};
                            draw_runtime.Draw(draw_state,*program,pass,{0,0,32,32,0,1},{0,0,32,32},
                                              {vertices,3},buffers,images,samplers);
                            const auto pixels=transfers.Readback({target,0,0,MTLRegionMake2D(0,0,32,32)},32*4);
                            for(unsigned c=0;c<4;++c) results[manual][c]=std::to_integer<unsigned>(pixels[(16*32+16)*4+c]);
                        }
                        for(unsigned c=0;c<4;++c) if(std::abs(int(results[0][c])-int(results[1][c]))>1)
                            throw std::runtime_error("Native/manual min-mag LOD clamp disagreement min="+std::to_string(min_linear)+
                                " mip="+std::to_string(mip_linear)+" bounds="+std::to_string(bounds)+" one_level="+
                                std::to_string(one_level)+" lod="+
                                std::to_string(lod)+" channel="+std::to_string(c)+" native="+std::to_string(results[0][c])+
                                " manual="+std::to_string(results[1][c]));
                        ++native_comparisons;
                    }
                std::cout<<native_comparisons<<" native/manual sampler comparisons: distinct min/mag and mip filters with LOD min/max clamps, single/multiple physical mip levels matched away from border texels\n";
            }
            std::cout << "Concurrent native shader cache publication, custom-normal keys, clear lifetime and failed-compilation isolation passed\n";
            for (bool reversed : {false, true}) {
                for (bool flipped : {false, true}) {
                    auto vertices = input_stream.Allocate(3 * sizeof(Metal::CapturedVertex));
                    std::memcpy(vertices.data, overlap.data, vertices.size);
                    auto* data = static_cast<Metal::CapturedVertex*>(vertices.data);
                    if (reversed) std::swap(data[1], data[2]);
                    if (flipped) for (unsigned i = 0; i < 3; ++i) data[i].position[1] = -data[i].position[1];
                    for (unsigned mode = 0; mode < 4; ++mode) {
                        auto cull_regs = pica_regs;
                        cull_regs.framebuffer.framebuffer.flip.Assign(!flipped);
                        cull_regs.rasterizer.cull_mode.Assign(Pica::RasterizerRegs::CullMode(mode));
                        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
                        draw_runtime.Draw(Metal::BuildPicaDrawState(cull_regs), *pica_fragment, pass,
                                          {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32}, {vertices, 3}, pica_buffers, {}, {});
                        const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                        const bool visible = mode == 0 || mode == 3 || (mode == 1 ? reversed : !reversed);
                        for (std::size_t i = 0; i < pixels.size(); i += 4) {
                            if (std::to_integer<unsigned>(pixels[i]) != (visible ? 255U : 0U))
                                throw std::runtime_error("Native cull mismatch: reversed " + std::to_string(reversed) +
                                    " flipped " + std::to_string(flipped) + " mode " + std::to_string(mode));
                        }
                    }
                }
            }
            std::cout << "All 4 PICA cull modes with both windings and framebuffer flips passed\n";
            {
                constexpr std::array<NSUInteger, 3> counts{24 * 256 + 128, 384, 512};
                const std::array formats{MTLPixelFormatRG32Float, MTLPixelFormatRG32Float, MTLPixelFormatRGBA32Float};
                std::array<Metal::TextureBinding, 3> lut_bindings{};
                for (NSUInteger lut = 0; lut < 3; ++lut) {
                    const NSUInteger components = lut == 2 ? 4 : 2;
                    const auto bytes = counts[lut] * components * sizeof(float);
                    const auto alignment = [device.Handle() minimumTextureBufferAlignmentForPixelFormat:formats[lut]];
                    auto buffer = device.CreateBuffer(bytes + (alignment - bytes % alignment) % alignment, true);
                    std::memset(buffer.contents, 0, buffer.length);
                    auto* data = static_cast<float*>(buffer.contents);
                    data[(counts[lut] - 1) * components + lut] = float(lut + 1) / 5;
                    lut_bindings[lut] = {device.CreateBufferTexture(buffer, 0, counts[lut], formats[lut]), lut};
                }
                auto shader = Metal::TranslateGuestGLSL(
                    "layout(set=0,binding=3) uniform samplerBuffer lf;"
                    "layout(set=0,binding=4) uniform samplerBuffer rg;"
                    "layout(set=0,binding=5) uniform samplerBuffer rgba;"
                    "layout(location=0) out vec4 color; void main() {"
                    "color=vec4(texelFetch(lf,6271).r,texelFetch(rg,383).g,texelFetch(rgba,511).b,1.0); }",
                    Metal::ShaderStage::Fragment, Metal::RasterizerShaderBindings());
                const auto library = device.CompileLibrary(shader.source, "Native PICA LUT buffer views");
                const Metal::CompiledFragmentShader fragment{{std::move(shader), {}}, library};
                bool rejected_missing{};
                try { draw_runtime.Draw(draw_state, fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32}, {overlap, 3}, {}, {}, {}); }
                catch (const std::invalid_argument&) { rejected_missing = true; }
                Check(rejected_missing, "Native draw accepted missing LUT texture bindings");
                draw_runtime.Draw(draw_state, fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                                  {overlap, 3}, {}, lut_bindings, {});
                const auto pixels = transfers.Readback({target, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
                const std::array<unsigned, 4> expected{51, 102, 153, 255};
                for (std::size_t i = 0; i < pixels.size(); ++i)
                    Check(std::to_integer<unsigned>(pixels[i]) == expected[i % 4],
                          "Native LUT buffer view bounds, texture table or backing-buffer residency mismatch");
                bool rejected{};
                try { device.CreateBufferTexture(lut_bindings[0].texture.buffer, 1, 4, MTLPixelFormatRG32Float); }
                catch (const std::invalid_argument&) { rejected = true; }
                Check(rejected, "Native LUT accepted an unaligned buffer view");
            }
            std::cout << "Native draw runtime, all three PICA LUT buffer views and backing-buffer residency passed\n";
            auto wbuffer_config = pica_config;
            wbuffer_config.framebuffer.depthmap_enable.Assign(Pica::RasterizerRegs::DepthBuffering::WBuffering);
            auto wbuffer_shader = shader_runtime.Fragment(wbuffer_config, {}, Metal::FragmentColorFormat::RGBA8);
            pipeline_description.fragment_library = wbuffer_shader->Library();
            pipeline_description.fragment_entry = wbuffer_shader->program.shader.entry_point;
            auto wbuffer_pipeline = pipelines.Render(pipeline_description);
            auto wbuffer_vertices = input_stream.Allocate(3 * sizeof(Metal::CapturedVertex));
            std::memcpy(wbuffer_vertices.data, overlap.data, wbuffer_vertices.size);
            auto* wvertices = static_cast<Metal::CapturedVertex*>(wbuffer_vertices.data);
            for (unsigned i = 0; i < 3; ++i) {
                for (unsigned c = 0; c < 4; ++c) wvertices[i].position[c] *= 2;
            }
            auto wbuffer_uniforms = input_stream.Allocate(sizeof(*fs_uniform));
            std::memset(wbuffer_uniforms.data, 0, wbuffer_uniforms.size);
            static_cast<Pica::Shader::Generator::FSUniformData*>(wbuffer_uniforms.data)->depth_offset = 0.25f;
            const std::array<Metal::BufferBinding, 2> wbuffer_bindings{{
                {wbuffer_vertices.buffer, wbuffer_vertices.offset, 3}, {wbuffer_uniforms.buffer, wbuffer_uniforms.offset, 2},
            }};
            auto wbuffer_table = scheduler.MakeArguments(wbuffer_bindings);
            auto* wdepth_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                        width:32 height:32 mipmapped:NO];
            wdepth_descriptor.storageMode = MTLStorageModePrivate;
            wdepth_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            auto wdepth = device.CreateTexture(wdepth_descriptor);
            auto wdepth_state = pipelines.DepthStencil({.depth_test = true, .depth_write = true});
            pass.depthAttachment.texture = wdepth;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = 0;
            scheduler.KeepAlive(wbuffer_pipeline);
            scheduler.KeepAlive(wdepth_state);
            auto wbuffer_encoder = scheduler.BeginRender(pass);
            [wbuffer_encoder setRenderPipelineState:wbuffer_pipeline];
            [wbuffer_encoder setDepthStencilState:wdepth_state];
            [wbuffer_encoder setArgumentTable:wbuffer_table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
            [wbuffer_encoder setViewport:MTLViewport{0, 0, 32, 32, 0, 1}];
            [wbuffer_encoder setScissorRect:MTLScissorRect{0, 0, 32, 32}];
            [wbuffer_encoder setCullMode:MTLCullModeNone];
            [wbuffer_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            scheduler.EndEncoding();
            const auto wdepth_pixels = transfers.Readback({wdepth, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
            for (std::size_t i = 0; i < wdepth_pixels.size(); i += sizeof(float)) {
                float value;
                std::memcpy(&value, wdepth_pixels.data() + i, sizeof(value));
                Check(value == 0.5f, "Native PICA W-buffering did not divide by reciprocal clip W");
            }
            pass.depthAttachment.texture = nil;
            auto noop_config = pica_config;
            noop_config.framebuffer.requested_logic_op = Pica::FramebufferRegs::LogicOp::NoOp;
            const auto noop_fragment = shader_runtime.Fragment(noop_config, {}, Metal::FragmentColorFormat::RGBA8);
            Check(!noop_fragment->program.shader.uses_framebuffer_fetch,
                  "NoOp incorrectly requires a color attachment in a depth-only pass");
            auto depth_only_state = draw_state;
            depth_only_state.blend.logic_operation = 6;
            depth_only_state.depth.depth_test = true;
            depth_only_state.depth.depth_write = true;
            pass.colorAttachments[0].texture = nil;
            pass.depthAttachment.texture = wdepth;
            draw_runtime.Draw(depth_only_state, *noop_fragment, pass, {0, 0, 32, 32, 0, 1}, {0, 0, 32, 32},
                              {overlap, 3}, pica_buffers, {}, {});
            const auto depth_only_pixels = transfers.Readback({wdepth, 0, 0, MTLRegionMake2D(0, 0, 32, 32)}, 32 * 4);
            for (std::size_t i = 0; i < depth_only_pixels.size(); i += sizeof(float)) {
                float value;
                std::memcpy(&value, depth_only_pixels.data() + i, sizeof(value));
                Check(value == 0.5f, "Native NoOp depth-only draw dropped a depth write");
            }
            pass.colorAttachments[0].texture = target;
            pass.depthAttachment.texture = nil;
            std::cout << "Native PICA NoOp depth-only draw without color fetch passed\n";
            std::cout << "Real PICA W-buffering with clip W=2 and native depth readback passed\n";
            Metal::FormatConversionRuntime conversions{device, scheduler};
            auto* depth_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                                       width:17 height:11 mipmapped:NO];
            depth_descriptor.storageMode = MTLStorageModePrivate;
            depth_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            auto depth_texture = device.CreateTexture(depth_descriptor);
            auto depth_data = device.CreateBuffer(17 * 11 * sizeof(float), true);
            auto stencil_data = device.CreateBuffer(17 * 11, true);
            auto* depth_values = static_cast<float*>(depth_data.contents);
            auto* stencil_values = static_cast<std::uint8_t*>(stencil_data.contents);
            for (unsigned y = 0; y < 11; ++y) {
                for (unsigned x = 0; x < 17; ++x) {
                    depth_values[y * 17 + x] = float((x * 100003U + y * 200017U) & 0xFFFFFFU) / 16777215.0f;
                    stencil_values[y * 17 + x] = std::uint8_t(x * 17 + y * 3);
                }
            }
            depth_values[1 * 17 + 2] = 0.0f;
            depth_values[9 * 17 + 14] = 1.0f;
            const Metal::TextureRegion full_depth{depth_texture, 0, 0, MTLRegionMake2D(0, 0, 17, 11)};
            transfers.Upload(depth_data, {0, 17 * 4, 0, MTLBlitOptionDepthFromDepthStencil}, full_depth);
            transfers.Upload(stencil_data, {0, 17, 0, MTLBlitOptionStencilFromDepthStencil}, full_depth);
            const Metal::TextureRegion source_region{depth_texture, 0, 0, MTLRegionMake2D(2, 1, 13, 9)};
            constexpr NSUInteger packed_pitch = 60;
            auto packed = device.CreateBuffer(16 + packed_pitch * 9 + 16, true);
            transfers.FillBuffer(packed, NSMakeRange(0, packed.length), 0xA5);
            conversions.DepthStencilToBuffer(source_region, packed, 16, packed_pitch);
            scheduler.Finish();
            const auto* packed_bytes = static_cast<const std::uint8_t*>(packed.contents);
            for (NSUInteger i = 0; i < packed.length; ++i) {
                const bool written = i >= 16 && i < 16 + packed_pitch * 9 &&
                                     (i - 16) % packed_pitch < 13 * 4;
                if (!written) Check(packed_bytes[i] == 0xA5, "Native depth conversion changed output padding or guard bytes");
            }
            for (unsigned y = 0; y < 9; ++y) {
                for (unsigned x = 0; x < 13; ++x) {
                    const auto index = (y + 1) * 17 + x + 2;
                    const auto expected_depth = std::uint32_t(depth_values[index] * 16777215.0f);
                    const auto expected = (expected_depth << 8U) | stencil_values[index];
                    std::uint32_t actual;
                    std::memcpy(&actual, packed_bytes + 16 + y * packed_pitch + x * 4, 4);
                    Check(actual == expected, "Native packed depth/stencil conversion lost an offset, endpoint or edge pixel");
                }
            }
            auto* converted_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                           width:17 height:11 mipmapped:NO];
            converted_descriptor.storageMode = MTLStorageModePrivate;
            converted_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                                        MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
            auto converted_texture = device.CreateTexture(converted_descriptor);
            transfers.ClearColor(converted_texture, MTLClearColorMake(17.0 / 255, 34.0 / 255, 51.0 / 255, 68.0 / 255));
            conversions.DepthStencilToRGBA8(source_region,
                {converted_texture, 0, 0, MTLRegionMake2D(1, 1, 13, 9)});
            const auto converted = transfers.Readback({converted_texture, 0, 0, MTLRegionMake2D(0, 0, 17, 11)}, 17 * 4);
            for (unsigned y = 0; y < 11; ++y) {
                for (unsigned x = 0; x < 17; ++x) {
                    std::array<unsigned, 4> expected{17, 34, 51, 68};
                    if (x >= 1 && x < 14 && y >= 1 && y < 10) {
                        const auto index = y * 17 + x + 1;
                        const auto d = std::uint32_t(depth_values[index] * 16777215.0f);
                        expected = {stencil_values[index], (d >> 16U) & 255U, (d >> 8U) & 255U, d & 255U};
                    }
                    for (unsigned c = 0; c < 4; ++c) {
                        Check(std::to_integer<unsigned>(converted[(y * 17 + x) * 4 + c]) == expected[c],
                              "Native depth/stencil reinterpretation changed channel order, edge pixels or untouched regions");
                    }
                }
            }
            bool rejected_pitch{};
            try { conversions.DepthStencilToBuffer(source_region, packed, 16, 4); }
            catch (const std::invalid_argument&) { rejected_pitch = true; }
            Check(rejected_pitch, "Native depth conversion accepted a short row pitch");
            std::cout << "Native depth/stencil plane upload, packed readback and RGBA8 partial reinterpretation with odd extents, endpoints and guard bytes passed\n";
            std::cout << "Metal 4 GPU vertex capture, list/strip/fan assembly, 8/16/32-bit indices, quaternion correction and invalid-index clipping passed\n";
            std::cout << "Native Metal 4 render pipeline, clip distances, fragment linkage and 32x32 offscreen readback passed\n";
            std::cout << "Native pipeline/depth-state cache reuse and all 16 PICA color-write masks passed\n";
            if (render_corpus) {
                const auto generic = pipelines.Stats(), pica = draw_runtime.PipelineStats();
                Check(generic.prewarm_hits && pica.prewarm_hits &&
                      generic.render_builds == generic_render_baseline && pica.render_builds == pica_render_baseline,
                      "Prepared render corpus performed a new render descriptor/PSO build during GPU requests");
                std::cout << "Full native render manifest corpus: generic ready hits=" << generic.prewarm_hits
                          << " PICA ready hits=" << pica.prewarm_hits
                          << "; 0 render builds during requests; all GPU cases passed\n";
            }
            if (argc == 3) {
                const auto vertex_stats = vertex_runtime.DiskStats();
                const auto fragment_stats = shader_runtime.DiskStats();
                std::cout << "Native source corpus " << argv[2] << ": vertex hits=" << vertex_stats.hits
                          << " translations=" << vertex_stats.translations << " rejected=" << vertex_stats.rejected
                          << "; fragment hits=" << fragment_stats.hits << " translations=" << fragment_stats.translations
                          << " rejected=" << fragment_stats.rejected << '\n';
                if (std::string_view{argv[2]} == "warm") {
                    Check(vertex_stats.hits && fragment_stats.hits && !vertex_stats.translations &&
                          !fragment_stats.translations && !vertex_stats.rejected && !fragment_stats.rejected,
                          "Fresh-process PICA corpus regenerated or rejected persisted source/IR/reflection");
                }
                if (prewarm_corpus) {
                    Check(vertex_stats.prewarm_hits && fragment_stats.prewarm_hits &&
                          !vertex_stats.translations && !fragment_stats.translations &&
                          !vertex_stats.rejected && !fragment_stats.rejected &&
                          vertex_stats.library_compiles == vs_library_baseline &&
                          fragment_stats.library_compiles == fs_library_baseline,
                          "Prewarmed PICA corpus performed additional guest/PICA library compilation");
                    std::cout << "Native prewarmed corpus: VS hits=" << vertex_stats.prewarm_hits
                              << " FS hits=" << fragment_stats.prewarm_hits
                              << "; 0 guest translations/0 PICA library compiles during requests; all GPU cases passed\n";
                }
                Check(device.FlushDiskCache(), "Native corpus pipeline archive publication failed");
            }
            {
                constexpr std::array<unsigned, 3> order{1, 2, 3};
                auto end = geometry.Assemble(captured, 4, Metal::TriangleTopology::List,
                    3, UINT32_MAX - 3, UINT32_MAX - 2, {}, 0, true);
                Verify(Download(transfers, scheduler, device, end), order, -1.f);
                bool overflow{};
                try { geometry.Assemble(captured, 4, Metal::TriangleTopology::List,
                    3, UINT32_MAX - 3, UINT32_MAX - 1, {}, 0, true); }
                catch (const std::invalid_argument&) { overflow = true; }
                Check(overflow, "Primitive assembly accepted a wrapped nonindexed vertex range");
                auto unused = geometry.CompileVertex(std::string{VertexSource} +
                    "\nlayout(set=0,binding=0,std140) uniform unused_block { vec4 unused_values[4]; };\n");
                Check(std::any_of(unused.shader.resources.begin(), unused.shader.resources.end(), [](const auto& resource) {
                    return resource.descriptor_set == 0 && resource.binding == 0 && !resource.active;
                }), "Unused vertex uniform was not represented as inactive reflection metadata");
                const auto inactive_capture = geometry.Capture(unused, inputs, 4, 0, uniforms);
                auto inactive = geometry.Assemble(inactive_capture, 4, Metal::TriangleTopology::List, 3, 0, 1, {}, 0, true);
                Verify(Download(transfers, scheduler, device, inactive), order, -1.f);
                bool capture_overflow{};
                try { geometry.Capture(program, inputs, 4, UINT32_MAX - 1, uniforms); }
                catch (const std::invalid_argument&) { capture_overflow = true; }
                Check(capture_overflow, "Native capture accepted a wrapped first-vertex range");
                std::cout << "Native geometry boundaries: UINT32_MAX last vertex, wrapped assembly/capture rejection and inactive real shader uniform omission preserve every captured position/color/quaternion/clip field passed\n";
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
