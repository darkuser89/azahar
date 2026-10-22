// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#include "mtl_shader_machine.h"
#include "mtl_shaders.h"
#include "mtl_transfer.h"
#include "mtl_input_reference.h"
#include "mtl_vertex_diagnostic.h"
#include <bit>
#include <filesystem>
#include "common/hash.h"
#include "common/logging/log.h"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <utility>
#include <stdexcept>
#include <string>
#include <bitset>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
namespace Metal {
// Opt-in synchronous GPU diagnostics. Legacy requests retain their16-class,
//64-record bounds. "epoch frame" captures every observed VS batch of one
// frontend frame, including duplicates, bounded by512 records and128MiB.
struct ShaderMachineRuntime::CaptureState {
    static constexpr std::uint64_t MaxPayload=16u*1024u*1024u;
    static constexpr std::uint64_t MaxFrameBytes=128u*1024u*1024u;
    std::filesystem::path directory;
    std::set<std::array<std::uint64_t,5>> seen;
    std::uint64_t checked_frame{UINT64_MAX},request{},sequence{};
    std::uint32_t requested_elements{};
    bool request_valid{},frame_mode{},frame_open{},frame_done{};
    unsigned total{};
    std::uint64_t frame_id{},frame_request{},attempted{},selected{},written{},reserved{},written_bytes{};
    std::uint64_t skipped_alias{},skipped_size{},skipped_limit{},skipped_budget{},conversion_records{};
    explicit CaptureState(const char* path):directory{path} {
        std::filesystem::create_directories(directory);
    }
    static void Publish(const std::filesystem::path& path,const void* bytes,std::size_t size) {
        NSError* error=nil;
        auto* data=[NSData dataWithBytes:bytes length:size];
        const auto temporary=std::filesystem::path(path.string()+".part");
        auto* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:temporary.c_str()]];
        if(![data writeToURL:url options:NSDataWritingWithoutOverwriting error:&error])
            throw std::runtime_error("Unable to write requested Metal 4 vertex diagnostic");
        try {std::filesystem::create_hard_link(temporary,path);}
        catch (...) {std::filesystem::remove(temporary);throw;}
        std::filesystem::remove(temporary);
    }
    void CloseFrame() {
        // Only a later observed frame closes this record. Destroying the
        // runtime early leaves no complete-frame manifest to be mistaken for one.
        const auto json=std::string("{\n  \"schema\":1,\n  \"stage\":\"shader_machine_vertex_batches\",\n  \"request\":")+
            std::to_string(frame_request)+",\n  \"frame\":"+std::to_string(frame_id)+
            ",\n  \"attempted\":"+std::to_string(attempted)+",\n  \"selected\":"+std::to_string(selected)+
            ",\n  \"written\":"+std::to_string(written)+",\n  \"reserved_bytes\":"+std::to_string(reserved)+
            ",\n  \"written_bytes\":"+std::to_string(written_bytes)+
            ",\n  \"skipped_alias\":"+std::to_string(skipped_alias)+
            ",\n  \"skipped_size\":"+std::to_string(skipped_size)+
            ",\n  \"skipped_record_limit\":"+std::to_string(skipped_limit)+
            ",\n  \"skipped_byte_budget\":"+std::to_string(skipped_budget)+
            ",\n  \"conversion_records\":"+std::to_string(conversion_records)+
            ",\n  \"complete_observed_stage\":"+(attempted==written && reserved==written_bytes ? "true":"false")+
            ",\n  \"complete_render_frame\":false\n}\n";
        Publish(directory/("frame-"+std::to_string(frame_request)+"-"+
            std::to_string(frame_id)+".json"),json.data(),json.size());
        frame_open=false;frame_done=true;
    }
    bool Select(const ShaderMachineProgram& program,const ShaderMachineUniforms& uniforms,
                std::uint64_t frame,std::uint32_t count,std::uint32_t index_width,
                bool aliases_state,std::uint64_t payload_bytes,bool conversion_metadata) {
        if(frame_done) return false;
        ++sequence;
        if(checked_frame!=frame) {
            if(frame_open) {CloseFrame();return false;}
            checked_frame=frame;
            std::ifstream arm(directory/"capture.request");
            std::uint64_t next{},elements{};bool next_frame_mode=false;
            request_valid=bool(arm>>next) && next;
            arm>>std::ws;
            if(request_valid && !arm.eof()) {
                if(arm.peek()=='f') {
                    std::string mode;arm>>mode;
                    request_valid=mode=="frame";next_frame_mode=request_valid;
                } else request_valid=bool(arm>>elements) && elements<=UINT32_MAX;
                arm>>std::ws;request_valid=request_valid && arm.eof();
            }
            if(request_valid && next!=request) {
                request=next;requested_elements=std::uint32_t(elements);seen.clear();
                frame_mode=next_frame_mode;
                if(frame_mode) {frame_open=true;frame_id=frame;frame_request=request;}
            }
        }
        if(frame_open) {
            ++attempted;
            if(aliases_state) {++skipped_alias;return false;}
            if(payload_bytes>MaxPayload) {++skipped_size;return false;}
            if(selected==512) {++skipped_limit;return false;}
            const auto bytes=payload_bytes+sizeof(VertexDiagnostic::Header)+sizeof(program.code)+
                sizeof(program.swizzle)+sizeof(uniforms)+2u*1824u+
                (conversion_metadata ? sizeof(GeometryEmissionConfig):0u);
            if(bytes>MaxFrameBytes-reserved) {++skipped_budget;return false;}
            ++selected;reserved+=bytes;return true;
        }
        if(aliases_state || payload_bytes>MaxPayload || total>=64 || !request_valid ||
           !request || frame_mode || (requested_elements && count!=requested_elements) || seen.size()>=16)
            return false;
        std::array<std::uint32_t,17> control{uniforms.b};
        std::memcpy(control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
        const std::array<std::uint64_t,5> key{
            Common::ComputeHash64(program.code.data(),sizeof(program.code)),
            Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),
            (std::uint64_t(program.entry)<<32)|program.output_mask,
            Common::ComputeHash64(control.data(),sizeof(control)),
            (std::uint64_t(index_width)<<32)|std::bit_width(count)};
        if(!seen.insert(key).second) return false;
        ++total;return true;
    }
    void Write(const std::vector<std::byte>& bytes,std::uint64_t frame,const GeometryEmissionConfig* config) {
        const auto path=directory/("draw-"+std::to_string(request)+"-"+
            std::to_string(frame)+"-"+std::to_string(sequence)+".vdc");
        // The final frame manifest publishes this pair only after both files
        // exist. VDC version1 stays compatible with existing scalar replayers.
        if(config) Publish(std::filesystem::path(path.string()+".capture.bin"),config,sizeof(*config));
        Publish(path,bytes.data(),bytes.size());
        if(frame_open && frame==frame_id) {
            ++written;written_bytes+=bytes.size()+(config ? sizeof(*config):0u);
            if(config) ++conversion_records;
        }
    }
};

std::bitset<4096> ShaderMachineRuntime::ReachableVertexInstructions(const ShaderMachineProgram& program) {
    struct Frame { unsigned end{},resume{},begin{}; };
    struct State {
        unsigned pc{},call_depth{},branch_depth{},loop_depth{};
        std::array<Frame,4> calls{},loops{};
        std::array<Frame,8> branches{};
    };
    std::bitset<4096> domain;
    if(program.entry>=4096) return domain.set();
    std::set<std::array<unsigned,52>> reached;
    std::vector<State> pending;
    bool exhausted=false;
    const auto enqueue=[&](State state) {
        if(state.pc>=4096) return; // The native PC guard fails before instruction fetch.
        std::array<unsigned,52> key{};unsigned field=0;
        key[field++]=state.pc;key[field++]=state.call_depth;
        key[field++]=state.branch_depth;key[field++]=state.loop_depth;
        const auto stack_key=[&](const auto& stack,unsigned depth) {
            for(unsigned n=0;n<stack.size();++n) {
                if(n<depth) {key[field]=stack[n].end;key[field+1]=stack[n].resume;key[field+2]=stack[n].begin;}
                field+=3;
            }
        };
        stack_key(state.calls,state.call_depth);stack_key(state.branches,state.branch_depth);
        stack_key(state.loops,state.loop_depth);
        if(reached.contains(key)) return;
        // Bounded compiler work. If structural contexts grow too far, preserve
        // the entire native PC domain rather than discarding an unproven PC.
        if(reached.size()==8192) {exhausted=true;return;}
        reached.insert(key);pending.push_back(state);
    };
    const auto push=[](auto& stack,unsigned& depth,Frame frame) {
        if(depth==stack.size()) std::move(stack.begin()+1,stack.end(),stack.begin());
        else ++depth;
        stack[depth-1]=frame;
    };
    const auto advance=[&](State state,unsigned next,unsigned jump,bool breaking=false) {
        unsigned result=jump,call_pc=next,call_result=next,pops=0;bool call_pop=false;
        while(state.call_depth && call_pc==state.calls[state.call_depth-1].end) {
            call_pc=state.calls[--state.call_depth].resume;
            if(++pops<4) call_result=call_pc; // PICA loses the fourth return update.
            call_pop=true;
        }
        if(call_pop) result=call_result;
        if(state.branch_depth && next==state.branches[state.branch_depth-1].end)
            result=state.branches[--state.branch_depth].resume;
        if(state.loop_depth && (breaking || next==state.loops[state.loop_depth-1].end)) {
            const auto frame=state.loops[state.loop_depth-1];
            // Ignore remaining counts and explore both outcomes. This includes
            // every 1..256 uniform loop count, address-dependent condition and
            // nested ring-stack context without unrolling vertex arithmetic.
            if(!breaking) {auto repeat=state;repeat.pc=frame.begin;enqueue(repeat);}
            --state.loop_depth;state.pc=frame.resume;enqueue(state);return;
        }
        state.pc=result;enqueue(state);
    };
    State initial{};initial.pc=program.entry;enqueue(initial);
    while(!pending.empty() && !exhausted) {
        auto state=pending.back();pending.pop_back();domain.set(state.pc);
        const auto word=program.code[state.pc],op=word>>26;
        const unsigned next=state.pc+1,target=(word>>10)&4095u,length=word&255u;
        if(op==0x22) continue; // END stops before any stack boundary is applied.
        if(op==0x20 || op==0x23) {
            if(op==0x23) advance(state,next,next);
            if(state.loop_depth) advance(state,next,next,true);
            continue;
        }
        if(op==0x2a || op==0x2b) {
            if(program.allow_emission) advance(state,next,next);
            continue;
        }
        if(op>=0x24 && op<=0x2d) {
            if(op==0x2c || op==0x2d) {
                advance(state,next,next);advance(state,next,target);continue;
            }
            if(op==0x29) {
                if(target<next || target>=4096) continue;
                push(state.loops,state.loop_depth,{target+1,target+1,next});
                advance(state,next,next);continue;
            }
            const bool branch=op==0x27 || op==0x28;
            if(target+length>4096 || (branch && target<next)) continue;
            const auto apply=[&](State path,bool take) {
                unsigned jump=next;
                if(branch) {
                    if(take) {
                        if(next==target) jump=target+length;
                        else push(path.branches,path.branch_depth,{target,target+length,0});
                    } else jump=target;
                } else if(take && length) {
                    push(path.calls,path.call_depth,{target+length,next,0});jump=target;
                }
                advance(path,next,jump);
            };
            apply(state,true);
            if(op!=0x24) apply(state,false);
            continue;
        }
        const bool arithmetic=op<=0x0f || op==0x12 || op==0x13 ||
                              (op>=0x18 && op<=0x1b) || op==0x2e || op==0x2f || op>=0x30;
        if(op==0x21 || arithmetic) advance(state,next,next);
        // Unsupported opcodes fail before the next instruction/stack advance.
    }
    return exhausted ? domain.set():domain;
}
namespace {
ShaderMachineRuntime::VertexExecution AnalyzeVerticesImpl(const ShaderMachineProgram& program,
    const ShaderMachineUniforms& uniforms,std::uint32_t* unproven_instruction,std::uint32_t* dependency_passes,
    std::array<std::uint32_t,2>* recurrence_mask,std::uint32_t* recurrence_capacity,
    std::array<std::uint32_t,5>* always_written,std::array<std::uint32_t,5>* possibly_written,
    const GeometryEmissionConfig* native_reset_capture,bool* capture_independent,
    bool skip_independent_raw=false) {
    using VertexExecution=ShaderMachineRuntime::VertexExecution;
    if(capture_independent) *capture_independent=false;
    if(dependency_passes) *dependency_passes=0;
    if(recurrence_mask) recurrence_mask->fill(0);
    if(recurrence_capacity) *recurrence_capacity=0;
    if(always_written) always_written->fill(0);
    if(possibly_written) possibly_written->fill(0);
    if(unproven_instruction) *unproven_instruction=program.entry;
    if(program.allow_emission || program.entry>=4096 || program.output_mask>0xffff) return VertexExecution::Ordered;
    using Dependencies=std::bitset<136>;
    struct Frame { unsigned end{},resume{},begin{},remaining{}; };
    struct Path {
        // Output/temporary lanes, address components and condition flags written so far.
        std::array<unsigned,34> written{},may_written{};
        std::array<std::array<Dependencies,4>,34> dependencies{},written_dependencies{};
        std::array<unsigned,64> temporary_writes{};
        Dependencies control_dependencies;
        std::array<Frame,4> calls{},loops{};
        std::array<Frame,8> branches{};
        unsigned call_depth{},branch_depth{},loop_depth{},pc{},next{},jump{},steps{};
        bool breaking{};
    };
    const auto push=[](auto& stack,unsigned& depth,Frame frame) {
        if(depth==stack.size()) std::move(stack.begin()+1,stack.end(),stack.begin());
        else ++depth;
        stack[depth-1]=frame;
    };
    Path initial{};initial.pc=initial.next=initial.jump=program.entry;
    for(unsigned reg=0;reg<34;++reg) for(unsigned component=0;component<4;++component)
        initial.dependencies[reg][component].set(reg*4+component);
    std::vector<Path> pending{initial};
    std::array<unsigned,34> must_write{},may_write{};
    Dependencies control_reads;
    std::array<Dependencies,136> final_dependencies;
    std::array<Dependencies,64> recurrence_operands;
    std::bitset<64> unsupported_recurrence;
    std::array<unsigned,64> maximum_writes{};
    bool ended=false;unsigned total_steps=0;
    // Compiler/dataflow analysis only: no vertex values or arithmetic run on the host.
    // Explore both sides of input-dependent control. Reads must follow a write or address a lane
    // that no invocation changes. Conditional writes require an ordered GPU merge. Untouched
    // lanes retain the common initial state rather than an earlier vertex's result.
    // Exact ring stacks and boundary priority match the native GPU machine.
    // Join only equal PCs and complete active ring-stack contexts. Dependencies and
    // possible writes grow; definite writes shrink. Track written-value dependencies
    // separately from preserved incoming values so a skipped write is not mistaken
    // for a self recurrence. Revisit successors whenever any abstract fact grows.
    // Flow states are looked up only; key order never participates in traversal.
    // Keep every stack/counter word in equality, including after hash collisions.
    struct FlowHash {
        std::size_t operator()(const std::array<unsigned,68>& state) const {
            return Common::ComputeHash64(state.data(),sizeof(state));
        }
    };
    std::unordered_map<std::array<unsigned,68>,Path,FlowHash> reached;
    while(!pending.empty()) {
        auto path=std::move(pending.back());pending.pop_back();
        for(;;) {
            // This is bounded compiler work cached per control-uniform state. Branch
            // combinations can require many short paths even in a small shader.
            if(++total_steps>1048576 || path.steps>=4096) return VertexExecution::Ordered;
            if(path.steps) {
                unsigned result=path.jump,call_pc=path.next,call_result=path.next,pops=0;bool call_pop=false;
                while(path.call_depth && call_pc==path.calls[path.call_depth-1].end) {
                    call_pc=path.calls[--path.call_depth].resume;
                    if(++pops<4) call_result=call_pc;
                    call_pop=true;
                }
                if(call_pop) result=call_result;
                if(path.branch_depth && path.next==path.branches[path.branch_depth-1].end)
                    result=path.branches[--path.branch_depth].resume;
                if(path.loop_depth && (path.breaking || path.next==path.loops[path.loop_depth-1].end)) {
                    auto& frame=path.loops[path.loop_depth-1];
                    if(!path.breaking && --frame.remaining) result=frame.begin;
                    else { result=frame.resume;--path.loop_depth; }
                }
                path.pc=result;
            }
            path.breaking=false;
            if(path.pc>=4096) return VertexExecution::Ordered;
            if(unproven_instruction) *unproven_instruction=path.pc;
            for(unsigned reg=0;reg<34;++reg) path.may_written[reg]|=path.written[reg];
            std::array<unsigned,68> flow{};unsigned field=0;
            flow[field++]=path.pc;flow[field++]=path.call_depth;
            flow[field++]=path.loop_depth;flow[field++]=path.branch_depth;
            const auto stack_key=[&](const auto& stack,unsigned depth) {
                for(unsigned n=0;n<stack.size();++n) {
                    if(n<depth) {flow[field]=stack[n].end;flow[field+1]=stack[n].resume;
                        flow[field+2]=stack[n].begin;flow[field+3]=stack[n].remaining;}
                    field+=4;
                }
            };
            stack_key(path.calls,path.call_depth);stack_key(path.loops,path.loop_depth);
            stack_key(path.branches,path.branch_depth);
            if(reached.size()==8192 && !reached.contains(flow)) return VertexExecution::Ordered;
            const auto [position,fresh]=reached.try_emplace(flow,path);
            if(!fresh) {
                auto& known=position->second;bool changed=false;
                const auto widen=[&](auto& old,const auto& value) {if(old!=value) {old=value;changed=true;}};
                for(unsigned reg=0;reg<34;++reg) {
                    widen(known.written[reg],known.written[reg]&path.written[reg]);
                    widen(known.may_written[reg],known.may_written[reg]|path.may_written[reg]);
                    for(unsigned lane=0;lane<4;++lane) {
                        widen(known.dependencies[reg][lane],known.dependencies[reg][lane]|path.dependencies[reg][lane]);
                        widen(known.written_dependencies[reg][lane],known.written_dependencies[reg][lane]|path.written_dependencies[reg][lane]);
                    }
                }
                for(unsigned lane=0;lane<64;++lane)
                    widen(known.temporary_writes[lane],std::max(known.temporary_writes[lane],path.temporary_writes[lane]));
                widen(known.control_dependencies,known.control_dependencies|path.control_dependencies);
                widen(known.steps,std::max(known.steps,path.steps));
                if(!changed) break;
                path=known;
            }
            const auto word=program.code[path.pc++],op=word>>26;
            const auto target=(word>>10)&4095u,length=word&255u;
            path.next=path.jump=path.pc;++path.steps;
            if(op==0x21) continue;
            if(op==0x22) {
                if(!ended) {must_write=path.written;may_write=path.may_written;}
                else for(unsigned lane=0;lane<path.written.size();++lane) {
                    must_write[lane]&=path.written[lane];may_write[lane]|=path.may_written[lane];
                }
                control_reads|=path.control_dependencies;
                for(unsigned lane=0;lane<64;++lane)
                    maximum_writes[lane]=std::max(maximum_writes[lane],path.temporary_writes[lane]);
                for(unsigned reg=0;reg<34;++reg) for(unsigned component=0;component<4;++component)
                    if(path.may_written[reg]&(reg<32 ? 8u>>component:1u<<component))
                        final_dependencies[reg*4+component]|=path.written_dependencies[reg][component];
                ended=true;break;
            }
            if(op==0x29) {
                if(target<path.pc || target>=4096) return VertexExecution::Ordered;
                path.written[32]|=4;path.dependencies[32][2].reset();path.written_dependencies[32][2].reset();
                push(path.loops,path.loop_depth,{target+1,target+1,path.pc,(uniforms.i[(word>>22)&3u][0]&255u)+1});
                continue;
            }
            if(op==0x20 || op==0x23 || (op>=0x24 && op<=0x28) || op==0x2c || op==0x2d) {
                const bool conditional=op==0x23 || op==0x25 || op==0x28 || op==0x2c;
                if(conditional) {
                    const unsigned mode=(word>>22)&3u,required=mode==2 ? 1:mode==3 ? 2:3;
                    for(unsigned component=0;component<2;++component) if(required&(1u<<component))
                        path.control_dependencies|=path.dependencies[33][component];
                }
                const auto apply=[&](Path& branch,bool take)->bool {
                    if(op==0x20 || op==0x23) {
                        if(take) { if(!branch.loop_depth) return false;branch.breaking=true; }
                    } else if(op==0x2c || op==0x2d) {
                        if(take) branch.jump=target;
                    } else {
                        if(target+length>4096) return false;
                        if(op==0x27 || op==0x28) {
                            if(target<branch.pc) return false;
                            if(take) {
                                if(branch.pc==target) branch.jump=target+length;
                                else push(branch.branches,branch.branch_depth,{target,target+length,0,0});
                            } else branch.jump=target;
                        } else if(take && length) {
                            push(branch.calls,branch.call_depth,{target+length,branch.pc,0,0});branch.jump=target;
                        }
                    }
                    return true;
                };
                if(conditional) {
                    if(pending.size()==128) return VertexExecution::Ordered;
                    auto other=path;
                    if(!apply(other,true) || !apply(path,false)) return VertexExecution::Ordered;
                    pending.push_back(std::move(other));
                } else {
                    bool take=op==0x20 || op==0x24 || bool(uniforms.b&(1u<<((word>>22)&15u)));
                    if(op==0x2d) take=bool(uniforms.b&(1u<<((word>>22)&15u)))!=bool(length&1u);
                    if(!apply(path,take)) return VertexExecution::Ordered;
                }
                continue;
            }
            const bool mad=op>=0x30;
            if(!mad && !(op<=0x0f || op==0x12 || op==0x13 || (op>=0x18 && op<=0x1b) || op==0x2e || op==0x2f))
                return VertexExecution::Ordered;
            const bool inverted=op==0x18 || op==0x19 || op==0x1a || op==0x1b || (mad && op<0x38);
            const auto desc=program.swizzle[word&(mad ? 31u:127u)];
            const auto address=(word>>(mad ? 22:19))&3u;
            const auto src1=mad ? (word>>17)&31u:(word>>(inverted ? 14:12))&(inverted ? 31u:127u);
            const auto src2=mad ? (word>>(inverted ? 12:10))&(inverted ? 31u:127u):(word>>7)&(inverted ? 127u:31u);
            const auto src3=(word>>5)&(inverted ? 127u:31u);
            const unsigned mask=desc&15u;
            unsigned a=mask,b=mask;
            if(!mad) switch(op) {
            case 0x01:a=b=mask ? 14:0;break; // DP3 never consumes w.
            case 0x02:a=b=mask ? 15:0;break;
            case 0x03:case 0x18:a=mask ? 14:0;b=mask ? 15:0;break;
            case 0x04:a=mask&6;b=mask&5;break;
            case 0x19:a=mask&5;b=mask&6;break;
            case 0x05:case 0x06:case 0x0e:case 0x0f:a=mask ? 8:0;b=0;break;
            case 0x07:a=9|(mask&4);b=0;break; // LIT always writes both condition flags.
            case 0x0b:case 0x13:b=0;break;
            case 0x12:a=mask&12;b=0;break;
            case 0x2e:case 0x2f:a=b=12;break;
            default:break;
            }
            const auto source_dependency=[&](unsigned reg,unsigned relative,unsigned source,unsigned component) {
                const unsigned shift=source==0 ? 5:source==1 ? 14:23;
                const unsigned selected=(desc>>(shift+6-component*2))&3u;
                if(reg>=16 && reg<32) return path.dependencies[reg][selected];
                if(reg>=32 && relative) return path.dependencies[32][relative-1];
                return Dependencies{};
            };
            std::array<Dependencies,4> av{},bv{},cv{},value{};
            for(unsigned component=0;component<4;++component) {
                av[component]=source_dependency(src1,(!mad && !inverted) ? address:0,0,component);
                bv[component]=source_dependency(src2,((mad && !inverted) || (!mad && inverted)) ? address:0,1,component);
                cv[component]=source_dependency(src3,(mad && inverted) ? address:0,2,component);
                value[component]=av[component]|bv[component]|(mad ? cv[component]:Dependencies{});
            }
            if(!mad) switch(op) {
            case 0x01:case 0x02:case 0x03:case 0x18: {
                Dependencies dot;
                for(unsigned component=0;component<4;++component) {
                    if(a&(8u>>component)) dot|=av[component];
                    if(b&(8u>>component)) dot|=bv[component];
                }
                value.fill(dot);break;
            }
            case 0x04:value={Dependencies{},av[1]|bv[1],av[2],bv[3]};break;
            case 0x19:value={Dependencies{},av[1]|bv[1],bv[2],av[3]};break;
            case 0x05:case 0x06:case 0x0e:case 0x0f:value.fill(av[0]);break;
            case 0x07:
                value={av[0],av[1],Dependencies{},av[3]};
                path.written_dependencies[33][0]=path.dependencies[33][0]=av[0];path.written_dependencies[33][1]=path.dependencies[33][1]=av[3];
                path.written[33]|=3;break;
            case 0x0b:case 0x13:value=av;break;
            case 0x12:
                for(unsigned component=0;component<2;++component) if(desc&(8u>>component)) {
                    path.written[32]|=1u<<component;path.written_dependencies[32][component]=path.dependencies[32][component]=av[component];
                }
                continue;
            case 0x2e:case 0x2f:
                path.written[33]|=3;
                for(unsigned component=0;component<2;++component)
                    path.written_dependencies[33][component]=path.dependencies[33][component]=((word>>(component ? 21:24))&7u)>=6 ?
                        Dependencies{}:av[component]|bv[component];
                continue;
            default:break;
            }
            const auto dest=(word>>(mad ? 24:21))&31u;
            path.written[dest]|=mask;
            for(unsigned component=0;component<4;++component) if(mask&(8u>>component)) {
                if(dest>=16) {
                    const unsigned lane=(dest-16)*4+component,initial_lane=dest*4+component;
                    ++path.temporary_writes[lane];
                    const auto selected=[&](unsigned source) {
                        return (desc>>((source==0 ? 5:14)+6-component*2))&3u;
                    };
                    const bool self_a=src1==dest && selected(0)==component;
                    const bool self_b=src2==dest && selected(1)==component;
                    // Replay direct scalar operations without reassociation. A reset may be
                    // any operation whose value is independent of every private recurrence.
                    // Intermediate values consumed by an update must be independent too.
                    if(!mad && op==0x13 && self_a) {
                    } else if(!mad && (op==0x00 || op==0x08) && self_a!=self_b) {
                        recurrence_operands[lane]|=self_a ? bv[component]:av[component];
                    } else {
                        recurrence_operands[lane]|=value[component];
                        if(value[component].test(initial_lane)) unsupported_recurrence.set(lane);
                    }
                }
                path.written_dependencies[dest][component]=path.dependencies[dest][component]=value[component];
            }
        }
    }
    if(!ended) return VertexExecution::Ordered;
    Dependencies changing;
    for(unsigned reg=0;reg<34;++reg) for(unsigned component=0;component<4;++component)
        if(may_write[reg]&(reg<32 ? 8u>>component:1u<<component)) changing.set(reg*4+component);
    // A pass may speculate on arithmetic state, never on a branch or error path that
    // depends on changing inherited registers. Conditional writes on input/uniform
    // predicates are safe because the GPU prefix preserves unwritten components.
    if((control_reads&changing).any()) return VertexExecution::Ordered;
    if(native_reset_capture) {
        const auto& capture=*native_reset_capture;
        const auto independent=[&] {
            if(capture.attribute_count>7 || capture.flip_viewport>1 || capture.enable_clip>1 ||
               capture.fix_quaternions || capture.honor_winding) return false;
            const auto attributes=std::min(capture.attribute_count,unsigned(std::popcount(program.output_mask)));
            unsigned attribute=0;
            for(unsigned reg=0;reg<16;++reg) if(program.output_mask&(1u<<reg)) {
                if(attribute==attributes) break;
                for(unsigned c=0;c<4;++c) {
                    const auto semantic=(capture.semantics[attribute]>>(8*c))&31u;
                    if(semantic>=24 && semantic!=31) return false;
                    if(semantic>=24 || semantic==17 || semantic==21) continue;
                    if(!(must_write[reg]&(8u>>c)) || (final_dependencies[reg*4+c]&changing).any())
                        return false;
                }
                ++attribute;
            }
            return true;
        }();
        // Both verdicts consume these exact complete traversal facts. Capture
        // success does not imply an independent full raw/state execution class.
        if(capture_independent) {
            *capture_independent=independent;
            // A successful native selection never needs the full raw proof.
            // Preserve its old classification cost and raw-cache occupancy.
            if(independent && skip_independent_raw) return VertexExecution::Independent;
        }
        else return independent ? VertexExecution::Independent:VertexExecution::Ordered;
    }
    Dependencies recurrences;
    for(unsigned lane=64;lane<128;++lane) {
        const auto dependencies=final_dependencies[lane]&changing;
        if(!changing.test(lane) || !dependencies.test(lane) || dependencies.count()!=1 ||
            unsupported_recurrence.test(lane-64)) continue;
        bool isolated=true;
        for(unsigned other=0;other<136;++other)
            if(other!=lane && changing.test(other) && final_dependencies[other].test(lane)) isolated=false;
        if(isolated) recurrences.set(lane);
    }
    // All replay operands, including assignments after a reset, must be independent
    // of ALL replayed lanes. Otherwise retain the general ordered native GPU machine.
    for(unsigned lane=64;lane<128;++lane) if(recurrences.test(lane) &&
        (recurrence_operands[lane-64]&recurrences).any()) return VertexExecution::Ordered;
    unsigned capacity=0;
    for(unsigned lane=64;lane<128;++lane) if(recurrences.test(lane)) capacity+=maximum_writes[lane-64];
    if(capacity>64) return VertexExecution::Ordered;
    std::array<unsigned,136> level{},visiting{};
    const auto depth=[&](auto&& self,unsigned lane)->unsigned {
        if(!changing.test(lane) || recurrences.test(lane)) return 0;
        if(visiting[lane]==1) return 137; // A retained-state recurrence needs ordered execution.
        if(visiting[lane]==2) return level[lane];
        visiting[lane]=1;unsigned result=1;
        const auto dependencies=final_dependencies[lane]&changing;
        for(unsigned source=0;source<136;++source) if(dependencies.test(source)) {
            const auto parent=self(self,source);
            if(parent>=137) return 137;
            result=std::max(result,parent+1);
        }
        visiting[lane]=2;return level[lane]=result;
    };
    unsigned passes=1;
    for(unsigned lane=0;lane<136;++lane) {
        const auto required=depth(depth,lane);
        if(required>=137) return VertexExecution::Ordered;
        passes=std::max(passes,required);
    }
    if(dependency_passes) *dependency_passes=passes;
    // Successful complete paths provide conservative per-component write facts.
    // Keep these per control-uniform proof, not per compiled program/pipeline.
    for(unsigned lane=0;lane<133;++lane) {
        const unsigned reg=lane<128 ? lane/4:lane<131 ? 32:33;
        const unsigned bit=lane<128 ? 8u>>(lane%4):1u<<(lane<131 ? lane-128:lane-131);
        if(always_written && (must_write[reg]&bit)) (*always_written)[lane/32]|=1u<<(lane%32);
        if(possibly_written && (may_write[reg]&bit)) (*possibly_written)[lane/32]|=1u<<(lane%32);
    }
    if(recurrences.any()) {
        if(recurrence_mask) for(unsigned lane=0;lane<64;++lane) if(recurrences.test(64+lane))
            (*recurrence_mask)[lane/32]|=1u<<(lane%32);
        if(recurrence_capacity) *recurrence_capacity=capacity;
        return VertexExecution::PrivateRecurrences;
    }
    if(passes>1) return VertexExecution::DependencyPasses;
    return must_write==may_write ? VertexExecution::Independent:VertexExecution::MergeState;
}
}
ShaderMachineRuntime::VertexExecution ShaderMachineRuntime::AnalyzeVertices(const ShaderMachineProgram& program,
    const ShaderMachineUniforms& uniforms,std::uint32_t* unproven_instruction,std::uint32_t* dependency_passes,
    std::array<std::uint32_t,2>* recurrence_mask,std::uint32_t* recurrence_capacity,
    std::array<std::uint32_t,5>* always_written,std::array<std::uint32_t,5>* possibly_written,
    const GeometryEmissionConfig* native_reset_capture) {
    return AnalyzeVerticesImpl(program,uniforms,unproven_instruction,dependency_passes,recurrence_mask,
        recurrence_capacity,always_written,possibly_written,native_reset_capture,nullptr);
}
namespace {
bool FitsStructuredVertexSource(const ShaderMachineProgram& program,
                                const std::bitset<4096>& reachable) {
    // Calls are inlined into native GPU control flow. Bound the expanded source,
    // not just the number of distinct PCs, so repeated callees cannot grow it
    // exponentially. The separate structural proof validates every boundary.
    unsigned instructions{};
    const auto block=[&](auto&& self,unsigned begin,unsigned end,unsigned depth)->bool {
        if(depth>128 || begin>end || end>4096) return false;
        for(unsigned pc=begin;pc<end;) {
            if(!reachable.test(pc)) break;
            if(++instructions>1024) return false;
            const auto word=program.code[pc],op=word>>26;
            const unsigned next=pc+1,target=(word>>10)&4095u,length=word&255u;
            if(op==0x22) return true;
            if(op>=0x24 && op<=0x26) {
                if(length && !self(self,target,target+length,depth+1)) return false;
                pc=next;
            } else if(op==0x27 || op==0x28) {
                if(!self(self,next,target,depth+1) ||
                   !self(self,target,target+length,depth+1)) return false;
                pc=target+length;
            } else if(op==0x29) {
                if(!self(self,next,target+1,depth+1)) return false;
                pc=target+1;
            } else pc=next;
        }
        return true;
    };
    return block(block,program.entry,4096,0);
}
std::string SpecializedVertexSource(const ShaderMachineProgram& program,
                                    std::string_view kernel,bool track_writes,
                                    const std::array<std::uint32_t,2>& compiled_recurrences,
                                    bool reset_vertex,bool small_serial=false,bool defer_final_state=false,
                                    bool bounded_execution=false,bool bounded_ordered_execution=false) {
    // Specialize the existing native machine rather than maintaining another arithmetic
    // implementation. Instruction words, descriptors and register indices become compiler
    // constants; ring stacks, instruction bounds and retained state keep the same code.
    const std::string source=ShaderMachineSource;
    const auto execute=source.find("template<bool TrackWrites=false,bool TrackRecurrences=false>");
    const auto instruction=source.find("        const uint word=code[pc],op=word>>26;",execute);
    const auto tail=source.find("    if (!ended && !error) error=6;",instruction);
    const auto function_end=source.find("\nkernel void pica_shader_machine",tail);
    const auto batch=source.find("struct VertexBatchParams",function_end);
    const auto batch_end=source.find("\nuint batch_vertex",batch);
    if(execute==std::string::npos || instruction==std::string::npos || tail==std::string::npos ||
        function_end==std::string::npos || batch==std::string::npos || batch_end==std::string::npos)
        throw std::logic_error("Native shader specialization source boundary is missing");
    auto prefix=source.substr(execute,instruction-execute);
    prefix.erase(0,prefix.find('\n')+1);
    const std::string default_argument="thread uint* event_count=nullptr) {";
    const auto argument=prefix.find(default_argument);
    if(argument==std::string::npos) throw std::logic_error("Native shader specialization signature is missing");
    prefix.replace(argument,default_argument.size(),
        "thread uint* event_count=nullptr) {\n    constexpr bool TrackWrites="+
        std::string(track_writes ? "true;":"false;")+"\n    constexpr bool TrackRecurrences="+
        std::string(kernel=="pica_vertex_machine_recurrence" ? "true;":"false;"));
    // Exclude the outer while's final brace, retaining all instruction semantics verbatim.
    auto body=source.substr(instruction,tail-instruction);
    const auto loop_close=body.rfind("    }");
    if(loop_close==std::string::npos) throw std::logic_error("Native shader specialization loop boundary is missing");
    body.erase(loop_close);
    // The pipeline cache includes this proven mask. Fold event bookkeeping for
    // unrelated temporary components without changing any arithmetic or writes.
    if(kernel=="pica_vertex_machine_recurrence") {
        const std::string mask="recurrence_mask[lane/32]";
        const auto position=body.find(mask);
        if(position==std::string::npos)
            throw std::logic_error("Native recurrence mask expression is missing");
        body.replace(position,mask.size(),"(lane<32 ? "+
            std::to_string(compiled_recurrences[0])+"u:"+
            std::to_string(compiled_recurrences[1])+"u)");
    }
    // Keep the previous default instruction group so pruning other programs
    // does not change the common switch path chosen by the Metal compiler.
    // Linear programs retain their existing entry..END default selection.
    unsigned first=0,last=program.code.size();
    for(unsigned pc=program.entry;pc<program.code.size();++pc) {
        const auto op=program.code[pc]>>26;
        if(op==0x22) {first=program.entry;last=pc+1;break;}
        const bool scalar_emission=kernel=="pica_shader_machine" && (op==0x2a || op==0x2b);
        if(op==0x20 || ((op>=0x23 && op<=0x2d) && !scalar_emission)) break;
    }
    const auto reachable=ShaderMachineRuntime::ReachableVertexInstructions(program);
    std::map<std::pair<std::uint32_t,std::uint32_t>,std::vector<unsigned>> groups;
    for(unsigned pc=first;pc<last;++pc) {
        const auto word=program.code[pc];
        groups[{word,program.swizzle[word&((word>>26)>=0x30 ? 31u:127u)]}].push_back(pc);
    }
    const auto most_common=std::max_element(groups.begin(),groups.end(),
        [](const auto& a,const auto& b) { return a.second.size()<b.second.size(); });
    // The structural proof covers every input/control-uniform path, including
    // calls below entry. Runtime limits and native instruction bodies stay intact.
    for(auto it=groups.begin();it!=groups.end();) {
        if(it==most_common) {++it;continue;}
        std::erase_if(it->second,[&](unsigned pc) {return !reachable.test(pc);});
        if(it->second.empty()) it=groups.erase(it);
        else ++it;
    }
    // Copy only the statically selected instruction bodies from the interpreter.
    // No arithmetic/control implementation is duplicated or evaluated on the CPU.
    const auto specialize_body=[&](std::uint32_t word,std::uint32_t descriptor_value) {
        auto specialized=body;
        const std::string word_load="const uint word=code[pc],op=word>>26;";
        const auto word_position=specialized.find(word_load);
        if(word_position==std::string::npos) throw std::logic_error("Native opcode source load is missing");
        specialized.replace(word_position,word_load.size(),
            "const uint word="+std::to_string(word)+"u,op=word>>26;");
        const auto op=word>>26;
        const std::string arithmetic="        const bool mad=op>=0x30;";
        const auto arithmetic_position=specialized.find(arithmetic);
        if(arithmetic_position==std::string::npos) throw std::logic_error("Native arithmetic source boundary is missing");
        if(op>=0x20 && op<=0x2d) {
            // Every control opcode exits this step through continue/break before arithmetic.
            specialized.erase(arithmetic_position);
            return specialized;
        }
        const std::string next="        next=jump=pc;\n";
        const auto next_position=specialized.find(next);
        if(next_position==std::string::npos || next_position+next.size()>arithmetic_position)
            throw std::logic_error("Native instruction flow boundary is missing");
        specialized.erase(next_position+next.size(),arithmetic_position-next_position-next.size());
        const std::string descriptor="const uint desc=swizzle[word&(mad ? 31u : 127u)];";
        const auto descriptor_position=specialized.find(descriptor);
        if(descriptor_position==std::string::npos) throw std::logic_error("Native instruction descriptor is missing");
        specialized.replace(descriptor_position,descriptor.size(),
            "const uint desc="+std::to_string(descriptor_value)+"u;");
        const std::string switch_begin="        else switch (op) {\n";
        const std::string switch_end="        }\n        if (error) break;";
        const auto begin=specialized.find(switch_begin),end=specialized.find(switch_end,begin);
        if(begin==std::string::npos || end==std::string::npos)
            throw std::logic_error("Native arithmetic switch boundary is missing");
        if(op>=0x30) {
            specialized.erase(begin,end+std::string_view("        }\n").size()-begin);
            return specialized;
        }
        const auto cases=specialized.substr(begin+switch_begin.size(),end-begin-switch_begin.size());
        const char* digits="0123456789abcdef";
        const std::string wanted=std::string("case 0x")+digits[op>>4]+digits[op&15]+":";
        std::string selected,fallback;
        for(std::size_t pos=0;pos<cases.size();) {
            const auto next_case=cases.find("\n        case ",pos+1);
            const auto next_default=cases.find("\n        default:",pos+1);
            const auto last=std::min(next_case==std::string::npos ? cases.size():next_case+1,
                                     next_default==std::string::npos ? cases.size():next_default+1);
            const auto group=cases.substr(pos,last-pos);
            const auto last_nonspace=group.find_last_not_of(" \t\r\n");
            const auto trimmed=group.substr(0,last_nonspace==std::string::npos ? 0:last_nonspace+1);
            if(!trimmed.ends_with("break;") && !trimmed.ends_with("continue;"))
                throw std::logic_error("Native arithmetic case has unsupported fallthrough");
            const auto header=group.substr(0,group.find('\n'));
            if(header.find(wanted)!=std::string::npos) selected=group;
            if(header.starts_with("        default:")) fallback=group;
            pos=last;
        }
        if(fallback.empty()) throw std::logic_error("Native arithmetic default is missing");
        if(selected.empty()) selected=fallback;
        specialized.replace(begin+switch_begin.size(),cases.size(),selected);
        return specialized;
    };
    // A bounded entry-to-END sequence without control flow has no stack or PC
    // dispatch transitions. Execute those original GPU instruction bodies in
    // order, retaining the guard before every step and the original status tail.
    const bool linear_program=first==program.entry && last>first && last-first<=512 &&
        (program.code[last-1]>>26)==0x22;
    const bool direct_blocks=(kernel=="pica_vertex_machine_batch" ||
        kernel=="pica_vertex_machine_dependency" || kernel=="pica_vertex_machine_recurrence" ||
        kernel=="pica_vertex_machine_carry" || kernel=="pica_vertex_machine_small_dependency" ||
        kernel=="pica_shader_machine" || small_serial) && reachable.count()<=512;
    // The structural proof excludes overlapping stack boundaries and overflow.
    // A bounded CALL/IF/LOOP domain can therefore use native structured GPU
    // control while retaining the exact original instruction bodies. Other
    // programs keep their precise machine/stack dispatch.
    const bool ordered_calls=kernel=="pica_vertex_machine_batch" && !small_serial;
    bool structured=(kernel=="pica_vertex_machine_small_dependency" || small_serial || ordered_calls) &&
        reachable.count()<=512 && ShaderMachineRuntime::HasNativeResetControlFlow(program);
    if(structured) for(unsigned pc=0;pc<program.code.size();++pc) if(reachable.test(pc)) {
        const auto op=program.code[pc]>>26;
        if(op==0x2c || op==0x2d) {structured=false;break;}
    }
    if(structured && !FitsStructuredVertexSource(program,reachable)) structured=false;
    std::string result;
    if(!direct_blocks && !linear_program && !structured) {
        result=source.substr(0,execute)+prefix+"        switch(pc) {\n";
        for(const auto& [key,pcs]:groups) {
            if(key==most_common->first) result+="        default:\n";
            else for(const auto pc:pcs) result+="        case "+std::to_string(pc)+"u:\n";
            auto specialized=body;
            const std::string word_load="const uint word=code[pc],op=word>>26;";
            specialized.replace(specialized.find(word_load),word_load.size(),
                "const uint word="+std::to_string(key.first)+"u,op=word>>26;");
            const std::string descriptor="const uint desc=swizzle[word&(mad ? 31u : 127u)];";
            const auto descriptor_position=specialized.find(descriptor);
            if(descriptor_position==std::string::npos) throw std::logic_error("Native shader specialization descriptor is missing");
            specialized.replace(descriptor_position,descriptor.size(),
                "const uint desc="+std::to_string(key.second)+"u;");
            result+="        {\n"+specialized+"            break;\n        }\n";
        }
        result+="        }\n    }\n"+source.substr(tail,function_end-tail);
    }
    // Execute arithmetic runs directly between every possible stack boundary.
    // An omitted machine_advance has no stack end to process and no pending
    // jump/break. All instruction bodies and per-instruction limits stay intact.
    // Bound generated code growth when many identical PCs share one instruction.
    if(structured) {
        const auto loop=prefix.find("    while (!ended && !error && steps<info.limit) {");
        if(loop==std::string::npos)
            throw std::logic_error("Native structured instruction loop boundary is missing");
        constexpr std::string_view guard="!ended && !error && steps<info.limit";
        const auto block=[&](auto&& self,unsigned begin,unsigned end)->std::string {
            std::string emitted;
            for(unsigned pc=begin;pc<end;) {
                if(!reachable.test(pc)) break;
                const auto word=program.code[pc],op=word>>26;
                const unsigned next=pc+1,target=(word>>10)&4095u,length=word&255u;
                emitted+="    if("+std::string(guard)+") {\n";
                if(op>=0x24 && op<=0x26) {
                    emitted+="        pc="+std::to_string(next)+"u; ++steps;\n";
                    if(op!=0x24) {
                        emitted+="        if(";
                        emitted+=op==0x25 ? "machine_condition("+std::to_string(word)+"u,unit)":
                            "bool(uniforms.b&"+std::to_string(1u<<((word>>22)&15u))+"u)";
                        emitted+=") {\n";
                    }
                    emitted+=self(self,target,target+length);
                    if(op!=0x24) emitted+="        }\n";
                    pc=next;
                } else if(op==0x27 || op==0x28) {
                    emitted+="        pc="+std::to_string(next)+"u; ++steps;\n        if(";
                    emitted+=op==0x28 ? "machine_condition("+std::to_string(word)+"u,unit)":
                        "bool(uniforms.b&"+std::to_string(1u<<((word>>22)&15u))+"u)";
                    emitted+=") {\n"+self(self,next,target)+"        } else {\n"+
                        self(self,target,target+length)+"        }\n";
                    pc=target+length;
                } else if(op==0x29) {
                    const auto id=std::to_string(pc);
                    emitted+="        pc="+std::to_string(next)+"u; ++steps;\n"+
                        "        const uint4 loop_"+id+"=uniforms.i["+std::to_string((word>>22)&3u)+
                        "];\n        unit.address.z=int(loop_"+id+".y&255u);\n"+
                        "        if(TrackWrites) writes[32]|=4;\n"+
                        "        for(uint remaining_"+id+"=(loop_"+id+".x&255u)+1u;remaining_"+
                        id+";--remaining_"+id+") {\n"+self(self,next,target+1)+
                        "            if(!("+std::string(guard)+")) break;\n"+
                        "            unit.address.z+=int(loop_"+id+".z&255u);\n        }\n";
                    pc=target+1;
                } else {
                    emitted+="        pc="+std::to_string(pc)+"u; do {\n"+
                        specialize_body(word,program.swizzle[word&(op>=0x30 ? 31u:127u)])+
                        "        } while(false);\n";
                    ++pc;
                }
                emitted+="    }\n";
                if(op==0x22) break;
            }
            return emitted;
        };
        result=source.substr(0,execute)+prefix.substr(0,loop)+
            block(block,program.entry,4096)+
            "    if(!ended && !error && steps<info.limit) { pc=4096u;error=1; }\n"+
            source.substr(tail,function_end-tail);
    } else if(linear_program) {
        const auto loop=prefix.find("    while (!ended && !error && steps<info.limit) {");
        if(loop==std::string::npos)
            throw std::logic_error("Native linear instruction loop boundary is missing");
        result=source.substr(0,execute)+prefix.substr(0,loop)+"    do {\n";
        for(unsigned pc=first;pc<last;++pc) {
            const auto word=program.code[pc],op=word>>26;
            auto specialized=specialize_body(word,program.swizzle[word&(op>=0x30 ? 31u:127u)]);
            result+="        if(ended || error || steps>=info.limit) break;\n"
                "        { do {\n"+specialized+"        } while(false); }\n";
        }
        result+="    } while(false);\n"+source.substr(tail,function_end-tail);
    } else if(direct_blocks) {
        std::bitset<4097> boundaries;
        for(unsigned boundary_pc=0;boundary_pc<program.code.size();++boundary_pc) {
            if(!reachable.test(boundary_pc)) continue;
            const auto word=program.code[boundary_pc];
            const auto op=word>>26,target=(word>>10)&4095u,length=word&255u;
            if(op>=0x24 && op<=0x26) {
                if(target+length<=4096) boundaries.set(target+length);
            } else if(op==0x27 || op==0x28) boundaries.set(target);
            else if(op==0x29) boundaries.set(target+1);
        }
        result=source.substr(0,execute)+prefix+"        switch(pc) {\n";
        for(unsigned pc=0;pc<program.code.size();++pc) {
            if(!reachable.test(pc)) continue;
            const auto word=program.code[pc],op=word>>26;
            // Keep the exact interpreter step/stack tail, but omit arithmetic
            // cases that this immutable instruction can never execute. In
            // particular queued scalar programs must not clone the entire
            // arithmetic interpreter for every reachable PC.
            auto specialized=specialize_body(word,program.swizzle[word&(op>=0x30 ? 31u:127u)]);
            result+="        case "+std::to_string(pc)+"u: {\n            do {\n"+specialized+
                "            } while(false);\n";
            const bool arithmetic=op!=0x20 && op!=0x22 && !(op>=0x23 && op<=0x2d);
            const bool direct=arithmetic && pc+1<program.code.size() && reachable.test(pc+1) && !boundaries.test(pc+1);
            if(direct) result+="            if(ended || error || steps>=info.limit) break;\n        }\n        [[fallthrough]];\n";
            else result+="            break;\n        }\n";
        }
        // Keep the previous default instruction behavior outside the proven PC set.
        auto specialized=body;
        const std::string word_load="const uint word=code[pc],op=word>>26;";
        specialized.replace(specialized.find(word_load),word_load.size(),
            "const uint word="+std::to_string(most_common->first.first)+"u,op=word>>26;");
        const std::string descriptor="const uint desc=swizzle[word&(mad ? 31u : 127u)];";
        specialized.replace(specialized.find(descriptor),descriptor.size(),
            "const uint desc="+std::to_string(most_common->first.second)+"u;");
        result+="        default: {\n            do {\n"+specialized+
                "            } while(false);\n            break;\n        }\n        }\n    }\n"+
                source.substr(tail,function_end-tail);
    }
    result+=source.substr(kernel=="pica_shader_machine" ? function_end:batch);
    // Emit only the requested batch kernel (and its exact finalizer for independent
    // vertices). Keep shared structs/helpers, but do not instantiate unused kernels.
    // Every arithmetic/control-flow body above is copied from the native interpreter.
    for(std::size_t pos=0;(pos=result.find("kernel void ",pos))!=std::string::npos;) {
        const auto name=pos+12,end_name=result.find('(',name);
        const auto function=result.substr(name,end_name-name);
        const auto open=result.find('{',end_name);
        if(end_name==std::string::npos || open==std::string::npos)
            throw std::logic_error("Native shader kernel boundary is missing");
        std::size_t end=open+1;unsigned braces=1;
        while(end<result.size() && braces) {
            if(result[end]=='{') ++braces;
            else if(result[end]=='}') --braces;
            ++end;
        }
        if(braces) throw std::logic_error("Native shader kernel body is incomplete");
        const bool keep=function==kernel || (kernel=="pica_vertex_machine_parallel" &&
                                             function=="pica_vertex_machine_finalize");
        if(keep) pos=end;
        else result.erase(pos,end-pos);
    }
    for(const std::string call:{"machine_execute<true,true>","machine_execute<true>"})
        for(std::size_t pos=0;(pos=result.find(call,pos))!=std::string::npos;)
            result.replace(pos,call.size(),"machine_execute");
    // These values are immutable members of this cached program. Vertex data,
    // control uniforms and the per-request instruction limit remain GPU inputs.
    const auto replace_constant=[&](std::string_view from,std::string_view to) {
        auto position=result.find(from);
        if(position==std::string::npos) return;
        // Preserve every emitted byte, including replacement order. Repeated
        // in-place shortening moves the entire remaining multi-MiB source at
        // each match; a single append pass copies each original range once.
        std::string replaced;
        replaced.reserve(result.size());
        std::size_t begin=0;
        do {
            replaced.append(result,begin,position-begin);
            replaced.append(to);
            begin=position+from.size();
            position=result.find(from,begin);
        } while(position!=std::string::npos);
        replaced.append(result,begin,result.size()-begin);
        result.swap(replaced);
    };
    // This variant is selected only for validated nonindexed ranges of at most
    // eight vertices. Fold out the ordered kernel's unused index FIFO without
    // changing its complete per-vertex machine state or error commit boundary.
    if(small_serial) replace_constant("info.index_size","0u");
    if(linear_program) {
        const auto entry=std::to_string(program.entry)+"u";
        replace_constant("info.machine.entry",entry);
        replace_constant("info.entry",entry);
        if(!program.allow_emission) replace_constant("info.allow_emission","0u");
    }
    if(reset_vertex) {
        // A reset VS starts with bank zero. VS rejects EMIT/SETEMIT before
        // mutation, so no instruction can switch the bank. Seeded dependency
        // passes retain that invariant. Keep the complete physical machine
        // state and use the ordinary pipeline for all inherited states.
        replace_constant("info.machine.reset","1u");
        replace_constant("info.allow_emission","0u");
        replace_constant("unit.output[unit.flags.z]","unit.output[0]");
    }
    if(defer_final_state) {
        if(!reset_vertex || !small_serial || kernel!="pica_vertex_machine_batch")
            throw std::logic_error("Deferred full state requires a fresh small ordered VS");
        constexpr std::string_view store="    stored=unit;\n";
        const auto position=result.find(store);
        if(position==std::string::npos || result.find(store,position+store.size())!=std::string::npos)
            throw std::logic_error("Native deferred VS final-state store boundary is missing or ambiguous");
        // Keep every invocation and raw output unchanged. A strong owned-input
        // descriptor represents the complete state until its exact GPU replay.
        result.erase(position,store.size());
    }
    if(bounded_execution || bounded_ordered_execution) {
        if(!structured || result.find("if (steps)")!=std::string::npos ||
           (bounded_execution && !defer_final_state) ||
           (bounded_ordered_execution && (!reset_vertex || small_serial ||
                                         kernel!="pica_vertex_machine_batch")))
            throw std::logic_error("Bounded execution requires its proven native structured domain");
        if(result.find("steps<info.limit")==std::string::npos)
            throw std::logic_error("Native bounded instruction guard boundary is missing");
        // Selected only with a separate conservative <=1M termination/cost
        // proof and immutable control snapshot. Keep END/error/PC semantics.
        // This private batch consumes only error/code/PC; diagnostics and every
        // public/raw state replay retain the original complete step accounting.
        replace_constant("steps<info.limit","true");
        // Ordered batches retain their complete state store and exact status
        // accounting. Only the separately proven termination guards disappear.
        if(bounded_execution) {
            constexpr std::string_view status="return uint4(error,count,steps,pc);";
            const auto position=result.find(status);
            if(position==std::string::npos || result.find(status,position+status.size())!=std::string::npos)
                throw std::logic_error("Native private instruction status boundary is missing or ambiguous");
            result.replace(position,status.size(),"return uint4(error,count,0u,pc);");
        }
    }
    return result;
}

std::string OwnedDependencySource(const ShaderMachineProgram& program,
    const std::array<std::uint32_t,2>& mask,
    const std::array<std::uint32_t,5>& always,const std::array<std::uint32_t,5>& possible) {
    auto result=SpecializedVertexSource(program,"pica_vertex_machine_small_dependency",true,mask,true);
    if(result.find("if (steps)")!=std::string::npos || result.find("steps<info.limit")==std::string::npos)
        throw std::logic_error("Owned SIMD prefix requires structured bounded control");
    const auto replace=[&](std::string_view from,std::string_view to) {
        for(std::size_t pos=0;(pos=result.find(from,pos))!=std::string::npos;pos+=to.size())result.replace(pos,from.size(),to);
    };
    replace("steps<info.limit","true");
    const auto once=[&](std::string_view from,std::string_view to) {
        const auto pos=result.find(from);
        if(pos==std::string::npos || result.find(from,pos+from.size())!=std::string::npos)
            throw std::logic_error("Owned SIMD source boundary missing or ambiguous");
        result.replace(pos,from.size(),to);
    };
    once("return uint4(error,count,steps,pc);","return uint4(error,count,0u,pc);");
    // This kernel consumes the local output bank directly. Its inclusive SIMD
    // last-writer prefix is the sole complete raw-output store.
    once("    for (uint reg=0;reg<16;++reg) output[reg]=unit.output[0][reg];","");
    const auto kernel=result.find("kernel void pica_vertex_machine_small_dependency(");
    if(kernel==std::string::npos)throw std::logic_error("Owned SIMD prefix kernel missing");
    const auto open=result.find('{',kernel);auto end=open+1;unsigned depth=1;
    while(end<result.size() && depth){if(result[end]=='{')++depth;else if(result[end]=='}')--depth;++end;}
    if(depth)throw std::logic_error("Owned SIMD prefix kernel incomplete");
    const std::string body=R"SIMD({
    // One SIMD group for six fresh owned vertices and two proven passes.
    // Inactive lanes still participate
    // in every ballot/shuffle. No cross-lane memory or shared state store.
    const bool active=element<info.element_count;
    MachineUnit unit={};
    if(active) {
        const uint vertex_id=info.first_vertex-info.vertex_base+element;
        for(uint reg=0;reg<16;++reg) unit.input[reg]=input[ulong(vertex_id)*16+reg];
    }
    uint writes[34]={};
    for(uint pass=0;pass<info.pad0;++pass) {
        for(uint reg=0;reg<34;++reg) writes[reg]=0;
        if(active && !atomic_load_explicit(status,memory_order_relaxed))
            batch_error(status,machine_execute(code,swizzle,uniforms,unit,
                output+ulong(element)*16,unused_emissions,info.machine,writes));
        if(pass+1<info.pad0) {
            #pragma clang loop unroll(full)
            for(uint component=0;component<72;++component) {
                const bool immutable=component==67 || component>=70;
                const uint retained_lane=component<67 ? component+64:component-68+131;
                const bool possible=!immutable && vertex_may_write(info,retained_lane);
                const bool always=!immutable && bool(info.always_written[retained_lane/32]&(1u<<(retained_lane%32)));
                uint value=0;
                if(possible) {
                    value=component<64 ? as_type<uint>(unit.temporary[component/4][component%4]):
                        component<68 ? as_type<uint>(unit.address[component-64]):unit.flags[component-68];
                    if(always) {
                        const uint previous=simd_shuffle_up(value,1);
                        value=element ? previous:0u;
                    } else {
                        const bool written=component<64 ? bool(writes[16+component/4]&(8u>>(component%4))):
                            component<67 ? bool(writes[32]&(1u<<(component-64))):
                            component>=68 && component<70 ? bool(writes[33]&(1u<<(component-68))):false;
                        const uint candidates=uint(static_cast<simd_vote::vote_t>(simd_ballot(active && written)))&((1u<<element)-1u);
                        const uint writer=candidates ? 31u-clz(candidates):0u;
                        const uint selected=simd_shuffle(value,writer);
                        value=candidates ? selected:0u;
                    }
                }
                if(component<64) unit.temporary[component/4][component%4]=as_type<float>(value);
                else if(component<68) unit.address[component-64]=as_type<int>(value);
                else unit.flags[component-68]=value;
            }
        }
    }
    const bool success=!atomic_load_explicit(status,memory_order_relaxed);
    const uint inclusive=element==31 ? ~0u:(1u<<(element+1))-1u;
    #pragma clang loop unroll(full)
    for(uint component=0;component<64;++component) {
        const uint reg=component/4,lane=component%4;
        const bool possible=vertex_may_write(info,component);
        const bool always=bool(info.always_written[component/32]&(1u<<(component%32)));
        uint value=0;
        if(possible) {
            value=as_type<uint>(unit.output[0][reg][lane]);
            if(!always) {
                const uint candidates=uint(static_cast<simd_vote::vote_t>(simd_ballot(active && success && bool(writes[reg]&(8u>>lane)))))&inclusive;
                const uint writer=candidates ? 31u-clz(candidates):0u;
                const uint selected=simd_shuffle(value,writer);
                value=candidates ? selected:0u;
            }
        }
        if(active && success) reinterpret_cast<device uint*>(output)[ulong(element)*64+component]=value;
    }
})SIMD";
    result.replace(open,end-open,body);
    std::string constants="constant uint compiled_always[5]={",conditional="constant uint compiled_conditional[34]={";
    for(unsigned i=0;i<5;++i)constants+=std::to_string(always[i])+"u,";
    constants+="};\nconstant uint compiled_possible[5]={";
    for(unsigned i=0;i<5;++i)constants+=std::to_string(possible[i])+"u,";
    constants+="};\n";
    const auto may=[&](unsigned lane){return (possible[lane/32]&~always[lane/32])&(1u<<(lane%32));};
    for(unsigned reg=0;reg<34;++reg) {
        unsigned bits=0;
        for(unsigned c=0;c<(reg<32 ? 4u:reg==32 ? 3u:2u);++c) {
            const auto lane=reg<32 ? reg*4+c:reg==32 ? 128+c:131+c;
            if(may(lane))bits|=reg<32 ? 8u>>c:1u<<c;
        }
        conditional+=std::to_string(bits)+"u,";
    }
    conditional+="};\n";
    const std::string marker="using namespace metal;";
    const auto at=result.find(marker);
    if(at==std::string::npos)throw std::logic_error("Missing Metal namespace");
    result.insert(at+marker.size(),"\n"+constants+conditional);
    replace("info.always_written[","compiled_always[");
    replace("info.possibly_written[","compiled_possible[");
    replace("if(TrackWrites) writes[dest]|=8u>>component;","if(TrackWrites && (compiled_conditional[dest]&(8u>>component))) writes[dest]|=8u>>component;");
    replace("if(TrackWrites) writes[32]|=1u<<i;","if(TrackWrites && (compiled_conditional[32]&(1u<<i))) writes[32]|=1u<<i;");
    replace("if(TrackWrites) writes[32]|=4;","if(TrackWrites && (compiled_conditional[32]&4u)) writes[32]|=4;");
    replace("if(TrackWrites) writes[33]|=3;","if(TrackWrites) writes[33]|=compiled_conditional[33];");
    return result;
}

}
bool ShaderMachineRuntime::HasIndependentVertices(const ShaderMachineProgram& program,
    const ShaderMachineUniforms& uniforms,std::uint32_t* unproven_instruction) {
    return AnalyzeVertices(program,uniforms,unproven_instruction)==VertexExecution::Independent;
}
struct ShaderMachineState::DeferredBatch {
    std::shared_ptr<const ShaderMachineProgram> program;
    ShaderMachineUniforms uniforms;
    BufferSlice input,indices;
    std::shared_ptr<const NativeVertexInputSnapshot> references;
    std::uint32_t vertex_count,element_count,vertex_base,first_vertex,index_size,instruction_limit;
};
ShaderMachineRuntime::~ShaderMachineRuntime() = default;
namespace {
struct NativeResetFlowProof {
    const ShaderMachineProgram& program;
    unsigned steps{};
    const char* reason{"structured control flow"};
    // Encoded absolute boundaries retain their kind and active nesting order.
    // A called range must not accidentally process another stack's absolute end.
    std::map<std::vector<unsigned>,bool> memo;
    unsigned recursion{};
    bool Reject(const char* why) {reason=why;return false;}
    bool Block(unsigned begin,unsigned end,std::vector<unsigned> boundaries,
               unsigned calls,unsigned branches,unsigned loops) {
        if(recursion>=128) return Reject("native proof recursion bound");
        struct Depth {unsigned& value;explicit Depth(unsigned& v):value{v} {++value;} ~Depth(){--value;}} depth{recursion};
        if(begin>end || end>4096) return Reject("invalid structured range");
        std::vector<unsigned> key{begin,end,calls,branches,loops};key.insert(key.end(),boundaries.begin(),boundaries.end());
        if(const auto it=memo.find(key);it!=memo.end()) return it->second;
        // The active boundary test rejects recursion; this false entry is also
        // a bounded guard against malformed cyclic subroutine descriptions.
        memo.emplace(key,false);
        for(unsigned pc=begin;pc<end;) {
            if(++steps>65536) return Reject("native structural proof budget");
            const auto word=program.code[pc],op=word>>26;
            const unsigned next=pc+1,target=(word>>10)&4095u,length=word&255u;
            const auto boundary_at=[&](unsigned offset) {
                return std::any_of(boundaries.begin(),boundaries.end(),[&](unsigned b) {return b/4==offset;});
            };
            if(op==0x22) {memo[key]=true;return true;}
            if(op==0x21) {pc=next;continue;}
            if(op==0x2c || op==0x2d) {
                if(target<next || target>=end || boundary_at(next) || boundary_at(target))
                    return Reject("JMP leaves its direct structured range");
                bool direct=false;
                for(unsigned label=begin;label<end;) {
                    if(label==target) {direct=true;break;}
                    const auto w=program.code[label],o=w>>26,t=(w>>10)&4095u,n=w&255u;
                    if(o==0x27 || o==0x28) {
                        if(t<label+1 || t+n>end) break;
                        label=t+n;
                    } else if(o==0x29) {
                        if(t<label+1 || t+1>end) break;
                        label=t+1;
                    } else ++label;
                }
                if(!direct) return Reject("JMP enters a nested structured range");
                if(!Block(next,end,boundaries,calls,branches,loops) ||
                   !Block(target,end,boundaries,calls,branches,loops)) return false;
                memo[key]=true;return true;
            }
            if(op>=0x24 && op<=0x26) {
                if(target+length>4096 || boundary_at(next)) return Reject("call at another stack boundary");
                if(length) {
                    if(calls>=3) return Reject("native call depth bound");
                    for(const auto b:boundaries) if(b/4>target && b/4<=target+length)
                        return Reject("called range crosses a foreign stack boundary");
                    auto child=boundaries;child.push_back((target+length)*4);
                    if(!Block(target,target+length,std::move(child),calls+1,branches,loops)) return false;
                }
                pc=next;continue;
            }
            if(op==0x27 || op==0x28) {
                if(target<next || target+length>end || boundary_at(next) || boundary_at(target) ||
                   boundary_at(target+length))
                    return Reject("IF crosses/collides with an active structured boundary");
                if(branches>=8) return Reject("native IF stack bound");
                auto child=boundaries;child.push_back(target*4+1);
                if(!Block(next,target,std::move(child),calls,branches+1,loops) ||
                   !Block(target,target+length,boundaries,calls,branches,loops)) return false;
                pc=target+length;continue;
            }
            if(op==0x29) {
                if(target<next || target+1>end || boundary_at(target+1))
                    return Reject("LOOP crosses/collides with an active structured boundary");
                if(loops>=4) return Reject("native LOOP stack bound");
                auto child=boundaries;child.push_back((target+1)*4+2);
                if(!Block(next,target+1,std::move(child),calls,branches,loops+1)) return false;
                pc=target+1;continue;
            }
            const bool arithmetic=(op<=0x0f && op!=4 && op!=7) || op==0x12 || op==0x13 ||
                op==0x18 || op==0x1a || op==0x1b || op>=0x30 ||
                ((op==0x2e || op==0x2f) && ((word>>24)&7)<=5 && ((word>>21)&7)<=5);
            if(!arithmetic) return Reject("instruction requires precise machine semantics");
            pc=next;
        }
        memo[key]=true;return true;
    }
    bool Analyze() {
        if(program.allow_emission || program.entry>=4096 || program.output_mask>65535)
            return Reject("invalid native reset VS metadata");
        return Block(program.entry,4096,{},0,0,0);
    }
};
}
bool ShaderMachineRuntime::HasNativeResetControlFlow(const ShaderMachineProgram& program) {
    return NativeResetFlowProof{program}.Analyze();
}
std::optional<std::uint32_t> ShaderMachineRuntime::NativeVertexInstructionBound(
    const ShaderMachineProgram& program,const ShaderMachineUniforms& uniforms) {
    if(!HasNativeResetControlFlow(program) ||
       !FitsStructuredVertexSource(program,ReachableVertexInstructions(program))) return std::nullopt;
    struct Bound {std::uint64_t steps;bool ended;};
    constexpr std::uint64_t limit=1000000;
    unsigned visits=0;
    const auto block=[&](auto&& self,unsigned begin,unsigned end,unsigned depth)->std::optional<Bound> {
        if(depth>128 || begin>end || end>4096) return std::nullopt;
        std::uint64_t steps=0;
        for(unsigned pc=begin;pc<end;) {
            if(++visits>65536) return std::nullopt;
            const auto word=program.code[pc],op=word>>26;
            const unsigned next=pc+1,target=(word>>10)&4095u,length=word&255u;
            if(op==0x22) return steps<limit ? std::optional<Bound>{{steps+1,true}}:std::nullopt;
            if(op>=0x24 && op<=0x26) {
                if(target+length>4096) return std::nullopt;
                const auto body=self(self,target,target+length,depth+1);
                if(!body) return std::nullopt;
                // Count the taken path for both conditional calls. Their
                // condition may depend on GPU flags; it is never evaluated here.
                steps+=1+body->steps;
                if(steps>limit) return std::nullopt;
                if(op==0x24 && body->ended) return Bound{steps,true};
                pc=next;
            } else if(op==0x27 || op==0x28) {
                if(target<next || target+length>end) return std::nullopt;
                const auto yes=self(self,next,target,depth+1);
                const auto no=self(self,target,target+length,depth+1);
                if(!yes || !no) return std::nullopt;
                steps+=1+std::max(yes->steps,no->steps);
                if(steps>limit) return std::nullopt;
                if(yes->ended && no->ended) return Bound{steps,true};
                pc=target+length;
            } else if(op==0x29) {
                if(target<next || target+1>end) return std::nullopt;
                const auto body=self(self,next,target+1,depth+1);
                if(!body) return std::nullopt;
                const auto count=(uniforms.i[(word>>22)&3u][0]&255u)+1u;
                steps+=1+std::uint64_t(count)*body->steps;
                if(steps>limit) return std::nullopt;
                if(body->ended) return Bound{steps,true};
                pc=target+1;
            } else {
                // NativeResetFlowProof validated each arithmetic instruction.
                // Forward/backward JMP and BREAK are deliberately not
                // accepted by this separate cost proof, regardless of dataflow.
                if(op!=0x21 && !((op<=0x0f && op!=4 && op!=7) || op==0x12 || op==0x13 ||
                    op==0x18 || op==0x1a || op==0x1b || op>=0x30 || op==0x2e || op==0x2f))
                    return std::nullopt;
                if(++steps>limit) return std::nullopt;
                pc=next;
            }
        }
        return Bound{steps,false};
    };
    const auto bound=block(block,program.entry,4096,0);
    if(!bound || !bound->ended) return std::nullopt;
    return static_cast<std::uint32_t>(bound->steps);
}
bool ShaderMachineRuntime::CanCaptureNativeReset(const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config) {
    return FindNativeResetProgram(source,uniforms,config)!=nullptr;
}
ShaderMachineRuntime::CachedProgram* ShaderMachineRuntime::FindNativeResetProgram(
    const ShaderMachineProgramView& source,const ShaderMachineUniforms& uniforms,
    const GeometryEmissionConfig& config) {
    (void)scheduler.LastUseTick(nil);
    // Instrumented diagnosis retains its established full raw-output capture path.
    if(capture || source.allow_emission || source.entry>=4096 || source.output_mask>65535 ||
       config.attribute_count>7 || config.flip_viewport>1 || config.enable_clip>1 ||
       config.fix_quaternions || config.honor_winding) return nullptr;
    auto& cached=GetProgram(source);
    if(!cached.native_reset_control_flow)
        cached.native_reset_control_flow=HasNativeResetControlFlow(cached.source);
    if(!*cached.native_reset_control_flow) return nullptr;
    std::array<std::uint32_t,17> control{uniforms.b};
    std::memcpy(control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
    std::array<std::uint32_t,8> semantics{};
    semantics[7]=std::min(config.attribute_count,unsigned(std::popcount(source.output_mask)));
    for(unsigned i=0;i<semantics[7];++i) semantics[i]=config.semantics[i]&0x1f1f1f1fu;
    const auto key=std::make_pair(control,semantics);
    if(const auto found=cached.native_reset_proofs.find(key);found!=cached.native_reset_proofs.end())
        return found->second ? &cached:nullptr;
    if(cached.native_reset_proofs.size()>=128) cached.native_reset_proofs.clear();
    bool independent=false;
    (void)GetVertexProof(cached,uniforms,&config,&independent);
    LOG_DEBUG(Render,"Metal4 native reset selection: {} (entry={}, output_mask={:#x}, attributes={})",
        independent ? "proven native capture":"precise GPU dependency capture",
        source.entry,source.output_mask,semantics[7]);
    cached.native_reset_proofs.emplace(key,independent);return independent ? &cached:nullptr;
}
ShaderMachineState::ShaderMachineState(const Device& device):storage{device.CreateBuffer(1824,false)} {}
ShaderMachineRuntime::CachedProgram& ShaderMachineRuntime::GetProgram(const ShaderMachineProgramView& program) {
    // After repeated use, compare every source field before hashing its 32 KiB
    // again. A source change resumes normal hashing until repetition is observed;
    // alternating programs must not pay an extra full comparison on every draw.
    if(recent_program_repeated)
        if(const auto found=program_cache.find(recent_program_hash);found!=program_cache.end())
            for(auto& candidate:found->second) if(program.Matches(candidate.source)) return candidate;
    const auto hash=Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
        Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission);
    recent_program_repeated=hash==recent_program_hash;
    recent_program_hash=hash;
    if(const auto found=program_cache.find(hash);found!=program_cache.end())
        for(auto& candidate:found->second) if(program.Matches(candidate.source)) return candidate;
    // Scheduler argument bindings retain in-flight uploads even after bounded cache eviction.
    if(cached_program_count==256) { program_cache.clear();cached_program_count=0; }
    auto buffer=device.CreateBuffer(sizeof(program.code)+sizeof(program.swizzle),true);
    std::memcpy(buffer.contents,program.code.data(),sizeof(program.code));
    std::memcpy(static_cast<std::byte*>(buffer.contents)+sizeof(program.code),program.swizzle.data(),sizeof(program.swizzle));
    auto& bucket=program_cache[hash];
    CachedProgram created{};
    created.source={program.code,program.swizzle,program.entry,program.output_mask,program.allow_emission};
    created.instructions=buffer;
    bucket.push_back(std::move(created));++cached_program_count;
    return bucket.back();
}
ShaderMachineRuntime::VertexProof* ShaderMachineRuntime::GetVertexProof(
    CachedProgram& cached,const ShaderMachineUniforms& uniforms,
    const GeometryEmissionConfig* native_reset_capture,bool* capture_independent) {
    std::array<std::uint32_t,17> control{uniforms.b};
    std::memcpy(control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
    if(const auto found=cached.control_proofs.find(control);found!=cached.control_proofs.end()) {
        // A different capture semantic can query an already cached raw proof.
        // Without retaining the large graph, keep its exact original query.
        if(capture_independent) *capture_independent=AnalyzeVertices(cached.source,uniforms,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,native_reset_capture)==VertexExecution::Independent;
        return &found->second;
    }
    // Completion eligibility and batch execution need the same complete proof
    // under the same immutable program and all bool/integer control uniforms.
    // Retain every write/recurrence fact here, including a negative Ordered
    // result, instead of analyzing the same first-use draw twice.
    VertexProof proof{};
    proof.execution=AnalyzeVerticesImpl(cached.source,uniforms,&proof.unproven,&proof.passes,
        &proof.recurrence_mask,&proof.recurrence_capacity,&proof.always_written,&proof.possibly_written,
        native_reset_capture,capture_independent,true);
    if(capture_independent && *capture_independent) return nullptr;
    if(cached.control_proofs.size()>=128) cached.control_proofs.clear();
    // Only the existing <=512-PC native structured generator can drop step
    // bookkeeping. The generic machine uses its first-step count for advance.
    if((proof.execution==VertexExecution::DependencyPasses || proof.execution==VertexExecution::Ordered) &&
       ReachableVertexInstructions(cached.source).count()<=512)
        proof.instruction_bound=NativeVertexInstructionBound(cached.source,uniforms).value_or(0);
    auto& result=cached.control_proofs.emplace(control,std::move(proof)).first->second;
    LOG_DEBUG(Render,"Metal4 vertex execution: {} (passes={}, entry={}, instruction={}, opcode={:#x})",
        result.execution==VertexExecution::Independent ? "parallel":
        result.execution==VertexExecution::MergeState ? "parallel + state merge":
        result.execution==VertexExecution::DependencyPasses ? "parallel dependency passes":
        result.execution==VertexExecution::PrivateRecurrences ? "parallel + exact scalar recurrence":"ordered",
        result.passes,cached.source.entry,result.unproven,cached.source.code[result.unproven]>>26);
    return &result;
}
ShaderMachineRuntime::ShaderMachineRuntime(Device& device_,Scheduler& scheduler_)
    :device{device_},scheduler{scheduler_},parameters{device,scheduler,1<<20},scalar_outputs{device,scheduler} {
    if (const auto* path=std::getenv("AZAHAR_METAL4_PROFILE");path && *path) {
        // Frame cadence diagnostics must not add shader dumps or per-draw
        // program/control hashing to the otherwise unchanged execution path.
        const auto* host_only=std::getenv("AZAHAR_METAL4_PROFILE_HOST_ONLY");
        profile_vertices=!host_only || std::string_view{host_only}!="1";
        profile_vertices_bounded=std::getenv("AZAHAR_METAL4_PROFILE_TRIGGER") ||
            std::getenv("AZAHAR_METAL4_PROFILE_FRAME_LIMIT");
    }
    if (const auto* path=std::getenv("AZAHAR_METAL4_CAPTURE_VERTEX_DRAWS");path && *path)
        capture=std::make_shared<CaptureState>(path);
    library=device.CompileLibrary(ShaderMachineSource,"Azahar Metal 4 PICA shader machine");
    auto* desc=[[MTL4ComputePipelineDescriptor alloc] init];
    constexpr std::array<const char*,6> geometry_names{"pica_vertex_machine_batch","pica_vertex_machine_parallel",
        "pica_vertex_machine_carry","pica_vertex_machine_dependency","pica_vertex_machine_recurrence","pica_vertex_machine_batch"};
    for(unsigned i=0;i<geometry_names.size();++i) {
        desc.computeFunctionDescriptor=Device::Function(library,geometry_names[i]);
        desc.maxTotalThreadsPerThreadgroup=i==0 || i==5 ? 1:64;
        geometry_vertex_pipelines[i]=device.CompileCompute(desc);
    }
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_machine_finalize");
    desc.maxTotalThreadsPerThreadgroup=1;geometry_finalize_pipeline=device.CompileCompute(desc);
    desc.maxTotalThreadsPerThreadgroup=64;geometry_simd_finalize_pipeline=device.CompileCompute(desc);
    if(geometry_simd_finalize_pipeline.threadExecutionWidth<16 || geometry_simd_finalize_pipeline.threadExecutionWidth>64)
        throw std::runtime_error("Native geometry vertex finalization requires a SIMD width between16 and64");
    desc.computeFunctionDescriptor=Device::Function(library,"pica_shader_machine");
    desc.maxTotalThreadsPerThreadgroup=1;
    pipeline=device.CompileCompute(desc);
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_machine_merge");
    desc.maxTotalThreadsPerThreadgroup=256;
    merge_pipeline=device.CompileCompute(desc);
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_machine_plan");
    desc.maxTotalThreadsPerThreadgroup=64;
    plan_pipeline=device.CompileCompute(desc);
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_machine_resolve");
    resolve_pipeline=device.CompileCompute(desc);
    if(resolve_pipeline.threadExecutionWidth<16 || resolve_pipeline.threadExecutionWidth>64)
        throw std::runtime_error("Native final-state resolution requires a SIMD width between 16 and 64");
    if(plan_pipeline.threadExecutionWidth<16 || plan_pipeline.threadExecutionWidth>64)
        throw std::runtime_error("Native vertex FIFO requires a SIMD width between 16 and 64");
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_machine_prefix");
    desc.maxTotalThreadsPerThreadgroup=64;
    prefix_pipeline=device.CompileCompute(desc);
    if(prefix_pipeline.threadExecutionWidth<16 || prefix_pipeline.threadExecutionWidth>64 ||
       !std::has_single_bit(unsigned(prefix_pipeline.threadExecutionWidth)))
        throw std::runtime_error("Native prefix requires power-of-two SIMD width between 16 and 64");
    desc.computeFunctionDescriptor=Device::Function(library,"pica_geometry_emissions");
    desc.maxTotalThreadsPerThreadgroup=64;
    emission_pipeline=device.CompileCompute(desc);
    desc.computeFunctionDescriptor=Device::Function(library,"pica_geometry_count");
    desc.maxTotalThreadsPerThreadgroup=1;emission_count_pipeline=device.CompileCompute(desc);
    desc.computeFunctionDescriptor=Device::Function(library,"pica_vertex_capture");
    desc.maxTotalThreadsPerThreadgroup=64;vertex_pipeline=device.CompileCompute(desc);
    empty_input=device.CreateBuffer(4,true);
    std::memset(empty_input.contents,0,4);
}
void ShaderMachineRuntime::DeferResetVertexBatch(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const BufferSlice& input,std::uint32_t vertex_count,
    std::uint32_t element_count,std::uint32_t vertex_base,std::uint32_t first_vertex,
    const BufferSlice& indices,std::uint32_t index_size,std::uint32_t instruction_limit) {
    (void)scheduler.LastUseTick(nil);
    if(source.allow_emission || source.entry>=4096 || source.output_mask>0xffff ||
       !instruction_limit || instruction_limit>1000000 || !vertex_count || !element_count ||
       state.storage.device!=device.Handle() || std::uint64_t(vertex_base)+vertex_count>std::uint64_t(UINT32_MAX)+1 ||
       (index_size!=0 && index_size!=1 && index_size!=2 && index_size!=4) ||
       (!index_size && (first_vertex<vertex_base ||
            std::uint64_t(first_vertex)+element_count>std::uint64_t(vertex_base)+vertex_count)) ||
       !input.buffer || input.buffer.device!=device.Handle() || input.buffer==state.storage || input.offset%16 ||
       input.offset>input.buffer.length || input.size>input.buffer.length-input.offset ||
       std::uint64_t(vertex_count)*sizeof(VertexRegisters)>input.size)
        throw std::invalid_argument("Invalid deferred reset vertex batch");
    if(index_size && (!indices.buffer || indices.buffer.device!=device.Handle() || indices.buffer==state.storage ||
        indices.offset>indices.buffer.length || indices.size>indices.buffer.length-indices.offset ||
        std::uint64_t(element_count)*index_size>indices.size))
        throw std::invalid_argument("Invalid deferred reset vertex indices");
    const auto op=source.code[source.entry]>>26;
    if(op==0x2a || op==0x2b) throw ShaderMachineError(2,source.entry+1);
    auto& cached=GetProgram(source);
    if(!cached.retained_source) cached.retained_source=std::make_shared<const ShaderMachineProgram>(cached.source);
    const auto input_bytes=NSUInteger(vertex_count)*sizeof(VertexRegisters);
    auto pending=std::make_shared<ShaderMachineState::DeferredBatch>();
    pending->program=cached.retained_source;pending->uniforms=uniforms;
    pending->vertex_count=vertex_count;pending->element_count=element_count;
    pending->vertex_base=vertex_base;pending->first_vertex=first_vertex;
    pending->index_size=index_size;pending->instruction_limit=instruction_limit;
    pending->input=AcquireBatchStorage(input_bytes,input.buffer,index_size ? indices.buffer:nil);
    if(index_size) {
        const auto bytes=NSUInteger(element_count)*index_size;
        pending->indices=AcquireBatchStorage(bytes,input.buffer,indices.buffer);
    }
    // The snapshot destinations are owned, disjoint private allocations.
    // Both source reads can share one encoder and its producer dependency.
    scheduler.UseResource(input.buffer);scheduler.UseResource(pending->input.buffer);
    if(index_size) {scheduler.UseResource(indices.buffer);scheduler.UseResource(pending->indices.buffer);}
    const auto encoder=scheduler.BeginCompute();
    [encoder copyFromBuffer:input.buffer sourceOffset:input.offset toBuffer:pending->input.buffer
        destinationOffset:pending->input.offset size:input_bytes];
    if(index_size) [encoder copyFromBuffer:indices.buffer sourceOffset:indices.offset toBuffer:pending->indices.buffer
        destinationOffset:pending->indices.offset size:pending->indices.size];
    scheduler.EndEncoding();
    // Publish only after complete host validation, ownership and copy encoding.
    // Queue order fixes the snapshot before subsequent GPU writes to callers.
    state.deferred=std::move(pending);state.reset=true;
}
void ShaderMachineRuntime::MaterializeDeferred(const ShaderMachineState& state) {
    if(!state.deferred) return;
    (void)scheduler.LastUseTick(nil);
    auto pending=std::move(state.deferred);state.reset=true;
    try {
        // RunVertexBatch changes only mutable reset bookkeeping and the pointed-to
        // GPU allocation. This preserves SaveState's logically const contract.
        auto input = pending->input;
        if (pending->references) {
            if (!input_reference_runtime) input_reference_runtime = std::make_unique<InputReferenceRuntime>(device, scheduler);
            input = input_reference_runtime->Materialize(pending->references->registers, pending->vertex_count);
        }
        RunVertexBatch(const_cast<ShaderMachineState&>(state),*pending->program,pending->uniforms,
            input,pending->vertex_count,pending->element_count,pending->vertex_base,
            pending->first_vertex,pending->indices,pending->index_size,pending->instruction_limit,false);
    } catch(...) {
        state.reset=true;state.deferred=std::move(pending);throw;
    }
}
ShaderMachineResult ShaderMachineRuntime::Run(ShaderMachineState& state,const ShaderMachineProgramView& program,
    const ShaderMachineUniforms& uniforms,const BufferSlice& input,std::uint32_t input_mask,
    std::uint32_t emission_capacity,std::uint32_t instruction_limit) {
    const auto constants=parameters.Allocate(sizeof(uniforms));
    std::memcpy(constants.data,&uniforms,sizeof(uniforms));
    return RunWithUniforms(state,program,constants,input,input_mask,emission_capacity,instruction_limit);
}
ShaderMachineResult ShaderMachineRuntime::RunWithUniforms(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const BufferSlice& constants,const BufferSlice& input,std::uint32_t input_mask,
    std::uint32_t emission_capacity,std::uint32_t instruction_limit) {
    return RunWithUniformsImpl(state,source,constants,input,input_mask,emission_capacity,instruction_limit,false);
}
BufferSlice ShaderMachineRuntime::RunVertexDeferred(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const BufferSlice& input,std::uint32_t input_mask,
    std::uint32_t instruction_limit) {
    if(source.allow_emission) throw std::invalid_argument("Deferred scalar vertex execution cannot emit geometry");
    const auto constants=parameters.Allocate(sizeof(uniforms));
    std::memcpy(constants.data,&uniforms,sizeof(uniforms));
    // A VS cannot legally emit. Keep the kernel's required nonempty emission
    // binding without reserving 192 unused GS emission records per vertex.
    auto result=RunWithUniformsImpl(state,source,constants,input,input_mask,1,instruction_limit,true).output;
    result.data=nullptr;
    return result;
}
ShaderMachineResult ShaderMachineRuntime::RunWithUniformsImpl(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const BufferSlice& constants,const BufferSlice& input,std::uint32_t input_mask,
    std::uint32_t emission_capacity,std::uint32_t instruction_limit,bool defer_vertex) {
    auto pending=EncodeScalar(state,source,constants,input,input_mask,emission_capacity,instruction_limit,{},nullptr,defer_vertex);
    const auto status=pending.status.buffer;
    const auto status_offset=pending.status.offset;
    const auto validate=[status,status_offset] {
        const auto* flags=reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::byte*>(status.contents)+status_offset);
        if(flags[0]) throw ShaderMachineError(flags[0],flags[3]);
    };
    if(defer_vertex) {
        scheduler.CheckAfterCompletion(validate);
        return pending.result;
    }
    try { scheduler.Finish(Scheduler::WaitReason::Readback);return ResolvePending(pending); }
    catch(...) { state.reset=pending.needs_reset;throw; }
}
BufferSlice ShaderMachineRuntime::BeginGeometryBatch() {
    auto result=scalar_outputs.Allocate(16);
    std::memset(result.data,0,16);
    return result;
}
std::optional<std::uint32_t> ShaderMachineRuntime::LinearGeometryStorageBound(const ShaderMachineProgramView& program) {
    if (!program.allow_emission || program.entry>=program.code.size() || program.output_mask>0xffff)
        return std::nullopt;
    std::uint32_t vertices{};
    for (auto pc=program.entry;pc<program.code.size();++pc) {
        const auto op=program.code[pc]>>26;
        if (op==0x22) return std::max(vertices,3u);
        if (op==0x2a) vertices+=3;
        else if (op==0x21 || op==0x2b || op<0x20 || op>=0x30) continue;
        else return std::nullopt;
    }
    return std::nullopt;
}

std::uint32_t ShaderMachineRuntime::EmissionStorageCapacity(const ShaderMachinePendingResult& pending) const {
    if (!pending.storage_proof) return pending.capacity;
    const auto& proof=*pending.storage_proof;
    const auto same=[](const BufferSlice& a,const BufferSlice& b) {
        return a.buffer==b.buffer && a.offset==b.offset && a.size==b.size;
    };
    if (pending.capacity!=proof.logical_capacity || !same(pending.result.emissions,proof.emissions) ||
        !same(pending.status,proof.status) || !proof.storage_capacity || proof.storage_capacity>pending.capacity)
        throw std::invalid_argument("Native compact GS storage proof does not match its result");
    return proof.storage_capacity;
}
bool ShaderMachineRuntime::ProvesGeometryVertexCompletion(const ShaderMachineProgramView& source,
                                                         const ShaderMachineUniforms& uniforms) {
    if(source.allow_emission || source.entry>=4096 || source.output_mask>0xffff) return false;
    auto& cached=GetProgram(source);
    // A successful existing dataflow proof explored all data-dependent paths,
    // checked every opcode/stack boundary and reached END below4096 steps.
    // Ordered is inconclusive: keep the immediate-draw error publication path.
    return GetVertexProof(cached,uniforms)->execution!=VertexExecution::Ordered;
}
ShaderMachinePendingResult ShaderMachineRuntime::QueueGeometry(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const BufferSlice& constants,const BufferSlice& input,const BufferSlice& abort,std::uint32_t input_mask,
    std::uint32_t emission_capacity) {
    return QueueGeometryImpl(state,source,constants,input,abort,input_mask,emission_capacity,nullptr);
}
ShaderMachinePendingResult ShaderMachineRuntime::QueueGeometryImpl(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const BufferSlice& constants,const BufferSlice& input,const BufferSlice& abort,std::uint32_t input_mask,
    std::uint32_t emission_capacity,const GeometryCapture* capture) {
    if(!source.allow_emission || !abort.buffer || !abort.buffer.contents || abort.buffer.device!=device.Handle() || abort.offset%4 ||
        abort.offset>abort.buffer.length || abort.size>abort.buffer.length-abort.offset || abort.size<4)
        throw std::invalid_argument("Invalid guarded native geometry invocation");
    return EncodeScalar(state,source,constants,input,input_mask,emission_capacity,1000000,abort,capture,true);
}
ShaderMachinePendingResult ShaderMachineRuntime::RetryGeometry(ShaderMachineState& state,
    const ShaderMachinePendingResult& pending,std::uint32_t emission_capacity) {
    (void)PendingStatus(pending);
    if(!pending.program || !pending.program->allow_emission || !pending.abort.buffer ||
        state.storage.device!=device.Handle() || pending.unit!=state.storage) throw std::invalid_argument("Invalid native geometry retry");
    state.reset=pending.needs_reset;
    return QueueGeometry(state,*pending.program,pending.uniforms,pending.input,pending.abort,pending.input_mask,emission_capacity);
}
std::array<std::uint32_t,4> ShaderMachineRuntime::PendingStatus(const ShaderMachinePendingResult& pending) const {
    const auto& status=pending.status;
    if(!status.buffer || status.buffer.device!=device.Handle() || !status.buffer.contents || status.offset%4 ||
        status.offset>status.buffer.length || status.size>status.buffer.length-status.offset || status.size<16 ||
        !scheduler.IsComplete(pending.tick)) throw std::invalid_argument("Native shader result is invalid or not complete");
    std::array<std::uint32_t,4> result{};
    std::memcpy(result.data(),static_cast<const std::byte*>(status.buffer.contents)+status.offset,16);
    return result;
}
ShaderMachineResult ShaderMachineRuntime::ResolvePending(const ShaderMachinePendingResult& pending) const {
    const auto flags=PendingStatus(pending);
    if(flags[0]) throw ShaderMachineError(flags[0],flags[3]);
    if(flags[1]>EmissionStorageCapacity(pending)) throw std::runtime_error("Native geometry result exceeds its output reservation");
    auto result=pending.result;result.emission_count=flags[1];return result;
}
void ShaderMachineRuntime::RestoreFailedGeometryReset(ShaderMachineState& state,const ShaderMachinePendingResult& pending) {
    if(state.storage.device!=device.Handle() || pending.unit!=state.storage || !PendingStatus(pending)[0])
        throw std::invalid_argument("Native geometry reset restoration requires a completed failure");
    // The guarded kernel leaves the failed unit and every later unit untouched.
    state.reset=pending.needs_reset;
}
ShaderMachinePendingResult ShaderMachineRuntime::EncodeScalar(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const BufferSlice& constants,const BufferSlice& input,std::uint32_t input_mask,
    std::uint32_t emission_capacity,std::uint32_t instruction_limit,const BufferSlice& abort,const GeometryCapture* capture,
    bool specialize) {
    if ((capture && !specialize) || source.entry>=4096 || source.output_mask>0xffff || input_mask>0xffff || !instruction_limit || instruction_limit>1000000 ||
        !emission_capacity || emission_capacity>1000000 || state.storage.device!=device.Handle())
        throw std::invalid_argument("Invalid native shader machine program/state/dimensions");
    if (input_mask && (!input.buffer || input.buffer.device!=device.Handle() || input.offset>input.buffer.length ||
        input.size>input.buffer.length-input.offset || input.size<sizeof(VertexRegisters) || input.offset%16))
        throw std::invalid_argument("Native shader machine input is missing or too small");
    if (!constants.buffer || constants.buffer.device!=device.Handle() || constants.offset%16 ||
        constants.offset>constants.buffer.length || constants.size>constants.buffer.length-constants.offset ||
        constants.size<sizeof(ShaderMachineUniforms))
        throw std::invalid_argument("Native shader machine uniforms are missing, misaligned or too small");
    auto storage_capacity=emission_capacity;
    if (capture && capture->compact) {
        const auto bound=LinearGeometryStorageBound(source);
        if (!bound || capture->emissions.buffer)
            throw std::invalid_argument("Compact GS storage requires a complete linear emission proof");
        storage_capacity=std::min(emission_capacity,*bound);
    }
    if (capture) {
        const auto& history=capture->history;
        if(!source.allow_emission || !abort.buffer || !history.buffer || history.buffer.device!=device.Handle() ||
            history.offset%16 || history.offset>history.buffer.length || history.size>history.buffer.length-history.offset ||
            history.size!=2128 || constants.buffer!=history.buffer || constants.offset!=history.offset ||
            constants.size!=sizeof(ShaderMachineUniforms) || input.buffer!=history.buffer ||
            input.offset!=history.offset+sizeof(ShaderMachineUniforms)+sizeof(VertexRegisters) || input.size!=sizeof(VertexRegisters) ||
            !capture->invoked_pipeline || capture->invoked_pipeline.device!=device.Handle() ||
            capture->invoked_binding.buffer!=capture->sources[0] || capture->invoked_binding.offset || capture->invoked_binding.index)
            throw std::invalid_argument("Invalid native GS snapshot/dispatch chain");
        constexpr std::array<NSUInteger,3> sizes{sizeof(ShaderMachineUniforms),sizeof(VertexRegisters),sizeof(VertexRegisters)};
        for(unsigned i=0;i<sizes.size();++i)
            if(!capture->sources[i] || capture->sources[i].device!=device.Handle() || capture->sources[i].length!=sizes[i] ||
                capture->sources[i]==history.buffer)
                throw std::invalid_argument("Invalid native GS snapshot source bank");
        const auto& output=capture->emissions;
        if(output.buffer && (output.buffer.device!=device.Handle() || output.offset%16 ||
            output.offset>output.buffer.length || output.size>output.buffer.length-output.offset ||
            NSUInteger(emission_capacity)*sizeof(ShaderMachineEmission)>output.size || output.buffer==history.buffer ||
            output.buffer==abort.buffer || std::find(capture->sources.begin(),capture->sources.end(),output.buffer)!=capture->sources.end()))
            throw std::invalid_argument("Invalid native streaming GS emission storage");
    }
    const auto upload=[&](const auto& value) {
        auto slice=parameters.Allocate(sizeof(value));
        std::memcpy(slice.data,&value,sizeof(value));
        return slice;
    };
    MaterializeDeferred(state);
    auto& cached=GetProgram(source);
    const auto& program=cached.source;
    auto selected=pipeline;
    std::shared_future<id<MTLComputePipelineState>> deferred;
    if(specialize) {
        if(cached.queued_scalar_pipeline.valid() &&
           cached.queued_scalar_pipeline.wait_for(std::chrono::seconds{0})==std::future_status::ready) {
            try { (void)cached.queued_scalar_pipeline.get(); }
            catch (...) { cached.queued_scalar_pipeline={}; }
        }
        if(!cached.queued_scalar_pipeline.valid()) {
            // Queued VS/GS invocations retain their complete scalar semantics.
            // Only immutable instruction/descriptor words become GPU compiler
            // constants; uniforms, state banks, limits and emission guards stay
            // dynamic. Synchronous Run remains the independent generic GPU oracle.
            auto owned=std::make_shared<const ShaderMachineProgram>(program);
            cached.queued_scalar_pipeline=device.DeferCompilation([native=&device,owned] {
                const auto& program=*owned;
                const auto library=native->CompileLibrary(SpecializedVertexSource(program,
                    "pica_shader_machine",false,{},false),"Azahar Metal4 specialized queued scalar machine");
                auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
                descriptor.computeFunctionDescriptor=Device::Function(library,"pica_shader_machine");
                descriptor.maxTotalThreadsPerThreadgroup=1;
                const auto hash=Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
                    Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission);
                descriptor.label=[NSString stringWithFormat:@"pica_shader_machine specialized id=%llu entry=%u emission=%u",
                    static_cast<unsigned long long>(hash),program.entry,unsigned(program.allow_emission)];
                auto result=native->CompileCompute(descriptor);
                LOG_INFO(Render,"Metal4 specialized queued scalar active: entry={}, mask={:#x}, geometry={}",
                    program.entry,program.output_mask,program.allow_emission);
                return result;
            },sizeof(ShaderMachineProgram));
        }
        deferred=cached.queued_scalar_pipeline;
    }
    const std::array<std::uint32_t,8> config{program.entry,program.output_mask,input_mask,state.reset ? 1U:0U,
        emission_capacity,instruction_limit,program.allow_emission ? 1U:0U,abort.buffer ? 1U:0U};
    const auto params=upload(config);
    const auto emission_bytes=NSUInteger(storage_capacity)*sizeof(ShaderMachineEmission);
    const bool streaming=capture && capture->emissions.buffer;
    const auto storage=scalar_outputs.Allocate(sizeof(VertexRegisters)+(streaming ? 0:emission_bytes)+16);
    // Keep one CPU lease for both returned ranges. A retained result must remain
    // immutable even after later invocations finish and the arena wraps.
    const BufferSlice result{storage.buffer,storage.offset,sizeof(VertexRegisters),storage.data,storage.lease};
    const BufferSlice emissions=streaming ? BufferSlice{capture->emissions.buffer,capture->emissions.offset,emission_bytes,nullptr,{}}:
        BufferSlice{storage.buffer,storage.offset+sizeof(VertexRegisters),emission_bytes,
        static_cast<std::byte*>(storage.data)+sizeof(VertexRegisters),storage.lease};
    const auto status_offset=storage.offset+sizeof(VertexRegisters)+(streaming ? 0:emission_bytes);
    auto* flags=static_cast<std::uint32_t*>(static_cast<void*>(
        static_cast<std::byte*>(storage.data)+sizeof(VertexRegisters)+(streaming ? 0:emission_bytes)));
    std::memset(flags,0,16);
    const std::array<BufferBinding,10> bindings{{
        {cached.instructions,0,0},{cached.instructions,sizeof(program.code),1},{constants.buffer,constants.offset,2},
        {input_mask ? input.buffer:empty_input,input_mask ? input.offset:0,3},{state.storage,0,4},
        {result.buffer,result.offset,5},{emissions.buffer,emissions.offset,6},
        {storage.buffer,status_offset,7},{params.buffer,params.offset,8},
        {abort.buffer ? abort.buffer:empty_input,abort.buffer ? abort.offset:0,9}}};
    if(specialize) {
        std::vector<ComputeDispatch> dispatches;
        ComputeDispatch scalar;
        scalar.pending_pipeline=deferred;
        scalar.buffers.assign(bindings.begin(),bindings.end());
        dispatches.push_back(std::move(scalar));
        std::vector<ComputeBufferCopy> copies;
        if(capture) {
            NSUInteger offset=capture->history.offset;
            for(auto buffer:capture->sources) {
                copies.push_back({buffer,capture->history.buffer,0,offset,buffer.length});
                offset+=buffer.length;
            }
            ComputeDispatch invoked;
            invoked.pipeline=capture->invoked_pipeline;
            invoked.buffers.push_back(capture->invoked_binding);
            dispatches.push_back(std::move(invoked));
        }
        scheduler.QueueCompute(dispatches,copies);
    } else {
        const auto args=scheduler.MakeArguments(bindings);
        scheduler.KeepAlive(selected);
        const auto encoder=scheduler.BeginCompute(selected);
        [encoder setComputePipelineState:selected];[encoder setArgumentTable:args];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        scheduler.EndCompute();
    }
    if(abort.buffer && !cached.retained_source) cached.retained_source=std::make_shared<const ShaderMachineProgram>(program);
    ShaderMachinePendingResult pending{{result,emissions,0},
        {storage.buffer,status_offset,16,flags,storage.lease},constants,input,abort,
        abort.buffer ? cached.retained_source:nullptr,state.storage,input_mask,emission_capacity,instruction_limit,bool(config[3]),scheduler.CurrentTick()};
    if (storage_capacity!=emission_capacity) {
        auto proof=std::shared_ptr<NativeEmissionStorage>{new NativeEmissionStorage};
        proof->emissions=emissions;proof->status=pending.status;
        proof->logical_capacity=emission_capacity;proof->storage_capacity=storage_capacity;
        pending.storage_proof=std::move(proof);
    }
    state.reset=false;
    return pending;
}
BufferSlice ShaderMachineRuntime::AcquireBatchStorage(NSUInteger size,id<MTLBuffer> avoid_a,id<MTLBuffer> avoid_b,
                                                     std::span<const BufferSlice> avoid_inputs,std::span<const BufferBinding> avoid_bindings) {
    constexpr NSUInteger budget=8*1024*1024;
    PrivateBatchBuffer* available=nullptr;
    NSUInteger retained=0;
    for(auto& entry:batch_buffers) {
        retained+=entry.buffer.length;
        if(entry.buffer && entry.buffer!=avoid_a && entry.buffer!=avoid_b &&
           std::none_of(avoid_inputs.begin(),avoid_inputs.end(),[&](const auto& input){return input.buffer==entry.buffer;}) &&
           std::none_of(avoid_bindings.begin(),avoid_bindings.end(),[&](const auto& binding){return binding.buffer==entry.buffer;}) &&
           entry.lease.use_count()==1 && entry.buffer.length>=size &&
           (!available || entry.buffer.length<available->buffer.length)) available=&entry;
    }
    if(available) return {available->buffer,0,size,nullptr,available->lease};
    // Cache retention is bounded. Held outputs and oversized valid allocations
    // receive their own private buffer, with the identical native GPU execution.
    if(size<=budget) for(auto& entry:batch_buffers) {
        if((!entry.buffer || entry.lease.use_count()==1) &&
           retained-entry.buffer.length<=budget-size) {
            entry.buffer=device.CreateBuffer(size,false);
            entry.lease=std::make_shared<std::byte>();
            return {entry.buffer,0,size,nullptr,entry.lease};
        }
    }
    return {device.CreateBuffer(size,false),0,size,nullptr,std::make_shared<std::byte>()};
}
namespace {
void ValidateVertexCaptureConfig(std::uint32_t mask,const GeometryEmissionConfig& config) {
    if(mask>0xffff || config.attribute_count>7 || config.flip_viewport>1 || config.enable_clip>1 ||
       config.fix_quaternions || config.honor_winding)
        throw std::invalid_argument("Invalid native raw VS capture input/configuration");
    for(unsigned i=0;i<config.attribute_count;++i) for(unsigned c=0;c<4;++c) {
        const auto semantic=(config.semantics[i]>>(8*c))&31u;
        if(semantic>=24 && semantic!=31) throw std::invalid_argument("Unknown native raw VS output semantic");
    }
}
}
std::optional<ShaderMachineRuntime::NativeResetSelection> ShaderMachineRuntime::SelectNativeReset(
    const ShaderMachineProgramView& source,const ShaderMachineUniforms& uniforms,
    const GeometryEmissionConfig& config) {
    (void)scheduler.LastUseTick(nil);
    ValidateVertexCaptureConfig(source.output_mask,config);
    auto* cached=FindNativeResetProgram(source,uniforms,config);
    if(!cached) return std::nullopt;
    if(!cached->retained_source) cached->retained_source=std::make_shared<const ShaderMachineProgram>(cached->source);
    auto pending=std::make_shared<ShaderMachineState::DeferredBatch>();
    pending->program=cached->retained_source;pending->uniforms=uniforms;
    return NativeResetSelection{this,std::move(pending)};
}
NativeProgramSnapshot ShaderMachineRuntime::NativeProgram(const NativeResetSelection& selection) const {
    (void)scheduler.LastUseTick(nil);
    if(selection.owner!=this || !selection.pending || !selection.pending->program)
        throw std::invalid_argument("Missing, foreign or consumed native program selection");
    return NativeProgramSnapshot{selection.pending->program};
}
void ShaderMachineRuntime::DeferNativeReferenceBatch(ShaderMachineState& state,
    NativeResetSelection&& selection,std::shared_ptr<const NativeVertexInputSnapshot> snapshot,
    std::uint32_t vertex_count, std::uint32_t element_count, std::uint32_t vertex_base,
    std::uint32_t first_vertex, std::uint32_t index_size, std::uint32_t instruction_limit) {
    (void)scheduler.LastUseTick(nil);
    if(selection.owner!=this || !selection.pending || !selection.pending->program)
        throw std::invalid_argument("Missing, foreign or consumed native vertex selection");
    if (!snapshot ||
        !instruction_limit || instruction_limit > 1000000 || !vertex_count || !element_count ||
        vertex_count > UINT32_MAX / 16 || state.storage.device != device.Handle() ||
        std::uint64_t(vertex_base) + vertex_count > std::uint64_t(UINT32_MAX) + 1 ||
        (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) ||
        (!index_size && (first_vertex < vertex_base ||
            std::uint64_t(first_vertex) + element_count > std::uint64_t(vertex_base) + vertex_count)))
        throw std::invalid_argument("Invalid immutable native vertex reference batch");
    const auto& indices = snapshot->indices;
    if (index_size && (!indices.buffer || indices.buffer.device != device.Handle() ||
        indices.offset > indices.buffer.length || indices.size > indices.buffer.length - indices.offset ||
        std::uint64_t(element_count) * index_size > indices.size))
        throw std::invalid_argument("Invalid immutable native vertex reference indices");
    if (snapshot->validated_device != device.Handle() || snapshot->validated_count != vertex_count) {
        // Borrowed/test snapshots retain the full check. The rasterizer's
        // immutable records already carry this exact device/extent validation.
        if (!input_reference_runtime) input_reference_runtime = std::make_unique<InputReferenceRuntime>(device, scheduler);
        input_reference_runtime->Validate(snapshot->registers, vertex_count);
    }
    auto pending=std::move(selection.pending);selection.owner=nullptr;
    pending->references = std::move(snapshot);
    pending->indices = pending->references->indices;
    pending->vertex_count = vertex_count;
    pending->element_count = element_count;
    pending->vertex_base = vertex_base;
    pending->first_vertex = first_vertex;
    pending->index_size = index_size;
    pending->instruction_limit = instruction_limit;
    state.deferred = std::move(pending);
    state.reset = true;
}
BufferSlice ShaderMachineRuntime::RunVertexBatch(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const BufferSlice& input,std::uint32_t vertex_count,
    std::uint32_t element_count,std::uint32_t vertex_base,std::uint32_t first_vertex,
    const BufferSlice& indices,std::uint32_t index_size,std::uint32_t instruction_limit,bool defer_validation) {
    return RunVertexBatchImpl(state,source,uniforms,input,vertex_count,element_count,vertex_base,first_vertex,
        indices,index_size,instruction_limit,defer_validation,nullptr,nullptr);
}
ShaderMachineVertexBatchResult ShaderMachineRuntime::RunVertexBatchAndCapture(ShaderMachineState& state,
    const ShaderMachineProgramView& source,const ShaderMachineUniforms& uniforms,const BufferSlice& input,
    const GeometryEmissionConfig& config,std::uint32_t vertex_count,std::uint32_t element_count,
    std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size,
    std::uint32_t instruction_limit,bool defer_validation) {
    // Invalid conversion metadata never queues VS work or changes retained state.
    ValidateVertexCaptureConfig(source.output_mask,config);
    ShaderMachineVertexBatchResult result{};
    result.output=RunVertexBatchImpl(state,source,uniforms,input,vertex_count,element_count,vertex_base,first_vertex,
        indices,index_size,instruction_limit,defer_validation,&config,&result.captured);
    return result;
}
ShaderMachineFetchedVertexBatchResult ShaderMachineRuntime::RunFetchedVertexBatchAndCapture(
    VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config,
    std::uint32_t vertex_count,std::uint32_t element_count,std::uint32_t vertex_base,
    std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size,
    std::uint32_t instruction_limit,bool defer_validation) {
    return RunFetchedVertexBatchAndCaptureImpl(fetch,input_state,inputs,layout,state,source,uniforms,
        config,vertex_count,element_count,vertex_base,first_vertex,indices,index_size,instruction_limit,
        defer_validation,false);
}
ShaderMachineVertexBatchResult ShaderMachineRuntime::RunOwnedFetchedVertexBatchAndCapture(
    VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config,
    std::uint32_t vertex_count,std::uint32_t element_count,std::uint32_t vertex_base,
    std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size,
    std::uint32_t instruction_limit,bool defer_validation) {
    auto result=RunFetchedVertexBatchAndCaptureImpl(fetch,input_state,inputs,layout,state,source,uniforms,
        config,vertex_count,element_count,vertex_base,first_vertex,indices,index_size,instruction_limit,
        defer_validation,true);
    return {std::move(result.output),std::move(result.captured)};
}
ShaderMachineFetchedVertexBatchResult ShaderMachineRuntime::RunFetchedVertexBatchAndCaptureImpl(
    VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config,
    std::uint32_t vertex_count,std::uint32_t element_count,std::uint32_t vertex_base,
    std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size,
    std::uint32_t instruction_limit,bool defer_validation,bool owned_input) {
    ValidateVertexCaptureConfig(source.output_mask,config);
    if(&fetch.device!=&device || &fetch.scheduler!=&scheduler)
        throw std::invalid_argument("Fetch and shader runtimes use different device/queue contexts");
    // Reject invalid VS metadata before fetch allocations can recycle upload pages.
    if(source.allow_emission || source.entry>=4096 || !instruction_limit || instruction_limit>1000000 ||
       !vertex_count || !element_count || state.storage.device!=device.Handle())
        throw std::invalid_argument("Invalid fetched vertex machine metadata");
    const auto entry_opcode=source.code[source.entry]>>26;
    if(entry_opcode==0x2a || entry_opcode==0x2b) throw ShaderMachineError(2,source.entry+1);
    auto prepared=fetch.PreparePica(input_state,inputs,layout,vertex_count,element_count,
                                  vertex_base,first_vertex,indices,index_size);
    ShaderMachineFetchedVertexBatchResult result{};
    result.input=std::move(prepared.output);
    result.output=RunVertexBatchImpl(state,source,uniforms,result.input,vertex_count,element_count,
        vertex_base,first_vertex,indices,index_size,instruction_limit,defer_validation,&config,
        &result.captured,&fetch,&input_state,&prepared,false,owned_input);
    return result;
}
BufferSlice ShaderMachineRuntime::RunFetchedGeometryVertices(
    VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,std::uint32_t vertex_count,std::uint32_t element_count,
    std::uint32_t vertex_base,std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size) {
    if(&fetch.device!=&device || &fetch.scheduler!=&scheduler ||
       state.storage.device!=device.Handle() || !vertex_count || !element_count ||
       !ProvesGeometryVertexCompletion(source,uniforms))
        throw std::invalid_argument("Unproved fetched geometry vertex batch");
    auto prepared=fetch.PreparePica(input_state,inputs,layout,vertex_count,element_count,
                                  vertex_base,first_vertex,indices,index_size);
    // No semantic capture: the GS input assembler consumes the original raw
    // registers at each element offset. Fetch and VS share their indexed plan;
    // a FIFO hit must advance neither retained input nor the VS unit.
    const auto input=prepared.output;
    return RunVertexBatchImpl(state,source,uniforms,input,vertex_count,element_count,
        vertex_base,first_vertex,indices,index_size,1000000,true,nullptr,nullptr,
        &fetch,&input_state,&prepared,true);
}
BufferSlice ShaderMachineRuntime::RunFetchedNativeResetAndCapture(
    VertexFetchRuntime& fetch,VertexInputState& input_state,std::span<const BufferSlice> inputs,
    const PicaVertexLayout& layout,ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const GeometryEmissionConfig& config,
    GeometryRuntime& geometry,const VertexCaptureProgram& native,std::span<const BufferBinding> bindings,
    std::uint32_t vertex_count,std::uint32_t element_count,std::uint32_t vertex_base,
    std::uint32_t first_vertex,const BufferSlice& indices,std::uint32_t index_size,std::uint32_t instruction_limit) {
    (void)scheduler.LastUseTick(nil);
    ValidateVertexCaptureConfig(source.output_mask,config);
    if(&fetch.device!=&device || &fetch.scheduler!=&scheduler ||
       &geometry.device!=&device || &geometry.scheduler!=&scheduler ||
       source.allow_emission || source.entry>=4096 || source.output_mask>0xffff ||
       !instruction_limit || instruction_limit>1000000 || !vertex_count || !element_count ||
       vertex_count>UINT32_MAX/16 || state.storage.device!=device.Handle() ||
       std::uint64_t(vertex_base)+vertex_count>std::uint64_t(UINT32_MAX)+1 ||
       (index_size!=0 && index_size!=1 && index_size!=2 && index_size!=4) ||
       (!index_size && (first_vertex<vertex_base ||
           std::uint64_t(first_vertex)+element_count>std::uint64_t(vertex_base)+vertex_count)))
        throw std::invalid_argument("Invalid owned native reset vertex batch");
    if(index_size && (!indices.buffer || indices.buffer.device!=device.Handle() || indices.buffer==state.storage ||
       indices.offset>indices.buffer.length || indices.size>indices.buffer.length-indices.offset ||
       std::uint64_t(element_count)*index_size>indices.size))
        throw std::invalid_argument("Invalid owned native reset vertex indices");
    auto* selected=FindNativeResetProgram(source,uniforms,config);
    if(!selected)
        throw std::invalid_argument("Owned native reset capture requires proven independent outputs");
    auto& cached=*selected;
    if(!cached.retained_source) cached.retained_source=std::make_shared<const ShaderMachineProgram>(cached.source);
    auto pending=std::make_shared<ShaderMachineState::DeferredBatch>();
    pending->program=cached.retained_source;pending->uniforms=uniforms;
    pending->vertex_count=vertex_count;pending->element_count=element_count;
    pending->vertex_base=vertex_base;pending->first_vertex=first_vertex;
    pending->index_size=index_size;pending->instruction_limit=instruction_limit;
    pending->input=AcquireBatchStorage(NSUInteger(vertex_count)*sizeof(VertexRegisters),indices.buffer,state.storage,inputs,bindings);
    if(index_size) pending->indices=AcquireBatchStorage(NSUInteger(element_count)*index_size,indices.buffer,state.storage,inputs,bindings);
    auto prepared=fetch.PreparePica(input_state,inputs,layout,vertex_count,element_count,
                                  vertex_base,first_vertex,index_size ? pending->indices:BufferSlice{},index_size,pending->input);
    auto capture=geometry.PrepareCapture(native,pending->input,vertex_count,first_vertex,bindings);
    // All host validation and all potentially submitting allocations precede
    // encoding or publication of either state. The canonical destination is
    // privately owned and can never be changed by a later caller upload.
    const auto fetch_commands=fetch.PreparePicaDispatches(input_state,prepared);
    const std::array dispatches{fetch_commands[0],fetch_commands[1],GeometryRuntime::CaptureDispatch(capture)};
    std::vector<ComputeBufferCopy> copies;
    if(index_size) copies.push_back({indices.buffer,pending->indices.buffer,
        indices.offset,pending->indices.offset,pending->indices.size});
    scheduler.QueueCompute(dispatches,copies);
    fetch.CommitPica(input_state);
    state.deferred=std::move(pending);state.reset=true;
    return std::move(capture.output);
}
BufferSlice ShaderMachineRuntime::RunVertexBatchImpl(ShaderMachineState& state,const ShaderMachineProgramView& source,
    const ShaderMachineUniforms& uniforms,const BufferSlice& input,std::uint32_t vertex_count,
    std::uint32_t element_count,std::uint32_t vertex_base,std::uint32_t first_vertex,
    const BufferSlice& indices,std::uint32_t index_size,std::uint32_t instruction_limit,bool defer_validation,
    const GeometryEmissionConfig* capture_config,BufferSlice* captured_output,
    VertexFetchRuntime* fetch_runtime,VertexInputState* fetch_state,
    VertexFetchRuntime::PreparedPicaFetch* prepared_fetch,bool generic_geometry,bool owned_input) {
    if (source.allow_emission || source.entry>=4096 || source.output_mask>0xffff ||
        !instruction_limit || instruction_limit>1000000 || !vertex_count || !element_count ||
        state.storage.device!=device.Handle() || std::uint64_t(vertex_base)+vertex_count>std::uint64_t(UINT32_MAX)+1 ||
        (index_size!=0 && index_size!=1 && index_size!=2 && index_size!=4) ||
        (!index_size && (first_vertex<vertex_base ||
            std::uint64_t(first_vertex)+element_count>std::uint64_t(vertex_base)+vertex_count)) ||
        !input.buffer || input.buffer.device!=device.Handle() || input.offset%16 ||
        input.offset>input.buffer.length || input.size>input.buffer.length-input.offset ||
        std::uint64_t(vertex_count)*sizeof(VertexRegisters)>input.size)
        throw std::invalid_argument("Invalid native vertex machine batch");
    if (index_size && (!indices.buffer || indices.buffer.device!=device.Handle() ||
        indices.offset>indices.buffer.length || indices.size>indices.buffer.length-indices.offset ||
        std::uint64_t(element_count)*index_size>indices.size))
        throw std::invalid_argument("Invalid native vertex machine batch indices");
    // An emission at the entry point is unconditionally illegal in VS mode. Reject it
    // before queuing work; errors reached through control flow remain GPU-validated.
    const auto entry_opcode=source.code[source.entry]>>26;
    if (entry_opcode==0x2a || entry_opcode==0x2b) throw ShaderMachineError(2,source.entry+1);
    MaterializeDeferred(state);
    auto& cached=GetProgram(source);
    const auto& program=cached.source;
    const bool profile_draw=profile_vertices && (!profile_vertices_bounded || scheduler.ProfilingEnabled());
    std::vector<std::byte> diagnostic;
    VertexDiagnostic::Header diagnostic_header{};
    const auto append=[&](const void* bytes,std::size_t size) {
        const auto* begin=static_cast<const std::byte*>(bytes);
        if (size) diagnostic.insert(diagnostic.end(),begin,begin+size);
    };
    // State aliases need a coupled storage replay and are intentionally excluded
    // from this version. Oversized records are excluded before any GPU readback.
    const std::uint64_t input_bytes=std::uint64_t(vertex_count)*sizeof(VertexRegisters);
    const std::uint64_t index_bytes=std::uint64_t(element_count)*index_size;
    const std::uint64_t output_bytes=std::uint64_t(element_count)*sizeof(VertexRegisters);
    const bool capture_draw=capture && capture->Select(program,uniforms,
        scheduler.CurrentFrame(),element_count,index_size,
        input.buffer==state.storage || (index_size && indices.buffer==state.storage),
        input_bytes+index_bytes+output_bytes,capture_config!=nullptr);
    if (capture_draw) {
        // The opt-in diagnostic needs completed fetched inputs before its readback.
        if(fetch_runtime) {
            fetch_runtime->EncodePica(*fetch_state,*prepared_fetch);
            scheduler.EndEncoding();fetch_runtime=nullptr;
        }
        static_assert(std::endian::native==std::endian::little);
        scheduler.MarkDiagnosticDraw();
        using D=VertexDiagnostic;
        diagnostic_header[D::Magic]=D::Signature;diagnostic_header[D::Version]=1;
        diagnostic_header[D::HeaderBytes]=sizeof(diagnostic_header);
        diagnostic_header[D::CodeWords]=program.code.size();
        diagnostic_header[D::SwizzleWords]=program.swizzle.size();
        diagnostic_header[D::UniformBytes]=sizeof(uniforms);
        diagnostic_header[D::StateBytes]=1824;
        diagnostic_header[D::VertexBytes]=sizeof(VertexRegisters);
        diagnostic_header[D::Entry]=program.entry;diagnostic_header[D::OutputMask]=program.output_mask;
        diagnostic_header[D::Vertices]=vertex_count;diagnostic_header[D::Elements]=element_count;
        diagnostic_header[D::VertexBase]=vertex_base;diagnostic_header[D::FirstVertex]=first_vertex;
        diagnostic_header[D::IndexWidth]=index_size;diagnostic_header[D::InstructionLimit]=instruction_limit;
        diagnostic_header[D::InitialReset]=state.reset;
        const auto split=[&](unsigned field,std::uint64_t value) {
            diagnostic_header[field]=std::uint32_t(value);diagnostic_header[field+1]=std::uint32_t(value>>32);
        };
        split(D::FrameLow,scheduler.CurrentFrame());split(D::DrawLow,capture->sequence);
        split(D::TickLow,scheduler.CurrentTick());split(D::RequestLow,capture->request);
        split(D::ProgramLow,Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
            Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission));
        std::array<std::uint32_t,17> diagnostic_control{uniforms.b};
        std::memcpy(diagnostic_control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
        split(D::ControlLow,Common::ComputeHash64(diagnostic_control.data(),sizeof(diagnostic_control)));
        diagnostic_header[D::InputBytes]=input_bytes;diagnostic_header[D::IndexBytes]=index_bytes;
        diagnostic_header[D::OutputBytes]=output_bytes;
        diagnostic_header[D::InputIndexAlias]=index_size && input.buffer==indices.buffer;
        append(diagnostic_header.data(),sizeof(diagnostic_header));
        append(program.code.data(),sizeof(program.code));append(program.swizzle.data(),sizeof(program.swizzle));
        append(&uniforms,sizeof(uniforms));
        const auto readback=device.CreateBuffer(1824+input_bytes+index_bytes,true);
        std::memset(readback.contents,0,1824);
        TransferRuntime transfer{device,scheduler};
        if (!state.reset) transfer.CopyBuffer(state.storage,0,readback,0,1824);
        transfer.CopyBuffer(input.buffer,input.offset,readback,1824,input_bytes);
        if(index_size) transfer.CopyBuffer(indices.buffer,indices.offset,readback,1824+input_bytes,index_bytes);
        scheduler.Finish(Scheduler::WaitReason::Readback);
        append(readback.contents,readback.length);
    }
    const auto upload=[&](const auto& value) {
        const auto slice=parameters.Allocate(sizeof(value));
        std::memcpy(slice.data,&value,sizeof(value));return slice;
    };
    const auto constants=upload(uniforms);
    std::array<std::uint32_t,30> config{program.entry,program.output_mask,0xffff,state.reset ? 1U:0U,
        1,instruction_limit,0,0,vertex_count,element_count,vertex_base,first_vertex,index_size,0,0,0,0,0,0,0};
    const auto status_slice=parameters.Allocate(16,16);
    const auto status=status_slice.buffer;
    const auto status_offset=status_slice.offset;
    std::memset(status_slice.data,0,16);
    std::array<std::uint32_t,17> control{uniforms.b};
    std::memcpy(control.data()+1,uniforms.i.data(),sizeof(uniforms.i));
    auto& proof=*GetVertexProof(cached,uniforms);
    if (profile_draw && !proof.profile_dumped) {
        const auto* profile=std::getenv("AZAHAR_METAL4_PROFILE");
        const auto hash = Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
            Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission);
        const auto path = std::string(profile) + ".shader-" + std::to_string(hash) + "-" +
            std::to_string(Common::ComputeHash64(control.data(),sizeof(control))) + ".bin";
        std::ofstream dump(path, std::ios::binary | std::ios::trunc);
        const std::array<std::uint32_t,12> header{0x34544c4d,1,4096,4096,sizeof(uniforms),
            program.entry,program.output_mask,program.allow_emission ? 1U:0U,
            static_cast<std::uint32_t>(proof.execution),proof.passes,proof.unproven,0};
        dump.write(reinterpret_cast<const char*>(header.data()),sizeof(header));
        dump.write(reinterpret_cast<const char*>(program.code.data()),sizeof(program.code));
        dump.write(reinterpret_cast<const char*>(program.swizzle.data()),sizeof(program.swizzle));
        dump.write(reinterpret_cast<const char*>(&uniforms),sizeof(uniforms));
        if (!dump) throw std::runtime_error("Unable to write requested Metal 4 shader diagnostic");
        proof.profile_dumped=true;
    }
    if (capture_draw) {
        using D=VertexDiagnostic;
        diagnostic_header[D::Execution]=static_cast<std::uint32_t>(proof.execution);
        diagnostic_header[D::Passes]=proof.passes;
        diagnostic_header[D::RecurrenceLow]=proof.recurrence_mask[0];
        diagnostic_header[D::RecurrenceHigh]=proof.recurrence_mask[1];
        diagnostic_header[D::RecurrenceCapacity]=proof.recurrence_capacity;
    }
    // Small dependent ranges execute the original ordered GPU kernel in one
    // dispatch. Keeping the complete unit in a single GPU thread avoids the
    // cooperative carry/seed prefix and its threadgroup barriers.
    const bool small_dependency=proof.execution==VertexExecution::DependencyPasses &&
        !index_size && element_count<=8 && input.buffer!=state.storage;
    std::shared_ptr<ShaderMachineState::DeferredBatch> deferred_state;
    if(owned_input && small_dependency && state.reset && input.lease && fetch_runtime &&
       !generic_geometry && defer_validation && !capture && instruction_limit==1000000) {
        if(!cached.retained_source)
            cached.retained_source=std::make_shared<const ShaderMachineProgram>(program);
        deferred_state=std::make_shared<ShaderMachineState::DeferredBatch>();
        deferred_state->program=cached.retained_source;deferred_state->uniforms=uniforms;
        // PreparedPica owns this immutable canonical output. Unlike the public
        // borrowed-input API, no caller can enqueue writes to this range.
        deferred_state->input=input;
        deferred_state->vertex_count=vertex_count;deferred_state->element_count=element_count;
        deferred_state->vertex_base=vertex_base;deferred_state->first_vertex=first_vertex;
        deferred_state->index_size=0;deferred_state->instruction_limit=instruction_limit;
    }
    const bool bounded_execution=deferred_state && proof.instruction_bound;
    // This proof bounds every conditional path per invocation independently of
    // guest float values. FIFO order, inherited registers between vertices and
    // complete state/status publication remain the ordinary ordered GPU path.
    const bool bounded_ordered=!generic_geometry && state.reset &&
        proof.execution==VertexExecution::Ordered && instruction_limit==1000000 && proof.instruction_bound;
    // Measured six-vertex/two-pass class only. Full inherited/diagnostic state
    // still uses its existing guarded GPU transport. No guest values are assumed.
    const bool owned_simd=bounded_execution && element_count==6 && proof.passes==2;
    const auto execution=proof.execution;
    const bool recurrence=execution==VertexExecution::PrivateRecurrences;
    config[16]=proof.recurrence_mask[0];config[17]=proof.recurrence_mask[1];
    // Small batches keep a single native merge dispatch; the SIMD resolver
    // amortizes its additional dispatch on batches of at least 128 elements.
    config[14]=element_count>=128u;
    config[18]=proof.recurrence_capacity;
    config[19]=input.buffer==state.storage || (index_size && indices.buffer==state.storage);
    if(small_dependency) config[13]=proof.passes;
    std::copy(proof.always_written.begin(),proof.always_written.end(),config.begin()+20);
    std::copy(proof.possibly_written.begin(),proof.possibly_written.end(),config.begin()+25);
    const bool shared_plan=fetch_runtime && index_size && execution!=VertexExecution::Ordered;
    config[15]=shared_plan;
    const auto params=upload(config);
    const bool independent=execution!=VertexExecution::Ordered;
    const bool dependency=!small_dependency && (execution==VertexExecution::DependencyPasses || recurrence);
    const bool merge_state=!small_dependency && (execution==VertexExecution::MergeState || dependency);
    const bool indexed_plan=(merge_state || shared_plan) && index_size;
    // These private ranges have the same submission lifetime. Pack them without
    // overlap to avoid a separate allocation/residency entry for every scratch
    // bank. Keep large valid batches within each native buffer's device limit.
    // Their arithmetic, dispatch order and device barriers remain unchanged.
    const std::array<NSUInteger,5> storage_bytes{
        NSUInteger(element_count)*sizeof(VertexRegisters),
        merge_state ? NSUInteger(element_count)*432:0,
        dependency ? NSUInteger(element_count)*288:0,
        indexed_plan ? (NSUInteger(element_count)+(shared_plan ? 1u:0u))*8:0,
        recurrence ? NSUInteger(element_count)*(config[18]+1u)*16:0};
    std::array<BufferBinding,5> storage{};
    std::array<BufferSlice,5> storage_leases{};
    const auto limit=device.Handle().maxBufferLength;
    for(const auto bytes:storage_bytes)
        if(bytes>limit) throw std::invalid_argument("Metal buffer size exceeds the device limits");
    for(unsigned first=0;first<storage.size();) {
        if(!storage_bytes[first]) {storage[first++].buffer=empty_input;continue;}
        unsigned last=first;NSUInteger size=0;
        for(;last<storage.size();++last) {
            if(!storage_bytes[last]) continue;
            const NSUInteger padding=(-size)&255u;
            if(padding>limit-size || storage_bytes[last]>limit-size-padding) break;
            storage[last].offset=size+padding;
            size+=padding+storage_bytes[last];
        }
        storage_leases[first]=AcquireBatchStorage(size);
        const auto buffer=storage_leases[first].buffer;
        for(unsigned i=first;i<last;++i)
            storage[i].buffer=storage_bytes[i] ? buffer:empty_input;
        first=last;
    }
    const auto result=storage[0].buffer;
    BufferSlice capture_params{},captured{};
    if(capture_config) {
        struct CaptureParams {GeometryEmissionConfig config;std::array<std::uint32_t,4> count;};
        static_assert(sizeof(CaptureParams)==80);
        const CaptureParams info{*capture_config,{element_count,program.output_mask,0,0}};
        // Upload and allocate before argument registration or opening the VS encoder.
        capture_params=parameters.Allocate(sizeof(info));std::memcpy(capture_params.data,&info,sizeof(info));
        captured=AcquireBatchStorage(NSUInteger(element_count)*sizeof(CapturedVertex));
    }
    std::array<BufferBinding,15> bindings{{
        {cached.instructions,0,0},{cached.instructions,sizeof(program.code),1},{constants.buffer,constants.offset,2},
        {input.buffer,input.offset,3},{state.storage,0,4},{result,0,5},{empty_input,0,6},
        {status,status_offset,7},{params.buffer,params.offset,8},{index_size ? indices.buffer:empty_input,index_size ? indices.offset:0,9},
        {storage[1].buffer,storage[1].offset,10},{storage[2].buffer,storage[2].offset,11},
        {storage[3].buffer,storage[3].offset,12},{empty_input,0,13},
        {storage[4].buffer,storage[4].offset,14}}};
    // Every variant publishes its complete native vertex/finalizer set only
    // after all required compiler/ABI checks have succeeded. The worker owns
    // immutable program/proof snapshots, never a cache node or this runtime.
    const bool reset_specialized=state.reset;
    const auto name=owned_simd ? "pica_vertex_machine_small_dependency":small_dependency ? "pica_vertex_machine_batch":recurrence ? "pica_vertex_machine_recurrence":dependency ? "pica_vertex_machine_dependency":merge_state ?
        "pica_vertex_machine_carry":independent ? "pica_vertex_machine_parallel":"pica_vertex_machine_batch";
    struct PipelineRef {
        id<MTLComputePipelineState> ready{};
        std::shared_future<id<MTLComputePipelineState>> pending{};
    };
    PipelineRef vertex_ref,finalize_ref,simd_finalize_ref;
    NSString* pipeline_label=nil;
    if(generic_geometry) {
        vertex_ref.ready=geometry_vertex_pipelines[small_dependency ? 5u:static_cast<unsigned>(execution)];
        finalize_ref.ready=geometry_finalize_pipeline;simd_finalize_ref.ready=geometry_simd_finalize_pipeline;
        pipeline_label=vertex_ref.ready.label;
    } else {
        auto& recurrence_pipelines=reset_specialized ? cached.reset_recurrence_pipelines:cached.recurrence_pipelines;
        auto& vertex_pipelines=reset_specialized ? cached.reset_vertex_pipelines:cached.vertex_pipelines;
        if(recurrence && recurrence_pipelines.size()==128 && !recurrence_pipelines.contains(proof.recurrence_mask))
            recurrence_pipelines.clear();
        std::array<std::uint32_t,10> write_masks{};
        if(owned_simd) {
            std::copy(proof.always_written.begin(),proof.always_written.end(),write_masks.begin());
            std::copy(proof.possibly_written.begin(),proof.possibly_written.end(),write_masks.begin()+5);
            if(cached.owned_simd_vertex_pipelines.size()==128 && !cached.owned_simd_vertex_pipelines.contains(write_masks))
                cached.owned_simd_vertex_pipelines.clear();
        }
        auto& selected=bounded_ordered ? cached.reset_bounded_ordered_vertex_pipeline:owned_simd ? cached.owned_simd_vertex_pipelines[write_masks]:bounded_execution ? cached.owned_bounded_vertex_pipeline:deferred_state ? cached.owned_small_vertex_pipeline:recurrence ?
            recurrence_pipelines[proof.recurrence_mask]:vertex_pipelines[small_dependency ? 5u:static_cast<unsigned>(execution)];
        if(selected.vertex.valid() && selected.vertex.wait_for(std::chrono::seconds{0})==std::future_status::ready) {
            try { (void)selected.vertex.get(); } catch(...) { selected={}; }
        }
        if(!selected.vertex.valid()) {
            struct Publication { std::array<std::promise<id<MTLComputePipelineState>>,3> values; };
            auto publication=std::make_shared<Publication>();
            BatchPipelines pending{publication->values[0].get_future().share(),publication->values[1].get_future().share(),publication->values[2].get_future().share()};
            auto owned=std::make_shared<const ShaderMachineProgram>(program);
            (void)device.DeferCompilation([native=&device,publication,owned,proof,name,owned_simd,merge_state,reset_specialized,
                small_dependency,defer_full_state=bool(deferred_state),bounded_execution,bounded_ordered,execution,
                independent,profile=profile_vertices] {
                try {
                    const auto& program=*owned;
                    const auto specialized=native->CompileLibrary(owned_simd ? OwnedDependencySource(program,proof.recurrence_mask,proof.always_written,proof.possibly_written):SpecializedVertexSource(program,name,merge_state,proof.recurrence_mask,reset_specialized,small_dependency,defer_full_state,bounded_execution,bounded_ordered),
                        "Azahar Metal 4 specialized PICA vertex machine");
                    auto* descriptor=[[MTL4ComputePipelineDescriptor alloc] init];
                    descriptor.computeFunctionDescriptor=Device::Function(specialized,name);
                    if(profile) {
                        const auto hash=Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
                            Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission);
                        descriptor.label=[NSString stringWithFormat:@"%s id=%llu",name,static_cast<unsigned long long>(hash)];
                    }
                    descriptor.maxTotalThreadsPerThreadgroup=owned_simd ? 32:independent && !small_dependency ? 64:1;
                    std::array<id<MTLComputePipelineState>,3> ready{native->CompileCompute(descriptor),nil,nil};
                    if(owned_simd && ready[0].threadExecutionWidth!=32)
                        throw std::runtime_error("Owned dependency execution requires a32lane SIMD group");
                    if(execution==VertexExecution::Independent) {
                        descriptor.computeFunctionDescriptor=Device::Function(specialized,"pica_vertex_machine_finalize");
                        descriptor.label=nil;descriptor.maxTotalThreadsPerThreadgroup=1;
                        ready[1]=native->CompileCompute(descriptor);
                        descriptor.maxTotalThreadsPerThreadgroup=64;ready[2]=native->CompileCompute(descriptor);
                        if(ready[2].threadExecutionWidth<16 || ready[2].threadExecutionWidth>64)
                            throw std::runtime_error("Native vertex finalization requires a SIMD width between 16 and 64");
                    }
                    if(bounded_ordered) LOG_INFO(Render,
                        "Metal4 bounded ordered vertex active: entry={}, output mask={:#x}, proved instruction bound={}",
                        program.entry,program.output_mask,proof.instruction_bound);
                    if(small_dependency) LOG_INFO(Render,
                        "Metal4 small vertex dependency active: entry={}, output mask={:#x}, dependency depth={}, deferred full state={}, proved instruction bound={}, SIMD prefix={}",
                        program.entry,program.output_mask,proof.passes,defer_full_state,bounded_execution ? proof.instruction_bound:0u,owned_simd);
                    for(unsigned i=0;i<ready.size();++i)publication->values[i].set_value(ready[i]);
                } catch(...) {
                    const auto failure=std::current_exception();
                    for(auto& value:publication->values)try{value.set_exception(failure);}catch(const std::future_error&){}
                }
            },sizeof(ShaderMachineProgram)+sizeof(VertexProof));
            selected=std::move(pending);
        }
        vertex_ref.pending=selected.vertex;finalize_ref.pending=selected.finalize;simd_finalize_ref.pending=selected.simd_finalize;
        if(profile_vertices) {
            const auto hash=Common::HashCombine(Common::ComputeHash64(program.code.data(),sizeof(program.code)),
                Common::ComputeHash64(program.swizzle.data(),sizeof(program.swizzle)),program.entry,program.output_mask,program.allow_emission);
            pipeline_label=[NSString stringWithFormat:@"%s id=%llu",name,static_cast<unsigned long long>(hash)];
        }
    }
    // Prepare all uploads before opening the bounded compute pass: recycling an
    // upload page can submit/wait and would invalidate a live encoder reference.
    BufferSlice seeded_params{};
    if(dependency) {
        auto seeded_config=config;seeded_config[13]=1;
        seeded_params=upload(seeded_config);
    }
    // Every upload precedes fetch encoding. Separate binding snapshots replace
    // the formerly mutable cache-owned argument tables.
    const std::vector<BufferBinding> initial_bindings{bindings.begin(),bindings.end()};
    auto seeded_bindings=initial_bindings;
    if(dependency) seeded_bindings[8]={seeded_params.buffer,seeded_params.offset,8};
    std::vector<ComputeDispatch> dispatches;
    const auto dispatch=[&](const PipelineRef& selected,const std::vector<BufferBinding>& buffers,
                            NSUInteger count,NSUInteger width=0,NSUInteger simd_groups=0) {
        ComputeDispatch command;
        command.pipeline=selected.ready;command.pending_pipeline=selected.pending;command.buffers=buffers;
        if(simd_groups) command.simd_groups=simd_groups;
        else {
            command.threads=MTLSizeMake(count,1,1);
            command.group=MTLSizeMake(width ? width:count==1 ? 1:64,1,1);
        }
        dispatches.push_back(std::move(command));
    };
    if(indexed_plan && !shared_plan) dispatch({plan_pipeline,{}},initial_bindings,1,0,1);
    if(small_dependency) dispatch(vertex_ref,initial_bindings,owned_simd ? 32:1,owned_simd ? 32:1);
    else if(dependency) {
        for(unsigned pass=0;pass<proof.passes;++pass) {
            dispatch(vertex_ref,pass ? seeded_bindings:initial_bindings,element_count);
            if(pass+1<proof.passes) dispatch({prefix_pipeline,{}},initial_bindings,
                ((element_count+63u)&~63u)+72u*unsigned(prefix_pipeline.threadExecutionWidth));
        }
    } else dispatch(vertex_ref,initial_bindings,independent ? element_count:1);
    if(merge_state) {
        if(config[14]) dispatch({resolve_pipeline,{}},initial_bindings,133,0,133);
        dispatch({merge_pipeline,{}},initial_bindings,133,config[19] ? 256:0);
    } else if(independent && !small_dependency) {
        if(index_size) dispatch(simd_finalize_ref,initial_bindings,1,0,1);
        else dispatch(finalize_ref,initial_bindings,1);
    }
    if(capture_config) {
        const std::vector<BufferBinding> capture_bindings{{result,0,0},
            {capture_params.buffer,capture_params.offset,1},{captured.buffer,captured.offset,2}};
        dispatch({this->vertex_pipeline,{}},capture_bindings,element_count,64);
    }
    if(fetch_runtime) {
        auto fetch_commands=fetch_runtime->PreparePicaDispatches(*fetch_state,*prepared_fetch,
            shared_plan ? BufferSlice{storage[3].buffer,storage[3].offset,storage_bytes[3],nullptr,{}}:BufferSlice{},
            shared_plan ? status_slice:BufferSlice{});
        dispatches.insert(dispatches.begin(),std::make_move_iterator(fetch_commands.begin()),
            std::make_move_iterator(fetch_commands.end()));
    }
    NSString* label=nil;
    if(profile_draw) label=[NSString stringWithFormat:
        fetch_runtime ? @"fetch+history+VS+capture %@ control=%llu elements=%u vertices=%u index=%u passes=%u reset=%u":
                  @"pica_vertex_batch %@ control=%llu elements=%u vertices=%u index=%u passes=%u reset=%u",
        pipeline_label,static_cast<unsigned long long>(Common::ComputeHash64(control.data(),sizeof(control))),
        element_count,vertex_count,index_size,proof.passes,config[3]];
    try { scheduler.QueueCompute(dispatches,{},label); }
    catch(...) { scheduler.EndEncoding();throw; }
    if(fetch_runtime) fetch_runtime->CommitPica(*fetch_state);
    // A continuation remains owned by the worker until a direct consumer,
    // submission or explicit boundary drains it. Do not synchronously drain a
    // cold compiler task just because this deferred API is returning.
    if(!defer_validation || capture_draw) scheduler.EndEncoding();
    // Completion checks run before the upload arena recycles a page. Retain its
    // buffer and offset without extending the CPU lease across submissions.
    const auto validate=[status,status_offset] {
        const auto* flags=reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::byte*>(status.contents)+status_offset);
        if (flags[0]) throw ShaderMachineError(flags[0],flags[3]);
    };
    if (defer_validation && !capture_draw) scheduler.CheckAfterCompletion(validate);
    else { scheduler.Finish(Scheduler::WaitReason::Readback);validate(); }
    if(deferred_state) {
        state.deferred=std::move(deferred_state);state.reset=true;
    } else state.reset=false;
    if (capture_draw) {
        const auto readback=device.CreateBuffer(output_bytes+1824,true);
        TransferRuntime transfer{device,scheduler};
        transfer.CopyBuffer(result,0,readback,0,output_bytes);
        transfer.CopyBuffer(state.storage,0,readback,output_bytes,1824);
        scheduler.Finish(Scheduler::WaitReason::Readback);append(readback.contents,readback.length);
        std::memcpy(diagnostic.data(),diagnostic_header.data(),sizeof(diagnostic_header));
        capture->Write(diagnostic,scheduler.CurrentFrame(),capture_config);
    }
    if(captured_output) *captured_output=std::move(captured);
    return {result,0,storage_bytes[0],nullptr,storage_leases[0].lease};
}
namespace {
void ValidateEmissionConfig(const GeometryEmissionConfig& config) {
    if(config.attribute_count>7 || config.flip_viewport>1 || config.enable_clip>1 ||
        config.fix_quaternions>1 || config.honor_winding>1)
        throw std::invalid_argument("Invalid native geometry emission configuration");
    for(unsigned a=0;a<config.attribute_count;++a) for(unsigned c=0;c<4;++c) {
        const auto semantic=(config.semantics[a]>>(c*8))&31;
        if(semantic>=24 && semantic!=31) throw std::invalid_argument("Unknown native geometry output semantic");
    }
}
}

AssembledVertices ShaderMachineRuntime::ConvertEmissionListIndirect(const ShaderMachinePendingResult& pending,
    const GeometryEmissionConfig& config,PrimitiveStream& stream,const BufferSlice& destination) {
    (void)scheduler.LastUseTick(nil);
    ValidateEmissionConfig(config);
    const auto valid=[&](const BufferSlice& slice,NSUInteger bytes,NSUInteger alignment) {
        return slice.buffer && slice.buffer.device==device.Handle() && slice.offset%alignment==0 &&
            slice.offset<=slice.buffer.length && slice.size<=slice.buffer.length-slice.offset && bytes<=slice.size;
    };
    const auto storage_capacity=EmissionStorageCapacity(pending);
    if(pending.capacity<3 || pending.capacity>1000000 || storage_capacity<3 || stream.history.device!=device.Handle() ||
        stream.topology!=TriangleTopology::List || stream.pending ||
        !valid(pending.status,16,4) || !pending.status.buffer.contents ||
        !valid(pending.result.emissions,NSUInteger(storage_capacity)*sizeof(ShaderMachineEmission),16))
        throw std::invalid_argument("Invalid native indirect GS count/source/list");
    const auto capacity=storage_capacity-storage_capacity%3;
    const auto bytes=NSUInteger(capacity)*sizeof(CapturedVertex);
    if(destination.buffer && (!valid(destination,bytes,16) || destination.buffer==pending.result.emissions.buffer ||
        destination.buffer==pending.status.buffer || destination.buffer==stream.history))
        throw std::invalid_argument("Invalid native indirect GS destination");
    const auto previous=stream.deferred_list;
    if(previous.buffer && (!valid(previous,64,16) || !previous.buffer.contents))
        throw std::invalid_argument("Invalid prior native indirect list metadata");
    const auto output=destination.buffer ? destination:pending.storage_proof ? scalar_outputs.Allocate(bytes):
        AcquireBatchStorage(bytes,pending.result.emissions.buffer,pending.status.buffer);
    const auto args=scalar_outputs.Allocate(64);
    struct Params { GeometryEmissionConfig config;std::array<std::uint32_t,4> count; };
    const Params info{config,{capacity,0,0,1}};
    const std::array<std::uint32_t,4> count_info{storage_capacity,stream.needs_reset ? 1u:0u,stream.history_offset,previous.buffer ? 1u:0u};
    const auto params=parameters.Allocate(sizeof(info)),count_params=parameters.Allocate(sizeof(count_info));
    std::memcpy(params.data,&info,sizeof(info));std::memcpy(count_params.data,count_info.data(),sizeof(count_info));
    const std::array<BufferBinding,5> count_bindings{{
        {pending.status.buffer,pending.status.offset,0},{count_params.buffer,count_params.offset,1},
        {args.buffer,args.offset,2},{stream.history,0,3},
        {previous.buffer ? previous.buffer:args.buffer,previous.buffer ? previous.offset:args.offset,4}}};
    const auto count_table=scheduler.UpdateArguments(emission_count_arguments,count_bindings);
    const std::array<BufferBinding,5> bindings{{
        {pending.result.emissions.buffer,pending.result.emissions.offset,0},{params.buffer,params.offset,1},
        {output.buffer,output.offset,2},{stream.history,0,3},{args.buffer,args.offset,4}}};
    const auto table=scheduler.UpdateArguments(indirect_emission_arguments,bindings);
    scheduler.KeepAlive(emission_count_pipeline);scheduler.KeepAlive(emission_pipeline);
    const auto encoder=scheduler.BeginCompute(emission_count_pipeline);
    [encoder setComputePipelineState:emission_count_pipeline];[encoder setArgumentTable:count_table];
    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
    [encoder setComputePipelineState:emission_pipeline];[encoder setArgumentTable:table];
    [encoder dispatchThreadgroupsWithIndirectBuffer:args.buffer.gpuAddress+args.offset+16
                            threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    scheduler.EndCompute();
    auto proof=std::shared_ptr<NativeIndirectDraw>{new NativeIndirectDraw};
    proof->arguments=args;proof->status=pending.status;proof->capacity=capacity;proof->tick=scheduler.CurrentTick();
    stream.deferred_list=args;stream.known_zero_history=false;
    return {output,capacity,std::move(proof)};
}
void ShaderMachineRuntime::ValidateIndirectResult(const AssembledVertices& vertices) const {
    (void)scheduler.LastUseTick(nil);
    if(!vertices.indirect || vertices.indirect->capacity!=vertices.vertex_count ||
        !scheduler.IsComplete(vertices.indirect->tick))
        throw std::invalid_argument("Native indirect result is missing or not complete");
    const auto& proof=*vertices.indirect;
    const auto* status=reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(proof.status.buffer.contents)+proof.status.offset);
    const auto* args=reinterpret_cast<const std::uint32_t*>(static_cast<const std::byte*>(proof.arguments.buffer.contents)+proof.arguments.offset);
    if(status[0]) throw ShaderMachineError(status[0],status[3]);
    if(args[12]) throw ShaderMachineError(args[12],status[3]);
}

AssembledVertices ShaderMachineRuntime::ConvertEmissions(const ShaderMachineResult& source,
    const GeometryEmissionConfig& config) {
    return ConvertEmissionsImpl(source,config,nullptr);
}
AssembledVertices ShaderMachineRuntime::ConvertEmissionList(const ShaderMachineResult& source,
    const GeometryEmissionConfig& config,PrimitiveStream& stream) {
    (void)scheduler.LastUseTick(nil);
    if (stream.history.device!=device.Handle() ||
        (stream.topology==TriangleTopology::List && stream.pending))
        throw std::invalid_argument("Complete GS list requires an empty list stream on the owning device");
    return ConvertEmissionsImpl(source,config,&stream);
}
AssembledVertices ShaderMachineRuntime::ConvertEmissionsImpl(const ShaderMachineResult& source,
    const GeometryEmissionConfig& config,PrimitiveStream* stream) {
    ValidateEmissionConfig(config);
    if(source.emission_count%3) throw std::invalid_argument("Invalid native geometry emission count");
    if (!source.emission_count) return {};
    const auto& input=source.emissions;
    if (!input.buffer || input.buffer.device!=device.Handle() || input.offset%16 ||
        input.offset>input.buffer.length || input.size>input.buffer.length-input.offset ||
        std::uint64_t(source.emission_count)*sizeof(ShaderMachineEmission)>input.size)
        throw std::invalid_argument("Native geometry emission buffer is missing, misaligned or too small");
    struct Params { GeometryEmissionConfig config; std::array<std::uint32_t,4> count; };
    static_assert(sizeof(Params)==80);
    const Params info{config,{source.emission_count,0,stream ? 1u:0u,0}};
    const auto params=parameters.Allocate(sizeof(info));
    std::memcpy(params.data,&info,sizeof(info));
    const auto output=AcquireBatchStorage(NSUInteger(source.emission_count)*sizeof(CapturedVertex));
    // Preserve the destination-half parity of the former per-triangle appends.
    const auto next_history_offset=stream ? stream->history_offset ^ ((source.emission_count/3)%2 ? 272u:0u):0u;
    const std::array<BufferBinding,5> bindings{{
        {input.buffer,input.offset,0},{params.buffer,params.offset,1},{output.buffer,output.offset,2},
        {stream ? stream->history:input.buffer,stream ? next_history_offset:input.offset,3},
        {input.buffer,input.offset,4}}};
    if (emission_arguments) scheduler.RebindBufferArguments(emission_arguments,bindings);
    else emission_arguments=scheduler.MakeArguments(bindings);
    const auto args=emission_arguments;
    scheduler.KeepAlive(emission_pipeline);
    const auto encoder=scheduler.BeginCompute(emission_pipeline);
    [encoder setComputePipelineState:emission_pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(source.emission_count/3,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    scheduler.EndEncoding();
    if (stream) {
        stream->history_offset=next_history_offset;
        stream->topology=TriangleTopology::List;stream->pending=0;
        stream->needs_reset=false;stream->known_zero_history=true;
        stream->deferred_list={};
    }
    return {output,source.emission_count};
}
BufferSlice ShaderMachineRuntime::CaptureVertex(const BufferSlice& input,std::uint32_t mask,
    const GeometryEmissionConfig& config) {
    return CaptureVertices(input,mask,config,1);
}
BufferSlice ShaderMachineRuntime::CaptureVertices(const BufferSlice& input,std::uint32_t mask,
    const GeometryEmissionConfig& config,std::uint32_t count) {
    if(!input.buffer || input.buffer.device!=device.Handle() ||
       !count || input.offset%16 || input.offset>input.buffer.length || input.size>input.buffer.length-input.offset ||
       std::uint64_t(count)*sizeof(VertexRegisters)>input.size)
        throw std::invalid_argument("Invalid native raw VS capture input/configuration");
    ValidateVertexCaptureConfig(mask,config);
    struct Params { GeometryEmissionConfig config;std::array<std::uint32_t,4> count; };
    static_assert(sizeof(Params)==80);
    const Params info{config,{count,mask,0,0}};
    const auto params=parameters.Allocate(sizeof(info));std::memcpy(params.data,&info,sizeof(info));
    const auto output=AcquireBatchStorage(NSUInteger(count)*sizeof(CapturedVertex));
    const std::array<BufferBinding,3> bindings{{{input.buffer,input.offset,0},{params.buffer,params.offset,1},{output.buffer,output.offset,2}}};
    if (capture_arguments) scheduler.RebindBufferArguments(capture_arguments,bindings);
    else capture_arguments=scheduler.MakeArguments(bindings);
    const auto args=capture_arguments;scheduler.KeepAlive(vertex_pipeline);
    const auto encoder=scheduler.BeginCompute(vertex_pipeline);[encoder setComputePipelineState:vertex_pipeline];[encoder setArgumentTable:args];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];scheduler.EndEncoding();
    return output;
}
ShaderMachineSnapshot ShaderMachineRuntime::SaveState(const ShaderMachineState& state) {
    if (state.storage.device!=device.Handle()) throw std::invalid_argument("Shader state belongs to another device");
    MaterializeDeferred(state);
    ShaderMachineSnapshot result{};result.needs_reset=state.reset;
    if (!state.reset) {
        const auto readback=device.CreateBuffer(result.history.size(),true);
        TransferRuntime transfer{device,scheduler};
        transfer.CopyBuffer(state.storage,0,readback,0,readback.length);scheduler.Finish(Scheduler::WaitReason::Readback);
        std::memcpy(result.history.data(),readback.contents,result.history.size());
    }
    return result;
}
void ShaderMachineRuntime::ValidateState(const ShaderMachineSnapshot& snapshot) {
    if (snapshot.needs_reset) {
        if (std::any_of(snapshot.history.begin(),snapshot.history.end(),[](std::byte v) { return v!=std::byte{}; }))
            throw std::invalid_argument("Reset shader snapshot contains noncanonical state");
        return;
    }
    const auto word=[&](unsigned offset) {
        std::uint32_t value{};
        for(unsigned i=0;i<4;++i) value|=std::uint32_t(snapshot.history[offset+i])<<(i*8);
        return value;
    };
    if (word(1804)!=0 || word(1808)>1 || word(1812)>1 || word(1816)>1 || word(1820)>15)
        throw std::invalid_argument("Shader snapshot contains invalid condition/output-bank/emitter flags");
}
void ShaderMachineRuntime::LoadState(ShaderMachineState& state,const ShaderMachineSnapshot& snapshot) {
    if (state.storage.device!=device.Handle()) throw std::invalid_argument("Shader state belongs to another device");
    ValidateState(snapshot);
    if (!snapshot.needs_reset) {
        const auto source=parameters.Allocate(snapshot.history.size());
        std::memcpy(source.data,snapshot.history.data(),snapshot.history.size());
        TransferRuntime transfer{device,scheduler};
        transfer.CopyBuffer(source.buffer,source.offset,state.storage,0,snapshot.history.size());
    }
    state.deferred.reset();
    state.reset=snapshot.needs_reset;
}
}
