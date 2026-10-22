// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <string_view>
namespace Metal {
inline constexpr std::string_view VertexMeshShaderGLSL=R"GLSL(layout(set=4,binding=0,std140) uniform MeshParams {
 uint elements;uint index_size;uint vertex_base;uint first_vertex;
 uint topology;uint pending;uint reset;uint fix_quaternions;
 uint triangles;uint debug_output;uint reserved0;uint reserved1;
} mesh_params;
layout(set=4,binding=1,std430) readonly buffer MeshIndices {uint8_t mesh_indices[];};
layout(set=4,binding=2,std430) readonly buffer MeshHistory {
 AzaharCapturedVertex history_vertices[2];uint history_index;uint history_ready;uint history_valid[2];
};
layout(set=4,binding=3,std430) writeonly buffer MeshNextHistory {
 AzaharCapturedVertex next_vertices[2];uint next_index;uint next_ready;uint next_valid[2];
};
out gl_MeshPerVertexEXT {vec4 gl_Position;float gl_ClipDistance[2];} gl_MeshVerticesEXT[];
layout(location=1) out vec4 mesh_color[];
layout(location=2) out vec2 mesh_tex0[];
layout(location=3) out vec2 mesh_tex1[];
layout(location=4) out vec2 mesh_tex2[];
layout(location=5) out float mesh_texw[];
layout(location=6) out vec4 mesh_quat[];
layout(location=7) out vec3 mesh_view[];
shared AzaharCapturedVertex mesh_v[96];
shared uint mesh_valid[96];
AzaharCapturedVertex mesh_zero() {
 return AzaharCapturedVertex(vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0),vec4(0));
}
uint mesh_source_index(uint element) {
 if(mesh_params.index_size==0)return mesh_params.first_vertex+element;
 uint byte_offset=element*mesh_params.index_size;
 uint value=0u;
 for(uint b=0u;b<mesh_params.index_size;++b)value|=uint(mesh_indices[byte_offset+b])<<(b*8u);
 return value;
}
AzaharCapturedVertex mesh_stream_vertex(uint logical,out bool valid) {
 if(logical<mesh_params.pending) {
  uint slot=logical;
  if(mesh_params.topology==1u && mesh_params.pending==2u)slot^=history_index;
  valid=mesh_params.reset==0u && history_valid[slot]!=0u;
  return valid ? history_vertices[slot]:mesh_zero();
 }
 uint index=mesh_source_index(logical-mesh_params.pending);
 valid=index>=mesh_params.vertex_base && index-mesh_params.vertex_base<azahar_capture.vertex_count;
 return valid ? azahar_evaluate(index-mesh_params.vertex_base):mesh_zero();
}
void mesh_write_history() {
 uint total=mesh_params.pending+mesh_params.elements;
 uint remaining=mesh_params.topology==0u ? total%3u:min(total,2u);
 uint next_slot=mesh_params.topology==1u ?
   ((mesh_params.reset==0u ? history_index:0u)+mesh_params.elements)&1u:remaining;
 bool valid_a=false,valid_b=false;
 AzaharCapturedVertex a=mesh_zero(),b=mesh_zero();
 if(remaining!=0u) {
  uint first=mesh_params.topology==2u ? 0u:total-remaining;
  a=mesh_stream_vertex(first,valid_a);
  if(remaining==2u)b=mesh_stream_vertex(total-1u,valid_b);
 }
 uint slot=mesh_params.topology==1u && remaining==2u ? next_slot:0u;
 next_vertices[slot]=a;next_vertices[slot^1u]=b;
 next_valid[slot]=valid_a ? 1u:0u;next_valid[slot^1u]=valid_b ? 1u:0u;
 next_index=mesh_params.topology==2u ? 1u:next_slot;
 next_ready=mesh_params.topology!=0u && remaining==2u ? 1u:0u;
}
void main() {
 uint local=gl_LocalInvocationIndex;
 uint first_triangle=mesh_params.reserved1+gl_WorkGroupID.x*32u;
 uint triangle=first_triangle+local/3u;
 uint primitive_count=first_triangle<mesh_params.triangles ? min(32u,mesh_params.triangles-first_triangle):0u;
 bool live=triangle<mesh_params.triangles;
 SetMeshOutputsEXT(primitive_count*3u,primitive_count);
 if(triangle==0u && local==0u)mesh_write_history();
 if(live) {
  uvec3 elements;
  if(mesh_params.topology==0u)elements=uvec3(triangle*3u,triangle*3u+1u,triangle*3u+2u);
  else if(mesh_params.topology==1u) {
   elements=uvec3(triangle,triangle+1u,triangle+2u);
   uint parity=mesh_params.pending==2u && mesh_params.reset==0u ? history_index:0u;
   if(((triangle^parity)&1u)!=0u)elements=elements.yxz;
  }else elements=uvec3(0u,triangle+1u,triangle+2u);
  bool valid;
  mesh_v[local]=mesh_stream_vertex(elements[local%3u],valid);
  mesh_valid[local]=valid ? 1u:0u;
 }
 barrier();
 if(live) {
  uint first_vertex=(local/3u)*3u;
  AzaharCapturedVertex v=mesh_v[local];
  if((mesh_valid[first_vertex]&mesh_valid[first_vertex+1u]&mesh_valid[first_vertex+2u])==0u) {
   v=mesh_zero();v.position=vec4(0,0,2,1);v.clip=vec4(-1,-1,0,0);
  }else if(mesh_params.fix_quaternions!=0u && local%3u!=0u && dot(mesh_v[first_vertex].quaternion,v.quaternion)<0.0) {
   v.quaternion=-v.quaternion;
  }
  if(mesh_params.debug_output!=0u)azahar_outputs[triangle*3u+local%3u]=v;
  gl_MeshVerticesEXT[local].gl_Position=v.position;
  gl_MeshVerticesEXT[local].gl_ClipDistance[0]=v.clip.x;
  gl_MeshVerticesEXT[local].gl_ClipDistance[1]=v.clip.y;
  mesh_color[local]=v.color;
  mesh_tex0[local]=v.texcoord0.xy;mesh_tex1[local]=v.texcoord1.xy;mesh_tex2[local]=v.texcoord2.xy;
  mesh_texw[local]=v.clip.z;mesh_quat[local]=v.quaternion;mesh_view[local]=v.view.xyz;
 }
 if(local<primitive_count)gl_PrimitiveTriangleIndicesEXT[local]=uvec3(local*3u,local*3u+1u,local*3u+2u);
}
)GLSL";
}
