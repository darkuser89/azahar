// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once
#include <string_view>
namespace Metal {
inline constexpr std::string_view InputReferenceShaderGLSL=R"GLSL(layout(buffer_reference,buffer_reference_align=1,std430) readonly buffer MeshBytes {uint8_t bytes[];};
struct InputReference {MeshBytes data;uint stride;uint format;uint components;uint kind;uint reserved0;uint reserved1;vec4 fixed_value;};
layout(set=5,binding=4,std430) readonly buffer MeshInputReferences {InputReference input_refs[16];};
vec4 mesh_load_register(uint reg,uint vertex) {
 if(input_refs[reg].kind==0u)return input_refs[reg].fixed_value;
 MeshBytes data=input_refs[reg].data;
 uint stride=input_refs[reg].stride;
 uint format=input_refs[reg].format;
 uint components=input_refs[reg].components;
 vec4 value=vec4(0,0,0,1);
 uint bytes=format==3u ? 4u:format==2u ? 2u:1u;
 for(uint c=0u;c<components;++c){
  uint raw=0u;
  for(uint b=0u;b<bytes;++b)raw|=uint(data.bytes[vertex*stride+c*bytes+b])<<(b*8u);
  value[c]=format==3u ? uintBitsToFloat(raw):format==2u ? float(int((raw^32768u)-32768u)):
           format==0u ? float(int((raw^128u)-128u)):float(raw);
 }
 return value;
}
)GLSL";
}
