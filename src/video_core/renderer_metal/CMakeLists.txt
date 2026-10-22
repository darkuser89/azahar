# Copyright 2026 Azahar Emulator Project
# Licensed under GPLv2 or any later version

cmake_minimum_required(VERSION 3.25)

# This backend can be compiled in isolation while the renderer is being ported.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(azahar_metal4 LANGUAGES CXX OBJCXX)
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "The native Metal 4 backend requires macOS on Apple Silicon")
endif()
if(CMAKE_OSX_ARCHITECTURES)
    if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        message(FATAL_ERROR "Metal 4 supports arm64 only; Intel and universal builds are unsupported")
    endif()
elseif(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "Select CMAKE_OSX_ARCHITECTURES=arm64 for the Metal 4 backend")
endif()

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
find_library(METAL4_FRAMEWORK Metal REQUIRED)
find_library(METAL4_FOUNDATION Foundation REQUIRED)
find_library(METAL4_QUARTZCORE QuartzCore REQUIRED)

include(mtl_dependencies.cmake)
add_library(azahar_metal4_shader_translation STATIC
    mtl_shader_translation.h mtl_shader_translation.cpp mtl_translation_cache.h
    mtl_fragment_operations.h mtl_fragment_operations.cpp
)
target_compile_features(azahar_metal4_shader_translation PUBLIC cxx_std_20)
target_include_directories(azahar_metal4_shader_translation PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(azahar_metal4_shader_translation PRIVATE -Wall -Wextra -Werror)
target_link_libraries(azahar_metal4_shader_translation PRIVATE
    glslang glslang-default-resource-limits spirv-cross-msl
)

add_library(azahar_metal4_pica_shader_translation STATIC
    mtl_pica_shader_provider.h mtl_pica_shader_provider.cpp
)
target_compile_features(azahar_metal4_pica_shader_translation PUBLIC cxx_std_20)
target_include_directories(azahar_metal4_pica_shader_translation SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/boost"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/nihstro/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/xxHash"
)
target_compile_options(azahar_metal4_pica_shader_translation PRIVATE -Wall -Wextra -Werror)
target_link_libraries(azahar_metal4_pica_shader_translation PUBLIC
    azahar_metal4_shader_translation fmt::fmt
)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Compile the real, unchanged generator for standalone verification. In the integrated
    # build these definitions come from video_core and its existing common dependency.
    target_sources(azahar_metal4_pica_shader_translation PRIVATE
        ../shader/generator/glsl_fs_shader_gen.cpp
        ../shader/generator/pica_fs_config.cpp
        ../shader/generator/glsl_shader_gen.cpp
        ../shader/generator/glsl_shader_decompiler.cpp
        ../shader/generator/shader_gen.cpp
        ../pica/shader_setup.cpp
        ../rasterizer_cache/surface_base.cpp
        ../rasterizer_cache/surface_params.cpp
        ../rasterizer_cache/pixel_format.cpp
        ../texture/texture_decode.cpp
        ../texture/etc1.cpp
    )
    target_compile_definitions(azahar_metal4_pica_shader_translation PRIVATE XXH_INLINE_ALL)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeModules/GenerateSettingKeys.cmake" _metal_setting_keys)
    string(REPLACE "\"common/setting_keys.h.in\""
        "\"${CMAKE_CURRENT_SOURCE_DIR}/../../common/setting_keys.h.in\"" _metal_setting_keys "${_metal_setting_keys}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generate_metal_setting_keys.cmake" "${_metal_setting_keys}")
    include("${CMAKE_CURRENT_BINARY_DIR}/generate_metal_setting_keys.cmake")
    target_include_directories(azahar_metal4_pica_shader_translation SYSTEM PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}")
    set_source_files_properties(../shader/generator/shader_gen.cpp PROPERTIES
        COMPILE_DEFINITIONS ENABLE_METAL)
    # The unchanged decompiler intentionally partially initializes subroutine aggregates.
    set_source_files_properties(../shader/generator/glsl_shader_decompiler.cpp PROPERTIES
        COMPILE_OPTIONS "-Wno-missing-field-initializers")
endif()

add_library(azahar_metal4 STATIC
    mtl_compilation_queue.h
    mtl_gas_runtime.h
    mtl_gas_fragment.h
    mtl_gas_accumulation.h
    mtl_gas_runtime.mm
    mtl_gas_runtime.metal
    mtl_device.h
    mtl_device.mm
    mtl_pipeline_disk_cache.h
    mtl_pipeline_disk_cache.mm
    mtl_scheduler.h
    mtl_scheduler.mm
    mtl_stream_buffer.h
    mtl_stream_buffer.mm
    mtl_present.h
    mtl_present.mm
    mtl_present.metal
    mtl_transfer.h
    mtl_transfer.mm
    mtl_geometry.h
    mtl_geometry.mm
    mtl_geometry.metal
    mtl_pipeline_cache.h
    mtl_pipeline_cache.mm
    mtl_pipeline_manifest.h
    mtl_pipeline_manifest.mm
    mtl_format_conversion.h
    mtl_format_conversion.mm
    mtl_format_conversion.metal
    mtl_shader_runtime.h
    mtl_shader_runtime.mm
    mtl_shader_disk_cache.h
    mtl_shader_disk_cache.mm
    mtl_translation_source_store.h
    mtl_translation_source_store.mm
    mtl_vertex_fetch.h
    mtl_input_reference.h
    mtl_input_reference_shader.h
    mtl_input_reference.mm
    mtl_vertex_fetch.mm
    mtl_vertex_fetch.metal
    mtl_texture_runtime.h
    mtl_texture_runtime.mm
    mtl_texture_filter.h
    mtl_texture_filter.mm
    mtl_rasterizer_cache.mm
    mtl_guest_codec.h
    mtl_guest_codec.mm
    mtl_guest_codec.metal
    mtl_shader_machine.h
    mtl_shader_machine.mm
    mtl_shader_machine.metal
    mtl_geometry_program.h
    mtl_geometry_program.mm
    mtl_geometry_program.metal
    mtl_state.h
    mtl_draw_runtime.h
    mtl_draw_runtime.mm
    mtl_rasterizer.h
    mtl_rasterizer.mm
    mtl_screen_layout.h
    mtl_screen_layout.mm
    renderer_metal.h
    renderer_metal.mm
)
set_target_properties(azahar_metal4 PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
)
target_compile_options(azahar_metal4 PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc> -Wall -Wextra -Werror -Wunguarded-availability
)
target_link_libraries(azahar_metal4 PUBLIC
    "${METAL4_FRAMEWORK}" "${METAL4_FOUNDATION}" "${METAL4_QUARTZCORE}"
    azahar_metal4_shader_translation
    azahar_metal4_pica_shader_translation
)
target_include_directories(azahar_metal4 SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/dynarmic/externals/robin-map/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/dds-ktx"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/microprofile")
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    target_compile_definitions(azahar_metal4 PRIVATE ENABLE_METAL)
    target_include_directories(azahar_metal4 SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/microprofile")
else()
    target_include_directories(azahar_metal4 SYSTEM PUBLIC "${PROJECT_BINARY_DIR}/src")
    target_link_libraries(azahar_metal4 PUBLIC citra_common)
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_present.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_geometry.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_format_conversion.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_vertex_fetch.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_guest_codec.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_machine.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_geometry_program.metal"
)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_present.metal" METAL4_PRESENT_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_geometry.metal" METAL4_GEOMETRY_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_format_conversion.metal" METAL4_FORMAT_CONVERSION_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_vertex_fetch.metal" METAL4_VERTEX_FETCH_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_guest_codec.metal" METAL4_GUEST_CODEC_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_machine.metal" METAL4_SHADER_MACHINE_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_geometry_program.metal" METAL4_GEOMETRY_PROGRAM_SOURCE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_gas_runtime.metal" METAL4_GAS_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mtl_gas_runtime.metal")
foreach(_filter IN ITEMS bicubic xbrz_freescale x_gradient y_gradient refine)
    set(_filter_path "${CMAKE_CURRENT_SOURCE_DIR}/../host_shaders/texture_filtering/${_filter}.frag")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_filter_path}")
    string(TOUPPER "${_filter}" _filter_upper)
    file(READ "${_filter_path}" METAL4_FILTER_${_filter_upper})
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mtl_filter_mmpx.glsl")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_filter_mmpx.glsl" METAL4_FILTER_MMPX)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mtl_filter_scale_force.glsl")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/mtl_filter_scale_force.glsl" METAL4_FILTER_SCALE_FORCE)
configure_file(mtl_shaders.h.in mtl_shaders.h @ONLY)
file(GLOB _metal_cache_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../shader/generator/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../shader/generator/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pica/regs_*.h")
list(APPEND _metal_cache_inputs
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_geometry_program.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_cache_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_translation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_text.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_translation_cache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_translation_source_store.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_translation_source_store.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_translation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_input_reference_shader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_input_reference.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_mesh_shader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_fragment_operations.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_fragment_operations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pica_shader_provider.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pica_shader_provider.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_gas_fragment.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_gas_accumulation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_disk_cache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_disk_cache.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_runtime.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_shader_runtime.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pipeline_cache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pipeline_cache.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pipeline_manifest.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_pipeline_manifest.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_compilation_queue.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_device.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/mtl_dependencies.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pica/shader_setup.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pica/shader_setup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/CMakeLists.txt")
get_target_property(_metal_cross_source_dir spirv-cross-msl SOURCE_DIR)
file(GLOB_RECURSE _metal_compiler_inputs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/glslang/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/glslang/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/glslang/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/SPIRV/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../externals/glslang/SPIRV/*.h")
file(GLOB _metal_cross_inputs CONFIGURE_DEPENDS
    "${_metal_cross_source_dir}/*.cpp" "${_metal_cross_source_dir}/*.hpp"
    "${_metal_cross_source_dir}/CMakeLists.txt")
list(APPEND _metal_cache_inputs ${_metal_compiler_inputs} ${_metal_cross_inputs})
list(REMOVE_DUPLICATES _metal_cache_inputs)
list(SORT _metal_cache_inputs)
set(_metal_cache_hashes "azahar-metal4-source-v1-arm64-msl4-safe-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
foreach(_input IN LISTS _metal_cache_inputs)
    if(NOT _input MATCHES "/CMakeLists\\.txt$")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_input}")
    endif()
    file(SHA256 "${_input}" _hash)
    string(APPEND _metal_cache_hashes "${_hash}")
endforeach()
string(SHA256 METAL4_SHADER_CACHE_SOURCE_VERSION "${_metal_cache_hashes}")
configure_file(mtl_shader_cache_version.h.in mtl_shader_cache_version.h @ONLY)
target_include_directories(azahar_metal4 PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
set_source_files_properties(mtl_present.metal mtl_geometry.metal mtl_format_conversion.metal mtl_vertex_fetch.metal mtl_guest_codec.metal mtl_shader_machine.metal mtl_geometry_program.metal mtl_gas_runtime.metal PROPERTIES HEADER_FILE_ONLY ON)

# Xcode sometimes exposes an installed toolchain only at its mounted path. An explicit compiler
# override keeps this build usable without altering Xcode's installation or source directories.
set(METAL4_SHADER_COMPILER "" CACHE FILEPATH "Optional path to the Xcode Metal shader compiler")
if(METAL4_SHADER_COMPILER)
    set(METAL4_SHADER_COMMAND "${METAL4_SHADER_COMPILER}")
else()
    find_program(METAL4_XCRUN xcrun REQUIRED)
    set(METAL4_SHADER_COMMAND "${METAL4_XCRUN}" -sdk macosx metal)
endif()
foreach(_shader IN ITEMS mtl_present mtl_geometry mtl_format_conversion mtl_vertex_fetch mtl_guest_codec mtl_shader_machine mtl_geometry_program mtl_gas_runtime)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_shader}.air"
        COMMAND ${METAL4_SHADER_COMMAND} -std=metal4.0 -mmacosx-version-min=26.0
            "-fmodules-cache-path=${CMAKE_CURRENT_BINARY_DIR}/metal_module_cache"
            -Wall -Wextra -Werror -c "${CMAKE_CURRENT_SOURCE_DIR}/${_shader}.metal"
            -o "${CMAKE_CURRENT_BINARY_DIR}/${_shader}.air"
        DEPENDS "${_shader}.metal"
        VERBATIM
    )
    list(APPEND METAL4_SHADER_OUTPUTS "${CMAKE_CURRENT_BINARY_DIR}/${_shader}.air")
endforeach()
add_custom_target(azahar_metal4_shaders DEPENDS ${METAL4_SHADER_OUTPUTS})
add_dependencies(azahar_metal4 azahar_metal4_shaders)

option(METAL4_BUILD_OFFLINE_TESTS "Build CPU-only Metal shader translation tests" OFF)
if(METAL4_BUILD_OFFLINE_TESTS)
    add_executable(azahar_metal4_shader_tests tests/mtl_shader_translation_tests.cpp)
    target_compile_features(azahar_metal4_shader_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_shader_tests PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_shader_tests PRIVATE azahar_metal4_shader_translation)
endif()

if(METAL4_BUILD_OFFLINE_TESTS)
    add_executable(azahar_metal4_pica_shader_tests tests/mtl_pica_shader_tests.cpp)
    target_compile_features(azahar_metal4_pica_shader_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_pica_shader_tests PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_pica_shader_tests PRIVATE azahar_metal4_pica_shader_translation)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_pica_shader_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    else()
        # Reuse the integrated, unchanged PICA generators in the stable app
        # build tree instead of requiring a second standalone build.
        find_library(METAL4_CPU_TEST_APPKIT AppKit REQUIRED)
        find_library(METAL4_CPU_TEST_COREGRAPHICS CoreGraphics REQUIRED)
        target_link_libraries(azahar_metal4_pica_shader_tests PRIVATE video_core
            "${METAL4_CPU_TEST_APPKIT}" "${METAL4_CPU_TEST_COREGRAPHICS}")
    endif()
endif()

option(METAL4_BUILD_GPU_TESTS "Build headless Metal 4 GPU tests without starting Azahar" OFF)
option(METAL4_BUILD_CORE_TESTS "Build data-only native Metal core/cache tests without starting Azahar" OFF)
if(METAL4_BUILD_CORE_TESTS)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        message(FATAL_ERROR "Metal core tests require the integrated core libraries")
    endif()
    add_executable(azahar_metal4_gas_transport_tests tests/mtl_gas_transport_tests.mm)
    set_target_properties(azahar_metal4_gas_transport_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_gas_transport_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_gas_transport_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_shadow_bindings_tests tests/mtl_shadow_bindings_tests.mm)
    set_target_properties(azahar_metal4_shadow_bindings_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_shadow_bindings_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_shadow_bindings_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_proctex_extended_tests tests/mtl_proctex_extended_tests.mm)
    set_target_properties(azahar_metal4_proctex_extended_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_proctex_extended_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_proctex_extended_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_multilight_core_tests tests/mtl_multilight_core_tests.mm)
    set_target_properties(azahar_metal4_multilight_core_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_multilight_core_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_multilight_core_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_fog_proctex_tests tests/mtl_fog_proctex_tests.mm)
    set_target_properties(azahar_metal4_fog_proctex_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_fog_proctex_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_fog_proctex_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_lighting_shadow_tests tests/mtl_lighting_shadow_tests.mm)
    set_target_properties(azahar_metal4_lighting_shadow_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_lighting_shadow_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_lighting_shadow_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_lut_core_tests tests/mtl_lut_core_tests.mm)
    set_target_properties(azahar_metal4_lut_core_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_lut_core_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_lut_core_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_core_tests tests/mtl_core_tests.mm)
    set_target_properties(azahar_metal4_core_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_core_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_core_tests PRIVATE azahar_metal4 video_core citra_core)
    add_executable(azahar_metal4_custom_cube_tests tests/mtl_custom_cube_tests.mm
        ../../common/apple_authorization.cpp ../../common/apple_utils.cpp ../../common/apple_utils.mm)
    set_target_properties(azahar_metal4_custom_cube_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_custom_cube_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_custom_cube_tests PRIVATE azahar_metal4 video_core citra_core)
endif()
if(METAL4_BUILD_GPU_TESTS)
    add_executable(azahar_metal4_gas_stages_tests tests/mtl_gas_stages_tests.mm)
    set_target_properties(azahar_metal4_gas_stages_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_gas_stages_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_gas_stages_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        find_library(METAL4_APPKIT AppKit REQUIRED)
        target_link_libraries(azahar_metal4_gas_stages_tests PRIVATE video_core "${METAL4_APPKIT}")
    endif()
    add_executable(azahar_metal4_scheduler_tests tests/mtl_scheduler_tests.mm tests/mtl_stage_timing_tests.mm)
    set_target_properties(azahar_metal4_scheduler_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_scheduler_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_scheduler_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_link_libraries(azahar_metal4_scheduler_tests PRIVATE video_core)
    endif()
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        # Real Material/CustomTexture objects come from video_core; decoded fixtures
        # bypass image IO and never initialize the frontend or emulator.
        add_executable(azahar_metal4_custom_texture_tests tests/mtl_custom_texture_tests.mm)
        set_target_properties(azahar_metal4_custom_texture_tests PROPERTIES OBJCXX_STANDARD 20)
        add_executable(azahar_metal4_lighting_lut_tests tests/mtl_lighting_lut_tests.mm)
        set_target_properties(azahar_metal4_lighting_lut_tests PROPERTIES OBJCXX_STANDARD 20)
        target_compile_options(azahar_metal4_lighting_lut_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
        target_link_libraries(azahar_metal4_lighting_lut_tests PRIVATE azahar_metal4 video_core)
        add_executable(azahar_metal4_specular_tests tests/mtl_specular_tests.mm)
        set_target_properties(azahar_metal4_specular_tests PROPERTIES OBJCXX_STANDARD 20)
        target_compile_options(azahar_metal4_specular_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
        target_link_libraries(azahar_metal4_specular_tests PRIVATE azahar_metal4 video_core)
        add_executable(azahar_metal4_multilight_tests tests/mtl_multilight_tests.mm)
        set_target_properties(azahar_metal4_multilight_tests PROPERTIES OBJCXX_STANDARD 20)
        target_compile_options(azahar_metal4_multilight_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
        target_link_libraries(azahar_metal4_multilight_tests PRIVATE azahar_metal4 video_core)
        target_compile_options(azahar_metal4_custom_texture_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
        target_link_libraries(azahar_metal4_custom_texture_tests PRIVATE azahar_metal4 video_core)
    endif()
    add_executable(azahar_metal4_special_sampler_tests tests/mtl_special_sampler_tests.mm)
    set_target_properties(azahar_metal4_special_sampler_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_special_sampler_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_special_sampler_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_special_sampler_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    else()
        target_link_libraries(azahar_metal4_special_sampler_tests PRIVATE video_core)
    endif()
    add_executable(azahar_metal4_pipeline_manifest_tests tests/mtl_pipeline_manifest_tests.mm)
    add_executable(azahar_metal4_translation_source_tests tests/mtl_translation_source_tests.mm)
    set_target_properties(azahar_metal4_translation_source_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_translation_source_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_translation_source_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        find_library(METAL4_SOURCE_APPKIT AppKit REQUIRED)
        target_link_libraries(azahar_metal4_translation_source_tests PRIVATE "${METAL4_SOURCE_APPKIT}")
    else()
        target_sources(azahar_metal4_translation_source_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    endif()
    set_target_properties(azahar_metal4_pipeline_manifest_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_pipeline_manifest_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_pipeline_manifest_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        find_library(METAL4_MANIFEST_APPKIT AppKit REQUIRED)
        target_link_libraries(azahar_metal4_pipeline_manifest_tests PRIVATE "${METAL4_MANIFEST_APPKIT}")
    endif()
    add_executable(azahar_metal4_shader_disk_cache_tests tests/mtl_shader_disk_cache_tests.mm)
    set_target_properties(azahar_metal4_shader_disk_cache_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_shader_disk_cache_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_shader_disk_cache_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_shader_disk_cache_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    endif()
    add_executable(azahar_metal4_pipeline_disk_cache_tests tests/mtl_pipeline_disk_cache_tests.mm)
    set_target_properties(azahar_metal4_pipeline_disk_cache_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_pipeline_disk_cache_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_pipeline_disk_cache_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_pipeline_disk_cache_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        # citra_common propagates Apple display helpers to integrated executables.
        find_library(METAL4_APPKIT AppKit REQUIRED)
        target_link_libraries(azahar_metal4_pipeline_disk_cache_tests PRIVATE "${METAL4_APPKIT}")
    endif()
    add_executable(azahar_metal4_geometry_program_tests tests/mtl_geometry_program_tests.mm)
    set_target_properties(azahar_metal4_geometry_program_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_geometry_program_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_geometry_program_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_geometry_program_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp ../pica/output_vertex.cpp)
    endif()
    add_executable(azahar_metal4_shader_machine_tests tests/mtl_shader_machine_tests.mm)
    set_target_properties(azahar_metal4_shader_machine_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_shader_machine_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_shader_machine_tests PRIVATE azahar_metal4)
    target_include_directories(azahar_metal4_shader_machine_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    add_executable(azahar_metal4_native_vertex_tests tests/mtl_native_vertex_tests.mm)
    add_executable(azahar_metal4_reference_mesh_tests tests/mtl_reference_mesh_tests.mm)
    set_target_properties(azahar_metal4_reference_mesh_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_reference_mesh_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_reference_mesh_tests PRIVATE azahar_metal4)
    set_target_properties(azahar_metal4_native_vertex_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_native_vertex_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_native_vertex_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_vertex_diagnostic_tests tests/mtl_vertex_diagnostic_tests.mm)
    set_target_properties(azahar_metal4_vertex_diagnostic_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_vertex_diagnostic_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_vertex_diagnostic_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_present_tests tests/mtl_present_tests.mm)
    set_target_properties(azahar_metal4_present_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_present_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_present_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_present_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_present_spatial_tests tests/mtl_present_spatial_tests.mm)
    set_target_properties(azahar_metal4_present_spatial_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_present_spatial_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_present_spatial_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_include_directories(azahar_metal4_present_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_present_tests PRIVATE ENABLE_METAL)
        target_include_directories(azahar_metal4_present_spatial_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_present_spatial_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_present_spatial_tests PRIVATE "${METAL4_APPKIT}")
    endif()
    add_executable(azahar_metal4_geometry_tests tests/mtl_geometry_tests.mm)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_geometry_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
        target_sources(azahar_metal4_geometry_tests PRIVATE ../pica/primitive_assembly.cpp)
    endif()
    target_compile_features(azahar_metal4_geometry_tests PRIVATE cxx_std_20)
    set_target_properties(azahar_metal4_geometry_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_geometry_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_geometry_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_vertex_fetch_tests tests/mtl_vertex_fetch_tests.mm)
    set_target_properties(azahar_metal4_vertex_fetch_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_vertex_fetch_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_vertex_fetch_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_vertex_fetch_tests PRIVATE azahar_metal4)
    target_include_directories(azahar_metal4_vertex_fetch_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    add_executable(azahar_metal4_format_conversion_tests tests/mtl_format_conversion_tests.mm)
    set_target_properties(azahar_metal4_format_conversion_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_format_conversion_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_format_conversion_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_format_conversion_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_texture_runtime_tests tests/mtl_texture_runtime_tests.mm)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_texture_runtime_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    endif()
    set_target_properties(azahar_metal4_texture_runtime_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_texture_runtime_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_texture_runtime_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_texture_runtime_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_texture_filter_tests tests/mtl_texture_filter_tests.mm)
    add_executable(azahar_metal4_filter_formats_tests tests/mtl_filter_formats_tests.mm)
    add_executable(azahar_metal4_mmpx_geometry_tests tests/mtl_mmpx_geometry_tests.mm)
    add_executable(azahar_metal4_edge_filter_tests tests/mtl_edge_filter_tests.mm)
    add_executable(azahar_metal4_gas_fragment_tests tests/mtl_gas_fragment_tests.mm)
    add_executable(azahar_metal4_gas_accumulation_tests tests/mtl_gas_accumulation_tests.mm)
    set_target_properties(azahar_metal4_gas_accumulation_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_gas_accumulation_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_gas_accumulation_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_gas_accumulation_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
        target_include_directories(azahar_metal4_gas_accumulation_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_gas_accumulation_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_gas_accumulation_tests PRIVATE video_core "${METAL4_APPKIT}")
    endif()
    set_target_properties(azahar_metal4_gas_fragment_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_gas_fragment_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_gas_fragment_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_gas_fragment_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
        target_include_directories(azahar_metal4_gas_fragment_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_gas_fragment_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_gas_fragment_tests PRIVATE video_core "${METAL4_APPKIT}")
    endif()
    set_target_properties(azahar_metal4_edge_filter_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_edge_filter_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_edge_filter_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_include_directories(azahar_metal4_edge_filter_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_edge_filter_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_edge_filter_tests PRIVATE "${METAL4_APPKIT}")
    endif()
    set_target_properties(azahar_metal4_mmpx_geometry_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_mmpx_geometry_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_mmpx_geometry_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_include_directories(azahar_metal4_mmpx_geometry_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_mmpx_geometry_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_mmpx_geometry_tests PRIVATE "${METAL4_APPKIT}")
    endif()
    set_target_properties(azahar_metal4_filter_formats_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_options(azahar_metal4_filter_formats_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_filter_formats_tests PRIVATE azahar_metal4)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_include_directories(azahar_metal4_filter_formats_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_filter_formats_tests PRIVATE ENABLE_METAL)
    else()
        target_link_libraries(azahar_metal4_filter_formats_tests PRIVATE "${METAL4_APPKIT}")
    endif()
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_texture_filter_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
        target_include_directories(azahar_metal4_texture_filter_tests SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_compile_definitions(azahar_metal4_texture_filter_tests PRIVATE ENABLE_METAL)
    endif()
    set_target_properties(azahar_metal4_texture_filter_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_texture_filter_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_texture_filter_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_texture_filter_tests PRIVATE azahar_metal4)
    add_executable(azahar_metal4_guest_codec_tests tests/mtl_guest_codec_tests.mm)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        target_sources(azahar_metal4_guest_codec_tests PRIVATE tests/mtl_pica_test_diagnostics.cpp)
    endif()
    set_target_properties(azahar_metal4_guest_codec_tests PROPERTIES OBJCXX_STANDARD 20)
    target_compile_features(azahar_metal4_guest_codec_tests PRIVATE cxx_std_20)
    target_compile_options(azahar_metal4_guest_codec_tests PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    target_link_libraries(azahar_metal4_guest_codec_tests PRIVATE azahar_metal4)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        # The integrated build obtains unchanged PICA generators, cache helpers and
        # the primitive-assembly test reference from video_core rather than duplicate sources.
        find_library(METAL4_COREGRAPHICS CoreGraphics REQUIRED)
        foreach(_metal_gpu_test IN ITEMS translation_source shader_disk_cache geometry_program shader_machine vertex_diagnostic native_vertex reference_mesh present geometry vertex_fetch format_conversion texture_runtime texture_filter guest_codec)
            # Core's Apple refresh-rate helper needs these frameworks even when
            # the headless test does not link the Qt frontend.
            target_link_libraries(azahar_metal4_${_metal_gpu_test}_tests PRIVATE
                video_core "${METAL4_APPKIT}" "${METAL4_COREGRAPHICS}")
        endforeach()
    endif()
endif()
