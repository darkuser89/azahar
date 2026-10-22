# Copyright 2026 Azahar Emulator Project
# Licensed under GPLv2 or any later version

include(FetchContent)

function(azahar_metal4_shader_dependencies)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    if(NOT TARGET glslang)
        # The standalone backend uses the repository's GLSL compiler as a CPU translation tool.
        set(BUILD_EXTERNAL OFF)
        set(ENABLE_OPT OFF)
        set(ENABLE_GLSLANG_BINARIES OFF)
        set(ENABLE_SPVREMAPPER OFF)
        set(ENABLE_HLSL OFF)
        set(ENABLE_PCH OFF)
        set(GLSLANG_TESTS OFF)
        set(GLSLANG_ENABLE_INSTALL OFF)
        set(ENABLE_SPIRV ON)
        add_subdirectory("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../externals/glslang"
                         "${CMAKE_CURRENT_BINARY_DIR}/glslang" EXCLUDE_FROM_ALL)
    endif()

    set(SPIRV_CROSS_CLI OFF)
    set(SPIRV_CROSS_ENABLE_TESTS OFF)
    set(SPIRV_CROSS_ENABLE_HLSL OFF)
    set(SPIRV_CROSS_ENABLE_CPP OFF)
    set(SPIRV_CROSS_ENABLE_REFLECT OFF)
    set(SPIRV_CROSS_ENABLE_C_API OFF)
    set(SPIRV_CROSS_ENABLE_UTIL OFF)
    set(SPIRV_CROSS_ENABLE_GLSL ON)
    set(SPIRV_CROSS_ENABLE_MSL ON)
    set(SPIRV_CROSS_SKIP_INSTALL ON)
    FetchContent_Declare(azahar_spirv_cross
        URL https://codeload.github.com/KhronosGroup/SPIRV-Cross/tar.gz/aa217aeb6c9f0ace7a0ab233b28807edf45eb165
        URL_HASH SHA256=b99db6ee6182dbeb102257e755bf6d4978c23dcfd3329033304779279bc27212
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(azahar_spirv_cross)
endfunction()

azahar_metal4_shader_dependencies()

function(azahar_metal4_pica_dependencies)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    if(NOT TARGET fmt::fmt)
        set(FMT_TEST OFF)
        set(FMT_DOC OFF)
        set(FMT_INSTALL OFF)
        add_subdirectory("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../externals/fmt"
                         "${CMAKE_CURRENT_BINARY_DIR}/fmt" EXCLUDE_FROM_ALL)
    endif()
endfunction()

azahar_metal4_pica_dependencies()
