# Compute shader build target.
#
# glslc compiles GLSL → SPIR-V (Vulkan's binary shader format). The SPIR-V
# binaries are loaded at runtime by src/renderer/embedded_shaders.cpp.
#
# This file is invoked by the main CMakeLists.txt as add_subdirectory(shaders).

cmake_minimum_required(VERSION 3.28)
project(btquant_vulkan_shaders LANGUAGES NONE)

# glslc is the official Vulkan SDK shader compiler (Khronos).
# Find it as part of the Vulkan SDK.
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS ${Vulkan_GLSLANG_BIN_DIR} ${Vulkan_BINARY_DIR}
    DOC "glslc — Khronos Vulkan SDK shader compiler"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR
        "glslc not found — install the Vulkan SDK "
        "(provides glslc via glslang component) or set CMAKE_PROGRAM_PATH.")
endif()

# Output directory (relative to build/)
set(SHADER_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv")
file(MAKE_DIRECTORY "${SHADER_SPV_DIR}")

# Helper: compile <src>.comp → <out>.comp.spv
function(SHADER_COMPILE SRC_FILE)
    get_filename_component(SRC_NAME "${SRC_FILE}" NAME_WE)
    set(SPV_OUT "${SHADER_SPV_DIR}/${SRC_NAME}.comp.spv")
    add_custom_command(
        OUTPUT "${SPV_OUT}"
        COMMAND "${GLSLC_EXECUTABLE}"
                -fshader-stage=comp
                "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
                -o "${SPV_OUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}"
        COMMENT "Compiling shader ${SRC_FILE} → spirv/"
        VERBATIM
    )
    # Convenience target for "build all shaders"
    get_filename_component(SRC_BASE "${SRC_FILE}" NAME)
    set(SHADER_TARGET "shader_${SRC_NAME}")
    add_custom_target(${SHADER_TARGET} DEPENDS "${SPV_OUT}")
    list(APPEND BTQUANT_SHADER_SPV "${SPV_OUT}")
    set(BTQUANT_SHADER_SPV "${BTQUANT_SHADER_SPV}" PARENT_SCOPE)
endfunction()

# Shader registry — add new SHADER_COMPILE(<name>.comp) lines here.
SHADER_COMPILE(heatmap.comp)

# Top-level "build all shaders" target.
add_custom_target(btquant_vulkan_shaders
    DEPENDS ${BTQUANT_SHADER_SPV}
)
