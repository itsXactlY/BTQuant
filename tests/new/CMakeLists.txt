cmake_minimum_required(VERSION 3.10)
project(BTQuantUITests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenGL
find_package(OpenGL REQUIRED)

if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()

# GLFW
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GLFW REQUIRED glfw3>=3.3)
else()
    find_library(GLFW_LIBRARIES glfw3 REQUIRED)
    find_path(GLFW_INCLUDE_DIRS GL/glfw3.h REQUIRED)
endif()

# GLM - try multiple approaches
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        /sw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps
        ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/glm
        ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/glm
    PATH_SUFFIXES
        glm
)
if(NOT GLM_INCLUDE_DIR)
    # If GLM is not found, we'll define it as a subdirectory or download it
    message(WARNING "GLM not found, using fallback approach")
    # For now, just continue without GLM check since it might be in the dependencies
    set(GLM_INCLUDE_DIR "")
endif()

# imgui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/imgui-src)
set(IMGUI_BACKENDS_DIR ${CMAKE_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/imgui-src/backends)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ui
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/ccapi/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/imgui-src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/imgui-src/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/_deps/implot-src
    ${OPENGL_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
)

# Conditionally add GLM if found
if(GLM_INCLUDE_DIR)
    include_directories(${GLM_INCLUDE_DIR})
endif()

# Source files
set(SOURCES
    src/ui_main.cpp
    src/ui/haptic_feedback.cpp
    src/ui/tooltips.cpp
    src/ui/tutorial.cpp
    src/ui_components_test.cpp
    # Include other source files as needed
)

# Define imgui sources
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
)

# imgui opengl3 and glfw bindings
set(IMGUI_GLFW_BINDING
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Add executable
add_executable(ui_main
    src/ui_main.cpp
    src/ui/haptic_feedback.cpp
    src/ui/tooltips.cpp
    src/ui/tutorial.cpp
    ${IMGUI_SOURCES}
    ${IMGUI_GLFW_BINDING}
)

add_executable(haptic_feedback_test
    src/haptic_feedback_test.cpp
    src/ui/haptic_feedback.cpp
)

# Link libraries
target_link_libraries(ui_main
    ${CMAKE_THREAD_LIBS_INIT}
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARIES}
)


# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ui_main PRIVATE -Wall -Wextra)
    target_compile_options(haptic_feedback_test PRIVATE -Wall -Wextra)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(ui_main PRIVATE /W4)
    target_compile_options(haptic_feedback_test PRIVATE /W4)
endif()