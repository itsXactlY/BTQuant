# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(btquant_tests)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Inherit the parent's imgui/implot include paths so #include <imgui.h> works in our test source.
set(_parent_imgui_src "${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/imgui-src")
set(_parent_implot_src "${CMAKE_CURRENT_SOURCE_DIR}/../build/_deps/implot-src")
if(EXISTS "${_parent_imgui_src}/imgui.h")
    include_directories(${_parent_imgui_src} ${_parent_imgui_src}/backends)
endif()
if(EXISTS "${_parent_implot_src}/implot.h")
    include_directories(${_parent_implot_src})
endif()

# GLFW: always resolve to actual lib+include paths so vulkan_context.cpp links.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 glfw3)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${Vulkan_INCLUDE_DIRS})
if(glfw3_FOUND AND NOT GLFW3_INCLUDE_DIRS)
    find_path(glfw3_INCLUDE_DIR GLFW/glfw3.h)
    if(glfw3_INCLUDE_DIR)
        include_directories(${glfw3_INCLUDE_DIR})
    endif()
endif()
if(GLFW3_INCLUDE_DIRS)
    include_directories(${GLFW3_INCLUDE_DIRS})
endif()

# Add the test executable
add_executable(test_integration test_integration.cpp)

# Same compile flags as main target — BTQUANT_USE_GLFW must be defined for vulkan_context.cpp
target_compile_definitions(test_integration PRIVATE BTQUANT_USE_GLFW)

# Inherit imgui sources for stats_overlay etc that reference ImGui symbols.
target_sources(test_integration PRIVATE
    ${_parent_imgui_src}/imgui.cpp
    ${_parent_imgui_src}/imgui_draw.cpp
    ${_parent_imgui_src}/imgui_tables.cpp
    ${_parent_imgui_src}/imgui_widgets.cpp
)

# Link libraries
target_link_libraries(test_integration
    Vulkan::Vulkan Threads::Threads
)
if(GLFW3_LIBRARIES)
    target_link_libraries(test_integration ${GLFW3_LIBRARIES})
endif()

# Link with main project sources (for testing purposes)
target_sources(test_integration PRIVATE
    ../src/data/data_spine.cpp
    ../src/data/market_data.cpp
    ../src/data/ring_buffer.cpp
    ../src/core/vulkan_utils.cpp
    ../src/core/vulkan_context.cpp
    ../src/util/settings.cpp
    ../src/ui/stats_overlay.cpp
)

# Set C++ standard
set_target_properties(test_integration PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

add_test(NAME test_integration COMMAND test_integration)