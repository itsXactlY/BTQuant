cmake_minimum_required(VERSION 3.16)
project(BTQuant_Advanced_Dashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

find_package(Vulkan REQUIRED)
find_package(X11 REQUIRED)

# Add GLM
include(FetchContent)
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.8
)
FetchContent_Populate(glm)

# Add concurrentqueue
FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
  GIT_TAG v1.0.4
)
FetchContent_Populate(concurrentqueue)

# Add Dear ImGui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.89.9-docking
)
FetchContent_MakeAvailable(imgui)

# Build ImGui library
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp  # For compatibility, though we use X11
)

target_include_directories(imgui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(imgui PUBLIC Vulkan::Vulkan)

# Advanced dashboard executable
add_executable(dashboard_advanced
  src/main_advanced.cpp
)

target_include_directories(dashboard_advanced PRIVATE
   ${Vulkan_INCLUDE_DIRS}
   ${X11_INCLUDE_DIR}
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${glm_SOURCE_DIR}
   ${concurrentqueue_SOURCE_DIR}
   ${imgui_SOURCE_DIR}
   ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(dashboard_advanced PRIVATE
   Vulkan::Vulkan
   ${X11_LIBRARIES}
   imgui
   pthread
   rt
)

# Compiler optimizations for performance (WITHOUT -fno-exceptions, -fno-rtti)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dashboard_advanced PRIVATE
    -O3
    -march=native
    -ffast-math
  )
endif()

set_target_properties(dashboard_advanced PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "BTQuant Advanced Dashboard configured:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Output: ${CMAKE_BINARY_DIR}/bin/dashboard_advanced")