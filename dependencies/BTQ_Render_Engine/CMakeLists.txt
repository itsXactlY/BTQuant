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

# Add VMA (Vulkan Memory Allocator)
FetchContent_Declare(
  VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.0.1
)
FetchContent_Populate(VulkanMemoryAllocator)

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

# Source files for main dashboard
set(DASHBOARD_SOURCES
    src/main_advanced.cpp
    src/components/data_grid_component.cpp
    src/components/realtime_chart_component.cpp
    src/components/heatmap_component.cpp
    src/components/orderbook_component.cpp
    src/components/log_display_component.cpp
    src/layout/dashboard_layout_manager.cpp
    src/data/ui_data_manager.cpp
    src/data/hotspine_data_bridge.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/visualization/data_visualization_engine.cpp
    src/monitoring/performance_monitor.cpp
    src/config/dashboard_config.cpp
    src/interaction/interaction_manager.cpp
    src/ui/advanced_ui_features.cpp
    src/optimization/performance_optimizer.cpp
    src/analytics/trading_analytics.cpp
    src/trading/trading_interface.cpp
    src/system/system_optimizer.cpp
)

# Source files for integration test
set(TEST_SOURCES
    src/test_integration.cpp
    src/data/hotspine_data_bridge.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/monitoring/performance_monitor.cpp
    src/config/dashboard_config.cpp
)

# Source files for interactive features test
set(INTERACTIVE_TEST_SOURCES
    src/test_interactive_features.cpp
    src/interaction/interaction_manager.cpp
    src/ui/advanced_ui_features.cpp
    src/optimization/performance_optimizer.cpp
    src/analytics/trading_analytics.cpp
    src/trading/trading_interface.cpp
    src/system/system_optimizer.cpp
)

# External dependencies
set(EXTERNAL_SOURCES
    ../../tests/new/src/hotspine_reader.cpp
    ../../tests/new/src/symbol_registry.cpp
    ../../tests/new/src/utils/dynamic_logger.cpp
)

# Advanced dashboard executable
add_executable(dashboard_advanced ${DASHBOARD_SOURCES} ${EXTERNAL_SOURCES})

# Integration test executable
add_executable(dashboard_test ${TEST_SOURCES} ${EXTERNAL_SOURCES})

# Interactive features test executable
add_executable(test_interactive_features ${INTERACTIVE_TEST_SOURCES} ${EXTERNAL_SOURCES})

target_include_directories(dashboard_advanced PRIVATE
   ${Vulkan_INCLUDE_DIRS}
   ${X11_INCLUDE_DIR}
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
   ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
   ${glm_SOURCE_DIR}
   ${concurrentqueue_SOURCE_DIR}
   ${VulkanMemoryAllocator_SOURCE_DIR}/include
   ${imgui_SOURCE_DIR}
   ${imgui_SOURCE_DIR}/backends
)

target_include_directories(dashboard_test PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
)

target_link_libraries(dashboard_advanced PRIVATE
   Vulkan::Vulkan
   ${X11_LIBRARIES}
   imgui
   pthread
   rt
)

target_link_libraries(dashboard_test PRIVATE
   pthread
   rt
)

target_include_directories(test_interactive_features PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
   ${glm_SOURCE_DIR}
   ${concurrentqueue_SOURCE_DIR}
)

target_link_libraries(test_interactive_features PRIVATE
   pthread
   rt
   ${X11_LIBRARIES}
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