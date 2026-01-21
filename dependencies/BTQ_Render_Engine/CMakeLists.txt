cmake_minimum_required(VERSION 3.15)
project(BTQuant_Render_Engine LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# External Headers
include(FetchContent)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.2)
FetchContent_Declare(concurrentqueue GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git GIT_TAG v1.0.4)
FetchContent_MakeAvailable(glm json concurrentqueue)

# Path Configuration
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui)
set(IMPLOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui)

# Explicit Source Listing (Senior HFT Requirement)
set(LIB_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot_demo.cpp
)

set(DASHBOARD_SOURCES
    src/main_realtime_dashboard.cpp
    src/vulkan_dashboard_advanced.cpp
    src/system/VulkanCore.cpp
    src/data/hotspine_data_bridge.cpp
    src/components/quant_workspace_component.cpp
    src/components/panel_base.cpp
    src/components/chart_panel.cpp
    src/components/metrics_panel.cpp
    src/components/panel_manager.cpp
    src/components/chart_manager.cpp
    src/components/indicator_renderer.cpp
    # TODO: Update these for newer ImPlot API
    # src/components/realtime_dashboard_component.cpp
    # src/components/heatmap_panel.cpp
    # src/components/correlation_heatmap_component.cpp
    src/trading/trading_interface.cpp
    src/data/symbol_registry.cpp
    src/optimization/performance_optimizer.cpp
    src/system/GPUMemoryManager.cpp
    src/config/dashboard_config.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/monitoring/performance_monitor.cpp
)

add_executable(dashboard_advanced ${DASHBOARD_SOURCES} ${LIB_SOURCES})

target_include_directories(dashboard_advanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${glm_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
    ${concurrentqueue_SOURCE_DIR}
)

target_link_libraries(dashboard_advanced PRIVATE
    Vulkan::Vulkan
    glfw
    dl
    pthread
)

# Optimization
target_compile_definitions(dashboard_advanced PRIVATE IMGUI_ENABLE_DOCKING)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dashboard_advanced PRIVATE
            -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops
            -fstrict-aliasing)
        target_link_options(dashboard_advanced PRIVATE -flto)
    endif()

# Performance monitoring
target_compile_definitions(dashboard_advanced PRIVATE ENABLE_PERFORMANCE_MONITORING)
