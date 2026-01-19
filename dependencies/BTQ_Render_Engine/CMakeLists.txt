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
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui-1.92.5)
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
    src/data/ui_data_manager.cpp
    src/data/symbol_manager.cpp
    src/data/symbol_registry.cpp
    src/components/data_grid_component.cpp
    src/components/heatmap_component.cpp
    src/components/log_display_component.cpp
    src/components/depth_chart_component.cpp
    src/components/trading_components.cpp
    src/components/alert_component.cpp
    src/components/watchlist_component.cpp
    src/components/market_screener_component.cpp
    src/components/technical_indicators_component.cpp
    src/components/keyboard_shortcuts_component.cpp
    src/components/theme_customization_component.cpp
    src/analytics/trading_analytics.cpp
    src/layout/dashboard_layout_manager.cpp
    src/monitoring/performance_monitor.cpp
    # src/interaction/interaction_manager.cpp
    src/optimization/performance_optimizer.cpp
    src/system/GPUMemoryManager.cpp
    src/config/dashboard_config.cpp
    src/data/market_data_processor.cpp
)

add_executable(dashboard_advanced ${DASHBOARD_SOURCES} ${LIB_SOURCES})

target_include_directories(dashboard_advanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/stubs
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${glm_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
    ${concurrentqueue_SOURCE_DIR}
)

target_link_libraries(dashboard_advanced PRIVATE
    Vulkan::Vulkan
    Threads::Threads
    glfw
    X11
    Xi
    dl
    pthread
)

# Optimization
target_compile_definitions(dashboard_advanced PRIVATE IMGUI_ENABLE_DOCKING)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dashboard_advanced PRIVATE -Wall -Wextra -O3 -march=native -ffast-math)
endif()
