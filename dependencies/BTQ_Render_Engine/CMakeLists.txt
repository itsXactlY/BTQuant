cmake_minimum_required(VERSION 3.15)
project(BTQuant_Render_Engine LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Build options
option(BUILD_TESTS "Build test executables" OFF)
option(ENABLE_PGO "Enable Profile-Guided Optimization" OFF)
option(ENABLE_LTO "Enable Link-Time Optimization" ON)
option(ENABLE_NATIVE_ARCH "Enable -march=native for maximum performance" ON)

# C++26 Standard - Cutting Edge (GCC 15+ / Clang 18+) for main application
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Checking standard - done")

# =============================================================================
# AGGRESSIVE OPTIMIZATION FLAGS FOR ULTRA-LOW-LATENCY TRADING
# =============================================================================

# Base compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-deprecated-declarations")

# Release mode optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Aggressive optimization level
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    
    # Enable native CPU architecture for maximum SIMD performance
    if(ENABLE_NATIVE_ARCH)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
        message(STATUS "Enabled native CPU architecture optimization (-march=native)")
    endif()
    
    # Link-Time Optimization for whole-program optimization
    if(ENABLE_LTO)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=auto")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto=auto")
        message(STATUS "Enabled Link-Time Optimization (-flto)")
    endif()
    
    # Fast math for indicator calculations (use with caution)
    # Note: This breaks IEEE 754 compliance but significantly speeds up floating-point ops
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -fno-math-errno")
    
    # Enable auto-vectorization and SIMD optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize -fno-trapping-math")
    
    # Function inlining optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions -finline-functions-called-once")
    
    # Unroll loops for better performance
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
    
    # Profile-Guided Optimization support
    if(ENABLE_PGO)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate")
        message(STATUS "Enabled Profile-Guided Optimization (generation phase)")
    endif()
    
    message(STATUS "Release CXX Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Debug mode - enable sanitizers for development
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")
    # Optional: Enable address sanitizer for memory debugging
    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
endif()

# Find Dependencies
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Find GLSL compiler
find_program(GLSLC glslc)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc shader compiler not found")
endif()

# External Headers
include(FetchContent)

# =============================================================================
# SIMDJSON - SIMD-accelerated JSON parsing for ultra-low-latency market data
# Parses at up to 3.5 GB/s on modern CPUs - critical for high-frequency data
# =============================================================================
FetchContent_Declare(simdjson 
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git 
    GIT_TAG v3.9.4
)

FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.2)
FetchContent_Declare(concurrentqueue GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git GIT_TAG v1.0.4)
FetchContent_Declare(readerwriterqueue GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git GIT_TAG v1.0.6)

# ImGui Docking Branch
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG docking)

# ImPlot
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master)

# Make all dependencies available
message(STATUS "Fetching simdjson...")
FetchContent_MakeAvailable(simdjson)
message(STATUS "Fetching glm...")
FetchContent_MakeAvailable(glm)
message(STATUS "Fetching json...")
FetchContent_MakeAvailable(json)
message(STATUS "Fetching concurrentqueue...")
FetchContent_MakeAvailable(concurrentqueue)
message(STATUS "Fetching readerwriterqueue...")
FetchContent_MakeAvailable(readerwriterqueue)
message(STATUS "Fetching imgui...")
FetchContent_MakeAvailable(imgui)
message(STATUS "Fetching implot...")
FetchContent_MakeAvailable(implot)
message(STATUS "Fetching dependencies - done")

message(STATUS "simdjson include dir: ${simdjson_SOURCE_DIR}/include")

# Path Configuration
set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMPLOT_DIR ${implot_SOURCE_DIR})
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/spirv)

# Shader compilation
file(GLOB SHADERS "${SHADER_DIR}/*.comp"
                  "${SHADER_DIR}/*.vert"
                  "${SHADER_DIR}/*.frag")

set(COMPILED_SHADERS "")
foreach(SHADER ${SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME_WE)
    get_filename_component(FILE_EXT ${SHADER} LAST_EXT)
    
    # Include extension in output name to avoid conflicts (e.g., candlestick.vert.spv vs candlestick.frag.spv)
    string(REPLACE "." "" EXT_CLEAN "${FILE_EXT}")
    set(SPIRV_FILE "${SPIRV_DIR}/${FILE_NAME}.${EXT_CLEAN}.spv")
    
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_DIR}"
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_FILE}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER} to ${SPIRV_FILE}"
    )
    
    list(APPEND COMPILED_SHADERS ${SPIRV_FILE})
endforeach()

# Explicit Source Listing (Fetching from valid source dirs now)
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
    # Threading module
    src/threading/task_scheduler.cpp
    src/threading/lockfree_queue.cpp
    src/performance/frame_time_graph.cpp
    src/performance/regression_detector.cpp
    src/performance/cpu_profiler.cpp
    src/performance/memory_tracker.cpp
    src/performance/global_sync.cpp
    src/ui/loading_state_integration.cpp
    src/ui/tooltips.cpp
    src/ui/context_menus.cpp
    src/ui/quick_actions.cpp
    src/ui/font_manager.cpp
    src/logging/logger.cpp
    src/error_handling/error_utils.cpp
    src/error_handling/crash_reporter.cpp
)

set(DASHBOARD_SOURCES
    src/main_trading_terminal.cpp
    src/vulkan_dashboard_advanced.cpp
    src/system/VulkanCore.cpp
    src/data/hotspine_data_bridge.cpp
    src/data/hotspine_bridge.cpp
    src/components/quant_workspace_component.cpp
    src/components/panel_base.cpp
    src/components/chart_panel.cpp
    src/components/panel_manager.cpp
    src/components/chart_manager.cpp
    src/components/theme_manager.cpp
    src/components/interaction_manager.cpp
    src/rendering/panel_culler.cpp
    src/rendering/imgui_optimizer.cpp
    src/rendering/frame_pacer.cpp
    src/rendering/auto_quality.cpp
    src/rendering/orderbook_batcher.cpp
    src/rendering/vulkan_chart_pipeline.cpp
    src/components/hierarchical_selector.cpp
    src/components/tape_panel.cpp
    src/components/volume_profile_panel.cpp
    src/components/footprint_panel.cpp
    src/components/tpo_panel.cpp
    src/components/dom_surface_panel.cpp
    src/components/VulkanSynchronization.cpp
    src/components/histogram_panel.cpp
    src/components/time_histogram_panel.cpp
    src/components/performance_monitor_panel.cpp
    src/components/time_and_sales.cpp
    src/data/symbol_registry.cpp
    src/system/GPUMemoryManager.cpp
    src/config/dashboard_config.cpp
    src/ui/unified_theme_system.cpp
    src/ui/layout_manager.cpp
    src/ui/workspace_manager.cpp
    src/ui/settings_manager.cpp
    src/ui/appearance_settings.cpp
    src/ui/data_settings.cpp
    src/ui/alert_settings.cpp
    src/ui/keyboard_shortcuts.cpp
    src/ui/loading_state_integration.cpp
    src/ui/quick_actions.cpp
    src/system/system_optimizer.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/data/cache_manager.cpp
    src/data/persistence.cpp
    src/data/incremental_updater.cpp
    src/data/quality_monitor.cpp
    src/monitoring/performance_monitor.cpp
    src/performance/debug_overlay.cpp
    src/analytics/technical_analysis.cpp
    src/analytics/trading_analytics.cpp
    src/analytics/volume_calculator.cpp
    src/indicators/session_vwap.cpp
    src/indicators/rolling_vwap.cpp
    src/indicators/anchored_vwap.cpp
    src/indicators/vwap_alerts.cpp
    src/indicators/indicator_alerts.cpp
    src/data/ui_data_manager.cpp
    src/data/orderbook_snapshot_manager.cpp
    src/performance/panel_profiler.cpp
    src/ui/performance_settings.cpp
    src/vulkan/lob_heatmap_compute_pipeline.cpp
    src/vulkan/ssbo_snapshot_updater.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
)



# Main BTQuantTerminal executable combining both dashboard functionalities
set(BTQUANT_TERMINAL_SOURCES
    # Library sources
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    # Threading module
    src/threading/task_scheduler.cpp
    src/threading/lockfree_queue.cpp
    src/performance/frame_time_graph.cpp
    src/performance/regression_detector.cpp
    src/performance/cpu_profiler.cpp
    src/performance/memory_tracker.cpp
    src/performance/global_sync.cpp
    src/ui/loading_state_integration.cpp
    src/ui/tooltips.cpp
    src/ui/context_menus.cpp
    src/ui/quick_actions.cpp
    src/ui/font_manager.cpp
    src/logging/logger.cpp
    src/error_handling/error_utils.cpp
    src/error_handling/crash_reporter.cpp
    # Main application sources
    src/main_trading_terminal.cpp
    src/vulkan_dashboard_advanced.cpp
    src/system/VulkanCore.cpp
    src/data/hotspine_data_bridge.cpp
    src/data/hotspine_bridge.cpp
    src/components/quant_workspace_component.cpp
    src/components/panel_base.cpp
    src/components/chart_panel.cpp
    src/components/panel_manager.cpp
    src/components/chart_manager.cpp
    src/components/theme_manager.cpp
    src/components/interaction_manager.cpp
    src/rendering/panel_culler.cpp
    src/rendering/imgui_optimizer.cpp
    src/rendering/frame_pacer.cpp
    src/rendering/auto_quality.cpp
    src/rendering/orderbook_batcher.cpp
    src/rendering/vulkan_chart_pipeline.cpp
    src/components/hierarchical_selector.cpp
    src/components/tape_panel.cpp
    src/components/volume_profile_panel.cpp
    src/components/footprint_panel.cpp
    src/components/tpo_panel.cpp
    src/components/dom_surface_panel.cpp
    src/components/VulkanSynchronization.cpp
    src/components/histogram_panel.cpp
    src/components/time_histogram_panel.cpp
    src/components/performance_monitor_panel.cpp
    src/components/time_and_sales.cpp
    src/data/symbol_registry.cpp
    src/system/GPUMemoryManager.cpp
    src/config/dashboard_config.cpp
    src/ui/unified_theme_system.cpp
    src/ui/layout_manager.cpp
    src/ui/workspace_manager.cpp
    src/ui/settings_manager.cpp
    src/ui/appearance_settings.cpp
    src/ui/data_settings.cpp
    src/ui/alert_settings.cpp
    src/ui/keyboard_shortcuts.cpp
    src/ui/loading_state_integration.cpp
    src/ui/quick_actions.cpp
    src/system/system_optimizer.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/data/cache_manager.cpp
    src/data/persistence.cpp
    src/data/incremental_updater.cpp
    src/data/quality_monitor.cpp
    src/monitoring/performance_monitor.cpp
    src/performance/debug_overlay.cpp
    src/analytics/technical_analysis.cpp
    src/analytics/trading_analytics.cpp
    src/analytics/volume_calculator.cpp
    src/indicators/session_vwap.cpp
    src/indicators/rolling_vwap.cpp
    src/indicators/anchored_vwap.cpp
    src/indicators/vwap_alerts.cpp
    src/indicators/indicator_alerts.cpp
    src/data/ui_data_manager.cpp
    src/data/orderbook_snapshot_manager.cpp
    src/performance/panel_profiler.cpp
    src/ui/performance_settings.cpp
    src/vulkan/lob_heatmap_compute_pipeline.cpp
    src/vulkan/ssbo_snapshot_updater.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
)

add_executable(BTQuantTerminal ${BTQUANT_TERMINAL_SOURCES} ${COMPILED_SHADERS})

target_include_directories(BTQuantTerminal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/threading
    ${CMAKE_CURRENT_SOURCE_DIR}/include/performance
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/imgui-src
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/imgui-src/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/implot-src
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${glm_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/glm-src
    ${json_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/json-src/include
    ${concurrentqueue_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/concurrentqueue-src
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/concurrentqueue-src/c_api
    ${readerwriterqueue_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/readerwriterqueue-src
    # simdjson for ultra-fast JSON parsing
    ${simdjson_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/_deps/simdjson-src/include
)

message(STATUS "Finding TBB...")
find_package(TBB REQUIRED)
message(STATUS "Finding TBB - done")

target_link_libraries(BTQuantTerminal PRIVATE
    Vulkan::Vulkan
    glfw
    dl
    pthread
    TBB::tbb
    # simdjson for SIMD-accelerated JSON parsing
    simdjson
)

# Conditionally build tests if BUILD_TESTS is enabled
if(BUILD_TESTS)
    message(STATUS "Building with tests enabled")
    
    # Find test framework if needed
    enable_testing()
    
    # Example test executable - in a real scenario, you'd define actual test executables here
    # For now, we'll just ensure that test files are not accidentally included in the main build
else()
    message(STATUS "Building with tests disabled - test files will be excluded from build tree")
endif()
