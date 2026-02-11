cmake_minimum_required(VERSION 3.15)
project(BTQuant_Render_Engine LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# C++26 Standard - Cutting Edge (GCC 15+ / Clang 18+) for main application
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Disable treating warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-deprecated-declarations -Wno-deprecated-literal-operator")

# Enable Address Sanitizer for memory stability testing
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
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
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.2)
FetchContent_Declare(concurrentqueue GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git GIT_TAG v1.0.4)

# ImGui Docking Branch
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG docking)

# ImPlot
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master)

FetchContent_MakeAvailable(glm json concurrentqueue imgui implot)

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
    
    set(SPIRV_FILE "${SPIRV_DIR}/${FILE_NAME}.spv")
    
    add_custom_command(
        OUTPUT ${SPIRV_FILE}
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
    # Threading module - this is the main focus of our task
    src/threading/task_scheduler.cpp
    src/threading/lockfree_queue.cpp
    src/performance/frame_time_graph.cpp
    src/performance/regression_detector.cpp
    src/performance/cpu_profiler.cpp
    src/performance/memory_tracker.cpp
    src/ui/loading_states.cpp
    src/ui/loading_state_integration.cpp
    src/ui/empty_states.cpp
    src/ui/tooltips.cpp
    src/ui/context_menus.cpp
    src/ui/quick_actions.cpp
    src/ui/tutorial.cpp
    src/data/unified_data_pipeline.cpp
    src/logging/logger.cpp
    src/error_handling/error_utils.cpp
    src/error_handling/crash_reporter.cpp
)



# Main BTQuantTerminal executable combining both dashboard functionalities
set(BTQUANT_TERMINAL_SOURCES
    # Library sources (moved from btq_engine_lib to simplify dependency graph)
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    # Threading module - this is the main focus of our task
    src/threading/task_scheduler.cpp
    src/threading/lockfree_queue.cpp
    src/performance/frame_time_graph.cpp
    src/performance/regression_detector.cpp
    src/performance/cpu_profiler.cpp
    src/performance/memory_tracker.cpp
    src/ui/loading_states.cpp
    src/ui/loading_state_integration.cpp
    src/ui/empty_states.cpp
    src/ui/tooltips.cpp
    src/ui/context_menus.cpp
    src/ui/quick_actions.cpp
    src/ui/tutorial.cpp
    src/data/unified_data_pipeline.cpp
    src/logging/logger.cpp
    src/error_handling/error_utils.cpp
    src/error_handling/crash_reporter.cpp
    # Main application sources
    src/main_trading_terminal.cpp
    src/vulkan_dashboard_advanced.cpp
    src/system/VulkanCore.cpp
    src/data/hotspine_data_bridge.cpp
    src/components/quant_workspace_component.cpp
    src/components/panel_base.cpp
    src/components/chart_panel.cpp
    src/components/metrics_panel.cpp
    src/components/panel_manager.cpp
    src/components/chart_manager.cpp
    src/rendering/panel_culler.cpp
    src/rendering/chart_culler.cpp
    src/rendering/footprint_lod.cpp
    src/rendering/imgui_optimizer.cpp
    src/rendering/frame_pacer.cpp
    src/rendering/auto_quality.cpp
    src/rendering/orderbook_batcher.cpp
    src/components/indicator_renderer.cpp
    src/components/hierarchical_selector.cpp
    src/components/orderbook_panel.cpp
    src/components/orderbook_history.cpp
    src/components/alerts_panel.cpp
    src/components/status_bar_panel.cpp
    src/components/watchlist_panel.cpp
    src/components/watchlist_alerts.cpp
    src/components/tape_panel.cpp
    src/components/volume_profile_panel.cpp
    src/components/depth_chart_panel.cpp
    src/components/footprint_panel.cpp
    src/components/tpo_panel.cpp
    src/components/dom_surface_panel.cpp
    src/components/risk_analyzer_panel.cpp
    src/components/MarketMicrostructureRenderer.cpp
    src/components/VulkanSynchronization.cpp
    src/components/correlation_heatmap_component.cpp
    src/components/symbol_selector.cpp
    src/components/trading_orders_panel.cpp
    src/components/trading_positions_panel.cpp
    src/components/risk_metrics_panel.cpp
    src/components/histogram_panel.cpp
    src/components/scatter_plot_panel.cpp
    src/components/time_series_panel.cpp
    src/components/time_statistics_panel.cpp
    src/components/time_histogram_panel.cpp
    src/components/screener_panel.cpp
    src/components/log_panel.cpp
    src/components/multi_vwap_panel.cpp
    src/components/performance_monitor_panel.cpp
    src/components/time_and_sales.cpp
    src/components/historical_time_sales.cpp
    src/trading/trading_interface.cpp
    src/trading/order_manager.cpp
    src/trading/position_manager.cpp
    src/trading/risk_assessment.cpp
    src/trading/advanced_trading_features.cpp
    src/data/symbol_registry.cpp
    src/optimization/performance_optimizer.cpp
    src/system/GPUMemoryManager.cpp
    src/components/theme_manager.cpp
    src/components/interaction_manager.cpp
    src/components/chart_replay.cpp
    src/components/chart_replay_panel.cpp
    src/components/drawing_tools.cpp
    src/components/dashboard_controls.cpp
    src/components/global_alert_manager.cpp
    src/components/strategy_builder.cpp
    src/components/option_analytics_panel.cpp
    src/components/tabbed_panel.cpp
    src/config/dashboard_config.cpp
    src/ui/unified_theme_system.cpp
    src/ui/layout_manager.cpp
    src/ui/workspace_manager.cpp
    src/ui/settings_manager.cpp
    src/ui/appearance_settings.cpp
    src/ui/data_settings.cpp
    src/ui/alert_settings.cpp
    src/ui/keyboard_shortcuts.cpp
    src/ui/haptic_feedback.cpp
    src/ui/loading_states.cpp
    src/ui/loading_state_integration.cpp
    src/ui/empty_states.cpp
    src/ui/quick_actions.cpp
    src/ui/tutorial.cpp
    src/layout/dashboard_layout_manager.cpp
    src/system/system_optimizer.cpp
    src/components/chart_panel_settings.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/data/cache_manager.cpp
    src/data/persistence.cpp
    src/data/incremental_updater.cpp
    src/data/exchange_aggregator.cpp
    src/data/quality_monitor.cpp
    src/components/atomic_orderbook_panel.cpp
    src/monitoring/performance_monitor.cpp
    src/performance/debug_overlay.cpp
    src/analytics/technical_analysis.cpp
    src/analytics/trading_analytics.cpp
    src/analytics/cluster_engine.cpp
    src/analytics/volume_calculator.cpp
    src/analytics/tpoengine.cpp
    src/indicators/session_vwap.cpp
    src/indicators/rolling_vwap.cpp
    src/indicators/anchored_vwap.cpp
    src/indicators/vwap_alerts.cpp
    src/indicators/indicator_alerts.cpp
    src/data/ui_data_manager.cpp
    src/performance/panel_profiler.cpp
    src/layout/layout_presets.cpp
    src/ui/performance_settings.cpp
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
)

find_package(TBB REQUIRED)

target_link_libraries(BTQuantTerminal PRIVATE
    Vulkan::Vulkan
    glfw
    dl
    pthread
    TBB::tbb
)

