cmake_minimum_required(VERSION 3.15)
project(BTQuant_Render_Engine LANGUAGES CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# C++26 Standard - Cutting Edge (GCC 15+ / Clang 18+) for main application
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    ${IMPLOT_DIR}/implot_demo.cpp
)

set(DASHBOARD_SOURCES
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
    src/components/indicator_renderer.cpp
    src/components/hierarchical_selector.cpp
    src/components/orderbook_panel.cpp
    src/components/orderbook_history.cpp
    src/components/alerts_panel.cpp
    src/components/status_bar_panel.cpp
    src/components/watchlist_panel.cpp
    src/components/tape_panel.cpp
    src/components/volume_profile_panel.cpp
    src/components/depth_chart_panel.cpp
    src/components/footprint_panel.cpp
    src/components/tpo_panel.cpp
    src/components/dom_surface_panel.cpp
    src/components/MarketMicrostructureRenderer.cpp
    src/components/VulkanSynchronization.cpp
    src/components/realtime_dashboard_component.cpp
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
    src/config/dashboard_config.cpp
    src/ui/unified_theme_system.cpp
    src/data/market_data_processor.cpp
    src/data/symbol_manager.cpp
    src/monitoring/performance_monitor.cpp
    # Analytics and advanced features
    src/analytics/technical_analysis.cpp
    src/analytics/trading_analytics.cpp
    src/analytics/cluster_engine.cpp
    src/analytics/volume_calculator.cpp
    src/indicators/session_vwap.cpp
    src/indicators/rolling_vwap.cpp
    src/indicators/vwap_alerts.cpp
    src/layout/dashboard_layout_manager.cpp
    src/data/ui_data_manager.cpp
    src/system/system_optimizer.cpp
)

add_executable(dashboard_advanced ${DASHBOARD_SOURCES} ${LIB_SOURCES} ${COMPILED_SHADERS})

target_include_directories(dashboard_advanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${glm_SOURCE_DIR}
    ${json_SOURCE_DIR}/include
    ${concurrentqueue_SOURCE_DIR}
)

find_package(TBB REQUIRED)

target_link_libraries(dashboard_advanced PRIVATE
    Vulkan::Vulkan
    glfw
    dl
    pthread
    TBB::tbb
)

# Optimization
# target_compile_definitions(dashboard_advanced PRIVATE IMGUI_ENABLE_DOCKING)
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     target_compile_options(dashboard_advanced PRIVATE
#             -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops
#             -fstrict-aliasing)
#         target_link_options(dashboard_advanced PRIVATE -flto)
#     endif()

# # Performance monitoring
# target_compile_definitions(dashboard_advanced PRIVATE ENABLE_PERFORMANCE_MONITORING)

# # Shader compilation target
# add_custom_target(compile_shaders DEPENDS ${COMPILED_SHADERS})
# add_dependencies(dashboard_advanced compile_shaders)

# # Install compiled shaders
# install(FILES ${COMPILED_SHADERS} DESTINATION ${CMAKE_INSTALL_BINDIR}/shaders)

# # Trading Terminal Executable
# set(TRADING_TERMINAL_SOURCES
#     src/main_trading_terminal.cpp
#     src/vulkan_dashboard_advanced.cpp
#     src/system/VulkanCore.cpp
#     src/data/hotspine_data_bridge.cpp
#     src/components/quant_workspace_component.cpp
#     src/components/panel_base.cpp
#     src/components/chart_panel.cpp
#     src/components/metrics_panel.cpp
#     src/components/panel_manager.cpp
#     src/components/chart_manager.cpp
#     src/components/indicator_renderer.cpp
#     src/components/hierarchical_selector.cpp
#     src/components/orderbook_panel.cpp
#     src/components/alerts_panel.cpp
#     src/components/status_bar_panel.cpp
#     src/components/watchlist_panel.cpp
#     src/components/tape_panel.cpp
#     src/components/volume_profile_panel.cpp
#     src/components/depth_chart_panel.cpp
#     src/components/footprint_panel.cpp
#     src/components/tpo_panel.cpp
#     src/components/dom_surface_panel.cpp
#     src/components/MarketMicrostructureRenderer.cpp
#     src/components/VulkanSynchronization.cpp
#     src/components/realtime_dashboard_component.cpp
#     src/components/correlation_heatmap_component.cpp
#     src/components/symbol_selector.cpp
#     src/components/trading_orders_panel.cpp
#     src/components/trading_positions_panel.cpp
#     src/components/risk_metrics_panel.cpp
#     src/components/histogram_panel.cpp
#     src/components/scatter_plot_panel.cpp
#     src/components/time_series_panel.cpp
#     src/components/screener_panel.cpp
#     src/components/log_panel.cpp
#     src/trading/trading_interface.cpp
#     src/trading/order_manager.cpp
#     src/trading/position_manager.cpp
#     src/trading/risk_assessment.cpp
#     src/trading/advanced_trading_features.cpp
#     src/data/symbol_registry.cpp
#     src/optimization/performance_optimizer.cpp
#     src/system/GPUMemoryManager.cpp
#     src/components/theme_manager.cpp
#     src/components/interaction_manager.cpp
#     src/config/dashboard_config.cpp
#     src/data/market_data_processor.cpp
#     src/data/symbol_manager.cpp
#     src/monitoring/performance_monitor.cpp
#     # Analytics and advanced features
#     src/analytics/technical_analysis.cpp
#     src/analytics/trading_analytics.cpp
#     src/layout/dashboard_layout_manager.cpp
#     src/data/ui_data_manager.cpp
#     src/system/system_optimizer.cpp
# )

# add_executable(trading_terminal ${TRADING_TERMINAL_SOURCES} ${LIB_SOURCES} ${COMPILED_SHADERS})

# target_include_directories(trading_terminal PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${IMGUI_DIR}
#     ${IMGUI_DIR}/backends
#     ${IMPLOT_DIR}
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
#     ${Vulkan_INCLUDE_DIRS}
#     ${glm_SOURCE_DIR}
#     ${json_SOURCE_DIR}/include
#     ${concurrentqueue_SOURCE_DIR}
# )

# target_link_libraries(trading_terminal PRIVATE
#     Vulkan::Vulkan
#     glfw
#     dl
#     pthread
# )

# # Optimization
# target_compile_definitions(trading_terminal PRIVATE IMGUI_ENABLE_DOCKING)
# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     target_compile_options(trading_terminal PRIVATE
#             -Wall -Wextra -O3 -march=native -ffast-math -funroll-loops
#             -fstrict-aliasing)
#         target_link_options(trading_terminal PRIVATE -flto)
#     endif()

# # Performance monitoring
# target_compile_definitions(trading_terminal PRIVATE ENABLE_PERFORMANCE_MONITORING)

# # Shader compilation target
# add_dependencies(trading_terminal compile_shaders)

# Benchmark executable
# Comment out benchmark_performance target for now since benchmark library is not found
# add_executable(benchmark_performance tests/benchmark_performance.cpp)
# target_include_directories(benchmark_performance PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
#     ${Vulkan_INCLUDE_DIRS}
#     ${glm_SOURCE_DIR}
#     ${json_SOURCE_DIR}/include
#     ${concurrentqueue_SOURCE_DIR}
# )
# target_link_libraries(benchmark_performance PRIVATE
#     Vulkan::Vulkan
#     glfw
#     dl
#     pthread
# )

# Unit test executable
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    # Add our new LOD rendering test
    add_executable(test_lod_rendering
        ${PROJECT_SOURCE_DIR}/../../tests/test_lod_rendering.cpp
        src/components/footprint_panel.cpp
        src/components/panel_base.cpp
        src/components/theme_manager.cpp
        src/components/interaction_manager.cpp
        src/analytics/cluster_engine.cpp
        src/analytics/volume_calculator.cpp
        src/data/market_data_processor.cpp
        src/data/hotspine_data_bridge.cpp
        src/data/symbol_registry.cpp
        src/trading/order_manager.cpp
        src/trading/position_manager.cpp
        src/trading/risk_assessment.cpp
        src/trading/advanced_trading_features.cpp
        src/analytics/technical_analysis.cpp
        src/analytics/trading_analytics.cpp
        src/monitoring/performance_monitor.cpp
        src/system/system_optimizer.cpp
        src/data/symbol_manager.cpp
        src/config/dashboard_config.cpp
        src/components/panel_manager.cpp
        src/components/chart_manager.cpp
        src/components/chart_panel.cpp
        src/components/metrics_panel.cpp
        src/components/orderbook_panel.cpp
        src/components/watchlist_panel.cpp
        src/components/tape_panel.cpp
        src/components/volume_profile_panel.cpp
        src/components/depth_chart_panel.cpp
        src/components/tpo_panel.cpp
        src/components/dom_surface_panel.cpp
        src/components/trading_orders_panel.cpp
        src/components/trading_positions_panel.cpp
        src/components/risk_metrics_panel.cpp
        src/components/histogram_panel.cpp
        src/components/scatter_plot_panel.cpp
        src/components/time_series_panel.cpp
        src/components/screener_panel.cpp
        src/components/log_panel.cpp
        src/components/alerts_panel.cpp
        src/components/status_bar_panel.cpp
        src/components/quant_workspace_component.cpp
        src/components/realtime_dashboard_component.cpp
        src/components/correlation_heatmap_component.cpp
        src/components/symbol_selector.cpp
        src/components/hierarchical_selector.cpp
        src/components/indicator_renderer.cpp
        src/components/MarketMicrostructureRenderer.cpp
        src/components/VulkanSynchronization.cpp
        src/system/VulkanCore.cpp
        src/system/GPUMemoryManager.cpp
        src/optimization/performance_optimizer.cpp
        src/layout/dashboard_layout_manager.cpp
        src/data/ui_data_manager.cpp
        src/vulkan_dashboard_advanced.cpp
        ${LIB_SOURCES}
    )
    target_include_directories(test_lod_rendering PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_lod_rendering PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
endif()

# Unit test executable continued
# if(GTest_FOUND)
#     add_executable(component_enhancement_tests
#         tests/component_enhancement_tests.cpp
#         tests/footprint_panel_tests.cpp
#         src/data/market_data_processor.cpp
#         src/data/hotspine_data_bridge.cpp
#         src/data/symbol_registry.cpp
#         src/trading/order_manager.cpp
#         src/trading/position_manager.cpp
#         src/trading/risk_assessment.cpp
#         src/trading/advanced_trading_features.cpp
#         src/analytics/technical_analysis.cpp
#         src/analytics/trading_analytics.cpp
#         src/components/theme_manager.cpp
#         src/components/interaction_manager.cpp
#         src/monitoring/performance_monitor.cpp
#         src/system/system_optimizer.cpp
#         src/data/symbol_manager.cpp
#         src/config/dashboard_config.cpp
#         src/components/panel_manager.cpp
#         src/components/panel_base.cpp
#         src/components/chart_manager.cpp
#         src/components/chart_panel.cpp
#         src/components/metrics_panel.cpp
#         src/components/orderbook_panel.cpp
#         src/components/watchlist_panel.cpp
#         src/components/tape_panel.cpp
#         src/components/volume_profile_panel.cpp
#         src/components/depth_chart_panel.cpp
#         src/components/footprint_panel.cpp
#         src/components/tpo_panel.cpp
#         src/components/dom_surface_panel.cpp
#         src/components/trading_orders_panel.cpp
#         src/components/trading_positions_panel.cpp
#         src/components/risk_metrics_panel.cpp
#         src/components/histogram_panel.cpp
#         src/components/scatter_plot_panel.cpp
#         src/components/time_series_panel.cpp
#         src/components/screener_panel.cpp
#         src/components/log_panel.cpp
#         src/components/alerts_panel.cpp
#         src/components/status_bar_panel.cpp
#         src/components/quant_workspace_component.cpp
#         src/components/realtime_dashboard_component.cpp
#         src/components/correlation_heatmap_component.cpp
#         src/components/symbol_selector.cpp
#         src/components/hierarchical_selector.cpp
#         src/components/indicator_renderer.cpp
#         src/components/MarketMicrostructureRenderer.cpp
#         src/components/VulkanSynchronization.cpp
#         src/system/VulkanCore.cpp
#         src/system/GPUMemoryManager.cpp
#         src/optimization/performance_optimizer.cpp
#         src/layout/dashboard_layout_manager.cpp
#         src/data/ui_data_manager.cpp
#         src/vulkan_dashboard_advanced.cpp
#         src/performance_monitor.cpp
#         src/symbol_registry.cpp
#         src/components/technical_indicators_component.cpp
#     )
#
#     # Crosshair functionality tests
#     add_executable(crosshair_tests
#         tests/test_chart_panel_crosshair.cpp
#         src/data/market_data_processor.cpp
#         src/data/hotspine_data_bridge.cpp
#         src/data/symbol_registry.cpp
#         src/trading/order_manager.cpp
#         src/trading/position_manager.cpp
#         src/trading/risk_assessment.cpp
#         src/trading/advanced_trading_features.cpp
#         src/analytics/technical_analysis.cpp
#         src/analytics/trading_analytics.cpp
#         src/components/theme_manager.cpp
#         src/components/interaction_manager.cpp
#         src/monitoring/performance_monitor.cpp
#         src/data/symbol_manager.cpp
#         src/config/dashboard_config.cpp
#         src/components/panel_manager.cpp
#         src/components/panel_base.cpp
#         src/components/chart_manager.cpp
#         src/components/chart_panel.cpp
#         src/components/metrics_panel.cpp
#         src/components/orderbook_panel.cpp
#         src/components/watchlist_panel.cpp
#         src/components/tape_panel.cpp
#         src/components/volume_profile_panel.cpp
#         src/components/depth_chart_panel.cpp
#         src/components/footprint_panel.cpp
#         src/components/tpo_panel.cpp
#         src/components/dom_surface_panel.cpp
#         src/components/trading_orders_panel.cpp
#         src/components/trading_positions_panel.cpp
#         src/components/risk_metrics_panel.cpp
#         src/components/histogram_panel.cpp
#         src/components/scatter_plot_panel.cpp
#         src/components/time_series_panel.cpp
#         src/components/screener_panel.cpp
#         src/components/log_panel.cpp
#         src/components/alerts_panel.cpp
#         src/components/status_bar_panel.cpp
#         src/components/quant_workspace_component.cpp
#         src/components/realtime_dashboard_component.cpp
#         src/components/correlation_heatmap_component.cpp
#         src/components/symbol_selector.cpp
#         src/components/hierarchical_selector.cpp
#         src/components/indicator_renderer.cpp
#         src/components/MarketMicrostructureRenderer.cpp
#         src/components/VulkanSynchronization.cpp
#         src/system/GPUMemoryManager.cpp
#         src/optimization/performance_optimizer.cpp
#         src/layout/dashboard_layout_manager.cpp
#         src/data/ui_data_manager.cpp
#         src/performance_monitor.cpp
#         src/symbol_registry.cpp
#         src/components/technical_indicators_component.cpp
#     )
#     target_include_directories(component_enhancement_tests PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${Vulkan_INCLUDE_DIRS}
#         ${glm_SOURCE_DIR}
#         ${json_SOURCE_DIR}/include
#         ${concurrentqueue_SOURCE_DIR}
#         ${IMGUI_DIR}
#         ${IMGUI_DIR}/backends
#         ${IMPLOT_DIR}
#     )
#     target_link_libraries(component_enhancement_tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
#
#     target_include_directories(crosshair_tests PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${Vulkan_INCLUDE_DIRS}
#         ${glm_SOURCE_DIR}
#         ${json_SOURCE_DIR}/include
#         ${concurrentqueue_SOURCE_DIR}
#         ${IMGUI_DIR}
#         ${IMGUI_DIR}/backends
#         ${IMPLOT_DIR}
#     )
#     target_link_libraries(crosshair_tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
# endif()

# Data flow and layout tests
# if(GTest_FOUND)
#     add_executable(data_flow_layout_tests
#         tests/test_data_flow_layout.cpp
#         src/data/market_data_processor.cpp
#         src/data/hotspine_data_bridge.cpp
#         src/data/symbol_registry.cpp
#         src/trading/order_manager.cpp
#         src/trading/position_manager.cpp
#         src/trading/risk_assessment.cpp
#         src/trading/advanced_trading_features.cpp
#         src/analytics/technical_analysis.cpp
#         src/analytics/trading_analytics.cpp
#         src/components/theme_manager.cpp
#         src/components/interaction_manager.cpp
#         src/monitoring/performance_monitor.cpp
#         src/data/symbol_manager.cpp
#         src/config/dashboard_config.cpp
#         src/components/panel_manager.cpp
#         src/components/panel_base.cpp
#         src/components/chart_manager.cpp
#         src/components/chart_panel.cpp
#         src/components/metrics_panel.cpp
#         src/components/orderbook_panel.cpp
#         src/components/watchlist_panel.cpp
#         src/components/tape_panel.cpp
#         src/components/volume_profile_panel.cpp
#         src/components/depth_chart_panel.cpp
#         src/components/footprint_panel.cpp
#         src/components/tpo_panel.cpp
#         src/components/dom_surface_panel.cpp
#         src/components/trading_orders_panel.cpp
#         src/components/trading_positions_panel.cpp
#         src/components/risk_metrics_panel.cpp
#         src/components/histogram_panel.cpp
#         src/components/scatter_plot_panel.cpp
#         src/components/time_series_panel.cpp
#         src/components/screener_panel.cpp
#         src/components/log_panel.cpp
#         src/components/alerts_panel.cpp
#         src/components/status_bar_panel.cpp
#         src/components/quant_workspace_component.cpp
#         src/components/realtime_dashboard_component.cpp
#         src/components/correlation_heatmap_component.cpp
#         src/components/symbol_selector.cpp
#         src/components/hierarchical_selector.cpp
#         src/components/indicator_renderer.cpp
#         src/components/MarketMicrostructureRenderer.cpp
#         src/components/VulkanSynchronization.cpp
#         src/system/GPUMemoryManager.cpp
#         src/optimization/performance_optimizer.cpp
#         src/layout/dashboard_layout_manager.cpp
#         src/data/ui_data_manager.cpp
#         src/performance_monitor.cpp
#         src/symbol_registry.cpp
#         src/components/technical_indicators_component.cpp
#         src/data/unified_data_pipeline.cpp
#     )
#     target_include_directories(data_flow_layout_tests PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/include/components
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${IMGUI_DIR}
#         ${IMGUI_DIR}/backends
#         ${IMPLOT_DIR}
#         ${Vulkan_INCLUDE_DIRS}
#         ${glm_SOURCE_DIR}
#         ${json_SOURCE_DIR}/include
#         ${concurrentqueue_SOURCE_DIR}
#     )
#     # Use C++17 for tests due to JSON library compatibility issues
#     target_compile_features(data_flow_layout_tests PRIVATE cxx_std_17)
#     target_link_libraries(data_flow_layout_tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
# endif()
#
# # Integration tests
# if(GTest_FOUND)
#     add_executable(integration_tests
#         tests/integration_tests.cpp
#         src/components/panel_manager.cpp
#         src/components/panel_base.cpp
#         src/components/chart_manager.cpp
#         src/components/chart_panel.cpp
#         src/components/metrics_panel.cpp
#         src/components/orderbook_panel.cpp
#         src/components/watchlist_panel.cpp
#         src/components/tape_panel.cpp
#         src/components/volume_profile_panel.cpp
#         src/components/depth_chart_panel.cpp
#         src/components/footprint_panel.cpp
#         src/components/tpo_panel.cpp
#         src/components/dom_surface_panel.cpp
#         src/components/trading_orders_panel.cpp
#         src/components/trading_positions_panel.cpp
#         src/components/risk_metrics_panel.cpp
#         src/components/histogram_panel.cpp
#         src/components/scatter_plot_panel.cpp
#         src/components/time_series_panel.cpp
#         src/components/screener_panel.cpp
#         src/components/log_panel.cpp
#         src/components/alerts_panel.cpp
#         src/components/status_bar_panel.cpp
#         src/components/quant_workspace_component.cpp
#         src/components/realtime_dashboard_component.cpp
#         src/components/correlation_heatmap_component.cpp
#         src/components/symbol_selector.cpp
#         src/components/hierarchical_selector.cpp
#         src/components/indicator_renderer.cpp
#         src/components/MarketMicrostructureRenderer.cpp
#         src/components/VulkanSynchronization.cpp
#         src/components/theme_manager.cpp
#         src/components/interaction_manager.cpp
#         src/components/performance_monitor_panel.cpp
#         src/trading/trading_interface.cpp
#         src/trading/order_manager.cpp
#         src/trading/position_manager.cpp
#         src/trading/risk_assessment.cpp
#         src/trading/advanced_trading_features.cpp
#         src/data/symbol_registry.cpp
#         src/optimization/performance_optimizer.cpp
#         src/system/GPUMemoryManager.cpp
#         src/config/dashboard_config.cpp
#         src/data/market_data_processor.cpp
#         src/data/symbol_manager.cpp
#         src/monitoring/performance_monitor.cpp
#         src/system/VulkanCore.cpp
#         src/system/GPUMemoryManager.cpp
#         src/optimization/performance_optimizer.cpp
#         src/layout/dashboard_layout_manager.cpp
#         src/data/ui_data_manager.cpp
#         src/vulkan_dashboard_advanced.cpp
#         src/performance_monitor.cpp
#         src/symbol_registry.cpp
#         src/components/technical_indicators_component.cpp
#         src/data/hotspine_data_bridge.cpp
#         ${LIB_SOURCES}
#     )
#     target_include_directories(integration_tests PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${Vulkan_INCLUDE_DIRS}
#         ${glm_SOURCE_DIR}
#         ${json_SOURCE_DIR}/include
#         ${concurrentqueue_SOURCE_DIR}
#         ${IMGUI_DIR}
#         ${IMGUI_DIR}/backends
#         ${IMPLOT_DIR}
#     )
#     target_link_libraries(integration_tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
# endif()
#
# # Comprehensive tests
# if(GTest_FOUND)
#     add_executable(comprehensive_tests
#         tests/comprehensive_tests.cpp
#         src/data/market_data_processor.cpp
#         src/data/hotspine_data_bridge.cpp
#         src/data/symbol_registry.cpp
#         src/trading/order_manager.cpp
#         src/trading/position_manager.cpp
#         src/trading/risk_assessment.cpp
#         src/trading/advanced_trading_features.cpp
#         src/analytics/technical_analysis.cpp
#         src/analytics/trading_analytics.cpp
#         src/components/theme_manager.cpp
#         src/components/interaction_manager.cpp
#         src/monitoring/performance_monitor.cpp
#         src/system/system_optimizer.cpp
#         src/data/symbol_manager.cpp
#         src/config/dashboard_config.cpp
#         src/components/panel_manager.cpp
#         src/components/panel_base.cpp
#         src/components/chart_manager.cpp
#         src/components/chart_panel.cpp
#         src/components/metrics_panel.cpp
#         src/components/orderbook_panel.cpp
#         src/components/watchlist_panel.cpp
#         src/components/tape_panel.cpp
#         src/components/volume_profile_panel.cpp
#         src/components/depth_chart_panel.cpp
#         src/components/footprint_panel.cpp
#         src/components/tpo_panel.cpp
#         src/components/dom_surface_panel.cpp
#         src/components/trading_orders_panel.cpp
#         src/components/trading_positions_panel.cpp
#         src/components/risk_metrics_panel.cpp
#         src/components/histogram_panel.cpp
#         src/components/scatter_plot_panel.cpp
#         src/components/time_series_panel.cpp
#         src/components/screener_panel.cpp
#         src/components/log_panel.cpp
#         src/components/alerts_panel.cpp
#         src/components/status_bar_panel.cpp
#         src/components/quant_workspace_component.cpp
#         src/components/realtime_dashboard_component.cpp
#         src/components/correlation_heatmap_component.cpp
#         src/components/symbol_selector.cpp
#         src/components/hierarchical_selector.cpp
#         src/components/indicator_renderer.cpp
#         src/components/MarketMicrostructureRenderer.cpp
#         src/components/VulkanSynchronization.cpp
#         src/system/VulkanCore.cpp
#         src/system/GPUMemoryManager.cpp
#         src/optimization/performance_optimizer.cpp
#         src/layout/dashboard_layout_manager.cpp
#         src/data/ui_data_manager.cpp
#         src/vulkan_dashboard_advanced.cpp
#         src/performance_monitor.cpp
#         src/symbol_registry.cpp
#         src/components/technical_indicators_component.cpp
#     )
#     target_include_directories(comprehensive_tests PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${Vulkan_INCLUDE_DIRS}
#         ${glm_SOURCE_DIR}
#         ${json_SOURCE_DIR}/include
#         ${concurrentqueue_SOURCE_DIR}
#         ${IMGUI_DIR}
#         ${IMGUI_DIR}/backends
#         ${IMPLOT_DIR}
#     )
#     target_link_libraries(comprehensive_tests PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
# endif()

# Unit test for VolumeAnalysisType
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_volume_analysis_type tests/test_volume_analysis_type.cpp)
    target_include_directories(test_volume_analysis_type PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_volume_analysis_type PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_volume_analysis_type COMMAND test_volume_analysis_type)
endif()

# Unit test for Footprint Panel Zoom functionality (mathematical algorithm only)
if(GTest_FOUND)
    add_executable(test_footprint_zoom_math_only
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_footprint_zoom.cpp
    )
    target_include_directories(test_footprint_zoom_math_only PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_footprint_zoom_math_only PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_footprint_zoom_math_only COMMAND test_footprint_zoom_math_only)
endif()

# Unit test for Improved Footprint Panel Zoom functionality
if(GTest_FOUND)
    add_executable(test_improved_footprint_zoom
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_improved_footprint_zoom.cpp
    )
    target_include_directories(test_improved_footprint_zoom PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_improved_footprint_zoom PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_improved_footprint_zoom COMMAND test_improved_footprint_zoom)
endif()

# Unit test for Split Volume Display functionality
if(GTest_FOUND)
    add_executable(test_split_volume_display
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_split_volume_display.cpp
    )
    target_include_directories(test_split_volume_display PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_split_volume_display PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_split_volume_display COMMAND test_split_volume_display)
endif()

# Unit test for Profile Settings functionality
if(GTest_FOUND)
    add_executable(test_profile_settings
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_profile_settings.cpp
    )
    target_include_directories(test_profile_settings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_profile_settings PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_profile_settings COMMAND test_profile_settings)
endif()

# Unit test for Step Profile Rendering functionality
if(GTest_FOUND)
    add_executable(test_step_profile_rendering
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_step_profile_rendering.cpp
        src/components/volume_profile_panel.cpp
        src/components/panel_base.cpp
        src/components/theme_manager.cpp
        src/components/interaction_manager.cpp
        src/data/market_data_processor.cpp
        src/data/hotspine_data_bridge.cpp
        src/data/symbol_registry.cpp
        src/trading/order_manager.cpp
        src/trading/position_manager.cpp
        src/trading/risk_assessment.cpp
        src/trading/advanced_trading_features.cpp
        src/analytics/technical_analysis.cpp
        src/analytics/trading_analytics.cpp
        src/analytics/cluster_engine.cpp
        src/analytics/volume_calculator.cpp
        src/monitoring/performance_monitor.cpp
        src/system/system_optimizer.cpp
        src/data/symbol_manager.cpp
        src/config/dashboard_config.cpp
        src/components/panel_manager.cpp
        src/components/chart_manager.cpp
        src/components/chart_panel.cpp
        src/components/metrics_panel.cpp
        src/components/orderbook_panel.cpp
        src/components/watchlist_panel.cpp
        src/components/tape_panel.cpp
        src/components/depth_chart_panel.cpp
        src/components/footprint_panel.cpp
        src/components/tpo_panel.cpp
        src/components/dom_surface_panel.cpp
        src/components/trading_orders_panel.cpp
        src/components/trading_positions_panel.cpp
        src/components/risk_metrics_panel.cpp
        src/components/histogram_panel.cpp
        src/components/scatter_plot_panel.cpp
        src/components/time_series_panel.cpp
        src/components/screener_panel.cpp
        src/components/log_panel.cpp
        src/components/alerts_panel.cpp
        src/components/status_bar_panel.cpp
        src/components/quant_workspace_component.cpp
        src/components/realtime_dashboard_component.cpp
        src/components/correlation_heatmap_component.cpp
        src/components/symbol_selector.cpp
        src/components/hierarchical_selector.cpp
        src/components/indicator_renderer.cpp
        src/components/MarketMicrostructureRenderer.cpp
        src/components/VulkanSynchronization.cpp
        src/system/VulkanCore.cpp
        src/system/GPUMemoryManager.cpp
        src/optimization/performance_optimizer.cpp
        src/layout/dashboard_layout_manager.cpp
        src/data/ui_data_manager.cpp
        src/vulkan_dashboard_advanced.cpp
        ${LIB_SOURCES}
    )
    target_include_directories(test_step_profile_rendering PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${glm_SOURCE_DIR}
        ${json_SOURCE_DIR}/include
        ${concurrentqueue_SOURCE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${IMPLOT_DIR}
    )
    target_link_libraries(test_step_profile_rendering PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
    add_test(NAME test_step_profile_rendering COMMAND test_step_profile_rendering)
endif()

