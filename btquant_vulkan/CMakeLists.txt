cmake_minimum_required(VERSION 3.28)
project(btquant_vulkan VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(FetchContent)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG docking GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git GIT_TAG master GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(implot)

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json QUIET)

# Force GLFW (system-installed library; no CMake config on most distros).
# Fall back to pkg-config if no glfw3 CMake config is present.
# NOTE: SDL2 intentionally not used — this codebase is GLFW-only.
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    set(glfw3_FOUND TRUE)
    set(GLFW3_USES_PKGCONFIG TRUE)
endif()
if(NOT glfw3_FOUND)
    message(FATAL_ERROR "GLFW3 not found via CMake config or pkg-config")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
else()
    add_compile_options(-g -O0)
endif()

set(SOURCES
    src/main.cpp
    src/core/vulkan_context.cpp
    src/core/vulkan_utils.cpp
    src/data/data_spine.cpp
    src/data/ring_buffer.cpp
    src/data/market_data.cpp
    src/data/market_data_processor.cpp
    src/renderer/render_pipeline.cpp
    src/renderer/gpu_buffers.cpp
    src/renderer/heatmap_compute.cpp
    src/ui/ui_context.cpp
    src/ui/window_manager.cpp
    src/widgets/order_book_widget.cpp
    src/widgets/order_book_depth_widget.cpp
    src/widgets/footprint_widget.cpp
    src/widgets/vpvr_widget.cpp
    src/widgets/multi_vwap_widget.cpp
    src/widgets/risk_panel.cpp
    src/widgets/dom_widget.cpp
    src/widgets/trades_widget.cpp
    src/widgets/tpo_widget.cpp
    src/widgets/heatmap_widget.cpp
    src/widgets/alerts_panel.cpp
    src/widgets/watchlist_widget.cpp
    src/widgets/log_panel.cpp
    src/widgets/connection_panel.cpp
    src/widgets/profile_manager.cpp
    src/widgets/symbol_picker.cpp
    src/widgets/theme_editor.cpp
    src/widgets/recent_fills_panel.cpp
    src/util/settings.cpp
    src/util/theme_io.cpp
    src/util/hotkey_config.cpp
    src/util/layout_io.cpp
    src/widgets/position_calculator.cpp
    src/widgets/order_ticket.cpp
    src/widgets/position_panel.cpp
    src/widgets/risk_limits_panel.cpp
    src/widgets/mini_price_chart.cpp
    src/widgets/hotkey_editor.cpp
    src/data/position_book.cpp
    src/data/risk_guard.cpp
    src/data/trade_journal.cpp
    src/ui/stats_overlay.cpp
    src/data/mock_producer.cpp
)

set(HEADERS
    src/core/vulkan_context.hpp
    src/core/vulkan_utils.hpp
    src/data/data_spine.hpp
    src/data/ring_buffer.hpp
    src/data/market_data.hpp
    src/data/market_data_processor.hpp
    src/renderer/render_pipeline.hpp
    src/renderer/gpu_buffers.hpp
    src/renderer/embedded_shaders.hpp
    src/renderer/heatmap_compute.hpp
    src/ui/ui_context.hpp
    src/ui/window_manager.hpp
    src/widgets/order_book_widget.hpp
    src/widgets/order_book_depth_widget.hpp
    src/widgets/footprint_widget.hpp
    src/widgets/vpvr_widget.hpp
    src/widgets/multi_vwap_widget.hpp
    src/widgets/risk_panel.hpp
    src/widgets/dom_widget.hpp
    src/widgets/trades_widget.hpp
    src/widgets/tpo_widget.hpp
    src/widgets/heatmap_widget.hpp
    src/widgets/alerts_panel.hpp
    src/widgets/watchlist_widget.hpp
    src/widgets/log_panel.hpp
    src/widgets/connection_panel.hpp
    src/widgets/profile_manager.hpp
    src/widgets/symbol_picker.hpp
    src/widgets/theme_editor.hpp
    src/util/settings.hpp
    src/util/theme_io.hpp
    src/util/hotkey_config.hpp
    src/util/layout_io.hpp
    src/widgets/position_calculator.hpp
    src/widgets/order_ticket.hpp
    src/widgets/position_panel.hpp
    src/widgets/risk_limits_panel.hpp
    src/widgets/mini_price_chart.hpp
    src/widgets/hotkey_editor.hpp
    src/data/position_book.hpp
    src/data/risk_guard.hpp
    src/data/trade_journal.hpp
    src/ui/stats_overlay.hpp
    src/data/mock_producer.hpp
)

add_executable(btquant_vulkan ${SOURCES} ${HEADERS}
    # ImGui + ImPlot + backends (built directly into the executable via FetchContent)
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
)

target_compile_definitions(btquant_vulkan PRIVATE
    # ImPlot needs this if you use std::vector etc. without IMGUI_DEFINE_MATH_OPERATORS
    IMGUI_DEFINE_MATH_OPERATORS
)

target_include_directories(btquant_vulkan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR}
)

if(GLFW3_USES_PKGCONFIG)
    target_include_directories(btquant_vulkan PRIVATE ${GLFW3_INCLUDE_DIRS})
endif()

target_link_libraries(btquant_vulkan PRIVATE
    Vulkan::Vulkan Threads::Threads)

if(nlohmann_json_FOUND)
    target_link_libraries(btquant_vulkan PRIVATE nlohmann_json::nlohmann_json)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    enable_testing()
    add_subdirectory(test)
endif()

# Compute shader build target — compiled to SPIR-V and embedded into the binary.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/CMakeLists.txt")
    add_subdirectory(shaders)
endif()
if(glfw3_FOUND)
    if(GLFW3_USES_PKGCONFIG)
        target_link_libraries(btquant_vulkan PRIVATE ${GLFW3_LIBRARIES})
        target_link_directories(btquant_vulkan PRIVATE ${GLFW3_LIBRARY_DIRS})
    else()
        target_link_libraries(btquant_vulkan PRIVATE glfw)
    endif()
    target_compile_definitions(btquant_vulkan PRIVATE BTQUANT_USE_GLFW)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(btquant_vulkan PRIVATE pthread dl)
endif()

message(STATUS "=== btquant_vulkan Configuration ===")
message(STATUS "  Build: ${CMAKE_BUILD_TYPE}, C++${CMAKE_CXX_STANDARD}")
