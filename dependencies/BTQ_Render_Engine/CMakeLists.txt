cmake_minimum_required(VERSION 3.16)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(BTQuant_Advanced_Dashboard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wunused-variable")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
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
# Add nlohmann/json
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Build ImGui library
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
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
    src/vulkan_dashboard_advanced.cpp
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
# set(TEST_SOURCES
#     src/test_integration.cpp
#     src/data/hotspine_data_bridge.cpp
#     src/data/market_data_processor.cpp
#     src/data/symbol_manager.cpp
#     src/monitoring/performance_monitor.cpp
#     src/config/dashboard_config.cpp
# )

# # Source files for interactive features test
# set(INTERACTIVE_TEST_SOURCES
#     src/test_interactive_features.cpp
#     src/interaction/interaction_manager.cpp
#     src/ui/advanced_ui_features.cpp
#     src/optimization/performance_optimizer.cpp
#     src/analytics/trading_analytics.cpp
#     src/trading/trading_interface.cpp
#     src/system/system_optimizer.cpp
# )

# External dependencies
set(EXTERNAL_SOURCES
    ../../tests/new/src/hotspine_reader.cpp
    ../../tests/new/src/symbol_registry.cpp
    ../../tests/new/src/utils/dynamic_logger.cpp
)

# Advanced dashboard executable
add_executable(dashboard_advanced ${DASHBOARD_SOURCES} ${EXTERNAL_SOURCES})

# Minimal Vulkan demo for troubleshooting
add_executable(hotspine_vulkan_demo src/hotspine_vulkan_demo.cpp ${EXTERNAL_SOURCES})
target_include_directories(hotspine_vulkan_demo PRIVATE
   ${Vulkan_INCLUDE_DIRS}
   ${X11_INCLUDE_DIR}
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
   ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
   ${glm_SOURCE_DIR}
)
target_link_libraries(hotspine_vulkan_demo PRIVATE
   Vulkan::Vulkan
   ${X11_LIBRARIES}
   pthread
   rt
)
set_target_properties(hotspine_vulkan_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Integration test executable
# add_executable(dashboard_test ${TEST_SOURCES} ${EXTERNAL_SOURCES})

# # Interactive features test executable
# add_executable(test_interactive_features ${INTERACTIVE_TEST_SOURCES} ${EXTERNAL_SOURCES})

# # Comprehensive test suite executable
# add_executable(comprehensive_test_suite
#     tests/comprehensive_test_suite.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

# # Performance validation executable
# add_executable(performance_validation
#     tests/performance_validation.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

# # Functional tests executable
# add_executable(functional_tests
#     tests/functional_tests.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

# # Integration tests executable
# add_executable(integration_tests
#     tests/integration_tests.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

# # Stress tests executable
# add_executable(stress_tests
#     tests/stress_tests.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

# # Professional standards executable
# add_executable(professional_standards
#     tests/professional_standards.cpp
#     ${DASHBOARD_SOURCES}
#     ${EXTERNAL_SOURCES}
# )

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
   ${json_SOURCE_DIR}/include
)

# target_include_directories(dashboard_test PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
# )

target_link_libraries(dashboard_advanced PRIVATE
   Vulkan::Vulkan
   ${X11_LIBRARIES}
   -lXi
   imgui
   nlohmann_json::nlohmann_json
   pthread
   rt
)

# target_link_libraries(dashboard_test PRIVATE
#    pthread
#    rt
# )

# target_include_directories(test_interactive_features PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
# )

# target_link_libraries(test_interactive_features PRIVATE
#    pthread
#    rt
#    ${X11_LIBRARIES}
# )

# Include directories and link libraries for comprehensive test suite
# target_include_directories(comprehensive_test_suite PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(comprehensive_test_suite PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

# # Include directories and link libraries for performance validation
# target_include_directories(performance_validation PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(performance_validation PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

# # Include directories and link libraries for functional tests
# target_include_directories(functional_tests PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(functional_tests PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

# # Include directories and link libraries for integration tests
# target_include_directories(integration_tests PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(integration_tests PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

# # Include directories and link libraries for stress tests
# target_include_directories(stress_tests PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(stress_tests PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

# # Include directories and link libraries for professional standards
# target_include_directories(professional_standards PRIVATE
#    ${Vulkan_INCLUDE_DIRS}
#    ${X11_INCLUDE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR}/include
#    ${CMAKE_CURRENT_SOURCE_DIR}/../ccapi/example/src
#    ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/new/include
#    ${glm_SOURCE_DIR}
#    ${concurrentqueue_SOURCE_DIR}
#    ${VulkanMemoryAllocator_SOURCE_DIR}/include
#    ${imgui_SOURCE_DIR}
#    ${imgui_SOURCE_DIR}/backends
# )

# target_link_libraries(professional_standards PRIVATE
#    Vulkan::Vulkan
#    ${X11_LIBRARIES}
#    imgui
#    pthread
#    rt
# )

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
