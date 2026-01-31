    cmake_minimum_required(VERSION 3.15)
project(BTQuantManipulationDetector VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/config
    ${PROJECT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/include
    ${PROJECT_SOURCE_DIR}/../../dependencies/ccapi/example/src/market_data_collector
)

# Source files - Core
set(CORE_SOURCES
    src/hotspine_reader.cpp
    src/hotspine_extended_reader.cpp
    src/symbol_registry.cpp
    src/config/config_loader.cpp
    src/utils/dynamic_logger.cpp
    src/utils/shm_health_monitor.cpp
    src/utils/alert_manager.cpp
    src/utils/recovery_controller.cpp
    ../../dependencies/BTQ_Render_Engine/src/analytics/cluster_engine.cpp
    ../../dependencies/BTQ_Render_Engine/src/analytics/cumulative_delta_tracker.cpp
)

# Source files - Detectors
set(DETECTOR_SOURCES
    src/detectors/stop_hunt_detector.cpp
    src/detectors/liquidity_imbalance_detector.cpp
    src/detectors/whale_frontrun_detector.cpp
    src/detectors/spread_arbitrage_detector.cpp
    src/detectors/spoofing_detector.cpp
    src/detectors/detector_plugin.cpp
)

# Core library
add_library(btquant_core STATIC ${CORE_SOURCES})
target_link_libraries(btquant_core PUBLIC pthread rt)

# Detector library
add_library(btquant_detectors STATIC ${DETECTOR_SOURCES})
target_link_libraries(btquant_detectors PUBLIC btquant_core)

# Main monitor executable
add_executable(manipulation_monitor src/main_monitor.cpp)
target_link_libraries(manipulation_monitor PRIVATE btquant_detectors btquant_core)

# Examples
add_executable(simple_monitor examples/simple_monitor.cpp)
target_link_libraries(simple_monitor PRIVATE btquant_detectors btquant_core)

add_executable(multi_exchange_monitor examples/multi_exchange_monitor.cpp)
target_link_libraries(multi_exchange_monitor PRIVATE btquant_detectors btquant_core)

# Analytics test
add_executable(analytics_test src/analytics_test.cpp)
target_link_libraries(analytics_test PRIVATE btquant_core)

# Cumulative delta tracker test
add_executable(cumulative_delta_tracker_test src/cumulative_delta_tracker_test.cpp)
target_link_libraries(cumulative_delta_tracker_test PRIVATE btquant_core)

# Install targets
install(TARGETS manipulation_monitor simple_monitor multi_exchange_monitor
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/btquant)
install(DIRECTORY config/ DESTINATION etc/btquant)

# Print build info
message(STATUS "BTQuant Manipulation Detector v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
