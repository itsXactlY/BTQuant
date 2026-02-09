cmake_minimum_required(VERSION 3.20)
project(MarketDataCollector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# External Headers
include(FetchContent)
FetchContent_Declare(concurrentqueue GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git GIT_TAG v1.0.4)
FetchContent_MakeAvailable(concurrentqueue)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/BTQ_Render_Engine/include)
include_directories(${concurrentqueue_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Market Data Collector with Warm-Up Generator
add_executable(market_data_collector
    market_data_collector/src/main.cpp
    market_data_collector/src/data/hotspine_data_bridge.cpp
)

# Market Data Processor Library (Header-only implementation)
add_library(market_data_processor_lib INTERFACE)


# Link libraries
target_link_libraries(market_data_collector
    Threads::Threads
    market_data_processor_lib
)

# Compiler-specific options
if(MSVC)
    target_compile_options(market_data_collector PRIVATE /W4)
else()
    target_compile_options(market_data_collector PRIVATE -Wall -Wextra -pthread)
endif()

# Set output directory
set_target_properties(market_data_collector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Test executable for MarketDataProcessor
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_market_data_processor
        tests/test_market_data_processor.cpp
    )

    add_executable(latency_test
        tests/latency_test.cpp
    )

    target_include_directories(test_market_data_processor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_include_directories(latency_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(test_market_data_processor
        Threads::Threads
        market_data_processor_lib
    )

    target_link_libraries(latency_test
        Threads::Threads
        market_data_processor_lib
    )

    add_test(NAME test_market_data_processor COMMAND test_market_data_processor)
    add_test(NAME latency_test COMMAND latency_test)
endif()


# Add the Render Engine subdirectory to build BTQuantTerminal
add_subdirectory(dependencies/BTQ_Render_Engine)