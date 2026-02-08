cmake_minimum_required(VERSION 3.20)
project(MarketDataProcessorBenchmark)

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

# Define source files
set(SOURCES
    benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

# Create executables
add_executable(benchmark_market_data_processor 
    benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

add_executable(enhanced_benchmark_market_data_processor 
    enhanced_benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

add_executable(optimized_benchmark_market_data_processor 
    optimized_benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

add_executable(ultra_optimized_benchmark_market_data_processor 
    ultra_optimized_benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

add_executable(final_benchmark_market_data_processor 
    final_benchmark_market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp
    dependencies/BTQ_Render_Engine/src/data/cache_manager.cpp
)

# Link libraries
target_link_libraries(benchmark_market_data_processor 
    Threads::Threads
)

target_link_libraries(enhanced_benchmark_market_data_processor 
    Threads::Threads
)

# Compiler-specific options
if(MSVC)
    target_compile_options(benchmark_market_data_processor PRIVATE /W4)
else()
    target_compile_options(benchmark_market_data_processor PRIVATE -Wall -Wextra -pthread)
endif()

# Set output directory
set_target_properties(benchmark_market_data_processor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the Render Engine subdirectory to build BTQuantTerminal
add_subdirectory(dependencies/BTQ_Render_Engine)