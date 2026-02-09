cmake_minimum_required(VERSION 3.20)
project(MarketDataCollector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/BTQ_Render_Engine/moodycamel
)

# Source files
set(SOURCES
    src/main.cpp
)

# Add executable for the main market data collector
add_executable(market_data_collector ${SOURCES})

# Add executable for the soak test (minimal version)
add_executable(soak_test src/soak_test.cpp)

# Link libraries
target_link_libraries(market_data_collector 
    Threads::Threads
    atomic
)

target_link_libraries(soak_test 
    Threads::Threads
    atomic
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(market_data_collector PRIVATE -Wall -Wextra -O3)
    target_compile_options(soak_test PRIVATE -Wall -Wextra -O3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(market_data_collector PRIVATE /W4 /O2)
    target_compile_options(soak_test PRIVATE /W4 /O2)
endif()