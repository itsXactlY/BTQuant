cmake_minimum_required(VERSION 3.10)
project(BTQ_Render_Engine_Tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../_deps/concurrentqueue-src)

# Define test executables
add_executable(test_lockfree_queue test_lockfree_queue.cpp)
add_executable(test_memory_tracker
    test_memory_tracker.cpp
    ../src/performance/memory_tracker.cpp
)

# Link against Threads library
target_link_libraries(test_lockfree_queue Threads::Threads)
target_link_libraries(test_memory_tracker Threads::Threads)

# Set properties for the tests
set_target_properties(test_lockfree_queue PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_memory_tracker PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)