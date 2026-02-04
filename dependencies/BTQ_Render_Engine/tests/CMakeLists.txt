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
add_executable(test_lockfree_structures test_lockfree_structures.cpp)
add_executable(test_task_scheduler test_task_scheduler.cpp)
add_executable(test_memory_tracker
    test_memory_tracker.cpp
    ../src/performance/memory_tracker.cpp
)

# Link against Threads library
target_link_libraries(test_lockfree_queue Threads::Threads)
target_link_libraries(test_lockfree_structures Threads::Threads)

# For task scheduler test, we need to link against gtest and the main library
find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

target_link_libraries(test_task_scheduler
    Threads::Threads
    GTest::GTest
    GTest::Main
    btq_engine_lib
)

target_link_libraries(test_memory_tracker Threads::Threads)

# Define test executables for frame time
add_executable(test_frame_time_basic test_frame_time_basic.cpp)
add_executable(test_frame_time_graph_advanced test_frame_time_graph_advanced.cpp)

# Link frame time tests
target_link_libraries(test_frame_time_basic
    btq_engine_lib
    Threads::Threads
)

target_link_libraries(test_frame_time_graph_advanced
    btq_engine_lib
    Threads::Threads
)

# Add imgui optimizer test
add_executable(test_imgui_optimizer test_imgui_optimizer.cpp)

# Link imgui optimizer test
target_link_libraries(test_imgui_optimizer
    btq_engine_lib
    Threads::Threads
)

# Add performance regression detector test
add_executable(test_performance_regression test_performance_regression.cpp)

# Link performance regression test
target_link_libraries(test_performance_regression
    btq_engine_lib
    Threads::Threads
)

# Add loading states test
add_executable(test_loading_states test_loading_states.cpp)

# Link loading states test
target_link_libraries(test_loading_states
    btq_engine_lib
    Threads::Threads
)

# Add volume calculation tests
add_executable(volume_calculation_tests volume_calculation_tests.cpp)

# Link volume calculation tests
target_link_libraries(volume_calculation_tests
    btq_engine_lib
    GTest::GTest
    GTest::Main
    Threads::Threads
)

# Set properties for the tests
set_target_properties(test_lockfree_queue PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_memory_tracker PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_frame_time_basic PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_frame_time_graph_advanced PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(test_imgui_optimizer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)