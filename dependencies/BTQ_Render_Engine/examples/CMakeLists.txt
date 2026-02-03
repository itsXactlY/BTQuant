cmake_minimum_required(VERSION 3.15)
project(BTQ_Render_Engine_Examples)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Dependencies
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../_deps/concurrentqueue-src)

# Define example executables
add_executable(demo_threadsafe_communication demo_threadsafe_communication.cpp)

# Link against Threads library
target_link_libraries(demo_threadsafe_communication Threads::Threads)

# Set properties for the examples
set_target_properties(demo_threadsafe_communication PROPERTIES
    CXX_STANDARD 26
    CXX_STANDARD_REQUIRED ON
)