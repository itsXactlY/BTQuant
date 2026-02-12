cmake_minimum_required(VERSION 3.15)
project(PubBTQuant LANGUAGES CXX)

# Set C++ standard to C++26
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")

# Also ensure the standard is set via flag for compatibility
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26")

# Define source files for the main library
set(MAIN_PROJECT_SOURCES
    src/hotspine_layout_v3_impl.cpp
)

# Include subdirectories that have CMakeLists.txt files
add_subdirectory(dependencies/BTQ_Render_Engine)