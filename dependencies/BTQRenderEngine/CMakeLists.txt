cmake_minimum_required(VERSION 3.10)
project(BTQRenderEngine)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)

# Add the TPO engine library
add_library(tpoengine 
    src/analytics/tpoengine.cpp
)

# Create an executable for testing
add_executable(tpotest 
    test/tpotest.cpp
)

# Link the TPO engine library to the test executable
target_link_libraries(tpotest tpoengine)

# Set output directory for binaries
set_target_properties(tpotest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)