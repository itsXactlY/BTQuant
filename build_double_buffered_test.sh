#!/bin/bash

# Build script for double buffered state test
echo "Building double buffered state test..."

# Create build directory if it doesn't exist
mkdir -p build_double_buffered

# Compile the test
g++ -std=c++17 -Wall -Wextra -pthread \
    -I./include \
    test_double_buffered_state.cpp \
    -o build_double_buffered/test_double_buffered_state

# Compile the example
g++ -std=c++17 -Wall -Wextra -pthread \
    -I./include \
    example_double_buffered_state.cpp \
    -o build_double_buffered/example_double_buffered_state

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Binaries created in build_double_buffered/"
else
    echo "Build failed!"
    exit 1
fi