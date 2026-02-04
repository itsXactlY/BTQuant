#!/bin/bash

# Build script for PubBTQuant examples

echo "Building PubBTQuant examples..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run cmake
echo "Running cmake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build all examples
echo "Building examples..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "You can run the examples from the build directory:"
    echo "  ./volume_analysis_example"
    echo "  ./indicator_calculations_example"
    echo "  ./data_processing_example"
    echo "  ./correlation_analytics_example"
else
    echo "Build failed!"
    exit 1
fi