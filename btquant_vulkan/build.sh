#!/bin/bash

# btquant_vulkan Build Script

echo "Building btquant_vulkan..."

# Create build directory if it doesn't exist
mkdir -p build

# Configure with CMake
echo "Configuring with CMake..."
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
cmake --build build -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Executable location: build/btquant_vulkan"
    
    # Optionally run the tests
    echo "Running integration tests..."
    cd build && ./test/test_integration || echo "  (test failed — see output above)"
    cd ..
else
    echo "Build failed!"
    exit 1
fi

echo "To run the application:"
echo "  cd build && ./btquant_vulkan"