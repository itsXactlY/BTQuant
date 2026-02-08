#!/bin/bash

# Build and run the MarketDataProcessor benchmark

echo "Building MarketDataProcessor benchmark..."

# Create build directory if it doesn't exist
mkdir -p build_benchmark

# Navigate to build directory
cd build_benchmark

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the benchmark
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"

# Run the benchmark
echo "Running benchmark..."
./bin/benchmark_market_data_processor

if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully!"
else
    echo "Benchmark failed!"
    exit 1
fi