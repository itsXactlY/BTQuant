#!/bin/bash

# Script to run the MarketDataProcessor benchmark

echo "Running MarketDataProcessor 1M Messages/Sec Benchmark..."

# Build the project if needed
if [ ! -f "build_benchmark/final_benchmark_market_data_processor" ]; then
    echo "Building benchmark..."
    mkdir -p build_benchmark
    cd build_benchmark
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ..
fi

# Run the benchmark
echo "Executing benchmark..."
./build_benchmark/final_benchmark_market_data_processor

echo "Benchmark completed."