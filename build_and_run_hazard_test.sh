#!/bin/bash

# Build and run the hazard pointer leak check test

echo "Building hazard pointer leak check test..."

# Compile the test
g++ -std=c++20 -pthread -I. -o test_hazard_pointer_leak_check test_hazard_pointer_leak_check.cpp

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running hazard pointer leak check test..."
    ./test_hazard_pointer_leak_check
else
    echo "Build failed!"
    exit 1
fi