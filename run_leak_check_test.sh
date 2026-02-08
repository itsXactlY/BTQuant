#!/bin/bash

echo "Running Hazard Pointer 4-Hour Window Leak Check Test..."
echo "====================================================="

# Compile the test
echo "Compiling test..."
g++ -std=c++20 -pthread -I. -o test_hazard_pointer_4hour_window test_hazard_pointer_4hour_window.cpp

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run the test
echo "Running test..."
./test_hazard_pointer_4hour_window

if [ $? -eq 0 ]; then
    echo ""
    echo "TEST PASSED: Hazard Pointers correctly reclaim memory after the 4-hour window moves."
else
    echo ""
    echo "TEST FAILED: Memory leak detected in hazard pointer reclamation mechanism."
    exit 1
fi