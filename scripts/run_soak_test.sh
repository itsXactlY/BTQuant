#!/bin/bash

# Soak Test Runner Script
# Runs the soak test to simulate 1 million events per second
# and verifies UI remains responsive

echo "Starting Soak Test for BTQ Render Engine..."
echo "This test will simulate 1 million events per second"
echo "to verify UI remains responsive under high load."

# Build the soak test
echo "Building soak test..."
cd /home/alca/projects/PubBTQuant/market_data_collector
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) soak_test

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully."

# Run the soak test in the background
echo "Starting soak test (1M events/sec)..."
./soak_test &
SOAK_TEST_PID=$!

echo "Soak test started with PID: $SOAK_TEST_PID"

# Give it a moment to start
sleep 2

# Monitor the soak test
echo "Monitoring soak test performance..."
echo "Press Ctrl+C to stop the test and monitoring."

# Trap SIGINT to handle cleanup
trap 'echo -e "\nStopping soak test..."; kill $SOAK_TEST_PID 2>/dev/null; wait $SOAK_TEST_PID 2>/dev/null; echo "Soak test stopped."; exit 0' INT TERM

# Wait for the soak test to complete (or be interrupted)
wait $SOAK_TEST_PID

echo "Soak test completed."