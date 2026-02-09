# Soak Test for BTQ Render Engine

## Overview
This soak test validates the BTQ Render Engine's ability to maintain UI responsiveness under extreme load conditions. The test simulates 1 million market data events per second to stress-test the system.

## Purpose
- **Validate Performance**: Ensure the UI remains interactive (mouse hover, button clicks) under high load
- **Memory Stability**: Verify no memory leaks during extended operation
- **Thread Safety**: Confirm thread-safe operation under concurrent load
- **Latency Testing**: Measure response times under sustained load

## Test Parameters
- **Target Throughput**: 1,000,000 events per second
- **Event Characteristics**: 
  - Simulated market data processing
  - High-frequency event generation
  - Sequential timestamps
- **Test Duration**: Continuous until manually stopped

## Expected Results
- UI remains responsive (mouse hover works, buttons click instantly)
- No significant memory growth over time
- Consistent event processing rates
- No crashes or hangs

## How to Run
```bash
# Method 1: Using the script
./scripts/run_soak_test.sh

# Method 2: Building and running manually
cd market_data_collector
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) soak_test
./soak_test
```

## Integration with BTQ Render Engine
To run the soak test with the actual BTQ Render Engine:
1. Build the full application: `./dependencies/BTQ_Render_Engine/build_integration.sh`
2. Run the main application: `./dependencies/BTQ_Render_Engine/build/BTQuantTerminal`
3. Run this soak test in parallel: `./market_data_collector/build/soak_test`
4. Verify UI remains responsive (mouse hover, button clicks work instantly)

## Implementation Details
The soak test generates batches of 1000 events every millisecond to achieve the target rate of 1M events/sec. Each event simulates realistic market data processing. The test is designed to work with the HotSpineDataBridge when the full system is operational.

## Monitoring
The test outputs periodic statistics showing:
- Total events sent
- Current throughput rate
- Duration of the test

## Pass Criteria
- UI remains responsive throughout the test
- Consistent event processing rate near 1M events/sec
- No memory leaks or crashes during extended operation