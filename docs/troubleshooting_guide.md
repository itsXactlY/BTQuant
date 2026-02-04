# Troubleshooting Guide

This guide provides solutions to common issues, debugging techniques, and troubleshooting steps for the PubBTQuant platform. This document covers build problems, runtime issues, performance bottlenecks, graphics problems, threading issues, and data connectivity problems.

## Table of Contents
- [Build Issues](#build-issues)
- [Runtime Issues](#runtime-issues)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Graphics and Vulkan Issues](#graphics-and-vulkan-issues)
- [Threading and Concurrency Debugging](#threading-and-concurrency-debugging)
- [Data Bridge and Connectivity](#data-bridge-and-connectivity)
- [Debugging Techniques](#debugging-techniques)
- [Common Solutions Summary](#common-solutions-summary)

## Build Issues

### Missing Dependencies
**Problem**: Build fails due to missing system dependencies.

**Solution**:
1. Install required packages:
   ```bash
   # On Ubuntu/Debian systems
   sudo apt update
   sudo apt install build-essential cmake vulkan-sdk libglfw3-dev libvulkan-dev libtbb-dev git
   
   # Install modern GCC that supports C++26
   sudo apt install gcc-13 g++-13
   ```

2. Verify Vulkan installation:
   ```bash
   vulkaninfo
   ```

### C++26 Support Issues
**Problem**: Compiler errors related to C++26 features.

**Solution**:
1. Ensure you're using a compatible compiler:
   ```bash
   g++ --version  # Should be GCC 13 or newer
   ```
   
2. Explicitly specify the compiler during CMake configuration:
   ```bash
   cmake .. -DCMAKE_CXX_COMPILER=g++-13
   ```

### CMake Configuration Errors
**Problem**: CMake fails to configure the project.

**Solution**:
1. Clean previous build attempts:
   ```bash
   rm -rf build/
   mkdir build && cd build
   ```
   
2. Reconfigure with verbose output:
   ```bash
   cmake .. -DCMAKE_CXX_COMPILER=g++-13 --debug-output
   ```

### Linker Errors
**Problem**: Linker fails with undefined references.

**Solution**:
1. Ensure all dependencies are properly linked in CMakeLists.txt
2. Check that TBB libraries are correctly found:
   ```bash
   pkg-config --libs tbb
   ```
3. Verify Vulkan SDK is properly installed and linked

### Missing Vulkan Headers
**Problem**: Compilation fails with Vulkan header errors.

**Solution**:
1. Install Vulkan SDK:
   ```bash
   # On Ubuntu
   sudo apt install vulkan-sdk
   ```
   
2. Or install headers separately:
   ```bash
   sudo apt install libvulkan-dev spirv-tools
   ```

## Runtime Issues

### Application Crashes at Startup
**Problem**: The application crashes immediately upon launching.

**Diagnosis Steps**:
1. Run with debug output:
   ```bash
   gdb ./realtime_dashboard
   (gdb) run
   ```
   
2. Check for missing shared libraries:
   ```bash
   ldd ./realtime_dashboard
   ```

**Solutions**:
1. Install missing libraries identified by `ldd`
2. Set proper library path if needed:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```
3. Check Vulkan compatibility with your GPU:
   ```bash
   vulkaninfo | grep -i device
   ```

### Memory Allocation Failures
**Problem**: Application crashes with out-of-memory errors.

**Solutions**:
1. Monitor memory usage during startup:
   ```bash
   htop  # Look for memory consumption
   ```
   
2. Increase system memory limits:
   ```bash
   ulimit -v unlimited  # Virtual memory
   ulimit -m unlimited  # Resident set size
   ```

3. Check for memory leaks in initialization code
4. Verify memory pool configurations

### UI Not Responding
**Problem**: Application appears frozen but doesn't crash.

**Solutions**:
1. Check for deadlocks in threading code
2. Verify that main UI thread isn't blocked by long operations
3. Use threading analysis tools:
   ```bash
   valgrind --tool=helgrind ./realtime_dashboard
   ```

### Panel Loading Failures
**Problem**: Specific panels fail to load or display incorrectly.

**Solutions**:
1. Check panel-specific logs in application output
2. Verify panel configuration files exist and are readable
3. Reset to default layout:
   ```bash
   # Backup current layout
   cp current_layout.json current_layout.json.bak
   # Copy default layout
   cp default_layout.json current_layout.json
   ```

## Performance Troubleshooting

### High CPU Usage
**Symptoms**: CPU utilization consistently above 80%

**Diagnosis**:
1. Profile the application:
   ```bash
   perf record -g ./realtime_dashboard
   perf report
   ```
   
2. Use built-in profiler if available:
   - Access performance monitor through UI
   - Check frame timing graphs

**Solutions**:
1. Reduce polling frequency for data updates
2. Implement level-of-detail (LOD) for large datasets
3. Optimize rendering loops
4. Check for busy-wait loops
5. Use TBB for parallel processing where appropriate

### Memory Leaks
**Symptoms**: Gradually increasing memory usage over time

**Diagnosis**:
1. Use memory profiling tools:
   ```bash
   valgrind --tool=memcheck --leak-check=full ./realtime_dashboard
   ```
   
2. Monitor memory usage with:
   ```bash
   watch -n 1 'ps aux | grep realtime_dashboard'
   ```

**Solutions**:
1. Fix resource leaks in destructors
2. Verify RAII principles are followed
3. Check for circular references in data structures
4. Use smart pointers appropriately

### Rendering Performance Issues
**Symptoms**: Low FPS, stuttering, or inconsistent frame rates

**Diagnosis**:
1. Enable frame timing in application
2. Check Vulkan validation layers output
3. Monitor GPU usage

**Solutions**:
1. Implement frustum culling for off-screen elements
2. Reduce polygon count for distant objects
3. Use instanced rendering for similar objects
4. Optimize shader complexity
5. Implement texture atlasing

### Data Processing Bottlenecks
**Symptoms**: Delayed market data updates, lag in analytics

**Solutions**:
1. Optimize data pipeline algorithms
2. Use lock-free queues for inter-thread communication
3. Implement data batching where appropriate
4. Profile data processing functions
5. Consider SIMD optimizations for numerical computations

## Graphics and Vulkan Issues

### Vulkan Validation Layer Errors
**Problem**: Application exits with Vulkan validation errors.

**Solutions**:
1. Enable/disable validation layers based on environment:
   - Development: Enable all validation layers
   - Production: Disable validation layers for performance

2. Common fixes:
   - Ensure proper queue family indices
   - Verify descriptor set layouts match shader expectations
   - Check buffer memory alignment requirements

### GPU Compatibility Issues
**Problem**: Application fails to initialize graphics on certain GPUs.

**Solutions**:
1. Check GPU Vulkan support:
   ```bash
   vulkaninfo | grep -A 10 -B 10 "deviceName\|apiVersion"
   ```
   
2. Implement fallback graphics paths
3. Verify required GPU features are supported:
   - Compute shaders
   - Geometry shaders (if used)
   - Required extensions

### Shader Compilation Failures
**Problem**: Shaders fail to compile or behave unexpectedly.

**Solutions**:
1. Verify SPIR-V compilation:
   ```bash
   glslc shader.vert -o vert.spv  # Compile shader
   spirv-val vert.spv             # Validate SPIR-V
   ```
   
2. Check shader compatibility with target Vulkan version
3. Use shader reflection to verify bindings match

### Rendering Artifacts
**Problem**: Visual glitches, incorrect colors, or missing geometry.

**Solutions**:
1. Enable Vulkan validation layers for diagnostic output
2. Check depth buffer configuration
3. Verify viewport and scissor rectangle settings
4. Ensure proper synchronization between graphics/compute queues

## Threading and Concurrency Debugging

### Deadlock Detection
**Problem**: Application freezes due to thread deadlock.

**Diagnosis**:
1. Use thread analysis tools:
   ```bash
   valgrind --tool=helgrind ./realtime_dashboard
   # or
   valgrind --tool=drd ./realtime_dashboard
   ```
   
2. Generate thread dump if possible:
   ```bash
   kill -USR2 <pid>  # If signal handler is implemented
   ```

**Solutions**:
1. Implement lock ordering discipline
2. Use lock-free data structures where possible
3. Add timeout mechanisms to mutex locks
4. Use RAII lock guards consistently

### Race Conditions
**Problem**: Unpredictable behavior or crashes that occur intermittently.

**Solutions**:
1. Use atomic operations for simple shared variables
2. Implement proper synchronization primitives
3. Use thread sanitizer during development:
   ```bash
   g++ -fsanitize=thread -g source.cpp -o test
   ```
   
4. Add memory barriers where necessary
5. Use immutable data structures when possible

### Thread Starvation
**Problem**: Some threads receive insufficient CPU time.

**Solutions**:
1. Adjust thread priorities appropriately
2. Check for overly aggressive spin loops
3. Implement fair queuing mechanisms
4. Monitor thread scheduling behavior

### Lock-Free Queue Issues
**Problem**: Data corruption or lost messages in lock-free queues.

**Solutions**:
1. Verify ABA problem protection if using CAS operations
2. Check memory ordering requirements
3. Use established lock-free queue implementations (like moodycamel)
4. Add extensive testing with multiple producer/consumer scenarios

## Data Bridge and Connectivity

### Connection Failures
**Problem**: Unable to connect to market data feeds.

**Diagnosis**:
1. Check network connectivity:
   ```bash
   ping <data_feed_server>
   telnet <data_feed_server> <port>
   ```
   
2. Verify connection parameters in configuration files

**Solutions**:
1. Check firewall settings
2. Verify API credentials and permissions
3. Implement retry logic with exponential backoff
4. Use connection health checks

### Data Synchronization Issues
**Problem**: Data inconsistencies between different panels or delayed updates.

**Solutions**:
1. Implement proper data versioning
2. Use sequence numbers for data integrity
3. Add data staleness checks
4. Implement data reconciliation mechanisms

### High-Frequency Data Overload
**Problem**: Application becomes unresponsive under high data throughput.

**Solutions**:
1. Implement data sampling/downsampling
2. Use ring buffers for temporary data storage
3. Prioritize critical data updates
4. Implement flow control mechanisms

### Data Type Conversion Errors
**Problem**: Incorrect data interpretation or conversion failures.

**Solutions**:
1. Add comprehensive data validation
2. Implement proper error handling for conversion failures
3. Use strongly-typed data structures
4. Add unit tests for data conversion functions

## Debugging Techniques

### Logging Strategy
**Best Practices**:
1. Use structured logging with severity levels
2. Include timestamps and thread IDs
3. Log entry/exit of critical functions
4. Implement configurable log levels

**Tools**:
- spdlog for high-performance logging
- Custom logging macros for debugging sections

### Profiling Methods
**CPU Profiling**:
```bash
# Using perf
perf record -g -F 997 ./realtime_dashboard
perf script | stackcollapse-perf.pl | flamegraph.pl > perf.svg

# Using built-in profiler
# Access through application UI or enable via command line flag
```

**Memory Profiling**:
```bash
# Valgrind Memcheck
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./realtime_dashboard

# Address Sanitizer
g++ -fsanitize=address -g source.cpp -o debug_build
./debug_build
```

### Debugging Multi-threaded Code
**Techniques**:
1. Use thread-local storage for debugging information
2. Implement thread-safe logging
3. Use conditional breakpoints based on thread ID
4. Capture thread state dumps

**Tools**:
- GDB with thread debugging: `info threads`, `thread <n>`
- rr (record and replay debugger) for deterministic debugging
- Custom thread monitoring utilities

### Shader Debugging
**Techniques**:
1. Use RenderDoc for frame capture and analysis
2. Implement debug visualization modes
3. Add shader validation during development
4. Use fragment shader to output intermediate values

### Memory Pool Debugging
**Techniques**:
1. Add guard bytes to detect buffer overruns
2. Implement double-free detection
3. Track allocation statistics
4. Add memory pool validation functions

## Common Solutions Summary

### Quick Fixes
1. **Application won't start**: Check Vulkan installation and GPU compatibility
2. **High CPU usage**: Enable LOD, reduce polling frequency, profile with perf
3. **Memory issues**: Use Valgrind, implement RAII, check for leaks
4. **Threading problems**: Use thread sanitizers, implement proper synchronization
5. **Data connectivity**: Verify network settings, check credentials, implement retries

### Preventive Measures
1. Regular code reviews focusing on resource management
2. Automated testing for performance regressions
3. Continuous integration with static analysis
4. Comprehensive logging and monitoring
5. Regular profiling and optimization cycles

### Useful Commands
```bash
# System monitoring
htop                    # CPU/memory usage
nvidia-smi             # GPU status (if NVIDIA)
iotop                  # I/O monitoring

# Debugging
gdb ./realtime_dashboard
valgrind --tool=memcheck ./realtime_dashboard
perf top -p $(pgrep realtime_dashboard)

# Vulkan debugging
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./realtime_dashboard
```

### Getting Further Help
- Check existing documentation in the `docs/` directory
- Review PRDs and implementation summaries
- Examine test files for working examples
- Use the built-in diagnostic tools when available
- Contact the development team for complex architectural issues
```