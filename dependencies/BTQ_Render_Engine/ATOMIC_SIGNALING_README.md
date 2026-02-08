# Atomic Signaling Implementation (C++20/26)

## Overview

This implementation provides lock-free signaling mechanisms that replace traditional condition variables with atomic wait/notify operations for improved performance in high-frequency scenarios. The solution addresses the requirements from the BTQ Render Engine project to eliminate `std::mutex`, `std::shared_mutex`, and `std::condition_variable` usage in the hot path.

## Components

### 1. AtomicSignal Class
- Uses `std::atomic<uint32_t>` as the underlying signaling mechanism
- Provides `notify_one()` and `notify_all()` methods equivalent to condition variables
- Implements `wait()`, `wait_for()`, and `wait_until()` methods
- Supports predicate-based waiting

### 2. AtomicBooleanSignal Class
- Optimized for simple true/false signaling scenarios
- Uses `std::atomic<bool>` as the underlying type
- More efficient for binary state signaling

### 3. AtomicCounterSignal Class
- Designed for counting-based signaling
- Useful when waiting for a specific number of events
- Tracks both current count and target value

## Benefits

- **Eliminated Condition Variable Overhead**: Replaced kernel-level locking with atomic operations
- **Improved Performance**: Reduced context switches and system call overhead
- **Better Scalability**: More efficient in high-frequency signaling scenarios
- **Zero-Lock Design**: Part of the broader zero-lock modernization initiative

## Integration with Task Scheduler

The implementation has been integrated with the existing TaskScheduler to replace condition variables with atomic signals:

- Replaced `std::condition_variable` with `btq::threading::AtomicSignal`
- Updated worker loop to use atomic signaling for task availability
- Maintained thread safety without traditional mutex-based synchronization
- Preserved all existing functionality while improving performance

## Usage Examples

```cpp
#include "threading/atomic_signal.hpp"

using namespace btq::threading;

// Basic signaling
AtomicSignal signal;
std::thread producer([&signal]() {
    // Do work
    signal.notify_one();  // Signal completion
});

std::thread consumer([&signal]() {
    signal.wait();  // Wait for signal
    // Process data
});

// Boolean signaling
AtomicBooleanSignal bool_signal;
// ... use signal.signal() and bool_signal.wait()

// Counter-based signaling
AtomicCounterSignal counter_signal(3);  // Wait for 3 events
// ... use counter_signal.increment() and counter_signal.wait_for_target()
```

## Performance Characteristics

- **Low Latency**: Atomic operations have significantly lower overhead than system calls
- **High Throughput**: Better performance in high-frequency signaling scenarios
- **Scalability**: Maintains performance under high thread contention
- **Memory Efficiency**: Reduced memory footprint compared to condition variable implementations

## Compatibility

- Requires C++20 or later for atomic wait/notify support
- Compatible with existing threading patterns
- Drop-in replacement for many condition variable use cases