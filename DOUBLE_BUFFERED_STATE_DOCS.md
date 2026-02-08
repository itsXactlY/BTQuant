# Double Buffered State Implementation

## Overview

The `DoubleBufferedState<T>` class implements a lock-free double buffering mechanism that enables safe concurrent access to shared state between reader and writer threads. This design eliminates contention by allowing:

- Writer threads to update the "back" buffer without blocking readers
- Reader threads to access the "front" buffer without blocking writers  
- Atomic swapping of buffers when updates are complete

## Key Features

### Thread Safety
- Zero-contention reads: Multiple reader threads can access the front buffer simultaneously
- Lock-free design: Uses atomic operations for synchronization
- Memory safety: Automatic memory management with RAII

### Performance
- O(1) read and write operations
- Minimal synchronization overhead
- Efficient memory usage

### API
- `read()`: Access current state for reading (non-modifying)
- `write()`: Access back buffer for writing
- `swap()`: Atomically swap front and back buffers
- `update_and_swap()`: Update back buffer and swap in one operation
- `modify_and_swap()`: Apply function to back buffer and swap
- `read_with()`: Perform atomic read operation with callback

## Usage Example

```cpp
#include "threading/double_buffered_state.hpp"
#include <iostream>
#include <thread>

using namespace btq::threading;

int main() {
    // Create a double buffered state for integer values
    DoubleBufferedState<int> counter(0);
    
    // Writer thread
    std::thread writer([&counter]() {
        for (int i = 1; i <= 100; ++i) {
            counter.write() = i;  // Update back buffer
            counter.swap();       // Atomically make it visible to readers
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    // Reader thread
    std::thread reader([&counter]() {
        for (int i = 0; i < 50; ++i) {
            int current_value = counter.read();  // Read current state
            std::cout << "Current value: " << current_value << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    
    writer.join();
    reader.join();
    
    return 0;
}
```

## Implementation Details

The implementation uses atomic pointers to achieve lock-free access:

- Two buffers (front and back) store the state
- Atomic pointers reference the current front and back buffers
- Swapping involves exchanging the atomic pointers
- Copy and move operations are supported with proper synchronization

## Use Cases in Trading Systems

This implementation is particularly valuable for trading systems where:

- Market data updates arrive rapidly (writer thread)
- Multiple UI components need to access current state (reader threads)
- Consistency is important but blocking is unacceptable
- Real-time analytics need to access stable snapshots of data

## Thread Safety Guarantees

- Writers cannot interfere with readers
- Readers see consistent snapshots of data
- Multiple readers can access simultaneously
- Buffer swapping is atomic and safe

## Limitations

- Writers must coordinate among themselves (not thread-safe for multiple writers)
- Memory overhead is approximately 2x the size of T
- Updates involve copying the entire state

## Memory Ordering

The implementation uses appropriate memory ordering:
- `memory_order_acquire` for loads
- `memory_order_release` for stores
- `memory_order_acq_rel` for exchanges