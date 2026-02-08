# RCU for Configuration (`<rcu>` C++26) Implementation Guide

## Overview
This document describes the implementation of RCU (Read-Copy-Update) for configuration management using the upcoming C++26 standard. The implementation provides zero-contention access for readers while allowing safe updates to configuration data structures.

## C++26 RCU Standard Interface

The C++26 standard introduces the following RCU interface in the `<rcu>` header:

```cpp
#include <rcu>

namespace std {
    template<typename T>
    class rcu_obj_base;  // Base class for RCU-protected objects
    
    template<typename Key, typename Hash = hash<Key>, typename Pred = equal_to<Key>>
    using rcu_unordered_set = rcu_obj_base<unordered_set<Key, Hash, Pred>>;
    
    template<typename Key, typename Value, typename Hash = hash<Key>, typename Pred = equal_to<Key>>
    using rcu_unordered_map = rcu_obj_base<unordered_map<Key, Value, Hash, Pred>>;
    
    template<typename T>
    [[nodiscard]] rcu_reader<T> rcu_read_lock(const rcu_obj_base<T>& obj);
    
    void synchronize_rcu();
    template<typename Func>
    void synchronize_rcu(Func&& func);
}
```

## Current Implementation Status

The current implementation in `BTQ_Render_Engine/include/rcu.hpp` provides a forward-compatible interface that matches the expected C++26 standard. When C++26 becomes available, the implementation can be seamlessly upgraded.

## Usage Examples

### 1. Basic Usage with Configuration Maps

```cpp
#include "rcu.hpp"

// Create an RCU-protected configuration map
BTQ::rcu_unordered_map<std::string, std::string> config_map;

// Thread-safe insertion/update (writer)
config_map.insert_or_assign("database.host", "localhost");
config_map.insert_or_assign("database.port", "5432");

// Zero-contention read access (reader)
auto guard = config_map.rcu_read_lock();
auto host = config_map.get("database.host");  // Returns std::optional
if (host) {
    std::cout << "Database host: " << *host << std::endl;
}
```

### 2. Active Pairs Management in Market Data Processor

```cpp
#include "rcu.hpp"

// RCU-protected set of active trading pairs
BTQ::rcu_unordered_set<std::string> active_pairs;

// Add a new pair (writer thread)
active_pairs.insert("BTC:USD:spot");

// Check for active pairs (reader thread - zero contention)
auto guard = active_pairs.rcu_read_lock();
bool is_active = active_pairs.contains("BTC:USD:spot");
```

### 3. Complex Configuration Updates

```cpp
// Perform complex updates atomically
config_map.update([](auto& map) {
    map["param1"] = "value1";
    map["param2"] = "value2";
    // All changes are applied atomically to a copy
    // Readers continue to see the old version until update completes
});
```

## Thread Safety Guarantees

1. **Readers never block**: Multiple reader threads can access the data concurrently without blocking each other
2. **Updates are atomic**: Writers create a copy of the data, modify it, then atomically swap the pointers
3. **Memory safety**: Old data is only reclaimed after all readers have finished with it
4. **Consistency**: Readers always see a consistent snapshot of the data

## Performance Characteristics

- **Read operations**: O(1) with zero contention between readers
- **Write operations**: O(n) where n is the size of the container (due to copying)
- **Memory overhead**: Temporary doubling of memory during updates
- **Latency**: Minimal impact on readers during updates

## Migration Path to C++26

When C++26 becomes available with native RCU support:

1. Replace `#include "rcu.hpp"` with `#include <rcu>`
2. Change namespace from `BTQ::` to `std::`
3. The API remains identical, ensuring seamless migration

## Integration Points

### MarketDataProcessor
- `active_pairs_` member uses `BTQ::rcu_unordered_set<std::string>`
- Updated in `handleTradeMessage` without blocking readers
- Used for tracking active trading pairs across exchanges

### ConfigLoader
- `config_` member uses `BTQ::rcu_unordered_map<std::string, ConfigSection>`
- `source_map_` member uses `BTQ::rcu_unordered_map<std::string, ConfigSource>`
- Provides thread-safe access to application configuration

## Best Practices

1. **Keep read-side critical sections short**: Hold the read lock only as long as necessary
2. **Minimize write frequency**: RCU is optimized for read-heavy workloads
3. **Avoid recursive locking**: Don't call functions that might acquire RCU locks while holding one
4. **Consider memory usage**: Large containers will consume more memory during updates due to copying

## Testing Strategy

Comprehensive tests are provided in `tests/rcu_test.cpp` covering:
- Basic functionality
- Concurrent access patterns
- Configuration map operations
- Stress testing with multiple readers and writers

## Known Limitations

1. Current implementation simulates grace periods using `std::this_thread::yield()`
2. Memory reclamation timing may differ from true RCU implementations
3. Performance characteristics may vary when native C++26 RCU becomes available

## Future Enhancements

1. Integration with hardware-assisted RCU when available
2. Support for custom memory allocators
3. Additional container types beyond unordered_set and unordered_map