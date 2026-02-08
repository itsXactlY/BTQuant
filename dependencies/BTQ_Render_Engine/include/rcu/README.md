# RCU for Configuration (`<rcu>` C++26) Implementation

## Overview
This implementation provides a simulated version of the upcoming C++26 RCU (Read-Copy-Update) functionality for configuration management. The implementation follows the specification mentioned in the hmm.md file:

- Use `std::rcu_obj_base` for `active_pairs_` and configuration maps
- Readers access data via `std::rcu_read_lock`
- Updates happen via `synchronize_rcu()`, ensuring zero contention for readers

## Components

### 1. RCU Wrapper Header (`rcu_config_wrapper.hpp`)
- Provides `rcu_obj_base` template class with specializations for `unordered_set` and `unordered_map`
- Includes convenience aliases: `rcu_unordered_set` and `rcu_unordered_map`
- Implements copy-on-write semantics for thread-safe updates
- Provides read locks for contention-free reading

### 2. MarketDataProcessor Integration
- Updated `active_pairs_` member to use `BTQ::rcu_unordered_set<std::string>`
- Modified `handleTradeMessage` to use RCU for inserting pairs without blocking readers
- Removed mutex protection for the active_pairs_ access in the hot path

### 3. ConfigLoader Integration
- Updated `config_` and `source_map_` members to use RCU-enabled containers
- Modified all access methods to use RCU read locks and update operations
- Ensured thread-safe configuration loading and updates

## Benefits

1. **Zero Reader Contention**: Readers can access configuration data without blocking
2. **Safe Updates**: Updates use copy-on-write semantics to prevent corruption
3. **Performance**: Reduced lock contention in hot paths
4. **Future-Proof**: Interface matches the expected C++26 RCU API

## Usage Example

```cpp
// Creating an RCU-enabled container
BTQ::rcu_unordered_set<std::string> active_pairs;

// Reading (no locks, zero contention)
auto guard = active_pairs.read_lock();
bool exists = guard->contains("BTC:USD:spot");

// Updating (copy-on-write)
active_pairs.insert("ETH:USD:spot");
```

## Testing
The implementation includes comprehensive tests covering:
- Basic functionality
- Concurrent access patterns
- Configuration map operations

Run tests with: `./tests/rcu_test`

## Future Migration
When C++26 RCU becomes available, migration will involve:
1. Replacing `BTQ::rcu_*` types with `std::rcu_*` equivalents
2. Using actual `std::rcu_read_lock` and `std::synchronize_rcu`
3. Leveraging hardware-optimized RCU implementations