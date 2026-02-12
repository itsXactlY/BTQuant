#pragma once

#include <atomic>
#include <vector>
#include <cstdint>

// Atomic Symbol Information Structure (for lock-free access)
struct alignas(64) AtomicSymbolInfo {
    std::atomic<double> price{0.0};
    std::atomic<double> volume_24h{0.0};
    std::atomic<double> change_24h{0.0};
    std::atomic<double> high_24h{0.0};
    std::atomic<double> low_24h{0.0};
    std::atomic<uint64_t> last_update_ts{0};
};

class MarketDataProcessor {
public:
    static constexpr size_t MAX_SYMBOLS = 100000; // Pre-allocated to 100,000 slots

    MarketDataProcessor() : atomic_store_(MAX_SYMBOLS) {}

    // Zero-lock access method to get atomic snapshot
    const AtomicSymbolInfo* get_atomic_snapshot(uint32_t symbol_id) const {
        if (symbol_id >= MAX_SYMBOLS) {
            return nullptr; // Out of bounds check
        }
        // Return a pointer to the atomic struct which can be accessed lock-free
        // The atomic operations themselves provide thread safety
        return &(atomic_store_[symbol_id]);
    }

    // Method to update atomic data (for writers)
    AtomicSymbolInfo* get_mutable_atomic_snapshot(uint32_t symbol_id) {
        if (symbol_id >= MAX_SYMBOLS) {
            return nullptr; // Out of bounds check
        }
        return &(atomic_store_[symbol_id]);
    }

private:
    std::vector<AtomicSymbolInfo> atomic_store_;
};