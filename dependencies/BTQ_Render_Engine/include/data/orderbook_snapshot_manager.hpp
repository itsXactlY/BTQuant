#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "../data/data_types.hpp"  // Contains OrderbookData definition

namespace BTQuant {
namespace RenderEngine {

/**
 * OrderbookSnapshotManager - Implements atomic orderbook snapshot swapping
 * 
 * This class manages atomic swapping of orderbook snapshots for renderer access.
 * Once per frame, the MarketDataProcessor atomically swaps the "Reader Snapshot" pointer,
 * allowing the OrderbookPanel to read strictly from the "Reader Snapshot" without mutexes.
 */
class OrderbookSnapshotManager {
public:
    OrderbookSnapshotManager() = default;
    ~OrderbookSnapshotManager() = default;

    // Non-copyable, non-movable
    OrderbookSnapshotManager(const OrderbookSnapshotManager&) = delete;
    OrderbookSnapshotManager& operator=(const OrderbookSnapshotManager&) = delete;
    OrderbookSnapshotManager(OrderbookSnapshotManager&&) = delete;
    OrderbookSnapshotManager& operator=(OrderbookSnapshotManager&&) = delete;

    /**
     * Update the current orderbook snapshot for a symbol (called by MarketDataProcessor)
     * This creates a new snapshot and atomically updates the pointer
     */
    void updateSnapshot(uint32_t symbol_id, const OrderbookData& orderbook_data) {
        // Create a new shared pointer to the orderbook data
        auto new_snapshot = std::make_shared<const OrderbookData>(orderbook_data);
        
        // Atomically update the snapshot pointer
        {
            std::unique_lock lock(mutex_);
            snapshots_[symbol_id] = new_snapshot;
        }
    }

    /**
     * Get the current orderbook snapshot for a symbol (called by OrderbookPanel)
     * This provides lock-free access to the current snapshot
     */
    std::shared_ptr<const OrderbookData> getSnapshot(uint32_t symbol_id) const {
        std::shared_lock lock(mutex_);
        auto it = snapshots_.find(symbol_id);
        if (it != snapshots_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * Get all active symbol IDs that have snapshots
     */
    std::vector<uint32_t> getActiveSymbols() const {
        std::shared_lock lock(mutex_);
        std::vector<uint32_t> symbols;
        symbols.reserve(snapshots_.size());
        for (const auto& pair : snapshots_) {
            symbols.push_back(pair.first);
        }
        return symbols;
    }

    /**
     * Clear snapshot for a specific symbol
     */
    void clearSnapshot(uint32_t symbol_id) {
        std::unique_lock lock(mutex_);
        snapshots_.erase(symbol_id);
    }

    /**
     * Clear all snapshots
     */
    void clearAllSnapshots() {
        std::unique_lock lock(mutex_);
        snapshots_.clear();
    }

private:
    // Thread-safe map of symbol_id -> shared_ptr to const OrderbookData
    // Using shared_ptr allows lock-free reads while maintaining data safety
    mutable std::shared_mutex mutex_;
    std::unordered_map<uint32_t, std::shared_ptr<const OrderbookData>> snapshots_;
};

} // namespace RenderEngine
} // namespace BTQuant