#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "market_data_processor.hpp"

namespace BTQuant {
namespace RenderEngine {

/**
 * AtomicOrderbookPanel - Implements lock-free orderbook rendering using atomic snapshots
 * 
 * This panel reads strictly from the "Reader Snapshot" provided by the MarketDataProcessor
 * without using mutexes during render, implementing the double-buffering requirement.
 */
class AtomicOrderbookPanel {
public:
    AtomicOrderbookPanel(std::shared_ptr<MarketDataProcessor> processor)
        : market_data_processor_(processor) {}

    ~AtomicOrderbookPanel() = default;

    // Non-copyable, non-movable
    AtomicOrderbookPanel(const AtomicOrderbookPanel&) = delete;
    AtomicOrderbookPanel& operator=(const AtomicOrderbookPanel&) = delete;
    AtomicOrderbookPanel(AtomicOrderbookPanel&&) = delete;
    AtomicOrderbookPanel& operator=(AtomicOrderbookPanel&&) = delete;

    /**
     * Render the orderbook using the atomic snapshot
     * This method reads strictly from the "Reader Snapshot" without mutexes
     */
    void render(uint32_t symbol_id) {
        // Get the atomic snapshot from the processor (lock-free access)
        auto snapshot = market_data_processor_->get_orderbook_snapshot(symbol_id);
        
        if (!snapshot) {
            // No snapshot available, render empty orderbook or placeholder
            renderEmptyOrderbook();
            return;
        }

        // Render the orderbook using the atomic snapshot data
        renderFromSnapshot(*snapshot);
    }

private:
    std::shared_ptr<MarketDataProcessor> market_data_processor_;

    void renderEmptyOrderbook() {
        // Render placeholder when no snapshot is available
    }

    void renderFromSnapshot(const OrderbookData& snapshot) {
        // Render the orderbook using the snapshot data
        // This method should implement the actual rendering logic
        
        // Example rendering logic (would be implemented with ImGui/Vulkan):
        // 1. Render bids (typically on the left/top)
        // 2. Render asks (typically on the right/bottom)
        // 3. Show spread between best bid and ask
        // 4. Apply heatmap coloring based on volume
        // 5. Highlight large orders
    }
};

} // namespace RenderEngine
} // namespace BTQuant