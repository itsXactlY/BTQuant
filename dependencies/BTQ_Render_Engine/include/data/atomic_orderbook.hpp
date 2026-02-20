#pragma once

/**
 * @file atomic_orderbook.hpp
 * @brief Lock-Free Double-Buffered OrderBook for Ultra-Low-Latency Trading
 * 
 * This implementation provides a wait-free read path for the rendering thread
 * while allowing concurrent writes from the market data thread.
 * 
 * Key features:
 * - Double-buffered design: Writer updates inactive buffer, then swaps
 * - Atomic pointer swap for O(1) buffer switch
 * - Zero allocations on the hot path
 * - Cache-line aligned to prevent false sharing
 * - Sequence numbers for detecting stale reads
 * 
 * Architecture:
 *   [Buffer A] <-- Writer (inactive)
 *   [Buffer B] <-- Reader (active)
 *   
 *   Writer flow:
 *   1. Get inactive buffer
 *   2. Update with new data
 *   3. Atomic swap active/inactive pointers
 *   
 *   Reader flow:
 *   1. Load active buffer pointer (atomic)
 *   2. Read data (no locks needed)
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <chrono>
#include <algorithm>

namespace btq {
namespace data {

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Single price level in the order book
 */
struct alignas(16) OrderBookLevel {
    double price{0.0};           // Price at this level
    double quantity{0.0};        // Quantity available
    uint64_t order_count{0};     // Number of orders at this level
    
    OrderBookLevel() = default;
    OrderBookLevel(double p, double q, uint64_t c = 1)
        : price(p), quantity(q), order_count(c) {}
    
    bool operator<(const OrderBookLevel& other) const {
        return price < other.price;
    }
    
    bool operator>(const OrderBookLevel& other) const {
        return price > other.price;
    }
};

/**
 * @brief Order book side (bids or asks)
 */
struct OrderBookSide {
    static constexpr size_t MAX_LEVELS = 256;  // Maximum depth
    
    alignas(64) std::atomic<uint32_t> count{0};
    OrderBookLevel levels[MAX_LEVELS];
    
    OrderBookSide() {
        std::memset(levels, 0, sizeof(levels));
    }
    
    void clear() {
        count.store(0, std::memory_order_release);
    }
    
    uint32_t size() const {
        return count.load(std::memory_order_acquire);
    }
    
    const OrderBookLevel* begin() const {
        return levels;
    }
    
    const OrderBookLevel* end() const {
        return levels + size();
    }
    
    // Binary search for price level
    int32_t findLevel(double price) const {
        uint32_t n = size();
        int32_t left = 0, right = n - 1;
        
        while (left <= right) {
            int32_t mid = left + (right - left) / 2;
            if (levels[mid].price == price) {
                return mid;
            } else if (levels[mid].price < price) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;  // Not found
    }
};

/**
 * @brief Complete order book snapshot
 */
struct OrderBookSnapshot {
    alignas(64) OrderBookSide bids;   // Buy orders (sorted descending by price)
    alignas(64) OrderBookSide asks;   // Sell orders (sorted ascending by price)
    
    alignas(64) std::atomic<uint64_t> sequence{0};
    alignas(64) std::atomic<uint64_t> timestamp{0};
    alignas(64) std::atomic<bool> ready{false};
    
    std::atomic<double> best_bid{0.0};
    std::atomic<double> best_ask{0.0};
    std::atomic<double> spread{0.0};
    std::atomic<double> mid_price{0.0};
    
    void clear() {
        bids.clear();
        asks.clear();
        sequence.store(0, std::memory_order_release);
        timestamp.store(0, std::memory_order_release);
        ready.store(false, std::memory_order_release);
    }
    
    void updateDerived() {
        uint32_t bid_count = bids.size();
        uint32_t ask_count = asks.size();
        
        if (bid_count > 0 && ask_count > 0) {
            double bb = bids.levels[0].price;  // Best bid (highest)
            double ba = asks.levels[0].price;  // Best ask (lowest)
            
            best_bid.store(bb, std::memory_order_relaxed);
            best_ask.store(ba, std::memory_order_relaxed);
            spread.store(ba - bb, std::memory_order_relaxed);
            mid_price.store((bb + ba) / 2.0, std::memory_order_relaxed);
        }
    }
};

/**
 * @brief Double-buffered lock-free order book
 * 
 * This class maintains two order book snapshots. The writer (market data thread)
 * updates the inactive buffer, then atomically swaps the active pointer.
 * The reader (rendering thread) always reads from the active buffer.
 */
class DoubleBufferedOrderBook {
public:
    DoubleBufferedOrderBook() {
        buffers_[0] = new OrderBookSnapshot();
        buffers_[1] = new OrderBookSnapshot();
        active_buffer_.store(buffers_[0], std::memory_order_release);
        inactive_buffer_.store(buffers_[1], std::memory_order_release);
    }
    
    ~DoubleBufferedOrderBook() {
        delete buffers_[0];
        delete buffers_[1];
    }
    
    // Non-copyable, non-movable
    DoubleBufferedOrderBook(const DoubleBufferedOrderBook&) = delete;
    DoubleBufferedOrderBook& operator=(const DoubleBufferedOrderBook&) = delete;
    
    // =========================================================================
    // WRITER API (Market Data Thread)
    // =========================================================================
    
    /**
     * @brief Begin an update cycle - get the inactive buffer
     * @return Pointer to the inactive buffer for writing
     */
    OrderBookSnapshot* beginUpdate() {
        return inactive_buffer_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Commit the update - swap active and inactive buffers
     * @param buffer The buffer that was updated (must be the inactive one)
     * @param seq The new sequence number
     */
    void commitUpdate(OrderBookSnapshot* buffer, uint64_t seq) {
        buffer->sequence.store(seq, std::memory_order_release);
        buffer->timestamp.store(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count(),
            std::memory_order_release
        );
        buffer->ready.store(true, std::memory_order_release);
        
        // Atomic swap
        OrderBookSnapshot* old_active = active_buffer_.exchange(
            buffer, std::memory_order_acq_rel);
        inactive_buffer_.store(old_active, std::memory_order_release);
    }
    
    /**
     * @brief Update a bid level
     * @param price The price level
     * @param quantity The new quantity (0 to remove)
     */
    void updateBid(double price, double quantity) {
        OrderBookSnapshot* snapshot = beginUpdate();
        
        if (quantity > 0) {
            // Insert or update
            insertOrUpdateLevel(snapshot->bids, price, quantity, false);
        } else {
            // Remove
            removeLevel(snapshot->bids, price, false);
        }
        
        snapshot->updateDerived();
        commitUpdate(snapshot, snapshot->sequence.load() + 1);
    }
    
    /**
     * @brief Update an ask level
     * @param price The price level
     * @param quantity The new quantity (0 to remove)
     */
    void updateAsk(double price, double quantity) {
        OrderBookSnapshot* snapshot = beginUpdate();
        
        if (quantity > 0) {
            // Insert or update
            insertOrUpdateLevel(snapshot->asks, price, quantity, true);
        } else {
            // Remove
            removeLevel(snapshot->asks, price, true);
        }
        
        snapshot->updateDerived();
        commitUpdate(snapshot, snapshot->sequence.load() + 1);
    }
    
    /**
     * @brief Batch update multiple levels
     * @param bids Vector of bid updates (price, quantity)
     * @param asks Vector of ask updates (price, quantity)
     */
    void batchUpdate(const std::vector<std::pair<double, double>>& bids,
                     const std::vector<std::pair<double, double>>& asks) {
        OrderBookSnapshot* snapshot = beginUpdate();
        
        for (const auto& [price, qty] : bids) {
            if (qty > 0) {
                insertOrUpdateLevel(snapshot->bids, price, qty, false);
            } else {
                removeLevel(snapshot->bids, price, false);
            }
        }
        
        for (const auto& [price, qty] : asks) {
            if (qty > 0) {
                insertOrUpdateLevel(snapshot->asks, price, qty, true);
            } else {
                removeLevel(snapshot->asks, price, true);
            }
        }
        
        snapshot->updateDerived();
        commitUpdate(snapshot, snapshot->sequence.load() + 1);
    }
    
    /**
     * @brief Clear the order book
     */
    void clear() {
        OrderBookSnapshot* snapshot = beginUpdate();
        snapshot->clear();
        commitUpdate(snapshot, 0);
    }
    
    // =========================================================================
    // READER API (Rendering Thread)
    // =========================================================================
    
    /**
     * @brief Get the active buffer for reading
     * @return Pointer to the active buffer (read-only)
     */
    const OrderBookSnapshot* getActiveBuffer() const {
        return active_buffer_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get the best bid price
     */
    double getBestBid() const {
        return getActiveBuffer()->best_bid.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get the best ask price
     */
    double getBestAsk() const {
        return getActiveBuffer()->best_ask.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get the spread
     */
    double getSpread() const {
        return getActiveBuffer()->spread.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get the mid price
     */
    double getMidPrice() const {
        return getActiveBuffer()->mid_price.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get the current sequence number
     */
    uint64_t getSequence() const {
        return getActiveBuffer()->sequence.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get the timestamp of the last update
     */
    uint64_t getTimestamp() const {
        return getActiveBuffer()->timestamp.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get the number of bid levels
     */
    uint32_t getBidCount() const {
        return getActiveBuffer()->bids.size();
    }
    
    /**
     * @brief Get the number of ask levels
     */
    uint32_t getAskCount() const {
        return getActiveBuffer()->asks.size();
    }
    
    /**
     * @brief Get a specific bid level
     * @param index Level index (0 = best bid)
     * @return The bid level, or default if out of range
     */
    OrderBookLevel getBidLevel(uint32_t index) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        if (index < snapshot->bids.size()) {
            return snapshot->bids.levels[index];
        }
        return OrderBookLevel{};
    }
    
    /**
     * @brief Get a specific ask level
     * @param index Level index (0 = best ask)
     * @return The ask level, or default if out of range
     */
    OrderBookLevel getAskLevel(uint32_t index) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        if (index < snapshot->asks.size()) {
            return snapshot->asks.levels[index];
        }
        return OrderBookLevel{};
    }
    
    /**
     * @brief Copy all bid levels to a vector
     */
    void copyBids(std::vector<OrderBookLevel>& out) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        uint32_t count = snapshot->bids.size();
        out.resize(count);
        std::memcpy(out.data(), snapshot->bids.levels, count * sizeof(OrderBookLevel));
    }
    
    /**
     * @brief Copy all ask levels to a vector
     */
    void copyAsks(std::vector<OrderBookLevel>& out) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        uint32_t count = snapshot->asks.size();
        out.resize(count);
        std::memcpy(out.data(), snapshot->asks.levels, count * sizeof(OrderBookLevel));
    }
    
    /**
     * @brief Get the quantity at a specific bid price
     */
    double getBidQuantity(double price) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        int32_t idx = snapshot->bids.findLevel(price);
        if (idx >= 0) {
            return snapshot->bids.levels[idx].quantity;
        }
        return 0.0;
    }
    
    /**
     * @brief Get the quantity at a specific ask price
     */
    double getAskQuantity(double price) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        int32_t idx = snapshot->asks.findLevel(price);
        if (idx >= 0) {
            return snapshot->asks.levels[idx].quantity;
        }
        return 0.0;
    }
    
    /**
     * @brief Calculate the average price for a given quantity
     * @param is_buy true for buy (take asks), false for sell (take bids)
     * @param quantity The quantity to fill
     * @return Average price, or 0 if insufficient liquidity
     */
    double calculateAveragePrice(bool is_buy, double quantity) const {
        const OrderBookSnapshot* snapshot = getActiveBuffer();
        const OrderBookSide& side = is_buy ? snapshot->asks : snapshot->bids;
        
        double remaining = quantity;
        double total_cost = 0.0;
        
        for (uint32_t i = 0; i < side.size() && remaining > 0; ++i) {
            const OrderBookLevel& level = side.levels[i];
            double take = std::min(remaining, level.quantity);
            total_cost += take * level.price;
            remaining -= take;
        }
        
        if (remaining > 0) {
            return 0.0;  // Insufficient liquidity
        }
        
        return total_cost / quantity;
    }

private:
    // Insert or update a price level
    void insertOrUpdateLevel(OrderBookSide& side, double price, double quantity, bool ascending) {
        uint32_t count = side.size();
        
        // Find insertion point
        int32_t insert_idx = -1;
        for (uint32_t i = 0; i < count; ++i) {
            if (side.levels[i].price == price) {
                // Update existing
                side.levels[i].quantity = quantity;
                side.levels[i].order_count++;
                return;
            }
            
            // Check if this is the insertion point
            if (ascending) {
                if (side.levels[i].price > price) {
                    insert_idx = i;
                    break;
                }
            } else {
                if (side.levels[i].price < price) {
                    insert_idx = i;
                    break;
                }
            }
        }
        
        // If not found and count < max, insert at end
        if (insert_idx == -1 && count < OrderBookSide::MAX_LEVELS) {
            insert_idx = count;
        }
        
        if (insert_idx >= 0 && count < OrderBookSide::MAX_LEVELS) {
            // Shift elements
            for (int32_t i = count; i > insert_idx; --i) {
                side.levels[i] = side.levels[i - 1];
            }
            
            // Insert new level
            side.levels[insert_idx] = OrderBookLevel(price, quantity, 1);
            side.count.store(count + 1, std::memory_order_release);
        }
    }
    
    // Remove a price level
    void removeLevel(OrderBookSide& side, double price, bool ascending) {
        (void)ascending;  // Unused in removal
        
        int32_t idx = side.findLevel(price);
        if (idx >= 0) {
            uint32_t count = side.size();
            
            // Shift elements
            for (uint32_t i = idx; i < count - 1; ++i) {
                side.levels[i] = side.levels[i + 1];
            }
            
            side.count.store(count - 1, std::memory_order_release);
        }
    }
    
    // Double buffer storage
    alignas(64) std::atomic<OrderBookSnapshot*> active_buffer_;
    alignas(64) std::atomic<OrderBookSnapshot*> inactive_buffer_;
    OrderBookSnapshot* buffers_[2];
};

/**
 * @brief RAII helper for reading the order book
 */
class OrderBookReader {
public:
    explicit OrderBookReader(const DoubleBufferedOrderBook& ob)
        : snapshot_(ob.getActiveBuffer())
        , sequence_(snapshot_->sequence.load(std::memory_order_acquire))
    {}
    
    /**
     * @brief Check if the snapshot is still valid (not updated)
     */
    bool isValid() const {
        return snapshot_->sequence.load(std::memory_order_acquire) == sequence_;
    }
    
    const OrderBookSnapshot* operator->() const { return snapshot_; }
    const OrderBookSnapshot& operator*() const { return *snapshot_; }
    
private:
    const OrderBookSnapshot* snapshot_;
    uint64_t sequence_;
};

} // namespace data
} // namespace btq
