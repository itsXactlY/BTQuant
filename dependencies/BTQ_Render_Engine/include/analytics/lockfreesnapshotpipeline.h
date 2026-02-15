#ifndef PUBBTQUANT_LOCKFREESNAPSHOTPIPELINE_H
#define PUBBTQUANT_LOCKFREESNAPSHOTPIPELINE_H

#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <cstring>
#include <thread>
#include <functional>
#include <algorithm>

namespace BTQuant {
namespace RenderEngine {

// Structure representing market data that can be atomically updated
struct alignas(64) AtomicMarketData {
    std::atomic<double> price;
    std::atomic<double> volume;
    std::atomic<double> bid_price;
    std::atomic<double> ask_price;
    std::atomic<double> bid_volume;
    std::atomic<double> ask_volume;
    std::atomic<uint64_t> sequence_number;
    std::atomic<std::chrono::system_clock::time_point> timestamp;
    
    AtomicMarketData() : price(0.0), volume(0.0), bid_price(0.0), ask_price(0.0),
                         bid_volume(0.0), ask_volume(0.0), sequence_number(0) {
        timestamp.store(std::chrono::system_clock::now());
    }
    
    AtomicMarketData(double p, double v, double bp, double ap, double bv, double av) 
        : price(p), volume(v), bid_price(bp), ask_price(ap), bid_volume(bv), ask_volume(av),
          sequence_number(0) {
        timestamp.store(std::chrono::system_clock::now());
    }
    
    // Copy constructor
    AtomicMarketData(const AtomicMarketData& other) {
        price.store(other.price.load());
        volume.store(other.volume.load());
        bid_price.store(other.bid_price.load());
        ask_price.store(other.ask_price.load());
        bid_volume.store(other.bid_volume.load());
        ask_volume.store(other.ask_volume.load());
        sequence_number.store(other.sequence_number.load());
        timestamp.store(other.timestamp.load());
    }
    
    // Assignment operator
    AtomicMarketData& operator=(const AtomicMarketData& other) {
        if (this != &other) {
            price.store(other.price.load());
            volume.store(other.volume.load());
            bid_price.store(other.bid_price.load());
            ask_price.store(other.ask_price.load());
            bid_volume.store(other.bid_volume.load());
            ask_volume.store(other.ask_volume.load());
            sequence_number.store(other.sequence_number.load());
            timestamp.store(other.timestamp.load());
        }
        return *this;
    }
};

// Double-buffered snapshot system for lock-free reading
template<typename T>
class AtomicDoubleBuffer {
private:
    std::unique_ptr<T[]> buffers_[2];
    std::atomic<int> current_read_buffer_{0};
    std::atomic<int> current_write_buffer_{1};
    size_t capacity_;

public:
    explicit AtomicDoubleBuffer(size_t capacity) : capacity_(capacity) {
        buffers_[0] = std::make_unique<T[]>(capacity_);
        buffers_[1] = std::make_unique<T[]>(capacity_);

        // Initialize with default values
        for (size_t i = 0; i < capacity_; ++i) {
            new (&buffers_[0][i]) T();
            new (&buffers_[1][i]) T();
        }
    }

    ~AtomicDoubleBuffer() = default;

    // Atomically swap read/write buffers
    void swap_buffers() {
        // Atomically swap the read and write buffer indices
        int old_write = current_write_buffer_.load(std::memory_order_acquire);
        int new_read = old_write;
        int new_write = 1 - old_write;

        // Update write buffer first
        current_write_buffer_.store(new_write, std::memory_order_release);
        // Memory fence to ensure visibility of the write buffer update
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // Then update read buffer
        current_read_buffer_.store(new_read, std::memory_order_release);
    }

    T* get_write_buffer() {
        return buffers_[current_write_buffer_.load(std::memory_order_acquire)].get();
    }

    const T* get_read_buffer() const {
        return buffers_[current_read_buffer_.load(std::memory_order_acquire)].get();
    }

    size_t get_capacity() const { return capacity_; }
};

// Structure to hold ring buffer entries with symbol information
struct alignas(64) RingBufferEntry {
    std::atomic<uint32_t> symbol_id;
    AtomicMarketData data;
    
    RingBufferEntry() : symbol_id(UINT32_MAX) {}
};

// Lock-free snapshot pipeline for market data
class LockFreeSnapshotPipeline {
private:
    std::unique_ptr<AtomicDoubleBuffer<AtomicMarketData>> data_buffer_;
    std::atomic<bool> initialized_{false};
    std::atomic<uint32_t> symbol_count_{0};

    // Ring buffer for high-frequency updates
    static constexpr size_t RING_BUFFER_SIZE = 1048576; // 2^20
    static constexpr size_t RING_BUFFER_MASK = RING_BUFFER_SIZE - 1;
    std::unique_ptr<RingBufferEntry[]> ring_buffer_;
    std::atomic<uint64_t> write_head_{0};
    std::atomic<uint64_t> read_tail_{0};
    std::atomic<uint64_t> snapshot_head_{0};  // Signals when new data is ready for compute shader

    // Statistics
    std::atomic<uint64_t> total_updates_{0};
    std::atomic<uint64_t> dropped_updates_{0};

public:
    explicit LockFreeSnapshotPipeline(uint32_t max_symbols = 100000)
        : data_buffer_(std::make_unique<AtomicDoubleBuffer<AtomicMarketData>>(max_symbols)),
          ring_buffer_(std::make_unique<RingBufferEntry[]>(RING_BUFFER_SIZE)) {
        symbol_count_.store(max_symbols);
        initialized_.store(true);
    }

    ~LockFreeSnapshotPipeline() = default;

    // Write market data to the pipeline (non-blocking)
    bool write_market_data(uint32_t symbol_id, const AtomicMarketData& data) {
        if (!initialized_.load(std::memory_order_acquire) ||
            symbol_id >= symbol_count_.load(std::memory_order_acquire)) {
            return false;
        }

        // Write directly to the current write buffer for immediate availability
        data_buffer_->get_write_buffer()[symbol_id] = data;
        data_buffer_->get_write_buffer()[symbol_id].sequence_number.fetch_add(1, std::memory_order_relaxed);

        // Also write to ring buffer for high-frequency updates
        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);

        // Check if buffer is full
        if ((current_write - current_read) >= RING_BUFFER_SIZE) {
            dropped_updates_.fetch_add(1, std::memory_order_relaxed);
            // Still return true since we wrote to the main buffer
        } else {
            // Attempt to advance write head
            uint64_t new_write;
            do {
                new_write = current_write + 1;
            } while (!write_head_.compare_exchange_weak(current_write, new_write,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire));

            // Successfully acquired slot, write data to ring buffer
            size_t index = new_write & RING_BUFFER_MASK;
            ring_buffer_[index].data = data;
            ring_buffer_[index].symbol_id.store(symbol_id, std::memory_order_release);
        }

        total_updates_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Read market data snapshot (non-blocking)
    bool read_market_data_snapshot(uint32_t symbol_id, AtomicMarketData& out_data) const {
        if (!initialized_.load(std::memory_order_acquire) ||
            symbol_id >= symbol_count_.load(std::memory_order_acquire)) {
            return false;
        }

        // Search the ring buffer for the most recent update for this symbol
        // We need to scan from the most recent position backward to find the latest update
        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);

        // Look for the most recent update for this symbol in the ring buffer
        uint64_t scan_limit = current_write > RING_BUFFER_SIZE ? current_write - RING_BUFFER_SIZE : 0;
        for (uint64_t pos = current_write; pos > current_read && pos > scan_limit; ) {
            --pos;  // Decrement first to avoid underflow
            size_t index = pos & RING_BUFFER_MASK;
            uint32_t entry_symbol = ring_buffer_[index].symbol_id.load(std::memory_order_acquire);
            
            if (entry_symbol == symbol_id) {
                // Found the most recent update for this symbol
                const AtomicMarketData& ring_data = ring_buffer_[index].data;
                
                // Perform atomic reads of all fields
                out_data.price.store(ring_data.price.load(std::memory_order_acquire));
                out_data.volume.store(ring_data.volume.load(std::memory_order_acquire));
                out_data.bid_price.store(ring_data.bid_price.load(std::memory_order_acquire));
                out_data.ask_price.store(ring_data.ask_price.load(std::memory_order_acquire));
                out_data.bid_volume.store(ring_data.bid_volume.load(std::memory_order_acquire));
                out_data.ask_volume.store(ring_data.ask_volume.load(std::memory_order_acquire));
                out_data.sequence_number.store(ring_data.sequence_number.load(std::memory_order_acquire));
                out_data.timestamp.store(ring_data.timestamp.load(std::memory_order_acquire));

                return true;
            }
        }

        // Fallback to the double buffer if no ring buffer entry exists
        const AtomicMarketData* buffer = data_buffer_->get_read_buffer();
        if (buffer) {
            // Perform atomic reads of all fields
            out_data.price.store(buffer[symbol_id].price.load(std::memory_order_acquire));
            out_data.volume.store(buffer[symbol_id].volume.load(std::memory_order_acquire));
            out_data.bid_price.store(buffer[symbol_id].bid_price.load(std::memory_order_acquire));
            out_data.ask_price.store(buffer[symbol_id].ask_price.load(std::memory_order_acquire));
            out_data.bid_volume.store(buffer[symbol_id].bid_volume.load(std::memory_order_acquire));
            out_data.ask_volume.store(buffer[symbol_id].ask_volume.load(std::memory_order_acquire));
            out_data.sequence_number.store(buffer[symbol_id].sequence_number.load(std::memory_order_acquire));
            out_data.timestamp.store(buffer[symbol_id].timestamp.load(std::memory_order_acquire));

            return true;
        }

        return false;
    }

    // Batch read multiple symbols (non-blocking)
    size_t read_batch_snapshot(const uint32_t* symbol_ids, AtomicMarketData* out_data, size_t count) const {
        if (!initialized_.load(std::memory_order_acquire)) {
            return 0;
        }

        size_t successful_reads = 0;
        for (size_t i = 0; i < count; ++i) {
            uint32_t symbol_id = symbol_ids[i];
            if (symbol_id < symbol_count_.load(std::memory_order_acquire)) {
                // Use the single read method which handles both ring buffer and double buffer
                if (read_market_data_snapshot(symbol_id, out_data[i])) {
                    successful_reads++;
                }
            }
        }

        return successful_reads;
    }

    // Flush ring buffer to the current write buffer
    void flush_ring_buffer() {
        if (!initialized_.load(std::memory_order_acquire)) {
            return;
        }

        uint64_t current_read = read_tail_.load(std::memory_order_acquire);
        uint64_t current_write = write_head_.load(std::memory_order_acquire);

        // Process all pending updates in the ring buffer
        while (current_read < current_write) {
            size_t index = current_read & RING_BUFFER_MASK;
            
            // Get the symbol ID that was updated
            uint32_t symbol_id = ring_buffer_[index].symbol_id.load(std::memory_order_acquire);
            
            // Only process if this is a valid symbol update
            if (symbol_id < symbol_count_.load(std::memory_order_acquire)) {
                // Copy data from ring buffer to the current write buffer
                data_buffer_->get_write_buffer()[symbol_id] = ring_buffer_[index].data;
            }
            
            // Advance read tail
            read_tail_.store(++current_read, std::memory_order_release);
        }
    }

    // Swap buffers to make new data available for readers
    void commit_snapshot() {
        if (initialized_.load(std::memory_order_acquire)) {
            flush_ring_buffer();  // Flush ring buffer to main buffer first
            data_buffer_->swap_buffers();
            
            // Signal that new data is ready for compute shader by incrementing snapshot_head
            snapshot_head_.fetch_add(1, std::memory_order_release);
        }
    }

    // Get statistics about the pipeline
    struct PipelineStats {
        uint64_t total_updates;
        uint64_t dropped_updates;
        uint64_t write_head;
        uint64_t read_tail;
        size_t buffer_capacity;
    };

    // Getter for snapshot_head to signal VulkanCore when new data is ready
    std::atomic<uint64_t>& get_snapshot_head() { return snapshot_head_; }
    const std::atomic<uint64_t>& get_snapshot_head() const { return snapshot_head_; }
    
    // Get current value of snapshot_head
    uint64_t get_current_snapshot_head() const { return snapshot_head_.load(std::memory_order_acquire); }

    PipelineStats get_stats() const {
        PipelineStats stats;
        stats.total_updates = total_updates_.load(std::memory_order_acquire);
        stats.dropped_updates = dropped_updates_.load(std::memory_order_acquire);
        stats.write_head = write_head_.load(std::memory_order_acquire);
        stats.read_tail = read_tail_.load(std::memory_order_acquire);
        stats.buffer_capacity = data_buffer_->get_capacity();
        return stats;
    }

    // Reset statistics
    void reset_stats() {
        total_updates_.store(0, std::memory_order_release);
        dropped_updates_.store(0, std::memory_order_release);
    }
};

// Helper class for polling market data
class MarketDataPoller {
private:
    LockFreeSnapshotPipeline* pipeline_;
    std::vector<uint32_t> watched_symbols_;
    
public:
    explicit MarketDataPoller(LockFreeSnapshotPipeline* pipeline) : pipeline_(pipeline) {}
    
    void add_symbol_to_watch(uint32_t symbol_id) {
        watched_symbols_.push_back(symbol_id);
    }
    
    void remove_symbol_from_watch(uint32_t symbol_id) {
        watched_symbols_.erase(
            std::remove(watched_symbols_.begin(), watched_symbols_.end(), symbol_id),
            watched_symbols_.end()
        );
    }
    
    // Poll all watched symbols and call the callback for each
    template<typename Callback>
    void poll_watched_symbols(Callback&& callback) {
        if (!pipeline_) return;
        
        std::vector<AtomicMarketData> snapshots(watched_symbols_.size());
        size_t count = pipeline_->read_batch_snapshot(
            watched_symbols_.data(), 
            snapshots.data(), 
            watched_symbols_.size()
        );
        
        for (size_t i = 0; i < count; ++i) {
            callback(watched_symbols_[i], snapshots[i]);
        }
    }
};

} // namespace RenderEngine
} // namespace BTQuant

#endif // PUBBTQUANT_LOCKFREESNAPSHOTPIPELINE_H