#ifndef BTQUANT_MARKET_DATA_PROCESSOR_HPP
#define BTQUANT_MARKET_DATA_PROCESSOR_HPP

#include "data_spine.hpp"
#include "market_data.hpp"
#include "ring_buffer.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace btquant {

// Central real-time market data pipeline.
//
// Subscribes to a single DataSpine instance and pumps incoming ticks into
// a MarketDataAggregator. Maintains a thread-safe snapshot for widgets to
// query without locking the aggregator's internal vector.
//
// Threading model:
//   - Background thread polls DataSpine every poll_interval_ms
//   - Each polled tick is pushed to the aggregator AND broadcast to subscribers
//   - Widgets call snapshot() which returns a thread-safe copy of the latest
//     order book + metrics + trade list (rolling window)
//
// Single-symbol MVP — symbol_id 0 by default. Multi-symbol routing is the
// next iteration.
class MarketDataProcessor {
public:
    MarketDataProcessor();
    ~MarketDataProcessor();

    // Initialize + start background thread polling the given spine path
    // (typically /dev/shm/btquant_hotspine). Returns nullptr on success.
    [[nodiscard]] std::optional<std::string>
    start(const std::string& hotspine_path,
          uint32_t poll_interval_ms = 16);  // ~60 Hz

    // Stop the background thread and close the spine.
    void stop();

    // Snapshot of the current state — thread-safe, returns by-value copy.
    // Trade list is bounded to last_n_trades.
    struct Snapshot {
        data::OrderBook order_book;
        data::MarketMetrics metrics;
        std::vector<data::Trade> recent_trades;  // last N trades, newest first
        uint64_t snapshot_seq = 0;              // monotonically increasing
    };
    Snapshot snapshot(size_t last_n_trades = 100) const;

    bool isRunning() const noexcept { return m_running.load(std::memory_order_acquire); }

private:
    void runLoop();

    std::unique_ptr<data::DataSpine> m_spine;
    data::MarketDataAggregator m_aggregator;

    mutable std::mutex m_snapshotMutex;
    Snapshot m_latestSnapshot;

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};
    std::thread m_thread;
    uint32_t m_pollIntervalMs = 16;
    uint64_t m_snapshotSeq = 0;
};

}  // namespace btquant

#endif
