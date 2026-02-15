#ifndef PUBBTQUANT_RAWTRADETABLE_H
#define PUBBTQUANT_RAWTRADETABLE_H

#include <vector>
#include <deque>
#include <mutex>
#include <chrono>
#include <string>
#include <functional>
#include <memory>

// Structure to represent a single raw trade
struct RawTrade {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double volume;
    char side; // 'B' for buy, 'S' for sell
    std::string trade_id;

    RawTrade() : timestamp(std::chrono::system_clock::now()), price(0.0), volume(0.0), side('N'), trade_id("") {}
    RawTrade(std::chrono::system_clock::time_point ts, double p, double v, char s, const std::string& id)
        : timestamp(ts), price(p), volume(v), side(s), trade_id(id) {}
};

// Structure to represent trade statistics
struct TradeStats {
    size_t total_trades;
    double total_volume;
    double avg_trade_size;
    double largest_trade_size;
    double buy_volume;
    double sell_volume;
    int buy_count;
    int sell_count;
};

// Function type for trade notification callbacks
typedef std::function<void(const RawTrade&)> TradeNotificationCallback;

// Forward declaration
class TradeRingBuffer;

class RawTradeTable {
private:
    std::deque<RawTrade> trades_;
    size_t max_capacity_;
    mutable std::mutex trades_mutex_;

    // Callback for when new trades are added
    TradeNotificationCallback trade_callback_;

    // Trade ring buffer for high-frequency atomic access
    std::unique_ptr<TradeRingBuffer> trade_ring_buffer_;

public:
    explicit RawTradeTable(size_t max_capacity = 10000); // Default to 10k trades

    // Add a single trade to the table
    void add_trade(const RawTrade& trade);

    // Add multiple trades at once
    void add_trades(const std::vector<RawTrade>& trades);

    // Get the most recent trades (up to count)
    std::vector<RawTrade> get_recent_trades(size_t count = 100) const;

    // Get trades within a specific time range
    std::vector<RawTrade> get_trades_in_range(
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time) const;

    // Get trade statistics
    TradeStats get_trade_statistics() const;

    // Clear all trades
    void clear();

    // Get current number of trades stored
    size_t size() const;

    // Get maximum capacity
    size_t max_size() const { return max_capacity_; }

    // Set maximum capacity (will trim excess trades if reducing size)
    void set_max_capacity(size_t new_capacity);

    // Get all trades (for UI rendering - use carefully with large datasets)
    std::vector<RawTrade> get_all_trades() const;

    // Set callback function to be called when a new trade is added
    void set_trade_notification_callback(TradeNotificationCallback callback) {
        trade_callback_ = callback;
    }

    // Access to the trade ring buffer for atomic tail reading
    TradeRingBuffer* get_trade_ring_buffer() { return trade_ring_buffer_.get(); }
    const TradeRingBuffer* get_trade_ring_buffer() const { return trade_ring_buffer_.get(); }
};

#endif // PUBBTQUANT_RAWTRADETABLE_H