#include "rawtradetable.h"
#include "traderingbuffer.h"
#include <algorithm>
#include <numeric>
#include <cmath>

RawTradeTable::RawTradeTable(size_t max_capacity) : max_capacity_(max_capacity) {
    // Note: std::deque doesn't have a reserve method like std::vector
    // We can optionally set a max size hint, but it's not necessary
    
    // Initialize the trade ring buffer for high-frequency atomic access
    trade_ring_buffer_ = std::make_unique<TradeRingBuffer>();
}

void RawTradeTable::add_trade(const RawTrade& trade) {
    std::lock_guard<std::mutex> lock(trades_mutex_);

    trades_.push_front(trade);

    // Trim if we exceed capacity
    if (trades_.size() > max_capacity_) {
        trades_.pop_back();
    }

    // Call the trade notification callback if set
    if (trade_callback_) {
        trade_callback_(trade);
    }
    
    // Write to the trade ring buffer for atomic access
    if (trade_ring_buffer_) {
        trade_ring_buffer_->write_trade(trade);
    }
}

void RawTradeTable::add_trades(const std::vector<RawTrade>& trades) {
    std::lock_guard<std::mutex> lock(trades_mutex_);

    // Add trades in reverse order to maintain chronological order in the front
    for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
        trades_.push_front(*it);

        // Call the trade notification callback for each trade if set
        if (trade_callback_) {
            trade_callback_(*it);
        }
    }

    // Trim if we exceed capacity
    while (trades_.size() > max_capacity_) {
        trades_.pop_back();
    }
    
    // Write all trades to the ring buffer for atomic access
    if (trade_ring_buffer_) {
        for (const auto& trade : trades) {
            trade_ring_buffer_->write_trade(trade);
        }
    }
}

std::vector<RawTrade> RawTradeTable::get_recent_trades(size_t count) const {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    
    size_t actual_count = std::min(count, trades_.size());
    std::vector<RawTrade> result;
    result.reserve(actual_count);
    
    auto it = trades_.begin();
    for (size_t i = 0; i < actual_count && it != trades_.end(); ++i, ++it) {
        result.push_back(*it);
    }
    
    return result;
}

std::vector<RawTrade> RawTradeTable::get_trades_in_range(
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time) const {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    
    std::vector<RawTrade> result;
    
    for (const auto& trade : trades_) {
        if (trade.timestamp >= start_time && trade.timestamp <= end_time) {
            result.push_back(trade);
        }
    }
    
    // Sort by timestamp in descending order (most recent first)
    std::sort(result.begin(), result.end(), [](const RawTrade& a, const RawTrade& b) {
        return a.timestamp > b.timestamp;
    });
    
    return result;
}

TradeStats RawTradeTable::get_trade_statistics() const {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    
    TradeStats stats{};
    stats.total_trades = trades_.size();
    
    if (trades_.empty()) {
        return stats;
    }
    
    double total_volume = 0.0;
    double largest_trade = 0.0;
    double buy_vol = 0.0, sell_vol = 0.0;
    int buy_count = 0, sell_count = 0;
    
    for (const auto& trade : trades_) {
        total_volume += trade.volume;
        
        if (trade.volume > largest_trade) {
            largest_trade = trade.volume;
        }
        
        if (trade.side == 'B' || trade.side == 'b') {
            buy_vol += trade.volume;
            buy_count++;
        } else if (trade.side == 'S' || trade.side == 's') {
            sell_vol += trade.volume;
            sell_count++;
        }
    }
    
    stats.total_volume = total_volume;
    stats.avg_trade_size = stats.total_trades > 0 ? total_volume / stats.total_trades : 0.0;
    stats.largest_trade_size = largest_trade;
    stats.buy_volume = buy_vol;
    stats.sell_volume = sell_vol;
    stats.buy_count = buy_count;
    stats.sell_count = sell_count;
    
    return stats;
}

void RawTradeTable::clear() {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    trades_.clear();
}

size_t RawTradeTable::size() const {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    return trades_.size();
}

void RawTradeTable::set_max_capacity(size_t new_capacity) {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    max_capacity_ = new_capacity;
    
    // Trim if current size exceeds new capacity
    while (trades_.size() > max_capacity_) {
        trades_.pop_back();
    }
}

std::vector<RawTrade> RawTradeTable::get_all_trades() const {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    std::vector<RawTrade> result(trades_.begin(), trades_.end());
    return result;
}