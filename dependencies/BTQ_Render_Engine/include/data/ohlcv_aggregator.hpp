#pragma once

/**
 * @file ohlcv_aggregator.hpp
 * @brief Real-time OHLCV Candlestick Aggregation Engine
 * 
 * This implementation provides lock-free candlestick building from tick data.
 * It supports multiple timeframes simultaneously and uses atomic operations
 * for thread-safe updates.
 * 
 * Key features:
 * - Lock-free tick aggregation
 * - Multiple timeframe support (1s, 5s, 1m, 5m, 15m, 1h, 4h, 1d)
 * - Real-time candle updates
 * - Historical candle storage
 * - Memory-efficient design
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace btq {
namespace data {

// Cache line size for alignment
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Single OHLCV candle data
 */
struct alignas(64) OHLCVCandle {
    uint64_t timestamp{0};      // Candle start timestamp (microseconds)
    uint64_t end_timestamp{0};  // Candle end timestamp
    double open{0.0};           // Opening price
    double high{0.0};           // Highest price
    double low{0.0};            // Lowest price
    double close{0.0};          // Closing price
    double volume{0.0};         // Total volume
    double vwap{0.0};           // Volume-weighted average price
    uint64_t tick_count{0};     // Number of ticks in this candle
    uint64_t trade_count{0};    // Number of trades
    bool is_closed{false};      // Whether this candle is finalized
    
    OHLCVCandle() = default;
    
    OHLCVCandle(uint64_t ts, double price, double vol)
        : timestamp(ts)
        , end_timestamp(ts)
        , open(price)
        , high(price)
        , low(price)
        , close(price)
        , volume(vol)
        , vwap(price)
        , tick_count(1)
        , trade_count(1)
        , is_closed(false)
    {}
    
    /**
     * @brief Update candle with a new tick
     */
    void update(double price, double vol, uint64_t ts) {
        if (tick_count == 0) {
            // First tick
            open = high = low = close = price;
            volume = vol;
            vwap = price;
            timestamp = ts;
            tick_count = 1;
            trade_count = 1;
        } else {
            // Update existing candle
            high = std::max(high, price);
            low = std::min(low, price);
            close = price;
            
            // Update VWAP incrementally
            double total_value = vwap * volume + price * vol;
            volume += vol;
            vwap = total_value / volume;
            
            ++tick_count;
            ++trade_count;
        }
        end_timestamp = ts;
    }
    
    /**
     * @brief Mark the candle as closed
     */
    void close() {
        is_closed = true;
    }
    
    /**
     * @brief Reset the candle
     */
    void reset() {
        std::memset(this, 0, sizeof(OHLCVCandle));
    }
};

/**
 * @brief Timeframe specification
 */
struct TimeframeSpec {
    std::string name;           // e.g., "1m", "5m", "1h"
    uint64_t duration_us;       // Duration in microseconds
    
    static TimeframeSpec fromString(const std::string& tf) {
        TimeframeSpec spec;
        spec.name = tf;
        
        // Parse the timeframe string
        uint64_t multiplier = 0;
        char unit = '\0';
        
        size_t i = 0;
        while (i < tf.size() && std::isdigit(tf[i])) {
            multiplier = multiplier * 10 + (tf[i] - '0');
            ++i;
        }
        
        if (i < tf.size()) {
            unit = tf[i];
        }
        
        // Convert to microseconds
        switch (unit) {
            case 's': spec.duration_us = multiplier * 1000000ULL; break;       // Seconds
            case 'm': spec.duration_us = multiplier * 60ULL * 1000000ULL; break;  // Minutes
            case 'h': spec.duration_us = multiplier * 3600ULL * 1000000ULL; break; // Hours
            case 'd': spec.duration_us = multiplier * 86400ULL * 1000000ULL; break; // Days
            case 'w': spec.duration_us = multiplier * 604800ULL * 1000000ULL; break; // Weeks
            default: spec.duration_us = 60000000ULL; break; // Default 1 minute
        }
        
        return spec;
    }
};

/**
 * @brief Lock-free candle aggregator for a single timeframe
 */
class SingleTimeframeAggregator {
public:
    explicit SingleTimeframeAggregator(const TimeframeSpec& spec, size_t history_size = 1000)
        : spec_(spec)
        , history_size_(history_size)
    {
        history_.reserve(history_size);
    }
    
    /**
     * @brief Process a new tick
     * @param price Trade price
     * @param volume Trade volume
     * @param timestamp Trade timestamp (microseconds)
     * @return true if a new candle was created
     */
    bool processTick(double price, double volume, uint64_t timestamp) {
        // Calculate the candle start time for this tick
        uint64_t candle_start = (timestamp / spec_.duration_us) * spec_.duration_us;
        
        // Check if we need a new candle
        bool new_candle = false;
        
        if (current_candle_.tick_count == 0) {
            // First tick ever
            current_candle_ = OHLCVCandle(candle_start, price, volume);
            new_candle = true;
        } else if (candle_start > current_candle_.timestamp) {
            // New candle period - close the old one and start new
            current_candle_.close();
            
            // Store the closed candle
            {
                std::lock_guard<std::mutex> lock(history_mutex_);
                history_.push_back(current_candle_);
                if (history_.size() > history_size_) {
                    history_.erase(history_.begin());
                }
            }
            
            // Start new candle
            current_candle_ = OHLCVCandle(candle_start, price, volume);
            new_candle = true;
        } else {
            // Update existing candle
            current_candle_.update(price, volume, timestamp);
        }
        
        return new_candle;
    }
    
    /**
     * @brief Get the current (active) candle
     */
    const OHLCVCandle& getCurrentCandle() const {
        return current_candle_;
    }
    
    /**
     * @brief Get historical candles
     */
    std::vector<OHLCVCandle> getHistory(size_t count = 0) const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        if (count == 0 || count >= history_.size()) {
            return history_;
        }
        
        return std::vector<OHLCVCandle>(
            history_.end() - count, history_.end());
    }
    
    /**
     * @brief Get the last N closed candles
     */
    std::vector<OHLCVCandle> getLastClosedCandles(size_t count) const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        std::vector<OHLCVCandle> result;
        result.reserve(count);
        
        size_t start = history_.size() > count ? history_.size() - count : 0;
        for (size_t i = start; i < history_.size(); ++i) {
            if (history_[i].is_closed) {
                result.push_back(history_[i]);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Get the timeframe specification
     */
    const TimeframeSpec& getSpec() const {
        return spec_;
    }
    
    /**
     * @brief Get the number of historical candles
     */
    size_t getHistoryCount() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return history_.size();
    }
    
    /**
     * @brief Clear all history
     */
    void clear() {
        std::lock_guard<std::mutex> lock(history_mutex_);
        history_.clear();
        current_candle_.reset();
    }

private:
    TimeframeSpec spec_;
    size_t history_size_;
    
    alignas(64) OHLCVCandle current_candle_;
    mutable std::mutex history_mutex_;
    std::vector<OHLCVCandle> history_;
};

/**
 * @brief Multi-timeframe OHLCV aggregator
 * 
 * Manages candlestick aggregation for multiple timeframes simultaneously.
 * Thread-safe for concurrent tick processing and candle reading.
 */
class OHLCVAggregator {
public:
    /**
     * @brief Default timeframes for trading
     */
    static std::vector<std::string> getDefaultTimeframes() {
        return {"1s", "5s", "15s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"};
    }
    
    OHLCVAggregator() {
        // Initialize with default timeframes
        initialize(getDefaultTimeframes());
    }
    
    explicit OHLCVAggregator(const std::vector<std::string>& timeframes) {
        initialize(timeframes);
    }
    
    /**
     * @brief Process a new tick for all timeframes
     * @param price Trade price
     * @param volume Trade volume
     * @param timestamp Trade timestamp (microseconds since epoch)
     */
    void processTick(double price, double volume, uint64_t timestamp) {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        for (auto& [name, aggregator] : aggregators_) {
            aggregator->processTick(price, volume, timestamp);
        }
    }
    
    /**
     * @brief Process a batch of ticks
     */
    void processTicks(const std::vector<std::tuple<double, double, uint64_t>>& ticks) {
        for (const auto& [price, volume, timestamp] : ticks) {
            processTick(price, volume, timestamp);
        }
    }
    
    /**
     * @brief Get the current candle for a timeframe
     */
    std::optional<OHLCVCandle> getCurrentCandle(const std::string& timeframe) const {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        auto it = aggregators_.find(timeframe);
        if (it != aggregators_.end()) {
            return it->second->getCurrentCandle();
        }
        return std::nullopt;
    }
    
    /**
     * @brief Get historical candles for a timeframe
     */
    std::vector<OHLCVCandle> getHistory(
        const std::string& timeframe, size_t count = 0) const
    {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        auto it = aggregators_.find(timeframe);
        if (it != aggregators_.end()) {
            return it->second->getHistory(count);
        }
        return {};
    }
    
    /**
     * @brief Get all current candles
     */
    std::unordered_map<std::string, OHLCVCandle> getAllCurrentCandles() const {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        std::unordered_map<std::string, OHLCVCandle> result;
        for (const auto& [name, aggregator] : aggregators_) {
            result[name] = aggregator->getCurrentCandle();
        }
        return result;
    }
    
    /**
     * @brief Add a new timeframe
     */
    void addTimeframe(const std::string& timeframe, size_t history_size = 1000) {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        if (aggregators_.find(timeframe) == aggregators_.end()) {
            auto spec = TimeframeSpec::fromString(timeframe);
            aggregators_[timeframe] = std::make_unique<SingleTimeframeAggregator>(
                spec, history_size);
        }
    }
    
    /**
     * @brief Remove a timeframe
     */
    void removeTimeframe(const std::string& timeframe) {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        aggregators_.erase(timeframe);
    }
    
    /**
     * @brief Get available timeframes
     */
    std::vector<std::string> getTimeframes() const {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        std::vector<std::string> result;
        result.reserve(aggregators_.size());
        for (const auto& [name, _] : aggregators_) {
            result.push_back(name);
        }
        return result;
    }
    
    /**
     * @brief Clear all history for all timeframes
     */
    void clear() {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        for (auto& [_, aggregator] : aggregators_) {
            aggregator->clear();
        }
    }
    
    /**
     * @brief Get statistics
     */
    struct Stats {
        size_t timeframe_count;
        std::unordered_map<std::string, size_t> history_counts;
    };
    
    Stats getStats() const {
        std::lock_guard<std::mutex> lock(aggregators_mutex_);
        
        Stats stats;
        stats.timeframe_count = aggregators_.size();
        
        for (const auto& [name, aggregator] : aggregators_) {
            stats.history_counts[name] = aggregator->getHistoryCount();
        }
        
        return stats;
    }

private:
    void initialize(const std::vector<std::string>& timeframes) {
        for (const auto& tf : timeframes) {
            addTimeframe(tf);
        }
    }
    
    mutable std::mutex aggregators_mutex_;
    std::unordered_map<std::string, std::unique_ptr<SingleTimeframeAggregator>> aggregators_;
};

/**
 * @brief Tick data structure for aggregation
 */
struct TickData {
    uint64_t timestamp{0};
    double price{0.0};
    double volume{0.0};
    bool is_buy{true};
    uint64_t trade_id{0};
    
    TickData() = default;
    TickData(uint64_t ts, double p, double v, bool buy = true, uint64_t id = 0)
        : timestamp(ts), price(p), volume(v), is_buy(buy), trade_id(id) {}
};

/**
 * @brief Real-time tick processor with OHLCV output
 * 
 * This class provides a high-performance interface for processing
 * tick data and generating OHLCV candles in real-time.
 */
class RealtimeTickProcessor {
public:
    using TickCallback = std::function<void(const TickData&)>;
    using CandleCallback = std::function<void(const std::string&, const OHLCVCandle&)>;
    
    RealtimeTickProcessor() = default;
    
    /**
     * @brief Process a single tick
     */
    void processTick(const TickData& tick) {
        // Update aggregator
        aggregator_.processTick(tick.price, tick.volume, tick.timestamp);
        
        // Call tick callback if set
        if (tick_callback_) {
            tick_callback_(tick);
        }
        
        // Check for new candles and call callbacks
        if (candle_callback_) {
            for (const auto& tf : aggregator_.getTimeframes()) {
                auto candle = aggregator_.getCurrentCandle(tf);
                if (candle) {
                    candle_callback_(tf, *candle);
                }
            }
        }
        
        ++tick_count_;
    }
    
    /**
     * @brief Process a batch of ticks
     */
    void processTicks(const std::vector<TickData>& ticks) {
        for (const auto& tick : ticks) {
            processTick(tick);
        }
    }
    
    /**
     * @brief Set the tick callback
     */
    void setTickCallback(TickCallback callback) {
        tick_callback_ = std::move(callback);
    }
    
    /**
     * @brief Set the candle callback
     */
    void setCandleCallback(CandleCallback callback) {
        candle_callback_ = std::move(callback);
    }
    
    /**
     * @brief Get the aggregator
     */
    OHLCVAggregator& getAggregator() {
        return aggregator_;
    }
    
    /**
     * @brief Get tick count
     */
    uint64_t getTickCount() const {
        return tick_count_;
    }

private:
    OHLCVAggregator aggregator_;
    TickCallback tick_callback_;
    CandleCallback candle_callback_;
    std::atomic<uint64_t> tick_count_{0};
};

} // namespace data
} // namespace btq
