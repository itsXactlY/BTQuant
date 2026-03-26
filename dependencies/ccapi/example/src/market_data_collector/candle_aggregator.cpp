#include "candle_aggregator.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

using namespace MarketData;

CandleAggregator::CandleAggregator(
    const std::vector<std::string>& timeframes)
    : timeframes_(timeframes) {}

std::string CandleAggregator::makeKey(const std::string& exchange,
                                      const std::string& symbol,
                                      const std::string& timeframe) {
    return exchange + ":" + symbol + ":" + timeframe;
}

int64_t CandleAggregator::timeframeMillis(const std::string& tf) const {
    // return microseconds here
    if (tf.empty()) return 60'000'000LL;  // default 60s in µs

    char unit = tf.back();
    int64_t val = std::stoll(tf.substr(0, tf.size() - 1));

    constexpr int64_t SECOND = 1'000'000LL;
    constexpr int64_t MINUTE = 60LL * SECOND;
    constexpr int64_t HOUR   = 60LL * MINUTE;
    constexpr int64_t DAY    = 24LL * HOUR;

    switch (unit) {
        case 's': return val * SECOND;
        case 'm': return val * MINUTE;
        case 'h': return val * HOUR;
        case 'd': return val * DAY;
        default:  return val * MINUTE;
    }
}



int64_t CandleAggregator::alignTimestamp(
    int64_t timestamp_us,
    const std::string& timeframe) const {
    auto tf_ms = timeframeMillis(timeframe);
    return (timestamp_us / tf_ms) * tf_ms;
}

void CandleAggregator::processTrade(const Trade& trade) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& tf : timeframes_) {
        const int64_t bucket =
            alignTimestamp(trade.timestamp_us, tf);
        const std::string key =
            makeKey(trade.exchange, trade.symbol, tf);

        auto& cs = active_[key];

        // New candle or first trade in this timeframe
        if (!cs.is_initialized) {
            cs.open_time_ms  = bucket;
            cs.open          = trade.price;
            cs.high          = trade.price;
            cs.low           = trade.price;
            cs.close         = trade.price;
            cs.volume        = trade.quantity;
            cs.market_type   = trade.market_type;
            cs.is_initialized = true;
            continue;
        }

        // Bucket advanced => close old candle, start a new one
        if (bucket > cs.open_time_ms) {
            OHLCV c;
            c.timestamp_us = cs.open_time_ms;
            c.exchange     = trade.exchange;
            c.symbol       = trade.symbol;
            c.market_type  = cs.market_type;
            c.timeframe    = tf;
            c.open         = cs.open;
            c.high         = cs.high;
            c.low          = cs.low;
            c.close        = cs.close;
            c.volume       = cs.volume;
            completed_.push_back(std::move(c));

            cs.open_time_ms = bucket;
            cs.open         = trade.price;
            cs.high         = trade.price;
            cs.low          = trade.price;
            cs.close        = trade.price;
            cs.volume       = trade.quantity;
        } else {
            // Same bucket: update OHLCV
            cs.high   = std::max(cs.high, trade.price);
            cs.low    = std::min(cs.low,  trade.price);
            cs.close  = trade.price;
            cs.volume += trade.quantity;
        }
    }
}

std::vector<OHLCV> CandleAggregator::getAllCompletedCandles() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<OHLCV> out;
    out.swap(completed_);
    return out;
}

std::vector<OHLCV> CandleAggregator::flushAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& kv : active_) {
        const auto& key = kv.first;
        const auto& cs  = kv.second;
        if (!cs.is_initialized) continue;

        // extract timeframe from key ("ex:sym:tf")
        auto pos1 = key.rfind(':');
        if (pos1 == std::string::npos) continue;
        std::string tf = key.substr(pos1 + 1);

        // exchange:symbol
        std::string ex_sym = key.substr(0, pos1);
        auto pos2 = ex_sym.find(':');
        if (pos2 == std::string::npos) continue;

        std::string ex = ex_sym.substr(0, pos2);
        std::string sym = ex_sym.substr(pos2 + 1);

        OHLCV c;
        c.timestamp_us = cs.open_time_ms;
        c.exchange     = ex;
        c.symbol       = sym;
        c.market_type  = cs.market_type;
        c.timeframe    = tf;
        c.open         = cs.open;
        c.high         = cs.high;
        c.low          = cs.low;
        c.close        = cs.close;
        c.volume       = cs.volume;
        completed_.push_back(std::move(c));
    }

    active_.clear();

    std::vector<OHLCV> out;
    out.swap(completed_);
    return out;
}
