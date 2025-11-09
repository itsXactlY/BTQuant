#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "market_data_types.h"

class CandleAggregator {
public:
    explicit CandleAggregator(const std::vector<std::string>& timeframes);

    void processTrade(const MarketData::Trade& trade);

    // Returns all completed candles (for all symbols/timeframes)
    std::vector<MarketData::OHLCV> getAllCompletedCandles();

    // Force close all open candles (e.g. shutdown)
    std::vector<MarketData::OHLCV> flushAll();

private:
    struct CandleState {
        int64_t open_time_ms{0};
        double open{0.0};
        double high{0.0};
        double low{0.0};
        double close{0.0};
        double volume{0.0};
        std::string market_type;
        bool is_initialized{false};
    };

    // key: exchange:symbol:timeframe
    std::unordered_map<std::string, CandleState> active_;
    std::vector<MarketData::OHLCV> completed_;
    std::vector<std::string> timeframes_;
    std::mutex mutex_;

    int64_t alignTimestamp(int64_t timestamp_ms,
                           const std::string& timeframe) const;
    int64_t timeframeMillis(const std::string& timeframe) const;

    static std::string makeKey(const std::string& exchange,
                               const std::string& symbol,
                               const std::string& timeframe);
};
