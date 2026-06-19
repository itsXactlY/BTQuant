#ifndef BTQUANT_MARKET_DATA_HPP
#define BTQUANT_MARKET_DATA_HPP

#include <cstdint>
#include <vector>
#include <cmath>

namespace btquant::data {

struct MarketTick {
    double price = 0;
    double size = 0;
    uint64_t timestamp = 0;
    bool isBuy = true;
};

struct OrderBookLevel {
    double price = 0;
    double size = 0;
    double cumSize = 0;
};

struct OrderBook {
    static constexpr size_t MAX_LEVELS = 50;
    OrderBookLevel bids[MAX_LEVELS];
    OrderBookLevel asks[MAX_LEVELS];
    size_t bidCount = 0;
    size_t askCount = 0;
    double midPrice = 0;
    double spread = 0;
};

struct MarketMetrics {
    double vwap = 0;
    double high = 0;
    double low = 0;
    double volume = 0;
    double buyVolume = 0;
    double sellVolume = 0;
    double delta = 0;
    int64_t tradeCount = 0;
};

// One OHLCV candle over a fixed-duration bucket. startTime/endTime are
// microseconds since epoch; bucket width is determined by the aggregator
// (1 minute by default).
struct Candle {
    double open = 0;
    double high = 0;
    double low = 0;
    double close = 0;
    double volume = 0;
    double buyVolume = 0;   // sub-volume for isBuy=true ticks
    double sellVolume = 0;  // sub-volume for isBuy=false ticks
    double delta = 0;       // buyVolume - sellVolume
    uint64_t startTime = 0; // bucket start (us)
    uint64_t endTime = 0;   // bucket end (us)
    uint32_t tradeCount = 0;
};

struct Trade {
    uint64_t id = 0;
    double price = 0;
    double size = 0;
    uint64_t timestamp = 0;
    bool isBuy = true;
};

// Per-symbol aggregator. Holds the latest order-book snapshot, rolling
// metrics, the recent trade list, and a rolling window of bucketed candles.
//
// Candle aggregation: every update(tick) call folds the tick into the
// "current" minute bucket. When the timestamp crosses into a new minute,
// the previous bucket is finalized (pushed into m_candles) and a new
// one is started. m_candles is pruned to m_maxCandles (oldest first).
class MarketDataAggregator {
public:
    // Bucket width for candle aggregation, in microseconds. Default 60s.
    static constexpr uint64_t DEFAULT_CANDLE_BUCKET_US = 60ULL * 1'000'000ULL;
    static constexpr size_t  DEFAULT_MAX_CANDLES = 60;  // 1 hour at 1-min buckets

    explicit MarketDataAggregator(uint64_t candle_bucket_us = DEFAULT_CANDLE_BUCKET_US,
                                 size_t   max_candles     = DEFAULT_MAX_CANDLES);

    void update(const MarketTick& tick);
    void reset();

    [[nodiscard]] const OrderBook& orderBook() const { return m_orderBook; }
    [[nodiscard]] const MarketMetrics& metrics() const { return m_metrics; }
    [[nodiscard]] const std::vector<Trade>& trades() const { return m_trades; }
    [[nodiscard]] const std::vector<Candle>& candles() const { return m_candles; }
    [[nodiscard]] const Candle* currentCandle() const { return m_haveCurrent ? &m_current : nullptr; }

private:
    // Close out the in-progress candle and append it to m_candles.
    // Does NOT touch the order book or metrics — those stay live.
    void finalizeCurrentCandle();

    // Map a microsecond timestamp to a bucket-aligned Candle skeleton
    // (open=close=price, high=low=price, volume=0, start/end=aligned).
    static Candle makeBucket(uint64_t timestamp_us, double price, uint64_t bucket_us);

    OrderBook   m_orderBook;
    MarketMetrics m_metrics;
    std::vector<Trade> m_trades;
    std::vector<Candle> m_candles;

    Candle m_current;        // in-progress candle for the current bucket
    bool   m_haveCurrent = false;
    uint64_t m_currentBucketStart = 0;  // startTime of m_current (cached)
    uint64_t m_candleBucketUs = DEFAULT_CANDLE_BUCKET_US;
    size_t   m_maxCandles = DEFAULT_MAX_CANDLES;

    uint64_t m_lastTimestamp = 0;
    double   m_vwapSum = 0;
    double   m_vwapVolume = 0;
};

} // namespace btquant::data

#endif
