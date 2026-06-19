#include "market_data.hpp"

namespace btquant::data {

MarketDataAggregator::MarketDataAggregator(uint64_t candle_bucket_us, size_t max_candles)
    : m_candleBucketUs(candle_bucket_us)
    , m_maxCandles(max_candles) {}

Candle MarketDataAggregator::makeBucket(uint64_t timestamp_us, double price, uint64_t bucket_us) {
    // Align the bucket to fixed-width grid so all trades in the same minute
    // land in the same bucket regardless of when the first one arrives.
    uint64_t aligned = (timestamp_us / bucket_us) * bucket_us;
    Candle c{};
    c.open = price;
    c.high = price;
    c.low  = price;
    c.close = price;
    c.startTime = aligned;
    c.endTime   = aligned + bucket_us;
    return c;
}

void MarketDataAggregator::update(const MarketTick& tick) {
    // 1. Trade history (rolling, last 200 entries).
    m_trades.push_back({m_trades.size(), tick.price, tick.size, tick.timestamp, tick.isBuy});
    if (m_trades.size() > 200) {
        m_trades.erase(m_trades.begin());
    }

    // 2. Metrics (volume, VWAP, delta, high/low).
    if (tick.isBuy) m_metrics.buyVolume += tick.size;
    else            m_metrics.sellVolume += tick.size;
    m_metrics.volume += tick.size;
    m_metrics.delta = m_metrics.buyVolume - m_metrics.sellVolume;
    m_metrics.tradeCount++;
    m_vwapSum    += tick.price * tick.size;
    m_vwapVolume += tick.size;
    if (m_vwapVolume > 0) m_metrics.vwap = m_vwapSum / m_vwapVolume;
    if (tick.price > m_metrics.high || m_metrics.high == 0) m_metrics.high = tick.price;
    if (tick.price < m_metrics.low  || m_metrics.low  == 0) m_metrics.low  = tick.price;
    m_lastTimestamp = tick.timestamp;

    // 3. Candle aggregation — bucket the tick into its minute window.
    if (tick.timestamp == 0) return;  // guard against producer w/o timestamp

    uint64_t bucket_start = (tick.timestamp / m_candleBucketUs) * m_candleBucketUs;

    if (!m_haveCurrent) {
        // First tick ever — start a fresh bucket at this tick's bucket.
        m_current = makeBucket(tick.timestamp, tick.price, m_candleBucketUs);
        m_currentBucketStart = bucket_start;
        m_haveCurrent = true;
        // fall through to fold this tick into m_current
    } else if (bucket_start != m_currentBucketStart) {
        // Bucket boundary crossed — finalize the previous candle.
        m_current.endTime = m_currentBucketStart + m_candleBucketUs;
        finalizeCurrentCandle();
        // Start a new bucket.
        m_current = makeBucket(tick.timestamp, tick.price, m_candleBucketUs);
        m_currentBucketStart = bucket_start;
        m_haveCurrent = true;
    }

    // Fold tick into the (now-current) bucket.
    if (tick.price > m_current.high) m_current.high = tick.price;
    if (tick.price < m_current.low)  m_current.low  = tick.price;
    m_current.close = tick.price;
    m_current.volume += tick.size;
    m_current.tradeCount++;
    if (tick.isBuy) m_current.buyVolume  += tick.size;
    else            m_current.sellVolume += tick.size;
    m_current.delta = m_current.buyVolume - m_current.sellVolume;
    m_current.endTime = bucket_start + m_candleBucketUs;
}

void MarketDataAggregator::finalizeCurrentCandle() {
    m_candles.push_back(m_current);
    if (m_candles.size() > m_maxCandles) {
        // Drop oldest until under cap. Usually just 1 per minute so this
        // is a single erase per tick — cheap.
        size_t excess = m_candles.size() - m_maxCandles;
        m_candles.erase(m_candles.begin(),
                        m_candles.begin() + static_cast<std::ptrdiff_t>(excess));
    }
    m_haveCurrent = false;
}

void MarketDataAggregator::reset() {
    m_metrics = {};
    m_orderBook = {};
    m_trades.clear();
    m_candles.clear();
    m_haveCurrent = false;
    m_lastTimestamp = 0;
    m_vwapSum = 0;
    m_vwapVolume = 0;
}

} // namespace btquant::data
