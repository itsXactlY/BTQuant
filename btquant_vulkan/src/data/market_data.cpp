#include "market_data.hpp"

namespace btquant::data {

void MarketDataAggregator::update(const MarketTick& tick) {
    m_trades.push_back({m_trades.size(), tick.price, tick.size, tick.timestamp, tick.isBuy});

    if (tick.isBuy) {
        m_metrics.buyVolume += tick.size;
    } else {
        m_metrics.sellVolume += tick.size;
    }
    m_metrics.volume += tick.size;
    m_metrics.delta = m_metrics.buyVolume - m_metrics.sellVolume;
    m_metrics.tradeCount++;

    m_vwapSum += tick.price * tick.size;
    m_vwapVolume += tick.size;
    if (m_vwapVolume > 0) {
        m_metrics.vwap = m_vwapSum / m_vwapVolume;
    }

    if (tick.price > m_metrics.high || m_metrics.high == 0) m_metrics.high = tick.price;
    if (tick.price < m_metrics.low || m_metrics.low == 0) m_metrics.low = tick.price;

    m_lastTimestamp = tick.timestamp;
}

void MarketDataAggregator::reset() {
    m_metrics = {};
    m_trades.clear();
    m_vwapSum = 0;
    m_vwapVolume = 0;
    m_lastTimestamp = 0;
}

} // namespace btquant::data
