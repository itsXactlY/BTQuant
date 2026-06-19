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

struct Candle {
    double open = 0;
    double high = 0;
    double low = 0;
    double close = 0;
    double volume = 0;
    uint64_t startTime = 0;
    uint64_t endTime = 0;
};

struct Trade {
    uint64_t id = 0;
    double price = 0;
    double size = 0;
    uint64_t timestamp = 0;
    bool isBuy = true;
};

class MarketDataAggregator {
public:
    void update(const MarketTick& tick);
    void reset();

    [[nodiscard]] const OrderBook& orderBook() const { return m_orderBook; }
    [[nodiscard]] const MarketMetrics& metrics() const { return m_metrics; }
    [[nodiscard]] const std::vector<Trade>& trades() const { return m_trades; }

private:
    OrderBook m_orderBook;
    MarketMetrics m_metrics;
    std::vector<Trade> m_trades;
    uint64_t m_lastTimestamp = 0;
    double m_vwapSum = 0;
    double m_vwapVolume = 0;
};

} // namespace btquant::data

#endif
