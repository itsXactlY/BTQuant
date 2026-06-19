#ifndef BTQUANT_FOOTPRINT_WIDGET_HPP
#define BTQUANT_FOOTPRINT_WIDGET_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Cluster chart: shows bid/ask volume per price level within each candle.
// "x@y" cells — left of mid = bid (seller-initiated, isBuy=false), right
// of mid = ask (buyer-initiated, isBuy=true). Imbalance is visually
// obvious at a glance: cells with bid >> ask (left-heavy) are bearish,
// cells with ask >> bid (right-heavy) are bullish.
//
// Aggregates on-the-fly from snap.recent_trades. For larger datasets
// (10k+ trades) the MarketDataProcessor should grow a per-candle cluster
// aggregate; for now the rolling 256-trade window is enough to fill
// the last few candles at 10Hz producer rate.
class FootprintWidget {
public:
    FootprintWidget();
    ~FootprintWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setPriceBucketTicks(int ticks) { m_priceBucketTicks = ticks; }
    void setCandleCount(int n) { m_candlesShown = n; }

    bool showWindow = true;

private:
    int m_priceBucketTicks = 10;   // bucket size in "ticks" (price unit)
    int m_candlesShown = 5;
    bool m_initialized = false;

    ::btquant::MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
