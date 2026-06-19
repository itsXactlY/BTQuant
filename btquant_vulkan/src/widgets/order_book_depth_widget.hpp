#ifndef BTQUANT_ORDER_BOOK_DEPTH_WIDGET_HPP
#define BTQUANT_ORDER_BOOK_DEPTH_WIDGET_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Ladder-style order book with depth bars at each level and a top-of-book
// imbalance gauge. Reads OrderBook directly from MarketDataProcessor.
//
// Differences from OrderBookWidget:
//   * Depth bars per price level (size-proportional horizontal bars).
//   * Cumulative-depth column on the side (running total of sizes).
//   * Top-of-book highlight (best bid / best ask in bold).
//   * Imbalance gauge showing (bid_vol - ask_vol) / (bid_vol + ask_vol) %.
//   * Volume-weighted mid (microprice) instead of simple mid.
class OrderBookDepthWidget {
public:
    OrderBookDepthWidget();
    ~OrderBookDepthWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setLevels(int levels) { m_levels = levels; }
    void setAutoCenterTopOfBook(bool v) { m_autoCenter = v; }

    bool showWindow = true;

private:
    int m_levels = 15;
    bool m_autoCenter = true;
    bool m_initialized = false;

    ::btquant::MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
