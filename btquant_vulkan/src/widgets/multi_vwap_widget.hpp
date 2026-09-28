#ifndef BTQUANT_MULTI_VWAP_WIDGET_HPP
#define BTQUANT_MULTI_VWAP_WIDGET_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Multi-period VWAP — rolling volume-weighted average price at several
// lookback windows (50 / 100 / 200 / all-visible). Shown as:
//   * Horizontal lines on a small price chart (last N trades) so the
//     operator can see at a glance which VWAP price is acting as S/R.
//   * Numeric table with VWAP value, deviation from last price, and
//     volume covered.
//
// Pulls from snap.recent_trades — no extra data model needed.
class MultiVWAPWidget {
public:
    MultiVWAPWidget();
    ~MultiVWAPWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    bool showWindow = true;

private:
    bool m_initialized = false;

    ::btquant::MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
