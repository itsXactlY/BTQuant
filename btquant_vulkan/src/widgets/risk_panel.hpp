#ifndef BTQUANT_RISK_PANEL_HPP
#define BTQUANT_RISK_PANEL_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Position + P&L tracker. Reads recent_trades and computes:
//   * Net position (signed contracts: positive = long, negative = short).
//   * Average entry price of the open position.
//   * Realized P&L from closed portions of round-trip trades.
//   * Unrealized P&L = (mark - avg_entry) × net_position at the last price.
//   * Max absolute position seen in the rolling window.
//   * Approximate daily P&L (sum of realized P&L across the window).
//
// Pulls from snap.recent_trades — no new data model. Position state is
// purely derived on-the-fly from the trade history.
class RiskPanel {
public:
    RiskPanel();
    ~RiskPanel();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    bool showWindow = true;

private:
    bool m_initialized = false;

    ::btquant::MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
