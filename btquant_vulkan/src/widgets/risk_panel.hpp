#ifndef BTQUANT_RISK_PANEL_HPP
#define BTQUANT_RISK_PANEL_HPP

#include <vector>

#include "../data/market_data.hpp"

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Position + P&L tracker. Reads recent_trades and computes:
//   * Net position (signed contracts: positive = long, negative = short).
//   * Average entry price of the open position.
//   * Realized P&L from closed portions of round-trip trades.
//   * Unrealized P&L = (mark - avg_entry) × net_position at the last price.
//   * Max absolute position seen in the rolling window.
//   * Approximate daily P&L (sum of realized P&L across the window).
//   * Aggregate risk metrics: Sharpe (per-trade + sqrt(N) heuristic),
//     max drawdown on cumP&L, win rate, profit factor, expectancy.
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

// Plain-old-data structs exposed for testing.
struct PositionState {
    double netSize    = 0.0;
    double avgEntry   = 0.0;
    double realized   = 0.0;
    double maxAbsSize = 0.0;
};

struct RiskMetrics {
    int    tradeCount   = 0;
    int    buyCount     = 0;
    int    sellCount    = 0;
    double avgTradeSize = 0.0;
    double sharpePerTrade    = 0.0;
    double sharpeAnnualized  = 0.0;
    double maxDrawdown       = 0.0;
    double winRate           = 0.0;
    double profitFactor      = 0.0;
    double expectancy        = 0.0;
};

PositionState computePosition(const std::vector<::btquant::data::Trade>& trades);
RiskMetrics  computeMetrics (const std::vector<::btquant::data::Trade>& trades);

}  // namespace btquant::ui

#endif
