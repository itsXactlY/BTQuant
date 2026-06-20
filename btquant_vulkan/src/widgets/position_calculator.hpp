#ifndef BTQUANT_POSITION_CALCULATOR_HPP
#define BTQUANT_POSITION_CALCULATOR_HPP

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Position Calculator widget — trader-facing form for sizing new orders.
// Inputs:
//   - Account equity (USD)
//   - Risk per trade (% of equity)
//   - Entry price
//   - Stop-loss price
//   - Take-profit price (optional)
//
// Outputs:
//   - Position size in base units
//   - Notional value
//   - Risk amount (USD)
//   - R:R ratio
//   - P&L scenarios (1R, 2R, target)
//
// Hot-recalc mode: when a MarketDataProcessor is bound and auto-update
// is on, the entry price is overwritten each frame from the latest
// trade on the live symbol. The user can lock the entry by toggling
// auto-update off (or by manually editing — manual edits set the lock
// implicitly via the next live-snapshot boundary, see render()).
class PositionCalculator {
public:
    void render();

    // Bind a live data source for hot-recalc. Passing nullptr
    // disconnects (auto-update is then a no-op).
    void setMarketData(::btquant::MarketDataProcessor* data);

    // Toggle: when true (default), entry price tracks the live last
    // trade price from the bound MarketDataProcessor. Manual edits
    // are accepted but get overwritten on the next frame.
    void setAutoUpdateEntry(bool v) { m_autoUpdateEntry = v; }
    bool autoUpdateEntry() const   { return m_autoUpdateEntry; }

    // Test accessors.
    double computeSize    (double equity, double riskPct,
                           double entry, double stop) const;
    double computeNotional(double size, double price) const;
    double computeRR      (double entry, double stop, double target) const;

private:
    char m_equity  [32] = "10000.00";
    char m_riskPct [32] = "1.00";
    char m_entry   [32] = "67500.00";
    char m_stop    [32] = "67000.00";
    char m_target  [32] = "68500.00";
    char m_leverage[32] = "1.0";

    bool m_showHelp = true;
    bool m_autoUpdateEntry = true;
    // Cached latest price from the data spine — updated by render()
    // when auto-update is on. Public via getter for testing.
    double m_lastLivePrice = 0.0;

public:
    // Last live price the calculator observed (0.0 when no data).
    // Public so tests can assert the auto-update path populated the
    // entry field without driving the full render loop.
    double lastLivePrice() const { return m_lastLivePrice; }

private:
    class MarketDataProcessor* m_data = nullptr;
    void refreshLivePrice();
};

} // namespace btquant::ui

#endif
