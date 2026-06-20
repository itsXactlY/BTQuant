#ifndef BTQUANT_POSITION_CALCULATOR_HPP
#define BTQUANT_POSITION_CALCULATOR_HPP

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
// Pure math — no external data dependencies.
class PositionCalculator {
public:
    void render();

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
};

} // namespace btquant::ui

#endif
