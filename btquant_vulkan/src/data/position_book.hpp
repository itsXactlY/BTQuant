#ifndef BTQUANT_POSITION_BOOK_HPP
#define BTQUANT_POSITION_BOOK_HPP

#include <optional>
#include <string>

namespace btquant {

// Single-symbol position book. Tracks the open position for the currently
// active symbol and accumulates realized P&L as fills close/reduce it.
//
// Lifecycle:
//   1. OrderTicket submit callback calls fill(side, qty, price)
//   2. fill() averages-in same-direction, reduces-and-realizes on opposite
//   3. markToMarket(currentPrice) refreshes unrealizedPnL from snapshots
//
// Math is side-aware: long positions profit when price rises, shorts the
// inverse. A "fill" is always positive quantity (the sign of side carries
// direction). All P&L is in quote currency (USD for USDT pairs).
struct Position {
    std::string symbol;
    bool   isLong      = true;
    double size        = 0.0;     // absolute quantity in base units
    double avgEntry    = 0.0;     // weighted-average entry price
    double realizedPnL = 0.0;     // closed-trade cumulative P&L (USD)
    double unrealizedPnL = 0.0;   // mark-to-market P&L on the open position
    int    fillCount   = 0;       // number of fills applied
};

class PositionBook {
public:
    // Apply a fill. side=true → long (BUY), side=false → short (SELL).
    // qty must be > 0. price must be > 0.
    // Returns the realized P&L delta from this fill (0 if it just opens or
    // averages in).
    double fill(const std::string& symbol, bool isLong,
                double qty, double price);

    // Recompute unrealizedPnL against the latest mark. Returns the value
    // stored after the update so callers can render it without a separate
    // getter call.
    double markToMarket(double currentPrice);

    // Force-flatten at the given price — used by the kill-switch and on
    // symbol swap. Realizes any remaining open P&L into realizedPnL.
    double flatten(double price);

    // ---- Accessors ----
    bool   hasPosition() const { return m_pos.size > 0.0; }
    const Position& position() const { return m_pos; }
    double realizedPnL()   const { return m_pos.realizedPnL; }
    double unrealizedPnL() const { return m_pos.unrealizedPnL; }
    double totalPnL()      const {
        return m_pos.realizedPnL + m_pos.unrealizedPnL;
    }
    int    fillCount() const { return m_pos.fillCount; }

    // Reset to empty (does not touch realizedPnL — caller decides).
    void clear() { m_pos = Position{}; }
    void clearAll() { m_pos = Position{}; m_realizedClosed = 0.0; }

    // Session-aggregate realized P&L (includes closed portions even after
    // the position is flat). Resets only via clearAll().
    double sessionRealized() const { return m_realizedClosed; }

    // ---- Pure math (test surface) ----

    // Compute the new average entry when averaging into an existing
    // long (or short) position. Returns the updated avgEntry.
    static double averageEntry(double oldSize, double oldAvg,
                               double fillQty,  double fillPrice);

    // Compute realized P&L on a reducing fill. side=true means the fill is
    // in the same direction as the open position (BUY into a long), so a
    // reducing close is SELL — handled by caller flipping the sign.
    // Returns (realized_delta, new_size, new_avg). If closeQty >= size
    // the position is fully closed and new_size = 0, new_avg = 0.
    struct CloseResult {
        double realizedDelta;
        double newSize;
        double newAvg;
    };
    static CloseResult applyClose(double openSize, double openAvg,
                                  double closeQty, double closePrice,
                                  bool   openIsLong);

private:
    Position m_pos;
    double   m_realizedClosed = 0.0;  // accumulated even after flatten
};

} // namespace btquant

#endif
