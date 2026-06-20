#ifndef BTQUANT_RISK_GUARD_HPP
#define BTQUANT_RISK_GUARD_HPP

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace btquant {

// Pre-trade risk guard. Pure-function check + session P&L tracker.
// Default config is conservative for a retail account:
//
//   maxPositionSizeUSD : 100,000  // single-order notional cap
//   maxLeverage        : 10       // notional / equity ceiling
//   killOnDailyLossUSD : 5,000    // flatten when session losses hit this
//   equityUSD          : 10,000   // assumed account equity for leverage
//
// checkOrder() returns std::nullopt on accept, or a human-readable error
// message on reject. WindowManager's OrderTicket callback uses the
// result to either apply the fill or surface the rejection via the
// LogPanel.
struct RiskConfig {
    double maxPositionSizeUSD = 100000.0;
    double maxLeverage        = 10.0;
    double killOnDailyLossUSD = 5000.0;
    double equityUSD          = 10000.0;

    static RiskConfig conservative() { return RiskConfig{}; }
    static RiskConfig aggressive() {
        RiskConfig c;
        c.maxPositionSizeUSD = 1'000'000.0;
        c.maxLeverage        = 50.0;
        c.killOnDailyLossUSD = 25'000.0;
        c.equityUSD          = 10'000.0;
        return c;
    }
};

class RiskGuard {
public:
    explicit RiskGuard(RiskConfig cfg = RiskConfig{});

    // Pre-trade check. side=true → buy (long), side=false → sell (short).
    // The check covers notional cap + leverage cap; existing-position
    // concerns (e.g. would this flip past max?) are the caller's job.
    std::optional<std::string>
    checkOrder(double qty, double price, bool isLong) const;

    // Symbol-aware overload. Same checks as the 3-arg form, plus an
    // optional per-symbol notional cap. When a per-symbol cap has been
    // registered via setMaxOrderNotionalUSDForSymbol(), the order's
    // notional is also tested against that cap. When no per-symbol
    // override exists, the global maxPositionSizeUSD is the only
    // notional limit (3-arg behaviour).
    //
    // Empty symbol = same as 3-arg form (no per-symbol check).
    std::optional<std::string>
    checkOrder(double qty, double price, bool isLong,
               const std::string& symbol) const;

    // Session P&L tracking. Add a realized delta (positive for wins,
    // negative for losses). The kill switch trips when the cumulative
    // session realized drops to or below -killOnDailyLossUSD.
    //
    // The 1-arg form (no symbol) updates only the aggregate total —
    // useful for non-trade P&L (e.g. manual journal entries that
    // pre-date the per-symbol feature, or synthetic bookkeeping). It
    // is intentionally preserved for back-compat with the original
    // API.
    //
    // The 2-arg form (with symbol) updates BOTH the aggregate total
    // AND the per-symbol bucket. The aggregate always equals the sum
    // of per-symbol buckets when only the 2-arg form has been used,
    // so RiskPanel can display the per-symbol breakdown as the source
    // of truth for which position is killing the kill budget.
    void addRealized(double delta);
    void addRealized(double delta, const std::string& symbol);

    void resetSession();

    bool isKillTripped() const;
    double sessionRealized() const { return m_sessionRealized; }
    // How much more session loss the account can absorb before the kill
    // switch trips. Starts at killOnDailyLossUSD and shrinks as
    // sessionRealized goes negative. Goes to ≤ 0 once tripped.
    double remainingLossBudget() const {
        return m_cfg.killOnDailyLossUSD + m_sessionRealized;
    }

    // Per-symbol session realized. Returns 0 for symbols that have
    // not been booked against (we never throw — the panel must keep
    // rendering even when one symbol has no contribution yet).
    double sessionRealizedFor(const std::string& symbol) const;

    // Per-symbol breakdown, sorted by absolute contribution DESCENDING
    // so the biggest bleeders surface at the top of the RiskPanel
    // readout. Each pair is {symbol, signed_pnl}. The vector is a
    // snapshot copy — safe to iterate without holding a lock.
    std::vector<std::pair<std::string, double>>
    sessionRealizedBySymbol() const;

    // Symbol names that have ever booked a delta this session, in
    // insertion order (insertion = first delta for that symbol).
    // Useful when the panel wants to display a stable column order
    // rather than re-sort on every frame.
    std::vector<std::string> symbolsBookedThisSession() const;

    const RiskConfig& config() const { return m_cfg; }
    void setConfig(const RiskConfig& c) {
        m_cfg = c;
        // Widen/tighten the kill threshold silently — caller's call.
    }

    // ---- Per-symbol notional caps ----
    //
    // Override the global maxPositionSizeUSD for a single symbol.
    // Typical use: cap BTCUSDT at $250k but leave the rest of the
    // book at the global $100k. usd <= 0 clears the override.
    // Overrides survive setConfig() (changing the global cap does
    // not touch the symbol-specific overrides — they're independent
    // levers). Pass usd <= 0 (or use clearMaxOrderNotionalUSDForSymbol)
    // to drop back to the global cap.
    void setMaxOrderNotionalUSDForSymbol(const std::string& sym, double usd);
    void clearMaxOrderNotionalUSDForSymbol(const std::string& sym);

    // Returns the override if one exists, otherwise the global
    // maxPositionSizeUSD. Callers that want to distinguish "no
    // override" from "explicit zero" should use hasMaxOrderNotionalUSDForSymbol.
    double maxOrderNotionalUSDForSymbol(const std::string& sym) const;

    bool hasMaxOrderNotionalUSDForSymbol(const std::string& sym) const;

    // Snapshot of {symbol, cap} pairs, sorted alphabetically by symbol
    // for stable UI rendering. Caps ≤ 0 are not included — those are
    // functionally equivalent to "no override".
    std::vector<std::pair<std::string, double>>
    maxOrderNotionalBySymbol() const;

    // ---- Pure math (test surface) ----

    // Compute effective leverage for a given notional. Returns
    // notional / equity, or 0 if equity <= 0.
    static double effectiveLeverage(double notionalUSD, double equityUSD);

    // Format a kill-switch reason string for logging.
    static std::string killReason(double sessionRealized,
                                  double killOnDailyLossUSD);

private:
    RiskConfig m_cfg;
    double     m_sessionRealized = 0.0;
    // Per-symbol session realized. Keyed by symbol; value is the
    // signed running total of addRealized(delta, symbol) calls this
    // session. Order is preserved separately in m_symbolOrder so the
    // panel can render a stable layout.
    std::unordered_map<std::string, double> m_sessionRealizedBySymbol;
    std::vector<std::string>                m_symbolOrder;

    // Per-symbol notional caps. Keyed by symbol; value is the override
    // for maxPositionSizeUSD when checking that symbol's orders.
    // Absent key = fall back to global cap. setConfig() doesn't touch
    // this map (per-symbol overrides are a separate lever from the
    // global notional cap).
    std::unordered_map<std::string, double> m_maxOrderNotionalBySymbol;
};

} // namespace btquant

#endif
