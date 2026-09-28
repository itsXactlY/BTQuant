#ifndef BTQUANT_RISK_GUARD_HPP
#define BTQUANT_RISK_GUARD_HPP

#include <chrono>
#include <ctime>
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
    // Symbol-aware kill check. Returns true when EITHER the global
    // session losses have tripped the global threshold OR the
    // per-symbol session losses for `symbol` have tripped that
    // symbol's override (if any). The global threshold is always
    // the backstop — per-symbol overrides tighten but never
    // loosen the kill switch.
    bool isKillTrippedForSymbol(const std::string& symbol) const;
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

    // ---- Session-realized history (Sprint #61) ----
    //
    // Bounded time-series of sessionRealized() snapshots for the
    // RiskPanel equity-curve sparkline. WindowManager calls
    // sampleSessionRealized() each render frame; the guard only
    // pushes to history when the value actually changes (so idle
    // frames don't pollute the series). Capped at kMaxRealizedHistory
    // (1000) — at typical 60fps that's ~16s of dense sampling, but
    // since we dedupe on change, the cap is more of a long-session
    // safety than an active limit. Old samples are evicted FIFO.
    static constexpr std::size_t kMaxRealizedHistory = 1000;

    // Push a snapshot of sessionRealized() to the history if it
    // differs from the last sample. No-op when value unchanged.
    void sampleSessionRealized();

    // Snapshot of session-realized samples, oldest first. The
    // sparkline renderer reads this directly.
    const std::vector<double>& sessionRealizedHistory() const {
        return m_realizedHistory;
    }

    // Clear the history (used by tests + the "Reset session" UI
    // button when the trader wants to start fresh).
    void clearSessionRealizedHistory() { m_realizedHistory.clear(); }

    // ---- Session-day tracking (Sprint #69) ----
    //
    // The "session" tracked by sessionRealized is normally aligned
    // with the trader's working day — i.e. resets at local midnight,
    // not on app restart. These accessors let WindowManager detect
    // when the wall-clock day has rolled over since the last reset
    // (or first construction) and call resetSession() automatically.
    //
    // Wall-clock time, not monotonic — we explicitly want to react
    // to the calendar day changing, not just elapsed seconds.
    using Clock     = std::chrono::system_clock;
    using TimePoint = Clock::time_point;

    // When the current session started (last manual reset OR last
    // auto-reset-at-midnight, whichever is later). Construction
    // captures Clock::now() so the first session starts at app boot.
    TimePoint sessionStartTime() const { return m_sessionStartTime; }

    // True when the current wall-clock local day differs from the
    // session start's local day. Wall-clock midnight rollover =
    // a new trading day for the guard. Time-of-day doesn't matter
    // — only the calendar date in the local timezone.
    bool isNewSessionDay(const TimePoint& now = Clock::now()) const;

    // If isNewSessionDay(now), reset the session and stamp the
    // start time to `now`. Otherwise a no-op. Returns true if
    // a reset actually fired (caller can log it).
    bool autoResetIfNewDay(const TimePoint& now = Clock::now());

    // Test helpers — prime the session-start stamp directly so
    // tests can exercise the day-rollover path without driving
    // the system clock.
    void setSessionStartTimeForTest(const TimePoint& t) {
        m_sessionStartTime = t;
    }

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

    // ---- Per-symbol kill thresholds ----
    //
    // Override the global killOnDailyLossUSD for a single symbol.
    // Typical use: tighter kill on illiquid alt-coins (-$500) while
    // leaving the majors at the global -$5,000. The global threshold
    // is always the backstop — per-symbol overrides tighten, never
    // loosen. usd <= 0 clears the override.
    void setKillOnDailyLossUSDForSymbol(const std::string& sym, double usd);
    void clearKillOnDailyLossUSDForSymbol(const std::string& sym);

    double killOnDailyLossUSDForSymbol(const std::string& sym) const;
    bool   hasKillOnDailyLossUSDForSymbol(const std::string& sym) const;

    // Per-symbol remaining loss budget. Returns the override budget
    // (if set) or the global budget (otherwise). Mirrors the global
    // remainingLossBudget() formula but per symbol.
    double remainingLossBudgetForSymbol(const std::string& symbol) const;

    // Alpha-sorted snapshot of {symbol, kill_threshold} pairs.
    std::vector<std::pair<std::string, double>>
    killOnDailyLossBySymbol() const;

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

    // Per-symbol kill thresholds. Keyed by symbol; value is the
    // override for killOnDailyLossUSD when checking that symbol's
    // session losses. Absent key = use global threshold. setConfig()
    // does not touch this map.
    std::unordered_map<std::string, double> m_killOnDailyLossBySymbol;

    // Session-realized time-series for the RiskPanel sparkline
    // (Sprint #61). Oldest first. Capped at kMaxRealizedHistory.
    std::vector<double> m_realizedHistory;

    // Wall-clock time the current session started. Set on
    // construction; updated by resetSession() and the auto-reset-
    // at-midnight path in autoResetIfNewDay().
    TimePoint m_sessionStartTime;
};

} // namespace btquant

#endif
