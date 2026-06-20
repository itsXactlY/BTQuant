#ifndef BTQUANT_RISK_GUARD_HPP
#define BTQUANT_RISK_GUARD_HPP

#include <optional>
#include <string>

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

    // Session P&L tracking. Add a realized delta (positive for wins,
    // negative for losses). The kill switch trips when the cumulative
    // session realized drops to or below -killOnDailyLossUSD.
    void addRealized(double delta);
    void resetSession();

    bool isKillTripped() const;
    double sessionRealized() const { return m_sessionRealized; }
    // How much more session loss the account can absorb before the kill
    // switch trips. Starts at killOnDailyLossUSD and shrinks as
    // sessionRealized goes negative. Goes to ≤ 0 once tripped.
    double remainingLossBudget() const {
        return m_cfg.killOnDailyLossUSD + m_sessionRealized;
    }

    const RiskConfig& config() const { return m_cfg; }
    void setConfig(const RiskConfig& c) {
        m_cfg = c;
        // Widen/tighten the kill threshold silently — caller's call.
    }

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
};

} // namespace btquant

#endif
