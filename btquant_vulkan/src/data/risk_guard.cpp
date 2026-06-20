#include "risk_guard.hpp"

#include <cmath>
#include <cstdio>

namespace btquant {

RiskGuard::RiskGuard(RiskConfig cfg) : m_cfg(cfg) {}

double RiskGuard::effectiveLeverage(double notionalUSD, double equityUSD) {
    if (equityUSD <= 0.0) return 0.0;
    return notionalUSD / equityUSD;
}

std::string RiskGuard::killReason(double sessionRealized,
                                  double killOnDailyLossUSD) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
        "kill switch tripped: session realized %s$%.2f ≤ -$%.2f limit",
        sessionRealized >= 0 ? "+" : "", sessionRealized,
        killOnDailyLossUSD);
    return std::string(buf);
}

std::optional<std::string>
RiskGuard::checkOrder(double qty, double price, bool /*isLong*/) const {
    if (qty <= 0.0 || price <= 0.0) {
        return std::string("invalid order: qty and price must be positive");
    }
    if (isKillTripped()) {
        return killReason(m_sessionRealized, m_cfg.killOnDailyLossUSD);
    }
    double notional = qty * price;
    if (notional > m_cfg.maxPositionSizeUSD) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "notional cap: $%.2f exceeds maxPositionSizeUSD $%.2f",
            notional, m_cfg.maxPositionSizeUSD);
        return std::string(buf);
    }
    double lev = effectiveLeverage(notional, m_cfg.equityUSD);
    if (lev > m_cfg.maxLeverage) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "leverage cap: %.2fx exceeds maxLeverage %.2fx "
            "(equity $%.2f)",
            lev, m_cfg.maxLeverage, m_cfg.equityUSD);
        return std::string(buf);
    }
    return std::nullopt;
}

void RiskGuard::addRealized(double delta) {
    m_sessionRealized += delta;
}

void RiskGuard::resetSession() {
    m_sessionRealized = 0.0;
}

bool RiskGuard::isKillTripped() const {
    return m_sessionRealized <= -m_cfg.killOnDailyLossUSD;
}

} // namespace btquant
