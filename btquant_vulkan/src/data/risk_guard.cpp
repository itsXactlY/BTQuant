#include "risk_guard.hpp"

#include <algorithm>
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
    return checkOrder(qty, price, false, std::string{});
}

std::optional<std::string>
RiskGuard::checkOrder(double qty, double price, bool /*isLong*/,
                      const std::string& symbol) const {
    if (qty <= 0.0 || price <= 0.0) {
        return std::string("invalid order: qty and price must be positive");
    }
    if (isKillTripped()) {
        return killReason(m_sessionRealized, m_cfg.killOnDailyLossUSD);
    }
    double notional = qty * price;
    // Global notional cap first. The per-symbol check below only
    // fires if the order's notional ALSO exceeds the (possibly
    // tighter) per-symbol override.
    if (notional > m_cfg.maxPositionSizeUSD) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "notional cap: $%.2f exceeds maxPositionSizeUSD $%.2f",
            notional, m_cfg.maxPositionSizeUSD);
        return std::string(buf);
    }
    // Per-symbol override — only when the symbol is non-empty AND
    // an override exists AND it's stricter than the global cap
    // (which it usually is). If the override is wider than the
    // global cap, the global already covers it and we don't
    // double-warn.
    if (!symbol.empty()) {
        auto it = m_maxOrderNotionalBySymbol.find(symbol);
        if (it != m_maxOrderNotionalBySymbol.end() &&
            it->second > 0.0 &&
            it->second < m_cfg.maxPositionSizeUSD &&
            notional > it->second) {
            char buf[240];
            std::snprintf(buf, sizeof(buf),
                "per-symbol notional cap: $%.2f exceeds %s cap $%.2f "
                "(global cap $%.2f)",
                notional, symbol.c_str(),
                it->second, m_cfg.maxPositionSizeUSD);
            return std::string(buf);
        }
    }
    double lev = effectiveLeverage(notional, m_cfg.equityUSD);
    if (lev > m_cfg.maxLeverage) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "leverage cap: %.2fx exceeds maxLeverage %.2fx "
            "(equity $%.2f)",
            lev, m_cfg.maxLeverage, m_cfg.equityUSD);
        return std::string(buf);
    }
    return std::nullopt;
}

void RiskGuard::setMaxOrderNotionalUSDForSymbol(const std::string& sym,
                                                 double usd) {
    if (sym.empty()) return;  // empty key is never meaningful
    if (usd <= 0.0) {
        // ≤0 = "clear the override". Equivalent to calling
        // clearMaxOrderNotionalUSDForSymbol(sym) but lets the caller
        // chain symmetrically: "set this cap" without first asking
        // whether it already exists.
        m_maxOrderNotionalBySymbol.erase(sym);
        return;
    }
    m_maxOrderNotionalBySymbol[sym] = usd;
}

void RiskGuard::clearMaxOrderNotionalUSDForSymbol(const std::string& sym) {
    m_maxOrderNotionalBySymbol.erase(sym);
}

double RiskGuard::maxOrderNotionalUSDForSymbol(const std::string& sym) const {
    auto it = m_maxOrderNotionalBySymbol.find(sym);
    if (it == m_maxOrderNotionalBySymbol.end() || it->second <= 0.0)
        return m_cfg.maxPositionSizeUSD;
    return it->second;
}

bool RiskGuard::hasMaxOrderNotionalUSDForSymbol(const std::string& sym) const {
    auto it = m_maxOrderNotionalBySymbol.find(sym);
    return it != m_maxOrderNotionalBySymbol.end() && it->second > 0.0;
}

std::vector<std::pair<std::string, double>>
RiskGuard::maxOrderNotionalBySymbol() const {
    std::vector<std::pair<std::string, double>> out;
    out.reserve(m_maxOrderNotionalBySymbol.size());
    for (const auto& kv : m_maxOrderNotionalBySymbol) {
        if (kv.second > 0.0) out.emplace_back(kv.first, kv.second);
    }
    // Alphabetical for stable panel layout (B, E, S → BTCUSDT,
    // ETHUSDT, SOLUSDT) — unlike the per-symbol P&L breakdown
    // (where |contribution| DESC is the useful sort because the
    // bleeder should be first), here the user wants to FIND a
    // specific symbol, not see the worst one.
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, double>& a,
                 const std::pair<std::string, double>& b) {
                  return a.first < b.first;
              });
    return out;
}

void RiskGuard::addRealized(double delta) {
    m_sessionRealized += delta;
}

void RiskGuard::addRealized(double delta, const std::string& symbol) {
    m_sessionRealized += delta;
    if (symbol.empty()) return;  // empty key = 1-arg-style update only
    auto it = m_sessionRealizedBySymbol.find(symbol);
    if (it == m_sessionRealizedBySymbol.end()) {
        m_sessionRealizedBySymbol.emplace(symbol, delta);
        m_symbolOrder.push_back(symbol);
    } else {
        it->second += delta;
    }
}

void RiskGuard::resetSession() {
    m_sessionRealized = 0.0;
    m_sessionRealizedBySymbol.clear();
    m_symbolOrder.clear();
}

double RiskGuard::sessionRealizedFor(const std::string& symbol) const {
    auto it = m_sessionRealizedBySymbol.find(symbol);
    if (it == m_sessionRealizedBySymbol.end()) return 0.0;
    return it->second;
}

std::vector<std::pair<std::string, double>>
RiskGuard::sessionRealizedBySymbol() const {
    std::vector<std::pair<std::string, double>> out;
    out.reserve(m_sessionRealizedBySymbol.size());
    for (const auto& kv : m_sessionRealizedBySymbol) {
        out.emplace_back(kv.first, kv.second);
    }
    // Sort by absolute contribution DESC so the biggest bleeder is
    // first. Ties broken by symbol name (stable, deterministic for
    // tests).
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, double>& a,
                 const std::pair<std::string, double>& b) {
                  double aa = std::fabs(a.second);
                  double bb = std::fabs(b.second);
                  if (aa != bb) return aa > bb;
                  return a.first < b.first;
              });
    return out;
}

std::vector<std::string>
RiskGuard::symbolsBookedThisSession() const {
    return m_symbolOrder;
}

bool RiskGuard::isKillTripped() const {
    return m_sessionRealized <= -m_cfg.killOnDailyLossUSD;
}

} // namespace btquant
