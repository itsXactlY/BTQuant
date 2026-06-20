#include "position_calculator.hpp"
#include "../data/market_data_processor.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <imgui.h>

namespace btquant::ui {

namespace {
double parseOrZero(const char* s) {
    if (!s || !*s) return 0.0;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    return (end == s) ? 0.0 : v;
}
} // namespace

void PositionCalculator::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void PositionCalculator::resetToDefaults() {
    // Same values as the field-initializers in the header. The
    // strings are snprintf'd so trailing-zero precision is preserved.
    std::snprintf(m_equity,   sizeof(m_equity),   "10000.00");
    std::snprintf(m_riskPct,  sizeof(m_riskPct),  "1.00");
    std::snprintf(m_entry,    sizeof(m_entry),    "67500.00");
    std::snprintf(m_stop,     sizeof(m_stop),     "67000.00");
    std::snprintf(m_target,   sizeof(m_target),   "68500.00");
    std::snprintf(m_leverage, sizeof(m_leverage), "1.0");
    m_lastLivePrice = 0.0;
}

void PositionCalculator::refreshLivePrice() {
    if (!m_data) return;
    auto snap = m_data->snapshot(1);
    if (snap.snapshot_seq == 0) return;
    if (snap.recent_trades.empty()) return;
    const auto& t = snap.recent_trades.back();
    if (t.price <= 0.0) return;
    m_lastLivePrice = t.price;
    if (m_autoUpdateEntry) {
        // Overwrite the entry field with the live price. We use
        // snprintf with enough precision for BTC-scale prices.
        std::snprintf(m_entry, sizeof(m_entry), "%.2f", t.price);
    }
}

double PositionCalculator::computeSize(double equity, double riskPct,
                                      double entry, double stop) const {
    if (equity <= 0.0 || riskPct <= 0.0) return 0.0;
    double riskUSD = equity * (riskPct / 100.0);
    double perUnitRisk = std::fabs(entry - stop);
    if (perUnitRisk <= 0.0) return 0.0;
    return riskUSD / perUnitRisk;
}

double PositionCalculator::computeNotional(double size, double price) const {
    return std::fabs(size) * price;
}

double PositionCalculator::computeRR(double entry, double stop, double target) const {
    double risk  = std::fabs(entry - stop);
    double reward = std::fabs(target - entry);
    if (risk <= 0.0) return 0.0;
    return reward / risk;
}

void PositionCalculator::render() {
    if (!ImGui::Begin("Position Calculator", nullptr,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    // Pull the latest trade price BEFORE the user edits the field.
    // If auto-update is on, the entry field below gets overwritten
    // with the live price; if off, the field stays at whatever the
    // user typed (or the last live price before they toggled).
    refreshLivePrice();

    ImGui::Text("Inputs (edit, results update live):");
    ImGui::PushItemWidth(160);
    ImGui::InputText("Equity (USD)",   m_equity,   sizeof(m_equity));
    ImGui::InputText("Risk per trade %", m_riskPct, sizeof(m_riskPct));
    ImGui::InputText("Entry price",     m_entry,    sizeof(m_entry));
    ImGui::InputText("Stop-loss price", m_stop,     sizeof(m_stop));
    ImGui::InputText("Take-profit price", m_target, sizeof(m_target));
    ImGui::InputText("Leverage (x)",    m_leverage, sizeof(m_leverage));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Checkbox("Show help", &m_showHelp);
    ImGui::SameLine();
    // Auto-update toggle — when on, the entry field above is
    // overwritten each frame with the live last-trade price. Shows
    // the live price next to the checkbox for context (only when
    // m_data is wired).
    ImGui::Checkbox("Auto-update entry", &m_autoUpdateEntry);
    if (m_lastLivePrice > 0.0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(live $%.2f)", m_lastLivePrice);
    }
    // Sprint #70: Reset button next to the auto-update toggle so
    // the trader can start a fresh calculation without clearing
    // each field by hand. Restores the constructor defaults for
    // equity / risk / entry / stop / target / leverage. Does NOT
    // touch auto-update or show-help (those are preferences).
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset")) {
        resetToDefaults();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Restore default equity / risk / entry / "
                          "stop / target / leverage values");
    }
    ImGui::Separator();

    double equity   = parseOrZero(m_equity);
    double riskPct  = parseOrZero(m_riskPct);
    double entry    = parseOrZero(m_entry);
    double stop     = parseOrZero(m_stop);
    double target   = parseOrZero(m_target);
    double leverage = parseOrZero(m_leverage);
    if (leverage <= 0.0) leverage = 1.0;

    double size      = computeSize(equity, riskPct, entry, stop);
    double notional  = computeNotional(size, entry);
    double riskUSD   = equity * (riskPct / 100.0);
    double rr        = computeRR(entry, stop, target);

    ImGui::Columns(2, "PosCalc", false);
    ImGui::SetColumnWidth(0, 200);

    ImGui::Text("Position size (base)"); ImGui::NextColumn();
    if (size > 0.0) ImGui::Text("%.6f", size); else ImGui::TextDisabled("—");
    ImGui::NextColumn();

    ImGui::Text("Notional (USD)");       ImGui::NextColumn();
    if (notional > 0.0) ImGui::Text("$%.2f", notional);
    else                ImGui::TextDisabled("—");
    ImGui::NextColumn();

    ImGui::Text("Risk amount (USD)");    ImGui::NextColumn();
    ImGui::Text("$%.2f", riskUSD); ImGui::NextColumn();

    ImGui::Text("R:R ratio");            ImGui::NextColumn();
    if (rr > 0.0) {
        ImVec4 col = rr >= 2.0 ? ImVec4(0.30f, 0.95f, 0.40f, 1.0f)
                     : rr >= 1.0 ? ImVec4(1.00f, 0.85f, 0.30f, 1.0f)
                                  : ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
        ImGui::TextColored(col, "%.2f R", rr);
    } else {
        ImGui::TextDisabled("—");
    }
    ImGui::NextColumn();

    ImGui::Text("Effective leverage");   ImGui::NextColumn();
    if (size > 0.0 && equity > 0.0) {
        double effLev = notional / equity;
        ImVec4 col = effLev > leverage * 1.01f ? ImVec4(1.0f, 0.5f, 0.3f, 1.0f)
                                                : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::TextColored(col, "%.2fx (input %.2fx)", effLev, leverage);
    } else {
        ImGui::TextDisabled("—");
    }
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Separator();
    ImGui::Text("P&L scenarios:");

    if (size > 0.0 && entry > 0.0) {
        // 1R = risk amount. 2R = double profit.
        // Also show actual target profit.
        double oneR        = riskUSD;
        double twoR        = riskUSD * 2.0;
        double targetProfit = (target > entry ? 1.0 : -1.0) *
                              size * std::fabs(target - entry);

        ImGui::Columns(4, "scenarios", false);
        ImGui::Text("1R win"); ImGui::NextColumn();
        ImGui::Text("2R win"); ImGui::NextColumn();
        ImGui::Text("Target win"); ImGui::NextColumn();
        ImGui::Text("Stop loss"); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.30f, 0.95f, 0.40f, 1.0f), "+$%.2f", oneR);
        ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.30f, 0.95f, 0.40f, 1.0f), "+$%.2f", twoR);
        ImGui::NextColumn();
        ImVec4 tcol = targetProfit >= 0 ? ImVec4(0.30f, 0.95f, 0.40f, 1.0f)
                                        : ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
        ImGui::TextColored(tcol, "%s$%.2f",
                           targetProfit >= 0 ? "+" : "", targetProfit);
        ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.95f, 0.30f, 0.30f, 1.0f), "-$%.2f", riskUSD);
        ImGui::Columns(1);
    } else {
        ImGui::TextDisabled("Set equity, risk %, entry, and stop to see scenarios.");
    }

    if (m_showHelp) {
        ImGui::Separator();
        ImGui::TextWrapped(
            "Position size = (equity × risk%) / |entry - stop|. "
            "If R:R < 1, the trade has more downside than upside relative to "
            "your stop. Effective leverage = notional / equity; if it exceeds "
            "your input leverage, you're over-sizing.");
    }

    ImGui::End();
}

} // namespace btquant::ui
