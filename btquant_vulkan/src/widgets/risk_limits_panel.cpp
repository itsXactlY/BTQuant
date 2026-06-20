#include "risk_limits_panel.hpp"

#include "../data/risk_guard.hpp"
#include "../data/position_book.hpp"
#include "log_panel.hpp"

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

double RiskLimitsPanel::editedMaxPositionSizeUSD() const {
    return parseOrZero(m_maxPos);
}
double RiskLimitsPanel::editedMaxLeverage() const {
    return parseOrZero(m_maxLev);
}
double RiskLimitsPanel::editedKillOnDailyLossUSD() const {
    return parseOrZero(m_killUSD);
}
double RiskLimitsPanel::editedEquityUSD() const {
    return parseOrZero(m_equity);
}

void RiskLimitsPanel::syncFromGuard() {
    if (!m_guard) return;
    const auto& c = m_guard->config();
    std::snprintf(m_maxPos,  sizeof(m_maxPos),  "%.2f", c.maxPositionSizeUSD);
    std::snprintf(m_maxLev,  sizeof(m_maxLev),  "%.2f", c.maxLeverage);
    std::snprintf(m_killUSD, sizeof(m_killUSD), "%.2f", c.killOnDailyLossUSD);
    std::snprintf(m_equity,  sizeof(m_equity),  "%.2f", c.equityUSD);
}

void RiskLimitsPanel::applyToGuard() {
    if (!m_guard) return;
    ::btquant::RiskConfig c = m_guard->config();
    c.maxPositionSizeUSD = editedMaxPositionSizeUSD();
    c.maxLeverage        = editedMaxLeverage();
    c.killOnDailyLossUSD = editedKillOnDailyLossUSD();
    c.equityUSD          = editedEquityUSD();
    m_guard->setConfig(c);
    BTQ_LOG_INFO("RiskLimits: applied caps $%.0f / %.2fx / kill $%.0f / eq $%.0f",
                 c.maxPositionSizeUSD, c.maxLeverage,
                 c.killOnDailyLossUSD, c.equityUSD);
}

void RiskLimitsPanel::render() {
    if (!m_open) return;

    // Lazy sync — first render pulls current values from the guard.
    static bool synced = false;
    if (!synced && m_guard) { syncFromGuard(); synced = true; }

    ImGui::SetNextWindowSize(ImVec2(440, 520), ImGuiCond_Appearing);
    if (!ImGui::Begin("Risk Dashboard", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!m_guard) {
        ImGui::TextDisabled("No RiskGuard bound — limits panel inert.");
        ImGui::End();
        return;
    }

    // ---- Status banner ----
    bool tripped = m_guard->isKillTripped();
    double remaining = m_guard->remainingLossBudget();
    double sessionReal = m_guard->sessionRealized();
    double fracLeft = m_guard->config().killOnDailyLossUSD > 0.0
                        ? std::max(0.0, remaining /
                                       m_guard->config().killOnDailyLossUSD)
                        : 0.0;

    ImVec4 bannerCol;
    const char* bannerText;
    if (tripped) {
        bannerCol = ImVec4(0.85f, 0.15f, 0.15f, 1.0f);
        bannerText = "KILL SWITCH TRIPPED — orders blocked";
    } else if (fracLeft < 0.20) {
        bannerCol = ImVec4(1.00f, 0.70f, 0.20f, 1.0f);
        bannerText = "WARNING — loss budget low";
    } else {
        bannerCol = ImVec4(0.20f, 0.70f, 0.30f, 1.0f);
        bannerText = "ARMED — risk checks active";
    }
    ImGui::PushStyleColor(ImGuiCol_Header,        bannerCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, bannerCol);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  bannerCol);
    ImGui::Selectable(bannerText, false,
                      ImGuiSelectableFlags_None);
    ImGui::PopStyleColor(3);

    ImGui::Separator();

    // ---- Session stats ----
    ImGui::Columns(2, "risk_stats", false);
    ImGui::SetColumnWidth(0, 180);
    ImGui::Text("Session realized"); ImGui::NextColumn();
    ImVec4 rc = sessionReal >= 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                                 : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(rc, "%s$%.2f",
                       sessionReal >= 0 ? "+" : "", sessionReal);
    ImGui::NextColumn();

    ImGui::Text("Remaining loss budget"); ImGui::NextColumn();
    ImVec4 bc = remaining > 0 ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                              : ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    ImGui::TextColored(bc, "$%.2f (%.0f%%)", remaining, fracLeft * 100.0);
    ImGui::NextColumn();

    ImGui::Text("Kill threshold"); ImGui::NextColumn();
    ImGui::Text("-$%.2f", m_guard->config().killOnDailyLossUSD);
    ImGui::NextColumn();

    // Visual progress bar — red as it fills toward kill.
    ImGui::Text("Budget used"); ImGui::NextColumn();
    float used = std::max(0.0f, std::min(1.0f, 1.0f - (float)fracLeft));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
        used > 0.8f ? ImVec4(0.85f, 0.15f, 0.15f, 1.0f)
        : used > 0.5f ? ImVec4(1.00f, 0.70f, 0.20f, 1.0f)
                     : ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
    ImGui::ProgressBar(used, ImVec2(-FLT_MIN, 0), "");
    ImGui::PopStyleColor();
    ImGui::Columns(1);

    ImGui::Separator();

    // ---- Current exposure ----
    ImGui::Text("Current exposure:");
    if (m_book && m_book->hasPosition()) {
        const auto& p = m_book->position();
        double notional = p.size * p.avgEntry;
        double fracNotional = m_guard->config().maxPositionSizeUSD > 0.0
            ? notional / m_guard->config().maxPositionSizeUSD
            : 0.0;
        ImGui::BulletText("%s %.4f @ $%.2f = $%.2f (%.0f%% of cap)",
                          p.isLong ? "LONG" : "SHORT",
                          p.size, p.avgEntry, notional, fracNotional * 100.0);
    } else {
        ImGui::BulletText("flat (no open position)");
    }

    ImGui::Separator();

    // ---- Edit limits ----
    ImGui::Text("Limits (apply with the button):");
    ImGui::PushItemWidth(180);
    ImGui::InputText("Max position (USD)",   m_maxPos,  sizeof(m_maxPos));
    ImGui::InputText("Max leverage (x)",     m_maxLev,  sizeof(m_maxLev));
    ImGui::InputText("Kill on loss (USD)",   m_killUSD, sizeof(m_killUSD));
    ImGui::InputText("Account equity (USD)", m_equity,  sizeof(m_equity));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Apply")) {
        applyToGuard();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        syncFromGuard();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear session")) {
        m_guard->resetSession();
        BTQ_LOG_WARN("RiskLimits: session P&L cleared (kill-trip reset)");
    }

    ImGui::Separator();
    if (ImGui::Button("Apply conservative preset")) {
        m_guard->setConfig(::btquant::RiskConfig::conservative());
        syncFromGuard();
        BTQ_LOG_INFO("RiskLimits: switched to conservative preset");
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply aggressive preset")) {
        m_guard->setConfig(::btquant::RiskConfig::aggressive());
        syncFromGuard();
        BTQ_LOG_INFO("RiskLimits: switched to aggressive preset");
    }

    ImGui::End();
}

} // namespace btquant::ui
