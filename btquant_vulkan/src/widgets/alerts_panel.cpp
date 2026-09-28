#include "alerts_panel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <imgui.h>

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

namespace btquant::ui {

namespace {
constexpr double kEmaAlpha = 0.05;  // smoothing for avg-volume baseline
} // namespace

void AlertsPanel::render() {
    if (!showInStatusBar && !ImGui::Begin("Alerts", &showInStatusBar)) {
        ImGui::End();
        return;
    }
    if (m_data) {
        auto snap = m_data->snapshot(0, 0);
        if (m_startTimeSec == 0.0) {
            m_startTimeSec = static_cast<double>(ImGui::GetTime());
        }
        double now = static_cast<double>(ImGui::GetTime()) - m_startTimeSec;
        evaluateAndPush(snap, now);
    }

    ImGui::Text("Alerts (%zu)", m_alerts.size());
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) m_alerts.clear();
    ImGui::SameLine();
    ImGui::Checkbox("Sound", &soundEnabled);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("Δ price %", reinterpret_cast<float*>(&priceMovePctThreshold),
                       0.01f, 2.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderFloat("Vol ×avg", reinterpret_cast<float*>(&volumeSpikeMultiplier),
                       1.0f, 20.0f, "%.1f");

    ImGui::Separator();
    if (ImGui::BeginTable("alerts", 3, ImGuiTableFlags_RowBg |
                                            ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("t",   ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("sev", ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("message", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (auto it = m_alerts.rbegin(); it != m_alerts.rend(); ++it) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.1fs", it->timestampSec);
            ImGui::TableNextColumn();
            ImVec4 col = it->severity >= 2 ? ImVec4(1.0f, 0.30f, 0.30f, 1.0f)
                       : it->severity == 1 ? ImVec4(1.0f, 0.85f, 0.30f, 1.0f)
                                            : ImVec4(0.70f, 0.85f, 1.0f, 1.0f);
            ImGui::TextColored(col, "%s",
                it->severity >= 2 ? "CRIT" : it->severity == 1 ? "warn" : "info");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(it->message.c_str());
        }

        ImGui::EndTable();
    }

    if (showInStatusBar) ImGui::End();
}

void AlertsPanel::evaluateAndPush(const MarketDataProcessor::Snapshot& snap,
                                   double nowSec) {
    if (snap.recent_trades.empty()) return;
    const double price = snap.recent_trades.front().price;  // newest trade

    // First-frame seed: just record baseline.
    if (m_lastPrice == 0.0) {
        m_lastPrice   = price;
        m_lastPriceAt = nowSec;
        for (const auto& t : snap.recent_trades) {
            m_volumeEma = m_volumeEma == 0.0 ? std::fabs(t.size) : m_volumeEma;
            if (m_volumeEma == 0.0) m_volumeEma = std::fabs(t.size);
        }
        return;
    }

    const double dt = nowSec - m_lastPriceAt;
    if (dt < 0.5) {
        // Too soon — don't spam alerts. Re-baseline EMA though.
        for (const auto& t : snap.recent_trades) {
            m_volumeEma = m_volumeEma == 0.0 ? std::fabs(t.size)
                                              : m_volumeEma * (1.0 - kEmaAlpha)
                                              + std::fabs(t.size) * kEmaAlpha;
        }
        return;
    }

    // Update EMA with the trades that arrived since last eval.
    for (const auto& t : snap.recent_trades) {
        const double sz = std::fabs(t.size);
        if (m_volumeEma == 0.0) m_volumeEma = sz;
        else                    m_volumeEma = m_volumeEma * (1.0 - kEmaAlpha) + sz * kEmaAlpha;
        // Per-trade volume spike check.
        if (m_volumeEma > 0 && sz > m_volumeEma * volumeSpikeMultiplier) {
            Alert a;
            a.message      = "Vol spike " + formatVolume(sz)
                              + " (" + std::to_string(static_cast<int>(sz / m_volumeEma))
                              + "x avg) @ " + formatPrice(t.price);
            a.timestampSec = nowSec;
            a.severity     = 1;
            m_alerts.push_back(std::move(a));
            ++m_alertsEmitted;
            if (soundEnabled) std::fputc('\a', stderr);
        }
    }

    // Price-move alert (one per window).
    const double priceMovePct = std::fabs(price - m_lastPrice) / std::max(m_lastPrice, 1e-9) * 100.0;
    if (priceMovePct >= priceMovePctThreshold) {
        Alert a;
        a.message      = std::string(price > m_lastPrice ? "Price UP " : "Price DOWN ")
                          + std::to_string(static_cast<int>(priceMovePct * 100) / 100.0)
                          + "%  " + formatPrice(m_lastPrice) + " -> " + formatPrice(price);
        a.timestampSec = nowSec;
        a.severity     = priceMovePct >= priceMovePctThreshold * 3.0 ? 2 : 1;
        m_alerts.push_back(std::move(a));
        ++m_alertsEmitted;
        if (soundEnabled) std::fputc('\a', stderr);
    }

    // Roll baseline forward.
    m_lastPrice   = price;
    m_lastPriceAt = nowSec;

    // Trim.
    while (m_alerts.size() > kMaxAlerts) m_alerts.pop_front();
}

std::string AlertsPanel::formatPrice(double p) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", p);
    return buf;
}

std::string AlertsPanel::formatVolume(double v) {
    char buf[32];
    if (v >= 1.0) std::snprintf(buf, sizeof(buf), "%.2f", v);
    else          std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

} // namespace btquant::ui
