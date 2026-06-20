#include "equity_curve_panel.hpp"

#include "imgui.h"

#include "../data/trade_journal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>

namespace btquant::ui {

namespace {

// Pick a label for the X axis (timestamp → "MM-DD HH:MM" or
// just "MM-DD"). The axis granularity auto-zooms with the
// time range covered by the data.
const char* xLabel(char* buf, size_t bufsz,
                   uint64_t ts_us, double span_secs) {
    std::time_t secs = static_cast<std::time_t>(ts_us / 1000000ULL);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &secs);
#else
    localtime_r(&secs, &tm);
#endif
    if (span_secs < 86400.0) {
        // Sub-day range — show HH:MM.
        std::strftime(buf, bufsz, "%H:%M", &tm);
    } else if (span_secs < 86400.0 * 90) {
        // Sub-quarter range — show MM-DD.
        std::strftime(buf, bufsz, "%m-%d", &tm);
    } else {
        // Year+ range — show YYYY-MM.
        std::strftime(buf, bufsz, "%Y-%m", &tm);
    }
    return buf;
}

}  // namespace

void EquityCurvePanel::render() {
    if (!m_journal) {
        ImGui::TextDisabled("Equity Curve: journal not wired");
        return;
    }

    auto eq    = m_journal->equityCurve();
    auto dd    = m_journal->equityDrawdownSeries();
    if (eq.empty()) {
        ImGui::TextDisabled("Equity Curve: no fills yet");
        return;
    }

    // Optional point cap for huge journals.
    if (m_maxPoints > 0 && eq.size() > m_maxPoints) {
        // Keep the LATEST m_maxPoints — the recent view is what
        // matters for live trading.
        size_t drop = eq.size() - m_maxPoints;
        eq.erase(eq.begin(), eq.begin() + drop);
        dd.erase(dd.begin(), dd.begin() + drop);
    }

    // ---- Compute ranges ----
    double eqMin   = eq.front().cumulative;
    double eqMax   = eq.front().cumulative;
    double ddMax   = 0.0;
    uint64_t tMin  = eq.front().timestamp_us;
    uint64_t tMax  = eq.front().timestamp_us;
    for (const auto& p : eq) {
        if (p.cumulative < eqMin) eqMin = p.cumulative;
        if (p.cumulative > eqMax) eqMax = p.cumulative;
        if (p.timestamp_us < tMin) tMin = p.timestamp_us;
        if (p.timestamp_us > tMax) tMax = p.timestamp_us;
    }
    for (const auto& p : dd) {
        if (p.drawdown > ddMax) ddMax = p.drawdown;
    }
    // Pad the equity Y range so the curve doesn't touch the top
    // edge of the chart frame.
    double eqPad = (eqMax - eqMin) * 0.05;
    if (eqPad < 1e-9) eqPad = 1.0;
    eqMin -= eqPad;
    eqMax += eqPad;
    double span_secs =
        static_cast<double>(tMax - tMin) / 1000000.0;
    if (span_secs < 1.0) span_secs = 1.0;

    // ---- Header summary ----
    double finalEq   = eq.back().cumulative;
    double maxDD     = 0.0;
    for (const auto& p : dd) if (p.drawdown > maxDD) maxDD = p.drawdown;
    double currentDD = dd.back().drawdown;
    int    nFills    = static_cast<int>(eq.size());

    char header[256];
    std::snprintf(header, sizeof(header),
        "Equity: %s$%.2f   Max DD: $%.2f   "
        "Current DD: $%.2f   Fills: %d",
        finalEq >= 0 ? "+" : "-", std::fabs(finalEq),
        maxDD, currentDD, nFills);
    ImGui::TextUnformatted(header);
    ImGui::Spacing();

    // ---- Layout: top chart (equity + drawdown overlay), bottom chart (DD series) ----
    const float topH    = ImGui::GetContentRegionAvail().y * 0.65f;
    const float bottomH = ImGui::GetContentRegionAvail().y * 0.30f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 orig    = ImGui::GetCursorScreenPos();

    // ==== TOP: equity curve with drawdown overlay ====
    ImVec2 topTL = orig;
    ImVec2 topBR = ImVec2(orig.x + ImGui::GetContentRegionAvail().x,
                          orig.y + topH);
    // Frame.
    dl->AddRectFilled(topTL, topBR,
                      IM_COL32(20, 20, 20, 255));
    dl->AddRect(topTL, topBR, IM_COL32(80, 80, 80, 255));

    auto project = [&](uint64_t ts, double val,
                       double yMin, double yMax) -> ImVec2 {
        double xf = (static_cast<double>(ts - tMin) / 1000000.0) /
                    span_secs;
        double yf = (val - yMin) / (yMax - yMin);
        return ImVec2(topTL.x + static_cast<float>(xf) *
                      (topBR.x - topTL.x),
                      topBR.y - static_cast<float>(yf) *
                      (topBR.y - topTL.y));
    };

    // Fill drawdown region: from peak line (top) down to equity line.
    // Polygon strip — one triangle per segment.
    if (eq.size() >= 2 && maxDD > 1e-9) {
        ImVec2 peakTL = topTL, peakBR = topBR;
        (void)peakTL; (void)peakBR;
        // Build poly: walk forward, push (peak point, equity point).
        // Use the drawdown series (which already has running_peak).
        for (size_t i = 1; i < dd.size(); ++i) {
            const auto& prev = dd[i - 1];
            const auto& cur  = dd[i];
            const auto& prevEq = eq[i - 1];
            const auto& curEq  = eq[i];
            // Only fill if underwater.
            if (prev.drawdown < 1e-9 && cur.drawdown < 1e-9)
                continue;
            ImVec2 pPeakA = project(prev.timestamp_us,
                                    prev.running_peak, eqMin, eqMax);
            ImVec2 pPeakB = project(cur.timestamp_us,
                                    cur.running_peak, eqMin, eqMax);
            ImVec2 pEqA   = project(prevEq.timestamp_us,
                                    prevEq.cumulative, eqMin, eqMax);
            ImVec2 pEqB   = project(curEq.timestamp_us,
                                    curEq.cumulative, eqMin, eqMax);
            dl->AddQuadFilled(pPeakA, pPeakB, pEqB, pEqA,
                              IM_COL32(180, 50, 50, 80));
        }
    }

    // Peak line (dim cyan, dashed look approximated by skipping pixels).
    {
        ImVec2 prev = ImVec2(-1, -1);
        for (size_t i = 0; i < dd.size(); ++i) {
            ImVec2 cur = project(dd[i].timestamp_us,
                                 dd[i].running_peak,
                                 eqMin, eqMax);
            if (prev.x >= 0) {
                dl->AddLine(prev, cur, IM_COL32(80, 80, 130, 200), 1.0f);
            }
            prev = cur;
        }
    }

    // Equity curve (bright green/red depending on sign of final).
    {
        ImU32 col = (finalEq >= 0.0)
            ? IM_COL32(80, 220, 120, 255)
            : IM_COL32(220, 80, 80, 255);
        ImVec2 prev = ImVec2(-1, -1);
        for (const auto& p : eq) {
            ImVec2 cur = project(p.timestamp_us, p.cumulative,
                                 eqMin, eqMax);
            if (prev.x >= 0) {
                dl->AddLine(prev, cur, col, 2.0f);
            }
            prev = cur;
        }
    }

    // Zero line if the curve crosses zero.
    if (eqMin < 0.0 && eqMax > 0.0) {
        double yf = (0.0 - eqMin) / (eqMax - eqMin);
        float yPx = topBR.y - static_cast<float>(yf) *
                    (topBR.y - topTL.y);
        dl->AddLine(ImVec2(topTL.x, yPx), ImVec2(topBR.x, yPx),
                    IM_COL32(120, 120, 120, 120), 1.0f);
    }

    // Y-axis labels (min / max).
    char yMinBuf[32], yMaxBuf[32];
    std::snprintf(yMinBuf, sizeof(yMinBuf), "%.0f", eqMin);
    std::snprintf(yMaxBuf, sizeof(yMaxBuf), "%.0f", eqMax);
    dl->AddText(ImVec2(topTL.x + 4, topBR.y - 14),
                IM_COL32(150, 150, 150, 255), yMinBuf);
    dl->AddText(ImVec2(topTL.x + 4, topTL.y + 2),
                IM_COL32(150, 150, 150, 255), yMaxBuf);

    // X-axis labels (start / end).
    char xMinBuf[16], xMaxBuf[16];
    xLabel(xMinBuf, sizeof(xMinBuf), tMin, span_secs);
    xLabel(xMaxBuf, sizeof(xMaxBuf), tMax, span_secs);
    dl->AddText(ImVec2(topTL.x + 4, topBR.y - 28),
                IM_COL32(150, 150, 150, 255), xMinBuf);
    dl->AddText(ImVec2(topBR.x - 50, topBR.y - 28),
                IM_COL32(150, 150, 150, 255), xMaxBuf);

    // Reserve the vertical space.
    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, topH));
    ImGui::Spacing();

    // ==== BOTTOM: drawdown bars ====
    ImVec2 botTL = ImGui::GetCursorScreenPos();
    ImVec2 botBR = ImVec2(botTL.x +
                          ImGui::GetContentRegionAvail().x,
                          botTL.y + bottomH);
    dl->AddRectFilled(botTL, botBR, IM_COL32(20, 20, 20, 255));
    dl->AddRect(botTL, botBR, IM_COL32(80, 80, 80, 255));

    // Zero line at the TOP of the bottom chart (DD grows downward).
    dl->AddLine(botTL, ImVec2(botBR.x, botTL.y),
                IM_COL32(120, 120, 120, 200), 1.0f);

    if (ddMax > 1e-9) {
        // Render DD as filled bars — one per segment.
        for (size_t i = 1; i < dd.size(); ++i) {
            if (dd[i].drawdown < 1e-9) continue;
            double yf = dd[i].drawdown / ddMax;
            float x0 = botTL.x + static_cast<float>(
                (static_cast<double>(dd[i - 1].timestamp_us - tMin) /
                 1000000.0) / span_secs) *
                       (botBR.x - botTL.x);
            float x1 = botTL.x + static_cast<float>(
                (static_cast<double>(dd[i].timestamp_us - tMin) /
                 1000000.0) / span_secs) *
                       (botBR.x - botTL.x);
            float yBar = botTL.y + static_cast<float>(yf) *
                         (botBR.y - botTL.y);
            dl->AddRectFilled(ImVec2(x0, botTL.y),
                              ImVec2(x1 + 1.0f, yBar),
                              IM_COL32(180, 50, 50, 180));
        }
    }
    // DD axis label.
    char ddBuf[32];
    std::snprintf(ddBuf, sizeof(ddBuf), "DD %.0f", ddMax);
    dl->AddText(ImVec2(botTL.x + 4, botTL.y + 2),
                IM_COL32(150, 100, 100, 255), ddBuf);

    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, bottomH));

    // ---- Tooltip on hover ----
    if (ImGui::IsItemHovered()) {
        // Cross-hair at the closest data point.
        ImVec2 mouse = ImGui::GetMousePos();
        float relX = mouse.x - topTL.x;
        if (relX < 0) relX = 0;
        if (relX > topBR.x - topTL.x)
            relX = topBR.x - topTL.x;
        size_t idx = static_cast<size_t>(
            (relX / (topBR.x - topTL.x)) * (eq.size() - 1));
        if (idx >= eq.size()) idx = eq.size() - 1;
        const auto& p  = eq[idx];
        const auto& dp = dd[idx];
        char tip[256];
        std::time_t secs = static_cast<std::time_t>(
            p.timestamp_us / 1000000ULL);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &secs);
#else
        localtime_r(&secs, &tm);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
        std::snprintf(tip, sizeof(tip),
            "%s\nFill: %s$%.2f\nCumulative: %s$%.2f\nDD: $%.2f",
            ts,
            p.realized >= 0 ? "+" : "-", std::fabs(p.realized),
            p.cumulative >= 0 ? "+" : "-", std::fabs(p.cumulative),
            dp.drawdown);
        ImGui::SetTooltip("%s", tip);
    }
}

}  // namespace btquant::ui
