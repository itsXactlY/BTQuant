#include "risk_panel.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

#include <imgui.h>
#include <vector>
#include <chrono>
#include <cmath>

namespace btquant::ui {

RiskPanel::RiskPanel() = default;
RiskPanel::~RiskPanel() = default;

void RiskPanel::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

// Compute position & P&L from a newest-first trade list.
// Walks forward (oldest first), treating each trade as either:
//   * Building the position (same-direction or fresh).
//   * Reducing / flipping the position (opposite direction); the closed
//     portion realizes P&L at the average-entry price.
struct PositionState {
    double netSize = 0;       // signed (positive = long, negative = short)
    double avgEntry = 0;     // avg entry price of the OPEN position
    double realized = 0;     // total realized P&L in the window
    double maxAbsSize = 0;   // peak |position| in the window
};

static PositionState computePosition(const std::vector<data::Trade>& trades) {
    PositionState s;
    // Walk oldest → newest.
    for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
        const auto& t = *it;
        // isBuy=true means buyer-initiated → +size (long)
        // isBuy=false means seller-initiated → -size (short)
        double deltaSigned = t.isBuy ? t.size : -t.size;

        if (s.netSize == 0) {
            // Open fresh.
            s.netSize = deltaSigned;
            s.avgEntry = t.price;
        } else if ((s.netSize > 0 && deltaSigned > 0) || (s.netSize < 0 && deltaSigned < 0)) {
            // Adding to existing position — update avg entry.
            double totalSize = std::abs(s.netSize) + std::abs(deltaSigned);
            s.avgEntry = (std::abs(s.netSize) * s.avgEntry + std::abs(deltaSigned) * t.price) / totalSize;
            s.netSize += deltaSigned;
        } else {
            // Opposite direction — closing (or flipping) the position.
            double closeSize = std::min(std::abs(deltaSigned), std::abs(s.netSize));
            // Realized P&L per contract = (exit - entry) for long, (entry - exit) for short.
            double pnlPerUnit = (s.netSize > 0) ? (t.price - s.avgEntry) : (s.avgEntry - t.price);
            s.realized += closeSize * pnlPerUnit;
            s.netSize += deltaSigned;
            // If position flipped, the remainder opens a new position at this price.
            if (s.netSize != 0 && ((s.netSize > 0) != (deltaSigned > 0))) {
                // flipped
                s.avgEntry = t.price;
            } else if (s.netSize == 0) {
                s.avgEntry = 0;
            }
        }

        if (std::abs(s.netSize) > s.maxAbsSize) s.maxAbsSize = std::abs(s.netSize);
    }
    return s;
}

void RiskPanel::render() {
    if (!m_initialized) m_initialized = true;
    if (!showWindow) return;

    ImGui::Begin("Risk Panel", &showWindow);

    bool isLive = false;
    std::vector<data::Trade> trades;
    if (m_data) {
        auto snap = m_data->snapshot(1024);  // larger window for position tracking
        trades = std::move(snap.recent_trades);
        isLive = snap.snapshot_seq > 0 && !trades.empty();
    }

    // Synthetic fallback.
    if (!isLive) {
        static std::vector<data::Trade> synth;
        static auto lastUpdate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 200) {
            uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            for (int i = 0; i < 6; ++i) {
                data::Trade t{};
                t.id = synth.size();
                t.price = 100.0 + (rand() % 200) / 100.0;
                t.size = 1.0 + (rand() % 50) / 10.0;
                t.timestamp = now_us - (uint64_t)(rand() % 60'000'000);
                t.isBuy = (rand() % 2 == 0);
                synth.push_back(t);
                if (synth.size() > 1024) synth.erase(synth.begin());
            }
            lastUpdate = now;
        }
        trades = synth;
    }

    ImGui::Text("Position + P&L — %s   (last %zu trades)",
                isLive ? "LIVE" : "synthetic", trades.size());

    if (trades.empty()) {
        ImGui::Text("No trades");
        ImGui::End();
        return;
    }

    PositionState st = computePosition(trades);
    double markPrice = trades.front().price;  // newest trade's price
    double unrealized = 0.0;
    if (st.netSize != 0) {
        double sign = (st.netSize > 0) ? 1.0 : -1.0;
        unrealized = sign * (markPrice - st.avgEntry) * std::abs(st.netSize);
    }
    double totalPnl = st.realized + unrealized;

    // Position color.
    ImU32 posCol = IM_COL32(220, 220, 220, 255);
    const char* posLabel = "FLAT";
    if (st.netSize > 0.0001) { posLabel = "LONG";  posCol = IM_COL32(0, 220, 120, 255); }
    else if (st.netSize < -0.0001) { posLabel = "SHORT"; posCol = IM_COL32(220, 80, 80, 255); }

    // Big status block.
    ImGui::PushStyleColor(ImGuiCol_Text, posCol);
    ImGui::Text("Position: %s   %.3f contracts @ avg %.4f",
                posLabel, std::abs(st.netSize), st.avgEntry);
    ImGui::PopStyleColor();

    // P&L table.
    ImGui::Columns(2, "RiskTable", false);
    ImGui::SetColumnWidth(0, 200);

    ImGui::Text("Mark price");            ImGui::NextColumn();
    ImGui::Text("%.4f", markPrice);      ImGui::NextColumn();
    ImGui::Text("Unrealized P&L");        ImGui::NextColumn();
    if (unrealized > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else if (unrealized < 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.4f", unrealized);
    if (unrealized != 0) ImGui::PopStyleColor();
    ImGui::NextColumn();
    ImGui::Text("Realized P&L (window)"); ImGui::NextColumn();
    if (st.realized > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else if (st.realized < 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.4f", st.realized);
    if (st.realized != 0) ImGui::PopStyleColor();
    ImGui::NextColumn();
    ImGui::Text("Total P&L");             ImGui::NextColumn();
    if (totalPnl > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else if (totalPnl < 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.4f", totalPnl);
    if (totalPnl != 0) ImGui::PopStyleColor();
    ImGui::NextColumn();
    ImGui::Text("Max position (window)"); ImGui::NextColumn();
    ImGui::Text("%.3f contracts", st.maxAbsSize); ImGui::NextColumn();
    ImGui::Text("Entry Δ (mark - avg)");   ImGui::NextColumn();
    if (st.netSize != 0)
        ImGui::Text("%+.4f", markPrice - st.avgEntry);
    else
        ImGui::Text("—"); ImGui::NextColumn();

    ImGui::Columns(1);

    // Mini risk bar — visualize unrealized P&L as a horizontal bar.
    {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float totalW = 380.0f;
        float h = 18.0f;
        dl->AddRectFilled(pos, ImVec2(pos.x + totalW, pos.y + h),
                          IM_COL32(20, 20, 25, 200));
        // Center line.
        float midX = pos.x + totalW * 0.5f;
        dl->AddLine(ImVec2(midX, pos.y), ImVec2(midX, pos.y + h),
                    IM_COL32(255, 255, 255, 120));
        // Scale: ±20 → full width, so 1 unit = totalW/40.
        double v = unrealized + st.realized;
        double clamped = std::max(-20.0, std::min(20.0, v));
        float fill = static_cast<float>(std::abs(clamped) / 20.0 * (totalW * 0.5));
        ImU32 col = (v >= 0) ? IM_COL32(0, 200, 100, 220) : IM_COL32(220, 60, 60, 220);
        if (v >= 0) dl->AddRectFilled(ImVec2(midX, pos.y + 2),
                                        ImVec2(midX + fill, pos.y + h - 2), col);
        else       dl->AddRectFilled(ImVec2(midX - fill, pos.y + 2),
                                        ImVec2(midX, pos.y + h - 2), col);
        ImGui::Dummy(ImVec2(totalW, h));
    }

    ImGui::Separator();

    // Notes:
    // - Position tracker is naive — does not handle partial fills, fees, or
    //   cross-symbol netting. Suitable for a single-symbol demo where every
    //   trade is a clean +/− on the open position.
    // - "Daily P&L" here is window-scoped, not time-of-day filtered. To
    //   get true daily P&L, group trades by date in the data model.
    ImGui::TextDisabled("Notes: window-scoped; naive avg-entry; no fees/partials.");

    ImGui::End();
}

}  // namespace btquant::ui
