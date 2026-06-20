#include "risk_panel.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include "../data/risk_guard.hpp"

#include <imgui.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>

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
RiskMetrics computeMetrics(const std::vector<data::Trade>& trades) {
    RiskMetrics m{};
    if (trades.empty()) return m;
    m.tradeCount = static_cast<int>(trades.size());

    double sumSize = 0.0;
    for (const auto& t : trades) {
        if (t.isBuy) ++m.buyCount;
        else         ++m.sellCount;
        sumSize += std::fabs(t.size);
    }
    m.avgTradeSize = sumSize / static_cast<double>(m.tradeCount);

    // Trade returns — consecutive mid-price pct moves.
    std::vector<double> returns;
    returns.reserve(trades.size());
    // trades.front() is newest (per the data processor). Walk from newest
    // backwards so returns are in chronological order.
    for (size_t i = 1; i < trades.size(); ++i) {
        const double prev = trades[i].price;
        const double cur  = trades[i - 1].price;
        if (prev > 0.0) returns.push_back((cur - prev) / prev);
    }
    if (returns.size() >= 2) {
        double sum = 0.0;
        for (double r : returns) sum += r;
        double mean = sum / static_cast<double>(returns.size());
        double var  = 0.0;
        for (double r : returns) { double d = r - mean; var += d * d; }
        double stddev = std::sqrt(var / static_cast<double>(returns.size()));
        if (stddev > 1e-12) {
            m.sharpePerTrade = mean / stddev;
            m.sharpeAnnualized = m.sharpePerTrade * std::sqrt(static_cast<double>(returns.size()));
        }
    }

    // Round-trip P&L deltas via running position. Each trade contributes:
    //   dPnl = sign(position_after) * (price_now - price_prev)
    // where position_after uses the trade's direction (isBuy=true → +1).
    // We track both gross profit and gross loss for profit factor / win rate.
    double cumPnl = 0.0;
    double peak   = 0.0;
    m.maxDrawdown = 0.0;
    int    wins    = 0;
    int    rounds  = 0;
    double grossProfit = 0.0;
    double grossLoss   = 0.0;
    double sumRTpnl    = 0.0;

    // trades[] is newest-first, so iterating i from oldest (back) to newest
    // (front) gives chronological order.
    int pos = 0;
    for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
        const double p = it->price;
        if (pos == 0) { /* baseline tick — open the position as +1 */ }
        // Update cumPnl: change in P&L since last tick = pos * (price - prevPrice).
        static thread_local double prevPrice = 0.0;
        if (pos != 0 && prevPrice > 0.0) {
            double dpnl = static_cast<double>(pos) * (p - prevPrice);
            cumPnl += dpnl;
            // Track as a "round-trip candidate" — counted toward win rate.
            if (dpnl > 0.0) { ++wins; grossProfit += dpnl; }
            else if (dpnl < 0.0) { grossLoss += dpnl; }
            ++rounds;
            sumRTpnl += dpnl;
        }
        if (cumPnl > peak) peak = cumPnl;
        double dd = peak - cumPnl;
        if (dd > m.maxDrawdown) m.maxDrawdown = dd;
        // Open position for next tick: 1 for buy, -1 for sell.
        pos = it->isBuy ? 1 : -1;
        prevPrice = p;
    }
    if (rounds > 0) m.winRate = static_cast<double>(wins) / static_cast<double>(rounds);
    if (grossLoss < -1e-12) m.profitFactor = grossProfit / (-grossLoss);
    else if (grossProfit > 0) m.profitFactor = std::numeric_limits<double>::infinity();
    if (rounds > 0) m.expectancy = sumRTpnl / static_cast<double>(rounds);

    return m;
}

PositionState computePosition(const std::vector<data::Trade>& trades) {
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

    // Aggregate risk metrics over the trade window.
    RiskMetrics m = computeMetrics(trades);
    ImGui::Text("Risk Metrics — %d trades (%d buy / %d sell, avg size %.3f)",
                m.tradeCount, m.buyCount, m.sellCount, m.avgTradeSize);
    ImGui::Columns(2, "RiskMetricsTable", false);
    ImGui::SetColumnWidth(0, 200);

    ImGui::Text("Sharpe (per-trade)");   ImGui::NextColumn();
    if (m.sharpePerTrade >= 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else                       ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.3f", m.sharpePerTrade);
    ImGui::PopStyleColor();             ImGui::NextColumn();

    ImGui::Text("Sharpe (sqrt N, heuristic)");  ImGui::NextColumn();
    if (m.sharpeAnnualized >= 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else                         ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.3f", m.sharpeAnnualized);
    ImGui::PopStyleColor();          ImGui::NextColumn();

    ImGui::Text("Max drawdown");           ImGui::NextColumn();
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("-%.4f", m.maxDrawdown);
    ImGui::PopStyleColor();                 ImGui::NextColumn();

    ImGui::Text("Win rate");               ImGui::NextColumn();
    ImGui::Text("%.1f%%", m.winRate * 100.0);  ImGui::NextColumn();

    ImGui::Text("Profit factor");          ImGui::NextColumn();
    if (std::isinf(m.profitFactor)) ImGui::Text("inf");
    else                            ImGui::Text("%.2f", m.profitFactor);
    ImGui::NextColumn();

    ImGui::Text("Expectancy / tick");      ImGui::NextColumn();
    if (m.expectancy >= 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
    else                   ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
    ImGui::Text("%+.4f", m.expectancy);
    ImGui::PopStyleColor();
    ImGui::Columns(1);

    ImGui::Separator();

    // Notes:
    // - Position tracker is naive — does not handle partial fills, fees, or
    //   cross-symbol netting. Suitable for a single-symbol demo where every
    //   trade is a clean +/− on the open position.
    // - "Daily P&L" here is window-scoped, not time-of-day filtered. To
    //   get true daily P&L, group trades by date in the data model.
    // - Sharpe "annualized" uses sqrt(N) heuristic, not 252-trading-days
    //   scaling. For a 60s candle stream at ~10 trades/sec, this gives
    //   an N-tick Sharpe, not a yearly one. Calibrate the multiplier
    //   before relying on it for strategy comparison.
    ImGui::TextDisabled("Notes: window-scoped; naive avg-entry; no fees/partials; "
                        "Sharpe = sqrt(N) heuristic.");

    ImGui::Separator();
    if (m_riskGuard) {
        // Live daily-loss progress — the trader can see at a glance
        // how much of the kill threshold has been consumed. The bar
        // goes red once we cross the threshold. We use the absolute
        // value of sessionRealized against killOnDailyLossUSD (both
        // are positive magnitudes; sessionRealized is signed in the
        // underlying API because losses are negative).
        double sessionRealized = m_riskGuard->sessionRealized();
        double killThreshold   = m_riskGuard->config().killOnDailyLossUSD;
        if (killThreshold > 0.0) {
            double frac = std::min(1.0, std::fabs(sessionRealized) / killThreshold);
            // Colour: green below 50%, yellow 50–80%, red ≥ 80%.
            ImVec4 barCol;
            if (frac < 0.5)      barCol = ImVec4(0.30f, 0.85f, 0.40f, 1.0f);
            else if (frac < 0.8) barCol = ImVec4(0.95f, 0.85f, 0.30f, 1.0f);
            else                 barCol = ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barCol);
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay), "%s$%.0f / -$%.0f",
                          sessionRealized >= 0 ? "+" : "",
                          std::fabs(sessionRealized), killThreshold);
            ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
            ImGui::PopStyleColor();
            // Remaining budget readout (live; recomputed each frame).
            double remaining = m_riskGuard->remainingLossBudget();
            ImGui::Text("Remaining loss budget: %s$%.2f",
                        remaining >= 0 ? "+" : "",
                        std::fabs(remaining));
        }
        // Reset Session button — clears the running P&L counter so the
        // trader can start a new trading day without restarting the app.
        // Confirmation popup prevents accidental clicks.
        ImGui::SameLine();
        if (ImGui::Button("Reset session")) {
            ImGui::OpenPopup("Confirm reset session");
        }
        if (ImGui::BeginPopupModal("Confirm reset session", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Reset session realized P&L to $0.00?");
            ImGui::Text("This clears today's kill-switch counter.");
            if (ImGui::Button("Confirm")) {
                m_riskGuard->resetSession();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    } else {
        ImGui::TextDisabled("(RiskGuard not bound — no daily-loss tracking)");
    }

    ImGui::End();
}

}  // namespace btquant::ui
