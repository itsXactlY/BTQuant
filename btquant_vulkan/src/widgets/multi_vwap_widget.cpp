#include "multi_vwap_widget.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>

namespace btquant::ui {

MultiVWAPWidget::MultiVWAPWidget() = default;
MultiVWAPWidget::~MultiVWAPWidget() = default;

void MultiVWAPWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

// Compute volume-weighted average price over the last k trades (newest-first
// vector). Returns 0 if no trades.
static double rollingVWAP(const std::vector<data::Trade>& trades, size_t k) {
    if (trades.empty() || k == 0) return 0.0;
    k = std::min(k, trades.size());
    double sumPV = 0.0, sumV = 0.0;
    for (size_t i = 0; i < k; ++i) {
        sumPV += trades[i].price * trades[i].size;
        sumV  += trades[i].size;
    }
    return sumV > 0 ? sumPV / sumV : 0.0;
}

void MultiVWAPWidget::render() {
    if (!m_initialized) m_initialized = true;
    if (!showWindow) return;

    ImGui::Begin("Multi-VWAP", &showWindow);

    bool isLive = false;
    std::vector<data::Trade> trades;
    if (m_data) {
        auto snap = m_data->snapshot(512);
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
                if (synth.size() > 512) synth.erase(synth.begin());
            }
            lastUpdate = now;
        }
        trades = synth;
    }

    // Periods to compute.
    const size_t periods[] = {20, 50, 100, 200, 0};  // 0 = "all visible"
    const char* labels[] = {"VWAP-20", "VWAP-50", "VWAP-100", "VWAP-200", "VWAP-all"};

    ImGui::Text("Multi-period VWAP — %s   (last %zu trades)",
                isLive ? "LIVE" : "synthetic", trades.size());

    if (trades.empty()) {
        ImGui::Text("No trades");
        ImGui::End();
        return;
    }

    double lastPrice = trades.front().price;

    // Mini-chart with VWAP lines.
    if (ImPlot::BeginPlot("Price vs VWAP bands", ImVec2(-1, 220))) {
        // Price line (last N trades).
        std::vector<double> xs, ys;
        xs.reserve(trades.size());
        ys.reserve(trades.size());
        for (size_t i = 0; i < trades.size(); ++i) {
            xs.push_back(static_cast<double>(i));
            ys.push_back(trades[i].price);
        }
        ImPlot::SetupAxes("trade #", "price");
        ImPlot::PlotLine("price", xs.data(), ys.data(), (int)ys.size());

        // Horizontal VWAP lines (one per period). ImPlot in this version
        // doesn't expose per-line color or line-weight vars, so the bands
        // share the default line color and weight — the legend still labels
        // each one via PlotLine's first-string-arg.
        for (size_t p = 0; p < std::size(periods); ++p) {
            double vwap = rollingVWAP(trades, periods[p]);
            if (vwap > 0) {
                double xs2[2] = {0.0, static_cast<double>(trades.size() - 1)};
                double ys2[2] = {vwap, vwap};
                ImPlot::PlotLine(labels[p], xs2, ys2, 2);
            }
        }
        ImPlot::EndPlot();
    }

    // Numeric table.
    ImGui::Columns(5, "VWAPTable", true);
    ImGui::SetColumnWidth(0, 90);
    ImGui::SetColumnWidth(1, 90);
    ImGui::SetColumnWidth(2, 90);
    ImGui::SetColumnWidth(3, 90);
    ImGui::SetColumnWidth(4, 70);

    ImGui::Text("Period"); ImGui::NextColumn();
    ImGui::Text("VWAP"); ImGui::NextColumn();
    ImGui::Text("Δ vs Last"); ImGui::NextColumn();
    ImGui::Text("Δ %%"); ImGui::NextColumn();
    ImGui::Text("Volume"); ImGui::NextColumn();
    ImGui::Separator();

    for (size_t p = 0; p < std::size(periods); ++p) {
        size_t k = periods[p];
        double vwap = rollingVWAP(trades, k);
        if (vwap <= 0) {
            ImGui::Text("%s", labels[p]); ImGui::NextColumn();
            ImGui::Text("—"); ImGui::NextColumn();
            ImGui::Text("—"); ImGui::NextColumn();
            ImGui::Text("—"); ImGui::NextColumn();
            ImGui::Text("—"); ImGui::NextColumn();
            continue;
        }
        double sumV = 0;
        size_t actual = std::min(k, trades.size());
        for (size_t i = 0; i < actual; ++i) sumV += trades[i].size;
        double delta = lastPrice - vwap;
        double deltaPct = (delta / vwap) * 100.0;

        ImGui::Text("%s", labels[p]); ImGui::NextColumn();
        ImGui::Text("%.4f", vwap); ImGui::NextColumn();
        // Color-code delta direction.
        if (delta > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 100, 255));
        else if (delta < 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 80, 80, 255));
        ImGui::Text("%+.4f", delta);
        if (delta != 0) ImGui::PopStyleColor();
        ImGui::NextColumn();
        if (delta > 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 100, 255));
        else if (delta < 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 80, 80, 255));
        ImGui::Text("%+.2f%%", deltaPct);
        if (delta != 0) ImGui::PopStyleColor();
        ImGui::NextColumn();
        ImGui::Text("%.2f", sumV); ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Text("Last trade: %.4f   (used as Δ reference)", lastPrice);

    ImGui::End();
}

}  // namespace btquant::ui
