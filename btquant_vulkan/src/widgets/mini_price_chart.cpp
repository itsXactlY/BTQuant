#include "mini_price_chart.hpp"

#include "../data/market_data.hpp"
#include "../data/market_data_processor.hpp"

#include <algorithm>
#include <cstdio>
#include <imgui.h>
#include <implot.h>

namespace btquant::ui {

bool MiniPriceChart::validateCandle(const ::btquant::data::Candle& c) {
    if (c.open <= 0.0 || c.low <= 0.0) return false;
    double hi = std::max(c.open, c.close);
    double lo = std::min(c.open, c.close);
    if (c.high < hi) return false;
    if (c.low  > lo) return false;
    if (c.volume < 0.0) return false;
    return true;
}

int MiniPriceChart::validateSeries(
        const std::vector<::btquant::data::Candle>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (!validateCandle(v[i])) return static_cast<int>(i);
    }
    return -1;
}

void MiniPriceChart::render() {
    if (!m_open) return;

    ImGui::SetNextWindowSize(ImVec2(640, 420), ImGuiCond_Appearing);
    if (!ImGui::Begin("Mini Price Chart", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!m_data) {
        ImGui::TextDisabled("No MarketDataProcessor bound — chart inert.");
        ImGui::End();
        return;
    }

    // Pull the latest snapshot.
    auto snap = m_data->snapshot(0, m_historyN);
    const auto& candles = snap.recent_candles;

    // Header summary.
    if (candles.empty()) {
        ImGui::TextDisabled("Waiting for candle data…");
        ImGui::End();
        return;
    }
    const auto& last = candles.back();
    ImGui::Text("Symbol: %s  |  Last close: $%.2f  |  Candles: %zu",
                m_data->symbol().c_str(), last.close, candles.size());
    ImGui::Separator();

    // Validate the series before plotting — bad candles would render as
    // garbage. Skip the chart and show a warning instead.
    int badIdx = validateSeries(candles);
    if (badIdx >= 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.2f, 1.0f));
        ImGui::Text("⚠ Skipping render — bad candle at index %d", badIdx);
        ImGui::PopStyleColor();
        ImGui::End();
        return;
    }

    // Build arrays for ImPlot.
    const int N = static_cast<int>(candles.size());
    std::vector<double> xs(N);
    std::vector<double> closes(N);
    std::vector<double> buyVols(N), sellVols(N);
    std::vector<double> wickLow(N), wickHigh(N);   // per-candle wick y-range
    for (int i = 0; i < N; ++i) {
        xs[i]       = static_cast<double>(i);
        closes[i]   = candles[i].close;
        buyVols[i]  = candles[i].buyVolume;
        sellVols[i] = candles[i].sellVolume;
        wickLow[i]  = candles[i].low;
        wickHigh[i] = candles[i].high;
    }

    // Price subplot — line chart of close prices + per-candle wick lines.
    // (This ImPlot version lacks PlotCandlestick + ImPlotCol_Fill, so we
    // render the close line and wicks manually. Bodies are not drawn —
    // good enough for an at-a-glance trend read.)
    if (ImPlot::BeginPlot("##price", ImVec2(-1, 280))) {
        ImPlot::SetupAxes("bar #", "price");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, N - 1, ImPlotCond_Always);
        // Close line.
        ImPlot::PlotLine("close", xs.data(), closes.data(), N);
        // Wick: each candle's low→high as a 2-point segment.
        for (int i = 0; i < N; ++i) {
            double wx[2] = { xs[i], xs[i] };
            double wy[2] = { wickLow[i], wickHigh[i] };
            ImPlot::PlotLine("wick", wx, wy, 2);
        }
        ImPlot::EndPlot();
    }

    // Volume subplot — PlotBars uses a single value from y=0, so we draw
    // buy + sell as separate series with default coloring.
    if (ImPlot::BeginPlot("##volume", ImVec2(-1, 100))) {
        ImPlot::SetupAxes("bar #", "vol");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, N - 1, ImPlotCond_Always);
        ImPlot::PlotBars("buy",  buyVols.data(),  N, 0.4);
        ImPlot::PlotBars("sell", sellVols.data(), N, 0.4);
        ImPlot::EndPlot();
    }

    ImGui::End();
}

} // namespace btquant::ui
