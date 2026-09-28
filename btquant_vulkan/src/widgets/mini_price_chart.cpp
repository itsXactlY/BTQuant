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

float MiniPriceChart::priceToPixelY(double price, double yMin, double yMax,
                                    float canvasY, float canvasH) {
    // Defensive: degenerate range → return canvas center.
    if (yMax <= yMin) return canvasY + canvasH * 0.5f;
    double frac = (price - yMin) / (yMax - yMin);
    // Higher price → smaller Y. Clamp to canvas so out-of-range prices
    // (rare but possible if the user drags the slider aggressively) don't
    // produce negative or off-canvas pixel coords that the draw-list
    // would silently clip.
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return canvasY + static_cast<float>((1.0 - frac) * canvasH);
}

float MiniPriceChart::indexToPixelX(int idx, int count, float canvasX,
                                    float canvasW, float bodyFrac) {
    if (count <= 0) return canvasX + canvasW * 0.5f;
    float slotW = canvasW / static_cast<float>(count);
    float bodyW = slotW * bodyFrac;
    // Centered on slot midpoint. Fractional bodyFrac below 1 leaves a
    // gap between candles so they don't visually merge.
    return canvasX + slotW * (idx + 0.5f) - bodyW * 0.5f;
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
    // Mode toggle — Line = cheap at-a-glance trend read, Candle = real
    // OHLC bodies drawn via raw draw-list (filled rect + wick line).
    ImGui::SameLine();
    ImGui::RadioButton("Line",   reinterpret_cast<int*>(&m_renderMode),
                       static_cast<int>(RenderMode::Line));
    ImGui::SameLine();
    ImGui::RadioButton("Candle", reinterpret_cast<int*>(&m_renderMode),
                       static_cast<int>(RenderMode::Candle));
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

    if (m_renderMode == RenderMode::Candle) {
        // Candlestick branch — raw draw-list rendering. We bypass ImPlot
        // here because the version vendored here lacks PlotCandlestick
        // + ImPlotCol_Fill, so the ImPlot path can only paint wicks.
        // Drawing bodies ourselves is both simpler and lets us control
        // the body width fraction + color scheme directly.
        //
        // Compute Y range across all candles with a small padding so
        // the highest wick / lowest wick doesn't sit on the canvas edge.
        double yMin = candles[0].low, yMax = candles[0].high;
        for (const auto& c : candles) {
            if (c.low  < yMin) yMin = c.low;
            if (c.high > yMax) yMax = c.high;
        }
        double yPad = (yMax - yMin) * 0.05;
        yMin -= yPad;
        yMax += yPad;
        if (yMax <= yMin) { yMin -= 0.5; yMax += 0.5; }  // degenerate guard

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize(-1, 280);
        ImGui::InvisibleButton("##candle_canvas", canvasSize);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(canvasPos,
                          ImVec2(canvasPos.x + ImGui::GetContentRegionAvail().x,
                                 canvasPos.y + canvasSize.y),
                          IM_COL32(20, 22, 32, 255));

        const float canvasW = ImGui::GetContentRegionAvail().x;
        const float canvasH = canvasSize.y;
        const float bodyFrac = 0.7f;
        const ImU32 colUp   = IM_COL32( 60, 200, 100, 255);  // green
        const ImU32 colDown = IM_COL32(220,  60,  60, 255);  // red
        const ImU32 colWick = IM_COL32(200, 200, 200, 200);

        for (int i = 0; i < N; ++i) {
            const auto& c = candles[i];
            float bodyX = indexToPixelX(i, N, canvasPos.x, canvasW, bodyFrac);
            float bodyW = (canvasW / static_cast<float>(N)) * bodyFrac;
            float yOpen  = priceToPixelY(c.open,  yMin, yMax, canvasPos.y, canvasH);
            float yClose = priceToPixelY(c.close, yMin, yMax, canvasPos.y, canvasH);
            float yHigh  = priceToPixelY(c.high,  yMin, yMax, canvasPos.y, canvasH);
            float yLow   = priceToPixelY(c.low,   yMin, yMax, canvasPos.y, canvasH);
            float bodyTop    = std::min(yOpen, yClose);
            float bodyBottom = std::max(yOpen, yClose);
            // Ensure a visible body even for doji (open == close).
            if (bodyBottom - bodyTop < 1.0f) bodyBottom = bodyTop + 1.0f;
            bool up = c.close >= c.open;
            ImU32 bodyCol = up ? colUp : colDown;

            // Wick (low → high) as a single vertical line through the body.
            float centerX = bodyX + bodyW * 0.5f;
            dl->AddLine(ImVec2(centerX, yHigh),
                        ImVec2(centerX, yLow),
                        colWick, 1.0f);
            // Body as a filled rectangle.
            dl->AddRectFilled(ImVec2(bodyX, bodyTop),
                              ImVec2(bodyX + bodyW, bodyBottom),
                              bodyCol);
            // Thin outline so adjacent doji bodies stay distinguishable.
            dl->AddRect(ImVec2(bodyX, bodyTop),
                        ImVec2(bodyX + bodyW, bodyBottom),
                        IM_COL32(0, 0, 0, 180), 0.0f, 0, 1.0f);
        }

        // Last-close price tag on the right edge — small label so the
        // trader can read the current price at a glance.
        char tagBuf[32];
        std::snprintf(tagBuf, sizeof(tagBuf), "%.2f", last.close);
        float tagY = priceToPixelY(last.close, yMin, yMax,
                                   canvasPos.y, canvasH);
        dl->AddText(ImVec2(canvasPos.x + canvasW - 60.0f, tagY - 8.0f),
                    IM_COL32(255, 255, 255, 230), tagBuf);
    } else {
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
    }

    ImGui::End();
}

} // namespace btquant::ui
