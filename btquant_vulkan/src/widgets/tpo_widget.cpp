#include "tpo_widget.hpp"
#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <map>
#include <vector>

namespace btquant::ui {

TPOWidget::TPOWidget() = default;
TPOWidget::~TPOWidget() = default;

void TPOWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void TPOWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("TPO (Time Price Opportunity)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Controls:");
    ImGui::SliderInt("Session Period (min)", &m_sessionPeriod, 5, 120);

    ImGui::Separator();

    // Aggregate recent trades into price bins (TPO histogram).
    std::map<double, int> tpoData;
    bool isLive = false;
    if (m_data) {
        auto snap = m_data->snapshot(256);
        if (snap.snapshot_seq > 0 && !snap.recent_trades.empty()) {
            // Price-bin width: 0.1% of mid price (or 0.0001 fallback).
            double mid = (snap.metrics.high + snap.metrics.low) * 0.5;
            if (mid <= 0) {
                mid = snap.recent_trades.front().price;
            }
            double binWidth = std::max(mid * 0.0001, 1e-6);
            for (const auto& t : snap.recent_trades) {
                double bin = std::round(t.price / binWidth) * binWidth;
                tpoData[bin]++;
            }
            isLive = true;
        }
    }

    // Synthetic fallback.
    if (!isLive) {
        static double lastPrice = 100.0;
        static std::map<double, int> synthTpo;
        static auto lastUpdate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 1000) {
            // Push a fake candle.
            double bin = std::round(lastPrice / 0.01) * 0.01;
            synthTpo[bin]++;
            lastPrice += ((rand() % 100) - 50) / 5000.0;
            lastUpdate = now;
        }
        tpoData = synthTpo;
    }

    if (ImPlot::BeginPlot(isLive ? "TPO Chart (LIVE)" : "TPO Chart (synthetic)", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Price", "Time Periods Active");

        if (!tpoData.empty()) {
            std::vector<double> prices, counts;
            for (const auto& [p, c] : tpoData) {
                prices.push_back(p);
                counts.push_back(static_cast<double>(c));
            }
            if (!prices.empty()) {
                ImPlot::PlotBars("TPO Activity", counts.data(), (int)counts.size(), 0.67);
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::Separator();

    // Show OHLC candles — pull directly from the snapshot's recent_candles
    // (computed by the MarketDataProcessor background thread). No more
    // widget-side reconstruction from trades.
    std::vector<data::Candle> candles;
    if (m_data) {
        auto snap = m_data->snapshot(1, 60);
        candles = std::move(snap.recent_candles);
        // The current in-progress candle (if any) shows live activity for
        // the current minute; append it after the finalized ones for the
        // table view so the user sees ongoing volume build-up.
        if (snap.current_candle) {
            candles.push_back(*snap.current_candle);
        }
        isLive = snap.snapshot_seq > 0 && !candles.empty();
    } else {
        // Synthetic fallback candles.
        static std::vector<data::Candle> synthCandles;
        static auto lastUpdate = std::chrono::steady_clock::now();
        static double lastPrice = 100.0;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 1000) {
            data::Candle candle;
            candle.open = lastPrice;
            candle.close = lastPrice + ((rand() % 100 - 50) / 1000.0);
            candle.high = std::max(candle.open, candle.close) + (rand() % 50) / 1000.0;
            candle.low = std::min(candle.open, candle.close) - (rand() % 50) / 1000.0;
            candle.volume = 100.0 + (rand() % 100);
            candle.startTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() - 60000000;
            candle.endTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            synthCandles.insert(synthCandles.begin(), candle);
            if (synthCandles.size() > 50) synthCandles.pop_back();
            lastPrice = candle.close;
            lastUpdate = now;
        }
        candles = synthCandles;
    }

    if (!candles.empty()) {
        ImGui::Text("Recent Candles (OHLC) — %zu finalized + current (%s)",
                    candles.size() - (m_data && candles.size() > 0 ? 1 : 0),
                    isLive ? "LIVE" : "synthetic");
        ImGui::Columns(8, "CandleData", true);
        ImGui::SetColumnWidth(0, 60);  // Time
        ImGui::SetColumnWidth(1, 60);  // Open
        ImGui::SetColumnWidth(2, 60);  // High
        ImGui::SetColumnWidth(3, 60);  // Low
        ImGui::SetColumnWidth(4, 60);  // Close
        ImGui::SetColumnWidth(5, 60);  // Volume
        ImGui::SetColumnWidth(6, 55);  // Delta
        ImGui::SetColumnWidth(7, 45);  // #Trades

        ImGui::Text("Time"); ImGui::NextColumn();
        ImGui::Text("Open"); ImGui::NextColumn();
        ImGui::Text("High"); ImGui::NextColumn();
        ImGui::Text("Low"); ImGui::NextColumn();
        ImGui::Text("Close"); ImGui::NextColumn();
        ImGui::Text("Vol"); ImGui::NextColumn();
        ImGui::Text("Delta"); ImGui::NextColumn();
        ImGui::Text("#"); ImGui::NextColumn();
        ImGui::Separator();

        int shown = 0;
        for (auto it = candles.rbegin(); it != candles.rend() && shown < 10; ++it, ++shown) {
            const auto& candle = *it;
            auto timePoint = std::chrono::system_clock::time_point(
                std::chrono::microseconds(candle.startTime));
            auto timeT = std::chrono::system_clock::to_time_t(timePoint);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&timeT), "%H:%M");
            // Mark the in-progress candle with a "*" so the user can see it
            // building in real time.
            if (candle.startTime == candles.back().startTime) {
                ss << "*";
            }
            ImGui::Text("%s", ss.str().c_str()); ImGui::NextColumn();
            ImGui::Text("%.2f", candle.open); ImGui::NextColumn();
            ImGui::Text("%.2f", candle.high); ImGui::NextColumn();
            ImGui::Text("%.2f", candle.low); ImGui::NextColumn();
            ImGui::Text("%.2f", candle.close); ImGui::NextColumn();
            ImGui::Text("%.1f", candle.volume); ImGui::NextColumn();
            // Color delta: green positive, red negative, gray zero.
            if (candle.delta > 0.01)
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
            else if (candle.delta < -0.01)
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::Text("%+.1f", candle.delta);
            if (candle.delta > 0.01 || candle.delta < -0.01)
                ImGui::PopStyleColor();
            ImGui::NextColumn();
            ImGui::Text("%u", candle.tradeCount); ImGui::NextColumn();
        }
        ImGui::Columns(1);
    }

    ImGui::Separator();

    ImGui::Text("Session Period: %d minutes", m_sessionPeriod);
    if (m_data) {
        auto snap = m_data->snapshot(1);
        if (snap.snapshot_seq > 0 && (snap.metrics.high > 0 || snap.metrics.low > 0)) {
            ImGui::Text("LIVE: High=%.4f  Low=%.4f  Range=%.4f  Vol=%.2f  Trades=%lld",
                        snap.metrics.high, snap.metrics.low,
                        snap.metrics.high - snap.metrics.low,
                        snap.metrics.volume,
                        static_cast<long long>(snap.metrics.tradeCount));
        } else {
            ImGui::Text("(waiting for live data from producer)");
        }
    }

    ImGui::End();
}

void TPOWidget::setSessionPeriod(int minutes) {
    m_sessionPeriod = minutes;
}

}  // namespace btquant::ui
