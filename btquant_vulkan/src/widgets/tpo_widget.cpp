#include "tpo_widget.hpp"
#include "../ui/ui_context.hpp"
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

void TPOWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("TPO (Time Price Opportunity)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Controls
    ImGui::Text("Controls:");
    ImGui::SliderInt("Session Period (min)", &m_sessionPeriod, 5, 120);
    
    ImGui::Separator();

    // Mock TPO data for demonstration
    static std::map<double, int> tpoData; // price -> time periods count
    static std::vector<data::Candle> mockCandles;
    
    // Generate mock TPO data periodically
    static auto lastUpdate = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 1000) { // Every second
        // Add mock candle
        data::Candle candle;
        static double lastPrice = 100.0;
        candle.open = lastPrice;
        candle.close = lastPrice + ((rand() % 100 - 50) / 1000.0); // Small random movement
        candle.high = std::max(candle.open, candle.close) + (rand() % 50) / 1000.0;
        candle.low = std::min(candle.open, candle.close) - (rand() % 50) / 1000.0;
        candle.volume = 100.0 + (rand() % 100);
        candle.startTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() - 60000000; // 1 minute ago
        candle.endTime = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        mockCandles.insert(mockCandles.begin(), candle);
        lastPrice = candle.close;
        
        // Keep only the last 50 candles
        if (mockCandles.size() > 50) {
            mockCandles.pop_back();
        }
        
        lastUpdate = now;
    }

    // Display TPO chart
    if (ImPlot::BeginPlot("TPO Chart", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Price", "Time Periods Active");
        
        // Prepare data for plotting
        if (!tpoData.empty()) {
            std::vector<double> prices, counts;
            for (const auto& pair : tpoData) {
                prices.push_back(pair.first);
                counts.push_back(pair.second);
            }
            
            if (!prices.empty()) {
                ImPlot::PlotBars("TPO Activity", counts.data(), (int)counts.size(), 0.67);
            }
        }
        
        ImPlot::EndPlot();
    }

    ImGui::Separator();
    
    // Display recent candles
    if (!mockCandles.empty()) {
        ImGui::Text("Recent Candles (OHLC)");
        ImGui::Columns(6, "CandleData", true);
        ImGui::SetColumnWidth(0, 60);  // Time
        ImGui::SetColumnWidth(1, 70);  // Open
        ImGui::SetColumnWidth(2, 70);  // High
        ImGui::SetColumnWidth(3, 70);  // Low
        ImGui::SetColumnWidth(4, 70);  // Close
        ImGui::SetColumnWidth(5, 70);  // Volume
        
        ImGui::Text("Time"); ImGui::NextColumn();
        ImGui::Text("Open"); ImGui::NextColumn();
        ImGui::Text("High"); ImGui::NextColumn();
        ImGui::Text("Low"); ImGui::NextColumn();
        ImGui::Text("Close"); ImGui::NextColumn();
        ImGui::Text("Volume"); ImGui::NextColumn();
        ImGui::Separator();

        // Show last 10 candles
        int shown = 0;
        for (const auto& candle : mockCandles) {
            if (shown++ >= 10) break;
            
            // Time
            auto timePoint = std::chrono::system_clock::time_point(std::chrono::microseconds(candle.startTime));
            auto timeT = std::chrono::system_clock::to_time_t(timePoint);
            
            std::stringstream ss;
            ss << std::put_time(std::localtime(&timeT), "%H:%M");
            
            ImGui::Text("%s", ss.str().c_str());
            ImGui::NextColumn();
            
            // Open
            ImGui::Text("%.4f", candle.open);
            ImGui::NextColumn();
            
            // High
            ImGui::Text("%.4f", candle.high);
            ImGui::NextColumn();
            
            // Low
            ImGui::Text("%.4f", candle.low);
            ImGui::NextColumn();
            
            // Close
            ImGui::Text("%.4f", candle.close);
            ImGui::NextColumn();
            
            // Volume
            ImGui::Text("%.2f", candle.volume);
            ImGui::NextColumn();
        }

        ImGui::Columns(1);
    }

    ImGui::Separator();
    
    // Stats
    ImGui::Text("Session Period: %d minutes", m_sessionPeriod);
    if (!mockCandles.empty()) {
        double highestHigh = mockCandles[0].high;
        double lowestLow = mockCandles[0].low;
        double totalVolume = 0;
        
        for (const auto& candle : mockCandles) {
            highestHigh = std::max(highestHigh, candle.high);
            lowestLow = std::min(lowestLow, candle.low);
            totalVolume += candle.volume;
        }
        
        ImGui::Text("Session High: %.4f", highestHigh);
        ImGui::Text("Session Low: %.4f", lowestLow);
        ImGui::Text("Session Range: %.4f", highestHigh - lowestLow);
        ImGui::Text("Total Volume: %.2f", totalVolume);
    }

    ImGui::End();
}

void TPOWidget::setSessionPeriod(int minutes) {
    m_sessionPeriod = minutes;
}

} // namespace btquant::ui