#include "trades_widget.hpp"
#include "../ui/ui_context.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace btquant::ui {

TradesWidget::TradesWidget() = default;
TradesWidget::~TradesWidget() = default;

void TradesWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Trades", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Controls
    ImGui::Text("Controls:");
    static const double filter_min = 0.0, filter_max = 1000.0;
    ImGui::SliderScalar("Min Size Filter", ImGuiDataType_Double, &m_filterSize, &filter_min, &filter_max, "%.2f");
    
    ImGui::Separator();

    // Mock trade data for demonstration
    static std::vector<data::Trade> mockTrades;
    
    // Generate mock trades periodically
    static auto lastUpdate = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 500) { // Every 500ms
        // Add a few mock trades
        for (int i = 0; i < 3; i++) {
            data::Trade trade;
            trade.id = mockTrades.size();
            trade.price = 99.5 + (rand() % 100) / 100.0; // Random price around 100
            trade.size = 10.0 + (rand() % 100); // Random size between 10-110
            trade.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            trade.isBuy = (rand() % 2 == 0); // Random buy/sell
            
            mockTrades.insert(mockTrades.begin(), trade);
            
            // Keep only the last 50 trades
            if (mockTrades.size() > 50) {
                mockTrades.pop_back();
            }
        }
        lastUpdate = now;
    }

    // Display trade table
    ImGui::Text("Recent Trades");
    ImGui::Columns(4, "TradesTable", true);
    ImGui::SetColumnWidth(0, 80);  // Time
    ImGui::SetColumnWidth(1, 80);  // Price
    ImGui::SetColumnWidth(2, 80);  // Size
    ImGui::SetColumnWidth(3, 60);  // Side
    
    ImGui::Text("Time"); ImGui::NextColumn();
    ImGui::Text("Price"); ImGui::NextColumn();
    ImGui::Text("Size"); ImGui::NextColumn();
    ImGui::Text("Side"); ImGui::NextColumn();
    ImGui::Separator();

    // Display trades
    for (const auto& trade : mockTrades) {
        // Only show trades above the filter size
        if (trade.size < m_filterSize) continue;
        
        // Time
        auto timePoint = std::chrono::system_clock::time_point(std::chrono::microseconds(trade.timestamp));
        auto timeT = std::chrono::system_clock::to_time_t(timePoint);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timePoint.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&timeT), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        
        ImGui::Text("%s", ss.str().c_str());
        ImGui::NextColumn();
        
        // Price
        ImGui::Text("%.4f", trade.price);
        ImGui::NextColumn();
        
        // Size
        ImGui::Text("%.2f", trade.size);
        ImGui::NextColumn();
        
        // Side with color
        if (trade.isBuy) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));  // Green
            ImGui::Text("BUY ");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));  // Red
            ImGui::Text("SELL");
        }
        ImGui::PopStyleColor();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Separator();
    
    // Stats
    int totalTrades = mockTrades.size();
    double totalVolume = 0;
    double buyVolume = 0, sellVolume = 0;
    
    for (const auto& trade : mockTrades) {
        totalVolume += trade.size;
        if (trade.isBuy) {
            buyVolume += trade.size;
        } else {
            sellVolume += trade.size;
        }
    }
    
    ImGui::Text("Total Trades: %d", totalTrades);
    ImGui::Text("Total Volume: %.2f", totalVolume);
    ImGui::Text("Buy Volume: %.2f", buyVolume);
    ImGui::Text("Sell Volume: %.2f", sellVolume);
    ImGui::Text("Delta (Buy-Sell): %.2f", buyVolume - sellVolume);

    ImGui::End();
}

void TradesWidget::setFilter(double minSize) {
    m_filterSize = minSize;
}

void TradesWidget::reset() {
    // Reset filters and stats
    m_filterSize = 0;
}

} // namespace btquant::ui