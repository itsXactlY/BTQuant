#include "order_book_widget.hpp"
#include "../ui/ui_context.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace btquant::ui {

OrderBookWidget::OrderBookWidget() = default;
OrderBookWidget::~OrderBookWidget() = default;

void OrderBookWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Order Book", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Controls
    ImGui::Text("Controls:");
    static const double price_min = 0.001, price_max = 1.0;
    ImGui::SliderScalar("Price Grouping", ImGuiDataType_Double, &m_priceGrouping, &price_min, &price_max, "%.4f");
    ImGui::SliderInt("Max Levels", &m_maxLevels, 10, 50);
    ImGui::Checkbox("Show USD", &m_showUSD);
    
    ImGui::Separator();

    // Mock data for demonstration
    static std::vector<data::OrderBookLevel> mockBids, mockAsks;
    
    // Generate mock data if empty
    if (mockBids.empty() || mockAsks.empty()) {
        // Generate mock bid levels
        double bidPrice = 100.0;
        for (int i = 0; i < m_maxLevels; i++) {
            data::OrderBookLevel level;
            level.price = bidPrice;
            level.size = 100.0 + (rand() % 100);  // Random size between 100-200
            level.cumSize = level.size + (i > 0 ? mockBids[i-1].cumSize : 0);
            mockBids.push_back(level);
            bidPrice -= m_priceGrouping;
        }
        
        // Generate mock ask levels
        double askPrice = 101.0;
        for (int i = 0; i < m_maxLevels; i++) {
            data::OrderBookLevel level;
            level.price = askPrice;
            level.size = 100.0 + (rand() % 100);  // Random size between 100-200
            level.cumSize = level.size + (i > 0 ? mockAsks[i-1].cumSize : 0);
            mockAsks.push_back(level);
            askPrice += m_priceGrouping;
        }
    }

    // Display order book
    ImGui::Columns(4, "OrderBook", true);
    ImGui::SetColumnWidth(0, 100);  // Bid Size
    ImGui::SetColumnWidth(1, 100);  // Bid Price
    ImGui::SetColumnWidth(2, 100);  // Ask Price
    ImGui::SetColumnWidth(3, 100);  // Ask Size
    
    ImGui::Text("Bid Size"); ImGui::NextColumn();
    ImGui::Text("Bid Price"); ImGui::NextColumn();
    ImGui::Text("Ask Price"); ImGui::NextColumn();
    ImGui::Text("Ask Size"); ImGui::NextColumn();
    ImGui::Separator();

    // Display bid levels
    for (int i = std::max(0, (int)mockBids.size() - m_maxLevels); i < (int)mockBids.size(); i++) {
        if (i < 0) continue;
        
        auto& level = mockBids[i];
        
        // Bid size with color
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));  // Green
        ImGui::Text("%.2f", level.size);
        ImGui::PopStyleColor();
        ImGui::NextColumn();
        
        // Bid price
        ImGui::Text("%.4f", level.price);
        ImGui::NextColumn();
        
        // Empty columns for asks (will fill later)
        ImGui::NextColumn();
        ImGui::NextColumn();
    }

    // Move back to ask columns
    ImGui::SetColumnOffset(2, ImGui::GetColumnOffset(1));
    ImGui::SetColumnOffset(3, ImGui::GetColumnOffset(2) + ImGui::GetColumnWidth(2));
    
    // Display ask levels
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - (mockBids.size() * ImGui::GetTextLineHeightWithSpacing()));
    
    for (int i = 0; i < std::min(m_maxLevels, (int)mockAsks.size()); i++) {
        auto& level = mockAsks[i];
        
        // Skip to the ask columns
        ImGui::NextColumn();
        ImGui::NextColumn();
        
        // Ask price
        ImGui::Text("%.4f", level.price);
        ImGui::NextColumn();
        
        // Ask size with color
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));  // Red
        ImGui::Text("%.2f", level.size);
        ImGui::PopStyleColor();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Separator();
    
    // Mid price indicator
    if (!mockBids.empty() && !mockAsks.empty()) {
        double midPrice = (mockBids.back().price + mockAsks[0].price) / 2.0;
        double spread = mockAsks[0].price - mockBids.back().price;
        double spreadPercent = (spread / midPrice) * 100;
        
        ImGui::Text("Mid Price: %.4f", midPrice);
        ImGui::Text("Spread: %.4f (%.2f%%)", spread, spreadPercent);
    }

    ImGui::End();
}

void OrderBookWidget::setPriceGrouping(double value) {
    m_priceGrouping = value;
}

void OrderBookWidget::setMaxLevels(int levels) {
    m_maxLevels = levels;
}

void OrderBookWidget::setShowUSD(bool usd) {
    m_showUSD = usd;
}

} // namespace btquant::ui