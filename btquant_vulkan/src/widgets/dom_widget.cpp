#include "dom_widget.hpp"
#include "../ui/ui_context.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace btquant::ui {

DOMWidget::DOMWidget() = default;
DOMWidget::~DOMWidget() = default;

void DOMWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Depth of Market (DOM)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // Controls
    ImGui::Text("Controls:");
    static const double price_min = 0.001, price_max = 1.0;
    ImGui::SliderScalar("Price Grouping", ImGuiDataType_Double, &m_priceGrouping, &price_min, &price_max, "%.4f");
    ImGui::SliderInt("Max Levels", &m_maxLevels, 10, 50);
    if (ImGui::BeginCombo("Alignment", m_alignment)) {
        if (ImGui::Selectable("Left")) m_alignment = "Left";
        if (ImGui::Selectable("Center")) m_alignment = "Center";
        if (ImGui::Selectable("Right")) m_alignment = "Right";
        ImGui::EndCombo();
    }
    
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

    // Display DOM chart
    ImGui::Text("Depth of Market Chart");
    
    // Calculate max size for scaling
    double maxSize = 0;
    for (const auto& level : mockBids) maxSize = std::max(maxSize, level.size);
    for (const auto& level : mockAsks) maxSize = std::max(maxSize, level.size);
    
    // Display depth chart
    ImVec2 canvasSize = ImVec2(ImGui::GetWindowWidth() - 40, 200);
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    
    ImGui::InvisibleButton("canvas", canvasSize);
    
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    
    // Draw background
    drawList->AddRectFilled(canvasPos, 
                           ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                           IM_COL32(30, 30, 50, 200));
    
    // Find center price (between highest bid and lowest ask)
    if (!mockBids.empty() && !mockAsks.empty()) {
        double centerPrice = (mockBids[0].price + mockAsks[0].price) / 2.0;
        
        // Draw price levels
        float priceRange = (mockBids[0].price - mockAsks.back().price);
        float pixelsPerPrice = canvasSize.y / priceRange;
        
        // Draw bid levels (green bars)
        for (size_t i = 0; i < std::min((size_t)m_maxLevels, mockBids.size()); i++) {
            auto& level = mockBids[i];
            
            float yPos = canvasPos.y + (centerPrice - level.price) * pixelsPerPrice;
            float barWidth = (level.size / maxSize) * (canvasSize.x / 2 - 10);
            
            // Bid bars on the left side
            ImVec2 p1(canvasPos.x + canvasSize.x / 2 - barWidth, yPos);
            ImVec2 p2(canvasPos.x + canvasSize.x / 2, yPos + 5);
            
            drawList->AddRectFilled(p1, p2, IM_COL32(0, 255, 0, 150)); // Green
            
            // Price label
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << level.price;
            drawList->AddText(ImVec2(canvasPos.x + canvasSize.x / 2 + 5, yPos), IM_COL32(200, 200, 200, 255), ss.str().c_str());
        }
        
        // Draw ask levels (red bars)
        for (size_t i = 0; i < std::min((size_t)m_maxLevels, mockAsks.size()); i++) {
            auto& level = mockAsks[i];
            
            float yPos = canvasPos.y + (centerPrice - level.price) * pixelsPerPrice;
            float barWidth = (level.size / maxSize) * (canvasSize.x / 2 - 10);
            
            // Ask bars on the right side
            ImVec2 p1(canvasPos.x + canvasSize.x / 2, yPos);
            ImVec2 p2(canvasPos.x + canvasSize.x / 2 + barWidth, yPos + 5);
            
            drawList->AddRectFilled(p1, p2, IM_COL32(255, 0, 0, 150)); // Red
            
            // Price label
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << level.price;
            drawList->AddText(ImVec2(canvasPos.x + canvasSize.x / 2 - 60, yPos), IM_COL32(200, 200, 200, 255), ss.str().c_str());
        }
        
        // Draw center line (mid price)
        float midY = canvasPos.y + (centerPrice - centerPrice) * pixelsPerPrice;
        drawList->AddLine(ImVec2(canvasPos.x, midY), ImVec2(canvasPos.x + canvasSize.x, midY), IM_COL32(255, 255, 255, 100));
    }

    ImGui::Separator();
    
    // Stats
    if (!mockBids.empty() && !mockAsks.empty()) {
        double midPrice = (mockBids[0].price + mockAsks[0].price) / 2.0;
        double spread = mockAsks[0].price - mockBids[0].price;
        double spreadPercent = (spread / midPrice) * 100;
        
        ImGui::Text("Best Bid: %.4f", mockBids[0].price);
        ImGui::Text("Best Ask: %.4f", mockAsks[0].price);
        ImGui::Text("Mid Price: %.4f", midPrice);
        ImGui::Text("Spread: %.4f (%.2f%%)", spread, spreadPercent);
    }

    ImGui::End();
}

void DOMWidget::setPriceGrouping(double value) {
    m_priceGrouping = value;
}

void DOMWidget::setMaxLevels(int levels) {
    m_maxLevels = levels;
}

void DOMWidget::setAlignment(const char* mode) {
    m_alignment = mode;
}

} // namespace btquant::ui