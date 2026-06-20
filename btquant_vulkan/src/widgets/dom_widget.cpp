#include "dom_widget.hpp"
#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace btquant::ui {

DOMWidget::DOMWidget() = default;
DOMWidget::~DOMWidget() = default;

void DOMWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void DOMWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Depth of Market (DOM)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

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
    // Heatmap toggle — swaps the bar-chart renderer for a heat-strip
    // view (one row per price level, full-width cell, color intensity
    // by size relative to the max). The user can flip between views at
    // runtime without losing data binding.
    ImGui::Checkbox("Heatmap view", &m_heatmapMode);
    if (m_heatmapMode) {
        ImGui::SliderFloat("Cell height (px)", &m_cellHeightPx, 1.0f, 16.0f, "%.1f");
    }

    ImGui::Separator();

    bool isLive = false;
    data::OrderBook book{};
    if (m_data) {
        auto snap = m_data->snapshot(1);
        if (snap.snapshot_seq > 0) {
            book = snap.order_book;
            isLive = true;
        }
    }

    // Synthetic fallback.
    if (!isLive) {
        static double fallbackMid = 100.0;
        for (int i = 0; i < m_maxLevels; ++i) {
            double bid = fallbackMid - (i + 1) * m_priceGrouping - (rand() % 100) / 5000.0;
            double ask = fallbackMid + (i + 1) * m_priceGrouping + (rand() % 100) / 5000.0;
            double size = 100.0 + (rand() % 100);
            book.bids[i] = {bid, size, 0};
            book.asks[i] = {ask, size, 0};
            book.bidCount = book.askCount = i + 1;
        }
        book.midPrice = fallbackMid;
        fallbackMid += ((rand() % 100) - 50) / 5000.0;
    }

    ImGui::Text("Depth of Market Chart%s", isLive ? "" : " (synthetic)");

    double maxSize = 1.0;
    for (size_t i = 0; i < book.bidCount; ++i) maxSize = std::max(maxSize, book.bids[i].size);
    for (size_t i = 0; i < book.askCount; ++i) maxSize = std::max(maxSize, book.asks[i].size);

    ImVec2 canvasSize = ImVec2(ImGui::GetWindowWidth() - 40, 200);
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("canvas", canvasSize);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(canvasPos,
        ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
        IM_COL32(30, 30, 50, 200));

    if (book.bidCount > 0 && book.askCount > 0) {
        double centerPrice = (book.bids[book.bidCount - 1].price + book.asks[0].price) * 0.5;
        double priceRange = book.bids[0].price - book.asks[book.askCount - 1].price;
        if (priceRange <= 0) priceRange = m_priceGrouping * m_maxLevels;
        float pixelsPerPrice = canvasSize.y / static_cast<float>(priceRange);

        // Bid bars (green, left side)
        for (size_t i = 0; i < book.bidCount && i < static_cast<size_t>(m_maxLevels); ++i) {
            const auto& level = book.bids[i];
            float yPos = canvasPos.y + static_cast<float>(centerPrice - level.price) * pixelsPerPrice;
            float barWidth = static_cast<float>((level.size / maxSize) * (canvasSize.x / 2 - 10));
            ImVec2 p1(canvasPos.x + canvasSize.x / 2 - barWidth, yPos);
            ImVec2 p2(canvasPos.x + canvasSize.x / 2, yPos + 5);
            drawList->AddRectFilled(p1, p2, IM_COL32(0, 255, 0, 150));
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << level.price;
            drawList->AddText(ImVec2(canvasPos.x + canvasSize.x / 2 + 5, yPos),
                              IM_COL32(200, 200, 200, 255), ss.str().c_str());
        }
        // Ask bars (red, right side)
        for (size_t i = 0; i < book.askCount && i < static_cast<size_t>(m_maxLevels); ++i) {
            const auto& level = book.asks[i];
            float yPos = canvasPos.y + static_cast<float>(centerPrice - level.price) * pixelsPerPrice;
            float barWidth = static_cast<float>((level.size / maxSize) * (canvasSize.x / 2 - 10));
            ImVec2 p1(canvasPos.x + canvasSize.x / 2, yPos);
            ImVec2 p2(canvasPos.x + canvasSize.x / 2 + barWidth, yPos + 5);
            drawList->AddRectFilled(p1, p2, IM_COL32(255, 0, 0, 150));
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << level.price;
            drawList->AddText(ImVec2(canvasPos.x + canvasSize.x / 2 - 60, yPos),
                              IM_COL32(200, 200, 200, 255), ss.str().c_str());
        }
        // Mid line
        float midY = canvasPos.y;
        drawList->AddLine(ImVec2(canvasPos.x, midY),
                          ImVec2(canvasPos.x + canvasSize.x, midY),
                          IM_COL32(255, 255, 255, 100));
    }

    if (m_heatmapMode && book.bidCount > 0 && book.askCount > 0) {
        // Heatmap branch — vertical heat strip. One row per price
        // level, full canvas width, color intensity by size relative
        // to the local max. Bid rows fade green→cyan, ask rows fade
        // red→yellow. Largest level on each side gets a price label.
        double centerPrice = (book.bids[book.bidCount - 1].price + book.asks[0].price) * 0.5;
        double priceRange = book.bids[0].price - book.asks[book.askCount - 1].price;
        if (priceRange <= 0) priceRange = m_priceGrouping * m_maxLevels;
        float pixelsPerPrice = canvasSize.y / static_cast<float>(priceRange);
        float cellH = std::max(1.0f, m_cellHeightPx);
        int maxLabelLevel = std::max(1, m_maxLevels / 4);  // label only top quartile

        // Bid rows (bottom-up: best bid at mid, lowest bid at bottom).
        for (size_t i = 0; i < book.bidCount && i < static_cast<size_t>(m_maxLevels); ++i) {
            const auto& level = book.bids[i];
            float yPos = canvasPos.y + static_cast<float>(centerPrice - level.price) * pixelsPerPrice;
            float intensity = std::min(1.0f, static_cast<float>(level.size / maxSize));
            // Green base (0,200,80) lerp to cyan (0,255,255) by intensity.
            ImU32 col = IM_COL32(
                static_cast<int>(0 * intensity),
                static_cast<int>(200 + 55 * intensity),
                static_cast<int>(80 + 175 * intensity),
                static_cast<int>(120 + 135 * intensity));
            drawList->AddRectFilled(
                ImVec2(canvasPos.x, yPos),
                ImVec2(canvasPos.x + canvasSize.x, yPos + cellH),
                col);
            if (static_cast<int>(i) < maxLabelLevel) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << level.price
                   << "  " << static_cast<int>(level.size);
                drawList->AddText(ImVec2(canvasPos.x + 4, yPos),
                                  IM_COL32(255, 255, 255, 220), ss.str().c_str());
            }
        }
        // Ask rows (top-down: best ask at mid, highest ask at top).
        for (size_t i = 0; i < book.askCount && i < static_cast<size_t>(m_maxLevels); ++i) {
            const auto& level = book.asks[i];
            float yPos = canvasPos.y + static_cast<float>(centerPrice - level.price) * pixelsPerPrice;
            float intensity = std::min(1.0f, static_cast<float>(level.size / maxSize));
            // Red base (220,40,40) lerp to yellow (255,230,40) by intensity.
            ImU32 col = IM_COL32(
                static_cast<int>(220 + 35 * intensity),
                static_cast<int>(40 + 190 * intensity),
                static_cast<int>(40 + 0 * intensity),
                static_cast<int>(120 + 135 * intensity));
            drawList->AddRectFilled(
                ImVec2(canvasPos.x, yPos),
                ImVec2(canvasPos.x + canvasSize.x, yPos + cellH),
                col);
            if (static_cast<int>(i) < maxLabelLevel) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << level.price
                   << "  " << static_cast<int>(level.size);
                drawList->AddText(ImVec2(canvasPos.x + canvasSize.x - 90, yPos),
                                  IM_COL32(255, 255, 255, 220), ss.str().c_str());
            }
        }
        // Mid line on top.
        float midY = canvasPos.y;
        drawList->AddLine(ImVec2(canvasPos.x, midY),
                          ImVec2(canvasPos.x + canvasSize.x, midY),
                          IM_COL32(255, 255, 255, 200));
    }

    ImGui::Separator();

    if (isLive && book.bidCount > 0 && book.askCount > 0) {
        double mid = (book.bids[book.bidCount - 1].price + book.asks[0].price) * 0.5;
        double spread = book.asks[0].price - book.bids[book.bidCount - 1].price;
        ImGui::Text("Best Bid: %.4f   Best Ask: %.4f   Mid: %.4f   Spread: %.4f",
                    book.bids[book.bidCount - 1].price, book.asks[0].price, mid, spread);
    } else {
        ImGui::Text("Best Bid: %.4f   Best Ask: %.4f   (no producer)",
                    book.bids[book.bidCount - 1].price,
                    book.askCount > 0 ? book.asks[0].price : 0.0);
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

void DOMWidget::setHeatmapMode(bool on) {
    m_heatmapMode = on;
}

}  // namespace btquant::ui
