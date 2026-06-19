#include "order_book_depth_widget.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace btquant::ui {

OrderBookDepthWidget::OrderBookDepthWidget() = default;
OrderBookDepthWidget::~OrderBookDepthWidget() = default;

void OrderBookDepthWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void OrderBookDepthWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }
    if (!showWindow) return;

    ImGui::Begin("Order Book Depth", &showWindow);

    ImGui::SliderInt("Levels", &m_levels, 5, 50);
    ImGui::Checkbox("Auto-center on mid", &m_autoCenter);

    bool isLive = false;
    data::OrderBook book{};
    if (m_data) {
        auto snap = m_data->snapshot(1);
        if (snap.snapshot_seq > 0) {
            book = snap.order_book;
            isLive = true;
        }
    }

    // Compute totals for imbalance bar.
    double totalBid = 0.0, totalAsk = 0.0;
    for (size_t i = 0; i < book.bidCount; ++i) totalBid += book.bids[i].size;
    for (size_t i = 0; i < book.askCount; ++i) totalAsk += book.asks[i].size;
    double totalVol = totalBid + totalAsk;
    double imbalance = (totalVol > 0) ? (totalBid - totalAsk) / totalVol : 0.0;
    double imbalancePct = imbalance * 100.0;

    // Imbalance gauge at the top.
    ImGui::PushStyleColor(ImGuiCol_Text,
        imbalance > 0 ? IM_COL32(0, 255, 0, 255) : IM_COL32(255, 80, 80, 255));
    ImGui::Text("Imbalance: %+.1f%%  (Bid vol %.2f / Ask vol %.2f)",
                imbalancePct, totalBid, totalAsk);
    ImGui::PopStyleColor();

    // Imbalance bar (horizontal).
    {
        ImVec2 barPos = ImGui::GetCursorScreenPos();
        float barWidth = ImGui::GetContentRegionAvail().x;
        float barHeight = 14.0f;
        // Center line.
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(barPos,
            ImVec2(barPos.x + barWidth, barPos.y + barHeight),
            IM_COL32(30, 30, 30, 200));
        float midX = barPos.x + barWidth * 0.5f;
        dl->AddLine(ImVec2(midX, barPos.y), ImVec2(midX, barPos.y + barHeight),
                    IM_COL32(255, 255, 255, 120));
        // Fill from center toward buy or sell side.
        float fillW = std::abs(imbalance) * barWidth * 0.5f;
        if (imbalance >= 0) {
            dl->AddRectFilled(ImVec2(midX, barPos.y + 2),
                              ImVec2(midX + fillW, barPos.y + barHeight - 2),
                              IM_COL32(0, 200, 100, 220));
        } else {
            dl->AddRectFilled(ImVec2(midX - fillW, barPos.y + 2),
                              ImVec2(midX, barPos.y + barHeight - 2),
                              IM_COL32(220, 60, 60, 220));
        }
        ImGui::Dummy(ImVec2(barWidth, barHeight));
    }

    ImGui::Separator();

    // Determine display window of prices.
    double centerPrice = book.midPrice;
    if (centerPrice <= 0 && book.bidCount > 0 && book.askCount > 0) {
        centerPrice = (book.bids[book.bidCount - 1].price + book.asks[0].price) * 0.5;
    }

    // Build rows: ask side reversed (lowest ask first, then upward), then bid side.
    // Render: asks ABOVE the mid line, bids BELOW.
    // For visualization, show the best N levels on each side.
    const int showLevels = std::min({m_levels,
        static_cast<int>(book.askCount),
        static_cast<int>(book.bidCount)});
    if (showLevels == 0) {
        ImGui::Text("No book data");
        ImGui::End();
        return;
    }

    // Compute cumulative depths.
    double maxCum = 1.0;
    {
        double cum = 0;
        for (int i = static_cast<int>(book.askCount) - 1; i >= 0; --i) {
            cum += book.asks[i].size;
            maxCum = std::max(maxCum, cum);
        }
        cum = 0;
        for (size_t i = 0; i < book.bidCount; ++i) {
            cum += book.bids[i].size;
            maxCum = std::max(maxCum, cum);
        }
    }

    ImGui::Columns(5, "OrderBookDepth", true);
    ImGui::SetColumnWidth(0, 70);
    ImGui::SetColumnWidth(1, 70);
    ImGui::SetColumnWidth(2, 90);
    ImGui::SetColumnWidth(3, 70);
    ImGui::SetColumnWidth(4, 70);

    ImGui::Text("Size"); ImGui::NextColumn();
    ImGui::Text("Cum"); ImGui::NextColumn();
    ImGui::Text("Price"); ImGui::NextColumn();
    ImGui::Text("Cum"); ImGui::NextColumn();
    ImGui::Text("Size"); ImGui::NextColumn();
    ImGui::Separator();

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Ask side (top of book first → lowest ask first in display).
    double askCum = 0.0;
    double askCumPrev = 0.0;
    for (int i = 0; i < showLevels; ++i) {
        const auto& level = book.asks[i];
        askCum += level.size;
        // Size col
        ImGui::Text("%.2f", level.size); ImGui::NextColumn();
        // Cum col (running total)
        ImGui::Text("%.2f", askCum); ImGui::NextColumn();
        // Price col (highlight best ask)
        if (i == 0) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 80, 80, 255));
        ImGui::Text("%.4f", level.price);
        if (i == 0) ImGui::PopStyleColor();
        // draw depth bar on ask side (right half)
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            float rowH = ImGui::GetTextLineHeight();
            float barW = (level.size / maxCum) * 90.0f;
            dl->AddRectFilled(ImVec2(p.x, p.y),
                              ImVec2(p.x + barW, p.y + rowH),
                              IM_COL32(255, 60, 60, 100));
        }
        ImGui::NextColumn();
        // Cum from top
        askCumPrev += level.size;
        ImGui::Text("%.0f", askCumPrev); ImGui::NextColumn();
        ImGui::Text("%.2f", level.size); ImGui::NextColumn();
    }

    // Mid separator.
    if (centerPrice > 0) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 200));
        // Compute volume-weighted microprice: (bid*askSize + ask*bidSize) / (bidSize + askSize).
        double microPrice = centerPrice;
        if (book.bidCount > 0 && book.askCount > 0) {
            double bb = book.bids[book.bidCount - 1].price;
            double bs = book.bids[book.bidCount - 1].size;
            double aa = book.asks[0].price;
            double as = book.asks[0].size;
            if (bs + as > 0) microPrice = (bb * as + aa * bs) / (bs + as);
        }
        ImGui::Text("%.4f", microPrice);
        ImGui::PopStyleColor();
        // Skip remaining columns.
        for (int c = 1; c < 5; ++c) ImGui::NextColumn();
    }

    // Bid side (top of book first → highest bid first).
    double bidCum = 0.0;
    double bidCumPrev = 0.0;
    for (int i = static_cast<int>(book.bidCount) - 1;
         i >= static_cast<int>(book.bidCount) - showLevels; --i) {
        const auto& level = book.bids[i];
        bidCum += level.size;
        // Size col
        ImGui::Text("%.2f", level.size); ImGui::NextColumn();
        // Cum col
        ImGui::Text("%.0f", bidCumPrev + level.size); ImGui::NextColumn();
        // Price col (highlight best bid)
        if (i == static_cast<int>(book.bidCount) - 1)
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 100, 255));
        ImGui::Text("%.4f", level.price);
        if (i == static_cast<int>(book.bidCount) - 1)
            ImGui::PopStyleColor();
        // depth bar on bid side (left half)
        {
            ImVec2 p = ImGui::GetCursorScreenPos();
            float rowH = ImGui::GetTextLineHeight();
            float barW = (level.size / maxCum) * 90.0f;
            dl->AddRectFilled(ImVec2(p.x + 70 - barW, p.y),
                              ImVec2(p.x + 70, p.y + rowH),
                              IM_COL32(0, 200, 100, 100));
        }
        ImGui::NextColumn();
        bidCumPrev += level.size;
        ImGui::Text("%.2f", bidCumPrev); ImGui::NextColumn();
        ImGui::Text("%.2f", level.size); ImGui::NextColumn();
    }

    ImGui::Columns(1);

    ImGui::Separator();
    if (isLive) {
        ImGui::Text("Source: LIVE   Spread: %.4f   Levels: %d/%d",
                    (book.askCount > 0 && book.bidCount > 0)
                        ? book.asks[0].price - book.bids[book.bidCount - 1].price : 0.0,
                    showLevels, m_levels);
    } else {
        ImGui::Text("Source: synthetic (no producer)");
    }

    ImGui::End();
}

}  // namespace btquant::ui
