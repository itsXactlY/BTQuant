#include "order_book_widget.hpp"
#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace btquant::ui {

OrderBookWidget::OrderBookWidget() = default;
OrderBookWidget::~OrderBookWidget() = default;

void OrderBookWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

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

    // Source selection: live data or synthetic fallback.
    bool isLive = false;
    data::OrderBook book{};
    if (m_data) {
        auto snap = m_data->snapshot(1);
        if (snap.snapshot_seq > 0) {
            book = snap.order_book;
            isLive = true;
        }
    }

    if (!isLive) {
        // Synthetic fallback so the widget still shows something when no
        // producer is running. Sticks close to 100.0 with small drift.
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

    // Display order book table
    ImGui::Columns(4, "OrderBook", true);
    ImGui::SetColumnWidth(0, 100);
    ImGui::SetColumnWidth(1, 100);
    ImGui::SetColumnWidth(2, 100);
    ImGui::SetColumnWidth(3, 100);

    ImGui::Text("Bid Size"); ImGui::NextColumn();
    ImGui::Text("Bid Price"); ImGui::NextColumn();
    ImGui::Text("Ask Price"); ImGui::NextColumn();
    ImGui::Text("Ask Size"); ImGui::NextColumn();
    ImGui::Separator();

    // Display bid levels (top of book = back of vector = highest bid)
    int shown = 0;
    for (int i = static_cast<int>(book.bidCount) - 1; i >= 0 && shown < m_maxLevels; --i) {
        const auto& level = book.bids[i];
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));  // green
        ImGui::Text("%.2f", level.size); ImGui::NextColumn();
        ImGui::PopStyleColor();
        ImGui::Text("%.4f", level.price); ImGui::NextColumn();
        ImGui::NextColumn(); ImGui::NextColumn();
        ++shown;
    }

    // Display ask levels (front of vector = lowest ask)
    for (int i = 0; i < static_cast<int>(book.askCount) && i < m_maxLevels; ++i) {
        const auto& level = book.asks[i];
        ImGui::NextColumn(); ImGui::NextColumn();
        ImGui::Text("%.4f", level.price); ImGui::NextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));  // red
        ImGui::Text("%.2f", level.size); ImGui::PopStyleColor();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Separator();

    if (isLive && book.bidCount > 0 && book.askCount > 0) {
        double midPrice = (book.bids[book.bidCount - 1].price + book.asks[0].price) * 0.5;
        double spread = book.asks[0].price - book.bids[book.bidCount - 1].price;
        double spreadPct = (spread / midPrice) * 100.0;
        ImGui::Text("Mid Price: %.4f   Spread: %.4f (%.2f%%)   Source: LIVE",
                    midPrice, spread, spreadPct);
    } else {
        ImGui::Text("Mid Price: %.4f   Source: synthetic fallback (no producer)",
                    book.midPrice);
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

}  // namespace btquant::ui
