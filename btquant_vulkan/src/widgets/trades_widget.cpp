#include "trades_widget.hpp"
#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"
#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace btquant::ui {

TradesWidget::TradesWidget() = default;
TradesWidget::~TradesWidget() = default;

void TradesWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void TradesWidget::render() {
    if (!m_initialized) {
        m_initialized = true;
    }

    ImGui::Begin("Trades", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Controls:");
    static const double filter_min = 0.0, filter_max = 1000.0;
    ImGui::SliderScalar("Min Size Filter", ImGuiDataType_Double, &m_filterSize, &filter_min, &filter_max, "%.2f");

    ImGui::Separator();

    // Source: live snapshot OR synthetic fallback.
    std::vector<data::Trade> trades;
    bool isLive = false;
    uint64_t seq = 0;
    if (m_data) {
        auto snap = m_data->snapshot(50);
        if (snap.snapshot_seq > 0) {
            trades = std::move(snap.recent_trades);
            seq = snap.snapshot_seq;
            isLive = true;
        }
    }
    if (!isLive) {
        // Synthetic fallback when no producer is running.
        static auto lastUpdate = std::chrono::steady_clock::now();
        static std::vector<data::Trade> fallbackTrades;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 500) {
            for (int i = 0; i < 3; ++i) {
                data::Trade trade;
                trade.id = fallbackTrades.size();
                trade.price = 99.5 + (rand() % 100) / 100.0;
                trade.size = 10.0 + (rand() % 100);
                trade.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                trade.isBuy = (rand() % 2 == 0);
                fallbackTrades.insert(fallbackTrades.begin(), trade);
                if (fallbackTrades.size() > 50) fallbackTrades.pop_back();
            }
            lastUpdate = now;
        }
        trades = fallbackTrades;
    }

    ImGui::Text("Recent Trades (%s, count=%zu)", isLive ? "LIVE" : "synthetic", trades.size());
    ImGui::Columns(4, "TradesTable", true);
    ImGui::SetColumnWidth(0, 80);
    ImGui::SetColumnWidth(1, 80);
    ImGui::SetColumnWidth(2, 80);
    ImGui::SetColumnWidth(3, 60);

    ImGui::Text("Time"); ImGui::NextColumn();
    ImGui::Text("Price"); ImGui::NextColumn();
    ImGui::Text("Size"); ImGui::NextColumn();
    ImGui::Text("Side"); ImGui::NextColumn();
    ImGui::Separator();

    for (const auto& trade : trades) {
        if (trade.size < m_filterSize) continue;

        auto timePoint = std::chrono::system_clock::time_point(std::chrono::microseconds(trade.timestamp));
        auto timeT = std::chrono::system_clock::to_time_t(timePoint);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timePoint.time_since_epoch()) % 1000;
        std::stringstream ss;
        ss << std::put_time(std::localtime(&timeT), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        ImGui::Text("%s", ss.str().c_str());
        ImGui::NextColumn();

        ImGui::Text("%.4f", trade.price);
        ImGui::NextColumn();
        ImGui::Text("%.2f", trade.size);
        ImGui::NextColumn();
        if (trade.isBuy) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
            ImGui::Text("BUY ");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
            ImGui::Text("SELL");
        }
        ImGui::PopStyleColor();
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Separator();

    double totalVol = 0, buyVol = 0, sellVol = 0;
    for (const auto& t : trades) {
        totalVol += t.size;
        if (t.isBuy) buyVol += t.size; else sellVol += t.size;
    }
    ImGui::Text("Total: %zu   Volume: %.2f   Buy: %.2f   Sell: %.2f   Delta: %.2f%s",
                trades.size(), totalVol, buyVol, sellVol, buyVol - sellVol,
                isLive ? "" : "   [snap#0]");

    ImGui::End();
}

void TradesWidget::setFilter(double minSize) {
    m_filterSize = minSize;
}

void TradesWidget::reset() {
    m_filterSize = 0;
}

}  // namespace btquant::ui
