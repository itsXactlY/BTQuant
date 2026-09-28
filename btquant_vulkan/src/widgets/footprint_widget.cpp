#include "footprint_widget.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>

namespace btquant::ui {

FootprintWidget::FootprintWidget() = default;
FootprintWidget::~FootprintWidget() = default;

void FootprintWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

// Aggregated cluster cell — one price level within one candle minute.
struct ClusterCell {
    double bidSize = 0;   // seller-initiated volume (isBuy=false)
    double askSize = 0;   // buyer-initiated volume (isBuy=true)
    uint32_t trades = 0;
};

void FootprintWidget::render() {
    if (!m_initialized) m_initialized = true;
    if (!showWindow) return;

    ImGui::Begin("Footprint Chart", &showWindow);

    ImGui::SliderInt("Price Bucket (ticks)", &m_priceBucketTicks, 1, 100);
    ImGui::SliderInt("Candles Shown", &m_candlesShown, 1, 12);

    bool isLive = false;
    std::vector<data::Trade> trades;
    if (m_data) {
        auto snap = m_data->snapshot(256);
        trades = std::move(snap.recent_trades);
        isLive = snap.snapshot_seq > 0 && !trades.empty();
    }

    // Synthesise a fallback when no producer is running: generate trades
    // bucketed into the last 3 minutes with random price/size/buy.
    if (!isLive) {
        static std::vector<data::Trade> synth;
        static auto lastUpdate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 200) {
            uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            uint64_t minute = (now_us / 60'000'000ULL) * 60'000'000ULL;
            for (int i = 0; i < 6; ++i) {
                data::Trade t{};
                t.id = synth.size();
                t.price = 100.0 + (rand() % 200) / 100.0;
                t.size = 1.0 + (rand() % 50) / 10.0;
                // Spread across 3 minutes.
                t.timestamp = minute - (uint64_t)(rand() % 3) * 60'000'000ULL
                              + (uint64_t)(rand() % 60'000'000ULL);
                t.isBuy = (rand() % 2 == 0);
                synth.push_back(t);
                if (synth.size() > 256) synth.erase(synth.begin());
            }
            lastUpdate = now;
        }
        trades = synth;
    }

    if (trades.empty()) {
        ImGui::Text("No trades");
        ImGui::End();
        return;
    }

    // Determine price range from trades.
    double pmin = std::numeric_limits<double>::max();
    double pmax = -std::numeric_limits<double>::max();
    uint64_t tmin = std::numeric_limits<uint64_t>::max();
    uint64_t tmax = 0;
    for (const auto& t : trades) {
        if (t.price < pmin) pmin = t.price;
        if (t.price > pmax) pmax = t.price;
        if (t.timestamp < tmin) tmin = t.timestamp;
        if (t.timestamp > tmax) tmax = t.timestamp;
    }
    if (pmax <= pmin) pmax = pmin + 1.0;
    double range = pmax - pmin;

    // Bucket size: pick a "tick" from the price range (1/100 of range is
    // a reasonable default for illiquid instruments).
    double bucket = std::max(range * 0.005, 1e-4);
    if (m_priceBucketTicks > 1) bucket = std::max(bucket / m_priceBucketTicks, 1e-6);

    // Group trades by (minute_timestamp, price_bucket).
    using Key = std::pair<uint64_t, int64_t>;  // (minute, price_bucket_index)
    std::map<Key, ClusterCell> clusters;
    std::map<uint64_t, int> minuteOrder;  // minute -> index (chronological)
    std::map<int64_t, double> bucketMid;  // bucket idx -> midpoint price

    for (const auto& t : trades) {
        uint64_t minute = (t.timestamp / 60'000'000ULL) * 60'000'000ULL;
        int64_t bucketIdx = static_cast<int64_t>(std::floor(t.price / bucket));
        double bucketPrice = bucketIdx * bucket;
        bucketMid[bucketIdx] = bucketPrice + bucket * 0.5;
        auto& cell = clusters[{minute, bucketIdx}];
        cell.trades++;
        if (t.isBuy) cell.askSize += t.size;  // buyer lifts the ask
        else        cell.bidSize += t.size;  // seller hits the bid
        if (minuteOrder.find(minute) == minuteOrder.end()) {
            minuteOrder[minute] = static_cast<int>(minuteOrder.size());
        }
    }

    if (clusters.empty()) {
        ImGui::Text("No clusters");
        ImGui::End();
        return;
    }

    // Pick the most recent N candles for display (column-major: time → x).
    std::vector<uint64_t> minutes;
    minutes.reserve(minuteOrder.size());
    for (const auto& [m, _] : minuteOrder) minutes.push_back(m);
    std::sort(minutes.begin(), minutes.end());
    if (static_cast<int>(minutes.size()) > m_candlesShown) {
        minutes.erase(minutes.begin(), minutes.begin() + (minutes.size() - m_candlesShown));
    }

    // Collect visible price buckets (rows).
    std::vector<int64_t> bucketIndices;
    for (const auto& [k, _] : clusters) {
        const auto& [minute, bidx] = k;
        if (std::find(minutes.begin(), minutes.end(), minute) != minutes.end()) {
            if (std::find(bucketIndices.begin(), bucketIndices.end(), bidx) == bucketIndices.end()) {
                bucketIndices.push_back(bidx);
            }
        }
    }
    std::sort(bucketIndices.begin(), bucketIndices.end(), std::greater<int64_t>());

    // Layout.
    const float colWidth = 80.0f;
    const float rowHeight = 22.0f;
    const float labelWidth = 70.0f;
    const float totalW = labelWidth + colWidth * minutes.size();
    const float totalH = rowHeight * (bucketIndices.size() + 1);

    ImGui::Text("Time axis (cols): newest %d minutes — Price axis (rows): %d buckets",
                (int)minutes.size(), (int)bucketIndices.size());
    ImGui::SameLine();
    ImGui::TextDisabled("| %s | %zu clusters", isLive ? "LIVE" : "synthetic", clusters.size());

    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Column headers (time labels).
    for (size_t c = 0; c < minutes.size(); ++c) {
        ImVec2 p(origin.x + labelWidth + colWidth * c, origin.y);
        auto timePoint = std::chrono::system_clock::time_point(
            std::chrono::microseconds(minutes[c]));
        auto tt = std::chrono::system_clock::to_time_t(timePoint);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&tt), "%H:%M");
        dl->AddText(ImVec2(p.x + 4, p.y), IM_COL32(200, 200, 200, 255), ss.str().c_str());
    }

    // Each row = a price bucket.
    double maxCell = 1.0;
    for (const auto& [_, cell] : clusters) {
        maxCell = std::max({maxCell, cell.bidSize, cell.askSize});
    }

    for (size_t r = 0; r < bucketIndices.size(); ++r) {
        int64_t bidx = bucketIndices[r];
        double mid = bucketMid[bidx];
        float y = origin.y + rowHeight * (r + 1);
        // Row label (price).
        char label[32];
        std::snprintf(label, sizeof(label), "%.4f", mid);
        dl->AddText(ImVec2(origin.x, y + 2), IM_COL32(220, 220, 220, 255), label);

        // Cells for each (minute, bucket).
        for (size_t c = 0; c < minutes.size(); ++c) {
            auto it = clusters.find({minutes[c], bidx});
            if (it == clusters.end()) continue;
            const ClusterCell& cell = it->second;
            float x = origin.x + labelWidth + colWidth * c;
            // Layout: |  bid @ ask  | in cell.
            // Draw bid (left half) and ask (right half) bars inside the cell.
            float midX = x + colWidth * 0.5f;
            float barMax = colWidth * 0.45f;
            float bidW = (cell.bidSize / maxCell) * barMax;
            float askW = (cell.askSize / maxCell) * barMax;

            // bid (red — sells eating bids) on left.
            dl->AddRectFilled(ImVec2(midX - bidW, y + 2),
                              ImVec2(midX, y + rowHeight - 2),
                              IM_COL32(220, 80, 80, 180));
            // ask (green — buys lifting asks) on right.
            dl->AddRectFilled(ImVec2(midX, y + 2),
                              ImVec2(midX + askW, y + rowHeight - 2),
                              IM_COL32(80, 220, 120, 180));

            // Numeric label centered if cell is wide enough.
            if (colWidth > 60.0f) {
                std::ostringstream ss;
                ss.precision(2);
                ss << std::fixed << cell.bidSize << "@" << cell.askSize;
                ImU32 col = (cell.askSize > cell.bidSize * 1.5f)
                    ? IM_COL32(80, 255, 120, 255)
                    : (cell.bidSize > cell.askSize * 1.5f)
                        ? IM_COL32(255, 80, 80, 255)
                        : IM_COL32(220, 220, 220, 255);
                dl->AddText(ImVec2(x + 4, y + 4), col, ss.str().c_str());
            }
        }
    }

    // Reserve the draw area + a legend below.
    ImGui::Dummy(ImVec2(totalW, totalH + 4));
    ImGui::Text("Legend: bid=red (sells hit bids) | ask=green (buys lift asks)");
    ImGui::Text("Cell color: bright green if ask >> bid, bright red if bid >> ask");

    ImGui::End();
}

}  // namespace btquant::ui
