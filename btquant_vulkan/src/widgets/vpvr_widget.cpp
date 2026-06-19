#include "vpvr_widget.hpp"

#include "../data/market_data_processor.hpp"
#include "../data/market_data.hpp"

#include <imgui.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>

namespace btquant::ui {

VPVRWidget::VPVRWidget() = default;
VPVRWidget::~VPVRWidget() = default;

void VPVRWidget::setMarketData(::btquant::MarketDataProcessor* data) {
    m_data = data;
}

void VPVRWidget::render() {
    if (!m_initialized) m_initialized = true;
    if (!showWindow) return;

    ImGui::Begin("Volume Profile (VPVR)", &showWindow);

    ImGui::SliderInt("Price Bucket Size (× tick)", &m_priceBucketTicks, 1, 50);
    ImGui::SliderFloat("Value Area %%", reinterpret_cast<float*>(&m_valueAreaPct), 0.50f, 0.95f, "%.0f%%");

    bool isLive = false;
    std::vector<data::Trade> trades;
    double vwap = 0.0;
    if (m_data) {
        auto snap = m_data->snapshot(512);  // larger window for VPVR
        trades = std::move(snap.recent_trades);
        vwap = snap.metrics.vwap;
        isLive = snap.snapshot_seq > 0 && !trades.empty();
    }

    // Synthetic fallback.
    if (!isLive) {
        static std::vector<data::Trade> synth;
        static auto lastUpdate = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate).count() > 200) {
            uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            for (int i = 0; i < 6; ++i) {
                data::Trade t{};
                t.id = synth.size();
                t.price = 100.0 + (rand() % 200) / 100.0;
                t.size = 1.0 + (rand() % 50) / 10.0;
                t.timestamp = now_us - (uint64_t)(rand() % 60'000'000);
                t.isBuy = (rand() % 2 == 0);
                synth.push_back(t);
                if (synth.size() > 512) synth.erase(synth.begin());
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

    // Price range + bucket size.
    double pmin = std::numeric_limits<double>::max();
    double pmax = -std::numeric_limits<double>::max();
    for (const auto& t : trades) {
        if (t.price < pmin) pmin = t.price;
        if (t.price > pmax) pmax = t.price;
    }
    if (pmax <= pmin) pmax = pmin + 1.0;
    double range = pmax - pmin;
    double bucket = std::max(range * 0.005, 1e-4);
    if (m_priceBucketTicks > 1) bucket *= m_priceBucketTicks;

    // Aggregate volume per price bucket.
    std::map<int64_t, double> volBuckets;   // bucket_idx → total volume
    std::map<int64_t, double> midByBucket;  // bucket_idx → midpoint price
    double totalVol = 0.0;
    double vwapSum = 0.0;
    for (const auto& t : trades) {
        int64_t bidx = static_cast<int64_t>(std::floor(t.price / bucket));
        volBuckets[bidx] += t.size;
        midByBucket[bidx] = bidx * bucket + bucket * 0.5;
        totalVol += t.size;
        vwapSum += t.price * t.size;
    }
    if (totalVol > 0 && vwap == 0.0) vwap = vwapSum / totalVol;

    // Find POC (bucket with max volume).
    int64_t pocIdx = 0;
    double pocVol = 0;
    for (const auto& [b, v] : volBuckets) {
        if (v > pocVol) { pocVol = v; pocIdx = b; }
    }

    // Compute Value Area: expand from POC outward (alternating up/down by
    // larger-volume neighbor) until cumulative volume reaches target %.
    std::set<int64_t> vaBuckets{pocIdx};
    double vaVol = pocVol;
    auto upIt = volBuckets.find(pocIdx);
    auto downIt = volBuckets.find(pocIdx);
    int64_t upIdx = pocIdx, downIdx = pocIdx;
    while (totalVol > 0 && vaVol / totalVol < m_valueAreaPct) {
        // Advance up and down alternately.
        ++upIdx;
        --downIdx;
        bool added = false;
        if (volBuckets.count(upIdx)) { vaBuckets.insert(upIdx); vaVol += volBuckets[upIdx]; added = true; }
        if (volBuckets.count(downIdx)) { vaBuckets.insert(downIdx); vaVol += volBuckets[downIdx]; added = true; }
        if (!added) break;  // ran off the data
    }
    int64_t vaHighIdx = pocIdx, vaLowIdx = pocIdx;
    for (auto b : vaBuckets) {
        if (b > vaHighIdx) vaHighIdx = b;
        if (b < vaLowIdx)  vaLowIdx  = b;
    }

    // Render.
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    const float totalW = 380.0f;
    const float totalH = 360.0f;

    // Header text.
    ImGui::Text("VPVR — last %zu trades (%s)", trades.size(),
                isLive ? "LIVE" : "synthetic");
    ImGui::SameLine();
    if (vwap > 0) ImGui::TextDisabled("| VWAP %.4f", vwap);

    // Background.
    dl->AddRectFilled(origin,
                      ImVec2(origin.x + totalW, origin.y + totalH),
                      IM_COL32(15, 15, 20, 240));
    dl->AddRect(origin,
                ImVec2(origin.x + totalW, origin.y + totalH),
                IM_COL32(60, 60, 80, 255));

    // Draw volume bars.
    double maxBarVol = 1.0;
    for (const auto& [_, v] : volBuckets) maxBarVol = std::max(maxBarVol, v);
    const float barLeft = origin.x + 60;     // leave 60px for price labels
    const float barMaxW = totalW - 60 - 8;

    // Sort bucket indices by price descending (so high prices render at top).
    std::vector<int64_t> sortedBuckets;
    sortedBuckets.reserve(volBuckets.size());
    for (const auto& [b, _] : volBuckets) sortedBuckets.push_back(b);
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), std::greater<int64_t>());

    // Map price range to vertical pixel range.
    int64_t idxMin = sortedBuckets.back();
    int64_t idxMax = sortedBuckets.front();
    if (idxMax == idxMin) idxMax = idxMin + 1;
    auto idxToY = [&](int64_t b) {
        double t = static_cast<double>(b - idxMax) / static_cast<double>(idxMin - idxMax);
        return origin.y + static_cast<float>(t) * totalH;
    };

    for (int64_t b : sortedBuckets) {
        float y = idxToY(b);
        double mid = midByBucket[b];
        double vol = volBuckets[b];
        float barW = static_cast<float>((vol / maxBarVol) * barMaxW);

        // Color: VA = blue, POC = orange, regular = grey-blue.
        ImU32 col;
        if (b == pocIdx) col = IM_COL32(255, 180, 60, 255);
        else if (vaBuckets.count(b)) col = IM_COL32(80, 160, 230, 180);
        else col = IM_COL32(70, 90, 140, 160);
        dl->AddRectFilled(ImVec2(barLeft, y), ImVec2(barLeft + barW, y + totalH / std::max<size_t>(1, sortedBuckets.size())),
                          col);

        // Price label on the left.
        char priceLabel[24];
        std::snprintf(priceLabel, sizeof(priceLabel), "%.4f", mid);
        dl->AddText(ImVec2(origin.x + 4, y - 2),
                    (b == pocIdx) ? IM_COL32(255, 220, 120, 255) : IM_COL32(220, 220, 220, 255),
                    priceLabel);
    }

    // VWAP horizontal line.
    if (vwap > pmin && vwap < pmax) {
        int64_t vwapIdx = static_cast<int64_t>(std::floor(vwap / bucket));
        if (vwapIdx >= idxMin && vwapIdx <= idxMax) {
            float y = idxToY(vwapIdx);
            dl->AddLine(ImVec2(barLeft, y), ImVec2(barLeft + barMaxW, y),
                        IM_COL32(255, 255, 255, 200), 1.5f);
            char vwapLabel[32];
            std::snprintf(vwapLabel, sizeof(vwapLabel), "VWAP %.4f", vwap);
            dl->AddText(ImVec2(barLeft + barMaxW - 70, y - 14),
                        IM_COL32(255, 255, 255, 230), vwapLabel);
        }
    }

    // Reserve the draw area.
    ImGui::Dummy(ImVec2(totalW, totalH));

    // Stats footer.
    double vaHighPrice = midByBucket[vaHighIdx];
    double vaLowPrice  = midByBucket[vaLowIdx];
    double pocPrice    = midByBucket[pocIdx];
    ImGui::Text("POC %.4f (vol %.2f)   VAH %.4f   VAL %.4f   Total vol %.2f",
                pocPrice, pocVol, vaHighPrice, vaLowPrice, totalVol);
    ImGui::Text("Value Area: %.0f%% of total volume (%zu of %zu buckets)",
                totalVol > 0 ? (vaVol / totalVol * 100.0) : 0.0,
                vaBuckets.size(), volBuckets.size());

    ImGui::End();
}

}  // namespace btquant::ui
