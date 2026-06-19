#ifndef BTQUANT_HEATMAP_WIDGET_HPP
#define BTQUANT_HEATMAP_WIDGET_HPP

#include "../renderer/heatmap_compute.hpp"
#include <cstdint>
#include <vector>

namespace btquant::ui {

// Displays the GPU-computed heatmap as an ImGui::Image.
// Owns a HeatmapCompute instance and a rolling buffer of recent trades
// (push() appends, render() flushes to GPU + draws).
class HeatmapWidget {
public:
    HeatmapWidget() = default;
    ~HeatmapWidget();

    // One-time setup. queueFamily is typically the same as graphics.
    [[nodiscard]] bool initialize(renderer::HeatmapCompute& compute);

    // Append a normalized (price, time, volume, side) trade to the buffer.
    // Called by MarketDataProcessor each tick.
    void push(float price_normalized, float time_normalized,
              float volume, uint32_t side);

    // Push a batch at once (efficient path).
    void pushBatch(const std::vector<renderer::TradeInput>& trades);

    // Flush buffer to GPU, dispatch compute, draw heatmap as ImGui::Image.
    // Call this once per frame inside an active ImGui window.
    void render();

    // Optional: limit visible range.
    void setPriceRange(float low, float high) { m_priceLow = low; m_priceHigh = high; }

    bool showHeatmap = true;

private:
    renderer::HeatmapCompute* m_compute = nullptr;
    std::vector<renderer::TradeInput> m_buffer;     // rolling trade buffer
    float m_priceLow = 0.0f;
    float m_priceHigh = 1.0f;
    uint64_t m_frameCounter = 0;
};

}  // namespace btquant::ui

#endif
