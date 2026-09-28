#include "heatmap_widget.hpp"

#include <imgui.h>

namespace btquant::ui {

HeatmapWidget::~HeatmapWidget() = default;

bool HeatmapWidget::initialize(renderer::HeatmapCompute& compute) {
    m_compute = &compute;
    return true;
}

void HeatmapWidget::push(float price_normalized, float time_normalized,
                          float volume, uint32_t side) {
    m_buffer.push_back({price_normalized, time_normalized, volume, side});
}

void HeatmapWidget::pushBatch(const std::vector<renderer::TradeInput>& trades) {
    m_buffer.insert(m_buffer.end(), trades.begin(), trades.end());
}

void HeatmapWidget::render() {
    if (!m_compute || !showHeatmap) return;

    ImGui::Begin("Heatmap (GPU Compute)", &showHeatmap);
    ImGui::Text("Trades in buffer: %zu (frame %llu)", m_buffer.size(),
                static_cast<unsigned long long>(m_frameCounter++));

    // The compute dispatch needs a Vulkan command buffer in recording state
    // AND a render pass (for the ImGui image barrier to be valid).
    // The mainLoop already records the graphics command buffer — we piggyback
    // by deferring the dispatch until mainLoop's cmd is available. For now
    // we just upload + draw; the actual vkCmdDispatch is triggered via a
    // callback registered with the VulkanContext. See main.cpp.
    m_compute->updateTrades(m_buffer.data(), static_cast<uint32_t>(m_buffer.size()));

    // Draw the heatmap (will be empty until the first dispatch lands).
    // 384x384 pixel display = 1.5x source resolution for visibility.
    ImGui::Image(m_compute->textureId(), ImVec2(384, 384));

    ImGui::Separator();
    ImGui::TextWrapped("GPU heatmap: 256x256 R8G8B8A8 storage image, "
                        "compute shader aggregates trade stream. "
                        "R = intensity, G = buy, B = sell.");

    // Clear buffer after upload — compute reads num_trades each frame.
    m_buffer.clear();
    ImGui::End();
}

}  // namespace btquant::ui
