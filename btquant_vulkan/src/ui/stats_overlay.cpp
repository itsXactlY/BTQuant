#include "stats_overlay.hpp"

#include <imgui.h>

namespace btquant::ui {

namespace {
constexpr double kEwmaAlpha = 0.10;  // weight for the most recent sample
} // namespace

void StatsOverlay::tick() {
    auto now = std::chrono::steady_clock::now();
    if (m_lastTick.time_since_epoch().count() == 0) {
        m_lastTick = now;
        return;  // first tick — no delta yet
    }
    double dt = std::chrono::duration<double, std::milli>(now - m_lastTick).count();
    m_lastTick = now;

    if (dt < 0.0) dt = 0.0;
    if (dt > 1000.0) dt = 1000.0;  // clamp huge spikes (stall detection)

    if (m_ewmaMs == 0.0) {
        m_ewmaMs = dt;
        m_minMs = dt;
        m_maxMs = dt;
    } else {
        m_ewmaMs = m_ewmaMs * (1.0 - kEwmaAlpha) + dt * kEwmaAlpha;
        if (dt < m_minMs) m_minMs = dt;
        if (dt > m_maxMs) m_maxMs = dt;
    }
}

void StatsOverlay::render(uint64_t tradeQueueDepth, uint64_t candleCount) {
    if (!m_enabled) return;

    // Pin to top-right, 8 px margin, no title bar, no input.
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 size = ImVec2(220, 90);
    ImVec2 pos = ImVec2(io.DisplaySize.x - size.x - 8.0f, 8.0f);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.65f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoInputs |
                             ImGuiWindowFlags_NoFocusOnAppearing;

    if (ImGui::Begin("##stats_overlay", nullptr, flags)) {
        double fps = avgFps();
        ImGui::Text("FPS       %.1f", fps);
        ImGui::Text("Frame     %.2f ms", m_ewmaMs);
        ImGui::Text("Min/Max   %.1f / %.1f ms", m_minMs, m_maxMs);
        ImGui::Text("Trades q  %lu", (unsigned long)tradeQueueDepth);
        ImGui::Text("Candles   %lu", (unsigned long)candleCount);
    }
    ImGui::End();
}

} // namespace btquant::ui
