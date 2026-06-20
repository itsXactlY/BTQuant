#ifndef BTQUANT_STATS_OVERLAY_HPP
#define BTQUANT_STATS_OVERLAY_HPP

#include <chrono>
#include <cstdint>

namespace btquant::ui {

// Top-right floating overlay with FPS, frame time, queue depth. Backed by a
// rolling 60-frame EWMA so the displayed value doesn't jitter.
class StatsOverlay {
public:
    StatsOverlay() = default;

    // Records one frame; call once per render loop iteration BEFORE render().
    void tick();

    // Render the overlay if enabled. Pinned to top-right corner with
    // semi-transparent background; no window title bar, no interaction.
    void render(uint64_t tradeQueueDepth = 0, uint64_t candleCount = 0);

    void setEnabled(bool e) { m_enabled = e; }
    bool enabled() const { return m_enabled; }

    // Public so tests can inspect the rolling average.
    double avgFrameTimeMs() const { return m_ewmaMs; }
    double avgFps() const { return m_ewmaMs > 0.0 ? 1000.0 / m_ewmaMs : 0.0; }

private:
    bool m_enabled = true;
    std::chrono::steady_clock::time_point m_lastTick{};
    double m_ewmaMs = 0.0;
    double m_minMs = 0.0;
    double m_maxMs = 0.0;
};

} // namespace btquant::ui

#endif
