#ifndef BTQUANT_ALERTS_PANEL_HPP
#define BTQUANT_ALERTS_PANEL_HPP

#include <cstdint>
#include <deque>
#include <string>

#include "../data/market_data_processor.hpp"

namespace btquant::ui {

// Compact alerts panel — detects price moves and volume spikes from the live
// snapshot stream and surfaces them as a rolling list. Optionally plays a
// terminal bell on each new alert. This is the "alerts_panel" module from
// BTQ_Render_Engine, trimmed to the essentials.
class AlertsPanel {
public:
    AlertsPanel() = default;

    void setMarketData(::btquant::MarketDataProcessor* data) { m_data = data; }

    // Render the panel (call inside an ImGui window or under the dockspace).
    void render();

    // Configuration. Persisted in Settings.
    double priceMovePctThreshold = 0.10;   // % move in 1 second
    double volumeSpikeMultiplier = 3.0;    // x avg volume
    bool   soundEnabled          = false;  // terminal bell on new alert
    bool   showInStatusBar       = true;

    // Public for tests.
    struct Alert {
        std::string message;
        double      timestampSec;  // seconds since render-loop start
        int         severity;     // 0 = info, 1 = warn, 2 = critical
    };
    const std::deque<Alert>& alerts() const { return m_alerts; }
    void clearAlerts() { m_alerts.clear(); }

private:
    void evaluateAndPush(const class MarketDataProcessor::Snapshot& snap, double nowSec);
    static std::string formatPrice(double p);
    static std::string formatVolume(double v);

    ::btquant::MarketDataProcessor* m_data = nullptr;
    std::deque<Alert> m_alerts;
    static constexpr size_t kMaxAlerts = 256;

    // Per-frame state for spike detection.
    double m_lastPrice        = 0.0;
    double m_lastPriceAt      = 0.0;
    double m_volumeEma        = 0.0;       // EWMA of |trade.size|
    uint64_t m_alertsEmitted  = 0;
    double  m_startTimeSec    = 0.0;
};

} // namespace btquant::ui

#endif
