#ifndef BTQUANT_CONNECTION_PANEL_HPP
#define BTQUANT_CONNECTION_PANEL_HPP

#include <cstdint>
#include <deque>
#include <string>

namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

// Connection status panel. Shows the data spine state:
//   - Source path
//   - Symbol
//   - Snapshot sequence number (monotonic)
//   - Tick rate (ticks/sec, EWMA)
//   - Total ticks received since start
//   - Parse errors (placeholder, future spine protocol parse-failures)
//   - Latency estimate (newest trade timestamp → now, in ms)
//   - Connection state (DISCONNECTED / SYNTHETIC / LIVE)
//
// Pulls from MarketDataProcessor each frame. Safe with a null processor.
class ConnectionPanel {
public:
    void setMarketData(::btquant::MarketDataProcessor* data) { m_data = data; }
    void render();

    // Test accessors.
    double tickRateEwma()  const { return m_tickRateEwma; }
    uint64_t lastSeqSeen() const { return m_lastSeq; }

private:
    enum class State { Disconnected, Synthetic, Live };
    const char* stateName(State s) const;

    ::btquant::MarketDataProcessor* m_data = nullptr;

    // Per-frame EWMA over (delta_ticks / delta_time). Updated each render().
    double   m_tickRateEwma  = 0.0;
    uint64_t m_lastTicksSeen = 0;
    uint64_t m_lastSeq       = 0;
    double   m_lastT         = 0.0;
};

} // namespace btquant::ui

#endif
