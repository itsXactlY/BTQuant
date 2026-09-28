#ifndef BTQUANT_CONNECTION_PANEL_HPP
#define BTQUANT_CONNECTION_PANEL_HPP

#include <cstdint>
#include <deque>
#include <string>

struct ImVec4;
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
    // Public state enum — the menu bar / status strip uses the same
    // states, so making it part of the API lets the badge in the
    // WindowManager menu bar match the panel's source-of-truth.
    enum class State { Disconnected, Synthetic, Live };

    void setMarketData(::btquant::MarketDataProcessor* data) { m_data = data; }
    void render();

    // Test accessors.
    double tickRateEwma()  const { return m_tickRateEwma; }
    uint64_t lastSeqSeen() const { return m_lastSeq; }

    // ---- Static helpers (public) ----
    // Compute the connection state from a market-data processor. Safe
    // with a null pointer — returns Disconnected in that case. The
    // rules are duplicated in render() to keep the static helper
    // self-contained (no instance state required for the simple case).
    static State computeState(const ::btquant::MarketDataProcessor* data);

    // Human-readable state label — used by both the panel and the
    // menu-bar status badge.
    static const char* stateName(State s);

    // RGBA colour for the state badge — used by both the panel and
    // the menu-bar status badge so the visual language matches.
    static ImVec4 stateColor(State s);

    // One-call helper: compute state from `data`, draw a coloured
    // dot + label. Returns the state that was drawn (so callers
    // can build tooltips or follow-on actions). Safe with a null
    // processor — renders the Disconnected badge.
    static State renderStateBadge(
        const ::btquant::MarketDataProcessor* data);

private:
    ::btquant::MarketDataProcessor* m_data = nullptr;

    // Per-frame EWMA over (delta_ticks / delta_time). Updated each render().
    double   m_tickRateEwma  = 0.0;
    uint64_t m_lastTicksSeen = 0;
    uint64_t m_lastSeq       = 0;
    double   m_lastT         = 0.0;
};

} // namespace btquant::ui

#endif
