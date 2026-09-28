#ifndef BTQUANT_RECENT_FILLS_PANEL_HPP
#define BTQUANT_RECENT_FILLS_PANEL_HPP

#include <cstddef>
#include <deque>
#include <string>

namespace btquant {
struct JournalFill;
}

namespace btquant::ui {

// Recent Fills — scrollable table of the trader's last N fills, newest
// first. Pushed by WindowManager on every successful fill (live and
// kill-flatten). Distinct from TradesWidget, which mirrors the live
// market tape — this is the trader's OWN history, in execution order,
// with the strategy tag attached. Together they form the full
// attribution loop:
//
//   OrderTicket → fill() → TradeJournal.append →
//                                ↓
//                          RecentFillsPanel (visual)
//                          RiskPanel per-symbol P&L (aggregate)
//
// Ring buffer capped at kMaxFills (default 50) so memory stays bounded
// across a long session. Older fills are evicted FIFO.
class RecentFillsPanel {
public:
    static constexpr std::size_t kMaxFills = 50;

    RecentFillsPanel();
    ~RecentFillsPanel();

    void render();

    // Push a fill. Called from the OrderTicket submit callback AND
    // the kill-switch flatten path. Copies the JournalFill into the
    // ring buffer; if at capacity, oldest is evicted.
    void addFill(const ::btquant::JournalFill& jf);

    // Clear the ring buffer. Used by "Clear history" button + tests.
    void clear();

    // Test accessors. Returns a copy so tests don't need to know
    // about the deque internals.
    std::size_t size() const { return m_fills.size(); }
    bool empty() const { return m_fills.empty(); }

    bool isOpen() const    { return m_open; }
    void setOpen(bool v)   { m_open = v; }

private:
    std::deque<::btquant::JournalFill> m_fills;  // newest at front
    bool m_open = false;
};

}  // namespace btquant::ui

#endif
