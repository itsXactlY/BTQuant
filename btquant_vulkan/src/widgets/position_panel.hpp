#ifndef BTQUANT_POSITION_PANEL_HPP
#define BTQUANT_POSITION_PANEL_HPP

#include <functional>
#include <string>
#include <vector>

namespace btquant { class PositionBook; }

namespace btquant::ui {

// Position Panel — read-only view of the active PositionBook plus a flat
// fill-history log. Hotkey Ctrl+B toggles visibility. Subscribes to the
// PositionBook via setPositionBook() and pulls the latest snapshot each
// frame; markToMarket is driven by the renderer's per-frame call.
class PositionPanel {
public:
    struct FillRecord {
        std::string symbol;
        bool   isLong  = true;
        double qty     = 0.0;
        double price   = 0.0;
        double realizedDelta = 0.0;
        int    seq     = 0;       // monotonically increasing
    };

    void setPositionBook(::btquant::PositionBook* book) { m_book = book; }

    // Push a fill into the in-panel history (called when an OrderTicket
    // submission applies a fill). Bounded to kMaxHistory entries.
    void recordFill(const FillRecord& r);

    void render();

    bool isOpen() const        { return m_open; }
    void setOpen(bool v)       { m_open = v; }
    size_t historySize() const { return m_history.size(); }
    static constexpr size_t kMaxHistory = 64;

private:
    ::btquant::PositionBook* m_book = nullptr;
    std::vector<FillRecord>  m_history;
    bool m_open = false;
    int  m_seq  = 0;
};

} // namespace btquant::ui

#endif
