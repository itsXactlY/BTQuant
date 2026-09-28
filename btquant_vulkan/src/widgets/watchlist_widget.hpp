#ifndef BTQUANT_WATCHLIST_WIDGET_HPP
#define BTQUANT_WATCHLIST_WIDGET_HPP

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace btquant::ui {

// Lightweight multi-symbol ticker. Symbol string -> rolling micro-sparkline
// of last prices + aggregated buy/sell volume + last-trade timestamp.
// Data is pushed from the main loop via update(symbol, price, size, isBuy).
//
// Click-to-switch (Sprint #63): the Symbol column in each row is
// wrapped in a Selectable so clicking it fires m_select(symbol).
// WindowManager wires the callback to whatever actually changes the
// active symbol (typically the same closure that the SymbolPicker
// uses). Selecting from the watchlist is the fastest way for a
// trader to switch context during a busy session — no popup, no
// typing, just click the row.
class WatchlistWidget {
public:
    static constexpr size_t kMaxSparkPoints = 64;

    struct Row {
        std::string symbol;
        std::deque<double> spark;
        double lastPrice  = 0.0;
        double prevPrice  = 0.0;
        double totalVol   = 0.0;
        double buyVol     = 0.0;
        double sellVol    = 0.0;
        uint64_t lastTs   = 0;
        uint64_t tickCount = 0;
    };

    void render();

    void clear() { m_rows.clear(); }
    bool empty() const { return m_rows.empty(); }

    // External data feed.
    void update(const std::string& symbol, double price, double size,
                bool isBuy, uint64_t timestamp);

    // Persistence-friendly.
    const std::vector<std::string>& symbols() const { return m_symbols; }
    void setSymbols(const std::vector<std::string>& syms);

    // Selection callback (Sprint #63). Fired with the clicked
    // symbol. Optional — when null, the Symbol column renders
    // non-interactive (Selectable without an on-click).
    using SelectFn = std::function<void(const std::string& symbol)>;
    void setSelectFn(SelectFn fn) { m_select = std::move(fn); }

    // Test accessors.
    Row* row(const std::string& symbol);
    size_t rowCount() const { return m_rows.size(); }

private:
    void drawSparkline(const Row& r, float width, float height) const;

    std::vector<std::string> m_symbols;
    std::unordered_map<std::string, Row> m_rows;
    SelectFn m_select;
};

}  // namespace btquant::ui

#endif
