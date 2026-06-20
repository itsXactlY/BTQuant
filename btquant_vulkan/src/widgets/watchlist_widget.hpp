#ifndef BTQUANT_WATCHLIST_WIDGET_HPP
#define BTQUANT_WATCHLIST_WIDGET_HPP

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace btquant::ui {

// Lightweight multi-symbol ticker. Symbol string -> rolling micro-sparkline
// of last prices + aggregated buy/sell volume + last-trade timestamp.
// Data is pushed from the main loop via update(symbol, price, size, isBuy).
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

    // Test accessors.
    Row* row(const std::string& symbol);
    size_t rowCount() const { return m_rows.size(); }

private:
    void drawSparkline(const Row& r, float width, float height) const;

    std::vector<std::string> m_symbols;
    std::unordered_map<std::string, Row> m_rows;
};

} // namespace btquant::ui

#endif
