#ifndef BTQUANT_TRADES_WIDGET_HPP
#define BTQUANT_TRADES_WIDGET_HPP

#include <cstddef>
#include <string>
#include <vector>

// MarketDataProcessor lives in the btquant:: namespace.
namespace btquant { class MarketDataProcessor; }
namespace btquant::data { struct Trade; }

namespace btquant::ui {

class TradesWidget {
public:
    TradesWidget();
    ~TradesWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setFilter(double minSize);
    void reset();

    // ---- CSV export (time-and-sales tape) ----
    //
    // formatTradesCSV() is a pure helper — same input → same output,
    // no global state, no ImGui dependency. Returns an RFC-4180-style
    // CSV: comma-separated, header row first, fields with commas or
    // quotes are quoted with internal quotes doubled. Timestamp is
    // ISO-8601 (UTC) for portability across spreadsheets + pandas.
    static std::string formatTradesCSV(const std::vector<data::Trade>& trades);

    // Write the current tape (respecting m_filterSize) to `path`.
    // Returns true on success, false on write/open failure.
    bool exportCSV(const std::string& path) const;

    // Get the most recent N trades (live or synthetic fallback, same as
    // what the render path shows). Public so tests + menu code can
    // snapshot the tape without re-implementing the data source logic.
    std::vector<data::Trade> snapshotTrades(size_t maxCount = 50) const;

    // Menu modal state — when true, the next render frame draws the
    // "Export trades to CSV…" filename input popup. WindowManager owns
    // the actual menu rendering; the widget just holds the state.
    void  setExportModalOpen(bool v) { m_exportModalOpen = v; }
    bool  exportModalOpen() const    { return m_exportModalOpen; }
    void  setExportFilename(const std::string& s) { m_exportFilename = s; }
    const std::string& exportFilename() const      { return m_exportFilename; }

private:
    double m_filterSize = 0;
    bool m_initialized = false;
    bool m_exportModalOpen = false;
    std::string m_exportFilename = "trades.csv";

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
