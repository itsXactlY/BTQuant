#ifndef BTQUANT_MINI_PRICE_CHART_HPP
#define BTQUANT_MINI_PRICE_CHART_HPP

#include <cstddef>
#include <vector>

namespace btquant { class MarketDataProcessor; }
namespace btquant::data { struct Candle; }

namespace btquant::ui {

// Mini candlestick chart. Pulls recent finalized candles from the
// MarketDataProcessor and renders them via ImPlot. Includes volume bars
// below the price plot for quick delta/flow inspection.
//
// Hotkey: Ctrl+M toggles visibility.
class MiniPriceChart {
public:
    void setMarketData(::btquant::MarketDataProcessor* data) { m_data = data; }

    void render();

    bool isOpen() const  { return m_open; }
    void setOpen(bool v) { m_open = v; }

    // How many recent candles to display. Persisted by the caller.
    void  setHistoryN(size_t n) { m_historyN = n; }
    size_t historyN() const     { return m_historyN; }

    // ---- Pure OHLC validation (test surface) ----
    //
    // A well-formed candle satisfies:
    //   open  > 0   (positive price)
    //   high  >= max(open, close)
    //   low   <= min(open, close)
    //   low   > 0   (positive price)
    //   volume >= 0
    static bool validateCandle(const ::btquant::data::Candle& c);

    // Validate an entire series — returns the index of the first bad
    // candle, or -1 if all are valid.
    static int validateSeries(const std::vector<::btquant::data::Candle>& v);

private:
    ::btquant::MarketDataProcessor* m_data = nullptr;
    bool  m_open     = false;
    size_t m_historyN = 60;   // last 60 finalized candles
};

} // namespace btquant::ui

#endif
