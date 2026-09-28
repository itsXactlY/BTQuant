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

    enum class RenderMode { Line = 0, Candle = 1 };
    void      setRenderMode(RenderMode m) { m_renderMode = m; }
    RenderMode renderMode() const        { return m_renderMode; }

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

    // ---- Pure pixel geometry (test surface) ----
    //
    // Map a price value into a pixel Y coordinate inside the chart
    // canvas. Higher price → smaller Y (screen coords). Pure: same
    // inputs always produce the same output, no global state.
    static float priceToPixelY(double price, double yMin, double yMax,
                               float canvasY, float canvasH);

    // Map a candle index into a pixel X coordinate (centered on the
    // candle body) given the canvas width, candle count, and a body
    // width fraction (0..1) of the slot. Pure.
    static float indexToPixelX(int idx, int count, float canvasX,
                               float canvasW, float bodyFrac);

private:
    ::btquant::MarketDataProcessor* m_data = nullptr;
    bool    m_open       = false;
    size_t  m_historyN   = 60;     // last 60 finalized candles
    RenderMode m_renderMode = RenderMode::Candle;  // default = real candles
};

} // namespace btquant::ui

#endif
