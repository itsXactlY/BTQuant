#ifndef BTQUANT_DOM_WIDGET_HPP
#define BTQUANT_DOM_WIDGET_HPP

// MarketDataProcessor lives in the btquant:: namespace.
namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

class DOMWidget {
public:
    DOMWidget();
    ~DOMWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setPriceGrouping(double value);
    void setMaxLevels(int levels);
    void setAlignment(const char* mode);
    void setHeatmapMode(bool on);
    bool heatmapMode() const { return m_heatmapMode; }

    // Heatmap row height in pixels — derived from Settings.heatmapDensity
    // (default 128 → ~6px rows; 512 → ~1.5px rows). Tests use the setter
    // to override the live setting without driving the full Settings chain.
    void setCellHeightPx(float px) { m_cellHeightPx = px; }
    float cellHeightPx() const { return m_cellHeightPx; }

private:
    double m_priceGrouping = 0.01;
    int m_maxLevels = 20;
    const char* m_alignment = "Center";
    bool m_initialized = false;
    bool m_heatmapMode = false;
    float m_cellHeightPx = 6.0f;  // overridable via setCellHeightPx()

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
