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

private:
    double m_priceGrouping = 0.01;
    int m_maxLevels = 20;
    const char* m_alignment = "Center";
    bool m_initialized = false;

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
