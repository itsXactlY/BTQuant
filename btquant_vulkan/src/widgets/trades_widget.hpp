#ifndef BTQUANT_TRADES_WIDGET_HPP
#define BTQUANT_TRADES_WIDGET_HPP

// MarketDataProcessor lives in the btquant:: namespace.
namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

class TradesWidget {
public:
    TradesWidget();
    ~TradesWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setFilter(double minSize);
    void reset();

private:
    double m_filterSize = 0;
    bool m_initialized = false;

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
