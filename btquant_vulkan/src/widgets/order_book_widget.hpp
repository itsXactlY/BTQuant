#ifndef BTQUANT_ORDER_BOOK_WIDGET_HPP
#define BTQUANT_ORDER_BOOK_WIDGET_HPP

// MarketDataProcessor lives in the btquant:: namespace (not btquant::ui).
// Forward-declare it at global scope so the member pointer below resolves
// to ::btquant::MarketDataProcessor and not btquant::ui::MarketDataProcessor.
namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

class OrderBookWidget {
public:
    OrderBookWidget();
    ~OrderBookWidget();
    void render();

    // Bind to a live MarketDataProcessor. Falls back to internal synthetic
    // mock data when null (preserves the original offline demo behaviour).
    void setMarketData(::btquant::MarketDataProcessor* data);

    void setPriceGrouping(double value);
    void setMaxLevels(int levels);
    void setShowUSD(bool usd);

private:
    double m_priceGrouping = 0.01;
    int m_maxLevels = 20;
    bool m_showUSD = false;
    bool m_initialized = false;

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
