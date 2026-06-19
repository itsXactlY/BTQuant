#ifndef BTQUANT_TPO_WIDGET_HPP
#define BTQUANT_TPO_WIDGET_HPP

// MarketDataProcessor lives in the btquant:: namespace.
namespace btquant { class MarketDataProcessor; }

namespace btquant::ui {

class TPOWidget {
public:
    TPOWidget();
    ~TPOWidget();
    void render();

    void setMarketData(::btquant::MarketDataProcessor* data);

    void setSessionPeriod(int minutes);

private:
    int m_sessionPeriod = 30;
    bool m_initialized = false;

    class MarketDataProcessor* m_data = nullptr;
};

}  // namespace btquant::ui

#endif
