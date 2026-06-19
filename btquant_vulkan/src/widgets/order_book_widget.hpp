#ifndef BTQUANT_ORDER_BOOK_WIDGET_HPP
#define BTQUANT_ORDER_BOOK_WIDGET_HPP

#include <string>

namespace btquant::ui {

class OrderBookWidget {
public:
    OrderBookWidget();
    ~OrderBookWidget();

    void render();

    void setPriceGrouping(double value);
    void setMaxLevels(int levels);
    void setShowUSD(bool usd);

private:
    double m_priceGrouping = 0.01;
    int m_maxLevels = 25;
    bool m_showUSD = true;
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
