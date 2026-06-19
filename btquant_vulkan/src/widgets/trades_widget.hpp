#ifndef BTQUANT_TRADES_WIDGET_HPP
#define BTQUANT_TRADES_WIDGET_HPP

namespace btquant::ui {

class TradesWidget {
public:
    TradesWidget();
    ~TradesWidget();
    void render();
    void setFilter(double minSize);
    void reset();

private:
    double m_filterSize = 0;
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
