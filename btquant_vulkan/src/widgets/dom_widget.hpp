#ifndef BTQUANT_DOM_WIDGET_HPP
#define BTQUANT_DOM_WIDGET_HPP

namespace btquant::ui {

class DOMWidget {
public:
    DOMWidget();
    ~DOMWidget();

    void render();

    void setPriceGrouping(double value);
    void setMaxLevels(int levels);
    void setAlignment(const char* mode);

private:
    double m_priceGrouping = 0.01;
    int m_maxLevels = 25;
    const char* m_alignment = "Center";
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
