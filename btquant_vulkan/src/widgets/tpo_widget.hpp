#ifndef BTQUANT_TPO_WIDGET_HPP
#define BTQUANT_TPO_WIDGET_HPP

namespace btquant::ui {

class TPOWidget {
public:
    TPOWidget();
    ~TPOWidget();
    void render();
    void setSessionPeriod(int minutes);

private:
    int m_sessionPeriod = 30;
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
