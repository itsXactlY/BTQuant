#ifndef BTQUANT_WINDOW_MANAGER_HPP
#define BTQUANT_WINDOW_MANAGER_HPP

#include <string>
#include <memory>
#include <cstdint>

namespace btquant::ui {

class WindowManager {
public:
    WindowManager();
    ~WindowManager();

    void initialize();
    void shutdown();
    void beginFrame();
    void endFrame();

    void showOrderBookWindow();
    void showDOMWindow();
    void showTradesWindow();
    void showTPOWindow();
    void showMainMenu();

    bool showOrderBook = true;
    bool showDOM = true;
    bool showTrades = true;
    bool showTPO = true;

private:
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
