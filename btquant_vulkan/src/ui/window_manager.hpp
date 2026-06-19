#ifndef BTQUANT_WINDOW_MANAGER_HPP
#define BTQUANT_WINDOW_MANAGER_HPP

#include <string>
#include <memory>
#include <cstdint>

// MarketDataProcessor is declared in btquant:: namespace (not btquant::ui).
// Forward-declare globally so the type is visible inside namespace btquant::ui.
namespace btquant { class MarketDataProcessor; }

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
    void showOrderBookDepthWindow();
    void showFootprintWindow();
    void showVPVRWindow();
    void showMultiVWAPWindow();
    void showDOMWindow();
    void showTradesWindow();
    void showTPOWindow();
    void showMainMenu();

    // Bind the live data source to all 4 trading widgets. Passing nullptr
    // disconnects them (widgets fall back to internal synthetic mock data).
    void setMarketData(::btquant::MarketDataProcessor* data);

    bool showOrderBook = true;
    bool showOrderBookDepth = true;
    bool showFootprint = true;
    bool showVPVR = true;
    bool showMultiVWAP = true;
    bool showDOM = true;
    bool showTrades = true;
    bool showTPO = true;

private:
    class OrderBookWidget* m_orderBookWidget = nullptr;
    class OrderBookDepthWidget* m_orderBookDepthWidget = nullptr;
    class FootprintWidget* m_footprintWidget = nullptr;
    class VPVRWidget* m_vpvrWidget = nullptr;
    class MultiVWAPWidget* m_multiVwapWidget = nullptr;
    class DOMWidget* m_domWidget = nullptr;
    class TradesWidget* m_tradesWidget = nullptr;
    class TPOWidget* m_tpoWidget = nullptr;
    bool m_initialized = false;
};

} // namespace btquant::ui

#endif
