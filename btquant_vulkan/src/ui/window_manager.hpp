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

    // Call once on the first frame after DockSpaceOverViewport() to build the
    // default 5-region layout (OrderBook | DOM | Trades | TPO/Risk/VPVR/VWAP
    // stacked). Idempotent — no-op on subsequent frames.
    void applyInitialDockLayoutIfNeeded();

    // Tear down the saved docking tree and rebuild the default layout on next
    // frame. Triggered by "View → Reset Layout".
    void requestDockLayoutReset();

    // Toggle the Settings window. Settings state lives in the application;
    // WindowManager just renders the window and exposes accessors.
    void showSettingsWindow();
    void setShowSettings(bool v) { showSettings = v; }

    bool showSettings = false;

    // Render-target fps limit (0 = uncapped). Read by main loop via
    // glfwSwapInterval; 60 → swap interval 1, anything > 0 → interval 1,
    // 0 → interval 0.
    long fpsLimit = 60;

    // Heatmap GPU texture side length (cells per side). 64–512; main loop
    // passes this to HeatmapWidget.resize() if it changes.
    long heatmapDensity = 128;

    void showOrderBookWindow();
    void showOrderBookDepthWindow();
    void showFootprintWindow();
    void showVPVRWindow();
    void showMultiVWAPWindow();
    void showRiskPanelWindow();
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
    bool showRiskPanel = true;
    bool showDOM = true;
    bool showTrades = true;
    bool showTPO = true;

private:
    void buildDockLayout();

    class OrderBookWidget* m_orderBookWidget = nullptr;
    class OrderBookDepthWidget* m_orderBookDepthWidget = nullptr;
    class FootprintWidget* m_footprintWidget = nullptr;
    class VPVRWidget* m_vpvrWidget = nullptr;
    class MultiVWAPWidget* m_multiVwapWidget = nullptr;
    class RiskPanel* m_riskPanel = nullptr;
    class DOMWidget* m_domWidget = nullptr;
    class TradesWidget* m_tradesWidget = nullptr;
    class TPOWidget* m_tpoWidget = nullptr;
    bool m_initialized = false;
    bool m_layoutApplied = false;
    bool m_layoutResetRequested = false;
};

} // namespace btquant::ui

#endif
