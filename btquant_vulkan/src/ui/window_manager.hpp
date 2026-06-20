#ifndef BTQUANT_WINDOW_MANAGER_HPP
#define BTQUANT_WINDOW_MANAGER_HPP

#include <string>
#include <memory>
#include <cstdint>

#include "stats_overlay.hpp"
#include "../util/settings.hpp"

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

    // Hotkey help overlay (toggled by `?` key). Shows all shortcuts in a table.
    void showHotkeyHelpWindow();
    bool showHotkeyHelp = false;

    // Theme — owned by UIContext but the WindowManager menu triggers changes.
    // We store the current theme here so we can detect changes and rebuild.
    long theme = 0;  // 0 = Dark, 1 = Light

    bool showSettings = false;

    // Alerts panel toggle.
    bool showAlerts = true;

    // Watchlist toggle.
    bool showWatchlist = true;

    // Top-right FPS / frame-time overlay (toggled by hotkey Shift+F1 or
    // menu item, persisted in Settings).
    bool showStatsOverlay = true;
    StatsOverlay& statsOverlay() { return m_statsOverlay; }

    // Last heatmap density we successfully applied (tracked here so the
    // render loop can detect changes from the slider).
    long lastAppliedHeatmapDensity = -1;

    // Set to true whenever a toggle/state mutation happens (View menu item,
    // F-key hotkey, slider change). Main loop saves state.ini periodically
    // while dirty and clears the flag. Cheap to test — just a bool compare.
    bool settingsDirty() const { return m_settingsDirty; }
    void markSettingsDirty() { m_settingsDirty = true; }
    void clearSettingsDirty() { m_settingsDirty = false; }

    // Render-target fps limit (0 = uncapped). Read by main loop via
    // glfwSwapInterval; 60 → swap interval 1, anything > 0 → interval 1,
    // 0 → interval 0.
    long fpsLimit = 60;

    // Heatmap GPU texture side length (cells per side). 64–512; main loop
    // passes this to HeatmapWidget.resize() if it changes.
    long heatmapDensity = 128;

    // Process global hotkeys via GLFW direct key access. Call once per frame
    // AFTER glfwPollEvents but BEFORE imgui::NewFrame so user input reaches
    // widgets when a text field has focus. F2..F12 toggle widgets, Ctrl+L
    // resets the docking layout.
    void processHotkeys(void* glfwWindow);

    // Record one frame for the stats overlay EWMA. Call once per render loop
    // iteration.
    void tickStatsOverlay() { m_statsOverlay.tick(); }

    // Render the FPS / frame-time / queue-depth overlay. Trade queue depth
    // and candle count are passed in from main.
    void renderStatsOverlay(uint64_t tradeQueueDepth, uint64_t candleCount);

    void showOrderBookWindow();
    void showOrderBookDepthWindow();
    void showFootprintWindow();
    void showVPVRWindow();
    void showMultiVWAPWindow();
    void showRiskPanelWindow();
    void showDOMWindow();
    void showTradesWindow();
    void showTPOWindow();
    void showAlertsWindow();
    void showWatchlistWindow();
    void updateWatchlist(const std::string& sym, double p, double s, bool b, uint64_t ts);
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
    void applyPreset(const ::btquant::util::Settings& s);

    class OrderBookWidget* m_orderBookWidget = nullptr;
    class OrderBookDepthWidget* m_orderBookDepthWidget = nullptr;
    class FootprintWidget* m_footprintWidget = nullptr;
    class VPVRWidget* m_vpvrWidget = nullptr;
    class MultiVWAPWidget* m_multiVwapWidget = nullptr;
    class RiskPanel* m_riskPanel = nullptr;
    class DOMWidget* m_domWidget = nullptr;
    class TradesWidget* m_tradesWidget = nullptr;
    class TPOWidget* m_tpoWidget = nullptr;
    class AlertsPanel* m_alertsPanel = nullptr;
    class WatchlistWidget* m_watchlistWidget = nullptr;
    StatsOverlay m_statsOverlay;
    bool m_initialized = false;
    bool m_layoutApplied = false;
    bool m_layoutResetRequested = false;
    bool m_settingsDirty = false;
};

} // namespace btquant::ui

#endif
