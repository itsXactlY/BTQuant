#ifndef BTQUANT_WINDOW_MANAGER_HPP
#define BTQUANT_WINDOW_MANAGER_HPP

#include <string>
#include <memory>
#include <cstdint>
#include <optional>

#include "stats_overlay.hpp"
#include "../util/settings.hpp"

// MarketDataProcessor is declared in btquant:: namespace (not btquant::ui).
// Forward-declare globally so the type is visible inside namespace btquant::ui.
#include "../util/hotkey_config.hpp"
#include "../util/layout_io.hpp"
#include "../widgets/theme_editor.hpp"

namespace btquant { class MarketDataProcessor; }
namespace btquant { class PositionBook; }
namespace btquant { class RiskGuard; }
namespace btquant { class TradeJournal; }
namespace btquant::util { class HotkeyMap; }
namespace btquant::widgets { class HotkeyEditor; }

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

    // Log panel toggle.
    bool showLog = true;

    // Connection panel toggle.
    bool showConnection = true;

    // Profile Manager toggle.
    bool showProfileManager = false;

    // Symbol Picker toggle (modal — flips showSymbolPickerOpen).
    bool showSymbolPickerOpen = false;

    // Theme Editor toggle (modal).
    bool showThemeEditorOpen = false;

    // Position Calculator toggle.
    bool showPositionCalculator = true;

    // Order Ticket toggle (Ctrl+Enter).
    bool showOrderTicket = false;

    // Position Panel toggle (Ctrl+B).
    bool showPositionPanel = false;

    // Risk Dashboard toggle (Ctrl+R).
    bool showRiskLimits = false;

    // Mini Price Chart toggle (Ctrl+M).
    bool showMiniPriceChart = false;

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
    void showLogWindow();
    void showConnectionWindow();
    void showProfileManagerWindow();
    void showSymbolPickerWindow();
    void showThemeEditorWindow();
    void applyPersistedTheme();
    void applyPersistedRiskConfig();
    // Layout profile save/load — writes a LayoutSnapshot (current
    // Settings + dock text) to ~/.config/btquant_vulkan/profiles/<name>.btqlayout
    // and reloads the same on demand. WindowManager's loadLayout(name)
    // applies show* booleans, theme, and risk config; dock text is
    // stored but the actual ImGui dock restore is a follow-on (it needs
    // ImGui::DockBuilderLoadNodes which needs a live dockspace).
    bool saveLayoutAs(const std::string& name);
    bool loadLayout(const std::string& name);
    // Switch to the Nth .btqlayout file from LayoutIO::list() (sorted
    // alphabetically). Out-of-range index = no-op (returns false).
    // Used by Ctrl+1..Ctrl+9 hotkeys.
    bool loadLayoutByIndex(size_t index);
    // Apply an already-loaded LayoutSnapshot in-place. Public so tests
    // can drive the apply logic without touching the file layer.
    void applyLayoutSnapshot(const ::btquant::util::LayoutSnapshot& snap);
    bool saveCurrentTheme();
    // Reset the live theme to ImGui's default dark style and persist it
    // to the theme file so the next launch picks up the reset state.
    // Uses the snapshot captured at construction (m_defaultStyleSnap) so
    // the result is deterministic regardless of how many times the user
    // has mutated the theme since launch.
    void resetThemeToDefault();
    void showPositionCalculatorWindow();
    void showOrderTicketWindow();
    void showPositionPanelWindow();
    void showRiskLimitsWindow();
    void showMiniPriceChartWindow();
    void showHotkeyEditorWindow();
    // Run the side-effect for a triggered action — single source of
    // truth for non-toggle hotkeys (toggles go through the kHotkeys
    // table member pointers).
    void dispatchAction(::btquant::util::HotkeyAction a);
    util::Settings captureCurrentSettings() const;
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
    // Pending dock layout text captured from a LayoutSnapshot. When
    // non-empty, applyInitialDockLayoutIfNeeded feeds it to
    // ImGui::DockBuilderLoadNodes on the next dock-reset so the user
    // gets the saved split layout restored (not just widget visibility).
    // Public so tests can assert plumbing without needing a live ImGui
    // context.
    std::string pendingDockLayout;

private:
    // Snapshot of the live ImGui style at the time WindowManager is
    // constructed. Used by resetThemeToDefault() to restore ImGui's
    // original look without depending on a globally-cached default.
    // Nullopt if the capture failed (e.g. tests run before ImGui is
    // initialised). Public so tests can assert the plumbing without
    // needing a live ImGui context.
public:
    std::optional<ThemeEditor::Snapshot> m_defaultStyleSnap;
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
    class LogPanel* m_logPanel = nullptr;
    class ConnectionPanel* m_connectionPanel = nullptr;
    class ProfileManager*  m_profileManager  = nullptr;
    class SymbolPicker*    m_symbolPicker    = nullptr;
    class ThemeEditor*     m_themeEditor     = nullptr;
    class PositionCalculator* m_positionCalculator = nullptr;
    class OrderTicket*       m_orderTicket        = nullptr;
    class PositionPanel*     m_positionPanel      = nullptr;
    class RiskLimitsPanel*   m_riskLimitsPanel    = nullptr;
    class MiniPriceChart*    m_miniPriceChart     = nullptr;
    StatsOverlay m_statsOverlay;
    // Cached MarketDataProcessor pointer — used by the SymbolPicker callback
    // to actually swap the active symbol on selection. Without this, the
    // picker would only log and the UI would still show the old symbol.
    ::btquant::MarketDataProcessor* m_marketData = nullptr;
    // Cached PositionBook — owns the open position. OrderTicket submits
    // apply fills here; main loop drives markToMarket each frame.
    ::btquant::PositionBook*       m_positionBook   = nullptr;
    // RiskGuard — pre-trade checks + session P&L + kill switch.
    ::btquant::RiskGuard*          m_riskGuard      = nullptr;
    // TradeJournal — append-only JSONL file at ~/.config/btquant_vulkan/journal.jsonl
    ::btquant::TradeJournal*       m_tradeJournal   = nullptr;
    // HotkeyMap — user-mappable hotkeys loaded from hotkeys.ini at startup.
    ::btquant::util::HotkeyMap*    m_hotkeyMap      = nullptr;
    // HotkeyEditor — Ctrl+H panel that lets the user remap bindings at runtime.
    ::btquant::widgets::HotkeyEditor* m_hotkeyEditor = nullptr;
    // Hotkey file path — saved to on each remap so user changes survive restart.
    std::string                    m_hotkeyPath;
    // Layout profile save/load UI state.
    bool   m_layoutSaveOpen  = false;
    bool   m_layoutLoadOpen  = false;
    char   m_layoutNameBuf[64] = "Custom";
    char   m_layoutExportBuf[256] = "/tmp/export.btqlayout";
    char   m_layoutImportBuf[256] = "/tmp/import.btqlayout";
    bool m_initialized = false;
    bool m_layoutApplied = false;
    bool m_layoutResetRequested = false;
    bool m_layoutExportOpen = false;
    bool m_layoutImportOpen = false;
    bool m_hotkeyEditorOpen = false;
    bool m_settingsDirty = false;
};
class PositionBook;
} // namespace btquant
namespace btquant::ui { class ThemeEditor; class RiskGuard; }

#endif
