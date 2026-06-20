#include "window_manager.hpp"

#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder*
#include <algorithm>          // std::find / std::rotate (symbol-picker callback)

#include "../widgets/order_book_widget.hpp"
#include "../widgets/order_book_depth_widget.hpp"
#include "../widgets/footprint_widget.hpp"
#include "../widgets/vpvr_widget.hpp"
#include "../widgets/multi_vwap_widget.hpp"
#include "../widgets/risk_panel.hpp"
#include "../widgets/dom_widget.hpp"
#include "../widgets/trades_widget.hpp"
#include "../widgets/tpo_widget.hpp"

#include "../data/market_data_processor.hpp"
#include "../util/settings.hpp"
#include "stats_overlay.hpp"
#include "../widgets/alerts_panel.hpp"
#include "../widgets/watchlist_widget.hpp"
#include "../widgets/log_panel.hpp"
#include "../widgets/connection_panel.hpp"
#include "../widgets/profile_manager.hpp"
#include "../widgets/symbol_picker.hpp"
#include "../widgets/theme_editor.hpp"
#include "../util/theme_io.hpp"
#include "../widgets/position_calculator.hpp"
#include "../widgets/order_ticket.hpp"
#include "../widgets/position_panel.hpp"
#include "../widgets/risk_limits_panel.hpp"
#include "../widgets/mini_price_chart.hpp"
#include "../data/position_book.hpp"
#include "../data/risk_guard.hpp"
#include "../data/trade_journal.hpp"
#include "../data/market_data.hpp"
#include "../util/hotkey_config.hpp"
#include "../widgets/hotkey_editor.hpp"

using btquant::ui::LogPanel;

// GLFW direct key access for hotkeys. Used because ImGui's keyboard input goes
// through imgui_impl_glfw and we want hotkeys to work even when no widget has
// focus. We check io.WantCaptureKeyboard so the user can still type into text
// fields unimpeded.
#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

// Static slider bounds (file-local) for the Settings window. SliderScalar
// needs typed pointers; using static const values avoids allocating per-frame.
// Hotkey bindings live INSIDE namespace btquant::ui so member pointers to
// WindowManager resolve correctly via unqualified lookup.
namespace btquant::ui {

const long kZero = 0;
const long kFps240 = 240;
const long kHeatmapMin = 64;
const long kHeatmapMax = 512;

// Stable references for theme radio buttons. MenuItem(selected=ptr) wants a
// stable bool that the radio can point at; these never change.
bool kBoolTrue = true;
bool kBoolFalse = false;

#ifdef BTQUANT_USE_GLFW
// Toggle widget given its F-key hotkey (F2-F12) — looks up the action in
// the runtime HotkeyMap so user remappings take effect, then dispatches
// via member pointer.
struct HotkeySlot {
    ::btquant::util::HotkeyAction action;
    bool WindowManager::*flag;
};
constexpr HotkeySlot kHotkeys[] = {
    { ::btquant::util::HotkeyAction::ToggleOrderBook,      &WindowManager::showOrderBook      },
    { ::btquant::util::HotkeyAction::ToggleOrderBookDepth, &WindowManager::showOrderBookDepth },
    { ::btquant::util::HotkeyAction::ToggleDOM,            &WindowManager::showDOM            },
    { ::btquant::util::HotkeyAction::ToggleTrades,         &WindowManager::showTrades         },
    { ::btquant::util::HotkeyAction::ToggleTPO,            &WindowManager::showTPO            },
    { ::btquant::util::HotkeyAction::ToggleFootprint,      &WindowManager::showFootprint      },
    { ::btquant::util::HotkeyAction::ToggleVPVR,           &WindowManager::showVPVR           },
    { ::btquant::util::HotkeyAction::ToggleAlerts,         &WindowManager::showAlerts         },
    { ::btquant::util::HotkeyAction::ToggleMultiVWAP,      &WindowManager::showMultiVWAP      },
    { ::btquant::util::HotkeyAction::ToggleRiskPanel,      &WindowManager::showRiskPanel      },
    { ::btquant::util::HotkeyAction::ToggleSettings,       &WindowManager::showSettings       },
};
#endif // BTQUANT_USE_GLFW

} // namespace btquant::ui (constants + hotkey table)

namespace btquant::ui {

WindowManager::WindowManager() {
    // Compute config dir once — used by TradeJournal, HotkeyMap, and the
    // RiskLimitsPanel persistence callback below.
    const char* home = std::getenv("HOME");
    std::string configDir = std::string(home ? home : "/tmp") +
                            "/.config/btquant_vulkan/";

    m_orderBookWidget = new OrderBookWidget();
    m_orderBookDepthWidget = new OrderBookDepthWidget();
    m_footprintWidget = new FootprintWidget();
    m_vpvrWidget = new VPVRWidget();
    m_multiVwapWidget = new MultiVWAPWidget();
    m_riskPanel = new RiskPanel();
    m_domWidget = new DOMWidget();
    m_tradesWidget = new TradesWidget();
    m_tpoWidget = new TPOWidget();
    m_alertsPanel = new AlertsPanel();
    m_watchlistWidget = new WatchlistWidget();
    m_watchlistWidget->setSymbols({"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"});
    m_logPanel = &LogPanel::instance();
    m_connectionPanel = new ConnectionPanel();
    m_profileManager  = new ProfileManager();
    m_profileManager->setCaptureFn([this]() { return captureCurrentSettings(); });
    m_profileManager->setApplyFn([this](const std::string& name) {
        util::Settings loaded = util::Settings::load(util::Settings::profilePath(name));
        applyPreset(loaded);
        markSettingsDirty();
    });

    m_symbolPicker = new SymbolPicker();
    m_symbolPicker->setSelectFn([this](const std::string& sym) {
        BTQ_LOG_INFO("SymbolPicker: selected %s", sym.c_str());
        // Forward the selection to the MarketDataProcessor — setSymbol()
        // atomically swaps the active spine index and resets the aggregator
        // so the UI immediately starts feeding ticks from the new symbol.
        if (m_marketData) {
            m_marketData->setSymbol(sym);
        }
        // Promote the picked symbol to the top of the watchlist so the user
        // sees it highlighted without scrolling.
        if (m_watchlistWidget) {
            std::vector<std::string> cur = m_watchlistWidget->symbols();
            auto it = std::find(cur.begin(), cur.end(), sym);
            if (it != cur.end()) std::rotate(cur.begin(), it, it + 1);
            else                  cur.insert(cur.begin(), sym);
            m_watchlistWidget->setSymbols(cur);
        }
    });

    m_themeEditor = new ThemeEditor();
    m_positionCalculator = new PositionCalculator();
    m_orderTicket = new OrderTicket();
    m_positionBook  = new ::btquant::PositionBook();
    m_positionPanel = new PositionPanel();
    m_positionPanel->setPositionBook(m_positionBook);
    m_riskGuard     = new ::btquant::RiskGuard();
    m_riskLimitsPanel = new RiskLimitsPanel();
    m_riskLimitsPanel->setRiskGuard(m_riskGuard);
    m_riskLimitsPanel->setPositionBook(m_positionBook);
    // Persist edits to ~/.config/btquant_vulkan/state.ini. Capture
    // state.ini path now so the closure doesn't dereference `this`.
    std::string settingsPath = configDir + "state.ini";
    m_riskLimitsPanel->setPersistFn(
        [this, settingsPath](const ::btquant::RiskGuard& g) {
            util::Settings s = util::Settings::load(settingsPath);
            const auto& c = g.config();
            s.risk_maxPositionSizeUSD = c.maxPositionSizeUSD;
            s.risk_maxLeverage        = c.maxLeverage;
            s.risk_killOnDailyLossUSD = c.killOnDailyLossUSD;
            s.risk_equityUSD          = c.equityUSD;
            s.save(settingsPath);
            BTQ_LOG_INFO("RiskLimits: persisted to %s", settingsPath.c_str());
        });
    m_miniPriceChart = new MiniPriceChart();
    m_miniPriceChart->setMarketData(m_marketData);
    m_hotkeyEditor = new ::btquant::widgets::HotkeyEditor();
    // HotkeyEditor is wired after m_hotkeyMap is constructed (below).
    // Trade journal lives in the user's config dir alongside settings.ini.
    std::string journalPath = configDir + "journal.jsonl";
    m_tradeJournal  = new ::btquant::TradeJournal(journalPath);
    {
        int skipped = 0;
        size_t onDisk = m_tradeJournal->count();
        auto history  = m_tradeJournal->loadAll(&skipped);
        BTQ_LOG_INFO("TradeJournal: %zu fills on disk at %s (skipped %d)",
                     onDisk, journalPath.c_str(), skipped);
        // Rehydrate the PositionBook from the persistent journal so the
        // open position + session realized P&L survive restart. Without
        // this, every launch starts from empty even though the journal
        // has the complete fill history.
        if (m_positionBook && m_tradeJournal && onDisk > 0) {
            double lastPx = 0.0;
            size_t applied = m_positionBook->replay(*m_tradeJournal, &lastPx);
            BTQ_LOG_INFO("PositionBook: replayed %zu/%zu fills (last price %.2f)",
                         applied, onDisk, lastPx);
            if (m_positionBook->hasPosition() && lastPx > 0.0) {
                m_positionBook->markToMarket(lastPx);
            }
        }
    }

    // HotkeyMap — load user customizations from hotkeys.ini; fall back
    // to the built-in defaults (which mirror the previous hardcoded
    // bindings) if no file exists or it's malformed.
    std::string hotkeyPath = configDir + "hotkeys.ini";
    m_hotkeyPath = hotkeyPath;
    auto loadedMap = ::btquant::util::HotkeyMap::loadFromFile(hotkeyPath);
    if (loadedMap.has_value()) {
        m_hotkeyMap = new ::btquant::util::HotkeyMap(*loadedMap);
        BTQ_LOG_INFO("HotkeyMap: loaded %d bindings from %s",
                     static_cast<int>(m_hotkeyMap->enumerate().size()),
                     hotkeyPath.c_str());
    } else {
        m_hotkeyMap = new ::btquant::util::HotkeyMap(
            ::btquant::util::HotkeyMap::defaults());
        BTQ_LOG_INFO("HotkeyMap: using built-in defaults (no %s)",
                     hotkeyPath.c_str());
    }
    // Save back so the user has a template to edit.
    if (m_hotkeyMap) m_hotkeyMap->saveToFile(hotkeyPath);
    if (m_hotkeyEditor) m_hotkeyEditor->setHotkeyMap(m_hotkeyMap);

    // OrderTicket submit → PositionBook.fill(). The ticket's sign-aware
    // size (positive for buy, negative for sell) is what feeds the book;
    // we split it into direction + magnitude for clarity.
    m_orderTicket->setSubmitFn([this](const std::string& summary) {
        BTQ_LOG_INFO("OrderTicket.submit: %s", summary.c_str());
        if (!m_orderTicket || !m_positionBook) return;
        double qty    = m_orderTicket->quantity();
        bool   isBuy  = m_orderTicket->isBuy();
        // Use the most recent snapshot price as the fill reference when
        // the live processor is connected; otherwise fall back to the
        // ticket's limit-price input.
        double price  = m_orderTicket->limitPrice();
        std::string sym = m_marketData ? m_marketData->symbol()
                                       : std::string("BTC/USDT");
        if (m_marketData) {
            auto snap = m_marketData->snapshot(1, 0);
            if (!snap.recent_trades.empty()) {
                price = snap.recent_trades.front().price;
            }
        }
        if (qty <= 0.0 || price <= 0.0) {
            BTQ_LOG_WARN("OrderTicket.submit ignored: qty=%.4f price=%.2f",
                         qty, price);
            return;
        }
        // Pre-trade risk check — reject before mutating the book.
        if (m_riskGuard) {
            auto reject = m_riskGuard->checkOrder(qty, price, isBuy);
            if (reject.has_value()) {
                BTQ_LOG_WARN("OrderTicket REJECTED: %s", reject->c_str());
                return;
            }
        }
        double realized = m_positionBook->fill(sym, isBuy, qty, price);
        if (m_riskGuard && std::fabs(realized) > 0.0) {
            m_riskGuard->addRealized(realized);
            if (m_riskGuard->isKillTripped()) {
                BTQ_LOG_ERROR("RiskGuard: %s",
                    ::btquant::RiskGuard::killReason(
                        m_riskGuard->sessionRealized(),
                        m_riskGuard->config().killOnDailyLossUSD).c_str());
            }
        }
        // Append to trade journal for cross-restart persistence.
        if (m_tradeJournal) {
            ::btquant::JournalFill jf;
            jf.timestamp_us  = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            jf.symbol        = sym;
            jf.isLong        = isBuy;
            jf.qty           = qty;
            jf.price         = price;
            jf.realizedDelta = realized;
            if (!m_tradeJournal->append(jf)) {
                BTQ_LOG_WARN("TradeJournal.append failed at %s",
                             m_tradeJournal->path().c_str());
            }
        }
        if (m_positionPanel) {
            PositionPanel::FillRecord r;
            r.symbol         = sym;
            r.isLong         = isBuy;
            r.qty            = qty;
            r.price          = price;
            r.realizedDelta  = realized;
            m_positionPanel->recordFill(r);
        }
        if (std::fabs(realized) > 0.0) {
            BTQ_LOG_INFO("PositionBook.fill: realized %s$%.4f on %s %s %.4f",
                         realized >= 0 ? "+" : "", realized,
                         sym.c_str(), isBuy ? "BUY" : "SELL", qty);
        }
    });
}

WindowManager::~WindowManager() {
    delete m_orderBookWidget;
    delete m_orderBookDepthWidget;
    delete m_footprintWidget;
    delete m_vpvrWidget;
    delete m_multiVwapWidget;
    delete m_riskPanel;
    delete m_domWidget;
    delete m_tradesWidget;
    delete m_tpoWidget;
    delete m_alertsPanel;
    delete m_watchlistWidget;
    delete m_connectionPanel;
    delete m_profileManager;
    delete m_symbolPicker;
    delete m_themeEditor;
    delete m_positionCalculator;
    delete m_orderTicket;
    delete m_positionPanel;
    delete m_positionBook;
    delete m_riskLimitsPanel;
    delete m_hotkeyEditor;
    delete m_miniPriceChart;
    delete m_hotkeyMap;
    delete m_riskGuard;
    delete m_tradeJournal;
    // m_logPanel is a singleton — do not delete.
}

void WindowManager::initialize() {
    m_initialized = true;
    applyPersistedTheme();
    applyPersistedRiskConfig();
}

void WindowManager::applyPersistedTheme() {
    if (!m_themeEditor) return;
    auto snap = ThemeIO::load(ThemeIO::defaultPath());
    if (!snap) {
        BTQ_LOG_INFO("no persisted theme — using ImGui default (dark)");
        return;
    }
    ThemeEditor::applySnapshot(ImGui::GetStyle(), *snap);
    BTQ_LOG_INFO("applied persisted theme from %s",
                 ThemeIO::defaultPath().string().c_str());
}

void WindowManager::applyPersistedRiskConfig() {
    if (!m_riskGuard) return;
    const char* home = std::getenv("HOME");
    std::string path = std::string(home ? home : "/tmp") +
                       "/.config/btquant_vulkan/state.ini";
    auto s = util::Settings::load(path);
    ::btquant::RiskConfig c = m_riskGuard->config();
    c.maxPositionSizeUSD = s.risk_maxPositionSizeUSD;
    c.maxLeverage        = s.risk_maxLeverage;
    c.killOnDailyLossUSD = s.risk_killOnDailyLossUSD;
    c.equityUSD          = s.risk_equityUSD;
    m_riskGuard->setConfig(c);
    BTQ_LOG_INFO("applied persisted risk config (cap=$%.0f lev=%.2fx kill=$%.0f eq=$%.0f)",
                 c.maxPositionSizeUSD, c.maxLeverage,
                 c.killOnDailyLossUSD, c.equityUSD);
}

bool WindowManager::saveCurrentTheme() {
    if (!m_themeEditor) return false;
    auto snap = ThemeEditor::capture(ImGui::GetStyle());
    auto path = ThemeIO::defaultPath();
    if (!ThemeIO::save(path, snap)) {
        BTQ_LOG_WARN("failed to save theme to %s", path.string().c_str());
        return false;
    }
    BTQ_LOG_INFO("saved theme to %s", path.string().c_str());
    return true;
}

void WindowManager::shutdown() {
    m_initialized = false;
}

void WindowManager::buildDockLayout() {
    // Anchor the layout on the main dockspace (ID 0 = root dockspace created
    // by DockSpaceOverViewport in main.cpp).
    const ImGuiID dockspaceId = 0;
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGuiID root = ImGui::DockBuilderAddNode(dockspaceId,
                                             ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(root, ImGui::GetMainViewport()->Size);

    // Split horizontal first: [ LEFT (OrderBook) | CENTER (DOM) | RIGHT (VWAP/VPVR/Footprint) ]
    ImGuiID left = 0, center = 0, right = 0;
    ImGui::DockBuilderSplitNode(root, ImGuiDir_Left, 0.22f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, &right, &center);

    // LEFT: split vertically → OrderBook (top), OrderBookDepth (bottom).
    ImGuiID ob_top = 0, ob_bot = 0;
    ImGui::DockBuilderSplitNode(left, ImGuiDir_Up, 0.55f, &ob_top, &ob_bot);

    // RIGHT: split vertically → MultiVWAP (top), VPVR (mid), Footprint (bottom).
    ImGuiID r_top = 0, r_mid_bot = 0;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.40f, &r_top, &r_mid_bot);
    ImGuiID r_mid = 0, r_bot = 0;
    ImGui::DockBuilderSplitNode(r_mid_bot, ImGuiDir_Up, 0.50f, &r_mid, &r_bot);

    // CENTER (DOM): split horizontally → Trades (top), DOM (mid), RiskPanel + TPO (bottom).
    ImGuiID c_top = 0, c_mid_bot = 0;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Up, 0.20f, &c_top, &c_mid_bot);
    ImGuiID c_mid = 0, c_bot = 0;
    ImGui::DockBuilderSplitNode(c_mid_bot, ImGuiDir_Up, 0.65f, &c_mid, &c_bot);
    ImGuiID c_bot_l = 0, c_bot_r = 0;
    ImGui::DockBuilderSplitNode(c_bot, ImGuiDir_Left, 0.50f, &c_bot_l, &c_bot_r);

    // Bind windows to nodes.
    ImGui::DockBuilderDockWindow("Order Book",         ob_top);
    ImGui::DockBuilderDockWindow("Order Book Depth",  ob_bot);
    ImGui::DockBuilderDockWindow("Multi VWAP",        r_top);
    ImGui::DockBuilderDockWindow("VPVR",              r_mid);
    ImGui::DockBuilderDockWindow("Footprint",         r_bot);
    ImGui::DockBuilderDockWindow("Trades",            c_top);
    ImGui::DockBuilderDockWindow("DOM",               c_mid);
    ImGui::DockBuilderDockWindow("Risk Panel",        c_bot_l);
    ImGui::DockBuilderDockWindow("TPO",               c_bot_r);
    ImGui::DockBuilderDockWindow("Heatmap",           root);  // heatmap as floating overlay

    ImGui::DockBuilderFinish(root);
}

void WindowManager::applyInitialDockLayoutIfNeeded() {
    if (m_layoutResetRequested) {
        m_layoutResetRequested = false;
        m_layoutApplied = false;
    }
    if (m_layoutApplied) return;

    // Only build on first frame after at least one widget has been rendered
    // (ImGui needs a frame to register the DockSpace ID).
    const ImGuiID dockspaceId = 0;
    if (ImGui::DockBuilderGetNode(dockspaceId) == nullptr) {
        // DockSpaceOverViewport hasn't run yet — wait one frame.
        return;
    }

    buildDockLayout();
    m_layoutApplied = true;
}

void WindowManager::requestDockLayoutReset() {
    m_layoutResetRequested = true;
}

void WindowManager::applyPreset(const ::btquant::util::Settings& s) {
    s.applyTo(showOrderBook, showOrderBookDepth, showFootprint, showVPVR,
              showMultiVWAP, showRiskPanel, showDOM, showTrades, showTPO,
              theme, heatmapDensity);
    markSettingsDirty();
    // The next applyInitialDockLayoutIfNeeded() call will rebuild the dock
    // tree to match the new widget set. We request a reset so the layout
    // re-applies cleanly even if the user just hid a window.
    requestDockLayoutReset();
}

::btquant::util::Settings WindowManager::captureCurrentSettings() const {
    ::btquant::util::Settings s;
    s.showOrderBook      = showOrderBook;
    s.showOrderBookDepth = showOrderBookDepth;
    s.showFootprint      = showFootprint;
    s.showVPVR           = showVPVR;
    s.showMultiVWAP      = showMultiVWAP;
    s.showRiskPanel      = showRiskPanel;
    s.showDOM            = showDOM;
    s.showTrades         = showTrades;
    s.showTPO            = showTPO;
    s.theme              = theme;
    s.heatmapDensity     = heatmapDensity;
    return s;
}

void WindowManager::processHotkeys(void* glfwWindow) {
#ifdef BTQUANT_USE_GLFW
    if (!glfwWindow) return;
    auto* win = static_cast<GLFWwindow*>(glfwWindow);

    ImGuiIO& io = ImGui::GetIO();
    bool textFieldFocus = io.WantCaptureKeyboard && io.WantTextInput;

    // Live modifier snapshot — used by every per-action edge-trigger
    // check below.
    bool ctrlDown  = glfwGetKey(win, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
                     glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    bool shiftDown = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS ||
                     glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS;

    // F2..F12 toggle widgets — separate table because they toggle
    // WindowManager member bools via pointer; consults the map so user
    // remappings take effect.
    constexpr size_t kNumHotkeys = sizeof(kHotkeys) / sizeof(kHotkeys[0]);
    static bool prevPressed[kNumHotkeys] = {};
    bool currPressed[kNumHotkeys];
    for (size_t i = 0; i < kNumHotkeys; ++i) {
        int boundKey = GLFW_KEY_UNKNOWN;
        if (m_hotkeyMap) {
            int k = m_hotkeyMap->get(kHotkeys[i].action).glfwKey;
            if (k >= 0) boundKey = k;
        }
        bool ctrlReq  = m_hotkeyMap && m_hotkeyMap->get(kHotkeys[i].action).ctrl;
        bool shiftReq = m_hotkeyMap && m_hotkeyMap->get(kHotkeys[i].action).shift;
        currPressed[i] = !textFieldFocus && boundKey != GLFW_KEY_UNKNOWN &&
                         glfwGetKey(win, boundKey) == GLFW_PRESS &&
                         (ctrlReq  ? ctrlDown  : true) &&
                         (shiftReq ? shiftDown : true);
        if (currPressed[i] && !prevPressed[i]) {
            this->*(kHotkeys[i].flag) = !(this->*(kHotkeys[i].flag));
            markSettingsDirty();
        }
        prevPressed[i] = currPressed[i];
    }

    // Ctrl+H opens the HotkeyEditor. When the editor is in capture mode,
    // route the next non-modifier keypress to it instead of the regular
    // hotkey dispatch.
    static bool prevCtrlH = false;
    bool currCtrlH = !textFieldFocus &&
                     glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS &&
                     (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    if (currCtrlH && !prevCtrlH) {
        if (m_hotkeyEditor) m_hotkeyEditor->toggleOpen();
        if (m_hotkeyEditor) m_hotkeyEditorOpen = m_hotkeyEditor->isOpen();
        markSettingsDirty();
    }
    prevCtrlH = currCtrlH;

    // If the editor is capturing, scan every GLFW key for a rising edge
    // and inject the first one into the editor. We iterate all 348
    // GLFW_KEY_LAST slots so letter, digit, function, arrow and
    // punctuation keys all flow through the same path.
    if (m_hotkeyEditor && m_hotkeyEditor->isOpen()) {
        static bool prevCapturedKeys[512] = {};
        for (int key = 32; key < 512; ++key) {
            bool down = glfwGetKey(win, key) == GLFW_PRESS;
            if (down && !prevCapturedKeys[key] && !textFieldFocus) {
                m_hotkeyEditor->injectCapture(key, ctrlDown, shiftDown);
                prevCapturedKeys[key] = true;
                if (m_hotkeyMap) m_hotkeyMap->saveToFile(m_hotkeyPath);
                break;
            }
            prevCapturedKeys[key] = down;
        }
    }

    // Map-driven dispatch for everything outside the F-key toggle table
    // (Ctrl-prefixed actions, Shift-prefixed actions, and any remapping
    // the user has applied). One pass over the HotkeyMap → one dispatch.
    // Edge-triggered per action via prevAction[].
    if (m_hotkeyMap) {
        using HA = ::btquant::util::HotkeyAction;
        constexpr int kNumActions = static_cast<int>(HA::COUNT);
        static bool prevAction[kNumActions] = {};
        auto rows = m_hotkeyMap->enumerate();
        for (size_t i = 0; i < rows.size(); ++i) {
            HA action = rows[i].first;
            const auto& b = rows[i].second;
            if (b.glfwKey < 0) continue;     // unbound
            int idx = static_cast<int>(action);
            if (idx < 0 || idx >= kNumActions) continue;
            bool curr = !textFieldFocus &&
                        glfwGetKey(win, b.glfwKey) == GLFW_PRESS &&
                        (b.ctrl  ? ctrlDown  : true) &&
                        (b.shift ? shiftDown : true);
            if (curr && !prevAction[idx]) {
                dispatchAction(action);
            }
            prevAction[idx] = curr;
        }
    }
#endif // BTQUANT_USE_GLFW
}

void WindowManager::dispatchAction(::btquant::util::HotkeyAction a) {
    using HA = ::btquant::util::HotkeyAction;
    switch (a) {
        case HA::ResetLayout:
            requestDockLayoutReset();
            markSettingsDirty();
            break;
        case HA::OpenSymbolPicker:
            showSymbolPickerOpen = !showSymbolPickerOpen;
            if (m_symbolPicker) m_symbolPicker->setOpen(showSymbolPickerOpen);
            markSettingsDirty();
            break;
        case HA::OpenThemeEditor:
            showThemeEditorOpen = !showThemeEditorOpen;
            if (m_themeEditor) m_themeEditor->setOpen(showThemeEditorOpen);
            markSettingsDirty();
            break;
        case HA::ToggleOrderTicket:
            showOrderTicket = !showOrderTicket;
            if (m_orderTicket) m_orderTicket->setOpen(showOrderTicket);
            markSettingsDirty();
            break;
        case HA::TogglePositionPanel:
            showPositionPanel = !showPositionPanel;
            if (m_positionPanel) m_positionPanel->setOpen(showPositionPanel);
            markSettingsDirty();
            break;
        case HA::ToggleRiskLimits:
            showRiskLimits = !showRiskLimits;
            if (m_riskLimitsPanel) m_riskLimitsPanel->setOpen(showRiskLimits);
            markSettingsDirty();
            break;
        case HA::ToggleMiniPriceChart:
            showMiniPriceChart = !showMiniPriceChart;
            if (m_miniPriceChart) m_miniPriceChart->setOpen(showMiniPriceChart);
            markSettingsDirty();
            break;
        case HA::ToggleStats:
            showStatsOverlay = !showStatsOverlay;
            m_statsOverlay.setEnabled(showStatsOverlay);
            markSettingsDirty();
            break;
        case HA::ToggleHotkeyHelp:
            showHotkeyHelp = !showHotkeyHelp;
            markSettingsDirty();
            break;
        case HA::ToggleHotkeyEditor:
            if (m_hotkeyEditor) m_hotkeyEditor->toggleOpen();
            if (m_hotkeyEditor) m_hotkeyEditorOpen = m_hotkeyEditor->isOpen();
            markSettingsDirty();
            break;
        case HA::KillSwitch:
            if (m_positionBook && m_positionBook->hasPosition()) {
                double px = 0.0;
                if (m_marketData) {
                    auto snap = m_marketData->snapshot(1, 0);
                    if (!snap.recent_trades.empty()) {
                        px = snap.recent_trades.front().price;
                    } else if (snap.order_book.midPrice > 0.0) {
                        px = snap.order_book.midPrice;
                    }
                }
                if (px <= 0.0) {
                    BTQ_LOG_WARN("KillSwitch ignored: no live price available");
                } else {
                    double realized = m_positionBook->flatten(px);
                    if (m_riskGuard) m_riskGuard->addRealized(realized);
                    if (m_tradeJournal) {
                        ::btquant::JournalFill jf;
                        jf.timestamp_us  = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        jf.symbol        = m_positionBook->position().symbol;
                        jf.isLong        = !m_positionBook->position().isLong;
                        jf.qty           = m_positionBook->position().size;
                        jf.price         = px;
                        jf.realizedDelta = realized;
                        m_tradeJournal->append(jf);
                    }
                    BTQ_LOG_WARN("KILL SWITCH: flattened %s at $%.2f, "
                                 "realized %s$%.2f, session P&L %s$%.2f",
                                 m_positionBook->position().symbol.c_str(),
                                 px,
                                 realized >= 0 ? "+" : "", realized,
                                 (m_riskGuard ? m_riskGuard->sessionRealized() : 0.0)
                                    >= 0 ? "+" : "",
                                 m_riskGuard ? m_riskGuard->sessionRealized() : 0.0);
                }
            } else {
                BTQ_LOG_INFO("KillSwitch: no open position to flatten");
            }
            break;
        // F2..F12 toggle widgets are dispatched by the kHotkeys table
        // loop (uses member pointers); dispatchAction is not called for
        // those — they never reach here.
        default:
            break;
    }
}

void WindowManager::renderStatsOverlay(uint64_t tradeQueueDepth,
                                       uint64_t candleCount) {
    m_statsOverlay.setEnabled(showStatsOverlay);
    m_statsOverlay.render(tradeQueueDepth, candleCount);
}

void WindowManager::setMarketData(::btquant::MarketDataProcessor* data) {
    m_marketData = data;
    if (m_orderTicket) m_orderTicket->setMarketData(data);
    if (m_miniPriceChart) m_miniPriceChart->setMarketData(data);
    if (m_orderBookWidget) m_orderBookWidget->setMarketData(data);
    if (m_orderBookDepthWidget) m_orderBookDepthWidget->setMarketData(data);
    if (m_footprintWidget) m_footprintWidget->setMarketData(data);
    if (m_vpvrWidget) m_vpvrWidget->setMarketData(data);
    if (m_multiVwapWidget) m_multiVwapWidget->setMarketData(data);
    if (m_riskPanel) m_riskPanel->setMarketData(data);
    if (m_domWidget) m_domWidget->setMarketData(data);
    if (m_tradesWidget) m_tradesWidget->setMarketData(data);
    if (m_tpoWidget) m_tpoWidget->setMarketData(data);
    if (m_alertsPanel) m_alertsPanel->setMarketData(data);
    if (m_connectionPanel) m_connectionPanel->setMarketData(data);
    // Watchlist uses push-only API (WindowManager::updateWatchlist) — no
    // setMarketData hook needed.
}

void WindowManager::showOrderBookWindow() {
    if (!showOrderBook) return;
    m_orderBookWidget->render();
}

void WindowManager::showOrderBookDepthWindow() {
    if (!showOrderBookDepth) return;
    m_orderBookDepthWidget->render();
}

void WindowManager::showFootprintWindow() {
    if (!showFootprint) return;
    m_footprintWidget->render();
}

void WindowManager::showVPVRWindow() {
    if (!showVPVR) return;
    m_vpvrWidget->render();
}

void WindowManager::showMultiVWAPWindow() {
    if (!showMultiVWAP) return;
    m_multiVwapWidget->render();
}

void WindowManager::showRiskPanelWindow() {
    if (!showRiskPanel) return;
    m_riskPanel->render();
}

void WindowManager::showDOMWindow() {
    if (!showDOM) return;
    m_domWidget->render();
}

void WindowManager::showTradesWindow() {
    if (!showTrades) return;
    m_tradesWidget->render();
}

void WindowManager::showTPOWindow() {
    if (!showTPO) return;
    m_tpoWidget->render();
}

void WindowManager::showAlertsWindow() {
    if (!showAlerts) return;
    m_alertsPanel->render();
}

void WindowManager::showWatchlistWindow() {
    if (!showWatchlist) return;
    m_watchlistWidget->render();
}

void WindowManager::updateWatchlist(const std::string& sym, double p, double s,
                                    bool b, uint64_t ts) {
    if (m_watchlistWidget) m_watchlistWidget->update(sym, p, s, b, ts);
}

void WindowManager::showLogWindow() {
    if (!showLog) return;
    if (m_logPanel) m_logPanel->render();
}

void WindowManager::showConnectionWindow() {
    if (!showConnection) return;
    if (m_connectionPanel) m_connectionPanel->render();
}

void WindowManager::showProfileManagerWindow() {
    if (!showProfileManager) return;
    if (m_profileManager) m_profileManager->render();
}

void WindowManager::showSymbolPickerWindow() {
    if (!showSymbolPickerOpen) return;
    if (m_symbolPicker) m_symbolPicker->render();
}

void WindowManager::showThemeEditorWindow() {
    if (!showThemeEditorOpen) return;
    if (m_themeEditor) m_themeEditor->render();
}

void WindowManager::showPositionCalculatorWindow() {
    if (!showPositionCalculator) return;
    if (m_positionCalculator) m_positionCalculator->render();
}

void WindowManager::showOrderTicketWindow() {
    if (!showOrderTicket) return;
    if (m_orderTicket) m_orderTicket->render();
}

void WindowManager::showPositionPanelWindow() {
    if (!showPositionPanel) return;
    // Drive mark-to-market off the latest snapshot mid price before render
    // so the panel shows live unrealized P&L. Cheap — locks the snapshot
    // mutex briefly then renders.
    if (m_positionBook && m_positionBook->hasPosition() && m_marketData) {
        auto snap = m_marketData->snapshot(1, 0);
        if (!snap.recent_trades.empty()) {
            m_positionBook->markToMarket(snap.recent_trades.front().price);
        } else if (snap.order_book.midPrice > 0.0) {
            m_positionBook->markToMarket(snap.order_book.midPrice);
        }
    }
    if (m_positionPanel) m_positionPanel->render();
}

void WindowManager::showRiskLimitsWindow() {
    if (!showRiskLimits) return;
    if (m_riskLimitsPanel) m_riskLimitsPanel->render();
}

void WindowManager::showMiniPriceChartWindow() {
    if (!showMiniPriceChart) return;
    if (m_marketData) {
        m_miniPriceChart->setMarketData(m_marketData);
    }
    if (m_miniPriceChart) m_miniPriceChart->render();
}

void WindowManager::showHotkeyEditorWindow() {
    if (!m_hotkeyEditor) return;
    m_hotkeyEditor->setOpen(m_hotkeyEditorOpen);
    m_hotkeyEditor->render();
    // Sync back in case the user closed the window via the [X] button.
    m_hotkeyEditorOpen = m_hotkeyEditor->isOpen();
    // Persist on close with pending changes.
    if (!m_hotkeyEditorOpen && m_hotkeyEditor->isDirty()) {
        m_hotkeyEditor->clearDirty();
        if (m_hotkeyMap) m_hotkeyMap->saveToFile(m_hotkeyPath);
    }
}

void WindowManager::showMainMenu() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Order Book",        nullptr, &showOrderBook))       markSettingsDirty();
            if (ImGui::MenuItem("Order Book Depth",  nullptr, &showOrderBookDepth))  markSettingsDirty();
            if (ImGui::MenuItem("DOM",               nullptr, &showDOM))             markSettingsDirty();
            if (ImGui::MenuItem("Trades",            nullptr, &showTrades))          markSettingsDirty();
            if (ImGui::MenuItem("TPO",               nullptr, &showTPO))             markSettingsDirty();
            if (ImGui::MenuItem("Footprint",         nullptr, &showFootprint))       markSettingsDirty();
            if (ImGui::MenuItem("VPVR",              nullptr, &showVPVR))            markSettingsDirty();
            if (ImGui::MenuItem("Multi VWAP",        nullptr, &showMultiVWAP))       markSettingsDirty();
            if (ImGui::MenuItem("Risk Panel",        nullptr, &showRiskPanel))       markSettingsDirty();
            if (ImGui::MenuItem("Alerts",            nullptr, &showAlerts))          markSettingsDirty();
            if (ImGui::MenuItem("Watchlist",         nullptr, &showWatchlist))       markSettingsDirty();
            if (ImGui::MenuItem("Log Panel",         nullptr, &showLog))             markSettingsDirty();
            if (ImGui::MenuItem("Connection",        nullptr, &showConnection))      markSettingsDirty();
            if (ImGui::MenuItem("Profile Manager…",  nullptr, &showProfileManager))  markSettingsDirty();
            if (ImGui::MenuItem("Symbol Picker… (Ctrl+P)", nullptr, &showSymbolPickerOpen)) markSettingsDirty();
            if (ImGui::MenuItem("Theme Editor… (Ctrl+T)",   nullptr, &showThemeEditorOpen))  markSettingsDirty();
            if (ImGui::MenuItem("Position Calculator",  nullptr, &showPositionCalculator)) markSettingsDirty();
            if (ImGui::MenuItem("Order Ticket (Ctrl+Enter)", nullptr, &showOrderTicket))  markSettingsDirty();
            if (ImGui::MenuItem("Position Panel (Ctrl+B)",     nullptr, &showPositionPanel))markSettingsDirty();
            if (ImGui::MenuItem("Risk Dashboard (Ctrl+R)",      nullptr, &showRiskLimits))   markSettingsDirty();
            if (ImGui::MenuItem("Mini Price Chart (Ctrl+M)",    nullptr, &showMiniPriceChart))markSettingsDirty();
            if (ImGui::MenuItem("Hotkey Editor (Ctrl+H)",       nullptr, &m_hotkeyEditorOpen)) {
                if (m_hotkeyEditor) m_hotkeyEditor->setOpen(m_hotkeyEditorOpen);
                markSettingsDirty();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Settings…",         nullptr, &showSettings))        markSettingsDirty();
            if (ImGui::MenuItem("Hotkey Help…",      nullptr, &showHotkeyHelp))      markSettingsDirty();
            ImGui::Separator();
            if (ImGui::BeginMenu("Theme")) {
                if (ImGui::MenuItem("Dark (Kraken Purple)", nullptr, theme == 0 ? &kBoolTrue : &kBoolFalse)) {
                    theme = 0; markSettingsDirty();
                }
                if (ImGui::MenuItem("Light (off-white)",    nullptr, theme == 1 ? &kBoolTrue : &kBoolFalse)) {
                    theme = 1; markSettingsDirty();
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::BeginMenu("Profiles")) {
                if (ImGui::MenuItem("Scalper"))      { applyPreset(util::Settings::presetScalper());      }
                if (ImGui::MenuItem("Market Maker")) { applyPreset(util::Settings::presetMarketMaker()); }
                if (ImGui::MenuItem("Volatility"))   { applyPreset(util::Settings::presetVolatility());   }
                if (ImGui::MenuItem("Fullscreen"))   { applyPreset(util::Settings::presetFullscreen());   }
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Reset Layout")) {
                requestDockLayoutReset();
                markSettingsDirty();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Hotkey Reference", nullptr, &showHotkeyHelp)) markSettingsDirty();
            ImGui::MenuItem("About", nullptr, nullptr);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void WindowManager::showHotkeyHelpWindow() {
    if (!showHotkeyHelp) return;
    ImGui::SetNextWindowSize(ImVec2(440, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Hotkey Reference", &showHotkeyHelp)) {
        ImGui::End();
        return;
    }

    ImGui::TextWrapped("Global shortcuts. Hotkeys are suppressed while typing into a "
                       "text field so search bars stay usable. The table reflects your "
                       "current binding layout from ~/.config/btquant_vulkan/hotkeys.ini "
                       "— open the Hotkey Editor (Ctrl+H) to remap.");
    ImGui::Separator();

    if (ImGui::BeginTable("hotkeys", 2, ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Key",  ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Action");
        ImGui::TableHeadersRow();

        // Reflect the user's current binding map, not the hardcoded
        // defaults — if they've remapped F2 → F3, the help window
        // shows F3.
        auto display = [&](::btquant::util::HotkeyAction a, const char* desc) {
            std::string keyStr = "(unbound)";
            if (m_hotkeyMap && m_hotkeyMap->has(a)) {
                keyStr = m_hotkeyMap->get(a).label();
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(keyStr.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(desc);
        };
        using HA = ::btquant::util::HotkeyAction;
        display(HA::ToggleOrderBook,        "Toggle Order Book");
        display(HA::ToggleOrderBookDepth,   "Toggle Order Book Depth");
        display(HA::ToggleDOM,              "Toggle DOM");
        display(HA::ToggleTrades,           "Toggle Trades");
        display(HA::ToggleTPO,              "Toggle TPO");
        display(HA::ToggleFootprint,        "Toggle Footprint");
        display(HA::ToggleVPVR,             "Toggle VPVR");
        display(HA::ToggleAlerts,           "Toggle Alerts");
        display(HA::ToggleMultiVWAP,        "Toggle Multi VWAP");
        display(HA::ToggleRiskPanel,        "Toggle Risk Panel");
        display(HA::ToggleSettings,         "Toggle Settings window");
        display(HA::ToggleStats,            "Toggle Stats overlay");
        display(HA::ToggleHotkeyHelp,       "Toggle this Hotkey Reference");
        display(HA::ResetLayout,            "Reset docking layout");
        display(HA::OpenSymbolPicker,       "Open Symbol Picker");
        display(HA::OpenThemeEditor,        "Open Theme Editor");
        display(HA::ToggleOrderTicket,      "Toggle Order Ticket");
        display(HA::TogglePositionPanel,    "Toggle Position Panel");
        display(HA::ToggleRiskLimits,       "Toggle Risk Dashboard");
        display(HA::ToggleMiniPriceChart,   "Toggle Mini Price Chart");
        display(HA::ToggleHotkeyEditor,     "Open Hotkey Editor (remap bindings)");
        display(HA::KillSwitch,             "Kill switch — flatten open position at market");

        ImGui::EndTable();
    }
    ImGui::Separator();
    ImGui::TextDisabled("ESC: close topmost popup / window");
    if (ImGui::Button("Close")) showHotkeyHelp = false;
    ImGui::End();
}

void WindowManager::showSettingsWindow() {
    if (!showSettings) return;
    ImGui::SetNextWindowSize(ImVec2(420, 260), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Settings", &showSettings)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Rendering");
    if (ImGui::SliderScalar("FPS limit (0 = uncapped)", ImGuiDataType_S64,
                            &fpsLimit, &kZero, &kFps240, "%ld")) {
        markSettingsDirty();
    }
    if (ImGui::SliderScalar("Heatmap density", ImGuiDataType_S64,
                            &heatmapDensity, &kHeatmapMin, &kHeatmapMax, "%ld")) {
        markSettingsDirty();
    }

    ImGui::Separator();
    ImGui::Text("Visible widgets");
    if (ImGui::MenuItem("Order Book",        nullptr, &showOrderBook))       markSettingsDirty();
    if (ImGui::MenuItem("Order Book Depth",  nullptr, &showOrderBookDepth))  markSettingsDirty();
    if (ImGui::MenuItem("DOM",               nullptr, &showDOM))             markSettingsDirty();
    if (ImGui::MenuItem("Trades",            nullptr, &showTrades))          markSettingsDirty();
    if (ImGui::MenuItem("TPO",               nullptr, &showTPO))             markSettingsDirty();
    if (ImGui::MenuItem("Footprint",         nullptr, &showFootprint))       markSettingsDirty();
    if (ImGui::MenuItem("VPVR",              nullptr, &showVPVR))            markSettingsDirty();
    if (ImGui::MenuItem("Multi VWAP",        nullptr, &showMultiVWAP))       markSettingsDirty();
    if (ImGui::MenuItem("Risk Panel",        nullptr, &showRiskPanel))       markSettingsDirty();
    if (ImGui::MenuItem("Stats overlay",     nullptr, &showStatsOverlay))    markSettingsDirty();

    ImGui::Separator();
    if (ImGui::Button("Reset Layout")) {
        requestDockLayoutReset();
        markSettingsDirty();
    }
    ImGui::SameLine();
    if (ImGui::Button("Close")) {
        showSettings = false;
        markSettingsDirty();
    }

    ImGui::End();
}

} // namespace btquant::ui
