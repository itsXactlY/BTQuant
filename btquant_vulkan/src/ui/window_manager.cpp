#include "window_manager.hpp"

#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder*

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
#include "stats_overlay.hpp"

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
// Toggle widget given its F-key hotkey (F2-F9) — looks up in a static table.
struct HotkeyBinding { int glfwKey; bool WindowManager::*flag; const char* name; };
constexpr HotkeyBinding kHotkeys[] = {
    { GLFW_KEY_F2,  &WindowManager::showOrderBook,      "Order Book"        },
    { GLFW_KEY_F3,  &WindowManager::showOrderBookDepth, "Order Book Depth"  },
    { GLFW_KEY_F4,  &WindowManager::showDOM,            "DOM"               },
    { GLFW_KEY_F5,  &WindowManager::showTrades,         "Trades"            },
    { GLFW_KEY_F6,  &WindowManager::showTPO,            "TPO"               },
    { GLFW_KEY_F7,  &WindowManager::showFootprint,      "Footprint"         },
    { GLFW_KEY_F8,  &WindowManager::showVPVR,           "VPVR"              },
    // F9 is taken by ImGui's default for "show demo window" — we skip it.
    { GLFW_KEY_F10, &WindowManager::showMultiVWAP,      "Multi VWAP"        },
    { GLFW_KEY_F11, &WindowManager::showRiskPanel,      "Risk Panel"        },
    { GLFW_KEY_F12, &WindowManager::showSettings,       "Settings"          },
};
#endif // BTQUANT_USE_GLFW

} // namespace btquant::ui (constants + hotkey table)

namespace btquant::ui {

WindowManager::WindowManager() {
    m_orderBookWidget = new OrderBookWidget();
    m_orderBookDepthWidget = new OrderBookDepthWidget();
    m_footprintWidget = new FootprintWidget();
    m_vpvrWidget = new VPVRWidget();
    m_multiVwapWidget = new MultiVWAPWidget();
    m_riskPanel = new RiskPanel();
    m_domWidget = new DOMWidget();
    m_tradesWidget = new TradesWidget();
    m_tpoWidget = new TPOWidget();
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
}

void WindowManager::initialize() {
    m_initialized = true;
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

void WindowManager::processHotkeys(void* glfwWindow) {
#ifdef BTQUANT_USE_GLFW
    if (!glfwWindow) return;
    auto* win = static_cast<GLFWwindow*>(glfwWindow);

    ImGuiIO& io = ImGui::GetIO();
    bool textFieldFocus = io.WantCaptureKeyboard && io.WantTextInput;

    // F2..F12 toggle widgets. Edge-triggered: fire only on the rising edge
    // (key was up last frame, is down now) so holding the key down doesn't
    // rapidly retoggle the widget.
    constexpr size_t kNumHotkeys = sizeof(kHotkeys) / sizeof(kHotkeys[0]);
    static bool prevPressed[kNumHotkeys] = {};
    bool currPressed[kNumHotkeys];
    for (size_t i = 0; i < kNumHotkeys; ++i) {
        currPressed[i] = !textFieldFocus &&
                         glfwGetKey(win, kHotkeys[i].glfwKey) == GLFW_PRESS;
        if (currPressed[i] && !prevPressed[i]) {
            this->*(kHotkeys[i].flag) = !(this->*(kHotkeys[i].flag));
            markSettingsDirty();
        }
        prevPressed[i] = currPressed[i];
    }

    // Ctrl+L — reset layout (also edge-triggered so it fires once).
    static bool prevCtrlL = false;
    bool currCtrlL = !textFieldFocus &&
                     glfwGetKey(win, GLFW_KEY_L) == GLFW_PRESS &&
                     (glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    if (currCtrlL && !prevCtrlL) {
        requestDockLayoutReset();
        markSettingsDirty();
    }
    prevCtrlL = currCtrlL;

    // Shift+F1 toggles stats overlay.
    static bool prevShiftF1 = false;
    bool currShiftF1 = !textFieldFocus &&
                       glfwGetKey(win, GLFW_KEY_F1) == GLFW_PRESS &&
                       (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                        glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    if (currShiftF1 && !prevShiftF1) {
        showStatsOverlay = !showStatsOverlay;
        m_statsOverlay.setEnabled(showStatsOverlay);
        markSettingsDirty();
    }
    prevShiftF1 = currShiftF1;

    // ? (Shift+/) toggles the hotkey reference overlay. Suppressed when user
    // is typing into a text field so search filters work normally.
    static bool prevQuestionMark = false;
    bool currQuestionMark = !textFieldFocus &&
                            glfwGetKey(win, GLFW_KEY_SLASH) == GLFW_PRESS &&
                            (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                             glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    if (currQuestionMark && !prevQuestionMark) {
        showHotkeyHelp = !showHotkeyHelp;
        markSettingsDirty();
    }
    prevQuestionMark = currQuestionMark;
#endif // BTQUANT_USE_GLFW
}

void WindowManager::renderStatsOverlay(uint64_t tradeQueueDepth,
                                       uint64_t candleCount) {
    m_statsOverlay.setEnabled(showStatsOverlay);
    m_statsOverlay.render(tradeQueueDepth, candleCount);
}

void WindowManager::setMarketData(::btquant::MarketDataProcessor* data) {
    if (m_orderBookWidget) m_orderBookWidget->setMarketData(data);
    if (m_orderBookDepthWidget) m_orderBookDepthWidget->setMarketData(data);
    if (m_footprintWidget) m_footprintWidget->setMarketData(data);
    if (m_vpvrWidget) m_vpvrWidget->setMarketData(data);
    if (m_multiVwapWidget) m_multiVwapWidget->setMarketData(data);
    if (m_riskPanel) m_riskPanel->setMarketData(data);
    if (m_domWidget) m_domWidget->setMarketData(data);
    if (m_tradesWidget) m_tradesWidget->setMarketData(data);
    if (m_tpoWidget) m_tpoWidget->setMarketData(data);
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
                       "text field so search bars stay usable.");
    ImGui::Separator();

    if (ImGui::BeginTable("hotkeys", 2, ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Key",  ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("Action");
        ImGui::TableHeadersRow();

        auto row = [](const char* key, const char* action) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(key);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(action);
        };

        row("F2",       "Toggle Order Book");
        row("F3",       "Toggle Order Book Depth");
        row("F4",       "Toggle DOM");
        row("F5",       "Toggle Trades");
        row("F6",       "Toggle TPO");
        row("F7",       "Toggle Footprint");
        row("F8",       "Toggle VPVR");
        row("F10",      "Toggle Multi VWAP");
        row("F11",      "Toggle Risk Panel");
        row("F12",      "Toggle Settings window");
        row("Shift+F1", "Toggle Stats overlay");
        row("?",        "Toggle this Hotkey Reference");
        row("Ctrl+L",   "Reset docking layout");
        row("ESC",      "Close topmost popup / window");

        ImGui::EndTable();
    }

    ImGui::Separator();
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
