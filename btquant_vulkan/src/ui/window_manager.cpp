#include "window_manager.hpp"

#include <imgui.h>

#include "../widgets/order_book_widget.hpp"
#include "../widgets/order_book_depth_widget.hpp"
#include "../widgets/footprint_widget.hpp"
#include "../widgets/vpvr_widget.hpp"
#include "../widgets/dom_widget.hpp"
#include "../widgets/trades_widget.hpp"
#include "../widgets/tpo_widget.hpp"

#include "../data/market_data_processor.hpp"

namespace btquant::ui {

WindowManager::WindowManager() {
    m_orderBookWidget = new OrderBookWidget();
    m_orderBookDepthWidget = new OrderBookDepthWidget();
    m_footprintWidget = new FootprintWidget();
    m_vpvrWidget = new VPVRWidget();
    m_domWidget = new DOMWidget();
    m_tradesWidget = new TradesWidget();
    m_tpoWidget = new TPOWidget();
}

WindowManager::~WindowManager() {
    delete m_orderBookWidget;
    delete m_orderBookDepthWidget;
    delete m_footprintWidget;
    delete m_vpvrWidget;
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

void WindowManager::beginFrame() {}
void WindowManager::endFrame() {}

void WindowManager::setMarketData(::btquant::MarketDataProcessor* data) {
    if (m_orderBookWidget) m_orderBookWidget->setMarketData(data);
    if (m_orderBookDepthWidget) m_orderBookDepthWidget->setMarketData(data);
    if (m_footprintWidget) m_footprintWidget->setMarketData(data);
    if (m_vpvrWidget) m_vpvrWidget->setMarketData(data);
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
            ImGui::MenuItem("Order Book", nullptr, &showOrderBook);
            ImGui::MenuItem("DOM", nullptr, &showDOM);
            ImGui::MenuItem("Trades", nullptr, &showTrades);
            ImGui::MenuItem("TPO", nullptr, &showTPO);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("About", nullptr, nullptr);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

} // namespace btquant::ui
