#include "window_manager.hpp"

#include <imgui.h>

namespace btquant::ui {

WindowManager::WindowManager() = default;
WindowManager::~WindowManager() = default;

void WindowManager::initialize() {
    m_initialized = true;
}

void WindowManager::shutdown() {
    m_initialized = false;
}

void WindowManager::beginFrame() {}

void WindowManager::endFrame() {}

void WindowManager::showOrderBookWindow() {
    if (!showOrderBook) return;
    ImGui::Begin("Order Book", &showOrderBook);
    ImGui::Text("Order Book Widget");
    ImGui::End();
}

void WindowManager::showDOMWindow() {
    if (!showDOM) return;
    ImGui::Begin("DOM", &showDOM);
    ImGui::Text("Depth of Market");
    ImGui::End();
}

void WindowManager::showTradesWindow() {
    if (!showTrades) return;
    ImGui::Begin("Trades", &showTrades);
    ImGui::Text("Trades Feed");
    ImGui::End();
}

void WindowManager::showTPOWindow() {
    if (!showTPO) return;
    ImGui::Begin("TPO", &showTPO);
    ImGui::Text("Time Price Opportunity");
    ImGui::End();
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
