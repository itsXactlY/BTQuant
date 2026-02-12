/**
 * BTQuant Trading Terminal - Main Entry Point
 *
 * Professional DOM trading terminal with real-time market data visualization
 * Built on BTQ_Render_Engine with Vulkan backend
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// Project headers
#include "components/panel_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "components/theme_manager.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "performance/debug_overlay.hpp"
#include "system/system_optimizer.hpp"
#include "ui/haptic_feedback.hpp"
#include "ui/layout_manager.hpp"
#include "ui/tutorial.hpp"
#include "vulkan_dashboard_advanced.hpp"

using namespace BTQuant;
using namespace BTQuant::RenderEngine;

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::cout << "========================================" << std::endl;
  std::cout << "  BTQuant Trading Terminal v1.0.0" << std::endl;
  std::cout << "========================================" << std::endl;

  // 1. System Optimization
  std::cout << "[Step 1] Initializing System Optimizer..." << std::endl;
  auto system_optimizer = std::make_unique<BTQuant::System::SystemOptimizer>();
  system_optimizer->optimize();
  std::cout << "[Step 1] System Optimizer initialized." << std::endl;

  // 1.5 Initialize haptic feedback system
  std::cout << "[Step 1.5] Initializing Haptic Feedback..." << std::endl;
  BTQuant::UI::HapticFeedback::getInstance().initialize();
  std::cout << "[Step 1.5] Haptic Feedback initialized." << std::endl;

  // 2. Data Layer Initialization
  std::cout << "[Step 2] Initializing Data Layer..." << std::endl;

  // Market Processor
  std::cout << "[Step 2.1] Creating MarketDataProcessor..." << std::endl;
  auto market_processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
  std::cout << "[Step 2.1] MarketDataProcessor created." << std::endl;

  // Data Bridge
  std::cout << "[Step 2.2] Creating HotSpineDataBridge..." << std::endl;
  auto data_bridge = std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  data_bridge->setMarketDataProcessor(market_processor);
  std::cout << "[Step 2.2] HotSpineDataBridge created." << std::endl;

  std::cout << "[Step 2.3] Starting Data Bridge..." << std::endl;
  if (auto res = data_bridge->start(); !res) {
    std::cerr << "✗ Failed to initialize HotSpine data bridge: " << res.error() << std::endl;
    return 1;
  }
  std::cout << "✓ Data Pipeline active" << std::endl;

  // 3. Initialize Dashboard (Vulkan + ImGui)
  std::cout << "[Step 3] Initializing Dashboard..." << std::endl;
  VulkanDashboardConfig dashboard_config;
  dashboard_config.enable_validation_layers = false;

  std::cout << "[Step 3.1] Creating VulkanDashboard instance..." << std::endl;
  auto dashboard = std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, data_bridge,
                                                              market_processor, dashboard_config);
  std::cout << "[Step 3.1] VulkanDashboard instance created." << std::endl;

  std::cout << "[Step 3.2] Calling dashboard->initialize()..." << std::endl;
  if (auto res = dashboard->initialize(); !res) {
    std::cerr << "✗ Failed to initialize Vulkan dashboard: " << res.error() << std::endl;
    return 1;
  }
  std::cout << "✓ Vulkan Dashboard initialized" << std::endl;

  // 4. Configure Theme
  std::cout << "[Step 4] Configuring Theme..." << std::endl;
  ThemeManager::getInstance().applyTheme(ThemeType::DarkNeon);
  std::cout << "[Step 4] Theme configured." << std::endl;

  // 5. Configure Layout (via Workspace Component)
  // We access the internal components to set up the default trading layout
  std::cout << "[Step 5] Configuring Layout..." << std::endl;
  if (auto* workspace = dashboard->get_workspace_component()) {
    std::cout << "Configuring Default Layout..." << std::endl;

    workspace->set_layout(BTQuant::LayoutPreset::PRO_QUANT);
  }
  std::cout << "[Step 5] Layout configured." << std::endl;

  // 5.5 Setup Custom Menu Bar
  dashboard->set_custom_menubar_callback([&dashboard]() {
    if (ImGui::BeginMenu("File")) {
      auto* workspace = dashboard->get_workspace_component();
      auto* panel_mgr = workspace ? workspace->getPanelManager() : nullptr;

      if (ImGui::MenuItem("Save Layout")) {
        if (panel_mgr) panel_mgr->save_layout("default_layout.json");
      }
      if (ImGui::MenuItem("Save as Default")) {
        if (panel_mgr) panel_mgr->save_default_layout();
      }
      if (ImGui::MenuItem("Load Layout")) {
        if (panel_mgr) panel_mgr->load_layout("default_layout.json");
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit")) {
        std::cout << "Exit requested via menu" << std::endl;
        // Properly signal the window to close using GLFW
        glfwSetWindowShouldClose(dashboard->get_window(), true);  // Need to expose the window handle
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      static bool show_perf = true;
      if (ImGui::MenuItem("Performance Overlay", nullptr, &show_perf)) {
        dashboard->set_show_performance_overlay(show_perf);
      }

      if (ImGui::MenuItem("Debug Overlay (F12)")) {
        g_debug_overlay.toggle_visibility();
      }

      auto* workspace = dashboard->get_workspace_component();
      if (ImGui::MenuItem("Auto Arrange Panels")) {
        if (workspace && workspace->getPanelManager())
          workspace->getPanelManager()->auto_arrange_panels();
      }

      // Add Always on Top toggle
      if (ImGui::MenuItem("Always on Top", nullptr, dashboard->is_always_on_top())) {
        dashboard->set_always_on_top(!dashboard->is_always_on_top());
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Theme")) {
      if (ImGui::MenuItem("Toggle Theme")) {
        ThemeManager::getInstance().toggleTheme();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("About")) {
        // Open modal logic would go here, or simple log
        std::cout << "BTQuant Version 1.0.0" << std::endl;
      }
      ImGui::EndMenu();
    }
  });

  std::cout << "\nStarting main loop..." << std::endl;

  // 7. Main Loop
  // Target 144 FPS
  const auto target_frame_time = std::chrono::microseconds(6944);

  auto last_frame_time = std::chrono::steady_clock::now();
  bool tutorial_checked = false;  // Flag to check tutorial only on first frame

  while (!dashboard->should_close()) {
    auto frame_begin = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(frame_begin - last_frame_time).count();
    (void)dt;  // Suppress unused variable warning
    last_frame_time = frame_begin;

    // Data Sync - Handled by background thread
    // data_bridge->sync();

    // Event Handling
    dashboard->handle_events();

    // Render
    dashboard->render_frame();

    // Check for tutorial on first frame only (after frame is rendered to ensure ImGui context is
    // ready)
    if (!tutorial_checked) {
      BTQuant::UI::show_tutorial_if_first_run();
      tutorial_checked = true;
    }

    // Render tutorial if active
    BTQuant::UI::render_tutorial();

    // FPS Limiter
    auto frame_end = std::chrono::steady_clock::now();
    auto elapsed = frame_end - frame_begin;
    if (elapsed < target_frame_time) {
      std::this_thread::sleep_for(target_frame_time - elapsed);
    }
  }

  // 8. Cleanup
  std::cout << "\nCleaning up..." << std::endl;

  // Save layout on exit
  if (auto* ws = dashboard->get_workspace_component()) {
    if (auto* pm = ws->getPanelManager()) {
      pm->save_layout("default_layout.json");
    }
  }

  dashboard->shutdown();
  data_bridge->stop();

  std::cout << "Shutdown complete. Goodbye!" << std::endl;
  return 0;
}
