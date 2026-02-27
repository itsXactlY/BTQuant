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

  // 1. Initialize MarketDataProcessor (The Data Core) - Created ABSOLUTELY FIRST to be destroyed LAST
  std::cout << "Initializing Data Layer..." << std::endl;
  auto market_processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();

  // 2. System Optimization & Logger
  auto system_optimizer = std::make_unique<BTQuant::System::SystemOptimizer>();
  system_optimizer->optimize();

  // 3. Data Bridge
  auto data_bridge = std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  data_bridge->setMarketDataProcessor(market_processor);

  if (auto res = data_bridge->start(); !res) {
    std::cerr << "✗ Failed to initialize HotSpine data bridge: " << res.error() << std::endl;
    return 1;
  }
  std::cout << "✓ Data Pipeline active" << std::endl;

  // 4. Initialize Dashboard (Vulkan + ImGui) - Creates Workspace + PanelManager + Trading Systems internally
  VulkanDashboardConfig dashboard_config;
  dashboard_config.enable_validation_layers = false;

  auto dashboard = std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, data_bridge, market_processor, dashboard_config);

  if (auto res = dashboard->initialize(); !res) {
    std::cerr << "✗ Failed to initialize Vulkan dashboard: " << res.error() << std::endl;
    return 1;
  }
  std::cout << "✓ Vulkan Dashboard initialized" << std::endl;

  // 5. Configure Theme
  ThemeManager::getInstance().applyTheme(ThemeType::DarkNeon);

  // 5.1 Apply default layout preset (MODERN_TRADING)
  if (auto* workspace = dashboard->get_workspace_component()) {
    if (auto* panel_mgr = workspace->getPanelManager()) {
      panel_mgr->apply_layout_preset(LayoutPreset::MODERN_TRADING);
      std::cout << "✓ Applied MODERN_TRADING layout preset" << std::endl;
    }
  }

  // 6. Setup Custom Menu Bar
  dashboard->set_custom_menubar_callback([&dashboard]() {
    if (ImGui::BeginMenu("File")) {
      auto* workspace = dashboard->get_workspace_component();
      auto* panel_mgr = workspace ? workspace->getPanelManager() : nullptr;

      if (ImGui::MenuItem("Save Layout")) {
        if (panel_mgr) panel_mgr->save_layout("default_layout.json");
      }
      if (ImGui::MenuItem("Load Layout")) {
        if (panel_mgr) panel_mgr->load_layout("default_layout.json");
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit")) {
        std::cout << "Exit requested via menu" << std::endl;
        exit(0);
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
      auto* panel_mgr = workspace ? workspace->getPanelManager() : nullptr;

      if (ImGui::MenuItem("Auto Arrange Panels")) {
        if (panel_mgr) panel_mgr->auto_arrange_panels();
      }

      ImGui::Separator();

      // Layout Presets submenu
      if (ImGui::BeginMenu("Layout Presets")) {
        if (ImGui::MenuItem("Default", "Ctrl+1")) {
          if (panel_mgr) panel_mgr->apply_layout_preset(LayoutPreset::DEFAULT);
        }
        if (ImGui::MenuItem("Modern Trading", "Ctrl+2")) {
          if (panel_mgr) panel_mgr->apply_layout_preset(LayoutPreset::MODERN_TRADING);
        }
        if (ImGui::MenuItem("Dashboard Only", "Ctrl+3")) {
          if (panel_mgr) panel_mgr->apply_layout_preset(LayoutPreset::DASHBOARD_ONLY);
        }
        if (ImGui::MenuItem("Chart Focus", "Ctrl+4")) {
          if (panel_mgr) panel_mgr->apply_layout_preset(LayoutPreset::CHART_FOCUS);
        }
        if (ImGui::MenuItem("Risk Monitoring", "Ctrl+5")) {
          if (panel_mgr) panel_mgr->apply_layout_preset(LayoutPreset::RISK_MONITORING);
        }
        ImGui::EndMenu();
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

  while (!dashboard->should_close()) {
    auto frame_begin = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(frame_begin - last_frame_time).count();
    (void)dt;  // Suppress unused variable warning
    last_frame_time = frame_begin;

    // Data Sync - Single sync point in main loop
    data_bridge->sync();

    // Event Handling
    dashboard->handle_events();

    // Render
    dashboard->render_frame();

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
