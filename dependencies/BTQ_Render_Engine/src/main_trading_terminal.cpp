/**
 * BTQuant Trading Terminal - Main Entry Point
 *
 * Professional DOM trading terminal with real-time market data visualization
 * Built on BTQ_Render_Engine with Vulkan backend
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <execution>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Vulkan headers
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

// ImGui headers
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

// GLM for math
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Project headers
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "performance_monitor.hpp"
#include "symbol_manager.hpp"
#include "vulkan_base_types.hpp"
#include "vulkan_dashboard_advanced.hpp"

// Component headers
#include "components/interaction_manager.hpp"
#include "components/panel_base.hpp"
#include "components/panel_manager.hpp"
#include "components/theme_manager.hpp"

// Trading headers
#include "analytics/trading_analytics.hpp" // Correct path
#include "trading/order_manager.hpp"
#include "trading/position_manager.hpp"
#include "trading/risk_assessment.hpp"

// Config headers
#include "dashboard_config.hpp"

// System headers
#include "system/system_optimizer.hpp"

using namespace BTQuant;
using namespace BTQuant::RenderEngine;

// ============================================================================
// Application State
// ============================================================================

struct ApplicationState {
  std::atomic<bool> running{true};
  std::atomic<bool> paused{false};
  std::atomic<uint64_t> frame_count{0};

  // Performance metrics
  double current_fps{0.0};
  double frame_time_ms{0.0};
  double cpu_usage{0.0};
  double gpu_usage{0.0};

  // Market data
  std::string current_symbol{"BTCUSDT"};
  std::string current_timeframe{"15m"};

  // UI state
  bool show_demo_window{false};
  bool show_performance_overlay{true};
  bool show_debug_info{false};

  // Theme
  std::string current_theme{"dark"};
};

// ============================================================================
// Global State
// ============================================================================

static ApplicationState g_app_state;
static std::unique_ptr<BTQuant::VulkanDashboard> g_dashboard;
static std::unique_ptr<BTQuant::PanelManager> g_panel_manager;

// Core Systems
static std::shared_ptr<BTQuant::HotSpineDataBridge> g_data_bridge;
static std::shared_ptr<BTQuant::RenderEngine::MarketDataProcessor>
    g_market_processor;
static std::unique_ptr<BTQuant::RenderEngine::SymbolManager> g_symbol_manager;
static std::unique_ptr<BTQuant::System::SystemOptimizer> g_system_optimizer;

// Trading Systems
static std::shared_ptr<BTQuant::OrderManager> g_order_manager;
static std::shared_ptr<BTQuant::PositionManager> g_position_manager;
static std::shared_ptr<BTQuant::RiskAssessment> g_risk_assessment;
static std::unique_ptr<BTQuant::TradingAnalytics> g_analytics;

// Microstructure Renderer (optional/stub for now if not fully integrated in
// main)
static BTQuant::RenderEngine::MarketMicrostructureRenderer *g_micro_renderer =
    nullptr;

// ============================================================================
// Performance Monitoring
// ============================================================================

void update_performance_metrics() {
  // Rely on g_performance_monitor for calculations
  g_app_state.current_fps = g_performance_monitor.get_fps();
  g_app_state.frame_time_ms = g_performance_monitor.get_frame_time_ms();
}

// ============================================================================
// UI Rendering
// ============================================================================

void render_performance_overlay() {
  if (!g_app_state.show_performance_overlay) {
    return;
  }

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Performance", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize)) {

    ImGui::Text("FPS: %.1f", g_app_state.current_fps);
    ImGui::Text("Frame Time: %.2f ms", g_app_state.frame_time_ms);
    ImGui::Text("Frame Count: %lu", g_app_state.frame_count.load());

    ImGui::Separator();

    auto metrics = g_performance_monitor.get_metrics();
    for (const auto &m : metrics) {
      ImGui::Text("%s: %.2f %s", m.name.c_str(), m.value, m.unit.c_str());
    }

    ImGui::Separator();

    ImGui::Text("Symbol: %s", g_app_state.current_symbol.c_str());
    ImGui::Text("Timeframe: %s", g_app_state.current_timeframe.c_str());

    if (g_market_processor) {
      auto stats = g_market_processor->getPerformanceMetrics();
      ImGui::Text("Trades/sec: %.0f", stats.trades_per_second);
      ImGui::Text("Orderbook Updates/sec: %.0f", stats.orderbooks_per_second);
    }
  }

  ImGui::End();
}

void render_main_menu() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Save Layout")) {
        if (g_panel_manager)
          g_panel_manager->save_layout("default_layout.json");
      }
      if (ImGui::MenuItem("Load Layout")) {
        if (g_panel_manager)
          g_panel_manager->load_layout("default_layout.json");
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit")) {
        g_app_state.running.store(false);
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Performance Overlay", nullptr,
                          g_app_state.show_performance_overlay)) {
        g_app_state.show_performance_overlay =
            !g_app_state.show_performance_overlay;
      }
      if (ImGui::MenuItem("Auto Arrange Panels")) {
        if (g_panel_manager)
          g_panel_manager->auto_arrange_panels();
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
        ImGui::OpenPopup("About");
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  // About popup
  if (ImGui::BeginPopupModal("About", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("BTQuant Trading Terminal");
    ImGui::Separator();
    ImGui::Text("Version: 1.0.0");
    ImGui::Text("Built on BTQ_Render_Engine");
    ImGui::Separator();
    ImGui::Text("© 2026 BTQuant");
    if (ImGui::Button("Close")) {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
}

void render_dockspace() {
  // Docking code disabled due to missing headers
  render_main_menu();
}

// ============================================================================
// Main Render Loop
// ============================================================================

void render_frame() {
  if (!g_dashboard)
    return;

  auto *core = g_dashboard->get_vulkan_core();
  if (!core)
    return;

  g_performance_monitor.start_frame();
  update_performance_metrics();

  // Prepare Vulkan Frame
  uint32_t imageIndex;
  VkResult result = core->PrepareFrame(imageIndex);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    // Swapchain rebuild handled by core/dashboard usually, or we need to handle
    // resize For now, return and let next frame handle it if core did rebuild
    // internally
    return;
  } else if (result != VK_SUCCESS) {
    return;
  }

  // Start New ImGui Frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // ---------------------------------------------------------
  // RENDER UI CONTENT
  // ---------------------------------------------------------
  render_dockspace();
  render_performance_overlay();

  if (g_panel_manager) {
    g_panel_manager->update(1.0f / 60.0f);
    g_panel_manager->render();
  }

  // ---------------------------------------------------------
  // FINALIZE IMGUI
  // ---------------------------------------------------------
  ImGui::Render();

  // ---------------------------------------------------------
  // SUBMIT COMMAND BUFFER
  // ---------------------------------------------------------
  core->RecordCommandBuffer(imageIndex, ImGui::GetDrawData(),
                            [](VkCommandBuffer cmd) {
                              // Optional: Insert custom non-ImGui graphics
                              // encoding here e.g.
                              // g_micro_renderer->executeGraphics(cmd);
                            });

  // ---------------------------------------------------------
  // PRESENT
  // ---------------------------------------------------------
  core->PresentFrame(imageIndex);

  g_performance_monitor.end_frame();
  g_app_state.frame_count.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================================
// Initialization
// ============================================================================

bool initialize_application() {
  std::cout << "Initializing BTQuant Trading Terminal..." << std::endl;

  // Initialize Singletons
  InteractionManager::getInstance(); // just to init

  // Initialize symbol manager
  g_symbol_manager = std::make_unique<BTQuant::RenderEngine::SymbolManager>();
  g_symbol_manager->initialize("/dev/shm/btquant_symbols.json");
  std::cout << "✓ Symbol manager initialized" << std::endl;

  // Initialize market data processor (shared)
  g_market_processor =
      std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
  std::cout << "✓ Market data processor initialized" << std::endl;

  // Initialize data bridge (takes shared processor)
  g_data_bridge =
      std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  g_data_bridge->setMarketDataProcessor(g_market_processor);

  if (auto res = g_data_bridge->start(); !res) {
    std::cerr << "✗ Failed to initialize HotSpine data bridge: " << res.error()
              << std::endl;
    return false;
  }
  std::cout << "✓ HotSpine data bridge initialized" << std::endl;

  // Initialize Trading Systems
  g_order_manager = std::make_shared<BTQuant::OrderManager>();
  g_position_manager = std::make_shared<BTQuant::PositionManager>();
  g_risk_assessment = std::make_shared<BTQuant::RiskAssessment>();
  g_analytics = std::make_unique<BTQuant::TradingAnalytics>();

  // Initialize system optimizer
  g_system_optimizer = std::make_unique<BTQuant::System::SystemOptimizer>();
  g_system_optimizer->optimize();

  // Initialize dashboard (Vulkan)
  VulkanDashboardConfig dashboard_config;
  dashboard_config.enable_validation_layers = false;

  g_dashboard = std::make_unique<BTQuant::VulkanDashboard>(
      1920, 1080, g_data_bridge, g_market_processor, dashboard_config);

  if (auto res = g_dashboard->initialize(); !res) {
    std::cerr << "✗ Failed to initialize Vulkan dashboard: " << res.error()
              << std::endl;
    return false;
  }
  std::cout << "✓ Vulkan dashboard initialized" << std::endl;

  // Initialize ThemeManager AFTER dashboard (ImGui context created)
  ThemeManager::getInstance().initialize();
  ThemeManager::getInstance().applyTheme(ThemeType::DarkNeon);
  std::cout << "✓ Theme & Interaction managers initialized" << std::endl;

  // Initialize panel manager
  g_panel_manager = std::make_unique<BTQuant::PanelManager>(
      g_data_bridge, g_market_processor, g_order_manager, g_position_manager,
      g_risk_assessment, g_micro_renderer);
  g_panel_manager->initialize();

  // Add default panels
  g_panel_manager->add_panel(PanelType::CHART, "Chart", 0, 0, 2, 2);
  g_panel_manager->add_panel(PanelType::HEATMAP, "DOM Surface", 2, 0, 1,
                             2); // Assuming HEATMAP type for DOM logic
  g_panel_manager->add_panel(PanelType::ORDERBOOK, "Orderbook", 3, 0, 1, 2);
  g_panel_manager->add_panel(PanelType::TAPE, "Tape", 4, 0, 1, 2);
  g_panel_manager->add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 0, 2,
                             1, 1);
  g_panel_manager->add_panel(PanelType::STATUS_BAR, "Status", 0, 3, 5, 1);

  std::cout << "\n✓ Application initialized successfully!" << std::endl;
  return true;
}

// ============================================================================
// Cleanup
// ============================================================================

void cleanup_application() {
  std::cout << "\nCleaning up..." << std::endl;

  if (g_panel_manager)
    g_panel_manager->save_layout("default_layout.json");

  // Reset in reverse dependency order
  g_panel_manager.reset();

  if (g_dashboard)
    g_dashboard->shutdown();
  g_dashboard.reset();

  g_system_optimizer.reset();
  g_analytics.reset();
  g_risk_assessment.reset();
  g_position_manager.reset();
  g_order_manager.reset();

  if (g_data_bridge)
    g_data_bridge->stop();
  g_data_bridge.reset();

  g_market_processor.reset();
  g_symbol_manager.reset();

  std::cout << "✓ Cleanup complete" << std::endl;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::cout << "========================================" << std::endl;
  std::cout << "  BTQuant Trading Terminal v1.0.0" << std::endl;
  std::cout << "========================================" << std::endl;

  // Initialize application
  if (!initialize_application()) {
    std::cerr << "\n✗ Failed to initialize application" << std::endl;
    return 1;
  }

  std::cout << "\nStarting main loop..." << std::endl;

  // Main render loop - sync bridge here as well to ensure data flow
  while (g_app_state.running.load()) {
    // Poll events
    if (g_dashboard) {
      g_dashboard->handle_events();
      if (g_dashboard->should_close()) {
        g_app_state.running.store(false);
        break;
      }

      // Sync data bridge! (Was likely missing in previous loop)
      if (g_data_bridge) {
        g_data_bridge->sync();
      }

      // Render frame
      render_frame();

      // Limit FPS?
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else {
      break;
    }
  }

  // Cleanup
  cleanup_application();

  std::cout << "\nShutdown complete. Goodbye!" << std::endl;
  return 0;
}
