/**
 * BTQuant Real-time Dashboard (Unified Architecture)
 *
 * This is the refined entry point that eliminates DashboardOrchestrator.
 * It directly coordinates the HotSpineDataBridge and VulkanDashboard.
 */

#include "../include/hotspine_data_bridge.hpp"
#include "../include/vulkan_dashboard_advanced.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

int main(int, char **) {
  std::cout << "[Main] Starting BTQuant Realtime Dashboard (Unified)..."
            << std::endl;

  // 1. Init Data Bridge
  std::cout << "[Main] Initializing HotSpine Data Bridge..." << std::endl;
  auto bridge = std::make_shared<BTQuant::RenderEngine::HotSpineDataBridge>();

  if (!bridge->start()) {
    std::cerr << "[Main] Failed to start HotSpineDataBridge!" << std::endl;
    return -1;
  }
  std::cout << "[Main] Data Bridge connected." << std::endl;

  // 2. Configure Dashboard
  BTQuant::VulkanDashboardConfig config;
  config.enable_validation_layers = true;
  config.enable_msaa = true;
  config.msaa_samples = VK_SAMPLE_COUNT_4_BIT;
  // Removed start_maximized as it is not in the config struct

  // 3. Create Dashboard (this implicitly creates VulkanCore and Window)
  std::cout << "[Main] Creating VulkanDashboard..." << std::endl;
  auto dashboard =
      std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, bridge, config);

  dashboard->initialize();

  // Apply default layout
  dashboard->set_active_symbol("BTC-USDT");

  // 4. Main Application Loop
  std::cout << "[Main] Entering Render Loop..." << std::endl;

  while (true) {
    // Poll Bridge Data & Check Health
    if (!bridge->isConnected()) {
      // Optional: Try reconnect or log warning
    }

    // Synchronize Market Data
    dashboard->synchronize_market_data();

    // Render Frame
    if (!dashboard->run_frame()) {
      break; // Window closed
    }
  }

  // Cleaning up
  std::cout << "[Main] Shutting down..." << std::endl;
  dashboard->shutdown();
  bridge->stop();

  return 0;
}