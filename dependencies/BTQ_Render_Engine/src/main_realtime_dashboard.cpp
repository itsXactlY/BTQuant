/**
 * BTQuant Advanced Vulkan Dashboard - Entry Point
 */

#include "hotspine_data_bridge.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include <iostream>
#include <memory>

using namespace BTQuant;
using namespace BTQuant::RenderEngine;

// Local DisplayConfig for this context
struct DisplayConfig {
  uint32_t window_width;
  uint32_t window_height;
  std::string title;
  bool vsync_enabled;
};

int main(int argc, char *argv[]) {
  try {
    // 1. Initialize HotSpine Data Bridge (Single Instance)
    // Using shared_ptr to pass to Dashboard and Components
    std::cout << "[Main] Initializing HotSpine Data Bridge..." << std::endl;
    auto bridge = std::make_shared<HotSpineDataBridge>(
        "/btquant_hotspine", "/dev/shm/btquant_symbols.json");

    // Start the bridge connection immediately
    if (bridge->start()) {
      std::cout << "[Main] HotSpine Bridge Connected Successfully."
                << std::endl;
    } else {
      std::cerr << "[Main] Warning: HotSpine Bridge failed to connect or start "
                   "initially."
                << std::endl;
    }

    // 2. Display Configuration
    DisplayConfig display_config;
    display_config.window_width = 1920;
    display_config.window_height = 1080;
    display_config.title = "BTQuant HFT Dashboard";
    display_config.vsync_enabled = true;

    // 3. Initialize Vulkan Dashboard
    // Passing the bridge instance to centralize data management
    std::cout << "[Main] Initializing Vulkan Dashboard..." << std::endl;
    auto vulkan_dashboard = std::make_unique<VulkanDashboard>(
        display_config.window_width, display_config.window_height, bridge);

    vulkan_dashboard->initialize();

    std::cout << "[Main] Dashboard Initialized. Entering Main Loop..."
              << std::endl;

    // 4. Run Main Loop
    // VulkanDashboard now handles the loop, rendering, and component updates
    // internally
    vulkan_dashboard->main_loop();

  } catch (const std::exception &e) {
    std::cerr << "[Main] Fatal Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}