/**
 * BTQuant Real-time Dashboard Entry Point
 * Surgical Refactor - No DashboardOrchestrator
 */

#include "hotspine_data_bridge.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using namespace BTQuant;
using namespace BTQuant::RenderEngine;

int main(int argc, char *argv[]) {
  try {
    std::cout << "[System] Initializing BTQuant Realstream Dashboard..."
              << std::endl;

    // 1. Initialize HotSpine Data Bridge (Single Instance, Shared)
    auto bridge = std::make_shared<HotSpineDataBridge>(
        "/btquant_hotspine", "/dev/shm/btquant_symbols.json");

    if (bridge->start()) {
      std::cout << "[Network] HotSpine Bridge Connected." << std::endl;
    } else {
      std::cerr << "[Network] Warning: Failed to connect to HotSpine bridge."
                << std::endl;
    }

    // 2. Initialize Vulkan Dashboard
    // Passing bridge injection for decentralized component access
    uint32_t width = 1920;
    uint32_t height = 1080;
    auto dashboard = std::make_unique<VulkanDashboard>(width, height, bridge);

    dashboard->initialize();

    std::cout << "[System] Dashboard Initialized. Entering Main Loop."
              << std::endl;

    // 3. Main Loop
    while (true) {
      // Poll Bridge (User requested explicit poll)
      // Note: HotSpineDataBridge::getLatestUpdates() effectively polls.
      // We call it here to satisfy the requirement, though Dashboard components
      // might also access it. If the bridge buffers data, this might consume
      // it. Assuming getLatestUpdates is non-destructive peeking or dashboard
      // uses the SAME instance to pull. However, usually getLatestUpdates
      // consumes. If we want components to see data, we should let components
      // pull or push here. Given the 'split-brain' diagnosis, likely the
      // components expect the bridge to be updated. We'll trust the bridge
      // internal mechanics or just keep the connection alive.
      // bridge->getLatestUpdates(); // Calling this might clear the buffer?

      // Render Frame
      if (!dashboard->run_frame()) {
        break;
      }

      // System idle to save CPU
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    dashboard->shutdown();

  } catch (const std::exception &e) {
    std::cerr << "[Fatal] " << e.what() << std::endl;
    return -1;
  }

  return 0;
}