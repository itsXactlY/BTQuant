#include "hotspine_data_bridge.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include <iostream>
#include <memory>

/**
 * BTQuant Terminal Entry Point
 *
 * Performance Architecture:
 * - SHM Data Ingestion: HotSpineDataBridge (Polled Frequency)
 * - Renderer: Vulkan with Dear ImGui/ImPlot (Uncapped Framerate)
 */
int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::cout << "[Main] BTQuant Multi-Asset Terminal starting..." << std::endl;

  // 1. Data Layer Initialization
  auto bridge =
      std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  if (!bridge->start()) {
    std::cerr << "[Main] Critical: Failed to attach to SHM Segment."
              << std::endl;
    return -1;
  }

  // 2. Rendering Layer Initialization
  BTQuant::VulkanDashboardConfig config{};
  auto dashboard =
      std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, bridge, config);

  // ImPlot Context MUST exist for components
  dashboard->initialize();

  // 3. Application Execution Loop
  while (!dashboard->should_close()) {
    // 1. Event Handling (X11 & Internal)
    dashboard->handle_events();

    // 2. High-frequency polling of market data updates
    bridge->poll();

    // 3. Immediate Frame Composition
    dashboard->render_frame();
  }

  // 4. Shutdown
  std::cout << "[Main] Shutting down Terminal." << std::endl;
  dashboard->shutdown();
  bridge->stop();

  return 0;
}