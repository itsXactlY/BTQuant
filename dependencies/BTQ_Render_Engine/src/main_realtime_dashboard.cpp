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
  int frame_count = 0;
  while (!dashboard->should_close()) {
    if (frame_count++ < 5)
      std::cout << "[Main] Frame " << frame_count << " start" << std::endl;

    // 1. Event Handling (X11 & Internal)
    dashboard->handle_events();
    if (frame_count < 5)
      std::cout << "[Main] handle_events done" << std::endl;

    // 2. High-frequency polling of market data updates
    bridge->poll();
    if (frame_count < 5)
      std::cout << "[Main] poll done" << std::endl;

    // 3. Immediate Frame Composition
    dashboard->render_frame();
    if (frame_count < 5)
      std::cout << "[Main] render_frame done" << std::endl;
  }

  // 4. Shutdown
  std::cout << "[Main] Shutting down Terminal." << std::endl;
  dashboard->shutdown();
  bridge->stop();

  return 0;
}