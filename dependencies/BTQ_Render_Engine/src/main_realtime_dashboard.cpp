#include "hotspine_data_bridge.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include <iostream>
#include <memory>

/**
 * BTQuant Real-Time Professional Terminal
 *
 * Goal: Wire up HotSpineDataBridge -> QuantWorkspaceComponent -> ImPlot ->
 * VulkanDashboard.
 */
int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  std::cout << "[Main] BTQuant Real-Time Terminal Starting..." << std::endl;

  // 1. Data Layer Initialization (Shared Ptr)
  auto bridge = std::make_shared<BTQuant::HotSpineDataBridge>("/btquant");
  if (!bridge->start()) {
    std::cerr << "[Main] Critical: Failed to start Data Bridge." << std::endl;
    return -1;
  }

  // 2. Rendering Layer Initialization
  // VulkanCore is initialized internally by VulkanDashboard
  BTQuant::VulkanDashboardConfig config{};
  auto dashboard =
      std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, bridge, config);

  // Initialize Dashboard and its internal components (including
  // QuantWorkspaceComponent)
  dashboard->initialize();

  // 3. Execution Loop
  std::cout << "[Main] Entering real-time monitoring loop..." << std::endl;
  while (!dashboard->should_close()) {
    // Poll for internal UI events/X11
    dashboard->handle_events();

    // Poll for high-frequency market data (Shared Memory or Simulation)
    bridge->poll();

    // Render Frame (ImGui + ImPlot + Vulkan)
    dashboard->render_frame();
  }

  // 4. Graceful Shutdown
  std::cout << "[Main] Terminal shutdown initiated." << std::endl;
  dashboard->shutdown();
  bridge->stop();

  return 0;
}