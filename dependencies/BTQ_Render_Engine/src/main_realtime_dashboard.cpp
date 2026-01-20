#include "hotspine_data_bridge.hpp"
#include "implot.h"
#include "market_data_processor.hpp"
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
  auto bridge =
      std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  if (!bridge->start()) {
    std::cerr << "[Main] Critical: Failed to start Data Bridge." << std::endl;
    return -1;
  }

  // 2. Market Data Processor
  auto processor =
      std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
  bridge->setMarketDataProcessor(processor);

  // 3. Rendering Layer Initialization
  BTQuant::VulkanDashboardConfig config{};
  auto dashboard = std::make_unique<BTQuant::VulkanDashboard>(
      1920, 1080, bridge, processor, config);

  dashboard->initialize();

  // 2b. ImPlot Context (Must be after ImGui initialization in
  // dashboard->initialize)
  ImPlot::CreateContext();

  // 3. Execution Loop
  while (!dashboard->should_close()) {
    dashboard->handle_events();
    bridge->poll();
    dashboard->render_frame();
  }

  // 4. Graceful Shutdown
  std::cout << "[Main] Terminal shutdown initiated." << std::endl;
  dashboard->shutdown();
  bridge->stop();

  return 0;
}
