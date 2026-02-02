#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "components/realtime_dashboard_component.hpp"
#include "hotspine_data_bridge.hpp"
#include "implot.h"
#include "market_data_processor.hpp"
#include "performance_monitor.hpp"
#include "rendering/frame_pacer.hpp"
#include "vulkan_dashboard_advanced.hpp"

/**
 * BTQuant Real-Time Professional Terminal
 *
 * Goal: Wire up HotSpineDataBridge -> QuantWorkspaceComponent -> ImPlot ->
 * VulkanDashboard.
 */
int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  std::println("[Main] BTQuant Real-Time Terminal Starting...");

  // 1. Data Layer Initialization (Shared Ptr)
  auto bridge = std::make_shared<BTQuant::HotSpineDataBridge>("/btquant_hotspine");
  auto start_res = bridge->start();
  if (!start_res) [[unlikely]] {
    std::println(stderr, "[Main] Critical: Failed to start Data Bridge: {}", start_res.error());
    return -1;
  }

  // 2. Market Data Processor
  auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
  bridge->setMarketDataProcessor(processor);

  // 3. Rendering Layer Initialization
  BTQuant::VulkanDashboardConfig config{};
  auto dashboard =
      std::make_unique<BTQuant::VulkanDashboard>(1920, 1080, bridge, processor, config);

  if (auto init_res = dashboard->initialize(); !init_res) [[unlikely]] {
    std::println(stderr, "[Main] Critical: Failed to initialize Dashboard: {}", init_res.error());
    return -1;
  }

  // 2b. ImPlot Context (Must be after ImGui initialization in
  // dashboard->initialize)
  std::println("[Main] Creating ImPlot context...");
  ImPlot::CreateContext();
  std::println("[Main] ImPlot context created");

  // 3. Initialize Frame Pacer
  RenderEngine::FramePacer::Config pacer_config;
  pacer_config.target_fps = 144;  // Target 144 FPS for high-refresh displays
  pacer_config.enable_adaptive_sync = true;
  pacer_config.enable_frame_smoothing = true;
  pacer_config.enable_burst_reduction = true;
  RenderEngine::FramePacer frame_pacer(pacer_config);

  while (!dashboard->should_close()) {
    frame_pacer.begin_frame();

    BTQuant::g_performance_monitor.start_frame();
    dashboard->handle_events();
    bridge->sync();
    dashboard->render_frame();
    BTQuant::g_performance_monitor.end_frame();

    frame_pacer.end_frame();
    frame_pacer.wait_for_next_frame();
  }

  // 4. Graceful Shutdown
  std::println("[Main] Terminal shutdown initiated.");
  dashboard->shutdown();
  bridge->stop();

  ImPlot::DestroyContext();

  return 0;
}
