#include "CandlePipeline.h"
#include "DashboardOrchestrator.hpp"
#include "OffscreenChartRenderer.h"
#include "hotspine_data_bridge.hpp"
#include "vulkan_dashboard_advanced.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <thread>

using namespace BTQuant;
using namespace BTQuant::RenderEngine;

// Local DisplayConfig for this context
struct DisplayConfig {
  uint32_t window_width;
  uint32_t window_height;
  std::string title;
  bool vsync_enabled;
};

// Simplified Main for patched integration
int main(int argc __attribute__((unused)),
         char *argv[] __attribute__((unused))) {
  try {
    // 1. Display Configuration
    DisplayConfig display_config;
    display_config.window_width = 1920;
    display_config.window_height = 1080;
    display_config.title = "BTQuant HFT Dashboard";
    display_config.vsync_enabled = true;

    // 2. Initialize Vulkan Dashboard
    auto vulkan_dashboard = std::make_unique<VulkanDashboard>(
        display_config.window_width, display_config.window_height);

    vulkan_dashboard->initialize();
    // if (!vulkan_dashboard->initialize()) {
    //   std::cerr << "Failed to initialize Vulkan Dashboard" << std::endl;
    //   return -1;
    // }

    // 3. Initialize Orchestrator
    auto orchestrator =
        std::make_unique<DashboardOrchestrator>(vulkan_dashboard->get_core());

    // 4. Initialize & Inject Components
    // Bridge (Correct namespace: BTQuant::RenderEngine)
    auto bridge = std::make_unique<BTQuant::RenderEngine::HotSpineDataBridge>(
        "/hotspine_shm", "symbols.json");
    if (bridge->isConnected()) {
      std::cout << "Connected to HotSpine" << std::endl;
    }
    orchestrator->SetBridge(std::move(bridge));

    // Renderer
    auto renderer =
        std::make_unique<OffscreenChartRenderer>(vulkan_dashboard->get_core());
    renderer->create_resources(800, 600); // Initial size
    // Note: RenderPass dependency for pipeline
    VkRenderPass chartPass = renderer->GetRenderPass();
    orchestrator->SetRenderer(std::move(renderer));

    // Pipeline
    auto pipeline = std::make_unique<CandlePipeline>(
        vulkan_dashboard->get_core(), chartPass);
    orchestrator->SetPipeline(std::move(pipeline));

    // 5. Wire Up ImGui
    vulkan_dashboard->set_on_gui_callback([&]() { orchestrator->RenderUI(); });

    std::cout << "System Wired. Starting Main Loop..." << std::endl;

    bool running = true;
    while (running) {
      // 1. Data Pump
      orchestrator->Update();

      // 2. Offscreen Render Pass
      // Get a command buffer just for offscreen work
      VkCommandBuffer cmd =
          vulkan_dashboard->get_core()->begin_single_time_commands();
      orchestrator->Draw(cmd); // Renders chart to texture
      vulkan_dashboard->get_core()->end_single_time_commands(cmd);

      // 3. Present (ImGui + Swapchain)
      if (!vulkan_dashboard->run_frame()) {
        running = false;
      }

      // Sleep to prevent melting CPU in this simple loop
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}