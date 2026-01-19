#include "../include/vulkan_dashboard_advanced.hpp"
#include <cstdio>
#include <cstdlib>

using namespace BTQuant;

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  fprintf(stderr,
          "\n╔════════════════════════════════════════════════════════════╗\n");
  fprintf(stderr,
          "║   BTQuant Advanced Vulkan Dashboard (Production Ready)    ║\n");
  fprintf(stderr,
          "║   Modern Vulkan | Real-time Data | Professional UI        ║\n");
  fprintf(stderr,
          "╚════════════════════════════════════════════════════════════╝\n\n");

  fprintf(stderr,
          "[Main] Initializing advanced Vulkan dashboard (1280x720)...\n");
  fprintf(
      stderr,
      "[Main] Features: MSAA, HDR, Compute Shaders, HotSpine Integration\n");
  fprintf(stderr, "[Main] Target: 60 FPS with <1ms latency\n");
  fprintf(stderr, "[Main] Close window to exit\n\n");

  try {
    // Create dashboard with default configuration
    VulkanDashboardConfig config;
    config.enable_msaa = false;
    config.enable_validation_layers = false;
    auto hotspine_bridge = std::make_shared<HotSpineDataBridge>();
    auto market_data_processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    
    // Set processor for data aggregation
    hotspine_bridge->setMarketDataProcessor(market_data_processor);
    
    auto dashboard = std::make_unique<VulkanDashboard>(1280, 720, hotspine_bridge,
                                                     market_data_processor, config);

    // Initialize and run
    dashboard->initialize();
    while (!dashboard->should_close()) {
        dashboard->handle_events();
        dashboard->render_frame();
    }
    dashboard->shutdown();

    fprintf(stderr, "\n[Main] Dashboard shutdown complete\n");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "\n[FATAL] Dashboard error: %s\n", e.what());
    return 1;
  }
}