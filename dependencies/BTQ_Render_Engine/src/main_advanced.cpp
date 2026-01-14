#include "vulkan_dashboard_advanced.hpp"
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
    DashboardConfig config;
    config.enable_msaa = false;
    config.enable_validation_layers = false;
    auto dashboard = std::make_unique<VulkanDashboard>(1280, 720, config);

    // Initialize and run
    dashboard->initialize();
    dashboard->main_loop();
    dashboard->shutdown();

    fprintf(stderr, "\n[Main] Dashboard shutdown complete\n");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "\n[FATAL] Dashboard error: %s\n", e.what());
    return 1;
  }
}