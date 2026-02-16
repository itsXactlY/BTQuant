#include <iostream>
#include <memory>

#include "components/panel_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "ui/unified_theme_system.hpp"
#include "vulkan_dashboard_advanced.hpp"

using namespace BTQuant;

int main(int argc, char** argv) {
  try {
    std::cout << "[INIT] Booting BTQ Render Engine..." << std::endl;

    // 1. Create the HotSpine data bridge
    // DEPRECATED - Legacy hotspine
    auto bridge = std::make_shared<HotSpineDataBridge>();

    // 2. The Data Spine (Destroyed Last)
    auto data_processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    // 3. The Workspace (creates PanelManager internally)
    auto workspace = std::make_shared<QuantWorkspaceComponent>(bridge, data_processor);

    // 4. Force The BTQ/Quantower Layout Default
    if (auto* pm = workspace->getPanelManager()) {
      pm->apply_layout_preset(LayoutPreset::MODERN_TRADING);
    }

    // 5. Apply Deep Void Theme
    BTQuant::UI::UnifiedThemeManager::getInstance().initialize();
    BTQuant::UI::UnifiedThemeManager::getInstance().set_current_theme("dark");

    // 6. The Renderer (Destroyed First)
    VulkanDashboardConfig config;
    config.target_fps = 144;
    auto dashboard = std::make_unique<VulkanDashboard>(1920, 1080, bridge, data_processor, config);

    auto init_result = dashboard->initialize();
    if (!init_result) {
      std::cerr << "[FATAL] Dashboard init failed: " << init_result.error() << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << "[INIT] Entering Main 144Hz Render Loop..." << std::endl;
    while (!dashboard->should_close()) {
      dashboard->handle_events();
      dashboard->render_frame();
    }

    // 7. Clean Teardown
    dashboard->shutdown();
    std::cout << "[SHUTDOWN] Terminated successfully." << std::endl;
    return EXIT_SUCCESS;

  } catch (const std::exception& e) {
    std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}