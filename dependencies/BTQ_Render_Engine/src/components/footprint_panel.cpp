#include "components/footprint_panel.hpp"
#include "imgui.h"

namespace BTQuant {

FootprintPanel::FootprintPanel(
    const PanelConfig &config,
    RenderEngine::MarketMicrostructureRenderer *renderer)
    : PanelBase(config), renderer_(renderer) {}

void FootprintPanel::update(float dt) {
  // Update logic if needed
}

void FootprintPanel::render() {
  begin_panel_window();

  if (!renderer_) {
    ImGui::Text("Renderer not available");
    end_panel_window();
    return;
  }

  auto stats = renderer_->getStats();
  ImGui::Text("Footprint Chart (Vulkan Native)");
  ImGui::Text("Clusters: %u", stats.footprintCellsRendered);
  ImGui::Text("Avg Frame Time: %.3f ms", stats.averageFrameTimeMs);

  auto clusters = renderer_->getFootprintClusters();
  ImDrawList *drawList = ImGui::GetWindowDrawList();
  ImVec2 winPos = ImGui::GetWindowPos();
  ImVec2 winSize = ImGui::GetWindowSize();

  // Find price range for scaling
  float minPrice = 1e9f, maxPrice = -1e9f;
  for (const auto &c : clusters) {
    minPrice = std::min(minPrice, c.centerY);
    maxPrice = std::max(maxPrice, c.centerY);
  }
  float priceRange = (maxPrice - minPrice) > 0 ? (maxPrice - minPrice) : 10.0f;

  for (const auto &c : clusters) {
    // Map normalized time [0, 1] to width
    float x = winPos.x + c.centerX * winSize.x;
    // Map price to height
    float y =
        winPos.y + (1.0f - (c.centerY - minPrice) / priceRange) * winSize.y;

    char label[64];
    std::snprintf(label, sizeof(label), "%u | %u", c.bidVolume, c.askVolume);

    // Draw background for readability? (Optional)
    drawList->AddText(ImVec2(x + 5, y - 10), IM_COL32(255, 255, 255, 200),
                      label);
  }

  // renderer_->setFootprintViewport(winPos.x, winPos.y, winSize.x, winSize.y);

  end_panel_window();
}

} // namespace BTQuant
