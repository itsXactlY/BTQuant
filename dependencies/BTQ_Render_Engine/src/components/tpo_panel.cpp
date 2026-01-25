#include "../../include/components/tpo_panel.hpp"
#include "imgui.h"

namespace BTQuant {

TpoPanel::TpoPanel(const PanelConfig &config,
                   RenderEngine::MarketMicrostructureRenderer *renderer)
    : PanelBase(config), renderer_(renderer) {}

void TpoPanel::update(float dt) {
  // Update logic if needed
}

void TpoPanel::render() {
  begin_panel_window();

  if (!renderer_) {
    ImGui::Text("Renderer not available");
    end_panel_window();
    return;
  }

  auto stats = renderer_->getStats();
  ImGui::Text("TPO Profile (Atomic Compute)");
  ImGui::Text("Trade Updates: %u", stats.tradeUpdates);
  ImGui::Text("Avg Frame Time: %.3f ms", stats.averageFrameTimeMs);

  end_panel_window();
}

} // namespace BTQuant
