#include "components/dom_surface_panel.hpp"

#include <imgui.h>

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor) {}

DomSurfacePanel::~DomSurfacePanel() {}

void DomSurfacePanel::setSymbol(uint32_t symbol_id) { current_symbol_id_ = symbol_id; }

void DomSurfacePanel::render_content() {
  if (current_symbol_id_ == 0) {
    ImGui::Text("No Data / Select Symbol");
    return;
  }

  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("DOMGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for DOM Surface.");
  }
}

void DomSurfacePanel::render_panel_header() { PanelBase::render_panel_header(); }

void DomSurfacePanel::updateLivePrice(double price) {
  live_price_.store(price, std::memory_order_release);
}

void DomSurfacePanel::onDataUpdate(uint32_t, RenderEngine::NotificationType) {}

}  // namespace BTQuant
