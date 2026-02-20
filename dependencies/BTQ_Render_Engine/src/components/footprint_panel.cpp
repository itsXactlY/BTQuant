#include "components/footprint_panel.hpp"

#include <imgui.h>

namespace BTQuant {

FootprintPanel::FootprintPanel(const PanelConfig& config) : PanelBase(config) {}

void FootprintPanel::update(float /*dt*/) {}

void FootprintPanel::render_content() {
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("FootprintGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for Footprint.");
  }
}

}  // namespace BTQuant
