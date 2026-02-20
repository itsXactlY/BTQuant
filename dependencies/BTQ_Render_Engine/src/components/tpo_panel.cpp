#include "components/tpo_panel.hpp"

#include <imgui.h>

namespace BTQuant {

TpoPanel::TpoPanel(const PanelConfig& config) : PanelBase(config) {}

void TpoPanel::update(float /*dt*/) {}

void TpoPanel::render_content() {
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("TPOGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for TPO.");
  }
}

}  // namespace BTQuant
