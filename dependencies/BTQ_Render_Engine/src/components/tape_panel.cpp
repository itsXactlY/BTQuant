#include "components/tape_panel.hpp"

#include <imgui.h>

namespace BTQuant {

TapePanel::TapePanel(const PanelConfig& config,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(processor) {}

void TapePanel::render_content() {
  begin_panel_window();

  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("TapeGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for Tape.");
  }

  end_panel_window();
}

}  // namespace BTQuant
