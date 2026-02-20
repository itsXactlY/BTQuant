#include "components/histogram_panel.hpp"

#include <imgui.h>

namespace BTQuant {

HistogramPanel::HistogramPanel(const PanelConfig& config) : PanelBase(config) {}

void HistogramPanel::render_content() {
  begin_panel_window();

  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("HistogramGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for Histogram.");
  }

  end_panel_window();
}

}  // namespace BTQuant
