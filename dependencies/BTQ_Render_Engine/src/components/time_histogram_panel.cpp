#include "components/time_histogram_panel.hpp"

#include <imgui.h>

namespace BTQuant {

TimeHistogramPanel::TimeHistogramPanel(const PanelConfig& config) : PanelBase(config) {}

void TimeHistogramPanel::initialize() { PanelBase::initialize(); }

void TimeHistogramPanel::render_content() {
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("HistogramGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for Time Histogram.");
  }
}

}  // namespace BTQuant
