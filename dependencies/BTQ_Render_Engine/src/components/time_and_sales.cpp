#include "components/time_and_sales.hpp"

#include <imgui.h>

namespace BTQuant {

TimeAndSalesPanel::TimeAndSalesPanel(const PanelConfig& config,
                                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(processor) {}

TimeAndSalesPanel::~TimeAndSalesPanel() = default;

void TimeAndSalesPanel::render_content() {
  ImVec2 canvas_size = ImGui::GetContentRegionAvail();
  if (canvas_size.x > 50.0f && canvas_size.y > 50.0f) {
    ImGui::InvisibleButton("TimeSalesGPUCanvas", canvas_size);
    ImGui::Text("GPU Hook Ready for Time & Sales.");
  }
}

}  // namespace BTQuant
