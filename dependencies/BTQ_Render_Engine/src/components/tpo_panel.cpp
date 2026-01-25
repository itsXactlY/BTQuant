#include "../../include/components/tpo_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <format>

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
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Renderer unavailable");
    end_panel_window();
    return;
  }

  // Toolbar
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  static bool show_text = true;
  ImGui::Checkbox("Show Text", &show_text);

  // Get Data
  auto clusters = renderer_->getFootprintClusters();

  if (ImPlot::BeginPlot("##FootprintChart", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    const auto &colors = ThemeManager::getInstance().getColors();
    auto *draw_list = ImPlot::GetPlotDrawList();

    for (const auto &cluster : clusters) {
      // Determine color based on Delta (Ask - Bid)
      int delta = (int)cluster.askVolume - (int)cluster.bidVolume;
      ImU32 color;
      if (delta > 0) {
        // Buying pressure -> Green gradient based on intensity
        float intensity = std::clamp((float)delta / 1000.0f, 0.2f, 1.0f);
        color = ImGui::GetColorU32(ImVec4(colors.accent_green.x,
                                          colors.accent_green.y,
                                          colors.accent_green.z, intensity));
      } else {
        // Selling pressure -> Red
        float intensity = std::clamp((float)(-delta) / 1000.0f, 0.2f, 1.0f);
        color =
            ImGui::GetColorU32(ImVec4(colors.accent_red.x, colors.accent_red.y,
                                      colors.accent_red.z, intensity));
      }

      // Draw Box
      // CenterX is time. Width is time duration ? Or visual width?
      // Assuming width/height are in Plot Coordinates
      double x1 = cluster.centerX - cluster.width * 0.45;
      double x2 = cluster.centerX + cluster.width * 0.45;
      double y1 = cluster.centerY - cluster.height * 0.45;
      double y2 = cluster.centerY + cluster.height * 0.45;

      ImVec2 p1 = ImPlot::PlotToPixels(x1, y1);
      ImVec2 p2 = ImPlot::PlotToPixels(x2, y2);

      draw_list->AddRectFilled(p1, p2, color);

      // Draw Text if zoomed in enough
      if (show_text && (p2.y - p1.y) > 15) { // Only if cell is tall enough
        std::string label = std::format("{}", delta);
        // Center text
        ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
        ImVec2 text_pos = ImVec2((p1.x + p2.x - text_size.x) * 0.5f,
                                 (p1.y + p2.y - text_size.y) * 0.5f);
        draw_list->AddText(text_pos, IM_COL32_WHITE, label.c_str());
      }
    }

    ImPlot::EndPlot();
  }

  // Debug Stats Overlay
  auto stats = renderer_->getStats();
  ImGui::SetCursorPos(ImVec2(10, 40));
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1));
  ImGui::Text("Clusters: %zu | Updates: %u", clusters.size(),
              stats.tradeUpdates);
  ImGui::PopStyleColor();

  end_panel_window();
}

} // namespace BTQuant
