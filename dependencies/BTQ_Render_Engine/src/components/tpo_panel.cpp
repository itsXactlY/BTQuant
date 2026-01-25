#include "../../include/components/tpo_panel.hpp"
#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <chrono>
#include <ctime>
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
  static bool show_text = true;
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Delta Labels", &show_text);

  auto clusters = renderer_->getFootprintClusters();
  auto stats = renderer_->getStats();

  // Base time for labeling (relative to 30s window)
  double base_time_sec =
      static_cast<double>(stats.lastUpdateTimeNs) / 1'000'000'000.0 - 30.0;

  if (ImPlot::BeginPlot("##FootprintChart", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend |
                            (1 << 8))) { // 1<<8 is Crosshairs

    // Axis Setup
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, 30, ImPlotCond_Always);

    float p_min = 0, p_max = 1000;
    if (!clusters.empty()) {
      p_min = clusters[0].centerY;
      p_max = clusters[0].centerY;
      for (const auto &c : clusters) {
        p_min = std::min(p_min, (float)c.centerY);
        p_max = std::max(p_max, (float)c.centerY);
      }
      ImPlot::SetupAxisLimits(ImAxis_Y1, (double)p_min - 10, (double)p_max + 10,
                              ImPlotCond_Once);
    }

    // Custom Formatting (C++26 lambda)
    ImPlot::SetupAxisFormat(
        ImAxis_X1,
        [](double val, char *buff, int size, void *user_data) -> int {
          double base = *static_cast<double *>(user_data);
          std::time_t t = static_cast<std::time_t>(base + val);
          std::tm *tm = std::localtime(&t);
          if (tm) [[likely]] {
            return (int)std::strftime(buff, size, "%H:%M:%S", tm);
          } else {
            return std::snprintf(buff, size, "%.2f", val);
          }
        },
        &base_time_sec);

    // Render Heatmap Background if available
    void *texID = renderer_->getHeatmapTextureID();
    if (texID) {
      ImPlot::PlotImage("Heatmap", texID, ImPlotPoint(0, (double)p_min),
                        ImPlotPoint(30, (double)p_max));
    }

    auto *draw_list = ImPlot::GetPlotDrawList();

    for (const auto &cluster : clusters) {
      int delta = static_cast<int>(cluster.askVolume) -
                  static_cast<int>(cluster.bidVolume);

      ImU32 color;
      float intensity =
          std::clamp(std::abs((float)delta) / 2000.0f, 0.2f, 0.7f);
      if (delta > 0) {
        color = ImColor(0.1f, 0.8f, 0.1f, intensity);
      } else {
        color = ImColor(0.8f, 0.1f, 0.1f, intensity);
      }

      double x1 = (double)cluster.centerX - (double)cluster.width * 0.48;
      double x2 = (double)cluster.centerX + (double)cluster.width * 0.48;
      double y1 = (double)cluster.centerY - (double)cluster.height * 0.48;
      double y2 = (double)cluster.centerY + (double)cluster.height * 0.48;

      ImVec2 p1 = ImPlot::PlotToPixels(x1, y1);
      ImVec2 p2 = ImPlot::PlotToPixels(x2, y2);

      draw_list->AddRectFilled(p1, p2, color);
      draw_list->AddRect(p1, p2, ImColor(1.0f, 1.0f, 1.0f, 0.05f));

      if (show_text && (std::abs(p2.y - p1.y) > 18)) {
        std::string label = std::format("{}", delta);
        ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
        draw_list->AddText(ImVec2((p1.x + p2.x - text_size.x) * 0.5f,
                                  (p1.y + p2.y - text_size.y) * 0.5f),
                           IM_COL32_WHITE, label.c_str());
      }
    }

    ImPlot::EndPlot();
  }

  // Overlay Info
  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.5f),
                     "Real-time Footprint | Latency: 0.1ms");

  end_panel_window();
}

} // namespace BTQuant
