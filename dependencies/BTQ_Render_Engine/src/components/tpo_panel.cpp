#include "../../include/components/tpo_panel.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>

#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TpoPanel::TpoPanel(const PanelConfig& config, RenderEngine::MarketMicrostructureRenderer* renderer)
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

  // Enhanced toolbar with more options
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  static bool show_text = true;
  ImGui::Checkbox("Delta Labels", &show_text);
  ImGui::SameLine();
  static bool show_grid = true;
  ImGui::Checkbox("Grid", &show_grid);
  ImGui::SameLine();
  static bool show_heatmap = true;
  ImGui::Checkbox("Heatmap", &show_heatmap);

  // Time window configuration
  static float time_window = 30.0f;
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::SliderFloat("Time Window", &time_window, 10.0f, 300.0f, "%.0f s");

  auto clusters = renderer_->getFootprintClusters();
  auto stats = renderer_->getStats();

  // Calculate TPO statistics
  double local_poc_price = 0.0;
  double max_volume = 0.0;
  std::unordered_map<double, double> price_volumes;

  // Pre-calculate POC data
  for (const auto& cluster : clusters) {
    // Accumulate volume by price level for POC calculation
    price_volumes[cluster.centerY] += cluster.askVolume + cluster.bidVolume;
  }

  // Find Point of Control (POC) - price level with highest volume
  for (const auto& [price, volume] : price_volumes) {
    if (volume > max_volume) {
      max_volume = volume;
      local_poc_price = price;
    }
  }

  // Base time for labeling (relative to time window)
  double base_time_sec =
      static_cast<double>(stats.lastUpdateTimeNs) / 1'000'000'000.0 - time_window;

  if (ImPlot::BeginPlot("##TPOProfile", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {
    // Axis Setup - ALL Setup calls must happen BEFORE any locking functions
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

    // Enable grid if requested (must call SetupAxis before SetupAxisLimits)
    if (show_grid) {
      ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_None);
      ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_None);
    }

    // Calculate Y-axis limits before calling SetupAxisLimits
    float p_min = 0, p_max = 1000;
    if (!clusters.empty()) {
      p_min = clusters[0].centerY;
      p_max = clusters[0].centerY;
      for (const auto& c : clusters) {
        p_min = std::min(p_min, (float)c.centerY);
        p_max = std::max(p_max, (float)c.centerY);
      }
    }

    // Apply all axis limits at once
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, time_window, ImPlotCond_Always);
    if (!clusters.empty()) {
      ImPlot::SetupAxisLimits(ImAxis_Y1, (double)p_min - 10, (double)p_max + 10, ImPlotCond_Once);
    }

    // Custom Formatting (C++26 lambda)
    ImPlot::SetupAxisFormat(
        ImAxis_X1,
        [](double val, char* buff, int size, void* user_data) -> int {
          double base = *static_cast<double*>(user_data);
          std::time_t t = static_cast<std::time_t>(base + val);
          std::tm* tm = std::localtime(&t);
          if (tm) [[likely]] {
            return (int)std::strftime(buff, size, "%H:%M:%S", tm);
          } else {
            return std::snprintf(buff, size, "%.2f", val);
          }
        },
        &base_time_sec);

    // Render Heatmap Background if available
    if (show_heatmap) {
      void* texID = renderer_->getHeatmapTextureID();
      if (texID) {
        ImPlot::PlotImage("Heatmap", texID, ImPlotPoint(0, (double)p_min),
                          ImPlotPoint(time_window, (double)p_max));
      }
    }

    auto* draw_list = ImPlot::GetPlotDrawList();

    for (const auto& cluster : clusters) {
      int delta = static_cast<int>(cluster.askVolume) - static_cast<int>(cluster.bidVolume);

      ImU32 color;
      float intensity = std::clamp(std::abs((float)delta) / 2000.0f, 0.2f, 0.7f);
      if (delta > 0) {
        color = ImColor(0.1f, 0.8f, 0.1f, intensity);  // Green for positive delta
      } else {
        color = ImColor(0.8f, 0.1f, 0.1f, intensity);  // Red for negative delta
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
        draw_list->AddText(
            ImVec2((p1.x + p2.x - text_size.x) * 0.5f, (p1.y + p2.y - text_size.y) * 0.5f),
            IM_COL32_WHITE, label.c_str());
      }
    }

    // Draw POC line if found (using pre-calculated value)
    if (local_poc_price > 0) {
      double poc_line_x[2] = {0, time_window};
      double poc_line_y[2] = {local_poc_price, local_poc_price};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
      ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
      ImPlot::PopStyleColor();
    }

    ImPlot::EndPlot();
  }

  // Enhanced Overlay Info
  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.5f), "TPO Profile | Clusters: %zu | POC: %.4f",
                     clusters.size(), local_poc_price > 0 ? local_poc_price : 0.0);

  end_panel_window();
}

}  // namespace BTQuant
