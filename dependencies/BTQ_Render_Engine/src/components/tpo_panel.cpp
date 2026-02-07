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

void TpoPanel::update(float /*dt*/) {
  // Update logic if needed
  // Data processing is handled in the render method to avoid duplication
}

void TpoPanel::render() {
  std::lock_guard<std::mutex> lock(data_mutex_);
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
  if (ImGui::Button("Clear TPO Data")) {
    tpo_engine_.clear();
    last_processed_timestamp_ns_ = 0;  // Reset the tracking timestamp
  }
  ImGui::SameLine();
  ImGui::Checkbox("Delta Labels", &show_text_);
  ImGui::SameLine();
  ImGui::Checkbox("Grid", &show_grid_);
  ImGui::SameLine();
  ImGui::Checkbox("Heatmap", &show_heatmap_);

  // Time window configuration
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::SliderFloat("Time Window", &time_window_, 10.0f, 300.0f, "%.0f s");

  // Get clusters and process them with the TPO engine
  auto clusters = renderer_->getFootprintClusters();
  auto stats = renderer_->getStats();

  // Only process new data if the timestamp has changed
  if (stats.lastUpdateTimeNs > last_processed_timestamp_ns_) {
    // Process clusters with TPO engine to generate TPO data
    // Convert clusters to PriceTicks and feed to TPO engine
    for (const auto& cluster : clusters) {
      // Create a timestamp based on the cluster's time
      auto timestamp =
          std::chrono::system_clock::time_point(std::chrono::nanoseconds(stats.lastUpdateTimeNs));

      // Create PriceTick from cluster data
      PriceTick tick;
      tick.timestamp = timestamp;
      tick.price = cluster.centerY;  // Use center Y as the price
      tick.volume = static_cast<double>(cluster.askVolume + cluster.bidVolume);  // Total volume

      // Process the tick with the TPO engine
      tpo_engine_.process_tick(tick);
    }

    // Update the last processed timestamp
    last_processed_timestamp_ns_ = stats.lastUpdateTimeNs;
  }

  // Get POC and Value Area from the TPO engine
  auto [poc_price, value_area] = tpo_engine_.get_poc_and_value_area(70.0);
  double value_area_low = value_area.first;
  double value_area_high = value_area.second;

  // Base time for labeling (relative to time window)
  double base_time_sec =
      static_cast<double>(stats.lastUpdateTimeNs) / 1'000'000'000.0 - time_window_;

  if (ImPlot::BeginPlot("##TPOProfile", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {
    // Axis Setup - ALL Setup calls must happen BEFORE any locking functions
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

    // Enable grid if requested (must call SetupAxis before SetupAxisLimits)
    if (show_grid_) {
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
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, time_window_, ImPlotCond_Always);
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
    if (show_heatmap_) {
      void* texID = renderer_->getHeatmapTextureID();
      if (texID) {
        ImPlot::PlotImage("Heatmap", texID, ImPlotPoint(0, (double)p_min),
                          ImPlotPoint(time_window_, (double)p_max));
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

      if (show_text_ && (std::abs(p2.y - p1.y) > 18)) {
        std::string label = std::format("{}", delta);
        ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
        draw_list->AddText(
            ImVec2((p1.x + p2.x - text_size.x) * 0.5f, (p1.y + p2.y - text_size.y) * 0.5f),
            IM_COL32_WHITE, label.c_str());
      }
    }

    // Draw Value Area (shaded region between VAH and VAL)
    if (value_area_low < value_area_high && value_area_low > 0) {
      // Draw shaded area for Value Area
      double va_x[] = {0.0, time_window_, time_window_, 0.0};
      double va_y[] = {value_area_low, value_area_low, value_area_high, value_area_high};

      ImPlot::PushStyleColor(ImPlotCol_Fill,
                             ImVec4(1.0f, 0.84f, 0.0f, 0.2f));  // Semi-transparent gold
      ImPlot::PlotShaded("Value Area", va_x, va_y, 4);
      ImPlot::PopStyleColor();

      // Draw Value Area High (VAH) line
      double vah_line_x[2] = {0, time_window_};
      double vah_line_y[2] = {value_area_high, value_area_high};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));  // Orange
      ImPlot::PlotLine("VAH", vah_line_x, vah_line_y, 2);
      ImPlot::PopStyleColor();

      // Draw Value Area Low (VAL) line
      double val_line_x[2] = {0, time_window_};
      double val_line_y[2] = {value_area_low, value_area_low};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));  // Orange
      ImPlot::PlotLine("VAL", val_line_x, val_line_y, 2);
      ImPlot::PopStyleColor();
    }

    // Draw POC line if found (using TPO engine calculated value)
    if (poc_price > 0) {
      double poc_line_x[2] = {0, time_window_};
      double poc_line_y[2] = {poc_price, poc_price};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));  // Bright yellow
      ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);  // 1px line as requested
      ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();
    }

    ImPlot::EndPlot();
  }

  // Enhanced Overlay Info
  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.5f),
                     "TPO Profile | Clusters: %zu | POC: %.4f | VA: %.4f-%.4f", clusters.size(),
                     poc_price > 0 ? poc_price : 0.0, value_area_low > 0 ? value_area_low : 0.0,
                     value_area_high > 0 ? value_area_high : 0.0);

  end_panel_window();
}

}  // namespace BTQuant
