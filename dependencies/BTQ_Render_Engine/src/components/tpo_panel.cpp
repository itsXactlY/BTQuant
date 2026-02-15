#include "../../include/components/tpo_panel.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <vector>
#include <unordered_map>
#include <memory>

#include "components/theme_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "imgui.h"
#include "implot.h"

// Define dummy structures for compilation
namespace Data {
    struct Cluster {
        double centerX = 0.0;
        double centerY = 0.0;
        double width = 0.0;
        double height = 0.0;
        double askVolume = 0.0;
        double bidVolume = 0.0;
    };
    
    struct Stats {
        uint64_t lastUpdateTimeNs = 0;
    };
}

namespace BTQuant {

TpoPanel::TpoPanel(const PanelConfig& config)
    : PanelBase(config) {}

void TpoPanel::update(float /*dt*/) {
  // Update logic if needed
}

void TpoPanel::render() {
  begin_panel_window();

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

  // TODO: Implement actual TPO data retrieval
  // For now, using dummy data to allow compilation
  std::vector<Data::Cluster> clusters; // Dummy vector
  Data::Stats stats{}; // Dummy stats

  // Calculate TPO statistics
  double local_poc_price = 0.0;
  double max_volume = 0.0;
  std::unordered_map<double, double> price_volumes;
  std::unordered_map<double, int> tpo_counts; // Track TPO counts per price level

  // Pre-calculate POC data and TPO counts
  for (const auto& cluster : clusters) {
    // Accumulate volume by price level for POC calculation
    price_volumes[cluster.centerY] += cluster.askVolume + cluster.bidVolume;
    
    // Count TPO occurrences per price level (simulating TPO counts)
    // In a real implementation, this would come from the TPO engine
    tpo_counts[cluster.centerY]++;
  }

  // Find Point of Control (POC) - price level with highest volume
  for (const auto& [price, volume] : price_volumes) {
    if (volume > max_volume) {
      max_volume = volume;
      local_poc_price = price;
    }
  }
  
  // Calculate Value Area (70% of TPOs) - simplified implementation
  // In a real implementation, this would use the TPO engine's get_value_area method
  double value_area_low = local_poc_price - 5.0;  // Placeholder calculation
  double value_area_high = local_poc_price + 5.0; // Placeholder calculation
  
  // More accurate calculation based on TPO counts
  if (!tpo_counts.empty()) {
    // Calculate total TPO count
    int total_tpo_count = 0;
    for (const auto& [price, count] : tpo_counts) {
        total_tpo_count += count;
    }
    
    if (total_tpo_count > 0) {
        // Target 70% of total TPOs for value area
        int target_count = static_cast<int>(total_tpo_count * 0.70);
        
        // Sort price levels by distance from POC
        std::vector<std::pair<double, int>> sorted_by_distance;
        for (const auto& [price, count] : tpo_counts) {
            sorted_by_distance.emplace_back(price, count);
        }
        
        std::sort(sorted_by_distance.begin(), sorted_by_distance.end(),
                  [local_poc_price](const auto& a, const auto& b) {
                      return std::abs(a.first - local_poc_price) < std::abs(b.first - local_poc_price);
                  });
        
        // Expand from POC until we reach 70% of TPOs
        int accumulated_count = 0;
        value_area_low = local_poc_price;
        value_area_high = local_poc_price;
        
        for (const auto& [price, count] : sorted_by_distance) {
            if (accumulated_count >= target_count) break;
            
            accumulated_count += count;
            value_area_low = std::min(value_area_low, price);
            value_area_high = std::max(value_area_high, price);
        }
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
      // TODO: Implement heatmap texture rendering
      // For now, skip heatmap rendering to allow compilation
    }

    auto* draw_list = ImPlot::GetPlotDrawList();

    // Get the crosshair price for highlighting
    double crosshair_price = QuantWorkspaceComponent::g_crosshair_price.load(std::memory_order_relaxed);

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
      
      // Check if this cluster corresponds to the crosshair price
      // We'll consider it a match if the crosshair price falls within the cluster's vertical range
      bool is_highlighted = crosshair_price >= y1 && crosshair_price <= y2;
      
      if (is_highlighted) {
        // Draw a highlighted border around the TPO block
        draw_list->AddRect(p1, p2, ImColor(1.0f, 1.0f, 0.0f, 1.0f), 0.0f, ImDrawFlags_RoundCornersAll, 2.0f); // Yellow highlight with 2px thickness
      } else {
        // Draw the normal border
        draw_list->AddRect(p1, p2, ImColor(1.0f, 1.0f, 1.0f, 0.05f));
      }

      if (show_text && (std::abs(p2.y - p1.y) > 18)) {
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
      double va_x[] = {0.0, time_window, time_window, 0.0};
      double va_y[] = {value_area_low, value_area_low, value_area_high, value_area_high};
      
      // ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.84f, 0.0f, 0.2f)); // Semi-transparent gold
      ImPlot::PlotShaded("Value Area", va_x, va_y, 4);
      ImPlot::PopStyleColor();
      
      // Draw Value Area High (VAH) line
      double vah_line_x[2] = {0, time_window};
      double vah_line_y[2] = {value_area_high, value_area_high};
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f)); // Orange
      ImPlot::PlotLine("VAH", vah_line_x, vah_line_y, 2);
      ImPlot::PopStyleColor();
      
      // Draw Value Area Low (VAL) line
      double val_line_x[2] = {0, time_window};
      double val_line_y[2] = {value_area_low, value_area_low};
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.5f, 0.0f, 1.0f)); // Orange
      ImPlot::PlotLine("VAL", val_line_x, val_line_y, 2);
      ImPlot::PopStyleColor();
    }

    // Draw POC line if found (using pre-calculated value)
    if (local_poc_price > 0) {
      double poc_line_x[2] = {0, time_window};
      double poc_line_y[2] = {local_poc_price, local_poc_price};
      // ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Bright yellow
      // ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f); // 1px line as requested
      ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();
    }

    ImPlot::EndPlot();
  }

  // Enhanced Overlay Info
  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.5f), "TPO Profile | Clusters: %zu | POC: %.4f | VA: %.4f-%.4f",
                     clusters.size(), 
                     local_poc_price > 0 ? local_poc_price : 0.0,
                     value_area_low > 0 ? value_area_low : 0.0,
                     value_area_high > 0 ? value_area_high : 0.0);

  end_panel_window();
}

}  // namespace BTQuant
