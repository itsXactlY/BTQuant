#include "components/footprint_panel.hpp"
#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <cmath>

namespace BTQuant {

FootprintPanel::FootprintPanel(
    const PanelConfig &config,
    RenderEngine::MarketMicrostructureRenderer *renderer)
    : PanelBase(config), renderer_(renderer), data_type_(Data::UnifiedDataPipeline::DataType::FOOTPRINT) {}

void FootprintPanel::update(float dt) {
  // Update logic if needed
  // Aggregation is handled by MarketMicrostructureRenderer
}

ImU32 FootprintPanel::getCellColor(const FootprintCell& cell) const {
  // Calculate total volume for intensity
  double total_vol = cell.bid_volume + cell.ask_volume;
  
  // Calculate normalized delta for color coding
  double max_vol = std::max(cell.bid_volume, cell.ask_volume);
  double normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
  normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);
  
  // Calculate intensity based on total volume (0.2 to 0.8 alpha)
  float intensity = std::clamp(static_cast<float>(total_vol / 10000.0f), 0.2f, 0.8f);
  
  // Exocharts-style color coding
  // Green for positive delta (more bids), Red for negative delta (more asks)
  // Neutral gray for balanced
  
  if (normalized_delta > delta_threshold_) {
    // Positive delta - Green gradient
    float green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
    return IM_COL32(
        static_cast<int>(0),
        static_cast<int>(50 + 205 * green_intensity),
        static_cast<int>(50 + 205 * green_intensity),
        static_cast<int>(intensity * 255));
  } else if (normalized_delta < -delta_threshold_) {
    // Negative delta - Red gradient
    float red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
    return IM_COL32(
        static_cast<int>(50 + 205 * red_intensity),
        static_cast<int>(0),
        static_cast<int>(0),
        static_cast<int>(intensity * 255));
  } else {
    // Neutral - Gray
    return IM_COL32(
        static_cast<int>(80),
        static_cast<int>(80),
        static_cast<int>(80),
        static_cast<int>(intensity * 255));
  }
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  // Show the larger of bid/ask volume
  double max_vol = std::max(cell.bid_volume, cell.ask_volume);
  
  // Format based on volume size
  if (max_vol >= 1000.0) {
    return std::format("{:.1f}K", max_vol / 1000.0);
  } else if (max_vol >= 100.0) {
    return std::format("{:.0f}", max_vol);
  } else {
    return std::format("{:.1f}", max_vol);
  }
}

void FootprintPanel::renderCell(const FootprintCell& cell, ImDrawList* draw_list) {
  // Calculate cell corners in plot coordinates
  double x1 = cell.x - cell.width * 0.48;
  double x2 = cell.x + cell.width * 0.48;
  double y1 = cell.y - cell.height * 0.48;
  double y2 = cell.y + cell.height * 0.48;
  
  // Convert to pixel coordinates
  ImVec2 p1 = ImPlot::PlotToPixels(x1, y1);
  ImVec2 p2 = ImPlot::PlotToPixels(x2, y2);
  
  // Get cell color
  ImU32 color = getCellColor(cell);
  
  // Draw filled cell
  draw_list->AddRectFilled(p1, p2, color);
  
  // Draw subtle border for cell separation
  ImU32 border_color = IM_COL32(255, 255, 255, 13); // White, 5% alpha
  draw_list->AddRect(p1, p2, border_color, 0.0f, 0, 1.0f);
  
  // Draw volume label if enabled and cell is large enough
  if (show_volume_labels_ && (std::abs(p2.y - p1.y) > 18)) {
    std::string label = getCellLabel(cell);
    ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
    
    // Center text in cell
    ImVec2 text_pos(
        (p1.x + p2.x - text_size.x) * 0.5f,
        (p1.y + p2.y - text_size.y) * 0.5f);
    
    draw_list->AddText(text_pos, IM_COL32_WHITE, label.c_str());
  }
  
  // Draw delta indicator if enabled
  if (show_delta_indicator_) {
    double max_vol = std::max(cell.bid_volume, cell.ask_volume);
    if (max_vol > 0.0) {
      double normalized_delta = cell.delta / max_vol;
      
      // Draw small indicator bar at the bottom of the cell
      if (std::abs(normalized_delta) > delta_threshold_) {
        float bar_height = 3.0f;
        float bar_width = (p2.x - p1.x) * 0.8f;
        ImVec2 bar_pos(
            p1.x + (p2.x - p1.x - bar_width) * 0.5f,
            p2.y - bar_height - 1.0f);
        
        ImU32 bar_color = normalized_delta > 0 
            ? IM_COL32(0, 255, 0, 200)  // Green
            : IM_COL32(255, 0, 0, 200); // Red
        
        draw_list->AddRectFilled(
            ImVec2(bar_pos.x, bar_pos.y),
            ImVec2(bar_pos.x + bar_width, bar_pos.y + bar_height),
            bar_color);
      }
    }
  }
}

void FootprintPanel::render() {
  begin_panel_window();

  if (!renderer_) {
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Renderer unavailable");
    end_panel_window();
    return;
  }

  // Data type selector
  const char* data_type_names[] = {
    "OHLC", "ORDERBOOK", "TRADES", "VOLUME_PROFILE", "FOOTPRINT", "TPO", "METRICS", "ALERTS"
  };

  int current_data_type = static_cast<int>(data_type_);
  if (ImGui::BeginCombo("Data Type", data_type_names[current_data_type])) {
    for (int i = 0; i < 8; i++) {
      bool is_selected = (current_data_type == i);
      if (ImGui::Selectable(data_type_names[i], is_selected)) {
        current_data_type = i;
        data_type_ = static_cast<Data::UnifiedDataPipeline::DataType>(i);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();

  // Enhanced toolbar with more options
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Volume Labels", &show_volume_labels_);
  ImGui::SameLine();
  ImGui::Checkbox("Delta Indicator", &show_delta_indicator_);
  ImGui::SameLine();
  static bool show_grid = true;
  ImGui::Checkbox("Grid", &show_grid);

  // Grid size configuration
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::SliderInt("Cols", &grid_cols_, 30, 120);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::SliderInt("Rows", &grid_rows_, 50, 200);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::SliderFloat("Delta Thresh", &delta_threshold_, 0.0f, 1.0f, "%.2f");

  // Get clusters from renderer
  auto clusters = renderer_->getFootprintClusters();
  auto stats = renderer_->getStats();

  // Base time for absolute labeling (relative to 30s window)
  double base_time_sec =
      static_cast<double>(stats.lastUpdateTimeNs) / 1'000'000'000.0 - 30.0;

  if (ImPlot::BeginPlot("##FootprintPlot", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {

    // Axis Setup
    ImPlot::SetupAxes("Time (s)", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);

    // Enable grid if requested
    if (show_grid) {
        ImPlot::SetupAxis(ImAxis_X1, "Time (s)", ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_None);
    }

    // Auto-scale X-axis to time window
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, 30, ImPlotCond_Always);

    // Auto-scale Y-axis to data
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

    // Custom Time Formatting (C++26 lambda)
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

    // Get draw list for custom rendering
    auto *draw_list = ImPlot::GetPlotDrawList();

    // Render each cluster as a footprint cell
    for (const auto &cluster : clusters) {
      // Convert cluster to footprint cell
      FootprintCell cell(
          cluster.centerX,           // x (time)
          cluster.centerY,           // y (price)
          cluster.width,            // width (time duration)
          cluster.height,           // height (price range)
          cluster.bidVolume,        // bid_volume
          cluster.askVolume,        // ask_volume
          cluster.tradeCount,       // trade_count
          cluster.vwap             // vwap
      );

      // Render the cell
      renderCell(cell, draw_list);
    }

    ImPlot::EndPlot();
  }

  // Enhanced Debug Overlay
  if (!clusters.empty()) {
    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1),
                       "Clusters: %zu | Grid: %dx%d | Thresh: %.2f",
                       clusters.size(), grid_cols_, grid_rows_, delta_threshold_);

    // Calculate statistics
    double total_bid_vol = 0.0;
    double total_ask_vol = 0.0;
    double total_trade_count = 0;
    for (const auto &c : clusters) {
      total_bid_vol += c.bidVolume;
      total_ask_vol += c.askVolume;
      total_trade_count += c.tradeCount;
    }
    double total_delta = total_bid_vol - total_ask_vol;

    ImGui::Text("Total Bid: %.2f | Total Ask: %.2f | Delta: %.2f | Trades: %.0f",
                total_bid_vol, total_ask_vol, total_delta, total_trade_count);

    // Add VWAP line if available from market data
    // Note: This requires renderer to have access to market data processor
    // For now, we'll skip this to avoid compilation errors
    // auto analytics = renderer_->getMarketDataProcessor()->getSymbolAnalytics(symbol_id_);
    // if (analytics.vwap > 0) {
    //     ImGui::Text("VWAP: %.4f", analytics.vwap);
    // }
  }

  end_panel_window();
}

} // namespace BTQuant
