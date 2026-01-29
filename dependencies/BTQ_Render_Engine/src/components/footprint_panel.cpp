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
    : PanelBase(config), renderer_(renderer), data_type_(Data::UnifiedDataPipeline::DataType::FOOTPRINT),
      volume_data_type_(Data::VolumeDataType::Delta) {}

void FootprintPanel::update(float dt) {
  // Update logic if needed
  // Aggregation is handled by MarketMicrostructureRenderer
}

ImU32 FootprintPanel::getCellColor(const FootprintCell& cell) const {
  // Calculate values based on selected volume data type
  double value_to_display = 0.0;
  double total_vol = cell.bid_volume + cell.ask_volume;
  double intensity = 0.0f;

  switch (volume_data_type_) {
    case Data::VolumeDataType::Trades:
      value_to_display = static_cast<double>(cell.trade_count);
      intensity = std::clamp(static_cast<float>(value_to_display / 100.0f), 0.2f, 0.8f);
      break;

    case Data::VolumeDataType::Volume:
      value_to_display = total_vol;
      intensity = std::clamp(static_cast<float>(total_vol / 10000.0f), 0.2f, 0.8f);
      break;

    case Data::VolumeDataType::BuyVolume:
      value_to_display = cell.bid_volume;
      intensity = std::clamp(static_cast<float>(cell.bid_volume / 5000.0f), 0.2f, 0.8f);
      break;

    case Data::VolumeDataType::SellVolume:
      value_to_display = cell.ask_volume;
      intensity = std::clamp(static_cast<float>(cell.ask_volume / 5000.0f), 0.2f, 0.8f);
      break;

    case Data::VolumeDataType::BuySellVolume:
      value_to_display = cell.bid_volume - cell.ask_volume;
      // For BuySellVolume, normalize based on max of bid/ask volume
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_value = max_vol > 0.0 ? value_to_display / max_vol : 0.0;
        normalized_value = std::clamp(normalized_value, -1.0, 1.0);
        intensity = std::clamp(static_cast<float>(std::abs(value_to_display) / 5000.0f), 0.2f, 0.8f);

        // Return color based on sign of value
        if (normalized_value > delta_threshold_) {
          // Positive - Green gradient
          float green_intensity = std::clamp(static_cast<float>(normalized_value), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(0),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(intensity * 255));
        } else if (normalized_value < -delta_threshold_) {
          // Negative - Red gradient
          float red_intensity = std::clamp(static_cast<float>(-normalized_value), 0.0f, 1.0f);
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
      break;

    case Data::VolumeDataType::Delta:
    default: // Default to Delta
      value_to_display = cell.delta;
      // Calculate normalized delta for color coding
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
        normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

        // Calculate intensity based on total volume (0.2 to 0.8 alpha)
        intensity = std::clamp(static_cast<float>(total_vol / 10000.0f), 0.2f, 0.8f);

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
      break;

    case Data::VolumeDataType::DeltaPercent:
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        value_to_display = max_vol > 0.0 ? (cell.delta / max_vol) * 100.0 : 0.0;
        double normalized_delta = std::clamp(value_to_display / 100.0, -1.0, 1.0);
        intensity = std::clamp(static_cast<float>(total_vol / 10000.0f), 0.2f, 0.8f);

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
      break;
  }

  // For scalar values (Trades, Volume, BuyVolume, SellVolume), use a blue gradient
  float blue_intensity = std::clamp(static_cast<float>(value_to_display / 10000.0f), 0.0f, 1.0f);
  return IM_COL32(
      static_cast<int>(0),
      static_cast<int>(0),
      static_cast<int>(50 + 205 * blue_intensity),
      static_cast<int>(intensity * 255));
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  double value_to_display = 0.0;

  switch (volume_data_type_) {
    case Data::VolumeDataType::Trades:
      return std::format("{}", cell.trade_count);

    case Data::VolumeDataType::Volume:
      value_to_display = cell.bid_volume + cell.ask_volume;
      break;

    case Data::VolumeDataType::BuyVolume:
      value_to_display = cell.bid_volume;
      break;

    case Data::VolumeDataType::SellVolume:
      value_to_display = cell.ask_volume;
      break;

    case Data::VolumeDataType::BuySellVolume:
      value_to_display = cell.bid_volume - cell.ask_volume;
      break;

    case Data::VolumeDataType::Delta:
      value_to_display = cell.delta;
      break;

    case Data::VolumeDataType::DeltaPercent:
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        value_to_display = max_vol > 0.0 ? (cell.delta / max_vol) * 100.0 : 0.0;
        return std::format("{:.1f}%", value_to_display);
      }

    default:
      // For other types, default to showing the larger of bid/ask volume
      value_to_display = std::max(cell.bid_volume, cell.ask_volume);
      break;
  }

  // Format based on value size
  if (std::abs(value_to_display) >= 1000.0) {
    return std::format("{:.1f}K", value_to_display / 1000.0);
  } else if (std::abs(value_to_display) >= 100.0) {
    return std::format("{:.0f}", value_to_display);
  } else {
    return std::format("{:.1f}", value_to_display);
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

  // Volume data type selector for footprint visualization
  const char* volume_data_type_names[] = {
    "Trades", "BuyTrades", "SellTrades", "Volume", "BuyVolume", "SellVolume",
    "BuyVolume%", "SellVolume%", "BuySellVolume", "Delta", "Delta%", "CumulativeDelta",
    "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol", "FilteredVol"
  };

  int current_vol_data_type = static_cast<int>(volume_data_type_);
  if (ImGui::BeginCombo("Footprint Mode", volume_data_type_names[current_vol_data_type])) {
    for (int i = 0; i < 17; i++) {
      bool is_selected = (current_vol_data_type == i);
      if (ImGui::Selectable(volume_data_type_names[i], is_selected)) {
        current_vol_data_type = i;
        volume_data_type_ = static_cast<Data::VolumeDataType>(i);
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
    // Determine the current volume data type name for display
    const char* vol_type_names[] = {
      "Trades", "BuyTrades", "SellTrades", "Volume", "BuyVolume", "SellVolume",
      "BuyVol%", "SellVol%", "BuySellVol", "Delta", "Delta%", "CumulativeDelta",
      "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol", "FilteredVol"
    };

    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1),
                       "Mode: %s | Clusters: %zu | Grid: %dx%d | Thresh: %.2f",
                       vol_type_names[static_cast<int>(volume_data_type_)],
                       clusters.size(), grid_cols_, grid_rows_, delta_threshold_);

    // Calculate statistics based on selected volume data type
    double total_value = 0.0;
    double total_bid_vol = 0.0;
    double total_ask_vol = 0.0;
    double total_trade_count = 0;
    for (const auto &c : clusters) {
      total_bid_vol += c.bidVolume;
      total_ask_vol += c.askVolume;
      total_trade_count += c.tradeCount;

      switch (volume_data_type_) {
        case Data::VolumeDataType::Trades:
          total_value += c.tradeCount;
          break;

        case Data::VolumeDataType::Volume:
          total_value += c.bidVolume + c.askVolume;
          break;

        case Data::VolumeDataType::BuyVolume:
          total_value += c.bidVolume;
          break;

        case Data::VolumeDataType::SellVolume:
          total_value += c.askVolume;
          break;

        case Data::VolumeDataType::BuySellVolume:
          total_value += c.bidVolume - c.askVolume;
          break;

        case Data::VolumeDataType::Delta:
          total_value += c.bidVolume - c.askVolume;
          break;

        case Data::VolumeDataType::DeltaPercent:
          {
            double max_vol = std::max(c.bidVolume, c.askVolume);
            total_value += max_vol > 0.0 ? ((c.bidVolume - c.askVolume) / max_vol) * 100.0 : 0.0;
          }
          break;

        default:
          total_value += c.bidVolume + c.askVolume; // Default to total volume
          break;
      }
    }

    ImGui::Text("Total Value: %.2f | Bid: %.2f | Ask: %.2f | Trades: %.0f",
                total_value, total_bid_vol, total_ask_vol, total_trade_count);
  }

  end_panel_window();
}

} // namespace BTQuant
