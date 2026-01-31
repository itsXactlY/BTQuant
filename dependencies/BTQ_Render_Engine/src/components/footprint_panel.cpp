#include "components/footprint_panel.hpp"
#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <cmath>
#include <map>
#include "analytics/cluster_engine.hpp"

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

ImU32 FootprintPanel::getCellColor(const FootprintCell& cell, double max_volume) const {
  // Calculate values based on selected volume data type
  double value_to_display = 0.0;
  double total_vol = cell.bid_volume + cell.ask_volume;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Trades:
      value_to_display = static_cast<double>(cell.trade_count);
      break;

    case Data::VolumeAnalysisType::Volume:
      value_to_display = total_vol;
      break;

    case Data::VolumeAnalysisType::BuyVolume:
      value_to_display = cell.bid_volume;
      break;

    case Data::VolumeAnalysisType::SellVolume:
      value_to_display = cell.ask_volume;
      break;

    case Data::VolumeAnalysisType::BuySellVolume:
      value_to_display = cell.bid_volume - cell.ask_volume;
      // For BuySellVolume, normalize based on max of bid/ask volume
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_value = max_vol > 0.0 ? value_to_display / max_vol : 0.0;
        normalized_value = std::clamp(normalized_value, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        // Return color based on sign of value using blue-red gradient
        if (normalized_value > delta_threshold_) {
          // Positive - Blue gradient
          float blue_intensity = std::clamp(static_cast<float>(normalized_value), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(0),
              static_cast<int>(0),
              static_cast<int>(50 + 205 * blue_intensity),
              static_cast<int>(alpha * 255));
        } else if (normalized_value < -delta_threshold_) {
          // Negative - Red gradient
          float red_intensity = std::clamp(static_cast<float>(-normalized_value), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(50 + 205 * red_intensity),
              static_cast<int>(0),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::Delta:
    default: // Default to Delta
      value_to_display = cell.delta;
      // Calculate normalized delta for color coding
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
        normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        // Green-red gradient for delta
        if (normalized_delta > delta_threshold_) {
          // Positive delta - Green gradient
          float green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(0),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(alpha * 255));
        } else if (normalized_delta < -delta_threshold_) {
          // Negative delta - Red gradient
          float red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(50 + 205 * red_intensity),
              static_cast<int>(0),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::DeltaPercent:
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        value_to_display = max_vol > 0.0 ? (cell.delta / max_vol) * 100.0 : 0.0;
        double normalized_delta = std::clamp(value_to_display / 100.0, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        if (normalized_delta > delta_threshold_) {
          // Positive delta - Green gradient
          float green_intensity = std::clamp(static_cast<float>(normalized_delta/100.0), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(0),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(alpha * 255));
        } else if (normalized_delta < -delta_threshold_) {
          // Negative delta - Red gradient
          float red_intensity = std::clamp(static_cast<float>(-normalized_delta/100.0), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(50 + 205 * red_intensity),
              static_cast<int>(0),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::BuyTrades:
      value_to_display = static_cast<double>(cell.trade_count) * 0.6; // Placeholder
      break;

    case Data::VolumeAnalysisType::SellTrades:
      value_to_display = static_cast<double>(cell.trade_count) * 0.4; // Placeholder
      break;

    case Data::VolumeAnalysisType::BuyVolumePercent:
      {
        double total_vol_local = cell.bid_volume + cell.ask_volume;
        value_to_display = total_vol_local > 0.0 ? (cell.bid_volume / total_vol_local) * 100.0 : 0.0;
        double normalized_val = std::clamp(value_to_display / 100.0, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        if (normalized_val > delta_threshold_) {
          // Positive - Yellow gradient for volume intensity
          float yellow_intensity = std::clamp(static_cast<float>(normalized_val), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(255 * yellow_intensity),
              static_cast<int>(255 * yellow_intensity),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else if (normalized_val < -delta_threshold_) {
          // Negative - Orange gradient for volume intensity
          float orange_intensity = std::clamp(static_cast<float>(-normalized_val), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(255 * orange_intensity),
              static_cast<int>(165 * orange_intensity),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::SellVolumePercent:
      {
        double total_vol_local = cell.bid_volume + cell.ask_volume;
        value_to_display = total_vol_local > 0.0 ? (cell.ask_volume / total_vol_local) * 100.0 : 0.0;
        double normalized_val = std::clamp(value_to_display / 100.0, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        if (normalized_val > delta_threshold_) {
          // Positive - Yellow gradient for volume intensity
          float yellow_intensity = std::clamp(static_cast<float>(normalized_val), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(255 * yellow_intensity),
              static_cast<int>(255 * yellow_intensity),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else if (normalized_val < -delta_threshold_) {
          // Negative - Orange gradient for volume intensity
          float orange_intensity = std::clamp(static_cast<float>(-normalized_val), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(255 * orange_intensity),
              static_cast<int>(165 * orange_intensity),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::CumulativeDelta:
      value_to_display = cell.delta; // Using same as delta for demo
      // Calculate normalized delta for color coding
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
        normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        // Green-red gradient for cumulative delta
        if (normalized_delta > delta_threshold_) {
          // Positive delta - Green gradient
          float green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(0),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(50 + 205 * green_intensity),
              static_cast<int>(alpha * 255));
        } else if (normalized_delta < -delta_threshold_) {
          // Negative delta - Red gradient
          float red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
          return IM_COL32(
              static_cast<int>(50 + 205 * red_intensity),
              static_cast<int>(0),
              static_cast<int>(0),
              static_cast<int>(alpha * 255));
        } else {
          // Neutral - Gray
          return IM_COL32(
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(80),
              static_cast<int>(alpha * 255));
        }
      }
      break;

    case Data::VolumeAnalysisType::AverageSize:
      {
        int total_count = cell.trade_count;
        value_to_display = total_count > 0 ? (cell.bid_volume + cell.ask_volume) / static_cast<double>(total_count) : 0.0;
      }
      break;

    case Data::VolumeAnalysisType::AverageBuySize:
      {
        int buy_count = static_cast<int>(cell.trade_count * 0.6); // Placeholder
        value_to_display = buy_count > 0 ? cell.bid_volume / static_cast<double>(buy_count) : 0.0;
      }
      break;

    case Data::VolumeAnalysisType::AverageSellSize:
      {
        int sell_count = static_cast<int>(cell.trade_count * 0.4); // Placeholder
        value_to_display = sell_count > 0 ? cell.ask_volume / static_cast<double>(sell_count) : 0.0;
      }
      break;

    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      value_to_display = (cell.bid_volume + cell.ask_volume) * 0.1; // Placeholder
      break;

    case Data::VolumeAnalysisType::FilteredVolume:
      value_to_display = cell.bid_volume + cell.ask_volume; // Using same as total volume for demo
      break;
  }

  // Calculate adaptive alpha based on total volume relative to max possible volume
  float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

  // For scalar values (Trades, Volume, BuyVolume, SellVolume), use yellow-orange gradient for volume intensity
  if (volume_data_type_ == Data::VolumeAnalysisType::Volume ||
      volume_data_type_ == Data::VolumeAnalysisType::BuyVolume ||
      volume_data_type_ == Data::VolumeAnalysisType::SellVolume ||
      volume_data_type_ == Data::VolumeAnalysisType::FilteredVolume) {
    // Yellow-orange gradient for volume intensity
    float volume_intensity = std::clamp(static_cast<float>(value_to_display / 10000.0f), 0.0f, 1.0f);
    return IM_COL32(
        static_cast<int>(255 * volume_intensity),
        static_cast<int>(165 * volume_intensity),
        static_cast<int>(0),
        static_cast<int>(alpha * 255));
  } else {
    // Blue gradient for trades and other metrics
    float blue_intensity = std::clamp(static_cast<float>(value_to_display / 10000.0f), 0.0f, 1.0f);
    return IM_COL32(
        static_cast<int>(0),
        static_cast<int>(0),
        static_cast<int>(50 + 205 * blue_intensity),
        static_cast<int>(alpha * 255));
  }
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  double value_to_display = 0.0;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Trades:
      return std::format("{}", cell.trade_count);

    case Data::VolumeAnalysisType::Volume:
      value_to_display = cell.bid_volume + cell.ask_volume;
      break;

    case Data::VolumeAnalysisType::BuyVolume:
      value_to_display = cell.bid_volume;
      break;

    case Data::VolumeAnalysisType::SellVolume:
      value_to_display = cell.ask_volume;
      break;

    case Data::VolumeAnalysisType::BuySellVolume:
      value_to_display = cell.bid_volume - cell.ask_volume;
      break;

    case Data::VolumeAnalysisType::Delta:
      value_to_display = cell.delta;
      break;

    case Data::VolumeAnalysisType::DeltaPercent:
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        value_to_display = max_vol > 0.0 ? (cell.delta / max_vol) * 100.0 : 0.0;
        return std::format("{:.1f}%", value_to_display);
      }

    case Data::VolumeAnalysisType::BuyTrades:
      return std::format("{}", static_cast<int>(cell.trade_count * 0.6)); // Placeholder

    case Data::VolumeAnalysisType::SellTrades:
      return std::format("{}", static_cast<int>(cell.trade_count * 0.4)); // Placeholder

    case Data::VolumeAnalysisType::BuyVolumePercent:
      {
        double total_vol = cell.bid_volume + cell.ask_volume;
        value_to_display = total_vol > 0.0 ? (cell.bid_volume / total_vol) * 100.0 : 0.0;
        return std::format("{:.1f}%", value_to_display);
      }

    case Data::VolumeAnalysisType::SellVolumePercent:
      {
        double total_vol = cell.bid_volume + cell.ask_volume;
        value_to_display = total_vol > 0.0 ? (cell.ask_volume / total_vol) * 100.0 : 0.0;
        return std::format("{:.1f}%", value_to_display);
      }

    case Data::VolumeAnalysisType::CumulativeDelta:
      value_to_display = cell.delta; // Using same as delta for demo
      break;

    case Data::VolumeAnalysisType::AverageSize:
      {
        int total_count = cell.trade_count;
        value_to_display = total_count > 0 ? (cell.bid_volume + cell.ask_volume) / static_cast<double>(total_count) : 0.0;
        break;
      }

    case Data::VolumeAnalysisType::AverageBuySize:
      {
        int buy_count = static_cast<int>(cell.trade_count * 0.6); // Placeholder
        value_to_display = buy_count > 0 ? cell.bid_volume / static_cast<double>(buy_count) : 0.0;
        break;
      }

    case Data::VolumeAnalysisType::AverageSellSize:
      {
        int sell_count = static_cast<int>(cell.trade_count * 0.4); // Placeholder
        value_to_display = sell_count > 0 ? cell.ask_volume / static_cast<double>(sell_count) : 0.0;
        break;
      }

    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      value_to_display = (cell.bid_volume + cell.ask_volume) * 0.1; // Placeholder
      break;

    case Data::VolumeAnalysisType::FilteredVolume:
      value_to_display = cell.bid_volume + cell.ask_volume; // Using same as total volume for demo
      break;

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

void FootprintPanel::renderCell(const FootprintCell& cell, ImDrawList* draw_list, double max_volume) {
  // Call the overloaded version with empty imbalance vectors
  std::vector<FootprintCell> empty_diagonal;
  std::vector<FootprintCell> empty_stacked;
  renderCell(cell, draw_list, max_volume, empty_diagonal, empty_stacked);
}

void FootprintPanel::renderCell(const FootprintCell& cell, ImDrawList* draw_list, double max_volume,
                               const std::vector<FootprintCell> &diagonal_imbalances,
                               const std::vector<FootprintCell> &stacked_imbalances) {
  // Calculate cell corners in plot coordinates
  double x1 = cell.x - cell.width * 0.48;
  double x2 = cell.x + cell.width * 0.48;
  double y1 = cell.y - cell.height * 0.48;
  double y2 = cell.y + cell.height * 0.48;

  // Convert to pixel coordinates
  ImVec2 p1 = ImPlot::PlotToPixels(x1, y1);
  ImVec2 p2 = ImPlot::PlotToPixels(x2, y2);

  // Get cell color
  ImU32 color = getCellColor(cell, max_volume);

  // Draw filled cell
  draw_list->AddRectFilled(p1, p2, color);

  // Check if this cell is part of any imbalances
  bool is_diagonal = false;
  bool is_stacked = false;

  // Check against diagonal imbalances
  for (const auto& diag_cell : diagonal_imbalances) {
    if (std::abs(cell.x - diag_cell.x) < 0.001 && std::abs(cell.y - diag_cell.y) < 0.001) {
      is_diagonal = true;
      break;
    }
  }

  // Check against stacked imbalances
  for (const auto& stack_cell : stacked_imbalances) {
    if (std::abs(cell.x - stack_cell.x) < 0.001 && std::abs(cell.y - stack_cell.y) < 0.001) {
      is_stacked = true;
      break;
    }
  }

  // Draw border with special highlighting for imbalances
  if (is_diagonal || is_stacked) {
    // Thicker border for imbalanced cells
    float border_thickness = 3.0f; // Thicker border

    // Yellow border for diagonal imbalance
    if (is_diagonal) {
      ImU32 diagonal_border_color = IM_COL32(255, 255, 0, 255); // Yellow
      draw_list->AddRect(p1, p2, diagonal_border_color, 0.0f, 0, border_thickness);

      // Optional glow effect for diagonal imbalance
      ImU32 glow_color = IM_COL32(255, 255, 0, 100); // Semi-transparent yellow
      ImVec2 glow_offset(2.0f, 2.0f);
      draw_list->AddRect(ImVec2(p1.x - glow_offset.x, p1.y - glow_offset.y),
                         ImVec2(p2.x + glow_offset.x, p2.y + glow_offset.y),
                         glow_color, 0.0f, 0, 1.0f);
    }

    // Cyan border for stacked imbalance
    if (is_stacked) {
      ImU32 stacked_border_color = IM_COL32(0, 255, 255, 255); // Cyan
      draw_list->AddRect(p1, p2, stacked_border_color, 0.0f, 0, border_thickness);

      // Optional glow effect for stacked imbalance
      ImU32 glow_color = IM_COL32(0, 255, 255, 100); // Semi-transparent cyan
      ImVec2 glow_offset(2.0f, 2.0f);
      draw_list->AddRect(ImVec2(p1.x - glow_offset.x, p1.y - glow_offset.y),
                         ImVec2(p2.x + glow_offset.x, p2.y + glow_offset.y),
                         glow_color, 0.0f, 0, 1.0f);
    }

    // If both, draw both effects
    if (is_diagonal && is_stacked) {
      // Draw both borders with different thickness to distinguish
      ImU32 diagonal_border_color = IM_COL32(255, 255, 0, 255); // Yellow
      ImU32 stacked_border_color = IM_COL32(0, 255, 255, 255); // Cyan
      draw_list->AddRect(ImVec2(p1.x - 1, p1.y - 1), ImVec2(p2.x + 1, p2.y + 1),
                         diagonal_border_color, 0.0f, 0, 2.0f);
      draw_list->AddRect(ImVec2(p1.x - 2, p1.y - 2), ImVec2(p2.x + 2, p2.y + 2),
                         stacked_border_color, 0.0f, 0, 1.0f);
    }
  } else {
    // Draw subtle border for cell separation (normal case)
    ImU32 border_color = IM_COL32(255, 255, 255, 13); // White, 5% alpha
    draw_list->AddRect(p1, p2, border_color, 0.0f, 0, 1.0f);
  }

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

bool FootprintPanel::isDiagonalImbalance(const FootprintCell& cell, const std::vector<FootprintCell>& all_cells) const {
  // Define threshold for imbalance detection
  const double threshold = 3.0; // Standard threshold from cluster engine

  // Look for diagonal patterns: buy volume at price P compared to sell volume at price P-1
  for (const auto& other_cell : all_cells) {
    // Check if this is a diagonal neighbor (adjacent price level, same or nearby time)
    // Diagonal imbalance: comparing buy volume at one price level with sell volume at adjacent price level
    if (std::abs(std::abs(cell.y - other_cell.y) - 1.0) < 0.2 && std::abs(cell.x - other_cell.x) < 2.0) { // Adjacent price level, similar time
      // Check buy volume at current cell vs sell volume at other cell (or vice versa)
      if (cell.bid_volume > 0 && other_cell.ask_volume > 0) {
        double ratio = cell.bid_volume / other_cell.ask_volume;
        if (ratio > threshold) {
          return true;
        }
      }
      if (cell.ask_volume > 0 && other_cell.bid_volume > 0) {
        double ratio = cell.ask_volume / other_cell.bid_volume;
        if (ratio > threshold) {
          return true;
        }
      }
    }
  }

  return false;
}

bool FootprintPanel::isStackedImbalance(const FootprintCell& cell, const std::vector<FootprintCell>& all_cells) const {
  // Define threshold for imbalance detection
  const double threshold = 3.0; // Standard threshold from cluster engine

  // Look for stacked patterns: comparing volumes at same price level across consecutive time buckets
  for (const auto& other_cell : all_cells) {
    // Check if this is at the same price level but different time (stacked in time dimension)
    if (std::abs(cell.y - other_cell.y) < 0.2 && std::abs(std::abs(cell.x - other_cell.x) - 1.0) < 0.2) { // Same price, adjacent time
      // Check buy-side stacked imbalance (current buy vs previous buy)
      if (other_cell.bid_volume > 0) {
        double buy_ratio = cell.bid_volume / other_cell.bid_volume;
        if (buy_ratio > threshold) {
          return true;
        }
      }

      // Check sell-side stacked imbalance (current sell vs previous sell)
      if (other_cell.ask_volume > 0) {
        double sell_ratio = cell.ask_volume / other_cell.ask_volume;
        if (sell_ratio > threshold) {
          return true;
        }
      }

      // Check for opposite imbalances (potential reversal signals)
      // Current sell vs previous buy (bearish signal)
      if (other_cell.bid_volume > 0) {
        double sell_vs_prev_buy_ratio = cell.ask_volume / other_cell.bid_volume;
        if (sell_vs_prev_buy_ratio > threshold) {
          return true;
        }
      }

      // Current buy vs previous sell (bullish signal)
      if (other_cell.ask_volume > 0) {
        double buy_vs_prev_sell_ratio = cell.bid_volume / other_cell.ask_volume;
        if (buy_vs_prev_sell_ratio > threshold) {
          return true;
        }
      }
    }
  }

  return false;
}

void FootprintPanel::detectImbalances(const std::vector<FootprintCell>& cells,
                                     std::vector<FootprintCell>& diagonal_imbalances,
                                     std::vector<FootprintCell>& stacked_imbalances) const {
  diagonal_imbalances.clear();
  stacked_imbalances.clear();

  for (const auto& cell : cells) {
    if (isDiagonalImbalance(cell, cells)) {
      diagonal_imbalances.push_back(cell);
    }

    if (isStackedImbalance(cell, cells)) {
      stacked_imbalances.push_back(cell);
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
        // Mark data as dirty to trigger immediate rendering update
        data_dirty_.store(true, std::memory_order_release);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();

  // Volume data type selector for footprint visualization (16 types)
  const char* volume_data_type_names[] = {
    "Trades", "BuyTrades", "SellTrades", "Volume", "BuyVolume", "SellVolume",
    "BuyVolume%", "SellVolume%", "BuySellVolume", "Delta", "Delta%", "CumulativeDelta",
    "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol"  // 16 types (excluding FilteredVol)
  };

  int current_vol_data_type = static_cast<int>(volume_data_type_);
  if (ImGui::BeginCombo("Footprint Mode", volume_data_type_names[current_vol_data_type])) {
    for (int i = 0; i < 16; i++) {
      bool is_selected = (current_vol_data_type == i);
      if (ImGui::Selectable(volume_data_type_names[i], is_selected)) {
        current_vol_data_type = i;
        volume_data_type_ = static_cast<Data::VolumeDataType>(i);
        // Mark data as dirty to trigger immediate rendering update
        data_dirty_.store(true, std::memory_order_release);
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

    // Get visible plot limits to determine which time bars and price levels are visible
    ImPlotRect limits = ImPlot::GetPlotLimits();
    double x_min = limits.X.Min;
    double x_max = limits.X.Max;
    double y_min = limits.Y.Min;
    double y_max = limits.Y.Max;

    // Calculate max volume across all visible cells for adaptive alpha calculation
    double max_volume = 0.0;
    for (const auto &cluster : clusters) {
      double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
      if (total_vol > max_volume) {
        max_volume = total_vol;
      }
    }

    // Prevent division by zero
    if (max_volume <= 0.0) {
        max_volume = 1.0; // Default to 1 to prevent division by zero
    }

    // Collect all cells for imbalance detection
    std::vector<FootprintCell> all_cells;

    // Iterate through existing clusters and render them
    // In a real implementation, we would iterate through visible time bars and price levels
    // and retrieve ClusterCell data from the cluster engine
    for (const auto &cluster : clusters) {
      // Convert cluster to footprint cell
      FootprintCell cell(
          cluster.centerX,           // x (time)
          cluster.centerY,           // y (price)
          cluster.width,            // width (time duration)
          cluster.height,           // height (price range)
          static_cast<double>(cluster.bidVolume),        // bid_volume
          static_cast<double>(cluster.askVolume),        // ask_volume
          cluster.tradeCount,       // trade_count
          cluster.vwap             // vwap
      );

      // Apply volume analysis based on active VolumeAnalysisType
      // This is where we switch on the active VolumeAnalysisType to determine displayed value
      double value_for_display = 0.0;

      switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
          case BTQuant::Data::VolumeAnalysisType::Trades:
              value_for_display = static_cast<double>(cluster.tradeCount);
              break;

          case BTQuant::Data::VolumeAnalysisType::BuyTrades:
              // For this case, we'd need buy trade count from the cluster engine
              value_for_display = static_cast<double>(cluster.tradeCount) * 0.6; // Placeholder
              break;

          case BTQuant::Data::VolumeAnalysisType::SellTrades:
              // For this case, we'd need sell trade count from the cluster engine
              value_for_display = static_cast<double>(cluster.tradeCount) * 0.4; // Placeholder
              break;

          case BTQuant::Data::VolumeAnalysisType::Volume:
              value_for_display = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::BuyVolume:
              value_for_display = static_cast<double>(cluster.bidVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::SellVolume:
              value_for_display = static_cast<double>(cluster.askVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
              value_for_display = static_cast<double>(cluster.bidVolume - cluster.askVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::Delta:
              value_for_display = static_cast<double>(cluster.bidVolume - cluster.askVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
              {
                  double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                  value_for_display = total_vol > 0.0 ?
                      (static_cast<double>(cluster.bidVolume - cluster.askVolume) / total_vol) * 100.0 : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
              // For demo purposes, use the same calculation as Delta
              value_for_display = static_cast<double>(cluster.bidVolume - cluster.askVolume);
              break;

          case BTQuant::Data::VolumeAnalysisType::AverageSize:
              {
                  int total_count = cluster.tradeCount;
                  double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                  value_for_display = total_count > 0 ? total_vol / static_cast<double>(total_count) : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
              {
                  int buy_count = static_cast<int>(cluster.tradeCount * 0.6); // Placeholder
                  value_for_display = buy_count > 0 ?
                      static_cast<double>(cluster.bidVolume) / static_cast<double>(buy_count) : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
              {
                  int sell_count = static_cast<int>(cluster.tradeCount * 0.4); // Placeholder
                  value_for_display = sell_count > 0 ?
                      static_cast<double>(cluster.askVolume) / static_cast<double>(sell_count) : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
              // For demo purposes, use a fraction of total volume
              value_for_display = static_cast<double>(cluster.bidVolume + cluster.askVolume) * 0.1; // Placeholder
              break;

          case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
              {
                  double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                  value_for_display = total_vol > 0.0 ?
                      (static_cast<double>(cluster.bidVolume) / total_vol) * 100.0 : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
              {
                  double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                  value_for_display = total_vol > 0.0 ?
                      (static_cast<double>(cluster.askVolume) / total_vol) * 100.0 : 0.0;
              }
              break;

          case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
              // For demo purposes, use total volume
              value_for_display = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              break;
      }

      // Update the cell's values based on the selected analysis type for visualization
      cell.bid_volume = static_cast<double>(cluster.bidVolume);
      cell.ask_volume = static_cast<double>(cluster.askVolume);
      cell.delta = static_cast<double>(cluster.bidVolume - cluster.askVolume);
      cell.trade_count = cluster.tradeCount;

      // Add to all cells for imbalance detection
      all_cells.push_back(cell);
    }

    // Detect imbalances
    std::vector<FootprintCell> diagonal_imbalances;
    std::vector<FootprintCell> stacked_imbalances;
    detectImbalances(all_cells, diagonal_imbalances, stacked_imbalances);

    // Group clusters by time (x-coordinate) to calculate time-bar summaries
    std::map<double, std::vector<const RenderEngine::CandleCluster*>> clusters_by_time;
    for (const auto &cluster : clusters) {
      // Round x to nearest time unit to group clusters by time bar
      double time_key = std::round(cluster.centerX * 10.0) / 10.0; // Adjust precision as needed
      clusters_by_time[time_key].push_back(&cluster);
    }

    // Calculate cumulative delta and other summaries for each time bar
    std::map<double, double> cumulative_deltas;  // Cumulative delta by time
    std::map<double, std::pair<double, double>> poc_info;  // POC price and volume by time
    std::map<double, double> time_bar_net_deltas;  // Net delta by time
    std::map<double, double> time_bar_total_volumes;  // Total volume by time

    double running_cumulative_delta = 0.0;

    for (const auto& [time_key, time_clusters] : clusters_by_time) {
        double time_net_delta = 0.0;
        double time_total_volume = 0.0;
        double max_volume_in_time_bar = 0.0;
        double poc_price = 0.0;

        for (const auto* cluster : time_clusters) {
            double cluster_delta = static_cast<double>(cluster->bidVolume - cluster->askVolume);
            time_net_delta += cluster_delta;
            time_total_volume += static_cast<double>(cluster->bidVolume + cluster->askVolume);

            double cluster_total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            if (cluster_total_vol > max_volume_in_time_bar) {
                max_volume_in_time_bar = cluster_total_vol;
                poc_price = cluster->centerY;  // Price with highest volume in this time bar
            }
        }

        running_cumulative_delta += time_net_delta;
        cumulative_deltas[time_key] = running_cumulative_delta;
        time_bar_net_deltas[time_key] = time_net_delta;
        time_bar_total_volumes[time_key] = time_total_volume;
        poc_info[time_key] = std::make_pair(poc_price, max_volume_in_time_bar);
    }

    // Render all cells with imbalance highlighting
    for (const auto &cell : all_cells) {
      // Render the cell with the calculated max volume for adaptive alpha
      renderCell(cell, draw_list, max_volume, diagonal_imbalances, stacked_imbalances);
    }

    // Render header summaries above each time bar
    for (const auto& [time_key, time_clusters] : clusters_by_time) {
        // Get the summary information for this time bar
        double total_volume = time_bar_total_volumes[time_key];
        double net_delta = time_bar_net_deltas[time_key];
        double cumulative_delta = cumulative_deltas[time_key];
        double poc_price = poc_info[time_key].first;

        // Convert time to pixel coordinates for header positioning
        ImVec2 header_pos = ImPlot::PlotToPixels(time_key, y_max + 5.0); // Position header slightly above the highest price

        // Format the header text
        char header_text[256];
        snprintf(header_text, sizeof(header_text),
                 "Vol:%.0f D:%+.0f CD:%+.0f POC:%.2f",
                 total_volume, net_delta, cumulative_delta, poc_price);

        // Use monospace font for alignment
        ImFont* mono_font = nullptr;
        // First, try to find a monospace font by name
        for (int i = 0; i < ImGui::GetIO().Fonts->Fonts.Size; i++) {
            const char* font_name = ImGui::GetIO().Fonts->Fonts[i]->GetDebugName();
            if (font_name && (strstr(font_name, "Mono") != nullptr ||
                             strstr(font_name, "Consolas") != nullptr ||
                             strstr(font_name, "Courier") != nullptr)) {
                mono_font = ImGui::GetIO().Fonts->Fonts[i];
                break;
            }
        }

        // If no monospace font found, try to use the default font
        if (mono_font) {
            ImGui::PushFont(mono_font);
        }

        // Calculate text size for background rectangle
        ImVec2 text_size = ImGui::CalcTextSize(header_text);

        // Draw background rectangle for header
        draw_list->AddRectFilled(
            ImVec2(header_pos.x - text_size.x/2.0f, header_pos.y - text_size.y - 2.0f),
            ImVec2(header_pos.x + text_size.x/2.0f, header_pos.y + 2.0f),
            IM_COL32(30, 30, 40, 220)); // Dark semi-transparent background with border

        // Draw border around the header
        draw_list->AddRect(
            ImVec2(header_pos.x - text_size.x/2.0f, header_pos.y - text_size.y - 2.0f),
            ImVec2(header_pos.x + text_size.x/2.0f, header_pos.y + 2.0f),
            IM_COL32(100, 100, 150, 200)); // Border color

        // Draw the header text
        if (mono_font) {
            // When using PushFont/PopFont, AddText uses the current font automatically
            draw_list->AddText(
                ImVec2(header_pos.x - text_size.x/2.0f, header_pos.y - text_size.y),
                IM_COL32(255, 255, 255, 255), // White text
                header_text);
        } else {
            draw_list->AddText(
                ImVec2(header_pos.x - text_size.x/2.0f, header_pos.y - text_size.y),
                IM_COL32(255, 255, 255, 255), // White text
                header_text);
        }

        if (mono_font) {
            ImGui::PopFont();
        }
    }

    ImPlot::EndPlot();
  }

  // Calculate footer statistics: number of trades, average trade size, max single trade
  int total_trades = 0;
  double total_volume = 0.0;
  double max_single_trade_volume = 0.0;

  for (const auto &c : clusters) {
    total_trades += c.tradeCount;
    double cluster_total_volume = static_cast<double>(c.bidVolume + c.askVolume);
    total_volume += cluster_total_volume;

    // Update max single trade volume if this cluster has a larger volume
    if (cluster_total_volume > max_single_trade_volume) {
      max_single_trade_volume = cluster_total_volume;
    }
  }

  // Calculate average trade size
  double avg_trade_size = (total_trades > 0) ? total_volume / static_cast<double>(total_trades) : 0.0;

  // Render footer with smaller font below the cluster grid
  if (!clusters.empty()) {
    // Create a separator line above the footer
    ImGui::Separator();

    // Temporarily reduce font size for the footer using text scaling
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 3.0f));
    ImGui::Text("Trades: %d | Avg Size: %.2f | Max Trade: %.2f",
                total_trades, avg_trade_size, max_single_trade_volume);
    ImGui::PopStyleVar(2);
  }

  // Enhanced Debug Overlay
  if (!clusters.empty()) {
    // Determine the current volume data type name for display
    const char* vol_type_names[] = {
      "Trades", "BuyTrades", "SellTrades", "Volume", "BuyVolume", "SellVolume",
      "BuyVol%", "SellVol%", "BuySellVol", "Delta", "Delta%", "CumulativeDelta",
      "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol"  // 16 types (excluding FilteredVol)
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

      switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
        case Data::VolumeAnalysisType::Trades:
          total_value += c.tradeCount;
          break;

        case Data::VolumeAnalysisType::Volume:
          total_value += c.bidVolume + c.askVolume;
          break;

        case Data::VolumeAnalysisType::BuyVolume:
          total_value += c.bidVolume;
          break;

        case Data::VolumeAnalysisType::SellVolume:
          total_value += c.askVolume;
          break;

        case Data::VolumeAnalysisType::BuySellVolume:
          total_value += c.bidVolume - c.askVolume;
          break;

        case Data::VolumeAnalysisType::Delta:
          total_value += c.bidVolume - c.askVolume;
          break;

        case Data::VolumeAnalysisType::DeltaPercent:
          {
            double max_vol = std::max(c.bidVolume, c.askVolume);
            total_value += max_vol > 0.0 ? ((c.bidVolume - c.askVolume) / max_vol) * 100.0 : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::BuyTrades:
          total_value += c.tradeCount * 0.6; // Placeholder
          break;

        case Data::VolumeAnalysisType::SellTrades:
          total_value += c.tradeCount * 0.4; // Placeholder
          break;

        case Data::VolumeAnalysisType::BuyVolumePercent:
          {
            double total_vol = c.bidVolume + c.askVolume;
            total_value += total_vol > 0.0 ? (c.bidVolume / total_vol) * 100.0 : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::SellVolumePercent:
          {
            double total_vol = c.bidVolume + c.askVolume;
            total_value += total_vol > 0.0 ? (c.askVolume / total_vol) * 100.0 : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::CumulativeDelta:
          total_value += c.bidVolume - c.askVolume; // Same as delta for demo
          break;

        case Data::VolumeAnalysisType::AverageSize:
          {
            int count = c.tradeCount;
            double vol = c.bidVolume + c.askVolume;
            total_value += count > 0 ? vol / static_cast<double>(count) : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::AverageBuySize:
          {
            int buy_count = static_cast<int>(c.tradeCount * 0.6); // Placeholder
            total_value += buy_count > 0 ? c.bidVolume / static_cast<double>(buy_count) : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::AverageSellSize:
          {
            int sell_count = static_cast<int>(c.tradeCount * 0.4); // Placeholder
            total_value += sell_count > 0 ? c.askVolume / static_cast<double>(sell_count) : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::MaxOneTradeVolume:
          total_value += (c.bidVolume + c.askVolume) * 0.1; // Placeholder
          break;

        case Data::VolumeAnalysisType::FilteredVolume:
          total_value += c.bidVolume + c.askVolume; // Same as total volume for demo
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
