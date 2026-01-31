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
#include <sstream>
#include <iomanip>

namespace BTQuant {

// Helper function to format numbers according to the selected format
std::string FootprintPanel::formatNumber(double value, NumberFormat format, int decimal_places) {
  std::ostringstream oss;

  switch (format) {
    case NumberFormat::Raw:
      oss << std::fixed << std::setprecision(decimal_places) << value;
      break;

    case NumberFormat::ThousandsK:
      if (std::abs(value) >= 1000000.0) {
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000.0) << "M";
      } else if (std::abs(value) >= 1000.0) {
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1000.0) << "K";
      } else {
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::MillionsM:
      if (std::abs(value) >= 1000000.0) {
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1000000.0) << "M";
      } else if (std::abs(value) >= 1000.0) {
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1000.0) << "K";
      } else {
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::Scientific:
      oss << std::scientific << std::setprecision(decimal_places) << value;
      break;

    case NumberFormat::CustomDecimal:
      oss << std::fixed << std::setprecision(decimal_places) << value;
      break;
  }

  return oss.str();
}

FootprintPanel::FootprintPanel(
    const PanelConfig &config,
    RenderEngine::MarketMicrostructureRenderer *renderer)
    : PanelBase(config), renderer_(renderer), data_type_(Data::UnifiedDataPipeline::DataType::FOOTPRINT),
      volume_data_type_(Data::VolumeDataType::Delta),
      time_aggregation_type_(Data::TimeAggregationType::T_1MIN) {}

void FootprintPanel::update(float dt) {
  // Update logic if needed
  // Aggregation is handled by MarketMicrostructureRenderer
}

ImU32 FootprintPanel::getCellColor(const FootprintCell& cell, double max_volume) const {
  // Calculate values based on selected volume data type
  double value_to_display = 0.0;
  double total_vol = cell.bid_volume + cell.ask_volume;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Delta:
    case Data::VolumeAnalysisType::DeltaPercent:
    case Data::VolumeAnalysisType::CumulativeDelta:
      // Green-red gradient for delta
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
        normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        if (std::abs(normalized_delta) > delta_threshold_) {
          // Use different colors based on the sign of the delta
          if (normalized_delta > 0) {
            // Positive delta - Green gradient
            float green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(0),
                static_cast<int>(50 + 205 * green_intensity),
                static_cast<int>(0),
                static_cast<int>(alpha * 255));
          } else {
            // Negative delta - Red gradient
            float red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(50 + 205 * red_intensity),
                static_cast<int>(0),
                static_cast<int>(0),
                static_cast<int>(alpha * 255));
          }
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

    case Data::VolumeAnalysisType::BuyVolume:
    case Data::VolumeAnalysisType::SellVolume:
    case Data::VolumeAnalysisType::BuySellVolume:
      // Blue-red gradient for buy/sell volume
      {
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        double normalized_value = 0.0;

        if (volume_data_type_ == Data::VolumeAnalysisType::BuyVolume) {
          normalized_value = max_vol > 0.0 ? cell.bid_volume / max_vol : 0.0;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellVolume) {
          normalized_value = max_vol > 0.0 ? cell.ask_volume / max_vol : 0.0;
        } else { // BuySellVolume
          normalized_value = max_vol > 0.0 ? (cell.bid_volume - cell.ask_volume) / max_vol : 0.0;
        }

        normalized_value = std::clamp(normalized_value, -1.0, 1.0);

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        if (std::abs(normalized_value) > delta_threshold_) {
          if (normalized_value > 0) {
            // Positive - Blue gradient
            float blue_intensity = std::clamp(static_cast<float>(normalized_value), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(0),
                static_cast<int>(0),
                static_cast<int>(50 + 205 * blue_intensity),
                static_cast<int>(alpha * 255));
          } else {
            // Negative - Red gradient
            float red_intensity = std::clamp(static_cast<float>(-normalized_value), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(50 + 205 * red_intensity),
                static_cast<int>(0),
                static_cast<int>(0),
                static_cast<int>(alpha * 255));
          }
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

    case Data::VolumeAnalysisType::Volume:
    case Data::VolumeAnalysisType::BuyVolumePercent:
    case Data::VolumeAnalysisType::SellVolumePercent:
    case Data::VolumeAnalysisType::Trades:
    case Data::VolumeAnalysisType::BuyTrades:
    case Data::VolumeAnalysisType::SellTrades:
    case Data::VolumeAnalysisType::FilteredVolume:
      // Yellow-orange gradient for volume intensity
      {
        double normalized_value = 0.0;
        double max_possible_value = 0.0;

        if (volume_data_type_ == Data::VolumeAnalysisType::Volume) {
          normalized_value = total_vol;
          max_possible_value = max_volume;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::BuyVolumePercent) {
          double total_vol_local = cell.bid_volume + cell.ask_volume;
          normalized_value = total_vol_local > 0.0 ? (cell.bid_volume / total_vol_local) * 100.0 : 0.0;
          max_possible_value = 100.0;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellVolumePercent) {
          double total_vol_local = cell.bid_volume + cell.ask_volume;
          normalized_value = total_vol_local > 0.0 ? (cell.ask_volume / total_vol_local) * 100.0 : 0.0;
          max_possible_value = 100.0;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::Trades) {
          normalized_value = static_cast<double>(cell.trade_count);
          // Estimate max possible trades based on max volume
          max_possible_value = 1000.0; // Placeholder - in a real scenario, this would come from stats
        } else if (volume_data_type_ == Data::VolumeAnalysisType::BuyTrades) {
          // Estimate buy trades based on volume ratio
          double total_vol_local = cell.bid_volume + cell.ask_volume;
          if (total_vol_local > 0) {
            double buy_ratio = cell.bid_volume / total_vol_local;
            normalized_value = static_cast<double>(static_cast<int>(cell.trade_count * buy_ratio));
          } else {
            normalized_value = static_cast<double>(static_cast<int>(cell.trade_count * 0.5)); // Equal split if no volume
          }
          max_possible_value = 600.0; // Placeholder
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellTrades) {
          // Estimate sell trades based on volume ratio
          double total_vol_local = cell.bid_volume + cell.ask_volume;
          if (total_vol_local > 0) {
            double sell_ratio = cell.ask_volume / total_vol_local;
            normalized_value = static_cast<double>(static_cast<int>(cell.trade_count * sell_ratio));
          } else {
            normalized_value = static_cast<double>(static_cast<int>(cell.trade_count * 0.5)); // Equal split if no volume
          }
          max_possible_value = 400.0; // Placeholder
        } else { // FilteredVolume
          normalized_value = total_vol;
          max_possible_value = max_volume;
        }

        float intensity = max_possible_value > 0.0 ?
            std::clamp(static_cast<float>(normalized_value / max_possible_value), 0.0f, 1.0f) : 0.0f;

        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        // Yellow-orange gradient for volume intensity
        return IM_COL32(
            static_cast<int>(255 * intensity),
            static_cast<int>(165 * intensity),
            static_cast<int>(0),
            static_cast<int>(alpha * 255));
      }
      break;

    case Data::VolumeAnalysisType::AverageSize:
    case Data::VolumeAnalysisType::AverageBuySize:
    case Data::VolumeAnalysisType::AverageSellSize:
    case Data::VolumeAnalysisType::MaxOneTradeVolume:
    default:
      // For other metrics, use blue gradient
      {
        // Calculate adaptive alpha based on total volume relative to max possible volume
        float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.1f, 1.0f) : 0.5f;

        // Calculate normalized value for the specific metric
        double normalized_value = 0.0;
        double max_possible_value = 10000.0; // Default placeholder

        switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
          case Data::VolumeAnalysisType::AverageSize:
            {
              int total_count = cell.trade_count;
              normalized_value = total_count > 0 ? (cell.bid_volume + cell.ask_volume) / static_cast<double>(total_count) : 0.0;
            }
            break;

          case Data::VolumeAnalysisType::AverageBuySize:
            {
              // Estimate buy count based on volume ratio
              double total_vol_local = cell.bid_volume + cell.ask_volume;
              int buy_count = cell.trade_count; // Start with total count
              if (total_vol_local > 0) {
                double buy_ratio = cell.bid_volume / total_vol_local;
                buy_count = static_cast<int>(static_cast<double>(cell.trade_count) * buy_ratio);
              }
              normalized_value = buy_count > 0 ? cell.bid_volume / static_cast<double>(buy_count) : 0.0;
            }
            break;

          case Data::VolumeAnalysisType::AverageSellSize:
            {
              // Estimate sell count based on volume ratio
              double total_vol_local = cell.bid_volume + cell.ask_volume;
              int sell_count = cell.trade_count; // Start with total count
              if (total_vol_local > 0) {
                double sell_ratio = cell.ask_volume / total_vol_local;
                sell_count = static_cast<int>(static_cast<double>(cell.trade_count) * sell_ratio);
              }
              normalized_value = sell_count > 0 ? cell.ask_volume / static_cast<double>(sell_count) : 0.0;
            }
            break;

          case Data::VolumeAnalysisType::MaxOneTradeVolume:
            // Estimate max single trade volume as total volume divided by trade count
            normalized_value = cell.trade_count > 0 ?
                (cell.bid_volume + cell.ask_volume) / static_cast<double>(cell.trade_count) : 0.0;
            break;

          default:
            normalized_value = cell.bid_volume + cell.ask_volume;
            break;
        }

        float blue_intensity = std::clamp(static_cast<float>(normalized_value / max_possible_value), 0.0f, 1.0f);
        return IM_COL32(
            static_cast<int>(0),
            static_cast<int>(0),
            static_cast<int>(50 + 205 * blue_intensity),
            static_cast<int>(alpha * 255));
      }
      break;
  }
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  // Calculate values based on selected volume data type
  double value_to_display = 0.0;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Trades:
      // For trade counts, we'll use the number formatting
      return formatNumber(static_cast<double>(cell.trade_count), number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::BuyTrades:
      // For buy trade counts, estimate based on volume ratio
      {
        double total_vol = cell.bid_volume + cell.ask_volume;
        if (total_vol > 0) {
          double buy_ratio = cell.bid_volume / total_vol;
          return formatNumber(static_cast<double>(static_cast<int>(cell.trade_count * buy_ratio)), number_format_, custom_decimal_places_);
        } else {
          return formatNumber(static_cast<double>(static_cast<int>(cell.trade_count * 0.5)), number_format_, custom_decimal_places_);
        }
      }

    case Data::VolumeAnalysisType::SellTrades:
      // For sell trade counts, estimate based on volume ratio
      {
        double total_vol = cell.bid_volume + cell.ask_volume;
        if (total_vol > 0) {
          double sell_ratio = cell.ask_volume / total_vol;
          return formatNumber(static_cast<double>(static_cast<int>(cell.trade_count * sell_ratio)), number_format_, custom_decimal_places_);
        } else {
          return formatNumber(static_cast<double>(static_cast<int>(cell.trade_count * 0.5)), number_format_, custom_decimal_places_);
        }
      }

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
        // Estimate buy count based on volume ratio
        double total_vol = cell.bid_volume + cell.ask_volume;
        int buy_count = cell.trade_count; // Start with total count
        if (total_vol > 0) {
          double buy_ratio = cell.bid_volume / total_vol;
          buy_count = static_cast<int>(static_cast<double>(cell.trade_count) * buy_ratio);
        }
        value_to_display = buy_count > 0 ? cell.bid_volume / static_cast<double>(buy_count) : 0.0;
        break;
      }

    case Data::VolumeAnalysisType::AverageSellSize:
      {
        // Estimate sell count based on volume ratio
        double total_vol = cell.bid_volume + cell.ask_volume;
        int sell_count = cell.trade_count; // Start with total count
        if (total_vol > 0) {
          double sell_ratio = cell.ask_volume / total_vol;
          sell_count = static_cast<int>(static_cast<double>(cell.trade_count) * sell_ratio);
        }
        value_to_display = sell_count > 0 ? cell.ask_volume / static_cast<double>(sell_count) : 0.0;
        break;
      }

    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      // Estimate max single trade volume as total volume divided by trade count
      value_to_display = cell.trade_count > 0 ?
          (cell.bid_volume + cell.ask_volume) / static_cast<double>(cell.trade_count) : 0.0;
      break;

    case Data::VolumeAnalysisType::FilteredVolume:
      value_to_display = cell.bid_volume + cell.ask_volume; // Using same as total volume for demo
      break;

    default:
      // For other types, default to showing the larger of bid/ask volume
      value_to_display = std::max(cell.bid_volume, cell.ask_volume);
      break;
  }

  // Format the number using the selected formatting option
  return formatNumber(value_to_display, number_format_, custom_decimal_places_);
}

std::string FootprintPanel::getCellTooltip(const FootprintCell& cell) const {
  // Calculate delta percent
  double max_vol = std::max(cell.bid_volume, cell.ask_volume);
  double delta_percent = max_vol > 0.0 ? (cell.delta / max_vol) * 100.0 : 0.0;

  // For buy/sell trades, we'll estimate based on volume ratios
  // In a real implementation, we'd need separate counters for buy/sell trades
  double total_vol = cell.bid_volume + cell.ask_volume;
  int buy_trades = 0;
  int sell_trades = 0;

  if (total_vol > 0) {
    buy_trades = static_cast<int>(cell.trade_count * (cell.bid_volume / total_vol));
    sell_trades = cell.trade_count - buy_trades;
  } else {
    // If no volume, split trades equally
    buy_trades = cell.trade_count / 2;
    sell_trades = cell.trade_count - buy_trades;
  }

  // For max single trade, we'll estimate as a portion of total volume
  // In a real implementation, we'd need the actual max trade size
  double max_single_trade = total_vol > 0 ? total_vol / cell.trade_count : 0.0;

  // Format the tooltip text with all required information
  std::string tooltip = std::format(
    "Buy Volume: {:.2f}\n"
    "Sell Volume: {:.2f}\n"
    "Delta: {:.2f}\n"
    "Delta %: {:.2f}%\n"
    "Buy Trades: {}\n"
    "Sell Trades: {}\n"
    "Max Single Trade: {:.2f}\n"
    "Timestamp Range: {:.2f}-{:.2f}",
    cell.bid_volume,
    cell.ask_volume,
    cell.delta,
    delta_percent,
    buy_trades,
    sell_trades,
    max_single_trade,
    cell.x - cell.width/2.0, // Start time
    cell.x + cell.width/2.0  // End time
  );

  return tooltip;
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

  // Volume data type selector for footprint visualization (16 types as requested)
  const char* volume_data_type_names[] = {
    "Trades", "BuyTrades", "SellTrades", "Volume", "BuyVolume", "SellVolume",
    "BuyVol%", "SellVol%", "BuySellVol", "Delta", "Delta%", "CumulDelta",
    "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol"
  };

  int current_vol_data_type = static_cast<int>(volume_data_type_);
  if (ImGui::BeginCombo("Footprint Mode", volume_data_type_names[current_vol_data_type])) {
    for (int i = 0; i < 16; i++) {  // 16 types as requested
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

  // Time aggregation selector
  const char* time_agg_names[] = {
    "1min", "5min", "15min", "30min", "1hour", "2hour", "4hour", "Volume-based", "Tick-based"
  };

  int current_time_agg = static_cast<int>(time_aggregation_type_);
  if (ImGui::BeginCombo("Time Agg", time_agg_names[current_time_agg])) {
    for (int i = 0; i < 9; i++) {
      bool is_selected = (current_time_agg == i);
      if (ImGui::Selectable(time_agg_names[i], is_selected)) {
        current_time_agg = i;
        time_aggregation_type_ = static_cast<Data::TimeAggregationType>(i);
        // Mark data as dirty to trigger immediate rendering update
        data_dirty_.store(true, std::memory_order_release);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Price aggregation selector
  const char* price_agg_names[] = {
    "1 Tick", "5 Ticks", "10 Ticks", "0.1%", "0.5%", "1%", "Custom"
  };

  int current_price_agg = static_cast<int>(price_aggregation_type_);
  if (ImGui::BeginCombo("Price Agg", price_agg_names[current_price_agg])) {
    for (int i = 0; i < 7; i++) {
      bool is_selected = (current_price_agg == i);
      if (ImGui::Selectable(price_agg_names[i], is_selected)) {
        current_price_agg = i;
        price_aggregation_type_ = static_cast<Data::PriceAggregationType>(i);
        // Mark data as dirty to trigger immediate rendering update
        data_dirty_.store(true, std::memory_order_release);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Show custom value input if custom price aggregation is selected
  if (price_aggregation_type_ == Data::PriceAggregationType::P_CUSTOM) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    double temp_custom_value = custom_price_aggregation_value_;
    if (ImGui::InputDouble("##CustomPriceAgg", &temp_custom_value, 0.01, 0.1, "%.4f")) {
      custom_price_aggregation_value_ = temp_custom_value;
      // Mark data as dirty to trigger immediate rendering update
      data_dirty_.store(true, std::memory_order_release);
    }
  }

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

  // Number formatting options
  ImGui::Separator();
  ImGui::Text("Number Formatting:");
  ImGui::SameLine();

  // Combo box for number format selection
  const char* format_items[] = { "Raw", "K (Thousands)", "M (Millions)", "Scientific", "Custom Decimal" };
  int current_format = static_cast<int>(number_format_);
  if (ImGui::BeginCombo("Format", format_items[current_format])) {
    for (int i = 0; i < 5; i++) {
      bool is_selected = (current_format == i);
      if (ImGui::Selectable(format_items[i], is_selected)) {
        current_format = i;
        number_format_ = static_cast<NumberFormat>(i);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Slider for custom decimal places (only shown when Custom Decimal is selected)
  if (number_format_ == NumberFormat::CustomDecimal) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderInt("Decimals", &custom_decimal_places_, 0, 6);
  }

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

    // Iterate through visible time bars and price levels to find max volume
    for (const auto &cluster : clusters) {
        // Check if cluster is within visible bounds
        if (cluster.centerX >= x_min && cluster.centerX <= x_max &&
            cluster.centerY >= y_min && cluster.centerY <= y_max) {

            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            if (total_vol > max_volume) {
                max_volume = total_vol;
            }
        }
    }

    // Prevent division by zero
    if (max_volume <= 0.0) {
        max_volume = 1.0; // Default to 1 to prevent division by zero
    }

    // Collect all visible cells for imbalance detection
    std::vector<FootprintCell> all_cells;

    // MAIN RENDER LOOP: Iterate through visible time bars and price levels
    // Retrieve ClusterCell data and switch on active VolumeAnalysisType to determine displayed value
    for (const auto &cluster : clusters) {
        // Check if cluster is within visible bounds
        if (cluster.centerX >= x_min && cluster.centerX <= x_max &&
            cluster.centerY >= y_min && cluster.centerY <= y_max) {

            // Convert cluster to footprint cell
            FootprintCell cell(
                cluster.centerX,           // x (time)
                cluster.centerY,           // y (price)
                cluster.width,             // width (time duration)
                cluster.height,            // height (price range)
                static_cast<double>(cluster.bidVolume),        // bid_volume
                static_cast<double>(cluster.askVolume),        // ask_volume
                cluster.tradeCount,        // trade_count
                cluster.vwap               // vwap
            );

            // Update the cell's values based on the selected analysis type for visualization
            cell.bid_volume = static_cast<double>(cluster.bidVolume);
            cell.ask_volume = static_cast<double>(cluster.askVolume);
            cell.delta = static_cast<double>(cluster.bidVolume - cluster.askVolume);
            cell.trade_count = cluster.tradeCount;

            // Add to all cells for imbalance detection
            all_cells.push_back(cell);
        }
    }

    // Detect imbalances
    std::vector<FootprintCell> diagonal_imbalances;
    std::vector<FootprintCell> stacked_imbalances;
    detectImbalances(all_cells, diagonal_imbalances, stacked_imbalances);

    // Group only visible clusters by time (x-coordinate) to calculate time-bar summaries
    std::map<double, std::vector<const RenderEngine::CandleCluster*>> clusters_by_time;
    for (const auto &cluster : clusters) {
        // Check if cluster is within visible bounds before grouping
        if (cluster.centerX >= x_min && cluster.centerX <= x_max &&
            cluster.centerY >= y_min && cluster.centerY <= y_max) {
            // Round x to nearest time unit to group clusters by time bar
            double time_key = std::round(cluster.centerX * 10.0) / 10.0; // Adjust precision as needed
            clusters_by_time[time_key].push_back(&cluster);
        }
    }

    // Calculate cumulative and other summaries for each time bar based on active VolumeAnalysisType
    std::map<double, double> cumulative_values;  // Cumulative value by time based on active analysis type
    std::map<double, std::pair<double, double>> poc_info;  // POC price and volume by time
    std::map<double, double> time_bar_net_values;  // Net value by time based on active analysis type
    std::map<double, double> time_bar_total_values;  // Total value by time based on active analysis type

    double running_cumulative_value = 0.0;

    for (const auto& [time_key, time_clusters] : clusters_by_time) {
        double time_net_value = 0.0;
        double time_total_value = 0.0;
        double max_volume_in_time_bar = 0.0;
        double poc_price = 0.0;
        double max_analysis_value_in_time_bar = 0.0; // Track max value for POC based on analysis type

        for (const auto* cluster : time_clusters) {
            // Calculate value based on active VolumeAnalysisType
            double cluster_value = 0.0;

            switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
                case BTQuant::Data::VolumeAnalysisType::Trades:
                    cluster_value = static_cast<double>(cluster->tradeCount);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyTrades:
                    // Estimate buy trades based on volume ratio
                    {
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        if (total_vol > 0) {
                            double buy_ratio = static_cast<double>(cluster->bidVolume) / total_vol;
                            cluster_value = static_cast<double>(cluster->tradeCount) * buy_ratio;
                        } else {
                            cluster_value = static_cast<double>(cluster->tradeCount) * 0.5; // Equal split if no volume
                        }
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellTrades:
                    // Estimate sell trades based on volume ratio
                    {
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        if (total_vol > 0) {
                            double sell_ratio = static_cast<double>(cluster->askVolume) / total_vol;
                            cluster_value = static_cast<double>(cluster->tradeCount) * sell_ratio;
                        } else {
                            cluster_value = static_cast<double>(cluster->tradeCount) * 0.5; // Equal split if no volume
                        }
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::Volume:
                    cluster_value = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolume:
                    cluster_value = static_cast<double>(cluster->bidVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolume:
                    cluster_value = static_cast<double>(cluster->askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
                    cluster_value = static_cast<double>(cluster->bidVolume - cluster->askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::Delta:
                    cluster_value = static_cast<double>(cluster->bidVolume - cluster->askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
                    {
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        cluster_value = total_vol > 0.0 ?
                            (static_cast<double>(cluster->bidVolume - cluster->askVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
                    cluster_value = static_cast<double>(cluster->bidVolume - cluster->askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSize:
                    {
                        int total_count = cluster->tradeCount;
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        cluster_value = total_count > 0 ? total_vol / static_cast<double>(total_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
                    {
                        // Estimate buy count based on volume ratio
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        int buy_count = cluster->tradeCount; // Start with total count
                        if (total_vol > 0) {
                            double buy_ratio = static_cast<double>(cluster->bidVolume) / total_vol;
                            buy_count = static_cast<int>(static_cast<double>(cluster->tradeCount) * buy_ratio);
                        }
                        cluster_value = buy_count > 0 ?
                            static_cast<double>(cluster->bidVolume) / static_cast<double>(buy_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
                    {
                        // Estimate sell count based on volume ratio
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        int sell_count = cluster->tradeCount; // Start with total count
                        if (total_vol > 0) {
                            double sell_ratio = static_cast<double>(cluster->askVolume) / total_vol;
                            sell_count = static_cast<int>(static_cast<double>(cluster->tradeCount) * sell_ratio);
                        }
                        cluster_value = sell_count > 0 ?
                            static_cast<double>(cluster->askVolume) / static_cast<double>(sell_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
                    // Estimate max single trade volume as total volume divided by trade count
                    cluster_value = cluster->tradeCount > 0 ?
                        (static_cast<double>(cluster->bidVolume + cluster->askVolume) / static_cast<double>(cluster->tradeCount)) : 0.0;
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
                    {
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        cluster_value = total_vol > 0.0 ?
                            (static_cast<double>(cluster->bidVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
                    {
                        double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                        cluster_value = total_vol > 0.0 ?
                            (static_cast<double>(cluster->askVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
                    cluster_value = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                    break;

                default:
                    cluster_value = static_cast<double>(cluster->bidVolume + cluster->askVolume);
                    break;
            }

            // Update time bar summary values
            time_net_value += cluster_value;
            time_total_value += cluster_value;

            // Track max volume for POC calculation (still based on total volume)
            double cluster_total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            if (cluster_total_vol > max_volume_in_time_bar) {
                max_volume_in_time_bar = cluster_total_vol;
                poc_price = cluster->centerY;  // Price with highest volume in this time bar
            }

            // Also track max analysis value for potential POC based on analysis type
            if (std::abs(cluster_value) > std::abs(max_analysis_value_in_time_bar)) {
                max_analysis_value_in_time_bar = cluster_value;
                poc_price = cluster->centerY;  // Update POC if this cluster has higher analysis value
            }
        }

        running_cumulative_value += time_net_value;
        cumulative_values[time_key] = running_cumulative_value;
        time_bar_net_values[time_key] = time_net_value;
        time_bar_total_values[time_key] = time_total_value;
        poc_info[time_key] = std::make_pair(poc_price, max_volume_in_time_bar);
    }

    // Render all cells with imbalance highlighting
    for (const auto &cell : all_cells) {
      // Render the cell with the calculated max volume for adaptive alpha
      renderCell(cell, draw_list, max_volume, diagonal_imbalances, stacked_imbalances);
    }

    // Handle tooltip for the cell under the mouse cursor
    if (ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse_pos_plot = ImPlot::GetPlotMousePos();

      // Find the cell under the mouse cursor
      for (const auto &cell : all_cells) {
        double x1 = cell.x - cell.width * 0.48;
        double x2 = cell.x + cell.width * 0.48;
        double y1 = cell.y - cell.height * 0.48;
        double y2 = cell.y + cell.height * 0.48;

        if (mouse_pos_plot.x >= x1 && mouse_pos_plot.x <= x2 &&
            mouse_pos_plot.y >= y1 && mouse_pos_plot.y <= y2) {
          // Show tooltip for this cell
          ImGui::SetTooltip("%s", getCellTooltip(cell).c_str());
          break; // Only show tooltip for the first cell found under cursor
        }
      }
    }

    // Render header summaries above each visible time bar
    for (const auto& [time_key, time_clusters] : clusters_by_time) {
        // Only render headers for time bars that are actually visible
        if (time_key >= x_min && time_key <= x_max) {
            // Get the summary information for this time bar based on active VolumeAnalysisType
            double total_value = time_bar_total_values[time_key];
            double net_value = time_bar_net_values[time_key];
            double cumulative_value = cumulative_values[time_key];
            double poc_price = poc_info[time_key].first;

            // Format the header text based on the active VolumeAnalysisType
            char header_text[256];
            const char* analysis_type_label = "";

            switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
                case BTQuant::Data::VolumeAnalysisType::Trades:
                    analysis_type_label = "T";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyTrades:
                    analysis_type_label = "BT";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellTrades:
                    analysis_type_label = "ST";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::Volume:
                    analysis_type_label = "Vol";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolume:
                    analysis_type_label = "BV";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolume:
                    analysis_type_label = "SV";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
                    analysis_type_label = "BSV";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::Delta:
                    analysis_type_label = "D";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
                    analysis_type_label = "DP%";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.1f%% NV:%+.1f%% CV:%+.1f%% POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
                    analysis_type_label = "CD";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSize:
                    analysis_type_label = "AS";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.2f NV:%+.2f CV:%+.2f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
                    analysis_type_label = "ABS";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.2f NV:%+.2f CV:%+.2f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
                    analysis_type_label = "ASS";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.2f NV:%+.2f CV:%+.2f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
                    analysis_type_label = "MOTV";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.2f NV:%+.2f CV:%+.2f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
                    analysis_type_label = "BVP%";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.1f%% NV:%+.1f%% CV:%+.1f%% POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
                    analysis_type_label = "SVP%";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.1f%% NV:%+.1f%% CV:%+.1f%% POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
                    analysis_type_label = "FV";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;

                default:
                    analysis_type_label = "Vol";
                    snprintf(header_text, sizeof(header_text),
                             "%s:%.0f NV:%+.0f CV:%+.0f POC:%.2f",
                             analysis_type_label, total_value, net_value, cumulative_value, poc_price);
                    break;
            }

            // Convert time to pixel coordinates for header positioning
            ImVec2 header_pos = ImPlot::PlotToPixels(time_key, y_max + 5.0); // Position header slightly above the highest price

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
      "BuyVol%", "SellVol%", "BuySellVol", "Delta", "Delta%", "CumulDelta",
      "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol"  // 16 types
    };

    // Time aggregation type names for display
    const char* time_agg_names[] = {
      "1min", "5min", "15min", "30min", "1hour", "2hour", "4hour", "Volume-based", "Tick-based"
    };

    // Price aggregation type names for display
    const char* price_agg_names[] = {
      "1 Tick", "5 Ticks", "10 Ticks", "0.1%", "0.5%", "1%", "Custom"
    };

    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1),
                       "Mode: %s | Time Agg: %s | Price Agg: %s | Clusters: %zu | Grid: %dx%d | Thresh: %.2f",
                       vol_type_names[static_cast<int>(volume_data_type_)],
                       time_agg_names[static_cast<int>(time_aggregation_type_)],
                       price_agg_names[static_cast<int>(price_aggregation_type_)],
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

        case Data::VolumeAnalysisType::BuyTrades:
          // Estimate buy trades based on volume ratio
          {
            double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
            if (total_vol > 0) {
              double buy_ratio = static_cast<double>(c.bidVolume) / total_vol;
              total_value += static_cast<double>(c.tradeCount) * buy_ratio;
            } else {
              total_value += static_cast<double>(c.tradeCount) * 0.5; // Equal split if no volume
            }
          }
          break;

        case Data::VolumeAnalysisType::SellTrades:
          // Estimate sell trades based on volume ratio
          {
            double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
            if (total_vol > 0) {
              double sell_ratio = static_cast<double>(c.askVolume) / total_vol;
              total_value += static_cast<double>(c.tradeCount) * sell_ratio;
            } else {
              total_value += static_cast<double>(c.tradeCount) * 0.5; // Equal split if no volume
            }
          }
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
            // Estimate buy count based on volume ratio
            double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
            int buy_count = c.tradeCount; // Start with total count
            if (total_vol > 0) {
              double buy_ratio = static_cast<double>(c.bidVolume) / total_vol;
              buy_count = static_cast<int>(static_cast<double>(c.tradeCount) * buy_ratio);
            }
            total_value += buy_count > 0 ? static_cast<double>(c.bidVolume) / static_cast<double>(buy_count) : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::AverageSellSize:
          {
            // Estimate sell count based on volume ratio
            double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
            int sell_count = c.tradeCount; // Start with total count
            if (total_vol > 0) {
              double sell_ratio = static_cast<double>(c.askVolume) / total_vol;
              sell_count = static_cast<int>(static_cast<double>(c.tradeCount) * sell_ratio);
            }
            total_value += sell_count > 0 ? static_cast<double>(c.askVolume) / static_cast<double>(sell_count) : 0.0;
          }
          break;

        case Data::VolumeAnalysisType::MaxOneTradeVolume:
          // Estimate max single trade volume as total volume divided by trade count
          total_value = c.tradeCount > 0 ?
              (static_cast<double>(c.bidVolume + c.askVolume) / static_cast<double>(c.tradeCount)) : 0.0;
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
