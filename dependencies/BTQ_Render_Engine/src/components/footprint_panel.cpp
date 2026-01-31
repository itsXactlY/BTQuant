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

  // Calculate adaptive alpha based on cell_volume / max_bar_volume
  float alpha = max_volume > 0.0 ? std::clamp(static_cast<float>(total_vol / max_volume), 0.05f, 1.0f) : 0.05f;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Delta:
    case Data::VolumeAnalysisType::DeltaPercent:
    case Data::VolumeAnalysisType::CumulativeDelta:
      // Green-red gradient for delta (buy/sell imbalance)
      {
        // Use the delta value that was already calculated in the main render loop
        double normalized_delta = cell.delta;

        // For DeltaPercent, normalize the value to [-1, 1] range
        if (volume_data_type_ == Data::VolumeAnalysisType::DeltaPercent) {
            normalized_delta = std::clamp(normalized_delta, -100.0, 100.0) / 100.0;
        } else {
            // For other delta types, normalize based on the sum of volumes
            double max_vol = std::max(cell.bid_volume, cell.ask_volume);
            normalized_delta = max_vol > 0.0 ? cell.delta / max_vol : 0.0;
        }

        normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

        if (std::abs(normalized_delta) > delta_threshold_) {
          // Use different colors based on the sign of the delta
          if (normalized_delta > 0) {
            // Positive delta (buy pressure) - Pure green gradient
            float green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(50 * (1.0f - green_intensity)), // Reduce red as green increases
                static_cast<int>(100 + 155 * green_intensity),  // Full green range
                static_cast<int>(50 * (1.0f - green_intensity)), // Reduce blue as green increases
                static_cast<int>(alpha * 255));
          } else {
            // Negative delta (sell pressure) - Pure red gradient
            float red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(100 + 155 * red_intensity),   // Full red range
                static_cast<int>(50 * (1.0f - red_intensity)), // Reduce green as red increases
                static_cast<int>(50 * (1.0f - red_intensity)), // Reduce blue as red increases
                static_cast<int>(alpha * 255));
          }
        } else {
          // Neutral - Gray with reduced alpha
          return IM_COL32(
              static_cast<int>(100),
              static_cast<int>(100),
              static_cast<int>(100),
              static_cast<int>(alpha * 100)); // Reduced alpha for neutral cells
        }
      }
      break;

    case Data::VolumeAnalysisType::BuyVolume:
    case Data::VolumeAnalysisType::SellVolume:
    case Data::VolumeAnalysisType::BuySellVolume:
      // Blue-red gradient for buy/sell volume comparison
      {
        double normalized_value = 0.0;

        if (volume_data_type_ == Data::VolumeAnalysisType::BuyVolume) {
          // Use bid_volume which was already set in the main render loop
          normalized_value = cell.bid_volume;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellVolume) {
          // Use ask_volume which was already set in the main render loop
          normalized_value = cell.ask_volume;
        } else { // BuySellVolume
          // Use the delta which was already calculated in the main render loop
          normalized_value = cell.delta;
        }

        // Normalize based on max volume
        double max_vol = std::max(cell.bid_volume, cell.ask_volume);
        normalized_value = max_vol > 0.0 ? normalized_value / max_vol : 0.0;
        normalized_value = std::clamp(normalized_value, -1.0, 1.0);

        if (std::abs(normalized_value) > delta_threshold_) {
          if (normalized_value > 0) {
            // Positive - Blue gradient for buy volume dominance
            float blue_intensity = std::clamp(static_cast<float>(normalized_value), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(50 * (1.0f - blue_intensity)), // Reduce red as blue increases
                static_cast<int>(50 * (1.0f - blue_intensity)), // Reduce green as blue increases
                static_cast<int>(100 + 155 * blue_intensity),   // Full blue range
                static_cast<int>(alpha * 255));
          } else {
            // Negative - Red gradient for sell volume dominance
            float red_intensity = std::clamp(static_cast<float>(-normalized_value), 0.0f, 1.0f);
            return IM_COL32(
                static_cast<int>(100 + 155 * red_intensity),   // Full red range
                static_cast<int>(50 * (1.0f - red_intensity)), // Reduce green as red increases
                static_cast<int>(50 * (1.0f - red_intensity)), // Reduce blue as red increases
                static_cast<int>(alpha * 255));
          }
        } else {
          // Neutral - Gray with reduced alpha
          return IM_COL32(
              static_cast<int>(100),
              static_cast<int>(100),
              static_cast<int>(100),
              static_cast<int>(alpha * 100)); // Reduced alpha for neutral cells
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
          // Use the delta which contains the percentage value
          normalized_value = cell.delta;
          max_possible_value = 100.0;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellVolumePercent) {
          // Use the delta which contains the percentage value
          normalized_value = cell.delta;
          max_possible_value = 100.0;
        } else if (volume_data_type_ == Data::VolumeAnalysisType::Trades) {
          // Use bid_volume which was set to trade count in the main render loop
          normalized_value = cell.bid_volume;
          // Estimate max possible trades based on max volume
          max_possible_value = 1000.0; // Placeholder - in a real scenario, this would come from stats
        } else if (volume_data_type_ == Data::VolumeAnalysisType::BuyTrades) {
          // Use bid_volume which was set to buy trades count in the main render loop
          normalized_value = cell.bid_volume;
          max_possible_value = 600.0; // Placeholder
        } else if (volume_data_type_ == Data::VolumeAnalysisType::SellTrades) {
          // Use ask_volume which was set to sell trades count in the main render loop
          normalized_value = cell.ask_volume;
          max_possible_value = 400.0; // Placeholder
        } else { // FilteredVolume
          normalized_value = total_vol;
          max_possible_value = max_volume;
        }

        float intensity = max_possible_value > 0.0 ?
            std::clamp(static_cast<float>(normalized_value / max_possible_value), 0.0f, 1.0f) : 0.0f;

        // Yellow-orange gradient for volume intensity - transitioning from yellow (low intensity) to orange (high intensity)
        // Yellow: high R&G, low B; Orange: high R, medium G, low B
        float red_val = 200.0f + 55.0f * intensity;      // Range: 200-255 (higher for more intensity)
        float green_val = 150.0f + 105.0f * intensity;   // Range: 150-255 (increasing for more intensity)
        float blue_val = 0.0f;                           // Keep blue low for yellow/orange tones

        return IM_COL32(
            static_cast<int>(std::min(255.0f, red_val)),
            static_cast<int>(std::min(255.0f, green_val)),
            static_cast<int>(std::min(255.0f, blue_val)),
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
        // Use the delta value which was already calculated in the main render loop
        double normalized_value = cell.delta;
        double max_possible_value = 10000.0; // Default placeholder

        // Adjust max_possible_value based on the specific analysis type
        switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
          case Data::VolumeAnalysisType::AverageSize:
            max_possible_value = 1000.0; // Reasonable max for average size
            break;

          case Data::VolumeAnalysisType::AverageBuySize:
            max_possible_value = 1000.0; // Reasonable max for average buy size
            break;

          case Data::VolumeAnalysisType::AverageSellSize:
            max_possible_value = 1000.0; // Reasonable max for average sell size
            break;

          case Data::VolumeAnalysisType::MaxOneTradeVolume:
            max_possible_value = max_volume; // Use max volume as reference
            break;

          default:
            normalized_value = cell.bid_volume + cell.ask_volume;
            break;
        }

        float blue_intensity = max_possible_value > 0.0 ?
            std::clamp(static_cast<float>(normalized_value / max_possible_value), 0.0f, 1.0f) : 0.0f;
        return IM_COL32(
            static_cast<int>(50 * (1.0f - blue_intensity)), // Reduce red as blue increases
            static_cast<int>(50 * (1.0f - blue_intensity)), // Reduce green as blue increases
            static_cast<int>(100 + 155 * blue_intensity),   // Full blue range
            static_cast<int>(alpha * 255));
      }
      break;
  }
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  // Calculate values based on selected volume data type
  // Since the cell values were already updated in the main render loop based on the active VolumeAnalysisType,
  // we can now use them directly

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Trades:
      // For trade counts, we'll use the number formatting
      return formatNumber(static_cast<double>(cell.trade_count), number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::BuyTrades:
      // Use bid_volume which was set to buy trades count in the main render loop
      return formatNumber(cell.bid_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::SellTrades:
      // Use ask_volume which was set to sell trades count in the main render loop
      return formatNumber(cell.ask_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::Volume:
      // Use the sum of bid and ask volumes
      return formatNumber(cell.bid_volume + cell.ask_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::BuyVolume:
      // Use bid_volume which was already set in the main render loop
      return formatNumber(cell.bid_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::SellVolume:
      // Use ask_volume which was already set in the main render loop
      return formatNumber(cell.ask_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::BuySellVolume:
      // Use the delta which was already calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::Delta:
      // Use the delta which was already calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::DeltaPercent:
      // Use the delta which contains the percentage value calculated in the main render loop
      return std::format("{:.1f}%", cell.delta);

    case Data::VolumeAnalysisType::BuyVolumePercent:
      // Use the delta which contains the percentage value calculated in the main render loop
      return std::format("{:.1f}%", cell.delta);

    case Data::VolumeAnalysisType::SellVolumePercent:
      // Use the delta which contains the percentage value calculated in the main render loop
      return std::format("{:.1f}%", cell.delta);

    case Data::VolumeAnalysisType::CumulativeDelta:
      // Use the delta which was already calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::AverageSize:
      // Use the delta which contains the average size calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::AverageBuySize:
      // Use the delta which contains the average buy size calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::AverageSellSize:
      // Use the delta which contains the average sell size calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      // Use the delta which contains the max single trade volume calculated in the main render loop
      return formatNumber(cell.delta, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::FilteredVolume:
      // Use the sum of bid and ask volumes
      return formatNumber(cell.bid_volume + cell.ask_volume, number_format_, custom_decimal_places_);

    default:
      // For other types, default to showing the larger of bid/ask volume
      return formatNumber(std::max(cell.bid_volume, cell.ask_volume), number_format_, custom_decimal_places_);
  }
}

std::string FootprintPanel::getCellTooltip(const FootprintCell& cell) const {
  // Calculate delta percent based on the actual values
  double total_vol = cell.bid_volume + cell.ask_volume;
  double delta_percent = total_vol > 0.0 ? (cell.delta / total_vol) * 100.0 : 0.0;

  // Format the tooltip text with all required information
  // Include information about the active analysis type
  std::string analysis_type_label = "";
  double analysis_value = 0.0;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Trades:
      analysis_type_label = "Trades";
      analysis_value = static_cast<double>(cell.trade_count);
      break;
    case Data::VolumeAnalysisType::BuyTrades:
      analysis_type_label = "Buy Trades";
      analysis_value = cell.bid_volume; // Already set to buy trades count in main loop
      break;
    case Data::VolumeAnalysisType::SellTrades:
      analysis_type_label = "Sell Trades";
      analysis_value = cell.ask_volume; // Already set to sell trades count in main loop
      break;
    case Data::VolumeAnalysisType::Volume:
      analysis_type_label = "Volume";
      analysis_value = cell.bid_volume + cell.ask_volume;
      break;
    case Data::VolumeAnalysisType::BuyVolume:
      analysis_type_label = "Buy Volume";
      analysis_value = cell.bid_volume;
      break;
    case Data::VolumeAnalysisType::SellVolume:
      analysis_type_label = "Sell Volume";
      analysis_value = cell.ask_volume;
      break;
    case Data::VolumeAnalysisType::BuySellVolume:
      analysis_type_label = "Buy-Sell Volume";
      analysis_value = cell.delta; // Already set to buy-sell volume in main loop
      break;
    case Data::VolumeAnalysisType::Delta:
      analysis_type_label = "Delta";
      analysis_value = cell.delta;
      break;
    case Data::VolumeAnalysisType::DeltaPercent:
      analysis_type_label = "Delta %";
      analysis_value = cell.delta; // Already set to percentage in main loop
      break;
    case Data::VolumeAnalysisType::BuyVolumePercent:
      analysis_type_label = "Buy Vol %";
      analysis_value = cell.delta; // Already set to percentage in main loop
      break;
    case Data::VolumeAnalysisType::SellVolumePercent:
      analysis_type_label = "Sell Vol %";
      analysis_value = cell.delta; // Already set to percentage in main loop
      break;
    case Data::VolumeAnalysisType::CumulativeDelta:
      analysis_type_label = "Cumulative Delta";
      analysis_value = cell.delta;
      break;
    case Data::VolumeAnalysisType::AverageSize:
      analysis_type_label = "Avg Size";
      analysis_value = cell.delta; // Already set to average size in main loop
      break;
    case Data::VolumeAnalysisType::AverageBuySize:
      analysis_type_label = "Avg Buy Size";
      analysis_value = cell.delta; // Already set to average buy size in main loop
      break;
    case Data::VolumeAnalysisType::AverageSellSize:
      analysis_type_label = "Avg Sell Size";
      analysis_value = cell.delta; // Already set to average sell size in main loop
      break;
    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      analysis_type_label = "Max Trade Vol";
      analysis_value = cell.delta; // Already set to max trade volume in main loop
      break;
    case Data::VolumeAnalysisType::FilteredVolume:
      analysis_type_label = "Filtered Volume";
      analysis_value = cell.bid_volume + cell.ask_volume;
      break;
    default:
      analysis_type_label = "Volume";
      analysis_value = cell.bid_volume + cell.ask_volume;
      break;
  }

  // Convert nanosecond timestamps to readable format (seconds with decimals)
  double start_time_sec = static_cast<double>(cell.start_time_ns) / 1'000'000'000.0;
  double end_time_sec = static_cast<double>(cell.end_time_ns) / 1'000'000'000.0;

  // Use exact values from the extended FootprintCell structure
  std::string tooltip = std::format(
    "{}: {:.2f}\n"
    "Buy Volume: {:.2f}\n"
    "Sell Volume: {:.2f}\n"
    "Delta: {:.2f}\n"
    "Delta %: {:.2f}%\n"
    "Buy Trades: {}\n"
    "Sell Trades: {}\n"
    "Max Single Trade: {:.2f}\n"
    "Timestamp Range: {:.3f}-{:.3f}",
    analysis_type_label,
    analysis_value,
    cell.bid_volume,
    cell.ask_volume,
    cell.delta,
    delta_percent,
    cell.buy_trade_count,
    cell.sell_trade_count,
    cell.max_single_trade_volume,
    start_time_sec, // Start timestamp in seconds
    end_time_sec   // End timestamp in seconds
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
    float border_thickness = 3.0f; // Thicker border for imbalanced cells

    // If both diagonal and stacked imbalances exist, draw both effects with offset rectangles
    if (is_diagonal && is_stacked) {
      // Draw both borders with slight offset to distinguish them
      ImU32 diagonal_border_color = IM_COL32(255, 255, 0, 255); // Yellow
      ImU32 stacked_border_color = IM_COL32(0, 255, 255, 255); // Cyan

      // Draw stacked border first (outer) with slightly larger dimensions
      draw_list->AddRect(ImVec2(p1.x - 1, p1.y - 1), ImVec2(p2.x + 1, p2.y + 1),
                         stacked_border_color, 0.0f, 0, border_thickness);

      // Draw diagonal border second (inner) with normal dimensions
      draw_list->AddRect(p1, p2, diagonal_border_color, 0.0f, 0, border_thickness);

      // Add enhanced glow effects for both types
      // Glow for diagonal (inner)
      ImU32 diagonal_glow_color = IM_COL32(255, 255, 0, 80); // Semi-transparent yellow
      ImVec2 diagonal_glow_offset(3.0f, 3.0f);
      draw_list->AddRect(ImVec2(p1.x - diagonal_glow_offset.x, p1.y - diagonal_glow_offset.y),
                         ImVec2(p2.x + diagonal_glow_offset.x, p2.y + diagonal_glow_offset.y),
                         diagonal_glow_color, 0.0f, 0, 1.0f);

      // Glow for stacked (outer)
      ImU32 stacked_glow_color = IM_COL32(0, 255, 255, 80); // Semi-transparent cyan
      ImVec2 stacked_glow_offset(4.0f, 4.0f);
      draw_list->AddRect(ImVec2(p1.x - stacked_glow_offset.x, p1.y - stacked_glow_offset.y),
                         ImVec2(p2.x + stacked_glow_offset.x, p2.y + stacked_glow_offset.y),
                         stacked_glow_color, 0.0f, 0, 1.0f);
    }
    // Only diagonal imbalance
    else if (is_diagonal) {
      ImU32 diagonal_border_color = IM_COL32(255, 255, 0, 255); // Yellow
      draw_list->AddRect(p1, p2, diagonal_border_color, 0.0f, 0, border_thickness);

      // Enhanced glow effect for diagonal imbalance
      ImU32 glow_color = IM_COL32(255, 255, 0, 120); // More prominent yellow glow
      ImVec2 glow_offset(2.5f, 2.5f);
      draw_list->AddRect(ImVec2(p1.x - glow_offset.x, p1.y - glow_offset.y),
                         ImVec2(p2.x + glow_offset.x, p2.y + glow_offset.y),
                         glow_color, 0.0f, 0, 1.5f);
    }
    // Only stacked imbalance
    else if (is_stacked) {
      ImU32 stacked_border_color = IM_COL32(0, 255, 255, 255); // Cyan
      draw_list->AddRect(p1, p2, stacked_border_color, 0.0f, 0, border_thickness);

      // Enhanced glow effect for stacked imbalance
      ImU32 glow_color = IM_COL32(0, 255, 255, 120); // More prominent cyan glow
      ImVec2 glow_offset(2.5f, 2.5f);
      draw_list->AddRect(ImVec2(p1.x - glow_offset.x, p1.y - glow_offset.y),
                         ImVec2(p2.x + glow_offset.x, p2.y + glow_offset.y),
                         glow_color, 0.0f, 0, 1.5f);
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

  // Data type selector - 16 types (placed in header)
  const char* data_type_names[] = {
    "OHLC", "Order Book", "Trades", "Volume Profile", "Footprint", "TPO", "Metrics", "Alerts",
    "Heikin Ashi", "Renko", "Line Break", "Kagi", "Point & Figure", "Range Bars", "Volume Bars", "Tick Bars"
  };

  int current_data_type = static_cast<int>(data_type_);
  if (ImGui::BeginCombo("Data Type##DataTypeSelector", data_type_names[current_data_type])) {
    for (int i = 0; i < 16; i++) {
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
  if (ImGui::BeginCombo("Footprint Mode##VolumeDataTypeSelector", volume_data_type_names[current_vol_data_type])) {
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
        // Mark data as dirty to trigger immediate rendering update if needed
        data_dirty_.store(true, std::memory_order_release);
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
    if (ImGui::SliderInt("Decimals", &custom_decimal_places_, 0, 6)) {
      // Mark data as dirty to trigger immediate rendering update if decimal places change
      data_dirty_.store(true, std::memory_order_release);
    }
  }

  // Check if data needs to be refreshed due to selection changes
  if (data_dirty_.load(std::memory_order_acquire)) {
    // Mark data as dirty to force renderer to potentially refresh its data
    // The renderer itself handles the data pipeline integration
    data_dirty_.store(false, std::memory_order_release);
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

    // Collect all visible cells for imbalance detection
    std::vector<FootprintCell> all_cells;

    // ENHANCED MAIN RENDER LOOP: Iterate through visible time bars and price levels
    // Retrieve ClusterCell data and switch on active VolumeAnalysisType to determine displayed value

    // First, collect all visible clusters and organize them by time and price levels
    std::map<double, std::map<double, const RenderEngine::CandleCluster*>> visible_clusters;

    // Calculate max volume across all visible clusters for adaptive alpha calculation
    // This is done in the same loop to avoid a second iteration
    for (const auto &cluster : clusters) {
        // Check if cluster is within visible bounds
        if (cluster.centerX >= x_min && cluster.centerX <= x_max &&
            cluster.centerY >= y_min && cluster.centerY <= y_max) {

            // Organize clusters by time (x-axis) and price (y-axis) for efficient iteration
            visible_clusters[cluster.centerX][cluster.centerY] = &cluster;

            // Also calculate max volume for adaptive alpha calculation
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

    // ENHANCED: Iterate through visible time bars and price levels with improved efficiency
    // Process each visible cluster according to the active VolumeAnalysisType
    for (const auto &[time_level, price_clusters] : visible_clusters) {
        for (const auto &[price_level, cluster_ptr] : price_clusters) {
            const auto &cluster = *cluster_ptr;

            // Convert cluster to footprint cell with all relevant data
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

            // Populate the extended fields from the cluster
            cell.buy_trade_count = cluster.buyTradeCount;
            cell.sell_trade_count = cluster.sellTradeCount;
            cell.max_single_trade_volume = static_cast<double>(cluster.maxSingleTradeVolume);
            cell.start_time_ns = cluster.startTimeNs;
            cell.end_time_ns = cluster.endTimeNs;

            // ENHANCED: Calculate value based on active VolumeAnalysisType for this specific cell
            // This is the core logic that determines what value is displayed in each cell
            // The switch statement now handles all VolumeAnalysisType values comprehensively
            switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
                case BTQuant::Data::VolumeAnalysisType::Trades:
                    // Display total number of trades in the cell
                    cell.bid_volume = static_cast<double>(cluster.tradeCount);
                    cell.ask_volume = 0.0; // Not applicable for trades count
                    cell.delta = static_cast<double>(cluster.tradeCount);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyTrades:
                    // Estimate buy trades based on volume ratio
                    {
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        if (total_vol > 0) {
                            double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
                            cell.bid_volume = static_cast<double>(cluster.tradeCount) * buy_ratio;
                            cell.ask_volume = 0.0; // Not applicable for buy trades count
                            cell.delta = cell.bid_volume;
                        } else {
                            cell.bid_volume = static_cast<double>(cluster.tradeCount) * 0.5; // Equal split if no volume
                            cell.ask_volume = 0.0;
                            cell.delta = cell.bid_volume;
                        }
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellTrades:
                    // Estimate sell trades based on volume ratio
                    {
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        if (total_vol > 0) {
                            double sell_ratio = static_cast<double>(cluster.askVolume) / total_vol;
                            cell.ask_volume = static_cast<double>(cluster.tradeCount) * sell_ratio;
                            cell.bid_volume = 0.0; // Not applicable for sell trades count
                            cell.delta = -cell.ask_volume; // Negative for sell trades
                        } else {
                            cell.ask_volume = static_cast<double>(cluster.tradeCount) * 0.5; // Equal split if no volume
                            cell.bid_volume = 0.0;
                            cell.delta = -cell.ask_volume;
                        }
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::Volume:
                    // Display total volume (bid + ask)
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolume:
                    // Display buy volume only
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = 0.0; // Not applicable for buy volume
                    cell.delta = static_cast<double>(cluster.bidVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolume:
                    // Display sell volume only
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.bid_volume = 0.0; // Not applicable for sell volume
                    cell.delta = -static_cast<double>(cluster.askVolume); // Negative for sell volume
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
                    // Display difference between buy and sell volume (BuyVolume - SellVolume)
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume - cluster.askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::Delta:
                    // Display net difference between buy and sell volume (BuyVolume - SellVolume)
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume - cluster.askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
                    // Display delta as percentage of total volume
                    {
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = total_vol > 0.0 ?
                            (static_cast<double>(cluster.bidVolume - cluster.askVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
                    // Display running sum of delta values
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume - cluster.askVolume);
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSize:
                    // Display average trade size
                    {
                        int total_count = cluster.tradeCount;
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = total_count > 0 ? total_vol / static_cast<double>(total_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
                    // Display average size of buy trades
                    {
                        // Estimate buy count based on volume ratio
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        int buy_count = cluster.tradeCount; // Start with total count
                        if (total_vol > 0) {
                            double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
                            buy_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * buy_ratio);
                        }
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = buy_count > 0 ?
                            static_cast<double>(cluster.bidVolume) / static_cast<double>(buy_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
                    // Display average size of sell trades
                    {
                        // Estimate sell count based on volume ratio
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        int sell_count = cluster.tradeCount; // Start with total count
                        if (total_vol > 0) {
                            double sell_ratio = static_cast<double>(cluster.askVolume) / total_vol;
                            sell_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * sell_ratio);
                        }
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = sell_count > 0 ?
                            static_cast<double>(cluster.askVolume) / static_cast<double>(sell_count) : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
                    // Display maximum volume of a single trade
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = cluster.tradeCount > 0 ?
                        (static_cast<double>(cluster.bidVolume + cluster.askVolume) / static_cast<double>(cluster.tradeCount)) : 0.0;
                    break;

                case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
                    // Display percentage of buy volume
                    {
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = total_vol > 0.0 ?
                            (static_cast<double>(cluster.bidVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
                    // Display percentage of sell volume
                    {
                        double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                        cell.bid_volume = static_cast<double>(cluster.bidVolume);
                        cell.ask_volume = static_cast<double>(cluster.askVolume);
                        cell.delta = total_vol > 0.0 ?
                            (static_cast<double>(cluster.askVolume) / total_vol) * 100.0 : 0.0;
                    }
                    break;

                case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
                    // Display volume filtered by specific criteria
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                    break;

                default:
                    // Default to total volume if unknown type
                    cell.bid_volume = static_cast<double>(cluster.bidVolume);
                    cell.ask_volume = static_cast<double>(cluster.askVolume);
                    cell.delta = static_cast<double>(cluster.bidVolume + cluster.askVolume);
                    break;
            }

            // Update the cell's trade count regardless of analysis type
            cell.trade_count = cluster.tradeCount;

            // Add to all cells for imbalance detection
            all_cells.push_back(cell);
        }
    }

    // ENHANCED: Perform additional optimizations and calculations based on the active VolumeAnalysisType
    // This allows for more sophisticated analysis depending on the selected view
    switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
        case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
            // For cumulative delta, calculate running totals across time and price dimensions
            {
                // Sort cells by time to ensure proper cumulative calculation
                std::sort(all_cells.begin(), all_cells.end(),
                         [](const FootprintCell& a, const FootprintCell& b) {
                             return a.x < b.x || (a.x == b.x && a.y < b.y);
                         });

                double cumulative_sum = 0.0;
                for (auto& cell : all_cells) {
                    cumulative_sum += cell.delta; // Delta already contains the difference
                    cell.delta = cumulative_sum; // Update with cumulative value
                }
            }
            break;

        case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
            // For delta percent, ensure values are properly normalized
            // This is already handled in the individual cell processing above
            break;

        case BTQuant::Data::VolumeAnalysisType::AverageSize:
        case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
        case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
            // For average sizes, ensure calculations are consistent
            // This is already handled in the individual cell processing above
            break;

        default:
            // For other types, no additional processing needed
            break;
    }

    // ENHANCED: Post-process all visible cells to apply any additional transformations
    // based on the active VolumeAnalysisType
    for (auto& cell : all_cells) {
        // All cells have already been processed according to the active VolumeAnalysisType
        // Additional post-processing can be applied here if needed
    }

    // Detect imbalances
    std::vector<FootprintCell> diagonal_imbalances;
    std::vector<FootprintCell> stacked_imbalances;
    detectImbalances(all_cells, diagonal_imbalances, stacked_imbalances);

    // ENHANCED: Group only visible clusters by time (x-coordinate) to calculate time-bar summaries
    // This optimization reduces redundant processing by only considering visible clusters
    // The grouping now takes into account the active VolumeAnalysisType for more accurate aggregations
    std::map<double, std::vector<const RenderEngine::CandleCluster*>> clusters_by_time;

    // Iterate through visible clusters to group by time
    for (const auto &cluster : clusters) {
        // Check if cluster is within visible bounds before grouping
        if (cluster.centerX >= x_min && cluster.centerX <= x_max &&
            cluster.centerY >= y_min && cluster.centerY <= y_max) {

            // Round x to nearest time unit to group clusters by time bar
            // Using a more precise rounding method for better time alignment
            double time_key = std::round(cluster.centerX * 10.0) / 10.0; // Adjust precision as needed
            clusters_by_time[time_key].push_back(&cluster);
        }
    }

    // ENHANCED: Calculate cumulative and other summaries for each time bar based on active VolumeAnalysisType
    // Optimized to reduce redundant calculations and improve performance
    std::map<double, double> cumulative_values;  // Cumulative value by time based on active analysis type
    std::map<double, std::pair<double, double>> poc_info;  // POC price and volume by time
    std::map<double, double> time_bar_net_values;  // Net value by time based on active analysis type
    std::map<double, double> time_bar_total_values;  // Total value by time based on active analysis type

    double running_cumulative_value = 0.0;

    // Process each time bar to calculate aggregated values based on the active VolumeAnalysisType
    for (const auto& [time_key, time_clusters] : clusters_by_time) {
        double time_net_value = 0.0;
        double time_total_value = 0.0;
        double max_volume_in_time_bar = 0.0;
        double poc_price = 0.0;
        double max_analysis_value_in_time_bar = 0.0; // Track max value for POC based on analysis type

        // Process each cluster within the time bar
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

            // Update time bar summary values based on the active VolumeAnalysisType
            time_net_value += cluster_value;
            time_total_value += std::abs(cluster_value); // Use absolute value for total

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
        poc_info[time_key] = std::make_pair(poc_price, max_analysis_value_in_time_bar); // Use analysis value for POC
    }

    // ENHANCED: Render all cells with imbalance highlighting
    // Apply the calculated max volume for adaptive alpha and highlight imbalances
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
            double total_volume = time_bar_total_values[time_key];
            double net_delta = time_bar_net_values[time_key];
            double cumulative_delta = cumulative_values[time_key];
            double poc_price = poc_info[time_key].first;

            // Format the header text to show total volume, net delta, cumulative delta, and POC price
            char header_text[256];
            snprintf(header_text, sizeof(header_text),
                     "TV:%.0f ND:%+.0f CD:%+.0f POC:%.2f",
                     total_volume, net_delta, cumulative_delta, poc_price);

            // Convert time to pixel coordinates for header positioning
            ImVec2 header_pos = ImPlot::PlotToPixels(time_key, y_max + 5.0); // Position header slightly above the highest price

            // Use monospace font for alignment
            ImFont* mono_font = nullptr;
            // First, try to find a monospace font by name
            for (int i = 0; i < ImGui::GetIO().Fonts->Fonts.Size; i++) {
                const char* font_name = ImGui::GetIO().Fonts->Fonts[i]->GetDebugName();
                if (font_name && (strstr(font_name, "Mono") != nullptr ||
                                 strstr(font_name, "Consolas") != nullptr ||
                                 strstr(font_name, "Courier") != nullptr ||
                                 strstr(font_name, "Fixed") != nullptr)) {
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

    // Use smaller font by pushing text wrap position to make text appear smaller
    // Alternative approach: use ImGui::PushFont if a smaller font is available
    ImGui::Text("Trades: %d | Avg Size: %.2f | Max Trade: %.2f",
                total_trades, avg_trade_size, max_single_trade_volume);

    // Restore the original scale
    ImGui::PopStyleVar(2);
  }

  // Enhanced Debug Overlay
  if (!clusters.empty()) {
    // Data type names for display
    const char* data_type_names[] = {
      "OHLC", "Order Book", "Trades", "Volume Profile", "Footprint", "TPO", "Metrics", "Alerts",
      "Heikin Ashi", "Renko", "Line Break", "Kagi", "Point & Figure", "Range Bars", "Volume Bars", "Tick Bars"
    };

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
                       "Data: %s | Mode: %s | Time Agg: %s | Price Agg: %s | Clusters: %zu | Grid: %dx%d | Thresh: %.2f",
                       data_type_names[static_cast<int>(data_type_)],
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
            double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
            total_value += total_vol > 0.0 ? ((c.bidVolume - c.askVolume) / total_vol) * 100.0 : 0.0;
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
          total_value += c.tradeCount > 0 ?
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
