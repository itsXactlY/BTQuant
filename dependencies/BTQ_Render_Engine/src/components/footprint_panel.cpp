#include "components/footprint_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <format>
#include <iomanip>
#include <map>
#include <sstream>

#include "../../include/analytics/cluster_engine.hpp"
#include "analytics/cluster_engine.hpp"
#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

// Helper function to format numbers according to the selected format
std::string FootprintPanel::formatNumber(double value, NumberFormat format, int decimal_places) {
  std::ostringstream oss;

  // Limit decimal places to reasonable range to prevent overflow issues
  decimal_places = std::max(0, std::min(10, decimal_places));

  // Handle special cases (NaN, infinity)
  if (std::isnan(value) || std::isinf(value)) {
    oss << (std::isnan(value) ? "NaN" : (value > 0 ? "∞" : "-∞"));
    return oss.str();
  }

  switch (format) {
    case NumberFormat::Raw:
      // Raw number formatting with special handling for edge cases
      if (value == 0.0) {
        oss << "0.0";
      } else {
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::ThousandsK:
      if (std::abs(value) >= 1e12) {
        // Trillions - for very large numbers
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e12) << "T";
      } else if (std::abs(value) >= 1e9) {
        // Billions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e9) << "B";
      } else if (std::abs(value) >= 1e6) {
        // Millions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e6) << "M";
      } else if (std::abs(value) >= 1e3) {
        // Thousands - this is the primary unit for this format
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e3) << "K";
      } else {
        // Raw value for anything below 1000
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::MillionsM:

      // Format primarily in millions, with fallback to thousands for smaller values and billions
      // for larger values
      if (value == 0.0) {
        oss << "0.0";
      } else if (std::abs(value) >= 1e12) {
        // Trillions - show as trillions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e12) << "T";
      } else if (std::abs(value) >= 1e9) {
        // Billions - show as billions
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e9) << "B";
      } else if (std::abs(value) >= 1e6) {
        // Millions - show as millions (this is the primary unit for this format)
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e6) << "M";
      } else if (std::abs(value) >= 1e3) {
        // Thousands - show as thousands
        oss << std::fixed << std::setprecision(decimal_places) << (value / 1e3) << "K";
      } else {
        // Values below 1000 - show as raw value
        oss << std::fixed << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::Scientific:
      // Scientific notation with special handling for edge cases
      if (value == 0.0) {
        oss << "0.0";
      } else {
        oss << std::scientific << std::setprecision(decimal_places) << value;
      }
      break;

    case NumberFormat::CustomDecimal:
      // For custom decimal format, we can apply the same logic as Raw but with custom precision
      oss << std::fixed << std::setprecision(decimal_places) << value;
      break;
  }

  return oss.str();
}

FootprintPanel::FootprintPanel(const PanelConfig& config)
    : PanelBase(config),
      volume_data_type_(Data::VolumeDataType::Delta),
      time_aggregation_type_(Data::TimeAggregationType::T_1MIN),
      volume_based_n_contracts_(1000),
      tick_based_n_ticks_(100),
      price_aggregation_type_(Data::PriceAggregationType::P_1TICK),
      custom_price_aggregation_value_(0.1),
      zoom_sensitivity_(1.0f) {

  // Configure the LOD system with appropriate thresholds for footprint visualization
  lod_system_.setMinDetailZoom(0.1f);
  lod_system_.setMediumDetailZoom(1.0f);
  lod_system_.setMaxDetailZoom(3.0f);

  // Set cell size thresholds in pixels for different LOD levels
  lod_system_.setMinCellSizePx(4.0f);
  lod_system_.setMediumCellSizePx(12.0f);
  lod_system_.setMaxCellSizePx(24.0f);

  // Set thresholds for specific rendering elements
  lod_system_.setTextRenderThreshold(12.0f);
  lod_system_.setLabelRenderThreshold(20.0f);
  lod_system_.setDetailRenderThreshold(8.0f);
}

void FootprintPanel::update(float /*dt*/) {
  // Update logic if needed
}

ImU32 FootprintPanel::getCellColor(const FootprintCell& cell, double max_volume) const {
  // For SplitVolume mode, we don't use this method as the cell is drawn with split colors
  if (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_) ==
      Data::VolumeAnalysisType::SplitVolume) {
    // Return a default color that won't be used since split volume draws its own colors
    return IM_COL32(128, 128, 128, 255);  // Gray as default
  }

  // Calculate values based on selected volume data type
  double value_to_display = 0.0;
  double total_vol = cell.bid_volume + cell.ask_volume;

  // Calculate adaptive alpha based on cell_volume / max_bar_volume
  // Use the appropriate volume value based on the selected analysis type
  double cell_volume = 0.0;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Delta:
    case Data::VolumeAnalysisType::DeltaPercent:
    case Data::VolumeAnalysisType::CumulativeDelta:
      // For delta types, use absolute delta value for alpha calculation
      cell_volume = std::abs(cell.delta);
      break;

    case Data::VolumeAnalysisType::BuyVolume:
    case Data::VolumeAnalysisType::SellVolume:
    case Data::VolumeAnalysisType::BuySellVolume:
      // For buy/sell types, use the respective volumes
      if (volume_data_type_ == Data::VolumeAnalysisType::BuyVolume) {
        cell_volume = cell.bid_volume;
      } else if (volume_data_type_ == Data::VolumeAnalysisType::SellVolume) {
        cell_volume = cell.ask_volume;
      } else {  // BuySellVolume
        cell_volume = std::abs(cell.delta);
      }
      break;

    case Data::VolumeAnalysisType::Volume:
    case Data::VolumeAnalysisType::BuyVolumePercent:
    case Data::VolumeAnalysisType::SellVolumePercent:
    case Data::VolumeAnalysisType::Trades:
    case Data::VolumeAnalysisType::BuyTrades:
    case Data::VolumeAnalysisType::SellTrades:
    case Data::VolumeAnalysisType::FilteredVolume:
      // For volume intensity types, use total volume or trade count
      if (volume_data_type_ == Data::VolumeAnalysisType::Trades ||
          volume_data_type_ == Data::VolumeAnalysisType::BuyTrades ||
          volume_data_type_ == Data::VolumeAnalysisType::SellTrades) {
        cell_volume = static_cast<double>(cell.trade_count);
      } else {
        cell_volume = total_vol;
      }
      break;

    case Data::VolumeAnalysisType::AverageSize:
    case Data::VolumeAnalysisType::AverageBuySize:
    case Data::VolumeAnalysisType::AverageSellSize:
    case Data::VolumeAnalysisType::MaxOneTradeVolume:
    default:
      // For other metrics, use the delta field which contains the calculated value
      cell_volume = std::abs(cell.delta);
      break;
  }

  // Calculate adaptive alpha based on cell_volume / max_bar_volume with enhanced precision
  float alpha = max_volume > 0.0
                    ? std::clamp(static_cast<float>(cell_volume / max_volume), 0.05f, 1.0f)
                    : 0.05f;

  // Determine the appropriate color scheme based on the volume analysis type
  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Delta:
    case Data::VolumeAnalysisType::DeltaPercent:
    case Data::VolumeAnalysisType::CumulativeDelta:
      // Green-red gradient for delta (buy/sell imbalance)
      return getDeltaColor(cell, alpha);

    case Data::VolumeAnalysisType::BuyVolume:
    case Data::VolumeAnalysisType::SellVolume:
    case Data::VolumeAnalysisType::BuySellVolume:
      // Blue-red gradient for buy/sell volume comparison
      return getBuySellColor(cell, max_volume, alpha);

    case Data::VolumeAnalysisType::Volume:
    case Data::VolumeAnalysisType::BuyVolumePercent:
    case Data::VolumeAnalysisType::SellVolumePercent:
    case Data::VolumeAnalysisType::Trades:
    case Data::VolumeAnalysisType::BuyTrades:
    case Data::VolumeAnalysisType::SellTrades:
    case Data::VolumeAnalysisType::FilteredVolume:
    case Data::VolumeAnalysisType::AverageSize:
    case Data::VolumeAnalysisType::AverageBuySize:
    case Data::VolumeAnalysisType::AverageSellSize:
    case Data::VolumeAnalysisType::MaxOneTradeVolume:
    default:
      // Yellow-orange gradient for volume intensity
      return getVolumeIntensityColor(cell, max_volume, alpha);
  }
}

ImU32 FootprintPanel::getDeltaColor(const FootprintCell& cell, float alpha) const {
  // Use the delta value that was already calculated in the main render loop
  double normalized_delta = cell.delta;

  // For DeltaPercent, normalize the value to [-1, 1] range
  if (volume_data_type_ ==
      static_cast<Data::VolumeDataType>(Data::VolumeAnalysisType::DeltaPercent)) {
    normalized_delta = std::clamp(normalized_delta, -100.0, 100.0) / 100.0;
  } else {
    // For other delta types, normalize based on the sum of volumes
    double total_vol = cell.bid_volume + cell.ask_volume;
    normalized_delta = total_vol > 0.0 ? cell.delta / total_vol : 0.0;
    // Clamp to [-1, 1] range to ensure proper color mapping
    normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);
  }

  // Clamp to [-1, 1] range to ensure proper color mapping
  normalized_delta = std::clamp(normalized_delta, -1.0, 1.0);

  // Use smooth green-red gradient based on normalized delta value
  float red_intensity = 0.0f;
  float green_intensity = 0.0f;

  if (normalized_delta >= 0) {
    // Positive delta (buy pressure) - Green gradient
    green_intensity = std::clamp(static_cast<float>(normalized_delta), 0.0f, 1.0f);
    red_intensity = 1.0f - green_intensity;  // Reduce red as green increases
  } else {
    // Negative delta (sell pressure) - Red gradient
    red_intensity = std::clamp(static_cast<float>(-normalized_delta), 0.0f, 1.0f);
    green_intensity = 1.0f - red_intensity;  // Reduce green as red increases
  }

  // Create smooth transition from red (negative) to green (positive) through neutral
  return IM_COL32(
      static_cast<int>(128 + 127 * red_intensity - 127 * green_intensity),  // Red channel
      static_cast<int>(128 + 127 * green_intensity - 127 * red_intensity),  // Green channel
      64,  // Blue channel kept low for better contrast
      static_cast<int>(alpha * 255));
}

ImU32 FootprintPanel::getBuySellColor(const FootprintCell& cell, double max_volume,
                                      float alpha) const {
  double normalized_value = 0.0;
  double max_possible_value = 0.0;

  if (volume_data_type_ == static_cast<Data::VolumeDataType>(Data::VolumeAnalysisType::BuyVolume)) {
    // Use bid_volume which was already set in the main render loop
    normalized_value = cell.bid_volume;
    max_possible_value = max_volume;  // Use the max volume passed to the function
  } else if (volume_data_type_ ==
             static_cast<Data::VolumeDataType>(Data::VolumeAnalysisType::SellVolume)) {
    // Use ask_volume which was already set in the main render loop
    normalized_value = cell.ask_volume;
    max_possible_value = max_volume;  // Use the max volume passed to the function
  } else {                            // BuySellVolume
    // Use the delta which was already calculated in the main render loop
    normalized_value = cell.delta;
    max_possible_value = max_volume;  // Use the max volume passed to the function
  }

  // Normalize based on max possible value for this analysis type
  double normalized_ratio =
      max_possible_value > 0.0 ? std::clamp(normalized_value / max_possible_value, -1.0, 1.0) : 0.0;

  // Use blue-red gradient based on normalized ratio value
  float red_intensity = 0.0f;
  float blue_intensity = 0.0f;

  if (normalized_ratio >= 0) {
    // Positive - Blue gradient for buy volume dominance
    blue_intensity = std::clamp(static_cast<float>(normalized_ratio), 0.0f, 1.0f);
    red_intensity = 1.0f - blue_intensity;  // Reduce red as blue increases
  } else {
    // Negative - Red gradient for sell volume dominance
    red_intensity = std::clamp(static_cast<float>(-normalized_ratio), 0.0f, 1.0f);
    blue_intensity = 1.0f - red_intensity;  // Reduce blue as red increases
  }

  // Create smooth transition from blue (buy) to red (sell) through neutral
  return IM_COL32(
      static_cast<int>(128 + 127 * red_intensity - 127 * blue_intensity),  // Red channel
      64,  // Green channel kept low for better contrast
      static_cast<int>(128 + 127 * blue_intensity - 127 * red_intensity),  // Blue channel
      static_cast<int>(alpha * 255));
}

ImU32 FootprintPanel::getVolumeIntensityColor(const FootprintCell& cell, double max_volume,
                                              float alpha) const {
  double normalized_value = 0.0;
  double max_possible_value = 0.0;
  double total_vol = cell.bid_volume + cell.ask_volume;

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::Volume:
      normalized_value = total_vol;
      max_possible_value = max_volume;
      break;
    case Data::VolumeAnalysisType::BuyVolumePercent:
      // Use the delta which contains the percentage value
      normalized_value = std::abs(cell.delta);  // Use absolute value for color intensity
      max_possible_value = 100.0;
      break;
    case Data::VolumeAnalysisType::SellVolumePercent:
      // Use the delta which contains the percentage value
      normalized_value = std::abs(cell.delta);  // Use absolute value for color intensity
      max_possible_value = 100.0;
      break;
    case Data::VolumeAnalysisType::Trades:
      // Use trade count
      normalized_value = static_cast<double>(cell.trade_count);
      max_possible_value = max_volume;  // Use max volume as reference for scaling
      break;
    case Data::VolumeAnalysisType::BuyTrades:
      // Use bid_volume which was set to buy trades count in the main render loop
      normalized_value = cell.bid_volume;
      max_possible_value = max_volume;  // Use max volume as reference for scaling
      break;
    case Data::VolumeAnalysisType::SellTrades:
      // Use ask_volume which was set to sell trades count in the main render loop
      normalized_value = cell.ask_volume;
      max_possible_value = max_volume;  // Use max volume as reference for scaling
      break;
    case Data::VolumeAnalysisType::FilteredVolume:
      normalized_value = total_vol;
      max_possible_value = max_volume;
      break;
    case Data::VolumeAnalysisType::AverageSize:
    case Data::VolumeAnalysisType::AverageBuySize:
    case Data::VolumeAnalysisType::AverageSellSize:
    case Data::VolumeAnalysisType::MaxOneTradeVolume:
      // Use the delta value which was already calculated in the main render loop
      normalized_value = std::abs(cell.delta);  // Use absolute value for color intensity
      max_possible_value = max_volume;          // Use max volume as reference
      break;
    default:
      normalized_value = std::abs(total_vol);
      max_possible_value = max_volume;  // Use max volume as reference
      break;
  }

  float intensity =
      max_possible_value > 0.0
          ? std::clamp(static_cast<float>(normalized_value / max_possible_value), 0.0f, 1.0f)
          : 0.0f;

  // Yellow-orange gradient for volume intensity - transitioning from yellow (low intensity) to
  // orange (high intensity) Yellow: high R&G, low B; Orange: high R, medium G, low B
  float red_val = 200.0f + 55.0f * intensity;     // Range: 200-255 (higher for more intensity)
  float green_val = 150.0f + 105.0f * intensity;  // Range: 150-255 (increasing for more intensity)
  float blue_val =
      50.0f * (1.0f - intensity);  // Low blue that decreases with intensity for better contrast

  return IM_COL32(static_cast<int>(std::min(255.0f, red_val)),
                  static_cast<int>(std::min(255.0f, green_val)),
                  static_cast<int>(std::min(255.0f, blue_val)), static_cast<int>(alpha * 255));
}

std::string FootprintPanel::getCellLabel(const FootprintCell& cell) const {
  // Calculate values based on selected volume data type
  // Since the cell values were already updated in the main render loop based on the active
  // VolumeAnalysisType, we can now use them directly

  switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
    case Data::VolumeAnalysisType::SplitVolume:
      // For split volume mode, show both buy and sell volumes
      return formatNumber(cell.bid_volume, number_format_, custom_decimal_places_) + "/" +
             formatNumber(cell.ask_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::Trades:
      // For trade counts, we'll use the number formatting
      return formatNumber(static_cast<double>(cell.trade_count), number_format_,
                          custom_decimal_places_);

    case Data::VolumeAnalysisType::BuyTrades:
      // Use bid_volume which was set to buy trades count in the main render loop
      return formatNumber(cell.bid_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::SellTrades:
      // Use ask_volume which was set to sell trades count in the main render loop
      return formatNumber(cell.ask_volume, number_format_, custom_decimal_places_);

    case Data::VolumeAnalysisType::Volume:
      // Use the sum of bid and ask volumes
      return formatNumber(cell.bid_volume + cell.ask_volume, number_format_,
                          custom_decimal_places_);

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
      // Format percentage using the selected number format
      return formatNumber(cell.delta, number_format_, custom_decimal_places_) + "%";

    case Data::VolumeAnalysisType::BuyVolumePercent:
      // Use the delta which contains the percentage value calculated in the main render loop
      // Format percentage using the selected number format
      return formatNumber(cell.delta, number_format_, custom_decimal_places_) + "%";

    case Data::VolumeAnalysisType::SellVolumePercent:
      // Use the delta which contains the percentage value calculated in the main render loop
      // Format percentage using the selected number format
      return formatNumber(cell.delta, number_format_, custom_decimal_places_) + "%";

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
      return formatNumber(cell.bid_volume + cell.ask_volume, number_format_,
                          custom_decimal_places_);

    default:
      // For other types, default to showing the larger of bid/ask volume
      return formatNumber(std::max(cell.bid_volume, cell.ask_volume), number_format_,
                          custom_decimal_places_);
  }
}

std::string FootprintPanel::getCellTooltip(const FootprintCell& cell) const {
  // Calculate delta percent based on the actual buy and sell volumes
  double total_vol = cell.bid_volume + cell.ask_volume;
  double delta_percent =
      total_vol > 0.0 ? ((cell.bid_volume - cell.ask_volume) / total_vol) * 100.0 : 0.0;

  // Convert nanosecond timestamps to readable format (seconds with decimals)
  double start_time_sec = static_cast<double>(cell.start_time_ns) / 1'000'000'000.0;
  double end_time_sec = static_cast<double>(cell.end_time_ns) / 1'000'000'000.0;

  // Format the tooltip text with all required information in a clean, organized way
  std::string tooltip = std::format(
      "📊 FOOTPRINT CELL DATA\n"
      "═════════════════════\n"
      "• Exact Buy Volume: {:.2f}\n"
      "• Exact Sell Volume: {:.2f}\n"
      "• Delta (B-S): {:.2f}\n"
      "• Delta %: {:.2f}%\n"
      "═════════════════════\n"
      "• # Buy Trades: {}\n"
      "• # Sell Trades: {}\n"
      "• Max Single Trade: {:.2f}\n"
      "• Time Range: {:.3f}s - {:.3f}s",
      cell.bid_volume,                    // exact buy volume
      cell.ask_volume,                    // exact sell volume
      cell.bid_volume - cell.ask_volume,  // delta (actual buy - sell)
      delta_percent,                      // delta percent
      cell.buy_trade_count,               // number of buy trades
      cell.sell_trade_count,              // number of sell trades
      cell.max_single_trade_volume,       // max single trade
      start_time_sec,                     // start timestamp
      end_time_sec                        // end timestamp
  );

  return tooltip;
}

void FootprintPanel::renderCell(const FootprintCell& cell, ImDrawList* draw_list,
                                double max_volume) {
  // Call the overloaded version with empty imbalance vectors
  std::vector<FootprintCell> empty_diagonal;
  std::vector<FootprintCell> empty_stacked;
  renderCell(cell, draw_list, max_volume, empty_diagonal, empty_stacked,
             1.0);  // Default zoom factor of 1.0
}

void FootprintPanel::renderCell(const FootprintCell& cell, ImDrawList* draw_list, double max_volume,
                                const std::vector<FootprintCell>& diagonal_imbalances,
                                const std::vector<FootprintCell>& stacked_imbalances,
                                double zoom_factor) {
  // Use the LOD system to render the cell with appropriate level of detail
  lod_system_.applyLODToCell(cell, draw_list, static_cast<float>(zoom_factor), max_volume,
                             diagonal_imbalances, stacked_imbalances, this);
}

void FootprintPanel::renderFilteredCell(const FootprintCell& cell, ImDrawList* draw_list,
                                        double /*max_volume*/, double zoom_factor) {
  // For filtered cells, we'll create a temporary version with reduced visibility
  // Calculate cell corners in plot coordinates with zoom-based adjustment
  double base_padding = 0.48;

  // Adjust cell padding based on zoom level with more responsive transitions
  // Use the same sophisticated algorithm as the regular cell rendering for consistency
  double adjusted_padding;
  if (zoom_factor >= 1.0) {
    // When zoomed in: expand cells to show more detail
    // Use a logarithmic approach for smoother transitions at high zoom levels
    // The higher the zoom, the less padding (larger cells)
    double zoom_effect = std::log10(zoom_factor * zoom_sensitivity_ + 1.0) * 0.3;
    adjusted_padding = std::max(0.05, base_padding - zoom_effect);
  } else {
    // When zoomed out: shrink cells to show more of them, approaching squares
    // Use an inverse approach to make cells smaller when zoomed out
    double zoom_effect = std::pow(1.0 / (zoom_factor * zoom_sensitivity_), 0.8) - 1.0;
    // Increase padding to make cells appear smaller when zoomed out
    adjusted_padding = std::min(0.48, base_padding + zoom_effect * 0.15);
  }

  // Ensure padding stays within reasonable bounds to maintain visibility
  adjusted_padding = std::max(0.01, std::min(0.48, adjusted_padding));

  double x1 = cell.x - cell.width * adjusted_padding;
  double x2 = cell.x + cell.width * adjusted_padding;
  double y1 = cell.y - cell.height * adjusted_padding;
  double y2 = cell.y + cell.height * adjusted_padding;

  // Convert to pixel coordinates
  ImVec2 p1 = ImPlot::PlotToPixels(x1, y1);
  ImVec2 p2 = ImPlot::PlotToPixels(x2, y2);

  // Calculate cell dimensions in pixels for level-of-detail decisions
  float cell_width_px = std::abs(p2.x - p1.x);
  float cell_height_px = std::abs(p2.y - p1.y);
  float min_dimension_px = std::min(cell_width_px, cell_height_px);

  // Draw greyed-out cell for values below threshold
  // Use a light grey color with low alpha to indicate filtered cells
  ImU32 greyed_out_color = IM_COL32(128, 128, 128, 64);  // Grey with transparency

  // Draw filled cell with greyed-out appearance
  draw_list->AddRectFilled(p1, p2, greyed_out_color);

  // Draw subtle border for cell separation (normal case)
  ImU32 border_color = IM_COL32(128, 128, 128, 40);  // Grey, low alpha
  draw_list->AddRect(p1, p2, border_color, 0.0f, 0, 1.0f);

  // Draw volume label if enabled and cell is large enough
  // Implement LOD: skip text rendering when cell height < 12px
  if (show_volume_labels_ && cell_height_px >= 12.0f) {
    std::string label = getCellLabel(cell);
    ImVec2 text_size = ImGui::CalcTextSize(label.c_str());

    // Center text in cell
    ImVec2 text_pos((p1.x + p2.x - text_size.x) * 0.5f, (p1.y + p2.y - text_size.y) * 0.5f);

    // Use a lighter grey text for filtered cells
    draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 128),
                       label.c_str());  // Light grey text with transparency
  }
}

bool FootprintPanel::isDiagonalImbalance(const FootprintCell& cell,
                                         const std::vector<FootprintCell>& all_cells) const {
  // Define threshold for imbalance detection
  const double threshold = 3.0;  // Standard threshold from cluster engine

  // Look for diagonal patterns: buy volume at price P compared to sell volume at price P-1
  for (const auto& other_cell : all_cells) {
    // Check if this is a diagonal neighbor (adjacent price level, same or nearby time)
    // Diagonal imbalance: comparing buy volume at one price level with sell volume at adjacent
    // price level
    if (std::abs(std::abs(cell.y - other_cell.y) - 1.0) < 0.2 &&
        std::abs(cell.x - other_cell.x) < 2.0) {  // Adjacent price level, similar time
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

bool FootprintPanel::isStackedImbalance(const FootprintCell& cell,
                                        const std::vector<FootprintCell>& all_cells) const {
  // Define threshold for imbalance detection
  const double threshold = 3.0;  // Standard threshold from cluster engine

  // Look for stacked patterns: comparing volumes at same price level across consecutive time
  // buckets
  for (const auto& other_cell : all_cells) {
    // Check if this is at the same price level but different time (stacked in time dimension)
    if (std::abs(cell.y - other_cell.y) < 0.2 &&
        std::abs(std::abs(cell.x - other_cell.x) - 1.0) < 0.2) {  // Same price, adjacent time
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
  
  // Simple placeholder rendering to allow compilation
  ImGui::Text("Footprint Panel - Placeholder View");
  ImGui::Text("Symbol: %s", config_.symbol.c_str());
  
  // Show some basic controls
  ImGui::Separator();
  ImGui::Checkbox("Show Volume Labels", &show_volume_labels_);
  ImGui::Checkbox("Show Delta Indicator", &show_delta_indicator_);
  ImGui::SliderFloat("Delta Threshold", &delta_threshold_, -100.0f, 100.0f);
  
  // Number formatting options
  const char* number_formats[] = {"Raw", "Thousands (K)", "Millions (M)", "Scientific", "Custom Decimal"};
  int current_format = static_cast<int>(number_format_);
  if (ImGui::Combo("Number Format", &current_format, number_formats, 5)) {
    number_format_ = static_cast<NumberFormat>(current_format);
  }

  if (number_format_ == NumberFormat::CustomDecimal) {
    ImGui::SliderInt("Decimal Places", &custom_decimal_places_, 0, 8);
  }

  // Placeholder for the actual footprint chart
  ImGui::Separator();
  ImGui::Text("Footprint Chart Area");
  
  // Create a child window to simulate the chart area
  if (ImGui::BeginChild("FootprintChartArea", ImVec2(0, 300), true)) {
    ImGui::Text("Footprint visualization would appear here");
    ImGui::Text("X-axis: Time");
    ImGui::Text("Y-axis: Price");
    ImGui::Text("Color/Size: Volume/Delta");
    
    // Example of how cells might be visualized
    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   
    if (canvas_sz.x < 50) canvas_sz.x = 50.0f;
    if (canvas_sz.y < 50) canvas_sz.y = 50.0f;
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // Draw a simple grid as a placeholder
    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
    draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(30, 30, 30, 255));
    draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(200, 200, 200, 255));
    
    // Draw some sample "cells" as rectangles
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 5; j++) {
        ImVec2 cell_p0 = ImVec2(canvas_p0.x + i * (canvas_sz.x / 10), canvas_p0.y + j * (canvas_sz.y / 5));
        ImVec2 cell_p1 = ImVec2(cell_p0.x + (canvas_sz.x / 10) - 2, cell_p0.y + (canvas_sz.y / 5) - 2);
        
        // Random color based on position for demonstration
        ImU32 color = IM_COL32(100 + i * 15, 100 + j * 30, 150, 200);
        draw_list->AddRectFilled(cell_p0, cell_p1, color);
        draw_list->AddRect(cell_p0, cell_p1, IM_COL32(255, 255, 255, 100));
      }
    }
    
    ImGui::EndChild();
  }
  
  end_panel_window();
}
}
