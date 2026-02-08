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
#include "ui/haptic_feedback.hpp"
#include "ui/tooltips.hpp"

// Shorter aliases for commonly used types
using BTQuant::RenderEngine::OrderbookData;
using BTQuant::RenderEngine::TradeData;
// Note: CandleCluster is used with full qualification for consistency

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

FootprintPanel::FootprintPanel(const PanelConfig& config,
                               RenderEngine::MarketMicrostructureRenderer* renderer)
    : PanelBase(config),
      renderer_(renderer),
      data_type_(Data::UnifiedDataPipeline::DataType::FOOTPRINT),
      volume_data_type_(Data::VolumeDataType::Delta),
      time_aggregation_type_(Data::TimeAggregationType::T_1MIN),
      volume_based_n_contracts_(1000),
      tick_based_n_ticks_(100),
      price_aggregation_type_(Data::PriceAggregationType::P_1TICK),
      custom_price_aggregation_value_(0.1),
      zoom_sensitivity_(1.0f) {
  // Initialize renderer with the current aggregation settings
  if (renderer_) {
    renderer_->setTimeAggregationType(time_aggregation_type_);
    renderer_->setVolumeBasedNContracts(volume_based_n_contracts_);
    renderer_->setTickBasedNTicks(tick_based_n_ticks_);
    renderer_->setPriceAggregationType(price_aggregation_type_);
    renderer_->setCustomPriceAggregationValue(custom_price_aggregation_value_);
  }

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
  // Aggregation is handled by MarketMicrostructureRenderer
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
  std::lock_guard<std::mutex> lock(data_mutex_);
  begin_panel_window();

  if (!renderer_) {
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Renderer unavailable");
    end_panel_window();
    return;
  }

  // Main data type selector (all 16 types from UnifiedDataPipeline::DataType)
  const char* data_type_names[] = {"OHLC",           "Orderbook", "Trades",     "VolumeProfile",
                                   "Footprint",      "TPO",       "Metrics",    "Alerts",
                                   "HeikinAshi",     "Renko",     "LineBreak",  "Kagi",
                                   "PointAndFigure", "RangeBars", "VolumeBars", "TickBars"};

  int current_data_type = static_cast<int>(data_type_);
  if (ImGui::BeginCombo("Data Type##MainDataTypeSelector", data_type_names[current_data_type])) {
    for (int i = 0; i < 16; i++) {  // 16 types to match enum
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

  // Volume data type selector for footprint visualization (all 17 types)
  const char* volume_data_type_names[] = {
      "Trades",  "BuyTrades",  "SellTrades",  "Volume",      "BuyVolume", "SellVolume",
      "BuyVol%", "SellVol%",   "BuySellVol",  "Delta",       "Delta%",    "CumulDelta",
      "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol", "SplitVol"};

  int current_vol_data_type = static_cast<int>(volume_data_type_);
  if (ImGui::BeginCombo("Footprint Mode##VolumeDataTypeSelector",
                        volume_data_type_names[current_vol_data_type])) {
    for (int i = 0; i < 17; i++) {  // 17 types to match requirement (including SplitVol)
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
    // Add haptic feedback for button interaction
    BTQuant::UI::HapticFeedback::getInstance().triggerForSubtleInteraction();
  }
  // Show standardized tooltip for the button
  BTQuant::UI::show_control_tooltip("footprint_panel_reset_view");

  ImGui::SameLine();
  ImGui::Checkbox("Volume Labels", &show_volume_labels_);
  ImGui::SameLine();
  ImGui::Checkbox("Delta Indicator", &show_delta_indicator_);
  ImGui::SameLine();
  static bool show_grid = true;
  ImGui::Checkbox("Grid", &show_grid);

  // Time aggregation selector
  const char* time_agg_names[] = {"1min",  "5min",  "15min",        "30min",     "1hour",
                                  "2hour", "4hour", "Volume-based", "Tick-based"};

  int current_time_agg = static_cast<int>(time_aggregation_type_);
  if (ImGui::BeginCombo("Time Agg", time_agg_names[current_time_agg])) {
    for (int i = 0; i < 9; i++) {
      bool is_selected = (current_time_agg == i);
      if (ImGui::Selectable(time_agg_names[i], is_selected)) {
        current_time_agg = i;
        time_aggregation_type_ = static_cast<Data::TimeAggregationType>(i);
        // Update renderer with new time aggregation type
        if (renderer_) {
          renderer_->setTimeAggregationType(time_aggregation_type_);
        }
        // Mark data as dirty to trigger immediate rendering update
        data_dirty_.store(true, std::memory_order_release);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Show custom value inputs if volume-based or tick-based aggregation is selected
  if (time_aggregation_type_ == Data::TimeAggregationType::VOLUME_BASED) {
    ImGui::SameLine();
    ImGui::Text("Every");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    int temp_volume_n = volume_based_n_contracts_;
    if (ImGui::InputInt("##VolumeBasedN", &temp_volume_n, 1, 10)) {
      volume_based_n_contracts_ = std::max(1, temp_volume_n);  // Ensure minimum value of 1
      // Update renderer with new volume-based N contracts value
      if (renderer_) {
        renderer_->setVolumeBasedNContracts(volume_based_n_contracts_);
      }
      // Mark data as dirty to trigger immediate rendering update
      data_dirty_.store(true, std::memory_order_release);
    }
    ImGui::SameLine();
    ImGui::Text("contracts");
  } else if (time_aggregation_type_ == Data::TimeAggregationType::TICK_BASED) {
    ImGui::SameLine();
    ImGui::Text("Every");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    int temp_tick_n = tick_based_n_ticks_;
    if (ImGui::InputInt("##TickBasedN", &temp_tick_n, 1, 10)) {
      tick_based_n_ticks_ = std::max(1, temp_tick_n);  // Ensure minimum value of 1
      // Update renderer with new tick-based N ticks value
      if (renderer_) {
        renderer_->setTickBasedNTicks(tick_based_n_ticks_);
      }
      // Mark data as dirty to trigger immediate rendering update
      data_dirty_.store(true, std::memory_order_release);
    }
    ImGui::SameLine();
    ImGui::Text("ticks");
  }

  // Price aggregation selector
  const char* price_agg_names[] = {"1 Tick", "5 Ticks", "10 Ticks", "0.1%", "0.5%", "1%", "Custom"};

  int current_price_agg = static_cast<int>(price_aggregation_type_);
  if (ImGui::BeginCombo("Price Agg", price_agg_names[current_price_agg])) {
    for (int i = 0; i < 7; i++) {
      bool is_selected = (current_price_agg == i);
      if (ImGui::Selectable(price_agg_names[i], is_selected)) {
        current_price_agg = i;
        price_aggregation_type_ = static_cast<Data::PriceAggregationType>(i);
        // Update renderer with new price aggregation type
        if (renderer_) {
          renderer_->setPriceAggregationType(price_aggregation_type_);
          renderer_->notifyPriceAggregationChanged();
        }
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
      // Update renderer with new custom price aggregation value
      if (renderer_) {
        renderer_->setCustomPriceAggregationValue(custom_price_aggregation_value_);
        renderer_->notifyPriceAggregationChanged();
      }
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
  ImGui::SameLine();
  ImGui::Checkbox("Vol Filter", &enable_volume_filter_);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(120);
  double min_thresh = 0.0;
  double max_thresh = 100000.0;
  ImGui::SliderScalar("Vol Thresh", ImGuiDataType_Double, &volume_threshold_, &min_thresh,
                      &max_thresh, "%.0f");

  // Zoom sensitivity control - affects how much cell sizes change with zoom level
  ImGui::SameLine();
  ImGui::SetNextItemWidth(120);
  ImGui::SliderFloat("Zoom Sens", &zoom_sensitivity_, 0.1f, 3.0f, "%.1f");

  // Number formatting options
  ImGui::Separator();
  ImGui::Text("Number Formatting:");
  ImGui::SameLine();

  // Combo box for number format selection
  const char* format_items[] = {"Raw", "K (Thousands)", "M (Millions)", "Scientific",
                                "Custom Decimal"};
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
  auto cluster_cells = renderer_->getClusterCells();  // Get ClusterCell data
  auto stats = renderer_->getStats();

  // Base time for absolute labeling (relative to 30s window)
  double base_time_sec = static_cast<double>(stats.lastUpdateTimeNs) / 1'000'000'000.0 - 30.0;

  if (ImPlot::BeginPlot("##FootprintPlot", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {
    // Axis Setup
    ImPlot::SetupAxes("Time (s)", "Price", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

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
      for (const auto& c : clusters) {
        p_min = std::min(p_min, (float)c.centerY);
        p_max = std::max(p_max, (float)c.centerY);
      }
      ImPlot::SetupAxisLimits(ImAxis_Y1, (double)p_min - 10, (double)p_max + 10, ImPlotCond_Once);
    }

    // Custom Time Formatting (C++26 lambda)
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

    // Get draw list for custom rendering
    auto* draw_list = ImPlot::GetPlotDrawList();

    // Get visible plot limits to determine which time bars and price levels are visible
    ImPlotRect limits = ImPlot::GetPlotLimits();
    double x_min = limits.X.Min;
    double x_max = limits.X.Max;
    double y_min = limits.Y.Min;
    double y_max = limits.Y.Max;

    // Calculate zoom factors for adaptive cell sizing
    double x_range_full = 30.0;    // Assuming 30-second window as per base_time_sec setup
    double y_range_full = 1000.0;  // Placeholder - will be replaced by actual data range
    if (!clusters.empty()) {
      float p_min = clusters[0].centerY;
      float p_max = clusters[0].centerY;
      for (const auto& c : clusters) {
        p_min = std::min(p_min, (float)c.centerY);
        p_max = std::max(p_max, (float)c.centerY);
      }
      y_range_full = p_max - p_min + 20;  // Add some padding
    }

    // Calculate zoom factors - higher values mean more zoomed in
    double x_zoom_factor = x_range_full / (x_max - x_min);
    double y_zoom_factor = y_range_full / (y_max - y_min);

    // Use separate zoom factors for X and Y dimensions to preserve aspect ratio
    // This allows for more precise control over cell stretching in each direction
    double x_adjusted_zoom = std::max(0.01, x_zoom_factor);
    double y_adjusted_zoom = std::max(0.01, y_zoom_factor);

    // Calculate an overall zoom factor as geometric mean for general purposes
    double zoom_factor = std::sqrt(x_adjusted_zoom * y_adjusted_zoom);

    // Normalize zoom factor to a more intuitive range
    zoom_factor = std::max(0.01, zoom_factor);  // Prevent extremely small values

    // Store zoom factor for use in debug display
    double effective_zoom_factor = zoom_factor * static_cast<double>(zoom_sensitivity_);

    // Calculate max volume across all visible cells for adaptive alpha calculation
    double max_volume = 0.0;

    // Collect all visible cells for imbalance detection
    std::vector<FootprintCell> all_cells;

    // ENHANCED MAIN RENDER LOOP: Iterate through visible time bars and price levels
    // Retrieve ClusterCell data and switch on active VolumeAnalysisType to determine displayed
    // value

    // First, collect all visible clusters and organize them by time and price levels
    std::map<double, std::map<double, const BTQuant::RenderEngine::CandleCluster*>>
        visible_clusters;

    // Calculate max volume across all visible clusters for adaptive alpha calculation
    // This is done in the same loop to avoid a second iteration
    for (const auto& cluster : clusters) {
      // Check if cluster is within visible bounds
      if (cluster.centerX >= x_min && cluster.centerX <= x_max && cluster.centerY >= y_min &&
          cluster.centerY <= y_max) {
        // Organize clusters by time (x-axis) and price (y-axis) for efficient iteration
        visible_clusters[cluster.centerX][cluster.centerY] = &cluster;

        // Also calculate max volume for adaptive alpha calculation
        // Use the appropriate value based on the active VolumeAnalysisType
        double analysis_value = 0.0;

        switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
          case BTQuant::Data::VolumeAnalysisType::Trades:
            analysis_value = static_cast<double>(cluster.tradeCount);
            break;
          case BTQuant::Data::VolumeAnalysisType::BuyTrades: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            if (total_vol > 0) {
              double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
              analysis_value = static_cast<double>(cluster.tradeCount) * buy_ratio;
            } else {
              analysis_value =
                  static_cast<double>(cluster.tradeCount) * 0.5;  // Equal split if no volume
            }
          } break;
          case BTQuant::Data::VolumeAnalysisType::SellTrades: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            if (total_vol > 0) {
              double sell_ratio = static_cast<double>(cluster.askVolume) / total_vol;
              analysis_value = static_cast<double>(cluster.tradeCount) * sell_ratio;
            } else {
              analysis_value =
                  static_cast<double>(cluster.tradeCount) * 0.5;  // Equal split if no volume
            }
          } break;
          case BTQuant::Data::VolumeAnalysisType::Volume:
            analysis_value = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            break;
          case BTQuant::Data::VolumeAnalysisType::BuyVolume:
            analysis_value = static_cast<double>(cluster.bidVolume);
            break;
          case BTQuant::Data::VolumeAnalysisType::SellVolume:
            analysis_value = static_cast<double>(cluster.askVolume);
            break;
          case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
            analysis_value = std::abs(static_cast<double>(cluster.bidVolume - cluster.askVolume));
            break;
          case BTQuant::Data::VolumeAnalysisType::Delta:
            analysis_value = std::abs(static_cast<double>(cluster.bidVolume - cluster.askVolume));
            break;
          case BTQuant::Data::VolumeAnalysisType::DeltaPercent: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            analysis_value =
                total_vol > 0.0
                    ? std::abs(
                          (static_cast<double>(cluster.bidVolume - cluster.askVolume) / total_vol) *
                          100.0)
                    : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
            analysis_value = std::abs(static_cast<double>(cluster.bidVolume - cluster.askVolume));
            break;
          case BTQuant::Data::VolumeAnalysisType::AverageSize: {
            int total_count = cluster.tradeCount;
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            analysis_value = total_count > 0 ? total_vol / static_cast<double>(total_count) : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::AverageBuySize: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            int buy_count = cluster.tradeCount;
            if (total_vol > 0) {
              double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
              buy_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * buy_ratio);
            }
            analysis_value = buy_count > 0 ? static_cast<double>(cluster.bidVolume) /
                                                 static_cast<double>(buy_count)
                                           : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::AverageSellSize: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            int sell_count = cluster.tradeCount;
            if (total_vol > 0) {
              double sell_ratio = static_cast<double>(cluster.askVolume) / total_vol;
              sell_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * sell_ratio);
            }
            analysis_value = sell_count > 0 ? static_cast<double>(cluster.askVolume) /
                                                  static_cast<double>(sell_count)
                                            : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
            analysis_value = static_cast<double>(cluster.maxSingleTradeVolume);
            break;
          case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            analysis_value = total_vol > 0.0
                                 ? (static_cast<double>(cluster.bidVolume) / total_vol) * 100.0
                                 : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::SellVolumePercent: {
            double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            analysis_value = total_vol > 0.0
                                 ? (static_cast<double>(cluster.askVolume) / total_vol) * 100.0
                                 : 0.0;
          } break;
          case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
            analysis_value = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            break;
          case BTQuant::Data::VolumeAnalysisType::SplitVolume:
            // For split volume, use the maximum of buy or sell volume for scaling purposes
            analysis_value = std::max(static_cast<double>(cluster.bidVolume),
                                      static_cast<double>(cluster.askVolume));
            break;
          default:
            analysis_value = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            break;
        }

        if (analysis_value > max_volume) {
          max_volume = analysis_value;
        }
      }
    }

    // ENHANCED: Also process ClusterCell data if available
    // This provides more granular data for advanced analysis
    if (!cluster_cells.empty()) {
      // Iterate through the ClusterCell data structure [price_level][time_bucket]
      for (size_t price_idx = 0; price_idx < cluster_cells.size(); ++price_idx) {
        const auto& time_buckets = cluster_cells[price_idx];

        for (size_t time_idx = 0; time_idx < time_buckets.size(); ++time_idx) {
          const auto& cell = time_buckets[time_idx];

          // Calculate the price and time coordinates for this cell
          // This is a simplified mapping - in a real implementation, you'd need to map
          // the indices back to actual price/time coordinates
          double price_level =
              y_min + (static_cast<double>(price_idx) / cluster_cells.size()) * (y_max - y_min);
          double time_bucket =
              x_min + (static_cast<double>(time_idx) / time_buckets.size()) * (x_max - x_min);

          // Check if this cell is within visible bounds
          if (time_bucket >= x_min && time_bucket <= x_max && price_level >= y_min &&
              price_level <= y_max) {
            // Calculate analysis value based on active VolumeAnalysisType for ClusterCell data
            double analysis_value = 0.0;

            switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
              case BTQuant::Data::VolumeAnalysisType::Trades:
                analysis_value = static_cast<double>(cell.trade_count.load());
                break;
              case BTQuant::Data::VolumeAnalysisType::BuyTrades:
                analysis_value = static_cast<double>(cell.buy_trade_count.load());
                break;
              case BTQuant::Data::VolumeAnalysisType::SellTrades:
                analysis_value = static_cast<double>(cell.sell_trade_count.load());
                break;
              case BTQuant::Data::VolumeAnalysisType::Volume:
                analysis_value = cell.total_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::BuyVolume:
                analysis_value = cell.buy_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::SellVolume:
                analysis_value = cell.sell_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
                analysis_value = std::abs(cell.buy_volume - cell.sell_volume);
                break;
              case BTQuant::Data::VolumeAnalysisType::Delta:
                analysis_value = std::abs(cell.buy_volume - cell.sell_volume);
                break;
              case BTQuant::Data::VolumeAnalysisType::DeltaPercent: {
                double total_vol = cell.total_volume;
                analysis_value =
                    total_vol > 0.0
                        ? std::abs(((cell.buy_volume - cell.sell_volume) / total_vol) * 100.0)
                        : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
                analysis_value = cell.buy_volume - cell.sell_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::AverageSize: {
                int total_count = cell.trade_count.load();
                analysis_value =
                    total_count > 0 ? cell.sum_of_volumes / static_cast<double>(total_count) : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::AverageBuySize: {
                int buy_count = cell.buy_trade_count.load();
                analysis_value =
                    buy_count > 0 ? cell.buy_volume / static_cast<double>(buy_count) : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::AverageSellSize: {
                int sell_count = cell.sell_trade_count.load();
                analysis_value =
                    sell_count > 0 ? cell.sell_volume / static_cast<double>(sell_count) : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
                analysis_value = cell.max_single_trade_volume.load();
                break;
              case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent: {
                double total_vol = cell.total_volume;
                analysis_value = total_vol > 0.0 ? (cell.buy_volume / total_vol) * 100.0 : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::SellVolumePercent: {
                double total_vol = cell.total_volume;
                analysis_value = total_vol > 0.0 ? (cell.sell_volume / total_vol) * 100.0 : 0.0;
              } break;
              case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
                analysis_value = cell.total_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::SplitVolume:
                // For split volume, use the maximum of buy or sell volume for scaling purposes
                analysis_value = std::max(cell.buy_volume, cell.sell_volume);
                break;
              default:
                analysis_value = cell.total_volume;
                break;
            }

            if (analysis_value > max_volume) {
              max_volume = analysis_value;
            }

            // Add this ClusterCell to all_cells for tooltip and imbalance detection
            FootprintCell converted_cell(
                time_bucket,  // x (time)
                price_level,  // y (price)
                (x_max - x_min) /
                    static_cast<double>(time_buckets.size()),  // width (time duration)
                (y_max - y_min) /
                    static_cast<double>(cluster_cells.size()),  // height (price range)
                cell.buy_volume,                                // bid_volume
                cell.sell_volume,                               // ask_volume
                cell.trade_count.load(),                        // trade_count
                0.0                                             // vwap placeholder
            );

            // Populate the extended fields from the ClusterCell
            converted_cell.buy_trade_count = cell.buy_trade_count.load();
            converted_cell.sell_trade_count = cell.sell_trade_count.load();
            converted_cell.max_single_trade_volume = cell.max_single_trade_volume.load();

            // Set delta based on active VolumeAnalysisType
            switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
              case BTQuant::Data::VolumeAnalysisType::Delta:
              case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
                converted_cell.delta = cell.buy_volume - cell.sell_volume;
                break;
              case BTQuant::Data::VolumeAnalysisType::DeltaPercent: {
                double total_vol = cell.total_volume;
                converted_cell.delta =
                    total_vol > 0.0 ? ((cell.buy_volume - cell.sell_volume) / total_vol) * 100.0
                                    : 0.0;
              } break;
              default:
                converted_cell.delta = cell.buy_volume - cell.sell_volume;
                break;
            }

            all_cells.push_back(converted_cell);
          }
        }
      }
    }

    // Prevent division by zero
    if (max_volume <= 0.0) {
      max_volume = 1.0;  // Default to 1 to prevent division by zero
    }

    // ENHANCED: Iterate through visible time bars and price levels with improved efficiency
    // Process each visible cluster according to the active VolumeAnalysisType
    for (const auto& [time_level, price_clusters] : visible_clusters) {
      for (const auto& [price_level, cluster_ptr] : price_clusters) {
        const auto& cluster = *cluster_ptr;

        // Convert cluster to footprint cell with all relevant data
        FootprintCell cell(cluster.centerX,                         // x (time)
                           cluster.centerY,                         // y (price)
                           cluster.width,                           // width (time duration)
                           cluster.height,                          // height (price range)
                           static_cast<double>(cluster.bidVolume),  // bid_volume
                           static_cast<double>(cluster.askVolume),  // ask_volume
                           cluster.tradeCount,                      // trade_count
                           cluster.vwap                             // vwap
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
            cell.ask_volume = 0.0;  // Not applicable for trades count
            cell.delta = static_cast<double>(cluster.tradeCount);
            break;

          case BTQuant::Data::VolumeAnalysisType::BuyTrades:
            // Estimate buy trades based on volume ratio
            {
              double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              if (total_vol > 0) {
                double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
                cell.bid_volume = static_cast<double>(cluster.tradeCount) * buy_ratio;
                cell.ask_volume = 0.0;  // Not applicable for buy trades count
                cell.delta = cell.bid_volume;
              } else {
                cell.bid_volume =
                    static_cast<double>(cluster.tradeCount) * 0.5;  // Equal split if no volume
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
                cell.bid_volume = 0.0;          // Not applicable for sell trades count
                cell.delta = -cell.ask_volume;  // Negative for sell trades
              } else {
                cell.ask_volume =
                    static_cast<double>(cluster.tradeCount) * 0.5;  // Equal split if no volume
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
            cell.ask_volume = 0.0;  // Not applicable for buy volume
            cell.delta = static_cast<double>(cluster.bidVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::SellVolume:
            // Display sell volume only
            cell.ask_volume = static_cast<double>(cluster.askVolume);
            cell.bid_volume = 0.0;                                 // Not applicable for sell volume
            cell.delta = -static_cast<double>(cluster.askVolume);  // Negative for sell volume
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
              cell.delta =
                  total_vol > 0.0
                      ? (static_cast<double>(cluster.bidVolume - cluster.askVolume) / total_vol) *
                            100.0
                      : 0.0;
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
              int buy_count = cluster.tradeCount;  // Start with total count
              if (total_vol > 0) {
                double buy_ratio = static_cast<double>(cluster.bidVolume) / total_vol;
                buy_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * buy_ratio);
              }
              cell.bid_volume = static_cast<double>(cluster.bidVolume);
              cell.ask_volume = static_cast<double>(cluster.askVolume);
              cell.delta = buy_count > 0 ? static_cast<double>(cluster.bidVolume) /
                                               static_cast<double>(buy_count)
                                         : 0.0;
            }
            break;

          case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
            // Display average size of sell trades
            {
              // Estimate sell count based on volume ratio
              double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              int sell_count = cluster.tradeCount;  // Start with total count
              if (total_vol > 0) {
                double sell_ratio = static_cast<double>(cluster.askVolume) / total_vol;
                sell_count = static_cast<int>(static_cast<double>(cluster.tradeCount) * sell_ratio);
              }
              cell.bid_volume = static_cast<double>(cluster.bidVolume);
              cell.ask_volume = static_cast<double>(cluster.askVolume);
              cell.delta = sell_count > 0 ? static_cast<double>(cluster.askVolume) /
                                                static_cast<double>(sell_count)
                                          : 0.0;
            }
            break;

          case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
            // Display maximum volume of a single trade
            cell.bid_volume = static_cast<double>(cluster.bidVolume);
            cell.ask_volume = static_cast<double>(cluster.askVolume);
            cell.delta = static_cast<double>(cluster.maxSingleTradeVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
            // Display percentage of buy volume
            {
              double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              cell.bid_volume = static_cast<double>(cluster.bidVolume);
              cell.ask_volume = static_cast<double>(cluster.askVolume);
              cell.delta = total_vol > 0.0
                               ? (static_cast<double>(cluster.bidVolume) / total_vol) * 100.0
                               : 0.0;
            }
            break;

          case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
            // Display percentage of sell volume
            {
              double total_vol = static_cast<double>(cluster.bidVolume + cluster.askVolume);
              cell.bid_volume = static_cast<double>(cluster.bidVolume);
              cell.ask_volume = static_cast<double>(cluster.askVolume);
              cell.delta = total_vol > 0.0
                               ? (static_cast<double>(cluster.askVolume) / total_vol) * 100.0
                               : 0.0;
            }
            break;

          case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
            // Display volume filtered by specific criteria
            cell.bid_volume = static_cast<double>(cluster.bidVolume);
            cell.ask_volume = static_cast<double>(cluster.askVolume);
            cell.delta = static_cast<double>(cluster.bidVolume + cluster.askVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::SplitVolume:
            // For split volume mode, preserve original bid/ask volumes for split display
            cell.bid_volume = static_cast<double>(cluster.bidVolume);
            cell.ask_volume = static_cast<double>(cluster.askVolume);
            cell.delta = static_cast<double>(
                cluster.bidVolume - cluster.askVolume);  // Keep delta for other calculations
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

    // ENHANCED: Perform additional optimizations and calculations based on the active
    // VolumeAnalysisType This allows for more sophisticated analysis depending on the selected view
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
            cumulative_sum += cell.delta;  // Delta already contains the difference
            cell.delta = cumulative_sum;   // Update with cumulative value
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

    // Detect imbalances
    std::vector<FootprintCell> diagonal_imbalances;
    std::vector<FootprintCell> stacked_imbalances;
    detectImbalances(all_cells, diagonal_imbalances, stacked_imbalances);

    // ENHANCED: Group only visible clusters by time (x-coordinate) to calculate time-bar summaries
    // This optimization reduces redundant processing by only considering visible clusters
    // The grouping now takes into account the active VolumeAnalysisType for more accurate
    // aggregations
    std::map<double, std::vector<const BTQuant::RenderEngine::CandleCluster*>> clusters_by_time;

    // Iterate through visible clusters to group by time
    for (const auto& cluster : clusters) {
      // Check if cluster is within visible bounds before grouping
      if (cluster.centerX >= x_min && cluster.centerX <= x_max && cluster.centerY >= y_min &&
          cluster.centerY <= y_max) {
        // Round x to nearest time unit to group clusters by time bar
        // Using a more precise rounding method for better time alignment
        double time_key = std::round(cluster.centerX * 10.0) / 10.0;  // Adjust precision as needed
        clusters_by_time[time_key].push_back(&cluster);
      }
    }

    // ENHANCED: Calculate cumulative and other summaries for each time bar based on active
    // VolumeAnalysisType Optimized to reduce redundant calculations and improve performance
    std::map<double, double>
        cumulative_values;  // Cumulative value by time based on active analysis type
    std::map<double, std::pair<double, double>> poc_info;  // POC price and volume by time
    std::map<double, double>
        time_bar_net_values;  // Net value by time based on active analysis type
    std::map<double, double>
        time_bar_total_values;  // Total value by time based on active analysis type

    double running_cumulative_value = 0.0;

    // Process each time bar to calculate aggregated values based on the active VolumeAnalysisType
    for (const auto& [time_key, time_clusters] : clusters_by_time) {
      double time_net_value = 0.0;
      double time_total_value = 0.0;
      double max_volume_in_time_bar = 0.0;
      double poc_price = 0.0;
      double max_analysis_value_in_time_bar =
          0.0;  // Track max value for POC based on analysis type

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
                cluster_value =
                    static_cast<double>(cluster->tradeCount) * 0.5;  // Equal split if no volume
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
                cluster_value =
                    static_cast<double>(cluster->tradeCount) * 0.5;  // Equal split if no volume
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

          case BTQuant::Data::VolumeAnalysisType::DeltaPercent: {
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            cluster_value =
                total_vol > 0.0
                    ? (static_cast<double>(cluster->bidVolume - cluster->askVolume) / total_vol) *
                          100.0
                    : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
            cluster_value = static_cast<double>(cluster->bidVolume - cluster->askVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::AverageSize: {
            int total_count = cluster->tradeCount;
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            cluster_value = total_count > 0 ? total_vol / static_cast<double>(total_count) : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::AverageBuySize: {
            // Estimate buy count based on volume ratio
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            int buy_count = cluster->tradeCount;  // Start with total count
            if (total_vol > 0) {
              double buy_ratio = static_cast<double>(cluster->bidVolume) / total_vol;
              buy_count = static_cast<int>(static_cast<double>(cluster->tradeCount) * buy_ratio);
            }
            cluster_value = buy_count > 0 ? static_cast<double>(cluster->bidVolume) /
                                                static_cast<double>(buy_count)
                                          : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::AverageSellSize: {
            // Estimate sell count based on volume ratio
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            int sell_count = cluster->tradeCount;  // Start with total count
            if (total_vol > 0) {
              double sell_ratio = static_cast<double>(cluster->askVolume) / total_vol;
              sell_count = static_cast<int>(static_cast<double>(cluster->tradeCount) * sell_ratio);
            }
            cluster_value = sell_count > 0 ? static_cast<double>(cluster->askVolume) /
                                                 static_cast<double>(sell_count)
                                           : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
            // Use the actual max single trade volume from the cluster
            cluster_value = static_cast<double>(cluster->maxSingleTradeVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent: {
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            cluster_value = total_vol > 0.0
                                ? (static_cast<double>(cluster->bidVolume) / total_vol) * 100.0
                                : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::SellVolumePercent: {
            double total_vol = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            cluster_value = total_vol > 0.0
                                ? (static_cast<double>(cluster->askVolume) / total_vol) * 100.0
                                : 0.0;
          } break;

          case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
            cluster_value = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            break;

          case BTQuant::Data::VolumeAnalysisType::SplitVolume:
            // For split volume, use the maximum of buy or sell volume for aggregation
            cluster_value = std::max(static_cast<double>(cluster->bidVolume),
                                     static_cast<double>(cluster->askVolume));
            break;

          default:
            cluster_value = static_cast<double>(cluster->bidVolume + cluster->askVolume);
            break;
        }

        // Update time bar summary values based on the active VolumeAnalysisType
        time_net_value += cluster_value;
        time_total_value += std::abs(cluster_value);  // Use absolute value for total

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
      poc_info[time_key] =
          std::make_pair(poc_price, max_analysis_value_in_time_bar);  // Use analysis value for POC
    }

    // ENHANCED: Render all cells with volume threshold filtering
    // Apply the calculated max volume for adaptive alpha and highlight imbalances
    for (const auto& cell : all_cells) {
      // Calculate the volume value based on the active VolumeAnalysisType for threshold comparison
      double cell_volume = 0.0;

      switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
        case BTQuant::Data::VolumeAnalysisType::Trades:
          cell_volume = static_cast<double>(cell.trade_count);
          break;
        case BTQuant::Data::VolumeAnalysisType::BuyTrades:
          cell_volume = cell.bid_volume;  // Already contains buy trades estimate
          break;
        case BTQuant::Data::VolumeAnalysisType::SellTrades:
          cell_volume = cell.ask_volume;  // Already contains sell trades estimate
          break;
        case BTQuant::Data::VolumeAnalysisType::Volume:
          cell_volume = cell.bid_volume + cell.ask_volume;
          break;
        case BTQuant::Data::VolumeAnalysisType::BuyVolume:
          cell_volume = cell.bid_volume;
          break;
        case BTQuant::Data::VolumeAnalysisType::SellVolume:
          cell_volume = cell.ask_volume;
          break;
        case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
          cell_volume = std::abs(cell.bid_volume - cell.ask_volume);
          break;
        case BTQuant::Data::VolumeAnalysisType::Delta:
          cell_volume = std::abs(cell.delta);
          break;
        case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
          cell_volume = std::abs(cell.delta);  // Delta already contains percentage
          break;
        case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
          cell_volume = std::abs(cell.delta);
          break;
        case BTQuant::Data::VolumeAnalysisType::AverageSize:
          cell_volume = std::abs(cell.delta);  // Delta contains average size
          break;
        case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
          cell_volume = std::abs(cell.delta);  // Delta contains average buy size
          break;
        case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
          cell_volume = std::abs(cell.delta);  // Delta contains average sell size
          break;
        case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
          cell_volume = cell.max_single_trade_volume;
          break;
        case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
          cell_volume = std::abs(cell.delta);  // Delta contains percentage
          break;
        case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
          cell_volume = std::abs(cell.delta);  // Delta contains percentage
          break;
        case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
          cell_volume = cell.bid_volume + cell.ask_volume;
          break;
        case BTQuant::Data::VolumeAnalysisType::SplitVolume:
          // For split volume, use the maximum of buy or sell volume for threshold comparison
          cell_volume = std::max(cell.bid_volume, cell.ask_volume);
          break;
        default:
          cell_volume = cell.bid_volume + cell.ask_volume;  // Default to total volume
          break;
      }

      // Only apply volume threshold filtering if the feature is enabled
      if (enable_volume_filter_ && cell_volume < volume_threshold_) {
        // Render greyed-out cell for values below threshold
        renderFilteredCell(cell, draw_list, max_volume, zoom_factor);
      } else {
        // Render the cell normally with the calculated max volume for adaptive alpha
        renderCell(cell, draw_list, max_volume, diagonal_imbalances, stacked_imbalances,
                   zoom_factor);
      }
    }

    // Handle tooltip for the cell under the mouse cursor
    if (ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse_pos_plot = ImPlot::GetPlotMousePos();

      // Find the cell under the mouse cursor
      for (const auto& cell : all_cells) {
        double x1 = cell.x - cell.width * 0.48;
        double x2 = cell.x + cell.width * 0.48;
        double y1 = cell.y - cell.height * 0.48;
        double y2 = cell.y + cell.height * 0.48;

        if (mouse_pos_plot.x >= x1 && mouse_pos_plot.x <= x2 && mouse_pos_plot.y >= y1 &&
            mouse_pos_plot.y <= y2) {
          // Calculate the volume value based on the active VolumeAnalysisType for threshold
          // comparison
          double cell_volume = 0.0;

          switch (static_cast<BTQuant::Data::VolumeAnalysisType>(volume_data_type_)) {
            case BTQuant::Data::VolumeAnalysisType::Trades:
              cell_volume = static_cast<double>(cell.trade_count);
              break;
            case BTQuant::Data::VolumeAnalysisType::BuyTrades:
              cell_volume = cell.bid_volume;  // Already contains buy trades estimate
              break;
            case BTQuant::Data::VolumeAnalysisType::SellTrades:
              cell_volume = cell.ask_volume;  // Already contains sell trades estimate
              break;
            case BTQuant::Data::VolumeAnalysisType::Volume:
              cell_volume = cell.bid_volume + cell.ask_volume;
              break;
            case BTQuant::Data::VolumeAnalysisType::BuyVolume:
              cell_volume = cell.bid_volume;
              break;
            case BTQuant::Data::VolumeAnalysisType::SellVolume:
              cell_volume = cell.ask_volume;
              break;
            case BTQuant::Data::VolumeAnalysisType::BuySellVolume:
              cell_volume = std::abs(cell.bid_volume - cell.ask_volume);
              break;
            case BTQuant::Data::VolumeAnalysisType::Delta:
              cell_volume = std::abs(cell.delta);
              break;
            case BTQuant::Data::VolumeAnalysisType::DeltaPercent:
              cell_volume = std::abs(cell.delta);  // Delta already contains percentage
              break;
            case BTQuant::Data::VolumeAnalysisType::CumulativeDelta:
              cell_volume = std::abs(cell.delta);
              break;
            case BTQuant::Data::VolumeAnalysisType::AverageSize:
              cell_volume = std::abs(cell.delta);  // Delta contains average size
              break;
            case BTQuant::Data::VolumeAnalysisType::AverageBuySize:
              cell_volume = std::abs(cell.delta);  // Delta contains average buy size
              break;
            case BTQuant::Data::VolumeAnalysisType::AverageSellSize:
              cell_volume = std::abs(cell.delta);  // Delta contains average sell size
              break;
            case BTQuant::Data::VolumeAnalysisType::MaxOneTradeVolume:
              cell_volume = cell.max_single_trade_volume;
              break;
            case BTQuant::Data::VolumeAnalysisType::BuyVolumePercent:
              cell_volume = std::abs(cell.delta);  // Delta contains percentage
              break;
            case BTQuant::Data::VolumeAnalysisType::SellVolumePercent:
              cell_volume = std::abs(cell.delta);  // Delta contains percentage
              break;
            case BTQuant::Data::VolumeAnalysisType::FilteredVolume:
              cell_volume = cell.bid_volume + cell.ask_volume;
              break;
            case BTQuant::Data::VolumeAnalysisType::SplitVolume:
              // For split volume, use the maximum of buy or sell volume for threshold comparison
              cell_volume = std::max(cell.bid_volume, cell.ask_volume);
              break;
            default:
              cell_volume = cell.bid_volume + cell.ask_volume;  // Default to total volume
              break;
          }

          // Show tooltip for this cell regardless of whether it meets the threshold
          // Add info about whether the cell is filtered
          std::string tooltip = getCellTooltip(cell);
          if (enable_volume_filter_ && cell_volume < volume_threshold_) {
            tooltip += "\n[Filtered: Below volume threshold]";
          }

          ImGui::SetTooltip("%s", tooltip.c_str());
          break;  // Only show tooltip for the first cell found under cursor
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
        snprintf(header_text, sizeof(header_text), "VOL:%.0f Δ:%+.0f ΣΔ:%+.0f POC:%.2f",
                 total_volume, net_delta, cumulative_delta, poc_price);

        // Convert time to pixel coordinates for header positioning
        ImVec2 header_pos = ImPlot::PlotToPixels(
            time_key, y_max + 5.0);  // Position header slightly above the highest price

        // Try to use monospace font for alignment
        ImFont* mono_font = nullptr;

        // Look for a monospace font in the loaded fonts
        for (int i = 0; i < ImGui::GetIO().Fonts->Fonts.Size; i++) {
          ImFont* font = ImGui::GetIO().Fonts->Fonts[i];
          const char* font_name = font->GetDebugName();

          // Check if this looks like a monospace font by name
          if (font_name &&
              (strstr(font_name, "Mono") != nullptr || strstr(font_name, "Consolas") != nullptr ||
               strstr(font_name, "Courier") != nullptr || strstr(font_name, "Fixed") != nullptr ||
               strstr(font_name, "Code") != nullptr)) {
            mono_font = font;
            break;
          }
        }

        // Calculate text size for background rectangle (with potential font)
        ImVec2 text_size;
        if (mono_font) {
          ImGui::PushFont(mono_font);
          text_size = ImGui::CalcTextSize(header_text);
          ImGui::PopFont();
        } else {
          text_size = ImGui::CalcTextSize(header_text);
        }

        // Draw background rectangle for header with padding
        float padding_x = 6.0f;
        float padding_y = 4.0f;
        draw_list->AddRectFilled(
            ImVec2(header_pos.x - text_size.x / 2.0f - padding_x,
                   header_pos.y - text_size.y - padding_y),
            ImVec2(header_pos.x + text_size.x / 2.0f + padding_x, header_pos.y + padding_y),
            IM_COL32(40, 40, 50, 220));  // Dark semi-transparent background

        // Draw border around the header
        draw_list->AddRect(
            ImVec2(header_pos.x - text_size.x / 2.0f - padding_x,
                   header_pos.y - text_size.y - padding_y),
            ImVec2(header_pos.x + text_size.x / 2.0f + padding_x, header_pos.y + padding_y),
            IM_COL32(120, 120, 160, 200));  // Border color

        // Draw the header text with monospace font if available
        if (mono_font) {
          ImGui::PushFont(mono_font);
          draw_list->AddText(
              ImVec2(header_pos.x - text_size.x / 2.0f, header_pos.y - text_size.y - 2.0f),
              IM_COL32(220, 220, 255, 255),  // Light blue-white text for better contrast
              header_text);
          ImGui::PopFont();
        } else {
          // If no monospace font found, use regular text but try to load one
          draw_list->AddText(
              ImVec2(header_pos.x - text_size.x / 2.0f, header_pos.y - text_size.y - 2.0f),
              IM_COL32(220, 220, 255, 255),  // Light blue-white text
              header_text);
        }
      }
    }

    ImPlot::EndPlot();
  }

  // Calculate footer statistics: number of trades, average trade size, max single trade
  int total_trades = 0;
  double total_volume = 0.0;
  double max_single_trade_volume = 0.0;

  for (const auto& c : clusters) {
    total_trades += c.tradeCount;
    double cluster_total_volume = static_cast<double>(c.bidVolume + c.askVolume);
    total_volume += cluster_total_volume;

    // Update max single trade volume from the cluster's max single trade volume
    if (static_cast<double>(c.maxSingleTradeVolume) > max_single_trade_volume) {
      max_single_trade_volume = static_cast<double>(c.maxSingleTradeVolume);
    }
  }

  // Calculate average trade size
  double avg_trade_size =
      (total_trades > 0) ? total_volume / static_cast<double>(total_trades) : 0.0;

  // Render footer with smaller font below the cluster grid
  if (!clusters.empty()) {
    // Create a separator line above the footer
    ImGui::Separator();

    // Temporarily reduce font size for the footer using smaller padding
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 2.0f));

    // Scale the font down to make it smaller
    ImGui::SetWindowFontScale(0.8f);

    // Render the footer with the requested information
    ImGui::Text("Trades: %d | Avg Size: %.2f | Max Single Trade: %.2f", total_trades,
                avg_trade_size, max_single_trade_volume);

    // Restore the original font scale
    ImGui::SetWindowFontScale(1.0f);

    // Restore the original style
    ImGui::PopStyleVar(2);  // Pop ItemSpacing and FramePadding
  }

  // Enhanced Debug Overlay
  if (!clusters.empty()) {
    // Volume data type names for display
    const char* vol_type_names[] = {
        "Trades",  "BuyTrades",  "SellTrades",  "Volume",      "BuyVolume", "SellVolume",
        "BuyVol%", "SellVol%",   "BuySellVol",  "Delta",       "Delta%",    "CumulDelta",
        "AvgSize", "AvgBuySize", "AvgSellSize", "MaxTradeVol", "SplitVol"  // 17 types
    };

    // Time aggregation type names for display
    const char* time_agg_names[] = {"1min",  "5min",  "15min",        "30min",     "1hour",
                                    "2hour", "4hour", "Volume-based", "Tick-based"};

    // Price aggregation type names for display
    const char* price_agg_names[] = {"1 Tick", "5 Ticks", "10 Ticks", "0.1%",
                                     "0.5%",   "1%",      "Custom"};

    // Calculate effective zoom factor for debug display
    // Use a default value if we can't get plot limits
    double effective_zoom_factor =
        static_cast<double>(zoom_sensitivity_);  // Default to sensitivity value

    ImGui::SetCursorPos(ImVec2(10, 30));
    if (time_aggregation_type_ == Data::TimeAggregationType::VOLUME_BASED) {
      ImGui::TextColored(
          ImVec4(1, 1, 0, 1),
          "Mode: %s | Time Agg: %s (%d contracts) | Price Agg: %s | Clusters: %zu | Grid: %dx%d | "
          "δThresh: %.2f | VolThresh: %.0f | VolFilt: %s | Zoom: %.2fx",
          vol_type_names[static_cast<int>(volume_data_type_)],
          time_agg_names[static_cast<int>(time_aggregation_type_)], volume_based_n_contracts_,
          price_agg_names[static_cast<int>(price_aggregation_type_)], clusters.size(), grid_cols_,
          grid_rows_, delta_threshold_, volume_threshold_, enable_volume_filter_ ? "ON" : "OFF",
          effective_zoom_factor);
    } else if (time_aggregation_type_ == Data::TimeAggregationType::TICK_BASED) {
      ImGui::TextColored(
          ImVec4(1, 1, 0, 1),
          "Mode: %s | Time Agg: %s (%d ticks) | Price Agg: %s | Clusters: %zu | Grid: %dx%d | "
          "δThresh: %.2f | VolThresh: %.0f | VolFilt: %s | Zoom: %.2fx",
          vol_type_names[static_cast<int>(volume_data_type_)],
          time_agg_names[static_cast<int>(time_aggregation_type_)], tick_based_n_ticks_,
          price_agg_names[static_cast<int>(price_aggregation_type_)], clusters.size(), grid_cols_,
          grid_rows_, delta_threshold_, volume_threshold_, enable_volume_filter_ ? "ON" : "OFF",
          effective_zoom_factor);
    } else {
      ImGui::TextColored(ImVec4(1, 1, 0, 1),
                         "Mode: %s | Time Agg: %s | Price Agg: %s | Clusters: %zu | Grid: %dx%d | "
                         "δThresh: %.2f | VolThresh: %.0f | VolFilt: %s | Zoom: %.2fx",
                         vol_type_names[static_cast<int>(volume_data_type_)],
                         time_agg_names[static_cast<int>(time_aggregation_type_)],
                         price_agg_names[static_cast<int>(price_aggregation_type_)],
                         clusters.size(), grid_cols_, grid_rows_, delta_threshold_,
                         volume_threshold_, enable_volume_filter_ ? "ON" : "OFF",
                         effective_zoom_factor);
    }

    // Calculate statistics based on selected volume data type
    double total_value = 0.0;
    double total_bid_vol = 0.0;
    double total_ask_vol = 0.0;
    double total_trade_count = 0;
    for (const auto& c : clusters) {
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
              total_value += static_cast<double>(c.tradeCount) * 0.5;  // Equal split if no volume
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
              total_value += static_cast<double>(c.tradeCount) * 0.5;  // Equal split if no volume
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

        case Data::VolumeAnalysisType::DeltaPercent: {
          double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
          total_value += total_vol > 0.0 ? ((c.bidVolume - c.askVolume) / total_vol) * 100.0 : 0.0;
        } break;

        case Data::VolumeAnalysisType::BuyVolumePercent: {
          double total_vol = c.bidVolume + c.askVolume;
          total_value += total_vol > 0.0 ? (c.bidVolume / total_vol) * 100.0 : 0.0;
        } break;

        case Data::VolumeAnalysisType::SellVolumePercent: {
          double total_vol = c.bidVolume + c.askVolume;
          total_value += total_vol > 0.0 ? (c.askVolume / total_vol) * 100.0 : 0.0;
        } break;

        case Data::VolumeAnalysisType::CumulativeDelta:
          total_value += c.bidVolume - c.askVolume;  // Same as delta for demo
          break;

        case Data::VolumeAnalysisType::AverageSize: {
          int count = c.tradeCount;
          double vol = c.bidVolume + c.askVolume;
          total_value += count > 0 ? vol / static_cast<double>(count) : 0.0;
        } break;

        case Data::VolumeAnalysisType::AverageBuySize: {
          // Estimate buy count based on volume ratio
          double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
          int buy_count = c.tradeCount;  // Start with total count
          if (total_vol > 0) {
            double buy_ratio = static_cast<double>(c.bidVolume) / total_vol;
            buy_count = static_cast<int>(static_cast<double>(c.tradeCount) * buy_ratio);
          }
          total_value += buy_count > 0
                             ? static_cast<double>(c.bidVolume) / static_cast<double>(buy_count)
                             : 0.0;
        } break;

        case Data::VolumeAnalysisType::AverageSellSize: {
          // Estimate sell count based on volume ratio
          double total_vol = static_cast<double>(c.bidVolume + c.askVolume);
          int sell_count = c.tradeCount;  // Start with total count
          if (total_vol > 0) {
            double sell_ratio = static_cast<double>(c.askVolume) / total_vol;
            sell_count = static_cast<int>(static_cast<double>(c.tradeCount) * sell_ratio);
          }
          total_value += sell_count > 0
                             ? static_cast<double>(c.askVolume) / static_cast<double>(sell_count)
                             : 0.0;
        } break;

        case Data::VolumeAnalysisType::MaxOneTradeVolume:
          // Estimate max single trade volume as total volume divided by trade count
          total_value += c.tradeCount > 0 ? (static_cast<double>(c.bidVolume + c.askVolume) /
                                             static_cast<double>(c.tradeCount))
                                          : 0.0;
          break;

        case Data::VolumeAnalysisType::FilteredVolume:
          total_value += c.bidVolume + c.askVolume;  // Same as total volume for demo
          break;

        case Data::VolumeAnalysisType::SplitVolume:
          // For split volume, use the maximum of buy or sell volume for total calculation
          total_value += std::max(c.bidVolume, c.askVolume);
          break;

        default:
          total_value += c.bidVolume + c.askVolume;  // Default to total volume
          break;
      }
    }

    ImGui::Text("Total Value: %.2f | Bid: %.2f | Ask: %.2f | Trades: %.0f", total_value,
                total_bid_vol, total_ask_vol, total_trade_count);
  }

  end_panel_window();
}

}  // namespace BTQuant
