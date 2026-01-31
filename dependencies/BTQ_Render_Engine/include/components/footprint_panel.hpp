#pragma once

#include "MarketMicrostructureRenderer.h"
#include "panel_base.hpp"
#include "data/unified_data_pipeline.hpp"  // For DataType enum
#include "data/VolumeDataTypes.h"          // For VolumeAnalysisType and VolumeDataType enums
#include <memory>
#include <vector>
#include <atomic>

namespace BTQuant {

// Footprint Cell Structure for Exocharts-style visualization
struct FootprintCell {
  double x;             // Time position (X-axis)
  double y;             // Price position (Y-axis)
  double width;         // Cell width (time duration)
  double height;        // Cell height (price range)
  double bid_volume;    // Total bid volume
  double ask_volume;    // Total ask volume
  double delta;         // Delta (bid_volume - ask_volume)
  uint32_t trade_count; // Number of trades
  double vwap;          // Volume-weighted average price
  uint32_t buy_trade_count;      // Number of buy trades
  uint32_t sell_trade_count;     // Number of sell trades
  double max_single_trade_volume; // Maximum single trade volume
  uint64_t start_time_ns;        // Start timestamp in nanoseconds
  uint64_t end_time_ns;          // End timestamp in nanoseconds

  // Constructor
  FootprintCell(double x_pos, double y_pos, double w, double h, double bid_vol,
                double ask_vol, uint32_t count, double vwap_price)
      : x(x_pos), y(y_pos), width(w), height(h), bid_volume(bid_vol),
        ask_volume(ask_vol), delta(bid_vol - ask_vol), trade_count(count),
        vwap(vwap_price), buy_trade_count(0), sell_trade_count(0),
        max_single_trade_volume(0.0), start_time_ns(0), end_time_ns(0) {}
};

// Number Formatting Options
enum class NumberFormat {
  Raw,           // Raw numbers without any formatting
  ThousandsK,    // Format with K suffix for thousands
  MillionsM,     // Format with M suffix for millions
  Scientific,    // Scientific notation
  CustomDecimal  // Custom decimal places
};

class FootprintPanel : public PanelBase {
public:
  FootprintPanel(const PanelConfig &config,
                 RenderEngine::MarketMicrostructureRenderer *renderer);

  void update(float dt) override;
  void render() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    if (renderer_)
      renderer_->setSymbol(id);
  }

  // Configuration
  void setGridSize(int cols, int rows) {
    grid_cols_ = cols;
    grid_rows_ = rows;
  }
  void setShowVolumeLabels(bool show) { show_volume_labels_ = show; }
  void setShowDeltaIndicator(bool show) { show_delta_indicator_ = show; }
  void setDeltaThreshold(float threshold) { delta_threshold_ = threshold; }

  // Number formatting options
  void setNumberFormat(NumberFormat format) { number_format_ = format; }
  NumberFormat getNumberFormat() const { return number_format_; }
  void setCustomDecimalPlaces(int places) { custom_decimal_places_ = places; }
  int getCustomDecimalPlaces() const { return custom_decimal_places_; }

  // Data type selection
  void setDataType(Data::UnifiedDataPipeline::DataType type) { data_type_ = type; }
  Data::UnifiedDataPipeline::DataType getDataType() const { return data_type_; }

  // Volume data type selection for footprint visualization
  void setVolumeDataType(Data::VolumeDataType vol_type) { volume_data_type_ = vol_type; }
  Data::VolumeDataType getVolumeDataType() const { return volume_data_type_; }

  // Time aggregation type selection
  void setTimeAggregationType(Data::TimeAggregationType agg_type) { time_aggregation_type_ = agg_type; }
  Data::TimeAggregationType getTimeAggregationType() const { return time_aggregation_type_; }

  // Volume-based and Tick-based aggregation parameters
  void setVolumeBasedNContracts(int n) { volume_based_n_contracts_ = std::max(1, n); }
  int getVolumeBasedNContracts() const { return volume_based_n_contracts_; }
  void setTickBasedNTicks(int n) { tick_based_n_ticks_ = std::max(1, n); }
  int getTickBasedNTicks() const { return tick_based_n_ticks_; }

  // Price aggregation type selection
  void setPriceAggregationType(Data::PriceAggregationType agg_type) { price_aggregation_type_ = agg_type; }
  Data::PriceAggregationType getPriceAggregationType() const { return price_aggregation_type_; }

  // Custom price aggregation value
  void setCustomPriceAggregationValue(double value) { custom_price_aggregation_value_ = value; }
  double getCustomPriceAggregationValue() const { return custom_price_aggregation_value_; }

private:
  RenderEngine::MarketMicrostructureRenderer *renderer_;
  uint32_t symbol_id_ = 0;

  // Data Type Selection
  Data::UnifiedDataPipeline::DataType data_type_ = Data::UnifiedDataPipeline::DataType::FOOTPRINT;

  // Volume Data Type for Footprint Visualization
  Data::VolumeDataType volume_data_type_ = Data::VolumeDataType::Delta;

  // Grid Configuration (Exocharts-style: 60 columns × 100 rows)
  int grid_cols_ = 60;  // Number of time columns (minutes)
  int grid_rows_ = 100; // Number of price rows (ticks)

  // Visualization Options
  bool show_volume_labels_ = true;
  bool show_delta_indicator_ = true;
  float delta_threshold_ = 0.0f; // Threshold for delta coloring

  // Number Formatting Options
  NumberFormat number_format_ = NumberFormat::ThousandsK;  // Default to K suffix
  int custom_decimal_places_ = 2;  // Default decimal places for custom format

  // Time Aggregation Type
  Data::TimeAggregationType time_aggregation_type_ = Data::TimeAggregationType::T_1MIN;

  // Volume-based and Tick-based aggregation parameters
  int volume_based_n_contracts_ = 1000;  // Default: every 1000 contracts
  int tick_based_n_ticks_ = 100;         // Default: every 100 ticks

  // Price Aggregation Type
  Data::PriceAggregationType price_aggregation_type_ = Data::PriceAggregationType::P_1TICK;

  // Custom price aggregation value
  double custom_price_aggregation_value_ = 0.1;

  // Data dirty flag for immediate rendering updates
  std::atomic<bool> data_dirty_{true};

  // Cell Data (CPU-side aggregation)
  std::vector<FootprintCell> cells_;

  // Rendering Helpers
  ImU32 getCellColor(const FootprintCell &cell, double max_volume = 10000.0) const;
  ImU32 getDeltaColor(const FootprintCell &cell, float alpha) const;
  ImU32 getBuySellColor(const FootprintCell &cell, double max_volume, float alpha) const;
  ImU32 getVolumeIntensityColor(const FootprintCell &cell, double max_volume, float alpha) const;
  std::string getCellLabel(const FootprintCell &cell) const;
  void renderCell(const FootprintCell &cell, ImDrawList *draw_list, double max_volume = 10000.0);
  void renderCell(const FootprintCell &cell, ImDrawList *draw_list, double max_volume,
                 const std::vector<FootprintCell> &diagonal_imbalances,
                 const std::vector<FootprintCell> &stacked_imbalances);
  std::string getCellTooltip(const FootprintCell &cell) const;

  // Imbalance Detection
  bool isDiagonalImbalance(const FootprintCell& cell, const std::vector<FootprintCell>& all_cells) const;
  bool isStackedImbalance(const FootprintCell& cell, const std::vector<FootprintCell>& all_cells) const;
  void detectImbalances(const std::vector<FootprintCell>& cells,
                       std::vector<FootprintCell>& diagonal_imbalances,
                       std::vector<FootprintCell>& stacked_imbalances) const;

private:
  // Number formatting helper
  static std::string formatNumber(double value, NumberFormat format, int decimal_places);

  // Monospace font for header alignment
  mutable ImFont* monospace_font_ = nullptr;

  // Helper function to get or load monospace font
  ImFont* getOrCreateMonospaceFont() const;
};

} // namespace BTQuant
