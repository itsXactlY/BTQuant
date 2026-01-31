#pragma once

#include <imgui.h>

#include <memory>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "theme_manager.hpp"

namespace BTQuant {

// Enum for profile mode
enum class ProfileMode { Step, Right, Left, Custom };

// Struct for profile settings
struct ProfileSettings {
  int vaPercent = 70;         // Value Area percentage
  int tickStep = 1;           // Tick step size
  bool showPOC = true;        // Show Point of Control
  bool showValueArea = true;  // Show Value Area
  int colorScheme = 0;        // Color scheme index
};

/**
 * VolumeProfilePanel - Volume at Price display
 *
 * C++26 Reactive Architecture:
 * - Subscribes to MarketDataProcessor for TRADE notifications
 * - markDirty() called from callback, consumeDirty() in render()
 * - No polling timer - truly event-driven
 *
 * Shows horizontal bars representing volume traded at each price level:
 * - Buy volume (green) on right side
 * - Sell volume (red) on left side (mirrored)
 * - Point of Control (POC) highlighted
 */
class VolumeProfilePanel : public PanelBase {
 public:
  VolumeProfilePanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  ~VolumeProfilePanel() override;

  void render() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

  // Method to render mini histogram overlays on candlestick charts
  void render_mini_histograms_on_candles(ImDrawList* draw_list,
                                         const std::vector<RenderEngine::OHLCVCandle>& candles,
                                         const std::vector<double>& x_coords,
                                         const std::vector<double>& y_coords_high,
                                         const std::vector<double>& y_coords_low);

  // Enhanced method to render step profile histograms on candlesticks
  void render_step_profile_histograms(ImDrawList* draw_list,
                                      const std::vector<RenderEngine::OHLCVCandle>& candles,
                                      const std::vector<double>& x_coords,
                                      const std::vector<double>& y_coords_high,
                                      const std::vector<double>& y_coords_low,
                                      bool show_poc_line = true,
                                      int num_buckets = 8);

  // Method to render step profile histograms specifically for candlestick volume distribution
  // This is the main method for implementing the Step Profile rendering feature
  void render_candle_volume_distribution(ImDrawList* draw_list,
                                        const std::vector<RenderEngine::OHLCVCandle>& candles,
                                        const std::vector<double>& x_coords,
                                        const std::vector<double>& y_coords_high,
                                        const std::vector<double>& y_coords_low,
                                        bool show_poc_line = true,
                                        int num_buckets = 8);

  // Enhanced method to render step profile directly on candles with improved visualization
  void render_step_profile_on_candles(ImDrawList* draw_list,
                                     const std::vector<RenderEngine::OHLCVCandle>& candles,
                                     const std::vector<double>& x_coords,
                                     const std::vector<double>& y_coords_high,
                                     const std::vector<double>& y_coords_low,
                                     bool show_poc_line = true,
                                     int num_buckets_per_candle = 8);

  // Static utility method to render mini histograms directly without creating a panel instance
  // This is more efficient for use in chart panels where we don't need the full panel functionality
  static void render_mini_histograms_direct(ImDrawList* draw_list,
                                           const std::vector<RenderEngine::OHLCVCandle>& candles,
                                           const std::vector<double>& x_coords,
                                           const std::vector<double>& y_coords_high,
                                           const std::vector<double>& y_coords_low,
                                           const std::vector<RenderEngine::TradeData>& trades,
                                           bool show_poc_line = true,
                                           int num_buckets = 8);

  // Static method to render step profile directly on candles with improved visualization
  static void render_step_profile_on_candles_static(ImDrawList* draw_list,
                                                  const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                  const std::vector<double>& x_coords,
                                                  const std::vector<double>& y_coords_high,
                                                  const std::vector<double>& y_coords_low,
                                                  const std::vector<RenderEngine::TradeData>& trades,
                                                  bool show_poc_line = true,
                                                  int num_buckets_per_candle = 8);

  // Enhanced method to render step profile histograms on candlesticks with additional features
  static void render_enhanced_step_profile_on_candles(ImDrawList* draw_list,
                                                    const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                    const std::vector<double>& x_coords,
                                                    const std::vector<double>& y_coords_high,
                                                    const std::vector<double>& y_coords_low,
                                                    const std::vector<RenderEngine::TradeData>& trades,
                                                    bool show_poc_line = true,
                                                    int num_buckets_per_candle = 8,
                                                    float opacity_factor = 1.0f);

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Volume profile data
  struct VolumeLevel {
    double price;
    double buy_volume;
    double sell_volume;
    double total_volume;
  };

  std::vector<VolumeLevel> volume_profile_;
  double poc_price_ = 0.0;   // Point of Control (highest volume price)
  double max_volume_ = 0.0;  // For scaling bars
  double vah_price_ = 0.0;   // Value Area High
  double val_price_ = 0.0;   // Value Area Low

  // Configuration
  static constexpr size_t NUM_PRICE_LEVELS = 20;
  double price_bucket_size_ = 10.0;  // Price range per level

  // Profile settings
  ProfileMode profile_mode_ = ProfileMode::Step;
  ProfileSettings profile_settings_{};

  // Profile anchor markers for Custom Profile mode
  bool use_custom_time_range_ = false;  // Whether to use custom time range
  double custom_start_time_ = 0.0;     // Start time for custom range
  double custom_end_time_ = 0.0;       // End time for custom range
  bool start_time_drag_active_ = false; // Whether start time drag handle is active
  bool end_time_drag_active_ = false;   // Whether end time drag handle is active

  void build_volume_profile();
  void render_volume_bars();
  void render_controls();
  void render_step_profile(const double* xs, const double* ys, const double* neg_ys, int count,
                           double height);
  void render_split_profile(const double* xs, const double* buy_vols, const double* sell_vols,
                           int count, double height);
  void calculate_value_area();

  // Subscribe to processor notifications
  void subscribe_to_updates();
};

}  // namespace BTQuant
