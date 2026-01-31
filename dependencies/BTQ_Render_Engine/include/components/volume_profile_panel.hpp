#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "theme_manager.hpp"
#include <imgui.h>
#include <memory>
#include <vector>

namespace BTQuant {

// Enum for profile mode
enum class ProfileMode {
    Step,
    Right,
    Left,
    Custom
};

// Struct for profile settings
struct ProfileSettings {
    double vaPercent = 70.0;        // Value Area percentage
    int tickStep = 1;               // Tick step size
    bool showPOC = true;            // Show Point of Control
    bool showValueArea = true;      // Show Value Area
    int colorScheme = 0;            // Color scheme index
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
  VolumeProfilePanel(
      const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  ~VolumeProfilePanel() override;

  void render() override;
  void set_symbol(uint32_t symbol_id, const std::string &symbol_name);

  // Method to render mini histogram overlays on candlestick charts
  void render_mini_histograms_on_candles(ImDrawList* draw_list,
                                       const std::vector<RenderEngine::OHLCVCandle>& candles,
                                       const std::vector<double>& x_coords,
                                       const std::vector<double>& y_coords_high,
                                       const std::vector<double>& y_coords_low);

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
  double poc_price_ = 0.0;  // Point of Control (highest volume price)
  double max_volume_ = 0.0; // For scaling bars

  // Configuration
  static constexpr size_t NUM_PRICE_LEVELS = 20;
  double price_bucket_size_ = 10.0; // Price range per level

  // Profile settings
  ProfileMode profile_mode_ = ProfileMode::Step;
  int va_percent_ = 70; // Value Area percentage

  void build_volume_profile();
  void render_volume_bars();
  void render_controls();
  void render_step_profile(const double* xs, const double* ys,
                         const double* neg_ys, int count, double height);

  // Subscribe to processor notifications
  void subscribe_to_updates();
};

} // namespace BTQuant
