#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <imgui.h>
#include <memory>
#include <vector>

namespace BTQuant {

/**
 * VolumeProfilePanel - Volume at Price display
 *
 * Shows horizontal bars representing volume traded at each price level:
 * - Bid volume (green) on one side
 * - Ask volume (red) on the other side
 * - Point of Control (POC) highlighted
 */
class VolumeProfilePanel : public PanelBase {
public:
  VolumeProfilePanel(
      const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

  void set_symbol(uint32_t symbol_id, const std::string &symbol_name);

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

  // Update timing
  float update_timer_ = 0.0f;
  static constexpr float UPDATE_INTERVAL = 0.1f; // 100ms refresh

  // Configuration
  static constexpr size_t NUM_PRICE_LEVELS = 20;
  double price_bucket_size_ = 10.0; // Price range per level

  void build_volume_profile();
  void render_volume_bars();
  void render_controls();
};

} // namespace BTQuant
