#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <imgui.h>
#include <memory>

namespace BTQuant {

/**
 * TapePanel - Time & Sales display
 *
 * Shows a scrolling list of recent trades with:
 * - Timestamp (HH:MM:SS.mmm)
 * - Price
 * - Size
 * - Side (colored: green=buy, red=sell)
 */
class TapePanel : public PanelBase {
public:
  TapePanel(const PanelConfig &config,
            std::shared_ptr<HotSpineDataBridge> bridge,
            std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

  void set_symbol(uint32_t symbol_id, const std::string &symbol_name);

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Configuration
  static constexpr size_t MAX_VISIBLE_TRADES = 50;
  bool auto_scroll_ = true;

  // Cached trades for rendering
  std::vector<RenderEngine::TradeData> cached_trades_;
  float update_timer_ = 0.0f;
  static constexpr float UPDATE_INTERVAL = 0.05f; // 50ms refresh

  void render_trade_table();
  void render_controls();
};

} // namespace BTQuant
