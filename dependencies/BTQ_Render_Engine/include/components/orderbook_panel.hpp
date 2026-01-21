#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <imgui.h>
#include <memory>

namespace BTQuant {

// Real-time orderbook ladder display
class OrderbookPanel : public PanelBase {
public:
  OrderbookPanel(const PanelConfig &config,
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

  static constexpr int MAX_LEVELS = 20; // Number of bid/ask levels to show

  void render_orderbook_ladder(const RenderEngine::OrderbookData &orderbook);
  void render_market_depth_chart(const RenderEngine::OrderbookData &orderbook);
};

} // namespace BTQuant
