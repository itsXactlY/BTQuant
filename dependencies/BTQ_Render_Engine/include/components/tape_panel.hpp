#pragma once

#include <array>
#include <memory>
#include <string>

#include "../data/core_types.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class TapePanel : public PanelBase {
 public:
  TapePanel(const PanelConfig& config,
            std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~TapePanel() override;

  void update(float dt) override;
  void render_content() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Trade data buffer (non-consuming peek from TradeRing)
  static constexpr size_t MAX_VISIBLE_TRADES = 500;
  std::array<TradeData, MAX_VISIBLE_TRADES> recent_trades_{};
  size_t trade_count_ = 0;

  // UI controls
  float size_filter_ = 0.0f;

  // Rendering helpers
  void render_tape_row(const TradeData& trade, int row_index, const TradeData* prev_trade) const;
  float compute_size_alpha(const TradeData& trade) const;
};

}  // namespace BTQuant
