#pragma once

#include <memory>
#include <string>

#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class TimeAndSalesPanel : public PanelBase {
 public:
  TimeAndSalesPanel(const PanelConfig& config,
                    std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~TimeAndSalesPanel() override;

  void render_content() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";
};

}  // namespace BTQuant