#pragma once

#include <memory>

#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class TradingOrdersPanel : public PanelBase {
 public:
  TradingOrdersPanel(const PanelConfig& config, std::shared_ptr<OrderManager> order_manager,
                     std::shared_ptr<PositionManager> position_manager);

  void initialize() override;
  void render_content() override;

 private:
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  
  // Heatmap intensity configuration for resting limit orders
  float heatmap_intensity_ = 1.0f;  // Sensitivity of color mapping for resting limit orders (default 1.0)
};

}  // namespace BTQuant