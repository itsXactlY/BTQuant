#pragma once

#include "panel_base.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include <memory>

namespace BTQuant {

class TradingOrdersPanel : public PanelBase {
public:
    TradingOrdersPanel(const PanelConfig &config, 
                      std::shared_ptr<OrderManager> order_manager,
                      std::shared_ptr<PositionManager> position_manager);

    void initialize() override;
    void render() override;

private:
    std::shared_ptr<OrderManager> order_manager_;
    std::shared_ptr<PositionManager> position_manager_;
};

} // namespace BTQuant