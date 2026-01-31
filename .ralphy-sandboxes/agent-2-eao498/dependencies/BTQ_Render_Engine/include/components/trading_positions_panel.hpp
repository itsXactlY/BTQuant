#pragma once

#include "panel_base.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include <memory>

namespace BTQuant {

class TradingPositionsPanel : public PanelBase {
public:
    TradingPositionsPanel(const PanelConfig &config, 
                         std::shared_ptr<PositionManager> position_manager,
                         std::shared_ptr<RiskAssessment> risk_assessment);

    void initialize() override;
    void render() override;

private:
    std::shared_ptr<PositionManager> position_manager_;
    std::shared_ptr<RiskAssessment> risk_assessment_;
};

} // namespace BTQuant