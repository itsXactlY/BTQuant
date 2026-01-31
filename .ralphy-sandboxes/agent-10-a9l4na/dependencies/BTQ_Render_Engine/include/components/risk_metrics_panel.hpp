#pragma once

#include "panel_base.hpp"
#include "../trading/risk_assessment.hpp"
#include "../trading/position_manager.hpp"
#include <memory>

namespace BTQuant {

class RiskMetricsPanel : public PanelBase {
public:
    RiskMetricsPanel(const PanelConfig &config, 
                    std::shared_ptr<RiskAssessment> risk_assessment,
                    std::shared_ptr<PositionManager> position_manager);

    void initialize() override;
    void render() override;

private:
    std::shared_ptr<RiskAssessment> risk_assessment_;
    std::shared_ptr<PositionManager> position_manager_;
};

} // namespace BTQuant