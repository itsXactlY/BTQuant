#pragma once

#include <memory>

#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class RiskMetricsPanel : public PanelBase {
 public:
  RiskMetricsPanel(const PanelConfig& config, std::shared_ptr<RiskAssessment> risk_assessment,
                   std::shared_ptr<PositionManager> position_manager);

  void initialize() override;
  void render() override;

 private:
  std::shared_ptr<RiskAssessment> risk_assessment_;
  std::shared_ptr<PositionManager> position_manager_;
};

}  // namespace BTQuant