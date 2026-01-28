#include "../../include/components/risk_metrics_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <iostream>

namespace BTQuant {

RiskMetricsPanel::RiskMetricsPanel(const PanelConfig &config, 
                    std::shared_ptr<RiskAssessment> risk_assessment,
                    std::shared_ptr<PositionManager> position_manager)
    : PanelBase(config), risk_assessment_(risk_assessment), 
      position_manager_(position_manager) {}

void RiskMetricsPanel::initialize() {
    PanelBase::initialize();
}

void RiskMetricsPanel::render() {
    begin_panel_window();
    
    if (risk_assessment_) {
        auto metrics = risk_assessment_->get_risk_metrics();
        
        ImGui::Text("Portfolio Beta: %.2f", metrics.portfolio_beta);
        ImGui::Text("Unrealized P/L: $%.2f", metrics.unrealized_pnl);
        ImGui::Text("Max Drawdown: $%.2f", metrics.max_drawdown);
        ImGui::Text("Sharpe Ratio: %.2f", metrics.sharpe_ratio);
        ImGui::Separator();
        
        ImGui::Text("Daily P/L: $%.2f", metrics.daily_pnl);
        ImGui::Text("Current Leverage: %.2fx", metrics.current_leverage);
        ImGui::Text("Overall Risk Score: %.2f", metrics.overall_risk_score);
        ImGui::Separator();
        
        ImGui::Text("VaR (Current): $%.2f", metrics.current_var);
        ImGui::Text("Concentration Risk: %.2f", metrics.concentration_risk);
        ImGui::Text("Leverage Risk: %.2f", metrics.leverage_risk);
        ImGui::Text("Volatility Risk: %.2f", metrics.volatility_risk);
    }
    
    end_panel_window();
}

} // namespace BTQuant