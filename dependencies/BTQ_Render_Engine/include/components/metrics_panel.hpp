#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../market_data_processor.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "panel_base.hpp"

namespace BTQuant {

struct Metric {
  std::string name;
  std::string value;
  std::string unit;
  ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  bool is_percentage = false;
  float change = 0.0f;  // Percentage change
};

class MetricsPanel : public PanelBase {
 public:
  MetricsPanel(const PanelConfig& config, std::shared_ptr<PositionManager> position_manager,
               std::shared_ptr<RiskAssessment> risk_assessment,
               std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render_content() override;

 private:
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  std::vector<Metric> metrics_;
  float update_timer_ = 0.0f;
  const float UPDATE_INTERVAL = 1.0f;  // Update every second

  void update_metrics();
  void render_debug_info();
  void render_metric_grid();
  void render_metric_card(const Metric& metric, float width);
  ImVec4 get_metric_color(float value, bool is_positive_good = true);
};

}  // namespace BTQuant