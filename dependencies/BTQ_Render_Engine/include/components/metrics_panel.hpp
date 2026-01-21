#pragma once

#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "panel_base.hpp"
#include <memory>
#include <string>
#include <vector>

namespace BTQuant {

struct Metric {
  std::string name;
  std::string value;
  std::string unit;
  ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  bool has_progress = false;
  float progress = 0.0f;
};

class MetricsPanel : public PanelBase {
public:
  MetricsPanel(const PanelConfig &config,
               std::shared_ptr<PositionManager> position_manager,
               std::shared_ptr<RiskAssessment> risk_assessment);

  void update(float dt) override;
  void render() override;

private:
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;

  std::vector<Metric> metrics_;
  float update_timer_ = 0.0f;
  const float UPDATE_INTERVAL = 1.0f; // Update every second

  void update_metrics();
  void render_metric_grid();
  void render_metric_card(const Metric &metric, float width, float height);
  ImVec4 get_metric_color(bool is_positive, bool unused = true);
};

} // namespace BTQuant