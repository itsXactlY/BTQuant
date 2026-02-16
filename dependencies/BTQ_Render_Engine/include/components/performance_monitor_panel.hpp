#pragma once

#include <vector>

#include "../performance_monitor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class PerformanceMonitorPanel : public PanelBase {
 public:
  explicit PerformanceMonitorPanel(const PanelConfig& config);

  void update(float dt) override;
  void render_content() override;

  // Configuration
  void set_update_interval(float interval);
  float get_update_interval() const;

 private:
  // Performance metrics
  std::vector<PerformanceMetric> current_metrics_;

  // Update timer
  float update_timer_ = 0.0f;
  float update_interval_ = 1.0f;  // Update every second by default
};

}  // namespace BTQuant