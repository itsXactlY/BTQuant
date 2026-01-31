#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <chrono>
#include <string>

namespace BTQuant {

class StatusBarPanel : public PanelBase {
public:
  StatusBarPanel(const PanelConfig &config,
                 std::shared_ptr<HotSpineDataBridge> bridge,
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  // Status data
  bool connection_status_ = false;
  std::string connection_text_ = "Disconnected";
  RenderEngine::ProcessorPerformanceMetrics performance_metrics_;
  std::chrono::system_clock::time_point last_update_;

  // Performance tracking
  float fps_ = 0.0f;
  float frame_time_ms_ = 0.0f;

  void update_connection_status();
  void update_performance_metrics();
  void render_connection_status();
  void render_performance_metrics();
  void render_time_display();
};

} // namespace BTQuant