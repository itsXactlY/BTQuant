#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

struct ResourceUtilizationData {
  float cpu_usage = 0.0f;
  float gpu_usage = 0.0f;
  float memory_usage = 0.0f;
  float gpu_memory_usage = 0.0f;
  float cpu_temperature = 0.0f;
  float gpu_temperature = 0.0f;
  float network_throughput = 0.0f; // MB/s
  float disk_usage = 0.0f;

  double timestamp = 0.0;
};

struct ComponentResourceUsage {
  std::string name;
  float cpu_usage = 0.0f;
  float memory_usage = 0.0f;
  float gpu_usage = 0.0f;
  float gpu_memory_usage = 0.0f;

  ComponentResourceUsage(const std::string &n) : name(n) {}
};

class SystemResourceUtilizationComponent : public UIComponent {
public:
  explicit SystemResourceUtilizationComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
      std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor =
          nullptr);
  virtual ~SystemResourceUtilizationComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override;
  void clear_data() override;

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor_;

  std::vector<ResourceUtilizationData> historical_data_;
  std::vector<ComponentResourceUsage> component_usage_;
  size_t max_history_points_ = 200;

  bool show_cpu_chart_ = true;
  bool show_gpu_chart_ = true;
  bool show_memory_chart_ = true;
  bool show_temperature_chart_ = true;
  bool show_network_chart_ = true;

  bool auto_refresh_ = true;
  float refresh_interval_ = 1.0f;
  float refresh_timer_ = 0.0f;

  void collect_resource_data();
  void update_component_usage();
  void render_overview_panel();
  void render_detailed_charts();
  void render_component_breakdown();
  void render_heatmap_visualization();
  void render_resource_alerts();

  float calculate_system_health_score() const;
  std::string get_health_status(float score) const;
  ImVec4 get_health_color(float score) const;

  bool has_high_cpu_usage() const;
  bool has_high_gpu_usage() const;
  bool has_high_memory_usage() const;
  bool has_high_temperature() const;
};

} // namespace BTQuant
