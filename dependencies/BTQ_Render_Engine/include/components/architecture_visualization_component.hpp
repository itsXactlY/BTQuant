#pragma once

#include "../data_visualization_engine.hpp"
#include "../hotspine_data_bridge.hpp"
#include "../interaction_manager.hpp"
#include "../market_data_processor.hpp"
#include "../performance_monitor.hpp"
#include "../vulkan_base_types.hpp"
#include <chrono>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

namespace BTQuant {

// Architecture node for visualization
struct ArchitectureNode {
  std::string name;
  std::string type;
  ImVec2 position;
  ImVec2 size;
  ImVec4 color;
  bool active = true;
  double load = 0.0;
  double latency_ms = 0.0;

  ArchitectureNode(const std::string &n, const std::string &t, ImVec2 p,
                   ImVec2 s, ImVec4 c, bool a = true, double l = 0.0,
                   double lat = 0.0)
      : name(n), type(t), position(p), size(s), color(c), active(a), load(l),
        latency_ms(lat) {}
};

// Data flow connection
struct DataFlowConnection {
  std::string from_node;
  std::string to_node;
  ImVec2 from_pos;
  ImVec2 to_pos;
  ImVec4 color;
  double bandwidth_mbps = 0.0;
  uint32_t packet_count = 0;
  bool active = true;

  DataFlowConnection(const std::string &f, const std::string &t, ImVec2 fp,
                     ImVec2 tp, ImVec4 c, double b = 0.0, uint32_t p = 0,
                     bool a = true)
      : from_node(f), to_node(t), from_pos(fp), to_pos(tp), color(c),
        bandwidth_mbps(b), packet_count(p), active(a) {}
};

// Data pulse for animation
struct DataPulse {
  std::string from_node;
  std::string to_node;
  float progress = 0.0f; // 0.0 to 1.0
  float speed = 1.0f;
  ImVec4 color = {0.0f, 0.94f, 1.0f, 1.0f};
  bool active = true;
};

class ArchitectureVisualizationComponent : public UIComponent {
public:
  ArchitectureVisualizationComponent(
      std::shared_ptr<HotSpineDataBridge> hotspine_bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> market_processor,
      std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor =
          nullptr,
      RenderEngine::DataVisualizationEngine *visualization_engine = nullptr);

  virtual ~ArchitectureVisualizationComponent();

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override;
  void clear_data() override;

  // Get current spine health metrics
  SpineHealthMetrics getSpineHealthMetrics() const;

  // Get latency breakdown
  LatencyBreakdown getLatencyBreakdown() const;

  // Get system resource utilization
  std::vector<ComponentResourceUtilization> getResourceUtilization() const;

  // Reset visualization state
  void reset();

  // Enable/disable animations
  void setAnimationsEnabled(bool enabled);

  // Set visualization theme
  void setTheme(const std::string &theme);

private:
  // Initialize architecture nodes and connections
  void initializeArchitecture();

  // Render architecture diagram
  void renderArchitectureDiagram() const;

  // Render health metrics panel
  void renderHealthMetricsPanel() const;

  // Render latency breakdown panel
  void renderLatencyBreakdownPanel() const;

  // Render resource utilization panel
  void renderResourceUtilizationPanel() const;

  // Render system health score
  void renderHealthScore() const;

  // Calculate health score
  double calculateHealthScore() const;

  // Update node positions with animations
  void updateNodePositions();

  // Update node loads
  void updateNodeLoad();

  // Update health metrics from bridge
  void updateHealthMetrics();

  // Update latency breakdown from performance monitor
  void updateLatencyBreakdown();

  // Update resource utilization
  void updateResourceUtilization();

  // Update data pulses
  void updatePulses(float delta_time);

  // Render pulses along connections
  void renderPulses();

  // Render control bar
  void renderControlBar();

  // Smooth animation helper
  float smoothAnimation(float current, float target, float delta_time,
                        float smooth_factor = 0.1f);

  // Get node by name
  ArchitectureNode *getNodeByName(const std::string &name);
  const ArchitectureNode *getNodeByName(const std::string &name) const;

  // Get connection between nodes
  DataFlowConnection *getConnection(const std::string &from_node,
                                    const std::string &to_node);
  const DataFlowConnection *getConnection(const std::string &from_node,
                                          const std::string &to_node) const;

  // Generate color based on load/health
  ImVec4 getLoadColor(double load) const;
  ImVec4 getHealthColor(double health) const;

  // Performance monitoring
  std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor_;

  // System components
  std::shared_ptr<HotSpineDataBridge> hotspine_bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> market_processor_;
  RenderEngine::DataVisualizationEngine *visualization_engine_;

  // Architecture visualization data
  std::vector<ArchitectureNode> nodes_;
  std::vector<DataFlowConnection> connections_;
  std::vector<DataPulse> pulses_;

  // Health metrics
  SpineHealthMetrics health_metrics_;
  LatencyBreakdown latency_breakdown_;
  std::vector<ComponentResourceUtilization> resource_utilization_;

  // Visualization state
  bool animations_enabled_ = true;
  std::string current_theme_ = "quant";
  ImVec2 diagram_position_ = {0, 0};
  ImVec2 diagram_size_ = {800, 600};
  float zoom_level_ = 1.0f;
  bool show_detailed_metrics_ = true;
  bool show_node_labels_ = true;
  bool show_connection_bandwidth_ = true;

  // Animation state
  std::chrono::high_resolution_clock::time_point last_animation_time_;
  float animation_progress_ = 0.0f;
  bool animation_complete_ = false;
};

} // namespace BTQuant
