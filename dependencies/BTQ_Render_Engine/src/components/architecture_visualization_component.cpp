#include "components/architecture_visualization_component.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <imgui.h>
#include <implot.h>
#include <iostream>

namespace BTQuant {

ArchitectureVisualizationComponent::ArchitectureVisualizationComponent(
    std::shared_ptr<HotSpineDataBridge> hotspine_bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> market_processor,
    std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor,
    RenderEngine::DataVisualizationEngine *visualization_engine)
    : UIComponent(glm::vec2(0.0f, 0.0f), glm::vec2(800.0f, 600.0f)),
      performance_monitor_(performance_monitor),
      hotspine_bridge_(hotspine_bridge), market_processor_(market_processor),
      visualization_engine_(visualization_engine) {
  initializeArchitecture();
  last_animation_time_ = std::chrono::high_resolution_clock::now();
}

ArchitectureVisualizationComponent::~ArchitectureVisualizationComponent() {}

void ArchitectureVisualizationComponent::initialize_vulkan_resources(
    VulkanCore *core) {}

void ArchitectureVisualizationComponent::clear_data() {
  nodes_.clear();
  connections_.clear();
  pulses_.clear();
}

void ArchitectureVisualizationComponent::initializeArchitecture() {
  // Create architecture nodes
  nodes_.clear();
  nodes_.reserve(15);

  // Base layers
  nodes_.emplace_back("HotSpine", "Core Layer", ImVec2(150, 300),
                      ImVec2(200, 80), ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
  nodes_.emplace_back("DataBridge", "Infrastructure", ImVec2(350, 300),
                      ImVec2(180, 60), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("DataProcessor", "Processing Layer", ImVec2(600, 300),
                      ImVec2(180, 60), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("VisualizationEngine", "Rendering Layer",
                      ImVec2(850, 300), ImVec2(180, 60),
                      ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("VulkanCore", "Hardware Layer", ImVec2(1100, 300),
                      ImVec2(200, 80), ImVec4(0.2f, 0.3f, 0.6f, 1.0f));

  // Charting sub-system
  nodes_.emplace_back("ChartManager", "UI Controller", ImVec2(600, 150),
                      ImVec2(150, 50), ImVec4(0.4f, 0.5f, 0.8f, 1.0f));
  nodes_.emplace_back("Chart1", "Visual Component", ImVec2(550, 50),
                      ImVec2(120, 40), ImVec4(0.5f, 0.6f, 0.9f, 1.0f));
  nodes_.emplace_back("Chart2", "Visual Component", ImVec2(790, 50),
                      ImVec2(120, 40), ImVec4(0.5f, 0.6f, 0.9f, 1.0f));
  nodes_.emplace_back("Chart3", "Visual Component", ImVec2(940, 50),
                      ImVec2(120, 40), ImVec4(0.5f, 0.6f, 0.9f, 1.0f));

  // Analytics sub-system
  nodes_.emplace_back("TechnicalIndicators", "Analytics", ImVec2(600, 450),
                      ImVec2(150, 50), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("VolumeProfile", "Analytics", ImVec2(750, 450),
                      ImVec2(120, 40), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("MarketDepth", "Analytics", ImVec2(900, 450),
                      ImVec2(120, 40), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));
  nodes_.emplace_back("PatternRecognizer", "Analytics", ImVec2(1050, 450),
                      ImVec2(120, 40), ImVec4(0.3f, 0.4f, 0.7f, 1.0f));

  // Performance Monitoring
  nodes_.emplace_back("PerformanceMonitor", "monitoring", ImVec2(850, 550),
                      ImVec2(180, 60), ImVec4(0.4f, 0.5f, 0.8f, 1.0f));
  nodes_.emplace_back("SystemMetrics", "monitoring", ImVec2(1100, 550),
                      ImVec2(120, 40), ImVec4(0.4f, 0.5f, 0.8f, 1.0f));
  nodes_.emplace_back("LatencyTracking", "monitoring", ImVec2(1250, 550),
                      ImVec2(120, 40), ImVec4(0.4f, 0.5f, 0.8f, 1.0f));
  nodes_.emplace_back("HealthMonitoring", "monitoring", ImVec2(1400, 550),
                      ImVec2(120, 40), ImVec4(0.4f, 0.5f, 0.8f, 1.0f));

  // Create data flow connections
  connections_.clear();
  connections_.reserve(25);

  connections_.emplace_back("MarketData", "Exchange", ImVec2(175.0f, 160.0f),
                            ImVec2(175.0f, 200.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("Exchange", "HotSpine", ImVec2(175.0f, 260.0f),
                            ImVec2(175.0f, 300.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("ExternalExchange", "HotSpine",
                            ImVec2(50.0f, 330.0f), ImVec2(150.0f, 330.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("HotSpine", "DataBridge", ImVec2(250.0f, 330.0f),
                            ImVec2(350.0f, 330.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataBridge", "DataProcessor",
                            ImVec2(530.0f, 330.0f), ImVec2(600.0f, 330.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataProcessor", "VisualizationEngine",
                            ImVec2(780.0f, 330.0f), ImVec2(850.0f, 330.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("VisualizationEngine", "VulkanCore",
                            ImVec2(1030.0f, 330.0f), ImVec2(1100.0f, 330.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);

  connections_.emplace_back("DataProcessor", "ChartManager",
                            ImVec2(690.0f, 300.0f), ImVec2(690.0f, 150.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("ChartManager", "Chart1", ImVec2(690.0f, 50.0f),
                            ImVec2(690.0f, 150.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("ChartManager", "Chart2", ImVec2(790.0f, 150.0f),
                            ImVec2(790.0f, 50.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("ChartManager", "Chart3", ImVec2(940.0f, 150.0f),
                            ImVec2(940.0f, 50.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);

  connections_.emplace_back("DataProcessor", "TechnicalIndicators",
                            ImVec2(690.0f, 360.0f), ImVec2(690.0f, 450.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataProcessor", "VolumeProfile",
                            ImVec2(740.0f, 360.0f), ImVec2(740.0f, 450.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataProcessor", "MarketDepth",
                            ImVec2(890.0f, 360.0f), ImVec2(890.0f, 450.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataProcessor", "PatternRecognizer",
                            ImVec2(1040.0f, 360.0f), ImVec2(1040.0f, 450.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);

  connections_.emplace_back("VulkanCore", "PerformanceMonitor",
                            ImVec2(1190.0f, 360.0f), ImVec2(1190.0f, 550.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataProcessor", "PerformanceMonitor",
                            ImVec2(690.0f, 360.0f), ImVec2(850.0f, 580.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("DataBridge", "PerformanceMonitor",
                            ImVec2(440.0f, 360.0f), ImVec2(850.0f, 580.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("PerformanceMonitor", "SystemMetrics",
                            ImVec2(940.0f, 610.0f), ImVec2(1100.0f, 610.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("PerformanceMonitor", "LatencyTracking",
                            ImVec2(940.0f, 610.0f), ImVec2(1250.0f, 610.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
  connections_.emplace_back("PerformanceMonitor", "HealthMonitoring",
                            ImVec2(940.0f, 610.0f), ImVec2(1400.0f, 610.0f),
                            ImVec4(0.3f, 0.6f, 1.0f, 0.8f), 0.0, 0, true);
}

void ArchitectureVisualizationComponent::render_gui() {
  ImGui::Begin("Architecture Visualization", nullptr,
               ImGuiWindowFlags_NoScrollbar |
                   ImGuiWindowFlags_NoScrollWithMouse);

  // Draw architecture diagram
  renderArchitectureDiagram();

  // Draw pulses on top of connections
  renderPulses();

  // Draw overlays
  if (show_detailed_metrics_) {
    renderHealthMetricsPanel();
    renderLatencyBreakdownPanel();
    renderResourceUtilizationPanel();
    renderHealthScore();
  }

  renderControlBar();

  ImGui::End();
}

void ArchitectureVisualizationComponent::renderArchitectureDiagram() const {
  ImGui::Text("Shared Memory Spine Architecture");
  ImGui::Separator();

  // Main canvas area
  ImGui::BeginChild("ArchitectureCanvas", ImVec2(0, 0), true,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse |
                        ImGuiWindowFlags_NoMove);

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

  // Render nodes
  for (const auto &node : nodes_) {
    if (node.active) {
      ImVec2 node_screen_pos = {
          canvas_pos.x + diagram_position_.x + node.position.x * zoom_level_,
          canvas_pos.y + diagram_position_.y + node.position.y * zoom_level_};
      ImVec2 node_screen_size = {node.size.x * zoom_level_,
                                 node.size.y * zoom_level_};

      // Draw node box
      draw_list->AddRectFilled(node_screen_pos,
                               {node_screen_pos.x + node_screen_size.x,
                                node_screen_pos.y + node_screen_size.y},
                               ImGui::ColorConvertFloat4ToU32(node.color),
                               5.0f * zoom_level_);

      // Glow effect for high load
      if (node.load > 70.0) {
        float pulse = (float)(sin(ImGui::GetTime() * 5.0) * 0.5 + 0.5);
        draw_list->AddRect(node_screen_pos,
                           {node_screen_pos.x + node_screen_size.x,
                            node_screen_pos.y + node_screen_size.y},
                           IM_COL32(255, 0, 0, (int)(pulse * 255)),
                           5.0f * zoom_level_, 0, 2.0f * zoom_level_);
      }

      // Node label
      if (show_node_labels_) {
        ImVec2 text_size = ImGui::CalcTextSize(node.name.c_str());
        draw_list->AddText(
            {node_screen_pos.x + (node_screen_size.x - text_size.x) / 2.0f,
             node_screen_pos.y + (node_screen_size.y - text_size.y) / 2.0f},
            IM_COL32(255, 255, 255, 255), node.name.c_str());

        // Type label
        text_size = ImGui::CalcTextSize(node.type.c_str());
        draw_list->AddText(
            {node_screen_pos.x + (node_screen_size.x - text_size.x) / 2.0f,
             node_screen_pos.y + node_screen_size.y + 5.0f},
            IM_COL32(180, 180, 180, 255), node.type.c_str());
      }
    }
  }

  // Render connections
  for (const auto &connection : connections_) {
    if (connection.active) {
      ImVec2 from_screen_pos = {canvas_pos.x + diagram_position_.x +
                                    connection.from_pos.x * zoom_level_,
                                canvas_pos.y + diagram_position_.y +
                                    connection.from_pos.y * zoom_level_};
      ImVec2 to_screen_pos = {canvas_pos.x + diagram_position_.x +
                                  connection.to_pos.x * zoom_level_,
                              canvas_pos.y + diagram_position_.y +
                                  connection.to_pos.y * zoom_level_};

      draw_list->AddLine(from_screen_pos, to_screen_pos,
                         ImGui::ColorConvertFloat4ToU32(connection.color),
                         2.0f * zoom_level_);

      // Bandwidth indicator
      if (show_connection_bandwidth_ && connection.bandwidth_mbps > 0) {
        char bw_text[32];
        snprintf(bw_text, sizeof(bw_text), "%.1f MB/s",
                 connection.bandwidth_mbps);
        ImVec2 text_pos = {(from_screen_pos.x + to_screen_pos.x) / 2.0f,
                           (from_screen_pos.y + to_screen_pos.y) / 2.0f -
                               15.0f};
        draw_list->AddText(text_pos, IM_COL32(0, 255, 255, 200), bw_text);
      }
    }
  }

  ImGui::EndChild();
}

void ArchitectureVisualizationComponent::renderHealthMetricsPanel() const {
  ImGui::BeginChild("HealthMetrics", {300, 250}, true);
  ImGui::Text("Spine Health Metrics");
  ImGui::Separator();

  ImGui::Text("Trade Buffer: %.1f%%", health_metrics_.trade_buffer_utilization);
  ImGui::ProgressBar(health_metrics_.trade_buffer_utilization / 100.0f,
                     ImVec2(-1, 0));

  ImGui::Text("Orderbook Buffer: %.1f%%",
              health_metrics_.orderbook_buffer_utilization);
  ImGui::ProgressBar(health_metrics_.orderbook_buffer_utilization / 100.0f,
                     ImVec2(-1, 0));

  ImGui::Spacing();
  ImGui::Text("Lost Trades: %lu", health_metrics_.lost_trades);
  ImGui::Text("Lost Orderbooks: %lu", health_metrics_.lost_orderbooks);

  ImGui::Spacing();
  ImGui::Text("Connection Status:");
  ImGui::TextColored(health_metrics_.hotspine_connected ? ImVec4(0, 1, 0, 1)
                                                        : ImVec4(1, 0, 0, 1),
                     "HotSpine: %s",
                     health_metrics_.hotspine_connected ? "CONNECTED"
                                                        : "OFFLINE");
  ImGui::TextColored(health_metrics_.data_bridge_active ? ImVec4(0, 1, 0, 1)
                                                        : ImVec4(1, 0, 0, 1),
                     "DataBridge: %s",
                     health_metrics_.data_bridge_active ? "ACTIVE"
                                                        : "INACTIVE");

  ImGui::EndChild();
}

void ArchitectureVisualizationComponent::renderLatencyBreakdownPanel() const {
  ImGui::BeginChild("LatencyBreakdown", {300, 250}, true);
  ImGui::Text("Latency Breakdown (μs)");
  ImGui::Separator();

  if (ImPlot::BeginPlot("##LatencyPlot", {-1, -1},
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMenus)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels,
                      ImPlotAxisFlags_AutoFit);

    double x_vals[] = {0, 1, 2, 3};
    double y_vals[] = {latency_breakdown_.data_processing_us,
                       latency_breakdown_.gpu_transfer_us,
                       latency_breakdown_.display_update_us,
                       latency_breakdown_.network_latency_ms * 1000.0};
    const char *labels[] = {"Proc", "GPU", "Disp", "Net"};

    ImPlot::PlotBars("##LatencyBars", x_vals, y_vals, 4, 0.6);
    ImPlot::SetupAxisTicks(ImAxis_X1, x_vals, 4, labels);

    ImPlot::EndPlot();
  }
  ImGui::EndChild();
}

void ArchitectureVisualizationComponent::renderResourceUtilizationPanel()
    const {
  ImGui::BeginChild("ResourceUtilization", {300, 250}, true);
  ImGui::Text("Resource Utilization");
  ImGui::Separator();

  for (const auto &comp : resource_utilization_) {
    ImGui::Text("%s:", comp.component_name.c_str());
    ImGui::Text("  CPU: %.1f%% | GPU: %.1f%%", comp.cpu_usage * 100.0,
                comp.gpu_usage * 100.0);
    ImGui::Text("  Mem: %.1f MB | Temp: %.1f°C", comp.memory_usage_mb,
                comp.temperature_celsius);
    ImGui::Separator();
  }
  ImGui::EndChild();
}

void ArchitectureVisualizationComponent::renderHealthScore() const {
  ImGui::Text("System Health Score");
  ImGui::Separator();

  ImVec4 score_color = getHealthColor(health_metrics_.health_score);
  ImGui::PushStyleColor(ImGuiCol_Text, score_color);
  ImGui::SetWindowFontScale(2.0f);
  ImGui::Text("%.1f", health_metrics_.health_score);
  ImGui::SetWindowFontScale(1.0f);
  ImGui::PopStyleColor();

  ImGui::ProgressBar(health_metrics_.health_score / 100.0f, ImVec2(-1, 20));
}

void ArchitectureVisualizationComponent::renderControlBar() {
  ImGui::SetCursorPos({10, 10});
  if (ImGui::Button("Reset View")) {
    diagram_position_ = {0, 0};
    zoom_level_ = 1.0f;
  }
  ImGui::SameLine();
  ImGui::Checkbox("Animations", &animations_enabled_);
  ImGui::SameLine();
  ImGui::Checkbox("Details", &show_detailed_metrics_);
}

void ArchitectureVisualizationComponent::update(float delta_time) {
  // Update health metrics from bridge
  updateHealthMetrics();

  // Update latency from performance monitor
  updateLatencyBreakdown();

  // Update resource utilization
  updateResourceUtilization();

  // Update node positions and loads
  updateNodePositions();
  updateNodeLoad();

  // Update animations
  if (animations_enabled_) {
    updatePulses(delta_time);
  }

  // Calculate overall health
  health_metrics_.health_score = calculateHealthScore();
}

void ArchitectureVisualizationComponent::updateHealthMetrics() {
  if (hotspine_bridge_) {
    auto metrics = hotspine_bridge_->get_spine_health_metrics();
    health_metrics_.trade_buffer_utilization = metrics.trade_buffer_utilization;
    health_metrics_.orderbook_buffer_utilization =
        metrics.orderbook_buffer_utilization;
    health_metrics_.lost_trades = metrics.lost_trades;
    health_metrics_.lost_orderbooks = metrics.lost_orderbooks;
    health_metrics_.hotspine_connected = metrics.hotspine_connected;
    health_metrics_.data_bridge_active = true;
  }
}

void ArchitectureVisualizationComponent::updateLatencyBreakdown() {
  if (performance_monitor_) {
    latency_breakdown_ = performance_monitor_->getLatencyBreakdown();
  }
}

void ArchitectureVisualizationComponent::updateResourceUtilization() {
  if (performance_monitor_) {
    resource_utilization_ =
        performance_monitor_->getComponentResourceUtilization();
  }
}

void ArchitectureVisualizationComponent::updateNodePositions() {
  // Smoothly move nodes if needed (placeholder for dynamic graph layout)
}

void ArchitectureVisualizationComponent::updateNodeLoad() {
  // Map performance metrics to node load
  for (auto &node : nodes_) {
    if (node.name == "HotSpine") {
      node.load = health_metrics_.trade_buffer_utilization;
    } else if (node.name == "DataProcessor") {
      node.load = (latency_breakdown_.data_processing_us / 1000.0) * 100.0;
    } else if (node.name == "VulkanCore") {
      if (!resource_utilization_.empty())
        node.load = resource_utilization_[0].gpu_usage * 100.0;
    }
    // Clamp load
    node.load = std::min(100.0, node.load);
  }
}

double ArchitectureVisualizationComponent::calculateHealthScore() const {
  double score = 100.0;

  // Penalty for buffer utilization
  score -= (health_metrics_.trade_buffer_utilization > 80) ? 20 : 0;
  score -= (health_metrics_.orderbook_buffer_utilization > 80) ? 20 : 0;

  // Penalty for lost data
  score -= (health_metrics_.lost_trades > 0) ? 10 : 0;
  score -= (health_metrics_.lost_orderbooks > 0) ? 10 : 0;

  // Critical penalty for connectivity
  if (!health_metrics_.hotspine_connected)
    score -= 50;

  return std::max(0.0, score);
}

void ArchitectureVisualizationComponent::setAnimationsEnabled(bool enabled) {
  animations_enabled_ = enabled;
}

void ArchitectureVisualizationComponent::setTheme(const std::string &theme) {
  current_theme_ = theme;
}

ArchitectureNode *
ArchitectureVisualizationComponent::getNodeByName(const std::string &name) {
  auto it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&name](const ArchitectureNode &node) { return node.name == name; });
  return it != nodes_.end() ? &(*it) : nullptr;
}

const ArchitectureNode *ArchitectureVisualizationComponent::getNodeByName(
    const std::string &name) const {
  auto it = std::find_if(
      nodes_.begin(), nodes_.end(),
      [&name](const ArchitectureNode &node) { return node.name == name; });
  return it != nodes_.end() ? &(*it) : nullptr;
}

DataFlowConnection *
ArchitectureVisualizationComponent::getConnection(const std::string &from_node,
                                                  const std::string &to_node) {
  auto it = std::find_if(
      connections_.begin(), connections_.end(),
      [&from_node, &to_node](const DataFlowConnection &conn) {
        return conn.from_node == from_node && conn.to_node == to_node;
      });
  return it != connections_.end() ? &(*it) : nullptr;
}

const DataFlowConnection *ArchitectureVisualizationComponent::getConnection(
    const std::string &from_node, const std::string &to_node) const {
  auto it = std::find_if(
      connections_.begin(), connections_.end(),
      [&from_node, &to_node](const DataFlowConnection &conn) {
        return conn.from_node == from_node && conn.to_node == to_node;
      });
  return it != connections_.end() ? &(*it) : nullptr;
}

ImVec4 ArchitectureVisualizationComponent::getLoadColor(double load) const {
  if (load < 30)
    return ImVec4(0.0f, 0.8f, 0.0f, 1.0f); // Green (low load)
  if (load < 60)
    return ImVec4(0.8f, 0.8f, 0.0f, 1.0f); // Yellow (medium load)
  if (load < 80)
    return ImVec4(0.9f, 0.4f, 0.0f, 1.0f); // Orange (high load)
  return ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red (very high load)
}

ImVec4 ArchitectureVisualizationComponent::getHealthColor(double health) const {
  if (health > 90)
    return ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
  if (health > 60)
    return ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
  return ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
}

float ArchitectureVisualizationComponent::smoothAnimation(float current,
                                                          float target,
                                                          float delta_time,
                                                          float smooth_factor) {
  return current + (target - current) * smooth_factor * delta_time;
}

void ArchitectureVisualizationComponent::updatePulses(float delta_time) {
  // Update progress of active pulses
  for (auto it = pulses_.begin(); it != pulses_.end();) {
    it->progress += it->speed * delta_time;
    if (it->progress >= 1.0f) {
      it = pulses_.erase(it);
    } else {
      ++it;
    }
  }

  // Randomly generate new pulses based on connection activity/bandwidth
  static float last_spawn_time = 0;
  last_spawn_time += delta_time;
  if (last_spawn_time > 0.2f) { // Spawn every 200ms
    last_spawn_time = 0;

    if (!connections_.empty()) {
      int conn_idx = rand() % connections_.size();
      const auto &conn = connections_[conn_idx];

      DataPulse p;
      p.from_node = conn.from_node;
      p.to_node = conn.to_node;
      p.progress = 0.0f;
      p.speed = 1.0f + (static_cast<float>(rand()) / RAND_MAX) * 2.0f;
      p.color = conn.color;
      p.active = true;
      pulses_.push_back(p);
    }
  }
}

void ArchitectureVisualizationComponent::renderPulses() {
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

  for (const auto &pulse : pulses_) {
    const auto *from = getNodeByName(pulse.from_node);
    const auto *to = getNodeByName(pulse.to_node);

    if (from && to) {
      ImVec2 start = {
          canvas_pos.x + diagram_position_.x + from->position.x * zoom_level_ +
              from->size.x * zoom_level_ / 2.0f,
          canvas_pos.y + diagram_position_.y + from->position.y * zoom_level_ +
              from->size.y * zoom_level_ / 2.0f};
      ImVec2 end = {
          canvas_pos.x + diagram_position_.x + to->position.x * zoom_level_ +
              to->size.x * zoom_level_ / 2.0f,
          canvas_pos.y + diagram_position_.y + to->position.y * zoom_level_ +
              to->size.y * zoom_level_ / 2.0f};

      ImVec2 pos = {start.x + (end.x - start.x) * pulse.progress,
                    start.y + (end.y - start.y) * pulse.progress};

      draw_list->AddCircleFilled(pos, 4.0f * zoom_level_,
                                 ImGui::ColorConvertFloat4ToU32(pulse.color));

      // Outer glow
      draw_list->AddCircle(pos, 6.0f * zoom_level_,
                           IM_COL32(pulse.color.x * 255, pulse.color.y * 255,
                                    pulse.color.z * 255, 100),
                           12, 1.0f * zoom_level_);
    }
  }
}

SpineHealthMetrics
ArchitectureVisualizationComponent::getSpineHealthMetrics() const {
  return health_metrics_;
}

LatencyBreakdown
ArchitectureVisualizationComponent::getLatencyBreakdown() const {
  return latency_breakdown_;
}

std::vector<ComponentResourceUtilization>
ArchitectureVisualizationComponent::getResourceUtilization() const {
  return resource_utilization_;
}

} // namespace BTQuant
