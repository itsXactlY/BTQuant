#include "../../include/components/panel_manager.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/components/metrics_panel.hpp"
#include "../../include/components/orderbook_panel.hpp"
#include "imgui.h"
#include <iostream>

namespace BTQuant {

PanelManager::PanelManager(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<OrderManager> order_manager,
    std::shared_ptr<PositionManager> position_manager,
    std::shared_ptr<RiskAssessment> risk_assessment)
    : bridge_(bridge), processor_(processor), order_manager_(order_manager),
      position_manager_(position_manager), risk_assessment_(risk_assessment) {
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
}

PanelManager::~PanelManager() { panels_.clear(); }

void PanelManager::initialize() {
  // Create default panels
  add_panel(PanelType::CHART, "BTC-USDT Chart", 0, 0, 2, 1);
  add_panel(PanelType::METRICS, "Market Debug", 2, 0, 1, 1);
  add_panel(PanelType::ORDERBOOK, "BTC-USDT Orderbook", 0, 1, 1, 1);
  add_panel(PanelType::TRADING_POSITIONS, "Positions", 1, 1, 1, 1);
  add_panel(PanelType::RISK_METRICS, "Risk Dashboard", 2, 1, 1, 1);
}

void PanelManager::update(float dt) {
  chart_manager_->update();

  for (auto &[id, panel] : panels_) {
    panel->update(dt);
  }
}

void PanelManager::render() {
  // Update dashboard size
  dashboard_size_ = ImGui::GetIO().DisplaySize;

  for (auto &[id, panel] : panels_) {
    if (panel->is_visible()) {
      panel->render();
    }
  }
}

uint32_t PanelManager::add_panel(PanelType type, const std::string &title,
                                 int grid_x, int grid_y, int width,
                                 int height) {
  uint32_t panel_id = next_panel_id_++;

  PanelConfig config =
      create_panel_config(type, title, grid_x, grid_y, width, height);

  std::unique_ptr<PanelBase> panel;
  switch (type) {
  case PanelType::CHART:
    panel = std::make_unique<ChartPanel>(config, bridge_, processor_,
                                         chart_manager_.get());
    break;
  case PanelType::METRICS:
    panel = std::make_unique<MetricsPanel>(config, position_manager_,
                                           risk_assessment_, processor_);
    break;
  case PanelType::HEATMAP:
    // TODO: Re-enable when HeatmapPanel is updated for newer ImPlot API
    return 0;
  case PanelType::ORDERBOOK:
    panel = std::make_unique<OrderbookPanel>(config, bridge_, processor_);
    break;
  case PanelType::SCATTER_PLOT:
  case PanelType::TIME_SERIES:
  case PanelType::TRADING_ORDERS:
  case PanelType::TRADING_POSITIONS:
  case PanelType::RISK_METRICS:
  case PanelType::ALERTS:
  case PanelType::HISTOGRAM:
    // TODO: Implement these panel types
    return 0;
  default:
    return 0;
  }

  if (panel) {
    panel->initialize();
    panels_[panel_id] = std::move(panel);
  }

  return panel_id;
}

void PanelManager::remove_panel(uint32_t panel_id) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    panels_.erase(it);
  }
}

void PanelManager::move_panel(uint32_t panel_id, int new_grid_x,
                              int new_grid_y) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    auto &config = it->second->get_config();
    config.grid_x = new_grid_x;
    config.grid_y = new_grid_y;
    config.position = calculate_panel_position(new_grid_x, new_grid_y);
  }
}

void PanelManager::resize_panel(uint32_t panel_id, int new_width,
                                int new_height) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    auto &config = it->second->get_config();
    config.grid_width = new_width;
    config.grid_height = new_height;
    config.size = calculate_panel_size(new_width, new_height);
  }
}

void PanelManager::set_grid_layout(int columns, int rows) {
  grid_layout_.columns = columns;
  grid_layout_.rows = rows;
}

void PanelManager::auto_arrange_panels() {
  int i = 0;
  for (auto &[id, panel] : panels_) {
    int x = i % grid_layout_.columns;
    int y = i / grid_layout_.columns;
    move_panel(id, x, y);
    i++;
  }
}

ImVec2 PanelManager::get_panel_position(uint32_t panel_id) const {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    return it->second->get_config().position;
  }
  return ImVec2(0, 0);
}

ImVec2 PanelManager::get_panel_size(uint32_t panel_id) const {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    return it->second->get_config().size;
  }
  return ImVec2(0, 0);
}

void PanelManager::set_panel_visible(uint32_t panel_id, bool visible) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    it->second->set_visible(visible);
  }
}

PanelConfig PanelManager::create_panel_config(PanelType type,
                                              const std::string &title,
                                              int grid_x, int grid_y, int width,
                                              int height) {
  PanelConfig config;
  config.type = type;
  config.title = title.empty() ? get_default_panel_title(type) : title;
  config.grid_x = grid_x;
  config.grid_y = grid_y;
  config.grid_width = width;
  config.grid_height = height;
  config.position = calculate_panel_position(grid_x, grid_y);
  config.size = calculate_panel_size(width, height);
  config.visible = true;
  config.resizable = true;
  config.movable = true;
  return config;
}

ImVec2 PanelManager::calculate_panel_position(int grid_x, int grid_y) const {
  float x = static_cast<float>(grid_x) * (dashboard_size_.x / 3.0f);
  float y = static_cast<float>(grid_y) * (dashboard_size_.y / 2.0f);
  return ImVec2(x, y);
}

ImVec2 PanelManager::calculate_panel_size(int width, int height) const {
  float w = static_cast<float>(width) * (dashboard_size_.x / 3.0f);
  float h = static_cast<float>(height) * (dashboard_size_.y / 2.0f);
  return ImVec2(w, h);
}

std::string PanelManager::get_default_panel_title(PanelType type) {
  switch (type) {
  case PanelType::CHART:
    return "Chart";
  case PanelType::METRICS:
    return "Metrics";
  case PanelType::HEATMAP:
    return "Heatmap";
  case PanelType::TRADING_ORDERS:
    return "Orders";
  case PanelType::TRADING_POSITIONS:
    return "Positions";
  case PanelType::RISK_METRICS:
    return "Risk";
  case PanelType::ALERTS:
    return "Alerts";
  case PanelType::ORDERBOOK:
    return "Orderbook";
  default:
    return "Panel";
  }
}

void PanelManager::save_layout(const std::string &filename) {
  // TODO: Implement layout saving to JSON
  (void)filename;
}

void PanelManager::load_layout(const std::string &filename) {
  // TODO: Implement layout loading from JSON
  (void)filename;
}

std::string PanelManager::serialize_layout() const {
  // TODO: Implement layout serialization
  return "{}";
}

void PanelManager::deserialize_layout(const std::string &layout_json) {
  // TODO: Implement layout deserialization
  (void)layout_json;
}

} // namespace BTQuant