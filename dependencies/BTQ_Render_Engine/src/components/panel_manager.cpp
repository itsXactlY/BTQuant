#include "../../include/components/panel_manager.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/components/metrics_panel.hpp"
#include "../../include/components/orderbook_panel.hpp"
#include "../../include/components/status_bar_panel.hpp"
#include "../../include/components/watchlist_panel.hpp"
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
  // Set grid layout for billion-dollar terminal (3 columns, 4 rows)
  set_grid_layout(3, 4);

  // Create default panels for billion-dollar terminal layout
  add_panel(PanelType::STATUS_BAR, "Status Bar", 0, 0, 3, 1); // Full width status bar
  add_panel(PanelType::CHART, "BTC-USDT Chart", 0, 1, 2, 2);   // Large main chart
  add_panel(PanelType::ORDERBOOK, "BTC-USDT Orderbook", 2, 1, 1, 1);
  add_panel(PanelType::WATCHLIST, "Watchlist", 2, 2, 1, 1);
  add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 0, 3, 1, 1);
  add_panel(PanelType::DEPTH_CHART, "Depth Chart", 1, 3, 1, 1);
  add_panel(PanelType::TAPE, "Time & Sales", 2, 3, 1, 1);

  // Initialize orderbook with first active symbol
  auto active_symbols = bridge_->getActiveSymbols();
  if (!active_symbols.empty()) {
    uint32_t symbol_id = active_symbols[0];
    std::string symbol_name = bridge_->getSymbolName(symbol_id);
    if (!symbol_name.empty()) {
      set_active_symbol(symbol_id, symbol_name);

      // Add initial symbol to watchlist (find watchlist panel dynamically)
      for (auto &[id, panel] : panels_) {
        if (panel->get_config().type == PanelType::WATCHLIST) {
          auto watchlist_panel = dynamic_cast<WatchlistPanel*>(panel.get());
          if (watchlist_panel) {
            watchlist_panel->add_symbol(symbol_id, symbol_name,
                                       bridge_->getExchangeName(symbol_id));
          }
          break;
        }
      }
    }
  }
}

void PanelManager::update(float dt) {
  chart_manager_->update();

  for (auto &[id, panel] : panels_) {
    panel->update(dt);
  }
}

void PanelManager::render() {
  // Update dashboard size
  ImVec2 current_size = ImGui::GetIO().DisplaySize;
  if (current_size.x > 0 && current_size.y > 0) {
    dashboard_size_ = current_size;
  }

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
  case PanelType::STATUS_BAR:
    panel = std::make_unique<StatusBarPanel>(config, bridge_, processor_);
    break;
  case PanelType::WATCHLIST:
    panel = std::make_unique<WatchlistPanel>(config, bridge_, processor_);
    break;
  case PanelType::SCATTER_PLOT:
  case PanelType::TIME_SERIES:
  case PanelType::TRADING_ORDERS:
  case PanelType::TRADING_POSITIONS:
  case PanelType::RISK_METRICS:
  case PanelType::ALERTS:
  case PanelType::HISTOGRAM:
  case PanelType::SCREENER:
  case PanelType::TAPE:
  case PanelType::VOLUME_PROFILE:
  case PanelType::DEPTH_CHART:
  case PanelType::LOG_PANEL:
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

  // Special handling for status bar
  if (type == PanelType::STATUS_BAR) {
    config.position = ImVec2(0, 0);
    config.size = ImVec2(dashboard_size_.x, 30);
    config.resizable = false;
    config.movable = false;
  } else {
    config.position = calculate_panel_position(grid_x, grid_y);
    config.size = calculate_panel_size(width, height);
    config.resizable = true;
    config.movable = true;
  }

  config.visible = true;
  return config;
}

ImVec2 PanelManager::calculate_panel_position(int grid_x, int grid_y) const {
  float x = static_cast<float>(grid_x) * (dashboard_size_.x / grid_layout_.columns);
  float y = 30.0f + static_cast<float>(grid_y) * ((dashboard_size_.y - 30.0f) / grid_layout_.rows);
  return ImVec2(x, y);
}

ImVec2 PanelManager::calculate_panel_size(int width, int height) const {
  float w = static_cast<float>(width) * (dashboard_size_.x / grid_layout_.columns);
  float h = static_cast<float>(height) * ((dashboard_size_.y - 30.0f) / grid_layout_.rows);
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
  case PanelType::STATUS_BAR:
    return "Status";
  case PanelType::WATCHLIST:
    return "Watchlist";
  case PanelType::SCREENER:
    return "Screener";
  case PanelType::TAPE:
    return "Time & Sales";
  case PanelType::VOLUME_PROFILE:
    return "Volume Profile";
  case PanelType::DEPTH_CHART:
    return "Depth Chart";
  case PanelType::LOG_PANEL:
    return "Log";
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

void PanelManager::set_active_symbol(uint32_t symbol_id,
                                     const std::string &symbol_name) {
  // Propagate symbol to all orderbook panels
  for (auto &[id, panel] : panels_) {
    if (panel->get_config().type == PanelType::ORDERBOOK) {
      auto *orderbook = dynamic_cast<OrderbookPanel *>(panel.get());
      if (orderbook) {
        orderbook->set_symbol(symbol_id, symbol_name);
      }
    }
  }
}

} // namespace BTQuant