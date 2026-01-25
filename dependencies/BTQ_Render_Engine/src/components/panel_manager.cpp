#include "../../include/components/panel_manager.hpp"
#include "../../include/components/alerts_panel.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/components/depth_chart_panel.hpp"
#include "../../include/components/dom_surface_panel.hpp"
#include "../../include/components/footprint_panel.hpp"
#include "../../include/components/metrics_panel.hpp"
#include "../../include/components/orderbook_panel.hpp"
#include "../../include/components/status_bar_panel.hpp"
#include "../../include/components/tape_panel.hpp"
#include "../../include/components/tpo_panel.hpp"
#include "../../include/components/volume_profile_panel.hpp"
#include "../../include/components/watchlist_panel.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace BTQuant {

PanelManager::PanelManager(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<OrderManager> order_manager,
    std::shared_ptr<PositionManager> position_manager,
    std::shared_ptr<RiskAssessment> risk_assessment,
    RenderEngine::MarketMicrostructureRenderer *micro_renderer)
    : bridge_(bridge), processor_(processor), order_manager_(order_manager),
      position_manager_(position_manager), risk_assessment_(risk_assessment),
      micro_renderer_(micro_renderer) {
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
}

PanelManager::~PanelManager() { panels_.clear(); }

void PanelManager::initialize() {
  // Set grid layout (3 columns, 5 rows to fit 2x2 chart properly)
  set_grid_layout(3, 5);

  // Create default panels
  // Row 0: Status Bar
  // Row 0: Status Bar and Alerts
  add_panel(PanelType::STATUS_BAR, "Status Bar", 0, 0, 2, 1);
  add_panel(PanelType::ALERTS, "Alerts", 2, 0, 1, 1);

  // Row 1-2: Main Chart (2x2) and Depth Chart (1x2)
  add_panel(PanelType::CHART, "BTC-USDT Chart", 0, 1, 2, 2);
  add_panel(PanelType::DEPTH_CHART, "Depth Chart", 2, 1, 1, 2);

  // Row 3: Orderbook Ladder (2x1) and Tape (1x1)
  add_panel(PanelType::ORDERBOOK, "BTC-USDT Orderbook", 0, 3, 2, 1);
  add_panel(PanelType::TAPE, "Time & Sales", 2, 3, 1, 1);

  // Row 4: Volume Profile (2x1) and Watchlist (1x1)
  add_panel(PanelType::HEATMAP, "DOM Surface", 0, 4, 2, 1);
  add_panel(PanelType::WATCHLIST, "Watchlist", 2, 4, 1, 1);

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
          auto watchlist_panel = dynamic_cast<WatchlistPanel *>(panel.get());
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
    panel = std::make_unique<DomSurfacePanel>(processor_);
    break;
  case PanelType::ORDERBOOK:
    panel = std::make_unique<OrderbookPanel>(config, bridge_, processor_);
    break;
  case PanelType::STATUS_BAR:
    panel = std::make_unique<StatusBarPanel>(config, bridge_, processor_);
    break;
  case PanelType::WATCHLIST:
    panel = std::make_unique<WatchlistPanel>(config, bridge_, processor_);
    break;
  case PanelType::TAPE:
    panel = std::make_unique<TapePanel>(config, bridge_, processor_);
    break;
  case PanelType::VOLUME_PROFILE:
    panel = std::make_unique<VolumeProfilePanel>(config, bridge_, processor_);
    break;
  case PanelType::DEPTH_CHART:
    panel = std::make_unique<DepthChartPanel>(config, bridge_, processor_);
    break;
  case PanelType::FOOTPRINT_CHART:
    panel = std::make_unique<FootprintPanel>(config, micro_renderer_);
    break;
  case PanelType::TPO_PROFILE:
    panel = std::make_unique<TpoPanel>(config, micro_renderer_);
    break;
  case PanelType::ALERTS:
    panel = std::make_unique<AlertsPanel>(config);
    break;
  case PanelType::SCATTER_PLOT:
  case PanelType::TIME_SERIES:
  case PanelType::TRADING_ORDERS:
  case PanelType::TRADING_POSITIONS:
  case PanelType::RISK_METRICS:
  case PanelType::HISTOGRAM:
  case PanelType::SCREENER:
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
  float x =
      static_cast<float>(grid_x) * (dashboard_size_.x / grid_layout_.columns);
  float y = 30.0f + static_cast<float>(grid_y) *
                        ((dashboard_size_.y - 30.0f) / grid_layout_.rows);
  return ImVec2(x, y);
}

ImVec2 PanelManager::calculate_panel_size(int width, int height) const {
  float w =
      static_cast<float>(width) * (dashboard_size_.x / grid_layout_.columns);
  float h = static_cast<float>(height) *
            ((dashboard_size_.y - 30.0f) / grid_layout_.rows);
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
  case PanelType::FOOTPRINT_CHART:
    return "Footprint Chart";
  case PanelType::TPO_PROFILE:
    return "TPO Profile";
  case PanelType::LOG_PANEL:
    return "Log";
  default:
    return "Panel";
  }
}

void PanelManager::save_layout(const std::string &filename) {
  try {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << serialize_layout();
      file.close();
      std::cout << "Layout saved to " << filename << std::endl;
    } else {
      std::cerr << "Failed to open file for saving layout: " << filename
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error saving layout: " << e.what() << std::endl;
  }
}

void PanelManager::load_layout(const std::string &filename) {
  try {
    std::ifstream file(filename);
    if (file.is_open()) {
      std::string json_str((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
      deserialize_layout(json_str);
      file.close();
      std::cout << "Layout loaded from " << filename << std::endl;
    } else {
      std::cerr << "Failed to open file for loading layout: " << filename
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error loading layout: " << e.what() << std::endl;
  }
}

std::string PanelManager::serialize_layout() const {
  json layout_json;
  layout_json["grid"] = {{"columns", grid_layout_.columns},
                         {"rows", grid_layout_.rows}};

  json panels_json = json::array();
  for (const auto &[id, panel] : panels_) {
    const auto &config = panel->get_config();
    json panel_json;
    panel_json["type"] = static_cast<int>(config.type);
    panel_json["title"] = config.title;
    panel_json["visible"] = config.visible;
    panel_json["grid_x"] = config.grid_x;
    panel_json["grid_y"] = config.grid_y;
    panel_json["grid_width"] = config.grid_width;
    panel_json["grid_height"] = config.grid_height;

    // Optional: save exact position/size if manually moved (overriding
    // grid) panel_json["pos_x"] = config.position.x;
    // ...

    panels_json.push_back(panel_json);
  }
  layout_json["panels"] = panels_json;

  return layout_json.dump(4);
}

void PanelManager::deserialize_layout(const std::string &layout_json) {
  try {
    auto j = json::parse(layout_json);

    // clear existing panels
    panels_.clear();
    // Reset ID counter? Maybe risky if other things hold IDs, but typically
    // fine for full reload. However, if we don't reset, IDs grow
    // indefinitely. Let's reset for fresh start.
    next_panel_id_ = 1;

    if (j.contains("grid")) {
      set_grid_layout(j["grid"]["columns"], j["grid"]["rows"]);
    }

    if (j.contains("panels")) {
      for (const auto &p : j["panels"]) {
        PanelType type = static_cast<PanelType>(p["type"].get<int>());
        std::string title = p["title"].get<std::string>();
        int grid_x = p["grid_x"].get<int>();
        int grid_y = p["grid_y"].get<int>();
        int width = p["grid_width"].get<int>();
        int height = p["grid_height"].get<int>();
        bool visible = p["visible"].get<bool>();

        uint32_t id = add_panel(type, title, grid_x, grid_y, width, height);
        set_panel_visible(id, visible);
      }
    }

    // Re-initialize active symbol after load if possible,
    // or let the orchestrator handle it.
    // For now, minimal restoration.

  } catch (const std::exception &e) {
    std::cerr << "Error deserializing layout: " << e.what() << std::endl;
  }
}

void PanelManager::set_active_symbol(uint32_t symbol_id,
                                     const std::string &symbol_name) {
  // Track active symbol for new panels
  active_symbol_id_ = symbol_id;
  active_symbol_name_ = symbol_name;

  // Propagate symbol to all relevant panel types
  for (auto &[id, panel] : panels_) {
    switch (panel->get_config().type) {
    case PanelType::ORDERBOOK: {
      if (auto *orderbook = dynamic_cast<OrderbookPanel *>(panel.get())) {
        orderbook->set_symbol(symbol_id, symbol_name);
      }
      break;
    }
    case PanelType::CHART: {
      if (auto *chart = dynamic_cast<ChartPanel *>(panel.get())) {
        chart->set_symbol(symbol_name, bridge_->getExchangeName(symbol_id));
      }
      break;
    }
    case PanelType::WATCHLIST: {
      // Add symbol to watchlist if not already present
      if (auto *watchlist = dynamic_cast<WatchlistPanel *>(panel.get())) {
        watchlist->add_symbol(symbol_id, symbol_name,
                              bridge_->getExchangeName(symbol_id));
      }
      break;
    }
    case PanelType::TAPE: {
      if (auto *tape = dynamic_cast<TapePanel *>(panel.get())) {
        tape->set_symbol(symbol_id, symbol_name);
      }
      break;
    }
    case PanelType::VOLUME_PROFILE: {
      if (auto *vp = dynamic_cast<VolumeProfilePanel *>(panel.get())) {
        vp->set_symbol(symbol_id, symbol_name);
      }
      break;
    }
    case PanelType::DEPTH_CHART: {
      if (auto *dc = dynamic_cast<DepthChartPanel *>(panel.get())) {
        dc->set_symbol(symbol_id, symbol_name);
      }
      break;
    }
    case PanelType::FOOTPRINT_CHART: {
      if (auto *fp = dynamic_cast<FootprintPanel *>(panel.get())) {
        fp->set_symbol_id(symbol_id);
      }
      break;
    }
    case PanelType::TPO_PROFILE: {
      if (auto *tpo = dynamic_cast<TpoPanel *>(panel.get())) {
        tpo->set_symbol_id(symbol_id);
      }
      break;
    }
    case PanelType::HEATMAP: {
      if (auto *dom = dynamic_cast<DomSurfacePanel *>(panel.get())) {
        dom->setSymbol(symbol_id);
      }
      break;
    }
    default:
      break;
    }
  }
}

} // namespace BTQuant