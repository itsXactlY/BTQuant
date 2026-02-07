#include "../../include/components/panel_manager.hpp"

#include <fstream>
#include <iostream>
#include <tuple>
#include <unordered_map>

#include "../../include/components/time_and_sales.hpp"
#include "../../include/components/historical_time_sales.hpp"
#include "../../include/rendering/panel_culler.hpp"
#include <nlohmann/json.hpp>

#include "../../include/components/alerts_panel.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/components/depth_chart_panel.hpp"
#include "../../include/components/dom_surface_panel.hpp"
#include "../../include/components/footprint_panel.hpp"
#include "../../include/components/histogram_panel.hpp"
#include "../../include/components/log_panel.hpp"
#include "../../include/components/metrics_panel.hpp"
#include "../../include/components/orderbook_panel.hpp"
#include "../../include/components/performance_monitor_panel.hpp"
#include "../../include/components/risk_metrics_panel.hpp"
#include "../../include/components/scatter_plot_panel.hpp"
#include "../../include/components/screener_panel.hpp"
#include "../../include/components/status_bar_panel.hpp"
#include "../../include/components/tape_panel.hpp"
#include "../../include/components/tabbed_panel.hpp"
#include "../../include/components/time_series_panel.hpp"
#include "../../include/components/time_statistics_panel.hpp"
#include "../../include/components/time_histogram_panel.hpp"
#include "../../include/components/tpo_panel.hpp"
#include "../../include/components/trading_orders_panel.hpp"
#include "../../include/components/trading_positions_panel.hpp"
#include "../../include/components/volume_profile_panel.hpp"
#include "../../include/components/watchlist_panel.hpp"
#include "../../include/components/chart_replay_panel.hpp"
#include "../../include/components/risk_analyzer_panel.hpp"
#include "../../include/components/strategy_builder.hpp"
#include "../../include/components/option_analytics_panel.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/performance/panel_profiler.hpp"

using json = nlohmann::json;

namespace BTQuant {

PanelManager::PanelManager(std::shared_ptr<HotSpineDataBridge> bridge,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                           std::shared_ptr<OrderManager> order_manager,
                           std::shared_ptr<PositionManager> position_manager,
                           std::shared_ptr<RiskAssessment> risk_assessment,
                           RenderEngine::MarketMicrostructureRenderer* micro_renderer)
    : bridge_(bridge),
      processor_(processor),
      order_manager_(order_manager),
      position_manager_(position_manager),
      risk_assessment_(risk_assessment),
      micro_renderer_(micro_renderer) {
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
  context_menu_manager_ = std::make_unique<ContextMenuManager>(this);
  strategy_builder_ = std::make_unique<RenderEngine::StrategyBuilder>(PanelConfig{.title = "Strategy Builder", .type = PanelType::STRATEGY_BUILDER});

  // Set the panel manager reference in the MarketMicrostructureRenderer for dirty state updates
  if (micro_renderer_) {
    micro_renderer_->set_panel_manager(this);
  }
}

PanelManager::~PanelManager() {
  context_menu_manager_.reset(); // Explicitly reset context menu manager before other members
  strategy_builder_.reset(); // Explicitly reset strategy builder before other members
  panels_.clear();
}

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
      for (auto& [id, panel] : panels_) {
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

  for (auto& [id, panel] : panels_) {
    panel->update(dt);
  }
}

void PanelManager::render() {
  // Update dashboard size
  ImVec2 current_size = ImGui::GetIO().DisplaySize;
  if (current_size.x > 0 && current_size.y > 0) {
    dashboard_size_ = current_size;
  }

  // Use panel culling to avoid rendering off-screen or minimized panels
  RenderEngine::PanelCuller culler;
  culler.set_viewport_bounds(ImVec2(0.0f, 0.0f), dashboard_size_);

  // Collect all panels into a vector for batch culling along with their IDs
  std::vector<std::pair<uint32_t, const BTQuant::PanelBase*>> panels_with_ids;
  panels_with_ids.reserve(panels_.size());
  for (const auto& [id, panel] : panels_) {
    panels_with_ids.emplace_back(id, panel.get());
  }

  // Extract just the panel pointers for culling
  std::vector<const BTQuant::PanelBase*> all_panels;
  all_panels.reserve(panels_with_ids.size());
  for (const auto& [id, panel] : panels_with_ids) {
    all_panels.push_back(panel);
  }

  // Perform batch culling to get only visible panels
  auto visible_panels = culler.cull_panels(all_panels);

  // Create a map from panel pointers to IDs for quick lookup
  std::unordered_map<const BTQuant::PanelBase*, uint32_t> panel_to_id;
  panel_to_id.reserve(panels_with_ids.size());
  for (const auto& [id, panel] : panels_with_ids) {
    panel_to_id[panel] = id;
  }

  // Process context menu for only the visible panels
  for (const auto* panel : visible_panels) {
    // Cast back to non-const pointer to call handle_context_menu (since handle_context_menu is non-const)
    const_cast<BTQuant::PanelBase*>(panel)->handle_context_menu(*context_menu_manager_);
  }

  // Handle drag and drop for tabbed groups
  handle_panel_drag_drop();

  // Render only the visible panels
  // Grouped panels should be rendered together to ensure proper positioning
  std::unordered_set<uint32_t> processed_groups;
  for (const auto* panel : visible_panels) {
    uint32_t panel_id = panel_to_id.at(panel);  // Safe lookup with at()

    // Check if this panel is part of a group
    uint32_t group_id = get_panel_group_id(panel_id);

    if (group_id != 0 && processed_groups.find(group_id) == processed_groups.end()) {
      // This panel belongs to a group that hasn't been processed yet
      // Render all panels in this group together
      const auto* group = get_panel_group(group_id);
      if (group) {
        for (uint32_t group_panel_id : group->panel_ids) {
          auto group_panel_it = panels_.find(group_panel_id);
          if (group_panel_it != panels_.end()) {
            auto* group_panel = group_panel_it->second.get();

            // Start timing the panel render
            BTQuant::g_panel_profiler.start_panel_render(group_panel_id, group_panel->get_title());

            // Render the panel
            group_panel->render();

            // End timing the panel render
            BTQuant::g_panel_profiler.end_panel_render(group_panel_id);
          }
        }

        // Mark this group as processed to avoid duplicate rendering
        processed_groups.insert(group_id);
      }
    } else if (group_id == 0) {
      // This panel is not part of a group, render it individually

      // Start timing the panel render
      BTQuant::g_panel_profiler.start_panel_render(panel_id, panel->get_title());

      // Cast back to non-const pointer to call render (since render() is non-const)
      const_cast<BTQuant::PanelBase*>(panel)->render();

      // End timing the panel render
      BTQuant::g_panel_profiler.end_panel_render(panel_id);
    }
  }
}

uint32_t PanelManager::add_panel(PanelType type, const std::string& title, int grid_x, int grid_y,
                                 int width, int height) {
  uint32_t panel_id = next_panel_id_++;

  // If grid coordinates are not specified (-1), use auto-dock to find an appropriate position
  if (grid_x == -1 || grid_y == -1) {
    auto [auto_x, auto_y] = find_auto_dock_position(width, height);
    if (auto_x != -1 && auto_y != -1) {
      grid_x = auto_x;
      grid_y = auto_y;
    } else {
      // If no suitable dock position found, default to (0, 0)
      grid_x = 0;
      grid_y = 0;
    }
  }

  PanelConfig config = create_panel_config(type, title, grid_x, grid_y, width, height);

  std::unique_ptr<PanelBase> panel;
  switch (type) {
    case PanelType::CHART:
      panel = std::make_unique<ChartPanel>(config, bridge_, processor_, chart_manager_.get(), this);

      // Set up scroll synchronization from Chart to TimeStats (reverse direction)
      if (auto* chart_panel = dynamic_cast<ChartPanel*>(panel.get())) {
        chart_panel->set_scroll_sync_callback([this](uint64_t start_timestamp, uint64_t end_timestamp) {
          // Find the active time statistics panel and adjust its view to match the time range
          for (auto& [id, panel] : panels_) {
            if (auto* time_stats_panel = dynamic_cast<TimeStatisticsPanel*>(panel.get())) {
              // Scroll the time statistics panel to show the corresponding time range
              time_stats_panel->scroll_to_time_range(start_timestamp, end_timestamp);
              break; // Assuming we want to adjust the first time stats panel we find
            }
          }
        });
      }
      break;
    case PanelType::TIME_STATISTICS: {
      auto time_stats = std::make_unique<TimeStatisticsPanel>(config);

      // Set up the row double-click callback
      time_stats->set_row_double_clicked_callback([this](uint64_t timestamp) {
        // Find the active chart panel and center it on the clicked timestamp
        for (auto& [id, panel] : panels_) {
          if (auto* chart_panel = dynamic_cast<ChartPanel*>(panel.get())) {
            // Center the chart on the clicked timestamp
            chart_panel->center_on_timestamp(timestamp);
            break; // Assuming we want to center the first chart panel we find
          }
        }
      });

      // Set up scroll synchronization from TimeStats to Chart
      time_stats->set_scroll_sync_callback([this](uint64_t start_timestamp, uint64_t end_timestamp) {
        // Find the active chart panel and adjust its view to match the time range
        for (auto& [id, panel] : panels_) {
          if (auto* chart_panel = dynamic_cast<ChartPanel*>(panel.get())) {
            // Convert timestamps to the format used by the chart (seconds)
            double start_time_seconds = static_cast<double>(start_timestamp) / 1000000.0;
            double end_time_seconds = static_cast<double>(end_timestamp) / 1000000.0;

            // Update the chart's view range
            chart_panel->last_view_min_ = start_time_seconds;
            chart_panel->last_view_max_ = end_time_seconds;
            chart_panel->follow_latest_ = false; // Disable auto-follow to maintain the synchronized view

            break; // Assuming we want to adjust the first chart panel we find
          }
        }
      });

      panel = std::move(time_stats);
      break;
    }
    case PanelType::TIME_AND_SALES: {
      panel = std::make_unique<TimeAndSalesPanel>(config, bridge_, processor_);
      break;
    }
    case PanelType::HISTORICAL_TIME_SALES: {
      panel = std::make_unique<HistoricalTimeSalesPanel>(config, bridge_, processor_);
      break;
    }
    case PanelType::TIME_HISTOGRAM:
      panel = std::make_unique<TimeHistogramPanel>(config);
      break;
    case PanelType::METRICS:
      panel =
          std::make_unique<MetricsPanel>(config, position_manager_, risk_assessment_, processor_);
      break;
    case PanelType::HEATMAP:
      panel = std::make_unique<DomSurfacePanel>(processor_);
      break;
    case PanelType::ORDERBOOK:
      panel = std::make_unique<OrderbookPanel>(config, bridge_, processor_);
      break;
    case PanelType::PERFORMANCE_MONITOR:
      panel = std::make_unique<PerformanceMonitorPanel>(config);
      break;
    case PanelType::STATUS_BAR:
      panel = std::make_unique<StatusBarPanel>(config, bridge_, processor_);
      break;
    case PanelType::WATCHLIST: {
      auto watchlist = std::make_unique<WatchlistPanel>(config, bridge_, processor_);
      watchlist->set_symbol_selected_callback(
          [this](uint32_t symbol_id, const std::string& symbol_name) {
            this->set_active_symbol(symbol_id, symbol_name);
          });
      panel = std::move(watchlist);
      break;
    }
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
    case PanelType::OPTION_ANALYTICS:
      panel = std::make_unique<BTQuant::RenderEngine::OptionAnalyticsPanel>(processor_);
      break;
    case PanelType::ALERTS:
      panel = std::make_unique<AlertsPanel>(config);
      break;
    case PanelType::SCATTER_PLOT:
      panel = std::make_unique<ScatterPlotPanel>(config);
      break;
    case PanelType::TIME_SERIES:
      panel = std::make_unique<TimeSeriesPanel>(config);
      break;
    case PanelType::TRADING_ORDERS:
      panel = std::make_unique<TradingOrdersPanel>(config, order_manager_, position_manager_);
      break;
    case PanelType::TRADING_POSITIONS:
      panel = std::make_unique<TradingPositionsPanel>(config, position_manager_, risk_assessment_);
      break;
    case PanelType::RISK_METRICS:
      panel = std::make_unique<RiskMetricsPanel>(config, risk_assessment_, position_manager_);
      break;
    case PanelType::HISTOGRAM:
      panel = std::make_unique<HistogramPanel>(config);
      break;
    case PanelType::SCREENER:
      panel = std::make_unique<ScreenerPanel>(config);
      break;
    case PanelType::LOG_PANEL:
      panel = std::make_unique<LogPanel>(config);
      break;
    case PanelType::CHART_REPLAY:
      panel = std::make_unique<ChartReplayPanel>(config, bridge_, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, bridge_, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::TABBED_GROUP:
      panel = std::make_unique<TabbedPanel>(config, this);
      break;
    default:
      return 0;
  }

  if (panel) {
    panel->initialize();
    panels_[panel_id] = std::move(panel);

    // Special handling for connecting watchlist and alerts panels
    // Check if we now have both panels and connect them
    if (type == PanelType::WATCHLIST || type == PanelType::ALERTS) {
      WatchlistPanel* watchlist_panel = nullptr;
      AlertsPanel* alerts_panel = nullptr;

      // Look for both panels in the collection
      for (auto& [id, existing_panel] : panels_) {
        if (existing_panel->get_config().type == PanelType::WATCHLIST) {
          watchlist_panel = dynamic_cast<WatchlistPanel*>(existing_panel.get());
        } else if (existing_panel->get_config().type == PanelType::ALERTS) {
          alerts_panel = dynamic_cast<AlertsPanel*>(existing_panel.get());
        }

        if (watchlist_panel && alerts_panel) {
          break; // Both found, exit early
        }
      }

      // If both panels exist, connect them using a raw pointer (the panel manager owns both)
      if (watchlist_panel && alerts_panel) {
        watchlist_panel->set_alerts_panel_raw(alerts_panel);
      }
    }

    // Notify all registered callbacks about the new panel
    for (const auto& callback : panel_added_callbacks_) {
      if (callback) {
        callback(panel_id, type);
      }
    }
  }

  return panel_id;
}

uint32_t PanelManager::add_panel_with_symbol(PanelType type, const std::string& title,
                                             const std::string& symbol, int grid_x, int grid_y,
                                             int width, int height) {
  uint32_t panel_id = next_panel_id_++;

  // If grid coordinates are not specified (-1), use auto-dock to find an appropriate position
  if (grid_x == -1 || grid_y == -1) {
    auto [auto_x, auto_y] = find_auto_dock_position(width, height);
    if (auto_x != -1 && auto_y != -1) {
      grid_x = auto_x;
      grid_y = auto_y;
    } else {
      // If no suitable dock position found, default to (0, 0)
      grid_x = 0;
      grid_y = 0;
    }
  }

  PanelConfig config =
      create_panel_config_with_symbol(type, title, symbol, grid_x, grid_y, width, height);

  std::unique_ptr<PanelBase> panel;
  switch (type) {
    case PanelType::CHART:
      panel = std::make_unique<ChartPanel>(config, bridge_, processor_, chart_manager_.get(), this);

      // Set up scroll synchronization from Chart to TimeStats (reverse direction)
      if (auto* chart_panel = dynamic_cast<ChartPanel*>(panel.get())) {
        chart_panel->set_scroll_sync_callback([this](uint64_t start_timestamp, uint64_t end_timestamp) {
          // Find the active time statistics panel and adjust its view to match the time range
          for (auto& [id, panel] : panels_) {
            if (auto* time_stats_panel = dynamic_cast<TimeStatisticsPanel*>(panel.get())) {
              // Scroll the time statistics panel to show the corresponding time range
              time_stats_panel->scroll_to_time_range(start_timestamp, end_timestamp);
              break; // Assuming we want to adjust the first time stats panel we find
            }
          }
        });
      }
      break;
    case PanelType::TIME_AND_SALES: {
      panel = std::make_unique<TimeAndSalesPanel>(config, bridge_, processor_);
      break;
    }
    case PanelType::HISTORICAL_TIME_SALES: {
      panel = std::make_unique<HistoricalTimeSalesPanel>(config, bridge_, processor_);
      break;
    }
    case PanelType::TIME_HISTOGRAM:
      panel = std::make_unique<TimeHistogramPanel>(config);
      break;
    case PanelType::METRICS:
      panel =
          std::make_unique<MetricsPanel>(config, position_manager_, risk_assessment_, processor_);
      break;
    case PanelType::HEATMAP:
      panel = std::make_unique<DomSurfacePanel>(processor_);
      break;
    case PanelType::ORDERBOOK:
      panel = std::make_unique<OrderbookPanel>(config, bridge_, processor_);
      break;
    case PanelType::PERFORMANCE_MONITOR:
      panel = std::make_unique<PerformanceMonitorPanel>(config);
      break;
    case PanelType::STATUS_BAR:
      panel = std::make_unique<StatusBarPanel>(config, bridge_, processor_);
      break;
    case PanelType::WATCHLIST: {
      auto watchlist = std::make_unique<WatchlistPanel>(config, bridge_, processor_);
      watchlist->set_symbol_selected_callback(
          [this](uint32_t symbol_id, const std::string& symbol_name) {
            this->set_active_symbol(symbol_id, symbol_name);
          });
      panel = std::move(watchlist);
      break;
    }
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
    case PanelType::OPTION_ANALYTICS:
      panel = std::make_unique<BTQuant::RenderEngine::OptionAnalyticsPanel>(processor_);
      break;
    case PanelType::ALERTS:
      panel = std::make_unique<AlertsPanel>(config);
      break;
    case PanelType::SCATTER_PLOT:
      panel = std::make_unique<ScatterPlotPanel>(config);
      break;
    case PanelType::TIME_SERIES:
      panel = std::make_unique<TimeSeriesPanel>(config);
      break;
    case PanelType::TRADING_ORDERS:
      panel = std::make_unique<TradingOrdersPanel>(config, order_manager_, position_manager_);
      break;
    case PanelType::TRADING_POSITIONS:
      panel = std::make_unique<TradingPositionsPanel>(config, position_manager_, risk_assessment_);
      break;
    case PanelType::RISK_METRICS:
      panel = std::make_unique<RiskMetricsPanel>(config, risk_assessment_, position_manager_);
      break;
    case PanelType::HISTOGRAM:
      panel = std::make_unique<HistogramPanel>(config);
      break;
    case PanelType::SCREENER:
      panel = std::make_unique<ScreenerPanel>(config);
      break;
    case PanelType::LOG_PANEL:
      panel = std::make_unique<LogPanel>(config);
      break;
    case PanelType::CHART_REPLAY:
      panel = std::make_unique<ChartReplayPanel>(config, bridge_, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, bridge_, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::TABBED_GROUP:
      panel = std::make_unique<TabbedPanel>(config, this);
      break;
    default:
      return 0;
  }

  if (panel) {
    panel->initialize();
    panels_[panel_id] = std::move(panel);

    // Special handling for connecting watchlist and alerts panels
    // Check if we now have both panels and connect them
    if (type == PanelType::WATCHLIST || type == PanelType::ALERTS) {
      WatchlistPanel* watchlist_panel = nullptr;
      AlertsPanel* alerts_panel = nullptr;

      // Look for both panels in the collection
      for (auto& [id, existing_panel] : panels_) {
        if (existing_panel->get_config().type == PanelType::WATCHLIST) {
          watchlist_panel = dynamic_cast<WatchlistPanel*>(existing_panel.get());
        } else if (existing_panel->get_config().type == PanelType::ALERTS) {
          alerts_panel = dynamic_cast<AlertsPanel*>(existing_panel.get());
        }

        if (watchlist_panel && alerts_panel) {
          break; // Both found, exit early
        }
      }

      // If both panels exist, connect them using a raw pointer (the panel manager owns both)
      if (watchlist_panel && alerts_panel) {
        watchlist_panel->set_alerts_panel_raw(alerts_panel);
      }
    }

    // Notify all registered callbacks about the new panel
    for (const auto& callback : panel_added_callbacks_) {
      if (callback) {
        callback(panel_id, type);
      }
    }
  }

  return panel_id;
}

uint32_t PanelManager::find_panel_by_type(PanelType type) const {
  for (const auto& [id, panel] : panels_) {
    if (panel->get_config().type == type) {
      return id;
    }
  }
  return 0; // Return 0 if no panel of the specified type is found
}

PanelBase* PanelManager::get_panel_by_id(uint32_t panel_id) const {
  auto it = panels_.find(panel_id);
  return it != panels_.end() ? it->second.get() : nullptr;
}

// This method is not needed since we handle panel connections differently
// The connection happens in the set_alerts_panel method of WatchlistPanel
// when both panels are available

void PanelManager::remove_panel(uint32_t panel_id) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    // Check if the panel is part of a group
    auto group_it = panel_to_group_map_.find(panel_id);
    if (group_it != panel_to_group_map_.end()) {
      uint32_t group_id = group_it->second;
      // Remove the panel from its group
      remove_panel_from_group(group_id, panel_id);
    }

    panels_.erase(it);

    // Notify all registered callbacks about the removed panel
    for (const auto& callback : panel_removed_callbacks_) {
      if (callback) {
        callback(panel_id);
      }
    }
  }
}

void PanelManager::clear_panels() {
  panels_.clear();
  panel_groups_.clear();
  panel_to_group_map_.clear();
  next_panel_id_ = 1;
  next_group_id_ = 1;
}

void PanelManager::move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y) {
  // Check if the panel is part of a group
  uint32_t group_id = get_panel_group_id(panel_id);
  if (group_id != 0) {
    // Moving a panel in a group means moving the entire group
    auto* group = get_panel_group(group_id);
    if (group) {
      // Validate the new position to ensure the entire group fits within grid bounds
      int new_group_right = new_grid_x + group->total_width;
      int new_group_bottom = new_grid_y + group->total_height;
      
      if (new_group_right > grid_layout_.columns || new_group_bottom > grid_layout_.rows) {
        // Position would put the group outside the grid bounds, reject the move
        return;
      }
      
      // Check for collisions with other panels that are not in this group
      for (int x = new_grid_x; x < new_group_right; x++) {
        for (int y = new_grid_y; y < new_group_bottom; y++) {
          // Check if this grid cell is occupied by a panel not in this group
          for (const auto& [id, panel] : panels_) {
            const auto& config = panel->get_config();
            
            // Skip panels in the same group
            if (get_panel_group_id(id) == group_id) {
              continue;
            }
            
            // Check if this panel occupies the grid cell
            int panel_right = config.grid_x + config.grid_width;
            int panel_bottom = config.grid_y + config.grid_height;
            
            if (x >= config.grid_x && x < panel_right && 
                y >= config.grid_y && y < panel_bottom) {
              // Collision detected, reject the move
              return;
            }
          }
        }
      }

      // Calculate the offset between the current position and the new position
      int offset_x = new_grid_x - group->min_grid_x;
      int offset_y = new_grid_y - group->min_grid_y;

      // Move all panels in the group by the same offset
      for (uint32_t id : group->panel_ids) {
        auto panel_it = panels_.find(id);
        if (panel_it != panels_.end()) {
          auto& config = panel_it->second->get_config();
          config.grid_x += offset_x;
          config.grid_y += offset_y;
          config.position = calculate_panel_position(config.grid_x, config.grid_y);
        }
      }

      // Update the group's position
      group->min_grid_x = new_grid_x;
      group->min_grid_y = new_grid_y;
    }
  } else {
    // Panel is not in a group, move normally but validate the position
    auto it = panels_.find(panel_id);
    if (it != panels_.end()) {
      auto& config = it->second->get_config();
      
      // Check if the new position would cause overlap with other panels
      int new_right = new_grid_x + config.grid_width;
      int new_bottom = new_grid_y + config.grid_height;
      
      if (new_right <= grid_layout_.columns && new_bottom <= grid_layout_.rows) {
        // Check for collisions with other panels
        for (const auto& [id, panel] : panels_) {
          if (id == panel_id) continue; // Skip the panel being moved
          
          const auto& other_config = panel->get_config();
          int other_right = other_config.grid_x + other_config.grid_width;
          int other_bottom = other_config.grid_y + other_config.grid_height;
          
          // Check if rectangles overlap
          if (!(new_grid_x >= other_right || new_right <= other_config.grid_x ||
                new_grid_y >= other_bottom || new_bottom <= other_config.grid_y)) {
            // Collision detected, reject the move
            return;
          }
        }
        
        // No collision, proceed with the move
        config.grid_x = new_grid_x;
        config.grid_y = new_grid_y;
        config.position = calculate_panel_position(new_grid_x, new_grid_y);
      }
    }
  }
}

void PanelManager::resize_panel(uint32_t panel_id, int new_width, int new_height) {
  // Check if the panel is part of a group
  uint32_t group_id = get_panel_group_id(panel_id);
  if (group_id != 0) {
    // Resizing a panel in a locked group should resize the entire group
    auto* group = get_panel_group(group_id);
    if (group && group->locked) {
      // Calculate the scale factor for resizing
      float width_scale = static_cast<float>(new_width) / static_cast<float>(group->total_width);
      float height_scale = static_cast<float>(new_height) / static_cast<float>(group->total_height);

      // Resize all panels in the group proportionally
      for (uint32_t id : group->panel_ids) {
        auto panel_it = panels_.find(id);
        if (panel_it != panels_.end()) {
          auto& config = panel_it->second->get_config();

          // Calculate new dimensions based on the original proportions
          int orig_width = config.grid_width;
          int orig_height = config.grid_height;

          int new_panel_width = std::max(1, static_cast<int>(orig_width * width_scale));
          int new_panel_height = std::max(1, static_cast<int>(orig_height * height_scale));

          config.grid_width = new_panel_width;
          config.grid_height = new_panel_height;
          config.size = calculate_panel_size(new_panel_width, new_panel_height);
        }
      }

      // Update the group's total dimensions
      group->total_width = new_width;
      group->total_height = new_height;
    } else {
      // Group is unlocked or panel not in group, resize normally
      auto it = panels_.find(panel_id);
      if (it != panels_.end()) {
        auto& config = it->second->get_config();
        
        // Validate the new size to ensure it fits within grid bounds
        int new_right = config.grid_x + new_width;
        int new_bottom = config.grid_y + new_height;
        
        if (new_right <= grid_layout_.columns && new_bottom <= grid_layout_.rows) {
          // Check for collisions with other panels
          for (const auto& [id, panel] : panels_) {
            if (id == panel_id) continue; // Skip the panel being resized
            
            const auto& other_config = panel->get_config();
            int other_right = other_config.grid_x + other_config.grid_width;
            int other_bottom = other_config.grid_y + other_config.grid_height;
            
            // Check if rectangles overlap after resize
            if (!(new_right <= other_config.grid_x || new_bottom <= other_config.grid_y ||
                  config.grid_x >= other_right || config.grid_y >= other_bottom)) {
              // Collision detected, reject the resize
              return;
            }
          }
          
          // No collision, proceed with the resize
          config.grid_width = new_width;
          config.grid_height = new_height;
          config.size = calculate_panel_size(new_width, new_height);
        }
      }
    }
  } else {
    // Panel is not in a group, resize normally
    auto it = panels_.find(panel_id);
    if (it != panels_.end()) {
      auto& config = it->second->get_config();
      
      // Validate the new size to ensure it fits within grid bounds
      int new_right = config.grid_x + new_width;
      int new_bottom = config.grid_y + new_height;
      
      if (new_right <= grid_layout_.columns && new_bottom <= grid_layout_.rows) {
        // Check for collisions with other panels
        for (const auto& [id, panel] : panels_) {
          if (id == panel_id) continue; // Skip the panel being resized
          
          const auto& other_config = panel->get_config();
          int other_right = other_config.grid_x + other_config.grid_width;
          int other_bottom = other_config.grid_y + other_config.grid_height;
          
          // Check if rectangles overlap after resize
          if (!(new_right <= other_config.grid_x || new_bottom <= other_config.grid_y ||
                config.grid_x >= other_right || config.grid_y >= other_bottom)) {
            // Collision detected, reject the resize
            return;
          }
        }
        
        // No collision, proceed with the resize
        config.grid_width = new_width;
        config.grid_height = new_height;
        config.size = calculate_panel_size(new_width, new_height);
      }
    }
  }
}

void PanelManager::set_grid_layout(int columns, int rows) {
  grid_layout_.columns = columns;
  grid_layout_.rows = rows;
}

void PanelManager::auto_arrange_panels() {
  int i = 0;
  for (auto& [id, panel] : panels_) {
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

void PanelManager::set_panel_symbol(uint32_t panel_id, const std::string& symbol) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    // Update the panel's symbol if it has symbol-dependent functionality
    switch (it->second->get_config().type) {
      case PanelType::ORDERBOOK: {
        if (auto* orderbook = dynamic_cast<OrderbookPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            orderbook->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      case PanelType::CHART: {
        if (auto* chart = dynamic_cast<ChartPanel*>(it->second.get())) {
          // Extract exchange from the symbol or use a default
          std::string exchange = "binance";  // Default exchange
          chart->set_symbol(symbol, exchange);
        }
        break;
      }
      case PanelType::WATCHLIST: {
        if (auto* watchlist = dynamic_cast<WatchlistPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            watchlist->add_symbol(symbol_id, symbol, bridge_->getExchangeName(symbol_id));
          }
        }
        break;
      }
      case PanelType::TAPE: {
        if (auto* tape = dynamic_cast<TapePanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            tape->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      case PanelType::VOLUME_PROFILE: {
        if (auto* vp = dynamic_cast<VolumeProfilePanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            vp->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      case PanelType::DEPTH_CHART: {
        if (auto* dc = dynamic_cast<DepthChartPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            dc->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      case PanelType::FOOTPRINT_CHART: {
        if (auto* fp = dynamic_cast<FootprintPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            fp->set_symbol_id(symbol_id);
          }
        }
        break;
      }
      case PanelType::TPO_PROFILE: {
        if (auto* tpo = dynamic_cast<TpoPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            tpo->set_symbol_id(symbol_id);
          }
        }
        break;
      }
      case PanelType::HEATMAP: {
        if (auto* dom = dynamic_cast<DomSurfacePanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            dom->setSymbol(symbol_id);
          }
        }
        break;
      }
      case PanelType::TIME_AND_SALES: {
        if (auto* tas = dynamic_cast<TimeAndSalesPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            tas->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      case PanelType::HISTORICAL_TIME_SALES: {
        if (auto* hts = dynamic_cast<HistoricalTimeSalesPanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          if (bridge_) {
            // Use SymbolRegistry to find the symbol ID
            auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
            if (symbol_info_opt) {
              symbol_id = symbol_info_opt->id;
            }
          }
          if (symbol_id != 0) {
            hts->set_symbol(symbol_id, symbol);
          }
        }
        break;
      }
      default:
        break;
    }
  }
}

void PanelManager::register_panel_added_callback(PanelAddedCallback callback) {
  panel_added_callbacks_.push_back(callback);
}

void PanelManager::register_panel_removed_callback(PanelRemovedCallback callback) {
  panel_removed_callbacks_.push_back(callback);
}

std::vector<uint32_t> PanelManager::get_all_panel_ids() const {
  std::vector<uint32_t> ids;
  for (const auto& [id, panel] : panels_) {
    ids.push_back(id);
  }
  return ids;
}

PanelConfig PanelManager::get_panel_config(uint32_t panel_id) const {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    return it->second->get_config();
  }
  return PanelConfig{};  // Return default config if not found
}

void PanelManager::update_panel_config(uint32_t panel_id, const PanelConfig& config) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    // Update the panel's configuration
    auto& panel_config = it->second->get_config();
    panel_config.title = config.title;
    panel_config.position = config.position;
    panel_config.size = config.size;
    panel_config.visible = config.visible;
    panel_config.resizable = config.resizable;
    panel_config.movable = config.movable;
    panel_config.grid_x = config.grid_x;
    panel_config.grid_y = config.grid_y;
    panel_config.grid_width = config.grid_width;
    panel_config.grid_height = config.grid_height;
  }
}

PanelConfig PanelManager::create_panel_config(PanelType type, const std::string& title, int grid_x,
                                              int grid_y, int width, int height) {
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

PanelConfig PanelManager::create_panel_config_with_symbol(PanelType type, const std::string& title,
                                                          const std::string& symbol, int grid_x,
                                                          int grid_y, int width, int height) {
  PanelConfig config = create_panel_config(type, title, grid_x, grid_y, width, height);
  config.title = title.empty() ? get_default_panel_title(type) : title;
  // Use the symbol parameter to set the symbol in the config
  config.symbol = symbol;
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
    case PanelType::PERFORMANCE_MONITOR:
      return "Performance Monitor";
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
    case PanelType::TIME_STATISTICS:
      return "Time Statistics";
    case PanelType::TIME_HISTOGRAM:
      return "Time Histogram";
    case PanelType::CHART_REPLAY:
      return "Chart Replay";
    case PanelType::RISK_ANALYZER:
      return "Risk Analyzer";
    case PanelType::STRATEGY_BUILDER:
      return "Strategy Builder";
    case PanelType::OPTION_ANALYTICS:
      return "Option Analytics";
    default:
      return "Panel";
  }
}

void PanelManager::save_layout(const std::string& filename) {
  try {
    std::ofstream file(filename);
    if (file.is_open()) {
      file << serialize_layout();
      file.close();
      std::cout << "Layout saved to " << filename << std::endl;

      // Update current layout name based on filename
      std::string layout_name = filename;
      // Remove path and extension to get clean layout name
      size_t last_slash = layout_name.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        layout_name = layout_name.substr(last_slash + 1);
      }
      size_t last_dot = layout_name.find_last_of('.');
      if (last_dot != std::string::npos) {
        layout_name = layout_name.substr(0, last_dot);
      }
      set_current_layout_name(layout_name);
    } else {
      std::cerr << "Failed to open file for saving layout: " << filename << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error saving layout: " << e.what() << std::endl;
  }
}

void PanelManager::load_layout(const std::string& filename) {
  try {
    std::ifstream file(filename);
    if (file.is_open()) {
      std::string json_str((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
      deserialize_layout(json_str);
      file.close();
      std::cout << "Layout loaded from " << filename << std::endl;

      // Update current layout name based on filename
      std::string layout_name = filename;
      // Remove path and extension to get clean layout name
      size_t last_slash = layout_name.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        layout_name = layout_name.substr(last_slash + 1);
      }
      size_t last_dot = layout_name.find_last_of('.');
      if (last_dot != std::string::npos) {
        layout_name = layout_name.substr(0, last_dot);
      }
      set_current_layout_name(layout_name);
    } else {
      std::cerr << "Failed to open file for loading layout: " << filename << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error loading layout: " << e.what() << std::endl;
  }
}

std::string PanelManager::serialize_layout() const {
  json layout_json;
  layout_json["grid"] = {{"columns", grid_layout_.columns}, {"rows", grid_layout_.rows}};

  json panels_json = json::array();
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();
    json panel_json;
    panel_json["type"] = static_cast<int>(config.type);
    panel_json["title"] = config.title;
    panel_json["visible"] = config.visible;
    panel_json["grid_x"] = config.grid_x;
    panel_json["grid_y"] = config.grid_y;
    panel_json["grid_width"] = config.grid_width;
    panel_json["grid_height"] = config.grid_height;
    panel_json["position"] = {config.position.x, config.position.y};
    panel_json["size"] = {config.size.x, config.size.y};
    panel_json["symbol"] = config.symbol;
    panel_json["original_id"] = static_cast<int>(id);  // Store original ID for deserialization mapping

    // Check if this panel is part of a group and add group info
    auto group_it = panel_to_group_map_.find(id);
    if (group_it != panel_to_group_map_.end()) {
      panel_json["group_id"] = group_it->second;
    }

    // Check if this panel is part of a tabbed group and add tabbed group info
    uint32_t tabbed_group_id = get_containing_tabbed_group_id(id);
    if (tabbed_group_id != 0) {
      panel_json["tabbed_group_id"] = tabbed_group_id;
    }

    // Serialize panel-specific settings
    json settings_json;

    // TPO Panel specific settings
    if (auto* tpo_panel = dynamic_cast<TpoPanel*>(panel.get())) {
        settings_json["symbol_id"] = tpo_panel->get_symbol_id();
        settings_json["show_text"] = tpo_panel->get_show_text();
        settings_json["show_grid"] = tpo_panel->get_show_grid();
        settings_json["show_heatmap"] = tpo_panel->get_show_heatmap();
        settings_json["time_window"] = tpo_panel->get_time_window();
    }
    // DOM Surface Panel specific settings
    else if (auto* dom_panel = dynamic_cast<DomSurfacePanel*>(panel.get())) {
        settings_json["symbol_id"] = dom_panel->get_symbol_id();
        settings_json["price_range"] = dom_panel->get_price_range();
        settings_json["price_bins"] = dom_panel->get_price_bins();
        settings_json["auto_scale_price"] = dom_panel->get_auto_scale_price();
        settings_json["large_order_threshold"] = dom_panel->get_large_order_threshold();
        settings_json["enable_fade_out"] = dom_panel->get_enable_fade_out();
        settings_json["heatmap_intensity"] = dom_panel->get_heatmap_intensity();
        settings_json["persistence_threshold_ms"] = dom_panel->getPersistenceThresholdMs();
        settings_json["persistence_timeout_ms"] = dom_panel->getPersistenceTimeoutMs();
        settings_json["show_persistent_lines"] = dom_panel->getShowPersistentLines();
        settings_json["max_large_order_markers"] = dom_panel->getMaxLargeOrderMarkers();
    }
    // Option Analytics Panel specific settings
    else if (auto* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel*>(panel.get())) {
        settings_json["active_tab"] = option_panel->get_active_tab();
    }
    // Tabbed Panel specific settings
    else if (auto* tabbed_panel = dynamic_cast<TabbedPanel*>(panel.get())) {
        if (tabbed_panel->get_tab_count() > 0) {
            json tabbed_panels_json = json::array();
            const auto& tabbed_panel_ids = tabbed_panel->get_tabbed_panels();
            for (uint32_t tabbed_id : tabbed_panel_ids) {
                tabbed_panels_json.push_back(tabbed_id);
            }
            settings_json["tabbed_panels"] = tabbed_panels_json;
            settings_json["active_tab_id"] = tabbed_panel->get_active_tab();
        }
    }

    // Add settings if any were captured
    if (!settings_json.empty()) {
        panel_json["settings"] = settings_json;
    }

    panels_json.push_back(panel_json);
  }
  layout_json["panels"] = panels_json;

  // Serialize panel groups
  json groups_json = json::array();
  for (const auto& [group_id, group] : panel_groups_) {
    json group_json;
    group_json["id"] = group->group_id;
    group_json["min_grid_x"] = group->min_grid_x;
    group_json["min_grid_y"] = group->min_grid_y;
    group_json["total_width"] = group->total_width;
    group_json["total_height"] = group->total_height;
    group_json["locked"] = group->locked;
    group_json["prevent_overlap"] = group->prevent_overlap;

    json panel_ids_json = json::array();
    for (uint32_t panel_id : group->panel_ids) {
      panel_ids_json.push_back(panel_id);
    }
    group_json["panel_ids"] = panel_ids_json;

    groups_json.push_back(group_json);
  }
  layout_json["groups"] = groups_json;

  return layout_json.dump(4);
}

void PanelManager::deserialize_layout(const std::string& layout_json) {
  try {
    auto j = json::parse(layout_json);

    // clear existing panels and groups
    panels_.clear();
    panel_groups_.clear();
    panel_to_group_map_.clear();
    // Reset ID counters
    next_panel_id_ = 1;
    next_group_id_ = 1;

    if (j.contains("grid")) {
      set_grid_layout(j["grid"]["columns"], j["grid"]["rows"]);
    }

    // First, create all panels
    std::unordered_map<int, uint32_t> original_to_new_id_map; // Map original IDs to new IDs
    
    if (j.contains("panels")) {
      for (const auto& p : j["panels"]) {
        PanelType type = static_cast<PanelType>(p["type"].get<int>());
        std::string title = p["title"].get<std::string>();
        int grid_x = p["grid_x"].get<int>();
        int grid_y = p["grid_y"].get<int>();
        int width = p["grid_width"].get<int>();
        int height = p["grid_height"].get<int>();
        bool visible = p["visible"].get<bool>();

        // Create panel config with position and size
        PanelConfig config = create_panel_config(type, title, grid_x, grid_y, width, height);

        // Restore position and size if available
        if (p.contains("position")) {
            auto pos_array = p["position"];
            config.position = ImVec2(pos_array[0].get<float>(), pos_array[1].get<float>());
        }
        if (p.contains("size")) {
            auto size_array = p["size"];
            config.size = ImVec2(size_array[0].get<float>(), size_array[1].get<float>());
        }
        if (p.contains("symbol")) {
            config.symbol = p["symbol"].get<std::string>();
        }

        uint32_t id = add_panel(type, title, grid_x, grid_y, width, height);

        // Store the mapping from original ID to new ID
        original_to_new_id_map[p["original_id"].get<int>()] = id;

        // Get the newly created panel to apply specific settings
        auto* panel = get_panel_by_id(id);
        if (panel) {
            // Update the panel's config with restored position and size
            auto& panel_config = panel->get_config();
            panel_config.position = config.position;
            panel_config.size = config.size;

            // Apply panel-specific settings if available
            if (p.contains("settings")) {
                auto settings = p["settings"];

                // TPO Panel specific settings
                if (auto* tpo_panel = dynamic_cast<TpoPanel*>(panel)) {
                    if (settings.contains("symbol_id")) {
                        tpo_panel->set_symbol_id(settings["symbol_id"].get<uint32_t>());
                    }
                    if (settings.contains("show_text")) {
                        tpo_panel->set_show_text(settings["show_text"].get<bool>());
                    }
                    if (settings.contains("show_grid")) {
                        tpo_panel->set_show_grid(settings["show_grid"].get<bool>());
                    }
                    if (settings.contains("show_heatmap")) {
                        tpo_panel->set_show_heatmap(settings["show_heatmap"].get<bool>());
                    }
                    if (settings.contains("time_window")) {
                        tpo_panel->set_time_window(settings["time_window"].get<float>());
                    }
                }
                // DOM Surface Panel specific settings
                else if (auto* dom_panel = dynamic_cast<DomSurfacePanel*>(panel)) {
                    if (settings.contains("symbol_id")) {
                        dom_panel->setSymbol(settings["symbol_id"].get<uint32_t>());
                    }
                    if (settings.contains("price_range")) {
                        dom_panel->set_price_range(settings["price_range"].get<double>());
                    }
                    if (settings.contains("price_bins")) {
                        dom_panel->set_price_bins(settings["price_bins"].get<int>());
                    }
                    if (settings.contains("auto_scale_price")) {
                        dom_panel->set_auto_scale_price(settings["auto_scale_price"].get<bool>());
                    }
                    if (settings.contains("large_order_threshold")) {
                        dom_panel->set_large_order_threshold(settings["large_order_threshold"].get<double>());
                    }
                    if (settings.contains("enable_fade_out")) {
                        dom_panel->set_enable_fade_out(settings["enable_fade_out"].get<bool>());
                    }
                    if (settings.contains("heatmap_intensity")) {
                        dom_panel->set_heatmap_intensity(settings["heatmap_intensity"].get<float>());
                    }
                    if (settings.contains("persistence_threshold_ms")) {
                        dom_panel->setPersistenceThresholdMs(settings["persistence_threshold_ms"].get<uint64_t>());
                    }
                    if (settings.contains("persistence_timeout_ms")) {
                        dom_panel->setPersistenceTimeoutMs(settings["persistence_timeout_ms"].get<double>());
                    }
                    if (settings.contains("show_persistent_lines")) {
                        dom_panel->setShowPersistentLines(settings["show_persistent_lines"].get<bool>());
                    }
                    if (settings.contains("max_large_order_markers")) {
                        dom_panel->setMaxLargeOrderMarkers(settings["max_large_order_markers"].get<int>());
                    }
                }
                // Option Analytics Panel specific settings
                else if (auto* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel*>(panel)) {
                    if (settings.contains("active_tab")) {
                        option_panel->set_active_tab(settings["active_tab"].get<int>());
                    }
                }
                // Tabbed Panel specific settings
                else if (auto* tabbed_panel = dynamic_cast<TabbedPanel*>(panel)) {
                    if (settings.contains("tabbed_panels")) {
                        auto tabbed_panels_array = settings["tabbed_panels"];
                        for (const auto& tabbed_id_val : tabbed_panels_array) {
                            uint32_t original_tabbed_id = tabbed_id_val.get<uint32_t>();
                            // Map the original ID to the new ID
                            auto id_it = original_to_new_id_map.find(original_tabbed_id);
                            if (id_it != original_to_new_id_map.end()) {
                                uint32_t new_tabbed_id = id_it->second;
                                // Add the panel to the tabbed panel
                                tabbed_panel->add_panel_to_tab(new_tabbed_id);
                                
                                // Set the panel as invisible since it's now in a tab
                                if (auto* contained_panel = get_panel_by_id(new_tabbed_id)) {
                                    contained_panel->set_visible(false);
                                }
                            } else {
                                // If we can't find the mapping, try using the ID directly (for backward compatibility)
                                tabbed_panel->add_panel_to_tab(original_tabbed_id);
                            }
                        }
                        
                        // Set the active tab if specified
                        if (settings.contains("active_tab_id")) {
                            uint32_t original_active_tab_id = settings["active_tab_id"].get<uint32_t>();
                            auto id_it = original_to_new_id_map.find(original_active_tab_id);
                            if (id_it != original_to_new_id_map.end()) {
                                tabbed_panel->set_active_tab(id_it->second);
                            } else {
                                // If we can't find the mapping, try using the ID directly (for backward compatibility)
                                tabbed_panel->set_active_tab(original_active_tab_id);
                            }
                        }
                    }
                }
            }
        }

        set_panel_visible(id, visible);
      }
    }

    // Third, restore tabbed groups if they exist in the layout
    // We need to process tabbed groups after all panels are created
    if (j.contains("panels")) {
        for (const auto& p : j["panels"]) {
            if (p.contains("tabbed_group_id")) {
                uint32_t original_tabbed_group_id = p["tabbed_group_id"].get<uint32_t>();
                uint32_t original_panel_id = p["original_id"].get<int>();
                
                // Map the original IDs to new IDs
                auto tabbed_group_it = original_to_new_id_map.find(original_tabbed_group_id);
                auto panel_it = original_to_new_id_map.find(original_panel_id);
                
                if (tabbed_group_it != original_to_new_id_map.end() && panel_it != original_to_new_id_map.end()) {
                    uint32_t new_tabbed_group_id = tabbed_group_it->second;
                    uint32_t new_panel_id = panel_it->second;
                    
                    // Add the panel to the tabbed group
                    add_panel_to_tabbed_group(new_tabbed_group_id, new_panel_id);
                    
                    // Set the panel as invisible since it's now in a tab
                    if (auto* contained_panel = get_panel_by_id(new_panel_id)) {
                        contained_panel->set_visible(false);
                    }
                }
            }
        }
    }

    // Second, restore panel groups if they exist in the layout
    if (j.contains("groups")) {
      for (const auto& g : j["groups"]) {
        uint32_t group_id = g["id"].get<uint32_t>();
        int min_grid_x = g["min_grid_x"].get<int>();
        int min_grid_y = g["min_grid_y"].get<int>();
        int total_width = g["total_width"].get<int>();
        int total_height = g["total_height"].get<int>();
        bool locked = g["locked"].get<bool>();
        bool prevent_overlap = g.value("prevent_overlap", true); // Default to true for backward compatibility

        // Get the panel IDs for this group
        std::vector<uint32_t> panel_ids;
        if (g.contains("panel_ids")) {
          for (const auto& panel_id_val : g["panel_ids"]) {
            uint32_t original_panel_id = panel_id_val.get<uint32_t>();
            // Map the original ID to the new ID
            auto id_it = original_to_new_id_map.find(original_panel_id);
            if (id_it != original_to_new_id_map.end()) {
              panel_ids.push_back(id_it->second);
            } else {
              // If we can't find the mapping, try using the ID directly (for backward compatibility)
              panel_ids.push_back(original_panel_id);
            }
          }
        }

        // Create the group
        if (!panel_ids.empty()) {
          uint32_t new_group_id = create_panel_group(panel_ids);
          if (new_group_id != 0) {
            // Get the group and update its properties
            auto* group = get_panel_group(new_group_id);
            if (group) {
              group->min_grid_x = min_grid_x;
              group->min_grid_y = min_grid_y;
              group->total_width = total_width;
              group->total_height = total_height;
              group->locked = locked;
              group->prevent_overlap = prevent_overlap;

              // Update next_group_id if needed
              if (new_group_id >= next_group_id_) {
                next_group_id_ = new_group_id + 1;
              }
            }
          }
        }
      }
    }

    // Re-initialize active symbol after load if possible,
    // or let the orchestrator handle it.
    // For now, minimal restoration.

  } catch (const std::exception& e) {
    std::cerr << "Error deserializing layout: " << e.what() << std::endl;
  }
}

void PanelManager::set_active_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  // Track active symbol for new panels
  active_symbol_id_ = symbol_id;
  active_symbol_name_ = symbol_name;

  // Propagate symbol to all relevant panel types
  for (auto& [id, panel] : panels_) {
    switch (panel->get_config().type) {
      case PanelType::ORDERBOOK: {
        if (auto* orderbook = dynamic_cast<OrderbookPanel*>(panel.get())) {
          orderbook->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::CHART: {
        if (auto* chart = dynamic_cast<ChartPanel*>(panel.get())) {
          chart->set_symbol(symbol_name, bridge_->getExchangeName(symbol_id));
        }
        break;
      }
      case PanelType::WATCHLIST: {
        // Update the symbol for all watchlist panels
        if (auto* watchlist = dynamic_cast<WatchlistPanel*>(panel.get())) {
          watchlist->add_symbol(symbol_id, symbol_name, bridge_->getExchangeName(symbol_id));
        }
        break;
      }
      case PanelType::TAPE: {
        if (auto* tape = dynamic_cast<TapePanel*>(panel.get())) {
          tape->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::VOLUME_PROFILE: {
        if (auto* vp = dynamic_cast<VolumeProfilePanel*>(panel.get())) {
          vp->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::DEPTH_CHART: {
        if (auto* dc = dynamic_cast<DepthChartPanel*>(panel.get())) {
          dc->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::FOOTPRINT_CHART: {
        if (auto* fp = dynamic_cast<FootprintPanel*>(panel.get())) {
          fp->set_symbol_id(symbol_id);
        }
        break;
      }
      case PanelType::TPO_PROFILE: {
        if (auto* tpo = dynamic_cast<TpoPanel*>(panel.get())) {
          tpo->set_symbol_id(symbol_id);
        }
        break;
      }
      case PanelType::OPTION_ANALYTICS: {
        // OptionAnalyticsPanel doesn't typically require symbol-specific data
        break;
      }
      case PanelType::HEATMAP: {
        if (auto* dom = dynamic_cast<DomSurfacePanel*>(panel.get())) {
          dom->setSymbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::TIME_AND_SALES: {
        if (auto* tas = dynamic_cast<TimeAndSalesPanel*>(panel.get())) {
          tas->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::HISTORICAL_TIME_SALES: {
        if (auto* hts = dynamic_cast<HistoricalTimeSalesPanel*>(panel.get())) {
          hts->set_symbol(symbol_id, symbol_name);
        }
        break;
      }
      case PanelType::CHART_REPLAY: {
        // Chart replay panels don't need direct symbol updates as they manage their own replay data
        break;
      }
      case PanelType::TIME_STATISTICS: {
        // Time statistics panels are typically linked to charts and don't need direct symbol updates
        break;
      }
      case PanelType::TIME_HISTOGRAM: {
        // Time histogram panels don't typically require symbol-specific data
        break;
      }
      case PanelType::SCATTER_PLOT: {
        // Scatter plot panels don't typically require symbol-specific data
        break;
      }
      case PanelType::TIME_SERIES: {
        // Time series panels don't typically require symbol-specific data
        break;
      }
      case PanelType::HISTOGRAM: {
        // Histogram panels don't typically require symbol-specific data
        break;
      }
      case PanelType::SCREENER: {
        // Screener panels don't typically require symbol-specific data
        break;
      }
      case PanelType::PERFORMANCE_MONITOR: {
        // Performance monitor doesn't need symbol-specific data
        break;
      }
      default:
        break;
    }
  }
}

size_t PanelManager::get_panel_count() const { return panels_.size(); }

void PanelManager::save_all_panel_configs(const std::string& config_file) const {
  for (const auto& [id, panel] : panels_) {
    // Check if the panel is a WatchlistPanel and save its configuration
    if (auto* watchlist = dynamic_cast<WatchlistPanel*>(panel.get())) {
      watchlist->save_watchlist_order_to_config(config_file);
    }
  }
}

void PanelManager::load_all_panel_configs(const std::string& config_file) {
  for (auto& [id, panel] : panels_) {
    // Check if the panel is a WatchlistPanel and load its configuration
    if (auto* watchlist = dynamic_cast<WatchlistPanel*>(panel.get())) {
      watchlist->load_watchlist_order_from_config(config_file);
    }
  }
}

void PanelManager::apply_layout_preset(LayoutPreset preset) {
  clear_panels();

  switch (preset) {
    case LayoutPreset::DEFAULT:
      // Default layout: Basic trading setup with chart, orderbook, watchlist
      set_grid_layout(3, 5);

      // Row 0: Status Bar and Alerts
      add_panel(PanelType::STATUS_BAR, "Status Bar", 0, 0, 2, 1);
      add_panel(PanelType::ALERTS, "Alerts", 2, 0, 1, 1);

      // Row 1-2: Main Chart (2x2) and Depth Chart (1x2)
      add_panel(PanelType::CHART, "BTC-USDT Chart", 0, 1, 2, 2);
      add_panel(PanelType::DEPTH_CHART, "Depth Chart", 2, 1, 1, 2);

      // Row 3: Orderbook Ladder (2x1) and Tape (1x1)
      add_panel(PanelType::ORDERBOOK, "BTC-USDT Orderbook", 0, 3, 2, 1);
      add_panel(PanelType::TAPE, "Time & Sales", 2, 3, 1, 1);

      // Row 4: DOM Surface (2x1) and Watchlist (1x1)
      add_panel(PanelType::HEATMAP, "DOM Surface", 0, 4, 2, 1);
      add_panel(PanelType::WATCHLIST, "Watchlist", 2, 4, 1, 1);
      break;

    case LayoutPreset::MODERN_TRADING:
      // Modern Trading layout: Based on RealtimeDashboardComponent setup
      set_grid_layout(6, 5);

      // 1. Chart (Top Left, Large) - 4x3
      add_panel(PanelType::CHART, "BTCUSDT Chart", 0, 0, 4, 3);

      // 2. DOM Surface (Top Right) - 2x2
      add_panel(PanelType::HEATMAP, "DOM Surface", 4, 0, 2, 2);

      // 3. Orderbook (Middle Right) - 2x2
      add_panel(PanelType::ORDERBOOK, "Orderbook", 4, 2, 2, 2);

      // 4. Time & Sales (Bottom Left 1) - 2x1
      add_panel(PanelType::TAPE, "Time & Sales", 0, 3, 2, 1);

      // 5. Watchlist (Bottom Left 2) - 2x1
      add_panel(PanelType::WATCHLIST, "Watchlist", 2, 3, 2, 1);

      // 6. Positions / Risk (Bottom Row) - 6x1
      add_panel(PanelType::TRADING_POSITIONS, "Positions", 0, 4, 6, 1);
      break;

    case LayoutPreset::PRO_QUANT:
      // Pro Quant layout: Based on main_trading_terminal.cpp setup
      set_grid_layout(6, 10);

      // 1. Main Chart (Top Left, large)
      add_panel(PanelType::CHART, "BTC/USDT Chart", 0, 0, 4, 3);

      // 2. Orderbook / DOM (Right side)
      // DOM Surface (Heatmap)
      add_panel(PanelType::HEATMAP, "DOM Surface", 4, 0, 2, 2);
      // Classic Orderbook
      add_panel(PanelType::ORDERBOOK, "Orderbook", 4, 2, 2, 2);

      // 3. Bottom Row 1 (Tape / Orders)
      add_panel(PanelType::TAPE, "Time & Sales", 0, 3, 2, 1);
      add_panel(PanelType::TRADING_ORDERS, "Active Orders", 2, 3, 2, 1);

      // 4. Bottom Row 2 (Positions / Risk)
      add_panel(PanelType::TRADING_POSITIONS, "Positions", 0, 4, 2, 1);

      // 5. Add remaining components for complete integration
      // Volume Profile (Bottom Right)
      add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 2, 4, 2, 1);

      // Watchlist (Far Right Bottom)
      add_panel(PanelType::WATCHLIST, "Watchlist", 4, 4, 2, 1);

      // Add Footprint Chart and TPO Profile
      add_panel(PanelType::FOOTPRINT_CHART, "Footprint Chart", 0, 5, 3, 2);
      add_panel(PanelType::TPO_PROFILE, "TPO Profile", 3, 5, 3, 2);

      // Add Performance Monitor panel
      add_panel(PanelType::PERFORMANCE_MONITOR, "Performance Monitor", 0, 7, 6, 2);

      // Add Alerts panel
      add_panel(PanelType::ALERTS, "Alerts", 4, 5, 2, 2);

      // Add Strategy Builder as footer panel
      add_panel(PanelType::STRATEGY_BUILDER, "Strategy Builder Footer", 0, 8, 6, 1);
      break;

    case LayoutPreset::SCALPER_DOM:
      // Scalper DOM layout: Focus on DOM and order execution
      set_grid_layout(4, 6);

      // Main DOM heatmap taking most of the screen
      add_panel(PanelType::HEATMAP, "DOM Surface", 0, 0, 2, 4);
      
      // Large orderbook
      add_panel(PanelType::ORDERBOOK, "Orderbook", 2, 0, 2, 4);
      
      // Small chart for reference
      add_panel(PanelType::CHART, "Chart", 0, 4, 2, 2);
      
      // Time & Sales
      add_panel(PanelType::TAPE, "Time & Sales", 2, 4, 1, 1);
      
      // Active orders
      add_panel(PanelType::TRADING_ORDERS, "Active Orders", 3, 4, 1, 1);
      
      // Positions
      add_panel(PanelType::TRADING_POSITIONS, "Positions", 2, 5, 2, 1);
      
      // Watchlist
      add_panel(PanelType::WATCHLIST, "Watchlist", 0, 5, 2, 1);
      break;

    case LayoutPreset::ANALYTICS_FOCUS:
      // Analytics Focus layout: Charts and analytical tools
      set_grid_layout(6, 6);

      // Main chart
      add_panel(PanelType::CHART, "Main Chart", 0, 0, 3, 3);
      
      // Secondary chart
      add_panel(PanelType::CHART, "Secondary Chart", 3, 0, 3, 2);
      
      // Time statistics
      add_panel(PanelType::TIME_STATISTICS, "Time Statistics", 3, 2, 3, 2);
      
      // Volume profile
      add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 0, 3, 2, 2);
      
      // Footprint chart
      add_panel(PanelType::FOOTPRINT_CHART, "Footprint Chart", 2, 3, 2, 2);
      
      // TPO profile
      add_panel(PanelType::TPO_PROFILE, "TPO Profile", 4, 3, 2, 2);
      
      // Metrics
      add_panel(PanelType::METRICS, "Metrics", 0, 5, 2, 1);
      
      // Risk metrics
      add_panel(PanelType::RISK_METRICS, "Risk Metrics", 2, 5, 2, 1);
      
      // Watchlist
      add_panel(PanelType::WATCHLIST, "Watchlist", 4, 5, 2, 1);
      break;
  }
}

uint32_t PanelManager::create_panel_group(const std::vector<uint32_t>& panel_ids) {
  // Validate that all panels exist and are not already in a group
  for (uint32_t panel_id : panel_ids) {
    if (panels_.find(panel_id) == panels_.end()) {
      // Panel doesn't exist
      return 0;
    }
    if (panel_to_group_map_.find(panel_id) != panel_to_group_map_.end()) {
      // Panel is already in a group
      return 0;
    }
  }

  // Create a new group
  uint32_t group_id = next_group_id_++;
  auto group = std::make_unique<PanelGroup>(group_id);

  // Calculate the bounding rectangle of all panels in the group
  int min_x = INT_MAX, min_y = INT_MAX;
  int max_x = -1, max_y = -1;

  for (uint32_t panel_id : panel_ids) {
    auto it = panels_.find(panel_id);
    if (it != panels_.end()) {
      const auto& config = it->second->get_config();
      
      // Update the minimum coordinates
      min_x = std::min(min_x, config.grid_x);
      min_y = std::min(min_y, config.grid_y);
      
      // Update the maximum coordinates (considering panel dimensions)
      max_x = std::max(max_x, config.grid_x + config.grid_width);
      max_y = std::max(max_y, config.grid_y + config.grid_height);
      
      group->panel_ids.push_back(panel_id);
      panel_to_group_map_[panel_id] = group_id;
    }
  }

  // Set the group's position and dimensions
  group->min_grid_x = min_x;
  group->min_grid_y = min_y;
  group->total_width = max_x - min_x;
  group->total_height = max_y - min_y;

  // Store the group
  panel_groups_[group_id] = std::move(group);

  return group_id;
}

bool PanelManager::add_panel_to_group(uint32_t group_id, uint32_t panel_id) {
  // Check if group exists
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false;
  }

  // Check if panel exists
  if (panels_.find(panel_id) == panels_.end()) {
    return false;
  }

  // Check if panel is already in a group
  if (panel_to_group_map_.find(panel_id) != panel_to_group_map_.end()) {
    return false;
  }

  // Add panel to group
  auto& group = group_it->second;
  group->panel_ids.push_back(panel_id);
  panel_to_group_map_[panel_id] = group_id;

  // Recalculate group bounds
  const auto& config = panels_[panel_id]->get_config();
  int new_min_x = std::min(group->min_grid_x, config.grid_x);
  int new_min_y = std::min(group->min_grid_y, config.grid_y);
  int new_max_x = std::max(group->min_grid_x + group->total_width, config.grid_x + config.grid_width);
  int new_max_y = std::max(group->min_grid_y + group->total_height, config.grid_y + config.grid_height);

  group->min_grid_x = new_min_x;
  group->min_grid_y = new_min_y;
  group->total_width = new_max_x - new_min_x;
  group->total_height = new_max_y - new_min_y;

  // Validate that the expanded group still fits within grid bounds
  if (group->min_grid_x + group->total_width > grid_layout_.columns ||
      group->min_grid_y + group->total_height > grid_layout_.rows) {
    // Group would exceed grid bounds, remove the panel and return false
    group->panel_ids.pop_back();
    panel_to_group_map_.erase(panel_id);
    
    // Recalculate bounds without the added panel
    if (!group->panel_ids.empty()) {
      int min_x = INT_MAX, min_y = INT_MAX;
      int max_x = -1, max_y = -1;

      for (uint32_t id : group->panel_ids) {
        const auto& config = panels_[id]->get_config();
        min_x = std::min(min_x, config.grid_x);
        min_y = std::min(min_y, config.grid_y);
        max_x = std::max(max_x, config.grid_x + config.grid_width);
        max_y = std::max(max_y, config.grid_y + config.grid_height);
      }

      group->min_grid_x = min_x;
      group->min_grid_y = min_y;
      group->total_width = max_x - min_x;
      group->total_height = max_y - min_y;
    } else {
      // Group is now empty, could remove it, but we'll leave it for now
    }
    
    return false;
  }

  return true;
}

bool PanelManager::remove_panel_from_group(uint32_t group_id, uint32_t panel_id) {
  // Check if group exists
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false;
  }

  // Check if panel is in this group
  auto panel_group_it = panel_to_group_map_.find(panel_id);
  if (panel_group_it == panel_to_group_map_.end() || panel_group_it->second != group_id) {
    return false;
  }

  // Remove panel from group
  auto& group = group_it->second;
  auto& panel_list = group->panel_ids;
  
  panel_list.erase(
    std::remove(panel_list.begin(), panel_list.end(), panel_id),
    panel_list.end()
  );

  // Remove from mapping
  panel_to_group_map_.erase(panel_id);

  // If group is empty, remove it
  if (panel_list.empty()) {
    panel_groups_.erase(group_id);
  } else {
    // Recalculate group bounds
    if (!panel_list.empty()) {
      int min_x = INT_MAX, min_y = INT_MAX;
      int max_x = -1, max_y = -1;

      for (uint32_t id : panel_list) {
        const auto& config = panels_[id]->get_config();
        min_x = std::min(min_x, config.grid_x);
        min_y = std::min(min_y, config.grid_y);
        max_x = std::max(max_x, config.grid_x + config.grid_width);
        max_y = std::max(max_y, config.grid_y + config.grid_height);
      }

      group->min_grid_x = min_x;
      group->min_grid_y = min_y;
      group->total_width = max_x - min_x;
      group->total_height = max_y - min_y;
    }
  }

  return true;
}

bool PanelManager::destroy_panel_group(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false;
  }

  // Remove all panels from the group mapping
  for (uint32_t panel_id : group_it->second->panel_ids) {
    panel_to_group_map_.erase(panel_id);
  }

  // Remove the group
  panel_groups_.erase(group_it);

  return true;
}

bool PanelManager::is_panel_in_group(uint32_t panel_id) const {
  return panel_to_group_map_.find(panel_id) != panel_to_group_map_.end();
}

uint32_t PanelManager::get_panel_group_id(uint32_t panel_id) const {
  auto it = panel_to_group_map_.find(panel_id);
  if (it != panel_to_group_map_.end()) {
    return it->second;
  }
  return 0; // Return 0 if panel is not in a group
}

PanelManager::PanelGroup* PanelManager::get_panel_group(uint32_t group_id) {
  auto it = panel_groups_.find(group_id);
  if (it != panel_groups_.end()) {
    return it->second.get();
  }
  return nullptr;
}

bool PanelManager::can_place_group_at(uint32_t group_id, int grid_x, int grid_y) const {
  const auto* group = get_panel_group(group_id);
  if (!group) {
    return false;
  }

  // If the group doesn't prevent overlap, allow placement anywhere within bounds
  if (!group->prevent_overlap) {
    int group_right = grid_x + group->total_width;
    int group_bottom = grid_y + group->total_height;

    // Just check if the group fits within grid bounds
    return (group_right <= grid_layout_.columns && group_bottom <= grid_layout_.rows);
  }

  // Check if the group fits within grid bounds at the specified position
  int group_right = grid_x + group->total_width;
  int group_bottom = grid_y + group->total_height;

  if (group_right > grid_layout_.columns || group_bottom > grid_layout_.rows) {
    return false;
  }

  // Check for collisions with other panels that are not in this group
  for (int x = grid_x; x < group_right; x++) {
    for (int y = grid_y; y < group_bottom; y++) {
      // Check if this grid cell is occupied by a panel not in this group
      for (const auto& [id, panel] : panels_) {
        const auto& config = panel->get_config();

        // Skip panels in the same group
        if (get_panel_group_id(id) == group_id) {
          continue;
        }

        // Check if this panel occupies the grid cell
        int panel_right = config.grid_x + config.grid_width;
        int panel_bottom = config.grid_y + config.grid_height;

        if (x >= config.grid_x && x < panel_right &&
            y >= config.grid_y && y < panel_bottom) {
          // Collision detected
          return false;
        }
      }
    }
  }

  return true;
}

const PanelManager::PanelGroup* PanelManager::get_panel_group(uint32_t group_id) const {
  auto it = panel_groups_.find(group_id);
  if (it != panel_groups_.end()) {
    return it->second.get();
  }
  return nullptr;
}

uint32_t PanelManager::create_super_panel_from_adjacent(uint32_t panel1_id, uint32_t panel2_id) {
  // Check if both panels exist
  if (panels_.find(panel1_id) == panels_.end() || 
      panels_.find(panel2_id) == panels_.end()) {
    return 0;
  }

  // Check if panels are already in a group
  if (is_panel_in_group(panel1_id) || is_panel_in_group(panel2_id)) {
    return 0; // Cannot create super-panel from panels already in groups
  }

  // Check if panels are adjacent (share a common edge)
  const auto& config1 = panels_[panel1_id]->get_config();
  const auto& config2 = panels_[panel2_id]->get_config();

  bool adjacent = false;

  // Check horizontal adjacency (same row, touching sides)
  if (config1.grid_y == config2.grid_y) {
    if (config1.grid_x + config1.grid_width == config2.grid_x || 
        config2.grid_x + config2.grid_width == config1.grid_x) {
      adjacent = true;
    }
  }
  // Check vertical adjacency (same column, touching sides)
  else if (config1.grid_x == config2.grid_x) {
    if (config1.grid_y + config1.grid_height == config2.grid_y || 
        config2.grid_y + config2.grid_height == config1.grid_y) {
      adjacent = true;
    }
  }

  if (!adjacent) {
    return 0; // Panels are not adjacent
  }

  // Create a group with both panels
  std::vector<uint32_t> panel_ids = {panel1_id, panel2_id};
  return create_panel_group(panel_ids);
}

uint32_t PanelManager::create_super_panel_from_rectangular_region(int start_x, int start_y, int width, int height) {
  // Validate region bounds
  if (start_x < 0 || start_y < 0 || 
      start_x + width > grid_layout_.columns || 
      start_y + height > grid_layout_.rows) {
    return 0; // Region is out of bounds
  }

  // Find all panels that are completely within the specified region
  std::vector<uint32_t> panel_ids;
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();
    
    // Check if panel is completely within the region
    if (config.grid_x >= start_x && 
        config.grid_y >= start_y && 
        config.grid_x + config.grid_width <= start_x + width && 
        config.grid_y + config.grid_height <= start_y + height) {
      
      // Check if panel is not already in a group
      if (!is_panel_in_group(id)) {
        panel_ids.push_back(id);
      }
    }
  }

  if (panel_ids.empty()) {
    return 0; // No panels found in the region
  }

  // Create a group with all panels in the region
  uint32_t group_id = create_panel_group(panel_ids);
  
  // Set the group's position and size to match the specified region
  if (auto* group = get_panel_group(group_id)) {
    group->min_grid_x = start_x;
    group->min_grid_y = start_y;
    group->total_width = width;
    group->total_height = height;
  }

  return group_id;
}

void PanelManager::lock_panel_group(uint32_t group_id, bool locked) {
  auto* group = get_panel_group(group_id);
  if (group) {
    group->locked = locked;
  }
}

void PanelManager::set_prevent_overlap_for_group(uint32_t group_id, bool prevent) {
  auto* group = get_panel_group(group_id);
  if (group) {
    group->prevent_overlap = prevent;
  }
}

bool PanelManager::does_group_prevent_overlap(uint32_t group_id) const {
  const auto* group = get_panel_group(group_id);
  return group ? group->prevent_overlap : false;
}

bool PanelManager::is_panel_group_locked(uint32_t group_id) const {
  const auto* group = get_panel_group(group_id);
  return group ? group->locked : false;
}

uint32_t PanelManager::create_tabbed_group(uint32_t target_panel_id) {
  // Check if the target panel exists
  if (panels_.find(target_panel_id) == panels_.end()) {
    return 0;
  }

  // Create a new tabbed panel to contain the target panel
  PanelConfig config = get_panel_config(target_panel_id);
  config.type = PanelType::TABBED_GROUP;
  config.title = "Tabbed Group";
  
  uint32_t tabbed_group_id = add_panel(PanelType::TABBED_GROUP, "Tabbed Group", 
                                       config.grid_x, config.grid_y, 
                                       config.grid_width, config.grid_height);

  if (tabbed_group_id != 0) {
    // Move the target panel into the tabbed group
    if (auto* tabbed_panel = dynamic_cast<TabbedPanel*>(get_panel_by_id(tabbed_group_id))) {
      // Add the target panel to the tabbed group
      if (tabbed_panel->add_panel_to_tab(target_panel_id)) {
        // Update the target panel's position to match the tabbed group
        auto& target_config = panels_[target_panel_id]->get_config();
        target_config.position = config.position;
        target_config.size = ImVec2(config.size.x, config.size.y - 30); // Account for tab bar height
        
        return tabbed_group_id;
      } else {
        // If adding to tab failed, remove the tabbed panel we just created
        remove_panel(tabbed_group_id);
        return 0;
      }
    }
  }

  return 0;
}

bool PanelManager::add_panel_to_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_add_id) {
  // Check if both panels exist
  if (panels_.find(tabbed_group_id) == panels_.end() || 
      panels_.find(panel_to_add_id) == panels_.end()) {
    return false;
  }

  // Check if the target panel is actually a tabbed panel
  auto* tabbed_panel = dynamic_cast<TabbedPanel*>(get_panel_by_id(tabbed_group_id));
  if (!tabbed_panel) {
    return false;
  }

  // Add the panel to the tabbed group
  return tabbed_panel->add_panel_to_tab(panel_to_add_id);
}

bool PanelManager::remove_panel_from_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_remove_id) {
  // Check if both panels exist
  if (panels_.find(tabbed_group_id) == panels_.end() || 
      panels_.find(panel_to_remove_id) == panels_.end()) {
    return false;
  }

  // Check if the target panel is actually a tabbed panel
  auto* tabbed_panel = dynamic_cast<TabbedPanel*>(get_panel_by_id(tabbed_group_id));
  if (!tabbed_panel) {
    return false;
  }

  // Remove the panel from the tabbed group
  return tabbed_panel->remove_panel_from_tab(panel_to_remove_id);
}

bool PanelManager::is_panel_in_tabbed_group(uint32_t panel_id) const {
  // Check if the panel exists
  if (panels_.find(panel_id) == panels_.end()) {
    return false;
  }

  // Check if the panel is contained within a tabbed panel
  for (const auto& [id, panel] : panels_) {
    if (panel->get_config().type == PanelType::TABBED_GROUP) {
      if (auto* tabbed_panel = dynamic_cast<TabbedPanel*>(panel.get())) {
        const auto& tabbed_panels = tabbed_panel->get_tabbed_panels();
        if (std::find(tabbed_panels.begin(), tabbed_panels.end(), panel_id) != tabbed_panels.end()) {
          return true;
        }
      }
    }
  }

  return false;
}

uint32_t PanelManager::get_containing_tabbed_group_id(uint32_t panel_id) const {
  // Check if the panel exists
  if (panels_.find(panel_id) == panels_.end()) {
    return 0;
  }

  // Find which tabbed panel contains this panel
  for (const auto& [id, panel] : panels_) {
    if (panel->get_config().type == PanelType::TABBED_GROUP) {
      if (auto* tabbed_panel = dynamic_cast<TabbedPanel*>(panel.get())) {
        const auto& tabbed_panels = tabbed_panel->get_tabbed_panels();
        if (std::find(tabbed_panels.begin(), tabbed_panels.end(), panel_id) != tabbed_panels.end()) {
          return id;
        }
      }
    }
  }

  return 0;
}

bool PanelManager::can_drag_panel_to_target(uint32_t source_panel_id, uint32_t target_panel_id) const {
  // Check if both panels exist
  if (panels_.find(source_panel_id) == panels_.end() ||
      panels_.find(target_panel_id) == panels_.end()) {
    return false;
  }

  // Prevent dragging a panel onto itself
  if (source_panel_id == target_panel_id) {
    return false;
  }

  // Prevent dragging a panel that's already in a tabbed group (to prevent nested tabs)
  if (is_panel_in_tabbed_group(source_panel_id)) {
    return false;
  }

  // Allow dragging to an existing tabbed group
  if (panels_.at(target_panel_id)->get_config().type == PanelType::TABBED_GROUP) {
    return true;
  }

  // Allow dragging to a panel that's already in a tabbed group
  if (is_panel_in_tabbed_group(target_panel_id)) {
    return true;
  }

  // Otherwise, we can create a new tabbed group with these two panels
  return true;
}

std::pair<int, int> PanelManager::find_auto_dock_position(int width, int height) const {
  // If no panels exist, return (0, 0) as the default position
  if (panels_.empty()) {
    return {0, 0};
  }

  // Define the grid boundaries
  int max_cols = grid_layout_.columns;
  int max_rows = grid_layout_.rows;

  // Create a 2D grid to represent occupied cells
  std::vector<std::vector<bool>> occupied(max_rows, std::vector<bool>(max_cols, false));

  // Mark cells occupied by existing panels
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    // Mark the grid cells occupied by this panel
    for (int x = config.grid_x; x < config.grid_x + config.grid_width && x < max_cols; ++x) {
      for (int y = config.grid_y; y < config.grid_y + config.grid_height && y < max_rows; ++y) {
        if (x >= 0 && y >= 0) {  // Ensure valid indices
          occupied[y][x] = true;
        }
      }
    }
  }

  // Priority order for docking: Right, Below, Left, Above (most intuitive for users)
  // Store potential positions with priority scores
  std::vector<std::tuple<int, int, int>> potential_positions; // x, y, score

  // Look for adjacent empty spaces to existing panels in priority order

  // 1. Try placing to the right of existing panels (priority 1 - highest)
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    int right_edge = config.grid_x + config.grid_width;
    if (right_edge + width <= max_cols) {
      bool can_place = true;
      for (int x = right_edge; x < right_edge + width; ++x) {
        for (int y = config.grid_y; y < config.grid_y + std::min(height, config.grid_height); ++y) {
          if (y >= 0 && y < max_rows && occupied[y][x]) {
            can_place = false;
            break;
          }
        }
        if (!can_place) break;
      }

      if (can_place) {
        // Check if the entire panel can fit vertically
        bool full_fit = true;
        for (int x = right_edge; x < right_edge + width; ++x) {
          for (int y = config.grid_y; y < config.grid_y + height; ++y) {
            if (y >= max_rows || (y >= 0 && occupied[y][x])) {
              full_fit = false;
              break;
            }
          }
          if (!full_fit) break;
        }

        if (full_fit) {
          // Score based on adjacency (higher is better) and position (prefer top-left areas)
          int score = 1000 - (right_edge + config.grid_y * 10); // Prefer positions closer to top-left
          potential_positions.push_back({right_edge, config.grid_y, score});
        }
      }
    }
  }

  // 2. Try placing below existing panels (priority 2)
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    int bottom_edge = config.grid_y + config.grid_height;
    if (bottom_edge + height <= max_rows) {
      bool can_place = true;
      for (int y = bottom_edge; y < bottom_edge + height; ++y) {
        for (int x = config.grid_x; x < config.grid_x + std::min(width, config.grid_width); ++x) {
          if (x >= 0 && x < max_cols && occupied[y][x]) {
            can_place = false;
            break;
          }
        }
        if (!can_place) break;
      }

      if (can_place) {
        // Check if the entire panel can fit horizontally
        bool full_fit = true;
        for (int y = bottom_edge; y < bottom_edge + height; ++y) {
          for (int x = config.grid_x; x < config.grid_x + width; ++x) {
            if (x >= max_cols || (x >= 0 && occupied[y][x])) {
              full_fit = false;
              break;
            }
          }
          if (!full_fit) break;
        }

        if (full_fit) {
          // Score based on adjacency (medium-high) and position (prefer top-left areas)
          int score = 800 - (config.grid_x + bottom_edge * 10); // Prefer positions closer to top-left
          potential_positions.push_back({config.grid_x, bottom_edge, score});
        }
      }
    }
  }

  // 3. Try placing to the left of existing panels (priority 3)
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    int left_edge = config.grid_x - width;
    if (left_edge >= 0) {
      bool can_place = true;
      for (int x = left_edge; x < config.grid_x; ++x) {
        for (int y = config.grid_y; y < config.grid_y + std::min(height, config.grid_height); ++y) {
          if (y >= 0 && y < max_rows && occupied[y][x]) {
            can_place = false;
            break;
          }
        }
        if (!can_place) break;
      }

      if (can_place) {
        // Check if the entire panel can fit vertically
        bool full_fit = true;
        for (int x = left_edge; x < config.grid_x; ++x) {
          for (int y = config.grid_y; y < config.grid_y + height; ++y) {
            if (y >= max_rows || (y >= 0 && occupied[y][x])) {
              full_fit = false;
              break;
            }
          }
          if (!full_fit) break;
        }

        if (full_fit) {
          // Score based on adjacency (medium) and position (prefer top-left areas)
          int score = 600 - (left_edge + config.grid_y * 10); // Prefer positions closer to top-left
          potential_positions.push_back({left_edge, config.grid_y, score});
        }
      }
    }
  }

  // 4. Try placing above existing panels (priority 4 - lowest among adjacent)
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    int top_edge = config.grid_y - height;
    if (top_edge >= 0) {
      bool can_place = true;
      for (int y = top_edge; y < config.grid_y; ++y) {
        for (int x = config.grid_x; x < config.grid_x + std::min(width, config.grid_width); ++x) {
          if (x >= 0 && x < max_cols && occupied[y][x]) {
            can_place = false;
            break;
          }
        }
        if (!can_place) break;
      }

      if (can_place) {
        // Check if the entire panel can fit horizontally
        bool full_fit = true;
        for (int y = top_edge; y < config.grid_y; ++y) {
          for (int x = config.grid_x; x < config.grid_x + width; ++x) {
            if (x >= max_cols || (x >= 0 && occupied[y][x])) {
              full_fit = false;
              break;
            }
          }
          if (!full_fit) break;
        }

        if (full_fit) {
          // Score based on adjacency (lower) and position (prefer top-left areas)
          int score = 400 - (config.grid_x + top_edge * 10); // Prefer positions closer to top-left
          potential_positions.push_back({config.grid_x, top_edge, score});
        }
      }
    }
  }

  // If we found any adjacent positions, return the one with the highest score
  if (!potential_positions.empty()) {
    // Find the position with the highest score
    auto best_pos = std::max_element(potential_positions.begin(), potential_positions.end(),
                                     [](const auto& a, const auto& b) {
                                       return std::get<2>(a) < std::get<2>(b);
                                     });
    return {std::get<0>(*best_pos), std::get<1>(*best_pos)};
  }

  // If no adjacent position found, try to find any empty space in the grid
  // Start from top-left and scan row by row for the first available spot
  for (int y = 0; y < max_rows; ++y) {
    for (int x = 0; x < max_cols; ++x) {
      // Check if we can place the panel at this position
      if (x + width <= max_cols && y + height <= max_rows) {
        bool can_place = true;
        for (int dy = 0; dy < height; ++dy) {
          for (int dx = 0; dx < width; ++dx) {
            if (occupied[y + dy][x + dx]) {
              can_place = false;
              break;
            }
          }
          if (!can_place) break;
        }

        if (can_place) {
          return {x, y};
        }
      }
    }
  }

  // If no space found, return (-1, -1) indicating failure
  return {-1, -1};
}

void PanelManager::handle_panel_drag_drop() {
  // Check if we're currently dragging a panel
  if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
    // Check if any panel window is being dragged
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
      // Find which panel is being dragged by checking if the mouse is over any panel window
      for (auto& [panel_id, panel] : panels_) {
        // Skip if panel is not visible or is in a tabbed group already
        if (!panel->is_visible() || is_panel_in_tabbed_group(panel_id)) {
          continue;
        }

        // Get the panel's window rectangle
        ImVec2 panel_pos = panel->get_config().position;
        ImVec2 panel_size = panel->get_config().size;

        // Check if mouse is over this panel
        ImVec2 mouse_pos = ImGui::GetMousePos();
        if (mouse_pos.x >= panel_pos.x && mouse_pos.x <= panel_pos.x + panel_size.x &&
            mouse_pos.y >= panel_pos.y && mouse_pos.y <= panel_pos.y + panel_size.y) {

            // Check if this is the title bar area (top portion of the window)
            // Usually the title bar height is around 20-30 pixels
            if (mouse_pos.y <= panel_pos.y + 30) {  // Approximate title bar height
                if (!is_dragging_) {
                    dragged_panel_id_ = panel_id;
                    is_dragging_ = true;

                    // Print debug info
                    printf("Started dragging panel ID: %u\n", panel_id);
                }
                break;
            }
        }
      }
    }
  } else {
    // Mouse button is released - end drag operation
    if (is_dragging_ && dragged_panel_id_ != 0) {
      // Check if we're dropping onto another panel
      if (drag_target_panel_id_ != 0) {
        // Attempt to create a tabbed group or add to existing tabbed group
        if (panels_.at(drag_target_panel_id_)->get_config().type == PanelType::TABBED_GROUP) {
          // Add the dragged panel to the existing tabbed group
          add_panel_to_tabbed_group(drag_target_panel_id_, dragged_panel_id_);
        } else {
          // Check if target panel is already in a tabbed group
          uint32_t existing_tabbed_group_id = get_containing_tabbed_group_id(drag_target_panel_id_);
          if (existing_tabbed_group_id != 0) {
            // Add the dragged panel to the existing tabbed group that contains the target
            add_panel_to_tabbed_group(existing_tabbed_group_id, dragged_panel_id_);
          } else {
            // Create a new tabbed group with both panels
            uint32_t tabbed_group_id = create_tabbed_group(drag_target_panel_id_);
            if (tabbed_group_id != 0) {
              // Add the originally dragged panel to the new tabbed group
              add_panel_to_tabbed_group(tabbed_group_id, dragged_panel_id_);
            }
          }
        }
      }

      // Reset drag state
      is_dragging_ = false;
      dragged_panel_id_ = 0;
      drag_target_panel_id_ = 0;
    } else {
      // Reset drag state even if no drop occurred
      is_dragging_ = false;
      dragged_panel_id_ = 0;
      drag_target_panel_id_ = 0;
    }
  }

  // During drag, check for potential drop targets
  if (is_dragging_ && dragged_panel_id_ != 0) {
    ImVec2 mouse_pos = ImGui::GetMousePos();
    drag_target_panel_id_ = 0;  // Reset target

    // Find which panel we might be dropping onto
    for (auto& [panel_id, panel] : panels_) {
      // Skip if panel is not visible, is the dragged panel, or is in a tabbed group
      if (!panel->is_visible() || panel_id == dragged_panel_id_) {
        continue;
      }

      // Get the panel's window rectangle
      ImVec2 panel_pos = panel->get_config().position;
      ImVec2 panel_size = panel->get_config().size;

      // Check if mouse is over this panel
      if (mouse_pos.x >= panel_pos.x && mouse_pos.x <= panel_pos.x + panel_size.x &&
          mouse_pos.y >= panel_pos.y && mouse_pos.y <= panel_pos.y + panel_size.y) {

          // Check if this panel supports being a drop target
          if (can_drag_panel_to_target(dragged_panel_id_, panel_id)) {
              drag_target_panel_id_ = panel_id;

              // Enhanced visual feedback for drop target
              ImDrawList* draw_list = ImGui::GetForegroundDrawList();
              
              // Draw a thick border around the target panel
              ImVec2 p_min = ImVec2(panel_pos.x, panel_pos.y);
              ImVec2 p_max = ImVec2(panel_pos.x + panel_size.x, panel_pos.y + panel_size.y);
              draw_list->AddRect(p_min, p_max, IM_COL32(0, 255, 0, 255), 0.0f, 0, 6.0f); // Thicker green border
              
              // Draw a semi-transparent overlay to highlight the area
              draw_list->AddRectFilled(p_min, p_max, IM_COL32(0, 255, 0, 50)); // Semi-transparent green fill
              
              // Draw a label indicating this is a valid drop zone
              std::string label = "Drop to create tabbed group";
              ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
              ImVec2 label_pos = ImVec2(
                  panel_pos.x + (panel_size.x - label_size.x) * 0.5f,
                  panel_pos.y + (panel_size.y - label_size.y) * 0.5f
              );
              draw_list->AddText(label_pos, IM_COL32(255, 255, 255, 255), label.c_str());

              break;
          }
      }
    }
  }
}

}  // namespace BTQuant