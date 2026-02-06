#include "../../include/components/panel_manager.hpp"

#include <fstream>
#include <iostream>
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
#include "../../include/components/tabbed_panel.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/performance/panel_profiler.hpp"
#include "../../include/ui/layout_manager.hpp"

using json = nlohmann::json;

namespace BTQuant {

PanelManager::PanelManager(std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                           std::shared_ptr<OrderManager> order_manager,
                           std::shared_ptr<PositionManager> position_manager,
                           std::shared_ptr<RiskAssessment> risk_assessment,
                           RenderEngine::MarketMicrostructureRenderer* micro_renderer)
    : processor_(processor),
      order_manager_(order_manager),
      position_manager_(position_manager),
      risk_assessment_(risk_assessment),
      micro_renderer_(micro_renderer),
      next_group_id_(1) {
  chart_manager_ = std::make_unique<ChartManager>(processor);
  context_menu_manager_ = std::make_unique<ContextMenuManager>(this);
  strategy_builder_ = std::make_unique<RenderEngine::StrategyBuilder>(PanelConfig{.title = "Strategy Builder", .type = PanelType::STRATEGY_BUILDER});
  
  // Set up the callback to mark visualization panels as dirty when cluster engine processes a trade
  if (micro_renderer_) {
    micro_renderer_->set_on_cluster_engine_trade_callback([this]() {
      this->mark_visualization_panels_dirty();
    });
    
    // Also set up the cluster engine panel manager connection for direct dirty callback
    micro_renderer_->set_cluster_engine_panel_manager(this);
  }
}

PanelManager::~PanelManager() {
  context_menu_manager_.reset(); // Explicitly reset context menu manager before other members
  strategy_builder_.reset(); // Explicitly reset strategy builder before other members
  panels_.clear();
  panel_groups_.clear();
  panel_to_group_map_.clear();
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

  // Initialize with default symbol
  set_active_symbol(1, "BTC-USDT"); // Use a default symbol ID and name
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

  // Process drag-and-drop for visible panels
  process_panel_drag_and_drop(panels_with_ids, panel_to_id);

  // Render only the visible panels
  for (const auto* panel : visible_panels) {
    uint32_t panel_id = panel_to_id.at(panel);  // Safe lookup with at()

    // Start timing the panel render
    BTQuant::g_panel_profiler.start_panel_render(panel_id, panel->get_title());

    // Cast back to non-const pointer to call render (since render() is non-const)
    const_cast<BTQuant::PanelBase*>(panel)->render();

    // End timing the panel render
    BTQuant::g_panel_profiler.end_panel_render(panel_id);
  }
}

uint32_t PanelManager::add_panel(PanelType type, const std::string& title, int grid_x, int grid_y,
                                 int width, int height) {
  uint32_t panel_id = next_panel_id_++;

  // If default positions are used (-1, -1), find the best docking position
  if (grid_x == -1 && grid_y == -1) {
    std::tie(grid_x, grid_y) = find_best_docking_position(width, height);
  }

  PanelConfig config = create_panel_config(type, title, grid_x, grid_y, width, height);

  std::unique_ptr<PanelBase> panel;
  switch (type) {
    case PanelType::CHART:
      panel = std::make_unique<ChartPanel>(config, processor_, chart_manager_.get(), this);

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
      panel = std::make_unique<TimeAndSalesPanel>(config, processor_);
      break;
    }
    case PanelType::HISTORICAL_TIME_SALES: {
      panel = std::make_unique<HistoricalTimeSalesPanel>(config, processor_);
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
      panel = std::make_unique<OrderbookPanel>(config, processor_);
      break;
    case PanelType::PERFORMANCE_MONITOR:
      panel = std::make_unique<PerformanceMonitorPanel>(config);
      break;
    case PanelType::STATUS_BAR:
      panel = std::make_unique<StatusBarPanel>(config, processor_);
      break;
    case PanelType::WATCHLIST: {
      auto watchlist = std::make_unique<WatchlistPanel>(config, processor_);
      watchlist->set_symbol_selected_callback(
          [this](uint32_t symbol_id, const std::string& symbol_name) {
            this->set_active_symbol(symbol_id, symbol_name);
          });
      panel = std::move(watchlist);
      break;
    }
    case PanelType::TAPE:
      panel = std::make_unique<TapePanel>(config, processor_);
      break;
    case PanelType::VOLUME_PROFILE:
      panel = std::make_unique<VolumeProfilePanel>(config, processor_);
      break;
    case PanelType::DEPTH_CHART:
      panel = std::make_unique<DepthChartPanel>(config, processor_);
      break;
    case PanelType::FOOTPRINT_CHART:
      panel = std::make_unique<FootprintPanel>(config, micro_renderer_);
      break;
    case PanelType::TPO_PROFILE:
      panel = std::make_unique<TpoPanel>(config, micro_renderer_);
      break;
    case PanelType::OPTION_ANALYTICS:
      panel = std::make_unique<BTQuant::RenderEngine::OptionAnalyticsPanel>(strategy_builder_.get());
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
      panel = std::make_unique<ChartReplayPanel>(config, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::TABBED_PANEL:
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

  // If default positions are used (-1, -1), find the best docking position
  if (grid_x == -1 && grid_y == -1) {
    std::tie(grid_x, grid_y) = find_best_docking_position(width, height);
  }

  PanelConfig config =
      create_panel_config_with_symbol(type, title, symbol, grid_x, grid_y, width, height);

  std::unique_ptr<PanelBase> panel;
  switch (type) {
    case PanelType::CHART:
      panel = std::make_unique<ChartPanel>(config, processor_, chart_manager_.get(), this);

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
      panel = std::make_unique<TimeAndSalesPanel>(config, processor_);
      break;
    }
    case PanelType::HISTORICAL_TIME_SALES: {
      panel = std::make_unique<HistoricalTimeSalesPanel>(config, processor_);
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
      panel = std::make_unique<OrderbookPanel>(config, processor_);
      break;
    case PanelType::PERFORMANCE_MONITOR:
      panel = std::make_unique<PerformanceMonitorPanel>(config);
      break;
    case PanelType::STATUS_BAR:
      panel = std::make_unique<StatusBarPanel>(config, processor_);
      break;
    case PanelType::WATCHLIST: {
      auto watchlist = std::make_unique<WatchlistPanel>(config, processor_);
      watchlist->set_symbol_selected_callback(
          [this](uint32_t symbol_id, const std::string& symbol_name) {
            this->set_active_symbol(symbol_id, symbol_name);
          });
      panel = std::move(watchlist);
      break;
    }
    case PanelType::TAPE:
      panel = std::make_unique<TapePanel>(config, processor_);
      break;
    case PanelType::VOLUME_PROFILE:
      panel = std::make_unique<VolumeProfilePanel>(config, processor_);
      break;
    case PanelType::DEPTH_CHART:
      panel = std::make_unique<DepthChartPanel>(config, processor_);
      break;
    case PanelType::FOOTPRINT_CHART:
      panel = std::make_unique<FootprintPanel>(config, micro_renderer_);
      break;
    case PanelType::TPO_PROFILE:
      panel = std::make_unique<TpoPanel>(config, micro_renderer_);
      break;
    case PanelType::OPTION_ANALYTICS:
      panel = std::make_unique<BTQuant::RenderEngine::OptionAnalyticsPanel>(strategy_builder_.get());
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
      panel = std::make_unique<ChartReplayPanel>(config, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::TABBED_PANEL:
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
      // Remove panel from its group
      uint32_t group_id = group_it->second;
      auto panel_group_it = panel_groups_.find(group_id);
      if (panel_group_it != panel_groups_.end()) {
        PanelGroup& group = panel_group_it->second;

        // Check if group is locked
        if (!group.locked) {
          // Remove panel from the group
          group.panel_ids.erase(panel_id);
          panel_to_group_map_.erase(group_it);

          // If the group becomes empty, consider removing it
          if (group.panel_ids.empty()) {
            panel_groups_.erase(panel_group_it);
          } else if (group.super_panel) {
            // For super-panels, update the layout when a panel is removed
            update_group_position(group_id);
          }
        }
      }
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
  panel_to_group_map_.clear();
  panel_groups_.clear();
  next_panel_id_ = 1;
  next_group_id_ = 1;
}

void PanelManager::move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    // Check if the panel is part of a group
    auto group_it = panel_to_group_map_.find(panel_id);
    if (group_it != panel_to_group_map_.end()) {
      // Panel is part of a group, move the entire group
      uint32_t group_id = group_it->second;
      auto panel_group_it = panel_groups_.find(group_id);
      if (panel_group_it != panel_groups_.end()) {
        PanelGroup& group = panel_group_it->second;

        // Check if group is locked
        if (group.locked) {
          return; // Cannot move a locked group or any panel within it
        }

        // Update the group's position
        group.grid_x = new_grid_x;
        group.grid_y = new_grid_y;

        // Update all panels in the group
        update_group_position(group_id);
      }
    } else {
      // Panel is not part of a group, move individually
      auto& config = it->second->get_config();
      config.grid_x = new_grid_x;
      config.grid_y = new_grid_y;
      config.position = calculate_panel_position(new_grid_x, new_grid_y);
    }
  }
}

void PanelManager::resize_panel(uint32_t panel_id, int new_width, int new_height) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    // Check if the panel is part of a group
    auto group_it = panel_to_group_map_.find(panel_id);
    if (group_it != panel_to_group_map_.end()) {
      // Panel is part of a group, resize the entire group
      uint32_t group_id = group_it->second;
      auto panel_group_it = panel_groups_.find(group_id);
      if (panel_group_it != panel_groups_.end()) {
        PanelGroup& group = panel_group_it->second;

        // Check if group is locked
        if (group.locked) {
          return; // Cannot resize a locked group or any panel within it
        }

        // Update the group's size
        group.grid_width = new_width;
        group.grid_height = new_height;

        // Update all panels in the group
        update_group_size(group_id);
      }
    } else {
      // Panel is not part of a group, resize individually
      auto& config = it->second->get_config();
      config.grid_width = new_width;
      config.grid_height = new_height;
      config.size = calculate_panel_size(new_width, new_height);
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

void PanelManager::reset_to_factory_layout() {
  // Clear all panels and groups
  panels_.clear();
  panel_groups_.clear();
  panel_to_group_map_.clear();

  // Reset ID counters
  next_panel_id_ = 1;
  next_group_id_ = 1;

  // Set grid layout (3 columns, 5 rows to fit 2x2 chart properly)
  set_grid_layout(3, 5);

  // Create default factory panels with specific positions to avoid overlaps
  // Row 0: Status Bar and Alerts
  add_panel(PanelType::STATUS_BAR, "Status Bar", 0, 0, 2, 1);
  add_panel(PanelType::ALERTS, "Alerts", 2, 0, 1, 1);

  // Row 1-2: Main Chart (2x2) and Depth Chart (1x2)
  add_panel(PanelType::CHART, "BTC-USDT Chart", 0, 1, 2, 2);
  add_panel(PanelType::DEPTH_CHART, "Depth Chart", 2, 1, 1, 2);

  // Row 3: Orderbook Ladder (2x1) and Tape (1x1) - corrected to avoid overlap
  add_panel(PanelType::ORDERBOOK, "BTC-USDT Orderbook", 0, 3, 2, 1);
  add_panel(PanelType::TAPE, "Time & Sales", 2, 3, 1, 1);

  // Row 4: DOM Surface (2x1) and Watchlist (1x1)
  add_panel(PanelType::HEATMAP, "DOM Surface", 0, 4, 2, 1);
  add_panel(PanelType::WATCHLIST, "Watchlist", 2, 4, 1, 1);

  // Initialize with default symbol
  set_active_symbol(1, "BTC-USDT"); // Use a default symbol ID and name
  
  // Reset the layout manager to ensure clean state for any layout-related settings
  auto& layoutManager = BTQuant::UI::LayoutManager::getInstance();
  layoutManager.set_active_quick_slot(0); // Clear any active quick save slot
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
          }
          if (symbol_id != 0) {
            // Get exchange name from symbol registry
            std::string exchange = "Unknown";
            auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
            if (symbol_info.has_value()) {
              exchange = symbol_info->exchange;
            }
            watchlist->add_symbol(symbol_id, symbol, exchange);
          }
        }
        break;
      }
      case PanelType::TAPE: {
        if (auto* tape = dynamic_cast<TapePanel*>(it->second.get())) {
          // Find the symbol ID for the given symbol name
          uint32_t symbol_id = 0;
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
          // Use SymbolRegistry to find the symbol ID
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            symbol_id = symbol_info_opt->id;
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
    
    // Update the panel's config symbol
    it->second->get_config().symbol = symbol;
    
    // Update linked symbols in the same symbol link group
    update_linked_symbols(panel_id, symbol);
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
        settings_json["active_tab"] = tabbed_panel->get_active_tab();
        
        // Serialize the panel IDs in the tabbed panel
        json tabbed_panels_json = json::array();
        for (uint32_t panel_id : tabbed_panel->get_tabbed_panels()) {
            tabbed_panels_json.push_back(panel_id);
        }
        settings_json["tabbed_panels"] = tabbed_panels_json;
    }

    // Add settings if any were captured
    if (!settings_json.empty()) {
        panel_json["settings"] = settings_json;
    }

    // Add group information if panel is part of a group
    auto group_it = panel_to_group_map_.find(id);
    if (group_it != panel_to_group_map_.end()) {
        panel_json["group_id"] = group_it->second;
    }

    panels_json.push_back(panel_json);
  }
  layout_json["panels"] = panels_json;

  // Serialize panel groups
  json groups_json = json::array();
  for (const auto& [group_id, group] : panel_groups_) {
    json group_json;
    group_json["id"] = group_id;
    group_json["grid_x"] = group.grid_x;
    group_json["grid_y"] = group.grid_y;
    group_json["grid_width"] = group.grid_width;
    group_json["grid_height"] = group.grid_height;
    group_json["locked"] = group.locked;
    group_json["super_panel"] = group.super_panel;

    // Serialize panel IDs in the group
    json panel_ids_json = json::array();
    for (uint32_t panel_id : group.panel_ids) {
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
                    if (settings.contains("active_tab")) {
                        tabbed_panel->set_active_tab(settings["active_tab"].get<int>());
                    }
                    if (settings.contains("tabbed_panels")) {
                        auto tabbed_panels_array = settings["tabbed_panels"];
                        for (const auto& panel_id_val : tabbed_panels_array) {
                            uint32_t panel_id = panel_id_val.get<uint32_t>();
                            // Add panel to tabbed panel if it exists
                            if (panels_.find(panel_id) != panels_.end()) {
                                tabbed_panel->add_panel(panel_id);
                            }
                        }
                    }
                }
            }
        }

        set_panel_visible(id, visible);
        
        // Add to group if specified
        if (p.contains("group_id")) {
            uint32_t group_id = p["group_id"].get<uint32_t>();
            add_panel_to_group(group_id, id);
        }
      }
    }

    // Now recreate the panel groups if they exist in the layout
    if (j.contains("groups")) {
        for (const auto& g : j["groups"]) {
            uint32_t group_id = g["id"].get<uint32_t>();
            int grid_x = g["grid_x"].get<int>();
            int grid_y = g["grid_y"].get<int>();
            int grid_width = g["grid_width"].get<int>();
            int grid_height = g["grid_height"].get<int>();
            bool locked = g["locked"].get<bool>();
            bool super_panel = g.value("super_panel", false); // Default to false if not present for backward compatibility

            // Create the group
            PanelGroup group(grid_x, grid_y, grid_width, grid_height, super_panel);
            group.locked = locked;

            // Add panels to the group (these will be handled by the individual panel processing above)
            if (g.contains("panel_ids")) {
                for (const auto& panel_id_val : g["panel_ids"]) {
                    uint32_t panel_id = panel_id_val.get<uint32_t>();
                    if (panels_.find(panel_id) != panels_.end()) { // Check if panel exists
                        group.panel_ids.insert(panel_id);
                    }
                }
            }

            // Store the group
            panel_groups_[group_id] = group;

            // Update the next_group_id if needed
            if (group_id >= next_group_id_) {
                next_group_id_ = group_id + 1;
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

uint32_t PanelManager::find_panel_by_type_and_position(PanelType type, int grid_x, int grid_y) const {
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();
    if (config.type == type && config.grid_x == grid_x && config.grid_y == grid_y) {
      return id;
    }
  }
  return 0; // Not found
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
          // Get exchange name from symbol registry
          std::string exchange = "Unknown";
          auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
          if (symbol_info.has_value()) {
            exchange = symbol_info->exchange;
          }
          chart->set_symbol(symbol_name, exchange);
        }
        break;
      }
      case PanelType::WATCHLIST: {
        // Update the symbol for all watchlist panels
        if (auto* watchlist = dynamic_cast<WatchlistPanel*>(panel.get())) {
          // Get exchange name from symbol registry
          std::string exchange = "Unknown";
          auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
          if (symbol_info.has_value()) {
            exchange = symbol_info->exchange;
          }
          watchlist->add_symbol(symbol_id, symbol_name, exchange);
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
          dom->setSymbol(symbol_id);
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

uint32_t PanelManager::create_panel_group(int grid_x, int grid_y, int width, int height, bool is_super_panel) {
  uint32_t group_id = next_group_id_++;

  PanelGroup group(grid_x, grid_y, width, height, is_super_panel);
  panel_groups_[group_id] = group;

  return group_id;
}

uint32_t PanelManager::create_super_panel_group(int grid_x, int grid_y, int width, int height) {
  return create_panel_group(grid_x, grid_y, width, height, true);  // Create with super_panel = true
}

uint32_t PanelManager::bind_panels_together(const std::vector<uint32_t>& panel_ids, int grid_x, int grid_y, int width, int height) {
  // Create a new super-panel group
  uint32_t group_id = create_super_panel_group(grid_x, grid_y, width, height);
  
  // Add each panel to the group
  for (uint32_t panel_id : panel_ids) {
    add_panel_to_group(group_id, panel_id);
  }
  
  // Lock the group to prevent modifications
  lock_panel_group(group_id);
  
  return group_id;
}

bool PanelManager::add_panel_to_group(uint32_t group_id, uint32_t panel_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false; // Group doesn't exist
  }

  auto panel_it = panels_.find(panel_id);
  if (panel_it == panels_.end()) {
    return false; // Panel doesn't exist
  }

  PanelGroup& group = group_it->second;

  // Check if group is locked
  if (group.locked) {
    return false;
  }

  // Check if panel is already in another group
  auto existing_group_it = panel_to_group_map_.find(panel_id);
  if (existing_group_it != panel_to_group_map_.end()) {
    // Remove from existing group first
    uint32_t old_group_id = existing_group_it->second;
    auto& old_group = panel_groups_[old_group_id];
    old_group.panel_ids.erase(panel_id);
    panel_to_group_map_.erase(existing_group_it);
  }

  // Add panel to the new group
  group.panel_ids.insert(panel_id);
  panel_to_group_map_[panel_id] = group_id;

  // Update the panel's grid position and size based on group type
  if (group.super_panel) {
    // For super-panels, update the entire group layout
    update_group_position(group_id);
  } else {
    // For regular groups, update the panel's position to match the group
    auto& panel_config = panel_it->second->get_config();
    panel_config.grid_x = group.grid_x;
    panel_config.grid_y = group.grid_y;
    panel_config.position = calculate_panel_position(group.grid_x, group.grid_y);
  }

  return true;
}

bool PanelManager::remove_panel_from_group(uint32_t group_id, uint32_t panel_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false; // Group doesn't exist
  }
  
  PanelGroup& group = group_it->second;
  
  // Check if group is locked
  if (group.locked) {
    return false;
  }
  
  // Check if panel is in this group
  if (group.panel_ids.find(panel_id) == group.panel_ids.end()) {
    return false; // Panel is not in this group
  }
  
  // Remove panel from group
  group.panel_ids.erase(panel_id);
  panel_to_group_map_.erase(panel_id);
  
  return true;
}

bool PanelManager::lock_panel_group(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false; // Group doesn't exist
  }
  
  group_it->second.locked = true;
  return true;
}

bool PanelManager::unlock_panel_group(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false; // Group doesn't exist
  }
  
  group_it->second.locked = false;
  return true;
}

bool PanelManager::delete_panel_group(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return false; // Group doesn't exist
  }
  
  // Check if group is locked
  if (group_it->second.locked) {
    return false;
  }
  
  // Remove all panels from the group mapping
  for (uint32_t panel_id : group_it->second.panel_ids) {
    panel_to_group_map_.erase(panel_id);
  }
  
  // Remove the group
  panel_groups_.erase(group_it);
  
  return true;
}

std::vector<uint32_t> PanelManager::get_panel_groups_for_panel(uint32_t panel_id) const {
  std::vector<uint32_t> group_ids;
  
  auto it = panel_to_group_map_.find(panel_id);
  if (it != panel_to_group_map_.end()) {
    group_ids.push_back(it->second);
  }
  
  return group_ids;
}

bool PanelManager::is_panel_bound(uint32_t panel_id) const {
  return panel_to_group_map_.find(panel_id) != panel_to_group_map_.end();
}

void PanelManager::update_group_position(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return; // Group doesn't exist
  }

  const PanelGroup& group = group_it->second;

  if (group.super_panel) {
    // For super-panel, distribute panels evenly within the group's grid space
    int panel_count = static_cast<int>(group.panel_ids.size());
    if (panel_count > 0) {
      int num_cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(panel_count))));
      int num_rows = static_cast<int>(std::ceil(static_cast<double>(panel_count) / num_cols));
      
      // Ensure we don't exceed the group dimensions
      num_cols = std::min(num_cols, group.grid_width);
      num_rows = std::min(num_rows, group.grid_height);
      
      int col_width = group.grid_width / num_cols;
      int row_height = group.grid_height / num_rows;

      // Update position and size for all panels in the group
      int idx = 0;
      for (uint32_t panel_id : group.panel_ids) {
        auto panel_it = panels_.find(panel_id);
        if (panel_it != panels_.end()) {
          auto& config = panel_it->second->get_config();

          int col = idx % num_cols;
          int row = idx / num_cols;

          config.grid_x = group.grid_x + col * col_width;
          config.grid_y = group.grid_y + row * row_height;
          config.grid_width = col_width;
          config.grid_height = row_height;

          config.position = calculate_panel_position(config.grid_x, config.grid_y);
          config.size = calculate_panel_size(config.grid_width, config.grid_height);
        }
        idx++;
      }
    }
  } else {
    // For regular groups, update position for all panels in the group
    for (uint32_t panel_id : group.panel_ids) {
      auto panel_it = panels_.find(panel_id);
      if (panel_it != panels_.end()) {
        auto& config = panel_it->second->get_config();
        config.grid_x = group.grid_x;
        config.grid_y = group.grid_y;
        config.position = calculate_panel_position(group.grid_x, group.grid_y);
      }
    }
  }
}

void PanelManager::update_group_size(uint32_t group_id) {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return; // Group doesn't exist
  }

  const PanelGroup& group = group_it->second;

  // For super-panels, the size update is handled in update_group_position
  // since both position and size need to be calculated together
  if (group.super_panel) {
    update_group_position(group_id); // Handle both position and size for super-panels
  } else {
    // For regular groups, update size for all panels in the group
    int num_cols = 0;
    int num_rows = 0;

    // Determine how to distribute the space among panels (for simplicity, we'll arrange them in a grid)
    int panel_count = static_cast<int>(group.panel_ids.size());
    if (panel_count > 0) {
      num_cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(panel_count))));
      num_rows = static_cast<int>(std::ceil(static_cast<double>(panel_count) / num_cols));

      // Ensure we don't exceed the group dimensions
      num_cols = std::min(num_cols, group.grid_width);
      num_rows = std::min(num_rows, group.grid_height);
    }

    int col_width = group.grid_width / num_cols;
    int row_height = group.grid_height / num_rows;

    // Update size and position for all panels in the group
    int idx = 0;
    for (uint32_t panel_id : group.panel_ids) {
      auto panel_it = panels_.find(panel_id);
      if (panel_it != panels_.end()) {
        auto& config = panel_it->second->get_config();

        int col = idx % num_cols;
        int row = idx / num_cols;

        config.grid_x = group.grid_x + col * col_width;
        config.grid_y = group.grid_y + row * row_height;
        config.grid_width = col_width;
        config.grid_height = row_height;

        config.position = calculate_panel_position(config.grid_x, config.grid_y);
        config.size = calculate_panel_size(config.grid_width, config.grid_height);
      }
      idx++;
    }
  }
}

bool PanelManager::validate_panel_placement(uint32_t panel_id, int grid_x, int grid_y, int width, int height) const {
  // Check if the panel is part of a locked group
  auto group_it = panel_to_group_map_.find(panel_id);
  if (group_it != panel_to_group_map_.end()) {
    uint32_t group_id = group_it->second;
    auto panel_group_it = panel_groups_.find(group_id);
    if (panel_group_it != panel_groups_.end() && panel_group_it->second.locked) {
      // If panel is part of a locked group, placement must respect group constraints
      return false; // Cannot move individual panels in a locked group
    }
  }

  // Check for overlaps with other panels
  for (const auto& [id, panel] : panels_) {
    if (id == panel_id) continue; // Skip the panel we're checking

    const auto& config = panel->get_config();
    
    // Check if the new position overlaps with existing panel
    if (grid_x < config.grid_x + config.grid_width &&
        grid_x + width > config.grid_x &&
        grid_y < config.grid_y + config.grid_height &&
        grid_y + height > config.grid_y) {
      // Overlap detected
      return false;
    }
  }

  // Check if the placement is within grid bounds
  if (grid_x < 0 || grid_y < 0 || 
      grid_x + width > grid_layout_.columns || 
      grid_y + height > grid_layout_.rows) {
    return false;
  }

  return true;
}

std::vector<uint32_t> PanelManager::get_panels_in_group(uint32_t group_id) const {
  auto group_it = panel_groups_.find(group_id);
  if (group_it == panel_groups_.end()) {
    return {}; // Group doesn't exist
  }

  std::vector<uint32_t> panel_ids;
  panel_ids.reserve(group_it->second.panel_ids.size());
  for (uint32_t panel_id : group_it->second.panel_ids) {
    panel_ids.push_back(panel_id);
  }
  
  return panel_ids;
}

uint32_t PanelManager::create_tabbed_group(const std::vector<uint32_t>& panel_ids) {
  if (panel_ids.size() < 2) {
    return 0; // Need at least 2 panels to create a group
  }

  // Get the first panel to determine the position for the tabbed panel
  auto first_panel_it = panels_.find(panel_ids[0]);
  if (first_panel_it == panels_.end()) {
    return 0; // Panel doesn't exist
  }

  const auto& first_config = first_panel_it->second->get_config();

  // Create a new tabbed panel at the same position as the first panel
  uint32_t tabbed_panel_id = add_panel(PanelType::TABBED_PANEL, "Tabbed Group",
                                       first_config.grid_x, first_config.grid_y,
                                       first_config.grid_width, first_config.grid_height);

  if (tabbed_panel_id == 0) {
    return 0; // Failed to create tabbed panel
  }

  // Get the new tabbed panel
  TabbedPanel* tabbed_panel = dynamic_cast<TabbedPanel*>(get_panel_by_id(tabbed_panel_id));
  if (!tabbed_panel) {
    return 0; // Failed to cast to TabbedPanel
  }

  // Add all panels to the tabbed panel
  for (uint32_t panel_id : panel_ids) {
    auto panel_it = panels_.find(panel_id);
    if (panel_it != panels_.end()) {
      tabbed_panel->add_panel(panel_id);

      // Hide the original panel since it's now managed by the tabbed panel
      panel_it->second->set_visible(false);

      // Remove the panel from any existing groups
      auto group_ids = get_panel_groups_for_panel(panel_id);
      for (uint32_t group_id : group_ids) {
        remove_panel_from_group(group_id, panel_id);
      }
    }
  }

  return tabbed_panel_id;
}

bool PanelManager::are_panels_bound_together(const std::vector<uint32_t>& panel_ids) const {
  if (panel_ids.empty()) {
    return false;
  }

  // Get the group of the first panel
  auto first_group_it = panel_to_group_map_.find(panel_ids[0]);
  if (first_group_it == panel_to_group_map_.end()) {
    return false; // First panel is not in any group
  }

  uint32_t expected_group_id = first_group_it->second;

  // Check if all other panels are in the same group
  for (size_t i = 1; i < panel_ids.size(); ++i) {
    auto group_it = panel_to_group_map_.find(panel_ids[i]);
    if (group_it == panel_to_group_map_.end() || group_it->second != expected_group_id) {
      return false; // Panel is not in the same group
    }
  }

  // Check if the group is locked (making it a true "Super-panel")
  auto panel_group_it = panel_groups_.find(expected_group_id);
  if (panel_group_it != panel_groups_.end()) {
    return panel_group_it->second.locked;
  }

  return false;
}

void PanelManager::process_panel_drag_and_drop(
    const std::vector<std::pair<uint32_t, const BTQuant::PanelBase*>>& panels_with_ids,
    const std::unordered_map<const BTQuant::PanelBase*, uint32_t>& panel_to_id) {
  (void)panel_to_id; // Suppress unused parameter warning

  // Process drag-and-drop for all panels
  for (const auto& [source_id, source_panel] : panels_with_ids) {
    // Only process drag if the panel can be a drag source
    if (!source_panel->is_drag_source()) {
      continue;
    }

    // Make the panel a drag source
    std::string drag_source_id = "PANEL_DRAG_SOURCE_" + std::to_string(source_id);
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoDisableHover | ImGuiDragDropFlags_SourceAllowNullID)) {
      // Set the payload to the panel ID
      ImGui::SetDragDropPayload("PANEL_ID", &source_id, sizeof(uint32_t));

      // Show a preview of what's being dragged
      ImGui::Text("Dragging panel: %s", source_panel->get_title().c_str());
      
      // Add visual indicator for the drag operation
      ImGui::TextDisabled("(Drag to another panel to create tabbed group)");

      ImGui::EndDragDropSource();
    }
  }

  // Process drop targets - check if any panel can accept a dropped panel
  for (const auto& [target_id, target_panel] : panels_with_ids) {
    // Make the panel a drop target regardless of can_accept_drop() to enable tabbed group creation
    std::string drop_target_id = "PANEL_DROP_TARGET_" + std::to_string(target_id);
    ImGui::PushID(drop_target_id.c_str());

    if (ImGui::BeginDragDropTarget()) {
      // Visual feedback when a panel can accept a drop
      ImGui::PushStyleColor(ImGuiCol_DragDropTarget, IM_COL32(100, 150, 255, 200));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);
      
      if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PANEL_ID")) {
        if (payload->DataSize == sizeof(uint32_t)) {
          uint32_t source_panel_id = *(static_cast<const uint32_t*>(payload->Data));

          // Don't allow dropping a panel onto itself
          if (source_panel_id != target_id) {
            // Check if the target panel can accept drops directly
            PanelBase* target_panel_ptr = get_panel_by_id(target_id);
            if (target_panel_ptr && target_panel_ptr->can_accept_drop()) {
              if (target_panel_ptr->handle_drop(source_panel_id)) {
                // Successfully handled the drop - remove the source panel from the main panel list
                // since it's now managed by the target panel
                auto source_it = panels_.find(source_panel_id);
                if (source_it != panels_.end()) {
                  // For now, we'll just hide the source panel. In a real implementation,
                  // we might want to remove it from the main panel list entirely or manage it differently.
                  source_it->second->set_visible(false);

                  // Also remove it from any existing groups
                  auto group_ids = get_panel_groups_for_panel(source_panel_id);
                  for (uint32_t group_id : group_ids) {
                    remove_panel_from_group(group_id, source_panel_id);
                  }
                }
              }
            } else {
              // Target panel cannot accept drops directly, so create a tabbed panel to group them
              // First, check if either panel is already part of a tabbed panel
              bool source_is_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(source_panel_id)) != nullptr;
              bool target_is_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(target_id)) != nullptr;

              // If both panels are already in tabbed panels, merge the tabbed panels
              if (source_is_tabbed && target_is_tabbed) {
                // Merge the source tabbed panel into the target tabbed panel
                TabbedPanel* source_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(source_panel_id));
                TabbedPanel* target_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(target_id));

                if (source_tabbed && target_tabbed) {
                  // Transfer all panels from source tabbed panel to target tabbed panel
                  auto source_panels = source_tabbed->get_tabbed_panels();
                  for (uint32_t panel_id : source_panels) {
                    target_tabbed->add_panel(panel_id);

                    // Hide the transferred panel
                    PanelBase* panel = get_panel_by_id(panel_id);
                    if (panel) {
                      panel->set_visible(false);
                    }
                  }

                  // Hide the source tabbed panel since it's now merged
                  set_panel_visible(source_panel_id, false);

                  // Remove the source tabbed panel from any groups
                  auto source_group_ids = get_panel_groups_for_panel(source_panel_id);
                  for (uint32_t group_id : source_group_ids) {
                    remove_panel_from_group(group_id, source_panel_id);
                  }
                }
              }
              // If only the source panel is a tabbed panel, add it to the target tabbed panel
              else if (source_is_tabbed && !target_is_tabbed) {
                // Add the source tabbed panel to the target tabbed panel
                TabbedPanel* source_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(source_panel_id));
                TabbedPanel* target_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(target_id));

                if (source_tabbed && target_tabbed) {
                  // Transfer all panels from source tabbed panel to target tabbed panel
                  auto source_panels = source_tabbed->get_tabbed_panels();
                  for (uint32_t panel_id : source_panels) {
                    target_tabbed->add_panel(panel_id);

                    // Hide the transferred panel
                    PanelBase* panel = get_panel_by_id(panel_id);
                    if (panel) {
                      panel->set_visible(false);
                    }
                  }

                  // Hide the source tabbed panel since it's now merged
                  set_panel_visible(source_panel_id, false);

                  // Remove the source tabbed panel from any groups
                  auto source_group_ids = get_panel_groups_for_panel(source_panel_id);
                  for (uint32_t group_id : source_group_ids) {
                    remove_panel_from_group(group_id, source_panel_id);
                  }
                }
              }
              // If only the target panel is a tabbed panel, add the source panel to it
              else if (!source_is_tabbed && target_is_tabbed) {
                // Add the source panel to the existing target tabbed panel
                TabbedPanel* target_tabbed = dynamic_cast<TabbedPanel*>(get_panel_by_id(target_id));
                if (target_tabbed) {
                  target_tabbed->add_panel(source_panel_id);

                  // Hide the source panel since it's now managed by the tabbed panel
                  PanelBase* source_panel = get_panel_by_id(source_panel_id);
                  if (source_panel) {
                    source_panel->set_visible(false);
                  }

                  // Remove the source panel from any existing groups
                  auto source_group_ids = get_panel_groups_for_panel(source_panel_id);
                  for (uint32_t group_id : source_group_ids) {
                    remove_panel_from_group(group_id, source_panel_id);
                  }
                }
              }
              // If neither is a tabbed panel, create a new tabbed panel to group them
              else {
                // Create a new tabbed panel at the same position as the target panel
                auto target_config = get_panel_config(target_id);

                // Create the tabbed panel with the same position and size as the target
                uint32_t tabbed_panel_id = add_panel(PanelType::TABBED_PANEL, "Tabbed Group",
                                                     target_config.grid_x, target_config.grid_y,
                                                     target_config.grid_width, target_config.grid_height);

                if (tabbed_panel_id != 0) {
                  // Get the new tabbed panel
                  TabbedPanel* tabbed_panel = dynamic_cast<TabbedPanel*>(get_panel_by_id(tabbed_panel_id));
                  if (tabbed_panel) {
                    // Add both panels to the tabbed panel
                    tabbed_panel->add_panel(target_id);  // Add the target panel first
                    tabbed_panel->add_panel(source_panel_id);  // Then add the source panel

                    // Hide the original panels since they're now managed by the tabbed panel
                    set_panel_visible(target_id, false);
                    set_panel_visible(source_panel_id, false);

                    // Remove both panels from any existing groups
                    auto target_group_ids = get_panel_groups_for_panel(target_id);
                    for (uint32_t group_id : target_group_ids) {
                      remove_panel_from_group(group_id, target_id);
                    }

                    auto source_group_ids = get_panel_groups_for_panel(source_panel_id);
                    for (uint32_t group_id : source_group_ids) {
                      remove_panel_from_group(group_id, source_panel_id);
                    }

                    // Update the position and size of the tabbed panel to match the target panel
                    auto& tabbed_config = tabbed_panel->get_config();
                    tabbed_config.grid_x = target_config.grid_x;
                    tabbed_config.grid_y = target_config.grid_y;
                    tabbed_config.grid_width = target_config.grid_width;
                    tabbed_config.grid_height = target_config.grid_height;
                    tabbed_config.position = target_config.position;
                    tabbed_config.size = target_config.size;

                    // Log the successful creation of the tabbed group
                    std::cout << "Successfully created tabbed group with panels: "
                              << target_id << " and " << source_panel_id << std::endl;
                  }
                }
              }
            }
          }
        }
      } else {
        // Provide visual feedback when hovering over a potential drop target
        // Check if we're hovering over this drop target with a payload but haven't dropped yet
        ImGuiDragDropFlags target_flags = ImGuiDragDropFlags_AcceptBeforeDelivery;
        if (ImGui::AcceptDragDropPayload("PANEL_ID", target_flags)) {
          // This means we're hovering over the target with a payload
          // Provide visual feedback that this is a valid drop target for creating a tabbed group
          ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255, 255, 0, 255)); // Yellow border
          ImGui::Separator(); // Just to trigger a visual change
          ImGui::PopStyleColor();
        }
      }
      ImGui::PopStyleVar(); // FrameBorderSize
      ImGui::PopStyleColor(); // DragDropTarget

      ImGui::EndDragDropTarget();
    }
    ImGui::PopID();
  }
}

std::pair<int, int> PanelManager::find_best_docking_position(int width, int height) const {
  // If no panels exist, return (0,0) as the starting position
  if (panels_.empty()) {
    return std::make_pair(0, 0);
  }

  // Define the grid boundaries
  int max_cols = grid_layout_.columns;
  int max_rows = grid_layout_.rows;

  // Create a 2D grid to track occupied cells
  std::vector<std::vector<bool>> occupied(max_rows, std::vector<bool>(max_cols, false));

  // Mark occupied cells based on existing panels
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    // Only consider visible panels that are not part of locked groups
    if (config.visible) {
      bool is_locked = false;

      // Check if panel is part of a locked group
      auto group_it = panel_to_group_map_.find(id);
      if (group_it != panel_to_group_map_.end()) {
        uint32_t group_id = group_it->second;
        auto panel_group_it = panel_groups_.find(group_id);
        if (panel_group_it != panel_groups_.end() && panel_group_it->second.locked) {
          is_locked = true;
        }
      }

      if (!is_locked) {
        // Mark the grid cells occupied by this panel
        for (int y = config.grid_y; y < config.grid_y + config.grid_height && y < max_rows; ++y) {
          for (int x = config.grid_x; x < config.grid_x + config.grid_width && x < max_cols; ++x) {
            if (x >= 0 && y >= 0) {  // Ensure we don't access negative indices
              occupied[y][x] = true;
            }
          }
        }
      }
    }
  }

  // Priority 1: Try to dock to the edges of existing panels in a preferred order
  // Order: Right edge, Below, Left edge, Above (clockwise around existing panels)

  // Collect all potential docking positions with priority
  std::vector<std::pair<int, int>> potential_positions;

  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    // Only consider visible panels that are not part of locked groups
    if (!config.visible) continue;

    bool is_locked = false;
    auto group_it = panel_to_group_map_.find(id);
    if (group_it != panel_to_group_map_.end()) {
      uint32_t group_id = group_it->second;
      auto panel_group_it = panel_groups_.find(group_id);
      if (panel_group_it != panel_groups_.end() && panel_group_it->second.locked) {
        is_locked = true;
      }
    }

    if (is_locked) continue;

    // Try placing to the right of the current panel (priority 1)
    int right_x = config.grid_x + config.grid_width;
    int right_y = config.grid_y;
    if (right_x + width <= max_cols) {  // Check if it fits horizontally
      bool can_place_right = true;
      for (int dy = 0; dy < height && can_place_right; ++dy) {
        for (int dx = 0; dx < width && can_place_right; ++dx) {
          int check_x = right_x + dx;
          int check_y = right_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_y < max_rows && check_x < max_cols) {
            if (occupied[check_y][check_x]) {
              can_place_right = false;
              break;
            }
          } else {
            can_place_right = false;  // Out of bounds
            break;
          }
        }
      }
      if (can_place_right) {
        potential_positions.push_back({right_x, right_y});
      }
    }

    // Try placing below the current panel (priority 2)
    int below_x = config.grid_x;
    int below_y = config.grid_y + config.grid_height;
    if (below_y + height <= max_rows) {  // Check if it fits vertically
      bool can_place_below = true;
      for (int dy = 0; dy < height && can_place_below; ++dy) {
        for (int dx = 0; dx < width && can_place_below; ++dx) {
          int check_x = below_x + dx;
          int check_y = below_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_y < max_rows && check_x < max_cols) {
            if (occupied[check_y][check_x]) {
              can_place_below = false;
              break;
            }
          } else {
            can_place_below = false;  // Out of bounds
            break;
          }
        }
      }
      if (can_place_below) {
        potential_positions.push_back({below_x, below_y});
      }
    }

    // Try placing to the left of the current panel (priority 3)
    int left_x = config.grid_x - width;
    int left_y = config.grid_y;
    if (left_x >= 0) {  // Check if it fits horizontally
      bool can_place_left = true;
      for (int dy = 0; dy < height && can_place_left; ++dy) {
        for (int dx = 0; dx < width && can_place_left; ++dx) {
          int check_x = left_x + dx;
          int check_y = left_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_y < max_rows && check_x < max_cols) {
            if (occupied[check_y][check_x]) {
              can_place_left = false;
              break;
            }
          } else {
            can_place_left = false;  // Out of bounds
            break;
          }
        }
      }
      if (can_place_left) {
        potential_positions.push_back({left_x, left_y});
      }
    }

    // Try placing above the current panel (priority 4)
    int above_x = config.grid_x;
    int above_y = config.grid_y - height;
    if (above_y >= 0) {  // Check if it fits vertically
      bool can_place_above = true;
      for (int dy = 0; dy < height && can_place_above; ++dy) {
        for (int dx = 0; dx < width && can_place_above; ++dx) {
          int check_x = above_x + dx;
          int check_y = above_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_y < max_rows && check_x < max_cols) {
            if (occupied[check_y][check_x]) {
              can_place_above = false;
              break;
            }
          } else {
            can_place_above = false;  // Out of bounds
            break;
          }
        }
      }
      if (can_place_above) {
        potential_positions.push_back({above_x, above_y});
      }
    }
  }

  // If we found any potential docking positions, return the first one (which follows our priority order)
  if (!potential_positions.empty()) {
    return potential_positions[0];
  }

  // Priority 2: Look for the first available spot that fits the new panel in the grid
  for (int y = 0; y < max_rows; ++y) {
    for (int x = 0; x < max_cols; ++x) {
      // Check if the space starting at (x,y) is available for the panel size
      bool can_place = true;
      if (x + width > max_cols || y + height > max_rows) {
        can_place = false; // Panel would extend beyond grid boundaries
      } else {
        for (int dy = 0; dy < height && can_place; ++dy) {
          for (int dx = 0; dx < width && can_place; ++dx) {
            if (occupied[y + dy][x + dx]) {
              can_place = false;
              break;
            }
          }
        }
      }

      if (can_place) {
        return std::make_pair(x, y);
      }
    }
  }

  // Priority 3: If still no space found, try to expand the grid by looking for positions
  // just adjacent to existing panels even if they go beyond the original grid bounds
  // (within reason - we don't want to place too far away)
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();

    // Only consider visible panels that are not part of locked groups
    if (!config.visible) continue;

    bool is_locked = false;
    auto group_it = panel_to_group_map_.find(id);
    if (group_it != panel_to_group_map_.end()) {
      uint32_t group_id = group_it->second;
      auto panel_group_it = panel_groups_.find(group_id);
      if (panel_group_it != panel_groups_.end() && panel_group_it->second.locked) {
        is_locked = true;
      }
    }

    if (is_locked) continue;

    // Try expanding to the right (within reasonable bounds)
    int expand_right_x = config.grid_x + config.grid_width;
    int expand_right_y = config.grid_y;
    if (expand_right_x + width <= max_cols * 3) {  // Allow expansion up to 3x the column count
      bool can_expand_right = true;
      for (int dy = 0; dy < height && can_expand_right; ++dy) {
        for (int dx = 0; dx < width && can_expand_right; ++dx) {
          int check_x = expand_right_x + dx;
          int check_y = expand_right_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_y < max_rows * 3) {  // Allow expansion up to 3x the row count
            if (check_x < max_cols && check_y < max_rows) {
              // Within original grid - check occupation
              if (occupied[check_y][check_x]) {
                can_expand_right = false;
                break;
              }
            }
            // For expanded areas beyond original grid, check if coordinates are reasonable and not occupied
            else if (check_x < max_cols * 3 && check_y < max_rows * 3) {
              // Check if this expanded area is occupied by any panel
              bool expanded_area_occupied = false;
              for (const auto& [other_id, other_panel] : panels_) {
                const auto& other_config = other_panel->get_config();

                // Check if the new position overlaps with any existing panel in the expanded area
                if ((check_x < other_config.grid_x + other_config.grid_width) &&
                    (check_x + width > other_config.grid_x) &&
                    (check_y < other_config.grid_y + other_config.grid_height) &&
                    (check_y + height > other_config.grid_y)) {
                  expanded_area_occupied = true;
                  break;
                }
              }

              if (expanded_area_occupied) {
                can_expand_right = false;
                break;
              }
            } else {
              can_expand_right = false;  // Out of reasonable bounds
              break;
            }
          } else {
            can_expand_right = false;  // Out of reasonable bounds
            break;
          }
        }
      }
      if (can_expand_right) {
        return std::make_pair(expand_right_x, expand_right_y);
      }
    }

    // Try expanding below (within reasonable bounds)
    int expand_below_x = config.grid_x;
    int expand_below_y = config.grid_y + config.grid_height;
    if (expand_below_y + height <= max_rows * 3) {  // Allow expansion up to 3x the row count
      bool can_expand_below = true;
      for (int dy = 0; dy < height && can_expand_below; ++dy) {
        for (int dx = 0; dx < width && can_expand_below; ++dx) {
          int check_x = expand_below_x + dx;
          int check_y = expand_below_y + dy;
          if (check_x >= 0 && check_y >= 0 && check_x < max_cols * 3) {  // Allow expansion up to 3x the column count
            if (check_x < max_cols && check_y < max_rows) {
              // Within original grid - check occupation
              if (occupied[check_y][check_x]) {
                can_expand_below = false;
                break;
              }
            }
            // For expanded areas beyond original grid, check if coordinates are reasonable and not occupied
            else if (check_x < max_cols * 3 && check_y < max_rows * 3) {
              // Check if this expanded area is occupied by any panel
              bool expanded_area_occupied = false;
              for (const auto& [other_id, other_panel] : panels_) {
                const auto& other_config = other_panel->get_config();

                // Check if the new position overlaps with any existing panel in the expanded area
                if ((check_x < other_config.grid_x + other_config.grid_width) &&
                    (check_x + width > other_config.grid_x) &&
                    (check_y < other_config.grid_y + other_config.grid_height) &&
                    (check_y + height > other_config.grid_y)) {
                  expanded_area_occupied = true;
                  break;
                }
              }

              if (expanded_area_occupied) {
                can_expand_below = false;
                break;
              }
            } else {
              can_expand_below = false;  // Out of reasonable bounds
              break;
            }
          } else {
            can_expand_below = false;  // Out of reasonable bounds
            break;
          }
        }
      }
      if (can_expand_below) {
        return std::make_pair(expand_below_x, expand_below_y);
      }
    }

    // Try expanding to the left (within reasonable bounds)
    int expand_left_x = config.grid_x - width;
    int expand_left_y = config.grid_y;
    if (expand_left_x >= -max_cols * 2) {  // Allow expansion to negative coordinates up to 2x the column count
      bool can_expand_left = true;
      for (int dy = 0; dy < height && can_expand_left; ++dy) {
        for (int dx = 0; dx < width && can_expand_left; ++dx) {
          int check_x = expand_left_x + dx;
          int check_y = expand_left_y + dy;
          if (check_x >= -max_cols * 2 && check_y >= 0 && check_y < max_rows * 3) {  // Allow negative x coordinates
            if (check_x < max_cols && check_y < max_rows && check_x >= 0) {
              // Within original grid - check occupation
              if (occupied[check_y][check_x]) {
                can_expand_left = false;
                break;
              }
            }
            // For expanded areas beyond original grid, check if coordinates are reasonable and not occupied
            else if (check_x < max_cols * 3 && check_x >= -max_cols * 2 && check_y < max_rows * 3) {
              // Check if this expanded area is occupied by any panel
              bool expanded_area_occupied = false;
              for (const auto& [other_id, other_panel] : panels_) {
                const auto& other_config = other_panel->get_config();

                // Check if the new position overlaps with any existing panel in the expanded area
                if ((check_x < other_config.grid_x + other_config.grid_width) &&
                    (check_x + width > other_config.grid_x) &&
                    (check_y < other_config.grid_y + other_config.grid_height) &&
                    (check_y + height > other_config.grid_y)) {
                  expanded_area_occupied = true;
                  break;
                }
              }

              if (expanded_area_occupied) {
                can_expand_left = false;
                break;
              }
            } else {
              can_expand_left = false;  // Out of reasonable bounds
              break;
            }
          } else {
            can_expand_left = false;  // Out of reasonable bounds
            break;
          }
        }
      }
      if (can_expand_left) {
        return std::make_pair(expand_left_x, expand_left_y);
      }
    }

    // Try expanding above (within reasonable bounds)
    int expand_above_x = config.grid_x;
    int expand_above_y = config.grid_y - height;
    if (expand_above_y >= -max_rows * 2) {  // Allow expansion to negative coordinates up to 2x the row count
      bool can_expand_above = true;
      for (int dy = 0; dy < height && can_expand_above; ++dy) {
        for (int dx = 0; dx < width && can_expand_above; ++dx) {
          int check_x = expand_above_x + dx;
          int check_y = expand_above_y + dy;
          if (check_x >= 0 && check_x < max_cols * 3 && check_y >= -max_rows * 2) {  // Allow negative y coordinates
            if (check_x < max_cols && check_y < max_rows && check_y >= 0) {
              // Within original grid - check occupation
              if (occupied[check_y][check_x]) {
                can_expand_above = false;
                break;
              }
            }
            // For expanded areas beyond original grid, check if coordinates are reasonable and not occupied
            else if (check_x < max_cols * 3 && check_x >= 0 && check_y < max_rows * 3 && check_y >= -max_rows * 2) {
              // Check if this expanded area is occupied by any panel
              bool expanded_area_occupied = false;
              for (const auto& [other_id, other_panel] : panels_) {
                const auto& other_config = other_panel->get_config();

                // Check if the new position overlaps with any existing panel in the expanded area
                if ((check_x < other_config.grid_x + other_config.grid_width) &&
                    (check_x + width > other_config.grid_x) &&
                    (check_y < other_config.grid_y + other_config.grid_height) &&
                    (check_y + height > other_config.grid_y)) {
                  expanded_area_occupied = true;
                  break;
                }
              }

              if (expanded_area_occupied) {
                can_expand_above = false;
                break;
              }
            } else {
              can_expand_above = false;  // Out of reasonable bounds
              break;
            }
          } else {
            can_expand_above = false;  // Out of reasonable bounds
            break;
          }
        }
      }
      if (can_expand_above) {
        return std::make_pair(expand_above_x, expand_above_y);
      }
    }
  }

  // If still no space found adjacent to existing panels, create a position next to the first panel
  // This ensures we never fall back to (0,0) when there are existing panels
  if (!panels_.empty()) {
    // Find the first visible panel that is not locked
    for (const auto& [id, panel] : panels_) {
      const auto& config = panel->get_config();

      if (!config.visible) continue;

      bool is_locked = false;
      auto group_it = panel_to_group_map_.find(id);
      if (group_it != panel_to_group_map_.end()) {
        uint32_t group_id = group_it->second;
        auto panel_group_it = panel_groups_.find(group_id);
        if (panel_group_it != panel_groups_.end() && panel_group_it->second.locked) {
          is_locked = true;
        }
      }

      if (is_locked) continue;

      // Try to place to the right of the first panel
      int fallback_x = config.grid_x + config.grid_width;
      int fallback_y = config.grid_y;

      // Check if this position is valid and not overlapping
      bool position_valid = true;
      for (const auto& [other_id, other_panel] : panels_) {
        const auto& other_config = other_panel->get_config();

        if ((fallback_x < other_config.grid_x + other_config.grid_width) &&
            (fallback_x + width > other_config.grid_x) &&
            (fallback_y < other_config.grid_y + other_config.grid_height) &&
            (fallback_y + height > other_config.grid_y)) {
          position_valid = false;
          break;
        }
      }

      if (position_valid) {
        return std::make_pair(fallback_x, fallback_y);
      }

      // If right doesn't work, try below
      fallback_x = config.grid_x;
      fallback_y = config.grid_y + config.grid_height;

      position_valid = true;
      for (const auto& [other_id, other_panel] : panels_) {
        const auto& other_config = other_panel->get_config();

        if ((fallback_x < other_config.grid_x + other_config.grid_width) &&
            (fallback_x + width > other_config.grid_x) &&
            (fallback_y < other_config.grid_y + other_config.grid_height) &&
            (fallback_y + height > other_config.grid_y)) {
          position_valid = false;
          break;
        }
      }

      if (position_valid) {
        return std::make_pair(fallback_x, fallback_y);
      }
    }
  }

  // If no panels exist, return (0,0) as the starting position
  // This is the only legitimate case to return (0,0)
  if (panels_.empty()) {
    return std::make_pair(0, 0);
  }

  // If we have panels but couldn't find a suitable position, return a position far to the right
  // of the rightmost panel as a last resort
  int max_x = 0;
  for (const auto& [id, panel] : panels_) {
    const auto& config = panel->get_config();
    if (config.grid_x + config.grid_width > max_x) {
      max_x = config.grid_x + config.grid_width;
    }
  }

  return std::make_pair(max_x, 0);
}

void PanelManager::set_panel_symbol_link_group(uint32_t panel_id, int group) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    it->second->set_symbol_link_group(group);
  }
}

void PanelManager::update_linked_symbols(uint32_t source_panel_id, const std::string& new_symbol) {
  auto source_it = panels_.find(source_panel_id);
  if (source_it == panels_.end()) {
    return; // Source panel not found
  }
  
  int source_group = source_it->second->get_symbol_link_group();
  if (source_group <= 0) {
    return; // Source panel is not in a symbol link group
  }
  
  // Update all panels in the same symbol link group
  for (auto& [id, panel] : panels_) {
    if (id != source_panel_id && panel->get_symbol_link_group() == source_group) {
      // Update the panel's symbol if it has symbol-dependent functionality
      set_panel_symbol(id, new_symbol);
      
      // Update the panel's config symbol
      panel->get_config().symbol = new_symbol;
    }
  }
}

std::vector<uint32_t> PanelManager::get_panels_in_symbol_link_group(int group) const {
  std::vector<uint32_t> panel_ids;
  
  for (const auto& [id, panel] : panels_) {
    if (panel->get_symbol_link_group() == group) {
      panel_ids.push_back(id);
    }
  }
  
  return panel_ids;
}

void PanelManager::mark_visualization_panels_dirty() {
  for (auto& [id, panel] : panels_) {
    // Only mark panels as dirty if they are visualization panels that need to update when trades arrive
    switch (panel->get_config().type) {
      case PanelType::FOOTPRINT_CHART:
      case PanelType::TPO_PROFILE:
      case PanelType::HEATMAP:
      case PanelType::CHART:
      case PanelType::VOLUME_PROFILE:
      case PanelType::DEPTH_CHART:
        // Mark these panels as dirty to trigger a redraw when new trade data arrives
        panel->markDirty();
        break;
      default:
        // Other panels don't need to be marked dirty for every trade
        break;
    }
  }
}

}  // namespace BTQuant