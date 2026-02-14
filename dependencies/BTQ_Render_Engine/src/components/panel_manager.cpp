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
#include "../../include/components/correlation_heatmap_panel.hpp"
#include "../../include/components/multi_vwap_panel.hpp"
#include "../../include/components/technical_indicators_panel.hpp"
#include "../../include/components/theme_customization_panel.hpp"
#include "../../include/components/keyboard_shortcuts_panel.hpp"
#include "../../include/components/drawing_tools_panel.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/performance/panel_profiler.hpp"

using json = nlohmann::json;

namespace BTQuant {

PanelManager::PanelManager(std::shared_ptr<HotSpineDataBridge> bridge,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                           std::shared_ptr<OrderManager> order_manager,
                           std::shared_ptr<PositionManager> position_manager,
                           std::shared_ptr<RiskAssessment> risk_assessment)
    : bridge_(bridge),
      processor_(processor),
      order_manager_(order_manager),
      position_manager_(position_manager),
      risk_assessment_(risk_assessment) {
  // NOTE: Do NOT call apply_layout_preset() or add default panels in constructor.
  // Panel instantiation should be controlled by the layout system externally.
  // See: main_trading_terminal.cpp where workspace->set_layout() is called.
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
  context_menu_manager_ = std::make_unique<ContextMenuManager>(this);
  strategy_builder_ = std::make_unique<RenderEngine::StrategyBuilder>(PanelConfig{.title = "Strategy Builder", .type = PanelType::STRATEGY_BUILDER});
}

PanelManager::~PanelManager() {
  context_menu_manager_.reset(); // Explicitly reset context menu manager before other members
  strategy_builder_.reset(); // Explicitly reset strategy builder before other members
  panels_.clear();
}

void PanelManager::initialize() {
  // Set grid layout (3 columns, 5 rows to fit 2x2 chart properly)
  set_grid_layout(3, 5);

  // Initialize with no default panels - let the layout system dictate what gets created
  // The layout system should completely control panel instantiation
  // DO NOT add default panels here - this is handled by external layout system
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
      panel = std::make_unique<FootprintPanel>(config);
      break;
    case PanelType::TPO_PROFILE:
      panel = std::make_unique<TpoPanel>(config);
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
      panel = std::make_unique<ChartReplayPanel>(config, bridge_, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, bridge_, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::CORRELATION_HEATMAP:
      panel = std::make_unique<CorrelationHeatmapPanel>(config);
      break;
    case PanelType::DOM_SURFACE:
      panel = std::make_unique<DomSurfacePanel>(processor_);
      break;
    case PanelType::MULTI_VWAP:
      panel = std::make_unique<MultiVWAPPanel>(config);
      break;
    case PanelType::TECHNICAL_INDICATORS:
      panel = std::make_unique<TechnicalIndicatorsPanel>(config);
      break;
    case PanelType::THEME_CUSTOMIZATION:
      panel = std::make_unique<ThemeCustomizationPanel>(config);
      break;
    case PanelType::KEYBOARD_SHORTCUTS:
      panel = std::make_unique<KeyboardShortcutsPanel>(config);
      break;
    case PanelType::DRAWING_TOOLS:
      panel = std::make_unique<DrawingToolsPanel>(config);
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
      // TODO: Implement proper connection between watchlist and alerts panels
      // For now, skip this connection to allow compilation
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
      panel = std::make_unique<FootprintPanel>(config);
      break;
    case PanelType::TPO_PROFILE:
      panel = std::make_unique<TpoPanel>(config);
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
      panel = std::make_unique<ChartReplayPanel>(config, bridge_, processor_, chart_manager_.get());
      break;
    case PanelType::RISK_ANALYZER:
      panel = std::make_unique<RiskAnalyzerPanel>(config, bridge_, processor_);
      break;
    case PanelType::STRATEGY_BUILDER:
      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>(config);
      break;
    case PanelType::CORRELATION_HEATMAP:
      panel = std::make_unique<CorrelationHeatmapPanel>(config);
      break;
    case PanelType::DOM_SURFACE:
      panel = std::make_unique<DomSurfacePanel>(processor_);
      break;
    case PanelType::MULTI_VWAP:
      panel = std::make_unique<MultiVWAPPanel>(config);
      break;
    case PanelType::TECHNICAL_INDICATORS:
      panel = std::make_unique<TechnicalIndicatorsPanel>(config);
      break;
    case PanelType::THEME_CUSTOMIZATION:
      panel = std::make_unique<ThemeCustomizationPanel>(config);
      break;
    case PanelType::KEYBOARD_SHORTCUTS:
      panel = std::make_unique<KeyboardShortcutsPanel>(config);
      break;
    case PanelType::DRAWING_TOOLS:
      panel = std::make_unique<DrawingToolsPanel>(config);
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
      // TODO: Implement proper connection between watchlist and alerts panels
      // For now, skip this connection to allow compilation
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
  next_panel_id_ = 1;
}

void PanelManager::move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    auto& config = it->second->get_config();
    config.grid_x = new_grid_x;
    config.grid_y = new_grid_y;
    config.position = calculate_panel_position(new_grid_x, new_grid_y);
  }
}

void PanelManager::resize_panel(uint32_t panel_id, int new_width, int new_height) {
  auto it = panels_.find(panel_id);
  if (it != panels_.end()) {
    auto& config = it->second->get_config();
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
    case PanelType::CORRELATION_HEATMAP:
      return "Correlation Heatmap";
    case PanelType::DOM_SURFACE:
      return "DOM Surface";
    case PanelType::MULTI_VWAP:
      return "Multi VWAP";
    case PanelType::TECHNICAL_INDICATORS:
      return "Technical Indicators";
    case PanelType::THEME_CUSTOMIZATION:
      return "Theme";
    case PanelType::KEYBOARD_SHORTCUTS:
      return "Shortcuts";
    case PanelType::DRAWING_TOOLS:
      return "Drawing Tools";
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
    }
    // Option Analytics Panel specific settings
    else if (auto* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel*>(panel.get())) {
        settings_json["active_tab"] = option_panel->get_active_tab();
    }
    
    // Add settings if any were captured
    if (!settings_json.empty()) {
        panel_json["settings"] = settings_json;
    }

    panels_json.push_back(panel_json);
  }
  layout_json["panels"] = panels_json;

  return layout_json.dump(4);
}

void PanelManager::deserialize_layout(const std::string& layout_json) {
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
                }
                // Option Analytics Panel specific settings
                else if (auto* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel*>(panel)) {
                    if (settings.contains("active_tab")) {
                        option_panel->set_active_tab(settings["active_tab"].get<int>());
                    }
                }
            }
        }
        
        set_panel_visible(id, visible);
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

void PanelManager::apply_layout_preset(LayoutPreset preset) {
  // CRITICAL: Clear all existing panels first to prevent duplication
  clear_panels();

  // Apply the specific layout based on the preset
  switch (preset) {
    case LayoutPreset::DEFAULT:
      // Add default panels for the default layout
      add_panel(PanelType::CHART, "Chart", 0, 0, 2, 3);
      add_panel(PanelType::ORDERBOOK, "Orderbook", 2, 0, 1, 2);
      add_panel(PanelType::METRICS, "Metrics", 2, 2, 1, 1);
      break;
      
    case LayoutPreset::MODERN_TRADING:
      // Modern trading layout with multiple panels
      add_panel(PanelType::CHART, "Price Chart", 0, 0, 2, 2);
      add_panel(PanelType::ORDERBOOK, "Order Book", 2, 0, 1, 2);
      add_panel(PanelType::WATCHLIST, "Watchlist", 0, 2, 1, 1);
      add_panel(PanelType::TIME_AND_SALES, "Time & Sales", 1, 2, 1, 1);
      add_panel(PanelType::TRADING_ORDERS, "Orders", 2, 2, 1, 1);
      add_panel(PanelType::TRADING_POSITIONS, "Positions", 0, 3, 1, 1);
      add_panel(PanelType::RISK_METRICS, "Risk", 1, 3, 1, 1);
      add_panel(PanelType::STATUS_BAR, "Status", 2, 3, 1, 1);
      break;
      
    case LayoutPreset::DASHBOARD_ONLY:
      // Layout with only dashboard elements, no trading panels
      add_panel(PanelType::CHART, "Chart", 0, 0, 2, 2);
      add_panel(PanelType::METRICS, "Metrics", 2, 0, 1, 1);
      add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 2, 1, 1, 1);
      add_panel(PanelType::WATCHLIST, "Watchlist", 0, 2, 3, 1);
      break;
      
    case LayoutPreset::CHART_FOCUS:
      // Layout focused on charting with minimal other panels
      add_panel(PanelType::CHART, "Main Chart", 0, 0, 3, 3);
      add_panel(PanelType::ORDERBOOK, "Orderbook", 0, 3, 1, 1);
      add_panel(PanelType::TIME_AND_SALES, "T&S", 1, 3, 1, 1);
      add_panel(PanelType::STATUS_BAR, "Status", 2, 3, 1, 1);
      break;
      
    case LayoutPreset::RISK_MONITORING:
      // Layout focused on risk monitoring
      add_panel(PanelType::RISK_METRICS, "Risk Metrics", 0, 0, 1, 2);
      add_panel(PanelType::TRADING_POSITIONS, "Positions", 1, 0, 1, 2);
      add_panel(PanelType::CHART, "Chart", 2, 0, 1, 2);
      add_panel(PanelType::RISK_ANALYZER, "Risk Analyzer", 0, 2, 3, 1);
      add_panel(PanelType::STATUS_BAR, "Status", 0, 3, 3, 1);
      break;
  }
}

}  // namespace BTQuant