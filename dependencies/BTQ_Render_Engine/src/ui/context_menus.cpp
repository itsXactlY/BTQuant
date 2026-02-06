#include "ui/context_menus.hpp"

#include <functional>
#include <string>
#include <algorithm>

#include "imgui.h"
#include "implot.h"

// Need to include panel_manager.hpp to use its methods
#include "components/chart_panel.hpp"
#include "components/orderbook_panel.hpp"
#include "components/panel_manager.hpp"
#include "components/watchlist_panel.hpp"

// Include all panel headers needed for dynamic_cast operations
#include "components/alerts_panel.hpp"
#include "components/chart_replay_panel.hpp"
#include "components/depth_chart_panel.hpp"
#include "components/dom_surface_panel.hpp"
#include "components/footprint_panel.hpp"
#include "components/histogram_panel.hpp"
#include "components/log_panel.hpp"
#include "components/metrics_panel.hpp"
#include "components/performance_monitor_panel.hpp"
#include "components/risk_metrics_panel.hpp"
#include "components/scatter_plot_panel.hpp"
#include "components/screener_panel.hpp"
#include "components/time_and_sales.hpp"
#include "components/time_histogram_panel.hpp"
#include "components/time_series_panel.hpp"
#include "components/time_statistics_panel.hpp"
#include "components/tpo_panel.hpp"
#include "components/trading_orders_panel.hpp"
#include "components/trading_positions_panel.hpp"
#include "components/volume_profile_panel.hpp"
#include "components/risk_analyzer_panel.hpp"
#include "components/historical_time_sales.hpp"
#include "components/tape_panel.hpp"
#include "components/status_bar_panel.hpp"
#include "ui/screenshot_utility.hpp"
namespace BTQuant {

// Context menu manager implementation
ContextMenuManager::ContextMenuManager(PanelManager* panel_manager)
    : panel_manager_(panel_manager) {
  // Initialize context menu handlers for different panel types
  initialize_context_menus();
}

void ContextMenuManager::initialize_context_menus() {
  // Register context menu handlers for each panel type
  // Using generic approach that works with PanelBase
  context_menu_handlers_[PanelType::CHART] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "ChartContextMenu");
  };

  context_menu_handlers_[PanelType::WATCHLIST] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "WatchlistContextMenu");
  };

  context_menu_handlers_[PanelType::ORDERBOOK] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "OrderbookContextMenu");
  };

  context_menu_handlers_[PanelType::FOOTPRINT_CHART] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "FootprintChartContextMenu");
  };

  context_menu_handlers_[PanelType::VOLUME_PROFILE] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "VolumeProfileContextMenu");
  };

  context_menu_handlers_[PanelType::TPO_PROFILE] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TPOProfileContextMenu");
  };

  context_menu_handlers_[PanelType::HEATMAP] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "HeatmapContextMenu");
  };

  context_menu_handlers_[PanelType::ALERTS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "AlertsContextMenu");
  };

  context_menu_handlers_[PanelType::TIME_AND_SALES] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TimeAndSalesContextMenu");
  };

  context_menu_handlers_[PanelType::DEPTH_CHART] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "DepthChartContextMenu");
  };

  context_menu_handlers_[PanelType::PERFORMANCE_MONITOR] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "PerformanceMonitorContextMenu");
  };

  context_menu_handlers_[PanelType::TRADING_ORDERS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TradingOrdersContextMenu");
  };

  context_menu_handlers_[PanelType::TRADING_POSITIONS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TradingPositionsContextMenu");
  };

  context_menu_handlers_[PanelType::RISK_METRICS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "RiskMetricsContextMenu");
  };

  context_menu_handlers_[PanelType::METRICS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "MetricsContextMenu");
  };

  context_menu_handlers_[PanelType::SCATTER_PLOT] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "ScatterPlotContextMenu");
  };

  context_menu_handlers_[PanelType::HISTOGRAM] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "HistogramContextMenu");
  };

  context_menu_handlers_[PanelType::TIME_SERIES] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TimeSeriesContextMenu");
  };

  context_menu_handlers_[PanelType::TIME_STATISTICS] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TimeStatisticsContextMenu");
  };

  context_menu_handlers_[PanelType::TIME_HISTOGRAM] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TimeHistogramContextMenu");
  };

  context_menu_handlers_[PanelType::SCREENER] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "ScreenerContextMenu");
  };

  context_menu_handlers_[PanelType::LOG_PANEL] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "LogPanelContextMenu");
  };

  context_menu_handlers_[PanelType::CHART_REPLAY] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "ChartReplayContextMenu");
  };

  context_menu_handlers_[PanelType::RISK_ANALYZER] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "RiskAnalyzerContextMenu");
  };

  context_menu_handlers_[PanelType::HISTORICAL_TIME_SALES] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "HistoricalTimeSalesContextMenu");
  };

  context_menu_handlers_[PanelType::TAPE] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "TapeContextMenu");
  };

  context_menu_handlers_[PanelType::STATUS_BAR] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "StatusBarContextMenu");
  };

  context_menu_handlers_[PanelType::STRATEGY_BUILDER] = [this](PanelBase* panel) {
    render_generic_context_menu(panel, "StrategyBuilderContextMenu");
  };
}

void ContextMenuManager::show_context_menu(PanelBase* panel) {
  if (!panel) return;

  auto handler_it = context_menu_handlers_.find(panel->get_config().type);
  if (handler_it != context_menu_handlers_.end()) {
    handler_it->second(panel);
  }
}

void ContextMenuManager::render_generic_context_menu(PanelBase* panel, const char* popup_name) {
  // Check if the window is hovered and right mouse button was clicked
  if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    // Open the context menu
    ImGui::OpenPopup(popup_name);
  }

  // Create the context menu
  if (ImGui::BeginPopup(popup_name)) {
    // Display panel-specific menu items based on panel type
    switch (panel->get_config().type) {
      case PanelType::CHART:
        ImGui::Text("Chart Actions:");
        ImGui::Separator();
        if (ImGui::MenuItem("Add Technical Indicator")) {
          // Trigger event to add a technical indicator to the chart
          if (panel_manager_) {
            // This would typically send an event to the chart to open indicator selection
            // For now, we'll just log that the action was triggered
          }
        }
        if (ImGui::BeginMenu("Set Timeframe")) {
          // Show available timeframes as submenu items
          if (auto* chart_panel = dynamic_cast<ChartPanel*>(panel)) {
            // Find the panel ID by comparing with all panels in the manager
            uint32_t panel_id = 0;
            if (panel_manager_) {
              auto all_panel_ids = panel_manager_->get_all_panel_ids();
              for (uint32_t id : all_panel_ids) {
                PanelBase* manager_panel = panel_manager_->get_panel_by_id(id);
                if (manager_panel == panel) {
                  panel_id = id;
                  break;
                }
              }
            }

            // Define timeframe options with their display names
            const char* timeframe_names[] = {"1ms", "10ms", "100ms", "500ms", "1s",  "3s",  "5s",
                                             "15s", "30s",  "1m",    "2m",    "5m",  "15m", "30m",
                                             "1h",  "2h",   "4h",    "6h",    "12h", "1d",  "1w"};

            RenderEngine::TimeFrame timeframes[] = {
                RenderEngine::TimeFrame::TF_1MS,    RenderEngine::TimeFrame::TF_10MS,
                RenderEngine::TimeFrame::TF_100MS,  RenderEngine::TimeFrame::TF_500MS,
                RenderEngine::TimeFrame::TF_1SEC,   RenderEngine::TimeFrame::TF_3SEC,
                RenderEngine::TimeFrame::TF_5SEC,   RenderEngine::TimeFrame::TF_15SEC,
                RenderEngine::TimeFrame::TF_30SEC,  RenderEngine::TimeFrame::TF_1MIN,
                RenderEngine::TimeFrame::TF_2MIN,   RenderEngine::TimeFrame::TF_5MIN,
                RenderEngine::TimeFrame::TF_15MIN,  RenderEngine::TimeFrame::TF_30MIN,
                RenderEngine::TimeFrame::TF_1HOUR,  RenderEngine::TimeFrame::TF_2HOUR,
                RenderEngine::TimeFrame::TF_4HOUR,  RenderEngine::TimeFrame::TF_6HOUR,
                RenderEngine::TimeFrame::TF_12HOUR, RenderEngine::TimeFrame::TF_1DAY,
                RenderEngine::TimeFrame::TF_1WEEK};

            // Create menu items for each timeframe
            for (int i = 0; i < 21; ++i) {
              if (ImGui::MenuItem(timeframe_names[i])) {
                // Set the new timeframe on the chart panel
                chart_panel->set_timeframe(timeframes[i]);

                // Update the panel configuration in the panel manager to persist the change
                if (panel_manager_ && panel_id != 0) {
                  PanelConfig updated_config = panel_manager_->get_panel_config(panel_id);
                  updated_config.title =
                      chart_panel->get_config().title;  // Update title which includes timeframe
                  panel_manager_->update_panel_config(panel_id, updated_config);
                }
              }
            }
          }
          ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Export Data")) {
          // Export chart data to CSV or other format
          if (panel_manager_) {
            // This would typically trigger data export functionality
          }
        }
        if (ImGui::MenuItem("Take Screenshot")) {
          // Capture and save chart screenshot
          if (panel_manager_) {
            // This would typically trigger screenshot functionality
          }
        }
        if (ImGui::MenuItem("Reset Zoom")) {
          // Reset chart zoom level to default
          if (panel_manager_) {
            // This would typically reset the chart's view transformation
          }
        }
        if (ImGui::MenuItem("Apply Symbol to All")) {
          // Apply the current chart's symbol to all other panels
          if (panel_manager_ && !panel->get_config().symbol.empty()) {
            // Get the current symbol from this chart
            std::string current_symbol = panel->get_config().symbol;
            
            // Iterate through all panels and update their symbols
            auto all_panel_ids = panel_manager_->get_all_panel_ids();
            for (uint32_t id : all_panel_ids) {
              PanelBase* other_panel = panel_manager_->get_panel_by_id(id);
              if (other_panel && other_panel != panel) {  // Don't update the current panel
                // Update the panel's config
                PanelConfig updated_config = other_panel->get_config();
                updated_config.symbol = current_symbol;
                panel_manager_->update_panel_config(id, updated_config);
                
                // If the panel has a specific method to set symbol, call it
                // This would require casting to specific panel types, but for now we'll update the config
              }
            }
          }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Duplicate Chart")) {
          // Duplicate the current chart panel
          if (panel_manager_) {
            // Get the panel's type and title to create a duplicate
            PanelType type = panel->get_config().type;
            std::string title = panel->get_config().title + " Copy";

            // Calculate new position for the duplicated panel
            int new_grid_x = panel->get_config().grid_x + 1;
            int new_grid_y = panel->get_config().grid_y;

            // Add the new panel with the same properties
            panel_manager_->add_panel(type, title, new_grid_x, new_grid_y,
                                      panel->get_config().grid_width,
                                      panel->get_config().grid_height);
          }
        }
        if (ImGui::MenuItem("Close Chart")) {
          // Close the current chart panel
          if (panel_manager_) {
            // Find the panel ID by comparing with all panels in the manager
            uint32_t panel_id = 0;
            auto all_panel_ids = panel_manager_->get_all_panel_ids();
            for (uint32_t id : all_panel_ids) {
              PanelBase* manager_panel = panel_manager_->get_panel_by_id(id);
              if (manager_panel == panel) {
                panel_id = id;
                break;
              }
            }
            if (panel_id != 0) {
              panel_manager_->remove_panel(panel_id);
            }
          }
        }
        break;

      case PanelType::WATCHLIST:
        ImGui::Text("Watchlist Actions:");
        ImGui::Separator();
        if (ImGui::MenuItem("Add Symbol")) {
          // Cast the panel to WatchlistPanel to access specific methods
          if (auto* watchlist_panel = dynamic_cast<WatchlistPanel*>(panel)) {
            watchlist_panel->focus_add_symbol_input();
          }
        }
        if (ImGui::MenuItem("Import Symbols")) {
          // Generic action
        }
        if (ImGui::MenuItem("Export Symbols")) {
          // Generic action
        }
        if (ImGui::BeginMenu("Sort by...")) {
          // Cast the panel to WatchlistPanel to access specific methods
          if (auto* watchlist_panel = dynamic_cast<WatchlistPanel*>(panel)) {
            if (ImGui::MenuItem("Symbol (Asc)")) {
              watchlist_panel->set_sorting(0, true);  // Sort by Symbol (column 0) ascending
            }
            if (ImGui::MenuItem("Symbol (Desc)")) {
              watchlist_panel->set_sorting(0, false);  // Sort by Symbol (column 0) descending
            }
            if (ImGui::MenuItem("Exchange (Asc)")) {
              watchlist_panel->set_sorting(1, true);  // Sort by Exchange (column 1) ascending
            }
            if (ImGui::MenuItem("Exchange (Desc)")) {
              watchlist_panel->set_sorting(1, false);  // Sort by Exchange (column 1) descending
            }
            if (ImGui::MenuItem("Last Price (Asc)")) {
              watchlist_panel->set_sorting(2, true);  // Sort by Last Price (column 2) ascending
            }
            if (ImGui::MenuItem("Last Price (Desc)")) {
              watchlist_panel->set_sorting(2, false);  // Sort by Last Price (column 2) descending
            }
            if (ImGui::MenuItem("Change % (Asc)")) {
              watchlist_panel->set_sorting(3, true);  // Sort by Change % (column 3) ascending
            }
            if (ImGui::MenuItem("Change % (Desc)")) {
              watchlist_panel->set_sorting(3, false);  // Sort by Change % (column 3) descending
            }
            if (ImGui::MenuItem("Change $ (Asc)")) {
              watchlist_panel->set_sorting(4, true);  // Sort by Change $ (column 4) ascending
            }
            if (ImGui::MenuItem("Change $ (Desc)")) {
              watchlist_panel->set_sorting(4, false);  // Sort by Change $ (column 4) descending
            }
            if (ImGui::MenuItem("Volume (Asc)")) {
              watchlist_panel->set_sorting(5, true);  // Sort by Volume (column 5) ascending
            }
            if (ImGui::MenuItem("Volume (Desc)")) {
              watchlist_panel->set_sorting(5, false);  // Sort by Volume (column 5) descending
            }
            if (ImGui::MenuItem("High (Asc)")) {
              watchlist_panel->set_sorting(6, true);  // Sort by High (column 6) ascending
            }
            if (ImGui::MenuItem("High (Desc)")) {
              watchlist_panel->set_sorting(6, false);  // Sort by High (column 6) descending
            }
            if (ImGui::MenuItem("Low (Asc)")) {
              watchlist_panel->set_sorting(7, true);  // Sort by Low (column 7) ascending
            }
            if (ImGui::MenuItem("Low (Desc)")) {
              watchlist_panel->set_sorting(7, false);  // Sort by Low (column 7) descending
            }
            if (ImGui::MenuItem("Open (Asc)")) {
              watchlist_panel->set_sorting(8, true);  // Sort by Open (column 8) ascending
            }
            if (ImGui::MenuItem("Open (Desc)")) {
              watchlist_panel->set_sorting(8, false);  // Sort by Open (column 8) descending
            }
            if (ImGui::MenuItem("VWAP (Asc)")) {
              watchlist_panel->set_sorting(9, true);  // Sort by VWAP (column 9) ascending
            }
            if (ImGui::MenuItem("VWAP (Desc)")) {
              watchlist_panel->set_sorting(9, false);  // Sort by VWAP (column 9) descending
            }
          }
          ImGui::EndMenu();
        }
        if (ImGui::MenuItem("Configure Columns")) {
          // Generic action
        }
        if (ImGui::MenuItem("Clear Watchlist")) {
          // Cast the panel to WatchlistPanel to access specific methods
          if (auto* watchlist_panel = dynamic_cast<WatchlistPanel*>(panel)) {
            watchlist_panel->clear_all_symbols();
          }
        }
        break;

      case PanelType::ORDERBOOK:
        ImGui::Text("Orderbook Actions:");
        ImGui::Separator();
        if (dynamic_cast<OrderbookPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Aggregation")) {
            // Call specific orderbook panel method
          }
          if (ImGui::MenuItem("Toggle Log Scale")) {
            // Call specific orderbook panel method
          }
          if (ImGui::MenuItem("Export Orderbook Data")) {
            // Call specific orderbook panel method
          }
          if (ImGui::MenuItem("Place Market Order")) {
            // Call specific orderbook panel method
          }
          if (ImGui::MenuItem("Place Limit Order")) {
            // Call specific orderbook panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Aggregation")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Log Scale")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Orderbook Data")) {
            // Generic action
          }
          if (ImGui::MenuItem("Place Market Order")) {
            // Generic action
          }
          if (ImGui::MenuItem("Place Limit Order")) {
            // Generic action
          }
        }
        break;

      case PanelType::FOOTPRINT_CHART:
        ImGui::Text("Footprint Chart Actions:");
        ImGui::Separator();
        if (dynamic_cast<FootprintPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Delta Bars")) {
            // Call specific footprint panel method
          }
          if (ImGui::MenuItem("Toggle Volume Bars")) {
            // Call specific footprint panel method
          }
          if (ImGui::MenuItem("Change Color Scheme")) {
            // Call specific footprint panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific footprint panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific footprint panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Delta Bars")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Volume Bars")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Color Scheme")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::VOLUME_PROFILE:
        ImGui::Text("Volume Profile Actions:");
        ImGui::Separator();
        if (dynamic_cast<VolumeProfilePanel*>(panel)) {
          if (ImGui::MenuItem("Toggle POC Line")) {
            // Call specific volume profile panel method
          }
          if (ImGui::MenuItem("Toggle Value Area")) {
            // Call specific volume profile panel method
          }
          if (ImGui::MenuItem("Change Time Period")) {
            // Call specific volume profile panel method
          }
          if (ImGui::MenuItem("Adjust Bin Size")) {
            // Call specific volume profile panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific volume profile panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific volume profile panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle POC Line")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Value Area")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Time Period")) {
            // Generic action
          }
          if (ImGui::MenuItem("Adjust Bin Size")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::TPO_PROFILE:
        ImGui::Text("TPO Profile Actions:");
        ImGui::Separator();
        if (dynamic_cast<TpoPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Session Blocks")) {
            // Call specific TPO panel method
          }
          if (ImGui::MenuItem("Change Session Type")) {
            // Call specific TPO panel method
          }
          if (ImGui::MenuItem("Toggle VWAP")) {
            // Call specific TPO panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific TPO panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific TPO panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Session Blocks")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Session Type")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle VWAP")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::HEATMAP:
        ImGui::Text("Heatmap Actions:");
        ImGui::Separator();
        if (dynamic_cast<DomSurfacePanel*>(panel)) {
          if (ImGui::MenuItem("Change Color Map")) {
            // Call specific heatmap panel method
          }
          if (ImGui::MenuItem("Toggle Interpolation")) {
            // Call specific heatmap panel method
          }
          if (ImGui::MenuItem("Adjust Brightness")) {
            // Call specific heatmap panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific heatmap panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific heatmap panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Change Color Map")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Interpolation")) {
            // Generic action
          }
          if (ImGui::MenuItem("Adjust Brightness")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::ALERTS:
        ImGui::Text("Alerts Actions:");
        ImGui::Separator();
        if (dynamic_cast<AlertsPanel*>(panel)) {
          if (ImGui::MenuItem("Create New Alert")) {
            // Call specific alerts panel method
          }
          if (ImGui::MenuItem("Enable All Alerts")) {
            // Call specific alerts panel method
          }
          if (ImGui::MenuItem("Disable All Alerts")) {
            // Call specific alerts panel method
          }
          if (ImGui::MenuItem("Clear Completed Alerts")) {
            // Call specific alerts panel method
          }
          if (ImGui::MenuItem("Delete Selected")) {
            // Call specific alerts panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Create New Alert")) {
            // Generic action
          }
          if (ImGui::MenuItem("Enable All Alerts")) {
            // Generic action
          }
          if (ImGui::MenuItem("Disable All Alerts")) {
            // Generic action
          }
          if (ImGui::MenuItem("Clear Completed Alerts")) {
            // Generic action
          }
          if (ImGui::MenuItem("Delete Selected")) {
            // Generic action
          }
        }
        break;

      case PanelType::TIME_AND_SALES:
        ImGui::Text("Time & Sales Actions:");
        ImGui::Separator();
        if (dynamic_cast<TimeAndSalesPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Call specific time and sales panel method
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Call specific time and sales panel method
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Call specific time and sales panel method
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Call specific time and sales panel method
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Call specific time and sales panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Generic action
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Generic action
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Generic action
          }
        }
        break;

      case PanelType::DEPTH_CHART:
        ImGui::Text("Depth Chart Actions:");
        ImGui::Separator();
        if (dynamic_cast<DepthChartPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Log Scale")) {
            // Call specific depth chart panel method
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Call specific depth chart panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific depth chart panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific depth chart panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Log Scale")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::PERFORMANCE_MONITOR:
        ImGui::Text("Performance Monitor Actions:");
        ImGui::Separator();
        if (dynamic_cast<PerformanceMonitorPanel*>(panel)) {
          if (ImGui::MenuItem("Refresh Data")) {
            // Call specific performance monitor panel method
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Call specific performance monitor panel method
          }
          if (ImGui::MenuItem("Reset Counters")) {
            // Call specific performance monitor panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific performance monitor panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Refresh Data")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset Counters")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::TRADING_ORDERS:
        ImGui::Text("Trading Orders Actions:");
        ImGui::Separator();
        if (dynamic_cast<TradingOrdersPanel*>(panel)) {
          if (ImGui::MenuItem("Place New Order")) {
            // Call specific trading orders panel method
          }
          if (ImGui::MenuItem("Cancel Selected")) {
            // Call specific trading orders panel method
          }
          if (ImGui::MenuItem("Cancel All")) {
            // Call specific trading orders panel method
          }
          if (ImGui::MenuItem("Modify Selected")) {
            // Call specific trading orders panel method
          }
          if (ImGui::MenuItem("Export Orders")) {
            // Call specific trading orders panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Place New Order")) {
            // Generic action
          }
          if (ImGui::MenuItem("Cancel Selected")) {
            // Generic action
          }
          if (ImGui::MenuItem("Cancel All")) {
            // Generic action
          }
          if (ImGui::MenuItem("Modify Selected")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Orders")) {
            // Generic action
          }
        }
        break;

      case PanelType::TRADING_POSITIONS:
        ImGui::Text("Trading Positions Actions:");
        ImGui::Separator();
        if (dynamic_cast<TradingPositionsPanel*>(panel)) {
          if (ImGui::MenuItem("Close Selected Position")) {
            // Call specific trading positions panel method
          }
          if (ImGui::MenuItem("Close All Positions")) {
            // Call specific trading positions panel method
          }
          if (ImGui::MenuItem("Reverse Position")) {
            // Call specific trading positions panel method
          }
          if (ImGui::MenuItem("Calculate PnL")) {
            // Call specific trading positions panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Close Selected Position")) {
            // Generic action
          }
          if (ImGui::MenuItem("Close All Positions")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reverse Position")) {
            // Generic action
          }
          if (ImGui::MenuItem("Calculate PnL")) {
            // Generic action
          }
        }
        break;

      case PanelType::RISK_METRICS:
        ImGui::Text("Risk Metrics Actions:");
        ImGui::Separator();
        if (dynamic_cast<RiskMetricsPanel*>(panel)) {
          if (ImGui::MenuItem("Refresh Metrics")) {
            // Call specific risk metrics panel method
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Call specific risk metrics panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific risk metrics panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Refresh Metrics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::METRICS:
        ImGui::Text("Metrics Actions:");
        ImGui::Separator();
        if (dynamic_cast<MetricsPanel*>(panel)) {
          if (ImGui::MenuItem("Refresh Data")) {
            // Call specific metrics panel method
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Call specific metrics panel method
          }
          if (ImGui::MenuItem("Change Time Range")) {
            // Call specific metrics panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific metrics panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Refresh Data")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Metrics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Time Range")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::SCATTER_PLOT:
        ImGui::Text("Scatter Plot Actions:");
        ImGui::Separator();
        if (dynamic_cast<ScatterPlotPanel*>(panel)) {
          if (ImGui::MenuItem("Change Point Style")) {
            // Call specific scatter plot panel method
          }
          if (ImGui::MenuItem("Toggle Grid")) {
            // Call specific scatter plot panel method
          }
          if (ImGui::MenuItem("Toggle Legend")) {
            // Call specific scatter plot panel method
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Call specific scatter plot panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific scatter plot panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific scatter plot panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Change Point Style")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Grid")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Legend")) {
            // Generic action
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::HISTOGRAM:
        ImGui::Text("Histogram Actions:");
        ImGui::Separator();
        if (dynamic_cast<HistogramPanel*>(panel)) {
          if (ImGui::MenuItem("Change Bin Count")) {
            // Call specific histogram panel method
          }
          if (ImGui::MenuItem("Toggle Normalization")) {
            // Call specific histogram panel method
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Call specific histogram panel method
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Call specific histogram panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific histogram panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific histogram panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Change Bin Count")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Normalization")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Generic action
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::TIME_SERIES:
        ImGui::Text("Time Series Actions:");
        ImGui::Separator();
        if (dynamic_cast<TimeSeriesPanel*>(panel)) {
          if (ImGui::MenuItem("Add Overlay")) {
            // Call specific time series panel method
          }
          if (ImGui::MenuItem("Toggle Grid")) {
            // Call specific time series panel method
          }
          if (ImGui::MenuItem("Toggle Legend")) {
            // Call specific time series panel method
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Call specific time series panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific time series panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific time series panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Add Overlay")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Grid")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Legend")) {
            // Generic action
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::TIME_STATISTICS:
        ImGui::Text("Time Statistics Actions:");
        ImGui::Separator();
        if (dynamic_cast<TimeStatisticsPanel*>(panel)) {
          if (ImGui::MenuItem("Refresh Data")) {
            // Call specific time statistics panel method
          }
          if (ImGui::MenuItem("Export Statistics")) {
            // Call specific time statistics panel method
          }
          if (ImGui::MenuItem("Change Time Range")) {
            // Call specific time statistics panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific time statistics panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Refresh Data")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Statistics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Time Range")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::TIME_HISTOGRAM:
        ImGui::Text("Time Histogram Actions:");
        ImGui::Separator();
        if (dynamic_cast<TimeHistogramPanel*>(panel)) {
          if (ImGui::MenuItem("Change Time Interval")) {
            // Call specific time histogram panel method
          }
          if (ImGui::MenuItem("Toggle Normalization")) {
            // Call specific time histogram panel method
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Call specific time histogram panel method
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Call specific time histogram panel method
          }
          if (ImGui::MenuItem("Reset View")) {
            // Call specific time histogram panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific time histogram panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Change Time Interval")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Normalization")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Cumulative")) {
            // Generic action
          }
          if (ImGui::MenuItem("Fit to View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Reset View")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::SCREENER:
        ImGui::Text("Screener Actions:");
        ImGui::Separator();
        if (dynamic_cast<ScreenerPanel*>(panel)) {
          if (ImGui::MenuItem("Add Filter")) {
            // Call specific screener panel method
          }
          if (ImGui::MenuItem("Edit Filters")) {
            // Call specific screener panel method
          }
          if (ImGui::MenuItem("Clear Filters")) {
            // Call specific screener panel method
          }
          if (ImGui::MenuItem("Sort by Column")) {
            // Call specific screener panel method
          }
          if (ImGui::MenuItem("Export Results")) {
            // Call specific screener panel method
          }
          if (ImGui::MenuItem("Refresh Data")) {
            // Call specific screener panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Add Filter")) {
            // Generic action
          }
          if (ImGui::MenuItem("Edit Filters")) {
            // Generic action
          }
          if (ImGui::MenuItem("Clear Filters")) {
            // Generic action
          }
          if (ImGui::MenuItem("Sort by Column")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Results")) {
            // Generic action
          }
          if (ImGui::MenuItem("Refresh Data")) {
            // Generic action
          }
        }
        break;

      case PanelType::LOG_PANEL:
        ImGui::Text("Log Panel Actions:");
        ImGui::Separator();
        if (dynamic_cast<LogPanel*>(panel)) {
          if (ImGui::MenuItem("Clear Logs")) {
            // Call specific log panel method
          }
          if (ImGui::MenuItem("Export Logs")) {
            // Call specific log panel method
          }
          if (ImGui::MenuItem("Filter by Level")) {
            // Call specific log panel method
          }
          if (ImGui::MenuItem("Toggle Timestamps")) {
            // Call specific log panel method
          }
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Call specific log panel method
          }
          if (ImGui::MenuItem("Find in Logs")) {
            // Call specific log panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Clear Logs")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Logs")) {
            // Generic action
          }
          if (ImGui::MenuItem("Filter by Level")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Timestamps")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Generic action
          }
          if (ImGui::MenuItem("Find in Logs")) {
            // Generic action
          }
        }
        break;

      case PanelType::CHART_REPLAY:
        ImGui::Text("Chart Replay Actions:");
        ImGui::Separator();
        if (dynamic_cast<ChartReplayPanel*>(panel)) {
          if (ImGui::MenuItem("Start Replay")) {
            // Call specific chart replay panel method
          }
          if (ImGui::MenuItem("Pause Replay")) {
            // Call specific chart replay panel method
          }
          if (ImGui::MenuItem("Stop Replay")) {
            // Call specific chart replay panel method
          }
          if (ImGui::MenuItem("Change Speed")) {
            // Call specific chart replay panel method
          }
          if (ImGui::MenuItem("Jump to Time")) {
            // Call specific chart replay panel method
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Call specific chart replay panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Start Replay")) {
            // Generic action
          }
          if (ImGui::MenuItem("Pause Replay")) {
            // Generic action
          }
          if (ImGui::MenuItem("Stop Replay")) {
            // Generic action
          }
          if (ImGui::MenuItem("Change Speed")) {
            // Generic action
          }
          if (ImGui::MenuItem("Jump to Time")) {
            // Generic action
          }
          if (ImGui::MenuItem("Take Screenshot")) {
            // Generic action
          }
        }
        break;

      case PanelType::HISTORICAL_TIME_SALES:
        ImGui::Text("Historical Time & Sales Actions:");
        ImGui::Separator();
        if (dynamic_cast<HistoricalTimeSalesPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Call specific historical time and sales panel method
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Call specific historical time and sales panel method
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Call specific historical time and sales panel method
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Call specific historical time and sales panel method
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Call specific historical time and sales panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Generic action
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Generic action
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Generic action
          }
        }
        break;

      case PanelType::TAPE:
        ImGui::Text("Tape Actions:");
        ImGui::Separator();
        if (dynamic_cast<TapePanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Call specific tape panel method
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Call specific tape panel method
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Call specific tape panel method
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Call specific tape panel method
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Call specific tape panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Auto Scroll")) {
            // Generic action
          }
          if (ImGui::MenuItem("Clear Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Filter Buys/Sells")) {
            // Generic action
          }
          if (ImGui::MenuItem("Highlight Large Trades")) {
            // Generic action
          }
          if (ImGui::MenuItem("Export Trade Data")) {
            // Generic action
          }
        }
        break;

      case PanelType::STATUS_BAR:
        ImGui::Text("Status Bar Actions:");
        ImGui::Separator();
        if (dynamic_cast<StatusBarPanel*>(panel)) {
          if (ImGui::MenuItem("Toggle Connection Status")) {
            // Call specific status bar panel method
          }
          if (ImGui::MenuItem("Toggle Performance Metrics")) {
            // Call specific status bar panel method
          }
          if (ImGui::MenuItem("Toggle Time Display")) {
            // Call specific status bar panel method
          }
          if (ImGui::MenuItem("Configure Display Format")) {
            // Call specific status bar panel method
          }
        } else {
          // Fallback for when cast fails - still allow generic actions
          if (ImGui::MenuItem("Toggle Connection Status")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Performance Metrics")) {
            // Generic action
          }
          if (ImGui::MenuItem("Toggle Time Display")) {
            // Generic action
          }
          if (ImGui::MenuItem("Configure Display Format")) {
            // Generic action
          }
        }
        break;

      case PanelType::STRATEGY_BUILDER:
        ImGui::Text("Strategy Builder Actions:");
        ImGui::Separator();
        if (ImGui::MenuItem("New Strategy")) {
          // Generic action for strategy builder
        }
        if (ImGui::MenuItem("Load Strategy")) {
          // Generic action for strategy builder
        }
        if (ImGui::MenuItem("Save Strategy")) {
          // Generic action for strategy builder
        }
        if (ImGui::MenuItem("Run Backtest")) {
          // Generic action for strategy builder
        }
        if (ImGui::MenuItem("Export Strategy")) {
          // Generic action for strategy builder
        }
        break;

      default:
        ImGui::Text("Generic Actions:");
        ImGui::Separator();
        if (ImGui::MenuItem("Export Data")) {
          // Generic action
        }
        if (ImGui::MenuItem("Take Screenshot")) {
          // Generic action
        }
        if (ImGui::MenuItem("Close Panel")) {
          // Generic action
        }
        break;
    }

    // Add global panel actions at the bottom of each context menu
    ImGui::Separator();

    // Find the panel ID by comparing with all panels in the manager
    uint32_t panel_id = 0;
    if (panel_manager_) {
      auto all_panel_ids = panel_manager_->get_all_panel_ids();
      for (uint32_t id : all_panel_ids) {
        PanelBase* manager_panel = panel_manager_->get_panel_by_id(id);
        if (manager_panel == panel) {
          panel_id = id;
          break;
        }
      }
    }

    // Global Panel Actions - Available for all panel types
    if (ImGui::MenuItem("Duplicate Panel")) {
      // Call the panel manager to duplicate this panel
      if (panel_manager_ && panel_id != 0) {
        // Get the panel's type and title to create a duplicate
        PanelType type = panel->get_config().type;
        std::string title = panel->get_config().title + " Copy";

        // Calculate new position for the duplicated panel to avoid overlap
        int new_grid_x = panel->get_config().grid_x + 1;
        int new_grid_y = panel->get_config().grid_y;

        // Add the new panel with the same properties
        panel_manager_->add_panel(type, title, new_grid_x, new_grid_y,
                                  panel->get_config().grid_width, panel->get_config().grid_height);
      }
    }
    if (ImGui::MenuItem("Apply Symbol to All")) {
      // Apply the current panel's symbol to all other panels
      if (panel_manager_ && !panel->get_config().symbol.empty()) {
        // Get the current symbol from this panel
        std::string current_symbol = panel->get_config().symbol;
        
        // Iterate through all panels and update their symbols
        auto all_panel_ids = panel_manager_->get_all_panel_ids();
        for (uint32_t id : all_panel_ids) {
          if (id != panel_id) {  // Don't update the current panel
            PanelBase* other_panel = panel_manager_->get_panel_by_id(id);
            if (other_panel) {
              // Update the panel's config
              PanelConfig updated_config = other_panel->get_config();
              updated_config.symbol = current_symbol;
              panel_manager_->update_panel_config(id, updated_config);
              
              // If the panel has a specific method to set symbol, call it
              // This would require casting to specific panel types, but for now we'll update the config
            }
          }
        }
      }
    }
    if (ImGui::MenuItem("Screenshot")) {
      // Take a screenshot of the current panel
      if (panel_manager_) {
        // Generate a unique filename based on panel type and timestamp
        std::string panel_type_name = panel->get_config().title;
        // Sanitize the panel name for use in filename
        std::replace(panel_type_name.begin(), panel_type_name.end(), ' ', '_');
        std::replace(panel_type_name.begin(), panel_type_name.end(), '/', '_');
        
        // Call the screenshot utility to capture the panel
        ScreenshotUtility::capture_panel_screenshot(panel, panel_type_name);
      }
    }
    if (ImGui::MenuItem("Close Panel")) {
      // Call the panel manager to remove this panel
      if (panel_manager_ && panel_id != 0) {
        panel_manager_->remove_panel(panel_id);
      }
    }
    if (ImGui::MenuItem("Settings")) {
      // Call the panel's open_settings() method to open the settings dialog
      if (panel) {
        panel->open_settings();
      }
    }

    ImGui::EndPopup();
  }
}

}  // namespace BTQuant