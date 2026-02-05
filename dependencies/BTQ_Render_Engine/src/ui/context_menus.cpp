#include "ui/context_menus.hpp"
#include "imgui.h"
#include "implot.h"
#include <string>
#include <functional>

// Need to include panel_manager.hpp to use its methods
#include "components/panel_manager.hpp"

namespace BTQuant {

// Context menu manager implementation
ContextMenuManager::ContextMenuManager(PanelManager* panel_manager) : panel_manager_(panel_manager) {
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
                if (ImGui::MenuItem("Set Timeframe")) {
                    // Open timeframe selection dialog
                    if (panel_manager_) {
                        // This would typically send an event to change the chart's timeframe
                    }
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
                    // Generic action
                }
                if (ImGui::MenuItem("Import Symbols")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Export Symbols")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Sort by Change %")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Sort by Volume")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Configure Columns")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Clear Watchlist")) {
                    // Generic action
                }
                break;
                
            case PanelType::ORDERBOOK:
                ImGui::Text("Orderbook Actions:");
                ImGui::Separator();
                if (ImGui::MenuItem("Center View")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Reset Depth")) {
                    // Generic action
                }
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
                break;
                
            case PanelType::FOOTPRINT_CHART:
                ImGui::Text("Footprint Chart Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::VOLUME_PROFILE:
                ImGui::Text("Volume Profile Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TPO_PROFILE:
                ImGui::Text("TPO Profile Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::HEATMAP:
                ImGui::Text("Heatmap Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::ALERTS:
                ImGui::Text("Alerts Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TIME_AND_SALES:
                ImGui::Text("Time & Sales Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::DEPTH_CHART:
                ImGui::Text("Depth Chart Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::PERFORMANCE_MONITOR:
                ImGui::Text("Performance Monitor Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TRADING_ORDERS:
                ImGui::Text("Trading Orders Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TRADING_POSITIONS:
                ImGui::Text("Trading Positions Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::RISK_METRICS:
                ImGui::Text("Risk Metrics Actions:");
                ImGui::Separator();
                if (ImGui::MenuItem("Refresh Metrics")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Export Metrics")) {
                    // Generic action
                }
                if (ImGui::MenuItem("Take Screenshot")) {
                    // Generic action
                }
                break;
                
            case PanelType::METRICS:
                ImGui::Text("Metrics Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::SCATTER_PLOT:
                ImGui::Text("Scatter Plot Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::HISTOGRAM:
                ImGui::Text("Histogram Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TIME_SERIES:
                ImGui::Text("Time Series Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TIME_STATISTICS:
                ImGui::Text("Time Statistics Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::TIME_HISTOGRAM:
                ImGui::Text("Time Histogram Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::SCREENER:
                ImGui::Text("Screener Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::LOG_PANEL:
                ImGui::Text("Log Panel Actions:");
                ImGui::Separator();
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
                break;
                
            case PanelType::CHART_REPLAY:
                ImGui::Text("Chart Replay Actions:");
                ImGui::Separator();
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

        if (ImGui::MenuItem("Close Panel")) {
            // Call the panel manager to remove this panel
            if (panel_manager_ && panel_id != 0) {
                panel_manager_->remove_panel(panel_id);
            }
        }
        if (ImGui::MenuItem("Duplicate Panel")) {
            // Call the panel manager to duplicate this panel
            if (panel_manager_ && panel_id != 0) {
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
        if (ImGui::MenuItem("Panel Settings")) {
            // Call the panel's settings method if available
            panel->open_settings();
        }

        ImGui::EndPopup();
    }
}

} // namespace BTQuant