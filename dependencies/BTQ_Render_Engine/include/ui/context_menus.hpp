#pragma once

#include <functional>
#include <functional>  // for std::hash>
#include <unordered_map>

#include "../components/panel_base.hpp"

namespace BTQuant {

// Forward declaration for PanelManager to avoid circular dependency
class PanelManager;

// Hash function for enum class
struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

class ChartPanel;
class WatchlistPanel;
class OrderbookPanel;
class FootprintPanel;
class VolumeProfilePanel;
class TpoPanel;
class DomSurfacePanel;
class AlertsPanel;
class TimeAndSalesPanel;
class DepthChartPanel;
class PerformanceMonitorPanel;
class TradingOrdersPanel;
class TradingPositionsPanel;
class RiskMetricsPanel;
class MetricsPanel;
class ScatterPlotPanel;
class HistogramPanel;
class TimeSeriesPanel;
class TimeStatisticsPanel;
class TimeHistogramPanel;
class ScreenerPanel;
class LogPanel;
class ChartReplayPanel;

/**
 * @brief ContextMenuManager - Manages context menus for different UI elements
 *
 * Implements a centralized system for handling right-click context menus across
 * all panel types in the trading terminal. Each panel type has its own context
 * menu with relevant actions.
 */
class ContextMenuManager {
 public:
  ContextMenuManager(PanelManager* panel_manager = nullptr);

  /**
   * @brief Shows the appropriate context menu for the given panel
   * @param panel The panel to show context menu for
   */
  void show_context_menu(PanelBase* panel);

 private:
  void initialize_context_menus();

  // Generic context menu renderer for all panel types
  void render_generic_context_menu(PanelBase* panel, const char* popup_name);

  // Context menu handlers map
  std::unordered_map<PanelType, std::function<void(PanelBase*)>, EnumClassHash>
      context_menu_handlers_;

  // Storage for mouse position when context menu is opened
  struct {
    double x = 0.0;
    double y = 0.0;
  } clicked_mouse_pos_;

  // Pointer to the PanelManager to execute global actions
  PanelManager* panel_manager_;
};

}  // namespace BTQuant