#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "imgui.h"
#include "theme_manager.hpp"
#include "panel_settings_interface.hpp"

// Forward declaration to avoid circular dependency
namespace BTQuant {
    class ContextMenuManager;
}

namespace BTQuant {

enum class PanelType {
  CHART,
  METRICS,
  HEATMAP,
  HISTOGRAM,
  SCATTER_PLOT,
  TIME_SERIES,
  TRADING_ORDERS,
  TRADING_POSITIONS,
  RISK_METRICS,
  ALERTS,
  ORDERBOOK,
  WATCHLIST,
  SCREENER,
  TAPE,
  VOLUME_PROFILE,
  DEPTH_CHART,
  STATUS_BAR,
  LOG_PANEL,
  FOOTPRINT_CHART,
  TPO_PROFILE,
  PERFORMANCE_MONITOR,
  TIME_STATISTICS,
  TIME_HISTOGRAM,
  TIME_AND_SALES,
  HISTORICAL_TIME_SALES,
  CHART_REPLAY,
  RISK_ANALYZER,
  STRATEGY_BUILDER,
  OPTION_ANALYTICS,
  TABBED_GROUP
};

struct PanelConfig {
  std::string title = "Panel";
  PanelType type = PanelType::CHART;
  ImVec2 position = ImVec2(0, 0);
  ImVec2 size = ImVec2(400, 300);
  bool visible = true;
  bool minimized = false;  // Whether the panel is minimized/collapsed
  bool resizable = true;
  bool movable = true;
  int grid_x = 0;
  int grid_y = 0;
  int grid_width = 1;
  int grid_height = 1;
  std::string symbol = "";  // Trading symbol associated with the panel

  // Per-panel settings data
  std::string settings_key = "";  // Key for identifying panel-specific settings
};

/**
 * PanelBase - Base class for all UI panels
 *
 * C++26 Reactive Architecture:
 * - Panels subscribe to MarketDataProcessor for push notifications
 * - data_dirty_ flag set by notification callback (thread-safe atomic)
 * - render() checks consumeDirty() to know when to refresh data
 * - No polling timers needed - truly event-driven
 */
class PanelBase {
 public:
  PanelBase(const PanelConfig& config) : config_(config) {}
  virtual ~PanelBase() = default;

  virtual void update([[maybe_unused]] float dt) {}
  virtual void render() = 0;
  virtual void initialize() {}

  // Panel management
  void set_position(const ImVec2& pos) { config_.position = pos; }
  void set_size(const ImVec2& size) { config_.size = size; }
  void set_visible(bool visible) { config_.visible = visible; }
  void set_minimized(bool minimized) { config_.minimized = minimized; }
  void set_title(const std::string& title) { config_.title = title; }

  const PanelConfig& get_config() const { return config_; }
  PanelConfig& get_config() { return config_; }

  bool is_visible() const { return config_.visible; }
  bool is_minimized() const { return config_.minimized; }
  const std::string& get_title() const { return config_.title; }

  // Per-panel settings functionality
  virtual PanelSettingsInterface* get_settings_interface() { return nullptr; }
  virtual void open_settings() {}

  // Context menu functionality
  virtual void render_context_menu() {}  // Virtual method for context menu
  virtual void handle_context_menu(class ContextMenuManager& manager);  // Virtual method for context menu handling

 protected:
  PanelConfig config_;

  // C++26 Reactive Push Notification Support
  // Set by processor callback when new data arrives - atomic for thread safety
  std::atomic<bool> data_dirty_{true};  // Start dirty to force initial load
  uint64_t subscription_id_ = 0;        // ID from processor->subscribe()

  /**
   * Called by MarketDataProcessor notification callback
   * Thread-safe: uses release memory ordering for proper visibility
   */
  void markDirty() noexcept { data_dirty_.store(true, std::memory_order_release); }

  /**
   * Called in render() to check if data needs refresh
   * Atomically clears the flag and returns previous value
   * Uses acquire-release for proper synchronization with markDirty()
   */
  [[nodiscard]] bool consumeDirty() noexcept {
    return data_dirty_.exchange(false, std::memory_order_acq_rel);
  }

  // Helper methods for consistent styling
  void begin_panel_window();
  void end_panel_window();
  void render_panel_header();

  // Glass-morphism helpers wrappers
  void push_glass_style();
  void pop_glass_style();

  // Utility functions
  static const char* get_panel_type_name(PanelType type);
};

}  // namespace BTQuant