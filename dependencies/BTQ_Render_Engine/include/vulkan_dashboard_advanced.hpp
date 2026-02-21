#pragma once

// ============================================================================
// BTQuant Vulkan Dashboard - Modular Header
// Streamlined from 1189 lines to ~250 lines by extracting classes into:
// - trading/order_manager.hpp, position_manager.hpp, risk_assessment.hpp
// - analytics/technical_analysis.hpp
// - ui/ui_base.hpp
// ============================================================================

/**
 * @file vulkan_dashboard_advanced.hpp
 * @brief Advanced Vulkan-based trading dashboard for BTQuant platform
 *
 * This module implements a high-performance trading dashboard using Vulkan for
 * rendering and supports real-time market data visualization, trading
 * analytics, and algorithmic trading capabilities.
 *
 * Features:
 * - Vulkan-accelerated rendering
 * - Real-time market data processing
 * - Advanced charting and analytics
 * - Multi-exchange connectivity
 * - Risk management integration
 * - Customizable layouts and themes
 */

// Include modular headers
#include <imgui.h>

#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <print>
#include <string>
#include <vector>

#include "analytics/technical_analysis.hpp"
#include "components/VulkanSynchronization.h"
#include "components/theme_manager.hpp"

#include "market_data_processor.hpp"
#include "ui/ui_base.hpp"
#include "vulkan/lob_heatmap_compute_pipeline.hpp"
#include "vulkan/ssbo_snapshot_updater.hpp"
#include "vulkan_base_types.hpp"

namespace BTQuant {

// Global RenderEngine namespace for data structures
namespace RenderEngine {
struct OrderBookLevel {
  double price;
  double size;
};
}  // namespace RenderEngine

// Helper: Convert ImVec4 to glm::vec4
inline glm::vec4 to_glm(const ImVec4& v) { return glm::vec4(v.x, v.y, v.z, v.w); }

// ============================================================================
// Gesture \u0026 Touch Types
// ============================================================================

struct TouchPoint {
  int id;
  glm::vec2 position;
  std::chrono::high_resolution_clock::time_point timestamp;
};

enum class GestureType { Tap, DoubleTap, LongPress, Swipe, Pinch, Rotate };

struct GestureEvent {
  GestureType type;
  glm::vec2 center;
  float scale;
  float rotation;
  glm::vec2 velocity;
};

enum class KeyModifier : uint32_t {
  None = 0,
  Shift = 1 << 0,
  Ctrl = 1 << 1,
  Alt = 1 << 2,
  Super = 1 << 3
};

// ============================================================================
// UI Utilities
// ============================================================================

enum class LogLevel { Debug, Info, Warning, Error, Critical };

struct ScreenerResult {
  std::string symbol;
  double price;
  double change_24h = 0;
  double volume_24h = 0;
  double vol_spike_ratio = 1.0;
};

struct WatchlistEntry {
  std::string symbol;
  double price = 0;
  double change_24h = 0;
  double volume_24h = 0;
  uint64_t last_update_ts = 0;
};

struct CandlestickVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
  float open;
  float close;
};

enum class AlertCondition { PRICE_ABOVE, PRICE_BELOW, VOLUME_ABOVE };

// Different AlertRule structure for this specific context
struct DashboardAlertRule {
  std::string symbol;
  AlertCondition condition;
  double target_value;
  bool is_triggered;
};

class AlertManager {
 public:
  void update(float dt);
  void add_alert(const DashboardAlertRule& rule);
  void check_alerts(const std::string& symbol, double price);
  std::vector<DashboardAlertRule> get_alerts();
  void remove_alert(size_t index);
};

// Forward declarations
class VulkanDashboard;
class QuantWorkspaceComponent;

class ResizablePanel : public UIComponent {
 public:
  ResizablePanel(const glm::vec2& p, const glm::vec2& s, const std::string& t)
      : UIComponent(p, s), title_(t) {}
  void set_resizable(bool r) { resizable_ = r; }
  void set_snap_to_grid(bool s, float g) {
    snap_to_grid_ = s;
    grid_size_ = g;
  }
  void update(float) override {}
  void render_gui() override {}

 private:
  std::string title_;
  bool resizable_ = true;
  bool snap_to_grid_ = false;
  float grid_size_ = 10.0f;
};

class LayoutManager {
 public:
  void create_default_layouts();
  void save_layout(const std::string& name, const std::string& desc);
};

class SearchEngine {
 public:
  void index_symbol(const std::string& s, const std::string& d);
  std::vector<std::string> search(const std::string& q);
};

class DataFilter {
 public:
  enum class FilterType { Text, Numeric, Boolean };
  enum class ComparisonOperator { Equals, NotEquals, Greater, Less, Contains };
  struct FilterCriteria {
    std::string field_name;
    FilterType type;
    ComparisonOperator operator_;
    std::string value;
  };
  void add_filter(const FilterCriteria& c);
};

// ============================================================================
// Legacy Component Declarations (Archived but declared for compatibility)
// ============================================================================

struct StrategyControlComponent : public UIComponent {
  StrategyControlComponent(const glm::vec2& p, const glm::vec2& s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore* core) override;
};

struct RiskManagerComponent : public UIComponent {
  RiskManagerComponent(const glm::vec2& p, const glm::vec2& s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore* core) override;
};

struct TradingInterfaceComponent : public UIComponent {
  TradingInterfaceComponent(const glm::vec2& p, const glm::vec2& s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore* core) override;
};

// Forward declarations for other archived components
struct TapeComponent;
struct OrderManagementComponent;
struct PositionPanelComponent;
struct MarketOverviewPanel;
struct WatchlistComponent;
struct LogDisplayComponent;
struct HeatmapComponent;
struct MarketScreenerComponent;
struct AlertComponent;
struct DataGridComponent;
struct MarketDepthChartComponent;

// ============================================================================
// Main Dashboard Class
// ============================================================================

/**
 * @class VulkanDashboard
 * @brief Main dashboard class that manages the entire trading interface
 *
 * The VulkanDashboard class serves as the central hub for the trading
 * application, coordinating between the Vulkan rendering engine, market data
 * processing, user interface components, and trading systems. It handles window
 * management, rendering loops, event processing, and resource lifecycle
 * management.
 */
class VulkanDashboard {
 public:
  /**
   * @brief Construct a new VulkanDashboard object
   * @param width Window width in pixels
   * @param height Window height in pixels
   * @param bridge Shared pointer to the data bridge for market data
   * @param processor Shared pointer to the market data processor
   * @param config Configuration object for dashboard settings
   */
  VulkanDashboard(uint32_t width, uint32_t height, 
                  std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                  const VulkanDashboardConfig& config);

  /// @brief Destructor - cleans up all allocated resources
  ~VulkanDashboard();

  /**
   * @brief Initialize the dashboard and all its components
   * @return Expected<void, std::string> Success or error message
   */
  [[nodiscard]] std::expected<void, std::string> initialize();

  /// @brief Clean up and shut down the dashboard
  void shutdown();

  /// @brief Render a single frame of the dashboard
  void render_frame();

  /// @brief Handle input events (keyboard, mouse, etc.)
  void handle_events();

  /// @brief Check if the dashboard window should be closed
  bool should_close() const;

  /// @brief Set the currently active trading symbol
  void set_active_symbol(const std::string& s) { active_symbol_ = s; }

  /// @brief Get the currently active trading symbol
  std::string get_active_symbol() const { return active_symbol_; }

  /// @brief Get access to the underlying Vulkan core
  VulkanCore* get_vulkan_core() { return vulkan_core_.get(); }

  /// @brief Set a callback to render custom ImGui menu items
  void set_custom_menubar_callback(std::function<void()> callback) {
    custom_menubar_callback_ = callback;
  }

  /// @brief Get the workspace component
  QuantWorkspaceComponent* get_workspace_component() { return workspace_.get(); }

  /// @brief Toggle performance overlay
  void set_show_performance_overlay(bool show) { show_performance_overlay_ = show; }

 private:
  /// @brief Initialize all UI components
  void init_components();

  /// @brief Initialize the GLFW window
  void init_window();

  /// @brief Poll market data and feed it to the microstructure renderer
  void pollDataToRenderer();

  /// @brief Render the internal performance overlay
  void render_performance_overlay();

  /// @brief Render the layout indicator showing active layout
  void render_layout_indicator();

  uint32_t width_, height_;
  
  std::shared_ptr<RenderEngine::MarketDataProcessor> market_data_processor_;
  VulkanDashboardConfig config_;
  std::string active_symbol_ = "BTC-USDT";
  std::unique_ptr<VulkanCore> vulkan_core_;
  std::unique_ptr<QuantWorkspaceComponent> workspace_;
  std::unique_ptr<VulkanSyncContext> sync_context_;
  std::unique_ptr<TimelineSemaphore> timeline_semaphore_;

  // Heatmap compute pipeline
  LobHeatmapComputePipeline heatmap_pipeline_;
  SsboSnapshotUpdater ssbo_updater_;
  VkDescriptorSet heatmap_texture_ = VK_NULL_HANDLE;  // Registered texture descriptor set

  // Customization
  std::function<void()> custom_menubar_callback_;
  bool show_performance_overlay_ = false;

  uint32_t current_image_index_ = 0;
  bool is_running_ = true;
  bool window_resized_ = false;

  /**
   * @brief Callback for when the window framebuffer is resized
   * @param window Pointer to the GLFW window
   * @param width New width
   * @param height New height
   */
  static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
  GLFWwindow* window_ = nullptr;
};

}  // namespace BTQuant