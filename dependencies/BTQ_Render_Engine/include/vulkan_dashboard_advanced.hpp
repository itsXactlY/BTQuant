#pragma once

// ============================================================================
// Standard Library & Third-Party Includes
// ============================================================================
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Math library
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Platform-specific Vulkan includes
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>

// X11 includes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XInput2.h>

// Internal includes - these MUST come before any clashing macros are undefined
#include "CandlePipeline.h"
#include "DashboardLayer.h"
#include "OffscreenChartRenderer.h"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"

// Undefine ONLY the most clashing X11 macros.
#ifdef None
#undef None
#endif
#ifdef Success
#undef Success
#endif
#ifdef Status
#undef Status
#endif
#ifdef Bool
#undef Bool
#endif

// Concurrent data structures
#include <concurrentqueue.h>

// ImGui
#include "imgui.h"

namespace BTQuant {

// Usage aliases
using RenderEngine::OrderbookData;
using RenderEngine::TradeData;
using OrderBookData = RenderEngine::OrderbookData;

// Forward declarations
class VulkanDashboard;

// ============================================================================
// Enums
// ============================================================================

enum class InputEventType {
  KeyDown,
  KeyUp,
  MouseMove,
  MouseButton,
  Scroll,
  Resize,
  Focus,
  Blur
};

enum class MouseButton { Left, Right, Middle, None };

enum class KeyModifier : uint32_t {
  None = 0,
  Shift = 1 << 0,
  Ctrl = 1 << 1,
  Alt = 1 << 2,
  Super = 1 << 3
};

enum class GestureType {
  None,
  Pan,
  Zoom,
  Rotate,
  Swipe,
  Tap,
  LongPress,
  Pinch
};

enum class LogLevel { Debug, Info, Warning, Error, Critical };

// ============================================================================
// Structs
// ============================================================================

struct TouchPoint {
  int id;
  glm::vec2 position;
  glm::vec2 start_position;
  float pressure;
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct GestureEvent {
  GestureType type;
  glm::vec2 position;
  glm::vec2 delta;
  float scale;
  float rotation;
  glm::vec2 center;
  float velocity;
};

struct InputEvent {
  InputEventType type;
  glm::vec2 position;
  glm::vec2 delta;
  float scroll_delta;
  int key;
  int modifiers;
  bool pressed;
  MouseButton mouse_button;
  TouchPoint touch_point;
  GestureEvent gesture_event;
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct DepthBarVertex {
  glm::vec2 position;
  glm::vec2 size;
  glm::vec4 color;
};

using CandlestickVertex = DepthBarVertex;

struct OrderBookTextVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
};

struct OrderBookUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec2 component_size;
  glm::vec2 component_position;
  float row_height;
  float max_size_for_bars;
  float spread_highlight_intensity;
};

struct UIUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::mat4 model;
  glm::vec2 viewport_size;
};

struct HeatmapData {
  float value;
  glm::vec4 color;
};

struct ScreenerResult {
  std::string symbol;
  double price;
  double change_24h = 0.0;
  double volume_24h = 0.0;
  double vol_spike_ratio = 0.0;
};

struct WatchlistEntry {
  std::string symbol;
  double price;
  double change_24h;
  double volume_24h;
  uint64_t last_update_ts;
};

struct TapeEntry {
  uint64_t timestamp_us;
  double price;
  double size;
  bool is_buy;
  bool is_large_trade;
  bool is_whale_trade;
  float delta;
};

struct PositionData {
  std::string symbol;
  float quantity;
  float entry_price;
  float pnl;
  float pnl_percent;
  float liquidation_price;
};

struct TickerData {
  std::string symbol;
  float price;
  float change_pct;
};

struct StrategyData {
  std::string name;
  bool active;
  float pnl;
  float draw_down;
  int trades;
  std::string status;
};

struct DashboardTheme {
  ImVec4 background_main;
  ImVec4 background_panel;
  ImVec4 accent_primary;
  ImVec4 accent_secondary;
  ImVec4 text_primary;
  ImVec4 text_secondary;
  ImVec4 price_up;
  ImVec4 price_down;
  ImVec4 border_color;
};

// ============================================================================
// Base Class
// ============================================================================

class UIComponent {
public:
  UIComponent(const glm::vec2 &position, const glm::vec2 &size)
      : position_(position), size_(size), visible_(true), is_dirty_(true),
        vulkan_core_(nullptr) {}
  virtual ~UIComponent() = default;

  virtual void initialize_vulkan_resources(VulkanCore *vulkan_core) = 0;
  virtual void update(float delta_time) = 0;
  virtual void render(VkCommandBuffer cmd) = 0;
  virtual void render_gui() = 0;
  virtual void handle_input(const InputEvent &event) = 0;

  // Data handlers
  virtual void handle_trade(const BTQuant::RenderEngine::TradeData &trade) {}
  virtual void
  handle_orderbook(const BTQuant::RenderEngine::OrderbookData &data) {}
  virtual void clear_data() {}

  void set_position(const glm::vec2 &pos) {
    position_ = pos;
    mark_dirty();
  }
  void set_size(const glm::vec2 &s) {
    size_ = s;
    mark_dirty();
  }
  virtual void set_target_symbol(const std::string &symbol) {}

  void set_visible(bool v) { visible_ = v; }
  bool is_visible() const { return visible_; }

protected:
  glm::vec2 position_;
  glm::vec2 size_;
  bool visible_;
  bool is_dirty_;
  VulkanCore *vulkan_core_;
  void mark_dirty() { is_dirty_ = true; }
};

// ============================================================================
// VulkanDashboard
// ============================================================================

class VulkanDashboard {
public:
  VulkanDashboard(
      uint32_t width, uint32_t height,
      std::shared_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge,
      const VulkanDashboardConfig &config);
  ~VulkanDashboard();

  void initialize();
  void shutdown();

  void run();
  void stop();
  bool run_frame();
  void render_frame();

  void add_component(std::shared_ptr<UIComponent> component);
  void remove_component(std::shared_ptr<UIComponent> component);

  // Synchronization
  void synchronize_market_data();

  // Public accessors
  VulkanCore *get_vulkan_core() const { return vulkan_core_.get(); }
  Display *get_display() { return display_; }
  Window get_x_window() { return x_window_; }
  bool should_close() const { return should_close_; }
  void set_active_symbol(const std::string &symbol);
  std::string get_active_symbol() const { return active_symbol_; }

private:
  void init_x11();
  void init_vulkan();
  void init_imgui();
  void init_components();
  void process_events();

  VulkanDashboardConfig config_;
  uint32_t width_;
  uint32_t height_;
  std::vector<std::shared_ptr<UIComponent>> components_;

  // X11
  Display *display_ = nullptr;
  Window x_window_;
  Atom wm_delete_window_;

  // Vulkan
  std::unique_ptr<VulkanCore> vulkan_core_;
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

  // State
  bool should_close_ = false;
  std::string active_symbol_;

  // Dependencies
  std::unique_ptr<OffscreenChartRenderer> offscreen_renderer_;
  std::shared_ptr<BTQuant::RenderEngine::HotSpineDataBridge> hotspine_bridge_;
  std::unique_ptr<CandlePipeline> candle_pipeline_;

  friend class UIComponent;
};

// ============================================================================
// Helpers
// ============================================================================

inline glm::vec4 to_glm(const ImVec4 &v) {
  return glm::vec4(v.x, v.y, v.z, v.w);
}

// ============================================================================
// Component Classes
// ============================================================================

class HeatmapComponent : public UIComponent {
public:
  HeatmapComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~HeatmapComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void set_range(float min_val, float max_val);
  void clear_data() override;

private:
  void rebuild_geometry() {}
  void dispatch_compute_interpolation() {}
  void interpolate_color() {}

  struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_set_layout;
  } compute_;

  struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_set_layout;
  } render_;

  std::vector<HeatmapData> heatmap_data_;
};

class DataGridComponent : public UIComponent {
public:
  DataGridComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~DataGridComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void set_cell(int row, int col, const std::string &value);
  void clear_data() override;

private:
  struct CellData {
    std::string text;
    double numeric_value;
    bool is_numeric;
    bool highlight;
    ImVec4 color;
  };

  int rows_;
  int columns_;
  std::vector<std::vector<CellData>> grid_data_;
  DashboardTheme theme_;
};

class LogDisplayComponent : public UIComponent {
public:
  LogDisplayComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~LogDisplayComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void add_log(LogLevel level, const std::string &message);
  void clear_logs();
  void clear_data() override { clear_logs(); }

  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;

private:
  struct LogEntry {
    LogLevel level;
    std::string message;
    std::string timestamp;
  };

  std::deque<LogEntry> log_entries_;
  size_t max_entries_;
  LogLevel min_log_level_;
  std::string text_filter_;
  bool auto_scroll_;
  float scroll_offset_;
  DashboardTheme theme_;

  VkPipeline text_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
};

struct OrderBookLevel {
  double price;
  double size;
  float last_update_ts;
};

class OrderBookComponent : public UIComponent {
public:
  OrderBookComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~OrderBookComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void handle_orderbook(const OrderbookData &data) override;
  void set_target_symbol(const std::string &symbol) override {
    target_symbol_ = symbol;
  }
  void set_precision(int price_precision, int size_precision);

private:
  void update_orderbook(const OrderBookData &data);
  void setup_uniform_buffer(OrderBookUniformBuffer &ubo);
  void add_text_at_position(std::vector<OrderBookTextVertex> &vertices,
                            const std::string &text, float x, float y,
                            const glm::vec4 &color, float size);
  void add_centered_text(std::vector<OrderBookTextVertex> &vertices,
                         const std::string &text, float y,
                         const glm::vec4 &color, float size);

  std::string target_symbol_;
  struct {
    std::vector<OrderBookLevel> bids;
    std::vector<OrderBookLevel> asks;
    double spread;
    double mid_price;
    uint64_t timestamp;
  } current_data_;

  std::mutex data_mutex_;
  DashboardTheme theme_;

  // Vulkan resources
  VkPipeline bar_pipeline_ = VK_NULL_HANDLE;
  VkPipeline text_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout bar_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout text_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout bar_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout text_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet bar_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet text_descriptor_set_ = VK_NULL_HANDLE;

  BufferAllocation bar_vertex_buffer_;
  BufferAllocation text_vertex_buffer_;
  BufferAllocation bar_ubo_buffer_;
  BufferAllocation text_ubo_buffer_;
  BufferAllocation font_metrics_buffer_;

  VkImage font_image_ = VK_NULL_HANDLE;
  VkImageView font_image_view_ = VK_NULL_HANDLE;
  VkDeviceMemory font_memory_ = VK_NULL_HANDLE;
  VkSampler font_sampler_ = VK_NULL_HANDLE;
};

class MarketScreenerComponent : public UIComponent {
public:
  MarketScreenerComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketScreenerComponent() override = default;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

private:
  std::vector<ScreenerResult> results_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
};

class RealtimeChartComponent : public UIComponent {
public:
  RealtimeChartComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~RealtimeChartComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void handle_trade(const TradeData &trade) override;
  void handle_orderbook(const OrderbookData &data) override;
  void clear_data() override;

private:
  void update_textures();

  std::string symbol_;
  std::vector<CandleData> raw_candles_;
  std::recursive_mutex data_mutex_;

  VkDescriptorSet chart_texture_set_ = VK_NULL_HANDLE;
  VkDescriptorSet depth_texture_set_ = VK_NULL_HANDLE;

  // Offscreen rendering state
  uint32_t chart_width_ = 0;
  uint32_t chart_height_ = 0;

  // Pipeline from Orchestrator
  CandlePipeline *candle_pipeline_ = nullptr;

  // Zoom and pan state
  float view_zoom_ = 1.0f;
  float view_offset_ = 0.0f;
};

class WatchlistComponent : public UIComponent {
public:
  WatchlistComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~WatchlistComponent() override = default;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

  void add_symbol(const std::string &symbol);
  void update_quote(const std::string &symbol, double price, double change);

private:
  std::vector<WatchlistEntry> entries_;
  std::mutex data_mutex_;
  VulkanDashboard *dashboard_ = nullptr;
};

enum class AlertCondition { GreaterThan, LessThan, CrossAbove, CrossBelow };

struct AlertRule {
  std::string symbol;
  std::string field;
  AlertCondition condition;
  double value;
  bool active;
};

class AlertManager {
public:
  void add_rule(const AlertRule &rule);
  void check_alerts(const std::string &symbol, double price);
};

class AlertComponent : public UIComponent {
public:
  AlertComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
};

class RiskManagerComponent : public UIComponent {
public:
  RiskManagerComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

private:
  DashboardTheme theme_;
};

class TradingInterfaceComponent : public UIComponent {
public:
  TradingInterfaceComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
};

class TapeComponent : public UIComponent {
public:
  TapeComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~TapeComponent() override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void handle_trade(const RenderEngine::TradeData &trade) override;
  void clear_data() override;

private:
  std::string target_symbol_;
  std::deque<TapeEntry> entries_;
  float cumulative_delta_ = 0.0f;
  float large_trade_threshold_ = 5.0f;
  float whale_trade_threshold_ = 25.0f;
  std::mutex data_mutex_;
  DashboardTheme theme_;
};

class OrderManagementComponent : public UIComponent {
public:
  OrderManagementComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~OrderManagementComponent() override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
};

class PositionPanelComponent : public UIComponent {
public:
  PositionPanelComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~PositionPanelComponent() override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

  void rebuild_equity_geometry();

private:
  std::vector<PositionData> positions_;
  std::vector<float> equity_history_;
  float total_equity_ = 100000.0f;
  float available_balance_ = 85000.0f;
  DashboardTheme theme_;
  BufferAllocation equity_vertex_buffer_;
};

class MarketOverviewPanel : public UIComponent {
public:
  MarketOverviewPanel(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketOverviewPanel() override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

private:
  std::vector<TickerData> tickers_;
  float global_volume_ = 1500000000.0f;
  float system_latency_ms_ = 1.25f;
  DashboardTheme theme_;
};

class StrategyControlComponent : public UIComponent {
public:
  StrategyControlComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}

private:
  std::vector<StrategyData> strategies_;
};

class MarketDepthChartComponent : public UIComponent {
public:
  struct CurrentData {
    struct Level {
      double price;
      double size;
    };
    std::vector<Level> bids;
    std::vector<Level> asks;
  };

  MarketDepthChartComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketDepthChartComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void handle_orderbook(const OrderbookData &data) override;
  void set_target_symbol(const std::string &symbol) override {
    target_symbol_ = symbol;
  }

private:
  void rebuild_geometry();

  std::string target_symbol_;
  CurrentData current_data_;
  std::mutex data_mutex_;
  DashboardTheme theme_;

  BufferAllocation vertex_buffer_;
  uint32_t vertex_count_ = 0;
};

} // namespace BTQuant