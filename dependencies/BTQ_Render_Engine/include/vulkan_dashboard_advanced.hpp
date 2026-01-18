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
#ifndef VK_USE_PLATFORM_XLIB_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>

// X11 includes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XInput2.h>

// Internal includes
#include "CandlePipeline.h"
#include "DashboardLayer.h"
#include "OffscreenChartRenderer.h"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"

// Undefine clashing X11 macros
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

enum class AlertCondition {
  PRICE_ABOVE,
  PRICE_BELOW,
  VOLUME_ABOVE,
  GreaterThan,
  LessThan,
  CrossAbove,
  CrossBelow
};

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
  float intensity;
  uint32_t level_type;
};

struct LogTextVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
  uint32_t glyph_id;
  float font_size;
  uint32_t log_level;
};

using CandlestickVertex = DepthBarVertex;

struct OrderBookTextVertex {
  glm::vec2 position;
  glm::vec2 texcoord;
  glm::vec4 color;
  uint32_t glyph_id;
  float font_size;
};

struct OrderBookUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec2 component_size;
  glm::vec2 component_position;
  float row_height;
  float max_size_for_bars;
  float spread_highlight_intensity;
  float time;
  glm::vec4 bid_color;
  glm::vec4 ask_color;
  glm::vec4 spread_color;
  float animation_phase;
};

struct GlyphMetric {
  glm::vec4 atlas_coords;
  glm::vec2 bearing;
  float advance;
  float padding;
};

struct UIUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::mat4 model;
  glm::vec2 viewport_size;
  glm::vec2 dpi_scale;
  glm::vec4 global_tint;
};

struct TextUniformBuffer {
  glm::mat4 projection;
  glm::mat4 view;
  glm::vec2 viewport_size;
  glm::vec2 dpi_scale;
  float time;
  glm::vec4 global_text_color;
  uint32_t render_flags;
};

struct HeatmapData {
  float value;
  glm::vec4 color;
  std::string label;
  uint32_t symbol_id;
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
  double volume_24h_base;
  double volume_24h_quote;
  double price_change_pct_24h;
  float high_24h;
  float low_24h;
  float last_price;
  float prev_close_24h;
  uint64_t last_update_ts;
};

struct TapeEntry {
  uint64_t timestamp;
  double price;
  double size;
  bool is_buy;
  bool is_large_trade;
  bool is_whale_trade;
  float delta;
};

struct PositionData {
  std::string symbol;
  float entry_price;
  float mark_price;
  float quantity;
  float pnl;
  float pnl_percent;
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
  ImVec4 background_primary;
  ImVec4 background_secondary;
  ImVec4 background_panel;
  ImVec4 accent_primary;
  ImVec4 accent_secondary;
  ImVec4 text_primary;
  ImVec4 text_secondary;
  ImVec4 text_muted;
  ImVec4 price_up;
  ImVec4 price_down;
  ImVec4 price_neutral;
  ImVec4 border_color;
  ImVec4 status_connected;
  ImVec4 status_disconnected;
  ImVec4 status_warning;
  void *monospace_font = nullptr;
};

struct AlertRule {
  std::string symbol;
  std::string field;
  AlertCondition condition;
  double target_value;
  bool active;
  bool is_triggered;
};

// ============================================================================
// Helper Functions
// ============================================================================

inline glm::vec4 to_glm(const ImVec4 &v) {
  return glm::vec4(v.x, v.y, v.z, v.w);
}

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

  virtual void handle_trade(const BTQuant::RenderEngine::TradeData &trade) {
    (void)trade;
  }
  virtual void
  handle_orderbook(const BTQuant::RenderEngine::OrderbookData &data) {
    (void)data;
  }

  virtual void on_resize(uint32_t width, uint32_t height) {
    (void)width;
    (void)height;
  }

  virtual void clear_data() {}

  void set_position(const glm::vec2 &pos) {
    position_ = pos;
    mark_dirty();
  }
  void set_size(const glm::vec2 &s) {
    size_ = s;
    mark_dirty();
  }
  virtual void set_target_symbol(const std::string & /*symbol*/) {}

  glm::vec2 get_position() const { return position_; }
  glm::vec2 get_size() const { return size_; }

  void set_visible(bool v) { visible_ = v; }
  bool is_visible() const { return visible_; }

  bool is_dirty() const { return is_dirty_; }
  void mark_dirty() { is_dirty_ = true; }

protected:
  glm::vec2 position_;
  glm::vec2 size_;
  bool visible_;
  bool is_dirty_;
  VulkanCore *vulkan_core_;
};

// ============================================================================
// Components
// ============================================================================

class HeatmapComponent : public UIComponent {
public:
  HeatmapComponent(const glm::vec2 &position, const glm::vec2 &size,
                   size_t grid_width = 30, size_t grid_height = 20);
  ~HeatmapComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  void set_data(const std::vector<std::vector<HeatmapData>> &data);
  void update_cell(size_t x, size_t y, const HeatmapData &data);
  void set_color_scheme(const std::vector<glm::vec4> &colors);
  void set_value_range(float min_val, float max_val);
  void clear_data() override;

private:
  void rebuild_geometry();
  void dispatch_compute_interpolation();
  glm::vec4 interpolate_color(float value);

  size_t grid_width_;
  size_t grid_height_;
  std::vector<std::vector<HeatmapData>> heatmap_data_;
  std::vector<glm::vec4> color_scheme_;
  bool interpolation_enabled_ = true;
  float min_value_ = -1.0f;
  float max_value_ = 1.0f;
  std::mutex data_mutex_;
  DashboardTheme theme_;

  VkPipeline compute_pipeline_ = VK_NULL_HANDLE;
  VkPipeline render_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout render_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compute_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout render_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet compute_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet render_descriptor_set_ = VK_NULL_HANDLE;

  BufferAllocation compute_input_buffer_;
  BufferAllocation compute_output_buffer_;
  BufferAllocation compute_previous_buffer_;
  BufferAllocation compute_ubo_buffer_;
  BufferAllocation color_scheme_buffer_;

  BufferAllocation vertex_buffer_;
  BufferAllocation index_buffer_;
  uint32_t vertex_count_ = 0;
  uint32_t index_count_ = 0;
  int dirty_frames_ = 0;

  BufferAllocation resolve_ubo_buffer_;
  BufferAllocation render_ubo_buffer_;
  bool minimized_ = false;
};

class DataGridComponent : public UIComponent {
public:
  struct CellData {
    std::string text;
    float value; // Legacy support
    double numeric_value;
    bool is_numeric;
    bool highlight;
    glm::vec4 color;
  };

  DataGridComponent(const glm::vec2 &position, const glm::vec2 &size,
                    size_t rows = 20, size_t columns = 5);
  ~DataGridComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;

  // Management methods
  void set_cell_data(size_t row, size_t col, const CellData &data);
  void set_row_data(size_t row, const std::vector<CellData> &row_data);
  void set_column_header(size_t col, const std::string &header);
  void set_column_width(size_t col, float width);
  void enable_sorting(size_t column, bool ascending);
  void set_filter(const std::string &filter_text);
  void clear_data() override;

private:
  void rebuild_geometry();
  void sort_data();

  size_t rows_;
  size_t columns_;
  std::vector<std::vector<CellData>> grid_data_;
  std::vector<std::string> column_headers_;
  std::vector<float> column_widths_;
  int sort_column_ = -1;
  bool sort_ascending_ = true;
  bool minimized_ = false;
  VulkanDashboard *dashboard_ = nullptr;
  int dirty_frames_ = 0;
  DashboardTheme theme_;
  std::mutex data_mutex_;

  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  BufferAllocation vertex_buffer_;
  BufferAllocation index_buffer_;
  uint32_t vertex_count_ = 0;
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

  void add_log(LogLevel level, const std::string &message) {
    add_log_entry(level, message);
  }
  void add_log_entry(LogLevel level, const std::string &message);
  void clear_logs();
  void clear_data() override;

  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;
  void
  handle_orderbook(const BTQuant::RenderEngine::OrderbookData &data) override;
  void rebuild_text_geometry();

private:
  struct LogEntry {
    LogLevel level;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    glm::vec4 color;
  };

  std::deque<LogEntry> log_entries_;
  size_t max_entries_ = 1000;
  LogLevel min_log_level_ = LogLevel::Debug;
  std::string text_filter_;
  bool auto_scroll_ = true;
  float scroll_offset_ = 0.0f;
  int dirty_frames_ = 0;
  DashboardTheme theme_;

  VkPipeline text_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkSampler font_sampler_ = VK_NULL_HANDLE;
  VkImageView font_image_view_ = VK_NULL_HANDLE;
  VkImage font_image_ = VK_NULL_HANDLE;
  VkDeviceMemory font_memory_ = VK_NULL_HANDLE;

  BufferAllocation text_vertex_buffer_;
  BufferAllocation ubo_buffer_;
  BufferAllocation font_metrics_buffer_;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  char filter_buffer_[256] = "";

  glm::vec4 get_log_level_color(LogLevel level);
  std::string get_log_level_string(LogLevel level);
  std::string
  format_timestamp(const std::chrono::system_clock::time_point &time);
  std::vector<LogEntry> get_filtered_entries() const;
  VulkanDashboard *dashboard_ = nullptr;
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

  void handle_trade(const RenderEngine::TradeData &trade) override;
  void handle_orderbook(const OrderbookData &data) override;
  void set_target_symbol(const std::string &symbol) override {
    target_symbol_ = symbol;
  }
  void set_precision(int price_precision, int size_precision);
  void clear_data() override;

private:
  void update_orderbook(const RenderEngine::OrderbookData &data);
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

  int price_precision_ = 2;
  int size_precision_ = 4;
  int max_levels_ = 20;
  int dirty_frames_ = 0;
  bool minimized_ = false;
  std::string symbol_ = "BTC-USDT";
  glm::vec4 bid_bar_color_ = glm::vec4(0.0f, 0.8f, 0.0f, 0.3f);
  glm::vec4 ask_bar_color_ = glm::vec4(0.8f, 0.0f, 0.0f, 0.3f);

  void rebuild_geometry();
  std::string format_price(double price);
  std::string format_size(double size);
  void add_text_line(std::vector<OrderBookTextVertex> &vertices,
                     const std::string &p_str, const std::string &s_str,
                     const std::string &t_str, float y, const glm::vec4 &color,
                     float size);
  VulkanDashboard *dashboard_ = nullptr;
};

class MarketScreenerComponent : public UIComponent {
public:
  MarketScreenerComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketScreenerComponent() override = default;

  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  std::vector<ScreenerResult> results_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
};

class RealtimeChartComponent : public UIComponent {
public:
  RealtimeChartComponent(
      const glm::vec2 &position, const glm::vec2 &size,
      std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge);
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

  uint32_t chart_width_ = 0;
  uint32_t chart_height_ = 0;

  std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge_;
  std::unique_ptr<OffscreenChartRenderer> offscreen_renderer_;
  std::unique_ptr<CandlePipeline> candle_pipeline_;

  float view_zoom_ = 1.0f;
  float view_offset_ = 0.0f;
};

class WatchlistComponent : public UIComponent {
public:
  WatchlistComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~WatchlistComponent() override = default;

  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

  void add_symbol(const std::string &symbol);
  void remove_symbol(const std::string &symbol);
  void update_quote(const std::string &symbol, double price, double change,
                    double volume);

private:
  std::vector<WatchlistEntry> entries_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
  VulkanDashboard *dashboard_ = nullptr;
};

class AlertManager {
public:
  void add_rule(const AlertRule &rule);
  void add_alert(const AlertRule &rule) { add_rule(rule); }
  void check_alerts(const std::string &symbol, double price);
  std::vector<AlertRule> get_alerts();
  void remove_alert(size_t index);
};

class AlertComponent : public UIComponent {
public:
  AlertComponent(const glm::vec2 &position, const glm::vec2 &size,
                 AlertManager &manager);
  ~AlertComponent() override;
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  AlertManager &manager_;
  std::mutex data_mutex_;
  char symbol_buffer_[64] = "";
  int selected_condition_ = 0;
  float target_value_ = 0.0f;
};

class RiskManagerComponent : public UIComponent {
public:
  RiskManagerComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float /*delta_time*/) override {}
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  DashboardTheme theme_;
};

class TradingInterfaceComponent : public UIComponent {
public:
  TradingInterfaceComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float /*delta_time*/) override {}
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  std::string order_type_ = "Limit";
  float quantity_ = 0.1f;
  float price_ = 42000.0f;
  float stop_price_ = 0.0f;
  float trailing_pct_ = 1.0f;
  float iceberg_display_qty_ = 0.01f;
  int twap_duration_mins_ = 60;
  DashboardTheme theme_;
};

class TapeComponent : public UIComponent {
public:
  TapeComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~TapeComponent() override;
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}
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
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  std::string symbol_ = "BTC-USDT";
  float quantity_ = 0.0f;
  float price_ = 0.0f;
};

class PositionPanelComponent : public UIComponent {
public:
  PositionPanelComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~PositionPanelComponent() override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

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
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float delta_time) override;
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  std::vector<TickerData> tickers_;
  float global_volume_ = 1500000000.0f;
  float system_latency_ms_ = 1.25f;
  DashboardTheme theme_;
};

class StrategyControlComponent : public UIComponent {
public:
  StrategyControlComponent(const glm::vec2 &position, const glm::vec2 &size);
  void initialize_vulkan_resources(VulkanCore * /*vulkan_core*/) override {}
  void update(float /*delta_time*/) override {}
  void render(VkCommandBuffer /*cmd*/) override {}
  void render_gui() override;
  void handle_input(const InputEvent & /*event*/) override {}

private:
  std::vector<StrategyData> strategies_;
};

class MarketDepthChartComponent : public UIComponent {
public:
  struct CurrentData {
    struct Level {
      double price;
      double size;
      double total_size;
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

  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;
  void handle_orderbook(const OrderbookData &data) override;
  void set_target_symbol(const std::string &symbol) override {
    target_symbol_ = symbol;
  }
  void clear_data() override;

private:
  void rebuild_geometry();

  std::string target_symbol_;
  CurrentData current_data_;
  std::mutex data_mutex_;
  DashboardTheme theme_;

  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  BufferAllocation vertex_buffer_;
  uint32_t vertex_count_ = 0;
  int dirty_frames_ = 0;
};

class VulkanDashboard {
public:
  VulkanDashboard(uint32_t width, uint32_t height,
                  std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge,
                  const VulkanDashboardConfig &config);
  ~VulkanDashboard();

  void initialize();
  void run();
  bool run_frame();
  void shutdown();

  void handle_input(const InputEvent &event);
  void render_frame();

  void set_active_symbol(const std::string &symbol);
  std::string get_active_symbol() const { return active_symbol_; }

  void add_component(std::unique_ptr<UIComponent> component);

  VulkanCore *get_vulkan_core() { return vulkan_core_.get(); }
  AlertManager &get_alert_manager() { return alert_manager_; }

  void synchronize_market_data();

private:
  void init_x11();
  void init_vulkan();
  void init_components();
  void handle_x11_events();
  void cleanup_x11();

  VulkanDashboardConfig config_;
  uint32_t width_;
  uint32_t height_;
  std::shared_ptr<RenderEngine::HotSpineDataBridge> hotspine_bridge_;

  std::string active_symbol_ = "BTC-USDT";
  std::vector<std::unique_ptr<UIComponent>> components_;
  std::unique_ptr<VulkanCore> vulkan_core_;

  Display *display_ = nullptr;
  Window window_;
  Atom wm_delete_window_;
  bool is_running_ = true;

  AlertManager alert_manager_;
};

} // namespace BTQuant