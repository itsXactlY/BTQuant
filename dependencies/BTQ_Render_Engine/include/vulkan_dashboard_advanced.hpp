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

// Undefine ONLY the most clashing X11 macros.
// DO NOT undefine ButtonPress, MotionNotify, etc. as they are needed by
// InteractionManager.
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

// Internal includes
#include "CandlePipeline.h"
#include "DashboardLayer.h"
#include "OffscreenChartRenderer.h"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"

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
  MouseButton mouse_button; // Matches InteractionManager usage
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

struct VulkanDashboardConfig {
  bool enable_validation_layers = false;
  bool enable_msaa = true;
  VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;
};

// ============================================================================
// Base Class
// ============================================================================

class UIComponent {
public:
  UIComponent(const glm::vec2 &position, const glm::vec2 &size)
      : position_(position), size_(size) {}
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

  void set_position(const glm::vec2 &pos) { position_ = pos; }
  void set_size(const glm::vec2 &s) { size_ = s; }
  virtual void set_target_symbol(const std::string &symbol) {}

protected:
  glm::vec2 position_;
  glm::vec2 size_;
  bool is_dirty_ = true;
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
// UI Components
// ============================================================================

class RealtimeChartComponent : public UIComponent {
public:
  RealtimeChartComponent(
      const glm::vec2 &position, const glm::vec2 &size,
      std::shared_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge =
          nullptr);
  ~RealtimeChartComponent() override;

  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;
  void clear_data() override;

private:
  std::shared_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge_;
  std::recursive_mutex data_mutex_;
  std::vector<BTQuant::RenderEngine::TradeData> raw_trades_;
  std::vector<BTQuant::RenderEngine::TradeData> raw_candles_;

  VkPipeline candle_pipeline_ = VK_NULL_HANDLE;
  float view_zoom_ = 1.0f;
  glm::vec2 view_offset_ = {0.0f, 0.0f};
};

class OrderBookComponent : public UIComponent {
public:
  OrderBookComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~OrderBookComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void handle_orderbook(
      const BTQuant::RenderEngine::OrderbookData &orderbook) override;
  void clear_data() override;

  void update_orderbook(const OrderBookData &orderbook);

private:
  void rebuild_geometry();
  void setup_uniform_buffer(OrderBookUniformBuffer &ubo);
  std::string format_price(double price);
  std::string format_size(double size);
  void add_text_line(std::vector<OrderBookTextVertex> &vertices,
                     const std::string &price, const std::string &size,
                     const std::string &total, float y, const glm::vec4 &color,
                     float font_size);
  void add_centered_text(std::vector<OrderBookTextVertex> &vertices,
                         const std::string &text, float y,
                         const glm::vec4 &color, float font_size);
  void add_text_at_position(std::vector<OrderBookTextVertex> &vertices,
                            const std::string &text, float x, float y,
                            const glm::vec4 &color, float font_size);

  struct OrderBookLevel {
    double price;
    double size;
    float last_update_ts;
  };

  struct {
    std::vector<OrderBookLevel> bids;
    std::vector<OrderBookLevel> asks;
    double spread;
    double mid_price;
    uint64_t timestamp;
  } current_data_;

  BufferAllocation bar_vertex_buffer_;
  BufferAllocation text_vertex_buffer_;
  BufferAllocation bar_ubo_buffer_;
  BufferAllocation text_ubo_buffer_;
  BufferAllocation font_metrics_buffer_;

  VkPipeline bar_pipeline_ = VK_NULL_HANDLE;
  VkPipeline text_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout bar_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout text_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout bar_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout text_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet bar_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet text_descriptor_set_ = VK_NULL_HANDLE;

  VkSampler font_sampler_ = VK_NULL_HANDLE;
  VkImageView font_image_view_ = VK_NULL_HANDLE;
  VkImage font_image_ = VK_NULL_HANDLE;
  VkDeviceMemory font_memory_ = VK_NULL_HANDLE;

  std::string symbol_ = "BTC-USDT";
  std::mutex data_mutex_;
  size_t max_levels_ = 50;
  DashboardTheme theme_;
};

class HeatmapComponent : public UIComponent {
public:
  HeatmapComponent(const glm::vec2 &position, const glm::vec2 &size,
                   size_t grid_width = 100, size_t grid_height = 100);
  ~HeatmapComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;
  void handle_orderbook(
      const BTQuant::RenderEngine::OrderbookData &orderbook) override;

  void set_data(const std::vector<std::vector<HeatmapData>> &data);
  void update_cell(size_t x, size_t y, const HeatmapData &data);
  void set_color_scheme(const std::vector<glm::vec4> &scheme);
  void set_value_range(float min_val, float max_val);

private:
  void rebuild_geometry();
  void dispatch_compute_interpolation();
  glm::vec4 interpolate_color(float value);

  BufferAllocation vertex_buffer_;
  BufferAllocation index_buffer_;
  BufferAllocation compute_input_buffer_;
  BufferAllocation compute_output_buffer_;
  BufferAllocation compute_previous_buffer_;
  BufferAllocation compute_ubo_buffer_;
  BufferAllocation color_scheme_buffer_;
  BufferAllocation render_ubo_buffer_;

  VkPipeline render_pipeline_ = VK_NULL_HANDLE;
  VkPipeline compute_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout render_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;

  VkDescriptorSetLayout render_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compute_layout_ = VK_NULL_HANDLE;

  VkDescriptorSet render_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet compute_descriptor_set_ = VK_NULL_HANDLE;

  std::vector<float> grid_data_;
  int grid_width_ = 100;
  int grid_height_ = 100;
  std::mutex data_mutex_;
  std::vector<std::vector<HeatmapData>> heatmap_data_;

  std::vector<glm::vec4> color_scheme_;
  float min_value_ = 0.0f;
  float max_value_ = 1.0f;
  bool interpolation_enabled_ = true;
};

class DataGridComponent : public UIComponent {
public:
  struct CellData {
    std::string text;
    glm::vec4 text_color = {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 bg_color = {0.0f, 0.0f, 0.0f, 0.0f};
    double numeric_value = 0.0;
    bool is_numeric = false;
    bool highlight = false;
  };

  DataGridComponent(const glm::vec2 &position, const glm::vec2 &size, int rows,
                    int cols);
  ~DataGridComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void set_cell(int row, int col, const CellData &data);
  void set_column_name(int col, const std::string &name);

private:
  void rebuild_geometry();
  void sort_data();

  std::vector<std::vector<CellData>> cells_;
  std::vector<std::string> column_names_;
  std::vector<float> column_widths_;
  int rows_ = 0;
  int columns_ = 0;
  std::mutex data_mutex_;

  int sort_column_ = -1;
  bool sort_ascending_ = true;
  VulkanDashboard *dashboard_ = nullptr;

  BufferAllocation vertex_buffer_;
  BufferAllocation index_buffer_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class MarketDepthChartComponent : public UIComponent {
public:
  MarketDepthChartComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketDepthChartComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;
  void
  handle_orderbook(const BTQuant::RenderEngine::OrderbookData &data) override;
  void clear_data() override;
  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;

private:
  void rebuild_geometry();
  uint32_t vertex_count_ = 0;
  std::string target_symbol_ = "BTC-USDT";

  BufferAllocation vertex_buffer_;
  BufferAllocation index_buffer_;
  BufferAllocation compute_input_buffer_;
  BufferAllocation compute_output_buffer_;
  BufferAllocation compute_previous_buffer_;
  BufferAllocation compute_ubo_buffer_;
  BufferAllocation color_scheme_buffer_;
  BufferAllocation render_ubo_buffer_;

  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;

  struct CurrentData {
    struct Level {
      float price;
      float size;
      float total_size;
    };
    std::vector<Level> bids;
    std::vector<Level> asks;
  } current_data_;

  std::mutex data_mutex_;
};

// ============================================================================
// Alert System
// ============================================================================

enum class AlertCondition {
  PRICE_ABOVE,
  PRICE_BELOW,
  VOLUME_ABOVE,
  VOLUME_BELOW
};

struct AlertRule {
  std::string symbol;
  AlertCondition condition;
  float target_value;
  bool is_triggered = false;
  std::chrono::system_clock::time_point created_at;
  std::chrono::system_clock::time_point triggered_at;
};

class AlertManager {
public:
  void add_alert(const AlertRule &rule) {
    std::lock_guard<std::mutex> lock(mutex_);
    alerts_.push_back(rule);
  }

  void remove_alert(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index < alerts_.size()) {
      alerts_.erase(alerts_.begin() + index);
    }
  }

  std::vector<AlertRule> get_alerts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return alerts_;
  }

  // Basic check function - logic would be more complex in real app
  void check_alerts(const std::string &symbol, double price, double volume) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &alert : alerts_) {
      if (alert.symbol == symbol && !alert.is_triggered) {
        bool triggered = false;
        switch (alert.condition) {
        case AlertCondition::PRICE_ABOVE:
          triggered = (price >= alert.target_value);
          break;
        case AlertCondition::PRICE_BELOW:
          triggered = (price <= alert.target_value);
          break;
        case AlertCondition::VOLUME_ABOVE:
          triggered = (volume >= alert.target_value);
          break;
        default:
          break;
        }
        if (triggered) {
          alert.is_triggered = true;
          alert.triggered_at = std::chrono::system_clock::now();
        }
      }
    }
  }

private:
  std::vector<AlertRule> alerts_;
  mutable std::mutex mutex_;
};

class AlertComponent : public UIComponent {
public:
  AlertComponent(const glm::vec2 &position, const glm::vec2 &size,
                 AlertManager &manager);

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}

private:
  AlertManager &manager_;
  std::mutex data_mutex_;
  char symbol_buffer_[32] = "BTC-USDT";
  int selected_condition_ = 0;
  float target_value_ = 0.0f;
};

class LogDisplayComponent : public UIComponent {
public:
  struct LogEntry {
    LogLevel level;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    glm::vec4 color;
  };

  LogDisplayComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~LogDisplayComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void add_log_entry(LogLevel level, const std::string &message);
  void handle_orderbook(const BTQuant::RenderEngine::OrderbookData &) override;
  void handle_trade(const BTQuant::RenderEngine::TradeData &trade) override;
  void clear_data() override;

private:
  void rebuild_text_geometry();
  void clear_logs();
  std::vector<LogEntry> get_filtered_entries() const;
  glm::vec4 get_log_level_color(LogLevel level);
  std::string get_log_level_string(LogLevel level);
  std::string
  format_timestamp(const std::chrono::system_clock::time_point &time);

  std::deque<LogEntry> log_entries_;
  std::mutex data_mutex_;

  size_t max_entries_ = 1000;
  bool auto_scroll_ = true;
  float scroll_offset_ = 0.0f;
  LogLevel min_log_level_ = LogLevel::Debug;
  char filter_buffer_[256] = "";

  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  BufferAllocation font_metrics_buffer_;
  VkSampler font_sampler_ = VK_NULL_HANDLE;
};

class MarketScreenerComponent : public UIComponent {
public:
  MarketScreenerComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketScreenerComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

private:
  std::vector<ScreenerResult> results_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
};

class WatchlistComponent : public UIComponent {
public:
  WatchlistComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~WatchlistComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void add_symbol(const std::string &symbol);
  void remove_symbol(const std::string &symbol);
  void update_quote(const std::string &symbol, double price, double change_24h,
                    double volume_24h);

private:
  std::vector<WatchlistEntry> entries_;
  VulkanDashboard *dashboard_ = nullptr;
  std::mutex data_mutex_;
};

class RiskManagerComponent : public UIComponent {
public:
  RiskManagerComponent(const glm::vec2 &position, const glm::vec2 &size);

  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}
};

class TradingInterfaceComponent : public UIComponent {
public:
  TradingInterfaceComponent(const glm::vec2 &position, const glm::vec2 &size);

  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}

private:
  std::string order_type_ = "Limit";
  float quantity_ = 0.0f;
  float price_ = 0.0f;
  float stop_price_ = 0.0f;
  float trailing_pct_ = 0.0f;
  float iceberg_display_qty_ = 0.0f;
  int twap_duration_mins_ = 60;
};

class TapeComponent : public UIComponent {
public:
  TapeComponent(const glm::vec2 &position, const glm::vec2 &size);

  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}

private:
  struct TapeEntry {
    uint64_t timestamp_us;
    double price;
    double size;
    bool is_buy;
    bool is_large_trade;
    bool is_whale_trade;
    float delta;
  };

  std::deque<TapeEntry> entries_;
  std::mutex data_mutex_;
  std::string target_symbol_ = "BTC-USDT";
  float large_trade_threshold_ = 5.0f;
  float whale_trade_threshold_ = 50.0f;
  float cumulative_delta_ = 0.0f;
};

class OrderManagementComponent : public UIComponent {
public:
  OrderManagementComponent(const glm::vec2 &position, const glm::vec2 &size);

  void update(float delta_time) override {}
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}

private:
  std::string symbol_ = "BTC-USDT";
  float quantity_ = 0.0f;
  float price_ = 0.0f;
};

class PositionPanelComponent : public UIComponent {
public:
  PositionPanelComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~PositionPanelComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void rebuild_equity_geometry();

private:
  struct Position {
    std::string symbol;
    float entry_price;
    float mark_price;
    float quantity;
    float pnl;
    float pnl_percent;
  };

  std::vector<Position> positions_;
  std::vector<float> equity_history_;

  BufferAllocation equity_vertex_buffer_; // For custom graph

  float total_equity_ = 105423.50f;
  float available_balance_ = 45220.10f;
};

class MarketOverviewPanel : public UIComponent {
public:
  MarketOverviewPanel(const glm::vec2 &position, const glm::vec2 &size);
  ~MarketOverviewPanel() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override {}
  void render_gui() override;
  void handle_input(const InputEvent &event) override {}
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override {}

private:
  struct Ticker {
    std::string symbol;
    float price;
    float change_pct;
  };
  std::vector<Ticker> tickers_;
  float global_volume_ = 45200000000.0f;
  float system_latency_ms_ = 12.4f;
};

struct WatchlistEntry {
  std::string symbol;
  double price = 0.0;
  double change_24h = 0.0;
  double volume_24h = 0.0;
  uint64_t last_update_ts = 0;
};

class WatchlistComponent : public UIComponent {
public:
  WatchlistComponent(const glm::vec2 &position, const glm::vec2 &size);
  ~WatchlistComponent() override;

  void update(float delta_time) override;
  void render(VkCommandBuffer cmd) override;
  void render_gui() override;
  void handle_input(const InputEvent &event) override;
  void initialize_vulkan_resources(VulkanCore *vulkan_core) override;

  void add_symbol(const std::string &symbol);
  void remove_symbol(const std::string &symbol);
  void update_quote(const std::string &symbol, double price, double change,
                    double volume);

private:
  std::vector<WatchlistEntry> entries_;
  VulkanDashboard *dashboard_ = nullptr;
  std::mutex data_mutex_;
};

} // namespace BTQuant