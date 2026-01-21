#pragma once

// ============================================================================
// BTQuant Vulkan Dashboard - Modular Header
// Streamlined from 1189 lines to ~250 lines by extracting classes into:
// - trading/order_manager.hpp, position_manager.hpp, risk_assessment.hpp
// - analytics/technical_analysis.hpp
// - ui/ui_base.hpp
// ============================================================================

#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"

// Include modular headers
#include "analytics/technical_analysis.hpp"
#include "trading/order_manager.hpp"
#include "trading/position_manager.hpp"
#include "trading/risk_assessment.hpp"
#include "ui/ui_base.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <glm/glm.hpp>
#include <imgui.h>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace BTQuant {

// Global RenderEngine namespace for data structures
namespace RenderEngine {
struct OrderBookLevel {
  double price;
  double size;
};
} // namespace RenderEngine

// Helper: Convert ImVec4 to glm::vec4
inline glm::vec4 to_glm(const ImVec4 &v) {
  return glm::vec4(v.x, v.y, v.z, v.w);
}

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

struct AlertRule {
  std::string symbol;
  AlertCondition condition;
  double target_value;
  bool is_triggered;
};

class AlertManager {
public:
  void update(float dt);
  void add_alert(const AlertRule &rule);
  void check_alerts(const std::string &symbol, double price);
  std::vector<AlertRule> get_alerts();
  void remove_alert(size_t index);
};

// Forward declarations
class VulkanDashboard;
class QuantWorkspaceComponent;

class ResizablePanel : public UIComponent {
public:
  ResizablePanel(const glm::vec2 &p, const glm::vec2 &s, const std::string &t)
      : UIComponent(p, s), title_(t) {}
  void set_resizable(bool r) { resizable_ = r; }
  void set_snap_to_grid(bool s, float g) {
    snap_to_grid_ = s;
    grid_size_ = g;
  }
  void update(float dt) override {}
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
  void save_layout(const std::string &name, const std::string &desc);
};

class ThemeManager {
public:
  std::vector<std::string> get_available_themes();
  void set_theme(const std::string &name);
};

class SearchEngine {
public:
  void index_symbol(const std::string &s, const std::string &d);
  std::vector<std::string> search(const std::string &q);
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
  void add_filter(const FilterCriteria &c);
};

// ============================================================================
// Legacy Component Declarations (Archived but declared for compatibility)
// ============================================================================

struct StrategyControlComponent : public UIComponent {
  StrategyControlComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore *core) override;
};

struct RiskManagerComponent : public UIComponent {
  RiskManagerComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore *core) override;
};

struct TradingInterfaceComponent : public UIComponent {
  TradingInterfaceComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void initialize_vulkan_resources(VulkanCore *core) override;
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

class VulkanDashboard {
public:
  VulkanDashboard(uint32_t width, uint32_t height,
                  std::shared_ptr<HotSpineDataBridge> bridge,
                  std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                  const VulkanDashboardConfig &config);
  ~VulkanDashboard();
  void initialize();
  void shutdown();
  void render_frame();
  void handle_events();
  bool should_close() const;
  void set_active_symbol(const std::string &s) { active_symbol_ = s; }
  std::string get_active_symbol() const { return active_symbol_; }
  VulkanCore *get_vulkan_core() { return m_vulkanCore.get(); }

private:
  void init_window();
  void init_vulkan();
  void init_components();
  uint32_t width_, height_;
  VulkanDashboardConfig config_;
  std::shared_ptr<HotSpineDataBridge> hotspine_bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> market_data_processor_;
  std::string active_symbol_ = "BTC-USDT";
  std::unique_ptr<VulkanCore> m_vulkanCore;
  std::unique_ptr<QuantWorkspaceComponent> m_workspace;
  uint32_t m_currentImageIndex = 0;
  bool is_running_ = true;
  bool m_windowResized = false;
  GLFWwindow *window_ = nullptr;
};

} // namespace BTQuant