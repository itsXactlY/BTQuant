#pragma once

#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "performance_monitor.hpp"
#include "vulkan_base_types.hpp"
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

// Input Types
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
enum class MouseButton {
  Left = 1,
  Middle = 2,
  Right = 3,
  ScrollUp = 4,
  ScrollDown = 5
};

struct InputEvent {
  InputEventType type;
  int keycode;
  int key;
  glm::vec2 mouse_pos;
  glm::vec2 position;
  glm::vec2 delta;
  bool pressed;
  float scroll_delta;
  MouseButton mouse_button;
  std::chrono::high_resolution_clock::time_point timestamp;
  uint32_t modifiers;
};

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
// Technical Analysis & Advanced Analytics
// ============================================================================

class TechnicalIndicators {
public:
  struct OHLCV {
    double open, high, low, close, volume;
    uint64_t timestamp;
  };

  struct IndicatorResult {
    std::vector<double> values;
    std::vector<uint64_t> timestamps;
    std::string name;
    std::unordered_map<std::string, double> parameters;
  };

  static IndicatorResult simple_moving_average(const std::vector<OHLCV> &data,
                                               int period);
  static IndicatorResult
  exponential_moving_average(const std::vector<OHLCV> &data, int period);
  static std::vector<IndicatorResult>
  bollinger_bands(const std::vector<OHLCV> &data, int period,
                  double std_dev = 2.0);
  static IndicatorResult rsi(const std::vector<OHLCV> &data, int period = 14);
  static std::vector<IndicatorResult> macd(const std::vector<OHLCV> &data,
                                           int fast_period = 12,
                                           int slow_period = 26,
                                           int signal_period = 9);
  static std::vector<IndicatorResult> stochastic(const std::vector<OHLCV> &data,
                                                 int k_period = 14,
                                                 int d_period = 3);
};

struct ProcessedTrade {
  uint32_t symbol_id;
  double price;
  double size;
  uint64_t timestamp;
  bool is_buy;
  double price_change;
  double volume_weighted_price;
};

class VolumeProfileAnalyzer {
public:
  struct VolumeNode {
    double price_level;
    double volume;
    double buy_volume;
    double sell_volume;
    int trade_count;
  };

  struct VolumeProfile {
    std::vector<VolumeNode> nodes;
    double poc_price;
    double value_area_high;
    double value_area_low;
    double total_volume;
    uint64_t start_time;
    uint64_t end_time;
  };

  struct VolumeImbalance {
    double price_level;
    double imbalance_ratio;
    double buy_volume;
    double sell_volume;
    bool is_significant;
  };

  VolumeProfileAnalyzer(double tick_size = 0.01);
  VolumeProfile calculate_volume_profile(
      const std::vector<TechnicalIndicators::OHLCV> &candles,
      const std::vector<RenderEngine::TradeData> &trades);
  std::vector<VolumeImbalance>
  detect_volume_imbalances(const VolumeProfile &profile,
                           double threshold = 2.0);

private:
  double tick_size_;
  void calculate_value_area(const std::vector<VolumeNode> &nodes,
                            VolumeProfile &profile);
};

class MarketDepthAnalyzer {
public:
  struct DepthLevel {
    double price;
    double size;
    double cumulative_size;
    int order_count;
    double average_order_size;
  };

  struct MarketDepthSnapshot {
    std::vector<DepthLevel> bids;
    std::vector<DepthLevel> asks;
    double spread;
    double mid_price;
    double total_bid_volume;
    double total_ask_volume;
    double imbalance_ratio;
    uint64_t timestamp;
  };

  struct DepthAnalysis {
    double support_level;
    double resistance_level;
    double liquidity_score;
    double market_impact_estimate;
    std::vector<double> significant_levels;
  };

  struct LiquidityGap {
    double price_start;
    double price_end;
    double gap_size;
    bool is_bid_side;
  };

  MarketDepthAnalyzer() = default;
  MarketDepthSnapshot
  create_depth_snapshot(const RenderEngine::OrderbookData &orderbook);
  DepthAnalysis analyze_market_depth(const MarketDepthSnapshot &snapshot);
  std::vector<LiquidityGap>
  detect_liquidity_gaps(const MarketDepthSnapshot &snapshot,
                        double min_gap_size = 0.01);

private:
  double find_support_level(const std::vector<DepthLevel> &bids);
  double find_resistance_level(const std::vector<DepthLevel> &asks);
  double calculate_liquidity_score(const MarketDepthSnapshot &snapshot);
  double estimate_market_impact(const MarketDepthSnapshot &snapshot,
                                double order_value);
  std::vector<double>
  find_significant_levels(const MarketDepthSnapshot &snapshot);
};

class PatternRecognizer {
public:
  enum class PatternType {
    DoubleTop,
    DoubleBottom,
    HeadAndShoulders,
    InverseHeadAndShoulders,
    Triangle,
    Flag,
    Pennant,
    Cup,
    Hammer,
    Doji,
    EngulfingBullish,
    EngulfingBearish,
    Wedge
  };

  struct Pattern {
    PatternType type;
    std::string name;
    double confidence;
    uint64_t start_time;
    uint64_t end_time;
    double entry_price;
    double target_price;
    double stop_loss;
    std::vector<glm::vec2> key_points;
    std::string description;
  };

  PatternRecognizer() = default;
  std::vector<Pattern>
  detect_patterns(const std::vector<TechnicalIndicators::OHLCV> &data);

private:
  std::vector<Pattern>
  detect_double_top(const std::vector<TechnicalIndicators::OHLCV> &data);
  std::vector<Pattern>
  detect_double_bottom(const std::vector<TechnicalIndicators::OHLCV> &data);
  std::vector<Pattern> detect_head_and_shoulders(
      const std::vector<TechnicalIndicators::OHLCV> &data);
  std::vector<Pattern>
  detect_triangles(const std::vector<TechnicalIndicators::OHLCV> &data);
  std::vector<Pattern> detect_candlestick_patterns(
      const std::vector<TechnicalIndicators::OHLCV> &data);
  std::vector<size_t>
  find_peaks(const std::vector<TechnicalIndicators::OHLCV> &data,
             bool find_highs);
  double
  find_valley_between(const std::vector<TechnicalIndicators::OHLCV> &data,
                      size_t start, size_t end);
  bool is_local_high(const std::vector<TechnicalIndicators::OHLCV> &data,
                     size_t index);
  bool is_local_low(const std::vector<TechnicalIndicators::OHLCV> &data,
                    size_t index);
  std::pair<double, double>
  calculate_trend_line(const std::vector<glm::vec2> &points);
  bool is_hammer(const TechnicalIndicators::OHLCV &candle);
  bool is_doji(const TechnicalIndicators::OHLCV &candle);
  bool is_bullish_engulfing(const TechnicalIndicators::OHLCV &prev,
                            const TechnicalIndicators::OHLCV &curr);
  bool is_bearish_engulfing(const TechnicalIndicators::OHLCV &prev,
                            const TechnicalIndicators::OHLCV &curr);
  double calculate_double_top_confidence(
      const std::vector<TechnicalIndicators::OHLCV> &data, size_t peak1,
      size_t valley, size_t peak2);
  double calculate_double_bottom_confidence(
      const std::vector<TechnicalIndicators::OHLCV> &data, size_t trough1,
      size_t peak, size_t trough2);
  double calculate_head_shoulders_confidence(
      const std::vector<TechnicalIndicators::OHLCV> &data, size_t left,
      size_t head, size_t right);
};

// ============================================================================
// Trading & Risk Management
// ============================================================================

class OrderManager {
public:
  enum class OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop,
    Iceberg,
    TWAP,
    VWAP
  };
  enum class OrderSide { Buy, Sell };
  enum class OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired
  };
  enum class TimeInForce { GTC, IOC, FOK, DAY, GTD };

  struct Order {
    std::string order_id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;
    double stop_price = 0;
    double filled_quantity = 0;
    double average_fill_price = 0;
    OrderStatus status = OrderStatus::Pending;
    TimeInForce time_in_force = TimeInForce::GTC;
    uint64_t created_time = 0;
    uint64_t updated_time = 0;
    uint64_t expiry_time = 0;
    double trailing_amount = 0;
    double iceberg_visible_quantity = 0;
    double twap_duration_minutes = 0;
    std::string parent_order_id;
    std::vector<std::string> child_order_ids;
    double max_position_size = 0;
    double max_loss_amount = 0;
    bool reduce_only = false;
    std::string execution_venue;
    double slippage_tolerance = 0;
    bool post_only = false;
    std::unordered_map<std::string, std::string> custom_fields;
  };

  struct OrderExecution {
    std::string execution_id;
    std::string order_id;
    double quantity;
    double price;
    double commission;
    uint64_t timestamp;
    std::string venue;
    std::string liquidity_flag;
  };

  OrderManager();
  std::string place_order(const Order &order);
  bool modify_order(const std::string &order_id, double new_quantity,
                    double new_price);
  bool cancel_order(const std::string &order_id);
  std::vector<Order> get_orders(const std::string &symbol = "") const;
  std::vector<Order> get_active_orders(const std::string &symbol = "") const;
  void add_execution(const OrderExecution &execution);
  std::vector<OrderExecution> get_executions(const std::string &order_id) const;

  // Event callbacks
  using OrderUpdateCallback = std::function<void(const Order &)>;
  using ExecutionCallback = std::function<void(const OrderExecution &)>;
  void set_order_update_callback(OrderUpdateCallback callback);
  void set_execution_callback(ExecutionCallback callback);

private:
  std::unordered_map<std::string, Order> orders_;
  std::unordered_map<std::string, OrderExecution> executions_;
  std::unordered_map<std::string, std::vector<std::string>> order_executions_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      symbol_orders_;
  std::unordered_map<std::string, std::string> order_symbols_;
  OrderUpdateCallback order_update_callback_;
  ExecutionCallback execution_callback_;
  std::atomic<uint64_t> order_counter_{0};

  bool validate_order(const Order &order);
  std::string generate_order_id();
  uint64_t get_current_timestamp();
  void process_order(const Order &order);
  void simulate_market_order_execution(const Order &order);
  void add_to_order_book(const Order &order);
  void add_to_stop_orders(const Order &order);
  void add_to_trailing_stops(const Order &order);
  void process_iceberg_order(const Order &order);
  void process_algorithmic_order(const Order &order);
  std::string generate_execution_id();
  double get_current_market_price(const std::string &symbol, OrderSide side);
  double calculate_commission(double quantity, double price);
  void notify_order_update(const Order &order);
  void notify_execution(const OrderExecution &execution);
};

class PositionManager {
public:
  struct Position {
    std::string symbol;
    double quantity = 0;
    double average_price = 0;
    double unrealized_pnl = 0;
    double realized_pnl = 0;
    double market_value = 0;
    double cost_basis = 0;
    uint64_t first_trade_time = 0;
    uint64_t last_trade_time = 0;
    double max_drawdown = 0;
    double max_profit = 0;
    double var_95 = 0;
    double beta = 1.0;
    double sharpe_ratio = 0;
    std::vector<std::string> contributing_orders;
    double total_commission = 0;
    int trade_count = 0;
  };

  struct PortfolioSummary {
    double total_value = 0;
    double total_unrealized_pnl = 0;
    double total_realized_pnl = 0;
    double total_commission = 0;
    double cash_balance = 0;
    double buying_power = 0;
    double margin_used = 0;
    double portfolio_beta = 0;
    double portfolio_var = 0;
    double sharpe_ratio = 0;
    int position_count = 0;
    int trade_count = 0;
  };

  PositionManager();
  void update_position(const OrderManager::OrderExecution &execution);
  void
  update_market_prices(const std::unordered_map<std::string, double> &prices);
  std::vector<Position> get_positions() const;
  Position get_position(const std::string &symbol) const;
  PortfolioSummary get_portfolio_summary() const;
  void set_cash_balance(double balance);

  using PositionUpdateCallback = std::function<void(const Position &)>;
  void set_position_update_callback(PositionUpdateCallback callback);

private:
  std::unordered_map<std::string, Position> positions_;
  std::unordered_map<std::string, double> market_prices_;
  std::unordered_map<std::string, std::string> order_symbols_;
  PositionUpdateCallback position_update_callback_;
  double cash_balance_ = 100000.0;

  std::string get_symbol_from_order(const std::string &order_id);
  bool is_buy_execution(const OrderManager::OrderExecution &execution);
  void update_market_values();
  void calculate_risk_metrics(Position &position);
  double calculate_buying_power() const;
  double calculate_margin_used() const;
  double calculate_portfolio_beta() const;
  double calculate_portfolio_var() const;
  double calculate_portfolio_sharpe() const;
  void notify_position_update(const Position &position);
};

class RiskAssessment {
public:
  struct RiskLimits {
    double max_position_size;
    double max_portfolio_value;
    double max_daily_loss;
    double max_drawdown;
    double max_leverage;
    double max_concentration;
    double var_limit;
    std::unordered_map<std::string, double> symbol_limits;
    std::unordered_map<std::string, double> sector_limits;
  };

  struct RiskMetrics {
    double current_var = 0;
    double portfolio_beta = 1.0;
    double sharpe_ratio = 0;
    double max_drawdown = 0;
    double current_leverage = 0;
    double largest_position_pct = 0;
    double daily_pnl = 0;
    double unrealized_pnl = 0;
    double overall_risk_score = 0;
    double concentration_risk = 0;
    double leverage_risk = 0;
    double volatility_risk = 0;
    double liquidity_risk = 0;
  };

  struct RiskAlert {
    enum class Severity { Info, Warning, Critical };
    Severity severity;
    std::string message;
    std::string symbol;
    double current_value;
    double limit_value;
    uint64_t timestamp;
    bool acknowledged;
  };

  struct RiskReport {
    RiskMetrics metrics;
    std::vector<RiskAlert> alerts;
    std::vector<std::string> recommendations;
    double risk_adjusted_return;
    double maximum_trade_size;
    std::unordered_map<std::string, double> symbol_risk_scores;
  };

  RiskAssessment();
  void set_risk_limits(const RiskLimits &limits);
  RiskMetrics calculate_risk_metrics(
      const PositionManager::PortfolioSummary &summary,
      const std::vector<PositionManager::Position> &positions);
  std::vector<RiskAlert>
  check_risk_limits(const RiskMetrics &metrics,
                    const PositionManager::PortfolioSummary &portfolio,
                    const std::vector<PositionManager::Position> &positions);
  bool
  validate_order_risk(const OrderManager::Order &order,
                      const PositionManager::PortfolioSummary &portfolio,
                      const std::vector<PositionManager::Position> &positions);
  RiskReport
  generate_risk_report(const PositionManager::PortfolioSummary &portfolio,
                       const std::vector<PositionManager::Position> &positions);

private:
  RiskLimits risk_limits_;
  void initialize_default_limits();
  double calculate_portfolio_var(
      const std::vector<PositionManager::Position> &positions);
  double calculate_concentration_risk(
      const std::vector<PositionManager::Position> &positions,
      double total_value);
  double calculate_leverage_risk(double leverage);
  double calculate_volatility_risk(
      const std::vector<PositionManager::Position> &positions);
  double calculate_liquidity_risk(
      const std::vector<PositionManager::Position> &positions);
  PositionManager::Position simulate_order_impact(
      const OrderManager::Order &order,
      const std::vector<PositionManager::Position> &positions);
  std::vector<std::string>
  generate_recommendations(const RiskMetrics &metrics,
                           const std::vector<RiskAlert> &alerts);
  double
  calculate_max_trade_size(const PositionManager::PortfolioSummary &portfolio,
                           const RiskMetrics &metrics);
  double calculate_symbol_risk_score(const PositionManager::Position &position);
  uint64_t get_current_timestamp();
};

// ============================================================================
// System Resources & Optimization
// ============================================================================

class SystemResourceMonitor {
public:
  struct CPUInfo {
    int core_count;
    float overall_usage;
    std::vector<float> core_usage;
  };
  struct MemoryInfo {
    uint64_t total_bytes;
    uint64_t used_bytes;
    float usage_percent;
  };
  struct SystemHealth {
    int overall_score;
    std::vector<std::string> alerts;
  };

  CPUInfo get_cpu_info();
  MemoryInfo get_memory_info();
  SystemHealth get_system_health();
};

class MemoryLeakDetector {
public:
  struct LeakReport {
    uint64_t total_leaked_bytes;
    int leak_count;
  };
  void record_allocation(void *ptr, size_t size, const char *file, int line);
  void record_deallocation(void *ptr);
  LeakReport generate_leak_report();
};

class NetworkOptimizer {
public:
  struct NetworkMetrics {
    double average_latency_ms;
    double packet_loss_percent;
  };
  struct NetworkSettings {
    bool tcp_no_delay;
    int buffer_size;
  };
  void optimize_network_settings(const NetworkMetrics &metrics);
  NetworkSettings get_current_settings() const;
};

class CacheOptimizer {
public:
  struct CacheStats {
    float hit_ratio;
    float cache_utilization;
  };
  struct CacheConfig {
    uint64_t max_size_bytes;
    int eviction_policy;
  };
  void optimize_cache_settings(const CacheStats &stats);
  CacheConfig get_current_config() const;
};

class ThreadPoolOptimizer {
public:
  struct ThreadPoolStats {
    float thread_utilization;
    int queued_tasks;
  };
  struct ThreadPoolConfig {
    int max_threads;
    int task_priority_levels;
  };
  void optimize_thread_pool(const ThreadPoolStats &stats);
  ThreadPoolConfig get_current_config() const;
};

// ============================================================================
// UI Base & Components
// ============================================================================

// Essential Types
enum class LogLevel { Debug, Info, Warning, Error, Critical };

struct DashboardTheme {
  ImVec4 accent_primary = {0.0f, 0.95f, 1.0f, 1.0f};
  ImVec4 accent_secondary = {1.0f, 0.0f, 0.3f, 1.0f};
  ImVec4 price_up = {0.0f, 0.95f, 1.0f, 1.0f};
  ImVec4 price_down = {1.0f, 0.0f, 0.3f, 1.0f};
  ImVec4 background = {0.04f, 0.04f, 0.04f, 1.0f};
  ImVec4 background_secondary = {0.1f, 0.1f, 0.1f, 1.0f};
  ImVec4 background_panel = {0.08f, 0.08f, 0.08f, 1.0f};
  ImVec4 background_primary = {0.04f, 0.04f, 0.04f, 1.0f};
  ImVec4 border_color = {0.2f, 0.2f, 0.2f, 1.0f};
  ImVec4 text_primary = {0.9f, 0.9f, 0.9f, 1.0f};
  ImVec4 text_muted = {0.5f, 0.5f, 0.5f, 1.0f};
  void *monospace_font = nullptr;
};

// Common Structures
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

class VulkanDashboard;

struct UIComponent {
  UIComponent(const glm::vec2 &p, const glm::vec2 &s)
      : position_(p), size_(s), visible_(true), dirty_frames_(3) {}
  virtual ~UIComponent() = default;
  virtual void initialize_vulkan_resources(VulkanCore *) {}
  virtual void update(float dt) = 0;
  virtual void render_gui() = 0;
  virtual void clear_data() {}
  void mark_dirty() { dirty_frames_ = 3; }
  bool is_dirty() const { return dirty_frames_ > 0; }
  virtual void handle_trade(const RenderEngine::TradeData &) {}
  virtual void handle_orderbook(const RenderEngine::OrderbookData &) {}
  virtual void handle_input(const InputEvent &) {}
  bool is_visible() const { return visible_; }
  glm::vec2 get_position() const { return position_; }
  glm::vec2 get_size() const { return size_; }
  glm::vec2 position_, size_;
  bool visible_;
  int dirty_frames_;
};

// Forward declare Quant Workspace components
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

struct StrategyControlComponent : public UIComponent {
  StrategyControlComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void clear_data() override;
  struct StrategyEntry {
    std::string name;
    bool active;
    float pnl;
    float draw_pct;
    int trades;
    std::string status;
  };
  std::vector<StrategyEntry> strategies_;
};

struct RiskManagerComponent : public UIComponent {
  RiskManagerComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void clear_data() override;
  DashboardTheme theme_;
};

struct TradingInterfaceComponent : public UIComponent {
  TradingInterfaceComponent(const glm::vec2 &p, const glm::vec2 &s);
  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void clear_data() override;
  DashboardTheme theme_;
  std::string order_type_ = "Limit";
  float quantity_ = 0, price_ = 0, stop_price_ = 0, trailing_pct_ = 0,
        iceberg_display_qty_ = 0;
  int twap_duration_mins_ = 30;
};

struct TapeComponent : public UIComponent {
  TapeComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~TapeComponent() override;
  void handle_trade(const RenderEngine::TradeData &) override;
  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void clear_data() override;
  struct TapeEntry {
    uint64_t timestamp;
    double price;
    double size;
    bool is_buy;
    bool is_large_trade;
    bool is_whale_trade;
    float delta;
  };
  std::deque<TapeEntry> entries_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
  std::string target_symbol_ = "BTC-USDT";
  float cumulative_delta_ = 0, large_trade_threshold_ = 1.0f,
        whale_trade_threshold_ = 5.0f;
};

struct OrderManagementComponent : public UIComponent {
  OrderManagementComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~OrderManagementComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  std::string symbol_ = "BTC-USDT";
  float quantity_ = 0, price_ = 0;
  std::mutex data_mutex_;
};

struct PositionPanelComponent : public UIComponent {
  PositionPanelComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~PositionPanelComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void rebuild_equity_geometry();
  struct PositionEntry {
    std::string symbol;
    float quantity;
    float entry_price;
    float current_price;
    float pnl;
    float pnl_percent;
  };
  std::vector<PositionEntry> positions_;
  std::vector<float> equity_history_;
  DashboardTheme theme_;
  VulkanCore *vulkan_core_ = nullptr;
  BufferAllocation equity_vertex_buffer_;
  float total_equity_ = 100000.0f, available_balance_ = 85000.0f;
  std::mutex data_mutex_;
};

struct MarketOverviewPanel : public UIComponent {
  MarketOverviewPanel(const glm::vec2 &p, const glm::vec2 &s);
  ~MarketOverviewPanel() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  struct Ticker {
    std::string symbol;
    float price;
    float change_pct;
  };
  std::vector<Ticker> tickers_;
  DashboardTheme theme_;
  double global_volume_ = 1.2e9;
  float system_latency_ms_ = 2.4f;
  std::mutex data_mutex_;
};

struct WatchlistComponent : public UIComponent {
  WatchlistComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~WatchlistComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  void add_symbol(const std::string &symbol);
  void remove_symbol(const std::string &symbol);
  void update_quote(const std::string &, double, double, double);
  std::vector<WatchlistEntry> entries_;
  DashboardTheme theme_;
  VulkanDashboard *dashboard_ = nullptr;
  std::mutex data_mutex_;
};

struct LogDisplayComponent : public UIComponent {
  LogDisplayComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~LogDisplayComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void render(VkCommandBuffer);
  void clear_data() override;
  void handle_input(const InputEvent &) override;
  void handle_trade(const RenderEngine::TradeData &) override;
  void handle_orderbook(const RenderEngine::OrderbookData &) override;
  void add_log(LogLevel, const std::string &);
  void add_log_entry(LogLevel, const std::string &);
  void clear_logs();
  struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string message;
    glm::vec4 color;
  };
  std::vector<LogEntry> get_filtered_entries() const;
  void rebuild_text_geometry();
  glm::vec4 get_log_level_color(LogLevel);
  std::string get_log_level_string(LogLevel);
  std::string format_timestamp(const std::chrono::system_clock::time_point &);
  struct LogTextVertex {
    glm::vec2 position;
    glm::vec2 texcoord;
    glm::vec4 color;
    uint32_t glyph_id;
    float font_size;
    uint32_t log_level;
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
  struct GlyphMetric {
    glm::vec4 atlas_coords;
    glm::vec2 bearing;
    float advance;
  };
  std::deque<LogEntry> log_entries_;
  size_t max_entries_;
  bool auto_scroll_;
  LogLevel min_log_level_;
  std::string text_filter_;
  float scroll_offset_;
  char filter_buffer_[256];
  std::mutex log_mutex_;
  DashboardTheme theme_;
  VulkanCore *vulkan_core_ = nullptr;
  VkPipeline text_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  VkSampler font_sampler_ = VK_NULL_HANDLE;
  VkImageView font_image_view_ = VK_NULL_HANDLE;
  VkImage font_image_ = VK_NULL_HANDLE;
  VkDeviceMemory font_memory_ = VK_NULL_HANDLE;
  BufferAllocation text_vertex_buffer_, ubo_buffer_, font_metrics_buffer_;
};

struct HeatmapComponent : public UIComponent {
  HeatmapComponent(const glm::vec2 &p, const glm::vec2 &s, size_t w = 10,
                   size_t h = 10);
  ~HeatmapComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void render(VkCommandBuffer);
  void clear_data() override;
  void handle_input(const InputEvent &) override;
  void rebuild_geometry();
  void dispatch_compute_interpolation();
  struct HeatmapData {
    float value;
    glm::vec4 color;
    std::string label;
    uint32_t symbol_id;
  };
  void set_data(const std::vector<std::vector<HeatmapData>> &data);
  void update_cell(size_t x, size_t y, const HeatmapData &data);
  void set_color_scheme(const std::vector<glm::vec4> &colors);
  void set_value_range(float min_val, float max_val);
  glm::vec4 interpolate_color(float value);
  std::vector<std::vector<HeatmapData>> heatmap_data_;
  size_t grid_width_, grid_height_;
  std::vector<glm::vec4> color_scheme_;
  bool interpolation_enabled_;
  float min_value_, max_value_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
  VulkanCore *vulkan_core_ = nullptr;
  bool minimized_ = false;
  VkPipeline render_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout render_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout render_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet render_descriptor_set_ = VK_NULL_HANDLE;
  BufferAllocation vertex_buffer_, index_buffer_, render_ubo_buffer_;
  VkPipeline compute_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout compute_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet compute_descriptor_set_ = VK_NULL_HANDLE;
  BufferAllocation compute_input_buffer_, compute_output_buffer_,
      compute_previous_buffer_, compute_ubo_buffer_, color_scheme_buffer_;
};

struct MarketScreenerComponent : public UIComponent {
  MarketScreenerComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~MarketScreenerComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  std::vector<ScreenerResult> results_;
  std::mutex data_mutex_;
  DashboardTheme theme_;
};

struct AlertComponent : public UIComponent {
  AlertComponent(const glm::vec2 &p, const glm::vec2 &s, AlertManager &manager);
  ~AlertComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void clear_data() override;
  struct AlertEntry {
    std::string time;
    std::string symbol;
    std::string message;
    LogLevel level;
  };
  std::vector<AlertEntry> alerts_;
  std::mutex data_mutex_;
  AlertManager &manager_;
  DashboardTheme theme_;
  char symbol_buffer_[64];
  int selected_condition_ = 0;
  float target_value_ = 0.0f;
  bool visible_ = true;
};

struct DataGridComponent : public UIComponent {
  DataGridComponent(const glm::vec2 &p, const glm::vec2 &s, size_t r = 20,
                    size_t c = 5);
  ~DataGridComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void render(VkCommandBuffer);
  void clear_data() override;
  void handle_input(const InputEvent &) override;
  void rebuild_geometry();
  void sort_data();
  struct CellData {
    std::string text;
    float value;
    float numeric_value;
    bool is_numeric;
    bool highlight;
    glm::vec4 color;
  };
  void set_cell_data(size_t row, size_t col, const CellData &data);
  void set_row_data(size_t row, const std::vector<CellData> &row_data);
  void set_column_header(size_t col, const std::string &header);
  void set_column_width(size_t col, float width);
  void enable_sorting(size_t column, bool ascending);
  void set_filter(const std::string &filter);
  std::vector<std::vector<CellData>> grid_data_;
  size_t rows_, columns_;
  std::vector<std::string> column_headers_;
  std::vector<float> column_widths_;
  int sort_column_ = -1;
  bool sort_ascending_ = true;
  std::mutex data_mutex_;
  DashboardTheme theme_;
  VulkanDashboard *dashboard_ = nullptr;
  bool minimized_ = false;
  BufferAllocation vertex_buffer_, index_buffer_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

struct MarketDepthChartComponent : public UIComponent {
  MarketDepthChartComponent(const glm::vec2 &p, const glm::vec2 &s);
  ~MarketDepthChartComponent() override;
  void initialize_vulkan_resources(VulkanCore *) override;
  void update(float dt) override;
  void render_gui() override;
  void render(VkCommandBuffer);
  void clear_data() override;
  void handle_input(const InputEvent &) override;
  void handle_trade(const RenderEngine::TradeData &) override;
  void handle_orderbook(const RenderEngine::OrderbookData &) override;
  void rebuild_geometry();
  struct LevelData {
    double price;
    double size;
    double total_size;
  };
  struct OrderBookData {
    std::vector<LevelData> bids;
    std::vector<LevelData> asks;
  };
  OrderBookData current_data_;
  std::string target_symbol_ = "BTC-USDT";
  DashboardTheme theme_;
  VulkanCore *vulkan_core_ = nullptr;
  BufferAllocation vertex_buffer_;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  uint32_t vertex_count_ = 0;
};

using DepthChartComponent = MarketDepthChartComponent;

class QuantWorkspaceComponent;
class ArchitectureVisualizationComponent;
class SystemResourceUtilizationComponent;

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
  std::shared_ptr<RenderEngine::PerformanceMonitor> performance_monitor_;
  std::string active_symbol_ = "BTC-USDT";
  std::unique_ptr<VulkanCore> m_vulkanCore;
  std::unique_ptr<QuantWorkspaceComponent> m_workspace;
  std::unique_ptr<ArchitectureVisualizationComponent>
      m_architecture_visualization;
  std::unique_ptr<SystemResourceUtilizationComponent> m_system_resource_monitor;
  uint32_t m_currentImageIndex = 0;
  bool is_running_ = true;
  bool m_windowResized = false;
  GLFWwindow *window_ = nullptr;
};

} // namespace BTQuant