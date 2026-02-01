#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <list>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "chart_manager.hpp"
#include "indicator_renderer.hpp"
#include "panel_base.hpp"
#include "../indicators/anchored_vwap.hpp"
#include "../indicators/session_vwap.hpp"
#include "panel_manager.hpp"

namespace BTQuant {

class TimeStatisticsPanel; // Forward declaration

// Indicator configuration for chart panel
struct IndicatorConfig {
  // SMA configurations
  bool show_sma_9 = false;
  bool show_sma_10 = false;
  bool show_sma_20 = false;
  bool show_sma_50 = false;
  bool show_sma_200 = false;

  // EMA configurations
  bool show_ema_9 = false;
  bool show_ema_10 = false;
  bool show_ema_20 = false;
  bool show_ema_21 = false;
  bool show_ema_50 = false;
  bool show_ema_200 = false;

  // RSI configuration
  bool show_rsi = false;

  // MACD configuration
  bool show_macd = false;

  // Bollinger Bands configuration
  bool show_bollinger = false;

  // Stochastic configuration
  bool show_stochastic = false;

  // ATR configuration
  bool show_atr = false;

  bool show_volume_profile = true;
  bool show_fibonacci = false;
  bool show_crosshair_info = true;

  // Fibonacci configuration
  double fib_start_price = 0.0;
  double fib_end_price = 0.0;

  // RSI configuration
  int rsi_period = 14;
  double rsi_overbought = 70.0;
  double rsi_oversold = 30.0;

  // Bollinger Bands configuration
  int bollinger_period = 20;
  double bollinger_std_dev = 2.0;

  // MACD configuration
  int macd_fast_period = 12;
  int macd_slow_period = 26;
  int macd_signal_period = 9;

  // Stochastic configuration
  int stochastic_k_period = 14;
  int stochastic_d_period = 3;
  int stochastic_smooth_period = 3;

  // ATR configuration
  int atr_period = 14;
};

// Fibonacci Retracement Level
struct FibonacciLevel {
  double price;
  double ratio;
  const char* label;
  ImU32 color;
};

// Indicator representation for the overlay panel
struct IndicatorItem {
  std::string name;
  bool isVisible;
  ImVec4 color;
  std::map<std::string, float> parameters;  // Generic parameter storage
  int id;  // Unique identifier for the indicator

  IndicatorItem(const std::string& n, bool vis, ImVec4 c, int indicator_id)
    : name(n), isVisible(vis), color(c), id(indicator_id) {}
};

class ChartPanel : public PanelBase {
 public:
  using ScrollSyncCallback = std::function<void(uint64_t start_timestamp, uint64_t end_timestamp)>;

  ChartPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
             ChartManager* chart_manager,
             PanelManager* panel_manager = nullptr);

  void update(float dt) override;
  void render() override;
  void initialize() override;

  // Chart-specific methods
  void set_symbol(const std::string& symbol, const std::string& exchange = "Binance");
  void set_timeframe(RenderEngine::TimeFrame timeframe);
  uint32_t get_chart_id() const { return chart_id_; }

  // Method to center the chart on a specific timestamp
  void center_on_timestamp(uint64_t timestamp);

  // Set callback for scroll synchronization
  void set_scroll_sync_callback(ScrollSyncCallback callback) {
      on_scroll_sync_ = std::move(callback);
  }

  // Set associated time statistics panel for synchronization
  void set_associated_time_stats_panel(TimeStatisticsPanel* time_stats_panel) {
      associated_time_stats_panel_ = time_stats_panel;
  }

  // Get visible time range
  std::pair<uint64_t, uint64_t> get_visible_time_range() const;

  // Set callback for showing historical trades
  void set_show_historical_trades_callback(std::function<void(uint64_t, uint64_t)> callback) {
    on_show_historical_trades_ = std::move(callback);
  }

  // Method to initialize active indicators from current configuration
  void initialize_active_indicators();

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  ChartManager* chart_manager_;
  IndicatorRenderer* indicator_renderer_;

  std::string symbol_ = "BTC-USDT";
  std::string exchange_ = "Binance";
  RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::TF_1SEC;
  uint32_t chart_id_ = 0;

  IndicatorConfig indicator_config_;
  bool follow_latest_ = true;
  float auto_follow_window_ = 1000.0f;
  double last_view_min_ = 0.0;
  double last_view_max_ = 0.0;
  bool first_frame_ = true;

  // Cached indicator data to prevent recalculation on every render
  enum class IndicatorType {
    SMA, EMA, RSI, STOCH_K, STOCH_D, ATR, TRUE_RANGE, BB_UPPER, BB_MIDDLE, BB_LOWER,
    MACD_LINE, MACD_SIGNAL, MACD_HISTOGRAM
  };

  struct IndicatorCacheKey {
    IndicatorType type;
    size_t data_size;
    int period1;
    int period2;
    double param1;  // For additional parameters like standard deviation

    bool operator==(const IndicatorCacheKey& other) const {
      return type == other.type &&
             data_size == other.data_size &&
             period1 == other.period1 &&
             period2 == other.period2 &&
             param1 == other.param1;
    }
  };

  struct IndicatorCacheKeyHash {
    std::size_t operator()(const IndicatorCacheKey& k) const {
      std::size_t h1 = std::hash<int>{}(static_cast<int>(k.type));
      std::size_t h2 = std::hash<size_t>{}(k.data_size);
      std::size_t h3 = std::hash<int>{}(k.period1);
      std::size_t h4 = std::hash<int>{}(k.period2);
      std::size_t h5 = std::hash<double>{}(k.param1);
      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
  };

  // Unified cached indicators map
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_indicators_;

  // Legacy cached indicators (to be deprecated gradually)
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_sma_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_ema_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_rsi_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_stoch_k_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_atr_;

  // Track the last known data size to detect when cache needs invalidation
  size_t last_known_data_size_ = 0;

  // Callback for scroll synchronization
  ScrollSyncCallback on_scroll_sync_;

  // Associated time statistics panel for synchronization
  TimeStatisticsPanel* associated_time_stats_panel_ = nullptr;

  // Anchored VWAPs
  std::list<::btq::AnchoredVWAP> anchored_vwaps_;

  // Session VWAP
  ::btq::SessionVWAP session_vwap_;

  // Variables for storing clicked bar time range
  uint64_t clicked_bar_start_time_ = 0;
  uint64_t clicked_bar_end_time_ = 0;

  // Flag to show trades popup
  bool show_trades_popup_ = false;

  // Callback for showing historical trades
  std::function<void(uint64_t, uint64_t)> on_show_historical_trades_;

  // Pointer to panel manager for creating new panels
  PanelManager* panel_manager_ = nullptr;

  // Active indicators list for the overlay panel
  std::vector<IndicatorItem> active_indicators_;
  int next_indicator_id_ = 1;  // Counter for generating unique IDs

  // Multi-timeframe indicators support
  struct MultiTimeframeIndicator {
    std::string name;
    bool isVisible;
    ImVec4 color;
    int period;
    RenderEngine::TimeFrame source_timeframe;  // Timeframe from which to source the indicator
    std::vector<double> values;  // Cached values from the source timeframe
    std::vector<double> timestamps;  // Timestamps for the values

    MultiTimeframeIndicator(const std::string& n, bool vis, ImVec4 c, int p, RenderEngine::TimeFrame tf)
      : name(n), isVisible(vis), color(c), period(p), source_timeframe(tf) {}
  };

  std::vector<MultiTimeframeIndicator> multi_tf_indicators_;
  int next_multitf_indicator_id_ = 1000;  // Counter for multi-timeframe indicators (separate ID space)

  // Accessors for view range (needed for synchronization)
  friend class PanelManager; // Allow PanelManager to access private members for synchronization

  void render_chart_controls();
  void render_indicator_selector();
  void render_instrument_chart(const ChartInstance& chart);
  void render_candlestick(const ChartInstance& chart);
  void render_context_menu(const ChartInstance& chart);
  void render_anchored_vwap_overlay(const ChartInstance& chart);
  void render_session_vwap_overlay(const ChartInstance& chart);
  void create_anchored_vwap_at_time(uint64_t timestamp);
  void render_trades_popup();

  // Indicator rendering methods
  void render_sma_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_ema_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_bollinger_bands(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_rsi_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_macd_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_stochastic_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_atr_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_fibonacci_levels(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_crosshair_info(const ChartInstance& chart, double mouse_x, double mouse_y);
  void render_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                   const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                   const std::vector<double>& x_coords,
                                                   const std::vector<double>& y_coords_high,
                                                   const std::vector<double>& y_coords_low,
                                                   bool show_poc_line = true,
                                                   int num_buckets_per_candle = 8);
  void render_enhanced_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                           const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                           const std::vector<double>& x_coords,
                                                           const std::vector<double>& y_coords_high,
                                                           const std::vector<double>& y_coords_low,
                                                           bool show_poc_line = true,
                                                           int num_buckets_per_candle = 8,
                                                           float opacity = 1.0f,
                                                           bool show_labels = false);

  // Indicator calculation helpers
  std::vector<double> calculate_sma(const std::vector<float>& prices, int period);
  std::vector<double> calculate_ema(const std::vector<float>& prices, int period);
  std::vector<double> calculate_ema(const std::vector<double>& prices, int period);
  std::vector<double> calculate_bollinger_upper(const std::vector<float>& prices, int period,
                                                double std_dev);
  std::vector<double> calculate_bollinger_middle(const std::vector<float>& prices, int period);
  std::vector<double> calculate_bollinger_lower(const std::vector<float>& prices, int period,
                                                double std_dev);
  std::vector<double> calculate_rsi(const std::vector<float>& prices, int period);
  std::vector<double> calculate_macd_line(const std::vector<float>& prices, int fast, int slow);
  std::vector<double> calculate_macd_signal(const std::vector<double>& macd_line, int signal);
  std::vector<double> calculate_macd_histogram(const std::vector<double>& macd_line,
                                               const std::vector<double>& signal);
  std::vector<double> calculate_stochastic_k(const std::vector<float>& highs,
                                             const std::vector<float>& lows,
                                             const std::vector<float>& closes,
                                             int k_period);
  std::vector<double> calculate_stochastic_d(const std::vector<double>& stoch_k, int d_period);
  std::vector<double> calculate_true_range(const std::vector<float>& highs,
                                           const std::vector<float>& lows,
                                           const std::vector<float>& closes);
  std::vector<double> calculate_atr(const std::vector<float>& highs,
                                    const std::vector<float>& lows,
                                    const std::vector<float>& closes,
                                    int period);

  // Fibonacci calculation
  std::vector<FibonacciLevel> calculate_fibonacci_levels(double start_price, double end_price);

  // Methods for handling mouse drag interaction for custom profile creation
  void handleMouseDragInteraction();

  // Cached indicator calculation methods
  void calculate_all_indicators(const ChartInstance& chart);

  // Method to update indicator configuration from active indicators
  void update_indicator_config_from_active();

  // Method to render the indicator overlay panel
  void render_indicator_overlay_panel();

  // Method to sync active indicators with current configuration
  void sync_active_indicators_with_config();

  // Multi-timeframe indicator methods
  void add_multi_timeframe_indicator(const std::string& name, bool visible, ImVec4 color, int period, RenderEngine::TimeFrame source_timeframe);
  void remove_multi_timeframe_indicator(int index);
  void update_multi_timeframe_indicators(const ChartInstance& chart);
  void render_multi_timeframe_indicators(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  std::vector<double> get_indicator_values_from_timeframe(const std::string& indicator_name, int period, RenderEngine::TimeFrame timeframe, uint32_t symbol_id);
  std::vector<double> calculate_cached_sma(const std::vector<float>& prices, int period);
  std::vector<double> calculate_cached_ema(const std::vector<float>& prices, int period);
  std::vector<double> calculate_cached_rsi(const std::vector<float>& prices, int period);
  std::vector<double> calculate_cached_bollinger_upper(const std::vector<float>& prices, int period,
                                                double std_dev);
  std::vector<double> calculate_cached_bollinger_middle(const std::vector<float>& prices, int period);
  std::vector<double> calculate_cached_bollinger_lower(const std::vector<float>& prices, int period,
                                                double std_dev);
  std::vector<double> calculate_cached_macd_line(const std::vector<float>& prices, int fast, int slow);
  std::vector<double> calculate_cached_macd_signal(const std::vector<double>& macd_line, int signal);
  std::vector<double> calculate_cached_macd_histogram(const std::vector<double>& macd_line,
                                               const std::vector<double>& signal);
  std::vector<double> calculate_cached_stochastic_k(const std::vector<float>& highs,
                                             const std::vector<float>& lows,
                                             const std::vector<float>& closes,
                                             int k_period);
  std::vector<double> calculate_cached_stochastic_d(const std::vector<double>& stoch_k, int d_period);
  std::vector<double> calculate_cached_true_range(const std::vector<float>& highs,
                                           const std::vector<float>& lows,
                                           const std::vector<float>& closes);
  std::vector<double> calculate_cached_atr(const std::vector<float>& highs,
                                    const std::vector<float>& lows,
                                    const std::vector<float>& closes,
                                    int period);
  void calculate_cached_bollinger_bands(const std::vector<float>& prices, int period, double std_dev);
  void calculate_cached_macd(const std::vector<float>& prices, int fast, int slow, int signal);
  void calculate_cached_stochastic(const std::vector<float>& highs,
                              const std::vector<float>& lows,
                              const std::vector<float>& closes,
                              int k_period, int d_period);
};

}  // namespace BTQuant
