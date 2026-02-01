#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "chart_manager.hpp"
#include "indicator_renderer.hpp"
#include "panel_base.hpp"

namespace BTQuant {

// Indicator configuration for chart panel
struct IndicatorConfig {
  bool show_sma_10 = false;
  bool show_sma_20 = false;
  bool show_sma_50 = false;
  bool show_ema_10 = false;
  bool show_ema_20 = false;
  bool show_ema_50 = false;
  bool show_rsi = false;
  bool show_macd = false;
  bool show_bollinger = false;
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
};

// Fibonacci Retracement Level
struct FibonacciLevel {
  double price;
  double ratio;
  const char* label;
  ImU32 color;
};

class ChartPanel : public PanelBase {
 public:
  ChartPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
             ChartManager* chart_manager);

  void update(float dt) override;
  void render() override;
  void initialize() override;

  // Chart-specific methods
  void set_symbol(const std::string& symbol, const std::string& exchange = "Binance");
  void set_timeframe(RenderEngine::TimeFrame timeframe);
  uint32_t get_chart_id() const { return chart_id_; }

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
  struct IndicatorCacheKey {
    size_t data_size;
    int period;

    bool operator==(const IndicatorCacheKey& other) const {
      return data_size == other.data_size && period == other.period;
    }
  };

  struct IndicatorCacheKeyHash {
    std::size_t operator()(const IndicatorCacheKey& k) const {
      return std::hash<size_t>{}(k.data_size) ^ (std::hash<int>{}(k.period) << 1);
    }
  };

  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_sma_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_ema_;
  std::unordered_map<IndicatorCacheKey, std::vector<double>, IndicatorCacheKeyHash> cached_rsi_;

  // Track the last known data size to detect when cache needs invalidation
  size_t last_known_data_size_ = 0;

  void render_chart_controls();
  void render_indicator_selector();
  void render_instrument_chart(const ChartInstance& chart);
  void render_candlestick(const ChartInstance& chart);

  // Indicator rendering methods
  void render_sma_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_ema_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_bollinger_bands(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_rsi_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
  void render_macd_indicator(const ChartInstance& chart, size_t start_idx, size_t end_idx);
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
  std::vector<double> calculate_bollinger_lower(const std::vector<float>& prices, int period,
                                                double std_dev);
  std::vector<double> calculate_rsi(const std::vector<float>& prices, int period);
  std::vector<double> calculate_macd_line(const std::vector<float>& prices, int fast, int slow);
  std::vector<double> calculate_macd_signal(const std::vector<double>& macd_line, int signal);
  std::vector<double> calculate_macd_histogram(const std::vector<double>& macd_line,
                                               const std::vector<double>& signal);

  // Fibonacci calculation
  std::vector<FibonacciLevel> calculate_fibonacci_levels(double start_price, double end_price);

  // Methods for handling mouse drag interaction for custom profile creation
  void handleMouseDragInteraction();
};

}  // namespace BTQuant
