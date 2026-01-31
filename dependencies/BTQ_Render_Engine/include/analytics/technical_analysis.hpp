#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "vulkan_base_types.hpp"

namespace BTQuant {

// Forward declaration
namespace RenderEngine {
struct TradeData;
struct OrderbookData;
}  // namespace RenderEngine

// ============================================================================
// Technical Indicators
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

  static IndicatorResult simple_moving_average(const std::vector<OHLCV>& data, int period);
  static IndicatorResult exponential_moving_average(const std::vector<OHLCV>& data, int period);
  static std::vector<IndicatorResult> bollinger_bands(const std::vector<OHLCV>& data, int period,
                                                      double std_dev = 2.0);
  static IndicatorResult rsi(const std::vector<OHLCV>& data, int period = 14);
  static std::vector<IndicatorResult> macd(const std::vector<OHLCV>& data, int fast_period = 12,
                                           int slow_period = 26, int signal_period = 9);
  static std::vector<IndicatorResult> stochastic(const std::vector<OHLCV>& data, int k_period = 14,
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

// ============================================================================
// Volume Profile Analyzer
// ============================================================================

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
  VolumeProfile calculate_volume_profile(const std::vector<TechnicalIndicators::OHLCV>& candles,
                                         const std::vector<RenderEngine::TradeData>& trades);
  std::vector<VolumeImbalance> detect_volume_imbalances(const VolumeProfile& profile,
                                                        double threshold = 2.0);

 private:
  double tick_size_;
  void calculate_value_area(const std::vector<VolumeNode>& nodes, VolumeProfile& profile);
};

// ============================================================================
// Market Depth Analyzer
// ============================================================================

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
  MarketDepthSnapshot create_depth_snapshot(const RenderEngine::OrderbookData& orderbook);
  DepthAnalysis analyze_market_depth(const MarketDepthSnapshot& snapshot);
  std::vector<LiquidityGap> detect_liquidity_gaps(const MarketDepthSnapshot& snapshot,
                                                  double min_gap_size = 0.01);

 private:
  double find_support_level(const std::vector<DepthLevel>& bids);
  double find_resistance_level(const std::vector<DepthLevel>& asks);
  double calculate_liquidity_score(const MarketDepthSnapshot& snapshot);
  double estimate_market_impact(const MarketDepthSnapshot& snapshot, double order_value);
  std::vector<double> find_significant_levels(const MarketDepthSnapshot& snapshot);
};

// ============================================================================
// Pattern Recognizer
// ============================================================================

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
  std::vector<Pattern> detect_patterns(const std::vector<TechnicalIndicators::OHLCV>& data);

 private:
  std::vector<Pattern> detect_double_top(const std::vector<TechnicalIndicators::OHLCV>& data);
  std::vector<Pattern> detect_double_bottom(const std::vector<TechnicalIndicators::OHLCV>& data);
  std::vector<Pattern> detect_head_and_shoulders(
      const std::vector<TechnicalIndicators::OHLCV>& data);
  std::vector<Pattern> detect_triangles(const std::vector<TechnicalIndicators::OHLCV>& data);
  std::vector<Pattern> detect_candlestick_patterns(
      const std::vector<TechnicalIndicators::OHLCV>& data);
  std::vector<size_t> find_peaks(const std::vector<TechnicalIndicators::OHLCV>& data,
                                 bool find_highs);
  double find_valley_between(const std::vector<TechnicalIndicators::OHLCV>& data, size_t start,
                             size_t end);
  bool is_local_high(const std::vector<TechnicalIndicators::OHLCV>& data, size_t index);
  bool is_local_low(const std::vector<TechnicalIndicators::OHLCV>& data, size_t index);
  std::pair<double, double> calculate_trend_line(const std::vector<glm::vec2>& points);
  bool is_hammer(const TechnicalIndicators::OHLCV& candle);
  bool is_doji(const TechnicalIndicators::OHLCV& candle);
  bool is_bullish_engulfing(const TechnicalIndicators::OHLCV& prev,
                            const TechnicalIndicators::OHLCV& curr);
  bool is_bearish_engulfing(const TechnicalIndicators::OHLCV& prev,
                            const TechnicalIndicators::OHLCV& curr);
  double calculate_double_top_confidence(const std::vector<TechnicalIndicators::OHLCV>& data,
                                         size_t peak1, size_t valley, size_t peak2);
  double calculate_double_bottom_confidence(const std::vector<TechnicalIndicators::OHLCV>& data,
                                            size_t trough1, size_t peak, size_t trough2);
  double calculate_head_shoulders_confidence(const std::vector<TechnicalIndicators::OHLCV>& data,
                                             size_t left, size_t head, size_t right);
};

}  // namespace BTQuant
