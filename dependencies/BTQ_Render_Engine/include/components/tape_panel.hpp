#pragma once

#include <imgui.h>

#include <memory>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

/**
 * TapePanel - Time & Sales display
 *
 * C++26 Reactive Architecture:
 * - Subscribes to TRADE notifications from MarketDataProcessor
 * - markDirty() in callback, consumeDirty() in render()
 * - No polling timer - event-driven updates
 *
 * Shows a scrolling list of recent trades with:
 * - Timestamp (HH:MM:SS.mmm)
 * - Price
 * - Size
 * - Side (colored: green=buy, red=sell)
 */
class TapePanel : public PanelBase {
 public:
  TapePanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
            std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  ~TapePanel() override;

  void render_content() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Configuration
  // Note: With virtualized scrolling using ImGuiListClipper, we can efficiently handle 100,000+ trades
  // The MAX_VISIBLE_TRADES value represents the maximum number of trades stored in memory
  static constexpr size_t MAX_VISIBLE_TRADES = 100000;  // Increased to handle large datasets
  bool auto_scroll_ = true;

  // Audio alert configuration
  float volume_multiplier_threshold_ = 5.0f;  // Multiplier for average trade size to trigger alert
  int buy_tone_frequency_ = 800;              // Frequency in Hz for buy alerts
  int sell_tone_frequency_ = 400;             // Frequency in Hz for sell alerts
  int tone_duration_ms_ = 200;                // Duration of the alert tone in milliseconds

  // Filtering options (these hide trades that don't match)
  double min_size_filter_ = 0.0;
  double max_size_filter_ = 0.0;  // 0 means no upper limit
  std::string exchange_filter_ = "";  // Empty means no filter
  uint64_t start_time_filter_ = 0;    // 0 means no filter
  uint64_t end_time_filter_ = 0;      // 0 means no filter (until now)

  // Search options (these highlight matching trades without hiding others)
  double search_min_size_ = 0.0;
  double search_max_size_ = 0.0;  // 0 means no upper limit
  double search_min_price_ = 0.0; // 0 means no lower limit
  double search_max_price_ = 0.0; // 0 means no upper limit
  std::string search_exchange_ = "";  // Empty means no filter
  uint64_t search_start_time_ = 0;    // 0 means no filter
  uint64_t search_end_time_ = 0;      // 0 means no filter (until now)

  // UI state for filters
  bool show_histogram_ = false;
  bool show_search_ = false;  // New search panel
  char min_size_input_[32] = "0.0";
  char max_size_input_[32] = "";  // Empty means no upper limit
  char min_price_input_[32] = ""; // Empty means no lower limit
  char max_price_input_[32] = ""; // Empty means no upper limit
  char exchange_input_[64] = "";
  char start_time_input_[32] = "";
  char end_time_input_[32] = "";

  // Cached trades for rendering
  std::vector<RenderEngine::TradeData> cached_trades_;

  // Trade clustering detection parameters
  static constexpr uint64_t DEFAULT_CLUSTER_TIME_WINDOW_US = 1000000;  // 1 second in microseconds
  static constexpr int DEFAULT_MIN_CLUSTER_SIZE = 5;                   // Minimum trades to form a cluster
  static constexpr double DEFAULT_PRICE_MATCH_TOLERANCE = 0.0001;      // Tolerance for price matching

  // Runtime-configurable clustering parameters
  uint64_t cluster_time_window_us_ = DEFAULT_CLUSTER_TIME_WINDOW_US;
  int min_cluster_size_ = DEFAULT_MIN_CLUSTER_SIZE;
  double price_match_tolerance_ = DEFAULT_PRICE_MATCH_TOLERANCE;

  void render_trade_table();
  void render_controls();
  void render_filter_controls();
  void render_search_controls();  // New search controls
  void render_trade_size_histogram();
  uint64_t parseTimeString(const std::string& time_str);
  void subscribe_to_updates();

  // Audio alert methods
  void checkForLargeTradesAndAlert();
  void playTradeAlertSound(bool is_buy);

#ifdef __linux__
  void generateAndPlayTone(int frequency, int duration_ms, const std::string& type);
  void writeInt16(std::ofstream& file, int16_t value);
  void writeInt32(std::ofstream& file, int32_t value);
#endif

  // CSV Export functionality
  void exportTradesToCSV();

  // Trade clustering detection
  bool isTradeClustered(int index, const std::vector<RenderEngine::TradeData>& trades) const;

  // Slippage detection - consecutive trades < 50ms apart at different prices
  static constexpr uint64_t SLIPPAGE_TIME_THRESHOLD_US = 50000;  // 50ms in microseconds
  bool isSlippageTrade(int index, const std::vector<RenderEngine::TradeData>& trades) const;

  // Trade size histogram functions
  struct LogBucket {
    double lower_bound;
    double upper_bound;
    int count;
    double total_size;
  };

  std::vector<LogBucket> computeLogarithmicTradeSizeHistogram(const std::vector<RenderEngine::TradeData>& trades) const;

  // Trade pace indicator data structures
  struct TradePaceData {
    uint64_t timestamp;  // Time when the measurement was taken
    double trades_per_minute;  // Trades per minute at this time
  };

  // Trade pace history for different time windows
  std::vector<TradePaceData> trade_pace_1min_history_;
  std::vector<TradePaceData> trade_pace_5min_history_;
  std::vector<TradePaceData> trade_pace_15min_history_;

  // Methods for trade pace calculation
  double calculateTradesPerMinute(const std::vector<RenderEngine::TradeData>& trades, uint64_t window_microseconds) const;
  void updateTradePaceHistory();
  void renderTradePaceChart();
};

}  // namespace BTQuant
