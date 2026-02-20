#pragma once

#include <imgui.h>

#include <format>
#include <memory>


#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

// Structure to hold batched geometry data for heatmap backgrounds
struct HeatmapRect {
  ImVec2 pos_min;
  ImVec2 pos_max;
  ImU32 color;
};

// Aggregation modes for order book
enum class OrderbookAggregationMode {
  NONE,           // No aggregation
  TICK_SIZE,      // Group by tick size
  PERCENT_0_1,    // Group by 0.1%
  PERCENT_0_5,    // Group by 0.5%
  PERCENT_1,      // Group by 1%
  CUSTOM_VALUE    // Group by custom value
};

// Real-time orderbook ladder display
class OrderbookPanel : public PanelBase {
 public:
  OrderbookPanel(const PanelConfig& config, 
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
                 
  // Destructor to clean up the lock-free cache
  ~OrderbookPanel();

  void update(float dt) override;
  void render_content() override;
  void render_panel_header();

  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

 private:
  
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Configuration for number of levels to display
  int selected_levels_count_ = 20;  // Current selection (10, 20, 50, 100, 500, or -1 for unlimited)
  static constexpr int LEVEL_OPTIONS[] = {10, 20, 50, 100, 500, -1};  // -1 means unlimited
  static constexpr const char* LEVEL_OPTION_NAMES[] = {"10", "20", "50", "100", "500", "Unlimited"};

  // Orderbook aggregation configuration
  OrderbookAggregationMode aggregation_mode_ = OrderbookAggregationMode::NONE;
  double custom_aggregation_value_ = 1.0;  // Custom aggregation value when mode is CUSTOM_VALUE

  // Heatmap intensity configuration
  float heatmap_intensity_ = 1.0f;  // Sensitivity of color mapping for resting limit orders (default 1.0)
  
  int get_level_option_index();  // Helper to find the index of the current selection
  void render_orderbook_ladder(const OrderBookSnapshot& orderbook);
  void render_market_depth_chart(const OrderBookSnapshot& orderbook);

  // Getter and setter for large order threshold
  double getLargeOrderThresholdPercentage() const { return large_order_threshold_percentage_; }
  void setLargeOrderThresholdPercentage(double percentage) { large_order_threshold_percentage_ = percentage; }

  // Getter and setter for volume delta period
  uint64_t getVolumeDeltaPeriodMicroseconds() const { return volume_delta_period_us_; }
  void setVolumeDeltaPeriodMicroseconds(uint64_t microseconds) { volume_delta_period_us_ = microseconds; }

  // Aggregation functions
  double getAggregationValue(double price) const;
  std::vector<PriceLevel> aggregateOrderbookLevels(
      const std::vector<PriceLevel>& levels) const;

  // View control functions
  void center_price();
  void reset_depth();

  // Helper method to detect order flow events by comparing snapshots
  void detectOrderFlowEvents(const OrderBookSnapshot& current_snapshot, const OrderBookSnapshot& previous_snapshot);

  // Helper method to track volume changes for delta calculation
  void trackVolumeChanges(const OrderBookSnapshot& snapshot, uint64_t timestamp);

  // Helper method to update the lock-free orderbook cache
  void updateOrderbookCache();

  struct PriceLevelVolume {
    double bought = 0.0;
    double sold = 0.0;
  };

  // Structure to track order flow events per price level
  struct OrderFlowActivity {
    int additions = 0;      // Number of order additions
    int cancellations = 0;  // Number of order cancellations
    int executions = 0;     // Number of order executions
    uint64_t last_activity_ts = 0;  // Timestamp of last activity

    // Reset counters after a certain period
    void reset() {
      additions = 0;
      cancellations = 0;
      executions = 0;
    }
  };

  std::map<double, PriceLevelVolume> volume_profile_;
  std::map<double, OrderFlowActivity> order_flow_activity_;

  // Store previous snapshots for comparison
  std::map<uint32_t, OrderBookSnapshot> previous_snapshots_;

  // Batched geometry for heatmap backgrounds
  std::vector<HeatmapRect> heatmap_rects_;

  // Track processed trades to avoid double counting
  // This needs to be coordinated with the ring buffer index
  // For simplicity, we'll traverse the buffer backward until we hit a timestamp
  // older than last frame? Or if the bridge provides a monotonic index, use
  // that. HotSpineDataBridge doesn't seem to expose a monotonic trade index
  // publicly in getTradeBuffer() return type (std::span). But
  // SharedMemoryHeader has write_index.
  uint64_t last_processed_trade_ts_ = 0;
  uint64_t last_order_flow_update_ts_ = 0;

  // Large order detection
  double average_order_size_ = 0.0;
  double large_order_threshold_percentage_ = 200.0; // 200% means 2x the average size

  // Order flow visualization settings
  float order_flow_decay_factor_ = 0.95f;  // Decay factor for activity intensity over time
  uint64_t order_flow_reset_interval_ = 5000000; // Reset interval in microseconds (5 seconds)

  // Volume delta tracking for order book changes over time
  struct VolumeDeltaPoint {
    uint64_t timestamp;
    double size;
  };

  struct VolumeLevelHistory {
    std::vector<VolumeDeltaPoint> bid_history;
    std::vector<VolumeDeltaPoint> ask_history;

    void addBidPoint(uint64_t ts, double size) {
      bid_history.push_back({ts, size});
      // Keep only the last N seconds of data
      cleanupOldData(ts);
    }

    void addAskPoint(uint64_t ts, double size) {
      ask_history.push_back({ts, size});
      // Keep only the last N seconds of data
      cleanupOldData(ts);
    }

    double getBidDeltaOverPeriod(uint64_t current_time, uint64_t period_us = 5000000) const { // 5 seconds default
      uint64_t start_time = current_time >= period_us ? current_time - period_us : 0;

      // Find the size at the start of the period (closest point at or before start_time)
      double initial_size = 0.0;
      double final_size = bid_history.empty() ? 0.0 : bid_history.back().size;

      // Look for the most recent point at or before start_time
      bool found_initial = false;
      for (auto it = bid_history.rbegin(); it != bid_history.rend(); ++it) {
        if (it->timestamp <= start_time) {
          initial_size = it->size;
          found_initial = true;
          break;
        }
      }

      // If no point was found before the start time, use the first point in the period
      if (!found_initial && !bid_history.empty()) {
        for (const auto& point : bid_history) {
          if (point.timestamp >= start_time) {
            initial_size = point.size;
            break;
          }
        }
      }

      return final_size - initial_size;
    }

    double getAskDeltaOverPeriod(uint64_t current_time, uint64_t period_us = 5000000) const { // 5 seconds default
      uint64_t start_time = current_time >= period_us ? current_time - period_us : 0;

      // Find the size at the start of the period (closest point at or before start_time)
      double initial_size = 0.0;
      double final_size = ask_history.empty() ? 0.0 : ask_history.back().size;

      // Look for the most recent point at or before start_time
      bool found_initial = false;
      for (auto it = ask_history.rbegin(); it != ask_history.rend(); ++it) {
        if (it->timestamp <= start_time) {
          initial_size = it->size;
          found_initial = true;
          break;
        }
      }

      // If no point was found before the start time, use the first point in the period
      if (!found_initial && !ask_history.empty()) {
        for (const auto& point : ask_history) {
          if (point.timestamp >= start_time) {
            initial_size = point.size;
            break;
          }
        }
      }

      return final_size - initial_size;
    }

    void cleanupOldData(uint64_t current_time, uint64_t retention_period_us = 10000000) { // 10 seconds retention
      uint64_t cutoff_time = current_time >= retention_period_us ? current_time - retention_period_us : 0;

      // Remove old bid history points
      auto bid_it = std::remove_if(bid_history.begin(), bid_history.end(),
                                  [cutoff_time](const VolumeDeltaPoint& p) {
                                    return p.timestamp < cutoff_time;
                                  });
      bid_history.erase(bid_it, bid_history.end());

      // Remove old ask history points
      auto ask_it = std::remove_if(ask_history.begin(), ask_history.end(),
                                  [cutoff_time](const VolumeDeltaPoint& p) {
                                    return p.timestamp < cutoff_time;
                                  });
      ask_history.erase(ask_it, ask_history.end());
    }
  };

  // Lock-free read cache for orderbook data
  struct OrderbookCache {
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    double spread = 0.0;
    double imbalance = 0.0;
    uint64_t timestamp = 0;
    
    // Copy constructor for thread safety
    OrderbookCache(const OrderBookSnapshot& data) 
      : timestamp(data.timestamp_us) {}
      
    OrderbookCache() = default;
  };
  
  // Atomic pointer to the latest orderbook data for lock-free reads
  mutable std::atomic<OrderbookCache*> latest_orderbook_cache_{nullptr};
  
  std::map<double, VolumeLevelHistory> volume_level_history_;
  uint64_t volume_delta_period_us_ = 5000000; // 5 seconds in microseconds
};

}  // namespace BTQuant
