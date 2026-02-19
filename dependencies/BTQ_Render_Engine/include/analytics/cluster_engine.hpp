#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

#include "../../../../dependencies/ccapi/example/src/market_data_collector/market_data_types.h"
#include "../data/VolumeDataTypes.h"  // Include for Data::TimeAggregationType
#include "../hotspine_layout_v3.hpp"
#include "../trading/HotspineData.h"  // For CandleCluster with CAS operations

namespace Analytics {

struct ClusterCell {
  mutable std::mutex volume_mutex;  // Mutex to protect double values
  double total_volume{0.0};
  double buy_volume{0.0};
  double sell_volume{0.0};
  std::atomic<int> trade_count{0};
  std::atomic<int> buy_trade_count{0};
  std::atomic<int> sell_trade_count{0};
  std::atomic<double> max_single_trade_volume{0.0};
  double sum_of_volumes{0.0};  // for average calculations

  // Fields for statistical calculations
  std::vector<double> prices;  // Store prices for statistical calculations
  double sum_of_prices{0.0};   // Sum of all prices for mean calculation
  double sum_of_squared_prices{0.0};  // Sum of squared prices for variance calculation

  // Define copy constructor and assignment operator to handle mutex properly
  ClusterCell() = default;

  // Copy constructor - only copies the data values, not the mutex
  ClusterCell(const ClusterCell& other)
      : total_volume(other.total_volume),
        buy_volume(other.buy_volume),
        sell_volume(other.sell_volume),
        trade_count(other.trade_count.load()),
        buy_trade_count(other.buy_trade_count.load()),
        sell_trade_count(other.sell_trade_count.load()),
        max_single_trade_volume(other.max_single_trade_volume.load()),
        sum_of_volumes(other.sum_of_volumes),
        prices(other.prices),
        sum_of_prices(other.sum_of_prices),
        sum_of_squared_prices(other.sum_of_squared_prices) {}

  // Assignment operator
  ClusterCell& operator=(const ClusterCell& other) {
    if (this != &other) {
      std::lock_guard<std::mutex> lock_this(volume_mutex);
      std::lock_guard<std::mutex> lock_other(other.volume_mutex);

      total_volume = other.total_volume;
      buy_volume = other.buy_volume;
      sell_volume = other.sell_volume;
      trade_count.store(other.trade_count.load());
      buy_trade_count.store(other.buy_trade_count.load());
      sell_trade_count.store(other.sell_trade_count.load());
      max_single_trade_volume.store(other.max_single_trade_volume.load());
      sum_of_volumes = other.sum_of_volumes;
      prices = other.prices;
      sum_of_prices = other.sum_of_prices;
      sum_of_squared_prices = other.sum_of_squared_prices;
    }
    return *this;
  }
};

}  // namespace Analytics

namespace Analytics {

class ClusterEngine {
 public:
  explicit ClusterEngine(double tick_size)
      : tick_size_(tick_size), min_tick_index_(0), session_start_us_(0), session_low_(0.0) {
    // Reserve some initial space to avoid immediate reallocations
    canvas_.reserve(10000);
    cluster_canvas_.reserve(10000);  // Reserve similar space for cluster canvas
  }

  void set_session_start(int64_t start_us) { session_start_us_ = start_us; }

  // O(1) mostly, amortized
  void process_trade(const MarketData::Trade& trade) {
    if (session_start_us_ == 0) {
      session_start_us_ = trade.timestamp_us;
      session_low_ = trade.price;
    }

    // Track session low for O(1) binning
    if (trade.price < session_low_) {
      session_low_ = trade.price;
    }

    // O(1) constant-time price binning
    int64_t bin_index = static_cast<int64_t>((trade.price - session_low_) / tick_size_);

    // Handle Expansion Low (price went below session_low_)
    if (bin_index < 0) {
      session_low_ = trade.price;
      bin_index = 0;
    }

    // Initialize canvas_ on first trade
    if (canvas_.empty()) {
      size_t initial_size = 200;
      canvas_.resize(initial_size);
    }

    // Handle Expansion High
    if (static_cast<size_t>(bin_index) >= canvas_.size()) {
      size_t needed = static_cast<size_t>(bin_index) - canvas_.size() + 1;
      size_t padding = 100;
      canvas_.resize(canvas_.size() + needed + padding);
    }

    // Update Node
    auto& node = canvas_[bin_index];
    if (trade.is_buyer_maker) {
      // Buyer is maker -> Seller is taker -> Sell Volume
      node.sell_vol += static_cast<float>(trade.quantity);
    } else {
      // Seller is maker -> Buyer is taker -> Buy Volume
      node.buy_vol += static_cast<float>(trade.quantity);
    }
    node.trade_count++;

    // Update TPO Bits (30 min brackets)
    constexpr int64_t INTERVAL_US = 30LL * 60 * 1000000;
    int64_t elapsed = trade.timestamp_us - session_start_us_;
    if (elapsed >= 0) {
      int bucket = static_cast<int>(elapsed / INTERVAL_US);
      if (bucket >= 0 && bucket < 16) {
        node.tpo_bits |= (1 << bucket);
      }
    }
  }

  // Process trade with atomic updates to ClusterCell counters for given price level and time bucket
  void processTrade(const MarketData::Trade& trade, int time_bucket);

  // Process trade with time aggregation based on different aggregation types
  void processTradeWithTimeAggregation(const MarketData::Trade& trade,
                                       BTQuant::Data::TimeAggregationType agg_type,
                                       int n_contracts = 1000, int n_ticks = 100);

  // Process trade into active CandleCluster using Compare-And-Swap (CAS) atomic operations
  // This method is thread-safe and lock-free for concurrent volume accumulation
  void processTradeToCandleClusterCAS(const MarketData::Trade& trade,
                                      BTQuant::RenderEngine::CandleCluster& cluster);

  // Get reference to the active candle cluster for direct CAS operations
  BTQuant::RenderEngine::CandleCluster& getActiveCandleCluster() { return active_cluster_; }
  const BTQuant::RenderEngine::CandleCluster& getActiveCandleCluster() const { return active_cluster_; }

  // Set the active candle cluster parameters
  void setActiveCandleCluster(float center_price, float tick_size, uint64_t start_time_ns);

  // Detect diagonal imbalances by comparing buy_volume at price P with sell_volume at price P-1
  std::vector<std::tuple<int64_t, int, double, double, double>> detect_diagonal_imbalances(
      double threshold = 3.0) const;

  // Detect stacked imbalances by comparing buy/sell volumes at same price level across consecutive
  // time bars
  std::vector<std::tuple<int64_t, int, double, double, double>> detect_stacked_imbalances(
      double threshold = 3.0) const;

  // Detect exhaustion moves by identifying extreme buying/selling pressure followed by weakness
  std::vector<std::tuple<int64_t, int, double, double, double, std::string>> detect_exhaustion_moves(
      double threshold = 3.0) const;

  // Getter method to access the cluster canvas for visualization
  const std::vector<std::vector<ClusterCell>>& getClusterCanvas() const { return cluster_canvas_; }

  // Getter method to access session low price
  double getSessionLow() const { return session_low_; }

  // Calculate standard deviation for a specific price level and time bucket
  double calculateStandardDeviation(int64_t price_level, int time_bucket) const;

  // Calculate median price for a specific price level and time bucket
  double calculateMedianPrice(int64_t price_level, int time_bucket) const;

  void snapshot_to_viewport(HotSpine::V3::ClusterColumn& out, double center_price) {
    int64_t center_idx = static_cast<int64_t>(std::round(center_price / tick_size_));
    int64_t start_abs_index = center_idx - (HotSpine::V3::VIEWPORT_ROWS / 2);

    out.tick_size = tick_size_;
    out.base_tick_index = start_abs_index;

    // Calculate session low tick index for converting absolute tick indices to bin indices
    int64_t session_low_tick = static_cast<int64_t>(std::round(session_low_ / tick_size_));

    // Note: out.open/high/low/close are not populated here as they depend on
    // session/candle context, which this specific rasterizer doesn't
    // necessarily track. The processor loop should populate them.

    for (size_t i = 0; i < HotSpine::V3::VIEWPORT_ROWS; ++i) {
      int64_t current_abs_idx = start_abs_index + i;
      int64_t bin_index = current_abs_idx - session_low_tick;

      auto& row = out.rows[i];

      if (bin_index >= 0 && static_cast<size_t>(bin_index) < canvas_.size()) {
        row = canvas_[bin_index];
      } else {
        row = HotSpine::V3::VolumeNode{};
      }
    }
  }

 private:
  double tick_size_;
  int64_t min_tick_index_;
  int64_t session_start_us_;
  double session_low_;  // Lowest price seen in session for O(1) binning
  std::vector<HotSpine::V3::VolumeNode> canvas_;

  // Additional data structure for cluster cells with time buckets
  std::vector<std::vector<ClusterCell>> cluster_canvas_;  // [price_level][time_bucket]

  // Active candle cluster for CAS-based volume accumulation
  BTQuant::RenderEngine::CandleCluster active_cluster_;
  
  // Phase 5.5: Cumulative Volume Delta (CVD) - atomic global delta tracker
  // Adds Ask hits (buys), subtracts Bid hits (sells)
  std::atomic<int64_t> cumulative_volume_delta_{0};
  
  // Phase 5.4: Dynamic POC tracking - maintain running maximum without sorting
  // Uses atomic compare-and-swap to update POC price level
  std::atomic<double> poc_price_level_{0.0};
  std::atomic<double> poc_max_volume_{0.0};
  
 public:
  // Phase 5.5: Get/Set cumulative volume delta (thread-safe atomic)
  int64_t getCumulativeVolumeDelta() const { 
    return cumulative_volume_delta_.load(std::memory_order_acquire); 
  }
  
  void addBuyVolume(int64_t volume) {
    cumulative_volume_delta_.fetch_add(volume, std::memory_order_release);
  }
  
  void addSellVolume(int64_t volume) {
    cumulative_volume_delta_.fetch_sub(volume, std::memory_order_release);
  }
  
  // Phase 5.4: Dynamic POC update using CAS
  // Returns true if this trade set a new POC
  bool tryUpdatePOC(double price_level, double volume) {
    double current_max = poc_max_volume_.load(std::memory_order_acquire);
    while (volume > current_max) {
      if (poc_max_volume_.compare_exchange_weak(current_max, volume,
          std::memory_order_release, std::memory_order_acquire)) {
        poc_price_level_.store(price_level, std::memory_order_release);
        return true;
      }
      // current_max is updated by CAS on failure
    }
    return false;
  }
  
  double getPOCPriceLevel() const { 
    return poc_price_level_.load(std::memory_order_acquire); 
  }
  
  double getPOCMaxVolume() const { 
    return poc_max_volume_.load(std::memory_order_acquire); 
  }
  
  void resetPOC() {
    poc_price_level_.store(0.0, std::memory_order_release);
    poc_max_volume_.store(0.0, std::memory_order_release);
  }
};
}  // namespace Analytics
