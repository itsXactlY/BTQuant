#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "../../../../dependencies/ccapi/example/src/market_data_collector/market_data_types.h"
#include "../data/VolumeDataTypes.h"  // Include for Data::TimeAggregationType
#include "../hotspine_layout_v3.hpp"

namespace Analytics {

struct ClusterCell {
  // Atomic values to replace mutex-protected data
  // Note: std::atomic<double> doesn't support fetch_add in C++17, so we use std::atomic<uint64_t> 
  // and reinterpret_cast to handle double values for atomic operations
  std::atomic<uint64_t> total_volume_raw{0};  // reinterpret_cast<double> of the bit representation
  std::atomic<uint64_t> buy_volume_raw{0};
  std::atomic<uint64_t> sell_volume_raw{0};
  std::atomic<int> trade_count{0};
  std::atomic<int> buy_trade_count{0};
  std::atomic<int> sell_trade_count{0};
  std::atomic<uint64_t> max_single_trade_volume_raw{0};
  std::atomic<uint64_t> sum_of_volumes_raw{0};  // for average calculations

  // Fields for statistical calculations - using atomic operations for sums
  // For the vector of prices, we'll use a different approach since there's no atomic vector
  // We'll store the count of prices separately and use atomic operations for sums
  std::atomic<uint64_t> sum_of_prices_raw{0};   // Sum of all prices for mean calculation
  std::atomic<uint64_t> sum_of_squared_prices_raw{0};  // Sum of squared prices for variance calculation
  std::atomic<int> price_count{0};  // Count of prices added

  // Helper methods to safely access double values
  double getTotalVolume() const {
    uint64_t raw_val = total_volume_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setTotalVolume(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    total_volume_raw.store(raw_val, std::memory_order_release);
  }
  
  double getBuyVolume() const {
    uint64_t raw_val = buy_volume_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setBuyVolume(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    buy_volume_raw.store(raw_val, std::memory_order_release);
  }
  
  double getSellVolume() const {
    uint64_t raw_val = sell_volume_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setSellVolume(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    sell_volume_raw.store(raw_val, std::memory_order_release);
  }
  
  double getMaxSingleTradeVolume() const {
    uint64_t raw_val = max_single_trade_volume_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setMaxSingleTradeVolume(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    max_single_trade_volume_raw.store(raw_val, std::memory_order_release);
  }
  
  double getSumOfVolumes() const {
    uint64_t raw_val = sum_of_volumes_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setSumOfVolumes(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    sum_of_volumes_raw.store(raw_val, std::memory_order_release);
  }
  
  double getSumOfPrices() const {
    uint64_t raw_val = sum_of_prices_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setSumOfPrices(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    sum_of_prices_raw.store(raw_val, std::memory_order_release);
  }
  
  double getSumOfSquaredPrices() const {
    uint64_t raw_val = sum_of_squared_prices_raw.load(std::memory_order_acquire);
    return *reinterpret_cast<const double*>(&raw_val);
  }
  
  void setSumOfSquaredPrices(double val) {
    uint64_t raw_val = *reinterpret_cast<const uint64_t*>(&val);
    sum_of_squared_prices_raw.store(raw_val, std::memory_order_release);
  }

  // Atomic add operations for doubles using CAS loop
  void addTotalVolume(double increment) {
    uint64_t expected = total_volume_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!total_volume_raw.compare_exchange_weak(expected, new_val, 
                                                     std::memory_order_release, 
                                                     std::memory_order_acquire));
  }
  
  void addBuyVolume(double increment) {
    uint64_t expected = buy_volume_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!buy_volume_raw.compare_exchange_weak(expected, new_val, 
                                                   std::memory_order_release, 
                                                   std::memory_order_acquire));
  }
  
  void addSellVolume(double increment) {
    uint64_t expected = sell_volume_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!sell_volume_raw.compare_exchange_weak(expected, new_val, 
                                                    std::memory_order_release, 
                                                    std::memory_order_acquire));
  }
  
  void addSumOfVolumes(double increment) {
    uint64_t expected = sum_of_volumes_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!sum_of_volumes_raw.compare_exchange_weak(expected, new_val, 
                                                       std::memory_order_release, 
                                                       std::memory_order_acquire));
  }
  
  void addSumOfPrices(double increment) {
    uint64_t expected = sum_of_prices_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!sum_of_prices_raw.compare_exchange_weak(expected, new_val, 
                                                      std::memory_order_release, 
                                                      std::memory_order_acquire));
  }
  
  void addSumOfSquaredPrices(double increment) {
    uint64_t expected = sum_of_squared_prices_raw.load(std::memory_order_acquire);
    uint64_t new_val;
    double expected_dbl, new_dbl;
    
    do {
      expected_dbl = *reinterpret_cast<double*>(&expected);
      new_dbl = expected_dbl + increment;
      new_val = *reinterpret_cast<uint64_t*>(&new_dbl);
    } while (!sum_of_squared_prices_raw.compare_exchange_weak(expected, new_val, 
                                                              std::memory_order_release, 
                                                              std::memory_order_acquire));
  }

  // Define copy constructor and assignment operator
  ClusterCell() = default;

  // Copy constructor - only copies the data values
  ClusterCell(const ClusterCell& other)
      : total_volume_raw(other.total_volume_raw.load()),
        buy_volume_raw(other.buy_volume_raw.load()),
        sell_volume_raw(other.sell_volume_raw.load()),
        trade_count(other.trade_count.load()),
        buy_trade_count(other.buy_trade_count.load()),
        sell_trade_count(other.sell_trade_count.load()),
        max_single_trade_volume_raw(other.max_single_trade_volume_raw.load()),
        sum_of_volumes_raw(other.sum_of_volumes_raw.load()),
        sum_of_prices_raw(other.sum_of_prices_raw.load()),
        sum_of_squared_prices_raw(other.sum_of_squared_prices_raw.load()),
        price_count(other.price_count.load()) {}

  // Assignment operator
  ClusterCell& operator=(const ClusterCell& other) {
    if (this != &other) {
      total_volume_raw.store(other.total_volume_raw.load());
      buy_volume_raw.store(other.buy_volume_raw.load());
      sell_volume_raw.store(other.sell_volume_raw.load());
      trade_count.store(other.trade_count.load());
      buy_trade_count.store(other.buy_trade_count.load());
      sell_trade_count.store(other.sell_trade_count.load());
      max_single_trade_volume_raw.store(other.max_single_trade_volume_raw.load());
      sum_of_volumes_raw.store(other.sum_of_volumes_raw.load());
      sum_of_prices_raw.store(other.sum_of_prices_raw.load());
      sum_of_squared_prices_raw.store(other.sum_of_squared_prices_raw.load());
      price_count.store(other.price_count.load());
    }
    return *this;
  }
};

}  // namespace Analytics

namespace Analytics {

class ClusterEngine {
 public:
  explicit ClusterEngine(double tick_size)
      : tick_size_(tick_size), min_tick_index_(0), session_start_us_(0) {
    // Reserve some initial space to avoid immediate reallocations
    canvas_.reserve(10000);
    cluster_canvas_.reserve(10000);  // Reserve similar space for cluster canvas
  }

  void set_session_start(int64_t start_us) { session_start_us_ = start_us; }

  // O(1) mostly, amortized
  void process_trade(const MarketData::Trade& trade) {
    if (session_start_us_ == 0) {
      session_start_us_ = trade.timestamp_us;
    }

    int64_t abs_tick_index = static_cast<int64_t>(std::round(trade.price / tick_size_));

    // Initialize min_tick_index_ on first trade
    if (canvas_.empty()) {
      min_tick_index_ = abs_tick_index - 100;  // start with some padding
      canvas_.resize(200);
    }

    int64_t relative_index = abs_tick_index - min_tick_index_;

    // Handle Expansion Low
    if (relative_index < 0) {
      size_t deficit = -relative_index;
      size_t padding = 100;
      size_t final_insert = deficit + padding;

      canvas_.insert(canvas_.begin(), final_insert, HotSpine::V3::VolumeNode{});
      min_tick_index_ -= (int64_t)final_insert;
      relative_index = abs_tick_index - min_tick_index_;
    }

    // Handle Expansion High
    if (static_cast<size_t>(relative_index) >= canvas_.size()) {
      size_t needed = static_cast<size_t>(relative_index) - canvas_.size() + 1;
      size_t padding = 100;
      canvas_.resize(canvas_.size() + needed + padding);
    }

    // Update Node
    auto& node = canvas_[relative_index];
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

  // Method to pull VolumeData from ClusterEngine without mutexes using atomic operations
  double getVolumeDataAt(int64_t price_level, int time_bucket, BTQuant::Data::VolumeAnalysisType vol_type) const;

  // Getter methods for accessing internal properties
  double get_tick_size() const { return tick_size_; }
  int64_t get_min_tick_index() const { return min_tick_index_; }

  // Calculate standard deviation for a specific price level and time bucket
  double calculateStandardDeviation(int64_t price_level, int time_bucket) const;

  // Calculate median price for a specific price level and time bucket
  double calculateMedianPrice(int64_t price_level, int time_bucket) const;

  void snapshot_to_viewport(HotSpine::V3::ClusterColumn& out, double center_price) {
    int64_t center_idx = static_cast<int64_t>(std::round(center_price / tick_size_));
    int64_t start_abs_index = center_idx - (HotSpine::V3::VIEWPORT_ROWS / 2);

    out.tick_size = tick_size_;
    out.base_tick_index = start_abs_index;

    // Note: out.open/high/low/close are not populated here as they depend on
    // session/candle context, which this specific rasterizer doesn't
    // necessarily track. The processor loop should populate them.

    for (size_t i = 0; i < HotSpine::V3::VIEWPORT_ROWS; ++i) {
      int64_t current_abs_idx = start_abs_index + i;
      int64_t relative_idx = current_abs_idx - min_tick_index_;

      auto& row = out.rows[i];

      if (relative_idx >= 0 && static_cast<size_t>(relative_idx) < canvas_.size()) {
        row = canvas_[relative_idx];
      } else {
        row = HotSpine::V3::VolumeNode{};
      }
    }
  }

 private:
  double tick_size_;
  int64_t min_tick_index_;
  int64_t session_start_us_;
  std::vector<HotSpine::V3::VolumeNode> canvas_;

  // Additional data structure for cluster cells with time buckets
  std::vector<std::vector<ClusterCell>> cluster_canvas_;  // [price_level][time_bucket]
};
}  // namespace Analytics
