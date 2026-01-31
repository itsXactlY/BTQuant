#pragma once

#include "../../../ccapi/example/src/market_data_collector/market_data_types.h"
#include "../hotspine_layout_v3.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace Analytics {

struct ClusterCell {
    std::atomic<double> total_volume{0.0};
    std::atomic<double> buy_volume{0.0};
    std::atomic<double> sell_volume{0.0};
    std::atomic<int> trade_count{0};
    std::atomic<int> buy_trade_count{0};
    std::atomic<int> sell_trade_count{0};
    std::atomic<double> max_single_trade_volume{0.0};
    std::atomic<double> sum_of_volumes{0.0};  // for average calculations
};

}

namespace Analytics {

class ClusterEngine {
public:
  explicit ClusterEngine(double tick_size)
      : tick_size_(tick_size), min_tick_index_(0), session_start_us_(0) {
    // Reserve some initial space to avoid immediate reallocations
    canvas_.reserve(10000);
  }

  void set_session_start(int64_t start_us) { session_start_us_ = start_us; }

  // O(1) mostly, amortized
  void process_trade(const MarketData::Trade &trade) {
    if (session_start_us_ == 0) {
      session_start_us_ = trade.timestamp_us;
    }

    int64_t abs_tick_index =
        static_cast<int64_t>(std::round(trade.price / tick_size_));

    // Initialize min_tick_index_ on first trade
    if (canvas_.empty()) {
      min_tick_index_ = abs_tick_index - 100; // start with some padding
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
    auto &node = canvas_[relative_index];
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

  void snapshot_to_viewport(HotSpine::V3::ClusterColumn &out,
                            double center_price) {
    int64_t center_idx =
        static_cast<int64_t>(std::round(center_price / tick_size_));
    int64_t start_abs_index = center_idx - (HotSpine::V3::VIEWPORT_ROWS / 2);

    out.tick_size = tick_size_;
    out.base_tick_index = start_abs_index;

    // Note: out.open/high/low/close are not populated here as they depend on
    // session/candle context, which this specific rasterizer doesn't
    // necessarily track. The processor loop should populate them.

    for (size_t i = 0; i < HotSpine::V3::VIEWPORT_ROWS; ++i) {
      int64_t current_abs_idx = start_abs_index + i;
      int64_t relative_idx = current_abs_idx - min_tick_index_;

      auto &row = out.rows[i];

      if (relative_idx >= 0 &&
          static_cast<size_t>(relative_idx) < canvas_.size()) {
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
} // namespace Analytics
