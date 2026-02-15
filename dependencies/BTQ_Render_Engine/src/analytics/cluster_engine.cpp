#include "../../include/analytics/cluster_engine.hpp"

#include <limits>
#include <map>
#include <mutex>
#include <tuple>

namespace Analytics {

void ClusterEngine::processTrade(const MarketData::Trade& trade, int time_bucket) {
  if (session_start_us_ == 0) {
    session_start_us_ = trade.timestamp_us;
  }

  int64_t abs_tick_index = static_cast<int64_t>(std::round(trade.price / tick_size_));

  // Initialize min_tick_index_ on first trade
  if (cluster_canvas_.empty()) {
    min_tick_index_ = abs_tick_index - 100;  // start with some padding
    cluster_canvas_.resize(200, std::vector<ClusterCell>(16));
  }

  int64_t relative_index = abs_tick_index - min_tick_index_;

  // Handle Expansion Low
  if (relative_index < 0) {
    size_t deficit = -relative_index;
    size_t padding = 100;
    size_t final_insert = deficit + padding;

    cluster_canvas_.insert(cluster_canvas_.begin(), final_insert, std::vector<ClusterCell>(16));
    min_tick_index_ -= (int64_t)final_insert;
    relative_index = abs_tick_index - min_tick_index_;
  }

  // Handle Expansion High
  if (static_cast<size_t>(relative_index) >= cluster_canvas_.size()) {
    size_t needed = static_cast<size_t>(relative_index) - cluster_canvas_.size() + 1;
    size_t padding = 100;
    cluster_canvas_.resize(cluster_canvas_.size() + needed + padding, std::vector<ClusterCell>(16));
  }

  // Ensure time bucket is within bounds
  int adjusted_time_bucket = time_bucket;
  if (adjusted_time_bucket < 0) {
    adjusted_time_bucket = 0;
  } else if (adjusted_time_bucket >= 16) {
    adjusted_time_bucket = 15;
  }

  // Get reference to the cluster cell for this price level and time bucket
  auto& cell = cluster_canvas_[relative_index][adjusted_time_bucket];

  // Atomically update the volume counters and price statistics
  cell.total_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
  cell.sum_of_volumes.fetch_add(trade.quantity, std::memory_order_relaxed);

  if (trade.is_buyer_maker) {
    cell.sell_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
  } else {
    cell.buy_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
  }

  // Update price statistics for standard deviation and median calculations
  // Using atomic operations for the sums
  cell.sum_of_prices.fetch_add(trade.price, std::memory_order_relaxed);
  cell.sum_of_squared_prices.fetch_add(trade.price * trade.price, std::memory_order_relaxed);
  cell.price_count.fetch_add(1, std::memory_order_relaxed);

  // Atomically update trade count counters
  if (trade.is_buyer_maker) {
    cell.sell_trade_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    cell.buy_trade_count.fetch_add(1, std::memory_order_relaxed);
  }

  cell.trade_count.fetch_add(1, std::memory_order_relaxed);

  // Atomically update max single trade volume using compare-and-swap
  double current_max = cell.max_single_trade_volume.load(std::memory_order_acquire);
  double new_value = trade.quantity;
  while (new_value > current_max) {
    if (cell.max_single_trade_volume.compare_exchange_weak(
            current_max, new_value, std::memory_order_release, std::memory_order_acquire)) {
      break;
    }
  }
}

std::vector<std::tuple<int64_t, int, double, double, double>>
ClusterEngine::detect_diagonal_imbalances(double threshold) const {
  std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

  // Iterate through price levels (rows) and time buckets (columns)
  // Compare buy_volume at price P with sell_volume at price P-1
  for (size_t price_idx = 1; price_idx < cluster_canvas_.size();
       ++price_idx) {  // Start from 1 to compare with P-1
    for (int time_bucket = 0; time_bucket < 16; ++time_bucket) {
      // Get buy volume at current price level P (using atomic loads)
      double buy_volume_at_p = cluster_canvas_[price_idx][time_bucket].buy_volume.load(std::memory_order_acquire);
      double sell_volume_at_p = cluster_canvas_[price_idx][time_bucket].sell_volume.load(std::memory_order_acquire);

      // Get sell volume at previous price level P-1 (using atomic loads)
      double sell_volume_at_p_minus_1 = cluster_canvas_[price_idx - 1][time_bucket].sell_volume.load(std::memory_order_acquire);
      double buy_volume_at_p_minus_1 = cluster_canvas_[price_idx - 1][time_bucket].buy_volume.load(std::memory_order_acquire);

      // Calculate ratio of buy_volume at P to sell_volume at P-1
      if (sell_volume_at_p_minus_1 > 0) {
        double ratio = buy_volume_at_p / sell_volume_at_p_minus_1;

        // Check if ratio exceeds threshold
        if (ratio > threshold) {
          // Store: absolute_price_index, time_bucket, buy_volume_at_P, sell_volume_at_P_minus_1,
          // ratio
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket, buy_volume_at_p, sell_volume_at_p_minus_1, ratio);
        }
      }

      // Also check the reverse diagonal: sell_volume at price P with buy_volume at price P-1
      if (buy_volume_at_p_minus_1 > 0) {
        double reverse_ratio = sell_volume_at_p / buy_volume_at_p_minus_1;

        if (reverse_ratio > threshold) {
          // Store: absolute_price_index, time_bucket, sell_volume_at_P, buy_volume_at_P_minus_1,
          // ratio
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket, sell_volume_at_p, buy_volume_at_p_minus_1, reverse_ratio);
        }
      }

      // Enhanced diagonal detection: Look for multi-level diagonal patterns
      // Check for buy volume at P compared to sell volume at P-2 (extended diagonal)
      if (price_idx >= 2) {
        double sell_volume_at_p_minus_2 = cluster_canvas_[price_idx - 2][time_bucket].sell_volume.load(std::memory_order_acquire);

        if (sell_volume_at_p_minus_2 > 0) {
          double extended_ratio = buy_volume_at_p / sell_volume_at_p_minus_2;

          if (extended_ratio > threshold * 1.5) { // Higher threshold for extended patterns
            imbalances.emplace_back(
                static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                time_bucket, buy_volume_at_p, sell_volume_at_p_minus_2, extended_ratio);
          }
        }
      }

      // Check for sell volume at P compared to buy volume at P-2 (reverse extended diagonal)
      if (price_idx >= 2) {
        double buy_volume_at_p_minus_2 = cluster_canvas_[price_idx - 2][time_bucket].buy_volume.load(std::memory_order_acquire);

        if (buy_volume_at_p_minus_2 > 0) {
          double reverse_extended_ratio = sell_volume_at_p / buy_volume_at_p_minus_2;

          if (reverse_extended_ratio > threshold * 1.5) { // Higher threshold for extended patterns
            imbalances.emplace_back(
                static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                time_bucket, sell_volume_at_p, buy_volume_at_p_minus_2, reverse_extended_ratio);
          }
        }
      }
    }
  }

  return imbalances;
}

std::vector<std::tuple<int64_t, int, double, double, double>>
ClusterEngine::detect_stacked_imbalances(double threshold) const {
  std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

  // Vertical analysis comparing buy/sell at same price across consecutive bars
  // Iterate through price levels (rows) and time buckets (columns)
  for (size_t price_idx = 0; price_idx < cluster_canvas_.size(); ++price_idx) {
    // Compare consecutive time buckets for the same price level
    for (int time_bucket = 1; time_bucket < 16;
         ++time_bucket) {  // Start from 1 to compare with previous time bucket
      // Get buy and sell volumes for current time bucket (using atomic loads)
      double buy_volume_current = cluster_canvas_[price_idx][time_bucket].buy_volume.load(std::memory_order_acquire);
      double sell_volume_current = cluster_canvas_[price_idx][time_bucket].sell_volume.load(std::memory_order_acquire);

      // Get buy and sell volumes for previous time bucket (using atomic loads)
      double buy_volume_previous = cluster_canvas_[price_idx][time_bucket - 1].buy_volume.load(std::memory_order_acquire);
      double sell_volume_previous = cluster_canvas_[price_idx][time_bucket - 1].sell_volume.load(std::memory_order_acquire);

      // Primary stacked imbalance detection: Compare buy/sell at same price across consecutive bars
      // Bullish stacked imbalance: Significant increase in buy volume compared to previous bar's
      // buy volume
      if (buy_volume_previous > 0 && buy_volume_current > buy_volume_previous * threshold) {
        double buy_imbalance_ratio = buy_volume_current / buy_volume_previous;
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            buy_volume_current,                                 // Current buy volume
            buy_volume_previous,                                // Previous buy volume
            buy_imbalance_ratio  // Ratio of current to previous buy volume
        );
      }

      // Bearish stacked imbalance: Significant increase in sell volume compared to previous bar's
      // sell volume
      if (sell_volume_previous > 0 && sell_volume_current > sell_volume_previous * threshold) {
        double sell_imbalance_ratio = sell_volume_current / sell_volume_previous;
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            sell_volume_current,                                // Current sell volume
            sell_volume_previous,                               // Previous sell volume
            sell_imbalance_ratio  // Ratio of current to previous sell volume
        );
      }

      // Cross-imbalance detection: Compare current buy volume vs previous sell volume (potential
      // bullish signal)
      if (sell_volume_previous > 0 && buy_volume_current > sell_volume_previous * threshold) {
        double buy_vs_prev_sell_ratio = buy_volume_current / sell_volume_previous;
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            buy_volume_current,                                 // Current buy volume
            sell_volume_previous,                               // Previous sell volume
            buy_vs_prev_sell_ratio  // Ratio of current buy to previous sell
        );
      }

      // Cross-imbalance detection: Compare current sell volume vs previous buy volume (potential
      // bearish signal)
      if (buy_volume_previous > 0 && sell_volume_current > buy_volume_previous * threshold) {
        double sell_vs_prev_buy_ratio = sell_volume_current / buy_volume_previous;
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            sell_volume_current,                                // Current sell volume
            buy_volume_previous,                                // Previous buy volume
            sell_vs_prev_buy_ratio  // Ratio of current sell to previous buy
        );
      }

      // Net flow imbalance: Compare the net flow (buy - sell) between consecutive bars
      double current_net_flow = buy_volume_current - sell_volume_current;
      double previous_net_flow = buy_volume_previous - sell_volume_previous;

      // Detect significant shift in net flow direction or magnitude
      if (std::abs(previous_net_flow) > 0) {
        double net_flow_change_ratio = std::abs(current_net_flow) / std::abs(previous_net_flow);

        // Only consider significant changes in net flow
        if (net_flow_change_ratio > threshold) {
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              std::abs(current_net_flow),                         // Magnitude of current net flow
              std::abs(previous_net_flow),                        // Magnitude of previous net flow
              net_flow_change_ratio  // Ratio of current to previous net flow magnitude
          );
        }
      }

      // Enhanced stacked imbalance: Look for accumulation of one-sided pressure
      // Calculate the buy/sell ratio for current and previous time buckets
      double current_ratio =
          (sell_volume_current > 0)
              ? buy_volume_current / sell_volume_current
              : (buy_volume_current > 0 ? std::numeric_limits<double>::max() : 0);
      double previous_ratio =
          (sell_volume_previous > 0)
              ? buy_volume_previous / sell_volume_previous
              : (buy_volume_previous > 0 ? std::numeric_limits<double>::max() : 0);

      // Detect significant change in the buy/sell dynamic at the same price level
      if (previous_ratio > 0 && current_ratio > 0 &&
          previous_ratio != std::numeric_limits<double>::max() &&
          current_ratio != std::numeric_limits<double>::max()) {
        double ratio_change = current_ratio / previous_ratio;

        // Bullish shift: buy/sell ratio increased significantly
        if (ratio_change > threshold) {
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              buy_volume_current,                                 // Current buy volume
              sell_volume_current,                                // Current sell volume
              ratio_change                                        // Change in buy/sell ratio
          );
        }
        // Bearish shift: buy/sell ratio decreased significantly
        else if (ratio_change < 1.0 / threshold) {
          double inverse_ratio_change = 1.0 / ratio_change;
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              sell_volume_current,                                // Current sell volume
              buy_volume_current,                                 // Current buy volume
              inverse_ratio_change  // Inverse change in buy/sell ratio
          );
        }
      }
      // Handle cases where one side is zero (extreme imbalance)
      else if (sell_volume_current == 0 && sell_volume_previous > 0 && buy_volume_current > 0) {
        // Extreme bullish: current period has only buys where previous had sells
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            buy_volume_current,                                 // Current buy volume
            sell_volume_previous,                               // Previous sell volume
            threshold                                           // High ratio indicator
        );
      } else if (buy_volume_current == 0 && buy_volume_previous > 0 && sell_volume_current > 0) {
        // Extreme bearish: current period has only sells where previous had buys
        imbalances.emplace_back(
            static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
            time_bucket,                                        // Current time bucket
            sell_volume_current,                                // Current sell volume
            buy_volume_previous,                                // Previous buy volume
            threshold                                           // High ratio indicator
        );
      }

      // Enhanced stacked imbalance: Look for multi-timeframe patterns
      // Compare current time bucket with multiple previous time buckets to detect sustained imbalances
      double buy_volume_2_bars_ago = 0.0, sell_volume_2_bars_ago = 0.0;
      bool has_2_bars_ago = false;

      if (time_bucket >= 2) {
        has_2_bars_ago = true;
        // Get data from 2 time buckets ago (using atomic loads)
        buy_volume_2_bars_ago = cluster_canvas_[price_idx][time_bucket - 2].buy_volume.load(std::memory_order_acquire);
        sell_volume_2_bars_ago = cluster_canvas_[price_idx][time_bucket - 2].sell_volume.load(std::memory_order_acquire);

        // Detect sustained bullish pressure: increasing buy volume over 3 consecutive periods
        if (buy_volume_2_bars_ago > 0 && buy_volume_previous > buy_volume_2_bars_ago &&
            buy_volume_current > buy_volume_previous &&
            buy_volume_current > buy_volume_2_bars_ago * threshold) {
          double sustained_bullish_ratio = buy_volume_current / buy_volume_2_bars_ago;
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              buy_volume_current,                                 // Current buy volume
              buy_volume_2_bars_ago,                              // Volume 2 bars ago
              sustained_bullish_ratio                            // Sustained bullish ratio
          );
        }

        // Detect sustained bearish pressure: increasing sell volume over 3 consecutive periods
        if (sell_volume_2_bars_ago > 0 && sell_volume_previous > sell_volume_2_bars_ago &&
            sell_volume_current > sell_volume_previous &&
            sell_volume_current > sell_volume_2_bars_ago * threshold) {
          double sustained_bearish_ratio = sell_volume_current / sell_volume_2_bars_ago;
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              sell_volume_current,                                // Current sell volume
              sell_volume_2_bars_ago,                             // Volume 2 bars ago
              sustained_bearish_ratio                            // Sustained bearish ratio
          );
        }
      }

      // Enhanced exhaustion detection within stacked patterns
      // Detect when strong momentum suddenly weakens
      if (has_2_bars_ago) {
        double avg_buy_prev_2 = (buy_volume_2_bars_ago + buy_volume_previous) / 2.0;
        double avg_sell_prev_2 = (sell_volume_2_bars_ago + sell_volume_previous) / 2.0;

        // Potential bullish exhaustion: strong previous buying followed by current weakness
        if (avg_buy_prev_2 > 0 && buy_volume_current < avg_buy_prev_2 / threshold &&
            sell_volume_current > avg_sell_prev_2 * threshold) {
          double exh_ratio = (avg_buy_prev_2 / buy_volume_current) * (sell_volume_current / avg_sell_prev_2);
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              buy_volume_current,                                 // Current buy volume
              sell_volume_current,                                // Current sell volume
              exh_ratio                                          // Exhaustion ratio
          );
        }

        // Potential bearish exhaustion: strong previous selling followed by current weakness
        if (avg_sell_prev_2 > 0 && sell_volume_current < avg_sell_prev_2 / threshold &&
            buy_volume_current > avg_buy_prev_2 * threshold) {
          double exh_ratio = (avg_sell_prev_2 / sell_volume_current) * (buy_volume_current / avg_buy_prev_2);
          imbalances.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              buy_volume_current,                                 // Current buy volume
              sell_volume_current,                                // Current sell volume
              exh_ratio                                          // Exhaustion ratio
          );
        }
      }
    }
  }

  return imbalances;
}

std::vector<std::tuple<int64_t, int, double, double, double, std::string>>
ClusterEngine::detect_exhaustion_moves(double threshold) const {
  std::vector<std::tuple<int64_t, int, double, double, double, std::string>> exhaustion_moves;

  // Exhaustion moves typically occur when there's extreme buying/selling pressure followed by weakness
  // This can be detected by looking for:
  // 1. High volume in one direction (buy or sell)
  // 2. Followed by lower volume in the same direction or reversal
  // 3. Price movement stalling despite high volume
  
  for (size_t price_idx = 0; price_idx < cluster_canvas_.size(); ++price_idx) {
    for (int time_bucket = 1; time_bucket < 16; ++time_bucket) {
      // Get current and previous time bucket data (using atomic loads)
      double current_buy_volume = cluster_canvas_[price_idx][time_bucket].buy_volume.load(std::memory_order_acquire);
      double current_sell_volume = cluster_canvas_[price_idx][time_bucket].sell_volume.load(std::memory_order_acquire);
      double prev_buy_volume = cluster_canvas_[price_idx][time_bucket - 1].buy_volume.load(std::memory_order_acquire);
      double prev_sell_volume = cluster_canvas_[price_idx][time_bucket - 1].sell_volume.load(std::memory_order_acquire);

      // Calculate total volumes
      double current_total_volume = current_buy_volume + current_sell_volume;
      double prev_total_volume = prev_buy_volume + prev_sell_volume;

      // Calculate deltas
      double current_delta = current_buy_volume - current_sell_volume;
      double prev_delta = prev_buy_volume - prev_sell_volume;

      // Bullish exhaustion: Strong buying pressure followed by weakness or reversal
      if (prev_delta > 0 && std::abs(prev_delta) > threshold * 100) { // Strong previous buying
        if (current_delta < 0 && std::abs(current_delta) > threshold * 50) { // Reversal to selling
          // This indicates bullish exhaustion - buyers exhausted, sellers taking control
          exhaustion_moves.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              current_buy_volume,                                 // Current buy volume
              current_sell_volume,                                // Current sell volume
              std::abs(current_delta),                            // Magnitude of reversal
              "Bullish Exhaustion"                                // Type of exhaustion
          );
        } else if (current_total_volume < prev_total_volume / 2.0 && current_delta < prev_delta / 2.0) {
          // Volume dries up and momentum weakens - also bullish exhaustion
          exhaustion_moves.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              current_buy_volume,                                 // Current buy volume
              current_sell_volume,                                // Current sell volume
              std::abs(current_delta - prev_delta),               // Change in momentum
              "Bullish Exhaustion (Weak)"                         // Type of exhaustion
          );
        }
      }

      // Bearish exhaustion: Strong selling pressure followed by weakness or reversal
      if (prev_delta < 0 && std::abs(prev_delta) > threshold * 100) { // Strong previous selling
        if (current_delta > 0 && std::abs(current_delta) > threshold * 50) { // Reversal to buying
          // This indicates bearish exhaustion - sellers exhausted, buyers stepping in
          exhaustion_moves.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              current_buy_volume,                                 // Current buy volume
              current_sell_volume,                                // Current sell volume
              std::abs(current_delta),                            // Magnitude of reversal
              "Bearish Exhaustion"                                // Type of exhaustion
          );
        } else if (current_total_volume < prev_total_volume / 2.0 && std::abs(current_delta) < std::abs(prev_delta) / 2.0) {
          // Volume dries up and momentum weakens - also bearish exhaustion
          exhaustion_moves.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              current_buy_volume,                                 // Current buy volume
              current_sell_volume,                                // Current sell volume
              std::abs(prev_delta - current_delta),               // Change in momentum
              "Bearish Exhaustion (Weak)"                         // Type of exhaustion
          );
        }
      }

      // Hidden exhaustion: Extreme volume in one direction but price doesn't move proportionally
      if (current_total_volume > 0 && std::abs(current_delta) / current_total_volume < 0.1) {
        // High volume but little directional bias - potential exhaustion
        if (current_total_volume > threshold * 200) { // Very high total volume
          std::string exhaustion_type = current_delta > 0 ? "Hidden Bullish Exhaustion" : "Hidden Bearish Exhaustion";
          exhaustion_moves.emplace_back(
              static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
              time_bucket,                                        // Current time bucket
              current_buy_volume,                                 // Current buy volume
              current_sell_volume,                                // Current sell volume
              current_total_volume,                               // Total volume
              exhaustion_type                                      // Type of exhaustion
          );
        }
      }
    }
  }

  return exhaustion_moves;
}

void ClusterEngine::processTradeWithTimeAggregation(const MarketData::Trade& trade,
                                                    BTQuant::Data::TimeAggregationType agg_type,
                                                    int n_contracts, int n_ticks) {
  int64_t abs_tick_index = static_cast<int64_t>(std::round(trade.price / tick_size_));

  // Initialize session start time if not already set
  if (session_start_us_ == 0) {
    session_start_us_ = trade.timestamp_us;
  }

  // Determine the appropriate time bucket based on aggregation type
  int time_bucket = 0;

  switch (agg_type) {
    case BTQuant::Data::TimeAggregationType::T_1MIN:
      time_bucket = static_cast<int>((trade.timestamp_us - session_start_us_) / (60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_5MIN:
      time_bucket =
          static_cast<int>((trade.timestamp_us - session_start_us_) / (5LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_15MIN:
      time_bucket =
          static_cast<int>((trade.timestamp_us - session_start_us_) / (15LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_30MIN:
      time_bucket =
          static_cast<int>((trade.timestamp_us - session_start_us_) / (30LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_1HOUR:
      time_bucket =
          static_cast<int>((trade.timestamp_us - session_start_us_) / (60LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_2HOUR:
      time_bucket = static_cast<int>((trade.timestamp_us - session_start_us_) /
                                     (2LL * 60LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::T_4HOUR:
      time_bucket = static_cast<int>((trade.timestamp_us - session_start_us_) /
                                     (4LL * 60LL * 60LL * 1000000LL));
      break;
    case BTQuant::Data::TimeAggregationType::VOLUME_BASED: {
      // For volume-based aggregation, use static thread-local variables to accumulate volume at the
      // price level
      static thread_local std::map<int64_t, double>
          volume_accumulators;  // Accumulated volume by price level

      // For volume-based aggregation, accumulate volume at the price level
      volume_accumulators[abs_tick_index] += trade.quantity;

      // Determine which time bucket this belongs to based on accumulated volume
      time_bucket = static_cast<int>(volume_accumulators[abs_tick_index] / n_contracts);

      // If we've reached the threshold, reset the accumulator for this price level
      if (volume_accumulators[abs_tick_index] >= n_contracts) {
        volume_accumulators[abs_tick_index] =
            fmod(volume_accumulators[abs_tick_index], n_contracts);
      }
      break;
    }
    case BTQuant::Data::TimeAggregationType::TICK_BASED: {
      // For tick-based aggregation, use static thread-local variables to count ticks at the price
      // level
      static thread_local std::map<int64_t, int>
          tick_accumulators;  // Count of ticks by price level

      // For tick-based aggregation, count ticks at the price level
      tick_accumulators[abs_tick_index]++;

      // Determine which time bucket this belongs to based on tick count
      time_bucket = static_cast<int>(tick_accumulators[abs_tick_index] / n_ticks);

      // If we've reached the threshold, reset the counter for this price level
      if (tick_accumulators[abs_tick_index] >= n_ticks) {
        tick_accumulators[abs_tick_index] = tick_accumulators[abs_tick_index] % n_ticks;
      }
      break;
    }
    default:
      // Default to 1-minute aggregation
      time_bucket = static_cast<int>((trade.timestamp_us - session_start_us_) / (60LL * 1000000LL));
      break;
  }

  // Process the trade with the determined time bucket
  processTrade(trade, time_bucket);
}

// Calculate standard deviation for a specific price level and time bucket
double ClusterEngine::calculateStandardDeviation(int64_t price_level, int time_bucket) const {
  // Check if the price level and time bucket are valid
  if (price_level < min_tick_index_ ||
      static_cast<size_t>(price_level - min_tick_index_) >= cluster_canvas_.size() ||
      time_bucket < 0 || time_bucket >= 16) {
    return 0.0;  // Return 0 if invalid indices
  }

  int64_t relative_index = price_level - min_tick_index_;
  const auto& cell = cluster_canvas_[relative_index][time_bucket];

  // Use atomic loads to get the values
  double sum_of_prices = cell.sum_of_prices.load(std::memory_order_acquire);
  double sum_of_squared_prices = cell.sum_of_squared_prices.load(std::memory_order_acquire);
  int n = cell.price_count.load(std::memory_order_acquire);

  if (n <= 1) {
    return 0.0;  // Standard deviation is undefined for 0 or 1 data points
  }

  // Calculate mean
  double mean = sum_of_prices / n;

  // Calculate variance using the formula: variance = E[X^2] - (E[X])^2
  double variance = (sum_of_squared_prices / n) - (mean * mean);

  // Ensure variance is not negative due to floating-point precision issues
  if (variance < 0.0) {
    variance = 0.0;
  }

  // Standard deviation is the square root of variance
  return std::sqrt(variance);
}

// Calculate median price for a specific price level and time bucket
double ClusterEngine::calculateMedianPrice(int64_t price_level, int time_bucket) const {
  // Check if the price level and time bucket are valid
  if (price_level < min_tick_index_ ||
      static_cast<size_t>(price_level - min_tick_index_) >= cluster_canvas_.size() ||
      time_bucket < 0 || time_bucket >= 16) {
    return 0.0;  // Return 0 if invalid indices
  }

  int64_t relative_index = price_level - min_tick_index_;
  const auto& cell = cluster_canvas_[relative_index][time_bucket];

  // Use atomic loads to get the values
  double sum_of_prices = cell.sum_of_prices.load(std::memory_order_acquire);
  int n = cell.price_count.load(std::memory_order_acquire);

  if (n == 0) {
    return 0.0;  // Return 0 if no prices recorded
  }

  // Since we can't efficiently store and sort individual prices atomically,
  // we'll return the mean price as an approximation of the median
  // In a truly lock-free system, calculating the exact median would require
  // a more complex data structure or algorithm
  return sum_of_prices / n;
}

}  // namespace Analytics
