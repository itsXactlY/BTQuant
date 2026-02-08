#include "../../include/analytics/cluster_engine.hpp"

#include <execution>  // For std::execution::par_unseq
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

namespace Analytics {

void ClusterEngine::processTrade(const MarketData::Trade& trade, int time_bucket) {
  std::lock_guard<std::mutex> lock(engine_mutex_);
  processTradeInternal(trade, time_bucket);
}

void ClusterEngine::processTradeInternal(const MarketData::Trade& trade, int time_bucket) {
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

  // Thread-safely update the volume counters and price statistics using mutex
  {
    std::lock_guard<std::mutex> lock(cell.volume_mutex);
    cell.total_volume += trade.quantity;
    cell.sum_of_volumes += trade.quantity;

    if (trade.is_buyer_maker) {
      cell.sell_volume += trade.quantity;
    } else {
      cell.buy_volume += trade.quantity;
    }

    // Update price statistics for standard deviation and median calculations
    cell.prices.push_back(trade.price);
    cell.sum_of_prices += trade.price;
    cell.sum_of_squared_prices += trade.price * trade.price;
  }

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
  std::lock_guard<std::mutex> lock(engine_mutex_);
  std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

  // Iterate through price levels (rows) and time buckets (columns)
  // Compare buy_volume at price P with sell_volume at price P-1
  for (size_t price_idx = 1; price_idx < cluster_canvas_.size();
       ++price_idx) {  // Start from 1 to compare with P-1
    for (int time_bucket = 0; time_bucket < 16; ++time_bucket) {
      // Get buy volume at current price level P (with mutex protection)
      double buy_volume_at_p;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx][time_bucket].volume_mutex);
        buy_volume_at_p = cluster_canvas_[price_idx][time_bucket].buy_volume;
      }

      // Get sell volume at previous price level P-1 (with mutex protection)
      double sell_volume_at_p_minus_1;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx - 1][time_bucket].volume_mutex);
        sell_volume_at_p_minus_1 = cluster_canvas_[price_idx - 1][time_bucket].sell_volume;
      }

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
      double sell_volume_at_p;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx][time_bucket].volume_mutex);
        sell_volume_at_p = cluster_canvas_[price_idx][time_bucket].sell_volume;
      }

      double buy_volume_at_p_minus_1;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx - 1][time_bucket].volume_mutex);
        buy_volume_at_p_minus_1 = cluster_canvas_[price_idx - 1][time_bucket].buy_volume;
      }

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
    }
  }

  return imbalances;
}

std::vector<std::tuple<int64_t, int, double, double, double>>
ClusterEngine::detect_stacked_imbalances(double threshold) const {
  std::lock_guard<std::mutex> lock(engine_mutex_);
  std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

  // Vertical analysis comparing buy/sell at same price across consecutive bars
  // Iterate through price levels (rows) and time buckets (columns)
  for (size_t price_idx = 0; price_idx < cluster_canvas_.size(); ++price_idx) {
    // Compare consecutive time buckets for the same price level
    for (int time_bucket = 1; time_bucket < 16;
         ++time_bucket) {  // Start from 1 to compare with previous time bucket
      // Get buy and sell volumes for current time bucket (with mutex protection)
      double buy_volume_current;
      double sell_volume_current;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx][time_bucket].volume_mutex);
        buy_volume_current = cluster_canvas_[price_idx][time_bucket].buy_volume;
        sell_volume_current = cluster_canvas_[price_idx][time_bucket].sell_volume;
      }

      // Get buy and sell volumes for previous time bucket (with mutex protection)
      double buy_volume_previous;
      double sell_volume_previous;
      {
        std::lock_guard<std::mutex> lock(cluster_canvas_[price_idx][time_bucket - 1].volume_mutex);
        buy_volume_previous = cluster_canvas_[price_idx][time_bucket - 1].buy_volume;
        sell_volume_previous = cluster_canvas_[price_idx][time_bucket - 1].sell_volume;
      }

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
    }
  }

  return imbalances;
}

void ClusterEngine::processTradeWithTimeAggregation(const MarketData::Trade& trade,
                                                    BTQuant::Data::TimeAggregationType agg_type,
                                                    int n_contracts, int n_ticks) {
  std::lock_guard<std::mutex> lock(engine_mutex_);
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
  processTradeInternal(trade, time_bucket);
}

// Calculate standard deviation for a specific price level and time bucket
double ClusterEngine::calculateStandardDeviation(int64_t price_level, int time_bucket) const {
  std::lock_guard<std::mutex> lock(engine_mutex_);
  // Check if the price level and time bucket are valid
  if (price_level < min_tick_index_ ||
      static_cast<size_t>(price_level - min_tick_index_) >= cluster_canvas_.size() ||
      time_bucket < 0 || time_bucket >= 16) {
    return 0.0;  // Return 0 if invalid indices
  }

  int64_t relative_index = price_level - min_tick_index_;
  const auto& cell = cluster_canvas_[relative_index][time_bucket];

  int n = static_cast<int>(cell.prices.size());

  if (n <= 1) {
    return 0.0;  // Standard deviation is undefined for 0 or 1 data points
  }

  // Calculate mean
  double mean = cell.sum_of_prices / n;

  // Calculate variance using the formula: variance = E[X^2] - (E[X])^2
  double variance = (cell.sum_of_squared_prices / n) - (mean * mean);

  // Ensure variance is not negative due to floating-point precision issues
  if (variance < 0.0) {
    variance = 0.0;
  }

  // Standard deviation is the square root of variance
  return std::sqrt(variance);
}

// Calculate median price for a specific price level and time bucket
double ClusterEngine::calculateMedianPrice(int64_t price_level, int time_bucket) const {
  std::lock_guard<std::mutex> lock(engine_mutex_);
  // Check if the price level and time bucket are valid
  if (price_level < min_tick_index_ ||
      static_cast<size_t>(price_level - min_tick_index_) >= cluster_canvas_.size() ||
      time_bucket < 0 || time_bucket >= 16) {
    return 0.0;  // Return 0 if invalid indices
  }

  int64_t relative_index = price_level - min_tick_index_;
  const auto& cell = cluster_canvas_[relative_index][time_bucket];

  if (cell.prices.empty()) {
    return 0.0;  // Return 0 if no prices recorded
  }

  // Create a copy of the prices vector to sort without affecting the original
  std::vector<double> sorted_prices = cell.prices;
  std::sort(sorted_prices.begin(), sorted_prices.end());

  size_t n = sorted_prices.size();
  if (n % 2 == 0) {
    // Even number of elements: average of the two middle elements
    return (sorted_prices[n / 2 - 1] + sorted_prices[n / 2]) / 2.0;
  } else {
    // Odd number of elements: return the middle element
    return sorted_prices[n / 2];
  }
}

void ClusterEngine::process_trade_batch(const std::vector<MarketData::Trade>& trades) {
  // Use parallel execution to process volume calculations across the batch
  // This vector will store intermediate results to avoid race conditions during parallel processing
  std::vector<std::pair<int64_t, int>> trade_mappings;
  trade_mappings.reserve(trades.size());
  
  // Pre-calculate mappings from trades to price/time coordinates (parallelizable)
  std::for_each(std::execution::par_unseq, trades.begin(), trades.end(),
                [&trade_mappings, this](const MarketData::Trade& trade) {
                  int64_t abs_tick_index = static_cast<int64_t>(std::round(trade.price / this->tick_size_));
                  
                  // Calculate time bucket based on timestamp
                  int64_t time_bucket = 0;
                  if (this->session_start_us_ != 0) {
                    constexpr int64_t INTERVAL_US = 30LL * 60 * 1000000; // 30 min brackets
                    int64_t elapsed = trade.timestamp_us - this->session_start_us_;
                    if (elapsed >= 0) {
                      time_bucket = static_cast<int64_t>(elapsed / INTERVAL_US);
                    }
                  }
                  
                  // Store the mapping for later processing to avoid race conditions
                  trade_mappings.emplace_back(abs_tick_index, static_cast<int>(time_bucket));
                });
  
  // Now process the trades sequentially to update the cluster canvas safely
  // This avoids race conditions while still getting benefits from parallel computation
  std::lock_guard<std::mutex> lock(engine_mutex_);
  for (size_t i = 0; i < trades.size(); ++i) {
    const auto& trade = trades[i];
    const auto& mapping = trade_mappings[i];
    
    int64_t abs_tick_index = mapping.first;
    int time_bucket = mapping.second;
    
    processTradeInternal(trade, time_bucket);
  }
}

}  // namespace Analytics
