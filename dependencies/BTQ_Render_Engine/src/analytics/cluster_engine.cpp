#include "../../include/analytics/cluster_engine.hpp"
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
        min_tick_index_ = abs_tick_index - 100; // start with some padding
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

    // Thread-safely update the volume counters using mutex
    {
        std::lock_guard<std::mutex> lock(cell.volume_mutex);
        cell.total_volume += trade.quantity;
        cell.sum_of_volumes += trade.quantity;

        if (trade.is_buyer_maker) {
            cell.sell_volume += trade.quantity;
        } else {
            cell.buy_volume += trade.quantity;
        }
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
        if (cell.max_single_trade_volume.compare_exchange_weak(current_max, new_value, std::memory_order_release, std::memory_order_acquire)) {
            break;
        }
    }
}

std::vector<std::tuple<int64_t, int, double, double, double>> ClusterEngine::detect_diagonal_imbalances(double threshold) const {
    std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

    // Iterate through price levels (rows) and time buckets (columns)
    // Compare buy_volume at price P with sell_volume at price P-1
    for (size_t price_idx = 1; price_idx < cluster_canvas_.size(); ++price_idx) {  // Start from 1 to compare with P-1
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
                    // Store: absolute_price_index, time_bucket, buy_volume_at_P, sell_volume_at_P_minus_1, ratio
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,
                        buy_volume_at_p,
                        sell_volume_at_p_minus_1,
                        ratio
                    );
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
                    // Store: absolute_price_index, time_bucket, sell_volume_at_P, buy_volume_at_P_minus_1, ratio
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,
                        sell_volume_at_p,
                        buy_volume_at_p_minus_1,
                        reverse_ratio
                    );
                }
            }
        }
    }

    return imbalances;
}

std::vector<std::tuple<int64_t, int, double, double, double>> ClusterEngine::detect_stacked_imbalances(double threshold) const {
    std::vector<std::tuple<int64_t, int, double, double, double>> imbalances;

    // Vertical analysis comparing buy/sell at same price across consecutive bars
    // Iterate through price levels (rows) and time buckets (columns)
    for (size_t price_idx = 0; price_idx < cluster_canvas_.size(); ++price_idx) {
        // Compare consecutive time buckets for the same price level
        for (int time_bucket = 1; time_bucket < 16; ++time_bucket) {  // Start from 1 to compare with previous time bucket
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

            // Stacked imbalance detection: Compare buy/sell dynamics at same price level across consecutive bars
            // This looks for situations where the relationship between buyers and sellers changes significantly

            // Calculate buy/sell ratios for both current and previous time buckets
            double current_buy_sell_ratio = 0.0;
            if (sell_volume_current > 0) {
                current_buy_sell_ratio = buy_volume_current / sell_volume_current;
            }

            double previous_buy_sell_ratio = 0.0;
            if (sell_volume_previous > 0) {
                previous_buy_sell_ratio = buy_volume_previous / sell_volume_previous;
            }

            // Detect significant shifts in the buy/sell balance at the same price level
            if (previous_buy_sell_ratio > 0 && current_buy_sell_ratio > 0) {
                // Calculate how much the buy/sell ratio has changed from previous bar
                double ratio_change = current_buy_sell_ratio / previous_buy_sell_ratio;

                // Bullish stacked imbalance: buy pressure significantly increased compared to sell pressure
                if (ratio_change > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        buy_volume_current,                                // Current buy volume
                        sell_volume_current,                               // Current sell volume
                        ratio_change                                       // Change in buy/sell ratio
                    );
                }

                // Bearish stacked imbalance: sell pressure significantly increased compared to buy pressure
                double inverse_ratio_change = previous_buy_sell_ratio / current_buy_sell_ratio;
                if (inverse_ratio_change > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        sell_volume_current,                               // Current sell volume
                        buy_volume_current,                                // Current buy volume
                        inverse_ratio_change                               // Change in sell/buy ratio
                    );
                }
            }

            // Additional stacked imbalance detection: Look for accumulation of one-sided pressure
            // Compare current buy volume vs previous sell volume (bullish sign)
            if (sell_volume_previous > 0) {
                double buy_vs_prev_sell = buy_volume_current / sell_volume_previous;
                if (buy_vs_prev_sell > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        buy_volume_current,                                // Current buy volume
                        sell_volume_previous,                              // Previous sell volume
                        buy_vs_prev_sell                                   // Ratio of current buy to previous sell
                    );
                }
            }

            // Compare current sell volume vs previous buy volume (bearish sign)
            if (buy_volume_previous > 0) {
                double sell_vs_prev_buy = sell_volume_current / buy_volume_previous;
                if (sell_vs_prev_buy > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        sell_volume_current,                               // Current sell volume
                        buy_volume_previous,                               // Previous buy volume
                        sell_vs_prev_buy                                   // Ratio of current sell to previous buy
                    );
                }
            }

            // Detect stacked buying pressure: when current buy volume significantly exceeds both
            // previous buy volume and current sell volume
            if (buy_volume_previous > 0 && sell_volume_current > 0) {
                double stacked_buy_pressure = (buy_volume_current / buy_volume_previous) *
                                              (buy_volume_current / sell_volume_current);
                if (stacked_buy_pressure > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        buy_volume_current,                                // Current buy volume
                        buy_volume_previous,                               // Previous buy volume
                        stacked_buy_pressure                              // Combined buy pressure indicator
                    );
                }
            }

            // Detect stacked selling pressure: when current sell volume significantly exceeds both
            // previous sell volume and current buy volume
            if (sell_volume_previous > 0 && buy_volume_current > 0) {
                double stacked_sell_pressure = (sell_volume_current / sell_volume_previous) *
                                               (sell_volume_current / buy_volume_current);
                if (stacked_sell_pressure > threshold) {
                    imbalances.emplace_back(
                        static_cast<int64_t>(price_idx) + min_tick_index_,  // Absolute tick index
                        time_bucket,                                       // Current time bucket
                        sell_volume_current,                               // Current sell volume
                        sell_volume_previous,                              // Previous sell volume
                        stacked_sell_pressure                             // Combined sell pressure indicator
                    );
                }
            }
        }
    }

    return imbalances;
}

} // namespace Analytics