#include "../../../dependencies/BTQ_Render_Engine/include/analytics/cluster_engine.hpp"
#include <mutex>

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

    // Atomically update the counters
    cell.total_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
    cell.sum_of_volumes.fetch_add(trade.quantity, std::memory_order_relaxed);

    if (trade.is_buyer_maker) {
        cell.sell_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
        cell.sell_trade_count.fetch_add(1, std::memory_order_relaxed);
    } else {
        cell.buy_volume.fetch_add(trade.quantity, std::memory_order_relaxed);
        cell.buy_trade_count.fetch_add(1, std::memory_order_relaxed);
    }

    cell.trade_count.fetch_add(1, std::memory_order_relaxed);

    // Update max single trade volume atomically
    double current_max = cell.max_single_trade_volume.load(std::memory_order_relaxed);
    double new_value = trade.quantity;
    while (new_value > current_max) {
        if (cell.max_single_trade_volume.compare_exchange_weak(current_max, new_value, std::memory_order_relaxed)) {
            break;
        }
    }
}

} // namespace Analytics