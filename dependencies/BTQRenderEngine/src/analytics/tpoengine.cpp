#include "analytics/tpoengine.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <set>

// Implementation of TPONode constructor
TPONode::TPONode(std::chrono::system_clock::time_point start,
                 std::chrono::system_clock::time_point end,
                 double price)
    : time_start(start), time_end(end), price_level(price),
      count(0), total_volume(0.0), high_price(price), low_price(price) {}

// Implementation of TPOEngine methods
TPOEngine::TPOEngine(double bucket_size) : price_bucket_size(bucket_size) {}

// Calculate the time bucket start time for a given timestamp (30-minute intervals)
std::chrono::system_clock::time_point TPOEngine::get_time_bucket_start(
    const std::chrono::system_clock::time_point& timestamp) const {
    // Convert to seconds since epoch
    auto timestamp_seconds = std::chrono::time_point_cast<std::chrono::seconds>(timestamp).time_since_epoch().count();

    // Calculate the start of the 30-minute bucket (1800 seconds = 30 minutes)
    auto bucket_start_seconds = (timestamp_seconds / 1800) * 1800;

    return std::chrono::system_clock::time_point{std::chrono::seconds(bucket_start_seconds)};
}

// Calculate the price bucket for a given price
double TPOEngine::get_price_bucket(double price) const {
    if (price_bucket_size <= 0) {
        return price; // No bucketing if bucket size is invalid
    }
    return std::floor(price / price_bucket_size) * price_bucket_size;
}

// Process a single price tick and aggregate into TPO buckets
void TPOEngine::process_tick(const PriceTick& tick) {
    auto time_bucket_start = get_time_bucket_start(tick.timestamp);
    auto time_bucket_end = time_bucket_start + std::chrono::minutes(30);
    auto price_bucket = get_price_bucket(tick.price);

    // Create or update the TPONode for this time-price combination
    auto& node = tpo_data[time_bucket_start][price_bucket];

    // Initialize node if it's new
    if (node.count == 0) {
        node.time_start = time_bucket_start;
        node.time_end = time_bucket_end;
        node.price_level = price_bucket;
        node.high_price = tick.price;
        node.low_price = tick.price;
    } else {
        // Update high and low prices for this bucket
        node.high_price = std::max(node.high_price, tick.price);
        node.low_price = std::min(node.low_price, tick.price);
    }

    // Update statistics
    node.count++;
    node.total_volume += tick.volume;

    // Add the price level to the TPO profile with the corresponding time bracket
    tpo_profile.add_price_to_time_bracket(price_bucket, time_bucket_start);
}

// Process a vector of ticks
void TPOEngine::process_ticks(const std::vector<PriceTick>& ticks) {
    for (const auto& tick : ticks) {
        process_tick(tick);
    }
}

// Get TPO data for a specific time range
std::map<std::chrono::system_clock::time_point,
         std::map<double, TPONode>> TPOEngine::get_tpo_data_for_range(
             const std::chrono::system_clock::time_point& start_time,
             const std::chrono::system_clock::time_point& end_time) const {
    std::map<std::chrono::system_clock::time_point,
             std::map<double, TPONode>> result;

    for (auto it = tpo_data.lower_bound(start_time);
         it != tpo_data.upper_bound(end_time); ++it) {
        result[it->first] = it->second;
    }

    return result;
}

// Get all TPO data
const std::map<std::chrono::system_clock::time_point,
               std::map<double, TPONode>>& TPOEngine::get_all_tpo_data() const {
    return tpo_data;
}

// Get TPO data for a specific time bucket
const std::map<double, TPONode>* TPOEngine::get_tpo_data_for_time_bucket(
    const std::chrono::system_clock::time_point& time_bucket) const {
    auto it = tpo_data.find(time_bucket);
    if (it != tpo_data.end()) {
        return &(it->second);
    }
    return nullptr;
}

// Get the highest price in a specific time-price bucket
double TPOEngine::get_high_price(const std::chrono::system_clock::time_point& time_bucket,
                                 double price_bucket) const {
    auto time_it = tpo_data.find(time_bucket);
    if (time_it != tpo_data.end()) {
        auto price_it = time_it->second.find(price_bucket);
        if (price_it != time_it->second.end()) {
            return price_it->second.high_price;
        }
    }
    return 0.0;
}

// Get the lowest price in a specific time-price bucket
double TPOEngine::get_low_price(const std::chrono::system_clock::time_point& time_bucket,
                                double price_bucket) const {
    auto time_it = tpo_data.find(time_bucket);
    if (time_it != tpo_data.end()) {
        auto price_it = time_it->second.find(price_bucket);
        if (price_it != time_it->second.end()) {
            return price_it->second.low_price;
        }
    }
    return 0.0;
}

// Clear all stored data
void TPOEngine::clear() {
    tpo_data.clear();
    tpo_profile.clear();
}

// Print TPO data for debugging purposes
void TPOEngine::print_tpo_data() const {
    for (const auto& [time_bucket, price_buckets] : tpo_data) {
        auto time_t = std::chrono::system_clock::to_time_t(time_bucket);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");

        std::cout << "Time Bucket Start: " << ss.str() << "\n";

        for (const auto& [price_level, node] : price_buckets) {
            std::cout << "  Price Level: " << std::fixed << std::setprecision(2) << price_level
                     << ", Hits: " << node.count
                     << ", Volume: " << node.total_volume
                     << ", High: " << node.high_price
                     << ", Low: " << node.low_price << "\n";
        }
        std::cout << "\n";
    }
}

// Print TPO profile for debugging purposes
void TPOEngine::print_tpo_profile() const {
    tpo_profile.print_profile();
}

// Get statistics for a specific time period
TPOStatistics TPOEngine::get_statistics_for_period(
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time) const {

    TPOStatistics stats;
    stats.total_ticks_processed = 0;
    stats.total_volume = 0.0;
    stats.unique_time_buckets = 0;
    stats.unique_price_levels = 0;

    auto data_in_range = get_tpo_data_for_range(start_time, end_time);

    std::set<double> all_price_levels;

    for (const auto& [time_bucket, price_buckets] : data_in_range) {
        stats.unique_time_buckets++;

        for (const auto& [price_level, node] : price_buckets) {
            stats.total_ticks_processed += node.count;
            stats.total_volume += node.total_volume;
            all_price_levels.insert(price_level);
        }
    }

    stats.unique_price_levels = all_price_levels.size();

    return stats;
}