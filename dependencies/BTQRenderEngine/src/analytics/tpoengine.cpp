#include "analytics/tpoengine.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ctime>

// Implementation of TPONode constructor
TPONode::TPONode(std::chrono::system_clock::time_point start,
                 std::chrono::system_clock::time_point end,
                 double price)
    : time_start(start), time_end(end), price_level(price),
      count(0), total_volume(0.0) {}

// Implementation of TPOEngine methods
TPOEngine::TPOEngine(double bucket_size) : price_bucket_size(bucket_size) {}

// Calculate the time bucket start time for a given timestamp
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
    return std::floor(price / price_bucket_size) * price_bucket_size;
}

// Process a single price tick and aggregate into TPO buckets
void TPOEngine::process_tick(const PriceTick& tick) {
    auto time_bucket_start = get_time_bucket_start(tick.timestamp);
    auto time_bucket_end = time_bucket_start + std::chrono::minutes(30);
    auto price_bucket = get_price_bucket(tick.price);

    // Create or update the TPONode for this time-price combination
    if (tpo_data[time_bucket_start].find(price_bucket) == tpo_data[time_bucket_start].end()) {
        tpo_data[time_bucket_start][price_bucket] = TPONode(time_bucket_start, time_bucket_end, price_bucket);
    }

    tpo_data[time_bucket_start][price_bucket].count++;
    tpo_data[time_bucket_start][price_bucket].total_volume += tick.volume;

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
         it != tpo_data.lower_bound(end_time); ++it) {
        result[it->first] = it->second;
    }

    return result;
}

// Get all TPO data
const std::map<std::chrono::system_clock::time_point,
               std::map<double, TPONode>>& TPOEngine::get_all_tpo_data() const {
    return tpo_data;
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
        std::cout << "Time Bucket Start: " << std::put_time(std::localtime(&time_t), "%F %T") << "\n";

        for (const auto& [price_level, node] : price_buckets) {
            std::cout << "  Price Level: " << price_level
                     << ", Hits: " << node.count
                     << ", Volume: " << node.total_volume << "\n";
        }
        std::cout << "\n";
    }
}

// Print TPO profile for debugging purposes
void TPOEngine::print_tpo_profile() const {
    tpo_profile.print_profile();
}