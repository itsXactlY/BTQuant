#ifndef BTQRENDERENGINE_TPOENGINE_H
#define BTQRENDERENGINE_TPOENGINE_H

#include <vector>
#include <map>
#include <chrono>
#include <iostream>

// Structure to represent a single price tick
struct PriceTick {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double volume;
};

// Structure to represent a TPO (Time-Price Opportunity) bucket
struct TPONode {
    std::chrono::system_clock::time_point time_start;
    std::chrono::system_clock::time_point time_end;
    double price_level;
    int count; // Number of times this price level was hit in the time bucket
    double total_volume;

    // Default constructor
    TPONode() : time_start(), time_end(), price_level(0.0), count(0), total_volume(0.0) {}

    TPONode(std::chrono::system_clock::time_point start,
            std::chrono::system_clock::time_point end,
            double price);
};

class TPOEngine {
private:
    // Price bucket size - configurable based on instrument
    double price_bucket_size;

    // Map to store aggregated TPO data: time_bucket -> price_bucket -> TPONode
    std::map<std::chrono::system_clock::time_point,
             std::map<double, TPONode>> tpo_data;

public:
    explicit TPOEngine(double bucket_size = 0.25);

    // Calculate the time bucket start time for a given timestamp
    std::chrono::system_clock::time_point get_time_bucket_start(
        const std::chrono::system_clock::time_point& timestamp) const;

    // Calculate the price bucket for a given price
    double get_price_bucket(double price) const;

    // Process a single price tick and aggregate into TPO buckets
    void process_tick(const PriceTick& tick);

    // Process a vector of ticks
    void process_ticks(const std::vector<PriceTick>& ticks);

    // Get TPO data for a specific time range
    std::map<std::chrono::system_clock::time_point,
             std::map<double, TPONode>> get_tpo_data_for_range(
                 const std::chrono::system_clock::time_point& start_time,
                 const std::chrono::system_clock::time_point& end_time) const;

    // Get all TPO data
    const std::map<std::chrono::system_clock::time_point,
                   std::map<double, TPONode>>& get_all_tpo_data() const;

    // Clear all stored data
    void clear();

    // Print TPO data for debugging purposes
    void print_tpo_data() const;
};

#endif // BTQRENDERENGINE_TPOENGINE_H