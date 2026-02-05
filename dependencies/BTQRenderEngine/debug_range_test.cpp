#include "../include/analytics/tpoengine.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "Debugging TPOEngine range query..." << std::endl;

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Create ticks spanning multiple time buckets
    // Using time points from epoch to be more precise
    std::vector<PriceTick> ticks = {
        {std::chrono::system_clock::time_point(std::chrono::seconds(5*60)), 100.1, 100.0},   // Time: 0-30 min -> bucket at 0s
        {std::chrono::system_clock::time_point(std::chrono::seconds(35*60)), 100.1, 150.0},  // Time: 30-60 min -> bucket at 1800s
        {std::chrono::system_clock::time_point(std::chrono::seconds(65*60)), 100.1, 200.0},  // Time: 60-90 min -> bucket at 3600s
    };

    std::cout << "Processing ticks..." << std::endl;
    for (const auto& tick : ticks) {
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(tick.timestamp.time_since_epoch()).count();
        auto bucket_start = engine.get_time_bucket_start(tick.timestamp);
        auto bs_count = std::chrono::duration_cast<std::chrono::seconds>(bucket_start.time_since_epoch()).count();
        std::cout << "  Tick at " << ts << "s -> bucket at " << bs_count << "s" << std::endl;
    }

    engine.process_ticks(ticks);

    // Print all data to see what we have
    std::cout << "All TPO data:" << std::endl;
    const auto& all_data = engine.get_all_tpo_data();
    for (const auto& [time_bucket, price_buckets] : all_data) {
        auto time_count = std::chrono::duration_cast<std::chrono::seconds>(time_bucket.time_since_epoch()).count();
        std::cout << "  Time bucket start: " << time_count << " seconds since epoch" << std::endl;
    }
    std::cout << "Total time buckets: " << all_data.size() << std::endl;

    // Query for range covering first two buckets: [0, 3600) seconds
    auto start_time = std::chrono::system_clock::time_point(std::chrono::seconds(0));
    auto end_time = std::chrono::system_clock::time_point(std::chrono::seconds(3600)); // 60 minutes

    std::cout << "Querying range [" << std::chrono::duration_cast<std::chrono::seconds>(start_time.time_since_epoch()).count()
              << ", " << std::chrono::duration_cast<std::chrono::seconds>(end_time.time_since_epoch()).count() << ")" << std::endl;

    auto range_data = engine.get_tpo_data_for_range(start_time, end_time);

    std::cout << "Range data size: " << range_data.size() << std::endl;
    std::cout << "Range data contents:" << std::endl;
    for (const auto& [time_bucket, price_buckets] : range_data) {
        auto time_count = std::chrono::duration_cast<std::chrono::seconds>(time_bucket.time_since_epoch()).count();
        std::cout << "  Time bucket start: " << time_count << " seconds since epoch" << std::endl;
    }

    return 0;
}