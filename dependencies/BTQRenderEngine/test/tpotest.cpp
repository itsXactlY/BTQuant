#include "analytics/tpoengine.h"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Create a TPO engine with price bucket size of 0.5
    TPOEngine engine(0.5);
    
    // Create sample price ticks
    auto now = std::chrono::system_clock::now();
    std::vector<PriceTick> ticks = {
        {now, 100.1, 100.0},
        {now + std::chrono::minutes(5), 100.3, 150.0},
        {now + std::chrono::minutes(10), 100.1, 200.0},  // Same price level as first tick
        {now + std::chrono::minutes(15), 100.7, 75.0},
        {now + std::chrono::minutes(25), 100.1, 125.0},  // Same price level as first tick
        {now + std::chrono::minutes(35), 101.2, 300.0}, // Next 30-min bucket
        {now + std::chrono::minutes(40), 101.3, 175.0}, // Same 30-min bucket as above
        {now + std::chrono::minutes(65), 99.8, 225.0},  // Another 30-min bucket
    };
    
    // Process the ticks
    engine.process_ticks(ticks);
    
    // Print the aggregated TPO data
    std::cout << "TPO Engine Results:" << std::endl;
    engine.print_tpo_data();
    
    // Get all TPO data
    const auto& all_data = engine.get_all_tpo_data();
    std::cout << "Total time buckets: " << all_data.size() << std::endl;
    
    return 0;
}