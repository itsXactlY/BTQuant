#include <iostream>
#include <cassert>
#include "dependencies/BTQRenderEngine/include/analytics/tpoengine.h"

int main() {
    std::cout << "Testing Single Print Detection Feature...\n";
    
    // Create a TPO engine with 0.25 price bucket size
    TPOEngine engine(0.25);
    
    // Add some sample data points to simulate TPOs
    std::vector<PriceTick> ticks;
    
    // Create ticks with varying prices and times to simulate TPOs
    // This will create some price levels with single prints and others with multiple prints
    auto base_time = std::chrono::system_clock::now();
    
    // Price level 100.0 - single print (only hit in one time bracket)
    ticks.push_back({base_time, 100.0, 1.0});
    
    // Price level 101.0 - multiple prints (hit in multiple time brackets)
    ticks.push_back({base_time, 101.0, 1.0});
    ticks.push_back({base_time + std::chrono::minutes(35), 101.0, 1.0});  // Different 30-min bucket
    
    // Price level 102.0 - single print (only hit in one time bracket)
    ticks.push_back({base_time + std::chrono::minutes(65), 102.0, 1.0});
    
    // Price level 103.0 - multiple prints (hit in multiple time brackets)
    ticks.push_back({base_time, 103.0, 1.0});
    ticks.push_back({base_time + std::chrono::minutes(35), 103.0, 1.0});
    ticks.push_back({base_time + std::chrono::minutes(70), 103.0, 1.0});
    
    // Process all ticks
    engine.process_ticks(ticks);
    
    // Get the TPO profile
    const TPOProfile& profile = engine.get_tpo_profile();
    
    // Test single print detection
    std::vector<double> single_prints = profile.get_single_print_levels();
    
    std::cout << "Single print levels detected: ";
    for (double price : single_prints) {
        std::cout << price << " ";
    }
    std::cout << std::endl;
    
    // Verify that we have the expected single print levels
    // We expect 100.0 and 102.0 to be single prints
    assert(single_prints.size() >= 2); // At least the two we know should be single prints
    
    bool found_100 = false, found_102 = false;
    for (double price : single_prints) {
        if (std::abs(price - 100.0) < 0.25) found_100 = true;
        if (std::abs(price - 102.0) < 0.25) found_102 = true;
    }
    
    assert(found_100);  // 100.0 should be a single print
    assert(found_102);  // 102.0 should be a single print
    
    std::cout << "✓ Single print detection test passed!" << std::endl;
    
    // Also verify that 101.0 and 103.0 are NOT in single prints (they have multiple hits)
    bool found_101 = false, found_103 = false;
    for (double price : single_prints) {
        if (std::abs(price - 101.0) < 0.25) found_101 = true;
        if (std::abs(price - 103.0) < 0.25) found_103 = true;
    }
    
    assert(!found_101);  // 101.0 should NOT be a single print
    assert(!found_103);  // 103.0 should NOT be a single print
    
    std::cout << "✓ Confirmed that multi-print levels are correctly excluded from single prints!" << std::endl;
    
    // Test the letter sequences
    std::cout << "Letter sequences for each price level:" << std::endl;
    for (const auto& [price, letters] : profile.price_to_letters) {
        std::cout << "Price " << price << ": " << letters << " (length: " << letters.length() << ")" << std::endl;
        
        // Verify that single prints have length 1 and multi-prints have length > 1
        bool is_expected_single = (std::abs(price - 100.0) < 0.25) || (std::abs(price - 102.0) < 0.25);
        bool is_actual_single = letters.length() == 1;
        
        if (is_expected_single) {
            assert(is_actual_single);  // Expected single prints should indeed have length 1
        } else {
            assert(!is_actual_single);  // Expected multi prints should have length > 1
        }
    }
    
    std::cout << "\n✓ All single print detection tests passed successfully!" << std::endl;
    
    return 0;
}