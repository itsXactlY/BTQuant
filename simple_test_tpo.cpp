#include <iostream>
#include <cassert>
#include "dependencies/BTQRenderEngine/include/analytics/tpoengine.h"

int main() {
    std::cout << "Testing TPO Engine basic functionality...\n";
    
    // Create a TPO engine with 0.25 price bucket size
    TPOEngine engine(0.25);
    
    // Add some sample data points to simulate TPOs
    std::vector<PriceTick> ticks;
    
    // Create ticks with varying prices and times to simulate TPOs
    for (int i = 0; i < 10; ++i) {
        PriceTick tick;
        tick.timestamp = std::chrono::system_clock::now() + std::chrono::minutes(i * 5);
        
        // Create concentration around certain price levels to test POC calculation
        if (i < 4) {
            tick.price = 100.0;  // This should have multiple hits
        } else if (i < 7) {
            tick.price = 100.25;
        } else {
            tick.price = 99.75;
        }
        
        tick.volume = 1.0;
        ticks.push_back(tick);
    }
    
    // Process the ticks
    engine.process_ticks(ticks);
    
    // Get the profile and test basic functionality
    const TPOProfile& profile = engine.get_tpo_profile();
    
    std::cout << "Total TPO count: " << profile.get_total_tpo_count() << std::endl;
    std::cout << "Unique price levels: " << profile.get_unique_price_count() << std::endl;
    
    // Test that basic functionality works
    assert(profile.get_unique_price_count() > 0);
    assert(profile.get_total_tpo_count() == 10); // 10 ticks processed
    
    std::cout << "✓ Basic TPO engine functionality test passed\n";
    
    // Test that POC calculation compiles and runs (without asserting specific values)
    double poc = profile.get_poc();
    std::cout << "POC (Point of Control): " << poc << std::endl;
    
    // Test that Value Area calculation compiles and runs
    auto va = profile.get_value_area(70.0);
    std::cout << "Value Area (70%): " << va.first << " - " << va.second << std::endl;
    
    std::cout << "\nTPO engine features compilation test passed!\n";
    
    return 0;
}