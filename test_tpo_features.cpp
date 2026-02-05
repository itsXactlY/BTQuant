#include <iostream>
#include <cassert>
#include "dependencies/BTQRenderEngine/include/analytics/tpoengine.h"

int main() {
    std::cout << "Testing TPO Engine Value Area and POC features...\n";
    
    // Create a TPO engine with 0.25 price bucket size
    TPOEngine engine(0.25);
    
    // Add some sample data points to simulate TPOs
    std::vector<PriceTick> ticks;
    
    // Create ticks with varying prices and times to simulate TPOs
    for (int i = 0; i < 100; ++i) {
        PriceTick tick;
        tick.timestamp = std::chrono::system_clock::now() + std::chrono::minutes(i * 5);
        
        // Create concentration around certain price levels to test POC calculation
        if (i < 30) {
            tick.price = 100.0;  // This should be the POC (most hits)
        } else if (i < 50) {
            tick.price = 100.25;
        } else if (i < 70) {
            tick.price = 99.75;
        } else {
            tick.price = 100.50;
        }
        
        tick.volume = 1.0;
        ticks.push_back(tick);
    }
    
    // Process the ticks
    engine.process_ticks(ticks);
    
    // Get the profile and test POC calculation
    const TPOProfile& profile = engine.get_tpo_profile();
    
    std::cout << "Total TPO count: " << profile.get_total_tpo_count() << std::endl;
    std::cout << "Unique price levels: " << profile.get_unique_price_count() << std::endl;
    
    double poc = profile.get_poc();
    std::cout << "POC (Point of Control): " << poc << std::endl;
    
    // The POC should be around 100.0 since it has the most hits (30 hits)
    assert(poc == 100.0);
    std::cout << "✓ POC calculation test passed\n";
    
    // Test Value Area calculation (70% of TPOs)
    auto va = profile.get_value_area(70.0);
    std::cout << "Value Area (70%): " << va.first << " - " << va.second << std::endl;
    
    // The value area should contain the POC and extend to include 70% of total TPOs
    // 70% of 100 TPOs = 70 TPOs, so it should include the POC and some adjacent levels
    assert(va.first <= poc && va.second >= poc);
    std::cout << "✓ Value Area calculation test passed\n";
    
    // Test with different percentages
    auto va_50 = profile.get_value_area(50.0);
    std::cout << "Value Area (50%): " << va_50.first << " - " << va_50.second << std::endl;
    
    auto va_90 = profile.get_value_area(90.0);
    std::cout << "Value Area (90%): " << va_90.first << " - " << va_90.second << std::endl;
    
    // 90% should be wider than 50%
    assert((va_90.second - va_90.first) >= (va_50.second - va_50.first));
    std::cout << "✓ Different percentage Value Area test passed\n";
    
    std::cout << "\nAll TPO engine tests passed successfully!\n";
    
    return 0;
}