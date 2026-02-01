#include "../dependencies/BTQ_Render_Engine/include/analytics/technical_analysis.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace BTQuant;

int main() {
    // Create test data
    std::vector<TechnicalIndicators::OHLCV> data = {
        {100.0, 105.0, 95.0, 102.0, 1000.0, 1000000},  // open, high, low, close, volume, timestamp
        {102.0, 108.0, 100.0, 106.0, 1500.0, 1000001},
        {106.0, 110.0, 104.0, 108.0, 2000.0, 1000002},
        {108.0, 112.0, 106.0, 110.0, 1200.0, 1000003}
    };

    // Test VWAP calculation from start (index 0)
    auto vwap_result = TechnicalIndicators::calculate_vwap(data, 0);

    std::cout << "Testing VWAP calculation..." << std::endl;
    std::cout << "Number of VWAP values: " << vwap_result.values.size() << std::endl;

    // Expected calculation:
    // Point 0: (105+95+102)/3 * 1000 / 1000 = 100.6667
    // Point 1: ((105+95+102)/3 * 1000 + (108+100+106)/3 * 1500) / (1000+1500) = ...
    
    assert(!vwap_result.values.empty());
    assert(vwap_result.values.size() == 4);  // Should have 4 values
    
    std::cout << "VWAP values:" << std::endl;
    for (size_t i = 0; i < vwap_result.values.size(); ++i) {
        std::cout << "  " << vwap_result.values[i] << " at timestamp " << vwap_result.timestamps[i] << std::endl;
    }

    // Test VWAP calculation from index 1
    auto vwap_result_from_1 = TechnicalIndicators::calculate_vwap(data, 1);
    
    std::cout << "\nVWAP from index 1:" << std::endl;
    std::cout << "Number of VWAP values: " << vwap_result_from_1.values.size() << std::endl;
    
    assert(vwap_result_from_1.values.size() == 3);  // Should have 3 values starting from index 1
    
    std::cout << "VWAP values from index 1:" << std::endl;
    for (size_t i = 0; i < vwap_result_from_1.values.size(); ++i) {
        std::cout << "  " << vwap_result_from_1.values[i] << " at timestamp " << vwap_result_from_1.timestamps[i] << std::endl;
    }

    // Test edge cases
    std::vector<TechnicalIndicators::OHLCV> empty_data = {};
    auto empty_result = TechnicalIndicators::calculate_vwap(empty_data, 0);
    assert(empty_result.values.empty());
    
    std::cout << "\nAll tests passed!" << std::endl;
    
    return 0;
}