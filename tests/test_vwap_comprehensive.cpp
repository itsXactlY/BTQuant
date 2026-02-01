#include "../dependencies/BTQ_Render_Engine/include/analytics/technical_analysis.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace BTQuant;

// Manual VWAP calculation for verification
double manual_vwap_calculation(const std::vector<TechnicalIndicators::OHLCV>& data, size_t start, size_t end) {
    double cumulative_price_volume = 0.0;
    double cumulative_volume = 0.0;
    
    for (size_t i = start; i <= end; ++i) {
        double typical_price = (data[i].high + data[i].low + data[i].close) / 3.0;
        double price_times_volume = typical_price * data[i].volume;
        
        cumulative_price_volume += price_times_volume;
        cumulative_volume += data[i].volume;
    }
    
    if (cumulative_volume > 0) {
        return cumulative_price_volume / cumulative_volume;
    } else {
        return (data[end].high + data[end].low + data[end].close) / 3.0;
    }
}

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

    // Verify each VWAP value manually
    for (size_t i = 0; i < vwap_result.values.size(); ++i) {
        double expected_vwap = manual_vwap_calculation(data, 0, i);
        double calculated_vwap = vwap_result.values[i];
        
        std::cout << "Index " << i << ": Calculated=" << calculated_vwap << ", Expected=" << expected_vwap 
                  << ", Diff=" << std::abs(calculated_vwap - expected_vwap) << std::endl;
                  
        // Allow for small floating point differences
        assert(std::abs(calculated_vwap - expected_vwap) < 0.001);
    }

    // Test VWAP calculation from index 1
    auto vwap_result_from_1 = TechnicalIndicators::calculate_vwap(data, 1);
    
    std::cout << "\nVWAP from index 1:" << std::endl;
    std::cout << "Number of VWAP values: " << vwap_result_from_1.values.size() << std::endl;
    
    // Verify each VWAP value from index 1 manually
    for (size_t i = 1; i < data.size(); ++i) {
        size_t result_idx = i - 1; // Result index starts from 0
        double expected_vwap = manual_vwap_calculation(data, 1, i);
        double calculated_vwap = vwap_result_from_1.values[result_idx];
        
        std::cout << "Index " << i << " (result " << result_idx << "): Calculated=" << calculated_vwap 
                  << ", Expected=" << expected_vwap << ", Diff=" << std::abs(calculated_vwap - expected_vwap) << std::endl;
                  
        // Allow for small floating point differences
        assert(std::abs(calculated_vwap - expected_vwap) < 0.001);
    }

    // Test edge cases
    std::vector<TechnicalIndicators::OHLCV> empty_data = {};
    auto empty_result = TechnicalIndicators::calculate_vwap(empty_data, 0);
    assert(empty_result.values.empty());
    
    // Test with zero volume
    std::vector<TechnicalIndicators::OHLCV> zero_vol_data = {
        {100.0, 105.0, 95.0, 102.0, 0.0, 1000000}
    };
    auto zero_vol_result = TechnicalIndicators::calculate_vwap(zero_vol_data, 0);
    assert(!zero_vol_result.values.empty());
    // With zero volume, it should use typical price: (105+95+102)/3 = 100.667
    assert(std::abs(zero_vol_result.values[0] - 100.667) < 0.01);
    
    std::cout << "\nAll comprehensive tests passed!" << std::endl;
    
    return 0;
}