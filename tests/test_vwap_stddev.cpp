#include "../dependencies/BTQ_Render_Engine/include/analytics/technical_analysis.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace BTQuant;

int main() {
    // Create test data
    std::vector<TechnicalIndicators::OHLCV> data = {
        {100.0, 105.0, 95.0, 102.0, 1000.0, 1000000},  // open, high, low, close, volume, timestamp
        {102.0, 108.0, 100.0, 106.0, 1500.0, 1000001},
        {106.0, 110.0, 104.0, 108.0, 2000.0, 1000002},
        {108.0, 112.0, 106.0, 110.0, 1200.0, 1000003}
    };

    // Test VWAP standard deviation calculation from start (index 0)
    auto stddev_result = TechnicalIndicators::calculate_vwap_standard_deviation(data, 0);

    std::cout << "Testing VWAP Standard Deviation calculation..." << std::endl;
    std::cout << "Number of VWAP StdDev values: " << stddev_result.values.size() << std::endl;

    // Print out the calculated values
    for (size_t i = 0; i < stddev_result.values.size(); ++i) {
        std::cout << "Index " << i << ": VWAP StdDev=" << stddev_result.values[i] << std::endl;
    }

    // Verify that we got the expected number of results
    assert(stddev_result.values.size() == 4); // Should have 4 values for 4 data points
    
    // Verify that all values are non-negative (standard deviation is always non-negative)
    for (size_t i = 0; i < stddev_result.values.size(); ++i) {
        assert(stddev_result.values[i] >= 0.0);
    }

    // The first value should be 0 since there's only one data point to calculate deviation from
    assert(std::abs(stddev_result.values[0] - 0.0) < 0.001);

    // Test VWAP standard deviation calculation from index 1
    auto stddev_result_from_1 = TechnicalIndicators::calculate_vwap_standard_deviation(data, 1);

    std::cout << "\nVWAP StdDev from index 1:" << std::endl;
    std::cout << "Number of VWAP StdDev values: " << stddev_result_from_1.values.size() << std::endl;

    // Verify that we got the expected number of results
    assert(stddev_result_from_1.values.size() == 3); // Should have 3 values for data points from index 1 to end
    
    // Verify that all values are non-negative
    for (size_t i = 0; i < stddev_result_from_1.values.size(); ++i) {
        std::cout << "Index " << i << " (from 1): VWAP StdDev=" << stddev_result_from_1.values[i] << std::endl;
        assert(stddev_result_from_1.values[i] >= 0.0);
    }

    // Test edge case with empty data
    std::vector<TechnicalIndicators::OHLCV> empty_data = {};
    auto empty_result = TechnicalIndicators::calculate_vwap_standard_deviation(empty_data, 0);
    assert(empty_result.values.empty());

    std::cout << "\nAll VWAP Standard Deviation tests passed!" << std::endl;

    return 0;
}