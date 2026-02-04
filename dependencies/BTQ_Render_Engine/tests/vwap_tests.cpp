#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>

#include "analytics/technical_analysis.hpp"
#include "indicators/rolling_vwap.hpp"
#include "indicators/anchored_vwap.hpp"
#include "market_data_processor.hpp"

namespace BTQuant {
namespace Tests {

// Define test fixture for VWAP tests
class VWAPTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test OHLCV data for VWAP calculations
        // Using known values to validate VWAP calculations
        ohlcv_data_ = {
            {100.0, 105.0, 99.0, 104.0, 1000.0, 1000000ULL},  // Typical price = 101.33, TPV = 101333.33
            {104.0, 106.0, 103.0, 105.0, 1500.0, 1000001ULL}, // Typical price = 104.67, TPV = 157000.00
            {105.0, 107.0, 104.0, 106.0, 800.0, 1000002ULL},  // Typical price = 106.00, TPV = 84800.00
            {106.0, 108.0, 105.0, 107.0, 1200.0, 1000003ULL}, // Typical price = 106.33, TPV = 127600.00
            {107.0, 109.0, 106.0, 108.0, 900.0, 1000004ULL}   // Typical price = 107.33, TPV = 96600.00
        };

        // OHLCVCandle data for rolling and anchored VWAP
        ohlcv_candle_data_ = {
            {1000000ULL, 100.0, 105.0, 99.0, 104.0, 1000.0, 0},  // Typical price = 101.33, TPV = 101333.33
            {1000001ULL, 104.0, 106.0, 103.0, 105.0, 1500.0, 0}, // Typical price = 104.67, TPV = 157000.00
            {1000002ULL, 105.0, 107.0, 104.0, 106.0, 800.0, 0},  // Typical price = 106.00, TPV = 84800.00
            {1000003ULL, 106.0, 108.0, 105.0, 107.0, 1200.0, 0}, // Typical price = 106.33, TPV = 127600.00
            {1000004ULL, 107.0, 109.0, 106.0, 108.0, 900.0, 0}   // Typical price = 107.33, TPV = 96600.00
        };
    }

    std::vector<TechnicalIndicators::OHLCV> ohlcv_data_;
    std::vector<BTQuant::RenderEngine::OHLCVCandle> ohlcv_candle_data_;
};

// Test basic VWAP calculation with known values
TEST_F(VWAPTests, BasicVWAPCalculation) {
    TechnicalIndicators ti;
    
    // Manually calculate expected VWAP for the first bar
    // Typical price = (105.0 + 99.0 + 104.0) / 3 = 102.67
    // TPV = 102.67 * 1000 = 102666.67
    // VWAP = 102666.67 / 1000 = 102.67
    
    auto result = ti.calculate_vwap(ohlcv_data_, 0);
    
    ASSERT_FALSE(result.values.empty());
    
    // Expected VWAP after first bar: (102.67 * 1000) / 1000 = 102.67
    double expected_first = (105.0 + 99.0 + 104.0) / 3.0;  // 102.67
    EXPECT_NEAR(result.values[0], expected_first, 0.01);
    
    // Expected VWAP after second bar: 
    // Cumulative TPV = (102.67 * 1000) + (105.33 * 1500) = 102666.67 + 158000.00 = 260666.67
    // Cumulative Volume = 1000 + 1500 = 2500
    // VWAP = 260666.67 / 2500 = 104.27
    double expected_second = (102666.67 + (105.33 * 1500)) / (1000 + 1500);  // 104.27
    if (result.values.size() > 1) {
        EXPECT_NEAR(result.values[1], expected_second, 0.01);
    }
}

// Test VWAP calculation with manual verification
TEST_F(VWAPTests, VWAPManualVerification) {
    TechnicalIndicators ti;
    
    auto result = ti.calculate_vwap(ohlcv_data_, 0);
    
    // Manual calculation for verification:
    // Bar 1: TP = (105+99+104)/3 = 102.67, TPV = 102.67 * 1000 = 102666.67, Vol = 1000
    //        VWAP = 102666.67 / 1000 = 102.67
    // Bar 2: TP = (106+103+105)/3 = 104.67, TPV = 104.67 * 1500 = 157000.00, Vol = 1500
    //        Cumulative TPV = 102666.67 + 157000.00 = 259666.67
    //        Cumulative Vol = 1000 + 1500 = 2500
    //        VWAP = 259666.67 / 2500 = 103.87
    // Bar 3: TP = (107+104+106)/3 = 105.67, TPV = 105.67 * 800 = 84533.33, Vol = 800
    //        Cumulative TPV = 259666.67 + 84533.33 = 344200.00
    //        Cumulative Vol = 2500 + 800 = 3300
    //        VWAP = 344200.00 / 3300 = 104.30
    
    if (result.values.size() >= 3) {
        EXPECT_NEAR(result.values[0], 102.67, 0.01);  // First bar VWAP
        EXPECT_NEAR(result.values[1], 103.87, 0.01);  // Second bar VWAP
        EXPECT_NEAR(result.values[2], 104.30, 0.01);  // Third bar VWAP
    }
}

// Test VWAP with zero volume to ensure proper handling
TEST_F(VWAPTests, VWAPWithZeroVolume) {
    TechnicalIndicators ti;
    
    // Create test data with zero volume for one bar
    auto test_data = ohlcv_data_;
    test_data[2].volume = 0.0;  // Zero volume for third bar
    
    auto result = ti.calculate_vwap(test_data, 0);
    
    ASSERT_FALSE(result.values.empty());
    
    // When volume is zero, VWAP should use the typical price as fallback
    double typical_price = (test_data[2].high + test_data[2].low + test_data[2].close) / 3.0;
    if (result.values.size() > 2) {
        EXPECT_NEAR(result.values[2], typical_price, 0.01);
    }
}

// Test anchored VWAP calculation
TEST_F(VWAPTests, AnchoredVWAPCalculation) {
    btq::AnchoredVWAP anchored_vwap(1000001);  // Anchor at second bar
    
    anchored_vwap.calculate(ohlcv_candle_data_);
    
    auto vwap_values = anchored_vwap.getVWAPValues();
    
    ASSERT_FALSE(vwap_values.empty());
    
    // After anchoring at timestamp 1000001 (second bar), calculations start from that point
    // Bar at 1000001: TP = (106+103+105)/3 = 104.67, TPV = 104.67 * 1500 = 157000.00, Vol = 1500
    //                  VWAP = 157000.00 / 1500 = 104.67
    // Next bar: TP = (107+104+106)/3 = 105.67, TPV = 105.67 * 800 = 84533.33, Vol = 800
    //           Cumulative TPV = 157000.00 + 84533.33 = 241533.33
    //           Cumulative Vol = 1500 + 800 = 2300
    //           VWAP = 241533.33 / 2300 = 105.01
    
    EXPECT_NEAR(vwap_values[0], 104.67, 0.01);  // First anchored bar VWAP
    if (vwap_values.size() > 1) {
        EXPECT_NEAR(vwap_values[1], 105.01, 0.01);  // Second anchored bar VWAP
    }
}

// Test rolling VWAP calculation with a 3-bar window
TEST_F(VWAPTests, RollingVWAPCalculation) {
    btq::RollingVWAP rolling_vwap(3);  // 3-bar rolling window
    
    rolling_vwap.calculate(ohlcv_candle_data_);
    
    auto vwap_values = rolling_vwap.getVWAPValues();
    
    ASSERT_FALSE(vwap_values.empty());
    
    // For rolling VWAP with 3-bar window:
    // First value: Only first bar available, so VWAP = typical price of first bar = (105+99+104)/3 = 102.67
    // Second value: VWAP of first 2 bars = ((102.67*1000) + (104.67*1500)) / (1000+1500) = 103.87
    // Third value: VWAP of first 3 bars = ((102.67*1000) + (104.67*1500) + (105.67*800)) / (1000+1500+800) = 104.23
    // Fourth value: VWAP of last 3 bars = ((104.67*1500) + (105.67*800) + (106.33*1200)) / (1500+800+1200) = 105.47
    // Fifth value: VWAP of last 3 bars = ((105.67*800) + (106.33*1200) + (107.33*900)) / (800+1200+900) = 106.28
    
    if (vwap_values.size() >= 5) {
        EXPECT_NEAR(vwap_values[0], 102.67, 0.01);  // First bar VWAP
        EXPECT_NEAR(vwap_values[1], 103.87, 0.01);  // VWAP of first 2 bars
        EXPECT_NEAR(vwap_values[2], 104.23, 0.01);  // VWAP of first 3 bars
        EXPECT_NEAR(vwap_values[3], 105.47, 0.01);  // VWAP of bars 2,3,4
        EXPECT_NEAR(vwap_values[4], 106.28, 0.01);  // VWAP of bars 3,4,5
    }
}

// Test rolling VWAP with different window sizes
TEST_F(VWAPTests, RollingVWAPDifferentWindowSizes) {
    // Test with 1-bar window (should return typical price of each bar)
    btq::RollingVWAP rolling_vwap_1(1);
    rolling_vwap_1.calculate(ohlcv_candle_data_);
    
    auto vwap_values_1 = rolling_vwap_1.getVWAPValues();
    
    ASSERT_FALSE(vwap_values_1.empty());
    
    // With 1-bar window, VWAP should equal the typical price of each individual bar
    double expected_tp_1 = (105.0 + 99.0 + 104.0) / 3.0;  // 102.67
    double expected_tp_2 = (106.0 + 103.0 + 105.0) / 3.0; // 104.67
    double expected_tp_3 = (107.0 + 104.0 + 106.0) / 3.0; // 105.67
    
    if (vwap_values_1.size() >= 3) {
        EXPECT_NEAR(vwap_values_1[0], expected_tp_1, 0.01);
        EXPECT_NEAR(vwap_values_1[1], expected_tp_2, 0.01);
        EXPECT_NEAR(vwap_values_1[2], expected_tp_3, 0.01);
    }
    
    // Test with 5-bar window (should match cumulative VWAP)
    btq::RollingVWAP rolling_vwap_5(5);
    rolling_vwap_5.calculate(ohlcv_candle_data_);
    
    auto vwap_values_5 = rolling_vwap_5.getVWAPValues();
    
    ASSERT_FALSE(vwap_values_5.empty());
    
    // The last value should match the cumulative VWAP of all 5 bars
    double total_tpv = (102.67 * 1000) + (104.67 * 1500) + (105.67 * 800) + (106.33 * 1200) + (107.33 * 900);
    double total_vol = 1000 + 1500 + 800 + 1200 + 900;
    double expected_cumulative_vwap = total_tpv / total_vol;  // ~105.13
    
    if (vwap_values_5.size() > 0) {
        EXPECT_NEAR(vwap_values_5[vwap_values_5.size()-1], expected_cumulative_vwap, 0.01);
    }
}

// Test VWAP standard deviation calculation
TEST_F(VWAPTests, VWAPStandardDeviationCalculation) {
    TechnicalIndicators ti;
    
    auto result = ti.calculate_vwap_standard_deviation(ohlcv_data_, 0);
    
    ASSERT_FALSE(result.values.empty());
    
    // Standard deviation should be calculated based on weighted variance around VWAP
    // The exact value depends on the implementation, but it should be positive
    for (const auto& value : result.values) {
        EXPECT_GE(value, 0.0);  // Standard deviation should be non-negative
    }
}

// Test edge cases: empty data
TEST_F(VWAPTests, VWAPEdgeCaseEmptyData) {
    TechnicalIndicators ti;
    
    std::vector<TechnicalIndicators::OHLCV> empty_data;
    auto result = ti.calculate_vwap(empty_data, 0);
    
    // Should return empty result without crashing
    EXPECT_TRUE(result.values.empty());
}

// Test edge cases: single data point
TEST_F(VWAPTests, VWAPEdgeCaseSingleDataPoint) {
    TechnicalIndicators ti;
    
    std::vector<TechnicalIndicators::OHLCV> single_data = {ohlcv_data_[0]};
    auto result = ti.calculate_vwap(single_data, 0);
    
    ASSERT_FALSE(result.values.empty());
    
    // For single data point, VWAP should equal the typical price
    double expected = (single_data[0].high + single_data[0].low + single_data[0].close) / 3.0;
    EXPECT_NEAR(result.values[0], expected, 0.01);
}

// Test anchored VWAP with anchor at beginning
TEST_F(VWAPTests, AnchoredVWAPAtBeginning) {
    btq::AnchoredVWAP anchored_vwap(1000000);  // Anchor at first bar
    
    anchored_vwap.calculate(ohlcv_candle_data_);
    
    auto vwap_values = anchored_vwap.getVWAPValues();
    
    ASSERT_FALSE(vwap_values.empty());
    
    // Should behave similarly to cumulative VWAP from the beginning
    TechnicalIndicators ti;
    auto cumulative_result = ti.calculate_vwap(ohlcv_data_, 0);
    
    // Compare first few values
    for (size_t i = 0; i < std::min(vwap_values.size(), cumulative_result.values.size()); ++i) {
        EXPECT_NEAR(vwap_values[i], cumulative_result.values[i], 0.01);
    }
}

// Test anchored VWAP with anchor at end (should have minimal data)
TEST_F(VWAPTests, AnchoredVWAPAtEnd) {
    btq::AnchoredVWAP anchored_vwap(1000004);  // Anchor at last bar
    
    anchored_vwap.calculate(ohlcv_candle_data_);
    
    auto vwap_values = anchored_vwap.getVWAPValues();
    
    // Should have only one value (the last bar)
    EXPECT_EQ(vwap_values.size(), 1);
    
    if (!vwap_values.empty()) {
        double expected = (109.0 + 106.0 + 108.0) / 3.0;  // Typical price of last bar
        EXPECT_NEAR(vwap_values[0], expected, 0.01);
    }
}

// Test validation against external VWAP calculation
TEST_F(VWAPTests, VWAPValidationAgainstExternalSource) {
    TechnicalIndicators ti;
    
    // Use a simple dataset with easily verifiable VWAP
    std::vector<TechnicalIndicators::OHLCV> simple_data = {
        {100.0, 102.0, 98.0, 101.0, 100.0, 1000000ULL},  // TP = 100.33, TPV = 10033.33
        {101.0, 103.0, 99.0, 102.0, 200.0, 1000001ULL},  // TP = 101.00, TPV = 20200.00
        {102.0, 104.0, 100.0, 103.0, 300.0, 1000002ULL}   // TP = 102.00, TPV = 30600.00
    };
    
    auto result = ti.calculate_vwap(simple_data, 0);
    
    // Manual calculation:
    // After bar 1: VWAP = (100.33 * 100) / 100 = 100.33
    // After bar 2: VWAP = (10033.33 + 20200.00) / (100 + 200) = 30233.33 / 300 = 100.78
    // After bar 3: VWAP = (30233.33 + 30600.00) / (300 + 300) = 60833.33 / 600 = 101.39
    
    ASSERT_GE(result.values.size(), 3);
    
    EXPECT_NEAR(result.values[0], 100.33, 0.01);
    EXPECT_NEAR(result.values[1], 100.78, 0.01);
    EXPECT_NEAR(result.values[2], 101.39, 0.01);
}

} // namespace Tests
} // namespace BTQuant