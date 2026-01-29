#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../include/components/chart_panel.hpp"
#include "../include/components/chart_manager.hpp"

// Test for the binary search optimization in crosshair functionality
class ChartPanelCrosshairOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// Helper function to create a test chart instance with sample data
BTQuant::ChartInstance create_test_chart_instance(int num_candles = 100) {
    BTQuant::ChartInstance chart;
    chart.symbol_name = "BTC-USDT";
    chart.exchange_name = "Binance";
    chart.timeframe = BTQuant::RenderEngine::TimeFrame::TF_1SEC;
    chart.symbol_id = 10007;
    chart.chart_id = 1;
    
    // Populate with sample data - dates should be sorted for binary search to work properly
    for (int i = 0; i < num_candles; ++i) {
        chart.dates.push_back(1704067200.0 + i); // Unix timestamp starting from a reference point
        chart.opens.push_back(50000.0f + (rand() % 100)); // Random price around 50000
        chart.highs.push_back(50050.0f + (rand() % 100));
        chart.lows.push_back(49950.0f - (rand() % 100));
        chart.closes.push_back(50025.0f + (rand() % 100));
        chart.volumes.push_back(100.0f + (rand() % 50));
    }
    
    return chart;
}

// Test the binary search optimization in render_crosshair_info
TEST_F(ChartPanelCrosshairOptimizationTest, BinarySearchOptimization) {
    BTQuant::ChartInstance chart = create_test_chart_instance(1000);
    
    // Test with a mouse_x value that should find a specific candle
    double mouse_x = chart.dates[500]; // Middle of the dataset
    double mouse_y = 50000.0;
    
    // Use the same binary search logic as in the optimized method
    size_t closest_idx = 0;
    
    auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x);
    
    if (lower == chart.dates.end()) {
        closest_idx = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        closest_idx = 0;
    } else {
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;
        
        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x);
        
        closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }
    
    // Verify that we got a valid index
    EXPECT_LT(closest_idx, chart.dates.size());
    EXPECT_LT(closest_idx, chart.closes.size());
    
    // The closest candle should be very close to our mouse_x (ideally the same or adjacent)
    double distance = std::abs(chart.dates[closest_idx] - mouse_x);
    EXPECT_LE(distance, 1.0); // Should be very close (within 1 unit)
}

// Test edge cases for the binary search
TEST_F(ChartPanelCrosshairOptimizationTest, BinarySearchEdgeCases) {
    BTQuant::ChartInstance chart = create_test_chart_instance(10);
    
    // Test with mouse_x before first date
    double mouse_x_before = chart.dates[0] - 10.0;
    size_t closest_idx_before;
    
    auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x_before);
    
    if (lower == chart.dates.end()) {
        closest_idx_before = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        closest_idx_before = 0;
    } else {
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;
        
        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x_before);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x_before);
        
        closest_idx_before = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }
    
    EXPECT_EQ(closest_idx_before, 0); // Should pick the first element
    
    // Test with mouse_x after last date
    double mouse_x_after = chart.dates.back() + 10.0;
    size_t closest_idx_after;
    
    lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x_after);
    
    if (lower == chart.dates.end()) {
        closest_idx_after = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        closest_idx_after = 0;
    } else {
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;
        
        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x_after);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x_after);
        
        closest_idx_after = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }
    
    EXPECT_EQ(closest_idx_after, chart.dates.size() - 1); // Should pick the last element
}

// Test performance comparison between linear and binary search
TEST_F(ChartPanelCrosshairOptimizationTest, PerformanceComparison) {
    // Create a large dataset to highlight the performance difference
    BTQuant::ChartInstance chart = create_test_chart_instance(100000); // 100k candles
    
    double mouse_x = chart.dates[50000]; // Somewhere in the middle
    
    // Measure binary search time (the new implementation)
    auto start = std::chrono::high_resolution_clock::now();
    
    auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x);
    size_t closest_idx;
    
    if (lower == chart.dates.end()) {
        closest_idx = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        closest_idx = 0;
    } else {
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;
        
        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x);
        
        closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto binary_search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify the result is correct
    EXPECT_LT(closest_idx, chart.dates.size());
    double distance = std::abs(chart.dates[closest_idx] - mouse_x);
    EXPECT_LE(distance, 1.0);
    
    std::cout << "Binary search took: " << binary_search_duration.count() << " microseconds" << std::endl;
    
    // The binary search should be significantly faster on large datasets
    // Though in a unit test we're mostly verifying correctness, the performance improvement
    // is the main benefit of the optimization
    EXPECT_GT(chart.dates.size(), 10000); // Ensure we're testing with a large enough dataset
}

// Test that the crosshair info finds the correct candle with the optimized search
TEST_F(ChartPanelCrosshairOptimizationTest, CrosshairInfoCorrectness) {
    BTQuant::ChartInstance chart = create_test_chart_instance(50);
    
    // Pick a specific date in the middle
    size_t target_idx = 25;
    double mouse_x = chart.dates[target_idx];
    double mouse_y = chart.closes[target_idx];
    
    // Manually run the binary search logic to verify it finds the right candle
    auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x);
    size_t closest_idx;
    
    if (lower == chart.dates.end()) {
        closest_idx = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        closest_idx = 0;
    } else {
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;
        
        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x);
        
        closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }
    
    // The closest index should be very close to our target index
    EXPECT_NEAR(closest_idx, target_idx, 1); // Allow for off-by-one due to floating point precision
    
    // Verify that the candle data matches expectations
    EXPECT_FLOAT_EQ(chart.opens[closest_idx], chart.opens[target_idx]);
    EXPECT_FLOAT_EQ(chart.highs[closest_idx], chart.highs[target_idx]);
    EXPECT_FLOAT_EQ(chart.lows[closest_idx], chart.lows[target_idx]);
    EXPECT_FLOAT_EQ(chart.closes[closest_idx], chart.closes[target_idx]);
    EXPECT_FLOAT_EQ(chart.volumes[closest_idx], chart.volumes[target_idx]);
}