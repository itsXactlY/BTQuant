#include "../include/components/volume_profile_panel.hpp"
#include "../include/market_data_processor.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace BTQuant;

// Simple test to verify the step profile rendering functionality
TEST(StepProfileRenderingTest, MethodExistsAndCanBeCalled) {
    // Just verify that the method exists and can be called without crashing
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    PanelConfig config;
    config.title = "Test Volume Profile";

    VolumeProfilePanel panel(config, processor);

    // Create mock candle data
    std::vector<RenderEngine::OHLCVCandle> candles;
    std::vector<double> x_coords;
    std::vector<double> y_coords_high;
    std::vector<double> y_coords_low;

    // Add a few mock candles
    for (int i = 0; i < 3; ++i) {
        RenderEngine::OHLCVCandle candle;
        candle.timestamp = 1000000 + i * 100000;
        candle.open = 100.0 + i;
        candle.high = 105.0 + i;
        candle.low = 95.0 + i;
        candle.close = 102.0 + i;
        candle.volume = 100.0 + i * 10;
        candle.trade_count = 10 + i;

        candles.push_back(candle);
        x_coords.push_back(100.0 + i * 10);  // Mock x coordinates
        y_coords_high.push_back(105.0 + i);  // Mock y coordinates for high
        y_coords_low.push_back(95.0 + i);    // Mock y coordinates for low
    }

    // Create a mock draw list (we won't actually draw, just test the function doesn't crash)
    ImDrawList* draw_list = nullptr;  // In a real scenario, this would be a valid draw list

    // Call the method - this should not crash
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 8);
    });
}

TEST(StepProfileRenderingTest, EmptyCandlesDoesNotCrash) {
    // Test with empty candle data
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    PanelConfig config;
    config.title = "Test Volume Profile";

    VolumeProfilePanel panel(config, processor);

    // Create empty mock data
    std::vector<RenderEngine::OHLCVCandle> candles;
    std::vector<double> x_coords;
    std::vector<double> y_coords_high;
    std::vector<double> y_coords_low;

    // Create a mock draw list
    ImDrawList* draw_list = nullptr;

    // Call the method - this should not crash even with empty data
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 8);
    });
}

TEST(StepProfileRenderingTest, ZeroVolumeCandleHandled) {
    // Test with a candle that has zero volume
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    PanelConfig config;
    config.title = "Test Volume Profile";

    VolumeProfilePanel panel(config, processor);

    // Create mock candle data with zero volume
    std::vector<RenderEngine::OHLCVCandle> candles;
    std::vector<double> x_coords;
    std::vector<double> y_coords_high;
    std::vector<double> y_coords_low;

    RenderEngine::OHLCVCandle candle;
    candle.timestamp = 1000000;
    candle.open = 100.0;
    candle.high = 105.0;
    candle.low = 95.0;
    candle.close = 102.0;
    candle.volume = 0.0;  // Zero volume
    candle.trade_count = 0;

    candles.push_back(candle);
    x_coords.push_back(100.0);
    y_coords_high.push_back(105.0);
    y_coords_low.push_back(95.0);

    // Create a mock draw list
    ImDrawList* draw_list = nullptr;

    // Call the method - this should handle zero volume candles gracefully
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 8);
    });
}

TEST(StepProfileRenderingTest, DifferentBucketCounts) {
    // Test with different bucket counts
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    PanelConfig config;
    config.title = "Test Volume Profile";

    VolumeProfilePanel panel(config, processor);

    // Create mock candle data
    std::vector<RenderEngine::OHLCVCandle> candles;
    std::vector<double> x_coords;
    std::vector<double> y_coords_high;
    std::vector<double> y_coords_low;

    RenderEngine::OHLCVCandle candle;
    candle.timestamp = 1000000;
    candle.open = 100.0;
    candle.high = 105.0;
    candle.low = 95.0;
    candle.close = 102.0;
    candle.volume = 100.0;
    candle.trade_count = 10;

    candles.push_back(candle);
    x_coords.push_back(100.0);
    y_coords_high.push_back(105.0);
    y_coords_low.push_back(95.0);

    // Create a mock draw list
    ImDrawList* draw_list = nullptr;

    // Test with different bucket counts
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 4);
    });

    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 16);
    });
}

TEST(StepProfileRenderingTest, PocLineOption) {
    // Test with POC line enabled/disabled
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();

    PanelConfig config;
    config.title = "Test Volume Profile";

    VolumeProfilePanel panel(config, processor);

    // Create mock candle data
    std::vector<RenderEngine::OHLCVCandle> candles;
    std::vector<double> x_coords;
    std::vector<double> y_coords_high;
    std::vector<double> y_coords_low;

    RenderEngine::OHLCVCandle candle;
    candle.timestamp = 1000000;
    candle.open = 100.0;
    candle.high = 105.0;
    candle.low = 95.0;
    candle.close = 102.0;
    candle.volume = 100.0;
    candle.trade_count = 10;

    candles.push_back(candle);
    x_coords.push_back(100.0);
    y_coords_high.push_back(105.0);
    y_coords_low.push_back(95.0);

    // Create a mock draw list
    ImDrawList* draw_list = nullptr;

    // Test with POC line enabled
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, true, 8);
    });

    // Test with POC line disabled
    EXPECT_NO_THROW({
        panel.render_step_profile_histograms(draw_list, candles, x_coords,
                                           y_coords_high, y_coords_low, false, 8);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}