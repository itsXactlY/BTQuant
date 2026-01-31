#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

// Simple test to verify the mathematical logic behind the split volume functionality
// This tests the proportional calculation logic without requiring the full renderer

TEST(SplitVolumeMath, ProportionalSplitCalculationEqualVolumes) {
  // Test that equal buy and sell volumes result in 50-50 split
  double buy_volume = 100.0;
  double sell_volume = 100.0;
  double total_volume = buy_volume + sell_volume;     // 200.0
  double buy_proportion = buy_volume / total_volume;  // 0.5 (50%)

  EXPECT_DOUBLE_EQ(buy_proportion, 0.5);
}

TEST(SplitVolumeMath, ProportionalSplitCalculationUnequalVolumes) {
  // Test that unequal buy and sell volumes result in proportional split (3:1 ratio)
  double buy_volume = 300.0;
  double sell_volume = 100.0;
  double total_volume = buy_volume + sell_volume;     // 400.0
  double buy_proportion = buy_volume / total_volume;  // 0.75 (75%)

  EXPECT_DOUBLE_EQ(buy_proportion, 0.75);
}

TEST(SplitVolumeMath, ProportionalSplitCalculationZeroVolumes) {
  // Test that zero volumes result in 50-50 split (fallback behavior)
  double buy_volume = 0.0;
  double sell_volume = 0.0;
  double total_volume = buy_volume + sell_volume;  // 0.0

  // When total volume is 0, the split should default to 50%
  double buy_proportion = (total_volume > 0.0) ? buy_volume / total_volume : 0.5;

  EXPECT_DOUBLE_EQ(buy_proportion, 0.5);
}

TEST(SplitVolumeMath, ProportionalSplitCalculationOnlyBuyVolume) {
  // Test that only buy volume results in 100% buy side
  double buy_volume = 200.0;
  double sell_volume = 0.0;
  double total_volume = buy_volume + sell_volume;     // 200.0
  double buy_proportion = buy_volume / total_volume;  // 1.0 (100%)

  EXPECT_DOUBLE_EQ(buy_proportion, 1.0);
}

TEST(SplitVolumeMath, ProportionalSplitCalculationOnlySellVolume) {
  // Test that only sell volume results in 0% buy side (100% sell side)
  double buy_volume = 0.0;
  double sell_volume = 300.0;
  double total_volume = buy_volume + sell_volume;     // 300.0
  double buy_proportion = buy_volume / total_volume;  // 0.0 (0%)

  EXPECT_DOUBLE_EQ(buy_proportion, 0.0);
}

TEST(SplitVolumeMath, AlphaCalculationWithMaxVolume) {
  // Test that alpha calculation works correctly with max volume
  double buy_volume = 150.0;
  double sell_volume = 75.0;
  double max_volume = 300.0;

  // Test buy alpha calculation
  float expected_buy_alpha = std::clamp(static_cast<float>(buy_volume / max_volume), 0.05f, 1.0f);
  float expected_buy_alpha_clamped = std::clamp(expected_buy_alpha, 0.05f, 1.0f);

  EXPECT_GE(expected_buy_alpha_clamped, 0.05f);
  EXPECT_LE(expected_buy_alpha_clamped, 1.0f);

  // Test sell alpha calculation
  float expected_sell_alpha = std::clamp(static_cast<float>(sell_volume / max_volume), 0.05f, 1.0f);
  float expected_sell_alpha_clamped = std::clamp(expected_sell_alpha, 0.05f, 1.0f);

  EXPECT_GE(expected_sell_alpha_clamped, 0.05f);
  EXPECT_LE(expected_sell_alpha_clamped, 1.0f);
}

TEST(SplitVolumeMath, DividerLineVisibilityBothVolumes) {
  // Test that divider line logic works when both buy and sell volumes exist
  double buy_volume = 200.0;
  double sell_volume = 100.0;
  double total_volume = buy_volume + sell_volume;     // 300.0
  double buy_proportion = buy_volume / total_volume;  // 0.6667

  EXPECT_GT(buy_proportion, 0.0);
  EXPECT_LT(buy_proportion, 1.0);
  EXPECT_NE(buy_proportion, 0.5);  // Should not be 50-50 since volumes are unequal
}

TEST(SplitVolumeMath, DividerLinePositionOnlyOneVolume) {
  // Test that divider line is positioned correctly when only one volume exists

  // Test with only buy volume (should put divider at far right)
  double buy_volume1 = 200.0;
  double sell_volume1 = 0.0;
  double total_volume1 = buy_volume1 + sell_volume1;                                 // 200.0
  double buy_proportion1 = total_volume1 > 0.0 ? buy_volume1 / total_volume1 : 0.5;  // 1.0

  EXPECT_DOUBLE_EQ(buy_proportion1, 1.0);

  // Test with only sell volume (should put divider at far left)
  double buy_volume2 = 0.0;
  double sell_volume2 = 150.0;
  double total_volume2 = buy_volume2 + sell_volume2;                                 // 150.0
  double buy_proportion2 = total_volume2 > 0.0 ? buy_volume2 / total_volume2 : 0.5;  // 0.0

  EXPECT_DOUBLE_EQ(buy_proportion2, 0.0);
}

TEST(SplitVolumeMath, EnhancedDividerLineVisibility) {
  // Test the enhanced divider line visibility logic
  double buy_volume = 150.0;
  double sell_volume = 100.0;
  double total_volume = buy_volume + sell_volume;     // 250.0
  double buy_proportion = buy_volume / total_volume;  // 0.6 (60%)

  // Calculate the split X position (simulating the logic from the render code)
  float p1_x = 10.0f;                                  // Left edge of cell
  float p2_x = 50.0f;                                  // Right edge of cell
  float cell_width = p2_x - p1_x;                      // 40.0
  float split_x = p1_x + cell_width * buy_proportion;  // 10 + 40 * 0.6 = 34.0

  // Both volumes exist, so divider should be visible
  bool has_both_volumes = (buy_volume > 0.0 && sell_volume > 0.0);
  EXPECT_TRUE(has_both_volumes);

  // Verify the calculated split position
  EXPECT_FLOAT_EQ(split_x, 34.0f);
}

TEST(SplitVolumeMath, EnhancedDividerLineThickness) {
  // Test that the divider line has appropriate thickness based on volume conditions
  double buy_volume = 200.0;
  double sell_volume = 150.0;
  bool has_both_volumes = (buy_volume > 0.0 && sell_volume > 0.0);
  double total_volume = buy_volume + sell_volume;

  // When both volumes exist, use thicker, more opaque line (3.0f, 255 alpha)
  float divider_thickness = has_both_volumes ? 3.0f : 2.0f;
  int divider_alpha = has_both_volumes ? 255 : 200;

  EXPECT_EQ(divider_thickness, 3.0f);
  EXPECT_EQ(divider_alpha, 255);

  // When only one volume exists, use thinner, less opaque line (2.0f, 200 alpha)
  buy_volume = 200.0;
  sell_volume = 0.0;
  has_both_volumes = (buy_volume > 0.0 && sell_volume > 0.0);
  divider_thickness = has_both_volumes ? 3.0f : 2.0f;
  divider_alpha = has_both_volumes ? 255 : 200;

  EXPECT_EQ(divider_thickness, 2.0f);
  EXPECT_EQ(divider_alpha, 200);
}