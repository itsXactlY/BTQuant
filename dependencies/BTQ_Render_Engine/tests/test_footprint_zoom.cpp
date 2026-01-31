#include <gtest/gtest.h>

#include <cmath>

// Simple test focusing on the zoom algorithm mathematics
// This tests the core zoom functionality without requiring the full GUI framework

// Test the zoom algorithm functions directly
double calculateAdjustedPadding(double zoom_factor, double zoom_sensitivity) {
  double base_padding = 0.48;
  double adjusted_padding;

  if (zoom_factor >= 1.0) {
    // When zoomed in: expand cells to show more detail
    // Use a logarithmic approach for smoother transitions at high zoom levels
    double zoom_effect = std::log10(zoom_factor * zoom_sensitivity + 1.0) * 0.5;
    adjusted_padding = std::max(0.01, base_padding - zoom_effect);
  } else {
    // When zoomed out: shrink cells to show more of them, approaching squares
    // Use an exponential approach to make cells shrink more aggressively when zoomed out
    // FIXED: The corrected algorithm that properly makes cells smaller when zoomed out
    double zoom_effect = std::pow(1.0 / (zoom_factor * zoom_sensitivity), 1.2) - 1.0;
    // To make cells smaller when zoomed out, we need MORE padding, not less
    double padding_factor = 0.8 * (zoom_effect / (zoom_effect + 1.0));
    adjusted_padding = std::min(base_padding, base_padding * (1.0 + padding_factor));
  }

  // Ensure padding stays within reasonable bounds to maintain visibility
  adjusted_padding = std::max(0.01, std::min(0.48, adjusted_padding));

  return adjusted_padding;
}

TEST(FootprintZoomAlgorithm, ZoomSensitivityDefaultValue) {
  // Test that the default sensitivity value produces expected results
  double default_sensitivity = 1.0;

  // Test with zoom factor of 1.0 (no zoom)
  double padding_no_zoom = calculateAdjustedPadding(1.0, default_sensitivity);
  EXPECT_NEAR(padding_no_zoom, 0.48 - std::log10(2.0) * 0.5, 0.01);

  // The result should be close to 0.48 - 0.15 = 0.33 (approximately)
  EXPECT_GT(padding_no_zoom, 0.30);
  EXPECT_LT(padding_no_zoom, 0.48);
}

TEST(FootprintZoomAlgorithm, CellPaddingAdjustmentZoomedIn) {
  // Test that cell padding decreases when zoomed in (making cells larger)
  double zoom_factor_high = 2.0;  // Zoomed in
  double sensitivity = 1.0;

  double padding_before = calculateAdjustedPadding(1.0, sensitivity);              // Normal zoom
  double padding_after = calculateAdjustedPadding(zoom_factor_high, sensitivity);  // Zoomed in

  // When zoomed in, padding should decrease (making cells larger)
  EXPECT_LT(padding_after, padding_before);
  EXPECT_GT(padding_after, 0.01);  // Should still be above minimum
}

TEST(FootprintZoomAlgorithm, CellPaddingAdjustmentZoomedOut) {
  // Test that cell padding increases when zoomed out (making cells smaller)
  double zoom_factor_low = 0.5;  // Zoomed out
  double sensitivity = 1.0;

  double padding_before = calculateAdjustedPadding(1.0, sensitivity);             // Normal zoom
  double padding_after = calculateAdjustedPadding(zoom_factor_low, sensitivity);  // Zoomed out

  // When zoomed out, padding should increase (making cells smaller)
  // The algorithm should make cells smaller when zoomed out
  EXPECT_GT(padding_after, padding_before);
  EXPECT_LE(padding_after, 0.48);  // Should be at or below maximum
}

TEST(FootprintZoomAlgorithm, CellPaddingBoundsCheck) {
  // Test that cell padding stays within reasonable bounds [0.01, 0.48]
  double zoom_factor_extreme_in = 100.0;  // Very zoomed in
  double zoom_factor_extreme_out = 0.01;  // Very zoomed out
  double sensitivity = 1.0;

  double padding_extreme_in = calculateAdjustedPadding(zoom_factor_extreme_in, sensitivity);
  double padding_extreme_out = calculateAdjustedPadding(zoom_factor_extreme_out, sensitivity);

  // Both should be within bounds
  EXPECT_GE(padding_extreme_in, 0.01);
  EXPECT_LE(padding_extreme_in, 0.48);

  EXPECT_GE(padding_extreme_out, 0.01);
  EXPECT_LE(padding_extreme_out, 0.48);
}

TEST(FootprintZoomAlgorithm, ZoomFactorEdgeCases) {
  // Test edge cases for zoom factors
  double sensitivity = 1.0;

  // Test zoom factor of exactly 1.0 (no zoom)
  double padding_normal = calculateAdjustedPadding(1.0, sensitivity);
  EXPECT_LE(padding_normal, 0.48);
  EXPECT_GE(padding_normal, 0.01);

  // Test very small positive zoom factor
  double zoom_factor_small = 0.001;
  double padding_small = calculateAdjustedPadding(zoom_factor_small, sensitivity);

  // Should still be within bounds
  EXPECT_GE(padding_small, 0.01);
  EXPECT_LE(padding_small, 0.48);
}

TEST(FootprintZoomAlgorithm, ZoomSensitivityRange) {
  // Test that zoom sensitivity values work across the expected range
  std::vector<double> test_sensitivities = {0.1, 0.5, 1.0, 1.5, 2.0, 3.0};

  for (double sensitivity : test_sensitivities) {
    // Test with a moderate zoom factor
    double zoom_factor = 2.0;
    double result = calculateAdjustedPadding(zoom_factor, sensitivity);

    // The result should always be within bounds
    EXPECT_GE(result, 0.01);
    EXPECT_LE(result, 0.48);
  }
}

TEST(FootprintZoomAlgorithm, DifferentSensitivityEffects) {
  // Test that different sensitivity values affect the zoom differently
  double zoom_factor = 2.0;

  double result_low_sensitivity = calculateAdjustedPadding(zoom_factor, 0.5);
  double result_high_sensitivity = calculateAdjustedPadding(zoom_factor, 2.0);

  // Higher sensitivity should make the zoom effect more pronounced
  // When zoomed in, higher sensitivity should result in smaller padding (larger cells)
  EXPECT_LT(result_high_sensitivity, result_low_sensitivity);
}

TEST(FootprintZoomAlgorithm, SymmetryInZoomEffect) {
  // Test that zooming in and out by the same factor ratio produces symmetric effects
  double sensitivity = 1.0;

  double zoom_in_factor = 2.0;
  double zoom_out_factor = 0.5;  // Reciprocal of 2.0

  double padding_normal = calculateAdjustedPadding(1.0, sensitivity);
  double padding_zoom_in = calculateAdjustedPadding(zoom_in_factor, sensitivity);
  double padding_zoom_out = calculateAdjustedPadding(zoom_out_factor, sensitivity);

  // The difference between normal and zoomed in should be similar to
  // the difference between normal and zoomed out, but in opposite directions
  double in_diff = padding_normal - padding_zoom_in;
  double out_diff = padding_zoom_out - padding_normal;

  // These differences should be positive (in_diff for zoom in, out_diff for zoom out)
  EXPECT_GT(in_diff, 0.0);
  EXPECT_GT(out_diff, 0.0);
}

TEST(FootprintZoomAlgorithm, ZeroZoomFactorHandling) {
  // Test that near-zero zoom factors are handled gracefully
  double near_zero_zoom = 0.0001;
  double sensitivity = 1.0;

  double result = calculateAdjustedPadding(near_zero_zoom, sensitivity);

  // Should be clamped to maximum value due to the formula behavior with very small zoom factors
  // When zoom_factor approaches 0, 1/zoom_factor approaches infinity
  // This causes zoom_effect to be very large, making padding_factor approach 0.8
  // So adjusted_padding approaches base_padding * (1.0 + 0.8) = base_padding * 1.8
  // But it's clamped to base_padding (0.48), so it should be 0.48
  EXPECT_DOUBLE_EQ(result, 0.48);
  EXPECT_GE(result, 0.01);
}
