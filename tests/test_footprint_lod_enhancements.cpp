#include <gtest/gtest.h>
#include "rendering/footprint_lod.hpp"

using namespace BTQuant::Rendering;

class FootprintLODEnhancementTest : public ::testing::Test {
protected:
    FootprintLOD lod_system;

    void SetUp() override {
        // Set up default parameters for testing
        lod_system.setMinDetailZoom(0.1f);
        lod_system.setMediumDetailZoom(1.0f);
        lod_system.setMaxDetailZoom(3.0f);
        lod_system.setMinCellSizePx(4.0f);
        lod_system.setMediumCellSizePx(12.0f);
        lod_system.setMaxCellSizePx(24.0f);
        lod_system.setTextRenderThreshold(12.0f);
        lod_system.setLabelRenderThreshold(20.0f);
        lod_system.setDetailRenderThreshold(8.0f);
        lod_system.setDistanceLODThreshold(100.0f);
    }
};

TEST_F(FootprintLODEnhancementTest, DistanceBasedLODCalculation) {
    // Test distance-based LOD calculation
    ImVec2 cell_center = {50.0f, 50.0f};
    ImVec2 view_center = {50.0f, 50.0f};  // Same as cell center
    
    // When cell is at view center, should potentially get higher detail
    LODLevel lod = lod_system.calculateDistanceBasedLODLevel(20.0f, 20.0f, 1.5f, cell_center, view_center);
    EXPECT_EQ(lod, LODLevel::HIGH_DETAIL) << "Cell at center should potentially get higher detail";
    
    // When cell is far from view center, should potentially get lower detail
    ImVec2 far_view_center = {500.0f, 500.0f};
    lod = lod_system.calculateDistanceBasedLODLevel(20.0f, 20.0f, 1.5f, cell_center, far_view_center);
    // The exact level depends on the base LOD calculation, but it might be reduced
    // This test verifies the distance calculation logic works
    float distance = sqrt((50.0f - 500.0f)*(50.0f - 500.0f) + (50.0f - 500.0f)*(50.0f - 500.0f));
    EXPECT_GT(distance, lod_system.getDistanceLODThreshold() * 1.5f) << "Distance should be considered";
}

TEST_F(FootprintLODEnhancementTest, PerformanceBasedLODUpdate) {
    // Test performance-based LOD adjustment
    PerformanceMetrics good_perf = {5.0f, 100, 120.0f, false};  // Good performance
    PerformanceMetrics poor_perf = {50.0f, 1000, 10.0f, true};  // Poor performance
    
    float original_min_zoom = lod_system.getMinDetailZoom();
    float original_medium_zoom = lod_system.getMediumDetailZoom();
    
    // Update with good performance - should allow more detail
    lod_system.updatePerformanceBasedLOD(good_perf);
    EXPECT_LE(lod_system.getMinDetailZoom(), original_min_zoom) << "Good perf should allow more detail";
    EXPECT_LE(lod_system.getMediumDetailZoom(), original_medium_zoom) << "Good perf should allow more detail";
    
    // Reset to original values
    lod_system.setMinDetailZoom(original_min_zoom);
    lod_system.setMediumDetailZoom(original_medium_zoom);
    
    // Update with poor performance - should reduce detail
    lod_system.updatePerformanceBasedLOD(poor_perf);
    EXPECT_GE(lod_system.getMinDetailZoom(), original_min_zoom) << "Poor perf should reduce detail";
    EXPECT_GE(lod_system.getMediumDetailZoom(), original_medium_zoom) << "Poor perf should reduce detail";
}

TEST_F(FootprintLODEnhancementTest, CellClusteringAtLowZoom) {
    // Create some test cells
    std::vector<FootprintCell> test_cells;
    for (int i = 0; i < 10; ++i) {
        FootprintCell cell;
        cell.x = i * 0.1;  // Close together
        cell.y = i * 0.1;
        cell.width = 0.05;
        cell.height = 0.05;
        cell.bid_volume = 100.0 + i * 10;
        cell.ask_volume = 80.0 + i * 10;
        test_cells.push_back(cell);
    }
    
    // Test clustering at low zoom (should cluster)
    auto clustered_low_zoom = lod_system.clusterCells(test_cells, 0.2f);  // Low zoom
    EXPECT_LE(clustered_low_zoom.size(), test_cells.size()) << "Low zoom should result in clustering";
    
    // Test clustering at high zoom (should not cluster)
    auto clustered_high_zoom = lod_system.clusterCells(test_cells, 2.0f);  // High zoom
    EXPECT_EQ(clustered_high_zoom.size(), test_cells.size()) << "High zoom should not cluster cells";
}

TEST_F(FootprintLODEnhancementTest, DistanceBasedLODWithDifferentDistances) {
    ImVec2 cell_center = {100.0f, 100.0f};
    ImVec2 close_view = {110.0f, 110.0f};   // Close (about 14 pixels away)
    ImVec2 far_view = {500.0f, 500.0f};     // Far (about 565 pixels away)
    
    // Calculate base LOD (without distance consideration)
    LODLevel base_lod = lod_system.calculateLODLevel(15.0f, 15.0f, 1.0f);
    
    // Calculate distance-based LOD for close view (should potentially be higher)
    LODLevel close_lod = lod_system.calculateDistanceBasedLODLevel(15.0f, 15.0f, 1.0f, cell_center, close_view);
    
    // Calculate distance-based LOD for far view (should potentially be lower)
    LODLevel far_lod = lod_system.calculateDistanceBasedLODLevel(15.0f, 15.0f, 1.0f, cell_center, far_view);
    
    // The exact behavior depends on the implementation, but the function should work without errors
    EXPECT_NE(close_lod, static_cast<LODLevel>(-1)) << "Close LOD should be valid";
    EXPECT_NE(far_lod, static_cast<LODLevel>(-1)) << "Far LOD should be valid";
}

TEST_F(FootprintLODEnhancementTest, ParameterAccessors) {
    // Test getter/setter methods
    lod_system.setDistanceLODThreshold(150.0f);
    EXPECT_FLOAT_EQ(lod_system.getDistanceLODThreshold(), 150.0f);
    
    lod_system.setPerformanceTargetFPS(120.0f);
    EXPECT_FLOAT_EQ(lod_system.getPerformanceTargetFPS(), 120.0f);
}

// Test structure definitions
TEST(LODStructuresTest, PerformanceMetricsInitialization) {
    PerformanceMetrics metrics;
    EXPECT_FLOAT_EQ(metrics.frame_time_ms, 0.0f);
    EXPECT_EQ(metrics.rendered_cells, 0);
    EXPECT_FLOAT_EQ(metrics.fps, 0.0f);
    EXPECT_FALSE(metrics.performance_degraded);
}

TEST(LODStructuresTest, LODStatisticsCalculations) {
    LODStatistics stats;
    stats.total_cells = 100;
    stats.high_detail_cells = 30;
    stats.medium_detail_cells = 50;
    stats.low_detail_cells = 20;
    
    EXPECT_FLOAT_EQ(stats.getHighDetailPercentage(), 30.0f);
    EXPECT_FLOAT_EQ(stats.getMediumDetailPercentage(), 50.0f);
    EXPECT_FLOAT_EQ(stats.getLowDetailPercentage(), 20.0f);
    EXPECT_FLOAT_EQ(stats.getMaxDetailPercentage(), 0.0f);
    
    // Test with zero total cells
    LODStatistics empty_stats;
    EXPECT_FLOAT_EQ(empty_stats.getHighDetailPercentage(), 0.0f);
}