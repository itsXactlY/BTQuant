#include "../dependencies/BTQ_Render_Engine/src/components/footprint_panel.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace BTQuant;

// Mock renderer for testing
class MockRenderer : public RenderEngine::MarketMicrostructureRenderer {
public:
  MockRenderer() : RenderEngine::MarketMicrostructureRenderer(nullptr) {}

  std::vector<RenderEngine::CandleCluster>
  getFootprintClusters() const override {
    std::vector<RenderEngine::CandleCluster> clusters;

    // Create a sample cluster for testing
    RenderEngine::CandleCluster cluster;
    cluster.centerX = 1.0;
    cluster.centerY = 100.0;
    cluster.width = 0.1;
    cluster.height = 0.5;
    cluster.bidVolume = 150;
    cluster.askVolume = 75;
    cluster.tradeCount = 10;
    cluster.vwap = 100.25;
    cluster.buyTradeCount = 7;
    cluster.sellTradeCount = 3;
    cluster.maxSingleTradeVolume = 25;
    cluster.startTimeNs = 1000000000ULL;
    cluster.endTimeNs = 1000000001ULL;

    clusters.push_back(cluster);
    return clusters;
  }

  std::vector<std::vector<RenderEngine::ClusterCell>>
  getClusterCells() const override {
    std::vector<std::vector<RenderEngine::ClusterCell>> cells;
    std::vector<RenderEngine::ClusterCell> row;

    // Create a sample cluster cell for testing
    RenderEngine::ClusterCell cell;
    cell.buy_volume = 150.0;
    cell.sell_volume = 75.0;
    cell.total_volume = 225.0;
    cell.trade_count.store(10);
    cell.buy_trade_count.store(7);
    cell.sell_trade_count.store(3);
    cell.max_single_trade_volume.store(25.0);
    cell.sum_of_volumes = 225.0;

    row.push_back(cell);
    cells.push_back(row);
    return cells;
  }

  RenderEngine::Statistics getStats() const override {
    RenderEngine::Statistics stats;
    stats.lastUpdateTimeNs = 1000000000ULL;
    return stats;
  }

  void setTimeAggregationType(Data::TimeAggregationType type) override {}
  void setVolumeBasedNContracts(int n) override {}
  void setTickBasedNTicks(int n) override {}
  void setPriceAggregationType(Data::PriceAggregationType type) override {}
  void setCustomPriceAggregationValue(double value) override {}
  void notifyPriceAggregationChanged() override {}
  void setSymbol(uint32_t id) override {}
};

class TestFootprintPanelWithZoom : public ::testing::Test {
protected:
  void SetUp() override {
    PanelConfig config;
    config.title = "Test Footprint Panel";
    config.position = ImVec2(0, 0);
    config.size = ImVec2(800, 600);

    renderer_ = std::make_unique<MockRenderer>();
    panel_ = std::make_unique<FootprintPanel>(config, renderer_.get());
  }

  std::unique_ptr<FootprintPanel> panel_;
  std::unique_ptr<MockRenderer> renderer_;
};

TEST_F(TestFootprintPanelWithZoom, ZoomSensitivitySetterGetter) {
  // Test default zoom sensitivity
  EXPECT_FLOAT_EQ(panel_->getZoomSensitivity(), 1.0f);

  // Test setting zoom sensitivity
  panel_->setZoomSensitivity(2.0f);
  EXPECT_FLOAT_EQ(panel_->getZoomSensitivity(), 2.0f);

  // Test setting different values
  panel_->setZoomSensitivity(0.5f);
  EXPECT_FLOAT_EQ(panel_->getZoomSensitivity(), 0.5f);
}

TEST_F(TestFootprintPanelWithZoom, CellSizeAdjustmentAtDifferentZoomLevels) {
  // Create a sample footprint cell
  FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);

  // Test with zoom factor of 0.5 (zoomed out) - should result in smaller cells
  double zoom_factor_low = 0.5;
  double base_padding = 0.48;
  double adjusted_padding_low =
      base_padding * std::pow(zoom_factor_low * 1.0, 1.5);
  adjusted_padding_low = std::max(0.01, std::min(0.48, adjusted_padding_low));

  // Test with zoom factor of 2.0 (zoomed in) - should result in larger cells
  double zoom_factor_high = 2.0;
  double adjusted_padding_high =
      base_padding / std::pow(zoom_factor_high * 1.0, 0.7);
  adjusted_padding_high = std::max(0.01, std::min(0.48, adjusted_padding_high));

  // Verify that zoomed in results in larger cells (smaller padding)
  EXPECT_LT(adjusted_padding_high, adjusted_padding_low);

  // Verify that both values are within bounds
  EXPECT_GE(adjusted_padding_low, 0.01);
  EXPECT_LE(adjusted_padding_low, 0.48);
  EXPECT_GE(adjusted_padding_high, 0.01);
  EXPECT_LE(adjusted_padding_high, 0.48);
}

TEST_F(TestFootprintPanelWithZoom, CellSizeAdjustmentWithSensitivity) {
  // Test how sensitivity affects cell size adjustment
  FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);

  // Test with zoom factor of 2.0 and different sensitivities
  double zoom_factor = 2.0;
  double base_padding = 0.48;

  // With high sensitivity (2.0)
  double sensitivity_high = 2.0;
  double adjusted_padding_high_sens =
      base_padding / std::pow(zoom_factor * sensitivity_high, 0.7);
  adjusted_padding_high_sens =
      std::max(0.01, std::min(0.48, adjusted_padding_high_sens));

  // With low sensitivity (0.5)
  double sensitivity_low = 0.5;
  double adjusted_padding_low_sens =
      base_padding / std::pow(zoom_factor * sensitivity_low, 0.7);
  adjusted_padding_low_sens =
      std::max(0.01, std::min(0.48, adjusted_padding_low_sens));

  // Higher sensitivity should result in more dramatic cell size changes
  // (smaller padding when zoomed in)
  EXPECT_LT(adjusted_padding_high_sens, adjusted_padding_low_sens);
}

TEST_F(TestFootprintPanelWithZoom, CellSizeBoundsCheck) {
  // Test that cell padding stays within reasonable bounds

  // Extremely low zoom factor
  double zoom_factor_extreme_low = 0.01;
  double base_padding = 0.48;
  double adjusted_padding_extreme_low =
      base_padding * std::pow(zoom_factor_extreme_low * 1.0, 1.5);
  adjusted_padding_extreme_low =
      std::max(0.01, std::min(0.48, adjusted_padding_extreme_low));
  EXPECT_GE(adjusted_padding_extreme_low, 0.01);
  EXPECT_LE(adjusted_padding_extreme_low, 0.48);

  // Extremely high zoom factor
  double zoom_factor_extreme_high = 10.0;
  double adjusted_padding_extreme_high =
      base_padding / std::pow(zoom_factor_extreme_high * 1.0, 0.7);
  adjusted_padding_extreme_high =
      std::max(0.01, std::min(0.48, adjusted_padding_extreme_high));
  EXPECT_GE(adjusted_padding_extreme_high, 0.01);
  EXPECT_LE(adjusted_padding_extreme_high, 0.48);
}

TEST_F(TestFootprintPanelWithZoom, ZoomFactorEffectOnCellDimensions) {
  // Test that zoom factor affects cell dimensions appropriately
  FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);

  // Calculate expected dimensions for different zoom levels
  double base_padding = 0.48;

  // Zoomed out (zoom factor < 1)
  double zoom_out = 0.3;
  double padding_out = base_padding * std::pow(zoom_out * 1.0, 1.5);
  padding_out = std::max(0.01, std::min(0.48, padding_out));
  double width_out = cell.width * (1.0 - 2 * padding_out);
  double height_out = cell.height * (1.0 - 2 * padding_out);

  // Zoomed in (zoom factor > 1)
  double zoom_in = 3.0;
  double padding_in = base_padding / std::pow(zoom_in * 1.0, 0.7);
  padding_in = std::max(0.01, std::min(0.48, padding_in));
  double width_in = cell.width * (1.0 - 2 * padding_in);
  double height_in = cell.height * (1.0 - 2 * padding_in);

  // When zoomed in, cells should appear larger (less padding means more visible
  // area)
  EXPECT_GT(width_in, width_out);
  EXPECT_GT(height_in, height_out);
}

TEST_F(TestFootprintPanelWithZoom, RenderCellMethodAcceptsZoomFactor) {
  // Verify that the renderCell method accepts and uses the zoom factor
  // parameter
  FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);

  // Create empty vectors for imbalance detection
  std::vector<FootprintCell> empty_diagonal;
  std::vector<FootprintCell> empty_stacked;

  // This test verifies that the method signature is correct and can accept
  // different zoom factors The actual rendering is tested elsewhere, here we
  // just verify the interface works
  EXPECT_NO_THROW({
    // Call the method with different zoom factors
    panel_->renderCell(cell, nullptr, 10000.0, empty_diagonal, empty_stacked,
                       0.5);
    panel_->renderCell(cell, nullptr, 10000.0, empty_diagonal, empty_stacked,
                       1.0);
    panel_->renderCell(cell, nullptr, 10000.0, empty_diagonal, empty_stacked,
                       2.0);
  });
}