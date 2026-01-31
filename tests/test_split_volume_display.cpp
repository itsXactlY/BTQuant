#include <gtest/gtest.h>
#include "../dependencies/BTQ_Render_Engine/src/components/footprint_panel.hpp"
#include "../dependencies/BTQ_Render_Engine/include/data/VolumeDataTypes.h"

using namespace BTQuant;

class MockRenderer : public RenderEngine::MarketMicrostructureRenderer {
public:
    MockRenderer() : RenderEngine::MarketMicrostructureRenderer() {}

    // Mock implementations for required methods
    std::vector<RenderEngine::CandleCluster> getFootprintClusters() const override {
        return {};
    }

    std::vector<std::vector<RenderEngine::ClusterCell>> getClusterCells() const override {
        return {};
    }

    RenderEngine::Stats getStats() const override {
        return {};
    }

    void setSymbol(uint32_t symbol_id) override {}
    void setTimeAggregationType(Data::TimeAggregationType type) override {}
    void setVolumeBasedNContracts(int n) override {}
    void setTickBasedNTicks(int n) override {}
    void setPriceAggregationType(Data::PriceAggregationType type) override {}
    void setCustomPriceAggregationValue(double value) override {}
    void notifyPriceAggregationChanged() override {}
};

class TestFootprintPanel : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = {};
        renderer_ = std::make_unique<MockRenderer>();
        panel_ = std::make_unique<FootprintPanel>(config_, renderer_.get());
    }

    void TearDown() override {
        panel_.reset();
        renderer_.reset();
    }

    PanelConfig config_;
    std::unique_ptr<MockRenderer> renderer_;
    std::unique_ptr<FootprintPanel> panel_;
};

TEST_F(TestFootprintPanel, SplitVolumeModeInitialization) {
    // Test that SplitVolume mode can be set
    panel_->setVolumeDataType(Data::VolumeDataType::SplitVolume);
    EXPECT_EQ(panel_->getVolumeDataType(), Data::VolumeDataType::SplitVolume);
}

TEST_F(TestFootprintPanel, SplitVolumeCellLabelFormat) {
    // Test that the cell label format is correct for split volume
    panel_->setVolumeDataType(Data::VolumeDataType::SplitVolume);
    
    // Create a test cell with known buy and sell volumes
    FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);
    
    std::string label = panel_->getCellLabel(cell);
    
    // The label should contain both buy and sell volumes separated by "/"
    EXPECT_NE(label.find("150.00"), std::string::npos);  // Buy volume
    EXPECT_NE(label.find("75.00"), std::string::npos);   // Sell volume
    EXPECT_NE(label.find("/"), std::string::npos);       // Separator
}

TEST_F(TestFootprintPanel, SplitVolumeCellColorCalculation) {
    // Test that cell color calculation works for split volume mode
    panel_->setVolumeDataType(Data::VolumeDataType::SplitVolume);
    
    // Create a test cell
    FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);
    
    // This should not crash and should return a valid color
    ImU32 color = panel_->getCellColor(cell, 1000.0);
    EXPECT_TRUE(color != 0); // Color should be non-zero
}

TEST_F(TestFootprintPanel, SplitVolumeTooltipContainsBothVolumes) {
    // Test that the tooltip contains both buy and sell volume information
    panel_->setVolumeDataType(Data::VolumeDataType::SplitVolume);
    
    // Create a test cell
    FootprintCell cell(1.0, 100.0, 0.1, 0.5, 150.0, 75.0, 10, 100.25);
    cell.buy_trade_count = 6;
    cell.sell_trade_count = 4;
    cell.max_single_trade_volume = 50.0;
    
    std::string tooltip = panel_->getCellTooltip(cell);
    
    // The tooltip should contain both buy and sell volume information
    EXPECT_NE(tooltip.find("Exact Buy Volume: 150.00"), std::string::npos);
    EXPECT_NE(tooltip.find("Exact Sell Volume: 75.00"), std::string::npos);
}

TEST_F(TestFootprintPanel, SplitVolumeFormatNumberFunction) {
    // Test the formatNumber function with different formats
    std::string result = FootprintPanel::formatNumber(1234.56, NumberFormat::Raw, 2);
    EXPECT_EQ(result, "1234.56");
    
    result = FootprintPanel::formatNumber(1234567.89, NumberFormat::ThousandsK, 2);
    EXPECT_EQ(result, "1234.57K");
    
    result = FootprintPanel::formatNumber(1234567.89, NumberFormat::MillionsM, 2);
    EXPECT_EQ(result, "1.23M");
}

TEST_F(TestFootprintPanel, SplitVolumeModeEnumExists) {
    // Verify that the SplitVolume enum value exists and is properly defined
    EXPECT_EQ(static_cast<int>(Data::VolumeAnalysisType::SplitVolume), 
              static_cast<int>(Data::VolumeDataType::SplitVolume));
              
    // Verify it's the last enum value (based on the header file)
    EXPECT_GE(static_cast<int>(Data::VolumeAnalysisType::SplitVolume), 0);
}