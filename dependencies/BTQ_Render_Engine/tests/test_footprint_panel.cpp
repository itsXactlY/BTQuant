#include "components/footprint_panel.hpp"
#include "components/MarketMicrostructureRenderer.h"
#include <gtest/gtest.h>
#include <memory>

using namespace BTQuant;

// Mock that inherits from the actual renderer to match the interface
class MockMarketMicrostructureRenderer : public RenderEngine::MarketMicrostructureRenderer {
public:
    MockMarketMicrostructureRenderer()
        : RenderEngine::MarketMicrostructureRenderer(nullptr, nullptr, nullptr, {}) {}

    // Override minimal methods to prevent actual Vulkan calls
    std::expected<void, RenderEngine::RendererError> initialize() override {
        return std::expected<void, RenderEngine::RendererError>{};
    }

    void cleanup() noexcept override {}

    std::expected<void, RenderEngine::RendererError> prepare() override {
        return std::expected<void, RenderEngine::RendererError>{};
    }

    void executeCompute(VkCommandBuffer cmdBuffer) override {}
    void executeGraphics(VkCommandBuffer cmdBuffer) override {}
    void updateLOBData(const HotspineOrderBookSnapshot& snapshot) override {}
    void updateTradeData(std::span<const HotspineTradeTick> trades) override {}
    void updateFootprintClusters(std::span<const CandleCluster> clusters) override {}
    void updateConfig(const RenderEngine::RendererConfig& config) override {}
    RenderEngine::RendererStats getStats() const override {
        RenderEngine::RendererStats stats{};
        return stats;
    }
    void resetStats() override {}
    void setSymbol(uint32_t symbol_id) override {}
    void onMarketDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type) override {}
    void recordFrameStats() override {}
    void* getHeatmapTextureID() override { return nullptr; }
    std::vector<CandleCluster> getFootprintClusters() const {
        return std::vector<CandleCluster>();
    }
    void updateUniformBuffers(uint32_t currentFrame) override {}
    void updateStorageBuffers() override {}
};

TEST(FootprintPanelTest, ConstructorInitialization) {
    PanelConfig config;
    config.title = "Test Footprint Panel";

    MockMarketMicrostructureRenderer* renderer = new MockMarketMicrostructureRenderer();
    FootprintPanel panel(config, renderer);

    EXPECT_EQ(panel.get_symbol_id(), 0);
    EXPECT_EQ(panel.getDataType(), Data::UnifiedDataPipeline::DataType::FOOTPRINT);

    delete renderer;
}

TEST(FootprintPanelTest, SetAndGetSymbolId) {
    PanelConfig config;
    MockMarketMicrostructureRenderer* renderer = new MockMarketMicrostructureRenderer();
    FootprintPanel panel(config, renderer);

    panel.set_symbol_id(123);
    EXPECT_EQ(panel.get_symbol_id(), 123);

    delete renderer;
}

TEST(FootprintPanelTest, SetAndGetDataType) {
    PanelConfig config;
    MockMarketMicrostructureRenderer* renderer = new MockMarketMicrostructureRenderer();
    FootprintPanel panel(config, renderer);

    panel.setDataType(Data::UnifiedDataPipeline::DataType::TRADES);
    EXPECT_EQ(panel.getDataType(), Data::UnifiedDataPipeline::DataType::TRADES);

    panel.setDataType(Data::UnifiedDataPipeline::DataType::VOLUME_PROFILE);
    EXPECT_EQ(panel.getDataType(), Data::UnifiedDataPipeline::DataType::VOLUME_PROFILE);

    delete renderer;
}

TEST(FootprintPanelTest, ConfigurationMethods) {
    PanelConfig config;
    MockMarketMicrostructureRenderer* renderer = new MockMarketMicrostructureRenderer();
    FootprintPanel panel(config, renderer);

    // Test grid size configuration
    panel.setGridSize(50, 75);

    // Test visualization options
    panel.setShowVolumeLabels(false);
    panel.setShowDeltaIndicator(false);
    panel.setDeltaThreshold(0.5f);

    delete renderer;
}

TEST(FootprintPanelTest, DefaultValues) {
    PanelConfig config;
    MockMarketMicrostructureRenderer* renderer = new MockMarketMicrostructureRenderer();
    FootprintPanel panel(config, renderer);

    EXPECT_EQ(panel.getDataType(), Data::UnifiedDataPipeline::DataType::FOOTPRINT);
    EXPECT_EQ(panel.get_symbol_id(), 0);

    delete renderer;
}