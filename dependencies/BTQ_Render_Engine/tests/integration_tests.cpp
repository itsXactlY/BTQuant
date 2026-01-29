#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

// Project headers
#include "components/panel_manager.hpp"
#include "components/performance_monitor_panel.hpp"
#include "components/quant_workspace_component.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "data/market_data_processor.hpp"
#include "performance_monitor.hpp"

using namespace BTQuant;

class IntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock data bridge and processor
    processor_ = std::make_shared<RenderEngine::MarketDataProcessor>();
    bridge_ = std::make_shared<HotSpineDataBridge>("/integration_test");

    // Initialize managers
    order_manager_ = std::make_shared<OrderManager>();
    position_manager_ = std::make_shared<PositionManager>();
    risk_assessment_ = std::make_shared<RiskAssessment>();

    // Initialize micro renderer (for footprint and TPO)
    micro_renderer_ = nullptr;  // Will be mocked or null for basic tests

    // Initialize panel manager
    panel_manager_ = std::make_unique<PanelManager>(
        bridge_, processor_, order_manager_, position_manager_, risk_assessment_, micro_renderer_);
    panel_manager_->initialize();
  }

  void TearDown() override { panel_manager_.reset(); }

  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;
  RenderEngine::MarketMicrostructureRenderer* micro_renderer_;
  std::unique_ptr<PanelManager> panel_manager_;
};

TEST_F(IntegrationTest, PanelManagerInitialization) {
  ASSERT_NE(panel_manager_, nullptr);
  EXPECT_GT(panel_manager_->get_panel_count(), 0);  // Should have default panels
}

TEST_F(IntegrationTest, AddAllComponentPanels) {
  // Test adding all the panels mentioned in Phase 3

  // DOM Surface
  uint32_t dom_panel_id = panel_manager_->add_panel(PanelType::HEATMAP, "DOM Surface Test");
  EXPECT_NE(dom_panel_id, 0);

  // Chart System
  uint32_t chart_panel_id = panel_manager_->add_panel(PanelType::CHART, "Chart Test");
  EXPECT_NE(chart_panel_id, 0);

  // Volume Profile
  uint32_t vp_panel_id =
      panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "Volume Profile Test");
  EXPECT_NE(vp_panel_id, 0);

  // Time & Sales (Tape)
  uint32_t tape_panel_id = panel_manager_->add_panel(PanelType::TAPE, "Time & Sales Test");
  EXPECT_NE(tape_panel_id, 0);

  // Watchlist
  uint32_t watchlist_panel_id = panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist Test");
  EXPECT_NE(watchlist_panel_id, 0);

  // Footprint Chart
  uint32_t footprint_panel_id =
      panel_manager_->add_panel(PanelType::FOOTPRINT_CHART, "Footprint Chart Test");
  EXPECT_NE(footprint_panel_id, 0);

  // TPO Profile
  uint32_t tpo_panel_id = panel_manager_->add_panel(PanelType::TPO_PROFILE, "TPO Profile Test");
  EXPECT_NE(tpo_panel_id, 0);

  // Performance Monitor
  uint32_t perf_panel_id =
      panel_manager_->add_panel(PanelType::PERFORMANCE_MONITOR, "Performance Monitor Test");
  EXPECT_NE(perf_panel_id, 0);

  // Verify all panels were created
  EXPECT_EQ(panel_manager_->get_panel_count(), 8);  // 8 additional panels + default panels
}

TEST_F(IntegrationTest, PerformanceMonitorIntegration) {
  // Add a performance monitor panel
  uint32_t perf_panel_id =
      panel_manager_->add_panel(PanelType::PERFORMANCE_MONITOR, "Performance Monitor Test");
  EXPECT_NE(perf_panel_id, 0);

  // Simulate updating the performance monitor
  g_performance_monitor.start_frame();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  g_performance_monitor.end_frame();

  // Update the panel manager to trigger panel updates
  panel_manager_->update(0.016f);  // ~60fps

  // Verify the performance monitor is working
  EXPECT_GE(g_performance_monitor.get_fps(), 0.0);
  EXPECT_GE(g_performance_monitor.get_frame_time_ms(), 0.0);
}

TEST_F(IntegrationTest, SymbolPropagationAcrossPanels) {
  // Add multiple panels that should react to symbol changes
  uint32_t chart_panel_id = panel_manager_->add_panel(PanelType::CHART, "Chart Test");
  uint32_t orderbook_panel_id = panel_manager_->add_panel(PanelType::ORDERBOOK, "Orderbook Test");
  uint32_t tape_panel_id = panel_manager_->add_panel(PanelType::TAPE, "Tape Test");
  uint32_t vp_panel_id = panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "VP Test");

  // Set an active symbol
  panel_manager_->set_active_symbol(10007, "BTC-USDT");

  // Basic verification that the method executes without error
  // More detailed verification would require checking internal panel state
  SUCCEED();
}

TEST_F(IntegrationTest, PanelManagerGridLayout) {
  // Test grid layout functionality
  panel_manager_->set_grid_layout(4, 4);

  // Add some panels
  uint32_t panel1 = panel_manager_->add_panel(PanelType::CHART, "Panel 1", 0, 0, 2, 2);
  uint32_t panel2 = panel_manager_->add_panel(PanelType::HEATMAP, "Panel 2", 2, 0, 2, 2);
  uint32_t panel3 = panel_manager_->add_panel(PanelType::WATCHLIST, "Panel 3", 0, 2, 4, 2);

  EXPECT_NE(panel1, 0);
  EXPECT_NE(panel2, 0);
  EXPECT_NE(panel3, 0);

  // Test auto arrange
  panel_manager_->auto_arrange_panels();

  SUCCEED();
}

TEST_F(IntegrationTest, PanelVisibilityControl) {
  // Add a panel
  uint32_t panel_id = panel_manager_->add_panel(PanelType::CHART, "Visibility Test");
  EXPECT_NE(panel_id, 0);

  // Test visibility control
  panel_manager_->set_panel_visible(panel_id, false);
  // Note: We can't easily verify visibility without accessing internal state
  // This test ensures the method executes without error

  panel_manager_->set_panel_visible(panel_id, true);

  SUCCEED();
}

TEST_F(IntegrationTest, LayoutSerialization) {
  // Add some panels
  panel_manager_->add_panel(PanelType::CHART, "Chart 1", 0, 0, 2, 2);
  panel_manager_->add_panel(PanelType::HEATMAP, "DOM 1", 2, 0, 2, 2);
  panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist 1", 0, 2, 4, 2);

  // Test serialization
  std::string layout_json = panel_manager_->serialize_layout();
  EXPECT_FALSE(layout_json.empty());
  EXPECT_THAT(layout_json, ::testing::HasSubstr("panels"));
  EXPECT_THAT(layout_json, ::testing::HasSubstr("grid"));

  // Test deserialization with a basic layout
  panel_manager_->clear_panels();
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);

  // Deserialize the same layout (this should recreate the panels)
  panel_manager_->deserialize_layout(layout_json);
  // Note: This test might not perfectly recreate due to ID management
  // but it should execute without error

  SUCCEED();
}
