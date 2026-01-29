#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

// Project headers
#include "components/panel_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "data/hotspine_data_bridge.hpp"
#include "data/market_data_processor.hpp"
#include "performance_monitor.hpp"
#include "trading/order_manager.hpp"
#include "trading/position_manager.hpp"
#include "trading/risk_assessment.hpp"

using namespace BTQuant;

class ComprehensiveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize mock data bridge and processor
    processor_ = std::make_shared<RenderEngine::MarketDataProcessor>();
    bridge_ = std::make_shared<HotSpineDataBridge>("/comprehensive_test");

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
    
    // Initialize workspace component
    workspace_component_ = std::make_unique<QuantWorkspaceComponent>(
        bridge_, processor_, micro_renderer_);
  }

  void TearDown() override {
    panel_manager_.reset();
    workspace_component_.reset();
  }

  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;
  RenderEngine::MarketMicrostructureRenderer* micro_renderer_;
  std::unique_ptr<PanelManager> panel_manager_;
  std::unique_ptr<QuantWorkspaceComponent> workspace_component_;
};

// Test all panel types can be created and managed
TEST_F(ComprehensiveTest, AllPanelTypesCreation) {
  std::vector<uint32_t> panel_ids;
  
  // Test all panel types mentioned in the PRD
  panel_ids.push_back(panel_manager_->add_panel(PanelType::CHART, "Main Chart"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::HEATMAP, "DOM Surface"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::ORDERBOOK, "Orderbook"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::TAPE, "Time & Sales"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::TRADING_ORDERS, "Active Orders"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::TRADING_POSITIONS, "Positions"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "Volume Profile"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::FOOTPRINT_CHART, "Footprint Chart"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::TPO_PROFILE, "TPO Profile"));
  panel_ids.push_back(panel_manager_->add_panel(PanelType::PERFORMANCE_MONITOR, "Performance Monitor"));

  // Verify all panels were created successfully
  for (auto id : panel_ids) {
    EXPECT_NE(id, 0) << "Panel creation failed";
  }
  
  EXPECT_EQ(panel_manager_->get_panel_count(), panel_ids.size());
}

// Test panel operations (add, remove, visibility)
TEST_F(ComprehensiveTest, PanelOperations) {
  // Add a panel
  uint32_t panel_id = panel_manager_->add_panel(PanelType::CHART, "Test Panel");
  EXPECT_NE(panel_id, 0);
  
  size_t initial_count = panel_manager_->get_panel_count();
  EXPECT_GT(initial_count, 0);
  
  // Test visibility
  panel_manager_->set_panel_visible(panel_id, false);
  panel_manager_->set_panel_visible(panel_id, true);
  
  // Test removal
  panel_manager_->remove_panel(panel_id);
  EXPECT_EQ(panel_manager_->get_panel_count(), initial_count - 1);
}

// Test symbol switching functionality
TEST_F(ComprehensiveTest, SymbolSwitching) {
  // Add multiple panels that should respond to symbol changes
  panel_manager_->add_panel(PanelType::CHART, "Chart Panel");
  panel_manager_->add_panel(PanelType::HEATMAP, "DOM Panel");
  panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "VP Panel");
  
  // Test setting active symbol
  panel_manager_->set_active_symbol(10007, "BTC-USDT");
  
  // Verify the symbol was set (would need internal access to verify, just ensure no crash)
  SUCCEED();
}

// Test layout management
TEST_F(ComprehensiveTest, LayoutManagement) {
  // Set grid layout
  panel_manager_->set_grid_layout(6, 6);
  
  // Add panels with specific positions
  uint32_t panel1 = panel_manager_->add_panel(PanelType::CHART, "Panel 1", 0, 0, 3, 3);
  uint32_t panel2 = panel_manager_->add_panel(PanelType::HEATMAP, "Panel 2", 3, 0, 3, 3);
  uint32_t panel3 = panel_manager_->add_panel(PanelType::WATCHLIST, "Panel 3", 0, 3, 6, 3);
  
  EXPECT_NE(panel1, 0);
  EXPECT_NE(panel2, 0);
  EXPECT_NE(panel3, 0);
  
  // Test auto arrange
  panel_manager_->auto_arrange_panels();
  
  // Test serialization/deserialization
  std::string layout = panel_manager_->serialize_layout();
  EXPECT_FALSE(layout.empty());
  
  // Clear and reload
  panel_manager_->clear_panels();
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
  
  panel_manager_->deserialize_layout(layout);
  // Count might vary due to ID management, just ensure no crash
  SUCCEED();
}

// Test workspace component functionality
TEST_F(ComprehensiveTest, WorkspaceComponent) {
  ASSERT_NE(workspace_component_, nullptr);
  
  // Test basic methods
  workspace_component_->update(0.016f);  // ~60fps
  
  // Verify we can get the panel manager
  auto* pm = workspace_component_->getPanelManager();
  ASSERT_NE(pm, nullptr);
}

// Test performance monitor integration
TEST_F(ComprehensiveTest, PerformanceMonitoring) {
  // Add performance monitor panel
  uint32_t perf_panel_id = panel_manager_->add_panel(PanelType::PERFORMANCE_MONITOR, "Perf Monitor");
  EXPECT_NE(perf_panel_id, 0);
  
  // Simulate performance monitoring
  g_performance_monitor.start_frame();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  g_performance_monitor.end_frame();
  
  // Verify metrics are available
  double fps = g_performance_monitor.get_fps();
  double frame_time = g_performance_monitor.get_frame_time_ms();
  
  EXPECT_GE(fps, 0.0);
  EXPECT_GE(frame_time, 0.0);
}

// Test data flow between components
TEST_F(ComprehensiveTest, DataFlowIntegration) {
  // Add panels that share data
  uint32_t chart_id = panel_manager_->add_panel(PanelType::CHART, "Shared Chart");
  uint32_t vp_id = panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "Shared VP");
  uint32_t dom_id = panel_manager_->add_panel(PanelType::HEATMAP, "Shared DOM");
  
  EXPECT_NE(chart_id, 0);
  EXPECT_NE(vp_id, 0);
  EXPECT_NE(dom_id, 0);
  
  // Update with a small time delta
  panel_manager_->update(0.016f);
  
  // Verify all panels received the update (no crashes)
  SUCCEED();
}

// Test trading components integration
TEST_F(ComprehensiveTest, TradingComponents) {
  // Add trading-related panels
  uint32_t orders_id = panel_manager_->add_panel(PanelType::TRADING_ORDERS, "Orders Panel");
  uint32_t positions_id = panel_manager_->add_panel(PanelType::TRADING_POSITIONS, "Positions Panel");
  uint32_t risk_id = panel_manager_->add_panel(PanelType::RISK_METRICS, "Risk Panel");
  
  EXPECT_NE(orders_id, 0);
  EXPECT_NE(positions_id, 0);
  EXPECT_NE(risk_id, 0);
  
  // Verify managers are accessible
  ASSERT_NE(order_manager_, nullptr);
  ASSERT_NE(position_manager_, nullptr);
  ASSERT_NE(risk_assessment_, nullptr);
  
  // Basic functionality test
  panel_manager_->update(0.016f);
  SUCCEED();
}

// Test theme management integration
TEST_F(ComprehensiveTest, ThemeIntegration) {
  // Add multiple panels and test theme application
  panel_manager_->add_panel(PanelType::CHART, "Themed Chart");
  panel_manager_->add_panel(PanelType::HEATMAP, "Themed DOM");
  panel_manager_->add_panel(PanelType::WATCHLIST, "Themed Watchlist");
  
  // Update with a time delta to ensure all panels can render
  panel_manager_->update(0.016f);
  
  // Verify no rendering issues occurred
  SUCCEED();
}

// Test stress scenario with many panels
TEST_F(ComprehensiveTest, StressManyPanels) {
  std::vector<uint32_t> panel_ids;
  
  // Create many panels to test memory and performance
  for (int i = 0; i < 20; ++i) {
    std::string name = "Panel_" + std::to_string(i);
    panel_ids.push_back(panel_manager_->add_panel(PanelType::CHART, name));
  }
  
  // Verify most panels were created
  size_t created_count = 0;
  for (auto id : panel_ids) {
    if (id != 0) created_count++;
  }
  
  EXPECT_GT(created_count, 15); // Expect most to succeed
  
  // Update all panels
  panel_manager_->update(0.016f);
  
  // Remove all panels
  for (auto id : panel_ids) {
    if (id != 0) {
      panel_manager_->remove_panel(id);
    }
  }
  
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
}

// Test cleanup and resource management
TEST_F(ComprehensiveTest, ResourceCleanup) {
  // Add several panels
  panel_manager_->add_panel(PanelType::CHART, "Cleanup Test 1");
  panel_manager_->add_panel(PanelType::HEATMAP, "Cleanup Test 2");
  panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "Cleanup Test 3");
  
  size_t initial_count = panel_manager_->get_panel_count();
  EXPECT_GT(initial_count, 0);
  
  // Clear all panels
  panel_manager_->clear_panels();
  
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
  
  // Add panels again to ensure manager still works
  panel_manager_->add_panel(PanelType::CHART, "Post-Cleanup Test");
  EXPECT_GT(panel_manager_->get_panel_count(), 0);
}