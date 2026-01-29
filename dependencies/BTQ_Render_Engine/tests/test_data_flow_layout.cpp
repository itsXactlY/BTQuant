/**
 * @file test_data_flow_layout.cpp
 * @brief Tests for Phase 4: Data Flow & Layout features
 *
 * This file contains unit and integration tests for:
 * - Unified Data Pipeline (ID: 19)
 * - Global Symbol Switching (ID: 20)
 * - Flexible Panel Layout System (ID: 21)
 * - Panel Add/Remove Functionality (ID: 22)
 */

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

#include "MarketMicrostructureRenderer.h"
#include "components/panel_manager.hpp"
#include "data/ui_data_manager.hpp"
#include "data/unified_data_pipeline.hpp"
#include "hotspine_data_bridge.hpp"
#include "layout/dashboard_layout_manager.hpp"
#include "market_data_processor.hpp"
#include "trading/order_manager.hpp"
#include "trading/position_manager.hpp"
#include "trading/risk_assessment.hpp"

using namespace BTQuant;
using namespace BTQuant::Data;
using namespace BTQuant::Layout;

class DataFlowLayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create mock dependencies
    bridge_ = std::make_shared<HotSpineDataBridge>("/test_hotspine");
    processor_ = std::make_shared<RenderEngine::MarketDataProcessor>();
    order_manager_ = std::make_shared<OrderManager>();
    position_manager_ = std::make_shared<PositionManager>();
    risk_assessment_ = std::make_shared<RiskAssessment>();
    micro_renderer_ = nullptr;  // Not needed for these tests

    // Initialize the components we want to test
    unified_pipeline_ = std::make_shared<UnifiedDataPipeline>(bridge_, processor_, nullptr);
    ui_data_manager_ = std::make_shared<UIDataManager>();
    layout_manager_ = std::make_unique<DashboardLayoutManager>();
    panel_manager_ = std::make_unique<PanelManager>(
        bridge_, processor_, order_manager_, position_manager_, risk_assessment_, micro_renderer_);

    unified_pipeline_->initialize();
  }

  void TearDown() override { unified_pipeline_->shutdown(); }

  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;
  RenderEngine::MarketMicrostructureRenderer* micro_renderer_;

  std::shared_ptr<UnifiedDataPipeline> unified_pipeline_;
  std::shared_ptr<UIDataManager> ui_data_manager_;
  std::unique_ptr<DashboardLayoutManager> layout_manager_;
  std::unique_ptr<PanelManager> panel_manager_;
};

// Test Unified Data Pipeline
TEST_F(DataFlowLayoutTest, UnifiedDataPipelineInitialization) {
  EXPECT_NE(unified_pipeline_, nullptr);
  EXPECT_NE(unified_pipeline_->get_data_bridge(), nullptr);
  EXPECT_NE(unified_pipeline_->get_market_processor(), nullptr);
  EXPECT_NE(unified_pipeline_->get_ui_data_manager(), nullptr);
}

TEST_F(DataFlowLayoutTest, UnifiedDataPipelineSubscribeAndPublish) {
  bool callback_called = false;
  std::string received_symbol;

  // Subscribe to data
  DataSubscription subscription;
  subscription.symbol_id = 0;       // Subscribe to all symbols
  subscription.data_types = {"0"};  // OHLC data type
  subscription.callback = [&](const void* data) {
    callback_called = true;
    received_symbol = "BTCUSDT";
  };

  uint32_t sub_id = unified_pipeline_->subscribe(subscription);
  EXPECT_GT(sub_id, 0);

  // Publish some data
  int dummy_data = 42;
  unified_pipeline_->publish(UnifiedDataPipeline::DataType::OHLC, 123, "BTCUSDT", "binance",
                             &dummy_data, sizeof(dummy_data));

  // Process events
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  unified_pipeline_->process_events();

  EXPECT_TRUE(callback_called);

  // Unsubscribe
  unified_pipeline_->unsubscribe(sub_id);
}

// Test Global Symbol Switching
TEST_F(DataFlowLayoutTest, GlobalSymbolSwitching) {
  std::string new_symbol = "ETHUSDT";
  bool callback_called = false;

  // Register a symbol change callback
  ui_data_manager_->register_symbol_change_callback([&](const std::string& symbol) {
    EXPECT_EQ(symbol, new_symbol);
    callback_called = true;
  });

  // Change the symbol
  ui_data_manager_->set_current_symbol(new_symbol);

  // Verify the symbol was updated
  EXPECT_EQ(ui_data_manager_->get_current_symbol(), new_symbol);
  EXPECT_TRUE(callback_called);
}

TEST_F(DataFlowLayoutTest, GlobalSymbolSwitchingSameSymbol) {
  std::string symbol = "BTCUSDT";

  // Set initial symbol
  ui_data_manager_->set_current_symbol(symbol);
  EXPECT_EQ(ui_data_manager_->get_current_symbol(), symbol);

  // Set the same symbol again (should not trigger callback unnecessarily)
  bool callback_called = false;
  ui_data_manager_->register_symbol_change_callback(
      [&](const std::string& s) { callback_called = true; });

  ui_data_manager_->set_current_symbol(symbol);
  EXPECT_FALSE(callback_called);  // Should not be called since symbol didn't change
}

// Test Flexible Panel Layout System
TEST_F(DataFlowLayoutTest, FlexiblePanelLayoutOperations) {
  // Test creating and managing layouts
  layout_manager_->create_new_layout();

  auto layouts = layout_manager_->get_available_layouts();
  EXPECT_GT(layouts.size(), 0);

  // Test grid dimensions
  layout_manager_->set_grid_dimensions(4, 6);
  auto [cols, rows] = layout_manager_->get_grid_dimensions();
  EXPECT_EQ(cols, 4);
  EXPECT_EQ(rows, 6);

  // Test panel operations
  DashboardLayoutManager::PanelLayout panel;
  panel.panel_id = "test_panel_1";
  panel.panel_name = "Test Panel";
  panel.type = Layout::PanelType::CHART;
  panel.symbol = "BTCUSDT";

  layout_manager_->add_panel_to_layout(panel);

  auto current_layout = layout_manager_->get_current_layout();
  EXPECT_EQ(current_layout.panels.size(), 1);
  EXPECT_EQ(current_layout.panels[0].panel_name, "Test Panel");

  // Test getting panels for specific symbol
  auto symbol_panels = layout_manager_->get_panels_for_symbol("BTCUSDT");
  EXPECT_EQ(symbol_panels.size(), 1);

  // Test updating symbol for all panels
  layout_manager_->update_symbol_for_all_panels("BTCUSDT", "ETHUSDT");
  auto eth_panels = layout_manager_->get_panels_for_symbol("ETHUSDT");
  EXPECT_EQ(eth_panels.size(), 1);
}

TEST_F(DataFlowLayoutTest, PanelLayoutAutoArrange) {
  // Set up a grid
  layout_manager_->set_grid_dimensions(3, 3);

  // Add multiple panels
  for (int i = 0; i < 5; ++i) {
    DashboardLayoutManager::PanelLayout panel;
    panel.panel_id = "panel_" + std::to_string(i);
    panel.panel_name = "Panel " + std::to_string(i);
    panel.type = Layout::PanelType::CHART;
    panel.symbol = "BTCUSDT";

    layout_manager_->add_panel_to_layout(panel);
  }

  // Auto arrange panels
  layout_manager_->auto_arrange_panels();

  auto current_layout = layout_manager_->get_current_layout();
  // Verify panels are arranged in grid
  EXPECT_EQ(current_layout.panels.size(), 5);
}

// Test Panel Add/Remove Functionality
TEST_F(DataFlowLayoutTest, PanelAddRemoveFunctionality) {
  // Register callbacks to verify add/remove operations
  int panels_added = 0;
  int panels_removed = 0;

  panel_manager_->register_panel_added_callback(
      [&](uint32_t id, BTQuant::PanelType type) { panels_added++; });

  panel_manager_->register_panel_removed_callback([&](uint32_t id) { panels_removed++; });

  // Add a panel
  uint32_t panel_id =
      panel_manager_->add_panel(BTQuant::PanelType::CHART, "Test Chart", 0, 0, 2, 2);
  EXPECT_GT(panel_id, 0);
  EXPECT_EQ(panels_added, 1);

  // Verify panel count
  EXPECT_EQ(panel_manager_->get_panel_count(), 1);

  // Get all panel IDs
  auto all_ids = panel_manager_->get_all_panel_ids();
  EXPECT_EQ(all_ids.size(), 1);
  EXPECT_EQ(all_ids[0], panel_id);

  // Get panel config
  auto config = panel_manager_->get_panel_config(panel_id);
  EXPECT_EQ(config.title, "Test Chart");

  // Update panel config
  PanelConfig new_config = config;
  new_config.title = "Updated Chart";
  panel_manager_->update_panel_config(panel_id, new_config);

  // Verify update
  auto updated_config = panel_manager_->get_panel_config(panel_id);
  EXPECT_EQ(updated_config.title, "Updated Chart");

  // Remove the panel
  panel_manager_->remove_panel(panel_id);
  EXPECT_EQ(panels_removed, 1);

  // Verify panel count after removal
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
}

TEST_F(DataFlowLayoutTest, PanelWithSymbolFunctionality) {
  // Add a panel with a specific symbol
  uint32_t panel_id = panel_manager_->add_panel_with_symbol(BTQuant::PanelType::CHART, "BTC Chart",
                                                            "BTCUSDT", 0, 0, 2, 2);
  EXPECT_GT(panel_id, 0);

  // Verify panel was added
  EXPECT_EQ(panel_manager_->get_panel_count(), 1);

  // Update the panel's symbol
  panel_manager_->set_panel_symbol(panel_id, "ETHUSDT");

  // Remove the panel
  panel_manager_->remove_panel(panel_id);
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
}

TEST_F(DataFlowLayoutTest, MultiplePanelsManagement) {
  std::vector<uint32_t> panel_ids;

  // Add multiple panels
  for (int i = 0; i < 3; ++i) {
    uint32_t id = panel_manager_->add_panel(BTQuant::PanelType::ORDERBOOK,
                                            "Orderbook " + std::to_string(i), i, 0, 1, 1);
    panel_ids.push_back(id);
    EXPECT_GT(id, 0);
  }

  EXPECT_EQ(panel_manager_->get_panel_count(), 3);

  // Remove one panel
  panel_manager_->remove_panel(panel_ids[1]);
  EXPECT_EQ(panel_manager_->get_panel_count(), 2);

  // Clear all panels
  panel_manager_->clear_panels();
  EXPECT_EQ(panel_manager_->get_panel_count(), 0);
}

// Integration test: Verify all components work together
TEST_F(DataFlowLayoutTest, DataFlowLayoutIntegration) {
  // Set up symbol switching
  std::string current_symbol = "BTCUSDT";
  ui_data_manager_->set_current_symbol(current_symbol);

  // Add a panel
  uint32_t panel_id =
      panel_manager_->add_panel(BTQuant::PanelType::CHART, "Main Chart", 0, 0, 2, 2);
  EXPECT_GT(panel_id, 0);

  // Update the panel's symbol
  panel_manager_->set_panel_symbol(panel_id, "ETHUSDT");

  // Verify the panel exists
  EXPECT_EQ(panel_manager_->get_panel_count(), 1);

  // Test layout management
  DashboardLayoutManager::PanelLayout layout_panel;
  layout_panel.panel_id = "chart_panel";
  layout_panel.panel_name = "Chart Panel";
  layout_panel.type = Layout::PanelType::CHART;
  layout_panel.symbol = "ETHUSDT";

  layout_manager_->add_panel_to_layout(layout_panel);

  // Verify layout has the panel
  auto current_layout = layout_manager_->get_current_layout();
  EXPECT_EQ(current_layout.panels.size(), 1);
  EXPECT_EQ(current_layout.panels[0].symbol, "ETHUSDT");

  // Clean up
  panel_manager_->remove_panel(panel_id);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}