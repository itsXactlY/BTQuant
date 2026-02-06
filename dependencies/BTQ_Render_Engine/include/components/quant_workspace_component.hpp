#pragma once

#include <memory>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../symbol_registry.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "hierarchical_selector.hpp"
#include "imgui.h"
#include "implot.h"
#include "panel_manager.hpp"

namespace BTQuant {

class QuantWorkspaceComponent : public UIComponent {
 public:
  explicit QuantWorkspaceComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
      RenderEngine::MarketMicrostructureRenderer* micro_renderer = nullptr);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  PanelManager* getPanelManager() { return panel_manager_.get(); }

  // Pass-through methods to expose state to the Dashboard
  OrderManager* getOrderManager() const { return order_manager_.get(); }
  PositionManager* getPositionManager() const { return position_manager_.get(); }
  RiskAssessment* getRiskAssessment() const { return risk_assessment_.get(); }
  HotSpineDataBridge* getDataBridge() const { return bridge_.get(); }
  RenderEngine::MarketDataProcessor* getMarketDataProcessor() const { return processor_.get(); }
  
  // UI state getters
  const std::string& getSelectedSymbol() const { return selected_symbol_; }
  double getOrderQuantity() const { return order_quantity_; }
  double getOrderPrice() const { return order_price_; }
  int getSelectedOrderSide() const { return selected_order_side_; }  // 0 = Buy, 1 = Sell
  int getSelectedOrderType() const { return selected_order_type_; }  // 0 = Market, 1 = Limit
  bool getShowDashboardControls() const { return show_dashboard_controls_; }
  
  // Selector state getters
  const HierarchicalSelectorState& getSelectorState() const { return selector_state_; }
  const HierarchicalSelector& getHierarchicalSelector() const { return hierarchical_selector_; }

  void initialize_vulkan_resources(VulkanCore* core) override;
  void clear_data() override;

  void refresh_hierarchical_selector();  // Made public for workspace manager

 private:
  // Core systems
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  // Trading Systems
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;

  // New panel-based UI system
  std::unique_ptr<PanelManager> panel_manager_;

  // Hierarchical selector for Exchange -> Symbol -> Chart
  HierarchicalSelector hierarchical_selector_;
  HierarchicalSelectorState selector_state_;

  // UI state
  bool show_dashboard_controls_ = true;

  // Order input state
  double order_quantity_ = 1.0;
  double order_price_ = 0.0;
  std::string selected_symbol_ = "BTC/USDT";
  int selected_order_side_ = 0;  // 0 = Buy, 1 = Sell
  int selected_order_type_ = 0;  // 0 = Market, 1 = Limit
  std::vector<std::string> order_sides_ = {"Buy", "Sell"};
  std::vector<std::string> order_types_ = {"Market", "Limit"};

  // UI rendering methods
  void render_dashboard_controls();
  void render_orders_panel();
  void render_positions_panel();

};

}  // namespace BTQuant
