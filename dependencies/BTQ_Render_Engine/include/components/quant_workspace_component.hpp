#pragma once

#include <atomic>
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

// Global crosshair price - shared across all chart panels for synchronized horizontal line
extern std::atomic<double> g_crosshair_price;

namespace BTQuant {

class QuantWorkspaceComponent : public UIComponent {
 public:
  explicit QuantWorkspaceComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  PanelManager* getPanelManager() { return panel_manager_.get(); }

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

  // Crosshair synchronization state
  bool global_crosshair_enabled_ = true;  // Enable/disable global crosshair sync
  ImVec2 last_crosshair_position_{0, 0};  // Last recorded crosshair position
  bool crosshair_active_ = false;         // Whether crosshair is currently active

  // UI rendering methods
  void render_dashboard_controls();
  void render_orders_panel();
  void render_positions_panel();
  
  // Crosshair synchronization methods
  void handle_global_crosshair_sync();
  ChartPanel* get_chart_panel_under_cursor() const;

};

}  // namespace BTQuant
