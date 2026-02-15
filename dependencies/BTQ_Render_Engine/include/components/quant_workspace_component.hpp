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
#include "symbol_selector.hpp"

namespace BTQuant {

// Global crosshair synchronization data
struct GlobalCrosshair {
    std::atomic<double> price{0.0};
    std::atomic<uint64_t> time{0};
    std::atomic<bool> active{false};
    
    GlobalCrosshair() = default;
    GlobalCrosshair(double p, uint64_t t, bool a) : price(p), time(t), active(a) {}
};

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

  // Public access to global crosshair
  static GlobalCrosshair g_crosshair;

  // Additional global crosshair price variable
  static std::atomic<double> g_crosshair_price;
  
  // Additional global crosshair time variable
  static std::atomic<uint64_t> g_crosshair_time;

  // Helper methods for accessing global crosshair state
  static double get_global_crosshair_price() { return g_crosshair.price.load(); }
  static uint64_t get_global_crosshair_time() { return g_crosshair.time.load(); }
  static bool get_global_crosshair_active() { return g_crosshair.active.load(); }
  static void set_global_crosshair_price(double price) { g_crosshair.price.store(price); }
  static void set_global_crosshair_time(uint64_t time) { g_crosshair.time.store(time); }
  static void set_global_crosshair_active(bool active) { g_crosshair.active.store(active); }

  // Global atomic active symbol ID for cross-panel synchronization
  static std::atomic<uint32_t> g_active_symbol_id;

  // Helper methods for accessing global active symbol ID
  static uint32_t get_global_active_symbol_id() { return g_active_symbol_id.load(); }
  static void set_global_active_symbol_id(uint32_t symbol_id) { g_active_symbol_id.store(symbol_id); }

  // Global SymbolSelector instance
  static BTQuant::SymbolSelector g_symbol_selector;
  static BTQuant::SymbolSelectorState g_symbol_selector_state;

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
