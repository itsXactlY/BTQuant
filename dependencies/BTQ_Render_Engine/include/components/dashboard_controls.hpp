#pragma once

#include "../ui/ui_base.hpp"
#include "../vulkan_base_types.hpp"
#include "../market_data_processor.hpp"  // For TimeFrame enum
#include "global_alert_manager.hpp"      // For GlobalAlertManager

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace BTQuant {

// Forward declarations
class PanelManager;

class DashboardControls : public UIComponent {
 public:
  explicit DashboardControls(PanelManager* panel_manager);
  virtual ~DashboardControls() = default;

  void update(float dt) override;
  void render_gui() override;
  void initialize_vulkan_resources(VulkanCore* core) override;
  void clear_data() override;

  // Specific method to render dashboard controls
  void render_dashboard_controls();

  // Access to global alert manager
  std::shared_ptr<GlobalAlertManager> get_global_alert_manager() { return global_alert_manager_; }

 private:
  PanelManager* panel_manager_;
  
  // Global Alert Manager
  std::shared_ptr<GlobalAlertManager> global_alert_manager_;

  // Exchange selection state
  std::vector<std::string> all_exchanges_;         // All available exchanges
  std::vector<int> selected_exchanges_;            // Track which exchanges are selected (using int instead of bool due to vector<bool> issues)
  bool exchanges_loaded_ = false;                  // Flag to indicate if exchanges are loaded

  // Symbol selection state
  std::array<char, 256> symbol_input_buffer_ = {};  // For search input
  std::vector<std::string> filtered_symbols_;  // Filtered symbols for search
  std::vector<std::string> all_symbols_;       // All available symbols
  int selected_symbol_idx_ = -1;              // Selected symbol index
  bool symbols_loaded_ = false;               // Flag to indicate if symbols are loaded
  bool needs_refresh_ = true;                 // Flag to indicate refresh needed
  bool fetch_symbols_from_api_ = false;       // Flag to indicate fetching from exchange API

  // Methods for exchange API integration
  void fetch_symbols_from_exchange_api();

  // Helper method to check if an exchange is selected
  bool is_exchange_selected(const std::string& exchange_name) const;

  // Helper method to refresh symbols based on selected exchanges
  void refresh_symbols_for_selected_exchanges();

  // Timeframe selection state
  RenderEngine::TimeFrame current_timeframe_ = RenderEngine::TimeFrame::TF_1MIN;

  // Method to update all chart panels with the new timeframe
  void update_all_chart_timeframes(RenderEngine::TimeFrame timeframe);

  // Method to sync symbol to all panels
  void sync_symbol_to_all_panels(uint32_t symbol_id, const std::string& symbol);
  
  // Methods for managing global alerts
  void render_global_alerts_section();
  void render_create_alert_modal();
  void create_alert_from_modal();
  
  // UI state for alert creation modal
  bool show_create_alert_modal_ = false;
  char new_alert_name_[64] = "";
  char new_alert_symbol_[32] = "";
  char new_alert_threshold_[32] = "";
  int new_alert_type_ = 0;  // 0: Price Above, 1: Price Below, 2: Volume Above, 3: Volume Below
};

}  // namespace BTQuant