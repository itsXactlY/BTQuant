#pragma once

#include "../ui/ui_base.hpp"
#include "../vulkan_base_types.hpp"

#include <memory>
#include <string>
#include <vector>
#include <string>

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

 private:
  PanelManager* panel_manager_;

  // Symbol selection state
  std::string symbol_input_buffer_ = "";  // For search input
  std::vector<std::string> filtered_symbols_;  // Filtered symbols for search
  std::vector<std::string> all_symbols_;       // All available symbols
  int selected_symbol_idx_ = -1;              // Selected symbol index
  bool symbols_loaded_ = false;               // Flag to indicate if symbols are loaded
  bool needs_refresh_ = true;                 // Flag to indicate refresh needed
};

}  // namespace BTQuant