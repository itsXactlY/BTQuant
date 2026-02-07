#pragma once

#include <imgui.h>
#include <memory>
#include <vector>
#include <string>

#include "panel_base.hpp"

// Forward declaration to avoid circular dependency
namespace BTQuant {
    class PanelManager;
}

namespace BTQuant {

/**
 * TabbedPanel - Container panel that holds multiple child panels in tabs
 *
 * This panel allows combining multiple panels into a single window with tabs
 * at the bottom, enabling users to switch between different views within the same space.
 */
class TabbedPanel : public PanelBase {
 public:
  TabbedPanel(const PanelConfig& config, PanelManager* panel_manager);

  ~TabbedPanel() override;

  void render() override;
  void update(float dt) override;

  // Tab management
  bool add_panel_to_tab(uint32_t panel_id);
  bool remove_panel_from_tab(uint32_t panel_id);
  bool set_active_tab(uint32_t panel_id);
  uint32_t get_active_tab() const { return active_tab_id_; }

  // Get the number of tabs in this panel
  size_t get_tab_count() const { return tabbed_panel_ids_.size(); }

  // Get all panel IDs in tabs
  const std::vector<uint32_t>& get_tabbed_panels() const { return tabbed_panel_ids_; }

  // PanelBase overrides
  PanelSettingsInterface* get_settings_interface() override { return nullptr; }
  void open_settings() override {}

 private:
  PanelManager* panel_manager_;
  std::vector<uint32_t> tabbed_panel_ids_;  // IDs of panels contained in tabs
  uint32_t active_tab_id_ = 0;              // ID of currently active tab
  std::string tab_bar_id_;                  // Unique ID for the tab bar

  void render_tab_bar();
  void render_active_tab_content();
};

}  // namespace BTQuant