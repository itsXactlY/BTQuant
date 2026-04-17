#pragma once

#include <imgui.h>
#include <vector>
#include <memory>
#include <string>

#include "panel_base.hpp"

// Forward declaration to avoid circular dependency
namespace BTQuant {
    class PanelManager;
}

#include "panel_manager.hpp"

namespace BTQuant {

/**
 * TabbedPanel - A panel that hosts multiple child panels in tabs
 * 
 * This panel allows multiple panels to be grouped together in a tabbed interface,
 * enabling users to organize their workspace more efficiently.
 */
class TabbedPanel : public PanelBase {
public:
    TabbedPanel(const PanelConfig& config, PanelManager* panel_manager);
    
    void render() override;
    void add_panel(uint32_t panel_id);
    void remove_panel(uint32_t panel_id);
    void set_active_tab(int tab_index);
    
    // Accessor methods
    std::vector<uint32_t>& get_tabbed_panels() { return tabbed_panels_; }
    const std::vector<uint32_t>& get_tabbed_panels() const { return tabbed_panels_; }
    int get_active_tab() const { return active_tab_index_; }

    // Override drag and drop methods
    bool can_accept_drop() const override { return true; }  // Tabbed panels can accept dropped panels
    bool handle_drop(uint32_t source_panel_id) override;    // Handle a dropped panel

private:
    PanelManager* panel_manager_;
    std::vector<uint32_t> tabbed_panels_;  // IDs of panels in this tabbed panel
    int active_tab_index_ = 0;             // Index of currently active tab
    bool should_close_tab_ = false;        // Flag to indicate if a tab should be closed
    int tab_to_close_ = -1;                // Index of tab to close
    
    void render_tab_bar();
    void render_active_tab_content();
    void handle_tab_context_menu(int tab_index);
};

} // namespace BTQuant