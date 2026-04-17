#include "components/tabbed_panel.hpp"
#include "components/panel_manager.hpp"

#include <imgui.h>
#include <algorithm>

namespace BTQuant {

TabbedPanel::TabbedPanel(const PanelConfig& config, PanelManager* panel_manager)
    : PanelBase(config), panel_manager_(panel_manager), active_tab_index_(0) {}

void TabbedPanel::render() {
    if (tabbed_panels_.empty()) {
        ImGui::Text("No panels in tab group");
        return;
    }

    render_tab_bar();
    render_active_tab_content();
}

void TabbedPanel::add_panel(uint32_t panel_id) {
    // Avoid duplicates
    if (std::find(tabbed_panels_.begin(), tabbed_panels_.end(), panel_id)
        == tabbed_panels_.end()) {
        tabbed_panels_.push_back(panel_id);
    }
}

void TabbedPanel::remove_panel(uint32_t panel_id) {
    auto it = std::find(tabbed_panels_.begin(), tabbed_panels_.end(), panel_id);
    if (it != tabbed_panels_.end()) {
        tabbed_panels_.erase(it);
        if (active_tab_index_ >= static_cast<int>(tabbed_panels_.size())) {
            active_tab_index_ = std::max(0, static_cast<int>(tabbed_panels_.size()) - 1);
        }
    }
}

void TabbedPanel::set_active_tab(int tab_index) {
    if (tab_index >= 0 && tab_index < static_cast<int>(tabbed_panels_.size())) {
        active_tab_index_ = tab_index;
    }
}

bool TabbedPanel::handle_drop(uint32_t source_panel_id) {
    add_panel(source_panel_id);
    return true;
}

void TabbedPanel::render_tab_bar() {
    if (ImGui::BeginTabBar("TabbedPanelBar")) {
        for (int i = 0; i < static_cast<int>(tabbed_panels_.size()); ++i) {
            bool open = true;
            std::string tab_label = "Panel " + std::to_string(tabbed_panels_[i]);

            if (ImGui::BeginTabItem(tab_label.c_str(), &open)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }

            if (!open) {
                tab_to_close_ = i;
                should_close_tab_ = true;
            }
        }

        if (should_close_tab_ && tab_to_close_ >= 0) {
            remove_panel(tabbed_panels_[tab_to_close_]);
            should_close_tab_ = false;
            tab_to_close_ = -1;
        }

        ImGui::EndTabBar();
    }
}

void TabbedPanel::render_active_tab_content() {
    // The actual rendering of child panels is handled by PanelManager
    // This just provides a placeholder region
    if (active_tab_index_ >= 0 &&
        active_tab_index_ < static_cast<int>(tabbed_panels_.size())) {
        ImGui::Text("Tab content rendered by PanelManager for panel ID: %u",
                     tabbed_panels_[active_tab_index_]);
    }
}

void TabbedPanel::handle_tab_context_menu(int tab_index) {
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Close Tab")) {
            tab_to_close_ = tab_index;
            should_close_tab_ = true;
        }
        ImGui::EndPopup();
    }
}

}  // namespace BTQuant
