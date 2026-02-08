#include "../../include/components/panel_base.hpp"

#include "components/panel_manager.hpp"
#include "imgui.h"
#include "ui/context_menus.hpp"
#include "ui/haptic_feedback.hpp"
#include "ui/tooltips.hpp"

namespace BTQuant {

void PanelBase::begin_panel_window() {
  // Check if we're in a valid ImGui frame scope to prevent assertion errors
  // We can check this by attempting to get the current context and checking if it's valid
  ImGuiContext* g = ImGui::GetCurrentContext();
  if (g == nullptr) {
    // If there's no valid ImGui context, skip rendering this frame
    return;
  }

  // In newer versions of ImGui, we can't directly access WithinFrameScope
  // Instead, we'll just check if the context is valid and proceed with rendering
  // If we're not in a proper frame, ImGui will handle the error internally

  ImGui::SetNextWindowPos(config_.position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(config_.size, ImGuiCond_FirstUseEver);

  ImGuiWindowFlags flags = ImGuiWindowFlags_None;
  if (!config_.resizable) flags |= ImGuiWindowFlags_NoResize;
  if (!config_.movable) flags |= ImGuiWindowFlags_NoMove;

  // Use ThemeManager for glass style
  push_glass_style();

  // Ensure unique ID for the window
  std::string window_title =
      config_.title + "###panel_" + std::to_string(reinterpret_cast<uintptr_t>(this));

  ImGui::Begin(window_title.c_str(), &config_.visible, flags);
}

void PanelBase::end_panel_window() {
  ImGui::End();
  pop_glass_style();
}

void PanelBase::push_glass_style() { ThemeManager::getInstance().pushGlassStyle(); }

void PanelBase::pop_glass_style() { ThemeManager::getInstance().popGlassStyle(); }

void PanelBase::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Default content for placeholder panels
  ImGui::Text("Panel Type: %s", get_panel_type_name(config_.type));
  ImGui::Text("Implementation coming soon...");

  end_panel_window();

  // Render settings modal if available
  if (auto* settings = get_settings_interface()) {
    settings->render();
  }
}

void PanelBase::render_symbol_link_icon() {
  if (symbol_link_color_ == SymbolLinkGroupColor::NONE) {
    // Render an inactive/unlinked icon (gray circle)
    ImVec2 cursor_pos = ImGui::GetCursorPos();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 0.7f));  // Gray
    ImGui::Text("●");  // Circle icon representing unlinked state
    ImGui::PopStyleColor();

    // Add tooltip
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("Click to link this panel to others");
      ImGui::EndTooltip();
    }

    // Handle click to create a new link group
    if (ImGui::IsItemClicked()) {
      // Show a popup menu to select a color for the new link group
      ImGui::OpenPopup("LinkGroupColorPopup");
    }
  } else {
    // Determine color based on link group
    ImVec4 color;
    switch (symbol_link_color_) {
      case SymbolLinkGroupColor::RED:
        color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red
        break;
      case SymbolLinkGroupColor::GREEN:
        color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);  // Green
        break;
      case SymbolLinkGroupColor::BLUE:
        color = ImVec4(0.3f, 0.6f, 1.0f, 1.0f);  // Blue
        break;
      default:
        color = ImVec4(0.5f, 0.5f, 0.5f, 0.7f);  // Gray (shouldn't happen)
        break;
    }

    // Render the link icon with the appropriate color
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("●");  // Colored circle icon representing linked state
    ImGui::PopStyleColor();

    // Add tooltip showing the link status
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      const char* color_name;
      switch (symbol_link_color_) {
        case SymbolLinkGroupColor::RED:
          color_name = "Red";
          break;
        case SymbolLinkGroupColor::GREEN:
          color_name = "Green";
          break;
        case SymbolLinkGroupColor::BLUE:
          color_name = "Blue";
          break;
        default:
          color_name = "Unknown";
          break;
      }
      ImGui::Text("Linked to %s group - Symbols sync", color_name);
      ImGui::EndTooltip();
    }

    // Handle click to remove from link group
    if (ImGui::IsItemClicked()) {
      // Show a confirmation popup to unlink the panel
      ImGui::OpenPopup("UnlinkPanelPopup");
    }
  }

  // Popup for selecting a color when creating a new link group
  if (ImGui::BeginPopup("LinkGroupColorPopup")) {
    ImGui::Text("Select link color:");
    if (ImGui::Selectable("Red")) {
      if (panel_manager_) {
        // Check if this panel is already in a link group and remove it first
        uint32_t current_group_id = get_symbol_link_group_id();
        if (current_group_id != 0) {
          panel_manager_->remove_panel_from_symbol_link_group(current_group_id, get_panel_id());
        }

        // Find or create a red link group and add this panel to it
        uint32_t group_id = 0;
        for (const auto& [id, group] : panel_manager_->get_symbol_link_groups()) {
          if (group->color == SymbolLinkGroupColor::RED) {
            group_id = id;
            break;
          }
        }

        if (group_id == 0) {
          group_id = panel_manager_->create_symbol_link_group(SymbolLinkGroupColor::RED);
        }

        panel_manager_->add_panel_to_symbol_link_group(group_id, get_panel_id());
        set_symbol_link_group_id(group_id);
        set_symbol_link_color(SymbolLinkGroupColor::RED);
      }
      ImGui::CloseCurrentPopup();
    }
    if (ImGui::Selectable("Green")) {
      if (panel_manager_) {
        // Check if this panel is already in a link group and remove it first
        uint32_t current_group_id = get_symbol_link_group_id();
        if (current_group_id != 0) {
          panel_manager_->remove_panel_from_symbol_link_group(current_group_id, get_panel_id());
        }

        // Find or create a green link group and add this panel to it
        uint32_t group_id = 0;
        for (const auto& [id, group] : panel_manager_->get_symbol_link_groups()) {
          if (group->color == SymbolLinkGroupColor::GREEN) {
            group_id = id;
            break;
          }
        }

        if (group_id == 0) {
          group_id = panel_manager_->create_symbol_link_group(SymbolLinkGroupColor::GREEN);
        }

        panel_manager_->add_panel_to_symbol_link_group(group_id, get_panel_id());
        set_symbol_link_group_id(group_id);
        set_symbol_link_color(SymbolLinkGroupColor::GREEN);
      }
      ImGui::CloseCurrentPopup();
    }
    if (ImGui::Selectable("Blue")) {
      if (panel_manager_) {
        // Check if this panel is already in a link group and remove it first
        uint32_t current_group_id = get_symbol_link_group_id();
        if (current_group_id != 0) {
          panel_manager_->remove_panel_from_symbol_link_group(current_group_id, get_panel_id());
        }

        // Find or create a blue link group and add this panel to it
        uint32_t group_id = 0;
        for (const auto& [id, group] : panel_manager_->get_symbol_link_groups()) {
          if (group->color == SymbolLinkGroupColor::BLUE) {
            group_id = id;
            break;
          }
        }

        if (group_id == 0) {
          group_id = panel_manager_->create_symbol_link_group(SymbolLinkGroupColor::BLUE);
        }

        panel_manager_->add_panel_to_symbol_link_group(group_id, get_panel_id());
        set_symbol_link_group_id(group_id);
        set_symbol_link_color(SymbolLinkGroupColor::BLUE);
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  // Popup for confirming unlinking
  if (ImGui::BeginPopup("UnlinkPanelPopup")) {
    ImGui::Text("Remove from link group?");
    if (ImGui::Button("Yes")) {
      if (panel_manager_) {
        uint32_t current_group_id = get_symbol_link_group_id();
        if (current_group_id != 0) {
          panel_manager_->remove_panel_from_symbol_link_group(current_group_id, get_panel_id());
          set_symbol_link_group_id(0);
          set_symbol_link_color(SymbolLinkGroupColor::NONE);
        }
      }
      ImGui::CloseCurrentPopup();
      // Add haptic feedback for button interaction
      BTQuant::UI::HapticFeedback::getInstance().triggerForImportantInteraction();
    }
    // Show standardized tooltip for the button
    BTQuant::UI::show_control_tooltip("confirmation_dialog_yes");

    ImGui::SameLine();
    if (ImGui::Button("No")) {
      ImGui::CloseCurrentPopup();
      // Add haptic feedback for button interaction
      BTQuant::UI::HapticFeedback::getInstance().triggerForSubtleInteraction();
    }
    // Show standardized tooltip for the button
    BTQuant::UI::show_control_tooltip("confirmation_dialog_no");
    ImGui::EndPopup();
  }
}

void PanelBase::render_panel_header() {
  // Panel type indicator
  const char* type_name = get_panel_type_name(config_.type);
  ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "[%s]", type_name);

  ImGui::SameLine();

  // Render the symbol link icon before the title
  render_symbol_link_icon();
  ImGui::SameLine();

  ImGui::Text("%s", config_.title.c_str());

  // Settings button (left of close button)
  if (get_settings_interface() != nullptr) {
    float button_size = ImGui::GetTextLineHeight();
    ImGui::SameLine(ImGui::GetWindowWidth() - button_size * 2 -
                    15.0f);  // Position before close button
    if (ImGui::Button("⚙", ImVec2(button_size, button_size))) {
      open_settings();
      // Add haptic feedback for button interaction
      BTQuant::UI::HapticFeedback::getInstance().triggerForSubtleInteraction();
    }

    // Show standardized tooltip for the button
    BTQuant::UI::show_control_tooltip("panel_settings_button");
  }

  // Close button (right aligned)
  if (config_.visible) {
    float close_size = ImGui::GetTextLineHeight();
    ImGui::SameLine(ImGui::GetWindowWidth() - close_size - 10.0f);
    if (ImGui::Button("X", ImVec2(close_size, close_size))) {
      config_.visible = false;
      // Add haptic feedback for button interaction
      BTQuant::UI::HapticFeedback::getInstance().triggerForImportantInteraction();
    }

    // Show standardized tooltip for the button
    BTQuant::UI::show_control_tooltip("panel_close_button");
  }

  // Right-click context menu
  if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
    // Set focus to this panel so that hotkeys (e.g., Delete) apply to the correct panel
    ImGui::SetWindowFocus();
    ImGui::OpenPopup("PanelContextMenu");
  }

  // Render context menu if available
  render_context_menu();

  ImGui::Separator();
}

void PanelBase::handle_context_menu(ContextMenuManager& manager) {
  // Check if the window is hovered and right mouse button was clicked
  if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    // Set focus to this panel so that hotkeys (e.g., Delete) apply to the correct panel
    ImGui::SetWindowFocus();

    // If the manager has access to the panel manager, set this panel as the active panel
    if (auto* panel_manager = manager.get_panel_manager()) {
      // Find the panel ID by comparing with all panels in the manager
      auto all_panel_ids = panel_manager->get_all_panel_ids();
      for (uint32_t id : all_panel_ids) {
        PanelBase* manager_panel = panel_manager->get_panel_by_id(id);
        if (manager_panel == this) {
          panel_manager->set_active_panel_id(id);
          break;
        }
      }
    }
  }

  // Show the context menu for this panel
  manager.show_context_menu(this);
}

const char* PanelBase::get_panel_type_name(PanelType type) {
  switch (type) {
    case PanelType::CHART:
      return "Chart";
    case PanelType::METRICS:
      return "Metrics";
    case PanelType::HEATMAP:
      return "Heatmap";
    case PanelType::HISTOGRAM:
      return "Histogram";
    case PanelType::SCATTER_PLOT:
      return "Scatter";
    case PanelType::TIME_SERIES:
      return "Time Series";
    case PanelType::TRADING_ORDERS:
      return "Orders";
    case PanelType::TRADING_POSITIONS:
      return "Positions";
    case PanelType::RISK_METRICS:
      return "Risk";
    case PanelType::ALERTS:
      return "Alerts";
    case PanelType::ORDERBOOK:
      return "Orderbook";
    case PanelType::WATCHLIST:
      return "Watchlist";
    case PanelType::SCREENER:
      return "Screener";
    case PanelType::TAPE:
      return "Tape";
    case PanelType::VOLUME_PROFILE:
      return "Volume Profile";
    case PanelType::DEPTH_CHART:
      return "Depth Chart";
    case PanelType::STATUS_BAR:
      return "Status Bar";
    case PanelType::LOG_PANEL:
      return "Log Panel";
    case PanelType::FOOTPRINT_CHART:
      return "Footprint Chart";
    case PanelType::TPO_PROFILE:
      return "TPO Profile";
    case PanelType::PERFORMANCE_MONITOR:
      return "Performance Monitor";
    case PanelType::TIME_STATISTICS:
      return "Time Statistics";
    case PanelType::TIME_AND_SALES:
      return "Time & Sales";
    case PanelType::TABBED_GROUP:
      return "Tabbed Group";
    default:
      return "Unknown";
  }
}

}  // namespace BTQuant