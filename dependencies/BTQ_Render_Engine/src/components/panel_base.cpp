#include "../../include/components/panel_base.hpp"

#include "imgui.h"
#include "ui/context_menus.hpp"
#include "ui/ui_base.hpp"

namespace BTQuant {

void PanelBase::begin_panel_window() {
  ImGui::SetNextWindowPos(config_.position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(config_.size, ImGuiCond_FirstUseEver);

  ImGuiWindowFlags flags = UI::PANEL_DEFAULT_FLAGS;
  if (!config_.resizable) flags |= ImGuiWindowFlags_NoResize;
  if (!config_.movable) flags |= ImGuiWindowFlags_NoMove;

  // Use ThemeManager for glass style
  push_glass_style();

  // Use stable window ID for DockBuilder compatibility (Title###TypeName_N)
  std::string window_title = get_imgui_window_id();

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

  // Render background image if enabled using channel splitting to ensure it's behind other content
  if (use_background_image_ && background_texture_ != 0) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Split the draw list into channels: 0 for background, 1 for foreground
    draw_list->ChannelsSplit(2);

    // Switch to background channel (0)
    draw_list->ChannelsSetCurrent(0);

    // Get the current window position and size
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();

    // Define the rectangle for the background image spanning the entire window
    ImVec2 bg_min = window_pos;
    ImVec2 bg_max = ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y);

    // Add the image to the draw list, spanning the entire panel background
    draw_list->AddImage(background_texture_, bg_min, bg_max, ImVec2(0, 0),
                        ImVec2(1, 1));  // UV coordinates default to full texture

    // Switch back to foreground channel (1) for normal rendering
    draw_list->ChannelsSetCurrent(1);
  }

  render_panel_header();

  // Default content for placeholder panels
  ImGui::Text("Panel Type: %s", get_panel_type_name(config_.type));
  ImGui::Text("Implementation coming soon...");

  // Merge channels back together if we were using background image
  if (use_background_image_ && background_texture_ != 0) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->ChannelsMerge();
  }

  end_panel_window();

  // Render settings modal if available
  if (auto* settings = get_settings_interface()) {
    settings->render();
  }
}

void PanelBase::render_panel_header() {
  // Panel type indicator
  const char* type_name = get_panel_type_name(config_.type);
  ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "[%s]", type_name);

  ImGui::SameLine();
  ImGui::Text("%s", config_.title.c_str());

  // Settings button (left of close button)
  if (get_settings_interface() != nullptr) {
    float button_size = ImGui::GetTextLineHeight();
    ImGui::SameLine(ImGui::GetWindowWidth() - button_size * 2 -
                    15.0f);  // Position before close button
    if (ImGui::Button("⚙", ImVec2(button_size, button_size))) {
      open_settings();
    }
  }

  // Close button (right aligned)
  if (config_.visible) {
    float close_size = ImGui::GetTextLineHeight();
    ImGui::SameLine(ImGui::GetWindowWidth() - close_size - 10.0f);
    if (ImGui::Button("X", ImVec2(close_size, close_size))) {
      config_.visible = false;
    }
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
    case PanelType::TIME_HISTOGRAM:
      return "Time Histogram";
    case PanelType::TIME_AND_SALES:
      return "Time & Sales";
    case PanelType::HISTORICAL_TIME_SALES:
      return "Historical T&S";
    case PanelType::CHART_REPLAY:
      return "Chart Replay";
    case PanelType::RISK_ANALYZER:
      return "Risk Analyzer";
    case PanelType::STRATEGY_BUILDER:
      return "Strategy Builder";
    case PanelType::OPTION_ANALYTICS:
      return "Option Analytics";
    case PanelType::CORRELATION_HEATMAP:
      return "Correlation Heatmap";
    case PanelType::DOM_SURFACE:
      return "DOM Surface";
    case PanelType::MULTI_VWAP:
      return "Multi VWAP";
    case PanelType::TECHNICAL_INDICATORS:
      return "Technical Indicators";
    case PanelType::THEME_CUSTOMIZATION:
      return "Theme Customization";
    case PanelType::KEYBOARD_SHORTCUTS:
      return "Keyboard Shortcuts";
    case PanelType::DRAWING_TOOLS:
      return "Drawing Tools";
    default:
      return "Unknown";
  }
}

}  // namespace BTQuant