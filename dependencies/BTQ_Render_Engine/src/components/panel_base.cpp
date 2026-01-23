#include "../../include/components/panel_base.hpp"
#include "imgui.h"

namespace BTQuant {

void PanelBase::begin_panel_window() {
  ImGui::SetNextWindowPos(config_.position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(config_.size, ImGuiCond_FirstUseEver);

  ImGuiWindowFlags flags = ImGuiWindowFlags_None;
  if (!config_.resizable)
    flags |= ImGuiWindowFlags_NoResize;
  if (!config_.movable)
    flags |= ImGuiWindowFlags_NoMove;

  // Professional styling
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 0.95f));
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_TitleBgActive,
                        ImVec4(0.15f, 0.15f, 0.15f, 1.0f));

  std::string window_title = config_.title + "###panel_" +
                             std::to_string(reinterpret_cast<uintptr_t>(this));
  ImGui::Begin(window_title.c_str(), &config_.visible, flags);
}

void PanelBase::end_panel_window() {
  ImGui::End();
  ImGui::PopStyleColor(4);
  ImGui::PopStyleVar(2);
}

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
}

void PanelBase::render_panel_header() {
  // Panel type indicator
  const char *type_name = get_panel_type_name(config_.type);
  ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "[%s]", type_name);

  ImGui::SameLine();
  ImGui::Text("%s", config_.title.c_str());

  // Close button (right aligned)
  if (config_.visible) {
    float close_size = ImGui::GetTextLineHeight();
    ImGui::SameLine(ImGui::GetWindowWidth() - close_size - 10.0f);
    if (ImGui::Button("X", ImVec2(close_size, close_size))) {
      config_.visible = false;
    }
  }

  ImGui::Separator();
}

const char *PanelBase::get_panel_type_name(PanelType type) {
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
  default:
    return "Unknown";
  }
}

} // namespace BTQuant