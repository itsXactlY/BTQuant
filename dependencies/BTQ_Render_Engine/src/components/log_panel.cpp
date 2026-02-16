#include "../../include/components/log_panel.hpp"

#include <iostream>

#include "imgui.h"

namespace BTQuant {

LogPanel::LogPanel(const PanelConfig& config) : PanelBase(config) {}

void LogPanel::initialize() { PanelBase::initialize(); }

void LogPanel::render_content() {
  begin_panel_window();

  // Log controls
  static int log_level = 0;
  const char* log_levels[] = {"Debug", "Info", "Warning", "Error", "Critical"};
  ImGui::Combo("Log Level", &log_level, log_levels, 5);

  ImGui::Separator();

  // Sample log entries - in a real implementation this would connect to a logger
  static const char* log_entries[] = {
      "[10:30:15] INFO: Market data connection established",
      "[10:30:16] DEBUG: Orderbook snapshot received for BTC-USDT",
      "[10:30:17] WARNING: High latency detected: 45ms",
      "[10:30:18] ERROR: Connection timeout to exchange API",
      "[10:30:19] INFO: Reconnection attempt #1",
      "[10:30:20] DEBUG: Position update: BTC-USDT +1.5",
      "[10:30:21] CRITICAL: Risk threshold exceeded",
      "[10:30:22] INFO: Risk management activated",
      "[10:30:23] DEBUG: Order filled: LIMIT BUY BTC-USDT @ 45000.00",
      "[10:30:24] INFO: P/L update: +$125.50"};

  ImGui::BeginChild("LogScrollingRegion", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), false,
                    ImGuiWindowFlags_HorizontalScrollbar);

  for (const char* entry : log_entries) {
    // Color code based on log level
    if (strstr(entry, "DEBUG")) {
      ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s", entry);
    } else if (strstr(entry, "INFO")) {
      ImGui::Text("%s", entry);
    } else if (strstr(entry, "WARNING")) {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", entry);
    } else if (strstr(entry, "ERROR")) {
      ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "%s", entry);
    } else if (strstr(entry, "CRITICAL")) {
      ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s", entry);
    } else {
      ImGui::Text("%s", entry);
    }
  }

  ImGui::EndChild();

  // Auto-scroll to bottom
  if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }

  end_panel_window();
}

}  // namespace BTQuant