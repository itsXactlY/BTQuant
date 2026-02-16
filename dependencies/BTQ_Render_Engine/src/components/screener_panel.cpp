#include "../../include/components/screener_panel.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

ScreenerPanel::ScreenerPanel(const PanelConfig& config) : PanelBase(config) {}

void ScreenerPanel::initialize() { PanelBase::initialize(); }

void ScreenerPanel::render_content() {
  begin_panel_window();

  // Screener controls
  static char filter_text[128] = "";
  ImGui::InputText("Symbol Filter", filter_text, sizeof(filter_text));

  if (ImGui::BeginTable("ScreenerTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Symbol");
    ImGui::TableSetupColumn("Price");
    ImGui::TableSetupColumn("Change");
    ImGui::TableSetupColumn("Volume");
    ImGui::TableSetupColumn("Market Cap");
    ImGui::TableSetupColumn("Action");
    ImGui::TableHeadersRow();

    // Sample data - in a real implementation this would come from market data
    const char* symbols[] = {"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"};
    float prices[] = {175.32f, 2780.45f, 330.12f, 245.67f, 145.89f, 485.23f, 875.43f, 520.78f};
    float changes[] = {1.2f, -0.8f, 0.5f, 2.3f, -1.1f, 0.9f, 3.4f, -0.6f};
    float volumes[] = {45.2f, 1.8f, 28.7f, 89.3f, 62.1f, 15.4f, 32.6f, 21.9f};  // in millions
    float market_caps[] = {2.76, 2.75, 2.58, 780.2, 1.48, 1.02, 540.3, 220.5};  // in billions

    for (int i = 0; i < 8; ++i) {
      // Apply filter
      if (filter_text[0] != '\0' && strstr(symbols[i], filter_text) == nullptr) {
        continue;
      }

      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s", symbols[i]);

      ImGui::TableSetColumnIndex(1);
      ImGui::Text("$%.2f", prices[i]);

      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(
          changes[i] >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
          "%.2f%%", changes[i]);

      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%.1fM", volumes[i]);

      ImGui::TableSetColumnIndex(4);
      ImGui::Text("$%.1fB", market_caps[i]);

      ImGui::TableSetColumnIndex(5);
      if (ImGui::SmallButton(("Trade##" + std::to_string(i)).c_str())) {
        // In a real implementation, this would open a trading dialog
        std::cout << "Opening trade dialog for " << symbols[i] << std::endl;
      }
    }

    ImGui::EndTable();
  }

  end_panel_window();
}

}  // namespace BTQuant