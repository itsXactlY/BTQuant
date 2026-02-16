#include "../../include/components/trading_positions_panel.hpp"

#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TradingPositionsPanel::TradingPositionsPanel(const PanelConfig& config,
                                             std::shared_ptr<PositionManager> position_manager,
                                             std::shared_ptr<RiskAssessment> risk_assessment)
    : PanelBase(config), position_manager_(position_manager), risk_assessment_(risk_assessment) {}

void TradingPositionsPanel::initialize() { PanelBase::initialize(); }

void TradingPositionsPanel::render_content() {
  begin_panel_window();

  if (ImGui::BeginTable("PositionsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Symbol");
    ImGui::TableSetupColumn("Side");
    ImGui::TableSetupColumn("Quantity");
    ImGui::TableSetupColumn("Avg Price");
    ImGui::TableSetupColumn("Unrealized P/L");
    ImGui::TableHeadersRow();

    if (position_manager_) {
      auto positions = position_manager_->get_all_positions();

      for (const auto& position : positions) {
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", position.symbol.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%s",
                    position.quantity > 0 ? "Long" : (position.quantity < 0 ? "Short" : "Flat"));

        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.4f", position.quantity);

        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.4f", position.average_price);

        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", position.unrealized_pnl);
      }
    }

    ImGui::EndTable();
  }

  end_panel_window();
}

}  // namespace BTQuant