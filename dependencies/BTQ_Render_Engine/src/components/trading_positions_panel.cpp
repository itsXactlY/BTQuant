#include "../../include/components/trading_positions_panel.hpp"

#include <iostream>

#include "imgui.h"
#include "implot.h"
#include "market_data_processor.hpp"

namespace BTQuant {

TradingPositionsPanel::TradingPositionsPanel(const PanelConfig& config,
                                             std::shared_ptr<PositionManager> position_manager,
                                             std::shared_ptr<RiskAssessment> risk_assessment)
    : PanelBase(config), position_manager_(position_manager), risk_assessment_(risk_assessment) {}

void TradingPositionsPanel::initialize() { PanelBase::initialize(); }

void TradingPositionsPanel::render_content() {
  // V-Sync locked Mark-to-Market PnL update
  // This ensures atomic best_bid/best_ask are read just before rendering,
  // synchronized with the display refresh to prevent tearing
  if (position_manager_) {
    position_manager_->updateMarkToMarketPnL();
  }

  begin_panel_window();

  if (ImGui::BeginTable("PositionsTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Symbol");
    ImGui::TableSetupColumn("Side");
    ImGui::TableSetupColumn("Quantity");
    ImGui::TableSetupColumn("Avg Price");
    ImGui::TableSetupColumn("Mark Price");
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
        // Display atomic mark price (mid price from atomic BBO)
        ImGui::Text("%.4f", position.mtm_mid_price);

        ImGui::TableSetColumnIndex(5);
        // Color-code PnL for visual clarity
        ImVec4 pnl_color = (position.unrealized_pnl >= 0.0f)
            ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)  // Green for profit
            : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);  // Red for loss
        ImGui::PushStyleColor(ImGuiCol_Text, pnl_color);
        ImGui::Text("%.2f", position.unrealized_pnl);
        ImGui::PopStyleColor();
      }
    }

    ImGui::EndTable();
  }

  end_panel_window();
}

}  // namespace BTQuant