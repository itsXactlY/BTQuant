#include "../../include/components/trading_orders_panel.hpp"

#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TradingOrdersPanel::TradingOrdersPanel(const PanelConfig& config,
                                       std::shared_ptr<OrderManager> order_manager,
                                       std::shared_ptr<PositionManager> position_manager)
    : PanelBase(config), order_manager_(order_manager), position_manager_(position_manager) {}

void TradingOrdersPanel::initialize() { PanelBase::initialize(); }

void TradingOrdersPanel::render() {
  begin_panel_window();

  if (ImGui::BeginTable("OrdersTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("ID");
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Side");
    ImGui::TableSetupColumn("Qty");
    ImGui::TableSetupColumn("Price");
    ImGui::TableSetupColumn("Status");
    ImGui::TableHeadersRow();

    if (order_manager_) {
      auto orders = order_manager_->get_active_orders("");

      for (const auto& order : orders) {
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", order.order_id.c_str());

        ImGui::TableSetColumnIndex(1);
        // Determine type string
        std::string type_str;
        switch (order.type) {
          case OrderManager::OrderType::Market:
            type_str = "Market";
            break;
          case OrderManager::OrderType::Limit:
            type_str = "Limit";
            break;
          case OrderManager::OrderType::Stop:
            type_str = "Stop";
            break;
          default:
            type_str = "Other";
            break;
        }
        ImGui::Text("%s", type_str.c_str());

        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%s", order.side == OrderManager::OrderSide::Buy ? "Buy" : "Sell");

        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.4f", order.quantity);

        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.4f", order.price);

        ImGui::TableSetColumnIndex(5);
        // Determine status string
        std::string status_str;
        switch (order.status) {
          case OrderManager::OrderStatus::Pending:
            status_str = "Pending";
            break;
          case OrderManager::OrderStatus::PartiallyFilled:
            status_str = "Partial";
            break;
          case OrderManager::OrderStatus::Filled:
            status_str = "Filled";
            break;
          case OrderManager::OrderStatus::Cancelled:
            status_str = "Cancelled";
            break;
          default:
            status_str = "Other";
            break;
        }
        ImGui::Text("%s", status_str.c_str());
      }
    }

    ImGui::EndTable();
  }

  end_panel_window();
}

}  // namespace BTQuant