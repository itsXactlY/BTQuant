#include "../../include/components/trading_orders_panel.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TradingOrdersPanel::TradingOrdersPanel(const PanelConfig& config,
                                       std::shared_ptr<OrderManager> order_manager,
                                       std::shared_ptr<PositionManager> position_manager)
    : PanelBase(config), order_manager_(order_manager), position_manager_(position_manager) {}

void TradingOrdersPanel::initialize() { PanelBase::initialize(); }

void TradingOrdersPanel::render_content() {
  begin_panel_window();

  // Add heatmap intensity slider to the panel header for resting limit orders
  ImGui::Text("Heatmap Intensity:");
  ImGui::SameLine();
  ImGui::PushItemWidth(200);
  ImGui::SliderFloat("##HeatmapIntensity", &heatmap_intensity_, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##HeatmapIntensity")) {
    heatmap_intensity_ = 1.0f;
  }
  ImGui::Separator();

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

        // Calculate heatmap intensity for limit orders based on quantity and heatmap_intensity_
        bool is_limit_order = (order.type == OrderManager::OrderType::Limit);
        bool is_resting = (order.status == OrderManager::OrderStatus::Pending || 
                          order.status == OrderManager::OrderStatus::PartiallyFilled);
        
        // Apply visual effect for resting limit orders based on heatmap intensity
        if (is_limit_order && is_resting) {
          // Calculate relative size for heatmap effect
          float normalized_size = std::min(static_cast<float>(order.quantity) / 100.0f, 1.0f); // Assuming 100 as reference size
          float adjusted_intensity = static_cast<float>(std::pow(normalized_size, 1.0f / heatmap_intensity_));
          
          // Apply background color based on heatmap intensity
          if (adjusted_intensity > 0.1f) {
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 row_pos = ImGui::GetCursorScreenPos();
            float row_height = ImGui::GetTextLineHeightWithSpacing();
            float table_width = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();
            
            ImVec2 pos_min = row_pos;
            ImVec2 pos_max = ImVec2(row_pos.x + table_width, row_pos.y + row_height);
            
            // Different colors for buy/sell limit orders
            ImVec4 color = order.side == OrderManager::OrderSide::Buy ? 
                          ImVec4(0.0f, 0.6f, 1.0f, adjusted_intensity * 0.2f) : 
                          ImVec4(1.0f, 0.5f, 0.0f, adjusted_intensity * 0.2f);
                          
            ImU32 bg_color = ImGui::GetColorU32(color);
            draw_list->AddRectFilled(pos_min, pos_max, bg_color);
          }
        }

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
        
        // Highlight limit orders
        if (order.type == OrderManager::OrderType::Limit) {
          ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "%s", type_str.c_str());
        } else {
          ImGui::Text("%s", type_str.c_str());
        }

        ImGui::TableSetColumnIndex(2);
        ImVec4 side_color = order.side == OrderManager::OrderSide::Buy ? 
                           ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for buy
                           ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for sell
        ImGui::TextColored(side_color, "%s", order.side == OrderManager::OrderSide::Buy ? "Buy" : "Sell");

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
        
        // Highlight resting orders (Pending or Partially Filled)
        if (order.status == OrderManager::OrderStatus::Pending || 
            order.status == OrderManager::OrderStatus::PartiallyFilled) {
          ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", status_str.c_str());
        } else {
          ImGui::Text("%s", status_str.c_str());
        }
      }
    }

    ImGui::EndTable();
  }

  end_panel_window();
}

}  // namespace BTQuant