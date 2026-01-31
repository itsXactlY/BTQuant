// Trading panel rendering methods for QuantWorkspace
// Include this BEFORE the closing namespace brace

void QuantWorkspaceComponent::render_trading_panel() {
  ImGui::SetNextWindowPos(ImVec2(530, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Trading Panel", &show_trading_panel_)) {
    ImGui::Text("Order Entry");
    ImGui::Separator();

    // Symbol selector
    static char symbol_input[128] = "BTC-USDT";
    ImGui::InputText("Symbol", symbol_input, sizeof(symbol_input));
    selected_symbol_ = std::string(symbol_input);

    // Order type
    const char *order_types[] = {"Market", "Limit"};
    ImGui::Combo("Order Type", &selected_order_type_, order_types,
                 IM_ARRAYSIZE(order_types));

    // Order side (Buy/Sell)
    const char *order_sides[] = {"Buy", "Sell"};
    ImGui::Combo("Side", &selected_order_side_, order_sides,
                 IM_ARRAYSIZE(order_sides));

    // Quantity
    ImGui::InputDouble("Quantity", &order_quantity_, 0.01, 1.0, "%.4f");

    // Price (only for limit orders)
    if (selected_order_type_ == 1) {
      ImGui::InputDouble("Price", &order_price_, 0.01, 10.0, "%.2f");
    }

    ImGui::Separator();

    // Place order button
    ImVec4 button_color = (selected_order_side_ == 0)
                              ? ImVec4(0.0f, 0.8f, 0.2f, 1.0f)
                              : ImVec4(1.0f, 0.2f, 0.0f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, button_color);

    if (ImGui::Button("Place Order", ImVec2(-1, 40))) {
      OrderManager::Order order;
      order.symbol = selected_symbol_;
      order.quantity = order_quantity_;
      order.side = (selected_order_side_ == 0) ? OrderManager::OrderSide::Buy
                                               : OrderManager::OrderSide::Sell;
      order.type = (selected_order_type_ == 0) ? OrderManager::OrderType::Market
                                               : OrderManager::OrderType::Limit;
      order.price = order_price_;

      std::string order_id = order_manager_->place_order(order);
      std::cout << "[Trading] Order placed: " << order_id << std::endl;
    }

    ImGui::PopStyleColor();
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_orders_panel() {
  ImGui::SetNextWindowPos(ImVec2(890, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

  static bool show_orders = true;
  if (ImGui::Begin("Active Orders", &show_orders)) {
    auto orders = order_manager_->get_active_orders();

    ImGui::Text("Active Orders: %zu", orders.size());
    ImGui::Separator();

    if (ImGui::BeginTable("OrdersTable", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Symbol");
      ImGui::TableSetupColumn("Side");
      ImGui::TableSetupColumn("Type");
      ImGui::TableSetupColumn("Qty");
      ImGui::TableSetupColumn("Price");
      ImGui::TableHeadersRow();

      for (const auto &order : orders) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", order.symbol.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", (order.side == OrderManager::OrderSide::Buy)
                              ? "BUY"
                              : "SELL");
        ImGui::TableNextColumn();
        ImGui::Text("%s", (order.type == OrderManager::OrderType::Market)
                              ? "Market"
                              : "Limit");
        ImGui::TableNextColumn();
        ImGui::Text("%.4f", order.quantity);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", order.price);
      }

      ImGui::EndTable();
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_positions_panel() {
  ImGui::SetNextWindowPos(ImVec2(890, 320), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

  static bool show_positions = true;
  if (ImGui::Begin("Positions", &show_positions)) {
    auto positions = position_manager_->get_positions();
    auto summary = position_manager_->get_portfolio_summary();

    // Portfolio summary
    ImGui::Text("Portfolio Summary");
    ImGui::Separator();
    ImGui::Text("Total Value: $%.2f", summary.total_value);
    ImGui::Text("Unrealized P&L: $%.2f", summary.total_unrealized_pnl);
    ImGui::Text("Realized P&L: $%.2f", summary.total_realized_pnl);
    ImGui::Text("Positions: %d", summary.position_count);

    ImGui::Separator();
    ImGui::Text("Active Positions");
    ImGui::Separator();

    if (ImGui::BeginTable("PositionsTable", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Symbol");
      ImGui::TableSetupColumn("Qty");
      ImGui::TableSetupColumn("Avg Price");
      ImGui::TableSetupColumn("Market Val");
      ImGui::TableSetupColumn("P&L");
      ImGui::TableHeadersRow();

      for (const auto &pos : positions) {
        if (std::abs(pos.quantity) < 0.0001)
          continue; // Skip zero positions

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", pos.symbol.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%.4f", pos.quantity);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", pos.average_price);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", pos.market_value);
        ImGui::TableNextColumn();

        // Color-code P&L
        ImVec4 pnl_color = (pos.unrealized_pnl >= 0)
                               ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                               : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
        ImGui::TextColored(pnl_color, "%.2f", pos.unrealized_pnl);
      }

      ImGui::EndTable();
    }
  }
  ImGui::End();
}
