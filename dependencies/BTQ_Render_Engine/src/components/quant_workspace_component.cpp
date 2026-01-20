#include "../../include/components/quant_workspace_component.hpp"

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge), processor_(processor) {

  // Ensure ImPlot context is created (must be called once)
  static bool implot_init = false;
  if (!implot_init) {
    ImPlot::CreateContext();
    implot_init = true;
  }

  // Create chart manager and indicator renderer
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
  indicator_renderer_ = std::make_unique<IndicatorRenderer>(nullptr, processor);

  // Initialize Trading Systems (Phase 3)
  order_manager_ = std::make_unique<OrderManager>();
  position_manager_ = std::make_unique<PositionManager>();
  risk_assessment_ = std::make_unique<RiskAssessment>();

  // Set up callbacks for order execution -> position updates
  order_manager_->set_execution_callback(
      [this](const OrderManager::OrderExecution &execution) {
        position_manager_->update_position(execution);
      });
}

void QuantWorkspaceComponent::initialize_vulkan_resources(VulkanCore *core) {
  indicator_renderer_->initialize_vulkan_resources();
}

void QuantWorkspaceComponent::update(float dt) { chart_manager_->update(); }

void QuantWorkspaceComponent::render_gui() {
  // Render chart controls
  if (show_chart_controls_) {
    render_chart_controls();
  }

  // Render indicator selector
  if (show_indicator_selector_) {
    render_indicator_selector();
  }

  if (show_trading_panel_) {
    render_trading_panel();
    render_orders_panel();
    render_positions_panel();
  }

  // Render all visible charts
  for (const auto &chart : chart_manager_->get_visible_charts()) {
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    // Unique title with symbol, exchange, and ID to prevent ImGui collisions
    std::string title = chart.symbol_name + " [" + chart.exchange_name + "] (" +
                        std::to_string(static_cast<int>(chart.timeframe)) +
                        ")###chart_" + std::to_string(chart.chart_id);

    if (ImGui::Begin(title.c_str(), &open)) {
      // Render chart without direct access to InstrumentStore
      render_instrument_chart(chart.symbol_name, chart.timeframe, chart);
    }
    ImGui::End();

    if (!open) {
      chart_manager_->destroy_chart(chart.chart_id);
    }
  }
}

void QuantWorkspaceComponent::render_chart_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Chart Controls", &show_chart_controls_)) {
    // Timeframe selector
    const char *timeframes[] = {"1 Minute",   "5 Minutes", "15 Minutes",
                                "1 Hour",     "4 Hours",   "1 Day",
                                "1 Second",   "5 Seconds", "15 Seconds",
                                "30 Seconds", "500ms",     "100ms"};
    int selected = static_cast<int>(selected_timeframe_);
    if (ImGui::Combo("Timeframe", &selected, timeframes,
                     IM_ARRAYSIZE(timeframes))) {
      selected_timeframe_ = static_cast<RenderEngine::TimeFrame>(selected);
    }

    ImGui::Separator();

    // Create chart button
    if (ImGui::Button("Create New Chart")) {
      // Find ID for BTC-USDT (default)
      auto id_opt = chart_manager_->getSymbolId("BTC-USDT");
      uint32_t btc_id =
          id_opt ? *id_opt : 10007; // 10007 is binance BTCUSDT in registry
      chart_manager_->create_chart("BTC-USDT", "Binance", btc_id,
                                   selected_timeframe_);
    }

    ImGui::Separator();

    // Chart list
    ImGui::Text("Active Charts: %zu", chart_manager_->get_charts().size());
    for (const auto &[id, chart] : chart_manager_->get_charts()) {
      std::string chart_label =
          chart.symbol_name + " [" + chart.exchange_name + "] (" +
          std::to_string(static_cast<int>(chart.timeframe)) + ")###cb_" +
          std::to_string(chart.chart_id);
      if (ImGui::Checkbox(chart_label.c_str(),
                          &const_cast<ChartInstance &>(chart).visible)) {
        chart_manager_->toggle_chart_visibility(id);
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_indicator_selector() {
  ImGui::SetNextWindowPos(ImVec2(270, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Indicator Selector", &show_indicator_selector_)) {
    // Default indicator configuration for all charts
    static IndicatorConfig global_config;

    ImGui::Text("Global Indicators");
    ImGui::Separator();

    ImGui::Checkbox("Show SMA 10", &global_config.show_sma_10);
    ImGui::Checkbox("Show SMA 20", &global_config.show_sma_20);
    ImGui::Checkbox("Show SMA 50", &global_config.show_sma_50);
    ImGui::Checkbox("Show EMA 10", &global_config.show_ema_10);
    ImGui::Checkbox("Show EMA 20", &global_config.show_ema_20);
    ImGui::Checkbox("Show EMA 50", &global_config.show_ema_50);
    ImGui::Checkbox("Show RSI", &global_config.show_rsi);
    ImGui::Checkbox("Show MACD", &global_config.show_macd);
    ImGui::Checkbox("Show Bollinger Bands", &global_config.show_bollinger);
    ImGui::Checkbox("Show Stochastic", &global_config.show_stochastic);

    // Apply to all charts
    if (ImGui::Button("Apply to All")) {
      for (auto &[symbol, config] : indicator_configs_) {
        config = global_config;
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_instrument_chart(
    const std::string &symbol, RenderEngine::TimeFrame timeframe,
    const ChartInstance &chart) {

  if (chart.dates.empty()) {
    ImGui::Text("Initializing Stream for %s [%s]...", symbol.c_str(),
                chart.exchange_name.c_str());
    ImGui::Text("Symbol ID: %u", chart.symbol_id);
    return;
  }

  // Diagnostic Overlay
  ImGui::SetCursorPos(ImVec2(10, 30));
  ImGui::BeginGroup();
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 0, 1)); // yellow
  ImGui::Text("Candles: %zu", chart.dates.size());
  if (!chart.dates.empty()) {
    double last_ts = chart.dates.back();
    ImGui::Text("Last TS: %.2f", last_ts);
    ImGui::Text("Last Price: %.2f", chart.closes.back());
  }
  ImGui::PopStyleColor();
  ImGui::EndGroup();

  // Get indicator configuration for this symbol
  auto &indicator_config = indicator_configs_[symbol];

  // Prepare indicator parameters
  std::vector<IndicatorParams> indicators;

  if (indicator_config.show_sma_10) {
    indicators.push_back({IndicatorType::SMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.94f, 1.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_20) {
    indicators.push_back({IndicatorType::SMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {1.0f, 0.0f, 0.3f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_50) {
    indicators.push_back({IndicatorType::SMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_10) {
    indicators.push_back({IndicatorType::EMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_20) {
    indicators.push_back({IndicatorType::EMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.0f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_50) {
    indicators.push_back({IndicatorType::EMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_rsi) {
    indicators.push_back({IndicatorType::RSI_14,
                          14,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.8f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_macd) {
    indicators.push_back({IndicatorType::MACD,
                          12,
                          26,
                          9,
                          2.0,
                          {0.8f, 0.4f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_bollinger) {
    indicators.push_back({IndicatorType::BOLLINGER_MID,
                          20,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.6f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_stochastic) {
    indicators.push_back({IndicatorType::STOCHASTIC_K,
                          14,
                          3,
                          0,
                          2.0,
                          {0.6f, 0.0f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  // Cyber-Cyan: #00F0FF (0xFFFFF000), Neon-Red: #FF0033 (0xFF3300FF)
  ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetColorU32(ImVec4(
                                             0.0f, 0.94f, 1.0f, 1.0f))); // Cyan

  std::string plot_id = symbol + "##" + std::to_string(chart.chart_id);
  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Enable auto-fit for price axis
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_AutoFit);

    ImPlot::SetupAxis(ImAxis_Y2, "Volume",
                      ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_NoGridLines |
                          ImPlotAxisFlags_NoTickLabels);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y2, 0,
                                       1000000); // For Volume alignment

    int count = (int)chart.dates.size();

    if (count > 0) {
      // 1. Determine Visible Range for Culling (Optimization)
      ImPlotRect limits = ImPlot::GetPlotLimits();
      double x_min = limits.X.Min;
      double x_max = limits.X.Max;

      // Professional Candlestick Rendering (Batched)
      std::vector<double> up_wick_x, up_wick_y;
      std::vector<double> down_wick_x, down_wick_y;
      std::vector<double> up_b_x, up_b_y1, up_b_y2;
      std::vector<double> down_b_x, down_b_y1, down_b_y2;

      // Heuristic for candle width based on timeframe
      double candle_width = 30.0;
      if (timeframe == RenderEngine::TimeFrame::TF_100MS)
        candle_width = 0.08;
      else if (timeframe == RenderEngine::TimeFrame::TF_500MS)
        candle_width = 0.4;
      else if (timeframe == RenderEngine::TimeFrame::TF_1SEC)
        candle_width = 0.8;
      else if (timeframe == RenderEngine::TimeFrame::TF_5SEC)
        candle_width = 4.0;
      else if (timeframe == RenderEngine::TimeFrame::TF_1MIN)
        candle_width = 45.0;
      else if (timeframe == RenderEngine::TimeFrame::TF_5MIN)
        candle_width = 225.0;
      else if (timeframe == RenderEngine::TimeFrame::TF_1DAY)
        candle_width = 64800.0;

      for (int i = 0; i < count; ++i) {
        double x = chart.dates[i];

        // Simple Culling: Skip candles outside the view (with some padding)
        if (x < x_min - candle_width || x > x_max + candle_width)
          continue;

        if (chart.closes[i] >= chart.opens[i]) {
          // Bullish Wick
          up_wick_x.push_back(x);
          up_wick_x.push_back(x);
          up_wick_y.push_back(chart.highs[i]);
          up_wick_y.push_back(chart.lows[i]);

          // Bullish Body
          up_b_x.push_back(x);
          up_b_y1.push_back(chart.opens[i]);
          up_b_y2.push_back(chart.closes[i]);
        } else {
          // Bearish Wick
          down_wick_x.push_back(x);
          down_wick_x.push_back(x);
          down_wick_y.push_back(chart.highs[i]);
          down_wick_y.push_back(chart.lows[i]);

          // Bearish Body
          down_b_x.push_back(x);
          down_b_y1.push_back(chart.opens[i]);
          down_b_y2.push_back(chart.closes[i]);
        }
      }

      // Draw Wicks (Neon Style)
      ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
      if (!up_wick_x.empty()) {
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.4f, 1.0f));
        ImPlot::PlotLine("##UpWicks", up_wick_x.data(), up_wick_y.data(),
                         (int)up_wick_x.size(), ImPlotLineFlags_Segments);
      }
      if (!down_wick_x.empty()) {
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.2f, 1.0f));
        ImPlot::PlotLine("##DownWicks", down_wick_x.data(), down_wick_y.data(),
                         (int)down_wick_x.size(), ImPlotLineFlags_Segments);
      }
      ImPlot::PopStyleVar();

      // Draw Bodies (Using PlotBars for robustness and performance)
      if (!up_b_x.empty()) {
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 1.0f, 0.4f, 0.6f));
        ImPlot::PlotBars("##UpBodies", up_b_x.data(), up_b_y1.data(),
                         up_b_y2.data(), (int)up_b_x.size(),
                         candle_width * 0.82);
      }
      if (!down_b_x.empty()) {
        ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.0f, 0.2f, 0.6f));
        ImPlot::PlotBars("##DownBodies", down_b_x.data(), down_b_y1.data(),
                         down_b_y2.data(), (int)down_b_x.size(),
                         candle_width * 0.82);
      }
    }
  }

  // Plot indicators
  indicator_renderer_->render_indicators(symbol, timeframe, indicators);

  ImPlot::EndPlot();
}
ImPlot::PopStyleColor();
}

void QuantWorkspaceComponent::clear_data() {
  chart_manager_.reset();
  indicator_renderer_.reset();
  indicator_configs_.clear();
}

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

      order_manager_->place_order(order);
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

} // namespace BTQuant
