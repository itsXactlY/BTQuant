#include "../../include/components/realtime_dashboard_component.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace BTQuant {

RealtimeDashboardComponent::RealtimeDashboardComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge), processor_(processor) {

  // Initialize default panels
  add_panel(DashboardPanelType::PRICE_CHART, "BTC-USDT Price", ImVec2(10, 10),
            ImVec2(800, 400));
  add_panel(DashboardPanelType::VOLUME_CHART, "Volume", ImVec2(820, 10),
            ImVec2(400, 200));
  add_panel(DashboardPanelType::INDICATORS, "Technical Indicators",
            ImVec2(10, 420), ImVec2(800, 200));
  add_panel(DashboardPanelType::ORDER_BOOK, "Order Book", ImVec2(820, 220),
            ImVec2(400, 400));
  add_panel(DashboardPanelType::RECENT_TRADES, "Recent Trades", ImVec2(10, 630),
            ImVec2(400, 200));
  add_panel(DashboardPanelType::MARKET_STATS, "Market Stats", ImVec2(420, 630),
            ImVec2(400, 200));
  add_panel(DashboardPanelType::PERFORMANCE_METRICS, "Performance",
            ImVec2(1230, 10), ImVec2(400, 300));
  add_panel(DashboardPanelType::MULTI_SYMBOL_OVERVIEW, "Multi-Symbol",
            ImVec2(1230, 320), ImVec2(400, 300));
}

void RealtimeDashboardComponent::update(float dt) {
  // Update data sources if needed
  // Real-time updates are handled in render methods
}

void RealtimeDashboardComponent::render_gui() {
  render_dashboard_menu();

  // Render all panels
  for (const auto &panel : panels_) {
    if (panel.visible) {
      render_panel(panel);
    }
  }

  if (show_panel_config_) {
    render_panel_config_window();
  }
}

void RealtimeDashboardComponent::initialize_vulkan_resources(VulkanCore *core) {
  // Initialize any Vulkan resources if needed
}

void RealtimeDashboardComponent::clear_data() {
  // Clear any cached data
}

void RealtimeDashboardComponent::add_panel(DashboardPanelType type,
                                           const std::string &title, ImVec2 pos,
                                           ImVec2 size) {
  DashboardPanel panel;
  panel.type = type;
  panel.title = title;
  panel.position = pos;
  panel.size = size;
  panels_.push_back(panel);
}

void RealtimeDashboardComponent::remove_panel(int index) {
  if (index >= 0 && index < static_cast<int>(panels_.size())) {
    panels_.erase(panels_.begin() + index);
  }
}

void RealtimeDashboardComponent::render_panel(const DashboardPanel &panel) {
  ImGui::SetNextWindowPos(panel.position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(panel.size, ImGuiCond_FirstUseEver);

  std::string window_title =
      panel.title + "###panel_" + std::to_string(&panel - &panels_[0]);
  if (ImGui::Begin(window_title.c_str(), nullptr,
                   ImGuiWindowFlags_NoCollapse)) {
    switch (panel.type) {
    case DashboardPanelType::PRICE_CHART:
      render_price_chart_panel(panel);
      break;
    case DashboardPanelType::VOLUME_CHART:
      render_volume_chart_panel(panel);
      break;
    case DashboardPanelType::INDICATORS:
      render_indicators_panel(panel);
      break;
    case DashboardPanelType::ORDER_BOOK:
      render_order_book_panel(panel);
      break;
    case DashboardPanelType::RECENT_TRADES:
      render_recent_trades_panel(panel);
      break;
    case DashboardPanelType::MARKET_STATS:
      render_market_stats_panel(panel);
      break;
    case DashboardPanelType::PERFORMANCE_METRICS:
      render_performance_metrics_panel(panel);
      break;
    case DashboardPanelType::MULTI_SYMBOL_OVERVIEW:
      render_multi_symbol_overview_panel(panel);
      break;
    }
  }
  ImGui::End();
}

void RealtimeDashboardComponent::render_price_chart_panel(
    const DashboardPanel &panel) {
  auto ohlcv_data = get_ohlcv_data(panel.symbol, panel.timeframe, 500);

  if (ohlcv_data.empty()) {
    ImGui::Text("No data available");
    return;
  }

  // Prepare data for ImPlot
  std::vector<double> timestamps, opens, highs, lows, closes;

  for (const auto &candle : ohlcv_data) {
    timestamps.push_back(static_cast<double>(candle.timestamp));
    opens.push_back(candle.open);
    highs.push_back(candle.high);
    lows.push_back(candle.low);
    closes.push_back(candle.close);
  }

  if (ImPlot::BeginPlot("Price Chart", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_Time, 0);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%H:%M:%S");

    // Plot candlesticks
    ImPlot::PlotCandlestick("BTC-USDT", timestamps.data(), opens.data(),
                            highs.data(), lows.data(), closes.data(),
                            timestamps.size());

    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_volume_chart_panel(
    const DashboardPanel &panel) {
  auto ohlcv_data = get_ohlcv_data(panel.symbol, panel.timeframe, 500);

  if (ohlcv_data.empty()) {
    ImGui::Text("No data available");
    return;
  }

  std::vector<double> timestamps, volumes;

  for (const auto &candle : ohlcv_data) {
    timestamps.push_back(static_cast<double>(candle.timestamp));
    volumes.push_back(candle.volume);
  }

  if (ImPlot::BeginPlot("Volume", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Volume", ImPlotAxisFlags_Time, 0);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%H:%M:%S");

    ImPlot::PlotBars("Volume", timestamps.data(), volumes.data(),
                     timestamps.size());

    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_indicators_panel(
    const DashboardPanel &panel) {
  auto ohlcv_data = get_ohlcv_data(panel.symbol, panel.timeframe, 200);

  if (ohlcv_data.empty()) {
    ImGui::Text("No data available");
    return;
  }

  // Calculate RSI
  auto rsi_result = TechnicalIndicators::rsi(ohlcv_data, 14);

  // Calculate MACD
  auto macd_results = TechnicalIndicators::macd(ohlcv_data, 12, 26, 9);

  if (ImPlot::BeginPlot("Indicators", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Value", ImPlotAxisFlags_Time, 0);

    // Plot RSI
    if (!rsi_result.values.empty()) {
      std::vector<double> rsi_timestamps;
      for (const auto &ts : rsi_result.timestamps) {
        rsi_timestamps.push_back(static_cast<double>(ts));
      }
      ImPlot::PlotLine("RSI", rsi_timestamps.data(), rsi_result.values.data(),
                       rsi_timestamps.size());
    }

    // Plot MACD
    if (macd_results.size() >= 2) {
      std::vector<double> macd_timestamps;
      for (const auto &ts : macd_results[0].timestamps) {
        macd_timestamps.push_back(static_cast<double>(ts));
      }
      ImPlot::PlotLine("MACD", macd_timestamps.data(),
                       macd_results[0].values.data(), macd_timestamps.size());
      ImPlot::PlotLine("Signal", macd_timestamps.data(),
                       macd_results[1].values.data(), macd_timestamps.size());
    }

    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_order_book_panel(
    const DashboardPanel &panel) {
  auto orderbook = get_orderbook_data(panel.symbol);

  if (orderbook.bids.empty() && orderbook.asks.empty()) {
    ImGui::Text("No order book data");
    return;
  }

  ImGui::Columns(3, "OrderBook", false);
  ImGui::Text("Bid Price");
  ImGui::NextColumn();
  ImGui::Text("Bid Size");
  ImGui::NextColumn();
  ImGui::Text("Ask Price");
  ImGui::NextColumn();
  ImGui::Separator();

  size_t max_levels = std::max(orderbook.bids.size(), orderbook.asks.size());
  for (size_t i = 0; i < max_levels; ++i) {
    if (i < orderbook.bids.size()) {
      ImGui::Text("%.2f", orderbook.bids[i].price);
      ImGui::NextColumn();
      ImGui::Text("%.4f", orderbook.bids[i].size);
      ImGui::NextColumn();
    } else {
      ImGui::Text("");
      ImGui::NextColumn();
      ImGui::Text("");
      ImGui::NextColumn();
    }

    if (i < orderbook.asks.size()) {
      ImGui::Text("%.2f", orderbook.asks[i].price);
      ImGui::NextColumn();
    } else {
      ImGui::Text("");
      ImGui::NextColumn();
    }
  }
  ImGui::Columns(1);
}

void RealtimeDashboardComponent::render_recent_trades_panel(
    const DashboardPanel &panel) {
  auto trades = get_recent_trades(panel.symbol, 20);

  if (trades.empty()) {
    ImGui::Text("No recent trades");
    return;
  }

  ImGui::Columns(4, "Trades", false);
  ImGui::Text("Time");
  ImGui::NextColumn();
  ImGui::Text("Price");
  ImGui::NextColumn();
  ImGui::Text("Size");
  ImGui::NextColumn();
  ImGui::Text("Side");
  ImGui::NextColumn();
  ImGui::Separator();

  for (const auto &trade : trades) {
    auto time_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::time_point(
            std::chrono::milliseconds(trade.timestamp)));
    std::tm *tm = std::localtime(&time_t);
    char time_str[9];
    std::strftime(time_str, sizeof(time_str), "%H:%M:%S", tm);

    ImGui::Text("%s", time_str);
    ImGui::NextColumn();
    ImGui::Text("%.2f", trade.price);
    ImGui::NextColumn();
    ImGui::Text("%.4f", trade.size);
    ImGui::NextColumn();
    ImGui::TextColored(trade.is_buy ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                       trade.is_buy ? "BUY" : "SELL");
    ImGui::NextColumn();
  }
  ImGui::Columns(1);
}

void RealtimeDashboardComponent::render_market_stats_panel(
    const DashboardPanel &panel) {
  auto ohlcv_data =
      get_ohlcv_data(panel.symbol, RenderEngine::TimeFrame::TF_1DAY, 2);

  if (ohlcv_data.size() < 2) {
    ImGui::Text("Insufficient data for stats");
    return;
  }

  const auto &current = ohlcv_data.back();
  const auto &previous = ohlcv_data[ohlcv_data.size() - 2];

  double price_change = current.close - previous.close;
  double price_change_percent = (price_change / previous.close) * 100.0;
  double volume_change = current.volume - previous.volume;
  double volume_change_percent = (volume_change / previous.volume) * 100.0;

  ImGui::Text("Current Price: %.2f", current.close);
  ImGui::TextColored(
      price_change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
      "24h Change: %+.2f (%+.2f%%)", price_change, price_change_percent);
  ImGui::Text("24h High: %.2f", current.high);
  ImGui::Text("24h Low: %.2f", current.low);
  ImGui::Text("24h Volume: %.4f", current.volume);
  ImGui::TextColored(
      volume_change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
      "Volume Change: %+.4f (%+.2f%%)", volume_change, volume_change_percent);
}

void RealtimeDashboardComponent::render_performance_metrics_panel(
    const DashboardPanel &panel) {
  // This would integrate with performance monitoring
  // For now, show placeholder metrics
  ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
  ImGui::Text("Frame Time: %.2f ms", 1000.0f / ImGui::GetIO().Framerate);
  ImGui::Text("Memory Usage: N/A");
  ImGui::Text("Data Processed: N/A");
  ImGui::Text("Active Connections: N/A");
}

void RealtimeDashboardComponent::render_multi_symbol_overview_panel(
    const DashboardPanel &panel) {
  std::vector<std::string> symbols = {"BTC-USDT", "ETH-USDT", "BNB-USDT",
                                      "ADA-USDT", "SOL-USDT"};

  ImGui::Columns(4, "Symbols", false);
  ImGui::Text("Symbol");
  ImGui::NextColumn();
  ImGui::Text("Price");
  ImGui::NextColumn();
  ImGui::Text("24h Change");
  ImGui::NextColumn();
  ImGui::Text("Volume");
  ImGui::NextColumn();
  ImGui::Separator();

  for (const auto &symbol : symbols) {
    auto data = get_ohlcv_data(symbol, RenderEngine::TimeFrame::TF_1DAY, 2);

    ImGui::Text("%s", symbol.c_str());
    ImGui::NextColumn();

    if (data.size() >= 1) {
      ImGui::Text("%.2f", data.back().close);
      ImGui::NextColumn();

      if (data.size() >= 2) {
        double change = data.back().close - data[data.size() - 2].close;
        double change_percent = (change / data[data.size() - 2].close) * 100.0;
        ImGui::TextColored(change >= 0 ? ImVec4(0, 1, 0, 1)
                                       : ImVec4(1, 0, 0, 1),
                           "%+.2f%%", change_percent);
        ImGui::NextColumn();
        ImGui::Text("%.1fM", data.back().volume / 1000000.0);
        ImGui::NextColumn();
      } else {
        ImGui::Text("N/A");
        ImGui::NextColumn();
        ImGui::Text("%.1fM", data.back().volume / 1000000.0);
        ImGui::NextColumn();
      }
    } else {
      ImGui::Text("N/A");
      ImGui::NextColumn();
      ImGui::Text("N/A");
      ImGui::NextColumn();
      ImGui::Text("N/A");
      ImGui::NextColumn();
    }
  }
  ImGui::Columns(1);
}

void RealtimeDashboardComponent::render_panel_config_window() {
  if (ImGui::Begin("Dashboard Configuration", &show_panel_config_)) {
    if (ImGui::Button("Add Panel")) {
      ImGui::OpenPopup("AddPanelPopup");
    }

    if (ImGui::BeginPopup("AddPanelPopup")) {
      if (ImGui::Selectable("Price Chart")) {
        add_panel(DashboardPanelType::PRICE_CHART, "New Price Chart",
                  ImVec2(50, 50), ImVec2(400, 300));
      }
      if (ImGui::Selectable("Volume Chart")) {
        add_panel(DashboardPanelType::VOLUME_CHART, "New Volume Chart",
                  ImVec2(50, 50), ImVec2(400, 200));
      }
      // Add more panel types...
      ImGui::EndPopup();
    }

    ImGui::Separator();
    ImGui::Text("Existing Panels:");
    for (size_t i = 0; i < panels_.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      bool visible = panels_[i].visible;
      if (ImGui::Checkbox(("##visible" + std::to_string(i)).c_str(),
                          &visible)) {
        panels_[i].visible = visible;
      }
      ImGui::SameLine();
      ImGui::Text("%s", panels_[i].title.c_str());
      ImGui::SameLine();
      if (ImGui::Button(("Remove##" + std::to_string(i)).c_str())) {
        remove_panel(static_cast<int>(i));
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }
  }
  ImGui::End();
}

void RealtimeDashboardComponent::render_dashboard_menu() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Dashboard")) {
      if (ImGui::MenuItem("Configure Panels")) {
        show_panel_config_ = !show_panel_config_;
      }
      if (ImGui::MenuItem("Reset Layout")) {
        // Reset to default layout
        panels_.clear();
        // Re-add default panels
        add_panel(DashboardPanelType::PRICE_CHART, "BTC-USDT Price",
                  ImVec2(10, 10), ImVec2(800, 400));
        add_panel(DashboardPanelType::VOLUME_CHART, "Volume", ImVec2(820, 10),
                  ImVec2(400, 200));
        add_panel(DashboardPanelType::INDICATORS, "Technical Indicators",
                  ImVec2(10, 420), ImVec2(800, 200));
        add_panel(DashboardPanelType::ORDER_BOOK, "Order Book",
                  ImVec2(820, 220), ImVec2(400, 400));
        add_panel(DashboardPanelType::RECENT_TRADES, "Recent Trades",
                  ImVec2(10, 630), ImVec2(400, 200));
        add_panel(DashboardPanelType::MARKET_STATS, "Market Stats",
                  ImVec2(420, 630), ImVec2(400, 200));
        add_panel(DashboardPanelType::PERFORMANCE_METRICS, "Performance",
                  ImVec2(1230, 10), ImVec2(400, 300));
        add_panel(DashboardPanelType::MULTI_SYMBOL_OVERVIEW, "Multi-Symbol",
                  ImVec2(1230, 320), ImVec2(400, 300));
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

std::vector<TechnicalIndicators::OHLCV>
RealtimeDashboardComponent::get_ohlcv_data(const std::string &symbol,
                                           RenderEngine::TimeFrame timeframe,
                                           int limit) {
  // Parse symbol (assume format "SYMBOL-EXCHANGE" or just "SYMBOL")
  std::string exchange = "Binance"; // Default
  std::string symbol_name = symbol;

  size_t dash_pos = symbol.find('-');
  if (dash_pos != std::string::npos) {
    symbol_name = symbol.substr(0, dash_pos);
    exchange = symbol.substr(dash_pos + 1);
  }

  // Get symbol ID
  auto symbol_id_opt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbol_name);
  if (!symbol_id_opt) {
    return std::vector<TechnicalIndicators::OHLCV>();
  }

  // Get candles from processor
  auto candles = processor_->getCandles(*symbol_id_opt, timeframe);

  // Convert to OHLCV format and limit
  std::vector<TechnicalIndicators::OHLCV> result;
  size_t start =
      candles.size() > static_cast<size_t>(limit) ? candles.size() - limit : 0;

  for (size_t i = start; i < candles.size(); ++i) {
    TechnicalIndicators::OHLCV candle;
    candle.timestamp = candles[i].timestamp;
    candle.open = candles[i].open;
    candle.high = candles[i].high;
    candle.low = candles[i].low;
    candle.close = candles[i].close;
    candle.volume = candles[i].volume;
    result.push_back(candle);
  }

  return result;
}

RenderEngine::OrderbookData
RealtimeDashboardComponent::get_orderbook_data(const std::string &symbol) {
  // Parse symbol
  std::string exchange = "Binance";
  std::string symbol_name = symbol;

  size_t dash_pos = symbol.find('-');
  if (dash_pos != std::string::npos) {
    symbol_name = symbol.substr(0, dash_pos);
    exchange = symbol.substr(dash_pos + 1);
  }

  // Get symbol ID
  auto symbol_id_opt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbol_name);
  if (!symbol_id_opt) {
    return RenderEngine::OrderbookData();
  }

  // Get analytics which includes recent orderbooks
  auto analytics = processor_->getSymbolAnalytics(*symbol_id_opt);

  if (!analytics.recent_orderbooks.empty()) {
    return analytics.recent_orderbooks.back();
  }

  return RenderEngine::OrderbookData();
}

std::vector<RenderEngine::TradeData>
RealtimeDashboardComponent::get_recent_trades(const std::string &symbol,
                                              int limit) {
  // Parse symbol
  std::string exchange = "Binance";
  std::string symbol_name = symbol;

  size_t dash_pos = symbol.find('-');
  if (dash_pos != std::string::npos) {
    symbol_name = symbol.substr(0, dash_pos);
    exchange = symbol.substr(dash_pos + 1);
  }

  // Get symbol ID
  auto symbol_id_opt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbol_name);
  if (!symbol_id_opt) {
    return std::vector<RenderEngine::TradeData>();
  }

  // Get analytics which includes recent trades
  auto analytics = processor_->getSymbolAnalytics(*symbol_id_opt);

  std::vector<RenderEngine::TradeData> result;
  size_t start = analytics.recent_trades.size() > static_cast<size_t>(limit)
                     ? analytics.recent_trades.size() - limit
                     : 0;

  for (size_t i = start; i < analytics.recent_trades.size(); ++i) {
    result.push_back(analytics.recent_trades[i]);
  }

  return result;
}

} // namespace BTQuant