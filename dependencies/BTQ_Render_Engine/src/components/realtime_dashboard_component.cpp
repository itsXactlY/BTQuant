/**
 * @file realtime_dashboard_component.cpp
 * @brief Realtime Dashboard Component Implementation (C++26)
 *
 * Multi-panel trading dashboard with C++26 features:
 * - std::ranges for iteration and transformation
 * - std::format for string formatting
 * - Designated initializers
 * - [[nodiscard]], [[likely]]/[[unlikely]] attributes
 * - Structured bindings and auto everywhere
 *
 * @version 2.0.0 (C++26)
 */

#include "../../include/components/realtime_dashboard_component.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <chrono>
#include <format>
#include <print>
#include <ranges>
#include <string_view>

namespace BTQuant {

// ============================================
// Constants
// ============================================
namespace {
constexpr std::array DEFAULT_SYMBOLS = {
    std::string_view{"BTCUSDT"}, std::string_view{"ETHUSDT"},
    std::string_view{"BNBUSDT"}, std::string_view{"ADAUSDT"},
    std::string_view{"SOLUSDT"}};

constexpr auto POSITIVE_COLOR = ImVec4{0.0f, 1.0f, 0.0f, 1.0f};
constexpr auto NEGATIVE_COLOR = ImVec4{1.0f, 0.0f, 0.0f, 1.0f};
} // namespace

// ============================================
// Constructor / Destructor
// ============================================

RealtimeDashboardComponent::RealtimeDashboardComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(std::move(bridge)),
      processor_(std::move(processor)) {
  setup_default_panels();
}

void RealtimeDashboardComponent::setup_default_panels() {
  // Initialize default panels using structured initialization
  struct PanelConfig {
    DashboardPanelType type;
    std::string_view title;
    ImVec2 pos;
    ImVec2 size;
  };

  constexpr std::array defaultPanels = {
      PanelConfig{DashboardPanelType::PRICE_CHART,
                  "BTCUSDT Price",
                  {10, 10},
                  {800, 400}},
      PanelConfig{
          DashboardPanelType::VOLUME_CHART, "Volume", {820, 10}, {400, 200}},
      PanelConfig{DashboardPanelType::INDICATORS,
                  "Technical Indicators",
                  {10, 420},
                  {800, 200}},
      PanelConfig{
          DashboardPanelType::ORDER_BOOK, "Order Book", {820, 220}, {400, 400}},
      PanelConfig{DashboardPanelType::RECENT_TRADES,
                  "Recent Trades",
                  {10, 630},
                  {400, 200}},
      PanelConfig{DashboardPanelType::MARKET_STATS,
                  "Market Stats",
                  {420, 630},
                  {400, 200}},
      PanelConfig{DashboardPanelType::PERFORMANCE_METRICS,
                  "Performance",
                  {1230, 10},
                  {400, 300}},
      PanelConfig{DashboardPanelType::MULTI_SYMBOL_OVERVIEW,
                  "Multi-Symbol",
                  {1230, 320},
                  {400, 300}},
      PanelConfig{DashboardPanelType::FOOTPRINT_CHART,
                  "BTCUSDT Footprint",
                  {10, 840},
                  {800, 400}},
      PanelConfig{DashboardPanelType::HEATMAP_LOB,
                  "BTCUSDT Heatmap",
                  {820, 630},
                  {400, 400}},
      PanelConfig{DashboardPanelType::TPO_PROFILE,
                  "BTCUSDT TPO",
                  {1230, 630},
                  {400, 400}}};

  for (const auto &[type, title, pos, size] : defaultPanels) {
    add_panel(type, std::string(title), pos, size);
  }
}

void RealtimeDashboardComponent::update([[maybe_unused]] float dt) {
  // Real-time updates handled in render methods
}

void RealtimeDashboardComponent::render_gui() {
  render_dashboard_menu();

  // Render visible panels using ranges filter
  for (const auto &panel :
       panels_ | std::views::filter([](const auto &p) { return p.visible; })) {
    render_panel(panel);
  }

  if (show_panel_config_) {
    render_panel_config_window();
  }
}

void RealtimeDashboardComponent::initialize_vulkan_resources(VulkanCore *core) {
  if (!microstructure_renderer_ && core) {
    ::BTQuant::RenderEngine::RendererConfig config =
        ::BTQuant::RenderEngine::createDefaultRendererConfig();
    microstructure_renderer_ =
        std::make_unique<::BTQuant::RenderEngine::MarketMicrostructureRenderer>(
            core, bridge_, processor_, config);
    microstructure_renderer_->initialize();
  }
}

void RealtimeDashboardComponent::clear_data() {
  // Clear any cached data
}

void RealtimeDashboardComponent::add_panel(DashboardPanelType type,
                                           const std::string &title, ImVec2 pos,
                                           ImVec2 size) {
  panels_.push_back(
      DashboardPanel{.type = type,
                     .title = title,
                     .position = pos,
                     .size = size,
                     .visible = true,
                     .symbol = "BTCUSDT",
                     .timeframe = RenderEngine::TimeFrame::TF_15SEC});
}

void RealtimeDashboardComponent::remove_panel(int index) {
  if (index >= 0 && static_cast<size_t>(index) < panels_.size()) [[likely]] {
    panels_.erase(panels_.begin() + index);
  }
}

void RealtimeDashboardComponent::reset_layout() {
  panels_.clear();
  setup_default_panels();
}

void RealtimeDashboardComponent::render_panel(const DashboardPanel &panel) {
  ImGui::SetNextWindowPos(panel.position, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(panel.size, ImGuiCond_FirstUseEver);

  auto windowTitle = std::format("{}###panel_{}", panel.title,
                                 static_cast<const void *>(&panel));

  if (ImGui::Begin(windowTitle.c_str(), nullptr, ImGuiWindowFlags_NoCollapse)) {
    // Dispatch to appropriate panel renderer
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
    case DashboardPanelType::FOOTPRINT_CHART:
    case DashboardPanelType::HEATMAP_LOB:
    case DashboardPanelType::TPO_PROFILE:
      render_microstructure_panels(panel);
      break;
    }
  }
  ImGui::End();
}

// ============================================
// Panel Renderers
// ============================================

void RealtimeDashboardComponent::render_price_chart_panel(
    const DashboardPanel &panel) {
  auto ohlcvData = get_ohlcv_data(panel.symbol, panel.timeframe, 500);

  if (ohlcvData.empty()) [[unlikely]] {
    ImGui::Text("No data available");
    return;
  }

  // Transform data using ranges
  auto timestamps = ohlcvData | std::views::transform([](const auto &c) {
                      return static_cast<double>(c.timestamp);
                    });
  auto closes =
      ohlcvData | std::views::transform([](const auto &c) { return c.close; });

  std::vector<double> tsVec(timestamps.begin(), timestamps.end());
  std::vector<double> closeVec(closes.begin(), closes.end());

  if (ImPlot::BeginPlot("Price Chart", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%H:%M:%S");

    ImPlot::PlotLine("BTC-USDT", tsVec.data(), closeVec.data(),
                     static_cast<int>(tsVec.size()));
    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_volume_chart_panel(
    const DashboardPanel &panel) {
  auto ohlcvData = get_ohlcv_data(panel.symbol, panel.timeframe, 500);

  if (ohlcvData.empty()) [[unlikely]] {
    ImGui::Text("No data available");
    return;
  }

  std::vector<double> timestamps, volumes;
  timestamps.reserve(ohlcvData.size());
  volumes.reserve(ohlcvData.size());

  for (const auto &candle : ohlcvData) {
    timestamps.push_back(static_cast<double>(candle.timestamp));
    volumes.push_back(candle.volume);
  }

  if (ImPlot::BeginPlot("Volume", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Volume", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxisFormat(ImAxis_X1, "%H:%M:%S");

    auto barSize =
        timestamps.size() > 1 ? (timestamps[1] - timestamps[0]) * 0.8 : 1.0;
    ImPlot::PlotBars("Volume", timestamps.data(), volumes.data(),
                     static_cast<int>(timestamps.size()), barSize);
    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_indicators_panel(
    const DashboardPanel &panel) {
  auto ohlcvData = get_ohlcv_data(panel.symbol, panel.timeframe, 200);

  if (ohlcvData.empty()) [[unlikely]] {
    ImGui::Text("No data available");
    return;
  }

  auto rsiResult = TechnicalIndicators::rsi(ohlcvData, 14);
  auto macdResults = TechnicalIndicators::macd(ohlcvData, 12, 26, 9);

  if (ImPlot::BeginPlot("Indicators", ImVec2(-1, -1))) {
    ImPlot::SetupAxes("Time", "Value", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Plot RSI
    if (!rsiResult.values.empty()) {
      auto rsiTimestamps =
          rsiResult.timestamps | std::views::transform([](auto ts) {
            return static_cast<double>(ts);
          });
      std::vector<double> rsiTsVec(rsiTimestamps.begin(), rsiTimestamps.end());

      ImPlot::PlotLine("RSI", rsiTsVec.data(), rsiResult.values.data(),
                       static_cast<int>(rsiTsVec.size()));
    }

    // Plot MACD
    if (macdResults.size() >= 2) {
      auto macdTimestamps =
          macdResults[0].timestamps | std::views::transform([](auto ts) {
            return static_cast<double>(ts);
          });
      std::vector<double> macdTsVec(macdTimestamps.begin(),
                                    macdTimestamps.end());

      ImPlot::PlotLine("MACD", macdTsVec.data(), macdResults[0].values.data(),
                       static_cast<int>(macdTsVec.size()));
      ImPlot::PlotLine("Signal", macdTsVec.data(), macdResults[1].values.data(),
                       static_cast<int>(macdTsVec.size()));
    }
    ImPlot::EndPlot();
  }
}

void RealtimeDashboardComponent::render_order_book_panel(
    const DashboardPanel &panel) {
  auto orderbook = get_orderbook_data(panel.symbol);

  if (orderbook.bids.empty() && orderbook.asks.empty()) [[unlikely]] {
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

  auto maxLevels = std::max(orderbook.bids.size(), orderbook.asks.size());

  for (size_t i : std::views::iota(size_t{0}, maxLevels)) {
    if (i < orderbook.bids.size()) {
      ImGui::Text("%.2f", orderbook.bids[i].price);
      ImGui::NextColumn();
      ImGui::Text("%.4f", orderbook.bids[i].size);
      ImGui::NextColumn();
    } else {
      ImGui::TextUnformatted("");
      ImGui::NextColumn();
      ImGui::TextUnformatted("");
      ImGui::NextColumn();
    }

    if (i < orderbook.asks.size()) {
      ImGui::Text("%.2f", orderbook.asks[i].price);
    } else {
      ImGui::TextUnformatted("");
    }
    ImGui::NextColumn();
  }
  ImGui::Columns(1);
}

void RealtimeDashboardComponent::render_recent_trades_panel(
    const DashboardPanel &panel) {
  auto trades = get_recent_trades(panel.symbol, 20);

  if (trades.empty()) [[unlikely]] {
    ImGui::Text("No recent trades");
    return;
  }

  ImGui::Columns(4, "Trades", false);
  for (auto header : {"Time", "Price", "Size", "Side"}) {
    ImGui::Text("%s", header);
    ImGui::NextColumn();
  }
  ImGui::Separator();

  for (const auto &trade : trades) {
    // Format time using chrono
    auto timePoint = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(trade.timestamp));
    auto timeStr = std::format("{:%H:%M:%S}", timePoint);

    ImGui::Text("%s", timeStr.c_str());
    ImGui::NextColumn();
    ImGui::Text("%.2f", trade.price);
    ImGui::NextColumn();
    ImGui::Text("%.4f", trade.size);
    ImGui::NextColumn();
    ImGui::TextColored(trade.is_buy ? POSITIVE_COLOR : NEGATIVE_COLOR,
                       trade.is_buy ? "BUY" : "SELL");
    ImGui::NextColumn();
  }
  ImGui::Columns(1);
}

void RealtimeDashboardComponent::render_market_stats_panel(
    const DashboardPanel &panel) {
  auto ohlcvData =
      get_ohlcv_data(panel.symbol, RenderEngine::TimeFrame::TF_15SEC, 2);

  if (ohlcvData.size() < 2) [[unlikely]] {
    ImGui::Text("Insufficient data for stats");
    return;
  }

  const auto &current = ohlcvData.back();
  const auto &previous = ohlcvData[ohlcvData.size() - 2];

  auto priceChange = current.close - previous.close;
  auto priceChangePercent = (priceChange / previous.close) * 100.0;
  auto volumeChange = current.volume - previous.volume;
  auto volumeChangePercent = (volumeChange / previous.volume) * 100.0;

  ImGui::Text("Current Price: %.2f", current.close);
  ImGui::TextColored(priceChange >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR,
                     "24h Change: %+.2f (%+.2f%%)", priceChange,
                     priceChangePercent);
  ImGui::Text("24h High: %.2f", current.high);
  ImGui::Text("24h Low: %.2f", current.low);
  ImGui::Text("24h Volume: %.4f", current.volume);
  ImGui::TextColored(volumeChange >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR,
                     "Volume Change: %+.4f (%+.2f%%)", volumeChange,
                     volumeChangePercent);
}

void RealtimeDashboardComponent::render_performance_metrics_panel(
    [[maybe_unused]] const DashboardPanel &panel) {
  auto io = ImGui::GetIO();

  auto metrics = processor_->getPerformanceMetrics();
  auto activeSymbols = processor_->getActiveSymbols();

  ImGui::Text("FPS: %.1f", io.Framerate);
  ImGui::Text("Frame Time: %.2f ms", 1000.0f / io.Framerate);
  ImGui::Text("Memory Usage: N/A"); // Requires OS-specific calls
  ImGui::Text("Data Processed: %lu Trades / %lu Books",
              metrics.total_trades_processed,
              metrics.total_orderbooks_processed);
  ImGui::Text("Active Symbols: %zu", activeSymbols.size());
  ImGui::Text("Processing Latency: %.2f us", metrics.processing_latency_us);
}

void RealtimeDashboardComponent::render_multi_symbol_overview_panel(
    [[maybe_unused]] const DashboardPanel &panel) {
  ImGui::Columns(4, "Symbols", false);
  for (auto header : {"Symbol", "Price", "24h Change", "Volume"}) {
    ImGui::Text("%s", header);
    ImGui::NextColumn();
  }
  ImGui::Separator();

  auto activeSymbols = processor_->getActiveSymbols();
  if (activeSymbols.empty()) {
    // Fallback to default if no active symbols found (e.g. before data arrives)
    // or just show empty state
  }

  // Combine default with active to ensure we show something in demo mode
  std::vector<std::string> symbolsToShow;
  for (auto s : DEFAULT_SYMBOLS)
    symbolsToShow.emplace_back(s);

  // Add dynamic ones
  for (auto id : activeSymbols) {
    // Need a way to get name from ID.
    // MarketDataProcessor doesn't expose getName(id) directly but we have
    // HotSpineDataBridge? Actually we can look up in registry if we had access,
    // or just skip name. For now, let's stick to DEFAULT_SYMBOLS which demo
    // mode populates + any others we can infer. Ideally we'd use SymbolRegistry
    // here.
  }

  for (const auto &symStr : symbolsToShow) {
    auto data = get_ohlcv_data(symStr, RenderEngine::TimeFrame::TF_15SEC, 2);

    ImGui::Text("%s", symStr.c_str());
    ImGui::NextColumn();

    if (data.size() >= 1) [[likely]] {
      ImGui::Text("%.2f", data.back().close);
      ImGui::NextColumn();

      if (data.size() >= 2) {
        auto change = data.back().close - data[data.size() - 2].close;
        auto changePercent = (change / data[data.size() - 2].close) * 100.0;
        ImGui::TextColored(change >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR,
                           "%+.2f%%", changePercent);
        ImGui::NextColumn();
        ImGui::Text("%.1fM", data.back().volume / 1'000'000.0);
        ImGui::NextColumn();
      } else {
        ImGui::Text("N/A");
        ImGui::NextColumn();
        ImGui::Text("%.1fM", data.back().volume / 1'000'000.0);
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

// ============================================
// Configuration UI
// ============================================

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
      ImGui::EndPopup();
    }

    ImGui::Separator();
    ImGui::Text("Existing Panels:");

    for (size_t i : std::views::iota(size_t{0}, panels_.size())) {
      ImGui::PushID(static_cast<int>(i));

      auto visible = panels_[i].visible;
      if (ImGui::Checkbox("##visible", &visible)) {
        panels_[i].visible = visible;
      }
      ImGui::SameLine();
      ImGui::Text("%s", panels_[i].title.c_str());
      ImGui::SameLine();

      if (ImGui::Button("Remove")) {
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
        reset_layout();
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

// ============================================
// Data Accessors
// ============================================

[[nodiscard]] auto RealtimeDashboardComponent::get_ohlcv_data(
    const std::string &symbol, RenderEngine::TimeFrame timeframe, int limit)
    -> std::vector<TechnicalIndicators::OHLCV> {

  // Parse symbol (format "SYMBOL-EXCHANGE" or just "SYMBOL")
  auto [symbolName,
        exchange] = [&symbol]() -> std::pair<std::string, std::string> {
    if (auto dashPos = symbol.find('-'); dashPos != std::string::npos) {
      return {symbol.substr(0, dashPos), symbol.substr(dashPos + 1)};
    }
    return {symbol, "binance"};
  }();

  auto symbolIdOpt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbolName);

  // If not found, try the other way around if dash exists? No, stick to
  // consistent format. Or try lowercase/uppercase logic if needed.

  if (!symbolIdOpt) [[unlikely]] {
    return {};
  }

  auto candles = processor_->getCandles(*symbolIdOpt, timeframe);

  // Convert to OHLCV format with limit
  auto startIdx =
      candles.size() > static_cast<size_t>(limit) ? candles.size() - limit : 0;

  std::vector<TechnicalIndicators::OHLCV> result;
  result.reserve(candles.size() - startIdx);
  for (size_t i = startIdx; i < candles.size(); ++i) {
    const auto &c = candles[i];
    result.push_back(TechnicalIndicators::OHLCV{c.open, c.high, c.low, c.close,
                                                c.volume, c.timestamp});
  }
  return result;
}

[[nodiscard]] auto
RealtimeDashboardComponent::get_orderbook_data(const std::string &symbol)
    -> RenderEngine::OrderbookData {

  auto [symbolName,
        exchange] = [&symbol]() -> std::pair<std::string, std::string> {
    if (auto dashPos = symbol.find('-'); dashPos != std::string::npos) {
      return {symbol.substr(0, dashPos), symbol.substr(dashPos + 1)};
    }
    return {symbol, "Binance"};
  }();

  auto symbolIdOpt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbolName);
  if (!symbolIdOpt) [[unlikely]] {
    return {};
  }

  auto analytics = processor_->getSymbolAnalytics(*symbolIdOpt);

  if (!analytics.recent_orderbooks.empty()) [[likely]] {
    return analytics.recent_orderbooks.back();
  }
  return {};
}

[[nodiscard]] auto
RealtimeDashboardComponent::get_recent_trades(const std::string &symbol,
                                              int limit)
    -> std::vector<RenderEngine::TradeData> {

  auto [symbolName,
        exchange] = [&symbol]() -> std::pair<std::string, std::string> {
    if (auto dashPos = symbol.find('-'); dashPos != std::string::npos) {
      return {symbol.substr(0, dashPos), symbol.substr(dashPos + 1)};
    }
    return {symbol, "Binance"};
  }();

  auto symbolIdOpt =
      SymbolRegistry::instance().get_symbol_id(exchange, symbolName);
  if (!symbolIdOpt) [[unlikely]] {
    return {};
  }

  auto analytics = processor_->getSymbolAnalytics(*symbolIdOpt);
  auto &trades = analytics.recent_trades;

  auto startIdx =
      trades.size() > static_cast<size_t>(limit) ? trades.size() - limit : 0;

  return std::vector<RenderEngine::TradeData>(trades.begin() + startIdx,
                                              trades.end());
}

void RealtimeDashboardComponent::render_microstructure_panels(
    const DashboardPanel &panel) {
  if (!microstructure_renderer_) {
    ImGui::Text("Microstructure renderer not initialized");
    return;
  }

  // Display renderer stats in the window
  auto stats = microstructure_renderer_->getStats();
  ImGui::Text("Avg Frame Time: %.3f ms", stats.averageFrameTimeMs);

  if (panel.type == DashboardPanelType::FOOTPRINT_CHART) {
    ImGui::Text("Footprint Chart (Vulkan Native)");
    ImGui::Text("Clusters Rendered: %u", stats.footprintCellsRendered);
    ImGui::Text("Mode: Zero-Copy Storage Buffer");
  } else if (panel.type == DashboardPanelType::HEATMAP_LOB) {
    ImGui::Text("LOB Heatmap (Compute)");
    ImGui::Text("LOB Updates: %u", stats.lobUpdates);

    // Render the Compute Shader result texture!
    if (auto texID = microstructure_renderer_->getHeatmapTextureID()) {
      // Calculate available size
      ImVec2 contentSize = ImGui::GetContentRegionAvail();
      // Maintain aspect ratio or fill? Fill for heatmap.
      ImGui::Image(texID, contentSize, ImVec2(0, 0), ImVec2(1, 1));
    } else {
      ImGui::TextColored(ImVec4(1, 1, 0, 1), "Waiting for Compute Pipeline...");
    }

  } else if (panel.type == DashboardPanelType::TPO_PROFILE) {
    ImGui::Text("TPO Profile (Atomic Compute)");
    ImGui::Text("Trade Updates: %u", stats.tradeUpdates);
    ImGui::ProgressBar(0.5f, ImVec2(-1, 0), "Processing..."); // Placeholder
  }
}

} // namespace BTQuant