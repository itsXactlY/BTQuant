#include "../../include/components/chart_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <limits>

#include "../../include/components/volume_profile_panel.hpp"
#include "../../include/components/interaction_manager.hpp"
#include "../../include/components/historical_time_sales.hpp"
#include "../../include/trading/trade_command_queue.hpp"
#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"
#include "../../include/indicators/anchored_vwap.hpp"
#include "../../include/indicators/session_vwap.hpp"
#include "../../include/components/drawing_tools.hpp"
#include "../../include/components/chart_panel_settings.hpp"

namespace BTQuant {

// Helper method to calculate all indicators when new data arrives
void ChartPanel::calculate_all_indicators(const ChartInstance& chart) {
  if (chart.closes.empty()) return;

  // Calculate all enabled indicators based on indicator_config_

  // Calculate SMAs - Required: (9,20,50,200)
  if (indicator_config_.show_sma_9) calculate_cached_sma(chart.closes, 9);
  if (indicator_config_.show_sma_20) calculate_cached_sma(chart.closes, 20);
  if (indicator_config_.show_sma_50) calculate_cached_sma(chart.closes, 50);
  if (indicator_config_.show_sma_200) calculate_cached_sma(chart.closes, 200);

  // Calculate EMAs - Required: (9,21,50,200)
  if (indicator_config_.show_ema_9) calculate_cached_ema(chart.closes, 9);
  if (indicator_config_.show_ema_21) calculate_cached_ema(chart.closes, 21);
  if (indicator_config_.show_ema_50) calculate_cached_ema(chart.closes, 50);
  if (indicator_config_.show_ema_200) calculate_cached_ema(chart.closes, 200);

  // Calculate RSI
  if (indicator_config_.show_rsi) calculate_cached_rsi(chart.closes, indicator_config_.rsi_period);

  // Calculate Bollinger Bands
  if (indicator_config_.show_bollinger) {
    calculate_cached_bollinger_bands(chart.closes, indicator_config_.bollinger_period,
                                   indicator_config_.bollinger_std_dev);
  }

  // Calculate MACD
  if (indicator_config_.show_macd) {
    calculate_cached_macd(chart.closes, indicator_config_.macd_fast_period,
                        indicator_config_.macd_slow_period, indicator_config_.macd_signal_period);
  }

  // Calculate Stochastic
  if (indicator_config_.show_stochastic) {
    calculate_cached_stochastic(chart.highs, chart.lows, chart.closes,
                              indicator_config_.stochastic_k_period,
                              indicator_config_.stochastic_d_period);
  }

  // Calculate ATR
  if (indicator_config_.show_atr) {
    calculate_cached_atr(chart.highs, chart.lows, chart.closes, indicator_config_.atr_period);
  }
}

static std::string timeframe_to_string(RenderEngine::TimeFrame tf) {
  switch (tf) {
    case RenderEngine::TimeFrame::TF_1MS:
      return "1ms";
    case RenderEngine::TimeFrame::TF_10MS:
      return "10ms";
    case RenderEngine::TimeFrame::TF_100MS:
      return "100ms";
    case RenderEngine::TimeFrame::TF_500MS:
      return "500ms";
    case RenderEngine::TimeFrame::TF_1SEC:
      return "1s";
    case RenderEngine::TimeFrame::TF_3SEC:
      return "3s";
    case RenderEngine::TimeFrame::TF_5SEC:
      return "5s";
    case RenderEngine::TimeFrame::TF_15SEC:
      return "15s";
    case RenderEngine::TimeFrame::TF_30SEC:
      return "30s";
    case RenderEngine::TimeFrame::TF_1MIN:
      return "1m";
    case RenderEngine::TimeFrame::TF_2MIN:
      return "2m";
    case RenderEngine::TimeFrame::TF_5MIN:
      return "5m";
    case RenderEngine::TimeFrame::TF_15MIN:
      return "15m";
    case RenderEngine::TimeFrame::TF_30MIN:
      return "30m";
    case RenderEngine::TimeFrame::TF_1HOUR:
      return "1h";
    case RenderEngine::TimeFrame::TF_2HOUR:
      return "2h";
    case RenderEngine::TimeFrame::TF_4HOUR:
      return "4h";
    case RenderEngine::TimeFrame::TF_6HOUR:
      return "6h";
    case RenderEngine::TimeFrame::TF_12HOUR:
      return "12h";
    case RenderEngine::TimeFrame::TF_1DAY:
      return "1d";
    case RenderEngine::TimeFrame::TF_1WEEK:
      return "1w";
    default:
      return "Unknown";
  }
}

ChartPanel::ChartPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                       ChartManager* chart_manager,
                       PanelManager* panel_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor), chart_manager_(chart_manager), panel_manager_(panel_manager) {
  indicator_renderer_ = new IndicatorRenderer(nullptr, processor_);
  initialize_active_indicators();

  // Initialize the historical time & sales panel for showing trades
  if (panel_manager_) {
    // Create a temporary config for the historical time sales panel
    PanelConfig hts_config;
    hts_config.title = "Historical Time & Sales";
    hts_config.position = ImVec2(100, 100);
    hts_config.size = ImVec2(600, 400);

    historical_time_sales_panel_ = std::make_shared<HistoricalTimeSalesPanel>(hts_config, bridge_, processor_);
    historical_time_sales_panel_->set_symbol(chart_manager_->getSymbolId(symbol_).value_or(0), symbol_);
  }

  // Initialize drawing tools manager
  drawing_tools_manager_ = std::make_unique<DrawingToolsManager>();

  // Initialize panel settings
  settings_ = std::make_unique<ChartPanelSettings>(static_cast<void*>(this));
}

void ChartPanel::initialize_active_indicators() {
  // Clear existing indicators
  active_indicators_.clear();

  // Add SMA indicators - Required: (9,20,50,200)
  if (indicator_config_.show_sma_9) {
    active_indicators_.emplace_back("SMA 9", true, ImVec4(1.0f, 0.41f, 0.71f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 9.0f;
  }
  if (indicator_config_.show_sma_20) {
    active_indicators_.emplace_back("SMA 20", true, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 20.0f;
  }
  if (indicator_config_.show_sma_50) {
    active_indicators_.emplace_back("SMA 50", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 50.0f;
  }
  if (indicator_config_.show_sma_200) {
    active_indicators_.emplace_back("SMA 200", true, ImVec4(0.5f, 0.0f, 0.5f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 200.0f;
  }

  // Add EMA indicators - Required: (9,21,50,200)
  if (indicator_config_.show_ema_9) {
    active_indicators_.emplace_back("EMA 9", true, ImVec4(1.0f, 0.0f, 1.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 9.0f;
  }
  if (indicator_config_.show_ema_21) {
    active_indicators_.emplace_back("EMA 21", true, ImVec4(0.0f, 0.75f, 1.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 21.0f;
  }
  if (indicator_config_.show_ema_50) {
    active_indicators_.emplace_back("EMA 50", true, ImVec4(0.25f, 0.41f, 0.88f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 50.0f;
  }
  if (indicator_config_.show_ema_200) {
    active_indicators_.emplace_back("EMA 200", true, ImVec4(0.29f, 0.0f, 0.51f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = 200.0f;
  }

  // Add RSI indicator
  if (indicator_config_.show_rsi) {
    active_indicators_.emplace_back("RSI", true, ImVec4(0.5f, 0.5f, 0.5f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = static_cast<float>(indicator_config_.rsi_period);
  }

  // Add MACD indicator
  if (indicator_config_.show_macd) {
    active_indicators_.emplace_back("MACD", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["fast_period"] = static_cast<float>(indicator_config_.macd_fast_period);
    active_indicators_.back().parameters["slow_period"] = static_cast<float>(indicator_config_.macd_slow_period);
    active_indicators_.back().parameters["signal_period"] = static_cast<float>(indicator_config_.macd_signal_period);
  }

  // Add Bollinger Bands indicator
  if (indicator_config_.show_bollinger) {
    active_indicators_.emplace_back("Bollinger Bands", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = static_cast<float>(indicator_config_.bollinger_period);
    active_indicators_.back().parameters["std_dev"] = static_cast<float>(indicator_config_.bollinger_std_dev);
  }

  // Add Stochastic indicator
  if (indicator_config_.show_stochastic) {
    active_indicators_.emplace_back("Stochastic", true, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["k_period"] = static_cast<float>(indicator_config_.stochastic_k_period);
    active_indicators_.back().parameters["d_period"] = static_cast<float>(indicator_config_.stochastic_d_period);
  }

  // Add ATR indicator
  if (indicator_config_.show_atr) {
    active_indicators_.emplace_back("ATR", true, ImVec4(0.0f, 1.0f, 0.5f, 1.0f), next_indicator_id_++);
    active_indicators_.back().parameters["period"] = static_cast<float>(indicator_config_.atr_period);
  }

  // Add Fibonacci indicator
  if (indicator_config_.show_fibonacci) {
    active_indicators_.emplace_back("Fibonacci", true, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
  }

  // Add Volume Profile indicator
  if (indicator_config_.show_volume_profile) {
    active_indicators_.emplace_back("Volume Profile", true, ImVec4(0.7f, 0.7f, 0.7f, 1.0f), next_indicator_id_++);
  }

  // Add Crosshair Info indicator
  if (indicator_config_.show_crosshair_info) {
    active_indicators_.emplace_back("Crosshair Info", true, ImVec4(0.8f, 0.8f, 0.8f, 1.0f), next_indicator_id_++);
  }
}

void ChartPanel::initialize() {
  // Create chart
  auto id_opt = chart_manager_->getSymbolId(symbol_);
  uint32_t symbol_id = id_opt ? *id_opt : 10007;  // Default BTC-USDT
  chart_id_ = chart_manager_->create_chart(symbol_, exchange_, symbol_id, timeframe_);

  // Initialize active indicators based on current configuration
  sync_active_indicators_with_config();

  // Initialize multi-timeframe indicators
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    update_multi_timeframe_indicators(it->second);
  }
}

void ChartPanel::update(float dt) {
  // Chart manager handles updates
  (void)dt;

  // Get chart instance to check for new data
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it == charts.end()) {
    return;  // Chart not found, nothing to update
  }

  const ChartInstance& chart = it->second;

  // Invalidate cache if new data has arrived
  // NOTE: This check is lightweight and only compares sizes
  if (chart.closes.size() > last_known_data_size_) {
    // Clear all cached indicators
    cached_indicators_.clear();
    cached_sma_.clear();
    cached_ema_.clear();
    cached_rsi_.clear();
    cached_stoch_k_.clear();
    cached_atr_.clear();

    // Convert chart data to OHLCVCandle format for VWAP calculation
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      BTQuant::RenderEngine::OHLCVCandle bar;
      bar.timestamp = static_cast<uint64_t>(chart.dates[i] * 1000000); // Convert to microseconds
      bar.open = chart.opens[i];
      bar.high = chart.highs[i];
      bar.low = chart.lows[i];
      bar.close = chart.closes[i];
      bar.volume = chart.volumes[i];
      bar.trade_count = 1; // Placeholder value
      bars.push_back(bar);
    }

    // Calculate session VWAPs based on the chart data
    session_vwap_.calculate(bars);

    last_known_data_size_ = chart.closes.size();

    // Pre-calculate all enabled indicators with new data
    calculate_all_indicators(chart);
  }

  // Update multi-timeframe indicators when new data arrives
  // This ensures that multi-timeframe indicators remain synchronized with the chart data
  update_multi_timeframe_indicators(chart);

  // Update liquidity data periodically (every frame for real-time updates)
  if (show_liquidity_bars_) {
    update_liquidity_data();
  }
}

void ChartPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Get chart instance
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it == charts.end()) {
    ImGui::Text("Chart not found");
    end_panel_window();
    return;
  }

  const ChartInstance& chart = it->second;

  // ========================================================================
  // QUANTOWER-STYLE 5-PART LAYOUT (Phase 3)
  // ========================================================================
  
  // 3.1 Top Toolbar (Main Controls)
  render_top_toolbar();
  
  // Create a horizontal layout: Left Sidebar | Main Chart Area | Right Sidebar
  ImGui::BeginChild("MainChartArea", ImVec2(0, -30));  // Reserve space for bottom toolbar
  
  // 3.2 Left Sidebar (Tools & Objects)
  render_left_sidebar();
  
  ImGui::SameLine();
  
  // Main Chart Content Area
  ImGui::BeginChild("ChartContent", ImVec2(-120, 0));  // Reserve space for right sidebar
    
    // Render chart controls in a collapsible header (legacy - can be hidden in Quantower mode)
    if (ImGui::CollapsingHeader("Chart Controls")) {
      render_chart_controls();
    }

    // Render liquidity bars controls in a collapsible header
    if (ImGui::CollapsingHeader("Liquidity Bars")) {
      render_liquidity_bars_controls();
    }

    // Render indicator selector
    if (ImGui::CollapsingHeader("Indicators")) {
      render_indicator_selector();
    }

    // Render drawing tools controls
    if (ImGui::CollapsingHeader("Drawing Tools")) {
      if (drawing_tools_manager_) {
        drawing_tools_manager_->render_ui_controls();
      }
    }

    // Render chart with indicators
    render_instrument_chart(chart);
    
  ImGui::EndChild();
  
  ImGui::SameLine();
  
  // 3.4 Right Sidebar Order Entry
  render_right_sidebar_order_entry();
  
  ImGui::EndChild();  // End MainChartArea
  
  // 3.5 Bottom Toolbar (Volume Analysis)
  render_bottom_toolbar();

  end_panel_window();

  // Render the indicator overlay panel
  render_indicator_overlay_panel();

  // Render the trades popup if needed
  render_trades_popup();

  // Render the historical time & sales popup if needed
  if (historical_time_sales_panel_ && show_trades_popup_) {
    historical_time_sales_panel_->show_trades_popup(clicked_bar_start_time_, clicked_bar_end_time_, symbol_);
  }

  // Render settings modal if available
  if (settings_) {
    settings_->render();
  }
}

void ChartPanel::set_symbol(const std::string& symbol, const std::string& exchange) {
  symbol_ = symbol;
  exchange_ = exchange;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new symbol
  // Don't destroy old one, so we can switch back to it with state preserved
  initialize();

  // Update the historical time & sales panel with the new symbol
  if (historical_time_sales_panel_) {
    auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
    if (symbol_id_opt) {
      historical_time_sales_panel_->set_symbol(*symbol_id_opt, symbol_);
    }
  }
}

void ChartPanel::set_timeframe(RenderEngine::TimeFrame timeframe) {
  RenderEngine::TimeFrame old_timeframe = timeframe_;
  (void)old_timeframe;  // Suppress unused variable warning
  timeframe_ = timeframe;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new timeframe
  // Don't destroy old one, so we can switch back to it with state preserved
  initialize();

  // Update multi-timeframe indicators when timeframe changes
  // This ensures that the multi-timeframe indicators are recalculated to align with the new chart timeframe
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    update_multi_timeframe_indicators(it->second);
  }
}

void ChartPanel::open_indicator_dialog() {
  // Open the indicator dialog by showing the indicator selector section
  // This could be implemented by setting a flag to show the indicator dialog
  // or by opening a dedicated indicator management window

  // For now, we'll simulate this by toggling the visibility of the indicator selector
  // In a real implementation, this would open a dedicated dialog window
  ImGui::OpenPopup("IndicatorDialog");

  // Show a simple indicator dialog popup
  if (ImGui::BeginPopupModal("IndicatorDialog", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Select Indicators to Add:");
    ImGui::Separator();

    // Add common indicators that can be selected
    if (ImGui::Button("SMA")) {
      indicator_config_.show_sma_9 = true;
      sync_active_indicators_with_config();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("EMA")) {
      indicator_config_.show_ema_9 = true;
      sync_active_indicators_with_config();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("RSI")) {
      indicator_config_.show_rsi = true;
      sync_active_indicators_with_config();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("MACD")) {
      indicator_config_.show_macd = true;
      sync_active_indicators_with_config();
      ImGui::CloseCurrentPopup();
    }

    ImGui::Separator();
    if (ImGui::Button("Cancel")) {
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }
}

void ChartPanel::reset_view() {
  // Reset the chart view to show all available data
  follow_latest_ = false;

  // Reset view limits to show the full range of data
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    const ChartInstance& chart = it->second;

    if (!chart.dates.empty()) {
      last_view_min_ = chart.dates.front();
      last_view_max_ = chart.dates.back();

      // Add a small margin to the view
      double margin = (last_view_max_ - last_view_min_) * 0.05; // 5% margin
      last_view_min_ -= margin;
      last_view_max_ += margin;
    }
  }
}

void ChartPanel::export_data() {
  // Export chart data to a file
  // This would typically open a file dialog and save the chart data in a specified format

  // For now, we'll just print a message to indicate the export action
  std::cout << "[ChartPanel] Exporting data for symbol: " << symbol_
            << " with timeframe: " << timeframe_to_string(timeframe_) << std::endl;

  // Get chart data to export
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    const ChartInstance& chart = it->second;

    if (!chart.dates.empty()) {
      // In a real implementation, this would open a file dialog and export the data
      // For now, we'll just log the data size
      std::cout << "[ChartPanel] Data points to export: " << chart.dates.size() << std::endl;

      // Example: Export to CSV format
      // This would typically use a file dialog to let the user choose the destination
      std::string filename = symbol_ + "_" + timeframe_to_string(timeframe_) + ".csv";
      std::cout << "[ChartPanel] Would export to: " << filename << std::endl;

      // In a real implementation, we would write the actual data to the file
      // Format would be: timestamp,open,high,low,close,volume
    }
  }
}

void ChartPanel::render_chart_controls() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Symbol input
  static char symbol_input[32] = "BTC-USDT";
  std::strncpy(symbol_input, symbol_.c_str(), sizeof(symbol_input) - 1);
  if (ImGui::InputText("Symbol", symbol_input, sizeof(symbol_input))) {
    set_symbol(symbol_input, exchange_);
  }
  ImGui::SameLine();

  // Timeframe selector (Extended to include higher timeframes for multi-timeframe analysis)
  const char* timeframes[] = {"1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"};
  int selected = static_cast<int>(timeframe_);
  if (ImGui::Combo("Timeframe", &selected, timeframes, IM_ARRAYSIZE(timeframes))) {
    set_timeframe(static_cast<RenderEngine::TimeFrame>(selected));
  }

  // Auto-follow Window Size
  ImGui::SameLine();
  int window_size_int = static_cast<int>(auto_follow_window_);
  if (ImGui::SliderInt("Window", &window_size_int, 100, 10000, "%d")) {
    auto_follow_window_ = static_cast<float>(window_size_int);
  }

  // Auto-follow checkbox
  ImGui::SameLine();
  ImGui::Checkbox("Auto-follow", &follow_latest_);


  ImGui::PopStyleVar();
}

void ChartPanel::render_liquidity_bars_controls() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Toggle for showing liquidity bars
  ImGui::Checkbox("Show Liquidity Bars", &show_liquidity_bars_);

  // Slider for adjusting bar width
  ImGui::SliderFloat("Bar Width", &liquidity_bar_width_, 5.0f, 30.0f, "%.1f px");

  // Slider for adjusting opacity
  ImGui::SliderFloat("Opacity", &liquidity_bar_opacity_, 0.1f, 1.0f, "%.2f");

  // Color pickers for bid and ask colors
  ImGui::Text("Bid Color (Green):");
  ImGui::ColorEdit4("##BidColor", &liquidity_bids_color_.x, ImGuiColorEditFlags_NoInputs);
  ImGui::Text("Ask Color (Red):");
  ImGui::ColorEdit4("##AskColor", &liquidity_asks_color_.x, ImGuiColorEditFlags_NoInputs);

  ImGui::PopStyleVar();
}

void ChartPanel::render_indicator_selector() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Moving Averages - Required: (9,20,50,200)
  ImGui::SeparatorText("Moving Averages");
  if (ImGui::Checkbox("SMA 9", &indicator_config_.show_sma_9)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("SMA 20", &indicator_config_.show_sma_20)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("SMA 50", &indicator_config_.show_sma_50)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("SMA 200", &indicator_config_.show_sma_200)) {
    sync_active_indicators_with_config();
  }

  if (ImGui::Checkbox("EMA 9", &indicator_config_.show_ema_9)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("EMA 21", &indicator_config_.show_ema_21)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("EMA 50", &indicator_config_.show_ema_50)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("EMA 200", &indicator_config_.show_ema_200)) {
    sync_active_indicators_with_config();
  }

  // Oscillators
  ImGui::SeparatorText("Oscillators");
  if (ImGui::Checkbox("RSI", &indicator_config_.show_rsi)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("MACD", &indicator_config_.show_macd)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Stochastic", &indicator_config_.show_stochastic)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("ATR", &indicator_config_.show_atr)) {
    sync_active_indicators_with_config();
  }

  // Overlays
  ImGui::SeparatorText("Overlays");
  if (ImGui::Checkbox("Bollinger Bands", &indicator_config_.show_bollinger)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Fibonacci", &indicator_config_.show_fibonacci)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Vol Profile", &indicator_config_.show_volume_profile)) {
    sync_active_indicators_with_config();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("Crosshair Info", &indicator_config_.show_crosshair_info)) {
    sync_active_indicators_with_config();
  }

  ImGui::PopStyleVar();
}

// ============================================================================
// INDICATOR CALCULATION HELPERS
// ============================================================================

std::vector<double> ChartPanel::calculate_cached_sma(const std::vector<float>& prices, int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{IndicatorType::SMA, prices.size(), period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  // Calculate SMA if not cached
  std::vector<double> sma(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < period; ++j) {
      sum += static_cast<double>(prices[i - j]);
    }
    sma[i] = sum / period;
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = sma;
  return sma;
}

std::vector<double> ChartPanel::calculate_cached_ema(const std::vector<float>& prices, int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{IndicatorType::EMA, prices.size(), period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> ema(prices.size(), 0.0);

  // Initialize with SMA
  double sum = 0.0;
  for (int i = 0; i < std::min(period, static_cast<int>(prices.size())); ++i) {
    sum += static_cast<double>(prices[i]);
  }
  ema[period - 1] = sum / period;

  // Calculate EMA
  double multiplier = 2.0 / (period + 1.0);
  for (size_t i = period; i < prices.size(); ++i) {
    ema[i] = (static_cast<double>(prices[i]) - ema[i - 1]) * multiplier + ema[i - 1];
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = ema;
  return ema;
}

std::vector<double> ChartPanel::calculate_cached_rsi(const std::vector<float>& prices, int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{IndicatorType::RSI, prices.size(), period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> rsi(prices.size(), 50.0);  // Default to neutral

  if (prices.size() < static_cast<size_t>(period + 1)) {
    // Cache the result even if it's empty/default
    cached_indicators_[key] = rsi;
    return rsi;
  }

  for (size_t i = period; i < prices.size(); ++i) {
    double gains = 0.0;
    double losses = 0.0;

    for (int j = 1; j <= period; ++j) {
      double change = static_cast<double>(prices[i - j + 1]) - static_cast<double>(prices[i - j]);
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    double avg_gain = gains / period;
    double avg_loss = -losses / period;

    double rs = avg_loss == 0.0 ? 100.0 : avg_gain / avg_loss;
    rsi[i] = 100.0 - (100.0 / (1.0 + rs));
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = rsi;
  return rsi;
}

std::vector<double> ChartPanel::calculate_cached_bollinger_upper(const std::vector<float>& prices,
                                                          int period, double std_dev) {
  auto sma = calculate_cached_sma(prices, period);
  std::vector<double> upper_band(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int j = 0; j < period; ++j) {
      double diff = static_cast<double>(prices[i - j]) - sma[i];
      sum_sq_diff += diff * diff;
    }
    double variance = sum_sq_diff / period;
    double std_deviation = std::sqrt(variance);

    upper_band[i] = sma[i] + std_dev * std_deviation;
  }

  // Cache the result in the unified cache
  IndicatorCacheKey key{IndicatorType::BB_UPPER, prices.size(), period, 0, std_dev};
  cached_indicators_[key] = upper_band;

  return upper_band;
}

std::vector<double> ChartPanel::calculate_cached_bollinger_middle(const std::vector<float>& prices,
                                                          int period) {
  auto sma = calculate_cached_sma(prices, period);

  // Cache the result in the unified cache
  IndicatorCacheKey key{IndicatorType::BB_MIDDLE, prices.size(), period, 0, 0.0};
  cached_indicators_[key] = sma;

  return sma;
}

std::vector<double> ChartPanel::calculate_cached_bollinger_lower(const std::vector<float>& prices,
                                                          int period, double std_dev) {
  auto sma = calculate_cached_sma(prices, period);
  std::vector<double> lower_band(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int j = 0; j < period; ++j) {
      double diff = static_cast<double>(prices[i - j]) - sma[i];
      sum_sq_diff += diff * diff;
    }
    double variance = sum_sq_diff / period;
    double std_deviation = std::sqrt(variance);

    lower_band[i] = sma[i] - std_dev * std_deviation;
  }

  // Cache the result in the unified cache
  IndicatorCacheKey key{IndicatorType::BB_LOWER, prices.size(), period, 0, std_dev};
  cached_indicators_[key] = lower_band;

  return lower_band;
}

std::vector<double> ChartPanel::calculate_cached_macd_line(const std::vector<float>& prices, int fast,
                                                    int slow) {
  auto ema_fast = calculate_cached_ema(prices, fast);
  auto ema_slow = calculate_cached_ema(prices, slow);

  std::vector<double> macd_line(prices.size(), 0.0);
  for (size_t i = 0; i < prices.size(); ++i) {
    macd_line[i] = ema_fast[i] - ema_slow[i];
  }

  // Cache the result in the unified cache
  IndicatorCacheKey key{IndicatorType::MACD_LINE, prices.size(), fast, slow, 0.0};
  cached_indicators_[key] = macd_line;

  return macd_line;
}

std::vector<double> ChartPanel::calculate_cached_macd_signal(const std::vector<double>& macd_line,
                                                      int signal) {
  // Create cache key based on the macd_line size and signal period
  IndicatorCacheKey key{IndicatorType::MACD_SIGNAL, macd_line.size(), signal, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> signal_line = calculate_ema(macd_line, signal);

  // Cache the result in the unified cache
  cached_indicators_[key] = signal_line;
  return signal_line;
}

std::vector<double> ChartPanel::calculate_cached_macd_histogram(const std::vector<double>& macd_line,
                                                         const std::vector<double>& signal) {
  std::vector<double> histogram(macd_line.size(), 0.0);

  for (size_t i = 0; i < macd_line.size(); ++i) {
    histogram[i] = macd_line[i] - signal[i];
  }

  // Cache the result in the unified cache
  IndicatorCacheKey key{IndicatorType::MACD_HISTOGRAM, macd_line.size(), 0, 0, 0.0};
  cached_indicators_[key] = histogram;

  return histogram;
}

std::vector<double> ChartPanel::calculate_cached_stochastic_k(const std::vector<float>& highs,
                                                       const std::vector<float>& lows,
                                                       const std::vector<float>& closes,
                                                       int k_period) {
  if (highs.empty() || lows.empty() || closes.empty()) {
    return std::vector<double>();
  }

  // Create cache key based on data size and k_period
  IndicatorCacheKey key{IndicatorType::STOCH_K, highs.size(), k_period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> stoch_k(highs.size(), 50.0); // Default to neutral

  for (size_t i = k_period - 1; i < highs.size(); ++i) {
    float highest_high = highs[i];
    float lowest_low = lows[i];

    // Find highest high and lowest low in the k_period
    for (int j = 0; j < k_period; ++j) {
      if (i >= static_cast<size_t>(j)) {
        highest_high = std::max(highest_high, highs[i - j]);
        lowest_low = std::min(lowest_low, lows[i - j]);
      }
    }

    // Calculate %K
    if (highest_high != lowest_low) {
      stoch_k[i] = ((static_cast<double>(closes[i]) - lowest_low) / (highest_high - lowest_low)) * 100.0;
    } else {
      stoch_k[i] = 50.0; // Neutral if high equals low
    }
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = stoch_k;
  return stoch_k;
}

std::vector<double> ChartPanel::calculate_cached_stochastic_d(const std::vector<double>& stoch_k,
                                                       int slow_period) {
  if (stoch_k.empty()) {
    return std::vector<double>();
  }

  // Create cache key based on data size and slow_period
  IndicatorCacheKey key{IndicatorType::STOCH_D, stoch_k.size(), slow_period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> stoch_d(stoch_k.size(), 50.0); // Default to neutral

  for (size_t i = slow_period - 1; i < stoch_k.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < slow_period; ++j) {
      if (i >= static_cast<size_t>(j)) {
        sum += stoch_k[i - j];
      }
    }
    stoch_d[i] = sum / slow_period;
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = stoch_d;
  return stoch_d;
}

std::vector<double> ChartPanel::calculate_cached_true_range(const std::vector<float>& highs,
                                                     const std::vector<float>& lows,
                                                     const std::vector<float>& closes) {
  if (highs.empty() || lows.empty() || closes.empty()) {
    return std::vector<double>();
  }

  // Create cache key based on data size
  IndicatorCacheKey key{IndicatorType::TRUE_RANGE, highs.size(), 0, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> tr(highs.size(), 0.0);

  for (size_t i = 0; i < highs.size(); ++i) {
    if (i == 0) {
      // For the first period, use high - low
      tr[i] = static_cast<double>(highs[i]) - static_cast<double>(lows[i]);
    } else {
      // True Range is the maximum of:
      // 1. Current High - Current Low
      // 2. Absolute value of Current High - Previous Close
      // 3. Absolute value of Current Low - Previous Close
      double hl = static_cast<double>(highs[i]) - static_cast<double>(lows[i]);
      double hc = std::abs(static_cast<double>(highs[i]) - static_cast<double>(closes[i - 1]));
      double lc = std::abs(static_cast<double>(lows[i]) - static_cast<double>(closes[i - 1]));

      tr[i] = std::max({hl, hc, lc});
    }
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = tr;
  return tr;
}

std::vector<double> ChartPanel::calculate_cached_atr(const std::vector<float>& highs,
                                              const std::vector<float>& lows,
                                              const std::vector<float>& closes,
                                              int period) {
  if (highs.empty() || lows.empty() || closes.empty()) {
    return std::vector<double>();
  }

  // Create cache key based on data size and period
  IndicatorCacheKey key{IndicatorType::ATR, highs.size(), period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  auto tr = calculate_cached_true_range(highs, lows, closes);
  if (tr.empty()) {
    return std::vector<double>();
  }

  std::vector<double> atr(highs.size(), 0.0);

  // Calculate initial ATR using Simple Moving Average for the first value
  double sum = 0.0;
  for (int i = 0; i < period && i < static_cast<int>(tr.size()); ++i) {
    sum += tr[i];
  }

  if (static_cast<int>(tr.size()) >= period) {
    atr[period - 1] = sum / period;

    // Calculate remaining ATR values using Wilder's smoothing method
    for (size_t i = period; i < tr.size(); ++i) {
      // ATR = [(previous ATR) * (period - 1) + current TR] / period
      atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period;
    }
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = atr;
  return atr;
}

// Helper methods to calculate and cache combined indicators
void ChartPanel::calculate_cached_bollinger_bands(const std::vector<float>& prices, int period, double std_dev) {
  // This method calculates and caches all three Bollinger Band components
  calculate_cached_bollinger_upper(prices, period, std_dev);
  calculate_cached_bollinger_middle(prices, period);
  calculate_cached_bollinger_lower(prices, period, std_dev);
}

void ChartPanel::calculate_cached_macd(const std::vector<float>& prices, int fast, int slow, int signal) {
  // This method calculates and caches all three MACD components
  auto macd_line = calculate_cached_macd_line(prices, fast, slow);
  auto macd_signal = calculate_cached_macd_signal(macd_line, signal);
  calculate_cached_macd_histogram(macd_line, macd_signal);
}

void ChartPanel::calculate_cached_stochastic(const std::vector<float>& highs,
                                          const std::vector<float>& lows,
                                          const std::vector<float>& closes,
                                          int k_period, int d_period) {
  (void)d_period;  // Suppress unused parameter warning
  // This method calculates and caches both stochastic components
  auto stoch_k = calculate_cached_stochastic_k(highs, lows, closes, k_period);
  calculate_cached_stochastic_d(stoch_k, indicator_config_.stochastic_slow_period);
}

void ChartPanel::update_indicator_config_from_active() {
  // Reset all indicators to false initially
  indicator_config_.show_sma_9 = false;
  indicator_config_.show_sma_10 = false;  // Not required by spec
  indicator_config_.show_sma_20 = false;
  indicator_config_.show_sma_50 = false;
  indicator_config_.show_sma_200 = false;
  indicator_config_.show_ema_9 = false;
  indicator_config_.show_ema_10 = false;  // Not required by spec
  indicator_config_.show_ema_20 = false;  // Not required by spec
  indicator_config_.show_ema_21 = false;
  indicator_config_.show_ema_50 = false;
  indicator_config_.show_ema_200 = false;
  indicator_config_.show_rsi = false;
  indicator_config_.show_macd = false;
  indicator_config_.show_bollinger = false;
  indicator_config_.show_stochastic = false;
  indicator_config_.show_atr = false;
  indicator_config_.show_fibonacci = false;
  indicator_config_.show_volume_profile = false;
  indicator_config_.show_crosshair_info = false;

  // Update configuration based on active indicators
  for (const auto& indicator : active_indicators_) {
    if (!indicator.isVisible) continue;  // Skip invisible indicators

    if (indicator.name == "SMA 9") {
      indicator_config_.show_sma_9 = true;
    } else if (indicator.name == "SMA 20") {
      indicator_config_.show_sma_20 = true;
    } else if (indicator.name == "SMA 50") {
      indicator_config_.show_sma_50 = true;
    } else if (indicator.name == "SMA 200") {
      indicator_config_.show_sma_200 = true;
    } else if (indicator.name == "EMA 9") {
      indicator_config_.show_ema_9 = true;
    } else if (indicator.name == "EMA 21") {
      indicator_config_.show_ema_21 = true;
    } else if (indicator.name == "EMA 50") {
      indicator_config_.show_ema_50 = true;
    } else if (indicator.name == "EMA 200") {
      indicator_config_.show_ema_200 = true;
    } else if (indicator.name == "RSI") {
      indicator_config_.show_rsi = true;
      if (indicator.parameters.count("period") > 0) {
        indicator_config_.rsi_period = static_cast<int>(indicator.parameters.at("period"));
      }
    } else if (indicator.name == "MACD") {
      indicator_config_.show_macd = true;
      if (indicator.parameters.count("fast_period") > 0) {
        indicator_config_.macd_fast_period = static_cast<int>(indicator.parameters.at("fast_period"));
      }
      if (indicator.parameters.count("slow_period") > 0) {
        indicator_config_.macd_slow_period = static_cast<int>(indicator.parameters.at("slow_period"));
      }
      if (indicator.parameters.count("signal_period") > 0) {
        indicator_config_.macd_signal_period = static_cast<int>(indicator.parameters.at("signal_period"));
      }
    } else if (indicator.name == "Bollinger Bands") {
      indicator_config_.show_bollinger = true;
      if (indicator.parameters.count("period") > 0) {
        indicator_config_.bollinger_period = static_cast<int>(indicator.parameters.at("period"));
      }
      if (indicator.parameters.count("std_dev") > 0) {
        indicator_config_.bollinger_std_dev = indicator.parameters.at("std_dev");
      }
    } else if (indicator.name == "Stochastic") {
      indicator_config_.show_stochastic = true;
      if (indicator.parameters.count("k_period") > 0) {
        indicator_config_.stochastic_k_period = static_cast<int>(indicator.parameters.at("k_period"));
      }
      if (indicator.parameters.count("d_period") > 0) {
        indicator_config_.stochastic_d_period = static_cast<int>(indicator.parameters.at("d_period"));
      }
      if (indicator.parameters.count("slow_period") > 0) {
        indicator_config_.stochastic_slow_period = static_cast<int>(indicator.parameters.at("slow_period"));
      }
    } else if (indicator.name == "ATR") {
      indicator_config_.show_atr = true;
      if (indicator.parameters.count("period") > 0) {
        indicator_config_.atr_period = static_cast<int>(indicator.parameters.at("period"));
      }
    } else if (indicator.name == "Fibonacci") {
      indicator_config_.show_fibonacci = true;
    } else if (indicator.name == "Volume Profile") {
      indicator_config_.show_volume_profile = true;
    } else if (indicator.name == "Crosshair Info") {
      indicator_config_.show_crosshair_info = true;
    }
  }
}

void ChartPanel::sync_active_indicators_with_config() {
  // Create a temporary list to hold the new active indicators
  std::vector<IndicatorItem> new_active_indicators;

  // Add SMA indicators - Required: (9,20,50,200)
  if (indicator_config_.show_sma_9) {
    // Check if already exists in active_indicators_
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "SMA 9"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it); // Keep existing one
    } else {
      new_active_indicators.emplace_back("SMA 9", true, ImVec4(1.0f, 0.41f, 0.71f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 9.0f;
    }
  }
  if (indicator_config_.show_sma_20) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "SMA 20"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("SMA 20", true, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 20.0f;
    }
  }
  if (indicator_config_.show_sma_50) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "SMA 50"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("SMA 50", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 50.0f;
    }
  }
  if (indicator_config_.show_sma_200) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "SMA 200"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("SMA 200", true, ImVec4(0.5f, 0.0f, 0.5f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 200.0f;
    }
  }

  // Add EMA indicators - Required: (9,21,50,200)
  if (indicator_config_.show_ema_9) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "EMA 9"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("EMA 9", true, ImVec4(1.0f, 0.0f, 1.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 9.0f;
    }
  }
  if (indicator_config_.show_ema_21) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "EMA 21"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("EMA 21", true, ImVec4(0.0f, 0.75f, 1.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 21.0f;
    }
  }
  if (indicator_config_.show_ema_50) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "EMA 50"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("EMA 50", true, ImVec4(0.25f, 0.41f, 0.88f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 50.0f;
    }
  }
  if (indicator_config_.show_ema_200) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "EMA 200"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("EMA 200", true, ImVec4(0.29f, 0.0f, 0.51f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = 200.0f;
    }
  }

  // Add RSI indicator
  if (indicator_config_.show_rsi) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "RSI"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("RSI", true, ImVec4(0.5f, 0.5f, 0.5f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = static_cast<float>(indicator_config_.rsi_period);
    }
  }

  // Add MACD indicator
  if (indicator_config_.show_macd) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "MACD"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("MACD", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["fast_period"] = static_cast<float>(indicator_config_.macd_fast_period);
      new_active_indicators.back().parameters["slow_period"] = static_cast<float>(indicator_config_.macd_slow_period);
      new_active_indicators.back().parameters["signal_period"] = static_cast<float>(indicator_config_.macd_signal_period);
    }
  }

  // Add Bollinger Bands indicator
  if (indicator_config_.show_bollinger) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "Bollinger Bands"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("Bollinger Bands", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = static_cast<float>(indicator_config_.bollinger_period);
      new_active_indicators.back().parameters["std_dev"] = static_cast<float>(indicator_config_.bollinger_std_dev);
    }
  }

  // Add Stochastic indicator
  if (indicator_config_.show_stochastic) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "Stochastic"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("Stochastic", true, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["k_period"] = static_cast<float>(indicator_config_.stochastic_k_period);
      new_active_indicators.back().parameters["d_period"] = static_cast<float>(indicator_config_.stochastic_d_period);
      new_active_indicators.back().parameters["slow_period"] = static_cast<float>(indicator_config_.stochastic_slow_period);
    }
  }

  // Add ATR indicator
  if (indicator_config_.show_atr) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "ATR"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("ATR", true, ImVec4(0.0f, 1.0f, 0.5f, 1.0f), next_indicator_id_++);
      new_active_indicators.back().parameters["period"] = static_cast<float>(indicator_config_.atr_period);
    }
  }

  // Add Fibonacci indicator
  if (indicator_config_.show_fibonacci) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "Fibonacci"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("Fibonacci", true, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
    }
  }

  // Add Volume Profile indicator
  if (indicator_config_.show_volume_profile) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "Volume Profile"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("Volume Profile", true, ImVec4(0.7f, 0.7f, 0.7f, 1.0f), next_indicator_id_++);
    }
  }

  // Add Crosshair Info indicator
  if (indicator_config_.show_crosshair_info) {
    auto it = std::find_if(active_indicators_.begin(), active_indicators_.end(),
                          [](const IndicatorItem& item) { return item.name == "Crosshair Info"; });
    if (it != active_indicators_.end()) {
      new_active_indicators.push_back(*it);
    } else {
      new_active_indicators.emplace_back("Crosshair Info", true, ImVec4(0.8f, 0.8f, 0.8f, 1.0f), next_indicator_id_++);
    }
  }

  // Update the active indicators list
  active_indicators_ = std::move(new_active_indicators);
}

std::vector<double> ChartPanel::calculate_sma(const std::vector<float>& prices, int period) {
  // Redirect to the cached version
  return calculate_cached_sma(prices, period);
}

std::vector<double> ChartPanel::calculate_ema(const std::vector<float>& prices, int period) {
  // Redirect to the cached version
  return calculate_cached_ema(prices, period);
}

std::vector<double> ChartPanel::calculate_ema(const std::vector<double>& prices, int period) {
  // Redirect to the cached version (using the unified cache)
  // Create cache key based on data size and period
  IndicatorCacheKey key{IndicatorType::EMA, prices.size(), period, 0, 0.0};

  // Check if result is already cached in the unified cache
  auto it = cached_indicators_.find(key);
  if (it != cached_indicators_.end()) {
    return it->second;
  }

  std::vector<double> ema(prices.size(), 0.0);

  // Initialize with SMA
  double sum = 0.0;
  for (int i = 0; i < std::min(period, static_cast<int>(prices.size())); ++i) {
    sum += prices[i];
  }
  ema[period - 1] = sum / period;

  // Calculate EMA
  double multiplier = 2.0 / (period + 1.0);
  for (size_t i = period; i < prices.size(); ++i) {
    ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
  }

  // Cache the result in the unified cache
  cached_indicators_[key] = ema;
  return ema;
}

void ChartPanel::render_indicator_overlay_panel() {
  // Create a window for the indicator overlay panel
  const char* overlay_title = "Active Indicators";
  ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
  ImGui::Begin(overlay_title, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);

  // Add a button to add new indicators
  if (ImGui::Button("Add Indicator")) {
    ImGui::OpenPopup("AddIndicatorPopup");
  }

  // Add indicator popup
  if (ImGui::BeginPopup("AddIndicatorPopup")) {
    if (ImGui::Selectable("SMA")) {
      active_indicators_.emplace_back("SMA", true, ImVec4(1.0f, 0.41f, 0.71f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["period"] = 9.0f;
    }
    if (ImGui::Selectable("EMA")) {
      active_indicators_.emplace_back("EMA", true, ImVec4(1.0f, 0.0f, 1.0f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["period"] = 9.0f;
    }
    if (ImGui::Selectable("RSI")) {
      active_indicators_.emplace_back("RSI", true, ImVec4(0.5f, 0.5f, 0.5f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["period"] = 14.0f;
    }
    if (ImGui::Selectable("MACD")) {
      active_indicators_.emplace_back("MACD", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["fast_period"] = 12.0f;
      active_indicators_.back().parameters["slow_period"] = 26.0f;
      active_indicators_.back().parameters["signal_period"] = 9.0f;
    }
    if (ImGui::Selectable("Bollinger Bands")) {
      active_indicators_.emplace_back("Bollinger Bands", true, ImVec4(0.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["period"] = 20.0f;
      active_indicators_.back().parameters["std_dev"] = 2.0f;
    }
    if (ImGui::Selectable("Stochastic")) {
      active_indicators_.emplace_back("Stochastic", true, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["k_period"] = 14.0f;
      active_indicators_.back().parameters["d_period"] = 3.0f;
      active_indicators_.back().parameters["slow_period"] = 3.0f;
    }
    if (ImGui::Selectable("ATR")) {
      active_indicators_.emplace_back("ATR", true, ImVec4(0.0f, 1.0f, 0.5f, 1.0f), next_indicator_id_++);
      active_indicators_.back().parameters["period"] = 14.0f;
    }
    if (ImGui::Selectable("Fibonacci")) {
      active_indicators_.emplace_back("Fibonacci", true, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), next_indicator_id_++);
    }
    if (ImGui::Selectable("Volume Profile")) {
      active_indicators_.emplace_back("Volume Profile", true, ImVec4(0.7f, 0.7f, 0.7f, 1.0f), next_indicator_id_++);
    }
    if (ImGui::Selectable("Crosshair Info")) {
      active_indicators_.emplace_back("Crosshair Info", true, ImVec4(0.8f, 0.8f, 0.8f, 1.0f), next_indicator_id_++);
    }

    ImGui::EndPopup();
  }

  // Add multi-timeframe indicator button
  ImGui::SameLine();
  if (ImGui::Button("Add Multi-TF Indicator")) {
    ImGui::OpenPopup("AddMultiTFIndicatorPopup");
  }

  // Add multi-timeframe indicator popup
  if (ImGui::BeginPopup("AddMultiTFIndicatorPopup")) {
    if (ImGui::Selectable("Daily SMA")) {
      // Get symbol ID to fetch data from daily timeframe
      auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
      if (symbol_id_opt) {
        // Add a daily SMA indicator that will be displayed on the current chart
        add_multi_timeframe_indicator("Daily SMA 20", true, ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 20, RenderEngine::TimeFrame::TF_1DAY);
      }
    }
    if (ImGui::Selectable("Weekly SMA")) {
      auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
      if (symbol_id_opt) {
        add_multi_timeframe_indicator("Weekly SMA 20", true, ImVec4(0.0f, 0.0f, 1.0f, 1.0f), 20, RenderEngine::TimeFrame::TF_1WEEK);
      }
    }
    if (ImGui::Selectable("Hourly SMA")) {
      auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
      if (symbol_id_opt) {
        add_multi_timeframe_indicator("Hourly SMA 20", true, ImVec4(1.0f, 0.0f, 1.0f, 1.0f), 20, RenderEngine::TimeFrame::TF_1HOUR);
      }
    }

    ImGui::EndPopup();
  }

  ImGui::Separator();

  // Render the list of active indicators with improved layout
  if (!active_indicators_.empty()) {
    ImGui::Text("Active Indicators (%zu):", active_indicators_.size());

    // Create a table for better organization of indicator properties
    if (ImGui::BeginTable("IndicatorTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Visibility", ImGuiTableColumnFlags_WidthFixed, 50.0f);
      ImGui::TableSetupColumn("Color", ImGuiTableColumnFlags_WidthFixed, 50.0f);
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Parameters", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f); // For delete button

      ImGui::TableHeadersRow();

      for (auto it = active_indicators_.begin(); it != active_indicators_.end();) {
        auto& indicator = *it;

        ImGui::PushID(indicator.id); // Use unique ID for each indicator

        ImGui::TableNextRow();

        // Column 1: Visibility checkbox
        ImGui::TableSetColumnIndex(0);
        bool visible = indicator.isVisible;
        if (ImGui::Checkbox("##visible", &visible)) {
          indicator.isVisible = visible;
          update_indicator_config_from_active();
        }

        // Column 2: Color picker
        ImGui::TableSetColumnIndex(1);
        if (ImGui::ColorEdit4("##color", &indicator.color.x,
              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoTooltip)) {
          // Color changed, no additional action needed
        }

        // Column 3: Indicator name
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%s", indicator.name.c_str());

        // Column 4: Parameter inputs based on indicator type
        ImGui::TableSetColumnIndex(3);
        if (indicator.name.find("SMA") != std::string::npos ||
            indicator.name.find("EMA") != std::string::npos ||
            indicator.name.find("RSI") != std::string::npos ||
            indicator.name.find("ATR") != std::string::npos) {

          float period = indicator.parameters.count("period") > 0 ?
                        indicator.parameters["period"] : 9.0f;
          if (ImGui::DragFloat("##period", &period, 0.5f, 1.0f, 200.0f, "Period: %.0f")) {
            indicator.parameters["period"] = period;
            update_indicator_config_from_active();
          }
        } else if (indicator.name.find("MACD") != std::string::npos) {
          float fast_period = indicator.parameters.count("fast_period") > 0 ?
                             indicator.parameters["fast_period"] : 12.0f;
          float slow_period = indicator.parameters.count("slow_period") > 0 ?
                             indicator.parameters["slow_period"] : 26.0f;
          float signal_period = indicator.parameters.count("signal_period") > 0 ?
                               indicator.parameters["signal_period"] : 9.0f;

          if (ImGui::DragFloat("##fast", &fast_period, 0.5f, 1.0f, 50.0f, "Fast: %.0f")) {
            indicator.parameters["fast_period"] = fast_period;
            update_indicator_config_from_active();
          }
          ImGui::SameLine();
          if (ImGui::DragFloat("##slow", &slow_period, 0.5f, 1.0f, 100.0f, "Slow: %.0f")) {
            indicator.parameters["slow_period"] = slow_period;
            update_indicator_config_from_active();
          }
          ImGui::SameLine();
          if (ImGui::DragFloat("##signal", &signal_period, 0.5f, 1.0f, 50.0f, "Signal: %.0f")) {
            indicator.parameters["signal_period"] = signal_period;
            update_indicator_config_from_active();
          }
        } else if (indicator.name.find("Bollinger") != std::string::npos) {
          float period = indicator.parameters.count("period") > 0 ?
                        indicator.parameters["period"] : 20.0f;
          float std_dev = indicator.parameters.count("std_dev") > 0 ?
                         indicator.parameters["std_dev"] : 2.0f;

          if (ImGui::DragFloat("##bb_period", &period, 0.5f, 1.0f, 100.0f, "Period: %.0f")) {
            indicator.parameters["period"] = period;
            update_indicator_config_from_active();
          }
          ImGui::SameLine();
          if (ImGui::DragFloat("##bb_std", &std_dev, 0.1f, 0.1f, 5.0f, "Std Dev: %.1f")) {
            indicator.parameters["std_dev"] = std_dev;
            update_indicator_config_from_active();
          }
        } else if (indicator.name.find("Stochastic") != std::string::npos) {
          float k_period = indicator.parameters.count("k_period") > 0 ?
                          indicator.parameters["k_period"] : 14.0f;
          float d_period = indicator.parameters.count("d_period") > 0 ?
                          indicator.parameters["d_period"] : 3.0f;
          float slow_period = indicator.parameters.count("slow_period") > 0 ?
                             indicator.parameters["slow_period"] : 3.0f;

          if (ImGui::DragFloat("##stoch_k", &k_period, 0.5f, 1.0f, 50.0f, "K: %.0f")) {
            indicator.parameters["k_period"] = k_period;
            update_indicator_config_from_active();
          }
          ImGui::SameLine();
          if (ImGui::DragFloat("##stoch_d", &d_period, 0.5f, 1.0f, 50.0f, "D: %.0f")) {
            indicator.parameters["d_period"] = d_period;
            update_indicator_config_from_active();
          }
          ImGui::SameLine();
          if (ImGui::DragFloat("##stoch_slow", &slow_period, 0.5f, 1.0f, 50.0f, "Slow: %.0f")) {
            indicator.parameters["slow_period"] = slow_period;
            update_indicator_config_from_active();
          }
        }

        // Column 5: Delete button
        ImGui::TableSetColumnIndex(4);
        if (ImGui::Button("X##delete")) {
          // Set the corresponding configuration flag to false based on the indicator name
          if (indicator.name == "SMA 9") {
            indicator_config_.show_sma_9 = false;
          } else if (indicator.name == "SMA 20") {
            indicator_config_.show_sma_20 = false;
          } else if (indicator.name == "SMA 50") {
            indicator_config_.show_sma_50 = false;
          } else if (indicator.name == "SMA 200") {
            indicator_config_.show_sma_200 = false;
          } else if (indicator.name == "EMA 9") {
            indicator_config_.show_ema_9 = false;
          } else if (indicator.name == "EMA 21") {
            indicator_config_.show_ema_21 = false;
          } else if (indicator.name == "EMA 50") {
            indicator_config_.show_ema_50 = false;
          } else if (indicator.name == "EMA 200") {
            indicator_config_.show_ema_200 = false;
          } else if (indicator.name == "RSI") {
            indicator_config_.show_rsi = false;
          } else if (indicator.name == "MACD") {
            indicator_config_.show_macd = false;
          } else if (indicator.name == "Bollinger Bands") {
            indicator_config_.show_bollinger = false;
          } else if (indicator.name == "Stochastic") {
            indicator_config_.show_stochastic = false;
          } else if (indicator.name == "ATR") {
            indicator_config_.show_atr = false;
          } else if (indicator.name == "Fibonacci") {
            indicator_config_.show_fibonacci = false;
          } else if (indicator.name == "Volume Profile") {
            indicator_config_.show_volume_profile = false;
          } else if (indicator.name == "Crosshair Info") {
            indicator_config_.show_crosshair_info = false;
          }

          it = active_indicators_.erase(it);
          update_indicator_config_from_active();
          ImGui::PopID();
          continue; // Skip incrementing iterator since we removed an element
        }

        ImGui::PopID(); // Pop the ID for this indicator
        ++it;
      }

      ImGui::EndTable();
    }
  } else {
    ImGui::Text("No active indicators. Click 'Add Indicator' to add one.");
  }

  // Render multi-timeframe indicators
  if (!multi_tf_indicators_.empty()) {
    ImGui::Separator();
    ImGui::Text("Multi-Timeframe Indicators (%zu):", multi_tf_indicators_.size());

    if (ImGui::BeginTable("MultiTFIndicatorTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Visibility", ImGuiTableColumnFlags_WidthFixed, 50.0f);
      ImGui::TableSetupColumn("Color", ImGuiTableColumnFlags_WidthFixed, 50.0f);
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Parameters", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 40.0f); // For delete button

      ImGui::TableHeadersRow();

      for (auto it = multi_tf_indicators_.begin(); it != multi_tf_indicators_.end();) {
        auto& indicator = *it;

        ImGui::PushID(next_multitf_indicator_id_ + std::distance(multi_tf_indicators_.begin(), it)); // Use unique ID for each indicator

        ImGui::TableNextRow();

        // Column 1: Visibility checkbox
        ImGui::TableSetColumnIndex(0);
        bool visible = indicator.isVisible;
        if (ImGui::Checkbox("##multitf_visible", &visible)) {
          indicator.isVisible = visible;
        }

        // Column 2: Color picker
        ImGui::TableSetColumnIndex(1);
        if (ImGui::ColorEdit4("##multitf_color", &indicator.color.x,
              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoTooltip)) {
          // Color changed, no additional action needed
        }

        // Column 3: Indicator name and source timeframe
        ImGui::TableSetColumnIndex(2);
        char timeframe_str[64];
        snprintf(timeframe_str, sizeof(timeframe_str), "%s (%s)",
                 indicator.name.c_str(),
                 timeframe_to_string(indicator.source_timeframe).c_str());
        ImGui::Text("%s", timeframe_str);

        // Column 4: Parameter input for multi-timeframe indicators
        ImGui::TableSetColumnIndex(3);
        if (indicator.name.find("SMA") != std::string::npos) {
          float period = static_cast<float>(indicator.period);
          if (ImGui::DragFloat("##mtf_period", &period, 0.5f, 1.0f, 200.0f, "Period: %.0f")) {
            indicator.period = static_cast<int>(period);
          }
        }

        // Column 5: Delete button
        ImGui::TableSetColumnIndex(4);
        if (ImGui::Button("X##mtf_delete")) {
          it = multi_tf_indicators_.erase(it);
          ImGui::PopID();
          continue; // Skip incrementing iterator since we removed an element
        }

        ImGui::PopID(); // Pop the ID for this indicator
        ++it;
      }

      ImGui::EndTable();
    }
  } else {
    ImGui::Text("No multi-timeframe indicators. Click 'Add Multi-TF Indicator' to add one.");
  }

  ImGui::End();
}

std::vector<double> ChartPanel::calculate_bollinger_upper(const std::vector<float>& prices,
                                                          int period, double std_dev) {
  // Redirect to the cached version
  return calculate_cached_bollinger_upper(prices, period, std_dev);
}

std::vector<double> ChartPanel::calculate_bollinger_middle(const std::vector<float>& prices,
                                                          int period) {
  // Redirect to the cached version
  return calculate_cached_bollinger_middle(prices, period);
}

std::vector<double> ChartPanel::calculate_bollinger_lower(const std::vector<float>& prices,
                                                          int period, double std_dev) {
  // Redirect to the cached version
  return calculate_cached_bollinger_lower(prices, period, std_dev);
}

std::vector<double> ChartPanel::calculate_rsi(const std::vector<float>& prices, int period) {
  // Redirect to the cached version
  return calculate_cached_rsi(prices, period);
}

std::vector<double> ChartPanel::calculate_macd_line(const std::vector<float>& prices, int fast,
                                                    int slow) {
  // Redirect to the cached version
  return calculate_cached_macd_line(prices, fast, slow);
}

std::vector<double> ChartPanel::calculate_macd_signal(const std::vector<double>& macd_line,
                                                      int signal) {
  // Redirect to the cached version
  return calculate_cached_macd_signal(macd_line, signal);
}

std::vector<double> ChartPanel::calculate_macd_histogram(const std::vector<double>& macd_line,
                                                         const std::vector<double>& signal) {
  // Redirect to the cached version
  return calculate_cached_macd_histogram(macd_line, signal);
}

std::vector<FibonacciLevel> ChartPanel::calculate_fibonacci_levels(double start_price,
                                                                   double end_price) {
  std::vector<FibonacciLevel> levels;

  if (start_price == 0.0 || end_price == 0.0) return levels;

  double range = end_price - start_price;

  // Fibonacci ratios
  const double ratios[] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};
  const char* labels[] = {"0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"};
  const ImU32 colors[] = {
      IM_COL32(255, 255, 255, 255),  // White
      IM_COL32(0, 255, 255, 200),    // Cyan
      IM_COL32(0, 255, 0, 200),      // Green
      IM_COL32(255, 255, 0, 200),    // Yellow
      IM_COL32(255, 165, 0, 200),    // Orange
      IM_COL32(255, 0, 0, 200),      // Red
      IM_COL32(255, 255, 255, 200)   // White
  };

  for (int i = 0; i < 7; ++i) {
    FibonacciLevel level;
    level.price = start_price + range * ratios[i];
    level.ratio = ratios[i];
    level.label = labels[i];
    level.color = colors[i];
    levels.push_back(level);
  }

  return levels;
}

std::vector<double> ChartPanel::calculate_stochastic_k(const std::vector<float>& highs,
                                                       const std::vector<float>& lows,
                                                       const std::vector<float>& closes,
                                                       int k_period) {
  // Redirect to the cached version
  return calculate_cached_stochastic_k(highs, lows, closes, k_period);
}

std::vector<double> ChartPanel::calculate_stochastic_d(const std::vector<double>& stoch_k,
                                                       int slow_period) {
  // Redirect to the cached version
  return calculate_cached_stochastic_d(stoch_k, slow_period);
}

std::vector<double> ChartPanel::calculate_true_range(const std::vector<float>& highs,
                                                     const std::vector<float>& lows,
                                                     const std::vector<float>& closes) {
  // Redirect to the cached version
  return calculate_cached_true_range(highs, lows, closes);
}

std::vector<double> ChartPanel::calculate_atr(const std::vector<float>& highs,
                                              const std::vector<float>& lows,
                                              const std::vector<float>& closes,
                                              int period) {
  // Redirect to the cached version
  return calculate_cached_atr(highs, lows, closes, period);
}

// ============================================================================
// INDICATOR RENDERING METHODS
// ============================================================================

void ChartPanel::render_sma_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx) {
  if (chart.closes.empty()) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // SMA 9
  if (indicator_config_.show_sma_9) {
    IndicatorCacheKey key{IndicatorType::SMA, chart.closes.size(), 9, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& sma_9 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 8) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 8], sma_9[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_9[i]);
          draw_list->AddLine(p1, p2, IM_COL32(255, 105, 180, 200), 2.0f); // Hot pink
        }
      }
    }
  }

  // SMA 10
  if (indicator_config_.show_sma_10) {
    IndicatorCacheKey key{IndicatorType::SMA, chart.closes.size(), 10, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& sma_10 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 9) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 9], sma_10[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_10[i]);
          draw_list->AddLine(p1, p2, IM_COL32(255, 165, 0, 200), 2.0f); // Orange
        }
      }
    }
  }

  // SMA 20
  if (indicator_config_.show_sma_20) {
    IndicatorCacheKey key{IndicatorType::SMA, chart.closes.size(), 20, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& sma_20 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 19) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 19], sma_20[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_20[i]);
          draw_list->AddLine(p1, p2, IM_COL32(255, 255, 0, 200), 2.0f); // Yellow
        }
      }
    }
  }

  // SMA 50
  if (indicator_config_.show_sma_50) {
    IndicatorCacheKey key{IndicatorType::SMA, chart.closes.size(), 50, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& sma_50 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 49) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 49], sma_50[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_50[i]);
          draw_list->AddLine(p1, p2, IM_COL32(0, 255, 255, 200), 2.0f); // Cyan
        }
      }
    }
  }

  // SMA 200
  if (indicator_config_.show_sma_200) {
    IndicatorCacheKey key{IndicatorType::SMA, chart.closes.size(), 200, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& sma_200 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 199) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 199], sma_200[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_200[i]);
          draw_list->AddLine(p1, p2, IM_COL32(128, 0, 128, 200), 2.0f); // Purple
        }
      }
    }
  }
}

void ChartPanel::render_ema_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx) {
  if (chart.closes.empty()) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // EMA 9
  if (indicator_config_.show_ema_9) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 9, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_9 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 8) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 8], ema_9[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_9[i]);
          draw_list->AddLine(p1, p2, IM_COL32(255, 0, 255, 200), 2.0f); // Magenta
        }
      }
    }
  }

  // EMA 10
  if (indicator_config_.show_ema_10) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 10, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_10 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 9) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 9], ema_10[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_10[i]);
          draw_list->AddLine(p1, p2, IM_COL32(255, 0, 128, 200), 2.0f); // Medium violet red
        }
      }
    }
  }

  // EMA 20
  if (indicator_config_.show_ema_20) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 20, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_20 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 19) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 19], ema_20[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_20[i]);
          draw_list->AddLine(p1, p2, IM_COL32(138, 43, 226, 200), 2.0f); // Blue violet
        }
      }
    }
  }

  // EMA 21
  if (indicator_config_.show_ema_21) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 21, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_21 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 20) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 20], ema_21[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_21[i]);
          draw_list->AddLine(p1, p2, IM_COL32(0, 191, 255, 200), 2.0f); // Deep sky blue
        }
      }
    }
  }

  // EMA 50
  if (indicator_config_.show_ema_50) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 50, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_50 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 49) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 49], ema_50[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_50[i]);
          draw_list->AddLine(p1, p2, IM_COL32(65, 105, 225, 200), 2.0f); // Royal blue
        }
      }
    }
  }

  // EMA 200
  if (indicator_config_.show_ema_200) {
    IndicatorCacheKey key{IndicatorType::EMA, chart.closes.size(), 200, 0, 0.0};
    auto it = cached_indicators_.find(key);
    if (it != cached_indicators_.end()) {
      const auto& ema_200 = it->second;
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i >= 199) {
          ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 199], ema_200[i]);
          ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_200[i]);
          draw_list->AddLine(p1, p2, IM_COL32(75, 0, 130, 200), 2.0f); // Indigo
        }
      }
    }
  }
}

void ChartPanel::render_bollinger_bands(const ChartInstance& chart, size_t start_idx,
                                        size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_bollinger) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Get cached Bollinger Bands
  IndicatorCacheKey upper_key{IndicatorType::BB_UPPER, chart.closes.size(),
                              indicator_config_.bollinger_period, 0,
                              indicator_config_.bollinger_std_dev};
  IndicatorCacheKey middle_key{IndicatorType::BB_MIDDLE, chart.closes.size(),
                               indicator_config_.bollinger_period, 0, 0.0};
  IndicatorCacheKey lower_key{IndicatorType::BB_LOWER, chart.closes.size(),
                              indicator_config_.bollinger_period, 0,
                              indicator_config_.bollinger_std_dev};

  auto upper_it = cached_indicators_.find(upper_key);
  auto middle_it = cached_indicators_.find(middle_key);
  auto lower_it = cached_indicators_.find(lower_key);

  if (upper_it != cached_indicators_.end() &&
      middle_it != cached_indicators_.end() &&
      lower_it != cached_indicators_.end()) {

    const auto& upper_band = upper_it->second;
    const auto& middle_band = middle_it->second;
    const auto& lower_band = lower_it->second;

    // Prepare points for filled area between upper and lower bands
    std::vector<ImVec2> upper_points;
    std::vector<ImVec2> lower_points;

    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= static_cast<size_t>(indicator_config_.bollinger_period - 1)) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[i], upper_band[i]);
        ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[i], lower_band[i]);
        ImVec2 middle_point = ImPlot::PlotToPixels(chart.dates[i], middle_band[i]);

        // Add points for filled area
        upper_points.push_back(upper_point);
        lower_points.insert(lower_points.begin(), lower_point); // Insert at beginning to maintain order

        // Draw middle band (SMA)
        if (i > start_idx && i >= static_cast<size_t>(indicator_config_.bollinger_period - 1)) {
          ImVec2 prev_middle = ImPlot::PlotToPixels(chart.dates[i-1], middle_band[i-1]);
          draw_list->AddLine(prev_middle, middle_point, IM_COL32(255, 255, 0, 150), 1.0f); // Yellow
        }
      }
    }

    // Draw filled area between upper and lower bands
    if (!upper_points.empty() && !lower_points.empty()) {
      std::vector<ImVec2> filled_area_points;
      filled_area_points.insert(filled_area_points.end(), upper_points.begin(), upper_points.end());
      filled_area_points.insert(filled_area_points.end(), lower_points.begin(), lower_points.end());

      if (filled_area_points.size() >= 3) {
        draw_list->AddConvexPolyFilled(filled_area_points.data(),
                                     static_cast<int>(filled_area_points.size()),
                                     IM_COL32(0, 255, 255, 50)); // Semi-transparent cyan
      }
    }

    // Draw upper and lower band lines
    for (size_t i = start_idx + 1; i < end_idx; ++i) {
      if (i >= static_cast<size_t>(indicator_config_.bollinger_period - 1)) {
        ImVec2 prev_upper = ImPlot::PlotToPixels(chart.dates[i-1], upper_band[i-1]);
        ImVec2 curr_upper = ImPlot::PlotToPixels(chart.dates[i], upper_band[i]);
        ImVec2 prev_lower = ImPlot::PlotToPixels(chart.dates[i-1], lower_band[i-1]);
        ImVec2 curr_lower = ImPlot::PlotToPixels(chart.dates[i], lower_band[i]);

        draw_list->AddLine(prev_upper, curr_upper, IM_COL32(0, 255, 255, 150), 1.0f); // Cyan
        draw_list->AddLine(prev_lower, curr_lower, IM_COL32(0, 255, 255, 150), 1.0f); // Cyan
      }
    }
  }
}

void ChartPanel::render_rsi_indicator(const ChartInstance& chart, size_t start_idx,
                                      size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_rsi) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Get cached RSI values
  IndicatorCacheKey key{IndicatorType::RSI, chart.closes.size(), indicator_config_.rsi_period, 0, 0.0};
  auto it = cached_indicators_.find(key);
  if (it == cached_indicators_.end()) return; // No cached data available

  const auto& rsi = it->second;

  // Get plot limits for RSI scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.rsi_period)) {
      double rsi_value = rsi[i];

      // Map RSI to Y-axis (0-100)
      double y = limits.Y.Min + (rsi_value / 100.0) * (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Color based on overbought/oversold
      ImU32 color = IM_COL32(128, 128, 128, 200);
      if (rsi_value >= indicator_config_.rsi_overbought) {
        color = IM_COL32(255, 0, 0, 200);  // Red
      } else if (rsi_value <= indicator_config_.rsi_oversold) {
        color = IM_COL32(0, 255, 0, 200);  // Green
      }

      // Draw RSI line
      draw_list->AddLine(p, p, color, 1.5f);
    }
  }

  // Draw overbought/oversold lines
  double overbought_y =
      limits.Y.Min + (indicator_config_.rsi_overbought / 100.0) * (limits.Y.Max - limits.Y.Min);
  double oversold_y =
      limits.Y.Min + (indicator_config_.rsi_oversold / 100.0) * (limits.Y.Max - limits.Y.Min);

  ImVec2 ob_p1 = ImPlot::PlotToPixels(limits.X.Min, overbought_y);
  ImVec2 ob_p2 = ImPlot::PlotToPixels(limits.X.Max, overbought_y);
  draw_list->AddLine(ob_p1, ob_p2, IM_COL32(255, 0, 0, 100), 1.0f);

  ImVec2 os_p1 = ImPlot::PlotToPixels(limits.X.Min, oversold_y);
  ImVec2 os_p2 = ImPlot::PlotToPixels(limits.X.Max, oversold_y);
  draw_list->AddLine(os_p1, os_p2, IM_COL32(0, 255, 0, 100), 1.0f);
}

void ChartPanel::render_stochastic_indicator(const ChartInstance& chart, size_t start_idx,
                                           size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_stochastic) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Get cached Stochastic values
  IndicatorCacheKey k_key{IndicatorType::STOCH_K, chart.highs.size(),
                          indicator_config_.stochastic_k_period, 0, 0.0};
  IndicatorCacheKey d_key{IndicatorType::STOCH_D, chart.highs.size(),
                          indicator_config_.stochastic_d_period, 0, 0.0};

  auto k_it = cached_indicators_.find(k_key);
  auto d_it = cached_indicators_.find(d_key);

  if (k_it != cached_indicators_.end() && d_it != cached_indicators_.end()) {
    const auto& stoch_k = k_it->second;
    const auto& stoch_d = d_it->second;

    // Get plot limits for Stochastic scaling
    ImPlotRect limits = ImPlot::GetPlotLimits();

    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= static_cast<size_t>(indicator_config_.stochastic_k_period +
                                  indicator_config_.stochastic_d_period - 1)) {

        double k_value = stoch_k[i];
        double d_value = stoch_d[i];

        // Map Stochastic values to Y-axis (0-100)
        double k_y = limits.Y.Min + (k_value / 100.0) * (limits.Y.Max - limits.Y.Min);
        double d_y = limits.Y.Min + (d_value / 100.0) * (limits.Y.Max - limits.Y.Min);

        ImVec2 k_point = ImPlot::PlotToPixels(chart.dates[i], k_y);
        ImVec2 d_point = ImPlot::PlotToPixels(chart.dates[i], d_y);

        // Draw %K line (typically faster line)
        draw_list->AddLine(k_point, k_point, IM_COL32(255, 255, 0, 200), 1.5f); // Yellow

        // Draw %D line (typically slower line)
        draw_list->AddLine(d_point, d_point, IM_COL32(255, 0, 0, 200), 1.5f); // Red
      }
    }

    // Draw overbought/oversold lines (typically at 80 and 20)
    double overbought_y = limits.Y.Min + (80.0 / 100.0) * (limits.Y.Max - limits.Y.Min);
    double oversold_y = limits.Y.Min + (20.0 / 100.0) * (limits.Y.Max - limits.Y.Min);

    ImVec2 ob_p1 = ImPlot::PlotToPixels(limits.X.Min, overbought_y);
    ImVec2 ob_p2 = ImPlot::PlotToPixels(limits.X.Max, overbought_y);
    draw_list->AddLine(ob_p1, ob_p2, IM_COL32(255, 0, 0, 100), 1.0f); // Red

    ImVec2 os_p1 = ImPlot::PlotToPixels(limits.X.Min, oversold_y);
    ImVec2 os_p2 = ImPlot::PlotToPixels(limits.X.Max, oversold_y);
    draw_list->AddLine(os_p1, os_p2, IM_COL32(0, 255, 0, 100), 1.0f); // Green
  }
}

void ChartPanel::render_macd_indicator(const ChartInstance& chart, size_t start_idx,
                                       size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_macd) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Get cached MACD values
  IndicatorCacheKey line_key{IndicatorType::MACD_LINE, chart.closes.size(),
                            indicator_config_.macd_fast_period, indicator_config_.macd_slow_period, 0.0};
  IndicatorCacheKey signal_key{IndicatorType::MACD_SIGNAL, chart.closes.size(),
                               indicator_config_.macd_signal_period, 0, 0.0};
  IndicatorCacheKey histogram_key{IndicatorType::MACD_HISTOGRAM, chart.closes.size(), 0, 0, 0.0};

  // Get all cached MACD components
  auto line_it = cached_indicators_.find(line_key);
  auto signal_it = cached_indicators_.find(signal_key);
  auto histogram_it = cached_indicators_.find(histogram_key);

  if (line_it == cached_indicators_.end() ||
      signal_it == cached_indicators_.end() ||
      histogram_it == cached_indicators_.end()) {
    return; // No cached data available
  }

  const auto& macd_line = line_it->second;
  const auto& macd_signal = signal_it->second;
  const auto& macd_histogram = histogram_it->second;

  // Get plot limits for MACD scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.macd_slow_period +
                                 indicator_config_.macd_signal_period)) {
      double macd_value = macd_line[i];
      double signal_value = macd_signal[i];
      double hist_value = macd_histogram[i];

      // Map MACD to Y-axis
      double y = limits.Y.Min + ((macd_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                                    (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Draw MACD line
      draw_list->AddLine(p, p, IM_COL32(0, 255, 255, 200), 1.5f);

      // Draw signal line
      double signal_y =
          limits.Y.Min + ((signal_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                             (limits.Y.Max - limits.Y.Min);
      ImVec2 signal_p = ImPlot::PlotToPixels(chart.dates[i], signal_y);
      draw_list->AddLine(signal_p, signal_p, IM_COL32(255, 165, 0, 200), 1.5f);

      // Draw histogram
      double hist_y = limits.Y.Min + ((hist_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                                         (limits.Y.Max - limits.Y.Min);
      ImVec2 hist_p = ImPlot::PlotToPixels(chart.dates[i], hist_y);

      ImU32 hist_color = hist_value >= 0 ? IM_COL32(0, 255, 0, 150) : IM_COL32(255, 0, 0, 150);
      draw_list->AddRectFilled(ImVec2(hist_p.x - 2, hist_p.y),
                               ImVec2(hist_p.x + 2, hist_p.y + hist_y - signal_y), hist_color);
    }
  }
}

void ChartPanel::render_atr_indicator(const ChartInstance& chart, size_t start_idx,
                                     size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_atr) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Get cached ATR values
  IndicatorCacheKey key{IndicatorType::ATR, chart.highs.size(), indicator_config_.atr_period, 0, 0.0};
  auto it = cached_indicators_.find(key);
  if (it == cached_indicators_.end()) return; // No cached data available

  const auto& atr = it->second;

  // Get plot limits for scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  // Calculate min/max ATR values for normalization
  double min_atr = std::numeric_limits<double>::max();
  double max_atr = std::numeric_limits<double>::lowest();

  for (size_t i = indicator_config_.atr_period; i < atr.size(); ++i) {
    if (atr[i] < min_atr) min_atr = atr[i];
    if (atr[i] > max_atr) max_atr = atr[i];
  }

  // If all ATR values are the same, use a default range
  if (min_atr == max_atr) {
    min_atr = min_atr * 0.9;
    max_atr = max_atr * 1.1;
  }

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.atr_period)) {
      double atr_value = atr[i];

      // Normalize ATR value to fit within the plot area
      double normalized_atr = limits.Y.Min + ((atr_value - min_atr) / (max_atr - min_atr)) * (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], normalized_atr);

      // Draw ATR line
      draw_list->AddLine(p, p, IM_COL32(0, 255, 127, 200), 2.0f); // Spring green
    }
  }
}

void ChartPanel::render_fibonacci_levels(const ChartInstance& chart, size_t start_idx,
                                         size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_fibonacci) return;

  // Find swing high and low in visible range
  double swing_high = 0.0;
  double swing_low = 1e9;

  for (size_t i = start_idx; i < end_idx; ++i) {
    swing_high = std::max(swing_high, static_cast<double>(chart.highs[i]));
    swing_low = std::min(swing_low, static_cast<double>(chart.lows[i]));
  }

  if (swing_high == 0.0 || swing_low == 1e9) return;

  auto fib_levels = calculate_fibonacci_levels(swing_low, swing_high);
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Draw Fibonacci levels
  for (const auto& level : fib_levels) {
    ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[start_idx], level.price);
    ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[end_idx - 1], level.price);

    // Draw horizontal line
    draw_list->AddLine(p1, p2, level.color, 1.0f);

    // Draw label
    ImVec2 text_pos = ImVec2(p1.x + 5, p1.y - 10);
    draw_list->AddText(text_pos, level.color, level.label);
  }
}

void ChartPanel::render_crosshair_info(const ChartInstance& chart, double mouse_x, double mouse_y) {
  (void)mouse_y;  // Suppress unused parameter warning
  if (!indicator_config_.show_crosshair_info || chart.closes.empty()) return;

  // Find closest candle to mouse position using binary search
  size_t closest_idx = 0;

  // Use lower_bound to find the insertion point for mouse_x in the sorted dates vector
  auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x);

  if (lower == chart.dates.end()) {
    // Mouse x is beyond the last date, use the last element
    closest_idx = chart.dates.size() - 1;
  } else if (lower == chart.dates.begin()) {
    // Mouse x is before the first date, use the first element
    closest_idx = 0;
  } else {
    // Compare the distance to the element at lower and the one before it
    size_t idx_after = std::distance(chart.dates.begin(), lower);
    size_t idx_before = idx_after - 1;

    double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x);
    double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x);

    closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
  }

  if (closest_idx >= chart.closes.size()) return;

  // Get candle data
  double open = chart.opens[closest_idx];
  double high = chart.highs[closest_idx];
  double low = chart.lows[closest_idx];
  double close = chart.closes[closest_idx];
  double volume = chart.volumes[closest_idx];

  // Render crosshair info overlay
  ImGui::SetNextWindowPos(ImVec2(ImGui::GetMousePos().x + 20, ImGui::GetMousePos().y + 20));
  ImGui::SetNextWindowSize(ImVec2(200, 150));
  ImGui::Begin("Crosshair Info", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Candle Info:");
  ImGui::Separator();
  ImGui::Text("Open:  %.2f", open);
  ImGui::Text("High:  %.2f", high);
  ImGui::Text("Low:   %.2f", low);
  ImGui::Text("Close: %.2f", close);
  ImGui::Text("Volume: %.2f", volume);

  // Calculate change
  if (closest_idx > 0) {
    double prev_close = chart.closes[closest_idx - 1];
    double change = close - prev_close;
    double change_pct = (change / prev_close) * 100.0;

    ImGui::Separator();
    ImGui::TextColored(change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                       "Change: %.2f (%.2f%%)", change, change_pct);
  }

  ImGui::End();
}

void ChartPanel::render_instrument_chart(const ChartInstance& chart) {
  if (chart.dates.empty()) {
    ImGui::Text("Loading chart data for %s...", symbol_.c_str());
    return;
  }

  // Diagnostic info & Controls
  ImGui::TextColored(
      ImVec4(0.0f, 1.0f, 0.8f, 1.0f), " | Candles: %zu | Last: %.2f | TF: %.4gs",
      chart.dates.size(), chart.closes.empty() ? 0.0f : chart.closes.back(),
      RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) / 1000000.0);

  // Neon Chart Styling (ThemeManager)
  const auto& colors = ThemeManager::getInstance().getColors();
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, colors.background);
  ImPlot::PushStyleColor(ImPlotCol_PlotBg,
                         colors.panel_bg);  // Use panel bg or specific dark
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, colors.border);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 10));

  // Initialize chart configuration
  // Ensure we start with auto-follow enabled to show recent action
  if (first_frame_) {
    follow_latest_ = true;
    first_frame_ = false;
  }

  // Fetch Volume Profile Data Early (for auto-scaling axes)
  double vp_max_vol = 0;
  std::vector<float> vp_prices;
  std::vector<float> vp_volumes;

  if (indicator_config_.show_volume_profile) {
    auto id_opt = chart_manager_->getSymbolId(symbol_);
    if (id_opt) {
      auto profile = processor_->getVolumeProfile(*id_opt, timeframe_);
      if (!profile.empty()) {
        vp_prices.reserve(profile.size());
        vp_volumes.reserve(profile.size());

        for (const auto& level : profile) {
          vp_prices.push_back(level.price);
          vp_volumes.push_back(static_cast<float>(level.total_volume));
          if (level.total_volume > vp_max_vol) vp_max_vol = level.total_volume;
        }
      }
    }
  }

  // Use a unique ID string per symbol/tf/exchange to ensure ImPlot saves state
  // per chart
  std::string plot_id =
      "##Chart_" + symbol_ + "_" + exchange_ + "_" + timeframe_to_string(timeframe_);

  // Pre-calculate axis limits BEFORE BeginPlot to avoid calling locking functions during setup
  double x_axis_min_pre, x_axis_max_pre;
  double y_axis_min_pre = std::numeric_limits<double>::max();
  double y_axis_max_pre = std::numeric_limits<double>::lowest();

  if (follow_latest_) {
    double time_max = chart.dates.back();
    double duration_raw = RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);
    double duration_sec = duration_raw / 1000000.0;
    double window_size = duration_sec * auto_follow_window_;
    double padding = window_size * 0.05;

    last_view_min_ = time_max - window_size;
    last_view_max_ = time_max + padding;

    x_axis_min_pre = last_view_min_;
    x_axis_max_pre = last_view_max_;
  } else {
    x_axis_min_pre = chart.dates.front();
    x_axis_max_pre = chart.dates.back();
  }

  // Calculate Y-axis limits based on visible X range
  size_t start_idx_pre = 0;
  size_t end_idx_pre = chart.dates.size();

  auto lower_pre = std::lower_bound(chart.dates.begin(), chart.dates.end(), x_axis_min_pre);
  if (lower_pre != chart.dates.begin()) --lower_pre;
  start_idx_pre = std::distance(chart.dates.begin(), lower_pre);

  auto upper_pre = std::upper_bound(chart.dates.begin(), chart.dates.end(), x_axis_max_pre);
  if (upper_pre != chart.dates.end()) ++upper_pre;
  end_idx_pre = std::distance(chart.dates.begin(), upper_pre);

  end_idx_pre = std::min(end_idx_pre, chart.dates.size());

  if (start_idx_pre < end_idx_pre) {
    bool found_data = false;

    for (size_t i = start_idx_pre; i < end_idx_pre; ++i) {
      double low = chart.lows[i];
      double high = chart.highs[i];
      if (low > 0 && high > 0) {
        if (low < y_axis_min_pre) y_axis_min_pre = low;
        if (high > y_axis_max_pre) y_axis_max_pre = high;
        found_data = true;
      }
    }

    if (found_data) {
      double range = y_axis_max_pre - y_axis_min_pre;
      if (range == 0) range = y_axis_min_pre * 0.01;
      if (range == 0) range = 1.0;

      y_axis_min_pre -= range * 0.1;
      y_axis_max_pre += range * 0.1;
    } else {
      y_axis_min_pre = chart.lows.empty() ? 0 : chart.lows[0];
      y_axis_max_pre = chart.highs.empty() ? 1 : chart.highs[0];
    }
  } else {
    y_axis_min_pre =
        chart.lows.empty() ? 0 : *std::min_element(chart.lows.begin(), chart.lows.end());
    y_axis_max_pre =
        chart.highs.empty() ? 1 : *std::max_element(chart.highs.begin(), chart.highs.end());
  }

  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle | ImPlotFlags_Crosshairs)) {
    // ===== ALL SETUP CALLS MUST HAPPEN FIRST - BEFORE ANY LOCKING FUNCTIONS =====
    // Setup primary axes
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Setup X2 for Volume Profile (must happen before SetupAxisLimits for main axes)
    if (indicator_config_.show_volume_profile && vp_max_vol > 0) {
      ImPlot::SetupAxis(ImAxis_X2, nullptr,
                        ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines);
      ImPlot::SetupAxisLimits(ImAxis_X2, 0, vp_max_vol * 4.0, ImPlotCond_Always);
    }

    // Apply axis limits - these must ALL come before any locking functions
    ImPlot::SetupAxisLimits(ImAxis_X1, x_axis_min_pre, x_axis_max_pre,
                            follow_latest_ ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, static_cast<float>(y_axis_min_pre),
                            static_cast<float>(y_axis_max_pre), ImPlotCond_Always);
    // ===== END OF SETUP PHASE =====

    // NOW it's safe to call locking functions like IsPlotHovered(), GetPlotLimits(), etc.
    // Check for user interaction to break auto-follow
    bool user_interacted =
        ImPlot::IsPlotHovered() &&
        (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
         ImGui::IsMouseDragging(ImGuiMouseButton_Right) || ImGui::GetIO().MouseWheel != 0.0f);

    if (user_interacted) {
      follow_latest_ = false;
    }

    // VOLUME PROFILE OVERLAY - using direct draw to avoid axis switching after setup lock
    if (indicator_config_.show_volume_profile && !vp_prices.empty()) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // Use Theme Colors for a more integrated look
      ImVec4 vp_color = colors.text;
      vp_color.w = 0.25f;  // reduced alpha

      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, vp_color.w);
      ImPlot::PushStyleColor(ImPlotCol_Fill, vp_color);
      ImPlot::PushStyleColor(ImPlotCol_Line,
                             ImVec4(vp_color.x, vp_color.y, vp_color.z, 0.5f));  // clearer border

      // Calculate bar width in plot coordinates based on volume values
      double max_vol_display = vp_max_vol * 4.0;  // Same as used in SetupAxisLimits for X2
      (void)max_vol_display;  // Suppress unused variable warning

      // Draw volume profile bars using direct drawing to avoid axis switching after setup lock
      for (size_t i = 0; i < vp_prices.size(); ++i) {
        // Convert price (Y coordinate) and volume (X coordinate) to screen coordinates
        ImVec2 pos_screen = ImPlot::PlotToPixels(vp_volumes[i], vp_prices[i]);  // tip of the bar
        ImVec2 base_screen = ImPlot::PlotToPixels(0, vp_prices[i]);             // base of the bar

        // Calculate bar dimensions
        float bar_width = base_screen.x - pos_screen.x;  // width from volume value to zero
        float bar_height = 3.0f;                         // fixed height for visibility

        // Define bar corners
        ImVec2 bar_tl = ImVec2(base_screen.x - bar_width, pos_screen.y - bar_height / 2);
        ImVec2 bar_br = ImVec2(base_screen.x, pos_screen.y + bar_height / 2);

        // Draw the volume bar
        draw_list->AddRectFilled(bar_tl, bar_br, ImGui::GetColorU32(vp_color));
      }

      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar();
    }

    // Drag & Drop Target for Price Levels (Must be after ALL Setup calls)
    if (ImPlot::BeginDragDropTargetPlot()) {
      if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PRICE_LEVEL")) {
        double dropped_price = *(const double*)payload->Data;
        std::cout << "[ChartPanel] Dropped Price Level: " << dropped_price << std::endl;
      }
      ImPlot::EndDragDropTarget();
    }

    // Store previous view limits to detect changes
    double prev_view_min = last_view_min_;
    double prev_view_max = last_view_max_;

    // Now get actual limits being used for THIS frame's rendering and NEXT
    // frame's scaling This locks setup, so it must happen AFTER SetupAxisLimits
    ImPlotRect limits = ImPlot::GetPlotLimits();

    // Store for next frame
    last_view_min_ = limits.X.Min;
    last_view_max_ = limits.X.Max;

    // Check if the view has changed (scroll/zoom) and notify the time stats panel if needed
    if ((prev_view_min != last_view_min_ || prev_view_max != last_view_max_) && on_scroll_sync_) {
        uint64_t start_time = static_cast<uint64_t>(last_view_min_ * 1000000);
        uint64_t end_time = static_cast<uint64_t>(last_view_max_ * 1000000);
        on_scroll_sync_(start_time, end_time);
    }

    // Recalculate start/end for CULLING (Rendering optimization)
    size_t render_start_idx = 0;
    size_t render_end_idx = chart.dates.size();
    {
      auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), limits.X.Min);
      if (lower != chart.dates.begin()) --lower;
      render_start_idx = std::distance(chart.dates.begin(), lower);

      auto upper = std::upper_bound(chart.dates.begin(), chart.dates.end(), limits.X.Max);
      if (upper != chart.dates.end()) ++upper;
      render_end_idx =
          std::min((size_t)std::distance(chart.dates.begin(), upper), chart.dates.size());
    }

    // Calculate candle width based on timeframe
    double duration_sec =
        RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) / 1000000.0;
    double candle_half_width = duration_sec * 0.4;

    // Safeguard for very small TFs
    if (candle_half_width < 0.000001) candle_half_width = 0.000001;

    ImDrawList* draw_list = ImPlot::GetPlotDrawList();

    // Calculate minimum pixel width for candles
    const float MIN_BODY_WIDTH_PX = 3.0f;
    const float MIN_BODY_HEIGHT_PX = 1.0f;

    // Draw ONLY visible candles
    for (size_t i = render_start_idx; i < render_end_idx; ++i) {
      double x = chart.dates[i];
      if (x == 0) continue;

      float open = chart.opens[i];
      float high = chart.highs[i];
      float low = chart.lows[i];
      float close = chart.closes[i];

      // Skip invalid candles
      if (high == 0 || low == 0 || open == 0 || close == 0) continue;

      bool bullish = close >= open;
      const auto& colors = ThemeManager::getInstance().getColors();
      ImU32 color = bullish ? ImGui::ColorConvertFloat4ToU32(colors.candle_up)
                            : ImGui::ColorConvertFloat4ToU32(colors.candle_down);
      ImU32 wick_color = color;

      // Transform to screen coordinates
      ImVec2 wick_top = ImPlot::PlotToPixels(x, high);
      ImVec2 wick_bot = ImPlot::PlotToPixels(x, low);
      ImVec2 body_tl = ImPlot::PlotToPixels(x - candle_half_width, bullish ? close : open);
      ImVec2 body_br = ImPlot::PlotToPixels(x + candle_half_width, bullish ? open : close);

      // Ensure minimum body width in pixels
      float body_width = std::abs(body_br.x - body_tl.x);
      if (body_width < MIN_BODY_WIDTH_PX) {
        float extra = (MIN_BODY_WIDTH_PX - body_width) / 2.0f;
        body_tl.x -= extra;
        body_br.x += extra;
      }

      // Ensure minimum body height in pixels (for doji candles)
      float body_height = std::abs(body_br.y - body_tl.y);
      if (body_height < MIN_BODY_HEIGHT_PX) {
        float mid_y = (body_tl.y + body_br.y) / 2.0f;
        body_tl.y = mid_y - MIN_BODY_HEIGHT_PX / 2.0f;
        body_br.y = mid_y + MIN_BODY_HEIGHT_PX / 2.0f;
      }

      // Draw wick (vertical line)
      draw_list->AddLine(wick_top, wick_bot, wick_color, 1.0f);

      // Draw body (filled rectangle)
      draw_list->AddRectFilled(body_tl, body_br, color);
    }

    // Render mini volume profile histograms on candles if enabled
    if (indicator_config_.show_volume_profile) {
      // Extract the visible candle data for the mini histograms
      std::vector<RenderEngine::OHLCVCandle> visible_candles;
      std::vector<double> x_coords;
      std::vector<double> y_coords_high;
      std::vector<double> y_coords_low;

      for (size_t i = render_start_idx; i < render_end_idx; ++i) {
        RenderEngine::OHLCVCandle candle;
        candle.timestamp =
            static_cast<uint64_t>(chart.dates[i] * 1000000);  // Convert back to microseconds
        candle.open = chart.opens[i];
        candle.high = chart.highs[i];
        candle.low = chart.lows[i];
        candle.close = chart.closes[i];
        candle.volume = chart.volumes[i];
        candle.trade_count = 1;  // Placeholder

        visible_candles.push_back(candle);

        // Calculate screen coordinates for this candle
        ImVec2 wick_top = ImPlot::PlotToPixels(chart.dates[i], chart.highs[i]);
        ImVec2 wick_bot = ImPlot::PlotToPixels(chart.dates[i], chart.lows[i]);

        x_coords.push_back(wick_top.x);
        y_coords_high.push_back(wick_top.y);
        y_coords_low.push_back(wick_bot.y);
      }

      // Use the processor to get the volume profile data for this symbol and timeframe
      // This will allow us to render the mini histograms with actual volume distribution data
      auto id_opt = chart_manager_->getSymbolId(symbol_);
      if (id_opt) {
        uint32_t symbol_id = *id_opt;

        // Get recent trades for this symbol to populate the mini histograms
        auto analytics = processor_->getSymbolAnalytics(symbol_id);
        const auto& recent_trades = analytics.recent_trades;

        // If we have recent trades, render the step profile histograms showing volume distribution for each candle
        if (!recent_trades.empty()) {
          // Use the enhanced method for Step Profile rendering: draw mini histogram overlay on each candlestick bar
          // showing volume distribution for that bar's price range with additional visualization options
          // This is the main implementation for the task requirement
          // We'll call the method directly on this instance since we have access to the processor and symbol_id
          render_enhanced_step_profile_histograms_on_candle_bars(draw_list, visible_candles, x_coords,
                                                              y_coords_high, y_coords_low,
                                                              true, 8, 0.8f, false);  // Show POC line with 8 buckets per candle, 80% opacity, no labels
        }
      }
    }

    // Check for user interaction to break auto-follow
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      follow_latest_ = false;
    }

    // Render multi-timeframe indicators
    render_multi_timeframe_indicators(chart, render_start_idx, render_end_idx);

    // Render indicators
    render_sma_lines(chart, render_start_idx, render_end_idx);
    render_ema_lines(chart, render_start_idx, render_end_idx);
    render_bollinger_bands(chart, render_start_idx, render_end_idx);
    render_rsi_indicator(chart, render_start_idx, render_end_idx);
    render_macd_indicator(chart, render_start_idx, render_end_idx);
    render_stochastic_indicator(chart, render_start_idx, render_end_idx);
    render_atr_indicator(chart, render_start_idx, render_end_idx);
    render_fibonacci_levels(chart, render_start_idx, render_end_idx);

    // Render drawing tools
    if (drawing_tools_manager_) {
        drawing_tools_manager_->render_all();
    }

    // Render crosshair info if mouse is over plot
    // Handle mouse drag interaction for custom profile creation
    handleMouseDragInteraction();

    if (indicator_config_.show_crosshair_info && ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
      render_crosshair_info(chart, mouse_pos.x, mouse_pos.y);
    }

    // Handle drawing tools mouse events
    if (drawing_tools_manager_ && ImPlot::IsPlotHovered()) {
        drawing_tools_manager_->handle_mouse_events();

        // Check for mouse clicks to create drawing tools
        if (ImPlot::IsPlotSelected()) {
            ImPlotRect selection = ImPlot::GetPlotSelection();
            // Handle selection-based drawing tools like rectangles
            // For now, we'll just clear the selection
            ImPlot::EndPlot();
            ImPlot::SetNextAxisLimits(ImAxis_X1, selection.X.Min, selection.X.Max, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(ImAxis_Y1, selection.Y.Min, selection.Y.Max, ImGuiCond_Always);
        }

        // Handle right-click context menu for drawing tools
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
            (void)mouse_pos;  // Suppress unused variable warning
            // In a real implementation, we would show a context menu to select drawing tool type
            // For now, we'll just store the position for potential use
        }

        // Handle left mouse click for creating drawing tools
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImPlot::IsPlotHovered()) {
            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
            (void)mouse_pos;  // Suppress unused variable warning
            // In a real implementation, we would check if we're in drawing mode
            // and create the appropriate tool based on the selected tool type
        }

        // Handle mouse dragging for creating drawing tools
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImPlot::IsPlotHovered()) {
            // Check if we're in drawing mode
            static bool is_drawing = false;
            static ImPlotPoint start_point;
            static std::string current_tool_id;

            if (!is_drawing) {
                start_point = ImPlot::GetPlotMousePos();
                is_drawing = true;

                // Generate a unique ID for the new tool
                static int tool_counter = 0;
                current_tool_id = "tool_" + std::to_string(++tool_counter);
            }

            // During drag, we could preview the tool being drawn
            // For now, we'll just track the drag state
            ImPlotPoint current_pos = ImPlot::GetPlotMousePos();
            (void)current_pos;  // Suppress unused variable warning

            // When mouse is released, finalize the tool
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                // In a real implementation, we would create the tool based on the
                // selected tool type and the start/end points
                // For now, we'll just reset the drawing state
                is_drawing = false;
            }
        }
    }

    // Render context menu if right-clicked on plot
    render_context_menu(chart);

    // Render anchored VWAP overlays
    render_anchored_vwap_overlay(chart);

    // Render session VWAP overlays
    render_session_vwap_overlay(chart);

    // Render liquidity bars on the right-hand price axis
    if (show_liquidity_bars_) {
      render_liquidity_bars(chart);
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

// Helper method to render enhanced step profile histograms on candle bars
void ChartPanel::render_enhanced_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                                   const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                                   const std::vector<double>& x_coords,
                                                                   const std::vector<double>& y_coords_high,
                                                                   const std::vector<double>& y_coords_low,
                                                                   bool show_poc_line,
                                                                   int num_buckets_per_candle,
                                                                   float opacity,
                                                                   bool show_labels) {

  // This method implements the Step Profile rendering with additional visualization options
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get symbol ID to fetch recent trades
  auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
  if (!symbol_id_opt) {
    return;
  }

  uint32_t symbol_id = *symbol_id_opt;

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw enhanced step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, static_cast<int>(240 * opacity));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * opacity),
                            static_cast<int>(50 * volume_ratio * opacity), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * opacity), green_intensity,
                            static_cast<int>(50 * volume_ratio * opacity), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (opacity > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                           IM_COL32(0, 0, 0, static_cast<int>(80 * opacity)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, static_cast<int>(255 * opacity)),  // Bright yellow color for POC with adjustable opacity
                         2.5f);                        // Slightly thicker line for better visibility
    }

    // Optionally show labels for the mini histogram
    if (show_labels) {
      // Add a small label indicating this is a volume profile histogram
      char label[32];
      snprintf(label, sizeof(label), "%.0f", candle.volume);

      // Position the label at the top of the candle
      ImVec2 label_pos = ImVec2(x_center, y_high - 15.0f);

      // Draw the label with appropriate color
      draw_list->AddText(label_pos, IM_COL32(255, 255, 255, static_cast<int>(200 * opacity)), label);
    }
  }
}

// Method to handle mouse drag interaction for custom profile creation
void ChartPanel::handleMouseDragInteraction() {
  auto& interaction_mgr = InteractionManager::getInstance();

  // Check if we're in a state where we should handle mouse drag for profile creation
  if (!ImPlot::IsPlotHovered()) {
    return;
  }

  // Get the current mouse position in plot coordinates
  ImPlotPoint mouse_plot_pos = ImPlot::GetPlotMousePos();
  ImVec2 mouse_screen_pos = ImGui::GetMousePos();
  (void)mouse_screen_pos;  // Suppress unused variable warning

  // Convert ImPlotPoint to ImVec2 for interaction manager
  ImVec2 converted_plot_pos = ImVec2(static_cast<float>(mouse_plot_pos.x), static_cast<float>(mouse_plot_pos.y));

  // Check if left mouse button is pressed (starting drag) and no drag is currently active
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !interaction_mgr.isMouseDragActive()) {
    // Check if the user is holding a modifier key (e.g., Shift) to indicate custom profile creation
    if (ImGui::GetIO().KeyShift) {
      interaction_mgr.startTimeRangeSelection(converted_plot_pos);
    }
  }
  // If drag is active, update the position
  else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && interaction_mgr.isTimeRangeSelectionActive()) {
    interaction_mgr.updateTimeRangeSelection(converted_plot_pos);
  }
  // If mouse button is released, end the drag interaction
  else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && interaction_mgr.isTimeRangeSelectionActive()) {
    interaction_mgr.endTimeRangeSelection();

    // At this point, we have a completed time range selection
    // We could trigger profile creation or notify other components
    auto time_range = interaction_mgr.getTimeRangeSelection();

    // Optionally, we can set the volume profile panel to use this time range
    // This would require having access to the volume profile panel instance
    // For now, we'll just log the selection
    std::cout << "[ChartPanel] Time range selected: " << time_range.first << " to " << time_range.second << std::endl;
  }
}

void ChartPanel::center_on_timestamp(uint64_t timestamp) {
  // Convert the timestamp to the format used by the chart (seconds since epoch)
  double timestamp_seconds = static_cast<double>(timestamp) / 1000000.0; // Convert microseconds to seconds

  // Store the target timestamp to be used in the next render cycle
  // We can't directly set the plot limits from outside BeginPlot/EndPlot
  // So we'll store it and apply it during the next render
  last_view_min_ = timestamp_seconds - 10.0; // 10 seconds before
  last_view_max_ = timestamp_seconds + 10.0; // 10 seconds after
  follow_latest_ = false; // Disable auto-follow to keep the view centered
}

std::pair<uint64_t, uint64_t> ChartPanel::get_visible_time_range() const {
  // Return the currently visible time range in the chart
  // Convert from seconds (used by ImPlot) back to microseconds (our internal format)
  uint64_t start_time = static_cast<uint64_t>(last_view_min_ * 1000000);
  uint64_t end_time = static_cast<uint64_t>(last_view_max_ * 1000000);

  return {start_time, end_time};
}

// Render context menu when user right-clicks on the chart
void ChartPanel::render_context_menu(const ChartInstance& chart) {
  // Check if the plot is hovered and right mouse button was clicked
  if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    // Get the mouse position in plot coordinates
    ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();

    // Convert the x-coordinate (time) back to timestamp
    uint64_t clicked_timestamp = static_cast<uint64_t>(mouse_pos.x * 1000000); // Convert from seconds to microseconds
    (void)clicked_timestamp;  // Suppress unused variable warning

    // Find the closest candle to the clicked timestamp to determine the time range for the bar
    // Use binary search (std::lower_bound) to optimize from O(n) to O(log n)
    size_t closest_idx = 0;

    // Use lower_bound to find the insertion point for mouse_pos.x in the sorted dates vector
    auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_pos.x);

    if (lower == chart.dates.end()) {
        // Mouse x is beyond the last date, use the last element
        closest_idx = chart.dates.size() - 1;
    } else if (lower == chart.dates.begin()) {
        // Mouse x is before the first date, use the first element
        closest_idx = 0;
    } else {
        // Compare the distance to the element at lower and the one before it
        size_t idx_after = std::distance(chart.dates.begin(), lower);
        size_t idx_before = idx_after - 1;

        double dist_to_after = std::abs(chart.dates[idx_after] - mouse_pos.x);
        double dist_to_before = std::abs(chart.dates[idx_before] - mouse_pos.x);

        closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
    }

    // Calculate the time range for the clicked bar based on the timeframe
    uint64_t bar_duration = RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);
    uint64_t bar_start_time = static_cast<uint64_t>(chart.dates[closest_idx] * 1000000); // Convert to microseconds
    uint64_t bar_end_time = bar_start_time + bar_duration;

    // Store the time range for the clicked bar
    clicked_bar_start_time_ = bar_start_time;
    clicked_bar_end_time_ = bar_end_time;

    // Open the context menu
    ImGui::OpenPopup("ChartContextMenu");
  }

  // Create the context menu
  if (ImGui::BeginPopup("ChartContextMenu")) {
    if (ImGui::MenuItem("Show Trades for Bar")) {
      // Call the callback to show historical trades if available
      if (on_show_historical_trades_) {
        on_show_historical_trades_(clicked_bar_start_time_, clicked_bar_end_time_);
      } else {
        // Fallback to the popup if no callback is set
        show_trades_popup_ = true;
      }
    }

    if (ImGui::MenuItem("Anchor VWAP Here")) {
      // Get the current mouse position in plot coordinates
      ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();

      // Convert the x-coordinate (time) back to timestamp
      uint64_t anchor_timestamp = static_cast<uint64_t>(mouse_pos.x * 1000000); // Convert from seconds to microseconds

      // Create a new anchored VWAP at this timestamp
      create_anchored_vwap_at_time(anchor_timestamp);
    }

    ImGui::EndPopup();
  }
}

// Create an anchored VWAP at the specified timestamp
void ChartPanel::create_anchored_vwap_at_time(uint64_t timestamp) {
  // Create a new anchored VWAP with the given timestamp
  ::btq::AnchoredVWAP new_vwap(timestamp);

  // Get the chart data to calculate the VWAP
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    const ChartInstance& chart = it->second;

    // Convert the chart data to OHLCVCandle format for VWAP calculation
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      BTQuant::RenderEngine::OHLCVCandle bar;
      bar.timestamp = static_cast<uint64_t>(chart.dates[i] * 1000000); // Convert to microseconds
      bar.open = chart.opens[i];
      bar.high = chart.highs[i];
      bar.low = chart.lows[i];
      bar.close = chart.closes[i];
      bar.volume = chart.volumes[i];
      bar.trade_count = 1; // Placeholder value
      bars.push_back(bar);
    }

    // Calculate the VWAP from the anchor point
    new_vwap.calculate(bars);
  }

  // Add the new VWAP to our list
  anchored_vwaps_.push_back(new_vwap);
}

// Render the anchored VWAP overlay on the chart
void ChartPanel::render_anchored_vwap_overlay(const ChartInstance& chart) {
  if (anchored_vwaps_.empty()) {
    return;
  }

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Iterate through all anchored VWAPs and render them
  for (const auto& vwap : anchored_vwaps_) {
    const auto& vwap_values = vwap.getVWAPValues();
    const auto& sd1_upper = vwap.getSD1UpperBand();
    const auto& sd1_lower = vwap.getSD1LowerBand();
    const auto& sd2_upper = vwap.getSD2UpperBand();
    const auto& sd2_lower = vwap.getSD2LowerBand();
    const auto& sd3_upper = vwap.getSD3UpperBand();
    const auto& sd3_lower = vwap.getSD3LowerBand();

    // Find the starting index in the chart data that corresponds to the anchor timestamp
    uint64_t anchor_timestamp = vwap.getAnchorTimestamp();
    double anchor_time_seconds = static_cast<double>(anchor_timestamp) / 1000000.0;

    // Find the index in the chart where the anchor timestamp occurs
    size_t start_idx = 0;
    bool found_anchor = false;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      if (chart.dates[i] >= anchor_time_seconds) {
        start_idx = i;
        found_anchor = true;
        break;
      }
    }

    if (!found_anchor) {
      continue; // Anchor timestamp not found in current chart data
    }

    // Render the VWAP line as a smooth polyline with anti-aliasing
    if (vwap_values.size() > 1) {
      // Prepare points for polyline
      std::vector<ImVec2> points;
      points.reserve(vwap_values.size());

      for (size_t i = 0; i < vwap_values.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 point = ImPlot::PlotToPixels(chart.dates[start_idx + i], vwap_values[i]);
        points.push_back(point);
      }

      if (points.size() > 1) {
        // Draw the VWAP line as a smooth polyline in yellow with anti-aliasing
        draw_list->AddPolyline(points.data(), static_cast<int>(points.size()),
                              IM_COL32(255, 255, 0, 255), ImDrawListFlags_AntiAliasedLines, 2.0f);
      }
    }

    // Render SD1 bands as semi-transparent filled regions
    if (sd1_upper.size() > 1 && sd1_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd1_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd1_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd1_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD1 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(255, 255, 0, 80));  // Semi-transparent yellow
      }
    }

    // Render SD2 bands as semi-transparent filled regions
    if (sd2_upper.size() > 1 && sd2_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd2_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd2_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd2_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD2 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(0, 255, 255, 60));  // Semi-transparent cyan
      }
    }

    // Render SD3 bands as semi-transparent filled regions
    if (sd3_upper.size() > 1 && sd3_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd3_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd3_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd3_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD3 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(255, 0, 255, 40));  // More transparent magenta
      }
    }
  }
}

// Render the session VWAP overlay on the chart
void ChartPanel::render_session_vwap_overlay(const ChartInstance& chart) {
  if (chart.dates.empty()) {
    return;
  }

  // Get all sessions and render them (already calculated in update method)
  const auto& sessions = session_vwap_.getSessions();
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  for (const auto& session : sessions) {
    const auto& vwap_values = session.vwapValues;
    const auto& sd1_upper = session.sd1UpperBand;
    const auto& sd1_lower = session.sd1LowerBand;
    const auto& sd2_upper = session.sd2UpperBand;
    const auto& sd2_lower = session.sd2LowerBand;
    const auto& sd3_upper = session.sd3UpperBand;
    const auto& sd3_lower = session.sd3LowerBand;

    // Find the starting index in the chart data that corresponds to the session start time
    double session_start_seconds = static_cast<double>(session.startTime) / 1000000.0;

    // Find the index in the chart where the session starts
    size_t start_idx = 0;
    bool found_start = false;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      if (chart.dates[i] >= session_start_seconds) {
        start_idx = i;
        found_start = true;
        break;
      }
    }

    if (!found_start) {
      continue; // Session start time not found in current chart data
    }

    // Render the VWAP line as a smooth polyline with anti-aliasing
    if (vwap_values.size() > 1) {
      // Prepare points for polyline
      std::vector<ImVec2> points;
      points.reserve(vwap_values.size());

      for (size_t i = 0; i < vwap_values.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 point = ImPlot::PlotToPixels(chart.dates[start_idx + i], vwap_values[i]);
        points.push_back(point);
      }

      if (points.size() > 1) {
        // Use different colors for active vs historical sessions
        ImU32 color = session.isActive ? IM_COL32(0, 255, 255, 255) : IM_COL32(128, 128, 128, 200); // Cyan for active, gray for historical

        // Draw the VWAP line as a smooth polyline in different colors based on session status
        draw_list->AddPolyline(points.data(), static_cast<int>(points.size()),
                              color, ImDrawListFlags_AntiAliasedLines, 2.0f);
      }
    }

    // Render SD1 bands as semi-transparent filled regions
    if (sd1_upper.size() > 1 && sd1_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd1_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd1_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd1_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD1 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd1_color = session.isActive ? IM_COL32(0, 255, 255, 80) : IM_COL32(128, 128, 128, 60); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd1_color);
      }
    }

    // Render SD2 bands as semi-transparent filled regions
    if (sd2_upper.size() > 1 && sd2_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd2_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd2_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd2_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD2 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd2_color = session.isActive ? IM_COL32(0, 200, 200, 60) : IM_COL32(100, 100, 100, 40); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd2_color);
      }
    }

    // Render SD3 bands as semi-transparent filled regions
    if (sd3_upper.size() > 1 && sd3_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd3_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd3_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd3_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD3 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd3_color = session.isActive ? IM_COL32(0, 150, 150, 40) : IM_COL32(80, 80, 80, 20); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd3_color);
      }
    }
  }

}

void ChartPanel::render_trades_popup() {
  // This method is kept for backward compatibility but will be replaced by the HistoricalTimeSalesPanel popup
  // The actual popup is now handled by the HistoricalTimeSalesPanel.show_trades_popup method
  // which is called from the render method
}

// Multi-timeframe indicator methods implementation
void ChartPanel::add_multi_timeframe_indicator(const std::string& name, bool visible, ImVec4 color, int period, RenderEngine::TimeFrame source_timeframe) {
  multi_tf_indicators_.emplace_back(name, visible, color, period, source_timeframe);
}

void ChartPanel::remove_multi_timeframe_indicator(int index) {
  if (index >= 0 && index < static_cast<int>(multi_tf_indicators_.size())) {
    multi_tf_indicators_.erase(multi_tf_indicators_.begin() + index);
  }
}

void ChartPanel::update_multi_timeframe_indicators(const ChartInstance& chart) {
  (void)chart;  // Suppress unused parameter warning
  // Update multi-timeframe indicators by fetching data from the source timeframe
  for (auto& indicator : multi_tf_indicators_) {
    if (!indicator.isVisible) continue; // Skip invisible indicators

    // Get symbol ID to fetch data from the source timeframe
    auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
    if (!symbol_id_opt) continue;

    uint32_t symbol_id = *symbol_id_opt;

    // Get candles from the source timeframe
    auto source_candles = processor_->getCandles(symbol_id, indicator.source_timeframe);

    if (!source_candles.empty()) {
      // Extract closing prices from the source timeframe
      std::vector<float> source_closes;
      source_closes.reserve(source_candles.size());

      for (const auto& candle : source_candles) {
        source_closes.push_back(static_cast<float>(candle.close));
      }

      // Calculate the indicator values based on the indicator type
      if (indicator.name.find("SMA") != std::string::npos) {
        // Calculate SMA for the source timeframe
        indicator.values = calculate_cached_sma(source_closes, indicator.period);

        // Store the timestamps for alignment with the current chart
        indicator.timestamps.clear();
        for (const auto& candle : source_candles) {
          indicator.timestamps.push_back(static_cast<double>(candle.timestamp) / 1000000.0); // Convert microseconds to seconds
        }
      } else if (indicator.name.find("EMA") != std::string::npos) {
        // Calculate EMA for the source timeframe
        indicator.values = calculate_cached_ema(source_closes, indicator.period);

        // Store the timestamps for alignment with the current chart
        indicator.timestamps.clear();
        for (const auto& candle : source_candles) {
          indicator.timestamps.push_back(static_cast<double>(candle.timestamp) / 1000000.0); // Convert microseconds to seconds
        }
      }
      // Add other indicator types as needed (RSI, etc.)
    }
  }
}

void ChartPanel::render_multi_timeframe_indicators(const ChartInstance& chart, size_t start_idx, size_t end_idx) {
  if (chart.dates.empty() || multi_tf_indicators_.empty()) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  for (const auto& indicator : multi_tf_indicators_) {
    if (!indicator.isVisible || indicator.values.empty() || indicator.timestamps.empty()) continue;

    // Find the range of the multi-timeframe indicator values that overlap with the current chart's visible range
    double visible_start_time = chart.dates[start_idx];
    double visible_end_time = chart.dates[end_idx - 1];

    // Find the indices in the multi-timeframe data that correspond to the visible range
    size_t mt_start_idx = 0;
    size_t mt_end_idx = indicator.timestamps.size();

    // Find start index
    for (size_t i = 0; i < indicator.timestamps.size(); ++i) {
      if (indicator.timestamps[i] >= visible_start_time) {
        mt_start_idx = i;
        break;
      }
    }

    // Find end index
    for (size_t i = indicator.timestamps.size(); i > 0; --i) {
      if (indicator.timestamps[i - 1] <= visible_end_time) {
        mt_end_idx = i;
        break;
      }
    }

    // Draw the multi-timeframe indicator lines
    for (size_t i = mt_start_idx; i < mt_end_idx - 1 && i < indicator.values.size() - 1; ++i) {
      // Find the closest date in the current chart for the multi-timeframe timestamp
      auto current_timestamp_it = std::lower_bound(chart.dates.begin(), chart.dates.end(), indicator.timestamps[i]);
      auto next_timestamp_it = std::lower_bound(chart.dates.begin(), chart.dates.end(), indicator.timestamps[i + 1]);

      // If we can't find exact matches, we'll interpolate or find the closest points
      size_t current_chart_idx = 0;
      size_t next_chart_idx = 0;

      if (current_timestamp_it != chart.dates.end()) {
        current_chart_idx = std::distance(chart.dates.begin(), current_timestamp_it);
        // Ensure we don't go out of bounds
        if (current_chart_idx >= chart.dates.size()) current_chart_idx = chart.dates.size() - 1;
      } else {
        // If timestamp is beyond the chart data, use the last available index
        current_chart_idx = chart.dates.size() - 1;
      }

      if (next_timestamp_it != chart.dates.end()) {
        next_chart_idx = std::distance(chart.dates.begin(), next_timestamp_it);
        // Ensure we don't go out of bounds
        if (next_chart_idx >= chart.dates.size()) next_chart_idx = chart.dates.size() - 1;
      } else {
        // If timestamp is beyond the chart data, use the last available index
        next_chart_idx = chart.dates.size() - 1;
      }

      // Ensure we have valid indices and values
      if (i < indicator.values.size() && (i + 1) < indicator.values.size() &&
          current_chart_idx < chart.dates.size() && next_chart_idx < chart.dates.size()) {

        // Draw line segments connecting the multi-timeframe indicator values to the appropriate chart positions
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[current_chart_idx], indicator.values[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[next_chart_idx], indicator.values[i + 1]);

        // Convert color to ImU32
        ImU32 color = ImGui::ColorConvertFloat4ToU32(indicator.color);

        // Draw the line segment with a slightly different style to distinguish from regular indicators
        draw_list->AddLine(p1, p2, color, 2.5f); // Slightly thicker line for visibility
      }
    }

    // Additionally, for better visualization of multi-timeframe indicators, draw points at the exact
    // multi-timeframe data points that align with the chart
    for (size_t i = mt_start_idx; i < mt_end_idx && i < indicator.values.size(); ++i) {
      // Find the closest date in the current chart for the multi-timeframe timestamp
      auto chart_it = std::lower_bound(chart.dates.begin(), chart.dates.end(), indicator.timestamps[i]);

      if (chart_it != chart.dates.end()) {
        size_t chart_idx = std::distance(chart.dates.begin(), chart_it);

        if (chart_idx < chart.dates.size()) {
          ImVec2 point = ImPlot::PlotToPixels(chart.dates[chart_idx], indicator.values[i]);

          // Draw a small circle to mark the exact multi-timeframe data point
          ImU32 color = ImGui::ColorConvertFloat4ToU32(indicator.color);
          draw_list->AddCircleFilled(point, 3.0f, color);
        }
      }
    }
  }
}

std::vector<double> ChartPanel::get_indicator_values_from_timeframe(const std::string& indicator_name, int period, RenderEngine::TimeFrame timeframe, uint32_t symbol_id) {
  // Get candles from the specified timeframe
  auto candles = processor_->getCandles(symbol_id, timeframe);

  if (candles.empty()) {
    return std::vector<double>();
  }

  // Extract closing prices
  std::vector<float> closes;
  closes.reserve(candles.size());

  for (const auto& candle : candles) {
    closes.push_back(static_cast<float>(candle.close));
  }

  // Calculate the requested indicator
  if (indicator_name == "SMA" || indicator_name.find("SMA") != std::string::npos) {
    return calculate_cached_sma(closes, period);
  } else if (indicator_name == "EMA" || indicator_name.find("EMA") != std::string::npos) {
    return calculate_cached_ema(closes, period);
  } else if (indicator_name == "RSI" || indicator_name.find("RSI") != std::string::npos) {
    return calculate_cached_rsi(closes, period);
  }

  // Return empty vector if indicator type is not supported
  return std::vector<double>();
}

// Set callback for showing historical trades
void ChartPanel::set_show_historical_trades_callback(std::function<void(uint64_t, uint64_t)> callback) {
  on_show_historical_trades_ = [this, callback](uint64_t start_time, uint64_t end_time) {
    if (callback) {
      callback(start_time, end_time);
    } else if (historical_time_sales_panel_) {
      // Show the trades in the historical time & sales panel
      historical_time_sales_panel_->show_trades_popup(start_time, end_time, symbol_);
    }
  };
}

  // Panel settings methods
  void ChartPanel::open_settings() {
    if (settings_) {
      settings_->open();
    }
  }

  void ChartPanel::render_context_menu() {
    // Create a context menu that appears when right-clicking on the panel
    if (ImGui::BeginPopup("PanelContextMenu")) {
      if (ImGui::MenuItem("Panel Settings")) {
        open_settings();
      }

      ImGui::EndPopup();
    }
  }

// Method to update liquidity data from the market data processor
void ChartPanel::update_liquidity_data() {
  if (!processor_ || symbol_.empty()) return;

  // Get the current symbol ID from the chart manager
  auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
  if (!symbol_id_opt) return;

  uint32_t current_symbol_id = *symbol_id_opt;

  // Get the current orderbook data for the symbol
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id);
  if (!orderbook_opt) return;

  const auto& orderbook = *orderbook_opt;

  // Clear existing liquidity levels
  liquidity_levels_.clear();

  // Add bid levels (green bars)
  for (const auto& bid : orderbook.bids) {
    liquidity_levels_.emplace_back(bid.price, bid.size, true);
  }

  // Add ask levels (red bars)
  for (const auto& ask : orderbook.asks) {
    liquidity_levels_.emplace_back(ask.price, ask.size, false);
  }

  // Find max volume for scaling
  max_liquidity_volume_ = 1.0;
  for (const auto& level : liquidity_levels_) {
    if (level.volume > max_liquidity_volume_) {
      max_liquidity_volume_ = level.volume;
    }
  }
}

// Method to render liquidity bars on the right-hand price axis
void ChartPanel::render_liquidity_bars(const ChartInstance& /*chart*/) {
  if (liquidity_levels_.empty()) {
    update_liquidity_data();
    if (liquidity_levels_.empty()) return;
  }

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  if (!draw_list) return;

  // Get the plot limits to determine the coordinate system
  ImPlotRect limits = ImPlot::GetPlotLimits();

  // Get the plot position and size to calculate the right edge
  ImVec2 plot_pos = ImPlot::GetPlotPos();
  ImVec2 plot_size = ImPlot::GetPlotSize();
  ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

  // Calculate the right edge of the plot area in screen coordinates
  float right_edge_x = canvas_pos.x + plot_pos.x + plot_size.x;

  // Draw liquidity bars for each level
  for (const auto& level : liquidity_levels_) {
    // Convert the price to Y coordinate
    ImVec2 level_pos = ImPlot::PlotToPixels(limits.X.Max, level.price); // Use rightmost X for liquidity bars
    
    // Calculate bar width based on volume (relative to max volume)
    float bar_width = (static_cast<float>(level.volume) / static_cast<float>(max_liquidity_volume_)) * liquidity_bar_width_;
    
    // Determine color based on bid/ask
    ImVec4 color = level.is_bid ? liquidity_bids_color_ : liquidity_asks_color_;
    color.w *= liquidity_bar_opacity_; // Apply opacity
    ImU32 im_color = ImGui::ColorConvertFloat4ToU32(color);
    
    // Calculate the top and bottom Y positions for the bar (small height to make it look like a line)
    float bar_height = 2.0f; // Small height to make it appear as a horizontal line
    
    // Calculate the left edge of the bar (extending from the right axis inward)
    float bar_left_x = right_edge_x - bar_width;
    
    // Draw the liquidity bar as a horizontal line extending from the right axis
    ImVec2 bar_start = ImVec2(bar_left_x, level_pos.y - bar_height/2);
    ImVec2 bar_end = ImVec2(right_edge_x, level_pos.y + bar_height/2);
    
    draw_list->AddRectFilled(bar_start, bar_end, im_color);
  }
}

// ============================================================================
// QUANTOWER-STYLE 5-PART LAYOUT IMPLEMENTATION (Phase 3)
// ============================================================================

// 3.1 Top Toolbar (Main Controls)
void ChartPanel::render_top_toolbar() {
  // Render a horizontal ImGui bar at the top
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
  ImGui::BeginChild("TopBar", ImVec2(0, 30), true, ImGuiWindowFlags_NoScrollbar);
  
  // Symbol Lookup (InputText)
  ImGui::PushItemWidth(100);
  if (ImGui::InputText("##Symbol", symbol_input_buffer_, sizeof(symbol_input_buffer_), 
                       ImGuiInputTextFlags_EnterReturnsTrue)) {
    set_symbol(symbol_input_buffer_, exchange_);
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    set_symbol(symbol_input_buffer_, exchange_);
  }
  ImGui::PopItemWidth();
  
  // Tooltip for symbol input
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Enter symbol (e.g., BTC-USDT)");
  }
  
  ImGui::SameLine();
  
  // Timeframe Selector (Dropdown: 1m, 5m, 1H, 1D - simplified for toolbar)
  const char* toolbar_timeframes[] = {"1m", "5m", "15m", "1h", "4h", "1d"};
  int tf_index = 0;
  // Map current timeframe to index
  switch (timeframe_) {
    case RenderEngine::TimeFrame::TF_1MIN: tf_index = 0; break;
    case RenderEngine::TimeFrame::TF_5MIN: tf_index = 1; break;
    case RenderEngine::TimeFrame::TF_15MIN: tf_index = 2; break;
    case RenderEngine::TimeFrame::TF_1HOUR: tf_index = 3; break;
    case RenderEngine::TimeFrame::TF_4HOUR: tf_index = 4; break;
    case RenderEngine::TimeFrame::TF_1DAY: tf_index = 5; break;
    default: tf_index = 0; break;
  }
  
  ImGui::PushItemWidth(60);
  if (ImGui::Combo("##TF", &tf_index, toolbar_timeframes, IM_ARRAYSIZE(toolbar_timeframes))) {
    RenderEngine::TimeFrame new_tf = RenderEngine::TimeFrame::TF_1MIN;
    switch (tf_index) {
      case 0: new_tf = RenderEngine::TimeFrame::TF_1MIN; break;
      case 1: new_tf = RenderEngine::TimeFrame::TF_5MIN; break;
      case 2: new_tf = RenderEngine::TimeFrame::TF_15MIN; break;
      case 3: new_tf = RenderEngine::TimeFrame::TF_1HOUR; break;
      case 4: new_tf = RenderEngine::TimeFrame::TF_4HOUR; break;
      case 5: new_tf = RenderEngine::TimeFrame::TF_1DAY; break;
    }
    set_timeframe(new_tf);
  }
  ImGui::PopItemWidth();
  
  ImGui::SameLine();
  
  // Chart Style (Dropdown: Candle, Bar, Line, Area, Quantower)
  const char* chart_styles[] = {"Candle", "Bar", "Line", "Area", "Quantower"};
  int style_index = static_cast<int>(chart_style_);
  ImGui::PushItemWidth(80);
  if (ImGui::Combo("##Style", &style_index, chart_styles, IM_ARRAYSIZE(chart_styles))) {
    chart_style_ = static_cast<ChartStyle>(style_index);
  }
  ImGui::PopItemWidth();
  
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  
  // Mouse Trading vs. Keyboard Trading toggle button
  const char* trading_mode_label = (trading_mode_ == TradingMode::MOUSE_TRADING) ? "Mouse" : "Keyboard";
  ImVec4 button_color = (trading_mode_ == TradingMode::MOUSE_TRADING) 
                        ? ImVec4(0.2f, 0.6f, 0.2f, 1.0f)  // Green for mouse
                        : ImVec4(0.6f, 0.4f, 0.2f, 1.0f); // Orange for keyboard
  
  ImGui::PushStyleColor(ImGuiCol_Button, button_color);
  if (ImGui::Button(trading_mode_label, ImVec2(70, 0))) {
    trading_mode_ = (trading_mode_ == TradingMode::MOUSE_TRADING) 
                    ? TradingMode::KEYBOARD_TRADING 
                    : TradingMode::MOUSE_TRADING;
  }
  ImGui::PopStyleColor();
  
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Toggle between Mouse Trading and Keyboard Trading modes");
  }
  
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  
  // Auto-follow checkbox
  ImGui::Checkbox("Auto-follow", &follow_latest_);
  
  ImGui::SameLine();
  
  // Price Centering Mode indicator
  const char* centering_modes[] = {"Auto", "Centered", "In View", "Manual"};
  ImGui::Text("Y: %s", centering_modes[static_cast<int>(price_centering_mode_)]);
  
  ImGui::EndChild();
  ImGui::PopStyleVar();
}

// 3.2 Left Sidebar (Tools & Objects)
void ChartPanel::render_left_sidebar() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2, 4));
  ImGui::BeginChild("Sidebar", ImVec2(40, 0), true);
  
  // Crosshair button
  ImVec4 crosshair_color = show_crosshair_ ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, crosshair_color);
  if (ImGui::Button("+", ImVec2(32, 32))) {
    show_crosshair_ = !show_crosshair_;
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Crosshair");
  
  // Drawing Tools button
  ImGui::Spacing();
  ImVec4 drawing_color = show_drawing_tools_sidebar_ ? ImVec4(0.8f, 0.6f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, drawing_color);
  if (ImGui::Button("D", ImVec2(32, 32))) {
    show_drawing_tools_sidebar_ = !show_drawing_tools_sidebar_;
    if (show_drawing_tools_sidebar_) {
      ImGui::OpenPopup("DrawingToolsPopup");
    }
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drawing Tools");
  
  // Overlays button
  ImGui::Spacing();
  ImVec4 overlays_color = show_overlays_menu_ ? ImVec4(0.2f, 0.6f, 0.8f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, overlays_color);
  if (ImGui::Button("O", ImVec2(32, 32))) {
    show_overlays_menu_ = !show_overlays_menu_;
    if (show_overlays_menu_) {
      ImGui::OpenPopup("OverlaysPopup");
    }
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overlays");
  
  // Indicators button
  ImGui::Spacing();
  ImVec4 indicators_color = show_indicators_menu_ ? ImVec4(0.6f, 0.2f, 0.8f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, indicators_color);
  if (ImGui::Button("I", ImVec2(32, 32))) {
    show_indicators_menu_ = !show_indicators_menu_;
    if (show_indicators_menu_) {
      ImGui::OpenPopup("IndicatorsPopup");
    }
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Indicators");
  
  // Separator
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  // Favorite tools section
  ImGui::Text("Fav");
  ImGui::Spacing();
  
  // Initialize favorite tools if empty
  if (favorite_tools_.empty()) {
    favorite_tools_.emplace_back("Horizontal Line", "H", false);
    favorite_tools_.emplace_back("Trend Line", "T", false);
    favorite_tools_.emplace_back("Fibonacci", "F", true);
    favorite_tools_.emplace_back("Rectangle", "R", false);
  }
  
  // Render favorite tools
  for (size_t i = 0; i < favorite_tools_.size(); ++i) {
    const auto& tool = favorite_tools_[i];
    ImVec4 fav_color = tool.is_favorite ? ImVec4(1.0f, 0.8f, 0.0f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, fav_color);
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::Button(tool.icon.c_str(), ImVec2(32, 28))) {
      selected_drawing_tool_ = static_cast<int>(i);
    }
    ImGui::PopID();
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.name.c_str());
    ImGui::Spacing();
  }
  
  // Render popups
  render_drawing_tools_popup();
  render_overlays_popup();
  render_indicators_popup();
  
  ImGui::EndChild();
  ImGui::PopStyleVar();
}

void ChartPanel::render_drawing_tools_popup() {
  if (ImGui::BeginPopup("DrawingToolsPopup")) {
    ImGui::Text("Drawing Tools");
    ImGui::Separator();
    
    const char* tools[] = {"Horizontal Line", "Vertical Line", "Trend Line", 
                           "Fibonacci", "Rectangle", "Text", "Arrow"};
    
    for (int i = 0; i < IM_ARRAYSIZE(tools); ++i) {
      bool is_selected = (selected_drawing_tool_ == i);
      if (ImGui::Selectable(tools[i], is_selected)) {
        selected_drawing_tool_ = i;
      }
      
      // Context menu for favorites
      if (ImGui::BeginPopupContextItem()) {
        bool is_fav = false;
        for (const auto& fav : favorite_tools_) {
          if (fav.name == tools[i]) {
            is_fav = fav.is_favorite;
            break;
          }
        }
        if (ImGui::Checkbox("Add to Favorites", &is_fav)) {
          toggle_favorite_tool(tools[i]);
        }
        ImGui::EndPopup();
      }
    }
    
    ImGui::EndPopup();
  }
}

void ChartPanel::render_overlays_popup() {
  if (ImGui::BeginPopup("OverlaysPopup")) {
    ImGui::Text("Overlays");
    ImGui::Separator();
    
    ImGui::Checkbox("Volume Profile", &indicator_config_.show_volume_profile);
    ImGui::Checkbox("Bollinger Bands", &indicator_config_.show_bollinger);
    ImGui::Checkbox("Fibonacci Levels", &indicator_config_.show_fibonacci);
    ImGui::Checkbox("Session VWAP", &show_session_vwap_);
    
    // Anchored VWAP section
    ImGui::Separator();
    ImGui::Text("Anchored VWAPs: %zu", anchored_vwaps_.size());
    
    ImGui::EndPopup();
  }
}

void ChartPanel::render_indicators_popup() {
  if (ImGui::BeginPopup("IndicatorsPopup")) {
    ImGui::Text("Indicators");
    ImGui::Separator();
    
    // Moving Averages
    ImGui::Text("Moving Averages");
    if (ImGui::Checkbox("SMA 9", &indicator_config_.show_sma_9)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("SMA 20", &indicator_config_.show_sma_20)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("SMA 50", &indicator_config_.show_sma_50)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("SMA 200", &indicator_config_.show_sma_200)) {
      sync_active_indicators_with_config();
    }
    
    ImGui::Separator();
    
    if (ImGui::Checkbox("EMA 9", &indicator_config_.show_ema_9)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("EMA 21", &indicator_config_.show_ema_21)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("EMA 50", &indicator_config_.show_ema_50)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("EMA 200", &indicator_config_.show_ema_200)) {
      sync_active_indicators_with_config();
    }
    
    ImGui::Separator();
    
    // Oscillators
    ImGui::Text("Oscillators");
    if (ImGui::Checkbox("RSI", &indicator_config_.show_rsi)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("MACD", &indicator_config_.show_macd)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("Stochastic", &indicator_config_.show_stochastic)) {
      sync_active_indicators_with_config();
    }
    if (ImGui::Checkbox("ATR", &indicator_config_.show_atr)) {
      sync_active_indicators_with_config();
    }
    
    ImGui::EndPopup();
  }
}

void ChartPanel::toggle_favorite_tool(const std::string& tool_name) {
  // Find if tool exists in favorites
  auto it = std::find_if(favorite_tools_.begin(), favorite_tools_.end(),
                         [&tool_name](const FavoriteTool& t) { return t.name == tool_name; });
  
  if (it != favorite_tools_.end()) {
    // Toggle favorite status
    it->is_favorite = !it->is_favorite;
    
    // Remove if no longer favorite
    if (!it->is_favorite) {
      favorite_tools_.erase(it);
    }
  } else {
    // Add to favorites with first letter as icon
    std::string icon = tool_name.substr(0, 1);
    favorite_tools_.emplace_back(tool_name, icon, true);
  }
}

// 3.3 Price Centering Implementation
void ChartPanel::apply_price_centering_mode(const ChartInstance& chart, double last_price) {
  if (chart.closes.empty() || last_price <= 0) return;
  
  ImPlotRect limits = ImPlot::GetPlotLimits();
  
  switch (price_centering_mode_) {
    case PriceCenteringMode::AUTO:
      // Standard ImPlot AutoFit - let ImPlot handle it
      // This is the default behavior, no manual intervention needed
      break;
      
    case PriceCenteringMode::AUTO_CENTERED: {
      // Center on last price: (Y_max + Y_min)/2 == last_price
      double y_range = limits.Y.Max - limits.Y.Min;
      double half_range = y_range / 2.0;
      
      double new_y_min = last_price - half_range;
      double new_y_max = last_price + half_range;
      
      ImPlot::SetNextAxisLimits(ImAxis_Y1, new_y_min, new_y_max, ImGuiCond_Always);
      break;
    }
    
    case PriceCenteringMode::KEEP_IN_VIEW: {
      // Only adjust Y limits if last_price exceeds current bounds
      double y_min = limits.Y.Min;
      double y_max = limits.Y.Max;
      double margin = (y_max - y_min) * 0.1;  // 10% margin
      
      bool needs_adjustment = false;
      
      if (last_price < y_min + margin) {
        // Price is too low, shift down
        y_min = last_price - margin * 2;
        y_max = y_min + (limits.Y.Max - limits.Y.Min);
        needs_adjustment = true;
      } else if (last_price > y_max - margin) {
        // Price is too high, shift up
        y_max = last_price + margin * 2;
        y_min = y_max - (limits.Y.Max - limits.Y.Min);
        needs_adjustment = true;
      }
      
      if (needs_adjustment) {
        ImPlot::SetNextAxisLimits(ImAxis_Y1, y_min, y_max, ImGuiCond_Always);
      }
      break;
    }
    
    case PriceCenteringMode::MANUAL:
      // Disable all auto-fitting - use stored manual limits
      ImPlot::SetNextAxisLimits(ImAxis_Y1, manual_y_min_, manual_y_max_, ImGuiCond_Always);
      break;
  }
}

void ChartPanel::handle_y_axis_context_menu() {
  // Right-click on Y-Axis for mode selection
  if (ImGui::BeginPopup("YAxisContextMenu")) {
    ImGui::Text("Price Centering Mode");
    ImGui::Separator();
    
    int mode = static_cast<int>(price_centering_mode_);
    if (ImGui::RadioButton("Auto", mode == 0)) {
      price_centering_mode_ = PriceCenteringMode::AUTO;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Standard ImPlot AutoFit");
    
    if (ImGui::RadioButton("Auto Centered", mode == 1)) {
      price_centering_mode_ = PriceCenteringMode::AUTO_CENTERED;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Center on last price");
    
    if (ImGui::RadioButton("Keep in View", mode == 2)) {
      price_centering_mode_ = PriceCenteringMode::KEEP_IN_VIEW;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Only adjust if price exceeds bounds");
    
    if (ImGui::RadioButton("Manual", mode == 3)) {
      price_centering_mode_ = PriceCenteringMode::MANUAL;
      // Store current limits as manual limits
      ImPlotRect limits = ImPlot::GetPlotLimits();
      manual_y_min_ = limits.Y.Min;
      manual_y_max_ = limits.Y.Max;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Disable auto-fitting");
    
    ImGui::EndPopup();
  }
}

void ChartPanel::render_snap_to_last_button() {
  // Only visible if X-axis max < current time
  if (!show_snap_to_last_) return;
  
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
  if (ImGui::Button("Snap to Last", ImVec2(80, 0))) {
    // Reset X-axis to follow live data
    follow_latest_ = true;
    show_snap_to_last_ = false;
  }
  ImGui::PopStyleColor();
  
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Reset X-axis to follow live data");
  }
}

// 3.4 Right Sidebar Order Entry
void ChartPanel::render_right_sidebar_order_entry() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
  ImGui::BeginChild("OrderEntry", ImVec2(120, 0), true);
  
  ImGui::Text("Quick Order");
  ImGui::Separator();
  
  // Update cached quotes from atomic snapshot
  update_cached_quotes();
  
  // Market Buy button with Best Ask
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
  std::string buy_label = "BUY\n" + std::to_string(cached_best_ask_);
  if (ImGui::Button(buy_label.c_str(), ImVec2(100, 40))) {
    execute_market_order(true);  // Buy
  }
  ImGui::PopStyleColor();
  
  ImGui::Spacing();
  
  // Market Sell button with Best Bid
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.0f, 0.0f, 1.0f));
  std::string sell_label = "SELL\n" + std::to_string(cached_best_bid_);
  if (ImGui::Button(sell_label.c_str(), ImVec2(100, 40))) {
    execute_market_order(false);  // Sell
  }
  ImGui::PopStyleColor();
  
  ImGui::Separator();
  
  // Order Quantity input
  ImGui::Text("Quantity:");
  ImGui::PushItemWidth(90);
  ImGui::InputDouble("##Qty", &order_quantity_, 0.1, 1.0, "%.4f");
  ImGui::PopItemWidth();
  
  ImGui::Spacing();
  
  // Time In Force selector
  ImGui::Text("TIF:");
  const char* tif_options[] = {"GTC", "IOC", "FOK", "DAY"};
  int tif_index = static_cast<int>(selected_tif_);
  ImGui::PushItemWidth(90);
  if (ImGui::Combo("##TIF", &tif_index, tif_options, IM_ARRAYSIZE(tif_options))) {
    selected_tif_ = static_cast<TimeInForce>(tif_index);
  }
  ImGui::PopItemWidth();
  
  ImGui::Separator();
  
  // Display current quotes
  ImGui::Text("Best Bid: %.2f", cached_best_bid_);
  ImGui::Text("Best Ask: %.2f", cached_best_ask_);
  ImGui::Text("Spread: %.4f", cached_best_ask_ - cached_best_bid_);
  
  ImGui::EndChild();
  ImGui::PopStyleVar();
}

void ChartPanel::update_cached_quotes() {
  if (!processor_ || symbol_.empty()) return;
  
  // Get symbol ID
  auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
  if (!symbol_id_opt) return;
  
  uint32_t symbol_id = *symbol_id_opt;
  
  // Read from atomic snapshot (Phase 4.1 requirement)
  // This is a non-blocking call that reads from atomic data
  auto snapshot_opt = processor_->get_atomic_snapshot(symbol_id);
  if (snapshot_opt) {
    const auto& snapshot = *snapshot_opt;
    cached_best_bid_ = snapshot.best_bid;
    cached_best_ask_ = snapshot.best_ask;
    last_quote_update_ = snapshot.timestamp;
  }
}

void ChartPanel::execute_market_order(bool is_buy) {
  // Phase 4.2: Push to SPSC queue for async execution
  // The UI thread never waits for the HTTP/WebSocket response
  
  // Get symbol ID
  auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
  uint32_t sym_id = symbol_id_opt.value_or(0);
  
  // Create the trade command
  RenderEngine::TradeCommand cmd(
    sym_id,
    symbol_,
    exchange_,
    is_buy ? RenderEngine::OrderSide::BUY : RenderEngine::OrderSide::SELL,
    RenderEngine::OrderType::MARKET,
    order_quantity_,
    static_cast<RenderEngine::TimeInForce>(selected_tif_)
  );
  
  // Set additional fields
  cmd.price = is_buy ? cached_best_ask_ : cached_best_bid_;
  
  // Push to the global SPSC queue (non-blocking)
  bool pushed = RenderEngine::GlobalTradeQueue::push_command(std::move(cmd));
  
  if (pushed) {
    std::cout << "[ChartPanel] Market Order QUEUED: "
              << (is_buy ? "BUY" : "SELL") << " "
              << order_quantity_ << " " << symbol_
              << " @ " << (is_buy ? cached_best_ask_ : cached_best_bid_)
              << " TIF: " << static_cast<int>(selected_tif_)
              << std::endl;
  } else {
    std::cerr << "[ChartPanel] ERROR: Failed to queue order - SPSC queue full!" << std::endl;
  }
}

// 3.5 Bottom Toolbar (Volume Analysis)
void ChartPanel::render_bottom_toolbar() {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
  ImGui::BeginChild("BottomBar", ImVec2(0, 30), true, ImGuiWindowFlags_NoScrollbar);
  
  // Volume Profile toggle
  ImVec4 vp_color = show_volume_profile_overlay_ ? ImVec4(0.2f, 0.6f, 0.8f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, vp_color);
  if (ImGui::Button("Vol Profile", ImVec2(80, 0))) {
    show_volume_profile_overlay_ = !show_volume_profile_overlay_;
    indicator_config_.show_volume_profile = show_volume_profile_overlay_;
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Volume Profile overlay");
  
  ImGui::SameLine();
  
  // Delta toggle
  ImVec4 delta_color = show_delta_overlay_ ? ImVec4(0.8f, 0.6f, 0.2f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, delta_color);
  if (ImGui::Button("Delta", ImVec2(60, 0))) {
    show_delta_overlay_ = !show_delta_overlay_;
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Delta overlay");
  
  ImGui::SameLine();
  
  // Cumulative Delta toggle
  ImVec4 cum_delta_color = show_cumulative_delta_overlay_ ? ImVec4(0.6f, 0.8f, 0.2f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, cum_delta_color);
  if (ImGui::Button("Cum Delta", ImVec2(80, 0))) {
    show_cumulative_delta_overlay_ = !show_cumulative_delta_overlay_;
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Cumulative Delta overlay");
  
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  
  // Liquidity bars toggle
  ImVec4 liq_color = show_liquidity_bars_ ? ImVec4(0.4f, 0.8f, 0.4f, 1.0f) : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
  ImGui::PushStyleColor(ImGuiCol_Button, liq_color);
  if (ImGui::Button("Liquidity", ImVec2(70, 0))) {
    show_liquidity_bars_ = !show_liquidity_bars_;
  }
  ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Liquidity Bars on price axis");
  
  ImGui::SameLine();
  ImGui::Spacing();
  ImGui::SameLine();
  
  // Current volume info (if chart data available)
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    const ChartInstance& chart = it->second;
    if (!chart.volumes.empty()) {
      float last_volume = chart.volumes.back();
      ImGui::Text("Vol: %.0f", last_volume);
    }
  }
  
  ImGui::EndChild();
  ImGui::PopStyleVar();
}


}  // namespace BTQuant
