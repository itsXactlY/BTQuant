#include "../include/layout/layout_presets.hpp"

#include <algorithm>  // for std::find_if
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {
namespace Layout {

LayoutPresetManager::LayoutPresetManager() {
  initialize_presets_directory();
  load_presets();
}

LayoutPresetManager::~LayoutPresetManager() {
  // Save any modified presets if needed
}

void LayoutPresetManager::initialize_presets_directory() {
  // Create presets directory if it doesn't exist
  presets_directory_ = "presets";
  std::filesystem::create_directories(presets_directory_);
}

void LayoutPresetManager::load_presets() {
  presets_.clear();
  load_builtin_presets();
  load_user_presets();
}

#ifdef HAS_NLOHMANN_JSON
void LayoutPresetManager::load_builtin_presets() {
  // Trading-focused layouts
  LayoutPreset trading_pro;
  trading_pro.name = "Trading Pro";
  trading_pro.description = "Professional trading layout with order book, charts, and positions";
  trading_pro.category = "Trading";
  trading_pro.is_builtin = true;
  trading_pro.author = "BTQuant Team";
  trading_pro.version = "1.0";

  // Sample JSON data for trading pro layout
  nlohmann::json trading_json;
  trading_json["layout_name"] = "Trading Pro";
  trading_json["theme"] = "Dark Professional";
  trading_json["grid_columns"] = 6;
  trading_json["grid_rows"] = 9;

  nlohmann::json panels = nlohmann::json::array();

  // Add chart panel
  nlohmann::json chart_panel;
  chart_panel["panel_id"] = "chart_1";
  chart_panel["panel_name"] = "Price Chart";
  chart_panel["type"] = "CHART";
  chart_panel["x"] = 0.0f;
  chart_panel["y"] = 0.0f;
  chart_panel["width"] = 0.66f;
  chart_panel["height"] = 0.5f;
  chart_panel["symbol"] = "BTCUSD";
  panels.push_back(chart_panel);

  // Add order book panel
  nlohmann::json orderbook_panel;
  orderbook_panel["panel_id"] = "orderbook_1";
  orderbook_panel["panel_name"] = "Order Book";
  orderbook_panel["type"] = "ORDERBOOK";
  orderbook_panel["x"] = 0.66f;
  orderbook_panel["y"] = 0.0f;
  orderbook_panel["width"] = 0.34f;
  orderbook_panel["height"] = 0.5f;
  orderbook_panel["symbol"] = "BTCUSD";
  panels.push_back(orderbook_panel);

  // Add positions panel
  nlohmann::json positions_panel;
  positions_panel["panel_id"] = "positions_1";
  positions_panel["panel_name"] = "Positions";
  positions_panel["type"] = "TRADING_POSITIONS";
  positions_panel["x"] = 0.0f;
  positions_panel["y"] = 0.5f;
  positions_panel["width"] = 0.33f;
  positions_panel["height"] = 0.5f;
  positions_panel["symbol"] = "BTCUSD";
  panels.push_back(positions_panel);

  // Add orders panel
  nlohmann::json orders_panel;
  orders_panel["panel_id"] = "orders_1";
  orders_panel["panel_name"] = "Orders";
  orders_panel["type"] = "TRADING_ORDERS";
  orders_panel["x"] = 0.33f;
  orders_panel["y"] = 0.5f;
  orders_panel["width"] = 0.33f;
  orders_panel["height"] = 0.5f;
  orders_panel["symbol"] = "BTCUSD";
  panels.push_back(orders_panel);

  // Add watchlist panel
  nlohmann::json watchlist_panel;
  watchlist_panel["panel_id"] = "watchlist_1";
  watchlist_panel["panel_name"] = "Watchlist";
  watchlist_panel["type"] = "WATCHLIST";
  watchlist_panel["x"] = 0.66f;
  watchlist_panel["y"] = 0.5f;
  watchlist_panel["width"] = 0.34f;
  watchlist_panel["height"] = 0.5f;
  watchlist_panel["symbol"] = "";
  panels.push_back(watchlist_panel);

  trading_json["panels"] = panels;
  trading_pro.json_data = trading_json.dump(4);

  presets_.push_back(trading_pro);

  // Analysis-focused layout
  LayoutPreset analysis_pro;
  analysis_pro.name = "Analysis Pro";
  analysis_pro.description = "Advanced analysis layout with multiple charts and indicators";
  analysis_pro.category = "Analysis";
  analysis_pro.is_builtin = true;
  analysis_pro.author = "BTQuant Team";
  analysis_pro.version = "1.0";

  nlohmann::json analysis_json;
  analysis_json["layout_name"] = "Analysis Pro";
  analysis_json["theme"] = "Dark Professional";
  analysis_json["grid_columns"] = 8;
  analysis_json["grid_rows"] = 10;

  nlohmann::json analysis_panels = nlohmann::json::array();

  // Add main chart
  nlohmann::json main_chart;
  main_chart["panel_id"] = "main_chart_1";
  main_chart["panel_name"] = "Main Chart";
  main_chart["type"] = "CHART";
  main_chart["x"] = 0.0f;
  main_chart["y"] = 0.0f;
  main_chart["width"] = 0.5f;
  main_chart["height"] = 0.4f;
  main_chart["symbol"] = "BTCUSD";
  analysis_panels.push_back(main_chart);

  // Add secondary chart
  nlohmann::json secondary_chart;
  secondary_chart["panel_id"] = "secondary_chart_1";
  secondary_chart["panel_name"] = "Secondary Chart";
  secondary_chart["type"] = "CHART";
  secondary_chart["x"] = 0.5f;
  secondary_chart["y"] = 0.0f;
  secondary_chart["width"] = 0.5f;
  secondary_chart["height"] = 0.4f;
  secondary_chart["symbol"] = "ETHUSD";
  analysis_panels.push_back(secondary_chart);

  // Add volume profile
  nlohmann::json vp_panel;
  vp_panel["panel_id"] = "vp_1";
  vp_panel["panel_name"] = "Volume Profile";
  vp_panel["type"] = "VOLUME_PROFILE";
  vp_panel["x"] = 0.0f;
  vp_panel["y"] = 0.4f;
  vp_panel["width"] = 0.5f;
  vp_panel["height"] = 0.3f;
  vp_panel["symbol"] = "BTCUSD";
  analysis_panels.push_back(vp_panel);

  // Add footprint chart
  nlohmann::json fp_panel;
  fp_panel["panel_id"] = "fp_1";
  fp_panel["panel_name"] = "Footprint Chart";
  fp_panel["type"] = "FOOTPRINT_CHART";
  fp_panel["x"] = 0.5f;
  fp_panel["y"] = 0.4f;
  fp_panel["width"] = 0.5f;
  fp_panel["height"] = 0.3f;
  fp_panel["symbol"] = "BTCUSD";
  analysis_panels.push_back(fp_panel);

  // Add correlation heatmap
  nlohmann::json heatmap_panel;
  heatmap_panel["panel_id"] = "heatmap_1";
  heatmap_panel["panel_name"] = "Correlation Heatmap";
  heatmap_panel["type"] = "HEATMAP";
  heatmap_panel["x"] = 0.0f;
  heatmap_panel["y"] = 0.7f;
  heatmap_panel["width"] = 0.5f;
  heatmap_panel["height"] = 0.3f;
  heatmap_panel["symbol"] = "";
  analysis_panels.push_back(heatmap_panel);

  // Add TPO profile
  nlohmann::json tpo_panel;
  tpo_panel["panel_id"] = "tpo_1";
  tpo_panel["panel_name"] = "TPO Profile";
  tpo_panel["type"] = "TPO_PROFILE";
  tpo_panel["x"] = 0.5f;
  tpo_panel["y"] = 0.7f;
  tpo_panel["width"] = 0.5f;
  tpo_panel["height"] = 0.3f;
  tpo_panel["symbol"] = "BTCUSD";
  analysis_panels.push_back(tpo_panel);

  analysis_json["panels"] = analysis_panels;
  analysis_pro.json_data = analysis_json.dump(4);

  presets_.push_back(analysis_pro);

  // Monitoring-focused layout
  LayoutPreset monitoring_pro;
  monitoring_pro.name = "Monitoring Pro";
  monitoring_pro.description = "Comprehensive monitoring layout with multiple data streams";
  monitoring_pro.category = "Monitoring";
  monitoring_pro.is_builtin = true;
  monitoring_pro.author = "BTQuant Team";
  monitoring_pro.version = "1.0";

  nlohmann::json monitoring_json;
  monitoring_json["layout_name"] = "Monitoring Pro";
  monitoring_json["theme"] = "Dark Professional";
  monitoring_json["grid_columns"] = 12;
  monitoring_json["grid_rows"] = 12;

  nlohmann::json monitoring_panels = nlohmann::json::array();

  // Add performance monitor
  nlohmann::json perf_panel;
  perf_panel["panel_id"] = "perf_1";
  perf_panel["panel_name"] = "Performance Monitor";
  perf_panel["type"] = "PERFORMANCE_MONITOR";
  perf_panel["x"] = 0.0f;
  perf_panel["y"] = 0.0f;
  perf_panel["width"] = 0.33f;
  perf_panel["height"] = 0.25f;
  perf_panel["symbol"] = "";
  monitoring_panels.push_back(perf_panel);

  // Add risk metrics
  nlohmann::json risk_panel;
  risk_panel["panel_id"] = "risk_1";
  risk_panel["panel_name"] = "Risk Metrics";
  risk_panel["type"] = "RISK_METRICS";
  risk_panel["x"] = 0.33f;
  risk_panel["y"] = 0.0f;
  risk_panel["width"] = 0.33f;
  risk_panel["height"] = 0.25f;
  risk_panel["symbol"] = "";
  monitoring_panels.push_back(risk_panel);

  // Add alerts panel
  nlohmann::json alerts_panel;
  alerts_panel["panel_id"] = "alerts_1";
  alerts_panel["panel_name"] = "Alerts";
  alerts_panel["type"] = "ALERTS";
  alerts_panel["x"] = 0.66f;
  alerts_panel["y"] = 0.0f;
  alerts_panel["width"] = 0.34f;
  alerts_panel["height"] = 0.25f;
  alerts_panel["symbol"] = "";
  monitoring_panels.push_back(alerts_panel);

  // Add multiple charts
  for (int i = 0; i < 6; ++i) {
    nlohmann::json chart;
    chart["panel_id"] = "monitor_chart_" + std::to_string(i + 1);
    chart["panel_name"] = "Chart " + std::to_string(i + 1);
    chart["type"] = "CHART";
    chart["x"] = (i % 3) * 0.33f;
    chart["y"] = 0.25f + (i / 3) * 0.25f;
    chart["width"] = 0.33f;
    chart["height"] = 0.25f;
    chart["symbol"] = "SYMBOL" + std::to_string(i + 1);
    monitoring_panels.push_back(chart);
  }

  // Add watchlist
  nlohmann::json watchlist_monitor;
  watchlist_monitor["panel_id"] = "watchlist_monitor_1";
  watchlist_monitor["panel_name"] = "Watchlist";
  watchlist_monitor["type"] = "WATCHLIST";
  watchlist_monitor["x"] = 0.0f;
  watchlist_monitor["y"] = 0.75f;
  watchlist_monitor["width"] = 0.5f;
  watchlist_monitor["height"] = 0.25f;
  watchlist_monitor["symbol"] = "";
  monitoring_panels.push_back(watchlist_monitor);

  // Add log panel
  nlohmann::json log_panel;
  log_panel["panel_id"] = "log_1";
  log_panel["panel_name"] = "Log Panel";
  log_panel["type"] = "LOG_PANEL";
  log_panel["x"] = 0.5f;
  log_panel["y"] = 0.75f;
  log_panel["width"] = 0.5f;
  log_panel["height"] = 0.25f;
  log_panel["symbol"] = "";
  monitoring_panels.push_back(log_panel);

  monitoring_json["panels"] = monitoring_panels;
  monitoring_pro.json_data = monitoring_json.dump(4);

  presets_.push_back(monitoring_pro);

  // Scalper-focused layout
  LayoutPreset scalper_pro;
  scalper_pro.name = "Scalper Pro";
  scalper_pro.description = "Optimized layout for scalping with real-time data, order book, and quick execution tools";
  scalper_pro.category = "Trading Styles";
  scalper_pro.is_builtin = true;
  scalper_pro.author = "BTQuant Team";
  scalper_pro.version = "1.0";

  nlohmann::json scalper_json;
  scalper_json["layout_name"] = "Scalper Pro";
  scalper_json["theme"] = "Dark Professional";
  scalper_json["grid_columns"] = 8;
  scalper_json["grid_rows"] = 10;

  nlohmann::json scalper_panels = nlohmann::json::array();

  // Add main price chart
  nlohmann::json main_chart;
  main_chart["panel_id"] = "scalp_chart_1";
  main_chart["panel_name"] = "Main Price Chart";
  main_chart["type"] = static_cast<int>(PanelType::CHART);
  main_chart["x"] = 0.0f;
  main_chart["y"] = 0.0f;
  main_chart["width"] = 0.5f;
  main_chart["height"] = 0.4f;
  main_chart["symbol"] = "BTCUSD";
  main_chart["timeframe"] = "1m";
  scalper_panels.push_back(main_chart);

  // Add order book
  nlohmann::json orderbook_panel;
  orderbook_panel["panel_id"] = "scalp_orderbook_1";
  orderbook_panel["panel_name"] = "Order Book";
  orderbook_panel["type"] = static_cast<int>(PanelType::ORDERBOOK);
  orderbook_panel["x"] = 0.5f;
  orderbook_panel["y"] = 0.0f;
  orderbook_panel["width"] = 0.25f;
  orderbook_panel["height"] = 0.4f;
  orderbook_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(orderbook_panel);

  // Add depth chart
  nlohmann::json depth_panel;
  depth_panel["panel_id"] = "scalp_depth_1";
  depth_panel["panel_name"] = "Market Depth";
  depth_panel["type"] = static_cast<int>(PanelType::DEPTH_CHART);
  depth_panel["x"] = 0.75f;
  depth_panel["y"] = 0.0f;
  depth_panel["width"] = 0.25f;
  depth_panel["height"] = 0.4f;
  depth_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(depth_panel);

  // Add tape chart
  nlohmann::json tape_panel;
  tape_panel["panel_id"] = "scalp_tape_1";
  tape_panel["panel_name"] = "Time & Sales";
  tape_panel["type"] = static_cast<int>(PanelType::TAPE);
  tape_panel["x"] = 0.0f;
  tape_panel["y"] = 0.4f;
  tape_panel["width"] = 0.25f;
  tape_panel["height"] = 0.3f;
  tape_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(tape_panel);

  // Add trading positions
  nlohmann::json positions_panel;
  positions_panel["panel_id"] = "scalp_positions_1";
  positions_panel["panel_name"] = "Positions";
  positions_panel["type"] = static_cast<int>(PanelType::TRADING_POSITIONS);
  positions_panel["x"] = 0.25f;
  positions_panel["y"] = 0.4f;
  positions_panel["width"] = 0.25f;
  positions_panel["height"] = 0.3f;
  positions_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(positions_panel);

  // Add trading orders
  nlohmann::json orders_panel;
  orders_panel["panel_id"] = "scalp_orders_1";
  orders_panel["panel_name"] = "Orders";
  orders_panel["type"] = static_cast<int>(PanelType::TRADING_ORDERS);
  orders_panel["x"] = 0.5f;
  orders_panel["y"] = 0.4f;
  orders_panel["width"] = 0.25f;
  orders_panel["height"] = 0.3f;
  orders_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(orders_panel);

  // Add quick execution panel
  nlohmann::json execution_panel;
  execution_panel["panel_id"] = "scalp_execution_1";
  execution_panel["panel_name"] = "Quick Execution";
  execution_panel["type"] = static_cast<int>(PanelType::CUSTOM);
  execution_panel["x"] = 0.75f;
  execution_panel["y"] = 0.4f;
  execution_panel["width"] = 0.25f;
  execution_panel["height"] = 0.3f;
  execution_panel["symbol"] = "BTCUSD";
  scalper_panels.push_back(execution_panel);

  // Add watchlist
  nlohmann::json watchlist_panel;
  watchlist_panel["panel_id"] = "scalp_watchlist_1";
  watchlist_panel["panel_name"] = "Watchlist";
  watchlist_panel["type"] = static_cast<int>(PanelType::WATCHLIST);
  watchlist_panel["x"] = 0.0f;
  watchlist_panel["y"] = 0.7f;
  watchlist_panel["width"] = 0.5f;
  watchlist_panel["height"] = 0.3f;
  watchlist_panel["symbol"] = "";
  scalper_panels.push_back(watchlist_panel);

  // Add alerts panel
  nlohmann::json alerts_panel;
  alerts_panel["panel_id"] = "scalp_alerts_1";
  alerts_panel["panel_name"] = "Alerts";
  alerts_panel["type"] = static_cast<int>(PanelType::ALERTS);
  alerts_panel["x"] = 0.5f;
  alerts_panel["y"] = 0.7f;
  alerts_panel["width"] = 0.5f;
  alerts_panel["height"] = 0.3f;
  alerts_panel["symbol"] = "";
  scalper_panels.push_back(alerts_panel);

  scalper_json["panels"] = scalper_panels;
  scalper_pro.json_data = scalper_json.dump(4);

  presets_.push_back(scalper_pro);

  // Day Trader-focused layout
  LayoutPreset day_trader_pro;
  day_trader_pro.name = "Day Trader Pro";
  day_trader_pro.description = "Optimized layout for day trading with multiple timeframes and technical analysis tools";
  day_trader_pro.category = "Trading Styles";
  day_trader_pro.is_builtin = true;
  day_trader_pro.author = "BTQuant Team";
  day_trader_pro.version = "1.0";

  nlohmann::json day_trader_json;
  day_trader_json["layout_name"] = "Day Trader Pro";
  day_trader_json["theme"] = "Dark Professional";
  day_trader_json["grid_columns"] = 10;
  day_trader_json["grid_rows"] = 12;

  nlohmann::json day_trader_panels = nlohmann::json::array();

  // Add main chart
  nlohmann::json dt_main_chart;
  dt_main_chart["panel_id"] = "dt_main_chart_1";
  dt_main_chart["panel_name"] = "Main Chart";
  dt_main_chart["type"] = static_cast<int>(PanelType::CHART);
  dt_main_chart["x"] = 0.0f;
  dt_main_chart["y"] = 0.0f;
  dt_main_chart["width"] = 0.4f;
  dt_main_chart["height"] = 0.5f;
  dt_main_chart["symbol"] = "BTCUSD";
  dt_main_chart["timeframe"] = "5m";
  day_trader_panels.push_back(dt_main_chart);

  // Add secondary chart
  nlohmann::json dt_secondary_chart;
  dt_secondary_chart["panel_id"] = "dt_secondary_chart_1";
  dt_secondary_chart["panel_name"] = "Secondary Chart";
  dt_secondary_chart["type"] = static_cast<int>(PanelType::CHART);
  dt_secondary_chart["x"] = 0.4f;
  dt_secondary_chart["y"] = 0.0f;
  dt_secondary_chart["width"] = 0.3f;
  dt_secondary_chart["height"] = 0.3f;
  dt_secondary_chart["symbol"] = "ETHUSD";
  dt_secondary_chart["timeframe"] = "15m";
  day_trader_panels.push_back(dt_secondary_chart);

  // Add order book
  nlohmann::json dt_orderbook_panel;
  dt_orderbook_panel["panel_id"] = "dt_orderbook_1";
  dt_orderbook_panel["panel_name"] = "Order Book";
  dt_orderbook_panel["type"] = static_cast<int>(PanelType::ORDERBOOK);
  dt_orderbook_panel["x"] = 0.7f;
  dt_orderbook_panel["y"] = 0.0f;
  dt_orderbook_panel["width"] = 0.3f;
  dt_orderbook_panel["height"] = 0.3f;
  dt_orderbook_panel["symbol"] = "BTCUSD";
  day_trader_panels.push_back(dt_orderbook_panel);

  // Add volume profile
  nlohmann::json dt_vp_panel;
  dt_vp_panel["panel_id"] = "dt_vp_1";
  dt_vp_panel["panel_name"] = "Volume Profile";
  dt_vp_panel["type"] = static_cast<int>(PanelType::VOLUME_PROFILE);
  dt_vp_panel["x"] = 0.4f;
  dt_vp_panel["y"] = 0.3f;
  dt_vp_panel["width"] = 0.3f;
  dt_vp_panel["height"] = 0.2f;
  dt_vp_panel["symbol"] = "BTCUSD";
  day_trader_panels.push_back(dt_vp_panel);

  // Add footprint chart
  nlohmann::json dt_fp_panel;
  dt_fp_panel["panel_id"] = "dt_fp_1";
  dt_fp_panel["panel_name"] = "Footprint Chart";
  dt_fp_panel["type"] = static_cast<int>(PanelType::FOOTPRINT_CHART);
  dt_fp_panel["x"] = 0.7f;
  dt_fp_panel["y"] = 0.3f;
  dt_fp_panel["width"] = 0.3f;
  dt_fp_panel["height"] = 0.2f;
  dt_fp_panel["symbol"] = "BTCUSD";
  day_trader_panels.push_back(dt_fp_panel);

  // Add trading positions
  nlohmann::json dt_positions_panel;
  dt_positions_panel["panel_id"] = "dt_positions_1";
  dt_positions_panel["panel_name"] = "Positions";
  dt_positions_panel["type"] = static_cast<int>(PanelType::TRADING_POSITIONS);
  dt_positions_panel["x"] = 0.0f;
  dt_positions_panel["y"] = 0.5f;
  dt_positions_panel["width"] = 0.2f;
  dt_positions_panel["height"] = 0.5f;
  dt_positions_panel["symbol"] = "BTCUSD";
  day_trader_panels.push_back(dt_positions_panel);

  // Add trading orders
  nlohmann::json dt_orders_panel;
  dt_orders_panel["panel_id"] = "dt_orders_1";
  dt_orders_panel["panel_name"] = "Orders";
  dt_orders_panel["type"] = static_cast<int>(PanelType::TRADING_ORDERS);
  dt_orders_panel["x"] = 0.2f;
  dt_orders_panel["y"] = 0.5f;
  dt_orders_panel["width"] = 0.2f;
  dt_orders_panel["height"] = 0.5f;
  dt_orders_panel["symbol"] = "BTCUSD";
  day_trader_panels.push_back(dt_orders_panel);

  // Add watchlist
  nlohmann::json dt_watchlist_panel;
  dt_watchlist_panel["panel_id"] = "dt_watchlist_1";
  dt_watchlist_panel["panel_name"] = "Watchlist";
  dt_watchlist_panel["type"] = static_cast<int>(PanelType::WATCHLIST);
  dt_watchlist_panel["x"] = 0.4f;
  dt_watchlist_panel["y"] = 0.5f;
  dt_watchlist_panel["width"] = 0.3f;
  dt_watchlist_panel["height"] = 0.5f;
  dt_watchlist_panel["symbol"] = "";
  day_trader_panels.push_back(dt_watchlist_panel);

  // Add risk metrics
  nlohmann::json dt_risk_panel;
  dt_risk_panel["panel_id"] = "dt_risk_1";
  dt_risk_panel["panel_name"] = "Risk Metrics";
  dt_risk_panel["type"] = static_cast<int>(PanelType::RISK_METRICS);
  dt_risk_panel["x"] = 0.7f;
  dt_risk_panel["y"] = 0.5f;
  dt_risk_panel["width"] = 0.3f;
  dt_risk_panel["height"] = 0.5f;
  dt_risk_panel["symbol"] = "";
  day_trader_panels.push_back(dt_risk_panel);

  day_trader_json["panels"] = day_trader_panels;
  day_trader_pro.json_data = day_trader_json.dump(4);

  presets_.push_back(day_trader_pro);

  // Swing Trader-focused layout
  LayoutPreset swing_trader_pro;
  swing_trader_pro.name = "Swing Trader Pro";
  swing_trader_pro.description = "Optimized layout for swing trading with multi-day charts and fundamental analysis tools";
  swing_trader_pro.category = "Trading Styles";
  swing_trader_pro.is_builtin = true;
  swing_trader_pro.author = "BTQuant Team";
  swing_trader_pro.version = "1.0";

  nlohmann::json swing_trader_json;
  swing_trader_json["layout_name"] = "Swing Trader Pro";
  swing_trader_json["theme"] = "Dark Professional";
  swing_trader_json["grid_columns"] = 8;
  swing_trader_json["grid_rows"] = 10;

  nlohmann::json swing_trader_panels = nlohmann::json::array();

  // Add main chart with daily timeframe
  nlohmann::json st_main_chart;
  st_main_chart["panel_id"] = "st_main_chart_1";
  st_main_chart["panel_name"] = "Daily Chart";
  st_main_chart["type"] = static_cast<int>(PanelType::CHART);
  st_main_chart["x"] = 0.0f;
  st_main_chart["y"] = 0.0f;
  st_main_chart["width"] = 0.5f;
  st_main_chart["height"] = 0.4f;
  st_main_chart["symbol"] = "SPY";
  st_main_chart["timeframe"] = "1d";
  swing_trader_panels.push_back(st_main_chart);

  // Add weekly chart
  nlohmann::json st_weekly_chart;
  st_weekly_chart["panel_id"] = "st_weekly_chart_1";
  st_weekly_chart["panel_name"] = "Weekly Chart";
  st_weekly_chart["type"] = static_cast<int>(PanelType::CHART);
  st_weekly_chart["x"] = 0.5f;
  st_weekly_chart["y"] = 0.0f;
  st_weekly_chart["width"] = 0.5f;
  st_weekly_chart["height"] = 0.2f;
  st_weekly_chart["symbol"] = "SPY";
  st_weekly_chart["timeframe"] = "1w";
  swing_trader_panels.push_back(st_weekly_chart);

  // Add monthly chart
  nlohmann::json st_monthly_chart;
  st_monthly_chart["panel_id"] = "st_monthly_chart_1";
  st_monthly_chart["panel_name"] = "Monthly Chart";
  st_monthly_chart["type"] = static_cast<int>(PanelType::CHART);
  st_monthly_chart["x"] = 0.5f;
  st_monthly_chart["y"] = 0.2f;
  st_monthly_chart["width"] = 0.5f;
  st_monthly_chart["height"] = 0.2f;
  st_monthly_chart["symbol"] = "SPY";
  st_monthly_chart["timeframe"] = "1mo";
  swing_trader_panels.push_back(st_monthly_chart);

  // Add heatmap for sector analysis
  nlohmann::json st_heatmap_panel;
  st_heatmap_panel["panel_id"] = "st_heatmap_1";
  st_heatmap_panel["panel_name"] = "Sector Heatmap";
  st_heatmap_panel["type"] = static_cast<int>(PanelType::HEATMAP);
  st_heatmap_panel["x"] = 0.0f;
  st_heatmap_panel["y"] = 0.4f;
  st_heatmap_panel["width"] = 0.5f;
  st_heatmap_panel["height"] = 0.3f;
  st_heatmap_panel["symbol"] = "";
  swing_trader_panels.push_back(st_heatmap_panel);

  // Add screener
  nlohmann::json st_screener_panel;
  st_screener_panel["panel_id"] = "st_screener_1";
  st_screener_panel["panel_name"] = "Stock Screener";
  st_screener_panel["type"] = static_cast<int>(PanelType::SCREENER);
  st_screener_panel["x"] = 0.5f;
  st_screener_panel["y"] = 0.4f;
  st_screener_panel["width"] = 0.5f;
  st_screener_panel["height"] = 0.3f;
  st_screener_panel["symbol"] = "";
  swing_trader_panels.push_back(st_screener_panel);

  // Add positions
  nlohmann::json st_positions_panel;
  st_positions_panel["panel_id"] = "st_positions_1";
  st_positions_panel["panel_name"] = "Positions";
  st_positions_panel["type"] = static_cast<int>(PanelType::TRADING_POSITIONS);
  st_positions_panel["x"] = 0.0f;
  st_positions_panel["y"] = 0.7f;
  st_positions_panel["width"] = 0.33f;
  st_positions_panel["height"] = 0.3f;
  st_positions_panel["symbol"] = "SPY";
  swing_trader_panels.push_back(st_positions_panel);

  // Add performance monitor
  nlohmann::json st_perf_panel;
  st_perf_panel["panel_id"] = "st_perf_1";
  st_perf_panel["panel_name"] = "Performance";
  st_perf_panel["type"] = static_cast<int>(PanelType::PERFORMANCE_MONITOR);
  st_perf_panel["x"] = 0.33f;
  st_perf_panel["y"] = 0.7f;
  st_perf_panel["width"] = 0.33f;
  st_perf_panel["height"] = 0.3f;
  st_perf_panel["symbol"] = "";
  swing_trader_panels.push_back(st_perf_panel);

  // Add watchlist
  nlohmann::json st_watchlist_panel;
  st_watchlist_panel["panel_id"] = "st_watchlist_1";
  st_watchlist_panel["panel_name"] = "Watchlist";
  st_watchlist_panel["type"] = static_cast<int>(PanelType::WATCHLIST);
  st_watchlist_panel["x"] = 0.66f;
  st_watchlist_panel["y"] = 0.7f;
  st_watchlist_panel["width"] = 0.34f;
  st_watchlist_panel["height"] = 0.3f;
  st_watchlist_panel["symbol"] = "";
  swing_trader_panels.push_back(st_watchlist_panel);

  swing_trader_json["panels"] = swing_trader_panels;
  swing_trader_pro.json_data = swing_trader_json.dump(4);

  presets_.push_back(swing_trader_pro);

  // Analysis-focused layout
  LayoutPreset analysis_template;
  analysis_template.name = "Analysis Template";
  analysis_template.description = "Comprehensive analysis layout with multiple charts, indicators, and statistical tools";
  analysis_template.category = "Analysis";
  analysis_template.is_builtin = true;
  analysis_template.author = "BTQuant Team";
  analysis_template.version = "1.0";

  nlohmann::json analysis_template_json;
  analysis_template_json["layout_name"] = "Analysis Template";
  analysis_template_json["theme"] = "Dark Professional";
  analysis_template_json["grid_columns"] = 12;
  analysis_template_json["grid_rows"] = 12;

  nlohmann::json analysis_template_panels = nlohmann::json::array();

  // Add main analysis chart
  nlohmann::json at_main_chart;
  at_main_chart["panel_id"] = "at_main_chart_1";
  at_main_chart["panel_name"] = "Primary Analysis Chart";
  at_main_chart["type"] = static_cast<int>(PanelType::CHART);
  at_main_chart["x"] = 0.0f;
  at_main_chart["y"] = 0.0f;
  at_main_chart["width"] = 0.5f;
  at_main_chart["height"] = 0.4f;
  at_main_chart["symbol"] = "BTCUSD";
  at_main_chart["timeframe"] = "1h";
  analysis_template_panels.push_back(at_main_chart);

  // Add secondary analysis chart
  nlohmann::json at_secondary_chart;
  at_secondary_chart["panel_id"] = "at_secondary_chart_1";
  at_secondary_chart["panel_name"] = "Secondary Analysis Chart";
  at_secondary_chart["type"] = static_cast<int>(PanelType::CHART);
  at_secondary_chart["x"] = 0.5f;
  at_secondary_chart["y"] = 0.0f;
  at_secondary_chart["width"] = 0.3f;
  at_secondary_chart["height"] = 0.2f;
  at_secondary_chart["symbol"] = "ETHUSD";
  at_secondary_chart["timeframe"] = "1h";
  analysis_template_panels.push_back(at_secondary_chart);

  // Add correlation matrix
  nlohmann::json at_corr_panel;
  at_corr_panel["panel_id"] = "at_corr_1";
  at_corr_panel["panel_name"] = "Correlation Matrix";
  at_corr_panel["type"] = static_cast<int>(PanelType::HEATMAP);
  at_corr_panel["x"] = 0.8f;
  at_corr_panel["y"] = 0.0f;
  at_corr_panel["width"] = 0.2f;
  at_corr_panel["height"] = 0.2f;
  at_corr_panel["symbol"] = "";
  analysis_template_panels.push_back(at_corr_panel);

  // Add volume profile
  nlohmann::json at_vp_panel;
  at_vp_panel["panel_id"] = "at_vp_1";
  at_vp_panel["panel_name"] = "Volume Profile";
  at_vp_panel["type"] = static_cast<int>(PanelType::VOLUME_PROFILE);
  at_vp_panel["x"] = 0.5f;
  at_vp_panel["y"] = 0.2f;
  at_vp_panel["width"] = 0.3f;
  at_vp_panel["height"] = 0.2f;
  at_vp_panel["symbol"] = "BTCUSD";
  analysis_template_panels.push_back(at_vp_panel);

  // Add footprint chart
  nlohmann::json at_fp_panel;
  at_fp_panel["panel_id"] = "at_fp_1";
  at_fp_panel["panel_name"] = "Footprint Chart";
  at_fp_panel["type"] = static_cast<int>(PanelType::FOOTPRINT_CHART);
  at_fp_panel["x"] = 0.8f;
  at_fp_panel["y"] = 0.2f;
  at_fp_panel["width"] = 0.2f;
  at_fp_panel["height"] = 0.2f;
  at_fp_panel["symbol"] = "BTCUSD";
  analysis_template_panels.push_back(at_fp_panel);

  // Add histogram
  nlohmann::json at_hist_panel;
  at_hist_panel["panel_id"] = "at_hist_1";
  at_hist_panel["panel_name"] = "Histogram";
  at_hist_panel["type"] = static_cast<int>(PanelType::HISTOGRAM);
  at_hist_panel["x"] = 0.0f;
  at_hist_panel["y"] = 0.4f;
  at_hist_panel["width"] = 0.25f;
  at_hist_panel["height"] = 0.2f;
  at_hist_panel["symbol"] = "";
  analysis_template_panels.push_back(at_hist_panel);

  // Add scatter plot
  nlohmann::json at_scatter_panel;
  at_scatter_panel["panel_id"] = "at_scatter_1";
  at_scatter_panel["panel_name"] = "Scatter Plot";
  at_scatter_panel["type"] = static_cast<int>(PanelType::SCATTER_PLOT);
  at_scatter_panel["x"] = 0.25f;
  at_scatter_panel["y"] = 0.4f;
  at_scatter_panel["width"] = 0.25f;
  at_scatter_panel["height"] = 0.2f;
  at_scatter_panel["symbol"] = "";
  analysis_template_panels.push_back(at_scatter_panel);

  // Add time series analysis
  nlohmann::json at_ts_panel;
  at_ts_panel["panel_id"] = "at_ts_1";
  at_ts_panel["panel_name"] = "Time Series";
  at_ts_panel["type"] = static_cast<int>(PanelType::TIME_SERIES);
  at_ts_panel["x"] = 0.5f;
  at_ts_panel["y"] = 0.4f;
  at_ts_panel["width"] = 0.25f;
  at_ts_panel["height"] = 0.2f;
  at_ts_panel["symbol"] = "";
  analysis_template_panels.push_back(at_ts_panel);

  // Add TPO profile
  nlohmann::json at_tpo_panel;
  at_tpo_panel["panel_id"] = "at_tpo_1";
  at_tpo_panel["panel_name"] = "TPO Profile";
  at_tpo_panel["type"] = static_cast<int>(PanelType::TPO_PROFILE);
  at_tpo_panel["x"] = 0.75f;
  at_tpo_panel["y"] = 0.4f;
  at_tpo_panel["width"] = 0.25f;
  at_tpo_panel["height"] = 0.2f;
  at_tpo_panel["symbol"] = "BTCUSD";
  analysis_template_panels.push_back(at_tpo_panel);

  // Add multiple smaller charts for comparison
  for (int i = 0; i < 6; ++i) {
    nlohmann::json mini_chart;
    mini_chart["panel_id"] = "at_mini_chart_" + std::to_string(i + 1);
    mini_chart["panel_name"] = "Mini Chart " + std::to_string(i + 1);
    mini_chart["type"] = static_cast<int>(PanelType::CHART);
    mini_chart["x"] = (i % 3) * 0.33f;
    mini_chart["y"] = 0.6f + (i / 3) * 0.2f;
    mini_chart["width"] = 0.33f;
    mini_chart["height"] = 0.2f;
    mini_chart["symbol"] = "SYMBOL" + std::to_string(i + 1);
    mini_chart["timeframe"] = "4h";
    analysis_template_panels.push_back(mini_chart);
  }

  // Add watchlist
  nlohmann::json at_watchlist_panel;
  at_watchlist_panel["panel_id"] = "at_watchlist_1";
  at_watchlist_panel["panel_name"] = "Watchlist";
  at_watchlist_panel["type"] = static_cast<int>(PanelType::WATCHLIST);
  at_watchlist_panel["x"] = 0.0f;
  at_watchlist_panel["y"] = 0.8f;
  at_watchlist_panel["width"] = 0.5f;
  at_watchlist_panel["height"] = 0.2f;
  at_watchlist_panel["symbol"] = "";
  analysis_template_panels.push_back(at_watchlist_panel);

  // Add log panel
  nlohmann::json at_log_panel;
  at_log_panel["panel_id"] = "at_log_1";
  at_log_panel["panel_name"] = "Analysis Log";
  at_log_panel["type"] = static_cast<int>(PanelType::LOG_PANEL);
  at_log_panel["x"] = 0.5f;
  at_log_panel["y"] = 0.8f;
  at_log_panel["width"] = 0.5f;
  at_log_panel["height"] = 0.2f;
  at_log_panel["symbol"] = "";
  analysis_template_panels.push_back(at_log_panel);

  analysis_template_json["panels"] = analysis_template_panels;
  analysis_template.json_data = analysis_template_json.dump(4);

  presets_.push_back(analysis_template);
}
#else
void LayoutPresetManager::load_builtin_presets() {
  // Fallback implementation when JSON is not available
  // Just add some basic presets with empty JSON data
  LayoutPreset trading_pro;
  trading_pro.name = "Trading Pro";
  trading_pro.description = "Professional trading layout with order book, charts, and positions";
  trading_pro.category = "Trading";
  trading_pro.is_builtin = true;
  trading_pro.author = "BTQuant Team";
  trading_pro.version = "1.0";
  trading_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(trading_pro);

  LayoutPreset analysis_pro;
  analysis_pro.name = "Analysis Pro";
  analysis_pro.description = "Advanced analysis layout with multiple charts and indicators";
  analysis_pro.category = "Analysis";
  analysis_pro.is_builtin = true;
  analysis_pro.author = "BTQuant Team";
  analysis_pro.version = "1.0";
  analysis_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(analysis_pro);

  LayoutPreset monitoring_pro;
  monitoring_pro.name = "Monitoring Pro";
  monitoring_pro.description = "Comprehensive monitoring layout with multiple data streams";
  monitoring_pro.category = "Monitoring";
  monitoring_pro.is_builtin = true;
  monitoring_pro.author = "BTQuant Team";
  monitoring_pro.version = "1.0";
  monitoring_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(monitoring_pro);

  // Scalper Pro template
  LayoutPreset scalper_pro;
  scalper_pro.name = "Scalper Pro";
  scalper_pro.description = "Optimized layout for scalping with real-time data, order book, and quick execution tools";
  scalper_pro.category = "Trading Styles";
  scalper_pro.is_builtin = true;
  scalper_pro.author = "BTQuant Team";
  scalper_pro.version = "1.0";
  scalper_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(scalper_pro);

  // Day Trader Pro template
  LayoutPreset day_trader_pro;
  day_trader_pro.name = "Day Trader Pro";
  day_trader_pro.description = "Optimized layout for day trading with multiple timeframes and technical analysis tools";
  day_trader_pro.category = "Trading Styles";
  day_trader_pro.is_builtin = true;
  day_trader_pro.author = "BTQuant Team";
  day_trader_pro.version = "1.0";
  day_trader_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(day_trader_pro);

  // Swing Trader Pro template
  LayoutPreset swing_trader_pro;
  swing_trader_pro.name = "Swing Trader Pro";
  swing_trader_pro.description = "Optimized layout for swing trading with multi-day charts and fundamental analysis tools";
  swing_trader_pro.category = "Trading Styles";
  swing_trader_pro.is_builtin = true;
  swing_trader_pro.author = "BTQuant Team";
  swing_trader_pro.version = "1.0";
  swing_trader_pro.json_data = "{}";  // Empty JSON

  presets_.push_back(swing_trader_pro);

  // Analysis Template
  LayoutPreset analysis_template;
  analysis_template.name = "Analysis Template";
  analysis_template.description = "Comprehensive analysis layout with multiple charts, indicators, and statistical tools";
  analysis_template.category = "Analysis";
  analysis_template.is_builtin = true;
  analysis_template.author = "BTQuant Team";
  analysis_template.version = "1.0";
  analysis_template.json_data = "{}";  // Empty JSON

  presets_.push_back(analysis_template);
}
#endif

void LayoutPresetManager::load_user_presets() {
  if (!std::filesystem::exists(presets_directory_)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(presets_directory_)) {
    if (entry.path().extension() == ".json") {
      std::string preset_name = entry.path().stem().string();

      // Check if this preset is already loaded as builtin
      bool already_exists = false;
      for (const auto& preset : presets_) {
        if (preset.name == preset_name) {
          already_exists = true;
          break;
        }
      }

      if (!already_exists) {
        auto preset = load_preset_from_file(entry.path().string());
        if (!preset.name.empty()) {
          presets_.push_back(preset);
        }
      }
    }
  }
}

bool LayoutPresetManager::save_preset(const std::string& name, const std::string& description,
                                      const std::string& json_data, const std::string& category) {
  LayoutPreset preset;
  preset.name = name;
  preset.description = description;
  preset.json_data = json_data;
  preset.category = category;
  preset.is_builtin = false;
  preset.author = "User";
  preset.version = "1.0";

  if (save_preset_to_file(preset)) {
    // Check if preset already exists and update it, otherwise add new
    bool updated = false;
    for (auto& existing_preset : presets_) {
      if (existing_preset.name == name) {
        existing_preset = preset;
        updated = true;
        break;
      }
    }

    if (!updated) {
      presets_.push_back(preset);
    }

    return true;
  }

  return false;
}

bool LayoutPresetManager::apply_preset(const std::string& preset_name) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      // Here we would typically apply the preset to the layout manager
      // For now, just return true to indicate success
      std::cout << "Applied preset: " << preset_name << std::endl;
      return true;
    }
  }

  std::cerr << "Preset not found: " << preset_name << std::endl;
  return false;
}

bool LayoutPresetManager::delete_preset(const std::string& preset_name) {
  auto it = std::find_if(
      presets_.begin(), presets_.end(),
      [&preset_name](const LayoutPreset& preset) { return preset.name == preset_name; });

  if (it != presets_.end() && !it->is_builtin) {
    // Delete the file
    std::string file_path = get_preset_file_path(preset_name);
    std::filesystem::remove(file_path);

    presets_.erase(it);
    return true;
  }

  return false;
}

std::vector<LayoutPreset> LayoutPresetManager::get_all_presets() const { return presets_; }

std::vector<LayoutPreset> LayoutPresetManager::get_presets_by_category(
    const std::string& category) const {
  std::vector<LayoutPreset> result;
  for (const auto& preset : presets_) {
    if (preset.category == category) {
      result.push_back(preset);
    }
  }
  return result;
}

bool LayoutPresetManager::export_preset(const std::string& preset_name,
                                        const std::string& file_path) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      std::ofstream file(file_path);
      if (file.is_open()) {
        file << preset.json_data;
        file.close();
        return true;
      }
    }
  }
  return false;
}

bool LayoutPresetManager::import_preset(const std::string& file_path) {
#ifdef HAS_NLOHMANN_JSON
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return false;
  }

  try {
    nlohmann::json j;
    file >> j;

    if (!j.contains("layout_name")) {
      file.close();
      return false;
    }

    LayoutPreset preset;
    preset.name = j["layout_name"];
    preset.description = "Imported preset";
    preset.category = "Imported";
    preset.is_builtin = false;
    preset.author = "User";
    preset.version = "1.0";
    preset.json_data = j.dump(4);

    file.close();

    // Save the imported preset
    std::string new_file_path = get_preset_file_path(preset.name);
    std::ofstream out_file(new_file_path);
    if (out_file.is_open()) {
      out_file << preset.json_data;
      out_file.close();

      presets_.push_back(preset);
      return true;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error importing preset: " << e.what() << std::endl;
    file.close();
  }

  return false;
#else
  // Return false when JSON support is not available
  return false;
#endif
}

bool LayoutPresetManager::create_thumbnail(const std::string& preset_name,
                                           const std::string& thumbnail_path) {
  // This would typically capture a screenshot of the layout
  // For now, just return true
  return true;
}

bool LayoutPresetManager::validate_preset(const LayoutPreset& preset) const {
#ifdef HAS_NLOHMANN_JSON
  try {
    auto json = nlohmann::json::parse(preset.json_data);
    return json.contains("layout_name") && json.contains("panels");
  } catch (...) {
    return false;
  }
#else
  // When JSON support is not available, just check if the JSON data is not empty
  return !preset.json_data.empty();
#endif
}

#ifdef HAS_NLOHMANN_JSON
LayoutPreset LayoutPresetManager::load_preset_from_file(const std::string& file_path) {
  LayoutPreset preset;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return preset;
  }

  try {
    nlohmann::json j;
    file >> j;

    preset.name = j.value("layout_name", "");
    preset.description = j.value("description", "User-defined layout");
    preset.category = j.value("category", "Custom");
    preset.is_builtin = false;
    preset.author = j.value("author", "User");
    preset.version = j.value("version", "1.0");
    preset.json_data = j.dump(4);
  } catch (const std::exception& e) {
    std::cerr << "Error loading preset " << file_path << ": " << e.what() << std::endl;
  }

  file.close();
  return preset;
}

bool LayoutPresetManager::save_preset_to_file(const LayoutPreset& preset) {
  std::string file_path = get_preset_file_path(preset.name);

  std::ofstream file(file_path);
  if (file.is_open()) {
    file << preset.json_data;
    file.close();
    return true;
  }

  return false;
}
#else
// Fallback implementations when nlohmann/json is not available
LayoutPreset LayoutPresetManager::load_preset_from_file(const std::string& file_path) {
  LayoutPreset preset;
  // Return a default preset when JSON parsing is not available
  preset.name = "Default";
  preset.description = "Default preset when JSON support is not available";
  preset.category = "Default";
  preset.is_builtin = true;
  preset.author = "System";
  preset.version = "1.0";
  preset.json_data = "{}";
  return preset;
}

bool LayoutPresetManager::save_preset_to_file(const LayoutPreset& preset) {
  // Return false when JSON serialization is not available
  return false;
}
#endif

std::string LayoutPresetManager::get_preset_file_path(const std::string& preset_name) const {
  return presets_directory_ + "/" + preset_name + ".json";
}

}  // namespace Layout
}  // namespace BTQuant