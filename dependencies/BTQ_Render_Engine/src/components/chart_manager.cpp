#include "../../include/components/chart_manager.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../include/data/data_types.hpp"
#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include "imgui.h"

namespace BTQuant {

// DEPRECATED - Legacy hotspine
ChartManager::ChartManager(std::shared_ptr<HotSpineDataBridge> bridge,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : bridge_(bridge), processor_(processor), next_chart_id_(0) {}

uint32_t ChartManager::create_chart(const std::string& symbol_name,
                                    const std::string& exchange_name, uint32_t symbol_id,
                                    RenderEngine::TimeFrame timeframe) {
  // Check if a chart already exists for this symbol and timeframe
  for (const auto& [id, chart] : charts_) {
    if (chart.symbol_name == symbol_name && chart.timeframe == timeframe) {
      return id;
    }
  }

  ChartInstance chart;
  chart.symbol_name = symbol_name;
  chart.exchange_name = exchange_name;
  chart.symbol_id = symbol_id;
  chart.timeframe = timeframe;
  chart.chart_id = next_chart_id_++;
  chart.visible = true;
  chart.minimized = false;

  int chart_count = static_cast<int>(charts_.size());
  int grid_cols = 2;
  float x_offset = static_cast<float>(chart_count % grid_cols) * 650.0f + 10.0f;
  float y_offset = static_cast<float>(chart_count / grid_cols) * 450.0f + 10.0f;
  chart.position = {x_offset, y_offset};

  charts_[chart.chart_id] = chart;
  return chart.chart_id;
}

void ChartManager::destroy_chart(uint32_t chart_id) { charts_.erase(chart_id); }

void ChartManager::toggle_chart_visibility(uint32_t chart_id) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.visible = !it->second.visible;
  }
}

void ChartManager::toggle_chart_minimization(uint32_t chart_id) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.minimized = !it->second.minimized;
  }
}

void ChartManager::update_chart_position(uint32_t chart_id, const ImVec2& position) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.position = position;
  }
}

void ChartManager::update_chart_size(uint32_t chart_id, const ImVec2& size) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.size = size;
  }
}

const std::unordered_map<uint32_t, ChartInstance>& ChartManager::get_charts() const {
  return charts_;
}

std::vector<ChartInstance> ChartManager::get_visible_charts() const {
  std::vector<ChartInstance> visible_charts;
  visible_charts.reserve(charts_.size());
  for (const auto& [id, chart] : charts_) {
    if (chart.visible) {
      visible_charts.push_back(chart);
    }
  }
  return visible_charts;
}

std::vector<ChartInstance> ChartManager::get_charts_for_symbol(
    const std::string& symbol_name) const {
  std::vector<ChartInstance> symbol_charts;
  for (const auto& [id, chart] : charts_) {
    if (chart.symbol_name == symbol_name) {
      symbol_charts.push_back(chart);
    }
  }
  return symbol_charts;
}

std::optional<uint32_t> ChartManager::getSymbolId(const std::string& symbol_name) const {
  std::lock_guard<std::mutex> lock(id_map_mutex_);

  auto it = symbol_id_map_.find(symbol_name);
  if (it != symbol_id_map_.end()) {
    return it->second;
  }

  // Query SymbolRegistry for the actual ID used by HotSpine
  auto all_symbols = SymbolRegistry::instance().get_all_symbols();
  for (const auto& info : all_symbols) {
    if (info.symbol == symbol_name) {
      symbol_id_map_[symbol_name] = info.id;
      return info.id;
    }
  }

  return std::nullopt;  // Unknown symbol
}

void ChartManager::update() {
  auto active_symbol_ids = bridge_->getActiveSymbols();

  for (uint32_t symbol_id : active_symbol_ids) {
    std::string symbol_str = bridge_->getSymbolName(symbol_id);

    bool has_chart = false;
    uint32_t chart_id = 0;
    for (const auto& [id, chart] : charts_) {
      if (chart.symbol_name == symbol_str && chart.symbol_id == symbol_id) {
        has_chart = true;
        chart_id = id;
        break;
      }
    }

    if (!has_chart) {
      std::string exchange = "Unknown";
      auto info = SymbolRegistry::instance().get_symbol_info(symbol_id);
      if (info) {
        exchange = info->exchange;
      }

      chart_id = create_chart(symbol_str, exchange, symbol_id, RenderEngine::TimeFrame::TF_1SEC);
    }

    populate_chart_data(chart_id);
  }
}

void ChartManager::populate_chart_data(uint32_t chart_id) {
  auto it = charts_.find(chart_id);
  if (it == charts_.end()) return;

  auto& chart = it->second;
  uint32_t symbol_id = chart.symbol_id;

  auto candles = processor_->getCandles(symbol_id, chart.timeframe);
  if (candles.empty()) {
    return;
  }

  // Incremental Update Logic: Only append or update the latest
  // candle
  size_t start_idx = 0;
  if (chart.dates.empty()) {
    chart.dates.reserve(candles.size());
    chart.opens.reserve(candles.size());
    chart.highs.reserve(candles.size());
    chart.lows.reserve(candles.size());
    chart.closes.reserve(candles.size());
    chart.volumes.reserve(candles.size());

    for (const auto& candle : candles) {
      chart.dates.push_back(static_cast<double>(candle.timestamp) / 1000000.0);
      chart.opens.push_back(static_cast<float>(candle.open));
      chart.highs.push_back(static_cast<float>(candle.high));
      chart.lows.push_back(static_cast<float>(candle.low));
      chart.closes.push_back(static_cast<float>(candle.close));
      chart.volumes.push_back(static_cast<float>(candle.volume));
    }
  } else {
    double last_stored_ts = chart.dates.back();
    bool found_overlap = false;

    // Search from the end for the last matching candle
    for (int i = (int)candles.size() - 1; i >= 0; --i) {
      double candle_ts = static_cast<double>(candles[i].timestamp) / 1000000.0;
      // Precision-safe comparison for Unix timestamps in
      // seconds (approx 1.7e9) Double precision is sufficient
      // for ~0.001s, but let's be robust (1ms = 1e-3)
      if (std::abs(candle_ts - last_stored_ts) < 0.001) {
        // Update the last candle as it might still be
        // aggregating
        chart.opens.back() = static_cast<float>(candles[i].open);
        chart.highs.back() = static_cast<float>(candles[i].high);
        chart.lows.back() = static_cast<float>(candles[i].low);
        chart.closes.back() = static_cast<float>(candles[i].close);
        chart.volumes.back() = static_cast<float>(candles[i].volume);
        start_idx = i + 1;
        found_overlap = true;
        break;
      }
    }

    if (!found_overlap) {
      // If we didn't find an overlap, it could be a gap or a
      // reset. APPEND if the new data is strictly after our
      // last data.
      if (!candles.empty() &&
          (static_cast<double>(candles[0].timestamp) / 1000000.0 > last_stored_ts)) {
        start_idx = 0;  // Prepare to append everything from the
                        // new batch
      } else {
        // Real reset or backwards jump, clear and re-populate
        chart.dates.clear();
        chart.opens.clear();
        chart.highs.clear();
        chart.lows.clear();
        chart.closes.clear();
        chart.volumes.clear();
        start_idx = 0;
      }
    }
  }

  // Append ONLY new candles (or all if reset/gap)
  for (size_t i = start_idx; i < candles.size(); ++i) {
    chart.dates.push_back(static_cast<double>(candles[i].timestamp) / 1000000.0);
    chart.opens.push_back(static_cast<float>(candles[i].open));
    chart.highs.push_back(static_cast<float>(candles[i].high));
    chart.lows.push_back(static_cast<float>(candles[i].low));
    chart.closes.push_back(static_cast<float>(candles[i].close));
    chart.volumes.push_back(static_cast<float>(candles[i].volume));
  }
}

void ChartManager::update_all_chart_timeframes(RenderEngine::TimeFrame new_timeframe) {
  // Update the timeframe for all charts
  for (auto& [chart_id, chart] : charts_) {
    // Store the old timeframe to compare
    RenderEngine::TimeFrame old_timeframe = chart.timeframe;
    (void)old_timeframe;  // Suppress unused variable warning

    // Update the timeframe
    chart.timeframe = new_timeframe;

    // Clear the old chart data to force a reload with the new
    // timeframe
    chart.dates.clear();
    chart.opens.clear();
    chart.highs.clear();
    chart.lows.clear();
    chart.closes.clear();
    chart.volumes.clear();

    // Repopulate the chart data with the new timeframe
    populate_chart_data(chart_id);
  }
}

}  // namespace BTQuant
