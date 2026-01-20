#include "../../include/components/chart_manager.hpp"
#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"

namespace BTQuant {

ChartManager::ChartManager(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : bridge_(bridge), processor_(processor), next_chart_id_(0) {}

uint32_t ChartManager::create_chart(const std::string &symbol,
                                    RenderEngine::TimeFrame timeframe) {
  ChartInstance chart;
  chart.symbol = symbol;
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

void ChartManager::update_chart_position(uint32_t chart_id,
                                         const ImVec2 &position) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.position = position;
  }
}

void ChartManager::update_chart_size(uint32_t chart_id, const ImVec2 &size) {
  auto it = charts_.find(chart_id);
  if (it != charts_.end()) {
    it->second.size = size;
  }
}

const std::unordered_map<uint32_t, ChartInstance> &
ChartManager::get_charts() const {
  return charts_;
}

std::vector<ChartInstance> ChartManager::get_visible_charts() const {
  std::vector<ChartInstance> visible_charts;
  visible_charts.reserve(charts_.size());
  for (const auto &[id, chart] : charts_) {
    if (chart.visible) {
      visible_charts.push_back(chart);
    }
  }
  return visible_charts;
}

std::vector<ChartInstance>
ChartManager::get_charts_for_symbol(const std::string &symbol) const {
  std::vector<ChartInstance> symbol_charts;
  for (const auto &[id, chart] : charts_) {
    if (chart.symbol == symbol) {
      symbol_charts.push_back(chart);
    }
  }
  return symbol_charts;
}

uint32_t ChartManager::getSymbolId(const std::string &symbol) const {
  std::lock_guard<std::mutex> lock(id_map_mutex_);

  auto it = symbol_id_map_.find(symbol);
  if (it != symbol_id_map_.end()) {
    return it->second;
  }

  // Query SymbolRegistry for the actual ID used by HotSpine
  auto all_symbols = SymbolRegistry::instance().get_all_symbols();
  for (const auto &info : all_symbols) {
    if (info.symbol == symbol) {
      symbol_id_map_[symbol] = info.id;
      return info.id;
    }
  }

  return 0; // Unknown symbol
}

void ChartManager::update() {
  auto active_symbol_ids = bridge_->getActiveSymbols();

  for (uint32_t symbol_id : active_symbol_ids) {
    std::string symbol = bridge_->getSymbolName(symbol_id);

    bool has_chart = false;
    uint32_t chart_id = 0;
    for (const auto &[id, chart] : charts_) {
      if (chart.symbol == symbol) {
        has_chart = true;
        chart_id = id;
        break;
      }
    }

    if (!has_chart) {
      chart_id = create_chart(symbol, RenderEngine::TimeFrame::TF_1MIN);
    }

    populate_chart_data(chart_id);
  }
}

void ChartManager::populate_chart_data(uint32_t chart_id) {
  auto it = charts_.find(chart_id);
  if (it == charts_.end())
    return;

  auto &chart = it->second;

  uint32_t symbol_id = getSymbolId(chart.symbol);
  if (symbol_id == 0) {
    return;
  }

  auto candles = processor_->getCandles(symbol_id, chart.timeframe);
  if (candles.empty()) {
    return;
  }

  chart.dates.clear();
  chart.opens.clear();
  chart.highs.clear();
  chart.lows.clear();
  chart.closes.clear();
  chart.volumes.clear();

  chart.dates.reserve(candles.size());
  chart.opens.reserve(candles.size());
  chart.highs.reserve(candles.size());
  chart.lows.reserve(candles.size());
  chart.closes.reserve(candles.size());
  chart.volumes.reserve(candles.size());

  for (const auto &candle : candles) {
    chart.dates.push_back(static_cast<double>(candle.timestamp) / 1000000.0);
    chart.opens.push_back(static_cast<float>(candle.open));
    chart.highs.push_back(static_cast<float>(candle.high));
    chart.lows.push_back(static_cast<float>(candle.low));
    chart.closes.push_back(static_cast<float>(candle.close));
    chart.volumes.push_back(static_cast<float>(candle.volume));
  }
}

} // namespace BTQuant
