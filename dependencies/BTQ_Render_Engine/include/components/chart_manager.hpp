#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hotspine_data_bridge.hpp"
#include "imgui.h"
#include "implot.h"
#include "market_data_processor.hpp"

namespace BTQuant {

struct ChartInstance {
  std::string symbol_name;
  std::string exchange_name;  // To prevent collisions
  RenderEngine::TimeFrame timeframe;
  uint32_t symbol_id;
  uint32_t chart_id;
  bool visible = true;
  bool minimized = false;
  ImVec2 position = {0, 0};
  ImVec2 size = {600, 400};

  // Candle data for rendering
  std::vector<double> dates;
  std::vector<float> opens;
  std::vector<float> highs;
  std::vector<float> lows;
  std::vector<float> closes;
  std::vector<float> volumes;

  // Volume profile data
  std::vector<float> vp_prices;
  std::vector<float> vp_volumes;
};

class ChartManager {
 public:
  ChartManager(std::shared_ptr<HotSpineDataBridge> bridge,
               std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  uint32_t create_chart(const std::string& symbol_name, const std::string& exchange_name,
                        uint32_t symbol_id, RenderEngine::TimeFrame timeframe);
  void destroy_chart(uint32_t chart_id);
  void toggle_chart_visibility(uint32_t chart_id);
  void toggle_chart_minimization(uint32_t chart_id);
  void update_chart_position(uint32_t chart_id, const ImVec2& position);
  void update_chart_size(uint32_t chart_id, const ImVec2& size);

  const std::unordered_map<uint32_t, ChartInstance>& get_charts() const;
  std::vector<ChartInstance> get_visible_charts() const;
  std::vector<ChartInstance> get_charts_for_symbol(const std::string& symbol_name) const;

  void update();
  void populate_chart_data(uint32_t chart_id);

  // Helper to map symbol name to ID
  std::optional<uint32_t> getSymbolId(const std::string& symbol_name) const;

  // Getter for bridge access
  std::shared_ptr<HotSpineDataBridge> get_bridge() const { return bridge_; }

  // Method to update all chart timeframes
  void update_all_chart_timeframes(RenderEngine::TimeFrame new_timeframe);

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::unordered_map<uint32_t, ChartInstance> charts_;
  uint32_t next_chart_id_ = 0;

  // Symbol name to ID mapping (cached for performance)
  mutable std::unordered_map<std::string, uint32_t> symbol_id_map_;
  mutable std::mutex id_map_mutex_;
};

}  // namespace BTQuant
