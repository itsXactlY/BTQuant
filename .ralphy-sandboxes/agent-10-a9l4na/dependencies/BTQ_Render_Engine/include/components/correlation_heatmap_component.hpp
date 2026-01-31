#pragma once

#include "imgui.h"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {

class CorrelationHeatmapComponent {
public:
  CorrelationHeatmapComponent(
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt);
  void render_gui();

  void set_visible(bool visible) { visible_ = visible; }
  bool is_visible() const { return visible_; }

  // Configuration
  void set_symbols(const std::vector<std::string> &symbols);
  void set_timeframe(RenderEngine::TimeFrame timeframe) {
    timeframe_ = timeframe;
  }
  void set_lookback_period(int periods) { lookback_periods_ = periods; }

private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  bool visible_ = true;

  // Configuration
  std::vector<std::string> symbols_;
  RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::TF_1SEC;
  int lookback_periods_ = 100;

  // Cached correlation data
  struct CorrelationMatrix {
    std::vector<std::vector<double>> matrix;
    std::vector<std::string> labels;
    uint64_t last_update = 0;
  };

  CorrelationMatrix correlation_matrix_;
  std::mutex data_mutex_;

  // Rendering
  void render_heatmap();
  void render_controls();

  // Data processing
  void update_correlation_matrix();
  double calculate_correlation(const std::string &symbol1,
                               const std::string &symbol2);
  std::vector<double> get_returns(const std::string &symbol);
  std::optional<uint32_t> get_symbol_id(const std::string &symbol);

  // Color mapping
  ImVec4 get_correlation_color(double correlation) const;
  std::string format_correlation_value(double value) const;
};

} // namespace BTQuant