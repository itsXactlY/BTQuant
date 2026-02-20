#pragma once

#include <memory>
#include <string>
#include <vector>

#include "market_data_processor.hpp"

namespace BTQuant {

class CorrelationHeatmapComponent {
 public:
  CorrelationHeatmapComponent(std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt);
  void render_gui();

  void set_visible(bool visible) { visible_ = visible; }
  bool is_visible() const { return visible_; }

  void set_symbols(const std::vector<std::string>& symbols);

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  bool visible_ = true;
};

}  // namespace BTQuant