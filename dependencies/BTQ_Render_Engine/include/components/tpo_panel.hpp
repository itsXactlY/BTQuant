#pragma once

#include <memory>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  // DEPRECATED - Legacy hotspine
  TpoPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
           std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    // Note: Exchange connection management has been moved out of the renderer
    // The renderer now only handles rendering, not data subscription
  }

 private:
  // DEPRECATED - Legacy hotspine
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t symbol_id_ = 0;
};

}  // namespace BTQuant
