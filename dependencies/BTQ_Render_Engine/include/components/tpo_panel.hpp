#pragma once

#include <memory>

#include "MarketMicrostructureRenderer.h"
#include "panel_base.hpp"

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  TpoPanel(const PanelConfig& config, RenderEngine::MarketMicrostructureRenderer* renderer);

  void update(float dt) override;
  void render() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    if (renderer_) renderer_->setSymbol(id);
  }

 private:
  RenderEngine::MarketMicrostructureRenderer* renderer_;
  uint32_t symbol_id_ = 0;
};

}  // namespace BTQuant
