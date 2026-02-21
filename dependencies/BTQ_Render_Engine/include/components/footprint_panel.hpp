#pragma once

#include <memory>

#include "data/cluster_engine.hpp"
#include "panel_base.hpp"

namespace BTQuant {

class FootprintPanel : public PanelBase {
 public:
  FootprintPanel(const PanelConfig& config, ClusterEngine* engine);

  void update(float dt) override;
  void render_content() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) { symbol_id_ = id; }

 private:
  ClusterEngine* engine_ = nullptr;
  uint32_t symbol_id_ = 0;
  float scroll_x_ = 0.0f;  // Horizontal scroll offset
};

}  // namespace BTQuant
