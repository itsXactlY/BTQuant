#pragma once

#include <memory>

#include "panel_base.hpp"

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  TpoPanel(const PanelConfig& config);

  void update(float dt) override;
  void render_content() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    // Note: Exchange connection management has been moved out of the renderer
    // The renderer now only handles rendering, not data subscription
  }

 private:
  uint32_t symbol_id_ = 0;
};

}  // namespace BTQuant
