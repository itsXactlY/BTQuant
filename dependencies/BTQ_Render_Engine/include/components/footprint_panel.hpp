#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class FootprintPanel : public PanelBase {
 public:
  FootprintPanel(const PanelConfig& config);

  void update(float dt) override;
  void render_content() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) { symbol_id_ = id; }

 private:
  uint32_t symbol_id_ = 0;
};

}  // namespace BTQuant
