#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class TpoPanel : public PanelBase {
 public:
  TpoPanel(const PanelConfig& config);

  void update(float dt) override;
  void render_content() override;
};

}  // namespace BTQuant
