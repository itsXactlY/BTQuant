#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class ScreenerPanel : public PanelBase {
 public:
  ScreenerPanel(const PanelConfig& config);

  void initialize() override;
  void render_content() override;
};

}  // namespace BTQuant