#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class LogPanel : public PanelBase {
 public:
  LogPanel(const PanelConfig& config);

  void initialize() override;
  void render_content() override;
};

}  // namespace BTQuant