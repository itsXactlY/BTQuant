#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class TimeHistogramPanel : public PanelBase {
 public:
  TimeHistogramPanel(const PanelConfig& config);

  void initialize() override;
  void render_content() override;
};

}  // namespace BTQuant