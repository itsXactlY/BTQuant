#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class HistogramPanel : public PanelBase {
 public:
  HistogramPanel(const PanelConfig& config);

  void initialize() override;
  void render_content() override;
};

}  // namespace BTQuant