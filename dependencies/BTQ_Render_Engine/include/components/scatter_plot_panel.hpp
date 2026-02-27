#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class ScatterPlotPanel : public PanelBase {
 public:
  ScatterPlotPanel(const PanelConfig& config);

  void initialize() override;
  void render() override;
};

}  // namespace BTQuant