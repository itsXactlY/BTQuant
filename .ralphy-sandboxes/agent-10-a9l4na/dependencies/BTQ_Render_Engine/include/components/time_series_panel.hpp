#pragma once

#include "panel_base.hpp"

namespace BTQuant {

class TimeSeriesPanel : public PanelBase {
public:
    TimeSeriesPanel(const PanelConfig &config);

    void initialize() override;
    void render() override;
};

} // namespace BTQuant