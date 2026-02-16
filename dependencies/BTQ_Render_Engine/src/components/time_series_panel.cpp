#include "../../include/components/time_series_panel.hpp"

#include <cmath>
#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TimeSeriesPanel::TimeSeriesPanel(const PanelConfig& config) : PanelBase(config) {}

void TimeSeriesPanel::initialize() { PanelBase::initialize(); }

void TimeSeriesPanel::render_content() {
  begin_panel_window();

  if (ImPlot::BeginPlot("Time Series")) {
    // Generate sample time series data for demonstration
    static double x_data[1000], y_data[1000];
    static bool first_run = true;

    if (first_run) {
      for (int i = 0; i < 1000; ++i) {
        x_data[i] = i * 0.01;  // Time axis
        y_data[i] = sin(x_data[i]) +
                    0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);  // Sine wave with noise
      }
      first_run = false;
    }

    ImPlot::SetupAxes("Time", "Value");
    ImPlot::PlotLine("Series", x_data, y_data, 1000);
    ImPlot::EndPlot();
  }

  end_panel_window();
}

}  // namespace BTQuant