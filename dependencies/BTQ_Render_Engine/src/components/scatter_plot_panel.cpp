#include "../../include/components/scatter_plot_panel.hpp"

#include <iostream>
#include <random>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

ScatterPlotPanel::ScatterPlotPanel(const PanelConfig& config) : PanelBase(config) {}

void ScatterPlotPanel::initialize() { PanelBase::initialize(); }

void ScatterPlotPanel::render_content() {
  begin_panel_window();

  if (ImPlot::BeginPlot("Scatter Plot")) {
    // Generate sample data for demonstration
    static float x_data[100], y_data[100];
    static bool first_run = true;

    if (first_run) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<> dis(0.0, 1.0);

      for (int i = 0; i < 100; ++i) {
        x_data[i] = static_cast<float>(dis(gen));
        y_data[i] = static_cast<float>(dis(gen));
      }
      first_run = false;
    }

    ImPlot::SetupAxes("X", "Y");
    ImPlot::PlotScatter("Data Points", x_data, y_data, 100);
    ImPlot::EndPlot();
  }

  end_panel_window();
}

}  // namespace BTQuant