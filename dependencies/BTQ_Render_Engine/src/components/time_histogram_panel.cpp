#include "../../include/components/time_histogram_panel.hpp"

#include <iostream>
#include <random>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TimeHistogramPanel::TimeHistogramPanel(const PanelConfig& config)
    : PanelBase(config), volume_data_type_(Data::VolumeDataType::BuySellVolume) {}

void TimeHistogramPanel::initialize() { PanelBase::initialize(); }

void TimeHistogramPanel::render() {
  begin_panel_window();

  if (ImPlot::BeginPlot("Time Histogram", "Time", "Volume", ImVec2(-1, -1), ImPlotFlags_None, ImPlotAxisFlags_None, ImPlotAxisFlags_None)) {
    // Generate sample time-based histogram data for demonstration
    static double x_data[100], buy_data[100], sell_data[100];
    static bool first_run = true;

    if (first_run) {
      // Initialize time values (x-axis) and corresponding buy/sell volumes (y-axis)
      for (int i = 0; i < 100; ++i) {
        x_data[i] = static_cast<double>(i);  // Time slots

        // Create sample buy and sell volumes with some pattern
        // Using sine wave with random noise to simulate time-based volume data
        double base_volume = 100.0;
        double buy_sine = base_volume + 50.0 * sin(i * 0.2) + 20.0 * sin(i * 0.5);
        double sell_sine = base_volume + 50.0 * cos(i * 0.2) + 20.0 * cos(i * 0.5);

        double noise_buy = (static_cast<double>(rand()) / RAND_MAX) * 20.0 - 10.0;
        double noise_sell = (static_cast<double>(rand()) / RAND_MAX) * 20.0 - 10.0;

        buy_data[i] = buy_sine + noise_buy;
        sell_data[i] = sell_sine + noise_sell;
      }
      first_run = false;
    }

    // Configure the plot axes
    ImPlot::SetupAxes("Time", "Volume", ImPlotAxisFlags_None, ImPlotAxisFlags_None);

    // Handle different volume analysis types
    switch (static_cast<Data::VolumeAnalysisType>(volume_data_type_)) {
      case Data::VolumeAnalysisType::BuySellVolume: {
        // Create stacked bars: buy volume (green) on top, sell volume (red) on bottom
        // For visualization, we'll plot buy volume above zero and sell volume below zero

        // Create arrays for stacked representation
        static double buy_stack[100], sell_stack[100];
        for (int i = 0; i < 100; ++i) {
          buy_stack[i] = buy_data[i];  // Positive values for buy volume
          sell_stack[i] = -sell_data[i];  // Negative values for sell volume
        }

        // Plot buy volume (green) above x-axis
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Volume", x_data, buy_stack, 100, 0.8);
        ImPlot::PopStyleColor();

        // Plot sell volume (red) below x-axis
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Volume", x_data, sell_stack, 100, 0.8);
        ImPlot::PopStyleColor();

        break;
      }
      default: {
        // Fallback to regular histogram for other types
        static double y_data[100];
        for (int i = 0; i < 100; ++i) {
          y_data[i] = buy_data[i] - sell_data[i];  // Simple difference as fallback
        }
        ImPlot::PlotBars("Time Histogram", x_data, y_data, 100, 0.8);
        break;
      }
    }

    // Add grid for better readability (grid lines are shown by default)
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_None);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_None);

    ImPlot::EndPlot();
  }

  end_panel_window();
}

}  // namespace BTQuant