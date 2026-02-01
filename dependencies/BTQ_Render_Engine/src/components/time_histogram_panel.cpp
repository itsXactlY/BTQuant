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
      case Data::VolumeAnalysisType::Delta: {
        // Delta histogram: bars originate from zero line
        // Positive delta extends up (green), negative delta extends down (red)
        static double delta_data[100];
        for (int i = 0; i < 100; ++i) {
          delta_data[i] = buy_data[i] - sell_data[i];  // Delta = BuyVolume - SellVolume
        }

        // Separate positive and negative values for different coloring
        static double pos_values[100], neg_values[100];
        for (int i = 0; i < 100; ++i) {
          if (delta_data[i] >= 0) {
            pos_values[i] = delta_data[i];
            neg_values[i] = 0.0;
          } else {
            pos_values[i] = 0.0;
            neg_values[i] = delta_data[i];
          }
        }

        // Plot positive delta values (green)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Positive Delta", x_data, pos_values, 100, 0.8);
        ImPlot::PopStyleColor();

        // Plot negative delta values (red)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Negative Delta", x_data, neg_values, 100, 0.8);
        ImPlot::PopStyleColor();

        break;
      }
      case Data::VolumeAnalysisType::CumulativeDelta: {
        // Cumulative Delta: line chart overlay showing running sum of delta
        // Color transitions from red to green as cumulative delta crosses zero
        static double cum_delta_data[100];

        // Calculate cumulative delta (running sum)
        double running_sum = 0.0;
        for (int i = 0; i < 100; ++i) {
          double current_delta = buy_data[i] - sell_data[i];
          running_sum += current_delta;
          cum_delta_data[i] = running_sum;
        }

        // Plot the cumulative delta as a line chart with color transitions
        // We'll draw segments individually to allow color changes based on sign
        for (int i = 0; i < 99; ++i) {  // 99 segments for 100 points
          // Determine color based on the sign of the two points to handle zero crossings
          bool start_positive = cum_delta_data[i] >= 0;
          bool end_positive = cum_delta_data[i+1] >= 0;

          // If both points are on the same side of zero, use consistent color
          if (start_positive == end_positive) {
            ImVec4 color = start_positive ?
                           ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for positive
                           ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for negative

            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("", &x_data[i], &cum_delta_data[i], 2);  // Length 2: current and next point
            ImPlot::PopStyleColor();
          } else {
            // Line segment crosses zero, so we need to interpolate the crossing point
            // Find the x-coordinate where the line crosses zero
            double dx = x_data[i+1] - x_data[i];
            if (dx != 0.0) {  // Avoid division by zero
              double slope = (cum_delta_data[i+1] - cum_delta_data[i]) / dx;
              if (slope != 0.0) {  // Avoid division by zero when slope is zero
                double zero_cross_x = x_data[i] - (cum_delta_data[i] / slope);

                // Draw first segment with starting color
                ImVec4 start_color = start_positive ?
                                     ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for positive
                                     ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for negative

                // Create temporary arrays for the partial segment
                double partial_x[2] = {x_data[i], zero_cross_x};
                double partial_y[2] = {cum_delta_data[i], 0.0};

                ImPlot::PushStyleColor(ImPlotCol_Line, start_color);
                ImPlot::PlotLine("", partial_x, partial_y, 2);
                ImPlot::PopStyleColor();

                // Draw second segment with ending color
                ImVec4 end_color = end_positive ?
                                   ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for positive
                                   ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for negative

                double partial_x2[2] = {zero_cross_x, x_data[i+1]};
                double partial_y2[2] = {0.0, cum_delta_data[i+1]};

                ImPlot::PushStyleColor(ImPlotCol_Line, end_color);
                ImPlot::PlotLine("", partial_x2, partial_y2, 2);
                ImPlot::PopStyleColor();
              } else {
                // Slope is zero, meaning both values are zero (shouldn't happen in this branch)
                // Just draw with start color
                ImVec4 color = start_positive ?
                               ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for positive
                               ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for negative

                ImPlot::PushStyleColor(ImPlotCol_Line, color);
                ImPlot::PlotLine("", &x_data[i], &cum_delta_data[i], 2);
                ImPlot::PopStyleColor();
              }
            } else {
              // x coordinates are the same (shouldn't happen in normal time series)
              // Just draw with start color
              ImVec4 color = start_positive ?
                             ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for positive
                             ImVec4(1.0f, 0.0f, 0.0f, 1.0f);   // Red for negative

              ImPlot::PushStyleColor(ImPlotCol_Line, color);
              ImPlot::PlotLine("", &x_data[i], &cum_delta_data[i], 2);
              ImPlot::PopStyleColor();
            }
          }
        }

        // Add legend entry for the cumulative delta line
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
        ImPlot::PlotDummy("Cumulative Delta");  // Dummy plot for legend
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