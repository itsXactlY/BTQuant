#include "../../include/components/time_histogram_panel.hpp"

#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TimeHistogramPanel::TimeHistogramPanel(const PanelConfig& config)
    : PanelBase(config),
      volume_data_type_(Data::VolumeDataType::BuySellVolume),
      locked_min_y_(-50.0),
      locked_max_y_(150.0) {}

void TimeHistogramPanel::initialize() { PanelBase::initialize(); }

void TimeHistogramPanel::render() {
  begin_panel_window();

  // Add volume analysis type selector
  const char* volume_analysis_types[] = {
    "Trades",             // Total number of trades
    "BuyTrades",          // Number of buy trades
    "SellTrades",         // Number of sell trades
    "Volume",             // Total volume (bid + ask)
    "BuyVolume",          // Volume of buy trades
    "SellVolume",         // Volume of sell trades
    "BuyVolumePercent",   // Percentage of buy volume
    "SellVolumePercent",  // Percentage of sell volume
    "BuySellVolume",      // Difference between buy and sell volume (BuyVolume - SellVolume)
    "Delta",              // Net difference between buy and sell volume (BuyVolume - SellVolume)
    "DeltaPercent",       // Delta as percentage of total volume
    "CumulativeDelta",    // Running sum of delta values
    "AverageSize",        // Average trade size
    "AverageBuySize",     // Average size of buy trades
    "AverageSellSize",    // Average size of sell trades
    "MaxOneTradeVolume",  // Maximum volume of a single trade
    "FilteredVolume",     // Volume filtered by specific criteria
    "SplitVolume"         // Split volume display: buy volume on left half, sell volume on right half
  };

  int current_type = static_cast<int>(volume_data_type_);
  if (ImGui::Combo("Volume Analysis Type", &current_type, volume_analysis_types, IM_ARRAYSIZE(volume_analysis_types))) {
    volume_data_type_ = static_cast<Data::VolumeDataType>(current_type);
  }

  // Add auto-scaling and scale locking controls
  ImGui::Checkbox("Auto-Scale Y-Axis", &auto_scale_y_axis_);
  ImGui::SameLine();
  ImGui::Checkbox("Lock Scale", &lock_y_axis_scale_);

  // If auto-scaling is disabled and not locked, show manual range controls
  if (!auto_scale_y_axis_ && !lock_y_axis_scale_) {
    ImGui::SliderScalar("Min Y", ImGuiDataType_Double, &locked_min_y_, &locked_min_y_, &locked_max_y_, "%.2f");
    ImGui::SliderScalar("Max Y", ImGuiDataType_Double, &locked_max_y_, &locked_min_y_, &locked_max_y_, "%.2f");
  }

  // Define axis flags based on auto-scaling and locking settings
  ImPlotAxisFlags y_axis_flags = ImPlotAxisFlags_None;
  if (lock_y_axis_scale_) {
    // When locked, prevent user from changing the scale
    y_axis_flags |= ImPlotAxisFlags_Lock;
  }

  if (ImPlot::BeginPlot("Time Histogram", "Time", "Volume", ImVec2(-1, -1), ImPlotFlags_None, ImPlotAxisFlags_None, y_axis_flags)) {
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

    // Variables to track min/max values for auto-scaling
    double min_y_value = 0.0;
    double max_y_value = 100.0;
    bool has_valid_data = false;

    // Handle Y-axis scaling based on auto-scale and lock settings
    if (lock_y_axis_scale_) {
      // If scale is locked, set the fixed Y-axis range
      ImPlot::SetNextAxisLimits(ImAxis_Y1, locked_min_y_, locked_max_y_, ImGuiCond_Always);
    } else if (auto_scale_y_axis_) {
      // If auto-scaling is enabled, calculate min/max from the data to be plotted
      // This will be handled by setting appropriate limits after processing all data
    } else {
      // If auto-scaling is disabled but not locked, use manual range
      ImPlot::SetNextAxisLimits(ImAxis_Y1, locked_min_y_, locked_max_y_, ImGuiCond_FirstUseEver);
    }

    // Helper function to add tooltip to bars
    auto addBarTooltip = [](const char* label_id, const double* xs, const double* ys, int count, double bar_size) {
        // Check if mouse is hovering over the plot
        if (ImPlot::IsPlotHovered()) {
            double mouse_x = ImPlot::GetPlotMousePos().x;
            double mouse_y = ImPlot::GetPlotMousePos().y;

            // Find the closest bar to the mouse position
            for (int i = 0; i < count; ++i) {
                double bar_center_x = xs[i];
                double bar_value = ys[i];

                // Check if mouse is horizontally within the bar
                if (mouse_x >= bar_center_x - bar_size/2.0 && mouse_x <= bar_center_x + bar_size/2.0) {
                    // For bars that extend from y=0, check if mouse is vertically within the bar bounds
                    double bar_bottom = std::min(0.0, bar_value);
                    double bar_top = std::max(0.0, bar_value);

                    if (mouse_y >= bar_bottom && mouse_y <= bar_top) {
                        // Show tooltip with exact value
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(2) << "Value: " << bar_value;

                        // Create a unique ID for the tooltip
                        ImGui::SetTooltip("%s", ss.str().c_str());
                        break;
                    }
                }
            }
        }
    };

    // Handle different volume analysis types
    switch (static_cast<Data::VolumeAnalysisType>(volume_data_type_)) {
      case Data::VolumeAnalysisType::Trades: {
        // Total number of trades - using a constant value for demonstration
        static double trades_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate trade counts with some variation
          trades_data[i] = 10.0 + 5.0 * sin(i * 0.1) + 2.0 * cos(i * 0.3);
        }

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = trades_data[i];
              has_valid_data = true;
            } else {
              if (trades_data[i] < min_y_value) min_y_value = trades_data[i];
              if (trades_data[i] > max_y_value) max_y_value = trades_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.5f, 0.5f, 1.0f, 0.7f));  // Blue fill
        ImPlot::PlotBars("Trades", x_data, trades_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Trades", x_data, trades_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::BuyTrades: {
        // Number of buy trades - using a constant value for demonstration
        static double buy_trades_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate buy trade counts with some variation
          buy_trades_data[i] = 5.0 + 3.0 * sin(i * 0.15) + 1.5 * cos(i * 0.25);
        }

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = buy_trades_data[i];
              has_valid_data = true;
            } else {
              if (buy_trades_data[i] < min_y_value) min_y_value = buy_trades_data[i];
              if (buy_trades_data[i] > max_y_value) max_y_value = buy_trades_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Trades", x_data, buy_trades_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Buy Trades", x_data, buy_trades_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::SellTrades: {
        // Number of sell trades - using a constant value for demonstration
        static double sell_trades_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate sell trade counts with some variation
          sell_trades_data[i] = 5.0 + 3.0 * cos(i * 0.15) + 1.5 * sin(i * 0.25);
        }

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = sell_trades_data[i];
              has_valid_data = true;
            } else {
              if (sell_trades_data[i] < min_y_value) min_y_value = sell_trades_data[i];
              if (sell_trades_data[i] > max_y_value) max_y_value = sell_trades_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Trades", x_data, sell_trades_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Sell Trades", x_data, sell_trades_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::Volume: {
        // Total volume (bid + ask) - sum of buy and sell volumes
        static double total_vol_data[100];
        for (int i = 0; i < 100; ++i) {
          total_vol_data[i] = buy_data[i] + sell_data[i];
        }

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = total_vol_data[i];
              has_valid_data = true;
            } else {
              if (total_vol_data[i] < min_y_value) min_y_value = total_vol_data[i];
              if (total_vol_data[i] > max_y_value) max_y_value = total_vol_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 1.0f, 0.0f, 0.7f));  // Yellow fill
        ImPlot::PlotBars("Total Volume", x_data, total_vol_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Total Volume", x_data, total_vol_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::BuyVolume: {
        // Volume of buy trades
        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = buy_data[i];
              has_valid_data = true;
            } else {
              if (buy_data[i] < min_y_value) min_y_value = buy_data[i];
              if (buy_data[i] > max_y_value) max_y_value = buy_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Volume", x_data, buy_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Buy Volume", x_data, buy_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::SellVolume: {
        // Volume of sell trades
        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = sell_data[i];
              has_valid_data = true;
            } else {
              if (sell_data[i] < min_y_value) min_y_value = sell_data[i];
              if (sell_data[i] > max_y_value) max_y_value = sell_data[i];
            }
          }
        }

        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Volume", x_data, sell_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Sell Volume", x_data, sell_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::BuyVolumePercent: {
        // Percentage of buy volume relative to total volume
        static double buy_vol_percent[100];
        for (int i = 0; i < 100; ++i) {
          double total = buy_data[i] + sell_data[i];
          if (total > 0) {
            buy_vol_percent[i] = (buy_data[i] / total) * 100.0;
          } else {
            buy_vol_percent[i] = 0.0;
          }
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Volume %", x_data, buy_vol_percent, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Buy Volume %", x_data, buy_vol_percent, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::SellVolumePercent: {
        // Percentage of sell volume relative to total volume
        static double sell_vol_percent[100];
        for (int i = 0; i < 100; ++i) {
          double total = buy_data[i] + sell_data[i];
          if (total > 0) {
            sell_vol_percent[i] = (sell_data[i] / total) * 100.0;
          } else {
            sell_vol_percent[i] = 0.0;
          }
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Volume %", x_data, sell_vol_percent, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Sell Volume %", x_data, sell_vol_percent, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::BuySellVolume: {
        // Create stacked bars: buy volume (green) on top, sell volume (red) on bottom
        // For visualization, we'll plot buy volume above zero and sell volume below zero

        // Create arrays for stacked representation
        static double buy_stack[100], sell_stack[100];
        for (int i = 0; i < 100; ++i) {
          buy_stack[i] = buy_data[i];  // Positive values for buy volume
          sell_stack[i] = -sell_data[i];  // Negative values for sell volume
        }

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = buy_stack[i];
              has_valid_data = true;
            } else {
              if (buy_stack[i] < min_y_value) min_y_value = buy_stack[i];
              if (buy_stack[i] > max_y_value) max_y_value = buy_stack[i];
            }

            if (sell_stack[i] < min_y_value) min_y_value = sell_stack[i];
            if (sell_stack[i] > max_y_value) max_y_value = sell_stack[i];
          }
        }

        // Plot buy volume (green) above x-axis
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Volume", x_data, buy_stack, 100, 0.8);

        // Plot sell volume (red) below x-axis
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Volume", x_data, sell_stack, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality for both buy and sell volumes
        addBarTooltip("Buy Volume", x_data, buy_stack, 100, 0.8);
        addBarTooltip("Sell Volume", x_data, sell_stack, 100, 0.8);
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

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = delta_data[i];
              has_valid_data = true;
            } else {
              if (delta_data[i] < min_y_value) min_y_value = delta_data[i];
              if (delta_data[i] > max_y_value) max_y_value = delta_data[i];
            }
          }
        }

        // Plot positive delta values (green)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Positive Delta", x_data, pos_values, 100, 0.8);

        // Plot negative delta values (red)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Negative Delta", x_data, neg_values, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality for both positive and negative deltas
        addBarTooltip("Positive Delta", x_data, pos_values, 100, 0.8);
        addBarTooltip("Negative Delta", x_data, neg_values, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::DeltaPercent: {
        // Delta as percentage of total volume
        static double delta_percent[100];
        for (int i = 0; i < 100; ++i) {
          double total = buy_data[i] + sell_data[i];
          if (total > 0) {
            delta_percent[i] = ((buy_data[i] - sell_data[i]) / total) * 100.0;
          } else {
            delta_percent[i] = 0.0;
          }
        }

        // Separate positive and negative values for different coloring
        static double pos_values[100], neg_values[100];
        for (int i = 0; i < 100; ++i) {
          if (delta_percent[i] >= 0) {
            pos_values[i] = delta_percent[i];
            neg_values[i] = 0.0;
          } else {
            pos_values[i] = 0.0;
            neg_values[i] = delta_percent[i];
          }
        }

        // Plot positive delta percent values (green)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Positive Delta %", x_data, pos_values, 100, 0.8);

        // Plot negative delta percent values (red)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Negative Delta %", x_data, neg_values, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality for both positive and negative delta percentages
        addBarTooltip("Positive Delta %", x_data, pos_values, 100, 0.8);
        addBarTooltip("Negative Delta %", x_data, neg_values, 100, 0.8);
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

        // Update min/max values for auto-scaling if enabled
        if (auto_scale_y_axis_) {
          for (int i = 0; i < 100; ++i) {
            if (!has_valid_data) {
              min_y_value = max_y_value = cum_delta_data[i];
              has_valid_data = true;
            } else {
              if (cum_delta_data[i] < min_y_value) min_y_value = cum_delta_data[i];
              if (cum_delta_data[i] > max_y_value) max_y_value = cum_delta_data[i];
            }
          }
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
      case Data::VolumeAnalysisType::AverageSize: {
        // Average trade size - using a constant value for demonstration
        static double avg_size_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate average trade size with some variation
          avg_size_data[i] = 50.0 + 20.0 * sin(i * 0.05) + 10.0 * cos(i * 0.1);
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.7f, 0.3f, 1.0f, 0.7f));  // Purple fill
        ImPlot::PlotBars("Avg Size", x_data, avg_size_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Avg Size", x_data, avg_size_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::AverageBuySize: {
        // Average size of buy trades - using a constant value for demonstration
        static double avg_buy_size_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate average buy trade size with some variation
          avg_buy_size_data[i] = 45.0 + 15.0 * sin(i * 0.06) + 8.0 * cos(i * 0.12);
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 0.8f, 0.0f, 0.7f));  // Dark green fill
        ImPlot::PlotBars("Avg Buy Size", x_data, avg_buy_size_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Avg Buy Size", x_data, avg_buy_size_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::AverageSellSize: {
        // Average size of sell trades - using a constant value for demonstration
        static double avg_sell_size_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate average sell trade size with some variation
          avg_sell_size_data[i] = 48.0 + 18.0 * cos(i * 0.06) + 9.0 * sin(i * 0.12);
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.8f, 0.0f, 0.0f, 0.7f));  // Dark red fill
        ImPlot::PlotBars("Avg Sell Size", x_data, avg_sell_size_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Avg Sell Size", x_data, avg_sell_size_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::MaxOneTradeVolume: {
        // Maximum volume of a single trade - using a constant value for demonstration
        static double max_trade_vol_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate max trade volume with some variation
          max_trade_vol_data[i] = 100.0 + 40.0 * sin(i * 0.04) + 20.0 * cos(i * 0.08);
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.5f, 0.0f, 0.7f));  // Orange fill
        ImPlot::PlotBars("Max Trade Vol", x_data, max_trade_vol_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Max Trade Vol", x_data, max_trade_vol_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::FilteredVolume: {
        // Volume filtered by specific criteria - using a combination of buy and sell for demo
        static double filtered_vol_data[100];
        for (int i = 0; i < 100; ++i) {
          // Simulate filtered volume as weighted combination
          filtered_vol_data[i] = 0.6 * buy_data[i] + 0.4 * sell_data[i];
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 0.7f, 0.7f, 0.7f));  // Teal fill
        ImPlot::PlotBars("Filtered Volume", x_data, filtered_vol_data, 100, 0.8);
        ImPlot::PopStyleColor();

        // Add tooltip functionality
        addBarTooltip("Filtered Volume", x_data, filtered_vol_data, 100, 0.8);
        break;
      }
      case Data::VolumeAnalysisType::SplitVolume: {
        // Split volume display: buy volume on left half, sell volume on right half
        // This requires a different visualization approach
        // For this demo, we'll show both volumes side by side

        // Create arrays for split representation
        static double buy_split[100], sell_split[100];
        for (int i = 0; i < 100; ++i) {
          buy_split[i] = buy_data[i] / 2.0;  // Half height for split view
          sell_split[i] = sell_data[i] / 2.0;  // Half height for split view
        }

        // Plot buy volume (green)
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 1.0f, 0.0f, 0.7f));  // Green fill
        ImPlot::PlotBars("Buy Volume", x_data, buy_split, 100, 0.4);  // Narrower bars

        // Plot sell volume (red) shifted slightly to the right
        static double x_shifted[100];
        for (int i = 0; i < 100; ++i) {
          x_shifted[i] = x_data[i] + 0.2;  // Shift right
        }
        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f, 0.0f, 0.0f, 0.7f));  // Red fill
        ImPlot::PlotBars("Sell Volume", x_shifted, sell_split, 100, 0.4);  // Narrower bars
        ImPlot::PopStyleColor();

        // Add tooltip functionality for both buy and sell volumes
        addBarTooltip("Buy Volume", x_data, buy_split, 100, 0.4);
        addBarTooltip("Sell Volume", x_shifted, sell_split, 100, 0.4);
        break;
      }
      default: {
        // Fallback to regular histogram for other types
        static double y_data[100];
        for (int i = 0; i < 100; ++i) {
          y_data[i] = buy_data[i] - sell_data[i];  // Simple difference as fallback
        }
        ImPlot::PlotBars("Time Histogram", x_data, y_data, 100, 0.8);

        // Add tooltip functionality
        addBarTooltip("Time Histogram", x_data, y_data, 100, 0.8);
        break;
      }
    }

    // Add grid for better readability (grid lines are shown by default)
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_None);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_None);

    // Apply auto-scaling if enabled and we have valid data
    if (auto_scale_y_axis_ && has_valid_data && !lock_y_axis_scale_) {
      // Add some padding to the calculated range
      double range = max_y_value - min_y_value;
      if (range == 0) {
        range = 1.0; // Prevent division by zero if all values are the same
      }
      double padding = range * 0.05; // 5% padding

      ImPlot::SetNextAxisLimits(ImAxis_Y1, min_y_value - padding, max_y_value + padding, ImGuiCond_Always);

      // Update the locked values to reflect the current auto-scaled range
      locked_min_y_ = min_y_value - padding;
      locked_max_y_ = max_y_value + padding;
    }

    ImPlot::EndPlot();
  }

  end_panel_window();
}

}  // namespace BTQuant