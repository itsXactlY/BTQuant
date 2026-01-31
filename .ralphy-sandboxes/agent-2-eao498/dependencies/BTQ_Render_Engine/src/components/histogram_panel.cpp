#include "../../include/components/histogram_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <iostream>
#include <random>

namespace BTQuant {

HistogramPanel::HistogramPanel(const PanelConfig &config) : PanelBase(config) {}

void HistogramPanel::initialize() {
    PanelBase::initialize();
}

void HistogramPanel::render() {
    begin_panel_window();
    
    if (ImPlot::BeginPlot("Histogram", "Values", "Frequency")) {
        // Generate sample data for demonstration
        static float data[1000];
        static bool first_run = true;
        
        if (first_run) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dis(0.0, 1.0);
            
            for (int i = 0; i < 1000; ++i) {
                data[i] = static_cast<float>(dis(gen)); // Random values following normal distribution
            }
            first_run = false;
        }
        
        ImPlot::SetupAxes("X", "Y");
        ImPlot::PlotHistogram("Distribution", data, 1000, 50);
        ImPlot::EndPlot();
    }
    
    end_panel_window();
}

} // namespace BTQuant