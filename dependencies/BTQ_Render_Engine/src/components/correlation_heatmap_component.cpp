#include "../../include/components/correlation_heatmap_component.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace BTQuant {

CorrelationHeatmapComponent::CorrelationHeatmapComponent(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : processor_(processor) {
    // Default symbols for major crypto pairs
    symbols_ = {"BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT"};
}

void CorrelationHeatmapComponent::update(float dt) {
    // Update correlation matrix periodically (every 5 seconds)
    static float update_timer = 0.0f;
    update_timer += dt;

    if (update_timer >= 5.0f) {
        update_correlation_matrix();
        update_timer = 0.0f;
    }
}

void CorrelationHeatmapComponent::render_gui() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Correlation Heatmap", &visible_)) {
        render_controls();
        ImGui::Separator();
        render_heatmap();
    }
    ImGui::End();
}

void CorrelationHeatmapComponent::set_symbols(const std::vector<std::string>& symbols) {
    symbols_ = symbols;
    update_correlation_matrix();
}

void CorrelationHeatmapComponent::render_controls() {
    if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Timeframe selection
        const char* timeframes[] = {"1 Min", "5 Min", "15 Min", "1 Hour", "4 Hours", "1 Day"};
        int selected = static_cast<int>(timeframe_);
        if (ImGui::Combo("Timeframe", &selected, timeframes, IM_ARRAYSIZE(timeframes))) {
            timeframe_ = static_cast<RenderEngine::TimeFrame>(selected);
            update_correlation_matrix();
        }

        // Lookback period
        if (ImGui::SliderInt("Lookback Periods", &lookback_periods_, 20, 500)) {
            update_correlation_matrix();
        }

        // Symbol selection
        if (ImGui::Button("Add Symbol")) {
            // TODO: Implement symbol picker
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset to Defaults")) {
            symbols_ = {"BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT"};
            update_correlation_matrix();
        }
    }
}

void CorrelationHeatmapComponent::render_heatmap() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (correlation_matrix_.matrix.empty() || correlation_matrix_.labels.empty()) {
        ImGui::Text("No correlation data available");
        return;
    }

    int n = correlation_matrix_.labels.size();
    if (n == 0) return;

    // Create heatmap using ImPlot
    if (ImPlot::BeginPlot("Correlation Matrix", ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);

        // Set axis limits
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n - 0.5, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -0.5, n - 0.5, ImGuiCond_Always);

        // Prepare data for heatmap
        std::vector<double> xs, ys, values;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                xs.push_back(j);
                ys.push_back(i);
                values.push_back(correlation_matrix_.matrix[i][j]);
            }
        }

        // Color scale from -1 (blue) to 1 (red)
        ImPlot::ColormapScale("Correlation", -1.0, 1.0, ImVec2(0, 400), "%.2f",
                             ImPlotColormap_RdBu, true);

        // Plot heatmap
        ImPlot::PlotHeatmap("Correlation", xs.data(), ys.data(), values.data(),
                           n, n, -1.0, 1.0, nullptr, ImPlotPoint(0, 0), ImPlotPoint(1, 1));

        // Add labels
        for (int i = 0; i < n; ++i) {
            ImPlot::Annotate(i, -0.3, ImVec2(0.5f, 1.0f), ImVec2(0.5f, 0.0f),
                           correlation_matrix_.labels[i].c_str());
            ImPlot::Annotate(-0.3, i, ImVec2(1.0f, 0.5f), ImVec2(0.0f, 0.5f),
                           correlation_matrix_.labels[i].c_str());
        }

        ImPlot::EndPlot();
    }

    // Display correlation values on hover
    if (ImGui::IsItemHovered()) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 plot_pos = ImPlot::GetPlotPos();
        ImVec2 plot_size = ImPlot::GetPlotSize();

        float rel_x = (mouse_pos.x - plot_pos.x) / plot_size.x;
        float rel_y = (mouse_pos.y - plot_pos.y) / plot_size.y;

        int cell_x = static_cast<int>(rel_x * n);
        int cell_y = static_cast<int>((1.0f - rel_y) * n); // Y is inverted

        if (cell_x >= 0 && cell_x < n && cell_y >= 0 && cell_y < n) {
            double corr = correlation_matrix_.matrix[cell_y][cell_x];
            ImGui::BeginTooltip();
            ImGui::Text("%s vs %s: %s",
                       correlation_matrix_.labels[cell_y].c_str(),
                       correlation_matrix_.labels[cell_x].c_str(),
                       format_correlation_value(corr).c_str());
            ImGui::EndTooltip();
        }
    }
}

void CorrelationHeatmapComponent::update_correlation_matrix() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    int n = symbols_.size();
    correlation_matrix_.matrix.assign(n, std::vector<double>(n, 0.0));
    correlation_matrix_.labels = symbols_;

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double corr = calculate_correlation(symbols_[i], symbols_[j]);
            correlation_matrix_.matrix[i][j] = corr;
            correlation_matrix_.matrix[j][i] = corr; // Symmetric
        }
    }

    correlation_matrix_.last_update = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double CorrelationHeatmapComponent::calculate_correlation(const std::string& symbol1,
                                                         const std::string& symbol2) {
    if (symbol1 == symbol2) return 1.0;

    auto returns1 = get_returns(symbol1);
    auto returns2 = get_returns(symbol2);

    if (returns1.size() != returns2.size() || returns1.size() < 2) {
        return 0.0;
    }

    // Calculate Pearson correlation coefficient
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    size_t n = returns1.size();

    for (size_t i = 0; i < n; ++i) {
        sum_x += returns1[i];
        sum_y += returns2[i];
        sum_xy += returns1[i] * returns2[i];
        sum_x2 += returns1[i] * returns1[i];
        sum_y2 += returns2[i] * returns2[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    if (denominator == 0.0) return 0.0;

    return numerator / denominator;
}

std::vector<double> CorrelationHeatmapComponent::get_returns(const std::string& symbol) {
    // Get price data from processor
    auto candles = processor_->getCandles(symbol, timeframe_, lookback_periods_ + 1);
    std::vector<double> returns;

    if (candles.size() < 2) return returns;

    for (size_t i = 1; i < candles.size(); ++i) {
        double prev_close = candles[i-1].close;
        double curr_close = candles[i].close;
        if (prev_close > 0.0) {
            returns.push_back((curr_close - prev_close) / prev_close);
        }
    }

    return returns;
}

ImVec4 CorrelationHeatmapComponent::get_correlation_color(double correlation) const {
    // Blue for negative, red for positive correlation
    if (correlation < 0) {
        return ImVec4(0.0f, 0.5f + 0.5f * correlation, 1.0f, 1.0f);
    } else {
        return ImVec4(1.0f, 0.5f - 0.5f * correlation, 0.0f, 1.0f);
    }
}

std::string CorrelationHeatmapComponent::format_correlation_value(double value) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", value);
    return std::string(buf);
}

} // namespace BTQuant