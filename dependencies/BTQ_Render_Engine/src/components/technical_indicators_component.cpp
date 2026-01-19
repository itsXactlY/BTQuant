/**
 * BTQuant Technical Indicators Component
 *
 * Comprehensive technical analysis display with support for multiple indicators
 * and customizable parameters.
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include "../../include/indicator.hpp"
#include <imgui.h>
#include <implot.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace BTQuant {

struct TechnicalIndicatorsComponent : public UIComponent {
    TechnicalIndicatorsComponent(const glm::vec2 &p, const glm::vec2 &s)
        : UIComponent(p, s) {
        // Initialize default indicators
        initialize_indicators();
    }

    void initialize_indicators() {
        // Simple Moving Average (SMA)
        sma_10_ = std::make_unique<SMAIndicator>(10);
        sma_20_ = std::make_unique<SMAIndicator>(20);
        sma_50_ = std::make_unique<SMAIndicator>(50);

        // Exponential Moving Average (EMA)
        ema_10_ = std::make_unique<EMAIndicator>(10);
        ema_20_ = std::make_unique<EMAIndicator>(20);
        ema_50_ = std::make_unique<EMAIndicator>(50);

        // RSI
        rsi_14_ = std::make_unique<RSIIndicator>(14);

        // MACD
        macd_ = std::make_unique<MACDIndicator>(12, 26, 9);
    }

    void update(float dt) override {
        (void)dt;

        // Update indicators with new price data
        if (!price_data_.empty()) {
            float latest_price = price_data_.back();

            // Update MA indicators
            sma_10_->update(latest_price);
            sma_20_->update(latest_price);
            sma_50_->update(latest_price);
            ema_10_->update(latest_price);
            ema_20_->update(latest_price);
            ema_50_->update(latest_price);

            // Update RSI
            rsi_14_->update(latest_price);

            // Update MACD
            macd_->update(latest_price);

            // Store indicator values
            if (sma_10_->is_ready()) {
                sma_10_values_.push_back(sma_10_->get_value());
                if (sma_10_values_.size() > max_data_points_) {
                    sma_10_values_.erase(sma_10_values_.begin());
                }
            }

            if (sma_20_->is_ready()) {
                sma_20_values_.push_back(sma_20_->get_value());
                if (sma_20_values_.size() > max_data_points_) {
                    sma_20_values_.erase(sma_20_values_.begin());
                }
            }

            if (sma_50_->is_ready()) {
                sma_50_values_.push_back(sma_50_->get_value());
                if (sma_50_values_.size() > max_data_points_) {
                    sma_50_values_.erase(sma_50_values_.begin());
                }
            }

            if (ema_10_->is_ready()) {
                ema_10_values_.push_back(ema_10_->get_value());
                if (ema_10_values_.size() > max_data_points_) {
                    ema_10_values_.erase(ema_10_values_.begin());
                }
            }

            if (ema_20_->is_ready()) {
                ema_20_values_.push_back(ema_20_->get_value());
                if (ema_20_values_.size() > max_data_points_) {
                    ema_20_values_.erase(ema_20_values_.begin());
                }
            }

            if (ema_50_->is_ready()) {
                ema_50_values_.push_back(ema_50_->get_value());
                if (ema_50_values_.size() > max_data_points_) {
                    ema_50_values_.erase(ema_50_values_.begin());
                }
            }

            if (rsi_14_->is_ready()) {
                rsi_values_.push_back(rsi_14_->get_value());
                if (rsi_values_.size() > max_data_points_) {
                    rsi_values_.erase(rsi_values_.begin());
                }
            }

            if (macd_->is_ready()) {
                macd_line_values_.push_back(macd_->get_value());
                signal_line_values_.push_back(macd_->get_signal());
                macd_histogram_values_.push_back(macd_->get_histogram());

                if (macd_line_values_.size() > max_data_points_) {
                    macd_line_values_.erase(macd_line_values_.begin());
                    signal_line_values_.erase(signal_line_values_.begin());
                    macd_histogram_values_.erase(macd_histogram_values_.begin());
                }
            }
        }
    }

    void render_gui() override {
        ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

        if (ImGui::Begin("Technical Indicators", &visible_)) {
            // Indicator selection
            ImGui::Text("Indicator Settings");
            ImGui::Separator();

            // SMA settings
            ImGui::Checkbox("Show SMA 10", &show_sma_10_);
            ImGui::Checkbox("Show SMA 20", &show_sma_20_);
            ImGui::Checkbox("Show SMA 50", &show_sma_50_);
            ImGui::Checkbox("Show EMA 10", &show_ema_10_);
            ImGui::Checkbox("Show EMA 20", &show_ema_20_);
            ImGui::Checkbox("Show EMA 50", &show_ema_50_);
            ImGui::Checkbox("Show RSI", &show_rsi_);
            ImGui::Checkbox("Show MACD", &show_macd_);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // RSI Panel
            if (show_rsi_ && !rsi_values_.empty()) {
                if (ImPlot::BeginPlot("RSI (14)", ImVec2(-1, 150))) {
                    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoLabel);
                    ImPlot::SetupAxis(ImAxis_Y1, "RSI", ImPlotAxisFlags_RangeFit);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);

                    // Plot RSI line
                    ImPlot::PlotLine("RSI", rsi_values_.data(), static_cast<int>(rsi_values_.size()));

                    // Draw horizontal lines for 30, 70 (overbought/oversold levels)
                    ImVec2 p1 = ImPlot::PlotToPixels(0, 30);
                    ImVec2 p2 = ImPlot::PlotToPixels(static_cast<double>(rsi_values_.size() - 1), 30);
                    ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

                    p1 = ImPlot::PlotToPixels(0, 70);
                    p2 = ImPlot::PlotToPixels(static_cast<double>(rsi_values_.size() - 1), 70);
                    ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

                    ImPlot::EndPlot();
                }
            }

            ImGui::Spacing();

            // MACD Panel
            if (show_macd_ && !macd_line_values_.empty() && !signal_line_values_.empty() && !macd_histogram_values_.empty()) {
                if (ImPlot::BeginPlot("MACD (12,26,9)", ImVec2(-1, 150))) {
                    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoLabel);
                    ImPlot::SetupAxis(ImAxis_Y1, "MACD", ImPlotAxisFlags_RangeFit);

                    ImPlot::PlotLine("MACD", macd_line_values_.data(), static_cast<int>(macd_line_values_.size()));
                    ImPlot::PlotLine("Signal", signal_line_values_.data(), static_cast<int>(signal_line_values_.size()));

                    // Plot histogram
                    std::vector<double> macd_pos, macd_neg;
                    macd_pos.resize(macd_histogram_values_.size());
                    macd_neg.resize(macd_histogram_values_.size());

                    for (size_t i = 0; i < macd_histogram_values_.size(); ++i) {
                        if (macd_histogram_values_[i] > 0) {
                            macd_pos[i] = macd_histogram_values_[i];
                            macd_neg[i] = 0;
                        } else {
                            macd_pos[i] = 0;
                            macd_neg[i] = macd_histogram_values_[i];
                        }
                    }

                    ImPlot::PlotBars("MACD Hist", macd_pos.data(), static_cast<int>(macd_line_values_.size()));
                    ImPlot::PlotBars("MACD Hist", macd_neg.data(), static_cast<int>(macd_line_values_.size()));

                    ImPlot::EndPlot();
                }
            }
        }
        ImGui::End();
    }

    void initialize_vulkan_resources(VulkanCore *) override {}

    void clear_data() override {
        price_data_.clear();
        sma_10_values_.clear();
        sma_20_values_.clear();
        sma_50_values_.clear();
        ema_10_values_.clear();
        ema_20_values_.clear();
        ema_50_values_.clear();
        rsi_values_.clear();
        macd_line_values_.clear();
        signal_line_values_.clear();
        macd_histogram_values_.clear();

        initialize_indicators();
    }

    // Data update methods
    void update_price_data(float price) {
        price_data_.push_back(price);
        if (price_data_.size() > max_data_points_) {
            price_data_.erase(price_data_.begin());
        }
    }

    void update_ohlcv_data(const TechnicalIndicators::OHLCV &data) {
        update_price_data(static_cast<float>(data.close));
    }

private:
    // Indicators
    std::unique_ptr<SMAIndicator> sma_10_;
    std::unique_ptr<SMAIndicator> sma_20_;
    std::unique_ptr<SMAIndicator> sma_50_;
    std::unique_ptr<EMAIndicator> ema_10_;
    std::unique_ptr<EMAIndicator> ema_20_;
    std::unique_ptr<EMAIndicator> ema_50_;
    std::unique_ptr<RSIIndicator> rsi_14_;
    std::unique_ptr<MACDIndicator> macd_;

    // Indicator values
    std::vector<float> price_data_;
    std::vector<float> sma_10_values_;
    std::vector<float> sma_20_values_;
    std::vector<float> sma_50_values_;
    std::vector<float> ema_10_values_;
    std::vector<float> ema_20_values_;
    std::vector<float> ema_50_values_;
    std::vector<float> rsi_values_;
    std::vector<float> macd_line_values_;
    std::vector<float> signal_line_values_;
    std::vector<float> macd_histogram_values_;

    // Display settings
    bool show_sma_10_ = true;
    bool show_sma_20_ = true;
    bool show_sma_50_ = false;
    bool show_ema_10_ = false;
    bool show_ema_20_ = false;
    bool show_ema_50_ = false;
    bool show_rsi_ = true;
    bool show_macd_ = true;

    const size_t max_data_points_ = 200;
};

} // namespace BTQuant
