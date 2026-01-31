#include "components/technical_indicators_component.hpp"

#include <algorithm>
#include <cmath>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

TechnicalIndicatorsComponent::TechnicalIndicatorsComponent(const glm::vec2& p, const glm::vec2& s)
    : UIComponent(p, s) {
  initialize_indicators();
}

void TechnicalIndicatorsComponent::initialize_indicators() {
  sma_10_ = std::make_unique<SMAIndicator>(10);
  sma_20_ = std::make_unique<SMAIndicator>(20);
  sma_50_ = std::make_unique<SMAIndicator>(50);
  ema_10_ = std::make_unique<EMAIndicator>(10);
  ema_20_ = std::make_unique<EMAIndicator>(20);
  ema_50_ = std::make_unique<EMAIndicator>(50);
  rsi_14_ = std::make_unique<RSIIndicator>(14);
  macd_ = std::make_unique<MACDIndicator>(12, 26, 9);
}

void TechnicalIndicatorsComponent::update(float dt) {
  // Update indicators with latest price data
  if (!price_data_.empty()) {
    float latest_price = price_data_.back();

    sma_10_->update(latest_price);
    sma_20_->update(latest_price);
    sma_50_->update(latest_price);
    ema_10_->update(latest_price);
    ema_20_->update(latest_price);
    ema_50_->update(latest_price);
    rsi_14_->update(latest_price);
    macd_->update(latest_price);
  }
}

void TechnicalIndicatorsComponent::render_gui() {
  if (!visible_) return;

  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_FirstUseEver);

  if (ImGui::Begin(title_.c_str())) {
    // Controls
    ImGui::Checkbox("Show SMA 10", &show_sma_10_);
    ImGui::SameLine();
    ImGui::Checkbox("Show SMA 20", &show_sma_20_);
    ImGui::SameLine();
    ImGui::Checkbox("Show SMA 50", &show_sma_50_);
    ImGui::SameLine();
    ImGui::Checkbox("Show RSI", &show_rsi_);
    ImGui::SameLine();
    ImGui::Checkbox("Show MACD", &show_macd_);

    // Plot indicators
    if (ImGui::BeginChild("IndicatorsPlot", ImVec2(0, 0), true)) {
      if (ImPlot::BeginPlot("##Indicators", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Time", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        // Plot price data if available
        if (!price_data_.empty()) {
          ImPlot::PlotLine("Price", price_data_.data(), price_data_.size());

          // Plot SMAs if enabled
          if (show_sma_10_ && sma_10_->is_ready()) {
            ImPlot::PlotLine("SMA 10", sma_10_values_.data(), sma_10_values_.size());
          }
          if (show_sma_20_ && sma_20_->is_ready()) {
            ImPlot::PlotLine("SMA 20", sma_20_values_.data(), sma_20_values_.size());
          }
          if (show_sma_50_ && sma_50_->is_ready()) {
            ImPlot::PlotLine("SMA 50", sma_50_values_.data(), sma_50_values_.size());
          }

          // Plot EMAs if enabled
          if (show_ema_10_ && ema_10_->is_ready()) {
            ImPlot::PlotLine("EMA 10", ema_10_values_.data(), ema_10_values_.size());
          }
          if (show_ema_20_ && ema_20_->is_ready()) {
            ImPlot::PlotLine("EMA 20", ema_20_values_.data(), ema_20_values_.size());
          }
          if (show_ema_50_ && ema_50_->is_ready()) {
            ImPlot::PlotLine("EMA 50", ema_50_values_.data(), ema_50_values_.size());
          }

          // Plot RSI if enabled
          if (show_rsi_ && rsi_14_->is_ready()) {
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.2f);
            ImPlot::PlotShaded("RSI", rsi_values_.data(), rsi_values_.size(), 70, 0);
            ImPlot::PlotLine("RSI", rsi_values_.data(), rsi_values_.size());

            // Add horizontal lines at 30 and 70
            double y30[2] = {30, 30};
            double y70[2] = {70, 70};
            double x_vals[2] = {0, static_cast<double>(rsi_values_.size() - 1)};
            ImPlot::PlotLine("##RSI30", x_vals, y30, 2);
            ImPlot::PlotLine("##RSI70", x_vals, y70, 2);
            ImPlot::PopStyleVar();
          }

          // Plot MACD if enabled
          if (show_macd_ && macd_->is_ready()) {
            ImPlot::PlotLine("MACD", macd_line_values_.data(), macd_line_values_.size());
            ImPlot::PlotLine("Signal", signal_line_values_.data(), signal_line_values_.size());
            // Plot histogram as bars
            ImPlot::PlotBars("Histogram", macd_histogram_values_.data(),
                             macd_histogram_values_.size());
          }
        }

        ImPlot::EndPlot();
      }
    }
    ImGui::EndChild();
  }
  ImGui::End();
}

void TechnicalIndicatorsComponent::initialize_vulkan_resources(VulkanCore* vulkan_core) {
  // No Vulkan resources needed for this component
  (void)vulkan_core;
}

void TechnicalIndicatorsComponent::clear_data() {
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
}

void TechnicalIndicatorsComponent::update_price_data(float price) {
  price_data_.push_back(price);

  // Maintain max data points
  if (price_data_.size() > max_data_points_) {
    price_data_.erase(price_data_.begin());
  }

  // Update indicator values
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

void TechnicalIndicatorsComponent::update_ohlcv_data(const TechnicalIndicators::OHLCV& data) {
  // Update indicators with OHLCV data
  update_price_data(data.close);
}

}  // namespace BTQuant