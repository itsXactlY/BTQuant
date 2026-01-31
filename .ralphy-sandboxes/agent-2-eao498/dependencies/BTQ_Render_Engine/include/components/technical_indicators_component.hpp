#pragma once

#include "../vulkan_dashboard_advanced.hpp"
#include "../indicator.hpp"

namespace BTQuant {

struct TechnicalIndicatorsComponent : public UIComponent {
    TechnicalIndicatorsComponent(const glm::vec2 &p, const glm::vec2 &s);
    void update(float dt) override;
    void render_gui() override;
    void initialize_vulkan_resources(VulkanCore *) override;
    void clear_data() override;
    
    void update_price_data(float price);
    void update_ohlcv_data(const TechnicalIndicators::OHLCV &data);

private:
    void initialize_indicators();

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
