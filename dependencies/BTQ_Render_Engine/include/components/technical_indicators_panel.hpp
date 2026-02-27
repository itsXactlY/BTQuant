#pragma once

#include "panel_base.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace BTQuant {

/**
 * TechnicalIndicatorsPanel - Comprehensive technical indicators dashboard
 * 
 * Features (Stub):
 * - RSI, MACD, Bollinger Bands, Stochastic, ATR
 * - Moving averages (SMA, EMA, WMA, VWMA)
 * - Oscillators and momentum indicators
 * - Custom indicator parameters
 * - Multi-timeframe analysis
 */
class TechnicalIndicatorsPanel : public PanelBase {
public:
    explicit TechnicalIndicatorsPanel(const PanelConfig& config);
    ~TechnicalIndicatorsPanel() override = default;

    void initialize() override;
    void render() override;

    // Indicator management
    void add_indicator(const std::string& name, const std::string& description = "");
    void remove_indicator(int index);
    void clear_indicators();

private:
    struct IndicatorConfig {
        std::string name;
        std::string description;
        bool visible;
        ImVec4 color;
    };
    
    std::vector<IndicatorConfig> indicators_;
    int selected_timeframe_ = 0;
    int selected_indicator_ = -1;
    
    void render_indicator_list();
    void render_indicator_settings();
    void render_available_indicators();
};

}  // namespace BTQuant
