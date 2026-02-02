#pragma once

#include <memory>
#include <string>

#include "panel_settings_interface.hpp"

namespace BTQuant {

// Forward declaration using void* to avoid circular dependency
class ChartPanel;

// Settings class for ChartPanel
class ChartPanelSettings : public BasePanelSettings {
private:
    void* chart_panel_ptr_;  // Use void* to avoid circular dependency
    // We'll define the IndicatorConfig in the implementation file
    struct TempIndicatorConfig {
        // SMA configurations - Required: (9,20,50,200)
        bool show_sma_9 = false;
        bool show_sma_10 = false;  // Not required by spec
        bool show_sma_20 = false;
        bool show_sma_50 = false;
        bool show_sma_200 = false;

        // EMA configurations - Required: (9,21,50,200)
        bool show_ema_9 = false;
        bool show_ema_10 = false;  // Not required by spec
        bool show_ema_20 = false;  // Not required by spec
        bool show_ema_21 = false;
        bool show_ema_50 = false;
        bool show_ema_200 = false;

        // RSI configuration
        bool show_rsi = false;

        // MACD configuration
        bool show_macd = false;

        // Bollinger Bands configuration
        bool show_bollinger = false;

        // Stochastic configuration
        bool show_stochastic = false;

        // ATR configuration
        bool show_atr = false;

        bool show_volume_profile = true;
        bool show_fibonacci = false;
        bool show_crosshair_info = true;

        // Fibonacci configuration
        double fib_start_price = 0.0;
        double fib_end_price = 0.0;

        // RSI configuration
        int rsi_period = 14;
        double rsi_overbought = 70.0;
        double rsi_oversold = 30.0;

        // Bollinger Bands configuration
        int bollinger_period = 20;
        double bollinger_std_dev = 2.0;

        // MACD configuration
        int macd_fast_period = 12;
        int macd_slow_period = 26;
        int macd_signal_period = 9;

        // Stochastic configuration
        int stochastic_k_period = 14;
        int stochastic_d_period = 3;
        int stochastic_slow_period = 3;

        // ATR configuration
        int atr_period = 14;
    };

    TempIndicatorConfig temp_config_;  // Temporary config for editing

public:
    explicit ChartPanelSettings(void* chart_panel_ptr);

    void render() override;

    // Apply settings to the chart panel
    void apply_settings();

    // Reset temporary config to current chart config
    void reset_temp_config();

    // Save settings to persistent storage
    void save_settings();

    // Load settings from persistent storage
    void load_settings();

    // Cast the void pointer to ChartPanel (defined in implementation)
    ChartPanel* get_chart_panel() const;
};

} // namespace BTQuant