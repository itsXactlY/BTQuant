#pragma once

#include <vector>
#include <functional>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <ctime>
#include "../market_data_processor.hpp"

namespace btq {

// Alert types for various technical indicators
enum class IndicatorAlertType {
    PRICE_CROSSES_SMA,         // Price crosses above/below Simple Moving Average
    RSI_OVERSOLD,              // RSI enters oversold territory (typically < 30)
    RSI_OVERBOUGHT,            // RSI enters overbought territory (typically > 70)
    BOLLINGER_BAND_TOUCH_UPPER, // Price touches upper Bollinger Band
    BOLLINGER_BAND_TOUCH_LOWER, // Price touches lower Bollinger Band
    BOLLINGER_BAND_BREAKOUT_UPPER, // Price breaks out above upper Bollinger Band
    BOLLINGER_BAND_BREAKOUT_LOWER, // Price breaks out below lower Bollinger Band
    MACD_CROSS_SIGNAL,         // MACD line crosses signal line
    STOCHASTIC_OVERSOLD,       // Stochastic oscillator enters oversold (typically < 20)
    STOCHASTIC_OVERBOUGHT,     // Stochastic oscillator enters overbought (typically > 80)
    PRICE_CROSSES_EMA          // Price crosses above/below Exponential Moving Average
};

// Alert event structure
struct IndicatorAlertEvent {
    IndicatorAlertType alert_type;
    uint64_t timestamp;
    double price;
    double indicator_value;     // Main indicator value (SMA, RSI, etc.)
    double secondary_value;     // Secondary value (for MACD, stochastic, etc.)
    double upper_band;          // Upper band value (for Bollinger Bands)
    double lower_band;          // Lower band value (for Bollinger Bands)
    std::string symbol;
    bool is_bullish;            // Direction of crossing or condition

    IndicatorAlertEvent(IndicatorAlertType type, uint64_t ts, double p, double ind_val,
                        double sec_val, double upper, double lower, const std::string& sym, bool bull = false)
        : alert_type(type), timestamp(ts), price(p), indicator_value(ind_val),
          secondary_value(sec_val), upper_band(upper), lower_band(lower), 
          symbol(sym), is_bullish(bull) {}
};

// Callback function type for alert notifications
using IndicatorAlertCallback = std::function<void(const IndicatorAlertEvent&)>;

/**
 * @brief Technical Indicator Alerts Manager
 * Monitors price action relative to various technical indicators,
 * triggering notifications when specific conditions are met.
 */
class IndicatorAlerts {
public:
    // Constructor
    IndicatorAlerts();

    // Set the callback function for alert notifications
    void setAlertCallback(IndicatorAlertCallback callback);

    // Configure alert sensitivity thresholds
    void setSMACrossThreshold(double threshold);      // Threshold for price crossing SMA
    void setEMACrossThreshold(double threshold);      // Threshold for price crossing EMA
    void setRSIThresholds(int overbought, int oversold); // RSI overbought/oversold levels
    void setBollingerBandThreshold(double threshold); // Threshold for touching Bollinger Bands
    void setMACDThreshold(double threshold);          // Threshold for MACD signals

    // Logging and notification system
    void enableLogging(bool enable);                 // Enable/disable alert logging
    void setLogFilePath(const std::string& path);    // Set custom log file path
    void logAlert(const IndicatorAlertEvent& event); // Log an alert to file

    // Check for alerts based on current market data and indicators
    void checkAlerts(const BTQuant::RenderEngine::OHLCVCandle& current_bar,
                     const std::vector<double>& sma_values,
                     const std::vector<double>& ema_values,
                     const std::vector<double>& rsi_values,
                     const std::vector<double>& bb_upper_values,
                     const std::vector<double>& bb_lower_values,
                     const std::vector<double>& bb_middle_values,
                     const std::vector<double>& macd_values,
                     const std::vector<double>& macd_signal_values,
                     const std::vector<double>& stochastic_k_values,
                     const std::vector<double>& stochastic_d_values,
                     const std::string& symbol);

    // Check for alerts based on individual values (for real-time updates)
    void checkAlerts(double current_price,
                     double current_sma,
                     double current_ema,
                     double current_rsi,
                     double current_bb_upper,
                     double current_bb_lower,
                     double current_bb_middle,
                     double current_macd,
                     double current_macd_signal,
                     double current_stoch_k,
                     double current_stoch_d,
                     uint64_t timestamp,
                     const std::string& symbol);

private:
    IndicatorAlertCallback alert_callback_;
    double sma_cross_threshold_;
    double ema_cross_threshold_;
    int rsi_overbought_level_;
    int rsi_oversold_level_;
    double bollinger_band_threshold_;
    double macd_threshold_;

    // Store previous values for comparison
    double previous_price_;
    double previous_sma_;
    double previous_ema_;
    double previous_rsi_;
    double previous_bb_upper_;
    double previous_bb_lower_;
    double previous_bb_middle_;
    double previous_macd_;
    double previous_macd_signal_;
    double previous_stoch_k_;
    double previous_stoch_d_;

    // Previous state flags
    bool previous_price_above_sma_;
    bool previous_price_above_ema_;
    bool previous_rsi_overbought_;
    bool previous_rsi_oversold_;
    bool previous_price_above_bb_upper_;
    bool previous_price_below_bb_lower_;
    bool previous_macd_above_signal_;
    bool previous_stoch_k_above_oversold_;
    bool previous_stoch_k_below_overbought_;

    // Initialize previous values flag
    bool initialized_;

    // Logging members
    bool logging_enabled_;
    std::string log_file_path_;
    std::ofstream log_file_;

    // Helper methods
    bool isPriceCrossingSMA(double current_price, double current_sma, double previous_price, double previous_sma) const;
    bool isPriceCrossingEMA(double current_price, double current_ema, double previous_price, double previous_ema) const;
    bool isRSIOversold(double current_rsi) const;
    bool isRSIOverbought(double current_rsi) const;
    bool isPriceTouchingBollingerBands(double current_price, double bb_upper, double bb_lower) const;
    bool isPriceBreakingBollingerBands(double current_price, double bb_upper, double bb_lower) const;
    bool isMACDCrossSignal(double current_macd, double current_macd_signal, double previous_macd, double previous_macd_signal) const;
    bool isStochasticOversold(double current_stoch_k, double current_stoch_d) const;
    bool isStochasticOverbought(double current_stoch_k, double current_stoch_d) const;

    // Generate alert event
    void generateAlert(IndicatorAlertType type, uint64_t timestamp, double price,
                       double indicator_value, double secondary_value,
                       double upper_band, double lower_band, const std::string& symbol, bool bullish);
};

} // namespace btq