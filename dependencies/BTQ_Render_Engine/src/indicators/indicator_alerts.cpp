#include "indicators/indicator_alerts.hpp"
#include <cmath>
#include <iostream>

namespace btq {

IndicatorAlerts::IndicatorAlerts()
    : sma_cross_threshold_(0.0001),  // Small threshold for forex-like precision
      rsi_overbought_level_(70),
      rsi_oversold_level_(30),
      bollinger_band_threshold_(0.0001),
      macd_threshold_(0.00001),
      previous_price_(0.0),
      previous_sma_(0.0),
      previous_rsi_(0.0),
      previous_bb_upper_(0.0),
      previous_bb_lower_(0.0),
      previous_bb_middle_(0.0),
      previous_macd_(0.0),
      previous_macd_signal_(0.0),
      previous_stoch_k_(0.0),
      previous_stoch_d_(0.0),
      previous_price_above_sma_(false),
      previous_rsi_overbought_(false),
      previous_rsi_oversold_(false),
      previous_price_above_bb_upper_(false),
      previous_price_below_bb_lower_(false),
      previous_macd_above_signal_(false),
      previous_stoch_k_above_oversold_(false),
      previous_stoch_k_below_overbought_(false),
      initialized_(false) {
}

void IndicatorAlerts::setAlertCallback(IndicatorAlertCallback callback) {
    alert_callback_ = std::move(callback);
}

void IndicatorAlerts::setSMACrossThreshold(double threshold) {
    sma_cross_threshold_ = std::abs(threshold);
}

void IndicatorAlerts::setRSIThresholds(int overbought, int oversold) {
    rsi_overbought_level_ = std::min(100, std::max(0, overbought));
    rsi_oversold_level_ = std::min(100, std::max(0, oversold));
    
    // Ensure oversold level is less than overbought level
    if (rsi_oversold_level_ >= rsi_overbought_level_) {
        rsi_oversold_level_ = rsi_overbought_level_ - 10;
    }
}

void IndicatorAlerts::setBollingerBandThreshold(double threshold) {
    bollinger_band_threshold_ = std::abs(threshold);
}

void IndicatorAlerts::setMACDThreshold(double threshold) {
    macd_threshold_ = std::abs(threshold);
}

void IndicatorAlerts::checkAlerts(const BTQuant::RenderEngine::OHLCVCandle& current_bar,
                                  const std::vector<double>& sma_values,
                                  const std::vector<double>& rsi_values,
                                  const std::vector<double>& bb_upper_values,
                                  const std::vector<double>& bb_lower_values,
                                  const std::vector<double>& bb_middle_values,
                                  const std::vector<double>& macd_values,
                                  const std::vector<double>& macd_signal_values,
                                  const std::vector<double>& stochastic_k_values,
                                  const std::vector<double>& stochastic_d_values,
                                  const std::string& symbol) {

    if (sma_values.empty() || rsi_values.empty() || !alert_callback_) {
        return;
    }

    // Get the most recent indicator values
    double current_sma = sma_values.back();
    double current_rsi = rsi_values.empty() ? 0.0 : rsi_values.back();
    double current_bb_upper = bb_upper_values.empty() ? 0.0 : bb_upper_values.back();
    double current_bb_lower = bb_lower_values.empty() ? 0.0 : bb_lower_values.back();
    double current_bb_middle = bb_middle_values.empty() ? 0.0 : bb_middle_values.back();
    double current_macd = macd_values.empty() ? 0.0 : macd_values.back();
    double current_macd_signal = macd_signal_values.empty() ? 0.0 : macd_signal_values.back();
    double current_stoch_k = stochastic_k_values.empty() ? 0.0 : stochastic_k_values.back();
    double current_stoch_d = stochastic_d_values.empty() ? 0.0 : stochastic_d_values.back();

    // Use the closing price of the current bar
    double current_price = current_bar.close;
    uint64_t timestamp = current_bar.timestamp;

    checkAlerts(current_price, current_sma, current_rsi, current_bb_upper, current_bb_lower,
                current_bb_middle, current_macd, current_macd_signal,
                current_stoch_k, current_stoch_d, timestamp, symbol);
}

void IndicatorAlerts::checkAlerts(double current_price,
                                  double current_sma,
                                  double current_rsi,
                                  double current_bb_upper,
                                  double current_bb_lower,
                                  double current_bb_middle,
                                  double current_macd,
                                  double current_macd_signal,
                                  double current_stoch_k,
                                  double current_stoch_d,
                                  uint64_t timestamp,
                                  const std::string& symbol) {

    if (!alert_callback_) {
        return;
    }

    // Initialize previous values if this is the first call
    if (!initialized_) {
        previous_price_ = current_price;
        previous_sma_ = current_sma;
        previous_rsi_ = current_rsi;
        previous_bb_upper_ = current_bb_upper;
        previous_bb_lower_ = current_bb_lower;
        previous_bb_middle_ = current_bb_middle;
        previous_macd_ = current_macd;
        previous_macd_signal_ = current_macd_signal;
        previous_stoch_k_ = current_stoch_k;
        previous_stoch_d_ = current_stoch_d;
        
        previous_price_above_sma_ = current_price > current_sma;
        previous_rsi_overbought_ = current_rsi > rsi_overbought_level_;
        previous_rsi_oversold_ = current_rsi < rsi_oversold_level_;
        previous_price_above_bb_upper_ = current_price > current_bb_upper;
        previous_price_below_bb_lower_ = current_price < current_bb_lower;
        previous_macd_above_signal_ = current_macd > current_macd_signal;
        previous_stoch_k_above_oversold_ = current_stoch_k > rsi_oversold_level_;
        previous_stoch_k_below_overbought_ = current_stoch_k < rsi_overbought_level_;
        
        initialized_ = true;
        return;
    }

    // Check for price crossing SMA
    if (isPriceCrossingSMA(current_price, current_sma, previous_price_, previous_sma_)) {
        bool is_bullish = current_price > current_sma;
        generateAlert(IndicatorAlertType::PRICE_CROSSES_SMA, timestamp, current_price,
                      current_sma, 0.0, 0.0, 0.0, symbol, is_bullish);
    }

    // Check for RSI oversold
    if (isRSIOversold(current_rsi)) {
        generateAlert(IndicatorAlertType::RSI_OVERSOLD, timestamp, current_price,
                      current_rsi, 0.0, 0.0, 0.0, symbol, true);
    }

    // Check for RSI overbought
    if (isRSIOverbought(current_rsi)) {
        generateAlert(IndicatorAlertType::RSI_OVERBOUGHT, timestamp, current_price,
                      current_rsi, 0.0, 0.0, 0.0, symbol, false);
    }

    // Check for Bollinger Band touches
    if (isPriceTouchingBollingerBands(current_price, current_bb_upper, current_bb_lower)) {
        bool is_bullish = current_price >= current_bb_upper;
        generateAlert(is_bullish ? IndicatorAlertType::BOLLINGER_BAND_TOUCH_UPPER : 
                                 IndicatorAlertType::BOLLINGER_BAND_TOUCH_LOWER,
                      timestamp, current_price, current_bb_middle, 0.0, 
                      current_bb_upper, current_bb_lower, symbol, is_bullish);
    }

    // Check for Bollinger Band breakouts
    if (isPriceBreakingBollingerBands(current_price, current_bb_upper, current_bb_lower)) {
        bool is_bullish = current_price > current_bb_upper;
        generateAlert(is_bullish ? IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_UPPER : 
                                 IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_LOWER,
                      timestamp, current_price, current_bb_middle, 0.0, 
                      current_bb_upper, current_bb_lower, symbol, is_bullish);
    }

    // Check for MACD crossing signal line
    if (isMACDCrossSignal(current_macd, current_macd_signal, previous_macd_, previous_macd_signal_)) {
        bool is_bullish = current_macd > current_macd_signal;
        generateAlert(IndicatorAlertType::MACD_CROSS_SIGNAL, timestamp, current_price,
                      current_macd, current_macd_signal, 0.0, 0.0, symbol, is_bullish);
    }

    // Check for Stochastic oversold
    if (isStochasticOversold(current_stoch_k, current_stoch_d)) {
        generateAlert(IndicatorAlertType::STOCHASTIC_OVERSOLD, timestamp, current_price,
                      current_stoch_k, current_stoch_d, 0.0, 0.0, symbol, true);
    }

    // Check for Stochastic overbought
    if (isStochasticOverbought(current_stoch_k, current_stoch_d)) {
        generateAlert(IndicatorAlertType::STOCHASTIC_OVERBOUGHT, timestamp, current_price,
                      current_stoch_k, current_stoch_d, 0.0, 0.0, symbol, false);
    }

    // Update previous values for next comparison
    previous_price_ = current_price;
    previous_sma_ = current_sma;
    previous_rsi_ = current_rsi;
    previous_bb_upper_ = current_bb_upper;
    previous_bb_lower_ = current_bb_lower;
    previous_bb_middle_ = current_bb_middle;
    previous_macd_ = current_macd;
    previous_macd_signal_ = current_macd_signal;
    previous_stoch_k_ = current_stoch_k;
    previous_stoch_d_ = current_stoch_d;
    
    previous_price_above_sma_ = current_price > current_sma;
    previous_rsi_overbought_ = current_rsi > rsi_overbought_level_;
    previous_rsi_oversold_ = current_rsi < rsi_oversold_level_;
    previous_price_above_bb_upper_ = current_price > current_bb_upper;
    previous_price_below_bb_lower_ = current_price < current_bb_lower;
    previous_macd_above_signal_ = current_macd > current_macd_signal;
    previous_stoch_k_above_oversold_ = current_stoch_k > rsi_oversold_level_;
    previous_stoch_k_below_overbought_ = current_stoch_k < rsi_overbought_level_;
}

bool IndicatorAlerts::isPriceCrossingSMA(double current_price, double current_sma,
                                         double previous_price, double previous_sma) const {
    // Check if price crossed SMA from above to below or below to above
    bool current_above = current_price > current_sma;
    bool previous_above = previous_price > previous_sma;

    // Add threshold to avoid noise
    double price_sma_diff = std::abs(current_price - current_sma);
    if (price_sma_diff < sma_cross_threshold_) {
        return false; // Too close to call, avoid noise
    }

    return current_above != previous_above;
}

bool IndicatorAlerts::isRSIOversold(double current_rsi) const {
    // Check if RSI entered oversold territory (typically < 30)
    bool currently_oversold = current_rsi < rsi_oversold_level_;
    bool previously_not_oversold = previous_rsi_ >= rsi_oversold_level_;
    
    return currently_oversold && previously_not_oversold;
}

bool IndicatorAlerts::isRSIOverbought(double current_rsi) const {
    // Check if RSI entered overbought territory (typically > 70)
    bool currently_overbought = current_rsi > rsi_overbought_level_;
    bool previously_not_overbought = previous_rsi_ <= rsi_overbought_level_;
    
    return currently_overbought && previously_not_overbought;
}

bool IndicatorAlerts::isPriceTouchingBollingerBands(double current_price, double bb_upper, double bb_lower) const {
    // Check if price touched or crossed Bollinger Bands
    double upper_diff = std::abs(current_price - bb_upper);
    double lower_diff = std::abs(current_price - bb_lower);

    bool touched_upper = upper_diff <= bollinger_band_threshold_ && 
                         current_price >= bb_lower && current_price <= bb_upper + bollinger_band_threshold_;
    bool touched_lower = lower_diff <= bollinger_band_threshold_ && 
                         current_price >= bb_lower - bollinger_band_threshold_ && current_price <= bb_upper;

    return touched_upper || touched_lower;
}

bool IndicatorAlerts::isPriceBreakingBollingerBands(double current_price, double bb_upper, double bb_lower) const {
    // Check if price broke out above or below Bollinger Bands
    bool broke_upper = current_price > bb_upper && previous_price_ <= bb_upper;
    bool broke_lower = current_price < bb_lower && previous_price_ >= bb_lower;

    return broke_upper || broke_lower;
}

bool IndicatorAlerts::isMACDCrossSignal(double current_macd, double current_macd_signal,
                                       double previous_macd, double previous_macd_signal) const {
    // Check if MACD crossed the signal line
    bool current_above = current_macd > current_macd_signal;
    bool previous_above = previous_macd > previous_macd_signal;

    // Add threshold to avoid noise
    double macd_signal_diff = std::abs(current_macd - current_macd_signal);
    if (macd_signal_diff < macd_threshold_) {
        return false; // Too close to call, avoid noise
    }

    return current_above != previous_above;
}

bool IndicatorAlerts::isStochasticOversold(double current_stoch_k, double current_stoch_d) const {
    // Check if Stochastic K line entered oversold territory (typically < 20)
    // Also consider D line for confirmation
    bool k_oversold = current_stoch_k < 20;
    bool d_oversold = current_stoch_d < 20;
    
    // Check if we just entered oversold territory
    bool k_just_entered = k_oversold && previous_stoch_k_ >= 20;
    bool d_just_entered = d_oversold && previous_stoch_d_ >= 20;
    
    return k_just_entered || d_just_entered;
}

bool IndicatorAlerts::isStochasticOverbought(double current_stoch_k, double current_stoch_d) const {
    // Check if Stochastic K line entered overbought territory (typically > 80)
    // Also consider D line for confirmation
    bool k_overbought = current_stoch_k > 80;
    bool d_overbought = current_stoch_d > 80;
    
    // Check if we just entered overbought territory
    bool k_just_entered = k_overbought && previous_stoch_k_ <= 80;
    bool d_just_entered = d_overbought && previous_stoch_d_ <= 80;
    
    return k_just_entered || d_just_entered;
}

void IndicatorAlerts::generateAlert(IndicatorAlertType type, uint64_t timestamp, double price,
                                    double indicator_value, double secondary_value,
                                    double upper_band, double lower_band, const std::string& symbol, bool bullish) {
    IndicatorAlertEvent event(type, timestamp, price, indicator_value, secondary_value, 
                             upper_band, lower_band, symbol, bullish);

    // Call the registered callback
    alert_callback_(event);
}

} // namespace btq