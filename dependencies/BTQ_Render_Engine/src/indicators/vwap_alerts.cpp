#include "indicators/vwap_alerts.hpp"
#include <cmath>
#include <iostream>

namespace btq {

VWAPAlerts::VWAPAlerts() 
    : price_cross_threshold_(0.0001),  // Small threshold for forex-like precision
      touch_threshold_(0.0001),
      volume_spike_multiplier_(2.0),
      previous_price_(0.0),
      previous_vwap_(0.0),
      previous_price_above_vwap_(false),
      previous_price_above_sd1_upper_(false),
      previous_price_below_sd1_lower_(false),
      previous_price_above_sd2_upper_(false),
      previous_price_below_sd2_lower_(false),
      previous_price_above_sd3_upper_(false),
      previous_price_below_sd3_lower_(false),
      initialized_(false) {
}

void VWAPAlerts::setAlertCallback(VWAPAlertCallback callback) {
    alert_callback_ = std::move(callback);
}

void VWAPAlerts::setPriceCrossThreshold(double threshold) {
    price_cross_threshold_ = std::abs(threshold);
}

void VWAPAlerts::setTouchThreshold(double threshold) {
    touch_threshold_ = std::abs(threshold);
}

void VWAPAlerts::setVolumeSpikeMultiplier(double multiplier) {
    volume_spike_multiplier_ = std::max(1.0, multiplier);
}

void VWAPAlerts::checkAlerts(const BTQuant::RenderEngine::OHLCVCandle& current_bar,
                             const std::vector<double>& vwap_values,
                             const std::vector<double>& sd1_upper,
                             const std::vector<double>& sd1_lower,
                             const std::vector<double>& sd2_upper,
                             const std::vector<double>& sd2_lower,
                             const std::vector<double>& sd3_upper,
                             const std::vector<double>& sd3_lower,
                             const std::string& symbol) {
    
    if (vwap_values.empty() || !alert_callback_) {
        return;
    }
    
    // Get the most recent VWAP and SD values
    double current_vwap = vwap_values.back();
    double current_sd1_upper = sd1_upper.empty() ? 0.0 : sd1_upper.back();
    double current_sd1_lower = sd1_lower.empty() ? 0.0 : sd1_lower.back();
    double current_sd2_upper = sd2_upper.empty() ? 0.0 : sd2_upper.back();
    double current_sd2_lower = sd2_lower.empty() ? 0.0 : sd2_lower.back();
    double current_sd3_upper = sd3_upper.empty() ? 0.0 : sd3_upper.back();
    double current_sd3_lower = sd3_lower.empty() ? 0.0 : sd3_lower.back();
    
    // Use the closing price of the current bar
    double current_price = current_bar.close;
    uint64_t timestamp = current_bar.timestamp;
    
    checkAlerts(current_price, current_vwap, current_sd1_upper, current_sd1_lower,
                current_sd2_upper, current_sd2_lower, current_sd3_upper, current_sd3_lower,
                timestamp, symbol);
}

void VWAPAlerts::checkAlerts(double current_price, 
                             double current_vwap,
                             double current_sd1_upper,
                             double current_sd1_lower,
                             double current_sd2_upper,
                             double current_sd2_lower,
                             double current_sd3_upper,
                             double current_sd3_lower,
                             uint64_t timestamp,
                             const std::string& symbol) {
    
    if (!alert_callback_) {
        return;
    }
    
    // Initialize previous values if this is the first call
    if (!initialized_) {
        previous_price_ = current_price;
        previous_vwap_ = current_vwap;
        previous_price_above_vwap_ = current_price > current_vwap;
        previous_price_above_sd1_upper_ = current_price > current_sd1_upper;
        previous_price_below_sd1_lower_ = current_price < current_sd1_lower;
        previous_price_above_sd2_upper_ = current_price > current_sd2_upper;
        previous_price_below_sd2_lower_ = current_price < current_sd2_lower;
        previous_price_above_sd3_upper_ = current_price > current_sd3_upper;
        previous_price_below_sd3_lower_ = current_price < current_sd3_lower;
        initialized_ = true;
        return;
    }
    
    // Check for price crossing VWAP
    if (isPriceCrossingVWAP(current_price, current_vwap, previous_price_, previous_vwap_)) {
        bool is_bullish = current_price > current_vwap;
        generateAlert(VWAPAlertType::PRICE_CROSSES_VWAP, timestamp, current_price, 
                      current_vwap, current_sd1_upper, current_sd1_lower, 
                      current_sd2_upper, current_sd2_lower, 
                      current_sd3_upper, current_sd3_lower, symbol, is_bullish);
    }
    
    // Check for touching SD1 bands
    if (isPriceTouchingSD1(current_price, current_sd1_upper, current_sd1_lower)) {
        bool is_bullish = current_price >= current_sd1_upper;
        generateAlert(VWAPAlertType::PRICE_TOUCHES_SD1, timestamp, current_price, 
                      current_vwap, current_sd1_upper, current_sd1_lower, 
                      current_sd2_upper, current_sd2_lower, 
                      current_sd3_upper, current_sd3_lower, symbol, is_bullish);
    }
    
    // Check for touching SD2 bands
    if (isPriceTouchingSD2(current_price, current_sd2_upper, current_sd2_lower)) {
        bool is_bullish = current_price >= current_sd2_upper;
        generateAlert(VWAPAlertType::PRICE_TOUCHES_SD2, timestamp, current_price, 
                      current_vwap, current_sd1_upper, current_sd1_lower, 
                      current_sd2_upper, current_sd2_lower, 
                      current_sd3_upper, current_sd3_lower, symbol, is_bullish);
    }
    
    // Check for touching SD3 bands
    if (isPriceTouchingSD3(current_price, current_sd3_upper, current_sd3_lower)) {
        bool is_bullish = current_price >= current_sd3_upper;
        generateAlert(VWAPAlertType::PRICE_TOUCHES_SD3, timestamp, current_price, 
                      current_vwap, current_sd1_upper, current_sd1_lower, 
                      current_sd2_upper, current_sd2_lower, 
                      current_sd3_upper, current_sd3_lower, symbol, is_bullish);
    }
    
    // Check for VWAP direction change
    if (isVWAPDirectionChange(current_vwap, previous_vwap_)) {
        bool is_bullish = current_vwap > previous_vwap_;
        generateAlert(VWAPAlertType::VWAP_DIRECTION_CHANGE, timestamp, current_price, 
                      current_vwap, current_sd1_upper, current_sd1_lower, 
                      current_sd2_upper, current_sd2_lower, 
                      current_sd3_upper, current_sd3_lower, symbol, is_bullish);
    }
    
    // Update previous values for next comparison
    previous_price_ = current_price;
    previous_vwap_ = current_vwap;
    previous_price_above_vwap_ = current_price > current_vwap;
    previous_price_above_sd1_upper_ = current_price > current_sd1_upper;
    previous_price_below_sd1_lower_ = current_price < current_sd1_lower;
    previous_price_above_sd2_upper_ = current_price > current_sd2_upper;
    previous_price_below_sd2_lower_ = current_price < current_sd2_lower;
    previous_price_above_sd3_upper_ = current_price > current_sd3_upper;
    previous_price_below_sd3_lower_ = current_price < current_sd3_lower;
}

bool VWAPAlerts::isPriceCrossingVWAP(double current_price, double current_vwap, 
                                     double previous_price, double previous_vwap) const {
    // Check if price crossed VWAP from above to below or below to above
    bool current_above = current_price > current_vwap;
    bool previous_above = previous_price > previous_vwap;
    
    // Add threshold to avoid noise
    double price_vwap_diff = std::abs(current_price - current_vwap);
    if (price_vwap_diff < price_cross_threshold_) {
        return false; // Too close to call, avoid noise
    }
    
    return current_above != previous_above;
}

bool VWAPAlerts::isPriceTouchingSD1(double current_price, double sd1_upper, double sd1_lower) const {
    // Check if price touched or crossed SD1 bands
    double upper_diff = std::abs(current_price - sd1_upper);
    double lower_diff = std::abs(current_price - sd1_lower);
    
    bool touched_upper = upper_diff <= touch_threshold_ && current_price >= sd1_lower && current_price <= sd1_upper + touch_threshold_;
    bool touched_lower = lower_diff <= touch_threshold_ && current_price >= sd1_lower - touch_threshold_ && current_price <= sd1_upper;
    
    return touched_upper || touched_lower;
}

bool VWAPAlerts::isPriceTouchingSD2(double current_price, double sd2_upper, double sd2_lower) const {
    // Check if price touched or crossed SD2 bands
    double upper_diff = std::abs(current_price - sd2_upper);
    double lower_diff = std::abs(current_price - sd2_lower);
    
    bool touched_upper = upper_diff <= touch_threshold_ && current_price >= sd2_lower && current_price <= sd2_upper + touch_threshold_;
    bool touched_lower = lower_diff <= touch_threshold_ && current_price >= sd2_lower - touch_threshold_ && current_price <= sd2_upper;
    
    return touched_upper || touched_lower;
}

bool VWAPAlerts::isPriceTouchingSD3(double current_price, double sd3_upper, double sd3_lower) const {
    // Check if price touched or crossed SD3 bands
    double upper_diff = std::abs(current_price - sd3_upper);
    double lower_diff = std::abs(current_price - sd3_lower);
    
    bool touched_upper = upper_diff <= touch_threshold_ && current_price >= sd3_lower && current_price <= sd3_upper + touch_threshold_;
    bool touched_lower = lower_diff <= touch_threshold_ && current_price >= sd3_lower - touch_threshold_ && current_price <= sd3_upper;
    
    return touched_upper || touched_lower;
}

bool VWAPAlerts::isVWAPDirectionChange(double current_vwap, double previous_vwap) const {
    // A direction change occurs when the VWAP slope changes
    static double prev_slope = 0.0;
    double current_slope = current_vwap - previous_vwap;
    
    bool direction_changed = false;
    if (prev_slope != 0.0) {
        direction_changed = (prev_slope > 0 && current_slope <= 0) || (prev_slope < 0 && current_slope >= 0);
    }
    
    // Update previous slope for next comparison
    // Note: This is a simplified version - in practice, you might want to store this in the class
    static_cast<void>(current_slope); // Suppress unused warning in this context
    
    return direction_changed;
}

bool VWAPAlerts::isVolumeSpike(double current_volume, double average_volume) const {
    // Check if current volume is significantly higher than average
    if (average_volume <= 0) {
        return current_volume > 0; // If no average, any volume could be considered a spike
    }
    return current_volume > (average_volume * volume_spike_multiplier_);
}

void VWAPAlerts::generateAlert(VWAPAlertType type, uint64_t timestamp, double price, 
                               double vwap, double sd1u, double sd1l, double sd2u, double sd2l, 
                               double sd3u, double sd3l, const std::string& symbol, bool bullish) {
    VWAPAlertEvent event(type, timestamp, price, vwap, sd1u, sd1l, sd2u, sd2l, sd3u, sd3l, symbol, bullish);
    
    // Call the registered callback
    alert_callback_(event);
}

} // namespace btq