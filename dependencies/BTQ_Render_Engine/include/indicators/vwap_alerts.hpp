#pragma once

#include <vector>
#include <functional>
#include <string>
#include <memory>
#include "../market_data_processor.hpp"

namespace btq {

// Alert types for VWAP notifications
enum class VWAPAlertType {
    PRICE_CROSSES_VWAP,      // Price crosses above/below VWAP
    PRICE_TOUCHES_SD1,       // Price touches 1st standard deviation band
    PRICE_TOUCHES_SD2,       // Price touches 2nd standard deviation band
    PRICE_TOUCHES_SD3,       // Price touches 3rd standard deviation band
    VWAP_DIRECTION_CHANGE,   // VWAP trend direction changes
    VWAP_VOLUME_SPIKE        // Unusual volume activity around VWAP
};

// Alert event structure
struct VWAPAlertEvent {
    VWAPAlertType alert_type;
    uint64_t timestamp;
    double price;
    double vwap_value;
    double sd1_upper;
    double sd1_lower;
    double sd2_upper;
    double sd2_lower;
    double sd3_upper;
    double sd3_lower;
    std::string symbol;
    bool is_bullish;  // Direction of crossing or touch
    
    VWAPAlertEvent(VWAPAlertType type, uint64_t ts, double p, double vwap, 
                   double s1u, double s1l, double s2u, double s2l, 
                   double s3u, double s3l, const std::string& sym, bool bull = false)
        : alert_type(type), timestamp(ts), price(p), vwap_value(vwap),
          sd1_upper(s1u), sd1_lower(s1l), sd2_upper(s2u), sd2_lower(s2l),
          sd3_upper(s3u), sd3_lower(s3l), symbol(sym), is_bullish(bull) {}
};

// Callback function type for alert notifications
using VWAPAlertCallback = std::function<void(const VWAPAlertEvent&)>;

/**
 * @brief VWAP Alerts Manager
 * Monitors price action relative to VWAP and standard deviation bands,
 * triggering notifications when specific conditions are met.
 */
class VWAPAlerts {
public:
    // Constructor
    VWAPAlerts();
    
    // Set the callback function for alert notifications
    void setAlertCallback(VWAPAlertCallback callback);
    
    // Configure alert sensitivity thresholds
    void setPriceCrossThreshold(double threshold);      // Threshold for price crossing VWAP
    void setTouchThreshold(double threshold);           // Threshold for touching SD bands
    void setVolumeSpikeMultiplier(double multiplier);   // Multiplier for volume spike detection
    
    // Check for alerts based on current market data
    void checkAlerts(const BTQuant::RenderEngine::OHLCVCandle& current_bar,
                     const std::vector<double>& vwap_values,
                     const std::vector<double>& sd1_upper,
                     const std::vector<double>& sd1_lower,
                     const std::vector<double>& sd2_upper,
                     const std::vector<double>& sd2_lower,
                     const std::vector<double>& sd3_upper,
                     const std::vector<double>& sd3_lower,
                     const std::string& symbol);
    
    // Check for alerts based on individual values (for real-time updates)
    void checkAlerts(double current_price, 
                     double current_vwap,
                     double current_sd1_upper,
                     double current_sd1_lower,
                     double current_sd2_upper,
                     double current_sd2_lower,
                     double current_sd3_upper,
                     double current_sd3_lower,
                     uint64_t timestamp,
                     const std::string& symbol);

private:
    VWAPAlertCallback alert_callback_;
    double price_cross_threshold_;
    double touch_threshold_;
    double volume_spike_multiplier_;
    
    // Store previous values for comparison
    double previous_price_;
    double previous_vwap_;
    bool previous_price_above_vwap_;
    bool previous_price_above_sd1_upper_;
    bool previous_price_below_sd1_lower_;
    bool previous_price_above_sd2_upper_;
    bool previous_price_below_sd2_lower_;
    bool previous_price_above_sd3_upper_;
    bool previous_price_below_sd3_lower_;
    
    // Initialize previous values flag
    bool initialized_;
    
    // Helper methods
    bool isPriceCrossingVWAP(double current_price, double current_vwap, double previous_price, double previous_vwap) const;
    bool isPriceTouchingSD1(double current_price, double sd1_upper, double sd1_lower) const;
    bool isPriceTouchingSD2(double current_price, double sd2_upper, double sd2_lower) const;
    bool isPriceTouchingSD3(double current_price, double sd3_upper, double sd3_lower) const;
    bool isVWAPDirectionChange(double current_vwap, double previous_vwap) const;
    bool isVolumeSpike(double current_volume, double average_volume) const;
    
    // Generate alert event
    void generateAlert(VWAPAlertType type, uint64_t timestamp, double price, 
                       double vwap, double sd1u, double sd1l, double sd2u, double sd2l, 
                       double sd3u, double sd3l, const std::string& symbol, bool bullish);
};

} // namespace btq