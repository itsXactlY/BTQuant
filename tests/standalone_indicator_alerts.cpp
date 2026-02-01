// Standalone indicator alerts implementation
#include "standalone_indicator_alerts.hpp"
#include <cmath>
#include <iostream>
#include <ctime>

namespace btq {

IndicatorAlerts::IndicatorAlerts()
    : sma_cross_threshold_(0.0001),  // Small threshold for forex-like precision
      ema_cross_threshold_(0.0001),  // Small threshold for forex-like precision
      rsi_overbought_level_(70),
      rsi_oversold_level_(30),
      bollinger_band_threshold_(0.0001),
      macd_threshold_(0.00001),
      previous_price_(0.0),
      previous_sma_(0.0),
      previous_ema_(0.0),
      previous_rsi_(0.0),
      previous_bb_upper_(0.0),
      previous_bb_lower_(0.0),
      previous_bb_middle_(0.0),
      previous_macd_(0.0),
      previous_macd_signal_(0.0),
      previous_stoch_k_(0.0),
      previous_stoch_d_(0.0),
      previous_price_above_sma_(false),
      previous_price_above_ema_(false),
      previous_rsi_overbought_(false),
      previous_rsi_oversold_(false),
      previous_price_above_bb_upper_(false),
      previous_price_below_bb_lower_(false),
      previous_macd_above_signal_(false),
      previous_stoch_k_above_oversold_(false),
      previous_stoch_k_below_overbought_(false),
      logging_enabled_(false),
      log_file_path_("indicator_alerts.log"),
      email_notifications_enabled_(false),
      smtp_server_("smtp.gmail.com"),
      smtp_port_(587),
      email_username_(""),
      email_password_(""),
      email_recipient_(""),
      webhook_notifications_enabled_(false),
      webhook_url_(""),
      initialized_(false) {
}

void IndicatorAlerts::setAlertCallback(IndicatorAlertCallback callback) {
    alert_callback_ = std::move(callback);
}

void IndicatorAlerts::setSMACrossThreshold(double threshold) {
    sma_cross_threshold_ = std::abs(threshold);
}

void IndicatorAlerts::setEMACrossThreshold(double threshold) {
    ema_cross_threshold_ = std::abs(threshold);
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
                                  const std::vector<double>& ema_values,
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
    double current_ema = ema_values.empty() ? 0.0 : ema_values.back();
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

    checkAlerts(current_price, current_sma, current_ema, current_rsi, current_bb_upper, current_bb_lower,
                current_bb_middle, current_macd, current_macd_signal,
                current_stoch_k, current_stoch_d, timestamp, symbol);
}

void IndicatorAlerts::checkAlerts(double current_price,
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
                                  const std::string& symbol) {

    if (!alert_callback_) {
        return;
    }

    // Initialize previous values if this is the first call
    if (!initialized_) {
        previous_price_ = current_price;
        previous_sma_ = current_sma;
        previous_ema_ = current_ema;
        previous_rsi_ = current_rsi;
        previous_bb_upper_ = current_bb_upper;
        previous_bb_lower_ = current_bb_lower;
        previous_bb_middle_ = current_bb_middle;
        previous_macd_ = current_macd;
        previous_macd_signal_ = current_macd_signal;
        previous_stoch_k_ = current_stoch_k;
        previous_stoch_d_ = current_stoch_d;

        previous_price_above_sma_ = current_price > current_sma;
        previous_price_above_ema_ = current_price > current_ema;
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

    // Check for price crossing EMA
    if (isPriceCrossingEMA(current_price, current_ema, previous_price_, previous_ema_)) {
        bool is_bullish = current_price > current_ema;
        generateAlert(IndicatorAlertType::PRICE_CROSSES_EMA, timestamp, current_price,
                      current_ema, 0.0, 0.0, 0.0, symbol, is_bullish);
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
    previous_ema_ = current_ema;
    previous_rsi_ = current_rsi;
    previous_bb_upper_ = current_bb_upper;
    previous_bb_lower_ = current_bb_lower;
    previous_bb_middle_ = current_bb_middle;
    previous_macd_ = current_macd;
    previous_macd_signal_ = current_macd_signal;
    previous_stoch_k_ = current_stoch_k;
    previous_stoch_d_ = current_stoch_d;

    previous_price_above_sma_ = current_price > current_sma;
    previous_price_above_ema_ = current_price > current_ema;
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

    // Check if there was an actual crossing (different sides of the SMA)
    bool crossed = current_above != previous_above;
    
    // Additional validation: ensure the crossing is meaningful by checking
    // that the current price is sufficiently away from the SMA
    double price_sma_diff = std::abs(current_price - current_sma);
    if (price_sma_diff < sma_cross_threshold_) {
        return false; // Too close to call, avoid noise
    }

    // Also validate that the previous price was on the opposite side of the SMA
    // and sufficiently away from it to confirm a genuine cross
    double prev_price_sma_diff = std::abs(previous_price - previous_sma);
    if (prev_price_sma_diff < sma_cross_threshold_) {
        return false; // Previous position was too close to SMA to confirm a cross
    }

    return crossed;
}

bool IndicatorAlerts::isPriceCrossingEMA(double current_price, double current_ema,
                                         double previous_price, double previous_ema) const {
    // Check if price crossed EMA from above to below or below to above
    bool current_above = current_price > current_ema;
    bool previous_above = previous_price > previous_ema;

    // Check if there was an actual crossing (different sides of the EMA)
    bool crossed = current_above != previous_above;
    
    // Additional validation: ensure the crossing is meaningful by checking
    // that the current price is sufficiently away from the EMA
    double price_ema_diff = std::abs(current_price - current_ema);
    if (price_ema_diff < ema_cross_threshold_) {
        return false; // Too close to call, avoid noise
    }

    // Also validate that the previous price was on the opposite side of the EMA
    // and sufficiently away from it to confirm a genuine cross
    double prev_price_ema_diff = std::abs(previous_price - previous_ema);
    if (prev_price_ema_diff < ema_cross_threshold_) {
        return false; // Previous position was too close to EMA to confirm a cross
    }

    return crossed;
}

bool IndicatorAlerts::isRSIOversold(double current_rsi) const {
    // Check if RSI entered oversold territory (typically < 30)
    bool currently_oversold = current_rsi < rsi_oversold_level_;
    bool previously_not_oversold = previous_rsi_ >= rsi_oversold_level_;

    // Only trigger if we just entered the oversold zone
    return currently_oversold && previously_not_oversold;
}

bool IndicatorAlerts::isRSIOverbought(double current_rsi) const {
    // Check if RSI entered overbought territory (typically > 70)
    bool currently_overbought = current_rsi > rsi_overbought_level_;
    bool previously_not_overbought = previous_rsi_ <= rsi_overbought_level_;

    // Only trigger if we just entered the overbought zone
    return currently_overbought && previously_not_overbought;
}

bool IndicatorAlerts::isPriceTouchingBollingerBands(double current_price, double bb_upper, double bb_lower) const {
    // Check if price touched Bollinger Bands with proper threshold
    // A touch occurs when price comes within threshold distance of either band
    double upper_diff = std::abs(current_price - bb_upper);
    double lower_diff = std::abs(current_price - bb_lower);

    bool touched_upper = upper_diff <= bollinger_band_threshold_ && current_price <= bb_upper;
    bool touched_lower = lower_diff <= bollinger_band_threshold_ && current_price >= bb_lower;

    // Also check for actual crossing of bands (price moved from inside to outside)
    bool crossed_upper = current_price > bb_upper && previous_price_ <= bb_upper && 
                         std::abs(current_price - bb_upper) <= bollinger_band_threshold_;
    bool crossed_lower = current_price < bb_lower && previous_price_ >= bb_lower && 
                         std::abs(current_price - bb_lower) <= bollinger_band_threshold_;

    return touched_upper || touched_lower || crossed_upper || crossed_lower;
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
    if (alert_callback_) {
        alert_callback_(event);
    }

    // Log the alert if logging is enabled
    if (logging_enabled_) {
        logAlert(event);
    }
    
    // Send email notification if enabled
    if (email_notifications_enabled_) {
        sendEmailNotification(event);
    }
    
    // Send webhook notification if enabled
    if (webhook_notifications_enabled_) {
        sendWebhookNotification(event);
    }
}

void IndicatorAlerts::enableLogging(bool enable) {
    logging_enabled_ = enable;
    if (enable && !log_file_.is_open()) {
        log_file_.open(log_file_path_, std::ios::app);
    } else if (!enable && log_file_.is_open()) {
        log_file_.close();
    }
}

void IndicatorAlerts::setLogFilePath(const std::string& path) {
    log_file_path_ = path;
    if (logging_enabled_ && log_file_.is_open()) {
        log_file_.close();
        log_file_.open(log_file_path_, std::ios::app);
    }
}

void IndicatorAlerts::logAlert(const IndicatorAlertEvent& event) {
    if (!logging_enabled_) {
        return;
    }

    if (!log_file_.is_open()) {
        log_file_.open(log_file_path_, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << log_file_path_ << std::endl;
            return;
        }
    }

    // Convert timestamp to readable format
    time_t timestamp_seconds = static_cast<time_t>(event.timestamp / 1000000); // Convert microseconds to seconds
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&timestamp_seconds));

    std::string alert_type_str;
    switch (event.alert_type) {
        case IndicatorAlertType::PRICE_CROSSES_SMA:
            alert_type_str = "PRICE_CROSSES_SMA";
            break;
        case IndicatorAlertType::PRICE_CROSSES_EMA:
            alert_type_str = "PRICE_CROSSES_EMA";
            break;
        case IndicatorAlertType::RSI_OVERSOLD:
            alert_type_str = "RSI_OVERSOLD";
            break;
        case IndicatorAlertType::RSI_OVERBOUGHT:
            alert_type_str = "RSI_OVERBOUGHT";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_UPPER:
            alert_type_str = "BOLLINGER_BAND_TOUCH_UPPER";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_LOWER:
            alert_type_str = "BOLLINGER_BAND_TOUCH_LOWER";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_UPPER:
            alert_type_str = "BOLLINGER_BAND_BREAKOUT_UPPER";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_LOWER:
            alert_type_str = "BOLLINGER_BAND_BREAKOUT_LOWER";
            break;
        case IndicatorAlertType::MACD_CROSS_SIGNAL:
            alert_type_str = "MACD_CROSS_SIGNAL";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERSOLD:
            alert_type_str = "STOCHASTIC_OVERSOLD";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERBOUGHT:
            alert_type_str = "STOCHASTIC_OVERBOUGHT";
            break;
        default:
            alert_type_str = "UNKNOWN";
            break;
    }

    log_file_ << "[" << buffer << "." << (event.timestamp % 1000000)/1000 << "] "
              << "Symbol: " << event.symbol << ", "
              << "Alert: " << alert_type_str << ", "
              << "Price: " << event.price << ", "
              << "Value: " << event.indicator_value << ", "
              << "Direction: " << (event.is_bullish ? "Bullish" : "Bearish") << std::endl;

    log_file_.flush(); // Ensure the log is written immediately
}

void IndicatorAlerts::enableEmailNotifications(bool enable) {
    email_notifications_enabled_ = enable;
}

void IndicatorAlerts::setEmailConfig(const std::string& smtp_server, int port,
                                   const std::string& username, const std::string& password,
                                   const std::string& recipient) {
    smtp_server_ = smtp_server;
    smtp_port_ = port;
    email_username_ = username;
    email_password_ = password;
    email_recipient_ = recipient;
}

void IndicatorAlerts::sendEmailNotification(const IndicatorAlertEvent& event) {
    if (!email_notifications_enabled_) {
        return;
    }

    // In a real implementation, this would connect to an SMTP server and send an email
    // For now, we'll just log the attempt
    std::cout << "EMAIL NOTIFICATION WOULD BE SENT: ";
    
    std::string alert_type_str;
    switch (event.alert_type) {
        case IndicatorAlertType::PRICE_CROSSES_SMA:
            alert_type_str = "Price crosses SMA";
            break;
        case IndicatorAlertType::PRICE_CROSSES_EMA:
            alert_type_str = "Price crosses EMA";
            break;
        case IndicatorAlertType::RSI_OVERSOLD:
            alert_type_str = "RSI oversold";
            break;
        case IndicatorAlertType::RSI_OVERBOUGHT:
            alert_type_str = "RSI overbought";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_UPPER:
            alert_type_str = "Price touches upper Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_LOWER:
            alert_type_str = "Price touches lower Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_UPPER:
            alert_type_str = "Price breaks out above upper Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_LOWER:
            alert_type_str = "Price breaks out below lower Bollinger Band";
            break;
        case IndicatorAlertType::MACD_CROSS_SIGNAL:
            alert_type_str = "MACD crosses signal line";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERSOLD:
            alert_type_str = "Stochastic oversold";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERBOUGHT:
            alert_type_str = "Stochastic overbought";
            break;
        default:
            alert_type_str = "Unknown alert";
            break;
    }

    std::cout << "To: " << email_recipient_ 
              << " | Subject: Indicator Alert - " << alert_type_str
              << " | Symbol: " << event.symbol
              << " | Price: " << event.price
              << " | Value: " << event.indicator_value << std::endl;
}

void IndicatorAlerts::enableWebhookNotifications(bool enable) {
    webhook_notifications_enabled_ = enable;
}

void IndicatorAlerts::setWebhookUrl(const std::string& url) {
    webhook_url_ = url;
}

void IndicatorAlerts::sendWebhookNotification(const IndicatorAlertEvent& event) {
    if (!webhook_notifications_enabled_ || webhook_url_.empty()) {
        return;
    }

    // In a real implementation, this would make an HTTP POST request to the webhook URL
    // For now, we'll just log the attempt
    std::cout << "WEBHOOK NOTIFICATION WOULD BE SENT TO: " << webhook_url_ << std::endl;
    
    std::string alert_type_str;
    switch (event.alert_type) {
        case IndicatorAlertType::PRICE_CROSSES_SMA:
            alert_type_str = "Price crosses SMA";
            break;
        case IndicatorAlertType::PRICE_CROSSES_EMA:
            alert_type_str = "Price crosses EMA";
            break;
        case IndicatorAlertType::RSI_OVERSOLD:
            alert_type_str = "RSI oversold";
            break;
        case IndicatorAlertType::RSI_OVERBOUGHT:
            alert_type_str = "RSI overbought";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_UPPER:
            alert_type_str = "Price touches upper Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_TOUCH_LOWER:
            alert_type_str = "Price touches lower Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_UPPER:
            alert_type_str = "Price breaks out above upper Bollinger Band";
            break;
        case IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_LOWER:
            alert_type_str = "Price breaks out below lower Bollinger Band";
            break;
        case IndicatorAlertType::MACD_CROSS_SIGNAL:
            alert_type_str = "MACD crosses signal line";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERSOLD:
            alert_type_str = "Stochastic oversold";
            break;
        case IndicatorAlertType::STOCHASTIC_OVERBOUGHT:
            alert_type_str = "Stochastic overbought";
            break;
        default:
            alert_type_str = "Unknown alert";
            break;
    }

    std::cout << "Alert Type: " << alert_type_str
              << " | Symbol: " << event.symbol
              << " | Price: " << event.price
              << " | Value: " << event.indicator_value
              << " | Bullish: " << (event.is_bullish ? "Yes" : "No") << std::endl;
}

} // namespace btq