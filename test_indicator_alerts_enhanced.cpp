#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// Define simplified versions of the required structures
namespace BTQuant {
namespace RenderEngine {
    struct OHLCVCandle {
        uint64_t timestamp;
        double open;
        double high;
        double low;
        double close;
        double volume;
        uint64_t trade_count;
    };
}
}

// Include the indicator alerts header
#include "dependencies/BTQ_Render_Engine/include/indicators/indicator_alerts.hpp"

int main() {
    std::cout << "Testing Indicator Alerts System..." << std::endl;

    btq::IndicatorAlerts alerts;

    // Enable logging
    alerts.enableLogging(true);
    alerts.setLogFilePath("indicator_alerts_test.log");

    // Enable email notifications (simulated)
    alerts.enableEmailNotifications(true);
    alerts.setEmailConfig("smtp.example.com", 587, "user@example.com", "password", "recipient@example.com");

    // Enable webhook notifications (simulated)
    alerts.enableWebhookNotifications(true);
    alerts.setWebhookUrl("https://hooks.example.com/webhook");

    // Set up a callback to handle alerts
    alerts.setAlertCallback([](const btq::IndicatorAlertEvent& event) {
        std::cout << "ALERT TRIGGERED: ";

        switch(event.alert_type) {
            case btq::IndicatorAlertType::PRICE_CROSSES_SMA:
                std::cout << "Price crosses SMA";
                break;
            case btq::IndicatorAlertType::PRICE_CROSSES_EMA:
                std::cout << "Price crosses EMA";
                break;
            case btq::IndicatorAlertType::RSI_OVERSOLD:
                std::cout << "RSI oversold";
                break;
            case btq::IndicatorAlertType::RSI_OVERBOUGHT:
                std::cout << "RSI overbought";
                break;
            case btq::IndicatorAlertType::BOLLINGER_BAND_TOUCH_UPPER:
                std::cout << "Price touches upper Bollinger Band";
                break;
            case btq::IndicatorAlertType::BOLLINGER_BAND_TOUCH_LOWER:
                std::cout << "Price touches lower Bollinger Band";
                break;
            case btq::IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_UPPER:
                std::cout << "Price breaks out above upper Bollinger Band";
                break;
            case btq::IndicatorAlertType::BOLLINGER_BAND_BREAKOUT_LOWER:
                std::cout << "Price breaks out below lower Bollinger Band";
                break;
            case btq::IndicatorAlertType::MACD_CROSS_SIGNAL:
                std::cout << "MACD crosses signal line";
                break;
            case btq::IndicatorAlertType::STOCHASTIC_OVERSOLD:
                std::cout << "Stochastic oversold";
                break;
            case btq::IndicatorAlertType::STOCHASTIC_OVERBOUGHT:
                std::cout << "Stochastic overbought";
                break;
            default:
                std::cout << "Unknown alert";
                break;
        }

        std::cout << " | Symbol: " << event.symbol
                  << " | Price: " << event.price
                  << " | Value: " << event.indicator_value
                  << " | Bullish: " << (event.is_bullish ? "Yes" : "No") << std::endl;
    });

    // Test data
    std::string symbol = "TEST";
    BTQuant::RenderEngine::OHLCVCandle candle;
    candle.timestamp = 1672531200000000;
    candle.open = 100.0;
    candle.high = 101.0;
    candle.low = 99.0;
    candle.close = 100.5;
    candle.volume = 1000.0;
    candle.trade_count = 50;

    // Configure thresholds
    alerts.setSMACrossThreshold(0.001);
    alerts.setEMACrossThreshold(0.001);
    alerts.setRSIThresholds(70, 30);
    alerts.setBollingerBandThreshold(0.01);
    alerts.setMACDThreshold(0.0001);

    std::cout << "\nSimulating indicator alerts..." << std::endl;

    // Prepare indicator vectors
    std::vector<double> sma_values = {100.0};  // Previous SMA
    std::vector<double> ema_values = {100.0};  // Previous EMA
    std::vector<double> rsi_values = {50.0};   // Previous RSI
    std::vector<double> bb_upper_values = {101.0};  // Upper Bollinger Band
    std::vector<double> bb_lower_values = {99.0};   // Lower Bollinger Band
    std::vector<double> bb_middle_values = {100.0}; // Middle Bollinger Band
    std::vector<double> macd_values = {0.5};        // MACD value
    std::vector<double> macd_signal_values = {0.4}; // MACD signal
    std::vector<double> stochastic_k_values = {80.0}; // Stochastic K
    std::vector<double> stochastic_d_values = {85.0}; // Stochastic D

    // Test 1: Simulate price crossing SMA from below (bullish)
    std::cout << "\nTest 1: Price crossing SMA from below (bullish)..." << std::endl;
    candle.close = 100.5;  // Price above SMA
    sma_values[0] = 100.0; // SMA value
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Test 2: Simulate price crossing SMA from above (bearish)
    std::cout << "\nTest 2: Price crossing SMA from above (bearish)..." << std::endl;
    candle.close = 99.5;   // Price below SMA
    sma_values[0] = 100.0; // SMA value
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Test 3: Simulate RSI oversold
    std::cout << "\nTest 3: RSI oversold..." << std::endl;
    candle.close = 100.0;
    rsi_values[0] = 25.0;  // RSI below 30
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Reset RSI for next test
    rsi_values[0] = 50.0;

    // Test 4: Simulate RSI overbought
    std::cout << "\nTest 4: RSI overbought..." << std::endl;
    rsi_values[0] = 80.0;  // RSI above 70
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Reset RSI for next test
    rsi_values[0] = 50.0;

    // Test 5: Simulate Bollinger Band touch upper
    std::cout << "\nTest 5: Bollinger Band touch upper..." << std::endl;
    candle.close = 101.01; // Price touching upper band
    bb_upper_values[0] = 101.0;
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Test 6: Simulate Bollinger Band touch lower
    std::cout << "\nTest 6: Bollinger Band touch lower..." << std::endl;
    candle.close = 98.99;  // Price touching lower band
    bb_upper_values[0] = 101.0;
    bb_lower_values[0] = 99.0;
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Test 7: Simulate Bollinger Band breakout upper
    std::cout << "\nTest 7: Bollinger Band breakout upper..." << std::endl;
    candle.close = 102.0;  // Price breaking out above upper band
    bb_upper_values[0] = 101.0;
    bb_lower_values[0] = 99.0;
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    // Test 8: Simulate Bollinger Band breakout lower
    std::cout << "\nTest 8: Bollinger Band breakout lower..." << std::endl;
    candle.close = 98.0;   // Price breaking out below lower band
    bb_upper_values[0] = 101.0;
    bb_lower_values[0] = 99.0;
    alerts.checkAlerts(candle, sma_values, ema_values, rsi_values, bb_upper_values,
                      bb_lower_values, bb_middle_values, macd_values,
                      macd_signal_values, stochastic_k_values, stochastic_d_values, symbol);

    std::cout << "\nIndicator Alerts test completed!" << std::endl;
    std::cout << "Check indicator_alerts_test.log for logged alerts." << std::endl;

    return 0;
}