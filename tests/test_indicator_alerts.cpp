#include "dependencies/BTQ_Render_Engine/include/indicators/indicator_alerts.hpp"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Testing Indicator Alerts..." << std::endl;

    btq::IndicatorAlerts alerts;

    // Enable logging
    alerts.enableLogging(true);
    alerts.setLogFilePath("test_indicator_alerts.log");

    // Set up a simple callback to handle alerts
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

    // Configure thresholds
    alerts.setSMACrossThreshold(0.001);
    alerts.setEMACrossThreshold(0.001);
    alerts.setRSIThresholds(70, 30);
    alerts.setBollingerBandThreshold(0.01);
    alerts.setMACDThreshold(0.0001);

    std::cout << "\nSimulating indicator alerts..." << std::endl;

    // Simulate price crossing SMA from below
    std::cout << "\nTest 1: Price crossing SMA from below..." << std::endl;
    alerts.checkAlerts(100.5, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531200000000, symbol);

    // Simulate price crossing SMA from above
    std::cout << "\nTest 2: Price crossing SMA from above..." << std::endl;
    alerts.checkAlerts(99.5, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531201000000, symbol);

    // Simulate price crossing EMA from below
    std::cout << "\nTest 3: Price crossing EMA from below..." << std::endl;
    alerts.checkAlerts(100.5, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531202000000, symbol);

    // Simulate RSI oversold
    std::cout << "\nTest 4: RSI oversold..." << std::endl;
    alerts.checkAlerts(100.0, 100.0, 100.0, 25.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531203000000, symbol);

    // Simulate RSI overbought
    std::cout << "\nTest 5: RSI overbought..." << std::endl;
    alerts.checkAlerts(100.0, 100.0, 100.0, 80.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531204000000, symbol);

    // Simulate Bollinger Band touch
    std::cout << "\nTest 6: Bollinger Band touch upper..." << std::endl;
    alerts.checkAlerts(105.0, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531205000000, symbol);

    // Simulate Bollinger Band breakout
    std::cout << "\nTest 7: Bollinger Band breakout upper..." << std::endl;
    alerts.checkAlerts(110.0, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.5, 0.4, 80.0, 85.0, 1672531206000000, symbol);

    // Simulate MACD cross
    std::cout << "\nTest 8: MACD cross signal..." << std::endl;
    alerts.checkAlerts(100.0, 100.0, 100.0, 50.0, 101.0, 99.0, 100.0, 0.6, 0.4, 80.0, 85.0, 1672531207000000, symbol);

    std::cout << "\nIndicator Alerts test completed!" << std::endl;
    std::cout << "Check test_indicator_alerts.log for logged alerts." << std::endl;

    return 0;
}