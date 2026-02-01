#include "indicators/vwap_alerts.hpp"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Testing VWAP Alerts..." << std::endl;
    
    btq::VWAPAlerts alerts;
    
    // Set up a simple callback to handle alerts
    alerts.setAlertCallback([](const btq::VWAPAlertEvent& event) {
        std::cout << "ALERT TRIGGERED: ";
        
        switch(event.alert_type) {
            case btq::VWAPAlertType::PRICE_CROSSES_VWAP:
                std::cout << "Price crosses VWAP";
                break;
            case btq::VWAPAlertType::PRICE_TOUCHES_SD1:
                std::cout << "Price touches SD1 band";
                break;
            case btq::VWAPAlertType::PRICE_TOUCHES_SD2:
                std::cout << "Price touches SD2 band";
                break;
            case btq::VWAPAlertType::PRICE_TOUCHES_SD3:
                std::cout << "Price touches SD3 band";
                break;
            case btq::VWAPAlertType::VWAP_DIRECTION_CHANGE:
                std::cout << "VWAP direction change";
                break;
            case btq::VWAPAlertType::VWAP_VOLUME_SPIKE:
                std::cout << "VWAP volume spike";
                break;
        }
        
        std::cout << " | Symbol: " << event.symbol 
                  << " | Price: " << event.price 
                  << " | VWAP: " << event.vwap_value
                  << " | Bullish: " << (event.is_bullish ? "Yes" : "No") << std::endl;
    });
    
    // Test data
    std::string symbol = "TEST";
    
    // Simulate some price movements relative to VWAP
    std::cout << "\nSimulating price movements..." << std::endl;
    
    // Initial values
    alerts.checkAlerts(100.0, 100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 1672531200000000, symbol);
    
    // Price crosses VWAP from below
    alerts.checkAlerts(100.5, 100.1, 101.1, 99.1, 102.1, 98.1, 103.1, 97.1, 1672531201000000, symbol);
    
    // Price touches SD2 band
    alerts.checkAlerts(102.1, 100.2, 101.2, 99.2, 102.0, 98.2, 103.2, 97.2, 1672531202000000, symbol);
    
    // Price crosses VWAP from above
    alerts.checkAlerts(99.8, 100.3, 101.3, 99.3, 102.3, 98.3, 103.3, 97.3, 1672531203000000, symbol);
    
    std::cout << "\nVWAP Alerts test completed!" << std::endl;
    
    return 0;
}