#include "indicators/rolling_vwap.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create sample OHLCV data
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    
    // Add 10 sample bars with increasing prices and varying volumes
    for (int i = 0; i < 10; ++i) {
        BTQuant::RenderEngine::OHLCVCandle bar;
        bar.timestamp = 1000000 + i * 60000000; // 1-minute intervals in microseconds
        bar.open = 100.0 + i * 0.5;
        bar.high = 101.0 + i * 0.5;
        bar.low = 99.0 + i * 0.5;
        bar.close = 100.5 + i * 0.5;
        bar.volume = 1000 + i * 100;
        bar.trade_count = 50 + i * 5;
        bars.push_back(bar);
    }

    // Test rolling VWAP with window size of 5
    btq::RollingVWAP rollingVWAP(5);
    rollingVWAP.calculate(bars);

    std::cout << "Rolling VWAP Results (Window Size: 5):\n";
    std::cout << "Bar\tVWAP\t\tSD1 Lower\tSD1 Upper\tSD2 Lower\tSD2 Upper\n";
    std::cout << "---\t----\t\t---------\t---------\t---------\t---------\n";

    auto vwapValues = rollingVWAP.getVWAPValues();
    auto sd1Lower = rollingVWAP.getSD1LowerBand();
    auto sd1Upper = rollingVWAP.getSD1UpperBand();
    auto sd2Lower = rollingVWAP.getSD2LowerBand();
    auto sd2Upper = rollingVWAP.getSD2UpperBand();

    for (size_t i = 0; i < vwapValues.size(); ++i) {
        std::cout << i << "\t" 
                  << vwapValues[i] << "\t\t" 
                  << sd1Lower[i] << "\t\t" 
                  << sd1Upper[i] << "\t\t" 
                  << sd2Lower[i] << "\t\t" 
                  << sd2Upper[i] << std::endl;
    }

    std::cout << "\nTotal VWAP values calculated: " << rollingVWAP.size() << std::endl;
    std::cout << "Expected: " << bars.size() << " (one for each bar)" << std::endl;

    return 0;
}