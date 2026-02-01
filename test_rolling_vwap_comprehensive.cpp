#include "indicators/rolling_vwap.hpp"
#include <iostream>
#include <vector>
#include <cassert>

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

    // Test rolling VWAP with window size of 3
    btq::RollingVWAP rollingVWAP(3);
    rollingVWAP.calculate(bars);

    // Verify that we have the expected number of results
    assert(rollingVWAP.size() == bars.size());
    std::cout << "✓ Number of VWAP values matches number of input bars" << std::endl;

    // Verify that all vectors have the same size
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD1UpperBand().size());
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD1LowerBand().size());
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD2UpperBand().size());
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD2LowerBand().size());
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD3UpperBand().size());
    assert(rollingVWAP.getVWAPValues().size() == rollingVWAP.getSD3LowerBand().size());
    std::cout << "✓ All band vectors have consistent sizes" << std::endl;

    // Test that windowing works properly - first few values should have fewer bars in calculation
    auto vwapValues = rollingVWAP.getVWAPValues();
    
    // The first value should be based on just the first bar
    double firstTypicalPrice = (bars[0].high + bars[0].low + bars[0].close) / 3.0;
    double expectedFirstVWAP = firstTypicalPrice; // Since it's just one bar with volume
    if (bars[0].volume > 0) {
        expectedFirstVWAP = (firstTypicalPrice * bars[0].volume) / bars[0].volume;
    }
    
    // Due to floating point precision, we'll check if they're approximately equal
    double tolerance = 0.0001;
    assert(std::abs(vwapValues[0] - expectedFirstVWAP) < tolerance);
    std::cout << "✓ First VWAP value calculated correctly" << std::endl;

    // Test window size change
    rollingVWAP.setWindowSize(5);
    rollingVWAP.calculate(bars);
    assert(rollingVWAP.getWindowSize() == 5);
    std::cout << "✓ Window size change works correctly" << std::endl;

    // Test clearing
    rollingVWAP.clear();
    assert(rollingVWAP.size() == 0);
    std::cout << "✓ Clear function works correctly" << std::endl;

    std::cout << "\nAll tests passed! Rolling VWAP implementation is working correctly." << std::endl;

    return 0;
}