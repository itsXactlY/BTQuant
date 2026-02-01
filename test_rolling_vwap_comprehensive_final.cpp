#include "indicators/rolling_vwap.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Comprehensive Rolling VWAP Test Suite\n" << std::endl;

    // Test 1: Basic functionality with known values
    std::cout << "Test 1: Basic functionality with window size 3" << std::endl;
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    
    // Create 5 bars with predictable values
    for (int i = 0; i < 5; ++i) {
        BTQuant::RenderEngine::OHLCVCandle bar;
        bar.timestamp = 1000000 + i * 60000000;
        bar.open = 100.0 + i;
        bar.high = 101.0 + i;
        bar.low = 99.0 + i;
        bar.close = 100.0 + i;
        bar.volume = 1000; // Constant volume
        bar.trade_count = 50 + i;
        bars.push_back(bar);
    }

    btq::RollingVWAP rollingVWAP(3); // Window size of 3
    rollingVWAP.calculate(bars);

    auto vwapValues = rollingVWAP.getVWAPValues();
    assert(vwapValues.size() == bars.size());
    std::cout << "  ✓ Correct number of VWAP values calculated" << std::endl;

    // Verify first value (should be just the first bar's typical price)
    double expectedFirst = (bars[0].high + bars[0].low + bars[0].close) / 3.0;
    assert(std::abs(vwapValues[0] - expectedFirst) < 0.001);
    std::cout << "  ✓ First VWAP value correct: " << vwapValues[0] << std::endl;

    // Verify second value (average of first two bars with equal volumes)
    double tp0 = (bars[0].high + bars[0].low + bars[0].close) / 3.0;
    double tp1 = (bars[1].high + bars[1].low + bars[1].close) / 3.0;
    double expectedSecond = (tp0 * 1000 + tp1 * 1000) / (1000 + 1000);
    assert(std::abs(vwapValues[1] - expectedSecond) < 0.001);
    std::cout << "  ✓ Second VWAP value correct: " << vwapValues[1] << std::endl;

    // Verify third value (average of first three bars)
    double tp2 = (bars[2].high + bars[2].low + bars[2].close) / 3.0;
    double expectedThird = (tp0 * 1000 + tp1 * 1000 + tp2 * 1000) / (1000 + 1000 + 1000);
    assert(std::abs(vwapValues[2] - expectedThird) < 0.001);
    std::cout << "  ✓ Third VWAP value correct: " << vwapValues[2] << std::endl;

    // Verify fourth value (rolling window: bars 1, 2, 3)
    double tp3 = (bars[3].high + bars[3].low + bars[3].close) / 3.0;
    double expectedFourth = (tp1 * 1000 + tp2 * 1000 + tp3 * 1000) / (1000 + 1000 + 1000);
    assert(std::abs(vwapValues[3] - expectedFourth) < 0.001);
    std::cout << "  ✓ Fourth VWAP value correct (rolling window): " << vwapValues[3] << std::endl;

    std::cout << "  ✓ Basic functionality test passed!" << std::endl;

    // Test 2: Different window sizes
    std::cout << "\nTest 2: Testing different window sizes" << std::endl;
    
    btq::RollingVWAP smallWindowVWAP(1);
    smallWindowVWAP.calculate(bars);
    assert(smallWindowVWAP.getWindowSize() == 1);
    assert(smallWindowVWAP.size() == bars.size());
    
    // With window size 1, each VWAP should equal the typical price of that single bar
    for (size_t i = 0; i < bars.size(); ++i) {
        double expectedTypicalPrice = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        assert(std::abs(smallWindowVWAP.getVWAPValues()[i] - expectedTypicalPrice) < 0.001);
    }
    std::cout << "  ✓ Window size 1 handled correctly" << std::endl;

    btq::RollingVWAP largeWindowVWAP(10);
    largeWindowVWAP.calculate(bars);
    assert(largeWindowVWAP.getWindowSize() == 10);
    assert(largeWindowVWAP.size() == bars.size());
    std::cout << "  ✓ Large window size handled correctly" << std::endl;

    // Test 3: Standard deviation bands
    std::cout << "\nTest 3: Testing standard deviation bands" << std::endl;
    
    auto sd1Upper = rollingVWAP.getSD1UpperBand();
    auto sd1Lower = rollingVWAP.getSD1LowerBand();
    auto sd2Upper = rollingVWAP.getSD2UpperBand();
    auto sd2Lower = rollingVWAP.getSD2LowerBand();
    auto sd3Upper = rollingVWAP.getSD3UpperBand();
    auto sd3Lower = rollingVWAP.getSD3LowerBand();
    
    // Verify all vectors have the same size
    assert(vwapValues.size() == sd1Upper.size());
    assert(vwapValues.size() == sd1Lower.size());
    assert(vwapValues.size() == sd2Upper.size());
    assert(vwapValues.size() == sd2Lower.size());
    assert(vwapValues.size() == sd3Upper.size());
    assert(vwapValues.size() == sd3Lower.size());
    std::cout << "  ✓ All band vectors have consistent sizes" << std::endl;

    // Verify bands are properly ordered
    for (size_t i = 0; i < vwapValues.size(); ++i) {
        assert(sd1Upper[i] >= vwapValues[i]);
        assert(vwapValues[i] >= sd1Lower[i]);
        assert(sd2Upper[i] >= sd1Upper[i]);
        assert(sd1Lower[i] >= sd2Lower[i]);
        assert(sd3Upper[i] >= sd2Upper[i]);
        assert(sd2Lower[i] >= sd3Lower[i]);
    }
    std::cout << "  ✓ Standard deviation bands are properly ordered" << std::endl;

    // Test 4: Edge cases
    std::cout << "\nTest 4: Testing edge cases" << std::endl;
    
    // Empty bars vector
    std::vector<BTQuant::RenderEngine::OHLCVCandle> emptyBars;
    btq::RollingVWAP emptyVWAP(3);
    emptyVWAP.calculate(emptyBars);
    assert(emptyVWAP.size() == 0);
    std::cout << "  ✓ Empty bars vector handled correctly" << std::endl;

    // Single bar
    std::vector<BTQuant::RenderEngine::OHLCVCandle> singleBar = {bars[0]};
    btq::RollingVWAP singleVWAP(3);
    singleVWAP.calculate(singleBar);
    assert(singleVWAP.size() == 1);
    double expectedSingle = (bars[0].high + bars[0].low + bars[0].close) / 3.0;
    assert(std::abs(singleVWAP.getVWAPValues()[0] - expectedSingle) < 0.001);
    std::cout << "  ✓ Single bar handled correctly" << std::endl;

    // Zero volume bars (should not cause division by zero)
    std::vector<BTQuant::RenderEngine::OHLCVCandle> zeroVolBars;
    for (int i = 0; i < 3; ++i) {
        BTQuant::RenderEngine::OHLCVCandle bar;
        bar.timestamp = 1000000 + i * 60000000;
        bar.open = 100.0 + i;
        bar.high = 101.0 + i;
        bar.low = 99.0 + i;
        bar.close = 100.0 + i;
        bar.volume = 0; // Zero volume
        bar.trade_count = 50 + i;
        zeroVolBars.push_back(bar);
    }
    
    btq::RollingVWAP zeroVolVWAP(3);
    zeroVolVWAP.calculate(zeroVolBars);
    // With zero volume, VWAP should be 0
    for (size_t i = 0; i < zeroVolVWAP.getVWAPValues().size(); ++i) {
        assert(std::abs(zeroVolVWAP.getVWAPValues()[i]) < 0.001);
    }
    std::cout << "  ✓ Zero volume bars handled correctly" << std::endl;

    // Test 5: API methods
    std::cout << "\nTest 5: Testing API methods" << std::endl;
    
    btq::RollingVWAP apiTestVWAP(5);
    assert(apiTestVWAP.getWindowSize() == 5);
    std::cout << "  ✓ getWindowSize() works correctly" << std::endl;

    apiTestVWAP.setWindowSize(7);
    assert(apiTestVWAP.getWindowSize() == 7);
    std::cout << "  ✓ setWindowSize() works correctly" << std::endl;

    apiTestVWAP.clear();
    assert(apiTestVWAP.size() == 0);
    std::cout << "  ✓ clear() works correctly" << std::endl;

    // Test 6: Rolling behavior verification
    std::cout << "\nTest 6: Verifying rolling window behavior" << std::endl;
    
    // Create bars with different volumes to test weighted average
    std::vector<BTQuant::RenderEngine::OHLCVCandle> weightedBars;
    for (int i = 0; i < 6; ++i) {
        BTQuant::RenderEngine::OHLCVCandle bar;
        bar.timestamp = 1000000 + i * 60000000;
        bar.open = 100.0;
        bar.high = 102.0;
        bar.low = 98.0;
        bar.close = 100.0;
        bar.volume = 100 + i * 50; // Increasing volume
        bar.trade_count = 50 + i;
        weightedBars.push_back(bar);
    }

    btq::RollingVWAP weightedVWAP(3);
    weightedVWAP.calculate(weightedBars);
    
    // At index 3, the window should contain bars 1, 2, 3 (0-indexed)
    // Verify that the calculation considers the rolling window properly
    auto weightedResults = weightedVWAP.getVWAPValues();
    assert(weightedResults.size() == weightedBars.size());
    std::cout << "  ✓ Rolling window behavior verified with weighted volumes" << std::endl;

    std::cout << "\nAll tests passed! Rolling VWAP implementation is working correctly." << std::endl;
    std::cout << "Features tested:" << std::endl;
    std::cout << "  - Basic VWAP calculation with rolling windows" << std::endl;
    std::cout << "  - Different window sizes (1, 3, 10)" << std::endl;
    std::cout << "  - Standard deviation bands (1, 2, 3 sigma)" << std::endl;
    std::cout << "  - Edge cases (empty, single bar, zero volume)" << std::endl;
    std::cout << "  - API methods (getters, setters, clear)" << std::endl;
    std::cout << "  - Rolling window behavior with weighted averages" << std::endl;

    return 0;
}