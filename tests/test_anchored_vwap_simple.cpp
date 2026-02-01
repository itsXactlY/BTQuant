#include <iostream>
#include <cassert>
#include <vector>
#include <cstdint>

// Include our implementation directly - this already defines OHLCVCandle
#include "dependencies/BTQ_Render_Engine/include/indicators/anchored_vwap.hpp"

int main() {
    std::cout << "Testing AnchoredVWAP calculate method..." << std::endl;

    // Create some sample OHLCV bars
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    
    // Create bars with increasing timestamps
    uint64_t baseTime = 1672531200000; // Anchor time
    for (int i = 0; i < 5; ++i) {
        BTQuant::RenderEngine::OHLCVCandle bar;
        bar.timestamp = baseTime + (i * 1000); // Each bar 1 second apart
        bar.open = 100.0 + i;
        bar.high = 102.0 + i;
        bar.low = 98.0 + i;
        bar.close = 101.0 + i;
        bar.volume = 1000.0 + (i * 100);
        bar.trade_count = 50 + i;
        bars.push_back(bar);
    }
    
    // Create AnchoredVWAP with anchor time at baseTime
    btq::AnchoredVWAP anchoredVWAP(baseTime);
    
    // Calculate VWAP from anchor point forward
    anchoredVWAP.calculate(bars);
    
    // Verify that we have 5 VWAP values (one for each bar from anchor forward)
    assert(anchoredVWAP.size() == 5);
    std::cout << "✓ Calculate method produced correct number of values" << std::endl;
    
    // Verify that VWAP values are reasonable (should be around the typical prices)
    const auto& vwapValues = anchoredVWAP.getVWAPValues();
    assert(vwapValues[0] > 90.0 && vwapValues[0] < 110.0); // Reasonable range
    std::cout << "✓ VWAP values are in reasonable range" << std::endl;
    
    // Verify that standard deviation bands are properly calculated
    const auto& sd1Upper = anchoredVWAP.getSD1UpperBand();
    const auto& sd1Lower = anchoredVWAP.getSD1LowerBand();
    const auto& sd2Upper = anchoredVWAP.getSD2UpperBand();
    const auto& sd2Lower = anchoredVWAP.getSD2LowerBand();
    const auto& sd3Upper = anchoredVWAP.getSD3UpperBand();
    const auto& sd3Lower = anchoredVWAP.getSD3LowerBand();
    
    // Check that upper bands are >= lower bands (first bar might have equal bands)
    assert(sd1Upper[0] >= sd1Lower[0]);
    assert(sd2Upper[0] >= sd2Lower[0]);
    assert(sd3Upper[0] >= sd3Lower[0]);

    // For later bars, bands should be wider
    if (sd1Upper.size() > 1) {
        // Check that upper bands are above lower bands for second bar onwards
        assert(sd1Upper[1] > sd1Lower[1]);
        assert(sd2Upper[1] > sd2Lower[1]);
        assert(sd3Upper[1] > sd3Lower[1]);
        // And that wider bands are wider
        assert((sd2Upper[1] - sd2Lower[1]) > (sd1Upper[1] - sd1Lower[1]));
        assert((sd3Upper[1] - sd3Lower[1]) > (sd2Upper[1] - sd2Lower[1]));
    }
    std::cout << "✓ Standard deviation bands calculated correctly" << std::endl;
    
    // Test with anchor time that skips some bars
    uint64_t laterTime = baseTime + 2000; // Skip first 2 bars
    btq::AnchoredVWAP anchoredVWAP2(laterTime);
    anchoredVWAP2.calculate(bars);
    
    // Should have 3 values (from index 2 onwards)
    assert(anchoredVWAP2.size() == 3);
    std::cout << "✓ Calculate method handles anchor points correctly" << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}