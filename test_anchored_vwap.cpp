#include "indicators/anchored_vwap.hpp"
#include "market_data_processor.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Test the AnchoredVWAP class
    std::cout << "Testing AnchoredVWAP class..." << std::endl;
    
    // Test constructor with anchor timestamp
    uint64_t anchorTime = 1672531200000; // Example timestamp in milliseconds
    btq::AnchoredVWAP vwap(anchorTime);
    
    // Verify anchor timestamp
    assert(vwap.getAnchorTimestamp() == anchorTime);
    std::cout << "✓ Anchor timestamp test passed" << std::endl;
    
    // Test adding VWAP values with standard deviation bands
    vwap.addVWAPValue(100.5, 102.0, 99.0, 103.5, 97.5, 105.0, 96.0);
    vwap.addVWAPValue(101.2, 102.7, 99.7, 104.2, 98.2, 105.7, 96.7);
    vwap.addVWAPValue(100.8, 102.3, 99.3, 103.8, 97.8, 105.3, 95.8);
    
    // Check size
    assert(vwap.size() == 3);
    std::cout << "✓ Size test passed" << std::endl;
    
    // Check values
    const auto& vwapValues = vwap.getVWAPValues();
    const auto& sd1Upper = vwap.getSD1UpperBand();
    const auto& sd1Lower = vwap.getSD1LowerBand();
    const auto& sd2Upper = vwap.getSD2UpperBand();
    const auto& sd2Lower = vwap.getSD2LowerBand();
    const auto& sd3Upper = vwap.getSD3UpperBand();
    const auto& sd3Lower = vwap.getSD3LowerBand();
    
    assert(vwapValues[0] == 100.5);
    assert(sd1Upper[0] == 102.0);
    assert(sd1Lower[0] == 99.0);
    assert(sd2Upper[0] == 103.5);
    assert(sd2Lower[0] == 97.5);
    assert(sd3Upper[0] == 105.0);
    assert(sd3Lower[0] == 96.0);
    
    std::cout << "✓ Value storage test passed" << std::endl;
    
    // Test setters
    std::vector<double> newVWAP = {200.0, 201.0};
    vwap.setVWAPValues(newVWAP);
    assert(vwap.getVWAPValues().size() == 2);
    assert(vwap.getVWAPValues()[0] == 200.0);
    std::cout << "✓ Setter test passed" << std::endl;
    
    // Test clear
    vwap.clear();
    assert(vwap.size() == 0);
    std::cout << "✓ Clear test passed" << std::endl;

    // Test calculate method
    {
        std::cout << "\nTesting calculate method..." << std::endl;

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

        // Check that upper bands are above lower bands
        assert(sd1Upper[0] > sd1Lower[0]);
        assert(sd2Upper[0] > sd2Lower[0]);
        assert(sd3Upper[0] > sd3Lower[0]);
        // And that wider bands are wider
        assert((sd2Upper[0] - sd2Lower[0]) > (sd1Upper[0] - sd1Lower[0]));
        assert((sd3Upper[0] - sd3Lower[0]) > (sd2Upper[0] - sd2Lower[0]));
        std::cout << "✓ Standard deviation bands calculated correctly" << std::endl;
    }

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}