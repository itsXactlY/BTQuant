#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <deque>
#include <cstdint>

// Simplified OHLCVCandle structure for testing
struct OHLCVCandle {
    uint64_t timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    uint64_t trade_count;
};

// Simplified RollingVWAP implementation for testing
class RollingVWAP {
public:
    // Constructor
    RollingVWAP() : windowSize_(20) {}

    // Constructor with window size
    explicit RollingVWAP(size_t windowSize) : windowSize_(windowSize) {}

    // Calculate rolling VWAP for the last N bars using OHLCV bars
    void calculate(const std::vector<OHLCVCandle>& bars) {
        // Clear any existing data
        clear();

        if (bars.empty() || windowSize_ == 0) {
            return;
        }

        // Use a deque to maintain a sliding window of bar data
        std::deque<BarData> window;
        double cumulativeTPV = 0.0;  // Cumulative Total Price * Volume
        double cumulativeVolume = 0.0;  // Cumulative Total Volume

        for (size_t i = 0; i < bars.size(); ++i) {
            const auto& bar = bars[i];

            // Calculate typical price for this bar: (High + Low + Close) / 3
            double typicalPrice = (bar.high + bar.low + bar.close) / 3.0;
            double tpv = typicalPrice * bar.volume;

            // Add the current bar to the window
            window.push_back(BarData(typicalPrice, bar.volume, tpv));
            cumulativeTPV += tpv;
            cumulativeVolume += bar.volume;

            // If window exceeds the specified size, remove the oldest bar
            if (window.size() > windowSize_) {
                const auto& oldestBar = window.front();
                cumulativeTPV -= oldestBar.tpv;
                cumulativeVolume -= oldestBar.volume;
                window.pop_front();
            }

            // Calculate current VWAP
            double currentVWAP = 0.0;
            if (cumulativeVolume > 0.0) {
                currentVWAP = cumulativeTPV / cumulativeVolume;
            }

            // Calculate standard deviation bands
            double sd1Upper = 0.0, sd1Lower = 0.0;
            double sd2Upper = 0.0, sd2Lower = 0.0;
            double sd3Upper = 0.0, sd3Lower = 0.0;

            if (cumulativeVolume > 0.0 && window.size() > 1) {
                // Calculate variance for standard deviation
                double weightedSquaredDeviations = 0.0;

                // Calculate weighted variance for the current window
                for (const auto& windowBar : window) {
                    double deviation = windowBar.typicalPrice - currentVWAP;
                    weightedSquaredDeviations += (deviation * deviation) * windowBar.volume;
                }

                double variance = weightedSquaredDeviations / cumulativeVolume;
                double stdDev = std::sqrt(variance);

                // Calculate standard deviation bands
                sd1Upper = currentVWAP + stdDev;
                sd1Lower = currentVWAP - stdDev;
                sd2Upper = currentVWAP + (2.0 * stdDev);
                sd2Lower = currentVWAP - (2.0 * stdDev);
                sd3Upper = currentVWAP + (3.0 * stdDev);
                sd3Lower = currentVWAP - (3.0 * stdDev);
            } else {
                // For the first bar or when volume is zero, set bands equal to the VWAP value
                sd1Upper = currentVWAP;
                sd1Lower = currentVWAP;
                sd2Upper = currentVWAP;
                sd2Lower = currentVWAP;
                sd3Upper = currentVWAP;
                sd3Lower = currentVWAP;
            }

            // Add the calculated values to our storage
            addVWAPValue(currentVWAP, sd1Upper, sd1Lower, sd2Upper, sd2Lower, sd3Upper, sd3Lower);
        }
    }

    // Add individual VWAP value with corresponding standard deviation bands
    void addVWAPValue(double vwap, double sd1Upper, double sd1Lower,
                      double sd2Upper, double sd2Lower,
                      double sd3Upper, double sd3Lower) {
        vwapValues_.push_back(vwap);
        sd1UpperBand_.push_back(sd1Upper);
        sd1LowerBand_.push_back(sd1Lower);
        sd2UpperBand_.push_back(sd2Upper);
        sd2LowerBand_.push_back(sd2Lower);
        sd3UpperBand_.push_back(sd3Upper);
        sd3LowerBand_.push_back(sd3Lower);
    }

    // Clear all stored data
    void clear() {
        vwapValues_.clear();
        sd1UpperBand_.clear();
        sd1LowerBand_.clear();
        sd2UpperBand_.clear();
        sd2LowerBand_.clear();
        sd3UpperBand_.clear();
        sd3LowerBand_.clear();
    }

    // Get the count of stored values
    size_t size() const {
        return vwapValues_.size();
    }

    // Getters
    size_t getWindowSize() const { return windowSize_; }
    const std::vector<double>& getVWAPValues() const { return vwapValues_; }
    const std::vector<double>& getSD1UpperBand() const { return sd1UpperBand_; }
    const std::vector<double>& getSD1LowerBand() const { return sd1LowerBand_; }
    const std::vector<double>& getSD2UpperBand() const { return sd2UpperBand_; }
    const std::vector<double>& getSD2LowerBand() const { return sd2LowerBand_; }
    const std::vector<double>& getSD3UpperBand() const { return sd3UpperBand_; }
    const std::vector<double>& getSD3LowerBand() const { return sd3LowerBand_; }

    // Setter
    void setWindowSize(size_t windowSize) { windowSize_ = windowSize; }

private:
    size_t windowSize_;
    std::vector<double> vwapValues_;
    std::vector<double> sd1UpperBand_;
    std::vector<double> sd1LowerBand_;
    std::vector<double> sd2UpperBand_;
    std::vector<double> sd2LowerBand_;
    std::vector<double> sd3UpperBand_;
    std::vector<double> sd3LowerBand_;
    
    // Helper structure to store bar data for rolling window calculation
    struct BarData {
        double typicalPrice;
        double volume;
        double tpv; // Typical Price * Volume
        
        BarData(double tp, double vol, double t) : typicalPrice(tp), volume(vol), tpv(t) {}
    };
};

int main() {
    std::cout << "Testing Rolling VWAP implementation...\n" << std::endl;

    // Test 1: Basic functionality with window size 3
    std::cout << "Test 1: Basic functionality with window size 3" << std::endl;
    std::vector<OHLCVCandle> bars;
    
    // Add 6 sample bars with predictable values
    for (int i = 0; i < 6; ++i) {
        OHLCVCandle bar;
        bar.timestamp = 1000000 + i * 60000000;
        bar.open = 100.0 + i;
        bar.high = 101.0 + i;
        bar.low = 99.0 + i;
        bar.close = 100.0 + i;
        bar.volume = 1000; // Constant volume for easier calculation
        bar.trade_count = 50 + i;
        bars.push_back(bar);
    }

    RollingVWAP rollingVWAP(3); // Window size of 3
    rollingVWAP.calculate(bars);

    auto vwapValues = rollingVWAP.getVWAPValues();
    
    // For the first bar, VWAP should be just that bar's typical price (100.0)
    double expectedFirst = (100.0 + 101.0 + 100.0) / 3.0; // (high + low + close) / 3
    assert(std::abs(vwapValues[0] - expectedFirst) < 0.001);
    std::cout << "  ✓ First VWAP value correct: " << vwapValues[0] << " (expected ~" << expectedFirst << ")" << std::endl;

    // For the second bar, VWAP should be volume-weighted average of first two bars
    // Since volumes are equal (1000), it's just the average of typical prices
    double tp0 = (100.0 + 101.0 + 100.0) / 3.0;  // 100.0
    double tp1 = (101.0 + 102.0 + 101.0) / 3.0;  // 101.333
    double expectedSecond = (tp0 * 1000 + tp1 * 1000) / (1000 + 1000);  // Volume-weighted average
    assert(std::abs(vwapValues[1] - expectedSecond) < 0.001);
    std::cout << "  ✓ Second VWAP value correct: " << vwapValues[1] << " (expected ~" << expectedSecond << ")" << std::endl;

    // For the third bar, VWAP should be volume-weighted average of first three bars
    double tp2 = (102.0 + 103.0 + 102.0) / 3.0;  // 102.333
    double expectedThird = (tp0 * 1000 + tp1 * 1000 + tp2 * 1000) / (1000 + 1000 + 1000);  // Volume-weighted average
    assert(std::abs(vwapValues[2] - expectedThird) < 0.001);
    std::cout << "  ✓ Third VWAP value correct: " << vwapValues[2] << " (expected ~" << expectedThird << ")" << std::endl;

    // For the fourth bar and onwards, it should be a rolling window of 3 bars
    // At bar 4 (index 3), we have bars 1, 2, 3 in the window (bar 0 is dropped)
    double tp3 = (103.0 + 104.0 + 103.0) / 3.0;  // 103.333
    double expectedFourth = (tp1 * 1000 + tp2 * 1000 + tp3 * 1000) / (1000 + 1000 + 1000);  // Volume-weighted average
    if (vwapValues.size() > 3) {
        assert(std::abs(vwapValues[3] - expectedFourth) < 0.001);
        std::cout << "  ✓ Fourth VWAP value correct: " << vwapValues[3] << " (expected ~" << expectedFourth << ")" << std::endl;
    }

    std::cout << "  ✓ Rolling window functionality works correctly!" << std::endl;

    // Test 2: Different window sizes
    std::cout << "\nTest 2: Testing different window sizes" << std::endl;
    
    RollingVWAP rollingVWAPLarge(5);
    rollingVWAPLarge.calculate(bars);
    
    assert(rollingVWAPLarge.size() == bars.size());
    assert(rollingVWAPLarge.getWindowSize() == 5);
    std::cout << "  ✓ Large window size (5) handled correctly" << std::endl;

    RollingVWAP rollingVWAPSmall(1);
    rollingVWAPSmall.calculate(bars);
    
    assert(rollingVWAPSmall.size() == bars.size());
    assert(rollingVWAPSmall.getWindowSize() == 1);
    
    // With window size 1, each VWAP should equal the typical price of that single bar
    for (size_t i = 0; i < bars.size(); ++i) {
        double expectedTypicalPrice = (bars[i].high + bars[i].low + bars[i].close) / 3.0;
        assert(std::abs(rollingVWAPSmall.getVWAPValues()[i] - expectedTypicalPrice) < 0.001);
    }
    std::cout << "  ✓ Small window size (1) handled correctly" << std::endl;

    // Test 3: Edge cases
    std::cout << "\nTest 3: Testing edge cases" << std::endl;
    
    // Empty bars vector
    std::vector<OHLCVCandle> emptyBars;
    RollingVWAP emptyVWAP(3);
    emptyVWAP.calculate(emptyBars);
    assert(emptyVWAP.size() == 0);
    std::cout << "  ✓ Empty bars vector handled correctly" << std::endl;

    // Single bar
    std::vector<OHLCVCandle> singleBar = {bars[0]};
    RollingVWAP singleVWAP(3);
    singleVWAP.calculate(singleBar);
    assert(singleVWAP.size() == 1);
    double expectedSingle = (bars[0].high + bars[0].low + bars[0].close) / 3.0;
    assert(std::abs(singleVWAP.getVWAPValues()[0] - expectedSingle) < 0.001);
    std::cout << "  ✓ Single bar handled correctly" << std::endl;

    // Test 4: Standard deviation bands
    std::cout << "\nTest 4: Testing standard deviation bands" << std::endl;
    
    auto sd1Upper = rollingVWAP.getSD1UpperBand();
    auto sd1Lower = rollingVWAP.getSD1LowerBand();
    auto sd2Upper = rollingVWAP.getSD2UpperBand();
    auto sd2Lower = rollingVWAP.getSD2LowerBand();
    
    // Check that bands are properly ordered (upper > vwap > lower)
    for (size_t i = 0; i < vwapValues.size(); ++i) {
        assert(sd1Upper[i] >= vwapValues[i]);
        assert(vwapValues[i] >= sd1Lower[i]);
        assert(sd2Upper[i] >= sd1Upper[i]);
        assert(sd1Lower[i] >= sd2Lower[i]);
    }
    std::cout << "  ✓ Standard deviation bands are properly ordered" << std::endl;

    std::cout << "\nAll tests passed! Rolling VWAP implementation is working correctly." << std::endl;

    return 0;
}