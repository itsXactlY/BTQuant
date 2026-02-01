#include "indicators/anchored_vwap.hpp"
#include <cmath>
#include <algorithm>

namespace btq {

void AnchoredVWAP::calculate(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars) {
    // Clear any existing data
    clear();
    
    // Find the index of the first bar that has a timestamp >= anchor timestamp
    size_t startIndex = 0;
    bool anchorFound = false;
    
    for (size_t i = 0; i < bars.size(); ++i) {
        if (bars[i].timestamp >= anchorTimestamp_) {
            startIndex = i;
            anchorFound = true;
            break;
        }
    }
    
    // If no bars are found from the anchor point forward, return early
    if (!anchorFound) {
        return;
    }
    
    // Calculate running VWAP and standard deviation bands from the anchor point forward
    double cumulativeTPV = 0.0;  // Cumulative Total Price * Volume
    double cumulativeVolume = 0.0;  // Cumulative Total Volume
    
    for (size_t i = startIndex; i < bars.size(); ++i) {
        const auto& bar = bars[i];
        
        // Calculate typical price for this bar: (High + Low + Close) / 3
        double typicalPrice = (bar.high + bar.low + bar.close) / 3.0;
        
        // Add this bar's contribution to cumulative totals
        cumulativeTPV += typicalPrice * bar.volume;
        cumulativeVolume += bar.volume;
        
        // Calculate current VWAP
        double currentVWAP = 0.0;
        if (cumulativeVolume > 0.0) {
            currentVWAP = cumulativeTPV / cumulativeVolume;
        }
        
        // Calculate standard deviation bands if we have accumulated volume
        double sd1Upper = 0.0, sd1Lower = 0.0;
        double sd2Upper = 0.0, sd2Lower = 0.0;
        double sd3Upper = 0.0, sd3Lower = 0.0;

        if (cumulativeVolume > 0.0) {
            // Calculate variance for standard deviation
            // For anchored VWAP, we calculate the weighted variance from the anchor point
            double weightedSquaredDeviations = 0.0;

            // Count how many bars we have from anchor to current position
            size_t barCount = i - startIndex + 1;

            if (barCount > 1) { // Need at least 2 bars to calculate meaningful variance
                // Recalculate variance from the anchor point to current bar
                for (size_t j = startIndex; j <= i; ++j) {
                    const auto& calcBar = bars[j];
                    double calcTypicalPrice = (calcBar.high + calcBar.low + calcBar.close) / 3.0;
                    double deviation = calcTypicalPrice - currentVWAP;
                    weightedSquaredDeviations += (deviation * deviation) * calcBar.volume;
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
                // For the first bar, set bands equal to the VWAP value or use a minimum spread
                sd1Upper = currentVWAP;
                sd1Lower = currentVWAP;
                sd2Upper = currentVWAP;
                sd2Lower = currentVWAP;
                sd3Upper = currentVWAP;
                sd3Lower = currentVWAP;
            }
        }
        
        // Add the calculated values to our storage
        addVWAPValue(currentVWAP, sd1Upper, sd1Lower, sd2Upper, sd2Lower, sd3Upper, sd3Lower);
    }
}

} // namespace btq