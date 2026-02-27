#include "indicators/rolling_vwap.hpp"
#include <cmath>
#include <algorithm>
#include <deque>

namespace btq {

void RollingVWAP::calculate(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars) {
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

} // namespace btq