#include "indicators/session_vwap.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace btq {

bool SessionVWAP::isNewSession(uint64_t timestamp) const {
    // Check if this timestamp matches any of the predefined session start times
    for (uint64_t sessionStartTime : sessionStartTimes_) {
        // Allow for some tolerance in timestamp matching (e.g., 1 minute = 60000 ms)
        if (timestamp >= sessionStartTime && timestamp < sessionStartTime + 60000) {
            return true;
        }
    }
    
    // Also check if there are existing sessions and this timestamp is significantly later
    // than the last session start time, indicating a new trading day
    if (!sessions_.empty()) {
        const auto& lastSession = sessions_.back();
        // Assuming a new trading day if the gap is more than 4 hours (14400000 ms)
        if (timestamp > lastSession.startTime + 14400000) {
            // Check if this timestamp aligns with typical market open times
            // Convert to hour of day (assuming timestamp is in milliseconds)
            uint64_t secondsSinceEpoch = timestamp / 1000;
            uint64_t remainder = secondsSinceEpoch % 86400; // seconds in a day
            uint64_t hourOfDay = (remainder / 3600) % 24;
            
            // Typical market opens at 9:30 AM (9.5 hours = 34200 seconds since midnight)
            // Allow for some flexibility around market open time
            if (hourOfDay == 9 || hourOfDay == 10) { // Approximate market open time
                return true;
            }
        }
    }
    
    return false;
}

int SessionVWAP::getSessionIndexForTimestamp(uint64_t timestamp) const {
    for (size_t i = 0; i < sessions_.size(); ++i) {
        if (timestamp >= sessions_[i].startTime) {
            // Check if this is within the session duration
            // For now, assume the session continues until a new one starts or end of day
            bool nextSessionExists = (i + 1 < sessions_.size());
            if (!nextSessionExists || timestamp < sessions_[i + 1].startTime) {
                return static_cast<int>(i);
            }
        }
    }
    return -1; // Not found in any session
}

void SessionVWAP::createNewSession(uint64_t startTime) {
    // If there's an active session, mark it as inactive (historical)
    if (!sessions_.empty() && sessions_.back().isActive) {
        sessions_.back().isActive = false;
        // Set the end time of the previous session to just before the new session starts
        sessions_.back().endTime = startTime;
    }
    
    // Create a new active session
    sessions_.emplace_back(startTime, true);
}

void SessionVWAP::addVWAPValueToCurrentSession(double vwap, double sd1Upper, double sd1Lower,
                                               double sd2Upper, double sd2Lower,
                                               double sd3Upper, double sd3Lower) {
    VWAPSession* currentSession = getCurrentActiveSession();
    if (currentSession) {
        currentSession->vwapValues.push_back(vwap);
        currentSession->sd1UpperBand.push_back(sd1Upper);
        currentSession->sd1LowerBand.push_back(sd1Lower);
        currentSession->sd2UpperBand.push_back(sd2Upper);
        currentSession->sd2LowerBand.push_back(sd2Lower);
        currentSession->sd3UpperBand.push_back(sd3Upper);
        currentSession->sd3LowerBand.push_back(sd3Lower);
    }
}

void SessionVWAP::calculateSessionVWAP(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars, 
                                      size_t startIndex, size_t endIndex, 
                                      VWAPSession& session) {
    if (startIndex >= bars.size() || endIndex >= bars.size() || startIndex > endIndex) {
        return;
    }

    // Calculate VWAP for the session range
    double cumulativeTPV = 0.0;  // Cumulative Total Price * Volume
    double cumulativeVolume = 0.0;  // Cumulative Total Volume

    for (size_t i = startIndex; i <= endIndex; ++i) {
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
            double weightedSquaredDeviations = 0.0;

            // Count how many bars we have from session start to current position
            size_t barCount = i - startIndex + 1;

            if (barCount > 1) { // Need at least 2 bars to calculate meaningful variance
                // Recalculate variance from the session start to current bar
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

        // Add the calculated values to the session
        session.vwapValues.push_back(currentVWAP);
        session.sd1UpperBand.push_back(sd1Upper);
        session.sd1LowerBand.push_back(sd1Lower);
        session.sd2UpperBand.push_back(sd2Upper);
        session.sd2LowerBand.push_back(sd2Lower);
        session.sd3UpperBand.push_back(sd3Upper);
        session.sd3LowerBand.push_back(sd3Lower);
    }
}

void SessionVWAP::calculate(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars) {
    // Clear any existing data
    clear();

    if (bars.empty()) {
        return;
    }

    // Process each bar and determine if it belongs to a new session
    size_t sessionStartIndex = 0;
    
    for (size_t i = 0; i < bars.size(); ++i) {
        const auto& bar = bars[i];
        
        // Check if this bar starts a new session
        if (isNewSession(bar.timestamp)) {
            // If we're not at the beginning, finalize the previous session
            if (i > 0 && !sessions_.empty()) {
                // Calculate VWAP for the previous session range
                if (sessionStartIndex < i) {
                    calculateSessionVWAP(bars, sessionStartIndex, i - 1, sessions_.back());
                }
            }
            
            // Create a new session starting at this bar
            createNewSession(bar.timestamp);
            sessionStartIndex = i;
        }
    }
    
    // Calculate VWAP for the final session
    if (sessionStartIndex < bars.size() && !sessions_.empty()) {
        calculateSessionVWAP(bars, sessionStartIndex, bars.size() - 1, sessions_.back());
    }
}

} // namespace btq