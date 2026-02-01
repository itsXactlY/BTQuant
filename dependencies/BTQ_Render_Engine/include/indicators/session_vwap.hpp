#pragma once

#include <vector>
#include <cstdint>
#include <memory>

// Forward declaration to avoid circular dependency
// The actual OHLCVCandle struct is defined in market_data_processor.hpp
#include "../market_data_processor.hpp"

namespace btq {

struct VWAPSession {
    uint64_t startTime;
    uint64_t endTime;
    std::vector<double> vwapValues;
    std::vector<double> sd1UpperBand;
    std::vector<double> sd1LowerBand;
    std::vector<double> sd2UpperBand;
    std::vector<double> sd2LowerBand;
    std::vector<double> sd3UpperBand;
    std::vector<double> sd3LowerBand;
    bool isActive;
    
    VWAPSession(uint64_t start, bool active = true) 
        : startTime(start), endTime(0), isActive(active) {}
};

/**
 * @brief Session VWAP (Volume Weighted Average Price) indicator
 * Automatically creates new VWAP at session start (market open time),
 * marks previous session VWAP as historical
 */
class SessionVWAP {
public:
    // Constructor
    SessionVWAP();

    // Constructor with session start times
    explicit SessionVWAP(const std::vector<uint64_t>& sessionStartTimes);

    // Getters
    const std::vector<VWAPSession>& getSessions() const;
    const std::vector<double>& getCurrentVWAPValues() const;
    const std::vector<double>& getCurrentSD1UpperBand() const;
    const std::vector<double>& getCurrentSD1LowerBand() const;
    const std::vector<double>& getCurrentSD2UpperBand() const;
    const std::vector<double>& getCurrentSD2LowerBand() const;
    const std::vector<double>& getCurrentSD3UpperBand() const;
    const std::vector<double>& getCurrentSD3LowerBand() const;

    // Setters
    void setSessionStartTimes(const std::vector<uint64_t>& sessionStartTimes);

    // Clear all stored data
    void clear();

    // Get the count of stored values in current session
    size_t size() const;

    // Calculate VWAP for each session using OHLCV bars
    void calculate(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars);

    // Add individual VWAP value to current active session
    void addVWAPValueToCurrentSession(double vwap, double sd1Upper, double sd1Lower,
                                      double sd2Upper, double sd2Lower,
                                      double sd3Upper, double sd3Lower);

    // Check if a new session should start at the given timestamp
    bool isNewSession(uint64_t timestamp) const;

    // Get the current active session
    VWAPSession* getCurrentActiveSession();
    const VWAPSession* getCurrentActiveSession() const;

private:
    std::vector<uint64_t> sessionStartTimes_;
    std::vector<VWAPSession> sessions_;

    // Find the session that contains the given timestamp
    int getSessionIndexForTimestamp(uint64_t timestamp) const;

    // Create a new session starting at the given timestamp
    void createNewSession(uint64_t startTime);

    // Calculate VWAP for a specific session range
    void calculateSessionVWAP(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars, 
                             size_t startIndex, size_t endIndex, 
                             VWAPSession& session);
};

// Inline implementations
inline SessionVWAP::SessionVWAP() {}

inline SessionVWAP::SessionVWAP(const std::vector<uint64_t>& sessionStartTimes) 
    : sessionStartTimes_(sessionStartTimes) {}

inline const std::vector<VWAPSession>& SessionVWAP::getSessions() const {
    return sessions_;
}

inline const std::vector<double>& SessionVWAP::getCurrentVWAPValues() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->vwapValues;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD1UpperBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd1UpperBand;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD1LowerBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd1LowerBand;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD2UpperBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd2UpperBand;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD2LowerBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd2LowerBand;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD3UpperBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd3UpperBand;
        }
    }
    return empty;
}

inline const std::vector<double>& SessionVWAP::getCurrentSD3LowerBand() const {
    static std::vector<double> empty;
    if (sessions_.empty()) return empty;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return it->sd3LowerBand;
        }
    }
    return empty;
}

inline void SessionVWAP::setSessionStartTimes(const std::vector<uint64_t>& sessionStartTimes) {
    sessionStartTimes_ = sessionStartTimes;
}

inline void SessionVWAP::clear() {
    sessions_.clear();
}

inline size_t SessionVWAP::size() const {
    const auto* currentSession = getCurrentActiveSession();
    return currentSession ? currentSession->vwapValues.size() : 0;
}

inline VWAPSession* SessionVWAP::getCurrentActiveSession() {
    if (sessions_.empty()) return nullptr;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return &(*it);
        }
    }
    return nullptr;
}

inline const VWAPSession* SessionVWAP::getCurrentActiveSession() const {
    if (sessions_.empty()) return nullptr;
    
    for (auto it = sessions_.rbegin(); it != sessions_.rend(); ++it) {
        if (it->isActive) {
            return &(*it);
        }
    }
    return nullptr;
}

} // namespace btq