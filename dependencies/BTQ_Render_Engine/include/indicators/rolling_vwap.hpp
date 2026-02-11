#pragma once

#include <vector>
#include <cstdint>
#include <deque>
#include <memory>

// Forward declaration to avoid circular dependency
// The actual OHLCVCandle struct is defined in market_data_processor.hpp
#include "../market_data_processor.hpp"

// Forward declaration for TaskScheduler to avoid circular includes
namespace btq {
    class TaskScheduler;
}

namespace btq {

/**
 * @brief Rolling VWAP (Volume Weighted Average Price) indicator
 * Calculates VWAP for the last N bars in a sliding window fashion
 */
class RollingVWAP {
public:
    // Constructor
    RollingVWAP();

    // Constructor with window size
    explicit RollingVWAP(size_t windowSize);

    // Getters
    size_t getWindowSize() const;
    const std::vector<double>& getVWAPValues() const;
    const std::vector<double>& getSD1UpperBand() const;
    const std::vector<double>& getSD1LowerBand() const;
    const std::vector<double>& getSD2UpperBand() const;
    const std::vector<double>& getSD2LowerBand() const;
    const std::vector<double>& getSD3UpperBand() const;
    const std::vector<double>& getSD3LowerBand() const;

    // Setters
    void setWindowSize(size_t windowSize);
    void setVWAPValues(const std::vector<double>& values);
    void setSD1Bands(const std::vector<double>& upper, const std::vector<double>& lower);
    void setSD2Bands(const std::vector<double>& upper, const std::vector<double>& lower);
    void setSD3Bands(const std::vector<double>& upper, const std::vector<double>& lower);

    // Add individual VWAP value with corresponding standard deviation bands
    void addVWAPValue(double vwap, double sd1Upper, double sd1Lower,
                      double sd2Upper, double sd2Lower,
                      double sd3Upper, double sd3Lower);

    // Clear all stored data
    void clear();

    // Get the count of stored values
    size_t size() const;

    // Calculate rolling VWAP for the last N bars using OHLCV bars
    void calculate(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars);

    // Async calculation using TaskScheduler
    void calculateAsync(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& bars,
                       std::shared_ptr<btq::TaskScheduler> taskScheduler);

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

// Inline implementations
inline RollingVWAP::RollingVWAP() : windowSize_(20) {} // Default window size of 20

inline RollingVWAP::RollingVWAP(size_t windowSize) : windowSize_(windowSize) {}

inline size_t RollingVWAP::getWindowSize() const {
    return windowSize_;
}

inline const std::vector<double>& RollingVWAP::getVWAPValues() const {
    return vwapValues_;
}

inline const std::vector<double>& RollingVWAP::getSD1UpperBand() const {
    return sd1UpperBand_;
}

inline const std::vector<double>& RollingVWAP::getSD1LowerBand() const {
    return sd1LowerBand_;
}

inline const std::vector<double>& RollingVWAP::getSD2UpperBand() const {
    return sd2UpperBand_;
}

inline const std::vector<double>& RollingVWAP::getSD2LowerBand() const {
    return sd2LowerBand_;
}

inline const std::vector<double>& RollingVWAP::getSD3UpperBand() const {
    return sd3UpperBand_;
}

inline const std::vector<double>& RollingVWAP::getSD3LowerBand() const {
    return sd3LowerBand_;
}

inline void RollingVWAP::setWindowSize(size_t windowSize) {
    windowSize_ = windowSize;
}

inline void RollingVWAP::setVWAPValues(const std::vector<double>& values) {
    vwapValues_ = values;
}

inline void RollingVWAP::setSD1Bands(const std::vector<double>& upper, const std::vector<double>& lower) {
    sd1UpperBand_ = upper;
    sd1LowerBand_ = lower;
}

inline void RollingVWAP::setSD2Bands(const std::vector<double>& upper, const std::vector<double>& lower) {
    sd2UpperBand_ = upper;
    sd2LowerBand_ = lower;
}

inline void RollingVWAP::setSD3Bands(const std::vector<double>& upper, const std::vector<double>& lower) {
    sd3UpperBand_ = upper;
    sd3LowerBand_ = lower;
}

inline void RollingVWAP::addVWAPValue(double vwap, double sd1Upper, double sd1Lower,
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

inline void RollingVWAP::clear() {
    vwapValues_.clear();
    sd1UpperBand_.clear();
    sd1LowerBand_.clear();
    sd2UpperBand_.clear();
    sd2LowerBand_.clear();
    sd3UpperBand_.clear();
    sd3LowerBand_.clear();
}

inline size_t RollingVWAP::size() const {
    return vwapValues_.size();
}

} // namespace btq