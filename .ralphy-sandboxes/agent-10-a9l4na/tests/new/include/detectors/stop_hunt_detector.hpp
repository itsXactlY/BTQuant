#pragma once

#include "hotspine_extended_reader.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <optional>

namespace BTQuant {

struct StopHuntSignal {
    std::string symbol;
    std::string hunt_exchange;
    double hunt_price;
    double median_price;
    double hunt_deviation_pct;
    std::vector<std::pair<std::string, double>> stable_exchanges;
    uint64_t timestamp_us;
    bool is_long_signal;  // true = price dumped on one exchange, fade long
    
    std::string to_string() const;
};

class StopHuntDetector {
public:
    explicit StopHuntDetector(HotSpineExtendedReader& reader);
    
    // Detect stop hunts for a specific symbol
    std::optional<StopHuntSignal> detect(const std::string& symbol);
    
    // Configuration - load from config system
    void configure(const std::string& config_path);
    
    // Configuration setters
    void set_threshold_pct(double threshold) { threshold_pct_ = threshold; }
    void set_min_exchanges(size_t min_exchanges) { min_exchanges_ = min_exchanges; }
    
    // Statistics
    uint64_t get_detections() const { return detections_; }
    void reset_statistics() { detections_ = 0; }
    
private:
    HotSpineExtendedReader& reader_;
    std::optional<double> threshold_pct_;  // Configurable: deviation threshold
    std::optional<size_t> min_exchanges_;  // Configurable: minimum exchanges needed
    uint64_t detections_ = 0;
    
    double calculate_median(std::vector<double>& values);
};

} // namespace BTQuant