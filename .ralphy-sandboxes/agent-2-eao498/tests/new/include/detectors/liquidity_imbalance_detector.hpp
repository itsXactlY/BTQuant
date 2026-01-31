#pragma once

#include "hotspine_extended_reader.hpp"
#include <optional>
#include <vector>
#include <string>

namespace BTQuant {

enum class ImbalanceDirection {
    LONG,
    SHORT,
    NEUTRAL
};

struct LiquiditySignal {
    std::string symbol;
    std::string thin_exchange;
    std::string thick_exchange;
    double thin_depth;
    double thick_depth;
    double depth_ratio;  // thick/thin
    double thin_spread_bps;
    double thick_spread_bps;
    ImbalanceDirection direction;
    uint64_t timestamp_us;
    
    std::string to_string() const;
};

class LiquidityImbalanceDetector {
public:
    explicit LiquidityImbalanceDetector(HotSpineExtendedReader& reader);
    
    // Detect liquidity imbalances
    std::optional<LiquiditySignal> detect(const std::string& symbol);
    
    // Configuration
    void set_depth_ratio_threshold(double ratio) { depth_ratio_threshold_ = ratio; }
    void set_imbalance_threshold(double threshold) { imbalance_threshold_ = threshold; }
    
    // Statistics
    uint64_t get_detections() const { return detections_; }
    void reset_statistics() { detections_ = 0; }
    
private:
    HotSpineExtendedReader& reader_;
    double depth_ratio_threshold_ = 3.0;     // 3x depth difference
    double imbalance_threshold_ = 0.3;       // 30% bid/ask imbalance
    uint64_t detections_ = 0;
    
    double calculate_orderbook_depth(const OrderbookData& ob, double bps_range = 100.0);
    double calculate_bid_ask_imbalance(const OrderbookData& ob);
};

} // namespace BTQuant