#include "detectors/spoofing_detector.hpp"
#include "detectors/detection_signal.hpp"
#include <sstream>
#include <iomanip>

namespace BTQuant {

SpoofingDetector::SpoofingDetector(HotSpineExtendedReader& reader)
    : reader_(reader),
      min_cancel_count_(3),
      max_duration_ms_(5000.0),
      min_size_(0.1),
      detections_(0) {

    // Initialize with default configuration
    // In a real implementation, we would load from config
}

std::string SpoofingDetector::make_key(const std::string& exchange,
                                        const std::string& symbol) const {
    return exchange + ":" + symbol;
}

void SpoofingDetector::update_orderbook(const std::string& exchange,
                                         const std::string& symbol) {
    auto ob = reader_.get_latest_orderbook(exchange, symbol);
    if (!ob) return;
    
    uint64_t now = reader_.get_current_time_us();
    std::string key = make_key(exchange, symbol);
    
    // Track current bids
    std::set<double> current_bids;
    for (const auto& [price, size] : ob->bids) {
        current_bids.insert(price);
        
        auto& level = bid_history_[key][price];
        if (level.first_seen_us == 0) {
            level.first_seen_us = now;
            level.price = price;
        }
        level.last_seen_us = now;
        level.size = size;
    }
    
    // Detect cancelled bids
    for (auto it = bid_history_[key].begin(); it != bid_history_[key].end();) {
        if (current_bids.find(it->first) == current_bids.end()) {
            // Level disappeared
            it->second.cancel_count++;
            
            // Remove old levels (>10 seconds)
            if (now - it->second.last_seen_us > 10'000'000) {
                it = bid_history_[key].erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
    
    // Same for asks
    std::set<double> current_asks;
    for (const auto& [price, size] : ob->asks) {
        current_asks.insert(price);
        
        auto& level = ask_history_[key][price];
        if (level.first_seen_us == 0) {
            level.first_seen_us = now;
            level.price = price;
        }
        level.last_seen_us = now;
        level.size = size;
    }
    
    for (auto it = ask_history_[key].begin(); it != ask_history_[key].end();) {
        if (current_asks.find(it->first) == current_asks.end()) {
            it->second.cancel_count++;
            
            if (now - it->second.last_seen_us > 10'000'000) {
                it = ask_history_[key].erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

std::optional<SpoofingSignal> SpoofingDetector::detect(const std::string& exchange,
                                                        const std::string& symbol) {
    std::string key = make_key(exchange, symbol);
    uint64_t now = reader_.get_current_time_us();
    
    // Check bid history for spoofing patterns
    for (const auto& [price, level] : bid_history_[key]) {
        double duration_ms = (level.last_seen_us - level.first_seen_us) / 1000.0;
        
        if (level.cancel_count >= min_cancel_count_ &&
            duration_ms < max_duration_ms_ &&
            level.size >= min_size_) {
            
            SpoofingSignal signal;
            signal.symbol = symbol;
            signal.exchange = exchange;
            signal.is_bid_spoof = true;
            signal.spoof_price = price;
            signal.spoof_size = level.size;
            signal.spoof_duration_ms = duration_ms;
            signal.cancel_count = level.cancel_count;
            signal.timestamp_us = now;
            
            if (level.cancel_count >= 5) {
                signal.confidence = SpoofConfidence::HIGH;
            } else if (level.cancel_count >= 4) {
                signal.confidence = SpoofConfidence::MEDIUM;
            } else {
                signal.confidence = SpoofConfidence::LOW;
            }
            
            detections_++;
            return signal;
        }
    }
    
    // Check ask history
    for (const auto& [price, level] : ask_history_[key]) {
        double duration_ms = (level.last_seen_us - level.first_seen_us) / 1000.0;
        
        if (level.cancel_count >= min_cancel_count_ &&
            duration_ms < max_duration_ms_ &&
            level.size >= min_size_) {
            
            SpoofingSignal signal;
            signal.symbol = symbol;
            signal.exchange = exchange;
            signal.is_bid_spoof = false;
            signal.spoof_price = price;
            signal.spoof_size = level.size;
            signal.spoof_duration_ms = duration_ms;
            signal.cancel_count = level.cancel_count;
            signal.timestamp_us = now;
            
            if (level.cancel_count >= 5) {
                signal.confidence = SpoofConfidence::HIGH;
            } else if (level.cancel_count >= 4) {
                signal.confidence = SpoofConfidence::MEDIUM;
            } else {
                signal.confidence = SpoofConfidence::LOW;
            }
            
            detections_++;
            return signal;
        }
    }
    
    return std::nullopt;
}

} // namespace BTQuant::Detectors

namespace BTQuant {

std::string SpoofingSignal::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Spoofing(symbol=" << symbol
        << ", exchange=" << exchange
        << ", side=" << (is_bid_spoof ? "BID" : "ASK")
        << ", price=" << spoof_price
        << ", size=" << spoof_size
        << ", cancels=" << cancel_count
        << ", confidence=";
    
    switch (confidence) {
        case SpoofConfidence::LOW: oss << "LOW"; break;
        case SpoofConfidence::MEDIUM: oss << "MEDIUM"; break;
        case SpoofConfidence::HIGH: oss << "HIGH"; break;
    }
    
    oss << ")";
    return oss.str();
}

} // namespace BTQuant