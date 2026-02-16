#include "liquiditysweepdetector.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

LiquiditySweepDetector::LiquiditySweepDetector(double massive_liquidity_threshold, 
                                               double sweep_detection_ratio, 
                                               int comparison_window_seconds, 
                                               double price_bucket_size)
    : massive_liquidity_threshold_(massive_liquidity_threshold),
      sweep_detection_ratio_(sweep_detection_ratio),
      comparison_window_seconds_(comparison_window_seconds),
      price_bucket_size_(price_bucket_size),
      overlay_radius_(10.0),
      overlay_red_(1.0f), overlay_green_(1.0f), overlay_blue_(1.0f), overlay_alpha_(1.0f) {}

void LiquiditySweepDetector::process_liquidity_snapshot(const LiquiditySnapshot& snapshot) {
    liquidity_history_.push_back(snapshot);
    
    // Keep only recent snapshots within the comparison window
    auto now = snapshot.timestamp;
    auto cutoff_time = now - std::chrono::seconds(comparison_window_seconds_);
    
    liquidity_history_.erase(
        std::remove_if(liquidity_history_.begin(), liquidity_history_.end(),
            [cutoff_time](const LiquiditySnapshot& snap) {
                return snap.timestamp < cutoff_time;
            }),
        liquidity_history_.end());
}

void LiquiditySweepDetector::process_liquidity_snapshots(const std::vector<LiquiditySnapshot>& snapshots) {
    for (const auto& snapshot : snapshots) {
        process_liquidity_snapshot(snapshot);
    }
}

void LiquiditySweepDetector::detect_sweeps() {
    // Clear previous detections
    detected_sweeps_.clear();
    
    if (liquidity_history_.size() < 2) {
        return; // Need at least 2 snapshots to compare
    }
    
    // Group snapshots by price level for comparison
    std::map<double, std::vector<LiquiditySnapshot>> grouped_by_price;
    
    for (const auto& snapshot : liquidity_history_) {
        double price_bucket = std::floor(snapshot.price_level / price_bucket_size_) * price_bucket_size_;
        grouped_by_price[price_bucket].push_back(snapshot);
    }
    
    // For each price level, look for significant drops in liquidity
    for (auto& [price_level, snapshots] : grouped_by_price) {
        // Sort snapshots by time
        std::sort(snapshots.begin(), snapshots.end(),
                  [](const LiquiditySnapshot& a, const LiquiditySnapshot& b) {
                      return a.timestamp < b.timestamp;
                  });
        
        // Compare consecutive snapshots for liquidity drops
        for (size_t i = 1; i < snapshots.size(); ++i) {
            const auto& prev = snapshots[i-1];
            const auto& curr = snapshots[i];
            
            // Check for bid liquidity sweep
            if (prev.bid_volume > massive_liquidity_threshold_ && 
                curr.bid_volume < prev.bid_volume) {
                
                double drop_ratio = (prev.bid_volume - curr.bid_volume) / prev.bid_volume;
                if (drop_ratio >= sweep_detection_ratio_) {
                    LiquiditySweepEvent sweep(curr.timestamp, price_level, 
                                           prev.bid_volume, curr.bid_volume, true);
                    detected_sweeps_.push_back(sweep);
                }
            }
            
            // Check for ask liquidity sweep
            if (prev.ask_volume > massive_liquidity_threshold_ && 
                curr.ask_volume < prev.ask_volume) {
                
                double drop_ratio = (prev.ask_volume - curr.ask_volume) / prev.ask_volume;
                if (drop_ratio >= sweep_detection_ratio_) {
                    LiquiditySweepEvent sweep(curr.timestamp, price_level, 
                                           prev.ask_volume, curr.ask_volume, false);
                    detected_sweeps_.push_back(sweep);
                }
            }
        }
    }
}

const std::vector<LiquiditySweepEvent>& LiquiditySweepDetector::get_detected_sweeps() const {
    return detected_sweeps_;
}

std::vector<LiquiditySweepEvent> LiquiditySweepDetector::get_sweeps_in_range(
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time) const {
    
    std::vector<LiquiditySweepEvent> result;
    
    for (const auto& sweep : detected_sweeps_) {
        if (sweep.timestamp >= start_time && sweep.timestamp <= end_time) {
            result.push_back(sweep);
        }
    }
    
    return result;
}

void LiquiditySweepDetector::clear() {
    liquidity_history_.clear();
    detected_sweeps_.clear();
}

void LiquiditySweepDetector::print_sweeps() const {
    std::cout << "Detected Liquidity Sweeps:\n";
    
    if (detected_sweeps_.empty()) {
        std::cout << "No liquidity sweeps detected.\n";
        return;
    }
    
    for (const auto& sweep : detected_sweeps_) {
        auto time_t = std::chrono::system_clock::to_time_t(sweep.timestamp);
        std::cout << "Time: " << std::put_time(std::localtime(&time_t), "%F %T")
                  << ", Price: " << std::fixed << std::setprecision(2) << sweep.price_level
                  << ", Type: " << sweep.description
                  << ", Volume Before: " << sweep.volume_before
                  << ", Volume After: " << sweep.volume_after
                  << ", Swept: " << sweep.swept_volume << "\n";
    }
}

void LiquiditySweepDetector::set_massive_liquidity_threshold(double threshold) {
    massive_liquidity_threshold_ = threshold;
}

void LiquiditySweepDetector::set_sweep_detection_ratio(double ratio) {
    sweep_detection_ratio_ = ratio;
}

void LiquiditySweepDetector::set_comparison_window_seconds(int seconds) {
    comparison_window_seconds_ = seconds;
}

double LiquiditySweepDetector::get_massive_liquidity_threshold() const {
    return massive_liquidity_threshold_;
}

double LiquiditySweepDetector::get_sweep_detection_ratio() const {
    return sweep_detection_ratio_;
}

int LiquiditySweepDetector::get_comparison_window_seconds() const {
    return comparison_window_seconds_;
}

std::vector<LiquiditySweepDetector::VisualSweepOverlay> LiquiditySweepDetector::get_visual_overlays() const {
    std::vector<VisualSweepOverlay> overlays;
    
    for (const auto& sweep : detected_sweeps_) {
        VisualSweepOverlay overlay;
        
        // Convert timestamp to a normalized x position (this would be calculated based on chart dimensions)
        // For now, we'll use a simple conversion assuming a reference start time
        auto duration_since_epoch = sweep.timestamp.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration_since_epoch).count();
        overlay.x_position = static_cast<double>(seconds);
        
        // Y position is the price level where the sweep occurred
        overlay.y_position = sweep.price_level;
        
        // Set the radius and color based on configuration
        overlay.radius = overlay_radius_;
        overlay.red = overlay_red_;
        overlay.green = overlay_green_;
        overlay.blue = overlay_blue_;
        overlay.alpha = overlay_alpha_;
        
        // Set type based on sweep type (bid or ask)
        overlay.type = sweep.is_bid_sweep ? "hollow_circle_bid" : "hollow_circle_ask";
        
        overlays.push_back(overlay);
    }
    
    return overlays;
}

void LiquiditySweepDetector::set_overlay_radius(double radius) {
    overlay_radius_ = radius;
}

void LiquiditySweepDetector::set_overlay_color(float r, float g, float b, float a) {
    overlay_red_ = r;
    overlay_green_ = g;
    overlay_blue_ = b;
    overlay_alpha_ = a;
}