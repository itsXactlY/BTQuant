#ifndef PUBBTQUANT_LIQUIDITYSWEEPDETECTOR_H
#define PUBBTQUANT_LIQUIDITYSWEEPDETECTOR_H

#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>

// Structure to represent a liquidity sweep event
struct LiquiditySweepEvent {
    std::chrono::system_clock::time_point timestamp;
    double price_level;
    double volume_before;
    double volume_after;
    double swept_volume;
    bool is_bid_sweep;  // true for bid liquidity removal, false for ask liquidity removal
    std::string description;
    
    LiquiditySweepEvent()
        : timestamp(), price_level(0.0), volume_before(0.0), 
          volume_after(0.0), swept_volume(0.0), is_bid_sweep(true), description("") {}
    
    LiquiditySweepEvent(std::chrono::system_clock::time_point ts, double price, 
                       double vol_before, double vol_after, bool is_bid)
        : timestamp(ts), price_level(price), volume_before(vol_before), 
          volume_after(vol_after), swept_volume(vol_before - vol_after), 
          is_bid_sweep(is_bid) {
        description = is_bid ? "Bid Liquidity Sweep" : "Ask Liquidity Sweep";
    }
};

// Structure to represent liquidity at a price level over time
struct LiquiditySnapshot {
    std::chrono::system_clock::time_point timestamp;
    double price_level;
    double bid_volume;
    double ask_volume;
    int order_count;
    
    LiquiditySnapshot() : timestamp(), price_level(0.0), bid_volume(0.0), 
                         ask_volume(0.0), order_count(0) {}
    
    LiquiditySnapshot(std::chrono::system_clock::time_point ts, double price, 
                     double bid_vol, double ask_vol, int orders = 0)
        : timestamp(ts), price_level(price), bid_volume(bid_vol), 
          ask_volume(ask_vol), order_count(orders) {}
};

class LiquiditySweepDetector {
private:
    // Threshold for what constitutes "massive" liquidity
    double massive_liquidity_threshold_;
    
    // Minimum ratio change to trigger a sweep detection
    double sweep_detection_ratio_;
    
    // Time window for comparing liquidity snapshots (in seconds)
    int comparison_window_seconds_;
    
    // Historical liquidity snapshots
    std::vector<LiquiditySnapshot> liquidity_history_;
    
    // Detected sweep events
    std::vector<LiquiditySweepEvent> detected_sweeps_;
    
    // Price bucket size for grouping price levels
    double price_bucket_size_;

public:
    explicit LiquiditySweepDetector(double massive_liquidity_threshold = 10000.0, 
                                  double sweep_detection_ratio = 0.5, 
                                  int comparison_window_seconds = 300,  // 5 minutes
                                  double price_bucket_size = 0.25);

    // Process a liquidity snapshot and detect potential sweeps
    void process_liquidity_snapshot(const LiquiditySnapshot& snapshot);

    // Process multiple liquidity snapshots
    void process_liquidity_snapshots(const std::vector<LiquiditySnapshot>& snapshots);

    // Detect sweeps by comparing current liquidity with historical data
    void detect_sweeps();

    // Get detected sweep events
    const std::vector<LiquiditySweepEvent>& get_detected_sweeps() const;

    // Get sweep events within a specific time range
    std::vector<LiquiditySweepEvent> get_sweeps_in_range(
        const std::chrono::system_clock::time_point& start_time,
        const std::chrono::system_clock::time_point& end_time) const;

    // Clear all stored data
    void clear();

    // Print detected sweeps for debugging
    void print_sweeps() const;

    // Setters for configuration parameters
    void set_massive_liquidity_threshold(double threshold);
    void set_sweep_detection_ratio(double ratio);
    void set_comparison_window_seconds(int seconds);
    
    // Getters for configuration parameters
    double get_massive_liquidity_threshold() const;
    double get_sweep_detection_ratio() const;
    int get_comparison_window_seconds() const;
    
    // Visualization methods for drawing overlays where liquidity vanished
    // These methods would typically interface with the rendering system
    
    // Get sweep events formatted for visualization overlay
    struct VisualSweepOverlay {
        double x_position;      // X coordinate (typically time-based)
        double y_position;      // Y coordinate (typically price-based)
        double radius;          // Size of the overlay
        float red, green, blue, alpha;  // Color values
        std::string type;       // Type of overlay (circle, line, etc.)
    };
    
    std::vector<VisualSweepOverlay> get_visual_overlays() const;
    
    // Set visualization parameters
    void set_overlay_radius(double radius);
    void set_overlay_color(float r, float g, float b, float a = 1.0f);
    
private:
    // Visualization parameters
    double overlay_radius_;
    float overlay_red_, overlay_green_, overlay_blue_, overlay_alpha_;
};

#endif // PUBBTQUANT_LIQUIDITYSWEEPDETECTOR_H