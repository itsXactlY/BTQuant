#pragma once

#include <imgui.h>
#include <vector>
#include <map>
#include <deque>
#include <chrono>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

// Structure to represent a snapshot of the order book at a specific time
struct OrderbookSnapshot {
    uint64_t timestamp;                    // Timestamp of the snapshot
    uint32_t symbol_id;                    // Symbol identifier
    std::vector<PriceLevel> bids;          // Bid levels
    std::vector<PriceLevel> asks;          // Ask levels
    double spread;                         // Current spread
    double imbalance;                      // Current imbalance
    
    OrderbookSnapshot() : timestamp(0), symbol_id(0), spread(0.0), imbalance(0.0) {}
    
    OrderbookSnapshot(const RenderEngine::OrderbookData& data, uint32_t sym_id) 
        : timestamp(std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::high_resolution_clock::now().time_since_epoch()).count()),
          symbol_id(sym_id),
          bids(data.bids),
          asks(data.asks),
          spread(data.spread),
          imbalance(data.imbalance) {}
};

// Component for managing historical order book snapshots
class OrderbookHistoryPanel : public PanelBase {
public:
    OrderbookHistoryPanel(const PanelConfig& config,
                         std::shared_ptr<HotSpineDataBridge> bridge,
                         std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

    void update(float dt) override;
    void render_content() override;

    // Capture a snapshot of the current order book state
    void captureSnapshot(uint32_t symbol_id, const std::string& symbol_name);
    
    // Playback controls
    void startPlayback();
    void pausePlayback();
    void stopPlayback();
    void setPlaybackSpeed(float speed);
    
    // Navigation controls
    void goToPreviousSnapshot();
    void goToNextSnapshot();
    void goToSnapshot(int index);
    
    // Configuration
    void setCaptureIntervalMs(int interval_ms);  // Capture interval in milliseconds
    void setMaxSnapshots(size_t max_snapshots);  // Maximum number of snapshots to store
    
    // Comparison functions
    void enableComparison(bool enable);
    void setComparisonSnapshot(int index);
    
    // Getters
    size_t getSnapshotCount() const { return snapshots_.size(); }
    int getCurrentSnapshotIndex() const { return current_snapshot_index_; }
    bool isPlaying() const { return is_playing_; }

private:
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    
    // Snapshot storage
    std::deque<OrderbookSnapshot> snapshots_;
    int current_snapshot_index_ = -1;
    
    // Configuration
    int capture_interval_ms_ = 1000;  // Default: capture every 1 second
    size_t max_snapshots_ = 1000;     // Default: keep up to 1000 snapshots
    uint64_t last_capture_time_ = 0;  // Timestamp of last capture
    
    // Playback controls
    bool is_playing_ = false;
    float playback_speed_ = 1.0f;
    uint64_t last_playback_time_ = 0;
    
    // Comparison functionality
    bool enable_comparison_ = false;
    int comparison_snapshot_index_ = -1;
    
    // UI state
    uint32_t symbol_id_ = 0;
    std::string symbol_name_ = "BTC-USDT";
    int selected_levels_count_ = 20;
    static constexpr int LEVEL_OPTIONS[] = {10, 20, 50, 100, 500, -1};
    static constexpr const char* LEVEL_OPTION_NAMES[] = {"10", "20", "50", "100", "500", "Unlimited"};
    
    // Private helper methods
    int get_level_option_index();
    void render_orderbook_ladder(const std::vector<PriceLevel>& bids, 
                                const std::vector<PriceLevel>& asks,
                                const std::string& title = "");
    void render_comparison_view();
    void cleanupOldSnapshots();
    
    // Data processing helpers
    double calculateAverageOrderSize(const std::vector<PriceLevel>& bids, 
                                   const std::vector<PriceLevel>& asks) const;
    double getMaxVolume(const std::vector<PriceLevel>& bids, 
                       const std::vector<PriceLevel>& asks) const;
};

}  // namespace BTQuant