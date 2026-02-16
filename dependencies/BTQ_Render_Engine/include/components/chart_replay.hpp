#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "chart_manager.hpp"

namespace BTQuant {

struct ReplayConfig {
    double playback_speed = 1.0;  // 1.0 = real-time, 0.5 = half speed, 2.0 = double speed
    bool loop_enabled = false;
    uint64_t start_timestamp = 0;  // Start time for replay
    uint64_t end_timestamp = 0;    // End time for replay (0 = until latest)
    uint32_t symbol_id = 0;        // Symbol to replay
    RenderEngine::TimeFrame timeframe = RenderEngine::TimeFrame::TF_1MIN;  // Timeframe to replay
    std::string symbol_name = "";
    std::string exchange_name = "Binance";

    // Backtesting specific options
    bool enable_step_by_step = false;  // Allow stepping through bars one by one
    bool enable_manual_control = false; // Enable manual control of replay
    bool show_performance_metrics = true; // Show performance metrics during replay
};

struct BacktestMetrics {
    uint64_t total_bars_processed = 0;
    uint64_t current_bar_index = 0;
    uint64_t total_bars = 0;
    double progress_percentage = 0.0;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point current_time;
    double elapsed_seconds = 0.0;
    double replay_speed_actual = 0.0;  // Actual replay speed achieved
};

class ChartReplay {
public:
    // DEPRECATED - Legacy hotspine
    ChartReplay(std::shared_ptr<HotSpineDataBridge> bridge,
                std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                ChartManager* chart_manager);

    ~ChartReplay();

    // Configuration methods
    void set_replay_config(const ReplayConfig& config);
    ReplayConfig get_replay_config() const;

    // Control methods
    void start_replay();
    void pause_replay();
    void stop_replay();
    void reset_replay();
    void seek_to_time(uint64_t timestamp);
    void step_forward();  // Step to next bar (for manual control)
    void step_backward(); // Step to previous bar (for manual control)

    // Status methods
    bool is_playing() const { return is_playing_; }
    bool is_paused() const { return is_paused_; }
    bool is_stopped() const { return !is_playing_ && !is_paused_; }
    uint64_t get_current_time() const { return current_time_; }
    size_t get_current_bar_index() const { return current_bar_index_; }
    size_t get_total_bars() const { return historical_candles_.size(); }

    // UI rendering
    void render_replay_controls();

    // Data loading methods
    bool load_historical_data(const std::string& symbol, const std::string& exchange,
                             RenderEngine::TimeFrame timeframe,
                             uint64_t start_time, uint64_t end_time);

    // Load from CSV file
    bool load_from_csv(const std::string& csv_file_path,
                      const std::string& symbol,
                      RenderEngine::TimeFrame timeframe);

    // Load from market data processor
    bool load_from_market_data_processor(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                                        uint64_t start_time, uint64_t end_time);

    // Get performance metrics
    BacktestMetrics get_performance_metrics() const;

private:
    // Internal control methods
    void replay_thread_func();
    void process_next_bar();
    void update_chart_with_current_data();
    void update_performance_metrics();

    // Member variables
    // DEPRECATED - Legacy hotspine
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    ChartManager* chart_manager_;

    ReplayConfig config_;

    std::atomic<bool> is_playing_{false};
    std::atomic<bool> is_paused_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<uint64_t> current_time_{0};

    std::thread replay_thread_;
    mutable std::mutex config_mutex_;
    mutable std::mutex metrics_mutex_;

    // Historical data storage
    std::vector<RenderEngine::OHLCVCandle> historical_candles_;
    size_t current_bar_index_ = 0;

    // Timing control
    std::chrono::steady_clock::time_point last_update_time_;
    double accumulated_time_ = 0.0;

    // Chart ID for the replay chart
    uint32_t replay_chart_id_ = 0;

    // Performance metrics
    BacktestMetrics metrics_;
};

} // namespace BTQuant