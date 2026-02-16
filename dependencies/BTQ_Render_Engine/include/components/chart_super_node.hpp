#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <chrono>

#include "chart_manager.hpp"
#include "chart_panel.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "indicators/anchored_vwap.hpp"
#include "indicators/session_vwap.hpp"
#include "analytics/tpoengine.h"

namespace BTQuant {

// Central node that coordinates all chart-related activities
class ChartSuperNode {
public:
    // Structure to hold chart synchronization data
    struct SyncData {
        uint64_t start_timestamp = 0;
        uint64_t end_timestamp = 0;
        double min_price = 0.0;
        double max_price = 0.0;
        bool is_locked = false;
        
        SyncData() = default;
        SyncData(uint64_t start, uint64_t end, double min_p, double max_p, bool locked = false)
            : start_timestamp(start), end_timestamp(end), min_price(min_p), max_price(max_p), is_locked(locked) {}
    };

    // Structure to hold crosshair synchronization data
    struct CrosshairSyncData {
        double x_position = 0.0;
        double y_position = 0.0;
        bool is_active = false;
        uint32_t source_chart_id = 0;
        
        CrosshairSyncData() = default;
        CrosshairSyncData(double x, double y, bool active, uint32_t source_chart)
            : x_position(x), y_position(y), is_active(active), source_chart_id(source_chart) {}
    };

    // Structure to hold indicator calculation parameters
    struct IndicatorParams {
        int period = 14;
        double multiplier = 2.0;
        int fast_period = 12;
        int slow_period = 26;
        int signal_period = 9;
        int rsi_period = 14;
        double rsi_overbought = 70.0;
        double rsi_oversold = 30.0;
        
        // Default constructor
        IndicatorParams() = default;
        
        // Constructor with parameters
        IndicatorParams(int p, double m, int fp, int sp, int sigp, int rsp, double rsio, double rsis)
            : period(p), multiplier(m), fast_period(fp), slow_period(sp), 
              signal_period(sigp), rsi_period(rsp), rsi_overbought(rsio), rsi_oversold(rsis) {}
    };

    // Callback types for various events
    using ChartSyncCallback = std::function<void(const SyncData&)>;
    using CrosshairSyncCallback = std::function<void(const CrosshairSyncData&)>;
    using DataUpdateCallback = std::function<void(uint32_t chart_id)>;
    using IndicatorCalculationCallback = std::function<void(const std::string&, const std::vector<double>&)>;

    // DEPRECATED - Legacy hotspine
    explicit ChartSuperNode(std::shared_ptr<HotSpineDataBridge> bridge,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

    ~ChartSuperNode();

    // Initialize the super node
    void initialize();

    // Main update loop - called regularly to process updates
    void update();

    // Chart management
    uint32_t create_chart(const std::string& symbol_name, const std::string& exchange_name,
                         uint32_t symbol_id, RenderEngine::TimeFrame timeframe);
    void destroy_chart(uint32_t chart_id);
    void update_chart_data(uint32_t chart_id);

    // Synchronization methods
    void set_chart_sync_callback(ChartSyncCallback callback);
    void set_crosshair_sync_callback(CrosshairSyncCallback callback);
    void broadcast_chart_sync(const SyncData& sync_data);
    void broadcast_crosshair_sync(const CrosshairSyncData& crosshair_data);

    // Indicator management
    void calculate_indicator(const std::string& indicator_name, uint32_t chart_id,
                           const IndicatorParams& params);
    void calculate_indicator(const std::string& indicator_name, uint32_t chart_id);
    void register_indicator_calculation_callback(IndicatorCalculationCallback callback);

    // TPO Engine integration
    TPOEngine& get_tpo_engine();
    const TPOEngine& get_tpo_engine() const;

    // VWAP management
    void add_anchored_vwap(uint32_t chart_id, uint64_t anchor_time);
    void update_session_vwap(uint32_t chart_id);
    
    // Data access methods
    const std::unordered_map<uint32_t, ChartInstance>& get_all_charts() const;
    std::shared_ptr<ChartInstance> get_chart_instance(uint32_t chart_id);
    
    // Performance monitoring
    void enable_performance_monitoring(bool enabled);
    double get_average_update_time() const;
    size_t get_active_chart_count() const;

    // Thread safety methods
    void lock();
    void unlock();
    bool try_lock();
    
    // Advanced synchronization methods
    template<typename Func>
    auto execute_with_lock(Func&& func) -> decltype(func()) {
        std::lock_guard<std::mutex> lock(mutex_);
        return func();
    }
    
    // Atomic operations for thread-safe state management
    bool is_initialized() const { return initialized_.load(std::memory_order_acquire); }
    bool is_running() const { return running_.load(std::memory_order_acquire); }
    void set_running(bool running) { running_.store(running, std::memory_order_release); }

private:
    // Internal data structures
    // DEPRECATED - Legacy hotspine
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::unique_ptr<ChartManager> chart_manager_;
    
    // Synchronization callbacks
    ChartSyncCallback chart_sync_callback_;
    CrosshairSyncCallback crosshair_sync_callback_;
    IndicatorCalculationCallback indicator_callback_;
    
    // TPO engine for advanced analytics
    std::unique_ptr<TPOEngine> tpo_engine_;
    
    // VWAP instances
    std::unordered_map<uint32_t, std::list<::btq::AnchoredVWAP>> anchored_vwaps_;
    std::unordered_map<uint32_t, ::btq::SessionVWAP> session_vwaps_;
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point last_update_time_;
    double average_update_time_ms_ = 0.0;
    size_t update_count_ = 0;
    bool performance_monitoring_enabled_ = false;
    
    // Thread safety
    mutable std::mutex mutex_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    
    // Internal helper methods
    void process_pending_updates();
    void update_indicators_for_chart(uint32_t chart_id);
    void calculate_and_cache_indicators(uint32_t chart_id);
    void update_tpo_data_for_chart(uint32_t chart_id);
    void synchronize_charts();
    void cleanup_stale_resources();
};

} // namespace BTQuant