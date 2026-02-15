#include "chart_super_node.hpp"
#include <algorithm>
#include <numeric>

namespace BTQuant {

ChartSuperNode::ChartSuperNode(std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : bridge_(bridge)
    , processor_(processor)
    , chart_manager_(std::make_unique<ChartManager>(bridge, processor))
    , tpo_engine_(std::make_unique<TPOEngine>())
    , last_update_time_(std::chrono::high_resolution_clock::now()) {
}

ChartSuperNode::~ChartSuperNode() {
    running_ = false;
    // Clean up resources
    anchored_vwaps_.clear();
    session_vwaps_.clear();
}

void ChartSuperNode::initialize() {
    if (initialized_.load()) {
        return;
    }
    
    // Initialize any required resources
    // Set up default TPO engine parameters
    tpo_engine_->clear();
    
    initialized_ = true;
    running_ = true;
}

void ChartSuperNode::update() {
    if (!running_.load() || !initialized_.load()) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process pending updates
    process_pending_updates();
    
    // Update all chart data
    auto& charts = chart_manager_->get_charts();
    for (const auto& pair : charts) {
        update_chart_data(pair.first);
    }
    
    // Update indicators for all charts
    for (const auto& pair : charts) {
        update_indicators_for_chart(pair.first);
    }
    
    // Update TPO data
    for (const auto& pair : charts) {
        update_tpo_data_for_chart(pair.first);
    }
    
    // Synchronize charts if needed
    synchronize_charts();
    
    // Cleanup stale resources periodically
    cleanup_stale_resources();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update performance metrics
    if (performance_monitoring_enabled_) {
        average_update_time_ms_ = (average_update_time_ms_ * update_count_ + duration) / (update_count_ + 1);
        update_count_++;
    }
}

uint32_t ChartSuperNode::create_chart(const std::string& symbol_name, 
                                      const std::string& exchange_name,
                                      uint32_t symbol_id, 
                                      RenderEngine::TimeFrame timeframe) {
    std::lock_guard<std::mutex> lock(mutex_);
    return chart_manager_->create_chart(symbol_name, exchange_name, symbol_id, timeframe);
}

void ChartSuperNode::destroy_chart(uint32_t chart_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    chart_manager_->destroy_chart(chart_id);
    
    // Remove associated VWAPs
    anchored_vwaps_.erase(chart_id);
    session_vwaps_.erase(chart_id);
}

void ChartSuperNode::update_chart_data(uint32_t chart_id) {
    if (!running_.load()) {
        return;
    }
    
    chart_manager_->populate_chart_data(chart_id);
}

void ChartSuperNode::set_chart_sync_callback(ChartSyncCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    chart_sync_callback_ = std::move(callback);
}

void ChartSuperNode::set_crosshair_sync_callback(CrosshairSyncCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    crosshair_sync_callback_ = std::move(callback);
}

void ChartSuperNode::broadcast_chart_sync(const SyncData& sync_data) {
    if (chart_sync_callback_) {
        chart_sync_callback_(sync_data);
    }
}

void ChartSuperNode::broadcast_crosshair_sync(const CrosshairSyncData& crosshair_data) {
    if (crosshair_sync_callback_) {
        crosshair_sync_callback_(crosshair_data);
    }
}

void ChartSuperNode::calculate_indicator(const std::string& indicator_name, 
                                       uint32_t chart_id, 
                                       const IndicatorParams& params) {
    // This is a simplified version - in a real implementation, this would
    // perform the actual indicator calculation
    std::vector<double> result;
    
    // Example: Calculate a simple moving average
    if (indicator_name == "SMA") {
        auto chart_instance = get_chart_instance(chart_id);
        if (chart_instance && !chart_instance->closes.empty()) {
            result.resize(chart_instance->closes.size());
            
            if (chart_instance->closes.size() >= static_cast<size_t>(params.period)) {
                for (size_t i = params.period - 1; i < chart_instance->closes.size(); ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < params.period; ++j) {
                        sum += chart_instance->closes[i - j];
                    }
                    result[i] = sum / params.period;
                }
            }
        }
    }
    
    // Call the registered callback if available
    if (indicator_callback_) {
        indicator_callback_(indicator_name, result);
    }
}

void ChartSuperNode::calculate_indicator(const std::string& indicator_name, uint32_t chart_id) {
    // Call the main implementation with default parameters
    calculate_indicator(indicator_name, chart_id, IndicatorParams());
}

void ChartSuperNode::register_indicator_calculation_callback(IndicatorCalculationCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    indicator_callback_ = std::move(callback);
}

TPOEngine& ChartSuperNode::get_tpo_engine() {
    return *tpo_engine_;
}

const TPOEngine& ChartSuperNode::get_tpo_engine() const {
    return *tpo_engine_;
}

void ChartSuperNode::add_anchored_vwap(uint32_t chart_id, uint64_t anchor_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create a new anchored VWAP and add it to the list for this chart
    ::btq::AnchoredVWAP new_vwap(anchor_time);
    anchored_vwaps_[chart_id].push_back(new_vwap);
}

void ChartSuperNode::update_session_vwap(uint32_t chart_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update the session VWAP for this chart
    auto it = session_vwaps_.find(chart_id);
    if (it == session_vwaps_.end()) {
        session_vwaps_[chart_id] = ::btq::SessionVWAP();
        it = session_vwaps_.find(chart_id);
    }
    
    // Update with current market data
    // This is a simplified version - in practice, you'd get the latest trade data
    // and update the VWAP calculation
}

const std::unordered_map<uint32_t, ChartInstance>& ChartSuperNode::get_all_charts() const {
    return chart_manager_->get_charts();
}

std::shared_ptr<ChartInstance> ChartSuperNode::get_chart_instance(uint32_t chart_id) {
    const auto& charts = chart_manager_->get_charts();
    auto it = charts.find(chart_id);
    if (it != charts.end()) {
        return std::make_shared<ChartInstance>(it->second);
    }
    return nullptr;
}

void ChartSuperNode::enable_performance_monitoring(bool enabled) {
    performance_monitoring_enabled_ = enabled;
    if (!enabled) {
        average_update_time_ms_ = 0.0;
        update_count_ = 0;
    }
}

double ChartSuperNode::get_average_update_time() const {
    return average_update_time_ms_;
}

size_t ChartSuperNode::get_active_chart_count() const {
    return chart_manager_->get_charts().size();
}

void ChartSuperNode::lock() {
    mutex_.lock();
}

void ChartSuperNode::unlock() {
    mutex_.unlock();
}

bool ChartSuperNode::try_lock() {
    return mutex_.try_lock();
}

void ChartSuperNode::process_pending_updates() {
    // Process any pending updates or events
    // This could include incoming market data, user interactions, etc.
}

void ChartSuperNode::update_indicators_for_chart(uint32_t chart_id) {
    // Calculate and cache indicators for this chart
    calculate_and_cache_indicators(chart_id);
}

void ChartSuperNode::calculate_and_cache_indicators(uint32_t chart_id) {
    // In a real implementation, this would calculate various indicators
    // and cache them for efficient retrieval
    
    // Example: Calculate and cache common indicators
    IndicatorParams params;
    
    // Calculate SMA
    calculate_indicator("SMA", chart_id, params);
    
    // Additional indicators would be calculated here
}

void ChartSuperNode::update_tpo_data_for_chart(uint32_t chart_id) {
    // Update TPO data for this chart based on current market activity
    auto chart_instance = get_chart_instance(chart_id);
    if (!chart_instance) {
        return;
    }
    
    // Convert chart data to PriceTick format for TPO processing
    std::vector<PriceTick> ticks;
    for (size_t i = 0; i < chart_instance->dates.size(); ++i) {
        PriceTick tick;
        tick.timestamp = std::chrono::system_clock::time_point(std::chrono::milliseconds(static_cast<long long>(chart_instance->dates[i])));
        tick.price = chart_instance->closes[i];
        tick.volume = chart_instance->volumes[i];
        ticks.push_back(tick);
    }
    
    // Process the ticks through the TPO engine
    tpo_engine_->process_ticks(ticks);
}

void ChartSuperNode::synchronize_charts() {
    // Implement chart synchronization logic
    // This could include time axis alignment, price scaling, etc.
}

void ChartSuperNode::cleanup_stale_resources() {
    // Periodically clean up any stale resources
    // This could include old VWAPs, expired indicators, etc.
}

} // namespace BTQuant