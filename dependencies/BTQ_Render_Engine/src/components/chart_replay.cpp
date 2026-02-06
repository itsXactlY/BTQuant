#include "../../include/components/chart_replay.hpp"

#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

#include "imgui.h"
#include "implot.h"
#include "../../include/ui/tooltips.hpp"

namespace BTQuant {

ChartReplay::ChartReplay(std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                         ChartManager* chart_manager)
    : processor_(processor), chart_manager_(chart_manager) {
    // Initialize with default configuration
    config_.playback_speed = 1.0;
    config_.loop_enabled = false;
    config_.start_timestamp = 0;
    config_.end_timestamp = 0;
    config_.symbol_id = 0;
    config_.timeframe = RenderEngine::TimeFrame::TF_1MIN;
    config_.symbol_name = "BTC-USDT";
    config_.exchange_name = "Binance";
    config_.enable_step_by_step = false;
    config_.enable_manual_control = false;
    config_.show_performance_metrics = true;

    // Initialize metrics
    metrics_.start_time = std::chrono::steady_clock::now();
    metrics_.current_time = metrics_.start_time;
}

ChartReplay::~ChartReplay() {
    stop_replay();
    if (replay_thread_.joinable()) {
        replay_thread_.join();
    }
}

void ChartReplay::set_replay_config(const ReplayConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = config;
}

ReplayConfig ChartReplay::get_replay_config() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

void ChartReplay::start_replay() {
    if (is_playing_) {
        return; // Already playing
    }
    
    if (historical_candles_.empty()) {
        std::cerr << "No historical data loaded for replay!" << std::endl;
        return;
    }
    
    if (current_bar_index_ >= historical_candles_.size()) {
        // Reset to beginning if we've reached the end
        current_bar_index_ = 0;
    }
    
    is_playing_ = true;
    is_paused_ = false;
    should_stop_ = false;
    
    // Create a chart for replay if not already created
    if (replay_chart_id_ == 0) {
        replay_chart_id_ = chart_manager_->create_chart(
            config_.symbol_name, 
            config_.exchange_name, 
            config_.symbol_id, 
            config_.timeframe
        );
    }
    
    // Update current time to the start of the current bar
    if (!historical_candles_.empty() && current_bar_index_ < historical_candles_.size()) {
        current_time_ = historical_candles_[current_bar_index_].timestamp;
    }
    
    // Start the replay thread
    if (replay_thread_.joinable()) {
        replay_thread_.join();
    }
    
    replay_thread_ = std::thread(&ChartReplay::replay_thread_func, this);
}

void ChartReplay::pause_replay() {
    is_paused_ = true;
}

void ChartReplay::stop_replay() {
    should_stop_ = true;
    is_playing_ = false;
    is_paused_ = false;
    
    if (replay_thread_.joinable()) {
        replay_thread_.join();
    }
}

void ChartReplay::reset_replay() {
    stop_replay();
    current_bar_index_ = 0;
    if (!historical_candles_.empty()) {
        current_time_ = historical_candles_[0].timestamp;
    }
}

void ChartReplay::seek_to_time(uint64_t timestamp) {
    if (historical_candles_.empty()) {
        return;
    }

    // Find the closest bar to the requested timestamp
    for (size_t i = 0; i < historical_candles_.size(); ++i) {
        if (historical_candles_[i].timestamp >= timestamp) {
            current_bar_index_ = i;
            current_time_ = historical_candles_[i].timestamp;
            break;
        }
    }

    // Update the chart with the new position
    update_chart_with_current_data();
}

void ChartReplay::step_forward() {
    if (historical_candles_.empty() || current_bar_index_ >= historical_candles_.size()) {
        return;
    }

    // Move to next bar
    current_bar_index_++;
    if (current_bar_index_ < historical_candles_.size()) {
        current_time_ = historical_candles_[current_bar_index_].timestamp;
        update_chart_with_current_data();
        update_performance_metrics();
    }
}

void ChartReplay::step_backward() {
    if (historical_candles_.empty() || current_bar_index_ == 0) {
        return;
    }

    // Move to previous bar
    current_bar_index_--;
    current_time_ = historical_candles_[current_bar_index_].timestamp;
    update_chart_with_current_data();
    update_performance_metrics();
}

BacktestMetrics ChartReplay::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void ChartReplay::update_performance_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    metrics_.current_bar_index = current_bar_index_;
    metrics_.total_bars = historical_candles_.size();
    metrics_.current_time = std::chrono::steady_clock::now();

    if (metrics_.total_bars > 0) {
        metrics_.progress_percentage = (static_cast<double>(current_bar_index_) /
                                       static_cast<double>(metrics_.total_bars)) * 100.0;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        metrics_.current_time - metrics_.start_time).count();
    metrics_.elapsed_seconds = duration / 1000.0;

    // Calculate actual replay speed if we have processed some bars
    if (metrics_.elapsed_seconds > 0) {
        metrics_.replay_speed_actual = static_cast<double>(current_bar_index_) / metrics_.elapsed_seconds;
    }

    metrics_.total_bars_processed = current_bar_index_;
}

bool ChartReplay::load_historical_data(const std::string& symbol, const std::string& exchange,
                                      RenderEngine::TimeFrame timeframe,
                                      uint64_t start_time, uint64_t end_time) {
    // In a real implementation, this would load historical data from a database or file
    // For now, we'll simulate loading by generating synthetic data with more realistic patterns

    historical_candles_.clear();

    // Determine the time interval based on the timeframe
    uint64_t interval_ms = 0;
    switch (timeframe) {
        case RenderEngine::TimeFrame::TF_1MS:     interval_ms = 1; break;
        case RenderEngine::TimeFrame::TF_10MS:    interval_ms = 10; break;
        case RenderEngine::TimeFrame::TF_100MS:   interval_ms = 100; break;
        case RenderEngine::TimeFrame::TF_500MS:   interval_ms = 500; break;
        case RenderEngine::TimeFrame::TF_1SEC:    interval_ms = 1000; break;
        case RenderEngine::TimeFrame::TF_3SEC:    interval_ms = 3000; break;
        case RenderEngine::TimeFrame::TF_5SEC:    interval_ms = 5000; break;
        case RenderEngine::TimeFrame::TF_15SEC:   interval_ms = 15000; break;
        case RenderEngine::TimeFrame::TF_30SEC:   interval_ms = 30000; break;
        case RenderEngine::TimeFrame::TF_1MIN:    interval_ms = 60000; break;
        case RenderEngine::TimeFrame::TF_2MIN:    interval_ms = 120000; break;
        case RenderEngine::TimeFrame::TF_5MIN:    interval_ms = 300000; break;
        case RenderEngine::TimeFrame::TF_15MIN:   interval_ms = 900000; break;
        case RenderEngine::TimeFrame::TF_30MIN:   interval_ms = 1800000; break;
        case RenderEngine::TimeFrame::TF_1HOUR:   interval_ms = 3600000; break;
        case RenderEngine::TimeFrame::TF_2HOUR:   interval_ms = 7200000; break;
        case RenderEngine::TimeFrame::TF_4HOUR:   interval_ms = 14400000; break;
        case RenderEngine::TimeFrame::TF_6HOUR:   interval_ms = 21600000; break;
        case RenderEngine::TimeFrame::TF_12HOUR:  interval_ms = 43200000; break;
        case RenderEngine::TimeFrame::TF_1DAY:    interval_ms = 86400000; break;
        case RenderEngine::TimeFrame::TF_1WEEK:   interval_ms = 604800000; break;
        default:                                  interval_ms = 60000; break; // Default to 1 minute
    }

    // Generate synthetic data for demonstration with more realistic patterns
    uint64_t current_time = start_time;
    double current_price = 40000.0; // Starting price

    // Add some trend and volatility clustering for more realistic behavior
    double trend_factor = 0.0; // Current trend direction (-1 to 1)
    double volatility_factor = 0.02; // Base volatility level

    while (current_time <= end_time && historical_candles_.size() < 10000) { // Limit to 10k bars
        RenderEngine::OHLCVCandle candle;
        candle.timestamp = current_time;

        // Simulate mean reversion and momentum effects
        double momentum = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.02; // Small random momentum
        trend_factor = trend_factor * 0.95 + momentum * 0.05; // Smooth trend changes

        // Adjust volatility based on market conditions (volatility clustering)
        double volatility_change = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.1;
        volatility_factor = std::max(0.005, std::min(0.05, volatility_factor + volatility_change));

        // Generate price movement with trend and volatility
        double rand_change = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0 * volatility_factor;
        rand_change += trend_factor * 0.001; // Add trend component
        double new_price = current_price * (1.0 + rand_change);

        // Set OHLC values with realistic relationships
        double high_val, low_val;

        // Determine if this is a bullish or bearish candle
        bool is_bullish = new_price > current_price;

        // Calculate wick sizes based on market conditions
        double upper_wick_ratio = is_bullish ?
            0.1 + (static_cast<double>(rand()) / RAND_MAX) * 0.3 :  // 10-40% of range for upper wick in bullish
            0.3 + (static_cast<double>(rand()) / RAND_MAX) * 0.4;   // 30-70% of range for upper wick in bearish

        double lower_wick_ratio = is_bullish ?
            0.3 + (static_cast<double>(rand()) / RAND_MAX) * 0.4 :  // 30-70% of range for lower wick in bullish
            0.1 + (static_cast<double>(rand()) / RAND_MAX) * 0.3;   // 10-40% of range for lower wick in bearish

        double body_size = std::abs(new_price - current_price);
        double range = body_size + (upper_wick_ratio * body_size) + (lower_wick_ratio * body_size);

        if (is_bullish) {
            candle.open = current_price;
            candle.close = new_price;
            high_val = std::max(candle.open, candle.close) + (upper_wick_ratio * body_size);
            low_val = std::min(candle.open, candle.close) - (lower_wick_ratio * body_size);
        } else {
            candle.open = current_price;
            candle.close = new_price;
            high_val = std::max(candle.open, candle.close) + (upper_wick_ratio * body_size);
            low_val = std::min(candle.open, candle.close) - (lower_wick_ratio * body_size);
        }

        candle.high = high_val;
        candle.low = low_val;

        // Ensure OHLC values are consistent
        candle.high = std::max({candle.open, candle.close, candle.high});
        candle.low = std::min({candle.open, candle.close, candle.low});

        // Generate volume with some correlation to price movement and volatility
        double volume_base = 1000.0;
        double volume_multiplier = 0.5 + (static_cast<double>(rand()) / RAND_MAX) * 1.5; // 0.5x to 2x
        double volatility_multiplier = 0.8 + volatility_factor * 10.0; // Higher volume with higher volatility
        double movement_multiplier = 1.0 + std::abs(rand_change) * 50.0; // Higher volume with larger moves

        candle.volume = volume_base * volume_multiplier * volatility_multiplier * movement_multiplier;
        candle.trade_count = static_cast<uint64_t>(candle.volume * (0.1 + (static_cast<double>(rand()) / RAND_MAX) * 0.5)); // 10-60% of volume as trade count

        historical_candles_.push_back(candle);

        current_time += interval_ms;
        current_price = new_price;
    }

    std::cout << "Loaded " << historical_candles_.size() << " historical candles for replay." << std::endl;

    // Reset to beginning
    current_bar_index_ = 0;
    if (!historical_candles_.empty()) {
        current_time_ = historical_candles_[0].timestamp;
    }

    // Update metrics
    update_performance_metrics();

    return !historical_candles_.empty();
}

bool ChartReplay::load_from_csv(const std::string& csv_file_path,
                               const std::string& symbol,
                               RenderEngine::TimeFrame timeframe) {
    historical_candles_.clear();

    std::ifstream file(csv_file_path);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV file: " << csv_file_path << std::endl;
        return false;
    }

    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (!header_skipped) {
            header_skipped = true; // Skip header row
            continue;
        }

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        // Expected format: timestamp,open,high,low,close,volume
        if (tokens.size() < 6) {
            continue; // Skip malformed lines
        }

        try {
            RenderEngine::OHLCVCandle candle;

            // Parse timestamp (assuming it's in milliseconds)
            candle.timestamp = static_cast<uint64_t>(std::stoull(tokens[0]));
            candle.open = static_cast<double>(std::stod(tokens[1]));
            candle.high = static_cast<double>(std::stod(tokens[2]));
            candle.low = static_cast<double>(std::stod(tokens[3]));
            candle.close = static_cast<double>(std::stod(tokens[4]));
            candle.volume = static_cast<double>(std::stod(tokens[5]));
            candle.trade_count = tokens.size() > 6 ? static_cast<uint64_t>(std::stoull(tokens[6])) : 1;

            historical_candles_.push_back(candle);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing CSV line: " << line << " - " << e.what() << std::endl;
            continue;
        }
    }

    file.close();

    std::cout << "Loaded " << historical_candles_.size() << " candles from CSV file: " << csv_file_path << std::endl;

    // Reset to beginning
    current_bar_index_ = 0;
    if (!historical_candles_.empty()) {
        current_time_ = historical_candles_[0].timestamp;
    }

    // Update metrics
    update_performance_metrics();

    return !historical_candles_.empty();
}

void ChartReplay::replay_thread_func() {
    last_update_time_ = std::chrono::steady_clock::now();
    accumulated_time_ = 0.0;
    
    while (!should_stop_ && current_bar_index_ < historical_candles_.size()) {
        if (is_paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(now - last_update_time_).count();
        last_update_time_ = now;
        
        // Apply playback speed factor
        accumulated_time_ += elapsed * config_.playback_speed;
        
        // Process bars based on accumulated time
        uint64_t interval_ms = 0;
        switch (config_.timeframe) {
            case RenderEngine::TimeFrame::TF_1MS:     interval_ms = 1; break;
            case RenderEngine::TimeFrame::TF_10MS:    interval_ms = 10; break;
            case RenderEngine::TimeFrame::TF_100MS:   interval_ms = 100; break;
            case RenderEngine::TimeFrame::TF_500MS:   interval_ms = 500; break;
            case RenderEngine::TimeFrame::TF_1SEC:    interval_ms = 1000; break;
            case RenderEngine::TimeFrame::TF_3SEC:    interval_ms = 3000; break;
            case RenderEngine::TimeFrame::TF_5SEC:    interval_ms = 5000; break;
            case RenderEngine::TimeFrame::TF_15SEC:   interval_ms = 15000; break;
            case RenderEngine::TimeFrame::TF_30SEC:   interval_ms = 30000; break;
            case RenderEngine::TimeFrame::TF_1MIN:    interval_ms = 60000; break;
            case RenderEngine::TimeFrame::TF_2MIN:    interval_ms = 120000; break;
            case RenderEngine::TimeFrame::TF_5MIN:    interval_ms = 300000; break;
            case RenderEngine::TimeFrame::TF_15MIN:   interval_ms = 900000; break;
            case RenderEngine::TimeFrame::TF_30MIN:   interval_ms = 1800000; break;
            case RenderEngine::TimeFrame::TF_1HOUR:   interval_ms = 3600000; break;
            case RenderEngine::TimeFrame::TF_2HOUR:   interval_ms = 7200000; break;
            case RenderEngine::TimeFrame::TF_4HOUR:   interval_ms = 14400000; break;
            case RenderEngine::TimeFrame::TF_6HOUR:   interval_ms = 21600000; break;
            case RenderEngine::TimeFrame::TF_12HOUR:  interval_ms = 43200000; break;
            case RenderEngine::TimeFrame::TF_1DAY:    interval_ms = 86400000; break;
            case RenderEngine::TimeFrame::TF_1WEEK:   interval_ms = 604800000; break;
            default:                                  interval_ms = 60000; break;
        }
        
        // Convert interval to milliseconds for comparison
        double interval_double = static_cast<double>(interval_ms);
        
        // Process as many bars as accumulated time allows
        while (accumulated_time_ >= interval_double && current_bar_index_ < historical_candles_.size()) {
            process_next_bar();
            accumulated_time_ -= interval_double;
        }
        
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Ensure we update the chart with the final state
    if (!should_stop_ && !historical_candles_.empty()) {
        update_chart_with_current_data();
    }
    
    is_playing_ = false;
}

void ChartReplay::process_next_bar() {
    if (current_bar_index_ >= historical_candles_.size()) {
        // Reached the end of historical data
        if (config_.loop_enabled) {
            // Loop back to the beginning
            current_bar_index_ = 0;
        } else {
            // Stop playback
            should_stop_ = true;
            return;
        }
    }

    // Update current time
    current_time_ = historical_candles_[current_bar_index_].timestamp;

    // Update the chart with current data
    update_chart_with_current_data();

    // Move to next bar
    current_bar_index_++;

    // Update performance metrics
    update_performance_metrics();

    // If in step-by-step mode, pause after each bar
    if (config_.enable_step_by_step) {
        is_paused_ = true;
    }
}

void ChartReplay::update_chart_with_current_data() {
    if (replay_chart_id_ == 0 || historical_candles_.empty()) {
        return;
    }

    // Get the chart instance
    auto& charts = const_cast<std::unordered_map<uint32_t, ChartInstance>&>(chart_manager_->get_charts());
    auto it = charts.find(replay_chart_id_);
    if (it == charts.end()) {
        return;
    }

    ChartInstance& chart = it->second;

    // Clear previous data
    chart.dates.clear();
    chart.opens.clear();
    chart.highs.clear();
    chart.lows.clear();
    chart.closes.clear();
    chart.volumes.clear();

    // Add data up to the current bar index
    size_t end_index = std::min(current_bar_index_ + 1, historical_candles_.size()); // Include current bar
    for (size_t i = 0; i < end_index; ++i) {
        const auto& candle = historical_candles_[i];
        chart.dates.push_back(static_cast<double>(candle.timestamp));
        chart.opens.push_back(static_cast<float>(candle.open));
        chart.highs.push_back(static_cast<float>(candle.high));
        chart.lows.push_back(static_cast<float>(candle.low));
        chart.closes.push_back(static_cast<float>(candle.close));
        chart.volumes.push_back(static_cast<float>(candle.volume));
    }
}

void ChartReplay::render_replay_controls() {
    ImGui::SeparatorText("Replay Controls");

    // Playback speed slider
    float speed = static_cast<float>(config_.playback_speed);
    if (ImGui::SliderFloat("Speed", &speed, 0.1f, 10.0f, "%.1fx", ImGuiSliderFlags_Logarithmic)) {
        config_.playback_speed = static_cast<double>(speed);
    }
    BTQuant::UI::show_control_tooltip("chart_replay_speed");

    // Loop checkbox
    ImGui::Checkbox("Loop", &config_.loop_enabled);
    BTQuant::UI::show_control_tooltip("chart_replay_loop");

    // Step-by-step mode
    ImGui::Checkbox("Step-by-step mode", &config_.enable_step_by_step);
    BTQuant::UI::show_control_tooltip("chart_replay_step_by_step");

    // Manual control option
    ImGui::Checkbox("Manual control", &config_.enable_manual_control);
    BTQuant::UI::show_control_tooltip("chart_replay_manual_control");

    // Backtesting practice mode - separate from manual control
    static bool backtesting_practice_mode = false;
    if (ImGui::Checkbox("Backtesting practice mode", &backtesting_practice_mode)) {
        // Toggle both manual control and step-by-step when backtesting mode is enabled
        config_.enable_manual_control = backtesting_practice_mode;
        config_.enable_step_by_step = backtesting_practice_mode;
    }
    BTQuant::UI::show_control_tooltip("chart_replay_backtesting_mode");

    // Control buttons
    if (ImGui::Button(is_playing_ ? "Pause" : "Play")) {
        if (is_playing_) {
            pause_replay();
        } else {
            start_replay();
        }
    }
    BTQuant::UI::show_control_tooltip(is_playing_ ? "chart_replay_pause" : "chart_replay_play");

    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        stop_replay();
        reset_replay();
    }
    BTQuant::UI::show_control_tooltip("chart_replay_stop");

    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        reset_replay();
    }
    BTQuant::UI::show_control_tooltip("chart_replay_reset");

    // Step controls (only visible in manual control mode)
    if (config_.enable_manual_control) {
        ImGui::SameLine();
        if (ImGui::Button("<< Prev")) {
            step_backward();
        }
        BTQuant::UI::show_control_tooltip("chart_replay_prev");

        ImGui::SameLine();
        if (ImGui::Button("Next >>")) {
            step_forward();
        }
        BTQuant::UI::show_control_tooltip("chart_replay_next");
    }

    // Seek slider
    if (!historical_candles_.empty()) {
        int seek_pos = static_cast<int>(current_bar_index_);
        int max_pos = static_cast<int>(historical_candles_.size() - 1);

        if (ImGui::SliderInt("Progress", &seek_pos, 0, max_pos)) {
            if (static_cast<size_t>(seek_pos) < historical_candles_.size()) {
                current_bar_index_ = static_cast<size_t>(seek_pos);
                current_time_ = historical_candles_[current_bar_index_].timestamp;

                if (!is_playing_) {
                    update_chart_with_current_data();
                }
            }
        }
    }

    // Status display
    ImGui::Text("Status: %s",
                is_playing_ ? "Playing" :
                is_paused_ ? "Paused" : "Stopped");

    if (!historical_candles_.empty() && current_bar_index_ < historical_candles_.size()) {
        ImGui::Text("Current Bar: %zu/%zu", current_bar_index_, historical_candles_.size());
        ImGui::Text("Current Time: %llu", static_cast<unsigned long long>(current_time_));
    }

    // Performance metrics display
    if (config_.show_performance_metrics && !historical_candles_.empty()) {
        ImGui::SeparatorText("Performance Metrics");

        auto metrics = get_performance_metrics();
        ImGui::Text("Progress: %.2f%%", metrics.progress_percentage);
        ImGui::Text("Bars Processed: %llu/%llu", metrics.total_bars_processed, metrics.total_bars);
        ImGui::Text("Elapsed Time: %.2fs", metrics.elapsed_seconds);
        ImGui::Text("Actual Speed: %.2f bars/sec", metrics.replay_speed_actual);
    }

    // Load historical data section
    ImGui::SeparatorText("Load Historical Data");

    static char symbol_buffer[64] = "BTC-USDT";
    static char exchange_buffer[64] = "Binance";
    static uint64_t start_time_input = 0;
    static uint64_t end_time_input = 0;

    ImGui::InputText("Symbol", symbol_buffer, sizeof(symbol_buffer));
    ImGui::InputText("Exchange", exchange_buffer, sizeof(exchange_buffer));

    ImGui::InputScalar("Start Time", ImGuiDataType_U64, &start_time_input);
    ImGui::InputScalar("End Time", ImGuiDataType_U64, &end_time_input);

    // Timeframe selection
    const char* timeframes[] = {"1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"};
    int selected_timeframe = static_cast<int>(config_.timeframe);
    if (ImGui::Combo("Timeframe", &selected_timeframe, timeframes, IM_ARRAYSIZE(timeframes))) {
        config_.timeframe = static_cast<RenderEngine::TimeFrame>(selected_timeframe);
    }

    if (ImGui::Button("Load Data")) {
        stop_replay();
        reset_replay();

        if (load_historical_data(std::string(symbol_buffer), std::string(exchange_buffer),
                                config_.timeframe, start_time_input, end_time_input)) {
            std::cout << "Historical data loaded successfully!" << std::endl;

            // Update config with loaded parameters
            config_.symbol_name = std::string(symbol_buffer);
            config_.exchange_name = std::string(exchange_buffer);
        } else {
            std::cerr << "Failed to load historical data!" << std::endl;
        }
    }

    // CSV loading section
    ImGui::SeparatorText("Load from CSV");

    static char csv_path_buffer[256] = "./data/historical_data.csv";
    ImGui::InputText("CSV File Path", csv_path_buffer, sizeof(csv_path_buffer));

    if (ImGui::Button("Load from CSV")) {
        stop_replay();
        reset_replay();

        if (load_from_csv(std::string(csv_path_buffer), std::string(symbol_buffer), config_.timeframe)) {
            std::cout << "Historical data loaded from CSV successfully!" << std::endl;

            // Update config with loaded parameters
            config_.symbol_name = std::string(symbol_buffer);
        } else {
            std::cerr << "Failed to load historical data from CSV!" << std::endl;
        }
    }

    // Load from market data processor (if available)
    if (processor_) {
        ImGui::SeparatorText("Load from Market Data");

        static char symbol_buffer2[64] = "BTC-USDT";
        static uint32_t symbol_id_input = 1;
        static uint64_t start_time_input2 = 0;
        static uint64_t end_time_input2 = 0;

        ImGui::InputText("Symbol", symbol_buffer2, sizeof(symbol_buffer2));
        ImGui::InputScalar("Symbol ID", ImGuiDataType_U32, &symbol_id_input);

        ImGui::InputScalar("Start Time", ImGuiDataType_U64, &start_time_input2);
        ImGui::InputScalar("End Time", ImGuiDataType_U64, &end_time_input2);

        // Timeframe selection for market data
        const char* timeframes2[] = {"1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"};
        int selected_timeframe2 = static_cast<int>(config_.timeframe);
        if (ImGui::Combo("Timeframe", &selected_timeframe2, timeframes2, IM_ARRAYSIZE(timeframes2))) {
            config_.timeframe = static_cast<RenderEngine::TimeFrame>(selected_timeframe2);
        }

        if (ImGui::Button("Load from Market Data")) {
            stop_replay();
            reset_replay();

            // Attempt to load from market data processor
            if (load_from_market_data_processor(symbol_id_input, config_.timeframe, start_time_input2, end_time_input2)) {
                std::cout << "Historical data loaded from market data processor successfully!" << std::endl;

                // Update config with loaded parameters
                config_.symbol_name = std::string(symbol_buffer2);
            } else {
                std::cerr << "Failed to load historical data from market data processor!" << std::endl;
            }
        }
    }
}

bool ChartReplay::load_from_market_data_processor(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                                                 uint64_t start_time, uint64_t end_time) {
    if (!processor_) {
        std::cerr << "Market data processor not available!" << std::endl;
        return false;
    }

    historical_candles_.clear();

    try {
        // Get historical candles from the market data processor
        auto candles = processor_->getCandles(symbol_id, timeframe);

        // Filter candles by time range if specified
        for (const auto& candle : candles) {
            if ((start_time == 0 || candle.timestamp >= start_time) &&
                (end_time == 0 || candle.timestamp <= end_time)) {
                historical_candles_.push_back(candle);
            }
        }

        std::cout << "Loaded " << historical_candles_.size() << " historical candles from market data processor." << std::endl;

        // Reset to beginning
        current_bar_index_ = 0;
        if (!historical_candles_.empty()) {
            current_time_ = historical_candles_[0].timestamp;
        }

        // Update metrics
        update_performance_metrics();

        return !historical_candles_.empty();
    } catch (const std::exception& e) {
        std::cerr << "Error loading data from market data processor: " << e.what() << std::endl;
        return false;
    }
}

} // namespace BTQuant