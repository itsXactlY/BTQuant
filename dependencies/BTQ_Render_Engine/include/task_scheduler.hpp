#ifndef BTQ_TASK_SCHEDULER_HPP
#define BTQ_TASK_SCHEDULER_HPP

#include <thread>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <atomic>
#include <functional>
#include <future>
#include <chrono>
#include <system_error>
#include <limits>
#include <map>
#include "../build/_deps/concurrentqueue-src/concurrentqueue.h"
#include "threading/atomic_signal.hpp"

namespace btq {

// Forward declarations for data structures
struct Trade {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double volume;
};

struct Candle {
    std::chrono::system_clock::time_point timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Type alias for Bollinger Bands result: tuple of (upper band, middle band, lower band)
using BollingerBandsResult = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>;

class TaskScheduler {
public:
    explicit TaskScheduler(size_t num_threads = 0);  // 0 means use hardware concurrency
    ~TaskScheduler();

    // Delete copy constructor and assignment operator
    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;
    
    // Thread-safe method to check if scheduler is stopping
    bool is_stopping() const;

    // Volume Calculations
    std::future<std::vector<double>> calculate_volume_profile_async(
        const std::vector<Trade>& trades, 
        double min_price, 
        double max_price, 
        int resolution);

    std::future<double> calculate_volume_weighted_average_price_async(
        const std::vector<Trade>& trades);

    std::future<std::vector<double>> calculate_volume_by_time_async(
        const std::vector<Trade>& trades,
        int time_resolution_minutes);

    // Enhanced volume calculation methods
    std::future<std::vector<double>> calculate_rolling_volume_profile_async(
        const std::vector<Trade>& trades,
        double min_price,
        double max_price,
        int resolution,
        int window_size);

    std::future<std::vector<double>> calculate_time_based_volume_async(
        const std::vector<Trade>& trades,
        int time_resolution_seconds);

    // Additional advanced volume calculation methods
    std::future<std::vector<double>> calculate_volume_at_price_levels_async(
        const std::vector<Trade>& trades,
        const std::vector<double>& price_levels);

    std::future<std::vector<double>> calculate_time_weighted_volume_async(
        const std::vector<Trade>& trades,
        int time_window_minutes);

    // Additional advanced market profile and VWAP methods
    std::future<std::vector<std::pair<double, double>>> calculate_market_profile_async(
        const std::vector<Trade>& trades,
        double min_price,
        double max_price,
        int price_levels);

    std::future<std::vector<double>> calculate_time_based_vwap_async(
        const std::vector<Trade>& trades,
        int time_resolution_minutes);

    // Indicator Computations
    std::future<std::vector<double>> calculate_sma_async(
        const std::vector<double>& prices, 
        int period);

    std::future<std::vector<double>> calculate_ema_async(
        const std::vector<double>& prices, 
        int period);

    std::future<std::vector<double>> calculate_rsi_async(
        const std::vector<double>& prices, 
        int period);

    std::future<BollingerBandsResult>
    calculate_bollinger_bands_async(
        const std::vector<double>& prices,
        int period,
        double num_std_dev = 2.0);

    // Enhanced indicator computation methods
    std::future<std::vector<double>> calculate_adaptive_sma_async(
        const std::vector<double>& prices,
        int min_period,
        int max_period);

    std::future<std::vector<double>> calculate_parabolic_sar_async(
        const std::vector<Candle>& candles,
        double acceleration_factor_step = 0.02,
        double max_acceleration_factor = 0.2);

    // Additional advanced indicator computation methods
    std::future<std::vector<double>> calculate_hull_moving_average_async(
        const std::vector<double>& prices,
        int period);

    std::future<std::vector<double>> calculate_triangular_moving_average_async(
        const std::vector<double>& prices,
        int period);

    // Data Processing
    std::future<std::vector<Candle>> aggregate_candles_async(
        const std::vector<Trade>& trades, 
        std::chrono::seconds timeframe);

    std::future<std::vector<Trade>> filter_trades_async(
        const std::vector<Trade>& trades, 
        std::function<bool(const Trade&)> filter_func);

    std::future<std::vector<std::pair<double, double>>> calculate_histogram_async(
        const std::vector<double>& values,
        int num_bins);

    // Additional utility methods for parallel processing
    std::future<std::vector<double>> transform_data_parallel_async(
        const std::vector<double>& input,
        std::function<double(double)> transform_func);

    std::future<std::vector<double>> calculate_moving_average_async(
        const std::vector<double>& prices,
        int period);

    // Advanced multi-threaded calculation methods
    std::future<std::vector<double>> calculate_macd_async(
        const std::vector<double>& prices,
        int fast_period = 12,
        int slow_period = 26,
        int signal_period = 9);

    std::future<std::vector<double>> calculate_atr_async(
        const std::vector<Candle>& candles,
        int period = 14);

    std::future<std::vector<double>> calculate_stochastic_oscillator_async(
        const std::vector<Candle>& candles,
        int k_period = 14,
        int d_period = 3);

    std::future<std::vector<double>> calculate_on_balance_volume_async(
        const std::vector<Candle>& candles);

    // New method for calculating correlation between two series
    std::future<double> calculate_correlation_async(
        const std::vector<double>& series1,
        const std::vector<double>& series2);

    // Advanced multi-threaded batch processing methods
    std::future<std::vector<std::vector<double>>> calculate_batch_indicators_async(
        const std::vector<std::vector<double>>& price_series,
        const std::vector<std::pair<std::string, int>>& indicator_configs);

    std::future<std::vector<std::vector<double>>> calculate_multiple_timeframe_indicators_async(
        const std::vector<double>& prices,
        const std::vector<std::pair<std::string, std::vector<int>>>& indicator_configs);

    std::future<std::vector<std::vector<Trade>>> process_batch_trades_async(
        const std::vector<std::vector<Trade>>& trade_batches,
        std::function<std::vector<Trade>(const std::vector<Trade>&)> processor_func);

    std::future<std::vector<std::vector<Candle>>> aggregate_batch_candles_async(
        const std::vector<std::vector<Trade>>& trade_batches,
        std::chrono::seconds timeframe);

    // Enhanced data processing methods
    std::future<std::vector<std::vector<Trade>>> partition_and_process_trades_async(
        const std::vector<Trade>& trades,
        std::function<std::vector<Trade>(const std::vector<Trade>&)> processor_func,
        int num_partitions = 0);

    std::future<std::vector<Candle>> create_dynamic_timeframe_candles_async(
        const std::vector<Trade>& trades,
        std::chrono::seconds base_timeframe,
        double volume_threshold = 0.0);

    // Additional advanced data processing methods
    std::future<std::vector<std::vector<double>>> calculate_normalized_correlation_matrix_async(
        const std::vector<std::vector<double>>& data_series);

    std::future<std::vector<Trade>> apply_market_microstructure_filters_async(
        const std::vector<Trade>& trades,
        double tick_size,
        std::chrono::milliseconds min_time_diff = std::chrono::milliseconds(0));

    // Additional advanced multi-threaded methods for volume analysis
    std::future<std::vector<double>> calculate_cumulative_volume_delta_async(
        const std::vector<Trade>& trades,
        const std::vector<double>& benchmark_prices);

    std::future<std::vector<std::pair<double, double>>> calculate_volume_price_confirmation_indicator_async(
        const std::vector<Trade>& trades,
        int trend_period);

    // Additional advanced multi-threaded methods for indicator computations
    std::future<std::vector<double>> calculate_chandelier_exit_async(
        const std::vector<Candle>& candles,
        int period,
        double multiplier = 3.0);

    std::future<std::vector<double>> calculate_keltner_channels_async(
        const std::vector<Candle>& candles,
        int period,
        double multiplier = 2.0);

    // Additional advanced multi-threaded methods for data processing
    std::future<std::vector<std::vector<Trade>>> detect_and_classify_market_regimes_async(
        const std::vector<Trade>& trades,
        double volatility_threshold,
        double volume_threshold);

    std::future<std::vector<std::vector<std::vector<double>>>> calculate_rolling_correlation_matrix_async(
        const std::vector<std::vector<double>>& data_series,
        int window_size);

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    moodycamel::ConcurrentQueue<std::function<void()>> tasks_;

    btq::threading::AtomicSignal task_available_signal_;  // Atomic signal for task availability
    size_t num_threads_;
    std::atomic<bool> stop_;

public:
    void enqueue_task(std::function<void()> task);
};

} // namespace btq

#endif // BTQ_TASK_SCHEDULER_HPP