#ifndef BTQ_TASK_SCHEDULER_HPP
#define BTQ_TASK_SCHEDULER_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <functional>
#include <future>
#include <chrono>
#include <system_error>
#include <limits>
#include <map>

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

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
    size_t num_threads_;

public:
    void enqueue_task(std::function<void()> task);
};

} // namespace btq

#endif // BTQ_TASK_SCHEDULER_HPP