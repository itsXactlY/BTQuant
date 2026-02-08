#include "../../include/task_scheduler.hpp"
#include <iostream>
#include <future>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <atomic>

namespace btq {

TaskScheduler::TaskScheduler(size_t num_threads)
    : num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency()),
      stop_(false) {

    for (size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }
}

bool TaskScheduler::is_stopping() const {
    std::shared_lock<std::shared_mutex> lock(stop_mutex_);
    return stop_;
}

TaskScheduler::~TaskScheduler() {
    {
        std::unique_lock<std::shared_mutex> lock(stop_mutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void TaskScheduler::worker_loop() {
    while (true) {
        std::function<void()> task;

        // Try to dequeue a task
        if (tasks_.try_dequeue(task)) {
            if (task) {
                task();
            }
        } else {
            // No task available, check if we should stop
            {
                std::shared_lock<std::shared_mutex> stop_lock(stop_mutex_);
                if (stop_ && tasks_.size_approx() == 0) {
                    return;
                }
            }

            // Wait for a short time before checking again
            std::unique_lock<std::mutex> lock(notification_mutex_);
            condition_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                std::shared_lock<std::shared_mutex> stop_lock(stop_mutex_);
                return stop_ || tasks_.size_approx() > 0;
            });
        }
    }
}

void TaskScheduler::enqueue_task(std::function<void()> task) {
    {
        std::shared_lock<std::shared_mutex> stop_lock(stop_mutex_);
        if (stop_) {
            throw std::runtime_error("TaskScheduler is stopped");
        }
    }

    tasks_.enqueue(std::move(task));
    condition_.notify_one();
}

// Template method implementation moved to header file

// Volume Calculations
std::future<std::vector<double>> TaskScheduler::calculate_volume_profile_async(
    const std::vector<Trade>& trades,
    double min_price,
    double max_price,
    int resolution) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    // Main task runs on background thread via enqueue_task
    enqueue_task([trades, min_price, max_price, resolution, promise]() {
        try {
            std::vector<double> volume_profile(resolution, 0.0);
            double price_range = max_price - min_price;
            if (price_range <= 0) {
                promise->set_value(std::move(volume_profile));
                return;
            }

            double bin_size = price_range / resolution;

            // Use sequential processing to avoid thread nesting
            // The outer task is already running on a thread pool thread
            for (const auto& trade : trades) {
                int bin_index = static_cast<int>((trade.price - min_price) / bin_size);
                if (bin_index >= 0 && bin_index < resolution) {
                    volume_profile[bin_index] += trade.volume;
                }
            }

            promise->set_value(std::move(volume_profile));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<double> TaskScheduler::calculate_volume_weighted_average_price_async(
    const std::vector<Trade>& trades) {

    auto promise = std::make_shared<std::promise<double>>();
    auto future = promise->get_future();

    // Main task runs on background thread via enqueue_task
    enqueue_task([trades, promise]() {
        try {
            if (trades.empty()) {
                promise->set_value(0.0);
                return;
            }

            // Use sequential processing to avoid thread nesting
            // The outer task is already running on a thread pool thread
            double total_value = 0.0;
            double total_volume = 0.0;

            for (const auto& trade : trades) {
                total_value += trade.price * trade.volume;
                total_volume += trade.volume;
            }

            double vwap = (total_volume > 0) ? total_value / total_volume : 0.0;
            promise->set_value(vwap);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_volume_by_time_async(
    const std::vector<Trade>& trades,
    int time_resolution_minutes) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, time_resolution_minutes, promise]() {
        try {
            if (trades.empty()) {
                promise->set_value(std::vector<double>());
                return;
            }

            // Determine time range
            auto min_max_it = std::minmax_element(trades.begin(), trades.end(),
                [](const Trade& a, const Trade& b) {
                    return a.timestamp < b.timestamp;
                });

            auto start_time = min_max_it.first->timestamp;
            auto end_time = min_max_it.second->timestamp;

            // Calculate number of time bins
            auto duration_minutes = std::chrono::duration_cast<std::chrono::minutes>(
                end_time - start_time).count();
            int num_bins = static_cast<int>(std::ceil(static_cast<double>(duration_minutes) /
                                                      time_resolution_minutes)) + 1;

            std::vector<double> volume_by_time(num_bins, 0.0);
            auto bin_duration = std::chrono::minutes(time_resolution_minutes);

            // Use sequential processing to avoid thread nesting
            // The outer task is already running on a thread pool thread
            for (const auto& trade : trades) {
                auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                    trade.timestamp - start_time).count();
                int bin_index = static_cast<int>(time_diff / time_resolution_minutes);

                if (bin_index >= 0 && bin_index < num_bins) {
                    volume_by_time[bin_index] += trade.volume;
                }
            }

            promise->set_value(std::move(volume_by_time));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Indicator Computations
std::future<std::vector<double>> TaskScheduler::calculate_sma_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    // Main task runs on background thread via enqueue_task
    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> sma_values;
            if (prices.size() < static_cast<size_t>(period)) {
                sma_values.resize(prices.size());
                promise->set_value(std::move(sma_values));
                return;
            }

            sma_values.reserve(prices.size() - period + 1);

            // Calculate initial sum for the first period
            double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
            sma_values.push_back(sum / period);

            // Use sequential processing to avoid thread nesting
            // The outer task is already running on a thread pool thread
            for (size_t i = period; i < prices.size(); ++i) {
                sum += prices[i] - prices[i - period];
                sma_values.push_back(sum / period);
            }

            promise->set_value(std::move(sma_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_ema_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> ema_values;
            if (prices.empty()) {
                promise->set_value(std::move(ema_values));
                return;
            }

            ema_values.reserve(prices.size());
            double multiplier = 2.0 / (period + 1);

            // Start with SMA for the first EMA value
            int sma_period = std::min(period, static_cast<int>(prices.size()));
            double sma_sum = std::accumulate(prices.begin(), prices.begin() + sma_period, 0.0);
            double ema = sma_sum / sma_period;
            ema_values.push_back(ema);

            // Use parallel algorithm for large datasets
            if (prices.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // For EMA, we need to calculate sequentially since each value depends on the previous one
                // But we can still parallelize the initial SMA calculation if needed
                for (size_t i = sma_period; i < prices.size(); ++i) {
                    ema = (prices[i] - ema) * multiplier + ema;
                    ema_values.push_back(ema);
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = sma_period; i < prices.size(); ++i) {
                    ema = (prices[i] - ema) * multiplier + ema;
                    ema_values.push_back(ema);
                }
            }

            promise->set_value(std::move(ema_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_rsi_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> rsi_values;
            if (prices.size() < 2) {
                rsi_values.resize(prices.size(), 50.0); // Return neutral RSI if insufficient data
                promise->set_value(std::move(rsi_values));
                return;
            }

            rsi_values.reserve(prices.size());

            // Initialize with NaN for first value
            rsi_values.push_back(std::numeric_limits<double>::quiet_NaN());

            // Calculate price changes
            std::vector<double> changes;
            changes.reserve(prices.size() - 1);

            // Use sequential processing to avoid thread nesting
            // The outer task is already running on a thread pool thread
            for (size_t i = 1; i < prices.size(); ++i) {
                changes.push_back(prices[i] - prices[i-1]);
            }

            if (changes.size() < static_cast<size_t>(period)) {
                // Fill remaining with NaN if insufficient data for full RSI calculation
                for (size_t i = 1; i < prices.size(); ++i) {
                    rsi_values.push_back(std::numeric_limits<double>::quiet_NaN());
                }
                promise->set_value(std::move(rsi_values));
                return;
            }

            // Calculate initial average gain and loss for first period
            double avg_gain = 0.0, avg_loss = 0.0;
            for (int i = 0; i < period; ++i) {
                if (changes[i] > 0) {
                    avg_gain += changes[i];
                } else {
                    avg_loss -= changes[i];
                }
            }
            avg_gain /= period;
            avg_loss /= period;

            // Calculate RSI for first complete period
            double rs = (avg_loss != 0) ? avg_gain / avg_loss : 0;
            double rsi = 100.0 - (100.0 / (1.0 + rs));
            rsi_values.push_back(rsi);

            // Calculate subsequent RSI values using Wilder's smoothing method
            for (size_t i = period; i < changes.size(); ++i) {
                double current_change = changes[i];
                if (current_change > 0) {
                    avg_gain = (avg_gain * (period - 1) + current_change) / period;
                    avg_loss = (avg_loss * (period - 1)) / period;
                } else {
                    avg_gain = (avg_gain * (period - 1)) / period;
                    avg_loss = (avg_loss * (period - 1) + (-current_change)) / period; // Make sure we add the positive value
                }

                rs = (avg_loss != 0) ? avg_gain / avg_loss : 0;
                rsi = 100.0 - (100.0 / (1.0 + rs));
                rsi_values.push_back(rsi);
            }

            promise->set_value(std::move(rsi_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<BollingerBandsResult> TaskScheduler::calculate_bollinger_bands_async(
    const std::vector<double>& prices,
    int period,
    double num_std_dev) {

    auto promise = std::make_shared<std::promise<BollingerBandsResult>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, num_std_dev, promise]() {
        try {
            std::vector<double> upper_band, middle_band, lower_band;

            if (prices.size() < static_cast<size_t>(period)) {
                promise->set_value(std::make_tuple(std::move(upper_band), std::move(middle_band), std::move(lower_band)));
                return;
            }

            // Calculate SMA first
            std::vector<double> sma_values;
            sma_values.reserve(prices.size() - period + 1);

            double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
            sma_values.push_back(sum / period);

            // Use parallel algorithm for large datasets
            if (prices.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Parallelize the SMA calculation using sliding window
                size_t start_idx = period;
                size_t total_elements = prices.size() - start_idx;

                if (total_elements >= num_threads) {
                    sma_values.resize(prices.size() - period + 1);

                    std::vector<std::thread> processing_threads;
                    size_t chunk_size = total_elements / num_threads;

                    for (size_t t = 0; t < num_threads; ++t) {
                        size_t chunk_start = start_idx + t * chunk_size;
                        size_t chunk_end = (t == num_threads - 1) ? prices.size() : start_idx + (t + 1) * chunk_size;

                        processing_threads.emplace_back([chunk_start, chunk_end, &prices, period, &sma_values]() {
                            double local_sum = std::accumulate(prices.begin() + (chunk_start - period), prices.begin() + chunk_start, 0.0);

                            for (size_t i = chunk_start; i < chunk_end; ++i) {
                                // Update the sum using the sliding window technique
                                local_sum += prices[i] - prices[i - period];
                                sma_values[i - period + 1] = local_sum / period;
                            }
                        });
                    }

                    for (auto& thread : processing_threads) {
                        thread.join();
                    }
                } else {
                    // Sequential processing for smaller datasets
                    for (size_t i = period; i < prices.size(); ++i) {
                        sum += prices[i] - prices[i - period];
                        sma_values.push_back(sum / period);
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = period; i < prices.size(); ++i) {
                    sum += prices[i] - prices[i - period];
                    sma_values.push_back(sum / period);
                }
            }

            // Calculate standard deviation and bands
            middle_band = sma_values;
            upper_band.reserve(sma_values.size());
            lower_band.reserve(sma_values.size());

            // Sequential processing for standard deviation calculation to avoid thread nesting
            for (size_t i = 0; i < sma_values.size(); ++i) {
                size_t start_idx = i; // The SMA at index i corresponds to the window ending at index i+period-1

                // Calculate variance for the corresponding window
                double sum_sq_diff = 0.0;
                for (int j = 0; j < period; ++j) {
                    if ((start_idx + j) < prices.size()) {
                        double diff = prices[start_idx + j] - sma_values[i];
                        sum_sq_diff += diff * diff;
                    }
                }
                double variance = sum_sq_diff / period;
                double std_dev = std::sqrt(variance);

                upper_band.push_back(sma_values[i] + num_std_dev * std_dev);
                lower_band.push_back(sma_values[i] - num_std_dev * std_dev);
            }

            promise->set_value(std::make_tuple(std::move(upper_band), std::move(middle_band), std::move(lower_band)));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Data Processing
std::future<std::vector<Candle>> TaskScheduler::aggregate_candles_async(
    const std::vector<Trade>& trades,
    std::chrono::seconds timeframe) {

    auto promise = std::make_shared<std::promise<std::vector<Candle>>>();
    auto future = promise->get_future();

    // Main task runs on background thread via enqueue_task
    enqueue_task([trades, timeframe, promise]() {
        try {
            std::vector<Candle> candles;
            if (trades.empty()) {
                promise->set_value(std::move(candles));
                return;
            }

            // Sequential processing for smaller datasets
            // Group trades by time windows
            std::map<std::chrono::system_clock::time_point, Candle> candle_map;

            for (const auto& trade : trades) {
                // Calculate the start time of the candle period
                auto time_since_epoch = trade.timestamp.time_since_epoch();
                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                auto period_start = std::chrono::system_clock::time_point(seconds -
                    std::chrono::seconds(seconds.count() % timeframe.count()));

                if (candle_map.find(period_start) == candle_map.end()) {
                    // Initialize new candle
                    candle_map[period_start] = {
                        period_start,
                        trade.price,  // Open
                        trade.price,  // High
                        trade.price,  // Low
                        trade.price,  // Close
                        trade.volume  // Volume
                    };
                } else {
                    // Update existing candle
                    auto& candle = candle_map[period_start];
                    candle.high = std::max(candle.high, trade.price);
                    candle.low = std::min(candle.low, trade.price);
                    candle.close = trade.price;  // Last price becomes close
                    candle.volume += trade.volume;
                }
            }

            // Convert map to vector
            candles.reserve(candle_map.size());
            for (const auto& pair : candle_map) {
                candles.push_back(pair.second);
            }

            promise->set_value(std::move(candles));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<Trade>> TaskScheduler::filter_trades_async(
    const std::vector<Trade>& trades,
    std::function<bool(const Trade&)> filter_func) {

    auto promise = std::make_shared<std::promise<std::vector<Trade>>>();
    auto future = promise->get_future();

    enqueue_task([trades, filter_func, promise]() {
        try {
            std::vector<Trade> filtered_trades;

            // Sequential processing for smaller datasets
            filtered_trades.reserve(trades.size()); // Reserve to prevent reallocation

            for (const auto& trade : trades) {
                if (filter_func(trade)) {
                    filtered_trades.push_back(trade);
                }
            }

            // Shrink to fit to save memory
            filtered_trades.shrink_to_fit();
            promise->set_value(std::move(filtered_trades));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::pair<double, double>>> TaskScheduler::calculate_histogram_async(
    const std::vector<double>& values,
    int num_bins) {

    auto promise = std::make_shared<std::promise<std::vector<std::pair<double, double>>>>();
    auto future = promise->get_future();

    enqueue_task([values, num_bins, promise]() {
        try {
            std::vector<std::pair<double, double>> histogram;
            if (values.empty() || num_bins <= 0) {
                promise->set_value(std::move(histogram));
                return;
            }

            // Find min and max values
            auto min_max = std::minmax_element(values.begin(), values.end());
            double min_val = *min_max.first;
            double max_val = *min_max.second;

            if (min_val == max_val) {
                // Special case: all values are the same
                histogram.push_back({min_val, static_cast<double>(values.size())});
                promise->set_value(std::move(histogram));
                return;
            }

            double bin_width = (max_val - min_val) / num_bins;
            histogram.resize(num_bins, {0.0, 0.0});

            // Sequential processing for smaller datasets
            // Calculate bin boundaries and counts
            for (double val : values) {
                int bin_index = static_cast<int>((val - min_val) / bin_width);
                // Handle edge case where value equals max
                if (bin_index >= num_bins) {
                    bin_index = num_bins - 1;
                }

                histogram[bin_index].second += 1.0; // Increment count
            }

            // Set bin centers
            for (int i = 0; i < num_bins; ++i) {
                histogram[i].first = min_val + (i + 0.5) * bin_width;
            }

            promise->set_value(std::move(histogram));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional utility methods for parallel processing

std::future<std::vector<double>> TaskScheduler::transform_data_parallel_async(
    const std::vector<double>& input,
    std::function<double(double)> transform_func) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([input, transform_func, promise]() {
        try {
            std::vector<double> result(input.size());

            // Sequential processing for smaller datasets
            for (size_t i = 0; i < input.size(); ++i) {
                result[i] = transform_func(input[i]);
            }

            promise->set_value(std::move(result));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_moving_average_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> ma_values;

            if (prices.size() < static_cast<size_t>(period)) {
                ma_values.resize(prices.size());
                promise->set_value(std::move(ma_values));
                return;
            }

            ma_values.reserve(prices.size() - period + 1);

            // Calculate initial sum for the first period
            double sum = 0.0;
            for (int i = 0; i < period; ++i) {
                sum += prices[i];
            }
            ma_values.push_back(sum / period);

            // Sequential processing for smaller datasets
            for (size_t i = period; i < prices.size(); ++i) {
                sum += prices[i] - prices[i - period];
                ma_values.push_back(sum / period);
            }

            promise->set_value(std::move(ma_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional multi-threaded calculation methods

std::future<std::vector<double>> TaskScheduler::calculate_macd_async(
    const std::vector<double>& prices,
    int fast_period,
    int slow_period,
    int signal_period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, fast_period, slow_period, signal_period, promise]() {
        try {
            std::vector<double> macd_line, signal_line, histogram;

            if (prices.size() < static_cast<size_t>(std::max({fast_period, slow_period, signal_period}))) {
                promise->set_value(std::move(macd_line));
                return;
            }

            // Calculate EMAs for fast and slow periods using the same approach as calculate_ema_async
            // but inline to avoid spawning additional threads outside the thread pool

            // Calculate fast EMA
            std::vector<double> fast_ema_values;
            if (!prices.empty()) {
                fast_ema_values.reserve(prices.size());
                double multiplier = 2.0 / (fast_period + 1);

                // Start with SMA for the first EMA value
                int sma_period = std::min(fast_period, static_cast<int>(prices.size()));
                double sma_sum = std::accumulate(prices.begin(), prices.begin() + sma_period, 0.0);
                double ema = sma_sum / sma_period;
                fast_ema_values.push_back(ema);

                // Sequential processing for EMA since each value depends on the previous
                for (size_t i = sma_period; i < prices.size(); ++i) {
                    ema = (prices[i] - ema) * multiplier + ema;
                    fast_ema_values.push_back(ema);
                }
            }

            // Calculate slow EMA
            std::vector<double> slow_ema_values;
            if (!prices.empty()) {
                slow_ema_values.reserve(prices.size());
                double multiplier = 2.0 / (slow_period + 1);

                // Start with SMA for the first EMA value
                int sma_period = std::min(slow_period, static_cast<int>(prices.size()));
                double sma_sum = std::accumulate(prices.begin(), prices.begin() + sma_period, 0.0);
                double ema = sma_sum / sma_period;
                slow_ema_values.push_back(ema);

                // Sequential processing for EMA since each value depends on the previous
                for (size_t i = sma_period; i < prices.size(); ++i) {
                    ema = (prices[i] - ema) * multiplier + ema;
                    slow_ema_values.push_back(ema);
                }
            }

            // Use the calculated EMAs
            auto& fast_ema = fast_ema_values;
            auto& slow_ema = slow_ema_values;

            // Calculate MACD line
            std::vector<double> macd_raw;
            macd_raw.reserve(std::min(fast_ema.size(), slow_ema.size()));

            for (size_t i = 0; i < std::min(fast_ema.size(), slow_ema.size()); ++i) {
                macd_raw.push_back(fast_ema[i] - slow_ema[i]);
            }

            // Calculate signal line (EMA of MACD line)
            if (macd_raw.size() >= static_cast<size_t>(signal_period)) {
                signal_line.reserve(macd_raw.size() - signal_period + 1);

                // Calculate initial SMA for signal line
                double signal_sma_sum = std::accumulate(macd_raw.begin(), macd_raw.begin() + signal_period, 0.0);
                double signal_ema = signal_sma_sum / signal_period;
                signal_line.push_back(signal_ema);

                double signal_multiplier = 2.0 / (signal_period + 1);

                for (size_t i = signal_period; i < macd_raw.size(); ++i) {
                    signal_ema = (macd_raw[i] - signal_ema) * signal_multiplier + signal_ema;
                    signal_line.push_back(signal_ema);
                }
            }

            // Calculate histogram (MACD line - Signal line)
            histogram.reserve(std::min(macd_raw.size(), signal_line.size()));
            for (size_t i = 0; i < std::min(macd_raw.size(), signal_line.size()); ++i) {
                histogram.push_back(macd_raw[i] - signal_line[i]);
            }

            // The final MACD values would typically be the difference between MACD line and signal line
            // For simplicity, returning the histogram values
            promise->set_value(std::move(histogram));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_atr_async(
    const std::vector<Candle>& candles,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, period, promise]() {
        try {
            std::vector<double> atr_values;
            if (candles.size() < 2) {
                promise->set_value(std::move(atr_values));
                return;
            }

            // Calculate True Range for each candle
            std::vector<double> true_ranges(candles.size() - 1);

            // Use multi-threaded approach for large datasets when calculating True Range
            if (candles.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), candles.size());
                if (num_threads < 2) num_threads = 2;

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = (candles.size() - 1) / num_threads; // Skip first candle since TR needs previous close

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? candles.size() - 1 : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&candles, &true_ranges, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            double high = candles[i + 1].high;
                            double low = candles[i + 1].low;
                            double prev_close = candles[i].close;

                            double tr1 = high - low;
                            double tr2 = std::abs(high - prev_close);
                            double tr3 = std::abs(low - prev_close);

                            true_ranges[i] = std::max({tr1, tr2, tr3});
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 1; i < candles.size(); ++i) {
                    double high = candles[i].high;
                    double low = candles[i].low;
                    double prev_close = candles[i-1].close;

                    double tr1 = high - low;
                    double tr2 = std::abs(high - prev_close);
                    double tr3 = std::abs(low - prev_close);

                    double true_range = std::max({tr1, tr2, tr3});
                    true_ranges[i - 1] = true_range;
                }
            }

            if (true_ranges.size() < static_cast<size_t>(period)) {
                promise->set_value(std::move(atr_values));
                return;
            }

            // Calculate ATR using SMA for the first value, then smoothed moving average
            double initial_atr = std::accumulate(true_ranges.begin(), true_ranges.begin() + period, 0.0) / period;
            atr_values.push_back(initial_atr);

            // Calculate subsequent ATR values using the smoothing formula
            // This part must remain sequential since each value depends on the previous ATR
            for (size_t i = period; i < true_ranges.size(); ++i) {
                double current_atr = ((atr_values.back() * (period - 1)) + true_ranges[i]) / period;
                atr_values.push_back(current_atr);
            }

            promise->set_value(std::move(atr_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_stochastic_oscillator_async(
    const std::vector<Candle>& candles,
    int k_period,
    int d_period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, k_period, d_period, promise]() {
        try {
            std::vector<double> k_values, d_values;

            if (candles.size() < static_cast<size_t>(k_period)) {
                promise->set_value(std::move(k_values));
                return;
            }

            k_values.reserve(candles.size() - k_period + 1);

            // Calculate %K values
            for (size_t i = k_period - 1; i < candles.size(); ++i) {
                // Find highest high and lowest low in the k_period
                double highest_high = candles[i - k_period + 1].high;
                double lowest_low = candles[i - k_period + 1].low;

                for (int j = 0; j < k_period; ++j) {
                    size_t idx = i - k_period + 1 + j;
                    if (idx < candles.size()) {
                        highest_high = std::max(highest_high, candles[idx].high);
                        lowest_low = std::min(lowest_low, candles[idx].low);
                    }
                }

                if (highest_high != lowest_low) {
                    double current_close = candles[i].close;
                    double k_value = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0;
                    k_values.push_back(k_value);
                } else {
                    // If highest high equals lowest low, %K is undefined, use 50 as neutral
                    k_values.push_back(50.0);
                }
            }

            // Calculate %D values (moving average of %K)
            if (k_values.size() >= static_cast<size_t>(d_period)) {
                d_values.reserve(k_values.size() - d_period + 1);

                for (size_t i = d_period - 1; i < k_values.size(); ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < d_period; ++j) {
                        sum += k_values[i - d_period + 1 + j];
                    }
                    double d_value = sum / d_period;
                    d_values.push_back(d_value);
                }
            }

            // Return %K values as the primary result
            promise->set_value(std::move(k_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_on_balance_volume_async(
    const std::vector<Candle>& candles) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, promise]() {
        try {
            std::vector<double> obv_values;
            if (candles.size() <= 1) {
                obv_values.resize(candles.size(), 0.0);
                promise->set_value(std::move(obv_values));
                return;
            }

            obv_values.reserve(candles.size());
            double current_obv = 0.0; // Starting OBV is typically 0
            obv_values.push_back(current_obv);

            // Sequential processing is required for OBV since each value depends on the previous
            // However, for very large datasets we can at least precompute the daily changes
            if (candles.size() > 10000) {
                // Precompute the daily changes in parallel
                std::vector<double> daily_changes(candles.size() - 1);

                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), (candles.size() - 1));
                if (num_threads < 2) num_threads = 2;

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = (candles.size() - 1) / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? candles.size() - 1 : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&candles, &daily_changes, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            if (candles[i + 1].close > candles[i].close) {
                                daily_changes[i] = candles[i + 1].volume;  // Price went up, add volume
                            } else if (candles[i + 1].close < candles[i].close) {
                                daily_changes[i] = -candles[i + 1].volume;  // Price went down, subtract volume
                            } else {
                                daily_changes[i] = 0.0;  // No change in price
                            }
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Now compute the actual OBV values sequentially using precomputed changes
                for (size_t i = 0; i < candles.size() - 1; ++i) {
                    current_obv += daily_changes[i];
                    obv_values.push_back(current_obv);
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 1; i < candles.size(); ++i) {
                    if (candles[i].close > candles[i-1].close) {
                        current_obv += candles[i].volume;
                    } else if (candles[i].close < candles[i-1].close) {
                        current_obv -= candles[i].volume;
                    }
                    // If equal, OBV remains unchanged
                    obv_values.push_back(current_obv);
                }
            }

            promise->set_value(std::move(obv_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// New method for calculating correlation between two series
std::future<double> TaskScheduler::calculate_correlation_async(
    const std::vector<double>& series1,
    const std::vector<double>& series2) {

    auto promise = std::make_shared<std::promise<double>>();
    auto future = promise->get_future();

    enqueue_task([series1, series2, promise]() {
        try {
            if (series1.size() != series2.size() || series1.size() < 2) {
                promise->set_value(0.0);
                return;
            }

            size_t n = series1.size();
            double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
            double sum_x2 = 0.0, sum_y2 = 0.0;

            // Use multi-threaded approach for large datasets
            if (n > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), n);
                if (num_threads < 2) num_threads = 2;

                std::vector<std::array<double, 5>> thread_results(num_threads, {0.0, 0.0, 0.0, 0.0, 0.0});

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = n / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? n : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&series1, &series2, &thread_results, t, start, end]() {
                        double local_sum_x = 0.0, local_sum_y = 0.0, local_sum_xy = 0.0;
                        double local_sum_x2 = 0.0, local_sum_y2 = 0.0;

                        for (size_t i = start; i < end; ++i) {
                            double x = series1[i];
                            double y = series2[i];

                            local_sum_x += x;
                            local_sum_y += y;
                            local_sum_xy += x * y;
                            local_sum_x2 += x * x;
                            local_sum_y2 += y * y;
                        }

                        thread_results[t][0] = local_sum_x;
                        thread_results[t][1] = local_sum_y;
                        thread_results[t][2] = local_sum_xy;
                        thread_results[t][3] = local_sum_x2;
                        thread_results[t][4] = local_sum_y2;
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (size_t t = 0; t < num_threads; ++t) {
                    sum_x += thread_results[t][0];
                    sum_y += thread_results[t][1];
                    sum_xy += thread_results[t][2];
                    sum_x2 += thread_results[t][3];
                    sum_y2 += thread_results[t][4];
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < n; ++i) {
                    double x = series1[i];
                    double y = series2[i];

                    sum_x += x;
                    sum_y += y;
                    sum_xy += x * y;
                    sum_x2 += x * x;
                    sum_y2 += y * y;
                }
            }

            // Calculate correlation coefficient
            double numerator = n * sum_xy - sum_x * sum_y;
            double denominator_x = std::sqrt(n * sum_x2 - sum_x * sum_x);
            double denominator_y = std::sqrt(n * sum_y2 - sum_y * sum_y);
            double denominator = denominator_x * denominator_y;

            double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;

            // Clamp correlation to [-1, 1] range to handle floating-point precision issues
            correlation = std::max(-1.0, std::min(1.0, correlation));

            promise->set_value(correlation);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Advanced multi-threaded batch processing methods

std::future<std::vector<std::vector<double>>> TaskScheduler::calculate_batch_indicators_async(
    const std::vector<std::vector<double>>& price_series,
    const std::vector<std::pair<std::string, int>>& indicator_configs) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<double>>>>();
    auto future = promise->get_future();

    enqueue_task([price_series, indicator_configs, promise]() {
        try {
            std::vector<std::vector<double>> results;
            results.reserve(price_series.size());

            // Use multi-threaded approach for large datasets
            if (price_series.size() > std::thread::hardware_concurrency()) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), price_series.size());

                std::vector<std::vector<std::vector<double>>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = price_series.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? price_series.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&price_series, &indicator_configs, &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            std::vector<double> indicator_result;

                            // Apply the first configuration for demonstration
                            if (!indicator_configs.empty()) {
                                const auto& config = indicator_configs[0];
                                const auto& prices = price_series[i];

                                if (config.first == "sma") {
                                    int period = config.second;
                                    if (prices.size() >= static_cast<size_t>(period)) {
                                        double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
                                        indicator_result.push_back(sum / period); // Just the first SMA value for demo
                                    }
                                } else if (config.first == "ema") {
                                    int period = config.second;
                                    if (!prices.empty()) {
                                        double multiplier = 2.0 / (period + 1);
                                        double ema = prices[0]; // Use first value as initial for demo

                                        for (size_t j = 1; j < prices.size(); ++j) {
                                            ema = (prices[j] - ema) * multiplier + ema;
                                        }
                                        indicator_result.push_back(ema);
                                    }
                                }
                            }

                            thread_results[t].push_back(indicator_result);
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (const auto& thread_result : thread_results) {
                    results.insert(results.end(), thread_result.begin(), thread_result.end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& prices : price_series) {
                    std::vector<double> indicator_result;

                    if (!indicator_configs.empty()) {
                        const auto& config = indicator_configs[0];

                        if (config.first == "sma") {
                            int period = config.second;
                            if (prices.size() >= static_cast<size_t>(period)) {
                                double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
                                indicator_result.push_back(sum / period);
                            }
                        } else if (config.first == "ema") {
                            int period = config.second;
                            if (!prices.empty()) {
                                double multiplier = 2.0 / (period + 1);
                                double ema = prices[0]; // Use first value as initial for demo

                                for (size_t j = 1; j < prices.size(); ++j) {
                                    ema = (prices[j] - ema) * multiplier + ema;
                                }
                                indicator_result.push_back(ema);
                            }
                        }
                    }

                    results.push_back(indicator_result);
                }
            }

            promise->set_value(std::move(results));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::vector<Trade>>> TaskScheduler::process_batch_trades_async(
    const std::vector<std::vector<Trade>>& trade_batches,
    std::function<std::vector<Trade>(const std::vector<Trade>&)> processor_func) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<Trade>>>>();
    auto future = promise->get_future();

    enqueue_task([trade_batches, processor_func, promise]() {
        try {
            std::vector<std::vector<Trade>> results;
            results.reserve(trade_batches.size());

            // Use multi-threaded approach for large datasets
            if (trade_batches.size() > std::thread::hardware_concurrency()) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trade_batches.size());

                std::vector<std::vector<std::vector<Trade>>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trade_batches.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trade_batches.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trade_batches, &processor_func, &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& batch = trade_batches[i];
                            auto processed_batch = processor_func(batch);
                            thread_results[t].push_back(processed_batch);
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (const auto& thread_result : thread_results) {
                    results.insert(results.end(), thread_result.begin(), thread_result.end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& batch : trade_batches) {
                    auto processed_batch = processor_func(batch);
                    results.push_back(processed_batch);
                }
            }

            promise->set_value(std::move(results));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::vector<double>>> TaskScheduler::calculate_multiple_timeframe_indicators_async(
    const std::vector<double>& prices,
    const std::vector<std::pair<std::string, std::vector<int>>>& indicator_configs) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<double>>>>();
    auto future = promise->get_future();

    enqueue_task([prices, indicator_configs, promise]() {
        try {
            std::vector<std::vector<double>> results;
            results.reserve(indicator_configs.size());

            // Process each indicator configuration in parallel
            if (indicator_configs.size() > 1) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), indicator_configs.size());

                std::vector<std::vector<std::vector<double>>> thread_results(num_threads);
                std::vector<std::mutex> result_mutexes(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = indicator_configs.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? indicator_configs.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&prices, &indicator_configs, &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& config = indicator_configs[i];
                            const std::string& indicator_type = config.first;
                            const auto& periods = config.second;

                            std::vector<double> indicator_values;

                            for (int period : periods) {
                                if (indicator_type == "sma" && prices.size() >= static_cast<size_t>(period)) {
                                    double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
                                    indicator_values.push_back(sum / period);
                                } else if (indicator_type == "ema" && !prices.empty() && period > 0) {
                                    double multiplier = 2.0 / (period + 1);
                                    double ema = prices[0]; // Use first value as initial for demo

                                    for (size_t j = 1; j < prices.size(); ++j) {
                                        ema = (prices[j] - ema) * multiplier + ema;
                                    }
                                    indicator_values.push_back(ema);
                                } else if (indicator_type == "rsi" && prices.size() >= static_cast<size_t>(period + 1)) {
                                    // Simplified RSI calculation for demonstration
                                    double avg_gain = 0.0, avg_loss = 0.0;
                                    for (int k = 1; k <= period; ++k) {
                                        double change = prices[k] - prices[k-1];
                                        if (change > 0) {
                                            avg_gain += change;
                                        } else {
                                            avg_loss -= change;
                                        }
                                    }
                                    avg_gain /= period;
                                    avg_loss /= period;

                                    double rs = (avg_loss != 0) ? avg_gain / avg_loss : 0;
                                    double rsi = 100.0 - (100.0 / (1.0 + rs));
                                    indicator_values.push_back(rsi);
                                }
                            }

                            thread_results[t].push_back(indicator_values);
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                for (size_t t = 0; t < thread_results.size(); ++t) {
                    for (const auto& result : thread_results[t]) {
                        results.push_back(result);
                    }
                }
            } else {
                // Sequential processing for single configuration
                for (const auto& config : indicator_configs) {
                    const std::string& indicator_type = config.first;
                    const auto& periods = config.second;

                    std::vector<double> indicator_values;

                    for (int period : periods) {
                        if (indicator_type == "sma" && prices.size() >= static_cast<size_t>(period)) {
                            double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
                            indicator_values.push_back(sum / period);
                        } else if (indicator_type == "ema" && !prices.empty() && period > 0) {
                            double multiplier = 2.0 / (period + 1);
                            double ema = prices[0]; // Use first value as initial for demo

                            for (size_t j = 1; j < prices.size(); ++j) {
                                ema = (prices[j] - ema) * multiplier + ema;
                            }
                            indicator_values.push_back(ema);
                        } else if (indicator_type == "rsi" && prices.size() >= static_cast<size_t>(period + 1)) {
                            // Simplified RSI calculation for demonstration
                            double avg_gain = 0.0, avg_loss = 0.0;
                            for (int k = 1; k <= period; ++k) {
                                double change = prices[k] - prices[k-1];
                                if (change > 0) {
                                    avg_gain += change;
                                } else {
                                    avg_loss -= change;
                                }
                            }
                            avg_gain /= period;
                            avg_loss /= period;

                            double rs = (avg_loss != 0) ? avg_gain / avg_loss : 0;
                            double rsi = 100.0 - (100.0 / (1.0 + rs));
                            indicator_values.push_back(rsi);
                        }
                    }

                    results.push_back(indicator_values);
                }
            }

            promise->set_value(std::move(results));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::vector<Candle>>> TaskScheduler::aggregate_batch_candles_async(
    const std::vector<std::vector<Trade>>& trade_batches,
    std::chrono::seconds timeframe) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<Candle>>>>();
    auto future = promise->get_future();

    enqueue_task([trade_batches, timeframe, promise]() {
        try {
            std::vector<std::vector<Candle>> results;
            results.reserve(trade_batches.size());

            // Use multi-threaded approach for large datasets
            if (trade_batches.size() > std::thread::hardware_concurrency()) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trade_batches.size());

                std::vector<std::vector<std::vector<Candle>>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trade_batches.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trade_batches.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trade_batches, timeframe, &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trades = trade_batches[i];

                            std::vector<Candle> candles;
                            if (trades.empty()) {
                                thread_results[t].push_back(std::move(candles));
                                continue;
                            }

                            // Group trades by time windows
                            std::map<std::chrono::system_clock::time_point, Candle> candle_map;

                            for (const auto& trade : trades) {
                                // Calculate the start time of the candle period
                                auto time_since_epoch = trade.timestamp.time_since_epoch();
                                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                                auto period_start = std::chrono::system_clock::time_point(seconds -
                                    std::chrono::seconds(seconds.count() % timeframe.count()));

                                if (candle_map.find(period_start) == candle_map.end()) {
                                    // Initialize new candle
                                    candle_map[period_start] = {
                                        period_start,
                                        trade.price,  // Open
                                        trade.price,  // High
                                        trade.price,  // Low
                                        trade.price,  // Close
                                        trade.volume  // Volume
                                    };
                                } else {
                                    // Update existing candle
                                    auto& candle = candle_map[period_start];
                                    candle.high = std::max(candle.high, trade.price);
                                    candle.low = std::min(candle.low, trade.price);
                                    candle.close = trade.price;  // Last price becomes close
                                    candle.volume += trade.volume;
                                }
                            }

                            // Convert map to vector
                            std::vector<Candle> batch_candles;
                            batch_candles.reserve(candle_map.size());
                            for (const auto& pair : candle_map) {
                                batch_candles.push_back(pair.second);
                            }

                            thread_results[t].push_back(std::move(batch_candles));
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (const auto& thread_result : thread_results) {
                    results.insert(results.end(), thread_result.begin(), thread_result.end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& trades : trade_batches) {
                    std::vector<Candle> candles;
                    if (trades.empty()) {
                        results.push_back(std::move(candles));
                        continue;
                    }

                    // Group trades by time windows
                    std::map<std::chrono::system_clock::time_point, Candle> candle_map;

                    for (const auto& trade : trades) {
                        // Calculate the start time of the candle period
                        auto time_since_epoch = trade.timestamp.time_since_epoch();
                        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                        auto period_start = std::chrono::system_clock::time_point(seconds -
                            std::chrono::seconds(seconds.count() % timeframe.count()));

                        if (candle_map.find(period_start) == candle_map.end()) {
                            // Initialize new candle
                            candle_map[period_start] = {
                                period_start,
                                trade.price,  // Open
                                trade.price,  // High
                                trade.price,  // Low
                                trade.price,  // Close
                                trade.volume  // Volume
                            };
                        } else {
                            // Update existing candle
                            auto& candle = candle_map[period_start];
                            candle.high = std::max(candle.high, trade.price);
                            candle.low = std::min(candle.low, trade.price);
                            candle.close = trade.price;  // Last price becomes close
                            candle.volume += trade.volume;
                        }
                    }

                    // Convert map to vector
                    candles.reserve(candle_map.size());
                    for (const auto& pair : candle_map) {
                        candles.push_back(pair.second);
                    }

                    results.push_back(std::move(candles));
                }
            }

            promise->set_value(std::move(results));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Enhanced multi-threaded methods for volume calculations

std::future<std::vector<double>> TaskScheduler::calculate_rolling_volume_profile_async(
    const std::vector<Trade>& trades,
    double min_price,
    double max_price,
    int resolution,
    int window_size) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, min_price, max_price, resolution, window_size, promise]() {
        try {
            std::vector<double> rolling_volume_profile;

            if (trades.size() < static_cast<size_t>(window_size) || resolution <= 0) {
                promise->set_value(std::move(rolling_volume_profile));
                return;
            }

            rolling_volume_profile.reserve(trades.size() - window_size + 1);
            double price_range = max_price - min_price;
            if (price_range <= 0) {
                std::fill_n(std::back_inserter(rolling_volume_profile),
                           trades.size() - window_size + 1, 0.0);
                promise->set_value(std::move(rolling_volume_profile));
                return;
            }

            double bin_size = price_range / resolution;

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads);

                // Calculate how many rolling windows each thread will process
                size_t total_windows = trades.size() - window_size + 1;
                size_t chunk_size = total_windows / num_threads;

                std::vector<std::thread> processing_threads;
                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start_window = t * chunk_size;
                    size_t end_window = (t == num_threads - 1) ? total_windows : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, min_price, bin_size, resolution, window_size,
                                                   &thread_results, t, start_window, end_window]() {
                        for (size_t win_idx = start_window; win_idx < end_window; ++win_idx) {
                            std::vector<double> window_profile(resolution, 0.0);

                            // Calculate volume profile for this window
                            for (int offset = 0; offset < window_size; ++offset) {
                                size_t trade_idx = win_idx + offset;
                                if (trade_idx < trades.size()) {
                                    const auto& trade = trades[trade_idx];
                                    int bin_index = static_cast<int>((trade.price - min_price) / bin_size);
                                    if (bin_index >= 0 && bin_index < resolution) {
                                        window_profile[bin_index] += trade.volume;
                                    }
                                }
                            }

                            // For this example, we'll return the total volume in the highest volume bin
                            double max_bin_volume = *std::max_element(window_profile.begin(), window_profile.end());
                            thread_results[t].push_back(max_bin_volume);
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                rolling_volume_profile.reserve(total_windows);
                for (size_t t = 0; t < num_threads; ++t) {
                    rolling_volume_profile.insert(rolling_volume_profile.end(),
                                                thread_results[t].begin(),
                                                thread_results[t].end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i <= trades.size() - window_size; ++i) {
                    std::vector<double> window_profile(resolution, 0.0);

                    // Calculate volume profile for this window
                    for (int j = 0; j < window_size; ++j) {
                        const auto& trade = trades[i + j];
                        int bin_index = static_cast<int>((trade.price - min_price) / bin_size);
                        if (bin_index >= 0 && bin_index < resolution) {
                            window_profile[bin_index] += trade.volume;
                        }
                    }

                    // For this example, we'll return the total volume in the highest volume bin
                    double max_bin_volume = *std::max_element(window_profile.begin(), window_profile.end());
                    rolling_volume_profile.push_back(max_bin_volume);
                }
            }

            promise->set_value(std::move(rolling_volume_profile));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_time_based_volume_async(
    const std::vector<Trade>& trades,
    int time_resolution_seconds) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, time_resolution_seconds, promise]() {
        try {
            if (trades.empty() || time_resolution_seconds <= 0) {
                promise->set_value(std::vector<double>());
                return;
            }

            // Determine time range
            auto min_max_it = std::minmax_element(trades.begin(), trades.end(),
                [](const Trade& a, const Trade& b) {
                    return a.timestamp < b.timestamp;
                });

            auto start_time = min_max_it.first->timestamp;
            auto end_time = min_max_it.second->timestamp;

            // Calculate number of time bins
            auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();
            int num_bins = static_cast<int>(std::ceil(static_cast<double>(duration_seconds) /
                                                      time_resolution_seconds)) + 1;

            std::vector<double> volume_by_time(num_bins, 0.0);
            auto bin_duration = std::chrono::seconds(time_resolution_seconds);

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                // Use atomic operations or temporary storage per thread to avoid race conditions
                std::vector<std::vector<double>> thread_results(num_threads, std::vector<double>(num_bins, 0.0));

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, start_time, time_resolution_seconds, num_bins,
                                                   &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                                trade.timestamp - start_time).count();
                            int bin_index = static_cast<int>(time_diff / time_resolution_seconds);

                            if (bin_index >= 0 && bin_index < num_bins) {
                                thread_results[t][bin_index] += trade.volume;
                            }
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (int i = 0; i < num_bins; ++i) {
                    for (size_t t = 0; t < num_threads; ++t) {
                        volume_by_time[i] += thread_results[t][i];
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& trade : trades) {
                    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                        trade.timestamp - start_time).count();
                    int bin_index = static_cast<int>(time_diff / time_resolution_seconds);

                    if (bin_index >= 0 && bin_index < num_bins) {
                        volume_by_time[bin_index] += trade.volume;
                    }
                }
            }

            promise->set_value(std::move(volume_by_time));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Enhanced multi-threaded methods for indicator computations

std::future<std::vector<double>> TaskScheduler::calculate_adaptive_sma_async(
    const std::vector<double>& prices,
    int min_period,
    int max_period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, min_period, max_period, promise]() {
        try {
            std::vector<double> adaptive_sma_values;

            if (prices.size() < static_cast<size_t>(min_period) || min_period >= max_period || min_period <= 0) {
                adaptive_sma_values.resize(prices.size());
                promise->set_value(std::move(adaptive_sma_values));
                return;
            }

            adaptive_sma_values.reserve(prices.size());

            // Use multi-threaded approach for large datasets
            if (prices.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), prices.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = prices.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? prices.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([start, end, &prices, min_period, max_period, &thread_results, t]() {
                        std::vector<double> local_results;

                        for (size_t i = start; i < end; ++i) {
                            if (i < static_cast<size_t>(min_period - 1)) {
                                // Not enough data points yet
                                local_results.push_back(prices[i]);
                                continue;
                            }

                            // Determine the adaptive period based on volatility (simplified approach)
                            size_t lookback = std::min(i + 1, static_cast<size_t>(max_period));
                            size_t period = std::max(min_period, static_cast<int>(lookback));

                            // Calculate SMA for the adaptive period
                            size_t actual_start = (i >= static_cast<size_t>(period - 1)) ? i - (period - 1) : 0;
                            double sum = 0.0;

                            for (size_t p = 0; p < static_cast<size_t>(period) && (actual_start + p) <= static_cast<size_t>(i); ++p) {
                                sum += prices[actual_start + p];
                            }

                            double sma = sum / period;
                            local_results.push_back(sma);
                        }

                        thread_results[t] = std::move(local_results);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                adaptive_sma_values.reserve(prices.size());
                for (size_t t = 0; t < num_threads; ++t) {
                    adaptive_sma_values.insert(adaptive_sma_values.end(),
                                             thread_results[t].begin(),
                                             thread_results[t].end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < prices.size(); ++i) {
                    if (i < static_cast<size_t>(min_period - 1)) {
                        // Not enough data points yet
                        adaptive_sma_values.push_back(prices[i]);
                        continue;
                    }

                    // Determine the adaptive period based on volatility (simplified approach)
                    size_t lookback = std::min(i + 1, static_cast<size_t>(max_period));
                    size_t period = std::max(min_period, static_cast<int>(lookback));

                    // Calculate SMA for the adaptive period
                    size_t actual_start = (i >= static_cast<size_t>(period - 1)) ? i - (period - 1) : 0;
                    double sum = 0.0;

                    for (size_t p = 0; p < static_cast<size_t>(period) && (actual_start + p) <= static_cast<size_t>(i); ++p) {
                        sum += prices[actual_start + p];
                    }

                    double sma = sum / period;
                    adaptive_sma_values.push_back(sma);
                }
            }

            promise->set_value(std::move(adaptive_sma_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_parabolic_sar_async(
    const std::vector<Candle>& candles,
    double acceleration_factor_step,
    double max_acceleration_factor) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, acceleration_factor_step, max_acceleration_factor, promise]() {
        try {
            std::vector<double> sar_values;

            if (candles.size() < 2) {
                sar_values.resize(candles.size(), 0.0);
                promise->set_value(std::move(sar_values));
                return;
            }

            sar_values.reserve(candles.size());

            // For Parabolic SAR, we need to determine initial trend based on first few candles
            // This is difficult to parallelize completely, but we can parallelize some calculations

            // Initialize with first candle's close
            sar_values.push_back(candles[0].close);

            // Sequential calculation is required for SAR since each value depends on the previous
            // However, for very large datasets we can precompute some values in parallel
            if (candles.size() > 10000) {
                // Precompute extreme points and acceleration factors in parallel for optimization
                std::vector<double> ep_values(candles.size(), 0.0);  // Extreme Point
                std::vector<double> af_values(candles.size(), acceleration_factor_step);  // Acceleration Factor

                // This is a simplified version - full Parabolic SAR calculation requires sequential processing
                // But we can at least parallelize some initial computations
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), candles.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::thread> processing_threads;
                size_t chunk_size = (candles.size() - 1) / num_threads;  // Skip first candle

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = 1 + t * chunk_size;  // Start from second candle
                    size_t end = (t == num_threads - 1) ? candles.size() : 1 + (t + 1) * chunk_size;

                    processing_threads.emplace_back([&candles, &ep_values, &af_values, acceleration_factor_step, max_acceleration_factor, start, end]() {
                        // Initialize EP and AF values (simplified)
                        for (size_t i = start; i < end; ++i) {
                            ep_values[i] = std::max(candles[i].high, candles[i-1].high);  // Simplified EP calculation
                            af_values[i] = acceleration_factor_step;  // Start with minimum AF
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Now calculate SAR values sequentially using precomputed values
                bool is_long = true;  // Assume initial trend is long
                double sar = candles[0].close;
                double ep = candles[0].high;
                double af = acceleration_factor_step;

                sar_values[0] = sar;

                for (size_t i = 1; i < candles.size(); ++i) {
                    // Simplified Parabolic SAR calculation
                    sar = sar + af * (ep - sar);

                    // Check if trend should reverse
                    bool should_reverse = (is_long && sar > candles[i].low) || (!is_long && sar < candles[i].high);

                    if (should_reverse) {
                        is_long = !is_long;
                        sar = ep;  // Set SAR to the extreme point of previous trend
                        ep = is_long ? candles[i].high : candles[i].low;
                        af = acceleration_factor_step;  // Reset acceleration factor
                    } else {
                        // Update EP if appropriate
                        if (is_long && candles[i].high > ep) {
                            ep = candles[i].high;
                            if (af < max_acceleration_factor) {
                                af = std::min(af + acceleration_factor_step, max_acceleration_factor);
                            }
                        } else if (!is_long && candles[i].low < ep) {
                            ep = candles[i].low;
                            if (af < max_acceleration_factor) {
                                af = std::min(af + acceleration_factor_step, max_acceleration_factor);
                            }
                        }
                    }

                    sar_values.push_back(sar);
                }
            } else {
                // Sequential processing for smaller datasets
                bool is_long = true;  // Assume initial trend is long
                double sar = candles[0].close;
                double ep = candles[0].high;
                double af = acceleration_factor_step;

                for (size_t i = 1; i < candles.size(); ++i) {
                    // Simplified Parabolic SAR calculation
                    sar = sar + af * (ep - sar);

                    // Check if trend should reverse
                    bool should_reverse = (is_long && sar > candles[i].low) || (!is_long && sar < candles[i].high);

                    if (should_reverse) {
                        is_long = !is_long;
                        sar = ep;  // Set SAR to the extreme point of previous trend
                        ep = is_long ? candles[i].high : candles[i].low;
                        af = acceleration_factor_step;  // Reset acceleration factor
                    } else {
                        // Update EP if appropriate
                        if (is_long && candles[i].high > ep) {
                            ep = candles[i].high;
                            if (af < max_acceleration_factor) {
                                af = std::min(af + acceleration_factor_step, max_acceleration_factor);
                            }
                        } else if (!is_long && candles[i].low < ep) {
                            ep = candles[i].low;
                            if (af < max_acceleration_factor) {
                                af = std::min(af + acceleration_factor_step, max_acceleration_factor);
                            }
                        }
                    }

                    sar_values.push_back(sar);
                }
            }

            promise->set_value(std::move(sar_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Enhanced multi-threaded methods for data processing

std::future<std::vector<std::vector<Trade>>> TaskScheduler::partition_and_process_trades_async(
    const std::vector<Trade>& trades,
    std::function<std::vector<Trade>(const std::vector<Trade>&)> processor_func,
    int num_partitions) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<Trade>>>>();
    auto future = promise->get_future();

    enqueue_task([trades, processor_func, num_partitions, promise]() {
        try {
            std::vector<std::vector<Trade>> results;

            if (trades.empty()) {
                promise->set_value(std::move(results));
                return;
            }

            // Determine number of partitions
            size_t actual_partitions = static_cast<size_t>(num_partitions);
            if (actual_partitions == 0) {
                actual_partitions = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (actual_partitions == 0) actual_partitions = 1;
            }

            // Use multi-threaded approach for large datasets
            if (actual_partitions > 1 && trades.size() > 1000) {
                std::vector<std::vector<Trade>> partitions(actual_partitions);
                std::vector<std::thread> processing_threads;

                // Divide trades into partitions
                size_t partition_size = trades.size() / actual_partitions;
                std::vector<std::pair<size_t, size_t>> partition_bounds;

                for (size_t i = 0; i < actual_partitions; ++i) {
                    size_t start = i * partition_size;
                    size_t end = (i == actual_partitions - 1) ? trades.size() : (i + 1) * partition_size;
                    partition_bounds.emplace_back(start, end);
                }

                // Process each partition in parallel
                std::vector<std::future<std::vector<Trade>>> futures;
                for (size_t i = 0; i < actual_partitions; ++i) {
                    auto partition_trades = std::vector<Trade>(
                        trades.begin() + partition_bounds[i].first,
                        trades.begin() + partition_bounds[i].second
                    );

                    // Create packaged_task to get a future
                    auto task = std::make_shared<std::packaged_task<std::vector<Trade>()>>(
                        [processor_func, partition_trades]() {
                            return processor_func(partition_trades);
                        }
                    );

                    futures.push_back(task->get_future());

                    processing_threads.emplace_back([task]() {
                        (*task)();
                    });
                }

                // Wait for all threads to complete and collect results
                results.reserve(actual_partitions);
                for (auto& future : futures) {
                    results.push_back(future.get());
                }

                // Wait for all threads to finish
                for (auto& thread : processing_threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                results.push_back(processor_func(trades));
            }

            promise->set_value(std::move(results));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<Candle>> TaskScheduler::create_dynamic_timeframe_candles_async(
    const std::vector<Trade>& trades,
    std::chrono::seconds base_timeframe,
    double volume_threshold) {

    auto promise = std::make_shared<std::promise<std::vector<Candle>>>();
    auto future = promise->get_future();

    enqueue_task([trades, base_timeframe, volume_threshold, promise]() {
        try {
            std::vector<Candle> dynamic_candles;

            if (trades.empty()) {
                promise->set_value(std::move(dynamic_candles));
                return;
            }

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                // Partition trades for parallel processing
                std::vector<std::vector<Trade>> thread_partitions(num_threads);
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    for (size_t i = start; i < end; ++i) {
                        thread_partitions[t].push_back(trades[i]);
                    }
                }

                // Process each partition in parallel
                std::vector<std::thread> processing_threads;
                std::vector<std::vector<Candle>> thread_results(num_threads);

                for (size_t t = 0; t < num_threads; ++t) {
                    processing_threads.emplace_back([&thread_partitions, base_timeframe, volume_threshold, &thread_results, t]() {
                        const auto& partition = thread_partitions[t];
                        std::vector<Candle> local_candles;

                        if (!partition.empty()) {
                            // Create dynamic timeframe candles for this partition
                            std::map<std::chrono::system_clock::time_point, Candle> candle_map;

                            for (const auto& trade : partition) {
                                // Calculate the start time of the base timeframe period
                                auto time_since_epoch = trade.timestamp.time_since_epoch();
                                auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                                auto period_start = std::chrono::system_clock::time_point(seconds -
                                    std::chrono::seconds(seconds.count() % base_timeframe.count()));

                                if (candle_map.find(period_start) == candle_map.end()) {
                                    // Initialize new candle
                                    candle_map[period_start] = {
                                        period_start,
                                        trade.price,  // Open
                                        trade.price,  // High
                                        trade.price,  // Low
                                        trade.price,  // Close
                                        trade.volume  // Volume
                                    };
                                } else {
                                    // Update existing candle
                                    auto& candle = candle_map[period_start];
                                    candle.high = std::max(candle.high, trade.price);
                                    candle.low = std::min(candle.low, trade.price);
                                    candle.close = trade.price;  // Last price becomes close
                                    candle.volume += trade.volume;
                                }
                            }

                            // Convert map to vector
                            for (const auto& pair : candle_map) {
                                local_candles.push_back(pair.second);
                            }
                        }

                        thread_results[t] = std::move(local_candles);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results from all threads
                std::map<std::chrono::system_clock::time_point, Candle> final_candle_map;

                for (const auto& thread_result : thread_results) {
                    for (const auto& candle : thread_result) {
                        auto it = final_candle_map.find(candle.timestamp);
                        if (it == final_candle_map.end()) {
                            final_candle_map[candle.timestamp] = candle;
                        } else {
                            // Merge candles with same timestamp
                            auto& existing_candle = it->second;
                            existing_candle.high = std::max(existing_candle.high, candle.high);
                            existing_candle.low = std::min(existing_candle.low, candle.low);
                            existing_candle.close = candle.close;  // Use the later close price
                            existing_candle.volume += candle.volume;
                        }
                    }
                }

                // Apply volume threshold filtering if needed
                for (const auto& pair : final_candle_map) {
                    if (pair.second.volume >= volume_threshold) {
                        dynamic_candles.push_back(pair.second);
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                std::map<std::chrono::system_clock::time_point, Candle> candle_map;

                for (const auto& trade : trades) {
                    // Calculate the start time of the base timeframe period
                    auto time_since_epoch = trade.timestamp.time_since_epoch();
                    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                    auto period_start = std::chrono::system_clock::time_point(seconds -
                        std::chrono::seconds(seconds.count() % base_timeframe.count()));

                    if (candle_map.find(period_start) == candle_map.end()) {
                        // Initialize new candle
                        candle_map[period_start] = {
                            period_start,
                            trade.price,  // Open
                            trade.price,  // High
                            trade.price,  // Low
                            trade.price,  // Close
                            trade.volume  // Volume
                        };
                    } else {
                        // Update existing candle
                        auto& candle = candle_map[period_start];
                        candle.high = std::max(candle.high, trade.price);
                        candle.low = std::min(candle.low, trade.price);
                        candle.close = trade.price;  // Last price becomes close
                        candle.volume += trade.volume;
                    }
                }

                // Apply volume threshold filtering
                for (const auto& pair : candle_map) {
                    if (pair.second.volume >= volume_threshold) {
                        dynamic_candles.push_back(pair.second);
                    }
                }
            }

            promise->set_value(std::move(dynamic_candles));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional multi-threaded methods for volume calculations

std::future<std::vector<double>> TaskScheduler::calculate_volume_at_price_levels_async(
    const std::vector<Trade>& trades,
    const std::vector<double>& price_levels) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, price_levels, promise]() {
        try {
            std::vector<double> volumes_at_levels(price_levels.size(), 0.0);

            if (trades.empty() || price_levels.empty()) {
                promise->set_value(std::move(volumes_at_levels));
                return;
            }

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000 || price_levels.size() > 1000) {
                size_t num_threads = std::min({
                    static_cast<size_t>(std::thread::hardware_concurrency()),
                    trades.size(),
                    static_cast<size_t>(price_levels.size())
                });
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads, std::vector<double>(price_levels.size(), 0.0));

                // Process trades in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, &price_levels, &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];

                            // Find the closest price level to the trade price
                            double min_diff = std::abs(trade.price - price_levels[0]);
                            size_t closest_level_idx = 0;

                            for (size_t j = 1; j < price_levels.size(); ++j) {
                                double diff = std::abs(trade.price - price_levels[j]);
                                if (diff < min_diff) {
                                    min_diff = diff;
                                    closest_level_idx = j;
                                }
                            }

                            // Add volume to the closest level
                            thread_results[t][closest_level_idx] += trade.volume;
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (size_t i = 0; i < price_levels.size(); ++i) {
                    for (size_t t = 0; t < num_threads; ++t) {
                        volumes_at_levels[i] += thread_results[t][i];
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& trade : trades) {
                    // Find the closest price level to the trade price
                    double min_diff = std::abs(trade.price - price_levels[0]);
                    size_t closest_level_idx = 0;

                    for (size_t j = 1; j < price_levels.size(); ++j) {
                        double diff = std::abs(trade.price - price_levels[j]);
                        if (diff < min_diff) {
                            min_diff = diff;
                            closest_level_idx = j;
                        }
                    }

                    // Add volume to the closest level
                    volumes_at_levels[closest_level_idx] += trade.volume;
                }
            }

            promise->set_value(std::move(volumes_at_levels));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_time_weighted_volume_async(
    const std::vector<Trade>& trades,
    int time_window_minutes) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, time_window_minutes, promise]() {
        try {
            std::vector<double> twv_values;

            if (trades.empty() || time_window_minutes <= 0) {
                twv_values.resize(trades.size(), 0.0);
                promise->set_value(std::move(twv_values));
                return;
            }

            twv_values.reserve(trades.size());

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                // Pre-calculate time windows in parallel
                std::vector<std::vector<double>> thread_results(num_threads);

                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, time_window_minutes, &thread_results, t, start, end]() {
                        std::vector<double> local_twv_values;

                        for (size_t i = start; i < end; ++i) {
                            const auto& current_trade = trades[i];
                            double total_volume = 0.0;
                            double weighted_volume = 0.0;

                            // Look back within the time window
                            auto time_threshold = current_trade.timestamp - std::chrono::minutes(time_window_minutes);

                            for (int j = static_cast<int>(i); j >= 0; --j) {
                                if (trades[j].timestamp < time_threshold) {
                                    break; // Exceeded time window
                                }

                                auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                                    current_trade.timestamp - trades[j].timestamp).count();

                                // Weight decreases linearly with time distance
                                double weight = std::max(0.0, 1.0 - (static_cast<double>(time_diff) /
                                           (time_window_minutes * 60.0)));

                                weighted_volume += trades[j].volume * weight;
                                total_volume += trades[j].volume;
                            }

                            double twv = (total_volume > 0) ? weighted_volume / total_volume : 0.0;
                            local_twv_values.push_back(twv);
                        }

                        thread_results[t] = std::move(local_twv_values);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                twv_values.reserve(trades.size());
                for (size_t t = 0; t < num_threads; ++t) {
                    twv_values.insert(twv_values.end(),
                                      thread_results[t].begin(),
                                      thread_results[t].end());
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < trades.size(); ++i) {
                    const auto& current_trade = trades[i];
                    double total_volume = 0.0;
                    double weighted_volume = 0.0;

                    // Look back within the time window
                    auto time_threshold = current_trade.timestamp - std::chrono::minutes(time_window_minutes);

                    for (int j = static_cast<int>(i); j >= 0; --j) {
                        if (trades[j].timestamp < time_threshold) {
                            break; // Exceeded time window
                        }

                        auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                            current_trade.timestamp - trades[j].timestamp).count();

                        // Weight decreases linearly with time distance
                        double weight = std::max(0.0, 1.0 - (static_cast<double>(time_diff) /
                                   (time_window_minutes * 60.0)));

                        weighted_volume += trades[j].volume * weight;
                        total_volume += trades[j].volume;
                    }

                    double twv = (total_volume > 0) ? weighted_volume / total_volume : 0.0;
                    twv_values.push_back(twv);
                }
            }

            promise->set_value(std::move(twv_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional multi-threaded methods for indicator computations

std::future<std::vector<double>> TaskScheduler::calculate_hull_moving_average_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> hma_values;

            if (prices.size() < static_cast<size_t>(period) || period <= 0) {
                hma_values.resize(prices.size());
                promise->set_value(std::move(hma_values));
                return;
            }

            hma_values.reserve(prices.size() - period + 1);

            // The Hull Moving Average formula: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
            // For efficiency, we'll calculate this in parts

            // First, calculate WMA(n/2) and WMA(n) using multi-threading
            int half_period = period / 2;
            if (half_period < 1) half_period = 1;

            // Calculate WMA(n/2)
            auto wma_half_future = std::async(std::launch::async, [&prices, half_period]() {
                std::vector<double> wma_half;

                if (prices.size() < static_cast<size_t>(half_period)) {
                    wma_half.resize(prices.size());
                    return wma_half;
                }

                wma_half.reserve(prices.size() - half_period + 1);

                for (size_t i = half_period - 1; i < prices.size(); ++i) {
                    double numerator = 0.0;
                    double denominator = 0.0;

                    for (int j = 0; j < half_period; ++j) {
                        double weight = static_cast<double>(j + 1);
                        numerator += prices[i - half_period + 1 + j] * weight;
                        denominator += weight;
                    }

                    wma_half.push_back(numerator / denominator);
                }

                return wma_half;
            });

            // Calculate WMA(n)
            auto wma_full_future = std::async(std::launch::async, [&prices, period]() {
                std::vector<double> wma_full;

                if (prices.size() < static_cast<size_t>(period)) {
                    wma_full.resize(prices.size());
                    return wma_full;
                }

                wma_full.reserve(prices.size() - period + 1);

                for (size_t i = period - 1; i < prices.size(); ++i) {
                    double numerator = 0.0;
                    double denominator = 0.0;

                    for (int j = 0; j < period; ++j) {
                        double weight = static_cast<double>(j + 1);
                        numerator += prices[i - period + 1 + j] * weight;
                        denominator += weight;
                    }

                    wma_full.push_back(numerator / denominator);
                }

                return wma_full;
            });

            // Wait for WMAs to be calculated
            auto wma_half = wma_half_future.get();
            auto wma_full = wma_full_future.get();

            // Calculate 2*WMA(n/2) - WMA(n)
            std::vector<double> diff_values;
            size_t min_size = std::min(wma_half.size(), wma_full.size());
            diff_values.reserve(min_size);

            for (size_t i = 0; i < min_size; ++i) {
                diff_values.push_back(2.0 * wma_half[i] - wma_full[i]);
            }

            // Now calculate WMA of the difference with period sqrt(n)
            int sqrt_period = static_cast<int>(std::sqrt(static_cast<double>(period)));
            if (sqrt_period < 1) sqrt_period = 1;

            if (diff_values.size() >= static_cast<size_t>(sqrt_period)) {
                // Use multi-threaded approach for large datasets
                if (diff_values.size() > 10000) {
                    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), diff_values.size());
                    if (num_threads < 2) num_threads = 2;

                    hma_values.resize(diff_values.size() - sqrt_period + 1);

                    std::vector<std::thread> processing_threads;
                    size_t chunk_size = (diff_values.size() - sqrt_period + 1) / num_threads;

                    for (size_t t = 0; t < num_threads; ++t) {
                        size_t start = t * chunk_size;
                        size_t end = (t == num_threads - 1) ? (diff_values.size() - sqrt_period + 1) : (start + chunk_size);

                        processing_threads.emplace_back([&diff_values, sqrt_period, &hma_values, start, end]() {
                            for (size_t i = start; i < end; ++i) {
                                double numerator = 0.0;
                                double denominator = 0.0;

                                for (int j = 0; j < sqrt_period; ++j) {
                                    double weight = static_cast<double>(j + 1);
                                    numerator += diff_values[i + j] * weight;
                                    denominator += weight;
                                }

                                hma_values[i] = numerator / denominator;
                            }
                        });
                    }

                    // Wait for all threads to complete
                    for (auto& thread : processing_threads) {
                        thread.join();
                    }
                } else {
                    // Sequential processing for smaller datasets
                    for (size_t i = 0; i <= diff_values.size() - sqrt_period; ++i) {
                        double numerator = 0.0;
                        double denominator = 0.0;

                        for (int j = 0; j < sqrt_period; ++j) {
                            double weight = static_cast<double>(j + 1);
                            numerator += diff_values[i + j] * weight;
                            denominator += weight;
                        }

                        hma_values.push_back(numerator / denominator);
                    }
                }
            } else {
                hma_values.resize(diff_values.size());
            }

            promise->set_value(std::move(hma_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_triangular_moving_average_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> tma_values;

            if (prices.size() < static_cast<size_t>(period) || period <= 0) {
                tma_values.resize(prices.size());
                promise->set_value(std::move(tma_values));
                return;
            }

            // Triangular Moving Average = SMA of SMA
            // Period for inner SMA = ceil(period/2)
            int inner_period = static_cast<int>(std::ceil(static_cast<double>(period) / 2.0));

            // Calculate first SMA
            std::vector<double> first_sma;

            if (prices.size() >= static_cast<size_t>(inner_period)) {
                first_sma.reserve(prices.size() - inner_period + 1);

                double sum = std::accumulate(prices.begin(), prices.begin() + inner_period, 0.0);
                first_sma.push_back(sum / inner_period);

                for (size_t i = inner_period; i < prices.size(); ++i) {
                    sum += prices[i] - prices[i - inner_period];
                    first_sma.push_back(sum / inner_period);
                }
            }

            // Calculate second SMA (on the first SMA results)
            if (first_sma.size() >= static_cast<size_t>(inner_period)) {
                tma_values.reserve(first_sma.size() - inner_period + 1);

                // Use multi-threaded approach for large datasets
                if (first_sma.size() > 10000) {
                    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), first_sma.size());
                    if (num_threads < 2) num_threads = 2;

                    tma_values.resize(first_sma.size() - inner_period + 1);

                    std::vector<std::thread> processing_threads;
                    size_t chunk_size = (first_sma.size() - inner_period + 1) / num_threads;

                    for (size_t t = 0; t < num_threads; ++t) {
                        size_t start = t * chunk_size;
                        size_t end = (t == num_threads - 1) ? (first_sma.size() - inner_period + 1) : (start + chunk_size);

                        processing_threads.emplace_back([&first_sma, inner_period, &tma_values, start, end]() {
                            double local_sum = std::accumulate(first_sma.begin() + start, first_sma.begin() + start + inner_period, 0.0);
                            tma_values[start] = local_sum / inner_period;

                            for (size_t i = start + 1; i < end; ++i) {
                                local_sum += first_sma[i + inner_period - 1] - first_sma[i - 1];
                                tma_values[i] = local_sum / inner_period;
                            }
                        });
                    }

                    // Wait for all threads to complete
                    for (auto& thread : processing_threads) {
                        thread.join();
                    }
                } else {
                    // Sequential processing for smaller datasets
                    double sum = std::accumulate(first_sma.begin(), first_sma.begin() + inner_period, 0.0);
                    tma_values.push_back(sum / inner_period);

                    for (size_t i = inner_period; i < first_sma.size(); ++i) {
                        sum += first_sma[i] - first_sma[i - inner_period];
                        tma_values.push_back(sum / inner_period);
                    }
                }
            } else {
                tma_values.resize(first_sma.size());
            }

            promise->set_value(std::move(tma_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional multi-threaded methods for data processing

std::future<std::vector<std::vector<double>>> TaskScheduler::calculate_normalized_correlation_matrix_async(
    const std::vector<std::vector<double>>& data_series) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<double>>>>();
    auto future = promise->get_future();

    enqueue_task([data_series, promise]() {
        try {
            std::vector<std::vector<double>> correlation_matrix;

            if (data_series.empty()) {
                promise->set_value(std::move(correlation_matrix));
                return;
            }

            size_t n = data_series.size();
            correlation_matrix.resize(n, std::vector<double>(n, 0.0));

            // Fill diagonal with 1.0 (each series is perfectly correlated with itself)
            for (size_t i = 0; i < n; ++i) {
                correlation_matrix[i][i] = 1.0;
            }

            // Use multi-threaded approach for large datasets
            if (n > 10) {
                // Create a triangular matrix of correlations to calculate
                std::vector<std::pair<size_t, size_t>> pairs_to_calculate;
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = i + 1; j < n; ++j) {
                        pairs_to_calculate.emplace_back(i, j);
                    }
                }

                size_t num_pairs = pairs_to_calculate.size();
                if (num_pairs > 0) {
                    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), num_pairs);
                    if (num_threads < 2) num_threads = 2;

                    std::vector<std::thread> processing_threads;
                    size_t chunk_size = num_pairs / num_threads;

                    for (size_t t = 0; t < num_threads; ++t) {
                        size_t start = t * chunk_size;
                        size_t end = (t == num_threads - 1) ? num_pairs : (t + 1) * chunk_size;

                        processing_threads.emplace_back([&pairs_to_calculate, &data_series, &correlation_matrix, start, end]() {
                            for (size_t idx = start; idx < end; ++idx) {
                                size_t i = pairs_to_calculate[idx].first;
                                size_t j = pairs_to_calculate[idx].second;

                                // Calculate correlation between series i and j
                                const auto& series_i = data_series[i];
                                const auto& series_j = data_series[j];

                                if (series_i.size() != series_j.size() || series_i.size() < 2) {
                                    correlation_matrix[i][j] = 0.0;
                                    correlation_matrix[j][i] = 0.0;
                                    continue;
                                }

                                size_t n_vals = series_i.size();
                                double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                                double sum_x2 = 0.0, sum_y2 = 0.0;

                                for (size_t k = 0; k < n_vals; ++k) {
                                    double x = series_i[k];
                                    double y = series_j[k];

                                    sum_x += x;
                                    sum_y += y;
                                    sum_xy += x * y;
                                    sum_x2 += x * x;
                                    sum_y2 += y * y;
                                }

                                double numerator = n_vals * sum_xy - sum_x * sum_y;
                                double denominator_x = std::sqrt(n_vals * sum_x2 - sum_x * sum_x);
                                double denominator_y = std::sqrt(n_vals * sum_y2 - sum_y * sum_y);
                                double denominator = denominator_x * denominator_y;

                                double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;

                                // Clamp correlation to [-1, 1] range to handle floating-point precision issues
                                correlation = std::max(-1.0, std::min(1.0, correlation));

                                correlation_matrix[i][j] = correlation;
                                correlation_matrix[j][i] = correlation;
                            }
                        });
                    }

                    // Wait for all threads to complete
                    for (auto& thread : processing_threads) {
                        thread.join();
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = i + 1; j < n; ++j) {
                        // Calculate correlation between series i and j
                        const auto& series_i = data_series[i];
                        const auto& series_j = data_series[j];

                        if (series_i.size() != series_j.size() || series_i.size() < 2) {
                            correlation_matrix[i][j] = 0.0;
                            correlation_matrix[j][i] = 0.0;
                            continue;
                        }

                        size_t n_vals = series_i.size();
                        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                        double sum_x2 = 0.0, sum_y2 = 0.0;

                        for (size_t k = 0; k < n_vals; ++k) {
                            double x = series_i[k];
                            double y = series_j[k];

                            sum_x += x;
                            sum_y += y;
                            sum_xy += x * y;
                            sum_x2 += x * x;
                            sum_y2 += y * y;
                        }

                        double numerator = n_vals * sum_xy - sum_x * sum_y;
                        double denominator_x = std::sqrt(n_vals * sum_x2 - sum_x * sum_x);
                        double denominator_y = std::sqrt(n_vals * sum_y2 - sum_y * sum_y);
                        double denominator = denominator_x * denominator_y;

                        double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;

                        // Clamp correlation to [-1, 1] range to handle floating-point precision issues
                        correlation = std::max(-1.0, std::min(1.0, correlation));

                        correlation_matrix[i][j] = correlation;
                        correlation_matrix[j][i] = correlation;
                    }
                }
            }

            promise->set_value(std::move(correlation_matrix));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<Trade>> TaskScheduler::apply_market_microstructure_filters_async(
    const std::vector<Trade>& trades,
    double tick_size,
    std::chrono::milliseconds min_time_diff) {

    auto promise = std::make_shared<std::promise<std::vector<Trade>>>();
    auto future = promise->get_future();

    enqueue_task([trades, tick_size, min_time_diff, promise]() {
        try {
            std::vector<Trade> filtered_trades;

            if (trades.empty()) {
                promise->set_value(std::move(filtered_trades));
                return;
            }

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<Trade>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, tick_size, min_time_diff, &thread_results, t, start, end]() {
                        std::vector<Trade> local_filtered_trades;

                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];

                            // Apply tick size filter - round price to nearest tick
                            double rounded_price = std::round(trade.price / tick_size) * tick_size;

                            // Apply time filter - check if enough time has passed since previous trade
                            bool time_ok = true;
                            if (i > 0) {
                                auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    trade.timestamp - trades[i-1].timestamp);
                                time_ok = time_diff >= min_time_diff;
                            }

                            if (time_ok) {
                                Trade filtered_trade = trade;
                                filtered_trade.price = rounded_price;
                                local_filtered_trades.push_back(filtered_trade);
                            }
                        }

                        thread_results[t] = std::move(local_filtered_trades);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Count total elements to reserve space
                size_t total_filtered = 0;
                for (const auto& result : thread_results) {
                    total_filtered += result.size();
                }

                filtered_trades.reserve(total_filtered);

                // Merge results
                for (const auto& result : thread_results) {
                    filtered_trades.insert(filtered_trades.end(), result.begin(), result.end());
                }
            } else {
                // Sequential processing for smaller datasets
                filtered_trades.reserve(trades.size()); // Reserve to prevent reallocation

                for (size_t i = 0; i < trades.size(); ++i) {
                    const auto& trade = trades[i];

                    // Apply tick size filter - round price to nearest tick
                    double rounded_price = std::round(trade.price / tick_size) * tick_size;

                    // Apply time filter - check if enough time has passed since previous trade
                    bool time_ok = true;
                    if (i > 0) {
                        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                            trade.timestamp - trades[i-1].timestamp);
                        time_ok = time_diff >= min_time_diff;
                    }

                    if (time_ok) {
                        Trade filtered_trade = trade;
                        filtered_trade.price = rounded_price;
                        filtered_trades.push_back(filtered_trade);
                    }
                }
            }

            // Shrink to fit to save memory
            filtered_trades.shrink_to_fit();
            promise->set_value(std::move(filtered_trades));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional advanced multi-threaded methods for market profile analysis
std::future<std::vector<std::pair<double, double>>> TaskScheduler::calculate_market_profile_async(
    const std::vector<Trade>& trades,
    double min_price,
    double max_price,
    int price_levels) {

    auto promise = std::make_shared<std::promise<std::vector<std::pair<double, double>>>>();
    auto future = promise->get_future();

    enqueue_task([trades, min_price, max_price, price_levels, promise]() {
        try {
            std::vector<std::pair<double, double>> market_profile;

            if (trades.empty() || price_levels <= 0) {
                promise->set_value(std::move(market_profile));
                return;
            }

            market_profile.resize(price_levels, {0.0, 0.0}); // {price, volume}

            double price_step = (max_price - min_price) / price_levels;

            // Initialize price levels
            for (int i = 0; i < price_levels; ++i) {
                market_profile[i].first = min_price + (i + 0.5) * price_step;
            }

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                // Temporary storage for each thread
                std::vector<std::vector<double>> thread_volumes(num_threads, std::vector<double>(price_levels, 0.0));

                // Process trades in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, min_price, price_step, price_levels,
                                                   &thread_volumes, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];

                            // Determine which price level this trade belongs to
                            int level_idx = static_cast<int>((trade.price - min_price) / price_step);

                            // Bounds checking
                            if (level_idx >= 0 && level_idx < price_levels) {
                                thread_volumes[t][level_idx] += trade.volume;
                            }
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (int i = 0; i < price_levels; ++i) {
                    for (size_t t = 0; t < num_threads; ++t) {
                        market_profile[i].second += thread_volumes[t][i];
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& trade : trades) {
                    // Determine which price level this trade belongs to
                    int level_idx = static_cast<int>((trade.price - min_price) / price_step);

                    // Bounds checking
                    if (level_idx >= 0 && level_idx < price_levels) {
                        market_profile[level_idx].second += trade.volume;
                    }
                }
            }

            promise->set_value(std::move(market_profile));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Multi-threaded implementation for calculating VWAP over time periods
std::future<std::vector<double>> TaskScheduler::calculate_time_based_vwap_async(
    const std::vector<Trade>& trades,
    int time_resolution_minutes) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, time_resolution_minutes, promise]() {
        try {
            std::vector<double> vwap_values;

            if (trades.empty() || time_resolution_minutes <= 0) {
                promise->set_value(std::move(vwap_values));
                return;
            }

            // Determine time range
            auto min_max_it = std::minmax_element(trades.begin(), trades.end(),
                [](const Trade& a, const Trade& b) {
                    return a.timestamp < b.timestamp;
                });

            auto start_time = min_max_it.first->timestamp;
            auto end_time = min_max_it.second->timestamp;

            // Calculate number of time bins
            auto duration_minutes = std::chrono::duration_cast<std::chrono::minutes>(
                end_time - start_time).count();
            int num_bins = static_cast<int>(std::ceil(static_cast<double>(duration_minutes) /
                                                      time_resolution_minutes)) + 1;

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                // Storage for thread results: {total_value, total_volume} for each time bin
                std::vector<std::vector<std::pair<double, double>>> thread_results(num_threads,
                    std::vector<std::pair<double, double>>(num_bins, {0.0, 0.0}));

                // Process trades in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, start_time, time_resolution_minutes, num_bins,
                                                   &thread_results, t, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                                trade.timestamp - start_time).count();
                            int bin_index = static_cast<int>(time_diff / time_resolution_minutes);

                            if (bin_index >= 0 && bin_index < num_bins) {
                                // Accumulate price * volume and volume for VWAP calculation
                                thread_results[t][bin_index].first += trade.price * trade.volume;
                                thread_results[t][bin_index].second += trade.volume;
                            }
                        }
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results and calculate VWAP
                vwap_values.reserve(num_bins);
                for (int i = 0; i < num_bins; ++i) {
                    double total_value = 0.0;
                    double total_volume = 0.0;

                    for (size_t t = 0; t < num_threads; ++t) {
                        total_value += thread_results[t][i].first;
                        total_volume += thread_results[t][i].second;
                    }

                    double vwap = (total_volume > 0) ? total_value / total_volume : 0.0;
                    vwap_values.push_back(vwap);
                }
            } else {
                // Sequential processing for smaller datasets
                std::vector<std::pair<double, double>> bin_totals(num_bins, {0.0, 0.0});

                for (const auto& trade : trades) {
                    auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                        trade.timestamp - start_time).count();
                    int bin_index = static_cast<int>(time_diff / time_resolution_minutes);

                    if (bin_index >= 0 && bin_index < num_bins) {
                        // Accumulate price * volume and volume for VWAP calculation
                        bin_totals[bin_index].first += trade.price * trade.volume;
                        bin_totals[bin_index].second += trade.volume;
                    }
                }

                // Calculate VWAP for each bin
                vwap_values.reserve(num_bins);
                for (int i = 0; i < num_bins; ++i) {
                    double vwap = (bin_totals[i].second > 0) ? bin_totals[i].first / bin_totals[i].second : 0.0;
                    vwap_values.push_back(vwap);
                }
            }

            promise->set_value(std::move(vwap_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional advanced multi-threaded methods for cumulative volume analysis
std::future<std::vector<double>> TaskScheduler::calculate_cumulative_volume_delta_async(
    const std::vector<Trade>& trades,
    const std::vector<double>& benchmark_prices) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, benchmark_prices, promise]() {
        try {
            std::vector<double> cvd_values;

            if (trades.empty() || benchmark_prices.empty()) {
                cvd_values.resize(std::max(trades.size(), benchmark_prices.size()), 0.0);
                promise->set_value(std::move(cvd_values));
                return;
            }

            cvd_values.resize(trades.size(), 0.0);

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads);
                std::vector<std::mutex> mutexes(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, &benchmark_prices, &thread_results, t, start, end]() {
                        std::vector<double> local_cvd_values(end - start, 0.0);

                        double cumulative_delta = 0.0;
                        for (size_t i = start; i < end; ++i) {
                            size_t benchmark_idx = std::min(i, benchmark_prices.size() - 1);

                            if (trades[i].price > benchmark_prices[benchmark_idx]) {
                                cumulative_delta += trades[i].volume;  // Up volume
                            } else if (trades[i].price < benchmark_prices[benchmark_idx]) {
                                cumulative_delta -= trades[i].volume;  // Down volume
                            }
                            // If equal, no change to cumulative delta

                            local_cvd_values[i - start] = cumulative_delta;
                        }

                        thread_results[t] = std::move(local_cvd_values);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                size_t idx = 0;
                for (const auto& thread_result : thread_results) {
                    for (double val : thread_result) {
                        cvd_values[idx++] = val;
                    }
                }

                // Adjust for cumulative effect across thread boundaries
                double running_total = 0.0;
                for (size_t i = 0; i < thread_results.size(); ++i) {
                    for (size_t j = 0; j < thread_results[i].size(); ++j) {
                        size_t global_idx = (i * chunk_size) + j;
                        if (global_idx < cvd_values.size()) {
                            cvd_values[global_idx] += running_total;
                        }
                    }
                    if ((i + 1) * chunk_size < cvd_values.size() && !thread_results[i].empty()) {
                        running_total = cvd_values[(i + 1) * chunk_size - 1];
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                double cumulative_delta = 0.0;

                for (size_t i = 0; i < trades.size(); ++i) {
                    size_t benchmark_idx = std::min(i, benchmark_prices.size() - 1);

                    if (trades[i].price > benchmark_prices[benchmark_idx]) {
                        cumulative_delta += trades[i].volume;  // Up volume
                    } else if (trades[i].price < benchmark_prices[benchmark_idx]) {
                        cumulative_delta -= trades[i].volume;  // Down volume
                    }
                    // If equal, no change to cumulative delta

                    cvd_values[i] = cumulative_delta;
                }
            }

            promise->set_value(std::move(cvd_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::pair<double, double>>> TaskScheduler::calculate_volume_price_confirmation_indicator_async(
    const std::vector<Trade>& trades,
    int trend_period) {

    auto promise = std::make_shared<std::promise<std::vector<std::pair<double, double>>>>();
    auto future = promise->get_future();

    enqueue_task([trades, trend_period, promise]() {
        try {
            std::vector<std::pair<double, double>> vpci_values; // {price_component, volume_component}

            if (trades.size() < static_cast<size_t>(trend_period) || trend_period <= 0) {
                vpci_values.resize(trades.size(), {0.0, 0.0});
                promise->set_value(std::move(vpci_values));
                return;
            }

            vpci_values.reserve(trades.size());

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<std::pair<double, double>>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = std::max(static_cast<size_t>(trend_period), t * chunk_size);
                    size_t end = (t == num_threads - 1) ? trades.size() : std::max(static_cast<size_t>(trend_period), (t + 1) * chunk_size);

                    processing_threads.emplace_back([&trades, trend_period, &thread_results, t, start, end]() {
                        std::vector<std::pair<double, double>> local_vpci_values;

                        for (size_t i = start; i < end; ++i) {
                            // Calculate price component (based on trend)
                            double price_sum = 0.0;
                            for (int j = 0; j < trend_period && (i >= static_cast<size_t>(j)); ++j) {
                                price_sum += trades[i - j].price;
                            }
                            double avg_price = price_sum / std::min(static_cast<size_t>(trend_period), i + 1);

                            // Calculate volume component (relative volume)
                            double volume_sum = 0.0;
                            for (int j = 0; j < trend_period && (i >= static_cast<size_t>(j)); ++j) {
                                volume_sum += trades[i - j].volume;
                            }
                            double avg_volume = volume_sum / std::min(static_cast<size_t>(trend_period), i + 1);

                            local_vpci_values.emplace_back(avg_price, avg_volume);
                        }

                        thread_results[t] = std::move(local_vpci_values);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                vpci_values.resize(trades.size(), {0.0, 0.0});

                size_t idx = trend_period;
                for (const auto& thread_result : thread_results) {
                    for (const auto& val : thread_result) {
                        if (idx < vpci_values.size()) {
                            vpci_values[idx++] = val;
                        }
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < trades.size(); ++i) {
                    if (i < static_cast<size_t>(trend_period - 1)) {
                        vpci_values.emplace_back(trades[i].price, trades[i].volume);
                        continue;
                    }

                    // Calculate price component (based on trend)
                    double price_sum = 0.0;
                    for (int j = 0; j < trend_period; ++j) {
                        price_sum += trades[i - j].price;
                    }
                    double avg_price = price_sum / trend_period;

                    // Calculate volume component (relative volume)
                    double volume_sum = 0.0;
                    for (int j = 0; j < trend_period; ++j) {
                        volume_sum += trades[i - j].volume;
                    }
                    double avg_volume = volume_sum / trend_period;

                    vpci_values.emplace_back(avg_price, avg_volume);
                }
            }

            promise->set_value(std::move(vpci_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional advanced multi-threaded methods for indicator computations
std::future<std::vector<double>> TaskScheduler::calculate_chandelier_exit_async(
    const std::vector<Candle>& candles,
    int period,
    double multiplier) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, period, multiplier, promise]() {
        try {
            std::vector<double> chandelier_exit_values;

            if (candles.size() < static_cast<size_t>(period) || period <= 0) {
                chandelier_exit_values.resize(candles.size(), 0.0);
                promise->set_value(std::move(chandelier_exit_values));
                return;
            }

            chandelier_exit_values.reserve(candles.size());

            // Use multi-threaded approach for large datasets
            if (candles.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), candles.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = candles.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = std::max(static_cast<size_t>(period), t * chunk_size);
                    size_t end = (t == num_threads - 1) ? candles.size() : std::max(static_cast<size_t>(period), (t + 1) * chunk_size);

                    processing_threads.emplace_back([&candles, period, multiplier, &thread_results, t, start, end]() {
                        std::vector<double> local_chandelier_values;

                        for (size_t i = start; i < end; ++i) {
                            // Calculate ATR for the current period
                            double atr = 0.0;
                            if (i >= static_cast<size_t>(period)) {
                                std::vector<double> true_ranges;
                                true_ranges.reserve(period);

                                for (int j = 0; j < period && (i >= static_cast<size_t>(j)); ++j) {
                                    size_t idx = i - j;
                                    if (idx > 0) {
                                        double high = candles[idx].high;
                                        double low = candles[idx].low;
                                        double prev_close = candles[idx - 1].close;

                                        double tr1 = high - low;
                                        double tr2 = std::abs(high - prev_close);
                                        double tr3 = std::abs(low - prev_close);

                                        double true_range = std::max({tr1, tr2, tr3});
                                        true_ranges.push_back(true_range);
                                    }
                                }

                                if (!true_ranges.empty()) {
                                    double sum = std::accumulate(true_ranges.begin(), true_ranges.end(), 0.0);
                                    atr = sum / true_ranges.size();
                                }
                            }

                            // Calculate Chandelier Exit (long position)
                            double highest_high = candles[i].high;
                            for (int j = 0; j < period && (i >= static_cast<size_t>(j)); ++j) {
                                size_t idx = i - j;
                                if (idx < candles.size()) {
                                    highest_high = std::max(highest_high, candles[idx].high);
                                }
                            }

                            double chandelier_exit = highest_high - (atr * multiplier);
                            local_chandelier_values.push_back(chandelier_exit);
                        }

                        thread_results[t] = std::move(local_chandelier_values);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                chandelier_exit_values.resize(candles.size(), 0.0);

                size_t idx = period;
                for (const auto& thread_result : thread_results) {
                    for (const auto& val : thread_result) {
                        if (idx < chandelier_exit_values.size()) {
                            chandelier_exit_values[idx++] = val;
                        }
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < candles.size(); ++i) {
                    if (i < static_cast<size_t>(period - 1)) {
                        chandelier_exit_values.push_back(candles[i].close); // Use close as placeholder
                        continue;
                    }

                    // Calculate ATR for the current period
                    std::vector<double> true_ranges;
                    true_ranges.reserve(period);

                    for (int j = 0; j < period; ++j) {
                        size_t idx = i - j;
                        if (idx > 0) {
                            double high = candles[idx].high;
                            double low = candles[idx].low;
                            double prev_close = candles[idx - 1].close;

                            double tr1 = high - low;
                            double tr2 = std::abs(high - prev_close);
                            double tr3 = std::abs(low - prev_close);

                            double true_range = std::max({tr1, tr2, tr3});
                            true_ranges.push_back(true_range);
                        }
                    }

                    double atr = 0.0;
                    if (!true_ranges.empty()) {
                        double sum = std::accumulate(true_ranges.begin(), true_ranges.end(), 0.0);
                        atr = sum / true_ranges.size();
                    }

                    // Calculate Chandelier Exit (long position)
                    double highest_high = candles[i].high;
                    for (int j = 0; j < period; ++j) {
                        size_t idx = i - j;
                        if (idx < candles.size()) {
                            highest_high = std::max(highest_high, candles[idx].high);
                        }
                    }

                    double chandelier_exit = highest_high - (atr * multiplier);
                    chandelier_exit_values.push_back(chandelier_exit);
                }
            }

            promise->set_value(std::move(chandelier_exit_values));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<double>> TaskScheduler::calculate_keltner_channels_async(
    const std::vector<Candle>& candles,
    int period,
    double multiplier) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([candles, period, multiplier, promise]() {
        try {
            std::vector<double> keltner_channels; // Will store middle band values

            if (candles.size() < static_cast<size_t>(period) || period <= 0) {
                keltner_channels.resize(candles.size(), 0.0);
                promise->set_value(std::move(keltner_channels));
                return;
            }

            keltner_channels.reserve(candles.size());

            // Use multi-threaded approach for large datasets
            if (candles.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), candles.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = candles.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = std::max(static_cast<size_t>(period), t * chunk_size);
                    size_t end = (t == num_threads - 1) ? candles.size() : std::max(static_cast<size_t>(period), (t + 1) * chunk_size);

                    processing_threads.emplace_back([&candles, period, multiplier, &thread_results, t, start, end]() {
                        std::vector<double> local_keltner_values;

                        for (size_t i = start; i < end; ++i) {
                            // Calculate EMA of typical price
                            std::vector<double> typical_prices;
                            typical_prices.reserve(period);

                            for (int j = 0; j < period && (i >= static_cast<size_t>(j)); ++j) {
                                size_t idx = i - j;
                                if (idx < candles.size()) {
                                    double typical_price = (candles[idx].high + candles[idx].low + candles[idx].close) / 3.0;
                                    typical_prices.push_back(typical_price);
                                }
                            }

                            // Calculate EMA of typical prices
                            double ema_tp = 0.0;
                            if (!typical_prices.empty()) {
                                double sum = std::accumulate(typical_prices.begin(), typical_prices.end(), 0.0);
                                ema_tp = sum / typical_prices.size();
                            }

                            local_keltner_values.push_back(ema_tp);
                        }

                        thread_results[t] = std::move(local_keltner_values);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                keltner_channels.resize(candles.size(), 0.0);

                size_t idx = period;
                for (const auto& thread_result : thread_results) {
                    for (const auto& val : thread_result) {
                        if (idx < keltner_channels.size()) {
                            keltner_channels[idx++] = val;
                        }
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < candles.size(); ++i) {
                    if (i < static_cast<size_t>(period - 1)) {
                        double typical_price = (candles[i].high + candles[i].low + candles[i].close) / 3.0;
                        keltner_channels.push_back(typical_price);
                        continue;
                    }

                    // Calculate EMA of typical price
                    std::vector<double> typical_prices;
                    typical_prices.reserve(period);

                    for (int j = 0; j < period; ++j) {
                        size_t idx = i - j;
                        if (idx < candles.size()) {
                            double typical_price = (candles[idx].high + candles[idx].low + candles[idx].close) / 3.0;
                            typical_prices.push_back(typical_price);
                        }
                    }

                    // Calculate EMA of typical prices
                    double ema_tp = 0.0;
                    if (!typical_prices.empty()) {
                        double sum = std::accumulate(typical_prices.begin(), typical_prices.end(), 0.0);
                        ema_tp = sum / typical_prices.size();
                    }

                    keltner_channels.push_back(ema_tp);
                }
            }

            promise->set_value(std::move(keltner_channels));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

// Additional advanced multi-threaded methods for data processing
std::future<std::vector<std::vector<Trade>>> TaskScheduler::detect_and_classify_market_regimes_async(
    const std::vector<Trade>& trades,
    double volatility_threshold,
    double volume_threshold) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<Trade>>>>();
    auto future = promise->get_future();

    enqueue_task([trades, volatility_threshold, volume_threshold, promise]() {
        try {
            std::vector<std::vector<Trade>> regime_classified_trades;

            if (trades.size() < 2) {
                regime_classified_trades.push_back(trades);
                promise->set_value(std::move(regime_classified_trades));
                return;
            }

            // Use multi-threaded approach for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), trades.size());
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<std::vector<Trade>>> thread_results(num_threads);

                // Process data in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&trades, volatility_threshold, volume_threshold, &thread_results, t, start, end]() {
                        std::vector<std::vector<Trade>> local_regime_groups;
                        std::vector<Trade> current_group;

                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];

                            // Simple regime detection based on volatility and volume
                            bool is_high_volatility = false;
                            bool is_high_volume = trade.volume > volume_threshold;

                            // Check volatility by comparing to recent average
                            if (i > 0) {
                                double price_change = std::abs(trade.price - trades[i-1].price);
                                // For simplicity, we'll use a fixed threshold comparison
                                is_high_volatility = price_change > volatility_threshold;
                            }

                            // Classify trade based on regime
                            if (is_high_volatility || is_high_volume) {
                                // High volatility/volume regime
                                if (!current_group.empty()) {
                                    local_regime_groups.push_back(current_group);
                                    current_group.clear();
                                }
                                current_group.push_back(trade);
                            } else {
                                // Low volatility/volume regime
                                current_group.push_back(trade);
                            }
                        }

                        if (!current_group.empty()) {
                            local_regime_groups.push_back(current_group);
                        }

                        thread_results[t] = std::move(local_regime_groups);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results
                for (const auto& thread_result : thread_results) {
                    for (const auto& group : thread_result) {
                        regime_classified_trades.push_back(group);
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                std::vector<Trade> current_group;

                for (size_t i = 0; i < trades.size(); ++i) {
                    const auto& trade = trades[i];

                    // Simple regime detection based on volatility and volume
                    bool is_high_volatility = false;
                    bool is_high_volume = trade.volume > volume_threshold;

                    // Check volatility by comparing to recent average
                    if (i > 0) {
                        double price_change = std::abs(trade.price - trades[i-1].price);
                        is_high_volatility = price_change > volatility_threshold;
                    }

                    // Classify trade based on regime
                    if (is_high_volatility || is_high_volume) {
                        // High volatility/volume regime
                        if (!current_group.empty()) {
                            regime_classified_trades.push_back(current_group);
                            current_group.clear();
                        }
                        current_group.push_back(trade);
                    } else {
                        // Low volatility/volume regime
                        current_group.push_back(trade);
                    }
                }

                if (!current_group.empty()) {
                    regime_classified_trades.push_back(current_group);
                }
            }

            promise->set_value(std::move(regime_classified_trades));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

std::future<std::vector<std::vector<std::vector<double>>>> TaskScheduler::calculate_rolling_correlation_matrix_async(
    const std::vector<std::vector<double>>& data_series,
    int window_size) {

    auto promise = std::make_shared<std::promise<std::vector<std::vector<std::vector<double>>>>>();
    auto future = promise->get_future();

    enqueue_task([data_series, window_size, promise]() {
        try {
            std::vector<std::vector<std::vector<double>>> rolling_correlations;

            if (data_series.empty() || window_size <= 0) {
                promise->set_value(std::move(rolling_correlations));
                return;
            }

            size_t n_series = data_series.size();
            if (n_series == 0) {
                promise->set_value(std::move(rolling_correlations));
                return;
            }

            // Determine the minimum length across all series
            size_t min_length = std::numeric_limits<size_t>::max();
            for (const auto& series : data_series) {
                min_length = std::min(min_length, series.size());
            }

            if (min_length < static_cast<size_t>(window_size)) {
                // Not enough data for the specified window size
                promise->set_value(std::move(rolling_correlations));
                return;
            }

            // Calculate number of windows
            size_t n_windows = min_length - window_size + 1;
            rolling_correlations.reserve(n_windows);

            // Use multi-threaded approach for large datasets
            if (n_windows > 100) {
                size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), n_windows);
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<std::vector<std::vector<double>>>> thread_results(num_threads);

                // Process windows in chunks using separate threads
                std::vector<std::thread> processing_threads;
                size_t chunk_size = n_windows / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? n_windows : (t + 1) * chunk_size;

                    processing_threads.emplace_back([&data_series, window_size, n_series, &thread_results, t, start, end]() {
                        std::vector<std::vector<std::vector<double>>> local_correlations;

                        for (size_t win_idx = start; win_idx < end; ++win_idx) {
                            // Calculate correlation matrix for this window
                            std::vector<std::vector<double>> corr_matrix(n_series, std::vector<double>(n_series, 0.0));

                            // Fill diagonal with 1.0
                            for (size_t i = 0; i < n_series; ++i) {
                                corr_matrix[i][i] = 1.0;
                            }

                            // Calculate correlations between series
                            for (size_t i = 0; i < n_series; ++i) {
                                for (size_t j = i + 1; j < n_series; ++j) {
                                    // Extract windowed data for both series
                                    std::vector<double> series_i_win(window_size);
                                    std::vector<double> series_j_win(window_size);

                                    for (int k = 0; k < window_size; ++k) {
                                        series_i_win[k] = data_series[i][win_idx + k];
                                        series_j_win[k] = data_series[j][win_idx + k];
                                    }

                                    // Calculate correlation
                                    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                                    double sum_x2 = 0.0, sum_y2 = 0.0;

                                    for (int k = 0; k < window_size; ++k) {
                                        double x = series_i_win[k];
                                        double y = series_j_win[k];

                                        sum_x += x;
                                        sum_y += y;
                                        sum_xy += x * y;
                                        sum_x2 += x * x;
                                        sum_y2 += y * y;
                                    }

                                    double n = window_size;
                                    double numerator = n * sum_xy - sum_x * sum_y;
                                    double denominator_x = std::sqrt(n * sum_x2 - sum_x * sum_x);
                                    double denominator_y = std::sqrt(n * sum_y2 - sum_y * sum_y);
                                    double denominator = denominator_x * denominator_y;

                                    double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;

                                    // Clamp correlation to [-1, 1] range
                                    correlation = std::max(-1.0, std::min(1.0, correlation));

                                    corr_matrix[i][j] = correlation;
                                    corr_matrix[j][i] = correlation;
                                }
                            }

                            local_correlations.push_back(std::move(corr_matrix));
                        }

                        thread_results[t] = std::move(local_correlations);
                    });
                }

                // Wait for all threads to complete
                for (auto& thread : processing_threads) {
                    thread.join();
                }

                // Merge results in order
                for (size_t t = 0; t < num_threads; ++t) {
                    for (const auto& corr_matrix : thread_results[t]) {
                        rolling_correlations.push_back(corr_matrix);
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t win_idx = 0; win_idx < n_windows; ++win_idx) {
                    // Calculate correlation matrix for this window
                    std::vector<std::vector<double>> corr_matrix(n_series, std::vector<double>(n_series, 0.0));

                    // Fill diagonal with 1.0
                    for (size_t i = 0; i < n_series; ++i) {
                        corr_matrix[i][i] = 1.0;
                    }

                    // Calculate correlations between series
                    for (size_t i = 0; i < n_series; ++i) {
                        for (size_t j = i + 1; j < n_series; ++j) {
                            // Extract windowed data for both series
                            std::vector<double> series_i_win(window_size);
                            std::vector<double> series_j_win(window_size);

                            for (int k = 0; k < window_size; ++k) {
                                series_i_win[k] = data_series[i][win_idx + k];
                                series_j_win[k] = data_series[j][win_idx + k];
                            }

                            // Calculate correlation
                            double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
                            double sum_x2 = 0.0, sum_y2 = 0.0;

                            for (int k = 0; k < window_size; ++k) {
                                double x = series_i_win[k];
                                double y = series_j_win[k];

                                sum_x += x;
                                sum_y += y;
                                sum_xy += x * y;
                                sum_x2 += x * x;
                                sum_y2 += y * y;
                            }

                            double n = window_size;
                            double numerator = n * sum_xy - sum_x * sum_y;
                            double denominator_x = std::sqrt(n * sum_x2 - sum_x * sum_x);
                            double denominator_y = std::sqrt(n * sum_y2 - sum_y * sum_y);
                            double denominator = denominator_x * denominator_y;

                            double correlation = (denominator != 0.0) ? numerator / denominator : 0.0;

                            // Clamp correlation to [-1, 1] range
                            correlation = std::max(-1.0, std::min(1.0, correlation));

                            corr_matrix[i][j] = correlation;
                            corr_matrix[j][i] = correlation;
                        }
                    }

                    rolling_correlations.push_back(std::move(corr_matrix));
                }
            }

            promise->set_value(std::move(rolling_correlations));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

} // namespace btq