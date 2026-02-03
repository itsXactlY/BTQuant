#include "task_scheduler.hpp"
#include <iostream>
#include <future>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>

namespace btq {

TaskScheduler::TaskScheduler(size_t num_threads)
    : num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency()),
      stop_(false) {

    for (size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this]() { worker_loop(); });
    }
}

TaskScheduler::~TaskScheduler() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
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
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            
            if (stop_ && tasks_.empty()) {
                return;
            }
            
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }
        
        if (task) {
            task();
        }
    }
}

void TaskScheduler::enqueue_task(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("TaskScheduler is stopped");
        }
        tasks_.emplace(std::move(task));
    }
    condition_.notify_one();
}

// Volume Calculations
std::future<std::vector<double>> TaskScheduler::calculate_volume_profile_async(
    const std::vector<Trade>& trades,
    double min_price,
    double max_price,
    int resolution) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([trades, min_price, max_price, resolution, promise]() {
        try {
            std::vector<double> volume_profile(resolution, 0.0);
            double price_range = max_price - min_price;
            double bin_size = price_range / resolution;

            // Use parallel algorithm for large datasets
            if (trades.size() > 10000) {
                // For parallel processing, we need to use temporary storage per thread
                // and then merge results
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads, std::vector<double>(resolution, 0.0));

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &trades, min_price, bin_size, resolution, &thread_results, t]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            int bin_index = static_cast<int>((trade.price - min_price) / bin_size);
                            if (bin_index >= 0 && bin_index < resolution) {
                                thread_results[t][bin_index] += trade.volume;
                            }
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }

                // Merge results
                for (const auto& thread_result : thread_results) {
                    for (int i = 0; i < resolution; ++i) {
                        volume_profile[i] += thread_result[i];
                    }
                }
            } else {
                // Sequential processing for smaller datasets
                for (const auto& trade : trades) {
                    int bin_index = static_cast<int>((trade.price - min_price) / bin_size);
                    if (bin_index >= 0 && bin_index < resolution) {
                        volume_profile[bin_index] += trade.volume;
                    }
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

    enqueue_task([trades, promise]() {
        try {
            if (trades.empty()) {
                promise->set_value(0.0);
                return;
            }

            // Use parallel algorithm for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                std::vector<std::pair<double, double>> thread_results(num_threads, {0.0, 0.0}); // {total_value, total_volume}

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &trades, &thread_results, t]() {
                        double local_total_value = 0.0;
                        double local_total_volume = 0.0;

                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            local_total_value += trade.price * trade.volume;
                            local_total_volume += trade.volume;
                        }

                        thread_results[t] = {local_total_value, local_total_volume};
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }

                // Merge results
                double total_value = 0.0;
                double total_volume = 0.0;
                for (const auto& [val, vol] : thread_results) {
                    total_value += val;
                    total_volume += vol;
                }

                double vwap = (total_volume > 0) ? total_value / total_volume : 0.0;
                promise->set_value(vwap);
            } else {
                // Sequential processing for smaller datasets
                double total_value = 0.0;
                double total_volume = 0.0;

                for (const auto& trade : trades) {
                    total_value += trade.price * trade.volume;
                    total_volume += trade.volume;
                }

                double vwap = (total_volume > 0) ? total_value / total_volume : 0.0;
                promise->set_value(vwap);
            }
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

            // Use parallel algorithm for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                std::vector<std::vector<double>> thread_results(num_threads, std::vector<double>(num_bins, 0.0));

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &trades, start_time, time_resolution_minutes, num_bins, &thread_results, t]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                                trade.timestamp - start_time).count();
                            int bin_index = static_cast<int>(time_diff / time_resolution_minutes);

                            if (bin_index >= 0 && bin_index < num_bins) {
                                thread_results[t][bin_index] += trade.volume;
                            }
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
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
                    auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                        trade.timestamp - start_time).count();
                    int bin_index = static_cast<int>(time_diff / time_resolution_minutes);

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

// Indicator Computations
std::future<std::vector<double>> TaskScheduler::calculate_sma_async(
    const std::vector<double>& prices,
    int period) {

    auto promise = std::make_shared<std::promise<std::vector<double>>>();
    auto future = promise->get_future();

    enqueue_task([prices, period, promise]() {
        try {
            std::vector<double> sma_values;
            if (prices.size() < period) {
                sma_values.resize(prices.size());
                promise->set_value(std::move(sma_values));
                return;
            }

            sma_values.reserve(prices.size() - period + 1);

            // Calculate initial sum for the first period
            double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
            sma_values.push_back(sum / period);

            // Use parallel algorithm for large datasets
            if (prices.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // For SMA, we can parallelize the sliding window calculation
                size_t start_idx = period;
                size_t total_elements = prices.size() - start_idx;

                if (total_elements >= num_threads) {
                    sma_values.resize(prices.size() - period + 1);

                    // Use std::async for parallel processing to avoid blocking the thread pool
                    std::vector<std::future<void>> futures;
                    size_t chunk_size = total_elements / num_threads;

                    for (size_t t = 0; t < num_threads; ++t) {
                        size_t chunk_start = start_idx + t * chunk_size;
                        size_t chunk_end = (t == num_threads - 1) ? prices.size() : start_idx + (t + 1) * chunk_size;

                        futures.push_back(std::async(std::launch::async, [chunk_start, chunk_end, &prices, period, &sma_values]() {
                            double local_sum = 0.0;

                            // Calculate the initial sum for this chunk's starting point
                            if (chunk_start > period) {
                                // Need to calculate the sum at chunk_start
                                local_sum = std::accumulate(prices.begin() + (chunk_start - period), prices.begin() + chunk_start, 0.0);
                            } else {
                                // Use the initial sum calculated earlier
                                local_sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
                            }

                            for (size_t i = chunk_start; i < chunk_end; ++i) {
                                // Calculate SMA for position i-period+1
                                size_t sma_idx = i - period + 1;

                                // Update the sum using the sliding window technique
                                local_sum += prices[i] - prices[i - period];
                                sma_values[sma_idx] = local_sum / period;
                            }
                        }));
                    }

                    // Wait for all sub-tasks to complete
                    for (auto& f : futures) {
                        f.wait();
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

            // Use parallel algorithm for large datasets
            if (prices.size() > 10000) {
                changes.resize(prices.size() - 1);

                // Parallel computation of price changes
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = (prices.size() - 1) / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? prices.size() - 1 : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &prices, &changes]() {
                        for (size_t i = start; i < end; ++i) {
                            changes[i] = prices[i + 1] - prices[i];
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 1; i < prices.size(); ++i) {
                    changes.push_back(prices[i] - prices[i-1]);
                }
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

            if (prices.size() < period) {
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

            // Use parallel algorithm for large datasets when calculating standard deviation
            if (sma_values.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = sma_values.size() / num_threads;

                upper_band.resize(sma_values.size());
                lower_band.resize(sma_values.size());

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t chunk_start = t * chunk_size;
                    size_t chunk_end = (t == num_threads - 1) ? sma_values.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [chunk_start, chunk_end, &prices, &sma_values, period, num_std_dev, &upper_band, &lower_band]() {
                        for (size_t i = chunk_start; i < chunk_end; ++i) {
                            size_t start_idx = i + period - sma_values.size(); // Adjust index to align with original prices

                            // Calculate variance for the corresponding window
                            double sum_sq_diff = 0.0;
                            for (int j = 0; j < period; ++j) {
                                double diff = prices[start_idx + j] - sma_values[i];
                                sum_sq_diff += diff * diff;
                            }
                            double variance = sum_sq_diff / period;
                            double std_dev = std::sqrt(variance);

                            upper_band[i] = sma_values[i] + num_std_dev * std_dev;
                            lower_band[i] = sma_values[i] - num_std_dev * std_dev;
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }
            } else {
                // Sequential processing for smaller datasets
                for (size_t i = 0; i < sma_values.size(); ++i) {
                    size_t start_idx = i + period - sma_values.size(); // Adjust index to align with original prices

                    // Calculate variance for the corresponding window
                    double sum_sq_diff = 0.0;
                    for (int j = 0; j < period; ++j) {
                        double diff = prices[start_idx + j] - sma_values[i];
                        sum_sq_diff += diff * diff;
                    }
                    double variance = sum_sq_diff / period;
                    double std_dev = std::sqrt(variance);

                    upper_band.push_back(sma_values[i] + num_std_dev * std_dev);
                    lower_band.push_back(sma_values[i] - num_std_dev * std_dev);
                }
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

    enqueue_task([trades, timeframe, promise]() {
        try {
            std::vector<Candle> candles;
            if (trades.empty()) {
                promise->set_value(std::move(candles));
                return;
            }

            // Use parallel algorithm for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Partition trades among threads
                std::vector<std::map<std::chrono::system_clock::time_point, Candle>> thread_maps(num_threads);

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &trades, timeframe, &thread_maps, t]() {
                        for (size_t i = start; i < end; ++i) {
                            const auto& trade = trades[i];
                            // Calculate the start time of the candle period
                            auto time_since_epoch = trade.timestamp.time_since_epoch();
                            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since_epoch);
                            auto period_start = std::chrono::system_clock::time_point(seconds -
                                std::chrono::seconds(seconds.count() % timeframe.count()));

                            if (thread_maps[t].find(period_start) == thread_maps[t].end()) {
                                // Initialize new candle
                                thread_maps[t][period_start] = {
                                    period_start,
                                    trade.price,  // Open
                                    trade.price,  // High
                                    trade.price,  // Low
                                    trade.price,  // Close
                                    trade.volume  // Volume
                                };
                            } else {
                                // Update existing candle
                                auto& candle = thread_maps[t][period_start];
                                candle.high = std::max(candle.high, trade.price);
                                candle.low = std::min(candle.low, trade.price);
                                candle.close = trade.price;  // Last price becomes close
                                candle.volume += trade.volume;
                            }
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }

                // Merge thread maps into a single map
                std::map<std::chrono::system_clock::time_point, Candle> candle_map;

                for (const auto& thread_map : thread_maps) {
                    for (const auto& pair : thread_map) {
                        auto it = candle_map.find(pair.first);
                        if (it == candle_map.end()) {
                            candle_map[pair.first] = pair.second;
                        } else {
                            // Merge the candles
                            auto& existing_candle = it->second;
                            existing_candle.high = std::max(existing_candle.high, pair.second.high);
                            existing_candle.low = std::min(existing_candle.low, pair.second.low);
                            existing_candle.close = pair.second.close;  // Last price becomes close
                            existing_candle.volume += pair.second.volume;
                            // Note: Open time is already set correctly
                        }
                    }
                }

                // Convert map to vector
                candles.reserve(candle_map.size());
                for (const auto& pair : candle_map) {
                    candles.push_back(pair.second);
                }
            } else {
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

            // Use parallel algorithm for large datasets
            if (trades.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Partition the work among threads
                std::vector<std::vector<Trade>> thread_results(num_threads);

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = trades.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? trades.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &trades, &filter_func, &thread_results, t]() {
                        for (size_t i = start; i < end; ++i) {
                            if (filter_func(trades[i])) {
                                thread_results[t].push_back(trades[i]);
                            }
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
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

                for (const auto& trade : trades) {
                    if (filter_func(trade)) {
                        filtered_trades.push_back(trade);
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

            // Use parallel algorithm for large datasets
            if (values.size() > 10000) {
                size_t num_threads = std::thread::hardware_concurrency();
                if (num_threads < 2) num_threads = 2;

                // Create temporary histograms for each thread
                std::vector<std::vector<double>> thread_histograms(num_threads, std::vector<double>(num_bins, 0.0));

                // Use std::async for parallel processing to avoid blocking the thread pool
                std::vector<std::future<void>> futures;
                size_t chunk_size = values.size() / num_threads;

                for (size_t t = 0; t < num_threads; ++t) {
                    size_t start = t * chunk_size;
                    size_t end = (t == num_threads - 1) ? values.size() : (t + 1) * chunk_size;

                    futures.push_back(std::async(std::launch::async, [start, end, &values, min_val, bin_width, num_bins, &thread_histograms, t]() {
                        for (size_t i = start; i < end; ++i) {
                            double val = values[i];
                            int bin_index = static_cast<int>((val - min_val) / bin_width);
                            // Handle edge case where value equals max
                            if (bin_index >= num_bins) {
                                bin_index = num_bins - 1;
                            }

                            thread_histograms[t][bin_index] += 1.0; // Increment count
                        }
                    }));
                }

                // Wait for all sub-tasks to complete
                for (auto& f : futures) {
                    f.wait();
                }

                // Merge thread histograms
                for (int i = 0; i < num_bins; ++i) {
                    for (size_t t = 0; t < num_threads; ++t) {
                        histogram[i].second += thread_histograms[t][i];
                    }
                }
            } else {
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

} // namespace btq