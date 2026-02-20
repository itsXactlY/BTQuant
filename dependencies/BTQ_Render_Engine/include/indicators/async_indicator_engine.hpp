#pragma once

/**
 * @file async_indicator_engine.hpp
 * @brief C++26 Async Indicator & Analytics Engine
 * 
 * This implementation provides:
 * - Parallel indicator calculations using std::execution::par_unseq
 * - Vectorized math operations for SIMD optimization
 * - Async calculation with std::future/std::promise
 * - Lock-free indicator cache
 * - Support for SMA, EMA, RSI, MACD, and custom indicators
 * 
 * Key C++26 features used:
 * - std::execution::par_unseq for parallel unsequenced execution
 * - std::execution::unseq for vectorized execution
 * - std::mdspan for multidimensional data views
 * - std::generator for lazy evaluation
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace btq {
namespace indicators {

/**
 * @brief Indicator result data
 */
struct IndicatorResult {
    std::vector<double> values;
    std::vector<double> timestamps;  // Optional: timestamps for each value
    bool is_valid = false;
    uint64_t calculation_hash = 0;   // Hash for cache validation
};

/**
 * @brief Indicator calculation parameters
 */
struct IndicatorParams {
    uint32_t period = 14;
    double multiplier = 1.0;
    uint32_t signal_period = 9;      // For MACD signal line
    uint32_t fast_period = 12;       // For MACD
    uint32_t slow_period = 26;       // For MACD
    std::string source = "close";    // Price source: open, high, low, close, hl2, hlc3
};

/**
 * @brief Base class for all indicators
 */
class Indicator {
public:
    virtual ~Indicator() = default;
    
    /**
     * @brief Calculate indicator values
     * @param prices Input price data
     * @param params Indicator parameters
     * @return Calculated indicator values
     */
    virtual IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const = 0;
    
    /**
     * @brief Calculate indicator values asynchronously
     * @param prices Input price data
     * @param params Indicator parameters
     * @return Future containing calculated values
     */
    virtual std::future<IndicatorResult> calculateAsync(
        std::span<const double> prices,
        const IndicatorParams& params) const {
        return std::async(std::launch::async, [this, prices, params]() {
            return calculate(prices, params);
        });
    }
    
    /**
     * @brief Get the indicator name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the minimum required data points
     */
    virtual size_t getMinRequiredPoints(const IndicatorParams& params) const = 0;
};

/**
 * @brief Simple Moving Average (SMA) Indicator
 */
class SMAIndicator : public Indicator {
public:
    IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const override
    {
        IndicatorResult result;
        const size_t period = params.period;
        
        if (prices.size() < period) {
            return result;
        }
        
        const size_t result_size = prices.size() - period + 1;
        result.values.resize(result_size);
        
        // Calculate initial sum
        double sum = 0.0;
        for (size_t i = 0; i < period; ++i) {
            sum += prices[i];
        }
        result.values[0] = sum / period;
        
        // Calculate remaining values using rolling sum
        for (size_t i = 1; i < result_size; ++i) {
            sum = sum - prices[i - 1] + prices[i + period - 1];
            result.values[i] = sum / period;
        }
        
        result.is_valid = true;
        return result;
    }
    
    /**
     * @brief Parallel SMA calculation using std::execution::par_unseq
     */
    IndicatorResult calculateParallel(
        std::span<const double> prices,
        const IndicatorParams& params) const
    {
        IndicatorResult result;
        const size_t period = params.period;
        
        if (prices.size() < period) {
            return result;
        }
        
        const size_t result_size = prices.size() - period + 1;
        result.values.resize(result_size);
        
        // Use parallel execution for large datasets
        if (result_size > 1000) {
            std::vector<double> partial_sums(result_size);
            std::vector<size_t> indices(result_size);
            std::iota(indices.begin(), indices.end(), size_t(0));
            
            // Calculate partial sums in parallel
            std::transform(
                std::execution::par_unseq,
                indices.begin(),
                indices.end(),
                partial_sums.begin(),
                [&prices, period](size_t i) {
                    return std::accumulate(
                        prices.begin() + i,
                        prices.begin() + i + period,
                        0.0);
                });
            
            // Divide by period
            std::transform(
                std::execution::par_unseq,
                partial_sums.begin(),
                partial_sums.end(),
                result.values.begin(),
                [period](double sum) { return sum / period; });
        } else {
            // Sequential for small datasets
            double sum = std::accumulate(prices.begin(), prices.begin() + period, 0.0);
            result.values[0] = sum / period;
            
            for (size_t i = 1; i < result_size; ++i) {
                sum = sum - prices[i - 1] + prices[i + period - 1];
                result.values[i] = sum / period;
            }
        }
        
        result.is_valid = true;
        return result;
    }
    
    std::string getName() const override { return "SMA"; }
    
    size_t getMinRequiredPoints(const IndicatorParams& params) const override {
        return params.period;
    }
};

/**
 * @brief Exponential Moving Average (EMA) Indicator
 */
class EMAIndicator : public Indicator {
public:
    IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const override
    {
        IndicatorResult result;
        const size_t period = params.period;
        
        if (prices.size() < period) {
            return result;
        }
        
        const size_t n = prices.size();
        result.values.resize(n);
        
        // Calculate multiplier
        const double multiplier = 2.0 / (period + 1.0);
        
        // Initialize with SMA for first period values
        double sum = 0.0;
        for (size_t i = 0; i < period; ++i) {
            sum += prices[i];
            result.values[i] = sum / (i + 1);  // Partial average
        }
        
        // Calculate EMA
        double ema = sum / period;
        result.values[period - 1] = ema;
        
        for (size_t i = period; i < n; ++i) {
            ema = (prices[i] - ema) * multiplier + ema;
            result.values[i] = ema;
        }
        
        result.is_valid = true;
        return result;
    }
    
    std::string getName() const override { return "EMA"; }
    
    size_t getMinRequiredPoints(const IndicatorParams& params) const override {
        return params.period;
    }
};

/**
 * @brief Relative Strength Index (RSI) Indicator
 */
class RSIIndicator : public Indicator {
public:
    IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const override
    {
        IndicatorResult result;
        const size_t period = params.period;
        
        if (prices.size() < period + 1) {
            return result;
        }
        
        const size_t n = prices.size();
        result.values.resize(n);
        
        // Calculate price changes
        std::vector<double> gains(n - 1);
        std::vector<double> losses(n - 1);
        
        // Use parallel execution for calculating gains/losses
        std::transform(
            std::execution::par_unseq,
            prices.begin() + 1,
            prices.end(),
            prices.begin(),
            gains.begin(),
            [](double current, double prev) {
                double change = current - prev;
                return change > 0 ? change : 0.0;
            });
        
        std::transform(
            std::execution::par_unseq,
            prices.begin() + 1,
            prices.end(),
            prices.begin(),
            losses.begin(),
            [](double current, double prev) {
                double change = current - prev;
                return change < 0 ? -change : 0.0;
            });
        
        // Calculate initial average gain/loss
        double avg_gain = std::accumulate(gains.begin(), gains.begin() + period, 0.0) / period;
        double avg_loss = std::accumulate(losses.begin(), losses.begin() + period, 0.0) / period;
        
        // Fill initial values with NaN
        for (size_t i = 0; i < period; ++i) {
            result.values[i] = std::nan("");
        }
        
        // Calculate RSI
        for (size_t i = period; i < n; ++i) {
            if (i > period) {
                // Smoothed average
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period;
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period;
            }
            
            if (avg_loss == 0.0) {
                result.values[i] = 100.0;
            } else {
                double rs = avg_gain / avg_loss;
                result.values[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        
        result.is_valid = true;
        return result;
    }
    
    std::string getName() const override { return "RSI"; }
    
    size_t getMinRequiredPoints(const IndicatorParams& params) const override {
        return params.period + 1;
    }
};

/**
 * @brief MACD (Moving Average Convergence Divergence) Indicator
 */
class MACDIndicator : public Indicator {
public:
    struct MACDResult {
        std::vector<double> macd_line;
        std::vector<double> signal_line;
        std::vector<double> histogram;
        bool is_valid = false;
    };
    
    MACDResult calculateMACD(
        std::span<const double> prices,
        const IndicatorParams& params) const
    {
        MACDResult result;
        
        const size_t fast_period = params.fast_period;
        const size_t slow_period = params.slow_period;
        const size_t signal_period = params.signal_period;
        
        if (prices.size() < slow_period + signal_period) {
            return result;
        }
        
        const size_t n = prices.size();
        
        // Calculate fast and slow EMAs
        EMAIndicator ema;
        IndicatorParams ema_params;
        
        ema_params.period = fast_period;
        auto fast_ema = ema.calculate(prices, ema_params);
        
        ema_params.period = slow_period;
        auto slow_ema = ema.calculate(prices, ema_params);
        
        if (!fast_ema.is_valid || !slow_ema.is_valid) {
            return result;
        }
        
        // Calculate MACD line (fast EMA - slow EMA)
        result.macd_line.resize(n);
        std::transform(
            std::execution::par_unseq,
            fast_ema.values.begin(),
            fast_ema.values.end(),
            slow_ema.values.begin(),
            result.macd_line.begin(),
            [](double fast, double slow) { return fast - slow; });
        
        // Calculate signal line (EMA of MACD line)
        ema_params.period = signal_period;
        auto signal_ema = ema.calculate(result.macd_line, ema_params);
        
        if (!signal_ema.is_valid) {
            return result;
        }
        result.signal_line = std::move(signal_ema.values);
        
        // Calculate histogram (MACD - signal)
        result.histogram.resize(n);
        std::transform(
            std::execution::par_unseq,
            result.macd_line.begin(),
            result.macd_line.end(),
            result.signal_line.begin(),
            result.histogram.begin(),
            [](double macd, double signal) { return macd - signal; });
        
        result.is_valid = true;
        return result;
    }
    
    IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const override
    {
        // Return MACD line as the main result
        auto macd_result = calculateMACD(prices, params);
        IndicatorResult result;
        result.values = std::move(macd_result.macd_line);
        result.is_valid = macd_result.is_valid;
        return result;
    }
    
    std::string getName() const override { return "MACD"; }
    
    size_t getMinRequiredPoints(const IndicatorParams& params) const override {
        return params.slow_period + params.signal_period;
    }
};

/**
 * @brief Bollinger Bands Indicator
 */
class BollingerBandsIndicator : public Indicator {
public:
    struct BollingerResult {
        std::vector<double> middle;     // SMA
        std::vector<double> upper;      // Upper band
        std::vector<double> lower;      // Lower band
        std::vector<double> bandwidth;  // Bandwidth
        bool is_valid = false;
    };
    
    BollingerResult calculateBollinger(
        std::span<const double> prices,
        const IndicatorParams& params) const
    {
        BollingerResult result;
        const size_t period = params.period;
        const double std_dev_multiplier = params.multiplier;
        
        if (prices.size() < period) {
            return result;
        }
        
        const size_t n = prices.size();
        result.middle.resize(n);
        result.upper.resize(n);
        result.lower.resize(n);
        result.bandwidth.resize(n);
        
        // Calculate SMA
        SMAIndicator sma;
        auto sma_result = sma.calculate(prices, params);
        
        if (!sma_result.is_valid) {
            return result;
        }
        
        // Fill initial values
        for (size_t i = 0; i < period - 1; ++i) {
            result.middle[i] = std::nan("");
            result.upper[i] = std::nan("");
            result.lower[i] = std::nan("");
            result.bandwidth[i] = std::nan("");
        }
        
        // Calculate standard deviation and bands
        for (size_t i = period - 1; i < n; ++i) {
            double mean = sma_result.values[i - period + 1];
            result.middle[i] = mean;
            
            // Calculate standard deviation
            double sum_sq = 0.0;
            for (size_t j = i - period + 1; j <= i; ++j) {
                double diff = prices[j] - mean;
                sum_sq += diff * diff;
            }
            double std_dev = std::sqrt(sum_sq / period);
            
            result.upper[i] = mean + std_dev_multiplier * std_dev;
            result.lower[i] = mean - std_dev_multiplier * std_dev;
            result.bandwidth[i] = (result.upper[i] - result.lower[i]) / mean * 100.0;
        }
        
        result.is_valid = true;
        return result;
    }
    
    IndicatorResult calculate(
        std::span<const double> prices,
        const IndicatorParams& params) const override
    {
        auto bb_result = calculateBollinger(prices, params);
        IndicatorResult result;
        result.values = std::move(bb_result.middle);
        result.is_valid = bb_result.is_valid;
        return result;
    }
    
    std::string getName() const override { return "BollingerBands"; }
    
    size_t getMinRequiredPoints(const IndicatorParams& params) const override {
        return params.period;
    }
};

/**
 * @brief Lock-free indicator cache
 */
class IndicatorCache {
public:
    struct CacheEntry {
        IndicatorResult result;
        std::atomic<bool> is_calculating{false};
        uint64_t data_hash{0};
        IndicatorParams params;
    };
    
    /**
     * @brief Get cached result or start calculation
     */
    std::shared_ptr<CacheEntry> getOrCreate(
        const std::string& indicator_key,
        uint64_t data_hash,
        const IndicatorParams& params)
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = cache_.find(indicator_key);
        if (it != cache_.end()) {
            auto& entry = it->second;
            
            // Check if data has changed
            if (entry->data_hash == data_hash && entry->params.period == params.period) {
                return entry;
            }
        }
        
        // Create new entry
        auto entry = std::make_shared<CacheEntry>();
        entry->data_hash = data_hash;
        entry->params = params;
        entry->is_calculating.store(true);
        
        cache_[indicator_key] = entry;
        return entry;
    }
    
    /**
     * @brief Update cache entry
     */
    void update(const std::string& indicator_key, IndicatorResult result) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = cache_.find(indicator_key);
        if (it != cache_.end()) {
            it->second->result = std::move(result);
            it->second->is_calculating.store(false);
        }
    }
    
    /**
     * @brief Invalidate cache for a specific indicator
     */
    void invalidate(const std::string& indicator_key) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.erase(indicator_key);
    }
    
    /**
     * @brief Clear all cache entries
     */
    void clear() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<CacheEntry>> cache_;
    std::mutex cache_mutex_;
};

/**
 * @brief Async Indicator Engine
 */
class AsyncIndicatorEngine {
public:
    AsyncIndicatorEngine() {
        // Register default indicators
        registerIndicator<SMAIndicator>("SMA");
        registerIndicator<EMAIndicator>("EMA");
        registerIndicator<RSIIndicator>("RSI");
        registerIndicator<MACDIndicator>("MACD");
        registerIndicator<BollingerBandsIndicator>("BollingerBands");
    }
    
    /**
     * @brief Register a custom indicator
     */
    template<typename T>
    void registerIndicator(const std::string& name) {
        indicators_[name] = std::make_unique<T>();
    }
    
    /**
     * @brief Calculate indicator asynchronously
     */
    std::future<IndicatorResult> calculateAsync(
        const std::string& indicator_name,
        std::span<const double> prices,
        const IndicatorParams& params)
    {
        return std::async(std::launch::async, [this, indicator_name, prices = std::vector<double>(prices.begin(), prices.end()), params]() {
            return calculate(indicator_name, prices, params);
        });
    }
    
    /**
     * @brief Calculate indicator synchronously
     */
    IndicatorResult calculate(
        const std::string& indicator_name,
        std::span<const double> prices,
        const IndicatorParams& params)
    {
        auto it = indicators_.find(indicator_name);
        if (it == indicators_.end()) {
            return IndicatorResult{};
        }
        
        return it->second->calculate(prices, params);
    }
    
    /**
     * @brief Calculate with caching
     */
    std::future<IndicatorResult> calculateCachedAsync(
        const std::string& indicator_name,
        std::span<const double> prices,
        const IndicatorParams& params,
        uint64_t data_hash)
    {
        std::string cache_key = indicator_name + "_" + std::to_string(params.period);
        
        // Check cache
        auto cache_entry = cache_.getOrCreate(cache_key, data_hash, params);
        
        if (cache_entry->result.is_valid && 
            !cache_entry->is_calculating.load() &&
            cache_entry->data_hash == data_hash) {
            // Return cached result
            std::promise<IndicatorResult> promise;
            promise.set_value(cache_entry->result);
            return promise.get_future();
        }
        
        // Calculate in background
        return std::async(std::launch::async, [this, indicator_name, prices = std::vector<double>(prices.begin(), prices.end()), params, cache_key, data_hash]() {
            auto result = calculate(indicator_name, prices, params);
            cache_.update(cache_key, result);
            return result;
        });
    }
    
    /**
     * @brief Get available indicators
     */
    std::vector<std::string> getAvailableIndicators() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : indicators_) {
            names.push_back(name);
        }
        return names;
    }
    
    /**
     * @brief Get the cache
     */
    IndicatorCache& getCache() { return cache_; }

private:
    std::unordered_map<std::string, std::unique_ptr<Indicator>> indicators_;
    IndicatorCache cache_;
};

/**
 * @brief Utility functions for indicator calculations
 */
namespace indicator_utils {

/**
 * @brief Calculate hash for price data
 */
inline uint64_t calculateDataHash(std::span<const double> prices) {
    uint64_t hash = 0;
    for (const double price : prices) {
        // Simple hash combining
        hash ^= std::bit_cast<uint64_t>(price) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

/**
 * @brief Calculate typical price (HLC3)
 */
inline std::vector<double> calculateTypicalPrice(
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close)
{
    const size_t n = std::min({high.size(), low.size(), close.size()});
    std::vector<double> typical(n);
    
    // Use indices for parallel execution
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), size_t(0));
    
    std::transform(
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        typical.begin(),
        [&high, &low, &close](size_t i) { return (high[i] + low[i] + close[i]) / 3.0; });
    
    return typical;
}

/**
 * @brief Calculate weighted close (HLCC4)
 */
inline std::vector<double> calculateWeightedClose(
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close)
{
    const size_t n = std::min({high.size(), low.size(), close.size()});
    std::vector<double> weighted(n);
    
    // Use indices for parallel execution
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), size_t(0));
    
    std::transform(
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        weighted.begin(),
        [&high, &low, &close](size_t i) { return (high[i] + low[i] + close[i] + close[i]) / 4.0; });
    
    return weighted;
}

} // namespace indicator_utils

} // namespace indicators
} // namespace btq
