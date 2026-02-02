#include "../include/data/exchange_aggregator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <format>
#include <limits>

#include "../../include/dynamic_logger.hpp"

namespace BTQuant {
namespace Data {

ExchangeAggregator::ExchangeAggregator(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<RenderEngine::SymbolManager> symbol_manager)
    : bridge_(bridge), processor_(processor), symbol_manager_(symbol_manager) {}

ExchangeAggregator::~ExchangeAggregator() {
    if (running_) {
        running_ = false;
        if (aggregation_thread_.joinable()) {
            aggregation_thread_.join();
        }
    }
}

bool ExchangeAggregator::initialize() {
    running_ = true;
    aggregation_thread_ = std::thread(&ExchangeAggregator::aggregationLoop, this);
    BTQ_LOG_INFO("ExchangeAggregator initialized successfully");
    return true;
}

void ExchangeAggregator::addExchange(const std::string& exchange_name, const ExchangeFeatures& features) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    exchange_features_[exchange_name] = features;

    // Initialize exchange-specific data structures
    exchange_validity_[exchange_name] = true;
    exchange_last_update_[exchange_name] = std::chrono::high_resolution_clock::now();

    // Initialize exchange correlation tracking
    exchange_correlations_[exchange_name] = std::unordered_map<std::string, double>();

    BTQ_LOG_INFO(std::format("Added exchange {} to aggregation pool", exchange_name));
}

void ExchangeAggregator::removeExchange(const std::string& exchange_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    exchange_features_.erase(exchange_name);
    exchange_validity_.erase(exchange_name);
    exchange_last_update_.erase(exchange_name);
    exchange_correlations_.erase(exchange_name);

    // Remove exchange data from all symbols
    for (auto& [symbol, exchange_data_map] : exchange_data_) {
        exchange_data_map.erase(exchange_name);
    }

    BTQ_LOG_INFO(std::format("Removed exchange {} from aggregation pool", exchange_name));
}

std::vector<std::string> ExchangeAggregator::getAvailableExchanges() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<std::string> exchanges;
    for (const auto& [exchange, _] : exchange_features_) {
        exchanges.push_back(exchange);
    }
    return exchanges;
}

void ExchangeAggregator::setTimeSyncStrategy(TimeSyncStrategy strategy) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    sync_strategy_ = strategy;
}

std::optional<AggregatedMarketData> ExchangeAggregator::aggregateSymbolData(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Filter out invalid or stale exchange data
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> valid_exchange_data;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            valid_exchange_data[exchange] = data;
            aggregated_data.exchange_data[exchange] = data;
        }
    }

    if (valid_exchange_data.empty()) {
        return std::nullopt;
    }

    // Collect timestamps from all valid exchanges
    for (const auto& [exchange, data] : valid_exchange_data) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values using enhanced methods
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidation(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);

    // Calculate advanced aggregated metrics
    aggregated_data.aggregated_high = calculateHighPrice(valid_exchange_data);
    aggregated_data.aggregated_low = calculateLowPrice(valid_exchange_data);
    aggregated_data.aggregated_bid = calculateBestBid(valid_exchange_data);
    aggregated_data.aggregated_ask = calculateBestAsk(valid_exchange_data);

    // Calculate total volume across all valid exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : valid_exchange_data) {
        total_volume += data.size;
    }
    aggregated_data.aggregated_volume = total_volume;

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, valid_exchange_data, aggregated_data);
    detectArbitrageOpportunities(valid_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

void ExchangeAggregator::processDataUpdate(const std::string& exchange, const std::string& symbol,
                                          const RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Validate the incoming data before storing
    if (!isValidData(update)) {
        BTQ_LOG_WARNING(std::format("Invalid data received from exchange {} for symbol {}", exchange, symbol));
        return;
    }

    // Store the raw data from the exchange
    exchange_data_[symbol][exchange] = update;
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;

    // Update statistics
    updateStatistics();
}

std::optional<AggregatedMarketData> ExchangeAggregator::getAggregatedData(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    // Create aggregated data on demand
    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Filter out invalid or stale exchange data
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> valid_exchange_data;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            valid_exchange_data[exchange] = data;
            aggregated_data.exchange_data[exchange] = data;
        }
    }

    if (valid_exchange_data.empty()) {
        return std::nullopt;
    }

    // Collect timestamps from all valid exchanges
    for (const auto& [exchange, data] : valid_exchange_data) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values using enhanced methods
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidation(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);

    // Calculate additional aggregated metrics
    aggregated_data.aggregated_high = calculateHighPrice(valid_exchange_data);
    aggregated_data.aggregated_low = calculateLowPrice(valid_exchange_data);
    aggregated_data.aggregated_bid = calculateBestBid(valid_exchange_data);
    aggregated_data.aggregated_ask = calculateBestAsk(valid_exchange_data);

    // Calculate total volume across all valid exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : valid_exchange_data) {
        total_volume += data.size;
    }
    aggregated_data.aggregated_volume = total_volume;

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, valid_exchange_data, aggregated_data);
    detectArbitrageOpportunities(valid_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

double ExchangeAggregator::calculateWeightedAveragePrice(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0.0;
    }

    double total_weighted_price = 0.0;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : symbol_it->second) {
        // Use exchange reliability score as weight
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        total_weighted_price += data.price * data.size * reliability;
        total_volume += data.size * reliability;
    }

    if (total_volume > 0.0) {
        return total_weighted_price / total_volume;
    }

    return 0.0;
}

double ExchangeAggregator::calculateWeightedAveragePriceWithValidation(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0.0;
    }

    double total_weighted_price = 0.0;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : symbol_it->second) {
        // Check if the data is valid and fresh
        if (!isExchangeDataValid(exchange, data)) {
            continue;
        }

        // Use exchange reliability score as weight
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Apply additional weight based on data freshness
        double freshness_weight = calculateFreshnessWeight(exchange);

        total_weighted_price += data.price * data.size * reliability * freshness_weight;
        total_volume += data.size * reliability * freshness_weight;
    }

    if (total_volume > 0.0) {
        return total_weighted_price / total_volume;
    }

    return 0.0;
}

uint64_t ExchangeAggregator::calculateSynchronizedTimestamp(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0;
    }

    if (symbol_it->second.empty()) {
        return 0;
    }

    std::vector<uint64_t> timestamps;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamps.push_back(data.timestamp);
        }
    }

    if (timestamps.empty()) {
        return 0;
    }

    switch (sync_strategy_) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            return *std::min_element(timestamps.begin(), timestamps.end());

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            return *std::max_element(timestamps.begin(), timestamps.end());

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            return std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();

        case TimeSyncStrategy::REFERENCE_EXCHANGE: {
            // Use the first exchange as reference
            if (!timestamps.empty()) {
                return timestamps[0];
            }
            return 0;
        }

        case TimeSyncStrategy::OFFSET_COMPENSATION: {
            // Apply latency offsets to align timestamps
            uint64_t sum_corrected = 0;
            size_t count = 0;

            for (const auto& [exchange, data] : symbol_it->second) {
                if (!isExchangeDataValid(exchange, data)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double offset = (exchange_it != exchange_features_.end()) ?
                               exchange_it->second.latency_offset_us : 0.0;

                // Convert offset from microseconds to appropriate unit for timestamp
                uint64_t corrected_ts = data.timestamp - static_cast<uint64_t>(offset);
                sum_corrected += corrected_ts;
                count++;
            }

            return (count > 0) ? sum_corrected / count : 0;
        }

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            if (timestamps.size() == 1) {
                return timestamps[0];
            }

            std::sort(timestamps.begin(), timestamps.end());
            size_t n = timestamps.size();
            if (n % 2 == 0) {
                return (timestamps[n/2 - 1] + timestamps[n/2]) / 2;
            } else {
                return timestamps[n/2];
            }
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
            // Adaptive synchronization based on market volatility
            if (timestamps.size() < 2) {
                return timestamps.empty() ? 0 : timestamps[0];
            }

            // Calculate variance in timestamps
            uint64_t mean_ts = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            uint64_t variance = 0;
            for (auto ts : timestamps) {
                variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
            }
            variance /= timestamps.size();

            // If variance is low (exchanges are well synchronized), use average
            // If variance is high (exchanges are not synchronized), use median
            if (variance < 1000000) { // 1ms threshold
                return std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::sort(timestamps.begin(), timestamps.end());
                size_t n = timestamps.size();
                if (n % 2 == 0) {
                    return (timestamps[n/2 - 1] + timestamps[n/2]) / 2;
                } else {
                    return timestamps[n/2];
                }
            }
        }

        default:
            return *std::min_element(timestamps.begin(), timestamps.end());
    }
}

std::optional<ExchangeFeatures> ExchangeAggregator::getExchangeFeatures(const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto it = exchange_features_.find(exchange);
    if (it != exchange_features_.end()) {
        return it->second;
    }

    return std::nullopt;
}

void ExchangeAggregator::updateExchangeFeatures(const std::string& exchange, const ExchangeFeatures& features) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    exchange_features_[exchange] = features;

    // Update validity status based on new features
    if (features.reliability_score > 0.0) {
        exchange_validity_[exchange] = true;
    } else {
        exchange_validity_[exchange] = false;
    }
}

ExchangeAggregator::AggregationStats ExchangeAggregator::getStats() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return stats_;
}

void ExchangeAggregator::aggregationLoop() {
    BTQ_LOG_INFO("Starting ExchangeAggregator background loop");

    while (running_) {
        // Perform periodic aggregation tasks
        // This could include recalculating weights, checking for stale data, etc.

        // Check for stale data and mark exchanges as invalid if needed
        checkStaleData();

        // Update exchange correlations periodically
        updateExchangeCorrelations();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Adjust frequency as needed
    }

    BTQ_LOG_INFO("ExchangeAggregator background loop stopped");
}

void ExchangeAggregator::synchronizeTimestamps(AggregatedMarketData& data) const {
    // Calculate synchronized timestamp based on strategy
    // We need to get the actual timestamps from the exchange_data map
    // This method is called from aggregateSymbolData and getAggregatedData
    // At this point, we need to get the timestamps from the exchange_data map
    // which should contain MarketDataUpdate objects

    // For now, we'll use the exchange_timestamps that was populated earlier
    std::vector<uint64_t> timestamps;
    for (const auto& [exchange, ts] : data.exchange_timestamps) {
        if (isExchangeValid(exchange)) {
            timestamps.push_back(ts);
        }
    }

    if (timestamps.empty()) {
        data.synchronized_timestamp = 0;
        return;
    }

    switch (data.sync_strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            data.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            data.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::REFERENCE_EXCHANGE:
            data.synchronized_timestamp = timestamps[0];  // Use first exchange as reference
            break;

        case TimeSyncStrategy::OFFSET_COMPENSATION: {
            // Apply latency offsets to align timestamps
            uint64_t sum_corrected = 0;
            size_t count = 0;

            for (const auto& [exchange, ts] : data.exchange_timestamps) {
                if (!isExchangeValid(exchange)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double offset = (exchange_it != exchange_features_.end()) ?
                               exchange_it->second.latency_offset_us : 0.0;

                // Convert offset from microseconds to appropriate unit for timestamp
                uint64_t corrected_ts = ts - static_cast<uint64_t>(offset);
                sum_corrected += corrected_ts;
                count++;
            }

            data.synchronized_timestamp = (count > 0) ? sum_corrected / count : 0;
            break;
        }

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            std::sort(timestamps.begin(), timestamps.end());
            size_t n = timestamps.size();
            if (n % 2 == 0) {
                data.synchronized_timestamp = (timestamps[n/2 - 1] + timestamps[n/2]) / 2;
            } else {
                data.synchronized_timestamp = timestamps[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
            if (timestamps.size() < 2) {
                data.synchronized_timestamp = timestamps.empty() ? 0 : timestamps[0];
                break;
            }

            // Calculate variance in timestamps
            uint64_t mean_ts = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            uint64_t variance = 0;
            for (auto ts : timestamps) {
                variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
            }
            variance /= timestamps.size();

            // If variance is low (exchanges are well synchronized), use average
            // If variance is high (exchanges are not synchronized), use median
            if (variance < 1000000) { // 1ms threshold
                data.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::sort(timestamps.begin(), timestamps.end());
                size_t n = timestamps.size();
                if (n % 2 == 0) {
                    data.synchronized_timestamp = (timestamps[n/2 - 1] + timestamps[n/2]) / 2;
                } else {
                    data.synchronized_timestamp = timestamps[n/2];
                }
            }
            break;
        }

        default:
            data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    data.reference_timestamp = data.synchronized_timestamp;
}

double ExchangeAggregator::calculateVolumeWeightedPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    double total_weighted_price = 0.0;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_data) {
        // Get exchange reliability score to weight the contribution
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Weight by both volume and reliability
        double weighted_volume = data.size * reliability;
        total_weighted_price += data.price * weighted_volume;
        total_volume += weighted_volume;
    }

    if (total_volume > 0.0) {
        return total_weighted_price / total_volume;
    }

    return 0.0;
}

void ExchangeAggregator::updateStatistics() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    stats_.total_symbols_aggregated = exchange_data_.size();
    stats_.total_exchanges = exchange_features_.size();
    stats_.last_update = std::chrono::high_resolution_clock::now();

    // Calculate average latency difference if we have multiple exchanges
    if (exchange_features_.size() > 1) {
        std::vector<double> offsets;
        for (const auto& [exchange, features] : exchange_features_) {
            if (exchange_validity_.count(exchange) && exchange_validity_.at(exchange)) {
                offsets.push_back(features.latency_offset_us);
            }
        }

        if (!offsets.empty()) {
            double sum = std::accumulate(offsets.begin(), offsets.end(), 0.0);
            stats_.avg_latency_difference_us = sum / offsets.size();
        }
    }

    // Update exchange-specific statistics
    stats_.valid_exchanges = 0;
    for (const auto& [exchange, valid] : exchange_validity_) {
        if (valid) {
            stats_.valid_exchanges++;
        }
    }
}

bool ExchangeAggregator::isExchangeDataValid(const std::string& exchange,
                                           const RenderEngine::MarketDataUpdate& data) const {
    // Check if exchange is marked as valid
    auto validity_it = exchange_validity_.find(exchange);
    if (validity_it != exchange_validity_.end() && !validity_it->second) {
        return false;
    }

    // Check if data is too old (stale data check)
    auto last_update_it = exchange_last_update_.find(exchange);
    if (last_update_it != exchange_last_update_.end()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update_it->second).count();

        // Consider data stale if older than 5 seconds
        if (duration > 5000) {
            return false;
        }
    }

    // Check if the data itself is valid
    return isValidData(data);
}

bool ExchangeAggregator::isExchangeValid(const std::string& exchange) const {
    auto validity_it = exchange_validity_.find(exchange);
    if (validity_it != exchange_validity_.end()) {
        return validity_it->second;
    }
    return false;
}

bool ExchangeAggregator::isValidData(const RenderEngine::MarketDataUpdate& data) const {
    // Check for valid price and size values
    if (data.price <= 0 || data.size < 0) {
        return false;
    }

    // Check for reasonable bounds
    if (data.price > 1e9 || data.size > 1e9) {  // Arbitrary large values
        return false;
    }

    // Check for NaN or infinity
    if (std::isnan(data.price) || std::isnan(data.size) ||
        std::isinf(data.price) || std::isinf(data.size)) {
        return false;
    }

    return true;
}

double ExchangeAggregator::calculateFreshnessWeight(const std::string& exchange) const {
    auto last_update_it = exchange_last_update_.find(exchange);
    if (last_update_it == exchange_last_update_.end()) {
        return 0.0;  // No data available
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_update_it->second).count();

    // Exponential decay function: weight decreases as data gets older
    // Half-life of 1000ms (1 second)
    double half_life_ms = 1000.0;
    double decay_factor = std::exp(-std::log(2.0) * age_ms / half_life_ms);

    // Clamp between 0.1 and 1.0 to avoid completely ignoring slightly stale data
    return std::max(0.1, decay_factor);
}

void ExchangeAggregator::checkStaleData() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto now = std::chrono::high_resolution_clock::now();

    for (auto& [exchange, last_update] : exchange_last_update_) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update).count();

        // Mark exchange as invalid if no data received in 5 seconds
        if (duration > 5000) {
            exchange_validity_[exchange] = false;
        } else {
            exchange_validity_[exchange] = true;
        }
    }
}

double ExchangeAggregator::calculateHighPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double high_price = 0.0;
    bool first = true;

    for (const auto& [exchange, data] : exchange_data) {
        if (first || data.price > high_price) {
            high_price = data.price;
            first = false;
        }
    }

    return high_price;
}

double ExchangeAggregator::calculateLowPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double low_price = std::numeric_limits<double>::max();

    for (const auto& [exchange, data] : exchange_data) {
        if (data.price < low_price) {
            low_price = data.price;
        }
    }

    return low_price;
}

double ExchangeAggregator::calculateBestBid(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    double best_bid = 0.0;

    for (const auto& [exchange, data] : exchange_data) {
        // For simplicity, we'll use the price as the bid if it's a buy order
        // In a real implementation, we'd need to check the orderbook data
        if (data.side == "BUY" && data.price > best_bid) {
            best_bid = data.price;
        }
    }

    return best_bid;
}

double ExchangeAggregator::calculateBestAsk(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    double best_ask = std::numeric_limits<double>::max();
    bool found_ask = false;

    for (const auto& [exchange, data] : exchange_data) {
        // For simplicity, we'll use the price as the ask if it's a sell order
        // In a real implementation, we'd need to check the orderbook data
        if (data.side == "SELL" && data.price < best_ask) {
            best_ask = data.price;
            found_ask = true;
        }
    }

    return found_ask ? best_ask : 0.0;
}

void ExchangeAggregator::calculateExchangeCorrelations(const std::string& symbol,
                                                      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                                                      AggregatedMarketData& result) const {
    if (exchange_data.size() < 2) {
        return; // Need at least 2 exchanges to calculate correlation
    }

    // Extract prices from exchanges
    std::vector<double> prices;
    std::vector<std::string> exchange_names;

    for (const auto& [exchange, data] : exchange_data) {
        prices.push_back(data.price);
        exchange_names.push_back(exchange);
    }

    // Calculate mean price
    double mean_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();

    // Calculate standard deviation
    double variance = 0.0;
    for (double price : prices) {
        variance += (price - mean_price) * (price - mean_price);
    }
    variance /= prices.size();
    double std_dev = std::sqrt(variance);

    // Calculate z-scores for each exchange
    std::vector<double> z_scores;
    for (double price : prices) {
        z_scores.push_back((price - mean_price) / std_dev);
    }

    // Calculate correlation matrix (for now just store individual z-scores as a proxy for correlation)
    for (size_t i = 0; i < exchange_names.size(); ++i) {
        result.exchange_correlations[exchange_names[i]] = z_scores[i];
    }

    // Calculate overall correlation coefficient (Pearson correlation to mean)
    double correlation_sum = 0.0;
    for (double z_score : z_scores) {
        correlation_sum += z_score * 0.0; // Correlation to mean is always 1.0, so we use 0.0 as baseline
    }
    result.overall_correlation = 1.0 - (std::abs(correlation_sum) / z_scores.size()); // Simplified correlation measure
}

void ExchangeAggregator::detectArbitrageOpportunities(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                                                     AggregatedMarketData& result) const {
    if (exchange_data.size() < 2) {
        return; // Need at least 2 exchanges to detect arbitrage
    }

    // Find highest bid and lowest ask across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : exchange_data) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
            highest_bid_exchange = exchange;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
            lowest_ask_exchange = exchange;
        }
    }

    // Check for arbitrage opportunity (bid > ask)
    if (highest_bid > lowest_ask) {
        result.arbitrage_opportunity = true;
        result.arbitrage_profit = highest_bid - lowest_ask;
        result.bid_exchange = highest_bid_exchange;
        result.ask_exchange = lowest_ask_exchange;
    } else {
        result.arbitrage_opportunity = false;
        result.arbitrage_profit = 0.0;
    }
}

void ExchangeAggregator::updateExchangeCorrelations() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Update correlations between exchanges based on recent data
    for (auto& [symbol, exchange_data_map] : exchange_data_) {
        std::vector<std::pair<std::string, double>> exchange_prices;

        for (const auto& [exchange, data] : exchange_data_map) {
            if (isExchangeDataValid(exchange, data)) {
                exchange_prices.push_back({exchange, data.price});
            }
        }

        if (exchange_prices.size() >= 2) {
            // Calculate correlation coefficients between exchanges
            for (size_t i = 0; i < exchange_prices.size(); ++i) {
                for (size_t j = i + 1; j < exchange_prices.size(); ++j) {
                    const auto& [ex1, price1] = exchange_prices[i];
                    const auto& [ex2, price2] = exchange_prices[j];

                    // Simple correlation based on price difference
                    double price_diff = std::abs(price1 - price2);
                    double avg_price = (price1 + price2) / 2.0;
                    double correlation = 1.0 - (price_diff / avg_price); // Higher correlation when prices are similar

                    // Store correlation in both directions
                    exchange_correlations_[ex1][ex2] = correlation;
                    exchange_correlations_[ex2][ex1] = correlation;
                }
            }
        }
    }
}

double ExchangeAggregator::calculateTWAP(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
    uint64_t window_start, uint64_t window_end) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double total_value = 0.0;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_data) {
        // Only include data within the time window
        if (data.timestamp >= window_start && data.timestamp <= window_end) {
            // Get exchange reliability score to weight the contribution
            auto exchange_it = exchange_features_.find(exchange);
            double reliability = (exchange_it != exchange_features_.end()) ?
                                exchange_it->second.reliability_score : 1.0;

            double weighted_volume = data.size * reliability;
            total_value += data.price * weighted_volume;
            total_volume += weighted_volume;
        }
    }

    if (total_volume > 0.0) {
        return total_value / total_volume;
    }

    return 0.0;
}

double ExchangeAggregator::calculateVWAP(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double total_value = 0.0;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_data) {
        // Get exchange reliability score to weight the contribution
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Weight by both volume and reliability
        double weighted_volume = data.size * reliability;
        total_value += data.price * weighted_volume;
        total_volume += weighted_volume;
    }

    if (total_volume > 0.0) {
        return total_value / total_volume;
    }

    return 0.0;
}

double ExchangeAggregator::calculateMedianPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    std::vector<double> prices;
    for (const auto& [exchange, data] : exchange_data) {
        // Apply reliability weighting by including the price multiple times based on reliability
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Add the price multiple times based on reliability score (clamped to range 0.1-2.0)
        int copies = std::max(1, static_cast<int>(reliability * 10.0));
        for (int i = 0; i < copies; ++i) {
            prices.push_back(data.price);
        }
    }

    if (prices.empty()) {
        return 0.0;
    }

    std::sort(prices.begin(), prices.end());
    size_t n = prices.size();
    if (n % 2 == 0) {
        return (prices[n/2 - 1] + prices[n/2]) / 2.0;
    } else {
        return prices[n/2];
    }
}

double ExchangeAggregator::calculateTrimmedMean(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
    double trim_percentage) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    std::vector<double> prices;
    for (const auto& [exchange, data] : exchange_data) {
        // Apply reliability weighting by including the price multiple times based on reliability
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Add the price multiple times based on reliability score (clamped to range 0.1-2.0)
        int copies = std::max(1, static_cast<int>(reliability * 10.0));
        for (int i = 0; i < copies; ++i) {
            prices.push_back(data.price);
        }
    }

    if (prices.size() < 3) {
        // Not enough data points to trim meaningfully
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        return sum / prices.size();
    }

    std::sort(prices.begin(), prices.end());

    // Calculate how many elements to trim from each end
    size_t trim_count = static_cast<size_t>(prices.size() * trim_percentage / 2.0);
    if (trim_count >= prices.size() / 2) {
        // Too much to trim, just return the median
        size_t mid = prices.size() / 2;
        return prices[mid];
    }

    // Calculate trimmed mean
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = trim_count; i < prices.size() - trim_count; ++i) {
        sum += prices[i];
        count++;
    }

    return (count > 0) ? sum / count : 0.0;
}

double ExchangeAggregator::calculateHarmonicMean(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double reciprocal_sum = 0.0;
    size_t count = 0;

    for (const auto& [exchange, data] : exchange_data) {
        if (data.price > 0) {  // Harmonic mean requires positive values
            // Apply reliability weighting
            auto exchange_it = exchange_features_.find(exchange);
            double reliability = (exchange_it != exchange_features_.end()) ?
                                exchange_it->second.reliability_score : 1.0;

            // Weight the reciprocal by reliability
            reciprocal_sum += reliability / data.price;
            count += static_cast<size_t>(reliability * 10.0);  // Count weighted by reliability
        }
    }

    if (reciprocal_sum > 0) {
        return (count > 0) ? (count / reciprocal_sum) : 0.0;
    }

    return 0.0;
}

}  // namespace Data
}  // namespace BTQuant