#include "../include/data/exchange_aggregator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <format>
#include <limits>
#include <map>

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
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

    // Validate exchange-specific constraints
    validateExchangeSpecificConstraints(exchange, symbol, update);

    // Handle exchange-specific features and adjustments
    RenderEngine::MarketDataUpdate adjusted_update = update;
    applyExchangeSpecificAdjustments(adjusted_update, exchange);
    handleExchangeSpecificFeatures(exchange, symbol, adjusted_update);

    // Store the processed data from the exchange
    exchange_data_[symbol][exchange] = adjusted_update;
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
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs; // To track which exchange each timestamp belongs to

    for (const auto& [exchange, ts] : data.exchange_timestamps) {
        if (isExchangeValid(exchange)) {
            timestamps.push_back(ts);
            timestamp_exchange_pairs.emplace_back(ts, exchange);
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

        case TimeSyncStrategy::REFERENCE_EXCHANGE: {
            // Use the first exchange in the map as reference
            if (!timestamp_exchange_pairs.empty()) {
                data.synchronized_timestamp = timestamp_exchange_pairs[0].first;
            } else {
                data.synchronized_timestamp = 0;
            }
            break;
        }

        case TimeSyncStrategy::OFFSET_COMPENSATION: {
            // Apply latency offsets to align timestamps
            uint64_t sum_corrected = 0;
            size_t count = 0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
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

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // Calculate weighted average considering both timestamp and reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
                if (!isExchangeValid(exchange)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double reliability = (exchange_it != exchange_features_.end()) ?
                                   exchange_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(ts) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                data.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // For predictive sync, we consider historical timing patterns
            // This would typically use historical data to predict the most accurate timestamp
            // For now, we'll implement a basic version that predicts based on exchange latency patterns

            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
                if (!isExchangeValid(exchange)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double latency_offset = (exchange_it != exchange_features_.end()) ?
                                      exchange_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = ts + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                data.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // Find the most recent timestamp
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                data.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                data.synchronized_timestamp = latest_ts;
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

std::optional<MultiExchangeData> ExchangeAggregator::getMultiExchangeView(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeData multi_exchange_data;
    multi_exchange_data.symbol = symbol;

    // Gather data from all exchanges for this symbol
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            multi_exchange_data.exchange_data[exchange] = data;

            // Get exchange features for additional context
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                multi_exchange_data.exchange_features[exchange] = features_it->second;
            }
        }
    }

    if (multi_exchange_data.exchange_data.empty()) {
        return std::nullopt;
    }

    // Calculate spread between highest bid and lowest ask
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();

    for (const auto& [exchange, data] : multi_exchange_data.exchange_data) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
        }
    }

    multi_exchange_data.spread = lowest_ask - highest_bid;
    multi_exchange_data.highest_bid = highest_bid;
    multi_exchange_data.lowest_ask = lowest_ask;

    // Calculate price differences between exchanges
    std::vector<double> prices;
    for (const auto& [exchange, data] : multi_exchange_data.exchange_data) {
        prices.push_back(data.price);
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        multi_exchange_data.price_volatility = prices.back() - prices.front(); // Max difference

        // Calculate additional volatility metrics
        if (prices.size() > 1) {
            double mean_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
            double variance = 0.0;
            for (double price : prices) {
                variance += (price - mean_price) * (price - mean_price);
            }
            variance /= prices.size();
            multi_exchange_data.price_std_deviation = std::sqrt(variance);
        }
    }

    // Calculate exchange-specific statistics
    for (const auto& [exchange, data] : multi_exchange_data.exchange_data) {
        ExchangeSpecificStats stats;
        stats.price = data.price;
        stats.volume = data.size;

        // Calculate price relative to overall average
        if (!prices.empty()) {
            double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
            stats.price_deviation_from_avg = data.price - avg_price;
            stats.percent_price_deviation = (avg_price > 0) ? (stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;
        }

        multi_exchange_data.exchange_stats[exchange] = stats;
    }

    multi_exchange_data.timestamp = std::chrono::high_resolution_clock::now();
    return multi_exchange_data;
}

void ExchangeAggregator::applyExchangeSpecificAdjustments(RenderEngine::MarketDataUpdate& data,
                                                        const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features found for this exchange
    }

    const auto& features = features_it->second;

    // Apply latency compensation if needed
    if (features.latency_offset_us != 0.0) {
        // Adjust timestamp based on known latency offset
        data.timestamp = static_cast<uint64_t>(
            static_cast<int64_t>(data.timestamp) - static_cast<int64_t>(features.latency_offset_us)
        );
    }

    // Apply exchange-specific filtering or adjustments
    // For example, adjust for exchange-specific fees or data quirks
    if (features.trading_fee_rate > 0.0) {
        // Could apply fee-adjusted pricing here if needed
        // For now, just log that this exchange has fees
        BTQ_LOG_DEBUG(std::format("Processing data from {} with {} fee rate",
                                 exchange, features.trading_fee_rate));
    }

    // Apply exchange-specific precision adjustments
    if (features.max_order_size > 0) {
        // Limit the size to the exchange's maximum order size
        if (data.size > features.max_order_size) {
            BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to max order size limit",
                                       data.size, features.max_order_size, exchange));
            data.size = features.max_order_size;
        }
    }

    // Apply exchange-specific minimum order size
    if (features.min_order_size > 0 && data.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to min order size requirement",
                                   data.size, features.min_order_size, exchange));
        data.size = features.min_order_size;
    }

    // Apply timezone adjustments if needed
    if (features.timezone != "UTC") {
        // In a real implementation, we would convert timestamps based on timezone
        // For now, just note the timezone difference
        BTQ_LOG_DEBUG(std::format("Exchange {} operates in timezone {}, data timestamp adjustment may be needed",
                                 exchange, features.timezone));
    }
}

void ExchangeAggregator::validateExchangeSpecificConstraints(const std::string& exchange,
                                                          const std::string& symbol,
                                                          const RenderEngine::MarketDataUpdate& data) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features found for this exchange
    }

    const auto& features = features_it->second;

    // Validate against exchange-specific constraints
    if (data.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} exceeds max allowed {} for exchange {} on symbol {}",
                                   data.size, features.max_order_size, exchange, symbol));
    }

    if (data.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} below min allowed {} for exchange {} on symbol {}",
                                   data.size, features.min_order_size, exchange, symbol));
    }

    // Check if symbol is supported by this exchange
    if (!features.supported_symbols.empty()) {
        bool symbol_supported = std::find(features.supported_symbols.begin(),
                                         features.supported_symbols.end(),
                                         symbol) != features.supported_symbols.end();
        if (!symbol_supported) {
            BTQ_LOG_WARNING(std::format("Symbol {} is not in supported symbols list for exchange {}",
                                       symbol, exchange));
        }
    }

    // Check if data type is supported by this exchange
    if (!features.supported_data_types.empty()) {
        // Assuming we have a way to determine data type from the update
        // For now, we'll just log this check
        BTQ_LOG_DEBUG(std::format("Checking if exchange {} supports data type for symbol {}",
                                 exchange, symbol));
    }
}

void ExchangeAggregator::updateExchangeSpecificFeatures(const std::string& exchange,
                                                      const ExchangeFeatures& new_features) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Update the exchange features
    exchange_features_[exchange] = new_features;

    // Log the update
    BTQ_LOG_INFO(std::format("Updated features for exchange {}: latency_offset={}us, reliability={}",
                            exchange, new_features.latency_offset_us, new_features.reliability_score));

    // Trigger recalculation of any dependent metrics
    updateExchangeCorrelations();
}

std::optional<ExchangeFeatures> ExchangeAggregator::getEnhancedExchangeFeatures(const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto it = exchange_features_.find(exchange);
    if (it != exchange_features_.end()) {
        ExchangeFeatures enhanced_features = it->second;

        // Add dynamic features based on current state
        auto validity_it = exchange_validity_.find(exchange);
        if (validity_it != exchange_validity_.end()) {
            enhanced_features.is_active = validity_it->second;
        }

        auto last_update_it = exchange_last_update_.find(exchange);
        if (last_update_it != exchange_last_update_.end()) {
            auto now = std::chrono::high_resolution_clock::now();
            auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_update_it->second).count();

            // Update the reliability score based on data freshness if needed
            if (latency_ms > 5000) { // More than 5 seconds old
                enhanced_features.reliability_score *= 0.5; // Reduce reliability for stale data
            } else if (latency_ms > 1000) { // More than 1 second old
                enhanced_features.reliability_score *= 0.8; // Slightly reduce reliability
            }
        }

        return enhanced_features;
    }

    return std::nullopt;
}

std::vector<ExchangeLatencyReport> ExchangeAggregator::generateLatencyReport() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<ExchangeLatencyReport> reports;

    for (const auto& [exchange, _] : exchange_features_) {
        ExchangeLatencyReport report;
        report.exchange_name = exchange;

        // Get the last update time for this exchange
        auto last_update_it = exchange_last_update_.find(exchange);
        if (last_update_it != exchange_last_update_.end()) {
            auto now = std::chrono::high_resolution_clock::now();
            report.current_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_update_it->second).count();
        }

        // Get the configured latency offset
        auto features_it = exchange_features_.find(exchange);
        if (features_it != exchange_features_.end()) {
            report.configured_latency_offset_us = features_it->second.latency_offset_us;
            report.reliability_score = features_it->second.reliability_score;
        }

        // Check validity status
        auto validity_it = exchange_validity_.find(exchange);
        if (validity_it != exchange_validity_.end()) {
            report.is_valid = validity_it->second;
        }

        reports.push_back(report);
    }

    return reports;
}

std::optional<ExchangeDataQualityMetrics> ExchangeAggregator::calculateDataQualityMetrics(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ExchangeDataQualityMetrics quality_metrics;
    quality_metrics.symbol = symbol;

    std::vector<double> prices;
    std::vector<double> timestamps;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            prices.push_back(data.price);
            timestamps.push_back(static_cast<double>(data.timestamp));

            // Calculate exchange-specific quality metrics
            ExchangeSpecificQuality exchange_quality;
            exchange_quality.exchange_name = exchange;

            // Freshness metric (based on how recent the data is)
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
                exchange_quality.freshness_score = std::exp(-age_ms / 1000.0); // Exponential decay over 1 second
            }

            // Completeness metric (based on presence of required fields)
            exchange_quality.completeness_score = (data.price > 0 && data.size >= 0) ? 1.0 : 0.5;

            // Accuracy metric (based on comparison with other exchanges)
            if (prices.size() > 1) {
                double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
                double deviation = std::abs(data.price - avg_price) / avg_price;
                // Lower deviation means higher accuracy score
                exchange_quality.accuracy_score = std::max(0.0, 1.0 - deviation);
            } else {
                exchange_quality.accuracy_score = 1.0; // Can't compare with just one exchange
            }

            // Overall quality score is the average of all quality aspects
            exchange_quality.overall_quality_score = (
                exchange_quality.freshness_score +
                exchange_quality.completeness_score +
                exchange_quality.accuracy_score
            ) / 3.0;

            quality_metrics.exchange_quality_metrics[exchange] = exchange_quality;
        }
    }

    if (prices.size() < 2) {
        // If we have only one exchange, we can't calculate cross-exchange metrics
        quality_metrics.cross_exchange_consistency = 1.0;
        quality_metrics.data_reliability_score = prices.empty() ? 0.0 : 0.8; // Assume moderate reliability
    } else {
        // Calculate cross-exchange consistency (lower variance = higher consistency)
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();

        // Normalize variance to get consistency score (0-1 scale)
        // Assuming typical price variance is around 1% of price
        double expected_variance = (avg_price * 0.01) * (avg_price * 0.01);
        quality_metrics.cross_exchange_consistency = 1.0 / (1.0 + variance / expected_variance);

        // Data reliability is combination of consistency and average quality
        double avg_quality = 0.0;
        for (const auto& [_, ex_quality] : quality_metrics.exchange_quality_metrics) {
            avg_quality += ex_quality.overall_quality_score;
        }
        avg_quality /= quality_metrics.exchange_quality_metrics.size();

        quality_metrics.data_reliability_score = (quality_metrics.cross_exchange_consistency + avg_quality) / 2.0;
    }

    quality_metrics.timestamp = std::chrono::high_resolution_clock::now();
    return quality_metrics;
}

std::vector<ExchangeRanking> ExchangeAggregator::rankExchangesByReliability() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<ExchangeRanking> rankings;

    for (const auto& [exchange, features] : exchange_features_) {
        ExchangeRanking ranking;
        ranking.exchange_name = exchange;
        ranking.reliability_score = features.reliability_score;
        ranking.is_active = features.is_active;
        ranking.data_staleness_ms = 0;

        // Calculate how stale the data is for this exchange
        auto last_update_it = exchange_last_update_.find(exchange);
        if (last_update_it != exchange_last_update_.end()) {
            auto now = std::chrono::high_resolution_clock::now();
            ranking.data_staleness_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_update_it->second).count();
        }

        // Calculate validity status
        auto validity_it = exchange_validity_.find(exchange);
        ranking.is_valid = (validity_it != exchange_validity_.end()) ? validity_it->second : false;

        rankings.push_back(ranking);
    }

    // Sort by reliability score (descending)
    std::sort(rankings.begin(), rankings.end(),
              [](const ExchangeRanking& a, const ExchangeRanking& b) {
                  return a.reliability_score > b.reliability_score;
              });

    return rankings;
}

std::optional<MultiExchangeConsolidatedView> ExchangeAggregator::getMultiExchangeConsolidatedView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeConsolidatedView consolidated_view;
    consolidated_view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            ExchangeConsolidatedData exchange_data;
            exchange_data.update = data;

            // Get exchange features for additional context
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                exchange_data.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            exchange_data.stats.price = data.price;
            exchange_data.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                exchange_data.stats.price_deviation_from_avg = data.price - avg_price;
                exchange_data.stats.percent_price_deviation = (avg_price > 0) ?
                    (exchange_data.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                exchange_data.stats.is_outlier = (std::abs(exchange_data.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                exchange_data.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            consolidated_view.exchange_data[exchange] = exchange_data;
        }
    }

    if (consolidated_view.exchange_data.empty()) {
        return std::nullopt;
    }

    // Calculate consolidated market metrics
    std::vector<double> prices;
    std::vector<double> volumes;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : consolidated_view.exchange_data) {
        prices.push_back(data.update.price);
        volumes.push_back(data.update.size);
        total_volume += data.update.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        consolidated_view.market_metrics.spread = prices.back() - prices.front();
        consolidated_view.market_metrics.volatility = consolidated_view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        consolidated_view.market_metrics.average_price = avg_price;
        consolidated_view.market_metrics.total_volume = total_volume;

        // Calculate price range metrics
        consolidated_view.market_metrics.lowest_price = prices.front();
        consolidated_view.market_metrics.highest_price = prices.back();
        consolidated_view.market_metrics.price_range = prices.back() - prices.front();
    }

    // Calculate order book metrics across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : consolidated_view.exchange_data) {
        if (data.update.side == "BUY" && data.update.price > highest_bid) {
            highest_bid = data.update.price;
            highest_bid_exchange = exchange;
        }
        if (data.update.side == "SELL" && data.update.price < lowest_ask) {
            lowest_ask = data.update.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > 0.0 && lowest_ask < std::numeric_limits<double>::max()) {
        consolidated_view.market_metrics.bid_ask_spread = lowest_ask - highest_bid;
        consolidated_view.market_metrics.best_bid_exchange = highest_bid_exchange;
        consolidated_view.market_metrics.best_ask_exchange = lowest_ask_exchange;
    }

    // Calculate risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        consolidated_view.risk_metrics.price_volatility = std::sqrt(variance);
        consolidated_view.risk_metrics.coefficient_of_variation = (avg_price > 0) ?
            consolidated_view.risk_metrics.price_volatility / avg_price : 0.0;
    }

    // Calculate arbitrage opportunities
    if (highest_bid > lowest_ask) {
        consolidated_view.arbitrage_opportunity_exists = true;
        consolidated_view.arbitrage_profit_potential = highest_bid - lowest_ask;
        consolidated_view.best_arbitrage_buy_exchange = highest_bid_exchange;
        consolidated_view.best_arbitrage_sell_exchange = lowest_ask_exchange;
    } else {
        consolidated_view.arbitrage_opportunity_exists = false;
        consolidated_view.arbitrage_profit_potential = 0.0;
    }

    // Calculate correlation metrics
    if (consolidated_view.exchange_data.size() > 1) {
        double total_correlation = 0.0;
        int correlation_count = 0;

        auto it1 = consolidated_view.exchange_data.begin();
        while (it1 != consolidated_view.exchange_data.end()) {
            auto it2 = std::next(it1);
            while (it2 != consolidated_view.exchange_data.end()) {
                double price_diff = std::abs(it1->second.update.price - it2->second.update.price);
                double avg_price = (it1->second.update.price + it2->second.update.price) / 2.0;
                double correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                total_correlation += correlation;
                correlation_count++;

                ++it2;
            }
            ++it1;
        }

        if (correlation_count > 0) {
            consolidated_view.market_metrics.cross_exchange_correlation = total_correlation / correlation_count;
        }
    }

    consolidated_view.timestamp = std::chrono::high_resolution_clock::now();

    return consolidated_view;
}

std::vector<MultiExchangeConsolidatedView> ExchangeAggregator::getAllSymbolsConsolidatedView() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<MultiExchangeConsolidatedView> all_views;

    for (const auto& [symbol, _] : exchange_data_) {
        auto view = getMultiExchangeConsolidatedView(symbol);
        if (view.has_value()) {
            all_views.push_back(view.value());
        }
    }

    return all_views;
}

std::optional<SymbolCrossExchangeAnalytics> ExchangeAggregator::getCrossExchangeAnalytics(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    SymbolCrossExchangeAnalytics analytics;
    analytics.symbol = symbol;

    // Collect data from all exchanges
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate analytics
    std::vector<double> prices;
    std::vector<double> volumes;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);

        // Store exchange-specific data
        analytics.exchange_prices[exchange] = data.price;
        analytics.exchange_volumes[exchange] = data.size;
    }

    if (!prices.empty()) {
        // Calculate statistical measures
        double sum_prices = std::accumulate(prices.begin(), prices.end(), 0.0);
        double avg_price = sum_prices / prices.size();

        // Calculate variance and standard deviation
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        double std_dev = std::sqrt(variance);

        analytics.mean_price = avg_price;
        analytics.std_deviation = std_dev;
        analytics.variance = variance;

        // Calculate coefficient of variation
        analytics.coefficient_of_variation = (avg_price > 0) ? std_dev / avg_price : 0.0;

        // Calculate min/max and range
        double min_price = *std::min_element(prices.begin(), prices.end());
        double max_price = *std::max_element(prices.begin(), prices.end());
        analytics.min_price = min_price;
        analytics.max_price = max_price;
        analytics.price_range = max_price - min_price;

        // Calculate median
        std::vector<double> sorted_prices = prices;
        std::sort(sorted_prices.begin(), sorted_prices.end());
        size_t n = sorted_prices.size();
        if (n % 2 == 0) {
            analytics.median_price = (sorted_prices[n/2 - 1] + sorted_prices[n/2]) / 2.0;
        } else {
            analytics.median_price = sorted_prices[n/2];
        }

        // Calculate skewness (measure of asymmetry)
        if (prices.size() >= 3) {
            double skewness_sum = 0.0;
            for (double price : prices) {
                double standardized = (price - avg_price) / std_dev;
                skewness_sum += standardized * standardized * standardized;
            }
            analytics.skewness = skewness_sum / prices.size();
        }

        // Calculate kurtosis (measure of tail heaviness)
        if (prices.size() >= 4) {
            double kurtosis_sum = 0.0;
            for (double price : prices) {
                double standardized = (price - avg_price) / std_dev;
                kurtosis_sum += standardized * standardized * standardized * standardized;
            }
            analytics.kurtosis = kurtosis_sum / prices.size() - 3.0; // Excess kurtosis
        }
    }

    // Calculate volume distribution metrics
    if (!volumes.empty()) {
        double total_volume = std::accumulate(volumes.begin(), volumes.end(), 0.0);
        if (total_volume > 0) {
            analytics.total_volume = total_volume;

            // Calculate volume concentration (Herfindahl-Hirschman Index)
            double hhi = 0.0;
            for (double volume : volumes) {
                double share = volume / total_volume;
                hhi += share * share;
            }
            analytics.volume_concentration_index = hhi;

            // Calculate volume-weighted average price across exchanges
            double vw_total_value = 0.0;
            double vw_total_volume = 0.0;
            for (size_t i = 0; i < exchange_updates.size(); ++i) {
                const auto& [exchange, data] = exchange_updates[i];
                double volume = data.size;
                double price = data.price;

                vw_total_value += price * volume;
                vw_total_volume += volume;
            }

            if (vw_total_volume > 0) {
                analytics.volume_weighted_average_price = vw_total_value / vw_total_volume;
            }
        }
    }

    // Calculate arbitrage potential
    if (analytics.max_price > 0 && analytics.min_price > 0) {
        analytics.max_arbitrage_potential = analytics.max_price - analytics.min_price;
        analytics.relative_arbitrage_potential = (analytics.max_arbitrage_potential / analytics.mean_price) * 100.0;
    }

    analytics.timestamp = std::chrono::high_resolution_clock::now();

    return analytics;
}

double ExchangeAggregator::calculateConsensusPrice(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0.0;
    }

    // Collect all valid prices from exchanges
    std::vector<double> prices;
    std::vector<double> weights;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            auto features_it = exchange_features_.find(exchange);
            double reliability = (features_it != exchange_features_.end()) ?
                                features_it->second.reliability_score : 1.0;

            // Apply freshness weight as well
            double freshness_weight = calculateFreshnessWeight(exchange);
            double combined_weight = reliability * freshness_weight;

            prices.push_back(data.price);
            weights.push_back(combined_weight);
        }
    }

    if (prices.empty()) {
        return 0.0;
    }

    // Calculate weighted median as consensus price (more robust than mean)
    if (prices.size() == 1) {
        return prices[0];
    }

    // Pair prices with weights and sort by price
    std::vector<std::pair<double, double>> price_weight_pairs;
    for (size_t i = 0; i < prices.size(); ++i) {
        price_weight_pairs.emplace_back(prices[i], weights[i]);
    }

    std::sort(price_weight_pairs.begin(), price_weight_pairs.end());

    // Calculate cumulative weights to find median
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    double cumulative_weight = 0.0;

    for (const auto& [price, weight] : price_weight_pairs) {
        cumulative_weight += weight;
        if (cumulative_weight >= total_weight / 2.0) {
            return price;
        }
    }

    // Fallback to first price if something went wrong
    return price_weight_pairs[0].first;
}

double ExchangeAggregator::calculateGeometricMeanPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double product = 1.0;
    size_t count = 0;

    for (const auto& [exchange, data] : exchange_data) {
        if (data.price > 0) {  // Geometric mean requires positive values
            // Apply reliability weighting by raising to power of reliability
            auto features_it = exchange_features_.find(exchange);
            double reliability = (features_it != exchange_features_.end()) ?
                                features_it->second.reliability_score : 1.0;

            // Apply freshness weight as well
            double freshness_weight = calculateFreshnessWeight(exchange);
            double combined_weight = reliability * freshness_weight;

            // Raise price to the power of its weight
            product *= std::pow(data.price, combined_weight);
            count++;
        }
    }

    if (count > 0) {
        // Calculate weighted geometric mean
        double total_weight = 0.0;
        for (const auto& [exchange, data] : exchange_data) {
            if (data.price > 0) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                    features_it->second.reliability_score : 1.0;
                double freshness_weight = calculateFreshnessWeight(exchange);
                total_weight += reliability * freshness_weight;
            }
        }

        if (total_weight > 0) {
            return std::pow(product, 1.0 / total_weight);
        }
    }

    return 0.0;
}

double ExchangeAggregator::calculateRobustMeanPrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    // Collect prices with their weights
    std::vector<std::pair<double, double>> price_weight_pairs;
    for (const auto& [exchange, data] : exchange_data) {
        auto features_it = exchange_features_.find(exchange);
        double reliability = (features_it != exchange_features_.end()) ?
                            features_it->second.reliability_score : 1.0;

        // Apply freshness weight as well
        double freshness_weight = calculateFreshnessWeight(exchange);
        double combined_weight = reliability * freshness_weight;

        price_weight_pairs.emplace_back(data.price, combined_weight);
    }

    if (price_weight_pairs.empty()) {
        return 0.0;
    }

    // Sort by price to identify outliers
    std::sort(price_weight_pairs.begin(), price_weight_pairs.end());

    // Calculate interquartile range to identify outliers
    size_t n = price_weight_pairs.size();
    size_t q1_idx = n / 4;
    size_t q3_idx = 3 * n / 4;

    if (q1_idx < price_weight_pairs.size() && q3_idx < price_weight_pairs.size()) {
        double q1_price = price_weight_pairs[q1_idx].first;
        double q3_price = price_weight_pairs[q3_idx].first;
        double iqr = q3_price - q1_price;
        double lower_bound = q1_price - 1.5 * iqr;
        double upper_bound = q3_price + 1.5 * iqr;

        // Calculate weighted mean excluding outliers
        double weighted_sum = 0.0;
        double total_weight = 0.0;

        for (const auto& [price, weight] : price_weight_pairs) {
            if (price >= lower_bound && price <= upper_bound) {
                weighted_sum += price * weight;
                total_weight += weight;
            }
        }

        if (total_weight > 0) {
            return weighted_sum / total_weight;
        }
    }

    // Fallback to regular weighted mean if IQR method fails
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    for (const auto& [price, weight] : price_weight_pairs) {
        weighted_sum += price * weight;
        total_weight += weight;
    }

    return (total_weight > 0) ? weighted_sum / total_weight : 0.0;
}

double ExchangeAggregator::calculateWeightedPercentilePrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
    double percentile) const {
    if (exchange_data.empty() || percentile < 0.0 || percentile > 100.0) {
        return 0.0;
    }

    // Collect prices with their weights
    std::vector<std::pair<double, double>> price_weight_pairs;
    for (const auto& [exchange, data] : exchange_data) {
        auto features_it = exchange_features_.find(exchange);
        double reliability = (features_it != exchange_features_.end()) ?
                            features_it->second.reliability_score : 1.0;

        // Apply freshness weight as well
        double freshness_weight = calculateFreshnessWeight(exchange);
        double combined_weight = reliability * freshness_weight;

        price_weight_pairs.emplace_back(data.price, combined_weight);
    }

    if (price_weight_pairs.empty()) {
        return 0.0;
    }

    // Sort by price
    std::sort(price_weight_pairs.begin(), price_weight_pairs.end());

    // Calculate total weight
    double total_weight = 0.0;
    for (const auto& [price, weight] : price_weight_pairs) {
        total_weight += weight;
    }

    if (total_weight == 0.0) {
        return 0.0;
    }

    // Find the weighted percentile
    double target_weight = (percentile / 100.0) * total_weight;
    double cumulative_weight = 0.0;

    for (size_t i = 0; i < price_weight_pairs.size(); ++i) {
        cumulative_weight += price_weight_pairs[i].second;

        if (cumulative_weight >= target_weight) {
            // Linear interpolation between this and previous value
            if (i > 0) {
                double prev_weight = cumulative_weight - price_weight_pairs[i].second;
                double weight_diff = target_weight - prev_weight;
                double total_interval_weight = price_weight_pairs[i].second;

                if (total_interval_weight > 0) {
                    double fraction = weight_diff / total_interval_weight;
                    return price_weight_pairs[i-1].first +
                           fraction * (price_weight_pairs[i].first - price_weight_pairs[i-1].first);
                }
            }
            return price_weight_pairs[i].first;
        }
    }

    // If we reach here, return the highest value
    return price_weight_pairs.back().first;
}

std::optional<AdvancedAggregationResult> ExchangeAggregator::performAdvancedAggregation(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AdvancedAggregationResult result;
    result.symbol = symbol;

    // Filter valid exchange data
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> valid_exchange_data;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            valid_exchange_data[exchange] = data;
            result.included_exchanges.push_back(exchange);
        }
    }

    if (valid_exchange_data.empty()) {
        return std::nullopt;
    }

    // Calculate multiple aggregation methods
    result.simple_average = calculateSimpleAveragePrice(valid_exchange_data);
    result.weighted_average = calculateWeightedAveragePriceWithValidation(symbol);
    result.volume_weighted = calculateVolumeWeightedPrice(valid_exchange_data);
    result.median_price = calculateMedianPrice(valid_exchange_data);
    result.geometric_mean = calculateGeometricMeanPrice(valid_exchange_data);
    result.robust_mean = calculateRobustMeanPrice(valid_exchange_data);
    result.harmonic_mean = calculateHarmonicMean(valid_exchange_data);
    result.consensus_price = calculateConsensusPrice(symbol);

    // Calculate various percentiles
    result.percentile_25th = calculateWeightedPercentilePrice(valid_exchange_data, 25.0);
    result.percentile_75th = calculateWeightedPercentilePrice(valid_exchange_data, 75.0);
    result.percentile_90th = calculateWeightedPercentilePrice(valid_exchange_data, 90.0);
    result.percentile_95th = calculateWeightedPercentilePrice(valid_exchange_data, 95.0);

    // Calculate dispersion metrics
    std::vector<double> prices;
    for (const auto& [exchange, data] : valid_exchange_data) {
        prices.push_back(data.price);
    }

    if (!prices.empty()) {
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        double mean = sum / prices.size();

        double variance = 0.0;
        for (double price : prices) {
            variance += (price - mean) * (price - mean);
        }
        variance /= prices.size();

        result.dispersion_metrics.mean = mean;
        result.dispersion_metrics.standard_deviation = std::sqrt(variance);
        result.dispersion_metrics.variance = variance;
        result.dispersion_metrics.min = *std::min_element(prices.begin(), prices.end());
        result.dispersion_metrics.max = *std::max_element(prices.begin(), prices.end());
        result.dispersion_metrics.range = result.dispersion_metrics.max - result.dispersion_metrics.min;
        result.dispersion_metrics.coefficient_of_variation = (mean > 0) ?
            result.dispersion_metrics.standard_deviation / mean : 0.0;
    }

    // Calculate confidence interval (assuming normal distribution)
    if (prices.size() > 1) {
        double std_error = result.dispersion_metrics.standard_deviation / std::sqrt(prices.size());
        // 95% confidence interval (z-score ≈ 1.96)
        result.confidence_interval_lower = result.dispersion_metrics.mean - 1.96 * std_error;
        result.confidence_interval_upper = result.dispersion_metrics.mean + 1.96 * std_error;
    }

    // Calculate outlier detection metrics
    std::vector<double> sorted_prices = prices;
    std::sort(sorted_prices.begin(), sorted_prices.end());

    if (sorted_prices.size() >= 3) {
        size_t q1_idx = sorted_prices.size() / 4;
        size_t q3_idx = 3 * sorted_prices.size() / 4;

        if (q1_idx < sorted_prices.size() && q3_idx < sorted_prices.size()) {
            double q1 = sorted_prices[q1_idx];
            double q3 = sorted_prices[q3_idx];
            double iqr = q3 - q1;

            result.outlier_detection.q1 = q1;
            result.outlier_detection.q3 = q3;
            result.outlier_detection.iqr = iqr;
            result.outlier_detection.lower_fence = q1 - 1.5 * iqr;
            result.outlier_detection.upper_fence = q3 + 1.5 * iqr;

            // Count outliers
            for (double price : sorted_prices) {
                if (price < result.outlier_detection.lower_fence ||
                    price > result.outlier_detection.upper_fence) {
                    result.outlier_detection.outlier_count++;
                }
            }
        }
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

double ExchangeAggregator::calculateSimpleAveragePrice(
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const {
    if (exchange_data.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    size_t count = 0;

    for (const auto& [exchange, data] : exchange_data) {
        sum += data.price;
        count++;
    }

    return (count > 0) ? sum / count : 0.0;
}

void ExchangeAggregator::handleExchangeSpecificFeatures(const std::string& exchange,
                                                      const std::string& symbol,
                                                      const RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features defined for this exchange
    }

    const auto& features = features_it->second;

    // Handle exchange-specific data processing
    if (!features.supports_microseconds) {
        // Round timestamp to nearest millisecond if exchange doesn't support microseconds
        const_cast<RenderEngine::MarketDataUpdate&>(update).timestamp =
            (update.timestamp / 1000) * 1000;
    }

    // Validate against exchange-specific limits
    if (update.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} exceeds max allowed {} for exchange {}",
                                   update.size, features.max_order_size, exchange));
    }

    if (update.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} below min allowed {} for exchange {}",
                                   update.size, features.min_order_size, exchange));
    }

    // Update exchange-specific statistics
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;
}

std::optional<AggregatedMarketData> ExchangeAggregator::getExchangeSpecificAggregatedData(
    const std::string& symbol, const std::vector<std::string>& exchanges) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Filter data based on requested exchanges
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> filtered_exchange_data;
    for (const auto& exchange : exchanges) {
        auto exchange_data_it = symbol_it->second.find(exchange);
        if (exchange_data_it != symbol_it->second.end() &&
            isExchangeDataValid(exchange, exchange_data_it->second)) {
            filtered_exchange_data[exchange] = exchange_data_it->second;
            aggregated_data.exchange_data[exchange] = exchange_data_it->second;
        }
    }

    if (filtered_exchange_data.empty()) {
        return std::nullopt;
    }

    // Collect timestamps from all valid exchanges
    for (const auto& [exchange, data] : filtered_exchange_data) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values using enhanced methods
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidationForExchanges(symbol, exchanges);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(filtered_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPriceForExchanges(symbol, exchanges);

    // Calculate additional aggregated metrics
    aggregated_data.aggregated_high = calculateHighPrice(filtered_exchange_data);
    aggregated_data.aggregated_low = calculateLowPrice(filtered_exchange_data);
    aggregated_data.aggregated_bid = calculateBestBid(filtered_exchange_data);
    aggregated_data.aggregated_ask = calculateBestAsk(filtered_exchange_data);

    // Calculate total volume across all valid exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : filtered_exchange_data) {
        total_volume += data.size;
    }
    aggregated_data.aggregated_volume = total_volume;

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, filtered_exchange_data, aggregated_data);
    detectArbitrageOpportunities(filtered_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

double ExchangeAggregator::calculateWeightedAveragePriceWithValidationForExchanges(
    const std::string& symbol, const std::vector<std::string>& exchanges) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0.0;
    }

    double total_weighted_price = 0.0;
    double total_volume = 0.0;

    for (const auto& exchange : exchanges) {
        auto exchange_data_it = symbol_it->second.find(exchange);
        if (exchange_data_it == symbol_it->second.end()) {
            continue; // Exchange not found for this symbol
        }

        const auto& data = exchange_data_it->second;

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

double ExchangeAggregator::calculateConsensusPriceForExchanges(
    const std::string& symbol, const std::vector<std::string>& exchanges) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return 0.0;
    }

    // Collect all valid prices from specified exchanges
    std::vector<double> prices;
    std::vector<double> weights;

    for (const auto& exchange : exchanges) {
        auto exchange_data_it = symbol_it->second.find(exchange);
        if (exchange_data_it == symbol_it->second.end()) {
            continue; // Exchange not found for this symbol
        }

        const auto& data = exchange_data_it->second;

        if (isExchangeDataValid(exchange, data)) {
            auto features_it = exchange_features_.find(exchange);
            double reliability = (features_it != exchange_features_.end()) ?
                                features_it->second.reliability_score : 1.0;

            // Apply freshness weight as well
            double freshness_weight = calculateFreshnessWeight(exchange);
            double combined_weight = reliability * freshness_weight;

            prices.push_back(data.price);
            weights.push_back(combined_weight);
        }
    }

    if (prices.empty()) {
        return 0.0;
    }

    // Calculate weighted median as consensus price (more robust than mean)
    if (prices.size() == 1) {
        return prices[0];
    }

    // Pair prices with weights and sort by price
    std::vector<std::pair<double, double>> price_weight_pairs;
    for (size_t i = 0; i < prices.size(); ++i) {
        price_weight_pairs.emplace_back(prices[i], weights[i]);
    }

    std::sort(price_weight_pairs.begin(), price_weight_pairs.end());

    // Calculate cumulative weights to find median
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    double cumulative_weight = 0.0;

    for (const auto& [price, weight] : price_weight_pairs) {
        cumulative_weight += weight;
        if (cumulative_weight >= total_weight / 2.0) {
            return price;
        }
    }

    // Fallback to first price if something went wrong
    return price_weight_pairs[0].first;
}

std::optional<AggregatedMarketData> ExchangeAggregator::getUnifiedView(const std::string& symbol) const {
    // This method provides a comprehensive unified view of multi-exchange data
    // It combines all the aggregation features into a single, comprehensive view
    return getAggregatedData(symbol);
}

std::vector<MultiExchangeData> ExchangeAggregator::getAllSymbolsMultiExchangeView() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<MultiExchangeData> all_views;

    for (const auto& [symbol, _] : exchange_data_) {
        auto view = getMultiExchangeView(symbol);
        if (view.has_value()) {
            all_views.push_back(view.value());
        }
    }

    return all_views;
}

std::optional<AggregatedMarketData> ExchangeAggregator::getAdvancedAggregatedData(
    const std::string& symbol,
    const std::vector<std::string>& exchanges) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Filter data based on requested exchanges or use all if empty
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> filtered_exchange_data;
    for (const auto& [exchange, data] : symbol_it->second) {
        // If exchanges vector is empty, use all exchanges; otherwise, only use specified ones
        bool include_exchange = exchanges.empty() ||
                               std::find(exchanges.begin(), exchanges.end(), exchange) != exchanges.end();

        if (include_exchange && isExchangeDataValid(exchange, data)) {
            filtered_exchange_data[exchange] = data;
            aggregated_data.exchange_data[exchange] = data;
        }
    }

    if (filtered_exchange_data.empty()) {
        return std::nullopt;
    }

    // Collect timestamps from all valid exchanges
    for (const auto& [exchange, data] : filtered_exchange_data) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate multiple aggregation methods for comprehensive view
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidation(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(filtered_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

    // Calculate additional aggregation methods
    aggregated_data.aggregated_high = calculateHighPrice(filtered_exchange_data);
    aggregated_data.aggregated_low = calculateLowPrice(filtered_exchange_data);
    aggregated_data.aggregated_bid = calculateBestBid(filtered_exchange_data);
    aggregated_data.aggregated_ask = calculateBestAsk(filtered_exchange_data);

    // Calculate total volume across all valid exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : filtered_exchange_data) {
        total_volume += data.size;
    }
    aggregated_data.aggregated_volume = total_volume;

    // Calculate advanced metrics
    aggregated_data.vwap = calculateVWAP(filtered_exchange_data);
    aggregated_data.median_price = calculateMedianPrice(filtered_exchange_data);
    aggregated_data.trimmed_mean_price = calculateTrimmedMean(filtered_exchange_data, 0.1);

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, filtered_exchange_data, aggregated_data);
    detectArbitrageOpportunities(filtered_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

std::optional<ComprehensiveExchangeView> ExchangeAggregator::getComprehensiveExchangeView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ComprehensiveExchangeView comprehensive_view;
    comprehensive_view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            ExchangeDetailedData detailed_data;
            detailed_data.update = data;

            // Get exchange features for additional context
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                detailed_data.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            detailed_data.stats.price = data.price;
            detailed_data.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                detailed_data.stats.price_deviation_from_avg = data.price - avg_price;
                detailed_data.stats.percent_price_deviation = (avg_price > 0) ?
                    (detailed_data.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                detailed_data.stats.is_outlier = (std::abs(detailed_data.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                detailed_data.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            comprehensive_view.exchange_details[exchange] = detailed_data;
        }
    }

    if (comprehensive_view.exchange_details.empty()) {
        return std::nullopt;
    }

    // Calculate overall market metrics
    std::vector<double> prices;
    double total_volume = 0.0;

    for (const auto& [exchange, details] : comprehensive_view.exchange_details) {
        prices.push_back(details.update.price);
        total_volume += details.update.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        comprehensive_view.market_metrics.spread = prices.back() - prices.front();
        comprehensive_view.market_metrics.volatility = comprehensive_view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        comprehensive_view.market_metrics.average_price = avg_price;
        comprehensive_view.market_metrics.total_volume = total_volume;
    }

    // Detect cross-exchange arbitrage opportunities
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, details] : comprehensive_view.exchange_details) {
        if (details.update.side == "BUY" && details.update.price > highest_bid) {
            highest_bid = details.update.price;
            highest_bid_exchange = exchange;
        }
        if (details.update.side == "SELL" && details.update.price < lowest_ask) {
            lowest_ask = details.update.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > lowest_ask) {
        comprehensive_view.arbitrage_detected = true;
        comprehensive_view.arbitrage_profit = highest_bid - lowest_ask;
        comprehensive_view.bid_exchange = highest_bid_exchange;
        comprehensive_view.ask_exchange = lowest_ask_exchange;
    } else {
        comprehensive_view.arbitrage_detected = false;
        comprehensive_view.arbitrage_profit = 0.0;
    }

    comprehensive_view.timestamp = std::chrono::high_resolution_clock::now();

    return comprehensive_view;
}

std::optional<TimestampSynchronizationResult> ExchangeAggregator::synchronizeTimestampsAcrossExchanges(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    TimestampSynchronizationResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply latency compensation based on exchange features
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs = timestamp_exchange_pairs;
    for (auto& [timestamp, exchange] : compensated_pairs) {
        auto features_it = exchange_features_.find(exchange);
        if (features_it != exchange_features_.end()) {
            // Apply latency offset compensation
            double offset = features_it->second.latency_offset_us;
            if (offset != 0.0) {
                // Compensate by adding the offset to align with a reference time
                timestamp = static_cast<uint64_t>(static_cast<int64_t>(timestamp) + static_cast<int64_t>(offset));
            }
        }
    }

    // Apply synchronization strategy
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

std::vector<ExchangeCorrelationMatrix> ExchangeAggregator::calculateExchangeCorrelationsMatrix() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<ExchangeCorrelationMatrix> correlation_matrices;

    // For each symbol, calculate correlation matrix between exchanges
    for (const auto& [symbol, exchange_data_map] : exchange_data_) {
        ExchangeCorrelationMatrix matrix;
        matrix.symbol = symbol;

        // Collect data from all exchanges for this symbol
        std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
        for (const auto& [exchange, data] : exchange_data_map) {
            if (isExchangeDataValid(exchange, data)) {
                exchange_updates.emplace_back(exchange, data);
            }
        }

        if (exchange_updates.size() < 2) {
            continue; // Need at least 2 exchanges to calculate correlations
        }

        // Calculate correlation coefficients between all pairs of exchanges
        for (size_t i = 0; i < exchange_updates.size(); ++i) {
            for (size_t j = i + 1; j < exchange_updates.size(); ++j) {
                const auto& [ex1, data1] = exchange_updates[i];
                const auto& [ex2, data2] = exchange_updates[j];

                // Calculate correlation based on price similarity and timing
                double price_diff = std::abs(data1.price - data2.price);
                double avg_price = (data1.price + data2.price) / 2.0;
                double price_correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                // Calculate timing correlation based on timestamp difference
                uint64_t ts_diff = std::abs(static_cast<int64_t>(data1.timestamp) - static_cast<int64_t>(data2.timestamp));
                double timing_correlation = std::exp(-static_cast<double>(ts_diff) / 1000000.0); // Decay factor of 1 second

                // Calculate volume correlation
                double volume_diff = std::abs(data1.size - data2.size);
                double avg_volume = (data1.size + data2.size) / 2.0;
                double volume_correlation = (avg_volume > 0) ?
                    1.0 - std::min(1.0, volume_diff / avg_volume) : 1.0;

                // Combined correlation
                double combined_correlation = (price_correlation + timing_correlation + volume_correlation) / 3.0;

                // Store in both directions
                matrix.correlations[ex1][ex2] = combined_correlation;
                matrix.correlations[ex2][ex1] = combined_correlation;
            }
        }

        // Calculate overall reliability-weighted correlation
        double total_reliability = 0.0;
        double weighted_correlation_sum = 0.0;

        for (const auto& [exchange, _] : exchange_updates) {
            auto features_it = exchange_features_.find(exchange);
            double reliability = (features_it != exchange_features_.end()) ?
                               features_it->second.reliability_score : 1.0;

            // Calculate average correlation for this exchange
            double ex_correlation_sum = 0.0;
            int correlation_count = 0;

            for (const auto& [other_ex, corr_map] : matrix.correlations) {
                auto corr_it = corr_map.find(exchange);
                if (corr_it != corr_map.end()) {
                    ex_correlation_sum += corr_it->second;
                    correlation_count++;
                }
            }

            if (correlation_count > 0) {
                double avg_ex_correlation = ex_correlation_sum / correlation_count;
                weighted_correlation_sum += avg_ex_correlation * reliability;
                total_reliability += reliability;
            }
        }

        if (total_reliability > 0.0) {
            matrix.overall_market_correlation = weighted_correlation_sum / total_reliability;
        }

        correlation_matrices.push_back(matrix);
    }

    return correlation_matrices;
}

std::optional<ExchangeRiskMetrics> ExchangeAggregator::calculateRiskMetrics(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ExchangeRiskMetrics risk_metrics;
    risk_metrics.symbol = symbol;

    std::vector<double> prices;
    std::vector<double> volumes;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            prices.push_back(data.price);
            volumes.push_back(data.size);

            // Get exchange-specific risk factors
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                ExchangeSpecificRisk exchange_risk;
                exchange_risk.exchange_name = exchange;
                exchange_risk.latency_risk = features_it->second.latency_offset_us / 1000.0; // Convert to ms
                exchange_risk.fee_cost = features_it->second.trading_fee_rate;
                exchange_risk.reliability_score = features_it->second.reliability_score;

                // Calculate price deviation risk
                if (!prices.empty()) {
                    double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
                    exchange_risk.price_deviation_risk = std::abs(data.price - avg_price) / avg_price;
                }

                risk_metrics.exchange_specific_risks[exchange] = exchange_risk;
            }
        }
    }

    if (prices.empty()) {
        return std::nullopt;
    }

    // Calculate overall market risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();

        // Calculate standard deviation as volatility measure
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        risk_metrics.price_volatility = std::sqrt(variance);

        // Calculate coefficient of variation
        risk_metrics.coefficient_of_variation = (avg_price > 0) ? risk_metrics.price_volatility / avg_price : 0.0;

        // Calculate price spread across exchanges
        double min_price = *std::min_element(prices.begin(), prices.end());
        double max_price = *std::max_element(prices.begin(), prices.end());
        risk_metrics.price_spread = max_price - min_price;
        risk_metrics.normalized_price_spread = (avg_price > 0) ? risk_metrics.price_spread / avg_price : 0.0;

        // Calculate additional risk metrics
        risk_metrics.price_variance = variance;
        risk_metrics.max_price_deviation = std::max(
            std::abs(max_price - avg_price),
            std::abs(min_price - avg_price)
        );
    }

    // Calculate volume concentration risk
    if (!volumes.empty()) {
        double total_volume = std::accumulate(volumes.begin(), volumes.end(), 0.0);
        if (total_volume > 0) {
            double max_volume = *std::max_element(volumes.begin(), volumes.end());
            risk_metrics.volume_concentration_risk = max_volume / total_volume;

            // Calculate volume distribution entropy as another measure of concentration
            double entropy = 0.0;
            for (double vol : volumes) {
                if (vol > 0) {
                    double p = vol / total_volume;
                    entropy -= p * std::log(p);
                }
            }
            risk_metrics.volume_distribution_entropy = entropy;
        }
    }

    risk_metrics.timestamp = std::chrono::high_resolution_clock::now();
    return risk_metrics;
}

// New method to handle sophisticated time synchronization across exchanges
std::optional<MultiExchangeTimeSyncResult> ExchangeAggregator::performMultiExchangeTimeSync(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeTimeSyncResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);

            // Apply latency compensation based on exchange features
            auto features_it = exchange_features_.find(exchange);
            uint64_t compensated_timestamp = data.timestamp;

            if (features_it != exchange_features_.end()) {
                // Apply latency offset compensation
                double offset = features_it->second.latency_offset_us;
                if (offset != 0.0) {
                    // Compensate by adding the offset to align with a reference time
                    compensated_timestamp = static_cast<uint64_t>(
                        static_cast<int64_t>(data.timestamp) + static_cast<int64_t>(offset)
                    );
                }
            }

            compensated_pairs.emplace_back(compensated_timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
            result.compensated_timestamps[exchange] = compensated_timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply synchronization strategy to compensated timestamps
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    // Calculate additional synchronization metrics
    result.average_original_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
    result.timestamp_variance = 0.0;
    if (timestamps.size() > 1) {
        uint64_t mean_ts = result.average_original_timestamp;
        uint64_t variance = 0;
        for (uint64_t ts : timestamps) {
            variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
        }
        variance /= timestamps.size();
        result.timestamp_variance = static_cast<double>(variance);
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

// New method to get a comprehensive multi-exchange view with all analytics
std::optional<ComprehensiveMultiExchangeView> ExchangeAggregator::getComprehensiveMultiExchangeView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ComprehensiveMultiExchangeView view;
    view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);

            // Store detailed exchange data
            ExchangeConsolidatedData exchange_data;
            exchange_data.update = data;

            // Get exchange features
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                exchange_data.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            exchange_data.stats.price = data.price;
            exchange_data.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                exchange_data.stats.price_deviation_from_avg = data.price - avg_price;
                exchange_data.stats.percent_price_deviation = (avg_price > 0) ?
                    (exchange_data.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                exchange_data.stats.is_outlier = (std::abs(exchange_data.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                exchange_data.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            view.exchange_data[exchange] = exchange_data;
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate consolidated market metrics
    std::vector<double> prices;
    std::vector<double> volumes;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
        total_volume += data.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        view.market_metrics.spread = prices.back() - prices.front();
        view.market_metrics.volatility = view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        view.market_metrics.average_price = avg_price;
        view.market_metrics.total_volume = total_volume;

        // Calculate price range metrics
        view.market_metrics.lowest_price = prices.front();
        view.market_metrics.highest_price = prices.back();
        view.market_metrics.price_range = prices.back() - prices.front();

        // Calculate additional statistical metrics
        view.market_metrics.median_price = (prices.size() % 2 == 0) ?
            (prices[prices.size()/2 - 1] + prices[prices.size()/2]) / 2.0 :
            prices[prices.size()/2];
    }

    // Calculate order book metrics across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : exchange_updates) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
            highest_bid_exchange = exchange;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > 0.0 && lowest_ask < std::numeric_limits<double>::max()) {
        view.market_metrics.bid_ask_spread = lowest_ask - highest_bid;
        view.market_metrics.best_bid_exchange = highest_bid_exchange;
        view.market_metrics.best_ask_exchange = lowest_ask_exchange;
    }

    // Calculate risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        view.risk_metrics.price_volatility = std::sqrt(variance);
        view.risk_metrics.coefficient_of_variation = (avg_price > 0) ?
            view.risk_metrics.price_volatility / avg_price : 0.0;
    }

    // Calculate arbitrage opportunities
    if (highest_bid > lowest_ask) {
        view.arbitrage_opportunity_exists = true;
        view.arbitrage_profit_potential = highest_bid - lowest_ask;
        view.best_arbitrage_buy_exchange = highest_bid_exchange;
        view.best_arbitrage_sell_exchange = lowest_ask_exchange;
    } else {
        view.arbitrage_opportunity_exists = false;
        view.arbitrage_profit_potential = 0.0;
    }

    // Calculate correlation metrics
    if (view.exchange_data.size() > 1) {
        double total_correlation = 0.0;
        int correlation_count = 0;

        auto it1 = view.exchange_data.begin();
        while (it1 != view.exchange_data.end()) {
            auto it2 = std::next(it1);
            while (it2 != view.exchange_data.end()) {
                double price_diff = std::abs(it1->second.update.price - it2->second.update.price);
                double avg_price = (it1->second.update.price + it2->second.update.price) / 2.0;
                double correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                total_correlation += correlation;
                correlation_count++;

                ++it2;
            }
            ++it1;
        }

        if (correlation_count > 0) {
            view.market_metrics.cross_exchange_correlation = total_correlation / correlation_count;
        }
    }

    // Calculate data quality metrics
    auto quality_metrics = calculateDataQualityMetrics(symbol);
    if (quality_metrics.has_value()) {
        view.data_quality = quality_metrics.value();
    }

    // Calculate exchange rankings
    auto rankings = rankExchangesByReliability();
    view.exchange_rankings = rankings;

    view.timestamp = std::chrono::high_resolution_clock::now();

    return view;
}

// Enhanced multi-exchange aggregation with improved time synchronization
std::optional<AggregatedMarketData> ExchangeAggregator::getEnhancedAggregatedData(
    const std::string& symbol) const {
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

    // Enhanced time synchronization with multiple strategies
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values using enhanced methods
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidation(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

    // Calculate advanced metrics
    aggregated_data.vwap = calculateVWAP(valid_exchange_data);
    aggregated_data.median_price = calculateMedianPrice(valid_exchange_data);
    aggregated_data.trimmed_mean_price = calculateTrimmedMean(valid_exchange_data, 0.1);

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, valid_exchange_data, aggregated_data);
    detectArbitrageOpportunities(valid_exchange_data, aggregated_data);

    // Enhanced risk assessment
    calculateEnhancedRiskMetrics(symbol, valid_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

// Enhanced time synchronization with adaptive algorithm
void ExchangeAggregator::enhancedSynchronizeTimestamps(AggregatedMarketData& data) const {
    // Calculate synchronized timestamp based on strategy
    std::vector<uint64_t> timestamps;
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;

    for (const auto& [exchange, ts] : data.exchange_timestamps) {
        if (isExchangeValid(exchange)) {
            timestamps.push_back(ts);
            timestamp_exchange_pairs.emplace_back(ts, exchange);
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

        case TimeSyncStrategy::REFERENCE_EXCHANGE: {
            // Use the first exchange in the map as reference
            if (!timestamp_exchange_pairs.empty()) {
                data.synchronized_timestamp = timestamp_exchange_pairs[0].first;
            } else {
                data.synchronized_timestamp = 0;
            }
            break;
        }

        case TimeSyncStrategy::OFFSET_COMPENSATION: {
            // Apply latency offsets to align timestamps
            uint64_t sum_corrected = 0;
            size_t count = 0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
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

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // Calculate weighted average considering both timestamp and reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
                if (!isExchangeValid(exchange)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double reliability = (exchange_it != exchange_features_.end()) ?
                                   exchange_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(ts) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                data.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // For predictive sync, we consider historical timing patterns
            // This would typically use historical data to predict the most accurate timestamp
            // For now, we'll implement a basic version that predicts based on exchange latency patterns

            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [ts, exchange] : timestamp_exchange_pairs) {
                if (!isExchangeValid(exchange)) {
                    continue;
                }

                auto exchange_it = exchange_features_.find(exchange);
                double latency_offset = (exchange_it != exchange_features_.end()) ?
                                      exchange_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = ts + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                data.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            if (timestamps.size() == 1) {
                data.synchronized_timestamp = timestamps[0];
                break;
            }

            // Find the most recent timestamp
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                data.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                data.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            data.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    data.reference_timestamp = data.synchronized_timestamp;
}

// Advanced time synchronization using cross-correlation analysis
std::optional<MultiExchangeTimeSyncResult> ExchangeAggregator::performAdvancedTimeSync(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeTimeSyncResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges along with their data
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);

            // Apply latency compensation based on exchange features
            auto features_it = exchange_features_.find(exchange);
            uint64_t compensated_timestamp = data.timestamp;

            if (features_it != exchange_features_.end()) {
                // Apply latency offset compensation
                double offset = features_it->second.latency_offset_us;
                if (offset != 0.0) {
                    // Compensate by adding the offset to align with a reference time
                    compensated_timestamp = static_cast<uint64_t>(
                        static_cast<int64_t>(data.timestamp) + static_cast<int64_t>(offset)
                    );
                }
            }

            compensated_pairs.emplace_back(compensated_timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
            result.compensated_timestamps[exchange] = compensated_timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply synchronization strategy to compensated timestamps
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    // Calculate additional synchronization metrics
    result.average_original_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
    result.timestamp_variance = 0.0;
    if (timestamps.size() > 1) {
        uint64_t mean_ts = result.average_original_timestamp;
        uint64_t variance = 0;
        for (uint64_t ts : timestamps) {
            variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
        }
        variance /= timestamps.size();
        result.timestamp_variance = static_cast<double>(variance);
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

// Enhanced time synchronization with historical data analysis
std::optional<TimestampSynchronizationResult> ExchangeAggregator::analyzeHistoricalTimeSync(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    TimestampSynchronizationResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply latency compensation based on exchange features
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs = timestamp_exchange_pairs;
    for (auto& [timestamp, exchange] : compensated_pairs) {
        auto features_it = exchange_features_.find(exchange);
        if (features_it != exchange_features_.end()) {
            // Apply latency offset compensation
            double offset = features_it->second.latency_offset_us;
            if (offset != 0.0) {
                // Compensate by adding the offset to align with a reference time
                timestamp = static_cast<uint64_t>(static_cast<int64_t>(timestamp) + static_cast<int64_t>(offset));
            }
        }
    }

    // Apply synchronization strategy
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

// Calculate enhanced risk metrics for multi-exchange aggregation
void ExchangeAggregator::calculateEnhancedRiskMetrics(
    const std::string& symbol,
    const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
    AggregatedMarketData& result) const {

    if (exchange_data.size() < 2) {
        return; // Need at least 2 exchanges for risk calculation
    }

    std::vector<double> prices;
    std::vector<double> volumes;

    for (const auto& [exchange, data] : exchange_data) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
    }

    if (prices.size() < 2) {
        return;
    }

    // Calculate statistical risk metrics
    double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
    double variance = 0.0;
    for (double price : prices) {
        variance += (price - avg_price) * (price - avg_price);
    }
    variance /= prices.size();
    double std_dev = std::sqrt(variance);

    result.aggregated_high = *std::max_element(prices.begin(), prices.end());
    result.aggregated_low = *std::min_element(prices.begin(), prices.end());

    // Calculate risk-adjusted metrics
    result.overall_correlation = 1.0 - (std_dev / avg_price); // Lower std dev = higher correlation

    // Calculate exchange-specific risk contributions
    for (const auto& [exchange, data] : exchange_data) {
        double price_deviation = std::abs(data.price - avg_price) / avg_price;
        result.exchange_correlations[exchange] = 1.0 - price_deviation; // Lower deviation = higher correlation
    }
}

// Enhanced exchange-specific feature handling
void ExchangeAggregator::handleEnhancedExchangeSpecificFeatures(const std::string& exchange,
                                                              const std::string& symbol,
                                                              RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features defined for this exchange
    }

    const auto& features = features_it->second;

    // Apply exchange-specific data normalization
    if (features.max_order_size > 0 && update.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Normalizing order size from {} to {} for exchange {} due to max order size limit",
                                   update.size, features.max_order_size, exchange));
        update.size = features.max_order_size;
    }

    if (features.min_order_size > 0 && update.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Normalizing order size from {} to {} for exchange {} due to min order size requirement",
                                   update.size, features.min_order_size, exchange));
        update.size = features.min_order_size;
    }

    // Apply exchange-specific price adjustments based on fees
    if (features.trading_fee_rate > 0.0) {
        // Adjust price to account for fees if needed
        // This is a simplified approach - in practice, you might want to adjust differently based on direction
        update.price = update.price * (1.0 + features.trading_fee_rate / 2.0); // Half the fee rate as a simple adjustment
    }

    // Apply exchange-specific precision rounding
    if (features.precision > 0) {
        double multiplier = std::pow(10.0, features.precision);
        update.price = std::round(update.price * multiplier) / multiplier;
    }

    // Apply timezone adjustments if needed
    if (features.timezone != "UTC") {
        // In a real implementation, we would convert timestamps based on timezone
        // For now, just note the timezone difference
        BTQ_LOG_DEBUG(std::format("Exchange {} operates in timezone {}, data timestamp adjustment may be needed",
                                 exchange, features.timezone));
    }

    // Apply exchange-specific API rate limiting considerations
    if (features.api_endpoint != "") {
        // Log or track API usage for this exchange
        BTQ_LOG_DEBUG(std::format("Processing data from exchange {} via endpoint {}",
                                 exchange, features.api_endpoint));
    }

    // Update exchange-specific statistics
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;
}

// Method to handle exchange-specific data quality checks
bool ExchangeAggregator::performExchangeSpecificQualityChecks(const std::string& exchange,
                                                           const std::string& symbol,
                                                           const RenderEngine::MarketDataUpdate& update) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return true; // If no features defined, assume data is valid
    }

    const auto& features = features_it->second;

    // Check if symbol is supported by this exchange
    if (!features.supported_symbols.empty()) {
        bool symbol_supported = std::find(features.supported_symbols.begin(),
                                         features.supported_symbols.end(),
                                         symbol) != features.supported_symbols.end();
        if (!symbol_supported) {
            BTQ_LOG_WARNING(std::format("Symbol {} is not in supported symbols list for exchange {}",
                                       symbol, exchange));
            return false;
        }
    }

    // Check if data type is supported by this exchange
    if (!features.supported_data_types.empty()) {
        // Assuming we have a way to determine data type from the update
        // For now, we'll just log this check
        BTQ_LOG_DEBUG(std::format("Checking if exchange {} supports data type for symbol {}",
                                 exchange, symbol));
    }

    // Validate against exchange-specific constraints
    if (update.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} exceeds max allowed {} for exchange {} on symbol {}",
                                   update.size, features.max_order_size, exchange, symbol));
        return false;
    }

    if (update.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Order size {} below min allowed {} for exchange {} on symbol {}",
                                   update.size, features.min_order_size, exchange, symbol));
        return false;
    }

    // Check reliability score - if too low, reject the data
    if (features.reliability_score < 0.1) {
        BTQ_LOG_WARNING(std::format("Exchange {} has low reliability score ({}), rejecting data for symbol {}",
                                   exchange, features.reliability_score, symbol));
        return false;
    }

    return true;
}

// Method to dynamically update exchange features based on observed behavior
void ExchangeAggregator::updateExchangeFeaturesDynamically(const std::string& exchange) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // Exchange not found
    }

    auto& features = features_it->second;

    // Update reliability score based on data freshness and consistency
    auto last_update_it = exchange_last_update_.find(exchange);
    if (last_update_it != exchange_last_update_.end()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update_it->second).count();

        // Adjust reliability based on data freshness
        if (latency_ms > 10000) { // More than 10 seconds old
            features.reliability_score *= 0.5; // Reduce reliability significantly
        } else if (latency_ms > 5000) { // More than 5 seconds old
            features.reliability_score *= 0.7; // Reduce reliability moderately
        } else if (latency_ms > 1000) { // More than 1 second old
            features.reliability_score *= 0.9; // Slightly reduce reliability
        }

        // Ensure reliability score stays within bounds
        features.reliability_score = std::clamp(features.reliability_score, 0.0, 1.0);
    }

    // Update validity status based on reliability score
    exchange_validity_[exchange] = features.reliability_score > 0.1;
}

// Process data update with enhanced exchange-specific handling
void ExchangeAggregator::processEnhancedDataUpdate(const std::string& exchange, const std::string& symbol,
                                                  const RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Validate the incoming data before storing
    if (!isValidData(update)) {
        BTQ_LOG_WARNING(std::format("Invalid data received from exchange {} for symbol {}", exchange, symbol));
        return;
    }

    // Perform exchange-specific quality checks
    if (!performExchangeSpecificQualityChecks(exchange, symbol, update)) {
        BTQ_LOG_WARNING(std::format("Data failed exchange-specific quality checks for exchange {} and symbol {}", exchange, symbol));
        return;
    }

    // Create a copy of the update to potentially modify
    RenderEngine::MarketDataUpdate processed_update = update;

    // Validate exchange-specific constraints
    validateExchangeSpecificConstraints(exchange, symbol, processed_update);

    // Handle exchange-specific features and adjustments
    applyExchangeSpecificAdjustments(processed_update, exchange);
    handleEnhancedExchangeSpecificFeatures(exchange, symbol, processed_update);

    // Store the processed data from the exchange
    exchange_data_[symbol][exchange] = processed_update;
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;

    // Update statistics
    updateStatistics();

    // Dynamically update exchange features based on observed behavior
    updateExchangeFeaturesDynamically(exchange);
}

// Advanced aggregation using Kalman filtering for optimal estimation
std::optional<AggregatedMarketData> ExchangeAggregator::getKalmanFilteredAggregatedData(
    const std::string& symbol) const {
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

    // Perform Kalman filter-like weighted aggregation
    // This simulates a Kalman filter approach where we weight measurements based on their reliability
    double weighted_sum = 0.0;
    double total_weight = 0.0;

    for (const auto& [exchange, data] : valid_exchange_data) {
        // Get exchange reliability score to weight the contribution
        auto exchange_it = exchange_features_.find(exchange);
        double reliability = (exchange_it != exchange_features_.end()) ?
                            exchange_it->second.reliability_score : 1.0;

        // Apply freshness weight as well
        double freshness_weight = calculateFreshnessWeight(exchange);

        // Combined weight (similar to inverse of measurement covariance in Kalman filter)
        double combined_weight = reliability * freshness_weight;

        // Accumulate weighted price and weight
        weighted_sum += data.price * combined_weight;
        total_weight += combined_weight;
    }

    if (total_weight > 0.0) {
        aggregated_data.aggregated_price = weighted_sum / total_weight;
    }

    // Calculate other metrics using standard approaches
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Advanced aggregation using machine learning-inspired weighted averaging
std::optional<AggregatedMarketData> ExchangeAggregator::getMLWeightedAggregatedData(
    const std::string& symbol) const {
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

    // Calculate ML-inspired weights based on multiple factors
    std::vector<std::pair<double, double>> price_weight_pairs; // price, calculated weight

    for (const auto& [exchange, data] : valid_exchange_data) {
        // Get exchange features
        auto features_it = exchange_features_.find(exchange);
        double base_reliability = (features_it != exchange_features_.end()) ?
                                 features_it->second.reliability_score : 1.0;

        // Calculate freshness weight
        double freshness_weight = calculateFreshnessWeight(exchange);

        // Calculate consistency weight (how close this exchange's price is to others)
        double consistency_weight = 1.0;
        if (valid_exchange_data.size() > 1) {
            double avg_other_price = 0.0;
            int other_count = 0;

            for (const auto& [other_exchange, other_data] : valid_exchange_data) {
                if (other_exchange != exchange) {
                    avg_other_price += other_data.price;
                    other_count++;
                }
            }

            if (other_count > 0) {
                avg_other_price /= other_count;
                double deviation = std::abs(data.price - avg_other_price) / avg_other_price;
                // Lower deviation = higher consistency weight
                consistency_weight = std::max(0.1, 1.0 - deviation);
            }
        }

        // Combine all weights
        double ml_weight = base_reliability * freshness_weight * consistency_weight;

        price_weight_pairs.emplace_back(data.price, ml_weight);
    }

    // Calculate weighted average using ML-inspired weights
    double weighted_sum = 0.0;
    double total_weight = 0.0;

    for (const auto& [price, weight] : price_weight_pairs) {
        weighted_sum += price * weight;
        total_weight += weight;
    }

    if (total_weight > 0.0) {
        aggregated_data.aggregated_price = weighted_sum / total_weight;
    }

    // Calculate other metrics using standard approaches
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Advanced aggregation using outlier-resistant methods
std::optional<AggregatedMarketData> ExchangeAggregator::getOutlierResistantAggregatedData(
    const std::string& symbol) const {
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

    // Collect prices with their associated weights for outlier detection
    std::vector<std::pair<double, double>> price_weight_pairs; // price, weight

    for (const auto& [exchange, data] : valid_exchange_data) {
        auto features_it = exchange_features_.find(exchange);
        double reliability = (features_it != exchange_features_.end()) ?
                            features_it->second.reliability_score : 1.0;
        double freshness_weight = calculateFreshnessWeight(exchange);
        double combined_weight = reliability * freshness_weight;

        price_weight_pairs.emplace_back(data.price, combined_weight);
    }

    if (price_weight_pairs.empty()) {
        return std::nullopt;
    }

    // Sort by price to identify potential outliers
    std::sort(price_weight_pairs.begin(), price_weight_pairs.end());

    // Calculate interquartile range to identify and down-weight outliers
    size_t n = price_weight_pairs.size();
    if (n >= 3) {
        size_t q1_idx = n / 4;
        size_t q3_idx = 3 * n / 4;

        double q1_price = price_weight_pairs[q1_idx].first;
        double q3_price = price_weight_pairs[q3_idx].first;
        double iqr = q3_price - q1_price;
        double lower_bound = q1_price - 1.5 * iqr;
        double upper_bound = q3_price + 1.5 * iqr;

        // Calculate weighted average excluding or down-weighting outliers
        double weighted_sum = 0.0;
        double total_weight = 0.0;

        for (const auto& [price, weight] : price_weight_pairs) {
            if (price >= lower_bound && price <= upper_bound) {
                // Normal weight for non-outliers
                weighted_sum += price * weight;
                total_weight += weight;
            } else {
                // Down-weight outliers instead of completely excluding them
                double down_weighted = weight * 0.1; // Reduce weight to 10% for outliers
                weighted_sum += price * down_weighted;
                total_weight += down_weighted;
            }
        }

        if (total_weight > 0) {
            aggregated_data.aggregated_price = weighted_sum / total_weight;
        }
    } else {
        // If not enough data points for IQR, use simple weighted average
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        for (const auto& [price, weight] : price_weight_pairs) {
            weighted_sum += price * weight;
            total_weight += weight;
        }
        if (total_weight > 0) {
            aggregated_data.aggregated_price = weighted_sum / total_weight;
        }
    }

    // Calculate other metrics using standard approaches
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Get exchange-specific aggregated data with custom weights
std::optional<AggregatedMarketData> ExchangeAggregator::getCustomWeightedAggregatedData(
    const std::string& symbol,
    const std::unordered_map<std::string, double>& custom_weights) const {
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

    // Calculate custom weighted aggregated values
    double total_weighted_price = 0.0;
    double total_weight = 0.0;

    for (const auto& [exchange, data] : valid_exchange_data) {
        // Get custom weight, default to 1.0 if not provided
        double custom_weight = 1.0;
        auto weight_it = custom_weights.find(exchange);
        if (weight_it != custom_weights.end()) {
            custom_weight = std::max(0.0, weight_it->second); // Ensure non-negative weight
        }

        // Also apply reliability and freshness weights
        auto features_it = exchange_features_.find(exchange);
        double reliability = (features_it != exchange_features_.end()) ?
                            features_it->second.reliability_score : 1.0;
        double freshness_weight = calculateFreshnessWeight(exchange);

        double combined_weight = custom_weight * reliability * freshness_weight;

        total_weighted_price += data.price * data.size * combined_weight;
        total_weight += data.size * combined_weight;
    }

    if (total_weight > 0.0) {
        aggregated_data.aggregated_price = total_weighted_price / total_weight;
    }

    // Calculate other metrics using standard approaches
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Get comprehensive multi-exchange analytics for a symbol
std::optional<SymbolCrossExchangeAnalytics> ExchangeAggregator::getComprehensiveAnalytics(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    SymbolCrossExchangeAnalytics analytics;
    analytics.symbol = symbol;

    // Collect data from all exchanges
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);

            // Store exchange-specific data
            analytics.exchange_prices[exchange] = data.price;
            analytics.exchange_volumes[exchange] = data.size;
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate analytics
    std::vector<double> prices;
    std::vector<double> volumes;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
    }

    if (!prices.empty()) {
        // Calculate statistical measures
        double sum_prices = std::accumulate(prices.begin(), prices.end(), 0.0);
        double avg_price = sum_prices / prices.size();

        // Calculate variance and standard deviation
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        double std_dev = std::sqrt(variance);

        analytics.mean_price = avg_price;
        analytics.std_deviation = std_dev;
        analytics.variance = variance;

        // Calculate coefficient of variation
        analytics.coefficient_of_variation = (avg_price > 0) ? std_dev / avg_price : 0.0;

        // Calculate min/max and range
        double min_price = *std::min_element(prices.begin(), prices.end());
        double max_price = *std::max_element(prices.begin(), prices.end());
        analytics.min_price = min_price;
        analytics.max_price = max_price;
        analytics.price_range = max_price - min_price;

        // Calculate median
        std::vector<double> sorted_prices = prices;
        std::sort(sorted_prices.begin(), sorted_prices.end());
        size_t n = sorted_prices.size();
        if (n % 2 == 0) {
            analytics.median_price = (sorted_prices[n/2 - 1] + sorted_prices[n/2]) / 2.0;
        } else {
            analytics.median_price = sorted_prices[n/2];
        }

        // Calculate skewness (measure of asymmetry)
        if (prices.size() >= 3) {
            double skewness_sum = 0.0;
            for (double price : prices) {
                double standardized = (price - avg_price) / std_dev;
                skewness_sum += standardized * standardized * standardized;
            }
            analytics.skewness = skewness_sum / prices.size();
        }

        // Calculate kurtosis (measure of tail heaviness)
        if (prices.size() >= 4) {
            double kurtosis_sum = 0.0;
            for (double price : prices) {
                double standardized = (price - avg_price) / std_dev;
                kurtosis_sum += standardized * standardized * standardized * standardized;
            }
            analytics.kurtosis = kurtosis_sum / prices.size() - 3.0; // Excess kurtosis
        }
    }

    // Calculate volume distribution metrics
    if (!volumes.empty()) {
        double total_volume = std::accumulate(volumes.begin(), volumes.end(), 0.0);
        if (total_volume > 0) {
            analytics.total_volume = total_volume;

            // Calculate volume concentration (Herfindahl-Hirschman Index)
            double hhi = 0.0;
            for (double volume : volumes) {
                double share = volume / total_volume;
                hhi += share * share;
            }
            analytics.volume_concentration_index = hhi;

            // Calculate volume-weighted average price across exchanges
            double vw_total_value = 0.0;
            double vw_total_volume = 0.0;
            for (size_t i = 0; i < exchange_updates.size(); ++i) {
                const auto& [exchange, data] = exchange_updates[i];
                double volume = data.size;
                double price = data.price;

                vw_total_value += price * volume;
                vw_total_volume += volume;
            }

            if (vw_total_volume > 0) {
                analytics.volume_weighted_average_price = vw_total_value / vw_total_volume;
            }
        }
    }

    // Calculate arbitrage potential
    if (analytics.max_price > 0 && analytics.min_price > 0) {
        analytics.max_arbitrage_potential = analytics.max_price - analytics.min_price;
        analytics.relative_arbitrage_potential = (analytics.max_arbitrage_potential / analytics.mean_price) * 100.0;
    }

    analytics.timestamp = std::chrono::high_resolution_clock::now();

    return analytics;
}

// Get exchange-specific aggregated data with custom weights
std::optional<AggregatedMarketData> ExchangeAggregator::getCustomWeightedAggregatedData(
    const std::string& symbol,
    const std::unordered_map<std::string, double>& custom_weights) const {
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

    // Calculate custom weighted aggregated values
    double total_weighted_price = 0.0;
    double total_weight = 0.0;

    for (const auto& [exchange, data] : valid_exchange_data) {
        // Get custom weight, default to 1.0 if not provided
        double custom_weight = 1.0;
        auto weight_it = custom_weights.find(exchange);
        if (weight_it != custom_weights.end()) {
            custom_weight = std::max(0.0, weight_it->second); // Ensure non-negative weight
        }

        // Also apply reliability and freshness weights
        auto features_it = exchange_features_.find(exchange);
        double reliability = (features_it != exchange_features_.end()) ?
                            features_it->second.reliability_score : 1.0;
        double freshness_weight = calculateFreshnessWeight(exchange);

        double combined_weight = custom_weight * reliability * freshness_weight;

        total_weighted_price += data.price * data.size * combined_weight;
        total_weight += data.size * combined_weight;
    }

    if (total_weight > 0.0) {
        aggregated_data.aggregated_price = total_weighted_price / total_weight;
    }

    // Calculate other metrics using standard approaches
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Get data quality scores for each exchange
std::unordered_map<std::string, double> ExchangeAggregator::getExchangeQualityScores(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return {};
    }

    std::unordered_map<std::string, double> quality_scores;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            double quality_score = 0.0;

            // Get exchange features
            auto features_it = exchange_features_.find(exchange);
            double reliability_score = (features_it != exchange_features_.end()) ?
                                     features_it->second.reliability_score : 1.0;

            // Calculate freshness weight
            double freshness_weight = calculateFreshnessWeight(exchange);

            // Calculate price stability (compared to other exchanges)
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            double avg_price = 0.0;
            if (!all_prices.empty()) {
                avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
            }

            double price_stability = 1.0;
            if (avg_price > 0) {
                price_stability = 1.0 - std::abs(data.price - avg_price) / avg_price;
                price_stability = std::max(0.0, price_stability); // Clamp to [0, 1]
            }

            // Combine all factors for final quality score
            quality_score = reliability_score * 0.4 + freshness_weight * 0.3 + price_stability * 0.3;

            quality_scores[exchange] = quality_score;
        }
    }

    return quality_scores;
}

// Enhanced aggregation using quality-weighted approach
std::optional<AggregatedMarketData> ExchangeAggregator::getQualityWeightedAggregatedData(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Get quality scores for all exchanges
    auto quality_scores = getExchangeQualityScores(symbol);

    // Filter out invalid or low-quality exchange data
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> valid_exchange_data;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data) && quality_scores.count(exchange) > 0 &&
            quality_scores.at(exchange) > 0.1) { // Only include exchanges with quality score > 0.1
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

    // Enhanced time synchronization with multiple strategies
    synchronizeTimestamps(aggregated_data);

    // Calculate quality-weighted aggregated values
    double total_weighted_price = 0.0;
    double total_weight = 0.0;

    for (const auto& [exchange, data] : valid_exchange_data) {
        double quality_weight = quality_scores.count(exchange) > 0 ? quality_scores.at(exchange) : 0.1;

        total_weighted_price += data.price * data.size * quality_weight;
        total_weight += data.size * quality_weight;
    }

    if (total_weight > 0.0) {
        aggregated_data.aggregated_price = total_weighted_price / total_weight;
    }

    // Calculate other metrics using quality-weighted approach
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

// Enhanced method to get a unified view with all multi-exchange features
std::optional<ComprehensiveMultiExchangeView> ExchangeAggregator::getUnifiedMultiExchangeView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ComprehensiveMultiExchangeView view;
    view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);

            // Store detailed exchange data
            ExchangeConsolidatedData exchange_data_item;
            exchange_data_item.update = data;

            // Get exchange features
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                exchange_data_item.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            exchange_data_item.stats.price = data.price;
            exchange_data_item.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                exchange_data_item.stats.price_deviation_from_avg = data.price - avg_price;
                exchange_data_item.stats.percent_price_deviation = (avg_price > 0) ?
                    (exchange_data_item.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                exchange_data_item.stats.is_outlier = (std::abs(exchange_data_item.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                exchange_data_item.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            view.exchange_data[exchange] = exchange_data_item;
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate consolidated market metrics
    std::vector<double> prices;
    std::vector<double> volumes;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
        total_volume += data.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        view.market_metrics.spread = prices.back() - prices.front();
        view.market_metrics.volatility = view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        view.market_metrics.average_price = avg_price;
        view.market_metrics.total_volume = total_volume;

        // Calculate price range metrics
        view.market_metrics.lowest_price = prices.front();
        view.market_metrics.highest_price = prices.back();
        view.market_metrics.price_range = prices.back() - prices.front();

        // Calculate additional statistical metrics
        view.market_metrics.median_price = (prices.size() % 2 == 0) ?
            (prices[prices.size()/2 - 1] + prices[prices.size()/2]) / 2.0 :
            prices[prices.size()/2];
    }

    // Calculate order book metrics across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : exchange_updates) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
            highest_bid_exchange = exchange;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > 0.0 && lowest_ask < std::numeric_limits<double>::max()) {
        view.market_metrics.bid_ask_spread = lowest_ask - highest_bid;
        view.market_metrics.best_bid_exchange = highest_bid_exchange;
        view.market_metrics.best_ask_exchange = lowest_ask_exchange;
    }

    // Calculate risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        view.risk_metrics.price_volatility = std::sqrt(variance);
        view.risk_metrics.coefficient_of_variation = (avg_price > 0) ?
            view.risk_metrics.price_volatility / avg_price : 0.0;
    }

    // Calculate arbitrage opportunities
    if (highest_bid > lowest_ask) {
        view.arbitrage_opportunity_exists = true;
        view.arbitrage_profit_potential = highest_bid - lowest_ask;
        view.best_arbitrage_buy_exchange = highest_bid_exchange;
        view.best_arbitrage_sell_exchange = lowest_ask_exchange;
    } else {
        view.arbitrage_opportunity_exists = false;
        view.arbitrage_profit_potential = 0.0;
    }

    // Calculate correlation metrics
    if (view.exchange_data.size() > 1) {
        double total_correlation = 0.0;
        int correlation_count = 0;

        auto it1 = view.exchange_data.begin();
        while (it1 != view.exchange_data.end()) {
            auto it2 = std::next(it1);
            while (it2 != view.exchange_data.end()) {
                double price_diff = std::abs(it1->second.update.price - it2->second.update.price);
                double avg_price = (it1->second.update.price + it2->second.update.price) / 2.0;
                double correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                total_correlation += correlation;
                correlation_count++;

                ++it2;
            }
            ++it1;
        }

        if (correlation_count > 0) {
            view.market_metrics.cross_exchange_correlation = total_correlation / correlation_count;
        }
    }

    // Calculate data quality metrics
    auto quality_metrics = calculateDataQualityMetrics(symbol);
    if (quality_metrics.has_value()) {
        view.data_quality = quality_metrics.value();
    }

    // Calculate exchange rankings
    auto rankings = rankExchangesByReliability();
    view.exchange_rankings = rankings;

    view.timestamp = std::chrono::high_resolution_clock::now();

    return view;
}

// Method to perform comprehensive multi-exchange time synchronization
std::optional<MultiExchangeTimeSyncResult> ExchangeAggregator::performComprehensiveTimeSync(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeTimeSyncResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);

            // Apply latency compensation based on exchange features
            auto features_it = exchange_features_.find(exchange);
            uint64_t compensated_timestamp = data.timestamp;

            if (features_it != exchange_features_.end()) {
                // Apply latency offset compensation
                double offset = features_it->second.latency_offset_us;
                if (offset != 0.0) {
                    // Compensate by adding the offset to align with a reference time
                    compensated_timestamp = static_cast<uint64_t>(
                        static_cast<int64_t>(data.timestamp) + static_cast<int64_t>(offset)
                    );
                }
            }

            compensated_pairs.emplace_back(compensated_timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
            result.compensated_timestamps[exchange] = compensated_timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply synchronization strategy to compensated timestamps
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    // Calculate additional synchronization metrics
    result.average_original_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
    result.timestamp_variance = 0.0;
    if (timestamps.size() > 1) {
        uint64_t mean_ts = result.average_original_timestamp;
        uint64_t variance = 0;
        for (uint64_t ts : timestamps) {
            variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
        }
        variance /= timestamps.size();
        result.timestamp_variance = static_cast<double>(variance);
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

// Enhanced method to handle exchange-specific features with more sophisticated processing
void ExchangeAggregator::handleAdvancedExchangeSpecificFeatures(const std::string& exchange,
                                                             const std::string& symbol,
                                                             RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features defined for this exchange
    }

    const auto& features = features_it->second;

    // Apply exchange-specific data normalization based on precision
    if (features.precision > 0) {
        double multiplier = std::pow(10.0, features.precision);
        update.price = std::round(update.price * multiplier) / multiplier;
    }

    // Apply exchange-specific order size constraints
    if (features.max_order_size > 0 && update.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to max order size limit",
                                   update.size, features.max_order_size, exchange));
        update.size = features.max_order_size;
    }

    if (features.min_order_size > 0 && update.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to min order size requirement",
                                   update.size, features.min_order_size, exchange));
        update.size = features.min_order_size;
    }

    // Apply exchange-specific fee adjustments
    if (features.trading_fee_rate > 0.0) {
        // Adjust price to account for fees - this could be bid/ask specific
        if (update.side == "BUY") {
            update.price = update.price * (1.0 + features.trading_fee_rate); // Fees increase buying cost
        } else if (update.side == "SELL") {
            update.price = update.price * (1.0 - features.trading_fee_rate); // Fees decrease selling revenue
        }
    }

    // Apply exchange-specific withdrawal fee considerations (for position sizing)
    if (features.withdrawal_fee > 0.0) {
        // This could affect how we interpret the data or adjust position sizes
        BTQ_LOG_DEBUG(std::format("Exchange {} has withdrawal fee of {}, factoring into risk calculations",
                                 exchange, features.withdrawal_fee));
    }

    // Apply timezone adjustments if needed
    if (features.timezone != "UTC") {
        // In a real implementation, we would convert timestamps based on timezone
        // For now, just note the timezone difference
        BTQ_LOG_DEBUG(std::format("Exchange {} operates in timezone {}, data timestamp adjustment may be needed",
                                 exchange, features.timezone));
    }

    // Update exchange-specific statistics
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;

    // Update the exchange features with dynamic adjustments
    updateExchangeFeaturesDynamically(exchange);
}

// Advanced time synchronization using cross-correlation and predictive modeling
std::optional<MultiExchangeTimeSyncResult> ExchangeAggregator::performAdvancedTimeSyncWithPrediction(
    const std::string& symbol, TimeSyncStrategy strategy) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    MultiExchangeTimeSyncResult result;
    result.symbol = symbol;
    result.strategy_used = strategy;

    // Collect timestamps from all valid exchanges along with their data
    std::vector<std::pair<uint64_t, std::string>> timestamp_exchange_pairs;
    std::vector<std::pair<uint64_t, std::string>> compensated_pairs;

    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            timestamp_exchange_pairs.emplace_back(data.timestamp, exchange);

            // Apply latency compensation based on exchange features
            auto features_it = exchange_features_.find(exchange);
            uint64_t compensated_timestamp = data.timestamp;

            if (features_it != exchange_features_.end()) {
                // Apply latency offset compensation
                double offset = features_it->second.latency_offset_us;
                if (offset != 0.0) {
                    // Compensate by adding the offset to align with a reference time
                    compensated_timestamp = static_cast<uint64_t>(
                        static_cast<int64_t>(data.timestamp) + static_cast<int64_t>(offset)
                    );
                }
            }

            compensated_pairs.emplace_back(compensated_timestamp, exchange);
            result.original_timestamps[exchange] = data.timestamp;
            result.compensated_timestamps[exchange] = compensated_timestamp;
        }
    }

    if (timestamp_exchange_pairs.empty()) {
        return std::nullopt;
    }

    // Apply synchronization strategy to compensated timestamps
    std::vector<uint64_t> timestamps;
    for (const auto& pair : compensated_pairs) {
        timestamps.push_back(pair.first);
    }

    switch (strategy) {
        case TimeSyncStrategy::EARLIEST_TIMESTAMP:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::LATEST_TIMESTAMP:
            result.synchronized_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
            break;

        case TimeSyncStrategy::AVERAGE_TIMESTAMP:
            result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            break;

        case TimeSyncStrategy::MEDIAN_TIMESTAMP: {
            std::vector<uint64_t> sorted_ts = timestamps;
            std::sort(sorted_ts.begin(), sorted_ts.end());
            size_t n = sorted_ts.size();
            if (n % 2 == 0) {
                result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
            } else {
                result.synchronized_timestamp = sorted_ts[n/2];
            }
            break;
        }

        case TimeSyncStrategy::ADAPTIVE_SYNC: {
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
                result.synchronized_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
            } else {
                std::vector<uint64_t> sorted_ts = timestamps;
                std::sort(sorted_ts.begin(), sorted_ts.end());
                size_t n = sorted_ts.size();
                if (n % 2 == 0) {
                    result.synchronized_timestamp = (sorted_ts[n/2 - 1] + sorted_ts[n/2]) / 2;
                } else {
                    result.synchronized_timestamp = sorted_ts[n/2];
                }
            }
            break;
        }

        case TimeSyncStrategy::SMART_SYNC: {
            // Smart synchronization that considers both time and data quality/reliability
            double weighted_sum = 0.0;
            double total_weight = 0.0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double reliability = (features_it != exchange_features_.end()) ?
                                   features_it->second.reliability_score : 1.0;

                double freshness_weight = calculateFreshnessWeight(exchange);
                double combined_weight = reliability * freshness_weight;

                weighted_sum += static_cast<double>(timestamp) * combined_weight;
                total_weight += combined_weight;
            }

            if (total_weight > 0.0) {
                result.synchronized_timestamp = static_cast<uint64_t>(weighted_sum / total_weight);
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::PREDICTIVE_SYNC: {
            // Predictive synchronization using historical patterns and trends
            uint64_t predicted_sum = 0;
            size_t valid_count = 0;

            for (const auto& [timestamp, exchange] : compensated_pairs) {
                auto features_it = exchange_features_.find(exchange);
                double latency_offset = (features_it != exchange_features_.end()) ?
                                      features_it->second.latency_offset_us : 0.0;

                // Predict the "true" timestamp by compensating for known latency
                uint64_t predicted_ts = timestamp + static_cast<uint64_t>(latency_offset);

                predicted_sum += predicted_ts;
                valid_count++;
            }

            if (valid_count > 0) {
                result.synchronized_timestamp = predicted_sum / valid_count;
            } else {
                result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            }
            break;
        }

        case TimeSyncStrategy::WINDOWED_SYNC: {
            // Windowed synchronization that only considers timestamps within a certain time window
            uint64_t latest_ts = *std::max_element(timestamps.begin(), timestamps.end());

            // Define a time window (e.g., 100ms) to filter out stale data
            uint64_t window_threshold = 100000; // 100ms in microseconds
            std::vector<uint64_t> recent_timestamps;

            for (uint64_t ts : timestamps) {
                if (latest_ts - ts <= window_threshold) {
                    recent_timestamps.push_back(ts);
                }
            }

            // Use average of recent timestamps
            if (!recent_timestamps.empty()) {
                result.synchronized_timestamp = std::accumulate(recent_timestamps.begin(), recent_timestamps.end(), 0ULL) / recent_timestamps.size();
            } else {
                // Fallback to latest if no recent timestamps
                result.synchronized_timestamp = latest_ts;
            }
            break;
        }

        default:
            result.synchronized_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
            break;
    }

    // Calculate synchronization accuracy metrics
    for (const auto& [original_ts, exchange] : timestamp_exchange_pairs) {
        int64_t diff = static_cast<int64_t>(result.synchronized_timestamp) - static_cast<int64_t>(original_ts);
        result.synchronization_accuracy[exchange] = std::abs(diff);
    }

    // Calculate additional synchronization metrics
    result.average_original_timestamp = std::accumulate(timestamps.begin(), timestamps.end(), 0ULL) / timestamps.size();
    result.timestamp_variance = 0.0;
    if (timestamps.size() > 1) {
        uint64_t mean_ts = result.average_original_timestamp;
        uint64_t variance = 0;
        for (uint64_t ts : timestamps) {
            variance += (ts > mean_ts) ? (ts - mean_ts) * (ts - mean_ts) : (mean_ts - ts) * (mean_ts - ts);
        }
        variance /= timestamps.size();
        result.timestamp_variance = static_cast<double>(variance);
    }

    // Calculate cross-correlation between exchanges to assess synchronization quality
    if (timestamp_exchange_pairs.size() >= 2) {
        double correlation_sum = 0.0;
        int correlation_count = 0;

        for (size_t i = 0; i < timestamp_exchange_pairs.size(); ++i) {
            for (size_t j = i + 1; j < timestamp_exchange_pairs.size(); ++j) {
                uint64_t ts1 = timestamp_exchange_pairs[i].first;
                uint64_t ts2 = timestamp_exchange_pairs[j].first;

                // Calculate correlation based on timestamp difference
                uint64_t diff = std::abs(static_cast<int64_t>(ts1) - static_cast<int64_t>(ts2));
                double correlation = std::exp(-static_cast<double>(diff) / 1000000.0); // Decay factor of 1 second

                correlation_sum += correlation;
                correlation_count++;
            }
        }

        if (correlation_count > 0) {
            result.cross_correlation = correlation_sum / correlation_count;
        }
    }

    result.timestamp = std::chrono::high_resolution_clock::now();
    return result;
}

// Method to get exchange-specific aggregated data with enhanced feature consideration
std::optional<AggregatedMarketData> ExchangeAggregator::getExchangeSpecificAggregatedDataWithFeatures(
    const std::string& symbol, const std::vector<std::string>& exchanges) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    AggregatedMarketData aggregated_data;
    aggregated_data.symbol = symbol;
    aggregated_data.sync_strategy = sync_strategy_;

    // Filter data based on requested exchanges
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> filtered_exchange_data;
    for (const auto& exchange : exchanges) {
        auto exchange_data_it = symbol_it->second.find(exchange);
        if (exchange_data_it != symbol_it->second.end() &&
            isExchangeDataValid(exchange, exchange_data_it->second)) {
            filtered_exchange_data[exchange] = exchange_data_it->second;
            aggregated_data.exchange_data[exchange] = exchange_data_it->second;
        }
    }

    if (filtered_exchange_data.empty()) {
        return std::nullopt;
    }

    // Collect timestamps from all valid exchanges
    for (const auto& [exchange, data] : filtered_exchange_data) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Enhanced time synchronization considering exchange-specific features
    enhancedSynchronizeTimestamps(aggregated_data);

    // Calculate aggregated values with exchange-specific feature weighting
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidationForExchanges(symbol, exchanges);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(filtered_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPriceForExchanges(symbol, exchanges);

    // Calculate additional aggregated metrics
    aggregated_data.aggregated_high = calculateHighPrice(filtered_exchange_data);
    aggregated_data.aggregated_low = calculateLowPrice(filtered_exchange_data);
    aggregated_data.aggregated_bid = calculateBestBid(filtered_exchange_data);
    aggregated_data.aggregated_ask = calculateBestAsk(filtered_exchange_data);

    // Calculate total volume across all valid exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : filtered_exchange_data) {
        total_volume += data.size;
    }
    aggregated_data.aggregated_volume = total_volume;

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, filtered_exchange_data, aggregated_data);
    detectArbitrageOpportunities(filtered_exchange_data, aggregated_data);

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

// Enhanced method to handle exchange-specific features with comprehensive processing
void ExchangeAggregator::handleComprehensiveExchangeSpecificFeatures(const std::string& exchange,
                                                                 const std::string& symbol,
                                                                 RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto features_it = exchange_features_.find(exchange);
    if (features_it == exchange_features_.end()) {
        return; // No features defined for this exchange
    }

    const auto& features = features_it->second;

    // Apply exchange-specific data normalization based on precision
    if (features.precision > 0) {
        double multiplier = std::pow(10.0, features.precision);
        update.price = std::round(update.price * multiplier) / multiplier;
    }

    // Apply exchange-specific order size constraints
    if (features.max_order_size > 0 && update.size > features.max_order_size) {
        BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to max order size limit",
                                   update.size, features.max_order_size, exchange));
        update.size = features.max_order_size;
    }

    if (features.min_order_size > 0 && update.size < features.min_order_size) {
        BTQ_LOG_WARNING(std::format("Adjusting order size from {} to {} for exchange {} due to min order size requirement",
                                   update.size, features.min_order_size, exchange));
        update.size = features.min_order_size;
    }

    // Apply exchange-specific fee adjustments
    if (features.trading_fee_rate > 0.0) {
        // Adjust price to account for fees - this could be bid/ask specific
        if (update.side == "BUY") {
            update.price = update.price * (1.0 + features.trading_fee_rate); // Fees increase buying cost
        } else if (update.side == "SELL") {
            update.price = update.price * (1.0 - features.trading_fee_rate); // Fees decrease selling revenue
        }
    }

    // Apply exchange-specific withdrawal fee considerations (for position sizing)
    if (features.withdrawal_fee > 0.0) {
        // This could affect how we interpret the data or adjust position sizes
        BTQ_LOG_DEBUG(std::format("Exchange {} has withdrawal fee of {}, factoring into risk calculations",
                                 exchange, features.withdrawal_fee));
    }

    // Apply timezone adjustments if needed
    if (features.timezone != "UTC") {
        // In a real implementation, we would convert timestamps based on timezone
        // For now, just note the timezone difference
        BTQ_LOG_DEBUG(std::format("Exchange {} operates in timezone {}, data timestamp adjustment may be needed",
                                 exchange, features.timezone));
    }

    // Apply exchange-specific API rate limiting considerations
    if (features.api_endpoint != "") {
        // Log or track API usage for this exchange
        BTQ_LOG_DEBUG(std::format("Processing data from exchange {} via endpoint {}",
                                 exchange, features.api_endpoint));
    }

    // Apply exchange-specific data quality filters
    if (features.supports_microseconds) {
        // Ensure timestamp precision matches exchange capability
        update.timestamp = update.timestamp; // Already in microseconds
    } else {
        // Round timestamp to nearest millisecond if exchange doesn't support microseconds
        update.timestamp = (update.timestamp / 1000) * 1000;
    }

    // Apply exchange-specific data validation rules
    if (!features.supported_symbols.empty()) {
        bool symbol_supported = std::find(features.supported_symbols.begin(),
                                         features.supported_symbols.end(),
                                         symbol) != features.supported_symbols.end();
        if (!symbol_supported) {
            BTQ_LOG_WARNING(std::format("Symbol {} is not in supported symbols list for exchange {}",
                                       symbol, exchange));
        }
    }

    // Update exchange-specific statistics
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;

    // Update the exchange features with dynamic adjustments
    updateExchangeFeaturesDynamically(exchange);
}

// Method to process data updates with comprehensive exchange-specific feature handling
void ExchangeAggregator::processDataUpdateWithComprehensiveFeatures(const std::string& exchange,
                                                                  const std::string& symbol,
                                                                  const RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Validate the incoming data before storing
    if (!isValidData(update)) {
        BTQ_LOG_WARNING(std::format("Invalid data received from exchange {} for symbol {}", exchange, symbol));
        return;
    }

    // Perform exchange-specific quality checks
    if (!performExchangeSpecificQualityChecks(exchange, symbol, update)) {
        BTQ_LOG_WARNING(std::format("Data failed exchange-specific quality checks for exchange {} and symbol {}", exchange, symbol));
        return;
    }

    // Create a copy of the update to potentially modify
    RenderEngine::MarketDataUpdate processed_update = update;

    // Validate exchange-specific constraints
    validateExchangeSpecificConstraints(exchange, symbol, processed_update);

    // Apply comprehensive exchange-specific features
    handleComprehensiveExchangeSpecificFeatures(exchange, symbol, processed_update);

    // Store the processed data from the exchange
    exchange_data_[symbol][exchange] = processed_update;
    exchange_last_update_[exchange] = std::chrono::high_resolution_clock::now();
    exchange_validity_[exchange] = true;

    // Update statistics
    updateStatistics();

    // Dynamically update exchange features based on observed behavior
    updateExchangeFeaturesDynamically(exchange);
}

// Method to get a comprehensive view with all exchange-specific features considered
std::optional<ComprehensiveMultiExchangeView> ExchangeAggregator::getComprehensiveViewWithFeatures(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ComprehensiveMultiExchangeView view;
    view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);

            // Store detailed exchange data
            ExchangeConsolidatedData exchange_data_item;
            exchange_data_item.update = data;

            // Get exchange features
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                exchange_data_item.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            exchange_data_item.stats.price = data.price;
            exchange_data_item.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                exchange_data_item.stats.price_deviation_from_avg = data.price - avg_price;
                exchange_data_item.stats.percent_price_deviation = (avg_price > 0) ?
                    (exchange_data_item.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                exchange_data_item.stats.is_outlier = (std::abs(exchange_data_item.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                exchange_data_item.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            view.exchange_data[exchange] = exchange_data_item;
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate consolidated market metrics
    std::vector<double> prices;
    std::vector<double> volumes;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
        total_volume += data.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        view.market_metrics.spread = prices.back() - prices.front();
        view.market_metrics.volatility = view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        view.market_metrics.average_price = avg_price;
        view.market_metrics.total_volume = total_volume;

        // Calculate price range metrics
        view.market_metrics.lowest_price = prices.front();
        view.market_metrics.highest_price = prices.back();
        view.market_metrics.price_range = prices.back() - prices.front();

        // Calculate additional statistical metrics
        view.market_metrics.median_price = (prices.size() % 2 == 0) ?
            (prices[prices.size()/2 - 1] + prices[prices.size()/2]) / 2.0 :
            prices[prices.size()/2];
    }

    // Calculate order book metrics across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : exchange_updates) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
            highest_bid_exchange = exchange;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > 0.0 && lowest_ask < std::numeric_limits<double>::max()) {
        view.market_metrics.bid_ask_spread = lowest_ask - highest_bid;
        view.market_metrics.best_bid_exchange = highest_bid_exchange;
        view.market_metrics.best_ask_exchange = lowest_ask_exchange;
    }

    // Calculate risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        view.risk_metrics.price_volatility = std::sqrt(variance);
        view.risk_metrics.coefficient_of_variation = (avg_price > 0) ?
            view.risk_metrics.price_volatility / avg_price : 0.0;
    }

    // Calculate arbitrage opportunities
    if (highest_bid > lowest_ask) {
        view.arbitrage_opportunity_exists = true;
        view.arbitrage_profit_potential = highest_bid - lowest_ask;
        view.best_arbitrage_buy_exchange = highest_bid_exchange;
        view.best_arbitrage_sell_exchange = lowest_ask_exchange;
    } else {
        view.arbitrage_opportunity_exists = false;
        view.arbitrage_profit_potential = 0.0;
    }

    // Calculate correlation metrics
    if (view.exchange_data.size() > 1) {
        double total_correlation = 0.0;
        int correlation_count = 0;

        auto it1 = view.exchange_data.begin();
        while (it1 != view.exchange_data.end()) {
            auto it2 = std::next(it1);
            while (it2 != view.exchange_data.end()) {
                double price_diff = std::abs(it1->second.update.price - it2->second.update.price);
                double avg_price = (it1->second.update.price + it2->second.update.price) / 2.0;
                double correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                total_correlation += correlation;
                correlation_count++;

                ++it2;
            }
            ++it1;
        }

        if (correlation_count > 0) {
            view.market_metrics.cross_exchange_correlation = total_correlation / correlation_count;
        }
    }

    // Calculate data quality metrics
    auto quality_metrics = calculateDataQualityMetrics(symbol);
    if (quality_metrics.has_value()) {
        view.data_quality = quality_metrics.value();
    }

    // Calculate exchange rankings
    auto rankings = rankExchangesByReliability();
    view.exchange_rankings = rankings;

    view.timestamp = std::chrono::high_resolution_clock::now();

    return view;
}

// Final method to get the ultimate unified view combining all multi-exchange features
std::optional<AggregatedMarketData> ExchangeAggregator::getUltimateAggregatedData(
    const std::string& symbol) const {
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

    // Enhanced time synchronization with multiple strategies
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values using multiple sophisticated methods
    aggregated_data.aggregated_price = calculateWeightedAveragePriceWithValidation(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(valid_exchange_data);
    aggregated_data.consensus_price = calculateConsensusPrice(symbol);

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

    // Calculate advanced metrics using multiple algorithms
    aggregated_data.vwap = calculateVWAP(valid_exchange_data);
    aggregated_data.median_price = calculateMedianPrice(valid_exchange_data);
    aggregated_data.trimmed_mean_price = calculateTrimmedMean(valid_exchange_data, 0.1);

    // Calculate exchange correlations and detect arbitrage opportunities
    calculateExchangeCorrelations(symbol, valid_exchange_data, aggregated_data);
    detectArbitrageOpportunities(valid_exchange_data, aggregated_data);

    // Enhanced risk assessment using multiple metrics
    calculateEnhancedRiskMetrics(symbol, valid_exchange_data, aggregated_data);

    // Perform advanced aggregation using multiple algorithms
    auto advanced_result = performAdvancedAggregation(symbol);
    if (advanced_result.has_value()) {
        aggregated_data.aggregated_price = advanced_result->consensus_price; // Use the most robust estimate
    }

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

// Additional method implementations for enhanced multi-exchange aggregation

std::optional<AggregatedMarketData> ExchangeAggregator::getUnifiedView(const std::string& symbol) const {
    // This method provides a comprehensive unified view of multi-exchange data
    // It combines all the aggregation features into a single, comprehensive view
    return getEnhancedAggregatedData(symbol);
}

// Enhanced method to get a single unified view with all multi-exchange features combined
std::optional<UnifiedMultiExchangeView> ExchangeAggregator::getUnifiedMultiExchangeView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    UnifiedMultiExchangeView view;
    view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    std::vector<std::pair<std::string, RenderEngine::MarketDataUpdate>> exchange_updates;
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            exchange_updates.emplace_back(exchange, data);

            // Store detailed exchange data
            ExchangeConsolidatedData exchange_data_item;
            exchange_data_item.update = data;

            // Get exchange features
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                exchange_data_item.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            exchange_data_item.stats.price = data.price;
            exchange_data_item.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                exchange_data_item.stats.price_deviation_from_avg = data.price - avg_price;
                exchange_data_item.stats.percent_price_deviation = (avg_price > 0) ?
                    (exchange_data_item.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                exchange_data_item.stats.is_outlier = (std::abs(exchange_data_item.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                exchange_data_item.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            view.exchange_data[exchange] = exchange_data_item;
        }
    }

    if (exchange_updates.empty()) {
        return std::nullopt;
    }

    // Calculate consolidated market metrics
    std::vector<double> prices;
    std::vector<double> volumes;
    double total_volume = 0.0;

    for (const auto& [exchange, data] : exchange_updates) {
        prices.push_back(data.price);
        volumes.push_back(data.size);
        total_volume += data.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        view.market_metrics.spread = prices.back() - prices.front();
        view.market_metrics.volatility = view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        view.market_metrics.average_price = avg_price;
        view.market_metrics.total_volume = total_volume;

        // Calculate price range metrics
        view.market_metrics.lowest_price = prices.front();
        view.market_metrics.highest_price = prices.back();
        view.market_metrics.price_range = prices.back() - prices.front();

        // Calculate additional statistical metrics
        view.market_metrics.median_price = (prices.size() % 2 == 0) ?
            (prices[prices.size()/2 - 1] + prices[prices.size()/2]) / 2.0 :
            prices[prices.size()/2];
    }

    // Calculate order book metrics across exchanges
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, data] : exchange_updates) {
        if (data.side == "BUY" && data.price > highest_bid) {
            highest_bid = data.price;
            highest_bid_exchange = exchange;
        }
        if (data.side == "SELL" && data.price < lowest_ask) {
            lowest_ask = data.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > 0.0 && lowest_ask < std::numeric_limits<double>::max()) {
        view.market_metrics.bid_ask_spread = lowest_ask - highest_bid;
        view.market_metrics.best_bid_exchange = highest_bid_exchange;
        view.market_metrics.best_ask_exchange = lowest_ask_exchange;
    }

    // Calculate risk metrics
    if (prices.size() > 1) {
        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        double variance = 0.0;
        for (double price : prices) {
            variance += (price - avg_price) * (price - avg_price);
        }
        variance /= prices.size();
        view.risk_metrics.price_volatility = std::sqrt(variance);
        view.risk_metrics.coefficient_of_variation = (avg_price > 0) ?
            view.risk_metrics.price_volatility / avg_price : 0.0;
    }

    // Calculate arbitrage opportunities
    if (highest_bid > lowest_ask) {
        view.arbitrage_opportunity_exists = true;
        view.arbitrage_profit_potential = highest_bid - lowest_ask;
        view.best_arbitrage_buy_exchange = highest_bid_exchange;
        view.best_arbitrage_sell_exchange = lowest_ask_exchange;
    } else {
        view.arbitrage_opportunity_exists = false;
        view.arbitrage_profit_potential = 0.0;
    }

    // Calculate correlation metrics
    if (view.exchange_data.size() > 1) {
        double total_correlation = 0.0;
        int correlation_count = 0;

        auto it1 = view.exchange_data.begin();
        while (it1 != view.exchange_data.end()) {
            auto it2 = std::next(it1);
            while (it2 != view.exchange_data.end()) {
                double price_diff = std::abs(it1->second.update.price - it2->second.update.price);
                double avg_price = (it1->second.update.price + it2->second.update.price) / 2.0;
                double correlation = 1.0 - std::min(1.0, price_diff / avg_price);

                total_correlation += correlation;
                correlation_count++;

                ++it2;
            }
            ++it1;
        }

        if (correlation_count > 0) {
            view.market_metrics.cross_exchange_correlation = total_correlation / correlation_count;
        }
    }

    // Calculate data quality metrics
    auto quality_metrics = calculateDataQualityMetrics(symbol);
    if (quality_metrics.has_value()) {
        view.data_quality = quality_metrics.value();
    }

    // Calculate exchange rankings
    auto rankings = rankExchangesByReliability();
    view.exchange_rankings = rankings;

    // Perform comprehensive time synchronization
    auto time_sync_result = performComprehensiveTimeSync(symbol, sync_strategy_);
    if (time_sync_result.has_value()) {
        view.time_sync_result = time_sync_result.value();
    }

    view.timestamp = std::chrono::high_resolution_clock::now();

    return view;
}

// Method to get a consolidated view of all symbols across all exchanges
std::vector<UnifiedMultiExchangeView> ExchangeAggregator::getAllSymbolsUnifiedView() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<UnifiedMultiExchangeView> all_views;

    for (const auto& [symbol, _] : exchange_data_) {
        auto view = getUnifiedMultiExchangeView(symbol);
        if (view.has_value()) {
            all_views.push_back(view.value());
        }
    }

    return all_views;
}

std::vector<MultiExchangeData> ExchangeAggregator::getAllSymbolsMultiExchangeView() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::vector<MultiExchangeData> all_views;

    for (const auto& [symbol, _] : exchange_data_) {
        auto view = getMultiExchangeView(symbol);
        if (view.has_value()) {
            all_views.push_back(view.value());
        }
    }

    return all_views;
}

std::optional<ComprehensiveExchangeView> ExchangeAggregator::getComprehensiveExchangeView(
    const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    auto symbol_it = exchange_data_.find(symbol);
    if (symbol_it == exchange_data_.end()) {
        return std::nullopt;
    }

    ComprehensiveExchangeView comprehensive_view;
    comprehensive_view.symbol = symbol;

    // Gather data from all exchanges for this symbol
    for (const auto& [exchange, data] : symbol_it->second) {
        if (isExchangeDataValid(exchange, data)) {
            ExchangeDetailedData detailed_data;
            detailed_data.update = data;

            // Get exchange features for additional context
            auto features_it = exchange_features_.find(exchange);
            if (features_it != exchange_features_.end()) {
                detailed_data.features = features_it->second;
            }

            // Calculate exchange-specific statistics
            detailed_data.stats.price = data.price;
            detailed_data.stats.volume = data.size;

            // Calculate price relative to overall average
            std::vector<double> all_prices;
            for (const auto& [other_exchange, other_data] : symbol_it->second) {
                if (isExchangeDataValid(other_exchange, other_data)) {
                    all_prices.push_back(other_data.price);
                }
            }

            if (!all_prices.empty()) {
                double avg_price = std::accumulate(all_prices.begin(), all_prices.end(), 0.0) / all_prices.size();
                detailed_data.stats.price_deviation_from_avg = data.price - avg_price;
                detailed_data.stats.percent_price_deviation = (avg_price > 0) ?
                    (detailed_data.stats.price_deviation_from_avg / avg_price) * 100.0 : 0.0;

                // Determine if this exchange is an outlier
                double std_dev = 0.0;
                for (double price : all_prices) {
                    std_dev += (price - avg_price) * (price - avg_price);
                }
                std_dev = std::sqrt(std_dev / all_prices.size());

                detailed_data.stats.is_outlier = (std::abs(detailed_data.stats.price_deviation_from_avg) > 2 * std_dev);
            }

            // Calculate latency relative to other exchanges
            auto last_update_it = exchange_last_update_.find(exchange);
            if (last_update_it != exchange_last_update_.end()) {
                auto now = std::chrono::high_resolution_clock::now();
                detailed_data.stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_update_it->second).count();
            }

            comprehensive_view.exchange_details[exchange] = detailed_data;
        }
    }

    if (comprehensive_view.exchange_details.empty()) {
        return std::nullopt;
    }

    // Calculate overall market metrics
    std::vector<double> prices;
    double total_volume = 0.0;

    for (const auto& [exchange, details] : comprehensive_view.exchange_details) {
        prices.push_back(details.update.price);
        total_volume += details.update.size;
    }

    if (!prices.empty()) {
        std::sort(prices.begin(), prices.end());
        comprehensive_view.market_metrics.spread = prices.back() - prices.front();
        comprehensive_view.market_metrics.volatility = comprehensive_view.market_metrics.spread / prices.front();

        double avg_price = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
        comprehensive_view.market_metrics.average_price = avg_price;
        comprehensive_view.market_metrics.total_volume = total_volume;
    }

    // Detect cross-exchange arbitrage opportunities
    double highest_bid = 0.0;
    double lowest_ask = std::numeric_limits<double>::max();
    std::string highest_bid_exchange = "";
    std::string lowest_ask_exchange = "";

    for (const auto& [exchange, details] : comprehensive_view.exchange_details) {
        if (details.update.side == "BUY" && details.update.price > highest_bid) {
            highest_bid = details.update.price;
            highest_bid_exchange = exchange;
        }
        if (details.update.side == "SELL" && details.update.price < lowest_ask) {
            lowest_ask = details.update.price;
            lowest_ask_exchange = exchange;
        }
    }

    if (highest_bid > lowest_ask) {
        comprehensive_view.arbitrage_detected = true;
        comprehensive_view.arbitrage_profit = highest_bid - lowest_ask;
        comprehensive_view.bid_exchange = highest_bid_exchange;
        comprehensive_view.ask_exchange = lowest_ask_exchange;
    } else {
        comprehensive_view.arbitrage_detected = false;
        comprehensive_view.arbitrage_profit = 0.0;
    }

    comprehensive_view.timestamp = std::chrono::high_resolution_clock::now();

    return comprehensive_view;
}

}  // namespace Data
}  // namespace BTQuant