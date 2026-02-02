#include "../include/data/exchange_aggregator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <format>

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
    
    BTQ_LOG_INFO(std::format("Added exchange {} to aggregation pool", exchange_name));
}

void ExchangeAggregator::removeExchange(const std::string& exchange_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    exchange_features_.erase(exchange_name);

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
    aggregated_data.exchange_data.clear(); // We'll populate this with actual MarketDataUpdate objects

    // Collect timestamps from all exchanges
    for (const auto& [exchange, data] : symbol_it->second) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values
    aggregated_data.aggregated_price = calculateWeightedAveragePrice(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(symbol_it->second);

    // Calculate total volume across all exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : symbol_it->second) {
        total_volume += data.size;  // Assuming size represents volume
    }
    aggregated_data.aggregated_volume = total_volume;

    aggregated_data.last_updated = std::chrono::high_resolution_clock::now();

    return aggregated_data;
}

void ExchangeAggregator::processDataUpdate(const std::string& exchange, const std::string& symbol,
                                          const RenderEngine::MarketDataUpdate& update) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Store the raw data from the exchange
    exchange_data_[symbol][exchange] = update;

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
    aggregated_data.exchange_data = symbol_it->second;  // This assigns MarketDataUpdate objects to exchange_data

    // Collect timestamps from all exchanges
    for (const auto& [exchange, data] : symbol_it->second) {
        aggregated_data.exchange_timestamps[exchange] = data.timestamp;
    }

    // Synchronize timestamps based on strategy
    synchronizeTimestamps(aggregated_data);

    // Calculate aggregated values
    aggregated_data.aggregated_price = calculateWeightedAveragePrice(symbol);
    aggregated_data.weighted_price = calculateVolumeWeightedPrice(symbol_it->second);

    // Calculate total volume across all exchanges
    double total_volume = 0.0;
    for (const auto& [exchange, data] : symbol_it->second) {
        total_volume += data.size;  // Assuming size represents volume
    }
    aggregated_data.aggregated_volume = total_volume;

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
        timestamps.push_back(data.timestamp);
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
        timestamps.push_back(ts);
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
            offsets.push_back(features.latency_offset_us);
        }

        if (!offsets.empty()) {
            double sum = std::accumulate(offsets.begin(), offsets.end(), 0.0);
            stats_.avg_latency_difference_us = sum / offsets.size();
        }
    }
}

}  // namespace Data
}  // namespace BTQuant