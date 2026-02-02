#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "symbol_manager.hpp"
#include "unified_data_pipeline.hpp"
#include "data_types.hpp"

namespace BTQuant {
namespace Data {

// Time synchronization strategies for multi-exchange data
enum class TimeSyncStrategy {
  EARLIEST_TIMESTAMP,    // Use earliest timestamp among exchanges
  LATEST_TIMESTAMP,      // Use latest timestamp among exchanges
  AVERAGE_TIMESTAMP,     // Average timestamps from all exchanges
  REFERENCE_EXCHANGE,    // Use one exchange as time reference
  OFFSET_COMPENSATION,   // Apply calculated offsets to align times
  MEDIAN_TIMESTAMP       // Use median timestamp among exchanges
};

// Exchange-specific features and configurations
struct ExchangeFeatures {
  std::string exchange_name;
  double latency_offset_us = 0.0;        // Latency offset in microseconds
  bool supports_microseconds = true;     // Whether exchange provides microsecond precision
  std::vector<std::string> supported_symbols;
  std::vector<std::string> supported_data_types;
  double reliability_score = 1.0;        // 0.0 to 1.0, where 1.0 is most reliable
  std::string api_endpoint;
};

// Aggregated market data combining multiple exchanges
struct AggregatedMarketData {
  std::string symbol;
  TimeSyncStrategy sync_strategy = TimeSyncStrategy::EARLIEST_TIMESTAMP;

  // Data from different exchanges
  std::unordered_map<std::string, RenderEngine::MarketDataUpdate> exchange_data;

  // Aggregated values
  double aggregated_price = 0.0;
  double aggregated_volume = 0.0;
  double weighted_price = 0.0;
  double aggregated_high = 0.0;          // Highest price among exchanges
  double aggregated_low = 0.0;           // Lowest price among exchanges
  double aggregated_bid = 0.0;           // Best bid price among exchanges
  double aggregated_ask = 0.0;           // Best ask price among exchanges
  uint64_t synchronized_timestamp = 0;

  // Time synchronization info
  std::map<std::string, uint64_t> exchange_timestamps;
  uint64_t reference_timestamp = 0;

  std::chrono::high_resolution_clock::time_point last_updated;
};

// Exchange Aggregator for combining data from multiple exchanges
class ExchangeAggregator {
 public:
  ExchangeAggregator(std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                     std::shared_ptr<RenderEngine::SymbolManager> symbol_manager);
  ~ExchangeAggregator();

  // Initialize the aggregator
  bool initialize();

  // Add an exchange to the aggregation pool
  void addExchange(const std::string& exchange_name, const ExchangeFeatures& features);

  // Remove an exchange from the aggregation pool
  void removeExchange(const std::string& exchange_name);

  // Get available exchanges in the aggregation pool
  std::vector<std::string> getAvailableExchanges() const;

  // Set time synchronization strategy
  void setTimeSyncStrategy(TimeSyncStrategy strategy);

  // Aggregate data for a specific symbol across all exchanges
  std::optional<AggregatedMarketData> aggregateSymbolData(const std::string& symbol);

  // Process incoming data from an exchange
  void processDataUpdate(const std::string& exchange, const std::string& symbol,
                         const RenderEngine::MarketDataUpdate& update);

  // Get aggregated data for a symbol
  std::optional<AggregatedMarketData> getAggregatedData(const std::string& symbol) const;

  // Calculate weighted average price based on volume from different exchanges
  double calculateWeightedAveragePrice(const std::string& symbol) const;

  // Calculate weighted average price with validation of data quality
  double calculateWeightedAveragePriceWithValidation(const std::string& symbol) const;

  // Calculate synchronized timestamp based on strategy
  uint64_t calculateSynchronizedTimestamp(const std::string& symbol) const;

  // Get exchange-specific features
  std::optional<ExchangeFeatures> getExchangeFeatures(const std::string& exchange) const;

  // Update exchange features
  void updateExchangeFeatures(const std::string& exchange, const ExchangeFeatures& features);

  // Get statistics about aggregation performance
  struct AggregationStats {
    size_t total_symbols_aggregated = 0;
    size_t total_exchanges = 0;
    size_t valid_exchanges = 0;          // Number of exchanges with valid data
    double avg_latency_difference_us = 0.0;
    std::chrono::high_resolution_clock::time_point last_update;
  };

  AggregationStats getStats() const;

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<RenderEngine::SymbolManager> symbol_manager_;

  mutable std::mutex data_mutex_;
  std::unordered_map<std::string, std::unordered_map<std::string, RenderEngine::MarketDataUpdate>> exchange_data_;
  std::unordered_map<std::string, ExchangeFeatures> exchange_features_;

  // Additional data structures for enhanced functionality
  std::unordered_map<std::string, bool> exchange_validity_;              // Track validity of each exchange
  std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> exchange_last_update_;  // Track last update time

  TimeSyncStrategy sync_strategy_ = TimeSyncStrategy::EARLIEST_TIMESTAMP;
  AggregationStats stats_;

  // Thread for continuous aggregation
  std::atomic<bool> running_{false};
  std::thread aggregation_thread_;

  void aggregationLoop();
  void synchronizeTimestamps(AggregatedMarketData& data) const;
  double calculateVolumeWeightedPrice(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  void updateStatistics();

  // Validation and quality control methods
  bool isExchangeDataValid(const std::string& exchange, const RenderEngine::MarketDataUpdate& data) const;
  bool isExchangeValid(const std::string& exchange) const;
  bool isValidData(const RenderEngine::MarketDataUpdate& data) const;
  double calculateFreshnessWeight(const std::string& exchange) const;
  void checkStaleData();

  // Enhanced aggregation methods
  double calculateHighPrice(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateLowPrice(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateBestBid(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateBestAsk(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
};

}  // namespace Data
}  // namespace BTQuant