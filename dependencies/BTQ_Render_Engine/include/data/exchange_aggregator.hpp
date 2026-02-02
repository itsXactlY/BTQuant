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
  MEDIAN_TIMESTAMP,      // Use median timestamp among exchanges
  ADAPTIVE_SYNC,         // Adaptive synchronization based on market conditions
  SMART_SYNC,            // Smart synchronization considering reliability and freshness
  PREDICTIVE_SYNC,       // Predictive synchronization using historical patterns
  WINDOWED_SYNC          // Windowed synchronization considering only recent data
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
  double trading_fee_rate = 0.001;       // Trading fee rate (0.1% default)
  double withdrawal_fee = 0.0;           // Withdrawal fee in base currency
  int max_order_size = 1000000;          // Maximum order size allowed
  int min_order_size = 1;                // Minimum order size allowed
  std::string timezone = "UTC";          // Exchange timezone
  bool is_active = true;                 // Whether the exchange is currently active
};

// Exchange ranking for reliability assessment
struct ExchangeRanking {
  std::string exchange_name;
  double reliability_score = 0.0;
  bool is_active = false;
  bool is_valid = false;
  int64_t data_staleness_ms = 0;
};

// Exchange-specific statistics for detailed analysis
struct ExchangeSpecificStats {
  double price = 0.0;
  double volume = 0.0;
  double price_deviation_from_avg = 0.0;  // Difference from overall average price
  double percent_price_deviation = 0.0;   // Percentage deviation from average
  double latency_ms = 0.0;                // Latency compared to other exchanges
  bool is_outlier = false;                // Whether this exchange's data is an outlier
};

// Multi-exchange data view for comprehensive analysis
struct MultiExchangeData {
  std::string symbol;
  std::unordered_map<std::string, RenderEngine::MarketDataUpdate> exchange_data;
  std::unordered_map<std::string, ExchangeFeatures> exchange_features;
  std::unordered_map<std::string, ExchangeSpecificStats> exchange_stats;  // Detailed stats per exchange

  // Calculated metrics across exchanges
  double spread = 0.0;                    // Difference between highest bid and lowest ask
  double highest_bid = 0.0;               // Highest bid across all exchanges
  double lowest_ask = 0.0;                // Lowest ask across all exchanges
  double price_volatility = 0.0;          // Price variation across exchanges
  double price_std_deviation = 0.0;       // Standard deviation of prices across exchanges

  std::chrono::high_resolution_clock::time_point timestamp;
};

// Detailed data for a single exchange
struct ExchangeDetailedData {
  RenderEngine::MarketDataUpdate update;
  ExchangeFeatures features;
  ExchangeSpecificStats stats;
};

// Market metrics across all exchanges
struct MarketMetrics {
  double average_price = 0.0;
  double spread = 0.0;                    // Difference between highest and lowest prices
  double volatility = 0.0;                // Normalized price variation
  double total_volume = 0.0;              // Total volume across all exchanges
  double correlation_coefficient = 0.0;   // Correlation between exchanges
};

// Comprehensive view of all exchanges for a symbol
struct ComprehensiveExchangeView {
  std::string symbol;
  std::unordered_map<std::string, ExchangeDetailedData> exchange_details;  // Detailed data per exchange
  MarketMetrics market_metrics;                                           // Overall market metrics

  // Arbitrage detection
  bool arbitrage_detected = false;
  double arbitrage_profit = 0.0;
  std::string bid_exchange = "";
  std::string ask_exchange = "";

  std::chrono::high_resolution_clock::time_point timestamp;
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
  double consensus_price = 0.0;          // Consensus price using weighted median
  uint64_t synchronized_timestamp = 0;

  // Advanced aggregation metrics
  double vwap = 0.0;                     // Volume Weighted Average Price
  double median_price = 0.0;             // Median price across exchanges
  double trimmed_mean_price = 0.0;       // Trimmed mean to reduce outlier impact

  // Time synchronization info
  std::map<std::string, uint64_t> exchange_timestamps;
  uint64_t reference_timestamp = 0;

  // Exchange correlation data
  std::unordered_map<std::string, double> exchange_correlations;  // Correlation of each exchange to the aggregated price
  double overall_correlation = 0.0;                              // Overall correlation among exchanges

  // Arbitrage detection
  bool arbitrage_opportunity = false;
  double arbitrage_profit = 0.0;
  std::string bid_exchange = "";
  std::string ask_exchange = "";

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
  std::unordered_map<std::string, std::unordered_map<std::string, double>> exchange_correlations_;  // Track correlations between exchanges

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

  // Advanced multi-exchange methods
  void calculateExchangeCorrelations(const std::string& symbol,
                                   const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                                   AggregatedMarketData& result) const;
  void detectArbitrageOpportunities(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                                   AggregatedMarketData& result) const;
  void updateExchangeCorrelations();

  // Advanced aggregation algorithms
  double calculateTWAP(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                      uint64_t window_start, uint64_t window_end) const;
  double calculateVWAP(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateMedianPrice(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateTrimmedMean(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
                             double trim_percentage = 0.1) const;
  double calculateHarmonicMean(const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;

  // Multi-exchange aggregation methods
  std::optional<MultiExchangeData> getMultiExchangeView(const std::string& symbol) const;
  void applyExchangeSpecificAdjustments(RenderEngine::MarketDataUpdate& data,
                                     const std::string& exchange) const;
  void validateExchangeSpecificConstraints(const std::string& exchange,
                                        const std::string& symbol,
                                        const RenderEngine::MarketDataUpdate& data) const;
  std::vector<ExchangeRanking> rankExchangesByReliability() const;
  double calculateConsensusPrice(const std::string& symbol) const;
  void handleExchangeSpecificFeatures(const std::string& exchange,
                                    const std::string& symbol,
                                    const RenderEngine::MarketDataUpdate& update);
  std::optional<AggregatedMarketData> getExchangeSpecificAggregatedData(
      const std::string& symbol, const std::vector<std::string>& exchanges) const;
  double calculateWeightedAveragePriceWithValidationForExchanges(
      const std::string& symbol, const std::vector<std::string>& exchanges) const;
  double calculateConsensusPriceForExchanges(
      const std::string& symbol, const std::vector<std::string>& exchanges) const;

  // Enhanced multi-exchange aggregation methods
  std::optional<AggregatedMarketData> getUnifiedView(const std::string& symbol) const;
  std::vector<MultiExchangeData> getAllSymbolsMultiExchangeView() const;
  std::optional<AggregatedMarketData> getAdvancedAggregatedData(
      const std::string& symbol, const std::vector<std::string>& exchanges = {}) const;
  std::optional<ComprehensiveExchangeView> getComprehensiveExchangeView(
      const std::string& symbol) const;
};

}  // namespace Data
}  // namespace BTQuant