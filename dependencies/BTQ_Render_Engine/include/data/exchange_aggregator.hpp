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
  int precision = 8;                     // Decimal precision for price rounding
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

// Structures for enhanced multi-exchange aggregation
struct TimestampSynchronizationResult {
  std::string symbol;
  TimeSyncStrategy strategy_used;
  uint64_t synchronized_timestamp = 0;
  std::unordered_map<std::string, uint64_t> original_timestamps;  // Original timestamps from each exchange
  std::unordered_map<std::string, uint64_t> synchronization_accuracy;  // Accuracy of sync for each exchange (in microseconds)
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct ExchangeCorrelationMatrix {
  std::string symbol;
  std::unordered_map<std::string, std::unordered_map<std::string, double>> correlations;  // Correlation between exchanges
  double overall_market_correlation = 0.0;  // Overall correlation across all exchanges
  std::chrono::high_resolution_clock::time_point timestamp;
};

struct ExchangeSpecificRisk {
  std::string exchange_name;
  double latency_risk = 0.0;              // Risk due to latency (in milliseconds)
  double fee_cost = 0.0;                  // Trading fee cost
  double reliability_score = 0.0;         // Reliability score (0.0-1.0)
  double price_deviation_risk = 0.0;      // Risk due to price deviation from market average
};

struct ExchangeRiskMetrics {
  std::string symbol;
  std::unordered_map<std::string, ExchangeSpecificRisk> exchange_specific_risks;  // Risk metrics per exchange
  double price_volatility = 0.0;          // Overall price volatility across exchanges
  double coefficient_of_variation = 0.0;  // Coefficient of variation (volatility relative to mean)
  double price_spread = 0.0;              // Price spread between highest and lowest exchanges
  double normalized_price_spread = 0.0;   // Normalized price spread relative to average price
  double volume_concentration_risk = 0.0; // Risk due to volume concentration on single exchange
  double price_variance = 0.0;            // Price variance across exchanges
  double max_price_deviation = 0.0;       // Maximum deviation from average price
  double volume_distribution_entropy = 0.0; // Entropy measure of volume distribution across exchanges
  std::chrono::high_resolution_clock::time_point timestamp;
};

// Result of multi-exchange time synchronization
struct MultiExchangeTimeSyncResult {
  std::string symbol;
  TimeSyncStrategy strategy_used;
  uint64_t synchronized_timestamp = 0;
  std::unordered_map<std::string, uint64_t> original_timestamps;    // Original timestamps from each exchange
  std::unordered_map<std::string, uint64_t> compensated_timestamps; // Timestamps after latency compensation
  std::unordered_map<std::string, uint64_t> synchronization_accuracy;  // Accuracy of sync for each exchange (in microseconds)
  uint64_t average_original_timestamp = 0;  // Average of original timestamps
  double timestamp_variance = 0.0;          // Variance of timestamps across exchanges
  double cross_correlation = 0.0;           // Cross-correlation between exchanges
  std::chrono::high_resolution_clock::time_point timestamp;
};

// Comprehensive view of all exchanges for a symbol with all analytics
struct ComprehensiveMultiExchangeView {
  std::string symbol;
  std::unordered_map<std::string, ExchangeConsolidatedData> exchange_data;  // Detailed data per exchange
  ConsolidatedMarketMetrics market_metrics;                                 // Overall market metrics
  ConsolidatedRiskMetrics risk_metrics;                                     // Risk metrics
  ExchangeDataQualityMetrics data_quality;                                  // Data quality metrics
  std::vector<ExchangeRanking> exchange_rankings;                           // Rankings of exchanges by reliability

  // Arbitrage detection
  bool arbitrage_opportunity_exists = false;
  double arbitrage_profit_potential = 0.0;
  std::string best_arbitrage_buy_exchange = "";
  std::string best_arbitrage_sell_exchange = "";

  std::chrono::high_resolution_clock::time_point timestamp;
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

  // Exchange-specific quality metrics
struct ExchangeSpecificQuality {
  std::string exchange_name;
  double freshness_score = 0.0;        // Score based on data age (0-1)
  double completeness_score = 0.0;     // Score based on data completeness (0-1)
  double accuracy_score = 0.0;         // Score based on agreement with other exchanges (0-1)
  double overall_quality_score = 0.0;  // Combined quality score (0-1)
};

// Data quality metrics for multi-exchange aggregation
struct ExchangeDataQualityMetrics {
  std::string symbol;
  std::unordered_map<std::string, ExchangeSpecificQuality> exchange_quality_metrics;
  double cross_exchange_consistency = 0.0;  // How consistent prices are across exchanges (0-1)
  double data_reliability_score = 0.0;      // Overall reliability of the aggregated data (0-1)
  std::chrono::high_resolution_clock::time_point timestamp;
};

// Exchange latency report for monitoring
struct ExchangeLatencyReport {
  std::string exchange_name;
  int64_t current_latency_ms = 0;           // Current latency in milliseconds
  double configured_latency_offset_us = 0.0; // Configured offset in microseconds
  double reliability_score = 0.0;           // Current reliability score
  bool is_valid = false;                    // Whether the exchange is currently valid
};

// Market metrics for consolidated view
struct ConsolidatedMarketMetrics {
  double average_price = 0.0;
  double spread = 0.0;                    // Difference between highest and lowest prices
  double volatility = 0.0;                // Normalized price variation
  double total_volume = 0.0;              // Total volume across all exchanges
  double lowest_price = 0.0;              // Lowest price across exchanges
  double highest_price = 0.0;             // Highest price across exchanges
  double price_range = 0.0;               // Range between lowest and highest prices
  double bid_ask_spread = 0.0;            // Spread between best bid and ask
  std::string best_bid_exchange = "";     // Exchange with best bid
  std::string best_ask_exchange = "";     // Exchange with best ask
  double cross_exchange_correlation = 0.0; // Correlation between exchanges
};

// Risk metrics for consolidated view
struct ConsolidatedRiskMetrics {
  double price_volatility = 0.0;          // Standard deviation of prices
  double coefficient_of_variation = 0.0;  // Volatility relative to mean
};

// Consolidated data for a single exchange
struct ExchangeConsolidatedData {
  RenderEngine::MarketDataUpdate update;
  ExchangeFeatures features;
  ExchangeSpecificStats stats;
};

// Multi-exchange consolidated view
struct MultiExchangeConsolidatedView {
  std::string symbol;
  std::unordered_map<std::string, ExchangeConsolidatedData> exchange_data;  // Data per exchange
  ConsolidatedMarketMetrics market_metrics;                                 // Overall market metrics
  ConsolidatedRiskMetrics risk_metrics;                                     // Risk metrics
  bool arbitrage_opportunity_exists = false;                               // Whether arbitrage is possible
  double arbitrage_profit_potential = 0.0;                                 // Potential profit from arbitrage
  std::string best_arbitrage_buy_exchange = "";                           // Best exchange to buy from for arbitrage
  std::string best_arbitrage_sell_exchange = "";                          // Best exchange to sell to for arbitrage
  std::chrono::high_resolution_clock::time_point timestamp;
};

// Unified view combining all multi-exchange features
struct UnifiedMultiExchangeView {
  std::string symbol;
  std::unordered_map<std::string, ExchangeConsolidatedData> exchange_data;  // Detailed data per exchange
  ConsolidatedMarketMetrics market_metrics;                                 // Overall market metrics
  ConsolidatedRiskMetrics risk_metrics;                                     // Risk metrics
  ExchangeDataQualityMetrics data_quality;                                  // Data quality metrics
  std::vector<ExchangeRanking> exchange_rankings;                           // Rankings of exchanges by reliability
  MultiExchangeTimeSyncResult time_sync_result;                             // Time synchronization result

  // Arbitrage detection
  bool arbitrage_opportunity_exists = false;
  double arbitrage_profit_potential = 0.0;
  std::string best_arbitrage_buy_exchange = "";
  std::string best_arbitrage_sell_exchange = "";

  std::chrono::high_resolution_clock::time_point timestamp;
};

// Cross-exchange analytics for a symbol
struct SymbolCrossExchangeAnalytics {
  std::string symbol;
  std::unordered_map<std::string, double> exchange_prices;                // Price from each exchange
  std::unordered_map<std::string, double> exchange_volumes;               // Volume from each exchange

  // Statistical measures
  double mean_price = 0.0;                                               // Average price across exchanges
  double median_price = 0.0;                                             // Median price across exchanges
  double std_deviation = 0.0;                                            // Standard deviation of prices
  double variance = 0.0;                                                 // Variance of prices
  double coefficient_of_variation = 0.0;                                 // CV = std_dev / mean
  double min_price = 0.0;                                                // Minimum price across exchanges
  double max_price = 0.0;                                                // Maximum price across exchanges
  double price_range = 0.0;                                              // Range = max - min
  double skewness = 0.0;                                                 // Measure of asymmetry
  double kurtosis = 0.0;                                                 // Measure of tail heaviness

  // Volume metrics
  double total_volume = 0.0;                                             // Total volume across exchanges
  double volume_weighted_average_price = 0.0;                            // VWAP across exchanges
  double volume_concentration_index = 0.0;                               // Herfindahl-Hirschman Index for volume

  // Arbitrage metrics
  double max_arbitrage_potential = 0.0;                                  // Max potential profit from arbitrage
  double relative_arbitrage_potential = 0.0;                             // Relative arbitrage potential (%)

  std::chrono::high_resolution_clock::time_point timestamp;
};

// Dispersion metrics for advanced aggregation
struct DispersionMetrics {
  double mean = 0.0;
  double standard_deviation = 0.0;
  double variance = 0.0;
  double min = 0.0;
  double max = 0.0;
  double range = 0.0;
  double coefficient_of_variation = 0.0;
};

// Outlier detection metrics
struct OutlierDetectionMetrics {
  double q1 = 0.0;              // First quartile
  double q3 = 0.0;              // Third quartile
  double iqr = 0.0;             // Interquartile range
  double lower_fence = 0.0;     // Lower bound for outliers (Q1 - 1.5*IQR)
  double upper_fence = 0.0;     // Upper bound for outliers (Q3 + 1.5*IQR)
  int outlier_count = 0;        // Number of detected outliers
};

// Result of advanced aggregation
struct AdvancedAggregationResult {
  std::string symbol;
  std::vector<std::string> included_exchanges;

  // Different aggregation methods
  double simple_average = 0.0;
  double weighted_average = 0.0;
  double volume_weighted = 0.0;
  double median_price = 0.0;
  double geometric_mean = 0.0;
  double robust_mean = 0.0;
  double harmonic_mean = 0.0;
  double consensus_price = 0.0;

  // Percentile-based aggregations
  double percentile_25th = 0.0;
  double percentile_75th = 0.0;
  double percentile_90th = 0.0;
  double percentile_95th = 0.0;

  // Dispersion and statistical metrics
  DispersionMetrics dispersion_metrics;
  OutlierDetectionMetrics outlier_detection;

  // Confidence intervals
  double confidence_interval_lower = 0.0;
  double confidence_interval_upper = 0.0;

  std::chrono::high_resolution_clock::time_point timestamp;
};

// Enhanced multi-exchange aggregation methods
  std::optional<TimestampSynchronizationResult> synchronizeTimestampsAcrossExchanges(
      const std::string& symbol, TimeSyncStrategy strategy) const;
  std::vector<ExchangeCorrelationMatrix> calculateExchangeCorrelationsMatrix() const;
  std::optional<ExchangeRiskMetrics> calculateRiskMetrics(const std::string& symbol) const;

  // Enhanced multi-exchange time synchronization and comprehensive views
  std::optional<MultiExchangeTimeSyncResult> performMultiExchangeTimeSync(
      const std::string& symbol, TimeSyncStrategy strategy) const;
  std::optional<ComprehensiveMultiExchangeView> getComprehensiveMultiExchangeView(
      const std::string& symbol) const;
  std::optional<UnifiedMultiExchangeView> getUnifiedMultiExchangeView(
      const std::string& symbol) const;
  std::vector<UnifiedMultiExchangeView> getAllSymbolsUnifiedView() const;
  std::optional<MultiExchangeTimeSyncResult> performAdvancedTimeSyncWithPrediction(
      const std::string& symbol, TimeSyncStrategy strategy) const;

  // Exchange-specific feature handling methods
  void updateExchangeSpecificFeatures(const std::string& exchange, const ExchangeFeatures& new_features);
  std::optional<ExchangeFeatures> getEnhancedExchangeFeatures(const std::string& exchange) const;
  std::vector<ExchangeLatencyReport> generateLatencyReport() const;
  std::optional<ExchangeDataQualityMetrics> calculateDataQualityMetrics(const std::string& symbol) const;

  // Comprehensive multi-exchange view methods
  std::optional<MultiExchangeConsolidatedView> getMultiExchangeConsolidatedView(const std::string& symbol) const;
  std::vector<MultiExchangeConsolidatedView> getAllSymbolsConsolidatedView() const;
  std::optional<SymbolCrossExchangeAnalytics> getCrossExchangeAnalytics(const std::string& symbol) const;

  // Advanced aggregation algorithms
  double calculateGeometricMeanPrice(
      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateRobustMeanPrice(
      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  double calculateWeightedPercentilePrice(
      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
      double percentile) const;
  double calculateSimpleAveragePrice(
      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data) const;
  std::optional<AdvancedAggregationResult> performAdvancedAggregation(const std::string& symbol) const;

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

  // Enhanced multi-exchange aggregation with quality weighting and risk assessment
  std::optional<AggregatedMarketData> getEnhancedAggregatedData(const std::string& symbol) const;
  std::optional<AggregatedMarketData> getQualityWeightedAggregatedData(const std::string& symbol) const;
  std::unordered_map<std::string, double> getExchangeQualityScores(const std::string& symbol) const;

  // Enhanced time synchronization methods
  std::optional<TimestampSynchronizationResult> analyzeHistoricalTimeSync(
      const std::string& symbol, TimeSyncStrategy strategy) const;

  // Enhanced exchange-specific features handling
  void handleEnhancedExchangeSpecificFeatures(const std::string& exchange,
                                           const std::string& symbol,
                                           RenderEngine::MarketDataUpdate& update);
  void processEnhancedDataUpdate(const std::string& exchange, const std::string& symbol,
                               const RenderEngine::MarketDataUpdate& update);
  std::optional<AggregatedMarketData> getCustomWeightedAggregatedData(
      const std::string& symbol,
      const std::unordered_map<std::string, double>& custom_weights) const;
  void handleComprehensiveExchangeSpecificFeatures(const std::string& exchange,
                                               const std::string& symbol,
                                               RenderEngine::MarketDataUpdate& update);
  void processDataUpdateWithComprehensiveFeatures(const std::string& exchange,
                                               const std::string& symbol,
                                               const RenderEngine::MarketDataUpdate& update);

  // Exchange-specific quality and feature management
  bool performExchangeSpecificQualityChecks(const std::string& exchange,
                                         const std::string& symbol,
                                         const RenderEngine::MarketDataUpdate& update) const;
  void updateExchangeFeaturesDynamically(const std::string& exchange);

  // Advanced aggregation algorithms
  std::optional<AggregatedMarketData> getKalmanFilteredAggregatedData(
      const std::string& symbol) const;
  std::optional<AggregatedMarketData> getMLWeightedAggregatedData(
      const std::string& symbol) const;
  std::optional<AggregatedMarketData> getOutlierResistantAggregatedData(
      const std::string& symbol) const;
  std::optional<AggregatedMarketData> getUltimateAggregatedData(
      const std::string& symbol) const;

  // New methods for comprehensive multi-exchange aggregation
  std::optional<ComprehensiveMultiExchangeView> getUnifiedMultiExchangeView(
      const std::string& symbol) const;
  std::optional<MultiExchangeTimeSyncResult> performComprehensiveTimeSync(
      const std::string& symbol, TimeSyncStrategy strategy) const;

private:
  // Enhanced risk metrics calculation
  void calculateEnhancedRiskMetrics(
      const std::string& symbol,
      const std::unordered_map<std::string, RenderEngine::MarketDataUpdate>& exchange_data,
      AggregatedMarketData& result) const;

  // Enhanced time synchronization
  void enhancedSynchronizeTimestamps(AggregatedMarketData& data) const;
};

}  // namespace Data
}  // namespace BTQuant