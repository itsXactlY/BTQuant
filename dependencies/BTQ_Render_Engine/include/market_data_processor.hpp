#pragma once

#include "hotspine_data_bridge.hpp"
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace BTQuant {
namespace RenderEngine {

// Trade data for analytics
struct TradeData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp;
  double price;
  double size;
  bool is_buy;
};

// Orderbook data for analytics
struct OrderbookData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
  double spread;
  double spread_percent;
  double bid_depth;
  double ask_depth;
  double total_depth;
  double imbalance; // (bid_depth - ask_depth) / total_depth
};

// Comprehensive symbol analytics
struct SymbolAnalytics {
  // Basic data
  uint32_t symbol_id = 0;
  uint64_t last_update_time = 0;

  // Trade analytics
  std::vector<TradeData> recent_trades;
  uint64_t trade_count = 0;
  double last_trade_price = 0.0;
  double last_trade_size = 0.0;
  uint64_t last_trade_time = 0;

  // Volume analytics
  double volume_1m = 0.0;
  double volume_5m = 0.0;
  double volume_15m = 0.0;
  double buy_volume = 0.0;
  double sell_volume = 0.0;
  uint64_t buy_count = 0;
  uint64_t sell_count = 0;
  double buy_sell_ratio = 0.5; // 0.5 = balanced, >0.5 = more buying

  // Price analytics
  double vwap = 0.0;              // Volume Weighted Average Price
  double vwap_deviation = 0.0;    // Current price deviation from VWAP (%)
  double momentum = 0.0;          // Price momentum (%)
  double momentum_strength = 0.0; // Volatility of momentum
  double price_min = 0.0;
  double price_max = 0.0;
  double price_position = 0.0; // Position between min/max (0-100%)

  // Volatility analytics
  double volatility = 0.0;   // Annualized volatility
  double sharpe_ratio = 0.0; // Return/volatility ratio

  // Trade size analytics
  double avg_trade_size = 0.0;
  uint64_t large_trade_count = 0; // Trades > 2x average size

  // Orderbook analytics
  std::vector<OrderbookData> recent_orderbooks;
  double current_spread = 0.0;
  double current_spread_percent = 0.0;
  double avg_spread = 0.0;
  double avg_spread_percent = 0.0;
  double current_imbalance = 0.0;
  double avg_imbalance = 0.0;
  double market_depth = 0.0;
};

// Performance metrics for the processor
struct ProcessorPerformanceMetrics {
  uint64_t total_trades_processed = 0;
  uint64_t total_orderbooks_processed = 0;
  std::chrono::high_resolution_clock::time_point last_update_time;
  double processing_latency_us = 0.0;
};

// Ranking criteria for symbol sorting
enum class RankingCriteria { VOLUME, MOMENTUM, VOLATILITY, SPREAD, IMBALANCE };

// Symbol ranking result
struct SymbolRanking {
  uint32_t symbol_id;
  double value;
  std::string label;
};

// Market summary statistics
struct MarketSummary {
  size_t total_symbols = 0;
  size_t active_symbols = 0;
  size_t trending_up = 0;
  size_t trending_down = 0;
  double avg_volume = 0.0;
  double avg_momentum = 0.0;
  double avg_volatility = 0.0;
  std::chrono::high_resolution_clock::time_point last_update;
};

/**
 * MarketDataProcessor - Advanced market data analytics engine
 *
 * This class processes real-time market data updates and calculates
 * comprehensive analytics including:
 * - VWAP (Volume Weighted Average Price)
 * - Price momentum and volatility
 * - Spread analysis and market depth
 * - Trading volume patterns
 * - Market microstructure metrics
 */
class MarketDataProcessor {
public:
  MarketDataProcessor();
  ~MarketDataProcessor();

  // Non-copyable, non-movable
  MarketDataProcessor(const MarketDataProcessor &) = delete;
  MarketDataProcessor &operator=(const MarketDataProcessor &) = delete;
  MarketDataProcessor(MarketDataProcessor &&) = delete;
  MarketDataProcessor &operator=(MarketDataProcessor &&) = delete;

  /**
   * Process a trade update
   * @param update Market data update containing trade information
   */
  void processTradeUpdate(const MarketDataUpdate &update);

  /**
   * Process an orderbook update
   * @param update Market data update containing orderbook information
   */
  void processOrderbookUpdate(const MarketDataUpdate &update);

  /**
   * Get comprehensive analytics for a symbol
   * @param symbol_id Symbol ID to get analytics for
   * @return Symbol analytics data
   */
  SymbolAnalytics getSymbolAnalytics(uint32_t symbol_id) const;

  /**
   * Get list of all active symbols
   * @return Vector of symbol IDs that have recent data
   */
  std::vector<uint32_t> getActiveSymbols() const;

  /**
   * Get performance metrics for the processor
   * @return Current performance metrics
   */
  ProcessorPerformanceMetrics getPerformanceMetrics() const;

  /**
   * Get symbol rankings based on criteria
   * @param criteria Ranking criteria (volume, momentum, etc.)
   * @param limit Maximum number of results (0 = no limit)
   * @return Vector of ranked symbols
   */
  std::vector<SymbolRanking> getRankings(RankingCriteria criteria,
                                         size_t limit = 0) const;

  /**
   * Get market summary statistics
   * @return Market-wide summary data
   */
  MarketSummary getMarketSummary() const;

  /**
   * Clear analytics data for a specific symbol
   * @param symbol_id Symbol ID to clear
   */
  void clearSymbolData(uint32_t symbol_id);

  /**
   * Clear all analytics data
   */
  void clearAllData();

  /**
   * Configuration methods
   */
  void setVWAPWindow(size_t window_size);
  void setMomentumWindow(size_t window_size);
  void setVolatilityWindow(size_t window_size);

private:
  // Configuration parameters
  size_t vwap_window_size_;
  size_t momentum_window_size_;
  size_t volatility_window_size_;
  size_t spread_analysis_window_;

  // Data storage
  mutable std::mutex data_mutex_;
  std::unordered_map<uint32_t, SymbolAnalytics> symbol_analytics_;

  // Performance tracking
  ProcessorPerformanceMetrics performance_metrics_;

  // Private calculation methods
  void updateVWAP(SymbolAnalytics &symbol_data);
  void updateMomentum(SymbolAnalytics &symbol_data);
  void updateVolatility(SymbolAnalytics &symbol_data);
  void updateTradingMetrics(SymbolAnalytics &symbol_data,
                            const TradeData &trade);
  void updateSpreadAnalysis(SymbolAnalytics &symbol_data);

  // Helper methods
  double calculateMarketDepth(const std::vector<PriceLevel> &levels) const;
  double calculateVolumeInWindow(const std::vector<TradeData> &trades,
                                 uint64_t window_us) const;
};

} // namespace RenderEngine
} // namespace BTQuant