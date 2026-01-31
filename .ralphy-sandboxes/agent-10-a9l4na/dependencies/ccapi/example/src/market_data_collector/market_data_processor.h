#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Correct relative path from dependencies/ccapi/example/src/market_data_collector
#include "../../../../BTQ_Render_Engine/include/analytics/cluster_engine.hpp"
#include "../../../../BTQ_Render_Engine/include/hotspine_layout_v3.hpp"
#include "../hotspine/hotspine_writer.hpp"
#include "candle_aggregator.h"
#include "ccapi_cpp/ccapi_session.h"
#include "config_types.h"
#include "mssql_bulk_inserter.h"

class MarketDataProcessor : public ccapi::EventHandler {
 public:
  struct PairStats {
    uint64_t trades{0};
    uint64_t orderbooks{0};
  };

  struct Stats {
    uint64_t trades_received{0};
    uint64_t trades_inserted{0};
    uint64_t candles_generated{0};
    uint64_t candles_inserted{0};
    uint64_t orderbooks_received{0};
    uint64_t orderbooks_inserted{0};
    uint64_t errors{0};
    double avg_latency_ms{0.0};
    double trades_per_sec{0.0};
    double orderbooks_per_sec{0.0};
  };

  MarketDataProcessor(std::shared_ptr<MSSQLBulkInserter> db, std::shared_ptr<CandleAggregator> candle_agg,
                      std::shared_ptr<HotSpine::HotSpineWriter> hotspine_writer = nullptr, bool enable_exclusive_hotspine = false,
                      const ConfigTypes::DebugConfig& debug_config = {});

  void processEvent(const ccapi::Event& event, ccapi::Session* session) override;

  void setBufferLimits(std::size_t max_trades, std::size_t max_candles, std::size_t max_orderbooks);

  void logWebSocketDataFlowStats() const;
  void addWebSocketDebugging();
  void validateWebSocketDataFlow();
  void validateCCAPIConfiguration();

  void flushBuffers();  // called periodically by orchestrator

  // Connection health monitoring
  void checkDatabaseConnectionHealth();
  bool attemptDatabaseReconnect();

  // HotSpine writer update for reconnection
  void updateHotSpineWriter(std::shared_ptr<HotSpine::HotSpineWriter> new_writer);

  Stats getStats() const;
  std::string getStatsJson() const;
  std::unordered_map<std::string, PairStats> getPairStats() const;

 private:
  void initHotSpineV3();
  void threadAffinityCheck();

  Analytics::ClusterEngine engine_{0.5};
  HotSpine::V3::SharedMemoryLayoutV3* shm_v3_ = nullptr;
  int shm_v3_fd_ = -1;
  std::once_flag affinity_flag_;
  std::shared_ptr<MSSQLBulkInserter> db_;
  std::shared_ptr<CandleAggregator> candle_agg_;
  std::shared_ptr<HotSpine::HotSpineWriter> hotspine_writer_;

  std::vector<MarketData::Trade> trade_buffer_;
  std::vector<MarketData::OHLCV> candle_buffer_;
  std::vector<MarketData::OrderbookSnapshot> orderbook_buffer_;
  std::unordered_set<std::string> active_pairs_;

  mutable std::mutex buffer_mutex_;
  mutable std::mutex db_mutex_;

  std::unordered_map<std::string, PairStats> pair_stats_;
  mutable std::mutex stats_mutex_;

  std::atomic<uint64_t> trades_received_{0};
  std::atomic<uint64_t> errors_{0};
  Stats stats_{};

  // for msg/sec estimation
  mutable std::atomic<uint64_t> trades_last_window_{0};
  mutable std::atomic<uint64_t> orderbooks_last_window_{0};
  mutable std::atomic<int64_t> last_stats_ts_us_{0};
  mutable std::atomic<double> trades_per_sec_{0.0};
  mutable std::atomic<double> orderbooks_per_sec_{0.0};

  std::size_t max_trade_buffer_size_{500};
  std::size_t max_candle_buffer_size_{200};
  std::size_t max_orderbook_buffer_size_{100};

  bool enable_exclusive_hotspine_{false};
  ConfigTypes::DebugConfig debug_config_;

  // Connection health monitoring
  std::atomic<int64_t> last_connection_check_time_{0};
  static constexpr int64_t CONNECTION_CHECK_INTERVAL_MS = 30000;  // 30 seconds

  void debugLog(const std::string& msg) const;

  void handleTradeMessage(const ccapi::Message& msg);
  void handleOrderbookMessage(const ccapi::Message& msg);

  void flushTradesIfNeeded(bool force = false);
  void flushCandlesIfNeeded(bool force = false);
  void flushOrderbooksIfNeeded(bool force = false);

  static std::vector<std::string> split(const std::string& s, char delim);
};
