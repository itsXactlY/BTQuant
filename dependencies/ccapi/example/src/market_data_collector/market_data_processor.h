#pragma once

#include <atomic>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "ccapi_cpp/ccapi_session.h"
#include "candle_aggregator.h"
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
        double   avg_latency_ms{0.0};
        double trades_per_sec{0.0};
        double orderbooks_per_sec{0.0};
    };

    MarketDataProcessor(std::shared_ptr<MSSQLBulkInserter> db,
                        std::shared_ptr<CandleAggregator> candle_agg);

    void processEvent(const ccapi::Event& event,
                      ccapi::Session* session) override;

    void setBufferLimits(std::size_t max_trades,
                         std::size_t max_candles,
                         std::size_t max_orderbooks);

    void flushBuffers();  // called periodically by orchestrator

    Stats getStats() const;
    std::string getStatsJson() const;
    std::unordered_map<std::string, PairStats> getPairStats() const;

private:
    std::shared_ptr<MSSQLBulkInserter> db_;
    std::shared_ptr<CandleAggregator> candle_agg_;

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
    mutable std::atomic<int64_t>  last_stats_ts_us_{0};

    std::size_t max_trade_buffer_size_{500};
    std::size_t max_candle_buffer_size_{200};
    std::size_t max_orderbook_buffer_size_{100};

    void handleTradeMessage(const ccapi::Message& msg);
    void handleOrderbookMessage(const ccapi::Message& msg);

    void flushTradesIfNeeded(bool force = false);
    void flushCandlesIfNeeded(bool force = false);
    void flushOrderbooksIfNeeded(bool force = false);

    static std::vector<std::string> split(const std::string& s, char delim);
};
