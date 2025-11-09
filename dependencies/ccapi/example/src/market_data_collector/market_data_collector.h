#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "candle_aggregator.h"
#include "exchange_connection_manager.h"
#include "market_data_processor.h"
#include "mssql_bulk_inserter.h"

class MarketDataCollector {
public:
    struct Config {
        std::string db_connection_string;
        std::size_t bulk_insert_batch_size{500};

        std::vector<ExchangeConnectionManager::ExchangeConfig> exchanges;

        std::vector<std::string> timeframes{"1m", "5m", "15m", "1h"};

        std::size_t trade_buffer_size{1000};
        std::size_t candle_buffer_size{200};
        std::size_t orderbook_buffer_size{100};

        int flush_interval_ms{1000};
        int stats_report_interval_s{10};
    };

    explicit MarketDataCollector(const Config& cfg);
    ~MarketDataCollector();

    void start();
    void stop();
    void waitForShutdown();

    void printStats() const;

private:
    Config config_;
    std::shared_ptr<MSSQLBulkInserter> db_;
    std::shared_ptr<CandleAggregator> candle_agg_;
    std::shared_ptr<MarketDataProcessor> processor_;
    std::unique_ptr<ExchangeConnectionManager> conn_mgr_;

    std::thread flush_thread_;
    std::thread stats_thread_;
    std::atomic<bool> running_{false};

    void flushLoop();
    void statsLoop() const;
};
