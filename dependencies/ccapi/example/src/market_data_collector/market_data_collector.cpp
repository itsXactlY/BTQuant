#include "market_data_collector.h"

#include <chrono>
#include <iostream>
#include <thread>

MarketDataCollector::MarketDataCollector(const Config& cfg)
    : config_(cfg) {

    db_ = std::make_shared<MSSQLBulkInserter>(
        config_.db_connection_string);

    candle_agg_ = std::make_shared<CandleAggregator>(
        config_.timeframes);

    processor_ = std::make_shared<MarketDataProcessor>(db_, candle_agg_);
    processor_->setBufferLimits(config_.trade_buffer_size,
                                config_.candle_buffer_size,
                                config_.orderbook_buffer_size);

    conn_mgr_ = std::make_unique<ExchangeConnectionManager>(processor_);
}

MarketDataCollector::~MarketDataCollector() {
    stop();
}

void MarketDataCollector::start() {
    running_ = true;
    conn_mgr_->subscribe(config_.exchanges);
    conn_mgr_->start();

    flush_thread_ = std::thread(&MarketDataCollector::flushLoop, this);
    stats_thread_ = std::thread(&MarketDataCollector::statsLoop, this);
}

void MarketDataCollector::stop() {
    if (!running_) return;
    running_ = false;
    conn_mgr_->stop();

    if (flush_thread_.joinable()) flush_thread_.join();
    if (stats_thread_.joinable()) stats_thread_.join();

    // final flush
    processor_->flushBuffers();
    auto final_candles = candle_agg_->flushAll();
    if (!final_candles.empty()) {
        // group per table
        std::unordered_map<std::string,
                           std::vector<MarketData::OHLCV>> per_table;
        for (const auto& c : final_candles) {
            per_table[c.getTableName()].push_back(c);
        }
        for (auto& kv : per_table) {
            db_->bulkInsertOHLCV(kv.first, kv.second);
        }
    }
}

void MarketDataCollector::waitForShutdown() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void MarketDataCollector::printStats() const {
    auto st = processor_->getStats();
    std::cout << "=== Market Data Stats ===\n";
    std::cout << "Trades: received=" << st.trades_received
              << ", inserted=" << st.trades_inserted
              << ", rate=" << st.trades_per_sec << " /s\n";
    std::cout << "Candles: generated=" << st.candles_generated
              << ", inserted=" << st.candles_inserted << "\n";
    std::cout << "Orderbooks: received=" << st.orderbooks_received
              << ", inserted=" << st.orderbooks_inserted
              << ", rate=" << st.orderbooks_per_sec << " /s\n";
    std::cout << "Avg latency (ms): " << st.avg_latency_ms << "\n";
    std::cout << "Errors: " << st.errors << "\n";

    // scrape-friendly JSON line for Prom/Grafana → Loki/Tempo/etc.
    std::cout << "STATS_JSON " << processor_->getStatsJson() << "\n";
}

void MarketDataCollector::flushLoop() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.flush_interval_ms));
        processor_->flushBuffers();
    }
}

void MarketDataCollector::statsLoop() const {
    while (const_cast<std::atomic<bool>&>(running_)) {
        std::this_thread::sleep_for(
            std::chrono::seconds(config_.stats_report_interval_s));
        if (!const_cast<std::atomic<bool>&>(running_)) break;
        printStats();
    }
}
