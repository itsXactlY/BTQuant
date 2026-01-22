#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <random>
#include <future>
#include "../include/market_data_processor.hpp"

// Test Fixture für MarketDataProcessor
class MarketDataProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();
    }

    void TearDown() override {
        processor.reset();
    }

    std::unique_ptr<BTQuant::RenderEngine::MarketDataProcessor> processor;
};

// Test für ConcurrentQueue Ingestion
TEST_F(MarketDataProcessorTest, ConcurrentQueueIngestion) {
    const int num_producers = 4;
    const int updates_per_producer = 1000;
    const int total_updates = num_producers * updates_per_producer;

    std::vector<std::thread> producers;
    std::atomic<int> produced_count{0};

    // Starte Producer Threads
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([this, p, updates_per_producer, &produced_count]() {
            for (int i = 0; i < updates_per_producer; ++i) {
                BTQuant::RenderEngine::MarketDataUpdate update;
                update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
                update.symbol_id = (p * updates_per_producer + i) % 16; // Verteile über Shards
                update.timestamp = 1704067200000000ULL + (p * updates_per_producer + i);
                update.price = 50000.0 + (p * updates_per_producer + i) * 0.01;
                update.size = 1.0 + (i % 10);
                update.side = (i % 2 == 0) ? "buy" : "sell";

                processor->processTradeUpdate(update);
                produced_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Warte auf alle Producer
    for (auto& t : producers) {
        t.join();
    }

    EXPECT_EQ(produced_count.load(), total_updates);

    // Warte kurz für Verarbeitung
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Überprüfe, dass Updates verarbeitet wurden
    auto active_symbols = processor->getActiveSymbols();
    EXPECT_FALSE(active_symbols.empty());

    // Überprüfe Performance Metrics
    auto metrics = processor->getPerformanceMetrics();
    EXPECT_GE(metrics.total_trades_processed, total_updates);
}

// Test für Sharding Thread-Safety
TEST_F(MarketDataProcessorTest, ShardingThreadSafety) {
    const int num_threads = 8;
    const int operations_per_thread = 500;

    std::vector<std::thread> threads;
    std::vector<std::promise<bool>> promises(num_threads);

    // Starte Threads, die gleichzeitig auf verschiedene Shards zugreifen
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, operations_per_thread, &promises]() {
            try {
                uint32_t base_symbol = t * 1000; // Verschiedene Symbol-Bereiche für verschiedene Shards

                for (int i = 0; i < operations_per_thread; ++i) {
                    BTQuant::RenderEngine::MarketDataUpdate update;
                    update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
                    update.symbol_id = base_symbol + i;
                    update.timestamp = 1704067200000000ULL + i;
                    update.price = 50000.0 + i;
                    update.size = 1.0;
                    update.side = "buy";

                    processor->processTradeUpdate(update);

                    // Ab und zu Analytics lesen
                    if (i % 50 == 0) {
                        auto analytics = processor->getSymbolAnalytics(update.symbol_id);
                        EXPECT_EQ(analytics.symbol_id, update.symbol_id);
                    }
                }

                promises[t].set_value(true);
            } catch (...) {
                promises[t].set_value(false);
            }
        });
    }

    // Warte auf alle Threads
    for (auto& t : threads) {
        t.join();
    }

    // Überprüfe, dass alle Threads erfolgreich waren
    for (auto& promise : promises) {
        EXPECT_TRUE(promise.get_future().get());
    }

    // Überprüfe Gesamtanzahl der verarbeiteten Trades
    auto metrics = processor->getPerformanceMetrics();
    EXPECT_GE(metrics.total_trades_processed, num_threads * operations_per_thread);
}

// Test für Atomic Performance Metrics
TEST_F(MarketDataProcessorTest, AtomicPerformanceMetrics) {
    // Teste, dass Performance Metrics thread-safe aktualisiert werden

    std::vector<std::thread> threads;
    const int num_threads = 4;
    const int updates_per_thread = 100;

    // Starte Threads, die Updates senden
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, updates_per_thread]() {
            for (int i = 0; i < updates_per_thread; ++i) {
                BTQuant::RenderEngine::MarketDataUpdate update;
                update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
                update.symbol_id = t;
                update.timestamp = 1704067200000000ULL + i;
                update.price = 50000.0;
                update.size = 1.0;
                update.side = "buy";

                processor->processTradeUpdate(update);
            }
        });
    }

    // Warte auf Threads
    for (auto& t : threads) {
        t.join();
    }

    // Warte für Verarbeitung
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Überprüfe Metrics
    auto metrics = processor->getPerformanceMetrics();
    EXPECT_GE(metrics.total_trades_processed, num_threads * updates_per_thread);
    EXPECT_GE(metrics.trades_per_second, 0.0);
    EXPECT_GE(metrics.avg_latency_ms, 0.0);
}

// Test für Orderbook Processing
TEST_F(MarketDataProcessorTest, OrderbookProcessing) {
    const int num_orderbooks = 100;

    for (int i = 0; i < num_orderbooks; ++i) {
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::ORDERBOOK;
        update.symbol_id = 1;
        update.timestamp = 1704067200000000ULL + i * 1000; // 1ms Abstand

        // Erstelle Bids und Asks
        for (int level = 0; level < 10; ++level) {
            PriceLevel bid_level;
            bid_level.price = 50000.0 - level * 10.0;
            bid_level.size = 10.0 + level;
            update.bids.push_back(bid_level);

            PriceLevel ask_level;
            ask_level.price = 50000.0 + level * 10.0;
            ask_level.size = 10.0 + level;
            update.asks.push_back(ask_level);
        }

        processor->processOrderbookUpdate(update);
    }

    // Warte für Verarbeitung
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Überprüfe Analytics
    auto analytics = processor->getSymbolAnalytics(1);
    EXPECT_EQ(analytics.symbol_id, 1u);
    EXPECT_FALSE(analytics.recent_orderbooks.empty());

    // Überprüfe Performance Metrics
    auto metrics = processor->getPerformanceMetrics();
    EXPECT_GE(metrics.total_orderbooks_processed, num_orderbooks);
}

// Test für Batch Processing Performance
TEST_F(MarketDataProcessorTest, BatchProcessingPerformance) {
    const int batch_size = 10000;
    std::vector<BTQuant::RenderEngine::MarketDataUpdate> batch;

    // Erstelle großen Batch
    for (int i = 0; i < batch_size; ++i) {
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
        update.symbol_id = i % 100; // Verteile über verschiedene Symbole
        update.timestamp = 1704067200000000ULL + i;
        update.price = 50000.0 + (i % 1000);
        update.size = 1.0 + (i % 10);
        update.side = (i % 2 == 0) ? "buy" : "sell";
        batch.push_back(update);
    }

    // Messe Batch-Verarbeitungszeit
    auto start = std::chrono::high_resolution_clock::now();
    processor->processTradeUpdates(batch);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Batch processing time for " << batch_size << " updates: "
              << duration.count() << " ms" << std::endl;

    // Sollte unter 1 Sekunde sein für gute Performance
    EXPECT_LT(duration.count(), 1000);

    // Warte für Verarbeitung
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Überprüfe, dass alle Updates verarbeitet wurden
    auto metrics = processor->getPerformanceMetrics();
    EXPECT_GE(metrics.total_trades_processed, batch_size);
}

// Test für Memory Consistency mit Atomic Operations
TEST_F(MarketDataProcessorTest, MemoryConsistency) {
    // Teste, dass atomic operations korrekte memory ordering garantieren

    std::atomic<bool> data_ready{false};
    BTQuant::RenderEngine::SymbolAnalytics shared_analytics;

    std::thread writer([this, &data_ready, &shared_analytics]() {
        // Schreibe Daten
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
        update.symbol_id = 999;
        update.timestamp = 1704067200000000ULL;
        update.price = 50000.0;
        update.size = 1.0;
        update.side = "buy";

        processor->processTradeUpdate(update);

        // Warte kurz für Verarbeitung
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Lese Analytics
        shared_analytics = processor->getSymbolAnalytics(999);

        // Signalisiere, dass Daten bereit sind
        data_ready.store(true, std::memory_order_release);
    });

    std::thread reader([&data_ready, &shared_analytics]() {
        // Warte auf Daten
        while (!data_ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        // Daten sollten konsistent sein
        EXPECT_EQ(shared_analytics.symbol_id, 999u);
        EXPECT_GE(shared_analytics.last_trade_price, 0.0);
    });

    writer.join();
    reader.join();

    EXPECT_TRUE(data_ready.load());
}

// Test für Parallel Processing Configuration
TEST_F(MarketDataProcessorTest, ParallelProcessingConfig) {
    // Teste Parallel Processing Enable/Disable
    processor->setParallelProcessingEnabled(true);
    EXPECT_TRUE(processor->isParallelProcessingEnabled());

    processor->setParallelProcessingEnabled(false);
    EXPECT_FALSE(processor->isParallelProcessingEnabled());
}

// Stress Test für Concurrent Access
TEST_F(MarketDataProcessorTest, DISABLED_ConcurrentStressTest) {
    const int num_threads = 16;
    const int duration_seconds = 5;

    std::vector<std::thread> threads;
    std::atomic<bool> running{true};
    std::vector<std::atomic<uint64_t>> thread_update_counts(num_threads);

    // Starte Stress-Threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, &running, &thread_update_counts]() {
            uint64_t local_count = 0;
            while (running.load(std::memory_order_relaxed)) {
                BTQuant::RenderEngine::MarketDataUpdate update;
                update.type = (local_count % 2 == 0) ?
                    BTQuant::RenderEngine::MarketDataType::TRADE :
                    BTQuant::RenderEngine::MarketDataType::ORDERBOOK;
                update.symbol_id = t * 100 + (local_count % 10);
                update.timestamp = 1704067200000000ULL + local_count;
                update.price = 50000.0 + local_count % 1000;
                update.size = 1.0;

                if (update.type == BTQuant::RenderEngine::MarketDataType::TRADE) {
                    processor->processTradeUpdate(update);
                } else {
                    // Füge einige Orderbook levels hinzu
                    for (int l = 0; l < 5; ++l) {
                        PriceLevel level;
                        level.price = update.price + l * 10;
                        level.size = 10.0;
                        update.bids.push_back(level);
                        update.asks.push_back(level);
                    }
                    processor->processOrderbookUpdate(update);
                }

                local_count++;
            }
            thread_update_counts[t].store(local_count, std::memory_order_relaxed);
        });
    }

    // Lauf für angegebene Dauer
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    running.store(false, std::memory_order_relaxed);

    // Warte auf Threads
    for (auto& t : threads) {
        t.join();
    }

    // Sammle Statistiken
    uint64_t total_updates = 0;
    for (auto& count : thread_update_counts) {
        total_updates += count.load();
    }

    auto metrics = processor->getPerformanceMetrics();

    std::cout << "Stress test results:" << std::endl;
    std::cout << "Total updates sent: " << total_updates << std::endl;
    std::cout << "Trades processed: " << metrics.total_trades_processed << std::endl;
    std::cout << "Orderbooks processed: " << metrics.total_orderbooks_processed << std::endl;
    std::cout << "Trades/sec: " << metrics.trades_per_second << std::endl;
    std::cout << "Orderbooks/sec: " << metrics.orderbooks_per_second << std::endl;
    std::cout << "Avg latency: " << metrics.avg_latency_ms << " ms" << std::endl;

    // Performance Assertions
    EXPECT_GE(metrics.trades_per_second, 1000.0); // Mindestens 1000 Trades/sec
    EXPECT_LT(metrics.avg_latency_ms, 10.0); // Unter 10ms Latenz
}