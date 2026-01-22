#include <benchmark/benchmark.h>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <random>
#include <iostream>
#include "../include/market_data_processor.hpp"
#include "../include/hotspine_data_bridge.hpp"

// Benchmark für Trade Processing Latenz
static void BM_TradeProcessingLatency(benchmark::State& state) {
    auto processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();

    BTQuant::RenderEngine::MarketDataUpdate update;
    update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
    update.symbol_id = 1;
    update.timestamp = 1704067200000000ULL;
    update.price = 50000.0;
    update.size = 1.0;
    update.side = "buy";

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        processor->processTradeUpdate(update);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        state.SetIterationTime(elapsed.count() / 1e9); // In Sekunden für benchmark
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TradeProcessingLatency)->UseManualTime()->Threads(1);

// Benchmark für Trade Processing Durchsatz
static void BM_TradeProcessingThroughput(benchmark::State& state) {
    auto processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();
    const int batch_size = state.range(0);

    std::vector<BTQuant::RenderEngine::MarketDataUpdate> batch;
    batch.reserve(batch_size);

    // Erstelle Batch
    for (int i = 0; i < batch_size; ++i) {
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
        update.symbol_id = i % 100; // Verteile über verschiedene Symbole
        update.timestamp = 1704067200000000ULL + i;
        update.price = 50000.0 + (i % 1000) * 0.01;
        update.size = 1.0 + (i % 10) * 0.1;
        update.side = (i % 2 == 0) ? "buy" : "sell";
        batch.push_back(update);
    }

    for (auto _ : state) {
        processor->processTradeUpdates(batch);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_TradeProcessingThroughput)->Range(100, 10000)->Threads(1);

// Benchmark für Concurrent Trade Processing
static void BM_ConcurrentTradeProcessing(benchmark::State& state) {
    auto processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();
    const int num_threads = state.range(0);
    const int updates_per_thread = 1000;

    for (auto _ : state) {
        std::vector<std::thread> threads;
        std::vector<std::promise<void>> promises(num_threads);

        state.PauseTiming(); // Setup-Zeit nicht messen

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < updates_per_thread; ++i) {
                    BTQuant::RenderEngine::MarketDataUpdate update;
                    update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
                    update.symbol_id = (t * updates_per_thread + i) % 1000;
                    update.timestamp = 1704067200000000ULL + (t * updates_per_thread + i);
                    update.price = 50000.0 + i * 0.01;
                    update.size = 1.0;
                    update.side = "buy";

                    processor->processTradeUpdate(update);
                }
                promises[t].set_value();
            });
        }

        state.ResumeTiming(); // Jetzt timing starten

        // Warte auf alle Threads
        for (auto& promise : promises) {
            promise.get_future().wait();
        }

        // Cleanup
        for (auto& t : threads) {
            t.join();
        }
    }

    state.SetItemsProcessed(state.iterations() * num_threads * updates_per_thread);
}
BENCHMARK(BM_ConcurrentTradeProcessing)->RangeMultiplier(2)->Range(1, 8)->Threads(1);

// Benchmark für Orderbook Processing
static void BM_OrderbookProcessing(benchmark::State& state) {
    auto processor = std::make_unique<BTQuant::RenderEngine::MarketDataProcessor>();

    BTQuant::RenderEngine::MarketDataUpdate update;
    update.type = BTQuant::RenderEngine::MarketDataType::ORDERBOOK;
    update.symbol_id = 1;
    update.timestamp = 1704067200000000ULL;

    // Erstelle 20 Levels Bids und Asks
    for (int i = 0; i < 20; ++i) {
        PriceLevel bid;
        bid.price = 50000.0 - i * 10.0;
        bid.size = 10.0 + i;
        update.bids.push_back(bid);

        PriceLevel ask;
        ask.price = 50000.0 + i * 10.0;
        ask.size = 10.0 + i;
        update.asks.push_back(ask);
    }

    for (auto _ : state) {
        processor->processOrderbookUpdate(update);
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_OrderbookProcessing)->Threads(1);

// Benchmark für Atomic Operations Performance
static void BM_AtomicOperations(benchmark::State& state) {
    std::atomic<uint64_t> counter{0};
    const int num_operations = 1000000;

    for (auto _ : state) {
        for (int i = 0; i < num_operations; ++i) {
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    state.SetItemsProcessed(state.iterations() * num_operations);
}
BENCHMARK(BM_AtomicOperations)->Threads(1);

// Benchmark für Atomic vs Non-Atomic Performance Comparison
static void BM_AtomicVsNonAtomic(benchmark::State& state) {
    const bool use_atomic = state.range(0);
    const int num_operations = 100000;

    if (use_atomic) {
        std::atomic<uint64_t> counter{0};
        for (auto _ : state) {
            for (int i = 0; i < num_operations; ++i) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        uint64_t counter = 0; // Nicht thread-safe, nur für Vergleich
        for (auto _ : state) {
            for (int i = 0; i < num_operations; ++i) {
                counter++;
            }
        }
    }

    state.SetItemsProcessed(state.iterations() * num_operations);
}
BENCHMARK(BM_AtomicVsNonAtomic)->Arg(0)->Arg(1)->Threads(1);

// Benchmark für Ring Buffer Simulation (ähnlich wie in HotSpine)
static void BM_RingBufferProcessing(benchmark::State& state) {
    const size_t buffer_size = 10000;
    std::vector<BTQuant::HotTrade> trades(buffer_size);

    // Fülle Buffer mit Testdaten
    for (size_t i = 0; i < buffer_size; ++i) {
        trades[i].ts_exchange = 1704067200000000ULL + i;
        trades[i].price = 50000.0 + i * 0.01;
        trades[i].size = 1.0;
        trades[i].symbol_id = i % 100;
        trades[i].side = 0;
    }

    uint64_t read_index = 0;
    uint64_t write_index = buffer_size;

    for (auto _ : state) {
        // Simuliere Ring Buffer Processing
        while (read_index < write_index) {
            const auto& trade = trades[read_index % buffer_size];

            // Simuliere Validierung
            if (trade.ts_exchange >= 1704067200000000ULL &&
                trade.price > 0 && trade.size > 0) {
                // Trade wäre gültig
                benchmark::DoNotOptimize(trade);
            }

            read_index++;
        }
        read_index = 0; // Reset für nächste Iteration
    }

    state.SetItemsProcessed(state.iterations() * buffer_size);
}
BENCHMARK(BM_RingBufferProcessing)->Threads(1);

// Benchmark für Memory Access Patterns
static void BM_MemoryAccessPatterns(benchmark::State& state) {
    const bool sequential = state.range(0);
    const size_t data_size = 1000000;

    std::vector<uint64_t> data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        data[i] = i;
    }

    uint64_t sum = 0;

    if (sequential) {
        // Sequentieller Zugriff
        for (auto _ : state) {
            for (size_t i = 0; i < data_size; ++i) {
                sum += data[i];
            }
        }
    } else {
        // Random Zugriff (simuliert Sharding)
        std::vector<size_t> indices(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), std::mt19937{42});

        for (auto _ : state) {
            for (size_t i = 0; i < data_size; ++i) {
                sum += data[indices[i]];
            }
        }
    }

    benchmark::DoNotOptimize(sum);
    state.SetItemsProcessed(state.iterations() * data_size);
}
BENCHMARK(BM_MemoryAccessPatterns)->Arg(0)->Arg(1)->Threads(1);

// Benchmark für CPU Cache Efficiency
static void BM_CacheEfficiency(benchmark::State& state) {
    const size_t array_size = state.range(0);
    const size_t iterations = 1000000;

    // Simuliere verschiedene Cache Line Zugriffe
    std::vector<std::atomic<uint64_t>> atomic_array(array_size);
    std::vector<uint64_t> regular_array(array_size);

    // Initialisiere
    for (size_t i = 0; i < array_size; ++i) {
        atomic_array[i].store(i);
        regular_array[i] = i;
    }

    uint64_t sum = 0;

    for (auto _ : state) {
        for (size_t iter = 0; iter < iterations; ++iter) {
            size_t index = iter % array_size;
            sum += atomic_array[index].load(std::memory_order_relaxed);
        }
    }

    benchmark::DoNotOptimize(sum);
    state.SetItemsProcessed(state.iterations() * iterations);
}
BENCHMARK(BM_CacheEfficiency)->RangeMultiplier(2)->Range(64, 1024)->Threads(1);

// Benchmark für Real-time Scheduling Simulation
static void BM_RealtimeScheduling(benchmark::State& state) {
    // Simuliere die 10kHz Sync Loop aus HotSpineDataBridge
    const auto target_interval = std::chrono::microseconds(100); // 10kHz

    for (auto _ : state) {
        auto next_sync = std::chrono::steady_clock::now() + target_interval;

        // Simuliere Arbeit (minimal)
        benchmark::DoNotOptimize(1 + 1);

        // Simuliere sleep_until
        auto now = std::chrono::steady_clock::now();
        if (now < next_sync) {
            // In echtem Code: std::this_thread::sleep_until(next_sync);
            auto sleep_duration = next_sync - now;
            benchmark::DoNotOptimize(sleep_duration);
        }
    }
}
BENCHMARK(BM_RealtimeScheduling)->Threads(1);

// Hauptfunktion für Benchmarks
BENCHMARK_MAIN();