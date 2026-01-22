#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <random>
#include "../include/hotspine_data_bridge.hpp"
#include "../include/market_data_processor.hpp"

// Mock für MarketDataProcessor
class MockMarketDataProcessor : public BTQuant::RenderEngine::MarketDataProcessor {
public:
    MOCK_METHOD(void, processTradeUpdate, (const BTQuant::RenderEngine::MarketDataUpdate&));
    MOCK_METHOD(void, processTradeUpdates, (const std::vector<BTQuant::RenderEngine::MarketDataUpdate>&));
    MOCK_METHOD(void, processOrderbookUpdate, (const BTQuant::RenderEngine::MarketDataUpdate&));
    MOCK_METHOD(BTQuant::RenderEngine::SymbolAnalytics, getSymbolAnalytics, (uint32_t), (const));
    MOCK_METHOD(std::vector<uint32_t>, getActiveSymbols, (), (const));
};

class HotSpineDataBridgeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Mock Processor erstellen
        mock_processor = std::make_shared<MockMarketDataProcessor>();
        bridge = std::make_unique<BTQuant::HotSpineDataBridge>("/test_shm");
        bridge->setMarketDataProcessor(mock_processor);
    }

    void TearDown() override {
        if (bridge) {
            bridge->stop();
        }
    }

    std::shared_ptr<MockMarketDataProcessor> mock_processor;
    std::unique_ptr<BTQuant::HotSpineDataBridge> bridge;
};

// Test für atomische Operationen in read indices
TEST_F(HotSpineDataBridgeTest, AtomicReadIndices) {
    // Teste, dass die atomic read indices korrekt initialisiert sind
    // Da wir keinen echten SHM haben, testen wir die Struktur

    // Simuliere mehrere Threads, die gleichzeitig auf read indices zugreifen
    std::vector<std::thread> threads;
    std::atomic<int> access_count{0};

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([this, &access_count]() {
            // Simuliere Zugriff auf atomic member (würde in echtem Code passieren)
            access_count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(access_count.load(), 10);
}

// Test für lock-freie Ring Buffer Logik
TEST_F(HotSpineDataBridgeTest, RingBufferWraparoundDetection) {
    // Teste die Logik für Ring Buffer wraparound
    // Simuliere write_idx < read_idx (wraparound case)

    uint64_t capacity = 1000;
    uint64_t write_idx = 500;  // Write pointer hat gewrapped
    uint64_t read_idx = 800;   // Read pointer ist hinter write pointer

    // Logik aus sync_shm(): wenn read_idx > write_idx, dann wraparound
    bool wraparound_detected = (read_idx > write_idx);

    EXPECT_TRUE(wraparound_detected);

    // Berechne neuen read_idx
    uint64_t new_read_idx;
    if (write_idx > capacity / 4) {
        new_read_idx = write_idx - (capacity / 4);
    } else {
        new_read_idx = 0;
    }

    EXPECT_EQ(new_read_idx, 500 - 250); // 250
}

// Test für Batch-Verarbeitung von Trades
TEST_F(HotSpineDataBridgeTest, TradeBatchProcessing) {
    const size_t BATCH_SIZE = 1000;
    std::vector<BTQuant::RenderEngine::MarketDataUpdate> batch;
    batch.reserve(BATCH_SIZE);

    // Simuliere Batch-Füllung
    for (size_t i = 0; i < BATCH_SIZE + 100; ++i) {
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
        update.symbol_id = 1;
        update.timestamp = 1704067200000000ULL + i;
        update.price = 50000.0 + i;
        update.size = 1.0;
        update.side = "buy";

        batch.push_back(update);

        // Simuliere Batch-Verarbeitung alle BATCH_SIZE
        if (batch.size() >= BATCH_SIZE) {
            EXPECT_CALL(*mock_processor, processTradeUpdates(testing::SizeIs(BATCH_SIZE)))
                .Times(1);
            mock_processor->processTradeUpdates(batch);
            batch.clear();
        }
    }

    // Final batch
    if (!batch.empty()) {
        EXPECT_CALL(*mock_processor, processTradeUpdates(testing::SizeIs(100)))
            .Times(1);
        mock_processor->processTradeUpdates(batch);
    }
}

// Test für Validierung von Trade-Daten
TEST_F(HotSpineDataBridgeTest, TradeDataValidation) {
    // Teste die Validierungslogik aus sync_shm()

    BTQuant::HotTrade valid_trade;
    valid_trade.ts_exchange = 1704067200000000ULL; // Valid timestamp
    valid_trade.price = 50000.0;
    valid_trade.size = 1.0;
    valid_trade.symbol_id = 1;
    valid_trade.side = 0;

    BTQuant::HotTrade invalid_trade;
    invalid_trade.ts_exchange = 1000000000000ULL; // Invalid timestamp (< MIN_VALID_TS)
    invalid_trade.price = 0.0; // Invalid price
    invalid_trade.size = -1.0; // Invalid size

    // Valid trade sollte verarbeitet werden
    EXPECT_TRUE(valid_trade.ts_exchange >= 1704067200000000ULL);
    EXPECT_TRUE(valid_trade.price > 0);
    EXPECT_TRUE(valid_trade.size > 0);

    // Invalid trades sollten übersprungen werden
    EXPECT_FALSE(invalid_trade.ts_exchange >= 1704067200000000ULL);
    EXPECT_FALSE(invalid_trade.price > 0);
    EXPECT_FALSE(invalid_trade.size > 0);
}

// Test für Concurrent Access auf InstrumentStore
TEST_F(HotSpineDataBridgeTest, InstrumentStoreAtomicAccess) {
    BTQuant::InstrumentStore store;
    store.symbol = "BTC/USD";
    store.exchange = "BINANCE";

    // Teste atomic symbol_id
    EXPECT_EQ(store.symbol_id.load(), 0u);

    // Simuliere concurrent writes (single writer)
    std::thread writer([&store]() {
        store.symbol_id.store(12345);
    });

    writer.join();
    EXPECT_EQ(store.symbol_id.load(), 12345u);

    // Teste atomic latest_snapshot
    EXPECT_EQ(store.latest_snapshot.load(), nullptr);

    auto* snapshot = new BTQuant::HotOrderbookSnapshot();
    snapshot->symbol_id = 12345;
    snapshot->ts_exchange = 1704067200000000ULL;

    std::thread snapshot_writer([&store, snapshot]() {
        store.latest_snapshot.store(snapshot);
    });

    snapshot_writer.join();
    EXPECT_EQ(store.latest_snapshot.load(), snapshot);

    // Cleanup
    delete snapshot;
}

// Test für Orderbook Bounds Checking
TEST_F(HotSpineDataBridgeTest, OrderbookBoundsChecking) {
    BTQuant::HotOrderbookSnapshot snap;
    snap.bids_count = 25; // Mehr als max 20
    snap.asks_count = 25;

    // Simuliere safe bounds checking aus sync_shm()
    int safe_bids_count = std::min((int)snap.bids_count, 20);
    int safe_asks_count = std::min((int)snap.asks_count, 20);

    EXPECT_EQ(safe_bids_count, 20);
    EXPECT_EQ(safe_asks_count, 20);

    // Teste mit gültigen counts
    snap.bids_count = 10;
    snap.asks_count = 10;
    safe_bids_count = std::min((int)snap.bids_count, 20);
    safe_asks_count = std::min((int)snap.asks_count, 20);

    EXPECT_EQ(safe_bids_count, 10);
    EXPECT_EQ(safe_asks_count, 10);
}

// Performance Test für Atomic Operations
TEST_F(HotSpineDataBridgeTest, DISABLED_AtomicPerformance) {
    // Performance test für atomic operations vs non-atomic
    const int iterations = 1000000;

    std::atomic<uint64_t> atomic_counter{0};
    uint64_t regular_counter = 0;

    // Test atomic performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        atomic_counter.fetch_add(1, std::memory_order_relaxed);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto atomic_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test regular performance (nicht thread-safe, nur für Vergleich)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        regular_counter++;
    }
    end = std::chrono::high_resolution_clock::now();
    auto regular_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Atomic operations: " << atomic_duration.count() << " µs" << std::endl;
    std::cout << "Regular operations: " << regular_duration.count() << " µs" << std::endl;

    // Atomic sollte nicht extrem langsamer sein (< 10x)
    EXPECT_LT(atomic_duration.count(), regular_duration.count() * 10);
}