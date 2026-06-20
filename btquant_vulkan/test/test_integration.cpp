#include "../src/core/vulkan_context.hpp"
#include "../src/data/data_spine.hpp"
#include "../src/data/ring_buffer.hpp"
#include "../src/ui/ui_context.hpp"
#include "../src/ui/window_manager.hpp"
#include "../src/data/market_data.hpp"
#include "../src/util/settings.hpp"
#include "../src/ui/stats_overlay.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <thread>

int main() {
    std::cout << "Starting btquant_vulkan integration test..." << std::endl;
    
    // Test 1: Vulkan Context Creation
    std::cout << "Test 1: Creating Vulkan Context..." << std::endl;
    btquant::vulkan::VulkanContext vkContext;
    
    // Note: This test assumes a valid Vulkan installation and compatible GPU
    // In a real scenario, we would need to set up a window surface first
    try {
        // For this test, we'll just check if the instance creation works
        if (auto err = vkContext.createInstance()) {
            std::cout << "Warning: Could not create Vulkan instance: " << *err << std::endl;
            std::cout << "This may be due to missing GPU or Vulkan drivers." << std::endl;
        } else {
            std::cout << "✓ Vulkan instance created successfully" << std::endl;
            
            // Clean up the instance
            vkContext.cleanup();
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Exception during Vulkan context test: " << e.what() << std::endl;
    }
    
    // Test 2: Data Spine functionality
    std::cout << "\nTest 2: Testing Data Spine functionality..." << std::endl;
    btquant::data::DataSpine dataSpine;
    
    // Try to open a mock data file (this will fail in normal conditions)
    // In production, this would connect to /dev/shm/btquant_hotspine
    if (dataSpine.open("/tmp/mock_btquant_hotspine")) {
        std::cout << "✓ Data Spine opened successfully" << std::endl;
    } else {
        std::cout << "ℹ Data Spine could not open mock file (expected in test environment)" << std::endl;
    }
    
    // Test basic data structures
    btquant::data::MarketDataAggregator aggregator;
    btquant::data::MarketTick tick;
    tick.price = 100.0;
    tick.size = 10.0;
    tick.timestamp = 1234567890;
    tick.isBuy = true;
    
    aggregator.update(tick);
    auto metrics = aggregator.metrics();
    
    if (metrics.volume == 10.0) {
        std::cout << "✓ Market data aggregation working correctly" << std::endl;
    } else {
        std::cout << "✗ Market data aggregation failed" << std::endl;
    }
    
    // Test 3: Basic data structures
    std::cout << "\nTest 3: Testing basic data structures..." << std::endl;
    
    // Test OrderBook
    btquant::data::OrderBook book;
    book.midPrice = 100.5;
    if (book.midPrice == 100.5) {
        std::cout << "✓ OrderBook structure working" << std::endl;
    }
    
    // Test Trade
    btquant::data::Trade trade;
    trade.price = 99.9;
    trade.size = 5.5;
    if (trade.price == 99.9 && trade.size == 5.5) {
        std::cout << "✓ Trade structure working" << std::endl;
    }
    
    // Test Candle
    btquant::data::Candle candle;
    candle.open = 100.0;
    candle.close = 101.0;
    if (candle.open == 100.0 && candle.close == 101.0) {
        std::cout << "✓ Candle structure working" << std::endl;
    }
    
    // Test 4: Ring Buffer
    std::cout << "\nTest 4: Testing Ring Buffer functionality..." << std::endl;
    btquant::data::RingBuffer<int, 10> ringBuffer;
    
    // Test push/pop operations
    for (int i = 0; i < 5; ++i) {
        if (!ringBuffer.push(i)) {
            std::cout << "✗ Failed to push element " << i << std::endl;
        }
    }
    
    if (ringBuffer.size() == 5) {
        std::cout << "✓ Ring buffer size correct after pushes" << std::endl;
    } else {
        std::cout << "✗ Ring buffer size incorrect" << std::endl;
    }
    
    // Pop elements
    for (int i = 0; i < 5; ++i) {
        int value;
        if (ringBuffer.pop(value) && value == i) {
            std::cout << "✓ Ring buffer popped correct value: " << value << std::endl;
        } else {
            std::cout << "✗ Ring buffer pop failed or wrong value" << std::endl;
        }
    }
    
    if (ringBuffer.isEmpty()) {
        std::cout << "✓ Ring buffer is empty after pops" << std::endl;
    } else {
        std::cout << "✗ Ring buffer not empty after pops" << std::endl;
    }
    
    std::cout << "\nIntegration test completed." << std::endl;
    std::cout << "Note: Some tests may show warnings in environments without Vulkan support." << std::endl;

    // Test 5: Candle aggregation in MarketDataAggregator.
    std::cout << "\nTest 5: Testing candle aggregation..." << std::endl;
    {
        btquant::data::MarketDataAggregator agg(60ULL * 1'000'000ULL, 5);  // 1-min buckets, max 5

        // Three ticks all in minute 1 (timestamp 1_000_000 to 1_500_000 us).
        btquant::data::MarketTick t1{100.0, 1.0, 1'000'000ULL, true};
        btquant::data::MarketTick t2{102.0, 2.0, 1'100'000ULL, true};
        btquant::data::MarketTick t3{101.0, 3.0, 1'500'000ULL, false};
        agg.update(t1);
        agg.update(t2);
        agg.update(t3);

        // Current candle should have open=100, high=102, low=100, close=101,
        // volume=6, buy=3, sell=3, delta=0.
        auto* cur = agg.currentCandle();
        if (cur && cur->tradeCount == 3 && cur->open == 100.0 && cur->high == 102.0 &&
            cur->low == 100.0 && cur->close == 101.0 && cur->volume == 6.0 &&
            cur->buyVolume == 3.0 && cur->sellVolume == 3.0) {
            std::cout << "✓ Candle aggregation correctly folds in-progress ticks (open=100 high=102 low=100 close=101 vol=6 buy=3 sell=3)" << std::endl;
        } else {
            std::cout << "✗ Candle aggregation wrong: count=" << (cur ? cur->tradeCount : -1) << std::endl;
        }

        // Push a tick in the next minute — should finalize the first candle.
        btquant::data::MarketTick t4{105.0, 4.0, 65'000'000ULL, true};  // 65 sec = next minute
        agg.update(t4);

        if (agg.candles().size() == 1) {
            std::cout << "✓ Bucket boundary correctly finalized previous candle" << std::endl;
        } else {
            std::cout << "✗ Bucket boundary failed — candles=" << agg.candles().size() << std::endl;
        }
        if (agg.candles()[0].close == 101.0 && agg.candles()[0].volume == 6.0) {
            std::cout << "✓ Finalized candle preserved open/close/volume" << std::endl;
        } else {
            std::cout << "✗ Finalized candle data corrupted" << std::endl;
        }

        // Push many minutes — verify max_candles pruning.
        for (int i = 0; i < 10; ++i) {
            btquant::data::MarketTick t{100.0 + i, 1.0,
                                       120'000'000ULL + (uint64_t)i * 60'000'000ULL, true};
            agg.update(t);
        }
        if (agg.candles().size() == 5) {
            std::cout << "✓ max_candles pruning works (kept last 5 of 11+ finalized)" << std::endl;
        } else {
            std::cout << "✗ Pruning failed — candles=" << agg.candles().size() << std::endl;
        }
    }

    // Test 6: Settings load/save roundtrip.
    std::cout << "\nTest 6: Testing settings load/save roundtrip..." << std::endl;
    {
        namespace fs = std::filesystem;
        fs::path tmpDir = fs::temp_directory_path() / "btquant_test_settings";
        fs::create_directories(tmpDir);
        fs::path tmpFile = tmpDir / "state.ini";

        // Clean slate.
        std::error_code ec;
        fs::remove(tmpFile, ec);

        // Default load on missing file → all true / 60 / 128.
        auto def = btquant::util::Settings::load(tmpFile);
        if (def.showOrderBook && def.fpsLimit == 60 && def.heatmapDensity == 128) {
            std::cout << "✓ Defaults loaded on missing file" << std::endl;
        } else {
            std::cout << "✗ Defaults wrong (ob=" << def.showOrderBook
                      << " fps=" << def.fpsLimit
                      << " density=" << def.heatmapDensity << ")" << std::endl;
        }

        // Write a non-default state, read back.
        btquant::util::Settings s;
        s.showOrderBook       = false;
        s.showOrderBookDepth  = true;
        s.showFootprint       = false;
        s.showVPVR            = true;
        s.showMultiVWAP       = false;
        s.showRiskPanel       = true;
        s.showDOM             = true;
        s.showTrades          = false;
        s.showTPO             = true;
        s.fpsLimit            = 144;
        s.heatmapDensity      = 256;
        s.tradeWindowSeconds  = 30.0;
        s.save(tmpFile);

        auto loaded = btquant::util::Settings::load(tmpFile);
        bool ok = !loaded.showOrderBook && loaded.showOrderBookDepth &&
                  !loaded.showFootprint && loaded.showVPVR &&
                  !loaded.showMultiVWAP && loaded.showRiskPanel &&
                  loaded.showDOM && !loaded.showTrades && loaded.showTPO &&
                  loaded.fpsLimit == 144 && loaded.heatmapDensity == 256 &&
                  loaded.tradeWindowSeconds > 29.9 && loaded.tradeWindowSeconds < 30.1;
        if (ok) {
            std::cout << "✓ Roundtrip preserved all 12 fields" << std::endl;
        } else {
            std::cout << "✗ Roundtrip lost data (ob=" << loaded.showOrderBook
                      << " fps=" << loaded.fpsLimit
                      << " density=" << loaded.heatmapDensity
                      << " tw=" << loaded.tradeWindowSeconds << ")" << std::endl;
        }

        // Cleanup.
        fs::remove(tmpFile, ec);
        fs::remove(tmpDir, ec);
    }

    // Test 7: StatsOverlay EWMA tracking.
    std::cout << "\nTest 7: Testing stats overlay EWMA..." << std::endl;
    {
        btquant::ui::StatsOverlay ov;
        // First tick — no measurement, just seed.
        ov.tick();
        if (ov.avgFrameTimeMs() == 0.0) {
            std::cout << "✓ First tick correctly seeds EWMA to 0" << std::endl;
        } else {
            std::cout << "✗ First tick seeded to " << ov.avgFrameTimeMs() << std::endl;
        }

        // Tick 60 times with ~16ms sleeps (≈60 fps) — EWMA should converge near 16.
        for (int i = 0; i < 60; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            ov.tick();
        }
        double avg = ov.avgFrameTimeMs();
        double fps = ov.avgFps();
        if (avg > 10.0 && avg < 50.0 && fps > 20.0 && fps < 100.0) {
            std::cout << "✓ EWMA converges: avg=" << avg
                      << " ms, fps=" << fps << std::endl;
        } else {
            std::cout << "✗ EWMA out of range: avg=" << avg
                      << " ms, fps=" << fps << std::endl;
        }

        // Disable / re-enable.
        ov.setEnabled(false);
        if (!ov.enabled()) {
            std::cout << "✓ setEnabled(false) works" << std::endl;
        } else {
            std::cout << "✗ setEnabled(false) didn't stick" << std::endl;
        }
    }

    // Test 8: settingsDirty flag toggles correctly.
    std::cout << "\nTest 8: Testing settingsDirty flag..." << std::endl;
    {
        btquant::ui::WindowManager wm;
        if (!wm.settingsDirty()) {
            std::cout << "✓ Fresh WindowManager is clean" << std::endl;
        } else {
            std::cout << "✗ Fresh WindowManager reported dirty" << std::endl;
        }
        wm.markSettingsDirty();
        if (wm.settingsDirty()) {
            std::cout << "✓ markSettingsDirty() flips the flag" << std::endl;
        } else {
            std::cout << "✗ markSettingsDirty() didn't flip" << std::endl;
        }
        wm.clearSettingsDirty();
        if (!wm.settingsDirty()) {
            std::cout << "✓ clearSettingsDirty() resets the flag" << std::endl;
        } else {
            std::cout << "✗ clearSettingsDirty() didn't reset" << std::endl;
        }
    }

    return 0;
}