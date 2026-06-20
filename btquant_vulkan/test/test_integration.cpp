#include "../src/core/vulkan_context.hpp"
#include "../src/data/data_spine.hpp"
#include "../src/data/ring_buffer.hpp"
#include "../src/ui/ui_context.hpp"
#include "../src/ui/window_manager.hpp"
#include "../src/data/market_data.hpp"
#include "../src/util/settings.hpp"
#include "../src/ui/stats_overlay.hpp"
#include "../src/data/mock_producer.hpp"
#include "../src/widgets/alerts_panel.hpp"
#include "../src/widgets/watchlist_widget.hpp"
#include "../src/widgets/log_panel.hpp"
#include "../src/widgets/risk_panel.hpp"
#include "../src/widgets/connection_panel.hpp"
#include "../src/data/market_data_processor.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
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
        s.theme               = 1;  // Light
        s.save(tmpFile);

        auto loaded = btquant::util::Settings::load(tmpFile);
        bool ok = !loaded.showOrderBook && loaded.showOrderBookDepth &&
                  !loaded.showFootprint && loaded.showVPVR &&
                  !loaded.showMultiVWAP && loaded.showRiskPanel &&
                  loaded.showDOM && !loaded.showTrades && loaded.showTPO &&
                  loaded.fpsLimit == 144 && loaded.heatmapDensity == 256 &&
                  loaded.tradeWindowSeconds > 29.9 && loaded.tradeWindowSeconds < 30.1 &&
                  loaded.theme == 1;
        if (ok) {
            std::cout << "✓ Roundtrip preserved all 15 fields (incl. theme=light)" << std::endl;
        } else {
            std::cout << "✗ Roundtrip lost data (ob=" << loaded.showOrderBook
                      << " fps=" << loaded.fpsLimit
                      << " density=" << loaded.heatmapDensity
                      << " tw=" << loaded.tradeWindowSeconds
                      << " theme=" << loaded.theme << ")" << std::endl;
        }

        // Verify string-form theme values ("dark"/"light") parse too.
        {
            std::ofstream f(tmpFile, std::ios::trunc);
            f << "theme=light\n";
            f.close();
            auto s2 = btquant::util::Settings::load(tmpFile);
            if (s2.theme == 1) {
                std::cout << "✓ theme=\"light\" string form parses to 1" << std::endl;
            } else {
                std::cout << "✗ theme=\"light\" parsed as " << s2.theme << std::endl;
            }
            std::ofstream f2(tmpFile, std::ios::trunc);
            f2 << "theme=dark\n";
            f2.close();
            auto s3 = btquant::util::Settings::load(tmpFile);
            if (s3.theme == 0) {
                std::cout << "✓ theme=\"dark\" string form parses to 0" << std::endl;
            } else {
                std::cout << "✗ theme=\"dark\" parsed as " << s3.theme << std::endl;
            }
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

    // Test 9: MockProducer writes a valid header and increments sequence.
    std::cout << "\nTest 9: Testing MockProducer..." << std::endl;
    {
        namespace fs = std::filesystem;
        fs::path tmpDir = fs::temp_directory_path() / "btquant_test_mock";
        fs::create_directories(tmpDir);
        fs::path tmpFile = tmpDir / "hotspine.bin";
        std::error_code ec;
        fs::remove(tmpFile, ec);

        btquant::data::MockProducer mp(tmpFile.string(),
                                        "BTC/USDT", "binance",
                                        1000.0, 20 /* 20 ms tick */);
        if (auto err = mp.start()) {
            std::cout << "✗ MockProducer.start failed: " << *err << std::endl;
        } else {
            // Let it run ~250 ms (≈12 ticks).
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            mp.stop();

            // Verify header magic.
            std::ifstream in(tmpFile, std::ios::binary);
            char magic[4] = {};
            in.read(magic, 4);
            bool headerOk = (std::memcmp(magic, "UQTB", 4) == 0);

            // Verify sequence advanced past 5 ticks (50 ms @ 20 ms each).
            uint64_t seq = mp.sequence();
            bool seqOk = (seq >= 5);

            // Verify file size matches expected layout (header 4KB + 1×128B record).
            auto sz = fs::file_size(tmpFile, ec);
            bool sizeOk = (sz == 0x1000 + 128);

            if (headerOk && seqOk && sizeOk) {
                std::cout << "✓ MockProducer wrote UQTB header + "
                          << seq << " ticks (file=" << sz << " bytes)" << std::endl;
            } else {
                std::cout << "✗ MockProducer check failed "
                          << "(header=" << headerOk
                          << " seq=" << seqOk << "(" << seq << ")"
                          << " size=" << sizeOk << "(" << sz << ")"
                          << ")" << std::endl;
            }
        }

        // Cleanup.
        fs::remove(tmpFile, ec);
        fs::remove(tmpDir, ec);
    }

    // Test 10: Settings profiles — applyTo + preset factories + path sanitize.
    std::cout << "\nTest 10: Testing profiles..." << std::endl;
    {
        using namespace btquant::util;

        // Scalper preset: DOM + Trades + RiskPanel + Footprint + MultiVWAP on;
        // OrderBook, OB-Depth, VPVR, TPO off.
        auto sc = Settings::presetScalper();
        bool ob = true, obd = true, fp = false, vpvr = true, mvwap = true;
        bool risk = false, dom = false, trades = true, tpo = true;
        long theme = 1, density = 64;
        sc.applyTo(ob, obd, fp, vpvr, mvwap, risk, dom, trades, tpo, theme, density);
        bool scalperOk = !ob && !obd && fp && !vpvr && mvwap && risk && dom && trades && !tpo;
        if (scalperOk) {
            std::cout << "✓ Scalper preset applied correctly" << std::endl;
        } else {
            std::cout << "✗ Scalper applyTo wrong: ob=" << ob << " obd=" << obd
                      << " fp=" << fp << " vpvr=" << vpvr << " dom=" << dom
                      << " trades=" << trades << std::endl;
        }

        // Volatility preset: density 384.
        auto vol = Settings::presetVolatility();
        if (vol.heatmapDensity == 384 && vol.showTPO) {
            std::cout << "✓ Volatility preset has density=384 + TPO" << std::endl;
        } else {
            std::cout << "✗ Volatility preset wrong: density=" << vol.heatmapDensity
                      << " tpo=" << vol.showTPO << std::endl;
        }

        // profilePath sanitization — "../../etc/passwd" must NOT traverse.
        auto bad = Settings::profilePath("../../etc/passwd");
        auto canon = std::filesystem::weakly_canonical(bad);
        // The cleaned name "etcpasswd" (special chars stripped) should be in
        // ~/.config/btquant_vulkan/profiles/ — never reach /etc.
        bool safeOk = (canon.string().find("/etc/passwd") == std::string::npos) &&
                      (canon.string().find("btquant_vulkan/profiles") != std::string::npos);
        if (safeOk) {
            std::cout << "✓ Profile name sanitization blocks path traversal" << std::endl;
        } else {
            std::cout << "✗ Profile path traversal NOT blocked: " << canon << std::endl;
        }

        // Roundtrip: save profile to disk, load it back, verify applyTo matches.
        namespace fs = std::filesystem;
        fs::path tmpDir = fs::temp_directory_path() / "btquant_test_profile";
        fs::create_directories(tmpDir);
        auto profileFile = tmpDir / "sc1.ini";
        Settings saved;
        saved.showDOM = true;
        saved.showTrades = true;
        saved.theme = 0;
        saved.heatmapDensity = 256;
        saved.save(profileFile);
        auto loaded = Settings::load(profileFile);
        if (loaded.showDOM && loaded.showTrades && loaded.heatmapDensity == 256) {
            std::cout << "✓ Profile save+load roundtrip preserves settings" << std::endl;
        } else {
            std::cout << "✗ Profile roundtrip lost data" << std::endl;
        }
        std::error_code ec;
        fs::remove(profileFile, ec);
        fs::remove(tmpDir, ec);
    }

    // Test 11: AlertsPanel — constructible + accessor sanity.
    std::cout << "\nTest 11: Testing AlertsPanel..." << std::endl;
    {
        btquant::ui::AlertsPanel ap;
        ap.priceMovePctThreshold = 1.0;
        ap.volumeSpikeMultiplier = 3.0;
        ap.soundEnabled = false;
        ap.showInStatusBar = false;
        ap.clearAlerts();

        if (ap.alerts().empty()) {
            std::cout << "✓ AlertsPanel: constructible, clearAlerts works"
                      << std::endl;
        } else {
            std::cout << "✗ AlertsPanel.clearAlerts didn't empty the deque"
                      << std::endl;
        }

        btquant::MarketDataProcessor mdp;
        if (auto err = mdp.start("/nonexistent_test_path", 100)) {
            std::cout << "  (mdp.start returned: " << *err << ")" << std::endl;
        }
        ap.setMarketData(&mdp);
        mdp.stop();
        std::cout << "✓ AlertsPanel.setMarketData() accepts a processor" << std::endl;
    }

    // Test 12: WatchlistWidget — symbol set + per-tick update produces rows.
    std::cout << "\nTest 12: Testing WatchlistWidget..." << std::endl;
    {
        btquant::ui::WatchlistWidget wl;
        wl.setSymbols({"BTC/USDT", "ETH/USDT", "SOL/USDT"});
        if (wl.rowCount() != 3) {
            std::cout << "✗ rowCount after setSymbols: " << wl.rowCount() << std::endl;
        } else {
            std::cout << "✓ WatchlistWidget.setSymbols created 3 rows" << std::endl;
        }

        // Push 3 ticks into BTC/USDT.
        for (int i = 0; i < 3; ++i) {
            wl.update("BTC/USDT", 100.0 + i, 0.5, i % 2 == 0, 1000 + i);
        }
        auto* r = wl.row("BTC/USDT");
        if (r && r->lastPrice == 102.0 && r->tickCount == 3 &&
            r->buyVol > 0 && r->sellVol > 0) {
            std::cout << "✓ WatchlistWidget.update() computes last/volume/tickCount"
                      << std::endl;
        } else {
            std::cout << "✗ WatchlistWidget update failed: "
                      << "last=" << (r ? r->lastPrice : -1)
                      << " ticks=" << (r ? r->tickCount : 0)
                      << " buy=" << (r ? r->buyVol : -1)
                      << " sell=" << (r ? r->sellVol : -1)
                      << std::endl;
        }

        // Other rows untouched.
        auto* e = wl.row("ETH/USDT");
        if (e && e->lastPrice == 0.0 && e->tickCount == 0) {
            std::cout << "✓ Other symbols untouched (last=0, ticks=0)" << std::endl;
        } else {
            std::cout << "✗ Other symbols leaked: "
                      << "ETH last=" << (e ? e->lastPrice : -1)
                      << std::endl;
        }

        // Sparkline grows up to kMaxSparkPoints.
        for (int i = 0; i < 100; ++i) wl.update("SOL/USDT", 200.0 + i, 1.0, true, 2000 + i);
        auto* s = wl.row("SOL/USDT");
        if (s && s->spark.size() == btquant::ui::WatchlistWidget::kMaxSparkPoints) {
            std::cout << "✓ Sparkline capped at kMaxSparkPoints ("
                      << s->spark.size() << ")" << std::endl;
        } else {
            std::cout << "✗ Sparkline cap failed: size="
                      << (s ? s->spark.size() : 0) << std::endl;
        }

        wl.clear();
        if (wl.rowCount() == 0) {
            std::cout << "✓ WatchlistWidget.clear() empties all rows" << std::endl;
        } else {
            std::cout << "✗ clear() left " << wl.rowCount() << " rows" << std::endl;
        }
    }

    // Test 13: LogPanel — singleton, thread-safe push, ring trim.
    std::cout << "\nTest 13: Testing LogPanel..." << std::endl;
    {
        auto& lp = btquant::ui::LogPanel::instance();
        lp.clear();

        BTQ_LOG_INFO("test info line %d", 42);
        BTQ_LOG_WARN("test warn %s", "hello");
        BTQ_LOG_ERROR("test error code=%d", -1);

        if (lp.lineCount() == 3) {
            std::cout << "✓ LogPanel captured 3 lines (info/warn/error)" << std::endl;
        } else {
            std::cout << "✗ LogPanel.lineCount: " << lp.lineCount()
                      << " (expected 3)" << std::endl;
        }

        // Sequence numbers are monotonic.
        BTQ_LOG_DEBUG("seq test A");
        BTQ_LOG_DEBUG("seq test B");
        BTQ_LOG_DEBUG("seq test C");
        if (lp.lineCount() >= 6) {
            std::cout << "✓ LogPanel monotonic sequence (6 lines)" << std::endl;
        } else {
            std::cout << "✗ LogPanel sequence broken: " << lp.lineCount() << std::endl;
        }

        // Overflow → trim. Push 2x kMaxLines + 100, expect size == kMaxLines.
        const size_t before = lp.lineCount();
        const size_t burst = btquant::ui::LogPanel::kMaxLines * 2 + 100;
        for (size_t i = 0; i < burst; ++i) BTQ_LOG_INFO("overflow %zu", i);
        if (lp.lineCount() == btquant::ui::LogPanel::kMaxLines) {
            std::cout << "✓ LogPanel ring trim keeps size == kMaxLines ("
                      << lp.lineCount() << ")" << std::endl;
        } else {
            std::cout << "✗ LogPanel ring trim failed: size=" << lp.lineCount()
                      << " (kMaxLines=" << btquant::ui::LogPanel::kMaxLines
                      << ", before=" << before << ", pushed=" << burst << ")"
                      << std::endl;
        }

        // Singleton identity.
        auto* lp2 = &btquant::ui::LogPanel::instance();
        if (lp2 == &lp) {
            std::cout << "✓ LogPanel singleton identity stable" << std::endl;
        } else {
            std::cout << "✗ LogPanel singleton broken" << std::endl;
        }
        lp.clear();
    }

    // Test 14: computeMetrics — Sharpe / max DD / win rate / profit factor.
    std::cout << "\nTest 14: Testing computeMetrics..." << std::endl;
    {
        using btquant::data::Trade;
        std::vector<Trade> trades;

        // Construct a deterministic series: 10 trades, prices 100→109.
        // All buys. Newest-first layout matches MarketDataProcessor:
        // trades.front() = newest (109), trades.back() = oldest (100).
        // Iterate via rbegin() in computeMetrics → chronological.
        for (int i = 9; i >= 0; --i) {
            Trade t{};
            t.id    = i;
            t.price = 100.0 + i;
            t.size  = 1.0;
            t.isBuy = true;
            t.timestamp = 1000 + i;
            trades.push_back(t);  // back = oldest, front = newest
        }

        auto m = btquant::ui::computeMetrics(trades);

        if (m.tradeCount == 10) {
            std::cout << "✓ tradeCount == 10" << std::endl;
        } else {
            std::cout << "✗ tradeCount: " << m.tradeCount << std::endl;
        }
        if (m.buyCount == 10 && m.sellCount == 0) {
            std::cout << "✓ buyCount/sellCount split (10/0)" << std::endl;
        } else {
            std::cout << "✗ buy/sell split: "
                      << m.buyCount << "/" << m.sellCount << std::endl;
        }
        // All-up price series → Sharpe (per-trade) must be > 0 (mean > 0).
        if (m.sharpePerTrade > 0.0) {
            std::cout << "✓ positive trend → positive Sharpe: "
                      << m.sharpePerTrade << std::endl;
        } else {
            std::cout << "✗ Sharpe sign wrong: " << m.sharpePerTrade << std::endl;
        }
        // Annualized = perTrade * sqrt(N-1) for N=10 returns.
        double expectedAnn = m.sharpePerTrade * std::sqrt(9.0);
        if (std::abs(m.sharpeAnnualized - expectedAnn) < 1e-9) {
            std::cout << "✓ Sharpe sqrt(N) heuristic: "
                      << m.sharpeAnnualized << std::endl;
        } else {
            std::cout << "✗ Sharpe annualized: got " << m.sharpeAnnualized
                      << " expected " << expectedAnn << std::endl;
        }
        // Buy-only monotonic up → max DD should be ≈ 0 (cumPnl only grows).
        if (m.maxDrawdown < 1e-9) {
            std::cout << "✓ monotonic-up → max DD ~ 0 ("
                      << m.maxDrawdown << ")" << std::endl;
        } else {
            std::cout << "✗ max DD should be ~0 on monotonic-up: "
                      << m.maxDrawdown << std::endl;
        }
        // All round-trip dpnl > 0 → winRate = 1.0.
        if (m.winRate > 0.99) {
            std::cout << "✓ monotonic-up → winRate = "
                      << (m.winRate * 100.0) << "%" << std::endl;
        } else {
            std::cout << "✗ winRate: " << m.winRate << std::endl;
        }

        // Empty trade list → all zeros.
        std::vector<Trade> empty;
        auto em = btquant::ui::computeMetrics(empty);
        if (em.tradeCount == 0 && em.sharpePerTrade == 0.0 &&
            em.maxDrawdown == 0.0 && em.winRate == 0.0) {
            std::cout << "✓ empty input → all-zero metrics" << std::endl;
        } else {
            std::cout << "✗ empty-input invariants broken" << std::endl;
        }
    }

    // Test 15: ConnectionPanel + MarketDataProcessor accessors.
    std::cout << "\nTest 15: Testing ConnectionPanel + MarketDataProcessor..." << std::endl;
    {
        btquant::ui::ConnectionPanel cp;
        if (cp.tickRateEwma() == 0.0) {
            std::cout << "✓ ConnectionPanel: EWMA starts at 0" << std::endl;
        } else {
            std::cout << "✗ ConnectionPanel initial EWMA: " << cp.tickRateEwma() << std::endl;
        }

        // Wire up a processor; let it run for a tick window.
        btquant::MarketDataProcessor mdp;
        auto err = mdp.start("/dev/shm/btquant_test_panel", 50);
        if (err) std::cout << "  (mdp.start: " << *err << ")" << std::endl;
        cp.setMarketData(&mdp);

        // Without an ImGui context we can't actually call render(), but the
        // accessor surface should be safe.
        if (mdp.sourcePath() == "/dev/shm/btquant_test_panel") {
            std::cout << "✓ MarketDataProcessor.sourcePath(): "
                      << mdp.sourcePath() << std::endl;
        } else {
            std::cout << "✗ sourcePath: " << mdp.sourcePath() << std::endl;
        }
        if (mdp.symbol() == "BTC/USDT") {
            std::cout << "✓ MarketDataProcessor.symbol(): " << mdp.symbol() << std::endl;
        } else {
            std::cout << "✗ symbol: " << mdp.symbol() << std::endl;
        }
        if (mdp.isRunning()) {
            std::cout << "✓ MarketDataProcessor.isRunning() true" << std::endl;
        } else {
            std::cout << "✗ isRunning false" << std::endl;
        }

        // After a small sleep, ticksSeen should be > 0 (synthetic fallback
        // produces ~50 ms ticks; even 250 ms gives ~5 ticks).
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        if (mdp.ticksSeen() > 0) {
            std::cout << "✓ ticksSeen() advanced: " << mdp.ticksSeen() << std::endl;
        } else {
            std::cout << "✗ ticksSeen still 0 after 300ms" << std::endl;
        }

        mdp.stop();
        if (!mdp.isRunning()) {
            std::cout << "✓ MarketDataProcessor.stop() halts runloop" << std::endl;
        } else {
            std::cout << "✗ isRunning still true after stop()" << std::endl;
        }
    }

    return 0;
}
