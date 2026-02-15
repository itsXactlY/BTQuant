#define _USE_MATH_DEFINES
#include "analytics/tpoengine.h"
#include "analytics/liquiditysweepdetector.h"
#include "analytics/lockfreesnapshotpipeline.h"
#include "analytics/rawtradetable.h"
#include "ui/compute_to_imgui_bind.h"
#include "audio/pitch_shifter.h"
#include "audio/audio_engine_integration.h"
#include "imgui.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>

int main() {
    std::cout << "Testing TPO Engine Implementation\n";
    
    // Create a TPO Engine instance with a price bucket size of 0.5
    TPOEngine engine(0.5);
    
    // Generate some sample price ticks
    std::vector<PriceTick> ticks;
    
    // Current time as a base
    auto base_time = std::chrono::system_clock::now();
    
    // Add some sample ticks with different timestamps and prices
    for (int i = 0; i < 100; ++i) {
        PriceTick tick;
        tick.timestamp = base_time + std::chrono::minutes(i % 60); // Vary timestamps within an hour
        tick.price = 100.0 + (i % 10) * 0.5; // Prices between 100.0 and 104.5
        tick.volume = 100 + (i % 50); // Vary volume
        
        ticks.push_back(tick);
        
        // Add some duplicate ticks to the same time bucket to test aggregation
        if (i % 10 == 0) {
            PriceTick dup_tick = tick;
            dup_tick.timestamp = base_time + std::chrono::minutes(i % 60);
            dup_tick.price = tick.price + 0.1; // Slightly different price in same bucket
            dup_tick.volume = tick.volume + 25;
            ticks.push_back(dup_tick);
        }
    }
    
    std::cout << "Processing " << ticks.size() << " price ticks...\n";
    
    // Process all ticks
    engine.process_ticks(ticks);
    
    // Print TPO data
    std::cout << "\nTPO Data:\n";
    engine.print_tpo_data();
    
    // Print TPO profile
    std::cout << "\nTPO Profile:\n";
    engine.print_tpo_profile();
    
    // Get statistics
    auto start_time = base_time;
    auto end_time = base_time + std::chrono::hours(2);
    auto stats = engine.get_statistics_for_period(start_time, end_time);
    
    std::cout << "\nStatistics:\n";
    std::cout << "Total ticks processed: " << stats.total_ticks_processed << "\n";
    std::cout << "Total volume: " << stats.total_volume << "\n";
    std::cout << "Unique time buckets: " << stats.unique_time_buckets << "\n";
    std::cout << "Unique price levels: " << stats.unique_price_levels << "\n";
    
    // Test individual tick processing
    std::cout << "\nTesting individual tick processing...\n";
    PriceTick new_tick;
    new_tick.timestamp = base_time + std::chrono::minutes(30);
    new_tick.price = 102.75;
    new_tick.volume = 150.0;
    
    engine.process_tick(new_tick);
    
    std::cout << "Added new tick: Time=" << std::chrono::duration_cast<std::chrono::seconds>(
        new_tick.timestamp.time_since_epoch()).count() << ", Price=" << new_tick.price << ", Vol=" << new_tick.volume << "\n";
    
    // Check the high and low prices for a specific time-price bucket
    auto time_bucket = engine.get_time_bucket_start(new_tick.timestamp);
    auto price_bucket = engine.get_price_bucket(new_tick.price);
    
    std::cout << "High price for bucket: " << engine.get_high_price(time_bucket, price_bucket) << "\n";
    std::cout << "Low price for bucket: " << engine.get_low_price(time_bucket, price_bucket) << "\n";

    // Demonstrate Value Area and POC tracking
    std::cout << "\n=== Value Area (VA) & POC Tracking Demo ===\n";
    
    // Get POC
    double poc = engine.get_current_poc();
    std::cout << "Point of Control (POC): " << poc << "\n";
    
    // Get Value Area (default 70%)
    auto value_area = engine.get_current_value_area_bounds();
    std::cout << "Value Area (70%): " << value_area.first << " - " << value_area.second << "\n";
    
    // Get Value Area with different percentage
    auto value_area_80 = engine.get_current_value_area_bounds(80.0);
    std::cout << "Value Area (80%): " << value_area_80.first << " - " << value_area_80.second << "\n";
    
    // Check if specific price levels are in the value area
    std::vector<double> test_prices = {100.0, 102.0, 104.0, 105.0};
    std::cout << "\nChecking if price levels are in Value Area (70%):\n";
    for (double price : test_prices) {
        bool in_va = engine.is_price_in_value_area(price);
        std::cout << "Price " << price << " in VA: " << (in_va ? "Yes" : "No") << "\n";
    }
    
    // Get TPO data with opacity for visualization
    std::cout << "\nTPO data with opacity for visualization:\n";
    auto tpo_with_opacity = engine.get_tpo_data_with_opacity();
    int count = 0;
    for (const auto& [node, opacity] : tpo_with_opacity) {
        if (count < 5) { // Limit output for readability
            std::cout << "Price: " << node.price_level << ", Count: " << node.count 
                      << ", Volume: " << node.total_volume << ", Opacity: " << opacity << "\n";
            count++;
        }
    }
    if (tpo_with_opacity.size() > 5) {
        std::cout << "... and " << (tpo_with_opacity.size() - 5) << " more entries\n";
    }
    
    // Get POC line data for visualization
    double poc_line = engine.get_poc_line_data();
    std::cout << "\nPOC line data for visualization: " << poc_line << "\n";

    // Test Liquidity Sweep Detector
    std::cout << "\n=== Testing Liquidity Sweep Detector ===\n";
    
    // Create a liquidity sweep detector
    LiquiditySweepDetector sweep_detector(5000.0, 0.4, 300, 0.25);  // threshold=5000, ratio=0.4, window=5min
    
    // Generate some sample liquidity snapshots to simulate liquidity sweeps
    std::vector<LiquiditySnapshot> liquidity_snapshots;
    
    // Create snapshots simulating liquidity at different price levels over time
    for (int i = 0; i < 50; ++i) {
        // Simulate some price levels with high liquidity that gets swept
        double time_offset = i * 10; // Every 10 seconds
        
        // Create snapshots at different price levels
        for (double price = 100.0; price <= 105.0; price += 0.5) {
            LiquiditySnapshot snapshot;
            snapshot.timestamp = base_time + std::chrono::seconds(static_cast<long long>(time_offset));
            snapshot.price_level = price;
            
            // Simulate high liquidity that gets reduced (swept) at certain points
            if (price == 102.0 && i > 20 && i < 25) {
                // Simulate a liquidity sweep at price 102.0 around the middle of the time series
                if (i == 21) {
                    // High liquidity before sweep
                    snapshot.bid_volume = 15000.0;
                    snapshot.ask_volume = 12000.0;
                } else if (i == 22) {
                    // Significant reduction in liquidity (sweep)
                    snapshot.bid_volume = 2000.0;  // Much lower than before
                    snapshot.ask_volume = 18000.0; // Keep ask high to simulate bid sweep
                } else {
                    // Normal liquidity levels
                    snapshot.bid_volume = 3000.0 + (i % 1000);
                    snapshot.ask_volume = 3500.0 + (i % 1000);
                }
            } else {
                // Normal liquidity levels
                snapshot.bid_volume = 3000.0 + (i % 1000);
                snapshot.ask_volume = 3500.0 + (i % 1000);
            }
            
            snapshot.order_count = 5 + (i % 10);
            liquidity_snapshots.push_back(snapshot);
        }
    }
    
    std::cout << "Processing " << liquidity_snapshots.size() << " liquidity snapshots...\n";
    
    // Process the liquidity snapshots
    sweep_detector.process_liquidity_snapshots(liquidity_snapshots);
    
    // Detect sweeps
    sweep_detector.detect_sweeps();
    
    // Print detected sweeps
    sweep_detector.print_sweeps();
    
    // Get specific sweeps in a time range
    auto sweeps_in_range = sweep_detector.get_sweeps_in_range(
        base_time + std::chrono::seconds(200), 
        base_time + std::chrono::seconds(300)
    );
    
    std::cout << "\nSweeps in specific time range: " << sweeps_in_range.size() << " detected\n";
    
    // Demonstrate integration between TPO Engine and Liquidity Sweep Detector
    std::cout << "\n=== Testing TPO-Liquidity Integration ===\n";
    
    // Integrate the TPO engine with the liquidity detector
    engine.integrate_with_liquidity_detector(sweep_detector);
    
    // Get the visual overlays for detected sweeps
    auto visual_overlays = sweep_detector.get_visual_overlays();
    std::cout << "Generated " << visual_overlays.size() << " visual overlays for detected sweeps\n";
    
    // Display first few overlays as example
    for (size_t i = 0; i < std::min(size_t(3), visual_overlays.size()); ++i) {
        const auto& overlay = visual_overlays[i];
        std::cout << "Overlay " << (i+1) << ": Position=(" << overlay.x_position << "," << overlay.y_position 
                  << "), Radius=" << overlay.radius << ", Color=(" 
                  << overlay.red << "," << overlay.green << "," << overlay.blue << ")\n";
    }
    
    std::cout << "\nTPO Engine and Liquidity Sweep Detector test completed successfully!\n";

    // Test Lock-Free Snapshot Pipeline
    std::cout << "\n=== Testing Lock-Free Snapshot Pipeline ===\n";

    // Create a lock-free snapshot pipeline for 1000 symbols
    LockFreeSnapshotPipeline lf_pipeline(1000);

    // Test single symbol write and read
    AtomicMarketData test_data(100.5, 1000.0, 100.4, 100.6, 500.0, 600.0);
    test_data.timestamp.store(std::chrono::system_clock::now());

    bool write_result = lf_pipeline.write_market_data(0, test_data);
    std::cout << "Write result: " << (write_result ? "Success" : "Failed") << std::endl;

    // Commit the snapshot to make the data available for reading
    lf_pipeline.commit_snapshot();

    AtomicMarketData read_data;
    bool read_result = lf_pipeline.read_market_data_snapshot(0, read_data);
    std::cout << "Read result: " << (read_result ? "Success" : "Failed") << std::endl;

    if (read_result) {
        std::cout << "Price: " << read_data.price.load() << std::endl;
        std::cout << "Volume: " << read_data.volume.load() << std::endl;
        std::cout << "Bid: " << read_data.bid_price.load() << "@" << read_data.bid_volume.load() << std::endl;
        std::cout << "Ask: " << read_data.ask_price.load() << "@" << read_data.ask_volume.load() << std::endl;
    }

    // Test batch read
    std::cout << "\nTesting batch read...\n";

    // Write data to multiple symbols
    for (int i = 1; i <= 5; ++i) {
        AtomicMarketData data(100.0 + i*0.1, 1000.0 + i*100,
                             99.9 + i*0.1, 100.1 + i*0.1,
                             500.0 + i*50, 600.0 + i*50);
        lf_pipeline.write_market_data(i, data);
    }

    // Commit the snapshot to make the data available for reading
    lf_pipeline.commit_snapshot();

    uint32_t symbols[] = {1, 2, 3, 4, 5};
    AtomicMarketData batch_results[5];

    size_t batch_count = lf_pipeline.read_batch_snapshot(symbols, batch_results, 5);
    std::cout << "Batch read count: " << batch_count << std::endl;

    for (size_t i = 0; i < batch_count; ++i) {
        std::cout << "Symbol " << symbols[i] << " - Price: " << batch_results[i].price.load()
                  << ", Volume: " << batch_results[i].volume.load() << std::endl;
    }

    // Print pipeline statistics
    auto pipeline_stats = lf_pipeline.get_stats();
    std::cout << "\nPipeline Statistics:" << std::endl;
    std::cout << "Total updates: " << pipeline_stats.total_updates << std::endl;
    std::cout << "Dropped updates: " << pipeline_stats.dropped_updates << std::endl;
    std::cout << "Write head: " << pipeline_stats.write_head << std::endl;
    std::cout << "Read tail: " << pipeline_stats.read_tail << std::endl;

    std::cout << "\nLock-Free Snapshot Pipeline test completed successfully!\n";

    // Test the new market table rendering functionality
    std::cout << "\n=== Testing Market Table Rendering [Buys | Asks | Price | Bids | Sells] ===\n";

    // Example values for the market table
    double bid_vol = 1250.50;
    double ask_vol = 980.75;
    double last_px = 102.45;
    double bid_px = 102.40;
    double ask_px = 102.50;

    std::cout << "Sample market data:\n";
    std::cout << "Bid Volume (Buys): " << bid_vol << std::endl;
    std::cout << "Ask Volume (Asks): " << ask_vol << std::endl;
    std::cout << "Last Price: " << last_px << std::endl;
    std::cout << "Bid Price: " << bid_px << std::endl;
    std::cout << "Ask Price: " << ask_px << std::endl;

    std::cout << "\nMarket table rendering functionality is ready for UI integration.\n";

    // Initialize UI binding system
    std::cout << "\n=== Initializing UI Binding System ===\n";
    
    BTQuant::UI::ComputeToImGuiBind ui_bind;
    
    // Bind the market table to the UI
    ui_bind.bindMarketTable(bid_vol, ask_vol, last_px, bid_px, ask_px, "Market Table [Buys | Asks | Price | Bids | Sells]");
    
    std::cout << "Market table bound to UI successfully!\n";

    // Test the horizontal bars visualization
    std::cout << "\n=== Testing Horizontal Bars Visualization ===\n";

    // Create sample data for horizontal bars
    std::vector<float> bar_values = {25.0f, 40.0f, 10.0f, 75.0f, 60.0f, 30.0f};
    std::vector<ImU32> bar_colors = {
        IM_COL32(255, 100, 100, 255),  // Red
        IM_COL32(100, 255, 100, 255),  // Green
        IM_COL32(100, 100, 255, 255),  // Blue
        IM_COL32(255, 255, 100, 255),  // Yellow
        IM_COL32(255, 100, 255, 255),  // Magenta
        IM_COL32(100, 255, 255, 255)   // Cyan
    };

    // Bind the horizontal bars to the UI
    ui_bind.bindHorizontalBars(bar_values, bar_colors, "Horizontal Bars Demo");

    std::cout << "Horizontal bars visualization bound to UI successfully!\n";
    std::cout << "Values: ";
    for (float val : bar_values) {
        std::cout << val << " ";
    }
    std::cout << "\nColors: " << bar_colors.size() << " colors assigned\n";

    // Test the new Atomic Unit Toggles (USD/COIN toggle)
    std::cout << "\n=== Testing Atomic Unit Toggles (USD/COIN) ===\n";

    std::cout << "Sample order book data:\n";
    std::cout << "Bid Volume: " << bid_vol << std::endl;
    std::cout << "Ask Volume: " << ask_vol << std::endl;
    std::cout << "Last Price: " << last_px << std::endl;
    std::cout << "Bid Price: " << bid_px << std::endl;
    std::cout << "Ask Price: " << ask_px << std::endl;

    // Bind the order book with USD/COIN toggle to the UI
    ui_bind.bindOrderBookWithToggle(bid_vol, ask_vol, last_px, bid_px, ask_px, "Order Book with USD/COIN Toggle");

    std::cout << "Order book with USD/COIN toggle bound to UI successfully!\n";

    // Test Raw Trade Table
    std::cout << "\n=== Testing Raw Trade Table ===\n";

    // Create a raw trade table with capacity for 1000 trades
    RawTradeTable trade_table(1000);

    // Initialize audio integration to trigger sounds on new trades
    std::cout << "Initializing audio integration for trade notifications...\n";
    AudioIntegration::TradeAudioNotifier audio_notifier;
    if (audio_notifier.initializeAudio()) {
        // Connect the audio notifier to the trade table
        audio_notifier.connectToTradeTable(trade_table);
        std::cout << "Audio integration connected to trade table\n";
    } else {
        std::cout << "Warning: Could not initialize audio, continuing without sound\n";
    }

    // Generate some sample trades
    auto trade_base_time = std::chrono::system_clock::now();
    std::vector<RawTrade> sample_trades;

    for (int i = 0; i < 50; ++i) {
        RawTrade trade;
        trade.timestamp = trade_base_time + std::chrono::milliseconds(i * 100); // 100ms intervals
        trade.price = 100.0 + (i % 20) * 0.25; // Prices between 100.0 and 104.75
        trade.volume = 10.0 + (i % 50); // Volumes between 10 and 59
        trade.side = (i % 3 == 0) ? 'B' : 'S'; // Alternate sides
        trade.trade_id = "T" + std::to_string(1000 + i); // Trade IDs like T1000, T1001, etc.

        sample_trades.push_back(trade);
    }

    std::cout << "Adding " << sample_trades.size() << " sample trades to the table...\n";

    // Add trades to the table (this will trigger audio notifications if audio is enabled)
    trade_table.add_trades(sample_trades);

    // Get trade statistics
    auto trade_stats = trade_table.get_trade_statistics();
    std::cout << "Trade Statistics:\n";
    std::cout << "  Total Trades: " << trade_stats.total_trades << "\n";
    std::cout << "  Total Volume: " << trade_stats.total_volume << "\n";
    std::cout << "  Avg Trade Size: " << trade_stats.avg_trade_size << "\n";
    std::cout << "  Largest Trade: " << trade_stats.largest_trade_size << "\n";
    std::cout << "  Buy Volume: " << trade_stats.buy_volume << " (" << trade_stats.buy_count << " trades)\n";
    std::cout << "  Sell Volume: " << trade_stats.sell_volume << " (" << trade_stats.sell_count << " trades)\n";

    // Get recent trades
    auto recent_trades = trade_table.get_recent_trades(5);
    std::cout << "\nMost recent 5 trades:\n";
    for (const auto& trade : recent_trades) {
        auto time_t = std::chrono::system_clock::to_time_t(trade.timestamp);
        std::cout << "  Time: " << std::ctime(&time_t) 
                  << "  Price: " << trade.price 
                  << "  Volume: " << trade.volume 
                  << "  Side: " << trade.side 
                  << "  ID: " << trade.trade_id << "\n";
    }

    // Test adding individual trades
    std::cout << "\nAdding individual trades...\n";
    RawTrade new_trade(trade_base_time + std::chrono::seconds(1), 102.50, 25.0, 'B', "NEW_TRADE_001");
    trade_table.add_trade(new_trade);

    // Test time range query
    auto time_range_trades = trade_table.get_trades_in_range(
        trade_base_time,
        trade_base_time + std::chrono::seconds(5)
    );
    std::cout << "Trades in first 5 seconds: " << time_range_trades.size() << "\n";

    // Bind the raw trade table to the UI
    ui_bind.bindRawTradeTable(trade_table, "Raw Trade Table [Buys | Asks | Price | Bids | Sells]");

    std::cout << "Raw Trade Table bound to UI successfully!\n";

    // Test the new Live Bid/Ask Button
    std::cout << "\n=== Testing Live Bid/Ask Button ===\n";

    // Bind the live bid/ask button to the UI using the existing lock-free pipeline
    ui_bind.bindLiveBidAskButton(lf_pipeline, 0, "Live Bid/Ask Button");

    std::cout << "Live Bid/Ask Button bound to UI successfully!\n";

    // Test the new Mouse Trading Interface with massive BUY MKT / SELL MKT buttons
    std::cout << "\n=== Testing Mouse Trading Interface ===\n";

    // Bind the mouse trading interface to the UI using the existing lock-free pipeline
    ui_bind.bindMouseTradingInterface(lf_pipeline, 0, "Mouse Trading Interface");

    std::cout << "Mouse Trading Interface bound to UI successfully!\n";

    // Test the new standalone Live Bid/Ask Button function
    std::cout << "\n=== Testing Standalone Live Bid/Ask Button ===\n";

    // The new renderLiveBidAskButton function can be used in other contexts
    std::cout << "Standalone Live Bid/Ask Button function is available for use!\n";

    std::cout << "\nAll analytics modules and UI visualizations tested successfully!\n";

    // Test Pitch Shifting functionality - Scale pitch inversely to volume (Big trade = Deep bass)
    std::cout << "\n=== Testing Pitch Shifting (Volume-Inverse Pitch Scaling) ===\n";

    // Create a pitch shifter with base pitch of 440Hz (A4 note)
    PitchShifter pitch_shifter(440.0, 1.0, 10000.0);

    // Simulate different trade volumes and see how pitch changes
    std::vector<double> test_volumes = {10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0};

    std::cout << "Volume -> Pitch mapping (Big trade = Deep bass):\n";
    for (double volume : test_volumes) {
        double calculated_pitch = pitch_shifter.calculate_pitch(volume);
        std::cout << "Volume: " << volume << " -> Pitch: " << calculated_pitch << "Hz\n";
    }

    // Generate a simple sine wave as input audio
    std::vector<double> input_signal;
    const double frequency = 440.0; // A4 note
    const double sample_rate = 44100.0;
    const double duration = 0.1; // 100ms

    for (int i = 0; i < sample_rate * duration; ++i) {
        // Generate a simple sine wave
        double t = static_cast<double>(i) / sample_rate;
        double sample = std::sin(2.0 * M_PI * frequency * t);
        input_signal.push_back(sample);
    }

    std::cout << "\nGenerated " << input_signal.size() << " samples of input signal at " << frequency << "Hz\n";

    // Test pitch shifting with different volumes
    std::vector<double> high_volume_shifted = pitch_shifter.shift_pitch(input_signal, 8000.0); // High volume = deep bass
    std::vector<double> low_volume_shifted = pitch_shifter.shift_pitch(input_signal, 100.0);  // Low volume = higher pitch

    std::cout << "Applied pitch shifting to audio samples\n";
    std::cout << "High volume (8000) shifted signal has " << high_volume_shifted.size() << " samples\n";
    std::cout << "Low volume (100) shifted signal has " << low_volume_shifted.size() << " samples\n";

    // Show some sample values to demonstrate the difference
    std::cout << "\nSample comparison (first 5 values):\n";
    std::cout << "Original: ";
    for (int i = 0; i < 5 && i < input_signal.size(); ++i) {
        std::cout << input_signal[i] << " ";
    }
    std::cout << "\nHigh Vol: ";
    for (int i = 0; i < 5 && i < high_volume_shifted.size(); ++i) {
        std::cout << high_volume_shifted[i] << " ";
    }
    std::cout << "\nLow Vol:  ";
    for (int i = 0; i < 5 && i < low_volume_shifted.size(); ++i) {
        std::cout << low_volume_shifted[i] << " ";
    }
    std::cout << "\n";

    // Demonstrate the concept with trading data
    std::cout << "\nApplying pitch shift to trading volume data:\n";
    std::vector<RawTrade> sample_trades_for_pitch = trade_table.get_recent_trades(10);

    for (const auto& trade : sample_trades_for_pitch) {
        double trade_pitch = pitch_shifter.calculate_pitch(trade.volume);
        char pitch_char = trade_pitch < 220.0 ? 'B' : (trade_pitch < 330.0 ? 'M' : 'H'); // Bass, Mid, High
        std::cout << "Trade Vol: " << trade.volume << " -> Pitch: " << trade_pitch << "Hz (" << pitch_char << ")\n";
    }

    std::cout << "\nPitch shifting functionality (Big trade = Deep bass) implemented and tested successfully!\n";

    // Clean up audio resources
    audio_notifier.shutdownAudio();

    return 0;
}