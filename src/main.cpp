#include "analytics/tpoengine.h"
#include "analytics/liquiditysweepdetector.h"
#include "analytics/lockfreesnapshotpipeline.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

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

    std::cout << "\nLock-Free Snapshot Pipeline test completed successfully!\n";

    return 0;
}