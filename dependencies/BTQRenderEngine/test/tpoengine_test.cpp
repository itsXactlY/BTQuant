#include "../include/analytics/tpoengine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

void test_get_time_bucket_start() {
    std::cout << "Testing get_time_bucket_start function...\n";

    TPOEngine engine(0.25);

    // Test with a specific time point
    auto time1 = std::chrono::system_clock::time_point(std::chrono::seconds(0)); // 1970-01-01 00:00:00
    auto bucket1 = engine.get_time_bucket_start(time1);
    assert(bucket1 == time1); // Should be the same as the start time

    // Test with 15 minutes past the hour (should round down to start of hour)
    auto time2 = std::chrono::system_clock::time_point(std::chrono::minutes(15));
    auto bucket2 = engine.get_time_bucket_start(time2);
    auto expected2 = std::chrono::system_clock::time_point(std::chrono::minutes(0));
    assert(bucket2 == expected2);

    // Test with 30 minutes past the hour (should stay at 30 min mark)
    auto time3 = std::chrono::system_clock::time_point(std::chrono::minutes(30));
    auto bucket3 = engine.get_time_bucket_start(time3);
    assert(bucket3 == time3);

    // Test with 45 minutes past the hour (should round down to 30 min mark)
    auto time4 = std::chrono::system_clock::time_point(std::chrono::minutes(45));
    auto bucket4 = engine.get_time_bucket_start(time4);
    auto expected4 = std::chrono::system_clock::time_point(std::chrono::minutes(30));
    assert(bucket4 == expected4);

    // Test with 75 minutes past the hour (should round down to 60 min mark)
    auto time5 = std::chrono::system_clock::time_point(std::chrono::minutes(75));
    auto bucket5 = engine.get_time_bucket_start(time5);
    auto expected5 = std::chrono::system_clock::time_point(std::chrono::minutes(60));
    assert(bucket5 == expected5);

    std::cout << "  get_time_bucket_start test PASSED\n\n";
}

void test_get_price_bucket() {
    std::cout << "Testing get_price_bucket function...\n";

    TPOEngine engine(0.25); // Price bucket size of 0.25

    // Test with exact bucket boundary
    assert(std::abs(engine.get_price_bucket(100.0) - 100.0) < 0.0001);
    
    // Test with value within first bucket
    assert(std::abs(engine.get_price_bucket(100.1) - 100.0) < 0.0001);
    
    // Test with value at next bucket boundary
    assert(std::abs(engine.get_price_bucket(100.25) - 100.25) < 0.0001);
    
    // Test with value within second bucket
    assert(std::abs(engine.get_price_bucket(100.4) - 100.25) < 0.0001);
    
    // Test with negative values
    assert(std::abs(engine.get_price_bucket(-100.1) - (-100.25)) < 0.0001);

    std::cout << "  get_price_bucket test PASSED\n\n";
}

void test_process_tick_single() {
    std::cout << "Testing process_tick with single tick...\n";

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Create a single tick
    PriceTick tick;
    tick.timestamp = std::chrono::system_clock::time_point(std::chrono::minutes(10));
    tick.price = 100.3;
    tick.volume = 150.0;

    engine.process_tick(tick);

    // Get the data
    const auto& all_data = engine.get_all_tpo_data();
    
    // Should have one time bucket
    assert(all_data.size() == 1);
    
    // The time bucket should be at the 0-minute mark (rounded down from 10 minutes)
    auto expected_time_bucket = std::chrono::system_clock::time_point(std::chrono::minutes(0));
    assert(all_data.count(expected_time_bucket) == 1);
    
    // Within that bucket, should have one price bucket
    auto price_bucket = engine.get_price_bucket(100.3); // Should be 100.0
    assert(all_data.at(expected_time_bucket).size() == 1);
    assert(all_data.at(expected_time_bucket).count(price_bucket) == 1);
    
    // Check the node properties
    const auto& node = all_data.at(expected_time_bucket).at(price_bucket);
    assert(node.count == 1);
    assert(std::abs(node.total_volume - 150.0) < 0.0001);
    assert(std::abs(node.price_level - price_bucket) < 0.0001);

    std::cout << "  process_tick single test PASSED\n\n";
}

void test_process_ticks_multiple_same_bucket() {
    std::cout << "Testing process_ticks with multiple ticks in same bucket...\n";

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Create multiple ticks in the same time and price bucket
    std::vector<PriceTick> ticks = {
        {std::chrono::system_clock::time_point(std::chrono::minutes(5)), 100.1, 100.0},
        {std::chrono::system_clock::time_point(std::chrono::minutes(15)), 100.3, 150.0},
        {std::chrono::system_clock::time_point(std::chrono::minutes(20)), 100.2, 200.0},
    };

    engine.process_ticks(ticks);

    // Get the data
    const auto& all_data = engine.get_all_tpo_data();
    
    // Should have one time bucket (0-30 minute range)
    assert(all_data.size() == 1);
    
    // The time bucket should be at the 0-minute mark
    auto expected_time_bucket = std::chrono::system_clock::time_point(std::chrono::minutes(0));
    assert(all_data.count(expected_time_bucket) == 1);
    
    // Should have one price bucket (for 100.0)
    auto price_bucket = engine.get_price_bucket(100.1); // Should be 100.0
    assert(all_data.at(expected_time_bucket).size() == 1);
    assert(all_data.at(expected_time_bucket).count(price_bucket) == 1);
    
    // Check the aggregated properties
    const auto& node = all_data.at(expected_time_bucket).at(price_bucket);
    assert(node.count == 3); // 3 ticks
    assert(std::abs(node.total_volume - 450.0) < 0.0001); // 100 + 150 + 200

    std::cout << "  process_ticks multiple same bucket test PASSED\n\n";
}

void test_process_ticks_different_buckets() {
    std::cout << "Testing process_ticks with ticks in different buckets...\n";

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Create ticks in different time and price buckets
    std::vector<PriceTick> ticks = {
        {std::chrono::system_clock::time_point(std::chrono::minutes(5)), 100.1, 100.0},   // Time: 0-30 min, Price: 100.0
        {std::chrono::system_clock::time_point(std::chrono::minutes(35)), 100.1, 150.0},  // Time: 30-60 min, Price: 100.0
        {std::chrono::system_clock::time_point(std::chrono::minutes(5)), 100.6, 200.0},   // Time: 0-30 min, Price: 100.5
        {std::chrono::system_clock::time_point(std::chrono::minutes(35)), 100.6, 250.0},  // Time: 30-60 min, Price: 100.5
    };

    engine.process_ticks(ticks);

    // Get the data
    const auto& all_data = engine.get_all_tpo_data();
    
    // Should have two time buckets (0-30 min and 30-60 min)
    assert(all_data.size() == 2);
    
    // Check first time bucket (0-30 min)
    auto time_bucket_1 = std::chrono::system_clock::time_point(std::chrono::minutes(0));
    assert(all_data.count(time_bucket_1) == 1);
    assert(all_data.at(time_bucket_1).size() == 2); // Two price buckets: 100.0 and 100.5
    
    // Check second time bucket (30-60 min)
    auto time_bucket_2 = std::chrono::system_clock::time_point(std::chrono::minutes(30));
    assert(all_data.count(time_bucket_2) == 1);
    assert(all_data.at(time_bucket_2).size() == 2); // Two price buckets: 100.0 and 100.5
    
    // Check specific nodes
    // First bucket, first price level (100.0)
    auto price_bucket_100 = engine.get_price_bucket(100.1); // 100.0
    const auto& node1 = all_data.at(time_bucket_1).at(price_bucket_100);
    assert(node1.count == 1);
    assert(std::abs(node1.total_volume - 100.0) < 0.0001);
    
    // Second bucket, first price level (100.0)
    const auto& node2 = all_data.at(time_bucket_2).at(price_bucket_100);
    assert(node2.count == 1);
    assert(std::abs(node2.total_volume - 150.0) < 0.0001);
    
    // First bucket, second price level (100.5)
    auto price_bucket_1005 = engine.get_price_bucket(100.6); // 100.5
    const auto& node3 = all_data.at(time_bucket_1).at(price_bucket_1005);
    assert(node3.count == 1);
    assert(std::abs(node3.total_volume - 200.0) < 0.0001);
    
    // Second bucket, second price level (100.5)
    const auto& node4 = all_data.at(time_bucket_2).at(price_bucket_1005);
    assert(node4.count == 1);
    assert(std::abs(node4.total_volume - 250.0) < 0.0001);

    std::cout << "  process_ticks different buckets test PASSED\n\n";
}

void test_get_tpo_data_for_range() {
    std::cout << "Testing get_tpo_data_for_range function...\n";

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Create ticks spanning multiple time buckets
    std::vector<PriceTick> ticks = {
        {std::chrono::system_clock::time_point(std::chrono::seconds(5*60)), 100.1, 100.0},   // Time: 0-30 min -> bucket at 0s
        {std::chrono::system_clock::time_point(std::chrono::seconds(35*60)), 100.1, 150.0},  // Time: 30-60 min -> bucket at 1800s
        {std::chrono::system_clock::time_point(std::chrono::seconds(65*60)), 100.1, 200.0},  // Time: 60-90 min -> bucket at 3600s
    };

    engine.process_ticks(ticks);

    // Query for range covering first two buckets: [0, 3600) seconds
    // This should include time buckets that start at 0s and 1800s, but NOT 3600s
    auto start_time = std::chrono::system_clock::time_point(std::chrono::seconds(0));
    auto end_time = std::chrono::system_clock::time_point(std::chrono::seconds(3600)); // Exactly 60 minutes

    auto range_data = engine.get_tpo_data_for_range(start_time, end_time);

    // Should have 2 time buckets in the range [0, 3600): time buckets starting at 0s and 1800s
    // The bucket starting at 3600s should be excluded since end_time is exclusive
    assert(range_data.size() == 2);

    // Check that we have the first time bucket (0-30 min)
    auto time_bucket_1 = std::chrono::system_clock::time_point(std::chrono::seconds(0));
    assert(range_data.count(time_bucket_1) == 1); // Should be present

    // Check that we have the second time bucket (30-60 min)
    auto time_bucket_2 = std::chrono::system_clock::time_point(std::chrono::seconds(1800));
    assert(range_data.count(time_bucket_2) == 1); // Should be present

    // The third bucket (60-90 min) should NOT be included since its start time equals end_time
    auto time_bucket_3 = std::chrono::system_clock::time_point(std::chrono::seconds(3600));
    assert(range_data.count(time_bucket_3) == 0); // Should NOT be present

    std::cout << "  get_tpo_data_for_range test PASSED\n\n";
}

void test_clear_functionality() {
    std::cout << "Testing clear functionality...\n";

    TPOEngine engine(0.5); // Price bucket size of 0.5

    // Add some data
    std::vector<PriceTick> ticks = {
        {std::chrono::system_clock::time_point(std::chrono::minutes(5)), 100.1, 100.0},
        {std::chrono::system_clock::time_point(std::chrono::minutes(35)), 100.1, 150.0},
    };
    engine.process_ticks(ticks);

    // Verify data exists
    const auto& all_data_before = engine.get_all_tpo_data();
    assert(all_data_before.size() == 2); // Two time buckets

    // Clear the engine
    engine.clear();

    // Verify data is cleared
    const auto& all_data_after = engine.get_all_tpo_data();
    assert(all_data_after.empty());

    // Also check that the profile is cleared
    const TPOProfile& profile = engine.get_tpo_profile();
    assert(profile.price_to_letters.empty());
    assert(profile.time_bracket_to_letter.empty());
    assert(profile.letter_counter == 0);

    std::cout << "  clear functionality test PASSED\n\n";
}

void test_edge_cases() {
    std::cout << "Testing edge cases...\n";

    // Test with zero price bucket size (should handle gracefully)
    TPOEngine engine(0.0);
    
    // Create a tick
    PriceTick tick;
    tick.timestamp = std::chrono::system_clock::time_point(std::chrono::minutes(10));
    tick.price = 100.0;
    tick.volume = 100.0;

    // This should not crash even with 0.0 bucket size
    engine.process_tick(tick);

    // Test with very small price bucket size
    TPOEngine small_engine(0.001);
    tick.price = 100.12345;
    small_engine.process_tick(tick);

    // Get the data and verify it's reasonable
    const auto& all_data = small_engine.get_all_tpo_data();
    assert(all_data.size() == 1);

    std::cout << "  edge cases test PASSED\n\n";
}

int main() {
    std::cout << "Running TPOEngine tests...\n\n";

    test_get_time_bucket_start();
    test_get_price_bucket();
    test_process_tick_single();
    test_process_ticks_multiple_same_bucket();
    test_process_ticks_different_buckets();
    test_get_tpo_data_for_range();
    test_clear_functionality();
    test_edge_cases();

    std::cout << "All TPOEngine tests PASSED!\n";

    return 0;
}