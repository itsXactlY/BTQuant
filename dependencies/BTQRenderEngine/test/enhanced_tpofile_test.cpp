#include "../include/analytics/tpoengine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

void test_enhanced_methods() {
    std::cout << "Testing enhanced TPOProfile methods...\n";

    TPOProfile profile;

    // Create some test data
    auto time1 = std::chrono::system_clock::time_point(std::chrono::hours(1));
    auto time2 = std::chrono::system_clock::time_point(std::chrono::hours(2));
    auto time3 = std::chrono::system_clock::time_point(std::chrono::hours(3));

    // Add price levels to different time brackets
    profile.add_price_to_time_bracket(100.0, time1);  // A
    profile.add_price_to_time_bracket(100.0, time2);  // B - now "AB"
    profile.add_price_to_time_bracket(100.0, time3);  // C - now "ABC"
    
    profile.add_price_to_time_bracket(105.0, time1);  // A - "A"
    profile.add_price_to_time_bracket(105.0, time2);  // B - now "AB"
    
    profile.add_price_to_time_bracket(110.0, time2);  // B - "B"
    profile.add_price_to_time_bracket(110.0, time3);  // C - now "BC"

    // Test get_prices_for_time_bracket
    auto prices_time1 = profile.get_prices_for_time_bracket(time1);
    auto prices_time2 = profile.get_prices_for_time_bracket(time2);
    auto prices_time3 = profile.get_prices_for_time_bracket(time3);

    // Verify that the right prices are associated with each time bracket
    assert(prices_time1.size() == 2);  // Prices 100.0 and 105.0 were touched in time1
    assert(prices_time2.size() == 3);  // Prices 100.0, 105.0, and 110.0 were touched in time2
    assert(prices_time3.size() == 2);  // Prices 100.0 and 110.0 were touched in time3

    // Check specific prices in each time bracket
    std::sort(prices_time1.begin(), prices_time1.end());
    std::sort(prices_time2.begin(), prices_time2.end());
    std::sort(prices_time3.begin(), prices_time3.end());

    assert(prices_time1[0] == 100.0 && prices_time1[1] == 105.0);
    assert(prices_time2[0] == 100.0 && prices_time2[1] == 105.0 && prices_time2[2] == 110.0);
    assert(prices_time3[0] == 100.0 && prices_time3[1] == 110.0);

    // Test get_time_bracket_for_letter
    auto retrieved_time1 = profile.get_time_bracket_for_letter("A");
    auto retrieved_time2 = profile.get_time_bracket_for_letter("B");
    auto retrieved_time3 = profile.get_time_bracket_for_letter("C");

    assert(retrieved_time1 == time1);
    assert(retrieved_time2 == time2);
    assert(retrieved_time3 == time3);

    // Test get_touch_counts
    auto touch_counts = profile.get_touch_counts();
    assert(touch_counts[100.0] == 3);  // Touched in 3 time brackets: A, B, C
    assert(touch_counts[105.0] == 2);  // Touched in 2 time brackets: A, B
    assert(touch_counts[110.0] == 2);  // Touched in 2 time brackets: B, C

    // Test get_unique_price_count and get_unique_time_bracket_count
    assert(profile.get_unique_price_count() == 3);  // Prices 100.0, 105.0, 110.0
    assert(profile.get_unique_time_bracket_count() == 3);  // Times time1, time2, time3

    std::cout << "  Enhanced methods test PASSED\n\n";
}

void test_edge_cases() {
    std::cout << "Testing edge cases for enhanced methods...\n";

    TPOProfile profile;
    auto time1 = std::chrono::system_clock::time_point(std::chrono::hours(1));

    // Test with empty profile
    auto empty_prices = profile.get_prices_for_time_bracket(time1);
    assert(empty_prices.empty());

    auto empty_time = profile.get_time_bracket_for_letter("A");
    assert(empty_time == std::chrono::system_clock::time_point());  // Null time point

    auto empty_counts = profile.get_touch_counts();
    assert(empty_counts.empty());

    assert(profile.get_unique_price_count() == 0);
    assert(profile.get_unique_time_bracket_count() == 0);

    // Add one entry and test
    profile.add_price_to_time_bracket(100.0, time1);
    
    assert(profile.get_unique_price_count() == 1);
    assert(profile.get_unique_time_bracket_count() == 1);
    
    auto counts = profile.get_touch_counts();
    assert(counts[100.0] == 1);

    auto prices = profile.get_prices_for_time_bracket(time1);
    assert(prices.size() == 1 && prices[0] == 100.0);

    std::cout << "  Edge cases test PASSED\n\n";
}

int main() {
    std::cout << "Running enhanced TPOProfile tests...\n\n";

    test_enhanced_methods();
    test_edge_cases();

    std::cout << "All enhanced TPOProfile tests PASSED!\n";

    return 0;
}