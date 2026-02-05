#include "../include/analytics/tpoengine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>

void test_tp_profile_basic_functionality() {
    std::cout << "Testing TPOProfile basic functionality...\n";
    
    TPOProfile profile;
    
    // Test letter assignment
    auto time1 = std::chrono::system_clock::time_point(std::chrono::hours(1));
    auto time2 = std::chrono::system_clock::time_point(std::chrono::hours(2));
    auto time3 = std::chrono::system_clock::time_point(std::chrono::hours(3));
    
    // Assign letters to time brackets
    std::string letter1 = profile.assign_letter_to_time_bracket(time1);
    std::string letter2 = profile.assign_letter_to_time_bracket(time2);
    std::string letter3 = profile.assign_letter_to_time_bracket(time3);
    
    // Check that letters are assigned correctly (A, B, C)
    assert(letter1 == "A");
    assert(letter2 == "B");
    assert(letter3 == "C");
    
    std::cout << "  Letter assignments: " << letter1 << ", " << letter2 << ", " << letter3 << std::endl;
    
    // Add prices to time brackets
    profile.add_price_to_time_bracket(100.0, time1);
    profile.add_price_to_time_bracket(100.0, time2);  // Same price in different time bracket
    profile.add_price_to_time_bracket(105.0, time1);  // Different price in same time bracket
    
    // Check letter sequences
    std::string seq100 = profile.get_letter_sequence_for_price(100.0);
    std::string seq105 = profile.get_letter_sequence_for_price(105.0);
    
    std::cout << "  Price 100.0 sequence: " << seq100 << std::endl;
    std::cout << "  Price 105.0 sequence: " << seq105 << std::endl;
    
    assert(seq100 == "AB");  // Price 100.0 appeared in time brackets A and B
    assert(seq105 == "A");  // Price 105.0 appeared only in time bracket A
    
    std::cout << "  Basic functionality test PASSED\n\n";
}

void test_tp_profile_case_sensitivity() {
    std::cout << "Testing TPOProfile case sensitivity (A-Z, a-z)...\n";
    
    TPOProfile profile;
    
    // Fill up to 26 uppercase letters (A-Z)
    std::vector<std::chrono::system_clock::time_point> times;
    for (int i = 0; i < 52; i++) {  // 26 uppercase + 26 lowercase
        times.push_back(std::chrono::system_clock::time_point(std::chrono::hours(i)));
    }
    
    // Assign letters to all time brackets
    std::vector<std::string> letters;
    for (int i = 0; i < 52; i++) {
        std::string letter = profile.assign_letter_to_time_bracket(times[i]);
        letters.push_back(letter);
    }
    
    // Check that first 26 are uppercase A-Z
    for (int i = 0; i < 26; i++) {
        assert(letters[i] == std::string(1, 'A' + i));
    }
    
    // Check that next 26 are lowercase a-z
    for (int i = 26; i < 52; i++) {
        assert(letters[i] == std::string(1, 'a' + i - 26));
    }
    
    std::cout << "  Case sensitivity test PASSED\n\n";
}

void test_tp_engine_integration() {
    std::cout << "Testing TPOEngine integration with TPOProfile...\n";
    
    TPOEngine engine(0.25);
    
    // Create sample ticks spanning multiple time brackets
    std::vector<PriceTick> sample_ticks = {
        {std::chrono::system_clock::time_point(std::chrono::minutes(0)), 100.50, 100.0},
        {std::chrono::system_clock::time_point(std::chrono::minutes(5)), 100.50, 150.0},  // Same price, different time
        {std::chrono::system_clock::time_point(std::chrono::minutes(35)), 100.50, 75.0},  // Different time bracket
        {std::chrono::system_clock::time_point(std::chrono::minutes(40)), 101.00, 125.0}, // Different price, different time
    };
    
    // Process the ticks
    engine.process_ticks(sample_ticks);
    
    // Get the TPO profile
    const TPOProfile& profile = engine.get_tpo_profile();
    
    // Check that price levels have been mapped to letters
    std::string seq10050 = profile.get_letter_sequence_for_price(engine.get_price_bucket(100.50));
    std::string seq10100 = profile.get_letter_sequence_for_price(engine.get_price_bucket(101.00));
    
    std::cout << "  Price " << engine.get_price_bucket(100.50) << " sequence: " << seq10050 << std::endl;
    std::cout << "  Price " << engine.get_price_bucket(101.00) << " sequence: " << seq10100 << std::endl;
    
    // The sequence should contain at least 2 letters for price 100.50 (appeared in 2 time brackets)
    assert(seq10050.length() >= 2);
    assert(seq10100.length() >= 1);
    
    std::cout << "  Integration test PASSED\n\n";
}

void test_tp_profile_clear_functionality() {
    std::cout << "Testing TPOProfile clear functionality...\n";
    
    TPOProfile profile;
    
    // Add some data
    auto time1 = std::chrono::system_clock::time_point(std::chrono::hours(1));
    profile.add_price_to_time_bracket(100.0, time1);
    
    // Verify data exists
    assert(profile.get_letter_sequence_for_price(100.0) == "A");
    assert(!profile.price_to_letters.empty());
    assert(!profile.time_bracket_to_letter.empty());
    
    // Clear the profile
    profile.clear();
    
    // Verify data is cleared
    assert(profile.get_letter_sequence_for_price(100.0) == "");
    assert(profile.price_to_letters.empty());
    assert(profile.time_bracket_to_letter.empty());
    assert(profile.letter_counter == 0);
    
    std::cout << "  Clear functionality test PASSED\n\n";
}

int main() {
    std::cout << "Running TPOProfile tests...\n\n";
    
    test_tp_profile_basic_functionality();
    test_tp_profile_case_sensitivity();
    test_tp_engine_integration();
    test_tp_profile_clear_functionality();
    
    std::cout << "All TPOProfile tests PASSED!\n";
    
    return 0;
}