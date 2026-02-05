#ifndef BTQRENDERENGINE_TPOENGINE_H
#define BTQRENDERENGINE_TPOENGINE_H

#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>

// Structure to represent a single price tick
struct PriceTick {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double volume;
};

// Structure to represent a TPO (Time-Price Opportunity) bucket
struct TPONode {
    std::chrono::system_clock::time_point time_start;
    std::chrono::system_clock::time_point time_end;
    double price_level;
    int count; // Number of times this price level was hit in the time bucket
    double total_volume;

    // Default constructor
    TPONode() : time_start(), time_end(), price_level(0.0), count(0), total_volume(0.0) {}

    TPONode(std::chrono::system_clock::time_point start,
            std::chrono::system_clock::time_point end,
            double price);
};

// Structure to represent a TPO Profile that maps price levels to letters (A-Z, a-z) based on time brackets
struct TPOProfile {
    // Map price levels to letters (A-Z, a-z) indicating the sequence of time brackets they were touched
    std::map<double, std::string> price_to_letters;

    // Map time brackets to letters (A-Z, a-z) to maintain consistent letter assignment
    std::map<std::chrono::system_clock::time_point, std::string> time_bracket_to_letter;

    // Counter to assign letters A-Z, then a-z (total 52 unique letters)
    int letter_counter;

    TPOProfile() : letter_counter(0) {}

    // Get the next letter in sequence (A-Z, then a-z)
    std::string get_next_letter() {
        if (letter_counter < 26) {
            // Uppercase A-Z
            return std::string(1, 'A' + letter_counter);
        } else if (letter_counter < 52) {
            // Lowercase a-z
            return std::string(1, 'a' + letter_counter - 26);
        } else {
            // If we exceed 52 time brackets, cycle back or return a special indicator
            return std::string(1, 'A' + (letter_counter % 26));
        }
    }

    // Assign a letter to a time bracket if not already assigned
    std::string assign_letter_to_time_bracket(const std::chrono::system_clock::time_point& time_bracket) {
        if (time_bracket_to_letter.find(time_bracket) == time_bracket_to_letter.end()) {
            std::string letter = get_next_letter();
            time_bracket_to_letter[time_bracket] = letter;
            letter_counter++;
            return letter;
        }
        return time_bracket_to_letter[time_bracket];
    }

    // Add a price level to a time bracket (assign letter to price level)
    void add_price_to_time_bracket(double price_level, const std::chrono::system_clock::time_point& time_bracket) {
        std::string letter = assign_letter_to_time_bracket(time_bracket);

        // Add the letter to the price level's sequence
        if (price_to_letters.find(price_level) == price_to_letters.end()) {
            price_to_letters[price_level] = letter;
        } else {
            // Append the letter to the existing sequence
            price_to_letters[price_level] += letter;
        }
    }

    // Get the letter sequence for a specific price level
    std::string get_letter_sequence_for_price(double price_level) const {
        auto it = price_to_letters.find(price_level);
        if (it != price_to_letters.end()) {
            return it->second;
        }
        return "";
    }

    // Clear all data
    void clear() {
        price_to_letters.clear();
        time_bracket_to_letter.clear();
        letter_counter = 0;
    }

    // Print the profile for debugging
    void print_profile() const {
        std::cout << "TPO Profile:\n";
        for (const auto& [price, letters] : price_to_letters) {
            std::cout << "Price: " << price << " -> Letters: " << letters << "\n";
        }
        std::cout << "\nTime Bracket Mappings:\n";
        for (const auto& [time_bracket, letter] : time_bracket_to_letter) {
            auto time_t = std::chrono::system_clock::to_time_t(time_bracket);
            std::cout << "Time: " << std::put_time(std::localtime(&time_t), "%F %T") << " -> Letter: " << letter << "\n";
        }
    }
};

class TPOEngine {
private:
    // Price bucket size - configurable based on instrument
    double price_bucket_size;

    // Map to store aggregated TPO data: time_bucket -> price_bucket -> TPONode
    std::map<std::chrono::system_clock::time_point,
             std::map<double, TPONode>> tpo_data;

    // TPO Profile to map price levels to letters based on time brackets
    TPOProfile tpo_profile;

public:
    explicit TPOEngine(double bucket_size = 0.25);

    // Calculate the time bucket start time for a given timestamp
    std::chrono::system_clock::time_point get_time_bucket_start(
        const std::chrono::system_clock::time_point& timestamp) const;

    // Calculate the price bucket for a given price
    double get_price_bucket(double price) const;

    // Process a single price tick and aggregate into TPO buckets
    void process_tick(const PriceTick& tick);

    // Process a vector of ticks
    void process_ticks(const std::vector<PriceTick>& ticks);

    // Get TPO data for a specific time range
    std::map<std::chrono::system_clock::time_point,
             std::map<double, TPONode>> get_tpo_data_for_range(
                 const std::chrono::system_clock::time_point& start_time,
                 const std::chrono::system_clock::time_point& end_time) const;

    // Get all TPO data
    const std::map<std::chrono::system_clock::time_point,
                   std::map<double, TPONode>>& get_all_tpo_data() const;

    // Get the TPO profile
    const TPOProfile& get_tpo_profile() const { return tpo_profile; }

    // Get mutable reference to TPO profile for modification
    TPOProfile& get_tpo_profile() { return tpo_profile; }

    // Clear all stored data
    void clear();

    // Print TPO data for debugging purposes
    void print_tpo_data() const;

    // Print TPO profile for debugging purposes
    void print_tpo_profile() const;
};

#endif // BTQRENDERENGINE_TPOENGINE_H